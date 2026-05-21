from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn

from ta_src.utils.quiet import suppressed_stdout

log = logging.getLogger(__name__)

try:
    import torchreid
    _HAS_TORCHREID = True
except Exception:
    _HAS_TORCHREID = False


def _load_reid_weights(model, path: Path) -> None:
    """Apply torchreid ReID weights from a local .pth onto the model.

    Indirection point so tests can monkeypatch without touching the real
    torchreid loader. Caller is responsible for verifying that the path
    exists; this helper just performs the load."""
    torchreid.utils.load_pretrained_weights(model, str(path))
    log.info("OSNet: loaded ReID weights from %s", path)

# Pre-computed normalisation constants as tensors (allocated once, reused every call).
# Shape (3, 1, 1) for broadcasting against (3, H, W) tensors.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _crop_to_tensor(crop: np.ndarray, out_h: int, out_w: int) -> torch.Tensor:
    """Resize a uint8 RGB crop with cv2 and convert to a normalised float tensor.

    cv2.resize avoids the PIL round-trip used in the old torchvision pipeline,
    saving ~1.8× on the transform step alone (O3).
    """
    resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(resized).permute(2, 0, 1).float().div_(255.0)
    return (t - _MEAN) / _STD


class OSNetWrapper:
    def __init__(
        self,
        device: str,
        input_size: tuple[int, int] = (256, 128),
        pretrained_reid_path: Path | str | None = None,
    ):
        self.device = device
        self.input_size = input_size  # (H, W)
        self._use_amp = device != "cpu" and torch.cuda.is_available()
        self._amp_dtype = torch.bfloat16 if (
            self._use_amp and torch.cuda.is_bf16_supported()
        ) else torch.float16

        if _HAS_TORCHREID:
            # torchreid prints "Successfully loaded imagenet pretrained weights ..."
            with suppressed_stdout():
                model = torchreid.models.build_model(
                    name="osnet_x1_0", num_classes=1000, pretrained=True
                )
            if hasattr(model, "classifier"):
                model.classifier = nn.Identity()
            # ImageNet weights cannot discriminate persons within the "person"
            # class — bodies in similar poses end up too close in embedding
            # space. ReID-trained checkpoints from the torchreid model zoo
            # restore proper margins. Run scripts/download_osnet_weights.py to
            # fetch osnet_x1_0_msmt17.pt and point this config at it.
            if pretrained_reid_path is not None:
                path = Path(pretrained_reid_path)
                if not path.is_file():
                    raise RuntimeError(
                        f"OSNet ReID weights not found: {path} — run "
                        f"scripts/download_osnet_weights.py or remove the "
                        f"tracking.osnet_reid_weights config key."
                    )
                _load_reid_weights(model, path)
            else:
                log.warning(
                    "OSNet: no ReID checkpoint configured — falling back to "
                    "ImageNet weights, which are weak at person discrimination. "
                    "Set tracking.osnet_reid_weights to a torchreid checkpoint."
                )
        else:
            from torchvision.models import resnet50, ResNet50_Weights
            base = resnet50(weights=ResNet50_Weights.DEFAULT)
            model = nn.Sequential(
                *list(base.children())[:-1],
                nn.Flatten(),
                nn.Linear(2048, 512),
            )

        model.eval().to(self.device)

        if self._use_amp:
            # TF32 gives free matmul speedup on Ampere+ (RTX 30/40 series).
            torch.set_float32_matmul_precision("high")
            # dynamic=True produces one compiled graph for all batch sizes so
            # Triton autotuning runs once (at warmup below) instead of once per
            # unique batch count mid-inference. reduce-overhead (cudagraphs) is
            # incompatible with dynamic shapes, so we use default mode (O2).
            try:
                model = torch.compile(model, mode="default", dynamic=True)
            except Exception:
                pass  # compile unavailable (old PyTorch, CPU-only build, etc.)

        self._model = model
        if self._use_amp:
            self._prewarm()

    @torch.no_grad()
    def _prewarm(self, max_batch: int = 16) -> None:
        """Run dummy batches 1→max_batch to front-load all Triton autotuning.

        After the first ever run these benchmarks are cached to disk
        (TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR) and subsequent process
        starts load instantly — no mid-inference benchmark surprises.
        """
        import logging
        _log = logging.getLogger(__name__)
        _log.info("OSNet: pre-warming Triton kernels for batch sizes 1–%d…", max_batch)
        h, w = self.input_size
        dummy = torch.zeros(max_batch, 3, h, w, device=self.device)
        amp_ctx = torch.autocast(
            device_type="cuda",
            dtype=self._amp_dtype,
            enabled=self._use_amp,
        )
        with amp_ctx:
            # Run the full range in one shot (dynamic=True handles all sizes).
            self._model(dummy)
            # Also exercise batch=1 to ensure that specialisation is cached.
            self._model(dummy[:1])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _log.info("OSNet: pre-warm done")

    @torch.no_grad()
    def extract(self, frame_rgb: np.ndarray, boxes: list) -> np.ndarray:
        """
        Args:
            frame_rgb: (H, W, 3) uint8 RGB
            boxes: list of [x1, y1, x2, y2]
        Returns:
            (N, 512) float32 L2-normalized embeddings
        """
        if not boxes:
            return np.zeros((0, 512), dtype=np.float32)

        H, W = frame_rgb.shape[:2]
        out_h, out_w = self.input_size
        crops: list[torch.Tensor] = []
        valid_idx: list[int] = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(_crop_to_tensor(frame_rgb[y1:y2, x1:x2], out_h, out_w))
            valid_idx.append(i)

        out = np.zeros((len(boxes), 512), dtype=np.float32)
        if not crops:
            return out

        batch = torch.stack(crops).to(self.device, non_blocking=True)

        # BF16 autocast halves memory bandwidth through the OSNet backbone (O7).
        with torch.autocast(
            device_type="cuda" if self._use_amp else "cpu",
            dtype=self._amp_dtype,
            enabled=self._use_amp,
        ):
            feats = self._model(batch)

        feats = torch.nn.functional.normalize(feats.float(), p=2, dim=1)
        feats_np = feats.cpu().numpy().astype(np.float32)

        for k, i in enumerate(valid_idx):
            out[i] = feats_np[k]

        return out
