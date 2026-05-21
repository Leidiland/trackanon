"""Per-person crop → ComfyUI inpaint → paste-back orchestration."""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from omegaconf import DictConfig

from ta_src.anonymization.comfyui_client import ComfyUIClient
from ta_src.anonymization.pose_utils import (
    face_orientation_factor,
    has_valid_keypoints,
    render_skeleton,
    warp_crop_to_pose,
)
from ta_src.anonymization.color_matcher import ColorMatcher
from ta_src.anonymization.regeneration_gate import RegenerationGate

log = logging.getLogger(__name__)


@dataclass
class _PrepResult:
    """Per-track state threaded through inpaint_frame's pipeline stages."""
    gid: int
    bbox: list = field(default_factory=list)
    mask: Optional[np.ndarray] = None

    # Dispatch:
    kind: str = "submit"          # "skip" or "submit"
    skip_reason: str = ""

    # ROI extraction:
    crop_rgb: Optional[np.ndarray] = None
    crop_mask: Optional[np.ndarray] = None
    roi_bounds: Optional[tuple] = None   # (x1c, y1c, x2c, y2c)

    # Submit path:
    workflow: Optional[dict] = None
    save_node_id: Optional[str] = None
    skeleton: Optional[np.ndarray] = None
    seed: int = 0

    # Concurrent stage outputs:
    prompt_id: Optional[str] = None
    generated_crop: Optional[np.ndarray] = None

    # Paste:
    crop_to_paste: Optional[np.ndarray] = None
    # Keypoints stamped at submit time, cached alongside _prev_crops for G2.
    keypoints: Optional[np.ndarray] = None

# Multiplier used to spread global_id values into a wider seed space so that
# adjacent IDs do not produce visually similar appearances.
_ID_SEED_MULT = 2654435761  # Knuth multiplicative hash constant


def _stable_seed(global_id: int) -> int:
    """Derive a stable 32-bit seed from a global identity ID."""
    return int((global_id * _ID_SEED_MULT) & 0xFFFF_FFFF)


def _snap_to_64(v: int) -> int:
    """Round v up to the nearest multiple of 64 (SD latent grid requirement)."""
    return ((v + 63) // 64) * 64


def extract_roi(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    bbox: list | np.ndarray,
    padding_px: int = 16,
    target_short_side: int = 512,
    max_long_side: int = 768,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """Crop person ROI from the frame and resize to diffusion-friendly dimensions.

    Args:
        image_rgb:        (H, W, 3) uint8 full frame.
        mask:             (H, W) bool, full-frame person mask.
        bbox:             [x1, y1, x2, y2] in frame coordinates.
        padding_px:       extra pixels of context added around the bbox.
        target_short_side: shorter crop dimension after resize.
        max_long_side:    cap on the longer dimension after resize.

    Returns:
        crop_rgb:   (tH, tW, 3) resized crop.
        crop_mask:  (tH, tW) bool mask aligned with crop_rgb.
        x1c, y1c, x2c, y2c:  crop bounds in original frame (before resize).
    """
    fh, fw = image_rgb.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)

    # Expand by padding and clamp.
    x1c = max(0, x1 - padding_px)
    y1c = max(0, y1 - padding_px)
    x2c = min(fw, x2 + padding_px)
    y2c = min(fh, y2 + padding_px)

    cw = x2c - x1c
    ch = y2c - y1c
    if cw <= 0 or ch <= 0:
        raise ValueError(f"Empty crop after padding: bbox={bbox}")

    # Compute target dimensions preserving aspect ratio.
    scale = target_short_side / min(cw, ch)
    tw = min(int(cw * scale), max_long_side)
    th = min(int(ch * scale), max_long_side)

    # Snap to multiples of 64 (SD latent grid).
    tw = _snap_to_64(tw)
    th = _snap_to_64(th)

    crop_rgb = cv2.resize(
        image_rgb[y1c:y2c, x1c:x2c], (tw, th), interpolation=cv2.INTER_LANCZOS4
    )
    crop_mask_uint8 = cv2.resize(
        mask[y1c:y2c, x1c:x2c].astype(np.uint8), (tw, th),
        interpolation=cv2.INTER_NEAREST,
    )
    crop_mask = crop_mask_uint8.astype(bool)

    return crop_rgb, crop_mask, x1c, y1c, x2c, y2c


def composite_result(
    frame: np.ndarray,
    generated_crop: np.ndarray,
    mask: np.ndarray,
    x1c: int, y1c: int, x2c: int, y2c: int,
    blend_mode: str = "hard",
    dilate_px: int = 0,
    erode_px: int = 0,
    color_matcher=None,
    color_match_alpha: float = 0.0,
) -> np.ndarray:
    """Paste the generated person crop back into the frame.

    Args:
        frame:            (H, W, 3) uint8 RGB, modified in-place on a copy.
        generated_crop:   (tH, tW, 3) uint8 RGB from ComfyUI.
        mask:             (H, W) bool, full-frame person mask.
        x1c…y2c:          crop bounds in the original frame.
        blend_mode:       "hard" (fast, ~0.5 ms) or "poisson" (seamless, ~10 ms).
        erode_px:         erode the mask by this many pixels before paste-back
                          (pulls the boundary inward when SAM 3 over-includes
                          background pixels). Applied before dilate_px so equal
                          values cancel to the operator's no-op point.
        dilate_px:        dilate the mask by this many pixels before paste-back
                          (closes hair/edge fringes inside a roughly-correct mask).
    """
    out = frame.copy()
    orig_h, orig_w = y2c - y1c, x2c - x1c

    # Scale generated crop back to original crop pixel dimensions.
    gen_orig = cv2.resize(
        generated_crop, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4
    )

    mask_roi = mask[y1c:y2c, x1c:x2c]

    # LAB color match before paste-back: shift the generated crop's mask-
    # interior color statistics toward the surrounding scene's ring of
    # pixels. See ADR-0017 for the ring-sampling rationale.
    if color_matcher is not None and color_match_alpha > 0.0:
        frame_roi_pre = frame[y1c:y2c, x1c:x2c]
        gen_orig = color_matcher.match(
            gen_orig, frame_roi_pre, mask_roi, alpha=color_match_alpha,
        )

    if erode_px > 0:
        k = 2 * erode_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_roi = cv2.erode(mask_roi.astype(np.uint8), kernel).astype(bool)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_roi = cv2.dilate(mask_roi.astype(np.uint8), kernel).astype(bool)

    pasted = False
    if blend_mode == "poisson" and mask_roi.any():
        mask_uint8 = (mask_roi.astype(np.uint8) * 255)
        frame_roi = out[y1c:y2c, x1c:x2c].copy()
        try:
            blended = cv2.seamlessClone(
                cv2.cvtColor(gen_orig, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(frame_roi, cv2.COLOR_RGB2BGR),
                mask_uint8,
                (orig_w // 2, orig_h // 2),
                cv2.NORMAL_CLONE,
            )
            out[y1c:y2c, x1c:x2c] = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
            pasted = True
        except cv2.error as e:
            log.debug("Poisson blending failed (%s), falling back to hard paste", e)

    if not pasted:
        # Hard paste: copy generated pixels only where mask is True.
        frame_roi = out[y1c:y2c, x1c:x2c]
        frame_roi[mask_roi] = gen_orig[mask_roi]
        out[y1c:y2c, x1c:x2c] = frame_roi

    return out


class DiffusionPipeline:
    """Orchestrates per-person inpainting via ComfyUI for a single frame."""

    def __init__(
        self,
        cfg: DictConfig,
        client: ComfyUIClient,
        crop_cache=None,
        face_id_wrapper=None,
    ):
        self._cfg = cfg
        self._client = client
        self._crop_cache = crop_cache
        self._face_id_wrapper = face_id_wrapper
        self._workflow_default = self._load_workflow(cfg.workflow)
        pose_wf_path = str(cfg.get("workflow_pose", "")).strip()
        self._workflow_pose = (
            self._load_workflow(pose_wf_path) if pose_wf_path else None
        )
        self._node_ids_default = self._workflow_default.get("_node_ids", {})
        self._node_ids_pose = (
            self._workflow_pose.get("_node_ids", {})
            if self._workflow_pose else {}
        )
        # Face polish pass (ADR-0019). Loaded only when enabled to keep
        # default-off behavior bit-equivalent to today's pipeline.
        fp_cfg = cfg.get("face_polish", {}) or {}
        self._face_polish_enabled: bool = bool(fp_cfg.get("enabled", False))
        face_polish_wf_path = str(cfg.get("workflow_face_polish", "")).strip()
        self._workflow_face_polish = (
            self._load_workflow(face_polish_wf_path)
            if self._face_polish_enabled and face_polish_wf_path
            else None
        )
        self._face_polish_face_strength: float = float(fp_cfg.get("face_strength", 1.0))
        self._face_polish_body_strength: float = float(fp_cfg.get("body_strength", 0.0))
        self._face_polish_weight_type: str = str(fp_cfg.get("ipa_weight_type", "linear"))
        self._face_polish_start_at: float = float(fp_cfg.get("ipa_start_at", 0.0))
        self._face_polish_end_at: float = float(fp_cfg.get("ipa_end_at", 1.0))
        self._face_polish_steps: int = int(fp_cfg.get("steps", 12))
        self._face_polish_cfg_scale: float = float(fp_cfg.get("cfg_scale", 6.0))
        self._face_polish_denoise: float = float(fp_cfg.get("denoise", 0.25))
        self._face_polish_pad_ratio: float = float(fp_cfg.get("face_bbox_pad_ratio", 0.25))
        self._face_polish_min_face_px: int = int(fp_cfg.get("min_face_size_px", 96))
        # Cache of generated crops, used only by the composite-skip path
        # (regen_gate decides reuse vs regenerate). Never feeds back as init.
        self._prev_crops: dict[int, np.ndarray] = {}
        # Regen-time keypoints + ROI per gid, used by warp_crop_to_pose to
        # affine-align the cached crop to the current frame's pose on
        # skip-path reuse (ghosting mitigation).
        self._prev_kps: dict[int, np.ndarray] = {}
        self._prev_roi: dict[int, tuple[int, int, int, int]] = {}
        crops_dir = str(cfg.get("save_crops_dir", "")).strip()
        self._crops_dir: Path | None = Path(crops_dir) if crops_dir else None
        self._regen_gate = RegenerationGate(
            max_interval=int(cfg["anon_max_interval"]),
            motion_iou_threshold=float(cfg["anon_motion_iou_threshold"]),
        )

        self._color_matcher = ColorMatcher.from_config(cfg)
        self._color_match_alpha = float(cfg["color_match"]["alpha"])

        self._mask_dilate_px: int = int(cfg["mask_dilate_px"])
        self._mask_erode_px: int = int(cfg["mask_erode_px"])
        self._pose_min_joints: int = int(cfg["pose_min_joints"])

        # G2: affine-warp the cached crop to the current frame's keypoints
        # on regen-gate skip reuse (ghosting mitigation).
        self._keypoint_warp_skip: bool = bool(cfg["keypoint_warp_skip"])
        # F2: pose-aware face_strength scaling. When the keypoints suggest a
        # non-frontal head (profile / facing away), scale face_strength down
        # so the frontal-portrait Reference Crop doesn't dominate the head
        # orientation that ControlNet pose is trying to enforce.
        self._face_strength_pose_adaptive: bool = bool(cfg["face_strength_pose_adaptive"])
        self._face_strength_min_factor: float = float(cfg["face_strength_min_factor"])

        # Concurrent inpaint_frame: overlap per-person HTTP submit/poll with
        # the GPU's serial inference. paint_order_debug emits a DEBUG-level
        # paint-order line per frame; off by default (diagnostic).
        self._max_concurrency: int = max(1, int(cfg["max_concurrency"]))
        self._paint_order_debug: bool = bool(cfg.get("paint_order_debug", False))

        # IP-Adapter readiness gate: hard-abort if node, model, or LoRA is
        # missing — silent fallback weakens the face-anonymisation guarantee.
        self._face_strength: float = float(cfg["face_strength"])
        self._body_strength: float = float(cfg["body_strength"])
        self._lora_strength: float = float(cfg["lora_strength"])
        self._ipa_weight_type: str = str(cfg["ipa_weight_type"])
        self._ipa_start_at: float = float(cfg["ipa_start_at"])
        self._ipa_end_at: float = float(cfg["ipa_end_at"])
        self._ip_adapter_model: str = str(cfg["ip_adapter_model"])
        self._ip_adapter_lora: str = str(cfg["ip_adapter_lora"])
        ipa_path = Path("external/comfyui/models/ipadapter") / self._ip_adapter_model
        lora_path = Path("external/comfyui/models/loras") / self._ip_adapter_lora
        if not client.has_node("IPAdapterFaceID"):
            raise RuntimeError(
                "IPAdapterFaceID custom node not found in the connected ComfyUI server. "
                "Install ComfyUI-IPAdapter-plus (ADR-0004 requires FaceID-Plus-v2 nodes)."
            )
        if not ipa_path.exists():
            raise RuntimeError(
                f"IP-Adapter model file not found at '{ipa_path}'. Download "
                f"'{self._ip_adapter_model}' to external/comfyui/models/ipadapter/ "
                f"(ADR-0004)."
            )
        if not lora_path.exists():
            raise RuntimeError(
                f"FaceID-Plus-v2 LoRA file not found at '{lora_path}'. Download "
                f"'{self._ip_adapter_lora}' to external/comfyui/models/loras/ (ADR-0004)."
            )

    def reset(self):
        """Per-video reset: drop cached gen crops and regen-gate state."""
        self._prev_crops.clear()
        self._prev_kps.clear()
        self._prev_roi.clear()
        self._regen_gate.reset()

    def _steps(self) -> int:
        return int(self._cfg["steps"])

    def _cfg_scale(self) -> float:
        return float(self._cfg["cfg_scale"])

    @staticmethod
    def _load_workflow(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"ComfyUI workflow not found: {p}")
        return json.loads(p.read_text())

    # ------------------------------------------------------------------
    # Per-frame entry point (concurrent submit/collect over N tracks)
    # ------------------------------------------------------------------

    def inpaint_frame(
        self,
        frame_rgb: np.ndarray,
        tracks: list[dict],
        keypoints_per_track: list,
        frame_idx: int = 0,
    ) -> np.ndarray:
        """Replace N tracked persons in the frame with overlapping HTTP latency.

        Pipeline stages:
          1. PREPARE  (sequential) — per-track ROI extraction, regen-gate,
             workflow patching, IPA injection, uploads.
          2. SUBMIT   (concurrent) — queue_prompt per submit-kind prep.
          3. COLLECT  (concurrent) — wait_for_result + fetch per active prep.
          4. CACHE    (sequential, main thread) — _prev_crops, _regen_gate
             writes; no concurrent access to shared state.
          5. PASTE    (sequential, mask-area-ascending) — composite_result.

        Paint order is deterministic mask-area-ascending — smaller masks paste
        first so larger masks win overlapping boundary pixels. Primary
        rationale: temporal stability at overlap rings (iteration order was
        non-deterministic; obj_id rotation flipped overlap-pixel ownership
        between frames, surfacing as flicker). Secondary: Painter's-algorithm
        depth heuristic.

        Always returns a frame — tracks with no mask, failed prep, failed
        submit, or failed collect simply do not contribute to the paste loop.
        """
        if len(tracks) != len(keypoints_per_track):
            raise ValueError(
                f"tracks and keypoints_per_track length mismatch: "
                f"{len(tracks)} vs {len(keypoints_per_track)}"
            )

        t0 = time.monotonic()

        # Step 1: PREPARE (sequential)
        preps: list[_PrepResult] = [
            self._prepare_one(frame_rgb, t, kp, frame_idx)
            for t, kp in zip(tracks, keypoints_per_track)
        ]
        submit_preps = [p for p in preps if p.kind == "submit"]

        # Step 2: SUBMIT (concurrent queue_prompt)
        if submit_preps:
            workers = max(1, min(self._max_concurrency, len(submit_preps)))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                prompt_ids = list(ex.map(self._submit_one, submit_preps))
            for p, prompt_id in zip(submit_preps, prompt_ids):
                p.prompt_id = prompt_id  # None on failure

        # Step 3: COLLECT (concurrent wait + fetch)
        active = [p for p in submit_preps if p.prompt_id is not None]
        if active:
            workers = max(1, min(self._max_concurrency, len(active)))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                crops = list(ex.map(self._collect_one, active))
            for p, crop in zip(active, crops):
                p.generated_crop = crop

        # Step 4: CACHE WRITES (sequential, main thread)
        for p in submit_preps:
            if p.generated_crop is None:
                continue
            if p.gid != -1:
                self._prev_crops[p.gid] = p.generated_crop.copy()
                if p.keypoints is not None:
                    self._prev_kps[p.gid] = p.keypoints
                else:
                    self._prev_kps.pop(p.gid, None)
                if p.roi_bounds is not None:
                    self._prev_roi[p.gid] = tuple(p.roi_bounds)
                else:
                    self._prev_roi.pop(p.gid, None)
                self._regen_gate.record_generation(p.gid, p.bbox, frame_idx)
            if self._crops_dir is not None and p.gid != -1:
                self._save_crops(
                    frame_idx, p.gid, p.crop_rgb, p.generated_crop,
                    crop_mask=p.crop_mask, skeleton=p.skeleton,
                )
            p.crop_to_paste = p.generated_crop

        # Step 5: PASTE (sequential, mask-area-ascending)
        paste_list = [
            p for p in preps
            if (p.crop_to_paste is not None
                and p.mask is not None
                and p.roi_bounds is not None
                and p.crop_mask is not None)
        ]
        paste_list.sort(key=lambda p: int(p.crop_mask.sum()))

        if self._paint_order_debug:
            log.debug(
                "frame=%d paint_order=%s",
                frame_idx,
                [(p.gid, int(p.crop_mask.sum())) for p in paste_list],
            )

        out = frame_rgb
        blend_mode = str(self._cfg["blend_mode"])
        for p in paste_list:
            x1c, y1c, x2c, y2c = p.roi_bounds
            out = composite_result(
                out, p.crop_to_paste, p.mask, x1c, y1c, x2c, y2c,
                blend_mode=blend_mode,
                dilate_px=self._mask_dilate_px,
                erode_px=self._mask_erode_px,
                color_matcher=self._color_matcher,
                color_match_alpha=self._color_match_alpha,
            )

        # Perf log: parsed by tests/manual/analyze_batch_perf.py. n_regen counts
        # submit-path prompts whose generated_crop arrived; n_skip counts every
        # prep that did not submit (no-mask, ROI fail, regen-gate reuse, upload
        # fail). Analysis filters chunk-boundary frames and n_regen < 2.
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        n_regen = sum(1 for p in submit_preps if p.generated_crop is not None)
        n_skip = sum(1 for p in preps if p.kind == "skip")
        log.debug(
            "perf frame=%d anon_ms=%.0f n_regen=%d n_skip=%d max_conc=%d",
            frame_idx, elapsed_ms, n_regen, n_skip, self._max_concurrency,
        )

        return out

    def inpaint_person(
        self,
        frame_rgb: np.ndarray,
        track: dict,
        keypoints: np.ndarray | None,
        frame_idx: int = 0,
    ) -> np.ndarray:
        """N=1 single-track API. Thin wrapper over inpaint_frame.

        Returns the (possibly unchanged) frame. Where the legacy contract
        returned `None` (mask missing, generation failed), this now returns
        the input frame unchanged — semantically equivalent for the only
        consumer pattern `output = result if result is not None else output`.
        """
        return self.inpaint_frame(frame_rgb, [track], [keypoints], frame_idx)

    # ------------------------------------------------------------------
    # Pipeline-stage helpers
    # ------------------------------------------------------------------

    def _prepare_one(
        self,
        frame_rgb: np.ndarray,
        track: dict,
        keypoints,
        frame_idx: int,
    ) -> _PrepResult:
        """Sequential per-track preparation. ROI → regen-gate → upload → patch → IPA."""
        gid = track.get("global_id", -1)
        bbox = track.get("bbox")
        mask = track.get("mask")
        if mask is None:
            return _PrepResult(gid=gid, bbox=bbox or [], kind="skip",
                               skip_reason="no mask")

        seed = _stable_seed(gid) if gid != -1 else int(self._cfg["seed"])
        if seed < 0:
            import random
            seed = random.randint(0, 0xFFFF_FFFF)

        try:
            crop_rgb, crop_mask, x1c, y1c, x2c, y2c = extract_roi(
                frame_rgb, mask, bbox,
                padding_px=int(self._cfg["padding_px"]),
                target_short_side=int(self._cfg["crop_target_short"]),
                max_long_side=int(self._cfg["crop_max_long"]),
            )
        except Exception as e:
            log.debug("ROI extraction failed for gid=%d: %s", gid, e)
            return _PrepResult(gid=gid, bbox=bbox, mask=mask,
                               kind="skip", skip_reason="roi failed")

        # Regen gate: cache hit → skip ComfyUI, paste cached crop.
        if (gid != -1
                and gid in self._prev_crops
                and not self._regen_gate.evaluate(gid, bbox, frame_idx).regenerate):
            crop_to_paste = self._prev_crops[gid]
            if self._keypoint_warp_skip:
                warped = warp_crop_to_pose(
                    cached_crop=crop_to_paste,
                    prev_kps=self._prev_kps.get(gid),
                    prev_roi=self._prev_roi.get(gid),
                    cur_kps=keypoints,
                    cur_roi=(x1c, y1c, x2c, y2c),
                )
                if warped is not None:
                    crop_to_paste = warped
            return _PrepResult(
                gid=gid, bbox=bbox, mask=mask,
                crop_mask=crop_mask, roi_bounds=(x1c, y1c, x2c, y2c),
                kind="skip", skip_reason="regen gate reuse",
                crop_to_paste=crop_to_paste,
            )

        th, tw = crop_rgb.shape[:2]
        valid_joints = (
            int((np.asarray(keypoints)[:, 2] >= 0.3).sum())
            if keypoints is not None and len(keypoints) else 0
        )
        use_pose = (
            self._workflow_pose is not None
            and has_valid_keypoints(keypoints, min_joints=self._pose_min_joints)
        )

        # Face polish prep (ADR-0019). Only when enabled, pose pass is active,
        # and the template + wrapper are wired. NO_FACE / SUB_RESOLUTION fall
        # through to the body-pass-only path the regen would have taken.
        face_polish_active = False
        face_mask_in_crop: np.ndarray | None = None
        face_polish_status = "disabled"
        if (
            getattr(self, "_face_polish_enabled", False)
            and use_pose
            and getattr(self, "_workflow_face_polish", None) is not None
            and getattr(self, "_face_id_wrapper", None) is not None
        ):
            from ta_src.anonymization.face_mask_builder import (
                FaceMaskStatus,
                build_face_mask,
            )
            fm = build_face_mask(
                face_id_wrapper=self._face_id_wrapper,
                frame_rgb=frame_rgb,
                body_bbox=bbox,
                person_mask_in_crop=crop_mask,
                crop_bounds=(x1c, y1c, x2c, y2c),
                pad_ratio=float(self._face_polish_pad_ratio),
                min_face_size_px=int(self._face_polish_min_face_px),
            )
            face_polish_status = fm.status.value
            if fm.status is FaceMaskStatus.OK:
                face_polish_active = True
                face_mask_in_crop = fm.mask

        if face_polish_active:
            workflow_template = self._workflow_face_polish
            node_ids = self._workflow_face_polish.get("_node_ids", {})
        else:
            workflow_template = self._workflow_pose if use_pose else self._workflow_default
            node_ids = self._node_ids_pose if use_pose else self._node_ids_default
        has_mask_node = bool(node_ids.get("load_mask"))
        save_node_id = node_ids.get("save_image", "11")
        strength = (
            float(self._cfg["strength_pose"])
            if use_pose
            else float(self._cfg["strength"])
        )

        # init_rgb is always the current frame's crop (ADR-0009).
        tag = f"gid{gid}"
        skeleton: np.ndarray | None = None
        try:
            img_name = self._client.upload_image(crop_rgb, f"crop_{tag}.png")
            mask_name: str | None = None
            if has_mask_node:
                mask_name = self._client.upload_mask(crop_mask, f"mask_{tag}.png")
            pose_name: str | None = None
            if use_pose:
                skeleton = render_skeleton(
                    keypoints, x1c, y1c, x2c - x1c, y2c - y1c, tw, th
                )
                pose_name = self._client.upload_image(skeleton, f"pose_{tag}.png")
            face_mask_name: str | None = None
            if face_polish_active and face_mask_in_crop is not None:
                face_mask_name = self._client.upload_mask(
                    face_mask_in_crop, f"face_mask_{tag}.png"
                )
        except Exception as e:
            log.warning("Upload failed for gid=%d: %s", gid, e)
            return _PrepResult(gid=gid, bbox=bbox, mask=mask,
                               crop_mask=crop_mask,
                               roi_bounds=(x1c, y1c, x2c, y2c),
                               kind="skip", skip_reason="upload failed")

        prompt = (
            track.get("prompt")
            or str(self._cfg["prompt"])
        )

        workflow = self._client.patch_workflow(
            workflow_template,
            image_name=img_name,
            mask_name=mask_name,
            pose_name=pose_name,
            seed=seed,
            prompt=prompt,
            neg_prompt=str(self._cfg["negative_prompt"]),
            denoise=strength,
            steps=self._steps(),
            cfg_scale=self._cfg_scale(),
            output_prefix=f"anon_{tag}",
            node_ids=node_ids,
            controlnet_strength=float(self._cfg["controlnet_strength"]) if use_pose else None,
            controlnet_end_percent=float(self._cfg["controlnet_end_percent"]) if use_pose else None,
        )

        if face_polish_active:
            face_ks = node_ids.get("ksampler_face")
            face_mask_node = node_ids.get("load_face_mask")
            if face_ks and face_ks in workflow:
                workflow[face_ks]["inputs"]["seed"] = seed
                workflow[face_ks]["inputs"]["steps"] = int(self._face_polish_steps)
                workflow[face_ks]["inputs"]["cfg"] = float(self._face_polish_cfg_scale)
                workflow[face_ks]["inputs"]["denoise"] = float(self._face_polish_denoise)
            if face_mask_node and face_mask_node in workflow and face_mask_name:
                workflow[face_mask_node]["inputs"]["image"] = face_mask_name

        # IP-Adapter path (ADR-0004).
        reference_crop = (
            self._crop_cache.load(gid)
            if self._crop_cache is not None and gid != -1
            else None
        )
        ipa_injected = False
        face_factor = (
            face_orientation_factor(
                keypoints, min_factor=self._face_strength_min_factor,
                face_emb_present=track.get("face_emb_present"),
            )
            if self._face_strength_pose_adaptive else 1.0
        )
        face_strength_effective = self._face_strength * face_factor
        if reference_crop is not None:
            try:
                from ta_src.anonymization.ipadapter_injector import (
                    ApplyConfig,
                    inject as _ipa_inject,
                )
                ref_name = self._client.upload_image(reference_crop, f"ref_{tag}.png")
                if face_polish_active:
                    apply_configs = [
                        ApplyConfig(
                            target_ksampler_id=node_ids["ksampler_body"],
                            face_strength=face_strength_effective,
                            body_strength=self._body_strength,
                            weight_type=self._ipa_weight_type,
                            start_at=self._ipa_start_at,
                            end_at=self._ipa_end_at,
                        ),
                        ApplyConfig(
                            target_ksampler_id=node_ids["ksampler_face"],
                            face_strength=float(self._face_polish_face_strength),
                            body_strength=float(self._face_polish_body_strength),
                            weight_type=str(self._face_polish_weight_type),
                            start_at=float(self._face_polish_start_at),
                            end_at=float(self._face_polish_end_at),
                        ),
                    ]
                    workflow = _ipa_inject(
                        workflow,
                        ref_name,
                        lora_strength=self._lora_strength,
                        lora_filename=self._ip_adapter_lora,
                        apply_configs=apply_configs,
                    )
                else:
                    workflow = _ipa_inject(
                        workflow,
                        ref_name,
                        face_strength=face_strength_effective,
                        body_strength=self._body_strength,
                        lora_strength=self._lora_strength,
                        lora_filename=self._ip_adapter_lora,
                        weight_type=self._ipa_weight_type,
                        start_at=self._ipa_start_at,
                        end_at=self._ipa_end_at,
                    )
                ipa_injected = True
            except Exception as e:
                log.warning("IP-Adapter injection failed for gid=%d (%s) — using base workflow", gid, e)

        log.info(
            "inpaint gid=%d frame=%d workflow=%s face_polish=%s denoise=%.2f ipa=%s "
            "joints=%d/%d face_factor=%.2f face_strength=%.3f ref=%s prompt=%r",
            gid, frame_idx,
            "face_polish" if face_polish_active else ("pose" if use_pose else "default"),
            face_polish_status,
            strength,
            "yes" if ipa_injected else "no",
            valid_joints, self._pose_min_joints,
            face_factor, face_strength_effective,
            "yes" if reference_crop is not None else "no",
            prompt[:80],
        )

        return _PrepResult(
            gid=gid, bbox=bbox, mask=mask,
            crop_rgb=crop_rgb, crop_mask=crop_mask,
            roi_bounds=(x1c, y1c, x2c, y2c),
            kind="submit",
            workflow=workflow,
            save_node_id=save_node_id,
            skeleton=skeleton,
            seed=seed,
            keypoints=(
                np.asarray(keypoints, dtype=np.float32).copy()
                if keypoints is not None else None
            ),
        )

    def _submit_one(self, prep: _PrepResult) -> str | None:
        """Concurrent worker: queue_prompt. Returns prompt_id or None on failure."""
        try:
            return self._client.queue_prompt(prep.workflow)
        except Exception as e:
            log.warning("queue_prompt failed for gid=%d: %s", prep.gid, e)
            return None

    def _collect_one(self, prep: _PrepResult) -> np.ndarray | None:
        """Concurrent worker: wait_for_result + fetch. Returns crop or None on failure."""
        try:
            history = self._client.wait_for_result(
                prep.prompt_id,
                timeout=float(self._cfg["timeout"]),
            )
        except Exception as e:
            log.warning("wait_for_result failed for gid=%d: %s", prep.gid, e)
            return None

        img_info = self._client.extract_output_image(history, prep.save_node_id)
        if img_info is None:
            log.warning("No output image for gid=%d, prompt_id=%s",
                        prep.gid, prep.prompt_id)
            return None

        try:
            return self._client.fetch_image(
                img_info["filename"],
                subfolder=img_info.get("subfolder", ""),
                img_type=img_info.get("type", "output"),
            )
        except Exception as e:
            log.warning("fetch_image failed for gid=%d: %s", prep.gid, e)
            return None

    def _save_crops(
        self,
        frame_idx: int,
        gid: int,
        orig_crop: np.ndarray,
        gen_crop: np.ndarray,
        crop_mask: np.ndarray | None = None,
        skeleton: np.ndarray | None = None,
    ):
        try:
            self._crops_dir.mkdir(parents=True, exist_ok=True)
            stem = f"f{frame_idx:06d}_gid{gid:04d}"
            cv2.imwrite(
                str(self._crops_dir / f"{stem}_orig.png"),
                cv2.cvtColor(orig_crop, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                str(self._crops_dir / f"{stem}_gen.png"),
                cv2.cvtColor(gen_crop, cv2.COLOR_RGB2BGR),
            )
            if crop_mask is not None:
                cv2.imwrite(
                    str(self._crops_dir / f"{stem}_mask.png"),
                    (crop_mask.astype(np.uint8) * 255),
                )
            if skeleton is not None:
                cv2.imwrite(
                    str(self._crops_dir / f"{stem}_pose.png"),
                    cv2.cvtColor(skeleton, cv2.COLOR_RGB2BGR),
                )
        except Exception as exc:
            log.debug("Failed to save crops for frame=%d gid=%d: %s", frame_idx, gid, exc)
