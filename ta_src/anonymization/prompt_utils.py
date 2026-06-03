"""CLIP-based per-identity crop captioner.

Classifies demographic and clothing attributes from a masked person crop. Run
once per global_id on the first high-quality crop; result is cached by the
caller so the model is never invoked more than once per new track.

The captioner intentionally pulls on multiple axes (ethnicity, age band, hair
colour/length, facial hair, glasses, plus garment colour and type). Without
those axes Realistic Vision V6's prior collapses every anonymized identity to
the same young-Caucasian template; with them the prompt space is large enough
to differentiate identities meaningfully.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_GENDER_LABELS  = ["a man", "a woman"]
_ETHNICITY      = ["East Asian", "South Asian", "Black", "Hispanic", "Middle Eastern", "White"]
_AGE_BAND       = ["young adult", "middle-aged", "older"]
_HAIR_COLOR     = ["black", "brown", "blonde", "red", "grey"]
_HAIR_LENGTH    = ["short", "medium-length", "long"]
_FACIAL_HAIR    = ["clean-shaven", "with stubble", "with a full beard"]
_COLOUR_WORDS   = ["red", "blue", "black", "white", "grey", "green", "brown", "beige"]
_GARMENT_WORDS  = ["jacket", "shirt", "hoodie", "dress", "suit", "coat"]


class CropCaptioner:
    """Zero-shot CLIP ViT-B/32 attribute classifier for anonymization prompts."""

    def __init__(self, device: str | None = None):
        import os
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import CLIPModel, CLIPProcessor

        _MODEL_ID = "openai/clip-vit-base-patch32"
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading CLIP ViT-B/32 on %s", self._device)
        # Resolve to local snapshot directory so the is_local code path in
        # transformers picks up model.safetensors directly, bypassing the
        # online check that 404s and falls back to pytorch_model.bin.
        # (transformers >= 5.6 blocks torch.load on torch < 2.6 per CVE-2025-32434)
        try:
            _cfg = hf_hub_download(_MODEL_ID, "config.json", local_files_only=True)
            _src: str = os.path.dirname(_cfg)
            _local_only = True
        except Exception:
            # Cache miss: bootstrap from HF on first run.
            _src = _MODEL_ID
            _local_only = False
        self._model = CLIPModel.from_pretrained(_src, local_files_only=_local_only).to(self._device)
        self._proc = CLIPProcessor.from_pretrained(_src, local_files_only=_local_only)
        self._model.eval()

    def _top1(self, image_pil, texts: list[str]) -> int:
        import torch
        inputs = self._proc(text=texts, images=image_pil, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits_per_image[0]
        return int(logits.argmax())

    def describe(self, crop_rgb: np.ndarray, crop_mask: np.ndarray) -> str:
        """Return a descriptive prompt for the person visible in crop_rgb.

        crop_mask suppresses background pixels before CLIP inference so that
        scene colours don't bias clothing classification.
        """
        from PIL import Image

        masked = crop_rgb.copy()
        masked[~crop_mask] = 0
        img = Image.fromarray(masked)

        gender    = _GENDER_LABELS[self._top1(img, _GENDER_LABELS)]
        ethnicity = _ETHNICITY[self._top1(img, [f"a person of {e} descent" for e in _ETHNICITY])]
        age       = _AGE_BAND[self._top1(img, [f"a {a} person" for a in _AGE_BAND])]
        hair_col  = _HAIR_COLOR[self._top1(img, [f"a person with {c} hair" for c in _HAIR_COLOR])]
        hair_len  = _HAIR_LENGTH[self._top1(img, [f"a person with {l} hair" for l in _HAIR_LENGTH])]
        glasses   = self._top1(img, ["a person wearing glasses", "a person not wearing glasses"]) == 0
        colour    = _COLOUR_WORDS[
            self._top1(img, [f"a person wearing {c} clothing" for c in _COLOUR_WORDS])
        ]
        garment   = _GARMENT_WORDS[
            self._top1(img, [f"a person wearing a {g}" for g in _GARMENT_WORDS])
        ]

        parts = [
            f"a {age} {ethnicity} {gender.removeprefix('a ')}",
            f"with {hair_col} {hair_len} hair",
        ]
        if gender == "a man":
            facial = _FACIAL_HAIR[self._top1(img, [f"a man {fh}" for fh in _FACIAL_HAIR])]
            if facial != "clean-shaven":
                parts.append(facial)
            # Realistic_Vision_V6 drifts toward female-portrait mode on
            # androgynous prompts even with facial-hair tokens present. The
            # masculine anchor is cheap and additive — apply unconditionally
            # for any man, not just the clean-shaven case.
            parts.append("masculine features")
        if glasses:
            parts.append("wearing glasses")
        parts.append(f"wearing a {colour} {garment}")
        parts.append("photorealistic, high quality, warm skin tone, soft natural light, sharp focus")

        return ", ".join(parts)
