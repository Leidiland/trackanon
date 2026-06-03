"""Pose-wrapper construction behind a factory seam keyed by `cfg.kind`.

DWpose (rtmlib RTMW-DW-x-l) is the sole pose backend — it drives both
tracking keypoints (COCO-17 via `.run()`) and the diffusion ControlNet
skeleton (`.render()`). The seam keeps call sites backend-agnostic so a
future pose model can be added without touching them. Returns None when no
pose config is present.
"""
from __future__ import annotations

from ta_src.pose.dwpose_wrapper import DWposeWrapper


def build_poser(cfg, device: str):
    if cfg is None:
        return None
    kind = str(cfg.get("kind", "dwpose"))
    if kind == "dwpose":
        return DWposeWrapper.from_config(cfg, device)
    raise ValueError(f"unknown pose kind={kind!r} — expected 'dwpose'")
