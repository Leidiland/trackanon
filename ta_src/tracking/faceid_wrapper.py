from __future__ import annotations

import logging

import numpy as np

from ta_src.utils.quiet import suppressed_stdout

log = logging.getLogger(__name__)


def _default_providers() -> list[str]:
    """Prefer CUDA when onnxruntime-gpu reports it as available.

    Falls back to CPU if CUDA isn't usable (no GPU, missing cuDNN/cuBLAS,
    or onnxruntime-gpu not installed). Mirrors the pattern used by
    vitpose_wrapper.py.
    """
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception as e:
        log.debug("ORT provider probe failed (%s) — falling back to CPU", e)
    return ["CPUExecutionProvider"]


def _load_face_analysis(providers: list[str] | None = None):
    """Load InsightFace buffalo_l. Monkeypatch target for tests."""
    from insightface.app import FaceAnalysis

    chosen = providers or _default_providers()
    # InsightFace prints provider lists, model paths, and det-size to stdout.
    with suppressed_stdout():
        app = FaceAnalysis(name="buffalo_l", providers=chosen)
        # ctx_id=0 → GPU 0 when the chosen list starts with a CUDA provider; -1 → CPU.
        ctx_id = 0 if chosen and chosen[0] == "CUDAExecutionProvider" else -1
        app.prepare(ctx_id=ctx_id)
    log.info("InsightFace buffalo_l ready (providers=%s, ctx_id=%d)", chosen, ctx_id)
    return app


class FaceIDWrapper:
    """InsightFace buffalo_l wrapper.

    Per-frame quality = det_score × sqrt(face_area / body_area), used to
    weight each frame's contribution to a track's running face mean.
    """

    def __init__(
        self,
        min_face_width_px: int = 40,
        min_face_det_score: float = 0.6,
        providers: list[str] | None = None,
    ):
        self.min_face_width_px = min_face_width_px
        self.min_face_det_score = min_face_det_score
        try:
            self._app = _load_face_analysis(providers)
        except Exception as e:
            raise RuntimeError(
                f"InsightFace buffalo_l failed to load; pipeline aborts at startup (ADR-0003): {e}"
            ) from e

    def extract(self, body_crop: np.ndarray) -> np.ndarray | None:
        face = self._best_face(body_crop)
        if face is None:
            return None
        return np.asarray(face.normed_embedding, dtype=np.float32)

    def extract_face_obj(self, body_crop: np.ndarray):
        """Return the raw InsightFace face (det_score, pose, normed_embedding)
        for the highest-confidence valid face in `body_crop`, or None.

        Used by the prewarm best-of-N scorer (ADR-0008 addendum) which needs
        det_score and pose alongside the embedding.
        """
        return self._best_face(body_crop)

    def detect_face_bbox(
        self, image: np.ndarray
    ) -> tuple[float, float, float, float] | None:
        """Return the highest-confidence face bbox in image coordinates,
        or None when no usable face is found. Used at KPL build time to
        derive a body bbox so OSNet sees a tight crop (matching runtime)
        instead of the whole reference image."""
        face = self._best_face(image)
        if face is None:
            return None
        return tuple(float(v) for v in face.bbox)

    def extract_with_quality(
        self,
        body_crop: np.ndarray,
        body_bbox: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, float, float] | None:
        """Returns (embedding, quality, det_score) or None.

        det_score is the raw InsightFace per-detection confidence, surfaced
        for the Hungarian face-quality gate (ADR-0018). quality is the area-
        weighted accumulation weight used by the face_emb running mean."""
        face = self._best_face(body_crop)
        if face is None:
            return None
        det_score = float(face.det_score)
        if det_score < self.min_face_det_score:
            return None
        emb = np.asarray(face.normed_embedding, dtype=np.float32)
        quality = det_score * _area_ratio_sqrt(face.bbox, body_bbox)
        return emb, quality, det_score

    def _best_face(self, body_crop: np.ndarray):
        faces = self._app.get(body_crop)
        if not faces:
            return None
        face = max(faces, key=lambda f: float(f.bbox[2]) - float(f.bbox[0]))
        width = float(face.bbox[2]) - float(face.bbox[0])
        if width < self.min_face_width_px:
            return None
        return face


def _area_ratio_sqrt(
    face_bbox: np.ndarray,
    body_bbox: tuple[float, float, float, float],
) -> float:
    fx1, fy1, fx2, fy2 = (float(v) for v in face_bbox)
    bx1, by1, bx2, by2 = (float(v) for v in body_bbox)
    face_area = max(0.0, fx2 - fx1) * max(0.0, fy2 - fy1)
    body_area = max(1e-6, (bx2 - bx1) * (by2 - by1))
    ratio = min(1.0, face_area / body_area)
    return float(np.sqrt(ratio))
