"""Standalone per-image tooling — no tracking, no temporal state.

Blur every detected person, or render per-stage overlay figures
(masks / detections / track / keypoints) on still images. Identity for the
'track' overlay is assigned per image by face detection alone, matched to a KPL
gallery identity; nothing carries across frames. Persona diffusion is not
offered for stills — anonymisation runs through the video pipeline's windowed
VACE backend, which operates on clips, not single frames.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ta_src.segmentation.sam3_wrapper import scale_rows_to_frame
from ta_src.utils import visualization as vis
from ta_src.video.sam3_frame_workspace import Sam3FrameWorkspace

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def _bbox_centre(bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _contains(bbox, point) -> bool:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    px, py = point
    return x1 <= px <= x2 and y1 <= py <= y2


def _face_for_person(person_bbox, faces):
    # The face whose centre lands inside the person bbox; widest wins ties,
    # mirroring FaceIDWrapper._best_face.
    contained = [f for f in faces if _contains(person_bbox, _bbox_centre(f.bbox))]
    if not contained:
        return None
    return max(contained, key=lambda f: float(f.bbox[2]) - float(f.bbox[0]))


def make_sam3_rows_fn(sam3, tmp_root):
    """Return a callable that runs SAM 3 on one RGB image and yields every
    person row (mask + bbox + score), scaled to frame coords.

    Each call spins a fresh 1-frame workspace and session, then frees SAM 3's
    VRAM — same lifecycle as the chunked stage."""
    tmp_root = Path(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)
    counter = {"i": 0}

    def _rows(image_rgb):
        chunk_id = counter["i"]
        counter["i"] += 1
        ws = Sam3FrameWorkspace(parent_dir=tmp_root, chunk_id=chunk_id)
        try:
            ws.write_frames([image_rgb])
            rows_per_frame = sam3.process_chunk(ws.path, 0)
        finally:
            sam3.close_session_and_empty_cache()
            ws.close()
        if not rows_per_frame or not rows_per_frame[0]:
            return []
        rows = rows_per_frame[0]
        scale_rows_to_frame(rows, image_rgb.shape[:2])
        return rows

    return _rows


def assign_persons(person_rows, faces, gallery, *, face_match_floor: float):
    """Return one track dict per person row, stamped with a gallery global_id.

    A person with no contained face, or whose best gallery cosine falls below
    `face_match_floor`, gets global_id=-1. Used by the 'track' overlay."""
    identities = list(gallery)
    tracks: list[dict] = []
    for row in person_rows:
        track = dict(row)
        track.setdefault("mask_source", "detection")
        track["global_id"] = -1
        track["name"] = ""
        track["face_centroid"] = None
        track["face_emb_present"] = False
        track["match_cos"] = None

        face = _face_for_person(row["bbox"], faces)
        if face is not None:
            emb = np.asarray(face.normed_embedding, dtype=np.float32)
            track["face_emb_present"] = True
            best_gid, best_name, best_centroid, best_cos = -1, "", None, -1.0
            for ident in identities:
                cos = float(np.dot(emb, ident.face_centroid))
                if cos > best_cos:
                    best_gid, best_name = ident.global_id, ident.name
                    best_centroid, best_cos = ident.face_centroid, cos
            track["match_cos"] = best_cos
            if best_cos >= face_match_floor:
                track["global_id"] = best_gid
                track["name"] = best_name
                track["face_centroid"] = best_centroid
        tracks.append(track)
    return tracks


def anonymize_image_array(image_rgb, *, sam3_rows_fn, fallback, name: str = "image"):
    """Blur every detected person in one RGB image. No tracking, no diffusion —
    needs only SAM 3 and the FallbackAnonymizer."""
    rows = sam3_rows_fn(image_rgb)
    output = image_rgb.copy()
    for row in rows:
        mask = row.get("mask")
        fallback.apply(output, mask if mask is not None else row["bbox"])
    return output


def run_flags_for_stages(stages) -> dict:
    """Pipeline run-flags needed to build the components the stages require —
    track loads face+gallery, keypoints loads pose. No stage loads the windowed
    VACE backend (still images aren't anonymised by diffusion)."""
    stages = set(stages)
    return {
        "run_tracking": "track" in stages,
        "run_pose": "keypoints" in stages,
        "run_anonymization": False,
    }


_OVERLAY_LAYERS = ("masks", "detections", "track", "keypoints")


def render_image_figures(
    image_rgb,
    *,
    sam3_rows_fn,
    figures,
    face=None,
    gallery=None,
    poser=None,
    face_match_floor: float = 0.5,
    label_scale: float = 1.0,
    name: str = "image",
) -> dict:
    """Render composite overlay figures on one RGB image.

    `figures` maps a figure name to an ordered list of overlay layers from
    {masks, detections, track, keypoints}; each figure stacks its layers on a
    single canvas the way --save-visualization composes overlays.

    SAM 3 runs once; face recognition and pose are each computed once and shared
    across every figure, and collaborators are touched only by the layers that
    need them."""
    figures = {fname: list(layers) for fname, layers in figures.items()}
    used = set().union(*figures.values()) if figures else set()
    unknown = used.difference(_OVERLAY_LAYERS)
    if unknown:
        raise ValueError(f"Unknown overlay layer(s): {sorted(unknown)}")

    if "track" in used and (face is None or gallery is None):
        raise ValueError("The 'track' stage needs face and gallery to assign gids.")
    if "keypoints" in used and poser is None:
        raise ValueError("The 'keypoints' stage needs a poser.")

    rows = sam3_rows_fn(image_rgb)
    tracks = None
    if "track" in used:
        faces = face.detect_faces(image_rgb)
        tracks = assign_persons(rows, faces, gallery,
                                face_match_floor=face_match_floor)
    keypoints = None
    if "keypoints" in used:
        keypoints = poser.run(image_rgb, tracks if tracks is not None else rows)

    draw = {
        "masks": lambda c: vis._draw_masks(c, rows),
        "detections": lambda c: vis._draw_detections(c, rows, label_scale=label_scale),
        "track": lambda c: vis._draw_tracks(c, tracks, label_scale=label_scale),
        "keypoints": lambda c: vis.draw_pose_overlay(c, keypoints),
    }
    outputs: dict = {}
    for fname, layers in figures.items():
        canvas = image_rgb.copy()
        for layer in layers:
            draw[layer](canvas)
        outputs[fname] = canvas
    return outputs


def render_image_stages(image_rgb, *, sam3_rows_fn, stages, **kwargs) -> dict:
    """Render one overlay per requested atomic stage — a thin adapter over
    render_image_figures where each stage is a single-layer figure keyed by its
    own name."""
    stages = set(stages)
    overlay = stages & set(_OVERLAY_LAYERS)
    return render_image_figures(
        image_rgb, sam3_rows_fn=sam3_rows_fn,
        figures={s: [s] for s in overlay}, **kwargs,
    )


def _build_fallback(acfg):
    from ta_src.anonymization.fallback_anonymizer import FallbackAnonymizer
    return FallbackAnonymizer(
        kernel_min=int(acfg.get("fallback_kernel_min", 51)),
        kernel_frac=float(acfg.get("fallback_kernel_frac", 0.20)),
        dilate_min=int(acfg.get("fallback_dilate_min", 5)),
        dilate_frac=float(acfg.get("fallback_dilate_frac", 0.025)),
        feather_min=int(acfg.get("fallback_feather_min", 5)),
        feather_frac=float(acfg.get("fallback_feather_frac", 0.020)),
    )


def build_process_fn(cfg, *, face_match_floor: float = 0.5):
    """Construct the per-image blur anonymizer for the given config.

    Blur-only: SAM 3 alone, no gallery, face model, or ComfyUI. (Persona
    diffusion for stills is not supported — the VACE backend operates on clips.)"""
    fallback = _build_fallback(cfg.anonymization)
    device = cfg.pipeline.device
    tmp_root = Path(cfg.paths.get("temp", "data/temp")) / "sam3_image"

    from ta_src.segmentation.sam3_wrapper import SAM3ChunkedStage
    sam3 = SAM3ChunkedStage.from_config(cfg.sam3, device)
    rows_fn = make_sam3_rows_fn(sam3, tmp_root)

    def process_fn(rgb, name):
        return anonymize_image_array(rgb, sam3_rows_fn=rows_fn, fallback=fallback)
    return process_fn


def build_render_fn(cfg, figures, *, face_match_floor: float = 0.5,
                    label_scale: float = 1.0):
    """Build a render_fn(rgb, name) -> {figure: rgb} for the requested figures.

    Only the components the figures' layers need are loaded
    (run_flags_for_stages), so masks/detections need just SAM 3 while track/
    keypoints add the identity/pose models. Reuses the Pipeline construction;
    the resolver/OSNet load with tracking but stay unused (image mode does
    identity by face alone)."""
    figures = {fname: list(layers) for fname, layers in figures.items()}
    used = set().union(*figures.values()) if figures else set()
    flags = run_flags_for_stages(used)
    cfg.pipeline.run_tracking = flags["run_tracking"]
    cfg.pipeline.run_pose = flags["run_pose"]
    cfg.pipeline.run_anonymization = flags["run_anonymization"]

    from ta_src.pipeline.main_pipeline import Pipeline
    pipe = Pipeline(cfg)
    tmp_root = Path(cfg.paths.get("temp", "data/temp")) / "sam3_image"
    rows_fn = make_sam3_rows_fn(pipe._sam3_stage, tmp_root)

    def render_fn(rgb, name):
        return render_image_figures(
            rgb, sam3_rows_fn=rows_fn, figures=figures,
            face=pipe._face, gallery=pipe._gallery, poser=pipe._poser,
            face_match_floor=face_match_floor,
            label_scale=label_scale, name=name,
        )
    return render_fn


def iter_image_files(input_path) -> list[Path]:
    """Image files for `input_path` — the file itself, or every image in the
    directory (sorted). Non-image entries are skipped."""
    p = Path(input_path)
    if p.is_dir():
        return sorted(
            c for c in p.iterdir()
            if c.is_file() and c.suffix.lower() in _IMAGE_EXTS
        )
    if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
        return [p]
    return []


def anonymize_path(input_path, output_dir, process_fn) -> list[Path]:
    """Anonymize a single image or a folder of images, writing each result to
    `output_dir` under its original filename. `process_fn(rgb, name)` returns
    the anonymized RGB frame. Returns the paths written."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for path in iter_image_files(input_path):
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out_rgb = process_fn(rgb, path.stem)
        out_path = out_dir / path.name
        cv2.imwrite(str(out_path), cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
        written.append(out_path)
    return written


def render_stages_path(input_path, output_dir, render_fn) -> list[Path]:
    """Render per-stage overlays for an image or folder, writing each stage to
    its own subfolder (`output_dir/<stage>/<filename>`) so the figures stay
    independent. `render_fn(rgb, name)` returns a {stage: rgb} mapping. Returns
    the paths written."""
    out_dir = Path(output_dir)
    written: list[Path] = []
    for path in iter_image_files(input_path):
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for stage, stage_rgb in render_fn(rgb, path.stem).items():
            stage_dir = out_dir / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            out_path = stage_dir / path.name
            cv2.imwrite(str(out_path), cv2.cvtColor(stage_rgb, cv2.COLOR_RGB2BGR))
            written.append(out_path)
    return written
