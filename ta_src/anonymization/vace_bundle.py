"""Assemble a Wan-VACE generation bundle from the two-pass track cache.

A bundle is a crop-space, VACE-sized clip for one gid over one window:
  - control video: source crop with the gid's silhouette greyed (grey-control —
    forces VACE to drive appearance from the reference, not the original pixels)
  - mask video:    the silhouette, white = regenerate
  - reference:     the persona crop, letterboxed to the canvas aspect
The window is 4n+1 frames (VACE length constraint); the crop is the per-window
union bbox, padded and snapped to x16.
"""
from __future__ import annotations

import dataclasses

import cv2
import numpy as np

# Neutral grey VACE reads as "no content" in the masked region.
_GREY = 127


def grey_control_frame(crop_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Grey the masked region of the control crop so VACE drives appearance from
    the reference, not the original subject pixels (the grey-control recipe)."""
    out = crop_rgb.copy()
    out[np.asarray(mask).astype(bool)] = _GREY
    return out


# Wholebody body joints (shoulders, hips) — a usable structural skeleton needs
# these; a face-only skeleton degenerates VACE into a dark structureless blob.
_BODY_JOINTS = (5, 6, 11, 12)


def _has_body_joints(kpf, thr: float = 0.3) -> bool:
    """True if ≥2 body joints (shoulders/hips) are visible. When false, the
    pose path falls back to grey-control rather than feed VACE a face-only
    skeleton that collapses to a blob."""
    if kpf is None or len(kpf) <= max(_BODY_JOINTS):
        return False
    return sum(1 for j in _BODY_JOINTS if kpf[j][2] > thr) >= 2


def _gid_keypoints_full(reader, frame_idx: int, gid: int):
    """The gid's full-frame wholebody keypoints at this frame, or None. The
    cache stores keypoints index-aligned with tracks (write(frame, tracks, kps))."""
    tracks, kps = reader.read(frame_idx)
    for i, t in enumerate(tracks):
        if int(t.get("global_id", -1)) == int(gid) and i < len(kps):
            kpf = kps[i].get("keypoints_full")
            if kpf is None:
                kpf = kps[i].get("keypoints")
            return np.asarray(kpf) if kpf is not None else None
    return None


def pose_control_frame(
    ctrl_rgb, mask_bool, kpf, crop_box, canvas_hw, *,
    mask_dilate_px: int = 0, draw_face_hands: bool = True,
):
    """Pose-in-the-hole control: grey the silhouette (kill original appearance,
    as grey-control does) and draw the gid's pose skeleton inside it, so VACE
    regenerates the persona in the *original pose* and fills the silhouette
    instead of inventing a divergent pose. Scene outside the mask is untouched.
    The skeleton renderer gates keypoints to the mask, so nothing leaks past the
    silhouette onto the preserved scene."""
    from ta_src.pose.dwpose_wrapper import render_skeleton_from_keypoints
    mask_bool = np.asarray(mask_bool).astype(bool)
    out = ctrl_rgb.copy()
    out[mask_bool] = _GREY
    skel = render_skeleton_from_keypoints(
        np.asarray(kpf),
        (float(crop_box[0]), float(crop_box[1]), float(crop_box[2]), float(crop_box[3])),
        (int(canvas_hw[0]), int(canvas_hw[1])),
        draw_face_hands=draw_face_hands,
        mask_crop=mask_bool, mask_dilate_px=mask_dilate_px,
    )
    # Confine the skeleton to the silhouette: keypoint-gating drops out-of-mask
    # joints, but anti-aliased limb lines between in-mask joints can still bow a
    # few px past the edge — AND with the mask so nothing paints on the
    # preserved scene.
    drawn = skel.any(axis=2) & mask_bool
    out[drawn] = skel[drawn]
    return out


def pose_gen_control_frame(kpf, crop_box, canvas_hw, *, draw_face_hands: bool = True):
    """Full-canvas pose skeleton on black for the pose_gen generate-then-composite
    path: no scene pixels, no grey hole, no mask gating (the companion mask is
    all-white). VACE generates a clean posed persona over a hallucinated
    background that the stitch mattes out — so the skeleton can't bake into the
    preserved scene the way the in-the-hole pose path does."""
    from ta_src.pose.dwpose_wrapper import render_skeleton_from_keypoints
    return render_skeleton_from_keypoints(
        np.asarray(kpf),
        (float(crop_box[0]), float(crop_box[1]), float(crop_box[2]), float(crop_box[3])),
        (int(canvas_hw[0]), int(canvas_hw[1])),
        draw_face_hands=draw_face_hands, mask_crop=None,
    )


def letterbox_reference(ref_rgb: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Pad the persona reference to the canvas aspect (fill grey, content
    centred) so WanVaceToVideo's center-crop resize can't shave off the
    head/trousers when a portrait ref meets a wider canvas."""
    W, H = size_wh
    rh, rw = ref_rgb.shape[:2]
    s = min(W / rw, H / rh)
    nw, nh = int(round(rw * s)), int(round(rh * s))
    out = np.full((H, W, 3), _GREY, dtype=np.uint8)
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(
        ref_rgb, (nw, nh), interpolation=cv2.INTER_AREA
    )
    return out


@dataclasses.dataclass
class VaceWindow:
    gid: int
    frames: list[int]
    crop_box: tuple[float, float, float, float]  # padded union bbox, frame coords
    size: tuple[int, int]                          # (W, H), snapped to x16


def round16(x) -> int:
    return max(16, int(round(x / 16.0)) * 16)


def _vace_length(n: int) -> int:
    # Largest 4k+1 not exceeding n (VACE requires length == 4k+1).
    return max(1, ((n - 1) // 4) * 4 + 1)


def list_confirmed_gids(reader, *, thresholds=None) -> list[int]:
    """Sorted unique gids >= 0 present in the cache. Unbound (-1) is excluded
    (no reference persona). When `thresholds` is provided, a gid is included
    only if at least one of its cached AssignmentInfo entries passes the
    confidence gate, or is operator-assigned — so low-confidence AUTO bindings
    don't get pasted with the wrong persona (they inherit fallback-blur), while
    a human-vouched label is always honoured even below the face floor."""
    if thresholds is not None:
        from ta_src.anonymization.confidence_gate import evaluate as _evaluate
    gids: set[int] = set()
    for k in reader.frame_indices():
        tracks = reader.read_meta(k)          # gid enumeration needs no masks
        for t in tracks:
            g = t.get("global_id")
            if g is None or int(g) < 0:
                continue
            if thresholds is None:
                gids.add(int(g))
                continue
            if int(g) in gids:
                continue                       # already confirmed earlier in cache
            if t.get("operator_assigned"):
                gids.add(int(g))               # operator vouched for the identity; the gate only screens auto bindings
                continue
            info = t.get("assignment_info")
            if info is not None and _evaluate(info, thresholds).confirmed:
                gids.add(int(g))
    return sorted(gids)


def _gid_bbox(reader, frame_idx: int, gid: int):
    # read_meta skips RLE mask decode — planning only needs presence + bbox.
    for t in reader.read_meta(frame_idx):
        if t.get("global_id") == gid and t.get("has_mask"):
            return t["bbox"]
    return None


def _union_bbox(boxes, pad: float):
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    w, h = x1 - x0, y1 - y0
    return (x0 - w * pad, y0 - h * pad, x1 + w * pad, y1 + h * pad)


def _crop_to_canvas(img, box, size_wh, interp):
    """Crop `img` to the (float) box, padding to the unclamped box so every
    frame shares one canvas, then resize to size_wh."""
    W, H = size_wh
    ix0, iy0 = int(np.floor(box[0])), int(np.floor(box[1]))
    ix1, iy1 = int(np.ceil(box[2])), int(np.ceil(box[3]))
    fh, fw = img.shape[:2]
    cx0, cy0 = max(0, ix0), max(0, iy0)
    cx1, cy1 = min(fw, ix1), min(fh, iy1)
    region = img[cy0:cy1, cx0:cx1]
    canvas_shape = (iy1 - iy0, ix1 - ix0) + img.shape[2:]
    canvas = np.zeros(canvas_shape, dtype=img.dtype)
    canvas[cy0 - iy0:cy0 - iy0 + region.shape[0],
           cx0 - ix0:cx0 - ix0 + region.shape[1]] = region
    return cv2.resize(canvas, (W, H), interpolation=interp)


def _gid_mask(reader, frame_idx, gid):
    t = _gid_track(reader, frame_idx, gid)
    return t.get("mask") if t is not None else None


def _gid_track(reader, frame_idx, gid):
    tracks, _ = reader.read(frame_idx)
    for t in tracks:
        if t.get("global_id") == gid:
            return t
    return None


def _bbox_diag(box) -> float:
    if box is None:
        return 1.0
    x0, y0, x1, y1 = box
    w, h = max(1.0, float(x1) - float(x0)), max(1.0, float(y1) - float(y0))
    return float(np.hypot(w, h))


def _bbox_center(box):
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return (0.5 * (float(x0) + float(x1)), 0.5 * (float(y0) + float(y1)))


def _within_motion_eps(cur_bbox, last_bbox, eps_frac: float) -> bool:
    c, l = _bbox_center(cur_bbox), _bbox_center(last_bbox)
    if c is None or l is None:
        return False
    delta = float(np.hypot(c[0] - l[0], c[1] - l[1]))
    return delta <= eps_frac * _bbox_diag(last_bbox)


def select_paint_mask(
    reader, frame_idx: int, gid: int, *, state: dict,
    quality_thresholds: dict | None = None, motion_guard_eps: float = 0.15,
):
    """Decide which silhouette to paint for `gid` at `frame_idx`. Gates the
    cached mask (`quality_thresholds`, optional) and falls back to the last
    good mask when the bbox centre hasn't moved much. Returns the full-frame
    mask to use (None if there's no usable mask). `state` is mutated in place
    to carry (`last_mask`, `last_bbox`) across calls per gid."""
    from ta_src.anonymization.mask_quality import Pass, mask_quality_check

    t = _gid_track(reader, frame_idx, gid)
    cur_bbox = t.get("bbox") if t else None
    m = t.get("mask") if t else None

    if m is not None and quality_thresholds is not None:
        decision = mask_quality_check(
            mask=m, mask_score=t.get("mask_score"),
            bbox=cur_bbox if cur_bbox is not None else (0, 0, 0, 0),
            mask_source=t.get("mask_source", "detection"),
            ratio_floor=float(quality_thresholds["ratio_floor"]),
            score_floor=float(quality_thresholds["score_floor"]),
            score_pass_override=quality_thresholds.get("score_pass_override"),
        )
        if not isinstance(decision, Pass):
            m = None

    last_mask = state.get("last_mask")
    last_bbox = state.get("last_bbox")
    if m is None and last_mask is not None and _within_motion_eps(
        cur_bbox, last_bbox, motion_guard_eps,
    ):
        m = last_mask                          # carry over a recent good mask
    elif m is not None:
        state["last_mask"], state["last_bbox"] = m, cur_bbox
    return m


def build_frames(
    window: VaceWindow, frames_rgb, reader, *, grey_control: bool = True,
    quality_thresholds: dict | None = None, motion_guard_eps: float = 0.15,
    preserve_overlap: dict | None = None,
    control_mode: str = "grey", pose_mask_dilate_px: int = 0,
):
    """Per-window control + mask frames in canvas space. `preserve_overlap`
    maps cache frame index -> full-frame RGB; those positions are emitted with
    mask=0 (preserve) and control crop taken from the preserved frame, so VACE
    keeps them as-is and temporally conditions the regenerated rest of the
    window on them (cross-window identity carry-over)."""
    preserve = preserve_overlap or {}
    control, masks, state = [], [], {}
    for k in window.frames:
        if k in preserve:
            ctrl = _crop_to_canvas(preserve[k], window.crop_box, window.size, cv2.INTER_AREA)
            mcrop = np.zeros(window.size[::-1], np.uint8)
            control.append(ctrl)
            masks.append(mcrop)
            continue
        ctrl = _crop_to_canvas(frames_rgb[k], window.crop_box, window.size, cv2.INTER_AREA)
        m_full = select_paint_mask(
            reader, k, window.gid, state=state,
            quality_thresholds=quality_thresholds,
            motion_guard_eps=motion_guard_eps,
        )
        if m_full is None:
            mcrop = np.zeros(window.size[::-1], np.uint8)
        else:
            mcrop = _crop_to_canvas(
                m_full.astype(np.uint8) * 255, window.crop_box, window.size,
                cv2.INTER_NEAREST,
            )
        mbool = mcrop > 127
        if control_mode == "pose_gen":
            # full-canvas pose + all-white mask -> VACE generates a posed
            # persona on a hallucinated bg (stitch mattes + composites it into
            # the real scene, clipped to the original silhouette). Per-frame grey
            # fallback when there's no usable body skeleton.
            kpf = _gid_keypoints_full(reader, k, window.gid)
            if m_full is not None and kpf is not None and _has_body_joints(kpf):
                ctrl = pose_gen_control_frame(kpf, window.crop_box, window.size[::-1])
                mcrop = np.full(window.size[::-1], 255, np.uint8)
            elif grey_control:
                ctrl = grey_control_frame(ctrl, mbool)
        elif control_mode == "pose_masked":
            # full-canvas pose skeleton (pose_gen's bake-free control) but the mask
            # stays the silhouette, so VACE regenerates ONLY inside it — the
            # persona is bounded exactly by the original mask (no matte, no bg
            # leak, no feather). Per-frame grey fallback when no body joints.
            kpf = _gid_keypoints_full(reader, k, window.gid)
            if m_full is not None and kpf is not None and _has_body_joints(kpf):
                ctrl = pose_gen_control_frame(kpf, window.crop_box, window.size[::-1])
            elif grey_control:
                ctrl = grey_control_frame(ctrl, mbool)
        elif control_mode == "pose" and m_full is not None:
            kpf = _gid_keypoints_full(reader, k, window.gid)
            if kpf is not None and _has_body_joints(kpf):
                ctrl = pose_control_frame(
                    ctrl, mbool, kpf, window.crop_box, window.size[::-1],
                    mask_dilate_px=pose_mask_dilate_px,
                )
            else:
                ctrl = grey_control_frame(ctrl, mbool)   # no usable skeleton → grey fallback
        elif grey_control:
            ctrl = grey_control_frame(ctrl, mbool)
        control.append(ctrl)
        masks.append(mcrop)
    return control, masks


def build_bundle(
    out_dir, window: VaceWindow, frames_rgb, reader, reference_rgb, *,
    grey_control: bool = True, fps: int = 25,
    preserve_overlap: dict | None = None,
    control_mode: str = "grey", pose_mask_dilate_px: int = 0,
) -> dict:
    """Write a VACE bundle (control.mp4, mask.mp4, reference.png, meta.json) and
    return the meta dict the client needs (size, length, fps)."""
    from pathlib import Path
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    W, H = window.size
    control, masks = build_frames(
        window, frames_rgb, reader, grey_control=grey_control,
        preserve_overlap=preserve_overlap,
        control_mode=control_mode, pose_mask_dilate_px=pose_mask_dilate_px,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_c = cv2.VideoWriter(str(out / "control.mp4"), fourcc, fps, (W, H))
    vw_m = cv2.VideoWriter(str(out / "mask.mp4"), fourcc, fps, (W, H))
    # Inputs are RGB (pipeline convention); cv2 writers assume BGR, so convert
    # at the boundary or ComfyUI decodes the persona channel-swapped.
    for ctrl, mcrop in zip(control, masks):
        vw_c.write(cv2.cvtColor(ctrl, cv2.COLOR_RGB2BGR))
        vw_m.write(cv2.cvtColor(mcrop, cv2.COLOR_GRAY2BGR))
    vw_c.release(); vw_m.release()

    ref = letterbox_reference(reference_rgb, (W, H))
    cv2.imwrite(str(out / "reference.png"), cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))
    meta = {
        "gid": window.gid, "n": len(window.frames),
        "frames": [window.frames[0], window.frames[-1]],
        "size": [W, H], "fps": fps, "crop_box": list(window.crop_box),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


_MIN_WINDOW_FRAMES = 5  # smallest valid 4n+1 worth rendering separately


def plan_windows(
    reader, gid: int, *, window_len: int = 49, overlap: int = 13,
    crop_pad: float = 0.10, crop_height: int = 832, max_bridge: int = 2,
) -> list[VaceWindow]:
    """Plan VACE windows for one gid: sliding stride = window_len - overlap,
    each window 4n+1, last window slid back so it ends at the last presence
    frame (any uncovered tail would otherwise fall through to fallback_blur).
    Presence is split on any gap of more than `max_bridge` missing frames so
    VACE doesn't see a temporal jump-cut as one continuous clip; tiny gaps
    are bridged (the per-frame carry-over handles them at mask-paint time).
    Each window has its own per-window stabilized crop."""
    present = [k for k in reader.frame_indices() if _gid_bbox(reader, k, gid) is not None]
    if not present:
        return []
    windows: list[VaceWindow] = []
    for seg in _split_on_gaps(present, max_bridge=max_bridge):
        windows.extend(_plan_segment_windows(
            reader, gid, seg, window_len=window_len, overlap=overlap,
            crop_pad=crop_pad, crop_height=crop_height,
        ))
    return windows


def _split_on_gaps(present: list[int], *, max_bridge: int) -> list[list[int]]:
    """Split a sorted presence list on any gap of more than `max_bridge`
    missing frames. A gap of 1 means presence[i+1] - presence[i] == 2 (one
    missing frame in between); max_bridge=2 thus keeps gaps of 1 or 2 frames
    inside the same segment and splits anything wider."""
    segments: list[list[int]] = []
    cur: list[int] = []
    for k in present:
        if cur and k - cur[-1] > 1 + max_bridge:
            segments.append(cur)
            cur = []
        cur.append(k)
    if cur:
        segments.append(cur)
    return segments


def _make_window(reader, gid, frames, crop_pad, crop_height) -> VaceWindow:
    boxes = [_gid_bbox(reader, k, gid) for k in frames]
    box = _union_bbox(boxes, crop_pad)
    bw, bh = box[2] - box[0], box[3] - box[1]
    H = round16(crop_height)
    W = round16(int(H * bw / bh))
    return VaceWindow(gid=gid, frames=frames, crop_box=box, size=(W, H))


def _plan_segment_windows(
    reader, gid: int, present: list[int], *, window_len: int, overlap: int,
    crop_pad: float, crop_height: int,
) -> list[VaceWindow]:
    """Plan windows over one contiguous (modulo `max_bridge`) presence segment."""
    if not present:
        return []
    stride = max(1, window_len - overlap)

    windows: list[VaceWindow] = []
    i = 0
    while i < len(present):
        n = _vace_length(min(window_len, len(present) - i))
        if n < _MIN_WINDOW_FRAMES:
            break
        frames = present[i:i + n]
        # Skip windows that don't extend past the prior one (the tail after
        # stride lands inside an already-rendered window).
        if windows and frames[-1] <= windows[-1].frames[-1]:
            break
        windows.append(_make_window(reader, gid, frames, crop_pad, crop_height))
        if i + n >= len(present):
            break
        i += stride

    # Trailing-frame coverage: slide the last window back so its last frame is
    # the segment's last presence frame. Length stays the same (4n+1); the
    # overlap with the prior window grows past `overlap`, which the linear
    # blend handles. Guarded against overlapping the window before the prior
    # one, preserving the at-most-2-owners invariant in stitch_windows.
    if windows and windows[-1].frames[-1] < present[-1]:
        n = len(windows[-1].frames)
        start_idx = len(present) - n
        if start_idx >= 0:
            new_frames = present[start_idx:]
            prev2_last = windows[-3].frames[-1] if len(windows) >= 3 else -1
            if new_frames[0] > prev2_last:
                windows[-1] = _make_window(reader, gid, new_frames, crop_pad, crop_height)
    return windows
