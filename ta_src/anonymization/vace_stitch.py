"""Stitch per-window VACE renders into a full-frame video.

Each window is rendered into its own (W,H) crop canvas; this module pastes those
back at the window's crop_box rect (silhouette-masked, from the track cache),
and blends adjacent windows linearly across their overlap so the boundary frames
don't pop. Other gids / non-presence frames pass through raw source.
"""
from __future__ import annotations

import cv2
import numpy as np

from ta_src.anonymization.vace_bundle import select_paint_mask


def compose_frame(source_rgb, rendered_canvas, crop_box, mask_full, *, feather_px: int = 0):
    """Paste `rendered_canvas` into `source_rgb` at `crop_box` (resize to fit),
    keeping only the silhouette region from `mask_full`. The surround stays
    source pixels. feather_px>0 swaps the hard mask paste for a Gaussian-
    feathered alpha-blend, so VACE's small overshoot past the silhouette doesn't
    show as a step at the boundary."""
    ix0, iy0 = int(np.floor(crop_box[0])), int(np.floor(crop_box[1]))
    ix1, iy1 = int(np.ceil(crop_box[2])), int(np.ceil(crop_box[3]))
    rect_w, rect_h = ix1 - ix0, iy1 - iy0
    resized = cv2.resize(rendered_canvas, (rect_w, rect_h), interpolation=cv2.INTER_AREA)

    fh, fw = source_rgb.shape[:2]
    cx0, cy0 = max(0, ix0), max(0, iy0)
    cx1, cy1 = min(fw, ix1), min(fh, iy1)
    placed = np.zeros_like(source_rgb)
    placed[cy0:cy1, cx0:cx1] = resized[cy0 - iy0:cy0 - iy0 + (cy1 - cy0),
                                       cx0 - ix0:cx0 - ix0 + (cx1 - cx0)]

    sel = np.asarray(mask_full).astype(bool)
    if feather_px <= 0:
        out = source_rgb.copy()
        out[sel] = placed[sel]
        return out

    # Feathered alpha: blur the silhouette, then blend src/placed per-pixel.
    k = 2 * int(feather_px) + 1
    alpha = cv2.GaussianBlur(sel.astype(np.float32), (k, k), float(feather_px))[..., None]
    out = (1.0 - alpha) * source_rgb.astype(np.float32) + alpha * placed.astype(np.float32)
    return out.clip(0, 255).astype(np.uint8)


def clamp_crop_box(crop_box, frame_hw) -> tuple[int, int, int, int]:
    """crop_box (float, frame coords) -> integer (y0,x0,y1,x1) clamped to the
    frame. The rect the persona occupies; everything outside is raw source."""
    fh, fw = frame_hw
    ix0, iy0 = int(np.floor(crop_box[0])), int(np.floor(crop_box[1]))
    ix1, iy1 = int(np.ceil(crop_box[2])), int(np.ceil(crop_box[3]))
    return max(0, iy0), max(0, ix0), min(fh, iy1), min(fw, ix1)


def compose_crop(base_crop, cb, rendered_canvas, crop_box, mask_full, *, feather_px: int = 0):
    """Crop-space twin of `compose_frame`: composite the rendered canvas onto
    `base_crop` (the cb=(y0,x0,y1,x1) region of the stitch base — prior-window
    stitch or raw source) instead of allocating/blurring whole 4K frames. The
    silhouette lives inside crop_box (it carries crop_pad), so only this rect
    changes. Returns (rgb_sub, mask_sub) for StitchStore.put_crop."""
    y0, x0, y1, x1 = cb
    ix0, iy0 = int(np.floor(crop_box[0])), int(np.floor(crop_box[1]))
    ix1, iy1 = int(np.ceil(crop_box[2])), int(np.ceil(crop_box[3]))
    resized = cv2.resize(rendered_canvas, (ix1 - ix0, iy1 - iy0), interpolation=cv2.INTER_AREA)
    placed = resized[y0 - iy0:y1 - iy0, x0 - ix0:x1 - ix0]
    mask_sub = np.asarray(mask_full[y0:y1, x0:x1]).astype(bool)

    if feather_px <= 0:
        out = base_crop.copy()
        out[mask_sub] = placed[mask_sub]
    else:
        k = 2 * int(feather_px) + 1
        alpha = cv2.GaussianBlur(mask_sub.astype(np.float32), (k, k), float(feather_px))[..., None]
        out = ((1.0 - alpha) * base_crop.astype(np.float32)
               + alpha * placed.astype(np.float32)).clip(0, 255).astype(np.uint8)
    return out, mask_sub


def compute_lab_stats(stitched: dict, painted: dict):
    """Concat the painted-region L*a*b pixels across every stitched frame and
    return (mean, std) per channel. Used as the W0 color-anchor target — every
    subsequent window's painted region gets affine-pulled back to these stats,
    so VAE round-trip chroma loss can't accumulate across window boundaries."""
    pixels = []
    for f, rgb in stitched.items():
        m = painted.get(f)
        if m is None or not m.any():
            continue
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        pixels.append(lab[np.asarray(m).astype(bool)])
    if not pixels:
        return None
    cat = np.concatenate(pixels, axis=0).astype(np.float32)
    return cat.mean(axis=0), cat.std(axis=0)


def color_anchor_frame(frame_rgb, mask_bool, target_mean_lab, target_std_lab, *,
                       chroma_only: bool = False):
    """Affine-correct the masked region's L*a*b channels to (target_mean,
    target_std); identity outside the mask. Channel-wise: x' = (x - mu_src) *
    (sigma_tgt / sigma_src) + mu_tgt. A 1e-3 floor on sigma_src keeps the
    transform stable for nearly-uniform regions.

    chroma_only: touch a*/b* only (leave L exactly as rendered) and clamp the
    scale to <=1, so the anchor can only REDUCE chroma. Desaturates an
    oversaturated window toward the target hue without shifting lightness — the
    fix for the W2+ magenta that the full L*a*b match washed out into a glow."""
    mask_bool = np.asarray(mask_bool).astype(bool)
    if not mask_bool.any():
        return frame_rgb
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    px = lab[mask_bool]
    src_mean, src_std = px.mean(axis=0), px.std(axis=0)
    scale = target_std_lab / np.maximum(src_std, 1e-3)
    if chroma_only:
        scale = np.minimum(scale, 1.0)
        new = px.copy()
        for c in (1, 2):
            new[:, c] = (px[:, c] - src_mean[c]) * scale[c] + target_mean_lab[c]
        lab[mask_bool] = new
    else:
        lab[mask_bool] = (px - src_mean) * scale + target_mean_lab
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    # cv2 RGB↔LAB is non-bit-exact (~1 unit round-trip noise on every pixel);
    # restore exact source bytes outside the mask so we only ever touch the
    # painted region.
    out[~mask_bool] = frame_rgb[~mask_bool]
    return out


def stitch_windows(
    source_frames, windows, rendered_canvases, reader, *, feather_px: int = 0,
    quality_thresholds: dict | None = None, motion_guard_eps: float = 0.15,
) -> tuple[dict, dict]:
    """Composite each window's canvases back into the source, linear-alpha
    blending across overlap. The compose mask passes through the same gate +
    carry-over logic as build_frames, so the painted silhouette stays aligned
    with what VACE actually rendered. Returns:
      - stitched: {cache_idx: full-frame rgb}
      - painted_regions: {cache_idx: bool mask of pixels VACE painted} — used
        by the fallback path to blur the cached_silhouette \\ painted remainder.
    """
    if not windows:
        return {}, {}
    gid = windows[0].gid
    pos: list[dict[int, int]] = [{f: i for i, f in enumerate(w.frames)} for w in windows]
    all_frames = sorted({f for w in windows for f in w.frames})

    out: dict = {}
    painted: dict = {}
    state: dict = {}    # carries last good mask/bbox across frames for this gid
    for f in all_frames:
        owners = [i for i in range(len(windows)) if f in pos[i]]
        mask_full = select_paint_mask(
            reader, f, gid, state=state,
            quality_thresholds=quality_thresholds,
            motion_guard_eps=motion_guard_eps,
        )
        if mask_full is None or not owners:
            out[f] = source_frames[f]
            painted[f] = np.zeros(source_frames[f].shape[:2], dtype=bool)
            continue
        painted[f] = np.asarray(mask_full).astype(bool)
        if len(owners) == 1:
            i = owners[0]
            out[f] = compose_frame(
                source_frames[f], rendered_canvases[i][pos[i][f]],
                windows[i].crop_box, mask_full, feather_px=feather_px,
            )
            continue
        if len(owners) > 2:
            raise NotImplementedError(
                f"frame {f}: {len(owners)} owning windows; stride must exceed overlap"
            )
        iA, iB = owners
        cA = compose_frame(source_frames[f], rendered_canvases[iA][pos[iA][f]],
                           windows[iA].crop_box, mask_full, feather_px=feather_px)
        cB = compose_frame(source_frames[f], rendered_canvases[iB][pos[iB][f]],
                           windows[iB].crop_box, mask_full, feather_px=feather_px)
        first_b = windows[iB].frames[0]
        last_a = windows[iA].frames[-1]
        alpha = (f - first_b) / max(1, last_a - first_b)
        blended = (1 - alpha) * cA.astype(np.float32) + alpha * cB.astype(np.float32)
        out[f] = blended.clip(0, 255).astype(np.uint8)
    return out, painted
