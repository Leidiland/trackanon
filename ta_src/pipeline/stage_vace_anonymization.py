"""Windowed Wan-VACE generation — the Pass-2 anonymization backend.

VACE generates a whole 4n+1 clip per gid from a fixed reference, so there is no
per-frame regen-gate / collapse-reuse machinery here — VACE has no per-frame
instability to compensate for. The stage renders multiple overlapping windows
for the target gid, stitches them with a linear-alpha blend across each overlap,
and writes a full-frame video covering all requested frame indices (frames
outside the gid's coverage pass through raw).
"""
from __future__ import annotations

import logging
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from ta_src.anonymization.confidence_gate import ConfidenceThresholds
from ta_src.anonymization.fallback_anonymizer import FallbackAnonymizer
from ta_src.anonymization.vace_bundle import (
    build_bundle, list_confirmed_gids, plan_windows, select_paint_mask,
)
from ta_src.anonymization.vace_client import (
    SharedPortPool, VaceClient, VaceClientPool, build_graph,
)
from ta_src.anonymization.vace_stitch import (
    clamp_crop_box, color_anchor_frame, compose_crop, compute_lab_stats,
    stitch_windows,
)
from ta_src.utils.frame_store import StitchStore

log = logging.getLogger(__name__)


def _silhouette_bottom(box, mask_sub) -> int:
    """Lowest masked row of a stored silhouette in full-frame coords — the depth
    cue for the composite (feet lower in frame = closer to camera). Falls back to
    the diff-box bottom when the stored mask carries only a feather band."""
    y0, _x0, y1, _x1 = box
    rows = np.any(np.asarray(mask_sub), axis=1)
    nz = np.nonzero(rows)[0]
    return int(y0) + int(nz[-1]) if nz.size else int(y1)


def _painted_chroma(rgb_sub, mask_sub) -> float:
    """Mean a*b* distance from neutral over the painted region — the oversatur-
    ation signal the reference-anchor chroma guard gates on."""
    m = np.asarray(mask_sub).astype(bool)
    if not m.any():
        return 0.0
    lab = cv2.cvtColor(rgb_sub, cv2.COLOR_RGB2LAB)[m].astype(np.float32).mean(axis=0)
    return float(((lab[1] - 128) ** 2 + (lab[2] - 128) ** 2) ** 0.5)


class VaceAnonymizationStage:
    def __init__(self, cfg, *, client=None, pool=None):
        self._cfg = cfg
        self._reference_dir = Path(cfg["reference_dir"])
        # Pool wins; bare client wraps to a singleton pool (backwards-compat
        # for existing tests + the single-server local deployment).
        if pool is not None:
            self._pool = pool
        elif client is not None:
            self._pool = VaceClientPool(clients=[client])
        else:
            self._pool = self._build_pool(cfg)
        # Fallback blur for any cached track whose silhouette VACE didn't paint
        # (unbound -1, unconfirmed, in-presence-but-out-of-window). Off by
        # default; flip on to close the raw-face leak.
        self._fallback: FallbackAnonymizer | None = (
            FallbackAnonymizer(
                kernel_min=int(cfg.get("fallback_blur_kernel_min", 51)),
                kernel_frac=float(cfg.get("fallback_blur_kernel_frac", 0.20)),
                dilate_min=int(cfg.get("fallback_blur_dilate_min", 5)),
                dilate_frac=float(cfg.get("fallback_blur_dilate_frac", 0.025)),
                feather_min=int(cfg.get("fallback_blur_feather_min", 5)),
                feather_frac=float(cfg.get("fallback_blur_feather_frac", 0.020)),
            )
            if bool(cfg.get("fallback_blur", False))
            else None
        )
        # Reference-seeded color anchor: cached per-gid L*a*b* stats of the
        # persona crop, used as the anchor target instead of window-0's (which
        # is contaminated by saturated cuffs/skin and can't desaturate W2+).
        self._anchor_stats_cache: dict[int, object] = {}

    def _resolve_target_gids(self, reader) -> list[int]:
        # Explicit list wins; empty / unset falls back to all confirmed gids in
        # the track cache so a default run anonymises every bound participant.
        # When the confidence-gate thresholds are wired in, low-confidence
        # bindings drop out and inherit the fallback-blur path automatically.
        explicit = list(self._cfg.get("target_gids", []) or [])
        if explicit:
            return [int(g) for g in explicit]
        return list_confirmed_gids(reader, thresholds=self._confidence_thresholds())

    def _confidence_thresholds(self) -> ConfidenceThresholds | None:
        cfg = self._cfg
        if "confidence_face_threshold" not in cfg:
            return None                        # gate off when thresholds aren't wired
        return ConfidenceThresholds(
            face_sim_floor=float(cfg["confidence_face_threshold"]),
            osnet_abs_floor=float(cfg["confidence_osnet_abs_threshold"]),
            osnet_margin_floor=float(cfg["confidence_osnet_margin_threshold"]),
        )

    def _mask_quality_thresholds(self) -> dict | None:
        cfg = self._cfg
        if "mask_score_floor" not in cfg:
            return None
        return {
            "ratio_floor": float(cfg.get("mask_floor_ratio", 0.2)),
            "score_floor": float(cfg["mask_score_floor"]),
            "score_pass_override": cfg.get("mask_score_pass_override"),
        }

    @staticmethod
    def _build_pool(cfg) -> VaceClientPool:
        pool_cfg = dict(cfg.get("pool", {}) or {})
        workers = int(pool_cfg.get("workers", 1))
        base_port = int(pool_cfg.get("base_port", cfg["comfyui_port"]))
        return VaceClientPool(
            host=str(cfg["comfyui_host"]), base_port=base_port, workers=workers,
            input_dir_root=cfg["comfyui_input_dir"],
            output_dir_root=cfg["comfyui_output_dir"],
            timeout_s=float(cfg.get("timeout", 2700.0)),
            poll_s=float(cfg.get("poll_interval", 4.0)),
        )

    def is_ready(self) -> bool:
        return self._pool.is_alive()

    def free_comfyui_vram(self) -> bool:
        return self._pool.free_vram()

    def _load_reference(self, gid: int) -> np.ndarray:
        path = self._reference_dir / f"{gid}.png"
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"VACE reference crop not found: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _reference_anchor_stats(self, gid: int):
        # L*a*b* (mean, std) of the reference persona, the fixed low-chroma anchor
        # target. A centre box drops the studio-crop background margin; the crops
        # are framed portraits so this isolates the person without a parser.
        if gid not in self._anchor_stats_cache:
            rgb = self._load_reference(gid)
            h, w = rgb.shape[:2]
            m = np.zeros((h, w), bool)
            m[int(0.30 * h):int(0.95 * h), int(0.20 * w):int(0.80 * w)] = True
            self._anchor_stats_cache[gid] = compute_lab_stats({0: rgb}, {0: m})
        return self._anchor_stats_cache[gid]

    def _prompt(self, gid: int) -> str:
        # Per-gid reference-crop prompt overrides the generic config prompt — it
        # describes the persona VACE must render, not the original subject.
        p = self._reference_dir / f"{gid}.prompt.txt"
        if p.exists() and p.read_text().strip():
            return p.read_text().strip()
        return str(self._cfg.get("prompt", ""))

    def _render_window(
        self, sub_work, window, frames_rgb, reader, *, preserve_overlap=None,
    ) -> list[np.ndarray]:
        """Render one window through the client and decode the result to a list
        of (H,W,3) RGB canvas frames. Test-override seam."""
        cfg = self._cfg
        gid = window.gid
        meta = build_bundle(
            sub_work, window, frames_rgb, reader, self._load_reference(gid),
            grey_control=bool(cfg.get("grey_control", True)), fps=int(cfg["fps"]),
            preserve_overlap=preserve_overlap,
            control_mode=str(cfg.get("control_mode", "grey")),
            pose_mask_dilate_px=int(cfg.get("pose_mask_dilate_px", 0)),
        )
        # Unique-per-render upload names: every window's graph references its own
        # control/mask/ref filenames so ComfyUI's filename-keyed input cache can
        # never serve a stale (empty) render for a graph that otherwise looks
        # identical (same crop dims + prompt + constant seed).
        tok = uuid.uuid4().hex[:12]
        names = (f"vc_{tok}.mp4", f"vm_{tok}.mp4", f"vr_{tok}.png")
        graph = build_graph(
            meta, prompt=self._prompt(gid), negative=str(cfg["negative"]),
            unet=cfg.get("unet"), clip=str(cfg["clip"]), vae=str(cfg["vae"]),
            gguf=cfg.get("gguf"), seed=int(cfg["seed"]), steps=int(cfg["steps"]),
            cfg=float(cfg["cfg"]), shift=float(cfg["shift"]),
            sampler=str(cfg["sampler"]), scheduler=str(cfg["scheduler"]),
            strength=float(cfg["strength"]),
            lora=cfg.get("lora"), lora_strength=float(cfg.get("lora_strength", 1.0)),
            filename_prefix=f"vace/gid{gid}",
            control_in=names[0], mask_in=names[1], ref_in=names[2],
            weight_dtype=str(cfg.get("weight_dtype", "default")),
        )
        # Retry transient worker failures (a ComfyUI "produced no output" or a
        # timeout shouldn't lose the window) — re-acquire so a flaky worker is
        # likely swapped for a healthy one.
        attempts = int(cfg.get("render_retries", 2)) + 1
        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                with self._pool.acquire() as client:
                    rendered = client.generate(sub_work, graph, remote_names=names)
                return _decode_canvases(rendered, n_expected=meta["n"])
            except (RuntimeError, TimeoutError, OSError) as e:
                last_err = e
                log.warning("VACE render gid %d failed (attempt %d/%d): %s",
                            gid, attempt + 1, attempts, e)
        raise last_err

    def _prepare_stores(self, reader, frames_provider, work_dir):
        """Render every target gid to its own disk-backed StitchStore and return
        (z-ordered gids that painted something, {gid: store}). None when no gid
        resolves or none paints — the cross-gid composite is streamed later from
        these stores so no full-clip frame buffer is ever held in RAM."""
        cfg = self._cfg
        gids = self._resolve_target_gids(reader)
        if not gids:
            log.warning("VACE: no target gids resolved (cache empty or none bound)")
            return None

        stores = self._render_all_gids(
            gids, reader, frames_provider, Path(work_dir),
            quality_thr=self._mask_quality_thresholds(),
            motion_eps=float(cfg.get("motion_guard_eps", 0.15)),
            preserve_on=bool(cfg.get("preserve_overlap", True)),
        )
        # Keep `gids` order — same z-order across pool sizes (deterministic
        # output regardless of which worker finished first).
        ordered = [g for g in gids if stores[g].painted_frames]
        if not ordered:
            return None
        for g in ordered:
            log.info("VACE: gid %d -> %d frame(s) stitched", g, len(stores[g].painted_frames))
        return ordered, stores

    def _compute_output_frames(
        self, reader, frames_provider, frame_indices, work_dir,
    ) -> list[np.ndarray] | None:
        """In-memory materialisation of the streamed composite — the test seam.
        Production goes through `run`, which streams the same frames straight to
        the encoder without ever holding the whole clip."""
        prep = self._prepare_stores(reader, frames_provider, work_dir)
        if prep is None:
            return None
        ordered, stores = prep
        return list(self._iter_frames(ordered, stores, reader, frames_provider, frame_indices))

    def _render_all_gids(
        self, gids, reader, frames_provider, work_dir, *,
        quality_thr, motion_eps, preserve_on,
    ) -> dict[int, StitchStore]:
        """Dispatch each gid's render to a pool worker. Concurrency defaults to
        the pool size; `render_concurrency` can oversubscribe it so threads
        dropping into their (GPU-idle) CPU stitch don't leave pool clients
        unused — the pool's semaphore still caps in-flight GPU renders, so the
        extra threads only keep the GPUs fed. Costs O(concurrency × window 4K
        frames) RAM, so only raise it past the pool size on a big-RAM host."""
        def _one(gid):
            try:
                return gid, self._render_gid(
                    reader, frames_provider, gid, work_dir,
                    quality_thr=quality_thr, motion_eps=motion_eps,
                    preserve_on=preserve_on,
                )
            except Exception as e:
                # A whole-gid failure (e.g. planning) must not abort the others;
                # return an empty store so this gid falls through to fallback blur.
                log.warning("VACE: gid %d failed entirely (%s: %s)",
                            gid, type(e).__name__, e)
                return gid, StitchStore(Path(work_dir) / "stitched" / f"gid{gid}")


        if bool(self._cfg.get("render_process_pool", False)) and len(gids) > 1:
            return self._render_all_gids_mp(gids, reader, frames_provider, work_dir)
        concurrency = int(self._cfg.get("render_concurrency", 0) or self._pool.size)
        concurrency = max(1, min(concurrency, len(gids)))
        if concurrency <= 1 or len(gids) <= 1:
            return dict(_one(g) for g in gids)
        results: dict[int, StitchStore] = {}
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(_one, g): g for g in gids}
            for fut in as_completed(futures):
                gid, store = fut.result()
                results[gid] = store
        return results

    def _render_all_gids_mp(self, gids, reader, frames_provider, work_dir) -> dict[int, StitchStore]:
        """Process-pool variant: one OS process per gid sidesteps the GIL that
        serialises the (CPU-bound) crop-space stitch to ~2 cores in the thread
        path, so the stitch parallelises across cores and stops starving the GPU
        pool. Each child rebuilds a 1-client stage bound to one ComfyUI worker
        port (static gid->port round-robin) and re-opens the disk-backed
        reader/provider, so nothing un-picklable crosses the process boundary;
        the child returns only its frame-index sets and the main process
        reconstructs lightweight StitchStore handles for the composite (the npz
        already live on disk). Requires disk-backed reader/provider."""
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import Manager

        from omegaconf import OmegaConf

        cache_dir = str(reader._dir)
        frames_root = str(frames_provider.root)
        cfg_plain = OmegaConf.to_container(self._cfg, resolve=True)
        host = str(self._cfg.get("comfyui_host", "127.0.0.1"))
        base = int(self._cfg.get("comfyui_port", 8190))
        nports = max(1, self._pool.size)
        concurrency = int(self._cfg.get("render_concurrency", 0) or len(gids))
        concurrency = max(1, min(concurrency, len(gids)))
        results: dict[int, StitchStore] = {}
        # Work-stealing: a shared port queue (one entry per GPU worker) that every
        # gid process pulls from per render, so GPUs stay fed regardless of how
        # uneven the gids' window counts are — no static gid->port idle tail.
        with Manager() as mgr:
            port_q = mgr.Queue()
            for i in range(nports):
                port_q.put(base + i)
            payloads = [
                {"gid": int(g), "host": host, "n_ports": nports, "port_queue": port_q,
                 "cfg": cfg_plain, "cache_dir": cache_dir, "frames_root": frames_root,
                 "work_dir": str(work_dir)}
                for g in gids
            ]
            with ProcessPoolExecutor(max_workers=concurrency) as ex:
                for gid, frames, painted in ex.map(_render_gid_in_process, payloads):
                    st = StitchStore(Path(work_dir) / "stitched" / f"gid{gid}")
                    st.frames = set(frames)
                    st.painted_frames = set(painted)
                    results[gid] = st
                    log.info("VACE: gid %d done (%d painted frames)", gid, len(painted))
        return results

    def _render_gid(
        self, reader, frames_provider, gid, work_dir, *,
        quality_thr, motion_eps, preserve_on,
    ) -> StitchStore:
        """Render every window for one gid and stitch them progressively into a
        disk-backed StitchStore (against raw source — independent of other gids,
        so this whole call can run on its own worker and only one window of
        frames is ever resident). When `preserve_on`, window N's bundle takes
        the prior window's stitched output for the overlap zone and emits mask=0
        there, so VACE keeps those pixels and conditions the rest on them."""
        cfg = self._cfg
        store = StitchStore(Path(work_dir) / "stitched" / f"gid{gid}")
        windows = plan_windows(
            reader, gid, window_len=int(cfg["window_len"]),
            overlap=int(cfg.get("overlap", 13)),
            crop_pad=float(cfg["crop_pad"]), crop_height=int(cfg["crop_height"]),
            max_bridge=int(cfg.get("max_bridge", 2)),
        )
        if not windows:
            log.info("VACE: gid %d has no plannable window — skipping", gid)
            return store

        # Resume: a prior run's stitched frames + finished-window markers + the
        # window-0 colour anchor live on disk, so a re-render after a crash
        # skips done windows instead of redoing the whole gid (load-bearing for
        # the long native run, where one gid can be hours).
        store.load_existing()
        anchor_on = bool(cfg.get("color_anchor", True))
        # "window0" (default): anchor later windows to W0 (drift correction).
        # "reference": anchor every window to the persona crop's palette — fixes
        # the cross-window oversaturation W0-seeding can't (W2+ renders magenta).
        seed_mode = str(cfg.get("color_anchor_seed", "window0"))
        chroma_guard = float(cfg.get("color_anchor_chroma_guard", 0.0))
        target_stats = store.load_anchor() if anchor_on else None
        if anchor_on and seed_mode == "reference" and target_stats is None:
            target_stats = self._reference_anchor_stats(gid)
            if target_stats is not None:
                store.save_anchor(*target_stats)
        for wi, w in enumerate(windows):
            if store.window_done(wi):
                continue
            # Whole window (source prep + render + stitch + store) under one
            # guard: a single bad window — missing frame, render failure, short
            # decode, stitch error — must skip and let the run continue
            # (preserve-overlap + fallback blur cover the gap), never abort
            # the other 170+ windows.
            try:
                per_w_src = {k: frames_provider(k) for k in w.frames}
                preserve = None
                if preserve_on and wi > 0:
                    preserve = {
                        k: store.stitched_full(k, per_w_src[k])
                        for k in w.frames if k in store.frames
                    }
                canvases = self._render_window(
                    work_dir / f"gid{gid}_w{wi:02d}", w, per_w_src, reader,
                    preserve_overlap=preserve,
                )
                # Crop-space stitch: composite only inside each frame's crop rect
                # (~1.3MP) not whole 4K, so the (GIL-bound) stitch stays cheap
                # enough to feed the GPU pool. State carries the last-good mask
                # within this window, matching the old per-window stitch.
                cm = str(cfg.get("control_mode", "grey"))
                is_pose_gen = cm == "pose_gen"   # only pose_gen needs the matte
                # Pose modes want the persona to fill the silhouette with a crisp
                # edge, so they drop the grey path's boundary feather (default 0).
                crisp = cm in ("pose_gen", "pose_masked")
                feather = int(cfg.get("posegen_feather_px", 0)) if crisp else int(cfg.get("feather_px", 0))
                posegen_sil_dilate = int(cfg.get("posegen_silhouette_dilate_px", 0))
                pos = {f: j for j, f in enumerate(w.frames)}
                state: dict = {}
                win: dict = {}      # f -> (cb, rgb_sub, mask_sub)
                for f in w.frames:
                    if pos[f] >= len(canvases):
                        continue                # short decode: drop missing tail frames
                    mask_full = select_paint_mask(
                        reader, f, gid, state=state,
                        quality_thresholds=quality_thr, motion_guard_eps=motion_eps,
                    )
                    if mask_full is None:
                        continue                # frame passes through raw source
                    canvas_f = canvases[pos[f]]
                    if is_pose_gen:
                        # pose_gen: matte the persona out of the hallucinated bg, then
                        # extend the persona's own pixels over the rest of the
                        # canvas — so pasting the full silhouette covers the whole
                        # mask with ONLY persona-derived colour (no generated bg,
                        # no blurred residual). Optionally dilate the silhouette so
                        # the persona can extend slightly past a tight SAM3 mask.
                        from ta_src.anonymization.vace_matte import (
                            fill_nonpersona, persona_matte_canvas,
                        )
                        matte = persona_matte_canvas(canvas_f, dilate_px=0)
                        import os as _os
                        if _os.environ.get("VACE_DEBUG_DUMP") and f == w.frames[0]:
                            dd = Path(_os.environ["VACE_DEBUG_DUMP"]); dd.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(dd / f"gid{gid}_canvas_raw.png"), cv2.cvtColor(canvas_f, cv2.COLOR_RGB2BGR))
                            cv2.imwrite(str(dd / f"gid{gid}_matte.png"), matte.astype(np.uint8) * 255)
                            _filled = fill_nonpersona(canvas_f, matte)
                            cv2.imwrite(str(dd / f"gid{gid}_canvas_filled.png"), cv2.cvtColor(_filled, cv2.COLOR_RGB2BGR))
                        canvas_f = fill_nonpersona(canvas_f, matte)
                        mask_full = np.asarray(mask_full).astype(bool)
                        if posegen_sil_dilate > 0:
                            k = 2 * posegen_sil_dilate + 1
                            mask_full = cv2.dilate(
                                mask_full.astype(np.uint8), np.ones((k, k), np.uint8),
                            ).astype(bool)
                    cb = clamp_crop_box(w.crop_box, per_w_src[f].shape[:2])
                    base = store.stitched_crop(f, cb, per_w_src[f])
                    rgb_sub, mask_sub = compose_crop(
                        base, cb, canvas_f, w.crop_box, mask_full,
                        feather_px=feather,
                    )
                    win[f] = (cb, rgb_sub, mask_sub)
                # Pull each window's painted region toward the anchor target so
                # VAE-roundtrip drift can't accumulate. Reference seeding uses a
                # fixed low-chroma target (every window, W0 included); window0
                # seeding caches W0's stats and corrects later windows to them.
                if anchor_on and seed_mode == "reference":
                    if target_stats is not None:
                        t_mean, t_std = target_stats
                        # Guard is a margin ABOVE the reference's own chroma, so it
                        # adapts per persona: a legitimately colourful persona isn't
                        # pulled, only a window that overshoots its reference.
                        ref_chroma = float(((t_mean[1] - 128) ** 2 + (t_mean[2] - 128) ** 2) ** 0.5)
                        for f, (cb, rgb_sub, mask_sub) in win.items():
                            if not mask_sub.any():
                                continue
                            if chroma_guard > 0 and _painted_chroma(rgb_sub, mask_sub) < ref_chroma + chroma_guard:
                                continue        # window near its reference — leave it
                            win[f] = (cb, color_anchor_frame(rgb_sub, mask_sub, t_mean, t_std, chroma_only=True), mask_sub)
                elif anchor_on:
                    stitched = {f: v[1] for f, v in win.items()}
                    painted = {f: v[2] for f, v in win.items()}
                    if wi == 0:
                        target_stats = compute_lab_stats(stitched, painted)
                        if target_stats is not None:
                            store.save_anchor(*target_stats)   # persist for resume
                    elif target_stats is not None:
                        t_mean, t_std = target_stats
                        for f, (cb, rgb_sub, mask_sub) in win.items():
                            if mask_sub.any():
                                win[f] = (cb, color_anchor_frame(rgb_sub, mask_sub, t_mean, t_std), mask_sub)
                for f, (cb, rgb_sub, mask_sub) in win.items():
                    store.put_crop(f, cb, rgb_sub, mask_sub)
                store.mark_window_done(wi)      # only after a full clean window
            except Exception as e:
                log.warning("VACE: gid %d window %d skipped (%s: %s)",
                            gid, wi, type(e).__name__, e)
                continue
        return store

    def _composite_frame(self, k, ordered_gids, stores, reader, frames_provider):
        """Composite one output frame: layer gids' silhouettes in z-order over the
        source, then apply the fallback blur. The first gid to paint seeds the
        whole stitched region (feather band included); later gids overwrite only
        their hard silhouette. Independent per frame (stores are on disk), so the
        parallel composite path fans these out across CPU cores."""
        fb = self._fallback
        base = None
        seeded = False
        painted_full = None
        # Depth order: paint far (silhouette ending higher in frame) first, near
        # (feet lower) last, so a foreground persona is never overwritten by a
        # background neighbour whose mask overlaps it. gid number is a tie-break
        # only — a fixed gid order makes the lowest gid the permanent bottom layer
        # and it vanishes whenever a neighbour's mask encroaches.
        layers = []
        for gid in ordered_gids:
            st = stores[gid]
            if k not in st.painted_frames:
                continue
            rec = st.get(k)
            layers.append((_silhouette_bottom(rec[0], rec[2]), gid, rec))
        layers.sort(key=lambda L: (L[0], L[1]))
        for _bottom, gid, (box, rgb_sub, mask_sub) in layers:
            y0, x0, y1, x1 = box
            if base is None:
                base = frames_provider(k).copy()
            if not seeded:
                base[y0:y1, x0:x1] = rgb_sub
                seeded = True
            else:
                region = base[y0:y1, x0:x1]
                region[mask_sub] = rgb_sub[mask_sub]
            if fb is not None:
                if painted_full is None:
                    painted_full = np.zeros(base.shape[:2], bool)
                painted_full[y0:y1, x0:x1] |= mask_sub
        if fb is not None:
            base = self._apply_fallback_frame(reader, frames_provider, k, base, painted_full)
        return base if base is not None else frames_provider(k)

    def _iter_frames(self, ordered_gids, stores, reader, frames_provider, frame_indices):
        """Yield each output frame in order — one frame resident at a time."""
        for k in frame_indices:
            yield self._composite_frame(k, ordered_gids, stores, reader, frames_provider)

    def _apply_fallback_frame(self, reader, frames_provider, k, base, painted_full):
        # Blur each cached track's silhouette pixels VACE didn't paint
        # (cached_mask AND NOT painted_full). Closes the carry-over leak: a
        # smaller/displaced painted silhouette than the cached one gets the
        # remainder blurred instead of passing through raw.
        try:
            tracks, _ = reader.read(k)
        except FileNotFoundError:
            return base                        # frame not in cache (out of clip range)
        for t in tracks:
            mask = t.get("mask")
            if mask is None:
                continue
            cached = np.asarray(mask).astype(bool)
            if painted_full is not None:
                cached = cached & ~painted_full
            if not cached.any():
                continue
            if base is None:
                base = frames_provider(k).copy()
            self._fallback.apply(base, cached)
        return base

    def run(self, reader, frames_provider, frame_indices, *, work_dir, out_path):
        prep = self._prepare_stores(reader, frames_provider, work_dir)
        if prep is None:
            return None
        ordered, stores = prep
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame_indices = list(frame_indices)
        workers = int(self._cfg.get("composite_workers", 0) or 0)
        # Parallel composite: the per-frame layering is CPU-bound and independent
        # (stores on disk), but the GPUs are idle during it — so fan it across the
        # box's cores instead of the single-threaded stream. Needs ffmpeg for the
        # lossless chunk concat; falls back to the serial stream otherwise.
        if (workers > 1 and len(frame_indices) >= 2 * workers
                and shutil.which("ffmpeg")):
            n = self._stream_parallel(
                ordered, stores, reader, frames_provider, frame_indices,
                Path(work_dir), out, workers,
            )
        else:
            n = _stream_write(
                self._iter_frames(ordered, stores, reader, frames_provider, frame_indices),
                out, fps=int(self._cfg["fps"]),
            )
        log.info("VACE: wrote stitched video (%d frames) -> %s", n, out)
        return out

    def _stream_parallel(self, ordered, stores, reader, frames_provider,
                         frame_indices, work_dir, out_path, workers) -> int:
        """Composite frames across `workers` processes — each writes a contiguous
        chunk clip from the on-disk stores — then concat the chunks losslessly
        (ffmpeg copy). Requires disk-backed reader/provider. Turns the serial
        composite tail into ~workers-x faster, using the cores the GPUs leave idle."""
        from concurrent.futures import ProcessPoolExecutor

        from omegaconf import OmegaConf

        k = min(workers, len(frame_indices))
        size = (len(frame_indices) + k - 1) // k
        ranges = [frame_indices[i * size:(i + 1) * size] for i in range(k)]
        ranges = [r for r in ranges if r]
        chunk_dir = Path(work_dir) / "composite_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        cfg_plain = OmegaConf.to_container(OmegaConf.create(self._cfg), resolve=True)
        payloads = [
            {"cfg": cfg_plain, "cache_dir": str(reader._dir),
             "frames_root": str(frames_provider.root), "work_dir": str(work_dir),
             "ordered": list(ordered), "frames": rng,
             "out": str(chunk_dir / f"chunk_{i:03d}.mp4"), "fps": int(self._cfg["fps"])}
            for i, rng in enumerate(ranges)
        ]
        clips, total = [], 0
        with ProcessPoolExecutor(max_workers=k) as ex:
            for path, n in ex.map(_composite_chunk_in_process, payloads):
                clips.append(path)
                total += n
        _concat_clips(clips, out_path)
        return total


def _render_gid_in_process(payload: dict):
    """ProcessPoolExecutor entry point (module-level so it pickles). Rebuilds a
    1-client stage on the assigned ComfyUI port + re-opens the disk-backed
    reader/provider in this process, renders+stitches one gid to its on-disk
    StitchStore, and returns (gid, sorted(frames), sorted(painted_frames)) — the
    npz stay on disk for the parent's composite, so only small sets cross back."""
    from omegaconf import OmegaConf

    from ta_src.pipeline.track_cache import TrackCacheReader
    from ta_src.utils.frame_store import DiskFrameProvider

    cfg = OmegaConf.create(payload["cfg"])
    cfg.comfyui_host = payload["host"]
    cfg.render_process_pool = False          # no nested pooling inside the child
    # Work-stealing pool over the shared port queue: each render pulls any free
    # GPU worker, so this gid never waits on a statically-assigned busy port.
    pool = SharedPortPool(
        payload["port_queue"], host=payload["host"], n_ports=int(payload["n_ports"]),
        input_dir_root=cfg.get("comfyui_input_dir"),
        output_dir_root=cfg.get("comfyui_output_dir"),
        timeout_s=float(cfg.get("timeout", 2700.0)),
        poll_s=float(cfg.get("poll_interval", 4.0)),
    )
    stage = VaceAnonymizationStage(cfg, pool=pool)
    reader = TrackCacheReader(payload["cache_dir"])
    frames = DiskFrameProvider(payload["frames_root"])
    frames.existing_count()                  # restore _n so the tail-frame clamp applies
    gid = int(payload["gid"])
    try:
        store = stage._render_gid(
            reader, frames, gid, Path(payload["work_dir"]),
            quality_thr=stage._mask_quality_thresholds(),
            motion_eps=float(cfg.get("motion_guard_eps", 0.15)),
            preserve_on=bool(cfg.get("preserve_overlap", True)),
        )
    except Exception as e:
        # A whole-gid failure must not propagate through ex.map and abort the
        # other gids' renders; the gid falls back to blur for its frames.
        log.warning("VACE: gid %d render aborted in worker (%s: %s)",
                    gid, type(e).__name__, e)
        return gid, [], []
    return gid, sorted(store.frames), sorted(store.painted_frames)


def _composite_chunk_in_process(payload: dict):
    """ProcessPoolExecutor entry point for the parallel composite (module-level so
    it pickles). Rebuilds a pool-less stage + re-opens the disk-backed reader/
    provider/stores in this process, composites its contiguous frame range to a
    chunk clip, and returns (chunk_path, n_frames). No GPU / ComfyUI touched."""
    from omegaconf import OmegaConf

    from ta_src.pipeline.track_cache import TrackCacheReader
    from ta_src.utils.frame_store import DiskFrameProvider, StitchStore

    class _NoClient:
        def generate(self, *a, **k):
            raise RuntimeError("composite worker must not render")

    cfg = OmegaConf.create(payload["cfg"])
    cfg.render_process_pool = False
    stage = VaceAnonymizationStage(cfg, client=_NoClient())
    reader = TrackCacheReader(payload["cache_dir"])
    frames = DiskFrameProvider(payload["frames_root"])
    frames.existing_count()                  # restore _n for the tail-frame clamp
    work_dir = Path(payload["work_dir"])
    stores = {}
    for g in payload["ordered"]:
        st = StitchStore(work_dir / "stitched" / f"gid{g}")
        st.load_existing()
        stores[g] = st
    n = _stream_write(
        stage._iter_frames(payload["ordered"], stores, reader, frames, payload["frames"]),
        payload["out"], fps=int(payload["fps"]),
    )
    return payload["out"], n


def _concat_clips(clip_paths, out_path) -> None:
    """Losslessly concat same-codec chunk clips in order (ffmpeg concat demuxer,
    stream copy — no re-encode)."""
    import subprocess

    out = Path(out_path)
    listf = out.parent / "_concat_list.txt"
    listf.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in clip_paths))
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listf),
         "-c", "copy", str(out)],
        check=True, capture_output=True,
    )
    listf.unlink(missing_ok=True)


def _decode_canvases(path, n_expected) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) != n_expected:
        log.warning("VACE: decoded %d frames from %s, expected %d",
                    len(frames), path, n_expected)
    return frames


def _stream_write(frames_rgb, out_path, *, fps: int) -> int:
    """Encode an iterable of RGB frames straight to disk, opening the writer
    from the first frame's shape so we never need the whole clip up front."""
    vw = None
    n = 0
    try:
        for rgb in frames_rgb:
            if vw is None:
                h, w = rgb.shape[:2]
                vw = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h),
                )
            vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            n += 1
    finally:
        if vw is not None:
            vw.release()
    if n == 0:
        raise ValueError("VACE: no frames to write")
    return n
