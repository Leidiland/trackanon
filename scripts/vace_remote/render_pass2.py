#!/usr/bin/env python3
"""Run VACE Pass 2 directly on the box that hosts the ComfyUI workers.

The full pipeline keeps Pass 2 local and tunnels only bundles, so local RAM
bounds how many gids stitch at once. When the footage is non-sensitive and the
GPU box is trusted, that constraint disappears: ship the (already-populated)
track cache + the source clip to the box and stitch there, against the box's
large RAM and all its pinned GPU workers. Pass 1 is skipped (cache reused), so
this never imports SAM 3 / InsightFace / OSNet / torch — only the VACE stage.

Usage (on the box):
  python render_pass2.py --config configs/anonymization/vace_remote.yaml \
      --cache-dir outputs/eval/cache/meeting_01 \
      --source data/input_videos/<clip>.mp4 \
      --start 540 --end 600 --fps 25 --workers 4 \
      --out outputs/vace_remote_meeting60/<clip>_vace.mp4

--dry-run resolves gids + plans windows and exits (no ComfyUI needed) — used to
validate config/cache locally before paying for the box.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

from ta_src.anonymization.vace_bundle import list_confirmed_gids, plan_windows
from ta_src.pipeline.stage_vace_anonymization import VaceAnonymizationStage
from ta_src.pipeline.track_cache import TrackCacheReader
from ta_src.utils.frame_store import DiskFrameProvider
from ta_src.video.videohandler import read_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("render_pass2")


def _load_cfg(config_path: str, workers: int, host: str, port: int):
    # vace_remote.yaml is self-contained (no hydra defaults to compose); the
    # stage wants the nested `vace:` block — the same slice MainPipeline passes.
    root = OmegaConf.load(config_path)
    cfg = root.vace if "vace" in root else root
    cfg.pool = cfg.get("pool", {}) or {}
    cfg.pool.workers = workers
    cfg.pool.base_port = port
    cfg.comfyui_host = host
    cfg.comfyui_port = port
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=0,
                    help="gid render threads; 0=pool size. Oversubscribe past "
                         "workers on a big-RAM box to keep GPUs fed during stitch.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8190)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temp", default="data/temp")
    ap.add_argument("--steps", type=int, default=0, help="override sampler steps (0=config)")
    ap.add_argument("--cfg", type=float, default=0.0, help="override guidance cfg (0=config)")
    ap.add_argument("--no-lora", action="store_true",
                    help="disable the distill LoRA (native many-step VACE)")
    ap.add_argument("--window-len", type=int, default=0,
                    help="override 4n+1 window length (0=config). Raise to 81 on "
                         ">=48GB GPUs for fewer boundaries / less overlap waste")
    ap.add_argument("--crop-height", type=int, default=0,
                    help="override canvas height (0=config)")
    ap.add_argument("--gguf", default="",
                    help="14B GGUF filename (UnetLoaderGGUF); overrides config unet")
    ap.add_argument("--lora", default="",
                    help="enable a distill LoRA by filename (overrides config null)")
    ap.add_argument("--lora-strength", type=float, default=-1.0,
                    help="distill LoRA strength (-1=config)")
    ap.add_argument("--weight-dtype", default="",
                    help="UNET load precision override, e.g. fp8_e4m3fn for "
                         "~1.4-1.7x on fp8 tensor cores (empty=config)")
    ap.add_argument("--control-mode", default="",
                    choices=["", "grey", "pose", "pose_gen", "pose_masked"],
                    help="VACE control recipe: grey-the-hole (default), pose "
                         "skeleton-in-the-hole, pose_gen (B2: full-canvas pose + "
                         "white mask, matte persona into the silhouette), or "
                         "pose_masked (C: full-canvas pose + silhouette mask, "
                         "persona bounded exactly by the mask, no matte) (empty=config)")
    ap.add_argument("--b2-silhouette-dilate-px", type=int, default=-1,
                    help="pose_gen: dilate the paste silhouette so the persona "
                         "can extend slightly past a tight SAM3 mask; -1=config")
    ap.add_argument("--b2-feather-px", type=int, default=-1,
                    help="pose_gen: persona/scene boundary feather; 0=crisp edge, -1=config")
    ap.add_argument("--process-pool", action="store_true",
                    help="stitch gids in separate processes (no GIL) to feed the GPUs")
    ap.add_argument("--composite-workers", type=int, default=0,
                    help="parallelize the final composite/stream across N CPU procs "
                         "(needs ffmpeg; 0=serial). Set ~= vCPU count to kill the "
                         "single-threaded stream tail after rendering")
    ap.add_argument("--target-gids", default="",
                    help="comma-separated gid subset (default: all confirmed)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve gids + plan windows, then exit (no ComfyUI)")
    args = ap.parse_args(argv)

    cfg = _load_cfg(args.config, args.workers, args.host, args.port)
    if args.concurrency:
        cfg.render_concurrency = args.concurrency
    if args.steps:
        cfg.steps = args.steps
    if args.cfg:
        cfg.cfg = args.cfg
    if args.window_len:
        cfg.window_len = args.window_len
    if args.crop_height:
        cfg.crop_height = args.crop_height
    if args.gguf.strip():
        cfg.gguf = args.gguf.strip()
    if args.no_lora:
        cfg.lora = None
    elif args.lora.strip():
        cfg.lora = args.lora.strip()
    if args.lora_strength >= 0:
        cfg.lora_strength = args.lora_strength
    if args.weight_dtype.strip():
        cfg.weight_dtype = args.weight_dtype.strip()
    if args.control_mode.strip():
        cfg.control_mode = args.control_mode.strip()
    if args.b2_silhouette_dilate_px >= 0:
        cfg.b2_silhouette_dilate_px = args.b2_silhouette_dilate_px
    if args.b2_feather_px >= 0:
        cfg.b2_feather_px = args.b2_feather_px
    if args.process_pool:
        cfg.render_process_pool = True
    if args.composite_workers:
        cfg.composite_workers = args.composite_workers
    if args.target_gids.strip():
        cfg.target_gids = [int(g) for g in args.target_gids.split(",") if g.strip()]
    log.info("recipe: lora=%s steps=%s cfg=%s weight_dtype=%s control_mode=%s",
             cfg.get("lora"), cfg["steps"], cfg["cfg"],
             cfg.get("weight_dtype", "default"), cfg.get("control_mode", "grey"))
    reader = TrackCacheReader(args.cache_dir)

    # gid resolution mirrors the stage's own logic so --dry-run reports exactly
    # what a real run will render.
    explicit = list(cfg.get("target_gids", []) or [])
    if explicit:
        gids = [int(g) for g in explicit]
    else:
        gids = list_confirmed_gids(reader, thresholds=None)
    total_windows = 0
    for g in gids:
        w = plan_windows(
            reader, g, window_len=int(cfg["window_len"]),
            overlap=int(cfg.get("overlap", 13)), crop_pad=float(cfg["crop_pad"]),
            crop_height=int(cfg["crop_height"]), max_bridge=int(cfg.get("max_bridge", 2)),
        )
        total_windows += len(w)
        log.info("gid %d: %d windows", g, len(w))
    log.info("target gids=%s  total windows=%d  workers=%d",
             gids, total_windows, args.workers)

    if args.dry_run:
        log.info("dry-run: skipping decode + render")
        return 0

    temp_root = Path(args.temp)
    src_key = f"{Path(args.source).stem}_{args.start:g}_{args.end:g}_{args.fps:g}"
    src_cache = DiskFrameProvider(temp_root / "vace_src" / src_key)
    n_frames = src_cache.existing_count()
    if n_frames > 0:
        log.info("reusing decoded source cache (%d frames)", n_frames)
    else:
        log.info("decoding source %s [%s,%s] @ %sfps", args.source, args.start, args.end, args.fps)
        with read_video(args.source, start_time=args.start, end_time=args.end, fps=args.fps) as r:
            n_frames = src_cache.populate(r.frames())
        log.info("decoded %d frames", n_frames)

    # Guard against a stale/short cache: Pass-2 reuses the cache without re-running
    # SAM3, so masks must exist for every decoded source frame. A source LONGER
    # than the cache means the cache can't cover it (wrong/short cache) -> abort.
    # A source shorter than the cache is fine — a partial-range render (--end) of
    # a clip whose cache covers the whole thing.
    cache_frames = reader.frame_indices()
    cache_span = (max(cache_frames) + 1) if cache_frames else 0
    if n_frames > cache_span + 1:
        log.error("cache too short for source: cache spans %d frames, source decoded "
                  "%d — wrong cache for this clip? Aborting.", cache_span, n_frames)
        return 3

    stage = VaceAnonymizationStage(cfg)
    if not stage.is_ready():
        log.error("ComfyUI pool not ready on %s:%d..%d — are all %d workers up?",
                  args.host, args.port, args.port + args.workers - 1, args.workers)
        return 2

    out = Path(args.out)
    base = out.stem        # per-output stitch scratch so distinct runs don't collide
    res = stage.run(
        reader, frames_provider=src_cache,
        frame_indices=list(range(n_frames)),
        work_dir=temp_root / "vace" / base, out_path=out,
    )
    if res is None:
        log.warning("VACE produced no output (no target gids painted)")
        return 3
    log.info("VACE stitched output written: %s", res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
