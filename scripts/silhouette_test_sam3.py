"""Identity-blind SAM 3 diagnostic.

Mirror of scripts/silhouette_test.py for the chunked SAM 3 stage: paints
masks black and overlays the per-detection sam3_obj_id, so a developer can
eyeball within-chunk obj_id stability and mask quality without standing up
ComfyUI, KPL, or InsightFace.

Usage:
    .venv/bin/python scripts/silhouette_test_sam3.py \\
        --input <video> --output <video>
"""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parents[1]))

from ta_src.segmentation.sam3_wrapper import SAM3ChunkedStage, scale_rows_to_frame
from ta_src.video.sam3_prefetcher import RollingChunkPrefetcher
from ta_src.video.videohandler import read_video, save_video

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Path to input video")
    p.add_argument("--output", required=True, help="Path to output video")
    p.add_argument("--sam3-config", default="configs/sam3/default.yaml")
    p.add_argument("--paths-config", default="configs/paths/default.yaml")
    p.add_argument("--device", default=None, help="Override device (cuda)")
    p.add_argument("--start-time", default=None, type=float)
    p.add_argument("--end-time", default=None, type=float)
    p.add_argument("--fps", default=None, type=int)
    p.add_argument(
        "--dilate-px",
        type=int,
        default=16,
        help="Dilate SAM3 mask by N px before paint to close hair/edge leakage. 0 disables.",
    )
    return p.parse_args()


def render_silhouette_with_id(
    frame_rgb: np.ndarray, rows: list[dict], dilate_px: int = 0
) -> np.ndarray:
    out = frame_rgb.copy()
    kernel = None
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    for row in rows:
        mask = row.get("mask")
        if mask is not None:
            if kernel is not None:
                mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
            out[mask] = (0, 0, 0)
        x1, y1, _x2, _y2 = row["bbox"]
        cv2.putText(
            out,
            str(row["sam3_obj_id"]),
            (int(x1), max(15, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
    return out


def main():
    args = parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    sam3_cfg = OmegaConf.load(args.sam3_config)
    paths_cfg = OmegaConf.load(args.paths_config)

    log.info("Loading SAM 3 video predictor (%s)", device)
    stage = SAM3ChunkedStage.from_config(sam3_cfg, device)

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        log.error("Input not found: %s", input_path)
        sys.exit(1)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with read_video(
        str(input_path),
        start_time=args.start_time,
        end_time=args.end_time,
        fps=args.fps,
    ) as reader:
        info = reader.info

    run_id = uuid.uuid4().hex[:8]
    run_dir = Path(paths_cfg.temp) / "sam3_chunks" / run_id
    log.info("Chunk temp dir: %s", run_dir)

    def _frames():
        last_emitted: int = -1
        with read_video(
            str(input_path),
            start_time=args.start_time,
            end_time=args.end_time,
            fps=args.fps,
        ) as reader:
            with RollingChunkPrefetcher(
                frame_source=reader,
                run_dir=run_dir,
                chunk_size=int(sam3_cfg.chunk_size),
                overlap_L=int(sam3_cfg.overlap_L),
            ) as pf:
                for chunk in pf:
                    rows_per_frame = stage.process_chunk(
                        chunk.jpeg_dir, chunk.indices[0]
                    )
                    stage.close_session_and_empty_cache()
                    n = len(chunk.indices)
                    for i in range(n):
                        frame_idx = chunk.indices[i]
                        frame_rgb = chunk.frames[i]
                        rows = rows_per_frame[i]
                        chunk.frames[i] = None
                        rows_per_frame[i] = None
                        # Skip overlap frames already emitted from the prior chunk
                        if frame_idx <= last_emitted:
                            continue
                        last_emitted = frame_idx
                        scale_rows_to_frame(rows, frame_rgb.shape[:2])
                        if (frame_idx + 1) % 30 == 0:
                            log.info("frame %d — %d dets", frame_idx + 1, len(rows))
                        yield render_silhouette_with_id(frame_rgb, rows, args.dilate_px)

    save_video(
        _frames(),
        str(output_path),
        fps=info.fps,
        width=info.width,
        height=info.height,
    )
    log.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
