import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

# See run_pipeline.py — persistent Inductor cache so SAM 3 doesn't recompile.
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(Path.home() / ".cache" / "torchinductor-trackanon"),
)

from ta_src.utils.quiet import quiet_progress, silence_third_party

silence_third_party()
if "--verbose" not in sys.argv:
    quiet_progress()

import cv2
from hydra import compose, initialize_config_dir

log = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parents[1] / "configs"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="center_on_person",
        description=(
            "Normalize a folder of images to a uniform canvas centred on the "
            "person: SAM 3 detects the most-confident person, the image is "
            "scaled so that person fills a fixed height fraction, then cropped/"
            "padded to --width×--height. Unknown args become Hydra overrides."
        ),
    )
    p.add_argument("--input", required=True, metavar="PATH",
                   help="Image file or directory of images")
    p.add_argument("--output", required=True, metavar="DIR",
                   help="Directory to write normalized images into")
    p.add_argument("--width", type=int, default=1024, help="Output width (default 1024)")
    p.add_argument("--height", type=int, default=1365, help="Output height (default 1365, 3:4)")
    p.add_argument("--person-frac", type=float, default=0.8, metavar="F",
                   help="Person bbox height as a fraction of output height (default 0.8)")
    p.add_argument("--pad", choices=("white", "black"), default="white",
                   help="Fill color for off-image regions (default white)")
    p.add_argument("--device", choices=("cuda", "cpu"), default=None)
    p.add_argument("--verbose", action="store_true",
                   help="Surface third-party progress noise (SAM 3 logs, tqdm)")
    return p


def main() -> None:
    args, passthrough = _build_parser().parse_known_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    overrides = list(passthrough)
    if args.device:
        overrides.append(f"pipeline.device={args.device}")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    from ta_src.pipeline.image_anonymize import iter_image_files, make_sam3_rows_fn
    from ta_src.segmentation.sam3_wrapper import SAM3ChunkedStage
    from ta_src.utils.person_crop import center_on_person

    sam3 = SAM3ChunkedStage.from_config(cfg.sam3, cfg.pipeline.device)
    rows_fn = make_sam3_rows_fn(sam3, Path(cfg.paths.get("temp", "data/temp")) / "sam3_image")
    pad_value = 255 if args.pad == "white" else 0

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in iter_image_files(args.input):
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rows = rows_fn(rgb)
        if not rows:
            log.warning("No person detected in %s — skipped", path.name)
            continue
        person = max(rows, key=lambda r: float(r.get("score", r.get("mask_score", 0.0))))
        out_rgb = center_on_person(rgb, person["bbox"], out_w=args.width,
                                   out_h=args.height, person_height_frac=args.person_frac,
                                   pad_value=pad_value)
        cv2.imwrite(str(out_dir / path.name), cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
        written += 1
    log.info("Wrote %d normalized image(s) to %s", written, out_dir)


if __name__ == "__main__":
    main()
