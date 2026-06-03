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

from hydra import compose, initialize_config_dir

log = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parents[1] / "configs"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="anonymize_images",
        description=(
            "Run the pipeline on a single image or a folder of images — no "
            "tracking. With --draw-* flags, save per-stage overlay figures "
            "(masks/detections/track/keypoints) to their own subfolders. With "
            "no --draw-* flag: blur every detected person. Persona diffusion is "
            "not offered for stills (the VACE backend operates on clips). "
            "Unknown args are forwarded as Hydra overrides."
        ),
    )
    p.add_argument("--input", required=True, metavar="PATH",
                   help="Image file or directory of images")
    p.add_argument("--output", required=True, metavar="DIR",
                   help="Directory to write anonymized images into")
    p.add_argument("--face-floor", type=float, default=0.5, metavar="COS",
                   help="Minimum face-centroid cosine to bind a person to a gid "
                        "in the track overlay (default 0.5)")
    p.add_argument("--draw-masks", action="store_true",
                   help="Save a SAM 3 segmentation-mask overlay per image (output/masks/)")
    p.add_argument("--draw-detections", action="store_true",
                   help="Save a detection bbox overlay per image (output/detections/)")
    p.add_argument("--draw-track", action="store_true",
                   help="Save a recognition overlay — objid + gid + KPL name + cosine — "
                        "per image (output/track/)")
    p.add_argument("--draw-keypoints", action="store_true",
                   help="Save a DWpose skeleton overlay per image (output/keypoints/)")
    p.add_argument("--figure", action="append", default=[], metavar="LAYERS",
                   help="Composite figure: comma-separated layers from "
                        "masks,detections,track,keypoints stacked on one image "
                        "(like --save-visualization), saved to output/<layers>/ "
                        "(repeatable). E.g. --figure masks,detections")
    p.add_argument("--label-scale", type=float, default=4.0, metavar="X",
                   help="Multiplier for detection/track label font size in "
                        "figures (default 4.0)")
    p.add_argument("--device", choices=("cuda", "cpu"), default=None)
    p.add_argument("--debug", action="store_true",
                   help="Set root logger to DEBUG (surfaces per-person dispatch decisions)")
    p.add_argument("--verbose", action="store_true",
                   help="Surface third-party progress noise (SAM 3 logs, tqdm)")
    return p


_FIGURE_LAYERS = ("masks", "detections", "track", "keypoints")


def _figures_from_args(args, parser) -> dict:
    """Figure specs from the CLI: each --draw-<layer> is a single-layer figure,
    each --figure a,b,... a composite keyed by its layers joined with '_'."""
    figures: dict = {}
    for layer, enabled in (("masks", args.draw_masks),
                           ("detections", args.draw_detections),
                           ("track", args.draw_track),
                           ("keypoints", args.draw_keypoints)):
        if enabled:
            figures[layer] = [layer]
    for spec in args.figure:
        layers = [s.strip() for s in spec.split(",") if s.strip()]
        bad = [l for l in layers if l not in _FIGURE_LAYERS]
        if bad:
            parser.error(f"--figure: unknown layer(s) {bad}; "
                         f"choose from {','.join(_FIGURE_LAYERS)}")
        if layers:
            figures["_".join(layers)] = layers
    return figures


def main() -> None:
    args, passthrough = _build_parser().parse_known_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    overrides = list(passthrough)
    if args.device:
        overrides.append(f"pipeline.device={args.device}")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    from ta_src.pipeline.image_anonymize import (
        anonymize_path,
        build_process_fn,
        build_render_fn,
        render_stages_path,
    )

    figures = _figures_from_args(args, _build_parser())
    if figures:
        # Figure mode: each --draw-* is a single-layer figure, each --figure a
        # composite. Each figure writes to its own output subfolder.
        log.info("Figures: %s (input=%s)", sorted(figures), args.input)
        render_fn = build_render_fn(cfg, figures,
                                    face_match_floor=args.face_floor,
                                    label_scale=args.label_scale)
        written = render_stages_path(args.input, args.output, render_fn)
        log.info("Wrote %d figure(s) to %s", len(written), args.output)
        return

    log.info("Image anonymization: blur-only (input=%s)", args.input)
    process_fn = build_process_fn(cfg, face_match_floor=args.face_floor)
    written = anonymize_path(args.input, args.output, process_fn)
    log.info("Wrote %d anonymized image(s) to %s", len(written), args.output)


if __name__ == "__main__":
    main()
