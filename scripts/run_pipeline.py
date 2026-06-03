import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

# Persistent Inductor cache. Default /tmp/torchinductor_$USER is wiped by
# systemd-tmpfiles / WSL2 restarts, forcing SAM3's max-autotune mask_decoder
# to recompile on first-hit shapes (~15 min stalls). User shell override wins.
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(Path.home() / ".cache" / "torchinductor-trackanon"),
)

from ta_src.utils.quiet import quiet_progress, silence_third_party

silence_third_party()
# Quiet by default. --verbose has to take effect before sam3 imports
# (LOG_LEVEL is read at logger construction), so peek argv now and let the
# parser still surface the flag for --help.
if "--verbose" not in sys.argv:
    quiet_progress()

from hydra import compose, initialize_config_dir

from ta_src.pipeline.main_pipeline import Pipeline

log = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parents[1] / "configs"
OVERRIDES_DEST = "overrides"


class _AppendOverride(argparse.Action):
    # Format the flag's value into `template` and append to ns.overrides.
    def __init__(self, option_strings, dest, template, **kwargs):
        self._template = template
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        lst = getattr(namespace, self.dest, None) or []
        lst.append(self._template.format(values))
        setattr(namespace, self.dest, lst)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline",
        description=(
            "Run the trackanon pipeline. "
            "Unknown arguments are forwarded verbatim as Hydra overrides "
            "(e.g. pose=dwpose, temporal.window=8)."
        ),
    )

    # ---- value flags ----
    p.add_argument(
        "--input", action=_AppendOverride, template="paths.input={}",
        metavar="PATH", dest=OVERRIDES_DEST, help="Override paths.input",
    )
    p.add_argument(
        "--output", action=_AppendOverride, template="paths.output={}",
        metavar="PATH", dest=OVERRIDES_DEST, help="Override paths.output",
    )
    p.add_argument(
        "--device", action=_AppendOverride, template="pipeline.device={}",
        choices=("cuda", "cpu"), dest=OVERRIDES_DEST,
        help="Override pipeline.device",
    )

    # ---- boolean flags (each flips away from the config default) ----
    p.add_argument(
        "--debug", action="store_true",
        help="Set root logger to DEBUG (default INFO)",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help=(
            "Surface third-party progress noise (SAM 3 INFO logs + tqdm "
            "bars). Default is quiet — only the per-frame pipeline "
            "progress bar renders."
        ),
    )
    p.add_argument(
        "--visualize", action="append_const",
        const="pipeline.visualize=true", dest=OVERRIDES_DEST,
        help="Enable pipeline.visualize",
    )
    p.add_argument(
        "--save-visualization", action="append_const",
        const="pipeline.save_visualization=true", dest=OVERRIDES_DEST,
        help="Enable pipeline.save_visualization",
    )
    p.add_argument(
        "--op-edit", action="append_const",
        const="pipeline.op_edit.enabled=true", dest=OVERRIDES_DEST,
        help="Enable pipeline.op_edit.enabled",
    )
    p.add_argument(
        "--no-run-pose", action="append_const",
        const="pipeline.run_pose=false", dest=OVERRIDES_DEST,
        help="Disable pipeline.run_pose",
    )
    p.add_argument(
        "--no-run-tracking", action="append_const",
        const="pipeline.run_tracking=false", dest=OVERRIDES_DEST,
        help="Disable pipeline.run_tracking",
    )
    p.add_argument(
        "--no-run-anonymization", action="append_const",
        const="pipeline.run_anonymization=false", dest=OVERRIDES_DEST,
        help="Disable pipeline.run_anonymization",
    )
    return p


def main() -> None:
    args, passthrough = _build_parser().parse_known_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    overrides = (getattr(args, OVERRIDES_DEST) or []) + passthrough
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    # Make the run.log self-describing: the literal command + the resolved knobs
    # that are otherwise unrecoverable from the log (window, fps, backend).
    log.info("run command: %s", " ".join(sys.argv))
    log.info(
        "resolved run: input=%s output=%s window=[%s, %s] fps=%s generator=%s",
        cfg.paths.get("input"), cfg.paths.get("output"),
        cfg.temporal.get("start_time"), cfg.temporal.get("end_time"),
        cfg.temporal.get("fps"), cfg.anonymization.get("generator", "vace"),
    )
    Pipeline(cfg).run()


if __name__ == "__main__":
    main()
