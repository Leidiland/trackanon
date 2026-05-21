from __future__ import annotations

import contextlib
import logging
import os
import sys
import warnings


def silence_third_party() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def quiet_progress() -> None:
    """Silence third-party progress noise (SAM 3 INFO logs + tqdm bars from
    SAM 3's io_utils/predictor). The pipeline's own per-frame tqdm passes
    ``disable=False`` to opt out of the patch."""
    # SAM 3's logger.py reads this at import time; must be set before any
    # `from sam3...` line runs.
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    import tqdm as _tqdm

    _orig_init = _tqdm.tqdm.__init__

    def _quiet_init(self, *args, **kwargs):
        kwargs.setdefault("disable", True)
        _orig_init(self, *args, **kwargs)

    _tqdm.tqdm.__init__ = _quiet_init

    # torchreid: torch.load(weights_only=False) FutureWarning (osnet.py)
    warnings.filterwarnings(
        "ignore",
        message=r"You are using `torch\.load` with `weights_only=False`.*",
        category=FutureWarning,
    )
    # insightface: SimilarityTransform.estimate FutureWarning (face_align.py)
    warnings.filterwarnings(
        "ignore",
        message=r"`estimate` is deprecated since version 0\.26.*",
        category=FutureWarning,
    )
    # torch dynamo symbolic-shape interp: floods stderr with hundreds of
    # "failed while executing pow_by_natural(...)" log lines on any compiled
    # graph with x**-1 expressions (SAM 3 et al.). Not actionable.
    logging.getLogger("torch.utils._sympy.interp").setLevel(logging.ERROR)


@contextlib.contextmanager
def suppressed_stdout():
    saved = sys.stdout
    try:
        sys.stdout = open(os.devnull, "w")
        yield
    finally:
        try:
            sys.stdout.close()
        except Exception:
            pass
        sys.stdout = saved
