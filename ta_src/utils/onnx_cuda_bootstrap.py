"""Force-preload the NVIDIA shared libraries that onnxruntime-gpu needs.

onnxruntime-gpu 1.25+ requires cuDNN 9 / CUDA 12 .so files at runtime. Even
when the matching wheels (`nvidia-cudnn-cu12`, `nvidia-cublas-cu12`, etc.)
are pip-installed, ORT only finds them when they're on LD_LIBRARY_PATH —
which Python can't change after process start. We bridge that by ctypes-
preloading the .so files with RTLD_GLOBAL before any wrapper imports ORT.

Importing this module is idempotent and a no-op when:
  - the nvidia-* wheels aren't installed (ORT will fall back to CPU);
  - we're not on linux;
  - cuDNN is already on the system loader path.
"""
from __future__ import annotations

import ctypes
import glob
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_LOADED = False


def preload_nvidia_libs() -> None:
    global _LOADED
    if _LOADED or sys.platform != "linux":
        return
    _LOADED = True

    try:
        import nvidia  # noqa: F401  (sentinel: the nvidia-* wheels are present)
    except ImportError:
        return

    # Resolve nvidia .so files from the active venv's nvidia/ tree, not from
    # a system path — keeps preload aligned with whichever torch + CUDA wheel
    # set is installed here.
    try:
        import nvidia as _n
        nvidia_root = Path(_n.__file__).resolve().parent
    except Exception:
        return

    # Order matters: cublas → cudart → cudnn. cuDNN depends on cuBLAS at load.
    patterns = [
        "cublas/lib/libcublas.so.12",
        "cuda_runtime/lib/libcudart.so.12",
        "cudnn/lib/libcudnn.so.9",
    ]
    for pat in patterns:
        for path in glob.glob(str(nvidia_root / pat)):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                log.debug("preloaded %s", path)
            except OSError as e:
                log.debug("preload failed for %s: %s", path, e)


preload_nvidia_libs()
