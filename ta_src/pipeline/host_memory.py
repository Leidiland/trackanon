"""Return the pipeline process's freed heap pages to the OS.

The pipeline can't restart itself the way the ComfyUI server can — it holds the
IdentityResolver gallery and per-run confirmation state. This is the in-process
analog: collect reference cycles, then ask glibc to release the freed top of the
heap arena back to the OS so RSS doesn't ratchet up across chunks and videos.
"""
from __future__ import annotations

import ctypes
import gc
import logging

log = logging.getLogger(__name__)


def _malloc_trim() -> bool:
    # glibc-only; silently no-op on musl / non-Linux.
    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        return False
    if not hasattr(libc, "malloc_trim"):
        return False
    libc.malloc_trim(0)
    return True


def _rss_mib() -> int | None:
    # Best-effort diagnostic; must never break the run.
    try:
        import psutil
        return psutil.Process().memory_info().rss // (1024 * 1024)
    except Exception:
        return None


def trim() -> bool:
    """Collect ref cycles then trim the glibc arena. True if the arena trimmed."""
    before = _rss_mib()
    gc.collect()
    trimmed = _malloc_trim()
    after = _rss_mib()
    if before is not None and after is not None:
        log.debug("pipeline host RSS %d -> %d MiB after trim", before, after)
    return trimmed
