"""Shared op-edit exceptions — kept in their own module so session and web
can both import without a cycle."""
from __future__ import annotations


class OpEditAbort(RuntimeError):
    """Operator aborted the pause — pipeline shuts down cleanly."""
