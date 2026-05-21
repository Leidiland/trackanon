"""Shared validator for the operator's sam3_obj_id → global_id mapping.

The stdin path (op_edit.session) and the web path (op_edit.web) both feed
the operator's edits through this function so the closed-world rules
(known gid, one-to-one assignment) can't drift between UIs.
"""
from __future__ import annotations


class OpEditValidationError(ValueError):
    """The operator's mapping violates a closed-world rule."""


def validate_operator_mapping(
    by_obj: dict[int, int],
    known_gids: set[int] | frozenset[int],
) -> None:
    """Raise OpEditValidationError if the mapping is unusable."""
    for obj_id, gid in by_obj.items():
        if gid >= 0 and gid not in known_gids:
            raise OpEditValidationError(
                f"sam3_obj_id={obj_id}: gid={gid} not in KPL "
                f"(known: {sorted(known_gids)})"
            )
    assigned = [g for g in by_obj.values() if g >= 0]
    if len(assigned) != len(set(assigned)):
        raise OpEditValidationError(
            f"duplicate operator_gid in mapping: {sorted(assigned)} "
            "(Hungarian is one-to-one)"
        )
