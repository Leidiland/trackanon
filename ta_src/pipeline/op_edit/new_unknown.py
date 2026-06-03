"""Mid-clip op_edit trigger: pause for the operator the first time a genuinely
new unidentified person settles in frame, not just at frame 0.

Chunked SAM 3 renumbers sam3_obj_ids every chunk, so the same continuously
-unknown person would otherwise re-fire every chunk. Dedup is therefore by
location: an abstaining detection near a slot we've already handled is the same
person; a slot goes stale and is pruned once unseen, so a person who is labelled
and later genuinely lost re-fires. A short sustained-abstain window gives the
resolver a chance to bind first and filters transient SAM 3 noise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class _Slot:
    obj_id: int
    cx: float
    cy: float
    last_frame: int
    streak: int
    handled: bool


class NewUnknownTrigger:
    def __init__(
        self,
        *,
        sustained_frames: int = 10,
        dedup_centroid_frac: float = 0.06,
        dedup_recent_frames: int = 50,
        min_det_score: float = 0.5,
    ) -> None:
        self._sustained = int(sustained_frames)
        self._dedup_frac = float(dedup_centroid_frac)
        self._recent = int(dedup_recent_frames)
        self._min_score = float(min_det_score)
        self._slots: list[_Slot] = []

    def update(
        self, frame_idx: int, enriched_rows: list[dict],
        frame_shape: tuple[int, int],
    ) -> list[int]:
        """Returns sam3_obj_ids to prompt the operator about this frame."""
        h, w = frame_shape
        radius = self._dedup_frac * math.hypot(float(h), float(w))
        cands: list[tuple[int, float, float]] = []
        for row in enriched_rows:
            if int(row["global_id"]) >= 0:
                continue
            if float(row.get("score", 1.0)) < self._min_score:
                continue
            x1, y1, x2, y2 = (float(v) for v in row["bbox"])
            cands.append((int(row["sam3_obj_id"]), (x1 + x2) / 2.0, (y1 + y2) / 2.0))

        # Greedy nearest match of each candidate to a live slot within radius.
        pairs: list[tuple[float, int, int]] = []
        for ci, (_oid, cx, cy) in enumerate(cands):
            for si, slot in enumerate(self._slots):
                if frame_idx - slot.last_frame > self._recent:
                    continue
                dist = math.hypot(cx - slot.cx, cy - slot.cy)
                if dist <= radius:
                    pairs.append((dist, ci, si))
        pairs.sort()
        cand_to_slot: dict[int, int] = {}
        used_slots: set[int] = set()
        for _dist, ci, si in pairs:
            if ci in cand_to_slot or si in used_slots:
                continue
            cand_to_slot[ci] = si
            used_slots.add(si)

        to_prompt: list[int] = []
        for ci, (oid, cx, cy) in enumerate(cands):
            si = cand_to_slot.get(ci)
            if si is None:
                slot = _Slot(oid, cx, cy, frame_idx, 1, False)
                self._slots.append(slot)
            else:
                slot = self._slots[si]
                slot.obj_id, slot.cx, slot.cy = oid, cx, cy
                slot.last_frame = frame_idx
                slot.streak += 1
            if slot.streak >= self._sustained and not slot.handled:
                slot.handled = True
                to_prompt.append(oid)

        self._slots = [
            s for s in self._slots if frame_idx - s.last_frame <= self._recent
        ]
        return to_prompt
