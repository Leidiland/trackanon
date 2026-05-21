"""ADR-0015 op-edit frame-0 bootstrap: pause once at frame 0 so the operator
can correct the resolver's proposed sam3_obj_id → global_id mapping before
identity-conditioned generation locks in.

Two UIs over the same sidecar-JSON wire format:
- stdin (default if no daemon is injected): operator edits the JSON in an
  editor, hits Enter on the terminal.
- web (when an OpEditWebDaemon is injected): browser tab with the annotated
  frame + clickable boxes; daemon writes the JSON back on Apply.
The JSON is the audit trail and the apply contract for both paths.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import cv2
import numpy as np

from ta_src.pipeline.op_edit.errors import OpEditAbort
from ta_src.pipeline.op_edit.validation import (
    OpEditValidationError,
    validate_operator_mapping,
)
from ta_src.tracking.identity_memory import IdentityGallery

if TYPE_CHECKING:
    from ta_src.pipeline.op_edit.web import OpEditWebDaemon


class OpEditSession:
    def __init__(
        self,
        *,
        gallery: IdentityGallery,
        kpl_root: Path,
        artifact_dir: Path,
        stdin: Optional[Callable[[str], str]] = None,
        daemon: Optional["OpEditWebDaemon"] = None,
        # Diagnostic: reuse a previously-edited sidecar JSON instead of
        # overwriting it with a fresh proposal. ADR-0015 rejects this as a
        # default (inverts the trust model); only set true for repro runs.
        reuse_existing: bool = False,
    ) -> None:
        self._gallery = gallery
        self._kpl_root = Path(kpl_root)
        self._artifact_dir = Path(artifact_dir)
        self._input = stdin if stdin is not None else input
        self._daemon = daemon
        self._reuse_existing = bool(reuse_existing)

    def prompt(
        self,
        frame_rgb: np.ndarray,
        enriched_rows: list[dict],
        frame_idx: int,
        video_name: str,
    ) -> dict[int, Optional[int]]:
        if not enriched_rows:
            return {}
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        json_path = self._artifact_dir / f"{video_name}_frame{frame_idx}_map.json"
        png_path = self._artifact_dir / f"{video_name}_frame{frame_idx}.png"
        if not (self._reuse_existing and json_path.exists()):
            self._write_proposal_json(json_path, enriched_rows)
        self._write_frame_png(png_path, frame_rgb)
        known_gids = {ident.global_id for ident in self._gallery}
        live_obj_ids = {int(r["sam3_obj_id"]) for r in enriched_rows}
        if self._daemon is not None:
            token = self._daemon.register_pause(
                json_path=json_path, png_path=png_path, known_gids=known_gids,
            )
            self._daemon.open_pause_in_browser(token)
            self._daemon.wait_for(token)
            # Apply endpoint already validated; revalidate locally so the
            # in-process contract is identical to the stdin path.
            return self._read_and_validate(json_path, known_gids, live_obj_ids)
        while True:
            response = self._input(
                f"[op-edit] Edit operator_gid values in {json_path}, then "
                "press Enter to apply or 'a' to abort: "
            )
            if response.strip().lower() == "a":
                raise OpEditAbort(f"operator aborted at frame {frame_idx}")
            try:
                return self._read_and_validate(json_path, known_gids, live_obj_ids)
            except OpEditValidationError as exc:
                print(f"[op-edit] {exc} — fix the JSON and press Enter to retry.")

    def _write_proposal_json(self, path: Path, rows: list[dict]) -> None:
        known = [
            {"gid": ident.global_id, "name": ident.name}
            for ident in self._gallery
        ]
        objects = []
        for row in rows:
            gid = int(row["global_id"])
            name = self._gallery.get(gid).name if gid >= 0 else None
            objects.append({
                "sam3_obj_id": int(row["sam3_obj_id"]),
                "bbox": [float(v) for v in row["bbox"]],
                "proposed_gid": gid,
                "proposed_name": name,
                "operator_gid": gid,
            })
        path.write_text(json.dumps({
            "known_persons": known,
            "objects": objects,
        }, indent=2))

    def _write_frame_png(self, path: Path, frame_rgb: np.ndarray) -> None:
        # Raw frame only; the web SVG and the JSON's bbox field carry the
        # box / label rendering — drawing them in cv2 too would double up.
        cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    def _read_and_validate(
        self,
        path: Path,
        known_gids: set[int],
        live_obj_ids: set[int],
    ) -> dict[int, Optional[int]]:
        data = json.loads(path.read_text())
        by_obj = {int(o["sam3_obj_id"]): int(o["operator_gid"])
                  for o in data["objects"]}
        validate_operator_mapping(by_obj, known_gids)
        return {obj_id: (None if gid < 0 else gid)
                for obj_id, gid in by_obj.items()}


def _palette(idx: int) -> tuple[int, int, int]:
    hue = (idx * 47) % 180
    hsv = np.uint8([[[hue, 200, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])
