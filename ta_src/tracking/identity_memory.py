"""Closed-world Identity registry: every Identity seeded from a KPL sub-folder at startup.

Centroids frozen at KPL-seeded values (no runtime EMA drift). Per-track
running state lives on IdentityResolver.TrackBinding.
"""
from __future__ import annotations

from typing import Iterator, TYPE_CHECKING

from ta_src.tracking.identity import Identity

if TYPE_CHECKING:
    from ta_src.tracking.kpl_seeder import KPLSeed


class IdentityGallery:
    def __init__(self):
        self._identities: dict[int, Identity] = {}

    def seed_from_kpl(self, seeds: list["KPLSeed"]) -> None:
        for seed in seeds:
            self._identities[seed.global_id] = Identity(
                global_id=seed.global_id,
                face_centroid=seed.face_centroid,
                appearance_centroid=seed.appearance_centroid,
                prompt=seed.prompt,
                name=seed.name,
            )

    def global_ids(self) -> list[int]:
        return list(self._identities.keys())

    def get(self, global_id: int) -> Identity | None:
        return self._identities.get(global_id)

    def get_prompt(self, global_id: int) -> str | None:
        identity = self._identities.get(global_id)
        return identity.prompt if identity is not None else None

    def __iter__(self) -> Iterator[Identity]:
        return iter(self._identities.values())

    def __len__(self) -> int:
        return len(self._identities)
