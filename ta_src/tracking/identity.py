"""Frozen Persona record — immutable centroids + prompt seeded from the KPL.

Per-frame state lives on IdentityResolver.TrackBinding, not here.
"""
from __future__ import annotations

import numpy as np

from ta_src.tracking.hungarian_assigner import Candidate


class Identity:
    def __init__(
        self,
        global_id: int,
        face_centroid: np.ndarray,
        appearance_centroid: np.ndarray,
        prompt: str,
        name: str = "",
    ):
        self._global_id = global_id
        self._face_centroid = face_centroid
        self._appearance_centroid = appearance_centroid
        self._prompt = prompt
        self._name = name

    @property
    def global_id(self) -> int:
        return self._global_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def face_centroid(self) -> np.ndarray:
        return self._face_centroid

    @property
    def appearance_centroid(self) -> np.ndarray:
        return self._appearance_centroid

    def to_candidate(self) -> Candidate:
        return Candidate(
            global_id=self._global_id,
            face_centroid=self._face_centroid,
            appearance_centroid=self._appearance_centroid,
            kalman=None,
        )
