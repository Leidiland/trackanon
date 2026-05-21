from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Callable, Iterable, Protocol

import cv2
import numpy as np

log = logging.getLogger(__name__)


class _CropCacheLike(Protocol):
    def load(self, global_id: int) -> np.ndarray | None: ...
    def save_first_generation(
        self, global_id: int, crop_rgb: np.ndarray
    ) -> bool: ...


class _IdentityLike(Protocol):
    @property
    def global_id(self) -> int: ...
    @property
    def prompt(self) -> str: ...


def _resize_for_sd_latent(
    image: np.ndarray, short_side: int, long_side_max: int,
) -> np.ndarray:
    """Resize an init image so its latent stays in SD1.5's trained range.

    Without this, KPL photos at native resolution (>800px on a side) cause
    SD1.5 to "tile" — generate two or more copies of the subject side by
    side — because the latent grid exceeds what the UNet ever saw during
    training. The shorter edge is pinned at ``short_side`` (512 for SD1.5)
    and the longer edge clamped at ``long_side_max`` (typically 768) so
    aspect is preserved but no dimension blows past the comfort zone. Both
    output dimensions are snapped to multiples of 8 — VAEEncode requires it.
    """
    h, w = image.shape[:2]
    scale = short_side / float(min(h, w))
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    if max(new_h, new_w) > long_side_max:
        scale = long_side_max / float(max(h, w))
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
    new_h = max(64, (new_h // 8) * 8)
    new_w = max(64, (new_w // 8) * 8)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def score_candidate(
    face_obj, existing_embeddings, clip_alignment: float = 1.0
) -> float:
    """Diversity- and prompt-aware quality score for a prewarm candidate.

    score = det × frontality × diversity × clip_alignment

    clip_alignment is the cosine similarity (in [0, 1]) between the candidate
    image and the prompt. Default 1.0 keeps legacy callers unaffected. Without
    this multiplier the scorer optimises face-embedding distance only and can
    pick off-prompt candidates (e.g. a woman for a "man" prompt) because they
    happened to land furthest from the existing-embedding cone.
    """
    det = float(face_obj.det_score)
    yaw_deg = float(face_obj.pose[0])
    pitch_deg = float(face_obj.pose[1])
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    frontality = max(0.0, min(1.0, float(np.cos(yaw) * np.cos(pitch))))
    if not existing_embeddings:
        diversity = 1.0
    else:
        emb = np.asarray(face_obj.normed_embedding, dtype=np.float32)
        max_sim = max(float(np.dot(emb, np.asarray(e, dtype=np.float32)))
                      for e in existing_embeddings)
        diversity = max(0.0, 1.0 - max_sim)
    return det * frontality * diversity * max(0.0, float(clip_alignment))


def select_best(triples):
    """Return (crop, embedding) of the highest-scoring triple, or None on empty."""
    if not triples:
        return None
    best = max(triples, key=lambda t: t[0])
    return best[1], best[2]


class PrewarmFailure(RuntimeError):
    """Raised when face validation fails for every retry attempt.

    The exception carries enough context for the operator to identify which
    Identity needs prompt curation (gid + KPL folder come from the surrounding
    AnonymizationStage when re-raised; PrewarmGenerator carries gid/prompt/seed).
    """

    def __init__(self, global_id: int, prompt: str, last_seed: int, extra: str = ""):
        suffix = f" — {extra}" if extra else ""
        super().__init__(
            f"PrewarmGenerator: face validation failed after retries for "
            f"gid={global_id} prompt={prompt!r} last_seed={last_seed}{suffix}"
        )
        self.global_id = global_id
        self.prompt = prompt
        self.last_seed = last_seed


class PrewarmGenerator:
    """Synthetic Reference Crop generator: txt2img + InsightFace validation + retry.

    No persistent state; the on-disk cache write is the caller's responsibility.
    """

    def __init__(
        self,
        comfy_client,
        workflow: dict,
        node_ids: dict,
        faceid_wrapper,
        *,
        max_candidates: int,
        canvas_size: tuple[int, int],
        debug_dir: str | Path | None = None,
        captioner=None,
        clip_drop_quartile: float = 0.25,
        kpl_leak_threshold: float = 0.3,
    ):
        self._client = comfy_client
        self._workflow = workflow
        self._node_ids = node_ids
        self._faceid = faceid_wrapper
        self._max_candidates = int(max_candidates)
        self._canvas_size = canvas_size
        self._debug_dir = Path(debug_dir) if debug_dir else None
        # CLIP-based prompt-fidelity scoring. None disables both the
        # alignment multiplier and the bottom-quartile gate (legacy path).
        self._captioner = captioner
        self._clip_drop_quartile = float(clip_drop_quartile)
        # img2img anti-leak: cosine ceiling for FaceID similarity to the
        # KPL face centroid. Candidates above this are rejected; if every
        # candidate exceeds it, PrewarmFailure fires (operator must adjust
        # denoise upward or pick a less identifying KPL representative).
        self._kpl_leak_threshold = float(kpl_leak_threshold)
        # The workflow tells us which mode we're in: presence of an
        # ``empty_latent`` slot means txt2img; ``load_image`` means img2img.
        self._mode = "img2img" if "load_image" in node_ids else "txt2img"

    def generate_best_of_k(
        self,
        global_id: int,
        prompt: str,
        seed: int,
        existing_embeddings: list,
        init_image: np.ndarray | None = None,
        kpl_face_embs: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate K candidates and return the highest-scored (crop, embedding).

        K = ``max_candidates``. For each attempt ``k`` in 0..K-1 the workflow is
        seeded with ``seed + k``; the resulting crop is run through the FaceID
        wrapper to obtain a face object. Face-validated survivors are scored
        via :func:`score_candidate` against ``existing_embeddings`` and the
        highest-scoring (crop, embedding) pair is returned. Raises
        :class:`PrewarmFailure` (carrying gid, prompt, last seed) when zero
        candidates within the K budget produce a face embedding. The loop
        never early-exits on first success — the K budget is always honoured.
        """
        # img2img mode requires both an init image and the KPL face
        # embeddings for the anti-leak gate. Refusing to run without them
        # keeps the privacy guarantee at the architectural seam — never
        # silently fall back to a leakier path.
        if self._mode == "img2img":
            if init_image is None or not kpl_face_embs:
                raise RuntimeError(
                    f"PrewarmGenerator(img2img): gid={global_id} missing "
                    f"init_image or kpl_face_embs — both are mandatory for "
                    f"the anti-leak gate."
                )
            # SD1.5 tiles subjects when latent dimensions exceed its training
            # range; resize before upload to keep one-person-per-canvas.
            init_image_resized = _resize_for_sd_latent(
                init_image,
                short_side=self._canvas_size[0],
                long_side_max=self._canvas_size[1],
            )
            init_image_name = self._client.upload_image(
                init_image_resized, f"kpl_init_gid{global_id}.png",
            )
        else:
            init_image_name = None

        # Face-validated candidates carry their CLIP alignment so the
        # tier-2 gate can drop the worst quartile before final scoring.
        # Tuple: (crop, emb, face_obj, clip_alignment, seed).
        candidates: list[tuple[np.ndarray, np.ndarray, object, float, int]] = []
        rejected_by_leak = 0
        last_seed = seed
        debug_gid_dir = self._open_debug_dir(global_id)
        for attempt in range(self._max_candidates):
            current_seed = seed + attempt
            last_seed = current_seed
            wf = self._patch(
                prompt=prompt, seed=current_seed, gid=global_id,
                init_image_name=init_image_name,
            )
            prompt_id = self._client.queue_prompt(wf)
            history = self._client.wait_for_result(prompt_id)
            img_info = self._client.extract_output_image(
                history, self._node_ids["save_image"]
            )
            crop = self._client.fetch_image(
                img_info["filename"],
                subfolder=img_info.get("subfolder", ""),
                img_type=img_info.get("type", "output"),
            )
            self._dump_candidate(debug_gid_dir, current_seed, crop)
            face_obj = self._faceid.extract_face_obj(crop)
            if face_obj is None:
                log.warning(
                    "PrewarmGenerator: no face detected for gid=%d attempt=%d seed=%d",
                    global_id, attempt + 1, current_seed,
                )
                continue
            emb = np.asarray(face_obj.normed_embedding, dtype=np.float32)
            # Anti-leak: max cosine across every KPL photo embedding (not
            # the centroid). A candidate that matches any single KPL shot
            # closely leaks even when its centroid distance looks safe —
            # the centroid is a fictional midpoint when the KPL person
            # varies (glasses, expression, age, lighting).
            if kpl_face_embs:
                max_kpl_sim = max(
                    float(np.dot(emb, np.asarray(k, dtype=np.float32)))
                    for k in kpl_face_embs
                )
                if max_kpl_sim > self._kpl_leak_threshold:
                    rejected_by_leak += 1
                    log.info(
                        "PrewarmGenerator: gid=%d seed=%d rejected by anti-leak "
                        "(max cos=%.3f > %.3f over %d KPL photos)",
                        global_id, current_seed, max_kpl_sim,
                        self._kpl_leak_threshold, len(kpl_face_embs),
                    )
                    continue
            clip_align = (
                self._captioner.similarity(crop, prompt)
                if self._captioner is not None else 1.0
            )
            candidates.append((crop, emb, face_obj, clip_align, current_seed))

        if not candidates:
            if rejected_by_leak > 0:
                raise PrewarmFailure(
                    global_id, prompt, last_seed,
                    extra=f"all {rejected_by_leak} face-detectable candidates "
                          f"exceeded anti-leak ceiling ({self._kpl_leak_threshold}). "
                          f"Raise denoise or lower kpl_leak_threshold.",
                )
            raise PrewarmFailure(global_id, prompt, last_seed)

        # Tier-2: drop the bottom quartile by CLIP alignment when CLIP is
        # active and we have enough survivors (≥4) for the quartile to be
        # meaningful. Prevents pathological cases where every candidate
        # drifts off-prompt and the scorer still picks one of them.
        kept = candidates
        if (
            self._captioner is not None
            and len(candidates) >= 4
            and self._clip_drop_quartile > 0.0
        ):
            sorted_by_clip = sorted(candidates, key=lambda c: c[3])
            n_drop = int(len(sorted_by_clip) * self._clip_drop_quartile)
            if n_drop > 0:
                kept = sorted_by_clip[n_drop:]
                log.info(
                    "PrewarmGenerator: gid=%d dropped %d/%d candidates by CLIP "
                    "(min kept=%.3f)",
                    global_id, n_drop, len(candidates), kept[0][3],
                )

        scored = [
            (
                score_candidate(face_obj, existing_embeddings, clip_align),
                crop, emb, current_seed,
            )
            for (crop, emb, face_obj, clip_align, current_seed) in kept
        ]
        best = max(scored, key=lambda t: t[0])
        self._dump_chosen(debug_gid_dir, best[3])
        return best[1], best[2]

    def _open_debug_dir(self, global_id: int) -> Path | None:
        if self._debug_dir is None:
            return None
        gid_dir = self._debug_dir / str(global_id)
        gid_dir.mkdir(parents=True, exist_ok=True)
        return gid_dir

    @staticmethod
    def _dump_candidate(gid_dir: Path | None, seed: int, crop: np.ndarray) -> None:
        if gid_dir is None:
            return
        path = gid_dir / f"seed_{seed}.png"
        cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    @staticmethod
    def _dump_chosen(gid_dir: Path | None, chosen_seed: int) -> None:
        if gid_dir is None:
            return
        (gid_dir / "chosen.flag").write_text(f"{chosen_seed}\n")

    def extract_embedding(self, crop: np.ndarray) -> np.ndarray | None:
        """Helper for orchestrators: extract a face embedding for the
        diversity accumulator. Returns None when no face is detected."""
        face_obj = self._faceid.extract_face_obj(crop)
        if face_obj is None:
            return None
        return np.asarray(face_obj.normed_embedding, dtype=np.float32)

    def _patch(
        self, *, prompt: str, seed: int, gid: int,
        init_image_name: str | None = None,
    ) -> dict:
        wf = copy.deepcopy(self._workflow)
        nid_pos = self._node_ids["positive"]
        nid_ks = self._node_ids["ksampler"]
        nid_save = self._node_ids["save_image"]
        wf[nid_pos]["inputs"]["text"] = prompt
        wf[nid_ks]["inputs"]["seed"] = seed
        wf[nid_save]["inputs"]["filename_prefix"] = f"prewarm_gid{gid}_seed{seed}"
        if self._mode == "img2img":
            # Init image drives canvas size; no EmptyLatentImage to patch.
            wf[self._node_ids["load_image"]]["inputs"]["image"] = init_image_name
        else:
            nid_lat = self._node_ids["empty_latent"]
            w, h = self._canvas_size
            wf[nid_lat]["inputs"]["width"] = w
            wf[nid_lat]["inputs"]["height"] = h
        return wf


def prewarm_all(
    gallery: Iterable[_IdentityLike],
    prewarm_gen,
    crop_cache: _CropCacheLike,
    stable_seed_fn: Callable[[int], int],
    kpl_init: dict | None = None,
) -> None:
    """Generate Reference Crops for every Identity not already cached.

    Threads a diversity accumulator across the gallery loop: each subsequent
    gid's best-of-K scorer sees the embeddings chosen earlier in the same
    pass, so Personas spread across face-embedding space (ADR-0008 addendum).
    Seeded at start-of-pass from on-disk Reference Crops so partial
    re-prewarms stay distant from already-cached Personas.

    PrewarmFailure propagates as-is — this is a startup hard-abort path.
    """
    existing: list[np.ndarray] = []
    for cached_gid in crop_cache.iter_global_ids():
        cached_crop = crop_cache.load(cached_gid)
        if cached_crop is None:
            continue
        emb = prewarm_gen.extract_embedding(cached_crop)
        if emb is not None:
            existing.append(emb)

    for identity in gallery:
        gid = identity.global_id
        if crop_cache.load(gid) is not None:
            continue
        stored = crop_cache.load_prompt(gid) if hasattr(crop_cache, "load_prompt") else None
        prompt = stored if stored is not None else identity.prompt
        init_image: np.ndarray | None = None
        kpl_face_embs: list[np.ndarray] | None = None
        if kpl_init is not None and gid in kpl_init:
            init_image, kpl_face_embs = kpl_init[gid]
        crop, emb = prewarm_gen.generate_best_of_k(
            global_id=gid,
            prompt=prompt,
            seed=stable_seed_fn(gid),
            existing_embeddings=existing,
            init_image=init_image,
            kpl_face_embs=kpl_face_embs,
        )
        crop_cache.save_first_generation(global_id=gid, crop_rgb=crop)
        if hasattr(crop_cache, "save_prompt"):
            crop_cache.save_prompt(global_id=gid, prompt=prompt)
        existing.append(emb)
        log.info("Reference Crop saved for gid=%d", gid)
