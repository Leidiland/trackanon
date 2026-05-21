import json
import logging
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig

from ta_src.anonymization.confidence_gate import (
    ConfidenceThresholds,
    evaluate as evaluate_confidence,
)
from ta_src.anonymization.confidence_log import ConfidenceLog
from ta_src.anonymization.fallback_anonymizer import FallbackAnonymizer
from ta_src.anonymization.mask_quality import (
    FailNone,
    Pass,
    mask_quality_check,
)
from ta_src.anonymization.prewarm_generator import (
    PrewarmGenerator,
    prewarm_all,
)
from ta_src.pipeline.frame_context import FrameContext

log = logging.getLogger(__name__)


def _name_filter_allows_inpaint(name: str | None, target_names: list[str]) -> bool:
    # Empty target list = today's default (no per-name filtering).
    if not target_names:
        return True
    return bool(name) and name in target_names


def _mask_area_ratio(mask, bbox) -> float | None:
    # Same definition as mask_quality.mask_quality_check uses internally;
    # surfaced here for the dispatch log so the operator sees what the gate saw.
    if mask is None or bbox is None:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
        bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
        if bbox_area == 0:
            return None
        mask_area = int(mask.sum()) if hasattr(mask, "sum") else 0
        return mask_area / bbox_area
    except (TypeError, ValueError):
        return None


def _dispatch_log(frame_idx, gid, route, *, mask, bbox, mask_score, mask_source, name,
                  decision_kind=None):
    if not log.isEnabledFor(logging.DEBUG):
        return
    ratio = _mask_area_ratio(mask, bbox)
    log.debug(
        "dispatch frame=%d gid=%s route=%s name=%s mask_present=%s "
        "mask_area_ratio=%s mask_score=%s mask_source=%s reason=%s",
        frame_idx, gid, route, name, mask is not None,
        f"{ratio:.3f}" if ratio is not None else "None",
        f"{float(mask_score):.3f}" if mask_score is not None else "None",
        mask_source, decision_kind,
    )


class AnonymizationStage:
    def __init__(
        self,
        cfg: DictConfig,
        device: str,
        gallery=None,
        crop_cache=None,
        face=None,
        captioner=None,
    ):
        self.cfg = cfg
        self.device = device
        self._client = None
        self._pipeline = None
        self._gallery = gallery
        self._crop_cache = crop_cache
        self._face = face
        self._captioner = captioner
        self._prewarm_gen: PrewarmGenerator | None = None
        self._fallback = FallbackAnonymizer(
            kernel_size=int(cfg.get("fallback_kernel_size", 51)),
            sigma=float(cfg.get("fallback_sigma", 0.0)),
        )
        # Mask-quality dispatch floors. score_pass_override rescues a sub-ratio
        # detection mask when SAM 3's mask_score is high — defends crowded
        # scenes where the visible portion of an occluded person produces a
        # narrow-but-valid mask.
        self._mask_floor_ratio = float(cfg["mask_floor_ratio"])
        self._mask_score_floor = float(cfg["mask_score_floor"])
        spo = cfg.get("mask_score_pass_override", None)
        self._mask_score_pass_override: float | None = (
            float(spo) if spo not in (None, "", "null") else None
        )
        # Per-video confidence-gated dispatch state.
        self._gid_confirmed_in_run: set[int] = set()
        self._confidence_thr = ConfidenceThresholds(
            face_sim_floor=float(cfg["confidence_face_threshold"]),
            osnet_abs_floor=float(cfg["confidence_osnet_abs_threshold"]),
            osnet_margin_floor=float(cfg["confidence_osnet_margin_threshold"]),
        )
        log_dir = str(cfg.get("confidence_log_dir", "")).strip()
        self._confidence_log = ConfidenceLog(log_dir if log_dir else None)
        # Empty list = no per-name filtering (default). Non-empty restricts
        # diffusion dispatch to confirmed tracks whose name matches.
        self._inpaint_target_names: list[str] = list(
            cfg.get("inpaint_target_names", []) or []
        )
        self._load()

    def reset(self, video_name: str | None = None):
        # Confirmation is per-video — clear and rotate the log.
        self._gid_confirmed_in_run.clear()
        self._confidence_log.close()
        if video_name:
            self._confidence_log.open(video_name)
        if self._pipeline is not None:
            self._pipeline.reset()

    def free_comfyui_vram(self) -> bool:
        # Free ComfyUI's resident weights before SAM 3's next chunk runs.
        if self._client is None:
            return False
        return self._client.free_vram()

    def cleanup_artifacts(self) -> None:
        # End-of-run sweep of ComfyUI's input/output dirs. Per-call PNGs
        # (crop/mask/pose/ref uploads + anon_/prewarm_/spike_ saves) are
        # unused after fetch_image returns.
        if self._client is None:
            return
        if not bool(self.cfg.get("cleanup_comfyui_artifacts", True)):
            return
        input_prefixes = ["crop_", "mask_", "pose_", "ref_"]
        output_prefixes = ["anon_", "spike_"]
        if bool(self.cfg.get("cleanup_comfyui_prewarm", True)):
            output_prefixes.append("prewarm_")
        n_in, n_out = self._client.cleanup_artifacts(
            input_prefixes=input_prefixes,
            output_prefixes=output_prefixes,
        )
        log.info("Cleaned ComfyUI artifacts: %d input, %d output", n_in, n_out)

    def prewarm_references(self, gallery, kpl_seeds=None) -> None:
        # Synthesise Reference Crops. When the prewarm workflow is img2img,
        # kpl_seeds supplies the per-gid init image and KPL face centroid
        # used by the anti-leak gate. txt2img mode ignores both.
        if self._prewarm_gen is None or self._crop_cache is None:
            return
        from ta_src.anonymization.diffusion_pipeline import _stable_seed
        kpl_init: dict | None = None
        if kpl_seeds:
            kpl_init = {}
            for seed in kpl_seeds:
                bgr = cv2.imread(str(seed.representative_path))
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                kpl_init[seed.global_id] = (rgb, seed.face_embeddings)
        prewarm_all(gallery, self._prewarm_gen, self._crop_cache,
                    stable_seed_fn=_stable_seed, kpl_init=kpl_init)

    def _load(self):
        from ta_src.anonymization.comfyui_client import ComfyUIClient
        from ta_src.anonymization.diffusion_pipeline import DiffusionPipeline

        host = str(self.cfg["comfyui_host"])
        port = int(self.cfg["comfyui_port"])
        client = ComfyUIClient(host=host, port=port)

        # ComfyUI unreachable is a hard abort — partial anonymization is the
        # worst failure mode for a privacy tool.
        if not client.is_alive():
            raise RuntimeError(
                f"ComfyUI not reachable at {host}:{port} — start it via "
                f"`.venv/bin/python external/comfyui/main.py --listen {host} --port {port}` "
                f"or disable run_anonymization in the pipeline config."
            )

        self._pipeline = DiffusionPipeline(
            self.cfg, client,
            crop_cache=self._crop_cache,
            face_id_wrapper=self._face,
        )

        if self._face is not None:
            self._prewarm_gen = self._build_prewarm_generator(client)

        self._client = client
        log.info("AnonymizationStage ready (ComfyUI at %s:%d)", host, port)

    def _build_prewarm_generator(self, client) -> PrewarmGenerator:
        wf_path = str(self.cfg.get(
            "workflow_prewarm",
            self.cfg.get(
                "workflow_txt2img",
                "configs/anonymization/workflows/img2img_default.json",
            ),
        )).strip()
        p = Path(wf_path)
        if not p.exists():
            raise FileNotFoundError(
                f"PrewarmGenerator: txt2img workflow not found at '{p}'. "
                f"Set anonymization.workflow_txt2img in diffusion.yaml."
            )
        workflow = json.loads(p.read_text())
        node_ids = workflow.get("_node_ids", {})
        canvas = (
            int(self.cfg["crop_target_short"]),
            int(self.cfg["crop_max_long"]),
        )
        debug_dir = str(self.cfg.get("prewarm_debug_dir", "")).strip() or None
        return PrewarmGenerator(
            comfy_client=client,
            workflow=workflow,
            node_ids=node_ids,
            faceid_wrapper=self._face,
            max_candidates=int(self.cfg["prewarm_candidates"]),
            canvas_size=canvas,
            debug_dir=debug_dir,
            captioner=self._captioner,
            clip_drop_quartile=float(self.cfg.get("prewarm_clip_drop_quartile", 0.25)),
            kpl_leak_threshold=float(self.cfg["prewarm_kpl_leak_threshold"]),
        )

    def run(self, image_rgb: np.ndarray, ctx: FrameContext, frame_idx: int = 0) -> np.ndarray:
        tracks = ctx.tracks
        keypoints_list = ctx.keypoints

        # Copy so fallback.apply (which writes in-place) cannot mutate the
        # caller's frame — the vis renderer overlays on that same buffer and
        # must stay un-anonymized.
        output = image_rgb.copy()

        # Fallback-path mutations run sequentially first; eligible tracks are
        # collected and handed to inpaint_frame as one batched call so HTTP
        # submit/poll latency can overlap with ComfyUI's serial GPU work.
        inpaint_tracks: list[dict] = []
        inpaint_kps: list = []

        for i, track in enumerate(tracks):
            gid = track.get("global_id", -1)
            mask = track.get("mask")
            bbox = track.get("bbox")
            mask_score = track.get("mask_score")
            mask_source = track.get("mask_source", "detection")

            name = track.get("name")
            log_kw = dict(mask=mask, bbox=bbox, mask_score=mask_score,
                          mask_source=mask_source, name=name)

            if gid == -1:
                if mask is not None:
                    self._fallback.apply(output, mask)
                elif bbox is not None:
                    self._fallback.apply(output, bbox)
                _dispatch_log(frame_idx, gid, "fallback-unbound", **log_kw)
                continue

            # Mask-quality gate: failures route to Fallback (bbox blur), not skip.
            decision = mask_quality_check(
                mask=mask,
                mask_score=mask_score,
                bbox=bbox if bbox is not None else (0, 0, 0, 0),
                mask_source=mask_source,
                ratio_floor=self._mask_floor_ratio,
                score_floor=self._mask_score_floor,
                score_pass_override=self._mask_score_pass_override,
            )
            if not isinstance(decision, Pass):
                if isinstance(decision, FailNone) and bbox is None:
                    _dispatch_log(frame_idx, gid, "skip-no-mask-no-bbox",
                                  decision_kind="FailNone", **log_kw)
                    continue  # nothing to anonymize over
                if mask is not None:
                    self._fallback.apply(output, mask)
                elif bbox is not None:
                    self._fallback.apply(output, bbox)
                _dispatch_log(frame_idx, gid, "fallback-mask-quality",
                              decision_kind=type(decision).__name__, **log_kw)
                continue

            if self._pipeline is None:
                if mask is not None:
                    self._fallback.apply(output, mask)
                elif bbox is not None:
                    self._fallback.apply(output, bbox)
                _dispatch_log(frame_idx, gid, "fallback-no-pipeline", **log_kw)
                continue

            # Confidence-gated dispatch: a gid stays in Fallback until the
            # gate fires once; afterwards all matches go to diffusion.
            if not self._is_confirmed_for_dispatch(gid, track, frame_idx):
                if mask is not None:
                    self._fallback.apply(output, mask)
                elif bbox is not None:
                    self._fallback.apply(output, bbox)
                _dispatch_log(frame_idx, gid, "fallback-not-confirmed", **log_kw)
                continue

            # Per-name dispatch filter: confirmed tracks outside the target
            # list fall to blur, never to diffusion.
            if not _name_filter_allows_inpaint(
                track.get("name"), self._inpaint_target_names
            ):
                if mask is not None:
                    self._fallback.apply(output, mask)
                elif bbox is not None:
                    self._fallback.apply(output, bbox)
                _dispatch_log(frame_idx, gid, "fallback-name-filter", **log_kw)
                continue

            kp = (
                keypoints_list[i].get("keypoints")
                if i < len(keypoints_list)
                else None
            )

            # Reference-crop prompt overrides the resolver's gallery prompt:
            # IPAdapter conditions on the synthetic Reference Crop, so the
            # text prompt must describe that crop, not the original KPL person.
            if self._crop_cache is not None and hasattr(self._crop_cache, "load_prompt"):
                ref_prompt = self._crop_cache.load_prompt(gid)
                if ref_prompt:
                    track["prompt"] = ref_prompt

            inpaint_tracks.append(track)
            inpaint_kps.append(kp)
            _dispatch_log(frame_idx, gid, "diffusion-eligible", **log_kw)

        if inpaint_tracks and self._pipeline is not None:
            output = self._pipeline.inpaint_frame(
                output, inpaint_tracks, inpaint_kps, frame_idx,
            )

        return output

    def _is_confirmed_for_dispatch(
        self, gid: int, track: dict, frame_idx: int
    ) -> bool:
        if gid in self._gid_confirmed_in_run:
            return True
        # Operator-pinned bindings are confirmed by definition; the live face/
        # OSNet cosine gate would still reject them on profile / occluded frames.
        if track.get("operator_assigned"):
            self._gid_confirmed_in_run.add(gid)
            return True
        info = track.get("assignment_info")
        if info is None:
            return False
        decision = evaluate_confidence(info, self._confidence_thr)
        self._confidence_log.log(
            gid=gid,
            assigned_sim=info.assigned_sim,
            second_best_sim=info.second_best_sim,
            confirmed=decision.confirmed,
            confirmed_via=decision.confirmed_via,
            cost_path=info.cost_path,
            frame_idx=frame_idx,
        )
        if decision.confirmed:
            self._gid_confirmed_in_run.add(gid)
            return True
        return False
