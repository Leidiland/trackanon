import logging
import time
import uuid
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from ta_src.video.videohandler import read_video, save_video, _StreamingVideoWriter
from ta_src.utils.visualization import overlay_all
from ta_src.pose.keypoint_smoother import KeypointSmoother
from ta_src.pose.vitpose_wrapper import ViTPoseWrapper
from ta_src.pipeline.frame_context import FrameContext
from ta_src.pipeline.op_edit import OpEditAbort, OpEditSession
from ta_src.pipeline.op_edit.web import OpEditWebDaemon
from ta_src.pipeline.stage_anonymization import AnonymizationStage
from ta_src.segmentation.sam3_wrapper import SAM3ChunkedStage, scale_rows_to_frame
from ta_src.tracking.identity_memory import IdentityGallery
from ta_src.tracking.identity_resolver import IdentityResolver
from ta_src.tracking.hungarian_assigner import HungarianAssigner
from ta_src.tracking.kpl_body_bbox import expand_face_to_body_bbox
from ta_src.tracking.kpl_seeder import KPLSeeder
from ta_src.tracking.faceid_wrapper import FaceIDWrapper
from ta_src.tracking.osnet_wrapper import OSNetWrapper
from ta_src.anonymization.confidence_gate import ConfidenceThresholds
from ta_src.anonymization.prompt_utils import CropCaptioner
from ta_src.anonymization.reference_crop_cache import ReferenceCropCache
from ta_src.video.sam3_prefetcher import RollingChunkPrefetcher

log = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _resolve_output_path(input_path: Path, output_path: Path) -> Path:
    """Compose output_path with input filename when output looks like a
    directory — otherwise vis writes lose the .mp4 suffix and PyAV cannot
    infer the container format."""
    if output_path.is_dir() or output_path.suffix.lower() not in _VIDEO_EXTS:
        return output_path / input_path.name
    return output_path


def _format_duration(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


class Pipeline:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._poser: ViTPoseWrapper | None = None
        self._kp_smoother: KeypointSmoother | None = None
        self._anonymizer: AnonymizationStage | None = None
        self._gallery: IdentityGallery | None = None
        self._crop_cache: ReferenceCropCache | None = None
        self._sam3_stage: SAM3ChunkedStage | None = None
        self._identity_resolver: IdentityResolver | None = None
        self._captioner: CropCaptioner | None = None
        self._kpl_seeds: list = []
        self._op_edit_session: OpEditSession | None = None
        self._op_edit_daemon: OpEditWebDaemon | None = None
        self._load_stages()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_stages(self):
        device = self.cfg.pipeline.device
        resolver_cfg = self.cfg.sam3.resolver

        if self.cfg.pipeline.run_pose:
            self._poser = ViTPoseWrapper.from_config(self.cfg.get("pose"), device)
            self._kp_smoother = KeypointSmoother.from_config(self.cfg.get("pose"))
            if self._kp_smoother is not None:
                log.info("KeypointSmoother enabled (1€ filter per global_id)")

        face: FaceIDWrapper | None = None
        osnet: OSNetWrapper | None = None
        if self.cfg.pipeline.run_tracking:
            face = FaceIDWrapper(
                min_face_width_px=int(resolver_cfg["min_face_width_px"]),
                min_face_det_score=float(resolver_cfg["min_face_det_score"]),
            )
            osnet = OSNetWrapper(
                device,
                tuple(resolver_cfg.reid_input_size),
                pretrained_reid_path=resolver_cfg.get("osnet_reid_weights"),
            )
            self._gallery = self._seed_gallery(face=face, osnet=osnet)
            self._identity_resolver = self._build_identity_resolver(
                face=face, osnet=osnet, gallery=self._gallery,
            )
            log.info(
                "IdentityResolver loaded (%d Identities seeded)",
                len(self._gallery.global_ids()),
            )
            if str(self.cfg.pipeline.get("trace_dir", "")).strip():
                self._identity_resolver.enable_trace()

        # SAM 3 always needed — KPL + resolver alone don't produce detections.
        self._sam3_stage = SAM3ChunkedStage.from_config(self.cfg.sam3, device)
        log.info("SAM3ChunkedStage loaded (prompt=%r, chunk_size=%d)",
                 str(self.cfg.sam3["prompt"]),
                 int(self.cfg.sam3["chunk_size"]))

        if self.cfg.pipeline.run_anonymization:
            crop_dir = str(resolver_cfg.get("reference_crop_dir",
                                            "data/gallery/reference_crops"))
            self._crop_cache = ReferenceCropCache(crop_dir)
            self._anonymizer = AnonymizationStage(
                self.cfg.anonymization, device,
                gallery=self._gallery, crop_cache=self._crop_cache,
                face=face, captioner=self._captioner,
            )
            log.info("Anonymization stage loaded (%s)", self.cfg.anonymization.name)
            if self._gallery is not None:
                self._anonymizer.prewarm_references(self._gallery, self._kpl_seeds)
            # Share the per-video ConfidenceLog so the resolver's warm-gallery
            # telemetry lands in the same file as the dispatch-time gate log.
            if self._identity_resolver is not None:
                self._identity_resolver.set_confidence_log(
                    self._anonymizer._confidence_log,
                )

        op_edit_cfg = self.cfg.pipeline.get("op_edit", {})
        if bool(op_edit_cfg.get("enabled", False)) and self._gallery is not None:
            resolver_cfg = self.cfg.sam3.resolver
            ui_mode = str(op_edit_cfg.get("ui", "web")).lower()
            if ui_mode == "web":
                web_cfg = op_edit_cfg.get("web", {})
                self._op_edit_daemon = OpEditWebDaemon()
                base_url = self._op_edit_daemon.serve_in_background(
                    host=str(web_cfg.get("host", "127.0.0.1")),
                    port=int(web_cfg.get("port", 8765)),
                    auto_open=bool(web_cfg.get("open_browser", True)),
                )
                log.info("OpEditWebDaemon serving on %s", base_url)
            self._op_edit_session = OpEditSession(
                gallery=self._gallery,
                kpl_root=Path(resolver_cfg.get("kpl_root", "data/known_persons")),
                artifact_dir=Path(op_edit_cfg.get("artifact_dir", "data/temp/op_edit")),
                reuse_existing=bool(op_edit_cfg.get("reuse_existing", False)),
                daemon=self._op_edit_daemon,
            )
            log.info("OpEditSession enabled (ui=%s)", ui_mode)

    def _build_identity_resolver(
        self, *, face: FaceIDWrapper, osnet: OSNetWrapper,
        gallery: IdentityGallery,
    ) -> IdentityResolver:
        resolver_cfg = self.cfg.sam3.resolver
        thresholds = ConfidenceThresholds(
            face_sim_floor=float(self.cfg.anonymization["confidence_face_threshold"]),
            osnet_abs_floor=float(self.cfg.anonymization["confidence_osnet_abs_threshold"]),
            osnet_margin_floor=float(self.cfg.anonymization["confidence_osnet_margin_threshold"]),
        )
        warm_cfg = resolver_cfg["warm_gallery"]
        assigner = HungarianAssigner(
            emb_weight=float(resolver_cfg["emb_weight"]),
            spatial_scale=float(resolver_cfg["spatial_scale"]),
            warm_match_floor=float(warm_cfg["warm_match_floor"]),
            face_quality_floor=float(resolver_cfg["face_quality_floor"]),
        )
        return IdentityResolver(
            face_wrapper=face,
            gallery=gallery,
            assigner=assigner,
            thresholds=thresholds,
            face_sampling_K=int(resolver_cfg["face_sampling_K"]),
            face_confirm_M=int(resolver_cfg["face_confirm_M"]),
            low_confidence_face_cosine=float(resolver_cfg["low_confidence_face_cosine"]),
            low_confidence_streak_max=int(resolver_cfg["low_confidence_streak_max"]),
            osnet_wrapper=osnet,
            warm_gallery_enabled=bool(warm_cfg["enabled"]),
            warm_face_size=int(warm_cfg["face_size"]),
            warm_osnet_size=int(warm_cfg["osnet_size"]),
            warm_anchor_count=int(warm_cfg["anchor_count"]),
            warm_dedup_cosine=float(warm_cfg["dedup_cosine"]),
            warm_osnet_trust_window_frames=int(warm_cfg["osnet_trust_window_frames"]),
            warm_mask_sanity_iou_floor=float(warm_cfg["mask_sanity_iou_floor"]),
            warm_mask_sanity_jump_px_frac=float(warm_cfg["mask_sanity_jump_px_frac"]),
            min_face_det_score=float(resolver_cfg["min_face_det_score"]),
            intra_chunk_revival_iou_floor=float(resolver_cfg["intra_chunk_revival_iou_floor"]),
            intra_chunk_revival_max_age_frames=int(resolver_cfg["intra_chunk_revival_max_age_frames"]),
            partial_carry_iou_floor=float(resolver_cfg["partial_carry_iou_floor"]),
            osnet_column_loss_margin=float(resolver_cfg["osnet_column_loss_margin"]),
            face_consistency_cos_floor=float(resolver_cfg["face_consistency_cos_floor"]),
            face_consistency_det_score_floor=float(resolver_cfg["face_consistency_det_score_floor"]),
        )

    def _seed_gallery(self, *, face: FaceIDWrapper, osnet: OSNetWrapper) -> IdentityGallery:
        # KPL photos are tight body crops; OSNet runs over an anthropometric
        # face-to-body bbox expansion without further segmentation.
        resolver_cfg = self.cfg.sam3.resolver
        kpl_root = Path(resolver_cfg.get("kpl_root", "data/known_persons"))
        captioner = CropCaptioner(device="cpu")
        # Reused downstream by PrewarmGenerator for CLIP prompt-fidelity scoring.
        self._captioner = captioner

        def _osnet_on_kpl_image(img):
            face_bbox = face.detect_face_bbox(img)
            h, w = img.shape[:2]
            body_bbox = expand_face_to_body_bbox(face_bbox, img_h=h, img_w=w)
            return osnet.extract(img, [list(body_bbox)])[0]

        seeder = KPLSeeder(
            kpl_root=str(kpl_root),
            face_extract=face.extract,
            osnet_extract=_osnet_on_kpl_image,
            captioner=captioner,
        )
        seeds = seeder.seed()
        gallery = IdentityGallery()
        gallery.seed_from_kpl(seeds)
        self._kpl_seeds = seeds
        return gallery

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        input_path = Path(self.cfg.paths.input)
        output_path = Path(self.cfg.paths.output)

        try:
            if input_path.is_dir():
                for child in sorted(input_path.iterdir()):
                    if not child.is_file():
                        continue
                    if child.suffix.lower() in _IMAGE_EXTS:
                        self._reject_image_input(child)
                    elif child.suffix.lower() in _VIDEO_EXTS:
                        self._run_video_file(child, output_path / child.name)
            elif input_path.suffix.lower() in _IMAGE_EXTS:
                self._reject_image_input(input_path)
            elif input_path.suffix.lower() in _VIDEO_EXTS:
                self._run_video_file(
                    input_path, _resolve_output_path(input_path, output_path),
                )
            else:
                log.error("Unsupported or missing input: %s", input_path)
        finally:
            # opencv-python-headless lacks GUI symbols; skip if no windows opened.
            if self.cfg.pipeline.visualize:
                cv2.destroyAllWindows()
            if self._op_edit_daemon is not None:
                self._op_edit_daemon.shutdown()

    # ------------------------------------------------------------------
    # File-level dispatch
    # ------------------------------------------------------------------

    def _reject_image_input(self, input_path: Path):
        raise RuntimeError(
            f"Single-image input is not supported ({input_path.name}): SAM 3 "
            "is a video-only model. Wrap the image as a 1-frame video before "
            "passing it to the pipeline."
        )

    def _run_video_file(self, input_path: Path, output_path: Path):
        log.info("Processing video: %s", input_path.name)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with read_video(
            str(input_path),
            start_time=self.cfg.temporal.start_time,
            end_time=self.cfg.temporal.end_time,
            fps=self.cfg.temporal.fps,
        ) as reader:
            info = reader.info

        save_vis = self.cfg.pipeline.save_visualization
        do_vis = self.cfg.pipeline.visualize
        write_main = bool(self.cfg.pipeline.run_anonymization)
        stopped = False

        # Stream vis frames to a background encoder; avoids buffering in RAM.
        vis_writer: _StreamingVideoWriter | None = None
        if save_vis:
            vis_path = output_path.with_stem(output_path.stem + "_vis")
            vis_writer = _StreamingVideoWriter(
                str(vis_path), fps=info.fps, width=info.width, height=info.height
            )

        progress = tqdm(total=info.frame_count or None, unit="frame",
                        desc=input_path.name, disable=False)

        def _stream():
            nonlocal stopped
            for frame, ctx in self.process_video(
                str(input_path),
                start_time=self.cfg.temporal.start_time,
                end_time=self.cfg.temporal.end_time,
                fps=self.cfg.temporal.fps,
            ):
                if do_vis or save_vis:
                    vis = overlay_all(frame, ctx)
                    if do_vis:
                        cv2.imshow("result", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                        if cv2.waitKey(1) == ord("q"):
                            stopped = True
                            return
                    if save_vis and vis_writer is not None:
                        vis_writer.push(vis)
                progress.update(1)
                yield ctx.frame

        start = time.monotonic()
        try:
            if write_main:
                save_video(_stream(), str(output_path), fps=info.fps,
                           width=info.width, height=info.height)
            else:
                # Vis-only: drain the generator; ctx.frame is the unmodified input.
                for _ in _stream():
                    pass
        except OpEditAbort as exc:
            log.warning("op-edit abort: %s — no output written for %s",
                        exc, input_path.name)
            return
        finally:
            progress.close()
            if vis_writer is not None:
                vis_writer.close()
            if self._anonymizer is not None:
                self._anonymizer.cleanup_artifacts()
            trace_dir = str(self.cfg.pipeline.get("trace_dir", "")).strip()
            if trace_dir and self._identity_resolver is not None:
                td = Path(trace_dir)
                td.mkdir(parents=True, exist_ok=True)
                stem = input_path.stem
                try:
                    self._identity_resolver.dump_trace(
                        td / f"{stem}.tracking.jsonl",
                        td / f"{stem}.tracking.summary.txt",
                    )
                    log.info("Wrote resolver trace to %s/%s.tracking.jsonl",
                             td, stem)
                except RuntimeError:
                    pass  # trace was never enabled — non-fatal

        elapsed = time.monotonic() - start
        log.info("completed in %s", _format_duration(elapsed))

    # ------------------------------------------------------------------
    # Core processing — SAM 3 per-chunk dispatch
    # ------------------------------------------------------------------

    def process_video(
        self,
        path: str,
        start_time: float | None = None,
        end_time: float | None = None,
        fps: int | None = None,
    ) -> Iterator[tuple[np.ndarray, FrameContext]]:
        # Per chunk: SAM 3 runs solo, then releases the GPU to ComfyUI.
        # Resolver.update runs on every frame (including L overlap frames for
        # re-binding evidence); pose/anonymizer/yield skip already-emitted overlap frames.
        if self._anonymizer:
            self._anonymizer.reset(video_name=Path(path).stem)
        if self._kp_smoother is not None:
            self._kp_smoother.reset_all()

        if self._sam3_stage is None or self._identity_resolver is None:
            raise RuntimeError(
                "process_video requires SAM3ChunkedStage and IdentityResolver — "
                "neither is loaded. Enable run_tracking and rebuild the Pipeline."
            )

        chunk_size = int(self.cfg.sam3["chunk_size"])
        overlap_L = int(self.cfg.sam3["overlap_L"])
        temp_root = Path(self.cfg.paths.get("temp", "data/temp"))
        run_dir = temp_root / "sam3_chunks" / f"{Path(path).stem}_{uuid.uuid4().hex[:8]}"

        last_yielded_idx = -1
        with read_video(path, start_time=start_time, end_time=end_time, fps=fps) as reader:
            with RollingChunkPrefetcher(
                frame_source=reader,
                run_dir=run_dir,
                chunk_size=chunk_size,
                overlap_L=overlap_L,
            ) as prefetcher:
                for chunk_id, chunk in enumerate(prefetcher):
                    # Drop ComfyUI's ~5 GB resident inpaint stack before SAM 3
                    # propagates. Skip on chunk 0: nothing's loaded yet.
                    if chunk_id > 0 and self._anonymizer is not None:
                        self._anonymizer.free_comfyui_vram()
                    self._identity_resolver.start_chunk(chunk_id)
                    sam3_outputs = self._sam3_stage.process_chunk(
                        chunk.jpeg_dir, chunk.indices[0],
                    )
                    # Free SAM 3's session (~9 GB peak) before ComfyUI claims VRAM.
                    self._sam3_stage.close_session_and_empty_cache()

                    n = len(chunk.indices)
                    for i in range(n):
                        frame_idx = chunk.indices[i]
                        frame_rgb = chunk.frames[i]
                        sam3_rows = sam3_outputs[i]
                        # Release chunk refs so frame buffers can be reclaimed.
                        chunk.frames[i] = None
                        sam3_outputs[i] = None

                        # SAM 3 emits at output_mask_resolution; upscale to frame coords.
                        scale_rows_to_frame(sam3_rows, frame_rgb.shape[:2])

                        # Resolver runs on overlap frames too, so carryover rebind
                        # sees the new chunk's sam3_obj_ids before snapshots drop.
                        ctx = FrameContext(frame=frame_rgb)
                        ctx.detections = sam3_rows
                        enriched = self._identity_resolver.update(
                            frame_rgb, sam3_rows, frame_idx,
                        )
                        if (
                            self._op_edit_session is not None
                            and frame_idx == 0
                            and self.cfg.pipeline.op_edit.get("frame_0", True)
                        ):
                            mapping = self._op_edit_session.prompt(
                                frame_rgb, enriched, frame_idx,
                                video_name=Path(path).stem,
                            )
                            enriched = self._identity_resolver.apply_operator_overrides(
                                mapping, sam3_rows,
                            )
                        ctx.tracks = enriched
                        ctx.detections = enriched

                        if frame_idx <= last_yielded_idx:
                            # Overlap frame: already emitted in the previous chunk.
                            continue

                        if self._poser is not None and ctx.detections:
                            ctx.keypoints = self._poser.run(frame_rgb, ctx.detections)
                            if self._kp_smoother is not None:
                                ctx.keypoints = [
                                    self._kp_smoother.smooth(
                                        int(det.get("global_id", -1)), kp, frame_idx,
                                    )
                                    for det, kp in zip(ctx.detections, ctx.keypoints)
                                ]

                        if self._anonymizer is not None:
                            ctx.frame = self._anonymizer.run(frame_rgb, ctx, frame_idx)

                        last_yielded_idx = frame_idx
                        yield frame_rgb, ctx
