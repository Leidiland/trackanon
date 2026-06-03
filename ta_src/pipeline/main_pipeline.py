import hashlib
import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Iterator


def _cache_path_is_ephemeral(cache_dir: Path, temp_root: Path) -> bool:
    """Wipe-on-exit predicate. True when the cache lives under temp_root —
    the unconditional-wipe default; False when it was routed elsewhere
    (e.g. outputs/eval/cache/<scene>) and should survive for downstream
    analysis. Unresolvable paths default to False so a misconfig never
    deletes the wrong tree."""
    try:
        return cache_dir.resolve().is_relative_to(temp_root.resolve())
    except (ValueError, OSError):
        return False


def _cache_is_populated(cache_dir: Path) -> bool:
    """True when cache_dir holds at least one frame_*.npz file — the marker
    that Pass 1 completed at least one chunk. Used by _run_vace_file to skip
    Pass 1 when the user pre-computed it locally and pointed track_cache_dir
    at the result (load mode); false → write mode (Pass 1 runs fresh)."""
    if not cache_dir.exists():
        return False
    return any(cache_dir.glob("frame_*.npz"))


# vace_cfg keys that don't affect rendered window pixels — excluded from the
# resume hash so a box move (host/port), worker-count change, or timeout tweak
# still reuses already-rendered windows.
_VACE_WORKDIR_IGNORE_KEYS = frozenset({
    "comfyui_host", "comfyui_port", "comfyui_input_dir", "comfyui_output_dir",
    "pool", "render_concurrency", "timeout", "poll_interval",
})


def _vace_work_dirname(src_key: str, vace_cfg) -> str:
    """Deterministic name for the VACE stitch work dir so a re-run resumes
    completed windows instead of re-rendering from scratch. Keyed on the decode
    identity (src_key = clip+window+fps) plus a hash of the render-affecting
    config — a prompt/seed/lora/knob change mints a new dir (no stale-window
    reuse), while a host/port/pool/timeout change reuses the same one."""
    try:
        from omegaconf import DictConfig, OmegaConf
        plain = (OmegaConf.to_container(vace_cfg, resolve=True)
                 if isinstance(vace_cfg, DictConfig) else dict(vace_cfg))
    except Exception:
        plain = dict(vace_cfg)
    render = {k: v for k, v in plain.items() if k not in _VACE_WORKDIR_IGNORE_KEYS}
    blob = json.dumps(render, sort_keys=True, default=str)
    return f"{src_key}_{hashlib.sha1(blob.encode()).hexdigest()[:8]}"

import cv2
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from ta_src.video.videohandler import read_video, save_video, _StreamingVideoWriter
from ta_src.utils.visualization import overlay_all
from ta_src.utils.frame_store import DiskFrameProvider
from typing import TYPE_CHECKING

from ta_src.pose.factory import build_poser
from ta_src.pose.keypoint_smoother import KeypointSmoother

if TYPE_CHECKING:
    from ta_src.pose.dwpose_wrapper import DWposeWrapper
from ta_src.pipeline import host_memory
from ta_src.pipeline.frame_context import FrameContext
from ta_src.pipeline.op_edit import NewUnknownTrigger, OpEditAbort, OpEditSession
from ta_src.pipeline.op_edit.web import OpEditWebDaemon
from ta_src.pipeline.track_cache import TrackCacheReader, TrackCacheWriter
from ta_src.segmentation.sam3_wrapper import SAM3ChunkedStage, scale_rows_to_frame
from ta_src.tracking.identity_memory import IdentityGallery
from ta_src.tracking.identity_resolver import IdentityResolver
from ta_src.tracking.hungarian_assigner import HungarianAssigner
from ta_src.tracking.kpl_body_bbox import expand_face_to_body_bbox
from ta_src.tracking.kpl_centroid_cache import KPLCentroidCache
from ta_src.tracking.kpl_seeder import KPLSeeder
from ta_src.tracking.faceid_wrapper import FaceIDWrapper
from ta_src.tracking.osnet_wrapper import OSNetWrapper
from ta_src.anonymization.confidence_gate import ConfidenceThresholds
from ta_src.anonymization.confidence_log import ConfidenceLog
from ta_src.anonymization.prompt_utils import CropCaptioner
from ta_src.video.sam3_frame_workspace import Sam3FrameWorkspace
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


def _sam3_kpl_bbox_fn(*, sam3: SAM3ChunkedStage, tmp_root: Path, face: FaceIDWrapper):
    """Return a callable that runs SAM 3 on a single RGB image and returns
    the highest-score person bbox (in image coordinates). Each call spins a
    fresh session and workspace; on miss (no detection), returns None so the
    caller can fall back to the anthropometric face→body expansion.

    The face wrapper is only used to break ties when SAM 3 emits multiple
    detections — picks the one whose bbox best contains a face."""
    counter = {"i": 0}
    tmp_root.mkdir(parents=True, exist_ok=True)

    def _bbox(image_rgb: np.ndarray) -> tuple[float, float, float, float] | None:
        chunk_id = counter["i"]
        counter["i"] += 1
        ws = Sam3FrameWorkspace(parent_dir=tmp_root, chunk_id=chunk_id)
        try:
            ws.write_frames([image_rgb])
            rows_per_frame = sam3.process_chunk(ws.path, 0)
        finally:
            sam3.close_session_and_empty_cache()
            ws.close()
        if not rows_per_frame or not rows_per_frame[0]:
            return None
        rows = rows_per_frame[0]
        scale_rows_to_frame(rows, image_rgb.shape[:2])
        # Multiple detections: prefer the one whose bbox contains the
        # InsightFace-detected face; otherwise highest score.
        face_bbox = face.detect_face_bbox(image_rgb)
        if face_bbox is not None and len(rows) > 1:
            for row in sorted(
                rows, key=lambda r: float(r.get("score", 0.0)), reverse=True,
            ):
                if _bbox_contains(row["bbox"], face_bbox):
                    return tuple(float(v) for v in row["bbox"])
        best = max(rows, key=lambda r: float(r.get("score", 0.0)))
        return tuple(float(v) for v in best["bbox"])

    return _bbox


def _bbox_contains(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
) -> bool:
    return (
        outer[0] <= inner[0] and outer[1] <= inner[1]
        and outer[2] >= inner[2] and outer[3] >= inner[3]
    )


class Pipeline:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        # DWpose pose stage, constructed by build_poser.
        self._poser: "DWposeWrapper | None" = None
        self._kp_smoother: KeypointSmoother | None = None
        self._vace_stage = None
        self._gallery: IdentityGallery | None = None
        self._face: FaceIDWrapper | None = None
        self._sam3_stage: SAM3ChunkedStage | None = None
        self._identity_resolver: IdentityResolver | None = None
        self._kpl_seeds: list = []
        self._op_edit_session: OpEditSession | None = None
        self._op_edit_daemon: OpEditWebDaemon | None = None
        self._new_unknown_trigger: NewUnknownTrigger | None = None
        # Owned only in tracking-only runs; with anonymization on the resolver
        # shares the anonymizer's ConfidenceLog instead.
        self._resolver_confidence_log: ConfidenceLog | None = None
        # Two-pass track cache for the in-flight video; removed in _run_video_file.
        self._active_cache_dir: Path | None = None
        self._load_stages()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_stages(self):
        device = self.cfg.pipeline.device
        resolver_cfg = self.cfg.sam3.resolver

        if self.cfg.pipeline.run_pose:
            self._poser = build_poser(self.cfg.get("pose"), device)
            pose_cfg = self.cfg.get("pose")
            self._kp_smoother = KeypointSmoother.from_config(
                pose_cfg.get("smoothing") if pose_cfg is not None else None
            )
            if self._kp_smoother is not None:
                log.info("KeypointSmoother enabled (one_euro)")

        # SAM 3 loads before KPL seeding so the seeder can derive each KPL
        # photo's body bbox from a real SAM 3 mask envelope — matching the
        # runtime OSNet crop distribution. Centroids are cached on disk so
        # this cost is paid only when KPL photos change.
        self._sam3_stage = SAM3ChunkedStage.from_config(self.cfg.sam3, device)
        log.info("SAM3ChunkedStage loaded (prompt=%r, chunk_size=%d)",
                 str(self.cfg.sam3["prompt"]),
                 int(self.cfg.sam3["chunk_size"]))

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
            self._face = face
            self._gallery = self._seed_gallery(
                face=face, osnet=osnet, sam3=self._sam3_stage,
            )
            self._identity_resolver = self._build_identity_resolver(
                face=face, osnet=osnet, gallery=self._gallery,
            )
            log.info(
                "IdentityResolver loaded (%d Identities seeded)",
                len(self._gallery.global_ids()),
            )
            if str(self.cfg.pipeline.get("trace_dir", "")).strip():
                self._identity_resolver.enable_trace()

        if self.cfg.pipeline.run_anonymization:
            self._load_vace_stage()

        # The resolver's rebind ConfidenceLog: VACE never owns one (its render is
        # windowed, decoupled from the per-detection gate), so the resolver writes
        # its own here in every run, capturing warm-gallery rebind diagnostics
        # alongside the trace (confidence_log_dir overrides; else trace_dir).
        if self._identity_resolver is not None:
            clog_dir = (
                str(self.cfg.pipeline.get("confidence_log_dir", "")).strip()
                or str(self.cfg.pipeline.get("trace_dir", "")).strip()
                or None
            )
            self._resolver_confidence_log = ConfidenceLog(clog_dir)
            self._identity_resolver.set_confidence_log(self._resolver_confidence_log)

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
            if bool(op_edit_cfg.get("on_new_unknown", False)):
                nu_cfg = op_edit_cfg.get("new_unknown", {})
                self._new_unknown_trigger = NewUnknownTrigger(
                    sustained_frames=int(nu_cfg.get("sustained_frames", 10)),
                    dedup_centroid_frac=float(nu_cfg.get("dedup_centroid_frac", 0.06)),
                    dedup_recent_frames=int(nu_cfg.get("dedup_recent_frames", 50)),
                    min_det_score=float(nu_cfg.get("min_det_score", 0.5)),
                )
                log.info("OpEdit new-unknown mid-clip trigger enabled")

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
            osnet_confirm_warm_only=bool(warm_cfg["osnet_confirm_warm_only"]),
            osnet_confirm_min_warm=int(warm_cfg["osnet_confirm_min_warm"]),
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
            osnet_column_loss_streak_max=int(resolver_cfg["osnet_column_loss_streak_max"]),
            face_consistency_cos_floor=float(resolver_cfg["face_consistency_cos_floor"]),
            face_consistency_det_score_floor=float(resolver_cfg["face_consistency_det_score_floor"]),
            locality_max_jump_px=float(resolver_cfg.get("locality_max_jump_px", 300.0)),
            locality_max_stale_frames=int(resolver_cfg.get("locality_max_stale_frames", 5)),
            locality_max_speed_px=float(resolver_cfg.get("locality_max_speed_px", 100.0)),
        )

    def _seed_gallery(
        self,
        *,
        face: FaceIDWrapper,
        osnet: OSNetWrapper,
        sam3: SAM3ChunkedStage,
    ) -> IdentityGallery:
        # OSNet's KPL bbox is derived from SAM 3 (matches runtime distribution)
        # with a fallback to anthropometric face→body expansion when SAM 3
        # finds no person in the photo.
        resolver_cfg = self.cfg.sam3.resolver
        kpl_root = Path(resolver_cfg.get("kpl_root", "data/known_persons"))
        captioner = CropCaptioner(device="cpu")

        cache_dir = str(
            resolver_cfg.get("kpl_centroid_cache_dir", "data/gallery/kpl_centroids"),
        )
        centroid_cache = KPLCentroidCache(Path(cache_dir))

        sam3_tmp_root = Path(
            self.cfg.paths.get("temp", "data/temp"),
        ) / "sam3_kpl"
        bbox_fn = _sam3_kpl_bbox_fn(sam3=sam3, tmp_root=sam3_tmp_root, face=face)

        def _osnet_on_kpl_image(img):
            h, w = img.shape[:2]
            body_bbox = bbox_fn(img)
            if body_bbox is None:
                face_bbox = face.detect_face_bbox(img)
                body_bbox = expand_face_to_body_bbox(face_bbox, img_h=h, img_w=w)
            return osnet.extract(img, [list(body_bbox)])[0]

        seeder = KPLSeeder(
            kpl_root=str(kpl_root),
            face_extract=face.extract,
            osnet_extract=_osnet_on_kpl_image,
            captioner=captioner,
            centroid_cache=centroid_cache,
        )
        seeds = seeder.seed()
        gallery = IdentityGallery()
        gallery.seed_from_kpl(seeds)
        self._kpl_seeds = seeds
        return gallery

    def _load_vace_stage(self):
        from ta_src.pipeline.stage_vace_anonymization import VaceAnonymizationStage

        vace_cfg = self.cfg.anonymization.vace
        self._vace_stage = VaceAnonymizationStage(vace_cfg)
        # ComfyUI unreachable is a hard abort — partial anonymization is the
        # worst failure mode for a privacy tool.
        if not self._vace_stage.is_ready():
            raise RuntimeError(
                f"VACE ComfyUI not reachable at {vace_cfg['comfyui_host']}:"
                f"{vace_cfg['comfyui_port']} — start the portable VACE server "
                f"(external/comfyui-vace/run_server.sh)."
            )
        log.info("VACE backend loaded (target gids=%s, %s:%s)",
                 list(vace_cfg.get("target_gids", [])) or "auto",
                 vace_cfg["comfyui_host"], vace_cfg["comfyui_port"])

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

        # VACE backend is windowed, not per-frame, so it does not fit the
        # streaming save_video loop — it runs Pass 1 to cache then generates.
        if self._vace_stage is not None:
            self._run_vace_file(input_path, output_path)
            return

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
            for frame, ctx in self._select_frame_source(input_path):
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
            if self._resolver_confidence_log is not None:
                self._resolver_confidence_log.close()
            # The two-pass track cache is transient (~GBs of per-frame npz); drop
            # it whether the run completed or aborted so it never leaks. The next
            # video resets _active_cache_dir in _select_frame_source. Eval-routed
            # caches (track_cache_dir set to outputs/eval/cache/...) sit outside
            # temp_root and survive — that's the preservation surface.
            if self._active_cache_dir is not None:
                temp_root = Path(self.cfg.paths.get("temp", "data/temp"))
                if _cache_path_is_ephemeral(self._active_cache_dir, temp_root):
                    shutil.rmtree(self._active_cache_dir, ignore_errors=True)

        elapsed = time.monotonic() - start
        log.info("completed in %s", _format_duration(elapsed))

    def _run_vace_file(self, input_path: Path, output_path: Path):
        # Pass 1 (track -> cache) then windowed VACE generation. SAM 3 and the
        # VACE ComfyUI never co-reside; the artifact is a crop-space clip.
        # When track_cache_dir points at a pre-populated cache (e.g. one
        # produced earlier locally and reused on a remote VACE run), Pass 1
        # is skipped — the reader reads what's already there.
        start = self.cfg.temporal.start_time
        end = self.cfg.temporal.end_time
        fps = self.cfg.temporal.fps
        temp_root = Path(self.cfg.paths.get("temp", "data/temp"))
        base = f"{input_path.stem}_{uuid.uuid4().hex[:8]}"
        run_dir = temp_root / "sam3_chunks" / base
        cache_dir = self._resolve_cache_dir(base)
        self._active_cache_dir = cache_dir
        try:
            if _cache_is_populated(cache_dir):
                log.info(
                    "VACE: reusing pre-populated cache at %s (skipping Pass 1)",
                    cache_dir,
                )
                # Pass 2 reads the cache off disk and renders through the remote
                # VACE server — the local track models (SAM 3, resolver, FaceID,
                # OSNet, gallery) are dead weight from here, and at ~12GB resident
                # they dominate the local OOM budget. Drop them.
                self._free_track_models()
            else:
                # Hand the GPU to SAM 3 for the whole tracking pass — evict the
                # VACE server's resident weights up front so they don't co-reside.
                self._vace_stage.free_comfyui_vram()
                writer = TrackCacheWriter(cache_dir)
                for _ in self._track_pass(
                    str(input_path), start, end, fps, run_dir,
                    cache_writer=writer,
                ):
                    pass
            reader = TrackCacheReader(cache_dir)
            # Cache frame_idx == decode enumeration index. Decode the source once
            # to a disk-backed provider so the whole 4K clip never sits in RAM —
            # the stage reads frames by index on demand and streams its output.
            # Keyed on (clip, window, fps) so a re-run reuses the decode instead
            # of paying the ~25MB/frame 4K decode again.
            src_key = f"{input_path.stem}_{start}_{end}_{fps}"
            src_cache = DiskFrameProvider(temp_root / "vace_src" / src_key)
            n_frames = src_cache.existing_count()
            if n_frames > 0:
                log.info("VACE: reusing decoded source cache (%d frames) at %s",
                         n_frames, temp_root / "vace_src" / src_key)
            else:
                with read_video(str(input_path), start_time=start, end_time=end, fps=fps) as r:
                    n_frames = src_cache.populate(r.frames())
            # Cover the full decoded clip range; in-window frames are stitched,
            # out-of-window frames pass through raw source. Stage layers all
            # target gids (explicit list or auto-enumerated from the cache).
            # Deterministic, config-aware work dir: keyed on decode identity +
            # render-config hash, so a killed/restarted run resumes its already-
            # rendered windows (the StitchStore skips done windows on disk).
            vace_work = temp_root / "vace" / _vace_work_dirname(
                src_key, self.cfg.anonymization.vace)
            res = self._vace_stage.run(
                reader, frames_provider=src_cache,
                frame_indices=list(range(n_frames)),
                work_dir=vace_work,
                out_path=output_path.parent / f"{input_path.stem}_vace.mp4",
            )
            if res is not None:
                log.info("VACE stitched output written: %s", res)
            else:
                log.warning("VACE produced no output (no target gids in cache)")
        finally:
            # Keep the VACE stitch work dir (deterministic, config-keyed) so a
            # re-run resumes its rendered windows; keep the decoded source cache
            # (keyed on clip+window) so a re-run skips the 4K re-decode.
            if self._active_cache_dir is not None:
                if _cache_path_is_ephemeral(self._active_cache_dir, temp_root):
                    shutil.rmtree(self._active_cache_dir, ignore_errors=True)

    def _free_track_models(self) -> None:
        """Release the Pass-1 track models and return their heap to the OS.
        Only safe once the cache is populated and no further Pass 1 will run
        (VACE Pass 2 never touches them); the resolver trace is already on disk."""
        self._sam3_stage = None
        self._identity_resolver = None
        self._face = None
        self._gallery = None
        self._kpl_seeds = []
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        host_memory.trim()

    def _resolve_cache_dir(self, default_base: str) -> Path:
        # pipeline.track_cache_dir, when set, routes the per-frame cache to an
        # explicit path (eval runs do this to preserve it for downstream
        # analysis; survives the wipe-on-exit predicate). Empty falls back to
        # the ephemeral default under data/temp/track_cache/<uuid>.
        configured = str(self.cfg.pipeline.get("track_cache_dir", "")).strip()
        if configured:
            return Path(configured)
        temp_root = Path(self.cfg.paths.get("temp", "data/temp"))
        return temp_root / "track_cache" / default_base

    def _select_frame_source(
        self, input_path: Path,
    ) -> Iterator[tuple[np.ndarray, FrameContext]]:
        # Reached only by the streaming loop, which runs for tracking-only / vis
        # passes (anonymisation goes through the windowed VACE path in
        # _run_vace_file). Pass 1 alone: the cache is normally skipped (no Pass 2
        # reader), but eval runs opt in via track_cache_dir to keep the per-frame
        # npz for downstream analysis.
        start_time = self.cfg.temporal.start_time
        end_time = self.cfg.temporal.end_time
        fps = self.cfg.temporal.fps
        temp_root = Path(self.cfg.paths.get("temp", "data/temp"))
        base = f"{input_path.stem}_{uuid.uuid4().hex[:8]}"
        run_dir = temp_root / "sam3_chunks" / base
        self._active_cache_dir = None
        configured_cache = str(self.cfg.pipeline.get("track_cache_dir", "")).strip()
        cache_writer = None
        if configured_cache:
            self._active_cache_dir = Path(configured_cache)
            cache_writer = TrackCacheWriter(self._active_cache_dir)
        return self._track_only_stream(
            str(input_path), start_time, end_time, fps, run_dir,
            cache_writer=cache_writer,
        )

    def _track_only_stream(
        self,
        path: str,
        start_time: float | None,
        end_time: float | None,
        fps: int | None,
        run_dir: Path,
        cache_writer: TrackCacheWriter | None = None,
    ) -> Iterator[tuple[np.ndarray, FrameContext]]:
        # Pass 1 alone: yield the unmodified frame and its enriched ctx for vis.
        # cache_writer is non-None only when the eval path opted in via
        # pipeline.track_cache_dir.
        for _frame_idx, frame_rgb, ctx in self._track_pass(
            path, start_time, end_time, fps, run_dir,
            cache_writer=cache_writer,
        ):
            yield frame_rgb, ctx

    def track_video(
        self,
        path: str,
        start_time: float | None = None,
        end_time: float | None = None,
        fps: int | None = None,
    ) -> Iterator[tuple[np.ndarray, FrameContext]]:
        # Drive the tracking pass over a clip, yielding (frame_rgb, ctx) per
        # output frame. Tracking-only (no anonymisation) — for eval/debug drains.
        temp_root = Path(self.cfg.paths.get("temp", "data/temp"))
        run_dir = temp_root / "sam3_chunks" / f"{Path(path).stem}_{uuid.uuid4().hex[:8]}"
        for _frame_idx, frame_rgb, ctx in self._track_pass(
            path, start_time, end_time, fps, run_dir,
        ):
            yield frame_rgb, ctx

    # ------------------------------------------------------------------
    # Core processing — SAM 3 per-chunk dispatch
    # ------------------------------------------------------------------

    def _track_pass(
        self,
        path: str,
        start_time: float | None,
        end_time: float | None,
        fps: int | None,
        run_dir: Path,
        cache_writer=None,
    ) -> Iterator[tuple[int, np.ndarray, FrameContext]]:
        # Chunked SAM 3 -> resolver -> pose -> smoothing, one emitted output frame
        # at a time. ComfyUI is never run here (the windowed VACE render is a
        # separate Pass 2); each emitted frame's enriched tracks are written to
        # cache_writer when one is supplied.
        if self._sam3_stage is None or self._identity_resolver is None:
            raise RuntimeError(
                "Track pass requires SAM3ChunkedStage and IdentityResolver — "
                "neither is loaded. Enable run_tracking and rebuild the Pipeline."
            )
        # Resolver is reused across every video in a batch; clear per-video
        # state so chunk snapshots and per-track history don't bleed across.
        self._identity_resolver.reset_video()
        if self._resolver_confidence_log is not None:
            self._resolver_confidence_log.open(Path(path).stem)

        chunk_size = int(self.cfg.sam3["chunk_size"])
        overlap_L = int(self.cfg.sam3["overlap_L"])

        last_yielded_idx = -1
        with read_video(path, start_time=start_time, end_time=end_time, fps=fps) as reader:
            with RollingChunkPrefetcher(
                frame_source=reader,
                run_dir=run_dir,
                chunk_size=chunk_size,
                overlap_L=overlap_L,
            ) as prefetcher:
                for chunk_id, chunk in enumerate(prefetcher):
                    # Trim the pipeline's own host RSS before each SAM 3 peak (skip
                    # chunk 0 — nothing accumulated yet): it can't restart itself,
                    # so hand freed pages back to the OS.
                    if chunk_id > 0:
                        if bool(self.cfg.pipeline.get("trim_host_memory_per_chunk", True)):
                            host_memory.trim()
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

                        # SAM 3 emits boxes/masks in its processing resolution; upscale to frame coords.
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

                        # Mid-clip op-edit: pause the first time a genuinely new
                        # unidentified person settles in frame (after the overlap
                        # guard so each real frame counts once).
                        if self._new_unknown_trigger is not None and frame_idx > 0:
                            to_prompt = self._new_unknown_trigger.update(
                                frame_idx, enriched, frame_rgb.shape[:2],
                            )
                            if to_prompt:
                                mapping = self._op_edit_session.prompt(
                                    frame_rgb, enriched, frame_idx,
                                    video_name=Path(path).stem,
                                )
                                enriched = self._identity_resolver.apply_operator_overrides(
                                    mapping, sam3_rows,
                                )
                                ctx.tracks = enriched
                                ctx.detections = enriched

                        if self._poser is not None and ctx.detections:
                            ctx.keypoints = self._poser.run(frame_rgb, ctx.detections)
                            if self._kp_smoother is not None:
                                self._kp_smoother.apply(
                                    ctx.keypoints, ctx.detections, frame_idx,
                                )

                        if cache_writer is not None:
                            cache_writer.write(frame_idx, ctx.tracks, ctx.keypoints)

                        last_yielded_idx = frame_idx
                        yield frame_idx, frame_rgb, ctx
