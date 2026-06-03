# trackanon

Offline video anonymization pipeline: every person in a video is detected, identified against a Known Person Library, and replaced with a diffusion-generated fictional appearance.

## Language

### Identification layer

**Person**:
A physical individual visible in the source video.
_Avoid_: subject, target, detection

**Identity**:
The persistent record for one Known Person, keyed by `global_id`. Holds `face_centroid`, `appearance_centroid`, Persona prompt, and Reference Crop — all frozen at startup (KPL-seeded centroids and prompt; Reference Crop loaded from disk). All N Identities (N = number of KPL sub-folders) exist from startup; none are created or destroyed during a run. Per-detection-chain state (running embedding means, current gid binding) lives on `TrackBinding`, not on Identity.
_Avoid_: track, tracklet, person record, gallery entry

**TrackBinding**:
The per-`sam3_obj_id` entity owned by `IdentityResolver`. Holds quality-weighted running means of face and OSNet embeddings, the accumulated observation counts (`n_face_observations`, `n_osnet_observations`), the current `global_id` binding, the low-confidence (face) streak, the OSNet column-loss streak (sparse-KPL safety), and `blocked_gids` (per-chunk set of gids the binding was demoted from — prevents the demote → re-confirm-to-same-wrong-gid cycle). Keyed by `(chunk_id, sam3_obj_id)` and re-created each chunk. At chunk boundaries every active binding snapshots its full state; the next chunk's first-frame bindings inherit a snapshot via bbox-IoU match, so a confirmed identity carries straight across the boundary and unconfirmed evidence accumulates over the person's whole on-screen lifetime. Bindings bind to Identities; they do not allocate `global_id`s.
_Avoid_: track-id (the binding evolves), tracklet

**Known Person Library**:
A filesystem folder of real-world photo sets for Persons whose appearance is known before processing begins. Root path: `data/known_persons/`. Each sub-folder corresponds to one Person (folder name is for human use only; the Identity is keyed by `global_id` assigned by sorted folder order). Used at startup by the KPL Seeder to initialise each Identity's face centroid, appearance centroid, and Persona prompt. The real photos never enter the anonymisation layer — the Reference Crop remains always synthetic.
_Avoid_: identity gallery (reserved for the runtime data structure `IdentityGallery`), face database, reference gallery

**KPL Seeder**:
The startup module that scans `data/known_persons/`, extracts InsightFace face embeddings and OSNet appearance embeddings from each person's photos, averages them into centroids, and generates a Persona prompt by captioning one representative image. Runs on every startup. Raises a startup error if `data/known_persons/` is missing or empty (closed-world architecture requires KPL), or if any person's photos yield no detectable face — photo curation is the operator's responsibility.

**Hungarian Assigner**:
Solver that builds a cost matrix over `(detections × candidate Identities)` and runs the Hungarian algorithm to find the optimal assignment. Cost detail (face vs OSNet, spatial term, low-confidence warning) is documented under "Identification architecture" below. The assigner runs at *track confirmation time* — only over unconfirmed `TrackBinding`s that have reached M face / OSNet observations, against free (still-unassigned) Identities. The per-track-confirmation cadence and the chunk-boundary carryover replace a per-frame Hungarian.

**Coverage policy.** Every detection must be anonymized; assignment is at-most-one detection per Identity per frame. Hungarian runs against all N Identities, never duplicates a candidate, never allocates a new `global_id`. When `N_det > N` (closed-world violated), the worst-cost excess detections route to the Fallback Anonymizer.

Generation is further gated on **per-video confidence-confirmation**: a gid is VACE-rendered only when at least one of its cached `AssignmentInfo` entries clears the face or OSNet thresholds; otherwise every silhouette of that gid inherits the Fallback Anonymizer blur. Within a rendered Window the greyed silhouette region is regenerated and the rest of the frame passes through unchanged; a per-frame **mask-quality gate** (`mask_floor_ratio` / `mask_score_floor`, with a `motion_guard` that carries the last good mask onto degenerate frames) drops frames that fail it to fallback-blur rather than painting a bad silhouette. The carry-over leak path is closed: any `cached_silhouette \ painted_region` is blurred.

**Chunk**:
A contiguous N-frame window of source video (default N=60) that bounds one SAM 3 inference session. SAM 3's `start_session → propagate_in_video → close_session` runs per chunk, so per-frame state cost stays bounded regardless of total video length. Chunks overlap by `overlap_L=4` frames so the first frame of chunk K+1 is the last frame of chunk K, giving `IdentityResolver`'s carryover step near-identical bboxes to match across the boundary.

**Sam3FrameWorkspace**:
A temp JPEG directory holding the decoded frames of one chunk. `SAM3ChunkedStage.start_session(resource_path=...)` reads from this directory. Owned by `RollingChunkPrefetcher`; created when chunk K+1's prefetch begins, deleted after chunk K's processing finishes. Disk usage is bounded to ~3 chunks at any time, independent of video length.

**RollingChunkPrefetcher**:
The background thread that materialises chunk K+1's `Sam3FrameWorkspace` while the main thread is processing chunk K. Exposes an iterator of `Chunk` records (`indices`, `frames`, `jpeg_dir`). Frees each chunk's workspace as iteration advances; on exception or `close()` it drains pending workspaces. Bounded by a 2-slot queue (≤ 2 prefetched + 1 consumer-held = 3 on disk).

**IdentityResolver**:
The identification stage. Public surface: `start_chunk(chunk_id)`, `update(frame_rgb, sam3_rows, frame_idx)`. Per call: materialises or reuses a `TrackBinding` per `sam3_obj_id`, accumulates face + OSNet evidence at K-frame cadence, fires a confirmation Hungarian (gated by the per-video confidence gate) only when an unconfirmed binding reaches M observations, and enriches each row with `global_id` + `assignment_info`. In Pass 1 these enriched rows are written to the track cache for the Pass-2 anonymization stage to read back.

**FaceID Wrapper**:
Thin wrapper around InsightFace `FaceAnalysis(buffalo_l)`. Takes a person body crop (BGR ndarray), returns a 512-dim L2-normalised face embedding or `None` if no face is detected above the minimum crop width (default 40px). InsightFace is a hard dependency — the pipeline aborts at startup if `buffalo_l` cannot be loaded. Per-frame face-not-detected results legitimately return `None` and are handled by the cost matrix's no-face mode.

### Anonymization layer

**Persona**:
The fictional replacement appearance owned by one Identity — the (prompt, seed, Reference Crop) tuple that drives generation. Same Identity → same Persona across all frames and across runs. The prompt is captioned at startup from one KPL image via CropCaptioner (a per-gid `reference_crops/<global_id>.prompt.txt`, when present, overrides it); the seed is fixed; the Reference Crop is a pre-generated synthetic image loaded from disk. All three are frozen for a run.
_Avoid_: avatar, synthetic appearance, replacement

**Reference Crop**:
The synthetic image of a Persona, persisted at `data/gallery/reference_crops/<global_id>.png` and loaded at startup. It is the appearance anchor every VACE Window is conditioned on. The Reference Crop is a pre-generated asset — the pipeline no longer synthesises one at startup; curating a synthetic, leak-safe crop per Persona is an offline/operator step (the SD1.5 img2img prewarm/First-Generation path was retired).
_Avoid_: latent_z, appearance embedding, latent

**Backend (Wan-VACE)**:
Anonymisation runs on a windowed, reference-anchored video generator (Wan2.1-VACE) on its own portable ComfyUI (port 8190), in two passes:
- **Pass 1** — SAM 3 + IdentityResolver track the whole clip into the per-frame track cache (ComfyUI not resident).
- **Pass 2** — per confidence-passing gid: plan **Windows** over the gid's presence, build a VACE bundle (control video + reference + mask), render, **Stitch** the windows back into the frame, then blur any cached silhouette the render didn't paint.

The two backends never co-reside; SAM 3 owns the GPU for Pass 1, VACE for Pass 2.

**Window**:
A contiguous run of `window_len` frames (must be `4n+1` for the VAE's temporal compression; default 49) of one gid's on-screen presence — the unit VACE generates in one call. Adjacent windows of the same gid overlap by `overlap` frames and blend; with `preserve_overlap`, a later window's overlap band is pinned to the prior window's stitched pixels so VACE's temporal attention smooths the boundary identity shift a linear blend can't hide. Presence is gap-split: an absence longer than `max_bridge` frames starts a new window so a jump-cut isn't generated as one continuous clip.
_Avoid_: clip, segment (reserved for whole-video terms)

**Control recipe (grey-control)**:
Per window, VACE's control video is the source crop with the gid's silhouette greyed out (neutral grey reads as "no content"), so appearance is driven by the Reference Crop rather than the original subject's pixels. The `pose` `control_mode` additionally draws the subject's skeleton inside the greyed hole so the Persona adopts the original pose and fully fills the silhouette; it falls back to plain grey-control on frames with no usable body joints.
_Avoid_: inpaint mask (collides with the SAM 3 mask / Stitch boundary vocabulary)

**Stitch**:
Compositing a rendered window back into the full frame: the painted Persona is feathered onto the scene at the silhouette boundary (`feather_px`), and reference-anchored colour correction (`color_anchor`) pulls each window's chroma toward the Persona crop to stop the cross-window VAE round-trip drift (gradual darkening/over-saturation per boundary). Frames or gids the render skipped fall to the Fallback Anonymizer.
_Avoid_: paste-back (SD1.5-era term)

## Relationships

- Each **Identity** corresponds to exactly one **Person** (closed-world: all Persons are in the Known Person Library).
- Each **Identity** owns exactly one **Persona**.
- **TrackBinding** binds detections to Identities; a Person on screen across multiple chunks is represented by a sequence of `(chunk_id, sam3_obj_id)`-keyed bindings, each inheriting the previous chunk's snapshot via bbox-IoU carryover.

## Example dialogue

> **Dev:** "When a Person re-enters the frame after occlusion, do we create a new Persona?"
> **Domain expert:** "No — the Identity is always present. The Hungarian Assigner re-matches the detection to the same Identity via face recognition. The Reference Crop is already set; the re-entry just becomes another Window VACE renders for that gid in Pass 2."

> **Dev:** "What if ComfyUI is down on frame 50 but was up for frames 1–49?"
> **Domain expert:** "That can't happen in a valid run — ComfyUI is checked at startup and the pipeline aborts if it's unreachable. There's no mid-run fallback."

> **Dev:** "What if a Person is not in the Known Person Library?"
> **Domain expert:** "The Fallback Anonymizer handles it. Hungarian only assigns at-most-one detection per Identity, so excess bodies (when `N_det > N`) are routed to Gaussian-blur-over-mask anonymisation instead of diffusion. The unrecognised Person is anonymised but with a visibly distinct (blurred) treatment rather than silently wearing someone else's Persona."

**Fallback Anonymizer**:
Non-generative anonymization path. Applies Gaussian blur over the SAM 3 segmentation mask region; falls back to a blurred bounding-box rectangle when no mask is available. Lives in its own module; the VACE Pass-2 stage calls it for excess detections (closed-world violated: `N_det > N`), for unconfirmed gids, and for any cached silhouette a window didn't paint. Has no Persona, no Reference Crop, and no ComfyUI dependency.

## Tooling

**Silhouette Test**:
A diagnostic script (`dev/silhouette_test_sam3.py`) that runs the chunked SAM 3 stage and pose but skips identification and diffusion. Each Person is painted as a solid-black mask region (SAM 3 mask, or bbox rectangle when no mask is available) with the per-detection `sam3_obj_id` overlaid in colour. Purpose: eyeball within-chunk obj_id stability and mask quality without standing up ComfyUI, KPL, or InsightFace.

## Identification architecture

**Identification-first**: detections are assigned directly to Known Persons via the Hungarian Assigner. No open-world Track allocation. The sam3 backend runs this on a per-chunk → per-frame cadence.

Per-chunk sequence (`SAM3ChunkedStage` + `IdentityResolver`):

1. `RollingChunkPrefetcher` materialises a 60-frame chunk's `Sam3FrameWorkspace` on a background thread.
2. SAM 3 starts a session over the chunk's JPEG dir, prompts `text="person"`, propagates through the chunk, emits per-frame `(sam3_obj_id, mask, bbox)` rows.
3. For each frame in order: `IdentityResolver.update(...)` materialises a `TrackBinding` per `sam3_obj_id` (or inherits one from a cross-chunk carryover snapshot at the boundary), samples face + OSNet at K-frame cadence into the binding's running sums, and fires a confirmation Hungarian only over unconfirmed bindings that have reached M observations. Matches that don't clear the ConfidenceGate stay `gid=-1`.
4. The enriched rows are written to the track cache (Pass 1). In Pass 2 the gid-level confidence gate decides which gids VACE renders; unconfirmed gids and silhouettes no window painted route to Fallback.
5. SAM 3's within-chunk propagation gives masks that are temporally stable across the chunk by construction — no per-frame mask propagation step.

Enriched rows carry `global_id`, `track_id`, `assignment_info`, `mask_source`, `prompt` — the contract the Pass-2 anonymization stage and overlay tooling consume.

## Known Person Library — seeding rules

KPL is required: the pipeline aborts at startup if `data/known_persons/` is missing or empty. At each startup:

1. For each person sub-folder (sorted alphabetically, 1-indexed `global_id`): InsightFace face embeddings are extracted from every image (only those that yield a detectable face contribute), averaged, and L2-renormalised into `face_centroid`. OSNet embeddings are extracted from every image (regardless of face detectability — a back-facing photo still gives a useful appearance signal), averaged, and L2-renormalised into the appearance centroid. The KPL Seeder raises an error at startup if any person's photos yield no detectable face, and on empty sub-folders.
2. The Persona prompt is (re-)generated by captioning the alphabetically-first KPL image whose face was successfully detected (CropCaptioner, fed a whole-image mask) and stored on `IdentityRecord.prompt`. Re-runs every startup; CropCaptioner output is deterministic on a given input image.
3. Reference Crops are loaded from `data/gallery/reference_crops/<global_id>.png` and assigned to their Identity; a per-gid `<global_id>.prompt.txt`, when present, overrides the captioned Persona prompt. The pipeline does not synthesise Reference Crops at startup — they are pre-generated synthetic assets (the SD1.5 img2img prewarm / First-Generation path was retired). A missing crop for a target gid means that gid cannot be VACE-rendered and its silhouettes fall to fallback-blur.
4. `global_id` is assigned by sorted sub-folder order and is stable across runs as long as folder names don't change.
5. Appearance and face centroids are frozen at KPL-seeded values. No EMA drift from in-video observations.

## Flagged ambiguities

- "persona" was used informally in comments for the synthesized appearance — resolved: **Persona** is the canonical term, defined as a child of Identity.
- `latent_z` in code implies a spatial SD latent — resolved: the concept is a **Reference Crop**; `latent_z` is removed.
- "track" / "Track" — resolved: `TrackBinding` is the per-`sam3_obj_id` entity that accumulates evidence and binds to an Identity. It is **not** an open-world Track — it does not allocate `global_id`s, and it does not persist beyond its chunk's lifetime (state crosses chunk boundaries via a per-binding snapshot, not via a long-lived object).
