"""Pure helpers for the face-polish validation script (ADR-0019).

Cross-frame face-cosine std, per-frame Persona-cosine stats, per-gid
aggregation, cross-config diff, and operator-readable text formatting.
No I/O, no pipeline state; tests/manual/measure_face_polish.py composes
these with the live FaceIDWrapper.
"""
from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import numpy as np

_GEN_PNG_RE = re.compile(r"^f(\d+)_gid(\d+)_gen\.png$")


def collect_emb_rows_from_crops_dir(
    *,
    crops_dir: str | Path,
    face_id_wrapper,
    n_per_gid: int = 10,
) -> list[dict]:
    """Walk `crops_dir` for files named f{frame:06d}_gid{gid:04d}_gen.png
    (DiffusionPipeline.save_crops_dir convention), sample N evenly-spaced
    regen frames per gid, run the face_id_wrapper on each, and build
    JSONL-shaped rows. Rows where the wrapper returns None are recorded
    with `status='no_face'` rather than dropped — the validation-summary
    layer needs the failure count for diagnosability."""
    import cv2

    crops_dir = Path(crops_dir)
    per_gid: dict[int, list[tuple[int, Path]]] = {}
    for p in crops_dir.iterdir():
        m = _GEN_PNG_RE.match(p.name)
        if not m:
            continue
        frame_idx = int(m.group(1))
        gid = int(m.group(2))
        per_gid.setdefault(gid, []).append((frame_idx, p))
    for gid in per_gid:
        per_gid[gid].sort()  # ascending by frame_idx

    rows: list[dict] = []
    for gid in sorted(per_gid):
        frames = per_gid[gid]
        chosen = sample_n_evenly(frames, n_per_gid)
        for frame_idx, path in chosen:
            bgr = cv2.imread(str(path))
            if bgr is None:
                rows.append({"gid": gid, "frame_idx": frame_idx,
                             "status": "unreadable", "face_emb": None})
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            emb = face_id_wrapper.extract(rgb)
            if emb is None:
                rows.append({"gid": gid, "frame_idx": frame_idx,
                             "status": "no_face", "face_emb": None})
            else:
                rows.append({"gid": gid, "frame_idx": frame_idx,
                             "status": "ok",
                             "face_emb": np.asarray(emb, dtype=np.float32).tolist()})
    return rows


def format_summary_text(
    *,
    summaries_off: dict[int, dict],
    summaries_on: dict[int, dict],
    diffs: list[dict],
) -> str:
    """Operator-readable text summary. One section per config + a diff
    section keyed by gid; designed for `cat` not `jq`."""
    lines: list[str] = []
    lines.append("# face_polish.enabled = false")
    lines.extend(_summary_lines(summaries_off))
    lines.append("")
    lines.append("# face_polish.enabled = true")
    lines.extend(_summary_lines(summaries_on))
    lines.append("")
    lines.append("# diff (on - off)")
    for d in diffs:
        lines.append(
            f"gid={d['gid']:<3} variance_pct_change={d['variance_pct_change']:+.1f}%  "
            f"persona_mean_change={d['persona_mean_change']:+.3f}  "
            f"persona_min off={d['persona_min_off']:.3f} on={d['persona_min_on']:.3f}"
        )
    return "\n".join(lines) + "\n"


def _summary_lines(summaries: dict[int, dict]) -> list[str]:
    lines = []
    for gid in sorted(summaries):
        s = summaries[gid]
        lines.append(
            f"gid={gid:<3} std={s['cross_frame_std']:.3f}  "
            f"persona_mean={s['persona_mean']:.3f}  "
            f"min={s['persona_min']:.3f} max={s['persona_max']:.3f}  "
            f"n_ok={s['n_ok']} n_failed={s['n_failed']}"
        )
    return lines


def diff_summaries(
    summaries_off: dict[int, dict],
    summaries_on: dict[int, dict],
) -> list[dict]:
    """Per-gid cross-config diff for the slice-6 gate read. Negative
    variance_pct_change = polish reduced drift. Positive persona_mean_change
    = polish tightened identity toward Persona. Gids missing from either
    side are dropped (no diff to compute)."""
    out = []
    for gid in sorted(set(summaries_off) & set(summaries_on)):
        off = summaries_off[gid]
        on = summaries_on[gid]
        std_off = float(off["cross_frame_std"])
        std_on = float(on["cross_frame_std"])
        pct = ((std_on - std_off) / std_off * 100.0) if std_off > 0 else float("nan")
        out.append({
            "gid": gid,
            "variance_pct_change": pct,
            "persona_mean_change": float(on["persona_mean"]) - float(off["persona_mean"]),
            "persona_min_off": float(off["persona_min"]),
            "persona_min_on": float(on["persona_min"]),
        })
    return out


def summarize_gid(
    rows: list[dict],
    *,
    persona_centroid: np.ndarray,
) -> dict:
    """Per-gid aggregation. `rows` are JSONL records for a single gid+config
    with a `status` field. Only status=='ok' rows contribute to the metrics;
    others are surfaced as `n_failed` so callers can see fall-through volume."""
    gid = rows[0]["gid"] if rows else -1
    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("face_emb") is not None]
    embs = [np.asarray(r["face_emb"], dtype=np.float32) for r in ok_rows]
    persona = persona_cosine_stats(embs, persona_centroid)
    return {
        "gid": gid,
        "n_ok": len(ok_rows),
        "n_failed": len(rows) - len(ok_rows),
        "cross_frame_std": cross_frame_cosine_std(embs),
        "persona_mean": persona["mean"],
        "persona_min": persona["min"],
        "persona_max": persona["max"],
    }


def sample_n_evenly(items: list, n: int) -> list:
    """Pick N evenly-spaced items from `items`, including the first and last
    when N≥2. Returns all items when fewer than N are available; returns [] for
    an empty input. Used to sample regen frames across the clip's full span."""
    if not items or n <= 0:
        return []
    if len(items) <= n:
        return list(items)
    if n == 1:
        return [items[0]]
    step = (len(items) - 1) / (n - 1)
    return [items[int(round(i * step))] for i in range(n)]


def persona_cosine_stats(
    embeddings: list[np.ndarray],
    persona_centroid: np.ndarray,
) -> dict[str, float]:
    """Per-frame cosine of each synthesized face embedding to the Persona
    Reference Crop's centroid. Returns mean / min / max. An empty embedding
    list returns NaN for all three so callers can detect 'no samples'."""
    if not embeddings:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    c = np.asarray(persona_centroid, dtype=np.float64)
    c_norm = float(np.linalg.norm(c)) or 1.0
    sims = []
    for e in embeddings:
        e64 = np.asarray(e, dtype=np.float64)
        e_norm = float(np.linalg.norm(e64)) or 1.0
        sims.append(float(np.dot(e64, c) / (e_norm * c_norm)))
    return {
        "mean": float(np.mean(sims)),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
    }


def cross_frame_cosine_std(embeddings: list[np.ndarray]) -> float:
    """Std of pairwise cosine similarities across the embedding sequence.

    Lower = more consistent identity across the sampled regen frames.
    Returns 0.0 when fewer than two embeddings are supplied (no pair to score).
    """
    embs = [np.asarray(e, dtype=np.float64) for e in embeddings]
    if len(embs) < 2:
        return 0.0
    sims = [
        float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        for a, b in combinations(embs, 2)
    ]
    return float(np.std(sims))
