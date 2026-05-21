"""Materialises RGB frame chunks as JPEG temp dirs for SAM 3's start_session."""
from __future__ import annotations

import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ta_src.video.sam3_frame_workspace import Sam3FrameWorkspace


_SENTINEL = object()
_PRODUCER_TIMEOUT = 0.05  # seconds — keeps the producer responsive to close()
# Grace window for sibling dirs without a .pid file: covers the race where a
# concurrent pipeline run just mkdir'd its run_dir but hasn't written .pid yet.
_PIDLESS_SIBLING_GRACE_SECONDS = 60.0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # PID exists but is owned by another user — treat as alive.
        return True
    except OSError:
        return False
    return True


@dataclass(frozen=True)
class Chunk:
    indices: list[int]
    frames: list[np.ndarray]
    jpeg_dir: Path


class RollingChunkPrefetcher:
    def __init__(
        self,
        frame_source,
        run_dir: Path,
        chunk_size: int = 60,
        overlap_L: int = 4,
    ) -> None:
        self._frame_source = frame_source
        self._run_dir = Path(run_dir)
        self._chunk_size = chunk_size
        self._overlap_L = overlap_L
        self._sweep_stale_run_dirs(self._run_dir.parent)
        self._run_dir.mkdir(parents=True, exist_ok=False)
        # Pidfile lets a future run identify this dir as live (vs. orphaned
        # by a hard kill / OOM where close() never ran).
        (self._run_dir / ".pid").write_text(str(os.getpid()))

        # Queue size 1 — 4K raw frames at 1.4 GB/chunk pin too much RAM under WSL2.
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._iter_started = False
        self._workspaces: dict[Path, Sam3FrameWorkspace] = {}
        self._workspaces_lock = threading.Lock()

    def __iter__(self) -> Iterator[Chunk]:
        if not self._iter_started:
            self._iter_started = True
            self._thread = threading.Thread(target=self._producer, daemon=True)
            self._thread.start()

        prev_path: Path | None = None
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, tuple) and item and item[0] == "ERR":
                raise item[1]
            chunk: Chunk = item
            if prev_path is not None:
                self._close_workspace(prev_path)
            prev_path = chunk.jpeg_dir
            yield chunk
        if prev_path is not None:
            self._close_workspace(prev_path)

    def _producer(self) -> None:
        try:
            buf_frames: list[np.ndarray] = []
            buf_indices: list[int] = []
            chunk_id = 0
            L = self._overlap_L
            for global_idx, frame in enumerate(self._frame_source.frames()):
                if self._stop.is_set():
                    return
                buf_frames.append(frame)
                buf_indices.append(global_idx)
                if len(buf_frames) >= self._chunk_size:
                    if not self._enqueue(chunk_id, buf_indices, buf_frames):
                        return
                    chunk_id += 1
                    if L > 0:
                        buf_frames = buf_frames[-L:]
                        buf_indices = buf_indices[-L:]
                    else:
                        buf_frames = []
                        buf_indices = []
            # Tail chunk: skip if it would only re-emit the overlap carry-over
            # (i.e. ≤ L frames remaining after a full-chunk emit).
            if buf_frames and (chunk_id == 0 or len(buf_frames) > L):
                self._enqueue(chunk_id, buf_indices, buf_frames)
        except Exception as e:  # noqa: BLE001 — surface to consumer
            while not self._stop.is_set():
                try:
                    self._queue.put(("ERR", e), timeout=_PRODUCER_TIMEOUT)
                    break
                except queue.Full:
                    continue
        finally:
            # Must block until the consumer makes room — a 1-second timeout
            # was dropping the sentinel whenever a slow consumer hadn't
            # drained chunks 0/1 yet, leaving the consumer hung on get().
            while not self._stop.is_set():
                try:
                    self._queue.put(_SENTINEL, timeout=_PRODUCER_TIMEOUT)
                    break
                except queue.Full:
                    continue

    def _enqueue(self, chunk_id: int, indices: list[int],
                 frames: list[np.ndarray]) -> bool:
        ws = Sam3FrameWorkspace(parent_dir=self._run_dir, chunk_id=chunk_id)
        ws.write_frames(frames)
        with self._workspaces_lock:
            self._workspaces[ws.path] = ws
        chunk = Chunk(indices=list(indices), frames=list(frames), jpeg_dir=ws.path)
        while not self._stop.is_set():
            try:
                self._queue.put(chunk, timeout=_PRODUCER_TIMEOUT)
                return True
            except queue.Full:
                continue
        return False

    @staticmethod
    def _sweep_stale_run_dirs(parent: Path) -> None:
        if not parent.exists():
            return
        now = time.time()
        for sibling in parent.iterdir():
            if not sibling.is_dir():
                continue
            pid_file = sibling / ".pid"
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                except (OSError, ValueError):
                    pid = None
                if pid is not None and _pid_alive(pid):
                    continue
            else:
                try:
                    age = now - sibling.stat().st_mtime
                except OSError:
                    continue
                if age < _PIDLESS_SIBLING_GRACE_SECONDS:
                    continue
            shutil.rmtree(sibling, ignore_errors=True)

    def _close_workspace(self, path: Path) -> None:
        with self._workspaces_lock:
            ws = self._workspaces.pop(path, None)
        if ws is not None:
            ws.close()

    def _close_all_workspaces(self) -> None:
        with self._workspaces_lock:
            workspaces = list(self._workspaces.values())
            self._workspaces.clear()
        for ws in workspaces:
            ws.close()

    def __enter__(self) -> "RollingChunkPrefetcher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._stop.set()
        # Drain queue so a blocked producer can put and observe _stop.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._close_all_workspaces()
        # Per-run parent is empty after _close_all_workspaces; rmtree handles
        # the partial-failure case where a workspace's own close() errored out.
        shutil.rmtree(self._run_dir, ignore_errors=True)
