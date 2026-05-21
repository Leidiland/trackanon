import gc
import queue
import threading
from fractions import Fraction

import av


def _detect_encoder() -> tuple[str, dict, str]:
    """Return (codec, options, pix_fmt) for the fastest available H.264 encoder."""
    try:
        av.codec.Codec("h264_nvenc", "w")
        return "h264_nvenc", {"preset": "p4", "tune": "hq", "rc": "vbr", "cq": "23"}, "nv12"
    except Exception:
        pass
    return "libx264", {"preset": "medium", "crf": "18"}, "yuv420p"


_ENCODER, _ENCODER_OPTS, _ENCODER_PIX_FMT = _detect_encoder()

_SENTINEL = object()


class VideoInfo:
    def __init__(self, width, height, fps, frame_count, duration):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = frame_count
        self.duration = duration


class VideoReader:
    def __init__(self, path, start_time=None, end_time=None, fps=None):
        self._container = av.open(path)
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self._stream.codec_context.thread_count = 0  # auto
        self._time_base = float(self._stream.time_base)
        self.start_time = start_time or 0
        self.end_time = end_time
        self._target_fps = fps
        self._gc_counter = 0

    @property
    def info(self):
        s = self._stream
        src_fps = float(s.average_rate) if s.average_rate else 0.0
        src_duration = float(s.duration * s.time_base) if s.duration else 0.0
        target_fps = self._target_fps if self._target_fps else src_fps

        # frame_count must reflect what frames() actually yields, i.e. the
        # post-trim, post-downsample count — otherwise progress bars lie when
        # start_time/end_time/fps are set.
        clip_end = self.end_time if self.end_time is not None else src_duration
        effective_duration = max(0.0, clip_end - self.start_time)
        if target_fps > 0 and effective_duration > 0:
            frame_count = int(round(effective_duration * target_fps))
        else:
            frame_count = s.frames  # fallback for streams without duration

        return VideoInfo(
            width=s.width,
            height=s.height,
            fps=target_fps,
            frame_count=frame_count,
            duration=effective_duration if effective_duration > 0 else src_duration,
        )

    def frames(self):
        """Yield decoded frames as uint8 RGB numpy arrays (H, W, 3).

        Respects start_time, end_time, and target fps downsampling.
        """
        src_fps = float(self._stream.average_rate)
        target_fps = self._target_fps or src_fps
        frame_step = 1.0 / target_fps
        time_base = self._time_base

        if self.start_time > 0:
            self.seek(self.start_time)

        next_ts = float(self.start_time)

        for packet in self._container.demux(self._stream):
            for frame in packet.decode():
                if frame.pts is None:
                    continue
                ts = frame.pts * time_base
                if ts < self.start_time:
                    continue
                if self.end_time is not None and ts >= self.end_time:
                    return
                if ts >= next_ts:
                    next_ts += frame_step
                    self._occasional_gc()
                    yield frame.to_ndarray(format="rgb24")

    def _occasional_gc(self):
        self._gc_counter += 1
        if self._gc_counter % 10 == 9:
            gc.collect()

    def seek(self, timestamp):
        pts = int(timestamp / self._stream.time_base)
        self._container.seek(pts, stream=self._stream)

    def get_frame(self, index):
        fps = float(self._stream.average_rate)
        if fps <= 0:
            return None
        target_ts = index / fps
        self.seek(target_ts)
        for frame in self._container.decode(self._stream):
            if frame.pts is None:
                continue
            if frame.pts * self._time_base >= target_ts:
                return frame.to_ndarray(format="rgb24")
        return None

    def close(self):
        self._container.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class PrefetchReader:
    """Wraps a VideoReader and pre-decodes frames on a background thread (O5).

    Decoding (CPU) and GPU processing can then overlap, hiding the ~5–10ms
    per-frame decode latency inside the GPU compute time.

    Shutdown order matters: close() must (1) signal stop, (2) drain the queue
    to unblock any blocked put(), (3) join the thread, and only then (4) close
    the underlying reader.  Closing the reader while the producer thread is
    inside av.packet.decode() / frame.to_ndarray() causes heap corruption.

    Args:
        reader: an open VideoReader instance.
        buffer_size: number of decoded frames to buffer ahead.
    """

    def __init__(self, reader: VideoReader, buffer_size: int = 4):
        self._reader = reader
        self._q: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._stop = threading.Event()
        self._exc: BaseException | None = None
        self._thread = threading.Thread(
            target=self._producer, daemon=True, name="frame-prefetch"
        )
        self._thread.start()

    def _producer(self):
        try:
            for frame in self._reader.frames():
                if self._stop.is_set():
                    break
                # Use a timeout loop so the stop event is checked even when
                # the queue is full and put() would block indefinitely.
                while not self._stop.is_set():
                    try:
                        self._q.put(frame, timeout=0.05)
                        break
                    except queue.Full:
                        pass
        except Exception as e:
            self._exc = e
        finally:
            # Always put sentinel so frames() can exit cleanly.
            try:
                self._q.put(_SENTINEL, timeout=1.0)
            except queue.Full:
                pass

    def frames(self):
        while True:
            # Use a timeout so GeneratorExit (from generator.close()) can be
            # processed without blocking forever on an empty queue.
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            if item is _SENTINEL:
                if self._exc is not None:
                    raise self._exc
                return
            yield item

    @property
    def info(self):
        return self._reader.info

    def close(self):
        # Step 1: tell the producer to stop at the next checkpoint.
        self._stop.set()
        # Step 2: drain pending frames so the producer is never stuck on put().
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        # Step 3: wait for the thread to exit completely before touching the
        # underlying av container — accessing it from two threads concurrently
        # corrupts libav's internal state and causes a heap segfault.
        self._thread.join()
        # Step 4: now safe to close.
        self._reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read_video(path, start_time=None, end_time=None, fps=None, prefetch: bool = False):
    """Open a video for reading.

    Args:
        prefetch: when True wraps the reader with PrefetchReader so frame
                  decoding overlaps with GPU compute (O5).  Disabled by default
                  because PrefetchReader.close() deadlocks when the producer
                  thread is stuck inside libav's C-level demux() — the join()
                  before container.close() ordering cannot be safely interrupted.
                  With ComfyUI as the pipeline bottleneck (~1.5 s/person) the
                  few-ms decode overlap provides no measurable benefit.
    """
    reader = VideoReader(path, start_time=start_time, end_time=end_time, fps=fps)
    if prefetch:
        return PrefetchReader(reader)
    return reader


def save_video(frames, output_path, fps, width, height):
    """Encode an iterable of RGB numpy arrays (H, W, 3) and write to output_path.

    Streams frames directly — do not buffer them into a list before calling.
    Uses NVENC when available, otherwise libx264 medium/crf18.
    """
    rate = Fraction(fps).limit_denominator(10_000)
    with av.open(str(output_path), mode="w") as container:
        stream = container.add_stream(_ENCODER, rate=rate, options=_ENCODER_OPTS)
        stream.width = width
        stream.height = height
        stream.pix_fmt = _ENCODER_PIX_FMT
        for frame_arr in frames:
            frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class _StreamingVideoWriter:
    """Encodes frames on a background thread via a queue (O4).

    Allows the main pipeline loop to push frames without blocking on encoder
    I/O, and avoids buffering all visualization frames in RAM.
    """

    def __init__(self, output_path: str, fps: float, width: int, height: int,
                 maxsize: int = 16):
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(
            target=self._writer,
            args=(output_path, fps, width, height),
            daemon=True,
            name="vis-encoder",
        )
        self._exc: BaseException | None = None
        self._thread.start()

    def _writer(self, output_path, fps, width, height):
        try:
            save_video(self._drain(), output_path, fps, width, height)
        except Exception as e:
            self._exc = e

    def _drain(self):
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                return
            yield item

    def push(self, frame):
        """Non-blocking push; blocks only when the queue is full (back-pressure).

        Surfaces encoder-thread death as a RuntimeError instead of deadlocking
        on a full queue when the consumer has crashed.
        """
        while True:
            if not self._thread.is_alive():
                raise self._exc or RuntimeError("vis-encoder thread died")
            try:
                self._q.put(frame, timeout=0.5)
                return
            except queue.Full:
                continue

    def close(self):
        """Signal end-of-stream and wait for the encoder thread to finish."""
        self._q.put(_SENTINEL)
        self._thread.join()
        if self._exc is not None:
            raise self._exc

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
