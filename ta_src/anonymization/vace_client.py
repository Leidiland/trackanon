"""Wan-VACE generation against a ComfyUI server.

build_graph assembles the native WanVaceToVideo pipeline (no custom nodes for
1.3B; UnetLoaderGGUF for the 14B GGUF). VaceClient.generate uploads a bundle's
control/mask/reference, submits the graph, polls to completion, and returns the
rendered clip. The host/port are configurable so the same client drives the
local portable ComfyUI or a remote rented GPU.
"""
from __future__ import annotations

import contextlib
import json
import logging
import mimetypes
import queue
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

_CONTROL_IN, _MASK_IN, _REF_IN = "vace_control.mp4", "vace_mask.mp4", "vace_ref.png"


def build_graph(
    meta, *, prompt: str, negative: str, unet, clip: str, vae: str,
    seed: int = 42, steps: int = 20, cfg: float = 6.0, shift: float = 8.0,
    sampler: str = "uni_pc", scheduler: str = "simple", strength: float = 1.0,
    gguf: str | None = None, lora: str | None = None, lora_strength: float = 1.0,
    filename_prefix: str = "vace/out",
    control_in: str = _CONTROL_IN, mask_in: str = _MASK_IN, ref_in: str = _REF_IN,
    weight_dtype: str = "default",
) -> dict:
    """Native WanVaceToVideo graph (ComfyUI API format). When `lora` is set, a
    LoraLoaderModelOnly (e.g. the CausVid/self-forcing distill) feeds the sampler
    model, so a few low-cfg steps match many-step base quality. `weight_dtype`
    (e.g. fp8_e4m3fn) quantises the UNET on load for ~1.4-1.7x on fp8 tensor
    cores — helps distill and native alike (only the safetensors loader; GGUF
    carries its own quant)."""
    W, H = meta["size"]
    L = meta["n"]
    loader = (
        {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": gguf}} if gguf
        else {"class_type": "UNETLoader",
              "inputs": {"unet_name": unet, "weight_dtype": weight_dtype}}
    )
    # Distill LoRA (node "19") between the loader and ModelSamplingSD3 when set.
    sampling_model_src = ["1", 0]
    lora_node = {}
    if lora:
        lora_node = {"19": {"class_type": "LoraLoaderModelOnly",
                            "inputs": {"model": ["1", 0], "lora_name": lora,
                                       "strength_model": lora_strength}}}
        sampling_model_src = ["19", 0]
    return {
        "1": loader,
        **lora_node,
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "wan"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": sampling_model_src, "shift": shift}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}},
        "7": {"class_type": "LoadVideo", "inputs": {"file": control_in}},
        "8": {"class_type": "GetVideoComponents", "inputs": {"video": ["7", 0]}},
        "9": {"class_type": "LoadVideo", "inputs": {"file": mask_in}},
        "10": {"class_type": "GetVideoComponents", "inputs": {"video": ["9", 0]}},
        "11": {"class_type": "ImageToMask", "inputs": {"image": ["10", 0], "channel": "red"}},
        "12": {"class_type": "LoadImage", "inputs": {"image": ref_in}},
        "13": {"class_type": "WanVaceToVideo",
               "inputs": {"positive": ["5", 0], "negative": ["6", 0], "vae": ["3", 0],
                          "width": W, "height": H, "length": L, "batch_size": 1,
                          "strength": strength, "control_video": ["8", 0],
                          "control_masks": ["11", 0], "reference_image": ["12", 0]}},
        "14": {"class_type": "KSampler",
               "inputs": {"model": ["4", 0], "positive": ["13", 0], "negative": ["13", 1],
                          "latent_image": ["13", 2], "seed": seed, "steps": steps,
                          "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler,
                          "denoise": 1.0}},
        "15": {"class_type": "TrimVideoLatent",
               "inputs": {"samples": ["14", 0], "trim_amount": ["13", 3]}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["3", 0]}},
        "17": {"class_type": "CreateVideo", "inputs": {"images": ["16", 0], "fps": float(meta["fps"])}},
        "18": {"class_type": "SaveVideo",
               "inputs": {"video": ["17", 0], "filename_prefix": filename_prefix,
                          "format": "auto", "codec": "auto"}},
    }


class VaceClient:
    def __init__(
        self, *, host: str, port: int, input_dir=None, output_dir=None,
        timeout_s: float = 2700.0, poll_s: float = 4.0,
    ):
        self._addr = f"{host}:{port}"
        # Vestigial for the HTTP path (generate uploads via /upload/image and
        # fetches via /view) — tolerate None so the shared-port pool need not
        # fabricate staging dirs.
        self._input_dir = Path(input_dir) if input_dir is not None else None
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._timeout_s = timeout_s
        self._poll_s = poll_s

    def _raw_post(self, path, data=None):
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(
            f"http://{self._addr}{path}", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read()

    def _post(self, path, data=None):
        # JSON-returning endpoints (/prompt, /history, /system_stats).
        return json.loads(self._raw_post(path, data).decode())

    def is_alive(self) -> bool:
        try:
            self._post("/system_stats")
            return True
        except (urllib.error.URLError, OSError):
            return False

    def free_vram(self) -> bool:
        # Evict the VACE server's resident weights so it doesn't co-reside with
        # SAM 3 during Pass 1 (the WSL OOM/CUDA-hang the two-pass path guards).
        # /free returns 200 with an empty body, so don't parse JSON.
        try:
            self._raw_post("/free", {"unload_models": True, "free_memory": True})
            return True
        except (urllib.error.URLError, OSError):
            return False

    def generate(self, bundle_dir, graph, *, remote_names=None) -> Path:
        """Upload the bundle through ComfyUI's HTTP API, submit `graph`, poll
        to completion, and download the rendered clip. Works against a local
        ComfyUI on 127.0.0.1 OR a remote one through an SSH-tunneled port —
        no filesystem assumption. `remote_names`=(control,mask,ref) must match
        the filenames baked into `graph`'s load nodes; unique-per-render names
        stop ComfyUI's input cache from serving a stale render (its LoadVideo
        cache keys on filename, not content — identical names → empty output)."""
        bundle = Path(bundle_dir)
        c_in, m_in, r_in = remote_names or (_CONTROL_IN, _MASK_IN, _REF_IN)
        self._upload_file(bundle / "control.mp4", c_in)
        self._upload_file(bundle / "mask.mp4", m_in)
        self._upload_file(bundle / "reference.png", r_in)

        pid = self._post("/prompt", {"prompt": graph})["prompt_id"]
        saved = self._await(pid)
        if saved is None:
            raise RuntimeError(f"VACE render produced no output (prompt {pid})")
        dst = bundle / f"render_{saved['filename']}"
        self._download_file(saved["filename"], saved.get("subfolder", ""), dst)
        return dst

    def _upload_file(self, local_path: Path, remote_name: str) -> dict:
        """POST `local_path` to /upload/image with multipart/form-data. The
        endpoint is named "image" but ComfyUI writes the file bytes straight to
        its --input-directory without content-type checks, so .mp4 works."""
        boundary = f"----vace{uuid.uuid4().hex}"
        mime = mimetypes.guess_type(remote_name)[0] or "application/octet-stream"
        with open(local_path, "rb") as f:
            payload = f.read()
        head = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{remote_name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode()
        tail = f"\r\n--{boundary}--\r\n".encode()
        body = head + payload + tail
        req = urllib.request.Request(
            f"http://{self._addr}/upload/image", data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

    def _download_file(self, filename: str, subfolder: str, dest_path: Path) -> None:
        """GET the rendered output from /view and stream it to disk."""
        qs = urllib.parse.urlencode({
            "filename": filename, "subfolder": subfolder, "type": "output",
        })
        req = urllib.request.Request(f"http://{self._addr}/view?{qs}")
        with urllib.request.urlopen(req, timeout=300) as r, open(dest_path, "wb") as f:
            shutil.copyfileobj(r, f)

    def _await(self, pid):
        t0 = time.time()
        while True:
            time.sleep(self._poll_s)
            try:
                hist = self._post(f"/history/{pid}")
            except urllib.error.URLError:
                continue
            if pid in hist:
                st = hist[pid].get("status", {})
                if st.get("status_str") == "error":
                    raise RuntimeError(f"VACE render errored: {json.dumps(hist[pid])[:800]}")
                outs = hist[pid].get("outputs", {})
                if st.get("completed") or st.get("status_str") == "success" or outs:
                    for o in outs.values():
                        for key in ("images", "video", "gifs"):
                            for f in o.get(key, []) or []:
                                return f
                    return None
            if time.time() - t0 > self._timeout_s:
                raise TimeoutError(f"VACE render timed out after {self._timeout_s}s")


class VaceClientPool:
    """A semaphore-gated pool of N VaceClients on consecutive ports — the
    multi-worker dispatch axis. On a rented GPU box, N ComfyUI processes run on
    `base_port..base_port+N-1`, each with its own staging dirs; the pipeline
    acquires one for each gid render and lets the OS GPU scheduler interleave
    them. Default workers=1 collapses to a trivial pool (no behavioural change
    from the original single-client path)."""

    def __init__(
        self, *, clients: list[VaceClient] | None = None,
        host: str | None = None, base_port: int | None = None,
        workers: int = 1, input_dir_root=None, output_dir_root=None,
        timeout_s: float = 2700.0, poll_s: float = 4.0,
    ):
        if clients is not None:
            self._clients = list(clients)
        else:
            assert host is not None and base_port is not None
            self._clients = []
            for i in range(workers):
                in_dir, out_dir = _staged_dirs(
                    input_dir_root, output_dir_root, i, workers,
                )
                self._clients.append(VaceClient(
                    host=host, port=base_port + i,
                    input_dir=in_dir, output_dir=out_dir,
                    timeout_s=timeout_s, poll_s=poll_s,
                ))
        self._sem = threading.Semaphore(len(self._clients))
        self._lock = threading.Lock()
        self._free: list[VaceClient] = list(self._clients)

    @property
    def size(self) -> int:
        return len(self._clients)

    @contextlib.contextmanager
    def acquire(self):
        self._sem.acquire()
        with self._lock:
            client = self._free.pop()
        try:
            yield client
        finally:
            with self._lock:
                self._free.append(client)
            self._sem.release()

    def is_alive(self) -> bool:
        # All workers must answer — a stale one would silently bottleneck the pool.
        return all(c.is_alive() for c in self._clients)

    def free_vram(self) -> bool:
        return all(c.free_vram() for c in self._clients)


class SharedPortPool:
    """Pool backed by a shared (multiprocessing) queue of ports — the work-
    stealing dispatch axis for the process-pool render path. Each gid runs in
    its own process; per window render it pulls ANY free port (= GPU worker)
    from the queue and returns it after, so no GPU sits idle while some gid
    still has a window to render (vs static gid->port assignment, where uneven
    gid sizes leave some GPUs idle through the tail). A port is held by at most
    one process at a time (it's removed from the queue), so workers never get
    concurrent uploads. Same `acquire()` contract as VaceClientPool, so
    `_render_window` is unchanged."""

    def __init__(
        self, port_queue, *, host: str, n_ports: int,
        input_dir_root=None, output_dir_root=None,
        timeout_s: float = 2700.0, poll_s: float = 4.0,
        acquire_timeout_s: float | None = None,
    ):
        self._q = port_queue
        self._host = host
        self._n = int(n_ports)
        self._in = input_dir_root
        self._out = output_dir_root
        self._timeout_s = timeout_s
        self._poll_s = poll_s
        # Cap the wait for a free port so a fully-dead/wedged pool fails the gid
        # instead of blocking forever; > one render so normal saturation waits.
        self._acq = acquire_timeout_s if acquire_timeout_s is not None else timeout_s * 2

    @property
    def size(self) -> int:
        return self._n

    @contextlib.contextmanager
    def acquire(self):
        try:
            port = self._q.get(timeout=self._acq)
        except queue.Empty:
            raise RuntimeError(
                f"VACE pool exhausted: no GPU worker free within {self._acq}s "
                "(all dead or wedged)"
            )
        client = VaceClient(
            host=self._host, port=port,
            input_dir=self._in, output_dir=self._out,
            timeout_s=self._timeout_s, poll_s=self._poll_s,
        )
        requeue = True
        try:
            yield client
        except (urllib.error.URLError, OSError):
            # Connection-level failure: the worker may have crashed. If it's
            # truly unreachable, DON'T return the port — quarantine it so other
            # gid processes stop drawing a dead worker (the shared queue shrinks
            # for everyone). A transient blip on a live worker keeps the port.
            requeue = client.is_alive()
            if not requeue:
                log.warning("VACE: worker on port %d unreachable — quarantined "
                            "from the pool", port)
            raise
        finally:
            if requeue:
                self._q.put(port)

    def is_alive(self) -> bool:
        return True                         # liveness is checked once in the parent

    def free_vram(self) -> bool:
        return True


def _staged_dirs(input_root, output_root, i: int, workers: int):
    """workers=1 -> unchanged dirs (legacy single-server layout); workers>1 ->
    per-worker `_{i}` suffix so concurrent bundle uploads don't trample."""
    if workers == 1:
        return input_root, output_root
    return f"{input_root}_{i}", f"{output_root}_{i}"
