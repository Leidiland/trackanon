"""Synchronous HTTP client for the ComfyUI REST API."""
from __future__ import annotations

import copy
import logging
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import requests

log = logging.getLogger(__name__)

_DEFAULT_COMFYUI_ROOT = Path("external/comfyui")


def _clean_workflow(workflow: dict) -> dict:
    """Drop top-level metadata entries (keep nodes with class_type)."""
    return {
        k: v for k, v in workflow.items()
        if isinstance(v, dict) and "class_type" in v
    }


class ComfyUIClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        self._base = f"http://{host}:{port}"
        self._session = requests.Session()
        self._client_id = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_alive(self, timeout: float = 2.0) -> bool:
        """Return True if the ComfyUI server responds on /system_stats."""
        try:
            r = self._session.get(f"{self._base}/system_stats", timeout=timeout)
            return r.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def has_node(self, class_type: str, timeout: float = 5.0) -> bool:
        """Return True if a custom node class is registered in ComfyUI."""
        try:
            r = self._session.get(
                f"{self._base}/object_info/{class_type}", timeout=timeout
            )
            if not r.ok:
                return False
            return bool(r.json())
        except requests.exceptions.RequestException:
            return False

    def free_vram(self, timeout: float = 5.0, wait_for_drop_s: float = 2.0,
                  min_free_mib: int = 4096) -> bool:
        """Unload models + free allocator; poll until VRAM actually drops."""
        try:
            free_before = self._free_vram_mib(timeout=timeout)
            r = self._session.post(
                f"{self._base}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=timeout,
            )
            r.raise_for_status()
        except requests.exceptions.RequestException:
            return False

        if wait_for_drop_s <= 0 or free_before is None:
            return True
        deadline = time.monotonic() + wait_for_drop_s
        while time.monotonic() < deadline:
            free_after = self._free_vram_mib(timeout=timeout)
            if free_after is not None and free_after - free_before >= min_free_mib:
                return True
            time.sleep(0.1)
        return False

    def _free_vram_mib(self, timeout: float = 2.0) -> int | None:
        """Return GPU 0 free VRAM in MiB from /system_stats, or None on failure."""
        try:
            r = self._session.get(f"{self._base}/system_stats", timeout=timeout)
            r.raise_for_status()
            devices = r.json().get("devices") or []
            if not devices:
                return None
            vram_free = devices[0].get("vram_free")
            if vram_free is None:
                return None
            return int(vram_free) // (1024 * 1024)
        except (requests.exceptions.RequestException, ValueError, KeyError):
            return None

    # ------------------------------------------------------------------
    # Image upload
    # ------------------------------------------------------------------

    def upload_image(self, image_rgb: np.ndarray, filename: str) -> str:
        """Upload as input PNG; returns the server-side filename (may differ on collision)."""
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError(f"cv2.imencode failed for {filename}")
        r = self._session.post(
            f"{self._base}/upload/image",
            files={"image": (filename, buf.tobytes(), "image/png")},
            data={"type": "input", "overwrite": "true"},
            timeout=15.0,
        )
        r.raise_for_status()
        return r.json()["name"]

    def upload_mask(self, mask: np.ndarray, filename: str) -> str:
        """Upload boolean mask as 3-channel PNG (white = inpaint area)."""
        mask_uint8 = mask.astype(np.uint8) * 255
        mask_rgb = np.stack([mask_uint8, mask_uint8, mask_uint8], axis=-1)
        return self.upload_image(mask_rgb, filename)

    # ------------------------------------------------------------------
    # Workflow execution
    # ------------------------------------------------------------------

    def queue_prompt(self, workflow: dict) -> str:
        """POST a workflow to /prompt.  Returns the prompt_id string."""
        clean_wf = _clean_workflow(workflow)
        payload = {"prompt": clean_wf, "client_id": self._client_id}
        r = self._session.post(f"{self._base}/prompt", json=payload, timeout=15.0)
        if not r.ok:
            try:
                body = r.json()
                node_errs = body.get("node_errors", {})
                detail = "; ".join(
                    f"node {nid} ({info.get('class_type','')}): "
                    + ", ".join(e.get("details", e.get("message", "")) for e in info.get("errors", []))
                    for nid, info in node_errs.items()
                ) or body.get("error", {}).get("message", r.text)
                raise RuntimeError(f"ComfyUI {r.status_code}: {detail}")
            except (ValueError, KeyError):
                r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"ComfyUI rejected workflow: {data['error']}")
        return data["prompt_id"]

    def wait_for_result(
        self, prompt_id: str, timeout: float = 60.0, poll_interval: float = 0.3
    ) -> dict:
        """Poll /history until the job completes; raise TimeoutError on deadline."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                r = self._session.get(
                    f"{self._base}/history/{prompt_id}", timeout=10.0
                )
                r.raise_for_status()
                history = r.json()
                entry = history.get(prompt_id)
                if entry and entry.get("status", {}).get("completed"):
                    return entry
            except requests.exceptions.RequestException as e:
                log.debug("Polling error for %s: %s", prompt_id, e)
            time.sleep(poll_interval)
        raise TimeoutError(
            f"ComfyUI prompt {prompt_id} did not complete within {timeout:.0f}s"
        )

    def fetch_image(
        self,
        filename: str,
        subfolder: str = "",
        img_type: str = "temp",
    ) -> np.ndarray:
        """Fetch a generated image from /view and return as (H, W, 3) uint8 RGB."""
        r = self._session.get(
            f"{self._base}/view",
            params={"filename": filename, "subfolder": subfolder, "type": img_type},
            timeout=15.0,
        )
        r.raise_for_status()
        arr = np.frombuffer(r.content, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to decode image returned by ComfyUI: {filename}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # End-of-run filesystem cleanup
    # ------------------------------------------------------------------

    def cleanup_artifacts(
        self,
        *,
        input_prefixes: list[str] | None = None,
        output_prefixes: list[str] | None = None,
        comfyui_root: Path | None = None,
    ) -> tuple[int, int]:
        # Sweep our per-call PNGs from ComfyUI's input/ and output/ dirs.
        # The disk copies are unused after fetch_image returns; without this
        # sweep output/ grows unbounded across runs.
        root = Path(comfyui_root) if comfyui_root else _DEFAULT_COMFYUI_ROOT
        n_in = self._sweep_prefix(root / "input", input_prefixes or [])
        n_out = self._sweep_prefix(root / "output", output_prefixes or [])
        return n_in, n_out

    @staticmethod
    def _sweep_prefix(dir_path: Path, prefixes: list[str]) -> int:
        if not prefixes or not dir_path.exists():
            return 0
        n = 0
        for entry in dir_path.iterdir():
            if not entry.is_file():
                continue
            if not any(entry.name.startswith(p) for p in prefixes):
                continue
            try:
                entry.unlink()
                n += 1
            except OSError as e:
                log.warning("cleanup: failed to remove %s: %s", entry, e)
        return n

    # ------------------------------------------------------------------
    # Workflow patching
    # ------------------------------------------------------------------

    def patch_workflow(
        self,
        template: dict,
        *,
        image_name: str,
        mask_name: str,
        pose_name: str | None,
        seed: int,
        prompt: str,
        neg_prompt: str,
        denoise: float,
        steps: int,
        cfg_scale: float,
        output_prefix: str,
        node_ids: dict,
        controlnet_strength: float | None = None,
        controlnet_end_percent: float | None = None,
    ) -> dict:
        """Return a deep-copied workflow with per-call fields filled in."""
        wf = copy.deepcopy(template)

        def _set(node_id: str | None, key: str, value) -> None:
            if node_id and node_id in wf:
                wf[node_id]["inputs"][key] = value

        _set(node_ids.get("positive"),   "text",             prompt)
        _set(node_ids.get("negative"),   "text",             neg_prompt)
        _set(node_ids.get("load_image"), "image",            image_name)
        _set(node_ids.get("load_mask"),  "image",            mask_name)
        _set(node_ids.get("ksampler"),   "seed",             seed)
        _set(node_ids.get("ksampler"),   "denoise",          denoise)
        _set(node_ids.get("ksampler"),   "steps",            steps)
        _set(node_ids.get("ksampler"),   "cfg",              cfg_scale)
        _set(node_ids.get("save_image"), "filename_prefix",  output_prefix)

        if pose_name and node_ids.get("load_pose"):
            _set(node_ids["load_pose"], "image", pose_name)

        if controlnet_strength is not None and node_ids.get("controlnet"):
            _set(node_ids["controlnet"], "strength", controlnet_strength)
        if controlnet_end_percent is not None and node_ids.get("controlnet"):
            _set(node_ids["controlnet"], "end_percent", controlnet_end_percent)

        return wf

    # ------------------------------------------------------------------
    # History parsing
    # ------------------------------------------------------------------

    def extract_output_image(
        self, history_entry: dict, save_node_id: str
    ) -> dict | None:
        """Return {filename, subfolder, type} from the SaveImage node, or None."""
        try:
            images = history_entry["outputs"][save_node_id]["images"]
            if images:
                return images[0]
        except (KeyError, IndexError, TypeError):
            pass
        return None
