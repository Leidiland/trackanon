"""Op-edit web UI daemon — browser tab replaces the JSON-in-editor flow.

Wire format unchanged: the daemon reads the resolver-proposed sidecar JSON
on disk, the browser POSTs the operator's edits back, the daemon writes
the JSON, and OpEditSession re-reads it through the existing validator.
The browser is just a richer editor over the same artefact.
"""
from __future__ import annotations

import json
import logging
import secrets
import subprocess
import threading
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from ta_src.pipeline.op_edit.errors import OpEditAbort
from ta_src.pipeline.op_edit.validation import (
    OpEditValidationError,
    validate_operator_mapping,
)

log = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"
_INDEX_HTML = _STATIC_DIR / "index.html"


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def open_browser(url: str) -> bool:
    """Open *url* in the operator's browser. Returns True on success.

    On WSL, prefer explorer.exe (always present, opens Windows default
    browser) over xdg-open/wslview/webbrowser, which depend on X11/WSLg.
    On failure, return False so the caller can log the URL for the operator
    to click manually instead of raising into the pause.
    """
    if _is_wsl():
        try:
            subprocess.Popen(["explorer.exe", url])
            return True
        except (FileNotFoundError, OSError) as exc:
            log.warning("explorer.exe failed (%s); falling back to webbrowser", exc)
    try:
        return bool(webbrowser.open(url))
    except Exception as exc:  # noqa: BLE001 — never let the opener kill the pause
        log.warning("webbrowser.open failed (%s)", exc)
        return False


@dataclass
class _PendingPause:
    json_path: Path
    png_path: Path
    known_gids: frozenset[int]
    settled: threading.Event = field(default_factory=threading.Event)
    aborted: bool = False


class OpEditWebDaemon:
    def __init__(self) -> None:
        self._pauses: dict[str, _PendingPause] = {}
        self.app = FastAPI()
        self._wire_routes()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._base_url: str | None = None
        self._auto_open: bool = True

    def serve_in_background(
        self, *, host: str, port: int, auto_open: bool = True,
    ) -> str:
        """Spin up uvicorn in a daemon thread; return the resolved base URL.

        Idempotent — a second call returns the already-running URL. Passing
        port=0 asks the OS for an open port (useful in tests). When
        ``auto_open`` is False the daemon will log the operator URL but not
        invoke a browser (CI / remote SSH scenarios)."""
        self._auto_open = auto_open
        if self._base_url is not None:
            return self._base_url
        config = uvicorn.Config(
            self.app, host=host, port=port, log_level="warning", access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        # Wait for the server socket to be bound so callers know the URL is live.
        deadline = threading.Event()
        while not self._server.started:
            if deadline.wait(0.02):
                break
            if not self._thread.is_alive():
                raise RuntimeError("uvicorn thread died before becoming ready")
        # The bound port lives on the first server in server.servers.
        bound_port = self._server.servers[0].sockets[0].getsockname()[1]
        self._base_url = f"http://{host}:{bound_port}"
        return self._base_url

    def shutdown(self) -> None:
        """Stop the uvicorn server and join the thread."""
        if self._server is None:
            return
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._server = None
        self._thread = None
        self._base_url = None

    def register_pause(
        self,
        *,
        json_path: Path,
        png_path: Path,
        known_gids: set[int],
    ) -> str:
        token = secrets.token_urlsafe(8)
        self._pauses[token] = _PendingPause(
            json_path=Path(json_path),
            png_path=Path(png_path),
            known_gids=frozenset(known_gids),
        )
        return token

    def wait_for(self, token: str) -> None:
        """Block until the operator applies (or aborts) the pause."""
        pause = self._pauses[token]
        pause.settled.wait()
        if pause.aborted:
            raise OpEditAbort(f"operator aborted pause {token}")

    def pending_tokens(self) -> list[str]:
        """Tokens for pauses that haven't been settled yet (apply/abort)."""
        return [t for t, p in self._pauses.items() if not p.settled.is_set()]

    def open_pause_in_browser(self, token: str) -> None:
        """Log the operator URL and (unless auto_open is disabled) launch a
        browser tab pointed at this pause."""
        if self._base_url is None:
            return
        url = f"{self._base_url}/pause/{token}"
        log.info("op-edit pause ready: %s", url)
        if self._auto_open:
            open_browser(url)

    def _wire_routes(self) -> None:
        @self.app.get("/pause/{token}", response_class=HTMLResponse)
        def get_pause_page(token: str) -> HTMLResponse:
            if token not in self._pauses:
                raise HTTPException(status_code=404, detail="unknown pause token")
            return HTMLResponse(_INDEX_HTML.read_text())

        @self.app.get("/pause/{token}/data")
        def get_pause_data(token: str) -> dict:
            pause = self._pauses.get(token)
            if pause is None:
                raise HTTPException(status_code=404, detail="unknown pause token")
            payload = json.loads(pause.json_path.read_text())
            payload["image_url"] = f"/pause/{token}/image"
            return payload

        @self.app.get("/pause/{token}/image")
        def get_pause_image(token: str) -> FileResponse:
            pause = self._pauses.get(token)
            if pause is None:
                raise HTTPException(status_code=404, detail="unknown pause token")
            return FileResponse(pause.png_path, media_type="image/png")

        @self.app.post("/pause/{token}/apply")
        def apply_pause(token: str, payload: dict) -> dict:
            pause = self._pauses.get(token)
            if pause is None:
                raise HTTPException(status_code=404, detail="unknown pause token")
            by_obj = {int(o["sam3_obj_id"]): int(o["operator_gid"])
                      for o in payload.get("objects", [])}
            try:
                validate_operator_mapping(by_obj, pause.known_gids)
            except OpEditValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            data = json.loads(pause.json_path.read_text())
            for obj in data["objects"]:
                obj_id = int(obj["sam3_obj_id"])
                if obj_id in by_obj:
                    obj["operator_gid"] = by_obj[obj_id]
            pause.json_path.write_text(json.dumps(data, indent=2))
            pause.settled.set()
            return {"status": "applied"}

        @self.app.post("/pause/{token}/abort")
        def abort_pause(token: str) -> dict:
            pause = self._pauses.get(token)
            if pause is None:
                raise HTTPException(status_code=404, detail="unknown pause token")
            pause.aborted = True
            pause.settled.set()
            return {"status": "aborted"}
