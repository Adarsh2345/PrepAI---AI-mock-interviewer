from __future__ import annotations

"""
WebSocket event server for the frontend.

Runs alongside the voice pipeline in the same asyncio event loop.
The pipeline calls `broadcast()` to push JSON events to all connected clients.

Event schema (all events have a `type` field):

  { "type": "state",      "state": "IDLE"|"LISTENING"|"AGENT_SPEAKING"|"INTERRUPTED" }
  { "type": "transcript", "role": "user"|"agent", "text": "...", "partial": false }
  { "type": "latency",    "turn_id": 1, "stt_lag_ms": 250, "endpoint_to_llm_first_ms": 310,
                           "llm_first_to_tts_first_ms": 95, "tts_first_to_playback_ms": 8,
                           "total_response_latency_ms": 420, "barge_in": false }
  { "type": "barge_in" }
  { "type": "ping" }       ← sent every 5 s to keep connections alive
"""

import asyncio
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Optional import — server only activates if fastapi+uvicorn are installed
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    logger.warning("fastapi/uvicorn not installed — frontend server disabled. Run: pip install fastapi uvicorn")


class EventServer:
    """
    Manages connected WebSocket clients and broadcasts pipeline events to them.
    Thread-safe: pipeline tasks call broadcast() from the pipeline's event loop.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._app: Optional[FastAPI] = None
        self._server_task: Optional[asyncio.Task] = None
        self._current_state: Optional[str] = None
        self._interview_config: Optional[dict] = None
        self.commands: asyncio.Queue[dict] = asyncio.Queue()

    def build_app(self, frontend_html: str) -> FastAPI:
        app = FastAPI(title="Voice Agent")

        @app.get("/")
        async def index():
            return HTMLResponse(frontend_html)

        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            async with self._lock:
                self._clients.add(ws)
            # Send current state immediately so late-joining clients aren't stale
            if self._current_state:
                try:
                    await ws.send_text(json.dumps({"type": "state", "state": self._current_state}))
                except Exception:
                    pass
            # Send cached interview config if session already running
            if self._interview_config:
                try:
                    await ws.send_text(json.dumps(self._interview_config))
                except Exception:
                    pass
            logger.info("Frontend client connected (%d total)", len(self._clients))
            try:
                while True:
                    msg = await ws.receive_text()
                    try:
                        cmd = json.loads(msg)
                        if isinstance(cmd, dict) and cmd.get("type") == "command":
                            await self.commands.put(cmd)
                    except (json.JSONDecodeError, Exception):
                        pass
            except WebSocketDisconnect:
                pass
            finally:
                async with self._lock:
                    self._clients.discard(ws)
                logger.info("Frontend client disconnected (%d remaining)", len(self._clients))

        self._app = app
        return app

    async def broadcast(self, event: dict) -> None:
        if event.get("type") == "state":
            self._current_state = event.get("state")
        if event.get("type") == "questions_list":
            self._interview_config = event
        if not self._clients:
            return
        payload = json.dumps(event)
        async with self._lock:
            dead: set[WebSocket] = set()
            for ws in self._clients:
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.add(ws)
            self._clients -= dead

    async def start(self, frontend_html: str) -> None:
        if not _FASTAPI_AVAILABLE:
            logger.warning("Frontend server not started (fastapi/uvicorn missing)")
            return

        app = self.build_app(frontend_html)
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            loop="none",  # use the already-running asyncio loop
        )
        server = uvicorn.Server(config)

        # Patch uvicorn's signal handlers — we manage lifecycle ourselves
        server.install_signal_handlers = lambda: None

        self._server_task = asyncio.create_task(server.serve())
        logger.info("Frontend available at http://%s:%d", self.host, self.port)

        # Keep-alive ping loop
        asyncio.create_task(self._ping_loop())

    async def _ping_loop(self) -> None:
        while True:
            await asyncio.sleep(5)
            await self.broadcast({"type": "ping"})

    async def stop(self) -> None:
        if self._server_task:
            self._server_task.cancel()
