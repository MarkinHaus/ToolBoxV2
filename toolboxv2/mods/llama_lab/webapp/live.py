# file: toolboxv2/mods/llama_lab/webapp/live.py
"""Live (omni) mode — session tokens + a full-duplex WebSocket bridge.

A WS connection is bound (on first message, where the path param is available)
to an OmniSession running a LlamaOmniBackend against the served omni model.
Client frames {type:'audio'|'text'|'image'|'commit'} flow in; the backend's
TEXT/AUDIO/TOOL_CALL/TURN_END events are pushed back via app.ws_send.

Client wire format:
    {"type":"audio","data":"<base64 PCM int16 16k mono>"}
    {"type":"text","data":"hello"}
    {"type":"image","mime":"image/jpeg","data":"<base64>"}   # one-shot turn
    {"type":"commit"}                                        # push-to-talk end
"""

import base64
import secrets
import time

from . import db
from ..omni import LlamaOmniBackend, OmniSession


class LiveHandler:
    def __init__(self, models):
        self.models = models
        self.sessions: dict = {}             # token -> {model, base_url, key_id}

    def settings(self) -> dict:
        return db.load_config().get("live", {"enabled": True, "webcam": True, "screen": True})

    def create_session(self, model, key_id: int) -> dict:
        if not self.settings().get("enabled", True):
            return {"error": "live mode disabled"}
        m = self.models.find_for_request(model, needs_audio=True)
        if not m:
            return {"error": "no omni/audio model running — load one in the admin panel"}
        token = "live-" + secrets.token_urlsafe(24)
        self.sessions[token] = {"model": m["name"], "base_url": m["base_url"],
                                "key_id": key_id, "created": time.time()}
        return {"session_token": token, "model": m["name"],
                "ws_path": f"/v1/audio/live/ws/{token}", "settings": self.settings()}

    def get(self, token: str):
        return self.sessions.get(token)

    def close(self, token: str) -> bool:
        return self.sessions.pop(token, None) is not None


def _path_param(request, key):
    if request is None:
        return None
    pp = getattr(request, "path_params", None)
    if pp is None and isinstance(request, dict):
        pp = request.get("path_params")
    return (pp or {}).get(key)


def make_ws_class(handler: "LiveHandler", app):
    """FastTB websocket handler class bound to this LiveHandler + app (for push)."""

    live_cfg = db.load_config().get("live", {})

    class LiveWS:
        def __init__(self):
            self.sessions: dict = {}        # conn_id -> OmniSession

        async def on_connect(self, conn_id, session):
            return {"type": "connected", "conn_id": conn_id}

        async def _ensure(self, conn_id, token):
            if conn_id in self.sessions:
                return self.sessions[conn_id]
            meta = handler.get(token)
            if not meta:
                return None
            backend = LlamaOmniBackend(
                meta["base_url"], meta["model"],
                system=live_cfg.get("system", ""),
                tts_base_url=live_cfg.get("tts_base_url", ""),
                tts_model=live_cfg.get("tts_model", ""))
            os_ = OmniSession(backend, send=lambda d, c=conn_id: app.ws_send(c, d))
            await os_.start()
            self.sessions[conn_id] = os_
            return os_

        async def on_message(self, payload, conn_id, session, request=None):
            token = _path_param(request, "session_token")
            os_ = await self._ensure(conn_id, token)
            if os_ is None:
                return {"type": "error", "error": "unknown or expired session"}
            t = payload.get("type", "audio")
            if t == "audio":
                await os_.feed_audio(base64.b64decode(payload.get("data", "")))
            elif t == "text":
                await os_.feed_text(payload.get("data", ""))
            elif t in ("image", "video"):
                data = base64.b64decode(payload.get("data", ""))
                await os_.feed_media(data, payload.get("mime", "image/jpeg"),
                                     as_turn=(t == "image"))
            elif t == "commit":
                await os_.backend.commit_input()
            return None                     # responses are pushed via ws_send

        async def on_disconnect(self, conn_id, session):
            os_ = self.sessions.pop(conn_id, None)
            if os_:
                await os_.stop()

    return LiveWS
