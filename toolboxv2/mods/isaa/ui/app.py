"""
ISAA UI — FastTB application.

Single-user, single port. Reuses ToolBoxV2 ISAA module + FastTB infrastructure.

Launch:
    python -m toolboxv2.mods.isaa.ui.app
    # or via tb CLI:
    tb -m isaa launchUI --port 8765
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from toolboxv2.utils.workers.fast_tb import FastTB
from toolboxv2.utils.workers.upload_manager import get_upload_manager

from .chat_store import ChatStore
from .stream_bridge import StreamBridge, BridgeBroadcaster
from .routes import chats as chats_routes
from .routes import vfs as vfs_routes
from .routes import agents as agents_routes
from .routes import skills as skills_routes
from .routes import running as running_routes
from .routes import tools as tools_routes

logger = logging.getLogger("isaa.ui")

# ============================================================================
# Greeting (lightweight stand-in; uses isaa.extras.isaa_branding if available)
# ============================================================================

def _greeting(lang: str = "de") -> dict:
    try:
        from toolboxv2.mods.isaa.extras.isaa_branding import get_greeting  # type: ignore
        g = get_greeting(lang=lang)
        if isinstance(g, dict):
            return g
        if isinstance(g, str):
            return {"greeting": g, "time_interval": ""}
    except Exception:
        pass
    import time
    h = int(time.strftime("%H"))
    if h < 5:
        slot, msg_de, msg_en = "night", "Nacht", "Late night"
    elif h < 11:
        slot, msg_de, msg_en = "morning", "Guten Morgen", "Good morning"
    elif h < 14:
        slot, msg_de, msg_en = "noon", "Guten Tag", "Good day"
    elif h < 18:
        slot, msg_de, msg_en = "afternoon", "Schönen Nachmittag", "Good afternoon"
    elif h < 23:
        slot, msg_de, msg_en = "evening", "Guten Abend", "Good evening"
    else:
        slot, msg_de, msg_en = "night", "Späte Stunde", "Late night"
    return {"greeting": msg_de if lang == "de" else msg_en, "time_interval": slot}


# ============================================================================
# WS handler — fixed path /ws/chat
# ============================================================================

class ChatWSHandler:
    """One handler instance, dispatches per chat_id from hello frame."""

    def __init__(self):
        self.bridge: StreamBridge | None = None
        self.store: ChatStore | None = None
        # conn_id -> chat_id
        self._connections: dict[str, str] = {}
        # chat_id -> set of conn_ids
        self._channels: dict[str, set[str]] = {}
        # conn_id -> WebSocketContext (set on connect)
        self._contexts: dict[str, Any] = {}

    def attach(self, bridge: StreamBridge, store: ChatStore) -> None:
        self.bridge = bridge
        self.store = store

    async def on_connect(self, ctx) -> None:
        self._contexts[ctx.conn_id] = ctx
        logger.info("[ws] connect %s", ctx.conn_id)

    async def on_disconnect(self, ctx) -> None:
        cid = self._connections.pop(ctx.conn_id, None)
        if cid:
            self._channels.get(cid, set()).discard(ctx.conn_id)
        self._contexts.pop(ctx.conn_id, None)
        logger.info("[ws] disconnect %s (chat=%s)", ctx.conn_id, cid)

    async def on_message(self, payload, ctx) -> None:
        op = payload.get("op", "")
        if op == "hello":
            chat_id = payload.get("chat_id", "")
            last_seq = int(payload.get("last_seq", 0))
            if not chat_id or not self.store or not self.store.exists(chat_id):
                await ctx.send({"type": "error", "error": "unknown chat_id"})
                return
            self._connections[ctx.conn_id] = chat_id
            self._channels.setdefault(chat_id, set()).add(ctx.conn_id)
            # Replay all frames > last_seq
            for frame in self.store.replay(chat_id, after_seq=last_seq):
                await ctx.send(frame)
            await ctx.send({
                "type": "resync_done",
                "is_running": self.bridge.is_running(chat_id) if self.bridge else False,
                "run_id": self.bridge.get_run_id(chat_id) if self.bridge else None,
            })
            return

        chat_id = self._connections.get(ctx.conn_id)
        if not chat_id:
            await ctx.send({"type": "error", "error": "send hello first"})
            return

        if op == "send":
            text = (payload.get("text") or "").strip()
            atts = payload.get("attachments") or []
            if not text and not atts:
                return
            meta = self.store.get_meta(chat_id) if self.store else None
            agent = (meta.agent if meta else "") or "self"
            if not self.bridge:
                return
            started = await self.bridge.start(chat_id, agent, text, atts)
            if not started:
                await ctx.send({"type": "warning", "message": "already running"})
            # Auto-title from first user message if empty
            if meta and (not meta.title or meta.title == "New Chat"):
                self.store.update_meta(chat_id, title=text[:40])

        elif op == "pause":
            if self.bridge:
                await self.bridge.pause(chat_id)

        elif op == "cancel":
            if self.bridge:
                await self.bridge.cancel(chat_id)

        elif op == "ping":
            await ctx.send({"type": "pong"})

        elif op == "rollback":
            step_id = payload.get("step_id")
            if not step_id or not self.store:
                return
            if self.bridge and self.bridge.is_running(chat_id):
                await self.bridge.cancel(chat_id)
            new_last = self.store.truncate_after(chat_id, step_id)
            self.store.update_meta(chat_id, run_id=None)
            await self.broadcast(chat_id, {
                "type": "rollback_done",
                "new_last_seq": new_last,
            })

        elif op == "widget_action":
            # Widget emitted an action back to us. Three sub-actions:
            #   - "set_var": persist + broadcast (no agent involvement needed)
            #   - others: persist as event frame; the agent can pick it up next turn
            widget_id = payload.get("widget_id", "")
            action = payload.get("action", "")
            data = payload.get("payload", {})
            if action == "set_var":
                scope = (data or {}).get("scope", "agent")
                key = (data or {}).get("key", "")
                value = (data or {}).get("value")
                if not key:
                    return
                scope_key = "vars_" + ("agent" if scope == "agent" else "global")
                meta = self.store.get_meta(chat_id) if self.store else None
                if meta:
                    current = dict(meta.ui.get(scope_key, {}))
                    current[key] = value
                    self.store.update_meta(chat_id, ui={scope_key: current})
                await self.broadcast(chat_id, {
                    "type": "var_set", "scope": scope, "key": key, "value": value,
                })
            else:
                # Surface as an event frame so the agent (or later replay) sees it.
                frame = {
                    "type": "widget_event",
                    "widget_id": widget_id,
                    "action": action,
                    "payload": data,
                }
                if self.store:
                    self.store.append(chat_id, frame)
                await self.broadcast(chat_id, frame)

    async def broadcast(self, chat_id: str, frame: dict) -> None:
        """Send a frame to every WS connection subscribed to this chat."""
        conn_ids = list(self._channels.get(chat_id, set()))
        for cid in conn_ids:
            ctx = self._contexts.get(cid)
            if ctx is None:
                continue
            try:
                await ctx.send(frame)
            except Exception:
                logger.exception("[ws] send to %s failed", cid)


# ============================================================================
# App factory
# ============================================================================

def build_app(tb_app=None) -> FastTB:
    """Build and return the FastTB app, wired to a running ToolBoxV2 instance."""
    from toolboxv2 import get_app
    tb_app = tb_app or get_app(from_="isaa_ui")

    # Get ISAA module (sync get_mod; agents themselves are loaded async on demand).
    isaa = tb_app.get_mod("isaa")
    if isaa is None:
        raise RuntimeError("ISAA module not available. Load it before launching the UI.")

    # Storage paths
    ui_root = Path(str(tb_app.data_dir)) / "isaa_ui" / "chats"
    store = ChatStore(ui_root)
    uploads = get_upload_manager(tb_app)

    # WS handler (need it now so bridge can broadcast through it)
    ws_handler = ChatWSHandler()

    async def _broadcast(chat_id: str, frame: dict) -> None:
        await ws_handler.broadcast(chat_id, frame)

    bridge = StreamBridge(isaa, store, BridgeBroadcaster(send=_broadcast))
    ws_handler.attach(bridge, store)

    app = FastTB(title="ISAA UI", app_instance=tb_app)
    app.inject_style = True  # rely on _maybe_inject_style for tbjs-main.css

    # Static mount: relative to this file
    static_dir = Path(__file__).parent / "static"
    app.mount_static("/static", str(static_dir))

    ctx = {
        "isaa": isaa,
        "store": store,
        "bridge": bridge,
        "uploads": uploads,
    }

    # Routes
    chats_routes.register(app, ctx)
    vfs_routes.register(app, ctx)
    agents_routes.register(app, ctx)
    skills_routes.register(app, ctx)
    running_routes.register(app, ctx)
    tools_routes.register(app, ctx)

    # Welcome (greeting + agent list)
    @app.get("/api/welcome")
    async def welcome(request):
        qp = request.query_params
        lang = qp.get("lang", "de")
        if isinstance(lang, list):
            lang = lang[0] if lang else "de"
        g = _greeting(lang=lang)
        names = isaa.config.get("agents-name-list", []) or []
        return {
            "greeting": g.get("greeting", ""),
            "time_interval": g.get("time_interval", ""),
            "agents": names,
            "default_agent": names[0] if names else "self",
        }

    # Index
    @app.get("/")
    async def index():
        idx = static_dir / "index.html"
        return idx.read_text(encoding="utf-8")

    # WebSocket
    @app.websocket("/ws/open_chat")
    class _WS:
        async def on_connect(self, conn_id, session, ctx=None, **kw):
            # Accept either signature; build a WebSocketContext-like accessor.
            wctx = _resolve_ctx(conn_id, session, ctx, **kw)
            await ws_handler.on_connect(wctx)

        async def on_message(self, payload, conn_id, session, request=None, ctx=None, **kw):
            wctx = _resolve_ctx(conn_id, session, ctx, **kw)
            await ws_handler.on_message(payload, wctx)

        async def on_disconnect(self, conn_id, session, ctx=None, **kw):
            wctx = _resolve_ctx(conn_id, session, ctx, **kw)
            await ws_handler.on_disconnect(wctx)

    return app


def _resolve_ctx(conn_id, session, ctx, **kw):
    """Build a context object that exposes `conn_id` and an async `send(frame)`.

    The ToolBoxV2 WS infrastructure passes the WebSocketContext-equivalent
    differently depending on entry point. We need at minimum: conn_id + send.

    Resolution order:
      1. explicit `ctx` kwarg with .send → use directly
      2. `session` if it has .send → it IS the context (FastTB WebSocketContext shape)
      3. fallback: build a wrapper that calls app.ws_send(conn_id, ...)
    """
    if ctx is not None and hasattr(ctx, "send"):
        return ctx
    if session is not None and hasattr(session, "send") and hasattr(session, "conn_id"):
        return session
    # Build minimal fallback using ws_send via app
    from toolboxv2 import get_app
    app = get_app()

    class _MinCtx:
        def __init__(self, cid):
            self.conn_id = cid
        async def send(self, data: dict):
            if hasattr(app, "ws_send"):
                return await app.ws_send(self.conn_id, data)
            return False

    return _MinCtx(conn_id)


# ============================================================================
# Launcher
# ============================================================================

def main(host: str = "127.0.0.1", port: int = 8765) -> None:
    from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
    app = build_app()
    app.inject_style = False
    wsgi_app = FastTBHandler(app).as_wsgi_app(enable_ws=True)

    print(f"[ISAA UI] serving on http://{host}:{port}")
    try:
        from waitress import serve as _serve
        _serve(wsgi_app, host=host, port=port, threads=int(os.getenv("ISAA_UI_THREADS", "32")))
    except ImportError:
        logger.warning("[ISAA UI] waitress not installed, falling back to wsgiref.simple_server")
        from wsgiref.simple_server import make_server
        make_server(host, port, wsgi_app).serve_forever()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.getenv("ISAA_UI_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.getenv("ISAA_UI_PORT", "8765")))
    args = p.parse_args()
    main(host=args.host, port=args.port)
