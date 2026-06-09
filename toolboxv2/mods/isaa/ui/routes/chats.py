"""
Chat REST routes.

Mounted by ui/app.py. All handlers async.
"""
from __future__ import annotations

from typing import Any


def register(app, ctx):
    """ctx provides: store, bridge, isaa."""
def register(app, ctx):
    """ctx provides: store, bridge, isaa, global_vars."""
    store = ctx["store"]
    bridge = ctx["bridge"]
    global_vars = ctx.get("global_vars")
    isaa = ctx["isaa"]

    @app.get("/api/chats")
    async def list_chats():
        return store.list()

    @app.post("/api/chats")
    async def create_chat(request):
        body = request.json_data or {}
        agent_name = body.get("agent") or _default_agent_name(isaa)
        title = body.get("title", "")
        meta = store.create(agent=agent_name, title=title)
        return {
            "chat_id": meta.chat_id,
            "session_id": meta.session_id,
            "agent": meta.agent,
            "title": meta.title,
        }

    @app.get("/api/chats/{chat_id}")
    async def get_chat(chat_id: str):
        meta = store.get_meta(chat_id)
        if not meta:
            return (404, {"error": "Not found"})
        return {
            "chat_id": meta.chat_id,
            "title": meta.title,
            "agent": meta.agent,
            "session_id": meta.session_id,
            "run_id": meta.run_id,
            "messages": store.read_all(chat_id),
            "ui": meta.ui,
            "is_running": bridge.is_running(chat_id),
        }

    @app.delete("/api/chats/{chat_id}")
    async def delete_chat(chat_id: str):
        await bridge.cancel(chat_id)
        ok = store.delete(chat_id)
        return {"ok": ok}

    @app.put("/api/chats/{chat_id}")
    async def update_chat(chat_id: str, request):
        body = request.json_data or {}
        allowed = {"title", "ui"}
        fields = {k: v for k, v in body.items() if k in allowed}
        meta = store.update_meta(chat_id, **fields)
        if not meta:
            return (404, {"error": "Not found"})
        return {"ok": True}

    @app.post("/api/chats/{chat_id}/rollback")
    async def rollback(chat_id: str, request):
        body = request.json_data or {}
        step_id = body.get("step_id")
        if not step_id:
            return (400, {"error": "step_id required"})
        # Cancel any running stream first.
        if bridge.is_running(chat_id):
            await bridge.cancel(chat_id)
        new_last = store.truncate_after(chat_id, step_id)
        if new_last < 0:
            return (404, {"error": "step_id not found"})
        # Drop run_id pointer — next message will start a fresh execution.
        # Engine-side checkpoints for the dropped run are orphaned but harmless;
        # the ObservabilityLayer cleans stale live_*.jsonl on next begin_run.
        store.update_meta(chat_id, run_id=None)
        return {"ok": True, "new_last_seq": new_last}
return {"ok": True, "new_last_seq": new_last}

    # --- Global variables endpoint ---
    @app.get("/api/global-vars")
    async def get_global_vars():
        if global_vars is None:
            return {"vars_agent": {}, "vars_global": {}}
        return global_vars.get_all()


def _default_agent_name(isaa) -> str:
def _default_agent_name(isaa) -> str:
    """Pick an existing agent or fall back to 'self'."""
    try:
        names = isaa.config.get("agents-name-list", [])
        if names:
            return names[0]
    except Exception:
        pass
    return "self"
