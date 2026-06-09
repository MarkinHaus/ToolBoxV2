"""
stream_bridge.py — FlowAgent.a_stream → WS frames.

Each chunk yielded by `agent.a_stream(...)` is:
  1. Assigned step_id (stable across reconnects, per chat).
  2. Persisted to chat_store (gets seq).
  3. Broadcast to all WS connections subscribed to this chat.

Concurrency: one running stream per chat. Cancel/pause via tracker.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("isaa.ui.bridge")


def _ui_templates_dir(store, agent_name: str) -> Path:
    """Per-agent dir for persisted custom widget templates (sibling of chats dir)."""
    return Path(store.root).parent / "templates" / (agent_name or "_")


def _load_agent_templates(store, agent_name: str) -> list[dict]:
    """Read persisted templates → list of template_register frames."""
    d = _ui_templates_dir(store, agent_name)
    frames: list[dict] = []
    if not d.exists():
        return frames
    for fp in d.glob("*.json"):
        try:
            spec = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        frames.append({
            "type": "template_register",
            "template_id": spec.get("template_id") or fp.stem,
            "name": spec.get("name", fp.stem),
            "adapter": spec.get("adapter", "html"),
            "schema": spec.get("schema", {}),
            "render_js": spec.get("render_js"),
        })
    return frames

# Frame types that are turn-terminal (close the running stream tracker).
TERMINAL_TYPES = frozenset({"done", "max_iterations", "cancelled", "paused", "error"})


def make_step_id(run_id: str, iteration: int, sub_idx: int = 0) -> str:
    """Stable step id for navigation/rollback."""
    return f"{run_id}:{iteration}:{sub_idx}"


@dataclass
class RunningStream:
    chat_id: str
    agent_name: str
    task: asyncio.Task | None = None
    run_id: str | None = None
    current_iter: int = 0
    sub_idx: int = 0
    started_at: float = 0.0


@dataclass
class BridgeBroadcaster:
    """Callable that pushes a frame to all WS clients of this chat."""
    send: Callable[[str, dict], Awaitable[Any]]


class StreamBridge:
    """Owns the running streams. One running stream per chat allowed."""

    def __init__(self, isaa_mod, chat_store, broadcaster: BridgeBroadcaster):
def __init__(self, isaa_mod, chat_store, broadcaster: BridgeBroadcaster, global_vars=None):
        self.isaa = isaa_mod
        self.store = chat_store
        self.global_vars = global_vars
        self.broadcast = broadcaster.send
        self._running: dict[str, RunningStream] = {}
        self._lock = asyncio.Lock()
    def is_running(self, chat_id: str) -> bool:
        rs = self._running.get(chat_id)
        return rs is not None and rs.task is not None and not rs.task.done()

    def get_run_id(self, chat_id: str) -> str | None:
        rs = self._running.get(chat_id)
        return rs.run_id if rs else None

    async def start(
        self,
        chat_id: str,
        agent_name: str,
        text: str,
        attachments: list[dict] | None = None,
    ) -> bool:
        """Start a new stream for this chat. Returns False if one is already running."""
        async with self._lock:
            if self.is_running(chat_id):
                return False

            # Persist the user message first.
            user_frame = {
                "type": "user_msg",
                "text": text,
                "attachments": attachments or [],
                "agent": agent_name,
            }
            seq = self.store.append(chat_id, user_frame)
            user_frame["seq"] = seq
            await self.broadcast(chat_id, user_frame)

            rs = RunningStream(chat_id=chat_id, agent_name=agent_name)
            rs.started_at = asyncio.get_event_loop().time()
            rs.task = asyncio.create_task(self._run(rs, text, attachments or []))
            self._running[chat_id] = rs
            return True

    async def _run(self, rs: RunningStream, text: str, attachments: list[dict]) -> None:
        chat_id = rs.chat_id
        try:
            agent = await self.isaa.get_agent(rs.agent_name)
        except Exception as e:
            logger.exception("[bridge] get_agent failed")
            await self._emit(chat_id, {"type": "error", "error": f"get_agent: {e}"})
            await self._emit(chat_id, {"type": "done", "success": False, "final_answer": ""})
            self._running.pop(chat_id, None)
            return
        try:
            from toolboxv2.mods.isaa.extras.live_obs_server import register_agent_obs
            register_agent_obs(agent)
        except Exception as e:
            logger.exception("[bridge] live_obs_server failed")
            await self._emit(chat_id, {"type": "error", "error": f"register_agent_obs: {e}"})

        # Register UI tools (idempotent per agent instance)
_register_ui_tools(agent, self, self.store, self.global_vars)
            _register_ui_tools(agent, self, self.store)
        except Exception:
            logger.exception("[bridge] register_ui_tools failed (non-fatal)")

        # Re-broadcast persisted custom templates so they survive across chats.
        try:
            _ag_name = getattr(getattr(agent, "amd", None), "name", "") or rs.agent_name
            for _tf in _load_agent_templates(self.store, _ag_name):
                await self.broadcast(chat_id, _tf)
        except Exception:
            logger.exception("[bridge] template replay failed (non-fatal)")

        # Build query — append upload paths if any.
        query = text
        for att in attachments:
            vfs_path = att.get("vfs_path")
            if vfs_path:
                query += f"\n\n[Attached: {vfs_path}]"

        try:
            engine = agent._get_execution_engine() if hasattr(agent, "_get_execution_engine") else None

            async for chunk in agent.a_stream(query=query, session_id=chat_id):
                # First pass: try to learn the run_id from the engine session map.
                if rs.run_id is None and engine is not None:
                    try:
                        rs.run_id = engine._session_last_run.get(chat_id)
                    except AttributeError:
                        rs.run_id = None
                # Some chunks carry run_id explicitly (paused/cancelled).
                if "run_id" in chunk and chunk["run_id"]:
                    rs.run_id = chunk["run_id"]

                # Track iter for step_id
                if "iter" in chunk:
                    try:
                        rs.current_iter = int(chunk["iter"])
                    except (TypeError, ValueError):
                        pass

                # Assign step_id (only for chunks that have content the UI will anchor to)
                ctype = chunk.get("type", "")
                if rs.run_id and ctype in (
                    "iteration_start", "content", "reasoning",
                    "tool_start", "tool_result", "final_answer",
                    "max_iterations", "paused", "cancelled", "done",
                ):
                    chunk.setdefault("step_id", make_step_id(rs.run_id, rs.current_iter, rs.sub_idx))
                    if ctype == "tool_start":
                        rs.sub_idx += 1
                    if ctype == "iteration_start":
                        rs.sub_idx = 0

                await self._emit(chat_id, chunk)

                if ctype in TERMINAL_TYPES:
                    # Done emitted by the engine itself for normal completion.
                    pass

        except asyncio.CancelledError:
            await self._emit(chat_id, {"type": "cancelled", "run_id": rs.run_id or "", "answer": ""})
            await self._emit(chat_id, {"type": "done", "success": False, "final_answer": ""})
            raise
        except Exception as e:
            logger.exception("[bridge] stream failed")
            await self._emit(chat_id, {"type": "error", "error": str(e)})
            await self._emit(chat_id, {"type": "done", "success": False, "final_answer": ""})
        finally:
            # Persist run_id pointer on meta for future pause/resume.
            if rs.run_id:
                try:
                    self.store.update_meta(chat_id, run_id=rs.run_id)
                except Exception:
                    pass
            self._running.pop(chat_id, None)

    async def _emit(self, chat_id: str, frame: dict) -> None:
        """Persist + broadcast a single frame."""
        seq = self.store.append(chat_id, frame)
        frame = dict(frame)
        frame["seq"] = seq
        try:
            await self.broadcast(chat_id, frame)
        except Exception:
            logger.exception("[bridge] broadcast failed")

    async def pause(self, chat_id: str) -> bool:
        rs = self._running.get(chat_id)
        if not rs or not rs.run_id:
            return False
        try:
            agent = await self.isaa.get_agent(rs.agent_name)
            await agent.pause_execution(rs.run_id)
            return True
        except Exception:
            logger.exception("[bridge] pause failed")
            return False

    async def cancel(self, chat_id: str) -> bool:
        rs = self._running.get(chat_id)
        if not rs:
            return False
        if rs.run_id:
            try:
                agent = await self.isaa.get_agent(rs.agent_name)
                await agent.cancel_execution(rs.run_id)
            except Exception:
                logger.exception("[bridge] engine cancel failed; falling back to task cancel")
                if rs.task:
                    rs.task.cancel()
        elif rs.task:
            rs.task.cancel()
        return True

    async def shutdown(self) -> None:
        for cid, rs in list(self._running.items()):
            if rs.task:
                rs.task.cancel()
        self._running.clear()


# ============================================================================
# UI Tools on the agent (tasks 20, 21, 22)
# ============================================================================

def _register_ui_tools(agent, bridge: "StreamBridge", store, global_vars=None) -> None:
    """Register create_widget / update_widget / close_widget / set_var / get_var
    / define_template on the agent. Idempotent — flag-guarded per instance.

    Tools read `agent.active_session` to know which chat to broadcast to.
    """
    if getattr(agent, "_isaa_ui_tools_registered", False):
        return
    if not hasattr(agent, "add_tool"):
        return  # not a FlowAgent

    agent_name = getattr(getattr(agent, "amd", None), "name", "") or ""

    async def create_widget(template_id: str, props: dict, pin: dict | None = None) -> str:
        """Create a UI widget visible in the chat. Returns widget_id.

        template_id: built-in adapter ('markdown', 'vega', 'code', 'table', 'form')
                     or a previously registered custom template id.
        props: data for the adapter (e.g. {'spec': <vega-lite>} or {'md': '...'}).
        pin: optional {x, y, w, h} to fix size/position (also makes it non-draggable).
        """
        chat_id = getattr(agent, "active_session", None)
        if not chat_id:
            return "error: no active session"
        wid = uuid.uuid4().hex[:12]
        frame = {
            "type": "widget_create",
            "widget_id": wid,
            "template": template_id,
            "props": props or {},
            "pin": pin,
        }
        await bridge._emit(chat_id, frame)
        return wid

    async def update_widget(widget_id: str, props_patch: dict) -> bool:
        """Update a previously created widget with patched props."""
        chat_id = getattr(agent, "active_session", None)
        if not chat_id:
            return False
        await bridge._emit(chat_id, {
            "type": "widget_update",
            "widget_id": widget_id,
            "props_patch": props_patch or {},
        })
        return True

    async def close_widget(widget_id: str) -> bool:
        """Close a widget and remove it from the chat UI."""
        chat_id = getattr(agent, "active_session", None)
        if not chat_id:
            return False
        await bridge._emit(chat_id, {
            "type": "widget_close",
            "widget_id": widget_id,
        })
        return True

    async def set_var(scope: str, key: str, value) -> bool:
        """Set a variable. scope is 'agent' or 'global'. Both are server-wide.
        Broadcast to all widgets so they can re-render.
        """
        chat_id = getattr(agent, "active_session", None)
        if not chat_id:
            return False
        # Persist to global store (not per-chat)
        if global_vars is not None:
            global_vars.set(scope, key, value)
        await bridge._emit(chat_id, {
            "type": "var_set", "scope": scope, "key": key, "value": value,
        })
        return True
        })
        return True

    async def get_var(scope: str, key: str):
        """Read a variable. scope is 'agent' or 'global'. Reads from global store."""
        if global_vars is not None:
            return global_vars.get(scope, key)
        return None
    async def define_template(name: str, adapter: str, schema: dict,
                               render_js: str | None = None) -> str:
        """Register a custom widget template.

        adapter: one of 'markdown', 'vega', 'code', 'table', 'form', 'html'.
                 For 'html', render_js must be provided (runs in a sandboxed iframe).
        schema:  JSON-schema describing the props the template accepts.
        """
        chat_id = getattr(agent, "active_session", None)
        if not chat_id:
            return "error: no active session"
        template_id = name.lower().replace(" ", "_").replace("-", "_")
        try:
            _d = _ui_templates_dir(store, agent_name)
            _d.mkdir(parents=True, exist_ok=True)
            (_d / f"{template_id}.json").write_text(json.dumps({
                "template_id": template_id, "name": name, "adapter": adapter,
                "schema": schema or {}, "render_js": render_js,
            }), encoding="utf-8")
        except Exception:
            logger.exception("[bridge] template persist failed (non-fatal)")
        await bridge._emit(chat_id, {
            "type": "template_register",
            "template_id": template_id,
            "name": name,
            "adapter": adapter,
            "schema": schema or {},
            "render_js": render_js,
        })
        return template_id

    for fn, name, desc in [
        (create_widget, "create_widget",
         "Create a UI widget displayed in the chat (e.g. chart, table, form). Returns widget_id."),
        (update_widget, "update_widget",
         "Update a previously created widget with a props patch."),
        (close_widget, "close_widget",
         "Close a widget and remove it from the UI."),
        (set_var, "set_var",
         "Set a variable for widgets to read. scope='agent'|'global'. "
         "WARNING: keys prefixed 'storage:' belong to interactive html-doc widgets "
         "(their window.storage state, scope='agent', per chat). You MAY read/edit them, "
         "but the value is a JSON STRING in that page's own schema — parse before reading, "
         "and write back the SAME shape, or you will corrupt the page state."),
        (get_var, "get_var",
         "Read a variable set by set_var. scope='agent'|'global'. "
         "Keys prefixed 'storage:' are html-doc widget state (JSON string, scope='agent'); "
         "json.loads the returned value before using it."),
        (define_template, "define_template",
         "Register a custom widget template. Returns template_id."),
    ]:
        try:
            agent.add_tool(fn, name=name, description=desc, category=["ui"])
        except Exception as e:
            logger.warning("[bridge] add_tool %s failed: %s", name, e)

    agent._isaa_ui_tools_registered = True
    logger.info("[bridge] UI tools registered on agent=%s",
                getattr(agent.amd, "name", "?") if hasattr(agent, "amd") else "?")
