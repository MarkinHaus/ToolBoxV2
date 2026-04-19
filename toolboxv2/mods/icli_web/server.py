"""
icli_web — standalone FastAPI server.

Two WebSocket endpoints:
    /ws/orb    — browsers (orb UI)
    /ws/icli   — icli process (one connection at a time)

Server is a message router between them.

Run:
    python -m toolboxv2.mods.icli_web.server
    # or
    ICLI_WEB_PORT=5055 python server.py
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import secrets
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI, Depends, Header, Query, HTTPException,
    WebSocket, WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, StreamingResponse

_WEB_DIR = Path(__file__).parent / "web"
_KEY_FILE = Path.home() / ".toolbox" / "icli_web.key"

from toolboxv2 import get_logger

log = get_logger()


# ─── Key ─────────────────────────────────────────────────────────────────────

def load_key() -> str:
    env = os.environ.get("ICLI_WEB_API_KEY", "").strip()
    if env: return env
    try:
        if _KEY_FILE.exists():
            k = _KEY_FILE.read_text().strip()
            if k: return k
    except Exception: pass
    k = secrets.token_urlsafe(32)
    try:
        _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _KEY_FILE.write_text(k)
        if os.name != "nt": os.chmod(_KEY_FILE, 0o600)
    except Exception as e: log.warning("key persist: %s", e)
    return k


# ─── Router: connects orb sessions ↔ icli ────────────────────────────────────

class Router:
    """
    Holds the (one) icli WS, and N orb WS connections.

    Message flow:
        orb → /query → forward {cid, query, agent} to icli
        icli → {cid, type, ...} → look up orb by cid, forward

        icli → broadcast task events → fan out to all SSE monitor subs

    Audio chunks (from icli TTS via WebPlayer) are binary frames. icli sends
    a JSON `{cid, type:"audio", chunk_index}` first, then the binary blob.
    We forward both to the right orb AND keep a ring buffer per cid so a
    reconnecting orb can request resume-from-index.
    """
    def __init__(self):
        self.icli_ws: Optional[WebSocket] = None
        self.orb_by_cid: dict[str, WebSocket] = {}   # cid → orb WS
        self.orbs: set[WebSocket] = set()
        self.monitor_subs: list[asyncio.Queue] = []
        self.task_cache: dict[str, dict] = {}        # task_id → latest state
        # Last audio meta received from icli on this conn — next binary
        # frame is for the cid recorded here. Keyed by id(icli_ws) so
        # a reconnected icli starts fresh.
        self._last_icli_cid: dict[int, str] = {}
        # Ring buffer of recent audio chunks per cid (for reconnect resume).
        # Each entry: {"index": int, "meta": dict, "bytes": bytes}
        self.audio_buffer: dict[str, list[dict]] = {}
        # Highest chunk_index that was acked by the orb (server watermark)
        self.acked_index_by_cid: dict[str, int] = {}

CHUNK_BUFFER_MAX = 64

R = Router()


# ─── App ─────────────────────────────────────────────────────────────────────

def build_app(api_key: str) -> FastAPI:
    app = FastAPI(title="icli_web", version="0.5.0")

    def require_key(
        x_api_key: Optional[str] = Header(None),
        key: Optional[str] = Query(None),
    ):
        supplied = x_api_key or key
        if not supplied or not secrets.compare_digest(supplied, api_key):
            raise HTTPException(401, "invalid api key")

    def check_ws_key(ws: WebSocket) -> bool:
        k = ws.query_params.get("key")
        return bool(k) and secrets.compare_digest(k, api_key)

    # ── HTML pages (open) ────────────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse)
    @app.get("/orb", response_class=HTMLResponse)
    async def orb_page():
        return HTMLResponse((_WEB_DIR / "orb.html").read_text(encoding="utf-8"))

    @app.get("/monitor", response_class=HTMLResponse)
    async def monitor_page():
        return HTMLResponse((_WEB_DIR / "monitor.html").read_text(encoding="utf-8"))

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "icli_connected": R.icli_ws is not None,
            "orb_sessions": len(R.orbs),
            "version": "0.5.0",
        }

    # ── Capabilities (used by orb settings dropdowns) ────────────────────
    # icli publishes these at connect time; we cache and serve them.
    app.state.capabilities = {
        "tts": {"backends": [], "emotions": [], "voices": {},
                "supports_style_prompt": []},
        "stt": {"backends": [], "models": {}},
        "defaults": {},
    }

    @app.get("/capabilities")
    async def capabilities(_=Depends(require_key)):
        return app.state.capabilities

    @app.post("/monitor/clear")
    async def monitor_clear(_=Depends(require_key)):
        """Drop the task cache so new monitor connections start empty."""
        R.task_cache.clear()
        # Tell all existing monitor SSE subscribers to wipe too
        for q in list(R.monitor_subs):
            try: q.put_nowait(json.dumps({"type": "snapshot", "data": []}))
            except asyncio.QueueFull: pass
        return {"cleared": True}

    @app.post("/monitor/clear_done")
    async def monitor_clear_done(_=Depends(require_key)):
        """Drop only terminal tasks (completed/failed/error) from the cache.
        Running tasks stay — client sees them continue streaming."""
        _DONE = {"completed", "failed", "error"}
        for tid, tv in list(R.task_cache.items()):
            if (tv.get("status") or "").lower() in _DONE:
                R.task_cache.pop(tid, None)
        # Re-seed all monitor subscribers with the surviving tasks
        snap = json.dumps(
            {"type": "snapshot", "data": list(R.task_cache.values())},
            default=str,
        )
        for q in list(R.monitor_subs):
            try:
                q.put_nowait(snap)
            except asyncio.QueueFull:
                pass
        return {"cleared": True, "remaining": len(R.task_cache)}

    # ── SSE: task monitor ────────────────────────────────────────────────
    @app.get("/stream/tasks")
    async def stream_tasks(_=Depends(require_key)):
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        R.monitor_subs.append(q)
        snap = json.dumps({"type": "snapshot",
                           "data": list(R.task_cache.values())}, default=str)

        async def gen():
            try:
                yield f"data: {snap}\n\n".encode()
                while True:
                    try:
                        payload = await asyncio.wait_for(q.get(), timeout=15.0)
                        yield f"data: {payload}\n\n".encode()
                    except asyncio.TimeoutError:
                        yield b": ping\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                if q in R.monitor_subs: R.monitor_subs.remove(q)

        return StreamingResponse(
            gen(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache",
                     "X-Accel-Buffering": "no",
                     "Connection": "keep-alive"},
        )

    # ── WS: icli side ────────────────────────────────────────────────────
    @app.websocket("/ws/icli")
    async def ws_icli(ws: WebSocket):
        await ws.accept()
        if not check_ws_key(ws):
            await ws.close(code=4401); return
        if R.icli_ws is not None:
            # A previous icli is still registered. This can happen when
            # the old connection died but the server hasn't run the finally
            # block yet (e.g. OS-level TCP reset while server was awaiting
            # receive). Try to close the stale connection and take over,
            # instead of rejecting the new one.
            stale = R.icli_ws
            log.info("icli reconnect: evicting stale previous connection")
            try:
                await stale.close(code=1000)
            except Exception:
                pass
            R.icli_ws = None
            # Give the old ws_icli finally a moment to run
            await asyncio.sleep(0.1)

        R.icli_ws = ws
        log.info("icli connected")
        # Notify all orbs: icli is live
        await _broadcast_status(True)

        # Start heartbeat: server → icli ping every 15s
        hb_task = asyncio.create_task(_heartbeat(ws, "icli"))

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=45.0)
                except asyncio.TimeoutError:
                    log.warning("icli recv timeout — closing")
                    break
                mt = msg.get("type")
                if mt == "websocket.disconnect": break

                if "bytes" in msg and msg["bytes"] is not None:
                    cid = R._last_icli_cid.pop(id(ws), None)
                    if cid:
                        _buffer_audio_chunk(cid, msg["bytes"])
                        orb = R.orb_by_cid.get(cid)
                        if orb is not None:
                            try: await orb.send_bytes(msg["bytes"])
                            except Exception: pass
                    continue

                text = msg.get("text")
                if not text: continue
                try: data = json.loads(text)
                except json.JSONDecodeError: continue

                await handle_icli_msg(app, data, ws)

        except WebSocketDisconnect:
            pass
        except Exception as e:
            log.exception("ws_icli error: %s", e)
        finally:
            hb_task.cancel()
            R.icli_ws = None
            log.info("icli disconnected")
            # Tell all orbs their in-flight queries are dead
            for cid, orb in list(R.orb_by_cid.items()):
                try:
                    await orb.send_text(json.dumps({
                        "type": "error", "cid": cid,
                        "error": "icli disconnected",
                    }))
                    await orb.send_text(json.dumps(
                        {"type": "done", "cid": cid}))
                except Exception: pass
            R.orb_by_cid.clear()
            R._last_icli_cid.clear()
            R.audio_buffer.clear()
            R.acked_index_by_cid.clear()
            await _broadcast_status(False)

    # ── WS: orb side ─────────────────────────────────────────────────────
    @app.websocket("/ws/orb")
    async def ws_orb(ws: WebSocket):
        await ws.accept()
        if not check_ws_key(ws):
            await ws.close(code=4401); return

        R.orbs.add(ws)
        log.info("orb connected (total=%d)", len(R.orbs))
        await ws.send_text(json.dumps({
            "type": "status",
            "icli_connected": R.icli_ws is not None,
        }))

        hb_task = asyncio.create_task(_heartbeat(ws, "orb"))

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=45.0)
                except asyncio.TimeoutError:
                    log.warning("orb recv timeout — closing")
                    break
                mt = msg.get("type")
                if mt == "websocket.disconnect": break

                if "bytes" in msg and msg["bytes"] is not None:
                    cid = getattr(ws, "_active_cid", None)
                    if R.icli_ws and cid:
                        try:
                            await R.icli_ws.send_text(json.dumps({
                                "type": "audio_chunk_in", "cid": cid,
                            }))
                            await R.icli_ws.send_bytes(msg["bytes"])
                        except Exception: pass
                    continue

                text = msg.get("text")
                if not text: continue
                try: data = json.loads(text)
                except json.JSONDecodeError: continue

                await handle_orb_msg(data, ws)

        except WebSocketDisconnect:
            pass
        except Exception as e:
            log.exception("ws_orb error: %s", e)
        finally:
            hb_task.cancel()
            R.orbs.discard(ws)
            # Cancel all this orb's in-flight cids on icli side
            dead_cids = [cid for cid, w in R.orb_by_cid.items() if w is ws]
            for cid in dead_cids:
                R.orb_by_cid.pop(cid, None)
                if R.icli_ws:
                    try:
                        await R.icli_ws.send_text(json.dumps({
                            "type": "cancel", "cid": cid,
                        }))
                    except Exception: pass
            log.info("orb disconnected (remaining=%d)", len(R.orbs))

    return app


# ─── Heartbeat ───────────────────────────────────────────────────────────────

async def _heartbeat(ws: WebSocket, role: str):
    """Send a tiny JSON ping every 15s. Close on failure."""
    try:
        while True:
            await asyncio.sleep(15.0)
            try:
                await ws.send_text(json.dumps({"type": "ping"}))
            except Exception:
                log.info("%s heartbeat failed, closing", role)
                try: await ws.close()
                except Exception: pass
                return
    except asyncio.CancelledError:
        pass


async def _broadcast_status(icli_connected: bool):
    """Tell every orb whether icli is live."""
    msg = json.dumps({"type": "status", "icli_connected": icli_connected})
    for orb in list(R.orbs):
        try: await orb.send_text(msg)
        except Exception: pass


# ─── Message handlers ────────────────────────────────────────────────────────

async def handle_icli_msg(app: FastAPI, data: dict, icli_ws: WebSocket):
    t = data.get("type")

    if t == "ping" or t == "pong":
        return  # heartbeat, ignore

    if t == "hello":
        # Expected shape: {type:"hello", capabilities:{...}, defaults:{...},
        #                  supports:{...}, active_cids:[{cid, chunk_index}]}
        caps = data.get("capabilities")
        if caps:
            app.state.capabilities = caps
        log.info("icli hello — caps: tts=%d stt=%d backends",
                 len(caps.get("tts",{}).get("backends", [])) if caps else 0,
                 len(caps.get("stt",{}).get("backends", [])) if caps else 0)

        # Inform orbs which cids survived on the icli side. Orbs hold the
        # cid + last_played_index locally; a notification lets them decide
        # to send `action:"resume"` if they care. If icli returned with a
        # FRESH set (old cids gone), dead cids in R.orb_by_cid get cleared.
        alive = {c.get("cid") for c in (data.get("active_cids") or [])}
        if data.get("active_cids") is not None:
            dead = [cid for cid in list(R.orb_by_cid.keys()) if cid not in alive]
            for cid in dead:
                orb = R.orb_by_cid.pop(cid, None)
                R.audio_buffer.pop(cid, None)
                R.acked_index_by_cid.pop(cid, None)
                if orb is not None:
                    try: await orb.send_text(json.dumps({
                        "type": "error", "cid": cid,
                        "error": "session lost on icli restart",
                    }))
                    except Exception: pass
                    try: await orb.send_text(json.dumps(
                        {"type": "done", "cid": cid}))
                    except Exception: pass
        return

    if t == "task":
        # Monitor update: {type:"task", data:{task_id,...}}
        payload = data.get("data") or {}
        tid = payload.get("task_id")
        if tid:
            # Preserve narrator history on top of the replacement —
            # icli re-sends the full task state periodically but never
            # includes narrator_events, so we carry them forward.
            prev = R.task_cache.get(tid)
            if prev and prev.get("narrator_events"):
                payload["narrator_events"] = prev["narrator_events"]
            R.task_cache[tid] = payload
            _broadcast_monitor({"type": "task", "data": payload})
        return

    if t == "narrator_event":
        # Attach an agent-tagged narrator message to its task and push
        # a monitor update. Keeps a bounded ring (most recent 200) so
        # long-running tasks don't grow unbounded in memory.
        tid = data.get("task_id")
        if not tid: return
        task = R.task_cache.get(tid)
        if task is None:
            # Task not registered yet — stage the event so it's not lost.
            # A subsequent "task" update will preserve it via the branch
            # above that carries narrator_events forward.
            task = {"task_id": tid, "narrator_events": []}
            R.task_cache[tid] = task
        events = task.setdefault("narrator_events", [])
        events.append({
            "agent":     data.get("agent"),
            "is_sub":    bool(data.get("is_sub")),
            "iter":      data.get("iter"),
            "persona":   data.get("persona"),
            "text":      data.get("text", ""),
            "timestamp": data.get("timestamp"),
        })
        if len(events) > 200: del events[:len(events) - 200]
        _broadcast_monitor({
            "type": "narrator_event",
            "task_id": tid,
            "event": events[-1],
        })
        return

    # Everything else is per-cid stream reply → route to the right orb
    cid = data.get("cid")
    if not cid: return
    orb = R.orb_by_cid.get(cid)
    if orb is None: return

    if t == "audio":
        # Meta for upcoming binary frame. Stash cid so next binary from
        # this icli conn is routed + buffered under that cid. Also forward
        # the meta JSON (orb uses it to show per-chunk text/emotion).
        R._last_icli_cid[id(icli_ws)] = cid
        # Pre-register the meta entry in the ring buffer; bytes get
        # attached when the next binary frame arrives.
        chunk_idx = data.get("chunk_index")
        if chunk_idx is not None:
            _stage_audio_meta(cid, chunk_idx, data)
        try: await orb.send_text(json.dumps(data))
        except Exception: pass
        return

    # Normal reply chunks (transcription, text_chunk, response, done, error…)
    try: await orb.send_text(json.dumps(data))
    except Exception: pass

    if t in ("done", "error"):
        R.orb_by_cid.pop(cid, None)


async def handle_orb_msg(data: dict, orb_ws: WebSocket):
    action = data.get("action")

    if action == "query":
        # New query from orb. Generate cid, send to icli.
        # First: barge-in — cancel any existing cid for this orb.
        await _cancel_orb_inflight(orb_ws, reason="new query")

        cid = uuid.uuid4().hex[:12]
        R.orb_by_cid[cid] = orb_ws
        orb_ws._active_cid = cid  # type: ignore[attr-defined]

        # Tell orb its cid so it can tag ack/resume/cancel against it
        try: await orb_ws.send_text(json.dumps(
            {"type": "cid_assigned", "cid": cid}))
        except Exception: pass

        msg = {
            "type": "query", "cid": cid,
            "agent": data.get("agent", "self"),
            "query": data.get("query", ""),
            "tts": data.get("tts") or {},
            "stt": data.get("stt") or {},
            "context": data.get("context") or {},
        }
        if not R.icli_ws:
            await orb_ws.send_text(json.dumps({
                "type": "error", "cid": cid,
                "error": "icli not connected",
            }))
            R.orb_by_cid.pop(cid, None)
            return
        try: await R.icli_ws.send_text(json.dumps(msg))
        except Exception as e:
            await orb_ws.send_text(json.dumps({
                "type": "error", "cid": cid, "error": str(e),
            }))

    elif action == "tts_preview":
        # Bypass the agent entirely: orb just wants to hear TTS output for
        # its current settings ("test voice" button). icli will synthesize
        # and stream audio chunks back — no agent_task, no iteration, no
        # tool calls, no Zen+ execution record.
        # Barge-in: cancel any prior cid on this orb first.
        await _cancel_orb_inflight(orb_ws, reason="new tts preview")
        cid = uuid.uuid4().hex[:12]
        R.orb_by_cid[cid] = orb_ws
        orb_ws._active_cid = cid  # type: ignore[attr-defined]
        try: await orb_ws.send_text(json.dumps(
            {"type": "cid_assigned", "cid": cid}))
        except Exception: pass
        msg = {
            "type": "tts_preview", "cid": cid,
            "text": data.get("text") or "Voice preview.",
            "tts": data.get("tts") or {},
        }
        if not R.icli_ws:
            await orb_ws.send_text(json.dumps({
                "type": "error", "cid": cid,
                "error": "icli not connected",
            }))
            R.orb_by_cid.pop(cid, None)
            return
        try: await R.icli_ws.send_text(json.dumps(msg))
        except Exception as e:
            await orb_ws.send_text(json.dumps({
                "type": "error", "cid": cid, "error": str(e),
            }))

    elif action == "audio_start":
        # Orb announces it will stream audio for a new cid (server STT path).
        # Barge-in: cancel prior cid on this orb first.
        await _cancel_orb_inflight(orb_ws, reason="new audio")
        cid = uuid.uuid4().hex[:12]
        R.orb_by_cid[cid] = orb_ws
        orb_ws._active_cid = cid  # type: ignore[attr-defined]
        if R.icli_ws:
            try:
                await R.icli_ws.send_text(json.dumps({
                    "type": "audio_start", "cid": cid,
                    "agent": data.get("agent", "self"),
                    "tts": data.get("tts") or {},
                    "stt": data.get("stt") or {},
                    "context": data.get("context") or {},
                }))
            except Exception: pass
        # Tell orb its cid so UI can correlate
        await orb_ws.send_text(json.dumps({"type": "cid_assigned", "cid": cid}))

    elif action == "audio_end":
        cid = getattr(orb_ws, "_active_cid", None)
        if R.icli_ws and cid:
            try:
                await R.icli_ws.send_text(json.dumps({
                    "type": "audio_end", "cid": cid,
                }))
            except Exception: pass

    elif action == "stop_tts":
        # Stop the currently speaking TTS but KEEP the agent task running.
        cid = data.get("cid") or getattr(orb_ws, "_active_cid", None)
        if R.icli_ws and cid:
            try:
                await R.icli_ws.send_text(json.dumps({
                    "type": "stop_tts", "cid": cid,
                }))
            except Exception: pass

    elif action == "test_tts":
        # Direct TTS bypass: speak literal text with current TTS config,
        # skipping the agent. Used for Settings > test voice.
        await _cancel_orb_inflight(orb_ws, reason="test tts")
        cid = uuid.uuid4().hex[:12]
        R.orb_by_cid[cid] = orb_ws
        orb_ws._active_cid = cid  # type: ignore[attr-defined]
        try: await orb_ws.send_text(json.dumps(
            {"type": "cid_assigned", "cid": cid}))
        except Exception: pass
        if not R.icli_ws:
            await orb_ws.send_text(json.dumps({
                "type": "error", "cid": cid, "error": "icli not connected"}))
            R.orb_by_cid.pop(cid, None)
            return
        try:
            await R.icli_ws.send_text(json.dumps({
                "type": "test_tts", "cid": cid,
                "text": data.get("text", "Voice check."),
                "tts": data.get("tts") or {},
            }))
        except Exception as e:
            await orb_ws.send_text(json.dumps({
                "type": "error", "cid": cid, "error": str(e)}))

    elif action == "cancel":
        # Full barge-in: kill TTS AND the agent task for this cid.
        cid = data.get("cid") or getattr(orb_ws, "_active_cid", None)
        if cid:
            await _cancel_cid(cid, reason="user cancel")

    elif action == "ack":
        # Orb confirms it played up to chunk_index (inclusive).
        # We can drop older buffered chunks for this cid to free memory.
        cid = data.get("cid")
        idx = data.get("chunk_index")
        if cid and idx is not None:
            R.acked_index_by_cid[cid] = max(
                R.acked_index_by_cid.get(cid, -1), int(idx))
            buf = R.audio_buffer.get(cid)
            if buf:
                R.audio_buffer[cid] = [e for e in buf if e["index"] > int(idx)]

    elif action == "resume":
        # Reconnecting orb asks to resume a cid it was previously on.
        # Shape: {"action":"resume", "cid":"...", "from_index": 4}
        cid = data.get("cid")
        from_idx = int(data.get("from_index") or 0)
        if not cid: return
        # Re-bind this cid to the new orb socket.
        R.orb_by_cid[cid] = orb_ws
        orb_ws._active_cid = cid  # type: ignore[attr-defined]
        sent = await _resend_buffered(cid, orb_ws, from_idx)
        try: await orb_ws.send_text(json.dumps(
            {"type": "resumed", "cid": cid, "chunks": sent}))
        except Exception: pass


async def _cancel_cid(cid: str, reason: str = "") -> None:
    """Send cancel to icli, clean up server state for this cid."""
    if R.icli_ws:
        try:
            await R.icli_ws.send_text(json.dumps({
                "type": "cancel", "cid": cid,
            }))
        except Exception: pass
    R.orb_by_cid.pop(cid, None)
    R.audio_buffer.pop(cid, None)
    R.acked_index_by_cid.pop(cid, None)


async def _cancel_orb_inflight(orb_ws: WebSocket, reason: str = "") -> None:
    """Cancel every cid currently bound to this orb."""
    dead = [cid for cid, w in R.orb_by_cid.items() if w is orb_ws]
    for cid in dead:
        await _cancel_cid(cid, reason=reason)


def _stage_audio_meta(cid: str, index: int, meta: dict) -> None:
    """Store meta for an incoming audio chunk; bytes get attached later."""
    buf = R.audio_buffer.setdefault(cid, [])
    buf.append({"index": index, "meta": meta, "bytes": None})
    if len(buf) > CHUNK_BUFFER_MAX:
        del buf[: len(buf) - CHUNK_BUFFER_MAX]


def _buffer_audio_chunk(cid: str, payload: bytes) -> None:
    """Attach binary to the most recent meta entry for cid (the one staged
    by the preceding 'audio' JSON frame)."""
    buf = R.audio_buffer.get(cid)
    if not buf: return
    # Find newest meta entry with no bytes yet
    for entry in reversed(buf):
        if entry["bytes"] is None:
            entry["bytes"] = payload
            return


async def _resend_buffered(cid: str, orb_ws: WebSocket, from_index: int) -> int:
    """Re-send all buffered chunks with index >= from_index. Returns count."""
    buf = R.audio_buffer.get(cid) or []
    sent = 0
    for entry in buf:
        if entry["index"] < from_index: continue
        if entry["bytes"] is None: continue
        try:
            await orb_ws.send_text(json.dumps(entry["meta"]))
            await orb_ws.send_bytes(entry["bytes"])
            sent += 1
        except Exception: break
    return sent


def _broadcast_monitor(event: dict):
    payload = json.dumps(event, default=str)
    for q in list(R.monitor_subs):
        try: q.put_nowait(payload)
        except asyncio.QueueFull: pass


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    host = os.environ.get("ICLI_WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("ICLI_WEB_PORT", "5055"))
    key = load_key()
    os.environ["ICLI_WEB_API_KEY"] = key  # propagate if reloaded

    app = build_app(key)

    try: import uvicorn
    except ImportError:
        log.error("pip install fastapi uvicorn"); sys.exit(2)

    log.info("icli_web → http://%s:%d/?key=%s", host, port, key)
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
