"""
live_obs_server.py — FastTB live observability server (WS push, no polling).

Single-page operator UI showing all registered agents grouped:
  - Live runs (◐ pulsing) with steps appended as they arrive
  - Completed run history + interrupted runs
  - Per-step detail identical to obs_viewer (timing decomp, LLM I/O, tool calls, VFS)

Style + step renderer reuse obs_viewer.py (CSS extracted, not duplicated).

Wiring:
    from toolboxv2.mods.isaa.extras.live_obs_server import app, register_agent_obs
    register_agent_obs(my_flow_agent)        # for each agent

Standalone:
    uvicorn toolboxv2.mods.isaa.extras.live_obs_server:wsgi_app --port 8000

HTTPWorker:
    worker.run(fast_tb_app=app)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, TypeVar

from toolboxv2 import get_logger, get_app
from toolboxv2.mods.isaa.extras.obs_viewer import _TEMPLATE as _VIEWER_TEMPLATE
from toolboxv2.utils.workers.fast_tb import FastTB
from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler


ObservabilityLayer = TypeVar('ObservabilityLayer')
logger = get_logger()

app = FastTB(title="ISAA Live OBS")

# Extract CSS from obs_viewer template — single source of truth
_css_m = re.search(r"<style>(.*?)</style>", _VIEWER_TEMPLATE, re.DOTALL)
_VIEWER_CSS = _css_m.group(1) if _css_m else ""


# =============================================================================
# HUB — patches registered ObservabilityLayers to broadcast over WS
# =============================================================================


class LiveObsHub:
    _instance: "LiveObsHub | None" = None
    CHANNEL = "/ws/openLive"

    def __init__(self):
        self._registered: dict[str, ObservabilityLayer] = {}
        self._app: FastTB | None = None
        # Disk-scan fallback state
        self._scan_roots: list[str] = []
        self._scan_interval: float = 2.0
        self._scan_task: asyncio.Task | None = None
        self._scan_thread: Any = None  # threading.Thread when no asyncio loop avail
        self._handler_loop: asyncio.AbstractEventLoop | None = None
        self._live_wired: set[str] = set()  # agents with live on_step wrap
        # Per-agent scan state: line counts of live files + known completed run_ids
        self._scan_state: dict[str, dict] = {}
        # {agent: {"live": {run_id: line_count}, "completed": set[run_id]}}
        # agent_name -> VFS instance (enables revert actions; None for shadow agents)
        self._vfs_map: dict[str, Any] = {}

    @classmethod
    def get(cls) -> "LiveObsHub":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def attach_app(self, fast_app: FastTB) -> None:
        self._app = fast_app

    def register(self, obs: ObservabilityLayer) -> None:
        agent = obs.agent_name
        if agent in self._live_wired:
            return
        # May overwrite a disk-discovered shadow entry
        self._registered[agent] = obs
        self._live_wired.add(agent)

        # ---- on_step wrap ----
        prev_on_step = obs.on_step

        async def hub_on_step(step_dict: dict) -> None:
            await self._broadcast({"type": "step", "agent_name": agent, **step_dict})
            if prev_on_step:
                try:
                    await prev_on_step(step_dict)
                except Exception as e:
                    logger.debug(f"[LiveObsHub] prev on_step error: {e}")

        obs.on_step = hub_on_step

        # ---- begin_run wrap ----
        prev_begin = obs.begin_run

        def hub_begin(run_id, query, session_id="", persona="", skills=None,
                      is_resume=False, parent_run_id=""):
            prev_begin(run_id, query, session_id, persona, skills,
                       is_resume=is_resume, parent_run_id=parent_run_id)
            current = obs.active_run(run_id)
            self._fire({
                "type": "run_start",
                "agent_name": agent,
                "run_id": run_id,
                "query": query,
                "session_id": session_id,
                "persona": persona,
                "skills_matched": skills or [],
                "t_start": current.t_start if current else time.time(),
                "is_resume": is_resume,
                "parent_run_id": parent_run_id,
                "is_sub_agent": bool(parent_run_id),
            })

        obs.begin_run = hub_begin

        # ---- end_run wrap ----
        prev_end = obs.end_run

        def hub_end(success: bool, final_answer: str = ""):
            run = obs.active_run()
            rid = run.run_id if run else None
            parent_rid = run.parent_run_id if run else ""
            prev_end(success, final_answer)
            final_dict = None
            if rid:
                reloaded = obs.get_run(rid)
                if reloaded:
                    final_dict = reloaded.to_dict()
            self._fire({
                "type": "run_end",
                "agent_name": agent,
                "success": success,
                "parent_run_id": parent_rid,
                "is_sub_agent": bool(parent_rid),
                "run": final_dict or {"run_id": rid, "success": success, "steps": []},
            })

        obs.end_run = hub_end

    def snapshot(self) -> dict:
        agents: dict[str, Any] = {}
        for name, obs in self._registered.items():
            try:
                runs = obs.list_runs()
                interrupted = obs.get_interrupted_runs()
                # Enrich with mtime so client can render freshness
                for ir in interrupted:
                    try:
                        ir["mtime"] = os.path.getmtime(ir.get("live_file", ""))
                    except OSError:
                        ir["mtime"] = 0
                # Active runs: with parallel sub-agents there can be several.
                # Expose the top-level run as active_run (back-compat) and the
                # full set as active_runs so the UI can nest sub-agents.
                _actives = obs.active_runs()
                _top = next((r for r in _actives if not r.parent_run_id), None)
                active = _top.to_dict() if _top else None
                active_all = [r.to_dict() for r in _actives]
            except Exception as e:
                logger.warning(f"[LiveObsHub] snapshot {name} failed: {e}")
                runs, interrupted, active, active_all = [], [], None, []
            agents[name] = {
                "agent_name": name,
                "obs_dir": obs.obs_dir,
                "runs": runs,
                "interrupted": interrupted,
                "active_run": active,
                "active_runs": active_all,
            }
        return {"agents": agents, "ts": time.time(),
                "vfs_capable": list(self._vfs_map.keys())}

    async def _broadcast(self, msg: dict) -> None:
        if self._app is None:
            return
        # If we're running outside the handler's loop (scanner thread),
        # marshall the broadcast over to the handler loop where the WS
        # connections live.
        handler_loop = self._handler_loop
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if handler_loop is not None and current is not handler_loop:
            try:
                asyncio.run_coroutine_threadsafe(self._do_broadcast(msg), handler_loop)
            except Exception as e:
                logger.debug(f"[LiveObsHub] cross-loop schedule failed: {e}")
            return
        await self._do_broadcast(msg)

    async def _do_broadcast(self, msg: dict) -> None:
        try:
            res = self._app.ws_broadcast(self.CHANNEL, msg)
            if asyncio.iscoroutine(res):
                await res
        except Exception as e:
            logger.debug(f"[LiveObsHub] broadcast error: {e}")

    def _fire(self, msg: dict) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._broadcast(msg))
        except RuntimeError:
            pass  # no loop active (e.g. test setup)

    # =========================================================================
    # DISK SCANNER (fallback when no live-registered agents)
    # =========================================================================

    def add_scan_root(self, root: str) -> None:
        """Add a directory whose subfolders contain agent obs dirs.
        Convention: <root>/<agent_name>/obs/{_index.json,run_*.json,live_*.jsonl}
        """
        root = os.path.abspath(root)
        if root not in self._scan_roots:
            self._scan_roots.append(root)

    def set_scan_interval(self, seconds: float) -> None:
        self._scan_interval = max(0.5, float(seconds))

    def _ensure_scanner(self) -> None:
        """Start the periodic disk-scan in a dedicated background thread.
        Thread-based so it works under both ASGI (uvicorn) and WSGI (waitress)
        and survives WS-handler lifecycle boundaries.
        """
        if not self._scan_roots:
            return
        # Already alive?
        t = self._scan_thread
        if t is not None and getattr(t, "is_alive", lambda: False)():
            return
        import threading

        def _thread_main():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._scanner_loop())
            except Exception as e:
                logger.debug(f"[LiveObsHub] scanner thread crashed: {e}")
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

        self._scan_thread = threading.Thread(
            target=_thread_main, name="LiveObsScanner", daemon=True
        )
        self._scan_thread.start()

    async def _scanner_loop(self) -> None:
        # Initial fast pass, then steady cadence
        try:
            await self._scan_once()
        except Exception as e:
            logger.debug(f"[LiveObsHub] initial scan error: {e}")
        while True:
            try:
                await asyncio.sleep(self._scan_interval)
                await self._scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(f"[LiveObsHub] scan loop error: {e}")

    def _discover_agents(self, root: str) -> dict[str, str]:
        """Returns {agent_name: obs_dir} for subdirs matching the convention."""
        out: dict[str, str] = {}
        try:
            entries = os.listdir(root)
        except OSError as e:
            return out
        for name in entries:
            obs_dir = os.path.join(root, name, "obs")
            if not os.path.isdir(obs_dir):
                continue
            try:
                files = os.listdir(obs_dir)
            except OSError as e:
                continue
            if any(
                f == "_index.json" or f.startswith("live_") or f.startswith("run_")
                for f in files
            ):
                out[name] = obs_dir
        return out

    def _count_jsonl_lines(self, path: str) -> int:
        try:
            with open(path, "rb") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    def _read_jsonl_from(self, path: str, from_line: int) -> list[dict]:
        out: list[dict] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for i, raw in enumerate(f):
                    if i < from_line:
                        continue
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        out.append(json.loads(raw))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
        return out

    async def _scan_once(self) -> None:
        """One scan pass — diff disk against in-memory scan_state, broadcast deltas."""
        # 1) Discover agents across all configured roots
        discovered: dict[str, str] = {}
        for root in self._scan_roots:
            for name, obs_dir in self._discover_agents(root).items():
                discovered.setdefault(name, obs_dir)
        for name, obs_dir in discovered.items():
            # Skip agents wired live — they push via callbacks
            if name in self._live_wired:
                continue

            # Ensure ObservabilityLayer shadow registered for snapshot/get_run
            if name not in self._registered:
                try:
                    from toolboxv2.mods.isaa.base.Agent.observability import ObservabilityLayer
                    shadow = ObservabilityLayer(agent_name=name, obs_dir=obs_dir)
                except Exception as e:
                    logger.debug(f"[LiveObsHub] shadow obs {name} failed: {e}")
                    continue
                self._registered[name] = shadow
                self._scan_state.setdefault(
                    name, {"live": {}, "completed": set()}
                )
                # Tell connected clients about the new agent
                try:
                    runs = shadow.list_runs()
                    interrupted = shadow.get_interrupted_runs()
                    for ir in interrupted:
                        try:
                            ir["mtime"] = os.path.getmtime(ir.get("live_file", ""))
                        except OSError:
                            ir["mtime"] = 0
                except Exception:
                    runs, interrupted = [], []
                # Seed completed set so we don't re-broadcast existing runs as new
                self._scan_state[name]["completed"] = {
                    r.get("run_id") for r in runs if r.get("run_id")
                }
                await self._broadcast({
                    "type": "agent_added",
                    "agent_name": name,
                    "obs_dir": obs_dir,
                    "runs": runs,
                    "interrupted": interrupted,
                    "active_run": None,
                })

            state = self._scan_state.setdefault(
                name, {"live": {}, "completed": set()}
            )

            # 2) Diff live_*.jsonl files
            try:
                files = os.listdir(obs_dir)
            except OSError:
                continue

            live_files = [f for f in files if f.startswith("live_") and f.endswith(".jsonl")]
            seen_live_ids: set[str] = set()
            for fname in live_files:
                run_id = fname[len("live_"):-len(".jsonl")]
                seen_live_ids.add(run_id)
                path = os.path.join(obs_dir, fname)
                current_lines = self._count_jsonl_lines(path)
                last_lines = state["live"].get(run_id)

                if last_lines is None:
                    # New live run discovered
                    state["live"][run_id] = 0
                    # Try to read first line for t_start
                    first = self._read_jsonl_from(path, 0)
                    t_start = first[0].get("t_start") if first else time.time()
                    await self._broadcast({
                        "type": "run_start",
                        "agent_name": name,
                        "run_id": run_id,
                        "query": "(disk-scan)",
                        "session_id": "",
                        "persona": "",
                        "skills_matched": [],
                        "t_start": t_start,
                    })
                    last_lines = 0

                if current_lines > last_lines:
                    new_steps = self._read_jsonl_from(path, last_lines)
                    state["live"][run_id] = current_lines
                    for sd in new_steps:
                        # Mirror the shape of the live on_step broadcast
                        await self._broadcast({
                            "type": "step",
                            "agent_name": name,
                            "run_id": run_id,
                            **sd,
                        })

            # 3) Detect completions: run_*.json that wasn't in completed set
            for fname in files:
                if not (fname.startswith("run_") and fname.endswith(".json")):
                    continue
                run_id = fname[len("run_"):-len(".json")]
                if run_id in state["completed"]:
                    continue
                state["completed"].add(run_id)
                # Drop from live tracking
                state["live"].pop(run_id, None)
                try:
                    with open(os.path.join(obs_dir, fname), "r", encoding="utf-8") as f:
                        run_dict = json.load(f)
                except (OSError, json.JSONDecodeError) as e:
                    logger.debug(f"[LiveObsHub] read {fname} failed: {e}")
                    continue
                _prid = run_dict.get("parent_run_id", "")
                await self._broadcast({
                    "type": "run_end",
                    "agent_name": name,
                    "success": run_dict.get("success", False),
                    "parent_run_id": _prid,
                    "is_sub_agent": bool(_prid),
                    "run": run_dict,
                })

            # 4) Live files that vanished without a run_*.json — agent crashed / cleared
            for run_id in list(state["live"].keys()):
                if run_id not in seen_live_ids and run_id not in state["completed"]:
                    # Live file gone but never completed — treat as interrupted
                    state["live"].pop(run_id, None)
                    # No special event; will surface on next snapshot via interrupted list


def register_agent_obs(agent_or_obs: Any, vfs: Any = None) -> None:
    """Public helper. Accepts FlowAgent (with .obs) or an ObservabilityLayer directly.

    If `vfs` is provided (or auto-discovered on the agent via .vfs / .vfs_v2),
    revert actions become available in the live UI for runs of this agent.
    """
    obs = getattr(agent_or_obs, "obs", agent_or_obs)

    if vfs is None:
        vfs = getattr(agent_or_obs, "vfs", None) or getattr(agent_or_obs, "vfs_v2", None)
    hub = LiveObsHub.get()
    hub.attach_app(app)
    hub.register(obs)
    if vfs is not None:
        hub._vfs_map[obs.agent_name] = vfs


def start_disk_scanner(*roots: str, interval: float = 2.0) -> None:
    """Configure disk-scan fallback. Each root contains <agent_name>/obs/ subdirs.

    Call before serving (or any time — scanner starts on next WS connect).
    Live-registered agents are skipped by the scanner to avoid duplicate events.
    """
    hub = LiveObsHub.get()
    hub.attach_app(app)
    hub.set_scan_interval(interval)
    for r in roots:
        hub.add_scan_root(r)
    hub._ensure_scanner()


# =============================================================================
# WS endpoint
# =============================================================================


@app.websocket("/ws/openLive")
class LiveHandler:
    async def on_connect(self, conn_id, session):
        hub = LiveObsHub.get()
        hub.attach_app(app)
        # Capture the WS-handler event loop so the scanner thread can push
        # broadcasts onto it via run_coroutine_threadsafe.
        try:
            hub._handler_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        hub._ensure_scanner()
        return {"type": "snapshot", **hub.snapshot()}

    async def on_message(self, payload, conn_id, session, request=None):
        t = payload.get("type")
        hub = LiveObsHub.get()
        # Capture loop opportunistically (covers servers where on_connect
        # return may not be auto-sent to the client).
        try:
            hub._handler_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        hub._ensure_scanner()
        if t == "get_snapshot":
            return {"type": "snapshot", **hub.snapshot()}
        if t == "get_run":
            agent = payload.get("agent_name", "")
            rid = payload.get("run_id", "")
            obs = hub._registered.get(agent)
            if obs is None:
                return {"type": "error", "message": f"agent not registered: {agent}"}
            run = obs.get_run(rid)
            if run is None:
                res = obs.get_resumable_run(rid)
                if res:
                    run, _ = res
            if run is not None:
                return {
                    "type": "run_detail",
                    "agent_name": agent,
                    "run": run.to_dict(),
                }
            return {"type": "error", "message": f"run not found: {rid}"}
        if t == "search":
            return self._do_search(hub, payload.get("q", ""))
        if t == "revert":
            return self._do_revert(hub, payload)
        return {"type": "ack"}

    def _do_search(self, hub, q: str) -> dict:
        q = (q or "").strip().lower()
        if not q or len(q) < 2:
            return {"type": "search_results", "q": q, "results": []}
        results = []
        MAX = 200

        def add(agent, rid, step_id, matched, snippet):
            if len(results) >= MAX:
                return False
            results.append({
                "agent": agent, "run_id": rid, "step_id": step_id,
                "matched": matched, "snippet": (snippet or "")[:160],
            })
            return True

        def scan_run(agent, run):
            rid = run.run_id
            # Run-level fields
            for field in ("run_id", "query", "persona"):
                v = str(getattr(run, field, ""))
                if q in v.lower():
                    if not add(agent, rid, None, field, v):
                        return False
                    break
            for step in run.steps:
                if step.llm:
                    inp = ""
                    if step.llm.input_messages:
                        try:
                            inp = json.dumps(step.llm.input_messages, ensure_ascii=False)
                        except (TypeError, ValueError):
                            inp = str(step.llm.input_messages)
                    if q in inp.lower():
                        if not add(agent, rid, step.step_id, "llm_input", inp):
                            return False
                    out = step.llm.output_text or ""
                    if q in out.lower():
                        if not add(agent, rid, step.step_id, "llm_output", out):
                            return False
                for tc in step.tool_calls:
                    for fname, fval in (("tool_name", tc.name),
                                         ("tool_args", tc.args_summary),
                                         ("tool_result", tc.result_summary),
                                         ("tool_error", tc.error)):
                        s = str(fval or "")
                        if q in s.lower():
                            if not add(agent, rid, step.step_id,
                                       f"{fname}:{tc.name}", s):
                                return False
                            break
                for vd in step.vfs_deltas:
                    p = str(vd.get("path", ""))
                    if q in p.lower():
                        if not add(agent, rid, step.step_id, "vfs_path", p):
                            return False
                    # Also search file contents
                    for cf in ("before_content", "after_content"):
                        c = vd.get(cf) or ""
                        if c and q in str(c).lower():
                            if not add(agent, rid, step.step_id,
                                       f"vfs_{cf}:{p}", str(c)):
                                return False
                            break
            return True

        for name, obs in hub._registered.items():
            if q in name.lower():
                add(name, None, None, "agent_name", name)
                if len(results) >= MAX:
                    break
            try:
                summaries = obs.list_runs()
            except Exception:
                summaries = []
            for s in summaries:
                rid = s.get("run_id", "")
                if not rid:
                    continue
                run = obs.get_run(rid)
                if run is None:
                    continue
                if not scan_run(name, run):
                    break
            if len(results) >= MAX:
                break
            # Live/interrupted
            try:
                interrupted = obs.get_interrupted_runs()
            except Exception:
                interrupted = []
            for ir in interrupted:
                rid = ir.get("run_id", "")
                res = obs.get_resumable_run(rid)
                if not res:
                    continue
                run, _ = res
                if not scan_run(name, run):
                    break
            if len(results) >= MAX:
                break
        return {"type": "search_results", "q": q, "results": results,
                "truncated": len(results) >= MAX}

    def _do_revert(self, hub, payload: dict) -> dict:
        agent = payload.get("agent_name", "")
        rid = payload.get("run_id", "")
        mode = payload.get("mode", "all")
        path = payload.get("path")
        step_id = payload.get("step_id")
        delta_index = payload.get("delta_index")
        obs = hub._registered.get(agent)
        vfs = hub._vfs_map.get(agent)
        if obs is None:
            return {"type": "revert_result", "success": False,
                    "message": f"agent not registered: {agent}"}
        if vfs is None:
            return {"type": "revert_result", "success": False,
                    "message": f"no VFS bound for agent '{agent}'. "
                               f"Pass vfs= to register_agent_obs() to enable revert."}
        try:
            result = obs.revert_from_run(
                rid, vfs, mode=mode, path=path,
                step_id=step_id, delta_index=delta_index,
            )
            return {
                "type": "revert_result",
                "success": result.get("success", False),
                "message": result.get("error") or "ok",
                "result": result,
                "agent_name": agent, "run_id": rid,
                "mode": mode, "step_id": step_id, "path": path,
                "delta_index": delta_index,
            }
        except Exception as e:
            return {"type": "revert_result", "success": False,
                    "message": f"{type(e).__name__}: {e}",
                    "agent_name": agent, "run_id": rid}

    async def on_disconnect(self, conn_id, session):
        pass


# =============================================================================
# HTML
# =============================================================================


@app.get("/")
async def index():
    return _HTML_TEMPLATE.replace("__VIEWER_CSS__", _VIEWER_CSS) \
                         .replace("__HOT_RELOAD__", app.hot_reload_script())


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ISAA LIVE OBS</title>
<style>
__VIEWER_CSS__

/* === LIVE-server additions === */
body{display:grid;grid-template-columns:42ch 1fr;height:100vh;padding-bottom:0;overflow:hidden}
.sidebar{border-right:1px solid var(--bd);overflow-y:auto;background:var(--bg)}
.main{overflow-y:auto;padding:16px}
.sidebar-h{padding:8px 12px;border-bottom:1px solid var(--bd);position:sticky;top:0;background:var(--bg);z-index:50;display:flex;align-items:center;justify-content:space-between}
.sidebar-h .t{color:var(--primary);font-size:12px;font-weight:600;letter-spacing:1px}
.conn-st{font-size:10px;color:var(--fg2);font-family:var(--mono)}
.conn-st.on{color:var(--success)}
.conn-st.off{color:var(--error)}
.agent-grp{border-bottom:1px solid var(--bd)}
.agent-h{padding:6px 12px;color:var(--fg);font-size:11px;background:var(--bg1);display:flex;align-items:center;gap:6px;cursor:pointer;user-select:none}
.agent-h .chev{color:var(--fg2);font-size:9px;transition:transform 80ms linear}
.agent-grp.col .agent-h .chev{transform:rotate(-90deg)}
.agent-grp.col .agent-runs{display:none}
.agent-h .nm{flex:1;color:var(--primary)}
.agent-h .cnt{color:var(--fg2);font-size:10px}
.agent-h .live-dot{color:var(--warning);animation:pulse 1.2s linear infinite}
.agent-tag{font-size:10px;letter-spacing:1px;text-transform:uppercase;padding:1px 6px;border:1px solid var(--bd);margin-left:4px}
.agent-tag.run{color:var(--warning);border-color:var(--warning);animation:pulse 1.2s linear infinite}
.agent-tag.idle{color:var(--fg2)}
.run-li{padding:5px 12px 5px 24px;cursor:pointer;border-left:2px solid transparent;font-size:11px;display:flex;align-items:center;gap:6px}
.run-li:hover{background:var(--sel)}
.run-li.sel{background:var(--sel);border-left-color:var(--primary)}
.run-li.is-sub{padding-left:36px;border-left-color:var(--primary);opacity:.92}
.run-li .sub-badge{font-size:9px;color:var(--primary);border:1px solid var(--primary);border-radius:3px;padding:0 4px;margin-left:5px;letter-spacing:.5px}
.run-li .gl{width:10px;flex-shrink:0;text-align:center;font-size:11px}
.run-li .gl.run{color:var(--warning);animation:pulse 1.2s linear infinite}
.run-li .gl.ok{color:var(--success)}
.run-li .gl.fail{color:var(--error)}
.run-li .gl.int{color:var(--warning)}
.run-li .id{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--fg1)}
.run-li.sel .id{color:var(--fg)}
.run-li .dur{color:var(--fg2);font-size:10px;flex-shrink:0}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

.empty{padding:32px 16px;color:var(--fg2);font-size:11px;text-align:center}
.main-h{padding:8px 16px;border-bottom:1px solid var(--bd);margin:-16px -16px 16px;background:var(--bg);position:sticky;top:-16px;z-index:40;display:flex;align-items:center;gap:12px}
.main-h .crumb{font-size:11px;color:var(--fg1)}
.main-h .crumb .sep{color:var(--fg2);margin:0 6px}
.live-tag{color:var(--warning);font-size:10px;letter-spacing:1px;text-transform:uppercase;animation:pulse 1.2s linear infinite}

@media(max-width:960px){
  body{grid-template-columns:1fr;grid-template-rows:auto 1fr}
  .sidebar{max-height:38vh;border-right:none;border-bottom:1px solid var(--bd)}
}

/* === SEARCH === */
.search-wrap{padding:6px 12px;border-bottom:1px solid var(--bd);background:var(--bg)}
.search-in{width:100%;background:var(--bg2);border:1px solid var(--bd);color:var(--fg);font-family:var(--mono);font-size:11px;padding:5px 8px;outline:none}
.search-in:focus{border-color:var(--primary)}
.search-in::placeholder{color:var(--fg2)}
.search-results{padding:6px 12px;border-bottom:1px solid var(--bd);background:var(--bg1);max-height:40vh;overflow-y:auto}
.sr-hdr{font-size:10px;color:var(--fg2);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
.sr-item{padding:4px 6px;cursor:pointer;font-size:11px;border-left:2px solid transparent;display:grid;grid-template-columns:auto 1fr;gap:6px}
.sr-item:hover{background:var(--sel);border-left-color:var(--primary)}
.sr-key{color:var(--fg2);font-size:10px;text-transform:uppercase;letter-spacing:1px;align-self:center}
.sr-snip{color:var(--fg1);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sr-ag{color:var(--primary);font-size:10px}

/* === RUN-LI TIMESTAMP === */
.run-li{flex-wrap:wrap}
.run-li .ts{width:100%;color:var(--fg2);font-size:9px;padding-left:16px;margin-top:-2px}

/* === ACTION BUTTONS === */
.act-btn{font-family:var(--mono);font-size:10px;background:transparent;color:var(--fg1);border:1px solid var(--bd);padding:2px 8px;cursor:pointer;letter-spacing:1px;text-transform:uppercase;margin-left:6px;transition:all 100ms linear}
.act-btn:hover{border-color:var(--warning);color:var(--warning)}
.act-btn:disabled{opacity:.4;cursor:not-allowed}
.act-btn::before{content:"[ ";color:var(--fg2)}
.act-btn::after{content:" ]";color:var(--fg2)}
.act-btn:hover::before,.act-btn:hover::after{color:var(--warning)}
.act-btn.danger:hover{border-color:var(--error);color:var(--error)}
.act-btn.danger:hover::before,.act-btn.danger:hover::after{color:var(--error)}

/* === VFS DIFF === */
.vfs-row{cursor:pointer}
.vfs-row:hover{background:rgba(255,255,255,0.03)}
.vfs-detail{display:none;padding:6px 8px;background:var(--bg2);border:1px solid var(--bd);margin:2px 0 6px}
.vfs-detail.open{display:block}
.vfs-acts{display:flex;gap:4px;margin-bottom:6px;padding-bottom:6px;border-bottom:1px solid var(--bd)}
.diff-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:10px}
.diff-col{background:var(--bg);border:1px solid var(--bd);padding:6px 8px;max-height:280px;overflow:auto}
.diff-col-h{font-size:9px;color:var(--fg2);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;padding-bottom:3px;border-bottom:1px solid var(--bd)}
.diff-col pre{white-space:pre-wrap;word-break:break-word;color:var(--fg1);line-height:1.4;margin:0}
.diff-col.before pre{color:rgba(255,180,180,.85)}
.diff-col.after pre{color:rgba(180,255,180,.85)}
.toast{position:fixed;top:16px;right:16px;padding:8px 14px;background:var(--bg1);border:1px solid var(--bd);font-size:11px;color:var(--fg);z-index:1100;max-width:400px;animation:fadein 100ms linear}
.toast.ok{border-color:var(--success);color:var(--success)}
.toast.err{border-color:var(--error);color:var(--error)}
@keyframes fadein{from{opacity:0}to{opacity:1}}
@media(max-width:640px){.diff-grid{grid-template-columns:1fr}}
</style>
</head>
<body>

<div class="sidebar">
  <div class="sidebar-h">
    <span class="t">ISAA LIVE OBS</span>
    <span class="conn-st off" id="conn-st">○ connecting</span>
  </div>
  <div class="search-wrap">
    <input id="search-in" class="search-in" type="text" placeholder="> search runs / steps / content"
           autocomplete="off" spellcheck="false">
  </div>
  <div id="search-results"></div>
  <div id="agents-list">
    <div class="empty">waiting for snapshot…</div>
  </div>
</div>

<div class="main">
  <div class="main-h">
    <span class="crumb" id="crumb">no run selected</span>
    <span id="live-ind"></span>
  </div>
  <div id="content">
    <div class="empty">select a run from the sidebar.</div>
  </div>
</div>

__HOT_RELOAD__
<script>
// ════════════════════════════════════════════════════════════════════
// STATE
// ════════════════════════════════════════════════════════════════════
const STATE = {
  agents: {},        // agent_name -> {runs, interrupted, active_run}
  liveRuns: {},      // "{agent}::{run_id}" -> run obj (incl. steps[])
  liveActivity: {},  // "{agent}::{run_id}" -> last step receive ts (client ms)
  selected: null,    // {agent, run_id, kind}
  collapsed: new Set(),
  seenAgents: new Set(),  // tracks first-render so we collapse by default
  searchQ: '',
  searchTimer: null,
  vfsCapable: new Set(),  // agents that support revert (server tells us)
};
const FRESH_MS = 30000;  // a run is "running now" if activity within this window

// ── Utils (ported from obs_viewer) ──
const F = {
  dur(s){if(s==null||s===0) return '—'; return s<1?`${(s*1000).toFixed(0)}ms`:`${s.toFixed(2)}s`},
  durX(s){if(s==null) return '—'; if(s===0) return '0'; return s<1?`${(s*1000).toFixed(0)}ms`:`${s.toFixed(2)}s`},
  tok(n){return n==null?'—':n>=1000?`${(n/1000).toFixed(1)}k`:String(n)},
  pct(v,t){return t>0?`${((v/t)*100).toFixed(1)}%`:'0%'},
  ts(t){if(!t) return '—'; const d=new Date(t*1000);
    const pad=n=>n<10?'0'+n:''+n;
    return `${pad(d.getDate())}.${pad(d.getMonth()+1)} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  },
  tsFull(t){if(!t) return '—'; const d=new Date(t*1000);
    const pad=n=>n<10?'0'+n:''+n;
    return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  },
  tsRel(t){if(!t) return '—'; const s=(Date.now()/1000)-t;
    if(s<60) return Math.round(s)+'s ago';
    if(s<3600) return Math.round(s/60)+'min ago';
    if(s<86400) return Math.round(s/3600)+'h ago';
    return Math.round(s/86400)+'d ago';
  },
};
function esc(s){if(s==null)return'';return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function fmtJson(v){if(v==null)return'';if(typeof v==='string'){try{return JSON.stringify(JSON.parse(v),null,2)}catch(e){return v}}try{return JSON.stringify(v,null,2)}catch(e){return String(v)}}

function analyzeStep(step){
  const d=step.duration_s||0, llm=step.llm;
  let llmDur=0,preLlm=0,postLlm=0,toolDur=0;
  if(llm&&llm.t_start&&step.t_start){
    preLlm=Math.max(0,llm.t_start-step.t_start);
    llmDur=llm.duration_s||0;
    (step.tool_calls||[]).forEach(tc=>{toolDur+=tc.duration_s||0});
    postLlm=Math.max(0,d-preLlm-llmDur-toolDur);
  }else{
    (step.tool_calls||[]).forEach(tc=>{toolDur+=tc.duration_s||0});
    postLlm=Math.max(0,d-toolDur);
  }
  return{total:d,llm:llmDur,tool:toolDur,pre:preLlm,post:postLlm,
    ttft:llm?.ttft_s||0,
    ttftMissing:llm&&(!llm.t_first_token||llm.t_first_token===0),
    tokIn:llm?.tokens_in||0,tokOut:llm?.tokens_out||0,
    tokSec:llm?.tokens_per_sec||0,model:llm?.model||'',
    hasError:(step.tool_calls||[]).some(tc=>tc.status==='error')};
}

function analyzeRun(r){
  const steps=r.steps||[];
  let tLlm=0,tTool=0,tPre=0,tPost=0;
  let ttfts=[], models=new Set();
  const analyzed=steps.map(s=>{
    const a=analyzeStep(s);
    tLlm+=a.llm;tTool+=a.tool;tPre+=a.pre;tPost+=a.post;
    if(a.ttft>0) ttfts.push(a.ttft);
    if(a.model) models.add(a.model);
    return{step:s,a};
  });
  const totalDur = r.duration_s || (
    steps.length && steps[steps.length-1].t_end && steps[0].t_start
      ? steps[steps.length-1].t_end - steps[0].t_start : 0);
  const avgTtft=ttfts.length?ttfts.reduce((a,b)=>a+b,0)/ttfts.length:0;
  return{analyzed,totalDur,totalLlm:tLlm,totalTool:tTool,totalPre:tPre,totalPost:tPost,avgTtft,models:[...models]};
}

// ── LLM message renderers (from obs_viewer) ──
function decodeMaybe(s){
  if(typeof s!=='string') return s;
  if(s.indexOf('\n')!==-1) return s;
  if(!/\\[ntr"\\]/.test(s)) return s;
  return s.replace(/\\n/g,'\n').replace(/\\t/g,'\t').replace(/\\r/g,'\r')
          .replace(/\\"/g,'"').replace(/\\\\/g,'\\');
}
function renderLlmMessages(msgs){
  if(!msgs) return '<span style="color:var(--fg2)">no input data captured</span>';
  if(typeof msgs==='string') return '<pre>'+esc(decodeMaybe(msgs))+'</pre>';
  if(!Array.isArray(msgs)) return '<pre>'+esc(fmtJson(msgs))+'</pre>';
  return msgs.map(m=>{
    const role=m.role||'unknown';
    let content='';
    if(typeof m.content==='string') content=m.content;
    else if(Array.isArray(m.content)) content=m.content.map(c=>typeof c==='string'?c:(c.text||JSON.stringify(c))).join('\n');
    else content=fmtJson(m.content);
    return `<div class="msg role-${role}"><div class="msg-role ${role}">${esc(role)}</div><div class="msg-content">${esc(decodeMaybe(content))}</div></div>`;
  }).join('');
}
function renderLlmOutput(out){
  if(!out) return '<span style="color:var(--fg2)">no output data captured</span>';
  if(typeof out==='string') return '<pre>'+esc(decodeMaybe(out))+'</pre>';
  return '<pre>'+esc(fmtJson(out))+'</pre>';
}

// ── Single step renderer (1:1 port of obs_viewer step rendering) ──
function renderStep(step, idPrefix){
  const a=analyzeStep(step);
  const cls=a.hasError?' err':(a.pre>2||a.post>2?' warn':'');
  const stepEid=idPrefix+'-step-'+step.step_id;
  const maxT=a.total||1;
  const prePct=(a.pre/maxT*100).toFixed(1);
  const llmPct=(a.llm/maxT*100).toFixed(1);
  const toolPct=(a.tool/maxT*100).toFixed(1);
  const postPct=(a.post/maxT*100).toFixed(1);
  const llmOff=parseFloat(prePct);
  const toolOff=llmOff+parseFloat(llmPct);
  const postOff=toolOff+parseFloat(toolPct);

  const agent=STATE.selected?STATE.selected.agent:'';
  const runId=STATE.selected?STATE.selected.run_id:'';
  const canRevert=STATE.vfsCapable && STATE.vfsCapable.has(agent);
  const revertStepBtn=(canRevert && (step.vfs_deltas||[]).length)
    ? `<button class="act-btn danger" onclick="event.stopPropagation();doRevert('${esc(agent)}','${esc(runId)}','step',{step_id:${step.step_id}})">undo step</button>`
    : '';

  let h=`<div class="step${cls}" id="${stepEid}">
    <div class="step-hdr" onclick="toggleStep('${stepEid}')"><div class="step-h"><span class="step-id"><span class="chevron">▼</span>step ${step.step_id}</span><span class="step-dur">${F.dur(a.total)}${revertStepBtn}</span></div></div>
    <div class="step-body">`;

  h+=`<div class="t-bar-row">
    <span class="t-bar-label">timing</span>
    <div class="t-bar-track">
      ${a.pre>0?`<div class="t-bar-seg pre" style="left:0%;width:${prePct}%" title="pre-LLM ${F.dur(a.pre)}"></div>`:''}
      <div class="t-bar-seg llm" style="left:${llmOff}%;width:${Math.max(parseFloat(llmPct),0.5)}%" title="LLM ${F.dur(a.llm)}"></div>
      ${a.tool>0?`<div class="t-bar-seg tool" style="left:${toolOff}%;width:${toolPct}%" title="tools ${F.dur(a.tool)}"></div>`:''}
      ${a.post>0.01?`<div class="t-bar-seg post" style="left:${postOff}%;width:${Math.max(parseFloat(postPct),0.5)}%" title="post-LLM ${F.dur(a.post)}"></div>`:''}
    </div><span class="t-bar-val"></span>
  </div>`;

  h+=`<div style="display:flex;flex-wrap:wrap;gap:12px;font-size:10px;color:var(--fg2);margin:4px 0">`;
  if(a.pre>0.01) h+=`<span style="color:var(--warning)">pre: ${F.dur(a.pre)}</span>`;
  h+=`<span style="color:var(--primary)">llm: ${F.dur(a.llm)}</span>`;
  if(a.tool>0) h+=`<span style="color:var(--success)">tool: ${F.dur(a.tool)}</span>`;
  if(a.post>0.01) h+=`<span style="color:var(--error);opacity:.7">post: ${F.dur(a.post)}</span>`;
  h+=`</div>`;

  if(step.llm){
    const l=step.llm;
    const llmId=stepEid+'-llm';
    const ttftNote=a.ttftMissing?' <span style="color:var(--warning)">▲ not captured</span>':'';
    h+=`<div class="step-sec"><div class="step-sec-l">LLM</div>
      <div class="llm-toggle" onclick="event.stopPropagation();toggleDetail('${llmId}')" title="click to show input/output">
      <div style="display:flex;flex-wrap:wrap;gap:12px;font-size:11px;color:var(--fg1)">
        <span class="kv"><span class="k">model </span><span class="v">${esc(l.model||'?')}</span></span>
        <span class="kv"><span class="k">ttft </span><span class="v" style="color:var(--primary)">${a.ttft>0?F.dur(a.ttft):'—'}</span>${ttftNote}</span>
        <span class="kv"><span class="k">duration </span><span class="v">${F.dur(l.duration_s)}</span></span>
        <span class="kv"><span class="k">in </span><span class="v">${F.tok(l.tokens_in)}</span></span>
        <span class="kv"><span class="k">out </span><span class="v">${F.tok(l.tokens_out)}</span></span>
        <span class="kv"><span class="k">tok/s </span><span class="v">${a.tokSec?a.tokSec.toFixed(1):'—'}</span></span>
      </div></div>
      <div class="llm-detail" id="${llmId}">
        <div class="llm-detail-tab-bar">
          <button class="llm-detail-tab active" onclick="event.stopPropagation();switchLlmTab('${llmId}','input')">Input</button>
          <button class="llm-detail-tab" onclick="event.stopPropagation();switchLlmTab('${llmId}','output')">Output</button>
        </div>
        <div class="llm-detail-content" id="${llmId}-input">${renderLlmMessages(l.input_messages||l.messages||l.input)}</div>
        <div class="llm-detail-content" id="${llmId}-output" style="display:none">${renderLlmOutput(l.output||l.response||l.output_text)}</div>
      </div></div>`;
  }

  if(step.tool_calls?.length){
    h+=`<div class="step-sec"><div class="step-sec-l">Tools</div>`;
    step.tool_calls.forEach((tc,ti)=>{
      const isErr=tc.status==='error';
      const tcId=stepEid+'-tc-'+ti;
      h+=`<div class="tool-row-wrap">`;
      h+=`<div class="tool-row${isErr?' terr':''}" onclick="toggleDetail('${tcId}')" title="click to expand"><span>${isErr?'✕ ':'▸ '}${esc(tc.name)}</span><span>${F.dur(tc.duration_s)}</span><span>${esc(tc.status)}</span><span style="color:var(--fg2)">${esc((tc.result_summary||tc.error||tc.args_summary||'').substring(0,80))}</span></div>`;
      h+=`<div class="tool-detail" id="${tcId}">`;
      if(tc.args||tc.args_summary||tc.input) h+=`<div class="tool-detail-l">Input / Args</div><pre>${esc(fmtJson(tc.args||tc.input||tc.args_summary))}</pre>`;
      if(tc.result||tc.result_summary||tc.output||tc.error) h+=`<div class="tool-detail-l">${isErr?'Error':'Result'}</div><pre>${esc(fmtJson(tc.result||tc.output||tc.result_summary||tc.error))}</pre>`;
      h+=`</div></div>`;
    });
    h+='</div>';
  }

  if(step.vfs_deltas?.length){
    h+=`<div class="step-sec"><div class="step-sec-l">VFS</div>`;
    step.vfs_deltas.forEach((vd,vi)=>{
      const la=vd.lines_added!=null?vd.lines_added:(vd.after_content!=null?vd.after_content.split('\n').length:0);
      const lr=vd.lines_removed!=null?vd.lines_removed:(vd.before_content!=null?vd.before_content.split('\n').length:0);
      const vfsId=stepEid+'-vfs-'+vi;
      const before=vd.before_content||'';
      const after=vd.after_content||'';
      const hasDiff=(before||after);
      const deltaIdx=vd.index;
      h+=`<div class="vfs-row" onclick="toggleDetail('${vfsId}')"><span class="file-path">${esc(vd.path)}</span><span>${esc(vd.action)}</span><span class="file-d">+${la}</span><span class="file-d neg">-${lr}</span></div>`;
      h+=`<div class="vfs-detail" id="${vfsId}">`;
      if(canRevert){
        h+=`<div class="vfs-acts">`;
        if(deltaIdx!=null) h+=`<button class="act-btn danger" onclick="event.stopPropagation();doRevert('${esc(agent)}','${esc(runId)}','delta',{delta_index:${deltaIdx}})">undo this change</button>`;
        h+=`<button class="act-btn danger" onclick="event.stopPropagation();doRevert('${esc(agent)}','${esc(runId)}','file',{path:'${esc(vd.path)}'})">undo file</button>`;
        h+=`</div>`;
      }
      if(hasDiff){
        h+=`<div class="diff-grid">
          <div class="diff-col before"><div class="diff-col-h">before</div><pre>${esc(before)||'<span style="color:var(--fg2)">(empty)</span>'}</pre></div>
          <div class="diff-col after"><div class="diff-col-h">after</div><pre>${esc(after)||'<span style="color:var(--fg2)">(empty)</span>'}</pre></div>
        </div>`;
      } else {
        h+=`<div style="color:var(--fg2);font-size:10px">no content snapshot (metadata-only delta)</div>`;
      }
      h+=`</div>`;
    });
    h+='</div>';
  }

  if(step.compression){
    h+=`<div class="step-sec"><div class="step-sec-l">Compression</div>
      <div style="font-size:11px;color:var(--fg1)">kept: ${step.compression.kept||0} │ summarized: ${step.compression.summarized||0} │ dropped: ${step.compression.dropped||0}</div></div>`;
  }

  h+='</div></div>';
  return h;
}

function renderRunHeader(run, isLive){
  const ra=analyzeRun(run);
  const liveTag=isLive?' <span class="live-tag">◐ live</span>':'';
  const agent=STATE.selected?STATE.selected.agent:'';
  const canRevert=STATE.vfsCapable && STATE.vfsCapable.has(agent);
  const revertAllBtn=canRevert
    ? `<button class="act-btn danger" onclick="doRevert('${esc(agent)}','${esc(run.run_id)}','all',{})">undo all</button>`
    : '';
  return `<div style="padding:12px 0;border-bottom:1px solid var(--bd);margin-bottom:16px">
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:6px;align-items:center">
      <span class="kv"><span class="k">run </span><span class="v" style="color:var(--primary)">${esc(run.run_id||'')}</span>${liveTag}</span>
      <span class="kv"><span class="k">query </span><span class="v">${esc((run.query||'—').substring(0,160))}</span></span>
      <span class="kv"><span class="k">duration </span><span class="v">${F.dur(run.duration_s||ra.totalDur)}</span></span>
      <span class="kv"><span class="k">model </span><span class="v">${esc(ra.models.join(', ')||'?')}</span></span>
      <span class="kv"><span class="k">persona </span><span class="v">${esc(run.persona||'default')}</span></span>
      ${run.skills_matched?.length?`<span class="kv"><span class="k">skills </span><span class="v">${esc(run.skills_matched.join(', '))}</span></span>`:''}
      ${revertAllBtn}
    </div>
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:6px;font-size:10px;color:var(--fg2)">
      ${run.t_start?`<span>started ${F.tsFull(run.t_start)} (${F.tsRel(run.t_start)})</span>`:''}
      ${run.t_end?`<span>ended ${F.tsFull(run.t_end)}</span>`:''}
    </div>
  </div>
  <div class="stats-row">
    <div class="stat"><div class="stat-l">STEPS</div><div class="stat-v">${(run.steps||[]).length}</div></div>
    <div class="stat"><div class="stat-l">LLM TIME</div><div class="stat-v">${F.dur(ra.totalLlm)}</div><div class="stat-s">${F.pct(ra.totalLlm,ra.totalDur)} of total</div></div>
    <div class="stat"><div class="stat-l">TOOL TIME</div><div class="stat-v">${F.durX(ra.totalTool)}</div><div class="stat-s">${F.pct(ra.totalTool,ra.totalDur)} of total</div></div>
    <div class="stat"><div class="stat-l">TOK IN</div><div class="stat-v">${F.tok(run.total_tokens_in||0)}</div></div>
    <div class="stat"><div class="stat-l">TOK OUT</div><div class="stat-v">${F.tok(run.total_tokens_out||0)}</div></div>
    <div class="stat"><div class="stat-l">AVG TTFT</div><div class="stat-v">${F.dur(ra.avgTtft)}</div></div>
  </div>`;
}

function renderRun(run, isLive){
  const idPrefix=(STATE.selected?STATE.selected.agent:'x')+'-'+(run.run_id||'unk');
  let h=renderRunHeader(run, isLive);
  h+=`<div class="sec-label">Step Timeline</div>`;
  h+=`<div class="legend">
    <span class="legend-i"><span class="legend-s" style="background:var(--warning);opacity:.6"></span> pre-LLM</span>
    <span class="legend-i"><span class="legend-s" style="background:var(--primary)"></span> LLM</span>
    <span class="legend-i"><span class="legend-s" style="background:var(--success)"></span> tool</span>
    <span class="legend-i"><span class="legend-s" style="background:var(--error);opacity:.4"></span> post-LLM</span>
  </div>`;
  h+=`<div class="collapse-bar">
    <button class="collapse-btn" onclick="toggleAllSteps(true)">[ Min All ]</button>
    <button class="collapse-btn" onclick="toggleAllSteps(false)">[ Open All ]</button>
  </div>`;
  h+=`<div class="step-tl" id="step-tl">`;
  (run.steps||[]).forEach(s=>{h+=renderStep(s, idPrefix)});
  h+=`</div>`;
  return h;
}

// ════════════════════════════════════════════════════════════════════
// SIDEBAR
// ════════════════════════════════════════════════════════════════════
function isSelected(agent, runId){
  return STATE.selected && STATE.selected.agent===agent && STATE.selected.run_id===runId;
}

function isFreshLive(r){
  const key=r._agent+'::'+r.run_id;
  const ts=STATE.liveActivity[key];
  if(ts && Date.now()-ts < FRESH_MS) return true;
  // Fallback: mtime from disk (seconds since epoch)
  if(r.mtime && (Date.now()/1000 - r.mtime) < FRESH_MS/1000) return true;
  return false;
}

function renderSidebar(){
  const list=document.getElementById('agents-list');
  const names=Object.keys(STATE.agents).sort();
  if(names.length===0){
    list.innerHTML='<div class="empty">no agents registered yet…</div>';
    return;
  }
  let h='';
  for(const name of names){
    // First sighting → collapse by default
    if(!STATE.seenAgents.has(name)){
      STATE.seenAgents.add(name);
      STATE.collapsed.add(name);
    }
    const ag=STATE.agents[name];
    // Merge live: event-tracked runs + disk-discovered interrupted live_*.jsonl files
    const eventLive=Object.values(STATE.liveRuns).filter(r=>r._agent===name).map(r=>({...r,_agent:name}));
    const eventIds=new Set(eventLive.map(r=>r.run_id));
    const diskLive=(ag.interrupted||[])
      .filter(ir=>!eventIds.has(ir.run_id))
      .map(ir=>({...ir,_agent:name,_fromDisk:true}));
    const allLive=[...eventLive, ...diskLive];
    // Sort: fresh first, then by mtime/t_start desc
    allLive.sort((a,b)=>{
      const fa=isFreshLive(a)?1:0, fb=isFreshLive(b)?1:0;
      if(fa!==fb) return fb-fa;
      return (b.mtime||b.t_start||0)-(a.mtime||a.t_start||0);
    });
    const liveIds=new Set(allLive.map(r=>r.run_id));
    const completed=(ag.runs||[]).filter(r=>!liveIds.has(r.run_id));
    const total=allLive.length+completed.length;
    // Agent-level tag right next to the name
    const runningCount=allLive.filter(isFreshLive).length;
    const idleLiveCount=allLive.length-runningCount;
    let agentTag='';
    if(runningCount>0) agentTag=`<span class="agent-tag run">● running${runningCount>1?' ×'+runningCount:''}</span>`;
    else if(idleLiveCount>0) agentTag=`<span class="agent-tag idle">◐ ${idleLiveCount} live</span>`;

    const isCol=STATE.collapsed.has(name);
    h+=`<div class="agent-grp${isCol?' col':''}">
      <div class="agent-h" onclick="toggleAgent('${esc(name)}')">
        <span class="chev">▼</span>
        <span class="nm">${esc(name)}</span>
        ${agentTag}
        <span class="cnt">${total}</span>
      </div>
      <div class="agent-runs">`;
    allLive.forEach(r=>{
      const sel=isSelected(name,r.run_id);
      const fresh=isFreshLive(r);
      const stepCount=(r.steps||[]).length || r.step_count || 0;
      const tref=r.mtime||r.t_start||0;
      const subBadge=r.is_sub_agent?`<span class="sub-badge" title="sub-agent of ${esc(r.parent_run_id||'?')}">⊕ sub</span>`:'';
      h+=`<div class="run-li${sel?' sel':''}${r.is_sub_agent?' is-sub':''}" onclick="selectRun('${esc(name)}','${esc(r.run_id)}','live')">
        <span class="gl ${fresh?'run':'int'}">◐</span>
        <span class="id">${esc(r.run_id)}${subBadge}${r._fromDisk&&!fresh?' <span style="color:var(--fg2)">(stale)</span>':''}</span>
        <span class="dur">${stepCount} st</span>
        ${tref?`<span class="ts">${F.ts(tref)} · ${F.tsRel(tref)}</span>`:''}
      </div>`;
    });
    completed.forEach(r=>{
      const sel=isSelected(name,r.run_id);
      const gl=r.success?'●':'✕';
      const cls=r.success?'ok':'fail';
      h+=`<div class="run-li${sel?' sel':''}" onclick="selectRun('${esc(name)}','${esc(r.run_id)}','complete')">
        <span class="gl ${cls}">${gl}</span>
        <span class="id">${esc(r.run_id)}</span>
        <span class="dur">${F.dur(r.duration_s)}</span>
        ${r.t_start?`<span class="ts">${F.ts(r.t_start)} · ${F.tsRel(r.t_start)}</span>`:''}
      </div>`;
    });
    h+='</div></div>';
  }
  list.innerHTML=h;
}

function toggleAgent(name){
  if(STATE.collapsed.has(name)) STATE.collapsed.delete(name);
  else STATE.collapsed.add(name);
  renderSidebar();
}

// ════════════════════════════════════════════════════════════════════
// SELECTION + MAIN RENDER
// ════════════════════════════════════════════════════════════════════
function selectRun(agent, runId, kind){
  STATE.selected={agent,run_id:runId,kind};
  document.getElementById('crumb').innerHTML=
    `<span style="color:var(--primary)">${esc(agent)}</span><span class="sep">/</span><span>${esc(runId)}</span>`;
  document.getElementById('live-ind').innerHTML=kind==='live'?'<span class="live-tag">◐ live</span>':'';
  renderSidebar();

  if(kind==='live'){
    const run=STATE.liveRuns[agent+'::'+runId];
    if(run){ renderMain(run, true); return; }
  }
  document.getElementById('content').innerHTML='<div class="empty">loading…</div>';
  wsSend({type:'get_run',agent_name:agent,run_id:runId});
}

function renderMain(run, isLive){
  document.getElementById('content').innerHTML=renderRun(run, isLive);
}

// ════════════════════════════════════════════════════════════════════
// LIVE EVENT HANDLERS
// ════════════════════════════════════════════════════════════════════
function onRunStart(msg){
  const agent=msg.agent_name;
  const key=agent+'::'+msg.run_id;
  STATE.liveActivity[key]=Date.now();

  // Beim Resume die bisherigen Schritte im UI behalten!
  const existing = STATE.liveRuns[key];
  const keepSteps = (msg.is_resume && existing) ? existing.steps : [];

  STATE.liveRuns[key]={
    run_id:msg.run_id, agent_name:agent,
    query:msg.query||'', persona:msg.persona||'',
    skills_matched:msg.skills_matched||[], session_id:msg.session_id||'',
    t_start:msg.t_start||Date.now()/1000,
    parent_run_id:msg.parent_run_id||'', is_sub_agent:!!msg.parent_run_id,
    steps:keepSteps, _agent:agent, _live:true,
  };
  if(!STATE.agents[agent]) STATE.agents[agent]={agent_name:agent,runs:[],interrupted:[],active_run:null};
  renderSidebar();
}

function appendStepLive(agent, runId, stepDict){
  const key=agent+'::'+runId;
  STATE.liveActivity[key]=Date.now();  // mark fresh
  let run=STATE.liveRuns[key];
  if(!run){
    // step arrived before run_start (race) — synthesize stub
    run={run_id:runId,agent_name:agent,steps:[],_agent:agent,_live:true,
         query:'(starting…)',t_start:stepDict.t_start||Date.now()/1000};
    STATE.liveRuns[key]=run;
    if(!STATE.agents[agent]) STATE.agents[agent]={agent_name:agent,runs:[],interrupted:[],active_run:null};
  }
  // Strip transport-layer fields
  const step={...stepDict};
  delete step.type; delete step.agent_name; delete step.run_id;
  run.steps.push(step);

  // If currently viewed → append to DOM (no full re-render)
  if(STATE.selected && STATE.selected.agent===agent && STATE.selected.run_id===runId){
    const tl=document.getElementById('step-tl');
    if(tl){
      const idPrefix=agent+'-'+runId;
      tl.insertAdjacentHTML('beforeend', renderStep(step, idPrefix));
      tl.lastElementChild?.scrollIntoView({block:'nearest',behavior:'smooth'});
      // Also refresh header stats
      const headerHost=document.getElementById('content');
      const oldHdr=headerHost.querySelector('.stats-row');
      if(oldHdr){
        const tmp=document.createElement('div');
        tmp.innerHTML=renderRunHeader(run, true);
        const newStats=tmp.querySelector('.stats-row');
        if(newStats) oldHdr.replaceWith(newStats);
      }
    }else{
      renderMain(run, true);
    }
  }
  renderSidebar();
}

function onRunEnd(msg){
  const agent=msg.agent_name;
  const rid=msg.run.run_id;
  const key=agent+'::'+rid;
  delete STATE.liveRuns[key];
  delete STATE.liveActivity[key];
  const ag=STATE.agents[agent]||(STATE.agents[agent]={agent_name:agent,runs:[],interrupted:[],active_run:null});
  ag.interrupted=(ag.interrupted||[]).filter(r=>r.run_id!==rid);
  const summary={...msg.run}; delete summary.steps;
  ag.runs=[summary, ...(ag.runs||[]).filter(r=>r.run_id!==rid)];
  renderSidebar();
  if(STATE.selected && STATE.selected.agent===agent && STATE.selected.run_id===rid){
    STATE.selected.kind='complete';
    document.getElementById('live-ind').innerHTML='';
    renderMain(msg.run, false);
  }
}

// ════════════════════════════════════════════════════════════════════
// WS
// ════════════════════════════════════════════════════════════════════
let ws=null;
let reconnectDelay=500;

function wsConnect(){
  const proto=location.protocol==='https:'?'wss:':'ws:';
  const wsPort=window.__TB_WS_PORT__ || '8100';
  const url=proto+'//'+location.hostname+(wsPort?':'+wsPort:'')+'/ws/openLive';
  ws=new WebSocket(url);
  ws.onopen=()=>{
    const el=document.getElementById('conn-st');
    el.className='conn-st on'; el.textContent='● connected';
    reconnectDelay=500;
    // Explicit snapshot request — some servers don't auto-send on_connect's
    // return value as a frame, so we trigger the request/response path.
    wsSend({type:'get_snapshot'});
  };
  ws.onclose=()=>{
    const el=document.getElementById('conn-st');
    el.className='conn-st off'; el.textContent='✕ disconnected';
    setTimeout(wsConnect, reconnectDelay);
    reconnectDelay=Math.min(reconnectDelay*2, 5000);
  };
  ws.onerror=()=>{};
  ws.onmessage=(e)=>{
    let msg; try{msg=JSON.parse(e.data)}catch(err){return}
    handleMsg(msg);
  };
}
function wsSend(o){ if(ws && ws.readyState===1) ws.send(JSON.stringify(o)); }

function handleMsg(msg){
  switch(msg.type){
    case 'snapshot':
      STATE.agents=msg.agents||{};
      STATE.vfsCapable = new Set(msg.vfs_capable||[]);
      // Re-hydrate all active runs (parent + parallel sub-agents) into liveRuns
      STATE.liveRuns={};
      Object.entries(STATE.agents).forEach(([name,ag])=>{
        const actives = (ag.active_runs && ag.active_runs.length)
          ? ag.active_runs
          : (ag.active_run ? [ag.active_run] : []);
        actives.forEach(r=>{
          STATE.liveRuns[name+'::'+r.run_id]={
            ...r,_agent:name,_live:true,
            is_sub_agent:!!r.parent_run_id,
          };
        });
      });
      renderSidebar();
      break;
    case 'run_start': onRunStart(msg); break;
    case 'step': appendStepLive(msg.agent_name, msg.run_id, msg); break;
    case 'run_end': onRunEnd(msg); break;
    case 'agent_added':
      STATE.agents[msg.agent_name]={
        agent_name:msg.agent_name,
        obs_dir:msg.obs_dir||'',
        runs:msg.runs||[],
        interrupted:msg.interrupted||[],
        active_run:msg.active_run||null,
      };
      renderSidebar();
      break;
    case 'run_detail':
      if(STATE.selected && STATE.selected.agent===msg.agent_name && STATE.selected.run_id===msg.run.run_id){
        renderMain(msg.run, false);
      }
      break;
    case 'error':
      document.getElementById('content').innerHTML=
        `<div class="warn-box"><div class="wt">▲ error</div>${esc(msg.message||'')}</div>`;
      break;
    case 'search_results':
      renderSearchResults(msg.q, msg.results, msg.truncated);
      break;
    case 'revert_result':
      if(msg.success){
        toast(`✓ undo ${msg.mode}: ${(msg.result&&msg.result.reverted_count)||'ok'}`, 'ok');
      } else {
        toast(`✕ undo failed: ${msg.message||'unknown'}`, 'err');
      }
      // Refresh selected run after revert (the VFS changed)
      if(STATE.selected && msg.agent_name===STATE.selected.agent && msg.run_id===STATE.selected.run_id){
        wsSend({type:'get_run', agent_name:msg.agent_name, run_id:msg.run_id});
      }
      break;
    case 'vfs_capable':
      // Server can also push this proactively after register
      STATE.vfsCapable = new Set(msg.agents||[]);
      break;
  }
}

// ── UI helpers (from obs_viewer) ──
function toggleStep(id){document.getElementById(id)?.classList.toggle('minimized')}
function toggleAllSteps(min){document.querySelectorAll('.step-tl .step').forEach(el=>el.classList.toggle('minimized',min))}
function toggleDetail(id){document.getElementById(id)?.classList.toggle('open')}
function switchLlmTab(llmId, tab){
  const p=document.getElementById(llmId); if(!p) return;
  p.querySelectorAll('.llm-detail-tab').forEach(t=>t.classList.toggle('active',t.textContent.toLowerCase()===tab));
  const i=document.getElementById(llmId+'-input'), o=document.getElementById(llmId+'-output');
  if(i) i.style.display=tab==='input'?'':'none';
  if(o) o.style.display=tab==='output'?'':'none';
}

// ════════════════════════════════════════════════════════════════════
// SEARCH
// ════════════════════════════════════════════════════════════════════
function bindSearch(){
  const inp=document.getElementById('search-in');
  inp.addEventListener('input', ()=>{
    const q=inp.value.trim();
    STATE.searchQ=q;
    if(STATE.searchTimer) clearTimeout(STATE.searchTimer);
    if(!q){
      document.getElementById('search-results').innerHTML='';
      return;
    }
    STATE.searchTimer=setTimeout(()=>{
      wsSend({type:'search', q});
    }, 300);
  });
}

function renderSearchResults(q, results, truncated){
  const c=document.getElementById('search-results');
  if(!results || !results.length){
    c.innerHTML=`<div class="sr-hdr">no matches for "${esc(q)}"</div>`;
    return;
  }
  let h=`<div class="sr-hdr">${results.length} match${results.length===1?'':'es'}${truncated?' (truncated)':''}</div>`;
  results.forEach(r=>{
    const where = r.step_id!=null?`step ${r.step_id}`:(r.run_id?'run':'agent');
    h+=`<div class="sr-item" onclick="onSearchResultClick('${esc(r.agent||'')}','${esc(r.run_id||'')}',${r.step_id!=null?r.step_id:'null'})">
      <span class="sr-key">${esc(r.matched||'?')}</span>
      <span class="sr-snip"><span class="sr-ag">${esc(r.agent||'')}</span> · ${esc(r.run_id?r.run_id.substring(0,12):'—')} · ${esc(where)} — ${esc(r.snippet||'')}</span>
    </div>`;
  });
  c.innerHTML=h;
}

function onSearchResultClick(agent, runId, stepId){
  if(!agent) return;
  if(!runId){
    // Agent match — just expand
    STATE.collapsed.delete(agent);
    renderSidebar();
    return;
  }
  // Determine kind: live if currently in liveRuns or in agent's interrupted list
  let kind='complete';
  const key=agent+'::'+runId;
  if(STATE.liveRuns[key]) kind='live';
  else {
    const ag=STATE.agents[agent];
    if(ag && (ag.interrupted||[]).some(ir=>ir.run_id===runId)) kind='live';
  }
  STATE.collapsed.delete(agent);
  selectRun(agent, runId, kind);
  if(stepId!=null){
    // Scroll to the step once rendered
    setTimeout(()=>{
      const el=document.getElementById(agent+'-'+runId+'-step-'+stepId);
      el?.scrollIntoView({behavior:'smooth',block:'center'});
      if(el){
        el.style.outline='1px solid var(--warning)';
        setTimeout(()=>{el.style.outline=''}, 2000);
      }
    }, 150);
  }
}

// ════════════════════════════════════════════════════════════════════
// REVERT
// ════════════════════════════════════════════════════════════════════
function doRevert(agent, runId, mode, extra){
  const label=mode==='all'?'ALL changes in this run':
              mode==='step'?`step ${extra.step_id}`:
              mode==='file'?`file ${extra.path}`:
              mode==='delta'?`delta #${extra.delta_index}`:mode;
  if(!confirm(`Undo ${label}? This modifies the VFS.`)) return;
  wsSend({type:'revert', agent_name:agent, run_id:runId, mode, ...extra});
}

function toast(msg, kind){
  const t=document.createElement('div');
  t.className='toast '+(kind||'');
  t.textContent=msg;
  document.body.appendChild(t);
  setTimeout(()=>{t.style.opacity='0';t.style.transition='opacity 200ms linear'},2500);
  setTimeout(()=>t.remove(),3000);
}

// ── INIT ──
bindSearch();
wsConnect();
// Periodically re-render sidebar so freshness state (running → idle) decays naturally
setInterval(()=>{ if(Object.keys(STATE.agents).length) renderSidebar(); }, 5000);
</script>
</body>
</html>
"""


# =============================================================================
# WSGI / Standalone
# =============================================================================

async def main(host="127.0.0.1", port=7000, wit_static=True):
    print("\nRoutes:")
    for r in app.list_routes():
        print(f"  {r['method'].ljust(6)} {r['path']}")
    print("\nStandalone:")
    print("  uvicorn toolboxv2.mods.isaa.extras.live_obs_server:wsgi_app --port 8000")
    print("\nWire agents (live push):")
    print("  from toolboxv2.mods.isaa.extras.live_obs_server import register_agent_obs")
    print("  register_agent_obs(my_flow_agent)")
    print("\nDisk-only fallback (no live agents required):")
    print("  from toolboxv2.mods.isaa.extras.live_obs_server import start_disk_scanner")
    print("  start_disk_scanner('/path/to/Agents', interval=2.0)\n")

    if wit_static:
        start_disk_scanner(str(Path(get_app().data_dir) / "Agents"))

    await app.serve_async(host=host, port=port, blocking=False, module_path="obs")

if __name__ == "__main__":
    main()
