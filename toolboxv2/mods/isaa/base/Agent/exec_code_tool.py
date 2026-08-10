"""
exec_code — the agent's live Python harness.

Two modes over one persistent environment:

    execute  → run code like a Jupyter cell (state persists, top-level await)
    write    → persist an importable harness module and auto-register its
               @tool functions into the agent's ToolManager

Plus: ``state`` (introspection), ``read``, ``remove``, ``reset``.

No LLM is involved anywhere in this path — it is pure compile + exec.
"""

from __future__ import annotations

from typing import Any

from toolboxv2.mods.isaa.base.Agent.live_harness import LiveSession

_MODES = ("execute", "write", "state", "read", "remove", "reset")


# =============================================================================
# DYNAMIC PRESENTATION
# =============================================================================

def build_exec_code_description(live: LiveSession, max_chars: int = 1800) -> str:
    """Render the tool description from the *live* environment state.

    The agent must see what actually exists right now — its harness modules,
    the tools it wrote itself, the variables still in the namespace — not a
    static blurb.
    """
    try:
        st = live.state()
    except Exception:
        st = {}

    lines = [
        "Live Python harness (persistent Jupyter-style session). Two modes:",
        "  mode='execute' code='...'   run a cell. State persists across calls. "
        "Top-level await works. Last expression is returned as 'result'.",
        "  mode='write' name='mymod' code='...'   persist an importable module in "
        "your harness. Functions decorated @tool are registered as real agent "
        "tools immediately and survive restarts.",
        "  mode='state' | 'read' name=... | 'remove' name=... | 'reset'",
        "",
        "In-cell namespace: agent, session, vfs, tools, tool (decorator), "
        "HARNESS_DIR, WORK_DIR.",
        "Tool calls follow notebook semantics: `await tools.x(...)` for async "
        "tools, `tools.x(...)` for sync ones, `tools.x.sync(...)` to block when "
        "you cannot await. `dir(tools)` lists everything.",
    ]

    if st.get("privileged"):
        lines.append("Privileged: `app` (ToolBox app instance) and `mods.<Any>` are available.")
    else:
        allowed = st.get("mods") or []
        lines.append(
            f"Mods available via mods.<Name>: {', '.join(allowed) if allowed else 'none'} "
            "(no `app` access for this agent)."
        )

    hm = st.get("harness_modules") or []
    if hm:
        lines.append(f"Harness modules ({len(hm)}): {', '.join(hm)}")
        ht = st.get("harness_tools") or {}
        flat = [t for v in ht.values() for t in v]
        if flat:
            lines.append(f"Tools you wrote: {', '.join(flat)}")
    else:
        lines.append("Harness modules: none yet — mode='write' to create your first one.")

    fns = st.get("functions") or []
    if fns:
        lines.append(f"In-session functions: {', '.join(fns[:12])}")
    vs = st.get("variables") or []
    if vs:
        lines.append(f"In-session variables: {', '.join(vs[:12])}")

    ta = st.get("tools_available") or []
    if ta:
        lines.append(f"Callable via `await tools.X()`: {', '.join(ta[:25])}"
                     + (f" (+{len(ta) - 25} more, use dir(tools))" if len(ta) > 25 else ""))

    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars] + "\n…"


# =============================================================================
# TOOL FACTORY
# =============================================================================

def create_exec_code_tool(
    agent,
    session_id: str | None = None,
    allowed_mods: set[str] | list[str] | None = None,
    privileged: bool | None = None,
    persist: bool = True,
    default_timeout: float = 120.0,
) -> dict:
    """Build the exec_code tool definition for ``agent.add_tool(**definition)``."""

    sid = session_id or getattr(agent, "active_session", None) or "default"
    live = LiveSession.get(
        agent=agent,
        session_id=sid,
        allowed_mods=set(allowed_mods or set()),
        privileged=privileged,
        persist=persist,
    )

    async def exec_code(
        code: str = "",
        mode: str = "execute",
        name: str = "",
        append: bool = False,
        timeout: float = default_timeout,
    ) -> dict:
        """Run or write Python in your persistent live harness.

        Args:
            code: Python source. Cell code for mode='execute', module source
                  for mode='write'.
            mode: 'execute' | 'write' | 'state' | 'read' | 'remove' | 'reset'
            name: Harness module name (required for write/read/remove).
            append: For mode='write' — append to the module instead of replacing.
            timeout: Seconds before an execute cell is aborted.

        Returns:
            dict with success, plus mode-specific keys.
        """
        mode = (mode or "execute").strip().lower()
        if mode not in _MODES:
            return {"success": False, "error": f"Unknown mode '{mode}'. Use one of {_MODES}."}

        try:
            if mode == "execute":
                if not code.strip():
                    return {"success": False, "error": "mode='execute' needs `code`."}
                res = await live.execute(code, timeout=timeout)
                return res.to_dict()

            if mode == "write":
                if not name.strip():
                    return {"success": False,
                            "error": "mode='write' needs `name` (the module name)."}
                if not code.strip():
                    return {"success": False, "error": "mode='write' needs `code`."}
                return live.write_harness(name, code, append=append)

            if mode == "state":
                return {"success": True, **live.state()}

            if mode == "read":
                src = live.read_harness(name)
                if src is None:
                    return {"success": False,
                            "error": f"No harness module '{name}'. Have: {live.list_harness()}"}
                return {"success": True, "module": name, "code": src}

            if mode == "remove":
                return live.remove_harness(name)

            return live.reset(wipe_harness=bool(name == "all"))

        except Exception as e:
            import traceback
            return {"success": False,
                    "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}

    return {
        "tool_func": exec_code,
        "name": "exec_code",
        "description": build_exec_code_description(live),
        "category": ["code", "execution", "harness"],
        "flags": {"local_execution": True, "write": True, "read": True},
        "is_async": True,
        "live_test_inputs": [{"code": "x = 'health_check_ok'\nprint(x)", "mode": "execute"}],
        "result_contract": {
            "expected_type": dict,
            "semantic_check_hint": (
                "Muss ein dict mit 'success' zurückgeben. Bei success=True und "
                "mode='execute' enthält 'output' die stdout-Ausgabe; bei "
                "success=False steht in 'error' ein nicht-leerer Traceback-String."
            ),
        },
        "_live_session": live,
    }


def refresh_exec_code_description(agent, live: LiveSession | None = None) -> bool:
    """Re-render the tool description after the harness changed.

    Call this after ``mode='write'`` so the agent's next prompt shows the tools
    and modules it just created.
    """
    tm = getattr(agent, "tool_manager", None)
    if tm is None:
        return False
    entry = tm.get("exec_code")
    if entry is None:
        return False
    live = live or getattr(entry, "metadata", {}).get("live_session")
    if live is None:
        return False
    try:
        tm.update("exec_code", description=build_exec_code_description(live))
        return True
    except Exception:
        return False


def register_exec_code_tool(agent, **kw) -> dict:
    """Register exec_code on an agent and return the definition."""
    definition = create_exec_code_tool(agent, **kw)
    live = definition.pop("_live_session")
    definition.pop("is_async", None)
    agent.add_tool(**definition)
    entry = agent.tool_manager.get("exec_code")
    if entry is not None:
        entry.metadata["live_session"] = live
    live.bootstrap()
    refresh_exec_code_description(agent, live)
    return {"name": "exec_code", "live": live}
