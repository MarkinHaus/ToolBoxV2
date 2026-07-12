"""
sandbox_tools.py
================
Phase 3+4 of the VFS sandbox migration.

Replaces the mock vfs_shell command parser with a thin pass-through
("Schleuse") to a real sandbox shell. The agent gets the FULL sandbox:

  sandbox_connect    — bind to local (auto) or remote ("conn.folder" key)
  sandbox_shell      — real bash (pipes, rg, sed, git — everything)
  sandbox_code       — Jupyter python execution
  sandbox_browser    — screenshot / low-level action / info
  sandbox_view       — context-window read (line ranges) via file API
  sandbox_import     — Schleuse IN : host/VFS file  -> sandbox workdir
  sandbox_export     — Schleuse OUT: sandbox /out/  -> host (allowlist)
  sandbox_status     — backend + health + workdir

Schleuse policy:
  - import: source must be inside allowed_dirs (if configured on session)
  - export: sandbox source must live under <workdir>/out/ ; host target
    must be inside allowed_dirs (if configured)
  - every transfer is emitted to obs (schleuse.in / schleuse.out)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import json

from toolboxv2.mods.isaa.base.patch.sandbox_backend import (
    AIOSandboxBackend,
    SandboxPolicy,
    SandboxRegistry,
    get_policy,
)

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.agent_session_v2 import AgentSessionV2


def _err(msg: Any, returncode: int = 1) -> dict:
    return {"success": False, "stdout": "", "stderr": str(msg), "returncode": returncode}


def _obs_emitter(session: "AgentSessionV2"):
    """Phase 5 wiring: backend events -> ObservabilityLayer."""
    def emit(kind: str, payload: dict) -> None:
        obs = getattr(session, "obs", None) or getattr(getattr(session, "agent", None), "obs", None)
        if obs is None:
            return
        try:
            if kind in ("schleuse.in", "schleuse.out"):
                # file boundary crossing — track like a vfs delta
                path = payload.get("sb_dst") or payload.get("sb_src") or "?"
                obs.record_vfs_delta(path=path, action=kind)
            obs._audit(action=kind, resource=payload.get("workdir", "?"), details=payload)
        except Exception:
            pass
    return emit


def _backend(session: "AgentSessionV2") -> AIOSandboxBackend | dict:
    """Lazy: first call boots/binds the per-AGENT container."""
    be = getattr(session, "_sandbox_backend", None)
    if be is not None:
        return be
    agent_name = getattr(session, "agent_name", None) or "default"
    reg = SandboxRegistry.for_agent(agent_name)
    be = reg.backend_for_session(session.session_id, obs_emit=_obs_emitter(session))
    if isinstance(be, dict):  # error from lifecycle (docker missing etc.)
        return be
    session._sandbox_backend = be
    return be


# =============================================================================
# TOOL FACTORIES
# =============================================================================

def make_sandbox_connect(session: "AgentSessionV2"):
    def sandbox_connect(key: str | None = None) -> dict:
        """Connect this session to a sandbox.

        key=None            -> local per-agent Docker container (auto pull/start,
                               port auto-selected in 7000-7099), workdir /work/<session_id>
        key="conn.folder"   -> remote sandbox: conn part selects the registered
                               remote host, folder part selects /work/<folder>
        """
        try:
            if key:
                be = SandboxRegistry.connect_remote(key, obs_emit=_obs_emitter(session))
                session._sandbox_backend = be
                return {"success": True, "stdout": f"connected remote, workdir={be.workdir}",
                        "stderr": "", "returncode": 0}
            session._sandbox_backend = None
            be = _backend(session)
            if isinstance(be, dict):
                return be
            return {"success": True, "stdout": f"connected local, workdir={be.workdir}",
                    "stderr": "", "returncode": 0}
        except Exception as e:
            return _err(e)
    return sandbox_connect


def make_sandbox_shell(session: "AgentSessionV2"):
    def sandbox_shell(reason: str, command: str, timeout: float | None = None) -> dict:
        """Execute a REAL bash command in the isolated sandbox (persistent shell
        session, cwd = your workdir). Pipes, redirects, rg/grep/sed/git all work
        exactly like a normal Linux shell."""
        be = _backend(session)
        if isinstance(be, dict):
            return be
        return be.exec(command, timeout=timeout)
    return sandbox_shell


def make_sandbox_code(session: "AgentSessionV2"):
    def sandbox_code(code: str, timeout: int | None = None) -> dict:
        """Execute Python code in the sandbox Jupyter kernel (stateful between calls)."""
        be = _backend(session)
        if isinstance(be, dict):
            return be
        return be.code(code, timeout=timeout)
    return sandbox_code


def make_sandbox_browser(session: "AgentSessionV2"):
    _AREAS = {
        "page": "browser_page",        # navigate, click, fill, fill_form, evaluate(JS),
                                       # get_html, get_markdown, get_text, get_console,
                                       # get_elements, find_text, check, back, forward,
                                       # export_console, hot_key, hover, press_key, reload,
                                       # scroll, scroll_to, scroll_to_element, select_option,
                                       # type_text, uncheck, upload_file, wait
        "tabs": "browser_tabs",        # list, create, activate, close
        "network": "browser_network",  # get_requests, export_har, add_route, set_headers,
                                       # set_scoped_headers, remove_route
        "cookies": "browser_cookies",  # get_cookies, set_cookies, clear_cookies
        "state": "browser_state",      # save, load
        "captcha": "browser_captcha",  # detect, wait
    }

    # P1: LLM-friendly synonyms -> SDK uppercase enum values
    _ACTION_TYPE_MAP = {
        "move": "MOVE_TO", "move_to": "MOVE_TO", "moveto": "MOVE_TO",
        "move_rel": "MOVE_REL", "moverel": "MOVE_REL",
        "click": "CLICK", "tap": "CLICK",
        "mouse_down": "MOUSE_DOWN", "mousedown": "MOUSE_DOWN",
        "mouse_up": "MOUSE_UP", "mouseup": "MOUSE_UP",
        "right_click": "RIGHT_CLICK", "rightclick": "RIGHT_CLICK",
        "double_click": "DOUBLE_CLICK", "doubleclick": "DOUBLE_CLICK",
        "drag_to": "DRAG_TO", "dragto": "DRAG_TO",
        "drag_rel": "DRAG_REL", "dragrel": "DRAG_REL",
        "scroll": "SCROLL",
        "type": "TYPING", "typing": "TYPING", "input": "TYPING", "write": "TYPING",
        "press": "PRESS", "keypress": "PRESS",
        "key_down": "KEY_DOWN", "keydown": "KEY_DOWN",
        "key_up": "KEY_UP", "keyup": "KEY_UP",
        "hotkey": "HOTKEY", "hot_key": "HOTKEY",
        "wait": "WAIT",
    }

    def _normalize_action(action: dict) -> dict:
        """P1: normalize action_type; P2: disable clipboard for Linux container."""
        at = action.get("action_type")
        if isinstance(at, str):
            key = at.lower().strip()
            action["action_type"] = _ACTION_TYPE_MAP.get(key, at.upper().strip())
        # P2: TYPING defaults use_clipboard=True which needs clip.exe (not in Linux)
        if action.get("action_type") == "TYPING" and "use_clipboard" not in action:
            action["use_clipboard"] = False
        return action

    def _serialize_response(r) -> dict:
        """P3+P4+P8+P9: extract data, check success, serialize as JSON, include hint."""
        # P4: propagate SDK success=False as error
        if hasattr(r, "success") and r.success is False:
            msg = getattr(r, "message", None) or "browser operation failed"
            return {"success": False, "stdout": "", "stderr": msg, "returncode": 1}
        # Extract .data with fallback
        raw = getattr(r, "data", r)
        # P8: convert Pydantic models to plain dict
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        elif hasattr(raw, "dict") and callable(getattr(raw, "dict", None)):
            try:
                raw = raw.dict()
            except Exception:
                pass
        # P3: serialize as JSON for structured data, str() for primitives
        if isinstance(raw, (dict, list)):
            stdout = json.dumps(raw, ensure_ascii=False, default=str)
        else:
            stdout = str(raw) if raw is not None else ""
        result = {"success": True, "stdout": stdout, "stderr": "", "returncode": 0}
        # P9: include SDK hint field if present
        hint = getattr(r, "hint", None)
        if hint:
            result["hint"] = hint
        return result

    def sandbox_browser(area: str = "info", method: str | None = None,
                        args: dict | None = None, **kwargs) -> dict:
        """FULL sandbox browser incl. devtools — thin wrapper over the sandbox API.

        area='screenshot' | 'info' | 'restart' | 'action' (raw input action via args)
        area='page'|'tabs'|'network'|'cookies'|'state'|'captcha' + method + args:
          page.navigate {"url": "https://..."}                  -> navigate to URL (start here)
          page.evaluate {"expression": "document.title"}        -> run JS (devtools console)
          page.get_console {}                              -> console logs
          page.get_html / get_markdown / get_text          -> page content
          page.click / fill / fill_form / find_text        -> interaction
          network.get_requests / export_har                -> devtools network tab
          tabs.list / create {"url": ...} / activate       -> tab control
        Wrong method name -> returns the list of available methods for that area."""
        be = _backend(session)
        if isinstance(be, dict):
            return be
        args = args or {}
        try:
            if area == "screenshot":
                return be.browser_screenshot()
            if area == "info":
                return be.browser_info()
            if area == "restart":
                r = be.client.browser.restart()
                # P10: restart returns Response with data=None; return clean message
                return {"success": True, "stdout": "browser restarted", "stderr": "", "returncode": 0}
            if area == "action":
                # P1+P2: normalize action_type and disable clipboard
                return be.browser_action(_normalize_action(args))
            sub_name = _AREAS.get(area)
            if sub_name is None:
                return _err(f"unknown area '{area}' — use {list(_AREAS) + ['screenshot', 'info', 'action', 'restart']}")
            sub = getattr(be.client, sub_name)
            if not method or not hasattr(sub, method) or method.startswith("_"):
                avail = [m for m in dir(sub) if not m.startswith("_") and m != "with_raw_response"]
                return _err(f"method '{method}' not in area '{area}'. Available: {avail}")
            # P6: merge args+kwargs to avoid TypeError on duplicate keys
            merged = {**args, **kwargs}
            r = getattr(sub, method)(**merged)
            # P3+P4+P8+P9: unified response serialization
            return _serialize_response(r)
        except Exception as e:
            return _err(e)
    return sandbox_browser


def make_sandbox_view(session: "AgentSessionV2"):
    def sandbox_view(path: str, line_start: int | None = None, line_end: int | None = None) -> dict:
        """Read a sandbox file (optionally a line range) into the context window."""
        be = _backend(session)
        if isinstance(be, dict):
            return be
        r = be.read(path, start_line=line_start, end_line=line_end)
        if r["success"]:
            r["showing"] = f"{path}:{line_start or 1}-{line_end or 'end'}"
        return r
    return sandbox_view


def make_sandbox_import(session: "AgentSessionV2"):
    def sandbox_import(source_path: str, sandbox_path: str | None = None,
                       from_vfs: bool = False) -> dict:
        """Schleuse IN — copy a file INTO the sandbox workdir.
        source_path: host path (or VFS path if from_vfs=True).
        sandbox_path: target relative to workdir (default: basename)."""
        be = _backend(session)
        if isinstance(be, dict):
            return be
        try:
            import os
            if from_vfs:
                vfs = session.vfs
                content = vfs.files.get(vfs._normalize_path(source_path))
                if content is None:
                    return _err(f"VFS file not found: {source_path}")
                data = getattr(content, "content", None) or ""
                dst = sandbox_path or source_path.lstrip("/")
                return be.write(dst, data)
            pol = get_policy(session)
            allowed = pol.allowed_import_dirs
            if allowed and not any(os.path.abspath(source_path).startswith(os.path.abspath(a))
                                   for a in allowed):
                return _err(f"import blocked by policy: {source_path} outside "
                            f"allowed_import_dirs={allowed}")
            if os.path.isfile(source_path) and \
                    os.path.getsize(source_path) > pol.max_transfer_mb * 1024 * 1024:
                return _err(f"import blocked by policy: file > {pol.max_transfer_mb}MB")
            dst = sandbox_path or os.path.basename(source_path.rstrip("/\\"))
            if os.path.isdir(source_path):
                return be.upload_dir(source_path, dst)
            return be.upload(source_path, dst)
        except Exception as e:
            return _err(e)
    return sandbox_import


def make_sandbox_export(session: "AgentSessionV2"):
    def sandbox_export(sandbox_path: str, host_path: str) -> dict:
        """Schleuse OUT — copy a file FROM the sandbox to the host.
        Only files under <workdir>/out/ may leave the sandbox.
        host_path must be inside allowed_dirs if configured."""
        be = _backend(session)
        if isinstance(be, dict):
            return be
        try:
            import os
            pol = get_policy(session)
            src_abs = be._abs(sandbox_path)
            if pol.export_prefix:
                out_prefix = be.workdir.rstrip("/") + "/" + pol.export_prefix.strip("/") + "/"
                if not (src_abs.startswith(out_prefix) or src_abs + "/" == out_prefix.rstrip("/") + "/"):
                    return _err(f"export blocked by policy: only {out_prefix}* may leave the "
                                f"sandbox (move the file there first, or host sets export_prefix='')")
            allowed = pol.allowed_export_dirs
            if allowed and not any(os.path.abspath(host_path).startswith(os.path.abspath(a))
                                   for a in allowed):
                return _err(f"export blocked by policy: {host_path} outside "
                            f"allowed_export_dirs={allowed}")
            probe = be.exec(f"test -d '{src_abs}' && echo DIR || echo FILE")
            if probe.get("success") and "DIR" in probe.get("stdout", ""):
                return be.download_dir(sandbox_path, host_path)  # arrives as .zip
            return be.download(sandbox_path, host_path)
        except Exception as e:
            return _err(e)
    return sandbox_export


def make_sandbox_edit(session: "AgentSessionV2"):
    def sandbox_edit(command: str, path: str, file_text: str | None = None,
                     old_str: str | None = None, new_str: str | None = None,
                     insert_line: int | None = None,
                     view_range: list[int] | None = None,
                     replace_mode: str | None = None) -> dict:
        """PRECISE file editing in the sandbox (server-side, no read-modify-write race).

        command='view'        + optional view_range=[start, end]
        command='create'      + file_text
        command='str_replace' + old_str (must be unique) + new_str
                                (replace_mode='ALL'|'FIRST'|'LAST' for non-unique)
        command='insert'      + insert_line + new_str
        command='undo_edit'   — revert the last edit on this file"""
        be = _backend(session)
        if isinstance(be, dict):
            return be
        kw = {k: v for k, v in (("file_text", file_text), ("old_str", old_str),
                                ("new_str", new_str), ("insert_line", insert_line),
                                ("view_range", view_range), ("replace_mode", replace_mode))
              if v is not None}
        return be.edit(command, path, **kw)
    return sandbox_edit


def make_sandbox_permissions(session: "AgentSessionV2"):
    def sandbox_permissions() -> dict:
        """Show EXACTLY what you may do — the enforced policy, your workdir,
        and the live host mounts. Set from outside via session.sandbox_policy
        or TB_SANDBOX_* env vars; you cannot change it from in here."""
        from toolboxv2.mods.isaa.base.patch.sandbox_backend import configured_mounts
        pol = get_policy(session)
        be = getattr(session, "_sandbox_backend", None)
        info = {
            "policy": pol.to_dict(),
            "workdir": getattr(be, "workdir", f"/work/{session.session_id} (after connect)"),
            "backend": getattr(be, "label", "not connected"),
            "host_mounts": [{"host": h, "sandbox": c, "live": True} for h, c in configured_mounts()],
            "notes": [
                "Inside the sandbox you have FULL control (shell, code, browser, edits).",
                "Only the host boundary (Schleuse) is gated: see allowed_import_dirs / "
                "allowed_export_dirs / export_prefix / max_transfer_mb.",
                "/work/global is shared live across ALL agents and sessions.",
            ],
        }
        import json as _json
        return {"success": True, "stdout": _json.dumps(info, indent=2),
                "stderr": "", "returncode": 0, "permissions": info}
    return sandbox_permissions


def make_sandbox_status(session: "AgentSessionV2"):
    def sandbox_status() -> dict:
        """Backend label, workdir, health."""
        be = getattr(session, "_sandbox_backend", None)
        if be is None:
            return {"success": True, "stdout": "not connected (call sandbox_connect "
                    "or any sandbox tool to auto-connect)", "stderr": "", "returncode": 0}
        h = be.health()
        h["stdout"] = f"{be.label} workdir={be.workdir} :: {h['stdout'] or h['stderr']}"
        return h
    return sandbox_status


# =============================================================================
# REGISTRATION HELPER (used by init_session_tools)
# =============================================================================

SANDBOX_TOOL_HEALTH = {
    "sandbox_shell": {
        "live_test_inputs": [{"reason": "health check", "command": "echo ok"}],
        "result_contract": {
            "allow_none": False, "expected_type": "dict",
            "semantic_check_hint": "Keys success/stdout/stderr/returncode. echo ok -> stdout contains 'ok'.",
        },
        "cleanup_func": None,
    },
    "sandbox_view": {
        "live_test_inputs": [{"path": "/etc/hostname", "line_start": 1, "line_end": 1}],
        "result_contract": {
            "allow_none": False, "expected_type": "dict",
            "semantic_check_hint": "success=True -> non-empty 'content' and 'showing'.",
        },
        "cleanup_func": None,
    },
    "sandbox_code": {
        "live_test_inputs": [{"code": "print(1+1)"}],
        "result_contract": {
            "allow_none": False, "expected_type": "dict",
            "semantic_check_hint": "success=True and stdout contains '2'.",
        },
        "cleanup_func": None,
    },
    "sandbox_status": {
        "live_test_inputs": [{}],
        "result_contract": {
            "allow_none": False, "expected_type": "dict",
            "semantic_check_hint": "Must contain success (bool) and stdout (str, non-empty).",
        },
        "cleanup_func": None,
    },
    "sandbox_connect": {"flags": {"guaranteed_healthy": True}},
    "sandbox_browser": {"flags": {"guaranteed_healthy": True}},
    "sandbox_import": {"flags": {"guaranteed_healthy": True}},
    "sandbox_export": {"flags": {"guaranteed_healthy": True}},
    "sandbox_edit": {"flags": {"guaranteed_healthy": True}},
    "sandbox_permissions": {
        "live_test_inputs": [{}],
        "result_contract": {
            "allow_none": False, "expected_type": "dict",
            "semantic_check_hint": "Must contain 'permissions' dict with 'policy' and 'workdir'.",
        },
        "cleanup_func": None,
    },
}


def build_sandbox_tools(session: "AgentSessionV2") -> list[dict]:
    """Full toolset — drop-in for init_session_tools()."""
    return [
        {"tool_func": make_sandbox_connect(session), "name": "sandbox_connect",
         "category": ["sandbox"], "description": (
             "Connect to a sandbox. No key = local isolated container (auto-managed). "
             "key='<conn>.<folder>' = remote sandbox with your folder namespace.")},
        {"tool_func": make_sandbox_shell(session), "name": "sandbox_shell",
         "category": ["sandbox", "shell"], "description": (
             "REAL bash in an isolated sandbox (persistent session, cwd=workdir). "
             "Use it exactly like a normal Linux shell: ls, cat, rg, sed -i, git, "
             "pip install, python script.py. Returns {success, stdout, stderr, returncode}.")},
        {"tool_func": make_sandbox_code(session), "name": "sandbox_code",
         "category": ["sandbox", "code"], "description": (
             "Run Python in the sandbox Jupyter kernel. State persists between calls.")},
        {"tool_func": make_sandbox_browser(session), "name": "sandbox_browser",
         "category": ["sandbox", "web", "browser"], "description": (
             "FULL browser control incl. devtools. area+method+args: "
             "page (navigate/click/fill/evaluate(JS)/get_html/get_markdown/get_console/find_text), "
             "network (get_requests/export_har), tabs, cookies, state, captcha — "
             "plus 'screenshot'|'info'|'action'|'restart'. Wrong method returns the "
             "available method list.")},
        {"tool_func": make_sandbox_view(session), "name": "sandbox_view",
         "category": ["sandbox", "context"], "description": (
             "Read a sandbox file (line ranges) into the context window.")},
        {"tool_func": make_sandbox_import(session), "name": "sandbox_import",
         "category": ["sandbox", "schleuse"], "flags": {"filesystem_access": True},
         "description": ("Schleuse IN: copy a host/VFS file OR a whole host directory "
                         "into the sandbox workdir (dirs auto-tar'd). Note: /work/global "
                         "is a live shared mount across all agents — no import needed there.")},
        {"tool_func": make_sandbox_export(session), "name": "sandbox_export",
         "category": ["sandbox", "schleuse"], "flags": {"filesystem_access": True},
         "description": ("Schleuse OUT: export a sandbox file or whole directory "
                         "(directories arrive as .zip). Only <workdir>/out/* may leave. "
                         "Alternative for code: git push from inside sandbox_shell.")},
        {"tool_func": make_sandbox_edit(session), "name": "sandbox_edit",
         "category": ["sandbox", "edit"], "description": (
             "PRECISE edits on sandbox files: view/create/str_replace/insert/undo_edit "
             "(server-side editor — use this for code changes instead of cat/sed).")},
        {"tool_func": make_sandbox_permissions(session), "name": "sandbox_permissions",
         "category": ["sandbox", "diagnostics"], "description": (
             "Show your enforced permissions: Schleuse policy, workdir, live host mounts. "
             "Full control inside the sandbox; only the host boundary is gated.")},
        {"tool_func": make_sandbox_status(session), "name": "sandbox_status",
         "category": ["sandbox", "diagnostics"], "description": "Backend, workdir, health."},
    ]
