#!/usr/bin/env python3
"""
toolboxv2/utils/workers/fast/local_ui.py - Local Default UI (FastTB + HTMX)

The thin local entry. Three primary screens for unauth users
(Login / Services / Mods). After auth → local-root view that links into
the existing /mainPagen.html and beyond.

LOCAL-ONLY. Refuses requests where Host header is not loopback.
The public web (existing /mainPagen.html and friends) is untouched.

Style: TBJS Glass v3.0 — embedded fallback tokens; if /dist/tbjs/main.css
is present it overrides cleanly.

Mount: included by HTTPWorker at "/" (see WIRE-IN at bottom of file).
"""

import asyncio
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from toolboxv2 import Result, get_app, tb_root_dir
from toolboxv2.utils.workers.fast_tb import FastTB
from toolboxv2.utils.workers.server_worker import ParsedRequest
from toolboxv2.utils.workers.session import SessionData

app = FastTB(title="ToolBox Local")
# Global auth on: every route requires an authenticated session except those
# explicitly marked auth=False (the local sign-in flow under "/" and /local-ui/auth/*)
# and /health. The local UI no longer gates routes itself via _require_auth; the
# FastTB guard enforces it uniformly.
# app.auth = True
# The local UI is the local origin, so the login page (login.html) is served from
# our own dist/web rather than the remote site — mount it permanently.
app.serve_login_assets = True
app.inject_style = False

# =============================================================================
# Helpers
# =============================================================================

def _ensure_local(request: ParsedRequest) -> Optional[Result]:
    """Refuse anything that isn't loopback. Returns Result-error or None."""
    host = (request.headers.get("host") or "").split(":")[0].lower()
    if host not in ("127.0.0.1", "localhost", "[::1]", "::1"):
        return Result.default_user_error(
            info="This UI is local-only. Use the public site for remote access.",
            exec_code=403,
        )
    return None


def _user_facing(feature_yaml: Path) -> List[Dict[str, Any]]:
    """Read `user_facing` list from a feature.yaml — empty if missing.

    Each entry needs `name` (mod) and `function` (callable on the mod). Both
    are passed to `a_run_any((name, function))` when the user starts the mod.
    `ui_path` is optional; if absent, no "open" link is shown (the mod might
    serve its UI on a different port, or have no UI at all — author decides).
    """
    try:
        data = yaml.safe_load(feature_yaml.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    out = []
    for entry in data.get("user_facing", []) or []:
        if not isinstance(entry, dict) or "name" not in entry:
            continue
        out.append({
            "name": entry["name"],
            "function": entry.get("function") or entry.get("entry_point") or "start",
            "label": entry.get("label", entry["name"]),
            "icon": entry.get("icon", "▸"),
            "description": entry.get("description", ""),
            "ui_path": entry.get("ui_path"),  # may be None — no auto-default
            "feature": data.get("name", "unknown"),
        })
    return out


def _all_user_facing_mods() -> List[Dict[str, Any]]:
    """Aggregate user_facing across every installed feature."""
    features_dir = tb_root_dir / "features"
    if not features_dir.exists():
        return []
    out = []
    for fyaml in features_dir.glob("*/feature.yaml"):
        installed_marker = fyaml.parent / ".installed"
        if not installed_marker.exists():
            continue
        out.extend(_user_facing(fyaml))
    return out


def _session_user(session: Optional[SessionData]) -> Optional[str]:
    """Pull username off the session if any. JWT validated by HTTPWorker upstream."""
    if not session:
        return None
    data = getattr(session, "data", None) or {}
    # session_data is populated by HTTPWorker after JWT validation
    return data.get("user_name") or data.get("username") or data.get("user") or None


# Local-admin first-run = a magic-link token written directly into the
# AUTH_MAGIC_LINK::{token} DB namespace and printed to STDOUT instead of email.
# The user then submits it; verify_magic_link() (existing endpoint) consumes it,
# auto-creates UserData(email="local-admin@toolbox.local"), and returns JWT.
LOCAL_ADMIN_EMAIL = "local-admin@toolbox.local"
_FIRST_RUN_TOKEN_CACHE: Dict[str, float] = {}


async def _has_any_user() -> bool:
    """True if at least one user already exists in AUTH_USER namespace."""
    try:
        tb_app = get_app("local_ui.has_user")
        result = tb_app.run_any(("CloudM.Auth", "list_users"), get_results=True)
        if hasattr(result, "is_error") and result.is_error():
            return False
        data = result.get() if hasattr(result, "get") else result
        return bool(data) and len(data) > 0
    except Exception:
        return False


async def _trigger_first_run(host_url: str) -> Optional[str]:
    """Generate a magic-link token in DB, then ask Tauri (or the browser) to
    open the setup URL — never print the token to STDOUT. The token rides
    inside the URL as ?setup_token=… and is auto-verified by the index route.

    If a SSE subscriber is connected (Tauri), only emit the open_url event.
    If nobody is subscribed (headless / browser-only), fall back to
    webbrowser.open() so the user still lands at the right page.
    """
    now = time.time()
    # Re-use a still-valid token if we already have one cached in-process
    for tok, ts in list(_FIRST_RUN_TOKEN_CACHE.items()):
        if now - ts < 9 * 60:
            return tok
        del _FIRST_RUN_TOKEN_CACHE[tok]

    try:
        from toolboxv2.mods.CloudM.auth.db_helpers import _db_set
    except Exception as e:
        print(f"[local_ui] cannot import auth db_helpers: {e}", flush=True)
        return None

    tb_app = get_app("local_ui.first_run_token")
    token = secrets.token_urlsafe(32)
    await _db_set(tb_app, f"AUTH_MAGIC_LINK::{token}", {
        "email": LOCAL_ADMIN_EMAIL,
        "created_at": now,
        "verified": False,
        "local_admin": True,
    })
    _FIRST_RUN_TOKEN_CACHE[token] = now

    setup_url = f"{host_url}/?setup_token={token}"

    # Try Tauri first via tray-API
    from .tray_api import emit_open_url, has_active_subscribers
    emit_open_url(setup_url, target="main")

    # Give Tauri a brief moment to react; if nobody is subscribed, fall back
    # to the system browser so the URL still gets opened.
    await asyncio.sleep(0.3)
    if not has_active_subscribers():
        try:
            import webbrowser
            webbrowser.open(setup_url)
        except Exception as e:
            logger_fn = print  # avoid pulling logging into this hot path
            logger_fn(f"[local_ui] webbrowser.open failed: {e}", flush=True)

    return token


# =============================================================================
# CSS — TBJS Glass v3.0 essentials (embedded fallback, linked if available)
# =============================================================================

CSS = """
:root[data-theme="dark"] {
  --raw-primary: 55% 0.18 230;
  --raw-success: 65% 0.20 145;
  --raw-warning: 75% 0.18 85;
  --raw-error:   55% 0.22 25;
  --primary: oklch(var(--raw-primary));
  --success: oklch(var(--raw-success));
  --warning: oklch(var(--raw-warning));
  --error:   oklch(var(--raw-error));

  --bg-base: #08080d;
  --bg-elevated: rgba(15, 15, 25, 0.9);
  --bg-sunken: rgba(0, 0, 0, 0.3);
  --glass-bg: rgba(255, 255, 255, 0.02);
  --glass-border: rgba(255, 255, 255, 0.05);
  --glass-blur: 12px;
  --border-subtle: rgba(255, 255, 255, 0.08);

  --text-main: rgba(255, 255, 255, 0.85);
  --text-body: rgba(255, 255, 255, 0.7);
  --text-label: rgba(255, 255, 255, 0.4);
  --text-muted: rgba(255, 255, 255, 0.25);

  --highlight-inset: inset 0 1px 0 rgba(255, 255, 255, 0.05);
  --shadow-micro: 0 2px 4px rgba(0, 0, 0, 0.5);

  --surface-hover: color-mix(in oklch, var(--primary) 5%, transparent);
  --surface-active: color-mix(in oklch, var(--primary) 10%, transparent);
  --surface-badge: color-mix(in oklch, var(--primary) 15%, transparent);
  --border-active: color-mix(in oklch, var(--primary) 30%, transparent);

  --font-sans: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
  --font-mono: 'IBM Plex Mono', ui-monospace, Consolas, monospace;

  --text-h1: clamp(22px, 2.5vw, 28px);
  --text-h2: clamp(20px, 2.25vw, 24px);
  --text-h3: clamp(17px, 1.875vw, 20px);
  --text-base: 16px;
  --text-sm: 14px;
  --text-xs: 11px;

  --radius-sm: 5px;
  --radius-md: 8px;
  --radius-lg: 12px;

  --space-1: 5px;  --space-2: 10px; --space-3: 15px;
  --space-4: 20px; --space-5: 30px; --space-6: 40px;
  --space-8: 60px; --space-10: 80px; --space-12: 120px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { background: var(--bg-base); color: var(--text-main); }
body {
  font-family: var(--font-sans);
  font-size: var(--text-base);
  line-height: 1.5;
  min-height: 100vh;
}

.shell {
  max-width: 900px;
  margin: 0 auto;
  padding: var(--space-6) var(--space-4);
}

h1 { font-size: var(--text-h1); font-weight: 700; letter-spacing: -0.02em; }
h2 { font-size: var(--text-h2); font-weight: 700; letter-spacing: -0.02em; }
h3 { font-size: var(--text-h3); font-weight: 600; letter-spacing: -0.02em; }
p  { color: var(--text-body); margin-block-end: var(--space-3); max-inline-size: 65ch; }

.label, h6 {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 2.5px;
  color: var(--text-label);
  font-weight: 500;
}

a { color: var(--primary); text-decoration: none; }
a:hover { text-decoration: underline; }

/* Header */
.app-head {
  display: flex; align-items: center; justify-content: space-between;
  padding-block-end: var(--space-5);
  margin-block-end: var(--space-5);
  border-block-end: 1px solid var(--border-subtle);
}
.app-head .brand { display: flex; align-items: baseline; gap: var(--space-3); }
.app-head .brand h1 { line-height: 1; }
.app-head .brand .label { line-height: 1; }
.app-head nav { display: flex; gap: var(--space-2); }

/* Card / glass */
.card {
  background: var(--glass-bg);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-lg);
  padding: var(--space-5);
  backdrop-filter: blur(var(--glass-blur));
  box-shadow: var(--highlight-inset), var(--shadow-micro);
}
.card + .card { margin-block-start: var(--space-3); }
.card-row {
  display: flex; align-items: center; justify-content: space-between;
  gap: var(--space-4); padding-block: var(--space-3);
}
.card-row + .card-row { border-block-start: 1px solid var(--border-subtle); }
.card-row .meta {
  display: flex; flex-direction: column; gap: 2px; min-width: 0;
}
.card-row .name { color: var(--text-main); font-weight: 500; }
.card-row .desc { color: var(--text-muted); font-size: var(--text-sm); }

/* Buttons */
.btn {
  display: inline-flex; align-items: center; gap: var(--space-2);
  padding: var(--space-2) var(--space-4);
  background: var(--glass-bg);
  border: 1px solid var(--glass-border);
  color: var(--text-main);
  border-radius: var(--radius-md);
  font: 500 var(--text-sm) var(--font-sans);
  cursor: pointer;
  box-shadow: var(--highlight-inset);
  transition: transform 120ms ease, background 120ms ease, border-color 120ms ease;
}
.btn:hover { background: var(--surface-hover); transform: translateY(-1px); }
.btn:active { transform: none; }
.btn:focus-visible { outline: 1px solid var(--primary); outline-offset: 2px; }
.btn-primary { background: var(--surface-badge); border-color: var(--border-active); color: var(--text-main); }
.btn-primary:hover { background: var(--surface-active); }
.btn-ghost { background: transparent; border-color: transparent; }
.btn-ghost:hover { background: var(--surface-hover); border-color: var(--glass-border); }
.btn-danger { color: var(--error); border-color: color-mix(in oklch, var(--error) 30%, transparent); }
.btn-danger:hover { background: color-mix(in oklch, var(--error) 10%, transparent); }

/* Inputs */
input[type="text"], input[type="password"], input[type="email"] {
  width: 100%;
  padding: var(--space-2) var(--space-3);
  background: var(--bg-sunken);
  border: 1px solid var(--glass-border);
  color: var(--text-main);
  border-radius: var(--radius-md);
  font: var(--text-base) var(--font-sans);
}
input:focus { outline: 1px solid var(--primary); outline-offset: 1px; border-color: var(--border-active); }
input[type="text"][data-mono], input[type="password"][data-mono] { font-family: var(--font-mono); }
.field { display: flex; flex-direction: column; gap: var(--space-2); margin-block-end: var(--space-4); }
.field > label { color: var(--text-label); font: var(--text-xs) var(--font-mono); text-transform: uppercase; letter-spacing: 2.5px; }

/* Status dots */
.dot {
  display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  vertical-align: middle; margin-inline-end: var(--space-2);
  background: var(--text-muted);
}
.dot.on  { background: var(--success); box-shadow: 0 0 4px color-mix(in oklch, var(--success) 40%, transparent); }
.dot.off { background: var(--text-muted); }
.dot.err { background: var(--error); }

/* Grid layouts */
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-3); }
@media (max-width: 600px) { .grid-2 { grid-template-columns: 1fr; } }

/* Nav buttons (login providers) */
.providers { display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-3); margin-block-start: var(--space-4); }
@media (max-width: 480px) { .providers { grid-template-columns: 1fr; } }

/* Hint footer */
.hint {
  color: var(--text-muted);
  font: var(--text-sm) var(--font-mono);
  margin-block-start: var(--space-5);
  padding-block-start: var(--space-3);
  border-block-start: 1px dashed var(--border-subtle);
}
.hint code {
  color: var(--text-label);
  background: var(--bg-sunken);
  padding: 1px 6px;
  border-radius: var(--radius-sm);
}

/* Toast (HTMX response area) */
#toast {
  position: fixed; right: var(--space-4); bottom: var(--space-4);
  max-width: 320px;
  z-index: 50;
}
.toast-msg {
  background: var(--bg-elevated);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-md);
  padding: var(--space-3) var(--space-4);
  font-size: var(--text-sm);
  box-shadow: var(--highlight-inset), 0 10px 30px rgba(0,0,0,0.7);
  margin-block-start: var(--space-2);
  animation: toast-in 200ms ease-out;
}
.toast-msg.is-error { color: var(--error); }
.toast-msg.is-ok { color: var(--success); }
@keyframes toast-in { from { transform: translateY(8px); opacity: 0; } to { transform: none; opacity: 1; } }

/* Empty state */
.empty {
  text-align: center;
  padding: var(--space-8) var(--space-4);
  color: var(--text-muted);
}

/* Section header */
.section-head {
  display: flex; align-items: baseline; justify-content: space-between;
  margin-block-end: var(--space-3);
}
.section-head .label { margin: 0; }

/* HTMX request indicator */
.htmx-indicator { opacity: 0; transition: opacity 200ms ease; }
.htmx-request .htmx-indicator { opacity: 1; }
"""


# =============================================================================
# HTML shell + JS
# =============================================================================

JS = """
// Auto-detect Tauri (for future hooks)
window.__TB_IS_TAURI__ = !!(window.__TAURI__ || window.__TAURI_INTERNALS__);

// Toast helper for HTMX responses
document.body.addEventListener('htmx:responseError', (e) => {
  showToast(e.detail.xhr.responseText || 'Request failed', 'error');
});
document.body.addEventListener('htmx:afterRequest', (e) => {
  const trigger = e.detail.xhr.getResponseHeader('X-Toast');
  if (trigger) showToast(trigger, e.detail.successful ? 'ok' : 'error');
});

function showToast(msg, kind) {
  const el = document.createElement('div');
  el.className = 'toast-msg is-' + (kind || 'ok');
  el.textContent = msg;
  document.getElementById('toast').appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

// Theme toggle (persists to localStorage)
function toggleTheme() {
  const cur = document.documentElement.dataset.theme || 'dark';
  const next = cur === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem('tb-theme', next); } catch(e){}
}
(function initTheme(){
  try {
    const saved = localStorage.getItem('tb-theme');
    if (saved) document.documentElement.dataset.theme = saved;
  } catch(e){}
})();
"""


def _shell(content: str, title: str = "ToolBox") -> str:
    """Wrap a content fragment in the full HTML shell."""
    return f"""<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/dist/tbjs/main.css" onerror="this.remove()">
  <style>{CSS}</style>
  <script src="https://unpkg.com/htmx.org@1.9.10" defer></script>
</head>
<body>
  <main class="shell" id="shell">{content}</main>
  <div id="toast" aria-live="polite"></div>
  <script>{JS}</script>
</body>
</html>"""


# =============================================================================
# HTML fragments
# =============================================================================

def _head_fragment(username: Optional[str]) -> str:
    right = (
        f'<span class="label">user/</span> <span style="font-family: var(--font-mono);">{username}</span>'
        f' <button class="btn btn-ghost" hx-post="/local-ui/logout" hx-target="#shell" hx-swap="innerHTML">logout</button>'
        if username else
        '<span class="label">not authenticated</span>'
    )
    return f"""
<header class="app-head">
  <div class="brand">
    <h1>ToolBox</h1>
    <span class="label">local</span>
  </div>
  <nav>
    {right}
    <button class="btn btn-ghost" onclick="toggleTheme()" title="Toggle theme">◐</button>
  </nav>
</header>"""


def _waiting_for_setup_fragment() -> str:
    """Shown briefly during first-run while we auto-open the setup URL elsewhere."""
    return _head_fragment(None) + """
  <div class="card">
    <div class="section-head"><h2>Setting up…</h2><span class="label">first run</span></div>
    <p>Opening the local UI to finish setup. If a new window or tab didn't appear,
       reload this page — your setup link is already issued.</p>
  </div>
"""


def _login_fragment(_unused_first_run: bool = False) -> str:
    """Regular sign-in screen — passkey, magic-link, OAuth.

    The first-run *setup-token* paste flow has been removed: the URL itself
    carries the token (?setup_token=…) and is auto-verified by /. Users
    therefore never need to copy or type a token.
    """
    return _head_fragment(None) + """
  <div class="card">
    <div class="section-head"><h2>Sign in</h2><span class="label">passkey · magic · oauth</span></div>
    <div class="providers">
      <button class="btn btn-primary" hx-post="/local-ui/auth/passkey/start" hx-target="#shell" hx-swap="innerHTML">Passkey</button>
      <button class="btn" onclick="document.getElementById('magic-row').hidden = false; this.blur();">Magic link</button>
      <a class="btn" href="/local-ui/auth/discord">Discord</a>
      <a class="btn" href="/local-ui/auth/google">Google</a>
    </div>
    <div id="magic-row" hidden style="margin-block-start: var(--space-4);">
      <form hx-post="/local-ui/auth/magic/request" hx-target="#toast" hx-swap="beforeend">
        <div class="field">
          <label>email — we'll send the link</label>
          <input type="email" name="email" required>
        </div>
        <button type="submit" class="btn">Send magic link</button>
      </form>
    </div>
  </div>

  <p class="hint">More features: <code>tb feature install &lt;name&gt;</code>. Apps overview at <a href="/mainPagen.html">/mainPagen.html</a>.</p>
"""


def _root_fragment(username: str, mods: List[Dict[str, Any]], services_running: int) -> str:
    mod_preview = ""
    if mods:
        items = "".join(
            (f'<a class="btn btn-ghost" href="{m["ui_path"]}" title="{m["description"]}">{m["icon"]} {m["label"]}</a>'
             if m.get("ui_path") else
             f'<button class="btn btn-ghost" hx-post="/local-ui/api/mods/{m["name"]}/start" hx-target="#toast" hx-swap="beforeend" title="{m["description"]}">{m["icon"]} {m["label"]}</button>')
            for m in mods[:6]
        )
        more = f' <span class="label">+{len(mods) - 6} more</span>' if len(mods) > 6 else ""
        mod_preview = f"""
  <div class="card">
    <div class="section-head"><h2>Your apps</h2><a class="label" href="#mods" hx-get="/local-ui/partials/mods" hx-target="#shell" hx-swap="innerHTML">manage →</a></div>
    <div style="display: flex; flex-wrap: wrap; gap: var(--space-2);">{items}{more}</div>
  </div>"""
    else:
        mod_preview = """
  <div class="card">
    <div class="section-head"><h2>Your apps</h2><span class="label">none yet</span></div>
    <p class="empty">No user-facing mods installed. Add <code>user_facing:</code> to a feature.yaml, or install a feature.</p>
  </div>"""

    return _head_fragment(username) + f"""
  <div class="card">
    <div class="section-head"><h2>Welcome, {username}</h2><span class="label">local root</span></div>
    <p>You are signed in locally. Everything on this page can change your system. The public site is read-only.</p>
    <div class="grid-2" style="margin-block-start: var(--space-4);">
      <a class="btn btn-primary" href="/mainPagen.html">Open all apps</a>
      <button class="btn" hx-get="/local-ui/partials/services" hx-target="#shell" hx-swap="innerHTML">
        Services <span class="label">{services_running} running</span>
      </button>
    </div>
  </div>

  {mod_preview}

  <div hx-get="/local-ui/partials/mounted" hx-trigger="load, every 8s" hx-swap="innerHTML"></div>

  <p class="hint">Add a feature: <code>tb feature install isaa</code> · <code>tb feature install desktop</code> · …</p>
"""


def _services_fragment(username: str, services: List[Dict[str, Any]]) -> str:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for svc in services:
        grouped.setdefault(svc.get("category", "other"), []).append(svc)

    sections = []
    for category in ("core", "infrastructure", "extension", "other"):
        if category not in grouped:
            continue
        rows = []
        for svc in grouped[category]:
            running = svc.get("running", False)
            pid = svc.get("pid")
            auto = svc.get("auto_start", False)
            name = svc["name"]
            desc = svc.get("description", "")
            pid_str = f' <span class="label">pid {pid}</span>' if running and pid else ""
            action_btn = (
                f'<button class="btn btn-danger" hx-post="/local-ui/api/services/{name}/stop" hx-target="#shell" hx-swap="innerHTML">stop</button>'
                if running else
                f'<button class="btn btn-primary" hx-post="/local-ui/api/services/{name}/start" hx-target="#shell" hx-swap="innerHTML">start</button>'
            )
            auto_btn = (
                f'<button class="btn btn-ghost" hx-post="/local-ui/api/services/{name}/auto-toggle" hx-target="#shell" hx-swap="innerHTML">'
                f'{"✓ auto" if auto else "auto off"}</button>'
            )
            rows.append(f"""
    <div class="card-row">
      <div class="meta">
        <span class="name"><span class="dot {"on" if running else "off"}"></span>{name}{pid_str}</span>
        <span class="desc">{desc}</span>
      </div>
      <div style="display: flex; gap: var(--space-2);">{auto_btn} {action_btn}</div>
    </div>""")
        sections.append(f"""
  <div class="card">
    <div class="section-head"><h2>{category.title()}</h2><span class="label">{len(grouped[category])}</span></div>
    {''.join(rows)}
  </div>""")

    if not sections:
        sections = ['<div class="card"><p class="empty">No services registered.</p></div>']

    return _head_fragment(username) + f"""
  <div class="section-head" style="margin-block-end: var(--space-4);">
    <button class="btn btn-ghost" hx-get="/local-ui/partials/root" hx-target="#shell" hx-swap="innerHTML">← back</button>
    <span class="label">services</span>
  </div>
  {''.join(sections)}
  <p class="hint">Service Manager API in <code>tb services --help</code> for the full CLI.</p>
"""


def _mods_fragment(username: str, mods: List[Dict[str, Any]], features_status: Dict[str, bool]) -> str:
    mod_rows = []
    for m in mods:
        open_btn = (
            f'<a class="btn" href="{m["ui_path"]}">open</a>'
            if m.get("ui_path") else ""
        )
        mod_rows.append(f"""
    <div class="card-row">
      <div class="meta">
        <span class="name">{m["icon"]} {m["label"]}</span>
        <span class="desc">{m["description"]} <span class="label">from {m["feature"]} · {m["name"]}.{m["function"]}</span></span>
      </div>
      <div style="display: flex; gap: var(--space-2);">
        {open_btn}
        <button class="btn btn-primary" hx-post="/local-ui/api/mods/{m["name"]}/start" hx-target="#toast" hx-swap="beforeend">start</button>
      </div>
    </div>""")
    mods_section = (
        f'<div class="card"><div class="section-head"><h2>User apps</h2><span class="label">{len(mods)}</span></div>{"".join(mod_rows)}</div>'
        if mods else
        '<div class="card"><div class="section-head"><h2>User apps</h2><span class="label">empty</span></div>'
        '<p class="empty">Add <code>user_facing</code> entries to a <code>feature.yaml</code> to expose mods here.</p></div>'
    )

    feature_rows = []
    for feat, installed in sorted(features_status.items()):
        if installed:
            feature_rows.append(f"""
    <div class="card-row">
      <div class="meta">
        <span class="name"><span class="dot on"></span>{feat}</span>
        <span class="desc">installed</span>
      </div>
      <span class="label">active</span>
    </div>""")
        else:
            feature_rows.append(f"""
    <div class="card-row">
      <div class="meta">
        <span class="name"><span class="dot off"></span>{feat}</span>
        <span class="desc">not installed — run <code>tb feature install {feat}</code></span>
      </div>
      <button class="btn btn-ghost" hx-post="/local-ui/api/features/{feat}/install" hx-target="#toast" hx-swap="beforeend">how?</button>
    </div>""")

    return _head_fragment(username) + f"""
  <div class="section-head" style="margin-block-end: var(--space-4);">
    <button class="btn btn-ghost" hx-get="/local-ui/partials/root" hx-target="#shell" hx-swap="innerHTML">← back</button>
    <span class="label">mods · features</span>
  </div>
  {mods_section}
  <div class="card">
    <div class="section-head"><h2>Features</h2><span class="label">{sum(1 for v in features_status.values() if v)}/{len(features_status)}</span></div>
    {''.join(feature_rows)}
  </div>
  <p class="hint">Anything you change here takes effect on the next <code>tb</code> launch.</p>
"""


# =============================================================================
# Data wiring (Service Manager / Feature Loader)
# =============================================================================

def _list_services() -> List[Dict[str, Any]]:
    try:
        from toolboxv2.utils.clis.service_manager import ServiceManager
        mgr = ServiceManager()
        status = mgr.get_all_status(include_registry=True) or {}
        config = mgr.load_config() or {}
        out = []
        for name, info in status.items():
            out.append({
                "name": name,
                "category": info.get("category", "other"),
                "description": info.get("description", ""),
                "running": bool(info.get("running")),
                "pid": info.get("pid"),
                "auto_start": bool((config.get("services", {}).get(name, {}) or {}).get("auto_start")),
            })
        # stable order
        out.sort(key=lambda s: (s["category"], s["name"]))
        return out
    except Exception as e:
        print(f"[local_ui] service list failed: {e}")
        return []


def _features_status() -> Dict[str, bool]:
    try:
        from toolboxv2.feature_loader import (
            EXTRA_TO_FEATURES,
            is_feature_installed,
        )
        known = set()
        for feats in EXTRA_TO_FEATURES.values():
            known.update(feats)
        return {f: is_feature_installed(f) for f in sorted(known)}
    except Exception as e:
        print(f"[local_ui] feature status failed: {e}")
        return {}


# =============================================================================
# Routes — entry
# =============================================================================

@app.get("/", auth=False)
async def root(
    request: ParsedRequest,
    session: SessionData = None,
    setup_token: str = None,
):
    err = _ensure_local(request)
    if err is not None:
        return err

    # Auto-verify a setup_token in the URL — this is what first-run links carry.
    if setup_token:
        tb_app = get_app("local_ui.auto_verify")
        result = await tb_app.a_run_any(
            ("CloudM.Auth", "verify_magic_link"),
            token=setup_token.strip(), get_results=True,
        )
        payload = await _finalize_session(session, result)
        if payload:
            _FIRST_RUN_TOKEN_CACHE.pop(setup_token, None)
            mods = _all_user_facing_mods()
            running_count = sum(1 for s in _list_services() if s["running"])
            return _shell(
                _root_fragment(payload.get("username") or "you", mods, running_count),
                title=f"{payload.get('username') or 'you'} — ToolBox",
            )
        # Token invalid/expired — drop through to normal login screen

    username = _session_user(session)
    if username:
        mods = _all_user_facing_mods()
        running_count = sum(1 for s in _list_services() if s["running"])
        return _shell(_root_fragment(username, mods, running_count), title=f"{username} — ToolBox")

    # First-run path: no users yet → silently open the setup URL ourselves
    if not await _has_any_user():
        host = (request.headers.get("host") or f"127.0.0.1:{os.getenv('TB_HTTP_PORT', '5000')}")
        await _trigger_first_run(f"http://{host}")
        # Show a quiet waiting card; the auto-open will land them on / with ?setup_token
        return _shell(_waiting_for_setup_fragment(), title="ToolBox — first run")

    return _shell(_login_fragment(False), title="Sign in — ToolBox")


# =============================================================================
# Routes — HTMX partials
# =============================================================================

@app.get("/local-ui/partials/root")
async def partial_root(request: ParsedRequest, session: SessionData = None):
    err = _ensure_local(request)
    if err is not None: return err
    username = _session_user(session)
    if not username:
        return _login_fragment(False)
    running_count = sum(1 for s in _list_services() if s["running"])
    return _root_fragment(username, _all_user_facing_mods(), running_count)


@app.get("/local-ui/partials/services")
async def partial_services(request: ParsedRequest, session: SessionData = None):
    err = _ensure_local(request)
    if err is not None: return err
    username = _session_user(session)
    if not username:
        return _login_fragment(False)
    return _services_fragment(username, _list_services())


@app.get("/local-ui/partials/mods")
async def partial_mods(request: ParsedRequest, session: SessionData = None):
    err = _ensure_local(request)
    if err is not None: return err
    username = _session_user(session)
    if not username:
        return _login_fragment(False)
    return _mods_fragment(username, _all_user_facing_mods(), _features_status())

def _mounted_fragment(mounted: List[Dict[str, str]]) -> str:
    if not mounted:
        return ('<div class="card"><div class="section-head">'
                '<h2>Mounted apps</h2><span class="label">none</span>'
                '</div></div>')
    items = "".join(
        f'<a class="btn btn-ghost" href="{m["prefix"]}/" '
        f'title="source: {m["source"]}">{m["source"]} '
        f'<span class="label">{m["prefix"]}</span></a>'
        for m in mounted
    )
    return (f'<div class="card"><div class="section-head">'
            f'<h2>Mounted apps</h2><span class="label">{len(mounted)} live</span>'
            f'</div><div style="display:flex;flex-wrap:wrap;gap:var(--space-2);">'
            f'{items}</div></div>')


@app.get("/local-ui/partials/mounted")
async def partial_mounted(request: ParsedRequest, session: SessionData = None):
    err = _ensure_local(request)
    if err is not None:
        return err
    return _mounted_fragment(app.list_mounted())
# =============================================================================
# Routes — auth (binds to CloudM.Auth API)
# =============================================================================

async def _finalize_session(session: SessionData, auth_result: Any) -> Optional[Dict[str, Any]]:
    """Inflate session_data from a CloudM.Auth result (JWT-bearing dict).

    Returns the user dict on success, None on failure. The HTTPWorker session
    will pick this up on the next request via its session middleware.
    """
    if hasattr(auth_result, "is_error") and auth_result.is_error():
        return None
    payload = auth_result.get() if hasattr(auth_result, "get") else auth_result
    if not isinstance(payload, dict) or not payload.get("authenticated"):
        return None
    # Project the relevant fields onto the session.
    if session is not None:
        sdata = getattr(session, "data", None)
        if sdata is None:
            try:
                session.data = {}
                sdata = session.data
            except Exception:
                sdata = {}
        sdata.update({
            "user_name": payload.get("username"),
            "user_id": payload.get("user_id"),
            "email": payload.get("email"),
            "level": payload.get("level", 1),
            "provider": payload.get("provider", ""),
            "access_token": payload.get("access_token"),
            "refresh_token": payload.get("refresh_token"),
            "authenticated_at": time.time(),
        })
    return payload


@app.post("/local-ui/auth/magic/request", auth=False)
async def auth_magic_request(request: ParsedRequest, session: SessionData = None):
    err = _ensure_local(request)
    if err is not None: return err
    email = ((request.form_data or {}).get("email") or "").strip()
    if not email:
        return '<div class="toast-msg is-error">Email required.</div>'
    tb_app = get_app("local_ui.magic_request")
    result = await tb_app.a_run_any(
        ("CloudM.Auth", "request_magic_link"),
        email=email, get_results=True,
    )
    if hasattr(result, "is_error") and result.is_error():
        return f'<div class="toast-msg is-error">{(result.get() or {}).get("error", "Failed.")}</div>'
    return f'<div class="toast-msg is-ok">Link sent to {email}. Click it to sign in.</div>'


@app.post("/local-ui/auth/passkey/start", auth=False)
async def auth_passkey_start(request: ParsedRequest, session: SessionData = None):
    """Return a Passkey-login HTML fragment that drives WebAuthn in the browser."""
    err = _ensure_local(request)
    if err is not None: return err
    tb_app = get_app("local_ui.passkey_start")
    result = await tb_app.a_run_any(
        ("CloudM.Auth", "passkey_login_start"),
        get_results=True,
    )
    if hasattr(result, "is_error") and result.is_error():
        return _head_fragment(None) + (
            '<div class="card"><p class="empty">Passkey login is not available — '
            'install via <code>tb feature install web</code> (provides py_webauthn).</p>'
            '<button class="btn" hx-get="/local-ui/partials/root" hx-target="#shell" hx-swap="innerHTML">Back</button></div>'
        )

    options = result.get() if hasattr(result, "get") else result
    options_json = json.dumps(options).replace("</", "<\\/")
    return _head_fragment(None) + f"""
  <div class="card">
    <div class="section-head"><h2>Use your passkey</h2><span class="label">touch / face / pin</span></div>
    <p id="pk-status">Waiting for authenticator…</p>
    <button class="btn" hx-get="/local-ui/partials/root" hx-target="#shell" hx-swap="innerHTML">Cancel</button>
  </div>
  <script>
  (async () => {{
    const status = document.getElementById('pk-status');
    try {{
      const opts = {options_json};
      // base64url → ArrayBuffer
      const b64uToBuf = s => Uint8Array.from(atob(s.replace(/-/g,'+').replace(/_/g,'/').padEnd(s.length+(4-s.length%4)%4,'=')), c=>c.charCodeAt(0)).buffer;
      const bufToB64u = b => btoa(String.fromCharCode(...new Uint8Array(b))).replace(/\\+/g,'-').replace(/\\//g,'_').replace(/=+$/,'');
      opts.challenge = b64uToBuf(opts.challenge);
      (opts.allowCredentials || []).forEach(c => c.id = b64uToBuf(c.id));
      const cred = await navigator.credentials.get({{publicKey: opts}});
      const payload = {{
        id: cred.id,
        rawId: bufToB64u(cred.rawId),
        type: cred.type,
        response: {{
          clientDataJSON: bufToB64u(cred.response.clientDataJSON),
          authenticatorData: bufToB64u(cred.response.authenticatorData),
          signature: bufToB64u(cred.response.signature),
          userHandle: cred.response.userHandle ? bufToB64u(cred.response.userHandle) : null,
        }},
      }};
      const fd = new FormData();
      fd.append('challenge', bufToB64u(opts.challenge));
      fd.append('credential', JSON.stringify(payload));
      status.textContent = 'Verifying…';
      const r = await fetch('/local-ui/auth/passkey/finish', {{method:'POST', body: fd}});
      if (r.ok) {{
        document.getElementById('shell').innerHTML = await r.text();
      }} else {{
        status.textContent = 'Passkey rejected.';
      }}
    }} catch (e) {{
      status.textContent = 'Passkey error: ' + (e.message || e);
    }}
  }})();
  </script>
"""


@app.post("/local-ui/auth/passkey/finish", auth=False)
async def auth_passkey_finish(request: ParsedRequest, session: SessionData = None):
    err = _ensure_local(request)
    if err is not None: return err
    form = request.form_data or {}
    challenge = (form.get("challenge") or "").strip()
    credential_raw = form.get("credential") or "{}"
    try:
        credential = json.loads(credential_raw)
    except Exception:
        return Result.default_user_error(info="Malformed credential.", exec_code=400)

    tb_app = get_app("local_ui.passkey_finish")
    result = await tb_app.a_run_any(
        ("CloudM.Auth", "passkey_login_finish"),
        challenge=challenge, credential=credential, get_results=True,
    )
    payload = await _finalize_session(session, result)
    if not payload:
        return Result.default_user_error(info="Passkey verification failed.", exec_code=401)

    mods = _all_user_facing_mods()
    running_count = sum(1 for s in _list_services() if s["running"])
    return _root_fragment(payload.get("username") or "you", mods, running_count)


@app.get("/local-ui/auth/discord", auth=False)
async def auth_discord_redirect(request: ParsedRequest, session: SessionData = None):
    """Get the Discord OAuth URL and redirect the browser to it."""
    err = _ensure_local(request)
    if err is not None: return err
    tb_app = get_app("local_ui.discord_redirect")
    result = await tb_app.a_run_any(
        ("CloudM.Auth", "get_discord_auth_url"), get_results=True,
    )
    if hasattr(result, "is_error") and result.is_error():
        return result.lazy_return(1).default_internal_error(info="Discord not configured.")
    data = result.get() if hasattr(result, "get") else result
    url = (data or {}).get("auth_url")
    if not url:
        return result.lazy_return(1).default_internal_error(info="No auth URL returned.")
    return Result.redirect(url) if hasattr(Result, "redirect") else f'<meta http-equiv="refresh" content="0;url={url}">'


@app.get("/local-ui/auth/google", auth=False)
async def auth_google_redirect(request: ParsedRequest, session: SessionData = None):
    err = _ensure_local(request)
    if err is not None: return err
    tb_app = get_app("local_ui.google_redirect")
    result = await tb_app.a_run_any(
        ("CloudM.Auth", "get_google_auth_url"),
        redirect_after="/", get_results=True,
    )
    if hasattr(result, "is_error") and result.is_error():
        return result.lazy_return(1).default_internal_error(info="Google not configured.")
    data = result.get() if hasattr(result, "get") else result
    url = (data or {}).get("auth_url")
    if not url:
        return result.lazy_return(1).default_internal_error(info="No auth URL returned.")
    return Result.redirect(url) if hasattr(Result, "redirect") else f'<meta http-equiv="refresh" content="0;url={url}">'


@app.post("/local-ui/logout")
async def logout(request: ParsedRequest, session: SessionData = None):
    err = _ensure_local(request)
    if err is not None: return err
    # Blacklist the access token then clear local session data.
    token = None
    if session is not None:
        sdata = getattr(session, "data", None) or {}
        token = sdata.get("access_token")
    try:
        tb_app = get_app("local_ui.logout")
        if token:
            await tb_app.a_run_any(("CloudM.Auth", "logout"), token=token)
    except Exception:
        pass
    if session is not None:
        try:
            session.data = {}
        except Exception:
            pass
    return _login_fragment(False)


# =============================================================================
# Routes — OAuth callbacks (mirrored here so local-only setups work even
# when no HTTPWorker is mounted on this port; we just call the same
# CloudM.Auth.login_* exports that HTTPWorker would call.)
# =============================================================================

async def _oauth_callback(provider: str, code: str, state: str, session: SessionData):
    """Shared body for /auth/<provider>/callback."""
    fn = "login_discord" if provider == "discord" else "login_google"
    tb_app = get_app(f"local_ui.{provider}_cb")
    result = await tb_app.a_run_any(
        ("CloudM.Auth", fn),
        code=code, state=state, get_results=True,
    )
    payload = await _finalize_session(session, result)
    if not payload:
        return _shell(
            _head_fragment(None) + f"""
  <div class="card">
    <div class="section-head"><h2>{provider.title()} sign-in failed</h2><span class="label">try again</span></div>
    <p>The OAuth callback did not return a valid session. Go back and retry.</p>
    <a class="btn btn-primary" href="/">Back to sign in</a>
  </div>""",
            title=f"{provider} — failed",
        )
    mods = _all_user_facing_mods()
    running_count = sum(1 for s in _list_services() if s["running"])
    return _shell(
        _root_fragment(payload.get("username") or "you", mods, running_count),
        title=f"{payload.get('username') or 'you'} — ToolBox",
    )


@app.get("/auth/discord/callback", auth=False)
async def discord_callback(
    request: ParsedRequest,
    session: SessionData = None,
    code: str = None,
    state: str = None,
):
    err = _ensure_local(request)
    if err is not None: return err
    if not code:
        return Result.default_user_error(info="OAuth code missing.", exec_code=400)
    return await _oauth_callback("discord", code, state, session)


@app.get("/auth/google/callback", auth=False)
async def google_callback(
    request: ParsedRequest,
    session: SessionData = None,
    code: str = None,
    state: str = None,
):
    err = _ensure_local(request)
    if err is not None: return err
    if not code:
        return Result.default_user_error(info="OAuth code missing.", exec_code=400)
    return await _oauth_callback("google", code, state, session)


# =============================================================================
# Routes — services API
# =============================================================================

@app.post("/local-ui/api/services/{name}/start")
async def svc_start(request: ParsedRequest, name: str, session: SessionData = None):
    err = _ensure_local(request)
    if err is not None: return err
    try:
        from toolboxv2.utils.clis.service_manager import ServiceManager
        result = ServiceManager().start_service(name)
        if not result.success:
            return Result.default_internal_error(info=result.error or f"Could not start {name}")
    except Exception as e:
        return Result.default_internal_error(info=f"{e}")
    return _services_fragment(_session_user(session) or "you", _list_services())


@app.post("/local-ui/api/services/{name}/stop")
async def svc_stop(request: ParsedRequest, name: str, session: SessionData = None):
    err = _ensure_local(request)
    if err is not None: return err
    try:
        from toolboxv2.utils.clis.service_manager import ServiceManager
        ServiceManager().stop_service(name)
    except Exception as e:
        return Result.default_internal_error(info=f"{e}")
    return _services_fragment(_session_user(session) or "you", _list_services())


@app.post("/local-ui/api/services/{name}/auto-toggle")
async def svc_auto(request: ParsedRequest, name: str, session: SessionData = None):
    err = _ensure_local(request)
    if err is not None: return err
    try:
        from toolboxv2.utils.clis.service_manager import ServiceManager
        mgr = ServiceManager()
        cfg = mgr.load_config() or {}
        current = bool((cfg.get("services", {}).get(name, {}) or {}).get("auto_start"))
        mgr.configure_service(name, auto_start=not current)
    except Exception as e:
        return Result.default_internal_error(info=f"{e}")
    return _services_fragment(_session_user(session) or "you", _list_services())


# =============================================================================
# Routes — mods + features API
# =============================================================================

@app.post("/local-ui/api/mods/{name}/start")
async def mod_start(request: ParsedRequest, name: str, session: SessionData = None):
    err = _ensure_local(request)
    if err is not None: return err

    # Resolve the function from user_facing (so authors decide the entry point).
    function_name = "start"
    for m in _all_user_facing_mods():
        if m["name"] == name:
            function_name = m["function"]
            break

    try:
        tb_app = get_app("local_ui.mod_start")
        await tb_app.a_run_any((name, function_name))
        return f'<div class="toast-msg is-ok">{name}.{function_name} started.</div>'
    except Exception as e:
        return f'<div class="toast-msg is-error">{name}.{function_name}: {e}</div>'


@app.post("/local-ui/api/features/{name}/install")
async def feature_install_hint(request: ParsedRequest, name: str, session: SessionData = None):
    err = _ensure_local(request)
    if err is not None: return err
    # Per design: show the command, don't invoke install in the request flow.
    return (
        f'<div class="toast-msg is-ok">Run in your terminal:&nbsp;'
        f'<code style="color: var(--text-main);">tb feature install {name}</code></div>'
    )


# =============================================================================
# WIRE-IN
# =============================================================================
#
# 1) Mount this app on HTTPWorker. Wherever HTTPWorker.run() is called
#    (server_worker.py main(), tauri_integration.py _run_servers()), pass
#    fast_tb_app=local_ui_app:
#
#       from toolboxv2.utils.workers.fast.local_ui import app as local_ui_app
#       worker = HTTPWorker(worker_id, config, app=tb_app)
#       worker.run(host=..., port=..., fast_tb_app=local_ui_app, do_run=True)
#
#    The existing /auth/discord/*, /auth/google/*, /auth/magic/verify, etc.
#    routes in HTTPWorker.AUTH_ENDPOINTS continue to work in parallel; this UI
#    delegates to them.
#
# 2) __main__.py — profile routing. consumer/homelab → web UI via Tauri/browser,
#    developer → terminal local_cli, server/business stay as today. In
#    runner_setup(), register the two new entry points:
#
#       runner["local-ui"] = helper_gui            # already exists (Tauri)
#       runner["local-cli"] = lambda: __import__(
#           "toolboxv2.utils.clis.local_cli", fromlist=["main"]).main()
#
#    Then in main_helper() replace the profile branch:
#
#       if profile == "consumer":
#           runner_name = "local-ui"
#       elif profile == "homelab":
#           runner_name = "local-ui"     # web UI for daily use
#       elif profile == "developer":
#           runner_name = "local-cli"    # terminal version of the same screens
#       elif profile == "server":
#           _run_server_overview(); ...
#       elif profile == "business":
#           _run_business_overview(); ...
#       else:
#           runner_name = "default"      # legacy interactive dashboard
#
# 3) Add `user_facing` to any feature.yaml that should appear here. The
#    `function` field tells the UI/CLI which callable on the mod to invoke
#    when the user clicks "start":
#
#       user_facing:
#         - name: CloudM
#           function: open_dashboard    # → app.a_run_any(("CloudM", "open_dashboard"))
#           label: "Account & sync"
#           icon: "☁"
#           description: "Login, OAuth, account management"
#           ui_path: /mainPagen.html?app=CloudM   # optional, default = ?app=<name>
#         - name: DB
#           function: open_browser
#           label: "Storage"
#           icon: "▣"
#           description: "MinIO blob storage"
#
# 4) First-run flow. Index detects no users → a magic-link token is written
#    directly into AUTH_MAGIC_LINK:: in TBEF.DB and printed to STDOUT (no
#    email send). User pastes → existing verify_magic_link() consumes it,
#    auto-creates UserData(email="local-admin@toolbox.local") and returns
#    JWT. After login the UI prompts to register a passkey (covered in the
#    "Sign in" card on subsequent visits).
# =============================================================================


# Standalone runner (for testing this file in isolation)
if __name__ == "__main__":
    from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
    from waitress import serve

    print("\nRoutes:")
    for r in app.list_routes():
        print(f"  {r['method'].ljust(6)} {r['path']}")
    print("\nServing on http://127.0.0.1:5000  (local only)\n")

    handler = FastTBHandler(app)
    serve(handler.as_wsgi_app(enable_ws=False), host="127.0.0.1", port=5000)
