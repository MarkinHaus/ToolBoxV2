#!/usr/bin/env python3
"""
toolboxv2/utils/clis/local_cli.py - Terminal pendant of local_ui

Same three screens (Login / Services / Mods), terminal UX. Selected when the
profile is "developer" or when the user opts out of the GUI. Talks to the
same CloudM.Auth API as local_ui — never a second auth surface.

Run:
    python -m toolboxv2.utils.clis.local_cli
or via tb dispatcher: tb local-cli
"""

import asyncio
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from toolboxv2 import get_app, tb_root_dir

from .cli_printing import (
    Colors, c_print, print_box_header, print_box_content,
    print_box_footer, print_status, print_separator,
)
from .cli_oauth import cli_remote_login
from .cli_input import menu_select_async
from toolboxv2.mods.CloudM.LogInSystem import (
    _save_cli_token, _clear_cli_token, _check_existing_session,
)
# =============================================================================
# Shared helpers (mirror local_ui semantics)
# =============================================================================

from toolboxv2.mods.CloudM.auth.config import LOCAL_ADMIN_EMAIL


def _features_dir() -> Path:
    return tb_root_dir / "features"


def _user_facing(feature_yaml: Path) -> List[Dict[str, Any]]:
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
            "feature": data.get("name", "unknown"),
        })
    return out


def _all_user_facing_mods() -> List[Dict[str, Any]]:
    features_dir = _features_dir()
    if not features_dir.exists():
        return []
    out = []
    for fyaml in features_dir.glob("*/feature.yaml"):
        if not (fyaml.parent / ".installed").exists():
            continue
        out.extend(_user_facing(fyaml))
    return out



def _is_remote() -> bool:
    return bool(os.environ.get("TOOLBOXV2_REMOTE_BASE"))


def _err_text(result) -> str:
    info = getattr(result, "info", None)
    return getattr(info, "help_text", "invalid") if info else "invalid"

async def _remote_auth(tb_app, choice: str) -> Optional[dict]:
    """Auth via the aiohttp Session (session.py) against TOOLBOXV2_REMOTE_BASE."""
    s = tb_app.session
    if choice in ("token", "magic"):
        if choice == "magic":
            email = _prompt("Email").strip()
            if "@" not in email:
                print_status("Valid email required", "error"); return None
            req = await s.login_with_magic_link(email)
            if req.is_error():
                print_status(f"Could not send: {_err_text(req)}", "error"); return None
            print_status("Magic link sent — paste the token from the email.", "info")
        token = _prompt("Token").strip()
        if not token:
            return None
        res = await s.verify_magic_link(token)
    elif choice == "invite":
        code = _prompt("Device invite code").strip().replace(" ", "").replace("-", "")
        if not code:
            return None
        res = await s.login_with_invite_code(code)
    else:
        return None

    if res.is_error():
        print_status(f"Auth failed: {_err_text(res)}", "error"); return None
    data = res.get() or {}
    # session.py already set s.username/.access_token/.valid and persisted (see #6).
    return {"authenticated": True, **data}

# =============================================================================
# State (one CLI session = one in-memory session)
# =============================================================================

class CLISession:
    """In-memory session holding the JWT after login."""

    def __init__(self):
        self.username: Optional[str] = None
        self.user_id: Optional[str] = None
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.provider: str = ""

    def is_auth(self) -> bool:
        return bool(self.username and self.access_token)

    def adopt(self, payload: dict) -> bool:
        if not isinstance(payload, dict) or not payload.get("authenticated"):
            return False
        self.username = payload.get("username")
        self.user_id = payload.get("user_id")
        self.access_token = payload.get("access_token")
        self.refresh_token = payload.get("refresh_token")
        self.provider = payload.get("provider", "")
        return True

    def clear(self):
        self.username = None
        self.user_id = None
        self.access_token = None
        self.refresh_token = None
        self.provider = ""


# =============================================================================
# Auth screen
# =============================================================================

def _prompt(label: str, mask: bool = False) -> str:
    try:
        if mask:
            import getpass
            return getpass.getpass(f"  {Colors.CYAN}❯{Colors.RESET} {label}: ")
        return input(f"  {Colors.CYAN}❯{Colors.RESET} {label}: ")
    except (KeyboardInterrupt, EOFError):
        raise SystemExit(0)

async def _finalize(sess: CLISession, payload, tb_app, remote: bool) -> bool:
    if not payload or not sess.adopt(payload):
        print_status("Authentication failed", "error")
        return False
    if not remote:  # OAuth/Passkey authenticate against the local instance → persist local
        try:
            await _save_cli_token(tb_app, sess.username, {
                "username": sess.username, "email": payload.get("email", ""),
                "user_id": sess.user_id, "level": payload.get("level", 1),
                "access_token": sess.access_token, "refresh_token": sess.refresh_token,
                "provider": sess.provider, "authenticated_at": time.time(),
            })
        except Exception as e:
            print_status(f"Session not persisted: {e}", "warning")
    print_status(f"Signed in as {sess.username}", "success")
    await asyncio.sleep(0.6)
    return True

async def login_screen(sess: CLISession) -> bool:
    tb_app = get_app("local_cli.auth")
    remote = _is_remote()

    if not remote:
        # Zero-friction local mode: auto-login as the local root admin.
        # No token shown or required locally — tokens are remote-only (E3).
        from toolboxv2.mods.CloudM.auth.local_admin import ensure_local_admin
        from toolboxv2.mods.CloudM.auth.jwt_tokens import _generate_tokens
        user = await ensure_local_admin(tb_app)
        tokens = _generate_tokens(user, "local_admin")
        payload = {
            "authenticated": True,
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "level": user.level,
            "provider": "local_admin",
            **tokens,
        }
        return await _finalize(sess, payload, tb_app, remote=False)

    print_box_header("ToolBox · Sign in · remote", "🔐")

    options = [
        ("token",   "Magic token (paste)"),
        ("magic",   "Magic link by email"),
        ("invite",  "Device invite code"),
        ("discord", "Discord OAuth (browser)"),
        ("google",  "Google OAuth (browser)"),
        ("passkey", "Passkey (browser)"),
        ("quit",    "Quit"),
    ]
    choice = await menu_select_async(
        options, title="Choose a method:",
        hint="\u2191/\u2193 or W/S to move \u00b7 Enter to select \u00b7 q/Esc to quit",
    )
    if choice in (None, "quit"):
        raise SystemExit(0)

    if choice in ("discord", "google", "passkey"):
        # All three run remotely on simplecore (its keys / rp_id). The remote
        # login page bridges the issued tokens back to a local loopback; the
        # resulting account is a remote account validated via app.session.
        print_status(f"Opening browser for {choice} on simplecore\u2026 complete the login, then return here.", "info")
        payload = await cli_remote_login(choice)
        if payload:
            s = tb_app.session
            s.username = payload.get("username") or s.username
            s._save_session_token(
                payload.get("access_token", ""),
                payload.get("refresh_token", ""),
                payload.get("user_id", ""),
            )
            if not await s.login():
                print_status("Remote session validation failed", "error")
                return False
            payload["username"] = s.username or payload.get("username", "")
        return await _finalize(sess, payload, tb_app, remote=True)

    payload = await _remote_auth(tb_app, choice)
    return await _finalize(sess, payload, tb_app, remote)

# =============================================================================
# Services screen
# =============================================================================

def _list_services() -> List[Dict[str, Any]]:
    try:
        from .service_manager import ServiceManager
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
        out.sort(key=lambda s: (s["category"], s["name"]))
        return out
    except Exception as e:
        print_status(f"Service list failed: {e}", "error")
        return []

async def _add_custom_service():
    """Register a named custom service where the user defines the tb command.

    Each custom service has its own name (own PID + own config entry), so
    multiple can coexist. The command is whatever the user types after 'tb'
    (python -m toolboxv2 <args>). Persisted via configure_service(custom=True).
    """
    name = _prompt("Service name (unique)").strip()
    if not name:
        print_status("Name required", "error"); return
    if name in ("custom", "workers", "db", "gui", "api"):
        print_status(f"'{name}' is reserved — pick another name", "error"); return

    from .service_manager import ServiceManager
    mgr = ServiceManager()
    if name in (mgr.load_config().get("services", {}) or {}):
        print_status(f"Service '{name}' already exists", "error"); return

    c_print(f"  {Colors.DIM}You define the tb command. Enter the part after 'tb' — "
            f"e.g. 'run my-flow' or 'api start --port 9000'.{Colors.RESET}")
    raw = _prompt("tb").strip()
    if not raw:
        print_status("Command required", "error"); return
    args = raw.split()
    auto_raw = _prompt("Auto-start on boot? [y/N]").strip().lower()
    auto_start = auto_raw in ("y", "yes", "j", "ja")

    try:
        mgr.configure_service(
            name, args=args, auto_start=auto_start,
            custom=True, description=f"tb {raw}",
        )
        print_status(f"Custom service '{name}' added: tb {raw}", "success")
        result = mgr.start_service(name, args=args)
        if result.success:
            print_status(f"Started (pid {result.pid})", "success")
        else:
            print_status(f"Not started: {result.error}", "error")
    except Exception as e:
        print_status(f"Could not add service: {e}", "error")
    await asyncio.sleep(0.8)


async def services_screen(sess: CLISession):
    """Interactive service management."""
    while True:
        os.system("cls" if sys.platform == "win32" else "clear")
        print_box_header(f"Services · signed in as {sess.username}", "🔧")
        svcs = _list_services()
        if not svcs:
            print_box_content("No services registered.", "warning")
            print_box_footer()
            return

        # Group by category
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for svc in svcs:
            grouped.setdefault(svc["category"], []).append(svc)

        # Numbered list across groups
        flat: List[Dict[str, Any]] = []
        idx = 0
        for cat in ("core", "infrastructure", "extension", "other"):
            if cat not in grouped:
                continue
            print_separator()
            c_print(f"  {Colors.BOLD}{cat.upper()}{Colors.RESET}")
            for svc in grouped[cat]:
                idx += 1
                flat.append(svc)
                dot = f"{Colors.GREEN}●{Colors.RESET}" if svc["running"] else f"{Colors.DIM}○{Colors.RESET}"
                pid = f" pid {svc['pid']}" if svc["running"] and svc.get("pid") else ""
                auto = f" {Colors.CYAN}[auto]{Colors.RESET}" if svc["auto_start"] else ""
                c_print(f"  {Colors.CYAN}{idx:>2}){Colors.RESET} {dot} {svc['name']:<18}{Colors.DIM}{pid}{Colors.RESET}{auto}")
                if svc["description"]:
                    c_print(f"       {Colors.DIM}{svc['description']}{Colors.RESET}")

        print_box_footer()
        c_print(
            f"  {Colors.DIM}Enter a number to act on a service · 'a' add custom tb command · 'b' back.{Colors.RESET}")
        choice = _prompt("Service #").strip().lower()
        if choice in ("b", "back", "q"):
            return
        if choice in ("a", "add", "+"):
            await _add_custom_service()
            continue
        try:
            sel = flat[int(choice) - 1]
        except (ValueError, IndexError):
            continue

        # Per-service actions
        c_print(f"  Selected: {Colors.BOLD}{sel['name']}{Colors.RESET}")
        c_print("  1) start" if not sel["running"] else "  1) stop")
        c_print(f"  2) toggle auto-start ({'on' if sel['auto_start'] else 'off'})")
        c_print("  3) cancel")
        sub = _prompt("Action").strip()

        from .service_manager import ServiceManager
        mgr = ServiceManager()
        try:
            if sub == "1" and not sel["running"]:
                result = mgr.start_service(sel["name"])
                if result.success:
                    print_status(f"{sel['name']} started (pid {result.pid})", "success")
                else:
                    print_status(f"Failed: {result.error}", "error")
            elif sub == "1" and sel["running"]:
                mgr.stop_service(sel["name"])
                print_status(f"{sel['name']} stopped", "success")
            elif sub == "2":
                mgr.configure_service(sel["name"], auto_start=not sel["auto_start"])
                print_status(f"Auto-start toggled", "success")
        except Exception as e:
            print_status(f"Error: {e}", "error")
        await asyncio.sleep(0.6)


# =============================================================================
# Mods screen
# =============================================================================

def _features_status() -> Dict[str, bool]:
    try:
        from toolboxv2.feature_loader import EXTRA_TO_FEATURES, is_feature_installed
        known = set()
        for feats in EXTRA_TO_FEATURES.values():
            known.update(feats)
        return {f: is_feature_installed(f) for f in sorted(known)}
    except Exception:
        return {}


async def mods_screen(sess: CLISession):
    while True:
        os.system("cls" if sys.platform == "win32" else "clear")
        print_box_header(f"Apps & Features · {sess.username}", "▣")

        mods = _all_user_facing_mods()
        if mods:
            print_separator()
            c_print(f"  {Colors.BOLD}USER APPS{Colors.RESET}")
            for i, m in enumerate(mods, 1):
                c_print(f"  {Colors.CYAN}{i:>2}){Colors.RESET} {m['icon']} {Colors.BOLD}{m['label']}{Colors.RESET}  "
                        f"{Colors.DIM}{m['name']}.{m['function']} · from {m['feature']}{Colors.RESET}")
                if m["description"]:
                    c_print(f"       {Colors.DIM}{m['description']}{Colors.RESET}")
        else:
            print_box_content("No user_facing mods configured in any feature.yaml.", "info")

        feats = _features_status()
        if feats:
            print_separator()
            c_print(f"  {Colors.BOLD}FEATURES{Colors.RESET}")
            for feat, installed in feats.items():
                if installed:
                    c_print(f"     {Colors.GREEN}●{Colors.RESET} {feat:<12} {Colors.DIM}installed{Colors.RESET}")
                else:
                    c_print(f"     {Colors.DIM}○ {feat:<12} → tb feature install {feat}{Colors.RESET}")

        print_box_footer()
        c_print(f"  {Colors.DIM}Enter a mod # to start it, or 'b' to go back.{Colors.RESET}")
        choice = _prompt("Mod #").strip().lower()
        if choice in ("b", "back", "q"):
            return
        try:
            sel = mods[int(choice) - 1]
        except (ValueError, IndexError):
            continue

        tb_app = get_app("local_cli.mod_start")
        try:
            await tb_app.a_run_any((sel["name"], sel["function"]))
            print_status(f"{sel['name']}.{sel['function']} executed", "success")
        except Exception as e:
            print_status(f"{sel['name']}.{sel['function']} failed: {e}", "error")
        await asyncio.sleep(1.0)


# =============================================================================
# Root menu (post-auth)
# =============================================================================

async def root_menu(sess: CLISession):
    tb_app = get_app("local_cli.menu")
    while True:
        os.system("cls" if sys.platform == "win32" else "clear")
        print_box_header(f"ToolBox · {sess.username}", "🏠")
        print_box_content(f"{'remote' if _is_remote() else 'local'} · provider: {sess.provider or 'magic_link'}", "info")
        print_box_footer()
        choice = await menu_select_async([
            ("apps", "Apps"),
            ("services", "Services  (start/stop · auto-start · add custom tb command)"),
            ("web", "Open web UI (browser)"),
            ("passkey_reg", "Register passkey (browser)"),
            ("logout", "Logout"),
            ("quit", "Quit"),
        ], hint="\u2191/\u2193 or W/S \u00b7 Enter \u00b7 q to quit")

        if choice == "apps":
            await mods_screen(sess)
        elif choice == "services":
            await services_screen(sess)
        elif choice == "passkey_reg":
            # Passkey registration happens on simplecore (remote rp_id). Open the
            # remote login page; the user registers a passkey there while signed in.
            import webbrowser
            base = (tb_app.session.base or "").rstrip("/")
            if not base:
                print_status("No remote base configured (TOOLBOXV2_REMOTE_BASE)", "error")
            else:
                webbrowser.open(f"{base}/web/assets/login.html")
                print_status("Opened simplecore in browser — register your passkey there", "success")
            await asyncio.sleep(0.8)
        elif choice == "web":
            import webbrowser
            # TODO encur local ui is running and get the current port- rework local ui
            port = os.getenv("TB_HTTP_PORT", "8080")
            webbrowser.open(f"http://127.0.0.1:{port}/")
            print_status("Opened in browser", "success")
            await asyncio.sleep(0.6)
        elif choice == "logout":
            if sess.access_token:
                try:
                    await tb_app.a_run_any(("CloudM.Auth", "logout"), token=sess.access_token)
                except Exception:
                    pass
            try:
                if _is_remote():
                    await tb_app.session.logout()
                else:
                    await _clear_cli_token(tb_app, sess.username)
            except Exception:
                pass
            sess.clear()
            print_status("Signed out", "success")
            return
        elif choice in (None, "quit"):
            raise SystemExit(0)


# =============================================================================
# Main
# =============================================================================
async def login():
    for i in range(3):
        sess = CLISession()
        ok = await login_screen(sess)
        if not ok:
            _prompt("Login failed (press enter)", False)
            os.system("cls" if sys.platform == "win32" else "clear")
            continue
        else:
            break

async def main():
    sess = CLISession()
    while True:
        if not sess.is_auth():
            app = get_app("local_cli.login")
            os.system("cls" if sys.platform == "win32" else "clear")

            # Local profile (User A): sign in straight as the local root admin.
            # No token, no menu — ensure_local_admin already ran at CloudM start
            # and mirrored the user into app.session (see CloudM.module).
            if not _is_remote():
                from toolboxv2.mods.CloudM.auth.local_admin import ensure_local_admin
                from toolboxv2.mods.CloudM.auth.jwt_tokens import _generate_tokens
                user = await ensure_local_admin(app)
                tokens = _generate_tokens(user, "local_admin")
                sess.username = user.username
                sess.user_id = user.user_id
                sess.access_token = tokens["access_token"]
                sess.refresh_token = tokens["refresh_token"]
                sess.provider = "local_admin"
            else:
                ok = await login_screen(sess)
                if not ok:
                    _prompt("Login failed (press enter)", False)
                    os.system("cls" if sys.platform == "win32" else "clear")
                    continue
        await root_menu(sess)

_async_main = main

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        c_print(f"\n  {Colors.DIM}bye{Colors.RESET}\n")

