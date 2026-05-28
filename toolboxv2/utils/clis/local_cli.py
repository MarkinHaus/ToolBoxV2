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


# =============================================================================
# Shared helpers (mirror local_ui semantics)
# =============================================================================

LOCAL_ADMIN_EMAIL = "local-admin@toolbox.local"


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


async def _has_any_user() -> bool:
    try:
        tb_app = get_app("local_cli.has_user")
        result = tb_app.run_any(("CloudM.Auth", "list_users"), get_results=True)
        if hasattr(result, "is_error") and result.is_error():
            return False
        data = result.get() if hasattr(result, "get") else result
        return bool(data) and len(data) > 0
    except Exception:
        return False

_global_token = [None]
async def _emit_first_run_token() -> Optional[str]:
    if _global_token[0] is not None:
        return _global_token[0]
    try:
        from toolboxv2.mods.CloudM.auth.db_helpers import _db_set, _db_get
    except Exception as e:
        print_status(f"Cannot import auth db_helpers: {e}", "error")
        return None
    tb_app = get_app("local_cli.first_run_token")
    token = secrets.token_urlsafe(32)
    r =await _db_set(tb_app, f"AUTH_MAGIC_LINK::{token}", {
        "email": LOCAL_ADMIN_EMAIL,
        "created_at": time.time(),
        "verified": False,
        "local_admin": True,
    })
    c_print(r.print(show=False))
    _global_token[0] = token
    return token


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


async def login_screen(sess: CLISession) -> bool:
    """Drive the user through one of the available auth flows. Returns True on success."""


    first_run = not await _has_any_user()
    print_box_header("ToolBox · Sign in" + (" · first run" if first_run else ""), "🔐")

    if first_run:
        token = await _emit_first_run_token()
        if token:
            print_box_content("First run — your local setup token (single-use, 10 min):", "info")
            print()
            c_print(f"    {Colors.BOLD}{token}{Colors.RESET}")
            print()
            print_box_content("Paste it below to create the local admin account.", "info")

        print_separator()
    else:
        c_print("No local setup token found!", "")
        c_print("run tb login", "")

    c_print(f"  {Colors.BOLD}Choose a method:{Colors.RESET}")
    options = [
        ("token",   "Setup / magic-link token"),
        ("magic",   "Request magic link by email"),
        ("discord", "Discord OAuth (opens browser)"),
        ("google",  "Google OAuth (opens browser)"),
        ("quit",    "Quit"),
    ]
    for i, (_, label) in enumerate(options, 1):
        c_print(f"  {Colors.CYAN}{i}){Colors.RESET} {label}")
    print()
    raw = _prompt(f"Choose [1-{len(options)}]")
    try:
        idx = int(raw.strip()) - 1
        choice = options[idx][0]
    except (ValueError, IndexError):
        print_status("Invalid choice", "error")
        return False

    if choice == "quit":
        raise SystemExit(0)

    tb_app = get_app("local_cli.auth")

    if choice == "token":
        token = _global_token[0] or _prompt("Token").strip()
        if not token:
            print_status("Token required", "error")
            return False
        result = await tb_app.a_run_any(
            ("CloudM.Auth", "verify_magic_link"), token=token, get_results=True,
        )
        if hasattr(result, "is_error") and result.is_error():
            print(result)
            _prompt("Email").strip()
            print_status(f"Verification failed: {(result.get() or {}).get('error', 'invalid')}", "error")
            return False
        payload = result.get() if hasattr(result, "get") else result
        if not sess.adopt(payload):
            print_status("Authentication failed", "error")
            return False
        print_status(f"Signed in as {sess.username}", "success")
        return True

    if choice == "magic":
        email = _prompt("Email").strip()
        if not email or "@" not in email:
            print_status("Valid email required", "error")
            return False
        result = await tb_app.a_run_any(
            ("CloudM.Auth", "request_magic_link"), email=email, get_results=True,
        )
        if hasattr(result, "is_error") and result.is_error():
            print_status(f"Could not send: {(result.get() or {}).get('error', 'unknown')}", "error")
            return False
        print_status("Magic link sent. Open the link in a browser, or paste the token here.", "info")
        token = _prompt("Token from email (or skip with empty)")
        if not token.strip():
            return False
        result = await tb_app.a_run_any(
            ("CloudM.Auth", "verify_magic_link"), token=token.strip(), get_results=True,
        )
        payload = result.get() if hasattr(result, "get") else result
        if not sess.adopt(payload):
            print_status("Authentication failed", "error")
            return False
        print_status(f"Signed in as {sess.username}", "success")
        return True

    if choice in ("discord", "google"):

        print_status("workers must be running | start with | tb workers start", "")

        fn = "get_discord_auth_url" if choice == "discord" else "get_google_auth_url"
        result = await tb_app.a_run_any(
            ("CloudM.Auth", fn), redirect_after="/", get_results=True,
        )
        if hasattr(result, "is_error") and result.is_error():
            print_status(f"{choice.title()} not configured", "error")
            return False
        url = (result.get() if hasattr(result, "get") else result or {}).get("auth_url")
        if not url:
            print_status("No URL returned", "error")
            return False
        print_status(f"Open this URL in your browser:", "info")
        c_print(f"    {Colors.CYAN}{url}{Colors.RESET}")
        print_status("After confirming, the OAuth callback will create the session in HTTP. "
                     "For the terminal session, paste the JWT access token printed by the callback page.",
                     "info")
        token = _prompt("Access token")
        if not token.strip():
            return False
        # Treat the access_token as already-valid JWT — wrap into payload.
        sess.access_token = token.strip()
        sess.username = "oauth-user"
        sess.provider = choice
        print_status(f"Signed in via {choice}", "success")
        return True

    return False


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
        c_print(f"  {Colors.DIM}Enter a number to act on a service, or 'b' to go back.{Colors.RESET}")
        choice = _prompt("Service #").strip().lower()
        if choice in ("b", "back", "q"):
            return
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
    while True:
        os.system("cls" if sys.platform == "win32" else "clear")
        print_box_header(f"ToolBox · {sess.username}", "🏠")
        print_box_content(f"local root · provider: {sess.provider or 'magic_link'}", "info")
        print_box_footer()
        c_print(f"  {Colors.CYAN}1){Colors.RESET} Apps")
        c_print(f"  {Colors.CYAN}2){Colors.RESET} Services")
        c_print(f"  {Colors.CYAN}3){Colors.RESET} Open web UI (browser)")
        c_print(f"  {Colors.CYAN}4){Colors.RESET} Logout")
        c_print(f"  {Colors.CYAN}5){Colors.RESET} Quit")
        c_print(f"  {Colors.DIM}Tip: full overview at /mainPagen.html{Colors.RESET}")
        print()

        choice = _prompt("Choose [1-5]").strip()
        if choice == "1":
            await mods_screen(sess)
        elif choice == "2":
            await services_screen(sess)
        elif choice == "3":
            import webbrowser
            port = os.getenv("TB_HTTP_PORT", "8080")
            webbrowser.open(f"http://127.0.0.1:{port}/")
            print_status("Opened in browser", "success")
            await asyncio.sleep(0.6)
        elif choice == "4":
            tb_app = get_app("local_cli.logout")
            if sess.access_token:
                try:
                    await tb_app.a_run_any(("CloudM.Auth", "logout"), token=sess.access_token)
                except Exception:
                    pass
            sess.clear()
            print_status("Signed out", "success")
            return
        elif choice in ("5", "q", "quit", "exit"):
            raise SystemExit(0)


# =============================================================================
# Main
# =============================================================================

async def main():
    sess = CLISession()
    while True:
        if not sess.is_auth():
            app = get_app("local_cli.login")
            os.system("cls" if sys.platform == "win32" else "clear")
            status = False
            ok = False
            if not app.session.valid:
                status = await app.session.login()
            if status:
                sess.username = app.session.username
                sess.access_token = app.session.access_token
                sess.refresh_token = app.session.refresh_token
                sess.user_id = app.session.user_id
                sess.provider = "app.session"
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

