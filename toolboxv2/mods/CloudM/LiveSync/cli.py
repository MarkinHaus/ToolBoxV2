"""
LiveSync — Interactive Onboarding CLI
=====================================
Async TUI: create / join / manage shares + start / stop / status.

Run:
    python -m toolboxv2.mods.CloudM.LiveSync.cli

Navigation:
    ↑↓ / W S       move
    Enter          select
    Q / Esc        back / quit
"""
from __future__ import annotations

import asyncio
import os
import sys
from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple

from . import (
    create_share,
    get_sync_log,
    get_sync_status,
    join_share,
    list_shares,
    restart_sync,
    start_sync,
    stop_share,
    stop_sync,
)


# ── ANSI ─────────────────────────────────────────────────────────────

CSI = "\033["
CLR = f"{CSI}2J{CSI}H"
HIDE = f"{CSI}?25l"
SHOW = f"{CSI}?25h"
B = f"{CSI}1m"
D = f"{CSI}2m"
INV = f"{CSI}7m"
RST = f"{CSI}0m"
GRN = f"{CSI}32m"
RED = f"{CSI}31m"
YEL = f"{CSI}33m"
CYN = f"{CSI}36m"


def w(s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.flush()


# ── Terminal mode ────────────────────────────────────────────────────

_IS_WIN = sys.platform == "win32"


@contextmanager
def raw_mode():
    """ICANON+ECHO off on Unix; no-op on Windows (msvcrt reads raw natively)."""
    if _IS_WIN:
        yield
        return
    import termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] = new[3] & ~(termios.ICANON | termios.ECHO)
    new[6][termios.VMIN] = 1
    new[6][termios.VTIME] = 0
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, new)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key_win() -> str:
    import msvcrt
    ch = msvcrt.getch()
    if ch in (b"\x00", b"\xe0"):
        ch2 = msvcrt.getch()
        return {b"H": "up", b"P": "down", b"K": "left", b"M": "right"}.get(ch2, "")
    if ch == b"\r":
        return "enter"
    if ch == b"\x1b":
        return "esc"
    if ch == b"\x03":
        raise KeyboardInterrupt
    try:
        return ch.decode("utf-8", "ignore").lower()
    except Exception:
        return ""


def _read_key_unix() -> str:
    import select
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        r, _, _ = select.select([sys.stdin], [], [], 0.05)
        if not r:
            return "esc"
        if sys.stdin.read(1) != "[":
            return "esc"
        ch3 = sys.stdin.read(1)
        return {"A": "up", "B": "down", "D": "left", "C": "right"}.get(ch3, "")
    if ch in ("\r", "\n"):
        return "enter"
    if ch == "\x03":
        raise KeyboardInterrupt
    return ch.lower()


async def read_key() -> str:
    return await asyncio.to_thread(_read_key_win if _IS_WIN else _read_key_unix)


# ── Drawing ──────────────────────────────────────────────────────────

def _header(status: dict) -> None:
    running = status.get("running")
    dot = f"{GRN}●{RST}" if running else f"{RED}○{RST}"
    pid = status.get("pid")
    pid_s = f"pid {pid}" if pid else "stopped"
    shares = len(status.get("shares", []))
    w(f"{B}╔══ LiveSync ══════════════════════════════╗{RST}\n")
    w(f"{B}║{RST} {dot} {pid_s:<22}  shares: {shares:<3} {B+'   '}║{RST}\n")
    w(f"{B}╚══════════════════════════════════════════╝{RST}\n\n")


def _render_menu(title: str, items: List[Tuple[str, str]], idx: int) -> None:
    w(f"{CYN}{title}{RST}\n\n")
    for i, (_, label) in enumerate(items):
        if i == idx:
            w(f"  {INV} ▶ {label} {RST}\n")
        else:
            w(f"      {label}\n")
    w(f"\n{D}↑↓/WS move · Enter select · Q back{RST}\n")


# ── Async primitives ────────────────────────────────────────────────

async def select_menu(
    title: str,
    items: List[Tuple[str, str]],
    status_provider: Optional[Callable[[], dict]] = None,
) -> Optional[str]:
    """Arrow/WASD menu. Returns chosen key, or None on Q/Esc."""
    idx = 0
    w(HIDE)
    try:
        with raw_mode():
            while True:
                w(CLR)
                if status_provider:
                    try:
                        _header(status_provider())
                    except Exception as e:
                        w(f"{RED}status err: {e}{RST}\n\n")
                _render_menu(title, items, idx)
                k = await read_key()
                if k in ("up", "w"):
                    idx = (idx - 1) % len(items)
                elif k in ("down", "s"):
                    idx = (idx + 1) % len(items)
                elif k == "enter":
                    return items[idx][0]
                elif k in ("q", "esc"):
                    return None
    finally:
        w(SHOW)


async def confirm(msg: str) -> bool:
    w(f"\n{YEL}{msg}{RST} [y/N] ")
    with raw_mode():
        k = await read_key()
    w(k + "\n")
    return k == "y"


async def prompt(label: str, default: str = "") -> str:
    """Cooked-mode line prompt (terminal echo + line editing)."""
    suffix = f" [{default}]" if default else ""
    w(f"{CYN}{label}{RST}{suffix}: ")
    val = await asyncio.to_thread(sys.stdin.readline)
    val = (val or "").rstrip("\r\n")
    return val or default


async def pause(msg: str = "press any key…") -> None:
    w(f"\n{D}{msg}{RST}")
    with raw_mode():
        await read_key()
    w("\n")


# ── Actions ──────────────────────────────────────────────────────────

async def action_create() -> None:
    w(CLR)
    w(f"{B}── Create new share ──{RST}\n\n")
    vault = await prompt("Vault path", os.path.expanduser("~/vault"))
    if not vault:
        return
    if not os.path.isdir(vault):
        if await confirm(f"'{vault}' does not exist. Create?"):
            os.makedirs(vault, exist_ok=True)
        else:
            return
    host = await prompt("WS host (token endpoint)", "127.0.0.1")
    port_s = await prompt("WS port", "8765")
    try:
        port = int(port_s)
    except ValueError:
        w(f"{RED}invalid port{RST}\n")
        await pause()
        return

    w(f"\n{D}creating share…{RST}\n")
    res = await asyncio.to_thread(create_share, vault, host, port)
    if not res.get("ok"):
        w(f"{RED}error: {res.get('error')}{RST}\n")
        await pause()
        return

    w(f"\n{GRN}✓ Share created{RST}\n")
    w(f"  share_id: {B}{res['share_id']}{RST}\n")
    w(f"\n{CYN}Token (distribute to peers):{RST}\n\n")
    w(f"{res['token']}\n\n")
    w(f"{D}copy this token — peers use it to join{RST}\n")
    await pause()


async def action_join() -> None:
    w(CLR)
    w(f"{B}── Join existing share ──{RST}\n\n")
    vault = await prompt("Vault path", os.path.expanduser("~/vault"))
    if not vault:
        return
    os.makedirs(vault, exist_ok=True)
    token = await prompt("Share token")
    if not token:
        return
    w(f"\n{D}joining…{RST}\n")
    res = await asyncio.to_thread(join_share, vault, token)
    if res.get("ok"):
        w(f"\n{GRN}✓ Joined {res['share_id']}{RST}\n")
        cfg = res.get("config", {})
        w(f"  ws:    {cfg.get('ws_endpoint')}\n")
        w(f"  vault: {cfg.get('vault_path')}\n")
    else:
        w(f"\n{RED}error: {res.get('error')}{RST}\n")
    await pause()


async def action_share_detail(share_id: str) -> None:
    while True:
        choice = await select_menu(
            f"Share: {share_id}",
            [
                ("log", "view sync log"),
                ("stop", "stop & deregister"),
                ("back", "back"),
            ],
        )
        if choice in (None, "back"):
            return
        if choice == "log":
            res = await asyncio.to_thread(get_sync_log, share_id, "", 30)
            w(CLR)
            w(f"{B}── Sync log: {share_id} ──{RST}\n\n")
            if not res.get("ok"):
                w(f"{RED}{res.get('error')}{RST}\n")
            else:
                logs = res.get("data") or []
                if not logs:
                    w(f"{D}(empty){RST}\n")
                for e in logs:
                    ts = e.get("timestamp", 0)
                    w(f"  {D}{int(ts)}{RST}  "
                      f"{e.get('action', ''):<10}  {e.get('rel_path', '')}\n")
            await pause()
        elif choice == "stop":
            if await confirm(f"Stop share {share_id}?"):
                res = await asyncio.to_thread(stop_share, share_id)
                col = GRN if res.get("ok") else RED
                w(f"{col}{res}{RST}\n")
                await pause()
                return


async def action_manage_shares() -> None:
    while True:
        shares = await asyncio.to_thread(list_shares)
        if not shares:
            w(CLR)
            w(f"{YEL}no shares registered{RST}\n")
            w(f"{D}→ use 'Create new share' or 'Join share' first{RST}\n")
            await pause()
            return
        items: List[Tuple[str, str]] = []
        for s in shares:
            sid = s.get("share_id", "?")
            vp = s.get("vault_path", "?")
            items.append((sid, f"{sid}   {D}{vp}{RST}"))
        items.append(("__back__", f"{D}← back{RST}"))
        choice = await select_menu("Manage shares", items)
        if choice in (None, "__back__"):
            return
        await action_share_detail(choice)


async def action_server_control() -> None:
    while True:
        status = await asyncio.to_thread(get_sync_status)
        if status.get("running"):
            items = [
                ("stop", "stop server"),
                ("restart", "restart server"),
                ("back", "← back"),
            ]
        else:
            items = [
                ("start", "start server"),
                ("back", "← back"),
            ]
        choice = await select_menu("Server control", items, get_sync_status)
        if choice in (None, "back"):
            return
        if choice == "start":
            shares = await asyncio.to_thread(list_shares)
            default_vault = (
                shares[0]["vault_path"] if shares else os.path.expanduser("~/vault")
            )
            vault = await prompt("Vault path", default_vault)
            port_s = await prompt("WS port", "8765")
            port = int(port_s) if port_s.isdigit() else 8765
            res = await asyncio.to_thread(start_sync, vault, "default", port)
            col = GRN if res.get("ok") else RED
            w(f"{col}{res}{RST}\n")
            await pause()
        elif choice == "stop":
            res = await asyncio.to_thread(stop_sync)
            col = GRN if res.get("ok") else RED
            w(f"{col}{res}{RST}\n")
            await pause()
        elif choice == "restart":
            shares = await asyncio.to_thread(list_shares)
            vault = shares[0]["vault_path"] if shares else os.path.expanduser("~/vault")
            res = await asyncio.to_thread(restart_sync, vault, "default", 8765)
            col = GRN if res.get("ok") else RED
            w(f"{col}{res}{RST}\n")
            await pause()


async def action_status() -> None:
    w(CLR)
    status = await asyncio.to_thread(get_sync_status)
    w(f"{B}── Status ──{RST}\n\n")
    running = status.get("running")
    col = GRN if running else RED
    w(f"  running: {col}{running}{RST}\n")
    w(f"  pid:     {status.get('pid')}\n")
    shares = status.get("shares", [])
    w(f"  shares:  {len(shares)}\n\n")
    for s in shares:
        w(f"    • {B}{s.get('share_id')}{RST}  {D}{s.get('vault_path')}{RST}\n")
    await pause()


# ── Main loop ────────────────────────────────────────────────────────

async def main() -> None:
    items = [
        ("create",  "Create new share     (first sync, gen token)"),
        ("join",    "Join existing share  (paste peer's token)"),
        ("manage",  "Manage shares        (list / stop / log)"),
        ("server",  "Server control       (start / stop / restart)"),
        ("status",  "Status"),
        ("quit",    "Quit"),
    ]
    try:
        while True:
            choice = await select_menu("Main menu", items, get_sync_status)
            if choice in (None, "quit"):
                w(CLR)
                w(f"{D}bye.{RST}\n")
                return
            if choice == "create":
                await action_create()
            elif choice == "join":
                await action_join()
            elif choice == "manage":
                await action_manage_shares()
            elif choice == "server":
                await action_server_control()
            elif choice == "status":
                await action_status()
    except KeyboardInterrupt:
        w(CLR + SHOW)
        w(f"{D}interrupted.{RST}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        w(SHOW)
        sys.exit(0)
