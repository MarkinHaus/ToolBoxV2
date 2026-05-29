# file: toolboxv2/utils/clis/cli_input.py
"""
Keyboard input + interactive menu helpers for ToolBoxV2 CLIs.

Arrow keys and W/S navigate, Enter selects, q/Esc cancels. No number picking.
Cross-platform: Windows (msvcrt) / POSIX (termios+tty). Reusable across every
CLI screen (login, services, mods, ...).
"""

import sys

from .cli_printing import Colors, c_print

__all__ = ["read_key", "menu_select", "menu_select_async"]

_IS_WIN = sys.platform == "win32"


def read_key() -> str:
    """Read one keypress, return a normalized token.

    One of: 'up','down','left','right','enter','esc','backspace',
    or a single printable character. Raises KeyboardInterrupt on Ctrl+C.
    """
    return _read_key_win() if _IS_WIN else _read_key_posix()


def _read_key_win() -> str:
    import msvcrt
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):                       # extended-key prefix
        return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(
            msvcrt.getwch(), ""
        )
    if ch in ("\r", "\n"):
        return "enter"
    if ch == "\x1b":
        return "esc"
    if ch in ("\x08", "\x7f"):
        return "backspace"
    if ch == "\x03":
        raise KeyboardInterrupt
    return ch


def _read_key_posix() -> str:
    import select
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":                             # esc, or arrow sequence
            r, _, _ = select.select([sys.stdin], [], [], 0.0005)
            if not r:
                return "esc"
            seq = sys.stdin.read(2)
            return {"[A": "up", "[B": "down", "[D": "left", "[C": "right"}.get(
                seq, "esc"
            )
        if ch in ("\r", "\n"):
            return "enter"
        if ch in ("\x08", "\x7f"):
            return "backspace"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def menu_select(options, title: str = None, *, start: int = 0, hint: str = None):
    """Interactive single-choice menu navigated by arrows / W,S.

    options: list of (value, label) tuples, or list of label strings.
    Returns the selected value, or None if cancelled (Esc/q/Ctrl+C).
    """
    norm = [o if isinstance(o, tuple) else (o, str(o)) for o in options]
    if not norm:
        return None
    idx = max(0, min(start, len(norm) - 1))

    if title:
        c_print(f"  {Colors.BOLD}{title}{Colors.RESET}")
    if hint:
        c_print(f"  {Colors.DIM}{hint}{Colors.RESET}")

    def render(first: bool):
        if not first:
            c_print(f"\033[{len(norm)}A", end="")     # move up over option block
        for i, (_, label) in enumerate(norm):
            if i == idx:
                c_print(f"\033[K  {Colors.CYAN}\u276f{Colors.RESET} "
                        f"{Colors.BOLD}{label}{Colors.RESET}")
            else:
                c_print(f"\033[K    {Colors.DIM}{label}{Colors.RESET}")

    render(first=True)
    while True:
        try:
            key = read_key()
        except KeyboardInterrupt:
            return None
        if key in ("up", "w", "W"):
            idx = (idx - 1) % len(norm)
            render(first=False)
        elif key in ("down", "s", "S"):
            idx = (idx + 1) % len(norm)
            render(first=False)
        elif key == "enter":
            return norm[idx][0]
        elif key in ("esc", "q", "Q"):
            return None


async def menu_select_async(*args, **kwargs):
    """Async wrapper so event-loop based CLIs don't block the loop."""
    import asyncio
    return await asyncio.to_thread(menu_select, *args, **kwargs)
