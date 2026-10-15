"""Standalone, reusable terminal pretty-printer for the Page engine.

Turns the engine's *default* outputs — dicts, lists, JSON strings, tool results,
``code_run`` stdout — into nicely formatted, colored **cards** instead of raw
one-line dumps like ``[{"role": "user", "content": ...}]``.

Design goals (so it can be reused elsewhere, e.g. dropped into icli later):
  * **zero dependencies** — stdlib only, imports nothing from toolboxv2;
  * **pure functions** — everything returns ``str``; no printing, no globals, so
    a caller can route the output through prompt_toolkit HTML, plain ANSI, a log,
    or a test assertion;
  * **color is a toggle** on :class:`Theme` (``color=False`` → plain text), and
    :func:`strip_ansi` removes color for width calc / testing;
  * **never raises** on arbitrary input (bytes, None, deeply nested, huge).

Public API:
    Theme(color=..., width=..., palette=...)   — look & feel
    format_value(obj, *, theme, max_chars)     — any value -> pretty block
    format_tool_result(name, text, *, theme, ok)  — a tool result -> a Card
    Card(title, theme).line(...).value(...).render()  — build a bordered card
    strip_ansi(s)                               — remove ANSI codes
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# ---- 16-color ANSI (truecolor-free so it works on any terminal) -------------
_ANSI = {
    "reset": "\x1b[0m",
    "dim": "\x1b[2;37m",
    "cyan": "\x1b[36m",
    "green": "\x1b[32m",
    "red": "\x1b[31m",
    "amber": "\x1b[33m",
    "white": "\x1b[37m",
    "bright": "\x1b[1;37m",
    "blue": "\x1b[34m",
    "purple": "\x1b[35m",
    "grey": "\x1b[90m",
}
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(s: str) -> str:
    """Remove ANSI color codes (for width math or test assertions)."""
    return _ANSI_RE.sub("", s)


@dataclass
class Theme:
    """Look & feel.  ``color=False`` emits plain text (fully strip-equal)."""

    color: bool = True
    width: int = 72
    # role -> palette key
    palette: dict = field(default_factory=lambda: {
        "title": "bright", "key": "cyan", "value": "white", "muted": "grey",
        "ok": "green", "warn": "amber", "err": "red", "border": "dim",
        "accent": "blue", "json": "purple",
    })
    # box-drawing
    tl: str = "┌"
    bl: str = "└"
    h: str = "─"
    v: str = "│"

    def paint(self, text: str, role: str) -> str:
        if not self.color:
            return text
        code = _ANSI.get(self.palette.get(role, "white"), _ANSI["white"])
        return f"{code}{text}{_ANSI['reset']}"


# ================================================================== value formatting
def _coerce(obj):
    """Best-effort: parse JSON strings; decode bytes; leave the rest."""
    if isinstance(obj, (bytes, bytearray)):
        try:
            obj = obj.decode("utf-8", "replace")
        except Exception:
            return repr(obj)
    if isinstance(obj, str):
        s = obj.strip()
        if s and s[0] in "[{" and s[-1] in "]}":
            try:
                return json.loads(s)
            except (json.JSONDecodeError, ValueError):
                return obj
    return obj


def _clip(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = text[: max(1, max_chars - 40)]
    return f"{head}\n…[{len(text)} chars total, clipped]"


def _fmt(obj, theme: Theme, indent: int, max_chars: int) -> list[str]:
    """Recursively render a value into indented, colored lines."""
    pad = "  " * indent
    lines: list[str] = []

    if isinstance(obj, dict):
        if not obj:
            lines.append(pad + theme.paint("{}", "muted"))
        for k, v in obj.items():
            key = theme.paint(f"{k}:", "key")
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}{key}")
                lines.extend(_fmt(v, theme, indent + 1, max_chars))
            else:
                val = _scalar(v, theme, max_chars)
                lines.append(f"{pad}{key} {val}")
    elif isinstance(obj, (list, tuple)):
        if not obj:
            lines.append(pad + theme.paint("[]", "muted"))
        for item in obj:
            if isinstance(item, (dict, list)) and item:
                lines.append(pad + theme.paint("-", "muted"))
                lines.extend(_fmt(item, theme, indent + 1, max_chars))
            else:
                lines.append(f"{pad}{theme.paint('-', 'muted')} "
                             f"{_scalar(item, theme, max_chars)}")
    else:
        lines.append(pad + _scalar(obj, theme, max_chars))
    return lines


def _scalar(v, theme: Theme, max_chars: int) -> str:
    if v is None:
        return theme.paint("null", "muted")
    if isinstance(v, bool):
        return theme.paint(str(v), "accent")
    if isinstance(v, (int, float)):
        return theme.paint(str(v), "accent")
    text = v if isinstance(v, str) else str(v)
    text = _clip(text, max_chars)
    # multi-line scalars: keep readable, color each line as value
    if "\n" in text:
        return "\n".join(theme.paint(ln, "value") for ln in text.splitlines())
    return theme.paint(text, "value")


def format_value(obj, *, theme: Theme | None = None, max_chars: int = 1200) -> str:
    """Render any value into a pretty, optionally-colored multi-line block.

    JSON strings are detected and expanded; dicts/lists become aligned key/value
    trees; bytes/None never raise; long scalars are clipped with a marker.
    """
    theme = theme or Theme()
    try:
        coerced = _coerce(obj)
        return "\n".join(_fmt(coerced, theme, 0, max_chars))
    except Exception as e:  # formatter must never break a run
        return theme.paint(f"[unrenderable {type(obj).__name__}: {e}]", "muted")


# ================================================================== cards
class Card:
    """A bordered, titled card.  Build with ``line``/``value``, then ``render``."""

    def __init__(self, title: str, *, theme: Theme | None = None, accent: str = "border"):
        self.theme = theme or Theme()
        self.title = title
        self.accent = accent
        self._body: list[str] = []

    def line(self, text: str = "", role: str = "value") -> "Card":
        for ln in str(text).splitlines() or [""]:
            self._body.append(self.theme.paint(ln, role))
        return self

    def value(self, obj, max_chars: int = 1200) -> "Card":
        block = format_value(obj, theme=self.theme, max_chars=max_chars)
        self._body.extend(block.splitlines())
        return self

    def render(self) -> str:
        th = self.theme
        title = th.paint(f" {self.title} ", "title")
        bar_len = max(2, th.width - len(strip_ansi(title)) - 2)
        top = th.paint(th.tl + th.h, self.accent) + title + th.paint(
            th.h * bar_len, self.accent)
        rows = [f"{th.paint(th.v, self.accent)} {ln}" for ln in self._body]
        bottom = th.paint(th.bl + th.h * (th.width - 1), self.accent)
        return "\n".join([top, *rows, bottom])


# ================================================================== tool results
def format_tool_result(name: str, text: str, *, theme: Theme | None = None,
                       ok: bool = True, max_chars: int = 1600) -> str:
    """Render a tool result (``code_run`` stdout, ``recall`` hits, JSON, ...) as a Card."""
    theme = theme or Theme()
    accent = "ok" if ok else "err"
    card = Card(name if ok else f"{name} · error", theme=theme, accent=accent)

    raw = text if isinstance(text, str) else str(text)

    # code_run style: "ok\nstdout:\n...\nresult: X"
    if raw.lstrip().startswith("ok") and ("stdout:" in raw or "result:" in raw):
        for ln in raw.splitlines():
            role = ("muted" if ln.strip() in ("ok", "stdout:")
                    else "accent" if ln.startswith("result:")
                    else "value")
            card.line(ln, role)
        return card.render()

    coerced = _coerce(raw)
    if isinstance(coerced, (dict, list)):
        card.value(coerced, max_chars=max_chars)
    else:
        card.line(_clip(raw, max_chars), "err" if not ok else "value")
    return card.render()
