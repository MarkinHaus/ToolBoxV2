# -*- coding: utf-8 -*-
"""Widget-Baukasten: reine Zeilen-Fabrik + Modal/Card (textual lazy).

build_widget_lines ist die GEMEINSAME Grundlage fuer Modal (Overlay) und
Board-Karte - keine doppelte Render-Logik. Vorlage fuer Interaktion/Dismiss:
picker.py:24-116 (ModalScreen, dismiss, OptionList).
"""
from __future__ import annotations

MAX_LINE = 100  # Info-Text Zeilenbreite (TUI)

try:  # textual ist optional im Test-venv; Klassen entstehen nur wenn verfuegbar
    from textual.app import ComposeResult
    from textual.containers import Vertical
    from textual.screen import ModalScreen
    from textual.widget import Widget
    from textual.widgets import Static

    TEXTUAL = True
except Exception:  # pragma: no cover - Testpfad ohne textual
    TEXTUAL = False
    Static = Widget = Vertical = object
    ComposeResult = None
    ModalScreen = object


def _clip(s: str, width: int = MAX_LINE) -> str:
    return s if len(s) <= width else s[: width - 1] + "\u2026"


def build_widget_lines(spec: dict) -> list[str]:
    """Spec -> Anzeige-Zeilen (reine Funktion, Grundlage fuer Modal + Card)."""
    template = spec.get("template")
    lines: list[str] = []
    if template == "info":
        text = str(spec.get("text", ""))
        lines = [_clip(text[i:i + MAX_LINE]) for i in range(0, len(text), MAX_LINE)] or [""]
    elif template == "choice":
        for opt in spec.get("options", []):
            lines.append(_clip(str(opt.get("label", opt.get("value", "")))))
    elif template == "form":
        for f in spec.get("fields", []):
            head = f.get("label") or f.get("name", "")
            ftype = f.get("type")
            if ftype == "checkbox":
                mark = "[x]" if f.get("default") else "[ ]"
                lines.append(_clip(f"{mark} {head}"))
            elif ftype == "select":
                opts = " | ".join(str(o.get("label", o.get("value", "")))
                                  for o in f.get("options", []))
                lines.append(_clip(f"{head}: {opts}"))
            else:
                default = f.get("default")
                suffix = f" = {default}" if default is not None else ""
                lines.append(_clip(f"{head}{suffix}"))
    return lines


if TEXTUAL:

    class WidgetModal(ModalScreen[dict | None]):
        """Overlay-Widget (modus modal). Enter/Click bestaetigt, Esc bricht ab.
        dismiss({id, values}) bzw. dismiss(None) - Picker-Muster picker.py:33."""

        BINDINGS = []

        def __init__(self, widget_id: str, spec: dict, *, creator: tuple):
            super().__init__()
            self.widget_id = widget_id
            self.spec = spec
            self.creator = creator

        def compose(self) -> ComposeResult:
            with Vertical() as box:
                box.border_title = self.spec.get("title", "Widget")
                yield Static("\n".join(build_widget_lines(self.spec)))

        def on_mount(self) -> None:
            self._values: dict = {}

        def _on_key(self, event) -> None:  # pragma: no cover - Terminal-Pfad
            if event.key == "escape":
                self.dismiss(None)
            elif event.key in ("enter", "return"):
                self.dismiss({"id": self.widget_id, "values": dict(self._values)})

    class WidgetCard(Widget):
        """Karte im Board (modus pinned): Titelzeile + Zeilen + Schliessen."""

        def __init__(self, widget_id: str, spec: dict, *, creator: tuple):
            super().__init__()
            self.widget_id = widget_id
            self.spec = spec
            self.creator = creator

        def compose(self) -> ComposeResult:
            yield Static(self.spec.get("title", "Widget"), classes="w-title")
            yield Static("\n".join(build_widget_lines(self.spec)))

else:  # pragma: no cover
    WidgetModal = None
    WidgetCard = None
