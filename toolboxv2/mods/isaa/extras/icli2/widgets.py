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
    from rich.text import Text
    from textual import events
    from .theme import BG,G

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

    class Fade(Widget):
        """Eine Zeile Abstand ueber der Eingabe mit weichem Uebergang: Halbbloecke in einer Farbe zwischen
        Hintergrund und Eingabeleiste (eine Zeichenzeile kann keinen echten Verlauf zeigen)."""

        DEFAULT_CSS = "Fade { height: 1; }"
        TONE = "#0a1317"  # zwischen BG (#070d10) und Eingabe (#0c161c)

        def render(self) -> Text:
            return Text("▄" * max(1, self.size.width), style=f"{self.TONE} on {BG}")


    class FoldLine(Static):
        """Ein einziges Widget statt Collapsible (vier Widgets): Kopfzeile, aufgeklappt mit Text darunter.
        Klick oder Enter klappt um. Zugeklappt kostet ein neuer Text keinen Layout-Durchlauf."""

        can_focus = True

        def __init__(self, title: str, body: str = "", collapsed: bool = True, classes: str = "",
                     body_style: str = ""):
            super().__init__("", classes=classes, markup=False)
            self._title, self._fold_body, self._collapsed = title, body, collapsed
            self._body_style = body_style  # z. B. lesbare Farbe unter einer roten Fehlerzeile

        def on_mount(self) -> None:
            self._draw(force=True)  # waehrend on_mount gilt is_mounted noch nicht

        @property
        def title(self) -> str:
            return self._title

        @title.setter
        def title(self, value: str) -> None:
            if value != self._title:
                self._title = value
                self._draw()

        @property
        def body(self) -> str:
            return self._fold_body

        @body.setter
        def body(self, value: str) -> None:
            if value != self._fold_body:
                self._fold_body = value
                if not self._collapsed:  # zugeklappt: nichts zu zeichnen
                    self._draw()

        @property
        def collapsed(self) -> bool:
            return self._collapsed

        @collapsed.setter
        def collapsed(self, value: bool) -> None:
            if value != self._collapsed:
                self._collapsed = value
                self._draw()

        def _draw(self, force: bool = False) -> None:
            if not (force or self.is_mounted):
                return
            text = Text(("▶ " if self._collapsed else "▼ ") + self._title)
            if not self._collapsed and self._fold_body:
                for line in self._fold_body.splitlines() or [""]:
                    text.append("\n  │ ", style=G.L2_DARK)
                    text.append(line, style=self._body_style)
            self.update(text)

        def on_click(self, event: events.Click) -> None:
            event.stop()  # nicht an umgebende Kachel weiterreichen
            self.collapsed = not self._collapsed

        def key_enter(self) -> None:
            self.collapsed = not self._collapsed


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
    FoldLine = None
    Fade = None


def format_answer(spec: dict, values: dict) -> str:
    """Schoene Verlaufs-Zeile ohne ID/JSON - icli2-Stil (Picker-Label-Vorlage)."""
    title = spec.get("title", "Widget")
    template = spec.get("template")
    parts: list[str] = []
    if template == "form":
        for f in spec.get("fields", []):
            name = f.get("name")
            if name in values:
                parts.append(f"{f.get('label') or name}: {values[name]}")
        for k, v in values.items():                     # Werte ohne Felddef nicht verlieren
            if not any(f.get("name") == k for f in spec.get("fields", [])):
                parts.append(f"{k}: {v}")
    elif template == "choice":
        v = values.get("value", values.get("choice"))
        if v is not None:
            parts.append(str(v))
    else:
        parts = [f"{k}: {v}" for k, v in values.items()]
    sep = " \u2013 " if parts else ""
    return f"{title}{sep}" + ", ".join(str(p) for p in parts)
