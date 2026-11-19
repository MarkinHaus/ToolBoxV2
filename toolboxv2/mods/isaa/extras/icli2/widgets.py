# -*- coding: utf-8 -*-
"""Widget-Baukasten: reine Zeilen-Fabrik + Modal/Card (textual lazy).

build_widget_lines ist die GEMEINSAME Grundlage fuer Modal (Overlay) und
Board-Karte - keine doppelte Render-Logik. Vorlage fuer Interaktion/Dismiss:
picker.py:24-116 (ModalScreen, dismiss, OptionList).
"""
from __future__ import annotations

from typing import Any

MAX_LINE = 100  # Info-Text Zeilenbreite (TUI)

try:  # textual ist optional im Test-venv; Klassen entstehen nur wenn verfuegbar
    from textual.app import ComposeResult
    from textual.containers import Vertical, VerticalScroll
    from textual.screen import ModalScreen
    from textual.widget import Widget
    from textual.widgets import Button, Checkbox, Input, Label, OptionList, Static, TextArea
    from textual.widgets.option_list import Option
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


def _chart_lines(spec: dict) -> list[str]:
    """ASCII-Grafik fuer chart (passiv): line = Sparkline, bar = Balken je Wert.
    ponytail: Max 50 Werte (Spec); pipe/flow-Faelle deckt das progress-Template ab."""
    vals = [float(v) for v in spec.get("values", [])]
    if not vals:
        return [""]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    steps = "▁▂▃▄▅▆▇█"
    if spec.get("chart_type") == "bar":
        width = 40
        return [_clip("█" * max(1, round((v - lo) / span * width)) + f" {v:g}") for v in vals]
    spark = "".join(steps[max(0, min(7, round((v - lo) / span * 7)))] for v in vals)
    return [_clip(f"{spark}  {lo:g}..{hi:g}")]


def build_widget_lines(spec: dict) -> list[str]:
    """Spec -> Anzeige-Zeilen (reine Funktion, Grundlage fuer Modal + Card)."""
    template = spec.get("template")
    lines: list[str] = []
    if template == "mermaid":
        src = str(spec.get("text", "")).strip()
        lines = ["```mermaid"] + [_clip(x) for x in src.splitlines()] + ["```"]
    if template == "chart":
        lines = _chart_lines(spec)
    elif template == "info":
        text = str(spec.get("text", ""))
        lines = [_clip(text[i:i + MAX_LINE]) for i in range(0, len(text), MAX_LINE)] or [""]
    elif template in ("note", "progress"):
        text = str(spec.get("text", ""))
        if template == "progress":
            lines.append(_clip(f"{spec.get('progress', 0)}% - {text}"))
        else:
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



    class InlineCard(Static):
        """Karte im Chat-Verlauf (modus inline): note = Markdown-Text, progress = Text + Balken.
        update_from ersetzt Inhalt und Fortschritt (widget_update)."""

        DEFAULT_CSS = """
        InlineCard { width: 1fr; height: auto; margin: 0 0 1 0; }
        """

        def __init__(self, widget_id: str, spec: dict):
            self.widget_id = widget_id
            self.spec: dict = {}
            self._text = ""
            super().__init__("", classes="inline-note" if spec.get("template") == "note" else "inline-progress")
            self.update_from(spec)

        @property
        def source(self) -> str:                          # Lesezugriff wie Markdown.source
            return self._text

        @property
        def renderable(self) -> str:
            return self._text

        def update(self, content: object = "") -> None:
            self._text = str(content)
            super().update(content)

        def update_from(self, spec: dict) -> None:
            self.spec.update(spec)
            tmpl, text = spec.get("template"), str(spec.get("text", ""))
            if tmpl == "progress":
                pct = spec.get("progress") or 0
                bar = "█" * round(pct / 10) + "░" * (10 - round(pct / 10))
                self.update(f"{spec.get('title', '')}\n{bar} {pct}%\n{text}")
            elif tmpl == "chart":
                self.update(f"{spec.get('title', '')}\n" + "\n".join(_chart_lines(self.spec)))
            elif tmpl == "mermaid":
                self.update(f"{spec.get('title', '')}\n```mermaid\n{spec.get('text', '')}\n```")
            else:
                self.update(f"{spec.get('title', '')}\n{text}")

    class WidgetModal(ModalScreen[dict | None]):
        """Overlay-Widget (modus modal, oder ein gepinntes aus /widget). Picker-Muster picker.py:33.

        form:   Eingabefelder (Input/TextArea/Checkbox/OptionList); Enter in einem Feld oder 'Senden'
                prueft und schliesst mit {id, values}
        choice: OptionList; Enter/Klick auf eine Option schliesst mit {id, values: {value}}
        info:   nur Text; Enter/Esc schliesst ohne Antwort (None)
        Esc bricht immer ohne Antwort ab (None)."""

        DEFAULT_CSS = """
        WidgetModal { align: center middle; }
        WidgetModal > #w-box { width: 76; max-width: 95%; height: auto; max-height: 90%;
                               border: round $accent; background: $surface; padding: 0 1; }
        WidgetModal .w-label { color: $text-muted; margin-top: 1; }
        WidgetModal Input, WidgetModal TextArea { width: 100%; }
        WidgetModal TextArea { height: 4; }
        WidgetModal OptionList { height: auto; max-height: 8; }
        WidgetModal #w-error { color: $error; height: auto; }
        WidgetModal #w-send { margin-top: 1; }
        """
        BINDINGS = [("escape", "cancel", "Abbrechen"), ("ctrl+s", "submit", "Senden")]

        def __init__(self, widget_id: str, spec: dict, *, creator: tuple, values: dict | None = None):
            super().__init__()
            self.widget_id = widget_id
            self.spec = spec
            self.creator = creator
            self._prev = dict(values or {})              # schon beantwortet: Werte vorbelegen
            self._fields: list[tuple[dict, Any]] = []    # (Felddefinition, Eingabe-Widget)
            self._chosen: dict[int, int] = {}            # OptionList-Index im Formular -> gewaehlte Option

        def compose(self) -> ComposeResult:
            template = self.spec.get("template")
            with VerticalScroll(id="w-box") as box:
                box.border_title = self.spec.get("title", "Widget")
                box.border_subtitle = "Esc abbrechen"
                if template == "info":
                    yield Static(str(self.spec.get("text", "")), markup=False)
                    yield Button("OK", id="w-ok", variant="primary")
                elif template == "choice":
                    yield OptionList(*[Option(str(o["label"])) for o in self.spec.get("options", [])],
                                     id="w-choice")
                else:
                    for f in self.spec.get("fields", []):
                        yield from self._field(f)
                    yield Static("", id="w-error")
                    yield Button("Senden", id="w-send", variant="primary")

        def _field(self, f: dict):
            ftype, label = f.get("type"), f.get("label") or f.get("name", "")
            default = self._prev.get(f["name"], f.get("default"))
            if ftype == "checkbox":
                widget = Checkbox(label, value=bool(default))
            else:
                yield Label(label + (" *" if f.get("required") else ""), classes="w-label")
                if ftype == "textarea":
                    widget = TextArea("" if default is None else str(default))
                elif ftype == "select":
                    opts = f.get("options", [])
                    widget = OptionList(*[Option(str(o["label"])) for o in opts])
                    pick = next((i for i, o in enumerate(opts) if o["value"] == default), None)
                    if pick is not None:
                        self._chosen[len(self._fields)] = pick
                else:
                    widget = Input("" if default is None else str(default), placeholder=label,
                                   type="number" if ftype == "number" else "text")
            self._fields.append((f, widget))
            yield widget

        def on_mount(self) -> None:
            for index, (f, widget) in enumerate(self._fields):
                if isinstance(widget, OptionList):
                    widget.highlighted = self._chosen.get(index, 0)
            if self.spec.get("template") == "choice":
                self.query_one("#w-choice", OptionList).highlighted = 0
            focusable = [w for _, w in self._fields] or list(self.query(OptionList)) or list(self.query(Button))
            if focusable:
                focusable[0].focus()

        # -- Eingaben ----------------------------------------------------------
        def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
            event.stop()
            if self.spec.get("template") == "choice":
                opt = self.spec.get("options", [])[event.option_index]
                self.dismiss({"id": self.widget_id, "values": {"value": opt["value"]}})
                return
            for index, (_, widget) in enumerate(self._fields):
                if widget is event.option_list:
                    self._chosen[index] = event.option_index

        def on_input_submitted(self, event: Input.Submitted) -> None:
            event.stop()
            self.action_submit()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            event.stop()
            if event.button.id == "w-ok":
                self.dismiss(None)
            else:
                self.action_submit()

        def action_cancel(self) -> None:
            self.dismiss(None)

        def values(self) -> tuple[dict, list[str]]:
            """Aktuelle Formularwerte + Fehler (Pflichtfeld, Zahl, Grenzen)."""
            out, errors = {}, []
            for index, (f, widget) in enumerate(self._fields):
                name, ftype, label = f["name"], f.get("type"), f.get("label") or f["name"]
                if ftype == "checkbox":
                    out[name] = bool(widget.value)
                    continue
                if ftype == "select":
                    pick = self._chosen.get(index, widget.highlighted)
                    opts = f.get("options", [])
                    value = opts[pick]["value"] if pick is not None and 0 <= pick < len(opts) else None
                elif ftype == "textarea":
                    value = widget.text
                else:
                    value = widget.value
                if value in (None, ""):
                    if f.get("required"):
                        errors.append(f"{label}: Pflichtfeld")
                    out[name] = value if ftype != "number" else None
                    continue
                if ftype == "number":
                    try:
                        num = float(value)
                    except ValueError:
                        errors.append(f"{label}: keine Zahl")
                        continue
                    value = int(num) if num.is_integer() else num
                    if f.get("min") is not None and value < f["min"]:
                        errors.append(f"{label}: mindestens {f['min']}")
                    if f.get("max") is not None and value > f["max"]:
                        errors.append(f"{label}: hoechstens {f['max']}")
                out[name] = value
            return out, errors

        def action_submit(self) -> None:
            template = self.spec.get("template")
            if template == "info":
                self.dismiss(None)
                return
            if template == "choice":
                opts = self.query_one("#w-choice", OptionList)
                if opts.highlighted is not None:
                    opt = self.spec.get("options", [])[opts.highlighted]
                    self.dismiss({"id": self.widget_id, "values": {"value": opt["value"]}})
                return
            values, errors = self.values()
            if errors:
                self.query_one("#w-error", Static).update("  ".join(errors))
                return
            self.dismiss({"id": self.widget_id, "values": values})

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
