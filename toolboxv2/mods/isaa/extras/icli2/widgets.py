"""Kleine, guenstige Widgets der icli2 (je EIN Widget, damit Layout-Durchlaeufe billig bleiben)."""
from __future__ import annotations

from rich.text import Text
from textual import events
from textual.widgets import Static

from .theme import G


class FoldLine(Static):
    """Ein einziges Widget statt Collapsible (vier Widgets): Kopfzeile, aufgeklappt mit Text darunter.
    Klick oder Enter klappt um. Zugeklappt kostet ein neuer Text keinen Layout-Durchlauf."""

    can_focus = True

    def __init__(self, title: str, body: str = "", collapsed: bool = True, classes: str = ""):
        super().__init__("", classes=classes, markup=False)
        self._title, self._fold_body, self._collapsed = title, body, collapsed

    def on_mount(self) -> None:
        self._draw()

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
            if not self._collapsed:                     # zugeklappt: nichts zu zeichnen
                self._draw()

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    @collapsed.setter
    def collapsed(self, value: bool) -> None:
        if value != self._collapsed:
            self._collapsed = value
            self._draw()

    def _draw(self) -> None:
        if not self.is_mounted:
            return
        text = Text(("▶ " if self._collapsed else "▼ ") + self._title)
        if not self._collapsed and self._fold_body:
            for line in self._fold_body.splitlines() or [""]:
                text.append("\n  │ ", style=G.L2_DARK)
                text.append(line)
        self.update(text)

    def on_click(self, event: events.Click) -> None:
        event.stop()                                    # nicht an umgebende Kachel weiterreichen
        self.collapsed = not self._collapsed

    def key_enter(self) -> None:
        self.collapsed = not self._collapsed
