"""Fuzzy-Auswahl als Overlay. Wiederverwendbar fuer /agent, /model, /session, Palette."""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList
from textual.widgets.option_list import Option


def fuzzy(needle: str, hay: str) -> bool:
    """Teilfolgen-Treffer: 'cdr' findet 'coder'."""
    it = iter(hay.lower())
    return all(ch in it for ch in needle.lower())


class Picker(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", priority=True),
                Binding("down", "move(1)", priority=True),
                Binding("up", "move(-1)", priority=True)]
    DEFAULT_CSS = """
    Picker { align: center middle; background: $background 70%; }
    Picker > Vertical { width: 64; max-width: 90%; height: auto; max-height: 70%;
                        border: round $primary; background: $background; padding: 0 1; }
    Picker Input, Picker Input:focus { border: none; height: 1; padding: 0; background: transparent; }
    Picker OptionList { border: none; height: auto; max-height: 16; padding: 0;
                        background: transparent; }
    """

    def __init__(self, title: str, items: list[tuple[str, str, str]]):
        """items: (wert, anzeige, hinweis)"""
        super().__init__()
        self._title, self._items = title, items

    def compose(self) -> ComposeResult:
        with Vertical() as box:
            box.border_title = self._title
            yield Input(placeholder="filtern …")
            yield OptionList()

    def on_mount(self) -> None:
        self._fill("")

    def _fill(self, needle: str) -> None:
        options = self.query_one(OptionList)
        options.clear_options()
        for value, label, hint in self._items:
            if fuzzy(needle, label):
                options.add_option(Option(f"{label}  [dim]{hint}[/]" if hint else label, id=value))
        if options.option_count:
            options.highlighted = 0

    @on(Input.Changed)
    def _filter(self, message: Input.Changed) -> None:
        self._fill(message.value)

    def action_move(self, step: int) -> None:
        options = self.query_one(OptionList)
        if options.option_count:
            options.highlighted = ((options.highlighted or 0) + step) % options.option_count

    @on(Input.Submitted)
    def _enter(self) -> None:
        options = self.query_one(OptionList)
        if options.highlighted is not None and options.option_count:
            self.dismiss(options.get_option_at_index(options.highlighted).id)

    @on(OptionList.OptionSelected)
    def _click(self, message: OptionList.OptionSelected) -> None:
        self.dismiss(message.option.id)
