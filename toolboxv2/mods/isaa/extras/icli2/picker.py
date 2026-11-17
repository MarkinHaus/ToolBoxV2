"""Fuzzy-Auswahl als Overlay. Wiederverwendbar fuer /agent, /model, /session, Palette."""
from __future__ import annotations

from rich.markup import escape
from textual import events, on
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


BACK = "\x00back"        # Rueckgabewert: eine Ebene zurueck (statt ganz schliessen)
NAV = {"w": -1, "s": 1}  # WASD steuert, solange der Filter leer ist


class _Filter(Input):
    """Eingabefeld des Pickers: Steuertasten gehen zuerst an den Picker (das Feld wuerde sie sonst
    als Text schlucken)."""

    async def _on_key(self, event: events.Key) -> None:
        if not self.screen.handle_nav(event):
            await super()._on_key(event)


class Picker(ModalScreen[str | None]):
    """Bedienung: ↑↓ oder W/S waehlen, → / D / Enter oeffnet, ← / A geht zurueck, Esc schliesst.
    Tippen filtert; W/A/S/D steuern nur bei leerem Filter - '/' startet das Filtern ausdruecklich."""

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
        # nie an fehlerhaften Zeilen scheitern: alles zu Text machen, Zeilen ohne Wert verwerfen
        self._title = title
        self._items = [(str(v), str(l or v), str(h or "")) for v, l, h in items if v is not None]

    def compose(self) -> ComposeResult:
        with Vertical() as box:
            box.border_title = self._title
            yield _Filter(placeholder="filtern …   ↑↓/WS wählen · →/D öffnen · ←/A zurück")
            yield OptionList()

    def on_mount(self) -> None:
        self._fill("")

    def _fill(self, needle: str) -> None:
        options = self.query_one(OptionList)
        options.clear_options()
        self._shown = []                                 # Werte nach Position: doppelte Werte sind erlaubt
        for value, label, hint in self._items:
            if fuzzy(needle, label):
                text = escape(label) + (f"  [dim]{escape(hint)}[/]" if hint else "")
                options.add_option(Option(text))
                self._shown.append(value)
        if options.option_count:
            options.highlighted = 0

    @on(Input.Changed)
    def _filter(self, message: Input.Changed) -> None:
        self._fill(message.value.strip())

    def handle_nav(self, event: events.Key) -> bool:
        """True, wenn die Taste eine Steuertaste war und verarbeitet wurde."""
        box = self.query_one(Input)
        empty = not box.value
        key = event.key
        if key == "right" and (empty or box.cursor_position >= len(box.value)) or (empty and key == "d"):
            event.prevent_default().stop()
            self._enter()
        elif empty and key in ("left", "a"):
            event.prevent_default().stop()
            self.dismiss(BACK)
        elif empty and key in NAV:
            event.prevent_default().stop()
            self.action_move(NAV[key])
        elif empty and key == "slash":                   # ausdruecklich filtern (auch mit w/a/s/d am Anfang)
            event.prevent_default().stop()
            box.value = " "
            box.cursor_position = 1
        else:
            return False
        return True

    def action_move(self, step: int) -> None:
        options = self.query_one(OptionList)
        if options.option_count:
            options.highlighted = ((options.highlighted or 0) + step) % options.option_count

    @on(Input.Submitted)
    def _enter(self) -> None:
        options = self.query_one(OptionList)
        if options.highlighted is not None and options.option_count:
            self.dismiss(self._shown[options.highlighted])

    @on(OptionList.OptionSelected)
    def _click(self, message: OptionList.OptionSelected) -> None:
        self.dismiss(self._shown[message.option_index])
