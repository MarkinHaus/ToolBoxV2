"""Fertige Antwort-Abschnitte als EIN Widget.

Textuals Markdown-Widget legt fuer jeden Absatz, jede Ueberschrift, jeden Listenpunkt und jeden
Codeblock eigene Widgets an. Das ist zum Streamen noetig, kostet aber bei jedem Layout-Durchlauf
(Scrollen, neuer Text) Zeit proportional zur Gesamtzahl. Ist ein Abschnitt fertig, ersetzt die UI
ihn deshalb durch FrozenMarkdown: ein Static mit Richs Markdown.

Damit beim Einfrieren nichts springt, rendert ThemedMarkdown zeilengleich zum Live-Markdown der UI:
keine Leerzeile nach Ueberschriften (eine davor), "• " und " 1. " fuer Listen, Codeblock auf
Flaeche mit Abstand 1/2, Zitat mit farbigem Balken - und Farben aus dem ISAA-Branding.
"""
from __future__ import annotations

from pygments.style import Style as PygmentsStyle
from pygments.token import Comment, Keyword, Name, Number, Operator, Punctuation, String, Token
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import BlockQuote, CodeBlock, Heading, ListItem, Markdown
from rich.segment import Segment
from rich.style import Style
from rich.syntax import PygmentsSyntaxTheme, Syntax
from rich.theme import Theme
from textual.widgets import Static

from .theme import DIM, G, ZEN

FENCE_BG = "#0a191f"                                     # wie .answer MarkdownFence ($panel 40 %)
_HEADING = Style.from_meta({"icli2": "h"})               # Markierungen fuer die Nachbearbeitung
_LIST = Style.from_meta({"icli2": "li"})


class _CodeStyle(PygmentsStyle):
    background_color = FENCE_BG
    styles = {
        Token: "#d2d2d2", Punctuation: "#d2d2d2",           # Grundfarbe wie MarkdownFence live
        Keyword: G.L4_BRIGHT, Name.Builtin: G.L4_BRIGHT, Name.Function: f"underline {ZEN['amber']}",
        Number: ZEN["amber"], String: ZEN["green"], Comment: f"italic {DIM}", Operator: G.L3_MID,
    }


class _Heading(Heading):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        text = self.text.copy()
        text.justify = "left"                            # Rich wuerde zentrieren
        text.stylize(_HEADING)
        yield text


class _CodeBlock(CodeBlock):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield Syntax(str(self.text).rstrip(), self.lexer_name, theme=PygmentsSyntaxTheme(_CodeStyle),
                     word_wrap=True, padding=(1, 2), background_color=FENCE_BG)


class _BlockQuote(BlockQuote):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        lines = console.render_lines(self.elements, options.update(width=options.max_width - 2))
        bar = Segment("▌ ", Style(color=G.L2_DARK))
        for line in lines:
            yield bar
            yield from line
            yield Segment("\n")


class _ListItem(ListItem):
    def render_bullet(self, console: Console, options: ConsoleOptions) -> RenderResult:
        return self._render(console, options, "• ", console.get_style("markdown.item.bullet", default="none"))

    def render_number(self, console: Console, options: ConsoleOptions, number: int,
                      last_number: int) -> RenderResult:
        width = len(str(last_number)) + 3
        return self._render(console, options, f"{number}.".rjust(width - 1) + " ",
                            console.get_style("markdown.item.number", default="none"))

    def _render(self, console: Console, options: ConsoleOptions, mark: str, style) -> RenderResult:
        lines = console.render_lines(self.elements, options.update(width=options.max_width - len(mark)),
                                     style=self.style)
        style = style + _LIST
        for index, line in enumerate(lines):
            yield Segment(mark if index == 0 else " " * len(mark), style)
            yield from line
            yield Segment("\n")


class _Markdown(Markdown):
    elements = {**Markdown.elements, "heading_open": _Heading, "fence": _CodeBlock,
                "code_block": _CodeBlock, "blockquote_open": _BlockQuote, "list_item_open": _ListItem}


MD_THEME = Theme({
    **{f"markdown.h{n}": f"bold {G.L4_BRIGHT}" for n in (1, 2, 3)},
    **{f"markdown.h{n}": f"bold {G.L3_MID}" for n in (4, 5, 6)},
    "markdown.h1.border": G.L2_DARK,
    "markdown.code": f"{ZEN['amber']} on #1a1608",       # wie Textual: $text-warning auf $warning 10 %
    "markdown.block_quote": "none",
    "markdown.item.bullet": f"bold {G.L3_MID}",
    "markdown.item.number": G.L3_MID,
    "markdown.hr": G.L2_DARK,
    "markdown.link": G.L4_BRIGHT,
    "markdown.link_url": f"underline {G.L3_MID}",
    "markdown.table.border": G.L2_DARK,
    "markdown.table.header": f"bold {G.L4_BRIGHT}",
})


def _marked(line: list[Segment], mark: str) -> bool:
    return any(seg.style is not None and seg.style.meta.get("icli2") == mark for seg in line)


def _is_heading(line: list[Segment]) -> bool:
    return _marked(line, "h")


def _is_blank(line: list[Segment]) -> bool:
    return not "".join(seg.text for seg in line).strip()


class ThemedMarkdown:
    """Rich-Renderable: Markdown im ISAA-Stil, zeilengleich zum Live-Markdown der UI."""

    def __init__(self, source: str):
        self.md = _Markdown(source, hyperlinks=False)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        with console.use_theme(MD_THEME):
            lines = console.render_lines(self.md, options, pad=False, new_lines=False)
        out: list[list[Segment]] = []
        after_heading = False
        for line in lines:
            if after_heading and _is_blank(line):        # Live-Markdown: Text direkt unter der Ueberschrift
                after_heading = False
                continue
            after_heading = _is_heading(line)
            out.append(line)
        if out and _is_heading(out[0]):                  # Live-Markdown: Abstand ueber der ersten Ueberschrift
            out.insert(0, [])
        while out and _is_blank(out[-1]):
            out.pop()
        if out and not _marked(out[-1], "li"):         # Live-Markdown: Abstand unter Absatz/Code/Zitat
            out.append([])
        for line in out:
            yield from line
            yield Segment.line()


class FrozenMarkdown(Static):
    """Fertiger Abschnitt: ein Widget statt vieler; gerendert wird einmal je Breite (Cache)."""

    def __init__(self, source: str, classes: str = "answer"):
        super().__init__(ThemedMarkdown(source), classes=classes)
        self.source = source
