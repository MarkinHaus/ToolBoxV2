"""Eingefrorene Abschnitte (FrozenMarkdown) muessen genau so hoch sein wie das Live-Markdown,
sonst springt der Text beim Einfrieren. Prueft mehrere Beispiele bei drei Breiten."""
import asyncio, sys
import stub  # noqa
from textual.app import App
from textual.containers import VerticalScroll
from textual.widgets import Markdown
from toolboxv2.mods.isaa.extras.icli2.app import ICli2
from toolboxv2.mods.isaa.extras.icli2.richmd import FrozenMarkdown
from toolboxv2.mods.isaa.extras.icli2.theme import ISAA_THEME
SAMPLES = {
 "absatz": "Nur ein Absatz mit **fett** und `code`.",
 "zwei": "Erster Absatz.\n\nZweiter Absatz, etwas länger, damit er bei schmaler Breite umbricht und mehrere Zeilen braucht.",
 "ueberschrift": "## Titel\n\nText darunter.\n\n### Unter\n\nNoch Text.",
 "liste": "- a\n- b mit `x`\n\n1. eins\n2. zwei\n3. drei",
 "code": "Vorher.\n\n```python\ndef f():\n    return 1\n```\n\nNachher.",
 "zitat": "> zitiert\n\nText.",
 "mix": "## Ergebnis\n\nIch habe **x** geprüft.\n\n- p1\n- p2\n\n```py\nx = 1\n```\n\n### Details\n\nEnde.",
 "nur_titel_zuerst": "# Groß\n\nText",
 "lange_liste": "\n".join(f"{i}. Punkt {i}" for i in range(1, 12)),
}
class T(App):
    CSS = ICli2.CSS
    def compose(self):
        with VerticalScroll(id="chat"):
            for k, md in SAMPLES.items():
                yield Markdown(md, classes="answer", id=f"live_{k}")
                yield FrozenMarkdown(md)
    def on_mount(self):
        self.register_theme(ISAA_THEME); self.theme = "isaa"
async def check(width: int) -> None:
    app = T()
    async with app.run_test(size=(width, 260)) as pilot:
        await pilot.pause(0.5)
        lives, frozen = list(app.query(Markdown)), list(app.query(FrozenMarkdown))
        assert len(lives) == len(frozen) == len(SAMPLES)
        for (k, md), lw, fw in zip(SAMPLES.items(), lives, frozen):
            assert lw.outer_size.height == fw.outer_size.height, (width, k, lw.outer_size.height, fw.outer_size.height)
            assert fw.source == md


async def main():
    for width in (70, 34, 120):                          # Umbrueche fallen je Breite anders aus
        await check(width)
    print("render ok")


asyncio.run(main())
