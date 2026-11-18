# -*- coding: utf-8 -*-
"""Kindprozess fuer test_10: startet die icli2 ueber den Starter (flows/isaa/icli2.py:build) mit dem
Ersatz-LLM aus der Umgebung, bedient das Widget-Modal und meldet Marken 'BETRIEB ...' auf stdout.

Einziges Double ausser dem LLM: Embedding-Modell (Netzgrenze, wie icli2/tests/e2e/e2e_bg.py)."""
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

sys.modules["litellm"] = None          # eigener Prozess: litellm gesperrt, der Lauf muss ohne gehen


def mark(text: str) -> None:
    print("BETRIEB " + text, flush=True)


async def main() -> None:
    import numpy as np
    from toolboxv2.mods.isaa.base import local_embeddings as LE

    async def no_model(texts, dimensions=None):
        return np.zeros((len(texts), dimensions or 768), dtype="float32")
    LE.local_embed = no_model

    from textual.widgets import Checkbox, Input
    from toolboxv2 import get_app
    from toolboxv2.flows.isaa.icli2 import build
    from toolboxv2.mods.isaa.extras.icli2 import protocol as P
    from toolboxv2.mods.isaa.extras.icli2.widgets import WidgetModal

    app = get_app("isaa-host")
    import os
    if not str(Path(app.appdata).resolve()).startswith(str(Path(os.environ["TB_DATA_DIR"]).resolve())):
        mark(f"nicht isoliert: {app.appdata}")          # nie ins echte Profil schreiben (Regel 6)
        return
    engine, ui = build(app)
    spec = {"title": "Ziel", "template": "form", "mode": "modal", "fields": [
        {"name": "ort", "label": "Ort", "type": "text"},
        {"name": "eilig", "label": "Eilig", "type": "checkbox"}]}

    async def until(pilot, pred, timeout, what):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if pred():
                return True
            await pilot.pause(0.1)
        mark(f"timeout {what}")
        return False

    try:
        async with ui.run_test(size=(140, 45)) as pilot:
            if not await until(pilot, lambda: ui.ready, 60, "ready"):
                return
            ui.prompt.text = "CALL widget_show " + json.dumps(spec)
            await pilot.press("enter")
            if not await until(pilot, lambda: isinstance(ui.screen, WidgetModal), 60, "modal"):
                return
            mark("modal")
            scr = ui.screen
            await pilot.click(scr.query_one(Input))
            await pilot.press(*"Kiel")
            await pilot.click(scr.query_one(Checkbox))
            await pilot.click(scr.query_one(Input))
            await pilot.press("enter")
            path = Path(app.appdata) / "icli2" / "widgets.json"

            def values():
                try:
                    return next(iter(json.loads(path.read_text("utf-8")).values()))["values"]
                except Exception:
                    return {}
            if await until(pilot, lambda: bool(values()), 30, "werte"):
                mark("werte " + json.dumps(values()))
            if await until(pilot, lambda: any(v.origin == "widget" and v.status != "running"
                                              for v in ui.runs.values()), 60, "folgelauf"):
                run = next(v for v in ui.runs.values() if v.origin == "widget")
                mark(f"folgelauf {run.status}")
    finally:
        engine.send(P.SHUTDOWN)
        engine.join(30)
        mark("ende" if not engine._thread.is_alive() else "engine haengt beim Beenden")
        await app.a_exit()                               # beendet den Prozess (wie der Starter)


try:
    asyncio.run(main())
except BaseException:
    traceback.print_exc()
