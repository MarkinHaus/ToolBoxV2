# -*- coding: utf-8 -*-
"""Betriebsumgebung fuer icli2-Tests: echte ToolBox-App, echter isaa-Mod, echte Engine + IsaaBackend,
LLM ueber das Ersatz-LLM (HTTP, Prozessgrenze). Alle Daten in einem Temp-Ordner (Regel 6).

Einziges weiteres Double: `local_embeddings.local_embed` liefert Nullen. Grund: das echte Embedding-Modell
wird beim ersten Gebrauch aus dem Netz geladen (HuggingFace) - Netzgrenze, siehe icli2/tests/e2e/e2e_bg.py.

Ohne litellm: Das Modell heisst `gateway/fake` und geht ueber den nativen OpenAICompatAdapter des
CompletionRouter (aiohttp, `TB_LLM_GATEWAY_URL`). `litellm` ist im Testprozess gesperrt
(sys.modules["litellm"] = None): jeder Versuch, es zu laden, scheitert - auch wenn es installiert ist.

Schreibschutz: HOME, XDG_*, APPDATA, LOCALAPPDATA und TB_DATA_DIR zeigen in den Temp-Ordner. Zusaetzlich
vergleicht `Schreibschutz` die echten Zustandsordner (Profil, .data/.info/.config des Checkouts) vor und
nach dem Test; jede Aenderung dort laesst den Test scheitern.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from icli2_fake_llm import FakeLLM
from toolboxv2.tests.schreibschutz import Schreibschutz  # noqa: F401  (auch fuer die Tests)

TIMEOUT = 120
MODEL = "gateway/fake"


async def _zero_embed(texts, dimensions=None):
    import numpy as np
    return np.zeros((len(texts), dimensions or 768), dtype="float32")


@contextmanager
def _ohne_litellm():
    """litellm sperren (Import scheitert) und danach den alten Eintrag zurueck. Nur dieser Schluessel:
    patch.dict(sys.modules) wuerde beim Verlassen alle seither geladenen Module entfernen (numpy & Co.)."""
    missing = object()
    old = sys.modules.get("litellm", missing)
    sys.modules["litellm"] = None
    try:
        yield
    finally:
        if old is missing:
            sys.modules.pop("litellm", None)
        else:
            sys.modules["litellm"] = old


class Betrieb:
    """Mit `with Betrieb() as b:` - startet Ersatz-LLM, App, Engine; raeumt alles wieder ab."""

    def __init__(self, root: str | None = None):
        self.root = root or tempfile.mkdtemp(prefix="icli2_betrieb_")
        self.llm: FakeLLM | None = None
        self.app = None
        self.engine = None
        self.backend = None
        self.events: list = []
        self._stack = ExitStack()

    # -- Aufbau ------------------------------------------------------------
    def env(self) -> dict[str, str]:
        root = Path(self.root)
        return dict(TB_LLM_GATEWAY_URL=self.llm.base,
                    NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
                    FASTMODEL=MODEL, COMPLEXMODEL=MODEL, BLITZMODEL=MODEL, SUMMARYMODEL=MODEL,
                    TB_DATA_DIR=self.root, HOME=str(root / "home"),
                    XDG_CONFIG_HOME=str(root / "cfg"), XDG_DATA_HOME=str(root / "share"),
                    XDG_CACHE_HOME=str(root / "cache"), XDG_STATE_HOME=str(root / "state"),
                    APPDATA=str(root / "appdata"), LOCALAPPDATA=str(root / "localappdata"),
                    ISAA_AUTO_PACK="0")

    @classmethod
    @contextmanager
    def llm_only(cls):
        """Nur Ersatz-LLM + Umgebung (fuer einen icli2-Kindprozess); liefert die Umgebungsvariablen."""
        b = cls()
        b.llm = FakeLLM().start()
        schutz = Schreibschutz()
        try:
            yield b.env()
        finally:
            b.llm.stop()
        schutz.pruefen()

    def __enter__(self) -> "Betrieb":
        self._schutz = Schreibschutz()
        self.llm = FakeLLM().start()
        self._stack.callback(self.llm.stop)
        self._stack.enter_context(patch.dict(os.environ, self.env()))
        self._stack.enter_context(_ohne_litellm())
        from toolboxv2.mods.isaa.base import local_embeddings
        self._stack.enter_context(patch.object(local_embeddings, "local_embed", _zero_embed))
        self.start_engine()
        return self

    def start_engine(self) -> None:
        from toolboxv2 import get_app
        from toolboxv2.mods.isaa.extras.icli2.backend import IsaaBackend
        from toolboxv2.mods.isaa.extras.icli2.engine import Engine
        self.app = get_app("isaa-host")         # darf noch nicht existieren (kein Import legt sie an)
        root = str(Path(self.root).resolve())
        for path in (self.app.appdata, self.app.data_dir):
            if not str(Path(path).resolve()).startswith(root):
                raise RuntimeError(f"Testumgebung nicht isoliert: {path} liegt nicht unter {root}")
        self.backend = IsaaBackend(self.app)
        self.engine = Engine(self.backend)
        self.engine.start()

    def stop_engine(self) -> None:
        from toolboxv2.mods.isaa.extras.icli2 import protocol as P
        if self.engine is not None:
            self.engine.send(P.SHUTDOWN)
            self.engine.join(30)
            self.engine = None

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.stop_engine()
        finally:
            self._stack.close()
        try:
            self._schutz.pruefen()
        except AssertionError as leak:
            if exc is None:
                raise
            exc.add_note(str(leak))                     # Testfehler bleibt sichtbar, Leck auch

    # -- Engine-Seite (ohne UI) ---------------------------------------------
    def pump(self) -> list:
        new = self.engine.poll(1000)
        self.events.extend(new)
        return new

    def wait_event(self, pred, timeout: float = 60.0, what: str = "Event"):
        end = time.monotonic() + timeout
        seen = 0
        while time.monotonic() < end:
            self.pump()
            for ev in self.events[seen:]:
                if pred(ev):
                    return ev
            seen = len(self.events)
            time.sleep(0.05)
        kinds = [(e.kind, str(e.data)[:120]) for e in self.events[-15:]]
        raise AssertionError(f"{what} kam nicht innerhalb {timeout}s; letzte Events: {kinds}")

    def ready(self) -> dict:
        return self.wait_event(lambda e: e.kind == "ready", 90, "READY").data

    def chat(self, agent: str, session: str, query: str, tag: str) -> dict:
        """Wie die UI: CHAT-Command; wartet auf das Ende genau dieses Laufs."""
        from toolboxv2.mods.isaa.extras.icli2 import protocol as P
        self.engine.send(P.CHAT, query=query, agent=agent, session=session, tag=tag)
        start = self.wait_event(lambda e: e.kind == P.RUN_START and e.data.get("tag") == tag, 60, f"RUN_START {tag}")
        end = self.wait_event(lambda e: e.kind == P.RUN_END and e.run == start.run, 90, f"RUN_END {tag}")
        return end.data

    def call(self, coro, timeout: float = 60.0):
        """Koroutine im Engine-Loop ausfuehren (dort leben isaa und die Agenten)."""
        return asyncio.run_coroutine_threadsafe(coro, self.engine._loop).result(timeout)

    # -- Zustand auf der Platte ---------------------------------------------
    def widgets_json(self) -> dict:
        path = Path(self.app.appdata) / "icli2" / "widgets.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def settings_json(self) -> dict:
        path = Path(self.app.appdata) / "icli2" / "settings.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def call_query(tool: str, **args) -> str:
    """Nachricht, auf die das Ersatz-LLM mit genau diesem Tool-Aufruf antwortet."""
    return f"CALL {tool} {json.dumps(args, ensure_ascii=False)}"
