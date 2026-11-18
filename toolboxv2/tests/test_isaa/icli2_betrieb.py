# -*- coding: utf-8 -*-
"""Betriebsumgebung fuer icli2-Tests: echte ToolBox-App, echter isaa-Mod, echte Engine + IsaaBackend,
LLM ueber das Ersatz-LLM (HTTP, Prozessgrenze). Alle Daten in einem Temp-Ordner (Regel 6).

Einziges weiteres Double: `local_embeddings.local_embed` liefert Nullen. Grund: das echte Embedding-Modell
wird beim ersten Gebrauch aus dem Netz geladen (HuggingFace) - Netzgrenze, siehe icli2/tests/e2e/e2e_bg.py.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from icli2_fake_llm import FakeLLM

TIMEOUT = 120


async def _zero_embed(texts, dimensions=None):
    import numpy as np
    return np.zeros((len(texts), dimensions or 768), dtype="float32")


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
        return dict(OPENAI_API_BASE=self.llm.base, OPENAI_API_KEY="x",
                    NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
                    FASTMODEL="openai/fake", COMPLEXMODEL="openai/fake", BLITZMODEL="openai/fake",
                    SUMMARYMODEL="openai/fake", TB_DATA_DIR=self.root,
                    XDG_CONFIG_HOME=str(Path(self.root) / "cfg"), ISAA_AUTO_PACK="0")

    @classmethod
    @contextmanager
    def llm_only(cls):
        """Nur Ersatz-LLM + Umgebung (fuer einen icli2-Kindprozess); liefert die Umgebungsvariablen."""
        b = cls()
        b.llm = FakeLLM().start()
        try:
            yield b.env()
        finally:
            b.llm.stop()

    def __enter__(self) -> "Betrieb":
        self.llm = FakeLLM().start()
        self._stack.callback(self.llm.stop)
        self._stack.enter_context(patch.dict(os.environ, self.env()))
        from toolboxv2.mods.isaa.base import local_embeddings
        self._stack.enter_context(patch.object(local_embeddings, "local_embed", _zero_embed))
        self.start_engine()
        return self

    def start_engine(self) -> None:
        from toolboxv2 import get_app
        from toolboxv2.mods.isaa.extras.icli2.backend import IsaaBackend
        from toolboxv2.mods.isaa.extras.icli2.engine import Engine
        from toolboxv2.tests.a_util import reset_app_singleton
        # Eine schon vorhandene App (z. B. beim Import von icli2/app.py angelegt) zeigt aufs echte
        # Profil: verwerfen, damit die App mit der umgebogenen Umgebung entsteht (Regel 6).
        reset_app_singleton(force_exit=True)
        self.app = get_app("isaa-host")
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

    def __exit__(self, *exc) -> None:
        try:
            self.stop_engine()
        finally:
            self._stack.close()

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
