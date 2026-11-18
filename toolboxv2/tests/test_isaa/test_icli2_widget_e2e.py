# -*- coding: utf-8 -*-
"""Welle 7 E2E: 2 Agenten, Routing, Persistenz-Reload, Board-Schliessen."""
from pathlib import Path

from toolboxv2.mods.isaa.extras.icli2.backend import IsaaBackend


def _backend(tmp: Path, agent: str) -> object:
    class _App:
        data_dir = str(tmp / "data")
        appdata = str(tmp / "appdata")

        class logger:
            @staticmethod
            def warning(*a, **k):
                pass

    class _B(IsaaBackend):
        def __init__(self):
            self.app = _App()
            self.agent = agent
            self.session = "default"
            self.max_iterations = 5
            self._settings = {}

    return _B()


SPEC_A = {"title": "Alpha-Panel", "template": "info", "text": "A"}
SPEC_B = {"title": "Beta-Board", "template": "choice", "options": ["x", "y"], "mode": "pinned"}


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_e2e_two_agents_routing_persistence(tmp_path):
    tmp = Path(tmp_path)
    alpha = _backend(tmp, "alpha")
    beta = _backend(tmp, "beta")

    ida = _run(alpha.widget_show(SPEC_A))["id"]
    idb = _run(beta.widget_show(SPEC_B))["id"]
    assert ida.startswith("alpha-") and idb.startswith("beta-")

    reg = alpha.widget_registry_load()
    assert reg[ida]["creator"] == ["alpha", "default"]
    assert reg[idb]["creator"] == ["beta", "default"]

    got = {}
    async def fake_start(prompt):
        got["alpha"] = prompt
    alpha._widget_start_turn = fake_start
    res = _run(alpha._do_widget_answer({"id": ida, "values": {"text": "ok"}}))
    assert res["ok"]
    assert "Alpha-Panel" in got["alpha"]

    beta2 = _backend(tmp, "beta")
    reg2 = beta2.widget_registry_load()
    assert ida in reg2 and idb in reg2

    assert not _run(beta2.widget_close(ida))["ok"]
    assert _run(beta.widget_close(idb))["ok"]
    reg3 = _backend(tmp, "alpha").widget_registry_load()
    assert reg3[idb]["closed"] is True
