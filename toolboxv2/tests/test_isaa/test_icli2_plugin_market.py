# -*- coding: utf-8 -*-
"""T6: Marketplace-Eintraege mit relativem Pfad (./plugins/x) gegen Clone-Wurzel aufloesen."""
import json
from pathlib import Path

import pytest

from toolboxv2.mods.isaa.extras.icli2.plugin_install import _plugin_source


def test_relative_without_base_none():
    assert _plugin_source("./plugins/x") is None
    assert _plugin_source("../evil") is None


def test_relative_with_local_base(tmp_path):
    res = _plugin_source("./plugins/x", base=tmp_path)
    assert res == str((tmp_path / "plugins" / "x").resolve())


def test_relative_escape_blocked(tmp_path):
    assert _plugin_source("../evil", base=tmp_path) is None
    assert _plugin_source("./a/../../evil", base=tmp_path) is None


def test_dict_sources_unberuehrt():
    assert _plugin_source({"source": "github", "repo": "o/r"}) == "https://github.com/o/r"
    assert _plugin_source({"source": "git", "url": "https://x"}) == "https://x"
    assert _plugin_source(None) is None


def _make_market(root: Path, rel_source="./plugins/myplug"):
    plug_dir = root / "plugins" / "myplug"
    (plug_dir / ".claude-plugin").mkdir(parents=True)
    (plug_dir / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": "myplug"}), encoding="utf-8")
    (root / ".claude-plugin").mkdir()
    (root / ".claude-plugin" / "marketplace.json").write_text(
        json.dumps({"name": "localm", "plugins": [
            {"name": "myplug", "source": rel_source}]}), encoding="utf-8")
    return root


def _backend_with_market(tmp_path: Path):
    from toolboxv2.mods.isaa.extras.icli2.backend import IsaaBackend

    class _App:
        data_dir = str(tmp_path / "data")
        appdata = str(tmp_path / "appdata")

        class logger:
            @staticmethod
            def warning(*a, **k):
                pass

    class _B(IsaaBackend):
        def __init__(self):
            self.app = _App()
            self.agent = "self"
            self.session = "default"
            self.max_iterations = 5
            self._settings = {}
            self._markets = [str(_make_market(tmp_path / "mkt"))]

        def _markets_load(self):               # Testgrenze: statt Datei unsere Liste
            return list(self._markets)

    return _B()


def test_resolve_market_ref_relative(tmp_path):
    import asyncio
    b = _backend_with_market(tmp_path)
    src = asyncio.run(b._resolve_market_ref("myplug@localm"))
    assert src.endswith("plugins" + chr(92) + "myplug") or src.endswith("plugins/myplug")


def test_resolve_market_ref_remote_hint(tmp_path, monkeypatch):
    import asyncio
    from toolboxv2.mods.isaa.extras.icli2 import plugin_install as pi

    def fake_fetch(src):                           # Netzgrenze: Remote-Markt lokal gefaked
        return {"name": "otherm",
                "plugins": [{"name": "myplug", "source": "./plugins/myplug"}]}

    monkeypatch.setattr(pi, "_fetch_market", fake_fetch)
    b = _backend_with_market(tmp_path)
    b._markets = ["owner/repo"]                    # Pfad existiert nicht -> base=None
    with pytest.raises(ValueError) as e:
        asyncio.run(b._resolve_market_ref("myplug@otherm"))
    assert "lokal" in str(e.value) or "unterstuetzt" in str(e.value)
