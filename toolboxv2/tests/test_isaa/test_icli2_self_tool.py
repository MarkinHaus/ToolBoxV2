# -*- coding: utf-8 -*-
"""T7: Self-Tool - der `self`-Agent steuert icli2-Settings und andere Agenten."""
import json
from pathlib import Path

import pytest

from toolboxv2.mods.isaa.extras.icli2.backend import IsaaBackend
from toolboxv2.mods.isaa.extras.icli2.self_tool import SelfToolMixin


def _backend(tmp: Path, agent: str = "self", with_agent_json: bool = True) -> IsaaBackend:
    class _App:
        data_dir = str(tmp / "data")
        appdata = str(tmp / "appdata")

        class logger:
            @staticmethod
            def warning(*a, **k):
                pass

    class _Isaa:
        agent_data = {}

    class _B(SelfToolMixin, IsaaBackend):
        def __init__(self):
            self.app = _App()
            self.isaa = _Isaa()
            self.agent = agent
            self.session = "default"
            self.max_iterations = 5
            self._settings = {}

    b = _B()
    agents = Path(b.app.data_dir) / "Agents" / agent
    agents.mkdir(parents=True, exist_ok=True)
    if with_agent_json:
        (agents / "agent.json").write_text(
            json.dumps({"name": agent, "complex_llm_model": "keep-me"}), encoding="utf-8")
    return b


def test_1_guard_rejects_non_self(tmp_path):
    b = _backend(tmp_path, agent="worker")
    assert b.self_set_setting("max_iterations", "7") is None
    assert b.self_get_setting("max_iterations") is None
    assert b.self_get_agent_config("worker") is None
    assert b.self_set_agent_config("worker", {"fast_llm_model": "x"}) is None
    assert b.self_list_agents() is None


def test_2_set_get_roundtrip(tmp_path):
    b = _backend(tmp_path)
    res = b.self_set_setting("max_iterations", "7")
    assert res["text"]
    assert res["value"] == 7
    assert b.max_iterations == 7
    assert b.self_get_setting("max_iterations")["value"] == 7
    data = json.loads((Path(b.app.appdata) / "icli2" / "settings.json").read_text())
    assert data["max_iterations"] == 7


def test_3_setting_whitelist_violation(tmp_path):
    b = _backend(tmp_path)
    with pytest.raises(ValueError):
        b.self_set_setting("evil_key", "1")


def test_4_get_agent_config_missing_warns(tmp_path):
    b = _backend(tmp_path, with_agent_json=False)
    res = b.self_get_agent_config("self")
    assert res is not None
    assert res.get("level") == "warn"
    assert "agent.json" in res["text"] or "nicht gebaut" in res["text"]


def test_5_set_agent_config_only_whitelisted_keys(tmp_path):
    b = _backend(tmp_path)
    target = Path(b.app.data_dir) / "Agents" / "otherag"
    target.mkdir(parents=True, exist_ok=True)
    (target / "agent.json").write_text(
        json.dumps({"name": "otherag", "keep": "me"}), encoding="utf-8")
    with pytest.raises(ValueError):
        b.self_set_agent_config("otherag", {"fast_llm_model": "x", "evil_key": "boom"})
    b.self_set_agent_config("otherag", {"fast_llm_model": "groq/x"})
    data = json.loads((target / "agent.json").read_text())
    assert data["fast_llm_model"] == "groq/x"
    assert data["keep"] == "me"                       # Merge: andere Felder bleiben
    assert "evil_key" not in data                     # nur whitelisted Keys


def test_6_list_agents_skips_folders_without_agent_json(tmp_path):
    b = _backend(tmp_path, with_agent_json=False)
    base = Path(b.app.data_dir) / "Agents"
    for name, with_json in (("alpha", True), ("beta", True), ("gamma", False)):
        (base / name).mkdir(parents=True, exist_ok=True)
        if with_json:
            (base / name / "agent.json").write_text("{}", encoding="utf-8")
    res = b.self_list_agents()
    assert sorted(res["agents"]) == ["alpha", "beta"]
    assert res["count"] == 2
    assert "gamma" not in res["agents"]
