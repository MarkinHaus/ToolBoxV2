# -*- coding: utf-8 -*-
"""T4: Agent-Config ueber /set (nach dem Builder) - via _persist_config in agent.json."""
import json
from pathlib import Path

import pytest

from toolboxv2.mods.isaa.extras.icli2.backend import IsaaBackend


def _backend(tmp: Path, with_agent_json: bool = True) -> IsaaBackend:
    class _App:
        data_dir = str(tmp / "data")
        appdata = str(tmp / "appdata")

        class logger:
            @staticmethod
            def warning(*a, **k):
                pass

    class _Isaa:
        agent_data = {}

    class _B(IsaaBackend):
        def __init__(self):
            self.app = _App()
            self.isaa = _Isaa()
            self.agent = "testag"
            self.session = "default"
            self.max_iterations = 5
            self._settings = {}

    b = _B()
    agents = Path(b.app.data_dir) / "Agents" / "testag"
    agents.mkdir(parents=True, exist_ok=True)
    if with_agent_json:
        (agents / "agent.json").write_text(
            json.dumps({"name": "testag", "fast_llm_model": "a", "complex_llm_model": "b"}),
            encoding="utf-8")
    return b


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_set_model_fields(tmp_path):
    b = _backend(tmp_path)
    res = _run(b._do_set("fast_llm_model", "groq/x"))
    assert res["text"]
    data = json.loads((Path(b.app.data_dir) / "Agents" / "testag" / "agent.json").read_text())
    assert data["fast_llm_model"] == "groq/x"
    assert data["complex_llm_model"] == "b"          # Merge, kein Verlust


def test_set_offload_ratio(tmp_path):
    b = _backend(tmp_path)
    _run(b._do_set("immediate_offload_ratio", "0.7"))
    data = json.loads((Path(b.app.data_dir) / "Agents" / "testag" / "agent.json").read_text())
    assert data["immediate_offload_ratio"] == 0.7


def test_set_offload_ratio_validated(tmp_path):
    b = _backend(tmp_path)
    with pytest.raises(ValueError):
        _run(b._do_set("immediate_offload_ratio", "1.5"))
    with pytest.raises(ValueError):
        _run(b._do_set("immediate_offload_ratio", "abc"))


def test_set_heavy_hitter_tokens(tmp_path):
    b = _backend(tmp_path)
    _run(b._do_set("heavy_hitter_min_tokens", "2048"))
    data = json.loads((Path(b.app.data_dir) / "Agents" / "testag" / "agent.json").read_text())
    assert data["heavy_hitter_min_tokens"] == 2048


def test_set_unknown_key_still_rejected(tmp_path):
    b = _backend(tmp_path)
    with pytest.raises(ValueError):
        _run(b._do_set("unsinn_key", "1"))


def test_set_without_agent_json_hint(tmp_path):
    b = _backend(tmp_path, with_agent_json=False)
    res = _run(b._do_set("fast_llm_model", "groq/x"))
    assert "nicht gebaut" in res["text"] or "agent.json" in res["text"]
    assert res.get("level") == "warn"
