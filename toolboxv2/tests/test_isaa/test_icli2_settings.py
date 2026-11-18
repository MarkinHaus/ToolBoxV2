# -*- coding: utf-8 -*-
"""T1 Settings-Persistenz: voice devices + TTS in <appdata>/icli2/settings.json."""
import json
import tempfile
from pathlib import Path

import pytest


def _make_backend(tmp: Path):
    """Backend-Stub mit gleicher Schnittstelle wie IsaaBackend fuer Mixin-Isolation."""
    from toolboxv2.mods.isaa.extras.icli2.backend import IsaaBackend

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
            self.agent = "self"
            self.session = "default"
            self.max_iterations = 5

    return _B()


@pytest.fixture()
def backend(tmp_path):
    return _make_backend(tmp_path)


def test_settings_roundtrip_devices(backend):
    """Geraetewahl ueberlebt Neustart (neuer _load_settings-Aufruf)."""
    backend.save_setting("voice_input_device", 3)
    backend.save_setting("voice_output_device", 1)
    assert json.loads((backend._settings_file()).read_text(encoding="utf-8"))[
        "voice_input_device"] == 3
    backend2 = _make_backend(Path(backend.app.appdata).parent)
    backend2._load_settings()
    assert backend2.get_voice_devices() == (3, 1)


def test_defaults_without_file(backend):
    """Ohne settings.json: Defaults (None, None) + tts groq/autumn."""
    backend._load_settings()
    assert backend.get_voice_devices() == (None, None)
    assert backend.get_tts_config() == ("groq_tts", "autumn")


def test_corrupt_json_falls_back(backend):
    """Korruptes JSON: keine Exception, Defaults, Hinweis verfuegbar."""
    backend._settings_file().parent.mkdir(parents=True, exist_ok=True)
    backend._settings_file().write_text("{broken", encoding="utf-8")
    backend._load_settings()
    assert backend.get_voice_devices() == (None, None)


def test_tts_settings(backend):
    backend.save_setting("tts_voice", "algiebra")
    backend._load_settings()
    assert backend.get_tts_config() == ("groq_tts", "algiebra")


def test_voice_mixin_applies_devices(backend):
    """VoiceMixin liest Geraete aus Backend-Settings beim Init."""
    backend.save_setting("voice_input_device", 7)
    backend.save_setting("voice_output_device", 2)
    backend._load_settings()
    backend._voice_init()
    assert backend._v_dev_in == 7
    assert backend._v_dev_out == 2


def test_do_voice_persists_choice(backend):
    """Nach der Wahl in /voice bleibt sie gespeichert."""
    backend._voice_init()
    res = asyncio_run(backend._do_voice("in:4"))
    assert res["text"]
    assert json.loads(backend._settings_file().read_text(encoding="utf-8"))["voice_input_device"] == 4


def asyncio_run(coro):
    import asyncio as _a
    return _a.get_event_loop_policy().new_event_loop().run_until_complete(coro)
