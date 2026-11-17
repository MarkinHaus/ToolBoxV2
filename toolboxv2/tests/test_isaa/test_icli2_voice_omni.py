# -*- coding: utf-8 -*-
"""T8 Omni-Adapter-Integration: /voice omni nutzt die bestehenden omni.py-Klassen
(OmniSession/VoiceModeConfig) mit den persistierten Geraete-Settings aus T1.

Alle Audio-/Netz-Kollaborateure werden per sys.modules-Fake ersetzt (lazy Imports
in voice.py machen das moeglich) – kein sounddevice, kein Netz, kein Modell.
"""
import asyncio
import sys
import types

import pytest


# --------------------------- Fakes -----------------------------------------

class FakeOmniSession:
    """Capture-Stub: merkt Konstruktor-Argumente, zaehlt start/stop."""
    instances = []

    def __init__(self, backend, recorder=None, player=None, tools=None, **kw):
        self.backend, self.recorder, self.player, self.tools = backend, recorder, player, tools
        self.kw = kw
        self.is_active = False
        self.started = None
        self.stopped = False
        FakeOmniSession.instances.append(self)

    async def start(self, tool_specs=None):
        self.started = tool_specs

    async def stop(self):
        self.stopped = True


class FakeBackend:
    backend_name = "StubOmniBackend"

    def __init__(self):
        self.system_instruction = None

    async def start(self):
        pass

    async def stop(self):
        pass

    async def send_audio(self, chunk):
        pass

    async def send_tool_result(self, call_id, result):
        pass

    def events(self):
        import asyncio as _a
        q = _a.Queue()
        return q


class FakeRecorder:
    instances = []

    def __init__(self, device=None):
        self.device = device
        self.started = False
        FakeRecorder.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def frames(self):
        return iter(())


class FakePlayer:
    instances = []

    def __init__(self, device=None, sample_rate=None, **kw):
        self.device = device
        self.sample_rate = sample_rate
        self.started = False
        FakePlayer.instances.append(self)

    async def start(self):
        self.started = True

    async def stop(self):
        self.started = False

    async def queue_audio(self, chunk):
        pass


def _install_omni_fakes(monkeypatch, backend=None):
    """Ersetzt die 3 lazy-importierten Module in sys.modules durch Fakes."""
    omni = types.ModuleType("toolboxv2.mods.isaa.base.audio_io.omni")
    omni.OMNI_SYSTEM_INSTRUCTION = "OMNI-SI"
    omni.OmniSession = FakeOmniSession
    omni.OmniBackend = object
    omni.OmniEvent = object
    omni.OmniEventType = object

    class FakeVoiceModeConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def build_backend(self):
            return backend if backend is not None else FakeBackend()

    omni.VoiceModeConfig = FakeVoiceModeConfig
    rec = types.ModuleType("toolboxv2.mods.isaa.base.audio_io.audio_recorder")
    rec.LocalMicRecorder = FakeRecorder
    io = types.ModuleType("toolboxv2.mods.isaa.base.audio_io.audioIo")
    io.LocalPlayer = FakePlayer
    io.StreamingLocalPlayer = FakePlayer
    for name, mod in (("toolboxv2.mods.isaa.base.audio_io.omni", omni),
                      ("toolboxv2.mods.isaa.base.audio_io.audio_recorder", rec),
                      ("toolboxv2.mods.isaa.base.audio_io.audioIo", io)):
        monkeypatch.setitem(sys.modules, name, mod)
    return omni


class FakeAgent:
    tool_manager = None


# --------------------------- Backend-Stub (Vorlage: test_icli2_settings) ----

def _make_backend(tmp):
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

        async def _agent(self):
            return FakeAgent()

    return _B()


@pytest.fixture()
def backend(tmp_path, monkeypatch):
    FakeOmniSession.instances.clear()
    FakeRecorder.instances.clear()
    FakePlayer.instances.clear()
    _install_omni_fakes(monkeypatch)
    return _make_backend(tmp_path)


def asyncio_run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


# --------------------------- Tests -----------------------------------------

def test_voice_module_binds_omni_classes():
    """(a) voice.py importiert OmniSession+VoiceModeConfig aus dem bestehenden omni.py-Adapter."""
    import toolboxv2.mods.isaa.extras.icli2.voice as voice_mod
    src = open(voice_mod.__file__, encoding="utf-8").read()
    assert "from toolboxv2.mods.isaa.base.audio_io.omni import" in src
    assert "OmniSession" in src and "VoiceModeConfig" in src


def test_omni_toggle_starts_session_with_persisted_devices(backend):
    """(b+d) Start: OmniSession (omni.py) bekommt Recorder/Player mit den
    persistierten Geraete-Indizes; Session ist aktiv und gemerkt."""
    backend.save_setting("voice_input_device", 7)
    backend.save_setting("voice_output_device", 2)
    backend._load_settings()
    backend._voice_init()
    res = asyncio_run(backend._do_voice("omni"))
    assert res["text"]
    assert backend._v_omni is not None
    sess = FakeOmniSession.instances[-1]
    assert sess.is_active is True
    assert sess.started == []            # keine Tool-Specs (FakeAgent ohne Manager)
    assert FakeRecorder.instances[-1].device == 7
    assert FakePlayer.instances[-1].device == 2


def test_omni_toggle_stop_closes_session(backend):
    """Zweiter Toggle beendet: is_active False, stop() gefahren, Referenz frei."""
    asyncio_run(backend._do_voice("omni"))
    sess = backend._v_omni
    res = asyncio_run(backend._do_voice("omni"))
    assert res["text"]
    assert sess.is_active is False
    assert sess.stopped is True
    assert backend._v_omni is None


def test_omni_toggle_rejects_while_recording(backend):
    """Micro-Aufnahme (F4-Pfad) hat Vorrang: Omni-Start wird abgewiesen."""
    backend._voice_init()
    backend._v_recording = True
    with pytest.raises(RuntimeError):
        asyncio_run(backend._do_voice("omni"))
    assert backend._v_omni is None
    assert FakeOmniSession.instances == []


def test_omni_guard_without_backend(backend, monkeypatch):
    """build_backend()->None (z.B. 'pipeline' ohne Agent) -> klare Fehlermeldung."""
    omni = _install_omni_fakes(monkeypatch)
    monkeypatch.setattr(omni.VoiceModeConfig, "build_backend",
                        lambda self: None, raising=False)
    backend._voice_init()
    with pytest.raises(RuntimeError):
        asyncio_run(backend._do_voice("omni"))
    assert backend._v_omni is None


def test_voice_close_stops_omni_session(backend):
    """_voice_close raeumt auch die Omni-Session ab (stop + None)."""
    asyncio_run(backend._do_voice("omni"))
    sess = backend._v_omni
    asyncio_run(backend._voice_close())
    assert sess.stopped is True
    assert backend._v_omni is None
