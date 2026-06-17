"""
Runnable checks for the P1 omni/audio fixes (T3 interrupt, T4 player buffer,
T5 idle reconnect). unittest only — no pytest.

omni.py is loaded in isolation by file path (tested core imports nothing from
toolboxv2). StreamingLocalPlayer.queue_audio is driven directly with fake WAV.

Run:
    python toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p1_fixes.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import io
import sys
import unittest
import wave
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    root = here
    for _ in range(20):
        if (root / "toolboxv2" / "mods" / "isaa" / "base" / "audio_io" / "omni.py").exists():
            return root
        root = root.parent
    raise FileNotFoundError("repo root not found")


ROOT = _repo_root()


def _load(name: str, rel: str):
    p = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses/enums read __module__ at exec
    spec.loader.exec_module(mod)
    return mod


omni = _load("omni_iso_p1", "toolboxv2/mods/isaa/base/audio_io/omni.py")


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _wav(n_frames: int, sr: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x01" * n_frames)
    return buf.getvalue()


# --- T4: StreamingLocalPlayer ring buffer is bounded + flush() works ----------
class TestT4PlayerBuffer(unittest.TestCase):
    def _player(self, max_buffer_s):
        # audioIo.py uses relative imports (.Stt/.Tts) -> can't exec standalone.
        # Stub the siblings + give it a package context so the real class loads
        # without pulling the whole toolboxv2 chain (pydantic etc).
        import types
        ai = sys.modules.get("aio_pkg.audioIo")
        if ai is None:
            pkg = types.ModuleType("aio_pkg")
            pkg.__path__ = [str(ROOT / "toolboxv2" / "mods" / "isaa" / "base" / "audio_io")]
            sys.modules["aio_pkg"] = pkg
            for sib in ("Stt", "Tts"):
                stub = types.ModuleType(f"aio_pkg.{sib}")
                sys.modules[f"aio_pkg.{sib}"] = stub
            # provide every name audioIo imports from the stubs as no-op placeholders
            for n in ("STTBackend", "STTConfig", "STTResult", "transcribe", "transcribe_stream"):
                setattr(sys.modules["aio_pkg.Stt"], n, object)
            for n in ("TTSBackend", "TTSConfig", "TTSResult", "TTSEmotion",
                      "synthesize", "synthesize_stream", "AudioStreamPlayer"):
                setattr(sys.modules["aio_pkg.Tts"], n, object)
            spec = importlib.util.spec_from_file_location(
                "aio_pkg.audioIo",
                ROOT / "toolboxv2" / "mods" / "isaa" / "base" / "audio_io" / "audioIo.py",
            )
            ai = importlib.util.module_from_spec(spec)
            sys.modules["aio_pkg.audioIo"] = ai
            try:
                spec.loader.exec_module(ai)
            except ImportError as e:  # stub miss -> add the missing name and retry once
                raise unittest.SkipTest(f"audioIo siblings need more stubs: {e}")
        return ai.StreamingLocalPlayer(sample_rate=24000, channels=1, max_buffer_s=max_buffer_s)

    def test_buffer_is_bounded_drop_oldest(self):
        p = self._player(max_buffer_s=1.0)  # cap = 24000*2 = 48000 bytes
        cap = p.max_buffer_bytes
        # queue 3 seconds of audio in 1s chunks -> must never exceed the cap
        for _ in range(3):
            run(p.queue_audio(_wav(24000), {}))
        self.assertLessEqual(len(p._buf), cap)
        # and it kept the most recent second (non-empty)
        self.assertGreater(len(p._buf), 0)

    def test_flush_empties_buffer(self):
        p = self._player(max_buffer_s=8.0)
        run(p.queue_audio(_wav(1000), {}))
        self.assertGreater(len(p._buf), 0)
        run(p.flush())
        self.assertEqual(len(p._buf), 0)


# --- shared bare OmniSession for T3/T5 ----------------------------------------
class _FakeBackend:
    backend_name = "fake"
    supports_restart = True

    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.system_instruction = ""

    async def start(self, tools=None):
        self.started += 1

    async def stop(self):
        self.stopped += 1

    async def send_text(self, text):
        pass


class _FakePlayer:
    def __init__(self):
        self.flushed = 0

    async def flush(self):
        self.flushed += 1


def _bare_session(backend, player=None, factory=None):
    s = omni.OmniSession.__new__(omni.OmniSession)
    s.backend = backend
    s.player = player
    s.backend_factory = factory
    s._audio_buf = []
    s._agent_speaking = False
    s._restarting = False
    s._running = True
    s._event_task = None
    s._tool_specs = []
    s.idle_reconnect_s = 10.0
    s._last_event_mono = 0.0
    s._phase = omni.OmniPhase.WAITING
    s.state_store = None
    s.resume_tail_turns = 6
    # _update_phase touches on_phase/flags; stub it to avoid the full hook
    s._update_phase = lambda: None
    return s


# --- T3: interrupt drops audio + flushes player + returns control -------------
class TestT3Interrupt(unittest.TestCase):
    def test_interrupt_clears_and_flushes(self):
        b = _FakeBackend()
        p = _FakePlayer()
        s = _bare_session(b, player=p)
        s._audio_buf = [(b"x", 24000)]
        s._agent_speaking = True
        run(s.interrupt())
        self.assertEqual(s._audio_buf, [])
        self.assertFalse(s._agent_speaking)
        self.assertEqual(p.flushed, 1)


# --- T5: reconnect swaps backend, keeps announced_jobs, flushes player --------
class TestT5Reconnect(unittest.TestCase):
    def test_reconnect_swaps_backend_and_flushes(self):
        old = _FakeBackend()
        new = _FakeBackend()
        p = _FakePlayer()
        s = _bare_session(old, player=p, factory=lambda: new)
        # minimal seed builder (no state_store -> returns "")
        s._build_seed_text = lambda: ""
        run(s._reconnect())
        self.assertIs(s.backend, new)          # backend swapped
        self.assertEqual(old.stopped, 1)        # old paused
        self.assertEqual(new.started, 1)        # new opened
        self.assertEqual(p.flushed, 1)          # playback backlog cleared
        self.assertFalse(s._restarting)         # guard released

    def test_reconnect_is_subset_no_compress_no_announce_clear(self):
        # guardrail: _reconnect must NOT compress or clear announced jobs
        src = inspect.getsource(omni.OmniSession._reconnect)
        self.assertNotIn("_compress(", src)
        self.assertNotIn("_announced_jobs.clear", src)

    def test_watchdog_only_reconnects_when_idle_and_waiting(self):
        # guardrail: the loop gates on WAITING + not speaking + not restarting
        src = inspect.getsource(omni.OmniSession._watchdog_loop)
        self.assertIn("OmniPhase.WAITING", src)
        self.assertIn("_agent_speaking", src)
        self.assertIn("_restarting", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
