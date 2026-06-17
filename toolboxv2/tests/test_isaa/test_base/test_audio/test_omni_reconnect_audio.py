"""
Runnable checks for the reconnect/restart audio fixes (RC1 lost audio during
swap, RC2 stuck mic gate). unittest only.

The core guarantee: mic frames produced WHILE the backend is being swapped are
buffered and replayed to the NEW backend (no lost reopening speech), and the
half-duplex gate is reset so the mic feeds the new backend.

Run:
    python toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_reconnect_audio.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    root = here
    for _ in range(20):
        if (root / "toolboxv2" / "mods" / "isaa" / "base" / "audio_io" / "omni.py").exists():
            return root
        root = root.parent
    raise FileNotFoundError("repo root not found")


def _load_omni():
    p = _repo_root() / "toolboxv2" / "mods" / "isaa" / "base" / "audio_io" / "omni.py"
    spec = importlib.util.spec_from_file_location("omni_iso_recon", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["omni_iso_recon"] = m
    spec.loader.exec_module(m)
    return m


omni = _load_omni()


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class _FakeBackend:
    backend_name = "fake"
    supports_restart = True

    def __init__(self, connect_delay=0.0):
        self._connect_delay = connect_delay
        self._up = False
        self.received: list[bytes] = []
        self.started = 0
        self.stopped = 0
        self.system_instruction = ""

    async def start(self, tools=None):
        # mimic a WS connect that takes time; not 'up' until it completes
        if self._connect_delay:
            await asyncio.sleep(self._connect_delay)
        self._up = True
        self.started += 1

    async def stop(self):
        self._up = False
        self.stopped += 1

    async def send_audio(self, frame: bytes):
        if not self._up:
            return  # exactly like GeminiLiveBackend: drop while WS not connected
        self.received.append(frame)

    async def send_text(self, text):
        pass

    async def events(self):
        # never yields; just parks so _consume_events doesn't busy-loop
        while True:
            await asyncio.sleep(3600)


class _FakePlayer:
    async def flush(self):
        pass


def _bare_session(backend, factory):
    s = omni.OmniSession.__new__(omni.OmniSession)
    s.backend = backend
    s.backend_factory = factory
    s.player = _FakePlayer()
    s.sample_rate = 16000
    s._audio_buf = []
    s._pending_audio = []
    s._pending_audio_bytes = 0
    s._backend_swapping = False
    s._agent_speaking = False
    s._mic_muted_until = 999999.0   # stuck gate (RC2) -> must be reset by swap
    s._vad_hangover = 5
    s._event_task = None
    s._tool_specs = []
    s._restarting = False
    s._last_event_mono = 0.0
    s._update_phase = lambda: None
    s._build_seed_text = lambda: ""
    return s


class TestSwapBuffersAndReplays(unittest.TestCase):
    def test_audio_during_swap_is_replayed_to_new_backend(self):
        old = _FakeBackend()
        old._up = True
        new = _FakeBackend(connect_delay=0.05)
        s = _bare_session(old, factory=lambda: new)

        async def scenario():
            swap = asyncio.ensure_future(s._swap_backend(""))
            # while the new backend is still connecting, mic frames arrive
            await asyncio.sleep(0.01)
            self.assertTrue(s._backend_swapping)
            for i in range(3):
                if s._backend_swapping:
                    s._buffer_out_audio(bytes([i]) * 320)
            await swap
            return new

        nb = run(scenario())
        # the 3 frames buffered mid-swap must have reached the NEW backend
        self.assertEqual(len(nb.received), 3)
        self.assertEqual(s._pending_audio, [])
        # clean up the parked event task
        if s._event_task:
            s._event_task.cancel()

    def test_swap_resets_stuck_mic_gate(self):
        old = _FakeBackend(); old._up = True
        new = _FakeBackend()
        s = _bare_session(old, factory=lambda: new)
        run(s._swap_backend(""))
        # RC2: gate must be open again after a swap
        self.assertFalse(s._agent_speaking)
        self.assertEqual(s._mic_muted_until, 0.0)
        self.assertEqual(s._vad_hangover, 0)
        self.assertIs(s.backend, new)
        self.assertEqual(old.stopped, 1)
        self.assertEqual(new.started, 1)
        if s._event_task:
            s._event_task.cancel()


class TestBufferBounded(unittest.TestCase):
    def test_ring_drops_oldest_over_cap(self):
        s = omni.OmniSession.__new__(omni.OmniSession)
        s.sample_rate = 16000
        s._pending_audio = []
        s._pending_audio_bytes = 0
        cap = 16000 * 2 * 5
        frame = b"\x00" * 3200  # 0.1s
        for _ in range(int(cap / len(frame)) + 50):  # overshoot the cap
            s._buffer_out_audio(frame)
        self.assertLessEqual(s._pending_audio_bytes, cap)
        self.assertGreater(s._pending_audio_bytes, 0)


class TestPumpBuffersWhileSwapping(unittest.TestCase):
    def test_guardrail_pump_buffers_not_sends_during_swap(self):
        import inspect
        src = inspect.getsource(omni.OmniSession._pump_audio)
        # the pump must check the swap flag and buffer instead of send
        self.assertIn("_backend_swapping", src)
        self.assertIn("_buffer_out_audio", src)
        i_buf = src.find("_buffer_out_audio")
        i_send = src.find("self.backend.send_audio")
        self.assertLess(i_buf, i_send)  # buffer branch comes before the send


if __name__ == "__main__":
    unittest.main(verbosity=2)
