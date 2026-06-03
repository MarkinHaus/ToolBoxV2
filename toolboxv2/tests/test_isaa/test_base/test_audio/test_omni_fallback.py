import asyncio
import unittest

from toolboxv2.mods.isaa.base.audio_io.omni import (
    OmniSession, FallbackOmniBackend, OmniEventType, pcm16_to_wav,
)
from toolboxv2.mods.isaa.base.audio_io.audioIo import NullPlayer


class FakeAudioAgent:
    """Mirrors FlowAgent's a_audio / a_run / tts contract deterministically.
    a_audio -> (audio_output WAV, text_output, tool_calls, metadata)."""
    def __init__(self):
        self.calls = []

    async def a_audio(self, wav, session_id="default", **kw):
        self.calls.append(("a_audio", session_id))
        audio = pcm16_to_wav(b"\x01\x00" * 800, 24000)   # 50ms @24k mono
        return audio, "Hallo, hier ist die Antwort.", [], {"mode": "fallback"}

    async def a_run(self, query, session_id="default", **kw):
        return f"answer: {query}"

    async def tts(self, text, **kw):
        class _R:
            audio = pcm16_to_wav(b"\x02\x00" * 800, 24000)
        return _R()


class _FakeEngine:
    """Stand-in for LiveModeEngine: no VAD/recorder, test triggers utterances."""
    def __init__(self, on_utterance):
        self.on_utterance = on_utterance
        self.started = False
        self.fed = []
    async def start(self): self.started = True
    async def stop(self): self.started = False
    async def feed(self, pcm): self.fed.append(pcm)
    async def utter(self, wav, speaker=None): await self.on_utterance(wav, speaker)


class TestFallbackBackend(unittest.IsolatedAsyncioTestCase):
    def _make(self):
        holder = {}
        backend = FallbackOmniBackend(
            FakeAudioAgent(),
            engine_factory=lambda on_utt: holder.setdefault("eng", _FakeEngine(on_utt)),
        )
        return backend, holder

    async def test_emits_text_audio_turnend(self):
        backend, holder = self._make()
        await backend.start(tools=[])
        await holder["eng"].utter(pcm16_to_wav(b"\x03\x00" * 800, 16000))
        await backend.stop()  # sentinel -> events() terminates

        kinds = [ev.type async for ev in backend.events()]
        self.assertIn(OmniEventType.TEXT, kinds)
        self.assertIn(OmniEventType.AUDIO, kinds)
        self.assertIn(OmniEventType.TURN_END, kinds)

    async def test_runs_through_session_like_cloud(self):
        backend, holder = self._make()
        player = NullPlayer()
        texts = []
        session = OmniSession(backend, player=player,
                              on_text=lambda t: texts.append(t))
        await session.start(tool_specs=[])
        await holder["eng"].utter(pcm16_to_wav(b"\x04\x00" * 800, 16000))
        await asyncio.sleep(0.05)        # let _consume_events drain into the player
        await session.stop()

        self.assertTrue(player.received_chunks, "agent audio reached the player (like cloud)")
        self.assertIn("Hallo, hier ist die Antwort.", "".join(texts))


if __name__ == "__main__":
    unittest.main()
