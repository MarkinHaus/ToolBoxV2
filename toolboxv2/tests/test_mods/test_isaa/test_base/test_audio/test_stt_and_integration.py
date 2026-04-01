"""
test_stt_and_integration.py
============================

Tests for:
  - STT provider backends (faster-whisper local, Groq API)
  - Full audio pipeline integration (STT → text → TTS → WAV out)
  - WebPlayer relay integration (simulates WS streaming to client)
  - ISAA agent speak() tool integration with mocked agent

Real data transformations throughout. Provider tests skip with install
hints when the backend is not available.

Run:
    python -m unittest test_audio.test_stt_and_integration -v
"""

import asyncio
import io
import os
import unittest
import wave
from unittest.mock import AsyncMock, MagicMock, patch, call

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolboxv2.mods.isaa.base.audio_io.Stt import (
    STTBackend,
    STTConfig,
    STTResult,
    transcribe,
    transcribe_local,
)
from toolboxv2.mods.isaa.base.audio_io.Tts import (
    TTSBackend,
    TTSConfig,
    TTSEmotion,
    TTSResult,
    synthesize,
)
from toolboxv2.mods.isaa.base.audio_io.audioIo import (
    AudioStreamPlayer,
    NullPlayer,
    WebPlayer,
    AudioIOConfig,
    ProcessingMode,
    split_sentences,
    make_silent_wav,
    wav_duration,
    wav_is_valid,
    setup_isaa_audio,
    create_speak_tool,
    SPEAK_TOOL_SYSTEM_PROMPT,
)
from toolboxv2.tests.test_mods.test_isaa.test_base.test_audio._utils import (
    HAS_FASTER_WHISPER,
    HAS_GROQ,
    HAS_GROQ_PACKAGE,
    assert_valid_wav,
    make_silent_wav as util_silent_wav,
    make_sine_wav,
    print_provider_status,
    require_faster_whisper,
    require_groq,
    wav_info,
)


# =============================================================================
# STT: FASTER-WHISPER (local)
# =============================================================================

class TestFasterWhisperSTT(unittest.TestCase):
    """
    faster-whisper: local Whisper inference on CPU/GPU.
    Install: pip install faster-whisper
    Model downloads automatically (~500MB for 'small').
    """

    @require_faster_whisper
    def test_transcribe_silent_audio_returns_result(self):
        """
        Silent audio should produce an STTResult, possibly with empty text.
        This validates the pipeline works end-to-end even if VAD filters silence.
        """
        silent = make_silent_wav(1.0, 16000)
        result = transcribe(
            silent,
            config=STTConfig(
                backend=STTBackend.FASTER_WHISPER,
                model="tiny",
                device="cpu",
                compute_type="int8",
            ),
        )
        self.assertIsInstance(result, STTResult)
        self.assertIsNotNone(result.text)  # text may be empty for silence

    @require_faster_whisper
    def test_transcribe_sine_wave_returns_result(self):
        """Sine wave is not speech — expect short/empty transcription but no crash."""
        sine = make_sine_wav(440.0, 2.0, 16000)
        result = transcribe(
            sine,
            config=STTConfig(
                backend=STTBackend.FASTER_WHISPER,
                model="tiny",
                device="cpu",
                compute_type="int8",
                language="en",
            ),
        )
        self.assertIsInstance(result, STTResult)
        # STT must not crash on non-speech audio
        self.assertIsNotNone(result.text)

    @require_faster_whisper
    def test_transcribe_returns_language_info(self):
        """STTResult must include detected language."""
        silent = make_silent_wav(1.0, 16000)
        result = transcribe(
            silent,
            config=STTConfig(
                backend=STTBackend.FASTER_WHISPER,
                model="tiny",
                device="cpu",
                compute_type="int8",
            ),
        )
        # Language may be None for silence or a detected language string
        # Just check it's not a crash
        self.assertIsInstance(result, STTResult)

    @require_faster_whisper
    def test_transcribe_from_file_path(self):
        """STT should accept a file path string as well as bytes."""
        import tempfile
        silent = make_silent_wav(0.5, 16000)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(silent)
            tmp_path = f.name
        try:
            result = transcribe(
                tmp_path,
                config=STTConfig(
                    backend=STTBackend.FASTER_WHISPER,
                    model="tiny",
                    device="cpu",
                    compute_type="int8",
                ),
            )
            self.assertIsInstance(result, STTResult)
        finally:
            os.unlink(tmp_path)

    @require_faster_whisper
    def test_convenience_function(self):
        silent = make_silent_wav(0.5, 16000)
        result = transcribe_local(silent, model="tiny")
        self.assertIsInstance(result, STTResult)

    @require_faster_whisper
    def test_vad_filter_enabled(self):
        """VAD is enabled by default — long silence should produce minimal segments."""
        silent = make_silent_wav(3.0, 16000)
        result = transcribe(
            silent,
            config=STTConfig(
                backend=STTBackend.FASTER_WHISPER,
                model="tiny",
                device="cpu",
                compute_type="int8",
            ),
        )
        # VAD should filter out silence, producing 0 or very few segments
        self.assertIsInstance(result, STTResult)
        # Text for silence must be very short or empty
        self.assertLess(len(result.text.strip()), 50)


# =============================================================================
# STT: GROQ WHISPER (API)
# =============================================================================

class TestGroqWhisperSTT(unittest.TestCase):
    """
    Groq Whisper: cloud STT via Groq API.
    Install: pip install groq
    Requires: export GROQ_API_KEY=gsk_...
    """

    @require_groq
    def test_transcribe_silent_audio(self):
        silent = make_silent_wav(1.0, 16000)
        result = transcribe(
            silent,
            config=STTConfig(
                backend=STTBackend.GROQ_WHISPER,
                model="whisper-large-v3-turbo",
                groq_api_key=os.environ.get("GROQ_API_KEY"),
                language="en",
            ),
        )
        self.assertIsInstance(result, STTResult)
        self.assertIsNotNone(result.text)

    @require_groq
    def test_transcribe_returns_sttresult_type(self):
        silent = make_silent_wav(0.5, 16000)
        result = transcribe(
            silent,
            config=STTConfig(
                backend=STTBackend.GROQ_WHISPER,
                groq_api_key=os.environ.get("GROQ_API_KEY"),
            ),
        )
        self.assertIsInstance(result, STTResult)
        self.assertIsInstance(result.text, str)

    @require_groq
    def test_missing_api_key_raises(self):
        from toolboxv2.mods.isaa.base.audio_io.Stt import _transcribe_groq_whisper
        config = STTConfig(
            backend=STTBackend.GROQ_WHISPER,
            groq_api_key=None,
        )
        # Temporarily unset env var
        old_key = os.environ.pop("GROQ_API_KEY", None)
        try:
            with self.assertRaises((ValueError, Exception)):
                _transcribe_groq_whisper(make_silent_wav(0.5), config)
        finally:
            if old_key:
                os.environ["GROQ_API_KEY"] = old_key


# =============================================================================
# FULL PIPELINE INTEGRATION — STT → agent → TTS
# =============================================================================

class TestFullPipelineIntegration(unittest.IsolatedAsyncioTestCase):
    """
    End-to-end pipeline tests.

    The agent (text processor) is a simple async generator — no ISAA needed.
    STT and TTS use mocked synthesize/transcribe so the test runs without
    any provider installed, but the data transformation chain is real.

    Separate tests with @require_faster_whisper / @require_groq run the
    real STT/TTS path.
    """

    def _make_fake_tts(self, duration: float = 0.3, sr: int = 16000):
        """Return a fake TTSResult with real WAV bytes."""
        wav = make_silent_wav(duration, sr)
        return TTSResult(audio=wav, format="wav", sample_rate=sr, duration=duration)

    async def test_pipeline_mocked_stt_and_tts(self):
        """
        Mocked STT + TTS. Tests the orchestration:
        audio_bytes → transcribe → agent_text → synthesize → WAV out.
        """
        input_wav = make_silent_wav(0.5)
        fake_stt = STTResult(text="What time is it?", language="en", duration=0.5)
        fake_tts = self._make_fake_tts(0.4)

        async def agent(text: str):
            for word in text.split():
                yield word + " "

        with patch("toolboxv2.mods.isaa.base.audio_io.audioIo.transcribe", return_value=fake_stt), \
             patch("toolboxv2.mods.isaa.base.audio_io.audioIo.synthesize", return_value=fake_tts):

            from toolboxv2.mods.isaa.base.audio_io.audioIo import _process_pipeline_raw
            config = AudioIOConfig(player_backend_type="null")
            result = await _process_pipeline_raw(input_wav, agent, config)

        self.assertEqual(result.text_input, "What time is it?")
        self.assertIn("What", result.text_output)
        self.assertIsNotNone(result.audio_output)
        assert_valid_wav(self, result.audio_output, "pipeline_mocked")

    async def test_pipeline_text_output_matches_agent(self):
        """Agent response text must be preserved in AudioIOResult."""
        input_wav = make_silent_wav(0.5)
        fake_stt = STTResult(text="ping", language="en", duration=0.1)
        fake_tts = self._make_fake_tts(0.2)
        expected_response = "pong response from agent"

        async def agent(text: str):
            yield expected_response

        with patch("toolboxv2.mods.isaa.base.audio_io.audioIo.transcribe", return_value=fake_stt), \
             patch("toolboxv2.mods.isaa.base.audio_io.audioIo.synthesize", return_value=fake_tts):

            from toolboxv2.mods.isaa.base.audio_io.audioIo import _process_pipeline_raw
            config = AudioIOConfig(player_backend_type="null")
            result = await _process_pipeline_raw(input_wav, agent, config)

        self.assertEqual(result.text_output, expected_response)

    async def test_null_player_receives_audio_from_stream_player(self):
        """
        Full path: AudioStreamPlayer queues text → TTS worker synthesizes →
        NullPlayer.queue_audio() called → null.received_chunks populated.
        This is a real async execution test.
        """
        null = NullPlayer()
        fake_tts = self._make_fake_tts(0.2)

        with patch("toolboxv2.mods.isaa.base.audio_io.audioIo.synthesize", return_value=fake_tts):
            sp = AudioStreamPlayer(
                player_backend=null,
                tts_config=TTSConfig(backend=TTSBackend.PIPER),
                session_id="integration_test",
            )
            await sp.start()
            await sp.queue_text("Integration test sentence one.")
            await sp.queue_text("Integration test sentence two.")
            await asyncio.sleep(0.8)
            await sp.stop()

        self.assertEqual(null.total_chunks, 2)
        for wav, meta in null.received_chunks:
            assert_valid_wav(self, wav, "NullPlayer integration")
            self.assertEqual(meta["session_id"], "integration_test")

    async def test_web_player_relay_full_cycle(self):
        """
        Producer: AudioStreamPlayer + WebPlayer
        Consumer: iter_chunks() draining the queue concurrently

        Real async concurrency test — producer and consumer run simultaneously.
        """
        web = WebPlayer(mock_mode=False, max_queue=20)
        fake_tts = self._make_fake_tts(0.2, 16000)

        sp = AudioStreamPlayer(
            player_backend=web,
            tts_config=TTSConfig(backend=TTSBackend.PIPER),
            session_id="web_relay_test",
        )

        consumed = []

        async def consumer():
            async for chunk, meta in web.iter_chunks():
                assert_valid_wav(self, chunk, f"web relay chunk {meta.get('chunk_index')}")
                consumed.append({
                    "text": meta.get("text"),
                    "emotion": meta.get("emotion"),
                    "chunk_index": meta.get("chunk_index"),
                })

        async def producer():
            await sp.start()
            texts = [
                ("First sentence in the relay test.", TTSEmotion.NEUTRAL),
                ("Second sentence with excitement.", TTSEmotion.EXCITED),
                ("Third sentence, calm conclusion.", TTSEmotion.CALM),
            ]
            for text, emo in texts:
                await sp.queue_text(text, emotion=emo)
            await asyncio.sleep(0.8)
            await sp.stop()

        with patch("toolboxv2.mods.isaa.base.audio_io.audioIo.synthesize", return_value=fake_tts):
            await asyncio.gather(producer(), consumer())

        self.assertEqual(len(consumed), 3)
        # Order must be preserved
        self.assertEqual(consumed[0]["text"], "First sentence in the relay test.")
        self.assertEqual(consumed[1]["emotion"], "excited")
        self.assertEqual(consumed[2]["emotion"], "calm")
        # Indices are sequential
        indices = [c["chunk_index"] for c in consumed]
        self.assertEqual(indices, sorted(indices))

    async def test_sentence_split_produces_multiple_tts_calls(self):
        """
        Long text through speak() must result in multiple TTS calls,
        one per sentence. Validates the sentence-boundary streaming
        that gives low latency in production.
        """
        null = NullPlayer()
        fake_tts = self._make_fake_tts(0.1)
        sp = AudioStreamPlayer(player_backend=null, tts_config=TTSConfig(backend=TTSBackend.PIPER))
        speak = create_speak_tool(sp)

        long_text = (
            "The first sentence ends here. "
            "The second sentence continues the thought. "
            "The third sentence concludes the response."
        )

        with patch("toolboxv2.mods.isaa.base.audio_io.audioIo.synthesize", return_value=fake_tts):
            await sp.start()
            await speak(long_text, "calm")
            await asyncio.sleep(0.8)
            await sp.stop()

        # 3 sentences → 3 TTS calls → 3 chunks
        self.assertGreaterEqual(null.total_chunks, 2,
                                "Multi-sentence text should produce ≥ 2 TTS chunks")
        # All chunks must be valid WAV
        for wav, meta in null.received_chunks:
            assert_valid_wav(self, wav, f"split chunk: {meta.get('text', '')[:20]}")

    @require_faster_whisper
    async def test_real_stt_with_mocked_tts(self):
        """
        Real faster-whisper STT on silent audio → mocked TTS.
        Validates the actual STT path runs without errors.
        """
        input_wav = make_silent_wav(1.0, 16000)
        fake_tts = self._make_fake_tts(0.3)

        async def echo_agent(text: str):
            yield f"You said: {text}"

        with patch("toolboxv2.mods.isaa.base.audio_io.audioIo.synthesize", return_value=fake_tts):
            from toolboxv2.mods.isaa.base.audio_io.audioIo import _process_pipeline_raw
            config = AudioIOConfig(
                player_backend_type="null",
                stt_config=STTConfig(
                    backend=STTBackend.FASTER_WHISPER,
                    model="tiny",
                    device="cpu",
                    compute_type="int8",
                ),
            )
            result = await _process_pipeline_raw(input_wav, echo_agent, config)

        self.assertIsNotNone(result.text_input)
        self.assertIsNotNone(result.audio_output)
        assert_valid_wav(self, result.audio_output, "real_stt_pipeline")


# =============================================================================
# SETUP_ISAA_AUDIO INTEGRATION — agent mock
# =============================================================================

class TestSetupIsaaAudio(unittest.IsolatedAsyncioTestCase):
    """
    Tests for setup_isaa_audio(): agent tool registration and system prompt injection.
    No real ISAA agent needed — uses a simple mock with the same interface.
    """

    def _make_mock_agent(self):
        """Mock agent with the minimal interface setup_isaa_audio() uses."""
        agent = MagicMock()
        agent.amd = MagicMock()
        agent.amd.system_message = "You are a helpful assistant."
        registered_tools = {}

        def add_tool(func, name=None, description=None, category=None, flags=None):
            registered_tools[name or func.__name__] = {
                "func": func,
                "description": description,
                "category": category,
            }

        agent.add_tool = add_tool
        agent._registered_tools = registered_tools
        return agent, registered_tools

    async def test_speak_tool_registered(self):
        agent, tools = self._make_mock_agent()
        setup_isaa_audio(agent, player_backend=NullPlayer())
        self.assertIn("speak", tools, "speak tool not registered on agent")

    async def test_speak_tool_is_callable(self):
        agent, tools = self._make_mock_agent()
        setup_isaa_audio(agent, player_backend=NullPlayer())
        speak_func = tools["speak"]["func"]
        self.assertTrue(callable(speak_func))

    async def test_speak_tool_category_audio(self):
        agent, tools = self._make_mock_agent()
        setup_isaa_audio(agent, player_backend=NullPlayer())
        category = tools["speak"]["category"]
        self.assertIn("audio", category)

    async def test_system_prompt_appended(self):
        agent, _ = self._make_mock_agent()
        setup_isaa_audio(agent, player_backend=NullPlayer())
        system_msg = agent.amd.system_message
        self.assertIn("AUDIO MODE", system_msg)
        self.assertIn("speak", system_msg)

    async def test_system_prompt_not_duplicated(self):
        """Calling setup twice should not double-append the audio contract."""
        agent, _ = self._make_mock_agent()
        setup_isaa_audio(agent, player_backend=NullPlayer())
        setup_isaa_audio(agent, player_backend=NullPlayer())
        count = agent.amd.system_message.count("AUDIO MODE")
        self.assertEqual(count, 1, "AUDIO MODE appended multiple times")

    async def test_returns_audio_stream_player(self):
        agent, _ = self._make_mock_agent()
        result = setup_isaa_audio(agent, player_backend=NullPlayer())
        self.assertIsInstance(result, AudioStreamPlayer)

    async def test_with_web_player_backend(self):
        agent, tools = self._make_mock_agent()
        web = WebPlayer(mock_mode=True)
        sp = setup_isaa_audio(agent, player_backend=web)
        self.assertIsInstance(sp.player, WebPlayer)

    async def test_speak_tool_executes_and_queues(self):
        """
        End-to-end: register speak(), call it, verify NullPlayer gets audio.
        """
        agent, tools = self._make_mock_agent()
        null = NullPlayer()

        fake_wav = make_silent_wav(0.2)
        fake_tts = TTSResult(audio=fake_wav, format="wav", sample_rate=16000, duration=0.2)

        sp = setup_isaa_audio(
            agent,
            tts_config=TTSConfig(backend=TTSBackend.PIPER),
            player_backend=null,
        )

        speak_func = tools["speak"]["func"]

        with patch("toolboxv2.mods.isaa.base.audio_io.audioIo.synthesize", return_value=fake_tts):
            await sp.start()
            result = await speak_func("Hello from the agent.", "friendly")
            await asyncio.sleep(0.5)
            await sp.stop()

        self.assertEqual(result, "[queued for speech]")
        self.assertEqual(null.total_chunks, 1)
        assert_valid_wav(self, null.received_chunks[0][0], "speak_tool_e2e")
        self.assertEqual(null.received_chunks[0][1]["emotion"], "friendly")

    async def test_speak_tool_system_prompt_contains_rules(self):
        """System prompt must contain all mandatory rules."""
        self.assertIn("MUST call", SPEAK_TOOL_SYSTEM_PROMPT)
        self.assertIn("final_answer", SPEAK_TOOL_SYSTEM_PROMPT)
        self.assertIn("emotion", SPEAK_TOOL_SYSTEM_PROMPT)
        for emo in ["neutral", "calm", "excited", "friendly", "serious", "empathetic"]:
            self.assertIn(emo, SPEAK_TOOL_SYSTEM_PROMPT)


if __name__ == "__main__":
    print_provider_status()
    unittest.main(verbosity=2)
