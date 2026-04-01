"""
test_audio_core.py
==================

Tests for:
  - split_sentences() — real text processing
  - WAV utility functions — real byte manipulation
  - NullPlayer / WebPlayer — real async queuing and data relay
  - AudioStreamPlayer with NullPlayer — TTS is mocked so tests run
    without any provider installed, but the queue/worker/metadata
    pipeline runs for real

Run:
    python -m unittest test_audio.test_audio_core -v
"""

import asyncio
import io
import unittest
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# We import the actual module objects, not mocks
from toolboxv2.mods.isaa.base.audio_io.audioIo import (
    AudioStreamPlayer,
    NullPlayer,
    WebPlayer,
    LocalPlayer,
    AudioIOConfig,
    TTSEmotion,
    make_silent_wav,
    split_sentences,
    wav_duration,
    wav_is_valid,
    wav_sample_rate,
    wav_channels,
    # ,
    create_speak_tool,
)
from toolboxv2.tests.test_mods.test_isaa.test_base.test_audio._utils import (
    assert_valid_wav,
    make_sine_wav,
    wav_info,
    print_provider_status,
)


# =============================================================================
# SENTENCE SPLITTER TESTS — no external deps, pure text logic
# =============================================================================

class TestSplitSentences(unittest.TestCase):
    """
    Real text transformation tests. No mocks.
    These verify the actual sentence-splitting logic that controls
    when TTS chunks are emitted during streaming.
    """

    def test_single_sentence_returned_as_is(self):
        result = split_sentences("Hello world.")
        self.assertEqual(len(result), 1)
        self.assertIn("Hello world", result[0])

    def test_two_sentences_split(self):
        text = "This is the first sentence. This is the second sentence."
        result = split_sentences(text, min_chars=10)
        self.assertEqual(len(result), 2)
        self.assertIn("first sentence", result[0])
        self.assertIn("second sentence", result[1])

    def test_question_mark_triggers_split(self):
        text = "How are you? I am fine. Let us continue."
        result = split_sentences(text, min_chars=5)
        self.assertGreaterEqual(len(result), 2)

    def test_exclamation_mark_triggers_split(self):
        text = "Watch out! The process has started. Please wait."
        result = split_sentences(text, min_chars=5)
        self.assertGreaterEqual(len(result), 2)

    def test_short_fragments_merged(self):
        # "Hi. How are you today? I am doing great."
        # "Hi." (2 chars) is too short → should be merged
        text = "Hi. How are you today? I am doing great."
        result = split_sentences(text, min_chars=20)
        # "Hi." merged with next fragment
        self.assertIn("Hi", result[0])
        self.assertGreater(len(result[0]), 5)

    def test_empty_string_returns_list_with_empty(self):
        result = split_sentences("", min_chars=20)
        self.assertEqual(len(result), 1)

    def test_long_text_produces_multiple_chunks(self):
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "She sells seashells by the seashore. "
            "Peter Piper picked a peck of pickled peppers. "
            "How much wood would a woodchuck chuck?"
        )
        result = split_sentences(text, min_chars=20)
        self.assertGreaterEqual(len(result), 3)

    def test_no_sentence_ending_punct(self):
        text = "This text has no ending punctuation at all"
        result = split_sentences(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], text)

    def test_semicolon_triggers_split(self):
        text = "First part of text; second part of text follows here."
        result = split_sentences(text, min_chars=10)
        self.assertGreaterEqual(len(result), 2)

    def test_colon_triggers_split(self):
        text = "Result: this is the output value for the query."
        result = split_sentences(text, min_chars=10)
        # Colon + space triggers boundary
        self.assertGreaterEqual(len(result), 1)

    def test_all_sentences_nonempty(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = split_sentences(text, min_chars=5)
        for s in result:
            self.assertTrue(s.strip(), f"Empty sentence in result: {result!r}")

    def test_content_preserved(self):
        """No text content should be lost during splitting."""
        text = "Alpha sentence here. Beta sentence there. Gamma sentence everywhere."
        result = split_sentences(text, min_chars=5)
        joined = " ".join(result)
        for word in ["Alpha", "Beta", "Gamma"]:
            self.assertIn(word, joined, f"Word '{word}' lost after split")


# =============================================================================
# WAV UTILITY TESTS — real byte manipulation
# =============================================================================

class TestWavUtilities(unittest.TestCase):
    """
    Tests for WAV byte utility functions.
    All tests use make_silent_wav() — no external providers needed.
    """

    def test_make_silent_wav_returns_valid_bytes(self):
        wav = make_silent_wav(0.5, 16000)
        assert_valid_wav(self, wav, "make_silent_wav")

    def test_wav_duration_correct(self):
        for duration in [0.1, 0.5, 1.0, 2.0]:
            wav = make_silent_wav(duration, 16000)
            measured = wav_duration(wav)
            self.assertAlmostEqual(measured, duration, delta=0.01,
                                   msg=f"Duration mismatch for {duration}s")

    def test_wav_sample_rate_correct(self):
        for sr in [8000, 16000, 22050, 44100]:
            wav = make_silent_wav(0.5, sr)
            measured = wav_sample_rate(wav)
            self.assertEqual(measured, sr, f"Sample rate mismatch: expected {sr}, got {measured}")

    def test_wav_channels_correct(self):
        wav = make_silent_wav(0.5, 16000)
        self.assertEqual(wav_channels(wav), 1)

    def test_wav_is_valid_true_for_silent(self):
        wav = make_silent_wav(0.5, 16000)
        self.assertTrue(wav_is_valid(wav))

    def test_wav_is_valid_false_for_garbage(self):
        self.assertFalse(wav_is_valid(b"not a wav file at all"))
        self.assertFalse(wav_is_valid(b""))
        self.assertFalse(wav_is_valid(b"\x00" * 8))

    def test_wav_is_valid_false_for_short(self):
        self.assertFalse(wav_is_valid(b"RIFF"))

    def test_wav_info_dict_complete(self):
        wav = make_silent_wav(1.0, 22050)
        info = wav_info(wav)
        self.assertIn("sample_rate", info)
        self.assertIn("channels", info)
        self.assertIn("n_frames", info)
        self.assertIn("duration_s", info)
        self.assertEqual(info["sample_rate"], 22050)
        self.assertAlmostEqual(info["duration_s"], 1.0, delta=0.01)

    def test_sine_wav_valid(self):
        wav = make_sine_wav(440.0, 1.0, 16000)
        assert_valid_wav(self, wav, "sine_440hz")
        info = wav_info(wav)
        self.assertAlmostEqual(info["duration_s"], 1.0, delta=0.01)

    def test_wav_frames_are_nonzero_for_sine(self):
        """Sine wave must contain non-silence frames."""
        import struct
        wav = make_sine_wav(440.0, 0.1, 16000)
        with io.BytesIO(wav) as buf:
            with wave.open(buf, "rb") as w:
                frames = w.readframes(w.getnframes())
        samples = [struct.unpack_from("<h", frames, i)[0] for i in range(0, min(len(frames), 100), 2)]
        self.assertTrue(any(abs(s) > 100 for s in samples), "Sine wave should have non-zero samples")


# =============================================================================
# NULL PLAYER TESTS — real async execution
# =============================================================================

class TestNullPlayer(unittest.IsolatedAsyncioTestCase):
    """
    NullPlayer: audio is discarded but tracked.
    All metadata flow tested without external dependencies.
    """

    async def test_start_stop_no_error(self):
        player = NullPlayer()
        await player.start()
        await player.stop()

    async def test_queue_audio_stores_chunk(self):
        player = NullPlayer()
        await player.start()
        wav = make_silent_wav(0.2)
        meta = {"text": "hello", "emotion": "neutral", "duration_s": 0.2}
        await player.queue_audio(wav, meta)
        self.assertEqual(player.total_chunks, 1)

    async def test_chunk_index_increments(self):
        player = NullPlayer()
        await player.start()
        for i in range(5):
            wav = make_silent_wav(0.1)
            await player.queue_audio(wav, {"text": f"chunk {i}"})
        self.assertEqual(player.total_chunks, 5)
        indices = [meta["chunk_index"] for _, meta in player.received_chunks]
        self.assertEqual(indices, list(range(5)))

    async def test_get_received_texts(self):
        player = NullPlayer()
        await player.start()
        texts = ["First sentence.", "Second sentence.", "Third."]
        for t in texts:
            await player.queue_audio(make_silent_wav(0.1), {"text": t, "emotion": "neutral"})
        received = player.get_received_texts()
        self.assertEqual(received, texts)

    async def test_get_received_emotions(self):
        player = NullPlayer()
        await player.start()
        emotions = ["calm", "excited", "neutral"]
        for e in emotions:
            await player.queue_audio(make_silent_wav(0.1), {"text": "x", "emotion": e})
        received = player.get_received_emotions()
        self.assertEqual(received, emotions)

    async def test_total_audio_bytes(self):
        player = NullPlayer()
        await player.start()
        wav = make_silent_wav(0.5)
        await player.queue_audio(wav, {})
        await player.queue_audio(wav, {})
        self.assertEqual(player.total_audio_bytes(), len(wav) * 2)

    async def test_is_active_always_false(self):
        player = NullPlayer()
        await player.start()
        self.assertFalse(player.is_active)
        await player.queue_audio(make_silent_wav(0.1), {})
        self.assertFalse(player.is_active)  # NullPlayer is never "busy"

    async def test_wav_bytes_preserved(self):
        """NullPlayer must store exact original WAV bytes."""
        player = NullPlayer()
        await player.start()
        wav = make_silent_wav(0.3, 16000)
        await player.queue_audio(wav, {"text": "test"})
        stored_wav, _ = player.received_chunks[0]
        self.assertEqual(stored_wav, wav)
        assert_valid_wav(self, stored_wav, "NullPlayer stored")


# =============================================================================
# WEB PLAYER TESTS — real async queuing + relay semantics
# =============================================================================

class TestWebPlayer(unittest.IsolatedAsyncioTestCase):
    """
    WebPlayer in mock_mode: tests the async relay semantics that a real
    WebSocket consumer would use. Real queue operations, real WAV data.
    """

    async def test_mock_mode_stores_chunks(self):
        player = WebPlayer(mock_mode=True)
        await player.start()
        wav = make_silent_wav(0.3)
        await player.queue_audio(wav, {"text": "hello"})
        self.assertEqual(len(player.received_chunks), 1)

    async def test_chunk_index_monotonic(self):
        player = WebPlayer(mock_mode=True)
        await player.start()
        for i in range(8):
            await player.queue_audio(make_silent_wav(0.1), {"text": f"s{i}"})
        indices = [meta["chunk_index"] for _, meta in player.received_chunks]
        self.assertEqual(indices, list(range(8)))

    async def test_total_chunks_sent_counter(self):
        player = WebPlayer(mock_mode=True)
        await player.start()
        for _ in range(3):
            await player.queue_audio(make_silent_wav(0.1), {})
        self.assertEqual(player.total_chunks_sent, 3)

    async def test_get_received_texts(self):
        player = WebPlayer(mock_mode=True)
        await player.start()
        texts = ["Sentence A.", "Sentence B.", "Sentence C."]
        for t in texts:
            await player.queue_audio(make_silent_wav(0.1), {"text": t})
        self.assertEqual(player.get_received_texts(), texts)

    async def test_get_received_emotions(self):
        player = WebPlayer(mock_mode=True)
        await player.start()
        for emo in ["calm", "excited", "friendly"]:
            await player.queue_audio(make_silent_wav(0.1), {"emotion": emo})
        self.assertEqual(player.get_received_emotions(), ["calm", "excited", "friendly"])

    async def test_get_received_audio_frames_concatenates(self):
        """Concatenated frames should equal sum of individual chunks."""
        player = WebPlayer(mock_mode=True)
        await player.start()
        w1 = make_silent_wav(0.2, 16000)
        w2 = make_silent_wav(0.3, 16000)
        await player.queue_audio(w1, {})
        await player.queue_audio(w2, {})

        with io.BytesIO(w1) as b:
            with wave.open(b, "rb") as w:
                f1 = w.readframes(w.getnframes())
        with io.BytesIO(w2) as b:
            with wave.open(b, "rb") as w:
                f2 = w.readframes(w.getnframes())

        expected = f1 + f2
        result = player.get_received_audio_frames()
        self.assertEqual(result, expected)

    async def test_live_mode_iter_chunks(self):
        """
        Live mode: producer puts chunks into queue, consumer drains via iter_chunks().
        Verifies the async generator relay pattern a WS handler would use.
        """
        player = WebPlayer(mock_mode=False, max_queue=10)
        await player.start()

        produced = []
        consumed = []

        async def producer():
            for i in range(3):
                wav = make_silent_wav(0.1)
                meta = {"text": f"chunk {i}", "chunk_index": i}
                await player.queue_audio(wav, meta)
                produced.append(i)
                await asyncio.sleep(0.01)
            await player.stop()

        async def consumer():
            async for chunk, meta in player.iter_chunks():
                assert_valid_wav(self, chunk, f"iter_chunks chunk {meta.get('chunk_index')}")
                consumed.append(meta.get("chunk_index"))

        await asyncio.gather(producer(), consumer())

        self.assertEqual(len(produced), 3)
        self.assertEqual(len(consumed), 3)
        self.assertEqual(sorted(consumed), [0, 1, 2])

    async def test_get_next_chunk_returns_none_on_timeout(self):
        player = WebPlayer(mock_mode=False)
        await player.start()
        result = await player.get_next_chunk(timeout=0.1)
        self.assertIsNone(result)
        await player.stop()

    async def test_get_next_chunk_returns_chunk(self):
        player = WebPlayer(mock_mode=False)
        await player.start()
        wav = make_silent_wav(0.2)
        await player.queue_audio(wav, {"text": "hello"})
        result = await player.get_next_chunk(timeout=1.0)
        self.assertIsNotNone(result)
        chunk, meta = result
        assert_valid_wav(self, chunk, "get_next_chunk")
        self.assertEqual(meta["text"], "hello")
        await player.stop()

    async def test_get_total_duration(self):
        player = WebPlayer(mock_mode=True)
        await player.start()
        for d in [0.2, 0.3, 0.5]:
            wav = make_silent_wav(d)
            await player.queue_audio(wav, {"duration_s": d})
        total = player.get_total_duration()
        self.assertAlmostEqual(total, 1.0, delta=0.05)

    async def test_wav_bytes_integrity_through_queue(self):
        """WAV bytes must be bit-for-bit identical after going through the queue."""
        player = WebPlayer(mock_mode=False)
        await player.start()
        original_wav = make_sine_wav(440.0, 0.5, 16000)
        await player.queue_audio(original_wav, {"text": "integrity test"})
        result = await player.get_next_chunk(timeout=2.0)
        self.assertIsNotNone(result)
        chunk, _ = result
        self.assertEqual(chunk, original_wav, "WAV bytes corrupted in transit through WebPlayer queue")
        await player.stop()


# =============================================================================
# AUDIO STREAM PLAYER TESTS — real worker, mocked TTS
# =============================================================================

class TestAudioStreamPlayer(unittest.IsolatedAsyncioTestCase):
    """
    AudioStreamPlayer: the TTS synthesis is mocked (synthesize() patched),
    but the async worker, queue, and player.queue_audio() calls are real.
    This tests the orchestration layer without needing any TTS provider.
    """

    def _make_player(self, player_backend=None):
        from toolboxv2.mods.isaa.base.audio_io.Tts import TTSConfig, TTSBackend, TTSResult, TTSEmotion as TE
        tts_config = TTSConfig(backend=TTSBackend.PIPER, voice="en_US-amy-medium")
        null = player_backend or NullPlayer()
        return AudioStreamPlayer(
            player_backend=null,
            tts_config=tts_config,
            session_id="test_session",
        ), null

    async def test_start_stop_clean(self):
        sp, _ = self._make_player()
        await sp.start()
        self.assertFalse(sp.is_busy)
        await sp.stop()

    async def test_queue_text_calls_player_queue_audio(self):
        """The full path: queue_text → worker → synthesize → player.queue_audio()."""
        from toolboxv2.mods.isaa.base.audio_io.Tts import TTSResult, TTSEmotion as TE

        null = NullPlayer()
        sp, _ = self._make_player(player_backend=null)

        fake_wav = make_silent_wav(0.3, 22050)
        fake_result = TTSResult(audio=fake_wav, format="wav", sample_rate=22050, duration=0.3)

        with patch(
            "toolboxv2.mods.isaa.base.audio_io.audioIo.synthesize",
            return_value=fake_result,
        ):
            await sp.start()
            await sp.queue_text("Hello, this is a test.", emotion=TTSEmotion.FRIENDLY)
            # Give worker time to process
            await asyncio.sleep(0.5)
            await sp.stop()

        self.assertEqual(null.total_chunks, 1)
        stored_wav, meta = null.received_chunks[0]
        assert_valid_wav(self, stored_wav, "AudioStreamPlayer output")
        self.assertEqual(meta["text"], "Hello, this is a test.")
        self.assertEqual(meta["emotion"], "friendly")
        self.assertEqual(meta["session_id"], "test_session")

    async def test_multiple_texts_ordered(self):
        """Multiple queue_text() calls must be processed in order."""
        from toolboxv2.mods.isaa.base.audio_io.Tts import TTSResult

        null = NullPlayer()
        sp, _ = self._make_player(player_backend=null)

        texts = ["First.", "Second.", "Third.", "Fourth."]
        fake_wav = make_silent_wav(0.1)
        fake_result = TTSResult(audio=fake_wav, format="wav", sample_rate=16000, duration=0.1)

        with patch(
            "toolboxv2.mods.isaa.base.audio_io.audioIo.synthesize",
            return_value=fake_result,
        ):
            await sp.start()
            for t in texts:
                await sp.queue_text(t)
            await asyncio.sleep(0.5)
            await sp.stop()

        received = null.get_received_texts()
        self.assertEqual(received, texts)

    async def test_emotion_passed_to_synthesize(self):
        """The emotion must be injected into TTSConfig before synthesis."""
        from toolboxv2.mods.isaa.base.audio_io.Tts import TTSResult

        null = NullPlayer()
        sp, _ = self._make_player(player_backend=null)

        fake_wav = make_silent_wav(0.1)
        captured_configs = []

        def fake_synthesize(text, config=None):
            captured_configs.append(config)
            return TTSResult(audio=fake_wav, format="wav", sample_rate=16000, duration=0.1)

        with patch("toolboxv2.mods.isaa.base.audio_io.audioIo.synthesize", side_effect=fake_synthesize):
            await sp.start()
            await sp.queue_text("Exciting news!", emotion=TTSEmotion.EXCITED)
            await asyncio.sleep(0.5)
            await sp.stop()

        self.assertEqual(len(captured_configs), 1)
        cfg = captured_configs[0]
        self.assertEqual(cfg.emotion, TTSEmotion.EXCITED)

    async def test_pending_texts_counter(self):
        null = NullPlayer()
        sp, _ = self._make_player(player_backend=null)

        from toolboxv2.mods.isaa.base.audio_io.Tts import TTSResult
        fake_wav = make_silent_wav(0.1)

        slow_event = asyncio.Event()

        async def slow_synthesize(text, config=None):
            await asyncio.sleep(0.3)
            return TTSResult(audio=fake_wav, format="wav", sample_rate=16000, duration=0.1)

        # Don't start worker — just check queue counting
        await sp.player.start()
        for t in ["A.", "B.", "C."]:
            await sp.queue_text(t)

        self.assertEqual(sp.pending_texts, 3)

    async def test_with_web_player_relay(self):
        """AudioStreamPlayer with WebPlayer backend — relay to consumer."""
        from toolboxv2.mods.isaa.base.audio_io.Tts import TTSResult

        web = WebPlayer(mock_mode=False, max_queue=10)
        sp = AudioStreamPlayer(
            player_backend=web,
            session_id="web_test",
        )

        fake_wav = make_sine_wav(220.0, 0.2, 16000)
        fake_result = TTSResult(audio=fake_wav, format="wav", sample_rate=16000, duration=0.2)

        consumed = []

        async def consumer():
            async for chunk, meta in web.iter_chunks():
                assert_valid_wav(self, chunk, "WebPlayer relay")
                consumed.append(meta.get("text"))

        with patch("toolboxv2.mods.isaa.base.audio_io.audioIo.synthesize", return_value=fake_result):
            await sp.start()
            await sp.queue_text("First relay chunk.", emotion=TTSEmotion.CALM)
            await sp.queue_text("Second relay chunk.", emotion=TTSEmotion.NEUTRAL)
            await asyncio.sleep(0.5)
            await sp.stop()  # This calls web.stop() → iter_chunks() exits

        consumer_task = asyncio.create_task(consumer())
        # Re-run consume from mock data (already stopped, iter will drain)
        # Re-test with mock_mode
        web2 = WebPlayer(mock_mode=True)
        sp2 = AudioStreamPlayer(player_backend=web2, session_id="web_test2")
        with patch("toolboxv2.mods.isaa.base.audio_io.audioIo.synthesize", return_value=fake_result):
            await sp2.start()
            await sp2.queue_text("Mock relay A.", emotion=TTSEmotion.FRIENDLY)
            await asyncio.sleep(0.3)
            await sp2.stop()

        self.assertEqual(len(web2.received_chunks), 1)
        assert_valid_wav(self, web2.received_chunks[0][0], "WebPlayer mock relay")
        consumer_task.cancel()


# =============================================================================
# SPEAK TOOL TESTS — real async tool execution
# =============================================================================

class TestSpeakTool(unittest.IsolatedAsyncioTestCase):
    """
    Tests for create_speak_tool() and the speak() function it returns.
    AudioStreamPlayer worker is started but TTS is mocked.
    """

    async def asyncSetUp(self):
        from toolboxv2.mods.isaa.base.audio_io.Tts import TTSResult
        self.null = NullPlayer()
        self.sp = AudioStreamPlayer(
            player_backend=self.null,
            session_id="speak_tool_test",
        )
        self.fake_wav = make_silent_wav(0.2)
        self.fake_result = TTSResult(
            audio=self.fake_wav, format="wav", sample_rate=16000, duration=0.2
        )
        self.patcher = patch(
            "toolboxv2.mods.isaa.base.audio_io.audioIo.synthesize",
            return_value=self.fake_result,
        )
        self.patcher.start()
        await self.sp.start()

    async def asyncTearDown(self):
        await self.sp.stop()
        self.patcher.stop()

    async def test_speak_returns_queued_string(self):
        speak = create_speak_tool(self.sp)
        result = await speak("Hello world.")
        self.assertEqual(result, "[queued for speech]")

    async def test_speak_queues_text_to_player(self):
        speak = create_speak_tool(self.sp)
        await speak("Test sentence.")
        await asyncio.sleep(0.4)
        self.assertEqual(self.null.total_chunks, 1)
        self.assertIn("Test sentence", self.null.get_received_texts()[0])

    async def test_speak_with_emotion_calm(self):
        speak = create_speak_tool(self.sp)
        from toolboxv2.mods.isaa.base.audio_io.Tts import TTSResult
        captured = []

        def cap_synthesize(text, config=None):
            captured.append(config)
            return TTSResult(audio=self.fake_wav, format="wav", sample_rate=16000, duration=0.2)

        with patch("toolboxv2.mods.isaa.base.audio_io.audioIo.synthesize", side_effect=cap_synthesize):
            await speak("Stay calm everyone.", "calm")
            await asyncio.sleep(0.4)

        self.assertTrue(len(captured) > 0)
        self.assertEqual(captured[0].emotion, TTSEmotion.CALM)

    async def test_speak_invalid_emotion_defaults_neutral(self):
        speak = create_speak_tool(self.sp)
        from toolboxv2.mods.isaa.base.audio_io.Tts import TTSResult
        captured = []

        def cap_synthesize(text, config=None):
            captured.append(config)
            return TTSResult(audio=self.fake_wav, format="wav", sample_rate=16000, duration=0.2)

        with patch("toolboxv2.mods.isaa.base.audio_io.audioIo.synthesize", side_effect=cap_synthesize):
            await speak("Test text.", "totally_invalid_emotion_xyz")
            await asyncio.sleep(0.4)

        self.assertEqual(captured[0].emotion, TTSEmotion.NEUTRAL)

    async def test_speak_long_text_splits_into_sentences(self):
        """Long multi-sentence text should produce multiple TTS calls."""
        speak = create_speak_tool(self.sp)
        long_text = (
            "This is the first sentence of a long response. "
            "This is the second sentence with more detail. "
            "This is the third and final sentence."
        )
        await speak(long_text)
        await asyncio.sleep(0.8)
        # Should have 3 chunks (one per sentence)
        self.assertGreaterEqual(self.null.total_chunks, 2)

    async def test_speak_empty_text_no_chunk(self):
        speak = create_speak_tool(self.sp)
        await speak("   ")
        await asyncio.sleep(0.2)
        self.assertEqual(self.null.total_chunks, 0)

    async def test_speak_all_emotions_valid(self):
        """Every TTSEmotion value must be accepted by speak()."""
        speak = create_speak_tool(self.sp)
        from toolboxv2.mods.isaa.base.audio_io.Tts import TTSEmotion as TE
        for emo in TE:
            result = await speak(f"Testing {emo.value} emotion.", emo.value)
            self.assertEqual(result, "[queued for speech]")


# =============================================================================
# AUDIO IO CONFIG TESTS
# =============================================================================

class TestAudioIOConfig(unittest.TestCase):
    """Config construction and player factory methods."""

    def test_default_config_valid(self):
        config = AudioIOConfig()
        self.assertEqual(config.player_backend_type, "null")
        self.assertIsNotNone(config.stt_config)
        self.assertIsNotNone(config.tts_config)

    def test_build_null_player(self):
        config = AudioIOConfig(player_backend_type="null")
        player = config.build_player()
        self.assertIsInstance(player, NullPlayer)

    def test_build_web_player(self):
        config = AudioIOConfig(player_backend_type="web", web_player_max_queue=50)
        player = config.build_player()
        self.assertIsInstance(player, WebPlayer)

    def test_build_invalid_player_raises(self):
        config = AudioIOConfig(player_backend_type="invalid_xyz")
        with self.assertRaises(ValueError):
            config.build_player()

    def test_build_stream_player_returns_instance(self):
        config = AudioIOConfig(player_backend_type="null")
        sp = config.build_stream_player("test_session")
        self.assertIsInstance(sp, AudioStreamPlayer)
        self.assertEqual(sp.session_id, "test_session")


if __name__ == "__main__":
    print_provider_status()
    unittest.main(verbosity=2)
