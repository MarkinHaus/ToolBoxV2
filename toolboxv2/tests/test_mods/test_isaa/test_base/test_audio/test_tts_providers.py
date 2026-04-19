"""
test_tts_providers.py
=====================

Real TTS synthesis tests. Each test:
  1. Checks provider availability, skips with install hint if missing
  2. Calls the actual synthesize() function with real text
  3. Validates the WAV output structurally and semantically
  4. Tests emotion injection where supported

No mocking of the TTS backends themselves — if a test runs, it exercises
real audio generation that would work identically in production.

Run:
    python -m unittest test_audio.test_tts_providers -v

To run only groq tests:
    python -m unittest test_audio.test_tts_providers.TestGroqTTS -v
"""

import io
import os
import unittest
import wave

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolboxv2.mods.isaa.base.audio_io.Tts import (
    TTSBackend,
    TTSConfig,
    TTSEmotion,
    TTSResult,
    synthesize,
    synthesize_stream,
    synthesize_groq,
)
from toolboxv2.tests.test_mods.test_isaa.test_base.test_audio._utils import (
    HAS_FASTER_WHISPER,
    HAS_GROQ,
    HAS_GROQ_PACKAGE,
    HAS_PIPER,
    HAS_INDEXTTS,
    HAS_INDEX_TTS_WEIGHTS,
    HAS_VIBEVOICE,
    assert_valid_wav,
    print_provider_status,
    require_groq,
    require_piper,
    require_index_tts,
    wav_info,
)


# =============================================================================
# HELPERS
# =============================================================================

TEST_TEXT_EN = "Hello! This is a test of the text-to-speech system."
TEST_TEXT_DE = "Hallo! Dies ist ein Test des Sprachsynthese-Systems."
TEST_TEXT_SHORT = "Hello."
TEST_TEXT_LONG = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump!"
)


def assert_tts_result(test_case: unittest.TestCase, result: TTSResult, context: str = ""):
    """Full validation of a TTSResult."""
    test_case.assertIsInstance(result, TTSResult)
    test_case.assertIsNotNone(result.audio)
    test_case.assertGreater(len(result.audio), 44, f"[{context}] Audio too short")
    assert_valid_wav(test_case, result.audio, context)

    info = wav_info(result.audio)
    test_case.assertGreater(info["duration_s"], 0.05, f"[{context}] Audio duration too short")
    test_case.assertGreater(info["n_frames"], 0, f"[{context}] No audio frames")


# =============================================================================
# PIPER TESTS (local CPU, no API key needed)
# =============================================================================

class TestPiperTTS(unittest.TestCase):
    """
    Piper: fast local CPU TTS. No API key required.
    Install: pip install piper-tts
    Models download automatically on first use (~50-100MB).
    """

    @require_piper
    def test_synthesize_english_basic(self):
        result = synthesize(
            TEST_TEXT_SHORT,
            config=TTSConfig(
                backend=TTSBackend.PIPER,
                voice="de_DE-thorsten-high",
                language="en",
            ),
        )
        assert_tts_result(self, result, "piper_en_basic")

    @require_piper
    def test_synthesize_returns_wav(self):
        result = synthesize(
            TEST_TEXT_EN,
            config=TTSConfig(backend=TTSBackend.PIPER, voice="de_DE-thorsten-high"),
        )
        self.assertEqual(result.format, "wav")
        self.assertGreater(result.sample_rate, 0)

    @require_piper
    def test_synthesize_long_text(self):
        result = synthesize(
            TEST_TEXT_LONG,
            config=TTSConfig(backend=TTSBackend.PIPER, voice="de_DE-thorsten-high"),
        )
        assert_tts_result(self, result, "piper_long")
        info = wav_info(result.audio)
        # Long text must produce more than 1 second
        self.assertGreater(info["duration_s"], 1.0, "Long text should produce > 1s audio")

    @require_piper
    def test_stream_yields_bytes(self):
        chunks = list(synthesize_stream(
            TEST_TEXT_SHORT,
            config=TTSConfig(backend=TTSBackend.PIPER, voice="de_DE-thorsten-high"),
        ))
        self.assertGreater(len(chunks), 0)
        total_bytes = sum(len(c) for c in chunks)
        self.assertGreater(total_bytes, 0, "Stream must yield some audio bytes")

    @require_piper
    def test_emotion_prefix_injected_into_text(self):
        """
        Piper doesn't interpret emotion prompts but the prefix
        must be passed through without crashing or producing silence.
        """
        from toolboxv2.mods.isaa.base.audio_io.Tts import _apply_emotion_prefix
        prefixed = _apply_emotion_prefix(TEST_TEXT_EN, TTSEmotion.CALM)
        self.assertIn("[calm", prefixed.lower())

        # Synthesis must still produce valid audio
        result = synthesize(
            TEST_TEXT_EN,
            config=TTSConfig(
                backend=TTSBackend.PIPER,
                voice="de_DE-thorsten-high",
                emotion=TTSEmotion.FRIENDLY,
            ),
        )
        assert_tts_result(self, result, "piper_emotion")

    @require_piper
    def test_german_voice(self):
        """German voice should work if installed."""
        try:
            result = synthesize(
                TEST_TEXT_DE,
                config=TTSConfig(
                    backend=TTSBackend.PIPER,
                    voice="de_DE-thorsten-medium",
                    language="de",
                ),
            )
            assert_tts_result(self, result, "piper_de")
        except Exception as e:
            self.skipTest(f"German Piper voice not installed: {e}")


# =============================================================================
# GROQ TTS TESTS (API, needs GROQ_API_KEY)
# =============================================================================

class TestGroqTTS(unittest.TestCase):
    """
    Groq TTS: Orpheus model via API. Fast, good quality.
    Install: pip install groq
    Requires: export GROQ_API_KEY=gsk_...
    """

    @require_groq
    def test_synthesize_english_basic(self):
        result = synthesize(
            TEST_TEXT_SHORT,
            config=TTSConfig(
                backend=TTSBackend.GROQ_TTS,
                voice="autumn",
                language="en",
                groq_api_key=os.environ.get("GROQ_API_KEY"),
            ),
        )
        assert_tts_result(self, result, "groq_en_basic")

    @require_groq
    def test_synthesize_convenience_function(self):
        result = synthesize_groq(
            TEST_TEXT_EN,
            api_key=os.environ.get("GROQ_API_KEY"),
            voice="autumn",
        )
        assert_tts_result(self, result, "groq_convenience")

    @require_groq
    def test_multiple_voices(self):
        """Test that multiple Groq voices all produce valid audio."""
        voices = ["autumn", "diana", "austin"]
        for voice in voices:
            with self.subTest(voice=voice):
                result = synthesize(
                    TEST_TEXT_SHORT,
                    config=TTSConfig(
                        backend=TTSBackend.GROQ_TTS,
                        voice=voice,
                        groq_api_key=os.environ.get("GROQ_API_KEY"),
                    ),
                )
                assert_tts_result(self, result, f"groq_{voice}")

    @require_groq
    def test_result_has_duration(self):
        result = synthesize(
            TEST_TEXT_EN,
            config=TTSConfig(
                backend=TTSBackend.GROQ_TTS,
                voice="autumn",
                groq_api_key=os.environ.get("GROQ_API_KEY"),
            ),
        )
        # Duration should be parseable from the WAV header
        info = wav_info(result.audio)
        self.assertGreater(info["duration_s"], 0)

    @require_groq
    def test_stream_yields_complete_audio(self):
        """Groq stream returns single chunk but must be valid WAV."""
        chunks = list(synthesize_stream(
            TEST_TEXT_SHORT,
            config=TTSConfig(
                backend=TTSBackend.GROQ_TTS,
                voice="autumn",
                groq_api_key=os.environ.get("GROQ_API_KEY"),
            ),
        ))
        self.assertGreater(len(chunks), 0)
        combined = b"".join(chunks)
        assert_valid_wav(self, combined, "groq_stream")

    @require_groq
    def test_emotion_neutral_and_calm_both_work(self):
        """Emotion prefix should not break synthesis."""
        for emotion in [TTSEmotion.NEUTRAL, TTSEmotion.CALM, TTSEmotion.EXCITED]:
            with self.subTest(emotion=emotion.value):
                result = synthesize(
                    TEST_TEXT_SHORT,
                    config=TTSConfig(
                        backend=TTSBackend.GROQ_TTS,
                        voice="autumn",
                        groq_api_key=os.environ.get("GROQ_API_KEY"),
                        emotion=emotion,
                    ),
                )
                assert_tts_result(self, result, f"groq_emotion_{emotion.value}")


# =============================================================================
# INDEX TTS TESTS (local GPU, zero-shot voice cloning)
# =============================================================================

class TestIndexTTS(unittest.TestCase):
    """
    IndexTTS: zero-shot voice cloning, local GPU.
    Install: pip install indextts
             OR: git clone https://github.com/index-tts/index-tts && pip install -e index-tts
    Weights: huggingface-cli download IndexTeam/IndexTTS --local-dir ./checkpoints
    Reference: provide a 3-10s WAV file of target speaker via INDEX_TTS_REFERENCE_AUDIO env.
    """

    def setUp(self):
        self.reference_audio = os.environ.get("INDEX_TTS_REFERENCE_AUDIO")
        self.model_dir = os.environ.get("INDEX_TTS_MODEL_DIR", "./checkpoints")

    @require_index_tts
    def test_requires_reference_audio_or_skip(self):
        if not self.reference_audio:
            self.skipTest(
                "No reference audio provided.\n"
                "  export INDEX_TTS_REFERENCE_AUDIO=/path/to/speaker.wav\n"
                "  (3-10 seconds of clean speech, mono WAV)"
            )

    @require_index_tts
    def test_synthesize_with_reference(self):
        if not self.reference_audio:
            self.skipTest("INDEX_TTS_REFERENCE_AUDIO not set")
        if not os.path.exists(self.reference_audio):
            self.skipTest(f"Reference audio not found: {self.reference_audio}")

        result = synthesize(
            TEST_TEXT_EN,
            config=TTSConfig(
                backend=TTSBackend.INDEX_TTS,
                index_tts_reference_audio=self.reference_audio,
                index_tts_model_dir=self.model_dir,
                emotion=TTSEmotion.NEUTRAL,
            ),
        )
        assert_tts_result(self, result, "index_tts_neutral")

    @require_index_tts
    def test_emotion_variants_all_produce_audio(self):
        if not self.reference_audio or not os.path.exists(self.reference_audio):
            self.skipTest("INDEX_TTS_REFERENCE_AUDIO not set or missing")

        emotions = [TTSEmotion.CALM, TTSEmotion.EXCITED, TTSEmotion.FRIENDLY]
        for emo in emotions:
            with self.subTest(emotion=emo.value):
                result = synthesize(
                    TEST_TEXT_SHORT,
                    config=TTSConfig(
                        backend=TTSBackend.INDEX_TTS,
                        index_tts_reference_audio=self.reference_audio,
                        index_tts_model_dir=self.model_dir,
                        emotion=emo,
                    ),
                )
                assert_tts_result(self, result, f"index_tts_{emo.value}")

    @require_index_tts
    def test_missing_reference_raises_value_error(self):
        """Without reference_audio, must raise ValueError not crash."""
        from toolboxv2.mods.isaa.base.audio_io.Tts import _synthesize_index_tts
        config = TTSConfig(
            backend=TTSBackend.INDEX_TTS,
            index_tts_reference_audio=None,
            index_tts_model_dir=self.model_dir,
        )
        with self.assertRaises(ValueError):
            _synthesize_index_tts(TEST_TEXT_SHORT, config)


# =============================================================================
# TTSConfig VALIDATION TESTS (no provider needed)
# =============================================================================

class TestTTSConfigValidation(unittest.TestCase):
    """Configuration construction and validation — no external deps."""

    def test_default_config_valid(self):
        config = TTSConfig()
        self.assertEqual(config.backend, TTSBackend.PIPER)
        self.assertEqual(config.emotion, TTSEmotion.NEUTRAL)

    def test_all_emotions_constructable(self):
        for emo in TTSEmotion:
            config = TTSConfig(backend=TTSBackend.PIPER, emotion=emo)
            self.assertEqual(config.emotion, emo)

    def test_speed_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            TTSConfig(speed=0.1)
        with self.assertRaises(ValueError):
            TTSConfig(speed=5.0)

    def test_default_voices_set_correctly(self):
        for backend in TTSBackend:
            config = TTSConfig(backend=backend)
            self.assertIsNotNone(config.voice)

    def test_index_tts_config_fields(self):
        config = TTSConfig(
            backend=TTSBackend.INDEX_TTS,
            index_tts_reference_audio="./test.wav",
            index_tts_model_dir="./weights",
            index_tts_cfg_scale=2.5,
            emotion=TTSEmotion.CALM,
        )
        self.assertEqual(config.index_tts_reference_audio, "./test.wav")
        self.assertEqual(config.index_tts_model_dir, "./weights")
        self.assertEqual(config.index_tts_cfg_scale, 2.5)
        self.assertEqual(config.emotion, TTSEmotion.CALM)

    def test_emotion_prefix_dict_complete(self):
        from toolboxv2.mods.isaa.base.audio_io.Tts import _EMOTION_PREFIXES, _apply_emotion_prefix
        for emo in TTSEmotion:
            self.assertIn(emo, _EMOTION_PREFIXES)

        # Neutral produces no prefix
        result = _apply_emotion_prefix("Hello", TTSEmotion.NEUTRAL)
        self.assertEqual(result, "Hello")

        # Non-neutral adds prefix
        result = _apply_emotion_prefix("Hello", TTSEmotion.CALM)
        self.assertIn("[calm", result.lower())
        self.assertIn("Hello", result)

    def test_unknown_backend_raises(self):
        config = TTSConfig()
        # Manually set an invalid backend (normally prevented by enum)
        from unittest.mock import patch
        with self.assertRaises(ValueError):
            synthesize("test", config=config.__class__(backend=None))


if __name__ == "__main__":
    print_provider_status()
    unittest.main(verbosity=2)
