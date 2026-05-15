"""
test_qwen3_tts.py
=================

Real Qwen3-TTS synthesis tests. Follows the pattern of test_tts_providers.py.

Qwen3-TTS covers two modes, selected implicitly from TTSConfig fields:
  - VoiceDesign (no ref_audio) — prompt-driven timbre
  - Base/Clone  (ref_audio set) — zero-shot voice clone

Run:
    python -m unittest test_audio.test_qwen3_tts -v

Env:
    QWEN3_REF_AUDIO=/path/to/3_to_10s_speaker.wav   # enables clone tests
    QWEN3_REF_TEXT="transcript of the ref audio"    # optional, improves clone
    QWEN3_DEVICE=cuda                               # or cpu
"""

import os
import unittest
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolboxv2.mods.isaa.base.audio_io.Tts import (
    TTSBackend,
    TTSConfig,
    TTSEmotion,
    TTSResult,
    synthesize,
    synthesize_stream,
    synthesize_qwen3,
    _qwen3_select_model_id,
    _qwen3_resolve_instruct,
)


# =============================================================================
# HELPERS
# =============================================================================

TEST_TEXT_EN = "Hello! This is a Qwen three voice test."
TEST_TEXT_DE = "Hallo! Dies ist ein Qwen drei Sprachtest."
TEST_TEXT_SHORT = "Hello world."


def _qwen_tts_installed() -> bool:
    try:
        import qwen_tts  # noqa: F401
        return True
    except ImportError:
        return False


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


HAS_QWEN3 = _qwen_tts_installed()
HAS_CUDA = _cuda_available()

require_qwen3 = unittest.skipUnless(
    HAS_QWEN3,
    "qwen-tts not installed. pip install -U qwen-tts",
)
require_qwen3_gpu = unittest.skipUnless(
    HAS_QWEN3 and HAS_CUDA,
    f"qwen-tts + CUDA required for full integration tests {HAS_QWEN3=} {HAS_CUDA=}",
)


def assert_tts_result(tc: unittest.TestCase, result: TTSResult, ctx: str = ""):
    from toolboxv2.tests.test_mods.test_isaa.test_base.test_audio._utils import (
        assert_valid_wav, wav_info,
    )
    tc.assertIsInstance(result, TTSResult)
    tc.assertIsNotNone(result.audio)
    tc.assertGreater(len(result.audio), 44, f"[{ctx}] audio too short")
    assert_valid_wav(tc, result.audio, ctx)
    info = wav_info(result.audio)
    tc.assertGreater(info["duration_s"], 0.05, f"[{ctx}] duration too short")


# =============================================================================
# PURE-LOGIC TESTS (no qwen-tts install needed)
# =============================================================================

class TestQwen3VariantSelection(unittest.TestCase):
    """Model-id + instruct resolution — no actual synthesis."""

    def test_voice_design_when_no_ref_audio(self):
        cfg = TTSConfig(backend=TTSBackend.QWEN3_TTS)
        self.assertEqual(
            _qwen3_select_model_id(cfg),
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        )

    def test_base_when_ref_audio_set(self):
        cfg = TTSConfig(
            backend=TTSBackend.QWEN3_TTS,
            qwen3_ref_audio="./ref.wav",
        )
        self.assertEqual(
            _qwen3_select_model_id(cfg),
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        )

    def test_explicit_model_id_wins(self):
        cfg = TTSConfig(
            backend=TTSBackend.QWEN3_TTS,
            qwen3_model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            qwen3_ref_audio="./ref.wav",
        )
        self.assertEqual(
            _qwen3_select_model_id(cfg),
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        )


class TestQwen3InstructResolution(unittest.TestCase):

    def test_style_prompt_wins_over_emotion(self):
        cfg = TTSConfig(
            backend=TTSBackend.QWEN3_TTS,
            qwen3_style_prompt="angry robot voice",
            emotion=TTSEmotion.CALM,
        )
        self.assertEqual(_qwen3_resolve_instruct(cfg), "angry robot voice")

    def test_neutral_and_custom_without_prompt_return_none(self):
        for emo in (TTSEmotion.NEUTRAL, TTSEmotion.CUSTOM):
            cfg = TTSConfig(backend=TTSBackend.QWEN3_TTS, emotion=emo)
            self.assertIsNone(_qwen3_resolve_instruct(cfg), emo.value)

    def test_emotion_maps_to_phrase(self):
        cfg = TTSConfig(
            backend=TTSBackend.QWEN3_TTS,
            emotion=TTSEmotion.EXCITED,
        )
        out = _qwen3_resolve_instruct(cfg)
        self.assertIsNotNone(out)
        self.assertIn("excit", out.lower())


class TestQwen3Config(unittest.TestCase):

    def test_config_defaults(self):
        cfg = TTSConfig(backend=TTSBackend.QWEN3_TTS)
        self.assertIsNone(cfg.qwen3_style_prompt)
        self.assertIsNone(cfg.qwen3_ref_audio)
        self.assertIsNone(cfg.qwen3_ref_text)
        self.assertIsNone(cfg.qwen3_model_id)
        self.assertEqual(cfg.qwen3_device, "cuda")
        self.assertFalse(cfg.qwen3_split_sentences)

    def test_custom_emotion_constructable(self):
        cfg = TTSConfig(
            backend=TTSBackend.QWEN3_TTS,
            emotion=TTSEmotion.CUSTOM,
            qwen3_style_prompt="whispering nervously",
        )
        self.assertEqual(cfg.emotion, TTSEmotion.CUSTOM)


# =============================================================================
# REGISTRY TESTS (no real model load — mocked)
# =============================================================================

class TestRegistryDeviceFallback(unittest.TestCase):
    """CPU fallback + warning path — no real weights touched."""

    def test_cuda_requested_without_gpu_falls_back_to_cpu(self):
        from toolboxv2.mods.isaa.base.audio_io import model_registry as reg

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": fake_torch}):
            with self.assertLogs(reg.logger, level="WARNING") as cm:
                device, dtype = reg._resolve_device("cuda")
        self.assertEqual(device, "cpu")
        self.assertEqual(dtype, "float32")
        self.assertTrue(any("CUDA unavailable" in m for m in cm.output))

    def test_cpu_explicit(self):
        from toolboxv2.mods.isaa.base.audio_io import model_registry as reg
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": fake_torch}):
            device, dtype = reg._resolve_device("cpu")
        self.assertEqual(device, "cpu")
        self.assertEqual(dtype, "float32")

    def test_windows_never_picks_flash_attention(self):
        """On Windows, SDPA must be chosen even with Ampere GPU reported."""
        from toolboxv2.mods.isaa.base.audio_io import model_registry as reg

        fake_torch = MagicMock()
        fake_torch.cuda.get_device_capability.return_value = (9, 0)  # Hopper
        fake_torch.cuda.is_available.return_value = True

        with patch("platform.system", return_value="Windows"), \
            patch.dict("sys.modules", {"torch": fake_torch}):
            impl = reg._pick_attn_impl("cuda")
        self.assertEqual(impl, "sdpa")

    def test_linux_without_flash_attn_installed_falls_back_to_sdpa(self):
        """Linux + Ampere but flash_attn import fails → SDPA."""
        from toolboxv2.mods.isaa.base.audio_io import model_registry as reg
        import builtins

        fake_torch = MagicMock()
        fake_torch.cuda.get_device_capability.return_value = (8, 6)  # Ampere
        fake_torch.cuda.is_available.return_value = True

        real_import = builtins.__import__

        def block_flash(name, *a, **kw):
            if name == "flash_attn":
                raise ImportError("not installed")
            return real_import(name, *a, **kw)

        with patch("platform.system", return_value="Linux"), \
            patch.dict("sys.modules", {"torch": fake_torch}), \
            patch.object(builtins, "__import__", side_effect=block_flash):
            impl = reg._pick_attn_impl("cuda")
        self.assertEqual(impl, "sdpa")


class TestRegistryCacheIsolation(unittest.TestCase):
    """get_qwen3 must cache per (model_id, device, dtype) triple."""

    def test_same_args_return_same_instance(self):
        from toolboxv2.mods.isaa.base.audio_io import model_registry as reg

        reg.clear()
        sentinel = object()

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.float32 = "fp32-marker"

        fake_model_cls = MagicMock()
        fake_model_cls.from_pretrained.return_value = sentinel
        fake_qwen_tts = MagicMock(Qwen3TTSModel=fake_model_cls)

        with patch.dict("sys.modules", {
            "torch": fake_torch,
            "qwen_tts": fake_qwen_tts,
        }):
            a = reg.get_qwen3("Qwen/dummy", device="cpu")
            b = reg.get_qwen3("Qwen/dummy", device="cpu")

        self.assertIs(a, sentinel)
        self.assertIs(b, sentinel)
        fake_model_cls.from_pretrained.assert_called_once()
        reg.clear()


class TestRegistryClear(unittest.TestCase):

    def test_clear_backend_drops_one(self):
        from toolboxv2.mods.isaa.base.audio_io import model_registry as reg
        reg._caches["qwen3"]["x"] = object()
        reg._caches["piper"]["y"] = object()
        reg.clear_backend("qwen3")
        self.assertNotIn("x", reg._caches["qwen3"])
        self.assertIn("y", reg._caches["piper"])
        reg.clear()

    def test_clear_all(self):
        from toolboxv2.mods.isaa.base.audio_io import model_registry as reg
        reg._caches["qwen3"]["x"] = object()
        reg._caches["piper"]["y"] = object()
        reg.clear()
        self.assertEqual(reg._caches["qwen3"], {})
        self.assertEqual(reg._caches["piper"], {})


# =============================================================================
# INTEGRATION TESTS (real synthesis — require install + GPU)
# =============================================================================

class TestQwen3VoiceDesign(unittest.TestCase):
    """Prompt-driven synthesis — needs qwen-tts + ideally GPU."""

    @require_qwen3_gpu
    def test_voice_design_basic(self):
        result = synthesize(
            TEST_TEXT_SHORT,
            config=TTSConfig(
                backend=TTSBackend.QWEN3_TTS,
                language="English",
                qwen3_style_prompt="Calm, clear male voice.",
                emotion=TTSEmotion.CUSTOM,
            ),
        )
        assert_tts_result(self, result, "qwen3_design_basic")

    @require_qwen3_gpu
    def test_voice_design_emotion_variants(self):
        for emo in [TTSEmotion.CALM, TTSEmotion.EXCITED, TTSEmotion.URGENT]:
            with self.subTest(emotion=emo.value):
                result = synthesize(
                    TEST_TEXT_SHORT,
                    config=TTSConfig(
                        backend=TTSBackend.QWEN3_TTS,
                        language="English",
                        emotion=emo,
                    ),
                )
                assert_tts_result(self, result, f"qwen3_design_{emo.value}")

    @require_qwen3_gpu
    def test_german_voice_design(self):
        result = synthesize(
            TEST_TEXT_DE,
            config=TTSConfig(
                backend=TTSBackend.QWEN3_TTS,
                language="German",
                qwen3_style_prompt="Freundliche weibliche Stimme.",
                emotion=TTSEmotion.CUSTOM,
            ),
        )
        assert_tts_result(self, result, "qwen3_design_de")

    @require_qwen3_gpu
    def test_convenience_function(self):
        result = synthesize_qwen3(
            TEST_TEXT_EN,
            style_prompt="Warm narrator voice.",
            language="English",
        )
        assert_tts_result(self, result, "qwen3_convenience_design")


class TestQwen3Clone(unittest.TestCase):
    """Zero-shot voice clone — needs ref audio from env."""

    def setUp(self):
        self.ref_audio = os.environ.get("QWEN3_REF_AUDIO")
        self.ref_text = os.environ.get("QWEN3_REF_TEXT")

    @require_qwen3_gpu
    def test_clone_with_ref_audio(self):
        if not self.ref_audio:
            self.skipTest("QWEN3_REF_AUDIO not set")
        if not os.path.exists(self.ref_audio):
            self.skipTest(f"Ref audio not found: {self.ref_audio}")

        result = synthesize(
            TEST_TEXT_EN,
            config=TTSConfig(
                backend=TTSBackend.QWEN3_TTS,
                language="English",
                qwen3_ref_audio=self.ref_audio,
                qwen3_ref_text=self.ref_text,
            ),
        )
        assert_tts_result(self, result, "qwen3_clone")

    @require_qwen3_gpu
    def test_clone_convenience(self):
        if not self.ref_audio or not os.path.exists(self.ref_audio):
            self.skipTest("QWEN3_REF_AUDIO not set or missing")
        result = synthesize_qwen3(
            TEST_TEXT_SHORT,
            ref_audio=self.ref_audio,
            ref_text=self.ref_text,
            language="English",
        )
        assert_tts_result(self, result, "qwen3_convenience_clone")


class TestQwen3Stream(unittest.TestCase):

    @require_qwen3_gpu
    def test_stream_yields_one_full_chunk(self):
        chunks = list(synthesize_stream(
            TEST_TEXT_SHORT,
            config=TTSConfig(
                backend=TTSBackend.QWEN3_TTS,
                language="English",
                qwen3_style_prompt="Neutral voice.",
                emotion=TTSEmotion.CUSTOM,
            ),
        ))
        self.assertEqual(len(chunks), 1)
        self.assertGreater(len(chunks[0]), 44)


if __name__ == "__main__":
    unittest.main(verbosity=2)
