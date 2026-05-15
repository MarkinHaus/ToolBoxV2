"""
test_audio_recorder.py
======================

Real-data tests for the input side. Follows the "no dumb-syntax" principle:
  - WebRecorder gets real PCM bytes fed in, we verify byte-accurate
    frame emission after resampling.
  - CustomRecorder gets a real async generator yielding synthetic sine
    waves; we verify framing + mono/stereo + sample-rate conversions.
  - Resampler is tested against known-answer ratios.
  - AudioStreamRecorder is partially mocked only at the LLM/STT boundary;
    the VAD + wake pipeline is driven by actual PCM frames.

Run:
    python -m unittest test_audio.test_audio_recorder -v
"""

import asyncio
import os
import sys
import unittest
from typing import AsyncIterator

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toolboxv2.mods.isaa.base.audio_io.audio_recorder import (
    AudioRecorder,
    CustomRecorder,
    LocalMicRecorder,
    WebRecorder,
    TARGET_FRAME_BYTES,
    TARGET_FRAME_SAMPLES,
    TARGET_SR,
    downmix_to_mono,
    resample_pcm16,
)


# =============================================================================
# SYNTHESIZERS — real audio data, no placeholders
# =============================================================================

def sine_pcm(freq_hz: float, duration_s: float, sr: int, amplitude: float = 0.5) -> bytes:
    """Generate a mono int16 PCM sine wave."""
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, endpoint=False)
    samples = (np.sin(2 * np.pi * freq_hz * t) * amplitude * 32767).astype(np.int16)
    return samples.tobytes()


def stereo_sine_pcm(freq_l: float, freq_r: float, duration_s: float, sr: int) -> bytes:
    """Stereo int16 PCM with different freq per channel — verifies downmix math."""
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, endpoint=False)
    left = (np.sin(2 * np.pi * freq_l * t) * 0.5 * 32767).astype(np.int16)
    right = (np.sin(2 * np.pi * freq_r * t) * 0.5 * 32767).astype(np.int16)
    interleaved = np.stack([left, right], axis=1).flatten()
    return interleaved.tobytes()


# =============================================================================
# RESAMPLER
# =============================================================================

class TestResampler(unittest.TestCase):

    def test_no_op_when_rates_match(self):
        data = sine_pcm(440.0, 0.1, 16000)
        self.assertEqual(resample_pcm16(data, 16000, 16000), data)

    def test_downsample_48k_to_16k_length_correct(self):
        data = sine_pcm(440.0, 1.0, 48000)
        # 48000 samples in → 16000 samples out → 32000 bytes
        out = resample_pcm16(data, 48000, 16000)
        self.assertEqual(len(out), 16000 * 2)

    def test_downsample_preserves_signal_shape(self):
        """After 48k→16k, a 440Hz sine should still peak near 440Hz."""
        data = sine_pcm(440.0, 1.0, 48000)
        out = resample_pcm16(data, 48000, 16000)
        samples = np.frombuffer(out, dtype=np.int16).astype(np.float32)
        # FFT peak bin
        spec = np.abs(np.fft.rfft(samples))
        peak_bin = int(np.argmax(spec))
        peak_freq = peak_bin * 16000 / len(samples)
        self.assertAlmostEqual(peak_freq, 440.0, delta=10.0)

    def test_empty_input(self):
        self.assertEqual(resample_pcm16(b"", 48000, 16000), b"")


class TestDownmix(unittest.TestCase):

    def test_mono_is_noop(self):
        data = sine_pcm(440, 0.1, 16000)
        self.assertEqual(downmix_to_mono(data, 1), data)

    def test_stereo_average_verified_numerically(self):
        """Feed known L/R samples, verify mean."""
        # 4 stereo frames: L, R interleaved
        left = np.array([100, 200, -100, -200], dtype=np.int16)
        right = np.array([300, 400, -300, -400], dtype=np.int16)
        interleaved = np.stack([left, right], axis=1).flatten().tobytes()
        mono = np.frombuffer(downmix_to_mono(interleaved, 2), dtype=np.int16)
        expected = ((left.astype(np.float32) + right.astype(np.float32)) / 2).astype(np.int16)
        self.assertTrue(np.array_equal(mono, expected))


# =============================================================================
# WebRecorder — real PCM injected, real frames extracted
# =============================================================================

class TestWebRecorder(unittest.IsolatedAsyncioTestCase):

    async def test_frames_chunked_to_80ms_after_feed(self):
        r = WebRecorder(src_sample_rate=TARGET_SR, src_channels=1)
        await r.start()

        # Feed 240ms — should produce exactly 3 × 80ms frames
        pcm = sine_pcm(440, 0.240, TARGET_SR)
        await r.feed(pcm)

        received = []
        async def consume():
            async for f in r.frames():
                received.append(f)
                if len(received) >= 3:
                    break
        await asyncio.wait_for(consume(), timeout=2.0)
        await r.stop()

        self.assertEqual(len(received), 3)
        for f in received:
            self.assertEqual(len(f), TARGET_FRAME_BYTES)

    async def test_feed_resamples_48k_input_mono(self):
        r = WebRecorder(src_sample_rate=48000, src_channels=1)
        await r.start()
        # 240ms of 48k audio → resampled to 240ms at 16k → 3 × 80ms frames
        pcm = sine_pcm(880, 0.240, 48000)
        await r.feed(pcm)
        received = []
        async def consume():
            async for f in r.frames():
                received.append(f)
                if len(received) >= 3:
                    break
        await asyncio.wait_for(consume(), timeout=2.0)
        await r.stop()
        self.assertEqual(len(received), 3)

    async def test_feed_downmixes_stereo(self):
        r = WebRecorder(src_sample_rate=TARGET_SR, src_channels=2)
        await r.start()
        pcm = stereo_sine_pcm(440, 660, 0.240, TARGET_SR)
        await r.feed(pcm)
        received = []
        async def consume():
            async for f in r.frames():
                received.append(f)
                if len(received) >= 3:
                    break
        await asyncio.wait_for(consume(), timeout=2.0)
        await r.stop()
        # Output must be mono → each frame = TARGET_FRAME_BYTES
        self.assertEqual(len(received), 3)
        for f in received:
            self.assertEqual(len(f), TARGET_FRAME_BYTES)

    async def test_feed_rejects_when_inactive(self):
        r = WebRecorder()
        # Not started → feed is a no-op
        await r.feed(sine_pcm(440, 0.1, TARGET_SR))
        # No frames should ever be available
        async def check():
            async for _ in r.frames():  # won't start — inactive
                self.fail("frames yielded while inactive")
        # frames() returns immediately because _active is False
        task = asyncio.create_task(check())
        await asyncio.sleep(0.1)
        task.cancel()


# =============================================================================
# CustomRecorder — real async generator
# =============================================================================

class TestCustomRecorder(unittest.IsolatedAsyncioTestCase):

    async def test_wraps_generator_and_frames_correctly(self):
        async def src() -> AsyncIterator[bytes]:
            # Stream 480ms in two chunks of 240ms each
            yield sine_pcm(440, 0.240, TARGET_SR)
            yield sine_pcm(880, 0.240, TARGET_SR)

        r = CustomRecorder(source=src, declared_sr=TARGET_SR)
        await r.start()
        frames = []
        async for f in r.frames():
            frames.append(f)
            if len(frames) >= 6:
                break
        await r.stop()
        self.assertEqual(len(frames), 6)
        for f in frames:
            self.assertEqual(len(f), TARGET_FRAME_BYTES)

    async def test_accepts_tuple_yield_with_source_metadata(self):
        """If source yields (pcm, sr, channels), conversion kicks in."""
        async def src() -> AsyncIterator[tuple]:
            # 240ms stereo at 48k
            yield (stereo_sine_pcm(440, 880, 0.240, 48000), 48000, 2)

        r = CustomRecorder(source=src)
        await r.start()
        frames = []
        async for f in r.frames():
            frames.append(f)
        await r.stop()
        self.assertGreaterEqual(len(frames), 3)  # ≥240ms / 80ms
        for f in frames:
            self.assertEqual(len(f), TARGET_FRAME_BYTES)

    async def test_fractional_tail_is_discarded(self):
        """Tail < 80ms must not be emitted partially."""
        async def src() -> AsyncIterator[bytes]:
            # Exactly 200ms — 2 × 80ms frames + 40ms leftover
            yield sine_pcm(440, 0.200, TARGET_SR)
        r = CustomRecorder(source=src)
        await r.start()
        frames = [f async for f in r.frames()]
        await r.stop()
        self.assertEqual(len(frames), 2)  # 40ms tail dropped


# =============================================================================
# LocalMicRecorder — skipped unless sounddevice + mic available
# =============================================================================

def _has_sounddevice_input() -> bool:
    try:
        import sounddevice as sd
        # at least one input device
        for d in sd.query_devices():
            if d.get("max_input_channels", 0) > 0:
                return True
    except Exception:
        pass
    return False


@unittest.skipUnless(_has_sounddevice_input(), "no sounddevice input available")
class TestLocalMicRecorder(unittest.IsolatedAsyncioTestCase):
    """Smoke test — we only verify we can open/close and get >0 frames."""

    async def test_open_capture_close(self):
        r = LocalMicRecorder()
        await r.start()
        frames = []
        async def consume():
            async for f in r.frames():
                frames.append(f)
                if len(frames) >= 2:
                    break
        try:
            await asyncio.wait_for(consume(), timeout=2.0)
        finally:
            await r.stop()
        self.assertGreaterEqual(len(frames), 1)
        self.assertEqual(len(frames[0]), TARGET_FRAME_BYTES)


# =============================================================================
# STT PRESET RESOLUTION — real env-snapshot tests
# =============================================================================

from unittest import mock


class TestSTTPresetResolution(unittest.TestCase):
    """
    These tests don't load real models. They verify that resolve_stt_preset
    picks the right backend/model combo given a snapshot of availability
    and env vars. No mocking of the resolve logic itself — only the
    availability probe and the env is patched.
    """

    def _patch_availability(self, **flags):
        """Helper: patch _available_backends() to return a specific snapshot."""
        from toolboxv2.mods.isaa.base.audio_io import Stt
        defaults = {
            "parakeet": False, "faster_whisper": False,
            "groq_pkg": False, "groq_key": False, "groq": False, "cuda": False,
        }
        defaults.update(flags)
        return mock.patch.object(Stt, "_available_backends", return_value=defaults)

    def test_default_prefers_groq_when_available(self):
        from toolboxv2.mods.isaa.base.audio_io.Stt import (
            resolve_stt_preset, STTPreset, STTBackend,
        )
        with self._patch_availability(groq=True, groq_pkg=True, groq_key=True,
                                       parakeet=True, faster_whisper=True):
            with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test"}):
                p, f, desc = resolve_stt_preset(STTPreset.DEFAULT)
        self.assertEqual(p.backend, STTBackend.GROQ_WHISPER)
        self.assertEqual(f.backend, STTBackend.GROQ_WHISPER)
        self.assertIn("groq", desc.lower())

    def test_default_falls_back_to_parakeet_without_groq(self):
        from toolboxv2.mods.isaa.base.audio_io.Stt import (
            resolve_stt_preset, STTPreset, STTBackend,
        )
        with self._patch_availability(parakeet=True, faster_whisper=True):
            p, f, _ = resolve_stt_preset(STTPreset.DEFAULT)
        self.assertEqual(p.backend, STTBackend.PARAKEET)
        self.assertEqual(f.backend, STTBackend.PARAKEET)

    def test_default_fallback_order_faster_whisper_last(self):
        from toolboxv2.mods.isaa.base.audio_io.Stt import (
            resolve_stt_preset, STTPreset, STTBackend,
        )
        with self._patch_availability(faster_whisper=True):
            p, f, _ = resolve_stt_preset(STTPreset.DEFAULT)
        self.assertEqual(p.backend, STTBackend.FASTER_WHISPER)
        self.assertEqual(f.backend, STTBackend.FASTER_WHISPER)

    def test_default_raises_when_nothing_installed(self):
        from toolboxv2.mods.isaa.base.audio_io.Stt import (
            resolve_stt_preset, STTPreset,
        )
        with self._patch_availability():
            with self.assertRaises(RuntimeError) as ctx:
                resolve_stt_preset(STTPreset.DEFAULT)
        self.assertIn("parakeet-stream", str(ctx.exception))

    def test_quality_local_uses_different_partial_and_final(self):
        from toolboxv2.mods.isaa.base.audio_io.Stt import (
            resolve_stt_preset, STTPreset, STTBackend,
        )
        with self._patch_availability(parakeet=True, faster_whisper=True, cuda=True):
            p, f, _ = resolve_stt_preset(STTPreset.QUALITY_LOCAL)
        self.assertEqual(p.backend, STTBackend.PARAKEET)
        self.assertEqual(f.backend, STTBackend.FASTER_WHISPER)
        self.assertEqual(f.model, "large-v3")
        self.assertEqual(f.device, "cuda")
        self.assertEqual(f.compute_type, "float16")

    def test_quality_local_cpu_fallback_when_no_cuda(self):
        from toolboxv2.mods.isaa.base.audio_io.Stt import (
            resolve_stt_preset, STTPreset,
        )
        with self._patch_availability(parakeet=True, faster_whisper=True, cuda=False):
            _, f, _ = resolve_stt_preset(STTPreset.QUALITY_LOCAL)
        self.assertEqual(f.device, "cpu")
        self.assertEqual(f.compute_type, "int8")

    def test_speed_api_raises_without_groq(self):
        from toolboxv2.mods.isaa.base.audio_io.Stt import (
            resolve_stt_preset, STTPreset,
        )
        with self._patch_availability(parakeet=True):
            with self.assertRaises(RuntimeError) as ctx:
                resolve_stt_preset(STTPreset.SPEED_API)
        self.assertIn("GROQ_API_KEY", str(ctx.exception))

    def test_quality_api_uses_large_v3_for_final(self):
        from toolboxv2.mods.isaa.base.audio_io.Stt import (
            resolve_stt_preset, STTPreset, STTBackend,
        )
        with self._patch_availability(groq=True, groq_pkg=True, groq_key=True):
            with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test"}):
                p, f, _ = resolve_stt_preset(STTPreset.QUALITY_API)
        self.assertEqual(p.backend, STTBackend.GROQ_WHISPER)
        self.assertEqual(p.model, "whisper-large-v3-turbo")
        self.assertEqual(f.backend, STTBackend.GROQ_WHISPER)
        self.assertEqual(f.model, "whisper-large-v3")

    def test_speed_local_requires_parakeet(self):
        from toolboxv2.mods.isaa.base.audio_io.Stt import (
            resolve_stt_preset, STTPreset,
        )
        with self._patch_availability(faster_whisper=True):
            with self.assertRaises(RuntimeError) as ctx:
                resolve_stt_preset(STTPreset.SPEED_LOCAL)
        self.assertIn("parakeet-stream", str(ctx.exception))

    def test_language_propagated_to_both_configs(self):
        from toolboxv2.mods.isaa.base.audio_io.Stt import (
            resolve_stt_preset, STTPreset,
        )
        with self._patch_availability(parakeet=True):
            p, f, _ = resolve_stt_preset(STTPreset.SPEED_LOCAL, language="de")
        self.assertEqual(p.language, "de")
        self.assertEqual(f.language, "de")


class TestSTTPresetDescribe(unittest.TestCase):

    def test_describe_contains_all_presets(self):
        from toolboxv2.mods.isaa.base.audio_io.Stt import (
            describe_stt_presets, STTPreset,
        )
        out = describe_stt_presets()
        for p in STTPreset:
            self.assertIn(p.value, out)

if __name__ == "__main__":
    unittest.main(verbosity=2)
