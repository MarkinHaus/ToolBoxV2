"""
audio_recorder.py
=================

Abstract input side for the audio stack. Symmetric to AudioPlayer.

All recorders yield a uniform frame format:
    PCM int16, mono, 16000 Hz, ~80 ms per frame (1280 samples, 2560 bytes).
This matches what openWakeWord and Silero VAD consume — callers don't have
to re-chunk or resample.

Backends:
    LocalMicRecorder  — sounddevice.InputStream from the system mic
    WebRecorder       — asyncio.Queue fed by a web-socket handler
    CustomRecorder    — wraps an async iterator; used for file replay,
                        unittest fixtures, Discord voice-recv, phone calls,
                        YouTube ingest — any non-standard source.

A CustomRecorder accepts an optional resampler so the caller can feed
PCM at any sample rate; we downmix and resample to the uniform 16k mono
int16 contract before yielding.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

TARGET_SR = 16000
TARGET_FRAME_MS = 80
TARGET_FRAME_SAMPLES = TARGET_SR * TARGET_FRAME_MS // 1000   # 1280
TARGET_FRAME_BYTES = TARGET_FRAME_SAMPLES * 2                # int16 mono


# ---------------------------------------------------------------------------
# Resampler (linear; good enough for STT input — STT is robust to minor
# interpolation artifacts; for audiophile use swap with soxr/scipy later)
# ---------------------------------------------------------------------------

def resample_pcm16(pcm: bytes, src_sr: int, dst_sr: int = TARGET_SR) -> bytes:
    """
    Linear resample of int16 mono PCM bytes. No-op if src_sr == dst_sr.

    Not the highest-quality resampler — but fast, zero extra deps, and
    adequate for speech recognition. Swap for scipy.signal.resample_poly
    if you want sub-percent WER improvements.
    """
    if src_sr == dst_sr:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return b""
    duration = samples.size / src_sr
    dst_n = int(duration * dst_sr)
    if dst_n <= 0:
        return b""
    x_src = np.linspace(0, duration, num=samples.size, endpoint=False)
    x_dst = np.linspace(0, duration, num=dst_n,        endpoint=False)
    out = np.interp(x_dst, x_src, samples).astype(np.int16)
    return out.tobytes()


def downmix_to_mono(pcm: bytes, channels: int) -> bytes:
    """Channel downmix to mono by averaging. No-op for mono input."""
    if channels == 1:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).reshape(-1, channels)
    mono = samples.mean(axis=1).astype(np.int16)
    return mono.tobytes()


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class AudioRecorder(ABC):
    """
    Abstract base for audio input backends.

    Contract:
        await recorder.start()                 # open resources
        async for frame in recorder.frames():  # yields ~80ms int16 16k mono
            ...
        await recorder.stop()                  # teardown

    Subclasses MUST emit frames in the uniform format (TARGET_* constants).
    Non-matching formats are the subclass's responsibility to convert —
    see resample_pcm16 / downmix_to_mono helpers.
    """

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    def frames(self) -> AsyncIterator[bytes]:
        """Async iterator of PCM int16 mono 16kHz frames (~80ms each)."""

    @property
    @abstractmethod
    def is_active(self) -> bool: ...

    @property
    def recorder_type(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# LocalMicRecorder — sounddevice
# ---------------------------------------------------------------------------

class LocalMicRecorder(AudioRecorder):
    """
    Records from the system microphone via sounddevice InputStream.

    sounddevice is opened in float32 and converted internally to int16
    to match the uniform contract.
    """

    def __init__(
        self,
        device: Optional[int] = None,
        sample_rate: int = TARGET_SR,
        frame_ms: int = TARGET_FRAME_MS,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=128)
        self._stream = None
        self._active = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as e:
            raise ImportError("sounddevice required: pip install sounddevice") from e

        self._loop = asyncio.get_event_loop()
        frame_samples = int(self.sample_rate * self.frame_ms / 1000)

        def _cb(indata, frames, time_info, status):
            if status:
                logger.debug("LocalMicRecorder: sd status %s", status)
            pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            if self.sample_rate != TARGET_SR:
                pcm = resample_pcm16(pcm, self.sample_rate, TARGET_SR)
            try:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, pcm)
            except asyncio.QueueFull:
                logger.warning("LocalMicRecorder queue full — dropping frame")

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frame_samples,
            device=self.device,
            callback=_cb,
        )
        self._stream.start()
        self._active = True

    async def stop(self) -> None:
        self._active = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def frames(self) -> AsyncIterator[bytes]:
        while self._active:
            try:
                frame = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield frame
            except asyncio.TimeoutError:
                continue

    @property
    def is_active(self) -> bool:
        return self._active


# ---------------------------------------------------------------------------
# WebRecorder — external feeder (WebSocket)
# ---------------------------------------------------------------------------

class WebRecorder(AudioRecorder):
    """
    Receives PCM frames from an external source (typically an icli_web
    WebSocket handler) via the public `feed()` method.

    Expected incoming format: PCM int16. If src_sr != 16000 or channels != 1,
    the recorder resamples/downmixes on ingest.

    Typical browser side (hint, not part of this class):
        AudioWorklet captures Float32 → converts to Int16 → WS.send(buffer)
    """

    def __init__(
        self,
        src_sample_rate: int = TARGET_SR,
        src_channels: int = 1,
        queue_maxsize: int = 256,
    ):
        self.src_sample_rate = src_sample_rate
        self.src_channels = src_channels
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_maxsize)
        self._active = False

    async def start(self) -> None:
        self._active = True

    async def stop(self) -> None:
        self._active = False
        # Drain queue so stop + restart doesn't replay stale frames.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def feed(self, pcm: bytes) -> None:
        """Public entry point for the WebSocket handler."""
        if not self._active:
            return
        if self.src_channels != 1:
            pcm = downmix_to_mono(pcm, self.src_channels)
        if self.src_sample_rate != TARGET_SR:
            pcm = resample_pcm16(pcm, self.src_sample_rate, TARGET_SR)
        # Normalize to exact 80ms frames so downstream VAD chunking is clean.
        for i in range(0, len(pcm) - TARGET_FRAME_BYTES + 1, TARGET_FRAME_BYTES):
            try:
                self._queue.put_nowait(pcm[i:i + TARGET_FRAME_BYTES])
            except asyncio.QueueFull:
                logger.warning("WebRecorder queue full — dropping frame")

    async def frames(self) -> AsyncIterator[bytes]:
        while self._active:
            try:
                f = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield f
            except asyncio.TimeoutError:
                continue

    @property
    def is_active(self) -> bool:
        return self._active


# ---------------------------------------------------------------------------
# CustomRecorder — generic adapter for anything async
# ---------------------------------------------------------------------------

class CustomRecorder(AudioRecorder):
    """
    Generic recorder wrapping an async source callable.

    Use for: file replay, Discord voice-recv, SIP/phone, YouTube ingest,
    or any future non-standard audio source.

    Args:
        source: async generator function — called on .start(), must yield
                (pcm_bytes, sample_rate, channels) tuples OR plain pcm bytes
                in the uniform 16k/mono/int16 format.
        declared_sr: when `source` yields plain bytes, what sample rate
                     they come at. Defaults to TARGET_SR.
        declared_channels: same, for channel count.
    """

    def __init__(
        self,
        source: Callable[[], AsyncIterator[Any]],
        declared_sr: int = TARGET_SR,
        declared_channels: int = 1,
    ):
        self._source_factory = source
        self.declared_sr = declared_sr
        self.declared_channels = declared_channels
        self._active = False
        self._gen: Optional[AsyncIterator[Any]] = None

    async def start(self) -> None:
        self._gen = self._source_factory()
        self._active = True

    async def stop(self) -> None:
        self._active = False
        if self._gen is not None and hasattr(self._gen, "aclose"):
            try:
                await self._gen.aclose()
            except Exception as e:
                logger.debug("CustomRecorder: source aclose raised %s", e)
        self._gen = None

    async def frames(self) -> AsyncIterator[bytes]:
        if self._gen is None:
            return
        buffer = b""
        async for item in self._gen:
            if not self._active:
                break
            if isinstance(item, tuple):
                pcm, sr, ch = item
            else:
                pcm, sr, ch = item, self.declared_sr, self.declared_channels
            if ch != 1:
                pcm = downmix_to_mono(pcm, ch)
            if sr != TARGET_SR:
                pcm = resample_pcm16(pcm, sr, TARGET_SR)
            buffer += pcm
            while len(buffer) >= TARGET_FRAME_BYTES:
                yield buffer[:TARGET_FRAME_BYTES]
                buffer = buffer[TARGET_FRAME_BYTES:]
        # Tail: discard fractional trailing frame — downstream would mis-chunk
        # if it were yielded unpadded.

    @property
    def is_active(self) -> bool:
        return self._active
