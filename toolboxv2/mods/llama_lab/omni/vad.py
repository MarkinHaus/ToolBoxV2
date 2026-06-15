# file: toolboxv2/mods/llama_lab/omni/vad.py
"""Tiny energy VAD for turn segmentation on PCM int16 16k mono.

No external deps. Emits 'speech_start' / 'speech_end' transitions with a
hangover so short pauses don't cut a turn, and a min-speech guard so clicks
don't trigger a turn. Tuned for low latency (20 ms frames).
"""

import math
from array import array


class VAD:
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 20,
                 threshold: float = 450.0, hangover_ms: int = 500,
                 min_speech_ms: int = 160):
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2
        self.threshold = threshold
        self.hangover_frames = max(1, hangover_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self._buf = bytearray()
        self._in_speech = False
        self._silence = 0
        self._speech = 0

    @staticmethod
    def _rms(frame: bytes) -> float:
        if not frame:
            return 0.0
        s = array("h")
        s.frombytes(frame)
        if not len(s):
            return 0.0
        return math.sqrt(sum(x * x for x in s) / len(s))

    def feed(self, pcm: bytes):
        """Yield ('speech_start'|'speech_end') transitions for the frames in pcm."""
        self._buf += pcm
        while len(self._buf) >= self.frame_bytes:
            frame = bytes(self._buf[:self.frame_bytes])
            del self._buf[:self.frame_bytes]
            voiced = self._rms(frame) >= self.threshold
            if voiced:
                self._speech += 1
                self._silence = 0
                if not self._in_speech and self._speech >= self.min_speech_frames:
                    self._in_speech = True
                    yield "speech_start"
            else:
                if self._in_speech:
                    self._silence += 1
                    if self._silence >= self.hangover_frames:
                        self._in_speech = False
                        self._speech = 0
                        yield "speech_end"
                else:
                    self._speech = 0

    def reset(self):
        self._buf.clear()
        self._in_speech = False
        self._silence = self._speech = 0
