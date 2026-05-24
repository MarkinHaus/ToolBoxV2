"""Minimal streaming throughput tracker. Zero-alloc hot path."""

import time
from dataclasses import dataclass


@dataclass
class StreamMetrics:
    """Filled during streaming, read after."""
    t_start: float = 0.0
    t_first_token: float = 0.0
    t_end: float = 0.0
    chunk_count: int = 0
    token_count: int = 0  # from final usage chunk, not estimated

    @property
    def ttft_ms(self) -> float:
        """Time to first token in ms."""
        if not self.t_first_token:
            return 0.0
        return (self.t_first_token - self.t_start) * 1000

    @property
    def tps(self) -> float:
        """Tokens per second (output). Uses actual token count from provider."""
        duration = self.t_end - (self.t_first_token or self.t_start)
        if duration <= 0 or self.token_count <= 0:
            return 0.0
        return self.token_count / duration

    @property
    def total_ms(self) -> float:
        if not self.t_end:
            return 0.0
        return (self.t_end - self.t_start) * 1000

    def summary(self) -> dict:
        return {
            "ttft_ms": round(self.ttft_ms, 1),
            "tps": round(self.tps, 1),
            "total_ms": round(self.total_ms, 1),
            "chunks": self.chunk_count,
            "output_tokens": self.token_count,
        }
