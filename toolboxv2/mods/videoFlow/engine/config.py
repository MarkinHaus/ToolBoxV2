# toolboxv2/mods/videoFlow/engine/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

@dataclass
class CostTracker:
    """Comprehensive cost tracking for all APIs"""
    agent_cost: int = 0
    kokoro_calls: int = 0
    kokoro_cost: float = 0.0
    flux_schnell_calls: int = 0
    flux_schnell_cost: float = 0.0
    flux_krea_calls: int = 0
    flux_krea_cost: float = 0.0
    flux_kontext_calls: int = 0
    flux_kontext_cost: float = 0.0
    banana_calls: int = 0
    banana_cost: float = 0.0
    minimax_calls: int = 0  # New
    minimax_cost: float = 0.0  # New
    elevenlabs_calls = 0
    elevenlabs_tokens = 0
    elevenlabs_cost = 0

    # Cost per call
    COSTS = {
        'kokoro': 0.002,  # Per audio segment
        'flux_schnell': 0.003,  # Per image
        'flux_krea': 0.025,  # Per image
        'flux_kontext': 0.04,  # Per image with reference
        'banana': 0.039,  # Per edit
        'minimax': 0.017 # Per second
    }

    def add_elevenlabs_cost(self, char_count: int):
        """Add ElevenLabs TTS cost"""
        self.elevenlabs_calls += 1
        self.elevenlabs_tokens += char_count
        self.elevenlabs_cost += (char_count / 1000) * 0.3


    def add_minimax_cost(self, calls: int = 1, second:int=5):
        """Add Minimax video generation cost"""
        self.minimax_calls += calls
        self.minimax_cost += second * self.COSTS['minimax']

    def add_kokoro_cost(self, calls: int = 1):
        self.kokoro_calls += calls
        self.kokoro_cost += calls * self.COSTS['kokoro']

    def add_flux_schnell_cost(self, calls: int = 1):
        self.flux_schnell_calls += calls
        self.flux_schnell_cost += calls * self.COSTS['flux_schnell']

    def add_flux_krea_cost(self, calls: int = 1):
        self.flux_krea_calls += calls
        self.flux_krea_cost += calls * self.COSTS['flux_krea']

    def add_flux_kontext_cost(self, calls: int = 1):
        self.flux_kontext_calls += calls
        self.flux_kontext_cost += calls * self.COSTS['flux_kontext']

    def add_banana_cost(self, calls: int = 1):
        self.banana_calls += calls
        self.banana_cost += calls * self.COSTS['banana']

    @property
    def total_cost(self) -> float:
        return (self.agent_cost + self.kokoro_cost + self.flux_schnell_cost +
                self.flux_krea_cost + self.minimax_cost + self.flux_kontext_cost + self.banana_cost)

    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_cost_usd': round(self.total_cost, 3),
            'breakdown': {
                'agent': {'calls': 1, 'cost': round(self.agent_cost, 3)},
                'kokoro': {'calls': self.kokoro_calls, 'cost': round(self.kokoro_cost, 3)},
                'flux_schnell': {'calls': self.flux_schnell_calls, 'cost': round(self.flux_schnell_cost, 3)},
                'flux_krea': {'calls': self.flux_krea_calls, 'cost': round(self.flux_krea_cost, 3)},
                'flux_kontext': {'calls': self.flux_kontext_calls, 'cost': round(self.flux_kontext_cost, 3)},
                'banana': {'calls': self.banana_calls, 'cost': round(self.banana_cost, 3)},
                'minimax': {'calls': self.minimax_calls, 'cost': round(self.minimax_cost, 3)},
                'elevenlabs': {'calls': self.elevenlabs_calls, 'cost': round(self.elevenlabs_cost, 3), 'tokens': self.elevenlabs_tokens}
            }
        }

    @classmethod
    def from_summary(cls, summary: Dict[str, Any]) -> 'CostTracker':
        cost_tracker = cls()
        if 'breakdown' not in summary:
            return cost_tracker
        cost_tracker.agent_cost = summary['breakdown']['agent']['cost']
        cost_tracker.kokoro_calls = summary['breakdown']['kokoro']['calls']
        cost_tracker.kokoro_cost = summary['breakdown']['kokoro']['cost']
        cost_tracker.flux_schnell_calls = summary['breakdown']['flux_schnell']['calls']
        cost_tracker.flux_schnell_cost = summary['breakdown']['flux_schnell']['cost']
        cost_tracker.flux_krea_calls = summary['breakdown']['flux_krea']['calls']
        cost_tracker.flux_krea_cost = summary['breakdown']['flux_krea']['cost']
        cost_tracker.flux_kontext_calls = summary['breakdown']['flux_kontext']['calls']
        cost_tracker.flux_kontext_cost = summary['breakdown']['flux_kontext']['cost']
        cost_tracker.banana_calls = summary['breakdown']['banana']['calls']
        cost_tracker.banana_cost = summary['breakdown']['banana']['cost']
        cost_tracker.minimax_calls = summary['breakdown']['minimax']['calls']
        cost_tracker.minimax_cost = summary['breakdown']['minimax']['cost']
        cost_tracker.elevenlabs_chars = summary['breakdown']['elevenlabs']['calls']
        cost_tracker.elevenlabs_cost = summary['breakdown']['elevenlabs']['cost']
        cost_tracker.elevenlabs_tokens = summary['breakdown']['elevenlabs']['elevenlabs_tokens']
        return cost_tracker

class Config:
    """Production configuration"""
    BASE_OUTPUT_DIR = Path("./generated_stories")
    IMAGE_SIZE = "landscape_4_3"
    VIDEO_FPS = 30
    SCENE_TRANSITION = 1.0  # seconds

    # Kokoro TTS settings
    KOKORO_MODELS_DIR = Path.cwd() / "kokoro_models"

    # FAL API models
    FLUX_SCHNELL = "fal-ai/flux/schnell"
    FLUX_KREA = "fal-ai/flux/krea"
    FLUX_KONTEXT = "fal-ai/flux-pro/kontext"
    BANANA_EDIT = "fal-ai/nano-banana/edit"
