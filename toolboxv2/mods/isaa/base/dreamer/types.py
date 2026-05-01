"""
Dreamer V3 — Shared Types

DreamConfig lives here so both FlowAgent and tool_handler can import it
without circular dependencies.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DreamConfig:
    """Configuration for a dream cycle."""
    max_budget: int = int(os.getenv("DREAMER_BUDGET", "160000"))
    max_history_time: Optional[float] = None
    do_skill_split: bool = True
    do_skill_evolve: bool = True
    do_persona_evolve: bool = True
    do_create_new: bool = True
    hard_stop: bool = False
    publish_threshold: float = 0.8
    publish_min_version: int = 3
