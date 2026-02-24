"""AgentLiveState - Minimal live state for ExecutionEngine introspection.

Option A (property getter) + slim B (direct attrs). ~50 lines, not 500.
Read-only from outside, written by engine internals.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("isaa.engine.live")


class AgentPhase(str, Enum):
    IDLE = "idle"
    INIT = "init"
    LLM_CALL = "llm_call"
    TOOL_EXEC = "tool_exec"
    COMPRESSING = "compressing"
    FINALIZING = "finalizing"
    DONE = "done"
    ERROR = "error"
    PAUSED = "paused"


@dataclass(slots=True)
class ToolExecution:
    name: str = ""
    args_summary: str = ""
    t_start: float = 0.0


@dataclass(slots=True)
class TokenStream:
    content: str = ""
    reasoning: str = ""
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class AgentLiveState:
    """Lightweight mutable state bag. Engine writes, renderer reads."""

    phase: AgentPhase = AgentPhase.IDLE
    run_id: str = ""
    agent_name: str = ""
    iteration: int = 0
    max_iterations: int = 0
    is_sub: bool = False
    tool: ToolExecution = field(default_factory=ToolExecution)
    stream: TokenStream = field(default_factory=TokenStream)
    error: str = ""
    t_start: float = 0.0
    # -- visible to renderer (the user MUST see these) -------------------------
    thought: str = ""           # current agent thought / focus
    status_msg: str = ""        # one-liner: "Loading tools...", "LLM Error: ..."
    skills: list = field(default_factory=list)   # matched skill names
    tools_loaded: list = field(default_factory=list)  # currently loaded tool names
    persona: str = ""

    # -- convenience ----------------------------------------------------------
    def reset(self):
        self.phase = AgentPhase.IDLE
        self.iteration = 0
        self.stream = TokenStream()
        self.tool = ToolExecution()
        self.error = ""
        self.thought = ""
        self.status_msg = ""
        self.persona = ""
        self.t_start = 0.0

    @property
    def elapsed(self) -> float:
        return round(time.time() - self.t_start, 2) if self.t_start else 0.0

    # -- engine helpers (replaces every print) --------------------------------
    def log(self, msg: str, level: int = logging.DEBUG):
        logger.log(level, "[%s|%s/%s] %s", self.agent_name, self.iteration, self.max_iterations, msg)

    def enter(self, phase: AgentPhase, msg: str = ""):
        self.phase = phase
        if msg:
            self.log(msg)
