"""
AgentLiveNarrator
=================
Generates short live-status text for AgentLiveState.thought via a fast
"Blitz" LLM (e.g. groq/llama-3.1-8b-instant).

Activation:
    BLITZ_MODEL=groq/llama-3.1-8b-instant   # required – if empty, narrator is off
    AGENT_NARRATOR_ENABLED=true              # default true
    NARRATOR_LANG=auto                       # auto | de | en

Design rules
------------
* Fire-and-forget via asyncio.create_task – never blocks the engine.
* Cancel-pending: a new task cancels the old one unless the old is a
  think-triggered task (highest priority).
* Rate-limit: min 0.75 s between Blitz calls; soft token budget per run.
* Mock strings are set *synchronously before* any async call so the UI
  always has something to show immediately.
* Think-tool path: Blitz gets the full thinking chunk → builds / updates
  NarratorMiniState (plan, drift, repeat) → used as context in all future
  calls of this run.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config / Constants
# ---------------------------------------------------------------------------

BLITZ_MODEL: str = os.getenv("BLITZMODEL", "")
NARRATOR_ENABLED: bool = os.getenv("AGENT_NARRATOR_ENABLED", "true").lower() == "true"
NARRATOR_LANG: str = os.getenv("NARRATOR_LANG", "auto")  # auto | de | en

# Token budget per minute (sliding window, matches Groq free-tier limits)
NARRATOR_MAX_INPUT_TOKENS_PM: int = int(os.getenv("NARRATOR_MAX_INPUT_TPM", "3000"))
NARRATOR_MAX_OUTPUT_TOKENS_PM: int = int(os.getenv("NARRATOR_MAX_OUTPUT_TPM", "200"))
_BUDGET_WINDOW: float = 60.0  # seconds

# Timing
NARRATOR_MIN_INTERVAL: float = 0.75   # seconds between blitz calls
NARRATOR_STALE_THRESHOLD: float = 0.3  # if live.thought is fresher than this, discard blitz result

# Diff compression: keep only first N chars of each message content
DIFF_CONTENT_MAXCHARS: int = 140
THINK_CONTENT_MAXCHARS: int = 600

# Blitz prompt tokens (rough estimate per call for budget tracking)
BLITZ_APPROX_SYSTEM_TOKENS: int = 80
BLITZ_APPROX_PER_MSG: int = 25

# ---------------------------------------------------------------------------
# Mock string tables  (DE + EN, with {tool} placeholder where needed)
# ---------------------------------------------------------------------------

_MOCKS: dict[str, dict[str, list[str]]] = {
    "init": {
        "de": [
            "analysiere anfrage …",
            "starte verarbeitung …",
            "bereite mich vor …",
            "initialisiere lauf …",
        ],
        "en": [
            "analysing request …",
            "starting up …",
            "getting ready …",
            "initialising run …",
        ],
    },
    "llm_pre": {
        "de": [
            "denke nach …",
            "formuliere nächsten schritt …",
            "überlege weiter …",
            "plane vorgehen …",
        ],
        "en": [
            "thinking …",
            "planning next step …",
            "reasoning …",
            "considering options …",
        ],
    },
    "tool_start": {
        "de": [
            "starte {tool} …",
            "rufe {tool} auf …",
            "führe {tool} aus …",
            "wende {tool} an …",
        ],
        "en": [
            "running {tool} …",
            "calling {tool} …",
            "executing {tool} …",
            "invoking {tool} …",
        ],
    },
    "tool_end": {
        "de": [
            "werte ergebnis aus …",
            "verarbeite antwort …",
            "lass mich das zusammenfassen …",
            "prüfe resultat …",
        ],
        "en": [
            "processing result …",
            "evaluating output …",
            "let me summarise …",
            "checking result …",
        ],
    },
    "think": {
        "de": [
            "denke gründlich nach …",
            "analysiere problem …",
            "durchdenke lösung …",
            "erarbeite strategie …",
        ],
        "en": [
            "thinking deeply …",
            "analysing problem …",
            "working through solution …",
            "elaborating strategy …",
        ],
    },
    "summarise": {
        "de": [
            "lass mich das zusammenfassen …",
            "bereite abschlussbericht vor …",
            "fasse ergebnisse zusammen …",
        ],
        "en": [
            "let me summarise …",
            "preparing final answer …",
            "wrapping up results …",
        ],
    },
    # Appended to thought when mini-state flags are set
    "drift": {
        "de": " ⚠ plan-abweichung",
        "en": " ⚠ drift detected",
    },
    "repeat": {
        "de": " ↩ wiederholung",
        "en": " ↩ repetition",
    },
}

# ---------------------------------------------------------------------------
# NarratorMiniState – built from think-tool results
# ---------------------------------------------------------------------------

@dataclass
class NarratorMiniState:
    """
    Compact state the narrator maintains across iterations of one run.
    Updated exclusively from think-tool Blitz calls.
    """
    plan_summary: str = ""   # ≤ 1 sentence: what the agent intends to do
    drift: bool = False      # Blitz detected deviation from plan
    repeat: bool = False     # Blitz detected repetition
    last_think_hash: str = ""  # dedup: skip if identical thinking chunk


# ---------------------------------------------------------------------------
# Helper: language detection
# ---------------------------------------------------------------------------

def _lang(query: str = "") -> str:
    """Return 'de' or 'en'."""
    if NARRATOR_LANG in ("de", "en"):
        return NARRATOR_LANG
    # simple heuristic: count German stop-words
    de_words = {"ich", "du", "ist", "das", "die", "der", "und", "was", "wie",
                "nicht", "mit", "auf", "für", "von"}
    tokens = set(query.lower().split())
    return "de" if len(tokens & de_words) >= 2 else "en"


def _mock(key: str, lang: str, **fmt) -> str:
    pool = _MOCKS.get(key, {}).get(lang, _MOCKS.get(key, {}).get("de", ["…"]))
    s = random.choice(pool)
    return s.format(**fmt) if fmt else s


# ---------------------------------------------------------------------------
# History compression for diff
# ---------------------------------------------------------------------------

def _compress_diff(history: list[dict], cursor: int) -> list[dict]:
    """
    Return only new messages since cursor with content truncated.
    Skips system messages (loop-warning injections etc.).
    """
    new_msgs = history[cursor:]
    compressed = []
    for m in new_msgs:
        role = m.get("role", "")
        if role == "system":
            continue
        content = m.get("content") or ""
        if isinstance(content, list):
            # multipart – flatten to text
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        compressed.append({
            "role": role,
            "content": content[:DIFF_CONTENT_MAXCHARS],
        })
    return compressed


def _estimate_tokens(msgs: list[dict]) -> int:
    total_chars = sum(len(m.get("content", "")) for m in msgs)
    return BLITZ_APPROX_SYSTEM_TOKENS + len(msgs) * BLITZ_APPROX_PER_MSG + total_chars // 4


# ---------------------------------------------------------------------------
# Blitz prompt builders
# ---------------------------------------------------------------------------

_SYSTEM_NORMAL_DE = (
    "Du bist ein Agent-Monitor. Antworte NUR mit kompaktem JSON, kein Markdown:\n"
    '{"t":"<max 8 Wörter was agent jetzt tut>","d":0,"r":0}\n'
    "d=1 bei plan-abweichung, r=1 bei wiederholung. Kein anderes Feld."
)
_SYSTEM_NORMAL_EN = (
    "You are an agent monitor. Reply ONLY with compact JSON, no markdown:\n"
    '{"t":"<max 8 words what agent is doing now>","d":0,"r":0}\n'
    "d=1 if drifting from plan, r=1 if repeating. No other fields."
)

_SYSTEM_THINK_DE = (
    "Du bist ein Agent-Monitor. Der Agent hat gerade gedacht.\n"
    "Antworte NUR mit JSON:\n"
    '{"t":"<6-8 Wörter Zusammenfassung>","plan":"<1 Satz Kernplan>","d":0,"r":0}\n'
    "Kein Markdown, keine anderen Felder."
)
_SYSTEM_THINK_EN = (
    "You are an agent monitor. The agent just completed a think step.\n"
    "Reply ONLY with JSON:\n"
    '{"t":"<6-8 word summary>","plan":"<1 sentence core plan>","d":0,"r":0}\n'
    "No markdown, no other fields."
)


def _build_normal_prompt(
    lang: str,
    diff_msgs: list[dict],
    mini: NarratorMiniState,
    tool_hint: str = "",
) -> tuple[str, list[dict]]:
    system = _SYSTEM_NORMAL_DE if lang == "de" else _SYSTEM_NORMAL_EN
    parts = []
    if mini.plan_summary:
        label = "Plan" if lang == "en" else "Plan"
        parts.append(f"{label}: {mini.plan_summary}")
    if tool_hint:
        label = "Tool" if lang == "en" else "Tool"
        parts.append(f"{label}: {tool_hint}")
    if diff_msgs:
        label = "Recent" if lang == "en" else "Neu"
        parts.append(f"{label}:\n" + "\n".join(
            f"[{m['role']}] {m['content']}" for m in diff_msgs[-4:]
        ))
    user_content = "\n".join(parts) or ("." if lang == "en" else ".")
    return system, [{"role": "user", "content": user_content}]


def _build_think_prompt(
    lang: str,
    thinking_content: str,
    mini: NarratorMiniState,
) -> tuple[str, list[dict]]:
    system = _SYSTEM_THINK_DE if lang == "de" else _SYSTEM_THINK_EN
    label_prev = "Vorheriger Plan" if lang == "de" else "Previous plan"
    label_think = "Denken" if lang == "de" else "Thinking"
    parts = []
    if mini.plan_summary:
        parts.append(f"{label_prev}: {mini.plan_summary}")
    parts.append(f"{label_think}:\n{thinking_content[:THINK_CONTENT_MAXCHARS]}")
    return system, [{"role": "user", "content": "\n".join(parts)}]


# ---------------------------------------------------------------------------
# Core: blitz API call  (raw litellm, NOT through agent machinery)
# ---------------------------------------------------------------------------

async def _call_blitz(system: str, messages: list[dict]) -> dict | None:
    """
    Call BLITZ_MODEL via litellm directly.
    Returns parsed JSON dict or None on any error.
    """
    try:
        import litellm  # type: ignore
        response = await litellm.acompletion(
            model=BLITZ_MODEL,
            messages=[{"role": "system", "content": system}] + messages,
            max_tokens=40,
            temperature=0.3,
            stream=False,
            drop_params=True,
        )
        raw = response.choices[0].message.content or ""
        raw = raw.strip().strip("```json").strip("```").strip()
        data = json.loads(raw)
        usage = getattr(response, "usage", None)
        return {
            "data": data,
            "in": getattr(usage, "prompt_tokens", 0),
            "out": getattr(usage, "completion_tokens", 0),
        }
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.debug("Blitz call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# AgentLiveNarrator
# ---------------------------------------------------------------------------

class AgentLiveNarrator:
    """
    Attach to ExecutionEngine as ``self._narrator``.

    Usage pattern in engine::

        # synchronous set (immediate UI update)
        self._narrator.mock("tool_start", tool="calc")

        # async schedule (fire-and-forget, non-blocking)
        asyncio.create_task(self._narrator.on_tool_end(ctx))
    """

    def __init__(self, live: "AgentLiveState", agent: Any, do_narator: bool = True):
        self.live = live
        self.agent = agent  # kept for future: model preference routing
        self._enabled: bool = bool(BLITZ_MODEL) and NARRATOR_ENABLED and do_narator

        # Per-run state (reset on each execute call)
        self._mini: NarratorMiniState = NarratorMiniState()
        self._lang: str = "de"

        # Async task management
        self._pending_task: asyncio.Task | None = None
        self._pending_is_think: bool = False  # think tasks are NOT cancelled by normal tasks

        # Rate-limiting
        self._last_blitz_time: float = 0.0
        self._thought_set_time: float = 0.0   # when we last wrote live.thought

        # Per-minute sliding window budget  [(timestamp, in_tokens, out_tokens), ...]
        # NOT reset per-run – persists across runs (shared across the engine lifetime)
        self._token_log: list[tuple[float, int, int]] = []

        # History cursor (diff tracking)
        self._history_cursor: int = 0

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def reset(self, query: str = ""):
        """Call at the start of each execute() run."""
        self._mini = NarratorMiniState()
        self._lang = _lang(query)
        self._pending_task = None
        self._pending_is_think = False
        self._last_blitz_time = 0.0
        self._thought_set_time = 0.0
        # NOTE: _token_log is NOT reset here – it's a per-minute sliding window
        # that lives across runs so the PM limit is respected correctly.
        self._history_cursor = 0

    # ------------------------------------------------------------------
    # Synchronous mock setters (always instant, no async needed)
    # ------------------------------------------------------------------

    def mock(self, key: str, **fmt) -> None:
        """Set live.thought immediately with a predefined mock string."""
        text = _mock(key, self._lang, **fmt)
        self._set_thought(text)

    def _set_thought(self, text: str) -> None:
        self.live.narrator_msg = text
        self._thought_set_time = time.monotonic()

    def _append_state_flags(self, text: str) -> str:
        if self._mini.drift:
            text += _MOCKS["drift"][self._lang]
        if self._mini.repeat:
            text += _MOCKS["repeat"][self._lang]
        return text

    # ------------------------------------------------------------------
    # Event handlers  (called from engine – all are async but lightweight)
    # ------------------------------------------------------------------

    def on_init(self, query: str) -> None:
        """Call right after live.enter(INIT, ...). Sync – no blitz yet."""
        self._lang = _lang(query)
        self.mock("init")

    def on_llm_pre_call(self, history: list[dict]) -> None:
        """
        Called just before each LLM call.
        Always sets mock immediately.
        Does NOT schedule blitz (no new tool results yet → last info is better).
        """
        self.mock("llm_pre")
        # advance cursor so next diff only contains what happened AFTER this call
        self._history_cursor = len(history)

    def on_tool_start(self, tool_name: str) -> None:
        """Sync: set mock for tool start immediately."""
        if tool_name == "think":
            self.mock("think")
        else:
            self.mock("tool_start", tool=tool_name)

    def schedule_tool_end(
        self,
        tool_name: str,
        result_snippet: str,
        history: list[dict],
    ) -> None:
        """
        After tool finished. Sets mock immediately, then schedules
        blitz in background *if* rate-limit and budget allow.
        """
        self.mock("tool_end")

        if not self._enabled:
            return
        if not self._budget_ok(history):
            return
        if time.monotonic() - self._last_blitz_time < NARRATOR_MIN_INTERVAL:
            # still inside cool-down – mock string is fine
            return

        diff = _compress_diff(history, self._history_cursor)
        self._history_cursor = len(history)

        self._schedule(
            self._blitz_normal(diff, tool_hint=f"{tool_name}: {result_snippet[:60]}"),
            is_think=False,
        )

    def schedule_think_result(
        self,
        thinking_content: str,
        history: list[dict],
    ) -> None:
        """
        Special path: agent finished a think-tool call.
        Always tries to call Blitz (cancels any non-think pending task).
        """
        # dedup: same thinking chunk → skip
        chunk_hash = hashlib.md5(thinking_content[:200].encode()).hexdigest()[:8]
        if chunk_hash == self._mini.last_think_hash:
            return
        self._mini.last_think_hash = chunk_hash

        self._history_cursor = len(history)

        if not self._enabled:
            # still update thought with something sensible
            self.mock("think")
            return

        if not self._budget_ok(history):
            return

        # cancel only non-think pending tasks
        self._cancel_pending(force=True)

        self._schedule(
            self._blitz_think(thinking_content),
            is_think=True,
        )

    def on_summarise(self) -> None:
        """Called when agent is about to give final_answer."""
        self.mock("summarise")

    # ------------------------------------------------------------------
    # Internal: task scheduling & cancellation
    # ------------------------------------------------------------------

    def _schedule(self, coro, *, is_think: bool) -> None:
        """Wrap coroutine in a task, honouring cancel rules."""
        if not is_think:
            # normal tasks cancel previous normal task, but NOT think tasks
            self._cancel_pending(force=False)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no event loop – skip silently

        task = loop.create_task(coro)
        task.add_done_callback(self._on_task_done)
        self._pending_task = task
        self._pending_is_think = is_think

    def _cancel_pending(self, *, force: bool) -> None:
        if self._pending_task and not self._pending_task.done():
            if force or not self._pending_is_think:
                self._pending_task.cancel()
        self._pending_task = None
        self._pending_is_think = False

    def _on_task_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.debug("Narrator task raised: %s", exc)

    # ------------------------------------------------------------------
    # Internal: budget guard
    # ------------------------------------------------------------------

    def _budget_ok(self, history: list[dict]) -> bool:
        """
        Return True if per-minute token budget allows another Blitz call.
        Uses a sliding 60-second window – entries older than 60 s are pruned
        before every check so the limit is rolling, not fixed-interval.
        """
        now = time.monotonic()
        cutoff = now - _BUDGET_WINDOW
        # prune stale entries
        self._token_log = [(t, i, o) for t, i, o in self._token_log if t > cutoff]

        used_in  = sum(i for _, i, _ in self._token_log)
        used_out = sum(o for _, _, o in self._token_log)

        # estimate cost of the upcoming call
        diff = _compress_diff(history, self._history_cursor)
        est_in = _estimate_tokens(diff)

        if used_in + est_in > NARRATOR_MAX_INPUT_TOKENS_PM:
            logger.debug("Narrator: input TPM budget full (%d/%d) – skipping blitz",
                         used_in, NARRATOR_MAX_INPUT_TOKENS_PM)
            return False
        if used_out >= NARRATOR_MAX_OUTPUT_TOKENS_PM:
            logger.debug("Narrator: output TPM budget full (%d/%d) – skipping blitz",
                         used_out, NARRATOR_MAX_OUTPUT_TOKENS_PM)
            return False
        return True

    def _record_tokens(self, tokens_in: int, tokens_out: int) -> None:
        """Log actual token usage from a completed Blitz call."""
        self._token_log.append((time.monotonic(), tokens_in, tokens_out))

    # ------------------------------------------------------------------
    # Internal: blitz call coroutines
    # ------------------------------------------------------------------

    async def _blitz_normal(
        self,
        diff_msgs: list[dict],
        tool_hint: str = "",
    ) -> None:
        self._last_blitz_time = time.monotonic()
        system, messages = _build_normal_prompt(
            self._lang, diff_msgs, self._mini, tool_hint
        )
        result = await _call_blitz(system, messages)
        if result is None:
            return

        # Stale guard: if thought was refreshed <0.3 s ago by a more recent mock,
        # discard blitz result to avoid overwriting fresher info.
        if time.monotonic() - self._thought_set_time < NARRATOR_STALE_THRESHOLD:
            logger.debug("Narrator: blitz result discarded (stale)")
            return

        self._record_tokens(result["in"], result["out"])

        data = result["data"]
        thought = data.get("t", "")
        if not thought:
            return

        if data.get("d"):
            self._mini.drift = True
        if data.get("r"):
            self._mini.repeat = True

        self._set_thought(self._append_state_flags(thought))

    async def _blitz_think(self, thinking_content: str) -> None:
        self._last_blitz_time = time.monotonic()
        system, messages = _build_think_prompt(
            self._lang, thinking_content, self._mini
        )
        result = await _call_blitz(system, messages)
        if result is None:
            self.mock("think")
            return

        self._record_tokens(result["in"], result["out"])

        data = result["data"]
        thought = data.get("t", "")
        plan = data.get("plan", "")

        # Update mini-state with new plan if provided
        if plan:
            self._mini.plan_summary = plan
        if data.get("d"):
            self._mini.drift = True
        if data.get("r"):
            self._mini.repeat = True

        if thought:
            self._set_thought(self._append_state_flags(thought))
        else:
            self.mock("think")
