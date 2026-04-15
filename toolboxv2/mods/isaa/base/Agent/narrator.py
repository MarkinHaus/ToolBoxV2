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
import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from toolboxv2 import get_logger, get_app

logger = get_logger()

# ---------------------------------------------------------------------------
# Config / Constants
# ---------------------------------------------------------------------------

BLITZ_MODEL: str = os.getenv("BLITZMODEL", "")
NARRATOR_ENABLED: bool = os.getenv("AGENT_NARRATOR_ENABLED", "true").lower() == "true"
NARRATOR_LANG: str = os.getenv("NARRATOR_LANG", "auto")  # auto | de | en

# Token budget per minute (sliding window, matches Groq free-tier limits)
NARRATOR_MAX_INPUT_TOKENS_PM: int = int(os.getenv("NARRATOR_MAX_INPUT_TPM", "10000"))
NARRATOR_MAX_OUTPUT_TOKENS_PM: int = int(os.getenv("NARRATOR_MAX_OUTPUT_TPM", "1000"))
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
    Call BLITZ_MODEL via LiteLLMRateLimitHandler (with fallback + key rotation).
    Falls back to raw litellm.acompletion if no handler is initialised.
    Returns parsed JSON dict or None on any error.
    """
    try:
        all_messages = [{"role": "system", "content": system}] + messages

        if _blitz_handler is not None:
            import litellm as _litellm_mod

            response = await _blitz_handler.completion_with_rate_limiting(
                _litellm_mod,
                model=BLITZ_MODEL,
                messages=all_messages,
                max_tokens=40,
                temperature=0.3,
                stream=False,
                drop_params=True,
            )
        else:
            # Fallback: direkt litellm ohne handler
            import litellm  # type: ignore

            response = await litellm.acompletion(
                model=BLITZ_MODEL,
                messages=all_messages,
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
            "raw": raw,
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

_GROQ_DEFAULT_FALLBACKS: dict[str, list[str]] = {
    # Primary model → ordered fallback list
    #   Strategy: same-class first, then smaller/cheaper
    "groq/llama-3.1-8b-instant": [
        "groq/llama-3.3-70b-versatile",
        "groq/openai/gpt-oss-20b",
    ],
    "groq/llama-3.3-70b-versatile": [
        "groq/llama-3.1-8b-instant",
        "groq/openai/gpt-oss-20b",
    ],
    "groq/openai/gpt-oss-120b": [
        "groq/openai/gpt-oss-20b",
        "groq/llama-3.3-70b-versatile",
        "groq/llama-3.1-8b-instant",
    ],
    "groq/openai/gpt-oss-20b": [
        "groq/llama-3.1-8b-instant",
        "groq/llama-3.3-70b-versatile",
    ],
}

# Generic fallback when BLITZ_MODEL is a groq model not in the table above
_GROQ_GENERIC_FALLBACKS: list[str] = [
    "groq/llama-3.1-8b-instant",
    "groq/llama-3.3-70b-versatile",
    "groq/openai/gpt-oss-20b",
]


def _resolve_groq_fallbacks(primary: str) -> list[str]:
    """
    Return a fallback chain for a groq primary model.
    Uses the explicit table if available, otherwise the generic list
    (filtering out the primary itself).
    """
    if primary in _GROQ_DEFAULT_FALLBACKS:
        return _GROQ_DEFAULT_FALLBACKS[primary]
    # unknown groq model → generic chain minus itself
    return [m for m in _GROQ_GENERIC_FALLBACKS if m != primary]

_blitz_handler: "LiteLLMRateLimitHandler | None" = None


def init_narrator_handler(
    handler: "LiteLLMRateLimitHandler | None" = None,
    fallback_models: list[str] | None = None,
) -> "LiteLLMRateLimitHandler":
    """
    Initialise (or replace) the module-level rate-limit handler used by
    all Narrator Blitz calls.

    Args:
        handler:          Pass an existing handler to share with the rest of
                          the engine.  If None a fresh one is created.
        fallback_models:  Optional fallback chain for BLITZ_MODEL.
                          If None AND BLITZ_MODEL is a groq model,
                          a sensible default chain is auto-selected.

    Returns the active handler so the caller can further configure it.
    """
    global _blitz_handler

    if handler is not None:
        _blitz_handler = handler
        return _blitz_handler

    # Auto-detect groq fallbacks when none provided
    if fallback_models is None and "groq" in BLITZ_MODEL.lower():
        fallback_models = _resolve_groq_fallbacks(BLITZ_MODEL)
        logger.info(
            "Narrator: auto-selected groq fallbacks for %s → %s",
            BLITZ_MODEL, fallback_models,
        )

    # lazy import – keep module importable without the handler installed
    from toolboxv2.mods.isaa.base.IntelligentRateLimiter import LiteLLMRateLimitHandler

    _blitz_handler = LiteLLMRateLimitHandler(
        enable_model_fallback=bool(fallback_models),
        enable_key_rotation=True,
        key_rotation_mode="balance",
        max_retries=2,
    )

    if fallback_models:
        _blitz_handler.add_fallback_chain(
            primary_model=BLITZ_MODEL,
            fallback_models=fallback_models,
            fallback_duration=90.0,
        )

    return _blitz_handler


class AgentLiveNarrator:
    """
    Attach to ExecutionEngine as ``self._narrator``.

    Usage pattern in engine::

        # synchronous set (immediate UI update)
        self._narrator.mock("tool_start", tool="calc")

        # async schedule (fire-and-forget, non-blocking)
        asyncio.create_task(self._narrator.on_tool_end(ctx))
    """
    def __init__(self, live: "AgentLiveState", agent: Any, do_narator=True,
             handler: "LiteLLMRateLimitHandler | None" = None,
             fallback_models: list[str] | None = None):
        self.live = live
        self.agent = agent  # kept for future: model preference routing
        self._enabled: bool = bool(BLITZ_MODEL) and NARRATOR_ENABLED and do_narator

        if self._enabled:
            init_narrator_handler(handler=handler, fallback_models=fallback_models)

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
        from toolboxv2 import get_app
        get_app().print(text)
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

    async def blitz(
        self,
        system: str,
        messages: list[dict],
        schema: dict | None = None,
        respect_ratelimit: bool = False,
        history: list[dict] | None = None,
    ) -> dict | str | None:
        """
        Public raw Blitz-Model call – now routed through the rate-limit handler.

        Args:
            system:            System prompt.
            messages:          User/assistant messages.
            schema:            Optional expected JSON schema as dict.
            respect_ratelimit: If True, checks internal narrator PM budget
                               before calling (handler has its own limits too).
            history:           Working history – for internal budget estimation.

        Returns:
            Parsed dict if schema given and valid,
            raw string if no schema,
            None on error / budget exceeded.
        """
        if not self._enabled:
            return None

        if respect_ratelimit:
            if not self._budget_ok(history or []):
                return None

        result = await _call_blitz(system, messages)
        if result is None:
            return None

        if respect_ratelimit:
            self._record_tokens(result["in"], result["out"])

        data = result["data"]

        if schema is None:
            return data if isinstance(data, dict) else result.get("raw", "")

        for key, expected_type in schema.items():
            if key not in data:
                logger.debug("Blitz schema mismatch: missing key %r", key)
                return None
            if not isinstance(data[key], expected_type):
                logger.debug(
                    "Blitz schema mismatch: key %r expected %s got %s",
                    key, expected_type.__name__, type(data[key]).__name__,
                )
                return None

        return data

    def on_summarise(self) -> None:
        """Called when agent is about to give final_answer."""
        self.mock("summarise")
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


    # ------------------------------------------------------------------
    # 1. Skills – select relevant skills for current context
    # ------------------------------------------------------------------

    def schedule_skills_update(
        self,
        query: str,
        history: list[dict],
        skills_manager: Any,
        ctx,
    ) -> None:
        """
        Fire-and-forget: ask Blitz to pick the most relevant skills for the
        current context, then update live.skills.

        Call after on_init() and after schedule_think_result().
        """
        if not self._enabled:
            return
        if not self._budget_ok(history):
            return

        skill_index = _compress_skills_to_index(skills_manager)
        if skill_index == "(no skills)":
            return

        diff = _compress_diff(history, self._history_cursor)
        self._schedule(
            self._blitz_skills(query, diff, skill_index, skills_manager, ctx),
            is_think=False,
        )

    async def _blitz_skills(
        self,
        query: str,
        diff_msgs: list[dict],
        skill_index: str,
        skills_manager: Any,
        ctx: None
    ) -> None:
        system, messages = _build_skills_prompt(
            self._lang, query, diff_msgs, skill_index, self._mini
        )
        result = await _call_blitz(system, messages)
        if result is None:
            return

        self._record_tokens(result["in"], result["out"])

        data = result["data"]
        selected_ids: list[str] = data.get("ids", [])
        if not isinstance(selected_ids, list):
            return

        # Validate ids exist in skills_manager
        skills = getattr(skills_manager, "skills", {})
        valid_ids = [sid for sid in selected_ids if sid in skills]

        # Update live state
        valid_names = [skills[sid].name for sid in valid_ids]
        self.live.skills = valid_names
        if ctx:
            ctx.matched_skills = valid_ids

        logger.debug(
            "Narrator: skills updated → %s (reason: %s)",
            valid_ids, data.get("reason", ""),
        )

    # ------------------------------------------------------------------
    # 2. RuleSet – extract situation/intent and activate matching rules
    # ------------------------------------------------------------------

    def schedule_ruleset_update(
        self,
        history: list[dict],
        session: Any,
        ctx: Any,
    ) -> None:
        """
        Fire-and-forget: ask Blitz to extract situation + intent from history,
        update session.rule_set, and write result to VFS.

        Call after on_init() and after substantial tool results.
        """
        if not self._enabled:
            return
        rule_set = getattr(session, "rule_set", None)
        if rule_set is None:
            return
        if not self._budget_ok(history):
            return

        diff = _compress_diff(history, self._history_cursor)
        self._schedule(
            self._blitz_ruleset(diff, session, rule_set, ctx),
            is_think=False,
        )

    async def _blitz_ruleset(
        self,
        diff_msgs: list[dict],
        session: Any,
        rule_set: Any,
        ctx: Any,
    ) -> None:
        system, messages = _build_ruleset_prompt(self._lang, diff_msgs, self._mini)
        result = await _call_blitz(system, messages)
        if result is None:
            return

        self._record_tokens(result["in"], result["out"])

        data = result["data"]
        situation: str = data.get("situation", "").strip()
        intent: str = data.get("intent", "").strip()
        confidence: float = float(data.get("confidence", 0.0))

        if not situation or not intent or confidence < 0.3:
            logger.debug("Narrator: ruleset update skipped (low confidence %.2f)", confidence)
            return

        # Apply to rule_set (this activates matching tool groups)
        try:
            rule_set.set_situation(situation, intent)
        except Exception as exc:
            logger.debug("Narrator: rule_set.set_situation failed: %s", exc)
            return

        logger.debug(
            "Narrator: ruleset updated → situation=%r intent=%r (conf=%.2f)",
            situation, intent, confidence,
        )

        # Write updated VFS content
        try:
            vfs_content: str = rule_set.build_vfs_content()
            vfs_filename: str = rule_set.get_vfs_filename()
            vfs = getattr(session, "vfs", None)
            if vfs is not None:
                vfs.set_rules_file(vfs_content)
                rule_set.mark_clean()
                logger.debug("Narrator: VFS %r updated", vfs_filename)
        except Exception as exc:
            logger.debug("Narrator: VFS write failed: %s", exc)

    # ------------------------------------------------------------------
    # 3. Live Memory Extraction
    # ------------------------------------------------------------------

    def schedule_memory_extraction(
        self,
        query: str,
        history: list[dict],
        ctx: Any,
        session: Any,
    ) -> None:
        """
        Fire-and-forget: ask Blitz to extract useful facts from recent history,
        inject working-memory facts into ctx.working_history and persist
        user-level facts via session memory.

        Call after schedule_tool_end() for substantive tool results.
        """
        if not self._enabled:
            return
        if not self._budget_ok(history):
            return
        if time.monotonic() - self._last_blitz_time < NARRATOR_MIN_INTERVAL:
            return

        diff = _compress_diff(history, self._history_cursor)
        if not diff:
            return

        self._schedule(
            self._blitz_memory(query, diff, ctx, session),
            is_think=False,
        )

    async def _blitz_memory(
        self,
        query: str,
        diff_msgs: list[dict],
        ctx: Any,
        session: Any,
    ) -> None:
        self._last_blitz_time = time.monotonic()
        system, messages = _build_memory_prompt(self._lang, query, diff_msgs)
        result = await _call_blitz(system, messages)
        if result is None:
            return

        self._record_tokens(result["in"], result["out"])

        data = result["data"]
        if not data.get("found", False):
            return

        working_facts: list[str] = data.get("working", [])
        user_facts: list[str] = data.get("user", [])

        # --- Inject working facts into ctx.working_history ---
        if working_facts and hasattr(ctx, "working_history"):
            label = "Extrahierte Kontext-Fakten" if self._lang == "de" else "Extracted Context Facts"
            facts_str = "\n".join(f"- {f}" for f in working_facts+user_facts)
            inject_msg = {
                "role": "system",
                "content": f"[{label}]\n{facts_str}",
            }
            # Insert after the last system message to stay near context start
            insert_at = 0
            for i, m in enumerate(ctx.working_history):
                if m.get("role") == "system":
                    insert_at = i + 1
            ctx.working_history.insert(insert_at, inject_msg)
            logger.debug("Narrator: injected %d working facts", len(working_facts))


# =============================================================================
# NARRATOR EXTENSION: Skills · RuleSet · Live Memory
# =============================================================================
# Each of the three subsystems is self-contained:
#   - schedule_*   → public fire-and-forget entry points (called from Engine)
#   - _blitz_*     → async coroutines that do the actual Blitz call + side-effect
#
# All three respect the shared PM token budget and the cancel/priority rules
# already established for the core narrator tasks.
# =============================================================================

# ---------------------------------------------------------------------------
# System prompts  (kept outside the class so they are importable for tests)
# ---------------------------------------------------------------------------

_SKILLS_SYSTEM_DE = """\
Du bist ein Skill-Selektor für einen KI-Agenten. Deine einzige Aufgabe:
Wähle aus der gegebenen Skill-Liste die 1-3 am besten passenden Skills für den aktuellen Kontext aus.

Regeln:
- Antworte AUSSCHLIESSLICH mit kompaktem JSON, kein Markdown:
  {"ids": ["id1", "id2"], "reason": "<max 8 deutsche Wörter warum>"}
- Wähle NUR Skills die direkt zur aktuellen Aufgabe passen
- Lieber weniger als zu viele
- "ids" darf leer sein wenn kein Skill passt"""

_SKILLS_SYSTEM_EN = """\
You are a skill selector for an AI agent. Your sole task:
Select 1-3 skills from the provided list that best fit the current context.

Rules:
- Reply ONLY with compact JSON, no markdown:
  {"ids": ["id1", "id2"], "reason": "<max 8 English words why>"}
- Select ONLY skills directly relevant to the current task
- Fewer is better than too many
- "ids" may be empty if no skill fits"""

_RULESET_SYSTEM_DE = """\
Du bist ein Kontext-Analyzer für einen KI-Agenten. Deine Aufgabe:
Extrahiere aus dem Agent-Verlauf die aktuelle Situation und den Intent.

Regeln:
- Antworte AUSSCHLIESSLICH mit kompaktem JSON, kein Markdown:
  {"situation": "<5-8 Wörter Kontext>", "intent": "<5-8 Wörter Ziel>", "confidence": 0.0}
- "situation": beschreibt WO der Agent gerade arbeitet (z.B. "python datei analyse", "discord api arbeit")
- "intent": beschreibt WAS der Agent erreichen will (z.B. "fehler finden und beheben", "nachricht senden")
- "confidence": 0.0-1.0 wie sicher du dir bist
- Bleibe faktisch, keine Spekulation"""

_RULESET_SYSTEM_EN = """\
You are a context analyzer for an AI agent. Your task:
Extract the current situation and intent from the agent's history.

Rules:
- Reply ONLY with compact JSON, no markdown:
  {"situation": "<5-8 word context>", "intent": "<5-8 word goal>", "confidence": 0.0}
- "situation": describes WHERE the agent is working (e.g. "python file analysis", "discord api work")
- "intent": describes WHAT the agent wants to achieve (e.g. "find and fix error", "send message")
- "confidence": 0.0-1.0 how certain you are
- Stay factual, no speculation"""

_MEMORY_SYSTEM_DE = """\
Du bist ein Memory-Extraktor für einen KI-Agenten. Deine Aufgabe:
Extrahiere aus dem Agent-Verlauf NUR dauerhaft nützliche Fakten.

Regeln:
- Antworte AUSSCHLIESSLICH mit kompaktem JSON, kein Markdown:
  {"working": ["fakt1", "fakt2"], "user": ["user_fakt1"], "found": true}
- "working": Kurzlebige Fakten relevant für den aktuellen Lauf (z.B. "Datei X liegt in /pfad/Y", "Funktion Z gibt None zurück")
  * Max 3 Einträge, je max 15 Wörter
  * NUR wenn sie in den nächsten Schritten gebraucht werden
- "user": Stabile Fakten ÜBER DEN USER (z.B. "User bevorzugt deutsche Kommentare", "User arbeitet mit Python 3.12")
  * Max 2 Einträge, je max 12 Wörter
  * NUR echte neue Infos, keine Annahmen
- "found": false wenn keine relevanten Fakten vorhanden
- NIEMALS: temporäre Zustände, Zwischen-Ergebnisse, Meinungen"""

_MEMORY_SYSTEM_EN = """\
You are a memory extractor for an AI agent. Your task:
Extract ONLY permanently useful facts from the agent's history.

Rules:
- Reply ONLY with compact JSON, no markdown:
  {"working": ["fact1", "fact2"], "user": ["user_fact1"], "found": true}
- "working": Short-lived facts relevant for the current run (e.g. "File X is at /path/Y", "Function Z returns None")
  * Max 3 entries, max 15 words each
  * ONLY if needed in upcoming steps
- "user": Stable facts ABOUT THE USER (e.g. "User prefers English comments", "User works with Python 3.12")
  * Max 2 entries, max 12 words each
  * ONLY genuine new info, no assumptions
- "found": false if no relevant facts present
- NEVER: temporary states, intermediate results, opinions"""


def _build_skills_prompt(
    lang: str,
    query: str,
    diff_msgs: list[dict],
    skill_index: str,
    mini: "NarratorMiniState",
) -> tuple[str, list[dict]]:
    system = _SKILLS_SYSTEM_DE if lang == "de" else _SKILLS_SYSTEM_EN
    label_q = "Anfrage" if lang == "de" else "Query"
    label_plan = "Plan" if lang == "de" else "Plan"
    label_skills = "Verfügbare Skills" if lang == "de" else "Available Skills"
    label_ctx = "Letzter Kontext" if lang == "de" else "Recent Context"

    parts = [f"{label_q}: {query[:120]}"]
    if mini.plan_summary:
        parts.append(f"{label_plan}: {mini.plan_summary}")
    parts.append(f"{label_skills}:\n{skill_index}")
    if diff_msgs:
        recent = "\n".join(f"[{m['role']}] {m['content'][:80]}" for m in diff_msgs[-3:])
        parts.append(f"{label_ctx}:\n{recent}")

    return system, [{"role": "user", "content": "\n\n".join(parts)}]


def _build_ruleset_prompt(
    lang: str,
    diff_msgs: list[dict],
    mini: "NarratorMiniState",
) -> tuple[str, list[dict]]:
    system = _RULESET_SYSTEM_DE if lang == "de" else _RULESET_SYSTEM_EN
    label_plan = "Bekannter Plan" if lang == "de" else "Known Plan"
    label_ctx = "Agent-Verlauf" if lang == "de" else "Agent History"

    parts = []
    if mini.plan_summary:
        parts.append(f"{label_plan}: {mini.plan_summary}")
    if diff_msgs:
        recent = "\n".join(f"[{m['role']}] {m['content'][:100]}" for m in diff_msgs[-5:])
        parts.append(f"{label_ctx}:\n{recent}")
    if not parts:
        parts.append(".")

    return system, [{"role": "user", "content": "\n\n".join(parts)}]


def _build_memory_prompt(
    lang: str,
    query: str,
    diff_msgs: list[dict],
) -> tuple[str, list[dict]]:
    system = _MEMORY_SYSTEM_DE if lang == "de" else _MEMORY_SYSTEM_EN
    label_q = "Ursprüngliche Anfrage" if lang == "de" else "Original Query"
    label_ctx = "Neuer Verlauf" if lang == "de" else "New History"

    parts = [f"{label_q}: {query[:100]}"]
    if diff_msgs:
        recent = "\n".join(f"[{m['role']}] {m['content'][:120]}" for m in diff_msgs[-6:])
        parts.append(f"{label_ctx}:\n{recent}")

    return system, [{"role": "user", "content": "\n\n".join(parts)}]


def _compress_skills_to_index(skills_manager: Any) -> str:
    """
    Convert SkillsManager skills to a compact index string for Blitz prompt.
    Format: "id|name|trigger1,trigger2,trigger3"  (one per line, max 20 skills)
    """
    lines = []
    skills = getattr(skills_manager, "skills", {})
    for skill in list(skills.values()):
        if not getattr(skill, "is_active", lambda: True)():
            continue
        triggers = ",".join(getattr(skill, "triggers", [])[:3])
        lines.append(f"{skill.id}|{skill.name}|{triggers}")
    return "\n".join(lines) if lines else "(no skills)"
