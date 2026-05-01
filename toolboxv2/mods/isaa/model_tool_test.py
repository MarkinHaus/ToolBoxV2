"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        ISAA – Ollama Tool-Call Diagnostik & Benchmark                       ║
║                                                                              ║
║  Dieses Modul in module.py (am Ende, vor dem __main__-Block) einfügen.      ║
║                                                                              ║
║  Testet in 4 Phasen ob ein Modell über litellm + Ollama echte              ║
║  Tool-Calls im ISAA-Agenten-System ausführen kann:                          ║
║                                                                              ║
║  Phase 1 – Raw litellm:        Direktaufruf, keine ISAA-Schicht             ║
║  Phase 2 – tool_choice-Matrix: auto / required / None × stream / no-stream  ║
║  Phase 3 – FlowAgent-Wrapper:  a_run_llm_completion mit Tools               ║
║  Phase 4 – ExecutionEngine:    Voller Agenten-Lauf mit echtem Tool          ║
║                                                                              ║
║  Aufruf:                                                                     ║
║    python -m toolboxv2.mods.isaa.module --test-ollama [model] [--verbose]   ║
║  oder direkt:                                                                ║
║    python module.py --test-ollama ollama/qwen3.5:0.8b                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
import datetime
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Einfaches Test-Tool – zeitgestempelte Antwort, deterministisch prüfbar
# ─────────────────────────────────────────────────────────────────────────────

_TEST_TOOL_CALL_MARKER = "__TOOL_WAS_CALLED__"

TEST_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Returns the current date and time. "
                "Call this whenever the user asks what time or date it is."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Optional timezone name, e.g. 'UTC' or 'Europe/Berlin'.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Adds two numbers and returns the result. Use when asked to add or sum numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        },
    },
]


async def _execute_test_tool(name: str, args: dict) -> str:
    if name == "get_current_time":
        tz = args.get("timezone", "UTC")
        return f"{_TEST_TOOL_CALL_MARKER} Current time ({tz}): {datetime.datetime.now(datetime.UTC)}Z"
    if name == "add_numbers":
        a = float(args.get("a", 0))
        b = float(args.get("b", 0))
        return f"{_TEST_TOOL_CALL_MARKER} Result: {a + b}"
    return f"Unknown tool: {name}"


# ─────────────────────────────────────────────────────────────────────────────
# Ergebnis-Datenklasse
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhaseResult:
    name: str
    passed: bool = False
    tool_called: bool = False
    tool_name: str | None = None
    tool_args: dict = field(default_factory=dict)
    finish_reason: str | None = None
    duration_s: float = 0.0
    error: str | None = None
    raw_content: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class DiagnosticReport:
    model: str
    timestamp: str = field(default_factory=lambda:  datetime.datetime.now(datetime.UTC))
    phases: list[PhaseResult] = field(default_factory=list)
    recommendation: str = ""
    ollama_workaround_needed: bool = False
    working_config: dict | None = None

    def summary(self) -> str:
        lines = [
            "",
            "╔══════════════════════════════════════════════════════╗",
            f"║  OLLAMA TOOL-CALL DIAGNOSTIK  │  {self.model[:30]:<30}  ║",
            "╠══════════════════════════════════════════════════════╣",
        ]
        for p in self.phases:
            icon = "✅" if p.passed else ("⚠️ " if p.tool_called else "❌")
            dur = f"{p.duration_s:.1f}s"
            err_hint = f"  [{p.error[:40]}]" if p.error else ""
            lines.append(
                f"║  {icon}  {p.name:<35}  {dur:>5}{err_hint}"
            )
        lines += [
            "╠══════════════════════════════════════════════════════╣",
            f"║  Empfehlung: {self.recommendation[:46]:<46}  ║",
            "╚══════════════════════════════════════════════════════╝",
            "",
        ]
        if self.working_config:
            lines.append(f"Funktionierender Konfiguration: {json.dumps(self.working_config, indent=2)}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Hilfs-Funktionen
# ─────────────────────────────────────────────────────────────────────────────

def _log(msg: str, verbose: bool, level: str = "INFO"):
    if verbose:
        prefix = {"INFO": "ℹ️ ", "WARN": "⚠️ ", "ERR": "❌", "OK": "✅"}.get(level, "   ")
        print(f"  {prefix} {msg}", flush=True)


def _extract_tool_calls_from_response(response_msg) -> list[dict]:
    """Extrahiert Tool-Calls aus einer litellm Message (verschiedene Formate)."""
    calls = []

    # Standard: response.tool_calls Liste
    tool_calls = getattr(response_msg, "tool_calls", None) or []
    for tc in tool_calls:
        try:
            name = tc.function.name
            args_raw = tc.function.arguments or "{}"
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            calls.append({"name": name, "args": args, "id": getattr(tc, "id", "?")})
        except Exception:
            pass

    return calls


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 – Raw litellm Direct Call
# ─────────────────────────────────────────────────────────────────────────────

async def _phase1_raw_litellm(model: str, verbose: bool) -> PhaseResult:
    """
    Ruft litellm direkt auf, ohne ISAA-Schicht.
    Testet ob das Modell überhaupt tool_calls zurückgibt.
    """
    phase = PhaseResult(name="Phase 1 – Raw litellm (no-stream)")
    _log("Starte Phase 1: Direkter litellm-Aufruf...", verbose)

    try:
        import litellm

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Always use the provided tools when appropriate."},
            {"role": "user", "content": "What time is it right now? Please use the get_current_time tool."},
        ]

        t0 = time.time()
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            tools=TEST_TOOL_SCHEMA,
            tool_choice="auto",
            stream=False,
            request_timeout=60,
        )
        phase.duration_s = time.time() - t0

        msg = response.choices[0].message
        phase.finish_reason = response.choices[0].finish_reason
        phase.raw_content = getattr(msg, "content", "") or ""

        _log(f"finish_reason = {phase.finish_reason!r}", verbose)
        _log(f"content preview = {(phase.raw_content or '')[:120]!r}", verbose)

        calls = _extract_tool_calls_from_response(msg)
        if calls:
            phase.tool_called = True
            phase.tool_name = calls[0]["name"]
            phase.tool_args = calls[0]["args"]
            phase.passed = True
            phase.notes.append(f"Tool-Call erkannt: {phase.tool_name}({phase.tool_args})")
            _log(f"Tool-Call erkannt: {phase.tool_name}", verbose, "OK")
        else:
            phase.notes.append("Kein Tool-Call in der Antwort – Modell antwortete als Text.")
            _log("Kein tool_call in Response!", verbose, "WARN")
            if phase.raw_content:
                _log(f"Model-Antwort (Text): {phase.raw_content[:200]}", verbose)

    except Exception as e:
        phase.error = str(e)
        phase.duration_s = time.time() - t0 if "t0" in dir() else 0
        _log(f"Exception: {e}", verbose, "ERR")

    return phase


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 – tool_choice × stream Kombinationsmatrix
# ─────────────────────────────────────────────────────────────────────────────

async def _phase2_tool_choice_matrix(model: str, verbose: bool) -> tuple[PhaseResult, dict | None]:
    """
    Testet alle relevanten Kombinationen von tool_choice + stream.
    Gibt das erste funktionierende Konfiguration zurück.
    """
    phase = PhaseResult(name="Phase 2 – tool_choice × stream Matrix")
    working_config = None

    COMBOS = [
        # (tool_choice, stream, label)
        ("auto",     False, "tool_choice=auto,  stream=False"),
        ("required", False, "tool_choice=required, stream=False"),
        (None,       False, "tool_choice=None,  stream=False"),
        ("auto",     True,  "tool_choice=auto,  stream=True"),
        (None,       True,  "tool_choice=None,  stream=True"),
    ]

    _log("Starte Phase 2: Kombinationsmatrix...", verbose)

    try:
        import litellm
        litellm.drop_params = True  # genau wie in flow_agent.py

        messages = [
            {"role": "user", "content": "Please add the numbers 17 and 25 using the add_numbers tool."},
        ]

        t0 = time.time()
        for tool_choice, use_stream, label in COMBOS:
            _log(f"  Teste: {label}", verbose)
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "tools": TEST_TOOL_SCHEMA,
                    "stream": use_stream,
                    "request_timeout": 45,
                }
                if tool_choice is not None:
                    kwargs["tool_choice"] = tool_choice

                response = await litellm.acompletion(**kwargs)

                # Stream auslesen
                if use_stream:
                    msg_content = ""
                    all_tool_calls = []
                    async for chunk in response:
                        if chunk.choices:
                            delta = chunk.choices[0].delta
                            if delta.content:
                                msg_content += delta.content
                            if getattr(delta, "tool_calls", None):
                                for tc in delta.tool_calls:
                                    idx = tc.index if hasattr(tc, "index") else 0
                                    while len(all_tool_calls) <= idx:
                                        all_tool_calls.append({"name": "", "args": ""})
                                    if tc.function:
                                        if tc.function.name:
                                            all_tool_calls[idx]["name"] += tc.function.name
                                        if tc.function.arguments:
                                            all_tool_calls[idx]["args"] += tc.function.arguments

                    calls = []
                    for tc in all_tool_calls:
                        if tc["name"]:
                            try:
                                args = json.loads(tc["args"]) if tc["args"] else {}
                            except Exception:
                                args = {"raw": tc["args"]}
                            calls.append({"name": tc["name"], "args": args})
                else:
                    msg = response.choices[0].message
                    calls = _extract_tool_calls_from_response(msg)

                if calls:
                    _log(f"    ✅ Tool-Call via [{label}]: {calls[0]['name']}({calls[0]['args']})", verbose, "OK")
                    phase.tool_called = True
                    phase.tool_name = calls[0]["name"]
                    phase.tool_args = calls[0]["args"]
                    phase.passed = True
                    working_config = {
                        "tool_choice": tool_choice,
                        "stream": use_stream,
                        "combo_label": label,
                    }
                    phase.notes.append(f"Funktionierendes Kombo gefunden: {label}")
                    # Weiter testen um alle Ergebnisse zu sammeln, aber merken
                    break
                else:
                    _log(f"    ⚠️  Kein Tool-Call via [{label}]", verbose, "WARN")
                    phase.notes.append(f"Kein Tool-Call: {label}")

            except Exception as combo_err:
                _log(f"    ❌ Fehler bei [{label}]: {combo_err}", verbose, "ERR")
                phase.notes.append(f"Exception bei {label}: {str(combo_err)[:80]}")

        phase.duration_s = time.time() - t0

        if not phase.passed:
            phase.error = "Kein Kombo aus tool_choice × stream hat Tool-Calls produziert."
            _log(phase.error, verbose, "ERR")

    except Exception as e:
        phase.error = str(e)
        _log(f"Phase-2-Exception: {e}", verbose, "ERR")

    return phase, working_config


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 – FlowAgent a_run_llm_completion
# ─────────────────────────────────────────────────────────────────────────────

async def _phase3_flow_agent_wrapper(
    model: str, verbose: bool, working_config: dict | None
) -> PhaseResult:
    """
    Testet a_run_llm_completion des FlowAgents mit Tools.
    Prüft ob tool_calls korrekt rekonstruiert werden.
    """
    phase = PhaseResult(name="Phase 3 – FlowAgent.a_run_llm_completion")
    _log("Starte Phase 3: FlowAgent-Wrapper...", verbose)

    # Wähle beste Konfiguration aus Phase 2, Fallback auf defaults
    tool_choice = (working_config or {}).get("tool_choice", "auto")
    use_stream   = (working_config or {}).get("stream", False)

    t0 = time.time()
    try:
        from toolboxv2 import get_app
        from toolboxv2.mods.isaa.base.Agent.builder import AgentConfig, FlowAgentBuilder
        from toolboxv2.mods.isaa.base.Agent.types import AgentModelData

        # Minimalen FlowAgent aufbauen
        amd = AgentModelData(
            name="ollama_test_agent",
            fast_llm_model=model,
            complex_llm_model=model,
            system_message=(
                "You are a test agent. When the user asks what time it is, "
                "you MUST call the get_current_time tool. "
                "When asked to add numbers, use the add_numbers tool."
            ),
            temperature=0.1,
            max_tokens_output=512,
        )

        # Minimaler FlowAgent-Build ohne Checkpoints / Persistenz
        from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
        agent = FlowAgent(amd=amd, verbose=verbose, auto_load_checkpoint=False)

        messages = [
            {"role": "user", "content": "What is the current time? Use get_current_time."},
        ]

        response_msg = await agent.a_run_llm_completion(
            messages=messages,
            tools=TEST_TOOL_SCHEMA,
            tool_choice=tool_choice if tool_choice is not None else litellm_auto_or_none(model),
            stream=use_stream,
            get_response_message=True,
            with_context=False,
            task_id="ollama_tool_test",
        )

        phase.duration_s = time.time() - t0

        if response_msg is None:
            phase.error = "a_run_llm_completion returned None"
        else:
            calls = _extract_tool_calls_from_response(response_msg)
            phase.raw_content = getattr(response_msg, "content", "") or ""
            if calls:
                phase.tool_called = True
                phase.tool_name = calls[0]["name"]
                phase.tool_args = calls[0]["args"]
                phase.passed = True
                phase.notes.append(f"Tool-Call durch FlowAgent-Wrapper: {phase.tool_name}({phase.tool_args})")
                _log(f"FlowAgent lieferte Tool-Call: {phase.tool_name}", verbose, "OK")
            else:
                phase.notes.append(
                    "FlowAgent-Wrapper lieferte keinen Tool-Call. "
                    "Prüfe ob litellm.drop_params die Tools entfernt oder ob finish_reason='stop' statt 'tool_calls'."
                )
                _log("Kein Tool-Call aus FlowAgent-Wrapper!", verbose, "WARN")
                _log(f"Response content: {(phase.raw_content or '')[:200]}", verbose)

                # Hilfreicher Hinweis: finish_reason prüfen
                if hasattr(response_msg, "_raw_response"):
                    fr = getattr(response_msg._raw_response, "choices", [{}])[0]
                    _log(f"finish_reason im raw response: {getattr(fr, 'finish_reason', '?')}", verbose)

        await agent.close()

    except Exception as e:
        phase.duration_s = time.time() - t0
        phase.error = str(e)
        _log(f"Phase-3-Exception: {e}", verbose, "ERR")
        import traceback
        if verbose:
            traceback.print_exc()

    return phase


def litellm_auto_or_none(model: str) -> str | None:
    """Heuristik: Ollama-Modelle kennen tool_choice='auto' nicht immer."""
    if model.startswith("ollama"):
        return None  # Sicherer Fallback für ältere Ollama-Versionen
    return "auto"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 – Voller ExecutionEngine Lauf
# ─────────────────────────────────────────────────────────────────────────────

async def _phase4_execution_engine(
    model: str, verbose: bool
) -> PhaseResult:
    """
    Vollständiger Agenten-Lauf mit ExecutionEngine.
    Registriert ein echtes Test-Tool und prüft ob es aufgerufen wird.
    """
    phase = PhaseResult(name="Phase 4 – ExecutionEngine vollständig")
    _log("Starte Phase 4: Voller ExecutionEngine-Lauf...", verbose)

    tool_called_flag = {"called": False, "name": None, "args": None}

    async def get_current_time_impl(timezone: str = "UTC") -> str:
        """Returns the current time. Call this when asked about time."""
        tool_called_flag["called"] = True
        tool_called_flag["name"] = "get_current_time"
        tool_called_flag["args"] = {"timezone": timezone}
        return f"{_TEST_TOOL_CALL_MARKER} { datetime.datetime.now(datetime.UTC)}Z (tz={timezone})"

    async def add_numbers_impl(a: float, b: float) -> str:
        """Adds two numbers. Call this to compute a sum."""
        tool_called_flag["called"] = True
        tool_called_flag["name"] = "add_numbers"
        tool_called_flag["args"] = {"a": a, "b": b}
        return f"{_TEST_TOOL_CALL_MARKER} {a} + {b} = {a + b}"

    t0 = time.time()
    try:
        if verbose:
            os.putenv("AGENT_VERBOSE", str(verbose).lower())
        from toolboxv2.mods.isaa.base.Agent.types import AgentModelData
        from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
        from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionEngine

        amd = AgentModelData(
            name="ollama_ee_test",
            fast_llm_model=model,
            complex_llm_model=model,
            system_message=(
                "You are an assistant. "
                "You MUST use tools to answer questions about time or math. "
                "Never answer without calling the right tool first."
            ),
            temperature=0.1,
        )

        agent = FlowAgent(amd=amd, verbose=verbose, auto_load_checkpoint=False, stream=False)

        # Tools registrieren
        agent.tool_manager.register(
            func=get_current_time_impl,
            name="get_current_time",
            description="Returns the current time. Use when asked what time it is.",
            category=["time", "utility"],
        )
        agent.tool_manager.register(
            func=add_numbers_impl,
            name="add_numbers",
            description="Adds two numbers. Use for addition tasks.",
            category=["math", "utility"],
        )

        # Session erstellen
        session_id = f"ollama_diag_{int(time.time())}"
        agent.stream = False
        agent.verbose = verbose
        # Agenten laufen lassen – kurze Iteration damit der Test schnell ist
        result = await agent.a_run(
            query="Please tell me what time it is right now. Use the get_current_time tool.",
            session_id=session_id,
            max_iterations=5,
        )

        phase.duration_s = time.time() - t0
        phase.raw_content = str(result)[:500]

        if tool_called_flag["called"]:
            phase.tool_called = True
            phase.tool_name = tool_called_flag["name"]
            phase.tool_args = tool_called_flag["args"]
            phase.passed = True
            phase.notes.append(
                f"ExecutionEngine rief Tool auf: {phase.tool_name}({phase.tool_args})\n"
                f"Finale Antwort: {result[:200]}"
            )
            _log(f"Tool wurde aufgerufen: {phase.tool_name}", verbose, "OK")
            _log(f"Finale Antwort: {result[:300]}", verbose)
        else:
            phase.notes.append(
                f"ExecutionEngine rief KEIN Tool auf!\n"
                f"Finale Antwort des Agenten: {result[:300]}"
            )
            _log("ExecutionEngine hat kein Tool aufgerufen!", verbose, "ERR")
            _log(f"Agent-Antwort: {result[:300]}", verbose, "WARN")

            # Diagnosetipp ausgeben
            phase.notes.append(
                "DIAGNOSE-TIPP: Prüfe execution_engine.py Zeile ~1250 "
                "ob tool_choice='auto' bei Ollama zu Problemen führt. "
                "Workaround: tool_choice auf None setzen für ollama/* Modelle."
            )

        # await agent.close()

    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        phase.duration_s = time.time() - t0
        phase.error = str(e)
        _log(f"Phase-4-Exception: {e}", verbose, "ERR")
        if verbose:
            import traceback
            traceback.print_exc()

    return phase


# ─────────────────────────────────────────────────────────────────────────────
# Hauptfunktion
# ─────────────────────────────────────────────────────────────────────────────

async def test_ollama_tool_calling(
    model: str = "ollama/qwen3.5:0.8b",
    verbose: bool = True,
    skip_phase4: bool = False,
) -> DiagnosticReport:
    """
    Führt alle 4 Diagnose-Phasen durch und gibt einen strukturierten Report zurück.

    Args:
        model:        Vollständiger litellm Modell-String, z.B. "ollama/qwen3.5:0.8b"
        verbose:      Gibt Zwischenschritte auf stdout aus
        skip_phase4:  Phase 4 (ExecutionEngine) überspringen (schnellerer Test)

    Returns:
        DiagnosticReport mit Ergebnissen und Empfehlung
    """
    report = DiagnosticReport(model=model)

    print(f"\n🔬 Starte Ollama Tool-Call Diagnostik für: {model}")
    print("=" * 60)

    # ── Phase 1 ──
    p1 = await _phase1_raw_litellm(model, verbose)
    report.phases.append(p1)

    # ── Phase 2 ──
    p2, working_config = await _phase2_tool_choice_matrix(model, verbose)
    report.phases.append(p2)
    report.working_config = working_config

    # ── Phase 3 ──
    p3 = await _phase3_flow_agent_wrapper(model, verbose, working_config)
    report.phases.append(p3)

    # ── Phase 4 (optional) ──
    if not skip_phase4:
        p4 = await _phase4_execution_engine(model, verbose)
        report.phases.append(p4)
    else:
        report.phases.append(PhaseResult(
            name="Phase 4 – ExecutionEngine vollständig",
            notes=["Übersprungen (skip_phase4=True)"]
        ))

    # ── Empfehlung generieren ──
    any_passed = any(p.passed for p in report.phases)
    all_passed = all(p.passed for p in report.phases if not skip_phase4 or "Phase 4" not in p.name)

    if all_passed:
        report.recommendation = (
            f"Modell {model} unterstützt Tool-Calls vollständig. "
            "Keine Änderungen nötig."
        )
    elif any_passed:
        cfg_label = (working_config or {}).get("combo_label", "unbekannt")
        report.ollama_workaround_needed = True
        report.recommendation = (
            f"Tool-Calls funktionieren nur mit [{cfg_label}]. "
            "Füge Ollama-Workaround in execution_engine.py ein (siehe unten)."
        )
    else:
        report.ollama_workaround_needed = True
        report.recommendation = (
            f"Modell {model} unterstützt keine nativen Tool-Calls via litellm. "
            "Erwäge ein anderes Modell (z.B. qwen3.5:0.8b, qwen2.5, mistral-nemo) "
            "oder aktiviere den XML-Tool-Call-Fallback."
        )

    print(report.summary())

    if report.ollama_workaround_needed:
        _print_workaround(model, working_config)

    return report


def _print_workaround(model: str, working_config: dict | None):
    """Gibt den konkreten Code-Patch aus der in execution_engine.py eingefügt werden soll."""
    tc = (working_config or {}).get("tool_choice", None)
    stream = (working_config or {}).get("stream", False)

    print("\n" + "─" * 60)
    print("🔧  WORKAROUND – In execution_engine.py (Zeile ~1239) einfügen:")
    print("─" * 60)
    print(f"""
    # ── OLLAMA tool_choice Workaround ──────────────────────────
    _model_for_call = (
        ctx.active_persona.model_preference == "fast"
        and self.agent.amd.fast_llm_model
        or self.agent.amd.complex_llm_model
    )
    _is_ollama = _model_for_call.startswith("ollama")
    _tool_choice = None if _is_ollama else "auto"
    # ────────────────────────────────────────────────────────────

    response = await self.agent.a_run_llm_completion(
        messages=messages,
        tools=current_tools,
        tool_choice=_tool_choice,   # <── geändert von "auto"
        **llm_kwargs,
    )
    """)

    print("─" * 60)
    print("⚡  Alternativ: In flow_agent.py a_run_llm_completion,")
    print("    nach 'litellm.drop_params = True' einfügen:")
    print("""
    # Ollama ignoriert tool_choice='auto' bei einigen Modellen
    if model.startswith("ollama") and llm_kwargs.get("tool_choice") == "auto":
        llm_kwargs.pop("tool_choice", None)
    """)
    print("─" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone / __main__ Integration
# ─────────────────────────────────────────────────────────────────────────────

def run_ollama_tool_diagnostic_cli():
    """
    CLI-Einstiegspunkt.
    Aufruf: python module.py --test-ollama [model] [--verbose] [--skip-phase4]
    """
    import argparse

    parser = argparse.ArgumentParser(description="ISAA Ollama Tool-Call Diagnostik")
    parser.add_argument(
        "--test-ollama",
        nargs="?",
        const="ollama/qwen3.5:0.8b",
        metavar="MODEL",
        help="Modell-String, z.B. ollama/qwen2.5 (Standard: ollama/qwen3.5:0.8b)",
    )
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--skip-phase4", action="store_true", default=False,
                        help="Phase 4 (ExecutionEngine) überspringen für schnelleren Test")

    args, _ = parser.parse_known_args()

    if args.test_ollama:
        report = asyncio.run(
            test_ollama_tool_calling(
                model=args.test_ollama,
                verbose=args.verbose,
                skip_phase4=args.skip_phase4,
            )
        )
        # Exit-Code: 0 = alles ok, 1 = Probleme
        sys.exit(0 if all(p.passed for p in report.phases if not args.skip_phase4 or "Phase 4" not in p.name) else 1)


# ─────────────────────────────────────────────────────────────────────────────
# Integration in module.py – Als Methode auf Tools-Klasse hinzufügen:
# ─────────────────────────────────────────────────────────────────────────────
#
# In der Tools-Klasse (module.py), am Ende der Methoden-Definitionen:
#
#     async def test_ollama_tool_calling(
#         self,
#         model: str | None = None,
#         verbose: bool = True,
#         skip_phase4: bool = False,
#     ) -> dict:
#         """
#         Führt Ollama Tool-Call Diagnostik durch.
#         Gibt strukturierten Report als Dict zurück.
#
#         Args:
#             model: litellm Modell-String (Standard: self.config["FASTMODEL"])
#             verbose: Gibt Diagnose-Logs aus
#             skip_phase4: Nur Phasen 1-3 (schneller, kein ExecutionEngine-Lauf)
#         """
#         from toolboxv2.mods.isaa.ollama_tool_test import test_ollama_tool_calling as _run_test
#         _model = model or self.config.get("FASTMODEL", "ollama/qwen3.5:0.8b")
#         report = await _run_test(_model, verbose=verbose, skip_phase4=skip_phase4)
#         return {
#             "model": report.model,
#             "passed_all": all(p.passed for p in report.phases),
#             "ollama_workaround_needed": report.ollama_workaround_needed,
#             "working_config": report.working_config,
#             "recommendation": report.recommendation,
#             "phases": [
#                 {
#                     "name": p.name,
#                     "passed": p.passed,
#                     "tool_called": p.tool_called,
#                     "tool_name": p.tool_name,
#                     "duration_s": round(p.duration_s, 2),
#                     "error": p.error,
#                     "notes": p.notes,
#                 }
#                 for p in report.phases
#             ],
#         }
#
# ─────────────────────────────────────────────────────────────────────────────
# Und am Ende des __main__-Blocks in module.py:
#
#   if __name__ == "__main__":
#       import sys
#       if "--test-ollama" in sys.argv:
#           from toolboxv2.mods.isaa.ollama_tool_test import run_ollama_tool_diagnostic_cli
#           run_ollama_tool_diagnostic_cli()
#       else:
#           asyncio.run(test_isaa_tools())  # bisheriger Test
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Standalone-Aufruf: python ollama_tool_test.py ollama/qwen3.5:0.8b
    _model = sys.argv[1] if len(sys.argv) > 1 else "ollama/mistral:latest"
    _verbose = "--quiet" not in sys.argv
    _skip4 = False#"--skip-phase4" in sys.argv
    import litellm
    litellm.register_model(
        model_cost={
            _model: {
                "supports_function_calling": True
            }
        }
    )
    asyncio.run(test_ollama_tool_calling(_model, verbose=_verbose, skip_phase4=_skip4))
