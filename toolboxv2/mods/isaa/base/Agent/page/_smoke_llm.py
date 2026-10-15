"""Manual LLM-call smoke / E2E for the Page engine (lives in the page/ folder).

Validates the engine against a REAL ISAA agent + LLM router — the check that the
unit tests (which use ScriptedLLM/MockRouter) cannot do.  Primary purpose:
reproduce and confirm the fix for the GLM 1214 "messages parameter is illegal"
HTTP 400 that appeared on the action step right after a context_fold.

Run (Windows, UTF-8 required so emojis don't crash cp1252):

    $env:PYTHONUTF8=1
    cd C:\\Users\\Markin\\Workspace\\ToolBoxV2
    python -m toolboxv2.mods.isaa.base.Agent.page._smoke_llm self
    # or a quick drop-in (V3 adapter) check:
    python -m toolboxv2.mods.isaa.base.Agent.page._smoke_llm self --engine

What it does:
  1. Loads the ISAA agent (same chain as cli.py).
  2. Builds the Page engine with small caps so a fold happens quickly.
  3. Wraps the LLM port to CAPTURE every message array actually sent, then
     asserts each array is provider-safe (no trailing/mid system, assistant
     content never None, every tool msg has a matching assistant tool_call).
  4. Streams a multi-step query (forces tool turns -> fold -> next action),
     printing tokens live, and reports whether any provider error occurred.

Exit code 0 = all sent arrays were provider-safe AND no LLM error.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys


def _check_provider_safe(msgs: list[dict]) -> list[str]:
    """Return a list of rule violations (empty = safe)."""
    problems: list[str] = []
    roles = [m.get("role") for m in msgs]
    if roles and roles[-1] == "system":
        problems.append(f"trailing system: {roles}")
    first_non_sys = next((i for i, r in enumerate(roles) if r != "system"), len(roles))
    if any(r == "system" for r in roles[first_non_sys:]):
        problems.append(f"system after start: {roles}")
    open_ids: set = set()
    for m in msgs:
        if m.get("role") == "assistant":
            if m.get("content") is None and not m.get("tool_calls"):
                problems.append("assistant content None without tool_calls")
            for tc in m.get("tool_calls") or []:
                open_ids.add(tc.get("id"))
        if m.get("role") == "tool" and m.get("tool_call_id") not in open_ids:
            problems.append(f"orphan tool msg {m.get('tool_call_id')}")
    return problems


async def _load_agent(name: str):
    from toolboxv2 import get_app
    app = get_app("page-smoke")
    isaa = app.get_mod("isaa")
    if hasattr(isaa, "init_isaa"):
        try:
            await isaa.init_isaa(build=True)
        except Exception as e:
            print(f"[warn] init_isaa: {e}")
    return await isaa.get_agent(name)


def _wrap_capture(engine, captured: list):
    orig_stream = getattr(engine.llm, "stream", None)
    orig_complete = engine.llm.complete

    if orig_stream is not None:
        async def wrap_stream(messages, tools):
            captured.append(("stream", [dict(m) for m in messages]))
            async for c in orig_stream(messages, tools):
                yield c
        engine.llm.stream = wrap_stream

    async def wrap_complete(messages, tools):
        captured.append(("complete", [dict(m) for m in messages]))
        return await orig_complete(messages, tools)
    engine.llm.complete = wrap_complete


async def _run_direct(agent) -> int:
    from .isaa import make_page_engine
    from .core import Config

    engine = make_page_engine(agent, root=None, cfg=Config(tail_items=3))
    captured: list = []
    _wrap_capture(engine, captured)

    err = None
    try:
        async for ev in engine.stream(
            "Nenne in genau einem kurzen Satz was ein Mutex ist und in einem "
            "zweiten Satz ein Beispiel. Fasse dich kurz.",
            session="smoke",
        ):
            t = ev.get("type")
            if t == "content_chunk":
                sys.stdout.write(ev["chunk"]); sys.stdout.flush()
            elif t in ("tool_start", "tool_result", "final_answer",
                       "context_step", "iteration_start"):
                safe = {k: v for k, v in ev.items() if k not in ("args", "result")}
                print(f"\n[{t}] {json.dumps(safe)[:160]}")
    except Exception as e:
        err = e
        print(f"\n\n=== LLM/STREAM EXCEPTION === {type(e).__name__}: {str(e)[:400]}")

    violations = 0
    for idx, (kind, msgs) in enumerate(captured):
        probs = _check_provider_safe(msgs)
        if probs:
            violations += len(probs)
            print(f"[UNSAFE call #{idx} ({kind})] {probs}")
    print(f"\ncaptured LLM calls: {len(captured)} | unsafe: {violations} | "
          f"llm_error: {bool(err)}")
    await engine.close()
    return 0 if (violations == 0 and err is None) else 1


async def _run_engine_adapter(agent) -> int:
    """Exercise the V3 drop-in (PageExecutionEngine) through execute_stream."""
    from .isaa import make_page_execution_engine
    from .core import Config

    eng = make_page_execution_engine(agent, root=None, cfg=Config(tail_items=3))
    captured: list = []
    _wrap_capture(eng.engine, captured)
    err = None
    try:
        gen, ctx = await eng.execute_stream("Was ist 2+2? Kurz.", session_id="smoke2")
        async for ev in gen(ctx):
            if ev.get("type") == "content":
                sys.stdout.write(ev["chunk"]); sys.stdout.flush()
            elif ev.get("type") in ("final_answer", "done"):
                print(f"\n[{ev['type']}] {str(ev.get('answer') or ev.get('final_answer'))[:160]}")
    except Exception as e:
        err = e
        print(f"\n=== EXCEPTION === {type(e).__name__}: {str(e)[:400]}")
    violations = sum(len(_check_provider_safe(m)) for _, m in captured)
    print(f"\n[adapter] calls: {len(captured)} | unsafe: {violations} | error: {bool(err)}")
    await eng.close()
    return 0 if (violations == 0 and err is None) else 1


async def _main(args) -> int:
    print(f"Loading agent '{args.agent}' ...")
    agent = await _load_agent(args.agent)
    print(f"agent: {getattr(agent.amd, 'name', args.agent)} | "
          f"fast model: {getattr(agent.amd, 'fast_llm_model', '?')}")
    if args.engine:
        return await _run_engine_adapter(agent)
    return await _run_direct(agent)


def main() -> None:
    p = argparse.ArgumentParser(description="Manual LLM-call smoke for the Page engine")
    p.add_argument("agent", nargs="?", default="self")
    p.add_argument("--engine", action="store_true",
                   help="test via the PageExecutionEngine V3 adapter")
    args = p.parse_args()
    rc = asyncio.run(_main(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
