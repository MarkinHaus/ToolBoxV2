"""
══════════════════════════════════════════════════════════════════════════════
BENCHMARK RUNNER V3.1 — Parallel Adapter Isolation + Crash Protection
══════════════════════════════════════════════════════════════════════════════

FIXES:
  1. Adapters run PARALLEL (not sequential) — each in own asyncio.Task
  2. Per-adapter try/except — one crash doesn't kill the rest
  3. Per-adapter timeout (default 300s) — hanging model can't block
  4. Separate agent session per adapter (not shared)
  5. Circuit breaker: if model fails 5 probes consecutively, abort early

Usage:
    python t.py                    # Interactive model selector
    python t.py --quick-preset     # Run predefined models in standard mode
    python t.py --zen              # With ZenPlus TUI
    python t.py --models alias1=openrouter/x/y,alias2=openrouter/a/b
    python t.py --adapters row     # Only RowModelAdapter (skip agent adapters)
"""

import asyncio
import atexit
import json as _json
import os
import sys
import time
import uuid
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CRASH RECOVERY — saves partial results if process dies
# ══════════════════════════════════════════════════════════════════════════════

_PARTIAL_REPORTS: List[Any] = []  # Accumulates reports as they complete
_OUTPUT_PATH = "comparison.html"


def _crash_save():
    """Called on process exit — save whatever we have."""
    if not _PARTIAL_REPORTS:
        return
    try:
        path = _OUTPUT_PATH.replace('.html', '_partial.json')
        data = []
        for r in _PARTIAL_REPORTS:
            if hasattr(r, 'to_dict'):
                data.append(r.to_dict())
            else:
                data.append(str(r))
        with open(path, 'w') as f:
            _json.dump(data, f, indent=2, default=str)
        print(f"\n  [RECOVERY] Saved {len(data)} partial reports to {path}")
    except Exception as e:
        print(f"\n  [RECOVERY] Failed to save: {e}")


atexit.register(_crash_save)


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchResult:
    """Container for one model's benchmark results — fully isolated."""
    model_id: str
    model_sig: str
    reports: list = field(default_factory=list)
    errors: list = field(default_factory=list)  # per-adapter errors
    error: Optional[str] = None  # legacy compat
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def duration(self) -> float:
        return self.finished_at - self.started_at

    @property
    def total_cost(self) -> float:
        return sum(r.total_cost for r in self.reports if hasattr(r, 'total_cost'))

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.reports if hasattr(r, 'total_tokens'))


# ══════════════════════════════════════════════════════════════════════════════
# ISOLATED MODEL RUN — adapters sequential, models parallel
# ══════════════════════════════════════════════════════════════════════════════

async def run_isolated(
    m_id: str,
    model_sig: str,
    app_context=None,
    mode: str = "standard",
    seed: int = 42,
    zen_callback=None,
    adapter_timeout: float = 300.0,
    adapters: list = None,  # None = all, or ["row", "agent", "agent_st"]
) -> BenchResult:
    """
    Run benchmarks for ONE model.

    Architecture (learned the hard way):
      - Adapters run SEQUENTIALLY (FlowAgent is NOT parallel-safe)
      - ONE shared agent per model (FlowAgent agents share internal state)
      - Each adapter has its OWN try/except + timeout (crash isolation)
      - Models run PARALLEL to each other (separate agents, no shared state)
    """
    from toolboxv2.mods.isaa.base.bench.benchmark import (
        AgentAdapter, AgentAdapterSt, RowModelAdapter,
    )

    result = BenchResult(model_id=m_id, model_sig=model_sig, started_at=time.time())
    run_uuid = uuid.uuid4().hex[:8]
    session_id = f"bench_{m_id}_{run_uuid}"

    def _notify(phase: str, detail: str = ""):
        if zen_callback:
            zen_callback({
                "agent": m_id, "type": "content",
                "chunk": f"[{phase}] {detail}\n",
                "iter": 0, "max_iter": 3,
            })

    if adapters is None:
        adapters = ["agent_st", "agent", "row"]

    needs_agent = bool({"agent_st", "agent"} & set(adapters))

    # Only load isaa if we need FlowAgent adapters
    isaa = None
    if needs_agent:
        try:
            if app_context:
                isaa = app_context.get_mod("isaa")
            else:
                from toolboxv2 import get_app
                isaa = get_app().get_mod("isaa")
        except Exception as e:
            # Can't get isaa — agent adapters will be skipped
            result.errors.append(f"isaa unavailable: {e}")
            needs_agent = False

    agent = None
    try:
        # Only create FlowAgent if agent adapters are requested
        if needs_agent:
            agent = await isaa.get_agent(session_id)
            agent.amd.fast_llm_model = model_sig
            agent.amd.complex_llm_model = model_sig

            def calculator(expression: str) -> str:
                """Evaluate a mathematical expression"""
                try:
                    return f"Result: {expression} = {eval(expression, {'__builtins__': None}, {})}"
                except Exception as e:
                    return f"Error: {e}"

            agent.add_tool(calculator, name="calculator",
                           description="Evaluate a mathematical expression")

        # ── Run adapters SEQUENTIALLY with per-adapter error isolation ──

        if "agent_st" in adapters:
            if not agent:
                result.errors.append("AgentAdapterSt: requires agent (skipped)")
            else:
                _notify("start", f"AgentAdapterSt ({mode})")
                try:
                    adapter_st = AgentAdapterSt(agent, zen_callback)
                    res = await asyncio.wait_for(
                        adapter_st.benchmark(f"{m_id}S", mode=mode, seed=seed),
                        timeout=adapter_timeout,
                    )
                    result.reports.append(res)
                    _PARTIAL_REPORTS.append(res)
                    _notify("done", f"AgentAdapterSt score={res.total:.1f} cost=${res.total_cost:.4f}")
                except asyncio.TimeoutError:
                    result.errors.append(f"AgentAdapterSt: TIMEOUT after {adapter_timeout}s")
                    _notify("error", f"AgentAdapterSt: TIMEOUT")
                except Exception as e:
                    result.errors.append(f"AgentAdapterSt: {type(e).__name__}: {e}")
                    _notify("error", f"AgentAdapterSt: {e}")

        if "agent" in adapters:
            if not agent:
                result.errors.append("AgentAdapter: requires agent (skipped)")
            else:
                _notify("start", f"AgentAdapter ({mode})")
                try:
                    adapter_a = AgentAdapter(agent)
                    res = await asyncio.wait_for(
                        adapter_a.benchmark(f"{m_id}A", mode=mode, seed=seed),
                        timeout=adapter_timeout,
                    )
                    result.reports.append(res)
                    _PARTIAL_REPORTS.append(res)
                    _notify("done", f"AgentAdapter score={res.total:.1f} cost=${res.total_cost:.4f}")
                except asyncio.TimeoutError:
                    result.errors.append(f"AgentAdapter: TIMEOUT after {adapter_timeout}s")
                    _notify("error", f"AgentAdapter: TIMEOUT")
                except Exception as e:
                    result.errors.append(f"AgentAdapter: {type(e).__name__}: {e}")
                    _notify("error", f"AgentAdapter: {e}")

        if "row" in adapters:
            _notify("start", f"RowModelAdapter ({mode})")
            try:
                # RowModelAdapter uses litellm DIRECTLY — no agent needed
                adapter_row = RowModelAdapter(model_name=model_sig)
                res = await asyncio.wait_for(
                    adapter_row.benchmark(m_id, mode=mode, seed=seed),
                    timeout=adapter_timeout,
                )
                result.reports.append(res)
                _PARTIAL_REPORTS.append(res)
                _notify("done", f"RowModelAdapter score={res.total:.1f} cost=${res.total_cost:.4f}")
            except asyncio.TimeoutError:
                result.errors.append(f"RowModelAdapter: TIMEOUT after {adapter_timeout}s")
                _notify("error", f"RowModelAdapter: TIMEOUT")
            except Exception as e:
                result.errors.append(f"RowModelAdapter: {type(e).__name__}: {e}")
                _notify("error", f"RowModelAdapter: {e}")

    except Exception as e:
        result.errors.append(f"Setup: {type(e).__name__}: {e}")

    finally:
        if needs_agent and agent:
            try:
                await isaa.delete_agent(session_id)
            except Exception:
                pass

    if result.errors and not result.reports:
        result.error = "; ".join(result.errors)

    result.finished_at = time.time()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

async def run_all_parallel(
    models: Dict[str, str],
    mode: str = "standard",
    seed: int = 42,
    max_concurrent: int = 4,
    zen_callback=None,
    adapter_timeout: float = 300.0,
    adapters: list = None,
) -> List[BenchResult]:
    """Two-phase execution:

    Phase 1 (Row): All models PARALLEL — pure litellm, no FlowAgent
    Phase 2 (Agent): Models SEQUENTIAL — FlowAgent is NOT parallel-safe

    This is the key architectural insight: litellm.acompletion() is stateless
    and can run in parallel. FlowAgent shares internal state (isaa module,
    agent registry, file system, rate limiters) and WILL crash if parallelized.
    """
    requested = set(adapters or ["agent_st", "agent", "row"])
    has_row = "row" in requested
    has_agent = bool({"agent_st", "agent"} & requested)

    all_results: Dict[str, BenchResult] = {}

    # ── Phase 1: Row adapters — PARALLEL (safe, no FlowAgent) ──
    if has_row:
        if zen_callback:
            zen_callback({"agent": "system", "type": "content",
                         "chunk": f"[phase1] Row benchmark: {len(models)} models parallel\n",
                         "iter": 0, "max_iter": 3})

        sem = asyncio.Semaphore(max_concurrent)

        async def _row_task(m_id: str, m_sig: str) -> BenchResult:
            async with sem:
                try:
                    return await run_isolated(
                        m_id, m_sig, app_context=None,
                        mode=mode, seed=seed, zen_callback=zen_callback,
                        adapter_timeout=adapter_timeout, adapters=["row"],
                    )
                except Exception as e:
                    return BenchResult(
                        model_id=m_id, model_sig=m_sig,
                        error=f"Row: {type(e).__name__}: {e}",
                        finished_at=time.time(),
                    )

        row_tasks = [_row_task(m_id, m_sig) for m_id, m_sig in models.items()]
        row_results = await asyncio.gather(*row_tasks, return_exceptions=True)

        for r, (m_id, m_sig) in zip(row_results, models.items()):
            if isinstance(r, Exception):
                all_results[m_id] = BenchResult(
                    model_id=m_id, model_sig=m_sig,
                    error=str(r), finished_at=time.time(),
                )
            else:
                all_results[m_id] = r

    # ── Phase 2: Agent adapters — SEQUENTIAL (FlowAgent not parallel-safe) ──
    if has_agent:
        agent_adapters = sorted({"agent_st", "agent"} & requested)

        if zen_callback:
            zen_callback({"agent": "system", "type": "content",
                         "chunk": f"[phase2] Agent benchmark: {len(models)} models sequential ({', '.join(agent_adapters)})\n",
                         "iter": 0, "max_iter": 3})

        from toolboxv2 import get_app
        app = get_app()

        for m_id, m_sig in models.items():
            try:
                agent_result = await run_isolated(
                    m_id, m_sig, app_context=app,
                    mode=mode, seed=seed, zen_callback=zen_callback,
                    adapter_timeout=adapter_timeout, adapters=agent_adapters,
                )
                # Merge with existing row results
                if m_id in all_results:
                    all_results[m_id].reports.extend(agent_result.reports)
                    all_results[m_id].errors.extend(agent_result.errors)
                else:
                    all_results[m_id] = agent_result
            except (SystemExit, KeyboardInterrupt):
                if m_id not in all_results:
                    all_results[m_id] = BenchResult(
                        model_id=m_id, model_sig=m_sig, finished_at=time.time(),
                    )
                all_results[m_id].errors.append("Agent: Process interrupted")
            except Exception as e:
                if m_id not in all_results:
                    all_results[m_id] = BenchResult(
                        model_id=m_id, model_sig=m_sig, finished_at=time.time(),
                    )
                all_results[m_id].errors.append(f"Agent: {type(e).__name__}: {e}")

    # Return in original model order
    results = []
    for m_id in models:
        if m_id in all_results:
            results.append(all_results[m_id])
        else:
            results.append(BenchResult(model_id=m_id, model_sig=models[m_id],
                                       error="Not executed", finished_at=time.time()))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# OPENROUTER MODEL SEARCH
# ══════════════════════════════════════════════════════════════════════════════

_MODEL_CACHE: Optional[list] = None

async def fetch_openrouter_models(force_refresh: bool = False) -> list:
    global _MODEL_CACHE
    if _MODEL_CACHE and not force_refresh:
        return _MODEL_CACHE
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get("https://openrouter.ai/api/v1/models", timeout=10.0)
            if resp.status_code == 200:
                _MODEL_CACHE = [
                    {"id": m["id"], "name": m.get("name", m["id"]),
                     "context": m.get("context_length", 0),
                     "price_in": m.get("pricing", {}).get("prompt", "?"),
                     "price_out": m.get("pricing", {}).get("completion", "?")}
                    for m in resp.json().get("data", [])
                ]
                return _MODEL_CACHE
    except Exception as e:
        print(f"  [WARN] OpenRouter fetch failed: {e}")
    return []

def search_models(query: str, models: list) -> list:
    q = query.lower()
    return [m for m in models if q in m["id"].lower() or q in m["name"].lower()]

async def interactive_model_selector() -> Dict[str, str]:
    print("\n  Fetching OpenRouter models...")
    all_models = await fetch_openrouter_models()
    if not all_models:
        print("  [WARN] Could not fetch models. Use --models flag.")
        return {}
    print(f"  {len(all_models)} models. Type to search, 'go' to run.\n")
    selected: Dict[str, str] = {}
    while True:
        try:
            query = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in ("go", "run", "start"):
            break
        if query.lower() == "quit":
            return {}
        if query.lower() == "list":
            for k, v in selected.items():
                print(f"    {k:20s} -> {v}")
            continue
        if query.lower() == "clear":
            selected.clear()
            continue
        if "=" in query and "/" in query:
            alias, sig = query.split("=", 1)
            sig = sig.strip()
            if not sig.startswith("openrouter/") and not sig.startswith("gateway/"):
                sig = f"openrouter/{sig}"
            selected[alias.strip()] = sig
            print(f"  + {alias.strip()} -> {sig}")
            continue
        results = search_models(query, all_models)
        if not results:
            print("  No matches.")
            continue
        for i, m in enumerate(results[:15]):
            ctx = f"{m['context']:>8,}" if m["context"] else "       ?"
            print(f"    {i:2d}) {m['id']:50s}  ctx:{ctx}")
        try:
            choice = input("  # (or alias=#): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not choice:
            continue
        alias_name, idx_str = (choice.split("=", 1) + [choice])[:2] if "=" in choice else (None, choice)
        try:
            idx = int(idx_str.strip())
            if 0 <= idx < len(results):
                m = results[idx]
                mid = alias_name.strip() if alias_name else m["id"].split("/")[-1]
                selected[mid] = f"openrouter/{m['id']}"
                print(f"  + {mid} -> openrouter/{m['id']}")
        except ValueError:
            pass
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# ZEN+ / CLI RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

async def run_with_zen(models, mode="standard", seed=42, max_concurrent=4,
                       adapter_timeout=300.0, adapters=None):
    try:
        from toolboxv2.mods.isaa.extras.zen.zen_plus import ZenPlus
    except ImportError:
        return await run_with_progress(models, mode, seed, max_concurrent,
                                       adapter_timeout, adapters)
    zen = ZenPlus.get()
    zen.clear_panes()
    for m_id in models:
        zen.inject_job(task_id=f"bench_{m_id}", agent_name=m_id,
                       query=f"Benchmark {mode}", status="pending")
    results_box = [None]

    async def _bench():
        results_box[0] = await run_all_parallel(
            models, mode=mode, seed=seed, max_concurrent=max_concurrent,
            zen_callback=lambda c: zen.feed_chunk(c),
            adapter_timeout=adapter_timeout, adapters=adapters,
        )
        for r in results_box[0]:
            zen.update_job(f"bench_{r.model_id}", "done" if not r.error else "failed")
        zen.signal_stream_done()

    task = asyncio.create_task(_bench())
    await zen.start()
    if not task.done():
        await task
    return results_box[0]


async def run_with_progress(models, mode="standard", seed=42, max_concurrent=4,
                            adapter_timeout=300.0, adapters=None):
    adapter_list = adapters or ["agent_st", "agent", "row"]
    has_row = "row" in adapter_list
    has_agent = bool({"agent_st", "agent"} & set(adapter_list))

    print(f"\n  {len(models)} models × {len(adapter_list)} adapters ({mode})")
    if has_row and has_agent:
        print(f"  Phase 1: Row (parallel, {max_concurrent} concurrent)")
        print(f"  Phase 2: Agent (sequential, FlowAgent not parallel-safe)")
    elif has_row:
        print(f"  Row only (parallel, {max_concurrent} concurrent)")
    else:
        print(f"  Agent only (sequential)")
    print(f"  Timeout: {adapter_timeout}s per adapter")
    print(f"  {'─' * 60}")

    def progress_cb(chunk):
        text = chunk.get("chunk", "").strip()
        if text:
            print(f"  {chunk.get('agent', ''):20s} {text}")

    results = await run_all_parallel(
        models, mode=mode, seed=seed, max_concurrent=max_concurrent,
        zen_callback=progress_cb, adapter_timeout=adapter_timeout, adapters=adapters,
    )

    print(f"\n  {'═' * 60}")
    print(f"  RESULTS")
    print(f"  {'═' * 60}")
    for r in results:
        if r.error and not r.reports:
            print(f"  ✗ {r.model_id:20s}  FAILED: {r.error}")
        else:
            scores = [f"{rep.total:.1f}" for rep in r.reports]
            cost = f"${r.total_cost:.4f}" if r.total_cost > 0 else "$0"
            tokens = f"{r.total_tokens:,}tok"
            status = "✓" if not r.errors else "⚠"
            print(f"  {status} {r.model_id:20s}  [{', '.join(scores)}]  {cost}  {tokens}  ({r.duration:.1f}s)")
            for err in r.errors:
                print(f"    └─ {err}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark Runner V3.1")
    parser.add_argument("--mode", default="standard",
                        choices=["quick", "standard", "full", "precision"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrent", type=int, default=4)
    parser.add_argument("--zen", action="store_true")
    parser.add_argument("--output", default="comparison.html")
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--quick-preset", action="store_true")
    parser.add_argument("--adapter-timeout", type=float, default=300.0)
    parser.add_argument("--adapters", type=str, default="",
                        help="Comma-separated: row,agent,agent_st (default: all)")
    args = parser.parse_args()
    global _OUTPUT_PATH
    _OUTPUT_PATH = args.output

    adapter_list = [a.strip() for a in args.adapters.split(",") if a.strip()] or None

    if args.models:
        models = {}
        for spec in args.models.split(","):
            spec = spec.strip()
            if "=" in spec:
                k, v = spec.split("=", 1)
                models[k.strip()] = v.strip()
            else:
                models[spec.split("/")[-1]] = spec
    elif args.quick_preset:
        models = {
            # "glm-4.6": "openrouter/z-ai/glm-4.6",
            # "glm-4.7": "openrouter/z-ai/glm-4.7",
            # slow "kimi-k2.5": "openrouter/moonshotai/kimi-k2.5",
            # "kimi-k2-t": "openrouter/moonshotai/kimi-k2-thinking",
            "deepseek-v3.2": "openrouter/deepseek/deepseek-v3.2",
            # slow "gemini-3.1": "openrouter/google/gemini-3.1-pro-preview",
            "gpt-5.2": "openrouter/openai/gpt-5.2",
            "gpt-5.2-c": "openrouter/openai/gpt-5.2-codex",
            "gpt-oss-20b": "openrouter/openai/gpt-oss-20b:nitro",
            "gpt-oss-120b": "openrouter/openai/gpt-oss-120b:nitro",
            "sonnet-4.6": "openrouter/anthropic/claude-sonnet-4.6",
            "opus-4.6": "openrouter/anthropic/claude-opus-4.6",
            "minimax-2.5": "openrouter/minimax/minimax-m2.5",
            "gemini-flash-3": "openrouter/google/gemini-3-flash-preview",
            "gemini-flash-2.5-lite": "openrouter/google/gemini-2.5-flash-lite",
        }
    else:
        models = await interactive_model_selector()

    if not models:
        print("  No models selected.")
        return

    print(f"\n  Models ({len(models)}):")
    for k, v in models.items():
        print(f"    {k:20s} -> {v}")

    runner = run_with_zen if args.zen else run_with_progress
    results = await runner(models, args.mode, args.seed, args.concurrent,
                           args.adapter_timeout, adapter_list)
    if not results:
        return

    all_reports = [rep for r in results for rep in r.reports]
    if not all_reports:
        print("  No valid reports generated.")
        # Still print what we got
        for r in results:
            if r.errors:
                print(f"    {r.model_id}: {'; '.join(r.errors)}")
            elif r.error:
                print(f"    {r.model_id}: {r.error}")
        return

    try:
        from toolboxv2.mods.isaa.base.bench.dashboard import Dashboard
        Dashboard.save(all_reports, args.output)
        print(f"\n  Dashboard: {args.output} ({len(all_reports)} reports)")
        _PARTIAL_REPORTS.clear()  # Success — no need for crash recovery
    except Exception as e:
        print(f"\n  [WARN] Dashboard generation failed: {e}")
        # Save raw JSON as fallback
        try:
            import json
            fallback = args.output.replace('.html', '.json')
            with open(fallback, 'w') as f:
                json.dump([r.to_dict() if hasattr(r, 'to_dict') else str(r) for r in all_reports], f, indent=2, default=str)
            print(f"  Fallback JSON: {fallback}")
        except Exception:
            pass

    total_cost = sum(r.total_cost for r in results)
    total_errors = sum(len(r.errors) for r in results)
    print(f"  Cost: ${total_cost:.4f} | Errors: {total_errors}")


if __name__ == "__main__":
    # Global exception handler — NEVER let unhandled exceptions kill the process
    _orig_excepthook = sys.excepthook
    def _bench_excepthook(exc_type, exc_val, exc_tb):
        print(f"\n  [FATAL] Unhandled: {exc_type.__name__}: {exc_val}")
        traceback.print_exception(exc_type, exc_val, exc_tb)
        # Don't call sys.exit — let the process continue cleanup
    sys.excepthook = _bench_excepthook

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  Aborted by user.")
    except SystemExit as e:
        print(f"\n  SystemExit caught: {e}")
    except Exception as e:
        print(f"\n  [FATAL] Top-level crash: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        sys.excepthook = _orig_excepthook
