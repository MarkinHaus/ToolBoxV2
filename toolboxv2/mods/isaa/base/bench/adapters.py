"""
Adapters for different execution targets.
Recycled from old benchmark system, adapted for new Task-based runner.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Callable

from toolboxv2.mods.isaa.base.bench.core import Report, Task
from toolboxv2.mods.isaa.base.bench.loader import filter_by_modality, load_suite, load_tasks_from_dir, resolve_suite
from toolboxv2.mods.isaa.base.bench.runner import run_tasks


def _extract_agent_context(agent) -> dict:
    """Extract tool_calls and working_history from agent's ExecutionEngine context.

    Returns {"tool_calls": [...], "working_history": [...]}.
    tool_calls: [{"name": str, "args": dict, "result": Any}]
    working_history: [{"role": str, "content": str/list}] — full conversation trace
    """
    result = {"tool_calls": [], "working_history": []}
    try:
        engine = agent._get_execution_engine()
        if engine and engine._active_executions:
            ctx = engine._active_executions[-1]
            result["tool_calls"] = list(getattr(ctx, "tools_dict", []))
            result["working_history"] = list(getattr(ctx, "working_history", []))
    except Exception:
        pass
    return result


class _BaseAdapter:
    """Common adapter logic."""

    def __init__(
        self,
        task_dir: str | Path = "",
        suite_path: str | Path = "",
        model_modalities: list[str] | None = None,
        on_progress: Callable | None = None,
    ):
        self.task_dir = Path(task_dir) if task_dir else None
        self.suite_path = Path(suite_path) if suite_path else None
        self.model_modalities = model_modalities or ["text"]
        self.on_progress = on_progress

    def _load_tasks(self) -> list[Task]:
        """Load and filter tasks."""
        if self.task_dir:
            all_tasks = load_tasks_from_dir(self.task_dir)
        else:
            all_tasks = []

        if self.suite_path and self.suite_path.exists():
            suite = load_suite(self.suite_path)
            tasks = resolve_suite(suite, all_tasks)
        else:
            tasks = all_tasks

        # Filter by model capabilities
        tasks = filter_by_modality(tasks, self.model_modalities)
        return tasks


class RowModelAdapter(_BaseAdapter):
    """Direct LiteLLM model testing — no agent, just prompt → response."""

    def __init__(
        self,
        model_name: str,
        task_dir: str | Path = "",
        suite_path: str | Path = "",
        model_modalities: list[str] | None = None,
        timeout: float = 90.0,
        on_progress: Callable | None = None,
    ):
        super().__init__(task_dir, suite_path, model_modalities, on_progress)
        self.model_name = model_name
        self.timeout = timeout

    async def benchmark(
        self,
        model_id: str | None = None,
        mode: str = "standard",
        seed: int | None = None,
    ) -> Report:
        tasks = self._load_tasks()

        async def model_fn(prompt: str) -> tuple[str, dict]:
            import litellm
            litellm.suppress_debug_info = True
            t0 = time.perf_counter()
            try:
                r = await asyncio.wait_for(
                    litellm.acompletion(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                return f"Error: timeout {self.timeout}s", {}
            except Exception as e:
                return f"Error: {type(e).__name__}: {e}", {}

            exec_time = time.perf_counter() - t0
            content = ""
            if r.choices:
                content = r.choices[0].message.content or ""
            tokens_in = getattr(r.usage, "prompt_tokens", 0) if r.usage else 0
            tokens_out = getattr(r.usage, "completion_tokens", 0) if r.usage else 0

            total_cost = 0.0
            try:
                from litellm import completion_cost
                total_cost = completion_cost(r)
            except Exception:
                pass

            # Fallback: OpenRouter includes cost in response headers/metadata
            if total_cost == 0.0:
                try:
                    # litellm wraps openrouter cost in _hidden_params
                    hidden = getattr(r, "_hidden_params", {}) or {}
                    additional = hidden.get("additional_headers", {}) or {}
                    # OpenRouter sends x-openrouter-cost header
                    or_cost = additional.get("x-openrouter-cost")
                    if or_cost:
                        total_cost = float(or_cost)
                except Exception:
                    pass

            # Last fallback: estimate from token count if model pricing known
            if total_cost == 0.0 and (tokens_in or tokens_out):
                try:
                    from litellm import model_cost
                    pricing = model_cost.get(self.model_name, {})
                    if pricing:
                        cost_in = pricing.get("input_cost_per_token", 0) * tokens_in
                        cost_out = pricing.get("output_cost_per_token", 0) * tokens_out
                        total_cost = cost_in + cost_out
                except Exception:
                    pass

            return content, {
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "total_cost": total_cost,
                "execution_time_s": exec_time,
            }

        return await run_tasks(
            tasks, model_fn,
            model_id=model_id or self.model_name,
            suite_id=self.suite_path.stem if self.suite_path else "all",
            mode=mode,
            timeout=self.timeout,
                    on_progress=self.on_progress,
        )


class AgentAdapter(_BaseAdapter):
    """Adapter for ISAA FlowAgent — uses agent.a_run()."""

    def __init__(
        self,
        agent,
        task_dir: str | Path = "",
        suite_path: str | Path = "",
        model_modalities: list[str] | None = None,
        probe_timeout: float = 120.0,
        on_progress: Callable | None = None,
    ):
        super().__init__(task_dir, suite_path, model_modalities, on_progress)
        self.agent = agent
        self.probe_timeout = probe_timeout
        self._counter = 0

    async def benchmark(
        self,
        model_id: str = "agent",
        mode: str = "standard",
        seed: int | None = None,
    ) -> Report:
        tasks = self._load_tasks()

        async def model_fn(prompt: str) -> tuple[str, dict]:
            self._counter += 1
            sid = f"bench_{self._counter}"
            t0 = time.perf_counter()
            try:
                start_cost = getattr(self.agent, "total_cost_accumulated", 0) or 0
                start_tin = getattr(self.agent, "total_tokens_in", 0) or 0
                start_tout = getattr(self.agent, "total_tokens_out", 0) or 0

                try:
                    r = await asyncio.wait_for(
                        self.agent.a_run(query=prompt, session_id=sid),
                        timeout=self.probe_timeout,
                    )
                except asyncio.TimeoutError:
                    r = f"Error: timeout {self.probe_timeout}s"

                agent_ctx = _extract_agent_context(self.agent)
                cost_info = {
                    "total_cost": float(getattr(self.agent, "total_cost_accumulated", 0) or 0) - float(start_cost),
                    "tokens_in": int(getattr(self.agent, "total_tokens_in", 0) or 0) - int(start_tin),
                    "tokens_out": int(getattr(self.agent, "total_tokens_out", 0) or 0) - int(start_tout),
                    "execution_time_s": time.perf_counter() - t0,
                    "tool_calls": agent_ctx["tool_calls"],
                    "working_history": agent_ctx["working_history"],
                }
                try:
                    self.agent.clear_session_history(sid)
                except Exception:
                    pass
                return (r if isinstance(r, str) else str(r)), cost_info
            except Exception as e:
                return f"Error: {type(e).__name__}: {e}", {
                    "total_cost": 0, "tokens_in": 0, "tokens_out": 0,
                    "execution_time_s": time.perf_counter() - t0,
                }

        return await run_tasks(
            tasks, model_fn,
            model_id=model_id,
            suite_id=self.suite_path.stem if self.suite_path else "all",
            mode=mode,
            timeout=self.probe_timeout,
                    on_progress=self.on_progress,
        )


class AgentStreamAdapter(_BaseAdapter):
    """Adapter for FlowAgent streaming — uses agent.a_stream()."""

    def __init__(
        self,
        agent,
        zen_callback=None,
        task_dir: str | Path = "",
        suite_path: str | Path = "",
        model_modalities: list[str] | None = None,
        probe_timeout: float = 120.0,
        on_progress: Callable | None = None,
    ):
        super().__init__(task_dir, suite_path, model_modalities, on_progress)
        self.agent = agent
        self.zen_callback = zen_callback
        self.probe_timeout = probe_timeout
        self._counter = 0

    async def benchmark(
        self,
        model_id: str = "agent_stream",
        mode: str = "standard",
        seed: int | None = None,
    ) -> Report:
        tasks = self._load_tasks()

        async def model_fn(prompt: str) -> tuple[str, dict]:
            self._counter += 1
            sid = f"bench_st_{self._counter}"
            t0 = time.perf_counter()
            try:
                start_cost = getattr(self.agent, "total_cost_accumulated", 0) or 0
                start_tin = getattr(self.agent, "total_tokens_in", 0) or 0
                start_tout = getattr(self.agent, "total_tokens_out", 0) or 0
                r = ""

                async def _consume():
                    nonlocal r
                    async for chunk in self.agent.a_stream(
                        query=prompt, wait_for_hard=True, session_id=sid
                    ):
                        if self.zen_callback:
                            try:
                                self.zen_callback(chunk)
                            except Exception:
                                pass
                        if chunk.get("type") in ("done", "final_answer"):
                            r = chunk.get("final_answer", chunk.get("answer", ""))

                try:
                    await asyncio.wait_for(_consume(), timeout=self.probe_timeout)
                except asyncio.TimeoutError:
                    if not r:
                        r = f"Error: timeout {self.probe_timeout}s"
                except (GeneratorExit, StopAsyncIteration):
                    pass

                agent_ctx = _extract_agent_context(self.agent)
                cost_info = {
                    "total_cost": float(getattr(self.agent, "total_cost_accumulated", 0) or 0) - float(start_cost),
                    "tokens_in": int(getattr(self.agent, "total_tokens_in", 0) or 0) - int(start_tin),
                    "tokens_out": int(getattr(self.agent, "total_tokens_out", 0) or 0) - int(start_tout),
                    "execution_time_s": time.perf_counter() - t0,
                    "tool_calls": agent_ctx["tool_calls"],
                    "working_history": agent_ctx["working_history"],
                }
                try:
                    self.agent.clear_session_history(sid)
                except Exception:
                    pass
                return r or "", cost_info
            except Exception as e:
                return f"Error: {type(e).__name__}: {e}", {
                    "total_cost": 0, "tokens_in": 0, "tokens_out": 0,
                    "execution_time_s": time.perf_counter() - t0,
                }

        return await run_tasks(
            tasks, model_fn,
            model_id=model_id,
            suite_id=self.suite_path.stem if self.suite_path else "all",
            mode=mode,
            timeout=self.probe_timeout,
                    on_progress=self.on_progress,
        )


class MAKERAdapter(_BaseAdapter):
    """Adapter for FlowAgent accomplish."""

    def __init__(
        self,
        agent,
        task_dir: str | Path = "",
        suite_path: str | Path = "",
        model_modalities: list[str] | None = None,
        on_progress: Callable | None = None,
    ):
        super().__init__(task_dir, suite_path, model_modalities, on_progress)
        self.agent = agent

    async def benchmark(
        self,
        model_id: str = "maker",
        mode: str = "standard",
        seed: int | None = None,
    ) -> Report:
        tasks = self._load_tasks()

        async def model_fn(prompt: str) -> tuple[str, dict]:
            try:
                r = await self.agent.a_accomplish(task=prompt, min_complexity=3, max_parallel=3)
                cost_info = r.get("cost_info", {}) if isinstance(r, dict) else {}
                result = r.get("result", str(r)) if isinstance(r, dict) else str(r)
                return result, cost_info
            except Exception as e:
                return f"Error: {type(e).__name__}: {e}", {}

        return await run_tasks(
            tasks, model_fn,
            model_id=model_id,
            suite_id=self.suite_path.stem if self.suite_path else "all",
            mode=mode,
                    on_progress=self.on_progress,
        )
