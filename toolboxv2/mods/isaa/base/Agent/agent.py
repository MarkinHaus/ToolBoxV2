import asyncio
import json
import logging
import os
import pickle
import random
import re
import threading
import time
import types
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timedelta
from functools import wraps
from typing import Any

import yaml

from toolboxv2.mods.isaa.base.tbpocketflow import AsyncFlow, AsyncNode

from pydantic import BaseModel, ValidationError

from toolboxv2.mods.isaa.base.Agent.chain import CF, IS, Chain, ConditionalChain

from toolboxv2.utils.extras.Style import Spinner, print_prompt

# Framework imports with graceful degradation
try:
    import litellm
    from litellm import BudgetManager, Usage
    from litellm.utils import get_max_tokens
    LITELLM_AVAILABLE = True
    # prin litllm version
    print(f"INFO: LiteLLM version {litellm.api_version} found.")
except ImportError:
    LITELLM_AVAILABLE = False
    class BudgetManager: pass
    def get_max_tokens(*a, **kw): return 4096

try:
    from python_a2a import A2AClient, A2AServer, AgentCard
    from python_a2a import run_server as run_a2a_server_func
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False
    class A2AServer: pass
    class A2AClient: pass
    class AgentCard: pass

try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    class FastMCP: pass

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    class TracerProvider: pass

from toolboxv2 import get_logger
from toolboxv2.mods.isaa.base.Agent.types import *

logger = get_logger()
litllm_logger = logging.getLogger("LiteLLM")
litllm_logger.setLevel(logging.CRITICAL)
git_logger = logging.getLogger("git")
git_logger.setLevel(logging.CRITICAL) #(get_logger().level)
mcp_logger = logging.getLogger("mcp")
mcp_logger.setLevel(logging.CRITICAL)
urllib3_logger = logging.getLogger("urllib3")
urllib3_logger.setLevel(logging.CRITICAL)
chardet_logger = logging.getLogger("chardet")
chardet_logger.setLevel(logging.CRITICAL)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.CRITICAL)
asyncio_logger = logging.getLogger("asyncio")
asyncio_logger.setLevel(logging.CRITICAL)

AGENT_VERBOSE = os.environ.get("AGENT_VERBOSE", "false").lower() == "true"
rprint = print if AGENT_VERBOSE else lambda *a, **k: None
wprint = print if AGENT_VERBOSE else lambda *a, **k: None
eprint = print if AGENT_VERBOSE else lambda *a, **k: None
iprint = print if AGENT_VERBOSE else lambda *a, **k: None

TASK_TYPES = ["llm_call", "tool_call", "analysis", "generic"]


import functools

# Annahme: Die folgenden Klassen sind bereits definiert
# from your_project import AsyncNode, ProgressEvent, NodeStatus

# --- Dies ist der wiederverwendbare "Autohook"-Dekorator ---
def with_progress_tracking(cls):
    """
    Ein Klassendekorator, der die Methoden run_async, prep_async, exec_async,
    und exec_fallback_async automatisch mit umfassendem Progress-Tracking umwickelt.
    """

    # --- Wrapper für run_async ---
    original_run = getattr(cls, 'run_async', None)
    if original_run:
        @functools.wraps(original_run)
        async def wrapped_run_async(self, shared):
            progress_tracker = shared.get("progress_tracker")
            node_name = self.__class__.__name__

            if not progress_tracker:
                return await original_run(self, shared)

            timer_key = f"{node_name}_total"
            progress_tracker.start_timer(timer_key)
            await progress_tracker.emit_event(ProgressEvent(
                event_type="node_enter",
                timestamp=time.time(),
                node_name=node_name,
                session_id=shared.get("session_id"),
                task_id=shared.get("current_task_id"),
                plan_id=shared.get("current_plan", TaskPlan(id="none", name="none", description="none")).id if shared.get("current_plan") else None,
                status=NodeStatus.RUNNING,
                success=None
            ))

            try:
                # Hier wird die ursprüngliche Methode aufgerufen
                result = await original_run(self, shared)

                total_duration = progress_tracker.end_timer(timer_key)
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="node_exit",
                    timestamp=time.time(),
                    node_name=node_name,
                    status=NodeStatus.COMPLETED,
                    success=True,
                    node_duration=total_duration,
                    routing_decision=result,
                    session_id=shared.get("session_id"),
                    task_id=shared.get("current_task_id"),
                    metadata={"success": True}
                ))

                return result
            except Exception as e:
                total_duration = progress_tracker.end_timer(timer_key)
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="error",
                    timestamp=time.time(),
                    node_name=node_name,
                    status=NodeStatus.FAILED,
                    success=False,
                    node_duration=total_duration,
                    session_id=shared.get("session_id"),
                    metadata={"error": str(e), "error_type": type(e).__name__}
                ))
                raise

        cls.run_async = wrapped_run_async

    # --- Wrapper für prep_async ---
    original_prep = getattr(cls, 'prep_async', None)
    if original_prep:
        @functools.wraps(original_prep)
        async def wrapped_prep_async(self, shared):
            progress_tracker = shared.get("progress_tracker")
            node_name = self.__class__.__name__

            if not progress_tracker:
                return await original_prep(self, shared)
            timer_key = f"{node_name}_total_p"
            progress_tracker.start_timer(timer_key)
            timer_key = f"{node_name}_prep"
            progress_tracker.start_timer(timer_key)
            await progress_tracker.emit_event(ProgressEvent(
                event_type="node_phase",
                timestamp=time.time(),
                node_name=node_name,
                status=NodeStatus.STARTING,
                node_phase="prep",
                session_id=shared.get("session_id")
            ))

            try:
                result = await original_prep(self, shared)

                prep_duration = progress_tracker.end_timer(timer_key)
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="node_phase",
                    timestamp=time.time(),
                    status=NodeStatus.RUNNING,
                    success=True,
                    node_name=node_name,
                    node_phase="prep_complete",
                    node_duration=prep_duration,
                    session_id=shared.get("session_id")
                ))
                return result
            except Exception as e:
                progress_tracker.end_timer(timer_key)
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="error",
                    timestamp=time.time(),
                    node_name=node_name,
                    status=NodeStatus.FAILED,
                    success=False,
                    metadata={"error": str(e), "error_type": type(e).__name__},
                    node_phase="prep_failed"
                ))
                raise


        cls.prep_async = wrapped_prep_async

    # --- Wrapper für exec_async ---
    original_exec = getattr(cls, 'exec_async', None)
    if original_exec:
        @functools.wraps(original_exec)
        async def wrapped_exec_async(self, prep_res):
            progress_tracker = prep_res.get("progress_tracker") if isinstance(prep_res, dict) else None
            node_name = self.__class__.__name__

            if not progress_tracker:
                return await original_exec(self, prep_res)

            timer_key = f"{node_name}_exec"
            progress_tracker.start_timer(timer_key)
            await progress_tracker.emit_event(ProgressEvent(
                event_type="node_phase",
                timestamp=time.time(),
                node_name=node_name,
                status=NodeStatus.RUNNING,
                node_phase="exec",
                session_id=prep_res.get("session_id") if isinstance(prep_res, dict) else None
            ))

            # In exec gibt es normalerweise keine Fehlerbehandlung, da diese von run_async übernommen wird
            result = await original_exec(self, prep_res)

            exec_duration = progress_tracker.end_timer(timer_key)
            await progress_tracker.emit_event(ProgressEvent(
                event_type="node_phase",
                timestamp=time.time(),
                node_name=node_name,
                status=NodeStatus.RUNNING,
                success=True,
                node_phase="exec_complete",
                node_duration=exec_duration,
                session_id=prep_res.get("session_id") if isinstance(prep_res, dict) else None
            ))
            return result

        cls.exec_async = wrapped_exec_async

    # --- Wrapper für post_async ---
    original_post = getattr(cls, 'post_async', None)
    if original_post:
        @functools.wraps(original_post)
        async def wrapped_post_async(self, shared, prep_res, exec_res):
            if isinstance(exec_res, str):
                print("exec_res is string:", exec_res)
            progress_tracker = shared.get("progress_tracker")
            node_name = self.__class__.__name__

            if not progress_tracker:
                return await original_post(self, shared, prep_res, exec_res)

            timer_key_post = f"{node_name}_post"
            progress_tracker.start_timer(timer_key_post)
            await progress_tracker.emit_event(ProgressEvent(
                event_type="node_phase",
                timestamp=time.time(),
                node_name=node_name,
                status=NodeStatus.COMPLETING,  # Neue Phase "completing"
                node_phase="post",
                session_id=shared.get("session_id")
            ))

            try:
                # Die eigentliche post_async Methode aufrufen
                result = await original_post(self, shared, prep_res, exec_res)

                post_duration = progress_tracker.end_timer(timer_key_post)
                total_duration = progress_tracker.end_timer(f"{node_name}_total_p")  # Gesamtdauer stoppen

                # Sende das entscheidende "node_exit" Event nach erfolgreicher post-Phase
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="node_exit",
                    timestamp=time.time(),
                    node_name=node_name,
                    status=NodeStatus.COMPLETED,
                    success=True,
                    node_duration=total_duration,
                    routing_decision=result,
                    session_id=shared.get("session_id"),
                    task_id=shared.get("current_task_id"),
                    metadata={
                        "success": True,
                        "post_duration": post_duration
                    }
                ))

                return result
            except Exception as e:
                # Fehler in der post-Phase

                post_duration = progress_tracker.end_timer(timer_key_post)
                total_duration = progress_tracker.end_timer(f"{node_name}_total")
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="error",
                    timestamp=time.time(),
                    node_name=node_name,
                    status=NodeStatus.FAILED,
                    success=False,
                    node_duration=total_duration,
                    metadata={"error": str(e), "error_type": type(e).__name__, "phase": "post"},
                    node_phase="post_failed"
                ))
                raise

        cls.post_async = wrapped_post_async

    # --- Wrapper für exec_fallback_async ---
    original_fallback = getattr(cls, 'exec_fallback_async', None)
    if original_fallback:
        @functools.wraps(original_fallback)
        async def wrapped_fallback_async(self, prep_res, exc):
            progress_tracker = prep_res.get("progress_tracker") if isinstance(prep_res, dict) else None
            node_name = self.__class__.__name__

            if progress_tracker:
                timer_key = f"{node_name}_exec"
                exec_duration = progress_tracker.end_timer(timer_key)
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="node_phase",
                    timestamp=time.time(),
                    node_name=node_name,
                    node_phase="exec_fallback",
                    node_duration=exec_duration,
                    status=NodeStatus.FAILED,
                    success=False,
                    session_id=prep_res.get("session_id") if isinstance(prep_res, dict) else None,
                    metadata={"error": str(exc), "error_type": type(exc).__name__},
                ))

            return await original_fallback(self, prep_res, exc)

        cls.exec_fallback_async = wrapped_fallback_async

    return cls

# ===== CORE NODE IMPLEMENTATIONS =====

@with_progress_tracking
class TaskPlannerNode(AsyncNode):
    """Erweiterte Aufgabenplanung mit dynamischen Referenzen und Tool-Integration"""

    async def prep_async(self, shared):
        """Enhanced preparation with goals-based planning support"""

        # Check if this is a goals-based call from LLMReasonerNode
        replan_context = shared.get("replan_context", {})
        goals_list = replan_context.get("goals", [])

        if goals_list:
            # Goals-based planning (called by LLMReasonerNode)
            return {
                "goals": goals_list,
                "planning_mode": "goals_based",
                "query": shared.get("current_query", ""),
                "reasoning_context": replan_context.get("reasoning_context", ""),
                "triggered_by": replan_context.get("triggered_by", "unknown"),
                "tasks": shared.get("tasks", {}),
                "system_status": shared.get("system_status", "idle"),
                "tool_capabilities": shared.get("tool_capabilities", {}),
                "available_tools_names": shared.get("available_tools", []),
                "strategy": "goals_decomposition",  # New strategy type
                "fast_llm_model": shared.get("fast_llm_model"),
                "complex_llm_model": shared.get("complex_llm_model"),
                "agent_instance": shared.get("agent_instance"),
                "variable_manager": shared.get("variable_manager"),
            }
        else:
            # Legacy planning (original query-based approach)
            return {
                "query": shared.get("current_query", ""),
                "tasks": shared.get("tasks", {}),
                "system_status": shared.get("system_status", "idle"),
                "tool_capabilities": shared.get("tool_capabilities", {}),
                "available_tools_names": shared.get("available_tools", []),
                "strategy": shared.get("selected_strategy", "direct_response"),
                "fast_llm_model": shared.get("fast_llm_model"),
                "complex_llm_model": shared.get("complex_llm_model"),
                "agent_instance": shared.get("agent_instance"),
                "variable_manager": shared.get("variable_manager"),
                "planning_mode": "legacy"
            }

    async def exec_async(self, prep_res):
        if prep_res["strategy"] == "fast_simple_planning":
            return await self._create_simple_plan(prep_res)
        else:
            return await self._advanced_llm_decomposition(prep_res)

    async def post_async(self, shared, prep_res, exec_res):
        """Post-processing nach Plan-Erstellung"""

        if exec_res is None:
            shared["planning_error"] = "Plan creation returned None"
            return "planning_failed"

        if isinstance(exec_res, TaskPlan):

            progress_tracker = shared.get("progress_tracker")
            if progress_tracker:
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="plan_created",
                    node_name="TaskPlannerNode",
                    session_id=shared.get("session_id"),
                    status=NodeStatus.COMPLETED,
                    success=True,
                    plan_id=exec_res.id,
                    metadata={
                        "plan_name": exec_res.name,
                        "task_count": len(exec_res.tasks),
                        "strategy": exec_res.execution_strategy
                    }
                ))
                await asyncio.sleep(0.1)

            # Erfolgreicher Plan
            shared["current_plan"] = exec_res

            # Tasks in shared state für Executor verfügbar machen
            task_dict = {task.id: task for task in exec_res.tasks}
            if "tasks" not in shared:
                shared["tasks"] = task_dict
            else:
                shared["tasks"].update(task_dict)

            # Plan-Metadaten setzen
            shared["plan_created_at"] = datetime.now().isoformat()
            shared["plan_strategy"] = exec_res.execution_strategy
            shared["total_tasks_planned"] = len(exec_res.tasks)

            rprint(f"Plan created successfully: {exec_res.name} with {len(exec_res.tasks)} tasks")
            return "planned"

        else:
            # Plan creation failed
            shared["planning_error"] = "Invalid plan format returned"
            shared["current_plan"] = None
            eprint("Plan creation failed - invalid format")
            return "planning_failed"

    async def _create_simple_plan(self, prep_res) -> TaskPlan:
        """Fast lightweight planning for direct or simple multi-step queries."""
        taw = self._build_tool_intelligence(prep_res)
        rprint("You are a FAST "+ taw)
        prompt = f"""
You are a FAST abstract pattern recognizer and task planner.
Identify if the query needs a **single-step LLM answer** or a **simple 2–3 task plan** using available tools.
Output ONLY YAML.

## User Query
{prep_res['query']}

## Available Tools
{taw}

## Pattern Recognition (Internal Only)
- Detect if query is informational, action-based, or tool-eligible.
- Map to minimal plan type: "direct_llm" or "simple_tool_plus_llm".

## YAML Schema
```yaml
plan_name: string
description: string
execution_strategy: "sequential" | "parallel"
tasks:
  - id: string
    type: "LLMTask" | "ToolTask"
    description: string
    priority: int
    dependencies: [list]
Example 1 — Direct LLM
```yaml
plan_name: Direct Response
description: Quick answer from LLM
execution_strategy: sequential
tasks:
  - id: answer
    type: LLMTask
    description: Respond to query
    priority: 1
    dependencies: []
    prompt_template: Respond concisely to: {prep_res['query']}
    llm_config:
      model_preference: fast
      temperature: 0.3
```
Example 2 — Tool + LLM
```yaml
plan_name: Fetch and Answer
description: Get info from tool and summarize
execution_strategy: sequential
tasks:
  - id: fetch_info
    type: ToolTask
    description: Get required data
    priority: 1
    dependencies: []
    tool_name: info_api
    arguments:
      query: "{prep_res['query']}"
  - id: summarize
    type: LLMTask
    description: Summarize fetched data
    priority: 2
    dependencies: ["fetch_info"]
    prompt_template: Summarize: {{ results.fetch_info.data }}
    llm_config:
      model_preference: fast
      temperature: 0.3
```
Output Requirements
Use ONLY YAML for the final output
Pick minimal plan type for fastest completion!
focus on correct quotation and correct yaml format!
    """

        try:
            agent_instance = prep_res["agent_instance"]
            content = await agent_instance.a_run_llm_completion(
                model=prep_res.get("complex_llm_model", "openrouter/anthropic/claude-3-haiku"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4512,
                node_name="TaskPlannerNode", task_id="fast_simple_planning"
            )

            yaml_content = content.split("```yaml")[1].split("```")[0].strip() if "```yaml" in content else content
            plan_data = yaml.safe_load(yaml_content)
            # print("Simple", json.dumps(plan_data, indent=2))
            return TaskPlan(
                id=str(uuid.uuid4()),
                name=plan_data.get("plan_name", "Generated Plan"),
                description=plan_data.get("description", f"Plan for: {prep_res['query']}"),
                tasks=[
                    [LLMTask, ToolTask, DecisionTask, Task][["LLMTask", "ToolTask", "DecisionTask", "Task"].index(t.get("type"))](**t)
                    for t in plan_data.get("tasks", [])
                ],
                execution_strategy=plan_data.get("execution_strategy", "sequential")
            )

        except Exception as e:
            eprint(f"Simple plan creation failed: {e}")
            import traceback
            print(traceback.format_exc())
            return TaskPlan(
                id=str(uuid.uuid4()),
                name="Fallback Plan",
                description="Direct response only",
                tasks=[
                    LLMTask(
                        id="fast_simple_planning",
                        type="LLMTask",
                        description="Generate direct response",
                        priority=1,
                        dependencies=[],
                        prompt_template=f"Respond to the query: {prep_res['query']}",
                        llm_config={"model_preference": "fast"}
                    )
                ]
            )

    async def _advanced_llm_decomposition(self, prep_res) -> TaskPlan:
        """Enhanced LLM-based decomposition with goals-based planning support"""

        planning_mode = prep_res.get("planning_mode", "legacy")
        variable_manager = prep_res.get("variable_manager")
        tool_intelligence = self._build_tool_intelligence(prep_res)

        if planning_mode == "goals_based":
            # Goals-based planning from LLMReasonerNode
            goals_list = prep_res.get("goals", [])
            reasoning_context = prep_res.get("reasoning_context", "")

            prompt = f"""
You are an expert task planner specialized in creating execution plans from strategic goals.
Create a comprehensive plan that addresses all goals with proper dependencies and parallelization.

## Strategic Goals from Reasoner
{chr(10).join([f"{i + 1}. {goal}" for i, goal in enumerate(goals_list)])}

## Reasoning Context
{reasoning_context}

## Your Available Tools & Intelligence
{tool_intelligence}

{variable_manager.get_llm_variable_context() if variable_manager else ""}

## Goals-Based Planning Instructions
1. Analyze each goal for dependencies on other goals
2. Identify goals that can be executed in parallel
3. Create tasks that address each goal effectively
4. Use variable references {{ results.task_id.data }} for dependencies
5. Ensure proper sequencing and coordination

## YAML Schema
```yaml
plan_name: string
description: string
execution_strategy: "sequential" | "parallel" | "mixed"
tasks:
  - id: string
    type: "LLMTask" | "ToolTask" | "DecisionTask"
    description: string
    priority: int
    dependencies: [list of task ids]
    # Type-specific fields as needed
Goals Decomposition Strategy

Independent Goals: Create parallel tasks
Sequential Goals: Use dependencies array
Complex Goals: Break into sub-tasks with DecisionTask routing
Data Dependencies: Use variable references between tasks

Example for Multi-Goal Plan
yamlCopyplan_name: "Multi-Goal Strategic Plan"
description: "Execute multiple strategic objectives with proper coordination"
execution_strategy: "mixed"
tasks:
  - id: "goal_1_research"
    type: "ToolTask"
    description: "Research data for Goal 1"
    priority: 1
    dependencies: []
    tool_name: "search_web"
    arguments:
      query: "research topic for goal 1"

  - id: "goal_2_research"
    type: "ToolTask"
    description: "Research data for Goal 2"
    priority: 1
    dependencies: []
    tool_name: "search_web"
    arguments:
      query: "research topic for goal 2"

  - id: "analyze_combined"
    type: "LLMTask"
    description: "Analyze combined research results"
    priority: 2
    dependencies: ["goal_1_research", "goal_2_research"]
    prompt_template: |
      Analyze these research results:
      Goal 1 Data: {{ results.goal_1_research.data }}
      Goal 2 Data: {{ results.goal_2_research.data }}

      Provide comprehensive analysis addressing both goals.
    llm_config:
      model_preference: "complex"
      temperature: 0.3
Generate the execution plan for the strategic goals:
    """

        else:
            # Legacy single-query planning
            base_query = prep_res['query']
            prompt = f"""
You are an expert task planner with dynamic adaptation capabilities.
Create intelligent, adaptive execution plans for the user query.
User Query
{base_query}
Your Available Tools & Intelligence
{tool_intelligence}
{variable_manager.get_llm_variable_context() if variable_manager else ""}
TASK TYPES (Dataclass-Aligned)

LLMTask: Step that uses a language model
ToolTask: Step that calls an available tool
DecisionTask: Step that decides routing between tasks

YAML SCHEMA
yamlCopyplan_name: string
description: string
execution_strategy: "sequential" | "parallel" | "mixed"
tasks:
  - id: string
    type: "LLMTask" | "ToolTask" | "DecisionTask"
    description: string
    priority: int
    dependencies: [list of task ids]
    # Additional fields depending on type
Generate the adaptive execution plan:
            """

        try:
            model_to_use = prep_res.get("complex_llm_model", "openrouter/openai/gpt-4o")
            agent_instance = prep_res["agent_instance"]

            content = await agent_instance.a_run_llm_completion(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                #max_tokens=2048,
                node_name="TaskPlannerNode",
                task_id="goals_based_planning" if planning_mode == "goals_based" else "adaptive_planning"
            )

            if "```yaml" in content:
                yaml_content = content.split("```yaml")[1].split("```")[0].strip()
            else:
                yaml_content = content

            plan_data = yaml.safe_load(yaml_content)

            # Create specialized tasks
            tasks = []
            for task_data in plan_data.get("tasks", []):
                task_type = task_data.pop("type", "generic")
                task = create_task(task_type, **task_data)
                tasks.append(task)

            plan = TaskPlan(
                id=str(uuid.uuid4()),
                name=plan_data.get("plan_name", "Generated Plan"),
                description=plan_data.get("description",
                                          "Plan for goals-based execution" if planning_mode == "goals_based" else f"Plan for: {base_query}"),
                tasks=tasks,
                execution_strategy=plan_data.get("execution_strategy", "sequential"),
                metadata={
                    "planning_mode": planning_mode,
                    "goals_count": len(prep_res.get("goals", [])) if planning_mode == "goals_based" else 1
                }
            )

            rprint(f"Created {planning_mode} plan with {len(tasks)} tasks")
            return plan

        except Exception as e:
            eprint(f"Advanced planning failed: {e}")
            import traceback

            print(traceback.format_exc())
            return await self._create_simple_plan(prep_res)

    def _build_tool_intelligence(self, prep_res: dict) -> str:
        """Build detailed tool intelligence for planning"""

        agent_instance = prep_res.get("agent_instance")
        if not agent_instance or not hasattr(agent_instance, '_tool_capabilities'):
            return "No tool intelligence available."

        capabilities = agent_instance._tool_capabilities
        query = prep_res.get('query', '').lower()

        context_parts = []
        context_parts.append("### Intelligent Tool Analysis:")

        for tool_name, cap in capabilities.items():
            context_parts.append(f"\n{tool_name}:")
            context_parts.append(f"- Function: {cap.get('primary_function', 'Unknown')}")
            context_parts.append(f"- Arguments: {yaml.dump(cap.get('args_schema', 'takes no arguments!'), default_flow_style=False)}")

            # Check relevance to current query
            relevance_score = self._calculate_tool_relevance(query, cap)
            context_parts.append(f"- Query relevance: {relevance_score:.2f}")

            if relevance_score > 0.4:
                context_parts.append("- ⭐ HIGHLY RELEVANT - SHOULD USE THIS TOOL!")

            # Show trigger analysis
            triggers = cap.get('trigger_phrases', [])
            matched_triggers = [t for t in triggers if t.lower() in query]
            if matched_triggers:
                context_parts.append(f"- Matched triggers: {matched_triggers}")

            # Show use cases
            use_cases = cap.get('use_cases', [])[:3]
            context_parts.append(f"- Use cases: {', '.join(use_cases)}")

        return "\n".join(context_parts)

    def _calculate_tool_relevance(self, query: str, capabilities: dict) -> float:
        """Calculate how relevant a tool is to the current query"""

        query_words = set(query.lower().split())

        # Check trigger phrases
        trigger_score = 0.0
        triggers = capabilities.get('trigger_phrases', [])
        for trigger in triggers:
            trigger_words = set(trigger.lower().split())
            if trigger_words.intersection(query_words):
                trigger_score += 0.04
        # Check confidence triggers if available
        conf_triggers = capabilities.get('confidence_triggers', {})
        for phrase, confidence in conf_triggers.items():
            if phrase.lower() in query:
                trigger_score += confidence/10
        # Check indirect connections
        indirect = capabilities.get('indirect_connections', [])
        for connection in indirect:
            connection_words = set(connection.lower().split())
            if connection_words.intersection(query_words):
                trigger_score += 0.02
        return min(1.0, trigger_score)

@with_progress_tracking
class TaskExecutorNode(AsyncNode):
    """Vollständige Task-Ausführung als unabhängige Node mit LLM-unterstützter Planung"""

    def __init__(self, max_parallel: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_parallel = max_parallel
        self.results_store = {}  # Für {{ }} Referenzen
        self.execution_history = []  # Für LLM-basierte Optimierung
        self.agent_instance = None  # Wird gesetzt vom FlowAgent
        self.variable_manager = None
        self.fast_llm_model = None
        self.complex_llm_model = None
        self.progress_tracker = None

    async def prep_async(self, shared):
        """Enhanced preparation with unified variable system"""
        current_plan = shared.get("current_plan")
        tasks = shared.get("tasks", {})

        # Get unified variable manager
        self.variable_manager = shared.get("variable_manager")
        self.progress_tracker = shared.get("progress_tracker")
        if not self.variable_manager:
            self.variable_manager = VariableManager(shared.get("world_model", {}), shared)

        # Register all necessary scopes
        self.variable_manager.set_results_store(self.results_store)
        self.variable_manager.set_tasks_store(tasks)
        self.variable_manager.register_scope('user', shared.get('user_context', {}))
        self.variable_manager.register_scope('system', {
            'timestamp': datetime.now().isoformat(),
            'agent_name': shared.get('agent_instance', {}).amd.name if shared.get('agent_instance') else 'unknown'
        })

        # Stelle sicher, dass Agent-Referenz verfügbar ist
        if not self.agent_instance:
            self.agent_instance = shared.get("agent_instance")

        if not current_plan:
            return {"error": "No active plan", "tasks": tasks}

        # Rest of existing prep_async logic...
        ready_tasks = self._find_ready_tasks(current_plan, tasks)
        blocked_tasks = self._find_blocked_tasks(current_plan, tasks)

        execution_plan = await self._create_intelligent_execution_plan(
            ready_tasks, blocked_tasks, current_plan, shared
        )
        self.complex_llm_model = shared.get("complex_llm_model")
        self.fast_llm_model = shared.get("fast_llm_model")

        return {
            "plan": current_plan,
            "ready_tasks": ready_tasks,
            "blocked_tasks": blocked_tasks,
            "all_tasks": tasks,
            "execution_plan": execution_plan,
            "fast_llm_model": self.fast_llm_model,
            "complex_llm_model": self.complex_llm_model,
            "available_tools": shared.get("available_tools", []),
            "world_model": shared.get("world_model", {}),
            "results": self.results_store,
            "variable_manager": self.variable_manager,
            "progress_tracker": self.progress_tracker ,
        }

    def _find_ready_tasks(self, plan: TaskPlan, all_tasks: dict[str, Task]) -> list[Task]:
        """Finde Tasks die zur Ausführung bereit sind"""
        ready = []
        for task in plan.tasks:
            if task.status == "pending" and self._dependencies_satisfied(task, all_tasks):
                ready.append(task)
        return ready

    def _find_blocked_tasks(self, plan: TaskPlan, all_tasks: dict[str, Task]) -> list[Task]:
        """Finde blockierte Tasks für Analyse"""
        blocked = []
        for task in plan.tasks:
            if task.status == "pending" and not self._dependencies_satisfied(task, all_tasks):
                blocked.append(task)
        return blocked

    def _dependencies_satisfied(self, task: Task, all_tasks: dict[str, Task]) -> bool:
        """Prüfe ob alle Dependencies erfüllt sind"""
        for dep_id in task.dependencies:
            if dep_id in all_tasks:
                dep_task = all_tasks[dep_id]
                if dep_task.status not in ["completed"]:
                    return False
            else:
                # Dependency existiert nicht - könnte Problem sein
                wprint(f"Task {task.id} has missing dependency: {dep_id}")
                return False
        return True

    async def _create_intelligent_execution_plan(
        self,
        ready_tasks: list[Task],
        blocked_tasks: list[Task],
        plan: TaskPlan,
        shared: dict
    ) -> dict[str, Any]:
        """LLM-unterstützte intelligente Ausführungsplanung"""

        if not ready_tasks:
            return {
                "strategy": "waiting",
                "reason": "No ready tasks",
                "blocked_count": len(blocked_tasks),
                "recommendations": []
            }

        # Einfache Planung für wenige Tasks
        if len(ready_tasks) <= 2 and not LITELLM_AVAILABLE:
            return self._create_simple_execution_plan(ready_tasks, plan)

        # LLM-basierte intelligente Planung
        return await self._llm_execution_planning(ready_tasks, blocked_tasks, plan, shared)

    def _create_simple_execution_plan(self, ready_tasks: list[Task], plan: TaskPlan) -> dict[str, Any]:
        """Einfache heuristische Ausführungsplanung"""

        # Prioritäts-basierte Sortierung
        sorted_tasks = sorted(ready_tasks, key=lambda t: (t.priority, t.created_at))

        # Parallelisierbare Tasks identifizieren
        parallel_groups = []
        current_group = []

        for task in sorted_tasks:
            # ToolTasks können oft parallel laufen
            if isinstance(task, ToolTask) and len(current_group) < self.max_parallel:
                current_group.append(task)
            else:
                if current_group:
                    parallel_groups.append(current_group)
                    current_group = []
                current_group.append(task)

        if current_group:
            parallel_groups.append(current_group)

        strategy = "parallel" if len(parallel_groups) > 1 or len(parallel_groups[0]) > 1 else "sequential"

        return {
            "strategy": strategy,
            "execution_groups": parallel_groups,
            "total_groups": len(parallel_groups),
            "reasoning": "Simple heuristic: priority-based with tool parallelization",
            "estimated_duration": self._estimate_duration(sorted_tasks)
        }

    async def _llm_execution_planning(
        self,
        ready_tasks: list[Task],
        blocked_tasks: list[Task],
        plan: TaskPlan,
        shared: dict
    ) -> dict[str, Any]:
        """Erweiterte LLM-basierte Ausführungsplanung"""

        try:
            # Erstelle detaillierte Task-Analyse für LLM
            task_analysis = self._analyze_tasks_for_llm(ready_tasks, blocked_tasks)
            execution_context = self._build_execution_context(shared)

            prompt = f"""
Du bist ein Experte für Task-Ausführungsplanung. Analysiere die verfügbaren Tasks und erstelle einen optimalen Ausführungsplan.

## Verfügbare Tasks zur Ausführung
{task_analysis['ready_tasks_summary']}

## Blockierte Tasks (zur Information)
{task_analysis['blocked_tasks_summary']}

## Ausführungskontext
- Max parallele Tasks: {self.max_parallel}
- Plan-Strategie: {plan.execution_strategy}
- Verfügbare Tools: {', '.join(shared.get('available_tools', []))}
- Bisherige Ergebnisse: {len(self.results_store)} Tasks abgeschlossen
- Execution History: {len(self.execution_history)} vorherige Zyklen

## Bisherige Performance
{execution_context}

## Aufgabe
Erstelle einen optimierten Ausführungsplan. Berücksichtige:
1. Task-Abhängigkeiten und Prioritäten
2. Parallelisierungsmöglichkeiten
3. Resource-Optimierung (Tools, LLM-Aufrufe)
4. Fehlerwahrscheinlichkeit und Retry-Strategien
5. Dynamische Argument-Auflösung zwischen Tasks

Antworte mit YAML:

```yaml
strategy: "parallel"  # "parallel" | "sequential" | "hybrid"
execution_groups:
  - group_id: 1
    tasks: ["task_1", "task_2"]  # Task IDs
    execution_mode: "parallel"
    priority: "high"
    estimated_duration: 30  # seconds
    risk_level: "low"  # low | medium | high
    dependencies_resolved: true
  - group_id: 2
    tasks: ["task_3"]
    execution_mode: "sequential"
    priority: "medium"
    estimated_duration: 15
    depends_on_groups: [1]
reasoning: "Detailed explanation of the execution strategy"
optimization_suggestions:
  - "Specific optimization 1"
  - "Specific optimization 2"
risk_mitigation:
  - risk: "Tool timeout"
    mitigation: "Use shorter timeout for parallel calls"
  - risk: "Argument resolution failure"
    mitigation: "Validate references before execution"
total_estimated_duration: 45
confidence: 0.85
```"""

            model_to_use = shared.get("complex_llm_model", "openrouter/openai/gpt-4o")

            content = await self.agent_instance.a_run_llm_completion(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000,
                node_name="TaskExecutorNode", task_id="llm_execution_planning"
            )

            yaml_match = re.search(r"```yaml\s*(.*?)\s*```", content, re.DOTALL)
            yaml_str = yaml_match.group(1) if yaml_match else content.strip()

            execution_plan = yaml.safe_load(yaml_str)

            # Validiere und erweitere den Plan
            validated_plan = self._validate_execution_plan(execution_plan, ready_tasks)

            rprint(
                f"LLM execution plan created: {validated_plan.get('strategy')} with {len(validated_plan.get('execution_groups', []))} groups")
            return validated_plan

        except Exception as e:
            eprint(f"LLM execution planning failed: {e}")
            return self._create_simple_execution_plan(ready_tasks, plan)

    def _analyze_tasks_for_llm(self, ready_tasks: list[Task], blocked_tasks: list[Task]) -> dict[str, str]:
        """Analysiere Tasks für LLM-Prompt"""

        ready_summary = []
        for task in ready_tasks:
            task_info = f"- {task.id} ({task.type}): {task.description}"
            if hasattr(task, 'priority'):
                task_info += f" [Priority: {task.priority}]"
            if isinstance(task, ToolTask):
                task_info += f" [Tool: {task.tool_name}]"
                if task.arguments:
                    # Zeige dynamische Referenzen
                    dynamic_refs = [arg for arg in task.arguments.values() if isinstance(arg, str) and "{{" in arg]
                    if dynamic_refs:
                        task_info += f" [Dynamic refs: {len(dynamic_refs)}]"
            ready_summary.append(task_info)

        blocked_summary = []
        for task in blocked_tasks:
            deps = ", ".join(task.dependencies) if task.dependencies else "None"
            blocked_summary.append(f"- {task.id}: waiting for [{deps}]")

        return {
            "ready_tasks_summary": "\n".join(ready_summary) or "No ready tasks",
            "blocked_tasks_summary": "\n".join(blocked_summary) or "No blocked tasks"
        }

    def _build_execution_context(self, shared: dict) -> str:
        """Baue Kontext für LLM-Planung"""
        context_parts = []

        # Performance der letzten Executions
        if self.execution_history:
            recent = self.execution_history[-3:]  # Last 3 executions
            avg_duration = sum(h.get("duration", 0) for h in recent) / len(recent)
            success_rate = sum(1 for h in recent if h.get("success", False)) / len(recent)
            context_parts.append(f"Recent performance: {avg_duration:.1f}s avg, {success_rate:.1%} success rate")

        # Resource utilization
        if self.results_store:
            tool_usage = {}
            for task_result in self.results_store.values():
                metadata = task_result.get("metadata", {})
                task_type = metadata.get("task_type", "unknown")
                tool_usage[task_type] = tool_usage.get(task_type, 0) + 1
            context_parts.append(f"Resource usage: {tool_usage}")

        return "\n".join(context_parts) if context_parts else "No previous execution history"

    def _validate_execution_plan(self, plan: dict, ready_tasks: list[Task]) -> dict:
        """Validiere und korrigiere LLM-generierten Ausführungsplan"""

        # Standard-Werte setzen
        validated = {
            "strategy": plan.get("strategy", "sequential"),
            "execution_groups": [],
            "reasoning": plan.get("reasoning", "LLM-generated plan"),
            "total_estimated_duration": plan.get("total_estimated_duration", 60),
            "confidence": min(1.0, max(0.0, plan.get("confidence", 0.5)))
        }

        # Validiere execution groups
        task_ids_available = [t.id for t in ready_tasks]

        for group_data in plan.get("execution_groups", []):
            group_tasks = group_data.get("tasks", [])
            # Filtere nur verfügbare Tasks
            valid_tasks = [tid for tid in group_tasks if tid in task_ids_available]

            if valid_tasks:
                validated["execution_groups"].append({
                    "group_id": group_data.get("group_id", len(validated["execution_groups"]) + 1),
                    "tasks": valid_tasks,
                    "execution_mode": group_data.get("execution_mode", "sequential"),
                    "priority": group_data.get("priority", "medium"),
                    "estimated_duration": group_data.get("estimated_duration", 30),
                    "risk_level": group_data.get("risk_level", "medium")
                })

        # Falls keine validen Groups, erstelle Fallback
        if not validated["execution_groups"]:
            validated["execution_groups"] = [{
                "group_id": 1,
                "tasks": task_ids_available[:self.max_parallel],
                "execution_mode": "parallel",
                "priority": "high"
            }]

        return validated

    def _estimate_duration(self, tasks: list[Task]) -> int:
        """Schätze Ausführungsdauer in Sekunden"""
        duration = 0
        for task in tasks:
            if isinstance(task, ToolTask):
                duration += 10  # Tool calls meist schneller
            elif isinstance(task, LLMTask):
                duration += 20  # LLM calls brauchen länger
            else:
                duration += 15  # Standard
        return duration

    async def exec_async(self, prep_res):
        """Hauptausführungslogik mit intelligentem Routing"""

        if "error" in prep_res:
            return {"error": prep_res["error"]}

        execution_plan = prep_res["execution_plan"]

        if execution_plan["strategy"] == "waiting":
            return {
                "status": "waiting",
                "message": execution_plan["reason"],
                "blocked_count": execution_plan.get("blocked_count", 0)
            }

        # Starte Ausführung basierend auf Plan
        execution_start = datetime.now()

        try:
            if execution_plan["strategy"] == "parallel":
                results = await self._execute_parallel_plan(execution_plan, prep_res)
            elif execution_plan["strategy"] == "sequential":
                results = await self._execute_sequential_plan(execution_plan, prep_res)
            else:  # hybrid
                results = await self._execute_hybrid_plan(execution_plan, prep_res)

            execution_duration = (datetime.now() - execution_start).total_seconds()

            # Speichere Execution-History für LLM-Optimierung
            self.execution_history.append({
                "timestamp": execution_start.isoformat(),
                "strategy": execution_plan["strategy"],
                "duration": execution_duration,
                "tasks_executed": len(results),
                "success": all(r.get("status") == "completed" for r in results),
                "plan_confidence": execution_plan.get("confidence", 0.5)
            })

            # Behalte nur letzte 10 Executions
            if len(self.execution_history) > 10:
                self.execution_history = self.execution_history[-10:]

            return {
                "status": "executed",
                "results": results,
                "execution_duration": execution_duration,
                "strategy_used": execution_plan["strategy"],
                "completed_tasks": len([r for r in results if r.get("status") == "completed"]),
                "failed_tasks": len([r for r in results if r.get("status") == "failed"])
            }

        except Exception as e:
            eprint(f"Execution plan failed: {e}")
            return {
                "status": "execution_failed",
                "error": str(e),
                "results": []
            }

    async def _execute_parallel_plan(self, plan: dict, prep_res: dict) -> list[dict]:
        """Führe Plan mit parallelen Gruppen aus"""
        all_results = []

        for group in plan["execution_groups"]:
            group_tasks = self._get_tasks_by_ids(group["tasks"], prep_res)

            if group.get("execution_mode") == "parallel":
                # Parallele Ausführung innerhalb der Gruppe
                batch_results = await self._execute_parallel_batch(group_tasks)
            else:
                # Sequenzielle Ausführung innerhalb der Gruppe
                batch_results = await self._execute_sequential_batch(group_tasks)

            all_results.extend(batch_results)

            # Prüfe ob kritische Tasks fehlgeschlagen sind
            critical_failures = [
                r for r in batch_results
                if r.get("status") == "failed" and self._is_critical_task(r.get("task_id"), prep_res)
            ]

            if critical_failures:
                eprint(f"Critical task failures in group {group['group_id']}, stopping execution")
                break

        return all_results

    async def _execute_sequential_plan(self, plan: dict, prep_res: dict) -> list[dict]:
        """Führe Plan sequenziell aus"""
        all_results = []

        for group in plan["execution_groups"]:
            group_tasks = self._get_tasks_by_ids(group["tasks"], prep_res)
            batch_results = await self._execute_sequential_batch(group_tasks)
            all_results.extend(batch_results)

            # Stoppe bei kritischen Fehlern
            critical_failures = [
                r for r in batch_results
                if r.get("status") == "failed" and self._is_critical_task(r.get("task_id"), prep_res)
            ]

            if critical_failures:
                break

        return all_results

    async def _execute_hybrid_plan(self, plan: dict, prep_res: dict) -> list[dict]:
        """Hybride Ausführung - Groups parallel, innerhalb je nach Mode"""

        # Führe Gruppen parallel aus (wenn möglich)
        group_tasks_list = []
        for group in plan["execution_groups"]:
            group_tasks = self._get_tasks_by_ids(group["tasks"], prep_res)
            group_tasks_list.append((group, group_tasks))

        # Führe bis zu max_parallel Gruppen parallel aus
        batch_size = min(len(group_tasks_list), self.max_parallel)
        all_results = []

        for i in range(0, len(group_tasks_list), batch_size):
            batch = group_tasks_list[i:i + batch_size]

            # Erstelle Coroutines für jede Gruppe
            group_coroutines = []
            for group, tasks in batch:
                if group.get("execution_mode") == "parallel":
                    coro = self._execute_parallel_batch(tasks)
                else:
                    coro = self._execute_sequential_batch(tasks)
                group_coroutines.append(coro)

            # Führe Gruppen-Batch parallel aus
            batch_results = await asyncio.gather(*group_coroutines, return_exceptions=True)

            # Flache Liste der Ergebnisse
            for result_group in batch_results:
                if isinstance(result_group, Exception):
                    eprint(f"Group execution failed: {result_group}")
                    continue
                all_results.extend(result_group)

        return all_results

    def _get_tasks_by_ids(self, task_ids: list[str], prep_res: dict) -> list[Task]:
        """Hole Task-Objekte basierend auf IDs"""
        all_tasks = prep_res["all_tasks"]
        return [all_tasks[tid] for tid in task_ids if tid in all_tasks]

    def _is_critical_task(self, task_id: str, prep_res: dict) -> bool:
        """Prüfe ob Task kritisch ist"""
        task = prep_res["all_tasks"].get(task_id)
        if not task:
            return False
        return getattr(task, 'critical', False) or task.priority == 1

    async def _execute_parallel_batch(self, tasks: list[Task]) -> list[dict]:
        """Führe Tasks parallel aus"""
        if not tasks:
            return []

        # Limitiere auf max_parallel
        batch_size = min(len(tasks), self.max_parallel)
        batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

        all_results = []
        for batch in batches:
            batch_results = await asyncio.gather(
                *[self._execute_single_task(task) for task in batch],
                return_exceptions=True
            )

            # Handle exceptions
            processed_results = []
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    eprint(f"Task {batch[i].id} failed with exception: {result}")
                    processed_results.append({
                        "task_id": batch[i].id,
                        "status": "failed",
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)

            all_results.extend(processed_results)

        return all_results

    async def _execute_sequential_batch(self, tasks: list[Task]) -> list[dict]:
        """Führe Tasks sequenziell aus"""
        results = []

        for task in tasks:
            try:
                result = await self._execute_single_task(task)
                results.append(result)

                # Stoppe bei kritischen Fehlern in sequenzieller Ausführung
                if result.get("status") == "failed" and getattr(task, 'critical', False):
                    eprint(f"Critical task {task.id} failed, stopping sequential execution")
                    break

            except Exception as e:
                eprint(f"Sequential task {task.id} failed: {e}")
                results.append({
                    "task_id": task.id,
                    "status": "failed",
                    "error": str(e)
                })

                if getattr(task, 'critical', False):
                    break

        return results

    async def _execute_single_task(self, task: Task) -> dict:
        """Enhanced task execution with unified LLMToolNode usage"""
        if self.progress_tracker:
            await self.progress_tracker.emit_event(ProgressEvent(
                event_type="task_start",
                node_name="TaskExecutorNode",
                status=NodeStatus.RUNNING,
                task_id=task.id,
                plan_id=self.variable_manager.get("shared.current_plan.id"),
                metadata={
                    "description": task.description,
                    "type": task.type,
                    "priority": task.priority,
                    "dependencies": task.dependencies
                }
            ))

        task_start = time.perf_counter()
        try:
            task.status = "running"
            task.started_at = datetime.now()

            # Ensure metadata is initialized
            if not hasattr(task, 'metadata') or task.metadata is None:
                task.metadata = {}

            # Pre-process task with variable resolution
            if isinstance(task, ToolTask):
                resolved_args = self._resolve_task_variables(task.arguments)
                result = await self._execute_tool_task_with_validation(task, resolved_args)
            elif isinstance(task, LLMTask):
                # Use LLMToolNode for LLM tasks instead of direct execution
                result = await self._execute_llm_via_llmtool(task)
            elif isinstance(task, DecisionTask):
                # Enhanced decision task with context awareness
                result = await self._execute_decision_task_enhanced(task)
            else:
                # Use LLMToolNode for generic tasks as well
                result = await self._execute_generic_via_llmtool(task)

            # Store result in unified system
            self._store_task_result(task.id, result, True)

            task.result = result
            task.status = "completed"
            task.completed_at = datetime.now()

            task_duration = time.perf_counter() - task_start

            if self.progress_tracker:
                await self.progress_tracker.emit_event(ProgressEvent(
                    event_type="task_complete",
                    node_name="TaskExecutorNode",
                    task_id=task.id,
                    plan_id=self.variable_manager.get("shared.current_plan.id"),
                    status=NodeStatus.COMPLETED,
                    success=True,
                    duration=task_duration,
                    metadata={
                        "result_type": type(result).__name__,
                        "description": task.description
                    }
                ))

            return {
                "task_id": task.id,
                "status": "completed",
                "result": result
            }

        except Exception as e:
            task.error = str(e)
            task.status = "failed"
            task.retry_count += 1

            # Store error in unified system
            self._store_task_result(task.id, None, False, str(e))
            task_duration = time.perf_counter() - task_start

            if self.progress_tracker:
                await self.progress_tracker.emit_event(ProgressEvent(
                    event_type="task_error",  # Klarer Event-Typ
                    node_name="TaskExecutorNode",
                    task_id=task.id,
                    plan_id=self.variable_manager.get("shared.current_plan.id"),
                    status=NodeStatus.FAILED,
                    success=False,
                    duration=task_duration,
                    error_details={
                        "message": str(e),
                        "type": type(e).__name__
                    },
                    metadata={
                        "retry_count": task.retry_count,
                        "description": task.description
                    }
                ))

            eprint(f"Task {task.id} failed: {e}")
            return {
                "task_id": task.id,
                "status": "failed",
                "error": str(e),
                "retry_count": task.retry_count
            }

    async def _resolve_dynamic_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Enhanced dynamic argument resolution with full variable system"""
        resolved = {}

        for key, value in arguments.items():
            if isinstance(value, str):
                # FIXED: Use unified variable manager for all resolution
                resolved_value = self.variable_manager.format_text(value)

                # Log if variables weren't resolved (debugging)
                if "{{" in resolved_value and "}}" in resolved_value:
                    wprint(f"Unresolved variables in argument '{key}': {resolved_value}")

                resolved[key] = resolved_value
            else:
                resolved[key] = value

        return resolved

    async def _execute_tool_task_with_validation(self, task: ToolTask, resolved_args: dict[str, Any]) -> Any:
        """Tool execution with improved error detection and validation"""

        if not task.tool_name:
            raise ValueError(f"ToolTask {task.id} missing tool_name")

        agent = self.agent_instance
        if not agent:
            raise ValueError("Agent instance not available for tool execution")

        tool_start = time.perf_counter()

        # Track tool call start
        if self.progress_tracker:
            await self.progress_tracker.emit_event(ProgressEvent(
                event_type="tool_call",
                timestamp=time.time(),
                node_name="TaskExecutorNode",
                status=NodeStatus.RUNNING,
                task_id=task.id,
                tool_name=task.tool_name,
                tool_args=resolved_args,
                metadata={
                    "task_type": "ToolTask",
                    "hypothesis": task.hypothesis,
                    "validation_criteria": task.validation_criteria
                }
            ))

        try:
            rprint(f"Executing tool {task.tool_name} with resolved args: {resolved_args}")

            # Execute tool with timeout and retry logic
            result = await self._execute_tool_with_retries(task.tool_name, resolved_args, agent)

            tool_duration = time.perf_counter() - tool_start

            # Validate result before marking as success
            is_valid_result = self._validate_tool_result(result, task)

            if not is_valid_result:
                raise ValueError(f"Tool {task.tool_name} returned invalid result: {type(result).__name__}")

            # Track successful tool call
            if self.progress_tracker:
                await self.progress_tracker.emit_event(ProgressEvent(
                    event_type="tool_call",
                    timestamp=time.time(),
                    node_name="TaskExecutorNode",
                    task_id=task.id,
                    status=NodeStatus.COMPLETED,
                    tool_name=task.tool_name,
                    tool_args=resolved_args,
                    tool_result=result,
                    duration=tool_duration,
                    success=True,
                    metadata={
                        "task_type": "ToolTask",
                        "result_type": type(result).__name__,
                        "result_length": len(str(result)),
                        "validation_passed": is_valid_result
                    }
                ))

            # FIXED: Store in variable manager with correct path structure
            if self.variable_manager:
                self.variable_manager.set(f"results.{task.id}.data", result)
                self.variable_manager.set(f"tasks.{task.id}.result", result)

            return result

        except Exception as e:
            tool_duration = time.perf_counter() - tool_start
            import traceback
            print(traceback.format_exc())

            # Detailed error tracking
            if self.progress_tracker:
                await self.progress_tracker.emit_event(ProgressEvent(
                    event_type="tool_call",
                    timestamp=time.time(),
                    node_name="TaskExecutorNode",
                    task_id=task.id,
                    status=NodeStatus.FAILED,
                    tool_name=task.tool_name,
                    tool_args=resolved_args,
                    duration=tool_duration,
                    success=False,
                    tool_error=str(e),
                    metadata={
                        "task_type": "ToolTask",
                        "error_type": type(e).__name__,
                        "retry_attempted": hasattr(self, '_retry_count')
                    }
                ))

            eprint(f"Tool execution failed for {task.tool_name}: {e}")
            raise
    async def _execute_llm_via_llmtool(self, task: LLMTask) -> Any:
        """Execute LLM task via LLMToolNode for consistency"""

        # Prepare context for LLMToolNode
        llm_shared = {
            "current_task_description": task.description,
            "formatted_context": {
                "recent_interaction": f"Executing LLM task: {task.description}",
                "session_summary": "",
                "task_context": f"Task ID: {task.id}, Priority: {task.priority}"
            },
            "variable_manager": self.variable_manager,
            "agent_instance": self.agent_instance,
            "available_tools": self.agent_instance.shared.get("available_tools", []) if self.agent_instance else [],
            "tool_capabilities": self.agent_instance._tool_capabilities if self.agent_instance else {},
            "fast_llm_model": self.fast_llm_model,
            "complex_llm_model": self.complex_llm_model,
            "progress_tracker": self.progress_tracker,
            "session_id": getattr(self, 'session_id', 'task_executor'),
            "use_fast_response": task.llm_config.get("model_preference", "fast") == "fast"
        }

        # Create LLMToolNode instance
        llm_node = LLMToolNode()

        # Execute via LLMToolNode
        try:
            result = await llm_node.run_async(llm_shared)
            # shared["current_response"]
            # shared["tool_calls_made"]
            # shared["llm_tool_conversation"]
            # shared["synthesized_response"]
            return llm_shared["current_response"]
        except Exception as e:
            eprint(f"LLMToolNode execution failed for task {task.id}: {e}")
            # Fallback to direct execution
            import traceback
            print(traceback.format_exc())
            return await self._execute_llm_task_enhanced(task)

    async def _execute_llm_task_enhanced(self, task: LLMTask) -> Any:
        """Enhanced LLM task execution with unified variable system"""
        if not LITELLM_AVAILABLE:
            raise Exception("LiteLLM not available for LLM tasks")

        # Get model preference with variable support
        llm_config = task.llm_config
        model_preference = llm_config.get("model_preference", "fast")

        if model_preference == "complex":
            model_to_use = self.variable_manager.get("system.complex_llm_model", "openrouter/openai/gpt-4o")
        else:
            model_to_use = self.variable_manager.get("system.fast_llm_model", "openrouter/anthropic/claude-3-haiku")

        # Build context for prompt
        context_data = {}
        for context_key in task.context_keys:
            value = self.variable_manager.get(context_key)
            if value is not None:
                context_data[context_key] = value

        # Resolve prompt template with full variable system
        final_prompt = self.variable_manager.format_text(
            task.prompt_template,
            context=context_data
        )

        llm_start = time.perf_counter()

        try:

            response = await litellm.acompletion(
                model=model_to_use,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", 2048)
            )

            result = response


            # Store intermediate result for other tasks
            self.variable_manager.set(f"tasks.{task.id}.result", result)

            # Output schema validation if present
            if task.output_schema:
                stripped = result.strip()

                try:
                    # Try JSON first if it looks like JSON
                    if stripped.startswith('{') or stripped.startswith('['):
                        parsed = json.loads(stripped)
                    else:
                        parsed = yaml.safe_load(stripped)

                    # Ensure metadata is a dict before updating
                    if not isinstance(task.metadata, dict):
                        task.metadata = {}

                    # Save parsed result
                    task.metadata["parsed_output"] = parsed

                except (json.JSONDecodeError, yaml.YAMLError):
                    # Save info about failure without logging output
                    if not isinstance(task.metadata, dict):
                        task.metadata = {}
                    task.metadata["parsed_output_error"] = "Invalid JSON/YAML format"

                except Exception as e:
                    if not isinstance(task.metadata, dict):
                        task.metadata = {}
                    task.metadata["parsed_output_error"] = f"Unexpected error: {str(e)}"

            return result
        except Exception as e:
            llm_duration = time.perf_counter() - llm_start

            if self.progress_tracker:
                await self.progress_tracker.emit_event(ProgressEvent(
                    event_type="llm_call",
                    node_name="TaskExecutorNode",
                    task_id=task.id,
                    status=NodeStatus.FAILED,
                    success=False,
                    duration=llm_duration,
                    llm_model=model_to_use,
                    error_details={
                        "message": str(e),
                        "type": type(e).__name__
                    }
                ))

            raise

    async def _execute_generic_via_llmtool(self, task: Task) -> Any:
        """
        Execute a generic task by treating its description as a query for the LLMToolNode.
        This provides a flexible fallback for undefined task types, leveraging the full
        reasoning and tool-use capabilities of the LLMToolNode.
        """
        # Prepare a shared context dictionary for the LLMToolNode, treating the
        # generic task's description as the primary query.
        llm_shared = {
            "current_task_description": task.description,
            "current_query": task.description,
            "formatted_context": {
                "recent_interaction": f"Executing generic task: {task.description}",
                "session_summary": f"The system needs to complete the following task: {task.description}",
                "task_context": f"Task ID: {task.id}, Priority: {task.priority}, Type: Generic"
            },
            "variable_manager": self.variable_manager,
            "agent_instance": self.agent_instance,
            # Generic tasks might require tools, so provide full tool context.
            "available_tools": self.agent_instance.shared.get("available_tools", []) if self.agent_instance else [],
            "tool_capabilities": self.agent_instance._tool_capabilities if self.agent_instance else {},
            "fast_llm_model": self.fast_llm_model,
            "complex_llm_model": self.complex_llm_model,
            "progress_tracker": self.progress_tracker,
            "session_id": getattr(self, 'session_id', 'task_executor_generic'),
            # Default to a fast model, assuming generic tasks are often straightforward.
            "use_fast_response": True
        }

        # Instantiate the LLMToolNode for this specific execution.
        llm_node = LLMToolNode()

        try:
            # Execute the node. It will run its internal loop for reasoning, tool calling, and response generation.
            # The results of the execution will be populated back into the `llm_shared` dictionary.
            await llm_node.run_async(llm_shared)

            # Extract the final response from the shared context populated by the node.
            # Prioritize the structured 'synthesized_response' but fall back to 'current_response'.
            final_response = llm_shared.get("synthesized_response", {}).get("synthesized_response")
            if not final_response:
                final_response = llm_shared.get("current_response", f"Generic task '{task.id}' processed.")

            return final_response

        except Exception as e:
            eprint(f"LLMToolNode execution for generic task {task.id} failed: {e}")
            # Re-raise the exception to allow the higher-level execution loop in
            # _execute_single_task to catch and handle it appropriately (e.g., for retries).
            raise

    async def _execute_decision_task_enhanced(self, task: DecisionTask) -> str:
        """Enhanced DecisionTask with intelligent replan assessment"""

        if not LITELLM_AVAILABLE:
            raise Exception("LiteLLM not available for decision tasks")

        # Build comprehensive context for decision
        decision_context = self._build_decision_context(task)

        # Enhanced decision prompt with full context
        enhanced_prompt = f"""
You are making a critical routing decision for task execution. Analyze all context carefully.

## Current Situation
{task.decision_prompt}

## Execution Context
{decision_context}

## Available Routing Options
{json.dumps(task.routing_map, indent=2)}

## Decision Guidelines
1. Only trigger "replan_from_here" if there's a genuine failure that cannot be recovered
2. Use "route_to_task" for normal flow continuation
3. Consider the full context, not just immediate results
4. Be conservative with replanning - it's expensive and can cause loops

Based on ALL the context above, what is your decision?
Respond with EXACTLY one of these options: {', '.join(task.routing_map.keys())}

Your decision:"""

        model_to_use = self.fast_llm_model if hasattr(self, 'fast_llm_model') else "openrouter/anthropic/claude-3-haiku"

        try:
            response = await litellm.acompletion(
                model=model_to_use,
                messages=[{"role": "user", "content": enhanced_prompt}],
                temperature=0.1,
                max_tokens=50
            )

            decision = response.choices[0].message.content.strip().lower().split('\n')[0]

            # Find matching key (case-insensitive)
            matched_key = None
            for key in task.routing_map:
                if key.lower() == decision:
                    matched_key = key
                    break

            if not matched_key:
                wprint(f"Decision '{decision}' not in routing map, using first option")
                matched_key = list(task.routing_map.keys())[0] if task.routing_map else "continue"

            routing_instruction = task.routing_map.get(matched_key, matched_key)

            # Enhanced metadata with decision reasoning
            if not hasattr(task, 'metadata'):
                task.metadata = {}

            task.metadata.update({
                "decision_made": matched_key,
                "routing_instruction": routing_instruction,
                "decision_context": decision_context,
                "replan_justified": self._assess_replan_necessity(matched_key, routing_instruction, decision_context)
            })

            # Handle dynamic planning instructions
            if isinstance(routing_instruction, dict) and "action" in routing_instruction:
                action = routing_instruction["action"]

                if action == "replan_from_here":
                    # Add extensive context for replanning
                    task.metadata["replan_context"] = {
                        "new_goal": routing_instruction.get("new_goal", "Continue with alternative approach"),
                        "failure_reason": f"Decision task {task.id} determined: {matched_key}",
                        "original_task": task.id,
                        "context": routing_instruction.get("context", ""),
                        "execution_history": self._get_execution_history_summary(),
                        "failed_approaches": self._identify_failed_approaches(),
                        "success_indicators": self._identify_success_patterns()
                    }

                self.variable_manager.set(f"tasks.{task.id}.result", {
                    "decision": matched_key,
                    "action": action,
                    "instruction": routing_instruction,
                    "confidence": self._calculate_decision_confidence(decision_context)
                })

                return action

            else:
                # Traditional routing
                next_task_id = routing_instruction if isinstance(routing_instruction, str) else str(routing_instruction)

                task.metadata.update({
                    "next_task_id": next_task_id,
                    "routing_action": "route_to_task"
                })

                self.variable_manager.set(f"tasks.{task.id}.result", {
                    "decision": matched_key,
                    "next_task": next_task_id
                })

                return matched_key

        except Exception as e:
            eprint(f"Enhanced decision task failed: {e}")
            raise

    async def post_async(self, shared, prep_res, exec_res):
        """Erweiterte Post-Processing mit dynamischer Plan-Anpassung"""

        # Results store in shared state integrieren
        shared["results"] = self.results_store

        if exec_res is None or "error" in exec_res:
            shared["executor_performance"] = {"status": "error", "last_error": exec_res.get("error")}
            return "execution_error"

        if exec_res["status"] == "waiting":
            shared["executor_status"] = "waiting_for_dependencies"
            return "waiting"

        # Performance-Metriken speichern
        performance_data = {
            "execution_duration": exec_res.get("execution_duration", 0),
            "strategy_used": exec_res.get("strategy_used", "unknown"),
            "completed_tasks": exec_res.get("completed_tasks", 0),
            "failed_tasks": exec_res.get("failed_tasks", 0),
            "success_rate": exec_res.get("completed_tasks", 0) / max(len(exec_res.get("results", [])), 1),
            "timestamp": datetime.now().isoformat()
        }
        shared["executor_performance"] = performance_data

        # Check for dynamic planning actions
        planning_action_detected = False

        for result in exec_res.get("results", []):
            task_id = result["task_id"]
            if task_id in shared["tasks"]:
                task = shared["tasks"][task_id]
                task.status = result["status"]

                if result["status"] == "completed":
                    task.result = result["result"]

                    # Check for planning actions from DecisionTasks
                    if hasattr(task, 'metadata') and task.metadata:
                        routing_action = task.metadata.get("routing_action")

                        if routing_action == "replan_from_here":
                            shared["needs_dynamic_replan"] = True
                            shared["replan_context"] = task.metadata.get("replan_context", {})
                            planning_action_detected = True
                            rprint(f"Dynamic replan triggered by task {task_id}")

                        elif routing_action == "append_plan":
                            shared["needs_plan_append"] = True
                            shared["append_context"] = task.metadata.get("append_context", {})
                            planning_action_detected = True
                            rprint(f"Plan append triggered by task {task_id}")

                    # Store verification results if available
                    if result.get("verification"):
                        if not hasattr(task, 'metadata'):
                            task.metadata = {}
                        task.metadata["verification"] = result["verification"]

                elif result["status"] == "failed":
                    task.error = result.get("error", "Unknown error")

        # Return appropriate status based on planning actions
        if planning_action_detected:
            if shared.get("needs_dynamic_replan"):
                return "needs_dynamic_replan"  # Goes to PlanReflectorNode
            elif shared.get("needs_plan_append"):
                return "needs_plan_append"  # Goes to PlanReflectorNode

        # Regular completion checking
        current_plan = shared["current_plan"]
        if current_plan:
            all_finished = all(
                shared["tasks"][task.id].status in ["completed", "failed"]
                for task in current_plan.tasks
            )

            if all_finished:
                current_plan.status = "completed"
                shared["plan_completion_time"] = datetime.now().isoformat()
                rprint(f"Plan {current_plan.id} finished")
                return "plan_completed"
            else:
                ready_tasks = [
                    task for task in current_plan.tasks
                    if shared["tasks"][task.id].status == "pending"
                ]

                if ready_tasks:
                    return "continue_execution"
                else:
                    return "waiting"

        return "execution_complete"

    def get_execution_statistics(self) -> dict[str, Any]:
        """Erhalte detaillierte Ausführungsstatistiken"""
        if not self.execution_history:
            return {"message": "No execution history available"}

        history = self.execution_history

        return {
            "total_executions": len(history),
            "average_duration": sum(h["duration"] for h in history) / len(history),
            "success_rate": sum(1 for h in history if h["success"]) / len(history),
            "strategy_usage": {
                strategy: sum(1 for h in history if h["strategy"] == strategy)
                for strategy in set(h["strategy"] for h in history)
            },
            "total_tasks_executed": sum(h["tasks_executed"] for h in history),
            "average_confidence": sum(h["plan_confidence"] for h in history) / len(history),
            "recent_performance": history[-3:] if len(history) >= 3 else history
        }

    def _resolve_task_variables(self, data):
        """Unified variable resolution for any task data"""
        if isinstance(data, str):
            res = self.variable_manager.format_text(data)
            return res
        elif isinstance(data, dict):
            resolved = {}
            for key, value in data.items():
                resolved[key] = self._resolve_task_variables(value)
            return resolved
        elif isinstance(data, list):
            return [self._resolve_task_variables(item) for item in data]
        else:
            return data

    def _store_task_result(self, task_id: str, result: Any, success: bool, error: str = None):
        """Store task result in unified variable system"""
        result_data = {
            "data": result,
            "metadata": {
                "task_type": "task",
                "completed_at": datetime.now().isoformat(),
                "success": success
            }
        }

        if error:
            result_data["error"] = error
            result_data["metadata"]["success"] = False

        # Store in results_store and update variable manager
        self.results_store[task_id] = result_data
        self.variable_manager.set_results_store(self.results_store)

        # FIXED: Store actual result data, not the wrapper object
        self.variable_manager.set(f"results.{task_id}.data", result)
        self.variable_manager.set(f"results.{task_id}.metadata", result_data["metadata"])
        if error:
            self.variable_manager.set(f"results.{task_id}.error", error)

    def _build_decision_context(self, task: DecisionTask) -> str:
        """Build comprehensive context for decision making"""

        context_parts = []

        # Recent execution results
        recent_results = []
        for task_id, result_data in list(self.results_store.items())[-3:]:
            success = result_data.get("metadata", {}).get("success", False)
            status = "✓" if success else "✗"
            data_preview = str(result_data.get("data", ""))[:100] + "..."
            recent_results.append(f"{status} {task_id}: {data_preview}")

        if recent_results:
            context_parts.append("Recent Results:\n" + "\n".join(recent_results))

        # Variable context
        if self.variable_manager:
            available_vars = list(self.variable_manager.get_available_variables().keys())[:10]
            context_parts.append(f"Available Variables: {', '.join(available_vars)}")

        # Execution history
        execution_summary = self._get_execution_history_summary()
        if execution_summary:
            context_parts.append(f"Execution Summary: {execution_summary}")

        # Current world model insights
        world_insights = self._get_world_model_insights()
        if world_insights:
            context_parts.append(f"Known Facts: {world_insights}")

        return "\n\n".join(context_parts)

    def _assess_replan_necessity(self, decision: str, routing_instruction: Any, context: str) -> bool:
        """Assess if replanning is truly necessary"""

        if not isinstance(routing_instruction, dict):
            return False

        action = routing_instruction.get("action", "")
        if action != "replan_from_here":
            return False

        # Check if we have genuine failures
        genuine_failures = "error" in context.lower() or "failed" in context.lower()
        alternative_available = len(self.results_store) > 0  # Have some results to work with

        # Be conservative - only replan if really necessary
        return genuine_failures and not alternative_available

    async def _execute_tool_with_retries(self, tool_name: str, args: dict, agent, max_retries: int = 2) -> Any:
        """Execute tool with retry logic"""

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                result = await agent.arun_function(tool_name, **args)

                # Additional validation - check if result indicates success
                if self._is_tool_result_success(result):
                    return result
                elif attempt < max_retries:
                    wprint(f"Tool {tool_name} returned unclear result, retrying...")
                    continue
                else:
                    return result

            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    wprint(f"Tool {tool_name} failed (attempt {attempt + 1}), retrying: {e}")
                    await asyncio.sleep(0.5 * (attempt + 1))  # Progressive delay
                else:
                    eprint(f"Tool {tool_name} failed after {max_retries + 1} attempts")

        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Tool {tool_name} failed without exception")

    def _validate_tool_result(self, result: Any, task: ToolTask) -> bool:
        """Validate tool result to prevent false failures"""

        # Basic validation
        if result is None:
            return False

        # Check for common error indicators
        if isinstance(result, str):
            error_indicators = ["error", "failed", "exception", "timeout", "not found"]
            result_lower = result.lower()

            # If result contains error indicators but also has substantial content, it might still be valid
            has_errors = any(indicator in result_lower for indicator in error_indicators)
            has_content = len(result.strip()) > 20

            if has_errors and not has_content:
                return False

        # Check against expectation if provided
        if hasattr(task, 'expectation') and task.expectation:
            expectation_keywords = task.expectation.lower().split()
            result_text = str(result).lower()

            # At least one expectation keyword should be present
            if not any(keyword in result_text for keyword in expectation_keywords):
                wprint(f"Tool result doesn't match expectation: {task.expectation}")

        return True

    def _is_tool_result_success(self, result: Any) -> bool:
        """Determine if a tool result indicates success"""

        if result is None:
            return False

        if isinstance(result, bool):
            return result

        if isinstance(result, list | dict):
            return len(result) > 0

        if isinstance(result, str):
            # Check for explicit success/failure indicators
            result_lower = result.lower()

            success_indicators = ["success", "completed", "found", "retrieved", "generated"]
            failure_indicators = ["error", "failed", "not found", "timeout", "exception"]

            has_success = any(indicator in result_lower for indicator in success_indicators)
            has_failure = any(indicator in result_lower for indicator in failure_indicators)

            if has_success and not has_failure:
                return True
            elif has_failure and not has_success:
                return False
            else:
                # Ambiguous - assume success if there's substantial content
                return len(result.strip()) > 10

        # For other types, assume success if not None
        return True

    def _get_execution_history_summary(self) -> str:
        """Get concise execution history summary"""

        if not hasattr(self, 'execution_history') or not self.execution_history:
            return "No execution history"

        recent = self.execution_history[-3:]  # Last 3 executions
        summaries = []

        for hist in recent:
            status = "Success" if hist.get("success", False) else "Failed"
            duration = hist.get("duration", 0)
            strategy = hist.get("strategy", "Unknown")
            summaries.append(f"{strategy}: {status} ({duration:.1f}s)")

        return "; ".join(summaries)

    def _identify_failed_approaches(self) -> list[str]:
        """Identify approaches that have consistently failed"""

        failed_approaches = []

        # Analyze failed tasks
        for _task_id, result_data in self.results_store.items():
            if not result_data.get("metadata", {}).get("success", True):
                error = result_data.get("error", "")
                if "tool" in error.lower():
                    failed_approaches.append("direct_tool_approach")
                elif "search" in error.lower():
                    failed_approaches.append("search_based_approach")
                elif "llm" in error.lower():
                    failed_approaches.append("llm_direct_approach")

        return list(set(failed_approaches))

    def _identify_success_patterns(self) -> list[str]:
        """Identify patterns that have led to success"""

        success_patterns = []

        # Analyze successful tasks
        successful_results = [
            r for r in self.results_store.values()
            if r.get("metadata", {}).get("success", False)
        ]

        if successful_results:
            # Identify common patterns
            if len(successful_results) > 1:
                success_patterns.append("multi_step_approach")

            for result in successful_results:
                data = result.get("data", "")
                if isinstance(data, str) and len(data) > 100:
                    success_patterns.append("detailed_information_retrieval")

        return list(set(success_patterns))

    def _get_world_model_insights(self) -> str:
        """Get relevant insights from world model"""

        if not self.variable_manager:
            return ""

        world_data = self.variable_manager.scopes.get("world", {})
        if not world_data:
            return "No world model data"

        # Get most recent or relevant facts
        recent_facts = []
        for key, value in list(world_data.items())[:5]:  # Top 5 facts
            recent_facts.append(f"{key}: {str(value)[:50]}...")

        return "; ".join(recent_facts)

    def _calculate_decision_confidence(self, context: str) -> float:
        """Calculate confidence in decision based on context"""

        # Simple heuristic based on context richness
        base_confidence = 0.5

        # Boost confidence if we have rich context
        if len(context) > 200:
            base_confidence += 0.2

        # Boost if we have recent results
        if "Recent Results:" in context:
            base_confidence += 0.2

        # Reduce if there are many failures
        failure_count = context.lower().count("failed") + context.lower().count("error")
        base_confidence -= min(failure_count * 0.1, 0.3)

        return max(0.1, min(1.0, base_confidence))

@with_progress_tracking
class LLMToolNode(AsyncNode):
    """Enhanced LLM tool with automatic tool calling and agent loop integration"""

    def __init__(self, model: str = None, max_tool_calls: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.model = model or os.getenv("COMPLEXMODEL", "openrouter/qwen/qwen3-code")
        self.max_tool_calls = max_tool_calls
        self.call_log = []

    async def prep_async(self, shared):
        context = shared.get("formatted_context", {})
        task_description = shared.get("current_task_description", shared.get("current_query", ""))

        # Variable Manager integration
        variable_manager = shared.get("variable_manager")
        agent_instance = shared.get("agent_instance")

        return {
            "task_description": task_description,
            "context": context,
            "context_manager": shared.get("context_manager"),
            "session_id": shared.get("session_id"),
            "variable_manager": variable_manager,
            "agent_instance": agent_instance,
            "available_tools": shared.get("available_tools", []),
            "tool_capabilities": shared.get("tool_capabilities", {}),
            "persona_config": shared.get("persona_config"),
            "base_system_message": variable_manager.format_text(agent_instance.amd.get_system_message_with_persona()),
            "recent_interaction": context.get("recent_interaction", ""),
            "session_summary": context.get("session_summary", ""),
            "task_context": context.get("task_context", ""),
            "fast_llm_model": shared.get("fast_llm_model"),
            "complex_llm_model": shared.get("complex_llm_model"),
            "progress_tracker": shared.get("progress_tracker"),
            "tool_call_count": 0
        }

    async def exec_async(self, prep_res):
        """Main execution with tool calling loop"""
        if not LITELLM_AVAILABLE:
            return await self._fallback_response(prep_res)

        progress_tracker = prep_res.get("progress_tracker")

        conversation_history = []
        tool_call_count = 0
        final_response = None
        model_to_use = "auto"
        total_llm_calls = 0
        total_cost = 0.0
        total_tokens = 0

        # Initial system message with tool awareness
        system_message = self._build_tool_aware_system_message(prep_res)

        # Initial user prompt with variable resolution
        initial_prompt = await self._build_context_aware_prompt(prep_res)
        conversation_history.append({"role": "user", "content":  prep_res["variable_manager"].format_text(initial_prompt)})
        runs = 0
        while tool_call_count < self.max_tool_calls:
            runs += 1
            # Get LLM response
            messages = [{"role": "system", "content": system_message + ( "\nfist look at the context and reason over you intal step." if runs == 1 else "")}] + conversation_history

            model_to_use = self._select_optimal_model(prep_res["task_description"], prep_res)

            llm_start = time.perf_counter()

            try:
                agent_instance = prep_res["agent_instance"]
                response = await agent_instance.a_run_llm_completion(
                    model=model_to_use,
                    messages=messages,
                    temperature=0.7,
                    # max_tokens=2048,
                    node_name="LLMToolNode", task_id="llm_phase_" + str(runs)
                )

                llm_response = response
                if not llm_response:
                    final_response = "I encountered an error while processing your request."
                    break


                # Check for tool calls
                tool_calls = self._extract_tool_calls(llm_response)

                llm_response = prep_res["variable_manager"].format_text(llm_response)
                conversation_history.append({"role": "assistant", "content": llm_response})


                if not tool_calls:
                    # No more tool calls, this is the final response
                    final_response = llm_response
                    break

                # Execute tool calls
                tool_results = await self._execute_tool_calls(tool_calls, prep_res)
                tool_call_count += len(tool_calls)

                # Add tool results to conversation
                tool_results_text = self._format_tool_results(tool_results)
                conversation_history.append({"role": "user",
                                             "content": f"Tool results:\n{tool_results_text}\n\nPlease continue with the next action do nor repeat or provide your final response."})

                # Update variable manager with tool results
                self._update_variables_with_results(tool_results, prep_res["variable_manager"])

            except Exception as e:
                llm_duration = time.perf_counter() - llm_start

                if progress_tracker:
                    await progress_tracker.emit_event(ProgressEvent(
                        event_type="llm_call",  # Konsistenter Event-Typ
                        node_name="LLMToolNode",
                        session_id=prep_res.get("session_id"),
                        status=NodeStatus.FAILED,
                        success=False,
                        duration=llm_duration,
                        llm_model=model_to_use,
                        error_details={
                            "message": str(e),
                            "type": type(e).__name__
                        },
                        metadata={"call_number": total_llm_calls + 1}
                    ))
                eprint(f"LLM tool execution failed: {e}")
                final_response = f"I encountered an error while processing: {str(e)}"
                import traceback
                print(traceback.format_exc())
                break


        return {
            "success": True,
            "final_response": final_response or "I was unable to complete the request.",
            "tool_calls_made": tool_call_count,
            "conversation_history": conversation_history,
            "model_used": model_to_use,
            "llm_statistics": {
                "total_calls": total_llm_calls,
                "total_cost": total_cost,
                "total_tokens": total_tokens
            }
        }

    def _build_tool_aware_system_message(self, prep_res: dict) -> str:
        """Build a unified intelligent, tool-aware system message with context and relevance analysis."""

        # Base system message
        base_message = prep_res.get("base_system_message", "You are a helpful AI assistant.")
        available_tools = prep_res.get("available_tools", [])
        tool_capabilities = prep_res.get("tool_capabilities", {})
        variable_manager = prep_res.get("variable_manager")
        context = prep_res.get("context", {})
        agent_instance = prep_res.get("agent_instance")
        query = prep_res.get('task_description', '').lower()

        base_message += ("\n\nAlways follow this action pattern"
                         "**THINK** -> **PLAN** -> (**ADJUST**) and (**UPDATE**) or **ACT** using tools!\n"
                         "all progress must be stored to ( variable system, memory, external services )!\n"
                         "if working on code or file based tasks, update and crate the files!\n"
                         "all text only steps ar discarded and not stored! only the final response is stored! ")

        # --- Part 1: List available tools & capabilities ---
        if available_tools:
            base_message += f"\n\n## Available Tools\nYou have access to these tools: {', '.join(available_tools)}\n"
            base_message += "Results will be stored to results.{tool_name}.data"

            for tool_name in available_tools:
                if tool_name in tool_capabilities:
                    cap = tool_capabilities[tool_name]
                    base_message += f"\n**{tool_name}**: {cap.get('primary_function', 'No description')}"
                    use_cases = cap.get('use_cases', [])
                    if use_cases:
                        base_message += f"\n  Use cases: {', '.join(use_cases[:3])}"

            # base_message += "\n\n## Tool Usage\nTo use tools, respond with:\nTOOL_CALL: tool_name(arg1='value1', arg2='value2')\nYou can make multiple tool calls in one response."
            base_message += """
## Tool Usage
To use tools, respond with a YAML block:
```yaml
TOOL_CALLS:
  - tool: tool_name
    args:
      arg1: value1
      arg2: value2
  - tool: another_tool
    args:
      code: |
        def example():
            return "multi-line code"
      text: |
        Multi-line text
        with arbitrary content
```
You can call multiple tools in one response. Use | for multi-line strings containing code or complex text."""

        # --- Part 2: Add variable context ---
        if variable_manager:
            var_context = variable_manager.get_llm_variable_context()
            if var_context:
                base_message += f"\n\n## Variable Context\n{var_context}"

        # --- Part 3: Intelligent tool analysis ---
        if not agent_instance or not hasattr(agent_instance, '_tool_capabilities'):
            return base_message + "\n\n⚠ No intelligent tool analysis available."

        capabilities = agent_instance._tool_capabilities
        analysis_parts = ["\n\n## Intelligent Tool Analysis"]

        for tool_name, cap in capabilities.items():
            analysis_parts.append(f"\n{tool_name}{cap.get('args_schema', '()')}:")
            analysis_parts.append(f"- Function: {cap.get('primary_function', 'Unknown')}")

            # Calculate relevance score
            relevance_score = self._calculate_tool_relevance(query, cap)
            analysis_parts.append(f"- Query relevance: {relevance_score:.2f}")

            if relevance_score > 0.65:
                analysis_parts.append("- ⭐ HIGHLY RELEVANT - SHOULD USE THIS TOOL!")

            # Trigger phrase matching
            triggers = cap.get('trigger_phrases', [])
            matched_triggers = [t for t in triggers if t.lower() in query]
            if matched_triggers:
                analysis_parts.append(f"- Matched triggers: {matched_triggers}")

            # Show top use cases
            use_cases = cap.get('use_cases', [])[:3]
            if use_cases:
                analysis_parts.append(f"- Use cases: {', '.join(use_cases)}")

        # Combine everything into a final message
        return base_message + "\n"+ "\n".join(analysis_parts)

    def _calculate_tool_relevance(self, query: str, capabilities: dict) -> float:
        """Calculate how relevant a tool is to the current query"""

        query_words = set(query.lower().split())

        # Check trigger phrases
        trigger_score = 0.0
        triggers = capabilities.get('trigger_phrases', [])
        for trigger in triggers:
            trigger_words = set(trigger.lower().split())
            if trigger_words.intersection(query_words):
                trigger_score += 0.04
        # Check confidence triggers if available
        conf_triggers = capabilities.get('confidence_triggers', {})
        for phrase, confidence in conf_triggers.items():
            if phrase.lower() in query:
                trigger_score += confidence/10
        # Check indirect connections
        indirect = capabilities.get('indirect_connections', [])
        for connection in indirect:
            connection_words = set(connection.lower().split())
            if connection_words.intersection(query_words):
                trigger_score += 0.02
        return min(1.0, trigger_score)

    @staticmethod
    def _extract_tool_calls_custom(text: str) -> list[dict]:
        """Extract tool calls from LLM response"""

        tool_calls = []

        pattern = r'TOOL_CALL:'
        matches = _extract_meta_tool_calls(text, pattern)

        for tool_name, args_str in matches:
            try:
                # Parse arguments
                args = _parse_tool_args(args_str)
                tool_calls.append({
                    "tool_name": tool_name,
                    "arguments": args
                })
            except Exception as e:
                wprint(f"Failed to parse tool call {tool_name}: {e}")

        return tool_calls

    @staticmethod
    def _extract_tool_calls(text: str) -> list[dict]:
        """Extract tool calls from LLM response using YAML format"""
        import re

        import yaml

        tool_calls = []

        # Pattern to find YAML blocks with TOOL_CALLS
        yaml_pattern = r'```yaml\s*\n(.*?TOOL_CALLS:.*?)\n```'
        yaml_matches = re.findall(yaml_pattern, text, re.DOTALL | re.IGNORECASE)

        # Also try without code blocks for simpler cases
        if not yaml_matches:
            simple_pattern = r'TOOL_CALLS:\s*\n((?:.*\n)*?)(?=\n\S|\Z)'
            simple_matches = re.findall(simple_pattern, text, re.MULTILINE)
            if simple_matches:
                yaml_matches = [f"TOOL_CALLS:\n{match}" for match in simple_matches]

        for yaml_content in yaml_matches:
            try:
                # Parse YAML content
                parsed_yaml = yaml.safe_load(yaml_content)

                if not isinstance(parsed_yaml, dict) or 'TOOL_CALLS' not in parsed_yaml:
                    continue

                calls = parsed_yaml['TOOL_CALLS']
                if not isinstance(calls, list):
                    calls = [calls]  # Handle single tool call

                for call in calls:
                    if isinstance(call, dict) and 'tool' in call:
                        tool_call = {
                            "tool_name": call['tool'],
                            "arguments": call.get('args', {})
                        }
                        tool_calls.append(tool_call)

            except yaml.YAMLError as e:
                wprint(f"Failed to parse YAML tool calls: {e}")
            except Exception as e:
                wprint(f"Error processing tool calls: {e}")

        return tool_calls

    def _select_optimal_model(self, task_description: str, prep_res: dict) -> str:
        """Select optimal model based on task complexity and available resources"""
        complexity_score = self._estimate_task_complexity(task_description, prep_res)
        if complexity_score > 0.7:
            return prep_res.get("complex_llm_model", "openrouter/openai/gpt-4o")
        else:
            return prep_res.get("fast_llm_model", "openrouter/anthropic/claude-3-haiku")

    def _estimate_task_complexity(self, task_description: str, prep_res: dict) -> float:
        """Estimate task complexity based on description, length, and available tools"""
        # Simple heuristic: length + keyword matching + tool availability
        description_length_score = min(len(task_description) / 500, 1.0)  # cap at 1.0
        keywords = ["analyze", "research", "generate", "simulate", "complex", "deep", "strategy"]
        keyword_score = sum(1 for k in keywords if k in task_description.lower()) / len(keywords)
        tool_score = min(len(prep_res.get("available_tools", [])) / 10, 1.0)

        # Weighted sum
        complexity_score = (0.5 * description_length_score) + (0.3 * keyword_score) + (0.2 * tool_score)
        return round(complexity_score, 2)

    async def _fallback_response(self, prep_res: dict) -> dict:
        """Fallback response if LiteLLM is not available"""
        wprint("LiteLLM not available — using fallback response.")
        return {
            "success": False,
            "final_response": (
                "I'm unable to process this request fully right now because the LLM interface "
                "is not available. Please try again later or check system configuration."
            ),
            "tool_calls_made": 0,
            "conversation_history": [],
            "model_used": None
        }

    async def _execute_tool_calls(self, tool_calls: list[dict], prep_res: dict) -> list[dict]:
        """Execute tool calls via agent"""
        agent_instance = prep_res.get("agent_instance")
        variable_manager = prep_res.get("variable_manager")
        progress_tracker = prep_res.get("progress_tracker")

        results = []

        for tool_call in tool_calls:
            tool_name = tool_call["tool_name"]
            arguments = tool_call["arguments"]

            # Start tool tracking
            tool_start = time.perf_counter()

            if progress_tracker:
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="tool_call",
                    timestamp=time.time(),
                    status=NodeStatus.RUNNING,
                    node_name="LLMToolNode",
                    tool_name=tool_name,
                    tool_args=arguments,
                    session_id=prep_res.get("session_id"),
                    metadata={"tool_call_initiated": True}
                ))

            try:
                # Resolve variables in arguments
                if variable_manager:
                    resolved_args = {}
                    for key, value in arguments.items():
                        if isinstance(value, str):
                            resolved_args[key] = variable_manager.format_text(value)
                        else:
                            resolved_args[key] = value
                else:
                    resolved_args = arguments

                # Execute via agent
                result = await agent_instance.arun_function(tool_name, **resolved_args)
                tool_duration = time.perf_counter() - tool_start
                variable_manager.set(f"results.{tool_name}.data", result)
                if progress_tracker:
                    await progress_tracker.emit_event(ProgressEvent(
                        event_type="tool_call",
                        timestamp=time.time(),
                        node_name="LLMToolNode",
                        status=NodeStatus.COMPLETED,
                        tool_name=tool_name,
                        tool_args=resolved_args,
                        tool_result=result,
                        duration=tool_duration,
                        success=True,
                        session_id=prep_res.get("session_id"),
                        metadata={
                            "result_type": type(result).__name__,
                            "result_length": len(str(result))
                        }
                    ))
                results.append({
                    "tool_name": tool_name,
                    "arguments": resolved_args,
                    "success": True,
                    "result": result
                })

            except Exception as e:
                tool_duration = time.perf_counter() - tool_start
                error_message = str(e)
                error_type = type(e).__name__
                import traceback
                print(traceback.format_exc())


                if progress_tracker:
                    await progress_tracker.emit_event(ProgressEvent(
                        event_type="tool_call",
                        timestamp=time.time(),
                        node_name="LLMToolNode",
                        status=NodeStatus.FAILED,
                        tool_name=tool_name,
                        tool_args=arguments,
                        duration=tool_duration,
                        success=False,
                        tool_error=error_message,
                        session_id=prep_res.get("session_id"),
                        metadata={
                            "error": error_message,
                            "error_message": error_message,
                            "error_type": error_type
                        }
                    ))

                    # FIXED: Also send generic error event for error log
                    await progress_tracker.emit_event(ProgressEvent(
                        event_type="error",
                        timestamp=time.time(),
                        node_name="LLMToolNode",
                        status=NodeStatus.FAILED,
                        success=False,
                        tool_name=tool_name,
                        metadata={
                            "error": error_message,
                            "error_message": error_message,
                            "error_type": error_type,
                            "source": "tool_execution",
                            "tool_name": tool_name,
                            "tool_args": arguments
                        }
                    ))
                eprint(f"Tool execution failed {tool_name}: {e}")
                results.append({
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "success": False,
                    "error": str(e)
                })

        return results

    def _format_tool_results(self, results: list[dict]) -> str:
        """Format tool results for LLM"""
        formatted = []

        for result in results:
            if result["success"]:
                formatted.append(f"✓ {result['tool_name']}: {result['result']}")
            else:
                formatted.append(f"✗ {result['tool_name']}: ERROR - {result['error']}")

        return "\n".join(formatted)

    def _update_variables_with_results(self, results: list[dict], variable_manager):
        """Update variable manager with tool results"""
        if not variable_manager:
            return

        for i, result in enumerate(results):
            if result["success"]:
                tool_name = result['tool_name']
                result_data = result['result']

                # FIXED: Store result in proper variable paths
                variable_manager.set(f"results.{tool_name}.data", result_data)
                variable_manager.set(f"tools.{tool_name}.result", result_data)

                # Also store with index for multiple calls to same tool
                var_key = f"tool_result_{tool_name}_{i}"
                variable_manager.set(var_key, result_data)

    async def _build_context_aware_prompt(self, prep_res: dict) -> str:
        """Build context-aware prompt mit UnifiedContextManager Integration"""
        variable_manager = prep_res.get("variable_manager")
        agent_instance = prep_res.get("agent_instance")
        context = prep_res.get("context", {})

        #Get unified context if available
        context_manager = prep_res.get("context_manager")
        session_id = prep_res.get("session_id", "default")

        unified_context_parts = []

        if context_manager:
            try:
                # Get unified context für LLM Tool usage
                unified_context = await context_manager.build_unified_context(session_id, prep_res.get("task_description", ""))

                # Format unified context for LLM consumption
                chat_history = unified_context.get("chat_history", [])
                if chat_history:
                    unified_context_parts.append("## Recent Conversation from Session")
                    for msg in chat_history[-5:]:  # Last 5 messages
                        timestamp = msg.get('timestamp', '')[:19]
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')[:300] + ("..." if len(msg.get('content', '')) > 300 else "")
                        unified_context_parts.append(f"[{timestamp}] {role}: {content}")

                # Execution state from unified context
                execution_state = unified_context.get("execution_state", {})
                if execution_state:
                    system_status = execution_state.get('system_status', 'unknown')
                    active_tasks = execution_state.get('active_tasks', [])
                    recent_completions = execution_state.get('recent_completions', [])

                    unified_context_parts.append("\n## Current System State")
                    unified_context_parts.append(f"Status: {system_status}")
                    if active_tasks:
                        unified_context_parts.append(f"Active Tasks: {len(active_tasks)}")
                    if recent_completions:
                        unified_context_parts.append(f"Recent Completions: {len(recent_completions)}")

                # Available results from unified context
                variables_context = unified_context.get("variables", {})
                recent_results = variables_context.get("recent_results", [])
                if recent_results:
                    unified_context_parts.append("\n## Available Results")
                    for result in recent_results[:3]:  # Top 3 results
                        task_id = result.get("task_id", "unknown")
                        preview = result.get("preview", "")[:100] + "..."
                        success = "✅" if result.get("success") else "❌"
                        unified_context_parts.append(f"{success} {task_id}: {preview}")

                # World model facts from unified context
                relevant_facts = unified_context.get("relevant_facts", [])
                if relevant_facts:
                    unified_context_parts.append("\n## Relevant Known Facts")
                    for key, value in relevant_facts[:3]:  # Top 3 facts
                        fact_preview = str(value)[:100] + ("..." if len(str(value)) > 100 else "")
                        unified_context_parts.append(f"- {key}: {fact_preview}")

            except Exception as e:
                unified_context_parts.append(f"## Context Error\nUnified context unavailable: {str(e)}")

        # EXISTIEREND: Keep existing context building (backwards compatibility)
        prompt_parts = []

        # Add unified context first (primary)
        if unified_context_parts:
            prompt_parts.extend(unified_context_parts)

        # Add existing context sections (secondary)
        recent_interaction = prep_res.get("recent_interaction", "")
        session_summary = prep_res.get("session_summary", "")
        task_context = prep_res.get("task_context", "")

        if recent_interaction:
            prompt_parts.append(f"\n## Recent Interaction Context\n{recent_interaction}")
            prompt_parts.append("\n**Important**: NO META_TOOL_CALLs needed in this section! and not avalabel\n use tools from Intelligent Tool Analysis only!")
        if session_summary:
            prompt_parts.append(f"\n## Session Summary\n{session_summary}")
        if task_context:
            prompt_parts.append(f"\n## Task Context\n{task_context}")

        # Add main task
        task_description = prep_res.get("task_description", "")
        if task_description:
            prompt_parts.append(f"\n## Current Request\n{task_description}")

        # Variable suggestions (existing functionality)
        if variable_manager and task_description:
            suggestions = variable_manager.get_variable_suggestions(task_description)
            if suggestions:
                prompt_parts.append(f"\n## Available Variables\nYou can use: {', '.join(suggestions)}")

        # Final variable resolution
        final_prompt = "\n".join(prompt_parts)
        if variable_manager:
            final_prompt = variable_manager.format_text(final_prompt)

        return final_prompt

    async def post_async(self, shared, prep_res, exec_res):
        shared["current_response"] = exec_res.get("final_response", "Task completed.")
        shared["tool_calls_made"] = exec_res.get("tool_calls_made", 0)
        shared["llm_tool_conversation"] = exec_res.get("conversation_history", [])
        shared["synthesized_response"] = {"synthesized_response":exec_res.get("final_response", "Task completed."),
                                          "confidence": (0.7 if exec_res.get("model_used") == prep_res.get("complex_llm_model") else 0.6) if exec_res.get("success", False) else 0,
                                          "metadata": exec_res.get("metadata", {"model_used": exec_res.get("model_used")}),
                                          "synthesis_method": "llm_tool"}
        return "llm_tool_complete"

@with_progress_tracking
class StateSyncNode(AsyncNode):
    """Synchronize state between world model and shared store"""
    async def prep_async(self, shared):
        world_model = shared.get("world_model", {})
        session_data = shared.get("session_data", {})
        tasks = shared.get("tasks", {})
        system_status = shared.get("system_status", "idle")

        return {
            "world_model": world_model,
            "session_data": session_data,
            "tasks": tasks,
            "system_status": system_status,
            "sync_timestamp": datetime.now().isoformat()
        }

    async def exec_async(self, prep_res):
        # Perform intelligent state synchronization
        sync_result = {
            "world_model_updates": {},
            "session_updates": {},
            "task_updates": {},
            "conflicts_resolved": [],
            "sync_successful": True
        }

        # Update world model with new information
        if "current_response" in prep_res:
            # Extract learnable facts from responses
            extracted_facts = self._extract_facts(prep_res.get("current_response", ""))
            sync_result["world_model_updates"].update(extracted_facts)

        # Sync task states
        for task_id, task in prep_res["tasks"].items():
            if task.status == "completed" and task.result:
                # Store task results in world model
                fact_key = f"task_{task_id}_result"
                sync_result["world_model_updates"][fact_key] = task.result

        return sync_result

    def _extract_facts(self, text: str) -> dict[str, Any]:
        """Extract learnable facts from text"""
        facts = {}
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            # Look for definitive statements
            if ' is ' in line and not line.startswith('I ') and not '?' in line:
                parts = line.split(' is ', 1)
                if len(parts) == 2:
                    subject = parts[0].strip().lower()
                    predicate = parts[1].strip().rstrip('.')
                    if len(subject.split()) <= 3:  # Keep subjects simple
                        facts[subject] = predicate

        return facts

    async def post_async(self, shared, prep_res, exec_res):
        # Apply the synchronization results
        if exec_res["sync_successful"]:
            shared["world_model"].update(exec_res["world_model_updates"])
            shared["session_data"].update(exec_res["session_updates"])
            shared["last_sync"] = datetime.now()
            return "sync_complete"
        else:
            wprint("State synchronization failed")
            return "sync_failed"

@with_progress_tracking
class CompletionCheckerNode(AsyncNode):
    """Breaks infinite cycles by checking actual completion status"""

    def __init__(self):
        super().__init__()
        self.execution_count = 0
        self.max_cycles = 5  # Prevent infinite loops

    async def prep_async(self, shared):
        current_plan = shared.get("current_plan")
        tasks = shared.get("tasks", {})

        return {
            "current_plan": current_plan,
            "tasks": tasks,
            "execution_count": self.execution_count
        }

    async def exec_async(self, prep_res):
        self.execution_count += 1

        # Safety check: prevent infinite loops
        if self.execution_count > self.max_cycles:
            wprint(f"Max execution cycles ({self.max_cycles}) reached, terminating")
            return {
                "action": "force_terminate",
                "reason": "Max cycles reached"
            }

        current_plan = prep_res["current_plan"]
        tasks = prep_res["tasks"]

        if not current_plan:
            return {"action": "truly_complete", "reason": "No active plan"}

        # Check actual completion status
        pending_tasks = [t for t in current_plan.tasks if tasks[t.id].status == "pending"]
        running_tasks = [t for t in current_plan.tasks if tasks[t.id].status == "running"]
        completed_tasks = [t for t in current_plan.tasks if tasks[t.id].status == "completed"]
        failed_tasks = [t for t in current_plan.tasks if tasks[t.id].status == "failed"]

        total_tasks = len(current_plan.tasks)

        # Truly complete: all tasks done
        if len(completed_tasks) + len(failed_tasks) == total_tasks:
            if len(failed_tasks) == 0 or len(completed_tasks) > len(failed_tasks):
                return {"action": "truly_complete", "reason": "All tasks completed"}
            else:
                return {"action": "truly_complete", "reason": "Plan failed but cannot continue"}

        # Has pending tasks that can run
        if pending_tasks and not running_tasks:
            return {"action": "continue_execution", "reason": f"{len(pending_tasks)} tasks ready"}

        # Has running tasks, wait
        if running_tasks:
            return {"action": "continue_execution", "reason": f"{len(running_tasks)} tasks running"}

        # Need reflection if tasks are stuck
        if pending_tasks and not running_tasks:
            return {"action": "needs_reflection", "reason": "Tasks may be blocked"}

        # Default: we're done
        return {"action": "truly_complete", "reason": "No actionable tasks"}

    async def post_async(self, shared, prep_res, exec_res):
        action = exec_res["action"]

        # Reset counter on true completion
        if action == "truly_complete":
            self.execution_count = 0
            shared["flow_completion_reason"] = exec_res["reason"]
        elif action == "force_terminate":  # HINZUGEFÜGT
            self.execution_count = 0
            shared["flow_completion_reason"] = f"Force terminated: {exec_res['reason']}"
            shared["force_terminated"] = True
            wprint(f"Flow force terminated: {exec_res['reason']}")

        return action

# ===== FLOW COMPOSITIONS =====
@with_progress_tracking
class TaskManagementFlow(AsyncFlow):
    """
    Enhanced Task-Management-Flow with LLMReasonerNode as strategic core.
    The flow now starts with strategic reasoning and delegates to specialized sub-systems.
    """

    def __init__(self, max_parallel_tasks: int = 3):
        # Create the strategic reasoning core (new primary node)
        self.llm_reasoner = LLMReasonerNode()

        # Create specialized sub-system nodes (now supporting nodes)
        self.planner_node = TaskPlannerNode()
        self.executor_node = TaskExecutorNode(max_parallel=max_parallel_tasks)
        self.sync_node = StateSyncNode()
        self.llm_tool_node = LLMToolNode()

        # Store references for the reasoner to access sub-systems
        # These will be injected into shared state during execution

        # === NEW HIERARCHICAL FLOW STRUCTURE ===

        # Primary flow: LLMReasonerNode is the main orchestrator
        # It makes strategic decisions and routes to appropriate sub-systems

        # The reasoner can internally call any of these sub-systems:
        # - LLMToolNode for direct tool usage
        # - TaskPlanner + TaskExecutor for complex project management
        # - Direct response for simple queries

        # Only one main connection: reasoner completes -> response generation
        self.llm_reasoner - "reasoner_complete" >> self.sync_node

        # Fallback connections for error handling
        self.llm_reasoner - "error" >> self.sync_node
        self.llm_reasoner - "timeout" >> self.sync_node

        # The old linear connections are removed - the reasoner now controls the flow internally

        super().__init__(start=self.llm_reasoner)

    async def run_async(self, shared):
        """Enhanced run with sub-system injection"""

        # Inject sub-system references into shared state so reasoner can access them
        shared["llm_tool_node_instance"] = self.llm_tool_node
        shared["task_planner_instance"] = self.planner_node
        shared["task_executor_instance"] = self.executor_node

        # Store tool registry access for the reasoner
        agent_instance = shared.get("agent_instance")
        if agent_instance:
            shared["tool_registry"] = agent_instance._tool_registry
            shared["tool_capabilities"] = agent_instance._tool_capabilities

        # Execute the flow with the reasoner as starting point
        return await super().run_async(shared)


@with_progress_tracking
class ResponseGenerationFlow(AsyncFlow):
    """Intelligente Antwortgenerierung basierend auf Task-Ergebnissen"""

    def __init__(self, tools=None):
        # Nodes für Response-Pipeline
        self.context_aggregator = ContextAggregatorNode()
        self.result_synthesizer = ResultSynthesizerNode()
        self.response_formatter = ResponseFormatterNode()
        self.quality_checker = ResponseQualityNode()
        self.final_processor = ResponseFinalProcessorNode()

        # === RESPONSE GENERATION PIPELINE ===

        # Context Aggregation -> Synthesis
        self.context_aggregator - "context_ready" >> self.result_synthesizer
        self.context_aggregator - "no_context" >> self.response_formatter  # Fallback

        # Synthesis -> Formatting
        self.result_synthesizer - "synthesized" >> self.response_formatter
        self.result_synthesizer - "synthesis_failed" >> self.response_formatter

        # Formatting -> Quality Check
        self.response_formatter - "formatted" >> self.quality_checker
        self.response_formatter - "format_failed" >> self.final_processor  # Skip quality check

        # Quality Check -> Final Processing oder Retry
        self.quality_checker - "quality_good" >> self.final_processor
        self.quality_checker - "quality_poor" >> self.result_synthesizer  # Retry synthesis
        self.quality_checker - "quality_acceptable" >> self.final_processor

        super().__init__(start=self.context_aggregator)


# Neue spezialisierte Nodes für Response-Generation

@with_progress_tracking
class ContextAggregatorNode(AsyncNode):
    """Vereinfachte Context-Aggregation über UnifiedContextManager"""

    async def prep_async(self, shared):
        """Simplified preparation - delegate to UnifiedContextManager"""
        return {
            "context_manager": shared.get("context_manager"),
            "session_id": shared.get("session_id", "default"),
            "original_query": shared.get("current_query", ""),
            "tasks": shared.get("tasks", {}),
            "current_plan": shared.get("current_plan"),
            "world_model": shared.get("world_model", {}),
            "results": shared.get("results", {})
        }

    async def exec_async(self, prep_res):
        """VEREINFACHT: Get aggregated context from UnifiedContextManager"""

        context_manager = prep_res.get("context_manager")
        session_id = prep_res.get("session_id", "default")
        query = prep_res.get("original_query", "")

        if not context_manager:
            # Fallback: Create basic aggregated context
            return self._create_fallback_context(prep_res)

        try:
            #Get unified context from context manager
            unified_context = await context_manager.build_unified_context(session_id, query, "full")

            # Transform to expected aggregated_context format for compatibility
            aggregated_context = {
                "original_query": query,
                "successful_results": self._extract_successful_results(unified_context),
                "failed_attempts": self._extract_failed_attempts(prep_res["tasks"]),
                "key_discoveries": self._extract_key_discoveries(unified_context),
                "adaptation_summary": self._extract_adaptation_summary(prep_res),
                "confidence_scores": self._calculate_confidence_scores(unified_context),
                "unified_context": unified_context,  # Include full unified context
                "context_source": "unified_context_manager"
            }

            return aggregated_context

        except Exception as e:
            eprint(f"UnifiedContextManager aggregation failed: {e}")
            return self._create_fallback_context(prep_res)

    def _extract_successful_results(self, unified_context: dict[str, Any]) -> dict[str, Any]:
        """Extract successful results from unified context"""
        successful_results = {}

        try:
            # Get from variables context
            variables = unified_context.get("variables", {})
            recent_results = variables.get("recent_results", [])

            for result in recent_results:
                if result.get("success"):
                    task_id = result.get("task_id", f"result_{len(successful_results)}")
                    successful_results[task_id] = {
                        "task_description": f"Task {task_id}",
                        "task_type": "unified_context_result",
                        "result": result.get("preview", ""),
                        "metadata": {
                            "timestamp": result.get("timestamp"),
                            "source": "unified_context"
                        }
                    }

            # Also check execution state for completions
            execution_state = unified_context.get("execution_state", {})
            recent_completions = execution_state.get("recent_completions", [])

            for completion in recent_completions:
                task_id = completion.get("id", f"completion_{len(successful_results)}")
                successful_results[task_id] = {
                    "task_description": completion.get("description", "Completed task"),
                    "task_type": "execution_completion",
                    "result": f"Task completed at {completion.get('completed_at', 'unknown time')}",
                    "metadata": {
                        "completion_time": completion.get("completed_at"),
                        "source": "execution_state"
                    }
                }

            return successful_results

        except Exception as e:
            eprint(f"Error extracting successful results: {e}")
            return {}

    def _extract_failed_attempts(self, tasks: dict) -> dict[str, Any]:
        """Extract failed attempts from tasks (existing functionality)"""
        failed_attempts = {}

        try:
            for task_id, task in tasks.items():
                if task.status == "failed":
                    failed_attempts[task_id] = {
                        "description": task.description,
                        "error": task.error,
                        "retry_count": task.retry_count
                    }
            return failed_attempts
        except:
            return {}

    def _extract_key_discoveries(self, unified_context: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract key discoveries from unified context"""
        discoveries = []

        try:
            # Extract from relevant facts
            relevant_facts = unified_context.get("relevant_facts", [])
            for key, value in relevant_facts[:3]:  # Top 3 facts
                discoveries.append({
                    "discovery": f"Fact discovered: {key}",
                    "confidence": 0.8,  # Default confidence for facts
                    "result": value
                })

            # Extract from successful results
            variables = unified_context.get("variables", {})
            recent_results = variables.get("recent_results", [])

            for result in recent_results[:2]:  # Top 2 results
                if result.get("success"):
                    discoveries.append({
                        "discovery": f"Task result: {result.get('task_id', 'unknown')}",
                        "confidence": 0.7,
                        "result": result.get("preview", "")
                    })

            return discoveries

        except Exception as e:
            eprint(f"Error extracting discoveries: {e}")
            return []

    def _extract_adaptation_summary(self, prep_res: dict) -> str:
        """Extract adaptation summary"""
        try:
            current_plan = prep_res.get("current_plan")
            if current_plan and hasattr(current_plan, 'metadata'):
                adaptations = current_plan.metadata.get("adaptations", 0)
                if adaptations > 0:
                    return f"Plan was adapted {adaptations} times to handle unexpected results."
            return ""
        except:
            return ""

    def _calculate_confidence_scores(self, unified_context: dict[str, Any]) -> dict[str, float]:
        """Calculate confidence scores based on unified context"""
        try:
            scores = {"overall": 0.5}

            # Base confidence on available data
            chat_history = unified_context.get("chat_history", [])
            if chat_history:
                scores["conversation_context"] = min(len(chat_history) / 10, 1.0)

            variables = unified_context.get("variables", {})
            recent_results = variables.get("recent_results", [])
            successful_results = [r for r in recent_results if r.get("success")]

            if recent_results:
                scores["execution_results"] = len(successful_results) / len(recent_results)

            # Calculate overall confidence
            scores["overall"] = sum(scores.values()) / len(scores)

            return scores

        except:
            return {"overall": 0.3}

    def _create_fallback_context(self, prep_res: dict) -> dict[str, Any]:
        """Create fallback context when UnifiedContextManager is unavailable"""
        return {
            "original_query": prep_res.get("original_query", ""),
            "successful_results": {},
            "failed_attempts": self._extract_failed_attempts(prep_res.get("tasks", {})),
            "key_discoveries": [],
            "adaptation_summary": "Fallback context - UnifiedContextManager unavailable",
            "confidence_scores": {"overall": 0.2},
            "context_source": "fallback"
        }

    async def post_async(self, shared, prep_res, exec_res):
        """Store aggregated context for downstream nodes"""
        shared["aggregated_context"] = exec_res

        #Also store unified context reference for other nodes
        if "unified_context" in exec_res:
            shared["unified_context"] = exec_res["unified_context"]

        if exec_res.get("successful_results") or exec_res.get("key_discoveries"):
            return "context_ready"
        else:
            return "no_context"

@with_progress_tracking
class ResultSynthesizerNode(AsyncNode):
    """Synthetisiere finale Antwort aus allen Ergebnissen"""

    async def prep_async(self, shared):
        return {
            "aggregated_context": shared.get("aggregated_context", {}),
            "fast_llm_model": shared.get("fast_llm_model"),
            "complex_llm_model": shared.get("complex_llm_model"),
            "agent_instance": shared.get("agent_instance")
        }

    async def exec_async(self, prep_res):
        if not LITELLM_AVAILABLE:
            return await self._fallback_synthesis(prep_res)

        context = prep_res["aggregated_context"]
        persona = (prep_res['agent_instance'].amd.persona.to_system_prompt_addition() if not prep_res['agent_instance'].amd.persona.should_post_process() else '') if prep_res['agent_instance'].amd.persona else None
        prompt = f"""
Du bist ein Experte für Informationssynthese. Erstelle eine umfassende, hilfreiche Antwort basierend auf den gesammelten Ergebnissen.

## Ursprüngliche Anfrage
{context.get('original_query', '')}

## Erfolgreiche Ergebnisse
{self._format_successful_results(context.get('successful_results', {}))}

## Wichtige Entdeckungen
{self._format_key_discoveries(context.get('key_discoveries', []))}

## Plan-Adaptationen
{context.get('adaptation_summary', 'No adaptations were needed.')}

## Fehlgeschlagene Versuche
{self._format_failed_attempts(context.get('failed_attempts', {}))}

{persona}

## Anweisungen
1. Gib eine direkte, hilfreiche Antwort auf die ursprüngliche Anfrage
2. Integriere alle relevanten gefundenen Informationen
3. Erkläre kurz den Prozess, falls Adaptationen nötig waren
4. Sei ehrlich über Limitationen oder fehlende Informationen
5. Strukturiere die Antwort logisch und lesbar

Erstelle eine finale Antwort:"""

        try:
            # Verwende complex model für finale Synthesis
            model_to_use = prep_res.get("complex_llm_model", "openrouter/openai/gpt-4o")
            agent_instance = prep_res["agent_instance"]
            synthesized_response = await agent_instance.a_run_llm_completion(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500,
                node_name="ResultSynthesizerNode", task_id="response_synthesis"
            )

            return {
                "synthesized_response": synthesized_response,
                "synthesis_method": "llm",
                "model_used": model_to_use,
                "confidence": self._estimate_synthesis_confidence(context)
            }

        except Exception as e:
            eprint(f"LLM synthesis failed: {e}")
            return await self._fallback_synthesis(prep_res)

    def _format_successful_results(self, results: dict) -> str:
        formatted = []
        for _task_id, result_info in results.items():
            formatted.append(f"- {result_info['task_description']}: {str(result_info['result'])[:20000]}...")
        return "\n".join(formatted) if formatted else "No successful results to report."

    def _format_key_discoveries(self, discoveries: list) -> str:
        formatted = []
        for discovery in discoveries:
            confidence = discovery.get('confidence', 0.0)
            formatted.append(f"- {discovery['discovery']} (Confidence: {confidence:.2f})")
        return "\n".join(formatted) if formatted else "No key discoveries."

    def _format_failed_attempts(self, failed: dict) -> str:
        if not failed:
            return "No significant failures."
        formatted = [f"- {info['description']}: {info['error']}" for info in failed.values()]
        return "\n".join(formatted)

    async def _fallback_synthesis(self, prep_res) -> dict:
        """Fallback synthesis ohne LLM"""
        context = prep_res["aggregated_context"]

        # Einfache Template-basierte Synthese
        response_parts = []

        if context.get("key_discoveries"):
            response_parts.append("Based on my analysis, I found:")
            for discovery in context["key_discoveries"][:3]:  # Top 3
                response_parts.append(f"- {discovery['discovery']}")

        if context.get("successful_results"):
            response_parts.append("\nDetailed results:")
            for _task_id, result in list(context["successful_results"].items())[:2]:  # Top 2
                response_parts.append(f"- {result['task_description']}: {str(result['result'])[:150]}")

        if context.get("adaptation_summary"):
            response_parts.append(f"\n{context['adaptation_summary']}")

        fallback_response = "\n".join(
            response_parts) if response_parts else "I was unable to complete the requested task effectively."

        return {
            "synthesized_response": fallback_response,
            "synthesis_method": "fallback",
            "confidence": 0.3
        }

    def _estimate_synthesis_confidence(self, context: dict) -> float:
        """Schätze Confidence der Synthese"""
        confidence = 0.5  # Base confidence

        # Boost für erfolgreiche Ergebnisse
        successful_count = len(context.get("successful_results", {}))
        confidence += min(successful_count * 0.15, 0.3)

        # Boost für key discoveries mit hoher confidence
        for discovery in context.get("key_discoveries", []):
            discovery_conf = discovery.get("confidence", 0.0)
            confidence += discovery_conf * 0.1

        # Penalty für viele fehlgeschlagene Versuche
        failed_count = len(context.get("failed_attempts", {}))
        confidence -= min(failed_count * 0.1, 0.2)

        return max(0.1, min(1.0, confidence))

    async def post_async(self, shared, prep_res, exec_res):
        shared["synthesized_response"] = exec_res
        if exec_res.get("synthesized_response"):
            return "synthesized"
        else:
            return "synthesis_failed"


@with_progress_tracking
class ResponseFormatterNode(AsyncNode):
    """Formatiere finale Antwort für Benutzer"""

    async def prep_async(self, shared):
        return {
            "synthesized_response": shared.get("synthesized_response", {}),
            "original_query": shared.get("current_query", ""),
            "user_preferences": shared.get("user_preferences", {})
        }

    async def exec_async(self, prep_res):
        synthesis_data = prep_res["synthesized_response"]
        raw_response = synthesis_data.get("synthesized_response", "")

        if not raw_response:
            return {
                "formatted_response": "I apologize, but I was unable to generate a meaningful response to your query."}

        # Basis-Formatierung
        formatted_response = raw_response.strip()

        # Füge Metadaten hinzu falls gewünscht (für debugging/transparency)
        confidence = synthesis_data.get("confidence", 0.0)
        if confidence < 0.4:
            formatted_response += "\n\n*Note: This response has low confidence due to limited information.*"

        adaptation_note = ""
        synthesis_method = synthesis_data.get("synthesis_method", "unknown")
        if synthesis_method == "fallback":
            adaptation_note = "\n\n*Note: Response generated with limited processing capabilities.*"

        return {
            "formatted_response": formatted_response + adaptation_note,
            "confidence": confidence,
            "metadata": {
                "synthesis_method": synthesis_method,
                "response_length": len(formatted_response)
            }
        }

    async def post_async(self, shared, prep_res, exec_res):
        shared["formatted_response"] = exec_res
        return "formatted"

@with_progress_tracking
class ResponseQualityNode(AsyncNode):
    """Prüfe Qualität der generierten Antwort"""

    async def prep_async(self, shared):
        return {
            "formatted_response": shared.get("formatted_response", {}),
            "original_query": shared.get("current_query", ""),
            "format_config": self._get_format_config(shared),
            "fast_llm_model": shared.get("fast_llm_model"),
            "persona_config": shared.get("persona_config"),
            "agent_instance": shared.get("agent_instance"),
        }

    def _get_format_config(self, shared) -> FormatConfig | None:
        """Extrahiere Format-Konfiguration"""
        persona = shared.get("persona_config")
        if persona and hasattr(persona, 'format_config'):
            return persona.format_config
        return None

    async def exec_async(self, prep_res):
        response_data = prep_res["formatted_response"]
        response_text = response_data.get("formatted_response", "")
        original_query = prep_res["original_query"]
        format_config = prep_res["format_config"]

        # Basis-Qualitätsprüfung
        base_quality = self._heuristic_quality_check(response_text, original_query)

        # Format-spezifische Bewertung
        format_quality = await self._evaluate_format_adherence(response_text, format_config)

        # Längen-spezifische Bewertung
        length_quality = self._evaluate_length_adherence(response_text, format_config)

        # LLM-basierte Gesamtbewertung
        llm_quality = 0.5
        if LITELLM_AVAILABLE and len(response_text) > 500:
            llm_quality = await self._llm_format_quality_check(
                response_text, original_query, format_config, prep_res
            )

        # Gewichtete Gesamtbewertung
        total_quality = (
            base_quality * 0.3 +
            format_quality * 0.3 +
            length_quality * 0.2 +
            llm_quality * 0.2
        )

        quality_details = {
            "total_score": total_quality,
            "base_quality": base_quality,
            "format_adherence": format_quality,
            "length_adherence": length_quality,
            "llm_assessment": llm_quality,
            "format_config_used": format_config is not None
        }

        return {
            "quality_score": total_quality,
            "quality_assessment": self._score_to_assessment(total_quality),
            "quality_details": quality_details,
            "suggestions": self._generate_format_quality_suggestions(
                total_quality, response_text, format_config, quality_details
            )
        }

    async def _evaluate_format_adherence(self, response: str, format_config: FormatConfig | None) -> float:
        """Bewerte Format-Einhaltung"""
        if not format_config:
            return 0.8  # Neutral wenn kein Format vorgegeben

        format_type = format_config.response_format
        score = 0.5

        # Format-spezifische Checks
        if format_type == ResponseFormat.WITH_TABLES:
            if '|' in response or 'Table:' in response or '| ' in response:
                score += 0.4

        elif format_type == ResponseFormat.WITH_BULLET_POINTS:
            bullet_count = response.count('•') + response.count('-') + response.count('*')
            if bullet_count >= 2:
                score += 0.4
            elif bullet_count >= 1:
                score += 0.2

        elif format_type == ResponseFormat.WITH_LISTS:
            list_patterns = ['1.', '2.', '3.', 'a)', 'b)', 'c)']
            list_score = sum(1 for pattern in list_patterns if pattern in response)
            score += min(0.4, list_score * 0.1)

        elif format_type == ResponseFormat.MD_TEXT:
            md_elements = ['#', '**', '*', '`', '```', '[', ']', '(', ')']
            md_score = sum(1 for element in md_elements if element in response)
            score += min(0.4, md_score * 0.05)

        elif format_type == ResponseFormat.YAML_TEXT:
            if response.strip().startswith(('```yaml', '---')) or ': ' in response:
                score += 0.4

        elif format_type == ResponseFormat.JSON_TEXT:
            if response.strip().startswith(('{', '[')):
                try:
                    json.loads(response)
                    score += 0.4
                except:
                    score += 0.1  # Partial credit for JSON-like structure

        elif format_type == ResponseFormat.TEXT_ONLY:
            # Penalize if formatting elements are present
            format_elements = ['#', '*', '|', '```', '1.', '•', '-']
            format_count = sum(1 for element in format_elements if element in response)
            score += max(0.1, 0.5 - format_count * 0.05)

        elif format_type == ResponseFormat.PSEUDO_CODE:
            code_indicators = ['if ', 'for ', 'while ', 'def ', 'return ', 'function', 'BEGIN', 'END']
            code_score = sum(1 for indicator in code_indicators if indicator in response)
            score += min(0.4, code_score * 0.1)

        return max(0.0, min(1.0, score))

    def _evaluate_length_adherence(self, response: str, format_config: FormatConfig | None) -> float:
        """Bewerte Längen-Einhaltung"""
        if not format_config:
            return 0.8

        word_count = len(response.split())
        min_words, max_words = format_config.get_expected_word_range()

        if min_words <= word_count <= max_words:
            return 1.0
        elif word_count < min_words:
            # Zu kurz - sanfte Bestrafung
            ratio = word_count / min_words
            return max(0.3, ratio * 0.8)
        else:  # word_count > max_words
            # Zu lang - weniger Bestrafung als zu kurz
            excess_ratio = (word_count - max_words) / max_words
            return max(0.4, 1.0 - excess_ratio * 0.3)

    async def _llm_format_quality_check(
        self,
        response: str,
        query: str,
        format_config: FormatConfig | None,
        prep_res: dict
    ) -> float:
        """LLM-basierte Format- und Qualitätsbewertung"""
        if not format_config:
            return await self._standard_llm_quality_check(response, query, prep_res)

        format_desc = format_config.get_format_instructions()
        length_desc = format_config.get_length_instructions()

        prompt = f"""
Bewerte diese Antwort auf einer Skala von 0.0 bis 1.0 basierend auf Format-Einhaltung und Qualität:

Benutzer-Anfrage: {query}

Antwort: {response}

Erwartetes Format: {format_desc}
Erwartete Länge: {length_desc}

Bewertungskriterien:
1. Format-Einhaltung (40%): Entspricht die Antwort dem geforderten Format?
2. Längen-Angemessenheit (25%): Ist die Länge angemessen?
3. Inhaltliche Qualität (25%): Beantwortet die Anfrage vollständig?
4. Lesbarkeit und Struktur (10%): Ist die Antwort gut strukturiert?

Antworte nur mit einer Zahl zwischen 0.0 und 1.0:"""

        try:
            model_to_use = prep_res.get("fast_llm_model", "openrouter/anthropic/claude-3-haiku")
            agent_instance = prep_res["agent_instance"]
            score_text = (await agent_instance.a_run_llm_completion(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10,
                node_name="QualityAssessmentNode", task_id="format_quality_assessment"
            )).strip()

            return float(score_text)

        except Exception as e:
            wprint(f"LLM format quality check failed: {e}")
            return 0.6  # Neutral fallback

    def _generate_format_quality_suggestions(
        self,
        score: float,
        response: str,
        format_config: FormatConfig | None,
        quality_details: dict
    ) -> list[str]:
        """Generiere Format-spezifische Verbesserungsvorschläge"""
        suggestions = []

        if not format_config:
            return ["Consider defining a specific response format for better consistency"]

        # Format-spezifische Vorschläge
        if quality_details["format_adherence"] < 0.6:
            format_type = format_config.response_format

            if format_type == ResponseFormat.WITH_TABLES:
                suggestions.append("Add tables using markdown format (| Column | Column |)")
            elif format_type == ResponseFormat.WITH_BULLET_POINTS:
                suggestions.append("Use bullet points (•, -, *) to structure information")
            elif format_type == ResponseFormat.MD_TEXT:
                suggestions.append("Use markdown formatting (headers, bold, code blocks)")
            elif format_type == ResponseFormat.YAML_TEXT:
                suggestions.append("Format response as valid YAML structure")
            elif format_type == ResponseFormat.JSON_TEXT:
                suggestions.append("Format response as valid JSON")

        # Längen-spezifische Vorschläge
        if quality_details["length_adherence"] < 0.6:
            word_count = len(response.split())
            min_words, max_words = format_config.get_expected_word_range()

            if word_count < min_words:
                suggestions.append(f"Response too short ({word_count} words). Aim for {min_words}-{max_words} words")
            else:
                suggestions.append(f"Response too long ({word_count} words). Aim for {min_words}-{max_words} words")

        # Qualitäts-spezifische Vorschläge
        if score < 0.5:
            suggestions.append("Overall quality needs improvement - consider regenerating")
        elif score < 0.7:
            suggestions.append("Good response but could be enhanced with better format adherence")

        return suggestions

    async def _standard_llm_quality_check(self, response: str, query: str, prep_res: dict) -> float:
        """Standard LLM-Qualitätsprüfung ohne Format-Fokus"""
        # Bestehende Implementierung beibehalten
        return await self._llm_quality_check(response, query, prep_res)

    def _heuristic_quality_check(self, response: str, query: str) -> float:
        """Heuristische Qualitätsprüfung"""
        score = 0.5  # Base score

        # Length check
        if len(response) < 50:
            score -= 0.3
        elif len(response) > 100:
            score += 0.2

        # Query term coverage
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        coverage = len(query_terms.intersection(response_terms)) / max(len(query_terms), 1)
        score += coverage * 0.3

        # Structure indicators
        if any(indicator in response for indicator in [":", "-", "1.", "•"]):
            score += 0.1  # Structured response bonus

        return max(0.0, min(1.0, score))

    async def _llm_quality_check(self, response: str, query: str, prep_res: dict) -> float:
        """LLM-basierte Qualitätsprüfung"""
        try:
            prompt = f"""
Rate the quality of this response to the user's query on a scale of 0.0 to 1.0.

User Query: {query}

Response: {response}

Consider:
- Relevance to the query
- Completeness of information
- Clarity and readability
- Accuracy (if verifiable)

Respond with just a number between 0.0 and 1.0:"""

            model_to_use = prep_res.get("fast_llm_model", "openrouter/anthropic/claude-3-haiku")
            agent_instance = prep_res["agent_instance"]
            score_text = (await agent_instance.a_run_llm_completion(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10,
                node_name="QualityAssessmentNode", task_id="quality_assessment"
            )).strip()

            return float(score_text)

        except:
            return 0.5  # Fallback score

    def _score_to_assessment(self, score: float) -> str:
        if score >= 0.8:
            return "quality_good"
        elif score >= 0.5:
            return "quality_acceptable"
        else:
            return "quality_poor"

    async def post_async(self, shared, prep_res, exec_res):
        shared["quality_assessment"] = exec_res
        return exec_res["quality_assessment"]


@with_progress_tracking
class ResponseFinalProcessorNode(AsyncNode):
    """Finale Verarbeitung mit Persona-System"""

    async def prep_async(self, shared):
        return {
            "formatted_response": shared.get("formatted_response", {}),
            "quality_assessment": shared.get("quality_assessment", {}),
            "conversation_history": shared.get("conversation_history", []),
            "persona": shared.get("persona_config"),
            "fast_llm_model": shared.get("fast_llm_model"),
            "use_fast_response": shared.get("use_fast_response", True),
            "agent_instance": shared.get("agent_instance"),
        }

    async def exec_async(self, prep_res):
        response_data = prep_res["formatted_response"]
        raw_response = response_data.get("formatted_response", "I apologize, but I couldn't generate a response.")

        # Persona-basierte Anpassung
        if prep_res.get("persona") and LITELLM_AVAILABLE:
            final_response = await self._apply_persona_style(raw_response, prep_res)
        else:
            final_response = raw_response

        # Finale Metadaten
        processing_metadata = {
            "response_confidence": response_data.get("confidence", 0.0),
            "quality_score": prep_res.get("quality_assessment", {}).get("quality_score", 0.0),
            "processing_timestamp": datetime.now().isoformat(),
            "response_length": len(final_response),
            "persona_applied": prep_res.get("persona") is not None
        }

        return {
            "final_response": final_response,
            "metadata": processing_metadata,
            "status": "completed"
        }

    async def _apply_persona_style(self, response: str, prep_res: dict) -> str:
        """Optimized persona styling mit Konfiguration"""
        persona = prep_res["persona"]

        # Nur anwenden wenn post-processing konfiguriert
        if not persona.should_post_process():
            return response

        # Je nach Integration Level unterschiedliche Prompts
        if persona.integration_level == "light":
            style_prompt = f"Make this {persona.tone} and {persona.style}: {response}"
            max_tokens = 400
        elif persona.integration_level == "medium":
            style_prompt = f"""
    Apply {persona.name} persona (style: {persona.style}, tone: {persona.tone}) to:
    {response}

    Keep the same information, adjust presentation:"""
            max_tokens = 600
        else:  # heavy
            style_prompt = f"""
Completely transform as {persona.name}:
Style: {persona.style}, Tone: {persona.tone}
Traits: {', '.join(persona.personality_traits)}
Instructions: {persona.custom_instructions}

Original: {response}

As {persona.name}:"""
            max_tokens = 1000

        try:
            model_to_use = prep_res.get("fast_llm_model", "openrouter/anthropic/claude-3-haiku")
            agent_instance = prep_res["agent_instance"]
            if prep_res.get("use_fast_response", True):
                response = await agent_instance.a_run_llm_completion(
                    model=model_to_use,
                    messages=[{"role": "user", "content": style_prompt}],
                    temperature=0.5,
                    max_tokens=max_tokens, node_name="PersonaStylingNode", task_id="persona_styling_fast"
                )
            else:
                response = await agent_instance.a_run_llm_completion(
                    model=model_to_use,
                    messages=[{"role": "user", "content": style_prompt}],
                    temperature=0.6,
                    max_tokens=max_tokens + 200, node_name="PersonaStylingNode", task_id="persona_styling_ritch"
                )

            return response.strip()

        except Exception as e:
            wprint(f"Persona styling failed: {e}")
            return response

    async def post_async(self, shared, prep_res, exec_res):
        shared["current_response"] = exec_res["final_response"]
        shared["response_metadata"] = exec_res["metadata"]
        return "response_ready"


@with_progress_tracking
class LLMReasonerNode(AsyncNode):
    """
    Enhanced strategic reasoning core with outline-driven execution,
    context management, auto-recovery, and intensive variable system integration.
    """

    def __init__(self, max_reasoning_loops: int = 12, **kwargs):
        super().__init__(**kwargs)
        self.max_reasoning_loops = max_reasoning_loops
        self.reasoning_context = []
        self.internal_task_stack = []
        self.meta_tools_registry = {}
        self.current_loop_count = 0
        self.current_reasoning_count = 0
        self.agent_instance: FlowAgent = None

        # Enhanced tracking systems
        self.outline = None
        self.current_outline_step = 0
        self.step_completion_tracking = {}
        self.loop_detection_memory = []
        self.context_summary_threshold = 15
        self.max_context_size = 30
        self.performance_metrics = {
                "loop_times": [],
                "progress_loops": 0,
                "total_loops": 0
            }
        self.auto_recovery_attempts = 0
        self.max_auto_recovery = 8
        self.variable_manager = None

        # Anti-loop mechanisms
        self.last_action_signatures = []
        self.step_enforcement_active = True
        self.mandatory_progress_check = True

    async def prep_async(self, shared):
        """Enhanced initialization with variable system integration"""
        # Reset for new execution
        self.reasoning_context = []
        self.internal_task_stack = []
        self.current_loop_count = 0
        self.current_reasoning_count = 0
        self.outline = None
        self.current_outline_step = 0
        self.step_completion_tracking = {}
        self.loop_detection_memory = []
        self.performance_metrics = {
            "loop_times": [],
            "progress_loops": 0,
            "total_loops": 0
        }
        self.auto_recovery_attempts = 0
        self.last_action_signatures = []

        self.agent_instance = shared.get("agent_instance")

        # Enhanced variable manager integration
        self.variable_manager = shared.get("variable_manager", self.agent_instance.variable_manager)
        context_manager = shared.get("context_manager")

        if self.variable_manager:
            # Store reasoning session context
            session_context = {
                "session_id": shared.get("session_id", "default"),
                "start_time": datetime.now().isoformat(),
                "query": shared.get("current_query", ""),
                "reasoning_mode": "outline_driven"
            }
            self.variable_manager.set("reasoning.current_session", session_context)
            # Load previous successful patterns from variables
            self._load_historical_patterns()

        #Build comprehensive system context via UnifiedContextManager
        system_context = await self._build_enhanced_system_context_unified(shared, context_manager)

        return {
            "original_query": shared.get("current_query", ""),
            "session_id": shared.get("session_id", "default"),
            "agent_instance": shared.get("agent_instance"),
            "variable_manager": self.variable_manager,
            "context_manager": context_manager,  #Context Manager Reference
            "system_context": system_context,
            "available_tools": shared.get("available_tools", []),
            "tool_capabilities": shared.get("tool_capabilities", {}),
            "fast_llm_model": shared.get("fast_llm_model"),
            "complex_llm_model": shared.get("complex_llm_model"),
            "progress_tracker": shared.get("progress_tracker"),
            "formatted_context": shared.get("formatted_context", {}),
            "historical_context": await self._get_historical_context_unified(context_manager, shared.get("session_id")),
            "capabilities_summary": shared.get("capabilities_summary", ""),
            # Sub-system references
            "llm_tool_node": shared.get("llm_tool_node_instance"),
            "task_planner": shared.get("task_planner_instance"),
            "task_executor": shared.get("task_executor_instance"),
        }

    async def exec_async(self, prep_res):
        """Enhanced main reasoning loop with outline-driven execution"""
        if not LITELLM_AVAILABLE:
            return await self._fallback_direct_response(prep_res)

        original_query = prep_res["original_query"]
        agent_instance = prep_res["agent_instance"]
        progress_tracker = prep_res.get("progress_tracker")

        # Initialize enhanced reasoning context
        await self._initialize_reasoning_session(prep_res, original_query)

        # STEP 1: MANDATORY OUTLINE CREATION
        if not self.outline:
            outline_result = await self._create_initial_outline(prep_res)
            if not outline_result:
                return await self._create_error_response(original_query, "Failed to create initial outline")

        final_result = None
        consecutive_no_progress = 0
        max_no_progress = 3

        # Enhanced main reasoning loop with strict progress tracking
        while self.current_reasoning_count < self.max_reasoning_loops:
            self.current_loop_count += 1
            loop_start_time = time.time()

            # Check for infinite loops
            if self._detect_infinite_loop():
                await self._trigger_auto_recovery(prep_res)
                if self.auto_recovery_attempts >= self.max_auto_recovery:
                    break

            # Auto-context management
            await self._manage_context_size()

            # Progress tracking
            if progress_tracker:
                await progress_tracker.emit_event(ProgressEvent(
                    event_type="reasoning_loop",
                    timestamp=time.time(),
                    node_name="LLMReasonerNode",
                    status=NodeStatus.RUNNING,
                    metadata={
                        "loop_number": self.current_loop_count,
                        "outline_step": self.current_outline_step,
                        "outline_total": len(self.outline.get("steps", [])) if self.outline else 0,
                        "context_size": len(self.reasoning_context),
                        "task_stack_size": len(self.internal_task_stack),
                        "auto_recovery_attempts": self.auto_recovery_attempts,
                        "performance_metrics": self.performance_metrics
                    }
                ))

            try:
                # Build enhanced reasoning prompt with outline context
                reasoning_prompt = await self._build_outline_driven_prompt(prep_res)

                # Force progress check if needed
                if self.mandatory_progress_check and consecutive_no_progress >= 2:
                    reasoning_prompt += "\n\n**MANDATORY**: You must either complete current outline step or move to next step. No more analysis without action!"

                # LLM reasoning call
                model_to_use = prep_res.get("complex_llm_model", "openrouter/openai/gpt-4o")

                llm_response = await agent_instance.a_run_llm_completion(
                    model=model_to_use,
                    messages=[{"role": "user", "content": reasoning_prompt}],
                    temperature=0.2,  # Lower temperature for more focused execution
                    # max_tokens=3072,
                    node_name="LLMReasonerNode",
                    stop="<immediate_context>",
                    task_id=f"reasoning_loop_{self.current_loop_count}_step_{self.current_outline_step}"
                )

                # Add LLM response to context
                self.reasoning_context.append({
                    "type": "reasoning",
                    "content": llm_response,
                    "loop": self.current_loop_count,
                    "outline_step": self.current_outline_step,
                    "timestamp": datetime.now().isoformat()
                })

                # Parse and execute meta-tool calls with enhanced tracking
                progress_made = await self._parse_and_execute_meta_tools(llm_response, prep_res)

                action_taken = progress_made.get("action_taken", False)
                actual_progress = progress_made.get("progress_made", False)

                # Update performance with correct progress indication
                self._update_performance_metrics(loop_start_time, actual_progress)

                if not action_taken:
                    self.current_reasoning_count += 1
                    if self.current_outline_step > len(self.outline.get("steps", [])):
                        progress_made["final_result"] = llm_response
                        rprint("Final result reached forced by outline step count")
                    if self.current_outline_step < len(self.outline.get("steps", [])) and self.outline.get("steps", [])[self.current_outline_step].get("is_final", False):
                        progress_made["final_result"] = llm_response
                        rprint("Final result reached forced by outline step count final step")
                else:
                    self.current_reasoning_count -= 1

                # Check for final result
                if progress_made.get("final_result"):
                    final_result = progress_made["final_result"]
                    await self._finalize_reasoning_session(prep_res, final_result)
                    break

                # Progress monitoring
                if progress_made.get("action_taken"):
                    consecutive_no_progress = 0
                    self._update_performance_metrics(loop_start_time, True)
                else:
                    consecutive_no_progress += 1
                    self._update_performance_metrics(loop_start_time, False)

                # Check outline completion
                if self.outline and self.current_outline_step >= len(self.outline.get("steps", []))+self.max_reasoning_loops:
                    # All outline steps completed, force final response
                    final_result = await self._create_outline_completion_response(prep_res)
                    break

                # Emergency break for excessive no-progress
                if consecutive_no_progress >= max_no_progress:
                    await self._trigger_auto_recovery(prep_res)

            except Exception as e:
                await self._handle_reasoning_error(e, prep_res, progress_tracker)
                import traceback
                print(traceback.format_exc())
                if self.auto_recovery_attempts >= self.max_auto_recovery:
                    final_result = await self._create_error_response(original_query, str(e))
                    break


        # If no final result after max loops, create a comprehensive summary
        if not final_result:
            final_result = await self._create_enhanced_timeout_response(original_query, prep_res)

        return {
            "final_result": final_result,
            "reasoning_loops": self.current_loop_count,
            "reasoning_context": self.reasoning_context.copy(),
            "internal_task_stack": self.internal_task_stack.copy(),
            "outline": self.outline,
            "outline_completion": self.current_outline_step,
            "performance_metrics": self.performance_metrics,
            "auto_recovery_attempts": self.auto_recovery_attempts
        }

    async def _build_enhanced_system_context_unified(self, shared, context_manager) -> str:
        """Build comprehensive system context mit UnifiedContextManager"""
        context_parts = []

        # Enhanced agent capabilities
        available_tools = shared.get("available_tools", [])
        if available_tools:
            context_parts.append(f"Available external tools: {', '.join(available_tools)}")

        #Context Manager Status
        if context_manager:
            session_stats = context_manager.get_session_statistics()
            context_parts.append(f"Context System: Advanced with {session_stats['total_sessions']} active sessions")
            context_parts.append(f"Cache Status: {session_stats['cache_entries']} cached contexts")

        # Variable system context
        if self.variable_manager:
            var_info = self.variable_manager.get_scope_info()
            context_parts.append(f"Variable System: {len(var_info)} scopes available")

            # Recent results availability
            results_count = len(self.variable_manager.get("results", {}))
            if results_count:
                context_parts.append(f"Previous results: {results_count} task results available")

        #Enhanced system state mit Context-Awareness
        session_id = shared.get("session_id", "default")
        if context_manager and session_id in context_manager.session_managers:
            session = context_manager.session_managers[session_id]
            if hasattr(session, 'history'):
                context_parts.append(f"Session History: {len(session.history)} conversation entries available")
            elif isinstance(session, dict) and 'history' in session:
                context_parts.append(f"Session History: {len(session['history'])} conversation entries (fallback mode)")

        # System state with enhanced details
        tasks = shared.get("tasks", {})
        if tasks:
            active_tasks = len([t for t in tasks.values() if t.status == "running"])
            completed_tasks = len([t for t in tasks.values() if t.status == "completed"])
            context_parts.append(f"Execution state: {active_tasks} active, {completed_tasks} completed tasks")

        # Performance history
        if hasattr(self, 'historical_successful_patterns'):
            context_parts.append(
                f"Historical patterns: {len(self.historical_successful_patterns)} successful patterns loaded")

        return "\n".join(context_parts) if context_parts else "Basic system context available"

    async def _get_historical_context_unified(self, context_manager, session_id: str) -> str:
        """Get historical context from UnifiedContextManager"""
        if not context_manager:
            return ""

        try:
            #Get unified context for historical analysis
            unified_context = await context_manager.build_unified_context(session_id, None, "historical")

            context_parts = []

            # Chat history insights
            chat_history = unified_context.get("chat_history", [])
            if chat_history:
                context_parts.append(f"Conversation History: {len(chat_history)} messages available")

                # Analyze conversation patterns
                user_queries = [msg['content'] for msg in chat_history if msg.get('role') == 'user']
                if user_queries:
                    avg_query_length = sum(len(q) for q in user_queries) / len(user_queries)
                    context_parts.append(f"Query patterns: Avg length {avg_query_length:.0f} chars")

            # Execution history from variables
            if self.variable_manager:
                # Recent successful queries
                recent_successes = self.variable_manager.get("reasoning.recent_successes", [])
                if recent_successes:
                    context_parts.append(f"Recent successful queries: {len(recent_successes)}")

                # Performance history
                avg_loops = self.variable_manager.get("reasoning.performance.avg_loops", 0)
                if avg_loops:
                    context_parts.append(f"Average reasoning loops: {avg_loops}")

            # System insights from unified context
            execution_state = unified_context.get("execution_state", {})
            if execution_state.get("recent_completions"):
                completions = execution_state["recent_completions"]
                context_parts.append(f"Recent completions: {len(completions)} tasks finished")

            return "\n".join(context_parts)

        except Exception as e:
            eprint(f"Failed to get historical context: {e}")
            return "Historical context unavailable"

    def _load_historical_patterns(self):
        """Load successful patterns from previous reasoning sessions"""
        if not self.variable_manager:
            return

        # Load successful outline patterns
        successful_outlines = self.variable_manager.get("reasoning.successful_patterns.outlines", [])
        failed_patterns = self.variable_manager.get("reasoning.failed_patterns", [])

        self.historical_successful_patterns = successful_outlines[-5:]  # Last 5 successful
        self.historical_failed_patterns = failed_patterns[-10:]  # Last 10 failed

    def _get_historical_context(self) -> str:
        """Get historical context from variable system"""
        if not self.variable_manager:
            return ""

        context_parts = []

        # Recent successful queries
        recent_successes = self.variable_manager.get("reasoning.recent_successes", [])
        if recent_successes:
            context_parts.append(f"Recent successful queries: {len(recent_successes)}")

        # Performance history
        avg_loops = self.variable_manager.get("reasoning.performance.avg_loops", 0)
        if avg_loops:
            context_parts.append(f"Average reasoning loops: {avg_loops}")

        # Common failure patterns to avoid
        failure_patterns = self.variable_manager.get("reasoning.failure_patterns", [])
        if failure_patterns:
            context_parts.append(f"Known failure patterns: {len(failure_patterns)}")

        return "\n".join(context_parts)

    async def _initialize_reasoning_session(self, prep_res, original_query):
        """Initialize enhanced reasoning session with variable tracking"""
        # Initialize reasoning context
        self.reasoning_context.append({
            "type": "session_start",
            "content": f"Enhanced reasoning session started for: {original_query}",
            "timestamp": datetime.now().isoformat(),
            "session_id": prep_res.get("session_id")
        })

        # Store session in variables
        if self.variable_manager:
            session_data = {
                "query": original_query,
                "start_time": datetime.now().isoformat(),
                "max_loops": self.max_reasoning_loops,
                "context_management": "auto_summary",
                "outline_driven": True
            }
            self.variable_manager.set("reasoning.current_session.data", session_data)

        # Add enhanced system context
        self.reasoning_context.append({
            "type": "system_context",
            "content": prep_res["system_context"],
            "timestamp": datetime.now().isoformat()
        })

        # Add historical context if available
        historical = prep_res.get("historical_context")
        if historical:
            self.reasoning_context.append({
                "type": "historical_context",
                "content": historical,
                "timestamp": datetime.now().isoformat()
            })

    async def _create_initial_outline(self, prep_res) -> bool:
        """Create mandatory initial outline, with a fast path for simple queries."""
        original_query = prep_res["original_query"]
        agent_instance = prep_res["agent_instance"]

        outline_prompt = f"""You MUST create an initial execution outline for this query. This is mandatory.

**Query:** {original_query}

**Available Resources:**
- Tools: {', '.join(prep_res.get('available_tools', []))}
- Sub-systems: LLM Tool Node, Task Planner, Task Executor

LLM Tool Node is for all tool calls!
LLM Tool Node is best for simple multi-step tasks like fetching data from a tool and summarizing it.
Task Planner is best for complex tasks with multiple dependencies and complex task flows.

**Historical Context:** {prep_res.get('historical_context', 'None')}

**Fast Path for Simple Queries:**
If the query is simple and can be answered directly without needing tools or complex reasoning, you MUST create a single-step outline using the `direct_response` method.

Create a structured outline using this EXACT format:

OUTLINE_START
Step 1: [Brief description of first step]
- Method: [internal_reasoning | delegate_to_llm_tool_node | create_and_execute_plan | direct_response]
- Expected outcome: [What this step should achieve]
- Success criteria: [How to know this step is complete]

[For complex queries, continue with more steps as needed.]

Final Step: Synthesize results and provide comprehensive response
- Method: direct_response
- Expected outcome: Complete answer to user query
- Success criteria: User query fully addressed
OUTLINE_END

**Requirements:**
1. Outline must have between 1 and 7 steps.
2. For simple queries, a single "Final Step" using the 'direct_response' method is the correct approach.
3. Each step must have clear success criteria and build logically toward the answer.
4. Be specific about which meta-tools to use for each step. meta-tools ar not Tools ! avalabel meta-tools *Method* (internal_reasoning, delegate_to_llm_tool_node, create_and_execute_plan, direct_response) no exceptions

Create the outline now:"""

        try:
            llm_response = await agent_instance.a_run_llm_completion(
                model=prep_res.get("complex_llm_model", "openrouter/openai/gpt-4o"),
                messages=[{"role": "user", "content": outline_prompt}],
                temperature=0.2,  # Lower temperature for more deterministic outlining
                max_tokens=2048,
                node_name="LLMReasonerNode",
                task_id="create_initial_outline"
            )

            # Parse outline from response
            outline = self._parse_outline_from_response(llm_response)

            if self.agent_instance and self.agent_instance.progress_tracker:
                await self.agent_instance.progress_tracker.emit_event(ProgressEvent(
                    event_type="outline_created",
                    timestamp=time.time(),
                    node_name="LLMReasonerNode",
                    status=NodeStatus.COMPLETED,
                    task_id="create_initial_outline",
                    metadata={"outline": outline}
                ))

            if outline:
                self.outline = outline
                self.current_outline_step = 0

                # Store outline in variables
                if self.variable_manager:
                    self.variable_manager.set("reasoning.current_session.outline", outline)

                # Add to reasoning context
                self.reasoning_context.append({
                    "type": "outline_created",
                    "content": f"Created outline with {len(outline.get('steps', []))} steps",
                    "outline": outline,
                    "timestamp": datetime.now().isoformat()
                })

                return True
            else:
                return False

        except Exception as e:
            eprint(f"Failed to create initial outline: {e}")
            return False

    def _parse_outline_from_response(self, response: str) -> dict[str, Any]:
        """Parse structured outline from LLM response"""
        import re

        # Find outline section
        outline_match = re.search(r'OUTLINE_START(.*?)OUTLINE_END', response, re.DOTALL)
        if not outline_match:
            return None

        outline_text = outline_match.group(1).strip()

        # Parse steps
        steps = []
        current_step = None

        for line in outline_text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # New step
            if re.match(r'^Step \d+:', line):
                if current_step:
                    steps.append(current_step)

                current_step = {
                    "description": re.sub(r'^Step \d+:\s*', '', line),
                    "method": "",
                    "expected_outcome": "",
                    "success_criteria": "",
                    "status": "pending"
                }
            elif re.match(r'^Final Step:', line):
                if current_step:
                    steps.append(current_step)

                current_step = {
                    "description": re.sub(r'^Final Step:\s*', '', line),
                    "method": "direct_response",
                    "expected_outcome": "",
                    "success_criteria": "",
                    "status": "pending",
                    "is_final": True
                }
            elif current_step and line.startswith('- Method:'):
                current_step["method"] = line.replace('- Method:', '').strip()
            elif current_step and line.startswith('- Expected outcome:'):
                current_step["expected_outcome"] = line.replace('- Expected outcome:', '').strip()
            elif current_step and line.startswith('- Success criteria:'):
                current_step["success_criteria"] = line.replace('- Success criteria:', '').strip()

        # Add final step if exists
        if current_step:
            steps.append(current_step)

        if not steps:
            return None

        return {
            "steps": steps,
            "created_at": datetime.now().isoformat(),
            "total_steps": len(steps)
        }

    def _build_enhanced_system_context(self, shared) -> str:
        """Build comprehensive system context with variable system info"""
        context_parts = []

        # Enhanced agent capabilities
        available_tools = shared.get("available_tools", [])
        if available_tools:
            context_parts.append(f"Available external tools: {', '.join(available_tools)}")

        # Variable system context
        if self.variable_manager:
            var_info = self.variable_manager.get_scope_info()
            context_parts.append(f"Variable System: {len(var_info)} scopes available")

            # Recent results availability
            results_count = len(self.variable_manager.get("results", {}))
            if results_count:
                context_parts.append(f"Previous results: {results_count} task results available")

        # System state with enhanced details
        tasks = shared.get("tasks", {})
        if tasks:
            active_tasks = len([t for t in tasks.values() if t.status == "running"])
            completed_tasks = len([t for t in tasks.values() if t.status == "completed"])
            context_parts.append(f"Execution state: {active_tasks} active, {completed_tasks} completed tasks")

        # Session context with history
        formatted_context = shared.get("formatted_context", {})
        if formatted_context:
            recent_interaction = formatted_context.get("recent_interaction", "")
            if recent_interaction:
                context_parts.append(f"Recent interaction: {recent_interaction[:100000]}...")

        # Performance history
        if hasattr(self, 'historical_successful_patterns'):
            context_parts.append(
                f"Historical patterns: {len(self.historical_successful_patterns)} successful patterns loaded")

        return "\n".join(context_parts) if context_parts else "Basic system context available"

    async def _manage_context_size(self):
        """Auto-manage context size with intelligent summarization"""
        if len(self.reasoning_context) <= self.context_summary_threshold:
            return

        # Trigger summarization
        if len(self.reasoning_context) >= self.max_context_size:
            # Emergency summarization
            await self._emergency_context_summary()
        elif len(self.reasoning_context) >= self.context_summary_threshold:
            # Regular summarization
            await self._regular_context_summary()

    async def _regular_context_summary(self):
        """Regular context summarization when threshold is reached"""
        # Keep last 10 entries, summarize the rest
        keep_recent = self.reasoning_context[-10:]
        to_summarize = self.reasoning_context[:-10]

        summary = self._create_context_summary(to_summarize, "regular")

        # Replace old context with summary + recent
        self.reasoning_context = [
                                     {
                                         "type": "context_summary",
                                         "content": summary,
                                         "summarized_entries": len(to_summarize),
                                         "summary_type": "regular",
                                         "timestamp": datetime.now().isoformat()
                                     }
                                 ] + keep_recent

    async def _emergency_context_summary(self):
        """Emergency context summarization when max size is reached"""
        # Keep last 5 entries, summarize everything else
        keep_recent = self.reasoning_context[-5:]
        to_summarize = self.reasoning_context[:-5]

        summary = self._create_context_summary(to_summarize, "emergency")

        # Replace with emergency summary
        self.reasoning_context = [
                                     {
                                         "type": "context_summary",
                                         "content": summary,
                                         "summarized_entries": len(to_summarize),
                                         "summary_type": "emergency",
                                         "timestamp": datetime.now().isoformat()
                                     }
                                 ] + keep_recent

    def _create_context_summary(self, entries: list[dict], summary_type: str) -> str:
        """Create intelligent context summary"""
        if not entries:
            return "No context to summarize"

        summary_parts = []

        # Group by type
        by_type = {}
        for entry in entries:
            entry_type = entry.get("type", "unknown")
            if entry_type not in by_type:
                by_type[entry_type] = []
            by_type[entry_type].append(entry)

        # Summarize each type
        for entry_type, type_entries in by_type.items():
            if entry_type == "reasoning":
                reasoning_summary = f"Completed {len(type_entries)} reasoning cycles"
                # Extract key insights
                insights = []
                for entry in type_entries[-3:]:  # Last 3 reasoning entries
                    content = entry.get("content", "")[:1000] + "..."
                    insights.append(content)
                if insights:
                    reasoning_summary += f"\nKey recent reasoning: {'; '.join(insights)}"
                summary_parts.append(reasoning_summary)

            elif entry_type == "meta_tool_result":
                results_summary = f"Executed {len(type_entries)} meta-tool operations"
                # Extract significant results
                significant_results = [
                    entry.get("content", "")[:800]
                    for entry in type_entries
                    if len(entry.get("content", "")) > 50
                ]
                if significant_results:
                    results_summary += f"\nSignificant results: {'; '.join(significant_results[-3:])}"
                summary_parts.append(results_summary)

            else:
                summary_parts.append(f"{entry_type}: {len(type_entries)} entries")

        summary = f"[{summary_type.upper()} SUMMARY] " + "; ".join(summary_parts)

        # Store summary in variables for future reference
        if self.variable_manager:
            summary_data = {
                "type": summary_type,
                "entries_summarized": len(entries),
                "summary": summary,
                "timestamp": datetime.now().isoformat()
            }
            summaries = self.variable_manager.get("reasoning.context_summaries", [])
            summaries.append(summary_data)
            self.variable_manager.set("reasoning.context_summaries", summaries[-10:])  # Keep last 10

        return summary

    def _get_pending_tasks_summary(self) -> str:
        """Get summary of pending tasks requiring attention"""
        if not self.internal_task_stack:
            return "⚠️ NO TASKS IN STACK - You must create tasks from your outline immediately!"

        pending_tasks = [task for task in self.internal_task_stack if task.get("status", "pending") == "pending"]

        if not pending_tasks:
            return "✅ No pending tasks - ready for next outline step or completion"

        task_summaries = []
        for i, task in enumerate(pending_tasks[:3], 1):
            desc = task.get("description", "No description")[:150] + "..." if len(
                task.get("description", "")) > 50 else task.get("description", "")
            step_ref = task.get("outline_step_ref", "")
            step_info = f" ({step_ref})" if step_ref else ""
            task_summaries.append(f"{i}. {desc}{step_info}")

        if len(pending_tasks) > 3:
            task_summaries.append(f"... +{len(pending_tasks) - 3} more pending tasks")

        return f"📋 {len(pending_tasks)} pending tasks:\n" + "\n".join(task_summaries)

    async def _build_outline_driven_prompt(self, prep_res) -> str:
        """Build outline-driven reasoning prompt mit UnifiedContextManager Integration"""

        # Get current task with enhanced visibility
        current_stack_task = self._get_current_stack_task()

        #Enhanced context aus UnifiedContextManager
        context_manager = prep_res.get("context_manager")
        session_id = prep_res.get("session_id", "default")

        # Build unified context sections
        unified_context_summary = ""
        recent_results_context = ""

        if context_manager:
            try:
                # Get full unified context
                unified_context = await  context_manager.build_unified_context(session_id, prep_res.get('original_query'))

                unified_context_summary = self._format_unified_context_for_reasoning(unified_context)
                recent_results_context = self._build_recent_results_from_unified_context(unified_context)
            except Exception as e:
                eprint(f"Failed to get unified context in reasoning prompt: {e}")
                unified_context_summary = "Unified context unavailable"
                recent_results_context = "**No recent results available**"

        # Enhanced context summaries (keeping existing functionality)
        context_summary = self._summarize_reasoning_context()
        task_stack_summary = self._summarize_task_stack()
        outline_status = self._get_current_step_requirements()
        performance_context = self._get_performance_context()

        # Enhanced variable system integration with better suggestions
        variable_context = ""
        variable_suggestions = []
        if self.variable_manager:
            variable_context = self.variable_manager.get_llm_variable_context()
            query_text = prep_res.get('original_query', '')
            if current_stack_task:
                query_text += " " + current_stack_task.get('description', '')
            variable_suggestions = self.variable_manager.get_variable_suggestions(query_text)

        immediate_context = self._get_immediate_context_for_prompt()
        # Detect if we're in a potential loop situation
        loop_warning = self._generate_loop_warning()

        prompt = f"""You are the enhanced strategic reasoning core operating in OUTLINE-DRIVEN MODE with MANDATORY TASK STACK enforcement.
## ABSOLUTE REQUIREMENTS - VIOLATION = IMMEDIATE STOP:
1. **WORK ONLY THROUGH TASK STACK** - No work outside the stack permitted
2. **SEE CURRENT TASK DIRECTLY** - Your current task is shown below
3. **USE VARIABLE SYSTEM** - All results are automatically stored and accessible
4. **USE UNIFIED CONTEXT** - Rich conversation and execution history is available
5. **MARK TASKS COMPLETE** - Every finished task must be marked complete
6. **NO REPEATED ACTIONS** - Check variables first before re-doing work

{loop_warning}

## <CURRENT SITUATION>:
**Original Query:** {prep_res['original_query']}

**Unified Context Summary:**
{unified_context_summary}

**Current Context Summary:**
{context_summary}

**Current Outline Status:**
{outline_status}

** CURRENT TASK FROM STACK:**
{current_stack_task}

**Internal Task Stack:**
{task_stack_summary}

**Performance Metrics:**
{performance_context}

## ENHANCED CONTEXT INTEGRATION:
{variable_context}

** SUGGESTED VARIABLES for current task:**
{', '.join(variable_suggestions[:10]) if variable_suggestions else 'tool_capabilities, query, model_complex, available_tools, timestamp, use_fast_response, tool_registry, name, current_query, current_session'}

** UNIFIED CONTEXT RESULTS ACCESS:**
{recent_results_context}

</CURRENT SITUATION>

## MANDATORY TASK STACK ENFORCEMENT:
**CRITICAL RULE**: You MUST work exclusively through your internal task stack.

**TASK STACK WORKFLOW (MANDATORY):**
1. **CHECK CURRENT TASK**: Your current task is: {current_stack_task.get('description', 'NO CURRENT TASK - ADD TASKS FROM OUTLINE!') if current_stack_task else 'NO CURRENT TASK - VIOLATION!'}

2. **WORK ONLY ON STACK TASKS**: You can ONLY work on tasks that exist in your internal task stack
   - The task you're working on MUST be in the stack with status "pending"
   - Before any action: Verify the task exists in your stack

3. **MANDATORY TASK COMPLETION**: After completing any work, you MUST mark the task as complete
   - Use: META_TOOL_CALL: manage_internal_task_stack(action="complete", task_description="[exact task description]", outline_step_ref="step_X")

4. **CHECK UNIFIED CONTEXT FIRST**: Before any major action, focus your attention to the variable system to see if results already exist
   - Avalabel results are automatically stored in the variable system
   - The unified context above shows available conversation history and execution state

**CURRENT TASK ANALYSIS:**
{self._analyze_current_task(current_stack_task) if current_stack_task else "❌ NO CURRENT TASK - You must add tasks from your outline!"}

## AVAILABLE META-TOOLS:
You have access to these meta-tools to control sub-systems. Use the EXACT syntax shown:

**META_TOOL_CALL: internal_reasoning(thought: str, thought_number: int, total_thoughts: int, next_thought_needed: bool, current_focus: str, key_insights: list[str], potential_issues: list[str], confidence_level: float)**
- Purpose: Structure your thinking process explicitly
- Use for: Any complex analysis, planning, or problem decomposition
- Example: META_TOOL_CALL: internal_reasoning(thought="I need to break this down into steps", thought_number=1, total_thoughts=3, next_thought_needed=true, current_focus="problem analysis", key_insights=["Query requires multiple data sources"], potential_issues=["Data might not be available"], confidence_level=0.8)

**META_TOOL_CALL: manage_internal_task_stack(action: str, task_description: str)**
- Purpose: Manage your high-level to-do list
- Actions: "add", "remove", "complete", "get_current"
- Example: META_TOOL_CALL: manage_internal_task_stack(action="add", task_description="Research competitor analysis data")

**META_TOOL_CALL: delegate_to_llm_tool_node(task_description: str, tools_list: list[str])**
- Purpose: Delegate specific, self-contained tasks requiring external tools
- Use for: Web searches, file operations, API calls, single-, two-, or three-step tool usage
- Example: META_TOOL_CALL: delegate_to_llm_tool_node(task_description="Search for latest news about AI developments", tools_list=["search_web"])
- Rule: always validate delegate_to_llm_tool_node result. will be available in <immediate_context> after execution!

**META_TOOL_CALL: create_and_execute_plan(goals: list[str])**
- Purpose: Handle complex, multi-step projects with dependencies
- Use for: Tasks requiring coordination, parallel execution, or complex workflows
- Example: META_TOOL_CALL: create_and_execute_plan(goals=["Research company A financial data", "Research company B financial data", "Compare {{results.task_1.data}} and {{results.task_2.data}} and create report"])
- Rule: always validate create_and_execute_plan result. will be available in <immediate_context> after execution!

**META_TOOL_CALL: read_from_variables(scope: str, key: str, purpose: str)**
- Unified context data is available in various scopes
- Example: META_TOOL_CALL: read_from_variables(scope="user", key="name", purpose="Gather user information for later reference")

**META_TOOL_CALL: write_to_variables(scope: str, key: str, value: any, description: str)**
- Store important findings immediately
- Example: META_TOOL_CALL: write_to_variables(scope="user", key="name", value="User-Name", description="The users name for later reference")

**META_TOOL_CALL: advance_outline_step(step_completed: bool, completion_evidence: str, next_step_focus: str)**
- Mark outline steps complete when all related tasks done

**META_TOOL_CALL: direct_response(final_answer: str, outline_completion: bool, steps_completed: list[str])**
- ONLY when ALL outline steps complete or no META_TOOL_CALL needed
- final_answer must contain the full final answer for the user with all necessary context and informations ( format in persona style )
- Purpose: End reasoning and provide final answer to user
- Use when: Query is complete or can be answered directly
- Example: META_TOOL_CALL: direct_response(final_answer="Based on my analysis, here are the key findings...")

note: in this interaction only META_TOOL_CALL ar avalabel. for other tools use META_TOOL_CALL: delegate_to_llm_tool_node with the appropriate tool names!

## REASONING STRATEGY:
1. **Start with internal_reasoning** to understand the query and plan approach
2. **Use manage_internal_task_stack** to track high-level steps
3. **Choose the right delegation strategy:**
   - Simple queries → direct_response
   - Up to 3 tool tasks with llm action → delegate_to_llm_tool_node
   - Complex projects → create_and_execute_plan
4. **Monitor progress** and adapt your approach
5. **End with direct_response** when complete

## EXAMPLES OF GOOD REASONING PATTERNS:

**Simple Query Pattern:**
META_TOOL_CALL: internal_reasoning(thought="This is a straightforward question I can answer directly", thought_number=1, total_thoughts=1, next_thought_needed=false, current_focus="direct response", key_insights=["No external data needed"], potential_issues=[], confidence_level=0.9)
META_TOOL_CALL: direct_response(final_answer="...")

**Research Task Pattern:**
META_TOOL_CALL: internal_reasoning(thought="I need to gather information from external sources", ...)
META_TOOL_CALL: manage_internal_task_stack(action="add", task_description="Research topic X")
META_TOOL_CALL: delegate_to_llm_tool_node(task_description="Search for information about X", tools_list=["search_web"])
[Wait for result]
META_TOOL_CALL: internal_reasoning(thought="I have the research data, now I can formulate response", ...)
META_TOOL_CALL: direct_response(final_answer="Based on my research: ...")

**Complex Project Pattern:**
META_TOOL_CALL: internal_reasoning(thought="This requires multiple steps with dependencies", ...)
META_TOOL_CALL: create_and_execute_plan(goals=["Step 1: Gather data A", "Step 2: Gather data B", "Step 3: Analyze A and B together", "Step 4: Create final report"])
[Wait for plan completion]
META_TOOL_CALL: direct_response(final_answer="I've completed your complex request...")

## ENHANCED ANTI-LOOP ENFORCEMENT:
- Current Loop: {self.current_loop_count}/{self.max_reasoning_loops}
- Auto-Recovery Attempts: {getattr(self, 'auto_recovery_attempts', 0)}/{getattr(self, 'max_auto_recovery', 3)}
- Last Actions: {', '.join(getattr(self, 'last_action_signatures', [])[-3:]) if hasattr(self, 'last_action_signatures') else 'None'}

**⚠️ LOOP PREVENTION RULES:**
1. If you just read a variable, DO NOT read the same variable again
2. If you completed a task, DO NOT repeat the same work
3. If results exist in unified context, DO NOT recreate them
4. Always advance to next logical step

{self._get_current_step_requirements()}

## YOUR NEXT ACTION (Choose ONE):
Based on your current task, unified context, and available variables, what is your next concrete action?

**DECISION TREE:**
1. ❓ No current task? → Add tasks from outline
2. 📖 Current task needs data? → Check variables and unified context first (read_from_variables)
3. 🔧 Need to execute tools and reason over up to 3 steps? → Use delegate_to_llm_tool_node
4. ✅ Task complete? → Mark complete and advance
5. 🎯 All outline done? → Provide direct_response

Latest unified context: (note delegation results could be wrong or misleading)
<immediate_context>
{immediate_context}
</immediate_context>

must validate <immediate_context> output!
- validate the <immediate_context> output! before proceeding with the outline!
- output compleat fail -> direct_response
- informations missing or output recovery needed -> repeat step with a different strategy
- not enough structure -> use create_and_execute_plan meta-tool call
- output is valid -> continue with the outline!
- if dynamic Planing is needed, you must use the appropriate meta-tool call

**Remember**:
- work step by step max call 3 meta-tool calls in one run.
- only use direct_response if the outline is complete and context from <immediate_context> is enough to answer the query!
- Your job is to work systematically through your outline using your task stack, while leveraging the unified context system to avoid duplicate work and maintain context."""

        return prompt

    def _format_unified_context_for_reasoning(self, unified_context: dict[str, Any]) -> str:
        """Format unified context für reasoning prompt"""
        try:
            context_parts = []

            # Session info
            session_stats = unified_context.get('session_stats', {})
            context_parts.append(
                f"Session: {unified_context.get('session_id', 'unknown')} with {session_stats.get('current_session_length', 0)} messages")

            # Chat history summary
            chat_history = unified_context.get('chat_history', [])
            if chat_history:
                recent_messages = len([msg for msg in chat_history if msg.get('role') == 'user'])
                context_parts.append(f"Conversation: {recent_messages} user queries in current context")

                # Show last user message for reference
                last_user_msg = None
                for msg in reversed(chat_history):
                    if msg.get('role') == 'user':
                        last_user_msg = msg.get('content', '')[:100] + "..."
                        break
                if last_user_msg:
                    context_parts.append(f"Latest user query: {last_user_msg}")

            # Execution state
            execution_state = unified_context.get('execution_state', {})
            active_tasks = execution_state.get('active_tasks', [])
            recent_completions = execution_state.get('recent_completions', [])
            if active_tasks or recent_completions:
                context_parts.append(
                    f"Execution: {len(active_tasks)} active, {len(recent_completions)} completed tasks")

            # Available data
            variables = unified_context.get('variables', {})
            recent_results = variables.get('recent_results', [])
            if recent_results:
                context_parts.append(f"Available Results: {len(recent_results)} recent task results accessible")

            return "\n".join(context_parts)

        except Exception as e:
            return f"Error formatting unified context: {str(e)}"

    def _build_recent_results_from_unified_context(self, unified_context: dict[str, Any]) -> str:
        """Build recent results context from unified context"""
        try:
            variables = unified_context.get('variables', {})
            recent_results = variables.get('recent_results', [])

            if not recent_results:
                return "**No recent results available from unified context**"

            result_context = """**🔍 RECENT RESULTS FROM UNIFIED CONTEXT:**"""

            for i, result in enumerate(recent_results[:3], 1):  # Top 3 results
                task_id = result.get('task_id', f'result_{i}')
                preview = result.get('preview', 'No preview')
                success = result.get('success', False)
                status_icon = "✅" if success else "❌"

                result_context += f"\n{status_icon} {task_id}: {preview}"

            result_context += "\n\n**Quick Access Keys Available:**"
            result_context += "\n- Use read_from_variables(scope='results', key='task_id.data') for specific results"
            result_context += "\n- Check delegation.latest for most recent delegation results"

            return result_context

        except Exception as e:
            return f"**Error accessing recent results: {str(e)}**"

    def _generate_loop_warning(self) -> str:
        """Generate loop warning if repetitive behavior detected"""
        if len(self.last_action_signatures) >= 3:
            recent_actions = self.last_action_signatures[-3:]
            if len(set(recent_actions)) <= 2:
                return """
⚠️ **LOOP WARNING DETECTED** ⚠️
You are repeating similar actions. MUST change approach:
- If you just read variables, act on the results
- If you delegated tasks, check the results
- Complete current task and advance to next step
- DO NOT repeat the same meta-tool calls
    """
        return ""

    def _get_current_stack_task(self) -> dict[str, Any]:
        """Get current pending task from stack for direct visibility"""
        if not self.internal_task_stack:
            return {}

        pending_tasks = [task for task in self.internal_task_stack if task.get("status", "pending") == "pending"]
        if pending_tasks:
            current_task = pending_tasks[0]  # Get first pending task
            return {
                "description": current_task.get("description", ""),
                "outline_step_ref": current_task.get("outline_step_ref", ""),
                "status": current_task.get("status", "pending"),
                "added_at": current_task.get("added_at", ""),
                "task_index": self.internal_task_stack.index(current_task) + 1,
                "total_tasks": len(self.internal_task_stack)
            }

        return {}

    def _analyze_current_task(self, current_task: dict[str, Any]) -> str:
        """Analyze current task and provide guidance"""
        if not current_task:
            return "❌ NO CURRENT TASK - Add tasks from your outline immediately!"

        description = current_task.get("description", "")
        outline_ref = current_task.get("outline_step_ref", "")

        analysis = f"""CURRENT TASK IDENTIFIED:
Task: {description}
Outline Reference: {outline_ref}
Position: {current_task.get('task_index', '?')}/{current_task.get('total_tasks', '?')}

RECOMMENDED ACTION:"""

        # Analyze task content for recommendations
        if "read" in description.lower() or "file" in description.lower():
            analysis += "\n1. Check if file content already exists in variables (read_from_variables)"
            analysis += "\n2. If not found, use delegate_to_llm_tool_node with read_file tool"
        elif "write" in description.lower() or "create" in description.lower():
            analysis += "\n1. Check if content is ready in variables"
            analysis += "\n2. Use delegate_to_llm_tool_node with write_file tool"
        elif "analyze" in description.lower() or "question" in description.lower():
            analysis += "\n1. Read existing data from variables"
            analysis += "\n2. Process the information and provide direct_response"
        else:
            analysis += "\n1. Break down the task into specific actions"
            analysis += "\n2. Verify last Task Delegation results"

        return analysis

    def _get_immediate_context_for_prompt(self) -> str:
        """Get immediate context additions from recent meta-tool executions"""
        recent_results = [
            entry for entry in self.reasoning_context[-5:]  # Last 5 entries
            if entry.get("type") == "meta_tool_result"
        ]

        if not recent_results:
            return "No recent meta-tool results"

        context_parts = ["📊 IMMEDIATE CONTEXT FROM RECENT ACTIONS:"]

        for result in recent_results:
            meta_tool = result.get("meta_tool", "unknown")
            content = result.get("content", "")
            loop = result.get("loop", "?")

            # Format based on meta-tool type
            if meta_tool == "delegate_to_llm_tool_node":
                context_parts.append(f"✅ DELEGATION RESULT (Loop {loop}):")
                context_parts.append(f"   {content}")
            elif meta_tool == "read_from_variables":
                context_parts.append(f"📖 VARIABLE READ (Loop {loop}):")
                context_parts.append(f"   {content}")
            elif meta_tool == "manage_internal_task_stack":
                context_parts.append(f"📋 TASK UPDATE (Loop {loop}):")
                context_parts.append(f"   {content}")
            else:
                context_parts.append(f"🔧 {meta_tool.upper()} (Loop {loop}):")
                context_parts.append(f"   {content}")

        return "\n".join(context_parts)

    def _summarize_reasoning_context(self) -> str:
        """Enhanced reasoning context summary with immediate result visibility"""
        if not self.reasoning_context:
            return "No previous reasoning steps"

        # Separate different types of context entries
        reasoning_entries = []
        meta_tool_results = []
        errors = []

        for entry in self.reasoning_context:
            entry_type = entry.get("type", "unknown")

            if entry_type == "reasoning":
                reasoning_entries.append(entry)
            elif entry_type == "meta_tool_result":
                meta_tool_results.append(entry)
            elif entry_type == "error":
                errors.append(entry)

        summary_parts = []

        # Show recent meta-tool results FIRST for immediate visibility
        if meta_tool_results:
            summary_parts.append("🔍 RECENT RESULTS:")
            for result in meta_tool_results[-3:]:  # Last 3 results
                meta_tool = result.get("meta_tool", "unknown")
                content = result.get("content", "")[:3000] + "..."
                loop = result.get("loop", "?")
                summary_parts.append(f"  [{meta_tool}] Loop {loop}: {content}")

        # Show reasoning summary
        if reasoning_entries:
            summary_parts.append(f"\n💭 REASONING: {len(reasoning_entries)} reasoning cycles completed")

        # Show errors if any
        if errors:
            summary_parts.append(f"\n⚠️ ERRORS: {len(errors)} errors encountered")
            for error in errors[-2:]:  # Last 2 errors
                content = error.get("content", "")[:1500]
                summary_parts.append(f"  Error: {content}")

        return "\n".join(summary_parts)

    def _get_current_step_requirements(self) -> str:
        """Get requirements for current outline step"""
        if not self.outline or not self.outline.get("steps"):
            return "ERROR: No outline available"

        steps = self.outline["steps"]
        if self.current_outline_step >= len(steps):
            return "All outline steps completed - must provide final response"

        current_step = steps[self.current_outline_step]

        requirements = f"""CURRENT STEP FOCUS:
Description: {current_step.get('description', 'Unknown')}
Required Method: {current_step.get('method', 'Unknown')}
Expected Outcome: {current_step.get('expected_outcome', 'Unknown')}
Success Criteria: {current_step.get('success_criteria', 'Unknown')}
Current Status: {current_step.get('status', 'pending')}

You MUST use the specified method and achieve the expected outcome before advancing."""

        return requirements

    def _get_performance_context(self) -> str:
        """Get performance context with accurate metrics"""
        if not self.performance_metrics:
            return "No performance metrics available"

        metrics_parts = []

        # Core metrics
        avg_time = self.performance_metrics.get("avg_loop_time", 0)
        efficiency = self.performance_metrics.get("action_efficiency", 0)
        total_loops = self.performance_metrics.get("total_loops", 0)
        progress_loops = self.performance_metrics.get("progress_loops", 0)

        metrics_parts.append(f"Avg Loop Time: {avg_time:.2f}s")
        metrics_parts.append(f"Progress Rate: {efficiency:.1%}")
        metrics_parts.append(f"Action Efficiency: {efficiency:.1%}")

        # Performance warnings
        if total_loops > 3 and efficiency < 0.5:
            metrics_parts.append("⚠️ LOW EFFICIENCY - Need more progress actions")
        elif total_loops > 5 and efficiency < 0.3:
            metrics_parts.append("🔴 VERY LOW EFFICIENCY - Review approach")

        # Loop detection warning based on actual metrics
        if len(self.last_action_signatures) > 3:
            unique_recent = len(set(self.last_action_signatures[-3:]))
            if unique_recent <= 1:
                metrics_parts.append("⚠️ LOOP PATTERN DETECTED - Change approach required")

        return "; ".join(metrics_parts)

    def _track_action_type(self, action_type: str, success: bool = True):
        """Track specific action types for detailed performance analysis"""
        if not hasattr(self, 'action_tracking'):
            self.action_tracking = {}

        if action_type not in self.action_tracking:
            self.action_tracking[action_type] = {"total": 0, "successful": 0}

        self.action_tracking[action_type]["total"] += 1
        if success:
            self.action_tracking[action_type]["successful"] += 1

        # Update overall action efficiency based on all action types
        total_actions = sum(stats["total"] for stats in self.action_tracking.values())
        successful_actions = sum(stats["successful"] for stats in self.action_tracking.values())

        if total_actions > 0:
            self.performance_metrics["detailed_action_efficiency"] = successful_actions / total_actions


    def _detect_infinite_loop(self) -> bool:
        """Enhanced infinite loop detection with multiple patterns"""
        if len(self.last_action_signatures) < 3:
            return False

        # 1. Immediate repetition (same action 3+ times)
        recent_actions = self.last_action_signatures[-3:]
        if len(set(recent_actions)) == 1:
            return True

        # 2. Pattern repetition (AB-AB-AB pattern)
        if len(self.last_action_signatures) >= 6:
            pattern1 = self.last_action_signatures[-6:-3]
            pattern2 = self.last_action_signatures[-3:]
            if pattern1 == pattern2:
                return True

        # 3. Variable read loops (multiple reads of same variable)
        variable_reads = [sig for sig in self.last_action_signatures if sig.startswith("read_from_variables")]
        if len(variable_reads) >= 3:
            # Extract variable signatures from recent reads
            recent_var_reads = variable_reads[-3:]
            if len(set(recent_var_reads)) <= 2:  # Repeated variable reads
                return True

        # 4. No outline progress for extended loops
        if self.current_loop_count > 5:
            if not hasattr(self, '_last_step_progress_loop'):
                self._last_step_progress_loop = {}

            last_progress = self._last_step_progress_loop.get(self.current_outline_step, 0)
            if self.current_loop_count - last_progress > 4:  # No step progress for 4+ loops
                return True

        # 5. Same task stack state for multiple loops
        if hasattr(self, '_task_stack_states'):
            stack_signature = hash(
                str([(t.get('status'), t.get('description')[:20]) for t in self.internal_task_stack]))
            if stack_signature in self._task_stack_states:
                repetitions = self._task_stack_states[stack_signature]
                if repetitions >= 4:
                    return True
                self._task_stack_states[stack_signature] = repetitions + 1
            else:
                self._task_stack_states[stack_signature] = 1
        else:
            self._task_stack_states = {}

        return False

    async def _trigger_auto_recovery(self, prep_res):
        """Trigger auto-recovery mechanism"""
        self.auto_recovery_attempts += 1

        # Store failure pattern
        if self.variable_manager:
            failure_data = {
                "timestamp": datetime.now().isoformat(),
                "loop_count": self.current_loop_count,
                "outline_step": self.current_outline_step,
                "last_actions": self.last_action_signatures[-5:],
                "recovery_attempt": self.auto_recovery_attempts
            }
            failures = self.variable_manager.get("reasoning.failure_patterns", [])
            failures.append(failure_data)
            self.variable_manager.set("reasoning.failure_patterns", failures[-20:])  # Keep last 20

        # Recovery strategies
        if self.auto_recovery_attempts == 1:
            # Force outline step advancement
            await self._force_outline_advancement(prep_res)
        elif self.auto_recovery_attempts == 2:
            # Skip current step and move to next
            await self._emergency_step_skip(prep_res)
        else:
            # Final emergency: force completion
            await self._emergency_completion(prep_res)

    async def _force_outline_advancement(self, prep_res):
        """Force advancement to next outline step"""
        if self.outline and self.current_outline_step < len(self.outline["steps"]):
            current_step = self.outline["steps"][self.current_outline_step]
            current_step["status"] = "force_completed"
            current_step["completion_method"] = "auto_recovery"

            self.current_outline_step += 1

            # Add to context
            self.reasoning_context.append({
                "type": "auto_recovery",
                "content": f"Force advanced to step {self.current_outline_step + 1} due to loop detection",
                "recovery_attempt": self.auto_recovery_attempts,
                "timestamp": datetime.now().isoformat()
            })

    async def _emergency_step_skip(self, prep_res):
        """Emergency skip of problematic step"""
        if self.outline and self.current_outline_step < len(self.outline["steps"]) - 1:
            current_step = self.outline["steps"][self.current_outline_step]
            current_step["status"] = "emergency_skipped"
            current_step["skip_reason"] = "loop_recovery"

            self.current_outline_step += 1

            # Add to context
            self.reasoning_context.append({
                "type": "emergency_skip",
                "content": f"Emergency skipped step {self.current_outline_step} and advanced to step {self.current_outline_step + 1}",
                "recovery_attempt": self.auto_recovery_attempts,
                "timestamp": datetime.now().isoformat()
            })

    async def _emergency_completion(self, prep_res):
        """Emergency completion of reasoning"""
        # Mark all remaining steps as emergency completed
        if self.outline:
            for i in range(self.current_outline_step, len(self.outline["steps"])):
                self.outline["steps"][i]["status"] = "emergency_completed"

            self.current_outline_step = len(self.outline["steps"])

        # Add to context
        self.reasoning_context.append({
            "type": "emergency_completion",
            "content": "Emergency completion triggered due to excessive recovery attempts",
            "recovery_attempt": self.auto_recovery_attempts,
            "timestamp": datetime.now().isoformat()
        })

    def _update_performance_metrics(self, loop_start_time: float, progress_made: bool):
        """Update performance metrics with accurate action efficiency tracking"""
        loop_duration = time.time() - loop_start_time

        # Initialize metrics if needed
        if not hasattr(self, 'performance_metrics') or not self.performance_metrics:
            self.performance_metrics = {
                "loop_times": [],
                "progress_loops": 0,
                "total_loops": 0
            }

        # Update core metrics
        self.performance_metrics["loop_times"].append(loop_duration)
        self.performance_metrics["total_loops"] += 1

        if progress_made:
            self.performance_metrics["progress_loops"] += 1

        # Calculate derived metrics
        total = self.performance_metrics["total_loops"]
        progress = self.performance_metrics["progress_loops"]

        self.performance_metrics["avg_loop_time"] = sum(self.performance_metrics["loop_times"]) / len(
            self.performance_metrics["loop_times"])
        self.performance_metrics["action_efficiency"] = progress / total if total > 0 else 0.0
        self.performance_metrics["progress_rate"] = self.performance_metrics["action_efficiency"]  # Same metric

        # Keep only recent loop times for memory efficiency
        if len(self.performance_metrics["loop_times"]) > 10:
            self.performance_metrics["loop_times"] = self.performance_metrics["loop_times"][-10:]

    def _add_context_to_reasoning(self, context_addition: str, meta_tool_name: str,
                                  execution_details: dict = None) -> None:
        """Add context addition to reasoning context for immediate visibility in next LLM prompt"""
        if not context_addition:
            return

        # Create structured context entry
        context_entry = {
            "type": "meta_tool_result",
            "content": context_addition,
            "meta_tool": meta_tool_name,
            "loop": self.current_loop_count,
            "outline_step": getattr(self, 'current_outline_step', 0),
            "timestamp": datetime.now().isoformat()
        }

        # Add execution details if provided
        if execution_details:
            context_entry["execution_details"] = {
                "duration": execution_details.get("execution_duration", 0),
                "success": execution_details.get("execution_success", False),
                "tool_category": execution_details.get("tool_category", "unknown")
            }

        # Add to reasoning context for immediate visibility
        self.reasoning_context.append(context_entry)

        # Store in variables for persistent access
        if self.agent_instance:
            if not self.agent_instance.shared.get("system_context"):
                self.agent_instance.shared["system_context"] = {}
            if not self.agent_instance.shared["system_context"].get("reasoning_context"):
                self.agent_instance.shared["system_context"]["reasoning_context"] = {}

            result_key = f"reasoning.loop_{self.current_loop_count}_{meta_tool_name}"
            self.agent_instance.shared["system_context"]["reasoning_context"][result_key] = {
                "context_addition": context_addition,
                "meta_tool": meta_tool_name,
                "timestamp": datetime.now().isoformat(),
                "loop": self.current_loop_count
            }

    async def _parse_and_execute_meta_tools(self, llm_response: str, prep_res: dict) -> dict[str, Any]:
        """Enhanced meta-tool parsing with comprehensive progress tracking"""

        result = {
            "final_result": None,
            "action_taken": None,
            "progress_made": False,
            "context_addition": None
        }

        progress_tracker = prep_res.get("progress_tracker")
        session_id = prep_res.get("session_id")

        # Pattern to match META_TOOL_CALL: tool_name(args...)
        pattern = r'META_TOOL_CALL:'
        matches = _extract_meta_tool_calls(llm_response, pattern)

        if not matches and progress_tracker:
            # No meta-tools found in response
            await progress_tracker.emit_event(ProgressEvent(
                event_type="meta_tool_analysis",
                node_name="LLMReasonerNode",
                session_id=session_id,
                status=NodeStatus.COMPLETED,
                success=True,  # Die Analyse selbst war erfolgreich
                node_phase="analysis_complete",  # Verwendung des dedizierten Feldes
                llm_output=llm_response,  # Speichert die vollständige analysierte Antwort
                metadata={
                    "analysis_result": "no_meta_tools_detected",
                    "reasoning_loop": self.current_loop_count,
                    "outline_step": self.current_outline_step if hasattr(self, 'current_outline_step') else 0,
                    "context_size": len(self.reasoning_context),
                    "performance_warning": len(self.reasoning_context) > 10 and self.current_loop_count > 5
                }
            ))
            result["context_addition"] = "No action taken - this violates outline-driven execution requirements"
            self._add_context_to_reasoning(result["context_addition"], "invalid", {})

            return result

        for i, (tool_name, args_str) in enumerate(matches):
            meta_tool_start = time.perf_counter()

            # Track action signature for loop detection
            action_signature = f"{tool_name}:{hash(args_str) % 1000}"
            self.last_action_signatures.append(action_signature)
            if len(self.last_action_signatures) > 10:
                self.last_action_signatures = self.last_action_signatures[-10:]

            try:
                # Parse arguments with enhanced error handling
                args = _parse_tool_args(args_str)
                if progress_tracker:
                    await progress_tracker.emit_event(ProgressEvent(
                        event_type="tool_call",  # Vereinheitlicht auf "tool_call"
                        node_name="LLMReasonerNode",
                        session_id=session_id,
                        status=NodeStatus.RUNNING,
                        tool_name=tool_name,
                        is_meta_tool=True,  # Klares Flag für Meta-Tools
                        tool_args=args,
                        task_id=f"meta_tool_{tool_name}_{i + 1}",
                        metadata={
                            "reasoning_loop": self.current_loop_count,
                            "outline_step": self.current_outline_step if hasattr(self, 'current_outline_step') else 0
                        }
                    ))
                rprint(f"Parsed args: {args}")

                # Execute meta-tool with detailed tracking
                meta_result = None
                execution_details = {
                    "meta_tool_name": tool_name,
                    "parsed_args": args,
                    "execution_success": False,
                    "execution_duration": 0.0,
                    "reasoning_loop": self.current_loop_count,
                    "outline_step": self.current_outline_step if hasattr(self, 'current_outline_step') else 0,
                    "context_before_size": len(self.reasoning_context),
                    "task_stack_before_size": len(self.internal_task_stack),
                    "tool_category": self._get_tool_category(tool_name),
                    "execution_phase": "executing"
                }

                if tool_name == "internal_reasoning":
                    meta_result = await self._execute_enhanced_internal_reasoning(args, prep_res)
                    execution_details.update({
                        "thought_number": args.get("thought_number", 1),
                        "total_thoughts": args.get("total_thoughts", 1),
                        "current_focus": args.get("current_focus", ""),
                        "confidence_level": args.get("confidence_level", 0.5),
                        "key_insights": args.get("key_insights", []),
                        "key_insights_count": len(args.get("key_insights", [])),
                        "potential_issues_count": len(args.get("potential_issues", [])),
                        "next_thought_needed": args.get("next_thought_needed", False),
                        "internal_reasoning_log_size": len(getattr(self, 'internal_reasoning_log', [])),
                        "reasoning_depth": self._calculate_reasoning_depth(),
                        "outline_step_progress": args.get("outline_step_progress", "")
                    })
                    result["action_taken"] = False

                elif tool_name == "manage_internal_task_stack":
                    meta_result = await self._execute_enhanced_task_stack(args, prep_res)
                    execution_details.update({
                        "stack_action": args.get("action", "unknown"),
                        "task_description": args.get("task_description", ""),
                        "outline_step_ref": args.get("outline_step_ref", ""),
                        "stack_size_before": len(self.internal_task_stack),
                        "stack_size_after": 0  # Will be updated below
                    })
                    execution_details["stack_size_after"] = len(self.internal_task_stack)
                    execution_details["stack_change"] = execution_details["stack_size_after"] - execution_details[
                        "stack_size_before"]
                    result["action_taken"] = True

                elif tool_name == "delegate_to_llm_tool_node":
                    meta_result = await self._execute_enhanced_delegate_llm_tool(args, prep_res)
                    execution_details.update({
                        "delegated_task_description": args.get("task_description", ""),
                        "tools_list": args.get("tools_list", []),
                        "tools_count": len(args.get("tools_list", [])),
                        "delegation_target": "LLMToolNode",
                        "sub_system_execution": True,
                        "delegation_complexity": self._assess_delegation_complexity(args),
                        "outline_step_completion": args.get("outline_step_completion", False)
                    })
                    result["action_taken"] = True
                    result["progress_made"] = True

                elif tool_name == "create_and_execute_plan":
                    meta_result = await self._execute_enhanced_create_plan(args, prep_res)
                    execution_details.update({
                        "goals_list": args.get("goals", []),
                        "goals_count": len(args.get("goals", [])),
                        "plan_execution_target": "TaskPlanner_TaskExecutor",
                        "sub_system_execution": True,
                        "complex_workflow": True,
                        "estimated_complexity": self._estimate_plan_complexity(args.get("goals", [])),
                        "outline_step_completion": args.get("outline_step_completion", False)
                    })
                    result["action_taken"] = True
                    result["progress_made"] = True

                elif tool_name == "advance_outline_step":
                    meta_result = await self._execute_advance_outline_step(args, prep_res)
                    execution_details.update({
                        "step_completed": args.get("step_completed", False),
                        "completion_evidence": args.get("completion_evidence", ""),
                        "next_step_focus": args.get("next_step_focus", ""),
                        "outline_advancement": True,
                        "step_progression": f"{self.current_outline_step}/{len(self.outline.get('steps', [])) if self.outline else 0}"
                    })
                    result["action_taken"] = True
                    result["progress_made"] = True

                elif tool_name == "write_to_variables":
                    meta_result = await self._execute_write_to_variables(args)
                    execution_details.update({
                        "variable_scope": args.get("scope", "reasoning"),
                        "variable_key": args.get("key", ""),
                        "variable_description": args.get("description", ""),
                        "data_persistence": True,
                        "variable_system_operation": "write"
                    })
                    result["action_taken"] = True

                elif tool_name == "read_from_variables":
                    meta_result = await self._execute_read_from_variables(args)
                    execution_details.update({
                        "variable_scope": args.get("scope", "reasoning"),
                        "variable_key": args.get("key", ""),
                        "read_purpose": args.get("purpose", ""),
                        "variable_system_operation": "read",
                        "data_retrieval": True
                    })
                    result["action_taken"] = True

                elif tool_name == "direct_response":

                    final_answer = args.get("final_answer", "Task completed.").replace('\\n', '\n').replace('\\t', '\t')
                    execution_details.update({
                        "final_answer": final_answer,
                        "final_answer_length": len(final_answer),
                        "reasoning_complete": True,
                        "flow_termination": True,
                        "reasoning_summary": self._create_reasoning_summary(),
                        "total_reasoning_steps": len(self.reasoning_context),
                        "outline_completion": True,
                        "steps_completed": args.get("steps_completed", []),
                        "session_completion": True
                    })

                    completion_context = f"✅ REASONING COMPLETE: {final_answer}"
                    self._add_context_to_reasoning(completion_context, tool_name, execution_details)

                    # Store successful completion
                    await self._store_successful_completion(prep_res, final_answer)

                    if progress_tracker:
                        meta_tool_duration = time.perf_counter() - meta_tool_start
                        execution_details["execution_duration"] = meta_tool_duration
                        execution_details["execution_success"] = True

                        await progress_tracker.emit_event(ProgressEvent(
                            event_type="meta_tool_call",
                            timestamp=time.time(),
                            node_name="LLMReasonerNode",
                            status=NodeStatus.COMPLETED,
                            session_id=session_id,
                            task_id=f"meta_tool_{tool_name}_{i + 1}",
                            node_duration=meta_tool_duration,
                            success=True,
                            metadata=execution_details
                        ))

                    result["final_result"] = final_answer
                    result["action_taken"] = True
                    result["progress_made"] = True
                    return result

                # test if tool name is in agent tools if so try to run it
                elif tool_name in self.agent_instance.tool_registry:
                    meta_result = await self.agent_instance.arun_function(tool_name, **args)
                    result["action_taken"] = True
                    result["progress_made"] = True
                    execution_details.update({
                        "tool_name": tool_name,
                        "tool_args": args,
                        "tool_result": meta_result
                    })

                else:
                    execution_details.update({
                        "error_type": "unknown_meta_tool",
                        "error_message": f"Unknown meta-tool: {tool_name}",
                        "execution_success": False,
                        "available_meta_tools": ["internal_reasoning", "manage_internal_task_stack",
                                            "delegate_to_llm_tool_node", "create_and_execute_plan",
                                            "advance_outline_step", "write_to_variables", "read_from_variables",
                                            "direct_response"]
                    })

                    if progress_tracker:
                        meta_tool_duration = time.perf_counter() - meta_tool_start
                        await progress_tracker.emit_event(ProgressEvent(
                            event_type="meta_tool_call",
                            timestamp=time.time(),
                            node_name="LLMReasonerNode",
                            status=NodeStatus.FAILED,
                            session_id=session_id,
                            task_id=f"meta_tool_{tool_name}_{i + 1}",
                            node_duration=meta_tool_duration,
                            success=False,
                            metadata=execution_details
                        ))

                    error_context = f"❌ Unknown meta-tool: {tool_name}"
                    self._add_context_to_reasoning(error_context, tool_name, execution_details)
                    wprint(f"Unknown meta-tool: {tool_name}")
                    continue

                # Update execution details with results
                meta_tool_duration = time.perf_counter() - meta_tool_start
                execution_details.update({
                    "execution_duration": meta_tool_duration,
                    "execution_success": True,
                    "context_after_size": len(self.reasoning_context),
                    "task_stack_after_size": len(self.internal_task_stack),
                    "performance_score": self._calculate_tool_performance_score(meta_tool_duration, tool_name),
                    "execution_phase": "completed"
                })
                self._track_action_type(tool_name, success=True)

                # Add result to context
                if meta_result and meta_result.get("context_addition"):
                    result["context_addition"] = meta_result["context_addition"]
                    execution_details["context_addition_length"] = len(meta_result["context_addition"])

                    self._add_context_to_reasoning(meta_result["context_addition"], tool_name, execution_details)

                # Emit success event
                if progress_tracker:
                    await progress_tracker.emit_event(ProgressEvent(
                        event_type="meta_tool_call",
                        timestamp=time.time(),
                        node_name="LLMReasonerNode",
                        status=NodeStatus.COMPLETED,
                        session_id=session_id,
                        task_id=f"meta_tool_{tool_name}_{i + 1}",
                        node_duration=meta_tool_duration,
                        success=True,
                        metadata=execution_details
                    ))

            except Exception as e:
                meta_tool_duration = time.perf_counter() - meta_tool_start
                error_details = {
                    "meta_tool_name": tool_name,
                    "execution_success": False,
                    "execution_duration": meta_tool_duration,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "reasoning_loop": self.current_loop_count,
                    "outline_step": self.current_outline_step if hasattr(self, 'current_outline_step') else 0,
                    "parsed_args": args if 'args' in locals() else None,
                    "raw_args_string": args_str,
                    "execution_phase": "meta_tool_error",
                    "context_size_at_error": len(self.reasoning_context),
                    "task_stack_size_at_error": len(self.internal_task_stack),
                    "tool_category": self._get_tool_category(tool_name),
                    "error_context": self._get_error_context(e),
                    "recovery_recommended": self.auto_recovery_attempts < getattr(self, 'max_auto_recovery', 3)
                }

                if progress_tracker:
                    await progress_tracker.emit_event(ProgressEvent(
                        event_type="meta_tool_call",
                        timestamp=time.time(),
                        node_name="LLMReasonerNode",
                        status=NodeStatus.FAILED,
                        session_id=session_id,
                        task_id=f"meta_tool_{tool_name}_{i + 1}",
                        node_duration=meta_tool_duration,
                        success=False,
                        metadata=error_details
                    ))

                eprint(f"Meta-tool execution failed for {tool_name}: {e}")
                result["context_addition"] = f"Error executing {tool_name}: {str(e)}"

                self._add_context_to_reasoning(result["context_addition"], tool_name, execution_details)

        # Final summary event if multiple meta-tools were processed
        if len(matches) > 1 and progress_tracker:
            batch_performance = self._calculate_batch_performance(matches)
            reasoning_progress = self._assess_reasoning_progress()

            await progress_tracker.emit_event(ProgressEvent(
                event_type="meta_tool_batch_complete",
                timestamp=time.time(),
                node_name="LLMReasonerNode",
                status=NodeStatus.COMPLETED,
                session_id=session_id,
                metadata={
                    "total_meta_tools_processed": len(matches),
                    "reasoning_loop": self.current_loop_count,
                    "outline_step": self.current_outline_step if hasattr(self, 'current_outline_step') else 0,
                    "batch_execution_complete": True,
                    "final_context_size": len(self.reasoning_context),
                    "final_task_stack_size": len(self.internal_task_stack),
                    "meta_tools_executed": [match[0] for match in matches],
                    "execution_phase": "meta_tool_batch_summary",
                    "batch_performance": batch_performance,
                    "reasoning_progress": reasoning_progress,
                    "progress_made": result["progress_made"],
                    "action_taken": result["action_taken"],
                    "outline_status": {
                        "current_step": self.current_outline_step if hasattr(self, 'current_outline_step') else 0,
                        "total_steps": len(self.outline.get('steps', [])) if self.outline else 0,
                        "completion_ratio": (
                                self.current_outline_step / len(self.outline.get('steps', [1]))) if self.outline else 0
                    },
                    "performance_summary": {
                        "loop_efficiency": self.performance_metrics.get("action_efficiency", 0) if hasattr(self,
                                                                                                           'performance_metrics') else 0,
                        "recovery_attempts": getattr(self, 'auto_recovery_attempts', 0),
                        "context_management_active": len(self.reasoning_context) >= getattr(self,
                                                                                            'context_summary_threshold',
                                                                                            15)
                    }
                }
            ))

        return result

    async def _execute_enhanced_internal_reasoning(self, args: dict, prep_res: dict) -> dict[str, Any]:
        """Enhanced internal reasoning with outline step tracking"""
        # Standard internal reasoning execution
        result = await self._execute_internal_reasoning(args, prep_res)

        # Enhanced with outline step progress
        outline_step_progress = args.get("outline_step_progress", "")
        if outline_step_progress and result:
            result["context_addition"] += f"\nOutline Step Progress: {outline_step_progress}"

        # Track reasoning depth for current step
        if not hasattr(self, '_step_reasoning_depth'):
            self._step_reasoning_depth = {}

        current_step = self.current_outline_step
        self._step_reasoning_depth[current_step] = self._step_reasoning_depth.get(current_step, 0) + 1

        # Warn if too much reasoning without action
        if self._step_reasoning_depth[current_step] > 3:
            result["context_addition"] += "\n⚠️ WARNING: Too much reasoning without concrete action for current step"

        return result

    async def _execute_enhanced_task_stack(self, args: dict, prep_res: dict) -> dict[str, Any]:
        """Enhanced task stack management with outline step tracking"""
        # Get outline step reference
        outline_step_ref = args.get("outline_step_ref", f"step_{self.current_outline_step}")

        # Execute standard task stack management
        result = await self._execute_manage_task_stack(args, prep_res)

        # Enhanced with outline step reference
        if result:
            result["context_addition"] += f"\n[Linked to: {outline_step_ref}]"

        return result

    async def _execute_enhanced_delegate_llm_tool(self, args: dict, prep_res: dict) -> dict[str, Any]:
        """Enhanced delegation with immediate result visibility and guaranteed storage"""
        task_description = args.get("task_description", "")
        tools_list = args.get("tools_list", [])
        outline_step_completion = args.get("outline_step_completion", False)

        # Generate unique delegation ID for this execution
        delegation_id = f"delegation_loop_{self.current_loop_count}"

        # Prepare shared state for LLMToolNode with enhanced result capture
        llm_tool_shared = {
            "current_task_description": task_description,
            "current_query": task_description,
            "formatted_context": {
                "recent_interaction": f"Reasoner delegating task: {task_description}",
                "session_summary": self._get_reasoning_summary(),
                "task_context": f"Loop {self.current_loop_count} delegation - CAPTURE ALL RESULTS"
            },
            "variable_manager": prep_res.get("variable_manager"),
            "agent_instance": prep_res.get("agent_instance"),
            "available_tools": tools_list,
            "tool_capabilities": prep_res.get("tool_capabilities", {}),
            "fast_llm_model": prep_res.get("fast_llm_model"),
            "complex_llm_model": prep_res.get("complex_llm_model"),
            "progress_tracker": prep_res.get("progress_tracker"),
            "session_id": prep_res.get("session_id"),
            "use_fast_response": True
        }

        try:
            # Execute LLMToolNode
            llm_tool_node = LLMToolNode()
            await llm_tool_node.run_async(llm_tool_shared)

            # IMMEDIATE RESULT EXTRACTION - Critical for visibility
            final_response = llm_tool_shared.get("current_response", "No response captured")
            tool_calls_made = llm_tool_shared.get("tool_calls_made", 0)
            tool_results = llm_tool_shared.get("results", {})

            # GUARANTEED STORAGE - Multiple storage patterns for reliability
            delegation_result = {
                "task_description": task_description,
                "tools_used": tools_list,
                "tool_calls_made": tool_calls_made,
                "final_response": final_response,
                "results": tool_results,
                "timestamp": datetime.now().isoformat(),
                "delegation_id": delegation_id,
                "outline_step": self.current_outline_step,
                "reasoning_loop": self.current_loop_count,
                "success": True
            }

            # CRITICAL: Store immediately with multiple access patterns
            if self.variable_manager:
                # 1. Primary delegation storage
                self.variable_manager.set(f"delegation.loop_{self.current_loop_count}", delegation_result)

                # 2. Latest results quick access
                self.variable_manager.set("delegation.latest", delegation_result)

                # 3. Store individual tool results with direct access
                for result_id, result_data in tool_results.items():
                    self.variable_manager.set(f"results.{result_id}.data", result_data.get("data"))

                # 4. Create smart access keys for common patterns
                if "read_file" in tools_list and tool_results:
                    file_content = next((res.get("data") for res in tool_results.values()
                                         if res.get("data") and isinstance(res.get("data"), str)), None)
                    if file_content:
                        self.variable_manager.set("var.file_content", file_content)
                        self.variable_manager.set("latest_file_content", file_content)

                # 5. Update delegation index for discovery
                index = self.variable_manager.get("delegation.index", [])
                index.append({
                    "loop": self.current_loop_count,
                    "task": task_description[:100],
                    "tools": tools_list,
                    "timestamp": datetime.now().isoformat(),
                    "results_available": len(tool_results) > 0
                })
                self.variable_manager.set("delegation.index", index[-20:])

            # Create comprehensive context addition with IMMEDIATE VISIBILITY
            context_addition = f"""DELEGATION COMPLETED (Loop {self.current_loop_count}):
Task: {task_description}
Tools: {', '.join(tools_list)}
Calls Made: {tool_calls_made}
Results Captured: {len(tool_results)} items

FINAL RESULT: {final_response}

- reference variable: delegation.loop_{self.current_loop_count}
DELEGATION END
"""

            # Mark outline step completion if specified
            if outline_step_completion:
                await self._mark_step_completion(prep_res, "delegation_complete", context_addition)

            return {"context_addition": context_addition}

        except Exception as e:
            error_msg = f"❌ DELEGATION FAILED: {str(e)}"
            # Store error for debugging
            if self.variable_manager:
                error_data = {
                    "task": task_description,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "loop": self.current_loop_count
                }
                self.variable_manager.set(f"delegation.error.loop_{self.current_loop_count}", error_data)

            return {"context_addition": error_msg}

    async def _execute_enhanced_create_plan(self, args: dict, prep_res: dict) -> dict[str, Any]:
        """Enhanced plan creation with outline step completion tracking"""
        # Check if this completes the outline step
        outline_step_completion = args.get("outline_step_completion", False)

        # Execute standard plan creation
        result = await self._execute_create_plan(args, prep_res)

        # Enhanced with step completion tracking
        if outline_step_completion and result:
            await self._mark_step_completion(prep_res, "create_and_execute_plan", result["context_addition"])
            result["context_addition"] += f"\n✓ OUTLINE STEP {self.current_outline_step + 1} COMPLETED"

        return result

    async def _execute_advance_outline_step(self, args: dict, prep_res: dict) -> dict[str, Any]:
        """Execute outline step advancement"""
        step_completed = args.get("step_completed", False)
        completion_evidence = args.get("completion_evidence", "")
        next_step_focus = args.get("next_step_focus", "")

        if not self.outline or not self.outline.get("steps"):
            return {"context_addition": "Cannot advance: No outline available"}

        steps = self.outline["steps"]

        if self.current_outline_step >= len(steps):
            return {"context_addition": "Cannot advance: Already at final step"}

        if step_completed:
            # Mark current step as completed
            if self.current_outline_step < len(steps):
                current_step = steps[self.current_outline_step]
                current_step["status"] = "completed"
                current_step["completion_evidence"] = completion_evidence
                current_step["completed_at"] = datetime.now().isoformat()

            # Advance to next step
            self.current_outline_step += 1

            # Store advancement in variables
            if self.variable_manager:
                advancement_data = {
                    "step_completed": self.current_outline_step,
                    "completion_evidence": completion_evidence,
                    "next_step_focus": next_step_focus,
                    "timestamp": datetime.now().isoformat()
                }
                self.variable_manager.set(f"reasoning.step_completions.{self.current_outline_step - 1}",
                                          advancement_data)

            context_addition = f"""✓ STEP {self.current_outline_step} COMPLETED
Evidence: {completion_evidence}
Advanced to Step {self.current_outline_step + 1}/{len(steps)}"""

            if next_step_focus:
                context_addition += f"\nNext Step Focus: {next_step_focus}"

            if self.current_outline_step >= len(steps):
                context_addition += "\n🎯 ALL OUTLINE STEPS COMPLETED - Ready for direct_response"

        else:
            context_addition = f"Step {self.current_outline_step + 1} not yet completed - continue working on current step"

        return {"context_addition": context_addition}

    async def _execute_read_from_variables(self, args: dict) -> dict[str, Any]:
        """Enhanced variable reading with intelligent discovery and loop prevention"""
        if not self.variable_manager:
            return {"context_addition": "❌ Variable system not available"}

        scope = args.get("scope", args.get("query", "reasoning"))
        key = args.get("key", "")
        purpose = args.get("purpose", "")

        # CRITICAL: Check for repeated reads - prevent infinite loops
        read_signature = f"{scope}.{key}"
        if not hasattr(self, '_variable_read_history'):
            self._variable_read_history = []

        # Prevent reading same variable multiple times in short succession
        recent_reads = [r for r in self._variable_read_history if r['signature'] == read_signature]
        if len(recent_reads) >= 2:
            self._variable_read_history.append({
                'signature': read_signature,
                'timestamp': time.time(),
                'loop': self.current_loop_count
            })
            return {
                "context_addition": f"⚠️ LOOP PREVENTION: Already read {read_signature} {len(recent_reads)} times. Try different approach or advance to next task."
            }

        # Record this read attempt
        self._variable_read_history.append({
            'signature': read_signature,
            'timestamp': time.time(),
            'loop': self.current_loop_count
        })

        # Clean old read history (keep last 10)
        if len(self._variable_read_history) > 10:
            self._variable_read_history = self._variable_read_history[-10:]

        if not key:
            return {"context_addition": "❌ Cannot read: No key provided"}

        try:
            # Smart key resolution for common patterns
            resolved_key = self._resolve_smart_key(scope, key)

            # Try direct access first
            value = self.variable_manager.get(resolved_key)

            if value is not None:
                # Format value for display
                value_display = self._format_variable_value(value)

                context_addition = f"""{resolved_key}={value_display}
Access: Successfully retrieved from variable system"""

                return {"context_addition": context_addition}

            else:
                # Enhanced discovery when not found
                discovery_result = self._perform_smart_variable_discovery(scope, key, purpose)
                return {"context_addition": discovery_result}

        except Exception as e:
            return {"context_addition": f"❌ Variable read error: {str(e)}"}

    def _resolve_smart_key(self, scope: str, key: str) -> str:
        """Resolve smart key patterns for common access cases"""
        # Handle delegation results specially
        if scope == "delegation" and "loop_" in key:
            return f"delegation.{key}"
        elif scope == "results" and key.endswith(".data"):
            return f"results.{key}"
        elif scope == "var" or key.startswith("var."):
            return key if key.startswith("var.") else f"var.{key}"
        else:
            return f"{scope}.{key}" if scope != "reasoning" else f"reasoning.{key}"

    def _format_variable_value(self, value: any) -> str:
        """Format variable value for display with intelligent truncation"""
        if isinstance(value, dict | list):
            value_str = json.dumps(value, default=str, indent=2)
        else:
            value_str = str(value)

        # Smart truncation based on content type
        if len(value_str) > 200000:
            if isinstance(value, dict) and "results" in str(value):
                # For result dicts, show structure
                return f"RESULTS DICT ({len(value)} keys):\n" + value_str[:150000] + "\n... [TRUNCATED]"
            elif isinstance(value, str) and (value.startswith("# ") or "markdown" in value.lower()):
                # For file content, show beginning
                return f"FILE CONTENT ({len(value_str)} chars):\n" + value_str[:100000] + "\n... [FULL CONTENT AVAILABLE]"
            else:
                return value_str[:100000] + f"\n... [TRUNCATED - {len(value_str)} total chars]"

        return value_str

    def _perform_smart_variable_discovery(self, scope: str, key: str, purpose: str) -> str:
        """Perform intelligent variable discovery when key not found"""
        # Check latest delegation results first
        latest = self.variable_manager.get("delegation.latest")
        if latest:
            discovery_msg = f"❌ Variable not found: {scope}.{key}\n\n✨ LATEST DELEGATION RESULTS AVAILABLE:"
            discovery_msg += f"\nTask: {latest.get('task_description', 'Unknown')[:100]}"
            discovery_msg += f"\nResults: {len(latest.get('results', {}))} items available"
            discovery_msg += "\nAccess with: delegation.latest"

            # Show actual keys available
            if latest.get('results'):
                discovery_msg += "\n\n🔍 Available result keys:"
                for result_id in latest['results']:
                    discovery_msg += f"\n• results.{result_id}.data"

            return discovery_msg

        # Check delegation index for recent activity
        index = self.variable_manager.get("delegation.index", [])
        if index:
            recent = index[-3:]  # Last 3 delegations
            discovery_msg = f"❌ Variable not found: {scope}.{key}\n\n📚 RECENT DELEGATIONS:"
            for entry in recent:
                discovery_msg += f"\n• Loop {entry['loop']}: {entry['task'][:50]}..."
                discovery_msg += f"  Access: delegation.loop_{entry['loop']}"
            return discovery_msg

        # Fallback: show available scopes
        available_vars = self.variable_manager.get_available_variables()
        return f"❌ Variable not found: {scope}.{key}\n\n📋 Available scopes: {', '.join(available_vars.keys())}"


    async def _execute_write_to_variables(self, args: dict) -> dict[str, Any]:
        """Enhanced variable writing with automatic result storage"""
        if not self.variable_manager:
            return {"context_addition": "❌ Variable system not available"}

        scope = args.get("scope", "reasoning")
        key = args.get("key", "")
        value = args.get("value", "")
        description = args.get("description", "")

        if not key:
            return {"context_addition": "❌ Cannot write to variables: No key provided"}

        try:
            # Create scoped key
            full_key = f"{scope}.{key}" if scope != "reasoning" else f"reasoning.{key}"

            # Write to variables
            self.variable_manager.set(full_key, value)

            # Store enhanced metadata
            metadata = {
                "description": description,
                "written_at": datetime.now().isoformat(),
                "outline_step": getattr(self, 'current_outline_step', 0),
                "reasoning_loop": self.current_loop_count,
                "value_type": type(value).__name__,
                "value_size": len(str(value)) if value else 0,
                "auto_stored": False  # Manual storage
            }
            self.variable_manager.set(f"{full_key}_metadata", metadata)

            # Update storage index for easy discovery
            storage_index = self.variable_manager.get("reasoning.storage_index", [])
            storage_entry = {
                "key": full_key,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "loop": self.current_loop_count
            }
            storage_index.append(storage_entry)
            self.variable_manager.set("reasoning.storage_index", storage_index[-20:])  # Keep last 20

            context_addition = f"✅ Stored in variables: {full_key}"
            if description:
                context_addition += f"\n📄 Description: {description}"

            # Show how to access it
            context_addition += f"\n🔍 Access with: read_from_variables(scope=\"{scope}\", key=\"{key}\", purpose=\"...\")"

            return {"context_addition": context_addition}

        except Exception as e:
            return {"context_addition": f"❌ Failed to write to variables: {str(e)}"}

    def _auto_store_delegation_results(self, delegation_result: dict, task_description: str) -> str:
        """Automatically store delegation results with smart naming and comprehensive indexing"""
        if not self.variable_manager:
            return "\n❌ Variable system not available for auto-storage"

        storage_summary = []

        try:
            # Store main delegation result with loop reference
            main_key = f"delegation.loop_{self.current_loop_count}"
            self.variable_manager.set(main_key, delegation_result)
            storage_summary.append(f"• {main_key}")

            # Store individual tool results with smart naming
            results = delegation_result.get("results", {})
            smart_keys_created = []

            for result_id, result_data in results.items():
                # Smart naming based on task content and result type
                smart_key = self._generate_smart_key(task_description, result_id, result_data)

                # Store full result
                self.variable_manager.set(smart_key, result_data)
                storage_summary.append(f"• {smart_key}")
                smart_keys_created.append(smart_key)

                # Store data separately for direct access
                if result_data.get("data"):
                    data_key = f"{smart_key}.data"
                    self.variable_manager.set(data_key, result_data["data"])
                    storage_summary.append(f"• {data_key} (direct access)")

                    # Store with generic access pattern
                    generic_data_key = f"results.{result_id}.data"
                    self.variable_manager.set(generic_data_key, result_data["data"])
                    storage_summary.append(f"• {generic_data_key} (standard access)")

            # Update comprehensive quick access index
            quick_access = {
                "latest_delegation": main_key,
                "latest_task": task_description,
                "timestamp": datetime.now().isoformat(),
                "loop": self.current_loop_count,
                "outline_step": getattr(self, 'current_outline_step', 0),
                "stored_keys": [item.replace("• ", "") for item in storage_summary],
                "smart_keys": smart_keys_created,
                "access_patterns": {
                    "main_result": main_key,
                    "by_loop": f"delegation.loop_{self.current_loop_count}",
                    "latest": "reasoning.latest_results",
                    "data_direct": [key for key in storage_summary if ".data" in key]
                }
            }
            self.variable_manager.set("reasoning.latest_results", quick_access)

            # Update global delegation index for easy discovery
            delegation_index = self.variable_manager.get("delegation.index", [])
            index_entry = {
                "loop": self.current_loop_count,
                "task": task_description[:100] + ("..." if len(task_description) > 100 else ""),
                "keys_created": len(storage_summary),
                "timestamp": datetime.now().isoformat(),
                "main_key": main_key,
                "smart_keys": smart_keys_created
            }
            delegation_index.append(index_entry)
            self.variable_manager.set("delegation.index", delegation_index[-50:])  # Keep last 50

            # Store task-specific quick access
            task_hash = hash(task_description) % 10000
            self.variable_manager.set(f"delegation.by_task.{task_hash}", {
                "task_description": task_description,
                "results": quick_access,
                "created_at": datetime.now().isoformat()
            })

            return f"\n📊 Auto-stored results ({len(storage_summary)} entries):\n" + "\n".join(storage_summary[:8]) + (
                f"\n... +{len(storage_summary) - 8} more" if len(storage_summary) > 8 else "")

        except Exception as e:
            return f"\n❌ Auto-storage failed: {str(e)}"

    def _generate_smart_key(self, task_description: str, result_id: str, result_data: dict) -> str:
        """Generate intelligent storage keys based on task content and result type"""
        task_lower = task_description.lower()

        # Analyze task type
        if "read" in task_lower and "file" in task_lower:
            prefix = "file_content"
        elif "write" in task_lower and "file" in task_lower:
            prefix = "file_written"
        elif "create" in task_lower and "file" in task_lower:
            prefix = "file_created"
        elif "search" in task_lower or "find" in task_lower:
            prefix = "search_results"
        elif "analyze" in task_lower or "analysis" in task_lower:
            prefix = "analysis_results"
        elif "list" in task_lower or "directory" in task_lower:
            prefix = "directory_listing"
        elif "download" in task_lower or "fetch" in task_lower:
            prefix = "downloaded_content"
        else:
            # Analyze result data for hints
            result_str = str(result_data).lower()
            if "file" in result_str and "content" in result_str:
                prefix = "file_content"
            elif "search" in result_str or "results" in result_str:
                prefix = "search_results"
            elif "data" in result_str:
                prefix = "task_data"
            else:
                prefix = "task_result"

        # Create unique key with loop and result ID
        return f"{prefix}.loop_{self.current_loop_count}_{result_id}"

    async def _mark_step_completion(self, prep_res: dict, method: str, evidence: str):
        """Mark current outline step as completed"""
        if not self.outline or not self.outline.get("steps"):
            return

        steps = self.outline["steps"]
        if self.current_outline_step < len(steps):
            current_step = steps[self.current_outline_step]
            current_step["status"] = "completed"
            current_step["completion_method"] = method
            current_step["completion_evidence"] = evidence
            current_step["completed_at"] = datetime.now().isoformat()

            # Store in variables
            if self.variable_manager:
                completion_data = {
                    "step_number": self.current_outline_step,
                    "description": current_step.get("description", ""),
                    "method": method,
                    "evidence": evidence,
                    "timestamp": datetime.now().isoformat()
                }
                self.variable_manager.set(f"reasoning.step_completions.{self.current_outline_step}", completion_data)

    async def _store_successful_completion(self, prep_res: dict, final_answer: str):
        """Store successful completion data for future learning"""
        if not self.variable_manager:
            return

        success_data = {
            "query": prep_res["original_query"],
            "final_answer": final_answer,
            "reasoning_loops": self.current_loop_count,
            "outline": self.outline,
            "performance_metrics": self.performance_metrics,
            "auto_recovery_attempts": self.auto_recovery_attempts,
            "completion_timestamp": datetime.now().isoformat(),
            "session_id": prep_res.get("session_id", "default")
        }

        # Store in successful patterns
        successes = self.variable_manager.get("reasoning.successful_patterns", [])
        successes.append(success_data)
        self.variable_manager.set("reasoning.successful_patterns", successes[-20:])  # Keep last 20

        # Update performance statistics
        self._update_success_statistics()

    def _update_success_statistics(self):
        """Update success statistics in variables"""
        if not self.variable_manager:
            return

        # Get current stats
        current_stats = self.variable_manager.get("reasoning.performance.statistics", {})

        # Update stats
        current_stats["total_successful_sessions"] = current_stats.get("total_successful_sessions", 0) + 1
        current_stats["avg_loops_per_success"] = current_stats.get("avg_loops_per_success", 0)

        # Calculate new average
        total_sessions = current_stats["total_successful_sessions"]
        old_avg = current_stats["avg_loops_per_success"] * (total_sessions - 1)
        current_stats["avg_loops_per_success"] = (old_avg + self.current_loop_count) / total_sessions

        # Store updated stats
        self.variable_manager.set("reasoning.performance.statistics", current_stats)

    async def _create_outline_completion_response(self, prep_res: dict) -> str:
        """Create response when outline is completed"""
        if not self.outline:
            return "Outline completion response requested but no outline available"

        steps = self.outline.get("steps", [])
        completed_steps = [s for s in steps if
                           s.get("status") in ["completed", "force_completed", "emergency_completed"]]

        response_parts = []
        response_parts.append("I have completed the structured approach outlined for your request:")

        # Summarize completed steps
        for i, step in enumerate(completed_steps):
            status_indicator = "✓" if step.get("status") == "completed" else "⚠️"
            response_parts.append(f"{status_indicator} Step {i + 1}: {step.get('description', 'Unknown step')}")

            # Add evidence if available
            evidence = step.get("completion_evidence", "")
            if evidence and len(evidence) < 200:
                response_parts.append(f"   Result: {evidence}")

        # Get final results from variables if available
        if self.variable_manager:
            final_results = self.variable_manager.get("reasoning.final_results", {})
            if final_results:
                response_parts.append("\nKey findings:")
                for key, value in final_results.items():
                    if isinstance(value, str) and len(value) < 300:
                        response_parts.append(f"- {key}: {value}")

        response_parts.append(
            f"\nCompleted in {self.current_loop_count} reasoning cycles using outline-driven execution.")

        return "\n".join(response_parts)

    async def _create_enhanced_timeout_response(self, query: str, prep_res: dict) -> str:
        """Create enhanced timeout response with comprehensive progress summary"""
        response_parts = []
        response_parts.append(
            f"I reached my reasoning limit of {self.max_reasoning_loops} steps while working on: {query}")

        # Outline progress
        if self.outline:
            steps = self.outline.get("steps", [])
            completed_steps = [s for s in steps if
                               s.get("status") in ["completed", "force_completed", "emergency_completed"]]
            unfinished_steps = [s for s in steps if s not in completed_steps]

            response_parts.append(f"\nOutline Progress: {len(completed_steps)}/{len(steps)} steps completed")

            if completed_steps:
                response_parts.append("Completed steps:")
                for i, step in enumerate(completed_steps):
                    response_parts.append(f"✓ {step.get('description', f'Step {i + 1}')}")

            if unfinished_steps:
                response_parts.append("Unfinished steps:")
                for i, step in enumerate(unfinished_steps):
                    response_parts.append(f"✗ {step.get('description', f'Step {i + 1}')}")

        # Task stack progress
        if self.internal_task_stack:
            completed_tasks = [t for t in self.internal_task_stack if t.get("status") == "completed"]
            pending_tasks = [t for t in self.internal_task_stack if t.get("status") == "pending"]

            response_parts.append(f"\nTask Progress: {len(completed_tasks)} completed, {len(pending_tasks)} pending")

        # Performance metrics
        if self.performance_metrics:
            response_parts.append(
                f"\nPerformance: {self.performance_metrics.get('action_efficiency', 0):.1%} efficiency, {self.auto_recovery_attempts} recovery attempts")

        # Available results from variables
        if self.variable_manager:
            reasoning_results = self.variable_manager.get("reasoning", {})
            if reasoning_results:
                response_parts.append(f"\nStored findings: {len(reasoning_results)} entries in reasoning variables")

        return "\n".join(response_parts)

    async def _finalize_reasoning_session(self, prep_res: dict, final_result: str):
        """Finalize reasoning session with comprehensive data storage"""
        if not self.variable_manager:
            return

        # Store session completion data
        session_data = {
            "query": prep_res["original_query"],
            "final_result": final_result,
            "reasoning_loops": self.current_loop_count,
            "outline_completion": self.current_outline_step,
            "performance_metrics": self.performance_metrics,
            "auto_recovery_attempts": self.auto_recovery_attempts,
            "context_summaries": len([c for c in self.reasoning_context if c.get("type") == "context_summary"]),
            "completion_timestamp": datetime.now().isoformat(),
            "session_duration": time.time() - time.mktime(datetime.now().timetuple()),
            "success": True
        }

        # Store in session history
        session_history = self.variable_manager.get("reasoning.session_history", [])
        session_history.append(session_data)
        self.variable_manager.set("reasoning.session_history", session_history[-50:])  # Keep last 50 sessions

        # Store outline pattern for reuse
        if self.outline:
            outline_pattern = {
                "query_type": self._classify_query_type(prep_res["original_query"]),
                "outline": self.outline,
                "success": True,
                "loops_used": self.current_loop_count,
                "timestamp": datetime.now().isoformat()
            }
            patterns = self.variable_manager.get("reasoning.successful_patterns.outlines", [])
            patterns.append(outline_pattern)
            self.variable_manager.set("reasoning.successful_patterns.outlines", patterns[-10:])

    def _classify_query_type(self, query: str) -> str:
        """Classify query type for pattern matching"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["search", "find", "look up", "research"]):
            return "research"
        elif any(word in query_lower for word in ["analyze", "compare", "evaluate"]):
            return "analysis"
        elif any(word in query_lower for word in ["create", "generate", "write", "build"]):
            return "creation"
        elif any(word in query_lower for word in ["plan", "strategy", "approach"]):
            return "planning"
        else:
            return "general"

    async def _handle_reasoning_error(self, error: Exception, prep_res: dict, progress_tracker):
        """Enhanced error handling with auto-recovery"""
        eprint(f"Reasoning loop {self.current_loop_count} failed: {error}")

        # Store error in context
        self.reasoning_context.append({
            "type": "error",
            "content": f"Error in loop {self.current_loop_count}: {str(error)}",
            "error_type": type(error).__name__,
            "outline_step": self.current_outline_step,
            "timestamp": datetime.now().isoformat()
        })

        # Store in variables for learning
        if self.variable_manager:
            error_data = {
                "error": str(error),
                "error_type": type(error).__name__,
                "loop": self.current_loop_count,
                "outline_step": self.current_outline_step,
                "timestamp": datetime.now().isoformat(),
                "query": prep_res["original_query"]
            }
            errors = self.variable_manager.get("reasoning.error_log", [])
            errors.append(error_data)
            self.variable_manager.set("reasoning.error_log", errors[-100:])  # Keep last 100 errors

        # Trigger auto-recovery if not already in recovery
        if self.auto_recovery_attempts < self.max_auto_recovery:
            await self._trigger_auto_recovery(prep_res)

    # Keep all existing helper methods like _execute_internal_reasoning, etc.
    # but update them to use the enhanced variable system...

    async def post_async(self, shared, prep_res, exec_res):
        """Enhanced post-processing with comprehensive data storage"""
        final_result = exec_res.get("final_result", "Task processing incomplete")

        # Store comprehensive reasoning artifacts
        shared["reasoning_artifacts"] = {
            "reasoning_loops": exec_res.get("reasoning_loops", 0),
            "reasoning_context": exec_res.get("reasoning_context", []),
            "internal_task_stack": exec_res.get("internal_task_stack", []),
            "outline": exec_res.get("outline"),
            "outline_completion": exec_res.get("outline_completion", 0),
            "performance_metrics": exec_res.get("performance_metrics", {}),
            "auto_recovery_attempts": exec_res.get("auto_recovery_attempts", 0)
        }

        # Enhanced variable system updates
        if self.variable_manager:
            # Store final session results
            final_session_data = {
                "final_result": final_result,
                "completion_timestamp": datetime.now().isoformat(),
                "total_loops": exec_res.get("reasoning_loops", 0),
                "session_success": final_result != "Task processing incomplete",
                "outline_driven_execution": True
            }
            self.variable_manager.set("reasoning.current_session.final_data", final_session_data)

            # Update global performance statistics
            self._update_global_performance_stats(exec_res)

        # Set enhanced response data
        shared["llm_reasoner_result"] = final_result
        shared["current_response"] = final_result

        # Provide enhanced synthesis metadata
        shared["synthesized_response"] = {
            "synthesized_response": final_result,
            "confidence": self._calculate_confidence(exec_res),
            "metadata": {
                "synthesis_method": "outline_driven_reasoner",
                "reasoning_loops": exec_res.get("reasoning_loops", 0),
                "outline_completion": exec_res.get("outline_completion", 0),
                "performance_score": self._calculate_performance_score(exec_res),
                "auto_recovery_used": exec_res.get("auto_recovery_attempts", 0) > 0
            }
        }

        return "reasoner_complete"

    def _update_global_performance_stats(self, exec_res: dict):
        """Update global performance statistics in variables"""
        if not self.variable_manager:
            return

        stats = self.variable_manager.get("reasoning.global_performance", {})

        # Update counters
        stats["total_sessions"] = stats.get("total_sessions", 0) + 1
        stats["total_loops"] = stats.get("total_loops", 0) + exec_res.get("reasoning_loops", 0)
        stats["total_recoveries"] = stats.get("total_recoveries", 0) + exec_res.get("auto_recovery_attempts", 0)

        # Calculate averages
        stats["avg_loops_per_session"] = stats["total_loops"] / stats["total_sessions"]
        stats["recovery_rate"] = stats["total_recoveries"] / stats["total_sessions"]

        # Success tracking
        if exec_res.get("final_result") != "Task processing incomplete":
            stats["successful_sessions"] = stats.get("successful_sessions", 0) + 1
            stats["success_rate"] = stats["successful_sessions"] / stats["total_sessions"]

        self.variable_manager.set("reasoning.global_performance", stats)

    def _calculate_confidence(self, exec_res: dict) -> float:
        """Calculate confidence score based on execution results"""
        base_confidence = 0.5

        # Outline completion boosts confidence
        outline = exec_res.get("outline")
        if outline:
            completion_ratio = exec_res.get("outline_completion", 0) / len(outline.get("steps", [1]))
            base_confidence += 0.3 * completion_ratio

        # Low recovery attempts boost confidence
        recovery_attempts = exec_res.get("auto_recovery_attempts", 0)
        if recovery_attempts == 0:
            base_confidence += 0.15
        elif recovery_attempts == 1:
            base_confidence += 0.05

        # Reasonable loop count boosts confidence
        loops = exec_res.get("reasoning_loops", 0)
        if 3 <= loops <= 15:
            base_confidence += 0.1

        # Performance metrics
        performance = exec_res.get("performance_metrics", {})
        if performance.get("action_efficiency", 0) > 0.7:
            base_confidence += 0.1

        return min(1.0, max(0.0, base_confidence))

    def _calculate_performance_score(self, exec_res: dict) -> float:
        """Calculate overall performance score"""
        score = 0.5

        # Efficiency score
        performance = exec_res.get("performance_metrics", {})
        action_efficiency = performance.get("action_efficiency", 0)
        score += 0.3 * action_efficiency

        # Completion score
        outline = exec_res.get("outline")
        if outline:
            completion_ratio = exec_res.get("outline_completion", 0) / len(outline.get("steps", [1]))
            score += 0.4 * completion_ratio

        # Recovery penalty
        recovery_attempts = exec_res.get("auto_recovery_attempts", 0)
        score -= 0.1 * recovery_attempts

        return min(1.0, max(0.0, score))


    def _summarize_reasoning_context(self) -> str:
        """Summarize the current reasoning context"""
        if not self.reasoning_context:
            return "No previous reasoning steps"

        summary_parts = []
        for entry in self.reasoning_context[-5:]:  # Last 5 entries
            entry_type = entry.get("type", "unknown")
            content = entry.get("content", "")

            if entry_type == "reasoning":
                # Truncate long reasoning content
                content_preview = content[:20000] + "..." if len(content) > 20000 else content
                summary_parts.append(f"Loop {entry.get('loop', '?')}: {content_preview}")
            elif entry_type == "meta_tool_result":
                summary_parts.append(f"Result: {content[:150]}...")
            elif entry_type == "error":
                summary_parts.append(f"Error: {content}")

        return "\n".join(summary_parts)

    def _summarize_task_stack(self) -> str:
        """Summarize the internal task stack"""
        if not self.internal_task_stack:
            return "No tasks in stack"

        summary_parts = []
        for i, task in enumerate(self.internal_task_stack):
            status = task.get("status", "pending")
            description = task.get("description", "No description")
            summary_parts.append(f"{i + 1}. [{status.upper()}] {description}")

        return "\n".join(summary_parts)

    def _get_tool_category(self, tool_name: str) -> str:
        """Get category for meta-tool"""
        categories = {
            "internal_reasoning": "thinking",
            "manage_internal_task_stack": "planning",
            "delegate_to_llm_tool_node": "delegation",
            "create_and_execute_plan": "orchestration",
            "direct_response": "completion"
        }
        return categories.get(tool_name, "unknown")

    def _calculate_reasoning_depth(self) -> int:
        """Calculate current reasoning depth"""
        reasoning_entries = [entry for entry in self.reasoning_context if entry.get("type") == "reasoning"]
        return len(reasoning_entries)

    def _assess_delegation_complexity(self, args: dict) -> str:
        """Assess complexity of delegation task"""
        task_desc = args.get("task_description", "")
        tools_count = len(args.get("tools_list", []))

        if tools_count > 3 or len(task_desc) > 100:
            return "high"
        elif tools_count > 1 or len(task_desc) > 50:
            return "medium"
        else:
            return "low"

    def _estimate_plan_complexity(self, goals: list) -> str:
        """Estimate complexity of plan"""
        goal_count = len(goals)
        total_text = sum(len(str(goal)) for goal in goals)

        if goal_count > 5 or total_text > 500:
            return "high"
        elif goal_count > 2 or total_text > 200:
            return "medium"
        else:
            return "low"

    def _calculate_tool_performance_score(self, duration: float, tool_name: str) -> float:
        """Calculate performance score for tool execution"""
        # Expected durations by tool type
        expected_durations = {
            "internal_reasoning": 0.1,
            "manage_internal_task_stack": 0.05,
            "delegate_to_llm_tool_node": 3.0,
            "create_and_execute_plan": 10.0,
            "direct_response": 0.1
        }

        expected = expected_durations.get(tool_name, 1.0)
        if duration <= expected:
            return 1.0
        else:
            return max(0.0, expected / duration)

    def _create_reasoning_summary(self) -> str:
        """Create summary of reasoning process"""
        reasoning_entries = [entry for entry in self.reasoning_context if entry.get("type") == "reasoning"]
        task_entries = len(self.internal_task_stack)

        return f"Completed {len(reasoning_entries)} reasoning steps with {task_entries} tasks tracked"

    def _calculate_batch_performance(self, matches: list) -> dict[str, Any]:
        """Calculate performance metrics for batch execution"""
        tool_types = [match[0] for match in matches]
        return {
            "total_tools": len(matches),
            "tool_diversity": len(set(tool_types)),
            "most_used_tool": max(set(tool_types), key=tool_types.count) if tool_types else "none"
        }

    def _assess_reasoning_progress(self) -> str:
        """Assess overall reasoning progress"""
        if len(self.reasoning_context) < 3:
            return "early_stage"
        elif len(self.reasoning_context) < 8:
            return "developing"
        elif len(self.reasoning_context) < 15:
            return "mature"
        else:
            return "extensive"

    def _get_error_context(self, error: Exception) -> dict[str, Any]:
        """Get contextual information about an error"""
        return {
            "error_class": type(error).__name__,
            "reasoning_stage": f"loop_{self.current_loop_count}",
            "context_available": len(self.reasoning_context) > 0,
            "stack_state": "populated" if self.internal_task_stack else "empty"
        }

    async def _execute_internal_reasoning(self, args: dict, prep_res: dict) -> dict[str, Any]:
        """Execute internal reasoning meta-tool"""
        thought = args.get("thought", "")
        thought_number = args.get("thought_number", 1)
        total_thoughts = args.get("total_thoughts", 1)
        next_thought_needed = args.get("next_thought_needed", False)
        current_focus = args.get("current_focus", "")
        key_insights = args.get("key_insights", [])
        potential_issues = args.get("potential_issues", [])
        confidence_level = args.get("confidence_level", 0.5)

        # Structure the reasoning entry
        reasoning_entry = {
            "thought": thought,
            "thought_number": thought_number,
            "total_thoughts": total_thoughts,
            "next_thought_needed": next_thought_needed,
            "current_focus": current_focus,
            "key_insights": key_insights,
            "potential_issues": potential_issues,
            "confidence_level": confidence_level,
            "timestamp": datetime.now().isoformat()
        }

        # Add to internal reasoning log
        if not hasattr(self, 'internal_reasoning_log'):
            self.internal_reasoning_log = []
        self.internal_reasoning_log.append(reasoning_entry)

        # Format for context
        context_addition = f"""Internal Reasoning Step {thought_number}/{total_thoughts}:
Thought: {thought}
Focus: {current_focus}
Key Insights: {', '.join(key_insights) if key_insights else 'None'}
Potential Issues: {', '.join(potential_issues) if potential_issues else 'None'}
Confidence: {confidence_level}
Next Thought Needed: {next_thought_needed}"""

        return {"context_addition": context_addition}

    async def _execute_manage_task_stack(self, args: dict, prep_res: dict) -> dict[str, Any]:
        """Execute task stack management meta-tool"""
        action = args.get("action", "get_current").lower()
        task_description = args.get("task_description", "")

        if action == "add":
            self.internal_task_stack.append({
                "description": task_description,
                "status": "pending",
                "added_at": datetime.now().isoformat()
            })
            context_addition = f"Added to task stack: {task_description}"

        elif action == "remove":
            # Remove task by description match
            original_count = len(self.internal_task_stack)
            self.internal_task_stack = [
                task for task in self.internal_task_stack
                if task_description.lower() not in task["description"].lower()
            ]
            removed_count = original_count - len(self.internal_task_stack)
            context_addition = f"Removed {removed_count} task(s) matching: {task_description}"

        elif action == "complete":
            # Mark task as completed
            for task in self.internal_task_stack:
                if task_description.lower() in task["description"].lower():
                    task["status"] = "completed"
                    task["completed_at"] = datetime.now().isoformat()
            context_addition = f"Marked as completed: {task_description}"

        elif action == "get_current":
            if self.internal_task_stack:
                stack_summary = []
                for i, task in enumerate(self.internal_task_stack):
                    status = task["status"]
                    desc = task["description"]
                    stack_summary.append(f"{i + 1}. [{status.upper()}] {desc}")
                context_addition = "Current task stack:\n" + "\n".join(stack_summary)
            else:
                context_addition = "Task stack is empty"

        else:
            context_addition = f"Unknown task stack action: {action}"

        return {"context_addition": context_addition}

    async def _execute_delegate_llm_tool(self, args: dict, prep_res: dict) -> dict[str, Any]:
        """Execute delegation to LLMToolNode"""
        task_description = args.get("task_description", "")
        tools_list = args.get("tools_list", [])

        # Prepare shared state for LLMToolNode
        llm_tool_shared = {
            "current_task_description": task_description + '\nreturn all results in the final answer!',
            "current_query": task_description,
            "formatted_context": {
                "recent_interaction": f"Reasoner delegating task: {task_description}",
                "session_summary": self._get_reasoning_summary(),
                "task_context": f"Reasoning loop {self.current_loop_count}, delegated task. return all results!"
            },
            "variable_manager": prep_res.get("variable_manager"),
            "agent_instance": prep_res.get("agent_instance"),
            "available_tools": tools_list,  # Restrict to specific tools
            "tool_capabilities": prep_res.get("tool_capabilities", {}),
            "fast_llm_model": prep_res.get("fast_llm_model"),
            "complex_llm_model": prep_res.get("complex_llm_model"),
            "progress_tracker": prep_res.get("progress_tracker"),
            "session_id": prep_res.get("session_id"),
            "use_fast_response": True  # Use fast model for delegated tasks
        }

        # Execute LLMToolNode
        try:
            llm_tool_node = LLMToolNode()
            await llm_tool_node.run_async(llm_tool_shared)

            # Get results
            final_response = llm_tool_shared.get("current_response", "Task completed without specific result")
            tool_calls_made = llm_tool_shared.get("tool_calls_made", 0)

            context_addition = f"""Delegated Task Completed:
Task: {task_description}
Tools Available: {', '.join(tools_list)}
Tools Used: {tool_calls_made} tool calls made
Result: {final_response}"""

            return {"context_addition": context_addition}

        except Exception as e:
            context_addition = f"Delegation failed for task '{task_description}': {str(e)}"
            return {"context_addition": context_addition}

    async def _execute_create_plan(self, args: dict, prep_res: dict) -> dict[str, Any]:
        """Execute plan creation and execution"""
        goals = args.get("goals", [])

        if not goals:
            return {"context_addition": "No goals provided for plan creation"}

        try:
            # Prepare shared state for TaskPlanner
            planning_shared = prep_res.copy()
            planning_shared.update({
                "replan_context": {
                    "goals": goals,
                    "triggered_by": "llm_reasoner",
                    "reasoning_context": self._get_reasoning_summary()
                },
                "current_task_description": f"Execute plan with {len(goals)} goals",
                "current_query": f"Complex task: {'; '.join(goals)}"
            })

            # Execute TaskPlanner
            planner_node = TaskPlannerNode()
            plan_info = await planner_node.run_async(planning_shared)

            if plan_info == "planning_failed":
                return {"context_addition": f"Plan creation failed: {planning_shared.get('planning_error', 'Unknown error')}"}

            plan = planning_shared.get("current_plan")
            # Execute the plan using TaskExecutor
            executor_shared = planning_shared.copy()
            executor_node = TaskExecutorNode()

            # Execute plan to completion
            max_execution_cycles = 10
            execution_cycle = 0

            while execution_cycle < max_execution_cycles:
                execution_cycle += 1

                result = await executor_node.run_async(executor_shared)

                # Check completion status
                if result == "plan_completed" or result == "execution_error":
                    break
                elif result in ["continue_execution", "waiting"]:
                    continue
                else:
                    # Handle other results like reflection needs
                    if result in ["needs_dynamic_replan", "needs_plan_append"]:
                        # For now, just continue - could add reflection logic here
                        continue
                    break

            # Collect results
            completed_tasks = [
                task for task in plan.tasks
                if executor_shared["tasks"][task.id].status == "completed"
            ]

            failed_tasks = [
                task for task in plan.tasks
                if executor_shared["tasks"][task.id].status == "failed"
            ]

            # Build context addition with results
            results_summary = []
            results_store = executor_shared.get("results", {})

            for task in completed_tasks:
                task_result = results_store.get(task.id, {})
                if task_result.get("data"):
                    result_preview = str(task_result["data"])[:150] + "..."
                    results_summary.append(f"- {task.description}: {result_preview}")

            context_addition = f"""Plan Execution Completed:
Goals: {len(goals)} goals processed
Tasks Created: {len(plan.tasks)}
Tasks Completed: {len(completed_tasks)}
Tasks Failed: {len(failed_tasks)}
Execution Cycles: {execution_cycle}

Results Summary:
{chr(10).join(results_summary) if results_summary else 'No specific results captured'}"""

            return {"context_addition": context_addition}

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            context_addition = f"Plan execution failed: {str(e)}"
            return {"context_addition": context_addition}

    def _get_reasoning_summary(self) -> str:
        """Get a summary of the reasoning process so far"""
        if not self.reasoning_context:
            return "No reasoning context available"

        summary_parts = []
        reasoning_entries = [entry for entry in self.reasoning_context if entry.get("type") == "reasoning"]

        for entry in reasoning_entries[-3:]:  # Last 3 reasoning steps
            content = entry.get("content", "")[:50000] + "..."
            loop_num = entry.get("loop", "?")
            summary_parts.append(f"Loop {loop_num}: {content}")

        return "\n".join(summary_parts)

    async def _create_error_response(self, query: str, error: str) -> str:
        """Create an error response"""
        return f"I encountered an error while processing your request: {error}. I was working on: {query}"

    async def _fallback_direct_response(self, prep_res: dict) -> dict[str, Any]:
        """Fallback when LLM is not available"""
        query = prep_res["original_query"]
        fallback_response = f"I received your request: {query}. However, I'm currently unable to process complex requests due to limited capabilities."

        return {
            "final_result": fallback_response,
            "reasoning_loops": 0,
            "reasoning_context": [{"type": "fallback", "content": "LLM unavailable"}],
            "internal_task_stack": []
        }

# ===== Foramt Helper =====
class VariableManager:
    """Unified variable management system with advanced features"""

    def __init__(self, world_model: dict, shared_state: dict = None):
        self.world_model = world_model
        self.shared_state = shared_state or {}
        self.scopes = {
            'world': world_model,
            'shared': self.shared_state,
            'results': {},
            'tasks': {},
            'user': {},
            'system': {}
        }
        self._cache = {}

    def register_scope(self, name: str, data: dict):
        """Register a new variable scope"""
        self.scopes[name] = data
        self._cache.clear()

    def set_results_store(self, results_store: dict):
        """Set the results store for task result references"""
        self.scopes['results'] = results_store
        self._cache.clear()

    def set_tasks_store(self, tasks_store: dict):
        """Set tasks store for task metadata access"""
        self.scopes['tasks'] = tasks_store
        self._cache.clear()

    def _resolve_path(self, path: str):
        """
        Internal helper to navigate a path that can contain both
        dictionary keys and list indices.
        """
        parts = path.split('.')

        # Determine the starting point
        if len(parts) == 1:
            # Simple key in the top-level world_model
            current = self.world_model
        else:
            scope_name = parts[0]
            if scope_name not in self.scopes:
                raise KeyError(f"Scope '{scope_name}' not found")
            current = self.scopes[scope_name]
            parts = parts[1:]  # Continue with the rest of the path

        # Navigate through the parts
        for part in parts:
            if isinstance(current, list):
                try:
                    # It's a list, so the part must be an integer index
                    index = int(part)
                    current = current[index]
                except (ValueError, IndexError):
                    raise KeyError(f"Invalid list index '{part}' in path '{path}'")
            elif isinstance(current, dict):
                try:
                    # It's a dictionary, so the part is a key
                    current = current[part]
                except KeyError:
                    raise KeyError(f"Key '{part}' not found in path '{path}'")
            else:
                # We've hit a non-collection type (int, str, etc.) but the path continues
                raise KeyError(f"Path cannot descend into non-collection type at '{part}' in path '{path}'")

        return current

    def get(self, path: str, default=None, use_cache: bool = True):
        """Get variable with dot notation path support for dicts and lists."""
        if use_cache and path in self._cache:
            return self._cache[path]

        try:
            value = self._resolve_path(path)
            if use_cache:
                self._cache[path] = value
            return value
        except (KeyError, IndexError):
            # A KeyError or IndexError during resolution means the path is invalid
            return default

    def set(self, path: str, value, create_scope: bool = True):
        """Set variable with dot notation path support for dicts and lists."""
        # Invalidate cache for this path
        if path in self._cache:
            del self._cache[path]

        parts = path.split('.')

        if len(parts) == 1:
            # Simple key in world_model
            self.world_model[path] = value
            return

        scope_name = parts[0]
        if scope_name not in self.scopes:
            if create_scope:
                self.scopes[scope_name] = {}
            else:
                raise KeyError(f"Scope '{scope_name}' not found")

        current = self.scopes[scope_name]

        # Iterate to the second-to-last part to get the container
        for i, part in enumerate(parts[1:-1]):
            next_part = parts[i + 2]  # Look ahead to the next part in the path

            # Determine if the current part is a dictionary key or a list index
            try:
                # Try to treat it as a list index
                key = int(part)
                if not isinstance(current, list):
                    # If current is not a list, we can't use an integer index
                    raise TypeError(f"Attempted to use integer index '{key}' on non-list for path '{path}'")

                # Ensure list is long enough
                while len(current) <= key:
                    current.append(None)  # Pad with None

                # If the next level doesn't exist, create it based on the next part
                if current[key] is None:
                    current[key] = [] if next_part.isdigit() else {}

                current = current[key]

            except ValueError:
                # It's a dictionary key
                key = part
                if not isinstance(current, dict):
                    raise TypeError(f"Attempted to use string key '{key}' on non-dict for path '{path}'")

                if key not in current:
                    # Create the next level: a list if the next part is a number, else a dict
                    current[key] = [] if next_part.isdigit() else {}

                current = current[key]

        # Handle the final part (the actual assignment)
        last_part = parts[-1]

        if isinstance(current, list):
            try:
                key = int(last_part)
                while len(current) <= key:
                    current.append(None)
                current[key] = value
            except ValueError:
                current.append(value)
        elif isinstance(current, dict):
            current[last_part] = value
        elif scope_name == 'tasks' and hasattr(current, 'task_identification_attr'):# from tasks like Tooltask ... model dump and acces
            dict_data = asdict(current)
            dict_data[last_part] = value
            current = dict_data
            # update self.scopes['tasks'] with the updated task
            self.scopes['tasks'][parts[1]][last_part] = current
        else:
            raise TypeError(f"Final container is not a list or dictionary for path '{path}' its a {type(current)}")

        self._cache.clear()

    def format_text(self, text: str, context: dict = None) -> str:
        """Enhanced text formatting with multiple syntaxes"""
        if not text or not isinstance(text, str):
            return str(text) if text is not None else ""

        # Temporary context overlay
        if context:
            original_scopes = self.scopes.copy()
            self.scopes['context'] = context

        try:
            # Handle {{ variable }} syntax
            formatted = self._format_double_braces(text)

            # Handle {variable} syntax
            formatted = self._format_single_braces(formatted)

            # Handle $variable syntax
            formatted = self._format_dollar_syntax(formatted)

            return formatted

        finally:
            if context:
                self.scopes = original_scopes

    def _format_double_braces(self, text: str) -> str:
        """Handle {{ variable.path }} syntax with improved debugging"""
        import re

        def replace_var(match):
            var_path = match.group(1).strip()
            value = self.get(var_path)

            if value is None:
                # IMPROVED: Log missing variables for debugging
                available_vars = list(self.get_available_variables().keys())
                wprint(f"Variable '{var_path}' not found. Available: {available_vars[:10]}")
                return match.group(0)  # Keep original if not found

            return self._value_to_string(value)

        return re.sub(r'\{\{\s*([^}]+)\s*\}\}', replace_var, text)

    def _format_single_braces(self, text: str) -> str:
        """Handle {variable.path} syntax, including with spaces like { variable.path }."""
        import re

        def replace_var(match):
            # Extrahiert den Variablennamen und entfernt führende/nachfolgende Leerzeichen
            var_path = match.group(1).strip()

            # Ruft den Wert über die get-Methode ab, die die Punktnotation bereits verarbeitet
            value = self.get(var_path)

            # Gibt den konvertierten Wert oder das Original-Tag zurück, wenn der Wert nicht gefunden wurde
            return self._value_to_string(value) if value is not None else match.group(0)

        # Dieser Regex findet {beliebiger.inhalt} und erlaubt Leerzeichen um den Inhalt
        # Er schließt verschachtelte oder leere Klammern wie {} oder { {var} } aus.
        return re.sub(r'\{([^{}]+)\}', replace_var, text)

    def _format_dollar_syntax(self, text: str) -> str:
        """Handle $variable syntax"""
        import re

        def replace_var(match):
            var_name = match.group(1)
            value = self.get(var_name)
            return self._value_to_string(value) if value is not None else match.group(0)

        return re.sub(r'\$([a-zA-Z_][a-zA-Z0-9_]*)', replace_var, text)

    def _value_to_string(self, value) -> str:
        """Convert value to string representation"""
        if isinstance(value, str):
            return value
        elif isinstance(value, dict | list):
            return json.dumps(value, default=str)
        else:
            return str(value)

    def validate_references(self, text: str) -> dict[str, bool]:
        """Validate all variable references in text"""
        import re

        references = {}

        # Find all {{ }} references
        double_brace_refs = re.findall(r'\{\{\s*([^}]+)\s*\}\}', text)
        for ref in double_brace_refs:
            references["{{"+ref+"}}"] = self.get(ref.strip()) is not None

        # Find all {} references
        single_brace_refs = re.findall(r'\{([^{}\s]+)\}', text)
        for ref in single_brace_refs:
            if '.' not in ref:  # Only simple vars
                references["{"+ref+"}"] = self.get(ref.strip()) is not None

        # Find all $ references
        dollar_refs = re.findall(r'\$([a-zA-Z_][a-zA-Z0-9_]*)', text)
        for ref in dollar_refs:
            references[f"${ref}"] = self.get(ref) is not None

        return references

    def get_scope_info(self) -> dict[str, Any]:
        """Get information about all available scopes"""
        info = {}
        for scope_name, scope_data in self.scopes.items():
            if isinstance(scope_data, dict):
                info[scope_name] = {
                    'type': 'dict',
                    'keys': len(scope_data),
                    'sample_keys': list(scope_data.keys())[:5]
                }
            else:
                info[scope_name] = {
                    'type': type(scope_data).__name__,
                    'value': str(scope_data)[:100]
                }
        return info

    def _validate_task_references(self, task: Task) -> dict[str, Any]:
        """Validate all variable references in a task"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # Check different task types
        if isinstance(task, LLMTask):
            if task.prompt_template:
                refs = self.validate_references(task.prompt_template)
                for ref, is_valid in refs.items():
                    if not is_valid:
                        validation_results['errors'].append(f"Invalid reference in prompt: {ref}")
                        validation_results['valid'] = False

        elif isinstance(task, ToolTask):
            for key, value in task.arguments.items():
                if isinstance(value, str):
                    refs = self.validate_references(value)
                    for ref, is_valid in refs.items():
                        if not is_valid:
                            validation_results['warnings'].append(f"Invalid reference in {key}: {ref}")

        return validation_results

    def get_variable_suggestions(self, query: str) -> list[str]:
        """Get variable suggestions based on query content"""

        query_lower = query.lower()
        suggestions = []

        # Check all variables for relevance
        for scope in self.scopes.values():
            for name, var_def in scope.items():
                if name in ["system_context", "task_executor_instance",
                            "index", "tool_capabilities", "use_fast_response", "task_planner_instance"]:
                    continue
                # Name similarity
                if any(word in name.lower() for word in query_lower.split()):
                    suggestions.append(name)
                    continue

                # Description similarity
                if var_def and any(word in str(var_def).lower() for word in query_lower.split()):
                    suggestions.append(name)
                    continue


        return list(set(suggestions))[:10]

    def _document_structure(self, data: Any, path_prefix: str, docs: dict[str, dict]):
        """A recursive helper to document nested dictionaries and lists."""
        if isinstance(data, dict):
            for key, value in data.items():
                # Construct the full path for the current item
                current_path = f"{path_prefix}.{key}" if path_prefix else key

                # Generate a preview for the value
                if isinstance(value, str):
                    preview = value[:70] + "..." if len(value) > 70 else value
                elif isinstance(value, dict):
                    preview = f"Object with keys: {list(value.keys())[:3]}" + ("..." if len(value.keys()) > 3 else "")
                elif isinstance(value, list):
                    preview = f"List with {len(value)} items"
                else:
                    preview = str(value)

                # Store the documentation for the current path
                docs[current_path] = {
                    'preview': preview,
                    'type': type(value).__name__
                }

                # Recurse into nested structures
                if isinstance(value, dict | list):
                    self._document_structure(value, current_path, docs)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                # Construct the full path for the list item
                current_path = f"{path_prefix}.{i}"

                # Generate a preview for the item
                if isinstance(item, str):
                    preview = item[:70] + "..." if len(item) > 70 else item
                elif isinstance(item, dict):
                    preview = f"Object with keys: {list(item.keys())[:3]}" + ("..." if len(item.keys()) > 3 else "")
                elif isinstance(item, list):
                    preview = f"List with {len(item)} items"
                else:
                    preview = str(item)

                docs[current_path] = {
                    'preview': preview,
                    'type': type(item).__name__
                }

                # Recurse into nested structures
                if isinstance(item, dict | list):
                    self._document_structure(item, current_path, docs)

    def get_available_variables(self) -> dict[str, dict]:
        """
        Recursively documents all available variables from world_model and scopes
        to provide a comprehensive overview for an LLM.
        """
        all_vars_docs = {}

        # 1. Document the world_model (top-level variables)
        self._document_structure(self.world_model, "", all_vars_docs)

        # 2. Document each scope
        for scope_name, scope_data in self.scopes.items():
            # Add documentation for the scope root itself
            if scope_name == "shared":
                continue
            if isinstance(scope_data, dict):
                preview = f"Object with keys: {list(scope_data.keys())[:3]}" + (
                    "..." if len(scope_data.keys()) > 3 else "")
            elif isinstance(scope_data, list):
                preview = f"List with {len(scope_data)} items"
            elif isinstance(scope_data, str | int):
                preview = str(scope_data)
            else:
                continue

            all_vars_docs[scope_name] = {'preview': preview, 'type': type(scope_data).__name__}

            # Recurse into the scope's data
            self._document_structure(scope_data, scope_name, all_vars_docs)

        return all_vars_docs

    def get_llm_variable_context(self) -> str:
        """
        Generates a detailed variable context formatted for LLM consumption,
        explaining structure, access patterns, and listing all available variables.
        """
        context_parts = [
            "## Variable System Reference",
            "You can access a state management system to retrieve data using dot notation.",
            "Syntax: `{{ path.to.variable }}` or `$path.to.variable`.",
            "",
            "### How to Access Data",
            "The system contains nested objects (dictionaries) and lists (arrays).",
            "",
            "**1. Object (Dictionary) Access (Primary Usage):**",
            "Use a dot (`.`) to access values inside an object. This is the most common way to get data.",
            "Example: If a `user` object exists with a `profile`, you can get the name with `{{ user.profile.name }}`.",
            "",
            "**2. List (Array) Access:**",
            "If a variable is a list, use a dot (`.`) followed by a zero-based number (index) to access a specific item.",
            "Example: To get the first email from a user's email list, use `{{ user.emails.0 }}`.",
            "You can chain these access methods: `{{ user.emails.0.address }}`.",
            "",
            "### Available Variables",
            "Below is a list of all currently available variable paths, their type, and a preview of their content. (Note: Previews may be truncated).",
        ]

        variables = self.get_available_variables()
        if not variables:
            context_parts.append("- No variables are currently set.")
            return "\n".join(context_parts)

        if "shared" in variables:
            variables["shared"] = {'preview': "Shared state variables", 'type': "dict"}

        # yaml dump preview
        context_parts.append("```yaml")
        context_parts.append(yaml.dump(variables, default_flow_style=False, sort_keys=False))
        context_parts.append("```")

        # Add any final complex examples or notes
        context_parts.extend([
            "",
            "**Note on Task Results:**",
            "All task results are stored in the `results` scope. To access the data from a task, append `.data`.",
            "Example: `{{ results.'task-id-123'.data }}`"
        ])

        return "\n".join(context_parts)


class UnifiedContextManager:
    """
    Zentrale Orchestrierung aller Context-Quellen für einheitlichen und effizienten Datenzugriff.
    Vereinigt ChatSession, VariableManager, World Model und Task Results.
    """

    def __init__(self, agent):
        self.agent = agent
        self.session_managers: dict[str, Any] = {}  # ChatSession objects
        self.variable_manager: VariableManager = None
        self.compression_threshold = 15  # Messages before compression
        self._context_cache: dict[str, tuple[float, Any]] = {}  # (timestamp, data)
        self.cache_ttl = 300  # 5 minutes
        self._memory_instance = None

    async def initialize_session(self, session_id: str, max_history: int = 200):
        """Initialisiere oder lade existierende ChatSession als primäre Context-Quelle"""
        if session_id not in self.session_managers:
            try:
                # Get memory instance
                if not self._memory_instance:
                    from toolboxv2 import get_app
                    self._memory_instance = get_app().get_mod("isaa").get_memory()
                from toolboxv2.mods.isaa.extras.session import ChatSession
                # Create ChatSession as PRIMARY memory source
                session = ChatSession(
                    self._memory_instance,
                    max_length=max_history,
                    space_name=f"ChatSession/{self.agent.amd.name}.{session_id}.unified"
                )
                self.session_managers[session_id] = session

                # Integration mit VariableManager wenn verfügbar
                if self.variable_manager:
                    self.variable_manager.register_scope(f'session_{session_id}', {
                        'chat_session_active': True,
                        'history_length': len(session.history),
                        'last_interaction': None,
                        'session_id': session_id
                    })

                rprint(f"Unified session context initialized for {session_id}")
                return session

            except Exception as e:
                eprint(f"Failed to create ChatSession for {session_id}: {e}")
                # Fallback: Create minimal session manager
                self.session_managers[session_id] = {
                    'history': [],
                    'session_id': session_id,
                    'fallback_mode': True
                }
                return self.session_managers[session_id]

        return self.session_managers[session_id]

    async def add_interaction(self, session_id: str, role: str, content: str, metadata: dict = None) -> None:
        """Einheitlicher Weg um Interaktionen in ChatSession zu speichern"""
        session = await self.initialize_session(session_id)

        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'metadata': metadata or {}
        }

        # PRIMARY: Store in ChatSession
        if hasattr(session, 'add_message'):
            await session.add_message(message, direct=False)
        elif isinstance(session, dict) and 'history' in session:
            # Fallback mode
            session['history'].append(message)
            # Keep max length
            max_len = 200
            if len(session['history']) > max_len:
                session['history'] = session['history'][-max_len:]

        # SECONDARY: Update VariableManager
        if self.variable_manager:
            self.variable_manager.set(f'session_{session_id}.last_interaction', message)
            if hasattr(session, 'history'):
                self.variable_manager.set(f'session_{session_id}.history_length', len(session.history))
            elif isinstance(session, dict):
                self.variable_manager.set(f'session_{session_id}.history_length', len(session.get('history', [])))

        # Clear context cache for this session
        self._invalidate_cache(session_id)

    async def get_contextual_history(self, session_id: str, query: str = "", max_entries: int = 10) -> list[dict]:
        """Intelligente Auswahl relevanter Geschichte aus ChatSession"""
        session = self.session_managers.get(session_id)
        if not session:
            return []

        try:
            # ChatSession mode
            if hasattr(session, 'get_past_x'):
                recent_history = session.get_past_x(max_entries * 2, last_u=False)
                # await session.get_reference(query, limit=5)

                return recent_history[:max_entries]

            # Fallback mode
            elif isinstance(session, dict) and 'history' in session:
                history = session['history']
                # Return last max_entries, starting with last user message
                result = []
                for msg in reversed(history[-max_entries * 2:]):
                    result.append(msg)
                    if msg.get('role') == 'user' and len(result) >= max_entries:
                        break
                return list(reversed(result))[:max_entries]

        except Exception as e:
            eprint(f"Error getting contextual history: {e}")

        return []

    async def build_unified_context(self, session_id: str, query: str = None, context_type: str = "full") -> dict[
        str, Any]:
        """ZENTRALE Methode für vollständigen Context-Aufbau aus allen Quellen"""

        # Cache check
        cache_key = f"{session_id}_{hash(query or '')}_{context_type}"
        cached = self._get_cached_context(cache_key)
        if cached:
            return cached

        context = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'query': query,
            'context_type': context_type
        }

        try:
            # 1. CHAT HISTORY (Primary - from ChatSession)
            context['chat_history'] = await self.get_contextual_history(
                session_id, query or "", max_entries=15
            )

            # 2. VARIABLE SYSTEM STATE
            if self.variable_manager:
                context['variables'] = {
                    'available_scopes': list(self.variable_manager.scopes.keys()),
                    'total_variables': len(self.variable_manager.get_available_variables()),
                    'recent_results': self._get_recent_results(5)
                }
            else:
                context['variables'] = {'status': 'variable_manager_not_available'}

            # 3. WORLD MODEL FACTS
            if self.variable_manager:
                world_model = self.variable_manager.get('world', {})
                if world_model and query:
                    context['relevant_facts'] = self._extract_relevant_facts(world_model, query)
                else:
                    context['relevant_facts'] = list(world_model.items())[:5]  # Top 5 facts

            # 4. EXECUTION STATE
            context['execution_state'] = {
                'active_tasks': self._get_active_tasks(),
                'recent_completions': self._get_recent_completions(3),
                'system_status': self.agent.shared.get('system_status', 'idle')
            }

            # 5. SESSION STATISTICS
            context['session_stats'] = {
                'total_sessions': len(self.session_managers),
                'current_session_length': len(context['chat_history']),
                'cache_enabled': bool(self._context_cache)
            }

        except Exception as e:
            eprint(f"Error building unified context: {e}")
            context['error'] = str(e)
            context['fallback_mode'] = True

        # Cache result
        self._cache_context(cache_key, context)
        return context

    def get_formatted_context_for_llm(self, unified_context: dict[str, Any]) -> str:
        """Formatiere unified context für LLM consumption"""
        try:
            parts = []

            # Recent Chat History
            chat_history = unified_context.get('chat_history', [])
            if chat_history:
                parts.append("## Recent Conversation")
                for msg in chat_history[-5:]:  # Last 5 messages
                    timestamp = msg.get('timestamp', '')[:19]  # Remove microseconds
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')[:500] + ("..." if len(msg.get('content', '')) > 500 else "")
                    parts.append(f"[{timestamp}] {role}: {content}")

            # System Status
            execution_state = unified_context.get('execution_state', {})
            if execution_state:
                parts.append("\n## Current System State")
                parts.append(f"Status: {execution_state.get('system_status', 'unknown')}")

                active_tasks = execution_state.get('active_tasks', [])
                if active_tasks:
                    parts.append(f"Active Tasks: {len(active_tasks)}")

                recent_completions = execution_state.get('recent_completions', [])
                if recent_completions:
                    parts.append(f"Recent Completions: {len(recent_completions)}")

            # Available Data
            variables = unified_context.get('variables', {})
            if variables and variables.get('recent_results'):
                parts.append("\n## Available Results")
                recent_results = variables['recent_results']
                for result in recent_results[:3]:  # Top 3 results
                    parts.append(f"- {result.get('task_id', 'unknown')}: {str(result.get('preview', ''))[:100]}...")

            # World Model Facts
            relevant_facts = unified_context.get('relevant_facts', [])
            if relevant_facts:
                parts.append("\n## Known Facts")
                for key, value in relevant_facts[:5]:  # Top 5 facts
                    fact_preview = str(value)[:100] + ("..." if len(str(value)) > 100 else "")
                    parts.append(f"- {key}: {fact_preview}")

            parts.append(f"\n---\nContext generated at: {unified_context.get('timestamp', 'unknown')}")

            return "\n".join(parts)

        except Exception as e:
            eprint(f"Error formatting context for LLM: {e}")
            return f"Context formatting error: {str(e)}"

    def _merge_and_dedupe_history(self, recent_history: list[dict], relevant_refs: list) -> list[dict]:
        """Merge und dedupliziere History-Einträge"""
        try:
            merged = recent_history.copy()

            # Add relevant references if they're not already in recent history
            for ref in relevant_refs:
                # Convert ref to message format if needed
                if isinstance(ref, dict) and 'content' in ref:
                    # Check if not already in recent_history
                    is_duplicate = any(
                        msg.get('content', '') == ref.get('content', '') and
                        msg.get('timestamp', '') == ref.get('timestamp', '')
                        for msg in merged
                    )
                    if not is_duplicate:
                        merged.append(ref)

            # Sort by timestamp
            merged.sort(key=lambda x: x.get('timestamp', ''))

            return merged
        except:
            return recent_history

    def _get_recent_results(self, limit: int = 5) -> list[dict]:
        """Hole recent results aus dem shared state"""
        try:
            results_store = self.agent.shared.get("results", {})
            recent_results = []

            for task_id, result_data in list(results_store.items())[-limit:]:
                if result_data and result_data.get("data"):
                    preview = str(result_data["data"])[:150] + "..."
                    recent_results.append({
                        "task_id": task_id,
                        "preview": preview,
                        "success": result_data.get("metadata", {}).get("success", False),
                        "timestamp": result_data.get("metadata", {}).get("completed_at")
                    })

            return recent_results
        except:
            return []

    def _extract_relevant_facts(self, world_model: dict, query: str) -> list[tuple[str, Any]]:
        """Extrahiere relevante Facts basierend auf Query"""
        try:
            query_words = set(query.lower().split())
            relevant_facts = []

            for key, value in world_model.items():
                # Simple relevance scoring
                key_words = set(key.lower().split())
                value_words = set(str(value).lower().split())

                # Check for word overlap
                key_overlap = len(query_words.intersection(key_words))
                value_overlap = len(query_words.intersection(value_words))

                if key_overlap > 0 or value_overlap > 0:
                    relevance_score = key_overlap * 2 + value_overlap  # Key matches weighted higher
                    relevant_facts.append((relevance_score, key, value))

            # Sort by relevance and return top facts
            relevant_facts.sort(key=lambda x: x[0], reverse=True)
            return [(key, value) for _, key, value in relevant_facts[:5]]
        except:
            return list(world_model.items())[:5]

    def _get_active_tasks(self) -> list[dict]:
        """Hole aktive Tasks"""
        try:
            tasks = self.agent.shared.get("tasks", {})
            return [
                {"id": task_id, "description": task.description, "status": task.status}
                for task_id, task in tasks.items()
                if task.status == "running"
            ]
        except:
            return []

    def _get_recent_completions(self, limit: int = 3) -> list[dict]:
        """Hole recent completions"""
        try:
            tasks = self.agent.shared.get("tasks", {})
            completed = [
                {"id": task_id, "description": task.description, "completed_at": task.completed_at}
                for task_id, task in tasks.items()
                if task.status == "completed" and hasattr(task, 'completed_at') and task.completed_at
            ]
            # Sort by completion time
            completed.sort(key=lambda x: x.get('completed_at', ''), reverse=True)
            return completed[:limit]
        except:
            return []

    def _get_cached_context(self, cache_key: str) -> dict[str, Any] | None:
        """Hole Context aus Cache wenn noch gültig"""
        if cache_key in self._context_cache:
            timestamp, data = self._context_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return data
            else:
                del self._context_cache[cache_key]
        return None

    def _cache_context(self, cache_key: str, context: dict[str, Any]):
        """Speichere Context in Cache"""
        self._context_cache[cache_key] = (time.time(), context.copy())

        # Cleanup old cache entries
        if len(self._context_cache) > 50:  # Keep max 50 entries
            oldest_key = min(self._context_cache.keys(),
                             key=lambda k: self._context_cache[k][0])
            del self._context_cache[oldest_key]

    def _invalidate_cache(self, session_id: str = None):
        """Invalidate cache for specific session or all"""
        if session_id:
            # Remove all cache entries for this session
            keys_to_remove = [k for k in self._context_cache if session_id in k]
            for key in keys_to_remove:
                del self._context_cache[key]
        else:
            self._context_cache.clear()

    def get_session_statistics(self) -> dict[str, Any]:
        """Hole Statistiken über alle Sessions"""
        stats = {
            "total_sessions": len(self.session_managers),
            "active_sessions": [],
            "cache_entries": len(self._context_cache),
            "cache_hit_rate": 0.0  # Could be tracked if needed
        }

        for session_id, session in self.session_managers.items():
            session_info = {
                "session_id": session_id,
                "fallback_mode": isinstance(session, dict) and session.get('fallback_mode', False)
            }

            if hasattr(session, 'history'):
                session_info["message_count"] = len(session.history)
            elif isinstance(session, dict) and 'history' in session:
                session_info["message_count"] = len(session['history'])

            stats["active_sessions"].append(session_info)

        return stats

    async def cleanup_old_sessions(self, max_age_hours: int = 168) -> int:
        """Cleanup alte Sessions (default: 1 Woche)"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            removed_count = 0

            sessions_to_remove = []
            for session_id, session in self.session_managers.items():
                should_remove = False

                # Check last activity
                if hasattr(session, 'history') and session.history:
                    last_msg = session.history[-1]
                    last_timestamp = last_msg.get('timestamp')
                    if last_timestamp:
                        try:
                            last_time = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                            if last_time < cutoff_time:
                                should_remove = True
                        except:
                            pass
                elif isinstance(session, dict) and session.get('history'):
                    last_msg = session['history'][-1]
                    last_timestamp = last_msg.get('timestamp')
                    if last_timestamp:
                        try:
                            last_time = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                            if last_time < cutoff_time:
                                should_remove = True
                        except:
                            pass

                if should_remove:
                    sessions_to_remove.append(session_id)

            # Remove old sessions
            for session_id in sessions_to_remove:
                session = self.session_managers[session_id]
                if hasattr(session, 'on_exit'):
                    session.on_exit()  # Save ChatSession data
                del self.session_managers[session_id]
                removed_count += 1

                # Remove from variable manager
                if self.variable_manager:
                    scope_name = f'session_{session_id}'
                    if scope_name in self.variable_manager.scopes:
                        del self.variable_manager.scopes[scope_name]

            # Clear related cache entries
            self._invalidate_cache()

            return removed_count
        except Exception as e:
            eprint(f"Error cleaning up old sessions: {e}")
            return 0

# ===== MAIN AGENT CLASS =====
class FlowAgent:
    """Production-ready agent system built on PocketFlow """
    def __init__(
        self,
        amd: AgentModelData,
        world_model: dict[str, Any] = None,
        verbose: bool = False,
        enable_pause_resume: bool = True,
        checkpoint_interval: int = 300,  # 5 minutes
        max_parallel_tasks: int = 3,
        progress_callback: callable = None,
        **kwargs
    ):
        self.amd = amd
        self.world_model = world_model or {}
        self.verbose = verbose
        self.enable_pause_resume = enable_pause_resume
        self.checkpoint_interval = checkpoint_interval
        self.max_parallel_tasks = max_parallel_tasks
        self.progress_tracker = ProgressTracker(progress_callback, agent_name=amd.name)

        # Core state
        self.shared = {
            "world_model": self.world_model,
            "tasks": {},
            "task_plans": {},
            "system_status": "idle",
            "session_data": {},
            "performance_metrics": {},
            "conversation_history": [],
            "available_tools": [],
            "progress_tracker": self.progress_tracker
        }
        self.context_manager = UnifiedContextManager(self)
        self.variable_manager = VariableManager(self.shared["world_model"], self.shared)
        self.context_manager.variable_manager = self.variable_manager# Register default scopes

        self.shared["context_manager"] = self.context_manager
        self.shared["variable_manager"] = self.variable_manager
        # Flows
        self.task_flow = TaskManagementFlow(max_parallel_tasks=self.max_parallel_tasks)
        self.response_flow = ResponseGenerationFlow()

        if hasattr(self.task_flow, 'executor_node'):
            self.task_flow.executor_node.agent_instance = self

        # Agent state
        self.is_running = False
        self.is_paused = False
        self.last_checkpoint = None
        self.checkpoint_data = {}

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_parallel_tasks)
        self._shutdown_event = threading.Event()

        # Server components
        self.a2a_server: A2AServer = None
        self.mcp_server: FastMCP = None

        # Enhanced tool registry
        self._tool_registry = {}
        self._tool_capabilities = {}
        self._tool_analysis_cache = {}

        self.active_session = None
        # Tool analysis file path
        self.tool_analysis_file = self._get_tool_analysis_path()

        self._tool_capabilities.update(self._load_tool_analysis())
        if self.amd.budget_manager:
            self.amd.budget_manager.load_data()

        self._setup_variable_scopes()

        rprint(f"FlowAgent initialized: {amd.name}")

    @property
    def progress_callback(self):
        return self.progress_tracker.progress_callback

    @progress_callback.setter
    def progress_callback(self, value):
        self.progress_tracker.progress_callback = value

    def set_progress_callback(self, progress_callback: callable = None):
        self.progress_callback = progress_callback

    async def a_run_llm_completion(self, node_name="FlowAgentLLMCall",task_id="unknown",model_preference="fast", with_context=True, **kwargs) -> str:
        if "model" not in kwargs:
            kwargs["model"] = self.amd.fast_llm_model if model_preference == "fast" else self.amd.complex_llm_model

        llm_start = time.perf_counter()

        if self.progress_tracker:
            await self.progress_tracker.emit_event(ProgressEvent(
                event_type="llm_call",
                node_name=node_name,
                session_id=self.active_session,
                task_id=task_id,
                status=NodeStatus.RUNNING,
                llm_model=kwargs["model"],
                llm_temperature=kwargs.get("temperature", 0.7),
                llm_input=kwargs.get("messages", [{}])[-1].get("content", ""),  # Prompt direkt erfassen
                metadata={
                    "model_preference": kwargs.get("model_preference", "fast")
                }
            ))

        # auto api key addition supports (google, openrouter, openai, anthropic, azure, aws, huggingface, replicate, togetherai, groq)
        if "api_key" not in kwargs:
            # litellm model-prefix apikey mapp
            prefix = kwargs['model'].split("/")[0]
            model_prefix_map = {
                "openrouter": os.getenv("OPENROUTER_API_KEY"),
                "openai": os.getenv("OPENAI_API_KEY"),
                "anthropic": os.getenv("ANTHROPIC_API_KEY"),
                "google": os.getenv("GOOGLE_API_KEY"),
                "azure": os.getenv("AZURE_API_KEY"),
                "huggingface": os.getenv("HUGGINGFACE_API_KEY"),
                "replicate": os.getenv("REPLICATE_API_KEY"),
                "togetherai": os.getenv("TOGETHERAI_API_KEY"),
                "groq": os.getenv("GROQ_API_KEY"),
            }
            kwargs["api_key"] = model_prefix_map.get(prefix)

        if self.active_session and with_context:
            # Add context to fist messages as system message
            context_ = await self.get_context(self.active_session)
            kwargs["messages"] = [{"role": "system", "content": self.amd.get_system_message_with_persona()+'\n\nContext:\n\n'+context_}] + kwargs.get("messages", [])

        # build fallback dict using FALLBACKS_MODELS/PREM and _KEYS

        if 'fallbacks' not in kwargs:
            fallbacks_dict_list = []
            fallbacks = os.getenv("FALLBACKS_MODELS", '').split(',') if model_preference == "fast" else os.getenv(
                "FALLBACKS_MODELS_PREM", '').split(',')
            fallbacks_keys = os.getenv("FALLBACKS_MODELS_KEYS", '').split(
                ',') if model_preference == "fast" else os.getenv(
                "FALLBACKS_MODELS_KEYS_PREM", '').split(',')
            for model, key in zip(fallbacks, fallbacks_keys):
                fallbacks_dict_list.append({"model": model, "api_key": key})
            kwargs['fallbacks'] = fallbacks_dict_list
        try:

            response = await litellm.acompletion(**kwargs)

            llm_duration = time.perf_counter() - llm_start
            result = response.choices[0].message.content

            if AGENT_VERBOSE and self.verbose:
                kwargs["messages"] += [{"role": "assistant", "content": result}]
                print_prompt(kwargs)
            # else:
            #     print_prompt([{"role": "assistant", "content": result}])

            # Extract token usage and cost
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0

            call_cost = self.progress_tracker.calculate_llm_cost(kwargs["model"], input_tokens,
                                                            output_tokens, response) if self.progress_tracker else 0.0

            if self.progress_tracker:
                await self.progress_tracker.emit_event(ProgressEvent(
                    event_type="llm_call",
                    node_name=node_name,
                    task_id=task_id,
                    session_id=self.active_session,
                    status=NodeStatus.COMPLETED,
                    success=True,
                    duration=llm_duration,
                    llm_model=kwargs["model"],
                    llm_prompt_tokens=input_tokens,
                    llm_completion_tokens=output_tokens,
                    llm_total_tokens=total_tokens,
                    llm_cost=call_cost,
                    llm_temperature=kwargs.get("temperature", 0.7),
                    llm_output=result,
                    llm_input="",
                ))

            return result
        except Exception as e:
            llm_duration = time.perf_counter() - llm_start
            import traceback
            print(traceback.format_exc())

            if self.progress_tracker:
                await self.progress_tracker.emit_event(ProgressEvent(
                    event_type="llm_call",  # Event-Typ bleibt konsistent
                    node_name=node_name,
                    task_id=task_id,
                    session_id=self.active_session,
                    status=NodeStatus.FAILED,
                    success=False,
                    duration=llm_duration,
                    llm_model=kwargs["model"],
                    error_details={
                        "message": str(e),
                        "type": type(e).__name__
                    }
                ))

            raise

    async def a_run(
        self,
        query: str,
        session_id: str = "default",
        user_id: str = None,
        stream_callback: Callable = None,
        remember: bool = True,
        **kwargs
    ) -> str:
        """Main entry point für Agent-Ausführung mit UnifiedContextManager"""

        execution_start = self.progress_tracker.start_timer("total_execution")
        self.active_session = session_id
        result = None
        await self.progress_tracker.emit_event(ProgressEvent(
            event_type="execution_start",
            timestamp=time.time(),
            status=NodeStatus.RUNNING,
            node_name="FlowAgent",
            session_id=session_id,
            metadata={"query": query, "user_id": user_id}
        ))

        try:
            #Initialize or get session über UnifiedContextManager
            await self.initialize_session_context(session_id, max_history=200)

            #Store user message immediately in ChatSession wenn remember=True
            if remember:
                await self.context_manager.add_interaction(
                    session_id,
                    'user',
                    query,
                    metadata={"user_id": user_id}
                )

            # Set user context variables
            timestamp = datetime.now()
            self.variable_manager.register_scope('user', {
                'id': user_id,
                'session': session_id,
                'query': query,
                'timestamp': timestamp.isoformat()
            })

            # Update system variables
            self.variable_manager.set('system_context.timestamp', {'isoformat': timestamp.isoformat()})
            self.variable_manager.set('system_context.current_session', session_id)
            self.variable_manager.set('system_context.current_user', user_id)
            self.variable_manager.set('system_context.last_query', query)

            # Initialize with tool awareness
            await self.initialize_context_awareness()

            # VEREINFACHT: Prepare execution context - weniger Daten duplizieren
            self.shared.update({
                "current_query": query,
                "session_id": session_id,
                "user_id": user_id,
                "stream_callback": stream_callback,
                "remember": remember,
                # CENTRAL: Context Manager ist die primäre Context-Quelle
                "context_manager": self.context_manager,
                "variable_manager": self.variable_manager
            })

            # Set LLM models in shared context
            self.shared['fast_llm_model'] = self.amd.fast_llm_model
            self.shared['complex_llm_model'] = self.amd.complex_llm_model
            self.shared['persona_config'] = self.amd.persona
            self.shared['use_fast_response'] = self.amd.use_fast_response

            # Set system status
            self.shared["system_status"] = "running"
            self.is_running = True

            # Execute main orchestration flow
            result = await self._orchestrate_execution()

            #Store assistant response in ChatSession wenn remember=True
            if remember:
                await self.context_manager.add_interaction(
                    session_id,
                    'assistant',
                    result,
                    metadata={"user_id": user_id, "execution_duration": time.time() - execution_start}
                )

            total_duration = self.progress_tracker.end_timer("total_execution")

            await self.progress_tracker.emit_event(ProgressEvent(
                event_type="execution_complete",
                timestamp=time.time(),
                node_name="FlowAgent",
                status=NodeStatus.COMPLETED,
                node_duration=total_duration,
                session_id=session_id,
                metadata={
                    "result_length": len(result),
                    "summary": self.progress_tracker.get_summary(),
                    "remembered": remember
                }
            ))

            # Checkpoint if needed
            if self.enable_pause_resume:
                await self._maybe_checkpoint()

            return result

        except Exception as e:
            eprint(f"Agent execution failed: {e}", exc_info=True)
            error_response = f"I encountered an error: {str(e)}"
            result = error_response
            import traceback
            print(traceback.format_exc())

            # Store error in ChatSession wenn remember=True
            if remember:
                await self.context_manager.add_interaction(
                    session_id,
                    'assistant',
                    error_response,
                    metadata={
                        "user_id": user_id,
                        "error": True,
                        "error_type": type(e).__name__
                    }
                )

            total_duration = self.progress_tracker.end_timer("total_execution")

            await self.progress_tracker.emit_event(ProgressEvent(
                event_type="error",
                timestamp=time.time(),
                node_name="FlowAgent",
                status=NodeStatus.FAILED,
                node_duration=total_duration,
                session_id=session_id,
                metadata={"error": str(e), "error_type": type(e).__name__}
            ))

            return error_response

        finally:
            self.shared["system_status"] = "idle"
            self.is_running = False
            self.active_session = None

    def set_response_format(
        self,
        response_format: str,
        text_length: str,
        custom_instructions: str = "",
        quality_threshold: float = 0.7
    ):
        """Dynamische Format- und Längen-Konfiguration"""

        # Validiere Eingaben
        try:
            ResponseFormat(response_format)
            TextLength(text_length)
        except ValueError:
            available_formats = [f.value for f in ResponseFormat]
            available_lengths = [l.value for l in TextLength]
            raise ValueError(
                f"Invalid format or length. "
                f"Available formats: {available_formats}. "
                f"Available lengths: {available_lengths}"
            )

        # Erstelle oder aktualisiere Persona
        if not self.amd.persona:
            self.amd.persona = PersonaConfig(name="Assistant")

        # Erstelle Format-Konfiguration
        format_config = FormatConfig(
            response_format=ResponseFormat(response_format),
            text_length=TextLength(text_length),
            custom_instructions=custom_instructions,
            quality_threshold=quality_threshold
        )

        self.amd.persona.format_config = format_config

        # Aktualisiere Personality Traits mit Format-Hinweisen
        self._update_persona_with_format(response_format, text_length)

        # Update shared state
        self.shared["persona_config"] = self.amd.persona
        self.shared["format_config"] = format_config

        rprint(f"Response format set: {response_format}, length: {text_length}")

    def _update_persona_with_format(self, response_format: str, text_length: str):
        """Aktualisiere Persona-Traits basierend auf Format"""

        # Format-spezifische Traits
        format_traits = {
            "with-tables": ["structured", "data-oriented", "analytical"],
            "with-bullet-points": ["organized", "clear", "systematic"],
            "with-lists": ["methodical", "sequential", "thorough"],
            "md-text": ["technical", "formatted", "detailed"],
            "yaml-text": ["structured", "machine-readable", "precise"],
            "json-text": ["technical", "API-focused", "structured"],
            "text-only": ["conversational", "natural", "flowing"],
            "pseudo-code": ["logical", "algorithmic", "step-by-step"],
            "code-structure": ["technical", "systematic", "hierarchical"]
        }

        # Längen-spezifische Traits
        length_traits = {
            "mini-chat": ["concise", "quick", "to-the-point"],
            "chat-conversation": ["conversational", "friendly", "balanced"],
            "table-conversation": ["structured", "comparative", "organized"],
            "detailed-indepth": ["thorough", "comprehensive", "analytical"],
            "phd-level": ["academic", "scholarly", "authoritative"]
        }

        # Kombiniere Traits
        current_traits = set(self.amd.persona.personality_traits)

        # Entferne alte Format-Traits
        old_format_traits = set()
        for traits in format_traits.values():
            old_format_traits.update(traits)
        for traits in length_traits.values():
            old_format_traits.update(traits)

        current_traits -= old_format_traits

        # Füge neue Traits hinzu
        new_traits = format_traits.get(response_format, [])
        new_traits.extend(length_traits.get(text_length, []))

        current_traits.update(new_traits)
        self.amd.persona.personality_traits = list(current_traits)

    def get_available_formats(self) -> dict[str, list[str]]:
        """Erhalte verfügbare Format- und Längen-Optionen"""
        return {
            "formats": [f.value for f in ResponseFormat],
            "lengths": [l.value for l in TextLength],
            "format_descriptions": {
                f.value: FormatConfig(response_format=f).get_format_instructions()
                for f in ResponseFormat
            },
            "length_descriptions": {
                l.value: FormatConfig(text_length=l).get_length_instructions()
                for l in TextLength
            }
        }

    async def a_run_with_format(
        self,
        query: str,
        response_format: str = "frei-text",
        text_length: str = "chat-conversation",
        custom_instructions: str = "",
        **kwargs
    ) -> str:
        """Führe Agent mit spezifischem Format aus"""

        # Temporäre Format-Einstellung
        original_persona = self.amd.persona

        try:
            self.set_response_format(response_format, text_length, custom_instructions)
            response = await self.a_run(query, **kwargs)
            return response
        finally:
            # Restore original persona
            self.amd.persona = original_persona
            self.shared["persona_config"] = original_persona

    def get_format_quality_report(self) -> dict[str, Any]:
        """Erhalte detaillierten Format-Qualitätsbericht"""
        quality_assessment = self.shared.get("quality_assessment", {})

        if not quality_assessment:
            return {"status": "no_assessment", "message": "No recent quality assessment available"}

        quality_details = quality_assessment.get("quality_details", {})

        return {
            "overall_score": quality_details.get("total_score", 0.0),
            "format_adherence": quality_details.get("format_adherence", 0.0),
            "length_adherence": quality_details.get("length_adherence", 0.0),
            "content_quality": quality_details.get("base_quality", 0.0),
            "llm_assessment": quality_details.get("llm_assessment", 0.0),
            "suggestions": quality_assessment.get("suggestions", []),
            "assessment": quality_assessment.get("quality_assessment", "unknown"),
            "format_config_active": quality_details.get("format_config_used", False)
        }

    def get_variable_documentation(self) -> str:
        """Get comprehensive variable system documentation"""
        docs = []
        docs.append("# Variable System Documentation\n")

        # Available scopes
        docs.append("## Available Scopes:")
        scope_info = self.variable_manager.get_scope_info()
        for scope_name, info in scope_info.items():
            docs.append(f"- `{scope_name}`: {info['type']} with {info.get('keys', 'N/A')} keys")

        docs.append("\n## Syntax Options:")
        docs.append("- `{{ variable.path }}` - Full path resolution")
        docs.append("- `{variable}` - Simple variable (no dots)")
        docs.append("- `$variable` - Shell-style variable")

        docs.append("\n## Example Usage:")
        docs.append("- `{{ results.task_1.data }}` - Get result from task_1")
        docs.append("- `{{ user.name }}` - Get user name")
        docs.append("- `{agent_name}` - Simple agent name")
        docs.append("- `$timestamp` - System timestamp")

        # Available variables
        docs.append("\n## Available Variables:")
        variables = self.variable_manager.get_available_variables()
        for scope_name, scope_vars in variables.items():
            docs.append(f"\n### {scope_name}:")
            for _var_name, var_info in scope_vars.items():
                docs.append(f"- `{var_info['path']}`: {var_info['preview']} ({var_info['type']})")

        return "\n".join(docs)

    def _setup_variable_scopes(self):
        """Setup default variable scopes with enhanced structure"""
        self.variable_manager.register_scope('agent', {
            'name': self.amd.name,
            'model_fast': self.amd.fast_llm_model,
            'model_complex': self.amd.complex_llm_model
        })

        timestamp = datetime.now()
        self.variable_manager.register_scope('system', {
            'timestamp': timestamp.isoformat(),
            'version': '2.0',
            'capabilities': list(self._tool_capabilities.keys())
        })

        # ADDED: Initialize empty results and tasks scopes
        self.variable_manager.register_scope('results', {})
        self.variable_manager.register_scope('tasks', {})

        # Update shared state
        self.shared["variable_manager"] = self.variable_manager

    def set_variable(self, path: str, value: Any):
        """Set variable using unified system"""
        self.variable_manager.set(path, value)

    def get_variable(self, path: str, default=None):
        """Get variable using unified system"""
        return self.variable_manager.get(path, default)

    def format_text(self, text: str, **context) -> str:
        """Format text with variables"""
        return self.variable_manager.format_text(text, context)

    async def initialize_session_context(self, session_id: str = "default", max_history: int = 200) -> bool:
        """Vereinfachte Session-Initialisierung über UnifiedContextManager"""
        try:
            # Delegation an UnifiedContextManager
            session = await self.context_manager.initialize_session(session_id, max_history)

            # Ensure Variable Manager integration
            if not self.context_manager.variable_manager:
                self.context_manager.variable_manager = self.variable_manager

            # Update shared state (minimal - primary data now in context_manager)
            self.shared["active_session_id"] = session_id
            self.shared["session_initialized"] = True

            # Legacy support: Keep session_managers reference in shared for backward compatibility
            self.shared["session_managers"] = self.context_manager.session_managers

            rprint(f"Session context initialized for {session_id} via UnifiedContextManager")
            return True

        except Exception as e:
            eprint(f"Session context initialization failed: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    async def initialize_context_awareness(self):
        """Enhanced context awareness with session management"""

        # Initialize session if not already done
        session_id = self.shared.get("session_id", self.active_session)
        if not self.shared.get("session_initialized"):
            await self.initialize_session_context(session_id)

        # Ensure tool capabilities are loaded
        # add tqdm prigress bar

        from tqdm import tqdm

        if hasattr(self.task_flow, 'llm_reasoner'):
            if "read_from_variables" not in self.shared["available_tools"] and hasattr(self.task_flow.llm_reasoner, '_execute_read_from_variables'):
                await self.add_tool(lambda scope, key, purpose: self.task_flow.llm_reasoner._execute_read_from_variables({"scope": scope, "key": key, "purpose": purpose}), "read_from_variables", "Read from variables")
            if "write_to_variables" not in self.shared["available_tools"] and hasattr(self.task_flow.llm_reasoner, '_execute_write_to_variables'):
                await self.add_tool(lambda scope, key, value, description: self.task_flow.llm_reasoner._execute_write_to_variables({"scope": scope, "key": key, "value": value, "description": description}), "write_to_variables", "Write to variables")

            if "internal_reasoning" not in self.shared["available_tools"] and hasattr(self.task_flow.llm_reasoner, '_execute_internal_reasoning'):
                async def internal_reasoning_tool(thought:str, thought_number:int, total_thoughts:int, next_thought_needed:bool, current_focus:str, key_insights:list[str], potential_issues:list[str], confidence_level:float):
                    args = {
                        "thought": thought,
                        "thought_number": thought_number,
                        "total_thoughts": total_thoughts,
                        "next_thought_needed": next_thought_needed,
                        "current_focus": current_focus,
                        "key_insights": key_insights,
                        "potential_issues": potential_issues,
                        "confidence_level": confidence_level
                    }
                    return await self.task_flow.llm_reasoner._execute_internal_reasoning(args, self.shared)
                await self.add_tool(internal_reasoning_tool, "internal_reasoning", "Internal reasoning")

            if "manage_internal_task_stack" not in self.shared["available_tools"] and hasattr(self.task_flow.llm_reasoner, '_execute_manage_task_stack'):
                async def manage_internal_task_stack_tool(action:str, task_description:str, outline_step_ref:str):
                    args = {
                        "action": action,
                        "task_description": task_description,
                        "outline_step_ref": outline_step_ref
                    }
                    return await self.task_flow.llm_reasoner._execute_manage_task_stack(args, self.shared)
                await self.add_tool(manage_internal_task_stack_tool, "manage_internal_task_stack", "Manage internal task stack")

            if "outline_step_completion" not in self.shared["available_tools"] and hasattr(self.task_flow.llm_reasoner, '_execute_outline_step_completion'):
                async def outline_step_completion_tool(step_completed:bool, completion_evidence:str, next_step_focus:str):
                    args = {
                        "step_completed": step_completed,
                        "completion_evidence": completion_evidence,
                        "next_step_focus": next_step_focus
                    }
                    return await self.task_flow.llm_reasoner._execute_outline_step_completion(args, self.shared)
                await self.add_tool(outline_step_completion_tool, "outline_step_completion", "Outline step completion")


        registered_tools = set(self._tool_registry.keys())
        cached_capabilities = list(self._tool_capabilities.keys())  # Create a copy of
        for tool_name in cached_capabilities:
            if tool_name in self._tool_capabilities and tool_name not in registered_tools:
                del self._tool_capabilities[tool_name]
                print(f"Removed outdated capability for unavailable tool: {tool_name}")

        for tool_name in tqdm(self.shared["available_tools"], desc=f"Agent {self.amd.name} Analyzing Tools", unit="tool", colour="green", total=len(self.shared["available_tools"])):
            if tool_name not in self._tool_capabilities:
                tool_info = self._tool_registry.get(tool_name, {})
                description = tool_info.get("description", "No description")
                with Spinner(f"Analyzing tool {tool_name}"):
                    await self._analyze_tool_capabilities(tool_name, description, tool_info.get("args_schema", "()"))

            if tool_name in self._tool_capabilities:
                function = self._tool_registry[tool_name]["function"]
                self._tool_capabilities[tool_name]["args_schema"] = get_args_schema(function)

        # Set enhanced system context
        self.shared["system_context"] = {
            "capabilities_summary": self._build_capabilities_summary(),
            "tool_count": len(self.shared["available_tools"]),
            "analysis_loaded": len(self._tool_capabilities),
            "intelligence_level": "high" if self._tool_capabilities else "basic",
            "context_management": "advanced_session_aware",
            "session_managers": len(self.shared.get("session_managers", {})),
        }


        rprint("Advanced context awareness initialized with session management")

    async def get_context(self, session_id: str = None, format_for_llm: bool = True) -> str | dict[str, Any]:
        """
        ÜBERARBEITET: Get context über UnifiedContextManager statt verteilte Quellen
        """
        try:
            session_id = session_id or self.shared.get("session_id", self.active_session)
            query = self.shared.get("current_query", "")

            #Hole unified context über Context Manager
            unified_context = await self.context_manager.build_unified_context(session_id, query, "full")


            if format_for_llm:
                return self.context_manager.get_formatted_context_for_llm(unified_context)
            else:
                return unified_context

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            eprint(f"Failed to generate context via UnifiedContextManager: {e}")

            # FALLBACK: Fallback zu alter Methode falls UnifiedContextManager fehlschlägt
            if format_for_llm:
                return f"Error generating context: {str(e)}"
            else:
                return {
                    "error": str(e),
                    "generated_at": datetime.now().isoformat(),
                    "fallback_mode": True
                }

    def get_context_statistics(self) -> dict[str, Any]:
        """Get comprehensive context management statistics"""
        stats = {
            "context_system": "advanced_session_aware",
            "compression_threshold": 0.76,
            "max_tokens": getattr(self, 'max_input_tokens', 8000),
            "session_managers": {},
            "context_usage": {},
            "compression_stats": {}
        }

        # Session manager statistics
        session_managers = self.shared.get("session_managers", {})
        for name, manager in session_managers.items():
            stats["session_managers"][name] = {
                "history_length": len(manager.history),
                "max_length": manager.max_length,
                "space_name": manager.space_name
            }

        # Context node statistics if available
        if hasattr(self.task_flow, 'context_manager'):
            context_manager = self.task_flow.context_manager
            stats["compression_stats"] = {
                "compression_threshold": context_manager.compression_threshold,
                "max_tokens": context_manager.max_tokens,
                "active_sessions": len(context_manager.session_managers)
            }

        # LLM call statistics from enhanced node
        llm_stats = self.shared.get("llm_call_stats", {})
        if llm_stats:
            stats["context_usage"] = {
                "total_llm_calls": llm_stats.get("total_calls", 0),
                "context_compression_rate": llm_stats.get("context_compression_rate", 0.0),
                "average_context_tokens": llm_stats.get("context_tokens_used", 0) / max(llm_stats.get("total_calls", 1),
                                                                                        1)
            }

        return stats

    def set_persona(self, name: str, style: str = "professional", tone: str = "friendly",
                    personality_traits: list[str] = None, apply_method: str = "system_prompt",
                    integration_level: str = "light", custom_instructions: str = ""):
        """Set agent persona mit erweiterten Konfigurationsmöglichkeiten"""
        if personality_traits is None:
            personality_traits = ["helpful", "concise"]

        self.amd.persona = PersonaConfig(
            name=name,
            style=style,
            tone=tone,
            personality_traits=personality_traits,
            custom_instructions=custom_instructions,
            apply_method=apply_method,
            integration_level=integration_level
        )

        rprint(f"Persona set: {name} ({style}, {tone}) - Method: {apply_method}, Level: {integration_level}")

    def configure_persona_integration(self, apply_method: str = "system_prompt", integration_level: str = "light"):
        """Configure how persona is applied"""
        if self.amd.persona:
            self.amd.persona.apply_method = apply_method
            self.amd.persona.integration_level = integration_level
            rprint(f"Persona integration updated: {apply_method}, {integration_level}")
        else:
            wprint("No persona configured to update")

    def get_available_variables(self) -> dict[str, str]:
        """Get available variables for dynamic formatting"""
        return self.variable_manager.get_available_variables()

    async def _orchestrate_execution(self) -> str:
        """
        Enhanced orchestration with LLMReasonerNode as strategic core.
        The reasoner now handles both task management and response generation internally.
        """

        self.shared["agent_instance"] = self
        self.shared["session_id"] = self.active_session
        # === UNIFIED REASONING AND EXECUTION CYCLE ===
        rprint("Starting strategic reasoning and execution cycle")

        # The LLMReasonerNode now handles the complete cycle:
        # 1. Strategic analysis of the query
        # 2. Decision making about approach
        # 3. Orchestration of sub-systems (LLMToolNode, TaskPlanner/Executor)
        # 4. Response synthesis and formatting

        # Execute the unified flow
        task_management_result = await self.task_flow.run_async(self.shared)

        # Check for various completion states
        if self.shared.get("plan_halted"):
            error_response = f"Task execution was halted: {self.shared.get('halt_reason', 'Unknown reason')}"
            self.shared["current_response"] = error_response
            return error_response

        final_response = self.shared.get("current_response", "Task completed successfully.")
        # Execute ResponseGenerationFlow for persona application and formatting
        response_result = await self.response_flow.run_async(self.shared)

        # The reasoner provides the final response
        final_response = self.shared.get("current_response", "Task completed successfully.")

        # Add reasoning artifacts to response if available
        reasoning_artifacts = self.shared.get("reasoning_artifacts", {})
        if reasoning_artifacts and reasoning_artifacts.get("reasoning_loops", 0) > 1:
            # For debugging/transparency, could add reasoning info to metadata
            pass

        # Log enhanced statistics
        self._log_execution_stats()

        return final_response

    def _log_execution_stats(self):
        """Enhanced execution statistics with reasoning metrics"""
        tasks = self.shared.get("tasks", {})
        adaptations = self.shared.get("plan_adaptations", 0)
        reasoning_artifacts = self.shared.get("reasoning_artifacts", {})

        completed_tasks = sum(1 for t in tasks.values() if t.status == "completed")
        failed_tasks = sum(1 for t in tasks.values() if t.status == "failed")

        # Enhanced logging with reasoning metrics
        reasoning_loops = reasoning_artifacts.get("reasoning_loops", 0)

        stats_message = f"Execution complete - Tasks: {completed_tasks} completed, {failed_tasks} failed"

        if adaptations > 0:
            stats_message += f", {adaptations} adaptations"

        if reasoning_loops > 0:
            stats_message += f", {reasoning_loops} reasoning loops"

            # Add reasoning efficiency metric
            if completed_tasks > 0:
                efficiency = completed_tasks / max(reasoning_loops, 1)
                stats_message += f" (efficiency: {efficiency:.1f} tasks/loop)"

        rprint(stats_message)

        # Log reasoning context if significant
        if reasoning_loops > 3:
            internal_task_stack = reasoning_artifacts.get("internal_task_stack", [])
            completed_reasoning_tasks = len([t for t in internal_task_stack if t.get("status") == "completed"])

            if completed_reasoning_tasks > 0:
                rprint(f"Strategic reasoning: {completed_reasoning_tasks} high-level tasks completed")

    def _build_capabilities_summary(self) -> str:
        """Build summary of agent capabilities"""

        if not self._tool_capabilities:
            return "Basic LLM capabilities only"

        summaries = []
        for tool_name, cap in self._tool_capabilities.items():
            primary = cap.get('primary_function', 'Unknown function')
            summaries.append(f"{tool_name}{cap.get('args_schema', '()')}: {primary}")

        return f"Enhanced capabilities: {'; '.join(summaries)}"

    # Neue Hilfsmethoden für erweiterte Funktionalität

    async def get_task_execution_summary(self) -> dict[str, Any]:
        """Erhalte detaillierte Zusammenfassung der Task-Ausführung"""
        tasks = self.shared.get("tasks", {})
        results_store = self.shared.get("results", {})

        summary = {
            "total_tasks": len(tasks),
            "completed_tasks": [],
            "failed_tasks": [],
            "task_types_used": {},
            "tools_used": [],
            "adaptations": self.shared.get("plan_adaptations", 0),
            "execution_timeline": []
        }

        for task_id, task in tasks.items():
            task_info = {
                "id": task_id,
                "type": task.type,
                "description": task.description,
                "status": task.status,
                "duration": None
            }

            if task.started_at and task.completed_at:
                duration = (task.completed_at - task.started_at).total_seconds()
                task_info["duration"] = duration

            if task.status == "completed":
                summary["completed_tasks"].append(task_info)
                if isinstance(task, ToolTask):
                    summary["tools_used"].append(task.tool_name)
            elif task.status == "failed":
                task_info["error"] = task.error
                summary["failed_tasks"].append(task_info)

            # Task types counting
            task_type = task.type
            summary["task_types_used"][task_type] = summary["task_types_used"].get(task_type, 0) + 1

        return summary

    async def explain_reasoning_process(self) -> str:
        """Erkläre den Reasoning-Prozess des Agenten"""
        if not LITELLM_AVAILABLE:
            return "Reasoning explanation requires LLM capabilities."

        summary = await self.get_task_execution_summary()

        prompt = f"""
Erkläre den Reasoning-Prozess dieses AI-Agenten in verständlicher Form:

## Ausführungszusammenfassung
- Total Tasks: {summary['total_tasks']}
- Erfolgreich: {len(summary['completed_tasks'])}
- Fehlgeschlagen: {len(summary['failed_tasks'])}
- Plan-Adaptationen: {summary['adaptations']}
- Verwendete Tools: {', '.join(set(summary['tools_used']))}
- Task-Typen: {summary['task_types_used']}

## Task-Details
Erfolgreiche Tasks:
{self._format_tasks_for_explanation(summary['completed_tasks'])}

## Anweisungen
Erkläre in 2-3 Absätzen:
1. Welche Strategie der Agent gewählt hat
2. Wie er die Aufgabe in Tasks unterteilt hat
3. Wie er auf unerwartete Ergebnisse reagiert hat (falls Adaptationen)
4. Was die wichtigsten Erkenntnisse waren

Schreibe für einen technischen Nutzer, aber verständlich."""

        try:
            response = await self.a_run_llm_completion(
                model=self.amd.complex_llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=800,task_id="reasoning_explanation"
            )

            return response

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return f"Could not generate reasoning explanation: {e}"

    def _format_tasks_for_explanation(self, tasks: list[dict]) -> str:
        formatted = []
        for task in tasks[:5]:  # Top 5 tasks
            duration_info = f" ({task['duration']:.1f}s)" if task['duration'] else ""
            formatted.append(f"- {task['type']}: {task['description']}{duration_info}")
        return "\n".join(formatted)

    # ===== PAUSE/RESUME FUNCTIONALITY =====

    async def pause(self) -> bool:
        """Pause agent execution"""
        if not self.is_running:
            return False

        self.is_paused = True
        self.shared["system_status"] = "paused"

        # Create checkpoint
        checkpoint = await self._create_checkpoint()
        await self._save_checkpoint(checkpoint)

        rprint("Agent execution paused")
        return True

    async def resume(self) -> bool:
        """Resume agent execution"""
        if not self.is_paused:
            return False

        self.is_paused = False
        self.shared["system_status"] = "running"

        rprint("Agent execution resumed")
        return True

    # ===== CHECKPOINT MANAGEMENT =====

    async def _create_checkpoint(self) -> AgentCheckpoint:
        """Vereinfachte Checkpoint-Erstellung - fokussiert auf wesentliche Daten"""
        try:
            # Budget Manager Daten vor Checkpoint speichern
            if hasattr(self.amd, 'budget_manager') and self.amd.budget_manager:
                self.amd.budget_manager.save_data()

            # Bereite AMD-Daten vor (ohne budget_manager für Serialisierung)
            amd_data = self.amd.model_dump()
            amd_data['budget_manager'] = None

            # Sammle wesentliche Session-Daten (vereinfacht)
            session_data = {}
            if self.context_manager and self.context_manager.session_managers:
                for session_id, session in self.context_manager.session_managers.items():
                    try:
                        if hasattr(session, 'history') and session.history:
                            # Nur die letzten 20 Nachrichten pro Session für Checkpoint
                            recent_history = session.history[-20:]
                            session_data[session_id] = {
                                "history": recent_history,
                                "session_type": "chatsession",
                                "message_count": len(session.history)
                            }
                        elif isinstance(session, dict) and session.get('history'):
                            session_data[session_id] = {
                                "history": session['history'][-20:],
                                "session_type": "fallback",
                                "message_count": len(session['history'])
                            }
                    except Exception as e:
                        rprint(f"Skipping session {session_id} in checkpoint: {e}")

            # Sammle serialisierbare Variable-Scopes
            variable_scopes = {}
            if self.variable_manager:
                NON_SERIALIZABLE_KEYS = {
                    "tool_registry", "variable_manager", "context_manager", "agent_instance",
                    "llm_tool_node_instance", "task_planner_instance", "task_executor_instance",
                    "progress_tracker", "session_managers", "stream_callback"
                }

                for scope_name, scope_data in self.variable_manager.scopes.items():
                    if isinstance(scope_data, dict):
                        # Filtere nicht-serialisierbare Objekte heraus
                        clean_scope = {
                            k: v for k, v in scope_data.items()
                            if k not in NON_SERIALIZABLE_KEYS
                        }
                        variable_scopes[scope_name] = clean_scope
                    else:
                        try:
                            # Teste Serialisierbarkeit
                            pickle.dumps(scope_data)
                            variable_scopes[scope_name] = scope_data
                        except:
                            # Überspringe nicht-serialisierbare Scopes
                            pass

            # Erstelle konsolidierten Checkpoint
            checkpoint = AgentCheckpoint(
                timestamp=datetime.now(),
                agent_state={
                    "is_running": self.is_running,
                    "is_paused": self.is_paused,
                    "amd_data": amd_data,
                    "active_session": self.active_session,
                    "system_status": self.shared.get("system_status", "idle")
                },
                task_state={
                    task_id: asdict(task) for task_id, task in self.shared.get("tasks", {}).items()
                },
                world_model=self.shared.get("world_model", {}),
                active_flows=["task_flow", "response_flow"],
                metadata={
                    "session_id": self.shared.get("session_id", "default"),
                    "last_query": self.shared.get("current_query", ""),
                    "checkpoint_version": "3.0_simplified",
                    "agent_name": self.amd.name
                },
                # Konsolidierte Zusatzdaten
                session_data=session_data,
                variable_scopes=variable_scopes,
                results_store=self.shared.get("results", {}),
                conversation_history=self.shared.get("conversation_history", [])[-50:],  # Letzte 50 Nachrichten
                tool_capabilities=self._tool_capabilities.copy()
            )

            rprint(f"Vereinfachter Checkpoint erstellt mit {len(session_data)} Sessions")
            return checkpoint

        except Exception as e:
            eprint(f"Checkpoint-Erstellung fehlgeschlagen: {e}")
            import traceback
            print(traceback.format_exc())
            raise

    async def _save_checkpoint(self, checkpoint: AgentCheckpoint, filepath: str = None):
        """Vereinfachtes Checkpoint-Speichern - alles in eine Datei"""
        try:
            from toolboxv2 import get_app
            folder = str(get_app().data_dir) + '/Agents/checkpoint/' + self.amd.name
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)

            if not filepath:
                timestamp = checkpoint.timestamp.strftime("%Y%m%d_%H%M%S")
                filepath = f"agent_checkpoint_{timestamp}.pkl"
            filepath = os.path.join(folder, filepath)

            # Sessions vor dem Speichern synchronisieren
            if self.context_manager and self.context_manager.session_managers:
                for session_id, session in self.context_manager.session_managers.items():
                    try:
                        if hasattr(session, 'save'):
                            await session.save()
                        elif hasattr(session, '_save_to_memory'):
                            session._save_to_memory()
                    except Exception as e:
                        rprint(f"Session sync error für {session_id}: {e}")

            # Speichere Checkpoint direkt
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint, f)

            self.last_checkpoint = checkpoint.timestamp

            # Erstelle einfache Zusammenfassung
            summary_parts = []
            if hasattr(checkpoint, 'session_data') and checkpoint.session_data:
                summary_parts.append(f"{len(checkpoint.session_data)} sessions")
            if checkpoint.task_state:
                completed_tasks = len([t for t in checkpoint.task_state.values() if t.get("status") == "completed"])
                summary_parts.append(f"{completed_tasks} completed tasks")
            if hasattr(checkpoint, 'variable_scopes') and checkpoint.variable_scopes:
                summary_parts.append(f"{len(checkpoint.variable_scopes)} variable scopes")

            summary = "; ".join(summary_parts) if summary_parts else "Basic checkpoint"
            rprint(f"Checkpoint gespeichert: {filepath} ({summary})")
            return True

        except Exception as e:
            eprint(f"Checkpoint-Speicherung fehlgeschlagen: {e}")
            print(checkpoint)
            return False

    async def load_latest_checkpoint(self, auto_restore_history: bool = True, max_age_hours: int = 24) -> dict[
        str, Any]:
        """Vereinfachtes Checkpoint-Laden mit automatischer History-Wiederherstellung"""
        try:
            from toolboxv2 import get_app
            folder = str(get_app().data_dir) + '/Agents/checkpoint/' + self.amd.name

            if not os.path.exists(folder):
                return {"success": False, "error": "Kein Checkpoint-Verzeichnis gefunden"}

            # Finde neuesten Checkpoint
            checkpoint_files = []
            for file in os.listdir(folder):
                if file.endswith('.pkl') and file.startswith('agent_checkpoint_'):
                    filepath = os.path.join(folder, file)
                    try:
                        timestamp_str = file.replace('agent_checkpoint_', '').replace('.pkl', '')
                        if timestamp_str == 'final_checkpoint':
                            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                        else:
                            file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                        age_hours = (datetime.now() - file_time).total_seconds() / 3600
                        if age_hours <= max_age_hours:
                            checkpoint_files.append((filepath, file_time, age_hours))
                    except Exception:
                        continue

            if not checkpoint_files:
                return {"success": False, "error": f"Keine gültigen Checkpoints in {max_age_hours} Stunden gefunden"}

            # Lade neuesten Checkpoint
            checkpoint_files.sort(key=lambda x: x[1], reverse=True)
            latest_checkpoint_path, latest_timestamp, age_hours = checkpoint_files[0]

            rprint(f"Lade Checkpoint: {latest_checkpoint_path} (Alter: {age_hours:.1f}h)")

            with open(latest_checkpoint_path, 'rb') as f:
                checkpoint: AgentCheckpoint = pickle.load(f)

            # Stelle Agent-Status wieder her
            restore_stats = await self._restore_from_checkpoint_simplified(checkpoint, auto_restore_history)

            # Re-initialisiere Kontext-Awareness
            await self.initialize_context_awareness()

            return {
                "success": True,
                "checkpoint_file": latest_checkpoint_path,
                "checkpoint_age_hours": age_hours,
                "checkpoint_timestamp": latest_timestamp.isoformat(),
                "available_checkpoints": len(checkpoint_files),
                "restore_stats": restore_stats
            }

        except Exception as e:
            eprint(f"Checkpoint-Laden fehlgeschlagen: {e}")
            import traceback
            print(traceback.format_exc())
            return {"success": False, "error": str(e)}

    async def _restore_from_checkpoint_simplified(self, checkpoint: AgentCheckpoint, auto_restore_history: bool) -> \
    dict[
        str, Any]:
        """Vereinfachte Checkpoint-Wiederherstellung"""
        restore_stats = {
            "agent_state_restored": False,
            "world_model_restored": False,
            "tasks_restored": 0,
            "sessions_restored": 0,
            "variables_restored": 0,
            "conversation_restored": 0,
            "errors": []
        }

        try:
            # 1. Agent-Status wiederherstellen
            if checkpoint.agent_state:
                self.is_running = checkpoint.agent_state.get("is_running", False)
                self.is_paused = checkpoint.agent_state.get("is_paused", False)
                self.active_session = checkpoint.agent_state.get("active_session")

                # AMD-Daten selektiv wiederherstellen
                amd_data = checkpoint.agent_state.get("amd_data", {})
                if amd_data:
                    # Nur sichere Felder wiederherstellen
                    safe_fields = ["name", "use_fast_response", "max_input_tokens"]
                    for field in safe_fields:
                        if field in amd_data and hasattr(self.amd, field):
                            setattr(self.amd, field, amd_data[field])

                    # Persona wiederherstellen falls vorhanden
                    if "persona" in amd_data and amd_data["persona"]:
                        try:
                            persona_data = amd_data["persona"]
                            if isinstance(persona_data, dict):
                                self.amd.persona = PersonaConfig(**persona_data)
                        except Exception as e:
                            restore_stats["errors"].append(f"Persona restore failed: {e}")

                restore_stats["agent_state_restored"] = True

            # 2. World Model wiederherstellen
            if checkpoint.world_model:
                self.shared["world_model"] = checkpoint.world_model.copy()
                self.world_model = checkpoint.world_model.copy()
                restore_stats["world_model_restored"] = True

            # 3. Variable System wiederherstellen
            if hasattr(checkpoint, 'variable_scopes') and checkpoint.variable_scopes:
                # Variable Manager neu initialisieren
                self.variable_manager = VariableManager(self.shared["world_model"], self.shared)

                # Basis-Scopes einrichten
                self._setup_variable_scopes()

                # Gespeicherte Scopes wiederherstellen
                for scope_name, scope_data in checkpoint.variable_scopes.items():
                    try:
                        self.variable_manager.register_scope(scope_name, scope_data)
                        restore_stats["variables_restored"] += 1
                    except Exception as e:
                        restore_stats["errors"].append(f"Variable scope {scope_name}: {e}")

                # Runtime-Objekte wieder einsetzen
                self.variable_manager.set("shared", "variable_manager", self.variable_manager)
                self.variable_manager.set("shared", "context_manager", self.context_manager)
                self.variable_manager.set("shared", "agent_instance", self)

                self.shared["variable_manager"] = self.variable_manager

            # 4. Tasks wiederherstellen
            if checkpoint.task_state:
                restored_tasks = {}
                for task_id, task_data in checkpoint.task_state.items():
                    try:
                        task_type = task_data.get("type", "generic")
                        if task_type == "LLMTask":
                            restored_tasks[task_id] = LLMTask(**task_data)
                        elif task_type == "ToolTask":
                            restored_tasks[task_id] = ToolTask(**task_data)
                        elif task_type == "DecisionTask":
                            restored_tasks[task_id] = DecisionTask(**task_data)
                        else:
                            restored_tasks[task_id] = Task(**task_data)

                        restore_stats["tasks_restored"] += 1
                    except Exception as e:
                        restore_stats["errors"].append(f"Task {task_id}: {e}")

                self.shared["tasks"] = restored_tasks

            # 5. Results Store wiederherstellen
            if hasattr(checkpoint, 'results_store') and checkpoint.results_store:
                self.shared["results"] = checkpoint.results_store
                if self.variable_manager:
                    self.variable_manager.set_results_store(checkpoint.results_store)

            # 6. Sessions und Conversation wiederherstellen (falls gewünscht)
            if auto_restore_history:
                await self._restore_sessions_and_conversation_simplified(checkpoint, restore_stats)

            # 7. Tool Capabilities wiederherstellen
            if hasattr(checkpoint, 'tool_capabilities') and checkpoint.tool_capabilities:
                self._tool_capabilities = checkpoint.tool_capabilities.copy()

            # Status setzen
            self.shared["system_status"] = "restored"
            restore_stats["restoration_timestamp"] = datetime.now().isoformat()

            rprint(
                f"Checkpoint wiederhergestellt: {restore_stats['tasks_restored']} Tasks, {restore_stats['sessions_restored']} Sessions, {len(restore_stats['errors'])} Fehler")
            return restore_stats

        except Exception as e:
            eprint(f"Checkpoint-Wiederherstellung fehlgeschlagen: {e}")
            import traceback
            print(traceback.format_exc())
            restore_stats["errors"].append(f"Critical restore error: {e}")
            return restore_stats

    async def _restore_sessions_and_conversation_simplified(self, checkpoint: AgentCheckpoint, restore_stats: dict):
        """Vereinfachte Session- und Conversation-Wiederherstellung"""
        try:
            # Context Manager sicherstellen
            if not self.context_manager:
                self.context_manager = UnifiedContextManager(self)
                self.context_manager.variable_manager = self.variable_manager

            # Sessions wiederherstellen
            if hasattr(checkpoint, 'session_data') and checkpoint.session_data:
                for session_id, session_info in checkpoint.session_data.items():
                    try:
                        # Session über Context Manager initialisieren
                        max_length = session_info.get("message_count", 200)
                        restored_session = await self.context_manager.initialize_session(session_id, max_length)

                        # History wiederherstellen
                        history = session_info.get("history", [])
                        if history and hasattr(restored_session, 'history'):
                            # Direkt in Session-History einfügen
                            restored_session.history.extend(history)

                        restore_stats["sessions_restored"] += 1
                    except Exception as e:
                        restore_stats["errors"].append(f"Session {session_id}: {e}")

            # Conversation History wiederherstellen
            if hasattr(checkpoint, 'conversation_history') and checkpoint.conversation_history:
                self.shared["conversation_history"] = checkpoint.conversation_history
                restore_stats["conversation_restored"] = len(checkpoint.conversation_history)

            # Update shared context
            self.shared["context_manager"] = self.context_manager
            if self.context_manager.session_managers:
                self.shared["session_managers"] = self.context_manager.session_managers
                self.shared["session_initialized"] = True

        except Exception as e:
            restore_stats["errors"].append(f"Session/conversation restore failed: {e}")

    async def _maybe_checkpoint(self):
        """Vereinfachtes automatisches Checkpointing"""
        if not self.enable_pause_resume:
            return

        now = datetime.now()
        if (not self.last_checkpoint or
            (now - self.last_checkpoint).seconds >= self.checkpoint_interval):

            try:
                checkpoint = await self._create_checkpoint()
                await self._save_checkpoint(checkpoint)
            except Exception as e:
                eprint(f"Automatic checkpoint failed: {e}")


    async def save_context_to_session(self, session_id: str = None, context_type: str = "full") -> bool:
        """Save current context to ChatSession for persistent storage"""
        try:
            session_id = session_id or self.shared.get("session_id", "default")

            if not self.context_manager:
                eprint("Context manager not available")
                return False

            # Build comprehensive context
            unified_context = await self.context_manager.build_unified_context(session_id, None, context_type)

            # Create context message for session storage
            context_message = {
                "role": "system",
                "content": f"[CONTEXT_SNAPSHOT_{context_type.upper()}] " + json.dumps(unified_context, default=str),
                "timestamp": datetime.now().isoformat(),
                "context_type": context_type,
                "metadata": {
                    "is_context_snapshot": True,
                    "context_version": "2.0",
                    "agent_name": self.amd.name,
                    "session_stats": unified_context.get("session_stats", {}),
                    "variables_count": len(unified_context.get("variables", {}).get("recent_results", [])),
                    "execution_state": unified_context.get("execution_state", {}).get("system_status", "unknown")
                }
            }

            # Store in session
            await self.context_manager.add_interaction(
                session_id,
                "system",
                context_message["content"],
                metadata=context_message["metadata"]
            )

            rprint(f"Context snapshot saved to session {session_id} (type: {context_type})")
            return True

        except Exception as e:
            eprint(f"Failed to save context to session: {e}")
            return False

    async def load_context_from_session(self, session_id: str, context_type: str = "full") -> dict[str, Any]:
        """Load context from ChatSession storage"""
        try:
            if not self.context_manager:
                return {"error": "Context manager not available"}

            session = self.context_manager.session_managers.get(session_id)
            if not session:
                return {"error": f"Session {session_id} not found"}

            # Search for context snapshots in session history
            context_snapshots = []

            if hasattr(session, 'history'):
                for message in reversed(session.history):  # Search from newest
                    if (message.get("role") == "system" and
                        message.get("metadata", {}).get("is_context_snapshot") and
                        message.get("metadata", {}).get("context_type") == context_type):

                        try:
                            # Extract context data
                            content = message.get("content", "")
                            if content.startswith(f"[CONTEXT_SNAPSHOT_{context_type.upper()}]"):
                                json_data = content.replace(f"[CONTEXT_SNAPSHOT_{context_type.upper()}] ", "")
                                context_data = json.loads(json_data)
                                context_snapshots.append({
                                    "context": context_data,
                                    "timestamp": message.get("timestamp"),
                                    "metadata": message.get("metadata", {})
                                })
                        except Exception as e:
                            wprint(f"Failed to parse context snapshot: {e}")

            if context_snapshots:
                # Return most recent context snapshot
                latest_context = context_snapshots[0]
                rprint(f"Loaded context snapshot from session {session_id} (timestamp: {latest_context['timestamp']})")
                return latest_context["context"]
            else:
                return {"error": f"No context snapshots of type '{context_type}' found in session {session_id}"}

        except Exception as e:
            eprint(f"Failed to load context from session: {e}")
            return {"error": str(e)}

    async def cleanup_session_context(self, session_id: str = None, keep_count: int = 100,
                                      remove_old_snapshots: bool = True) -> dict[str, Any]:
        """Cleanup session context by removing old snapshots and entries"""
        try:
            session_id = session_id or self.shared.get("session_id", "default")

            if not self.context_manager:
                return {"error": "Context manager not available"}

            session = self.context_manager.session_managers.get(session_id)
            if not session or not hasattr(session, 'history'):
                return {"error": f"Session {session_id} not found or has no history"}

            cleanup_stats = {
                "original_message_count": len(session.history),
                "context_snapshots_removed": 0,
                "context_entries_removed": 0,
                "regular_messages_kept": 0,
                "cleanup_performed": False
            }

            if len(session.history) <= keep_count:
                return {**cleanup_stats, "message": "No cleanup needed"}

            # Separate different types of messages
            regular_messages = []
            context_snapshots = []
            context_entries = []

            for message in session.history:
                metadata = message.get("metadata", {})

                if metadata.get("is_context_snapshot"):
                    context_snapshots.append(message)
                elif metadata.get("is_context_entry"):
                    context_entries.append(message)
                else:
                    regular_messages.append(message)

            # Keep most recent regular messages
            messages_to_keep = regular_messages[-keep_count:]
            cleanup_stats["regular_messages_kept"] = len(messages_to_keep)

            # Keep most recent context snapshots (if not removing)
            if not remove_old_snapshots:
                recent_snapshots = context_snapshots[-5:]  # Keep last 5 snapshots
                messages_to_keep.extend(recent_snapshots)
            else:
                cleanup_stats["context_snapshots_removed"] = len(context_snapshots)

            # Keep persistent context entries
            persistent_entries = [
                entry for entry in context_entries
                if entry.get("persistent", True)
            ]
            messages_to_keep.extend(persistent_entries)
            cleanup_stats["context_entries_removed"] = len(context_entries) - len(persistent_entries)

            # Sort by timestamp and update session
            messages_to_keep.sort(key=lambda x: x.get("timestamp", ""))
            session.history = messages_to_keep

            cleanup_stats.update({
                "final_message_count": len(session.history),
                "cleanup_performed": True,
                "messages_removed": cleanup_stats["original_message_count"] - len(session.history)
            })

            rprint(f"Session cleanup completed: {cleanup_stats['messages_removed']} messages removed")
            return cleanup_stats

        except Exception as e:
            eprint(f"Failed to cleanup session context: {e}")
            return {"error": str(e)}

    def get_session_storage_stats(self) -> dict[str, Any]:
        """Get comprehensive session storage statistics"""
        try:
            stats = {
                "context_manager_active": bool(self.context_manager),
                "total_sessions": 0,
                "session_details": {},
                "storage_summary": {
                    "total_messages": 0,
                    "context_snapshots": 0,
                    "context_entries": 0,
                    "regular_messages": 0
                }
            }

            if not self.context_manager:
                return stats

            stats["total_sessions"] = len(self.context_manager.session_managers)

            for session_id, session in self.context_manager.session_managers.items():
                session_stats = {
                    "session_type": "chatsession" if hasattr(session, 'history') else "fallback",
                    "message_count": 0,
                    "context_snapshots": 0,
                    "context_entries": 0,
                    "regular_messages": 0,
                    "storage_size_estimate": 0
                }

                if hasattr(session, 'history'):
                    session_stats["message_count"] = len(session.history)

                    for message in session.history:
                        content_size = len(str(message))
                        session_stats["storage_size_estimate"] += content_size

                        metadata = message.get("metadata", {})
                        if metadata.get("is_context_snapshot"):
                            session_stats["context_snapshots"] += 1
                        elif metadata.get("is_context_entry"):
                            session_stats["context_entries"] += 1
                        else:
                            session_stats["regular_messages"] += 1

                elif isinstance(session, dict) and 'history' in session:
                    session_stats["message_count"] = len(session['history'])
                    session_stats["regular_messages"] = len(session['history'])
                    session_stats["storage_size_estimate"] = sum(len(str(msg)) for msg in session['history'])

                stats["session_details"][session_id] = session_stats

                # Update totals
                stats["storage_summary"]["total_messages"] += session_stats["message_count"]
                stats["storage_summary"]["context_snapshots"] += session_stats["context_snapshots"]
                stats["storage_summary"]["context_entries"] += session_stats["context_entries"]
                stats["storage_summary"]["regular_messages"] += session_stats["regular_messages"]

            # Estimate total storage size
            stats["storage_summary"]["estimated_total_size_kb"] = sum(
                details["storage_size_estimate"] for details in stats["session_details"].values()
            ) / 1024

            return stats

        except Exception as e:
            eprint(f"Failed to get session storage stats: {e}")
            return {"error": str(e)}

    def list_available_checkpoints(self, max_age_hours: int = 168) -> list[dict[str, Any]]:  # Default 1 week
        """List all available checkpoints with metadata"""
        try:
            from toolboxv2 import get_app
            folder = str(get_app().data_dir) + '/Agents/checkpoint/' + self.amd.name

            if not os.path.exists(folder):
                return []

            checkpoints = []
            for file in os.listdir(folder):
                if file.endswith('.pkl') and file.startswith('agent_checkpoint_'):
                    filepath = os.path.join(folder, file)
                    try:
                        # Get file info
                        file_stat = os.stat(filepath)
                        file_size = file_stat.st_size
                        modified_time = datetime.fromtimestamp(file_stat.st_mtime)

                        # Extract timestamp from filename
                        timestamp_str = file.replace('agent_checkpoint_', '').replace('.pkl', '')
                        if timestamp_str == 'final_checkpoint':
                            checkpoint_time = modified_time
                            checkpoint_type = "final"
                        else:
                            checkpoint_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                            checkpoint_type = "regular"

                        # Check age
                        age_hours = (datetime.now() - checkpoint_time).total_seconds() / 3600
                        if age_hours <= max_age_hours:

                            # Try to load checkpoint metadata without full loading
                            metadata = {}
                            try:
                                with open(filepath, 'rb') as f:
                                    checkpoint = pickle.load(f)
                                metadata = {
                                    "tasks_count": len(checkpoint.task_state) if checkpoint.task_state else 0,
                                    "world_model_entries": len(checkpoint.world_model) if checkpoint.world_model else 0,
                                    "session_id": checkpoint.metadata.get("session_id", "unknown") if hasattr(
                                        checkpoint, 'metadata') and checkpoint.metadata else "unknown",
                                    "last_query": checkpoint.metadata.get("last_query", "unknown")[:100] if hasattr(
                                        checkpoint, 'metadata') and checkpoint.metadata else "unknown"
                                }
                            except:
                                metadata = {"load_error": True}

                            checkpoints.append({
                                "filepath": filepath,
                                "filename": file,
                                "checkpoint_type": checkpoint_type,
                                "timestamp": checkpoint_time.isoformat(),
                                "age_hours": round(age_hours, 1),
                                "file_size_kb": round(file_size / 1024, 1),
                                "metadata": metadata
                            })

                    except Exception as e:
                        import traceback
                        print(traceback.format_exc())
                        wprint(f"Could not analyze checkpoint file {file}: {e}")
                        continue

            # Sort by timestamp (newest first)
            checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)

            return checkpoints

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            eprint(f"Failed to list checkpoints: {e}")
            return []

    async def delete_old_checkpoints(self, keep_count: int = 5, max_age_hours: int = 168) -> dict[str, Any]:
        """Delete old checkpoints, keeping the most recent ones"""
        try:
            checkpoints = self.list_available_checkpoints(
                max_age_hours=max_age_hours * 2)  # Look further back for deletion

            deleted_count = 0
            deleted_size_kb = 0
            errors = []

            if len(checkpoints) > keep_count:
                # Keep the newest, delete the rest (except final checkpoint)
                to_delete = checkpoints[keep_count:]

                for checkpoint in to_delete:
                    if checkpoint["checkpoint_type"] != "final":  # Never delete final checkpoint
                        try:
                            os.remove(checkpoint["filepath"])
                            deleted_count += 1
                            deleted_size_kb += checkpoint["file_size_kb"]
                            rprint(f"Deleted old checkpoint: {checkpoint['filename']}")
                        except Exception as e:
                            import traceback
                            print(traceback.format_exc())
                            errors.append(f"Failed to delete {checkpoint['filename']}: {e}")

            # Also delete checkpoints older than max_age_hours
            old_checkpoints = [cp for cp in checkpoints if
                               cp["age_hours"] > max_age_hours and cp["checkpoint_type"] != "final"]
            for checkpoint in old_checkpoints:
                if checkpoint not in checkpoints[keep_count:]:  # Don't double-delete
                    try:
                        os.remove(checkpoint["filepath"])
                        deleted_count += 1
                        deleted_size_kb += checkpoint["file_size_kb"]
                        rprint(f"Deleted aged checkpoint: {checkpoint['filename']}")
                    except Exception as e:
                        import traceback
                        print(traceback.format_exc())
                        errors.append(f"Failed to delete {checkpoint['filename']}: {e}")

            return {
                "success": True,
                "deleted_count": deleted_count,
                "freed_space_kb": round(deleted_size_kb, 1),
                "remaining_checkpoints": len(checkpoints) - deleted_count,
                "errors": errors
            }

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            eprint(f"Failed to delete old checkpoints: {e}")
            return {
                "success": False,
                "error": str(e),
                "deleted_count": 0
            }

    # ===== TOOL AND NODE MANAGEMENT =====
    def _get_tool_analysis_path(self) -> str:
        """Get path for tool analysis cache"""
        from toolboxv2 import get_app
        folder = str(get_app().data_dir) + '/Agents/capabilities/' + self.amd.name
        os.makedirs(folder, exist_ok=True)
        return folder + '/tool_capabilities.json'

    def _get_context_path(self, session_id=None) -> str:
        """Get path for tool analysis cache"""
        from toolboxv2 import get_app
        folder = str(get_app().data_dir) + '/Agents/context/' + self.amd.name
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_suffix = f"_session_{session_id}" if session_id else ""
        filepath = f"agent_context_{self.amd.name}_{timestamp}{session_suffix}.json"
        return folder + f'/{filepath}'

    def add_first_class_tool(self, tool_func: Callable, name: str, description: str):
        """
        Add a first-class meta-tool that can be used by the LLMReasonerNode.
        These are different from regular tools - they control agent sub-systems.

        Args:
            tool_func: The function to register as a meta-tool
            name: Name of the meta-tool
            description: Description of when and how to use it
        """

        # Validate the tool function
        if not callable(tool_func):
            raise ValueError("Tool function must be callable")

        # Register in the reasoner's meta-tool registry (if reasoner exists)
        if hasattr(self.task_flow, 'llm_reasoner'):
            if not hasattr(self.task_flow.llm_reasoner, 'meta_tools_registry'):
                self.task_flow.llm_reasoner.meta_tools_registry = {}

            self.task_flow.llm_reasoner.meta_tools_registry[name] = {
                "function": tool_func,
                "description": description,
                "added_at": datetime.now().isoformat()
            }

            rprint(f"First-class meta-tool added: {name}")
        else:
            wprint("LLMReasonerNode not available for first-class tool registration")

    async def add_tool(self, tool_func: Callable, name: str = None, description: str = None, is_new=False):
        """Enhanced tool addition with intelligent analysis"""
        if not asyncio.iscoroutinefunction(tool_func):
            @wraps(tool_func)
            async def async_wrapper(*args, **kwargs):
                return await asyncio.to_thread(tool_func, *args, **kwargs)

            effective_func = async_wrapper
        else:
            effective_func = tool_func

        tool_name = name or effective_func.__name__
        tool_description = description or effective_func.__doc__ or "No description"

        # Store in registry
        self._tool_registry[tool_name] = {
            "function": effective_func,
            "description": tool_description,
            "args_schema": get_args_schema(tool_func)
        }

        # Add to available tools list
        if tool_name not in self.shared["available_tools"]:
            self.shared["available_tools"].append(tool_name)

        # Intelligent tool analysis
        if is_new:
            await self._analyze_tool_capabilities(tool_name, tool_description)

        rprint(f"Tool added with analysis: {tool_name}")

    async def _analyze_tool_capabilities(self, tool_name: str, description: str, tool_args:str):
        """Analyze tool capabilities with LLM for smart usage"""

        # Try to load existing analysis
        existing_analysis = self._load_tool_analysis()

        if tool_name in existing_analysis:
            try:
                # Validate cached data against the Pydantic model
                ToolAnalysis.model_validate(existing_analysis[tool_name])
                self._tool_capabilities[tool_name] = existing_analysis[tool_name]
                rprint(f"Loaded and validated cached analysis for {tool_name}")
            except ValidationError as e:
                wprint(f"Cached data for {tool_name} is invalid and will be regenerated: {e}")
                del self._tool_capabilities[tool_name]

        if not LITELLM_AVAILABLE:
            # Fallback analysis
            self._tool_capabilities[tool_name] = {
                "use_cases": [description],
                "triggers": [tool_name.lower().replace('_', ' ')],
                "complexity": "unknown",
                "confidence": 0.3
            }
            return

        # LLM-based intelligent analysis
        prompt = f"""
Analyze this tool and identify ALL possible use cases, triggers, and connections:

Tool Name: {tool_name}
args: {tool_args}
Description: {description}


Provide a comprehensive analysis covering:

1. OBVIOUS use cases (direct functionality)
2. INDIRECT connections (when this tool might be relevant)
3. TRIGGER PHRASES (what user queries would benefit from this tool)
4. COMPLEX scenarios (non-obvious applications)
5. CONTEXTUAL usage (when combined with other information)

Example for a "get_user_name" tool:
- Obvious: When user asks "what is my name"
- Indirect: Personalization, greetings, user identification
- Triggers: "my name", "who am I", "hello", "introduce yourself", "personalize"
- Complex: User context in multi-step tasks, addressing user directly
- Contextual: Any response that could be personalized

Rule! no additional comments or text in the format !
schema:
 {yaml.dump(ToolAnalysis.model_json_schema())}

Respond in YAML format:
Example:
```yaml
primary_function: "Retrieves the current user's name."
use_cases:
  - "Responding to 'what is my name?'"
  - "Personalizing greeting messages."
trigger_phrases:
  - "my name"
  - "who am I"
  - "introduce yourself"
indirect_connections:
  - "User identification in multi-factor authentication."
  - "Tagging user-generated content."
complexity_scenarios:
  - "In a multi-step task, remembering the user's name to personalize the final output."
user_intent_categories:
  - "Personalization"
  - "User Identification"
confidence_triggers:
  "my name": 0.95
  "who am I": 0.9
tool_complexity: low/medium/high
```
"""
        model = os.getenv("BASEMODEL", self.amd.fast_llm_model)
        for i in range(3):
            try:
                response = await self.a_run_llm_completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    with_context=False,
                    temperature=0.3,
                    max_tokens=1000,
                    task_id=f"tool_analysis_{tool_name}"
                )

                content = response.strip()

                # Extract JSON
                if "```yaml" in content:
                    yaml_str = content.split("```yaml")[1].split("```")[0].strip()
                else:
                    yaml_str = content

                analysis = yaml.safe_load(yaml_str)

                # Store analysis
                self._tool_capabilities[tool_name] = analysis

                # Save to cache
                await self._save_tool_analysis()

                validated_analysis = ToolAnalysis.model_validate(analysis)
                rprint(f"Generated intelligent analysis for {tool_name}")
                break

            except Exception as e:
                import traceback
                print(traceback.format_exc())
                model = self.amd.complex_llm_model if i > 1 else self.amd.fast_llm_model
                eprint(f"Tool analysis failed for {tool_name}: {e}")
                # Fallback
                self._tool_capabilities[tool_name] = {
                    "primary_function": description,
                    "use_cases": [description],
                    "trigger_phrases": [tool_name.lower().replace('_', ' ')],
                    "tool_complexity": "medium"
                }

    def _load_tool_analysis(self) -> dict[str, Any]:
        """Load tool analysis from cache"""
        try:
            if os.path.exists(self.tool_analysis_file):
                with open(self.tool_analysis_file) as f:
                    return json.load(f)
        except Exception as e:
            wprint(f"Could not load tool analysis: {e}")
        return {}


    async def save_context_to_file(self, session_id: str = None) -> bool:
        """Save current context to file"""
        try:
            context = await self.get_context(session_id=session_id, format_for_llm=False)

            filepath = self._get_context_path(session_id)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(context, f, indent=2, ensure_ascii=False, default=str)

            rprint(f"Context saved to: {filepath}")
            return True

        except Exception as e:
            eprint(f"Failed to save context: {e}")
            return False

    async def _save_tool_analysis(self):
        """Save tool analysis to cache"""
        try:
            with open(self.tool_analysis_file, 'w') as f:
                json.dump(self._tool_capabilities, f, indent=2)
        except Exception as e:
            eprint(f"Could not save tool analysis: {e}")

    def add_custom_flow(self, flow: AsyncFlow, name: str):
        """Add a custom flow for dynamic execution"""
        self.add_tool(flow.run_async, name=name, description=f"Custom flow: {flow.__class__.__name__}")
        rprint(f"Custom node added: {name}")

    def get_tool_by_name(self, tool_name: str) -> Callable | None:
        """Get tool function by name"""
        return self._tool_registry.get(tool_name, {}).get("function")

    async def arun_function(self, function_name: str, *args, **kwargs) -> Any:
        """
        Asynchronously finds a function by its string name, executes it with
        the given arguments, and returns the result.
        """
        rprint(f"Attempting to run function: {function_name} with args: {args}, kwargs: {kwargs}")
        target_function = self.get_tool_by_name(function_name)

        start_time = time.perf_counter()
        if not target_function:
            raise ValueError(f"Function '{function_name}' not found in the {self.amd.name}'s registered tools.")

        try:
            if asyncio.iscoroutinefunction(target_function):
                result = await target_function(*args, **kwargs)
            else:
                # If the function is not async, run it in a thread pool
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: target_function(*args, **kwargs))

            if asyncio.iscoroutine(result):
                result = await result

            if self.progress_tracker:
                await self.progress_tracker.emit_event(ProgressEvent(
                    event_type="tool_call",  # Vereinheitlicht zu tool_call
                    node_name="FlowAgent",
                    status=NodeStatus.COMPLETED,
                    success=True,
                    duration=time.perf_counter() - start_time,
                    tool_name=function_name,
                    tool_args=kwargs,
                    tool_result=result,
                    is_meta_tool=False,  # Klarstellen, dass es kein Meta-Tool ist
                    metadata={
                        "result_type": type(result).__name__,
                        "result_length": len(str(result))
                    }
                ))
            rprint(f"Function {function_name} completed successfully with result: {result}")
            return result

        except Exception as e:
            eprint(f"Function {function_name} execution failed: {e}")
            raise

    # ===== FORMATTING =====

    async def a_format_class(self,
                             pydantic_model: type[BaseModel],
                             prompt: str,
                             message_context: list[dict] = None,
                             max_retries: int = 2, auto_context=True, session_id: str = None, **kwargs) -> dict[str, Any]:
        """
        State-of-the-art LLM-based structured data formatting using Pydantic models.

        Args:
            pydantic_model: The Pydantic model class to structure the response
            prompt: The main prompt for the LLM
            message_context: Optional conversation context messages
            max_retries: Maximum number of retry attempts

        Returns:
            dict: Validated structured data matching the Pydantic model

        Raises:
            ValidationError: If the LLM response cannot be validated against the model
            RuntimeError: If all retry attempts fail
        """

        if not LITELLM_AVAILABLE:
            raise RuntimeError("LiteLLM is required for structured formatting but not available")

        if session_id and self.active_session != session_id:
            self.active_session = session_id
        # Generate schema documentation
        schema = pydantic_model.model_json_schema()
        model_name = pydantic_model.__name__

        # Create enhanced prompt with schema
        enhanced_prompt = f"""
    {prompt}

    CRITICAL FORMATTING REQUIREMENTS:
    1. Respond ONLY in valid YAML format
    2. Follow the exact schema structure provided
    3. Use appropriate data types (strings, lists, numbers, booleans)
    4. Include ALL required fields
    5. No additional comments, explanations, or text outside the YAML

    SCHEMA FOR {model_name}:
    {yaml.dump(schema, default_flow_style=False, indent=2)}

    EXAMPLE OUTPUT FORMAT:
    ```yaml
    # Your response here following the schema exactly
    field_name: "value"
    list_field:
      - "item1"
      - "item2"
    boolean_field: true
    number_field: 42
Respond in YAML format only:
"""
        # Prepare messages
        messages = []
        if message_context:
            messages.extend(message_context)
        messages.append({"role": "user", "content": enhanced_prompt})

        # Retry logic with progressive adjustments
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Adjust parameters based on attempt
                temperature = 0.1 + (attempt * 0.1)  # Increase temperature slightly on retries
                max_tokens = min(2000 + (attempt * 500), 4000)  # Increase token limit on retries

                rprint(f"[{model_name}] Attempt {attempt + 1}/{max_retries + 1} (temp: {temperature})")

                # Generate LLM response
                response = await self.a_run_llm_completion(
                    model=self.amd.complex_llm_model,
                    messages=messages,
                    with_context=auto_context,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    task_id=f"format_{model_name.lower()}_{attempt}"
                )

                if not response or not response.strip():
                    raise ValueError("Empty response from LLM")

                # Extract YAML content with multiple fallback strategies
                yaml_content = self._extract_yaml_content(response)

                if not yaml_content:
                    raise ValueError("No valid YAML content found in response")

                # Parse YAML
                try:
                    parsed_data = yaml.safe_load(yaml_content)
                except yaml.YAMLError as e:
                    raise ValueError(f"Invalid YAML syntax: {e}")

                if not isinstance(parsed_data, dict):
                    raise ValueError(f"Expected dict, got {type(parsed_data)}")

                # Validate against Pydantic model
                try:
                    validated_instance = pydantic_model.model_validate(parsed_data)
                    validated_data = validated_instance.model_dump()

                    rprint(f"✅ Successfully formatted {model_name} on attempt {attempt + 1}")
                    return validated_data

                except ValidationError as e:
                    detailed_errors = []
                    for error in e.errors():
                        field_path = " -> ".join(str(x) for x in error['loc'])
                        detailed_errors.append(f"Field '{field_path}': {error['msg']}")

                    error_msg = "Validation failed:\n" + "\n".join(detailed_errors)
                    raise ValueError(error_msg)

            except Exception as e:
                last_error = e
                wprint(f"[{model_name}] Attempt {attempt + 1} failed: {str(e)}")

                if attempt < max_retries:
                    # Add error feedback for next attempt
                    error_feedback = f"\n\nPREVIOUS ATTEMPT FAILED: {str(e)}\nPlease correct the issues and provide valid YAML matching the schema exactly."
                    messages[-1]["content"] = enhanced_prompt + error_feedback

                    # Brief delay before retry
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    eprint(f"[{model_name}] All {max_retries + 1} attempts failed")

        # All attempts failed
        raise RuntimeError(f"Failed to format {model_name} after {max_retries + 1} attempts. Last error: {last_error}")

    def _extract_yaml_content(self, response: str) -> str:
        """Extract YAML content from LLM response with multiple strategies"""
        # Strategy 1: Extract from code blocks
        if "```yaml" in response:
            try:
                yaml_content = response.split("```yaml")[1].split("```")[0].strip()
                if yaml_content:
                    return yaml_content
            except IndexError:
                pass

        # Strategy 2: Extract from generic code blocks
        if "```" in response:
            try:
                parts = response.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 1:  # Odd indices are inside code blocks
                        # Skip if it starts with a language identifier
                        lines = part.strip().split('\n')
                        if lines and not lines[0].strip().isalpha():
                            return part.strip()
                        elif len(lines) > 1:
                            # Try without first line
                            return '\n'.join(lines[1:]).strip()
            except:
                pass

        # Strategy 3: Look for YAML-like patterns
        lines = response.split('\n')
        yaml_lines = []
        in_yaml = False

        for line in lines:
            stripped = line.strip()

            # Detect start of YAML-like content
            if ':' in stripped and not stripped.startswith('#'):
                in_yaml = True
                yaml_lines.append(line)
            elif in_yaml:
                if stripped == '' or stripped.startswith(' ') or stripped.startswith('-') or ':' in stripped:
                    yaml_lines.append(line)
                else:
                    # Potential end of YAML
                    break

        if yaml_lines:
            return '\n'.join(yaml_lines).strip()

        # Strategy 4: Return entire response if it looks like YAML
        if ':' in response and not response.strip().startswith('<'):
            return response.strip()

        return ""
    # ===== SERVER SETUP =====

    def setup_a2a_server(self, host: str = "0.0.0.0", port: int = 5000, **kwargs):
        """Setup A2A server for bidirectional communication"""
        if not A2A_AVAILABLE:
            wprint("A2A not available, cannot setup server")
            return

        try:
            self.a2a_server = A2AServer(
                host=host,
                port=port,
                agent_card=AgentCard(
                    name=self.amd.name,
                    description="Production-ready PocketFlow agent",
                    version="1.0.0"
                ),
                **kwargs
            )

            # Register agent methods
            @self.a2a_server.route("/run")
            async def handle_run(request_data):
                query = request_data.get("query", "")
                session_id = request_data.get("session_id", "a2a_session")

                response = await self.a_run(query, session_id=session_id)
                return {"response": response}

            rprint(f"A2A server setup on {host}:{port}")

        except Exception as e:
            eprint(f"Failed to setup A2A server: {e}")

    def setup_mcp_server(self, host: str = "0.0.0.0", port: int = 8000, name: str = None, **kwargs):
        """Setup MCP server"""
        if not MCP_AVAILABLE:
            wprint("MCP not available, cannot setup server")
            return

        try:
            server_name = name or f"{self.amd.name}_MCP"
            self.mcp_server = FastMCP(server_name)

            # Register agent as MCP tool
            @self.mcp_server.tool()
            async def agent_run(query: str, session_id: str = "mcp_session") -> str:
                """Execute agent with given query"""
                return await self.a_run(query, session_id=session_id)

            rprint(f"MCP server setup: {server_name}")

        except Exception as e:
            eprint(f"Failed to setup MCP server: {e}")

    # ===== LIFECYCLE MANAGEMENT =====

    async def start_servers(self):
        """Start all configured servers"""
        tasks = []

        if self.a2a_server:
            tasks.append(asyncio.create_task(self.a2a_server.start()))

        if self.mcp_server:
            tasks.append(asyncio.create_task(self.mcp_server.run()))

        if tasks:
            rprint(f"Starting {len(tasks)} servers...")
            await asyncio.gather(*tasks, return_exceptions=True)

    def clear_context(self, session_id: str = None) -> bool:
        """Clear context über UnifiedContextManager mit Session-spezifischer Unterstützung"""
        try:
            #Clear über Context Manager
            if session_id:
                # Clear specific session
                if session_id in self.context_manager.session_managers:
                    session = self.context_manager.session_managers[session_id]
                    if hasattr(session, 'history'):
                        session.history = []
                    elif isinstance(session, dict) and 'history' in session:
                        session['history'] = []

                    # Remove from session managers
                    del self.context_manager.session_managers[session_id]

                    # Clear variable manager scope for this session
                    if self.variable_manager:
                        scope_name = f'session_{session_id}'
                        if scope_name in self.variable_manager.scopes:
                            del self.variable_manager.scopes[scope_name]

                    rprint(f"Context cleared for session: {session_id}")
            else:
                # Clear all sessions
                for session_id, session in self.context_manager.session_managers.items():
                    if hasattr(session, 'history'):
                        session.history = []
                    elif isinstance(session, dict) and 'history' in session:
                        session['history'] = []

                self.context_manager.session_managers = {}
                rprint("Context cleared for all sessions")

            # Clear context cache
            self.context_manager._invalidate_cache(session_id)

            # Clear current execution context in shared
            context_keys_to_clear = [
                "current_query", "current_response", "current_plan", "tasks",
                "results", "task_plans", "session_data", "formatted_context",
                "synthesized_response", "quality_assessment", "plan_adaptations",
                "executor_performance", "llm_tool_conversation", "aggregated_context"
            ]

            for key in context_keys_to_clear:
                if key in self.shared:
                    if isinstance(self.shared[key], dict):
                        self.shared[key] = {}
                    elif isinstance(self.shared[key], list):
                        self.shared[key] = []
                    else:
                        self.shared[key] = None

            # Clear variable manager scopes (except core system variables)
            if hasattr(self, 'variable_manager'):
                # Clear user, results, tasks scopes
                self.variable_manager.register_scope('user', {})
                self.variable_manager.register_scope('results', {})
                self.variable_manager.register_scope('tasks', {})
                # Reset cache
                self.variable_manager._cache.clear()

            # Reset execution state
            self.is_running = False
            self.is_paused = False
            self.shared["system_status"] = "idle"

            # Clear progress tracking
            if hasattr(self, 'progress_tracker'):
                self.progress_tracker.reset_session_metrics()

            return True

        except Exception as e:
            eprint(f"Failed to clear context: {e}")
            return False

    async def clean_memory(self, deep_clean: bool = False) -> bool:
        """Clean memory and context of the agent"""
        try:
            # Clear current context first
            self.clear_context()

            # Clean world model
            self.shared["world_model"] = {}
            self.world_model = {}

            # Clean performance metrics
            self.shared["performance_metrics"] = {}

            # Deep clean session storage
            session_managers = self.shared.get("session_managers", {})
            if session_managers:
                for _manager_name, manager in session_managers.items():
                    if hasattr(manager, 'clear_all_history'):
                        await manager.clear_all_history()
                    elif hasattr(manager, 'clear_history'):
                        manager.clear_history()

            # Clear session managers entirely
            self.shared["session_managers"] = {}
            self.shared["session_initialized"] = False

            # Clean variable manager completely
            if hasattr(self, 'variable_manager'):
                # Reinitialize with clean state
                self.variable_manager = VariableManager({}, self.shared)
                self._setup_variable_scopes()

            # Clean tool analysis cache if deep clean
            if deep_clean:
                self._tool_capabilities = {}
                self._tool_analysis_cache = {}

                # Remove tool analysis file
                if hasattr(self, 'tool_analysis_file') and os.path.exists(self.tool_analysis_file):
                    try:
                        os.remove(self.tool_analysis_file)
                        rprint("Removed tool analysis cache file")
                    except:
                        pass

            # Clean checkpoint data
            self.checkpoint_data = {}
            self.last_checkpoint = None

            # Clean execution history
            if hasattr(self.task_flow, 'executor_node'):
                self.task_flow.executor_node.execution_history = []
                self.task_flow.executor_node.results_store = {}

            # Clean context manager sessions
            if hasattr(self.task_flow, 'context_manager'):
                self.task_flow.context_manager.session_managers = {}

            # Clean LLM call statistics
            self.shared.pop("llm_call_stats", None)

            # Force garbage collection
            import gc
            gc.collect()

            rprint(f"Memory cleaned (deep_clean: {deep_clean})")
            return True

        except Exception as e:
            eprint(f"Failed to clean memory: {e}")
            return False

    async def close(self):
        """Clean shutdown"""
        self.is_running = False
        self._shutdown_event.set()

        # Create final checkpoint
        if self.enable_pause_resume:
            checkpoint = await self._create_checkpoint()
            await self._save_checkpoint(checkpoint, "final_checkpoint.pkl")

        # Shutdown executor
        self.executor.shutdown(wait=True)

        # Close servers
        if self.a2a_server:
            await self.a2a_server.close()

        if self.mcp_server:
            await self.mcp_server.close()

        if hasattr(self, '_mcp_session_manager'):
            await self._mcp_session_manager.cleanup_all()

        rprint("Agent shutdown complete")

    @property
    def total_cost(self) -> float:
        """Get total cost if budget manager available"""
        if hasattr(self.amd, 'budget_manager') and self.amd.budget_manager:
            return getattr(self.amd.budget_manager, 'total_cost', 0.0)
        return 0.0

    def status(self, pretty_print: bool = False) -> dict[str, Any] | str:
        """Get comprehensive agent status with optional pretty printing"""

        # Core status information
        base_status = {
            "agent_info": {
                "name": self.amd.name,
                "version": "2.0",
                "type": "FlowAgent"
            },
            "runtime_status": {
                "status": self.shared.get("system_status", "idle"),
                "is_running": self.is_running,
                "is_paused": self.is_paused,
                "uptime_seconds": (datetime.now() - getattr(self, '_start_time', datetime.now())).total_seconds()
            },
            "task_execution": {
                "total_tasks": len(self.shared.get("tasks", {})),
                "active_tasks": len([t for t in self.shared.get("tasks", {}).values() if t.status == "running"]),
                "completed_tasks": len([t for t in self.shared.get("tasks", {}).values() if t.status == "completed"]),
                "failed_tasks": len([t for t in self.shared.get("tasks", {}).values() if t.status == "failed"]),
                "plan_adaptations": self.shared.get("plan_adaptations", 0)
            },
            "conversation": {
                "turns": len(self.shared.get("conversation_history", [])),
                "session_id": self.shared.get("session_id", self.active_session),
                "current_user": self.shared.get("user_id"),
                "last_query": self.shared.get("current_query", "")[:100] + "..." if len(
                    self.shared.get("current_query", "")) > 100 else self.shared.get("current_query", "")
            },
            "capabilities": {
                "available_tools": len(self.shared.get("available_tools", [])),
                "tool_names": list(self.shared.get("available_tools", [])),
                "analyzed_tools": len(self._tool_capabilities),
                "world_model_size": len(self.shared.get("world_model", {})),
                "intelligence_level": "high" if self._tool_capabilities else "basic"
            },
            "memory_context": {
                "session_initialized": self.shared.get("session_initialized", False),
                "session_managers": len(self.shared.get("session_managers", {})),
                "context_system": "advanced_session_aware" if self.shared.get("session_initialized") else "basic",
                "variable_scopes": len(self.variable_manager.get_scope_info()) if hasattr(self,
                                                                                          'variable_manager') else 0
            },
            "performance": {
                "total_cost": self.total_cost,
                "checkpoint_enabled": self.enable_pause_resume,
                "last_checkpoint": self.last_checkpoint.isoformat() if self.last_checkpoint else None,
                "max_parallel_tasks": self.max_parallel_tasks
            },
            "servers": {
                "a2a_server": self.a2a_server is not None,
                "mcp_server": self.mcp_server is not None,
                "server_count": sum([self.a2a_server is not None, self.mcp_server is not None])
            },
            "configuration": {
                "fast_llm_model": self.amd.fast_llm_model,
                "complex_llm_model": self.amd.complex_llm_model,
                "use_fast_response": getattr(self.amd, 'use_fast_response', False),
                "max_input_tokens": getattr(self.amd, 'max_input_tokens', 8000),
                "persona_configured": self.amd.persona is not None,
                "format_config": bool(getattr(self.amd.persona, 'format_config', None)) if self.amd.persona else False
            }
        }

        # Add detailed execution summary if tasks exist
        tasks = self.shared.get("tasks", {})
        if tasks:
            task_types_used = {}
            tools_used = []
            execution_timeline = []

            for task_id, task in tasks.items():
                # Count task types
                task_type = getattr(task, 'type', 'unknown')
                task_types_used[task_type] = task_types_used.get(task_type, 0) + 1

                # Collect tools used
                if hasattr(task, 'tool_name') and task.tool_name:
                    tools_used.append(task.tool_name)

                # Timeline info
                if hasattr(task, 'started_at') and task.started_at:
                    timeline_entry = {
                        "task_id": task_id,
                        "type": task_type,
                        "started": task.started_at.isoformat(),
                        "status": getattr(task, 'status', 'unknown')
                    }
                    if hasattr(task, 'completed_at') and task.completed_at:
                        timeline_entry["completed"] = task.completed_at.isoformat()
                        timeline_entry["duration"] = (task.completed_at - task.started_at).total_seconds()
                    execution_timeline.append(timeline_entry)

            base_status["task_execution"].update({
                "task_types_used": task_types_used,
                "tools_used": list(set(tools_used)),
                "execution_timeline": execution_timeline[-5:]  # Last 5 tasks
            })

        # Add context statistics
        if hasattr(self.task_flow, 'context_manager'):
            context_manager = self.task_flow.context_manager
            base_status["memory_context"].update({
                "compression_threshold": context_manager.compression_threshold,
                "max_tokens": context_manager.max_tokens,
                "active_context_sessions": len(getattr(context_manager, 'session_managers', {}))
            })

        # Add variable system info
        if hasattr(self, 'variable_manager'):
            available_vars = self.variable_manager.get_available_variables()
            scope_info = self.variable_manager.get_scope_info()

            base_status["variable_system"] = {
                "total_scopes": len(scope_info),
                "scope_names": list(scope_info.keys()),
                "total_variables": sum(len(vars) for vars in available_vars.values()),
                "scope_details": {
                    scope: {"type": info["type"], "variables": len(available_vars.get(scope, {}))}
                    for scope, info in scope_info.items()
                }
            }

        # Add format quality info if available
        quality_assessment = self.shared.get("quality_assessment", {})
        if quality_assessment:
            quality_details = quality_assessment.get("quality_details", {})
            base_status["format_quality"] = {
                "overall_score": quality_details.get("total_score", 0.0),
                "format_adherence": quality_details.get("format_adherence", 0.0),
                "length_adherence": quality_details.get("length_adherence", 0.0),
                "content_quality": quality_details.get("base_quality", 0.0),
                "assessment": quality_assessment.get("quality_assessment", "unknown"),
                "has_suggestions": bool(quality_assessment.get("suggestions", []))
            }

        # Add LLM usage statistics
        llm_stats = self.shared.get("llm_call_stats", {})
        if llm_stats:
            base_status["llm_usage"] = {
                "total_calls": llm_stats.get("total_calls", 0),
                "context_compression_rate": llm_stats.get("context_compression_rate", 0.0),
                "average_context_tokens": llm_stats.get("context_tokens_used", 0) / max(llm_stats.get("total_calls", 1),
                                                                                        1),
                "total_tokens_used": llm_stats.get("total_tokens_used", 0)
            }

        # Add timestamp
        base_status["timestamp"] = datetime.now().isoformat()

        if not pretty_print:
            return base_status

        # Pretty print using EnhancedVerboseOutput
        try:
            from toolboxv2.mods.isaa.extras.verbose_output import EnhancedVerboseOutput
            verbose_output = EnhancedVerboseOutput(verbose=True)

            # Header
            verbose_output.log_header(f"Agent Status: {base_status['agent_info']['name']}")

            # Runtime Status
            status_color = {
                "running": "SUCCESS",
                "paused": "WARNING",
                "idle": "INFO",
                "error": "ERROR"
            }.get(base_status["runtime_status"]["status"], "INFO")

            getattr(verbose_output, f"print_{status_color.lower()}")(
                f"Status: {base_status['runtime_status']['status'].upper()}"
            )

            # Task Execution Summary
            task_exec = base_status["task_execution"]
            if task_exec["total_tasks"] > 0:
                verbose_output.formatter.print_section(
                    "Task Execution",
                    f"Total: {task_exec['total_tasks']} | "
                    f"Completed: {task_exec['completed_tasks']} | "
                    f"Failed: {task_exec['failed_tasks']} | "
                    f"Active: {task_exec['active_tasks']}\n"
                    f"Adaptations: {task_exec['plan_adaptations']}"
                )

                if task_exec.get("tools_used"):
                    verbose_output.formatter.print_section(
                        "Tools Used",
                        ", ".join(task_exec["tools_used"])
                    )

            # Capabilities
            caps = base_status["capabilities"]
            verbose_output.formatter.print_section(
                "Capabilities",
                f"Intelligence Level: {caps['intelligence_level']}\n"
                f"Available Tools: {caps['available_tools']}\n"
                f"Analyzed Tools: {caps['analyzed_tools']}\n"
                f"World Model Size: {caps['world_model_size']}"
            )

            # Memory & Context
            memory = base_status["memory_context"]
            verbose_output.formatter.print_section(
                "Memory & Context",
                f"Context System: {memory['context_system']}\n"
                f"Session Managers: {memory['session_managers']}\n"
                f"Variable Scopes: {memory['variable_scopes']}\n"
                f"Session Initialized: {memory['session_initialized']}"
            )

            # Configuration
            config = base_status["configuration"]
            verbose_output.formatter.print_section(
                "Configuration",
                f"Fast LLM: {config['fast_llm_model']}\n"
                f"Complex LLM: {config['complex_llm_model']}\n"
                f"Max Tokens: {config['max_input_tokens']}\n"
                f"Persona: {'Configured' if config['persona_configured'] else 'Default'}\n"
                f"Format Config: {'Active' if config['format_config'] else 'None'}"
            )

            # Performance
            perf = base_status["performance"]
            verbose_output.formatter.print_section(
                "Performance",
                f"Total Cost: ${perf['total_cost']:.4f}\n"
                f"Checkpointing: {'Enabled' if perf['checkpoint_enabled'] else 'Disabled'}\n"
                f"Max Parallel Tasks: {perf['max_parallel_tasks']}\n"
                f"Last Checkpoint: {perf['last_checkpoint'] or 'None'}"
            )

            # Variable System Details
            if "variable_system" in base_status:
                var_sys = base_status["variable_system"]
                scope_details = []
                for scope, details in var_sys["scope_details"].items():
                    scope_details.append(f"{scope}: {details['variables']} variables ({details['type']})")

                verbose_output.formatter.print_section(
                    "Variable System",
                    f"Total Scopes: {var_sys['total_scopes']}\n"
                    f"Total Variables: {var_sys['total_variables']}\n" +
                    "\n".join(scope_details)
                )

            # Format Quality
            if "format_quality" in base_status:
                quality = base_status["format_quality"]
                verbose_output.formatter.print_section(
                    "Format Quality",
                    f"Overall Score: {quality['overall_score']:.2f}\n"
                    f"Format Adherence: {quality['format_adherence']:.2f}\n"
                    f"Length Adherence: {quality['length_adherence']:.2f}\n"
                    f"Content Quality: {quality['content_quality']:.2f}\n"
                    f"Assessment: {quality['assessment']}"
                )

            # LLM Usage
            if "llm_usage" in base_status:
                llm = base_status["llm_usage"]
                verbose_output.formatter.print_section(
                    "LLM Usage Statistics",
                    f"Total Calls: {llm['total_calls']}\n"
                    f"Avg Context Tokens: {llm['average_context_tokens']:.1f}\n"
                    f"Total Tokens: {llm['total_tokens_used']}\n"
                    f"Compression Rate: {llm['context_compression_rate']:.2%}"
                )

            # Servers
            servers = base_status["servers"]
            if servers["server_count"] > 0:
                server_status = []
                if servers["a2a_server"]:
                    server_status.append("A2A Server: Active")
                if servers["mcp_server"]:
                    server_status.append("MCP Server: Active")

                verbose_output.formatter.print_section(
                    "Servers",
                    "\n".join(server_status)
                )

            verbose_output.print_separator()
            verbose_output.print_info(f"Status generated at: {base_status['timestamp']}")

            return "Status printed above"

        except Exception:
            # Fallback to JSON if pretty print fails
            import json
            return json.dumps(base_status, indent=2, default=str)

    @property
    def tool_registry(self):
        return self._tool_registry

    def __rshift__(self, other):
        return Chain(self) >> other

    def __add__(self, other):
        return Chain(self) + other

    def __and__(self, other):
        return Chain(self) & other

    def __mod__(self, other):
        """Implements % operator for conditional branching"""
        return ConditionalChain(self, other)


def get_progress_summary(self) -> dict[str, Any]:
    """Get comprehensive progress summary from the agent"""
    if hasattr(self, 'progress_tracker'):
        return self.progress_tracker.get_summary()
    return {"error": "No progress tracker available"}

import inspect
import typing
from collections.abc import Callable
from typing import Any


def get_args_schema(func: Callable) -> str:
    """
    Generate a string representation of a function's arguments and annotations.
    Keeps *args and **kwargs indicators and handles modern Python type hints.
    """
    sig = inspect.signature(func)
    parts = []

    for name, param in sig.parameters.items():
        ann = ""
        if param.annotation is not inspect._empty:
            ann = f": {_annotation_to_str(param.annotation)}"

        default = ""
        if param.default is not inspect._empty:
            default = f" = {repr(param.default)}"

        prefix = ""
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            prefix = "*"
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            prefix = "**"

        parts.append(f"{prefix}{name}{ann}{default}")

    return f"({', '.join(parts)})"

def _annotation_to_str(annotation: Any) -> str:
    """
    Convert any annotation to a nice string, including | union syntax (PEP 604),
    Optional[T], generics, and forward references.
    """
    if isinstance(annotation, str):
        return annotation  # Forward reference as-is

    # Handle typing.Optional and typing.Union
    if getattr(annotation, "__origin__", None) is typing.Union:
        args = annotation.__args__
        if len(args) == 2 and type(None) in args:
            non_none = args[0] if args[1] is type(None) else args[1]
            return f"Optional[{_annotation_to_str(non_none)}]"
        return " | ".join(_annotation_to_str(a) for a in args)

    # Handle built-in Union syntax (PEP 604)
    if hasattr(annotation, "__args__") and getattr(annotation, "__origin__", None) is None and "|" in str(annotation):
        return str(annotation)

    # Handle generics like list[int], dict[str, Any]
    if getattr(annotation, "__origin__", None):
        origin = getattr(annotation.__origin__, "__name__", str(annotation.__origin__))
        args = getattr(annotation, "__args__", None)
        if args:
            return f"{origin}[{', '.join(_annotation_to_str(a) for a in args)}]"
        return origin

    # Handle normal classes and built-ins
    if hasattr(annotation, "__name__"):
        return annotation.__name__

    return repr(annotation)


from typing import Any


def _extract_meta_tool_calls(text: str, prefix="META_TOOL_CALL:") -> list[tuple[str, str]]:
    """Extract META_TOOL_CALL with proper bracket balance handling"""
    import re

    matches = []
    pattern = r'META_TOOL_CALL:\s*(\w+)\(' if prefix == "META_TOOL_CALL:" else r'TOOL_CALL:\s*(\w+)\('

    for match in re.finditer(pattern, text):
        tool_name = match.group(1)
        start_pos = match.end() - 1  # Position of opening parenthesis

        # Find matching closing parenthesis with bracket balancing
        paren_count = 0
        pos = start_pos
        in_string = False
        string_char = None
        escape_next = False

        while pos < len(text):
            char = text[pos]

            if escape_next:
                escape_next = False
            elif char == '\\':
                escape_next = True
            elif not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    string_char = char
                elif char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        # Found matching closing parenthesis
                        args_str = text[start_pos + 1:pos]
                        matches.append((tool_name, args_str))
                        break
            else:  # in_string is True
                if char == string_char:
                    in_string = False
                    string_char = None

            pos += 1

    return matches


def _parse_tool_args(args_str: str) -> dict[str, Any]:
    """Parse tool arguments from string format with enhanced error handling"""
    import ast

    # Handle simple key=value format
    if '=' in args_str and not args_str.strip().startswith('{'):
        args = {}
        # Split by commas but handle nested structures
        parts = []
        current_part = ""
        bracket_count = 0
        in_quotes = False
        quote_char = None

        for char in args_str:
            if char in ['"', "'"] and not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
            elif char in ['[', '{'] and not in_quotes:
                bracket_count += 1
            elif char in [']', '}'] and not in_quotes:
                bracket_count -= 1
            elif char == ',' and bracket_count == 0 and not in_quotes:
                parts.append(current_part.strip())
                current_part = ""
                continue

            current_part += char

        if current_part.strip():
            parts.append(current_part.strip())

        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip().strip('"').strip("'")
                value = value.strip()

                # Try to evaluate the value
                try:
                    if value.startswith('[') or value.startswith('{'):
                        args[key] = ast.literal_eval(value)
                    elif value.lower() in ['true', 'false']:
                        args[key] = value.lower() == 'true'
                    elif value.replace('.', '').replace('-', '').isdigit():
                        args[key] = float(value) if '.' in value else int(value)
                    else:
                        # Remove quotes if present
                        args[key] = value.strip('"').strip("'")
                except:
                    args[key] = value.strip('"').strip("'")

        return auto_unescape(args)

    # Handle JSON-like format
    try:
        return auto_unescape(ast.literal_eval(f"{{{args_str}}}"))
    except:
        return auto_unescape({"raw_args": args_str})


def unescape_string(text: str) -> str:
    """Universal string unescaping for any programming language."""
    if not isinstance(text, str) or len(text) < 2:
        return text

    # Remove outer quotes if wrapped
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]

    # Universal escape sequences
    escapes = {
        '\\n': '\n', '\\t': '\t', '\\r': '\r',
        '\\"': '"', "\\'": "'", '\\\\': '\\'
    }

    for escaped, unescaped in escapes.items():
        text = text.replace(escaped, unescaped)

    return text


def needs_unescaping(text: str) -> bool:
    """Detect if string likely needs unescaping."""
    return bool(re.search(r'\\[ntr"\'\\]', text)) or len(text) > 50


def process_nested(data: Any, max_depth: int = 20) -> Any:
    """Recursively process nested structures, unescaping strings that need it."""
    if max_depth <= 0:
        return data

    if isinstance(data, dict):
        return {k: process_nested(v, max_depth - 1) for k, v in data.items()}

    elif isinstance(data, list | tuple):
        processed = [process_nested(item, max_depth - 1) for item in data]
        return type(data)(processed)

    elif isinstance(data, str) and needs_unescaping(data):
        return unescape_string(data)

    return data


def auto_unescape(args: Any) -> Any:
    """Automatically unescape all strings in nested data structure."""
    return process_nested(args)



# Add this method to FlowAgent class
FlowAgent.get_progress_summary = get_progress_summary
if __name__ == "__main__3":

    agent_text = '''
First, let me examine the results of the previous tasks:

```yaml
TOOL_CALLS:
  - tool: list_directory
    args:
      dirpath: ImageCreation
```

I see that we've created the ImageCreation directory successfully, but the image move operation failed because the source image `unique_complex_image.png` couldn't be found in the current location. Let me check the current directory to locate the image file:

```yaml
TOOL_CALLS:
  - tool: list_directory
    args:
      dirpath: .
```

Now I'll check if we can find information about fractal pattern generation to prepare for creating our script:

```yaml
TOOL_CALLS:
  - tool: searchWeb
    args:
      query: python fractal pattern generation leaves trees stars houses
      max_results: 5
```

Let me also check for any existing agents that might be helpful for this task:

```yaml
TOOL_CALLS:
  - tool: list_agents
    args:
      detailed: false
```

Based on the search results, I'll now create a specialized agent that can help with generating complex fractal patterns:

```yaml
TOOL_CALLS:
  - tool: create_specialized_agent
    args:
      agent_name: fractal_artist
      system_prompt: "You are a specialized AI artist focused on creating complex fractal patterns. You understand mathematical concepts behind fractals and can generate code to create visual representations of fractals that resemble natural objects like leaves, trees, stars, and architectural patterns like houses. You're skilled in Python with libraries like PIL, NumPy, and matplotlib for image generation."
      model: null
```

I'll also look for specific libraries that might be useful for generating fractal images:

```yaml
TOOL_CALLS:
  - tool: context7-mcp_resolve-library-id
    args:
      libraryName: PIL
```

Let's also check for NumPy which is commonly used in fractal generation:

```yaml
TOOL_CALLS:
  - tool: context7-mcp_resolve-library-id
    args:
      libraryName: NumPy
```

Now I'll create the Python script for generating fractal patterns. First, let me check if we have the required libraries' documentation:

```yaml
TOOL_CALLS:
  - tool: context7-mcp_get-library-docs
    args:
      context7CompatibleLibraryID: PIL
      topic: Image generation
```

Let's also get documentation for NumPy:

```yaml
TOOL_CALLS:
  - tool: context7-mcp_get-library-docs
    args:
      context7CompatibleLibraryID: numpy
      topic: mathematical operations
```

Now I'll create a Python script that can generate complex fractal patterns:

```yaml
TOOL_CALLS:
  - tool: createScript
    args:
      name: generate_fractal_patterns.py
      description: Generates complex fractal patterns resembling leaves, trees, stars, and houses
      content: |
        import numpy as np
        from PIL import Image, ImageDraw
        import math
        import random
        import os

        def create_fractal_leaf(width=800, height=800):
            """Generate a fractal leaf pattern"""
            img = Image.new('RGB', (width, height), (0, 0, 0))
            draw = ImageDraw.Draw(img)

            def draw_branch(x, y, length, angle, depth):
                if depth == 0:
                    return

                # Calculate end point
                end_x = x + length * math.cos(angle)
                end_y = y + length * math.sin(angle)

                # Draw the branch
                color = (0, min(255, depth * 40), 0)
                draw.line((x, y, end_x, end_y), fill=color, width=depth)

                # Recursive calls for sub-branches
                new_length = length * 0.7
                draw_branch(end_x, end_y, new_length, angle - 0.3, depth - 1)
                draw_branch(end_x, end_y, new_length, angle + 0.3, depth - 1)

                # Occasionally add a leaf
                if depth <= 2 and random.random() > 0.7:
                    leaf_size = random.randint(2, 5)
                    draw.ellipse((end_x - leaf_size, end_y - leaf_size,
                                end_x + leaf_size, end_y + leaf_size),
                               fill=(0, min(255, depth * 50), 0))

            # Start with a main trunk
            draw_branch(width/2, height, 100, -math.pi/2, 8)
            return img

        def create_fractal_tree(width=800, height=800):
            """Generate a fractal tree pattern"""
            img = Image.new('RGB', (width, height), (0, 0, 0))
            draw = ImageDraw.Draw(img)

            def draw_tree(x, y, length, angle, depth):
                if depth == 0:
                    return

                # Calculate end point
                end_x = x + length * math.cos(angle)
                end_y = y + length * math.sin(angle)

                # Draw the branch with brown color
                color = (min(139, depth * 20), min(69, depth * 10), min(19, depth * 3))
                draw.line((x, y, end_x, end_y), fill=color, width=max(1, depth//2))

                # Recursive calls for sub-branches
                if depth > 1:
                    # Add some randomness to angles
                    angle1 = angle - random.uniform(0.1, 0.5)
                    angle2 = angle + random.uniform(0.1, 0.5)
                    angle3 = angle + random.uniform(-0.1, 0.1)

                    new_length = length * random.uniform(0.6, 0.8)
                    draw_tree(end_x, end_y, new_length, angle1, depth - 1)
                    draw_tree(end_x, end_y, new_length, angle2, depth - 1)

                    # Occasionally add a third branch
                    if random.random() > 0.6:
                        draw_tree(end_x, end_y, new_length * 0.7, angle3, depth - 1)

            # Start with a main trunk
            draw_tree(width/2, height, 120, -math.pi/2, 10)
            return img

        def create_fractal_stars(width=800, height=800):
            """Generate a fractal star pattern"""
            img = Image.new('RGB', (width, height), (0, 0, 20))
            draw = ImageDraw.Draw(img)

            def draw_star(x, y, size, depth):
                if depth == 0:
                    return

                # Draw a star shape
                points = []
                for i in range(10):
                    angle = math.pi/5 * i
                    radius = size if i % 2 == 0 else size * 0.4
                    px = x + radius * math.cos(angle)
                    py = y + radius * math.sin(angle)
                    points.append((px, py))

                # Draw the star with gradient color
                brightness = min(255, 200 + depth * 15)
                draw.polygon(points, fill=(brightness, brightness, brightness))

                # Recursive calls for smaller stars
                if depth > 1:
                    for i in range(5):
                        angle = math.pi/2.5 * i
                        distance = size * 1.5
                        new_x = x + distance * math.cos(angle)
                        new_y = y + distance * math.sin(angle)
                        new_size = size * 0.4
                        if 0 <= new_x < width and 0 <= new_y < height:
                            draw_star(new_x, new_y, new_size, depth - 1)

            # Draw several stars with different properties
            for _ in range(20):
                x = random.randint(0, width)
                y = random.randint(0, height)
                size = random.randint(10, 30)
                depth = random.randint(2, 4)
                draw_star(x, y, size, depth)

            return img

        def create_fractal_house(width=800, height=800):
            """Generate a fractal house pattern"""
            img = Image.new('RGB', (width, height), (135, 206, 235))  # Sky blue background
            draw = ImageDraw.Draw(img)

            def draw_house(x, y, size, depth):
                if depth == 0:
                    return

                # Draw main house structure
                house_color = (random.randint(180, 220), random.randint(60, 100), random.randint(40, 80))
                draw.rectangle([x, y, x+size, y+size], fill=house_color)

                # Draw roof
                roof_color = (random.randint(100, 150), random.randint(20, 60), random.randint(10, 40))
                roof_points = [(x-5, y), (x+size//2, y-size//3), (x+size+5, y)]
                draw.polygon(roof_points, fill=roof_color)

                # Draw door
                door_color = (101, 67, 33)
                draw.rectangle([x+size//3, y+size//2, x+2*size//3, y+size], fill=door_color)

                # Draw windows
                window_color = (173, 216, 230)
                draw.rectangle([x+size//6, y+size//4, x+size//3, y+size//2], fill=window_color)
                draw.rectangle([x+2*size//3, y+size//4, x+5*size//6, y+size//2], fill=window_color)

                # Recursive calls for smaller houses
                if depth > 1:
                    # Draw a smaller house next to the current one
                    new_size = size // 2
                    new_x = x + size + 20
                    new_y = y + size - new_size
                    if new_x + new_size < width and new_y + new_size < height:
                        draw_house(new_x, new_y, new_size, depth - 1)

                    # Draw a smaller house above the current one
                    new_size2 = size // 3
                    new_x2 = x + size//2 - new_size2//2
                    new_y2 = y - new_size2 - 10
                    if 0 <= new_x2 and new_x2 + new_size2 < width and 0 <= new_y2:
                        draw_house(new_x2, new_y2, new_size2, depth - 1)

            # Draw main house
            draw_house(width//4, height//2, 200, 3)

            # Draw some bushes
            for i in range(5):
                bush_x = random.randint(0, width)
                bush_y = height - 50
                bush_size = random.randint(20, 40)
                bush_color = (random.randint(0, 50), random.randint(100, 150), random.randint(0, 50))
                draw.ellipse([bush_x, bush_y, bush_x + bush_size, bush_y + bush_size//2], fill=bush_color)

            return img

        def create_fractal_mosaic(width=800, height=800):
            """Generate a fractal mosaic pattern"""
            img = Image.new('RGB', (width, height), (0, 0, 0))
            pixels = img.load()

            def mandelbrot(c, max_iter=100):
                z = 0
                for n in range(max_iter):
                    if abs(z) > 2:
                        return n
                    z = z*z + c
                return max_iter

            # Generate Mandelbrot set based pattern
            for x in range(width):
                for y in range(height):
                    # Convert pixel coordinate to complex number
                    c = complex(-2.0 + 3.0 * x / width, -1.5 + 3.0 * y / height)
                    # Compute the number of iterations
                    m = mandelbrot(c)
                    # Color based on iterations
                    hue = int(255 * m / 100)
                    pixels[x, y] = (hue, (hue * 2) % 256, (hue * 3) % 256)

            return img

        def main():
            """Generate all fractal patterns and save them"""
            # Create ImageCreation directory if it doesn't exist
            if not os.path.exists("ImageCreation"):
                os.makedirs("ImageCreation")

            # Generate and save all patterns
            patterns = [
                ("fractal_leaf.png", create_fractal_leaf),
                ("fractal_tree.png", create_fractal_tree),
                ("fractal_stars.png", create_fractal_stars),
                ("fractal_house.png", create_fractal_house),
                ("fractal_mosaic.png", create_fractal_mosaic)
            ]

            for filename, func in patterns:
                print(f"Generating {filename}...")
                img = func()
                img.save(f"ImageCreation/{filename}")
                print(f"Saved {filename}")

            print("All fractal patterns generated successfully!")

        if __name__ == "__main__":
            main()
      script_type: py
```

Now let's run the script to generate the fractal patterns:

```yaml
TOOL_CALLS:
  - tool: runScript
    args:
      name: generate_fractal_patterns.py
```

Let's check if the images were created successfully:

```yaml
TOOL_CALLS:
  - tool: list_directory
    args:
      dirpath: ImageCreation
```

Let me verify the content of the generated images by analyzing one of them:

```yaml
TOOL_CALLS:
  - tool: askAgent
    args:
      target_agent_name: fractal_artist
      instructions: "Analyze the fractal patterns generated in the ImageCreation folder. Specifically, examine the complexity of the generated images like fractal_tree.png and fractal_mosaic.png. Do they exhibit the complex, fractal-like patterns requested for leaves, trees, stars, houses, and mosaics? Please provide a detailed assessment."
```

Let me also create a summary report of what we've accomplished:

```yaml
TOOL_CALLS:
  - tool: write_file
    args:
      filepath: ImageCreation/GENERATION_REPORT.md
      content: |
        # Fractal Pattern Generation Report

        ## Overview
        This project successfully generated complex fractal patterns resembling natural and architectural forms including leaves, trees, stars, houses, and mosaics.

        ## Generated Patterns

        ### 1. Fractal Leaf (`fractal_leaf.png`)
        - Created using recursive branching algorithms
        - Features realistic vein structures
        - Color gradient from trunk to leaf tips

        ### 2. Fractal Tree (`fractal_tree.png`)
        - Multi-level branching structure
        - Random variations for natural appearance
        - Brown trunk with green foliage

        ### 3. Fractal Stars (`fractal_stars.png`)
        - Recursive star patterns
        - Depth-based brightness variations
        - Space-like background

        ### 4. Fractal House (`fractal_house.png`)
        - Recursive house structures
        - Roof and window details
        - Surrounding landscape elements

        ### 5. Fractal Mosaic (`fractal_mosaic.png`)
        - Mathematically-generated Mandelbrot set visualization
        - Colorful, intricate patterns
        - Infinite complexity at boundaries

        ## Technical Implementation
        - Language: Python
        - Libraries: PIL (Pillow), NumPy
        - Techniques: Recursive algorithms, mathematical transformations

        ## Complexity Assessment
        All generated images exhibit complex, fractal-like patterns with self-similarity at different scales. The recursive nature of the algorithms ensures that patterns repeat with variations, creating visually interesting and mathematically significant imagery.
```

Finally, let's provide a summary to the user about what we've accomplished:

```yaml
TOOL_CALLS:
  - tool: workspace_status
    args:
      include_git: false
```

I'll check the final status of our tasks:

```yaml
TOOL_CALLS:
  - tool: get_background_tasks_status
    args:
      show_completed: true
```'''
    print(LLMToolNode._extract_tool_calls(agent_text))


# Example usage and tests
async def tchains():
    class CustomFormat(BaseModel):
        value: str

    print("=== Testing Basic Chain ===")
    agent_a = FlowAgent(AgentModelData(name="A"))
    agent_b = FlowAgent(AgentModelData(name="B"))
    agent_c = FlowAgent(AgentModelData(name="C"))

    async def a_run(self, query: str):
        print(f"FlowAgent {self.amd.name} running query: {query}")
        return f"Answer from {self.amd.name}"

    async def a_format_class(self, pydantic_model: type[BaseModel],
                             prompt: str,
                             message_context: list[dict] = None,
                             max_retries: int = 2):
        print(f"FlowAgent {self.amd.name} formatting class: {pydantic_model.__name__}")
        return {"value": 'yes' if random.random() < 0.5 else 'no'}
    agent_a.a_run = types.MethodType(a_run, agent_a)
    agent_a.a_format_class = types.MethodType(a_format_class, agent_a)
    agent_b.a_run = types.MethodType(a_run, agent_b)
    agent_b.a_format_class = types.MethodType(a_format_class, agent_b)
    agent_c.a_run = types.MethodType(a_run, agent_c)
    agent_c.a_format_class =types.MethodType(a_format_class, agent_c)

    # Basic sequential chain
    c = agent_a >> agent_b
    result = await c.a_run("Hello World")
    print(f"Result: {result}\n")
    c.print_graph()

    # Three agent chain
    c = agent_a >> agent_c >> agent_b
    result = await c.a_run("Hello World")
    print(f"Three agent result: {result}\n")
    c.print_graph()

    print("=== Testing Format Chain ===")
    # Chain with formatting
    c = CF(CustomFormat) >> agent_a >> CF(CustomFormat) >> agent_b
    result = await c.a_run(CustomFormat(value="Hello World"))
    print(f"Format chain result: {result}\n")
    c.print_graph()

    print("=== Testing Parallel Execution ===")
    # Parallel execution
    c = agent_a + agent_b
    result = await c.a_run("Hello World")
    print(f"Parallel result: {result}\n")
    c.print_graph()

    print("=== Testing Mixed Chain ===")
    # Mixed parallel and sequential
    c = (agent_a & agent_b) >> CF(CustomFormat)
    result = await c.a_run("Hello World")
    print(f"Mixed chain result: {result}\n")
    c.print_graph()

    print("=== Testing Mixed Chain v2 ===")
    # Mixed parallel and sequential
    c = (agent_a >> agent_c >> agent_b & agent_b) >> CF(CustomFormat)
    result = await c.a_run("Hello World")
    print(f"Mixed chain result: {result}\n")
    c.print_graph()

    i = 0
    c: Chain = agent_a >> agent_b
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    c: Chain = agent_a >> agent_c >> agent_b
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    c: Chain = CF(CustomFormat) >> agent_a >> CF(
        CustomFormat) >> agent_b  # using a_format_class intelligently defalt all
    result = await c.a_run(CustomFormat(value="Hello World"))
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    c: Chain = agent_a >> CF(CustomFormat) >> agent_b  # using a_format_class intelligently same as above
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()
    c: Chain = agent_a >> CF(CustomFormat) - '*' >> agent_b  # using a_format_class intelligently same as above
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()
    c: Chain = agent_a >> CF(CustomFormat) - 'value' >> agent_b  # using a_format_class intelligently
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()
    c: Chain = agent_a >> CF(CustomFormat) - '*value' >> agent_b  # using a_format_class intelligently
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()
    c: Chain = agent_a >> CF(CustomFormat) - ('value', 'value2') >> agent_b  # using a_format_class intelligently
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    c: Chain = agent_a >> CF(
        CustomFormat) - 'value[n]' >> agent_b  # using a_format_class intelligently runs b n times parallel
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    c: Chain = agent_a >> CF(CustomFormat) - IS('value',
                                                'yes') >> agent_b % agent_c  # using a_format_class intelligently runs b n times parallel
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    print("Cinc")
    chain_x = agent_b >> CF(CustomFormat)
    chain_z = agent_c >> CF(CustomFormat)

    c: Chain = agent_a >> CF(CustomFormat) - IS('value',
                                                'yes') >> chain_x % chain_z  # using a_format_class intelligently runs b n times parallel
    result = await c.a_run("Hello World IS 12")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    c: Chain = agent_a >> CF(CustomFormat) - IS('value', 'yes') >> agent_b + agent_c | CF(
        CustomFormat) - 'error_reson_val_from_agent_a' >> agent_c  # using a_format_class intelligently runs b n times parallel
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    c: Chain = agent_a + agent_b  # runs a and p in parallel combines output inteligently
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    c: Chain = agent_a & agent_b  # same as above
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    c: Chain = agent_a & agent_b >> CF(CustomFormat) - 'value[n]' >> agent_b  # runs agent b n times parallel with different input
    result = await c.a_run("Hello World")
    print(f"test result: {i} {result}\n")
    i += 1
    c.print_graph()

    print("=== Testing Done ===")


async def run_new_custom_chain():
    # --- Agenten-Setup ---

    class CustomFormat(BaseModel):
        value: str
    # (Wir gehen davon aus, dass die Agenten wie im Beispiel-Code definiert sind)
    supervisor_agent = FlowAgent(AgentModelData(name="Supervisor"))
    writer_agent = FlowAgent(AgentModelData(name="Writer"))
    reviewer_agent = FlowAgent(AgentModelData(name="Reviewer"))
    notifier_agent = FlowAgent(AgentModelData(name="Notifier"))

    # Weisen Sie den Agenten die beispielhaften a_run und a_format_class Methoden zu
    # (Dieser Code ist der gleiche wie in Ihrem Beispiel)
    async def a_run(self, query: str):
        print(f"FlowAgent {self.amd.name} running query: {query}")
        return f"Answer from {self.amd.name}"

    async def a_format_class(self, pydantic_model: type[BaseModel],
                             prompt: str,
                             message_context: list[dict] = None,
                             max_retries: int = 2):
        print(f"FlowAgent {self.amd.name} formatting class: {pydantic_model.__name__}")
        # Simuliert eine zufällige Entscheidung
        decision = 'yes' if random.random() < 0.5 else 'no'
        print(f"--> Decision made: {decision}")
        return {"value": decision}

    for agent in [supervisor_agent, writer_agent, reviewer_agent, notifier_agent]:
        agent.a_run = types.MethodType(a_run, agent)
        agent.a_format_class = types.MethodType(a_format_class, agent)

    # --- Die neue übersichtliche Test-Chain ---
    # Logik: Supervisor -> Entscheidung -> (Writer + Reviewer) ODER nichts -> Notifier
    conditional_parallel_chain = (writer_agent + reviewer_agent)

    # Erstellen der vollständigen Kette
    # Wenn der Wert 'yes' ist, führe die conditional_parallel_chain aus.
    # % notifier_agent bedeutet: Wenn die Bedingung nicht erfüllt ist, gehe direkt zu diesem Agenten.
    # Der letzte >> notifier_agent stellt sicher, dass der Notifier immer am Ende läuft (sowohl für den 'yes'- als auch für den 'no'-Pfad).
    c: Chain = supervisor_agent >> CF(CustomFormat) - IS('value', 'yes') >> conditional_parallel_chain % notifier_agent >> notifier_agent

    print("--- Start: Neue Test-Chain ---")
    result = await c.a_run("Start the content creation process")
    print(f"\nFinal Result of the Chain: {result}\n")
    c.print_graph()
    print("--- Ende: Neue Test-Chain ---")

if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_new_custom_chain())

if __name__ == "__main__2":


    # Simple test
    async def _agent():
        amd = AgentModelData(
        name="TestAgent",
        fast_llm_model="groq/llama-3.3-70b-versatile",
        complex_llm_model="openrouter/qwen/qwen3-coder",
        persona=PersonaConfig(
            name="Isaa",
            style="light and perishes",
            tone="modern friendly",
            personality_traits=["intelligent", "autonomous", "duzen", "not formal"],
            custom_instructions="dos not like to Talk in to long sanitize and texts."
            )
        )
        agent = FlowAgent(amd, verbose=True)

        # Load latest checkpoint with full history restoration
        result = await agent.load_latest_checkpoint(auto_restore_history=True, max_age_hours=24)

        if result["success"]:
            print(f"Loaded checkpoint from {result['checkpoint_timestamp']}")
            print(f"Restored {result['restore_stats']['conversation_history_entries']} conversation entries")
            print(f"Restored {result['restore_stats']['tasks_restored']} tasks")
            print(f"Restored {result['restore_stats']['world_model_entries']} world model entries")
        else:
            print(f"Failed to load checkpoint: {result['error']}")

        # List available checkpoints
        checkpoints = agent.list_available_checkpoints(max_age_hours=168)  # 1 week
        for cp in checkpoints:
            print(f"Checkpoint: {cp['filename']} (age: {cp['age_hours']}h, size: {cp['file_size_kb']}kb)")

        # Clean up old checkpoints
        cleanup_result = await agent.delete_old_checkpoints(keep_count=5, max_age_hours=168)
        print(f"Deleted {cleanup_result['deleted_count']} old checkpoints, freed {cleanup_result['freed_space_kb']}kb")

        def get_user_name():
            return "Markin"

        print(agent.get_available_variables())
        await agent.add_tool(get_user_name, "get_user_name", "Get the user's name")
        print("online")
        import time
        t = time.perf_counter()
        response = await agent.a_run("is 1980 45 years ago?")
        print(f"Time: {time.perf_counter() - t}")
        print(f"Response: {response}")
        agent.status(pretty_print=True)

        while True:
            query = input("Query: ")
            if query == "r":
                res = await agent.explain_reasoning_process()
                print(res)
                continue
            if query == "exit":
                break
            response = await agent.a_run(query)
            print(f"Response: {response}")

        await agent.close()

    asyncio.run(_agent())
