import os
import re

import asyncio
import yaml
import json
import logging
import pickle
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, asdict, field
from pydantic import BaseModel, Field, create_model
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import uuid
import time

# PocketFlow imports
from pocketflow import AsyncNode, AsyncFlow, BatchNode, Flow, Node

# Framework imports with graceful degradation
try:
    import litellm
    from litellm import BudgetManager, Usage
    from litellm.utils import get_max_tokens
    LITELLM_AVAILABLE = True
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

logger = get_logger()
litllm_logger = logging.getLogger("LiteLLM")
litllm_logger.setLevel(logging.CRITICAL) #(get_logger().level)

TASK_TYPES = ["llm_call", "tool_call", "analysis", "generic"]

# ===== CORE DATA STRUCTURES =====
class dAsyncFlowT(AsyncFlow):
    """
    Pass through observer class
    adds print statements to
    async def prep_async(self,shared): pass
    async def exec_async(self,prep_res): pass
    async def exec_fallback_async(self,prep_res,exc): raise exc
    async def post_async(self,shared,prep_res,exec_res): pass
    async def run_async(self,shared):

    with args.
    """
    async def prep_async(self, shared):
        logger.info(f"Prep: {self.__class__.__name__}")
        return await super().prep_async(shared)

    async def exec_async(self, prep_res):
        logger.info(f"Exec: {self.__class__.__name__} args {json.dumps(prep_res, indent=2, errors='ignore') if isinstance(prep_res, dict) else prep_res}")
        return await super().exec_async(prep_res)

    async def exec_fallback_async(self, prep_res, exc):
        logger.info(f"ExecFallback: {self.__class__.__name__} args {json.dumps(prep_res, indent=2, errors='ignore') if isinstance(prep_res, dict) else prep_res} exc {exc}")
        return await super().exec_fallback_async(prep_res, exc)
    async def post_async(self, shared, prep_res, exec_res):
        logger.info(f"Post: {self.__class__.__name__} args {json.dumps(prep_res, indent=2, errors='ignore') if isinstance(prep_res, dict) else prep_res} exec_res {json.dumps(exec_res, indent=2, errors='ignore') if isinstance(exec_res, dict) else exec_res}")
        return await super().post_async(shared, prep_res, exec_res)
    async def run_async(self, shared):
        logger.info(f"Run: {self.__class__.__name__}")
        return await super().run_async(shared)

class AsyncNodeT(AsyncNode):
    """
    Pass through observer class
    adds print statements to
    async def prep_async(self,shared): pass
    async def exec_async(self,prep_res): pass
    async def exec_fallback_async(self,prep_res,exc): raise exc
    async def post_async(self,shared,prep_res,exec_res): pass
    async def run_async(self,shared):

    with args.
    """
    async def prep_async(self, shared):
        logger.info(f"Prep: {self.__class__.__name__}")
        return await super().prep_async(shared)

    async def exec_async(self, prep_res):
        logger.info(f"Exec: {self.__class__.__name__} args {json.dumps(prep_res, indent=2, errors='ignore') if isinstance(prep_res, dict) else prep_res}")
        return await super().exec_async(prep_res)

    async def exec_fallback_async(self, prep_res, exc):
        logger.info(f"ExecFallback: {self.__class__.__name__} args {json.dumps(prep_res, indent=2, errors='ignore') if isinstance(prep_res, dict) else prep_res} exc {exc}")
        return await super().exec_fallback_async(prep_res, exc)
    async def post_async(self, shared, prep_res, exec_res):
        logger.info(f"Post: {self.__class__.__name__} args {prep_res} exec_res {json.dumps(exec_res, indent=2, errors='ignore') if isinstance(exec_res, dict) else exec_res}")
        return await super().post_async(shared, prep_res, exec_res)
    async def run_async(self, shared):
        logger.info(f"Run: {self.__class__.__name__} ")
        return await super().run_async(shared)

AsyncFlowT = AsyncFlow
dAsyncNodeT = AsyncNode

@dataclass
class Task:
    id: str
    type: str
    description: str
    status: str = "pending"  # pending, running, completed, failed, paused
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    result: Any = None
    error: str = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    critical: bool = False

@dataclass
class TaskPlan:
    id: str
    name: str
    description: str
    tasks: List[Task] = field(default_factory=list)
    status: str = "created"  # created, running, paused, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_strategy: str = "sequential"  # sequential, parallel, mixed

@dataclass
class LLMTask(Task):
    """Spezialisierter Task für LLM-Aufrufe"""
    llm_config: Dict[str, Any] = field(default_factory=lambda: {
        "model_preference": "fast",  # "fast" | "complex"
        "temperature": 0.7,
        "max_tokens": 1024
    })
    prompt_template: str = ""
    context_keys: List[str] = field(default_factory=list)  # Keys aus shared state
    output_schema: Optional[Dict] = None  # JSON Schema für Validierung


@dataclass
class ToolTask(Task):
    """Spezialisierter Task für Tool-Aufrufe"""
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)  # Kann {{ }} Referenzen enthalten
    hypothesis: str = ""  # Was erwarten wir von diesem Tool?
    validation_criteria: str = ""  # Wie validieren wir das Ergebnis?
    expectation: str = ""  # Wie sollte das Ergebnis aussehen?


@dataclass
class DecisionTask(Task):
    """Task für dynamisches Routing"""
    decision_prompt: str = ""  # Kurze Frage an LLM
    routing_map: Dict[str, str] = field(default_factory=dict)  # Ergebnis -> nächster Task
    decision_model: str = "fast"  # Welches LLM für Entscheidung


@dataclass
class CompoundTask(Task):
    """Task der Sub-Tasks gruppiert"""
    sub_task_ids: List[str] = field(default_factory=list)
    execution_strategy: str = "sequential"  # "sequential" | "parallel"
    success_criteria: str = ""  # Wann ist der Compound-Task erfolgreich?


# Erweiterte Task-Erstellung
def create_task(task_type: str, **kwargs) -> Task:
    """Factory für Task-Erstellung mit korrektem Typ"""
    task_classes = {
        "llm_call": LLMTask,
        "tool_call": ToolTask,
        "decision": DecisionTask,
        "compound": CompoundTask,
        "generic": Task,
        "LLMTask": LLMTask,
        "ToolTask": ToolTask,
        "DecisionTask": DecisionTask,
        "CompoundTask": CompoundTask,
        "Task": Task,
    }

    task_class = task_classes.get(task_type, Task)

    # Standard-Felder setzen
    if "id" not in kwargs:
        kwargs["id"] = str(uuid.uuid4())
    if "type" not in kwargs:
        kwargs["type"] = task_type
    if "critical" not in kwargs:
        kwargs["critical"] = task_type in ["llm_call", "decision"]

    return task_class(**kwargs)

@dataclass
class AgentCheckpoint:
    timestamp: datetime
    agent_state: Dict[str, Any]
    task_state: Dict[str, Any]
    world_model: Dict[str, Any]
    active_flows: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonaConfig:
    name: str
    style: str = "professional"
    personality_traits: List[str] = field(default_factory=lambda: ["helpful", "concise"])
    tone: str = "friendly"
    response_format: str = "direct"
    custom_instructions: str = ""

    apply_method: str = "system_prompt"  # "system_prompt" | "post_process" | "both"
    integration_level: str = "light"  # "light" | "medium" | "heavy"

    def to_system_prompt_addition(self) -> str:
        """Convert persona to system prompt addition"""
        if self.apply_method in ["system_prompt", "both"]:
            additions = []
            additions.append(f"You are {self.name}.")
            additions.append(f"Your communication style is {self.style} with a {self.tone} tone.")

            if self.personality_traits:
                traits_str = ", ".join(self.personality_traits)
                additions.append(f"Your key traits are: {traits_str}.")

            if self.custom_instructions:
                additions.append(self.custom_instructions)

            return " ".join(additions)
        return ""

    def should_post_process(self) -> bool:
        """Check if post-processing should be applied"""
        return self.apply_method in ["post_process", "both"]

class AgentModelData(BaseModel):
    name: str = "FlowAgent"
    fast_llm_model: str = "openrouter/anthropic/claude-3-haiku"
    complex_llm_model: str = "openrouter/openai/gpt-4o"
    system_message: str = "You are a production-ready autonomous agent."
    temperature: float = 0.7
    max_tokens: int = 2048
    max_input_tokens: int = 32768
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    budget_manager: Optional[Any] = None
    caching: bool = True
    persona: Optional[PersonaConfig] = None
    use_fast_response: bool = True

    def get_system_message_with_persona(self) -> str:
        """Get system message with persona integration"""
        base_message = self.system_message

        if self.persona and self.persona.apply_method in ["system_prompt", "both"]:
            persona_addition = self.persona.to_system_prompt_addition()
            if persona_addition:
                base_message += f"\n\n## Persona Instructions\n{persona_addition}"

        return base_message

# ===== CORE NODE IMPLEMENTATIONS =====



class ContextManagerNode(AsyncNodeT):
    """Advanced context management with intelligent splitting"""

    def __init__(self, max_tokens: int = 8000, **kwargs):
        super().__init__(**kwargs)
        self.max_tokens = max_tokens

    async def prep_async(self, shared):
        session_id = shared.get("session_id", "default")
        current_query = shared.get("current_query", "")
        history = shared.get("conversation_history", [])
        world_model = shared.get("world_model", {})

        return {
            "session_id": session_id,
            "current_query": current_query,
            "history": history,
            "world_model": world_model,
            "tasks": shared.get("tasks", {}),
            "system_context": shared.get("system_context", {})
        }

    async def exec_async(self, prep_res):
        # Split context into 3 parts as requested
        recent_interaction = self._extract_recent_interaction(prep_res)
        instructions = self._generate_instructions(prep_res)
        compressed_context = await self._compress_context(prep_res)

        context = {
            "recent_interaction": recent_interaction,
            "instructions": instructions,
            "compressed_context": compressed_context,
            "total_tokens": self._estimate_tokens(recent_interaction + instructions + compressed_context)
        }

        return context

    def _extract_recent_interaction(self, prep_res):
        history = prep_res["history"]
        current_query = prep_res["current_query"]

        # Get last 2-3 exchanges
        recent = history[-6:] if len(history) > 6 else history
        if recent == current_query:
            return "No recent interaction."
        return f"Recent conversation:\n{self._format_history(recent)}\nCurrent query: {current_query}"

    def _generate_instructions(self, prep_res):
        tasks = prep_res["tasks"]
        active_tasks = [t for t in tasks.values() if t.status == "running"]

        instructions = "## Current Instructions\n"
        if active_tasks:
            instructions += "Active tasks:\n"
            for task in active_tasks:
                instructions += f"- {task.description} (Priority: {task.priority})\n"

        system_context = prep_res["system_context"]
        if system_context.get("strategy"):
            instructions += f"\nCurrent strategy: {system_context['strategy']}\n"

        return instructions

    async def _compress_context(self, prep_res):
        world_model = prep_res["world_model"]
        history = prep_res["history"]

        # Create compressed summary of relevant context
        relevant_facts = []
        for key, value in world_model.items():
            if self._is_relevant(key, prep_res["current_query"]):
                relevant_facts.append(f"{key}: {value}")

        compressed = "## Relevant Context\n"
        compressed += "\n".join(relevant_facts[:10])  # Top 10 most relevant

        return compressed

    def _is_relevant(self, key: str, query: str) -> bool:
        # Simple relevance check - can be enhanced with embeddings
        query_words = query.lower().split()
        key_words = key.lower().split()
        return any(word in key_words for word in query_words)

    def _format_history(self, history: List) -> str:
        formatted = []
        for entry in history:
            if isinstance(entry, dict):
                role = entry.get("role", "unknown")
                content = entry.get("content", "")
                formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    def _estimate_tokens(self, text: str) -> int:
        # Rough estimation: 4 chars per token
        return len(text) // 4

    async def post_async(self, shared, prep_res, exec_res):
        shared["formatted_context"] = exec_res
        shared["context_tokens"] = exec_res["total_tokens"]
        return "context_ready"

class YAMLFormatterNode(AsyncNodeT):
    """Enhanced YAML formatter with schema-based generation"""

    def __init__(self, schema_class: Optional[Type[BaseModel]] = None, **kwargs):
        super().__init__(**kwargs)
        self.schema_class = schema_class

    async def prep_async(self, shared):
        task_description = shared.get("current_task_description", "")
        schema_mode = shared.get("yaml_format_mode", "general")
        custom_schema = shared.get("custom_schema", {})
        raw_input = shared.get("raw_llm_output", "")

        return {
            "task_description": task_description,
            "schema_mode": schema_mode,
            "custom_schema": custom_schema,
            "raw_input": raw_input,
            "context": shared.get("formatted_context", {}),
            "fast_llm_model": shared.get("fast_llm_model"),
            "complex_llm_model": shared.get("complex_llm_model")
        }

    async def exec_async(self, prep_res):
        if prep_res["raw_input"]:
            # Parse existing LLM output into YAML
            return await self._parse_to_yaml(prep_res)
        else:
            # Generate new YAML based on schema
            return await self._generate_yaml_from_schema(prep_res)

    async def _parse_to_yaml(self, prep_res):
        raw_input = prep_res["raw_input"]

        try:
            # Try to extract YAML from markdown code blocks
            if "```yaml" in raw_input:
                yaml_content = raw_input.split("```yaml")[1].split("```")[0].strip()
            elif "```" in raw_input:
                yaml_content = raw_input.split("```")[1].split("```")[0].strip()
            else:
                yaml_content = raw_input

            parsed = yaml.safe_load(yaml_content)
            return {
                "success": True,
                "data": parsed,
                "raw_yaml": yaml_content
            }
        except Exception as e:
            logger.error(f"Failed to parse YAML: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback": {"raw_content": raw_input}
            }

    async def _generate_yaml_from_schema(self, prep_res):
        schema_mode = prep_res["schema_mode"]

        if self.schema_class:
            schema = self.schema_class.model_json_schema()
        else:
            schema = self._get_default_schema(schema_mode)

        # Generate LLM prompt to create YAML based on schema
        prompt = self._build_schema_prompt(schema, prep_res)

        if LITELLM_AVAILABLE:
            try:
                # Use fast model from shared context
                model_to_use = prep_res.get("fast_llm_model", "openrouter/anthropic/claude-3-haiku")
                logger.info(f"Using model {model_to_use} for YAML generation")
                response = await litellm.acompletion(
                    model=model_to_use,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                yaml_content = response.choices[0].message.content

                # Extract and validate YAML
                if "```yaml" in yaml_content:
                    yaml_str = yaml_content.split("```yaml")[1].split("```")[0].strip()
                else:
                    yaml_str = yaml_content.strip()

                parsed = yaml.safe_load(yaml_str)
                return {
                    "success": True,
                    "data": parsed,
                    "raw_yaml": yaml_str
                }
            except Exception as e:
                logger.error(f"LLM YAML generation failed: {e}")
                return self._generate_fallback_yaml(prep_res)
        else:
            return self._generate_fallback_yaml(prep_res)

    def _get_default_schema(self, mode: str) -> Dict:
        schemas = {
            "task_plan": {
                "type": "object",
                "properties": {
                    "plan_name": {"type": "string"},
                    "description": {"type": "string"},
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "description": {"type": "string"},
                                "priority": {"type": "integer"},
                                "dependencies": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    }
                }
            },
            "action": {
                "type": "object",
                "properties": {
                    "action_type": {"type": "string"},
                    "parameters": {"type": "object"},
                    "reasoning": {"type": "string"}
                }
            },
            "analysis": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                    "recommendations": {"type": "array", "items": {"type": "string"}}
                }
            }
        }
        return schemas.get(mode, schemas["analysis"])

    def _build_schema_prompt(self, schema: Dict, prep_res: Dict) -> str:
        return f"""
Generate a YAML structure based on the following schema and context:

## Task Description
{prep_res['task_description']}

## Required Schema
```yaml
{yaml.safe_dump(schema, indent=2)}
```
Context
{prep_res.get('context', {})}
Generate valid YAML that conforms to this schema.
Wrap your response in one
```yaml
```
code block!.
"""
    def _generate_fallback_yaml(self, prep_res):
        fallback = {
            "task_description": prep_res["task_description"],
            "schema_mode": prep_res["schema_mode"],
            "timestamp": datetime.now().isoformat()
        }
        return {
            "success": True,
            "data": fallback,
            "raw_yaml": yaml.dump(fallback)
        }

    async def post_async(self, shared, prep_res, exec_res):
        shared["formatted_yaml"] = exec_res
        if exec_res["success"]:
            shared["structured_data"] = exec_res["data"]
        return "formatted" if exec_res["success"] else "format_failed"

class StrategyOrchestratorNode(AsyncNodeT):
    """Strategic orchestration with meta-reasoning"""
    def __init__(self, strategies: Dict[str, Dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.strategies = strategies or self._get_default_strategies()

    async def prep_async(self, shared):
        current_query = shared.get("current_query", "")
        task_stack = shared.get("tasks", {})
        world_model = shared.get("world_model", {})
        system_status = shared.get("system_status", "idle")
        recent_performance = shared.get("performance_metrics", {})

        agent_instance = shared.get("agent_instance")
        tool_capabilities = {}
        if agent_instance and hasattr(agent_instance, '_tool_capabilities'):
            tool_capabilities = agent_instance._tool_capabilities
        return {
            "query": current_query,
            "tasks": task_stack,
            "world_model": world_model,
            "system_status": system_status,
            "performance": recent_performance,
            "available_strategies": list(self.strategies.keys()),
            "fast_llm_model": shared.get("fast_llm_model"),
            "complex_llm_model": shared.get("complex_llm_model"),
            "tool_capabilities": tool_capabilities,
            "available_tools_names": shared.get("available_tools", [])
        }

    async def exec_async(self, prep_res):
        # LLM-basierte Strategieauswahl
        strategy = await self._determine_strategy_llm(prep_res)

        # Generate execution plan
        execution_plan = await self._create_execution_plan(strategy, prep_res)

        return {
            "selected_strategy": strategy,
            "execution_plan": execution_plan,
            "reasoning": self._get_strategy_reasoning(strategy, prep_res),
            "estimated_complexity": self._estimate_complexity(prep_res)
        }

    async def _determine_strategy_llm(self, prep_res) -> str:
        """Enhanced strategy determination with tool awareness"""

        if not LITELLM_AVAILABLE:
            return "direct_response"

        # Build tool context
        tool_context = self._build_tool_awareness_context(prep_res)

        prompt = f"""
    You are a strategic AI agent. Analyze the query and your available capabilities to select the optimal strategy.

    ## User Query
    {prep_res['query']}

    ## Your Available Tools & Capabilities
    {tool_context}

    ## System Status
    - Active tasks: {len(prep_res['tasks'])}
    - System status: {prep_res['system_status']}

    ## Available Strategies:
    - direct_response: Simple response without tools (ONLY if no relevant tools available)
    - multi_step_planning: Complex task breakdown with tool usage
    - research_and_analyze: Information gathering using available tools
    - creative_generation: Content creation, may use tools for context
    - problem_solving: Analysis and problem solving with tools

    ## IMPORTANT DECISION CRITERIA:
    1. If you have tools that can DIRECTLY answer the query, use multi_step_planning
    2. If tools can provide context/data for better responses, use research_and_analyze
    3. Only use direct_response if NO tools are relevant
    4. Consider INDIRECT connections - tools might be useful even if not obvious

    ## Analysis Process:
    1. Check if any tools can directly fulfill the request
    2. Check if tools can provide supporting information
    3. Consider personalization opportunities
    4. Look for non-obvious connections

    Respond ONLY with the strategy name:"""

        try:
            model_to_use = prep_res.get("fast_llm_model", "openrouter/anthropic/claude-3-haiku")

            response = await litellm.acompletion(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Lower temperature for more consistent tool usage
                max_tokens=50
            )

            strategy = response.choices[0].message.content.strip().lower()

            if strategy in self.strategies:
                logger.info(f"Selected strategy: {strategy} (tool-aware)")
                return strategy
            else:
                logger.warning(f"Invalid strategy returned: {strategy}, using multi_step_planning as fallback")
                return "multi_step_planning"  # Better fallback than direct_response

        except Exception as e:
            logger.error(f"Strategy determination failed: {e}")
            return "multi_step_planning"  # Tool-aware fallback

    def _build_tool_awareness_context(self, prep_res: Dict) -> str:
        """Build comprehensive tool context for strategy decisions"""

        tool_capabilities = prep_res.get("tool_capabilities", {})
        available_tools = prep_res.get("available_tools_names", [])

        if not available_tools:
            return "No tools available."

        context_parts = []
        context_parts.append("### Available Tools:")

        for tool_name in available_tools:
            if tool_name in tool_capabilities:
                cap = tool_capabilities[tool_name]
                context_parts.append(f"\n**{tool_name}:**")
                context_parts.append(f"- Primary function: {cap.get('primary_function', 'Unknown')}")
                context_parts.append(f"- Use cases: {', '.join(cap.get('use_cases', [])[:3])}")
                context_parts.append(f"- Triggers: {', '.join(cap.get('trigger_phrases', [])[:5])}")

                # Add indirect connections
                indirect = cap.get('indirect_connections', [])
                if indirect:
                    context_parts.append(f"- Indirect uses: {', '.join(indirect[:3])}")
            else:
                # Fallback for tools without analysis
                context_parts.append(f"\n**{tool_name}:** Available but not analyzed")

        return "\n".join(context_parts)

    async def _create_execution_plan(self, strategy: str, prep_res: Dict) -> Dict:
        strategy_config = self.strategies[strategy]

        plan = {
            "strategy": strategy,
            "phases": strategy_config["phases"],
            "parallel_capable": strategy_config.get("parallel_capable", False),
            "estimated_steps": len(strategy_config["phases"]),
            "resource_requirements": strategy_config.get("resources", {}),
            "success_criteria": strategy_config.get("success_criteria", [])
        }

        return plan

    def _get_default_strategies(self) -> Dict[str, Dict]:
        return {
            "direct_response": {
                "phases": ["context_prep", "llm_call", "response_format"],
                "parallel_capable": False,
                "resources": {"llm_calls": 1, "complexity": "low"}
            },
            "multi_step_planning": {
                "phases": ["task_decomposition", "dependency_analysis", "parallel_execution", "result_synthesis"],
                "parallel_capable": True,
                "resources": {"llm_calls": "multiple", "complexity": "high"}
            },
            "research_and_analyze": {
                "phases": ["query_expansion", "information_gathering", "analysis", "synthesis"],
                "parallel_capable": True,
                "resources": {"llm_calls": "multiple", "tools": ["search", "analysis"]}
            },
            "creative_generation": {
                "phases": ["ideation", "structure_planning", "content_generation", "refinement"],
                "parallel_capable": False,
                "resources": {"llm_calls": "multiple", "complexity": "medium"}
            },
            "problem_solving": {
                "phases": ["problem_analysis", "solution_exploration", "implementation_planning", "validation"],
                "parallel_capable": True,
                "resources": {"llm_calls": "multiple", "tools": ["code_execution", "testing"]}
            }
        }

    def _get_strategy_reasoning(self, strategy: str, prep_res: Dict) -> str:
        return f"Selected '{strategy}' based on query analysis and current system state"

    def _estimate_complexity(self, prep_res: Dict) -> str:
        task_count = len(prep_res["tasks"])
        query_length = len(prep_res["query"].split())

        if task_count > 5 or query_length > 100:
            return "high"
        elif task_count > 2 or query_length > 20:
            return "medium"
        else:
            return "low"

    async def post_async(self, shared, prep_res, exec_res):
        shared["selected_strategy"] = exec_res["selected_strategy"]
        shared["execution_plan"] = exec_res["execution_plan"]
        shared["strategy_reasoning"] = exec_res["reasoning"]
        return exec_res["selected_strategy"]


class TaskPlannerNode(AsyncNodeT):
    """Erweiterte Aufgabenplanung mit dynamischen Referenzen und Tool-Integration"""

    async def prep_async(self, shared):
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
        }

    async def exec_async(self, prep_res):
        if prep_res["strategy"] == "direct_response":
            return self._create_simple_plan(prep_res)
        else:
            return await self._advanced_llm_decomposition(prep_res)

    async def post_async(self, shared, prep_res, exec_res):
        """Post-processing nach Plan-Erstellung"""

        if exec_res is None:
            shared["planning_error"] = "Plan creation returned None"
            return "planning_failed"

        if isinstance(exec_res, TaskPlan):
            # Erfolgreicher Plan
            shared["current_plan"] = exec_res

            # Tasks in shared state für Executor verfügbar machen
            task_dict = {task.id: task for task in exec_res.tasks}
            shared["tasks"].update(task_dict)

            # Plan-Metadaten setzen
            shared["plan_created_at"] = datetime.now().isoformat()
            shared["plan_strategy"] = exec_res.execution_strategy
            shared["total_tasks_planned"] = len(exec_res.tasks)

            logger.info(f"Plan created successfully: {exec_res.name} with {len(exec_res.tasks)} tasks")
            return "planned"

        else:
            # Plan creation failed
            shared["planning_error"] = "Invalid plan format returned"
            shared["current_plan"] = None
            logger.error("Plan creation failed - invalid format")
            return "planning_failed"

    def _create_simple_plan(self, prep_res) -> TaskPlan:
        """Fast lightweight planning for direct or simple multi-step queries."""
        taw = self._build_tool_intelligence(prep_res)
        logger.info("You are a FAST "+ taw)
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
plan_name: "Direct Response"
description: "Quick answer from LLM"
execution_strategy: "sequential"
tasks:
  - id: "answer"
    type: "LLMTask"
    description: "Respond to query"
    priority: 1
    dependencies: []
    prompt_template: "Respond concisely to: {prep_res['query']}"
    llm_config:
      model_preference: "fast"
      temperature: 0.3
```
Example 2 — Tool + LLM
```yaml
plan_name: "Fetch and Answer"
description: "Get info from tool and summarize"
execution_strategy: "sequential"
tasks:
  - id: "fetch_info"
    type: "ToolTask"
    description: "Get required data"
    priority: 1
    dependencies: []
    tool_name: "info_api"
    arguments:
      query: "{{ prep_res['query'] }}"
  - id: "summarize"
    type: "LLMTask"
    description: "Summarize fetched data"
    priority: 2
    dependencies: ["fetch_info"]
    prompt_template: "Summarize: {{ results.fetch_info.data }}"
    llm_config:
      model_preference: "fast"
      temperature: 0.3
```
Output Requirements
Use ONLY YAML for the final output
Pick minimal plan type for fastest completion!
    """

        try:
            response = litellm.completion(
                model=prep_res.get("fast_llm_model", "openrouter/anthropic/claude-3-haiku"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512
            )
            content = response.choices[0].message.content

            yaml_content = content.split("```yaml")[1].split("```")[0].strip() if "```yaml" in content else content
            plan_data = yaml.safe_load(yaml_content)
            print("Simple", plan_data)
            return TaskPlan(
                id=str(uuid.uuid4()),
                name=plan_data.get("plan_name", "Generated Plan"),
                description=plan_data.get("description", f"Plan for: {prep_res['query']}"),
                tasks=[
                    [LLMTask, ToolTask, DecisionTask, CompoundTask, Task][["LLMTask", "ToolTask", "DecisionTask", "CompoundTask", "Task"].index(t.get("type"))](**t)
                    for t in plan_data.get("tasks", [])
                ],
                execution_strategy=plan_data.get("execution_strategy", "sequential")
            )

        except Exception as e:
            logger.error(f"Simple plan creation failed: {e}")
            import traceback
            print(traceback.format_exc())
            return TaskPlan(
                id=str(uuid.uuid4()),
                name="Fallback Plan",
                description="Direct response only",
                tasks=[
                    LLMTask(
                        id="direct_response",
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
        """Erweiterte LLM-basierte Dekomposition mit Tool-Integration"""
        tool_intelligence = self._build_tool_intelligence(prep_res)
        prompt = f"""
You are an expert **task planner** and **abstract pattern recognizer** with access to specialized tools and task types.
Your goal: Create intelligent, structured **execution plans** in YAML format by first recognizing patterns in the query and tool capabilities.

## User Query
{prep_res['query']}

## Your Available Tools & Intelligence
{tool_intelligence}

## CRITICAL INSTRUCTIONS:
1. ALWAYS check if tools can directly answer the query
2. Look for INDIRECT ways tools provide value
3. Use tools for personalization when available
4. Create tool_call tasks for any relevant tools
5. Don't ignore tools - they are your primary capability!

## STAGE 1 — ABSTRACT PATTERN RECOGNITION & REASONING
Before creating any YAML plan:
1. Quickly identify the **type of user query** (informational, action-based, decision-based, multi-step, compound).
2. Recognize **patterns** in task structure: sequential chains, parallel steps, decision-routing, grouped subtasks.
3. Abstractly map query intent to **tool usage patterns** (direct tool calls, LLM augmentation, routing, compound processing).
4. Decide optimal **execution strategy** (sequential, parallel, mixed).
5. Keep reasoning concise but precise — this is for your own internal planning, not for the final YAML output.

**Pattern Recognition Output Format (Internal Only)**:
```

[Pattern Summary]: Short abstract description of plan type.
[Reasoning]: How the structure should be organized and why.

```

## STAGE 2 — YAML PLAN GENERATION
After reasoning, produce **only valid YAML** for the execution plan.
Follow the dataclass structure below.

## TASK TYPES (Dataclass-Aligned)
- **Task**: Generic step in the plan.
- **LLMTask**: Step that uses a language model.
- **ToolTask**: Step that calls an available tool.
- **DecisionTask**: Step that decides routing between tasks.
- **CompoundTask**: Step grouping sub-tasks.

## YAML SCHEMA
```yaml
plan_name: string
description: string
execution_strategy: "sequential" | "parallel" | "mixed"
tasks:
  - id: string
    type: "Task" | "LLMTask" | "ToolTask" | "DecisionTask" | "CompoundTask"
    description: string
    priority: int
    dependencies: [list of task ids]
    # Additional fields depending on type:
    # LLMTask: prompt_template, llm_config, context_keys
    # ToolTask: tool_name, arguments, hypothesis, validation_criteria, expectation
    # DecisionTask: decision_prompt, routing_map, decision_model
    # CompoundTask: sub_task_ids, execution_strategy, success_criteria
```

## EXAMPLES WITH PATTERN RECOGNITION
## IMPORTANT ALL TOOLS USED IN THE EXAMPLES AR NOT AUTOMATICALLY AVAILABLE! only tools from "Your Available Tools & Intelligence" AR Available!!!
### Example 1 — Basic ToolTask + LLMTask

**Pattern Summary**: Simple sequential pipeline — fetch → personalize.
**Reasoning**: Direct tool call retrieves data, then LLM formats/generates output.

```yaml
plan_name: "Personalized Greeting"
description: "Get user's name and greet them"
execution_strategy: "sequential"
tasks:
  - id: "get_user_name"
    type: "ToolTask"
    description: "Retrieve user's name"
    priority: 1
    dependencies: []
    tool_name: "get_user_name"
    arguments: {{}}
    hypothesis: "We will obtain the user's name"
    validation_criteria: "Must return a string"
  - id: "greet_user"
    type: "LLMTask"
    description: "Generate greeting"
    priority: 2
    dependencies: ["get_user_name"]
    prompt_template: "Greet the user named {{ results.get_user_name.data }}"
    llm_config:
      model_preference: "fast"
      temperature: 0.7
```

### Example 2 — DecisionTask + ToolTask + LLMTask

**Pattern Summary**: Branching logic — decision routing to alternate flows.
**Reasoning**: Decision task checks intent, then either calls weather tool or LLM for smalltalk.

```yaml
plan_name: "Weather Routing"
description: "Route based on user weather query"
execution_strategy: "sequential"
tasks:
  - id: "check_weather_intent"
    type: "DecisionTask"
    description: "Decide if weather info is needed"
    priority: 1
    dependencies: []
    decision_prompt: "Does the query ask about weather?"
    routing_map:
      yes: "get_weather"
      no: "smalltalk"
    decision_model: "fast"   # options fast/complx use fastest model for decision making
  - id: "get_weather"
    type: "ToolTask"
    description: "Fetch weather data"
    priority: 2
    dependencies: ["check_weather_intent"]
    tool_name: "weather_api"
    arguments:
      location: "{{ user.location }}"
  - id: "smalltalk"
    type: "LLMTask"
    description: "General conversation"
    priority: 2
    dependencies: ["check_weather_intent"]
    prompt_template: "Respond to user casually."
```

### Example 3 — CompoundTask with Mixed Subtasks

**Pattern Summary**: Multi-stage compound execution — data fetch → process → compile report.
**Reasoning**: Parallel processing possible for some subtasks; final report requires sequential completion.

```yaml
plan_name: "Data Report Generation"
description: "Generate and format report"
execution_strategy: "mixed"
tasks:
  - id: "fetch_data"
    type: "ToolTask"
    description: "Retrieve raw data"
    priority: 1
    dependencies: []
    tool_name: "data_fetcher"
    arguments: {{}}
  - id: "process_data"
    type: "LLMTask"
    description: "Analyze data"
    priority: 2
    dependencies: ["fetch_data"]
    prompt_template: "Summarize dataset: {{ results.fetch_data.data }}"
  - id: "generate_report"
    type: "CompoundTask"
    description: "Compile and format report"
    priority: 3
    dependencies: ["process_data"]
    sub_task_ids: ["process_data", "format_report"]
    execution_strategy: "sequential"
    success_criteria: "Final report file is ready"
  - id: "format_report"
    type: "LLMTask"
    description: "Format into PDF"
    priority: 4
    dependencies: ["process_data"]
    prompt_template: "Format summary into PDF layout"
```

## FINAL OUTPUT REQUIREMENTS

* Perform **Pattern Recognition** first (internal reasoning phase).
* Output **only the YAML plan** for the final answer.
* Combine up to **6 tasks** from any types.
* Ensure logical dependencies and optimal execution strategy.
* Include at least one **DecisionTask** or **CompoundTask** in complex plans.

## NOW GENERATE:

Produce a complete YAML plan for the given query using the above reasoning process.
"""
        try:
            model_to_use = prep_res.get("complex_llm_model", "openrouter/openai/gpt-4o")

            response = await litellm.acompletion(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2048
            )

            content = response.choices[0].message.content
            if "```yaml" in content:
                yaml_content = content.split("```yaml")[1].split("```")[0].strip()
            else:
                yaml_content = content

            plan_data = yaml.safe_load(yaml_content)
            print(f"Advanced Plan:\n{json.dumps(plan_data, indent=2)}")
            # Tasks mit spezialisierten Klassen erstellen
            tasks = []
            for task_data in plan_data.get("tasks", []):
                task_type = task_data.pop("type", "generic")

                # Spezialisierte Task-Erstellung
                if task_type == "tool_call":
                    task = ToolTask(
                        id=task_data.get("id", str(uuid.uuid4())),
                        type=task_type,
                        description=task_data.get("description", ""),
                        priority=task_data.get("priority", 1),
                        dependencies=task_data.get("dependencies", []),
                        tool_name=task_data.get("tool_name", ""),
                        arguments=task_data.get("arguments", {}),
                        hypothesis=task_data.get("hypothesis", ""),
                        validation_criteria=task_data.get("validation_criteria", ""),
                        critical=task_data.get("priority", 1) == 1
                    )
                elif task_type == "llm_call":
                    task = LLMTask(
                        id=task_data.get("id", str(uuid.uuid4())),
                        type=task_type,
                        description=task_data.get("description", ""),
                        priority=task_data.get("priority", 1),
                        dependencies=task_data.get("dependencies", []),
                        prompt_template=task_data.get("prompt_template", task_data.get("description", "")),
                        context_keys=task_data.get("context_keys", []),
                        llm_config=task_data.get("llm_config", {}),
                        critical=task_data.get("priority", 1) == 1
                    )
                elif task_type == "decision":
                    task = DecisionTask(
                        id=task_data.get("id", str(uuid.uuid4())),
                        type=task_type,
                        description=task_data.get("description", ""),
                        priority=task_data.get("priority", 1),
                        dependencies=task_data.get("dependencies", []),
                        decision_prompt=task_data.get("decision_prompt", ""),
                        routing_map=task_data.get("routing_map", {}),
                        critical=True  # Decisions sind immer kritisch
                    )
                else:
                    task = create_task(task_type, **task_data)

                tasks.append(task)

            plan = TaskPlan(
                id=str(uuid.uuid4()),
                name=plan_data.get("plan_name", "Generated Plan"),
                description=plan_data.get("description", f"Plan for: {prep_res['query']}"),
                tasks=tasks,
                execution_strategy=plan_data.get("execution_strategy", "sequential")
            )

            logger.info(f"Created plan with {len(tasks)} specialized tasks")
            return plan

        except Exception as e:
            logger.error(f"Advanced task decomposition failed: {e}")
            import traceback
            print(traceback.format_exc())
            return self._create_simple_plan(prep_res)

    def _build_tool_intelligence(self, prep_res: Dict) -> str:
        """Build detailed tool intelligence for planning"""

        agent_instance = prep_res.get("agent_instance")
        if not agent_instance or not hasattr(agent_instance, '_tool_capabilities'):
            return "No tool intelligence available."

        capabilities = agent_instance._tool_capabilities
        query = prep_res.get('query', '').lower()

        context_parts = []
        context_parts.append("### Intelligent Tool Analysis:")

        for tool_name, cap in capabilities.items():
            context_parts.append(f"\n**{tool_name}:**")
            context_parts.append(f"- Function: {cap.get('primary_function', 'Unknown')}")

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

    def _calculate_tool_relevance(self, query: str, capabilities: Dict) -> float:
        """Calculate how relevant a tool is to the current query"""

        query_words = set(query.lower().split())

        # Check trigger phrases
        trigger_score = 0.0
        triggers = capabilities.get('trigger_phrases', [])
        for trigger in triggers:
            trigger_words = set(trigger.lower().split())
            if trigger_words.intersection(query_words):
                trigger_score += 0.4

        # Check confidence triggers if available
        conf_triggers = capabilities.get('confidence_triggers', {})
        for phrase, confidence in conf_triggers.items():
            if phrase.lower() in query:
                trigger_score += confidence

        # Check indirect connections
        indirect = capabilities.get('indirect_connections', [])
        for connection in indirect:
            connection_words = set(connection.lower().split())
            if connection_words.intersection(query_words):
                trigger_score += 0.2

        return min(1.0, trigger_score)


class TaskExecutorNode(AsyncNodeT):
    """Vollständige Task-Ausführung als unabhängige Node mit LLM-unterstützter Planung"""

    def __init__(self, max_parallel: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_parallel = max_parallel
        self.results_store = {}  # Für {{ }} Referenzen
        self.execution_history = []  # Für LLM-basierte Optimierung
        self.agent_instance = None  # Wird gesetzt vom FlowAgent

    async def prep_async(self, shared):
        """Intelligente Vorbereitung der Task-Ausführung mit LLM-Unterstützung"""

        current_plan = shared.get("current_plan")
        tasks = shared.get("tasks", {})

        # Stelle sicher, dass Agent-Referenz verfügbar ist
        if not self.agent_instance:
            self.agent_instance = shared.get("agent_instance")

        if not current_plan:
            return {"error": "No active plan", "tasks": tasks}

        # Analysiere verfügbare Tasks
        ready_tasks = self._find_ready_tasks(current_plan, tasks)
        blocked_tasks = self._find_blocked_tasks(current_plan, tasks)

        # LLM-basierte Ausführungsplanung
        execution_plan = await self._create_intelligent_execution_plan(
            ready_tasks, blocked_tasks, current_plan, shared
        )

        # Integriere results_store aus shared wenn vorhanden
        if "results_store" in shared:
            self.results_store.update(shared["results_store"])

        return {
            "plan": current_plan,
            "ready_tasks": ready_tasks,
            "blocked_tasks": blocked_tasks,
            "all_tasks": tasks,
            "execution_plan": execution_plan,
            "fast_llm_model": shared.get("fast_llm_model"),
            "complex_llm_model": shared.get("complex_llm_model"),
            "available_tools": shared.get("available_tools", []),
            "world_model": shared.get("world_model", {}),
            "results_store": self.results_store
        }

    def _find_ready_tasks(self, plan: TaskPlan, all_tasks: Dict[str, Task]) -> List[Task]:
        """Finde Tasks die zur Ausführung bereit sind"""
        ready = []
        for task in plan.tasks:
            if task.status == "pending" and self._dependencies_satisfied(task, all_tasks):
                ready.append(task)
        return ready

    def _find_blocked_tasks(self, plan: TaskPlan, all_tasks: Dict[str, Task]) -> List[Task]:
        """Finde blockierte Tasks für Analyse"""
        blocked = []
        for task in plan.tasks:
            if task.status == "pending" and not self._dependencies_satisfied(task, all_tasks):
                blocked.append(task)
        return blocked

    def _dependencies_satisfied(self, task: Task, all_tasks: Dict[str, Task]) -> bool:
        """Prüfe ob alle Dependencies erfüllt sind"""
        for dep_id in task.dependencies:
            if dep_id in all_tasks:
                dep_task = all_tasks[dep_id]
                if dep_task.status not in ["completed"]:
                    return False
            else:
                # Dependency existiert nicht - könnte Problem sein
                logger.warning(f"Task {task.id} has missing dependency: {dep_id}")
                return False
        return True

    async def _create_intelligent_execution_plan(
        self,
        ready_tasks: List[Task],
        blocked_tasks: List[Task],
        plan: TaskPlan,
        shared: Dict
    ) -> Dict[str, Any]:
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

    def _create_simple_execution_plan(self, ready_tasks: List[Task], plan: TaskPlan) -> Dict[str, Any]:
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
        ready_tasks: List[Task],
        blocked_tasks: List[Task],
        plan: TaskPlan,
        shared: Dict
    ) -> Dict[str, Any]:
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

            response = await litellm.acompletion(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )

            content = response.choices[0].message.content
            yaml_match = re.search(r"```yaml\s*(.*?)\s*```", content, re.DOTALL)
            yaml_str = yaml_match.group(1) if yaml_match else content.strip()

            execution_plan = yaml.safe_load(yaml_str)

            # Validiere und erweitere den Plan
            validated_plan = self._validate_execution_plan(execution_plan, ready_tasks)

            logger.info(
                f"LLM execution plan created: {validated_plan.get('strategy')} with {len(validated_plan.get('execution_groups', []))} groups")
            return validated_plan

        except Exception as e:
            logger.error(f"LLM execution planning failed: {e}")
            return self._create_simple_execution_plan(ready_tasks, plan)

    def _analyze_tasks_for_llm(self, ready_tasks: List[Task], blocked_tasks: List[Task]) -> Dict[str, str]:
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

    def _build_execution_context(self, shared: Dict) -> str:
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

    def _validate_execution_plan(self, plan: Dict, ready_tasks: List[Task]) -> Dict:
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

    def _estimate_duration(self, tasks: List[Task]) -> int:
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
            logger.error(f"Execution plan failed: {e}")
            return {
                "status": "execution_failed",
                "error": str(e),
                "results": []
            }

    async def _execute_parallel_plan(self, plan: Dict, prep_res: Dict) -> List[Dict]:
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
                logger.error(f"Critical task failures in group {group['group_id']}, stopping execution")
                break

        return all_results

    async def _execute_sequential_plan(self, plan: Dict, prep_res: Dict) -> List[Dict]:
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

    async def _execute_hybrid_plan(self, plan: Dict, prep_res: Dict) -> List[Dict]:
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
                    logger.error(f"Group execution failed: {result_group}")
                    continue
                all_results.extend(result_group)

        return all_results

    def _get_tasks_by_ids(self, task_ids: List[str], prep_res: Dict) -> List[Task]:
        """Hole Task-Objekte basierend auf IDs"""
        all_tasks = prep_res["all_tasks"]
        return [all_tasks[tid] for tid in task_ids if tid in all_tasks]

    def _is_critical_task(self, task_id: str, prep_res: Dict) -> bool:
        """Prüfe ob Task kritisch ist"""
        task = prep_res["all_tasks"].get(task_id)
        if not task:
            return False
        return getattr(task, 'critical', False) or task.priority == 1

    async def _execute_parallel_batch(self, tasks: List[Task]) -> List[Dict]:
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
                    logger.error(f"Task {batch[i].id} failed with exception: {result}")
                    processed_results.append({
                        "task_id": batch[i].id,
                        "status": "failed",
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)

            all_results.extend(processed_results)

        return all_results

    async def _execute_sequential_batch(self, tasks: List[Task]) -> List[Dict]:
        """Führe Tasks sequenziell aus"""
        results = []

        for task in tasks:
            try:
                result = await self._execute_single_task(task)
                results.append(result)

                # Stoppe bei kritischen Fehlern in sequenzieller Ausführung
                if result.get("status") == "failed" and getattr(task, 'critical', False):
                    logger.error(f"Critical task {task.id} failed, stopping sequential execution")
                    break

            except Exception as e:
                logger.error(f"Sequential task {task.id} failed: {e}")
                results.append({
                    "task_id": task.id,
                    "status": "failed",
                    "error": str(e)
                })

                if getattr(task, 'critical', False):
                    break

        return results

    async def _execute_single_task(self, task: Task) -> Dict:
        """Task-Ausführung (unverändert von vorher)"""
        try:
            task.status = "running"
            task.started_at = datetime.now()

            # Dynamische Argumentauflösung vor Ausführung
            if isinstance(task, ToolTask):
                resolved_args = await self._resolve_dynamic_arguments(task.arguments)
                result = await self._execute_tool_task_enhanced(task, resolved_args)
            elif isinstance(task, LLMTask):
                result = await self._execute_llm_task_enhanced(task)
            elif isinstance(task, DecisionTask):
                result = await self._execute_decision_task(task)
            else:
                result = await self._execute_llm_task(task)

            # Ergebnis strukturiert speichern
            self.results_store[task.id] = {
                "data": result,
                "metadata": {
                    "task_type": task.type,
                    "completed_at": datetime.now().isoformat(),
                    "success": True
                }
            }

            task.result = result
            task.status = "completed"
            task.completed_at = datetime.now()

            # Verifikation für ToolTasks
            verification_result = None
            if isinstance(task, ToolTask) and task.hypothesis:
                verification_result = await self._verify_tool_result_enhanced(task)
                task.metadata["verification"] = verification_result

            return {
                "task_id": task.id,
                "status": "completed",
                "result": result,
                "verification": verification_result
            }

        except Exception as e:
            task.error = str(e)
            task.status = "failed"
            task.retry_count += 1

            # Auch Fehler in results_store speichern
            self.results_store[task.id] = {
                "error": str(e),
                "metadata": {
                    "task_type": task.type,
                    "failed_at": datetime.now().isoformat(),
                    "success": False
                }
            }

            logger.error(f"Task {task.id} failed: {e}")
            return {
                "task_id": task.id,
                "status": "failed",
                "error": str(e),
                "retry_count": task.retry_count
            }

    async def _resolve_dynamic_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Löse {{ results.task_id.key }} Referenzen auf"""
        resolved = {}

        for key, value in arguments.items():
            if isinstance(value, str) and "{{" in value and "}}" in value:
                # Referenz-Pattern finden: {{ results.task_id.key }}
                pattern = r'\{\{\s*results\.([^.]+)\.([^}\s]+)\s*\}\}'
                matches = re.findall(pattern, value)

                resolved_value = value
                for task_id, result_key in matches:
                    if task_id in self.results_store:
                        task_result = self.results_store[task_id]

                        # Navigiere zum gewünschten Wert
                        try:
                            if result_key == "data":
                                replacement = task_result["data"]
                            elif result_key in task_result:
                                replacement = task_result[result_key]
                            elif "data" in task_result and isinstance(task_result["data"], dict):
                                replacement = task_result["data"].get(result_key, f"KEY_NOT_FOUND:{result_key}")
                            else:
                                replacement = f"REFERENCE_ERROR:results.{task_id}.{result_key}"

                            # Ersetze die Referenz
                            ref_pattern = f"{{\\s*results\\.{task_id}\\.{result_key}\\s*}}"
                            resolved_value = re.sub(ref_pattern, str(replacement), resolved_value)

                        except Exception as e:
                            logger.warning(f"Failed to resolve reference results.{task_id}.{result_key}: {e}")
                            resolved_value = f"RESOLVE_ERROR:{task_id}.{result_key}"

                resolved[key] = resolved_value
            else:
                resolved[key] = value

        return resolved

    async def _execute_tool_task_enhanced(self, task: ToolTask, resolved_args: Dict[str, Any]) -> Any:
        """Erweiterte Tool-Ausführung mit aufgelösten Argumenten"""

        if not task.tool_name:
            raise ValueError(f"ToolTask {task.id} missing tool_name")

        agent = self.agent_instance
        if not agent:
            raise ValueError("Agent instance not available for tool execution")

        try:
            logger.info(f"Executing tool {task.tool_name} with resolved args: {resolved_args}")
            result = await agent.arun_function(task.tool_name, **resolved_args)

            return result

        except Exception as e:
            logger.error(f"Enhanced tool execution failed for {task.tool_name}: {e}")
            raise

    async def _execute_llm_task_enhanced(self, task: LLMTask) -> Any:
        """Erweiterte LLM-Task Ausführung mit Template-System"""

        if not LITELLM_AVAILABLE:
            raise Exception("LiteLLM not available for LLM tasks")

        # Model-Präferenz auflösen
        llm_config = task.llm_config
        model_preference = llm_config.get("model_preference", "fast")

        if model_preference == "complex":
            model_to_use = task.metadata.get("complex_llm_model", "openrouter/openai/gpt-4o")
        else:
            model_to_use = task.metadata.get("fast_llm_model", "openrouter/anthropic/claude-3-haiku")

        # Prompt aus Template erstellen
        prompt = task.prompt_template

        # Context-Keys auflösen
        for context_key in task.context_keys:
            if context_key.startswith("results."):
                # Dynamische Referenz auflösen
                pattern = r'results\.([^.]+)\.(.+)'
                match = re.match(pattern, context_key)
                if match:
                    task_id, result_key = match.groups()
                    if task_id in self.results_store:
                        context_value = self.results_store[task_id].get("data", "")
                        placeholder = f"{{{{{context_key}}}}}"
                        prompt = prompt.replace(placeholder, str(context_value))

        response = await litellm.acompletion(
            model=model_to_use,
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 1024)
        )

        result = response.choices[0].message.content

        # Output-Schema Validierung falls vorhanden
        if task.output_schema:
            try:
                # Versuche JSON zu parsen und gegen Schema zu validieren
                if result.strip().startswith('{') or result.strip().startswith('['):
                    parsed = json.loads(result)
                    # Hier könnte eine Schema-Validierung implementiert werden
                    logger.info(f"LLM output validated against schema for task {task.id}")
            except json.JSONDecodeError:
                logger.warning(f"LLM output for task {task.id} is not valid JSON")

        return result

    async def _execute_decision_task(self, task: DecisionTask) -> str:
        """Führe Decision Task aus und gib Routing-Entscheidung zurück"""

        if not LITELLM_AVAILABLE:
            raise Exception("LiteLLM not available for decision tasks")

        # Fast model für Entscheidungen
        model_to_use = task.metadata.get("fast_llm_model", "openrouter/anthropic/claude-3-haiku")

        # Decision prompt mit Context erweitern
        enhanced_prompt = f"""
{task.decision_prompt}

Available routing options: {', '.join(task.routing_map.keys())}

Respond with EXACTLY one of these options - nothing else:"""

        response = await litellm.acompletion(
            model=model_to_use,
            messages=[{"role": "user", "content": enhanced_prompt}],
            temperature=0.1,
            max_tokens=50
        )

        decision = response.choices[0].message.content.strip().upper()

        # Validiere Decision gegen routing_map
        if decision not in task.routing_map:
            logger.warning(f"Decision '{decision}' not in routing map, using first option")
            decision = list(task.routing_map.keys())[0] if task.routing_map else "DEFAULT"

        # Speichere nächsten Task in metadata für Router
        next_task_id = task.routing_map.get(decision, "")
        task.metadata["next_task_id"] = next_task_id
        task.metadata["decision_made"] = decision

        return decision

    async def _verify_tool_result_enhanced(self, task: ToolTask) -> Dict[str, Any]:
        """Erweiterte Verifikation mit strukturierter Bewertung"""

        result = task.result
        hypothesis = task.hypothesis
        validation_criteria = task.validation_criteria

        if not LITELLM_AVAILABLE:
            return {"status": "no_verification", "reason": "LiteLLM unavailable"}

        prompt = f"""
Bewerte das Tool-Ausführungsergebnis systematisch:

## Tool-Kontext
Tool: {task.tool_name}
Hypothese: {hypothesis}
Validierungskriterien: {validation_criteria}

## Tatsächliches Ergebnis
{result}

## Bewertungsaufgabe
Bewerte das Ergebnis auf einer Skala von 0.0 bis 1.0:

Antworte NUR mit gültigem YAML:

```yaml
hypothesis_score: 0.85  # Wie gut wurde die Hypothese erfüllt?
criteria_score: 0.92    # Wie gut erfüllt es die Validierungskriterien?
usefulness_score: 0.88  # Wie nützlich ist das Ergebnis insgesamt?
overall_status: "success"  # success | partial_success | failure
confidence: 0.87        # Wie sicher ist diese Bewertung?
reasoning: "Brief explanation"
recommendations:
  - "Specific suggestion 1"
  - "Specific suggestion 2"
next_actions:
  - action: "refine_search"
    reason: "Need more specific results"
    priority: "medium"
```"""

        try:
            model_to_use = task.metadata.get("complex_llm_model", "openrouter/openai/gpt-4o")

            response = await litellm.acompletion(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800
            )

            content = response.choices[0].message.content
            yaml_match = re.search(r"```yaml\s*(.*?)\s*```", content, re.DOTALL)
            yaml_str = yaml_match.group(1) if yaml_match else content.strip()

            verification = yaml.safe_load(yaml_str)

            # Validierung der Scores
            for score_key in ["hypothesis_score", "criteria_score", "usefulness_score", "confidence"]:
                if score_key in verification:
                    verification[score_key] = max(0.0, min(1.0, float(verification[score_key])))

            logger.info(f"Enhanced verification for task {task.id}: {verification.get('overall_status')}")
            return verification

        except Exception as e:
            logger.error(f"Enhanced tool verification failed: {e}")
            return {
                "status": "verification_error",
                "error": str(e),
                "hypothesis_score": 0.0,
                "criteria_score": 0.0,
                "usefulness_score": 0.0
            }



        ### ===========================================

    def _resolve_template_string(self, template: str) -> str:
        """Fallback method for template resolution"""
        # Simple replacement using results_store
        import re

        def replace_match(match):
            ref = match.group(1).strip()
            if ref.startswith("results.") and "." in ref[8:]:
                parts = ref.split(".")
                task_id = parts[1]
                field = ".".join(parts[2:])

                if task_id in self.results_store:
                    result_data = self.results_store[task_id]
                    if field == "data":
                        return str(result_data.get("data", ""))
                    else:
                        return str(result_data.get(field, ""))

            return match.group(0)  # Return original if not resolved

        return re.sub(r'\{\{\s*([^}]+)\s*}}', replace_match, template)

    async def post_async(self, shared, prep_res, exec_res):
        """Erweiterte Post-Processing mit Performance-Tracking"""

        # Results store in shared state integrieren
        shared["results_store"] = self.results_store

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

        # Task-Status updates
        for result in exec_res.get("results", []):
            task_id = result["task_id"]
            if task_id in shared["tasks"]:
                task = shared["tasks"][task_id]
                task.status = result["status"]
                if result["status"] == "completed":
                    task.result = result["result"]
                    # Speichere Verifikationsergebnisse
                    if result.get("verification"):
                        if not hasattr(task, 'metadata'):
                            task.metadata = {}
                        task.metadata["verification"] = result["verification"]
                elif result["status"] == "failed":
                    task.error = result.get("error", "Unknown error")

        # Plan completion check
        current_plan = shared["current_plan"]
        if current_plan:
            all_finished = all(
                shared["tasks"][task.id].status in ["completed", "failed"]
                for task in current_plan.tasks
            )

            if all_finished:
                current_plan.status = "completed"
                shared["plan_completion_time"] = datetime.now().isoformat()
                logger.info(f"Plan {current_plan.id} finished")
                return "plan_completed"  # Goes to completion checker
            else:
                # Still has work to do
                ready_tasks = [
                    task for task in current_plan.tasks
                    if shared["tasks"][task.id].status == "pending"
                ]

                if ready_tasks:
                    return "continue_execution"  # Goes to completion checker
                else:
                    return "waiting"  # Goes to completion checker

        return "execution_complete"  # Fallback

    def get_execution_statistics(self) -> Dict[str, Any]:
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

    async def optimize_future_executions(self, shared: Dict) -> Dict[str, Any]:
        """LLM-basierte Optimierung für zukünftige Ausführungen"""

        if not LITELLM_AVAILABLE or not self.execution_history:
            return {"status": "no_optimization", "reason": "No LLM or history available"}

        stats = self.get_execution_statistics()

        prompt = f"""
Du bist ein Performance-Optimierungsexperte für Task-Execution. Analysiere die Ausführungshistorie und gib Optimierungsempfehlungen.

## Aktuelle Performance-Statistiken
{yaml.safe_dump(stats, indent=2)}

## Letzte Ausführungen
{yaml.safe_dump(self.execution_history[-5:], indent=2)}

## Aufgabe
Basierend auf den Daten, identifiziere:
1. Leistungsengpässe
2. Optimierungspotentiale
3. Strategieempfehlungen
4. Resource-Optimierungen

Antworte mit YAML:

```yaml
analysis:
  performance_trend: "improving" | "declining" | "stable"
  bottlenecks:
    - type: "parallel_overhead"
      description: "Too much parallelization causing overhead"
      impact: "medium"
  strengths:
    - "Tool tasks execute efficiently"
    - "Sequential LLM tasks work well"

optimizations:
  - category: "strategy"
    recommendation: "Use hybrid execution for mixed workloads"
    expected_improvement: "15% faster execution"
    confidence: 0.8
  - category: "resource"
    recommendation: "Reduce parallel limit for tool-heavy tasks"
    expected_improvement: "Better resource utilization"
    confidence: 0.9

future_strategy_preferences:
  - condition: "mostly_tool_tasks"
    preferred_strategy: "parallel"
    max_parallel: 2
  - condition: "mixed_workload"
    preferred_strategy: "hybrid"
    max_parallel: 3

confidence: 0.75
```"""

        try:
            model_to_use = shared.get("complex_llm_model", "openrouter/openai/gpt-4o")

            response = await litellm.acompletion(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )

            content = response.choices[0].message.content
            yaml_match = re.search(r"```yaml\s*(.*?)\s*```", content, re.DOTALL)
            yaml_str = yaml_match.group(1) if yaml_match else content.strip()

            optimization = yaml.safe_load(yaml_str)

            # Speichere Optimierungsempfehlungen
            if not hasattr(self, 'optimization_recommendations'):
                self.optimization_recommendations = []

            self.optimization_recommendations.append({
                "timestamp": datetime.now().isoformat(),
                "optimization": optimization
            })

            logger.info(
                f"Execution optimization analysis completed with confidence {optimization.get('confidence', 0.0)}")
            return optimization

        except Exception as e:
            logger.error(f"Execution optimization failed: {e}")
            return {"status": "optimization_failed", "error": str(e)}


class PlanReflectorNode(AsyncNodeT):
    """Adaptive Plan-Reflexion und -Anpassung"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def prep_async(self, shared):
        current_plan = shared.get("current_plan")
        tasks = shared.get("tasks", {})
        results_store = shared.get("results_store", {})

        recently_completed = [
            task for task in (current_plan.tasks if current_plan else [])
            if task.status == "completed" and
               task.completed_at and
               (datetime.now() - task.completed_at).seconds < 300  # Letzte 5 min
        ]

        return {
            "current_plan": current_plan,
            "tasks": tasks,
            "results_store": results_store,
            "recently_completed": recently_completed,
            "fast_llm_model": shared.get("fast_llm_model"),
            "complex_llm_model": shared.get("complex_llm_model"),
            "agent_instance": shared.get("agent_instance")
        }

    async def exec_async(self, prep_res):
        if not prep_res["recently_completed"]:
            return {"action": "CONTINUE", "reason": "No recent completions to reflect on"}

        # Analysiere kürzlich abgeschlossene Tasks
        reflection_result = await self._analyze_recent_completions(prep_res)

        # Entscheide über nächste Aktionen
        action_decision = await self._decide_next_action(reflection_result, prep_res)

        # Führe adaptive Aktionen aus
        if action_decision["action"] == "ADAPT":
            adaptation_result = await self._adapt_plan(action_decision, prep_res)
            return adaptation_result
        elif action_decision["action"] == "REPLAN":
            replan_result = await self._trigger_replanning(action_decision, prep_res)
            return replan_result
        else:
            return action_decision

    async def _analyze_recent_completions(self, prep_res) -> Dict[str, Any]:
        """Analysiere kürzlich abgeschlossene Tasks auf Probleme und Erfolge"""

        recently_completed = prep_res["recently_completed"]
        results_store = prep_res["results_store"]

        analysis = {
            "successful_tasks": [],
            "problematic_tasks": [],
            "verification_issues": [],
            "unexpected_results": [],
            "missing_information": []
        }

        for task in recently_completed:
            task_analysis = await self._analyze_single_task(task, results_store)

            if task_analysis["status"] == "successful":
                analysis["successful_tasks"].append(task_analysis)
            elif task_analysis["status"] == "problematic":
                analysis["problematic_tasks"].append(task_analysis)
            elif task_analysis["status"] == "unexpected":
                analysis["unexpected_results"].append(task_analysis)

        return analysis

    async def _analyze_single_task(self, task: Task, results_store: Dict) -> Dict[str, Any]:
        """Detaillierte Analyse eines einzelnen Tasks"""

        task_result = results_store.get(task.id, {})
        verification = task.metadata.get("verification", {}) if hasattr(task, 'metadata') else {}

        # ToolTask spezielle Analyse
        if isinstance(task, ToolTask) and task.hypothesis:
            hypothesis_score = verification.get("hypothesis_score", 0.5)
            criteria_score = verification.get("criteria_score", 0.5)

            if hypothesis_score < 0.3 or criteria_score < 0.3:
                return {
                    "task_id": task.id,
                    "status": "problematic",
                    "issue": "hypothesis_not_met",
                    "details": {
                        "hypothesis": task.hypothesis,
                        "hypothesis_score": hypothesis_score,
                        "criteria_score": criteria_score,
                        "tool_name": task.tool_name
                    }
                }
            elif 0.3 <= hypothesis_score < 0.7 or 0.3 <= criteria_score < 0.7:
                return {
                    "task_id": task.id,
                    "status": "unexpected",
                    "issue": "partial_success",
                    "details": {
                        "hypothesis": task.hypothesis,
                        "actual_result": task_result.get("data", ""),
                        "verification": verification
                    }
                }
            else:
                return {
                    "task_id": task.id,
                    "status": "successful",
                    "details": verification
                }

        # DecisionTask Analyse
        elif isinstance(task, DecisionTask):
            decision_made = task.metadata.get("decision_made", "") if hasattr(task, 'metadata') else ""
            if decision_made and decision_made in task.routing_map:
                return {
                    "task_id": task.id,
                    "status": "successful",
                    "decision": decision_made,
                    "next_task": task.routing_map[decision_made]
                }
            else:
                return {
                    "task_id": task.id,
                    "status": "problematic",
                    "issue": "invalid_decision",
                    "decision": decision_made
                }

        # Standard Task Analyse
        else:
            if task.result and not task.error:
                return {"task_id": task.id, "status": "successful"}
            else:
                return {
                    "task_id": task.id,
                    "status": "problematic",
                    "issue": "execution_error",
                    "error": task.error
                }

    async def _decide_next_action(self, reflection_result: Dict, prep_res: Dict) -> Dict[str, Any]:
        """LLM-basierte Entscheidung über nächste Aktionen"""

        if not LITELLM_AVAILABLE:
            return {"action": "CONTINUE", "reason": "No LLM available for decision making"}

        prompt = f"""
Du bist ein Plan-Reflexionssystem. Analysiere die Task-Ergebnisse und entscheide über die nächste Aktion.

## Analyse der kürzlich abgeschlossenen Tasks
Erfolgreiche Tasks: {len(reflection_result['successful_tasks'])}
Problematische Tasks: {len(reflection_result['problematic_tasks'])}
Unerwartete Ergebnisse: {len(reflection_result['unexpected_results'])}

## Detaillierte Probleme
{self._format_analysis_for_prompt(reflection_result)}

## Verfügbare Aktionen
- CONTINUE: Alles läuft nach Plan, weitermachen
- ADAPT: Plan um zusätzliche Tasks erweitern (für unerwartete Ergebnisse)
- REPLAN: Kompletter Neustart der Planung (bei kritischen Fehlern)
- HALT_FAILURE: Plan abbrechen (bei unüberwindbaren Problemen)

## Entscheidung
Antworte mit YAML:

```yaml
action: "ADAPT"  # Eine der obigen Aktionen
confidence: 0.85  # Wie sicher ist diese Entscheidung?
reasoning: "Brief explanation of the decision"
priority: "high"  # high | medium | low
adaptation_suggestions:
  - task_type: "tool_call"
    description: "Additional search with refined query"
    reason: "Original search didn't find expected information"
  - task_type: "llm_call"
    description: "Analyze alternative data sources"
    reason: "Need to explore different approaches"
```"""

        try:
            model_to_use = prep_res.get("fast_llm_model", "openrouter/anthropic/claude-3-haiku")

            response = await litellm.acompletion(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=600
            )

            content = response.choices[0].message.content
            yaml_match = re.search(r"```yaml\s*(.*?)\s*```", content, re.DOTALL)
            yaml_str = yaml_match.group(1) if yaml_match else content.strip()

            decision = yaml.safe_load(yaml_str)

            # Validierung
            valid_actions = ["CONTINUE", "ADAPT", "REPLAN", "HALT_FAILURE"]
            if decision.get("action") not in valid_actions:
                decision["action"] = "CONTINUE"

            logger.info(
                f"Plan reflection decision: {decision.get('action')} (confidence: {decision.get('confidence', 0.0)})")
            return decision

        except Exception as e:
            logger.error(f"Plan reflection decision failed: {e}")
            return {"action": "CONTINUE", "reason": f"Decision error: {e}"}

    def _format_analysis_for_prompt(self, reflection_result: Dict) -> str:
        """Formatiere Analyse-Ergebnisse für LLM-Prompt"""
        formatted = []

        for problematic in reflection_result.get("problematic_tasks", []):
            formatted.append(f"PROBLEM - Task {problematic['task_id']}: {problematic.get('issue', 'Unknown')}")

        for unexpected in reflection_result.get("unexpected_results", []):
            formatted.append(f"UNEXPECTED - Task {unexpected['task_id']}: {unexpected.get('issue', 'Unknown')}")

        return "\n".join(formatted) if formatted else "No significant issues detected."

    async def _adapt_plan(self, decision: Dict, prep_res: Dict) -> Dict[str, Any]:
        """Führe Plan-Adaptation aus basierend auf Entscheidung"""

        current_plan = prep_res["current_plan"]
        agent_instance = prep_res["agent_instance"]

        adaptation_suggestions = decision.get("adaptation_suggestions", [])
        new_tasks = []

        for suggestion in adaptation_suggestions:
            # Erstelle neue Tasks basierend auf Suggestions
            task_id = f"adapt_{str(uuid.uuid4())[:8]}"

            if suggestion.get("task_type") == "tool_call":
                new_task = ToolTask(
                    id=task_id,
                    type="tool_call",
                    description=suggestion.get("description", ""),
                    priority=2,  # Mittlere Priorität für adaptive Tasks
                    dependencies=[],  # Können sofort ausgeführt werden
                    tool_name=self._infer_tool_name(suggestion, agent_instance),
                    arguments=self._generate_tool_arguments(suggestion),
                    hypothesis=f"This adaptation will address: {suggestion.get('reason', '')}",
                    validation_criteria="Results should provide the missing information"
                )
            elif suggestion.get("task_type") == "llm_call":
                new_task = LLMTask(
                    id=task_id,
                    type="llm_call",
                    description=suggestion.get("description", ""),
                    priority=2,
                    dependencies=[],
                    prompt_template=f"Analyze the situation: {suggestion.get('reason', '')}",
                    llm_config={"model_preference": "fast"}
                )
            else:
                new_task = create_task("generic",
                                       id=task_id,
                                       description=suggestion.get("description", ""),
                                       priority=2
                                       )

            new_tasks.append(new_task)

        # Füge neue Tasks zum Plan hinzu
        current_plan.tasks.extend(new_tasks)

        # Update shared state
        task_dict = {task.id: task for task in new_tasks}
        prep_res["tasks"].update(task_dict)

        logger.info(f"Plan adapted with {len(new_tasks)} additional tasks")

        return {
            "action": "ADAPT_COMPLETED",
            "new_tasks": [task.id for task in new_tasks],
            "reason": decision.get("reasoning", ""),
            "total_tasks": len(current_plan.tasks)
        }

    def _infer_tool_name(self, suggestion: Dict, agent_instance) -> str:
        """Inferiere Tool-Name basierend auf Suggestion"""
        description = suggestion.get("description", "").lower()
        available_tools = getattr(agent_instance, 'shared', {}).get("available_tools", [])

        # Simple Heuristik für Tool-Auswahl
        if "search" in description and any("search" in tool for tool in available_tools):
            return next(tool for tool in available_tools if "search" in tool)
        elif "analyze" in description and any("analy" in tool for tool in available_tools):
            return next(tool for tool in available_tools if "analy" in tool)
        else:
            return available_tools[0] if available_tools else "generic_tool"

    def _generate_tool_arguments(self, suggestion: Dict) -> Dict[str, Any]:
        """Generiere Tool-Argumente basierend auf Suggestion"""
        description = suggestion.get("description", "")
        reason = suggestion.get("reason", "")

        # Basis-Argumente basierend auf Kontext
        if "search" in description.lower():
            return {
                "query": f"refined search based on: {reason}",
                "max_results": 5
            }
        elif "analyze" in description.lower():
            return {
                "text": "{{ results.previous_task.data }}",
                "analysis_type": "comprehensive"
            }
        else:
            return {"input": f"{description} - {reason}"}

    async def _trigger_replanning(self, decision: Dict, prep_res: Dict) -> Dict[str, Any]:
        """Triggere komplette Neuplanung"""

        logger.warning(f"Triggering complete replan: {decision.get('reasoning', 'Unknown reason')}")

        # Erstelle neuen Planungskontext mit gelernten Informationen
        original_query = prep_res.get("current_query", "")
        learned_context = self._extract_learned_context(prep_res)

        replan_context = {
            "original_query": original_query,
            "previous_attempt_failed": True,
            "learned_context": learned_context,
            "failure_reason": decision.get("reasoning", ""),
            "avoid_approaches": self._extract_failed_approaches(prep_res)
        }

        return {
            "action": "REPLAN_TRIGGERED",
            "replan_context": replan_context,
            "reason": decision.get("reasoning", ""),
            "previous_plan_id": prep_res["current_plan"].id if prep_res["current_plan"] else None
        }

    def _extract_learned_context(self, prep_res: Dict) -> str:
        """Extrahiere gelernten Kontext aus bisherigen Ergebnissen"""
        results_store = prep_res.get("results_store", {})
        learned_facts = []

        for task_id, result in results_store.items():
            if result.get("metadata", {}).get("success", False):
                learned_facts.append(f"Task {task_id} discovered: {str(result.get('data', ''))[:100]}...")

        return "\\n".join(learned_facts)

    def _extract_failed_approaches(self, prep_res: Dict) -> List[str]:
        """Extrahiere fehlgeschlagene Ansätze um sie zu vermeiden"""
        failed_approaches = []
        tasks = prep_res.get("tasks", {})

        for task in tasks.values():
            if task.status == "failed" and isinstance(task, ToolTask):
                failed_approaches.append(f"Tool {task.tool_name} with args {task.arguments}")

        return failed_approaches

    async def post_async(self, shared, prep_res, exec_res):
        """Update shared state basierend auf Reflexions-Ergebnisse"""

        action = exec_res.get("action", "CONTINUE")

        if action == "ADAPT_COMPLETED":
            # Neue Tasks wurden hinzugefügt
            shared["plan_adaptations"] = shared.get("plan_adaptations", 0) + 1
            shared["last_adaptation"] = datetime.now()
            return "adapted"

        elif action == "REPLAN_TRIGGERED":
            # Markiere für komplette Neuplanung
            shared["replan_context"] = exec_res.get("replan_context", {})
            shared["needs_replanning"] = True
            return "needs_replan"

        elif action == "HALT_FAILURE":
            # Plan abbrechen
            shared["plan_halted"] = True
            shared["halt_reason"] = exec_res.get("reason", "Critical failure")
            return "plan_halted"

        else:
            # Normale Fortsetzung
            return "continue"

class LLMToolNode(AsyncNodeT):
    """Enhanced LLM tool with task-specific processing"""
    def __init__(self, model: str | None = None,tools=None, **kwargs):
        super().__init__(**kwargs)
        if model is None:
            model = os.getenv("DEFAULTMODEL1", "openrouter/qwen/qwen3-code")
        self.model = model
        self.tools = tools

    async def prep_async(self, shared):
        context = shared.get("formatted_context", {})
        task_description = shared.get("current_task_description", "")
        hypothesis = shared.get("hypothesis", "")
        expectation = shared.get("expectation", "")
        reasoning = shared.get("reasoning", "")
        tools_available = shared.get("available_tools", [])
        complex_llm_model = shared.get("complex_llm_model", "openrouter/openai/gpt-4o")

        # Variable Manager und Persona hinzufügen
        variable_manager = shared.get("variable_manager")
        persona_config = shared.get("persona_config")

        # Base system message from agent
        agent_instance = shared.get("agent_instance")
        base_system_message = "You are a helpful AI assistant."
        if agent_instance and hasattr(agent_instance, 'amd'):
            base_system_message = agent_instance.amd.get_system_message_with_persona()

        return {
            "context": context,
            "task_description": task_description,
            "hypothesis": hypothesis,
            "expectation": expectation,
            "reasoning": reasoning,
            "tools_available": tools_available,
            "complex_llm_model": complex_llm_model,
            "fast_llm_model": shared.get("fast_llm_model"),
            "variable_manager": variable_manager,
            "persona_config": persona_config,
            "base_system_message": base_system_message
        }

    async def exec_async(self, prep_res):
        if not LITELLM_AVAILABLE:
            return {
                "success": False,
                "error": "LiteLLM not available",
                "fallback_response": "I'm unable to process this request due to missing LLM capabilities."
            }

        prompt = self._build_enhanced_prompt(prep_res)

        # Persona-Integration in System Message
        persona_config = prep_res.get("persona_config")
        system_message = prep_res.get("base_system_message", "You are a helpful AI assistant.")

        if persona_config and persona_config.apply_method in ["system_prompt", "both"]:
            persona_addition = persona_config.to_system_prompt_addition()
            if persona_addition:
                system_message += f"\n\n{persona_addition}"

        # Model selection basierend auf task complexity
        task_description = prep_res.get("task_description", "")
        model_to_use = prep_res.get("complex_llm_model", "openrouter/openai/gpt-4o")

        logger.info(f"Using model {model_to_use} for task {task_description} in LLMToolNode")

        # Messages mit System-Prompt
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        response = await litellm.acompletion(
            model=model_to_use,
            messages=messages,
            temperature=0.7,
            max_tokens=2048
        )

        content = response.choices[0].message.content
        processed_response = await self._process_llm_response(content, prep_res)

        # Post-Process Persona anwenden falls konfiguriert
        if persona_config and persona_config.should_post_process():
            processed_response = await self._process_llm_response(content, prep_res)

        return {
            "success": True,
            "raw_response": content,
            "processed_response": processed_response,
            "model_used": model_to_use,
            "persona_applied": persona_config is not None,
            "usage": response.usage.model_dump() if response.usage else {}
        }

    async def exec_fallback_async(self, prep_res, exc):
        return {
            "success": False,
            "error": str(exc),
            "fallback_response": "I'm unable to process this request due to an error."
        }

    def _build_enhanced_prompt(self, prep_res: Dict) -> str:
        """Enhanced prompt building mit Variable-System"""
        context = prep_res["context"]
        var_manager = prep_res.get("variable_manager")

        # Basis-Prompt erstellen
        prompt = f"""# Task Processing Request
    Task Description
    {prep_res['task_description']}
    Context Information
    {context.get('recent_interaction', '')}
    {context.get('instructions', '')}
    {context.get('compressed_context', '')}
    """

        # Variable-Informationen hinzufügen
        if var_manager:
            available_vars = var_manager.get_available_variables()
            if available_vars:
                prompt += f"\n## Available Variables\n"
                for var_name, preview in list(available_vars.items())[:10]:  # Zeige nur erste 10
                    prompt += f"- {var_name}: {preview}\n"

        if prep_res["hypothesis"]:
            hypothesis_text = prep_res['hypothesis']
            if var_manager:
                hypothesis_text = var_manager.format_text(hypothesis_text)
            prompt += f"\n## Hypothesis\n{hypothesis_text}"

        if prep_res["expectation"]:
            expectation_text = prep_res['expectation']
            if var_manager:
                expectation_text = var_manager.format_text(expectation_text)
            prompt += f"\n## Expected Outcome\n{expectation_text}"

        if prep_res["reasoning"]:
            reasoning_text = prep_res['reasoning']
            if var_manager:
                reasoning_text = var_manager.format_text(reasoning_text)
            prompt += f"\n## Additional Reasoning Context\n{reasoning_text}"

        if prep_res["tools_available"]:
            prompt += f"\n## Available Tools\n{', '.join(prep_res['tools_available'])}"

        prompt += "\n## Response Requirements\nProvide a comprehensive response that addresses the task while considering the context and any constraints mentioned."

        # Finale Variable-Formatierung
        if var_manager:
            prompt = var_manager.format_text(prompt)

        return prompt

    async def _process_llm_response(self, response: str, prep_res: Dict) -> Dict:
        """Process and enhance the LLM response based on task requirements"""

        processed = {
            "main_response": response,
            "confidence": self._estimate_confidence(response),
            "task_completion": self._assess_task_completion(response, prep_res),
            "follow_up_needed": self._identify_follow_up_needs(response),
            "extracted_data": self._extract_structured_data(response)
        }

        return processed

    def _estimate_confidence(self, response: str) -> float:
        # Simple confidence estimation based on language patterns
        confidence_indicators = ["certainly", "definitely", "clearly", "obviously"]
        uncertainty_indicators = ["might", "possibly", "perhaps", "unclear", "uncertain"]

        conf_count = sum(1 for indicator in confidence_indicators if indicator in response.lower())
        uncertain_count = sum(1 for indicator in uncertainty_indicators if indicator in response.lower())

        base_confidence = 0.7
        confidence = base_confidence + (conf_count * 0.1) - (uncertain_count * 0.15)
        return max(0.1, min(1.0, confidence))

    def _assess_task_completion(self, response: str, prep_res: Dict) -> str:
        task_desc = prep_res["task_description"].lower()
        response_lower = response.lower()

        # Check if response addresses key elements of the task
        key_terms = task_desc.split()[:5]  # First 5 words of task
        addressed_terms = sum(1 for term in key_terms if term in response_lower)

        completion_ratio = addressed_terms / max(len(key_terms), 1)

        if completion_ratio > 0.8:
            return "complete"
        elif completion_ratio > 0.5:
            return "partial"
        else:
            return "incomplete"

    def _identify_follow_up_needs(self, response: str) -> List[str]:
        follow_up_patterns = [
            "need more information",
            "would require",
            "further investigation",
            "additional details",
            "follow up"
        ]

        needs = []
        for pattern in follow_up_patterns:
            if pattern in response.lower():
                needs.append(pattern)

        return needs

    def _extract_structured_data(self, response: str) -> Dict[str, Any]:
        """Extract any structured data from the response"""
        extracted = {}

        # Extract lists
        lines = response.split('\n')
        current_list = []
        list_name = None

        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                current_list.append(line[2:])
            elif line.endswith(':') and current_list:
                if list_name:
                    extracted[list_name] = current_list
                list_name = line[:-1].lower().replace(' ', '_')
                current_list = []

        if list_name and current_list:
            extracted[list_name] = current_list

        # Extract key-value pairs
        for line in lines:
            if ':' in line and not line.strip().endswith(':'):
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                if key and value:
                    extracted[key] = value

        return extracted

    async def post_async(self, shared, prep_res, exec_res):
        shared["last_llm_response"] = exec_res

        if exec_res["success"]:
            shared["current_response"] = exec_res["processed_response"]["main_response"]
            shared["response_confidence"] = exec_res["processed_response"]["confidence"]
            return "llm_success"
        else:
            shared["current_response"] = exec_res["fallback_response"]
            return "llm_failed"

class StateSyncNode(AsyncNodeT):
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

    def _extract_facts(self, text: str) -> Dict[str, Any]:
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
            logger.warning("State synchronization failed")
            return "sync_failed"


class CompletionCheckerNode(AsyncNodeT):
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
            logger.warning(f"Max execution cycles ({self.max_cycles}) reached, terminating")
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
            logger.warning(f"Flow force terminated: {exec_res['reason']}")

        return action

# ===== ADVANCED BATCH NODE FOR PARALLEL EXECUTION =====
class ParallelTaskBatch(BatchNode):
    """Batch node for parallel task execution"""
    def prep(self, shared):
        ready_tasks = []
        for task_id, task in shared.get("tasks", {}).items():
            if task.status == "pending":
                ready_tasks.append(task)
        return ready_tasks

    def exec(self, task: Task):
        # This runs in parallel for each task
        try:
            # Simulate task execution
            time.sleep(0.1)
            result = f"Batch result for task: {task.description}"
            return {
                "task_id": task.id,
                "status": "completed",
                "result": result
            }
        except Exception as e:
            return {
                "task_id": task.id,
                "status": "failed",
                "error": str(e)
            }

    def post(self, shared, prep_res, exec_res_list):
        # Process all batch results
        completed_count = 0
        failed_count = 0

        for result in exec_res_list:
            task_id = result["task_id"]
            if task_id in shared["tasks"]:
                task = shared["tasks"][task_id]
                task.status = result["status"]
                if result["status"] == "completed":
                    task.result = result["result"]
                    completed_count += 1
                else:
                    task.error = result.get("error", "Unknown error")
                    failed_count += 1

        shared["batch_execution_stats"] = {
            "completed": completed_count,
            "failed": failed_count,
            "total": len(exec_res_list)
        }

        return "batch_complete"

# ===== FLOW COMPOSITIONS =====
class TaskManagementFlow(AsyncFlowT):
    """Fixed Task-Management-Flow with proper termination"""

    def __init__(self, max_parallel_tasks: int = 3):
        # Create all nodes
        self.strategy_node = StrategyOrchestratorNode()
        self.planner_node = TaskPlannerNode()
        self.executor_node = TaskExecutorNode(max_parallel=max_parallel_tasks)
        self.reflector_node = PlanReflectorNode()
        self.sync_node = StateSyncNode()

        # Add a completion checker node to break cycles
        self.completion_checker = CompletionCheckerNode()

        # === MAIN FLOW (Linear with controlled cycles) ===

        # Strategy -> Planning Phase
        self.strategy_node - "multi_step_planning" >> self.planner_node
        self.strategy_node - "research_and_analyze" >> self.planner_node
        self.strategy_node - "creative_generation" >> self.planner_node
        self.strategy_node - "problem_solving" >> self.planner_node
        self.strategy_node - "direct_response" >> self.executor_node
        self.strategy_node - "default" >> self.planner_node

        # Planning -> Execution
        self.planner_node - "planned" >> self.executor_node
        self.planner_node - "planning_failed" >> self.sync_node
        self.planner_node - "default" >> self.sync_node

        # === EXECUTION CYCLE WITH TERMINATION CONTROL ===

        # Executor -> Completion Checker (instead of direct to reflector)
        self.executor_node - "plan_completed" >> self.completion_checker
        self.executor_node - "continue_execution" >> self.completion_checker
        self.executor_node - "execution_error" >> self.reflector_node
        self.executor_node - "waiting" >> self.completion_checker
        self.executor_node - "execution_complete" >> self.completion_checker  # HINZUGEFÜGT
        self.executor_node - "default" >> self.completion_checker  # GEÄNDERT von sync_node

        # Completion Checker decides next action
        self.completion_checker - "truly_complete" >> self.sync_node  # TERMINATE
        self.completion_checker - "needs_reflection" >> self.reflector_node
        self.completion_checker - "force_terminate" >> self.sync_node  # HINZUGEFÜGT
        self.completion_checker - "continue_execution" >> self.executor_node
        self.completion_checker - "default" >> self.sync_node

        # Reflector actions (limited cycles)
        self.reflector_node - "continue" >> self.completion_checker  # Back to checker
        self.reflector_node - "adapted" >> self.completion_checker  # Back to checker
        self.reflector_node - "needs_replan" >> self.planner_node  # Restart planning
        self.reflector_node - "final_complete" >> self.sync_node
        self.reflector_node - "plan_halted" >> self.sync_node  # TERMINATE
        self.reflector_node - "default" >> self.sync_node

        super().__init__(start=self.strategy_node)


class ResponseGenerationFlow(AsyncFlowT):
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

class ContextAggregatorNode(AsyncNodeT):
    """Aggregiere alle relevanten Kontexte und Ergebnisse"""

    async def prep_async(self, shared):
        return {
            "results_store": shared.get("results_store", {}),
            "tasks": shared.get("tasks", {}),
            "current_plan": shared.get("current_plan"),
            "original_query": shared.get("current_query", ""),
            "conversation_history": shared.get("conversation_history", []),
            "world_model": shared.get("world_model", {}),
            "plan_adaptations": shared.get("plan_adaptations", 0)
        }

    async def exec_async(self, prep_res):
        """Aggregiere intelligente Kontextinformationen"""

        aggregated_context = {
            "original_query": prep_res["original_query"],
            "successful_results": {},
            "failed_attempts": {},
            "key_discoveries": [],
            "adaptation_summary": "",
            "confidence_scores": {}
        }

        # Sammle erfolgreiche Task-Ergebnisse
        for task_id, task in prep_res["tasks"].items():
            if task.status == "completed" and task.result:
                result_data = prep_res["results_store"].get(task_id, {})

                aggregated_context["successful_results"][task_id] = {
                    "task_description": task.description,
                    "task_type": task.type,
                    "result": task.result,
                    "metadata": getattr(task, 'metadata', {}),
                    "verification": result_data.get("verification", {})
                }

                # Extrahiere key discoveries
                if isinstance(task, ToolTask) and task.hypothesis:
                    verification = result_data.get("verification", {})
                    if verification.get("hypothesis_score", 0) > 0.7:
                        aggregated_context["key_discoveries"].append({
                            "discovery": f"Tool {task.tool_name} confirmed: {task.hypothesis}",
                            "confidence": verification.get("hypothesis_score", 0.0),
                            "result": task.result
                        })

        # Sammle fehlgeschlagene Versuche
        for task_id, task in prep_res["tasks"].items():
            if task.status == "failed":
                aggregated_context["failed_attempts"][task_id] = {
                    "description": task.description,
                    "error": task.error,
                    "retry_count": task.retry_count
                }

        # Plan adaptations summary
        if prep_res["plan_adaptations"] > 0:
            aggregated_context[
                "adaptation_summary"] = f"Plan was adapted {prep_res['plan_adaptations']} times to handle unexpected results."

        return aggregated_context

    async def post_async(self, shared, prep_res, exec_res):
        shared["aggregated_context"] = exec_res
        if exec_res["successful_results"] or exec_res["key_discoveries"]:
            return "context_ready"
        else:
            return "no_context"


class ResultSynthesizerNode(AsyncNodeT):
    """Synthetisiere finale Antwort aus allen Ergebnissen"""

    async def prep_async(self, shared):
        return {
            "aggregated_context": shared.get("aggregated_context", {}),
            "fast_llm_model": shared.get("fast_llm_model"),
            "complex_llm_model": shared.get("complex_llm_model")
        }

    async def exec_async(self, prep_res):
        if not LITELLM_AVAILABLE:
            return await self._fallback_synthesis(prep_res)

        context = prep_res["aggregated_context"]

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

            response = await litellm.acompletion(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )

            synthesized_response = response.choices[0].message.content

            return {
                "synthesized_response": synthesized_response,
                "synthesis_method": "llm",
                "model_used": model_to_use,
                "confidence": self._estimate_synthesis_confidence(context)
            }

        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return await self._fallback_synthesis(prep_res)

    def _format_successful_results(self, results: Dict) -> str:
        formatted = []
        for task_id, result_info in results.items():
            formatted.append(f"- {result_info['task_description']}: {str(result_info['result'])[:200]}...")
        return "\n".join(formatted) if formatted else "No successful results to report."

    def _format_key_discoveries(self, discoveries: List) -> str:
        formatted = []
        for discovery in discoveries:
            confidence = discovery.get('confidence', 0.0)
            formatted.append(f"- {discovery['discovery']} (Confidence: {confidence:.2f})")
        return "\n".join(formatted) if formatted else "No key discoveries."

    def _format_failed_attempts(self, failed: Dict) -> str:
        if not failed:
            return "No significant failures."
        formatted = [f"- {info['description']}: {info['error']}" for info in failed.values()]
        return "\n".join(formatted)

    async def _fallback_synthesis(self, prep_res) -> Dict:
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
            for task_id, result in list(context["successful_results"].items())[:2]:  # Top 2
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

    def _estimate_synthesis_confidence(self, context: Dict) -> float:
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


class ResponseFormatterNode(AsyncNodeT):
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


class ResponseQualityNode(AsyncNodeT):
    """Prüfe Qualität der generierten Antwort"""

    async def prep_async(self, shared):
        return {
            "formatted_response": shared.get("formatted_response", {}),
            "original_query": shared.get("current_query", ""),
            "fast_llm_model": shared.get("fast_llm_model")
        }

    async def exec_async(self, prep_res):
        response_data = prep_res["formatted_response"]
        response_text = response_data.get("formatted_response", "")
        original_query = prep_res["original_query"]

        # Heuristische Qualitätsprüfung
        quality_score = self._heuristic_quality_check(response_text, original_query)

        # LLM-basierte Qualitätsprüfung falls verfügbar
        if LITELLM_AVAILABLE and len(response_text) > 50:
            llm_quality = await self._llm_quality_check(response_text, original_query, prep_res)
            # Kombiniere beide Scores
            quality_score = (quality_score + llm_quality) / 2

        return {
            "quality_score": quality_score,
            "quality_assessment": self._score_to_assessment(quality_score),
            "suggestions": self._generate_quality_suggestions(quality_score, response_text)
        }

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

    async def _llm_quality_check(self, response: str, query: str, prep_res: Dict) -> float:
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

            llm_response = await litellm.acompletion(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )

            score_text = llm_response.choices[0].message.content.strip()
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

    def _generate_quality_suggestions(self, score: float, response: str) -> List[str]:
        suggestions = []

        if score < 0.5:
            suggestions.append("Response may need more relevant information")
        if len(response) < 100:
            suggestions.append("Response could be more detailed")
        if score < 0.3:
            suggestions.append("Consider regenerating with different approach")

        return suggestions

    async def post_async(self, shared, prep_res, exec_res):
        shared["quality_assessment"] = exec_res
        return exec_res["quality_assessment"]


class ResponseFinalProcessorNode(AsyncNodeT):
    """Finale Verarbeitung mit Persona-System"""

    async def prep_async(self, shared):
        return {
            "formatted_response": shared.get("formatted_response", {}),
            "quality_assessment": shared.get("quality_assessment", {}),
            "conversation_history": shared.get("conversation_history", []),
            "persona": shared.get("persona_config"),
            "fast_llm_model": shared.get("fast_llm_model"),
            "use_fast_response": shared.get("use_fast_response", True)
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

    async def _apply_persona_style(self, response: str, prep_res: Dict) -> str:
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

            if prep_res.get("use_fast_response", True):
                response = await litellm.acompletion(
                    model=model_to_use,
                    messages=[{"role": "user", "content": style_prompt}],
                    temperature=0.5,
                    max_tokens=max_tokens
                )
            else:
                response = await litellm.acompletion(
                    model=model_to_use,
                    messages=[{"role": "user", "content": style_prompt}],
                    temperature=0.6,
                    max_tokens=max_tokens + 200
                )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"Persona styling failed: {e}")
            return response

    async def post_async(self, shared, prep_res, exec_res):
        shared["current_response"] = exec_res["final_response"]
        shared["response_metadata"] = exec_res["metadata"]
        return "response_ready"

# ===== Foramt Helper =====
class VariableManager:
    """Variable management system using world_model"""

    def __init__(self, world_model: Dict):
        self.world_model = world_model

    def get(self, key: str, default=None):
        """Get variable from world_model"""
        return self.world_model.get(key, default)

    def set(self, key: str, value):
        """Set variable in world_model"""
        self.world_model[key] = value

    def format_text(self, text: str) -> str:
        """Format text with variables using {{ variable_name }} syntax"""
        import re

        def replace_var(match):
            var_path = match.group(1).strip()
            try:
                # Handle nested access like results.task_id.data
                if '.' in var_path:
                    parts = var_path.split('.')
                    value = self.world_model
                    for part in parts:
                        if isinstance(value, dict):
                            value = value.get(part, {})
                        else:
                            return f"{{{{ {var_path} }}}}"  # Return original if not found
                else:
                    value = self.world_model.get(var_path, f"{{{{ {var_path} }}}}")

                return str(value) if value is not None else ""
            except:
                return f"{{{{ {var_path} }}}}"

        # Replace {{ variable }} patterns
        formatted = re.sub(r'\{\{\s*([^}]+)\s*\}\}', replace_var, text)
        return formatted

    def get_available_variables(self) -> Dict[str, str]:
        """Get list of available variables with descriptions"""
        variables = {}
        for key, value in self.world_model.items():
            if isinstance(value, str):
                preview = value[:50] + "..." if len(value) > 50 else value
            else:
                preview = str(type(value).__name__)
            variables[key] = preview
        return variables

# ===== MAIN AGENT CLASS =====
class FlowAgent:
    """Production-ready agent system built on PocketFlow"""
    def __init__(
        self,
        amd: AgentModelData,
        world_model: Dict[str, Any] = None,
        verbose: bool = False,
        enable_pause_resume: bool = True,
        checkpoint_interval: int = 300,  # 5 minutes
        max_parallel_tasks: int = 3,
        **kwargs
    ):
        self.amd = amd
        self.world_model = world_model or {}
        self.verbose = verbose
        self.enable_pause_resume = enable_pause_resume
        self.checkpoint_interval = checkpoint_interval
        self.max_parallel_tasks = max_parallel_tasks

        # Core state
        self.shared = {
            "world_model": self.world_model.copy(),
            "tasks": {},
            "task_plans": {},
            "system_status": "idle",
            "session_data": {},
            "performance_metrics": {},
            "conversation_history": [],
            "available_tools": []
        }
        self.variable_manager = VariableManager(self.shared["world_model"])

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
        self.a2a_server: Optional[A2AServer] = None
        self.mcp_server: Optional[FastMCP] = None

        # Enhanced tool registry
        self._tool_registry = {}
        self._tool_capabilities = {}
        self._tool_analysis_cache = {}

        # Tool analysis file path
        self.tool_analysis_file = self._get_tool_analysis_path()

        logger.info(f"FlowAgent initialized: {amd.name}")

    async def a_run(
        self,
        query: str,
        session_id: str = "default",
        user_id: str = None,
        stream_callback: Optional[Callable] = None,
        **kwargs
    ) -> str:
        """Main entry point for agent execution"""

        try:
            # Initialize with tool awareness
            await self._initialize_context_awareness()

            # Prepare execution context
            self.shared.update({
                "current_query": query,
                "session_id": session_id,
                "user_id": user_id,
                "stream_callback": stream_callback
            })

            # Set LLM models in shared context
            self.shared['fast_llm_model'] = self.amd.fast_llm_model
            self.shared['complex_llm_model'] = self.amd.complex_llm_model
            self.shared['persona_config'] = self.amd.persona
            self.shared['use_fast_response'] = self.amd.use_fast_response
            self.shared['variable_manager'] = self.variable_manager
            # Add to conversation history
            self.shared["conversation_history"].append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            })

            # Set system status
            self.shared["system_status"] = "running"
            self.is_running = True

            # Execute main orchestration flow
            result = await self._orchestrate_execution()

            # Add response to history
            self.shared["conversation_history"].append({
                "role": "assistant",
                "content": result,
                "timestamp": datetime.now().isoformat()
            })

            # Checkpoint if needed
            if self.enable_pause_resume:
                await self._maybe_checkpoint()

            return result

        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            error_response = f"I encountered an error: {str(e)}"

            self.shared["conversation_history"].append({
                "role": "assistant",
                "content": error_response,
                "timestamp": datetime.now().isoformat()
            })

            return error_response

        finally:
            self.shared["system_status"] = "idle"
            self.is_running = False

    def set_persona(self, name: str, style: str = "professional", tone: str = "friendly",
                    personality_traits: List[str] = None, apply_method: str = "system_prompt",
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

        logger.info(f"Persona set: {name} ({style}, {tone}) - Method: {apply_method}, Level: {integration_level}")

    def configure_persona_integration(self, apply_method: str = "system_prompt", integration_level: str = "light"):
        """Configure how persona is applied"""
        if self.amd.persona:
            self.amd.persona.apply_method = apply_method
            self.amd.persona.integration_level = integration_level
            logger.info(f"Persona integration updated: {apply_method}, {integration_level}")
        else:
            logger.warning("No persona configured to update")

    def get_available_variables(self) -> Dict[str, str]:
        """Get available variables for dynamic formatting"""
        return self.variable_manager.get_available_variables()

    async def _orchestrate_execution(self) -> str:
        """Vollständig adaptive Orchestrierung mit separaten Phasen"""

        self.shared["agent_instance"] = self

        # === PHASE 1: TASK MANAGEMENT CYCLE ===
        logger.info("Starting adaptive task management cycle")

        # Führe Task-Management-Flow aus (adaptiv mit Reflexion)
        task_management_result = await self.task_flow.run_async(self.shared)

        if self.shared.get("plan_halted"):
            error_response = f"Task execution was halted: {self.shared.get('halt_reason', 'Unknown reason')}"
            self.shared["current_response"] = error_response
            return error_response

        # === PHASE 2: RESPONSE GENERATION ===
        logger.info("Starting response generation flow")

        # Führe Response-Generation-Flow aus
        response_result = await self.response_flow.run_async(self.shared)

        # === PHASE 3: FINAL RESULT ===
        final_response = self.shared.get("current_response", "Task completed successfully.")

        # Logge Statistiken
        self._log_execution_stats()

        return final_response

    def _log_execution_stats(self):
        """Logge Ausführungsstatistiken"""
        tasks = self.shared.get("tasks", {})
        adaptations = self.shared.get("plan_adaptations", 0)

        completed_tasks = sum(1 for t in tasks.values() if t.status == "completed")
        failed_tasks = sum(1 for t in tasks.values() if t.status == "failed")

        logger.info(
            f"Execution complete - Tasks: {completed_tasks} completed, {failed_tasks} failed, {adaptations} adaptations")

    async def _initialize_context_awareness(self):
        """Initialize system self-awareness"""

        # Ensure tool capabilities are loaded/analyzed
        for tool_name in self.shared["available_tools"]:
            if tool_name not in self._tool_capabilities:
                tool_info = self._tool_registry.get(tool_name, {})
                description = tool_info.get("description", "No description")
                await self._analyze_tool_capabilities(tool_name, description)

        # Set system context with capabilities
        self.shared["system_context"] = {
            "capabilities_summary": self._build_capabilities_summary(),
            "tool_count": len(self.shared["available_tools"]),
            "analysis_loaded": len(self._tool_capabilities),
            "intelligence_level": "high" if self._tool_capabilities else "basic"
        }

        logger.info(f"Context awareness initialized: {len(self._tool_capabilities)} tools analyzed")

    def _build_capabilities_summary(self) -> str:
        """Build summary of agent capabilities"""

        if not self._tool_capabilities:
            return "Basic LLM capabilities only"

        summaries = []
        for tool_name, cap in self._tool_capabilities.items():
            primary = cap.get('primary_function', 'Unknown function')
            summaries.append(f"{tool_name}: {primary}")

        return f"Enhanced capabilities: {'; '.join(summaries)}"

    # Neue Hilfsmethoden für erweiterte Funktionalität

    async def get_task_execution_summary(self) -> Dict[str, Any]:
        """Erhalte detaillierte Zusammenfassung der Task-Ausführung"""
        tasks = self.shared.get("tasks", {})
        results_store = self.shared.get("results_store", {})

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
            response = await litellm.acompletion(
                model=self.amd.complex_llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=800
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Could not generate reasoning explanation: {e}"

    def _format_tasks_for_explanation(self, tasks: List[Dict]) -> str:
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

        logger.info("Agent execution paused")
        return True

    async def resume(self) -> bool:
        """Resume agent execution"""
        if not self.is_paused:
            return False

        self.is_paused = False
        self.shared["system_status"] = "running"

        logger.info("Agent execution resumed")
        return True

    async def _create_checkpoint(self) -> AgentCheckpoint:
        """Create a checkpoint of current state"""
        return AgentCheckpoint(
            timestamp=datetime.now(),
            agent_state={
                "is_running": self.is_running,
                "is_paused": self.is_paused,
                "amd": self.amd.model_dump_json(indent=2) if hasattr(self.amd, 'model_dump_json') else str(self.amd)
            },
            task_state={
                task_id: asdict(task) for task_id, task in self.shared.get("tasks", {}).items()
            },
            world_model=self.shared["world_model"].copy(),
            active_flows=["task_flow", "response_flow"],  # Track active flows
            metadata={
                "session_id": self.shared.get("session_id", "default"),
                "last_query": self.shared.get("current_query", "")
            }
        )

    async def _save_checkpoint(self, checkpoint: AgentCheckpoint, filepath: str = None):
        """Save checkpoint to file"""
        from toolboxv2 import get_app
        folder = str(get_app().data_dir) + '/Agents/checkpoint/' + self.amd.name
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        if not filepath:
            timestamp = checkpoint.timestamp.strftime("%Y%m%d_%H%M%S")
            filepath = f"agent_checkpoint_{timestamp}.pkl"
        filepath = folder + '/' + filepath
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint, f)

            self.last_checkpoint = checkpoint.timestamp
            logger.info(f"Checkpoint saved: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    async def load_checkpoint(self, filepath: str) -> bool:
        """Load checkpoint from file"""
        try:
            with open(filepath, 'rb') as f:
                checkpoint: AgentCheckpoint = pickle.load(f)

            # Restore state
            self.shared["world_model"] = checkpoint.world_model
            self.shared["tasks"] = {
                task_id: Task(**task_data)
                for task_id, task_data in checkpoint.task_state.items()
            }

            # Restore agent state
            agent_state = checkpoint.agent_state
            self.is_running = agent_state.get("is_running", False)
            self.is_paused = agent_state.get("is_paused", False)

            self.last_checkpoint = checkpoint.timestamp
            logger.info(f"Checkpoint loaded: {filepath}")

            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    async def _maybe_checkpoint(self):
        """Create checkpoint if interval has passed"""
        now = datetime.now()
        if (not self.last_checkpoint or
            (now - self.last_checkpoint).seconds >= self.checkpoint_interval):

            checkpoint = await self._create_checkpoint()
            await self._save_checkpoint(checkpoint)

    # ===== TOOL AND NODE MANAGEMENT =====
    def _get_tool_analysis_path(self) -> str:
        """Get path for tool analysis cache"""
        from toolboxv2 import get_app
        folder = str(get_app().data_dir) + '/Agents/capabilities/' + self.amd.name
        os.makedirs(folder, exist_ok=True)
        return folder + '/tool_capabilities.json'

    async def add_tool(self, tool_func: Callable, name: str = None, description: str = None):
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
            "description": tool_description
        }

        # Add to available tools list
        if tool_name not in self.shared["available_tools"]:
            self.shared["available_tools"].append(tool_name)

        # Intelligent tool analysis
        await self._analyze_tool_capabilities(tool_name, tool_description)

        logger.info(f"Tool added with analysis: {tool_name}")

    async def _analyze_tool_capabilities(self, tool_name: str, description: str):
        """Analyze tool capabilities with LLM for smart usage"""

        # Try to load existing analysis
        existing_analysis = await self._load_tool_analysis()

        if tool_name in existing_analysis:
            self._tool_capabilities[tool_name] = existing_analysis[tool_name]
            logger.info(f"Loaded cached analysis for {tool_name}")
            return

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

Respond in YAML format:
primary_function: Main purpose of the tool
use_cases:
  - Specific use case 1
  - Specific use case 2
trigger_phrases:
  - Trigger phrase 1
  - Trigger phrase 2
indirect_connections:
  - Non-obvious connection 1
  - Non-obvious connection 2
complexity_scenarios:
  - Complex scenario 1
  - Complex scenario 2
user_intent_categories:
  - Category 1
  - Category 2
confidence_triggers:
  phrase: confidence_score
tool_complexity: low/medium/high
"""

        try:
            response = await litellm.acompletion(
                model=self.amd.complex_llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )

            content = response.choices[0].message.content.strip()

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

            logger.info(f"Generated intelligent analysis for {tool_name}")

        except Exception as e:
            logger.error(f"Tool analysis failed for {tool_name}: {e}")
            # Fallback
            self._tool_capabilities[tool_name] = {
                "primary_function": description,
                "use_cases": [description],
                "trigger_phrases": [tool_name.lower().replace('_', ' ')],
                "tool_complexity": "medium"
            }

    async def _load_tool_analysis(self) -> Dict[str, Any]:
        """Load tool analysis from cache"""
        try:
            if os.path.exists(self.tool_analysis_file):
                with open(self.tool_analysis_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load tool analysis: {e}")
        return {}

    async def _save_tool_analysis(self):
        """Save tool analysis to cache"""
        try:
            with open(self.tool_analysis_file, 'w') as f:
                json.dump(self._tool_capabilities, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save tool analysis: {e}")

    def add_custom_flow(self, flow: AsyncFlowT, name: str):
        """Add a custom flow for dynamic execution"""
        self.add_tool(flow.run_async, name=name, description=f"Custom flow: {flow.__class__.__name__}")
        logger.info(f"Custom node added: {name}")

    def get_tool_by_name(self, tool_name: str) -> Callable | None:
        """Get tool function by name"""
        return self._tool_registry.get(tool_name, {}).get("function")

    async def arun_function(self, function_name: str, *args, **kwargs) -> Any:
        """
        Asynchronously finds a function by its string name, executes it with
        the given arguments, and returns the result.
        """
        print(f"Attempting to run function: {function_name}")
        logger.info(f"Attempting to run function: {function_name}")
        target_function = self.get_tool_by_name(function_name)

        if not target_function:
            raise ValueError(f"Function '{function_name}' not found in the agent's registered tools.")

        if asyncio.iscoroutinefunction(target_function):
            return await target_function(*args, **kwargs)
        else:
            # If the function is not async, run it in a thread pool
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: target_function(*args, **kwargs))

    async def execute_custom_node(self, node_name: str, **kwargs) -> Any:
        """Execute a custom node dynamically"""
        if not hasattr(self, '_node_registry') or node_name not in self._node_registry:
            raise ValueError(f"Node '{node_name}' not found")

        node = self._node_registry[node_name]

        # Create temporary shared state with kwargs
        temp_shared = self.shared.copy()
        temp_shared.update(kwargs)

        # Execute the node
        result = await node.run_async(temp_shared)

        # Merge back any changes
        self.shared.update(temp_shared)

        return result

    # ===== SERVER SETUP =====

    def setup_a2a_server(self, host: str = "0.0.0.0", port: int = 5000, **kwargs):
        """Setup A2A server for bidirectional communication"""
        if not A2A_AVAILABLE:
            logger.warning("A2A not available, cannot setup server")
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

            logger.info(f"A2A server setup on {host}:{port}")

        except Exception as e:
            logger.error(f"Failed to setup A2A server: {e}")

    def setup_mcp_server(self, host: str = "0.0.0.0", port: int = 8000, name: str = None, **kwargs):
        """Setup MCP server"""
        if not MCP_AVAILABLE:
            logger.warning("MCP not available, cannot setup server")
            return

        try:
            server_name = name or f"{self.amd.name}_MCP"
            self.mcp_server = FastMCP(server_name)

            # Register agent as MCP tool
            @self.mcp_server.tool()
            async def agent_run(query: str, session_id: str = "mcp_session") -> str:
                """Execute agent with given query"""
                return await self.a_run(query, session_id=session_id)

            logger.info(f"MCP server setup: {server_name}")

        except Exception as e:
            logger.error(f"Failed to setup MCP server: {e}")

    # ===== LIFECYCLE MANAGEMENT =====

    async def start_servers(self):
        """Start all configured servers"""
        tasks = []

        if self.a2a_server:
            tasks.append(asyncio.create_task(self.a2a_server.start()))

        if self.mcp_server:
            tasks.append(asyncio.create_task(self.mcp_server.run()))

        if tasks:
            logger.info(f"Starting {len(tasks)} servers...")
            await asyncio.gather(*tasks, return_exceptions=True)

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

        logger.info("Agent shutdown complete")

    @property
    def total_cost(self) -> float:
        """Get total cost if budget manager available"""
        if hasattr(self.amd, 'budget_manager') and self.amd.budget_manager:
            return getattr(self.amd.budget_manager, 'total_cost', 0.0)
        return 0.0

    @property
    def status(self, full=True) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            "name": self.amd.name,
            "status": self.shared.get("system_status", "idle"),
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "total_tasks": len(self.shared.get("tasks", {})),
            "active_tasks": len([t for t in self.shared.get("tasks", {}).values() if t.status == "running"]),
            "completed_tasks": len([t for t in self.shared.get("tasks", {}).values() if t.status == "completed"]),
            "total_cost": self.total_cost,
            "last_checkpoint": self.last_checkpoint.isoformat() if self.last_checkpoint else None,
            "conversation_turns": len(self.shared.get("conversation_history", [])),
            "world_model_size": len(self.shared.get("world_model", {})),
            "available_tools": len(self.shared.get("available_tools", [])),
            "servers": {
                "a2a": self.a2a_server is not None,
                "mcp": self.mcp_server is not None
            },
            "full_state": self.shared if full else None

        }

if __name__ == "__main__":
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

        def get_user_name():
            return "Markin"

        await agent.add_tool(get_user_name, "get_user_name", "Get the user's name")
        print("online")
        response = await agent.a_run("Hello. what is my name?")
        print(f"Response: {response}")
        print(f"Status: {agent.status}")

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
            print(f"Status: {agent.status}")

        await agent.close()

    asyncio.run(_agent())

    x = [
            {'id': 'respond_to_name_query',
             'type': 'LLMTask',
             'description': "Generate response to user's name question",
             'priority': 1,
             'dependencies': [],
             'prompt_template': "The user asked: 'What is my name?' Please provide a helpful and polite response.",
             'llm_config': {'model_preference': 'fast', 'temperature': 0.7}}
    ]
