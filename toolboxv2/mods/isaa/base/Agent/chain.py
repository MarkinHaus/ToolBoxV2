import types

import asyncio
import random
from enum import Enum
from typing import Any, Union, List, Dict, Tuple, Optional
from pydantic import BaseModel
import copy

from toolboxv2.mods.isaa.base.Agent.types import NodeStatus, ProgressEvent


class ChainRunType(Enum):
    auto = "auto"
    a_run = "a_run"
    format_class = "format_class"

class CF:
    """Chain Format - handles formatting between agents"""

    def __init__(self, format_class: type[BaseModel]):
        self.format_class = format_class
        self.extract_key = None
        self.extract_multiple = False
        self.parallel_count = None

    def __sub__(self, key):
        """Implements - operator for key extraction"""
        new_cf = copy.copy(self)
        if isinstance(key, str):
            if key == '*':
                new_cf.extract_key = '*'  # Extract all fields
            elif key.startswith('*'):
                new_cf.extract_key = key[1:]  # Remove * prefix
            elif '[n]' in key:
                new_cf.extract_key = key.replace('[n]', '')
                new_cf.parallel_count = 'n'  # Will be determined at runtime
            else:
                new_cf.extract_key = key
        elif isinstance(key, tuple):
            new_cf.extract_key = key
        return new_cf

    def __rshift__(self, other):
        """Implements >> operator after CF"""
        return Chain._create_chain([self, other])


class IS:
    """Conditional check"""

    def __init__(self, key: str, expected_value: Any):
        self.key = key
        self.expected_value = expected_value

    def __rshift__(self, other):
        """Implements >> operator after IS"""
        return ConditionalChain(self, other)


class ParallelChain:
    """Handles parallel execution of agents"""

    def __init__(self, agents: List[Union['FlowAgent', 'Chain']]):
        self.agents = agents

    async def a_run(self, query, **kwargs):
        """Run all agents in parallel"""
        tasks = []

        for agent in self.agents:
            if hasattr(agent, 'a_run'):
                tasks.append(agent.a_run(query, **kwargs))
            else:
                tasks.append(agent.run(query))  # For other chain types

        results = await asyncio.gather(*tasks)
        return self._combine_results(results)

    def _combine_results(self, results):
        """Intelligently combine parallel results"""
        if len(results) == 1:
            return results[0]

        # Simple combination - could be made more sophisticated
        if all(isinstance(r, str) for r in results):
            return " | ".join(results)
        elif all(isinstance(r, dict) for r in results):
            combined = {}
            for i, result in enumerate(results):
                combined[f"agent_{i}"] = result
            return combined
        else:
            return results

    def __rshift__(self, other):
        """Implements >> operator"""
        return Chain._create_chain([self, other])

    def __or__(self, other):
        """Implements | operator for error handling"""
        return ErrorHandlingChain(self, other)


    def __call__(self, *args, **kwargs):
        return self._Runner(self, args, kwargs)

    class _Runner:
        def __init__(self, parent, args, kwargs):
            self.parent = parent
            self.args = args
            self.kwargs = kwargs

        def __call__(self):
            # Normaler Aufruf → run
            return self.parent.run(*self.args, **self.kwargs)

        def __await__(self):
            # Await → arun
            return self.parent.a_run(*self.args, **self.kwargs).__await__()



class ConditionalChain:
    """Handles conditional execution"""

    def __init__(self, condition, true_branch, false_branch=None):
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.kwargs_for_agent = {}

    def __mod__(self, other):
        """Implements % operator for false branch"""
        return ConditionalChain(self.condition, self.true_branch, other)

    async def a_run(self, query_or_data, **kwargs):
        """Execute based on condition"""
        self.kwargs_for_agent = kwargs
        if isinstance(self.condition, IS):
            # Check condition
            if isinstance(query_or_data, dict) and self.condition.key in query_or_data:
                condition_met = query_or_data[self.condition.key] == self.condition.expected_value
            else:
                condition_met = False

            if condition_met:
                return await self._run_branch(self.true_branch, query_or_data)
            elif self.false_branch:
                return await self._run_branch(self.false_branch, query_or_data)
            else:
                return query_or_data
        else:
            # Direct conditional execution
            return await self._run_branch(self.true_branch, query_or_data)

    async def _run_branch(self, branch, data):
        if hasattr(branch, 'a_run'):
            return await branch.a_run(data, **self.kwargs_for_agent)
        elif hasattr(branch, 'run'):
            return await branch.run(data, **self.kwargs_for_agent)
        else:
            return data


    def __call__(self, *args, **kwargs):
        return self._Runner(self, args, kwargs)

    class _Runner:
        def __init__(self, parent, args, kwargs):
            self.parent = parent
            self.args = args
            self.kwargs = kwargs

        def __call__(self):
            # Normaler Aufruf → run
            return self.parent.run(*self.args, **self.kwargs)

        def __await__(self):
            # Await → arun
            return self.parent.a_run(*self.args, **self.kwargs).__await__()



class ErrorHandlingChain:
    """Handles error cases with fallback"""

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback

    async def a_run(self, query, **kwargs):
        try:
            return await self.primary.a_run(query, **kwargs)
        except Exception as e:
            print(f"Primary chain failed with {e}, trying fallback")
            if hasattr(self.fallback, 'a_run'):
                return await self.fallback.a_run(query, **kwargs)
            else:
                return await self.fallback.run(query, **kwargs)


    def __call__(self, *args, **kwargs):
        return self._Runner(self, args, kwargs)

    class _Runner:
        def __init__(self, parent, args, kwargs):
            self.parent = parent
            self.args = args
            self.kwargs = kwargs

        def __call__(self):
            # Normaler Aufruf → run
            return self.parent.run(*self.args, **self.kwargs)

        def __await__(self):
            # Await → arun
            return self.parent.a_run(*self.args, **self.kwargs).__await__()



class Chain:
    def __init__(self, agent: 'FlowAgent' = None):
        if agent:
            self.agent = agent
            self.tasks = [agent]
        else:
            self.tasks = []
        self._name = "chain"
        self.progress_tracker = None
        self.description = "A chain of tasks"
        self.return_key = "chain"
        self.use = "chain"

        self.amd = lambda: None
        self.amd.name = "chain"

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, value):
        self._name = value
        self.amd.name = value

    @classmethod
    def _create_chain(cls, components):
        """Create a chain from a list of components"""
        chain = cls()
        chain.tasks = components
        return chain

    def _extract_data(self, data, cf: CF):
        """Extract data based on CF configuration with parallel support"""
        if not isinstance(data, dict):
            return data

        if cf.extract_key == '*':
            return data  # Return all
        elif isinstance(cf.extract_key, tuple):
            return {k: data.get(k) for k in cf.extract_key if k in data}
        elif cf.extract_key in data:
            extracted = data[cf.extract_key]

            # Handle parallel extraction
            if cf.parallel_count == 'n' and isinstance(extracted, (list, tuple)):
                return extracted  # Return list for parallel processing
            else:
                return extracted
        else:
            return data

    async def a_run(self, query: Union[str, BaseModel], task_id=None, **kwargs):
        """Execute the chain asynchronously with auto-parallel support"""
        import time

        self.kwargs_for_agent = kwargs
        current_data = query
        node_name = self.name or "Unnamed Chain"

        if self.progress_tracker:
            await self.progress_tracker.emit_event(ProgressEvent(
                event_type="chain_start",
                timestamp=time.time(),
                status=NodeStatus.RUNNING,
                node_name=node_name,
                task_id=task_id,
                metadata={"input": str(query)}
            ))

        for i, task in enumerate(self.tasks):
            task_name = getattr(task, 'name', type(task).__name__)

            if self.progress_tracker:
                await self.progress_tracker.emit_event(ProgressEvent(
                    event_type="task_start",
                    timestamp=time.time(),
                    status=NodeStatus.RUNNING,
                    node_name=task_name,
                    task_id=task_id,
                ))

            try:
                # Handle CF with parallel extraction
                if hasattr(task, 'format_class') and hasattr(task, 'parallel_count') and task.parallel_count == 'n':
                    extracted_data = self._extract_data(current_data, task)

                    # If we have a list and there's a next task, run it in parallel
                    if isinstance(extracted_data, (list, tuple)) and i + 1 < len(self.tasks):
                        next_task = self.tasks[i + 1]

                        if hasattr(next_task, 'a_run') or hasattr(next_task, 'run'):
                            # Create parallel execution
                            parallel_tasks = []
                            for item in extracted_data:
                                if hasattr(next_task, 'a_run'):
                                    parallel_tasks.append(next_task.a_run(str(item), **self.kwargs_for_agent))
                                else:
                                    parallel_tasks.append(next_task.run(str(item)))

                            # Execute in parallel
                            parallel_results = await asyncio.gather(*parallel_tasks)
                            current_data = parallel_results

                            # Skip the next task since we already executed it
                            continue
                        else:
                            current_data = extracted_data
                    else:
                        current_data = extracted_data

                elif isinstance(task, (ParallelChain, ConditionalChain, ErrorHandlingChain)):
                    current_data = await task.a_run(current_data, **self.kwargs_for_agent)
                elif hasattr(task, 'format_class'):
                    # Regular CF processing
                    current_data = self._extract_data(current_data, task)
                elif hasattr(task, 'a_run'):
                    if isinstance(current_data, BaseModel):
                        current_data = await task.a_run(str(current_data.model_dump()), **self.kwargs_for_agent)
                    else:
                        current_data = await task.a_run(str(current_data), **self.kwargs_for_agent)

                if self.progress_tracker:
                    await self.progress_tracker.emit_event(ProgressEvent(
                        event_type="task_end",
                        timestamp=time.time(),
                        status=NodeStatus.COMPLETED,
                        node_name=task_name,
                        task_id=task_id,
                    ))
            except Exception as e:
                if self.progress_tracker:
                    await self.progress_tracker.emit_event(ProgressEvent(
                        event_type="task_error",
                        timestamp=time.time(),
                        status=NodeStatus.FAILED,
                        node_name=task_name,
                        task_id=task_id,
                        metadata={"error": str(e)}
                    ))
                raise e

        if self.progress_tracker:
            await self.progress_tracker.emit_event(ProgressEvent(
                event_type="chain_end",
                timestamp=time.time(),
                status=NodeStatus.COMPLETED,
                node_name=node_name,
                task_id=task_id,
                metadata={"output": str(current_data)}
            ))

        return current_data

    def run(self, query: Union[str, BaseModel], use="auto", **kwargs):
        """Synchronous wrapper"""
        return asyncio.run(self.a_run(query, use, **kwargs))

    def __rshift__(self, other):
        """Implements >> operator for extending chains"""
        new_tasks = self.tasks + [other]
        return Chain._create_chain(new_tasks)

    def __add__(self, other):
        """Implements + operator for parallel execution"""
        return ParallelChain([self, other])

    def __and__(self, other):
        """Implements & operator for parallel execution"""
        return ParallelChain([self, other])

    def __mod__(self, other):
        """Implements % operator for conditional branching"""
        return ConditionalChain(None, self, other)

    def __or__(self, other):
        """Implements | operator for error handling"""
        return ErrorHandlingChain(self, other)


    def set_progress_callback(self, progress_tracker):
        """
        Sets the progress callback for every agent in the chain.
        """
        for task in self.tasks:
            if hasattr(task, 'set_progress_callback'):
                task.set_progress_callback(progress_tracker)
            else:
                ok = False
                for typ in (ParallelChain, ConditionalChain, ErrorHandlingChain):
                    if isinstance(task, typ):
                        ok = True
                if not ok:
                    continue
                if hasattr(task, 'agents'): # ParallelChain
                    for agent in task.agents:
                         if hasattr(agent, 'set_progress_callback'):
                            agent.set_progress_callback(progress_tracker)
                if hasattr(task, 'true_branch'): # ConditionalChain
                    if hasattr(task.true_branch, 'set_progress_callback'):
                        task.true_branch.set_progress_callback(progress_tracker)
                if hasattr(task, 'false_branch') and task.false_branch: # ConditionalChain
                    if hasattr(task.false_branch, 'set_progress_callback'):
                        task.false_branch.set_progress_callback(progress_tracker)
                if hasattr(task, 'primary'): # ErrorHandlingChain
                     if hasattr(task.primary, 'set_progress_callback'):
                        task.primary.set_progress_callback(progress_tracker)
                if hasattr(task, 'fallback'): # ErrorHandlingChain
                     if hasattr(task.fallback, 'set_progress_callback'):
                        task.fallback.set_progress_callback(progress_tracker)

    def __call__(self, *args, **kwargs):
        return self._Runner(self, args, kwargs)

    class _Runner:
        def __init__(self, parent, args, kwargs):
            self.parent = parent
            self.args = args
            self.kwargs = kwargs

        def __call__(self):
            # Normaler Aufruf → run
            return self.parent.run(*self.args, **self.kwargs)

        def __await__(self):
            # Await → arun
            return self.parent.a_run(*self.args, **self.kwargs).__await__()


def chain_to_graph(self) -> Dict[str, Any]:
    """Convert chain to hierarchical structure with complete component detection."""

    def process_component(comp, depth=0, visited=None):
        if visited is None:
            visited = set()

        # Prevent infinite recursion
        comp_id = id(comp)
        if comp_id in visited or depth > 20:
            return {"type": "Circular", "display": "[CIRCULAR_REF]", "depth": depth}
        visited.add(comp_id)

        if comp is None:
            return {"type": "Error", "display": "[NULL]", "depth": depth}

        try:
            # Agent detection
            if hasattr(comp, 'amd') and comp.amd:
                return {
                    "type": "Agent",
                    "display": f"[Agent] {comp.amd.name}",
                    "name": comp.amd.name,
                    "depth": depth
                }

            # Format detection (CF) with parallel detection
            if hasattr(comp, 'format_class'):
                name = comp.format_class.__name__
                display = f"[Format] {name}"

                result = {
                    "type": "Format",
                    "display": display,
                    "format_class": name,
                    "extract_key": getattr(comp, 'extract_key', None),
                    "depth": depth,
                    "creates_parallel": False
                }

                # Extract key visualization
                if hasattr(comp, 'extract_key') and comp.extract_key:
                    key = comp.extract_key
                    if key == '*':
                        display += " \033[90m(*all*)\033[0m"
                    elif isinstance(key, str):
                        display += f" \033[90m(→{key})\033[0m"
                    elif isinstance(key, tuple):
                        display += f" \033[90m(→{','.join(key)})\033[0m"

                # Parallel detection
                if hasattr(comp, 'parallel_count') and comp.parallel_count == 'n':
                    display += " \033[95m[PARALLEL]\033[0m"
                    result["creates_parallel"] = True
                    result["parallel_type"] = "auto_n"

                result["display"] = display
                return result

            # Condition detection (IS)
            if hasattr(comp, 'key') and hasattr(comp, 'expected_value'):
                return {
                    "type": "Condition",
                    "display": f"[Condition] IS {comp.key}=='{comp.expected_value}'",
                    "condition_key": comp.key,
                    "expected_value": comp.expected_value,
                    "depth": depth
                }

            # Parallel chain detection
            if hasattr(comp, 'agents') and isinstance(comp.agents, (list, tuple)):
                branches = []
                for i, agent in enumerate(comp.agents):
                    if agent:
                        branch_data = process_component(agent, depth + 1, visited.copy())
                        branch_data["branch_id"] = i
                        branches.append(branch_data)

                return {
                    "type": "Parallel",
                    "display": f"[Parallel] {len(branches)} branches",
                    "branches": branches,
                    "branch_count": len(branches),
                    "execution_type": "concurrent",
                    "depth": depth
                }

            # Conditional chain detection
            if hasattr(comp, 'condition') and hasattr(comp, 'true_branch'):
                condition_data = process_component(comp.condition, depth + 1,
                                                   visited.copy()) if comp.condition else None
                true_data = process_component(comp.true_branch, depth + 1, visited.copy()) if comp.true_branch else None
                false_data = None

                if hasattr(comp, 'false_branch') and comp.false_branch:
                    false_data = process_component(comp.false_branch, depth + 1, visited.copy())

                return {
                    "type": "Conditional",
                    "display": "[Conditional] Branch Logic",
                    "condition": condition_data,
                    "true_branch": true_data,
                    "false_branch": false_data,
                    "has_false_branch": false_data is not None,
                    "depth": depth
                }

            # Error handling detection
            if hasattr(comp, 'primary') and hasattr(comp, 'fallback'):
                primary_data = process_component(comp.primary, depth + 1, visited.copy()) if comp.primary else None
                fallback_data = process_component(comp.fallback, depth + 1, visited.copy()) if comp.fallback else None

                return {
                    "type": "ErrorHandling",
                    "display": "[Try-Catch] Error Handler",
                    "primary": primary_data,
                    "fallback": fallback_data,
                    "has_fallback": fallback_data is not None,
                    "depth": depth
                }

            # Regular chain detection
            if hasattr(comp, 'tasks') and isinstance(comp.tasks, (list, tuple)):
                tasks = []
                for i, task in enumerate(comp.tasks):
                    if task is not None:
                        task_data = process_component(task, depth + 1, visited.copy())
                        task_data["task_id"] = i
                        tasks.append(task_data)

                # Analyze chain characteristics
                has_conditionals = any(t.get("type") == "Conditional" for t in tasks)
                has_parallels = any(t.get("type") == "Parallel" for t in tasks)
                has_error_handling = any(t.get("type") == "ErrorHandling" for t in tasks)
                has_auto_parallel = any(t.get("creates_parallel", False) for t in tasks)

                chain_type = "Sequential"
                if has_auto_parallel:
                    chain_type = "Auto-Parallel"
                elif has_conditionals and has_parallels:
                    chain_type = "Complex"
                elif has_conditionals:
                    chain_type = "Conditional"
                elif has_parallels:
                    chain_type = "Mixed-Parallel"
                elif has_error_handling:
                    chain_type = "Error-Handling"

                return {
                    "type": "Chain",
                    "display": f"[Chain] {chain_type}",
                    "tasks": tasks,
                    "task_count": len(tasks),
                    "chain_type": chain_type,
                    "has_conditionals": has_conditionals,
                    "has_parallels": has_parallels,
                    "has_error_handling": has_error_handling,
                    "has_auto_parallel": has_auto_parallel,
                    "depth": depth
                }

            # Fallback for unknown types
            return {
                "type": "Unknown",
                "display": f"[Unknown] {type(comp).__name__}",
                "class_name": type(comp).__name__,
                "depth": depth
            }

        except Exception as e:
            return {
                "type": "Error",
                "display": f"[ERROR] {str(e)[:50]}",
                "error": str(e),
                "depth": depth
            }
        finally:
            visited.discard(comp_id)

    return {"structure": process_component(self)}


def print_graph(self):
    """Enhanced chain visualization with complete functionality coverage and parallel detection."""

    # Enhanced color scheme with parallel indicators
    COLORS = {
        "Agent": "\033[94m",  # Blue
        "Format": "\033[92m",  # Green
        "Condition": "\033[93m",  # Yellow
        "Parallel": "\033[95m",  # Magenta
        "Conditional": "\033[96m",  # Cyan
        "ErrorHandling": "\033[91m",  # Red
        "Chain": "\033[97m",  # White
        "Unknown": "\033[31m",  # Dark Red
        "Error": "\033[91m",  # Red
        "AutoParallel": "\033[105m",  # Bright Magenta Background
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    PARALLEL_ICON = "⚡"
    BRANCH_ICON = "🔀"
    ERROR_ICON = "🚨"

    def style_component(comp, override_color=None):
        """Apply enhanced styling with parallel indicators."""
        if not comp:
            return f"{COLORS['Error']}[NULL]{RESET}"

        comp_type = comp.get("type", "Unknown")
        display = comp.get("display", f"[{comp_type}]")

        # Special handling for parallel-creating formats
        if comp_type == "Format" and comp.get("creates_parallel", False):
            color = override_color or COLORS["AutoParallel"]
            return f"{color}{PARALLEL_ICON} {display}{RESET}"
        else:
            color = override_color or COLORS.get(comp_type, COLORS['Unknown'])
            return f"{color}{display}{RESET}"

    def print_section_header(title, details=None):
        """Print formatted section header."""
        print(f"\n{BOLD}{'=' * 60}{RESET}")
        print(f"{BOLD}🔗 {title}{RESET}")
        if details:
            print(f"{DIM}{details}{RESET}")
        print(f"{BOLD}{'=' * 60}{RESET}")

    def render_task_flow(tasks, indent="", show_parallel_creation=True):
        """Render tasks with parallel creation detection."""
        if not tasks:
            print(f"{indent}{DIM}(No tasks){RESET}")
            return

        for i, task in enumerate(tasks):
            if not task:
                continue

            is_last = i == len(tasks) - 1
            connector = "└─ " if is_last else "├─ "
            next_indent = indent + ("    " if is_last else "│   ")

            task_type = task.get("type", "Unknown")

            # Handle different task types
            if task_type == "Format" and task.get("creates_parallel", False):
                print(f"{indent}{connector}{style_component(task)}")

                # Show what happens next
                if i + 1 < len(tasks):
                    next_task = tasks[i + 1]
                    print(f"{next_indent}├─ {DIM}Creates parallel execution for:{RESET}")
                    print(f"{next_indent}└─ {PARALLEL_ICON} {style_component(next_task)}")
                    # Skip the next task in main loop since we showed it here
                    continue

            elif task_type == "Parallel":
                print(f"{indent}{connector}{style_component(task)}")
                branches = task.get("branches", [])

                for j, branch in enumerate(branches):
                    if branch:
                        branch_last = j == len(branches) - 1
                        branch_conn = "└─ " if branch_last else "├─ "
                        branch_indent = next_indent + ("    " if branch_last else "│   ")

                        print(f"{next_indent}{branch_conn}{BRANCH_ICON} Branch {j + 1}:")

                        if branch.get("type") == "Chain":
                            render_task_flow(branch.get("tasks", []), branch_indent, False)
                        else:
                            print(f"{branch_indent}└─ {style_component(branch)}")

            elif task_type == "Conditional":
                print(f"{indent}{connector}{style_component(task)}")

                # Condition
                condition = task.get("condition")
                if condition:
                    print(f"{next_indent}├─ {style_component(condition)}")

                # True branch
                true_branch = task.get("true_branch")
                false_branch = task.get("false_branch")
                has_false = false_branch is not None

                if true_branch:
                    true_conn = "├─ " if has_false else "└─ "
                    print(f"{next_indent}{true_conn}{COLORS['Conditional']}✓ TRUE:{RESET}")
                    true_indent = next_indent + ("│   " if has_false else "    ")

                    if true_branch.get("type") == "Chain":
                        render_task_flow(true_branch.get("tasks", []), true_indent, False)
                    else:
                        print(f"{true_indent}└─ {style_component(true_branch)}")

                if false_branch:
                    print(f"{next_indent}└─ {COLORS['Conditional']}✗ FALSE:{RESET}")
                    false_indent = next_indent + "    "

                    if false_branch.get("type") == "Chain":
                        render_task_flow(false_branch.get("tasks", []), false_indent, False)
                    else:
                        print(f"{false_indent}└─ {style_component(false_branch)}")

            elif task_type == "ErrorHandling":
                print(f"{indent}{connector}{style_component(task)}")

                primary = task.get("primary")
                fallback = task.get("fallback")
                has_fallback = fallback is not None

                if primary:
                    prim_conn = "├─ " if has_fallback else "└─ "
                    print(f"{next_indent}{prim_conn}{COLORS['Chain']}🎯 PRIMARY:{RESET}")
                    prim_indent = next_indent + ("│   " if has_fallback else "    ")

                    if primary.get("type") == "Chain":
                        render_task_flow(primary.get("tasks", []), prim_indent, False)
                    else:
                        print(f"{prim_indent}└─ {style_component(primary)}")

                if fallback:
                    print(f"{next_indent}└─ {ERROR_ICON} FALLBACK:")
                    fallback_indent = next_indent + "    "

                    if fallback.get("type") == "Chain":
                        render_task_flow(fallback.get("tasks", []), fallback_indent, False)
                    else:
                        print(f"{fallback_indent}└─ {style_component(fallback)}")

            else:
                print(f"{indent}{connector}{style_component(task)}")

    # Main execution
    try:
        # Generate graph structure
        graph_data = self.chain_to_graph()
        structure = graph_data.get("structure")

        if not structure:
            print_section_header("Empty Chain")
            return

        # Determine chain characteristics
        chain_type = structure.get("chain_type", "Unknown")
        has_auto_parallel = structure.get("has_auto_parallel", False)
        has_parallels = structure.get("has_parallels", False)
        has_conditionals = structure.get("has_conditionals", False)
        has_error_handling = structure.get("has_error_handling", False)
        task_count = structure.get("task_count", 0)

        # Build header info
        info_parts = [f"Tasks: {task_count}"]
        if has_auto_parallel:
            info_parts.append(f"{PARALLEL_ICON} Auto-Parallel")
        if has_parallels:
            info_parts.append(f"{BRANCH_ICON} Parallel Branches")
        if has_conditionals:
            info_parts.append("🔀 Conditionals")
        if has_error_handling:
            info_parts.append(f"{ERROR_ICON} Error Handling")

        print_section_header(f"Chain Visualization - {chain_type}", " | ".join(info_parts))

        # Handle different structure types
        struct_type = structure.get("type", "Unknown")

        if struct_type == "Chain":
            tasks = structure.get("tasks", [])
            render_task_flow(tasks)

        elif struct_type == "Parallel":
            print(f"{style_component(structure)}")
            branches = structure.get("branches", [])
            for i, branch in enumerate(branches):
                is_last = i == len(branches) - 1
                conn = "└─ " if is_last else "├─ "
                indent = "    " if is_last else "│   "

                print(f"{conn}{BRANCH_ICON} Branch {i + 1}:")
                if branch.get("type") == "Chain":
                    render_task_flow(branch.get("tasks", []), indent, False)
                else:
                    print(f"{indent}└─ {style_component(branch)}")

        elif struct_type == "Conditional":
            render_task_flow([structure])

        elif struct_type == "ErrorHandling":
            render_task_flow([structure])

        else:
            print(f"└─ {style_component(structure)}")

        print(f"\n{DIM}{'─' * 60}{RESET}")

    except Exception as e:
        print(f"\n{COLORS['Error']}{BOLD}[VISUALIZATION ERROR]{RESET}")
        print(f"{COLORS['Error']}Error: {str(e)}{RESET}")

        # Emergency fallback
        print(f"\n{DIM}--- Emergency Info ---{RESET}")
        try:
            attrs = []
            for attr in ['tasks', 'agents', 'condition', 'true_branch', 'false_branch', 'primary', 'fallback']:
                if hasattr(self, attr):
                    val = getattr(self, attr)
                    if val is not None:
                        if isinstance(val, (list, tuple)):
                            attrs.append(f"{attr}: {len(val)} items")
                        else:
                            attrs.append(f"{attr}: {type(val).__name__}")

            if attrs:
                print("Chain attributes:")
                for attr in attrs:
                    print(f"  • {attr}")
        except:
            print("Complete inspection failed")

        print(f"{DIM}--- End Emergency Info ---{RESET}\n")


# Attach methods to all chain classes
Chain.chain_to_graph = chain_to_graph
ParallelChain.chain_to_graph = chain_to_graph
ConditionalChain.chain_to_graph = chain_to_graph
ErrorHandlingChain.chain_to_graph = chain_to_graph

Chain.print_graph = print_graph
ParallelChain.print_graph = print_graph
ConditionalChain.print_graph = print_graph
ErrorHandlingChain.print_graph = print_graph

ParallelChain.print_graph = Chain.set_progress_callback
ConditionalChain.print_graph = Chain.set_progress_callback
ErrorHandlingChain.print_graph = Chain.set_progress_callback
