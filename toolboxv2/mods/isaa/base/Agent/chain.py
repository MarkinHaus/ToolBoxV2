
import asyncio
import copy
from enum import Enum
from typing import Any, Union
from pydantic import BaseModel


class ChainRunType(Enum):
    auto = "auto"
    a_run = "a_run"
    format_class = "format_class"


class CF:
    """Chain Format - handles formatting and data extraction between tasks."""

    def __init__(self, format_class: type[BaseModel]):
        self.format_class = format_class
        self.extract_key: str | tuple | None = None
        self.is_parallel_extraction = False

    def __sub__(self, key: str | tuple):
        """Implements the - operator for data extraction keys."""
        new_cf = copy.copy(self)
        if isinstance(key, str):
            if '[n]' in key:
                new_cf.extract_key = key.replace('[n]', '')
                new_cf.is_parallel_extraction = True
            else:
                new_cf.extract_key = key
        elif isinstance(key, tuple):
            new_cf.extract_key = key
        return new_cf

class IS:
    """Conditional check for branching logic."""

    def __init__(self, key: str, expected_value: Any = True):
        self.key = key
        self.expected_value = expected_value


# --- Kernarchitektur der Chains ---

class ChainBase:
    """Abstract base class for all chain types, providing common operators."""

    def __rshift__(self, other: Any) -> 'Chain':
        """Implements the >> operator to chain tasks sequentially."""
        if isinstance(self, Chain):
            new_tasks = self.tasks + [other]
            return Chain._create_chain(new_tasks)
        return Chain._create_chain([self, other])

    def __add__(self, other: Any) -> 'ParallelChain':
        """Implements the + operator for parallel execution."""
        return ParallelChain([self, other])

    def __and__(self, other: Any) -> 'ParallelChain':
        """Implements the & operator, an alias for parallel execution."""
        return ParallelChain([self, other])

    def __or__(self, other: Any) -> 'ErrorHandlingChain':
        """Implements the | operator for defining a fallback/error handling path."""
        return ErrorHandlingChain(self, other)

    def __mod__(self, other: Any) -> 'ConditionalChain':
        """Implements the % operator for defining a false/else branch in a condition."""
        # This is typically used after a conditional chain.
        if isinstance(self, ConditionalChain):
            self.false_branch = other
            return self
        # Allows creating a conditional chain directly
        return ConditionalChain(None, self, other)

    def set_progress_callback(self, progress_tracker: 'ProgressTracker'):
        """Recursively sets the progress callback for all tasks in the chain."""
        tasks_to_process = []
        if hasattr(self, 'tasks'): tasks_to_process.extend(self.tasks)  # Chain
        if hasattr(self, 'agents'): tasks_to_process.extend(self.agents)  # ParallelChain
        if hasattr(self, 'true_branch'): tasks_to_process.append(self.true_branch)  # ConditionalChain
        if hasattr(self, 'false_branch') and self.false_branch: tasks_to_process.append(
            self.false_branch)  # ConditionalChain
        if hasattr(self, 'primary'): tasks_to_process.append(self.primary)  # ErrorHandlingChain
        if hasattr(self, 'fallback'): tasks_to_process.append(self.fallback)  # ErrorHandlingChain

        for task in tasks_to_process:
            if hasattr(task, 'set_progress_callback'):
                task.set_progress_callback(progress_tracker)

    def __call__(self, *args, **kwargs):
        """Allows the chain to be called like a function, returning an awaitable runner."""
        return self._Runner(self, args, kwargs)

    class _Runner:
        def __init__(self, parent, args, kwargs):
            self.parent = parent
            self.args = args
            self.kwargs = kwargs

        def __call__(self):
            """Synchronous execution."""
            return asyncio.run(self.parent.a_run(*self.args, **self.kwargs))

        def __await__(self):
            """Asynchronous execution."""
            return self.parent.a_run(*self.args, **self.kwargs).__await__()

class Function(ChainBase):
    """A wrapper to treat native Python functions as chainable components."""

    def __init__(self, func: callable):
        if not callable(func):
            raise TypeError("Function object must be initialized with a callable.")
        self.func = func
        # Get a meaningful name for visualization
        self.func_name = getattr(func, '__name__', 'anonymous_lambda')

    async def a_run(self, data: Any, **kwargs):
        """Executes the wrapped function, handling both sync and async cases."""
        # Note: kwargs from the chain run are not passed to the native function
        # to maintain a simple, predictable (data in -> data out) interface.
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(data)
        else:
            return self.func(data)

    def __repr__(self):
        return f"Function(name='{self.func_name}')"

class ParallelChain(ChainBase):
    """Handles parallel execution of multiple agents or chains."""

    def __init__(self, agents: list[Union['FlowAgent', ChainBase]]):
        self.agents = agents

    async def a_run(self, query: Any, **kwargs):
        """Runs all agents/chains in parallel."""
        tasks = [agent.a_run(query, **kwargs) for agent in self.agents]
        results = await asyncio.gather(*tasks)
        return self._combine_results(results)

    def _combine_results(self, results: list[Any]) -> Any:
        """Intelligently combines parallel results."""
        if all(isinstance(r, str) for r in results):
            return " | ".join(results)
        return results


class ConditionalChain(ChainBase):
    """Handles conditional execution based on a condition."""

    def __init__(self, condition: IS, true_branch: Any, false_branch: Any = None):
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch

    async def a_run(self, data: Any, **kwargs):
        """Executes the true or false branch based on the condition."""
        condition_met = False
        if isinstance(self.condition, IS) and isinstance(data, dict):
            if data.get(self.condition.key) == self.condition.expected_value:
                condition_met = True

        if condition_met:
            return await self.true_branch.a_run(data, **kwargs)
        elif self.false_branch:
            return await self.false_branch.a_run(data, **kwargs)
        return data  # Return original data if condition not met and no false branch


class ErrorHandlingChain(ChainBase):
    """Handles exceptions in a primary chain by executing a fallback chain."""

    def __init__(self, primary: Any, fallback: Any):
        self.primary = primary
        self.fallback = fallback

    async def a_run(self, query: Any, **kwargs):
        """Tries the primary chain and executes the fallback on failure."""
        try:
            return await self.primary.a_run(query, **kwargs)
        except Exception as e:
            print(f"Primary chain failed with error: {e}. Running fallback.")
            return await self.fallback.a_run(query, **kwargs)


class Chain(ChainBase):
    """The main class for creating and executing sequential chains of tasks."""

    def __init__(self, agent: 'FlowAgent' = None):
        self.tasks: list[Any] = [agent] if agent else []
        self.progress_tracker: 'ChainPrinter' | None = None

    @classmethod
    def _create_chain(cls, components: list[Any]) -> 'Chain':
        chain = cls()
        chain.tasks = components
        return chain

    def _extract_data(self, data: dict, cf: CF) -> Any:
        """Extracts data from a dictionary based on the CF configuration."""
        if not isinstance(data, dict):
            return data

        key = cf.extract_key
        if key == '*':
            return data
        if isinstance(key, tuple):
            return {k: data.get(k) for k in key if k in data}
        if isinstance(key, str) and key in data:
            return data[key]
        return data  # Return original data if key not found

    async def a_run(self, query: Any, **kwargs):
        """
        Executes the chain of tasks asynchronously with dynamic method selection,
        data extraction, and auto-parallelization.
        """
        current_data = query

        # We need to iterate with an index to look ahead
        i = 0
        while i < len(self.tasks):
            task = self.tasks[i]

            # --- Auto-Erkennung und Ausführung ---
            if hasattr(task, 'a_run') and hasattr(task, 'a_format_class'):
                next_task = self.tasks[i + 1] if (i + 1) < len(self.tasks) else None
                task.active_session = kwargs.get("session_id", "default")
                # Dynamische Entscheidung: a_format_class oder a_run aufrufen?
                if isinstance(next_task, CF):
                    # Nächste Aufgabe ist Formatierung, also a_format_class aufrufen
                    current_data = await task.a_format_class(
                        next_task.format_class, str(current_data), **kwargs
                    )
                else:
                    # Standardausführung
                    current_data = await task.a_run(str(current_data), **kwargs)
                task.active_session = None

            elif isinstance(task, CF):
                # --- Auto-Extraktion und Parallelisierung ---
                if task.extract_key:
                    extracted_data = self._extract_data(current_data, task)

                    if task.is_parallel_extraction and isinstance(extracted_data, list):
                        next_task_for_parallel = self.tasks[i + 1] if (i + 1) < len(self.tasks) else None
                        if next_task_for_parallel:
                            # Erstelle eine temporäre Parallel-Kette und führe sie aus
                            parallel_runner = ParallelChain([next_task_for_parallel] * len(extracted_data))

                            # Führe jeden Task mit dem entsprechenden Datenelement aus
                            parallel_tasks = [
                                next_task_for_parallel.a_run(item, **kwargs) for item in extracted_data
                            ]
                            current_data = await asyncio.gather(*parallel_tasks)

                            print("Parallel results:", type(current_data))
                            print("Parallel results:", len(current_data))
                            # Überspringe die nächste Aufgabe, da sie bereits parallel ausgeführt wurde
                            i += 1
                        else:
                            current_data = extracted_data
                    else:
                        current_data = extracted_data
                else:
                    # Keine Extraktion, Daten bleiben unverändert (CF dient nur als Marker)
                    pass

            elif isinstance(task, ParallelChain | ConditionalChain | ErrorHandlingChain):
                current_data = await task.a_run(current_data, **kwargs)

            elif callable(task) and not isinstance(task, (ChainBase, type)):
                # Check if the function is async, then await it
                if asyncio.iscoroutinefunction(task):
                    current_data = await task(current_data)
                # Otherwise, run the synchronous function normally
                else:
                    current_data = task(current_data)
            elif hasattr(task, 'a_run'):
                current_data = await task.a_run(current_data, **kwargs)
            elif isinstance(task, IS):
                # IS needs to be paired with >> to form a ConditionalChain
                next_task_for_cond = self.tasks[i + 1] if (i + 1) < len(self.tasks) else None
                if next_task_for_cond:
                    # Form a conditional chain on the fly
                    conditional_task = ConditionalChain(task, next_task_for_cond)
                    # Check for a false branch defined with %
                    next_next_task = self.tasks[i + 2] if (i + 2) < len(self.tasks) else None
                    if isinstance(next_next_task, ConditionalChain) and next_next_task.false_branch:
                        conditional_task.false_branch = next_next_task.false_branch
                        i += 1  # also skip the false branch marker

                    current_data = await conditional_task.a_run(current_data, **kwargs)
                    i += 1  # Skip the next task as it's part of the conditional
                else:
                    raise ValueError("IS condition must be followed by a task to execute.")

            i += 1  # Gehe zur nächsten Aufgabe

        return current_data


def chain_to_graph(self) -> dict[str, Any]:
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
            if hasattr(comp, 'agents') and isinstance(comp.agents, list | tuple):
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

            if isinstance(comp, Function):
                return {
                    "type": "Function",
                    "display": f"[Func] {comp.func_name}",
                    "function_name": comp.func_name,
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
            if hasattr(comp, 'tasks') and isinstance(comp.tasks, list | tuple):
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
        "Function": "\033[35m",  # Light Purple
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
    FUNCTION_ICON = "ƒ"

    def style_component(comp, override_color=None):
        """Apply enhanced styling with parallel indicators."""
        if not comp:
            return f"{COLORS['Error']}[NULL]{RESET}"

        comp_type = comp.get("type", "Unknown")
        display = comp.get("display", f"[{comp_type}]")
        color = override_color or COLORS.get(comp_type, COLORS['Unknown'])
        # Special handling for parallel-creating formats
        if comp_type == "Format" and comp.get("creates_parallel", False):
            return f"{color}{PARALLEL_ICON} {display}{RESET}"
        elif comp_type == "Function":
            return f"{color}{FUNCTION_ICON} {display}{RESET}"
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

        elif struct_type == "Conditional" or struct_type == "ErrorHandling":
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
                        if isinstance(val, list | tuple):
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
