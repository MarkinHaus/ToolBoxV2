import io
import json
import pickle
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pydantic import BaseModel, Field
from pathlib import Path

from toolboxv2 import Style, Spinner, get_app
from toolboxv2.mods.isaa.extras.modes import get_free_agent

from inspect import getdoc, signature, isfunction, ismethod, currentframe
import ast
from collections import Counter
from typing import Optional, Dict, Any, List, Union, Type, Tuple
from copy import deepcopy
import re

from toolboxv2.mods.isaa.extras.session import ChatSession


### ---- Styles ------- ###

from enum import Enum, auto
from dataclasses import dataclass


class VerboseFormatter:
    def __init__(self, spinner_style: str = "d"):
        self.style = Style()
        self.current_spinner = None
        self.spinner_style = spinner_style

    def print_header(self, text: str):
        """Print a formatted header with separator line"""
        width = 80
        print(f"\n{self.style.BLUE('=' * width)}")
        print(self.style.BLUE2(f"⚡ {text.center(width - 4)} ⚡"))
        print(f"{self.style.BLUE('=' * width)}\n")

    def print_section(self, title: str, content: str):
        """Print a formatted section with title and content"""
        print(f"{self.style.YELLOW('┌─')} {self.style.YELLOW2(title)}")
        for line in content.split('\n'):
            print(f"{self.style.YELLOW('│')} {line}")
        print(f"{self.style.YELLOW('└─')} {self.style.GREY('End of section')}\n")

    def print_iteration(self, current: int, maximum: int):
        """Print iteration progress with visual bar"""
        progress = int((current / maximum) * 20)
        bar = "█" * progress + "░" * (20 - progress)
        print(f"\r{self.style.CYAN(f'Iteration [{bar}] {current}/{maximum}')}  ", end='')

    def print_state(self, state: str, details: Optional[Dict[str, Any]] = None):
        """Print current state with optional details"""
        state_color = {
            'THINKING': self.style.GREEN2,
            'PROCESSING': self.style.YELLOW2,
            'BRAKE': self.style.RED2,
            'DONE': self.style.BLUE2
        }.get(state, self.style.WHITE2)

        print(f"\n{self.style.Bold(f'Current State:')} {state_color(state)}")

        if details:
            for key, value in details.items():
                print(f"  {self.style.GREY('├─')} {self.style.CYAN(key)}: {value}")

    def print_method_update(self, method_update: 'MethodUpdate'):
        """Print a formatted view of a MethodUpdate structure"""
        # Header with class and method name
        print(f"\n{self.style.BLUE('┏━')} {self.style.Bold('Method Update Details')}")

        # Class and method information
        print(f"{self.style.BLUE('┣━')} Class: {self.style.GREEN2(method_update.class_name)}")
        print(f"{self.style.BLUE('┣━')} Method: {self.style.YELLOW2(method_update.method_name)}")

        # Description if available
        if method_update.description:
            print(f"{self.style.BLUE('┣━')} Description:")
            for line in method_update.description.split('\n'):
                print(f"{self.style.BLUE('┃')}  {self.style.GREY(line)}")

        # Code section
        print(f"{self.style.BLUE('┣━')} Code:")
        code_lines = method_update.code.split('\n')
        for i, line in enumerate(code_lines):
            # Different styling for first and last lines
            if i == 0:
                print(f"{self.style.BLUE('┃')}  {self.style.CYAN('┌─')} {line}")
            elif i == len(code_lines) - 1:
                print(f"{self.style.BLUE('┃')}  {self.style.CYAN('└─')} {line}")
            else:
                print(f"{self.style.BLUE('┃')}  {self.style.CYAN('│')} {line}")

        # Footer
        print(f"{self.style.BLUE('┗━')} {self.style.GREY('End of method update')}\n")

    async def process_with_spinner(self, message: str, coroutine):
        """Execute a coroutine with a spinner indicator"""
        with Spinner(message, symbols=self.spinner_style) as spinner:
            result = await coroutine
            return result




class EnhancedVerboseOutput:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.formatter = VerboseFormatter()

    async def log_message(self, role: str, content: str):
        """Log chat messages with role-based formatting"""
        if not self.verbose:
            return

        role_formats = {
            'user': (self.formatter.style.GREEN, "👤"),
            'assistant': (self.formatter.style.BLUE, "🤖"),
            'system': (self.formatter.style.YELLOW, "⚙️")
        }

        color_func, icon = role_formats.get(role, (self.formatter.style.WHITE, "•"))
        print(f"\n{icon} {color_func(f'[{role}]')}")
        print(f"{self.formatter.style.GREY('└─')} {content}\n")

    async def log_think_result(self, result: Dict[str, Any]):
        """Log thinking results with structured formatting"""
        if not self.verbose:
            return

        self.formatter.print_section(
            "Thinking Result",
            f"Action: {result.get('action', 'N/A')}\n"
            f"context: {result.get('context', 'N/A')}\n"
            f"Content: {result.get('content', '')}"
        )

    async def log_process_result(self, result: Dict[str, Any]):
        """Log processing results with structured formatting"""
        if not self.verbose:
            return

        self.formatter.print_section(
            "Process Result",
            f"Completed: {result.get('is_completed', False)}\n"
            f"Effectiveness: {result.get('effectiveness', 'N/A')}\n"
            f"Recommendations: \n{result.get('recommendations', 'None')}\n"
            f"workflow: \n{result.get('workflow', 'None')}\n"
            f"errors: {result.get('errors', 'None')}\n"
            f"text: {result.get('text', 'None')}"
        )

    async def log_method_update(self, method_update: 'MethodUpdate'):
        """Log method update with structured formatting"""
        if not self.verbose:
            return

    def log_header(self, text: str):
        """Log method update with structured formatting"""
        if not self.verbose:
            return

        self.formatter.print_header(text)

    def log_state(self, state: str, user_ns:dict):
        """Log method update with structured formatting"""
        if not self.verbose:
            return

        self.formatter.print_state(state, user_ns)

    async def process(self, message: str, coroutine):
        if not self.verbose:
            return await coroutine
        return await self.formatter.process_with_spinner(message, coroutine)

### -- TYPESs --- ###

class ThinkState(Enum):
    THINKING = auto()
    PROCESSING = auto()
    BRAKE = auto()
    DONE = auto()


class MethodUpdate(BaseModel):
    class_name: str = Field(..., description="Name of the class to update")
    method_name: str = Field(..., description="Name of the method to update")
    code: str = Field(..., description="Python code for the method implementation")
    description: Optional[str] = Field(None, description="Description of what the method does")


class ThinkResult(BaseModel):
    action: str = Field(..., description="Next action to take: 'code', 'method_update'', 'brake', 'done'")
    content: str = Field(..., description="Content related to the action")
    context: Optional[Dict] = Field(default_factory=dict, description="Additional context for the action")


@dataclass
class ExecutionRecord:
    code: str
    result: Any
    error: Optional[str] = None

    def __str__(self):
        return  '' if self.result is None and self.error is None else f"Output -> {self.result if self.result else ''}{'(error: '+self.error+')' if self.error else ''}"


@dataclass
class PipelineResult:
    variables: Dict[str, Any]
    result: Any
    execution_history: List[ExecutionRecord]
    message: List[Dict[str, str]]


### ---- logic ---- ###

class MockIPython:
    def __init__(self, _session_dir=None):
        self.user_ns: Dict[str, Any] = {
            '__name__': '__main__',
            '__builtins__': __builtins__,
        }
        self.output_history = {}
        self._execution_count = 0
        self._session_dir = _session_dir or Path(get_app().appdata) / '.pipeline_sessions'
        self._session_dir.mkdir(exist_ok=True)

    def reset(self):
        """Reset the interpreter state"""
        self.user_ns = {
            '__name__': '__main__',
            '__builtins__': __builtins__,
        }
        self.output_history.clear()
        self._execution_count = 0

    def get_namespace(self) -> Dict[str, Any]:
        """Get current namespace"""
        return self.user_ns.copy()

    def update_namespace(self, variables: Dict[str, Any]):
        """Update namespace with new variables"""
        self.user_ns.update(variables)


    def _parse_code(self, code: str) -> Tuple[Any, Optional[Any]]:
        """Parse code and separate last expression if present"""
        try:
            tree = ast.parse(code)
            if not tree.body:
                return None, None

            if isinstance(tree.body[-1], ast.Expr):
                # Split into statements and expression
                exec_code = ast.Module(
                    body=tree.body[:-1],
                    type_ignores=[],
                )
                eval_code = ast.Expression(
                    body=tree.body[-1].value
                )

                return (
                    compile(exec_code, '<string>', 'exec'),
                    compile(eval_code, '<string>', 'eval')
                )
            return compile(tree, '<string>', 'exec'), None

        except SyntaxError as e:
            raise SyntaxError(f"Invalid syntax: {str(e)}")

    def run_cell(self, code: str) -> Any:
        stdout = io.StringIO()
        stderr = io.StringIO()
        result = None
        error = None
        tb = None

        try:
            # Parse and compile code
            exec_code, eval_code = self._parse_code(code)
            if exec_code is None:
                return "No executable"

            # Execute in captured context
            with redirect_stdout(stdout), redirect_stderr(stderr):
                # Execute main code
                exec(exec_code, self.user_ns)

                # Evaluate final expression if present
                if eval_code is not None:
                    result = eval(eval_code, self.user_ns)
                    self.user_ns['_'] = result

        except Exception as e:
            error = str(e)
            tb = traceback.format_exc()
            stderr.write(f"{error}\n{tb}")

        finally:
            self._execution_count += 1
            d = {
                'code': code,
                'stdout': stdout.getvalue(),
                'stderr': stderr.getvalue(),
                'result': result if result else "stdout"
            }
            self.output_history[self._execution_count] = d
            if not result:
                result = ""
            if d.get('stdout'):
                result = f"{result}\nstdout:{d.get('stdout')}"
            if d.get('stderr'):
                result = f"{result}\nstderr:{d.get('stderr')}"
            return result

    def save_session(self, name: str):
        """Save current session to file"""
        session_file = self._session_dir / f"{name}.pkl"
        user_ns = self.user_ns.copy()
        output_history = self.output_history.copy()
        import pickle
        for key, value in user_ns.items():
            try:
                pickle.dumps(value)
            except Exception as e:
                user_ns[key] = f"not serializable: {str(value)}"
                continue
        for key, value in output_history.items():
            try:
                pickle.dumps(value)
            except Exception as e:
                output_history[key] = f"not serializable: {str(value)}"
                continue
        session_data = {
            'user_ns': user_ns,
            'output_history': output_history
        }
        with open(session_file, 'wb') as f:
            pickle.dump(session_data, f)

    def load_session(self, name: str):
        """Load session from file"""
        session_file = self._session_dir / f"{name}.pkl"
        if session_file.exists():
            with open(session_file, 'rb') as f:
                session_data = pickle.load(f)
                self.user_ns.update(session_data['user_ns'])
                self.output_history.update(session_data['output_history'])

    def __str__(self):
        """String representation of current session"""
        output = []
        for count, data in self.output_history.items():
            output.append(f"In [{count}]: {data['code']}")
            if data['stdout']:
                output.append(data['stdout'])
            if data['stderr']:
                output.append(f"Error: {data['stderr']}")
            if data['result'] is not None:
                output.append(f"Out[{count}]: {data['result']}")
        return "\n".join(output)


class Pipeline:
    """
        A pipeline for executing AI agent-driven tasks with interactive code execution and variable management.

        The Pipeline class provides a structured environment for AI agents to:
        1. Execute code in a controlled environment
        2. Manage and track variables
        3. Update methods dynamically
        4. Save and load session states
        5. Generate detailed variable descriptions

        Attributes:
            agent: The AI agent instance used for task execution
            task (str): The task to be performed
            mas_iter (int): Maximum number of iterations allowed (default: 12)
            variables (Dict[str, Any]): Dictionary of variables available to the pipeline
            top_n (Optional[int]): Limit variable descriptions to top N most used
            include_usage (bool): Include usage statistics in variable descriptions
            execution_history (List[ExecutionRecord]): History of executed code and results
            session_name (Optional[str]): Name of the current session if saved
            ipython: IPython or MockIPython instance for code execution

        Example:
            >>> agent = get_free_agent("demo", "anthropic/claude-3-haiku-20240307")
            >>> pipeline = Pipeline(
            ...     agent=agent,
            ...     task="Calculate fibonacci sequence",
            ...     variables={"n": 10}
            ... )
            >>> result = pipeline.run()
            >>> print(result.result)

        Notes:
            - The pipeline uses either IPython if available or a MockIPython implementation
            - Variables can be provided as either a dictionary or list
            - Session state can be saved and loaded
            - Method updates are handled through a structured BaseModel approach
        """
    def __init__(
        self,
        agent: Any,
        verbose: bool=False,
        max_iter: int= 6,
        variables: Optional[Union[Dict[str, Any], List[Any]]] = None,
        top_n: Optional[bool] = None,
        include_usage: Optional[bool] = None,
        max_think_after_think = None,
    ):
        """
        Initialize the Pipeline.

        Args:
            agent: AI agent instance to use for task execution
            verbose: print internal results
            max_iter: Maximum number of iterations (default: 12)
            variables: Dictionary or list of variables to make available
            top_n: Limit variable descriptions to top N most used
            include_usage: Include usage statistics in variable descriptions
        """
        self.include_usage = include_usage
        self.top_n = top_n
        self.max_iter = max_iter
        self.max_think_after_think = max_think_after_think or max_iter // 2
        self.agent = agent
        self.task = None
        self.verbose_output = EnhancedVerboseOutput(verbose=verbose)
        self.variables = self._process_variables(variables or {})
        self.execution_history = []
        self.session_name = None

        _session_dir = Path(get_app().appdata) / 'ChatSession'
        self.ipython = MockIPython(_session_dir)
        self.chat_session = ChatSession(agent.memory, space_name=f"ChatSession/Pipeline_{agent.amd.name}", max_length=max_iter)
        self.process_memory = ChatSession(agent.memory, space_name=f"ChatSession/Process_{agent.amd.name}")

        # Initialize interpreter with variables
        self.ipython.user_ns.update(self.variables)

    def on_exit(self):
        self.chat_session.on_exit()
        self.process_memory.on_exit()
        self.save_session(f"Pipeline_Session_{self.agent.amd.name}")

    def save_session(self, name: str):
        """Save current session"""
        self.session_name = name
        self.ipython.save_session(name)

    def load_session(self, name: str):
        """Load saved session"""
        self.ipython.load_session(name)
        self.variables.update(self.ipython.user_ns)

    @staticmethod
    def _process_variables(variables: Union[Dict[str, Any], List[Any]]) -> Dict[str, Any]:
        if isinstance(variables, list):
            return {f"var_{i}": var for i, var in enumerate(variables)}
        return variables

    def _generate_variable_descriptions(
        self,
        top_n: Optional[int] = None,
        include_usage: bool = False
    ) -> str:
        """
        Generate detailed descriptions of variables for the agent, including instance state.

        Args:
            top_n: Optional limit to show only top N variables
            include_usage: Whether to include usage statistics
        """
        if top_n is None:
            top_n = self.top_n
        if include_usage is None:
            include_usage = self.include_usage

        def get_type_info(var: Any) -> str:
            """Helper to get detailed type information"""
            if isinstance(var, type):
                return f"class '{var.__name__}'"
            elif isinstance(var, BaseModel):
                return f"Pydantic model '{var.__class__.__name__}'"
            elif hasattr(var, '__class__'):
                type_name = var.__class__.__name__
                module_name = var.__class__.__module__
                if module_name != 'builtins':
                    return f"{module_name}.{type_name}"
                return type_name
            return type(var).__name__

        def get_instance_state(var: Any) -> Dict[str, Any]:
            """Helper to get current instance state"""
            state = {}
            if hasattr(var, '__dict__'):
                for attr_name, attr_value in var.__dict__.items():
                    # Skip private attributes and methods
                    if not attr_name.startswith('_') and not callable(attr_value):
                        try:
                            # Handle basic types directly
                            if isinstance(attr_value, (int, float, bool, str)):
                                state[attr_name] = attr_value
                            # Handle lists, tuples, sets with preview
                            elif isinstance(attr_value, (list, tuple, set)):
                                preview = str(list(attr_value)[:5])[:-1] + ", ...]"
                                state[attr_name] = f"{len(attr_value)} items: {preview}"
                            # Enhanced dictionary handling
                            elif isinstance(attr_value, dict):
                                preview_items = []
                                for i, (k, v) in enumerate(attr_value.items()):
                                    if i >= 5:  # Limit to first 5 items
                                        preview_items.append("...")
                                        break
                                    # Format key and value, handling different types
                                    key_repr = repr(k) if isinstance(k, (
                                    str, int, float, bool)) else f"<{type(k).__name__}>"
                                    if isinstance(v, (str, int, float, bool)):
                                        val_repr = repr(v)
                                    elif isinstance(v, (list, tuple, set)):
                                        val_repr = f"<{type(v).__name__}[{len(v)} items]>"
                                    elif isinstance(v, dict):
                                        val_repr = f"<dict[{len(v)} items]>"
                                    else:
                                        val_repr = f"<{type(v).__name__}>"
                                    preview_items.append(f"{key_repr}: {val_repr}")

                                preview = ", ".join(preview_items)
                                state[attr_name] = f"{len(attr_value)} pairs: {{{preview}}}"
                            # Handle other objects with type info
                            else:
                                state[attr_name] = f"<{type(attr_value).__name__}> value: {attr_value}"
                        except:
                            state[attr_name] = "<error getting value>"
            return state

        def get_value_preview(var: Any) -> Optional[str]:
            """Helper to get a preview of the value"""
            try:
                if isinstance(var, (int, float, bool, str)):
                    return f"`{repr(var)}`"
                elif isinstance(var, (list, tuple, set)):
                    items = len(var)
                    sample = str(list(var)[:3])[:-1] + ", ...]"
                    return f"{items} items: {sample}"
                elif isinstance(var, dict):
                    items = len(var)
                    return f"{items} key/value pairs"
                return None
            except:
                return None

        # Track usage if requested
        variables = self.variables.items()
        if include_usage and hasattr(self, 'execution_history'):
            usage_counter = Counter()
            for record in self.execution_history:
                try:
                    tree = ast.parse(record.code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Name):
                            if node.id in self.variables:
                                usage_counter[node.id] += 1
                except:
                    continue

            if usage_counter:
                variables = sorted(
                    variables,
                    key=lambda x: usage_counter.get(x[0], 0),
                    reverse=True
                )

        # Limit to top N if specified
        if top_n is not None:
            variables = list(variables)[:top_n]

        descriptions = []
        for name, var in variables:
            desc_parts = [f"### {name}"]

            # Type information
            type_info = get_type_info(var)
            desc_parts.append(f"**Type:** `{type_info}`")

            # Value preview
            preview = get_value_preview(var)
            if preview:
                desc_parts.append(f"**Value:** {preview}")

            # Instance state (current values)
            if not isinstance(var, type):  # Only for instances, not classes
                instance_state = get_instance_state(var)
                if instance_state:
                    desc_parts.append("**Current Instance State:**")
                    for attr_name, attr_value in instance_state.items():
                        desc_parts.append(f"- `{attr_name}` = {attr_value}")

            # Get docstring if available
            if hasattr(var, '__doc__') and var.__doc__:
                doc = var.__doc__.strip()
                desc_parts.append(f"**Documentation:**\n{doc}")

            # Handle class instances
            if hasattr(var, '__class__'):
                # Get class docstring if different from instance docstring
                class_doc = getdoc(var.__class__)
                instance_doc = getdoc(var)
                if class_doc and class_doc != instance_doc:
                    desc_parts.append(f"**Class Documentation:**\n{class_doc}")

                # Get method information
                if hasattr(var, '__dict__'):
                    methods = []
                    for attr_name, attr_value in var.__dict__.items():
                        if isfunction(attr_value) or ismethod(attr_value):
                            try:
                                sig = signature(attr_value)
                                methods.append(f"- `{attr_name}{sig}`")
                                # Add method docstring if available
                                if attr_value.__doc__:
                                    doc = attr_value.__doc__.strip().split('\n')[0]  # First line only
                                    methods.append(f"  {doc}")
                            except ValueError:
                                methods.append(f"- `{attr_name}()`")

                    if methods:
                        desc_parts.append("**Methods:**\n" + "\n".join(methods))

            # Handle Pydantic models
            if isinstance(var, BaseModel):
                fields = []
                for field_name, field in var.model_fields.items():
                    field_type = field.annotation
                    if hasattr(field_type, '__name__'):
                        type_name = field_type.__name__
                    else:
                        type_name = str(field_type)

                    # Add current value for the field
                    current_value = getattr(var, field_name, None)
                    field_desc = f"- `{field_name}: {type_name}` = {repr(current_value)}"
                    if field.description:
                        field_desc += f"\n  {field.description}"
                    fields.append(field_desc)

                if fields:
                    desc_parts.append("**Fields:**\n" + "\n".join(fields))

            descriptions.append("\n".join(desc_parts))

        return "\n\n".join(descriptions)

    async def _handle_method_update(self, class_name: str, method_name: str, content: str) -> Optional[ExecutionRecord]:
        """Handle method updates using a BaseModel to structure the update"""
        prompt = f"""Generate method update for:
        content: {content}
        Class: {class_name}
        Method: {method_name}
        Current variables: {self._generate_variable_descriptions()}

        Provide the method implementation following the MethodUpdate model structure.
        """

        try:
            # Get method update using format_class
            update_dict = await self.agent.a_format_class(
                MethodUpdate,
                prompt
            )
            await self.verbose_output.log_method_update(update_dict)
            method_update = MethodUpdate(**update_dict)

            # Execute the code
            result = await self._execute_code(method_update.code)

            # If there's an error, try to fix it
            if isinstance(result.error, SyntaxError):
                fix_prompt = f"""Fix syntax error in code:
                Original code: {method_update.code}
                Error: {result.error}
                """

                fixed_dict = await self.agent.a_format_class(
                    MethodUpdate,
                    fix_prompt
                )
                fixed_update = MethodUpdate(**fixed_dict)
                result = await self._execute_code(fixed_update.code)

            return result

        except Exception as e:
            return ExecutionRecord(
                code=f"Failed to update method {method_name}",
                result=None,
                error=str(e)
            )

    async def _execute_code(self, code: str) -> ExecutionRecord:
        """Execute code and track results"""
        try:
            result = self.ipython.run_cell(code)

            # Update pipeline variables from IPython namespace
            for var_name in self.variables:
                if var_name in self.ipython.user_ns:
                    self.variables[var_name] = self.ipython.user_ns[var_name]

            record = ExecutionRecord(code=code, result=result, error=None)
            self.execution_history.append(record)
            return record

        except Exception as e:
            record = ExecutionRecord(code=code, result=None, error=str(e))
            self.execution_history.append(record)
            return record

    def __str__(self):
        """String representation of pipeline session"""
        return str(self.ipython)

    async def _process_think_result(self, think_result: ThinkResult) -> Tuple[ThinkState,  Optional[ExecutionRecord | str]]:
        """Process the result of agent thinking"""
        if think_result.action == 'brake':
            return ThinkState.BRAKE, think_result.content

        elif think_result.action == 'method_update':
            class_name = think_result.context.get('class_name')
            method_name = think_result.context.get('method_name')
            result = None
            if class_name is not None and method_name is not None:
                result = await self._handle_method_update(class_name, method_name, think_result.content)
            return ThinkState.THINKING, result

        elif think_result.action == 'code':
            result = await self._execute_code(think_result.content)
            return ThinkState.PROCESSING, result

        elif think_result.action == 'done':
            return ThinkState.DONE, think_result.content

        elif think_result.action == 'infos':
            infos = await self.chat_session.get_reference(think_result.content, to_str=True)
            return ThinkState.THINKING, infos

        elif think_result.action == 'guide':
            details = await self.process_memory.get_reference(think_result.content, to_str=True, unified_retrieve=True)
            plan = await self.agent.a_mini_task(f"Help Guide The next action, details {details}"+ think_result.content)
            return ThinkState.THINKING, plan

        return ThinkState.THINKING, None

    def execute(self, code:str):
        return str(self._execute_code(code))

    def clear(self):
        self.chat_session.mem.delete_memory(self.chat_session.space_name)
        self.chat_session.mem.create_memory(self.chat_session.space_name)
        self.chat_session.history = []
        self.process_memory.mem.delete_memory(self.chat_session.space_name)
        self.process_memory.mem.create_memory(self.chat_session.space_name)
        self.process_memory.history = []
        self.execution_history = []

    async def get_process_hint(self, task):
        return await self.process_memory.get_reference(task, to_str=True), await self.chat_session.get_reference(task, to_str=True, unified_retrieve=True)

    async def run(self, task) -> PipelineResult:
        """Run the pipeline with separated thinking and processing phases"""
        state = ThinkState.THINKING
        result = None
        code_follow_up_prompt = f"""
You are an AI assistant responsible for evaluating task completion and providing feedback on the execution process. Your goal is to determine if a given task has been completed based on the execution result, and to offer insights for future improvements.

You will be provided with two inputs:
<task_description>
{task}
</task_description>

<execution_result>
#EXECUTION_RESULT#
</execution_result>

First, carefully analyze the task description and the execution result. Determine whether the task has been completed successfully based on the information provided.

If the task is completed:
1. Prepare a brief statement indicating that the task is done.
2. Summarize the output for the user in a clear and concise manner.

If the task is not completed:
1. Prepare a brief statement indicating that the task is not done.
2. Identify the specific aspects of the task that remain incomplete.

Regardless of task completion status, evaluate the procedure and effectiveness of the execution:
1. Analyze the workflow: Describe the steps taken in the execution process.
2. Assess effectiveness: Determine how well the procedure achieved the desired outcome.
3. Identify errors: Pinpoint any mistakes or inefficiencies in the execution.
4. Provide recommendations: Suggest improvements for future task executions.

Ensure that your evaluation is thorough, constructive, and provides actionable insights for improving future task executions."""
        initial_prompt = """
You are an AI agent designed to perform tasks that involve thinking before and after code iteration. Your goal is to complete the given task while demonstrating a clear thought process throughout the execution.

Before you begin, here are the available variables for context:
<available_variables>
#LOCALS#
</available_variables>

You will use a structure called ThinkResult to organize your thoughts and actions.
For each step of your task, follow this process:

1. Choose an action:
   - 'code': Write code it will be executed, call functions and print or return vars! most Important action use to think evaluate and write code!
   - 'method_update': Update a code snippet from the avalabel variables.
   - 'infos': to get ref's infos form past chat
   - 'guide': if current step is unclear use 'guide' to help you understand the next step
   - 'brake': Pause and reassess the situation
   - 'done': Indicate that the task is complete

2. Execute the action:
   - If 'code', write the necessary code content must be valid python code (hint: its run in an ipy session as cell!). the context:dict context is a dict and must include only 'reason' as key and value is -> write your Reasoning one 6 words max!
   - If 'method_update', explain the changes to your approach the context must include 'class_name' and 'method_name'
   - If 'infos', write the necessary information you seek to content to receive a infos if avalabel
   - If 'brake', explain why you're pausing and what you need to reassess
   - If 'done', summarize the completed task and results

if their is missing of informations try running code to get the infos or break with a user question.

<code> example code : start
x = 1
y = 2
x + y # so i can see the result
x # so i can see x
y # so i can see y end
</code>

<process_memory_hints>
#PHINT#
</process_memory_hints>

<chat_memory_context>
#CHINT#
</chat_memory_context>

For each step, output your thoughts and actions using the format
Use the context to begine or do the next step.
Continue this process until the task is complete. When you've finished the task, use the 'done' action and provide a summary of the results.

Remember to demonstrate clear reasoning throughout the process and explain your decisions. If you encounter any difficulties or need to make assumptions, clearly state them in your thinking process.

Begin the task now or to the next step!
current iteration #ITER#
current state : #STATE#
!!DO NOT REPEAT UR ACTIONS!!"""
        p_hint, c_hint = await self.get_process_hint(task)
        initial_prompt = initial_prompt.replace('#PHINT#', p_hint)
        initial_prompt = initial_prompt.replace('#CHINT#', c_hint)
        print(initial_prompt)
        iter_i = 0
        iter_p = 0
        iter_tat = 0
        await self.chat_session.add_message({'role': 'user', 'content': task})
        # await self.verbose_output.log_message('user', task)
        self.verbose_output.log_header(task)
        while state != ThinkState.DONE:
            if iter_i > self.max_iter:
                break
            iter_i += 1

            prompt = initial_prompt.replace('#ITER#', f'{iter_i} max {self.max_iter}')
            prompt = prompt.replace('#STATE#', f'{state.name}')
            prompt = prompt.replace('#LOCALS#', f'{self._generate_variable_descriptions()}')

            self.verbose_output.log_state(state.name, {})
            self.verbose_output.formatter.print_iteration(iter_i, self.max_iter)
            if state == ThinkState.THINKING:
                iter_tat +=1
                if iter_tat > self.max_think_after_think:
                    state = ThinkState.BRAKE
            else:
                iter_tat = 0

            if state == ThinkState.THINKING:
                # Get agent's thoughts
                think_dict = await self.agent.a_format_class(
                    ThinkResult,
                    prompt,
                    message=self.chat_session.get_past_x(self.max_iter*2).copy(),
                )
                await self.verbose_output.log_think_result(think_dict)
                think_result = ThinkResult(**think_dict)
                state, result = await self.verbose_output.process(think_dict.get("action"), self._process_think_result(think_result))
                await self.chat_session.add_message({'role': 'assistant', 'content': think_result.content})
                if result and think_result.content[8:-8] != str(result).replace("Output -> ", ""):
                    await self.chat_session.add_message({'role': 'system', 'content': 'Evaluation: '+str(result)})
                    await self.verbose_output.log_message('system', str(result))

            elif state == ThinkState.PROCESSING:
                # Get agent's thoughts
                class Next(BaseModel):
                    is_completed: bool
                    recommendations: str
                    errors: str
                    effectiveness: str
                    workflow: str
                    text: str
                # Format the agent's thoughts into a structured response
                next_dict = await self.agent.a_format_class(
                    Next,
                    code_follow_up_prompt,
                    message=self.chat_session.get_past_x(self.max_iter*2).copy(),
                )
                next_infos = json.dumps(next_dict, indent=2)
                await self.verbose_output.log_process_result(next_dict)
                await self.process_memory.add_message({'role': 'assistant', 'content': next_infos})
                iter_p += 1
                if not next_dict.get('is_completed', True):
                    state = ThinkState.THINKING
                    continue
                elif next_dict.get('is_completed', False):
                    state = ThinkState.DONE
                    continue
                else:
                    result = next_dict.get('text', '')
                    break

            elif state == ThinkState.BRAKE:
                break

        self.verbose_output.log_state(state.name, {})

        return PipelineResult(
            variables=self.variables,
            result=result,
            execution_history=self.execution_history,
            message=self.chat_session.get_past_x(iter_i*2)+self.process_memory.get_past_x(iter_p),
        )

### -- extra -- ###

@dataclass
class SyncReport:
    """Report of variables synced from namespace to pipeline"""
    added: Dict[str, str]
    skipped: Dict[str, str]  # var_name -> reason
    errors: Dict[str, str]  # var_name -> error message

    def __str__(self) -> str:
        parts = []
        if self.added:
            parts.append("Added variables:")
            for name, type_ in self.added.items():
                parts.append(f"  - {name}: {type_}")
        if self.skipped:
            parts.append("\nSkipped variables:")
            for name, reason in self.skipped.items():
                parts.append(f"  - {name}: {reason}")
        if self.errors:
            parts.append("\nErrors:")
            for name, error in self.errors.items():
                parts.append(f"  - {name}: {error}")
        return "\n".join(parts)


def sync_globals_to_vars(
    pipeline: Any,
    namespace: Optional[Dict[str, Any]] = None,
    prefix: Optional[str] = None,
    include_types: Optional[Union[Type, List[Type]]] = None,
    exclude_patterns: Optional[List[str]] = None,
    exclude_private: bool = True,
    deep_copy: bool = False,
    only_serializable: bool = False
) -> SyncReport:
    """
    Sync global variables or a specific namespace to pipeline variables.

    Args:
        pipeline: Pipeline instance to sync variables to
        namespace: Optional dictionary of variables (defaults to globals())
        prefix: Optional prefix for variable names (e.g., 'global_')
        include_types: Only include variables of these types
        exclude_patterns: List of regex patterns to exclude
        exclude_private: Exclude variables starting with underscore
        deep_copy: Create deep copies of variables instead of references
        only_serializable: Only include variables that can be serialized

    Returns:
        SyncReport with details about added, skipped and error variables

    Usage example:
# Basic usage - sync all globals
report = sync_globals_to_vars(pipeline)

# Sync only numeric types with prefix
report = sync_globals_to_vars(
    pipeline,
    include_types=[int, float],
    prefix="global_"
)

# Sync from specific namespace
import numpy as np
namespace = {"arr": np.array([1,2,3])}
report = sync_globals_to_vars(pipeline, namespace=namespace)

# Sync with deep copy and serialization check
report = sync_globals_to_vars(
    pipeline,
    deep_copy=True,
    only_serializable=True
)
    """
    # Initialize report
    report = SyncReport(
        added={},
        skipped={},
        errors={}
    )

    # Get namespace
    if namespace is None:
        # Get caller's globals
        namespace = currentframe().f_back.f_globals

    # Compile exclude patterns
    if exclude_patterns:
        patterns = [re.compile(pattern) for pattern in exclude_patterns]
    else:
        patterns = []

    # Normalize include_types
    if include_types and not isinstance(include_types, (list, tuple, set)):
        include_types = [include_types]
    def get_type_info(var: Any) -> str:
        """Helper to get detailed type information"""
        if isinstance(var, type):
            return f"class '{var.__name__}'"
        elif isinstance(var, BaseModel):
            return f"Pydantic model '{var.__class__.__name__}'"
        elif hasattr(var, '__class__'):
            type_name = var.__class__.__name__
            module_name = var.__class__.__module__
            if module_name != 'builtins':
                return f"{module_name}.{type_name}"
            return type_name
        return type(var).__name__
    # Process each variable
    for name, value in namespace.items():
        try:
            # Skip if matches exclude criteria
            if exclude_private and name.startswith('_'):
                report.skipped[name] = "private variable"
                continue

            if any(pattern.match(name) for pattern in patterns):
                report.skipped[name] = "matched exclude pattern"
                continue

            if include_types and not isinstance(value, tuple(include_types)):
                report.skipped[name] = f"type {type(value).__name__} not in include_types"
                continue

            # Test serialization if required
            if only_serializable:
                try:
                    import pickle
                    pickle.dumps(value)
                except Exception as e:
                    report.skipped[name] = f"not serializable: {str(e)}"
                    continue

            # Prepare variable
            var_value = deepcopy(value) if deep_copy else value
            var_name = f"{prefix}{name}" if prefix else name

            # Add to pipeline variables
            pipeline.variables[var_name] = var_value
            report.added[var_name] = get_type_info(value)

        except Exception as e:
            report.errors[name] = str(e)

    return report


if __name__ == '__main__':
    agent = get_free_agent("demo", "anthropic/claude-3-haiku-20240307")

    __pipeline = Pipeline(
        agent=agent,
        task="Enahnce the promt ahnd gud the resoning and refaction mor",
        variables=[]
    )
    # print(str(sync_globals_to_vars(__pipeline, namespace={"Any": Any, "sync_globals_to_vars": sync_globals_to_vars})))
    final_result = __pipeline.run(True)
    print(str(__pipeline))
    print(f"Final variables: {final_result.variables}")
    print(f"Result: {final_result.result}")
    print(f"Result: {final_result.message}")
    for record in final_result.execution_history:
        print(f"\nCode: {record.code}")
        print(f"Result: {record.result}")
        if record.error:
            print(f"Error: {record.error}")


