import ast
import io
import json
import pickle
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Dict, List, Union, Optional, Any, Tuple
from pydantic import BaseModel, Field
from IPython import get_ipython
from enum import Enum, auto
from dataclasses import dataclass
from pathlib import Path

from toolboxv2.mods.isaa.extras.modes import get_free_agent
from typing import Dict, Any, Optional
from inspect import getdoc, signature, isfunction
import ast
from collections import Counter


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


class MockIPython:
    def __init__(self):
        self.user_ns: Dict[str, Any] = {
            '__name__': '__main__',
            '__builtins__': __builtins__,
        }
        self.output_history = {}
        self._execution_count = 0
        self._session_dir = Path.home() / '.pipeline_sessions'
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
        session_data = {
            'user_ns': self.user_ns,
            'output_history': self.output_history
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
        task: str,
        mas_iter: int= 12,
        variables: Optional[Union[Dict[str, Any], List[Any]]] = None,
        top_n: Optional[bool] = None,
        include_usage: Optional[bool] = None,
    ):
        """
        Initialize the Pipeline.

        Args:
            agent: AI agent instance to use for task execution
            task: Task description string
            mas_iter: Maximum number of iterations (default: 12)
            variables: Dictionary or list of variables to make available
            top_n: Limit variable descriptions to top N most used
            include_usage: Include usage statistics in variable descriptions
        """
        self.include_usage = include_usage
        self.top_n = top_n
        self.mas_iter = mas_iter
        self.agent = agent
        self.task = task
        self.variables = self._process_variables(variables or {})
        self.execution_history = []
        self.session_name = None

        # Try to get IPython or create mock
        try:
            self.ipython = get_ipython()
            if self.ipython is None:
                raise NameError
        except NameError:
            self.ipython = MockIPython()

        # Initialize interpreter with variables
        self.ipython.user_ns.update(self.variables)

    def save_session(self, name: str):
        """Save current session"""
        self.session_name = name
        if isinstance(self.ipython, MockIPython):
            self.ipython.save_session(name)
        else:
            # Using IPython's store magic
            for var_name, value in self.variables.items():
                self.ipython.run_cell(f"%store {var_name}")
            self.ipython.run_cell(f"%store pipeline_session_{name}")

    def load_session(self, name: str):
        """Load saved session"""
        if isinstance(self.ipython, MockIPython):
            self.ipython.load_session(name)
            self.variables.update(self.ipython.user_ns)
        else:
            # Using IPython's store magic
            self.ipython.run_cell(f"%store -r pipeline_session_{name}")
            for var_name in self.variables:
                self.ipython.run_cell(f"%store -r {var_name}")
                if var_name in self.ipython.user_ns:
                    self.variables[var_name] = self.ipython.user_ns[var_name]

    def _process_variables(self, variables: Union[Dict[str, Any], List[Any]]) -> Dict[str, Any]:
        if isinstance(variables, list):
            return {f"var_{i}": var for i, var in enumerate(variables)}
        return variables

    def _generate_variable_descriptions(
        self,
        top_n: Optional[int] = None,
        include_usage: bool = False
    ) -> str:
        """
        Generate detailed descriptions of variables for the agent.

        Args:
            top_n: Optional limit to show only top N variables
            include_usage: Whether to include usage statistics
        """
        if top_n is None:
            top_n = self.top_n
        if include_usage is None:
            include_usage = self.include_usage
        descriptions = []
        variables = self.variables.items()

        # Track usage if requested
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
        variables = self.variables.items()

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

        for name, var in variables:
            desc_parts = [f"### {name}"]

            # Type information
            type_info = get_type_info(var)
            desc_parts.append(f"**Type:** `{type_info}`")

            # Value preview
            preview = get_value_preview(var)
            if preview:
                desc_parts.append(f"**Value:** {preview}")

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
                        if isfunction(attr_value) or inspect.ismethod(attr_value):
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

                    field_desc = f"- `{field_name}: {type_name}`"
                    if field.description:
                        field_desc += f"\n  {field.description}"
                    fields.append(field_desc)

                if fields:
                    desc_parts.append("**Fields:**\n" + "\n".join(fields))

            descriptions.append("\n".join(desc_parts))

        return "\n\n".join(descriptions)

    def _handle_method_update(self, class_name: str, method_name: str, content: str) -> Optional[ExecutionRecord]:
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
            update_dict = self.agent.format_class(
                MethodUpdate,
                prompt
            )

            method_update = MethodUpdate(**update_dict)

            # Execute the code
            result = self._execute_code(method_update.code)

            # If there's an error, try to fix it
            if isinstance(result.error, SyntaxError):
                fix_prompt = f"""Fix syntax error in code:
                Original code: {method_update.code}
                Error: {result.error}
                """

                fixed_dict = self.agent.format_class(
                    MethodUpdate,
                    fix_prompt
                )
                fixed_update = MethodUpdate(**fixed_dict)
                result = self._execute_code(fixed_update.code)

            return result

        except Exception as e:
            return ExecutionRecord(
                code=f"Failed to update method {method_name}",
                result=None,
                error=str(e)
            )

    def _execute_code(self, code: str) -> ExecutionRecord:
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
        if isinstance(self.ipython, MockIPython):
            return str(self.ipython)
        else:
            return "IPython session active"

    def _process_think_result(self, think_result: ThinkResult) -> Tuple[ThinkState,  Optional[ExecutionRecord | str]]:
        """Process the result of agent thinking"""
        if think_result.action == 'brake':
            return ThinkState.BRAKE, think_result.content

        elif think_result.action == 'method_update':
            class_name = think_result.context.get('class_name')
            method_name = think_result.context.get('method_name')
            result = None
            if class_name is not None and method_name is not None:
                result = self._handle_method_update(class_name, method_name, think_result.content)
            return ThinkState.THINKING, result

        elif think_result.action == 'code':
            result = self._execute_code(think_result.content)
            return ThinkState.PROCESSING, result

        elif think_result.action == 'done':
            return ThinkState.DONE, think_result.content

        elif think_result.action == 'inform':
            print("inform:", think_result.content)
            print("", think_result.context)
            return ThinkState.THINKING, think_result.content

        return ThinkState.THINKING, None

    def execute(self, code:str):
        return str(self._execute_code(code))

    def run(self, verbose=False) -> PipelineResult:
        """Run the pipeline with separated thinking and processing phases"""
        state = ThinkState.THINKING
        result = None


        message = [{'role': 'user', 'content': self.task}]
        iter_i = 0
        while iter_i > self.mas_iter or state != ThinkState.DONE:
            iter_i += 1
            initial_prompt = f"""
You are an AI agent designed to perform tasks that involve thinking before and after code iteration. Your goal is to complete the given task while demonstrating a clear thought process throughout the execution.

Before you begin, here are the available variables for context:
<available_variables>
{self._generate_variable_descriptions()}
</available_variables>

You will use a structure called ThinkResult to organize your thoughts and actions.
For each step of your task, follow this process:

1. Think before action:
   - Analyze the current state of the task
   - Consider possible next steps
   - Evaluate potential outcomes
   - write your detail reasoning and revaluations to content!

2. Choose an action:
   - 'code': Write code it will be executed, call functions and print or return vars
   - 'method_update': Update your approach or methodology
   - 'brake': Pause and reassess the situation
   - 'done': Indicate that the task is complete

3. Execute the action:
   - If 'code', write the necessary code content must be valid python code (hint: its run in an ipy session as cell!). the context:dict context is a dict and must include only 'reason' as key and value is -> write your Reasoning one 6 words max!
   - If 'method_update', explain the changes to your approach the context must include 'class_name' and 'method_name'
   - If 'brake', explain why you're pausing and what you need to reassess
   - If 'done', summarize the completed task and results

4. Think after action:
   - Evaluate the results of your action
   - Consider any new information or insights gained
   - Determine the next steps or if the task is complete

if their is missing of informations try running code to get the infos or break with a user question.

<code> example code : start
x = 1
y = 2
x + y # so i can see the result
x # so i can see x
y # so i can see y end
</code>

For each step, output your thoughts and actions using the format
Use the context to begine or do the next step.
Continue this process until the task is complete. When you've finished the task, use the 'done' action and provide a summary of the results.

Remember to demonstrate clear reasoning throughout the process and explain your decisions. If you encounter any difficulties or need to make assumptions, clearly state them in your thinking process.

Begin the task now or to the next step!
current iteration {iter_i} max {self.mas_iter}
current state : {state.name}
!!DO NOT REPEAT UR ACTIONS!!"""
            code_follow_up_prompt =f"""
### **After-Code Evaluation and Reassessment Prompt**

You are an AI agent responsible for critically evaluating the outcomes of a prior code iteration. Your task is to assess whether to **proceed**, **pause**, or **rethink the approach** without executing new code actions.

#### **Evaluation Process:**
1. **Review the Current State:**
   - Assess the results from the last executed code iteration.
   - Identify discrepancies, inefficiencies, or unexpected behaviors.
   - Analyze the logical soundness of the approach.

2. **Structured Decision-Making:**
   - **‘contains_final_result’** → If the task is successfully completed!
   - **‘contains_error’** →  If the task Failed!
   - **‘contains_thoughts’** → On an Internal Step

3. **Reevaluation Constraints:**
   - **Do not propose or execute new code.**
   - Focus on **logical evaluation** and **final result** rather than implementation details.
   - Base decisions on observed results and theoretical expectations.

Proceed with your evaluation based on the current state of the task."""

            print(f"Iteration : {iter_i} of {self.mas_iter}")
            if state == ThinkState.THINKING:
                # Get agent's thoughts
                think_dict = self.agent.format_class(
                    ThinkResult,
                    initial_prompt,
                    message=message.copy(),
                )
                if verbose:
                    print(json.dumps(think_dict, indent=2))
                think_result = ThinkResult(**think_dict)
                state, result = self._process_think_result(think_result)
                message.append({'role': 'assistant', 'content': think_result.content})
                if result and think_result.content[8:-8] != str(result).replace("Output -> ", ""):
                    message.append({'role': 'system', 'content': 'Evaluation: '+str(result)})
                if verbose:
                    print(think_result.content)
                    print(str(result))

            elif state == ThinkState.PROCESSING:
                # Get agent's thoughts
                class Next(BaseModel):
                    contains_final_result: bool
                    contains_thoughts: bool
                    contains_error: bool
                # Format the agent's thoughts into a structured response
                next_dict = self.agent.format_class(
                    Next,
                    code_follow_up_prompt,
                    message=message.copy(),
                )
                if verbose:
                    print(json.dumps(next_dict, indent=2))
                if next_dict.get('contains_thoughts', False):
                    state = ThinkState.THINKING
                    continue
                else:
                    break

            elif state == ThinkState.BRAKE:
                break

        return PipelineResult(
            variables=self.variables,
            result=result,
            execution_history=self.execution_history,
            message=message,
        )


from typing import Dict, Any, Optional, List, Union, Type, Set
import inspect
from copy import deepcopy
import re
from dataclasses import dataclass
from collections import defaultdict


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
        namespace = inspect.currentframe().f_back.f_globals

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
    initial_prompt = f"""
    You are an AI agent designed to perform tasks that involve thinking before and after code iteration. Your goal is to complete the given task while demonstrating a clear thought process throughout the execution.

    Before you begin, here are the available variables for context:
    <available_variables>
f
    </available_variables>

    You will use a structure called ThinkResult to organize your thoughts and actions.
    For each step of your task, follow this process:

    1. Think before action:
       - Analyze the current state of the task
       - Consider possible next steps
       - Evaluate potential outcomes
       - write your detail reasoning and revaluations to content!

    2. Choose an action:
       - 'code': Write code it will be executed, call functions and print or return vars
       - 'method_update': Update your approach or methodology
       - 'brake': Pause and reassess the situation
       - 'done': Indicate that the task is complete

    3. Execute the action:
       - If 'code', write the necessary code content must be valid python code (hint: its run in an ipy session as cell!). the context:dict context is a dict and must include only 'reason' as key and value is -> write your Reasoning one 6 words max!
       - If 'method_update', explain the changes to your approach the context must include 'class_name' and 'method_name'
       - If 'brake', explain why you're pausing and what you need to reassess
       - If 'done', summarize the completed task and results

    4. Think after action:
       - Evaluate the results of your action
       - Consider any new information or insights gained
       - Determine the next steps or if the task is complete

    if their is missing of informations try running code to get the infos or break with a user question.

    <code> example code : start
    x = 1
    y = 2
    x + y # so i can see the result
    x # so i can see x
    y # so i can see y end
    </code>

    For each step, output your thoughts and actions using the format
    Use the context to begine or do the next step.
    Continue this process until the task is complete. When you've finished the task, use the 'done' action and provide a summary of the results.

    Remember to demonstrate clear reasoning throughout the process and explain your decisions. If you encounter any difficulties or need to make assumptions, clearly state them in your thinking process.

    Begin the task now or to the next step!
    current iteration x max y
    current state :
    !!DO NOT REPEAT UR ACTIONS!!"""
    __pipeline = Pipeline(
        agent=agent,
        task="Enahnce the promt ahnd gud the resoning and refaction mor",
        variables=[initial_prompt]
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


