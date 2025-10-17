# core/executors.py
import io
import logging
import os
import sys
import time
from typing import Any

from pydantic import BaseModel

from toolboxv2 import get_logger

# --- Dependency Check ---
try:
    import restrictedpython
    from restrictedpython import compile_restricted, safe_globals
    RESTRICTEDPYTHON_AVAILABLE = True
except ImportError:
    os.system(f"{sys.executable} -m pip install restrictedpython")
    RESTRICTEDPYTHON_AVAILABLE = False
    def restrictedpython():
        return None

    def _print_(*args, **kwargs):
        print(*args, **kwargs)


    def _getattr_(obj, attr):
        return getattr(obj, attr)


    def _getitem_(obj, key):
        return obj[key]


    def _write_(obj, attr, value):
        setattr(obj, attr, value)

    restrictedpython.PrintCollector = _print_
    restrictedpython.safe_getattr = _getattr_
    restrictedpython.safe_getitem = _getitem_
    restrictedpython.guarded_setattr = _write_

try:
    import docker
    from docker.errors import APIError, ContainerError, ImageNotFound
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

# --- Base Executor Abstraction ---
# Inherit from ADK's BaseCodeExecutor if available for compatibility
try:
    from google.adk.code_executors import (
        BaseCodeExecutor,
        CodeExecutionInput,
        CodeExecutionResult,
        InvocationContext,
    )
    _BaseExecutorClass = BaseCodeExecutor
    ADK_EXEC_AVAILABLE = True
except ImportError:
    # Define dummy types if ADK not installed
    class CodeExecutionInput(BaseModel):
        code: str
        language: str = "python"
        # other ADK fields if needed

    class CodeExecutionResult(BaseModel):
        output: str
        outcome: str = "OUTCOME_UNSPECIFIED" # Or map to success/failure

    class InvocationContext: pass # Dummy context
    _BaseExecutorClass = object
    ADK_EXEC_AVAILABLE = False

logger = logging.getLogger("CodeExecutor")
logger.setLevel(get_logger().level)
safe_globals = {}
# --- RestrictedPython Executor ---

class RestrictedPythonExecutor(_BaseExecutorClass):
    """
    Executes Python code using restrictedpython.

    Safer than exec() but NOT a full sandbox. Known vulnerabilities exist.
    Use with extreme caution and only with trusted code sources or for
    low-risk operations. Docker is strongly recommended for untrusted code.
    """
    DEFAULT_ALLOWED_GLOBALS = {
        **safe_globals,
        '_print_': restrictedpython.PrintCollector,
        '_getattr_': restrictedpython.safe_getattr,
        '_getitem_': restrictedpython.safe_getitem,
        '_write_': restrictedpython.guarded_setattr, # Allows modifying specific safe objects if needed
        # Add other safe builtins or modules carefully
        'math': __import__('math'),
        'random': __import__('random'),
        'datetime': __import__('datetime'),
        'time': __import__('time'),
        # 'requests': None, # Example: Explicitly disallow
    }

    def __init__(self, allowed_globals: dict | None = None, max_execution_time: int = 5):
        if not RESTRICTEDPYTHON_AVAILABLE:
            raise ImportError("restrictedpython is not installed. Cannot use RestrictedPythonExecutor.")
        self.allowed_globals = allowed_globals or self.DEFAULT_ALLOWED_GLOBALS
        self.max_execution_time = max_execution_time # Basic timeout (not perfectly enforced by restrictedpython)
        logger.warning("Initialized RestrictedPythonExecutor. This provides LIMITED sandboxing. Use Docker for untrusted code.")

    def _execute(self, code: str) -> dict[str, Any]:
        """Internal execution logic."""
        start_time = time.monotonic()
        result = {"stdout": "", "stderr": "", "error": None, "exit_code": None}
        local_vars = {}
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Basic timeout check (not preemptive)
            if time.monotonic() - start_time > self.max_execution_time:
                 raise TimeoutError(f"Execution exceeded max time of {self.max_execution_time}s (pre-check).")

            # Compile the code in restricted mode
            byte_code = compile_restricted(code, filename='<inline code>', mode='exec')

            # Add a print collector to capture output
            self.allowed_globals['_print_'] = restrictedpython.PrintCollector
            print_collector = self.allowed_globals['_print_']()
            exec_globals = {**self.allowed_globals, '_print': print_collector}

            # Execute the compiled code
            # Note: restrictedpython does not inherently support robust timeouts during exec
            exec(byte_code, exec_globals, local_vars)

            # Check execution time again
            duration = time.monotonic() - start_time
            if duration > self.max_execution_time:
                logger.warning(f"Execution finished but exceeded max time ({duration:.2f}s > {self.max_execution_time}s).")
                # Potentially treat as an error or partial success

            result["stdout"] = print_collector.printed_text # Access collected prints
            result["exit_code"] = 0 # Assume success if no exception

        except TimeoutError as e:
            result["stderr"] = f"TimeoutError: {e}"
            result["error"] = str(e)
            result["exit_code"] = -1 # Indicate timeout
        except SyntaxError as e:
            result["stderr"] = f"SyntaxError: {e}"
            result["error"] = str(e)
            result["exit_code"] = 1
        except Exception as e:
            # Capture other potential execution errors allowed by restrictedpython
            error_type = type(e).__name__
            error_msg = f"{error_type}: {e}"
            result["stderr"] = error_msg
            result["error"] = str(e)
            result["exit_code"] = 1
            logger.warning(f"RestrictedPython execution caught exception: {error_msg}", exc_info=False) # Avoid logging potentially sensitive details from code
        finally:
            stdout_capture.close() # Not used directly with PrintCollector
            stderr_capture.close()

        return result

    # --- ADK Compatibility Method ---
    if ADK_EXEC_AVAILABLE:
        def execute_code(self, invocation_context: InvocationContext, code_input: CodeExecutionInput) -> CodeExecutionResult:
            logger.debug(f"RestrictedPythonExecutor executing ADK request (lang: {code_input.language}). Code: {code_input.code[:100]}...")
            if code_input.language.lower() != 'python':
                 return CodeExecutionResult(output=f"Error: Unsupported language '{code_input.language}'. Only Python is supported.", outcome="OUTCOME_FAILURE")

            exec_result = self._execute(code_input.code)

            output_str = ""
            if exec_result["stdout"]:
                output_str += f"Stdout:\n{exec_result['stdout']}\n"
            if exec_result["stderr"]:
                 output_str += f"Stderr:\n{exec_result['stderr']}\n"
            if not output_str and exec_result["exit_code"] == 0:
                 output_str = "Execution successful with no output."
            elif not output_str and exec_result["exit_code"] != 0:
                 output_str = f"Execution failed with no output (Exit code: {exec_result['exit_code']}). Error: {exec_result['error']}"


            outcome = "OUTCOME_SUCCESS" if exec_result["exit_code"] == 0 else "OUTCOME_FAILURE"

            return CodeExecutionResult(output=output_str.strip(), outcome=outcome)
    # --- End ADK Compatibility ---

    # --- Direct Call Method ---
    def execute(self, code: str) -> dict[str, Any]:
        """Directly execute code, returning detailed dictionary."""
        logger.debug(f"RestrictedPythonExecutor executing direct call. Code: {code[:100]}...")
        return self._execute(code)
    # --- End Direct Call ---


# --- Docker Executor (Recommended for Untrusted Code) ---

class DockerCodeExecutor(_BaseExecutorClass):
    """
    Executes Python code in a sandboxed Docker container.

    Requires Docker to be installed and running, and the 'docker' Python SDK.
    """
    DEFAULT_DOCKER_IMAGE = "python:3.10-slim" # Use a minimal image
    DEFAULT_TIMEOUT = 10 # Seconds
    DEFAULT_MEM_LIMIT = "128m"
    DEFAULT_CPUS = 0.5

    def __init__(self,
                 docker_image: str = DEFAULT_DOCKER_IMAGE,
                 timeout: int = DEFAULT_TIMEOUT,
                 mem_limit: str = DEFAULT_MEM_LIMIT,
                 cpus: float = DEFAULT_CPUS,
                 network_mode: str = "none", # Disable networking by default for security
                 docker_client_config: dict | None = None):
        if not DOCKER_AVAILABLE:
            raise ImportError("Docker SDK not installed ('pip install docker'). Cannot use DockerCodeExecutor.")

        self.docker_image = docker_image
        self.timeout = timeout
        self.mem_limit = mem_limit
        self.cpus = cpus
        self.network_mode = network_mode
        try:
            self.client = docker.from_env(**(docker_client_config or {}))
            self.client.ping() # Check connection
            # Ensure image exists locally or pull it
            try:
                self.client.images.get(self.docker_image)
                logger.info(f"Docker image '{self.docker_image}' found locally.")
            except ImageNotFound:
                logger.warning(f"Docker image '{self.docker_image}' not found locally. Attempting to pull...")
                try:
                    self.client.images.pull(self.docker_image)
                    logger.info(f"Successfully pulled Docker image '{self.docker_image}'.")
                except APIError as pull_err:
                    raise RuntimeError(f"Failed to pull Docker image '{self.docker_image}': {pull_err}") from pull_err
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Docker daemon: {e}. Is Docker running?") from e
        logger.info(f"DockerCodeExecutor initialized (Image: {docker_image}, Timeout: {timeout}s, Network: {network_mode})")

    def _execute(self, code: str) -> dict[str, Any]:
        """Internal execution logic."""
        result = {"stdout": "", "stderr": "", "error": None, "exit_code": None}
        container = None

        try:
            logger.debug(f"Creating Docker container from image '{self.docker_image}'...")
            container = self.client.containers.run(
                image=self.docker_image,
                command=["python", "-c", code],
                detach=True,
                mem_limit=self.mem_limit,
                nano_cpus=int(self.cpus * 1e9),
                network_mode=self.network_mode,
                # Security considerations: Consider read-only filesystem, dropping capabilities
                read_only=True,
                # working_dir="/app", # Define a working dir if needed
                # volumes={...} # Mount volumes carefully if required
            )
            logger.debug(f"Container '{container.short_id}' started.")

            # Wait for container completion with timeout
            container_result = container.wait(timeout=self.timeout)
            result["exit_code"] = container_result.get("StatusCode", None)

            # Retrieve logs
            result["stdout"] = container.logs(stdout=True, stderr=False).decode('utf-8', errors='replace').strip()
            result["stderr"] = container.logs(stdout=False, stderr=True).decode('utf-8', errors='replace').strip()

            logger.debug(f"Container '{container.short_id}' finished with exit code {result['exit_code']}.")
            if result["exit_code"] != 0:
                 logger.warning(f"Container stderr: {result['stderr'][:500]}...") # Log stderr on failure

        except ContainerError as e:
            result["error"] = f"ContainerError: {e}"
            result["stderr"] = e.stderr.decode('utf-8', errors='replace').strip() if e.stderr else str(e)
            result["exit_code"] = e.exit_status
            logger.error(f"Container '{container.short_id if container else 'N/A'}' failed: {result['error']}\nStderr: {result['stderr']}")
        except APIError as e:
            result["error"] = f"Docker APIError: {e}"
            result["exit_code"] = -1
            logger.error(f"Docker API error during execution: {e}")
        except Exception as e:
            # Catch potential timeout errors from container.wait or other unexpected issues
            result["error"] = f"Unexpected execution error: {type(e).__name__}: {e}"
            result["exit_code"] = -1
            # Check if it looks like a timeout
            if isinstance(e, TimeoutError) or "Timeout" in str(e): # docker SDK might raise requests.exceptions.ReadTimeout
                result["stderr"] = f"Execution timed out after {self.timeout} seconds."
                logger.warning(f"Container execution timed out ({self.timeout}s).")
            else:
                logger.error(f"Unexpected error during Docker execution: {e}", exc_info=True)
        finally:
            if container:
                try:
                    logger.debug(f"Removing container '{container.short_id}'...")
                    container.remove(force=True)
                except APIError as rm_err:
                    logger.warning(f"Failed to remove container {container.short_id}: {rm_err}")

        return result

     # --- ADK Compatibility Method ---
    if ADK_EXEC_AVAILABLE:
        def execute_code(self, invocation_context: InvocationContext, code_input: CodeExecutionInput) -> CodeExecutionResult:
            logger.debug(f"DockerCodeExecutor executing ADK request (lang: {code_input.language}). Code: {code_input.code[:100]}...")
            if code_input.language.lower() != 'python':
                 return CodeExecutionResult(output=f"Error: Unsupported language '{code_input.language}'. Only Python is supported.", outcome="OUTCOME_FAILURE")

            exec_result = self._execute(code_input.code)

            output_str = ""
            if exec_result["stdout"]:
                output_str += f"Stdout:\n{exec_result['stdout']}\n"
            if exec_result["stderr"]:
                 output_str += f"Stderr:\n{exec_result['stderr']}\n"
            if not output_str and exec_result["exit_code"] == 0:
                 output_str = "Execution successful with no output."
            elif not output_str and exec_result["exit_code"] != 0:
                 output_str = f"Execution failed with no output (Exit code: {exec_result['exit_code']}). Error: {exec_result['error']}"

            outcome = "OUTCOME_SUCCESS" if exec_result["exit_code"] == 0 else "OUTCOME_FAILURE"

            return CodeExecutionResult(output=output_str.strip(), outcome=outcome)
    # --- End ADK Compatibility ---

    # --- Direct Call Method ---
    def execute(self, code: str) -> dict[str, Any]:
        """Directly execute code, returning detailed dictionary."""
        logger.debug(f"DockerCodeExecutor executing direct call. Code: {code[:100]}...")
        return self._execute(code)
    # --- End Direct Call ---

# --- Factory function ---
def get_code_executor(config: 'AgentConfig') -> RestrictedPythonExecutor | DockerCodeExecutor | BaseCodeExecutor | None:
    """Creates a code executor instance based on configuration."""
    executor_type = config.code_executor_type
    executor_config = config.code_executor_config or {}

    if executor_type == "restricted":
        if not RESTRICTEDPYTHON_AVAILABLE:
            logger.error("RestrictedPython executor configured but library not installed. Code execution disabled.")
            return None
        return RestrictedPythonExecutor(**executor_config)
    elif executor_type == "docker":
        if not DOCKER_AVAILABLE:
            logger.error("Docker executor configured but library not installed or Docker not running. Code execution disabled.")
            return None
        try:
            return DockerCodeExecutor(**executor_config)
        except Exception as e:
            logger.error(f"Failed to initialize DockerCodeExecutor: {e}. Code execution disabled.")
            return None
    elif executor_type == "none":
        logger.info("Code execution explicitly disabled in configuration.")
        return None
    elif executor_type and ADK_EXEC_AVAILABLE and isinstance(executor_type, BaseCodeExecutor):
        # Allow passing a pre-configured ADK executor instance
        logger.info(f"Using pre-configured ADK code executor: {type(executor_type).__name__}")
        return executor_type
    else:
        logger.warning(f"Unknown or unsupported code_executor_type: '{executor_type}'. Code execution disabled.")
        return None
