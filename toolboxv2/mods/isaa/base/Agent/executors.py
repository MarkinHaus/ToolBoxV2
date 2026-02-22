"""
Code Executor - MockIPython Integration for FlowAgent

Uses the proven MockIPython.run_cell() for code execution:
- LocalCodeExecutor - uses MockIPython directly with agent tools
- DockerCodeExecutor - uses MockIPython in toolboxv2:latest container

Author: FlowAgent V2
"""

import asyncio
import io
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# =============================================================================
# MOCK IPYTHON IMPORTS
# =============================================================================
from toolboxv2.mods.isaa.CodingAgent.live import (
    MockIPython,
    VirtualFileSystem,
    VirtualEnvContext,
    TeeStream,
)


# =============================================================================
# LOCAL CODE EXECUTOR (with MockIPython)
# =============================================================================

class LocalCodeExecutor:
    """
    Local code executor using MockIPython.

    Provides:
    - Top-level await support
    - Async code execution
    - Agent tool access via 'tools' proxy
    - VFS integration
    - Output capturing
    """

    def __init__(self, agent=None, session_dir: Path = None, timeout: int = 30):
        """
        Initialize LocalCodeExecutor with MockIPython.

        Args:
            agent: FlowAgent instance (for tool access)
            session_dir: Directory for session files (default: appdata)
            timeout: Execution timeout in seconds
        """
        from toolboxv2 import get_app

        self.agent = agent
        self.timeout = timeout

        # Create session directory
        if session_dir is None:
            app = get_app()
            session_dir = Path(app.appdata) / '.code_executor_sessions'

        self._session_dir = session_dir / str(id(self))
        self._session_dir.mkdir(parents=True, exist_ok=True)

        # Create MockIPython instance
        self._mock_ipython = MockIPython(
            _session_dir=self._session_dir,
            auto_remove=True  # Cleanup temp files on reset
        )

        # Inject agent tools into namespace
        if agent and hasattr(agent, 'tool_manager'):
            self._mock_ipython.user_ns['tools'] = AgentToolProxy(agent)
            self._mock_ipython.user_ns['agent'] = agent

    async def execute(self, code: str) -> dict:
        """
        Execute Python code using MockIPython.run_cell().

        Args:
            code: Python code to execute

        Returns:
            Dict with success, output, error keys
        """
        try:
            # Run code with MockIPython (handles top-level await, async, etc.)
            mock_result = await self._mock_ipython.run_cell(code, live_output=True)

            # MockIPython returns a formatted string like:
            # "value\nstdout: actual_output\nstderr: error"
            # Or just the value for simple expressions

            output = ""

            if isinstance(mock_result, str):
                # Parse the formatted output
                lines = mock_result.split('\n')
                for line in lines:
                    if line.startswith('stdout:'):
                        output += line[7:] + "\n"
                    elif line.startswith('stderr:'):
                        # Error case handled below
                        pass
                    elif line.startswith('Error executing code:'):
                        # Error from MockIPython
                        return {
                            'success': False,
                            'output': '',
                            'error': mock_result
                        }
                    else:
                        # Regular output
                        output += line + "\n"

            elif isinstance(mock_result, dict):
                # Dict format (direct access)
                stdout_val = mock_result.get('stdout', '')
                if stdout_val:
                    output = stdout_val

                result_val = mock_result.get('result')
                if result_val and result_val != "stdout":
                    if output:
                        output += "\n"
                    output += str(result_val)
            else:
                # Direct value
                if mock_result and mock_result != "No executable code":
                    output = str(mock_result)

            return {
                'success': True,
                'output': output.strip(),
                'error': None
            }

        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            }

    def get_namespace(self) -> dict:
        """Get current execution namespace"""
        return self._mock_ipython.get_namespace()

    def reset(self):
        """Reset the interpreter state"""
        self._mock_ipython.reset()


# =============================================================================
# PROXY FOR AGENT TOOLS
# =============================================================================

class AgentToolProxy:
    """
    Proxy for accessing agent tools from executed code.

    Usage in code:
        result = tools.vfs_list("/")
        result = tools.vfs_read("/main.py")
    """

    def __init__(self, agent):
        self._agent = agent

    def __getattr__(self, tool_name: str):
        """Get a tool function by name."""
        tool_func = self._agent.tool_manager.get_function(tool_name)
        if tool_func is None:
            raise AttributeError(f"Tool '{tool_name}' not found")

        # Return wrapper that handles sync/async
        def wrapper(*args, **kwargs):
            result = tool_func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # In async context - need to create task
                        # For now, raise error with hint
                        raise RuntimeError(
                            f"Tool '{tool_name}' is async. Use 'await tools.{tool_name}()' "
                            "in async context."
                        )
                    return loop.run_until_complete(result)
                except RuntimeError:
                    # No event loop - create new one
                    return asyncio.run(result)
            return result

        return wrapper

    def __dir__(self):
        """List available tools."""
        return list(self._agent.tool_manager._registry.keys())

    def __repr__(self):
        available = list(self._agent.tool_manager._registry.keys())
        return f"<AgentToolProxy: {len(available)} tools>"


# =============================================================================
# DOCKER CODE EXECUTOR (with MockIPython in Container)
# =============================================================================

class DockerCodeExecutor:
    """
    Docker-based code executor using toolboxv2:latest image.

    Runs MockIPython.run_cell() inside the container.
    VFS files are synced to /workspace (or custom working_dir).
    """

    def __init__(self, agent=None, working_dir: str = None, timeout: int = 30):
        """
        Initialize DockerCodeExecutor.

        Args:
            agent: FlowAgent instance (for VFS sync)
            working_dir: Optional custom working directory (default: temp dir)
            timeout: Execution timeout in seconds
        """
        self.agent = agent
        self.timeout = timeout
        self._container_id: str | None = None
        self._workspace_dir: str

        # Use temp dir if no working_dir specified
        if working_dir:
            self._workspace_dir = working_dir
        else:
            import tempfile
            self._workspace_dir = f"/tmp/toolbox_exec_{id(self)}"

    async def execute(self, code: str) -> dict:
        """
        Execute Python code in Docker container using MockIPython.

        Args:
            code: Python code to execute

        Returns:
            Dict with success, output, error keys
        """
        # Ensure container is running
        if not self._container_id:
            create_result = await self._ensure_container()
            if not create_result.get("success"):
                return {
                    'success': False,
                    'output': '',
                    'error': create_result.get("error", "Failed to create container")
                }

        # Sync VFS to container if available
        if self.agent and hasattr(self.agent, 'active_session'):
            from toolboxv2.mods.isaa.base.Agent.session_manager import SessionManager
            session = SessionManager.instance().get(self.agent.active_session)
            if session and hasattr(session, 'vfs'):
                await self._sync_vfs(session.vfs)

        # Execute code using a simplified script that handles top-level await
        # We inline the execution logic to avoid toolboxv2 import issues in container

        # Escape the code for Python string embedding
        escaped_code = code.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

        exec_script = f'''
import asyncio
import sys
import io
import json
import ast

code_to_exec = """{escaped_code}"""

async def run_code():
    """Execute code with top-level await support"""
    result = None
    output = ""

    # Parse to detect top-level await
    try:
        tree = ast.parse(code_to_exec)
        has_await = any(isinstance(node, ast.Await) for node in ast.walk(tree))

        # Capture output
        old_stdout = sys.stdout
        stdout_capture = io.StringIO()
        sys.stdout = stdout_capture

        try:
            if has_await:
                # Simple wrapper for async code - just exec with top-level await support
                # Python 3.8+ allows top-level await in async functions
                import copy

                # Build wrapper function
                wrapper_code = "async def __wrapper():\\n"
                for line in code_to_exec.split('\\n'):
                    wrapper_code += "    " + line + "\\n"

                exec(wrapper_code, {{}})

                # Execute the async function
                await __wrapper__()

            else:
                # No await - use regular exec/eval
                # Check if last statement is an expression
                if isinstance(tree.body[-1], ast.Expr):
                    # Exec all but last, eval last
                    if len(tree.body) > 1:
                        exec_code = ast.Module(body=tree.body[:-1], type_ignores=[])
                        exec(compile(exec_code, "<exec>", "exec"), {{}})
                    eval_code = ast.Expression(body=tree.body[-1].value)
                    result = eval(compile(eval_code, "<eval>", "eval"), {{}})
                else:
                    exec(code_to_exec, {{}})

            output = stdout_capture.getvalue()

        finally:
            sys.stdout = old_stdout

        # Add result to output if exists
        if result is not None:
            if output:
                output += "\\n"
            output += str(result)

        return {{"success": True, "output": output, "error": None}}

    except Exception as e:
        import traceback
        error_output = stdout_capture.getvalue() if 'stdout_capture' in locals() else ""
        return {{"success": False, "output": error_output, "error": str(e)}}

# Run and output JSON
result = asyncio.run(run_code())
print(json.dumps(result))
'''

        cmd = [
            "docker", "exec", "-i",
            self._container_id,
            "python", "-c", exec_script
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )

            output = stdout.decode('utf-8', errors='replace')

            # Parse JSON result
            for line in reversed(output.split('\n')):
                line = line.strip()
                if line.startswith('{'):
                    try:
                        result = json.loads(line)
                        return result
                    except json.JSONDecodeError:
                        continue

            # Fallback: raw output
            return {
                'success': process.returncode == 0,
                'output': output,
                'error': stderr.decode('utf-8', errors='replace') if process.returncode != 0 else None
            }

        except asyncio.TimeoutError:
            return {
                'success': False,
                'output': '',
                'error': f"Execution timed out after {self.timeout} seconds"
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': str(e)
            }

    def _parse_mock_output(self, output: str) -> str:
        """Parse MockIPython output format."""
        lines = output.split('\n')
        result = ""

        for line in lines:
            if line.startswith('stdout:'):
                result += line[7:] + "\n"
            elif line.startswith('stderr:'):
                pass  # Skip stderr
            elif line.startswith('Error executing code:'):
                # This is an error, will be handled separately
                pass
            else:
                result += line + "\n"

        return result.strip()

    async def _ensure_container(self) -> dict:
        """Ensure container is running."""
        if self._container_id:
            # Check if container still exists
            try:
                process = await asyncio.create_subprocess_exec(
                    "docker", "inspect", self._container_id,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
                if process.returncode == 0:
                    return {"success": True}
            except Exception:
                self._container_id = None

        # Create new container
        import uuid
        name = f"toolbox_exec_{uuid.uuid4().hex[:8]}"

        cmd = [
            "docker", "run", "-d",
            "--name", name,
            "-w", self._workspace_dir,
            "toolboxv2:latest",
            "tail", "-f", "/dev/null"
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=60
            )

            if process.returncode != 0:
                err_msg = stderr.decode('utf-8', errors='replace')
                return {"success": False, "error": f"Failed to create container: {err_msg}"}

            self._container_id = stdout.decode().strip()
            return {"success": True}

        except asyncio.TimeoutError:
            return {"success": False, "error": "Timeout creating container"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _sync_vfs(self, vfs):
        """Sync VFS files to container workspace."""
        import tarfile

        tar_buffer = io.BytesIO()

        with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
            for path, vfs_file in vfs.files.items():
                if vfs_file.readonly:
                    continue

                rel_path = path.lstrip('/')
                if not rel_path:
                    continue

                content = vfs_file.content.encode('utf-8')
                tarinfo = tarfile.TarInfo(name=rel_path)
                tarinfo.size = len(content)
                tar.addfile(tarinfo, io.BytesIO(content))

        tar_buffer.seek(0)

        # Create workspace dir in container
        mkdir_cmd = ["docker", "exec", self._container_id, "mkdir", "-p", self._workspace_dir]
        mkdir_proc = await asyncio.create_subprocess_exec(
            *mkdir_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await mkdir_proc.communicate()

        # Copy tar to container
        process = await asyncio.create_subprocess_exec(
            "docker", "cp", "-",
            f"{self._container_id}:{self._workspace_dir}",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate(input=tar_buffer.read())

    async def cleanup(self):
        """Remove container."""
        if self._container_id:
            try:
                process = await asyncio.create_subprocess_exec(
                    "docker", "rm", "-f", self._container_id,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
            except Exception:
                pass
            self._container_id = None


# =============================================================================
# CONFIG
# =============================================================================

class DockerConfig:
    """Docker configuration for code executor (legacy, kept for compatibility)."""
    base_image: str = "toolboxv2:latest"
    workspace_dir: str = "/tmp/toolbox_workspace"  # Default temp dir
    memory_limit: str = "1g"
    cpu_limit: float = 0.5
    timeout_seconds: int = 60

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


# =============================================================================
# TOOL DEFINITIONS FOR REGISTRATION
# =============================================================================

def create_local_code_exec_tool(agent) -> dict:
    """
    Create local code execution tool for agent registration.

    Args:
        agent: FlowAgent instance

    Returns:
        Tool definition dict for add_tool()
    """
    executor = LocalCodeExecutor(agent=agent)

    async def exec_code(code: str) -> dict:
        """
        Execute Python code locally using MockIPython.

        Supports top-level await, async functions, and agent tool access.

        Args:
            code: Python code to execute

        Returns:
            Execution result with output and success status
        """
        return await executor.execute(code)

    return {
        "tool_func": exec_code,
        "name": "exec_code",
        "description": "Execute Python code (MockIPython with top-level await). Access agent tools via 'tools.tool_name()'.",
        "category": ["code", "execution"],
        "flags": {"local_execution": True},
        "is_async": True,
    }


def create_docker_code_exec_tool(agent, working_dir: str = None) -> dict:
    """
    Create Docker code execution tool for agent registration.

    Args:
        agent: FlowAgent instance
        working_dir: Optional working directory in container (default: temp dir)

    Returns:
        Tool definition dict for add_tool()
    """
    executor = DockerCodeExecutor(agent=agent, working_dir=working_dir)

    async def exec_code(code: str) -> dict:
        """
        Execute Python code in Docker container (toolboxv2:latest).

        Supports top-level await and async functions.
        VFS files are synced to the workspace directory.

        Args:
            code: Python code to execute

        Returns:
            Execution result with output and success status
        """
        return await executor.execute(code)

    return {
        "tool_func": exec_code,
        "name": "exec_code_docker",
        "description": "Execute Python code in Docker container (toolboxv2:latest, top-level await support)",
        "category": ["code", "execution", "docker"],
        "flags": {"requires_docker": True},
        "is_async": True,
    }


# =============================================================================
# HELPER FOR SESSION INIT
# =============================================================================

def register_code_exec_tools(agent, docker=False):
    """
    Register both code execution tools for a session.

    Call this from init_session_tools().

    Args:
        agent: FlowAgent instance
        docker: with docker ( in production )
    Returns:
        List of registered tool entries
    """
    local_tool = create_local_code_exec_tool(agent)

    agent.add_tool(**local_tool)


    if docker:
        docker_tool = create_docker_code_exec_tool(agent)
        agent.add_tool(**docker_tool)
        return [local_tool, docker_tool]
    return [local_tool]
