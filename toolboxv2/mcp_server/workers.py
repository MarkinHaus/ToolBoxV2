"""
ToolBoxV2 MCP Server - Workers
==============================
Stateless logic handlers for tool execution
Following ToolBox V2 Architecture Guidelines
"""

import asyncio
import contextlib
import io
import json
import logging
import sys
import traceback
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from .models import ToolResult, ResponseFormat
from .managers import PythonContextManager, CacheManager

logger = logging.getLogger("mcp.workers")


# =============================================================================
# SAFE IO CONTEXT
# =============================================================================


class MCPSafeIO:
    """
    Redirects stdout to stderr to prevent breaking JSON-RPC over stdio.

    In stdio mode, sys.stdout is the exclusive channel for JSON-RPC messages.
    Any print() calls would corrupt the protocol. This context manager
    redirects all output to stderr where it appears in logs/inspector.
    """

    def __init__(self):
        self._stdout = None

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = sys.stderr
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stdout:
            sys.stdout = self._stdout
        return False  # Don't suppress exceptions


# =============================================================================
# PYTHON EXECUTION WORKER
# =============================================================================


class PythonWorker:
    """
    Secure Python code execution with persistent state.

    Features:
    - Uses exec() for full statement support (not eval())
    - Persistent globals across calls
    - Stdout/stderr capture
    - Timeout protection
    - ToolBox app integration
    """

    def __init__(self, context_manager: PythonContextManager):
        self.ctx_mgr = context_manager
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="python")

    async def execute(
        self, code: str, app: Any, timeout: int = 30, capture_output: bool = True
    ) -> ToolResult:
        """
        Execute Python code with full exec() support.

        Args:
            code: Python code to execute
            app: ToolBoxV2 App instance
            timeout: Execution timeout in seconds
            capture_output: Whether to capture stdout/stderr

        Returns:
            ToolResult with execution output
        """
        start_time = time.time()
        # auto print wrapper if onle line
        if "\n" not in code:
            code = f"print({code})"

        try:
            # Get persistent context
            exec_globals = await self.ctx_mgr.get_context(app)

            # Prepare output buffer
            output_buffer = io.StringIO()

            # Run in thread pool with timeout
            loop = asyncio.get_running_loop()

            async def _execute():
                def _sync_exec():
                    result = None

                    with MCPSafeIO():
                        if capture_output:
                            with contextlib.redirect_stdout(output_buffer):
                                with contextlib.redirect_stderr(output_buffer):
                                    # Try exec first for statements
                                    try:
                                        exec(code, exec_globals, exec_globals)
                                    except SyntaxError:
                                        # Might be an expression, try eval
                                        try:
                                            result = eval(
                                                code, exec_globals, exec_globals
                                            )
                                            if result is not None:
                                                output_buffer.write(str(result))
                                        except:
                                            raise
                        else:
                            exec(code, exec_globals, exec_globals)

                    return result

                return await loop.run_in_executor(self._executor, _sync_exec)

            await asyncio.wait_for(_execute(), timeout=timeout)

            # Update persistent context with new variables
            await self.ctx_mgr.update_context(exec_globals)
            await self.ctx_mgr.increment_count()

            # Get output
            output = output_buffer.getvalue()
            if not output.strip():
                output = "✅ Code executed successfully (no output)"

            execution_time = time.time() - start_time

            return ToolResult(success=True, content=output, execution_time=execution_time)

        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                content=f"⏱️ Execution timed out after {timeout}s",
                execution_time=timeout,
                error="TimeoutError",
            )
        except Exception as e:
            tb = traceback.format_exc()
            return ToolResult(
                success=False,
                content=f"❌ Execution error:\n```\n{tb}\n```",
                execution_time=time.time() - start_time,
                error=str(e),
            )

    def close(self):
        """Cleanup executor."""
        self._executor.shutdown(wait=False)


# =============================================================================
# DOCUMENTATION WORKER
# =============================================================================


class DocsWorker:
    """
    Documentation system interface (v2.1 compatible).

    Features:
    - Query caching
    - Multiple format support
    - Source code lookup
    - Task context generation
    """

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager

    async def reader(
        self,
        app: Any,
        query: Optional[str] = None,
        section_id: Optional[str] = None,
        file_path: Optional[str] = None,
        tags: Optional[List[str]] = None,
        format_type: str = "markdown",
        max_results: int = 20,
        use_cache: bool = True,
    ) -> ToolResult:
        """Read documentation with optional caching."""
        start_time = time.time()

        # Check if docs system available
        if not hasattr(app, "docs_reader"):
            return ToolResult(
                success=False,
                content="❌ Documentation system not available. Update ToolBoxV2 to v2.1+",
                execution_time=time.time() - start_time,
                error="DocsNotAvailable",
            )

        # Build cache key
        cache_key = None
        if use_cache:
            cache_key = self.cache.make_key(
                {
                    "query": query,
                    "section_id": section_id,
                    "file_path": file_path,
                    "tags": tags,
                    "format": format_type,
                    "max": max_results,
                }
            )

            cached = await self.cache.get(cache_key)
            if cached:
                return ToolResult(
                    success=True,
                    content=cached,
                    execution_time=time.time() - start_time,
                    cached=True,
                )

        try:
            with MCPSafeIO():
                result = await app.docs_reader(
                    query=query,
                    section_id=section_id,
                    file_path=file_path,
                    tags=tags,
                    format_type=format_type,
                    max_results=max_results,
                )

            # Format output
            if isinstance(result, dict):
                if "error" in result:
                    return ToolResult(
                        success=False,
                        content=f"❌ {result['error']}",
                        execution_time=time.time() - start_time,
                        error=result["error"],
                    )

                if format_type == "markdown" and "content" in result:
                    content = result["content"]
                else:
                    content = json.dumps(
                        result, indent=2, ensure_ascii=False, default=str
                    )
            else:
                content = str(result)

            # Cache successful results
            if cache_key and use_cache:
                await self.cache.set(cache_key, content)

            return ToolResult(
                success=True, content=content, execution_time=time.time() - start_time
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Documentation error: {e}",
                execution_time=time.time() - start_time,
                error=str(e),
            )

    async def writer(
        self,
        app: Any,
        action: str,
        file_path: Optional[str] = None,
        section_title: Optional[str] = None,
        content: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Write/update documentation."""
        start_time = time.time()

        if not hasattr(app, "docs_writer"):
            return ToolResult(
                success=False,
                content="❌ Documentation writer not available",
                execution_time=time.time() - start_time,
                error="DocsWriterNotAvailable",
            )

        try:
            with MCPSafeIO():
                result = await app.docs_writer(
                    action=action,
                    file_path=file_path,
                    section_title=section_title,
                    content=content,
                    **kwargs,
                )

            if isinstance(result, dict) and "error" in result:
                return ToolResult(
                    success=False,
                    content=f"❌ {result['error']}",
                    execution_time=time.time() - start_time,
                    error=result["error"],
                )

            # Invalidate related cache entries
            await self.cache.clear()  # Simple approach - clear all

            content = (
                json.dumps(result, indent=2, default=str)
                if isinstance(result, dict)
                else str(result)
            )

            return ToolResult(
                success=True,
                content=f"✅ Documentation {action} completed:\n```json\n{content}\n```",
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Documentation writer error: {e}",
                execution_time=time.time() - start_time,
                error=str(e),
            )

    async def lookup(
        self,
        app: Any,
        element_name: str,
        file_path: Optional[str] = None,
        element_type: Optional[str] = None,
        max_results: int = 25,
        include_code: bool = True,
    ) -> ToolResult:
        """Look up source code elements."""
        start_time = time.time()

        if not hasattr(app, "docs_lookup"):
            return ToolResult(
                success=False,
                content="❌ Source code lookup not available",
                execution_time=time.time() - start_time,
                error="LookupNotAvailable",
            )

        try:
            with MCPSafeIO():
                result = await app.docs_lookup(
                    name=element_name,
                    file_path=file_path,
                    element_type=element_type,
                    max_results=max_results,
                    include_code=include_code,
                )

            if isinstance(result, dict) and "error" in result:
                return ToolResult(
                    success=False,
                    content=f"❌ {result['error']}",
                    execution_time=time.time() - start_time,
                    error=result["error"],
                )

            matches = result.get("matches", []) if isinstance(result, dict) else []
            content = f"Found {len(matches)} matches for '{element_name}':\n\n"
            content += json.dumps(result, indent=2, default=str)

            return ToolResult(
                success=True, content=content, execution_time=time.time() - start_time
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Lookup error: {e}",
                execution_time=time.time() - start_time,
                error=str(e),
            )

    async def get_task_context(
        self, app: Any, files: List[str], intent: str
    ) -> ToolResult:
        """Get optimized context for an editing task (Graph-based)."""
        start_time = time.time()

        if not hasattr(app, "get_task_context"):
            return ToolResult(
                success=False,
                content="❌ Task context engine not available. Update ToolBoxV2.",
                execution_time=time.time() - start_time,
                error="TaskContextNotAvailable",
            )

        try:
            with MCPSafeIO():
                result = await app.get_task_context(files=files, intent=intent)

            content = json.dumps(result, indent=2, default=str)

            return ToolResult(
                success=True, content=content, execution_time=time.time() - start_time
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Task context error: {e}",
                execution_time=time.time() - start_time,
                error=str(e),
            )


# =============================================================================
# TOOLBOX EXECUTION WORKER
# =============================================================================


class ToolboxWorker:
    """
    Generic ToolBox module execution.

    Features:
    - Any module/function execution
    - Result handling
    - Timeout protection
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="toolbox")

    async def execute(
        self,
        app: Any,
        module_name: str,
        function_name: str,
        args: Optional[List] = None,
        kwargs: Optional[Dict] = None,
        get_results: bool = False,
        timeout: int = 30,
    ) -> ToolResult:
        """Execute a ToolBox module function."""
        start_time = time.time()

        if not app:
            return ToolResult(
                success=False,
                content="❌ ToolBox not initialized",
                execution_time=time.time() - start_time,
                error="NotInitialized",
            )

        try:
            with MCPSafeIO():
                result = await asyncio.wait_for(
                    app.a_run_any(
                        (module_name, function_name),
                        args_=args or [],
                        get_results=get_results,
                        **(kwargs or {}),
                    ),
                    timeout=timeout,
                )

            # Format result
            if get_results and hasattr(result, "as_dict"):
                result_text = json.dumps(result.as_dict(), indent=2, default=str)
            else:
                result_text = str(result)

            content = f"**Executed:** `{module_name}.{function_name}`\n\n**Result:**\n```\n{result_text}\n```"

            return ToolResult(
                success=True, content=content, execution_time=time.time() - start_time
            )

        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                content=f"⏱️ Execution timed out after {timeout}s",
                execution_time=timeout,
                error="TimeoutError",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Execution error: {e}\n\n```\n{traceback.format_exc()}\n```",
                execution_time=time.time() - start_time,
                error=str(e),
            )

    def close(self):
        """Cleanup executor."""
        self._executor.shutdown(wait=False)


# =============================================================================
# SYSTEM INFO WORKER
# =============================================================================


class SystemWorker:
    """
    System information and status.
    """

    @staticmethod
    async def get_status(
        app: Any,
        include_modules: bool = True,
        include_flows: bool = True,
        include_functions: bool = False,
        metrics: Optional[Dict] = None,
    ) -> ToolResult:
        """Get comprehensive system status."""
        start_time = time.time()

        if not app:
            return ToolResult(
                success=False,
                content="❌ ToolBox not initialized",
                execution_time=time.time() - start_time,
                error="NotInitialized",
            )

        try:
            status = {
                "🏗️ System": {
                    "app_id": getattr(app, "id", "unknown"),
                    "version": getattr(app, "version", "unknown"),
                    "debug_mode": getattr(app, "debug", False),
                    "alive": getattr(app, "alive", False),
                }
            }

            if include_modules:
                modules = list(getattr(app, "functions", {}).keys())
                status["📦 Modules"] = {"count": len(modules), "list": modules}

            if include_flows:
                flows = list(getattr(app, "flows", {}).keys())
                status["🔄 Flows"] = {"count": len(flows), "list": flows}

            if include_functions and include_modules:
                func_details = {}
                for mod_name, mod_funcs in getattr(app, "functions", {}).items():
                    if isinstance(mod_funcs, dict):
                        func_details[mod_name] = list(mod_funcs.keys())
                status["🔧 Functions"] = func_details

            # Add docs status
            status["📚 Documentation"] = {
                "docs_reader": hasattr(app, "docs_reader"),
                "docs_writer": hasattr(app, "docs_writer"),
                "docs_lookup": hasattr(app, "docs_lookup"),
                "task_context": hasattr(app, "get_task_context"),
            }

            if metrics:
                status["⚡ Performance"] = metrics

            content = "# 🚀 ToolBoxV2 System Status\n\n"
            content += json.dumps(status, indent=2, ensure_ascii=False, default=str)

            return ToolResult(
                success=True, content=content, execution_time=time.time() - start_time
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Status error: {e}",
                execution_time=time.time() - start_time,
                error=str(e),
            )

    @staticmethod
    async def get_info(
        app: Any,
        info_type: str,
        target: Optional[str] = None,
        include_examples: bool = False,
    ) -> ToolResult:
        """Get system information."""
        start_time = time.time()

        try:
            if info_type == "modules":
                modules = list(getattr(app, "functions", {}).keys())
                content = "# 📦 Available Modules\n\n"
                for mod in sorted(modules):
                    content += f"- **{mod}**\n"

                if include_examples:
                    content += "\n## 💡 Usage Example\n"
                    content += "```\ntoolbox_execute(module='module_name', function='function_name')\n```"

            elif info_type == "functions" and target:
                funcs = getattr(app, "functions", {}).get(target, {})
                if isinstance(funcs, dict):
                    content = f"# 🔧 Functions in {target}\n\n"
                    for func_name in sorted(funcs.keys()):
                        content += f"- `{func_name}`\n"
                else:
                    content = f"Module '{target}' not found or has no functions"

            elif info_type == "flows":
                flows = list(getattr(app, "flows", {}).keys())
                content = "# 🔄 Available Flows\n\n"
                for flow in sorted(flows):
                    content += f"- **{flow}**\n"

            elif info_type == "python_guide":
                from .models import PYTHON_EXECUTION_TEMPLATE

                content = PYTHON_EXECUTION_TEMPLATE

            elif info_type == "performance_guide":
                from .models import PERFORMANCE_GUIDE_TEMPLATE

                content = PERFORMANCE_GUIDE_TEMPLATE.format(
                    cache_ttl=300,
                    max_cache_size=100,
                    requests=0,
                    avg_time=0.0,
                    hit_rate=0.0,
                )

            else:
                content = f"Unknown info type: {info_type}"

            return ToolResult(
                success=True, content=content, execution_time=time.time() - start_time
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Info error: {e}",
                execution_time=time.time() - start_time,
                error=str(e),
            )
