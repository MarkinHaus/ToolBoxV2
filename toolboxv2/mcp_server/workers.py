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

from toolboxv2.mcp_server.models import ToolResult, ResponseFormat
from toolboxv2.mcp_server.managers import PythonContextManager, CacheManager

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
        self,
        code: str,
        app: Any,
        timeout: int = 30,
        capture_output: bool = True
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

        if '\n' not in code:
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
                                            result = eval(code, exec_globals, exec_globals)
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

            return ToolResult(
                success=True,
                content=output,
                execution_time=execution_time
            )

        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                content=f"⏱️ Execution timed out after {timeout}s",
                execution_time=timeout,
                error="TimeoutError"
            )
        except Exception as e:
            tb = traceback.format_exc()
            return ToolResult(
                success=False,
                content=f"❌ Execution error:\n```\n{tb}\n```",
                execution_time=time.time() - start_time,
                error=str(e)
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
        use_cache: bool = True
    ) -> ToolResult:
        """Read documentation with optional caching."""
        start_time = time.time()

        # Check if docs system available
        if not hasattr(app, 'docs_reader'):
            return ToolResult(
                success=False,
                content="❌ Documentation system not available. Update ToolBoxV2 to v2.1+",
                execution_time=time.time() - start_time,
                error="DocsNotAvailable"
            )

        # Build cache key
        cache_key = None
        if use_cache:
            cache_key = self.cache.make_key({
                "query": query,
                "section_id": section_id,
                "file_path": file_path,
                "tags": tags,
                "format": format_type,
                "max": max_results
            })

            cached = await self.cache.get(cache_key)
            if cached:
                return ToolResult(
                    success=True,
                    content=cached,
                    execution_time=time.time() - start_time,
                    cached=True
                )

        try:
            with MCPSafeIO():
                result = await app.docs_reader(
                    query=query,
                    section_id=section_id,
                    file_path=file_path,
                    tags=tags,
                    format_type=format_type,
                    max_results=max_results
                )

            # Format output
            if isinstance(result, dict):
                if 'error' in result:
                    return ToolResult(
                        success=False,
                        content=f"❌ {result['error']}",
                        execution_time=time.time() - start_time,
                        error=result['error']
                    )

                if format_type == "markdown" and 'content' in result:
                    content = result['content']
                else:
                    content = json.dumps(result, indent=2, ensure_ascii=False, default=str)
            else:
                content = str(result)

            # Cache successful results
            if cache_key and use_cache:
                await self.cache.set(cache_key, content)

            return ToolResult(
                success=True,
                content=content,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Documentation error: {e}",
                execution_time=time.time() - start_time,
                error=str(e)
            )

    async def writer(
        self,
        app: Any,
        action: str,
        file_path: Optional[str] = None,
        section_title: Optional[str] = None,
        content: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Write/update documentation."""
        start_time = time.time()

        if not hasattr(app, 'docs_writer'):
            return ToolResult(
                success=False,
                content="❌ Documentation writer not available",
                execution_time=time.time() - start_time,
                error="DocsWriterNotAvailable"
            )

        try:
            with MCPSafeIO():
                result = await app.docs_writer(
                    action=action,
                    file_path=file_path,
                    section_title=section_title,
                    content=content,
                    **kwargs
                )

            if isinstance(result, dict) and 'error' in result:
                return ToolResult(
                    success=False,
                    content=f"❌ {result['error']}",
                    execution_time=time.time() - start_time,
                    error=result['error']
                )

            # Invalidate related cache entries
            await self.cache.clear()  # Simple approach - clear all

            content = json.dumps(result, indent=2, default=str) if isinstance(result, dict) else str(result)

            return ToolResult(
                success=True,
                content=f"✅ Documentation {action} completed:\n```json\n{content}\n```",
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Documentation writer error: {e}",
                execution_time=time.time() - start_time,
                error=str(e)
            )

    async def lookup(
        self,
        app: Any,
        element_name: str,
        file_path: Optional[str] = None,
        element_type: Optional[str] = None,
        max_results: int = 25,
        include_code: bool = True
    ) -> ToolResult:
        """Look up source code elements."""
        start_time = time.time()

        if not hasattr(app, 'docs_lookup'):
            return ToolResult(
                success=False,
                content="❌ Source code lookup not available",
                execution_time=time.time() - start_time,
                error="LookupNotAvailable"
            )

        try:
            with MCPSafeIO():
                result = await app.docs_lookup(
                    name=element_name,
                    file_path=file_path,
                    element_type=element_type,
                    max_results=max_results,
                    include_code=include_code
                )

            if isinstance(result, dict) and 'error' in result:
                return ToolResult(
                    success=False,
                    content=f"❌ {result['error']}",
                    execution_time=time.time() - start_time,
                    error=result['error']
                )

            matches = result.get('matches', []) if isinstance(result, dict) else []
            content = f"Found {len(matches)} matches for '{element_name}':\n\n"
            content += json.dumps(result, indent=2, default=str)

            return ToolResult(
                success=True,
                content=content,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Lookup error: {e}",
                execution_time=time.time() - start_time,
                error=str(e)
            )

    async def get_task_context(
        self,
        app: Any,
        files: List[str],
        intent: str
    ) -> ToolResult:
        """Get optimized context for an editing task (Graph-based)."""
        start_time = time.time()

        if not hasattr(app, 'get_task_context'):
            return ToolResult(
                success=False,
                content="❌ Task context engine not available. Update ToolBoxV2.",
                execution_time=time.time() - start_time,
                error="TaskContextNotAvailable"
            )

        try:
            with MCPSafeIO():
                result = await app.get_task_context(
                    files=files,
                    intent=intent
                )

            content = json.dumps(result, indent=2, default=str)

            return ToolResult(
                success=True,
                content=content,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Task context error: {e}",
                execution_time=time.time() - start_time,
                error=str(e)
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
        timeout: int = 30
    ) -> ToolResult:
        """Execute a ToolBox module function."""
        start_time = time.time()

        if not app:
            return ToolResult(
                success=False,
                content="❌ ToolBox not initialized",
                execution_time=time.time() - start_time,
                error="NotInitialized"
            )

        try:
            with MCPSafeIO():
                result = await asyncio.wait_for(
                    app.a_run_any(
                        (module_name, function_name),
                        args_=args or [],
                        get_results=get_results,
                        **(kwargs or {})
                    ),
                    timeout=timeout
                )

            # Format result
            if get_results and hasattr(result, 'as_dict'):
                result_text = json.dumps(result.as_dict(), indent=2, default=str)
            else:
                result_text = str(result)

            content = f"**Executed:** `{module_name}.{function_name}`\n\n**Result:**\n```\n{result_text}\n```"

            return ToolResult(
                success=True,
                content=content,
                execution_time=time.time() - start_time
            )

        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                content=f"⏱️ Execution timed out after {timeout}s",
                execution_time=timeout,
                error="TimeoutError"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Execution error: {e}\n\n```\n{traceback.format_exc()}\n```",
                execution_time=time.time() - start_time,
                error=str(e)
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
        metrics: Optional[Dict] = None
    ) -> ToolResult:
        """Get comprehensive system status."""
        start_time = time.time()

        if not app:
            return ToolResult(
                success=False,
                content="❌ ToolBox not initialized",
                execution_time=time.time() - start_time,
                error="NotInitialized"
            )

        try:
            status = {
                "🏗️ System": {
                    "app_id": getattr(app, 'id', 'unknown'),
                    "version": getattr(app, 'version', 'unknown'),
                    "debug_mode": getattr(app, 'debug', False),
                    "alive": getattr(app, 'alive', False)
                }
            }

            if include_modules:
                modules = list(getattr(app, 'functions', {}).keys())
                status["📦 Modules"] = {
                    "count": len(modules),
                    "list": modules
                }

            if include_flows:
                flows = list(getattr(app, 'flows', {}).keys())
                status["🔄 Flows"] = {
                    "count": len(flows),
                    "list": flows
                }

            if include_functions and include_modules:
                func_details = {}
                for mod_name, mod_funcs in getattr(app, 'functions', {}).items():
                    if isinstance(mod_funcs, dict):
                        func_details[mod_name] = list(mod_funcs.keys())
                status["🔧 Functions"] = func_details

            # Add docs status
            status["📚 Documentation"] = {
                "docs_reader": hasattr(app, 'docs_reader'),
                "docs_writer": hasattr(app, 'docs_writer'),
                "docs_lookup": hasattr(app, 'docs_lookup'),
                "task_context": hasattr(app, 'get_task_context')
            }

            if metrics:
                status["⚡ Performance"] = metrics

            content = "# 🚀 ToolBoxV2 System Status\n\n"
            content += json.dumps(status, indent=2, ensure_ascii=False, default=str)

            return ToolResult(
                success=True,
                content=content,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Status error: {e}",
                execution_time=time.time() - start_time,
                error=str(e)
            )

    @staticmethod
    async def get_info(
        app: Any,
        info_type: str,
        target: Optional[str] = None,
        include_examples: bool = False
    ) -> ToolResult:
        """
        Get comprehensive system information.

        Info types:
        - modules: List all modules with function counts
        - functions: List functions in a module with signatures
        - function_detail: Detailed info about a specific function
        - flows: List all flows with descriptions
        - flow_detail: Detailed info about a specific flow
        - python_guide: Python execution guide
        - performance_guide: Performance optimization guide
        - discovery: Full system discovery
        """
        start_time = time.time()

        try:
            # =================================================================
            # MODULES
            # =================================================================
            if info_type == "modules":
                functions_dict = getattr(app, 'functions', {})

                content = "# 📦 Available Modules\n\n"
                content += f"**Total Modules**: {len(functions_dict)}\n\n"

                # Sort by function count
                module_stats = []
                for mod_name, mod_funcs in functions_dict.items():
                    if isinstance(mod_funcs, dict):
                        func_count = len(mod_funcs)
                        module_stats.append((mod_name, func_count))

                module_stats.sort(key=lambda x: -x[1])  # Sort by count desc

                content += "| Module | Functions |\n"
                content += "|--------|----------:|\n"
                for mod_name, count in module_stats:
                    content += f"| `{mod_name}` | {count} |\n"

                if include_examples:
                    content += "\n## 💡 Usage Examples\n\n"
                    content += "```python\n"
                    content += "# List functions in a module\n"
                    content += "toolbox_info(info_type='functions', target='CloudM')\n\n"
                    content += "# Get function details\n"
                    content += "toolbox_info(info_type='function_detail', target='CloudM.get_user')\n\n"
                    content += "# Execute a function\n"
                    content += "toolbox_execute(module='CloudM', function='get_user', kwargs={'username': 'test'})\n"
                    content += "```"

            # =================================================================
            # FUNCTIONS IN MODULE
            # =================================================================
            elif info_type == "functions":
                if not target:
                    return ToolResult(
                        success=False,
                        content="❌ `target` parameter required for info_type='functions'\n\n"
                                "Usage: `toolbox_info(info_type='functions', target='ModuleName')`",
                        execution_time=time.time() - start_time,
                        error="MissingTarget"
                    )

                functions_dict = getattr(app, 'functions', {})
                mod_funcs = functions_dict.get(target, {})

                if not mod_funcs or not isinstance(mod_funcs, dict):
                    # Try case-insensitive match
                    for mod_name in functions_dict:
                        if mod_name.lower() == target.lower():
                            mod_funcs = functions_dict[mod_name]
                            target = mod_name
                            break

                if not mod_funcs:
                    available = list(functions_dict.keys())[:20]
                    return ToolResult(
                        success=False,
                        content=f"❌ Module `{target}` not found.\n\n"
                                f"**Available modules** (first 20):\n" +
                                "\n".join(f"- `{m}`" for m in sorted(available)),
                        execution_time=time.time() - start_time,
                        error="ModuleNotFound"
                    )

                content = f"# 🔧 Functions in `{target}`\n\n"
                content += f"**Total Functions**: {len(mod_funcs)}\n\n"

                # Group by API availability
                api_funcs = []
                internal_funcs = []

                for func_name, func_data in mod_funcs.items():
                    if isinstance(func_data, dict):
                        is_api = func_data.get('api', False)
                        params = func_data.get('params', [])
                        state = func_data.get('state', False)

                        func_info = {
                            'name': func_name,
                            'params': params,
                            'api': is_api,
                            'state': state,
                            'methods': func_data.get('api_methods', ['GET'])
                        }

                        if is_api:
                            api_funcs.append(func_info)
                        else:
                            internal_funcs.append(func_info)
                    else:
                        internal_funcs.append({'name': func_name, 'params': [], 'api': False})

                if api_funcs:
                    content += "## 🌐 API Functions\n\n"
                    for f in sorted(api_funcs, key=lambda x: x['name']):
                        params_str = ", ".join(f['params'][:5])
                        if len(f['params']) > 5:
                            params_str += ", ..."
                        content += f"- `{f['name']}({params_str})`"
                        if f.get('state'):
                            content += " [stateful]"
                        content += "\n"

                if internal_funcs:
                    content += "\n## 🔒 Internal Functions\n\n"
                    for f in sorted(internal_funcs, key=lambda x: x['name'])[:30]:
                        params_str = ", ".join(f.get('params', [])[:3])
                        if len(f.get('params', [])) > 3:
                            params_str += ", ..."
                        content += f"- `{f['name']}({params_str})`\n"

                    if len(internal_funcs) > 30:
                        content += f"\n*...and {len(internal_funcs) - 30} more*\n"

                if include_examples and api_funcs:
                    content += "\n## 💡 Example Call\n\n"
                    example_func = api_funcs[0]
                    content += "```python\n"
                    content += f"toolbox_execute(\n"
                    content += f"    module='{target}',\n"
                    content += f"    function='{example_func['name']}',\n"
                    if example_func['params']:
                        content += f"    kwargs={{'{example_func['params'][0]}': '...'}}\n"
                    content += ")\n```"

            # =================================================================
            # FUNCTION DETAIL
            # =================================================================
            elif info_type == "function_detail":
                if not target:
                    return ToolResult(
                        success=False,
                        content="❌ `target` parameter required (format: 'Module.function')",
                        execution_time=time.time() - start_time,
                        error="MissingTarget"
                    )

                # Parse target
                if '.' in target:
                    mod_name, func_name = target.split('.', 1)
                else:
                    return ToolResult(
                        success=False,
                        content="❌ Invalid format. Use 'Module.function' format.\n\n"
                                "Example: `toolbox_info(info_type='function_detail', target='CloudM.get_user')`",
                        execution_time=time.time() - start_time,
                        error="InvalidFormat"
                    )

                functions_dict = getattr(app, 'functions', {})
                mod_funcs = functions_dict.get(mod_name, {})

                if not mod_funcs:
                    return ToolResult(
                        success=False,
                        content=f"❌ Module `{mod_name}` not found",
                        execution_time=time.time() - start_time,
                        error="ModuleNotFound"
                    )

                func_data = mod_funcs.get(func_name)
                if not func_data:
                    available = list(mod_funcs.keys())[:10]
                    return ToolResult(
                        success=False,
                        content=f"❌ Function `{func_name}` not found in `{mod_name}`.\n\n"
                                f"**Available functions** (first 10):\n" +
                                "\n".join(f"- `{f}`" for f in sorted(available)),
                        execution_time=time.time() - start_time,
                        error="FunctionNotFound"
                    )

                content = f"# 📋 Function: `{mod_name}.{func_name}`\n\n"

                if isinstance(func_data, dict):
                    # Parameters
                    params = func_data.get('params', [])
                    content += f"## Parameters\n\n"
                    if params:
                        content += "| Parameter | Required |\n"
                        content += "|-----------|----------|\n"
                        for p in params:
                            # First few are usually required
                            required = "Yes" if params.index(p) < 2 else "No"
                            content += f"| `{p}` | {required} |\n"
                    else:
                        content += "*No parameters*\n"

                    # Metadata
                    content += "\n## Metadata\n\n"
                    content += f"- **API Exposed**: {'Yes ✅' if func_data.get('api') else 'No ❌'}\n"
                    content += f"- **Stateful**: {'Yes' if func_data.get('state') else 'No'}\n"
                    content += f"- **API Methods**: {func_data.get('api_methods', ['N/A'])}\n"

                    # Docstring if available
                    func_obj = func_data.get('func')
                    if func_obj and hasattr(func_obj, '__doc__') and func_obj.__doc__:
                        content += f"\n## Documentation\n\n"
                        content += f"```\n{func_obj.__doc__[:500]}\n```\n"

                    content += f"\n## Example Call\n\n"
                    content += "```python\n"
                    content += f"result = toolbox_execute(\n"
                    content += f"    module='{mod_name}',\n"
                    content += f"    function='{func_name}',\n"
                    if params:
                        kwargs_example = ", ".join(f"'{p}': '...'" for p in params[:3])
                        content += f"    kwargs={{{kwargs_example}}},\n"
                    content += f"    get_results=True\n"
                    content += ")\n```"
                else:
                    content += "*Limited information available for this function*"

            # =================================================================
            # FLOWS
            # =================================================================
            elif info_type == "flows":
                flows = getattr(app, 'flows', {})

                content = "# 🔄 Available Flows\n\n"
                content += f"**Total Flows**: {len(flows)}\n\n"

                if not flows:
                    content += "*No flows registered*\n"
                else:
                    for flow_name in sorted(flows.keys()):
                        flow_func = flows[flow_name]

                        # Try to get docstring
                        doc = ""
                        if hasattr(flow_func, '__doc__') and flow_func.__doc__:
                            doc = flow_func.__doc__.split('\n')[0][:80]

                        content += f"- **`{flow_name}`**"
                        if doc:
                            content += f" - {doc}"
                        content += "\n"

                if include_examples:
                    content += "\n## 💡 Usage\n\n"
                    content += "```python\n"
                    content += "# Start a flow\n"
                    content += "flow_start(flow_name='flow_name')\n\n"
                    content += "# Provide input when waiting\n"
                    content += "flow_input(session_id='...', input='your input')\n\n"
                    content += "# Check status\n"
                    content += "flow_status(session_id='...')\n"
                    content += "```"

            # =================================================================
            # FLOW DETAIL
            # =================================================================
            elif info_type == "flow_detail":
                if not target:
                    return ToolResult(
                        success=False,
                        content="❌ `target` parameter required for flow detail",
                        execution_time=time.time() - start_time,
                        error="MissingTarget"
                    )

                flows = getattr(app, 'flows', {})
                flow_func = flows.get(target)

                if not flow_func:
                    available = list(flows.keys())[:15]
                    return ToolResult(
                        success=False,
                        content=f"❌ Flow `{target}` not found.\n\n"
                                f"**Available flows**:\n" +
                                "\n".join(f"- `{f}`" for f in sorted(available)),
                        execution_time=time.time() - start_time,
                        error="FlowNotFound"
                    )

                content = f"# 🔄 Flow: `{target}`\n\n"

                # Docstring
                if hasattr(flow_func, '__doc__') and flow_func.__doc__:
                    content += f"## Description\n\n{flow_func.__doc__}\n\n"

                # Signature
                import inspect
                try:
                    sig = inspect.signature(flow_func)
                    content += f"## Signature\n\n"
                    content += f"```python\n{target}{sig}\n```\n\n"

                    # Parameters
                    params = list(sig.parameters.items())
                    if len(params) > 2:  # Skip app, args_sto
                        content += "## Parameters\n\n"
                        content += "| Parameter | Default | Description |\n"
                        content += "|-----------|---------|-------------|\n"
                        for name, param in params[2:]:  # Skip app, args_sto
                            default = param.default if param.default != inspect.Parameter.empty else "Required"
                            content += f"| `{name}` | `{default}` | - |\n"
                except Exception:
                    pass

                # Async?
                import asyncio
                is_async = asyncio.iscoroutinefunction(flow_func)
                content += f"\n## Properties\n\n"
                content += f"- **Type**: {'Async' if is_async else 'Sync'}\n"

                content += f"\n## Example\n\n"
                content += "```python\n"
                content += f"# Start this flow\n"
                content += f"flow_start(flow_name='{target}')\n"
                content += "```"

            # =================================================================
            # PYTHON GUIDE
            # =================================================================
            elif info_type == "python_guide":
                from toolboxv2.mcp_server.models import PYTHON_EXECUTION_TEMPLATE
                content = PYTHON_EXECUTION_TEMPLATE

            # =================================================================
            # PERFORMANCE GUIDE
            # =================================================================
            elif info_type == "performance_guide":
                from toolboxv2.mcp_server.models import PERFORMANCE_GUIDE_TEMPLATE
                content = PERFORMANCE_GUIDE_TEMPLATE.format(
                    cache_ttl=300,
                    max_cache_size=100,
                    requests=0,
                    avg_time=0.0,
                    hit_rate=0.0
                )

            # =================================================================
            # DISCOVERY (Full system overview)
            # =================================================================
            elif info_type == "discovery":
                functions_dict = getattr(app, 'functions', {})
                flows = getattr(app, 'flows', {})

                content = "# 🔍 ToolBoxV2 System Discovery\n\n"

                # Summary
                total_funcs = sum(
                    len(f) if isinstance(f, dict) else 0
                    for f in functions_dict.values()
                )
                content += "## 📊 Summary\n\n"
                content += f"- **Modules**: {len(functions_dict)}\n"
                content += f"- **Functions**: {total_funcs}\n"
                content += f"- **Flows**: {len(flows)}\n\n"

                # Top modules
                content += "## 📦 Top Modules (by function count)\n\n"
                module_counts = [
                    (name, len(funcs) if isinstance(funcs, dict) else 0)
                    for name, funcs in functions_dict.items()
                ]
                module_counts.sort(key=lambda x: -x[1])

                for name, count in module_counts[:10]:
                    content += f"- `{name}` ({count} functions)\n"

                # Flows
                if flows:
                    content += f"\n## 🔄 Available Flows\n\n"
                    for flow_name in sorted(list(flows.keys())[:15]):
                        content += f"- `{flow_name}`\n"

                content += "\n## 🛠 Available Tools\n\n"
                content += "- `toolbox_execute` - Execute module functions\n"
                content += "- `python_execute` - Run Python code\n"
                content += "- `docs_reader` - Search documentation\n"
                content += "- `flow_start` / `flow_input` - Run interactive flows\n"
                content += "- `toolbox_info` - Get detailed info (this tool)\n"

                content += "\n## 💡 Quick Start\n\n"
                content += "```python\n"
                content += "# 1. Explore a module\n"
                content += "toolbox_info(info_type='functions', target='isaa')\n\n"
                content += "# 2. Get function details\n"
                content += "toolbox_info(info_type='function_detail', target='isaa.run_agent')\n\n"
                content += "# 3. Execute a function\n"
                content += "toolbox_execute(module='isaa', function='run_agent', kwargs={...})\n"
                content += "```"

            # =================================================================
            # UNKNOWN
            # =================================================================
            else:
                valid_types = [
                    "modules", "functions", "function_detail",
                    "flows", "flow_detail",
                    "python_guide", "performance_guide",
                    "discovery"
                ]
                content = f"❌ Unknown info type: `{info_type}`\n\n"
                content += "**Valid types**:\n"
                for t in valid_types:
                    content += f"- `{t}`\n"

            return ToolResult(
                success=True,
                content=content,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            import traceback
            return ToolResult(
                success=False,
                content=f"❌ Info error: {e}\n\n```\n{traceback.format_exc()}\n```",
                execution_time=time.time() - start_time,
                error=str(e)
            )

    @staticmethod
    async def search_functions(
        app: Any,
        query: str,
        max_results: int = 20
    ) -> ToolResult:
        """
        Search for functions across all modules.

        Args:
            app: ToolBoxV2 App instance
            query: Search query (matches function names, module names)
            max_results: Maximum results to return

        Returns:
            ToolResult with matching functions
        """
        start_time = time.time()

        try:
            functions_dict = getattr(app, 'functions', {})
            query_lower = query.lower()

            matches = []

            for mod_name, mod_funcs in functions_dict.items():
                if not isinstance(mod_funcs, dict):
                    continue

                # Check if module name matches
                mod_match = query_lower in mod_name.lower()

                for func_name, func_data in mod_funcs.items():
                    # Check if function name matches
                    func_match = query_lower in func_name.lower()

                    # Check docstring if available
                    doc_match = False
                    doc = ""
                    if isinstance(func_data, dict):
                        func_obj = func_data.get('func')
                        if func_obj and hasattr(func_obj, '__doc__') and func_obj.__doc__:
                            doc = func_obj.__doc__[:200]
                            doc_match = query_lower in doc.lower()

                    if mod_match or func_match or doc_match:
                        params = []
                        is_api = False
                        if isinstance(func_data, dict):
                            params = func_data.get('params', [])
                            is_api = func_data.get('api', False)

                        matches.append({
                            'module': mod_name,
                            'function': func_name,
                            'params': params[:5],
                            'api': is_api,
                            'doc': doc[:100] if doc else None,
                            'relevance': (
                                3 if func_match else 0 +
                                2 if mod_match else 0 +
                                1 if doc_match else 0
                            )
                        })

            # Sort by relevance
            matches.sort(key=lambda x: -x['relevance'])
            matches = matches[:max_results]

            if not matches:
                content = f"🔍 No functions found matching `{query}`\n\n"
                content += "**Tips:**\n"
                content += "- Try a shorter search term\n"
                content += "- Use `toolbox_info(info_type='modules')` to browse all modules\n"
                content += "- Use `toolbox_info(info_type='discovery')` for an overview"
            else:
                content = f"# 🔍 Search Results for `{query}`\n\n"
                content += f"Found **{len(matches)}** matches:\n\n"

                for m in matches:
                    params_str = ", ".join(m['params'][:3])
                    if len(m['params']) > 3:
                        params_str += ", ..."

                    api_badge = " 🌐" if m['api'] else ""
                    content += f"### `{m['module']}.{m['function']}`{api_badge}\n"
                    content += f"```python\n{m['function']}({params_str})\n```\n"
                    if m['doc']:
                        content += f"_{m['doc']}..._\n"
                    content += "\n"

                content += "---\n"
                content += f"*Use `toolbox_info(info_type='function_detail', target='Module.function')` for more details*"

            return ToolResult(
                success=True,
                content=content,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Search error: {e}",
                execution_time=time.time() - start_time,
                error=str(e)
            )

    @staticmethod
    async def get_module_tree(
        app: Any,
        module: Optional[str] = None,
        depth: int = 2
    ) -> ToolResult:
        """
        Get a tree view of modules and their functions.

        Args:
            app: ToolBoxV2 App instance
            module: Specific module to show (None = all)
            depth: Tree depth (1 = modules only, 2 = with functions)

        Returns:
            ToolResult with tree structure
        """
        start_time = time.time()

        try:
            functions_dict = getattr(app, 'functions', {})

            content = "# 📂 Module Tree\n\n"
            content += "```\n"
            content += "ToolBoxV2/\n"

            modules_to_show = functions_dict.items()
            if module:
                if module in functions_dict:
                    modules_to_show = [(module, functions_dict[module])]
                else:
                    return ToolResult(
                        success=False,
                        content=f"❌ Module `{module}` not found",
                        execution_time=time.time() - start_time,
                        error="ModuleNotFound"
                    )

            for mod_name, mod_funcs in sorted(modules_to_show, key=lambda x: x[0]):
                func_count = len(mod_funcs) if isinstance(mod_funcs, dict) else 0
                content += f"├── 📦 {mod_name}/ ({func_count} functions)\n"

                if depth >= 2 and isinstance(mod_funcs, dict):
                    func_items = sorted(mod_funcs.items(), key=lambda x: x[0])
                    for i, (func_name, func_data) in enumerate(func_items[:15]):
                        is_last = i == len(func_items[:15]) - 1
                        prefix = "│   └──" if is_last else "│   ├──"

                        api_marker = "🌐" if (isinstance(func_data, dict) and func_data.get('api')) else "🔧"
                        content += f"{prefix} {api_marker} {func_name}()\n"

                    if len(func_items) > 15:
                        content += f"│   └── ... ({len(func_items) - 15} more)\n"

            content += "```\n\n"
            content += "**Legend**: 🌐 = API exposed, 🔧 = Internal"

            return ToolResult(
                success=True,
                content=content,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Tree error: {e}",
                execution_time=time.time() - start_time,
                error=str(e)
            )

    @staticmethod
    async def get_callable_summary(
        app: Any,
        module: str
    ) -> ToolResult:
        """
        Get a concise summary of callable functions in a module,
        optimized for LLM tool use.

        Args:
            app: ToolBoxV2 App instance
            module: Module name

        Returns:
            ToolResult with callable signatures
        """
        start_time = time.time()

        try:
            functions_dict = getattr(app, 'functions', {})
            mod_funcs = functions_dict.get(module, {})

            if not mod_funcs:
                return ToolResult(
                    success=False,
                    content=f"❌ Module `{module}` not found",
                    execution_time=time.time() - start_time,
                    error="ModuleNotFound"
                )

            content = f"# {module} - Callable Summary\n\n"
            content += "```python\n"
            content += f"# Module: {module}\n"
            content += f"# Execute via: toolbox_execute(module='{module}', function='...', kwargs={{...}})\n\n"

            for func_name, func_data in sorted(mod_funcs.items()):
                if not isinstance(func_data, dict):
                    continue

                params = func_data.get('params', [])
                is_api = func_data.get('api', False)

                if not is_api:
                    continue  # Only show API functions

                # Build signature
                param_strs = []
                for p in params[:6]:
                    param_strs.append(p)
                if len(params) > 6:
                    param_strs.append("...")

                sig = f"def {func_name}({', '.join(param_strs)})"
                content += f"{sig}\n"

                # Add docstring if available
                func_obj = func_data.get('func')
                if func_obj and hasattr(func_obj, '__doc__') and func_obj.__doc__:
                    doc_line = func_obj.__doc__.split('\n')[0].strip()[:60]
                    content += f'    """{doc_line}"""\n'

                content += "\n"

            content += "```"

            return ToolResult(
                success=True,
                content=content,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Summary error: {e}",
                execution_time=time.time() - start_time,
                error=str(e)
            )
