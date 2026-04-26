import asyncio
import contextlib
import os
from typing import Any

try:
    from mcp import ClientSession
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = object
    print("MCP not available, skipping MCP session manager")

AGENT_VERBOSE = os.environ.get("AGENT_VERBOSE", "false").lower() == "true"
eprint = print if AGENT_VERBOSE else lambda *a, **k: None
wprint = print if AGENT_VERBOSE else lambda *a, **k: None
iprint = print if AGENT_VERBOSE else lambda *a, **k: None

import json as _json
from pathlib import Path as _Path


def _make_mcp_tool_func(session, server_name: str, tool_name: str, tool_info: dict):
    """
    Returns an async callable that calls session.call_tool().
    Works without FlowAgentBuilder.
    """
    input_schema = tool_info.get("input_schema", {}) or {}
    props = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    async def mcp_tool_func(**kwargs) -> str:
        try:
            # MCPSessionManager stores live sessions — use it directly
            result = await session.call_tool(tool_name, arguments=kwargs)
            # result.content is a list of ContentBlock objects
            if hasattr(result, "content"):
                parts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                    elif hasattr(block, "data"):
                        parts.append(str(block.data))
                    else:
                        parts.append(str(block))
                return "\n".join(parts)
            return str(result)
        except Exception as e:
            return f"MCP tool error ({server_name}.{tool_name}): {e}"

    # Docstring for tool_manager description lookup
    mcp_tool_func.__name__ = f"{server_name}_{tool_name}"
    mcp_tool_func.__doc__ = tool_info.get("description", f"MCP tool: {tool_name}")

    return mcp_tool_func


# ─────────────────────────────────────────────────────────────────────────────
# Register one MCP server config on an already-obtained session
# Returns count of registered tools
# ─────────────────────────────────────────────────────────────────────────────

async def _register_mcp_session(
    agent,
    session,  # live ClientSession
    server_name: str,
    mcp_session_manager,
) -> int:
    """
    Extract capabilities from session and register tools on agent.tool_manager.
    Returns number of tools registered.
    """
    if session is None:
        return 0

    caps = await mcp_session_manager.extract_capabilities_with_timeout(session, server_name)
    tools = caps.get("tools", {})

    count = 0
    for t_name, t_info in tools.items():
        wrapper_name = f"{server_name}_{t_name}"
        func = _make_mcp_tool_func(session, server_name, t_name, t_info)

        # agent.add_tools() accepts a list of dicts OR callables — use dict form
        # which matches how FlowAgentBuilder registers MCP tools:
        agent.add_tools([{
            "tool_func": func,
            "name": wrapper_name,
            "description": t_info.get("description", f"{server_name}: {t_name}"),
            "category": [f"mcp_{server_name}", "mcp"],
            "flags": {"mcp": True, "server": server_name},
        }])
        count += 1

    return count


# ─────────────────────────────────────────────────────────────────────────────
# Claude-Code config loader
# Supports: .mcp.json  /  claude_desktop_config.json  /  ~/.claude.json
# ─────────────────────────────────────────────────────────────────────────────

def _load_claude_mcp_configs(path: str) -> dict[str, dict]:
    """
    Parse a Claude-Code-compatible MCP config file.

    Supported formats:
      { "mcpServers": { "name": { "command": "...", "args": [...], "env": {} } } }
      { "mcp": { "servers": [ { "name": "...", "command": "...", ... } ] } }

    Returns dict: server_name → server_config
    """
    p = _Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    with open(p, encoding="utf-8") as f:
        raw = _json.load(f)

    servers: dict[str, dict] = {}

    # Format 1: Claude Desktop / Claude Code  {"mcpServers": {...}}
    if "mcpServers" in raw:
        for name, cfg in raw["mcpServers"].items():
            servers[name] = {
                "command": cfg.get("command", ""),
                "args": cfg.get("args", []),
                "env": cfg.get("env", {}),
                "transport": cfg.get("transport", "stdio"),
            }

    # Format 2: toolboxv2 legacy  {"mcp": {"servers": [...]}}
    elif "mcp" in raw and "servers" in raw.get("mcp", {}):
        for entry in raw["mcp"]["servers"]:
            name = entry.get("name", "unknown")
            servers[name] = {
                "command": entry.get("command", ""),
                "args": entry.get("args", []),
                "env": entry.get("env", {}),
                "transport": entry.get("transport", "stdio"),
            }

    # Format 3: flat list  [{"name":..., "command":...}]
    elif isinstance(raw, list):
        for entry in raw:
            name = entry.get("name", "unknown")
            servers[name] = {
                "command": entry.get("command", ""),
                "args": entry.get("args", []),
                "env": entry.get("env", {}),
                "transport": entry.get("transport", "stdio"),
            }

    return servers


# ─────────────────────────────────────────────────────────────────────────────
# Default config file search order (Claude Code compatible)
# ─────────────────────────────────────────────────────────────────────────────

def _find_default_mcp_configs() -> list[str]:
    candidates = [
        ".mcp.json",
        "mcp.json",
        str(_Path.home() / ".claude.json"),
        str(_Path.home() / ".config" / "claude" / "mcp.json"),
        str(_Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"),
        str(_Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"),
    ]
    return [c for c in candidates if _Path(c).expanduser().exists()]


class MCPSessionManager:
    """Manages persistent MCP sessions with automatic reconnection and parallel processing"""

    def __init__(self):
        self.sessions: dict[str, ClientSession] = {}
        self.connections: dict[str, Any] = {}
        self.capabilities_cache: dict[str, dict] = {}
        self.retry_count: dict[str, int] = {}
        self.max_retries = 3
        self.connection_timeout = 15.0  # 10 seconds timeout
        self.operation_timeout = 10.0  # 5 seconds for operations

    async def get_session_with_timeout(self, server_name: str, server_config: dict[str, Any]) -> ClientSession | None:
        """Get session with timeout protection"""
        try:
            return await asyncio.wait_for(
                self.get_session(server_name, server_config),
                timeout=self.connection_timeout
            )
        except TimeoutError:
            eprint(f"MCP session creation timeout for {server_name}")
            return None

    async def get_session(self, server_name: str, server_config: dict[str, Any]) -> ClientSession | None:
        """Get or create persistent MCP session with proper context management"""
        if server_name in self.sessions:
            try:
                # Test if session is still alive with timeout
                session = self.sessions[server_name]
                # Quick connectivity test
                await asyncio.wait_for(session.list_tools(), timeout=2.0)
                return session
            except Exception as e:
                wprint(f"MCP session {server_name} failed, recreating: {e}")
                # Clean up the old session
                if server_name in self.sessions:
                    del self.sessions[server_name]
                if server_name in self.connections:
                    del self.connections[server_name]

        return await self._create_session(server_name, server_config)

    async def _create_session(self, server_name: str, server_config: dict[str, Any]) -> ClientSession | None:
        """Create new MCP session with improved error handling"""
        try:
            command = server_config.get('command')
            args = server_config.get('args', [])
            env = server_config.get('env', {})
            transport_type = server_config.get('transport', 'stdio')

            if not command:
                eprint(f"No command specified for MCP server {server_name}")
                return None

            iprint(f"Creating MCP session for {server_name} (transport: {transport_type})")

            session = None

            # Create connection based on transport type
            if transport_type == 'stdio':
                session = await self._create_stdio_session(server_name, command, args, env)
            elif transport_type in ['http', 'streamable-http']:
                session = await self._create_http_session(server_name, server_config)
            else:
                eprint(f"Unsupported transport type: {transport_type}")
                return None

            if session:
                self.sessions[server_name] = session
                self.retry_count[server_name] = 0
                iprint(f"✓ MCP session created successfully: {server_name}")
                return session

            return None

        except Exception as e:
            self.retry_count[server_name] = self.retry_count.get(server_name, 0) + 1
            if self.retry_count[server_name] <= self.max_retries:
                wprint(f"MCP session creation failed (attempt {self.retry_count[server_name]}/{self.max_retries}): {e}")
                await asyncio.sleep(1.0)  # Longer delay before retry
                return await self._create_session(server_name, server_config)
            else:
                eprint(f"✗ MCP session creation failed after {self.max_retries} attempts: {e}")
                return None

    async def _create_stdio_session(self, server_name: str, command: str, args: list[str], env: dict[str, str]) -> \
    ClientSession | None:
        """Create stdio MCP session with fixed async context handling"""
        try:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            # Prepare environment
            process_env = os.environ.copy()
            process_env.update(env)

            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=process_env
            )

            # Create the stdio client and session in a single task context
            stdio_connection = stdio_client(server_params)

            # Enter the context manager
            read_stream, write_stream = await stdio_connection.__aenter__()

            # Store the connection for cleanup later
            self.connections[server_name] = stdio_connection

            # Create session
            session = ClientSession(read_stream, write_stream)

            # Initialize session in the same context
            await session.__aenter__()
            await asyncio.wait_for(session.initialize(), timeout=self.connection_timeout)

            return session

        except Exception as e:
            eprint(f"Failed to create stdio session for {server_name}: {e}")
            # Cleanup on failure
            if server_name in self.connections:
                with contextlib.suppress(Exception):
                    await self.connections[server_name].__aexit__(None, None, None)
                del self.connections[server_name]
            return None

    async def _create_http_session(self, server_name: str, server_config: dict[str, Any]) -> ClientSession | None:
        """Create HTTP MCP session with timeout"""
        try:
            import httpx
            from mcp.client.streamable_http import streamable_http_client

            url = server_config.get("url", f"http://localhost:{server_config.get('port', 8000)}/mcp")
            headers = server_config.get("headers", {})
            timeout = server_config.get("timeout", self.connection_timeout)

            # Build httpx client with configurable headers/auth/timeout
            http_client = httpx.AsyncClient(
                headers=headers,
                timeout=httpx.Timeout(float(timeout)),
            )
            # streamable_http_client() yields (read_stream, write_stream)
            streams = streamable_http_client(url, http_client=http_client)
            read_stream, write_stream = await asyncio.wait_for(
                streams.__aenter__(), timeout=self.connection_timeout
            )

            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            await asyncio.wait_for(session.initialize(), timeout=self.connection_timeout)

            # Store http_client for cleanup
            self.connections[server_name] = (streams, http_client)
            return session

        except Exception as e:
            eprint(f"Failed to create HTTP session for {server_name}: {e}")
            return None

    async def extract_capabilities_with_timeout(self, session: ClientSession, server_name: str) -> dict[str, dict]:
        """Extract capabilities with timeout protection"""
        try:
            return await asyncio.wait_for(
                self.extract_capabilities(session, server_name),
                timeout=self.operation_timeout
            )
        except TimeoutError:
            eprint(f"Capability extraction timeout for {server_name}")
            return {'tools': {}, 'resources': {}, 'resource_templates': {}, 'prompts': {}, 'images': {}}

    async def extract_capabilities(self, session: ClientSession, server_name: str) -> dict[str, dict]:
        """Extract all capabilities from MCP session"""
        if server_name in self.capabilities_cache:
            return self.capabilities_cache[server_name]

        capabilities = {
            'tools': {},
            'resources': {},
            'resource_templates': {},
            'prompts': {},
            'images': {}
        }

        try:
            # Extract tools with individual timeouts
            try:
                tools_response = await asyncio.wait_for(session.list_tools(), timeout=3.0)
                for tool in tools_response.tools:
                    capabilities['tools'][tool.name] = {
                        'name': tool.name,
                        'description': tool.description or '',
                        'input_schema': tool.inputSchema,
                        'output_schema': getattr(tool, 'outputSchema', None),
                        'display_name': getattr(tool, 'title', tool.name)
                    }
            except TimeoutError:
                wprint(f"Tools extraction timeout for {server_name}")
            except Exception as e:
                wprint(f"Failed to extract tools from {server_name}: {e}")

            # Extract resources with timeout
            try:
                resources_response = await asyncio.wait_for(session.list_resources(), timeout=3.0)
                for resource in resources_response.resources:
                    capabilities['resources'][str(resource.uri)] = {
                        'uri': str(resource.uri),
                        'name': resource.name or str(resource.uri),
                        'description': resource.description or '',
                        'mime_type': getattr(resource, 'mimeType', None)
                    }
            except TimeoutError:
                wprint(f"Resources extraction timeout for {server_name}")
            except Exception as e:
                wprint(f"Failed to extract resources from {server_name}: {e}")

            # Extract resource templates with timeout
            try:
                templates_response = await asyncio.wait_for(session.list_resource_templates(), timeout=3.0)
                for template in templates_response.resourceTemplates:
                    capabilities['resource_templates'][template.uriTemplate] = {
                        'uri_template': template.uriTemplate,
                        'name': template.name or template.uriTemplate,
                        'description': template.description or ''
                    }
            except TimeoutError:
                wprint(f"Resource templates extraction timeout for {server_name}")
            except Exception as e:
                wprint(f"Failed to extract resource templates from {server_name}: {e}")

            # Extract prompts with timeout
            try:
                prompts_response = await asyncio.wait_for(session.list_prompts(), timeout=3.0)
                for prompt in prompts_response.prompts:
                    capabilities['prompts'][prompt.name] = {
                        'name': prompt.name,
                        'description': prompt.description or '',
                        'arguments': [
                            {
                                'name': arg.name,
                                'description': arg.description or '',
                                'required': arg.required
                            } for arg in (prompt.arguments or [])
                        ]
                    }
            except TimeoutError:
                wprint(f"Prompts extraction timeout for {server_name}")
            except Exception as e:
                wprint(f"Failed to extract prompts from {server_name}: {e}")

            self.capabilities_cache[server_name] = capabilities

            total_caps = (len(capabilities['tools']) + len(capabilities['resources']) +
                          len(capabilities['resource_templates']) + len(capabilities['prompts']))
            iprint(f"✓ Extracted {total_caps} capabilities from {server_name}")

        except Exception as e:
            eprint(f"Failed to extract capabilities from {server_name}: {e}")

        return capabilities

    async def _cleanup_session(self, server_name: str):
        """Clean up a specific session with proper context management"""
        try:
            # Clean up session first
            if server_name in self.sessions:
                try:
                    session = self.sessions[server_name]
                    await asyncio.wait_for(session.__aexit__(None, None, None), timeout=2.0)
                except (TimeoutError, Exception) as e:
                    wprint(f"Session cleanup warning for {server_name}: {e}")
                finally:
                    del self.sessions[server_name]

            # Clean up connection
            if server_name in self.connections:
                try:
                    streams, http_client = self.connections[server_name]
                    await asyncio.wait_for(streams.__aexit__(None, None, None), timeout=2.0)
                    await http_client.aclose()
                except (TimeoutError, Exception) as e:
                    wprint(f"Connection cleanup warning for {server_name}: {e}")
                finally:
                    del self.connections[server_name]

            # Clear cache
            if server_name in self.capabilities_cache:
                del self.capabilities_cache[server_name]

            # Reset retry count
            if server_name in self.retry_count:
                del self.retry_count[server_name]

        except Exception as e:
            wprint(f"Cleanup error for {server_name}: {e}")

    async def cleanup_all(self):
        """Clean up all sessions with timeout and proper error handling"""
        cleanup_tasks = []
        for server_name in list(self.sessions.keys()):
            task = asyncio.create_task(self._cleanup_session(server_name))
            cleanup_tasks.append(task)

        if cleanup_tasks:
            try:
                # Use gather with return_exceptions=True to collect all results
                results = await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=5.0
                )

                # Log any non-cancellation exceptions
                for i, result in enumerate(results):
                    if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                        wprint(f"Error cleaning up MCP session: {result}")

            except (asyncio.TimeoutError, asyncio.CancelledError):
                # Handle timeout and cancellation
                wprint("MCP session cleanup timeout or cancelled")
                # Cancel all tasks
                for task in cleanup_tasks:
                    if not task.done():
                        task.cancel()

                # Give tasks a moment to cancel cleanly
                try:
                    await asyncio.wait(cleanup_tasks, timeout=1.0)
                except asyncio.CancelledError:
                    pass

            except Exception as e:
                wprint(f"Unexpected error during MCP session cleanup: {e}")
                # Cancel remaining tasks
                for task in cleanup_tasks:
                    if not task.done():
                        task.cancel()


if not MCP_AVAILABLE:
    class MCPSessionManager:
        def __getitem__(self, key):
            raise ImportError("MCP is not available")

    MCPSessionManager = MCPSessionManager
