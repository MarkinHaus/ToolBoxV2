"""
Flow Handlers for MCP Server
============================

Integrates FlowSessionManager with MCP protocol.
Provides tools for AI agents to interact with ToolBoxV2 flows.

Tools:
- flow_list_available: List all registered flows
- flow_start: Start a flow session
- flow_input: Provide input to waiting flow
- flow_status: Get session status
- flow_cancel: Cancel running flow
- flow_list_sessions: List active sessions
"""

from __future__ import annotations

import time
import json
import logging
from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from toolboxv2.utils.toolbox import App

from toolboxv2.mcp_server.flow_manager import FlowSessionManager, FlowState

logger = logging.getLogger("mcp.flow_handlers")


class FlowToolResult:
    """Result container for flow tools."""

    __slots__ = ("success", "content", "execution_time", "error", "data")

    def __init__(
        self,
        success: bool,
        content: str,
        execution_time: float,
        error: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.content = content
        self.execution_time = execution_time
        self.error = error
        self.data = data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "content": self.content,
            "execution_time": self.execution_time,
            "error": self.error,
            "data": self.data
        }


class FlowHandlers:
    """
    MCP tool handlers for flow operations.

    Bridges MCP protocol with FlowSessionManager.
    All methods return FlowToolResult.
    """

    def __init__(self, session_manager: FlowSessionManager):
        self._manager = session_manager

    # -------------------------------------------------------------------------
    # TOOL DEFINITIONS
    # -------------------------------------------------------------------------

    @staticmethod
    def get_tool_definitions() -> List[Dict[str, Any]]:
        """Get MCP tool definitions for flow operations."""
        return [
            {
                "name": "flow_list_available",
                "description": (
                    "List all available flows registered in ToolBoxV2. "
                    "Returns flow names that can be started with flow_start."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "filter": {
                            "type": "string",
                            "description": "Optional filter string to match flow names"
                        }
                    }
                }
            },
            {
                "name": "flow_start",
                "description": (
                    "Start a new flow session. Flows are interactive scripts that may "
                    "request input via prompts. After starting, use flow_input to provide "
                    "responses when the flow is waiting for input. "
                    "Returns session_id for subsequent operations."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "flow_name": {
                            "type": "string",
                            "description": "Name of the flow to start"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional custom session ID"
                        },
                        "auto_start": {
                            "type": "boolean",
                            "description": "Start execution immediately (default: true)",
                            "default": True
                        },
                        "kwargs": {
                            "type": "object",
                            "description": "Additional arguments to pass to the flow"
                        }
                    },
                    "required": ["flow_name"]
                }
            },
            {
                "name": "flow_input",
                "description": (
                    "Provide input to a flow that is waiting for user input. "
                    "Check flow_status or the waiting_input field to know when input is needed. "
                    "The prompt field shows what the flow is asking for."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "The session ID returned by flow_start"
                        },
                        "input": {
                            "type": "string",
                            "description": "The input to provide (simulates user typing)"
                        }
                    },
                    "required": ["session_id", "input"]
                }
            },
            {
                "name": "flow_status",
                "description": (
                    "Get detailed status of a flow session including output buffer, "
                    "current prompt (if waiting for input), history, and result."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "The session ID"
                        }
                    },
                    "required": ["session_id"]
                }
            },
            {
                "name": "flow_cancel",
                "description": "Cancel a running flow session.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "The session ID to cancel"
                        }
                    },
                    "required": ["session_id"]
                }
            },
            {
                "name": "flow_list_sessions",
                "description": "List all active flow sessions with their current states.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]

    # -------------------------------------------------------------------------
    # TOOL HANDLERS
    # -------------------------------------------------------------------------

    async def handle_list_available(
        self,
        app: "App",
        filter_str: Optional[str] = None
    ) -> FlowToolResult:
        """List available flows."""
        start_time = time.time()

        try:
            if not app or not hasattr(app, "flows"):
                return FlowToolResult(
                    success=False,
                    content="❌ Flow system not available",
                    execution_time=time.time() - start_time,
                    error="FlowsNotAvailable"
                )

            flows = list(app.flows.keys())

            if filter_str:
                filter_lower = filter_str.lower()
                flows = [f for f in flows if filter_lower in f.lower()]

            if not flows:
                content = "📋 No flows available"
                if filter_str:
                    content += f" matching '{filter_str}'"
            else:
                content = f"📋 **Available Flows** ({len(flows)})\n\n"
                for flow in sorted(flows):
                    content += f"- `{flow}`\n"
                content += "\nUse `flow_start(flow_name='...')` to start a flow."

            return FlowToolResult(
                success=True,
                content=content,
                execution_time=time.time() - start_time,
                data={"flows": flows}
            )

        except Exception as e:
            logger.exception("Error listing flows")
            return FlowToolResult(
                success=False,
                content=f"❌ Error: {e}",
                execution_time=time.time() - start_time,
                error=str(e)
            )

    async def handle_start(
        self,
        app: "App",
        flow_name: str,
        session_id: Optional[str] = None,
        auto_start: bool = True,
        kwargs: Optional[Dict[str, Any]] = None
    ) -> FlowToolResult:
        """Start a flow session."""
        start_time = time.time()

        try:
            # Ensure manager has app reference
            if not self._manager._app:
                self._manager.set_app(app)

            # Create session
            session = await self._manager.create(
                flow_name=flow_name,
                session_id=session_id,
                auto_start=auto_start,
                **(kwargs or {})
            )

            # Get initial status
            if auto_start:
                status = await self._manager.get_status(session.session_id)
            else:
                status = session.to_dict()
                status["output"] = []

            # Format response
            content = f"🚀 **Flow Started**\n\n"
            content += f"- **Flow**: `{flow_name}`\n"
            content += f"- **Session ID**: `{session.session_id}`\n"
            content += f"- **State**: {status.get('state', 'created')}\n"

            # Include output if any
            output = status.get("output", [])
            if output:
                content += f"\n**Output:**\n"
                for entry in output:
                    content += f"```\n{entry.get('content', '')}\n```\n"

            # Check if waiting for input
            if status.get("waiting_input"):
                prompt = status.get("prompt", "")
                content += f"\n⏳ **Waiting for input**"
                if prompt:
                    content += f": {prompt}"
                content += f"\n\nUse `flow_input(session_id='{session.session_id}', input='...')` to respond."

            return FlowToolResult(
                success=True,
                content=content,
                execution_time=time.time() - start_time,
                data=status
            )

        except ValueError as e:
            return FlowToolResult(
                success=False,
                content=f"❌ {e}",
                execution_time=time.time() - start_time,
                error=str(e)
            )
        except Exception as e:
            logger.exception("Error starting flow")
            return FlowToolResult(
                success=False,
                content=f"❌ Error starting flow: {e}",
                execution_time=time.time() - start_time,
                error=str(e)
            )

    async def handle_input(
        self,
        session_id: str,
        input_value: str
    ) -> FlowToolResult:
        """Provide input to a waiting flow."""
        start_time = time.time()

        try:
            result = await self._manager.provide_input(session_id, input_value)

            # Format response
            state = result.get("state", "unknown")
            content = f"📝 **Input Provided**\n\n"
            content += f"- **Session**: `{session_id}`\n"
            content += f"- **Input**: `{input_value}`\n"
            content += f"- **State**: {state}\n"

            # Include output
            output = result.get("output", [])
            if output:
                content += f"\n**Output:**\n"
                for entry in output:
                    entry_type = entry.get("type", "stdout")
                    entry_content = entry.get("content", "")
                    if entry_type == "prompt":
                        content += f"❓ {entry_content}\n"
                    elif entry_type == "stderr":
                        content += f"⚠️ {entry_content}\n"
                    elif entry_type == "result":
                        content += f"✅ {entry_content}\n"
                    else:
                        content += f"{entry_content}\n"

            # Check state
            if result.get("waiting_input"):
                prompt = result.get("prompt", "")
                content += f"\n⏳ **Waiting for input**"
                if prompt:
                    content += f": {prompt}"
            elif state == "completed":
                content += f"\n✅ **Flow completed**"
                if result.get("result"):
                    content += f"\n\n**Result:**\n```json\n{json.dumps(result['result'], indent=2)}\n```"
            elif state == "error":
                content += f"\n❌ **Error**: {result.get('error', 'Unknown error')}"

            return FlowToolResult(
                success=state not in ("error",),
                content=content,
                execution_time=time.time() - start_time,
                error=result.get("error"),
                data=result
            )

        except ValueError as e:
            return FlowToolResult(
                success=False,
                content=f"❌ {e}",
                execution_time=time.time() - start_time,
                error=str(e)
            )
        except Exception as e:
            logger.exception("Error providing input")
            return FlowToolResult(
                success=False,
                content=f"❌ Error: {e}",
                execution_time=time.time() - start_time,
                error=str(e)
            )

    async def handle_status(self, session_id: str) -> FlowToolResult:
        """Get flow session status."""
        start_time = time.time()

        try:
            status = await self._manager.get_status(session_id)

            # Format response
            state = status.get("state", "unknown")
            content = f"📊 **Flow Status**\n\n"
            content += f"- **Session**: `{session_id}`\n"
            content += f"- **Flow**: `{status.get('flow_name', 'unknown')}`\n"
            content += f"- **State**: {state}\n"
            content += f"- **History**: {len(status.get('history', []))} entries\n"

            # Current state details
            if status.get("waiting_input"):
                prompt = status.get("prompt", "")
                content += f"\n⏳ **Waiting for input**"
                if prompt:
                    content += f": {prompt}"
            elif state == "completed":
                content += f"\n✅ **Completed**"
            elif state == "error":
                content += f"\n❌ **Error**: {status.get('error', 'Unknown')}"
            elif state == "running":
                content += f"\n🔄 **Running**"

            # Pending output
            output = status.get("output", [])
            if output:
                content += f"\n\n**Pending Output:**\n"
                for entry in output[-10:]:  # Last 10 entries
                    content += f"- [{entry.get('type')}] {entry.get('content', '')[:100]}\n"

            # Recent history
            history = status.get("history", [])[-5:]
            if history:
                content += f"\n**Recent History:**\n"
                for h in history:
                    content += f"- {h[:80]}...\n" if len(h) > 80 else f"- {h}\n"

            return FlowToolResult(
                success=True,
                content=content,
                execution_time=time.time() - start_time,
                data=status
            )

        except ValueError as e:
            return FlowToolResult(
                success=False,
                content=f"❌ Session not found: {session_id}",
                execution_time=time.time() - start_time,
                error=str(e)
            )
        except Exception as e:
            logger.exception("Error getting status")
            return FlowToolResult(
                success=False,
                content=f"❌ Error: {e}",
                execution_time=time.time() - start_time,
                error=str(e)
            )

    async def handle_cancel(self, session_id: str) -> FlowToolResult:
        """Cancel a flow session."""
        start_time = time.time()

        try:
            success = await self._manager.cancel(session_id)

            if success:
                return FlowToolResult(
                    success=True,
                    content=f"✅ Flow session `{session_id}` cancelled",
                    execution_time=time.time() - start_time
                )
            else:
                return FlowToolResult(
                    success=False,
                    content=f"❌ Could not cancel session `{session_id}` (not running or not found)",
                    execution_time=time.time() - start_time,
                    error="CancelFailed"
                )

        except Exception as e:
            logger.exception("Error cancelling flow")
            return FlowToolResult(
                success=False,
                content=f"❌ Error: {e}",
                execution_time=time.time() - start_time,
                error=str(e)
            )

    async def handle_list_sessions(self) -> FlowToolResult:
        """List all active sessions."""
        start_time = time.time()

        try:
            sessions = await self._manager.list_sessions()

            if not sessions:
                return FlowToolResult(
                    success=True,
                    content="📋 No active flow sessions",
                    execution_time=time.time() - start_time,
                    data={"sessions": []}
                )

            content = f"📋 **Active Flow Sessions** ({len(sessions)})\n\n"

            for s in sessions:
                state = s.get("state", "unknown")
                state_emoji = {
                    "created": "🆕",
                    "running": "🔄",
                    "waiting_input": "⏳",
                    "completed": "✅",
                    "error": "❌",
                    "cancelled": "🚫"
                }.get(state, "❓")

                content += f"- {state_emoji} `{s['session_id']}`: {s['flow_name']} ({state})\n"

            return FlowToolResult(
                success=True,
                content=content,
                execution_time=time.time() - start_time,
                data={"sessions": sessions}
            )

        except Exception as e:
            logger.exception("Error listing sessions")
            return FlowToolResult(
                success=False,
                content=f"❌ Error: {e}",
                execution_time=time.time() - start_time,
                error=str(e)
            )

    # -------------------------------------------------------------------------
    # ROUTER
    # -------------------------------------------------------------------------

    async def route(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        app: "App"
    ) -> FlowToolResult:
        """
        Route tool call to appropriate handler.

        Args:
            tool_name: Name of the flow tool
            arguments: Tool arguments
            app: ToolBoxV2 App instance

        Returns:
            FlowToolResult
        """
        if tool_name == "flow_list_available":
            return await self.handle_list_available(
                app,
                filter_str=arguments.get("filter")
            )

        elif tool_name == "flow_start":
            return await self.handle_start(
                app,
                flow_name=arguments["flow_name"],
                session_id=arguments.get("session_id"),
                auto_start=arguments.get("auto_start", True),
                kwargs=arguments.get("kwargs")
            )

        elif tool_name == "flow_input":
            return await self.handle_input(
                session_id=arguments["session_id"],
                input_value=arguments["input"]
            )

        elif tool_name == "flow_status":
            return await self.handle_status(
                session_id=arguments["session_id"]
            )

        elif tool_name == "flow_cancel":
            return await self.handle_cancel(
                session_id=arguments["session_id"]
            )

        elif tool_name == "flow_list_sessions":
            return await self.handle_list_sessions()

        else:
            return FlowToolResult(
                success=False,
                content=f"❌ Unknown flow tool: {tool_name}",
                execution_time=0,
                error="UnknownTool"
            )


# =============================================================================
# FACTORY
# =============================================================================


def create_flow_handlers(
    app: Optional["App"] = None,
    max_sessions: int = 100,
    timeout: int = 3600
) -> FlowHandlers:
    """
    Create FlowHandlers with configured manager.

    Args:
        app: ToolBoxV2 App instance (can be set later)
        max_sessions: Maximum concurrent sessions
        timeout: Session timeout in seconds

    Returns:
        Configured FlowHandlers
    """
    manager = FlowSessionManager(app=app, max_sessions=max_sessions, timeout=timeout)
    return FlowHandlers(manager)
