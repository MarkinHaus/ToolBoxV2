"""
Execution Engine Patch - Enable Default Tools in ALL Responses

This patch modifies the ExecutionEngine to:
1. ALWAYS include default tools (VFS, context, meta)
2. Allow tool calling even in "immediate" responses
3. Enforce honesty by requiring tools for factual claims

Apply by importing in execution_engine.py:
    from toolboxv2.mods.isaa.base.Agent.execution_engine_patch import patch_execution_engine
    patch_execution_engine(ExecutionEngine)

Author: FlowAgent V2
"""

from typing import TYPE_CHECKING
import asyncio

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionEngine, ExecutionState
    from toolboxv2.mods.isaa.base.Agent.agent_session import AgentSession


def patch_execution_engine(ExecutionEngineClass):
    """
    Patch ExecutionEngine to always include default tools.

    Changes:
    1. _immediate_response now can use tools
    2. Default tools always available
    3. Honesty enforcement
    """

    # Store original methods
    _original_immediate_response = ExecutionEngineClass._immediate_response
    _original_react_loop = ExecutionEngineClass._react_loop
    _original_get_action_native = ExecutionEngineClass._get_action_native

    async def patched_immediate_response(
        self,
        state: 'ExecutionState',
        session: 'AgentSession',
        with_context: bool
    ) -> str:
        """
        Modified immediate response that CAN use default tools.

        Instead of just generating text, we:
        1. Include default tools in the call
        2. Let the model decide if it needs them
        3. Execute any tool calls
        4. Then generate final response
        """
        from toolboxv2.mods.isaa.base.Agent.default_tools import (
            get_default_tools_litellm,
            is_default_tool,
            create_default_tools_handler
        )

        self._emit_intermediate("Bearbeite Anfrage...")

        # Get default tools
        default_tools = get_default_tools_litellm()

        # Build messages
        messages = session.get_history_for_llm(last_n=5)

        # System message with VFS context
        vfs_context = session.build_vfs_context()
        system_msg = self.agent.amd.get_system_message()

        # Add instruction about using tools for facts
        honesty_instruction = """
WICHTIG: Du hast Zugriff auf Tools. Nutze sie wenn:
- Du Informationen aus Dateien brauchst → vfs_read, vfs_list
- Du dir unsicher bist → get_context
- Du etwas speichern willst → vfs_write, remember

EHRLICHKEITS-REGEL:
- Behaupte NUR was du durch Tools verifiziert hast
- Sage "Ich weiß nicht" wenn du keine Information hast
- Erfinde KEINE Fakten
"""

        messages = [
            {"role": "system", "content": f"{system_msg}\n\n{honesty_instruction}\n\n{vfs_context}"}
        ] + messages

        # Make LLM call WITH tools
        response = await self.agent.a_run_llm_completion(
            messages=messages,
            tools=default_tools,
            tool_choice="auto",  # Let model decide
            model_preference="fast",
            stream=False,
            get_response_message=True,
            task_id=f"{state.execution_id}_immediate_with_tools",
            session_id=state.session_id,
            with_context=False
        )

        # Check for tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Create handler
            handler = create_default_tools_handler(self.agent, session)

            # Execute each tool call
            tool_results = []
            for tc in response.tool_calls:
                import json
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except:
                    args = {}

                self._emit_intermediate(f"Verwende {tc.function.name}...")

                try:
                    result = await handler.execute(tc.function.name, **args)
                    tool_results.append({
                        "tool": tc.function.name,
                        "result": result
                    })

                    # Check for control flow tools
                    if isinstance(result, dict):
                        if result.get("type") == "final_answer":
                            state.final_answer = result.get("answer", "")
                            state.success = True
                            state.phase = "completed"
                            return state.final_answer

                        if result.get("type") == "need_info":
                            state.final_answer = f"Mir fehlt: {result.get('missing', 'Information')}"
                            state.success = True
                            return state.final_answer

                except Exception as e:
                    tool_results.append({
                        "tool": tc.function.name,
                        "error": str(e)
                    })

            # Generate final response with tool results
            tool_context = "\n".join([
                f"Tool {tr['tool']}: {tr.get('result', tr.get('error', 'Error'))}"
                for tr in tool_results
            ])

            messages.append({
                "role": "assistant",
                "content": response.content or ""
            })
            messages.append({
                "role": "user",
                "content": f"Tool-Ergebnisse:\n{tool_context}\n\nBitte antworte basierend auf diesen Ergebnissen."
            })

            final_response = await self.agent.a_run_llm_completion(
                messages=messages,
                model_preference="fast",
                stream=False,
                task_id=f"{state.execution_id}_immediate_final",
                session_id=state.session_id
            )

            state.final_answer = final_response
        else:
            # No tool calls - use direct response
            state.final_answer = response.content if hasattr(response, 'content') else str(response)

        state.success = True
        state.phase = "completed"

        return state.final_answer

    async def patched_get_action_native(
        self,
        state: 'ExecutionState',
        session: 'AgentSession',
        vfs_context: str
    ):
        """
        Modified to ALWAYS include default tools alongside selected tools.
        """
        from toolboxv2.mods.isaa.base.Agent.default_tools import get_default_tools_litellm
        from toolboxv2.mods.isaa.base.Agent.execution_engine import VFS_TOOLS_LITELLM, REACT_SYSTEM_PROMPT
        import json

        # Build messages
        system_prompt = REACT_SYSTEM_PROMPT.format(
            max_open_files=5,
            tools=", ".join(state.selected_tools + ["+ default tools"])
        )

        messages = [
            {"role": "system", "content": f"{system_prompt}\n\n{vfs_context}"},
            {"role": "user", "content": state.query}
        ]

        # Add history
        for i, (thought, obs) in enumerate(zip(state.thoughts, state.observations)):
            messages.append({"role": "assistant", "content": thought})
            messages.append({"role": "user", "content": f"Observation: {obs}"})

        # Get selected tools in LiteLLM format
        selected_tools_litellm = [
            t for t in self.agent.tool_manager.get_all_litellm()
            if any(t['function']['name'] == name for name in state.selected_tools)
        ]

        # ALWAYS add default tools + VFS tools
        default_tools = get_default_tools_litellm()

        # Combine: default + VFS + selected (avoid duplicates)
        all_tool_names = set()
        all_tools = []

        for tool in default_tools + VFS_TOOLS_LITELLM + selected_tools_litellm:
            name = tool['function']['name']
            if name not in all_tool_names:
                all_tool_names.add(name)
                all_tools.append(tool)

        # Make LLM call
        response = await self.agent.a_run_llm_completion(
            messages=messages,
            tools=all_tools,
            tool_choice="auto",
            model_preference="fast" if not state.escalated else "complex",
            stream=False,
            get_response_message=True,
            task_id=f"{state.execution_id}_react_{state.iteration}",
            session_id=state.session_id,
            with_context=False
        )

        # Parse response
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tc = response.tool_calls[0]  # Take first tool call

            try:
                args = json.loads(tc.function.arguments or "{}")
            except:
                args = {}

            action_type = tc.function.name

            return {
                'type': action_type,
                'tool': tc.function.name,
                'args': args,
                **args
            }

        # No tool call - check for text response
        if hasattr(response, 'content') and response.content:
            state.thoughts.append(response.content)
            return {
                'type': 'thinking',
                'thought': response.content
            }

        return None

    # Apply patches
    ExecutionEngineClass._immediate_response = patched_immediate_response
    ExecutionEngineClass._get_action_native = patched_get_action_native

    print("✓ ExecutionEngine patched with default tools support")

    return ExecutionEngineClass


# =============================================================================
# ALTERNATIVE: Mixin Class
# =============================================================================

class DefaultToolsMixin:
    """
    Mixin that adds default tools support to ExecutionEngine.

    Usage:
        class PatchedExecutionEngine(DefaultToolsMixin, ExecutionEngine):
            pass
    """

    async def _get_all_tools_for_state(self, state: 'ExecutionState') -> list[dict]:
        """Get all tools including defaults"""
        from toolboxv2.mods.isaa.base.Agent.default_tools import get_default_tools_litellm
        from toolboxv2.mods.isaa.base.Agent.execution_engine import VFS_TOOLS_LITELLM

        # Selected tools
        selected = [
            t for t in self.agent.tool_manager.get_all_litellm()
            if t['function']['name'] in state.selected_tools
        ]

        # Combine with defaults
        all_tools = get_default_tools_litellm() + VFS_TOOLS_LITELLM + selected

        # Deduplicate
        seen = set()
        unique = []
        for tool in all_tools:
            name = tool['function']['name']
            if name not in seen:
                seen.add(name)
                unique.append(tool)

        return unique

    async def _execute_default_tool(
        self,
        session: 'AgentSession',
        tool_name: str,
        args: dict
    ):
        """Execute a default tool"""
        from toolboxv2.mods.isaa.base.Agent.default_tools import (
            is_default_tool,
            create_default_tools_handler
        )

        if not is_default_tool(tool_name):
            raise ValueError(f"Not a default tool: {tool_name}")

        handler = create_default_tools_handler(self.agent, session)
        return await handler.execute(tool_name, **args)


# =============================================================================
# AUTO-PATCH ON IMPORT
# =============================================================================

def auto_patch():
    """Auto-patch ExecutionEngine if available"""
    try:
        from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionEngine
        patch_execution_engine(ExecutionEngine)
    except ImportError as e:
        print(f"Failed to auto-patch ExecutionEngine: {e}")
        pass


# Uncomment to auto-patch on import:
auto_patch()
