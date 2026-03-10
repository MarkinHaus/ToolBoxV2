"""
Sub-Agent Resume Extension for ExecutionEngine

This extension allows the main agent to resume a sub-agent that hit max iterations.

Features:
- Sub-agents can signal they are resumable (made progress)
- Main agent can resume with additional iterations + budget
- Context is preserved (working_history, tools_used, etc.)
- Optional additional context can be injected on resume

Author: FlowAgent Team
Date: 2025-01-30
"""

from typing import Optional


# ============================================================================
# NEW TOOL: resume_sub_agent
# ============================================================================

RESUME_SUB_AGENT_TOOL = {
    "type": "function",
    "function": {
        "name": "resume_sub_agent",
        "description": """Resume a paused or max-iterations sub-agent with additional iterations.

Use this when:
- A sub-agent hit max_iterations but made progress
- You want to give it more iterations to complete the task
- The sub-agent status is 'max_iterations' or 'paused'

The sub-agent will continue from where it stopped with its full context preserved.

You can optionally provide additional context that will be injected into the sub-agent's working history.""",
        "parameters": {
            "type": "object",
            "properties": {
                "sub_agent_id": {
                    "type": "string",
                    "description": "ID of the sub-agent to resume (from spawn_sub_agent or wait_for result)"
                },
                "additional_iterations": {
                    "type": "integer",
                    "description": "How many more iterations to allow (default: 10)",
                    "default": 10
                },
                "additional_budget": {
                    "type": "integer",
                    "description": "Additional token budget (default: 3000)",
                    "default": 3000
                },
                "context": {
                    "type": "string",
                    "description": "Optional additional context/instructions to inject into the sub-agent (e.g., 'Focus on testing error cases', 'Use the new API instead')"
                },
                "wait": {
                    "type": "boolean",
                    "description": "Wait for completion (true) or run async (false)",
                    "default": True
                }
            },
            "required": ["sub_agent_id"]
        }
    }
}


# ============================================================================
# ExecutionEngine Extension Methods
# ============================================================================

class SubAgentResumeExtension:
    """Mixin for ExecutionEngine to add resume capabilities"""

    async def _handle_sub_agent_max_iterations(
        self,
        ctx: 'ExecutionContext',
        query: str,
        max_iterations: int
    ) -> tuple[str, bool]:
        """
        Handle max iterations for sub-agent.

        Returns: (response, should_mark_resumable)
        """
        from toolboxv2.mods.isaa.base.Agent.execution_engine import HistoryCompressor

        # Generate graceful response
        summary = HistoryCompressor.compress_to_summary(ctx.working_history, ctx.run_id)
        summary_text = summary["content"] if summary else "Keine Aktionen durchgeführt."

        response = f"""⏱️ Max Iterations erreicht ({max_iterations})

{summary_text}

**Was wurde erreicht:**
- Durchgeführte Tools: {', '.join(ctx.tools_used[-5:])}
- Aktuelle Iteration: {ctx.current_iteration}/{max_iterations}
- Token verwendet: {self._tokens_used}

**Status:** Die Aufgabe ist noch nicht abgeschlossen, aber es wurde Fortschritt gemacht.

**Main-Agent:** Du kannst mich mit `resume_sub_agent('{ctx.run_id}')` fortsetzen, wenn du denkst, dass ich die Aufgabe mit mehr Iterationen abschließen kann."""

        # Mark as resumable if tools were used (= progress was made)
        should_mark_resumable = len(ctx.tools_used) > 0

        return response, should_mark_resumable

    async def _tool_resume_sub_agent(
        self,
        sub_agent_id: str,
        additional_iterations: int = 10,
        additional_budget: int = 3000,
        wait: bool = True,
        context: Optional[str] = None
    ) -> str:
        if not self._sub_agent_manager:
            return "ERROR: SubAgentManager not initialized (nur Main-Agent kann resumieren)"

        if self.is_sub_agent:
            return "ERROR: Sub-Agents können keine anderen Sub-Agents resumieren"

        state = self._sub_agent_manager._sub_agents.get(sub_agent_id)
        if not state:
            return f"ERROR: Sub-Agent '{sub_agent_id}' nicht gefunden"

        if not getattr(state, 'resumable', False):
            return f"ERROR: Sub-Agent '{sub_agent_id}' kann nicht resumed werden (status: {state.status.value})"

        from toolboxv2.mods.isaa.base.Agent.sub_agent import SubAgentStatus
        if state.status not in [SubAgentStatus.MAX_ITERATIONS, SubAgentStatus.PAUSED]:
            return f"ERROR: Sub-Agent '{sub_agent_id}' ist nicht pausiert (status: {state.status.value})"

        ctx = getattr(state, 'execution_context', None)
        if not ctx:
            return f"ERROR: Sub-Agent '{sub_agent_id}' hat keinen erhaltenen Context"

        try:
            state.status = SubAgentStatus.RUNNING

            if context:
                ctx.working_history.append({
                    "role": "system",
                    "content": f"ZUSÄTZLICHER KONTEXT VOM MAIN-AGENT:\n{context}"
                })

            new_budget = getattr(state, 'original_budget', 5000) + additional_budget
            if state.engine:
                state.engine.sub_agent_budget = new_budget

            result = await state.engine.execute(
                query=state.task,
                session_id=state.session_id,
                max_iterations=additional_iterations,
                ctx=ctx,
                get_ctx=True
            )

            if isinstance(result, tuple):
                final_response, ctx = result
            else:
                final_response = result

            state.tokens_used += getattr(state.engine, '_tokens_used', 0)
            state.iterations_used += ctx.current_iteration

            if ctx.current_iteration >= additional_iterations:
                state.status = SubAgentStatus.MAX_ITERATIONS
                state.resumable = len(ctx.tools_used) > 0
                state.execution_context = ctx if state.resumable else None
                success_msg = f"⏱️ Wieder Max-Iterations erreicht (total: {state.iterations_used})"
            else:
                state.status = SubAgentStatus.COMPLETED
                state.resumable = False
                state.execution_context = None
                success_msg = "✅ Sub-Agent erfolgreich fortgesetzt und abgeschlossen"

            state.result = final_response

            # Collect written files
            try:
                vfs = self._current_session.vfs
                ls_result = vfs.ls(state.output_dir, recursive=True)
                if ls_result.get("success"):
                    state.files_written = [f["path"] for f in ls_result.get("files", [])]
            except Exception:
                pass

            # Update SubAgentResult in _completed if present
            from toolboxv2.mods.isaa.base.Agent.sub_agent import SubAgentResult
            from datetime import datetime
            duration = 0.0
            if state.started_at:
                duration = (datetime.now() - state.started_at).total_seconds()

            completed_result = SubAgentResult(
                id=sub_agent_id,
                success=state.status == SubAgentStatus.COMPLETED,
                status=state.status,
                result=final_response,
                error=None,
                output_dir=state.output_dir,
                files_written=state.files_written,
                tokens_used=state.tokens_used,
                duration_seconds=duration,
                task=state.task,
                max_iterations_reached=(state.status == SubAgentStatus.MAX_ITERATIONS),
                resumable=state.resumable,
                iterations_used=state.iterations_used,
                execution_context=state.execution_context,
            )
            self._sub_agent_manager._completed[sub_agent_id] = completed_result

            files_str = ', '.join(state.files_written[:5])
            return (
                f"{success_msg}\n"
                f"Output: {state.output_dir}\n"
                f"Iterations: {state.iterations_used}\n"
                f"Tokens: {state.tokens_used}\n"
                f"Files: {files_str}\n\n"
                f"Result: {final_response[:500]}"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            state.status = SubAgentStatus.FAILED
            state.result = f"Resume failed: {str(e)}"
            return f"ERROR: Resume fehlgeschlagen: {str(e)}"
