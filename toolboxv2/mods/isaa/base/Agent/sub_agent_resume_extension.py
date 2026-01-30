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
        """
        Resume a sub-agent that hit max iterations.
        
        Args:
            sub_agent_id: ID of sub-agent to resume
            additional_iterations: Additional iterations to allow
            additional_budget: Additional token budget
            wait: Wait for completion
            context: Optional additional context to inject
            
        Returns:
            Result message
        """
        if not self._sub_agent_manager:
            return "ERROR: SubAgentManager not initialized (nur Main-Agent kann resumieren)"
        
        if self.is_sub_agent:
            return "ERROR: Sub-Agents können keine anderen Sub-Agents resumieren"
        
        # Get sub-agent state
        state = self._sub_agent_manager._sub_agents.get(sub_agent_id)
        if not state:
            return f"ERROR: Sub-Agent '{sub_agent_id}' nicht gefunden"
        
        # Check if resumable
        if not hasattr(state, 'resumable') or not state.resumable:
            return f"ERROR: Sub-Agent '{sub_agent_id}' kann nicht resumed werden (status: {state.status.value})"
        
        # Check status
        from toolboxv2.mods.isaa.base.Agent.sub_agent import SubAgentStatus
        if state.status not in [SubAgentStatus.MAX_ITERATIONS, SubAgentStatus.PAUSED]:
            return f"ERROR: Sub-Agent '{sub_agent_id}' ist nicht pausiert (status: {state.status.value})"
        
        # Check if context is preserved
        if not hasattr(state, 'execution_context') or not state.execution_context:
            return f"ERROR: Sub-Agent '{sub_agent_id}' hat keinen erhaltenen Context (kann nicht resumed werden)"
        
        try:
            # Update status
            state.status = SubAgentStatus.RUNNING
            
            # Get preserved context
            ctx = state.execution_context
            
            # Inject additional context if provided
            if context:
                ctx.working_history.append({
                    "role": "system",
                    "content": f"ZUSÄTZLICHER KONTEXT VOM MAIN-AGENT:\n{context}"
                })
                print(f"[Resume] Injected additional context: {context[:100]}...")
            
            # Increase budget
            original_budget = getattr(state, 'original_budget', 5000)
            new_budget = original_budget + additional_budget
            if hasattr(state, 'engine'):
                state.engine.sub_agent_budget = new_budget
            
            # Resume execution with existing context
            result = await state.engine.execute(
                query=state.task,
                session_id=state.session_id,
                max_iterations=additional_iterations,
                ctx=ctx,  # Pass existing context!
                get_ctx=True
            )
            
            if isinstance(result, tuple):
                final_response, ctx = result
            else:
                final_response = result
                ctx = state.execution_context
            
            # Update state
            state.tokens_used += getattr(state.engine, '_tokens_used', 0)
            state.iterations_used += ctx.current_iteration
            
            # Check if completed or hit max again
            if ctx.current_iteration >= additional_iterations:
                state.status = SubAgentStatus.MAX_ITERATIONS
                if hasattr(state, 'result'):
                    state.result.success = False
                    state.result.max_iterations_reached = True
                    state.result.resumable = len(ctx.tools_used) > 0
                success_msg = f"⏱️ Wieder Max-Iterations erreicht (total: {state.iterations_used})"
            else:
                state.status = SubAgentStatus.COMPLETED
                if hasattr(state, 'result'):
                    state.result.success = True
                    state.result.max_iterations_reached = False
                    state.result.resumable = False
                success_msg = "✅ Sub-Agent erfolgreich fortgesetzt und abgeschlossen"
            
            # Update result
            if hasattr(state, 'result'):
                state.result.result = final_response
                state.result.status = state.status
                state.result.iterations_used = state.iterations_used
                state.result.tokens_used = state.tokens_used
            
            # Update files written
            try:
                vfs = self._current_session.vfs
                files = vfs.list_files(state.output_dir)
                if hasattr(state, 'result'):
                    state.result.files_written = [f.path for f in files]
                else:
                    state.files_written = [f.path for f in files]
            except:
                pass
            
            if wait:
                files_str = ', '.join(state.result.files_written[:5]) if hasattr(state, 'result') else ', '.join(state.files_written[:5])
                return f"""{success_msg}
Output: {state.output_dir}
Iterations: {state.iterations_used}
Tokens: {state.tokens_used}
Files: {files_str}

Result: {final_response[:500]}"""
            else:
                return f"🔄 Sub-Agent '{sub_agent_id}' resumed (async)"
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            state.status = SubAgentStatus.FAILED
            if hasattr(state, 'result'):
                state.result.success = False
                state.result.error = f"Resume failed: {str(e)}"
            return f"ERROR: Resume fehlgeschlagen: {str(e)}"
