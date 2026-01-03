"""
Data Collection for FlowAgent RL Training

Collects training traces from FlowAgent checkpoints and runtime.
Extracts detailed execution information including tool calls,
reasoning steps, and actual outcomes - not just final outputs.
"""

import os
import json
import pickle
import glob
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Callable
from pathlib import Path
import hashlib


@dataclass
class ToolCallTrace:
    """Single tool call with inputs, outputs, and success status"""
    tool_name: str
    arguments: dict
    result: Any
    success: bool
    duration_ms: float
    error: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ReasoningStep:
    """Single reasoning step from LLMReasonerNode"""
    step_type: str  # internal_reasoning, outline_step, task_delegation
    content: str
    confidence: float = 0.0
    insights: list = field(default_factory=list)
    issues: list = field(default_factory=list)


@dataclass 
class ExecutionTrace:
    """Complete execution trace for a single agent run"""
    
    # Identification
    trace_id: str = ""
    session_id: str = ""
    timestamp: str = ""
    
    # Input
    user_query: str = ""
    
    # Execution Details (what the agent ACTUALLY did)
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    reasoning_steps: list[ReasoningStep] = field(default_factory=list)
    tasks_created: list[dict] = field(default_factory=list)
    tasks_completed: list[dict] = field(default_factory=list)
    tasks_failed: list[dict] = field(default_factory=list)
    
    # Outputs
    final_response: str = ""
    
    # Metrics
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost: float = 0.0
    execution_duration_ms: float = 0.0
    llm_calls_count: int = 0
    
    # Labels (for training)
    label: Optional[bool] = None  # True = good, False = bad
    reward_score: Optional[float] = None  # 0.0 - 1.0
    manual_review: bool = False
    review_notes: str = ""
    
    def __post_init__(self):
        if not self.trace_id:
            # Generate unique ID from content
            content = f"{self.session_id}:{self.user_query}:{self.timestamp}"
            self.trace_id = hashlib.md5(content.encode()).hexdigest()[:12]
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        """Convert to serializable dict"""
        data = asdict(self)
        # Convert nested dataclasses
        data["tool_calls"] = [asdict(tc) if hasattr(tc, "__dataclass_fields__") else tc 
                             for tc in self.tool_calls]
        data["reasoning_steps"] = [asdict(rs) if hasattr(rs, "__dataclass_fields__") else rs 
                                   for rs in self.reasoning_steps]
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionTrace":
        """Reconstruct from dict"""
        # Convert tool calls
        tool_calls = []
        for tc in data.get("tool_calls", []):
            if isinstance(tc, dict):
                tool_calls.append(ToolCallTrace(**tc))
            else:
                tool_calls.append(tc)
        data["tool_calls"] = tool_calls
        
        # Convert reasoning steps
        reasoning_steps = []
        for rs in data.get("reasoning_steps", []):
            if isinstance(rs, dict):
                reasoning_steps.append(ReasoningStep(**rs))
            else:
                reasoning_steps.append(rs)
        data["reasoning_steps"] = reasoning_steps
        
        return cls(**data)


class TraceCollector:
    """
    Collects execution traces from FlowAgent.
    
    Hooks into agent execution to capture detailed information about
    what the agent actually did, not just the final response.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize trace collector.
        
        Args:
            storage_path: Where to store collected traces
        """
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            try:
                from toolboxv2 import get_app
                self.storage_path = Path(get_app().data_dir) / "rl_traces"
            except:
                self.storage_path = Path.home() / ".toolbox" / "rl_traces"
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.current_trace: Optional[ExecutionTrace] = None
        self.traces: list[ExecutionTrace] = []
        
        # Hooks for agent integration
        self._tool_call_hook: Optional[Callable] = None
        self._reasoning_hook: Optional[Callable] = None
    
    def start_trace(self, session_id: str, user_query: str) -> ExecutionTrace:
        """Start collecting a new execution trace"""
        self.current_trace = ExecutionTrace(
            session_id=session_id,
            user_query=user_query,
            timestamp=datetime.now().isoformat()
        )
        return self.current_trace
    
    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        result: Any,
        success: bool,
        duration_ms: float,
        error: Optional[str] = None
    ):
        """Record a tool call during execution"""
        if self.current_trace is None:
            return
        
        trace = ToolCallTrace(
            tool_name=tool_name,
            arguments=arguments,
            result=str(result)[:2000] if result else "",  # Truncate large results
            success=success,
            duration_ms=duration_ms,
            error=error
        )
        self.current_trace.tool_calls.append(trace)
    
    def record_reasoning_step(
        self,
        step_type: str,
        content: str,
        confidence: float = 0.0,
        insights: list = None,
        issues: list = None
    ):
        """Record a reasoning step"""
        if self.current_trace is None:
            return
        
        step = ReasoningStep(
            step_type=step_type,
            content=content[:1000],  # Truncate
            confidence=confidence,
            insights=insights or [],
            issues=issues or []
        )
        self.current_trace.reasoning_steps.append(step)
    
    def record_task(self, task_data: dict, status: str):
        """Record task creation/completion/failure"""
        if self.current_trace is None:
            return
        
        task_info = {
            "task_id": task_data.get("id", "unknown"),
            "type": task_data.get("type", "unknown"),
            "description": str(task_data.get("description", ""))[:500],
            "timestamp": datetime.now().isoformat()
        }
        
        if status == "created":
            self.current_trace.tasks_created.append(task_info)
        elif status == "completed":
            task_info["result"] = str(task_data.get("result", ""))[:500]
            self.current_trace.tasks_completed.append(task_info)
        elif status == "failed":
            task_info["error"] = str(task_data.get("error", ""))[:500]
            self.current_trace.tasks_failed.append(task_info)
    
    def finish_trace(
        self,
        final_response: str,
        total_tokens_in: int = 0,
        total_tokens_out: int = 0,
        total_cost: float = 0.0,
        execution_duration_ms: float = 0.0,
        llm_calls_count: int = 0
    ) -> ExecutionTrace:
        """Complete the current trace and save it"""
        if self.current_trace is None:
            raise ValueError("No trace in progress")
        
        self.current_trace.final_response = final_response
        self.current_trace.total_tokens_in = total_tokens_in
        self.current_trace.total_tokens_out = total_tokens_out
        self.current_trace.total_cost = total_cost
        self.current_trace.execution_duration_ms = execution_duration_ms
        self.current_trace.llm_calls_count = llm_calls_count
        
        # Save trace
        self._save_trace(self.current_trace)
        self.traces.append(self.current_trace)
        
        finished = self.current_trace
        self.current_trace = None
        return finished
    
    def _save_trace(self, trace: ExecutionTrace):
        """Save trace to storage"""
        date_folder = self.storage_path / datetime.now().strftime("%Y-%m-%d")
        date_folder.mkdir(exist_ok=True)
        
        filepath = date_folder / f"{trace.trace_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trace.to_dict(), f, indent=2, ensure_ascii=False, default=str)
    
    def load_traces(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        labeled_only: bool = False,
        min_tool_calls: int = 0
    ) -> list[ExecutionTrace]:
        """
        Load traces from storage with optional filtering.
        
        Args:
            start_date: Filter traces from this date (YYYY-MM-DD)
            end_date: Filter traces until this date
            labeled_only: Only return traces that have been labeled
            min_tool_calls: Minimum number of tool calls required
        
        Returns:
            List of ExecutionTrace objects
        """
        traces = []
        
        # Find all trace files
        pattern = str(self.storage_path / "**" / "*.json")
        files = glob.glob(pattern, recursive=True)
        
        for filepath in files:
            try:
                # Date filtering based on folder name
                folder_name = Path(filepath).parent.name
                if start_date and folder_name < start_date:
                    continue
                if end_date and folder_name > end_date:
                    continue
                
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                trace = ExecutionTrace.from_dict(data)
                
                # Apply filters
                if labeled_only and trace.label is None:
                    continue
                if min_tool_calls > 0 and len(trace.tool_calls) < min_tool_calls:
                    continue
                
                traces.append(trace)
                
            except Exception as e:
                print(f"Warning: Could not load trace {filepath}: {e}")
                continue
        
        return traces
    
    def get_unlabeled_traces(self, limit: int = 100) -> list[ExecutionTrace]:
        """Get traces that need manual labeling"""
        all_traces = self.load_traces()
        unlabeled = [t for t in all_traces if t.label is None]
        return unlabeled[:limit]
    
    def label_trace(self, trace_id: str, label: bool, notes: str = ""):
        """Apply manual label to a trace"""
        # Find and update the trace file
        pattern = str(self.storage_path / "**" / f"{trace_id}.json")
        files = glob.glob(pattern, recursive=True)
        
        if not files:
            raise ValueError(f"Trace {trace_id} not found")
        
        filepath = files[0]
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        data["label"] = label
        data["manual_review"] = True
        data["review_notes"] = notes
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def get_statistics(self) -> dict:
        """Get statistics about collected traces"""
        traces = self.load_traces()
        
        if not traces:
            return {"total": 0}
        
        labeled = [t for t in traces if t.label is not None]
        positive = [t for t in labeled if t.label == True]
        
        return {
            "total": len(traces),
            "labeled": len(labeled),
            "unlabeled": len(traces) - len(labeled),
            "positive_labels": len(positive),
            "negative_labels": len(labeled) - len(positive),
            "avg_tool_calls": sum(len(t.tool_calls) for t in traces) / len(traces),
            "avg_reasoning_steps": sum(len(t.reasoning_steps) for t in traces) / len(traces),
            "avg_cost": sum(t.total_cost for t in traces) / len(traces),
        }


class CheckpointLoader:
    """
    Loads and extracts training data from FlowAgent checkpoints.
    
    Handles overlapping data from multiple checkpoints and
    deduplicates based on trace IDs.
    """
    
    def __init__(self, agent_name: str, checkpoint_path: Optional[str] = None):
        """
        Initialize checkpoint loader.
        
        Args:
            agent_name: Name of the FlowAgent
            checkpoint_path: Path to checkpoint directory
        """
        self.agent_name = agent_name
        
        if checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)
        else:
            try:
                from toolboxv2 import get_app
                self.checkpoint_path = Path(get_app().data_dir) / "Agents" / "checkpoint" / agent_name
            except:
                self.checkpoint_path = Path.home() / ".toolbox" / "checkpoints" / agent_name
    
    def list_checkpoints(self) -> list[dict]:
        """List available checkpoints with metadata"""
        if not self.checkpoint_path.exists():
            return []
        
        checkpoints = []
        for filepath in self.checkpoint_path.glob("*.pkl"):
            try:
                stat = filepath.stat()
                checkpoints.append({
                    "path": str(filepath),
                    "filename": filepath.name,
                    "size_mb": stat.st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except Exception as e:
                print(f"Warning: Could not stat {filepath}: {e}")
        
        checkpoints.sort(key=lambda x: x["modified"], reverse=True)
        return checkpoints
    
    def load_checkpoint(self, filepath: str) -> dict:
        """Load a single checkpoint file"""
        with open(filepath, "rb") as f:
            return pickle.load(f)
    
    def extract_traces_from_checkpoint(self, checkpoint) -> list[ExecutionTrace]:
        """
        Extract execution traces from a checkpoint.
        
        Looks into:
        - session_data for conversation history
        - task_state for task execution details
        - variable_scopes for stored results
        - conversation_history for context
        """
        traces = []
        
        # Extract from session data
        session_data = getattr(checkpoint, "session_data", {}) or {}
        for session_id, session_info in session_data.items():
            history = session_info.get("history", [])
            
            # Pair user messages with assistant responses
            i = 0
            while i < len(history) - 1:
                if history[i].get("role") == "user":
                    user_msg = history[i].get("content", "")
                    
                    # Find next assistant response
                    j = i + 1
                    while j < len(history) and history[j].get("role") != "assistant":
                        j += 1
                    
                    if j < len(history):
                        assistant_msg = history[j].get("content", "")
                        
                        trace = ExecutionTrace(
                            session_id=session_id,
                            user_query=user_msg,
                            final_response=assistant_msg,
                            timestamp=history[i].get("timestamp", "")
                        )
                        traces.append(trace)
                        i = j + 1
                    else:
                        i += 1
                else:
                    i += 1
        
        # Extract from task state
        task_state = getattr(checkpoint, "task_state", {}) or {}
        for task_id, task_data in task_state.items():
            # Check if this is a tool task with execution info
            if task_data.get("type") == "ToolTask":
                tool_name = task_data.get("tool_name", "")
                if tool_name:
                    # This would need to be matched with existing traces
                    # or create synthetic traces for standalone tool executions
                    pass
        
        return traces
    
    def load_all_traces(self, deduplicate: bool = True) -> list[ExecutionTrace]:
        """
        Load traces from all checkpoints.
        
        Args:
            deduplicate: Remove duplicate traces based on trace_id
        
        Returns:
            List of unique ExecutionTrace objects
        """
        all_traces = []
        seen_ids = set()
        
        checkpoints = self.list_checkpoints()
        for cp_info in checkpoints:
            try:
                checkpoint = self.load_checkpoint(cp_info["path"])
                traces = self.extract_traces_from_checkpoint(checkpoint)
                
                for trace in traces:
                    if deduplicate:
                        if trace.trace_id in seen_ids:
                            continue
                        seen_ids.add(trace.trace_id)
                    
                    all_traces.append(trace)
                    
            except Exception as e:
                print(f"Warning: Could not load checkpoint {cp_info['path']}: {e}")
        
        return all_traces
    
    def generate_synthetic_tasks(self, num_tasks: int = 100) -> list[dict]:
        """
        Generate synthetic training tasks from checkpoint data.
        
        Analyzes patterns in successful executions to create
        similar training prompts.
        """
        traces = self.load_all_traces()
        
        # Collect patterns from successful traces
        patterns = {
            "code_tasks": [],
            "shell_tasks": [],
            "general_tasks": [],
            "interaction_tasks": []
        }
        
        for trace in traces:
            if trace.label == True or len(trace.tasks_completed) > 0:
                query = trace.user_query.lower()
                
                if any(kw in query for kw in ["code", "python", "script", "function", "class"]):
                    patterns["code_tasks"].append(trace.user_query)
                elif any(kw in query for kw in ["run", "execute", "shell", "command", "terminal"]):
                    patterns["shell_tasks"].append(trace.user_query)
                elif any(kw in query for kw in ["remind", "help", "suggest", "what should"]):
                    patterns["interaction_tasks"].append(trace.user_query)
                else:
                    patterns["general_tasks"].append(trace.user_query)
        
        # Generate variations
        synthetic = []
        for category, examples in patterns.items():
            if examples:
                # Sample from existing patterns with variations
                for example in examples[:num_tasks // 4]:
                    synthetic.append({
                        "prompt": example,
                        "category": category,
                        "source": "checkpoint_derived"
                    })
        
        return synthetic[:num_tasks]


def hook_into_agent(agent, trace_collector: TraceCollector):
    """
    Hook TraceCollector into a FlowAgent instance.
    
    This monkey-patches the agent to record execution traces.
    Call this after creating the agent but before running it.
    
    Args:
        agent: FlowAgent instance
        trace_collector: TraceCollector instance
    """
    original_a_run = agent.a_run
    original_arun_function = agent.arun_function
    
    async def traced_a_run(query, session_id="default", **kwargs):
        # Start trace
        trace_collector.start_trace(session_id, query)
        
        try:
            # Run original
            result = await original_a_run(query, session_id=session_id, **kwargs)
            
            # Finish trace with metrics
            trace_collector.finish_trace(
                final_response=result,
                total_tokens_in=agent.total_tokens_in,
                total_tokens_out=agent.total_tokens_out,
                total_cost=agent.total_cost_accumulated,
                llm_calls_count=agent.total_llm_calls
            )
            
            return result
            
        except Exception as e:
            # Record failed execution
            if trace_collector.current_trace:
                trace_collector.current_trace.tasks_failed.append({
                    "error": str(e),
                    "type": "execution_error"
                })
                trace_collector.finish_trace(
                    final_response=f"Error: {e}",
                    total_tokens_in=agent.total_tokens_in,
                    total_tokens_out=agent.total_tokens_out,
                    total_cost=agent.total_cost_accumulated
                )
            raise
    
    async def traced_arun_function(function_name, *args, **kwargs):
        import time
        start = time.perf_counter()
        error = None
        success = True
        result = None
        
        try:
            result = await original_arun_function(function_name, *args, **kwargs)
            return result
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            trace_collector.record_tool_call(
                tool_name=function_name,
                arguments={"args": args, "kwargs": kwargs},
                result=result,
                success=success,
                duration_ms=duration_ms,
                error=error
            )
    
    # Apply hooks
    agent.a_run = traced_a_run
    agent.arun_function = traced_arun_function
    
    # Store collector reference
    agent._trace_collector = trace_collector
    
    print(f"TraceCollector hooked into agent {agent.amd.name}")
