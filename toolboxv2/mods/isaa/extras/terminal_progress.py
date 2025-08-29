import json
import time
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List, Set
from enum import Enum
from datetime import datetime
from collections import defaultdict, deque

try:
    from rich.console import Console
    from rich.text import Text
    from rich.tree import Tree
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    from rich.columns import Columns
    from rich.progress import Progress, SpinnerColumn, TextColumn

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: Rich not available. Install with: pip install rich")

from toolboxv2.mods.isaa.base.Agent.types import *
class VerbosityMode(Enum):
    MINIMAL = "minimal"  # Nur wichtigste Updates, kompakte Ansicht
    STANDARD = "standard"  # Standard-Detailgrad mit wichtigen Events
    VERBOSE = "verbose"  # Detaillierte Ansicht mit Reasoning und Metriken
    DEBUG = "debug"  # Vollständige Debugging-Info mit JSON
    REALTIME = "realtime"  # Live-Updates mit Spinner und Fortschritt


class NodeStatus(Enum):
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProgressEvent:
    """Enhanced progress event with better error handling"""
    event_type: str
    timestamp: float
    node_name: str
    event_id: str = ""

    #
    agent_name: Optional[str] = None

    # Status information
    status: Optional[NodeStatus] = None
    success: Optional[bool] = None
    error_details: Optional[Dict[str, Any]] = None

    # LLM-specific data
    llm_model: Optional[str] = None
    llm_prompt_tokens: Optional[int] = None
    llm_completion_tokens: Optional[int] = None
    llm_total_tokens: Optional[int] = None
    llm_cost: Optional[float] = None
    llm_duration: Optional[float] = None
    llm_temperature: Optional[float] = None

    # Tool-specific data
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    tool_duration: Optional[float] = None
    tool_success: Optional[bool] = None
    tool_error: Optional[str] = None

    # Node/Routing data
    routing_decision: Optional[str] = None
    routing_from: Optional[str] = None
    routing_to: Optional[str] = None
    node_phase: Optional[str] = None
    node_duration: Optional[float] = None

    # Context data
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    plan_id: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.event_id:
            self.event_id = f"{self.node_name}_{self.event_type}_{int(self.timestamp * 1000000)}"
        if 'error' in self.metadata or 'error_type' in self.metadata:
            if self.error_details is None:
                self.error_details = {}
            self.error_details['error'] = self.metadata.get('error')
            self.error_details['error_type'] = self.metadata.get('error_type')
            self.status = NodeStatus.FAILED
        if self.status == NodeStatus.FAILED:
            self.success = False
        if self.status == NodeStatus.COMPLETED:
            self.success = True


class ExecutionNode:
    """Enhanced execution node with better status management"""

    def __init__(self, name: str, node_type: str = "unknown"):
        self.name = name
        self.node_type = node_type
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
        self.phase: str = "unknown"

        # Enhanced status management
        self.status: NodeStatus = NodeStatus.PENDING
        self.previous_status: Optional[NodeStatus] = None
        self.status_history: List[Dict[str, Any]] = []

        # Error handling
        self.error: Optional[str] = None
        self.error_details: Optional[Dict[str, Any]] = None
        self.retry_count: int = 0

        # Child operations
        self.llm_calls: List[ProgressEvent] = []
        self.tool_calls: List[ProgressEvent] = []
        self.sub_events: List[ProgressEvent] = []

        # Enhanced metadata
        self.reasoning: Optional[str] = None
        self.strategy: Optional[str] = None
        self.routing_from: Optional[str] = None
        self.routing_to: Optional[str] = None
        self.completion_criteria: Optional[Dict[str, Any]] = None

        # Stats
        self.total_cost: float = 0.0
        self.total_tokens: int = 0
        self.performance_metrics: Dict[str, Any] = {}

    def update_status(self, new_status: NodeStatus, reason: str = "", error_details: Dict = None):
        """Update node status with history tracking"""
        if new_status != self.status:
            self.previous_status = self.status
            self.status_history.append({
                "from": self.status.value if self.status else None,
                "to": new_status.value,
                "timestamp": time.time(),
                "reason": reason,
                "error_details": error_details
            })
            self.status = new_status

            if error_details:
                self.error_details = error_details
                self.error = error_details.get("error", reason)

    def add_event(self, event: ProgressEvent):
        """Enhanced event processing with auto-completion detection"""
        # Categorize event
        if event.event_type == "llm_call":
            self.llm_calls.append(event)
            if event.llm_cost:
                self.total_cost += event.llm_cost
            if event.llm_total_tokens:
                self.total_tokens += event.llm_total_tokens

        elif event.event_type == "tool_call":
            self.tool_calls.append(event)

        else:
            self.sub_events.append(event)

        # Update node info from metadata
        if event.metadata:
            if "strategy" in event.metadata:
                self.strategy = event.metadata["strategy"]
            if "reasoning" in event.metadata:
                self.reasoning = event.metadata["reasoning"]

        # Update routing info
        if event.routing_from:
            self.routing_from = event.routing_from
        if event.routing_to:
            self.routing_to = event.routing_to

        # Auto-completion detection
        self._detect_completion(event)

        # Update timing
        if not self.start_time:
            self.start_time = event.timestamp

        # Status updates based on event
        self._update_status_from_event(event)

    def _detect_completion(self, event: ProgressEvent):
        """Detect node completion based on various criteria"""

        # Check for explicit completion signals from flows or the entire execution
        if event.event_type in ["node_exit", "execution_complete", "task_complete"] or event.success:
            # This logic correctly handles the completion of Flows (like TaskManagementFlow)
            if event.node_duration:
                self.duration = event.node_duration
                self.end_time = event.timestamp
                self.update_status(NodeStatus.COMPLETED, "Explicit completion signal")
                return

        # --- KORRIGIERTER ABSCHNITT START ---
        # General auto-completion for simple Nodes (not Flows) after their main action.
        # This replaces the hardcoded rule for just "StrategyOrchestratorNode".
        is_simple_node = "Flow" not in self.name
        is_finalizing_event = event.event_type in ["llm_call", "tool_call", "node_phase"] and event.success

        if is_simple_node and is_finalizing_event:
            # A simple node is often considered "done" after its last successful major operation.
            self.end_time = event.timestamp
            # If the event provides a duration for the whole node, use it. Otherwise, calculate from start.
            if event.node_duration:
                self.duration = event.node_duration
            elif self.start_time:
                self.duration = self.end_time - self.start_time

            self.update_status(NodeStatus.COMPLETED, f"Auto-detected completion after successful '{event.event_type}'")
            return

        # Error-based completion detection
        if event.event_type == "error" or event.success is False:
            self.update_status(NodeStatus.FAILED, "Error detected", {
                "error": event.metadata.get("error", (
                    event.tool_error if hasattr(event, 'tool_error') else "Unknown error") or "Unknown error"),
                "error_type": event.metadata.get("error_type", "UnknownError")
            })
            if event.node_duration:
                self.duration = event.node_duration
                self.end_time = event.timestamp

    def _update_status_from_event(self, event: ProgressEvent):
        """Update status based on incoming events"""

        if event.event_type == "node_enter":
            self.update_status(NodeStatus.STARTING, "Node entered")

        elif event.event_type in ["llm_call", "tool_call"] and self.status == NodeStatus.STARTING:
            self.update_status(NodeStatus.RUNNING, f"Started {event.event_type}")

        elif event.event_type == "error":
            self.update_status(NodeStatus.FAILED, "Error occurred", {
                "error": event.metadata.get("error"),
                "error_type": event.metadata.get("error_type")
            })

    def is_completed(self) -> bool:
        """Check if node is truly completed"""
        return self.status in [NodeStatus.COMPLETED, NodeStatus.FAILED, NodeStatus.SKIPPED]

    def is_active(self) -> bool:
        """Check if node is currently active"""
        return self.status in [NodeStatus.STARTING, NodeStatus.RUNNING, NodeStatus.WAITING]

    def get_status_icon(self) -> str:
        """Get appropriate status icon"""
        icons = {
            NodeStatus.PENDING: "⏸️",
            NodeStatus.STARTING: "🔄",
            NodeStatus.RUNNING: "🔄",
            NodeStatus.WAITING: "⏸️",
            NodeStatus.COMPLETING: "🔄",
            NodeStatus.COMPLETED: "✅",
            NodeStatus.FAILED: "❌",
            NodeStatus.SKIPPED: "⏭️"
        }
        return icons.get(self.status, "❓")

    def get_status_color(self) -> str:
        """Get appropriate color for rich console"""
        colors = {
            NodeStatus.PENDING: "yellow dim",
            NodeStatus.STARTING: "yellow",
            NodeStatus.RUNNING: "blue bold",
            NodeStatus.WAITING: "yellow dim",
            NodeStatus.COMPLETING: "green",
            NodeStatus.COMPLETED: "green bold",
            NodeStatus.FAILED: "red bold",
            NodeStatus.SKIPPED: "cyan dim"
        }
        return colors.get(self.status, "white")

    def get_duration_str(self) -> str:
        """Enhanced duration string with better formatting"""
        if self.duration:
            if self.duration < 1:
                return f"{self.duration * 1000:.0f}ms"
            elif self.duration < 60:
                return f"{self.duration:.1f}s"
            elif self.duration < 3600:
                minutes = int(self.duration // 60)
                seconds = self.duration % 60
                return f"{minutes}m{seconds:.1f}s"
            else:
                hours = int(self.duration // 3600)
                minutes = int((self.duration % 3600) // 60)
                return f"{hours}h{minutes}m"
        elif self.start_time and self.status in [NodeStatus.RUNNING, NodeStatus.STARTING]:
            elapsed = time.time() - self.start_time
            return f"~{elapsed:.1f}s"
        return "..."

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            "duration": self.duration,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "llm_calls": len(self.llm_calls),
            "tool_calls": len(self.tool_calls),
            "retry_count": self.retry_count,
            "status_changes": len(self.status_history),
            "efficiency_score": self._calculate_efficiency_score()
        }

    def _calculate_efficiency_score(self) -> float:
        """Calculate efficiency score based on various metrics"""
        if not self.duration:
            return 0.0

        # Base score
        score = 1.0

        # Penalize long durations (relative to complexity)
        complexity = len(self.llm_calls) + len(self.tool_calls)
        if complexity > 0:
            expected_duration = complexity * 2  # 2 seconds per operation
            if self.duration > expected_duration:
                score *= 0.8

        # Penalize retries
        if self.retry_count > 0:
            score *= (0.9 ** self.retry_count)

        # Bonus for successful completion
        if self.status == NodeStatus.COMPLETED:
            score *= 1.1

        return max(0.0, min(1.0, score))


class ExecutionTreeBuilder:
    """Enhanced execution tree builder with better error handling and metrics"""

    def __init__(self):
        self.nodes: Dict[str, ExecutionNode] = {}
        self.execution_flow: List[str] = []
        self.current_node: Optional[str] = None
        self.root_node: Optional[str] = None
        self.routing_history: List[Dict[str, str]] = []

        # Enhanced tracking
        self.error_log: List[Dict[str, Any]] = []
        self.completion_order: List[str] = []
        self.active_nodes: Set[str] = set()

        # Stats
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.total_cost: float = 0.0
        self.total_tokens: int = 0
        self.total_events: int = 0
        self.session_id: Optional[str] = None

    def add_event(self, event: ProgressEvent):
        """Enhanced event processing with better error handling"""
        try:
            self.total_events += 1

            if not self.start_time:
                self.start_time = event.timestamp
                self.session_id = event.session_id

            # Create or update node
            node_name = event.node_name
            if node_name not in self.nodes:
                self.nodes[node_name] = ExecutionNode(node_name, event.event_type)
                if not self.root_node:
                    self.root_node = node_name

            node = self.nodes[node_name]
            previous_status = node.status

            # Add event to node
            node.add_event(event)

            # Track status changes
            if previous_status != node.status:
                if node.status in [NodeStatus.RUNNING, NodeStatus.STARTING]:
                    self.active_nodes.add(node_name)
                elif node.is_completed():
                    self.active_nodes.discard(node_name)
                    if node_name not in self.completion_order:
                        self.completion_order.append(node_name)

            # Update current node tracking
            if event.event_type in ["node_enter", "llm_call", "tool_call"]:
                if self.current_node != node_name:
                    self.current_node = node_name
                    if node_name not in self.execution_flow:
                        self.execution_flow.append(node_name)

            # Track routing decisions
            if event.routing_from and event.routing_to:
                self.routing_history.append({
                    "from": event.routing_from,
                    "to": event.routing_to,
                    "timestamp": event.timestamp,
                    "decision": event.routing_decision or "unknown"
                })

            # Track errors
            error_message = None
            error_source = None

            # Check multiple places for error information
            if event.event_type == "error":
                # Direct error event
                error_message = event.metadata.get("error") or event.metadata.get("error_message")
                error_source = event.metadata.get("source", "unknown")
            elif event.event_type == "task_error":
                # Task-specific error
                error_message = event.metadata.get("error") or event.metadata.get("error_message")
                error_source = "task_execution"
            elif event.success is False:
                # Failed operation
                error_message = (event.metadata.get("error") or
                                 event.metadata.get("error_message") or
                                 getattr(event, 'tool_error', None) or
                                 "Operation failed")
                error_source = event.event_type
            elif event.metadata and (event.metadata.get("error") or event.metadata.get("error_message")):
                # Error in metadata
                error_message = event.metadata.get("error") or event.metadata.get("error_message")
                error_source = "metadata"

            # Add to error log if we found an error
            if error_message and error_message != "Unknown error":
                self.error_log.append({
                    "timestamp": event.timestamp,
                    "node": event.node_name or "Unknown",
                    "error": error_message,
                    "error_type": event.metadata.get("error_type", "Unknown") if event.metadata else "Unknown",
                    "source": error_source or "unknown",
                    "task_id": getattr(event, 'task_id', None),
                    "tool_name": getattr(event, 'tool_name', None)
                })

                # Limit error log size
                if len(self.error_log) > 150:
                    self.error_log = self.error_log[-100:]

            # Update global stats
            if event.llm_cost:
                self.total_cost += event.llm_cost
            if event.llm_total_tokens:
                self.total_tokens += event.llm_total_tokens

        except Exception as e:
            # Fallback error handling
            self.error_log.append({
                "timestamp": time.time(),
                "node": "SYSTEM",
                "event_type": "processing_error",
                "error": f"Failed to process event: {str(e)}",
                "error_type": "EventProcessingError",
                "original_event": event.event_id if hasattr(event, 'event_id') else "unknown"
            })

    def get_execution_summary(self) -> Dict[str, Any]:
        """Enhanced execution summary with detailed metrics"""
        current_time = time.time()

        return {
            "session_info": {
                "session_id": self.session_id,
                "total_nodes": len(self.nodes),
                "completed_nodes": len(self.completion_order),
                "active_nodes": len(self.active_nodes),
                "failed_nodes": len([n for n in self.nodes.values() if n.status == NodeStatus.FAILED])
            },
            "execution_flow": {
                "flow": self.execution_flow,
                "completion_order": self.completion_order,
                "current_node": self.current_node,
                "active_nodes": list(self.active_nodes)
            },
            "performance_metrics": {
                "total_cost": self.total_cost,
                "total_tokens": self.total_tokens,
                "total_events": self.total_events,
                "error_count": len(self.error_log),
                "routing_steps": len(self.routing_history)
            },
            "timing": {
                "start_time": self.start_time,
                "current_time": current_time,
                "elapsed": current_time - self.start_time if self.start_time else 0,
                "estimated_completion": self._estimate_completion_time()
            },
            "health_indicators": {
                "overall_health": self._calculate_health_score(),
                "error_rate": len(self.error_log) / max(self.total_events, 1),
                "completion_rate": len(self.completion_order) / max(len(self.nodes), 1),
                "average_node_efficiency": self._calculate_average_efficiency()
            }
        }

    def _estimate_completion_time(self) -> Optional[float]:
        """Estimate when execution might complete"""
        if not self.active_nodes or not self.start_time:
            return None

        # Simple heuristic based on current progress
        completed_ratio = len(self.completion_order) / max(len(self.nodes), 1)
        if completed_ratio > 0:
            elapsed = time.time() - self.start_time
            estimated_total = elapsed / completed_ratio
            return self.start_time + estimated_total

        return None

    def _calculate_health_score(self) -> float:
        """Calculate overall execution health score"""
        if not self.nodes:
            return 1.0

        scores = []
        for node in self.nodes.values():
            if node.status == NodeStatus.COMPLETED:
                scores.append(1.0)
            elif node.status == NodeStatus.FAILED:
                scores.append(0.0)
            elif node.status in [NodeStatus.RUNNING, NodeStatus.STARTING]:
                scores.append(0.7)  # In progress
            else:
                scores.append(0.5)  # Pending/waiting

        return sum(scores) / len(scores)

    def _calculate_average_efficiency(self) -> float:
        """Calculate average node efficiency"""
        efficiencies = [node._calculate_efficiency_score() for node in self.nodes.values()
                        if node.duration is not None]
        return sum(efficiencies) / max(len(efficiencies), 1)

def human_readable_time(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m"
    days, hours = divmod(hours, 24)
    if days < 7:
        return f"{days}d {hours}h"
    weeks, days = divmod(days, 7)
    return f"{weeks}w {days}d"

class ProgressiveTreePrinter:
    """Production-ready progressive tree printer with enhanced features"""

    def __init__(self, mode: VerbosityMode = VerbosityMode.STANDARD, use_rich: bool = True,
                 auto_refresh: bool = True, max_history: int = 1000,
                 realtime_minimal: bool = None):
        self.mode = mode
        self.agent_name = "self"
        self.use_rich = use_rich and RICH_AVAILABLE
        self.auto_refresh = auto_refresh
        self.max_history = max_history

        self.tree_builder = ExecutionTreeBuilder()
        self.print_history: List[Dict[str, Any]] = []

        # Optimized realtime option
        self.realtime_minimal = realtime_minimal if realtime_minimal is not None else False
        self._last_summary = ""
        self._needs_full_tree = False
        self._spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._spinner_index = 0

        # External accumulation storage
        self._accumulated_runs: List[Dict[str, Any]] = []
        self._current_run_id = 0
        self._global_start_time = time.time()

        # Rich console setup
        if self.use_rich:
            self.console = Console(record=True)
            if mode == VerbosityMode.REALTIME:
                self.progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                    transient=True
                )
                self.progress_task = None

        # State tracking
        self._last_print_hash = None
        self._print_counter = 0
        self._last_update_time = 0
        self._consecutive_errors = 0

        # Error handling
        self._error_threshold = 5
        self._fallback_mode = False

    def reset_global_start_time(self):
        """Reset global start time for new session"""
        self._global_start_time = time.time()

    def print_strategy_selection(self, strategy: str, event: ProgressEvent = None, context: Dict[str, Any] = None):
        """Print strategy selection information with descriptions based on verbosity mode"""

        # Strategy descriptions mapping
        strategy_descriptions = {
            "direct_response": "Simple LLM flow with optional tool calls",
            "fast_simple_planning": "Simple multi-step plan with tool orchestration",
            "slow_complex_planning": "Complex task breakdown with tool orchestration, use for tasks with more than 2 'and' words",
            "research_and_analyze": "Information gathering with variable integration",
            "creative_generation": "Content creation with personalization",
            "problem_solving": "Analysis with tool validation"
        }

        strategy_icons = {
            "direct_response": "💬",
            "fast_simple_planning": "⚡",
            "slow_complex_planning": "🔄",
            "research_and_analyze": "🔍",
            "creative_generation": "🎨",
            "problem_solving": "🧩"
        }

        try:
            if self._fallback_mode or not self.use_rich:
                self._print_strategy_fallback(strategy, strategy_descriptions, strategy_icons)
                return

            # Get strategy info
            icon = strategy_icons.get(strategy, "🎯")+" "+self.agent_name
            description = strategy_descriptions.get(strategy, "Unknown strategy")

            # Format based on verbosity mode
            if self.mode == VerbosityMode.MINIMAL:
                # Just show strategy name
                strategy_text = f"{icon} Strategy: {strategy}"
                self.console.print(strategy_text, style="cyan")

            elif self.mode == VerbosityMode.STANDARD:
                # Show strategy with description
                strategy_text = f"{icon} Strategy selected: [bold]{strategy}[/bold]\n📝 {description}"
                strategy_panel = Panel(
                    strategy_text,
                    title="🎯 Execution Strategy",
                    style="cyan",
                    box=box.ROUNDED
                )
                self.console.print(strategy_panel)

            elif self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                # Full details with context
                strategy_content = [
                    f"{icon} Strategy: [bold cyan]{strategy}[/bold cyan]",
                    f"📝 Description: {description}"
                ]

                # Add context information if available
                if context:
                    if context.get("reasoning"):
                        strategy_content.append(f"🧠 Reasoning: {context['reasoning']}")
                    if context.get("complexity_score"):
                        strategy_content.append(f"📊 Complexity: {context['complexity_score']}")
                    if context.get("estimated_steps"):
                        strategy_content.append(f"📋 Est. Steps: {context['estimated_steps']}")

                # Add event context in debug mode
                if self.mode == VerbosityMode.DEBUG and event:
                    strategy_content.append(
                        f"⏱️ Selected at: {datetime.fromtimestamp(event.timestamp).strftime('%H:%M:%S')}")
                    if event.node_name:
                        strategy_content.append(f"📍 Node: {event.node_name}")

                strategy_panel = Panel(
                    "\n".join(strategy_content),
                    title="🎯 Strategy Selection Details",
                    style="cyan bold",
                    box=box.ROUNDED
                )
                self.console.print()
                self.console.print(strategy_panel)

            elif self.mode == VerbosityMode.REALTIME:
                # Minimal output for realtime mode
                if not self.realtime_minimal:
                    strategy_text = f"\n{icon} Strategy: {strategy} - {description}"
                    self.console.print(strategy_text, style="cyan dim")

        except Exception as e:
            # Fallback on error
            self._consecutive_errors += 1
            if self._consecutive_errors <= self._error_threshold:
                print(f"⚠️ Strategy print error: {e}")
            self._print_strategy_fallback(strategy, strategy_descriptions, strategy_icons)

    def _print_strategy_fallback(self, strategy: str, descriptions: Dict[str, str], icons: Dict[str, str]):
        """Fallback strategy printing without Rich"""
        try:
            icon = icons.get(strategy, "🎯")
            description = descriptions.get(strategy, "Unknown strategy")

            if self.mode == VerbosityMode.MINIMAL:
                print(f"{icon} Strategy: {strategy}")

            elif self.mode == VerbosityMode.STANDARD:
                print(f"\n{'-' * 50}")
                print(f"{icon} Strategy selected: {strategy}")
                print(f"📝 {description}")
                print(f"{'-' * 50}")

            elif self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                print(f"\n{'=' * 60}")
                print(f"🎯 STRATEGY SELECTION")
                print(f"{'=' * 60}")
                print(f"{icon} Strategy: {strategy}")
                print(f"📝 Description: {description}")
                print(f"{'=' * 60}")

            elif self.mode == VerbosityMode.REALTIME and not self.realtime_minimal:
                print(f"{icon} Strategy: {strategy} - {description}")

        except Exception as e:
            # Ultimate fallback
            print(f"Strategy selected: {strategy}")

    def print_strategy_from_event(self, event: ProgressEvent):
        """Convenience method to print strategy from event metadata"""
        try:
            if not event.metadata or 'strategy' not in event.metadata:
                return

            strategy = event.metadata['strategy']
            context = {
                'reasoning': event.metadata.get('reasoning'),
                'complexity_score': event.metadata.get('complexity_score'),
                'estimated_steps': event.metadata.get('estimated_steps')
            }

            self.print_strategy_selection(strategy, event, context)

        except Exception as e:
            if self.mode == VerbosityMode.DEBUG:
                print(f"⚠️ Error printing strategy from event: {e}")

    def print_plan_from_event(self, event: ProgressEvent):
        """Convenience method to print plan from event metadata"""
        try:
            if not event.metadata or 'full_plan' not in event.metadata:
                return

            plan = event.metadata['full_plan']
            self.pretty_print_task_plan(plan)

        except Exception as e:
            if self.mode == VerbosityMode.DEBUG:
                print(f"⚠️ Error printing plan from event: {e}")

    def _should_print_update(self) -> bool:
        """Enhanced decision logic for when to print updates"""
        current_time = time.time()

        # Force full tree on errors or completion
        if self._needs_full_tree:
            self._last_update_time = current_time
            return True

        # In minimal realtime mode, only show one-line updates frequently
        if self.realtime_minimal and self.mode == VerbosityMode.REALTIME:
            # Update one-line summary more frequently (every 0.5s)
            return current_time - self._last_update_time > 0.5

        # Rate limiting for other modes - don't print too frequently
        if current_time - self._last_update_time < 1.5:
            return False

        try:
            # Create state hash for change detection
            summary = self.tree_builder.get_execution_summary()
            current_state = {
                "total_nodes": summary["session_info"]["total_nodes"],
                "completed_nodes": summary["session_info"]["completed_nodes"],
                "active_nodes": summary["session_info"]["active_nodes"],
                "failed_nodes": summary["session_info"]["failed_nodes"],
                "current_node": summary["execution_flow"]["current_node"],
                "total_events": summary["performance_metrics"]["total_events"],
                "error_count": summary["performance_metrics"]["error_count"]
            }

            current_hash = hash(str(sorted(current_state.items())))

            # Mode-specific update logic
            if self.mode == VerbosityMode.MINIMAL:
                should_update = (current_hash != self._last_print_hash and
                                 (current_state["completed_nodes"] !=
                                  getattr(self, '_last_completed_count', 0) or
                                  current_state["failed_nodes"] !=
                                  getattr(self, '_last_failed_count', 0)))

                self._last_completed_count = current_state["completed_nodes"]
                self._last_failed_count = current_state["failed_nodes"]

            elif self.mode in [VerbosityMode.STANDARD, VerbosityMode.VERBOSE]:
                should_update = current_hash != self._last_print_hash

            else:  # DEBUG mode
                should_update = True

            if should_update:
                self._last_print_hash = current_hash
                self._last_update_time = current_time
                return True

            return False

        except Exception as e:
            self._consecutive_errors += 1
            if self._consecutive_errors > self._error_threshold:
                self._fallback_mode = True
                print(f"⚠️  Printer error threshold exceeded, switching to fallback mode: {e}")
            return False

    def flush(self, run_name: str = None) -> Dict[str, Any]:
        """
        Flush current execution data and store externally for accumulation.
        Resets internal state for fresh execution timing.

        Args:
            run_name: Optional name for this run

        Returns:
            Dict containing the flushed execution data
        """
        try:
            # Generate run info
            current_time = time.time()
            if run_name is None:
                run_name = f"run_{self._current_run_id + 1}"

            # Collect current execution data
            summary = self.tree_builder.get_execution_summary()

            # Create comprehensive run data
            run_data = {
                "run_id": self._current_run_id + 1,
                "run_name": run_name,
                "flush_timestamp": current_time,
                "execution_summary": summary,
                "detailed_nodes": {},
                "execution_history": self.print_history.copy(),
                "error_log": self.tree_builder.error_log.copy(),
                "routing_history": self.tree_builder.routing_history.copy(),
                "print_counter": self._print_counter,
                "consecutive_errors": self._consecutive_errors,
                "fallback_mode": self._fallback_mode
            }

            # Add detailed node information
            for node_name, node in self.tree_builder.nodes.items():
                run_data["detailed_nodes"][node_name] = {
                    "status": node.status.value,
                    "duration": node.duration,
                    "start_time": node.start_time,
                    "end_time": node.end_time,
                    "total_cost": node.total_cost,
                    "total_tokens": node.total_tokens,
                    "llm_calls": len(node.llm_calls),
                    "tool_calls": len(node.tool_calls),
                    "error": node.error,
                    "retry_count": node.retry_count,
                    "performance_metrics": node.get_performance_summary(),
                    "strategy": node.strategy,
                    "reasoning": node.reasoning,
                    "routing_from": node.routing_from,
                    "routing_to": node.routing_to
                }

            # Store in accumulated runs
            self._accumulated_runs.append(run_data)

            # Reset internal state for fresh execution
            self._reset_for_fresh_execution()

            if self.use_rich:
                self.console.print(f"✅ Run '{run_name}' flushed and stored", style="green bold")
                self.console.print(f"📊 Total accumulated runs: {len(self._accumulated_runs)}", style="blue")
            else:
                print(f"✅ Run '{run_name}' flushed and stored")
                print(f"📊 Total accumulated runs: {len(self._accumulated_runs)}")

            return run_data

        except Exception as e:
            error_msg = f"❌ Error during flush: {e}"
            if self.use_rich:
                self.console.print(error_msg, style="red bold")
            else:
                print(error_msg)

            # Still try to reset for fresh execution
            self._reset_for_fresh_execution()

            return {"error": str(e), "timestamp": current_time}

    def pretty_print_task_plan(self, task_plan: Any):
        """Pretty print a Any with full details and structure"""
        try:
            if self._fallback_mode or not self.use_rich:
                self._print_task_plan_fallback(task_plan)
                return

            # Create main header
            self.console.print()
            header_text = f"📋 Task Plan: {task_plan.name}\n"
            header_text += f"Status: {task_plan.status.upper()} | Strategy: {task_plan.execution_strategy}\n"
            header_text += f"Created: {task_plan.created_at.strftime('%Y-%m-%d %H:%M:%S')} | Tasks: {len(task_plan.tasks)}"

            header = Panel(
                header_text,
                title="🚀 Task Plan Overview",
                style="cyan bold",
                box=box.ROUNDED
            )
            self.console.print(header)

            # Description panel
            if task_plan.description:
                desc_panel = Panel(
                    task_plan.description,
                    title="📝 Description",
                    style="blue",
                    box=box.ROUNDED
                )
                self.console.print(desc_panel)

            # Create task tree
            tree = Tree(f"🔗 Task Execution Flow ({len(task_plan.tasks)} tasks)", style="bold green")

            # Group tasks by type for better organization
            task_groups = {}
            for task in task_plan.tasks:
                task_type = task.type if hasattr(task, 'type') else type(task).__name__
                if task_type not in task_groups:
                    task_groups[task_type] = []
                task_groups[task_type].append(task)

            # Add tasks organized by dependencies and priority
            sorted_tasks = sorted(task_plan.tasks, key=lambda t: (t.priority, t.id))

            for i, task in enumerate(sorted_tasks):
                # Task status icon
                status_icon = self._get_task_status_icon(task)
                task_type = task.type if hasattr(task, 'type') else type(task).__name__

                # Main task info
                task_text = f"{status_icon} [{i + 1}] {task.id}"
                if task.priority != 1:
                    task_text += f" (Priority: {task.priority})"

                task_style = self._get_task_status_color(task)
                task_branch = tree.add(task_text, style=task_style)

                # Add task details based on verbosity mode
                if self.mode == VerbosityMode.MINIMAL:
                    # Only show basic info
                    task_branch.add(f"📄 {task.description[:80]}...", style="dim")
                else:
                    # Show full details
                    self._add_task_details(task_branch, task)

            self.console.print(tree)

            # Add metadata if available
            if task_plan.metadata and self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                self._print_task_plan_metadata(task_plan)

            # Add dependency analysis
            if self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                self._print_dependency_analysis(task_plan)

        except Exception as e:
            self.console.print(f"❌ Error printing task plan: {e}", style="red bold")
            self._print_task_plan_fallback(task_plan)

    def _get_task_status_icon(self, task: Any) -> str:
        """Get appropriate status icon for task"""
        status_icons = {
            "pending": "⏳",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
            "paused": "⏸️"
        }
        return status_icons.get(task.status, "❓")

    def _get_task_status_color(self, task: Any) -> str:
        """Get appropriate color styling for task status"""
        status_colors = {
            "pending": "yellow",
            "running": "white bold dim",
            "completed": "green bold",
            "failed": "red bold",
            "paused": "orange3"
        }
        return status_colors.get(task.status, "white")

    def _add_task_details(self, parent_branch: Tree, task: Any):
        """Add detailed task information based on task type"""
        # Description
        parent_branch.add(f"📄 {task.description}", style="white dim")

        # Dependencies
        if task.dependencies:
            deps_text = f"🔗 Dependencies: {', '.join(task.dependencies)}"
            parent_branch.add(deps_text, style="yellow dim")

        # Task type specific details

        self._add_llm_task_details(parent_branch, task)
        self._add_tool_task_details(parent_branch, task)
        self._add_decision_task_details(parent_branch, task)
        self._add_compound_task_details(parent_branch, task)

        # Timing info
        if hasattr(task, 'created_at') and task.created_at:
            timing_info = f"📅 Created: {task.created_at.strftime('%H:%M:%S')}"
            if hasattr(task, 'started_at') and task.started_at:
                timing_info += f" | Started: {task.started_at.strftime('%H:%M:%S')}"
            if hasattr(task, 'completed_at') and task.completed_at:
                timing_info += f" | Completed: {task.completed_at.strftime('%H:%M:%S')}"
            parent_branch.add(timing_info, style="cyan dim")

        # Error info
        if hasattr(task, 'error') and task.error:
            error_text = f"❌ Error: {task.error}"
            if hasattr(task, 'retry_count') and task.retry_count > 0:
                error_text += f" (Retries: {task.retry_count}/{task.max_retries})"
            parent_branch.add(error_text, style="red dim")

        # Critical flag
        if hasattr(task, 'critical') and task.critical:
            parent_branch.add("🚨 CRITICAL TASK", style="red bold")

    def _add_llm_task_details(self, parent_branch: Tree, task: Any):
        """Add LLM-specific task details"""
        if hasattr(task, 'llm_config') and task.llm_config:
            config_text = f"🧠 Model: {task.llm_config.get('model_preference', 'default')}"
            config_text += f" | Temp: {task.llm_config.get('temperature', 0.7)}"
            parent_branch.add(config_text, style="purple dim")

        if hasattr(task, 'context_keys') and task.context_keys:
            context_text = f"🔑 Context: {', '.join(task.context_keys)}"
            parent_branch.add(context_text, style="blue dim")

        if hasattr(task, 'prompt_template') and task.prompt_template and self.mode == VerbosityMode.DEBUG:
            prompt_preview = task.prompt_template[:100] + "..." if len(
                task.prompt_template) > 100 else task.prompt_template
            parent_branch.add(f"💬 Prompt: {prompt_preview}", style="green dim")

    def _add_tool_task_details(self, parent_branch: Tree, task: Any):
        """Add Tool-specific task details"""
        if hasattr(task, 'tool_name') and task.tool_name:
            parent_branch.add(f"🔧 Tool: {task.tool_name}", style="green dim")

        if hasattr(task, 'arguments') and task.arguments and self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
            args_text = f"⚙️ Args: {str(task.arguments)[:80]}..."
            parent_branch.add(args_text, style="yellow dim")

        if hasattr(task, 'hypothesis') and task.hypothesis:
            parent_branch.add(f"🔬 Hypothesis: {task.hypothesis}", style="blue dim")

        if hasattr(task, 'expectation') and task.expectation:
            parent_branch.add(f"🎯 Expected: {task.expectation}", style="cyan dim")

    def _add_decision_task_details(self, parent_branch: Tree, task: Any):
        """Add Decision-specific task details"""
        if hasattr(task, 'decision_model') and task.decision_model:
            parent_branch.add(f"🧠 Decision Model: {task.decision_model}", style="purple dim")

        if hasattr(task, 'routing_map') and task.routing_map and self.mode == VerbosityMode.DEBUG:
            routes_text = f"🗺️ Routes: {list(task.routing_map.keys())}"
            parent_branch.add(routes_text, style="orange dim")

    def _add_compound_task_details(self, parent_branch: Tree, task: Any):
        """Add Compound-specific task details"""
        if hasattr(task, 'sub_task_ids') and task.sub_task_ids:
            subtasks_text = f"📋 Subtasks: {', '.join(task.sub_task_ids)}"
            parent_branch.add(subtasks_text, style="magenta dim")

        if hasattr(task, 'execution_strategy') and task.execution_strategy:
            parent_branch.add(f"⚡ Strategy: {task.execution_strategy}", style="blue dim")

    def _print_task_plan_metadata(self, task_plan: Any):
        """Print task plan metadata in verbose modes"""
        if not task_plan.metadata:
            return

        metadata_table = Table(title="📊 Task Plan Metadata", box=box.ROUNDED)
        metadata_table.add_column("Key", style="cyan", min_width=15)
        metadata_table.add_column("Value", style="green", min_width=20)

        for key, value in task_plan.metadata.items():
            metadata_table.add_row(key, str(value))

        self.console.print()
        self.console.print(metadata_table)

    def _print_dependency_analysis(self, task_plan: Any):
        """Print dependency analysis"""
        try:
            # Build dependency graph
            dependency_info = self._analyze_dependencies(task_plan)

            if dependency_info["cycles"] or dependency_info["orphans"] or dependency_info["leaves"]:
                analysis_text = []

                if dependency_info["cycles"]:
                    analysis_text.append(f"🔄 Circular dependencies detected: {dependency_info['cycles']}")

                if dependency_info["orphans"]:
                    analysis_text.append(f"🏝️ Tasks without dependencies: {dependency_info['orphans']}")

                if dependency_info["leaves"]:
                    analysis_text.append(f"🍃 Final tasks: {dependency_info['leaves']}")

                analysis_text.append(f"📊 Max depth: {dependency_info['max_depth']} levels")

                analysis_panel = Panel(
                    "\n".join(analysis_text),
                    title="🔍 Dependency Analysis",
                    style="yellow"
                )
                self.console.print()
                self.console.print(analysis_panel)

        except Exception as e:
            if self.mode == VerbosityMode.DEBUG:
                self.console.print(f"⚠️ Dependency analysis error: {e}", style="red dim")

    def _analyze_dependencies(self, task_plan: Any) -> Dict[str, Any]:
        """Analyze task dependencies for insights"""
        task_map = {task.id: task for task in task_plan.tasks}

        cycles = []
        orphans = []
        leaves = []
        max_depth = 0

        # Find orphans (no dependencies)
        for task in task_plan.tasks:
            if not task.dependencies:
                orphans.append(task.id)

        # Find leaves (no one depends on them)
        all_deps = set()
        for task in task_plan.tasks:
            all_deps.update(task.dependencies)

        for task in task_plan.tasks:
            if task.id not in all_deps:
                leaves.append(task.id)

        # Calculate max depth (simplified)
        def get_depth(task_id, visited=None):
            if visited is None:
                visited = set()
            if task_id in visited:
                return 0  # Cycle detected
            if task_id not in task_map:
                return 0

            visited.add(task_id)
            task = task_map[task_id]
            if not task.dependencies:
                return 1

            return 1 + max((get_depth(dep, visited.copy()) for dep in task.dependencies), default=0)

        for task in task_plan.tasks:
            depth = get_depth(task.id)
            max_depth = max(max_depth, depth)

        return {
            "cycles": cycles,
            "orphans": orphans,
            "leaves": leaves,
            "max_depth": max_depth
        }

    def _print_task_plan_fallback(self, task_plan: Any):
        """Fallback task plan printing without Rich"""
        print(f"\n{'=' * 80}")
        print(f"📋 TASK PLAN: {task_plan.name}")
        print(f"{'=' * 80}")
        print(f"Description: {task_plan.description}")
        print(f"Status: {task_plan.status} | Strategy: {task_plan.execution_strategy}")
        print(f"Created: {task_plan.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Tasks: {len(task_plan.tasks)}")
        print(f"{'=' * 80}")

        print("\n📋 TASKS:")
        print(f"{'-' * 40}")

        sorted_tasks = sorted(task_plan.tasks, key=lambda t: (t.priority, t.id))
        for i, task in enumerate(sorted_tasks):
            status_icon = self._get_task_status_icon(task)
            task_type = task.type if hasattr(task, 'type') else type(task).__name__

            print(f"{status_icon} [{i + 1}] {task.id} ({task_type})")
            print(f"    📄 {task.description}")

            if task.dependencies:
                print(f"    🔗 Dependencies: {', '.join(task.dependencies)}")

            if hasattr(task, 'error') and task.error:
                print(f"    ❌ Error: {task.error}")

            if i < len(sorted_tasks) - 1:
                print()

        print(f"{'=' * 80}")

    def _reset_for_fresh_execution(self):
        """Reset internal state for a completely fresh execution"""
        try:
            # Increment run counter
            self._current_run_id += 1

            # Reset tree builder with completely fresh state
            self.tree_builder = ExecutionTreeBuilder()

            # Reset print history
            self.print_history = []

            # Reset timing and state tracking
            self._last_print_hash = None
            self._print_counter = 0
            self._last_update_time = 0

            # Reset realtime state
            self._last_summary = ""
            self._needs_full_tree = False
            self._spinner_index = 0

            # Reset error handling but don't reset fallback mode completely
            # (if we're in fallback mode due to Rich issues, stay there)
            self._consecutive_errors = 0

            # Reset Rich progress if exists
            if hasattr(self, 'progress') and self.progress:
                self.progress_task = None

            # Clear any cached state
            if hasattr(self, '_last_completed_count'):
                delattr(self, '_last_completed_count')
            if hasattr(self, '_last_failed_count'):
                delattr(self, '_last_failed_count')

        except Exception as e:
            print(f"⚠️ Error during reset: {e}")

    def get_accumulated_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all accumulated runs"""
        try:
            if not self._accumulated_runs:
                return {
                    "total_runs": 0,
                    "message": "No runs have been flushed yet"
                }

            # Calculate aggregate metrics
            total_cost = 0.0
            total_tokens = 0
            total_events = 0
            total_errors = 0
            total_nodes = 0
            total_duration = 0.0

            run_summaries = []

            for run in self._accumulated_runs:
                summary = run["execution_summary"]
                perf = summary["performance_metrics"]
                timing = summary["timing"]
                session_info = summary["session_info"]

                total_cost += perf["total_cost"]
                total_tokens += perf["total_tokens"]
                total_events += perf["total_events"]
                total_errors += perf["error_count"]
                total_nodes += session_info["total_nodes"]
                total_duration += timing["elapsed"]

                run_summaries.append({
                    "run_id": run["run_id"],
                    "run_name": run["run_name"],
                    "nodes": session_info["total_nodes"],
                    "completed": session_info["completed_nodes"],
                    "failed": session_info["failed_nodes"],
                    "duration": timing["elapsed"],
                    "cost": perf["total_cost"],
                    "tokens": perf["total_tokens"],
                    "errors": perf["error_count"],
                    "health_score": summary["health_indicators"]["overall_health"]
                })

            # Calculate averages
            num_runs = len(self._accumulated_runs)
            avg_duration = total_duration / num_runs
            avg_cost = total_cost / num_runs
            avg_tokens = total_tokens / num_runs
            avg_nodes = total_nodes / num_runs

            return {
                "total_runs": num_runs,
                "current_run_id": self._current_run_id,
                "global_start_time": self._global_start_time,
                "total_accumulated_time": time.time() - self._global_start_time,

                "aggregate_metrics": {
                    "total_cost": total_cost,
                    "total_tokens": total_tokens,
                    "total_events": total_events,
                    "total_errors": total_errors,
                    "total_nodes": total_nodes,
                    "total_duration": total_duration,
                },

                "average_metrics": {
                    "avg_duration": avg_duration,
                    "avg_cost": avg_cost,
                    "avg_tokens": avg_tokens,
                    "avg_nodes": avg_nodes,
                    "avg_error_rate": total_errors / max(total_events, 1),
                    "avg_health_score": sum(r["health_score"] for r in run_summaries) / num_runs
                },

                "run_summaries": run_summaries,

                "performance_insights": self._generate_accumulated_insights(run_summaries)
            }

        except Exception as e:
            return {"error": f"Error generating accumulated summary: {e}"}

    def _generate_accumulated_insights(self, run_summaries: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from accumulated run data"""
        insights = []

        if not run_summaries:
            return insights

        try:
            num_runs = len(run_summaries)

            # Performance trends
            if num_runs > 1:
                recent_runs = run_summaries[-3:]  # Last 3 runs
                older_runs = run_summaries[:-3] if len(run_summaries) > 3 else []

                if older_runs:
                    recent_avg_duration = sum(r["duration"] for r in recent_runs) / len(recent_runs)
                    older_avg_duration = sum(r["duration"] for r in older_runs) / len(older_runs)

                    if recent_avg_duration < older_avg_duration * 0.8:
                        insights.append("🚀 Performance improving: Recent runs 20% faster")
                    elif recent_avg_duration > older_avg_duration * 1.2:
                        insights.append("⚠️ Performance degrading: Recent runs 20% slower")

            # Error patterns
            error_rates = [r["errors"] / max(r["nodes"], 1) for r in run_summaries]
            avg_error_rate = sum(error_rates) / len(error_rates)

            if avg_error_rate == 0:
                insights.append("✨ Perfect reliability: Zero errors across all runs")
            elif avg_error_rate < 0.1:
                insights.append(f"✅ High reliability: {avg_error_rate:.1%} average error rate")
            elif avg_error_rate > 0.3:
                insights.append(f"🔧 Reliability concerns: {avg_error_rate:.1%} average error rate")

            # Cost efficiency
            costs = [r["cost"] for r in run_summaries if r["cost"] > 0]
            if costs:
                avg_cost = sum(costs) / len(costs)
                if avg_cost < 0.01:
                    insights.append(f"💚 Very cost efficient: ${avg_cost:.4f} average per run")
                elif avg_cost > 0.1:
                    insights.append(f"💸 High cost per run: ${avg_cost:.4f} average")

            # Consistency
            durations = [r["duration"] for r in run_summaries]
            if len(durations) > 1:
                import statistics
                duration_std = statistics.stdev(durations)
                duration_mean = statistics.mean(durations)
                cv = duration_std / duration_mean if duration_mean > 0 else 0

                if cv < 0.2:
                    insights.append("🎯 Highly consistent execution times")
                elif cv > 0.5:
                    insights.append("📊 Variable execution times - investigate bottlenecks")

            # Success patterns
            completion_rates = [r["completed"] / max(r["nodes"], 1) for r in run_summaries]
            avg_completion = sum(completion_rates) / len(completion_rates)

            if avg_completion > 0.95:
                insights.append(f"🎉 Excellent completion rate: {avg_completion:.1%}")
            elif avg_completion < 0.8:
                insights.append(f"⚠️ Low completion rate: {avg_completion:.1%}")

        except Exception as e:
            insights.append(f"⚠️ Error generating insights: {e}")

        return insights

    def print_accumulated_summary(self):
        """Print comprehensive summary of all accumulated runs"""
        try:
            summary = self.get_accumulated_summary()

            if summary.get("total_runs", 0) == 0:
                if self.use_rich:
                    self.console.print("📊 No accumulated runs to display", style="yellow")
                else:
                    print("📊 No accumulated runs to display")
                return

            if not self.use_rich:
                self._print_accumulated_summary_fallback(summary)
                return

            # Rich formatted output
            self.console.print()
            self.console.print("🗂️ [bold cyan]ACCUMULATED EXECUTION SUMMARY[/bold cyan] 🗂️")

            # Overview table
            overview_table = Table(title="📊 Aggregate Overview", box=box.ROUNDED)
            overview_table.add_column("Metric", style="cyan", min_width=20)
            overview_table.add_column("Value", style="green", min_width=15)
            overview_table.add_column("Average", style="blue", min_width=15)

            agg = summary["aggregate_metrics"]
            avg = summary["average_metrics"]

            overview_table.add_row("Total Runs", str(summary["total_runs"]), "")
            overview_table.add_row("Total Duration", f"{agg['total_duration']:.1f}s", f"{avg['avg_duration']:.1f}s")
            overview_table.add_row("Total Nodes", str(agg["total_nodes"]), f"{avg['avg_nodes']:.1f}")
            overview_table.add_row("Total Events", str(agg["total_events"]), "")

            if agg["total_cost"] > 0:
                overview_table.add_row("Total Cost", self._format_cost(agg["total_cost"]),
                                       self._format_cost(avg["avg_cost"]))

            if agg["total_tokens"] > 0:
                overview_table.add_row("Total Tokens", f"{agg['total_tokens']:,}",
                                       f"{avg['avg_tokens']:,.0f}")

            overview_table.add_row("Error Rate", f"{avg['avg_error_rate']:.1%}", "")
            overview_table.add_row("Health Score", f"{avg['avg_health_score']:.1%}", "")

            self.console.print(overview_table)

            # Individual runs table
            runs_table = Table(title="🏃 Individual Runs", box=box.ROUNDED)
            runs_table.add_column("Run", style="cyan")
            runs_table.add_column("Duration", style="blue")
            runs_table.add_column("Nodes", style="green")
            runs_table.add_column("Success", style="green")
            runs_table.add_column("Cost", style="yellow")
            runs_table.add_column("Health", style="magenta")

            for run in summary["run_summaries"]:
                success_rate = run["completed"] / max(run["nodes"], 1)
                cost_str = self._format_cost(run["cost"]) if run["cost"] > 0 else "-"

                runs_table.add_row(
                    run["run_name"],
                    f"{run['duration']:.1f}s",
                    f"{run['completed']}/{run['nodes']}",
                    f"{success_rate:.1%}",
                    cost_str,
                    f"{run['health_score']:.1%}"
                )

            self.console.print(runs_table)

            # Insights
            if summary.get("performance_insights"):
                insights_panel = Panel(
                    "\n".join(f"• {insight}" for insight in summary["performance_insights"]),
                    title="🔍 Performance Insights",
                    style="yellow"
                )
                self.console.print(insights_panel)

        except Exception as e:
            error_msg = f"❌ Error printing accumulated summary: {e}"
            if self.use_rich:
                self.console.print(error_msg, style="red bold")
            else:
                print(error_msg)

    def export_accumulated_data(self, filepath: str = None, extra_data: Dict[str, Any] = None) -> str:
        """Export all accumulated run data to file"""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"accumulated_execution_data_{timestamp}.json"

            export_data = {
                "export_timestamp": time.time(),
                "export_version": "1.0",
                "printer_config": {
                    "mode": self.mode.value,
                    "use_rich": self.use_rich,
                    "realtime_minimal": self.realtime_minimal
                },
                "accumulated_summary": self.get_accumulated_summary(),
                "all_runs": self._accumulated_runs,

            }

            export_data.update(extra_data or {})

            import json
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            if self.use_rich:
                self.console.print(f"📁 Accumulated data exported to: {filepath}", style="green bold")
                self.console.print(f"📊 Total runs exported: {len(self._accumulated_runs)}", style="blue")
            else:
                print(f"📁 Accumulated data exported to: {filepath}")
                print(f"📊 Total runs exported: {len(self._accumulated_runs)}")

            return filepath

        except Exception as e:
            error_msg = f"❌ Error exporting accumulated data: {e}"
            if self.use_rich:
                self.console.print(error_msg, style="red bold")
            else:
                print(error_msg)
            return ""

    def _print_accumulated_summary_fallback(self, summary: Dict[str, Any]):
        """Fallback accumulated summary without Rich"""
        try:
            print(f"\n{'=' * 80}")
            print("🗂️ ACCUMULATED EXECUTION SUMMARY 🗂️")
            print(f"{'=' * 80}")

            agg = summary["aggregate_metrics"]
            avg = summary["average_metrics"]

            print(f"Total Runs: {summary['total_runs']}")
            print(f"Total Duration: {agg['total_duration']:.1f}s (avg: {avg['avg_duration']:.1f}s)")
            print(f"Total Nodes: {agg['total_nodes']} (avg: {avg['avg_nodes']:.1f})")
            print(f"Total Events: {agg['total_events']}")

            if agg["total_cost"] > 0:
                print(f"Total Cost: {self._format_cost(agg['total_cost'])} (avg: {self._format_cost(avg['avg_cost'])})")

            if agg["total_tokens"] > 0:
                print(f"Total Tokens: {agg['total_tokens']:,} (avg: {avg['avg_tokens']:,.0f})")

            print(f"Average Error Rate: {avg['avg_error_rate']:.1%}")
            print(f"Average Health Score: {avg['avg_health_score']:.1%}")

            print(f"\n{'=' * 80}")
            print("🏃 INDIVIDUAL RUNS:")
            print(f"{'=' * 80}")

            for run in summary["run_summaries"]:
                success_rate = run["completed"] / max(run["nodes"], 1)
                cost_str = self._format_cost(run["cost"]) if run["cost"] > 0 else "N/A"

                print(f"• {run['run_name']}: {run['duration']:.1f}s | "
                      f"{run['completed']}/{run['nodes']} nodes ({success_rate:.1%}) | "
                      f"Cost: {cost_str} | Health: {run['health_score']:.1%}")

            # Insights
            if summary.get("performance_insights"):
                print(f"\n🔍 PERFORMANCE INSIGHTS:")
                print(f"{'-' * 40}")
                for insight in summary["performance_insights"]:
                    print(f"• {insight}")

            print(f"{'=' * 80}")

        except Exception as e:
            print(f"❌ Error printing fallback summary: {e}")

    def _create_one_line_summary(self) -> str:
        """Create a concise one-line summary of current execution state"""
        try:
            summary = self.tree_builder.get_execution_summary()
            current_node = summary["execution_flow"]["current_node"]
            active_nodes = summary["execution_flow"]["active_nodes"]
            timing = summary["timing"]

            # Get spinner
            spinner = f"@{self.agent_name} "

            # Format elapsed time
            elapsed = timing["elapsed"]
            if elapsed < 60:
                time_str = f"{elapsed:.1f}s"
            elif elapsed < 3600:
                minutes = int(elapsed // 60)
                seconds = elapsed % 60
                time_str = f"{minutes}m{seconds:.1f}s"
            else:
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                time_str = f"{hours}h{minutes}m"

            # Get current event info
            if current_node and current_node in self.tree_builder.nodes:
                node = self.tree_builder.nodes[current_node]

                # Get the most relevant info
                info_parts = []
                if node.strategy:
                    info_parts.append(f"strategy: {node.strategy}")
                if node.reasoning:
                    reasoning_short = node.reasoning[:50] + "..." if len(node.reasoning) > 50 else node.reasoning
                    info_parts.append(f"reasoning: {reasoning_short}")

                # Recent activity
                recent_activity = "processing"
                if node.llm_calls and node.llm_calls[-1].timestamp > time.time() - 5:
                    recent_activity = "llm_call"
                elif node.tool_calls and node.tool_calls[-1].timestamp > time.time() - 5:
                    recent_activity = f"tool: {node.tool_calls[-1].tool_name}"

                info_str = " | ".join(info_parts) if info_parts else recent_activity
                if len(info_str) > 80:
                    info_str = info_str[15:92] + "..."

                return f"{spinner} {current_node} → {recent_activity} | {info_str} | {time_str}" if recent_activity != info_str else f"{spinner} {current_node}  → | {info_str} | {time_str}"

            # Fallback summary
            session_info = summary["session_info"]
            progress_text = f"{session_info['completed_nodes']}/{session_info['total_nodes']} nodes"
            return f"{spinner} Processing {progress_text} | {time_str}"

        except Exception as e:
            return f"⚠️ Processing... | {time.time():.1f}s"

    def _print_one_line_summary(self):
        """Print or update the one-line summary"""
        try:
            summary_line = self._create_one_line_summary()

            if summary_line != self._last_summary:
                # Clear the previous line and print new summary
                if self._last_summary:
                    print(f"\r{' ' * len(self._last_summary)}", end="", flush=True)
                print(f"\r{summary_line}", end="", flush=True)
                self._last_summary = summary_line

        except Exception as e:
            print(f"\r⚠️ Error updating summary: {e}", end="", flush=True)

    def _create_execution_tree(self) -> Tree:
        """Create comprehensive execution tree with enhanced features"""
        try:
            summary = self.tree_builder.get_execution_summary()
            session_info = summary["session_info"]
            timing = summary["timing"]
            health = summary["health_indicators"]

            # Root tree with health indicator
            health_emoji = "🟢" if health["overall_health"] > 0.8 else "🟡" if health["overall_health"] > 0.5 else "🔴"
            root_title = f"{health_emoji} Agent Execution Flow"

            if timing["elapsed"] > 0:
                root_title += f" ({timing['elapsed']:.1f}s elapsed)"

            tree = Tree(root_title, style="bold cyan")

            # Execution status overview
            self._add_execution_overview(tree, summary)

            # Main execution flow
            self._add_execution_flow_branch(tree, summary)

            # Error log (if any errors)
            if self.tree_builder.error_log and self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG, VerbosityMode.STANDARD]:
                self._add_error_log_branch(tree)

            # Performance metrics
            if self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                self._add_performance_branch(tree, summary)

            # Routing history
            if (self.tree_builder.routing_history and
                self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG, VerbosityMode.STANDARD]):
                self._add_routing_branch(tree)

            return tree

        except Exception as e:
            # Fallback tree on error
            error_tree = Tree("❌ Error creating execution tree", style="red")
            error_tree.add(f"Error: {str(e)}", style="red dim")
            return error_tree

    def _add_execution_overview(self, tree: Tree, summary: Dict[str, Any]):
        """Add execution overview section"""
        session_info = summary["session_info"]
        health = summary["health_indicators"]

        overview_text = (f"📊 Status: {session_info['completed_nodes']}/{session_info['total_nodes']} completed "
                         f"({health['completion_rate']:.1%})")

        if session_info["active_nodes"] > 0:
            overview_text += f" | {session_info['active_nodes']} active"
        if session_info["failed_nodes"] > 0:
            overview_text += f" | {session_info['failed_nodes']} failed"

        overview_branch = tree.add(overview_text, style="bold yellow")

        # Health indicators
        if self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
            health_text = f"Health: {health['overall_health']:.1%} | Error Rate: {health['error_rate']:.1%}"
            overview_branch.add(health_text, style="blue dim")

    def _add_execution_flow_branch(self, tree: Tree, summary: Dict[str, Any]):
        """Add detailed execution flow branch"""
        flow_branch = tree.add("🔄 Execution Flow", style="bold blue")

        execution_flow = summary["execution_flow"]["flow"]
        active_nodes = set(summary["execution_flow"]["active_nodes"])
        completion_order = summary["execution_flow"]["completion_order"]

        for i, node_name in enumerate(execution_flow):
            if node_name not in self.tree_builder.nodes:
                continue

            node = self.tree_builder.nodes[node_name]

            # Status icon and styling
            status_icon = node.get_status_icon()
            status_style = node.get_status_color()

            # Node info with enhanced details
            node_text = f"{status_icon} [{i + 1}] {node_name}"

            # Add timing info
            duration_str = node.get_duration_str()
            if duration_str != "...":
                node_text += f" ({duration_str})"

            # Add performance indicator
            if node.is_completed() and node.duration:
                efficiency = node._calculate_efficiency_score()
                if efficiency > 0.8:
                    node_text += ""
                elif efficiency < 0.5:
                    node_text += " 🐌"

            node_branch = flow_branch.add(node_text, style=status_style)

            # Add detailed information based on verbosity
            if self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                self._add_node_details(node_branch, node)
            elif self.mode == VerbosityMode.STANDARD and node.error:
                # Show errors even in standard mode
                node_branch.add(f"❌ {node.error}", style="red dim")

    def _add_node_details(self, parent_branch: Tree, node: ExecutionNode):
        """Add comprehensive node details"""

        # Strategy and reasoning
        if node.strategy:
            parent_branch.add(f"🎯 Strategy: {node.strategy}", style="cyan dim")
        if node.reasoning:
            parent_branch.add(f"🧠 Reasoning: {node.reasoning[:100]}...", style="blue dim")

        # Error details
        if node.error and node.error_details:
            error_branch = parent_branch.add(f"❌ Error: {node.error}", style="red")
            if self.mode == VerbosityMode.DEBUG:
                for key, value in node.error_details.items():
                    error_branch.add(f"{key}: {value}", style="red dim")

        # LLM calls summary
        if node.llm_calls:
            llm_summary = f"🧠 LLM: {len(node.llm_calls)} calls"
            if node.total_cost > 0:
                llm_summary += f", ${node.total_cost:.4f}"
            if node.total_tokens > 0:
                llm_summary += f", {node.total_tokens:,} tokens"

            llm_branch = parent_branch.add(llm_summary, style="blue dim")

            # Show individual calls in debug mode
            if self.mode == VerbosityMode.DEBUG:
                for call in node.llm_calls[-3:]:  # Last 3 calls
                    call_info = f"{call.llm_model or 'Unknown'}"
                    if call.llm_duration:
                        call_info += f" ({call.llm_duration:.1f}s)"
                    llm_branch.add(call_info, style="blue dim")

        # Tool calls summary
        if node.tool_calls:
            tool_summary = f"🔧 Tools: {len(node.tool_calls)} calls"
            successful_tools = sum(1 for call in node.tool_calls if call.tool_success)
            if successful_tools < len(node.tool_calls):
                tool_summary += f" ({successful_tools}/{len(node.tool_calls)} successful)"

            tool_branch = parent_branch.add(tool_summary, style="green dim")

            # Show individual tool calls
            if self.mode == VerbosityMode.DEBUG:
                for call in node.tool_calls[-3:]:  # Last 3 calls
                    success_icon = "✓" if call.tool_success else "✗"
                    call_info = f"{success_icon} {call.tool_name}"
                    if call.tool_duration:
                        call_info += f" ({call.tool_duration:.1f}s)"
                    style = "green dim" if call.tool_success else "red dim"
                    tool_branch.add(call_info, style=style)

        # Performance metrics
        if node.is_completed() and self.mode == VerbosityMode.DEBUG:
            perf = node.get_performance_summary()
            perf_text = f"📈 Efficiency: {perf['efficiency_score']:.1%}"
            if perf['retry_count'] > 0:
                perf_text += f" (Retries: {perf['retry_count']})"
            parent_branch.add(perf_text, style="yellow dim")

    def _add_error_log_branch(self, tree: Tree):
        """Add enhanced error log branch with detailed error information"""
        if not self.tree_builder.error_log:
            return

        error_branch = tree.add(f"❌ Error Log ({len(self.tree_builder.error_log)})", style="red bold")

        # Show recent errors with enhanced details
        recent_errors = self.tree_builder.error_log[-5:]  # Last 5 errors
        for error in recent_errors:
            timestamp = datetime.fromtimestamp(error["timestamp"]).strftime("%H:%M:%S")
            node_name = error.get("node", "Unknown")
            error_message = error.get("error", "Unknown error")
            error_type = error.get("error_type", "Unknown")

            # FIXED: Use actual error message instead of "Unknown error"
            if error_message and error_message != "Unknown error":
                error_text = f"[{timestamp}] {node_name}: {error_message}"
            else:
                error_text = f"[{timestamp}] {node_name}: {error_type} error"

            # Add additional context if available
            if error.get("task_id"):
                error_text += f" (Task: {error['task_id']})"
            elif error.get("tool_name"):
                error_text += f" (Tool: {error['tool_name']})"

            # Add retry info if available
            if error.get("retry_count", 0) > 0:
                error_text += f" (Retry #{error['retry_count']})"

            # Use different colors based on error severity
            if "critical" in error_message.lower() or "fatal" in error_message.lower():
                style = "red bold"
            elif error.get("source") == "task_execution":
                style = "red"
            else:
                style = "red dim"

            error_branch.add(error_text, style=style)

    def _add_performance_branch(self, tree: Tree, summary: Dict[str, Any]):
        """Add performance metrics branch"""
        perf = summary["performance_metrics"]
        health = summary["health_indicators"]
        timing = summary["timing"]

        perf_branch = tree.add("📊 Performance Metrics", style="bold green")

        # Cost and token metrics
        if perf["total_cost"] > 0:
            cost_text = f"💰 Cost: {self._format_cost(perf['total_cost'])}"
            perf_branch.add(cost_text, style="green dim")

        if perf["total_tokens"] > 0:
            tokens_text = f"🎯 Tokens: {perf['total_tokens']:,}"
            if timing["elapsed"] > 0:
                tokens_per_sec = perf["total_tokens"] / timing["elapsed"]
                tokens_text += f" ({tokens_per_sec:.0f}/sec)"
            perf_branch.add(tokens_text, style="green dim")

        # Efficiency metrics
        if health["average_node_efficiency"] > 0:
            efficiency_text = f"⚡ Avg Efficiency: {health['average_node_efficiency']:.1%}"
            perf_branch.add(efficiency_text, style="green dim")

        # Event processing rate
        if timing["elapsed"] > 0:
            events_per_sec = perf["total_events"] / timing["elapsed"]
            processing_text = f"📝 Events: {perf['total_events']} ({events_per_sec:.1f}/sec)"
            perf_branch.add(processing_text, style="green dim")

    def _add_routing_branch(self, tree: Tree):
        """Add routing decisions branch"""
        if not self.tree_builder.routing_history:
            return

        routing_branch = tree.add(f"🧭 Routing History ({len(self.tree_builder.routing_history)})",
                                  style="bold purple")

        # Show recent routing decisions
        recent_routes = self.tree_builder.routing_history[-5:]  # Last 5
        for i, route in enumerate(recent_routes):
            timestamp = datetime.fromtimestamp(route["timestamp"]).strftime("%H:%M:%S")
            route_text = f"[{timestamp}] {route['from']} → {route['to']}"
            if route["decision"] != "unknown":
                route_text += f" ({route['decision']})"
            routing_branch.add(route_text, style="purple dim")

    def _format_cost(self, cost: float) -> str:
        """Enhanced cost formatting"""
        if cost < 0.0001:
            return f"${cost * 1000000:.1f}μ"
        elif cost < 0.001:
            return f"${cost * 1000:.1f}m"
        elif cost < 1:
            return f"${cost * 1000:.1f}m"
        else:
            return f"${cost:.4f}"

    def _print_tree_update(self):
        """Print tree update with minimal realtime support"""
        try:
            if self._fallback_mode:
                self._print_fallback()
                return

            if not self.use_rich:
                self._print_fallback()
                return

            # In minimal realtime mode, only print one-line summary unless full tree is needed
            if self.realtime_minimal and self.mode == VerbosityMode.REALTIME and not self._needs_full_tree:
                self._print_one_line_summary()
                return

            # Full tree printing (existing logic)
            self._print_counter += 1
            summary = self.tree_builder.get_execution_summary()

            # If we printed a one-line summary before, clear it and add newline
            if self._last_summary and self.realtime_minimal:
                print()  # Move to next line
                self._last_summary = ""

            # Clear screen in realtime mode only for full tree updates
            if self.mode == VerbosityMode.REALTIME and self._print_counter > 1 and not self.realtime_minimal:
                self.console.clear()

            # Create and print header
            header = self._create_header(summary)
            tree = self._create_execution_tree()

            # Print everything
            self.console.print()
            self.console.print(header)
            self.console.print(tree)

            # Update progress in realtime mode
            if self.mode == VerbosityMode.REALTIME:
                self._update_progress_display(summary)

            # Reset full tree flag
            self._needs_full_tree = False

            # Reset error counter on successful print
            self._consecutive_errors = 0

        except Exception as e:
            self._consecutive_errors += 1
            if self._consecutive_errors <= self._error_threshold:
                print(f"⚠️  Print error #{self._consecutive_errors}: {e}")
                if self._consecutive_errors == self._error_threshold:
                    print("🔄 Switching to fallback mode...")
                    self._fallback_mode = True

            # Always try fallback
            self._print_fallback()

    def _create_header(self, summary: Dict[str, Any]) -> Panel:
        """Create informative header panel"""
        session_info = summary["session_info"]
        timing = summary["timing"]
        health = summary["health_indicators"]

        # Status indicators
        status_parts = []
        if session_info["active_nodes"] > 0:
            status_parts.append(f"🔄 Running")
        elif session_info["failed_nodes"] > 0:
            status_parts.append(f"❌ Errors")
        elif session_info["completed_nodes"] == session_info["total_nodes"]:
            status_parts.append(f"✅ Complete")
        else:
            status_parts.append(f"⏸️ Waiting")

        status_parts[-1] += f" ({self.agent_name})"

        status_text = " | ".join(status_parts)

        # Progress info
        progress_text = (f"Progress: {session_info['completed_nodes']}/{session_info['total_nodes']} "
                         f"({health['completion_rate']:.1%})")

        # Timing info
        timing_text = f"Runtime: {human_readable_time(timing['elapsed'])}"
        if timing["estimated_completion"]:
            eta = timing["estimated_completion"] - time.time()
            if eta > 0:
                timing_text += f" | ETA: {human_readable_time(eta)}"

        # Performance info
        perf_metrics = summary["performance_metrics"]
        perf_text = f"Events: {perf_metrics['total_events']}"
        if perf_metrics["total_cost"] > 0:
            perf_text += f" | Cost: {self._format_cost(perf_metrics['total_cost'])}"

        header_content = f"{status_text}\n{progress_text} | {timing_text}\n{perf_text}"

        return Panel(
            header_content,
            title=f"📊 Update #{self._print_counter}",
            style="cyan",
            box=box.ROUNDED
        )

    def _update_progress_display(self, summary: Dict[str, Any]):
        """Update progress display for realtime mode"""
        if not hasattr(self, 'progress'):
            return

        session_info = summary["session_info"]

        if not self.progress_task:
            description = f"Processing {session_info['total_nodes']} nodes..."
            self.progress_task = self.progress.add_task(description, total=session_info['total_nodes'])

        # Update progress
        completed = session_info["completed_nodes"]
        self.progress.update(self.progress_task, completed=completed)

        # Update description
        if session_info["active_nodes"] > 0:
            current_node = summary["execution_flow"]["current_node"]
            description = f"Processing: {current_node}..."
        else:
            description = "Processing complete"

        self.progress.update(self.progress_task, description=description)

    def _print_fallback(self):
        """Enhanced fallback printing without Rich"""
        try:
            summary = self.tree_builder.get_execution_summary()
            session_info = summary["session_info"]
            timing = summary["timing"]
            perf = summary["performance_metrics"]

            print(f"\n{'=' * 80}")
            print(f"🚀 AGENT EXECUTION UPDATE #{self._print_counter}")
            print(f"Session: {summary.get('session_id', 'unknown')} | Runtime: {timing['elapsed']:.1f}s")
            print(f"Progress: {session_info['completed_nodes']}/{session_info['total_nodes']} nodes")

            if session_info["failed_nodes"] > 0:
                print(f"❌ Failures: {session_info['failed_nodes']}")
            if perf["total_cost"] > 0:
                print(f"💰 Cost: {self._format_cost(perf['total_cost'])}")

            print(f"{'=' * 80}")

            # Show execution flow
            print("\n🔄 Execution Flow:")
            for i, node_name in enumerate(summary["execution_flow"]["flow"]):
                if node_name not in self.tree_builder.nodes:
                    continue

                node = self.tree_builder.nodes[node_name]
                status_icon = node.get_status_icon()
                duration = node.get_duration_str()

                print(f"  {status_icon} [{i + 1}] {node_name} ({duration})")

                if node.error and self.mode in [VerbosityMode.STANDARD, VerbosityMode.VERBOSE]:
                    print(f"    ❌ {node.error}")

            # Show errors in verbose modes
            if (self.tree_builder.error_log and
                self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]):
                print(f"\n❌ Recent Errors:")
                for error in self.tree_builder.error_log[-3:]:
                    timestamp = datetime.fromtimestamp(error["timestamp"]).strftime("%H:%M:%S")
                    print(f"  [{timestamp}] {error['node']}: {error['error']}")

            print(f"{'=' * 80}")

        except Exception as e:
            # Ultimate fallback
            print(f"\n⚠️  EXECUTION UPDATE #{self._print_counter} - Basic fallback")
            print(f"Agent Name: {self.agent_name}")
            print(f"Total events processed: {self.tree_builder.total_events}")
            print(f"Nodes: {len(self.tree_builder.nodes)}")
            print(f"Errors encountered: {len(self.tree_builder.error_log)}")
            if e:
                print(f"Print error: {e}")

    async def progress_callback(self, event: ProgressEvent):
        """Enhanced progress callback with automatic task detection"""
        is_task_update = False
        try:
            # Add event to tree builder
            self.tree_builder.add_event(event)

            # Store in history with size limit
            self.print_history.append({
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "node_name": event.node_name,
                "event_id": event.event_id
            })

            # Maintain history size limit
            if len(self.print_history) > self.max_history:
                self.print_history = self.print_history[-self.max_history:]

            # Automatic task detection and printing
            if event.event_type.startswith('task_'):
                self.print_task_update_from_event(event)
                is_task_update = True

            if event.node_name == "LLMReasonerNode":
                self.print_reasoner_update_from_event(event)

            # Check if we need to show full tree (errors or completion)
            if self.realtime_minimal:
                if (event.event_type == "error" or
                    event.success is False or
                    (event.metadata and event.metadata.get("error"))):
                    self._needs_full_tree = True

                if (event.event_type in ["execution_complete", "task_complete", "node_exit"] or
                    (event.node_name in self.tree_builder.nodes and
                     self.tree_builder.nodes[event.node_name].is_completed())):
                    summary = self.tree_builder.get_execution_summary()
                    if (summary["session_info"]["completed_nodes"] + summary["session_info"][
                        "failed_nodes"] ==
                        summary["session_info"]["total_nodes"]):
                        self._needs_full_tree = True

            # Print debug info in debug mode
            if self.mode == VerbosityMode.DEBUG:
                self._print_debug_event(event)

            # Agent name extraction
            self.agent_name = event.agent_name if event.agent_name else event.metadata.get("agent_name",
                                                                                          self.agent_name)
            # Print strategy and plan updates
            if (not is_task_update and
                event.node_name != "LLMReasonerNode" and  # Don't double-print reasoner events
                (event.node_name == "FlowAgent" or self._should_print_update())):
                self.print_strategy_from_event(event)
                self.print_plan_from_event(event)
                self._print_tree_update()

        except Exception as e:
            # Emergency error handling
            self._consecutive_errors += 1
            print(f"⚠️ Progress callback error #{self._consecutive_errors}: {e}")

            if self._consecutive_errors > self._error_threshold:
                print("🚨 Progress printing disabled due to excessive errors")
                self.progress_callback = self._noop_callback

    def print_reasoner_update_from_event(self, event: ProgressEvent):
        """Print reasoner updates and meta-tool usage based on events for all verbosity modes"""
        try:
            # Handle reasoner-related events in all modes (not just verbose)
            if (event.node_name != "LLMReasonerNode" or
                not event.metadata or
                event.event_type not in ["reasoning_loop", "meta_tool_call", "meta_tool_batch_complete",
                                         "meta_tool_analysis"]):
                return

            if event.event_type == "reasoning_loop":
                self._print_reasoning_loop_update(event)
            elif event.event_type == "meta_tool_call":
                self._print_meta_tool_update(event)
            elif event.event_type == "meta_tool_batch_complete":
                self._print_meta_tool_batch_summary(event)
            elif event.event_type == "meta_tool_analysis":
                self._print_meta_tool_analysis_update(event)

        except Exception as e:
            if self.mode == VerbosityMode.DEBUG:
                print(f"⚠️ Error printing reasoner update: {e}")

    def _print_meta_tool_batch_summary(self, event: ProgressEvent):
        """Print summary when multiple meta-tools complete"""
        try:
            metadata = event.metadata
            total_meta_tools = metadata.get("total_meta_tools_processed", 0)
            reasoning_loop = metadata.get("reasoning_loop", "?")
            final_context_size = metadata.get("final_context_size", 0)
            final_task_stack_size = metadata.get("final_task_stack_size", 0)
            meta_tools_executed = metadata.get("meta_tools_executed", [])
            batch_performance = metadata.get("batch_performance", {})

            if self._fallback_mode or not self.use_rich:
                if self.mode != VerbosityMode.MINIMAL:  # Skip in minimal mode
                    print(f"🔧 Batch Complete: {total_meta_tools} meta-tools in loop {reasoning_loop}")
                return

            # Only show batch summaries in detailed modes
            if self.mode == VerbosityMode.MINIMAL:
                return  # Skip batch summaries in minimal mode

            elif self.mode == VerbosityMode.STANDARD:
                # Simple batch summary
                if total_meta_tools > 2:  # Only show for larger batches
                    summary_text = f"🔧 Completed {total_meta_tools} operations in loop {reasoning_loop}"
                    self.console.print(summary_text, style="purple dim")

            elif self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                # Detailed batch summary
                summary_text = f"🔧 Meta-Tool Batch Complete"
                details = []
                details.append(f"🎯 Tools executed: {', '.join(meta_tools_executed)}")
                details.append(f"🔄 Loop: {reasoning_loop}")
                details.append(f"📚 Final context size: {final_context_size}")
                details.append(f"📋 Final task stack: {final_task_stack_size}")

                if self.mode == VerbosityMode.DEBUG and batch_performance:
                    details.append(f"📊 Tool diversity: {batch_performance.get('tool_diversity', 0)}")
                    most_used = batch_performance.get('most_used_tool', 'none')
                    if most_used != 'none':
                        details.append(f"🏆 Most used: {most_used}")

                batch_panel = Panel(
                    "\n".join(details),
                    title=summary_text,
                    style="purple",
                    box=box.ROUNDED
                )
                self.console.print(batch_panel)

            elif self.mode == VerbosityMode.REALTIME:
                if not self.realtime_minimal and total_meta_tools > 1:
                    self.console.print(f"🔧 {total_meta_tools} tools completed", style="purple dim")

        except Exception as e:
            print(f"⚠️ Error printing batch summary: {e}")

    def _print_meta_tool_analysis_update(self, event: ProgressEvent):
        """Print meta-tool analysis updates (when no tools found)"""
        metadata = event.metadata
        analysis_result = metadata.get("analysis_result", "")
        llm_response_length = metadata.get("llm_response_length", 0)
        reasoning_loop = metadata.get("reasoning_loop", "?")

        # Only show analysis in verbose/debug modes
        if self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
            if analysis_result == "no_meta_tools_detected":
                analysis_text = f"🔍 Loop {reasoning_loop}: No meta-tools in LLM response ({llm_response_length} chars)"

                if self.mode == VerbosityMode.DEBUG:
                    preview = metadata.get("llm_response_preview", "")
                    if preview:
                        analysis_panel = Panel(
                            f"{analysis_text}\n\n📄 Response preview:\n{preview}",
                            title="🔍 Meta-Tool Analysis",
                            style="orange3",
                            box=box.ROUNDED
                        )
                        self.console.print(analysis_panel)
                    else:
                        self.console.print(analysis_text, style="orange3 dim")
                else:
                    self.console.print(analysis_text, style="orange3 dim")

    def _print_reasoning_loop_update(self, event: ProgressEvent):
        """Print reasoning loop progress update for all modes with enhanced timestamps"""
        try:
            metadata = event.metadata
            loop_number = metadata.get("loop_number", "?")
            context_size = metadata.get("context_size", 0)
            task_stack_size = metadata.get("task_stack_size", 0)
            outline_step = metadata.get("outline_step", 0)
            auto_recovery_attempts = metadata.get("auto_recovery_attempts", 0)
            performance_metrics = metadata.get("performance_metrics", {})

            # Enhanced timestamp formatting
            timestamp = datetime.fromtimestamp(event.timestamp)

            if self._fallback_mode or not self.use_rich:
                # Fallback for all modes with enhanced info
                if self.mode == VerbosityMode.MINIMAL:
                    if loop_number == 1:  # Only show first loop in minimal
                        time_str = timestamp.strftime("%H:%M:%S")
                        print(f"🧠 [{time_str}] Starting reasoning...")
                elif self.mode == VerbosityMode.REALTIME:
                    if self.realtime_minimal:
                        time_str = timestamp.strftime("%H:%M:%S")
                        print(f"\r🧠 [{time_str}] Thinking... (step {loop_number})", end="", flush=True)
                    else:
                        time_str = timestamp.strftime("%H:%M:%S")
                        outline_info = f" | Step: {outline_step}" if outline_step > 0 else ""
                        recovery_info = f" | Recovery: {auto_recovery_attempts}" if auto_recovery_attempts > 0 else ""
                        print(
                            f"🧠 [{time_str}] Loop {loop_number}{outline_info} | Context: {context_size} | Tasks: {task_stack_size}{recovery_info}")
                else:
                    time_str = timestamp.strftime("%H:%M:%S")
                    outline_info = f" | Outline Step: {outline_step}" if outline_step > 0 else ""
                    print(
                        f"🧠 [{time_str}] Reasoning Loop #{loop_number}{outline_info} | Context: {context_size} | Tasks: {task_stack_size}")
                return

            # Rich formatted output for all modes with enhanced timestamps
            if self.mode == VerbosityMode.MINIMAL:
                if loop_number == 1:
                    time_str = timestamp.strftime("%H:%M:%S")
                    self.console.print(f"🧠 [{time_str}] Starting reasoning process...", style="cyan")
                elif loop_number % 5 == 0:  # Every 5th loop
                    time_str = timestamp.strftime("%H:%M:%S")
                    self.console.print(f"🧠 [{time_str}] Reasoning progress: Step #{loop_number}", style="cyan dim")

            elif self.mode == VerbosityMode.STANDARD:
                time_str = timestamp.strftime("%H:%M:%S")
                if loop_number == 1 or context_size > 5 or task_stack_size > 0 or auto_recovery_attempts > 0:

                    content_lines = [f"📚 Context: {context_size} entries", f"📋 Tasks: {task_stack_size} items"]
                    if outline_step > 0:
                        content_lines.append(f"📍 Outline Step: {outline_step}")
                    if auto_recovery_attempts > 0:
                        content_lines.append(f"🔄 Recovery Attempts: {auto_recovery_attempts}")
                    if performance_metrics.get("action_efficiency"):
                        efficiency = performance_metrics["action_efficiency"]
                        content_lines.append(f"📊 Efficiency: {efficiency:.1%}")

                    loop_panel = Panel(
                        "\n".join(content_lines),
                        title=f"🧠 [{time_str}] Reasoning Step #{loop_number}",
                        style="cyan",
                        box=box.ROUNDED
                    )
                    self.console.print(loop_panel)
                else:
                    self.console.print(f"🧠 [{time_str}] Step #{loop_number}", style="cyan dim")

            elif self.mode == VerbosityMode.VERBOSE:
                time_str = timestamp.strftime("%H:%M:%S")
                loop_content = [
                    f"📚 Context: {context_size} entries",
                    f"📋 Task Stack: {task_stack_size} items",
                    f"⏱️ Time: {time_str}"
                ]

                if outline_step > 0:
                    loop_content.append(f"📍 Outline Step: {outline_step}")
                if auto_recovery_attempts > 0:
                    loop_content.append(f"🔄 Recovery Attempts: {auto_recovery_attempts}")
                if performance_metrics:
                    if performance_metrics.get("action_efficiency"):
                        loop_content.append(f"📊 Action Efficiency: {performance_metrics['action_efficiency']:.1%}")
                    if performance_metrics.get("avg_loop_time"):
                        loop_content.append(f"⚡ Avg Loop Time: {performance_metrics['avg_loop_time']:.2f}s")

                loop_panel = Panel(
                    "\n".join(loop_content),
                    title=f"🧠 Reasoning Loop #{loop_number}",
                    style="cyan",
                    box=box.ROUNDED
                )
                self.console.print(loop_panel)

            elif self.mode == VerbosityMode.DEBUG:
                timestamp_detailed = timestamp.strftime("%H:%M:%S.%f")[:-3]
                debug_info = [
                    f"📚 Context Size: {context_size} entries",
                    f"📋 Task Stack: {task_stack_size} items",
                    f"⏱️ Timestamp: {timestamp_detailed}",
                    f"📊 Event ID: {event.event_id}",
                    f"🔄 Status: {event.status.value if event.status else 'unknown'}"
                ]

                if outline_step > 0:
                    debug_info.append(f"📍 Outline Step: {outline_step}")
                if auto_recovery_attempts > 0:
                    debug_info.append(f"🔄 Recovery Attempts: {auto_recovery_attempts}")

                if performance_metrics:
                    debug_info.append("📈 Performance Metrics:")
                    for key, value in performance_metrics.items():
                        if isinstance(value, float):
                            debug_info.append(f"  • {key}: {value:.3f}")
                        else:
                            debug_info.append(f"  • {key}: {value}")

                debug_panel = Panel(
                    "\n".join(debug_info),
                    title=f"🧠 Debug: Reasoning Loop #{loop_number}",
                    style="cyan bold",
                    box=box.HEAVY
                )
                self.console.print(debug_panel)

            elif self.mode == VerbosityMode.REALTIME:
                time_str = timestamp.strftime("%H:%M:%S")
                if self.realtime_minimal:
                    progress_indicators = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                    spinner = progress_indicators[(loop_number - 1) % len(progress_indicators)]
                    outline_info = f" step:{outline_step}" if outline_step > 0 else ""
                    print(f"\r{spinner} [{time_str}] Loop {loop_number}{outline_info} (ctx:{context_size})", end="",
                          flush=True)
                else:
                    outline_info = f" | Step: {outline_step}" if outline_step > 0 else ""
                    recovery_info = f" | Rec: {auto_recovery_attempts}" if auto_recovery_attempts > 0 else ""
                    self.console.print(
                        f"🧠 [{time_str}] Loop #{loop_number}{outline_info} | Context: {context_size} | Tasks: {task_stack_size}{recovery_info}",
                        style="cyan dim")

        except Exception as e:
            print(f"⚠️ Error printing reasoning loop: {e}")

    def _print_meta_tool_update(self, event: ProgressEvent):
        """Print meta-tool execution updates for all verbosity modes with enhanced timestamps"""
        try:
            metadata = event.metadata
            meta_tool_name = metadata.get("meta_tool_name", "unknown")
            execution_phase = metadata.get("execution_phase", "unknown")
            tool_category = metadata.get("tool_category", "unknown")

            # Enhanced timestamp
            timestamp = datetime.fromtimestamp(event.timestamp)

            # Handle different phases based on verbosity mode
            if execution_phase == "meta_tool_start" and self.mode == VerbosityMode.MINIMAL:
                return  # Skip start phase in minimal mode

            if self._fallback_mode or not self.use_rich:
                self._print_meta_tool_fallback(event, metadata, timestamp)
                return

            # Route to specific tool handlers based on verbosity mode
            if self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                # Detailed mode - use specific handlers
                if meta_tool_name == "internal_reasoning":
                    self._print_internal_reasoning_update(event, metadata, timestamp)
                elif meta_tool_name == "manage_internal_task_stack":
                    self._print_task_stack_update(event, metadata, timestamp)
                elif meta_tool_name == "delegate_to_llm_tool_node":
                    self._print_delegation_update(event, metadata, timestamp)
                elif meta_tool_name == "create_and_execute_plan":
                    self._print_plan_execution_update(event, metadata, timestamp)
                elif meta_tool_name == "direct_response":
                    self._print_direct_response_update(event, metadata, timestamp)
                elif meta_tool_name in ["advance_outline_step", "write_to_variables", "read_from_variables"]:
                    self._print_enhanced_meta_tool_update(event, metadata, timestamp)
                else:
                    self._print_generic_meta_tool_update(event, metadata, timestamp)
            else:
                # Simpler modes - use unified handler
                self._print_unified_meta_tool_update(event, metadata, timestamp)

        except Exception as e:
            print(f"⚠️ Error printing meta-tool update: {e}")

    def _print_internal_reasoning_update(self, event: ProgressEvent, metadata: Dict[str, Any], timestamp: datetime):
        """Print internal reasoning specific updates with insights and enhanced timestamp support"""
        if not event.success:
            return

        thought_number = metadata.get("thought_number", "?")
        total_thoughts = metadata.get("total_thoughts", "?")
        current_focus = metadata.get("current_focus", "")
        confidence_level = metadata.get("confidence_level", 0.0)
        key_insights = metadata.get("key_insights", [])
        key_insights_count = len(key_insights)
        potential_issues = metadata.get("potential_issues", [])
        next_thought_needed = metadata.get("next_thought_needed", False)
        outline_step = metadata.get("outline_step", 0)
        outline_step_progress = metadata.get("outline_step_progress", "")
        reasoning_depth = metadata.get("reasoning_depth", 0)

        # Enhanced timestamp formatting
        time_str = timestamp.strftime("%H:%M:%S")

        # Create reasoning update with timestamp
        reasoning_text = f"💭 [{time_str}] Thought {thought_number}/{total_thoughts}"

        # Add outline step info if available
        if outline_step > 0:
            reasoning_text += f" (Step {outline_step})"

        if current_focus:
            focus_preview = current_focus[:60] + "..." if len(current_focus) > 60 else current_focus
            reasoning_text += f"\n🎯 Focus: {focus_preview}"

        details = []
        if key_insights_count > 0:
            details.append(f"💡 {key_insights_count} insights")
        if confidence_level > 0:
            details.append(f"📊 {confidence_level:.1%} confidence")
        if next_thought_needed:
            details.append("➡️ More thinking needed")
        if outline_step_progress:
            details.append(f"📍 Progress: {outline_step_progress[:40]}...")

        # Add performance info in debug mode
        if self.mode == VerbosityMode.DEBUG:
            duration = metadata.get("execution_duration", 0)
            if duration > 0:
                details.append(f"⏱️ {duration:.2f}s")
            if reasoning_depth > 0:
                details.append(f"🔄 Depth: {reasoning_depth}")

        if self.mode == VerbosityMode.DEBUG:
            # Show detailed insights in debug mode
            debug_content = [reasoning_text]

            if details:
                debug_content.append("\n📊 Metrics:")
                debug_content.extend(f"• {detail}" for detail in details)

            # Show actual insights
            if key_insights:
                debug_content.append("\n💡 Key Insights:")
                for i, insight in enumerate(key_insights[:3], 1):  # Show up to 3 insights
                    insight_preview = insight[:80] + "..." if len(insight) > 80 else insight
                    debug_content.append(f"  {i}. {insight_preview}")

                if len(key_insights) > 3:
                    debug_content.append(f"  ... +{len(key_insights) - 3} more insights")

            # Show potential issues
            if potential_issues:
                debug_content.append("\n⚠️ Potential Issues:")
                for i, issue in enumerate(potential_issues[:2], 1):  # Show up to 2 issues
                    issue_preview = issue[:80] + "..." if len(issue) > 80 else issue
                    debug_content.append(f"  {i}. {issue_preview}")

                if len(potential_issues) > 2:
                    debug_content.append(f"  ... +{len(potential_issues) - 2} more issues")

            # Show outline step progress if available
            if outline_step_progress and len(outline_step_progress) > 40:
                debug_content.append(f"\n📍 Outline Progress:")
                debug_content.append(f"  {outline_step_progress}")

            reasoning_panel = Panel(
                "\n".join(debug_content),
                title="🧠 Internal Reasoning Analysis",
                style="white",
                box=box.ROUNDED
            )
            self.console.print(reasoning_panel)
        else:
            # Verbose mode - simpler display with enhanced info
            if details:
                reasoning_text += f"\n{', '.join(details)}"
            self.console.print(reasoning_text, style="white")

    def _print_task_stack_update(self, event: ProgressEvent, metadata: Dict[str, Any], timestamp: datetime):
        """Print task stack management updates with enhanced timestamp and outline step tracking"""
        if not event.success:
            return

        stack_action = metadata.get("stack_action", "unknown")
        task_description = metadata.get("task_description", "")
        outline_step_ref = metadata.get("outline_step_ref", "")
        stack_size_before = metadata.get("stack_size_before", 0)
        stack_size_after = metadata.get("stack_size_after", 0)
        stack_change = metadata.get("stack_change", 0)
        outline_step = metadata.get("outline_step", 0)

        # Enhanced timestamp formatting
        time_str = timestamp.strftime("%H:%M:%S")

        # Action icons
        action_icons = {
            "add": "➕",
            "remove": "➖",
            "complete": "✅",
            "get_current": "📋"
        }

        action_icon = action_icons.get(stack_action, "🔄")
        stack_text = f"{action_icon} [{time_str}] Stack {stack_action.title()}"

        # Add outline step context
        if outline_step > 0:
            stack_text += f" (Step {outline_step})"

        if stack_action in ["add", "remove", "complete"] and task_description:
            preview = task_description[:60] + "..." if len(task_description) > 60 else task_description
            stack_text += f": {preview}"

        # Show size change with enhanced info
        if stack_change != 0:
            change_text = f" ({stack_change:+d})" if stack_change != 0 else ""
            stack_text += f"\n📊 Size: {stack_size_before} → {stack_size_after}{change_text}"
        elif stack_action == "get_current":
            stack_text += f"\n📊 Current size: {stack_size_after} items"

        # Add outline step reference if available
        if outline_step_ref and outline_step_ref != f"step_{outline_step}":
            stack_text += f"\n📍 Linked to: {outline_step_ref}"

        # Add performance info in debug mode
        if self.mode == VerbosityMode.DEBUG:
            duration = metadata.get("execution_duration", 0)
            if duration > 0:
                stack_text += f"\n⏱️ Duration: {duration:.3f}s"

            # Show additional debug info
            debug_details = []
            if metadata.get("reasoning_loop"):
                debug_details.append(f"Loop: {metadata['reasoning_loop']}")
            if metadata.get("context_before_size") and metadata.get("context_after_size"):
                ctx_before = metadata["context_before_size"]
                ctx_after = metadata.get("context_after_size", ctx_before)
                if ctx_after != ctx_before:
                    debug_details.append(f"Context: {ctx_before}→{ctx_after}")

            if debug_details:
                stack_text += f"\n🔧 Debug: {', '.join(debug_details)}"

        self.console.print(stack_text, style="yellow")

    def _print_delegation_update(self, event: ProgressEvent, metadata: Dict[str, Any], timestamp: datetime):
        """Print delegation to LLMToolNode updates with enhanced timestamp and variable system integration"""
        delegated_task = metadata.get("delegated_task_description", "")
        tools_list = metadata.get("tools_list", [])
        tools_count = metadata.get("tools_count", 0)
        execution_phase = metadata.get("execution_phase", "")
        delegation_complexity = metadata.get("delegation_complexity", "unknown")
        outline_step = metadata.get("outline_step", 0)
        raw = metadata.get("raw_args_string", "")
        if raw:
            task_desc = raw.split('task_description="', 1)[1].split('", tools_list=', 1)[0]
            tools = raw.split('tools_list=', 1)[1]
            task_preview = f"{tools} "
            task_preview += task_desc[:1000 if self.mode == VerbosityMode.VERBOSE or self.mode == VerbosityMode.DEBUG else 80] + '...'
        else:
            task_preview = ""
        outline_step_completion = metadata.get("outline_step_completion", False)

        # Enhanced timestamp formatting
        time_str = timestamp.strftime("%H:%M:%S")

        if execution_phase == "meta_tool_start":
            # Starting delegation - show task description in all modes
            if self.mode == VerbosityMode.VERBOSE or self.mode == VerbosityMode.DEBUG:
                # Detailed view for verbose/debug
                # Final minimalist output
                delegation_text = f"🎯 [{time_str}] Delegating: {task_preview}"

                # Add outline step context
                if outline_step > 0:
                    delegation_text += f" (Step {outline_step})"
                if outline_step_completion:
                    delegation_text += " [Step Completion Expected]"

                if tools_count > 0:
                    tools_preview = ", ".join(tools_list[:3])
                    if len(tools_list) > 3:
                        tools_preview += f" +{len(tools_list) - 3} more"
                    delegation_text += f"\n🔧 Tools: [{tools_preview}]"

                if self.mode == VerbosityMode.DEBUG:
                    delegation_text += f"\n📊 Complexity: {delegation_complexity}"

                    # Add variable system context if available
                    if metadata.get("variable_system_context"):
                        delegation_text += f"\n💾 Variables available: {metadata['variable_system_context']}"

                self.console.print(delegation_text, style="green")

            elif self.mode == VerbosityMode.STANDARD:
                # Standard mode - show task description in panel
                panel_content = f"📄 Task: {task_preview}"

                # Add outline context
                if outline_step > 0:
                    panel_content += f"\n📍 Outline Step: {outline_step}"
                    if outline_step_completion:
                        panel_content += " (Completion Expected)"

                if tools_count > 0:
                    tools_preview = ", ".join(tools_list[:4])
                    if len(tools_list) > 4:
                        tools_preview += f" +{len(tools_list) - 4} more"
                    panel_content += f"\n🔧 Available Tools: {tools_preview}"

                delegation_panel = Panel(
                    panel_content,
                    title=f"🎯 [{time_str}] Delegating Task to LLM Tool Node",
                    style="green",
                    box=box.ROUNDED
                )
                self.console.print(delegation_panel)

            else:
                # Minimalist output for other modes
                delegation_text = f"🎯 [{time_str}] Delegating task"
                if outline_step > 0:
                    delegation_text += f" (Step {outline_step})"
                self.console.print(delegation_text + task_preview[:50]+'...', style="green")

        elif event.success and execution_phase != "meta_tool_start":
            # Delegation completed with enhanced info
            duration = metadata.get("execution_duration", 0)
            duration_str = f" ({duration:.1f}s)" if duration > 0.1 else ""

            if self.mode == VerbosityMode.STANDARD:
                # Show completion with brief task reference in standard mode
                task_brief = delegated_task[:50] + "..." if len(delegated_task) > 50 else delegated_task
                completion_text = f"✅ [{time_str}] Task completed: {task_brief}"

                # Add outline step completion info
                if outline_step_completion:
                    completion_text += " ✓ Step Complete"
                elif outline_step > 0:
                    completion_text += f" (Step {outline_step})"

                completion_text += duration_str

                if tools_count > 0:
                    completion_text += f" | Used {tools_count} tools"

                # Show any variable system results
                if metadata.get("variable_results_stored"):
                    completion_text += f" | Results stored in variables"

            else:
                # Simpler completion for verbose/debug
                completion_text = f"✅ [{time_str}] Delegation completed"

                if outline_step > 0:
                    completion_text += f" (Step {outline_step})"
                if outline_step_completion:
                    completion_text += " ✓ Step Complete"

                completion_text += duration_str

                if tools_count > 0:
                    completion_text += f" | Used {tools_count} tools"

                # Debug mode: show additional delegation details
                if self.mode == VerbosityMode.DEBUG:
                    debug_details = []
                    if delegation_complexity != "unknown":
                        debug_details.append(f"Complexity: {delegation_complexity}")
                    if metadata.get("sub_system_execution"):
                        debug_details.append("Sub-system executed")
                    if metadata.get("variable_integration"):
                        debug_details.append("Variable system integrated")

                    if debug_details:
                        completion_text += f"\n🔧 Debug: {', '.join(debug_details)}"

            self.console.print(completion_text, style="green")

    def _print_plan_execution_update(self, event: ProgressEvent, metadata: Dict[str, Any], timestamp: datetime):
        """Print plan creation and execution updates with enhanced timeline and variable integration"""
        goals_list = metadata.get("goals_list", [])
        goals_count = metadata.get("goals_count", 0)
        execution_phase = metadata.get("execution_phase", "")
        estimated_complexity = metadata.get("estimated_complexity", "unknown")
        outline_step = metadata.get("outline_step", 0)
        outline_step_completion = metadata.get("outline_step_completion", False)

        # Enhanced timestamp formatting
        time_str = timestamp.strftime("%H:%M:%S")

        if execution_phase == "meta_tool_start":
            # Starting plan execution
            plan_text = f"📋 [{time_str}] Creating Plan: {goals_count} goals"

            # Add outline context
            if outline_step > 0:
                plan_text += f" (Step {outline_step})"
            if outline_step_completion:
                plan_text += " [Step Completion Expected]"

            if self.mode == VerbosityMode.DEBUG and goals_list:
                goals_preview = []
                for i, goal in enumerate(goals_list[:4], 1):
                    goal_short = goal[:50] + "..." if len(goal) > 50 else goal
                    goals_preview.append(f"{i}. {goal_short}")

                if len(goals_list) > 4:
                    goals_preview.append(f"... +{len(goals_list) - 4} more goals")

                debug_content = []
                debug_content.append(f"📊 Complexity: {estimated_complexity}")
                if outline_step > 0:
                    debug_content.append(f"📍 Outline Step: {outline_step}")
                if outline_step_completion:
                    debug_content.append("✓ Will complete outline step")

                # Add variable system integration info
                if metadata.get("variable_system_integration"):
                    debug_content.append("💾 Variable system integration enabled")

                debug_content.append("")
                debug_content.extend(goals_preview)

                plan_panel = Panel(
                    "\n".join(debug_content),
                    title=f"📋 [{time_str}] Plan Creation",
                    style="magenta",
                    box=box.ROUNDED
                )
                self.console.print(plan_panel)
            else:
                if estimated_complexity != "unknown":
                    plan_text += f" (complexity: {estimated_complexity})"
                self.console.print(plan_text, style="magenta")

        elif event.success and execution_phase != "meta_tool_start":
            # Plan execution completed with enhanced results
            duration = metadata.get("execution_duration", 0)
            duration_str = f" ({duration:.1f}s)" if duration > 0.5 else ""

            completion_text = f"✅ [{time_str}] Plan execution completed"

            # Add outline context
            if outline_step_completion:
                completion_text += " ✓ Step Complete"
            elif outline_step > 0:
                completion_text += f" (Step {outline_step})"

            completion_text += duration_str

            if goals_count > 0:
                completion_text += f" | {goals_count} goals processed"

            # Show additional execution details in verbose/debug modes
            if self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                additional_details = []

                if estimated_complexity != "unknown":
                    additional_details.append(f"Complexity: {estimated_complexity}")

                # Show task execution results if available
                if metadata.get("tasks_completed"):
                    additional_details.append(f"Tasks completed: {metadata['tasks_completed']}")
                if metadata.get("tasks_failed"):
                    additional_details.append(f"Tasks failed: {metadata['tasks_failed']}")

                # Show variable system integration results
                if metadata.get("results_stored_in_variables"):
                    additional_details.append("Results stored in variables")
                if metadata.get("variable_references_resolved"):
                    additional_details.append(f"Variable refs: {metadata['variable_references_resolved']}")

                if additional_details and self.mode == VerbosityMode.DEBUG:
                    completion_text += f"\n🔧 Details: {', '.join(additional_details)}"
                elif additional_details and self.mode == VerbosityMode.VERBOSE:
                    completion_text += f" | {additional_details[0]}"

            self.console.print(completion_text, style="magenta")

    def _print_direct_response_update(self, event: ProgressEvent, metadata: Dict[str, Any], timestamp: datetime):
        """Print direct response (flow termination) updates with enhanced session completion info"""
        final_answer_length = metadata.get("final_answer_length", 0)
        reasoning_complete = metadata.get("reasoning_complete", False)
        total_reasoning_steps = metadata.get("total_reasoning_steps", 0)
        outline_completion = metadata.get("outline_completion", False)
        steps_completed = metadata.get("steps_completed", [])
        session_completion = metadata.get("session_completion", False)
        reasoning_summary = metadata.get("reasoning_summary", "")

        # Enhanced timestamp formatting
        time_str = timestamp.strftime("%H:%M:%S")

        if reasoning_complete and session_completion:
            if self.mode == VerbosityMode.MINIMAL:
                self.console.print(f"✅ [{time_str}] Response ready", style="green bold")

            elif self.mode == VerbosityMode.STANDARD:
                response_text = f"✨ [{time_str}] Final response generated ({final_answer_length} characters)"

                # Add outline completion status
                if outline_completion and len(steps_completed) > 0:
                    response_text += f" | {len(steps_completed)} steps completed"

                self.console.print(response_text, style="green bold")

            else:  # VERBOSE/DEBUG
                response_text = f"✨ [{time_str}] Final Response Generated"
                details = [
                    f"📝 Length: {final_answer_length} characters",
                    f"🧠 Reasoning steps: {total_reasoning_steps}"
                ]

                # Add outline completion details
                if outline_completion:
                    details.append(f"📋 Outline completed: {len(steps_completed)} steps")

                # Add session completion info
                if session_completion:
                    details.append("🎯 Session successfully completed")

                if self.mode == VerbosityMode.DEBUG:
                    duration = metadata.get("execution_duration", 0)
                    if duration > 0:
                        details.append(f"⏱️ Generation time: {duration:.3f}s")

                    if reasoning_summary:
                        details.append(f"📊 {reasoning_summary}")

                    # Add variable system completion info
                    if metadata.get("results_stored_in_variables"):
                        details.append("💾 Results stored in variable system")
                    if metadata.get("session_data_archived"):
                        details.append("📚 Session data archived")

                    # Show completed steps in debug mode
                    if steps_completed and len(steps_completed) <= 5:
                        details.append("✅ Completed steps:")
                        for i, step in enumerate(steps_completed[:3], 1):
                            step_preview = step[:50] + "..." if len(step) > 50 else step
                            details.append(f"  {i}. {step_preview}")
                        if len(steps_completed) > 3:
                            details.append(f"  ... +{len(steps_completed) - 3} more")

                response_panel = Panel(
                    "\n".join(details),
                    title=response_text,
                    style="green bold",
                    box=box.ROUNDED
                )
                self.console.print(response_panel)

    def _print_generic_meta_tool_update(self, event: ProgressEvent, metadata: Dict[str, Any], timestamp: datetime):
        """Print generic meta-tool updates for unknown tools with enhanced variable system support"""
        meta_tool_name = metadata.get("meta_tool_name", "unknown")
        execution_phase = metadata.get("execution_phase", "")
        outline_step = metadata.get("outline_step", 0)

        # Enhanced timestamp formatting
        time_str = timestamp.strftime("%H:%M:%S")

        if execution_phase == "meta_tool_start":
            if self.mode in [VerbosityMode.STANDARD, VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                start_text = f"🔧 [{time_str}] {meta_tool_name.replace('_', ' ').title()} starting..."

                # Add outline context
                if outline_step > 0:
                    start_text += f" (Step {outline_step})"

                # Add any additional context in debug mode
                if self.mode == VerbosityMode.DEBUG:
                    debug_details = []
                    if metadata.get("tool_category"):
                        debug_details.append(f"Category: {metadata['tool_category']}")
                    if metadata.get("variable_system_operation"):
                        debug_details.append(f"Variable op: {metadata['variable_system_operation']}")
                    if metadata.get("reasoning_loop"):
                        debug_details.append(f"Loop: {metadata['reasoning_loop']}")

                    if debug_details:
                        start_text += f"\n🔧 {', '.join(debug_details)}"

                self.console.print(start_text, style="white dim")

        elif event.success:
            duration = metadata.get("execution_duration", 0)
            duration_str = f" ({duration:.1f}s)" if duration > 0.1 else ""

            completion_text = f"✅ [{time_str}] {meta_tool_name.replace('_', ' ').title()} completed"

            # Add outline context
            if outline_step > 0:
                completion_text += f" (Step {outline_step})"

            completion_text += duration_str

            # Add specific results based on metadata
            result_details = []

            # Variable system operations
            if metadata.get("variable_system_operation") == "write":
                var_scope = metadata.get("variable_scope", "")
                var_key = metadata.get("variable_key", "")
                if var_scope and var_key:
                    result_details.append(f"Stored: {var_scope}.{var_key}")
            elif metadata.get("variable_system_operation") == "read":
                var_scope = metadata.get("variable_scope", "")
                var_key = metadata.get("variable_key", "")
                if var_scope and var_key:
                    result_details.append(f"Retrieved: {var_scope}.{var_key}")

            # Performance scores
            if metadata.get("performance_score") and self.mode == VerbosityMode.DEBUG:
                score = metadata["performance_score"]
                result_details.append(f"Performance: {score:.1%}")

            # Context changes
            if (metadata.get("context_before_size") and metadata.get("context_after_size") and
                metadata["context_before_size"] != metadata["context_after_size"]):
                ctx_before = metadata["context_before_size"]
                ctx_after = metadata["context_after_size"]
                result_details.append(f"Context: {ctx_before}→{ctx_after}")

            # Add result details to completion text
            if result_details:
                if self.mode == VerbosityMode.DEBUG:
                    completion_text += f"\n🔧 Details: {', '.join(result_details)}"
                else:
                    completion_text += f" | {result_details[0]}"

            self.console.print(completion_text, style="white")

        else:
            error_message = metadata.get("error_message", "Unknown error")
            error_text = f"❌ [{time_str}] {meta_tool_name} failed"

            # Add outline context
            if outline_step > 0:
                error_text += f" (Step {outline_step})"

            error_text += f": {error_message}"

            # Add recovery info in debug mode
            if self.mode == VerbosityMode.DEBUG and metadata.get("recovery_recommended"):
                error_text += "\n🔄 Auto-recovery recommended"

            self.console.print(error_text, style="red")

    def _print_unified_meta_tool_update(self, event: ProgressEvent, metadata: Dict[str, Any], timestamp: datetime):
        """Unified meta-tool update with enhanced timestamp display"""
        meta_tool_name = metadata.get("meta_tool_name", "unknown")
        execution_phase = metadata.get("execution_phase", "unknown")
        tool_category = metadata.get("tool_category", "unknown")
        args_string = metadata.get("raw_args_string", "unknown")
        outline_step = metadata.get("outline_step", 0)

        if "purpose" in args_string:
            args_string = args_string.split("purpose")[1].split("}")[0]
        elif "description" in args_string:
            args_string = args_string.split("description")[1].split(',')[0]
        elif "thought" in args_string:
            args_string = args_string.split("thought")[1].split(',')[0]
        else:
            args_string = "..."

        # Tool icons and colors
        tool_icons = {
            "internal_reasoning": "💭",
            "manage_internal_task_stack": "📋",
            "delegate_to_llm_tool_node": "🎯",
            "create_and_execute_plan": "📋",
            "advance_outline_step": "➡️",
            "write_to_variables": "💾",
            "read_from_variables": "📖",
            "direct_response": "✨"
        }

        tool_colors = {
            "thinking": "white",
            "planning": "yellow",
            "delegation": "green",
            "orchestration": "magenta",
            "completion": "green bold"
        }

        icon = tool_icons.get(meta_tool_name, "🔧")
        color = tool_colors.get(tool_category, "white")
        time_str = timestamp.strftime("%H:%M:%S")

        if execution_phase == "meta_tool_start":
            if self.mode == VerbosityMode.MINIMAL:
                # Show more tools in minimal mode for better visibility
                if meta_tool_name in ["create_and_execute_plan", "direct_response", "delegate_to_llm_tool_node",
                                      "advance_outline_step"]:
                    tool_name_display = meta_tool_name.replace('_', ' ').title()
                    outline_info = f" (step {outline_step})" if outline_step > 0 else ""
                    self.console.print(f"{icon} [{time_str}] {tool_name_display}{outline_info}...", style=color)

            elif self.mode == VerbosityMode.STANDARD:
                # Show all tools with brief description
                tool_descriptions = {
                    "internal_reasoning": "Analyzing and thinking",
                    "manage_internal_task_stack": "Managing task queue",
                    "delegate_to_llm_tool_node": "Delegating to tool system",
                    "create_and_execute_plan": "Creating execution plan",
                    "advance_outline_step": "Advancing outline step",
                    "write_to_variables": "Storing data",
                    "read_from_variables": "Retrieving data",
                    "direct_response": "Generating final response"
                }

                description = tool_descriptions.get(meta_tool_name, meta_tool_name.replace('_', ' ').title())
                outline_info = f" {args_string} (step {outline_step})" if outline_step > 0 else ""
                self.console.print(f"{icon} [{time_str}] {description}{outline_info}...", style=color)

            elif self.mode == VerbosityMode.REALTIME:
                if self.realtime_minimal:
                    if meta_tool_name in ["delegate_to_llm_tool_node", "create_and_execute_plan", "direct_response",
                                          "advance_outline_step"]:
                        tool_brief = {
                            "delegate_to_llm_tool_node": "Delegating",
                            "create_and_execute_plan": "Planning",
                            "advance_outline_step": "Advancing",
                            "direct_response": "Responding"
                        }
                        brief_name = tool_brief.get(meta_tool_name, meta_tool_name.replace('_', ' '))
                        outline_info = f":s{outline_step}" if outline_step > 0 else ""
                        print(f"\r{icon} [{time_str}] {brief_name}{outline_info}...", end="", flush=True)
                else:
                    tool_display = meta_tool_name.replace('_', ' ').title()
                    outline_info = f" {args_string}  (step {outline_step})" if outline_step > 0 else ""
                    self.console.print(f"{icon} [{time_str}] {tool_display}{outline_info} starting...",
                                       style=f"{color} dim")

        elif event.success and execution_phase != "meta_tool_start":
            # Enhanced completion messages with timestamps
            duration = metadata.get("execution_duration", 0)
            duration_str = f" ({duration:.1f}s)" if duration > 0.1 else ""

            if self.mode == VerbosityMode.MINIMAL:
                # Show completion for important tools
                if meta_tool_name == "direct_response":
                    answer_length = metadata.get("final_answer_length", 0)
                    self.console.print(f"✅ [{time_str}] Response ready ({answer_length} chars){duration_str}",
                                       style="green bold")
                elif meta_tool_name == "create_and_execute_plan":
                    goals_count = metadata.get("goals_count", 0)
                    self.console.print(f"✅ [{time_str}] Plan completed ({goals_count} goals){duration_str}",
                                       style="green")
                elif meta_tool_name == "delegate_to_llm_tool_node":
                    self.console.print(f"✅ [{time_str}] Task delegated successfully{duration_str}", style="green")
                elif meta_tool_name == "advance_outline_step":
                    step_completed = metadata.get("step_completed", False)
                    if step_completed:
                        self.console.print(f"✅ [{time_str}] Outline step advanced{duration_str}", style="green")

            elif self.mode == VerbosityMode.STANDARD:
                # Show all completions with enhanced results
                if meta_tool_name == "internal_reasoning":
                    thought_num = metadata.get("thought_number", "?")
                    focus = metadata.get("current_focus", "")[:40] + "..." if len(
                        metadata.get("current_focus", "")) > 40 else metadata.get("current_focus", "")
                    confidence = metadata.get("confidence_level", 0)
                    confidence_str = f" ({confidence:.1%})" if confidence > 0 else ""
                    if focus:
                        self.console.print(
                            f"💭 [{time_str}] Thought {thought_num}: {focus}{confidence_str}{duration_str}",
                            style="white")

                elif meta_tool_name == "manage_internal_task_stack":
                    action = metadata.get("stack_action", "")
                    stack_size = metadata.get("stack_size_after", 0)
                    outline_ref = metadata.get("outline_step_ref", "")
                    ref_info = f" ({outline_ref})" if outline_ref else ""
                    self.console.print(
                        f"📋 [{time_str}] Task stack {action}: {stack_size} items{ref_info}{duration_str}",
                        style="yellow")

                elif meta_tool_name == "delegate_to_llm_tool_node":
                    task_desc = metadata.get("delegated_task_description", "")
                    outline_completion = metadata.get("outline_step_completion", False)
                    completion_info = " ✓ Step Complete" if outline_completion else ""
                    if task_desc:
                        task_brief = task_desc[:60] + "..." if len(task_desc) > 60 else task_desc
                        self.console.print(f"🎯 [{time_str}] Completed: {task_brief}{completion_info}{duration_str}",
                                           style="green")

                elif meta_tool_name == "advance_outline_step":
                    step_completed = metadata.get("step_completed", False)
                    completion_evidence = metadata.get("completion_evidence", "")[:50] + "..." if len(
                        metadata.get("completion_evidence", "")) > 50 else metadata.get("completion_evidence", "")
                    if step_completed:
                        self.console.print(f"➡️ [{time_str}] Step advanced: {completion_evidence}{duration_str}",
                                           style="green")

                elif meta_tool_name == "write_to_variables":
                    var_scope = metadata.get("variable_scope", "")
                    var_key = metadata.get("variable_key", "")
                    self.console.print(f"💾 [{time_str}] Stored: {var_scope}.{var_key}{duration_str}", style="blue")

                elif meta_tool_name == "read_from_variables":
                    var_scope = metadata.get("variable_scope", "")
                    var_key = metadata.get("variable_key", "")
                    self.console.print(f"📖 [{time_str}] Retrieved: {var_scope}.{var_key}{duration_str}", style="blue")

                elif meta_tool_name == "create_and_execute_plan":
                    goals_count = metadata.get("goals_count", 0)
                    complexity = metadata.get("estimated_complexity", "")
                    complexity_str = f" ({complexity})" if complexity and complexity != "unknown" else ""
                    outline_completion = metadata.get("outline_step_completion", False)
                    completion_info = " ✓ Step Complete" if outline_completion else ""
                    self.console.print(
                        f"📋 [{time_str}] Plan executed: {goals_count} goals{complexity_str}{completion_info}{duration_str}",
                        style="magenta")

                elif meta_tool_name == "direct_response":
                    answer_length = metadata.get("final_answer_length", 0)
                    total_steps = metadata.get("total_reasoning_steps", 0)
                    self.console.print(
                        f"✨ [{time_str}] Response generated ({answer_length} chars, {total_steps} reasoning steps){duration_str}",
                        style="green bold")

            elif self.mode == VerbosityMode.REALTIME:
                if self.realtime_minimal:
                    # Clear the line and show completion with time
                    if meta_tool_name == "direct_response":
                        print(f"\r✅ [{time_str}] Response ready                    ")
                    elif meta_tool_name == "create_and_execute_plan":
                        goals_count = metadata.get("goals_count", 0)
                        print(f"\r✅ [{time_str}] Plan done ({goals_count})           ")
                    elif meta_tool_name == "delegate_to_llm_tool_node":
                        print(f"\r✅ [{time_str}] Task completed                    ")
                    elif meta_tool_name == "advance_outline_step":
                        if metadata.get("step_completed", False):
                            print(f"\r➡️ [{time_str}] Step advanced                    ")
                else:
                    # Full realtime updates with timestamp and duration
                    tool_display = meta_tool_name.replace('_', ' ').title()
                    outline_info = f" (step {outline_step})" if outline_step > 0 else ""

                    if meta_tool_name == "direct_response":
                        answer_length = metadata.get("final_answer_length", 0)
                        self.console.print(
                            f"✅ [{time_str}] {tool_display} complete: {answer_length} chars{outline_info}{duration_str}",
                            style="green bold")
                    elif meta_tool_name == "create_and_execute_plan":
                        goals_count = metadata.get("goals_count", 0)
                        outline_completion = metadata.get("outline_step_completion", False)
                        completion_info = " ✓" if outline_completion else ""
                        self.console.print(
                            f"✅ [{time_str}] {tool_display} complete: {goals_count} goals{completion_info}{outline_info}{duration_str}",
                            style="magenta")
                    elif meta_tool_name == "delegate_to_llm_tool_node":
                        tools_count = metadata.get("tools_count", 0)
                        outline_completion = metadata.get("outline_step_completion", False)
                        completion_info = " ✓" if outline_completion else ""
                        self.console.print(
                            f"✅ [{time_str}] {tool_display} complete: {tools_count} tools{completion_info}{outline_info}{duration_str}",
                            style="green")
                    else:
                        self.console.print(f"✅ [{time_str}] {tool_display} completed{outline_info}{duration_str}",
                                           style=f"{color} dim")

        elif not event.success:
            # Error messages with timestamps - show in all modes except minimal realtime
            if not (self.mode == VerbosityMode.REALTIME and self.realtime_minimal):
                error_message = metadata.get("error_message", "Unknown error")
                outline_info = f" (step {outline_step})" if outline_step > 0 else ""
                if self.mode == VerbosityMode.REALTIME and self.realtime_minimal:
                    print(f"\r❌ [{time_str}] {meta_tool_name.replace('_', ' ').title()} failed      ")
                else:
                    self.console.print(
                        f"❌ [{time_str}] {meta_tool_name.replace('_', ' ').title()} failed{outline_info}: {error_message}",
                        style="red")

    def _print_enhanced_meta_tool_update(self, event: ProgressEvent, metadata: Dict[str, Any], timestamp: datetime):
        """Print updates for enhanced meta-tools (advance_outline_step, write_to_variables, read_from_variables)"""
        if not event.success:
            return

        meta_tool_name = metadata.get("meta_tool_name", "unknown")
        time_str = timestamp.strftime("%H:%M:%S")
        duration = metadata.get("execution_duration", 0)
        duration_str = f" ({duration:.2f}s)" if duration > 0.01 else ""

        if meta_tool_name == "advance_outline_step":
            step_completed = metadata.get("step_completed", False)
            completion_evidence = metadata.get("completion_evidence", "")
            next_step_focus = metadata.get("next_step_focus", "")
            step_progression = metadata.get("step_progression", "")

            if step_completed:
                advancement_text = f"➡️ [{time_str}] Outline Step Advanced"
                if step_progression:
                    advancement_text += f" ({step_progression})"

                details = []
                if completion_evidence:
                    evidence_preview = completion_evidence[:80] + "..." if len(
                        completion_evidence) > 80 else completion_evidence
                    details.append(f"✓ Evidence: {evidence_preview}")
                if next_step_focus:
                    focus_preview = next_step_focus[:60] + "..." if len(next_step_focus) > 60 else next_step_focus
                    details.append(f"🎯 Next Focus: {focus_preview}")

                if self.mode == VerbosityMode.DEBUG and details:
                    advancement_panel = Panel(
                        "\n".join(details),
                        title=advancement_text + duration_str,
                        style="green",
                        box=box.ROUNDED
                    )
                    self.console.print(advancement_panel)
                else:
                    if details and self.mode == VerbosityMode.VERBOSE:
                        self.console.print(f"{advancement_text}{duration_str}\n{details[0]}", style="green")
                    else:
                        self.console.print(f"{advancement_text}{duration_str}", style="green")

        elif meta_tool_name == "write_to_variables":
            var_scope = metadata.get("variable_scope", "")
            var_key = metadata.get("variable_key", "")
            var_description = metadata.get("variable_description", "")

            var_text = f"💾 [{time_str}] Stored Variable: {var_scope}.{var_key}{duration_str}"

            if var_description and self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                desc_preview = var_description[:60] + "..." if len(var_description) > 60 else var_description
                self.console.print(f"{var_text}\n📄 {desc_preview}", style="blue")
            else:
                self.console.print(var_text, style="blue")

        elif meta_tool_name == "read_from_variables":
            var_scope = metadata.get("variable_scope", "")
            var_key = metadata.get("variable_key", "")
            read_purpose = metadata.get("read_purpose", "")

            var_text = f"📖 [{time_str}] Retrieved Variable: {var_scope}.{var_key}{duration_str}"

            if read_purpose and self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                purpose_preview = read_purpose[:60] + "..." if len(read_purpose) > 60 else read_purpose
                self.console.print(f"{var_text}\n🎯 Purpose: {purpose_preview}", style="blue")
            else:
                self.console.print(var_text, style="blue")

    def _print_meta_tool_fallback(self, event: ProgressEvent, metadata: Dict[str, Any], timestamp: datetime):
        """Fallback meta-tool printing without Rich for all modes with timestamps"""
        try:
            meta_tool_name = metadata.get("meta_tool_name", "unknown")
            execution_phase = metadata.get("execution_phase", "")
            reasoning_loop = metadata.get("reasoning_loop", "?")
            outline_step = metadata.get("outline_step", 0)
            time_str = timestamp.strftime("%H:%M:%S")

            if execution_phase == "meta_tool_start":
                if self.mode == VerbosityMode.MINIMAL:
                    # Only show important tools
                    if meta_tool_name in ["create_and_execute_plan", "direct_response", "advance_outline_step"]:
                        outline_info = f" (step {outline_step})" if outline_step > 0 else ""
                        print(f"🔧 [{time_str}] {meta_tool_name.replace('_', ' ').title()}{outline_info}")
                else:
                    outline_info = f" step:{outline_step}" if outline_step > 0 else ""
                    print(f"🔧 [{time_str}] Loop {reasoning_loop}{outline_info}: {meta_tool_name} starting...")

            elif event.success:
                duration = metadata.get("execution_duration", 0)
                duration_str = f" ({duration:.1f}s)" if duration > 0 else ""
                outline_info = f" step:{outline_step}" if outline_step > 0 else ""

                if self.mode == VerbosityMode.MINIMAL:
                    # Only show completion for important tools
                    if meta_tool_name == "direct_response":
                        print(f"✅ [{time_str}] Response generated{duration_str}")
                    elif meta_tool_name == "create_and_execute_plan":
                        goals_count = metadata.get("goals_count", 0)
                        print(f"✅ [{time_str}] Plan executed ({goals_count} goals){duration_str}")
                    elif meta_tool_name == "advance_outline_step":
                        if metadata.get("step_completed", False):
                            print(f"➡️ [{time_str}] Step advanced{duration_str}")
                else:
                    print(
                        f"✅ [{time_str}] Loop {reasoning_loop}{outline_info}: {meta_tool_name} completed{duration_str}")

                    # Show specific results based on tool type with enhanced info
                    if meta_tool_name == "manage_internal_task_stack":
                        action = metadata.get("stack_action", "")
                        stack_size = metadata.get("stack_size_after", 0)
                        outline_ref = metadata.get("outline_step_ref", "")
                        ref_info = f" ({outline_ref})" if outline_ref else ""
                        print(f"   Stack {action}: {stack_size} items{ref_info}")

                    elif meta_tool_name == "internal_reasoning":
                        thought_num = metadata.get("thought_number", "?")
                        total_thoughts = metadata.get("total_thoughts", "?")
                        focus = metadata.get("current_focus", "")[:50] + "..." if len(
                            metadata.get("current_focus", "")) > 50 else metadata.get("current_focus", "")
                        confidence = metadata.get("confidence_level", 0)
                        confidence_str = f" ({confidence:.1%})" if confidence > 0 else ""
                        print(f"   Thought {thought_num}/{total_thoughts}: {focus}{confidence_str}")

                    elif meta_tool_name == "create_and_execute_plan":
                        goals_count = metadata.get("goals_count", 0)
                        complexity = metadata.get("estimated_complexity", "")
                        complexity_str = f" ({complexity})" if complexity and complexity != "unknown" else ""
                        print(f"   Plan executed: {goals_count} goals{complexity_str}")

                    elif meta_tool_name == "delegate_to_llm_tool_node":
                        tools_count = metadata.get("tools_count", 0)
                        task_desc = metadata.get("delegated_task_description", "")
                        if task_desc:
                            task_brief = task_desc[:50] + "..." if len(task_desc) > 50 else task_desc
                            print(f"   Task: {task_brief} | {tools_count} tools used")
                        else:
                            print(f"   Delegation: {tools_count} tools used")

                    elif meta_tool_name == "advance_outline_step":
                        if metadata.get("step_completed", False):
                            evidence = metadata.get("completion_evidence", "")[:50] + "..." if len(
                                metadata.get("completion_evidence", "")) > 50 else metadata.get("completion_evidence",
                                                                                                "")
                            print(f"   Step completed: {evidence}")

                    elif meta_tool_name == "write_to_variables":
                        var_scope = metadata.get("variable_scope", "")
                        var_key = metadata.get("variable_key", "")
                        print(f"   Stored: {var_scope}.{var_key}")

                    elif meta_tool_name == "read_from_variables":
                        var_scope = metadata.get("variable_scope", "")
                        var_key = metadata.get("variable_key", "")
                        print(f"   Retrieved: {var_scope}.{var_key}")
            else:
                error = metadata.get("error_message", "Unknown error")
                if self.mode != VerbosityMode.MINIMAL:
                    outline_info = f" step:{outline_step}" if outline_step > 0 else ""
                    print(f"❌ [{time_str}] Loop {reasoning_loop}{outline_info}: {meta_tool_name} failed - {error}")

        except Exception as e:
            print(f"⚠️ Fallback meta-tool print error: {e}")

    def print_task_update_from_event(self, event: ProgressEvent):
        """Print task updates from events with automatic task detection"""
        try:
            # Check if this is a task-related event
            if not event.event_type.startswith('task_'):
                return

            # Extract task object from metadata
            if not event.metadata or 'task' not in event.metadata:
                return

            task_dict = event.metadata['task']

            self._print_task_update(event, task_dict)

        except Exception as e:
            if self.mode == VerbosityMode.DEBUG:
                print(f"⚠️ Error printing task update from event: {e}")
            import traceback
            print(traceback.format_exc())

    def _print_task_update(self, event: ProgressEvent, task_dict: Dict[str, Any]):
        """Print task update based on verbosity mode"""
        try:
            if self._fallback_mode or not self.use_rich:
                self._print_task_update_fallback(event, task_dict)
                return

            # Get task info
            task_id = task_dict.get('id', 'unknown')
            task_type = task_dict.get('type', 'Task')
            task_status = task_dict.get('status', 'unknown')
            task_description = task_dict.get('description', 'No description')

            # Status icon and color
            status_icon = self._get_task_status_icon_from_dict(task_dict)
            status_color = self._get_task_status_color_from_dict(task_dict)

            # Format based on verbosity mode and event type
            if self.mode == VerbosityMode.MINIMAL:
                self._print_minimal_task_update(event, task_dict, status_icon)

            elif self.mode == VerbosityMode.STANDARD:
                self._print_standard_task_update(event, task_dict, status_icon, status_color)

            elif self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                self._print_detailed_task_update(event, task_dict, status_icon, status_color)

            elif self.mode == VerbosityMode.REALTIME:
                if not self.realtime_minimal:
                    self._print_realtime_task_update(event, task_dict, status_icon)

        except Exception as e:
            self._consecutive_errors += 1
            if self._consecutive_errors <= self._error_threshold:
                print(f"⚠️ Task update print error: {e}")
            self._print_task_update_fallback(event, task_dict)

    def _print_minimal_task_update(self, event: ProgressEvent, task_dict: Dict[str, Any], status_icon: str):
        """Minimal task update - only status changes"""
        if event.event_type in ['task_start', 'task_complete', 'task_error']:
            task_id = task_dict.get('id', 'unknown')
            task_text = f"{status_icon} {task_id}"

            if event.event_type == 'task_error' and task_dict.get('error'):
                task_text += f" - {task_dict['error']}"

            self.console.print(task_text, style=self._get_task_status_color_from_dict(task_dict))

    def _print_standard_task_update(self, event: ProgressEvent, task_dict: Dict[str, Any], status_icon: str,
                                    status_color: str):
        """Standard task update with panels"""
        task_id = task_dict.get('id', 'unknown')
        task_description = task_dict.get('description', 'No description')

        # Create update message based on event type
        if event.event_type == 'task_start':
            title = f"🚀 Task Starting: {task_id}"
            content = f"{status_icon} {task_description}"

        elif event.event_type == 'task_complete':
            title = f"✅ Task Completed: {task_id}"
            content = f"{status_icon} {task_description}"

            # Add timing if available
            if task_dict.get('started_at') and task_dict.get('completed_at'):
                try:
                    start = datetime.fromisoformat(task_dict['started_at']) if isinstance(task_dict['started_at'],
                                                                                          str) else task_dict[
                        'started_at']
                    end = datetime.fromisoformat(task_dict['completed_at']) if isinstance(task_dict['completed_at'],
                                                                                          str) else task_dict[
                        'completed_at']
                    duration = (end - start).total_seconds()
                    content += f"\n⏱️ Duration: {duration:.1f}s"
                except:
                    pass

        elif event.event_type == 'task_error':
            title = f"❌ Task Failed: {task_id}"
            content = f"{status_icon} {task_description}"

            if task_dict.get('error'):
                content += f"\n🚨 Error: {task_dict['error']}"

            retry_count = task_dict.get('retry_count', 0)
            max_retries = task_dict.get('max_retries', 0)
            if retry_count > 0:
                content += f"\n🔄 Retries: {retry_count}/{max_retries}"

        elif event.event_type == 'task_updating':
            old_status = event.metadata.get('old_status', 'unknown')
            new_status = event.metadata.get('new_status', 'unknown')
            title = f"🔄 Task Update: {task_id}"
            content = f"{status_icon} {old_status} → {new_status}"
        else:
            return  # Don't print other task events in standard mode

        # Create and print panel
        panel = Panel(
            content,
            title=title,
            style=status_color,
            box=box.ROUNDED
        )
        self.console.print(panel)

    def _print_detailed_task_update(self, event: ProgressEvent, task_dict: Dict[str, Any], status_icon: str,
                                    status_color: str):
        """Detailed task update with full information"""
        task_id = task_dict.get('id', 'unknown')
        task_type = task_dict.get('type', 'Task')

        # Build comprehensive task info
        content_lines = []
        content_lines.append(f"{status_icon} Type: {task_type}")
        content_lines.append(f"📄 {task_dict.get('description', 'No description')}")

        # Dependencies
        if task_dict.get('dependencies'):
            content_lines.append(f"🔗 Dependencies: {', '.join(task_dict['dependencies'])}")

        # Priority
        if task_dict.get('priority', 1) != 1:
            content_lines.append(f"⭐ Priority: {task_dict['priority']}")

        # Task-specific details
        if task_type == 'ToolTask':
            if task_dict.get('tool_name'):
                content_lines.append(f"🔧 Tool: {task_dict['tool_name']}")
            if task_dict.get('arguments') and self.mode == VerbosityMode.DEBUG:
                args_str = str(task_dict['arguments'])[:80] + "..." if len(str(task_dict['arguments'])) > 80 else str(
                    task_dict['arguments'])
                content_lines.append(f"⚙️ Args: {args_str}")
            if task_dict.get('hypothesis'):
                content_lines.append(f"🔬 Hypothesis: {task_dict['hypothesis']}")

        elif task_type == 'LLMTask':
            if task_dict.get('llm_config'):
                model = task_dict['llm_config'].get('model_preference', 'default')
                temp = task_dict['llm_config'].get('temperature', 0.7)
                content_lines.append(f"🧠 Model: {model} (temp: {temp})")
            if task_dict.get('context_keys'):
                content_lines.append(f"🔑 Context: {', '.join(task_dict['context_keys'])}")

        elif task_type == 'DecisionTask':
            if task_dict.get('routing_map') and self.mode == VerbosityMode.DEBUG:
                routes = list(task_dict['routing_map'].keys())
                content_lines.append(f"🗺️ Routes: {routes}")

        # Timing information
        timing_info = []
        if task_dict.get('created_at'):
            timing_info.append(f"Created: {self._format_timestamp(task_dict['created_at'])}")
        if task_dict.get('started_at'):
            timing_info.append(f"Started: {self._format_timestamp(task_dict['started_at'])}")
        if task_dict.get('completed_at'):
            timing_info.append(f"Completed: {self._format_timestamp(task_dict['completed_at'])}")

        if timing_info:
            content_lines.append(f"📅 {' | '.join(timing_info)}")

        # Error information
        if task_dict.get('error'):
            content_lines.append(f"❌ Error: {task_dict['error']}")
            retry_count = task_dict.get('retry_count', 0)
            max_retries = task_dict.get('max_retries', 0)
            if retry_count > 0:
                content_lines.append(f"🔄 Retries: {retry_count}/{max_retries}")

        # Result preview (in debug mode)
        if self.mode == VerbosityMode.DEBUG and task_dict.get('result'):
            result_preview = str(task_dict['result'])[:100] + "..." if len(str(task_dict['result'])) > 100 else str(
                task_dict['result'])
            content_lines.append(f"📊 Result: {result_preview}")

        # Critical flag
        if task_dict.get('critical'):
            content_lines.append("🚨 CRITICAL TASK")

        # Create title based on event type
        event_titles = {
            'task_start': f"🔄 Running Task: {task_id}",
            'task_complete': f"✅ Completed Task: {task_id}",
            'task_error': f"❌ Failed Task: {task_id}",
            'task_updating': f"🔄 Updating Task: {task_id}"
        }
        title = event_titles.get(event.event_type, f"📋 Task Update: {task_id}")

        # Create and print panel
        panel = Panel(
            "\n".join(content_lines),
            title=title,
            style=status_color,
            box=box.ROUNDED
        )
        self.console.print(panel)

    def _print_realtime_task_update(self, event: ProgressEvent, task_dict: Dict[str, Any], status_icon: str):
        """Realtime task update - brief but informative"""
        if event.event_type in ['task_start', 'task_complete', 'task_error']:
            task_id = task_dict.get('id', 'unknown')
            task_desc = task_dict.get('description', '')[:50] + "..." if len(
                task_dict.get('description', '')) > 50 else task_dict.get('description', '')

            update_text = f"{status_icon} {task_id}: {task_desc}"

            if event.event_type == 'task_error' and task_dict.get('error'):
                update_text += f" ({task_dict['error']})"

            self.console.print(update_text, style=self._get_task_status_color_from_dict(task_dict))

    def _print_task_update_fallback(self, event: ProgressEvent, task_dict: Dict[str, Any]):
        """Fallback task update printing without Rich"""
        try:
            task_id = task_dict.get('id', 'unknown')
            task_type = task_dict.get('type', 'Task')
            task_status = task_dict.get('status', 'unknown')
            task_description = task_dict.get('description', 'No description')

            status_icon = self._get_task_status_icon_from_dict(task_dict)

            if event.event_type == 'task_start':
                print(f"\n🚀 TASK STARTING: {task_id}")
                print(f"{status_icon} {task_description}")

            elif event.event_type == 'task_complete':
                print(f"\n✅ TASK COMPLETED: {task_id}")
                print(f"{status_icon} {task_description}")

                if task_dict.get('started_at') and task_dict.get('completed_at'):
                    try:
                        start = datetime.fromisoformat(task_dict['started_at']) if isinstance(task_dict['started_at'],
                                                                                              str) else task_dict[
                            'started_at']
                        end = datetime.fromisoformat(task_dict['completed_at']) if isinstance(task_dict['completed_at'],
                                                                                              str) else task_dict[
                            'completed_at']
                        duration = (end - start).total_seconds()
                        print(f"⏱️ Duration: {duration:.1f}s")
                    except:
                        pass

            elif event.event_type == 'task_error':
                print(f"\n❌ TASK FAILED: {task_id}")
                print(f"{status_icon} {task_description}")
                if task_dict.get('error'):
                    print(f"🚨 Error: {task_dict['error']}")

            elif event.event_type == 'task_updating':
                old_status = event.metadata.get('old_status', 'unknown')
                new_status = event.metadata.get('new_status', 'unknown')
                print(f"\n🔄 TASK UPDATE: {task_id}")
                print(f"{status_icon} {old_status} → {new_status}")

            if self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                print(f"Type: {task_type} | Priority: {task_dict.get('priority', 1)}")
                if task_dict.get('dependencies'):
                    print(f"Dependencies: {', '.join(task_dict['dependencies'])}")

            print("-" * 50)

        except Exception as e:
            print(f"⚠️ Error in fallback task print: {e}")

    def _get_task_status_icon_from_dict(self, task_dict: Dict[str, Any]) -> str:
        """Get status icon from task dict"""
        status = task_dict.get('status', 'unknown')
        status_icons = {
            "pending": "⏳",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
            "paused": "⏸️"
        }
        return status_icons.get(status, "❓")

    def _get_task_status_color_from_dict(self, task_dict: Dict[str, Any]) -> str:
        """Get status color from task dict"""
        status = task_dict.get('status', 'unknown')
        status_colors = {
            "pending": "yellow",
            "running": "white bold",
            "completed": "green bold",
            "failed": "red bold",
            "paused": "orange3"
        }
        return status_colors.get(status, "white")

    def _format_timestamp(self, timestamp) -> str:
        """Format timestamp for display"""
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp)
            else:
                dt = timestamp
            return dt.strftime('%H:%M:%S')
        except:
            return str(timestamp)


    def _print_debug_event(self, event: ProgressEvent):
        """Print individual event details in debug mode"""
        timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S.%f")[:-3]
        if self.use_rich:
            debug_text = f"[{timestamp}] {event.event_type.upper()} - {event.node_name} ({json.dumps({k: v for k, v in asdict(event).items() if v is not None}, default=str, ensure_ascii=False)})"
            if event.success is not None:
                success_icon = "✅" if event.success else "❌"
                debug_text += f" {success_icon}"
            self.console.print(debug_text, style="dim")
        else:
            print(f"[{timestamp}] {event.event_type.upper()} - {event.node_name} ({json.dumps({k: v for k, v in asdict(event).items() if v is not None}, default=str, ensure_ascii=False)})")

    async def _noop_callback(self, event: ProgressEvent):
        """No-op callback when printing is disabled"""
        pass

    def print_final_summary(self):
        """Print comprehensive final summary"""
        try:
            if self._fallback_mode:
                self._print_final_summary_fallback()
                return

            if not self.use_rich:
                self._print_final_summary_fallback()
                return

            summary = self.tree_builder.get_execution_summary()

            # Final completion message
            self.console.print()
            self.console.print("🎉 [bold green]EXECUTION COMPLETED[/bold green] 🎉")

            # Final execution tree
            final_tree = self._create_execution_tree()
            self.console.print(final_tree)

            # Comprehensive summary table
            self._print_final_summary_table(summary)

            # Performance analysis
            if self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                self._print_performance_analysis(summary)

        except Exception as e:
            print(f"⚠️  Error printing final summary: {e}")
            self._print_final_summary_fallback()

    def _print_final_summary_table(self, summary: Dict[str, Any]):
        """Print detailed final summary table"""
        session_info = summary["session_info"]
        timing = summary["timing"]
        perf = summary["performance_metrics"]
        health = summary["health_indicators"]

        table = Table(title="📊 Final Execution Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", min_width=20)
        table.add_column("Value", style="green", min_width=15)
        table.add_column("Details", style="dim", min_width=25)

        # Session metrics
        table.add_row("Session ID", str(summary.get("session_id", "N/A")), "")
        table.add_row("Total Runtime", f"{timing['elapsed']:.2f}s", "")
        table.add_row("Nodes Processed", str(session_info["total_nodes"]),
                      f"{session_info['completed_nodes']} completed, {session_info['failed_nodes']} failed")

        # Performance metrics
        table.add_row("Total Events", str(perf["total_events"]),
                      f"{perf['total_events'] / max(timing['elapsed'], 1):.1f} events/sec")
        table.add_row("Routing Steps", str(perf["routing_steps"]), "")

        if perf["total_cost"] > 0:
            table.add_row("Total Cost", self._format_cost(perf["total_cost"]), "")
        if perf["total_tokens"] > 0:
            tokens_per_sec = perf["total_tokens"] / max(timing["elapsed"], 1)
            table.add_row("Total Tokens", f"{perf['total_tokens']:,}", f"{tokens_per_sec:.0f} tokens/sec")

        # Health metrics
        table.add_row("Overall Health", f"{health['overall_health']:.1%}", "")
        table.add_row("Error Rate", f"{health['error_rate']:.1%}", f"{perf['error_count']} total errors")
        table.add_row("Completion Rate", f"{health['completion_rate']:.1%}", "")
        table.add_row("Avg Efficiency", f"{health['average_node_efficiency']:.1%}", "")

        self.console.print()
        self.console.print(table)

    def _print_performance_analysis(self, summary: Dict[str, Any]):
        """Print detailed performance analysis"""
        analysis_panel = Panel(
            self._generate_performance_insights(summary),
            title="🔍 Performance Analysis",
            style="yellow"
        )
        self.console.print()
        self.console.print(analysis_panel)

    def _generate_performance_insights(self, summary: Dict[str, Any]) -> str:
        """Generate performance insights"""
        insights = []

        health = summary["health_indicators"]
        timing = summary["timing"]
        perf = summary["performance_metrics"]
        session_info = summary["session_info"]

        # Health insights
        if health["overall_health"] > 0.9:
            insights.append("✨ Excellent execution with minimal issues")
        elif health["overall_health"] > 0.7:
            insights.append("✅ Good execution with minor issues")
        elif health["overall_health"] > 0.5:
            insights.append("⚠️ Moderate execution with some failures")
        else:
            insights.append("❌ Poor execution with significant issues")

        # Performance insights
        if timing["elapsed"] > 0:
            events_per_sec = perf["total_events"] / timing["elapsed"]
            if events_per_sec > 10:
                insights.append(f"⚡ High event processing rate: {events_per_sec:.1f}/sec")
            elif events_per_sec < 2:
                insights.append(f"🐌 Low event processing rate: {events_per_sec:.1f}/sec")

        # Error insights
        if perf["error_count"] == 0:
            insights.append("🎯 Zero errors - perfect execution")
        elif health["error_rate"] < 0.1:
            insights.append(f"✅ Low error rate: {health['error_rate']:.1%}")
        else:
            insights.append(f"⚠️ High error rate: {health['error_rate']:.1%} - review failed operations")

        # Cost insights
        if perf["total_cost"] > 0:
            cost_per_node = perf["total_cost"] / max(session_info["total_nodes"], 1)
            if cost_per_node < 0.001:
                insights.append(f"💚 Very cost-efficient: {self._format_cost(cost_per_node)}/node")
            elif cost_per_node > 0.01:
                insights.append(f"💸 High cost per node: {self._format_cost(cost_per_node)}/node")

        # Node efficiency insights
        if health["average_node_efficiency"] > 0.8:
            insights.append("🚀 High node efficiency - well-optimized execution")
        elif health["average_node_efficiency"] < 0.5:
            insights.append("🔧 Low node efficiency - consider optimization")

        return "\n".join(f"• {insight}" for insight in insights)

    def _print_final_summary_fallback(self):
        """Fallback final summary without Rich"""
        summary = self.tree_builder.get_execution_summary()
        session_info = summary["session_info"]
        timing = summary["timing"]
        perf = summary["performance_metrics"]
        health = summary["health_indicators"]

        print(f"\n{'=' * 80}")
        print("🎉 EXECUTION COMPLETED 🎉")
        print(f"{'=' * 80}")

        print(f"Session ID: {summary.get('session_id', 'N/A')}")
        print(f"Total Runtime: {timing['elapsed']:.2f}s")
        print(f"Nodes: {session_info['completed_nodes']}/{session_info['total_nodes']} completed")
        print(f"Events: {perf['total_events']}")
        print(f"Errors: {perf['error_count']}")
        print(f"Overall Health: {health['overall_health']:.1%}")

        if perf["total_cost"] > 0:
            print(f"Total Cost: {self._format_cost(perf['total_cost'])}")
        if perf["total_tokens"] > 0:
            print(f"Total Tokens: {perf['total_tokens']:,}")

        print(f"{'=' * 80}")

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get complete execution log for analysis"""
        return self.print_history.copy()

    def export_summary(self, filepath: str = None) -> Dict[str, Any]:
        """Export comprehensive execution summary"""
        summary = self.tree_builder.get_execution_summary()

        # Add detailed node information
        summary["detailed_nodes"] = {}
        for node_name, node in self.tree_builder.nodes.items():
            summary["detailed_nodes"][node_name] = {
                "status": node.status.value,
                "duration": node.duration,
                "start_time": node.start_time,
                "end_time": node.end_time,
                "total_cost": node.total_cost,
                "total_tokens": node.total_tokens,
                "llm_calls": len(node.llm_calls),
                "tool_calls": len(node.tool_calls),
                "error": node.error,
                "retry_count": node.retry_count,
                "performance_metrics": node.get_performance_summary()
            }

        # Add execution history
        summary["execution_history"] = self.print_history.copy()
        summary["error_log"] = self.tree_builder.error_log.copy()
        summary["routing_history"] = self.tree_builder.routing_history.copy()

        # Export to file if specified
        if filepath:
            import json
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

        return summary


# Demo and testing functions
async def demo_enhanced_printer():
    """Comprehensive demo of the enhanced progress printer showcasing all modes"""
    import asyncio

    print("🚀 Starting Enhanced Progress Printer Demo...")
    print("Choose demo type:")
    print("1. All Modes Demo - Show all verbosity modes with same scenario")
    print("2. Interactive Mode Selection - Choose specific mode")
    print("3. Strategy Selection Demo - Show strategy printing")
    print("4. Accumulated Runs Demo - Show multi-run accumulation")
    print("5. Complete Feature Demo - All features in sequence")
    print("6. Exit")

    try:
        choice = input("Enter choice (1-6) [default: 1]: ").strip() or "1"
    except:
        choice = "1"

    if choice == "6":
        return
    elif choice == "1":
        await demo_all_modes()
    elif choice == "2":
        await demo_interactive_mode()
    elif choice == "3":
        await demo_strategy_selection()
    elif choice == "4":
        await demo_accumulated_runs()
    elif choice == "5":
        await demo_complete_features()


async def demo_all_modes():
    """Demo all verbosity modes with the same scenario"""
    print("\n🎭 ALL MODES DEMONSTRATION")
    print("=" * 50)
    print("This demo will run the same scenario in all verbosity modes")
    print("to show the differences in output detail.")

    modes = [
        (VerbosityMode.MINIMAL, "MINIMAL - Only major updates"),
        (VerbosityMode.STANDARD, "STANDARD - Regular updates with panels"),
        (VerbosityMode.VERBOSE, "VERBOSE - Detailed information with metrics"),
        (VerbosityMode.DEBUG, "DEBUG - Full debugging info with all details"),
        (VerbosityMode.REALTIME, "REALTIME - Live updates (will show final tree)")
    ]

    for mode, description in modes:
        print(f"\n{'=' * 60}")
        print(f"🎯 NOW DEMONSTRATING: {description}")
        print(f"{'=' * 60}")

        await asyncio.sleep(2)

        printer = ProgressiveTreePrinter(mode=mode, realtime_minimal=False)

        # Strategy selection demo
        printer.print_strategy_selection(
            "research_and_analyze",
            context={
                "reasoning": "Complex query requires multi-source research and analysis",
                "complexity_score": 0.8,
                "estimated_steps": 5
            }
        )

        await asyncio.sleep(1)

        # Run scenario
        events = await create_demo_scenario()

        for event in events:
            await printer.progress_callback(event)
            if mode == VerbosityMode.REALTIME:
                await asyncio.sleep(0.5)
            else:
                await asyncio.sleep(0.3)

        # Final summary
        printer.print_final_summary()

        if mode != modes[-1][0]:  # Not the last mode
            input(f"\n⏸️  Press Enter to continue to next mode...")


async def demo_interactive_mode():
    """Interactive mode selection demo"""
    print("\n🎮 INTERACTIVE MODE SELECTION")
    print("Choose your preferred verbosity mode:")
    print("1. MINIMAL - Only major updates")
    print("2. STANDARD - Regular updates")
    print("3. VERBOSE - Detailed information")
    print("4. DEBUG - Full debugging info")
    print("5. REALTIME - Live updates")

    try:
        choice = input("Enter choice (1-5) [default: 2]: ").strip() or "2"
        modes = {
            "1": VerbosityMode.MINIMAL,
            "2": VerbosityMode.STANDARD,
            "3": VerbosityMode.VERBOSE,
            "4": VerbosityMode.DEBUG,
            "5": VerbosityMode.REALTIME
        }
        mode = modes.get(choice, VerbosityMode.STANDARD)
    except:
        mode = VerbosityMode.STANDARD

    printer = ProgressiveTreePrinter(mode=mode)
    print(f"\n🎯 Running demo in {mode.value.upper()} mode...")

    # Strategy selection
    printer.print_strategy_selection("slow_complex_planning", context={
        "reasoning": "Task has multiple 'and' conditions requiring complex breakdown",
        "complexity_score": 0.9,
        "estimated_steps": 8
    })

    await asyncio.sleep(1)

    events = await create_demo_scenario()
    for event in events:
        await printer.progress_callback(event)
        await asyncio.sleep(0.5 if mode == VerbosityMode.REALTIME else 0.8)

    printer.print_final_summary()


async def demo_strategy_selection():
    """Demo all strategy selection options"""
    print("\n🎯 STRATEGY SELECTION DEMONSTRATION")
    print("=" * 50)

    strategies = [
        ("direct_response", "Simple question that needs direct answer"),
        ("fast_simple_planning", "Task needs quick multi-step approach"),
        ("slow_complex_planning", "Complex task with multiple 'and' conditions"),
        ("research_and_analyze", "Needs information gathering and analysis"),
        ("creative_generation", "Content creation with personalization"),
        ("problem_solving", "Analysis with validation required")
    ]

    for mode in [VerbosityMode.MINIMAL, VerbosityMode.STANDARD, VerbosityMode.VERBOSE]:
        print(f"\n🔍 Strategy demo in {mode.value.upper()} mode:")
        print("-" * 40)

        printer = ProgressiveTreePrinter(mode=mode)

        for strategy, reasoning in strategies:
            complexity = 0.3 if "simple" in strategy else 0.7 if "complex" in strategy else 0.5

            printer.print_strategy_selection(
                strategy,
                context={
                    "reasoning": reasoning,
                    "complexity_score": complexity,
                    "estimated_steps": 1 if "direct" in strategy else 3 if "fast" in strategy else 6
                }
            )
            await asyncio.sleep(0.8)

        if mode != VerbosityMode.VERBOSE:
            input("\n⏸️  Press Enter for next mode...")


async def demo_accumulated_runs():
    """Demo accumulated runs functionality"""
    print("\n📊 ACCUMULATED RUNS DEMONSTRATION")
    print("=" * 50)
    print("This demo shows how multiple execution runs are accumulated and analyzed")

    printer = ProgressiveTreePrinter(mode=VerbosityMode.STANDARD)

    # Simulate 3 different runs
    runs = [
        ("Market Analysis", "research_and_analyze", True, 12.5, 0.045),
        ("Content Creation", "creative_generation", True, 8.2, 0.032),
        ("Problem Solving", "problem_solving", False, 15.8, 0.067)  # This one fails
    ]

    for i, (run_name, strategy, success, duration, cost) in enumerate(runs):
        print(f"\n🏃 Running execution {i + 1}/3: {run_name}")

        # Strategy selection
        printer.print_strategy_selection(strategy)
        await asyncio.sleep(1)

        # Quick execution simulation
        events = await create_demo_scenario(
            run_name=run_name,
            duration=duration,
            cost=cost,
            should_fail=not success
        )

        for event in events:
            await printer.progress_callback(event)
            await asyncio.sleep(0.2)  # Fast execution

        # Flush the run
        printer.flush(run_name)
        await asyncio.sleep(2)

    # Show accumulated summary
    print("\n📈 ACCUMULATED SUMMARY:")
    printer.print_accumulated_summary()

    # Export data
    if input("\n💾 Export accumulated data? (y/n): ").lower().startswith('y'):
        filepath = printer.export_accumulated_data()
        print(f"✅ Data exported to: {filepath}")


async def demo_complete_features():
    """Complete feature demonstration"""
    print("\n🚀 COMPLETE FEATURE DEMONSTRATION")
    print("=" * 50)
    print("This demo showcases all features in a comprehensive scenario")

    # Start with verbose mode
    printer = ProgressiveTreePrinter(mode=VerbosityMode.VERBOSE)

    print("\n1️⃣ STRATEGY SELECTION SHOWCASE:")
    strategies = ["direct_response", "research_and_analyze", "problem_solving"]
    for strategy in strategies:
        printer.print_strategy_selection(strategy, context={
            "reasoning": f"Demonstrating {strategy} strategy selection",
            "complexity_score": 0.6,
            "estimated_steps": 4
        })
        await asyncio.sleep(1)

    print("\n2️⃣ COMPLEX EXECUTION WITH ERRORS:")
    # Complex scenario with multiple nodes, errors, and recovery
    complex_events = await create_complex_scenario()

    for event in complex_events:
        await printer.progress_callback(event)
        await asyncio.sleep(0.4)

    printer.print_final_summary()

    print("\n3️⃣ MODE COMPARISON:")
    print("Switching to REALTIME mode for live demo...")
    await asyncio.sleep(2)

    # Switch to realtime mode
    realtime_printer = ProgressiveTreePrinter(
        mode=VerbosityMode.REALTIME,
        realtime_minimal=True
    )

    print("Running same scenario in REALTIME minimal mode:")
    simple_events = await create_demo_scenario()

    for event in simple_events:
        await realtime_printer.progress_callback(event)
        await asyncio.sleep(0.3)

    print("\n\n4️⃣ ACCUMULATED ANALYTICS:")
    # Flush both runs
    printer.flush("Complex Execution")
    realtime_printer.flush("Realtime Execution")

    # Transfer accumulated data to one printer for summary
    printer._accumulated_runs.extend(realtime_printer._accumulated_runs)
    printer.print_accumulated_summary()




async def create_demo_scenario(run_name="Demo Run", duration=10.0, cost=0.025, should_fail=False):
    """Create a demo scenario with configurable parameters"""
    base_time = time.time()
    events = []

    # Execution start
    events.append(ProgressEvent(
        event_type="execution_start",
        timestamp=base_time,
        node_name="FlowAgent",
        session_id=f"demo_session_{int(base_time)}",
        metadata={"query": f"Execute {run_name}", "user_id": "demo_user"}
    ))

    # Strategy orchestrator
    events.append(ProgressEvent(
        event_type="node_enter",
        timestamp=base_time + 0.1,
        node_name="StrategyOrchestratorNode"
    ))

    events.append(ProgressEvent(
        event_type="llm_call",
        timestamp=base_time + 1.2,
        node_name="StrategyOrchestratorNode",
        llm_model="gpt-4",
        llm_total_tokens=1200,
        llm_cost=cost * 0.4,
        llm_duration=1.1,
        success=True,
        metadata={"strategy": "research_and_analyze"}
    ))

    # Planning
    events.append(ProgressEvent(
        event_type="node_enter",
        timestamp=base_time + 2.5,
        node_name="PlannerNode"
    ))

    events.append(ProgressEvent(
        event_type="llm_call",
        timestamp=base_time + 3.8,
        node_name="PlannerNode",
        llm_model="gpt-3.5-turbo",
        llm_total_tokens=800,
        llm_cost=cost * 0.2,
        llm_duration=1.3,
        success=True
    ))
    # TaskPlan
    events.append(ProgressEvent(
        event_type="plan_created",
        timestamp=base_time + 4.0,
        node_name="PlannerNode",
        status=NodeStatus.COMPLETED,
        success=True,
        metadata={"plan_name": "Demo Plan", "task_count": 3, "full_plan": TaskPlan(id='bf5053ad-1eae-4dd2-9c08-0c7fab49f80d', name='File Cleanup Task', description='Remove turtle_on_bike.py and execution_summary.json if they exist', tasks=[LLMTask(id='analyze_files', type='LLMTask', description='Analyze the current directory for turtle_on_bike.py and execution_summary.json', status='pending', priority=1, dependencies=[], subtasks=[], result=None, error=None, created_at=datetime(2025, 8, 13, 23, 51, 38, 726320), started_at=None, completed_at=None, metadata={}),ToolTask(id='remove_files', type='ToolTask', description='Delete turtle_on_bike.py and execution_summary.json using shell command', status='pending', priority=1, dependencies=[], subtasks=[], result=None, error=None, created_at=datetime(2025, 8, 13, 23, 51, 38, 726320), started_at=None, completed_at=None, metadata={}, retry_count=0, max_retries=3, critical=False, tool_name='shell', arguments={'command': "Remove-Item -Path 'turtle_on_bike.py', 'execution_summary.json' -ErrorAction SilentlyContinue"}, hypothesis='', validation_criteria='', expectation='')], status='created', created_at=datetime(2025, 8, 13, 23, 51, 38, 726320), metadata={}, execution_strategy='sequential')}
    ))

    # Execution with tools
    events.append(ProgressEvent(
        event_type="node_enter",
        timestamp=base_time + 5.0,
        node_name="ExecutorNode"
    ))

    events.append(ProgressEvent(
        event_type="tool_call",
        timestamp=base_time + 6.2,
        node_name="ExecutorNode",
        tool_name="web_search",
        tool_duration=2.1,
        tool_success=not should_fail,
        tool_result="Search completed" if not should_fail else None,
        tool_error="Search failed" if should_fail else None,
        success=not should_fail,
        metadata={"error": "Search API timeout"} if should_fail else {}
    ))

    if not should_fail:
        # Analysis
        events.append(ProgressEvent(
            event_type="llm_call",
            timestamp=base_time + 8.5,
            node_name="AnalysisNode",
            llm_model="gpt-4",
            llm_total_tokens=1500,
            llm_cost=cost * 0.4,
            llm_duration=2.3,
            success=True
        ))

        # Completion
        events.append(ProgressEvent(
            event_type="execution_complete",
            timestamp=base_time + duration,
            node_name="FlowAgent",
            node_duration=duration,
            status=NodeStatus.COMPLETED,
            success=True,
            metadata={"result": "Successfully completed"}
        ))
    else:
        # Failed completion
        events.append(ProgressEvent(
            event_type="error",
            timestamp=base_time + duration * 0.7,
            node_name="ExecutorNode",
            status=NodeStatus.FAILED,
            success=False,
            metadata={
                "error": "Execution failed due to tool error",
                "error_type": "ToolError"
            }
        ))

    return events


async def create_complex_scenario():
    """Create a complex scenario with multiple nodes and error recovery"""
    base_time = time.time()
    events = []

    nodes = [
        "FlowAgent",
        "StrategyOrchestratorNode",
        "TaskPlannerFlow",
        "ResearchNode",
        "AnalysisNode",
        "ValidationNode",
        "ResponseGeneratorNode"
    ]

    # Start execution
    events.append(ProgressEvent(
        event_type="execution_start",
        timestamp=base_time,
        node_name="FlowAgent",
        session_id=f"complex_session_{int(base_time)}",
        metadata={"complexity": "high", "estimated_duration": 25}
    ))

    current_time = base_time

    for i, node in enumerate(nodes[1:], 1):
        # Node entry
        current_time += 0.5
        events.append(ProgressEvent(
            event_type="node_enter",
            timestamp=current_time,
            node_name=node
        ))

        # Main operation (LLM or tool call)
        current_time += 1.2
        if i % 3 == 0:  # Tool call
            success = i != 5  # Fail on ValidationNode
            events.append(ProgressEvent(
                event_type="tool_call",
                timestamp=current_time,
                node_name=node,
                tool_name=f"tool_{i}",
                tool_duration=1.8,
                tool_success=success,
                tool_result=f"Tool result {i}" if success else None,
                tool_error=f"Tool error {i}" if not success else None,
                success=success,
                metadata={"error": "Validation failed", "error_type": "ValidationError"} if not success else {}
            ))

            # Recovery if failed
            if not success:
                current_time += 2.0
                events.append(ProgressEvent(
                    event_type="tool_call",
                    timestamp=current_time,
                    node_name=node,
                    tool_name="recovery_tool",
                    tool_duration=1.5,
                    tool_success=True,
                    tool_result="Recovery successful"
                ))
        else:  # LLM call
            events.append(ProgressEvent(
                event_type="llm_call",
                timestamp=current_time,
                node_name=node,
                llm_model="gpt-4" if i % 2 == 0 else "gpt-3.5-turbo",
                llm_total_tokens=1200 + i * 200,
                llm_cost=0.024 + i * 0.005,
                llm_duration=1.5 + i * 0.3,
                success=True
            ))

        # Node completion
        current_time += 0.8
        if node.endswith("Node"):  # Simple nodes auto-complete
            events.append(ProgressEvent(
                event_type="node_phase",
                timestamp=current_time,
                node_name=node,
                success=True,
                node_duration=current_time - (base_time + i * 2.5)
            ))

    # Final completion
    events.append(ProgressEvent(
        event_type="execution_complete",
        timestamp=current_time + 1.0,
        node_name="FlowAgent",
        node_duration=current_time + 1.0 - base_time,
        status=NodeStatus.COMPLETED,
        success=True,
        metadata={"total_cost": 0.156, "total_tokens": 12500}
    ))

    return events


if __name__ == "__main__":
    print("🔧 Enhanced CLI Progress Printing System")
    print("=" * 50)

    # Run the enhanced demo
    import asyncio

    try:
        asyncio.run(demo_enhanced_printer())
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
