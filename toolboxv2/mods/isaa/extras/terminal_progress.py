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
    """Production-ready progressive tree printer with live terminal updates and intelligent display management"""

    def __init__(self, mode: VerbosityMode = VerbosityMode.STANDARD, use_rich: bool = True,
                 auto_refresh: bool = True, max_history: int = 1000,
                 realtime_minimal: bool = None, auto_manage_display: bool = True,
                 ):
        self.prompt_app = None
        self.mode = mode
        self.agent_name = "FlowAgent"
        self.use_rich = use_rich and RICH_AVAILABLE
        self.auto_refresh = auto_refresh
        self.max_history = max_history
        self.auto_manage_display = auto_manage_display
        self._layout_integration_attempted = False

        self.tree_builder = ExecutionTreeBuilder()
        self.print_history: List[Dict[str, Any]] = []

        # Live display configuration
        self.realtime_minimal = realtime_minimal if realtime_minimal is not None else (mode == VerbosityMode.REALTIME)
        self._last_summary = ""
        self._needs_full_tree = False
        self._spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._spinner_index = 0
        self._last_display_lines = 20

        # Auto-management state
        self._display_active = False
        self._last_activity_time = time.time()
        self._activity_timeout = 30.0  # Stop live display after 30s inactivity
        self._min_display_interval = 0.5  # Minimum time between updates

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

    # In ProgressiveTreePrinter class - neue Methode hinzufügen

    # In ProgressiveTreePrinter class - NEUE METHODEN HINZUFÜGEN

    def _detect_and_integrate_prompt_toolkit(self) -> Optional[Dict[str, Any]]:
        """
        Safe prompt_toolkit detection and integration with proper error handling.
        ERSETZT: Die bestehende _detect_and_integrate_prompt_toolkit Methode
        """
        try:
            from prompt_toolkit.application import get_app
            from prompt_toolkit.widgets import TextArea
            from prompt_toolkit.layout.containers import Window, HSplit, VSplit
            from prompt_toolkit.layout.controls import FormattedTextControl
            from prompt_toolkit.formatted_text import HTML
            from prompt_toolkit.layout.dimension import D

            app = self.prompt_app or get_app()
            if not app or not app.is_running:
                return None

            # Create status control (single line) - this is safer than TextArea
            if not hasattr(self, '_agent_status_control'):
                self._agent_status_control = FormattedTextControl(
                    text=HTML("🤖 <cyan>Agent Ready</cyan>")
                )

                # Wrap in proper Window container with fixed height
                self._agent_status_window = Window(
                    content=self._agent_status_control,
                    height=D.exact(1),  # Exact 1 line height
                    dont_extend_height=True,
                    wrap_lines=False
                )

            # For multi-line output, use a simpler approach
            if not hasattr(self, '_agent_output_control'):
                self._agent_output_lines = ["🤖 Agent Status: Ready"]  # Store lines as list
                self._agent_output_control = FormattedTextControl(
                    text=lambda: self._get_formatted_output_text()
                )

                self._agent_output_window = Window(
                    content=self._agent_output_control,
                    height=D.max(6),  # Maximum 6 lines
                    wrap_lines=True
                )

            return {
                'app': app,
                'status_control': self._agent_status_control,
                'status_window': self._agent_status_window,
                'output_control': self._agent_output_control,
                'output_window': self._agent_output_window,
                'integration_type': 'control',  # Using controls instead of widgets
                'layout_integration': False  # Will be set to True if successful
            }

        except ImportError:
            return None
        except Exception as e:
            if self.mode == VerbosityMode.DEBUG:
                print(f"⚠️ Prompt toolkit integration failed: {e}")
            return None

    def _get_formatted_output_text(self) -> str:
        """
        Returns formatted text for the output control.
        NEUE METHODE
        """
        try:
            if not hasattr(self, '_agent_output_lines'):
                return "🤖 Agent Ready"

            # Join lines and return as formatted text
            return '\n'.join(self._agent_output_lines[-6:])  # Last 6 lines only

        except Exception:
            return "🤖 Agent Status"

    def _integrate_with_application_layout(self, integration_info: Dict[str, Any]) -> bool:
        """
        Safe layout integration with multiple fallback strategies.
        ERSETZT: Die bestehende _integrate_with_application_layout Methode
        """
        try:
            app = integration_info['app']
            status_window = integration_info['status_window']
            output_window = integration_info['output_window']

            # Strategy 1: Try to modify existing layout
            if hasattr(app, 'layout') and app.layout:
                try:
                    current_layout = app.layout

                    if hasattr(current_layout, 'container'):
                        root_container = current_layout.container

                        # Check if it's HSplit and we can safely add
                        if (hasattr(root_container, 'children') and
                            hasattr(root_container, '__class__') and
                            'HSplit' in str(root_container.__class__)):

                            # Add status window if not already present
                            if status_window not in root_container.children:
                                root_container.children.append(status_window)

                            # For verbose modes, add output window too
                            if (self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG] and
                                output_window not in root_container.children):
                                root_container.children.append(output_window)

                            integration_info['layout_integration'] = True
                            return True

                except Exception as e:
                    if self.mode == VerbosityMode.DEBUG:
                        print(f"⚠️ Layout integration attempt 1 failed: {e}")

            # Strategy 2: Use invalidation-based updates (safer fallback)
            integration_info['layout_integration'] = False
            return True  # We can still use the controls, just not integrated in layout

        except Exception as e:
            if self.mode == VerbosityMode.DEBUG:
                print(f"⚠️ All layout integration attempts failed: {e}")
            return False

    def _update_prompt_toolkit_widget(self, integration_info: Dict[str, Any], content: str,
                                      is_status_line: bool = False):
        """
        Safe widget updates with proper error handling.
        ERSETZT: Die bestehende _update_prompt_toolkit_widget Methode
        """
        try:
            app = integration_info['app']

            if is_status_line:
                # Update status line control
                status_control = integration_info['status_control']
                from prompt_toolkit.formatted_text import HTML

                # Clean content for HTML safety
                safe_content = content.replace('<', '&lt;').replace('>', '&gt;')
                # Re-add our HTML tags
                safe_content = safe_content.replace('&lt;cyan&gt;', '<cyan>').replace('&lt;/cyan&gt;', '</cyan>')
                safe_content = safe_content.replace('&lt;blue&gt;', '<blue>').replace('&lt;/blue&gt;', '</blue>')
                safe_content = safe_content.replace('&lt;green&gt;', '<green>').replace('&lt;/green&gt;', '</green>')
                safe_content = safe_content.replace('&lt;yellow&gt;', '<yellow>').replace('&lt;/yellow&gt;',
                                                                                          '</yellow>')
                safe_content = safe_content.replace('&lt;red&gt;', '<red>').replace('&lt;/red&gt;', '</red>')

                status_control.text = HTML(safe_content)

            else:
                # Update output lines
                if not hasattr(self, '_agent_output_lines'):
                    self._agent_output_lines = []

                # Add new content line
                timestamp = time.strftime("%H:%M:%S")
                formatted_line = f"[{timestamp}] {content}"

                self._agent_output_lines.append(formatted_line)

                # Keep only recent lines
                max_lines = 20 if self.mode == VerbosityMode.MINIMAL else 50
                if len(self._agent_output_lines) > max_lines:
                    self._agent_output_lines = self._agent_output_lines[-max_lines:]

            # Trigger UI refresh - safe approach
            try:
                if hasattr(app, 'invalidate'):
                    app.invalidate()
            except Exception:
                # If invalidate fails, try alternative refresh
                pass

        except Exception as e:
            if self.mode == VerbosityMode.DEBUG:
                print(f"⚠️ Widget update failed, falling back: {e}")
            # Fallback: Just print to console
            if is_status_line:
                print(f"🤖 {content}")
            else:
                print(f"📊 {content}")

    # ============================================================================
    # AUTO-MANAGEMENT METHODS (NEW)
    # ============================================================================

    def _should_start_live_display(self) -> bool:
        """Intelligently determines when to start live display"""
        if not self.auto_manage_display:
            return True

        # Start conditions
        return (
            self.tree_builder.total_events > 2 and  # Have some activity
            len(self.tree_builder.active_nodes) > 0 and  # Has active nodes
            time.time() - self.tree_builder.start_time < 300  # Within 5 minutes
        )

    def _should_stop_live_display(self) -> bool:
        """Intelligently determines when to stop live display"""
        if not self.auto_manage_display:
            return False

        # Stop conditions
        return (
            len(self.tree_builder.active_nodes) == 0 and  # No active nodes
            time.time() - self._last_activity_time > self._activity_timeout  # Inactivity timeout
        ) or (
            self.tree_builder.total_events > 100 and  # Lots of events
            self._consecutive_errors > 3  # Multiple errors
        )

    def _manage_display_lifecycle(self):
        """Manages automatic start/stop of live display"""
        try:
            # Check if we should start live display
            if not self._display_active and self._should_start_live_display():
                self._start_live_display()

            # Check if we should stop live display
            elif self._display_active and self._should_stop_live_display():
                self._stop_live_display()

        except Exception as e:
            if self.mode == VerbosityMode.DEBUG:
                print(f"⚠️ Display lifecycle management error: {e}")

    def _start_live_display(self):
        """Start live display mode"""
        self._display_active = True
        self._needs_full_tree = True
        if self.use_rich and self.mode != VerbosityMode.MINIMAL:
            self.console.print("🚀 [cyan]Live display started[/cyan]", style="dim")

    def _stop_live_display(self):
        """Stop live display and show final summary"""
        self._display_active = False
        if self.use_rich and self.mode != VerbosityMode.MINIMAL:
            self.console.print("⏹️ [yellow]Live display stopped - showing final state[/yellow]", style="dim")
        self._print_fallback()  # One final display

    def _add_live_error_section(self, tree: Tree):
        """
        Add live error section with enhanced error information and recovery suggestions.
        Shows recent errors with context, timing, and potential solutions.
        """
        try:
            if not self.tree_builder.error_log:
                return

            # Get recent errors (last 10, or last 30 seconds)
            current_time = time.time()
            recent_errors = []

            for error in self.tree_builder.error_log:
                error_age = current_time - error.get("timestamp", 0)

                # Include errors from last 30 seconds, or last 10 errors max
                if error_age < 30 or len(recent_errors) < 10:
                    recent_errors.append({
                        **error,
                        "age": error_age
                    })

            # Sort by timestamp (most recent first)
            recent_errors.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            recent_errors = recent_errors[:8]  # Show max 8 errors

            if not recent_errors:
                return

            error_count = len(recent_errors)
            critical_errors = [e for e in recent_errors if
                               'critical' in e.get('error', '').lower() or 'fatal' in e.get('error', '').lower()]

            # Section title with severity indication
            if critical_errors:
                error_title = f"🚨 Critical Errors ({len(critical_errors)}/{error_count})"
                error_style = "red bold"
            elif error_count > 5:
                error_title = f"⚠️ Recent Errors ({error_count})"
                error_style = "red"
            else:
                error_title = f"❌ Error Log ({error_count})"
                error_style = "red dim"

            error_branch = tree.add(error_title, style=error_style)

            # Add errors based on verbosity mode
            for error in recent_errors[:5]:  # Show up to 5 errors
                timestamp = datetime.fromtimestamp(error["timestamp"]).strftime("%H:%M:%S")
                node_name = error.get("node", "Unknown")
                error_message = error.get("error", "Unknown error")
                age = error.get("age", 0)

                # Age indicator
                if age < 10:
                    age_indicator = "🔥"  # Very recent
                elif age < 30:
                    age_indicator = "⚠️"  # Recent
                else:
                    age_indicator = "📝"  # Older

                # Truncate long error messages
                if len(error_message) > 80:
                    error_message = error_message[:77] + "..."

                error_text = f"{age_indicator} [{timestamp}] {node_name}: {error_message}"

                error_item = error_branch.add(error_text, style="red")

                # Add context if available
                if error.get("task_id"):
                    error_item.add(f"📋 Task: {error['task_id']}", style="red dim")
                elif error.get("tool_name"):
                    error_item.add(f"🔧 Tool: {error['tool_name']}", style="red dim")

        except Exception as e:
            # Fallback error display
            try:
                fallback_branch = tree.add("⚠️ Error Log (display error)", style="red dim")
                fallback_branch.add(f"Error displaying errors: {str(e)[:50]}...", style="red dim")
            except:
                pass  # If even fallback fails, silently continue

    def _get_task_executor_progress(self) -> Dict[str, Any]:
        """
        Extract comprehensive task execution progress from TaskExecutorNode events.
        Tracks parallel execution, dependencies, and completion status.
        """
        try:
            task_progress = {
                'total_tasks': 0,
                'completed_tasks': 0,
                'failed_tasks': 0,
                'running_tasks': [],
                'waiting_tasks': [],
                'parallel_executing': [],
                'execution_groups': [],
                'current_strategy': 'unknown',
                'execution_duration': 0,
                'task_details': {},
                'dependency_chains': [],
                'performance_metrics': {}
            }

            # Look for TaskExecutorNode events
            task_events = []
            for node in self.tree_builder.nodes.values():
                if 'TaskExecutor' in node.name:
                    task_events.extend(node.llm_calls)
                    task_events.extend(node.sub_events)

            if not task_events:
                return task_progress

            # Sort by timestamp
            task_events.sort(key=lambda x: x.timestamp)

            # Extract task information from events
            for event in task_events:
                if not hasattr(event, 'metadata') or not event.metadata:
                    continue

                metadata = event.metadata

                # Track individual task progress
                if event.event_type in ['task_start', 'task_complete', 'task_error']:
                    task_data = metadata.get('task', {})
                    if task_data:
                        task_id = task_data.get('id', 'unknown')
                        task_progress['task_details'][task_id] = {
                            'id': task_id,
                            'description': task_data.get('description', ''),
                            'status': task_data.get('status', 'unknown'),
                            'type': task_data.get('type', 'unknown'),
                            'priority': task_data.get('priority', 1),
                            'dependencies': task_data.get('dependencies', []),
                            'started_at': task_data.get('started_at'),
                            'completed_at': task_data.get('completed_at'),
                            'error': task_data.get('error'),
                            'last_event': event.event_type,
                            'timestamp': event.timestamp
                        }

                # Track execution plan and strategy
                if 'execution_plan' in metadata:
                    plan = metadata['execution_plan']
                    task_progress['current_strategy'] = plan.get('strategy', 'unknown')
                    task_progress['execution_groups'] = plan.get('execution_groups', [])

                # Track performance metrics
                if event.event_type == 'task_complete' and 'execution_duration' in metadata:
                    task_progress['execution_duration'] += metadata['execution_duration']

            # Aggregate task status
            for task_id, task_info in task_progress['task_details'].items():
                status = task_info['status']

                if status == 'completed':
                    task_progress['completed_tasks'] += 1
                elif status == 'failed':
                    task_progress['failed_tasks'] += 1
                elif status == 'running':
                    task_progress['running_tasks'].append({
                        'id': task_id,
                        'description': task_info['description'][:50] + "..." if len(task_info['description']) > 50 else
                        task_info['description'],
                        'type': task_info['type'],
                        'running_time': time.time() - task_info.get('timestamp', time.time())
                    })
                elif status == 'pending':
                    # Check if waiting for dependencies
                    deps = task_info.get('dependencies', [])
                    unmet_deps = []
                    for dep in deps:
                        dep_task = task_progress['task_details'].get(dep)
                        if not dep_task or dep_task['status'] not in ['completed']:
                            unmet_deps.append(dep)

                    task_progress['waiting_tasks'].append({
                        'id': task_id,
                        'description': task_info['description'][:50] + "..." if len(task_info['description']) > 50 else
                        task_info['description'],
                        'waiting_for': unmet_deps,
                        'priority': task_info.get('priority', 1)
                    })

            # Total tasks
            task_progress['total_tasks'] = len(task_progress['task_details'])

            # Identify parallel execution
            current_time = time.time()
            recent_running = [
                task for task in task_progress['running_tasks']
                if task['running_time'] < 30  # Running in last 30 seconds
            ]

            if len(recent_running) > 1:
                task_progress['parallel_executing'] = recent_running

            # Calculate performance metrics
            if task_progress['total_tasks'] > 0:
                completion_rate = task_progress['completed_tasks'] / task_progress['total_tasks']
                failure_rate = task_progress['failed_tasks'] / task_progress['total_tasks']

                task_progress['performance_metrics'] = {
                    'completion_rate': completion_rate,
                    'failure_rate': failure_rate,
                    'avg_execution_time': task_progress['execution_duration'] / max(task_progress['completed_tasks'], 1)
                }

            return task_progress

        except Exception as e:
            if self.mode == VerbosityMode.DEBUG:
                print(f"⚠️ Task executor progress error: {e}")
            return {
                'total_tasks': 0, 'completed_tasks': 0, 'failed_tasks': 0,
                'running_tasks': [], 'waiting_tasks': [], 'parallel_executing': [],
                'execution_groups': [], 'current_strategy': 'unknown',
                'execution_duration': 0, 'task_details': {}, 'dependency_chains': [],
                'performance_metrics': {}
            }

    def _create_task_executor_display(self, summary: Dict[str, Any]) -> Optional[Panel]:
        """
        Create comprehensive task executor display showing parallel execution,
        dependencies, and real-time progress.
        """
        try:
            task_progress = self._get_task_executor_progress()

            if task_progress['total_tasks'] == 0:
                return None

            content_lines = []

            # Header with overall progress
            total = task_progress['total_tasks']
            completed = task_progress['completed_tasks']
            failed = task_progress['failed_tasks']
            running = len(task_progress['running_tasks'])
            waiting = len(task_progress['waiting_tasks'])

            # Progress bar
            if total > 0:
                progress_ratio = completed / total
                bar_width = 20
                filled_width = int(bar_width * progress_ratio)
                progress_bar = "█" * filled_width + "░" * (bar_width - filled_width)

                progress_line = f"📊 Tasks: [{progress_bar}] {completed}/{total} completed"
                if failed > 0:
                    progress_line += f", {failed} failed"
                content_lines.append(progress_line)

            # Current execution status
            status_parts = []
            if running > 0:
                status_parts.append(f"🔄 {running} running")
            if waiting > 0:
                status_parts.append(f"⏸️ {waiting} waiting")
            if len(task_progress['parallel_executing']) > 1:
                status_parts.append(f"⚡ {len(task_progress['parallel_executing'])} parallel")

            if status_parts:
                content_lines.append(" | ".join(status_parts))

            content_lines.append("")  # Separator

            # Current running tasks (detailed)
            if task_progress['running_tasks']:
                content_lines.append("🔄 Currently Executing:")
                for task in task_progress['running_tasks'][:4]:  # Show up to 4
                    running_time = task['running_time']
                    time_str = f"{running_time:.0f}s" if running_time < 60 else f"{running_time / 60:.1f}m"
                    task_line = f"  • {task['id']}: {task['description']} ({task['type']}, {time_str})"
                    content_lines.append(task_line)

                if len(task_progress['running_tasks']) > 4:
                    content_lines.append(f"  ... +{len(task_progress['running_tasks']) - 4} more")

            # Parallel execution info
            if len(task_progress['parallel_executing']) > 1:
                content_lines.append("")
                content_lines.append(f"⚡ Parallel Execution ({len(task_progress['parallel_executing'])} tasks):")
                for task in task_progress['parallel_executing'][:3]:
                    content_lines.append(f"  ⚡ {task['id']}: {task['type']}")

            # Waiting tasks (verbose mode)
            if task_progress['waiting_tasks'] and self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                content_lines.append("")
                content_lines.append("⏸️ Waiting for Dependencies:")
                for task in sorted(task_progress['waiting_tasks'], key=lambda x: x['priority'])[:3]:
                    deps_str = ", ".join(task['waiting_for']) if task['waiting_for'] else "unknown"
                    content_lines.append(f"  • {task['id']}: waiting for [{deps_str}]")

            # Execution strategy
            if task_progress['current_strategy'] != 'unknown':
                content_lines.append("")
                strategy_display = task_progress['current_strategy'].replace('_', ' ').title()
                content_lines.append(f"📋 Strategy: {strategy_display}")

                # Execution groups info
                if task_progress['execution_groups'] and self.mode == VerbosityMode.DEBUG:
                    groups_info = []
                    for group in task_progress['execution_groups']:
                        group_size = len(group.get('tasks', []))
                        group_mode = group.get('execution_mode', 'sequential')
                        groups_info.append(f"Group {group.get('group_id', '?')}: {group_size} tasks ({group_mode})")

                    if groups_info:
                        content_lines.append(f"  Groups: {' | '.join(groups_info)}")

            # Performance metrics (debug mode)
            if task_progress['performance_metrics'] and self.mode == VerbosityMode.DEBUG:
                metrics = task_progress['performance_metrics']
                content_lines.append("")
                content_lines.append("📈 Performance:")
                content_lines.append(f"  Completion Rate: {metrics['completion_rate']:.1%}")
                if metrics['failure_rate'] > 0:
                    content_lines.append(f"  Failure Rate: {metrics['failure_rate']:.1%}")
                content_lines.append(f"  Avg Task Time: {metrics['avg_execution_time']:.1f}s")

            # Create title with status
            if running > 0:
                title = f"🔄 Task Executor - {running} Active"
            elif waiting > 0:
                title = f"⏸️ Task Executor - {waiting} Waiting"
            else:
                title = f"✅ Task Executor - {completed}/{total} Complete"

            return Panel(
                "\n".join(content_lines),
                title=title,
                style="yellow",
                box=box.ROUNDED,
                width=90
            )

        except Exception as e:
            if self.mode == VerbosityMode.DEBUG:
                return Panel(f"Task executor display error: {e}", title="⚠️ Task Executor Error", style="red")
            return None

    # ============================================================================
    # ENHANCED PROGRESS CALLBACK (UPDATED)
    # ============================================================================

    async def progress_callback(self, event: ProgressEvent):
        """Enhanced progress callback with intelligent live display management"""
        is_task_update = False
        try:
            # Update activity time
            self._last_activity_time = time.time()

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

            # Handle specific event types
            if event.event_type.startswith('task_'):
                self.print_task_update_from_event(event)
                is_task_update = True

            if event.node_name == "LLMReasonerNode":
                self.print_reasoner_update_from_event(event)

            # Force full tree updates for important events
            if (event.event_type in ["error", "execution_complete", "plan_created"] or
                event.success is False or
                (event.metadata and event.metadata.get("error"))):
                self._needs_full_tree = True

            # Update agent name
            self.agent_name = event.agent_name if event.agent_name else event.metadata.get("agent_name",
                                                                                           self.agent_name)

            # Print strategy and plan updates (non-live components)
            if not is_task_update and event.node_name != "LLMReasonerNode":
                self.print_strategy_from_event(event)
                self.print_plan_from_event(event)

            # **INTELLIGENT DISPLAY MANAGEMENT**
            self._manage_display_lifecycle()

            # **LIVE DISPLAY UPDATE**
            if self._should_print_update():
                self._create_live_display()

            # Debug info
            if self.mode == VerbosityMode.DEBUG:
                self._print_debug_event(event)

        except Exception as e:
            self._consecutive_errors += 1
            print(f"⚠️ Progress callback error #{self._consecutive_errors}: {e}")
            if self._consecutive_errors > self._error_threshold:
                print("🚨 Progress printing disabled due to excessive errors")
                self.progress_callback = self._noop_callback

    def _should_print_update(self) -> bool:
        """Enhanced decision logic for when to print updates with live display awareness"""
        current_time = time.time()

        # Always update for important events
        if self._needs_full_tree:
            return True

        # In live display mode, update more frequently
        min_interval = 0.5 if self._display_active else 1.5

        # Rate limiting
        if current_time - self._last_update_time < min_interval:
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

            else:  # DEBUG mode or REALTIME
                should_update = True

            if should_update:
                self._last_print_hash = current_hash
                return True

            return False

        except Exception as e:
            self._consecutive_errors += 1
            if self._consecutive_errors > self._error_threshold:
                self._fallback_mode = True
                print(f"⚠️ Printer error threshold exceeded, switching to fallback mode: {e}")
            return False

    # ============================================================================
    # UTILITY METHODS (UPDATED)
    # ============================================================================

    def print_final_summary(self):
        """Print comprehensive final summary with live display cleanup"""
        try:
            # Stop live display
            if self._display_active:
                self._stop_live_display()

            # Clear any remaining one-line display
            if self._last_summary:
                print()  # Move to next line

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
            final_tree = self._render_dynamic_tree(summary)
            self.console.print(final_tree)

            # Comprehensive summary table
            self._print_final_summary_table(summary)

            # Performance analysis
            if self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                self._print_performance_analysis(summary)

        except Exception as e:
            print(f"⚠️ Error printing final summary: {e}")
            self._print_final_summary_fallback()

    def flush(self, run_name: str = None) -> Dict[str, Any]:
        """Enhanced flush with live display management"""
        try:
            # Stop live display before flushing
            if self._display_active:
                self._stop_live_display()

            # Clear any remaining display
            if self._last_summary:
                print()  # Move to next line

            # Generate run info
            current_time = time.time()
            if run_name is None:
                run_name = f"run_{self._current_run_id + 1}"

            # [Continue with existing flush logic...]
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
                "display_was_active": self._display_active
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
                    "performance_metrics": node.get_performance_summary()
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

    def _reset_for_fresh_execution(self):
        """Reset internal state for a completely fresh execution"""
        try:
            # Increment run counter
            self._current_run_id += 1

            # Reset display state
            self._display_active = False
            self._last_activity_time = time.time()
            self._needs_full_tree = False
            self._last_summary = ""
            self._last_display_lines = 20

            # Reset tree builder with completely fresh state
            self.tree_builder = ExecutionTreeBuilder()

            # Reset print history
            self.print_history = []

            # Reset timing and state tracking
            self._last_print_hash = None
            self._print_counter = 0
            self._last_update_time = 0
            self._spinner_index = 0

            # Reset error handling but don't reset fallback mode completely
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


    # ============================================================================
    # CORE LIVE DISPLAY METHODS
    # ============================================================================

    def _create_live_display(self) -> None:
        """
        Enhanced live display with combined agent outline and task executor progress.
        """
        try:
            if self._fallback_mode or not self.use_rich:
                self._print_fallback()
                return

            # Try prompt_toolkit integration first
            integration_info = self._detect_and_integrate_prompt_toolkit()
            summary = self.tree_builder.get_execution_summary()

            # Get combined progress information
            outline_info = self._get_current_outline_info()
            task_progress = self._get_task_executor_progress()

            if integration_info:
                self._handle_prompt_toolkit_display(integration_info, summary)
            else:
                # Enhanced terminal display with combined progress
                should_show_outline = self.mode != VerbosityMode.MINIMAL
                should_show_tasks = self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]

                self._handle_terminal_display(
                    summary,
                    show_outline=should_show_outline,
                    show_tasks=should_show_tasks
                )

            self._print_counter += 1
            self._needs_full_tree = False
            self._last_update_time = time.time()
            self._consecutive_errors = 0

        except Exception as e:
            self._consecutive_errors += 1
            if self._consecutive_errors <= self._error_threshold:
                print(f"⚠️ Live display error #{self._consecutive_errors}: {e}")
            self._print_fallback()

    def _handle_terminal_display(self, summary: Dict[str, Any], show_outline: bool = True, show_tasks: bool = False):
        """
        Handles the rendering of the live display in the terminal.
        This unified method is configurable to optionally show outline and task progress panels.

        Args:
            summary (Dict[str, Any]): The execution summary from the tree builder.
            show_outline (bool): If True, displays the execution outline panel.
            show_tasks (bool): If True, displays the task executor progress panel.
        """
        try:
            # Rate limiting to prevent flickering
            current_time = time.time()
            if current_time - self._last_update_time < self._min_display_interval:
                return

            # Clear terminal for a clean live update (except for the very first print)
            if self._print_counter > 1 and self._display_active:
                if self.mode == VerbosityMode.REALTIME:
                    self.console.clear()

            # --- Create all display components ---
            header = self._create_enhanced_header(summary)
            tree = self._render_dynamic_tree(summary)
            predictions = self._create_predictions_panel(summary)
            status_bar = self._create_live_status_bar(summary)

            # --- Render the complete display to the console ---
            self.console.print()
            self.console.print(header)

            # Conditionally create and display the outline panel
            if show_outline:
                outline_panel = self._create_outline_display(summary)
                if outline_panel:
                    self.console.print(outline_panel)

            # Conditionally create and display the task executor panel
            if show_tasks:
                task_panel = self._create_task_executor_display(summary)
                if task_panel:
                    self.console.print(task_panel)

            # Always display the main execution tree
            self.console.print(tree)

            # Display predictions in more verbose modes
            if predictions and self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                self.console.print(predictions)

            # Always show the status bar when the live display is active
            if self._display_active:
                self.console.print(status_bar)

        except Exception as e:
            print(f"⚠️ Enhanced terminal display error: {e}")
            self._print_fallback()

    def _handle_prompt_toolkit_display(self, integration_info: Dict[str, Any], summary: Dict[str, Any]):
        """
        Safe prompt_toolkit display handling with comprehensive error recovery.
        ERSETZT: Die bestehende _handle_prompt_toolkit_display Methode
        """
        try:
            # Try to integrate with application layout (only once)
            if not hasattr(self, '_layout_integration_attempted'):
                self._layout_integration_attempted = True
                integration_success = self._integrate_with_application_layout(integration_info)
                if not integration_success:
                    if self.mode == VerbosityMode.DEBUG:
                        print("⚠️ Layout integration failed, using fallback display")

            # Determine display approach based on integration success
            layout_integrated = integration_info.get('layout_integration', False)

            if self.realtime_minimal and self.mode == VerbosityMode.REALTIME and not self._needs_full_tree:
                # Single line status update
                self._update_prompt_toolkit_status_line(integration_info, summary)
            else:
                # Full display update
                if layout_integrated:
                    # Use integrated widgets
                    self._update_prompt_toolkit_full_display(integration_info, summary)
                else:
                    # Use console fallback with prompt_toolkit awareness
                    self._update_prompt_toolkit_console_fallback(integration_info, summary)

        except Exception as e:
            if self.mode == VerbosityMode.DEBUG:
                print(f"⚠️ Prompt toolkit display error, using terminal fallback: {e}")
            # Complete fallback to terminal display
            self._handle_terminal_display(summary, show_outline=True, show_tasks=False)

    def _update_prompt_toolkit_status_line(self, integration_info: Dict[str, Any], summary: Dict[str, Any]):
        """
        Safe status line update with error recovery.
        NEUE METHODE
        """
        try:
            # Create status content
            session_info = summary["session_info"]
            timing = summary["timing"]

            # Simple status without complex formatting
            self._spinner_index = (self._spinner_index + 1) % len(self._spinner_chars)
            spinner = self._spinner_chars[self._spinner_index]

            current_node = summary["execution_flow"]["current_node"]
            operation = "running"

            if current_node and current_node in self.tree_builder.nodes:
                node = self.tree_builder.nodes[current_node]
                operation = self._get_node_current_operation(node) or "processing"

            elapsed = timing["elapsed"]
            time_str = f"{elapsed:.0f}s" if elapsed < 60 else f"{elapsed // 60:.0f}m"
            progress = f"{session_info['completed_nodes']}/{session_info['total_nodes']}"

            # Build safe HTML status
            status_html = f"🤖 <cyan>{spinner} @{self.agent_name}</cyan> | <blue>{operation}</blue> | <green>{progress}</green> | <yellow>{time_str}</yellow>"

            # Error indicator
            if summary["performance_metrics"]["error_count"] > 0:
                status_html += f" | <red>⚠️{summary['performance_metrics']['error_count']}</red>"

            # Safe update
            self._update_prompt_toolkit_widget(integration_info, status_html, is_status_line=True)

        except Exception as e:
            if self.mode == VerbosityMode.DEBUG:
                print(f"⚠️ Status line update error: {e}")
            # Ultimate fallback
            print(f"🤖 Agent running... {summary['timing']['elapsed']:.0f}s")

    def _update_prompt_toolkit_console_fallback(self, integration_info: Dict[str, Any], summary: Dict[str, Any]):
        """
        Console fallback for prompt_toolkit environment (no ANSI codes).
        NEUE METHODE
        """
        try:
            # Use rich console but without clearing
            session_info = summary["session_info"]
            timing = summary["timing"]
            perf = summary["performance_metrics"]

            # Simple status update for prompt_toolkit
            timestamp = time.strftime("%H:%M:%S")

            # Current operation
            current_node = summary["execution_flow"]["current_node"]
            if current_node and current_node in self.tree_builder.nodes:
                node = self.tree_builder.nodes[current_node]
                operation = self._get_node_current_operation(node)
                if operation:
                    self.console.print(
                        f"🔄 [{timestamp}] {operation} | {session_info['completed_nodes']}/{session_info['total_nodes']} | {timing['elapsed']:.0f}s",
                        style="blue dim")

            # Show errors if any
            if perf["error_count"] > 0:
                self.console.print(f"⚠️ [{timestamp}] {perf['error_count']} errors detected", style="red dim")

            # Show completion
            if session_info["completed_nodes"] == session_info["total_nodes"] and session_info["total_nodes"] > 0:
                self.console.print(f"✅ [{timestamp}] Execution completed!", style="green bold")

        except Exception as e:
            # Ultimate fallback
            print(f"🤖 [{time.strftime('%H:%M:%S')}] Agent status update")

    def _update_prompt_toolkit_full_display(self, integration_info: Dict[str, Any], summary: Dict[str, Any]):
        """
        Updates the full display area in prompt_toolkit.
        NEUE METHODE
        """
        try:
            # Create rich content but render to string for prompt_toolkit
            import io
            from contextlib import redirect_stdout

            # Capture rich output as string
            string_io = io.StringIO()
            temp_console = Console(file=string_io, width=80, legacy_windows=False)

            # Create components with temporary console
            header = self._create_enhanced_header(summary)
            temp_console.print(header)

            # Only add outline if significant changes or debug mode
            if self._needs_full_tree or self.mode == VerbosityMode.DEBUG:
                outline_panel = self._create_outline_display(summary)
                if outline_panel:
                    temp_console.print(outline_panel)

            # Current operations summary
            current_ops = self._get_current_operations_summary(summary)
            if current_ops:
                temp_console.print(f"🔄 {current_ops}", style="blue")

            # Quick stats
            session_info = summary["session_info"]
            perf = summary["performance_metrics"]

            stats_parts = []
            if perf["total_cost"] > 0:
                stats_parts.append(f"💰 {self._format_cost(perf['total_cost'])}")
            if perf["total_tokens"] > 0:
                stats_parts.append(f"🎯 {perf['total_tokens']:,} tokens")
            if perf["error_count"] > 0:
                stats_parts.append(f"⚠️ {perf['error_count']} errors")

            if stats_parts:
                temp_console.print(" | ".join(stats_parts), style="dim")

            # Get rendered content
            display_content = string_io.getvalue()

            # Add timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            timestamped_content = f"[{timestamp}] Agent Update #{self._print_counter}\n{display_content}"

            # Update widget with captured content
            self._update_prompt_toolkit_widget(integration_info, timestamped_content, is_status_line=False)

        except Exception as e:
            if self.mode == VerbosityMode.DEBUG:
                print(f"⚠️ Full display update error: {e}")
            # Fallback to status line only
            self._update_prompt_toolkit_status_line(integration_info, summary)

    def _update_one_line_display_terminal(self) -> None:
        """
        Enhanced one-line display using detailed activity detection.
        """
        try:
            # Get detailed activity information
            activity_info = self._get_detailed_current_activity()
            tool_usage = self._get_tool_usage_summary()
            outline_info = self._get_current_outline_info()

            # Create spinner based on activity confidence
            confidence = activity_info.get('confidence', 0.0)
            if confidence > 0.8:
                spinner_chars = ["🎯", "🔥", "⚡", "🚀"]  # High confidence
            elif confidence > 0.5:
                spinner_chars = ["🔄", "⚙️", "🔧", "📊"]  # Medium confidence
            else:
                spinner_chars = ["❓", "🔍", "📋", "⏸️"]  # Low confidence

            self._spinner_index = (self._spinner_index + 1) % len(spinner_chars)
            spinner = spinner_chars[self._spinner_index]

            # Build status line
            status_parts = []
            status_parts.append(f"{spinner} @{self.agent_name}")

            # Current activity
            primary_activity = activity_info.get('primary_activity', 'Unknown')
            if primary_activity != 'Unknown':
                activity_brief = primary_activity
                if len(activity_brief) > 20:
                    activity_brief = activity_brief[:17] + "..."
                status_parts.append(activity_brief)

            # Progress from real outline
            if outline_info.get('outline_created'):
                current_step = outline_info['current_step']
                total_steps = outline_info['total_steps']
                completed_steps = len(outline_info.get('completed_steps', []))
                status_parts.append(f"step:{current_step}/{total_steps}")
                if completed_steps > 0:
                    completion_pct = (completed_steps / total_steps) * 100
                    status_parts.append(f"{completion_pct:.0f}%")
            else:
                # Fallback to system progress
                summary = self.tree_builder.get_execution_summary()
                session_info = summary["session_info"]
                progress = f"{session_info['completed_nodes']}/{session_info['total_nodes']}"
                status_parts.append(progress)

            # Tool activity
            if tool_usage['tools_active']:
                active_count = len(tool_usage['tools_active'])
                status_parts.append(f"tools:{active_count}")
            elif tool_usage['current_tool_operation']:
                tool_op = tool_usage['current_tool_operation']
                if len(tool_op) > 15:
                    tool_op = tool_op[:12] + "..."
                status_parts.append(tool_op)

            # Timing
            summary = self.tree_builder.get_execution_summary()
            elapsed = summary["timing"]["elapsed"]
            if elapsed < 60:
                time_str = f"{elapsed:.0f}s"
            elif elapsed < 3600:
                time_str = f"{elapsed // 60:.0f}m{elapsed % 60:.0f}s"
            else:
                time_str = f"{elapsed // 3600:.0f}h{(elapsed % 3600) // 60:.0f}m"
            status_parts.append(time_str)

            # Error indicator
            error_count = summary["performance_metrics"]["error_count"]
            if error_count > 0:
                status_parts.append(f"⚠️{error_count}")

            # Cost indicator (if significant)
            if summary["performance_metrics"]["total_cost"] > 0.01:
                cost_str = self._format_cost(summary["performance_metrics"]["total_cost"])
                status_parts.append(f"💰{cost_str}")

            # Build final status line
            status_line = " | ".join(status_parts)

            # Ensure line doesn't exceed terminal width
            max_width = 120
            if len(status_line) > max_width:
                # Truncate activity description if too long
                if len(status_parts) > 3 and len(status_parts[2]) > 15:
                    status_parts[2] = status_parts[2][:12] + "..."
                    status_line = " | ".join(status_parts)

                # Final truncation if still too long
                if len(status_line) > max_width:
                    status_line = status_line[:max_width - 3] + "..."

            # Terminal output with confidence-based coloring (if supported)
            print(f"\r{status_line:<{max_width}}", end="", flush=True)
            self._last_summary = status_line

        except Exception as e:
            # Error handling with safe terminal output
            error_msg = f"⚠️ Status update error: {str(e)[:50]}"
            print(f"\r{error_msg:<100}", end="", flush=True)
            self._last_summary = error_msg

    def _create_outline_display(self, summary: Dict[str, Any]) -> Optional[Panel]:
        """
        Enhanced outline display showing real agent progress through steps.
        Provides comprehensive view of execution state and progress.
        """
        try:
            outline_info = self._get_current_outline_info()
            if not outline_info or not outline_info.get('steps'):
                return None

            outline_steps = outline_info.get('steps', [])
            current_step = outline_info.get('current_step', 1)
            completed_steps = outline_info.get('completed_steps', [])
            current_step_progress = outline_info.get('current_step_progress', '')
            task_stack_items = outline_info.get('task_stack_items', 0)
            reasoning_loops = outline_info.get('reasoning_loops', 0)
            completion_percentage = outline_info.get('completion_percentage', 0)

            # Build enhanced outline content
            outline_content = []

            # Progress header with completion percentage
            if outline_info.get('total_steps', 0) > 0:
                progress_bar_width = 20
                filled_width = int(progress_bar_width * (completion_percentage / 100))
                progress_bar = "█" * filled_width + "░" * (progress_bar_width - filled_width)
                outline_content.append(f"📊 Progress: [{progress_bar}] {completion_percentage:.1f}%")
                outline_content.append("")

            # Current operation summary
            current_operation = self._get_current_step_operation()
            if current_operation:
                outline_content.append(f"🔄 Current: {current_operation}")
                if task_stack_items > 0:
                    outline_content.append(f"📋 Task Queue: {task_stack_items} items pending")
                if reasoning_loops > 0:
                    outline_content.append(f"🧠 Reasoning Depth: {reasoning_loops} loops")
                outline_content.append("")

            # Step-by-step progress
            for i, step in enumerate(outline_steps, 1):
                # Enhanced status indicators
                if i in completed_steps:
                    status_icon = "✅"
                    status_style = "[green]"
                    step_status = "COMPLETED"
                elif i == current_step:
                    # Animated indicator for current step
                    animation_chars = ["🔄", "⚡", "🎯", "🔧"]
                    anim_char = animation_chars[int(time.time() * 2) % len(animation_chars)]
                    status_icon = anim_char
                    status_style = "[blue bold]"
                    step_status = "ACTIVE"
                elif i < current_step:
                    status_icon = "✅"
                    status_style = "[green dim]"
                    step_status = "COMPLETED"
                else:
                    status_icon = "⏸️"
                    status_style = "[yellow dim]"
                    step_status = "PENDING"

                # Step description with method info
                step_desc = step.get('description', f'Step {i}')
                method = step.get('method', 'unknown')
                expected_outcome = step.get('expected_outcome', '')

                # Truncate long descriptions
                if len(step_desc) > 70:
                    step_desc = step_desc[:67] + "..."

                step_line = f"{status_icon} Step {i}: {step_desc}"

                # Add method info for verbose modes
                if self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                    method_display = method.replace('_', ' ').title()
                    step_line += f"\n    📋 Method: {method_display}"

                    if expected_outcome:
                        outcome_brief = expected_outcome[:60] + "..." if len(
                            expected_outcome) > 60 else expected_outcome
                        step_line += f"\n    🎯 Expected: {outcome_brief}"

                # Add current activity for active step
                if i == current_step:
                    if current_step_progress:
                        progress_brief = current_step_progress[:80] + "..." if len(
                            current_step_progress) > 80 else current_step_progress
                        step_line += f"\n    ▶️ Progress: {progress_brief}"
                    elif step.get('current_action'):
                        step_line += f"\n    ▶️ {step['current_action']}"

                    # Add timing estimate if available
                    if outline_info.get('estimated_completion'):
                        remaining_time = outline_info['estimated_completion']
                        if remaining_time > 0:
                            time_str = f"{remaining_time:.0f}s" if remaining_time < 60 else f"{remaining_time / 60:.1f}m"
                            step_line += f"\n    ⏱️ Est. completion: {time_str}"

                outline_content.append(step_line)

            # Summary footer
            completed_count = len(completed_steps)
            total_steps = len(outline_steps)

            if completed_count == total_steps and total_steps > 0:
                summary_line = "🎉 All outline steps completed!"
            elif completed_count > 0:
                remaining = total_steps - completed_count
                summary_line = f"📈 {completed_count}/{total_steps} steps completed, {remaining} remaining"
            else:
                summary_line = f"🚀 Starting execution: {total_steps} steps planned"

            outline_content.append("")
            outline_content.append(summary_line)

            # Create title with progress indication
            title = f"📋 Execution Outline - {completed_count}/{total_steps} Complete"
            if current_step <= total_steps:
                title += f" (Step {current_step})"

            return Panel(
                "\n".join(outline_content),
                title=title,
                style="cyan",
                box=box.ROUNDED,
                width=100
            )

        except Exception as e:
            if self.mode == VerbosityMode.DEBUG:
                return Panel(f"Outline display error: {e}", title="⚠️ Outline Error", style="red")
            return None

    def _get_current_outline_info(self) -> Dict[str, Any]:
        """
        Extract real outline information from create_initial_outline task and meta-tool metadata.
        Tracks actual agent progress through dynamically created outline steps.
        """
        try:
            outline_info = {
                'steps': [],
                'current_step': 1,
                'completed_steps': [],
                'total_steps': 0,
                'step_descriptions': {},
                'current_step_progress': "",
                'outline_raw_data': None,
                'outline_created': False,
                'actual_step_completions': []
            }

            # Look for the actual outline creation from create_initial_outline task
            outline_creation_event = None
            for node in self.tree_builder.nodes.values():
                for event in node.llm_calls:
                    if (hasattr(event, 'metadata') and event.metadata and
                        event.metadata.get('task_id') == 'create_initial_outline'):
                        outline_creation_event = event
                        break
                if outline_creation_event:
                    break

            # Extract real outline from LLM response
            if outline_creation_event and outline_creation_event.metadata.get('outline'):
                outline_data = outline_creation_event.metadata['outline']
                outline_info['outline_created'] = True
                outline_info['outline_raw_data'] = outline_data

                # Parse the actual outline structure
                if isinstance(outline_data, dict) and 'steps' in outline_data:
                    steps_data = outline_data['steps']
                    for i, step_data in enumerate(steps_data, 1):
                        parsed_step = {
                            'id': i,
                            'description': step_data.get('description', f'Step {i}'),
                            'method': step_data.get('method', 'unknown'),
                            'expected_outcome': step_data.get('expected_outcome', ''),
                            'success_criteria': step_data.get('success_criteria', ''),
                            'status': 'pending',
                            'actual_actions': [],
                            'completion_evidence': ''
                        }
                        outline_info['steps'].append(parsed_step)
                        outline_info['step_descriptions'][i] = parsed_step['description']

                    outline_info['total_steps'] = len(outline_info['steps'])

            # Track real step progression from meta-tool events
            meta_tool_events = []
            for node in self.tree_builder.nodes.values():
                for event in node.llm_calls + node.sub_events:
                    if (hasattr(event, 'metadata') and event.metadata and
                        event.metadata.get('meta_tool_name')):
                        meta_tool_events.append(event)

            # Sort by timestamp for chronological analysis
            meta_tool_events.sort(key=lambda x: x.timestamp)

            # Track actual step completions and current position
            for event in meta_tool_events:
                metadata = event.metadata

                # 1. Track current step position from meta-tool metadata
                if 'outline_step' in metadata:
                    step_num = metadata['outline_step']
                    if isinstance(step_num, int) and step_num > 0:
                        outline_info['current_step'] = max(outline_info['current_step'], step_num)

                        # Record actual actions for this step
                        if outline_info['steps'] and step_num <= len(outline_info['steps']):
                            step_idx = step_num - 1
                            action_description = f"{metadata['meta_tool_name']} at {time.strftime('%H:%M:%S', time.localtime(event.timestamp))}"
                            if action_description not in outline_info['steps'][step_idx]['actual_actions']:
                                outline_info['steps'][step_idx]['actual_actions'].append(action_description)

                # 2. Track step completions from advance_outline_step
                if (metadata.get('meta_tool_name') == 'advance_outline_step' and
                    event.success and metadata.get('step_completed')):

                    completed_step = metadata.get('outline_step', 0)
                    completion_evidence = metadata.get('completion_evidence', '')

                    if completed_step > 0:
                        # Mark step as completed
                        if completed_step not in outline_info['completed_steps']:
                            outline_info['completed_steps'].append(completed_step)

                        # Record completion details
                        completion_record = {
                            'step_id': completed_step,
                            'timestamp': event.timestamp,
                            'evidence': completion_evidence,
                            'method_used': metadata.get('meta_tool_name')
                        }
                        outline_info['actual_step_completions'].append(completion_record)

                        # Update step status and evidence
                        if outline_info['steps'] and completed_step <= len(outline_info['steps']):
                            step_idx = completed_step - 1
                            outline_info['steps'][step_idx]['status'] = 'completed'
                            outline_info['steps'][step_idx]['completion_evidence'] = completion_evidence

                        # Advance current step
                        outline_info['current_step'] = completed_step + 1

                # 3. Track step progress details
                if metadata.get('outline_step_progress'):
                    outline_info['current_step_progress'] = metadata['outline_step_progress']

                # 4. Track step completion expectations
                if metadata.get('outline_step_completion') and metadata.get('outline_step'):
                    step_num = metadata['outline_step']
                    if outline_info['steps'] and step_num <= len(outline_info['steps']):
                        step_idx = step_num - 1
                        outline_info['steps'][step_idx]['status'] = 'executing'
                        outline_info['steps'][step_idx]['expected_completion'] = True

            # Update current step status
            if outline_info['steps'] and outline_info['current_step'] <= len(outline_info['steps']):
                current_step_idx = outline_info['current_step'] - 1
                if outline_info['steps'][current_step_idx]['status'] == 'pending':
                    outline_info['steps'][current_step_idx]['status'] = 'active'

            # Calculate real completion percentage
            if outline_info['total_steps'] > 0:
                completed_count = len(outline_info['completed_steps'])
                outline_info['completion_percentage'] = (completed_count / outline_info['total_steps']) * 100

                # Calculate realistic time estimates based on actual step completion times
                if len(outline_info['actual_step_completions']) > 1:
                    completion_times = [c['timestamp'] for c in outline_info['actual_step_completions']]
                    completion_times.sort()

                    total_time_spent = completion_times[-1] - completion_times[0]
                    avg_time_per_step = total_time_spent / len(completion_times)

                    remaining_steps = outline_info['total_steps'] - completed_count
                    outline_info['estimated_completion'] = remaining_steps * avg_time_per_step

            return outline_info

        except Exception as e:
            if self.mode == VerbosityMode.DEBUG:
                print(f"⚠️ Real outline tracking error: {e}")
            return {
                'steps': [], 'current_step': 1, 'completed_steps': [], 'total_steps': 0,
                'step_descriptions': {}, 'current_step_progress': "", 'outline_raw_data': None,
                'outline_created': False, 'actual_step_completions': []
            }

    def _get_detailed_current_activity(self) -> Dict[str, Any]:
        """
        Detailed detection of current system activity with precise location and action.
        Shows exactly what the agent is doing and where it is in the process.
        """
        try:
            activity_info = {
                'primary_activity': 'Unknown',
                'detailed_description': 'System status unclear',
                'location_context': 'Unknown phase',
                'progress_indicators': [],
                'time_in_current_activity': 0,
                'expected_next_action': 'Uncertain',
                'confidence': 0.0
            }

            current_time = time.time()

            # Get most recent events (last 60 seconds)
            recent_events = []
            for node in self.tree_builder.nodes.values():
                for event in node.llm_calls + node.sub_events + node.tool_calls:
                    if event.timestamp > current_time - 60:
                        recent_events.append({
                            'event': event,
                            'node': node.name,
                            'age': current_time - event.timestamp
                        })

            if not recent_events:
                activity_info['primary_activity'] = 'Idle'
                activity_info['detailed_description'] = 'No recent system activity detected'
                return activity_info

            # Sort by recency
            recent_events.sort(key=lambda x: x['age'])
            most_recent = recent_events[0]

            event = most_recent['event']
            node_name = most_recent['node']
            time_since = most_recent['age']

            activity_info['time_in_current_activity'] = time_since

            # Analyze the most recent event for detailed activity
            if hasattr(event, 'metadata') and event.metadata:
                metadata = event.metadata

                # Meta-tool activity analysis
                if metadata.get('meta_tool_name'):
                    meta_tool = metadata['meta_tool_name']
                    execution_phase = metadata.get('execution_phase', 'unknown')
                    outline_step = metadata.get('outline_step', 0)

                    if meta_tool == 'internal_reasoning':
                        thought_num = metadata.get('thought_number', 1)
                        current_focus = metadata.get('current_focus', '')
                        confidence_level = metadata.get('confidence_level', 0)

                        activity_info['primary_activity'] = 'Deep Reasoning'
                        activity_info[
                            'detailed_description'] = f"Processing thought {thought_num}: {current_focus[:100]}..."
                        activity_info[
                            'location_context'] = f"Reasoning loop, step {outline_step}" if outline_step > 0 else "Reasoning loop"
                        activity_info['confidence'] = confidence_level

                        if metadata.get('next_thought_needed'):
                            activity_info['expected_next_action'] = 'Continue reasoning with next thought'
                        else:
                            activity_info['expected_next_action'] = 'Move to action phase'

                    elif meta_tool == 'delegate_to_llm_tool_node':
                        task_desc = metadata.get('delegated_task_description', '')
                        tools_count = metadata.get('tools_count', 0)

                        if execution_phase == 'meta_tool_start':
                            activity_info['primary_activity'] = 'Initiating Tool Delegation'
                            activity_info['detailed_description'] = f"Delegating: {task_desc[:80]}..."
                            activity_info['expected_next_action'] = f'Execute task using {tools_count} available tools'
                        else:
                            activity_info['primary_activity'] = 'Processing Tool Results'
                            activity_info['detailed_description'] = f"Analyzing results from tool delegation"
                            activity_info['expected_next_action'] = 'Integrate results and continue'

                        activity_info[
                            'location_context'] = f"Tool delegation phase, step {outline_step}" if outline_step > 0 else "Tool delegation phase"

                    elif meta_tool == 'create_and_execute_plan':
                        goals_count = metadata.get('goals_count', 0)

                        if execution_phase == 'meta_tool_start':
                            activity_info['primary_activity'] = 'Creating Execution Plan'
                            activity_info['detailed_description'] = f"Developing plan with {goals_count} goals"
                            activity_info['expected_next_action'] = 'Execute the created plan'
                        else:
                            activity_info['primary_activity'] = 'Executing Plan'
                            activity_info['detailed_description'] = f"Running plan with {goals_count} goals"
                            activity_info['expected_next_action'] = 'Complete plan execution and review results'

                        activity_info[
                            'location_context'] = f"Planning phase, step {outline_step}" if outline_step > 0 else "Planning phase"

                    elif meta_tool == 'manage_internal_task_stack':
                        stack_action = metadata.get('stack_action', 'unknown')
                        stack_size = metadata.get('stack_size_after', metadata.get('stack_size_before', 0))

                        activity_info['primary_activity'] = 'Managing Task Queue'
                        activity_info[
                            'detailed_description'] = f"Task stack {stack_action}, {stack_size} items in queue"
                        activity_info['location_context'] = "Task management"
                        activity_info['expected_next_action'] = 'Process next priority task'

                    elif meta_tool == 'advance_outline_step':
                        step_completed = metadata.get('step_completed', False)

                        if step_completed:
                            activity_info['primary_activity'] = 'Advancing Outline Step'
                            activity_info[
                                'detailed_description'] = f"Moving from step {outline_step} to step {outline_step + 1}"
                            activity_info['expected_next_action'] = f'Begin step {outline_step + 1} activities'
                        else:
                            activity_info['primary_activity'] = 'Preparing Step Advancement'
                            activity_info['detailed_description'] = f"Validating completion of step {outline_step}"
                            activity_info['expected_next_action'] = 'Advance to next outline step'

                        activity_info['location_context'] = f"Step transition, current step {outline_step}"

                    elif meta_tool == 'direct_response':
                        final_answer_length = metadata.get('final_answer_length', 0)

                        activity_info['primary_activity'] = 'Generating Final Response'
                        activity_info[
                            'detailed_description'] = f"Creating final answer ({final_answer_length} characters)"
                        activity_info['location_context'] = "Response generation phase"
                        activity_info['expected_next_action'] = 'Complete execution and return response'

                # LLM call analysis
                elif event.event_type == 'llm_call':
                    task_id = metadata.get('task_id', 'unknown')

                    if task_id == 'create_initial_outline':
                        activity_info['primary_activity'] = 'Creating Initial Outline'
                        activity_info['detailed_description'] = 'Analyzing query and creating execution outline'
                        activity_info['location_context'] = 'Initialization phase'
                        activity_info['expected_next_action'] = 'Begin executing outline steps'
                    else:
                        activity_info['primary_activity'] = 'LLM Processing'
                        activity_info['detailed_description'] = f'Processing LLM task: {task_id}'
                        activity_info['location_context'] = f'LLM call in {node_name}'
                        activity_info['expected_next_action'] = 'Process LLM response'

            # Tool call analysis
            elif hasattr(event, 'tool_name') and event.tool_name:
                tool_name = event.tool_name

                if event.tool_success is None:  # Still running
                    activity_info['primary_activity'] = f'Using {tool_name} Tool'
                    activity_info['detailed_description'] = f'Executing {tool_name} operation'
                    activity_info['location_context'] = f'Tool execution in {node_name}'
                    activity_info['expected_next_action'] = 'Process tool results'
                elif event.tool_success:
                    activity_info['primary_activity'] = 'Processing Tool Results'
                    activity_info['detailed_description'] = f'Analyzing results from {tool_name}'
                    activity_info['location_context'] = f'Post-tool processing in {node_name}'
                    activity_info['expected_next_action'] = 'Continue with next operation'

            # Add progress indicators
            outline_info = self._get_current_outline_info()
            if outline_info.get('outline_created'):
                current_step = outline_info['current_step']
                total_steps = outline_info['total_steps']
                completed_steps = len(outline_info.get('completed_steps', []))

                activity_info['progress_indicators'].append(f"Outline: Step {current_step}/{total_steps}")
                activity_info['progress_indicators'].append(f"Completed: {completed_steps}/{total_steps}")

                if outline_info.get('completion_percentage'):
                    activity_info['progress_indicators'].append(
                        f"Progress: {outline_info['completion_percentage']:.0f}%")

            # Set confidence based on data quality
            if time_since < 10:  # Very recent
                activity_info['confidence'] = 0.9
            elif time_since < 30:  # Recent
                activity_info['confidence'] = 0.7
            else:  # Older data
                activity_info['confidence'] = 0.4

            return activity_info

        except Exception as e:
            return {
                'primary_activity': 'System Error',
                'detailed_description': f'Error detecting activity: {str(e)}',
                'location_context': 'Error state',
                'progress_indicators': [],
                'time_in_current_activity': 0,
                'expected_next_action': 'Recover from error',
                'confidence': 0.0
            }

    def _infer_outline_from_nodes(self) -> Dict[str, Any]:
        """
        Fallback method to infer outline from node progression if no explicit outline found.
        ADD this method if missing:
        """
        try:
            nodes = list(self.tree_builder.nodes.keys())
            steps = []
            for i, node_name in enumerate(nodes, 1):
                node = self.tree_builder.nodes[node_name]
                steps.append({
                    'id': i,
                    'description': f"{node_name}: {node.status.value}",
                    'status': node.status.value,
                    'priority': 1
                })

            # Determine current step based on active nodes
            current_step = 1
            for i, node_name in enumerate(nodes, 1):
                if node_name in self.tree_builder.active_nodes:
                    current_step = i
                    break

            completed_steps = []
            for i, node_name in enumerate(nodes, 1):
                node = self.tree_builder.nodes[node_name]
                if node.status == NodeStatus.COMPLETED:
                    completed_steps.append(i)

            return {
                'steps': steps,
                'current_step': current_step,
                'completed_steps': completed_steps,
                'total_steps': len(steps)
            }

        except Exception as e:
            return {'steps': [], 'current_step': 1, 'completed_steps': [], 'total_steps': 0}

    def _get_current_step_operation(self) -> Optional[str]:
        """
        Enhanced current step operation detection from recent meta-tool activity.
        Provides detailed, real-time insight into what the agent is actually doing.
        """
        try:
            # Get recent meta-tool events (last 30 seconds)
            cutoff_time = time.time() - 30
            recent_meta_tools = []

            for node in self.tree_builder.nodes.values():
                for event in node.llm_calls + node.sub_events:
                    if (event.timestamp > cutoff_time and
                        hasattr(event, 'metadata') and event.metadata and
                        event.metadata.get('meta_tool_name')):
                        recent_meta_tools.append(event)

            if not recent_meta_tools:
                return None

            # Sort by timestamp to get most recent
            recent_meta_tools.sort(key=lambda x: x.timestamp, reverse=True)
            latest_event = recent_meta_tools[0]
            metadata = latest_event.metadata

            meta_tool_name = metadata.get('meta_tool_name')
            execution_phase = metadata.get('execution_phase', '')
            reasoning_loop = metadata.get('reasoning_loop', 0)

            # Detailed operation descriptions based on meta-tool and phase
            if meta_tool_name == 'internal_reasoning':
                thought_num = metadata.get('thought_number', 1)
                total_thoughts = metadata.get('total_thoughts', 1)
                current_focus = metadata.get('current_focus', '')

                if current_focus:
                    focus_brief = current_focus[:50] + "..." if len(current_focus) > 50 else current_focus
                    return f"Reasoning (thought {thought_num}/{total_thoughts}): {focus_brief}"
                else:
                    return f"Deep reasoning (thought {thought_num}/{total_thoughts})"

            elif meta_tool_name == 'manage_internal_task_stack':
                stack_action = metadata.get('stack_action', 'unknown')
                task_description = metadata.get('task_description', '')
                stack_size = metadata.get('stack_size_after', metadata.get('stack_size_before', 0))

                if stack_action == 'add' and task_description:
                    task_brief = task_description[:40] + "..." if len(task_description) > 40 else task_description
                    return f"Adding task: {task_brief} (stack: {stack_size})"
                elif stack_action == 'remove' or stack_action == 'complete':
                    return f"Completing task from stack (remaining: {stack_size})"
                elif stack_action == 'get_current':
                    return f"Reviewing task priorities ({stack_size} items)"
                else:
                    return f"Managing task stack ({stack_action}, {stack_size} items)"

            elif meta_tool_name == 'delegate_to_llm_tool_node':
                if execution_phase == 'meta_tool_start':
                    task_desc = metadata.get('delegated_task_description', '')
                    tools_count = metadata.get('tools_count', 0)

                    if task_desc:
                        task_brief = task_desc[:60] + "..." if len(task_desc) > 60 else task_desc
                        return f"Delegating: {task_brief} ({tools_count} tools available)"
                    else:
                        return f"Delegating task to tool system ({tools_count} tools)"
                else:
                    return "Processing delegated task results"

            elif meta_tool_name == 'create_and_execute_plan':
                goals_count = metadata.get('goals_count', 0)
                estimated_complexity = metadata.get('estimated_complexity', '')

                if execution_phase == 'meta_tool_start':
                    complexity_info = f" ({estimated_complexity})" if estimated_complexity != 'unknown' else ""
                    return f"Creating execution plan with {goals_count} goals{complexity_info}"
                else:
                    return f"Executing plan ({goals_count} goals)"

            elif meta_tool_name == 'advance_outline_step':
                step_completed = metadata.get('step_completed', False)
                completion_evidence = metadata.get('completion_evidence', '')
                next_step_focus = metadata.get('next_step_focus', '')

                if step_completed and next_step_focus:
                    focus_brief = next_step_focus[:50] + "..." if len(next_step_focus) > 50 else next_step_focus
                    return f"Step completed, focusing on: {focus_brief}"
                elif step_completed:
                    return "Advancing to next outline step"
                else:
                    return "Preparing to advance outline step"

            elif meta_tool_name == 'write_to_variables':
                var_scope = metadata.get('variable_scope', '')
                var_key = metadata.get('variable_key', '')
                var_desc = metadata.get('variable_description', '')

                if var_desc:
                    desc_brief = var_desc[:40] + "..." if len(var_desc) > 40 else var_desc
                    return f"Storing result: {desc_brief} ({var_scope}.{var_key})"
                else:
                    return f"Storing data in {var_scope}.{var_key}"

            elif meta_tool_name == 'read_from_variables':
                var_scope = metadata.get('variable_scope', '')
                var_key = metadata.get('variable_key', '')
                read_purpose = metadata.get('read_purpose', '')

                if read_purpose:
                    purpose_brief = read_purpose[:40] + "..." if len(read_purpose) > 40 else read_purpose
                    return f"Retrieving data for: {purpose_brief} ({var_scope}.{var_key})"
                else:
                    return f"Reading {var_scope}.{var_key}"

            elif meta_tool_name == 'direct_response':
                final_answer_length = metadata.get('final_answer_length', 0)
                steps_completed = metadata.get('steps_completed', [])

                if len(steps_completed) > 0:
                    return f"Generating final response ({len(steps_completed)} steps completed, {final_answer_length} chars)"
                else:
                    return f"Generating final response ({final_answer_length} characters)"

            else:
                # Generic meta-tool operation
                tool_display = meta_tool_name.replace('_', ' ').title()
                if reasoning_loop > 0:
                    return f"{tool_display} (reasoning loop {reasoning_loop})"
                else:
                    return tool_display

        except Exception as e:
            if self.mode == VerbosityMode.DEBUG:
                print(f"⚠️ Error detecting current step operation: {e}")
            return None

    def _render_dynamic_tree(self, summary: Dict[str, Any]) -> Tree:
        """
        Enhanced dynamic tree with combined agent outline and task executor progress.
        Shows both high-level agent progression and detailed task execution.
        """
        try:
            # Create root with combined status
            health = summary["health_indicators"]
            timing = summary["timing"]

            # Get both progress systems
            outline_info = self._get_current_outline_info()
            task_progress = self._get_task_executor_progress()

            # Dynamic root title with combined progress
            health_emoji = "🟢" if health["overall_health"] > 0.8 else "🟡" if health["overall_health"] > 0.5 else "🔴"

            elapsed_str = f"{timing['elapsed']:.1f}s" if timing[
                                                             'elapsed'] < 60 else f"{timing['elapsed'] // 60:.0f}m{timing['elapsed'] % 60:.0f}s"

            # Combined title showing both systems
            root_title = f"{health_emoji} @{self.agent_name} Execution ({elapsed_str})"

            # Add progress indicators
            progress_indicators = []
            if outline_info.get('outline_created') and outline_info.get('total_steps', 0) > 0:
                outline_progress = f"Outline: {len(outline_info.get('completed_steps', []))}/{outline_info['total_steps']}"
                progress_indicators.append(outline_progress)

            if task_progress['total_tasks'] > 0:
                task_indicator = f"Tasks: {task_progress['completed_tasks']}/{task_progress['total_tasks']}"
                if len(task_progress['running_tasks']) > 0:
                    task_indicator += f" ({len(task_progress['running_tasks'])} active)"
                progress_indicators.append(task_indicator)

            if progress_indicators:
                root_title += f" | {' | '.join(progress_indicators)}"

            tree = Tree(root_title, style="bold cyan")

            # Add combined overview
            self._add_combined_overview(tree, summary, outline_info, task_progress)

            # Agent outline progress (high-level)
            if outline_info.get('outline_created'):
                outline_branch = tree.add("🎯 Agent Outline Progress", style="bold blue")
                self._add_outline_progress_to_tree(outline_branch, outline_info)

            # Task executor progress (detailed)
            if task_progress['total_tasks'] > 0:
                task_branch = tree.add("⚙️ Task Execution Details", style="bold green")
                self._add_task_progress_to_tree(task_branch, task_progress)

            # Main execution flow (system level)
            if summary["execution_flow"]["flow"]:
                flow_branch = tree.add("🔄 System Execution Flow", style="bold purple")
                self._add_system_flow_to_tree(flow_branch, summary)

            # Error section if there are recent errors
            if self.tree_builder.error_log and self.mode != VerbosityMode.MINIMAL:
                self._add_live_error_section(tree)

            return tree

        except Exception as e:
            # Fallback tree
            error_tree = Tree("❌ Tree rendering error", style="red")
            error_tree.add(f"Error: {str(e)}", style="red dim")
            return error_tree

    def _add_combined_overview(self, tree: Tree, summary: Dict[str, Any], outline_info: Dict[str, Any],
                               task_progress: Dict[str, Any]):
        """Add combined overview showing both agent and task progress"""
        try:
            session_info = summary["session_info"]
            timing = summary["timing"]

            overview_lines = []

            # Agent-level progress
            if outline_info.get('outline_created'):
                current_step = outline_info['current_step']
                total_steps = outline_info['total_steps']
                completed_steps = len(outline_info.get('completed_steps', []))

                agent_progress = f"🎯 Agent: Step {current_step}/{total_steps} ({completed_steps} completed)"
                if outline_info.get('completion_percentage'):
                    agent_progress += f" - {outline_info['completion_percentage']:.0f}%"
                overview_lines.append(agent_progress)

            # Task-level progress
            if task_progress['total_tasks'] > 0:
                task_status_parts = []
                task_status_parts.append(f"{task_progress['completed_tasks']}/{task_progress['total_tasks']} completed")

                if len(task_progress['running_tasks']) > 0:
                    task_status_parts.append(f"{len(task_progress['running_tasks'])} running")
                if len(task_progress['parallel_executing']) > 1:
                    task_status_parts.append(f"{len(task_progress['parallel_executing'])} parallel")
                if len(task_progress['waiting_tasks']) > 0:
                    task_status_parts.append(f"{len(task_progress['waiting_tasks'])} waiting")

                task_progress_line = f"⚙️ Tasks: {' | '.join(task_status_parts)}"
                overview_lines.append(task_progress_line)

            # System health
            health_parts = []
            if session_info["active_nodes"] > 0:
                health_parts.append(f"{session_info['active_nodes']} active nodes")
            if session_info["failed_nodes"] > 0:
                health_parts.append(f"{session_info['failed_nodes']} failed")

            if health_parts:
                overview_lines.append(f"🔧 System: {' | '.join(health_parts)}")

            # Timing and ETA
            timing_parts = [f"Runtime: {human_readable_time(timing['elapsed'])}"]

            if outline_info.get('estimated_completion'):
                eta = outline_info['estimated_completion']
                if eta > 0:
                    eta_str = f"{human_readable_time(eta)}"
                    timing_parts.append(f"ETA: {eta_str}")

            overview_lines.append(f"⏱️ {' | '.join(timing_parts)}")

            if overview_lines:
                overview_branch = tree.add("📊 Live Progress Overview", style="bold yellow")
                for line in overview_lines:
                    overview_branch.add(line, style="yellow dim")

        except Exception as e:
            tree.add(f"⚠️ Overview error: {e}", style="red dim")

    def _add_outline_progress_to_tree(self, outline_branch: Tree, outline_info: Dict[str, Any]):
        """Add agent outline progress to tree branch"""
        try:
            steps = outline_info.get('steps', [])
            current_step = outline_info['current_step']
            completed_steps = outline_info.get('completed_steps', [])

            for step in steps:
                step_id = step['id']
                step_desc = step['description']

                # Status and progress
                if step_id in completed_steps:
                    status_icon = "✅"
                    status_style = "green"
                elif step_id == current_step:
                    # Show current activity
                    current_activity = self._get_detailed_current_activity()
                    if current_activity['primary_activity'] != 'Unknown':
                        status_icon = "🔄"
                        status_style = "blue bold"
                        step_desc += f" → {current_activity['primary_activity']}"
                    else:
                        status_icon = "🎯"
                        status_style = "blue bold"
                else:
                    status_icon = "⏸️"
                    status_style = "yellow dim"

                step_text = f"{status_icon} {step_desc}"
                step_branch = outline_branch.add(step_text, style=status_style)

                # Add step details in verbose mode
                if self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                    method = step.get('method', 'unknown')
                    step_branch.add(f"📋 Method: {method.replace('_', ' ').title()}", style=f"{status_style} dim")

                    if step.get('expected_outcome'):
                        outcome = step['expected_outcome'][:60] + "..." if len(step['expected_outcome']) > 60 else step[
                            'expected_outcome']
                        step_branch.add(f"🎯 Expected: {outcome}", style=f"{status_style} dim")

        except Exception as e:
            outline_branch.add(f"⚠️ Outline display error: {e}", style="red dim")

    def _add_task_progress_to_tree(self, task_branch: Tree, task_progress: Dict[str, Any]):
        """Add detailed task execution progress to tree branch"""
        try:
            # Running tasks
            if task_progress['running_tasks']:
                running_branch = task_branch.add(f"🔄 Running ({len(task_progress['running_tasks'])})", style="blue")
                for task in task_progress['running_tasks'][:5]:  # Show up to 5
                    running_time = task['running_time']
                    time_str = f"{running_time:.0f}s" if running_time < 60 else f"{running_time / 60:.1f}m"
                    task_text = f"{task['id']}: {task['description']} ({time_str})"
                    running_branch.add(task_text, style="blue dim")

            # Parallel execution
            if len(task_progress['parallel_executing']) > 1:
                parallel_branch = task_branch.add(f"⚡ Parallel ({len(task_progress['parallel_executing'])})",
                                                  style="green")
                for task in task_progress['parallel_executing']:
                    parallel_branch.add(f"{task['id']}: {task['type']}", style="green dim")

            # Waiting tasks
            if task_progress['waiting_tasks']:
                waiting_branch = task_branch.add(f"⏸️ Waiting ({len(task_progress['waiting_tasks'])})", style="yellow")
                for task in sorted(task_progress['waiting_tasks'], key=lambda x: x['priority'])[:3]:
                    deps_str = ", ".join(task['waiting_for']) if task['waiting_for'] else "dependencies"
                    task_text = f"{task['id']}: waiting for {deps_str}"
                    waiting_branch.add(task_text, style="yellow dim")

            # Completed summary
            if task_progress['completed_tasks'] > 0:
                completed_text = f"✅ Completed: {task_progress['completed_tasks']}/{task_progress['total_tasks']}"
                if task_progress['failed_tasks'] > 0:
                    completed_text += f" ({task_progress['failed_tasks']} failed)"
                task_branch.add(completed_text, style="green")

        except Exception as e:
            task_branch.add(f"⚠️ Task progress error: {e}", style="red dim")

    def _add_system_flow_to_tree(self, flow_branch: Tree, summary: Dict[str, Any]):
        """Add system-level execution flow (nodes)"""
        try:
            execution_flow = summary["execution_flow"]["flow"]
            active_nodes = set(summary["execution_flow"]["active_nodes"])

            for i, node_name in enumerate(execution_flow[-5:], 1):  # Show last 5 nodes
                if node_name not in self.tree_builder.nodes:
                    continue

                node = self.tree_builder.nodes[node_name]
                status_icon = node.get_status_icon()
                status_style = node.get_status_color()

                node_text = f"{status_icon} {node_name}"
                duration_str = node.get_duration_str()
                if duration_str != "...":
                    node_text += f" ({duration_str})"

                flow_branch.add(node_text, style=status_style)

        except Exception as e:
            flow_branch.add(f"⚠️ System flow error: {e}", style="red dim")

    def _add_dynamic_overview(self, tree: Tree, summary: Dict[str, Any]) -> None:
        """
        Adds dynamic overview section with live metrics and progress indicators.
        """
        try:
            session_info = summary["session_info"]
            perf = summary["performance_metrics"]
            health = summary["health_indicators"]

            # Create live progress indicator
            completed = session_info["completed_nodes"]
            total = session_info["total_nodes"]
            active = session_info["active_nodes"]
            failed = session_info["failed_nodes"]

            # Progress bar visualization
            if total > 0:
                progress_ratio = completed / total
                bar_width = 20
                filled_width = int(bar_width * progress_ratio)
                progress_bar = "█" * filled_width + "▒" * (bar_width - filled_width)
                progress_text = f"📊 Progress: [{progress_bar}] {completed}/{total}"
            else:
                progress_text = f"📊 Progress: {completed}/{total}"

            # Add active indicators
            if active > 0:
                progress_text += f" | {active} active"
            if failed > 0:
                progress_text += f" | {failed} failed"

            overview_branch = tree.add(progress_text, style="bold yellow")

            # Health and performance indicators (verbose modes only)
            if self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                health_text = f"♥️ Health: {health['overall_health']:.1%}"
                if perf["total_cost"] > 0:
                    health_text += f" | 💰 Cost: {self._format_cost(perf['total_cost'])}"
                if perf["total_tokens"] > 0:
                    health_text += f" | 🎯 Tokens: {perf['total_tokens']:,}"

                overview_branch.add(health_text, style="green dim")

        except Exception as e:
            tree.add(f"⚠️ Overview error: {e}", style="red dim")


    def _get_node_current_operation(self, node: ExecutionNode) -> Optional[str]:
        """
        Determines the current operation being performed by a node for live display.
        """
        try:
            # Check most recent meta-tool activity
            if node.llm_calls:
                recent_llm = node.llm_calls[-1]
                if (hasattr(recent_llm, 'metadata') and recent_llm.metadata and
                    recent_llm.timestamp > time.time() - 10):  # Within last 10 seconds

                    meta_tool = recent_llm.metadata.get('meta_tool_name')
                    if meta_tool:
                        # Compact operation descriptions
                        operations = {
                            'internal_reasoning': 'thinking',
                            'manage_internal_task_stack': 'managing tasks',
                            'delegate_to_llm_tool_node': 'delegating',
                            'create_and_execute_plan': 'planning',
                            'advance_outline_step': 'advancing step',
                            'direct_response': 'responding',
                            'write_to_variables': 'storing data',
                            'read_from_variables': 'reading data'
                        }
                        return operations.get(meta_tool, meta_tool.replace('_', ' '))

            # Check recent tool activity
            if node.tool_calls:
                recent_tool = node.tool_calls[-1]
                if recent_tool.timestamp > time.time() - 10:  # Within last 10 seconds
                    tool_name = recent_tool.tool_name
                    if len(tool_name) > 15:
                        return f"using {tool_name[:12]}..."
                    return f"using {tool_name}"

            # Check node status for general operation
            if node.status == NodeStatus.RUNNING:
                if 'Reasoner' in node.name:
                    return 'reasoning'
                elif 'Planner' in node.name:
                    return 'planning'
                elif 'Executor' in node.name:
                    return 'executing'
                elif 'Tool' in node.name:
                    return 'processing'
                else:
                    return 'working'

            return None

        except Exception as e:
            return None


    def _create_predictions_panel(self, summary: Dict[str, Any]) -> Optional[Panel]:
        """
        Enhanced predictions panel using real execution metrics and detailed activity analysis.
        """
        try:
            if self.mode not in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                return None

            # Use the enhanced prediction function with real metrics
            predictions = self._generate_execution_predictions(summary)
            activity_info = self._get_detailed_current_activity()

            if not predictions or not any([predictions.get('next_steps'), predictions.get('estimated_completion')]):
                return None

            prediction_lines = []

            # Current activity context for predictions
            if activity_info['primary_activity'] != 'Unknown':
                prediction_lines.append(f"🔄 Current: {activity_info['primary_activity']}")
                if activity_info.get('expected_next_action') and activity_info['expected_next_action'] != 'Uncertain':
                    prediction_lines.append(f"➡️ Next: {activity_info['expected_next_action']}")
                prediction_lines.append("")

            # Predicted next steps using real data
            if predictions.get('next_steps'):
                prediction_lines.append("🔮 Predicted Next Steps:")
                for i, step in enumerate(predictions['next_steps'][:3], 1):
                    confidence_icon = "🎯" if step['confidence'] > 0.8 else "🔄" if step['confidence'] > 0.5 else "❓"
                    step_text = f"  {i}. {step['action'][:60]}..."
                    prediction_lines.append(f"{confidence_icon} {step_text}")

                    # Add method and timing info
                    method = step.get('method', 'unknown')
                    duration = step.get('estimated_duration', 0)
                    if duration > 0:
                        if duration < 60:
                            time_str = f"{duration:.0f}s"
                        else:
                            time_str = f"{duration / 60:.1f}m"
                        prediction_lines.append(
                            f"     📋 Method: {method.replace('_', ' ').title()} | ⏱️ Est: {time_str} | 🎯 {step['confidence']:.0%}")
                    else:
                        prediction_lines.append(
                            f"     📋 Method: {method.replace('_', ' ').title()} | 🎯 {step['confidence']:.0%}")

            # Completion prediction with real data
            if predictions.get('estimated_completion'):
                completion = predictions['estimated_completion']
                prediction_lines.append("")

                time_remaining = completion['time_remaining']
                if time_remaining < 60:
                    time_str = f"{time_remaining:.0f}s"
                elif time_remaining < 3600:
                    time_str = f"{time_remaining / 60:.1f}m"
                else:
                    time_str = f"{time_remaining / 3600:.1f}h"

                prediction_lines.append(f"🎯 Completion Estimate: {time_str} remaining")
                prediction_lines.append(f"   Confidence: {completion['confidence']:.0%}")

                # Add cost prediction if available
                if completion.get('estimated_additional_cost', 0) > 0:
                    cost_str = self._format_cost(completion['estimated_additional_cost'])
                    prediction_lines.append(f"   💰 Additional cost: {cost_str}")

            # Potential issues using real metrics
            if predictions.get('potential_issues') and self.mode == VerbosityMode.DEBUG:
                high_impact_issues = [issue for issue in predictions['potential_issues'] if
                                      issue.get('impact') == 'high']
                if high_impact_issues:
                    prediction_lines.append("")
                    prediction_lines.append("⚠️ Potential Issues:")
                    for issue in high_impact_issues[:2]:
                        risk_icon = "🚨" if issue['probability'] > 0.7 else "⚠️"
                        prediction_lines.append(
                            f"  {risk_icon} {issue['description']} (Risk: {issue['probability']:.0%})")

            # Overall prediction confidence
            confidence_level = predictions.get('confidence_level', 0.0)
            if confidence_level > 0:
                prediction_lines.append("")
                if confidence_level > 0.8:
                    prediction_lines.append("🎯 High confidence predictions based on real execution data")
                elif confidence_level > 0.5:
                    prediction_lines.append("🔄 Moderate confidence predictions from partial execution data")
                else:
                    prediction_lines.append("❓ Low confidence predictions - limited execution data")

            if not prediction_lines:
                return None

            return Panel(
                "\n".join(prediction_lines),
                title="🔮 Execution Predictions (Real Metrics)",
                style="magenta",
                box=box.ROUNDED,
                width=90
            )

        except Exception as e:
            if self.mode == VerbosityMode.DEBUG:
                return Panel(f"Prediction error: {e}", title="⚠️ Prediction Error", style="red")
            return None

    def _generate_execution_predictions(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate predictions using real execution metrics and patterns.
        Uses actual meta-tool timing, outline progress, and system performance data.
        """
        try:
            predictions = {
                'next_steps': [],
                'estimated_completion': None,
                'potential_issues': [],
                'confidence_level': 0.0
            }

            # Get real outline progress
            outline_info = self._get_current_outline_info()
            tool_usage = self._get_tool_usage_summary()

            session_info = summary["session_info"]
            timing = summary["timing"]
            perf = summary["performance_metrics"]

            # Analyze real meta-tool performance for predictions
            meta_tool_timings = {}
            meta_tool_success_rates = {}

            for node in self.tree_builder.nodes.values():
                for event in node.llm_calls + node.sub_events:
                    if (hasattr(event, 'metadata') and event.metadata and
                        event.metadata.get('meta_tool_name')):

                        tool_name = event.metadata['meta_tool_name']
                        duration = event.metadata.get('execution_duration', 0)

                        if duration > 0:
                            if tool_name not in meta_tool_timings:
                                meta_tool_timings[tool_name] = []
                            meta_tool_timings[tool_name].append(duration)

                        # Track success rates
                        if tool_name not in meta_tool_success_rates:
                            meta_tool_success_rates[tool_name] = {'success': 0, 'total': 0}
                        meta_tool_success_rates[tool_name]['total'] += 1
                        if event.success:
                            meta_tool_success_rates[tool_name]['success'] += 1

            # Predict next steps based on real outline progress
            if outline_info.get('outline_created') and outline_info.get('steps'):
                current_step = outline_info['current_step']
                total_steps = outline_info['total_steps']
                completed_steps = outline_info.get('completed_steps', [])

                # Find next pending steps
                for step in outline_info['steps']:
                    if step['id'] > current_step and step['id'] not in completed_steps:
                        method = step.get('method', 'unknown')

                        # Calculate confidence based on historical performance
                        confidence = 0.7  # base confidence
                        estimated_duration = 10.0  # default

                        if method in meta_tool_success_rates:
                            success_rate = meta_tool_success_rates[method]['success'] / max(
                                meta_tool_success_rates[method]['total'], 1)
                            confidence = min(0.95, success_rate * 0.9)

                        if method in meta_tool_timings:
                            avg_duration = sum(meta_tool_timings[method]) / len(meta_tool_timings[method])
                            estimated_duration = avg_duration * 1.2  # Add buffer

                        predictions['next_steps'].append({
                            'action': step['description'],
                            'method': method,
                            'confidence': confidence,
                            'estimated_duration': estimated_duration
                        })

                        if len(predictions['next_steps']) >= 3:
                            break

            # Calculate realistic completion estimate
            if outline_info.get('total_steps', 0) > 0:
                completed_count = len(outline_info.get('completed_steps', []))
                remaining_steps = outline_info['total_steps'] - completed_count

                if completed_count > 0 and timing['elapsed'] > 0:
                    # Use real completion rate
                    actual_completion_rate = completed_count / timing['elapsed']
                    estimated_remaining_time = remaining_steps / actual_completion_rate

                    # Adjust based on remaining step complexity
                    if remaining_steps > 0:
                        remaining_methods = []
                        for step in outline_info['steps']:
                            if step['id'] not in outline_info.get('completed_steps', []):
                                remaining_methods.append(step.get('method', 'unknown'))

                        # Apply complexity multiplier based on methods
                        complexity_multiplier = 1.0
                        for method in remaining_methods:
                            if method == 'create_and_execute_plan':
                                complexity_multiplier *= 1.8
                            elif method == 'delegate_to_llm_tool_node':
                                complexity_multiplier *= 1.4
                            elif method == 'internal_reasoning':
                                complexity_multiplier *= 1.2

                        estimated_remaining_time *= complexity_multiplier

                    # Calculate confidence based on error rate and consistency
                    confidence = 0.8
                    if perf['error_count'] > 0:
                        error_rate = perf['error_count'] / perf['total_events']
                        confidence *= (1.0 - error_rate)

                    predictions['estimated_completion'] = {
                        'time_remaining': estimated_remaining_time,
                        'confidence': confidence,
                        'estimated_additional_cost': self._predict_additional_cost_real(summary,
                                                                                        estimated_remaining_time,
                                                                                        remaining_methods)
                    }

            # Real potential issues based on actual metrics
            error_rate = perf['error_count'] / max(perf['total_events'], 1) if perf['total_events'] > 0 else 0

            if error_rate > 0.15:
                predictions['potential_issues'].append({
                    'description': f'High error rate detected ({error_rate:.1%}) - may cause delays',
                    'probability': min(0.9, error_rate * 2),
                    'impact': 'high'
                })

            # Tool-specific issues
            for tool_name, rates in meta_tool_success_rates.items():
                success_rate = rates['success'] / max(rates['total'], 1)
                if success_rate < 0.7 and rates['total'] > 2:
                    predictions['potential_issues'].append({
                        'description': f'{tool_name} has low success rate ({success_rate:.1%})',
                        'probability': 1.0 - success_rate,
                        'impact': 'medium'
                    })

            # Performance degradation
            if timing['elapsed'] > 60:  # After 1 minute
                recent_events = [e for node in self.tree_builder.nodes.values()
                                 for e in node.llm_calls + node.sub_events
                                 if e.timestamp > time.time() - 30]

                if len(recent_events) < 5:  # Low activity
                    predictions['potential_issues'].append({
                        'description': 'System activity has decreased - possible performance issue',
                        'probability': 0.6,
                        'impact': 'low'
                    })

            # Overall confidence
            if predictions['next_steps']:
                avg_confidence = sum(step['confidence'] for step in predictions['next_steps']) / len(
                    predictions['next_steps'])
                predictions['confidence_level'] = avg_confidence

            return predictions

        except Exception as e:
            if self.mode == VerbosityMode.DEBUG:
                print(f"⚠️ Real metrics prediction error: {e}")
            return {'next_steps': [], 'estimated_completion': None, 'potential_issues': [], 'confidence_level': 0.0}

    def _predict_additional_cost_real(self, summary: Dict[str, Any], time_remaining: float,
                                      remaining_methods: List[str]) -> float:
        """Predict additional cost based on real method costs and remaining work"""
        try:
            # Calculate average cost per method from real data
            method_costs = {}

            for node in self.tree_builder.nodes.values():
                for event in node.llm_calls:
                    if (hasattr(event, 'metadata') and event.metadata and
                        event.metadata.get('meta_tool_name') and event.llm_cost):

                        method = event.metadata['meta_tool_name']
                        cost = event.llm_cost

                        if method not in method_costs:
                            method_costs[method] = []
                        method_costs[method].append(cost)

            # Estimate cost for remaining methods
            total_estimated_cost = 0.0
            for method in remaining_methods:
                if method in method_costs:
                    avg_cost = sum(method_costs[method]) / len(method_costs[method])
                    total_estimated_cost += avg_cost
                else:
                    # Default estimates based on method complexity
                    default_costs = {
                        'internal_reasoning': 0.002,
                        'delegate_to_llm_tool_node': 0.005,
                        'create_and_execute_plan': 0.010,
                        'direct_response': 0.003
                    }
                    total_estimated_cost += default_costs.get(method, 0.002)

            return total_estimated_cost

        except Exception:
            return 0.0

    def _create_enhanced_header(self, summary: Dict[str, Any]) -> Panel:
        """
        Enhanced header using detailed activity detection and tool usage information.
        """
        try:
            session_info = summary["session_info"]
            timing = summary["timing"]
            health = summary["health_indicators"]
            perf = summary["performance_metrics"]

            # Get detailed current activity
            activity_info = self._get_detailed_current_activity()
            tool_usage = self._get_tool_usage_summary()
            outline_info = self._get_current_outline_info()

            header_content = []

            # Main status line with detailed activity
            status_parts = []

            if session_info["active_nodes"] > 0:
                # Show detailed current activity
                primary_activity = activity_info['primary_activity']
                confidence = activity_info.get('confidence', 0.0)
                time_in_activity = activity_info.get('time_in_current_activity', 0)

                # Activity with confidence and timing
                activity_display = primary_activity
                if confidence > 0.8:
                    activity_display = f"🎯 {primary_activity}"
                elif confidence > 0.5:
                    activity_display = f"🔄 {primary_activity}"
                else:
                    activity_display = f"❓ {primary_activity}"

                if time_in_activity < 60:
                    activity_display += f" ({time_in_activity:.0f}s)"
                else:
                    activity_display += f" ({time_in_activity / 60:.1f}m)"

                status_parts.append(activity_display)
            elif session_info["failed_nodes"] > 0:
                status_parts.append(f"❌ ERRORS (@{self.agent_name})")
            elif session_info["completed_nodes"] == session_info["total_nodes"] and session_info["total_nodes"] > 0:
                status_parts.append(f"✅ COMPLETED (@{self.agent_name})")
            else:
                status_parts.append(f"▶️ RUNNING (@{self.agent_name})")

            header_content.append(" | ".join(status_parts))

            # Detailed activity description line
            if activity_info['detailed_description'] != 'System status unclear':
                detail_line = f"📋 {activity_info['detailed_description']}"
                if len(detail_line) > 100:
                    detail_line = detail_line[:97] + "..."
                header_content.append(detail_line)

            # Progress and context line
            progress_parts = []

            # Outline progress with real data
            if outline_info.get('outline_created') and outline_info.get('total_steps', 0) > 0:
                current_step = outline_info['current_step']
                total_steps = outline_info['total_steps']
                completed_steps = len(outline_info.get('completed_steps', []))
                completion_pct = outline_info.get('completion_percentage', 0)

                progress_parts.append(f"📋 Outline: {completed_steps}/{total_steps} ({completion_pct:.0f}%)")

                # Current step with location context
                location_context = activity_info.get('location_context', '')
                if location_context and location_context != 'Unknown phase':
                    progress_parts.append(f"📍 {location_context}")
            else:
                # Fallback to node progress
                if session_info["total_nodes"] > 0:
                    progress_ratio = session_info["completed_nodes"] / session_info["total_nodes"]
                    progress_percent = int(progress_ratio * 100)
                    progress_parts.append(
                        f"📊 Nodes: {session_info['completed_nodes']}/{session_info['total_nodes']} ({progress_percent}%)")

            # Tool usage information
            if tool_usage['tools_active']:
                active_tools = list(tool_usage['tools_active'])[:3]  # Show up to 3 active tools
                tools_display = f"🔧 Using: {', '.join(active_tools)}"
                if len(tool_usage['tools_active']) > 3:
                    tools_display += f" +{len(tool_usage['tools_active']) - 3} more"
                progress_parts.append(tools_display)
            elif tool_usage['tools_used']:
                used_count = len(tool_usage['tools_used'])
                available_count = len(tool_usage['tools_available'])
                progress_parts.append(f"🔧 Tools: {used_count} used, {available_count} available")

            if progress_parts:
                header_content.append(" | ".join(progress_parts))

            # Timing and performance line
            timing_parts = []

            elapsed = timing["elapsed"]
            if elapsed < 60:
                timing_str = f"Runtime: {elapsed:.1f}s"
            elif elapsed < 3600:
                minutes = int(elapsed // 60)
                seconds = elapsed % 60
                timing_str = f"Runtime: {minutes}m{seconds:.0f}s"
            else:
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                timing_str = f"Runtime: {hours}h{minutes}m"

            # Add ETA from real outline progress
            if outline_info.get('estimated_completion'):
                eta = outline_info['estimated_completion']
                if eta > 0:
                    if eta < 60:
                        timing_str += f" | ETA: {eta:.0f}s"
                    else:
                        eta_min = int(eta // 60)
                        timing_str += f" | ETA: {eta_min}m{eta % 60:.0f}s"

            timing_parts.append(timing_str)

            # Expected next action
            if activity_info.get('expected_next_action') and activity_info['expected_next_action'] != 'Uncertain':
                next_action = activity_info['expected_next_action']
                if len(next_action) > 50:
                    next_action = next_action[:47] + "..."
                timing_parts.append(f"➡️ Next: {next_action}")

            # Performance indicators
            if perf["error_count"] > 0:
                error_rate = perf["error_count"] / max(perf["total_events"], 1)
                timing_parts.append(f"⚠️ Errors: {perf['error_count']} ({error_rate:.1%})")

            # Resource usage
            if perf["total_cost"] > 0:
                timing_parts.append(f"💰 {self._format_cost(perf['total_cost'])}")

            # Health indicator
            health_emoji = "🟢" if health["overall_health"] > 0.8 else "🟡" if health["overall_health"] > 0.5 else "🔴"
            timing_parts.append(f"{health_emoji} {health['overall_health']:.0%}")

            header_content.append(" | ".join(timing_parts))

            # Determine header style based on activity confidence
            if session_info["failed_nodes"] > 0:
                header_style = "red"
            elif activity_info.get('confidence', 0) > 0.8:
                header_style = "green"
            elif session_info["active_nodes"] > 0:
                header_style = "blue"
            else:
                header_style = "cyan"

            return Panel(
                "\n".join(header_content),
                title=f"📊 Live Activity Status - Update #{self._print_counter}",
                style=header_style,
                box=box.ROUNDED
            )

        except Exception as e:
            return Panel(
                f"⚠️ Activity tracking error: {e}\nRuntime: {time.time() - self.tree_builder.start_time:.1f}s",
                title="📊 Status",
                style="red"
            )

    def _infer_outline_from_execution_pattern(self, outline_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Infer outline structure from execution patterns when no explicit outline is found.
        Creates a synthetic outline based on meta-tool usage and system activity.
        """
        try:
            # Look for common execution patterns
            meta_tool_sequence = []
            node_progression = []

            # Analyze recent meta-tool activity
            for node in self.tree_builder.nodes.values():
                for event in node.llm_calls + node.sub_events:
                    if (hasattr(event, 'metadata') and event.metadata and
                        event.metadata.get('meta_tool_name')):
                        meta_tool_sequence.append({
                            'tool': event.metadata['meta_tool_name'],
                            'timestamp': event.timestamp,
                            'success': event.success
                        })

                if node.name not in node_progression:
                    node_progression.append(node.name)

            # Sort by timestamp
            meta_tool_sequence.sort(key=lambda x: x['timestamp'])

            # Create synthetic outline based on patterns
            synthetic_steps = []

            # Pattern 1: Direct response (simple query)
            if any(tool['tool'] == 'direct_response' for tool in meta_tool_sequence):
                synthetic_steps.append({
                    'id': 1,
                    'description': 'Generate direct response to query',
                    'method': 'direct_response',
                    'expected_outcome': 'Complete answer provided',
                    'success_criteria': 'User query fully addressed',
                    'status': 'completed' if any(tool['tool'] == 'direct_response' and tool['success'] for tool in
                                                 meta_tool_sequence) else 'active'
                })

            # Pattern 2: Reasoning + Planning + Execution
            else:
                step_id = 1

                # Check for reasoning phase
                if any(tool['tool'] == 'internal_reasoning' for tool in meta_tool_sequence):
                    synthetic_steps.append({
                        'id': step_id,
                        'description': 'Analyze query and develop reasoning approach',
                        'method': 'internal_reasoning',
                        'expected_outcome': 'Clear understanding and approach',
                        'success_criteria': 'Problem decomposed and approach defined',
                        'status': 'completed'
                    })
                    step_id += 1

                # Check for delegation phase
                if any(tool['tool'] == 'delegate_to_llm_tool_node' for tool in meta_tool_sequence):
                    synthetic_steps.append({
                        'id': step_id,
                        'description': 'Execute tool-based operations',
                        'method': 'delegate_to_llm_tool_node',
                        'expected_outcome': 'Required data gathered and processed',
                        'success_criteria': 'All necessary information obtained',
                        'status': 'completed' if any(
                            tool['tool'] == 'delegate_to_llm_tool_node' and tool['success'] for tool in
                            meta_tool_sequence) else 'active'
                    })
                    step_id += 1

                # Check for planning phase
                if any(tool['tool'] == 'create_and_execute_plan' for tool in meta_tool_sequence):
                    synthetic_steps.append({
                        'id': step_id,
                        'description': 'Create and execute comprehensive plan',
                        'method': 'create_and_execute_plan',
                        'expected_outcome': 'Complex task completed systematically',
                        'success_criteria': 'All plan objectives achieved',
                        'status': 'completed' if any(
                            tool['tool'] == 'create_and_execute_plan' and tool['success'] for tool in
                            meta_tool_sequence) else 'active'
                    })
                    step_id += 1

                # Final response step
                synthetic_steps.append({
                    'id': step_id,
                    'description': 'Synthesize results and provide comprehensive response',
                    'method': 'direct_response',
                    'expected_outcome': 'Complete answer to user query',
                    'success_criteria': 'User query fully addressed',
                    'status': 'pending'
                })

            # Update outline_info with synthetic data
            outline_info['steps'] = synthetic_steps
            outline_info['total_steps'] = len(synthetic_steps)
            outline_info['outline_created'] = True  # Mark as created (synthetic)

            # Determine current step based on completed activities
            completed_tools = [tool['tool'] for tool in meta_tool_sequence if tool['success']]

            for step in synthetic_steps:
                if step['method'] in completed_tools:
                    if step['id'] not in outline_info['completed_steps']:
                        outline_info['completed_steps'].append(step['id'])
                    step['status'] = 'completed'
                elif step['status'] == 'active':
                    outline_info['current_step'] = step['id']
                    break
            else:
                # If no active step found, set current to first pending
                for step in synthetic_steps:
                    if step['status'] == 'pending':
                        outline_info['current_step'] = step['id']
                        step['status'] = 'active'
                        break

            # Build step descriptions dictionary
            outline_info['step_descriptions'] = {
                step['id']: step['description'] for step in synthetic_steps
            }

            return outline_info

        except Exception as e:
            if self.mode == VerbosityMode.DEBUG:
                print(f"⚠️ Error inferring outline from execution pattern: {e}")
            return outline_info

    def _create_live_status_bar(self, summary: Dict[str, Any]) -> Panel:
        """
        Enhanced live status bar with detailed activity information.
        """
        try:
            activity_info = self._get_detailed_current_activity()
            tool_usage = self._get_tool_usage_summary()
            outline_info = self._get_current_outline_info()

            status_items = []

            # Current activity with confidence indicator
            primary_activity = activity_info.get('primary_activity', 'Unknown')
            confidence = activity_info.get('confidence', 0.0)

            if confidence > 0.8:
                confidence_icon = "🎯"
            elif confidence > 0.5:
                confidence_icon = "🔄"
            else:
                confidence_icon = "❓"

            activity_brief = primary_activity
            if len(activity_brief) > 25:
                activity_brief = activity_brief[:22] + "..."

            status_items.append(f"{confidence_icon} {activity_brief}")

            # Progress with real outline data
            if outline_info.get('outline_created'):
                completed = len(outline_info.get('completed_steps', []))
                total = outline_info.get('total_steps', 0)
                if total > 0:
                    # Mini progress bar
                    bar_length = 8
                    filled = int(bar_length * (completed / total))
                    bar = "▰" * filled + "▱" * (bar_length - filled)
                    status_items.append(f"{bar} {completed}/{total}")
            else:
                # Fallback to node progress
                session_info = summary["session_info"]
                if session_info["total_nodes"] > 0:
                    progress_ratio = session_info["completed_nodes"] / session_info["total_nodes"]
                    bar_length = 8
                    filled = int(bar_length * progress_ratio)
                    bar = "▰" * filled + "▱" * (bar_length - filled)
                    status_items.append(f"{bar} {session_info['completed_nodes']}/{session_info['total_nodes']}")

            # Tool status
            if tool_usage['tools_active']:
                tool_count = len(tool_usage['tools_active'])
                status_items.append(f"🔧 {tool_count} active")
            elif tool_usage['tools_used']:
                used_count = len(tool_usage['tools_used'])
                status_items.append(f"🔧 {used_count} used")

            # Timing
            elapsed = summary["timing"]["elapsed"]
            time_str = f"{elapsed:.0f}s" if elapsed < 60 else f"{elapsed // 60:.0f}m{elapsed % 60:.0f}s"
            status_items.append(f"⏱️ {time_str}")

            # Expected next action (if space allows)
            expected_next = activity_info.get('expected_next_action', '')
            if expected_next and expected_next != 'Uncertain' and len(' | '.join(status_items)) < 80:
                next_brief = expected_next[:20] + "..." if len(expected_next) > 20 else expected_next
                status_items.append(f"➡️ {next_brief}")

            # Error indicator
            error_count = summary["performance_metrics"]["error_count"]
            if error_count > 0:
                status_items.append(f"⚠️ {error_count}")

            status_text = " | ".join(status_items)

            # Style based on confidence and activity
            if error_count > 0:
                style = "red dim"
            elif confidence > 0.8:
                style = "green dim"
            elif activity_info['primary_activity'] != 'Unknown':
                style = "blue dim"
            else:
                style = "yellow dim"

            return Panel(
                status_text,
                style=style,
                box=box.MINIMAL,
                height=3
            )

        except Exception as e:
            return Panel(f"Status: Active | ⚠️ {str(e)[:30]}...", style="red dim", box=box.MINIMAL, height=3)

    def _get_current_operations_summary(self, summary: Dict[str, Any]) -> Optional[str]:
        """
        Enhanced operations summary using detailed activity detection.
        """
        try:
            activity_info = self._get_detailed_current_activity()
            tool_usage = self._get_tool_usage_summary()

            operations = []

            # Primary activity
            if activity_info['primary_activity'] != 'Unknown':
                primary = activity_info['primary_activity']
                time_in_activity = activity_info.get('time_in_current_activity', 0)

                if time_in_activity > 0:
                    if time_in_activity < 60:
                        time_str = f"{time_in_activity:.0f}s"
                    else:
                        time_str = f"{time_in_activity / 60:.1f}m"
                    operations.append(f"{primary} ({time_str})")
                else:
                    operations.append(primary)

            # Tool operations
            if tool_usage['current_tool_operation']:
                operations.append(tool_usage['current_tool_operation'])
            elif tool_usage['tools_active']:
                active_tools = list(tool_usage['tools_active'])[:2]
                operations.append(f"Using {', '.join(active_tools)}")

            # Location context
            location_context = activity_info.get('location_context', '')
            if location_context and location_context not in ['Unknown phase', 'Error state']:
                operations.append(f"@ {location_context}")

            if operations:
                summary_text = " | ".join(operations)
                if len(summary_text) > 120:
                    summary_text = summary_text[:117] + "..."
                return summary_text

            return None

        except Exception as e:
            return f"Operations tracking error: {str(e)[:50]}..."

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
            print(f"Runtime: {human_readable_time(timing['elapsed'])}")
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
        """Enhanced delegation updates with detailed task descriptions and tool usage"""
        delegated_task = metadata.get("delegated_task_description", "")
        tools_list = metadata.get("tools_list", [])
        tools_count = metadata.get("tools_count", 0)
        execution_phase = metadata.get("execution_phase", "")
        outline_step = metadata.get("outline_step", 0)

        # Get tool usage context
        tool_usage = self._get_tool_usage_summary()

        time_str = timestamp.strftime("%H:%M:%S")

        if execution_phase == "meta_tool_start":
            if self.mode == VerbosityMode.VERBOSE or self.mode == VerbosityMode.DEBUG:
                # Show detailed task description
                task_preview = delegated_task[:120] + "..." if len(delegated_task) > 120 else delegated_task

                delegation_content = []
                delegation_content.append(f"📄 Task: {task_preview}")

                if outline_step > 0:
                    delegation_content.append(f"📍 Outline Step: {outline_step}")

                # Show available tools with success rates
                if tools_list:
                    tools_with_rates = []
                    for tool in tools_list[:6]:  # Show up to 6 tools
                        if tool in tool_usage['tool_success_rate']:
                            rate_info = tool_usage['tool_success_rate'][tool]
                            success_rate = rate_info['success'] / max(rate_info['total'], 1)
                            if success_rate == 1.0:
                                tools_with_rates.append(f"{tool} ✓")
                            elif success_rate > 0.8:
                                tools_with_rates.append(f"{tool} ({success_rate:.0%})")
                            else:
                                tools_with_rates.append(f"{tool} ⚠️{success_rate:.0%}")
                        else:
                            tools_with_rates.append(tool)

                    if len(tools_list) > 6:
                        tools_with_rates.append(f"+{len(tools_list) - 6} more")

                    delegation_content.append(f"🔧 Tools: {', '.join(tools_with_rates)}")

                delegation_panel = Panel(
                    "\n".join(delegation_content),
                    title=f"🎯 [{time_str}] Delegating to Tool System",
                    style="green",
                    box=box.ROUNDED
                )
                self.console.print(delegation_panel)

            elif self.mode == VerbosityMode.STANDARD:
                # Compact but informative
                task_brief = delegated_task[:80] + "..." if len(delegated_task) > 80 else delegated_task
                outline_info = f" (Step {outline_step})" if outline_step > 0 else ""

                self.console.print(
                    f"🎯 [{time_str}] Delegating{outline_info}: {task_brief}",
                    style="green"
                )

                # Show tools in a compact way
                if tools_list:
                    successful_tools = [t for t in tools_list if t in tool_usage['tools_used']]
                    tools_display = f"🔧 {len(successful_tools)}/{len(tools_list)} proven tools available"
                    self.console.print(f"   {tools_display}", style="green dim")

            else:  # MINIMAL/REALTIME
                task_brief = delegated_task[:60] + "..." if len(delegated_task) > 60 else delegated_task
                outline_info = f" (S{outline_step})" if outline_step > 0 else ""
                self.console.print(f"🎯 [{time_str}] Delegating{outline_info}: {task_brief}", style="green")

        elif event.success and execution_phase != "meta_tool_start":
            duration = metadata.get("execution_duration", 0)
            duration_str = f" ({duration:.1f}s)" if duration > 0.1 else ""

            # Show completion with tool usage summary
            tools_used_now = len([t for t in tools_list if t in tool_usage['tools_used']])
            completion_text = f"✅ [{time_str}] Delegation completed{duration_str}"

            if outline_step > 0:
                completion_text += f" (Step {outline_step})"

            if tools_used_now > 0:
                completion_text += f" | Used {tools_used_now}/{tools_count} tools"

            self.console.print(completion_text, style="green")

            # Show which specific tools were used (in verbose mode)
            if self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG] and tools_used_now > 0:
                used_tools = [t for t in tools_list if t in tool_usage['tools_used']][:4]
                tools_text = f"   🔧 Used: {', '.join(used_tools)}"
                if len(used_tools) < tools_used_now:
                    tools_text += f" +{tools_used_now - len(used_tools)} more"
                self.console.print(tools_text, style="green dim")

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

    def _get_tool_usage_summary(self) -> Dict[str, Any]:
        """
        Tracks real tool usage from delegation events and tool calls.
        Shows which tools were used, are being used, and are available.
        """
        try:
            tool_usage = {
                'tools_used': set(),
                'tools_active': set(),
                'tools_available': set(),
                'current_tool_operation': None,
                'tool_success_rate': {},
                'delegation_history': []
            }

            # Look for tool usage in recent events
            for node in self.tree_builder.nodes.values():
                # Check direct tool calls
                for tool_event in node.tool_calls:
                    tool_name = tool_event.tool_name
                    if tool_event.tool_success:
                        tool_usage['tools_used'].add(tool_name)

                    # Track success rate
                    if tool_name not in tool_usage['tool_success_rate']:
                        tool_usage['tool_success_rate'][tool_name] = {'success': 0, 'total': 0}
                    tool_usage['tool_success_rate'][tool_name]['total'] += 1
                    if tool_event.tool_success:
                        tool_usage['tool_success_rate'][tool_name]['success'] += 1

                    # Check if currently active (last 30 seconds)
                    if tool_event.timestamp > time.time() - 30:
                        if tool_event.tool_success is None:  # Still running
                            tool_usage['tools_active'].add(tool_name)
                            tool_usage['current_tool_operation'] = f"Using {tool_name}"

                # Check delegation events for tool availability and usage
                for event in node.llm_calls + node.sub_events:
                    if (hasattr(event, 'metadata') and event.metadata and
                        event.metadata.get('meta_tool_name') == 'delegate_to_llm_tool_node'):
                        tools_list = event.metadata.get('tools_list', [])
                        tool_usage['tools_available'].update(tools_list)

                        # Track delegation details
                        delegation_info = {
                            'timestamp': event.timestamp,
                            'task_description': event.metadata.get('delegated_task_description', ''),
                            'tools_requested': tools_list,
                            'success': event.success
                        }
                        tool_usage['delegation_history'].append(delegation_info)

            # Keep only recent delegations (last 10)
            tool_usage['delegation_history'] = tool_usage['delegation_history'][-10:]

            return tool_usage

        except Exception as e:
            return {
                'tools_used': set(), 'tools_active': set(), 'tools_available': set(),
                'current_tool_operation': None, 'tool_success_rate': {}, 'delegation_history': []
            }


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
