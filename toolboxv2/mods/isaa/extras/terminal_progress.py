import json
import time
from dataclasses import dataclass, asdict
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

        # Check for explicit completion signals
        if event.event_type in ["node_exit", "execution_complete", "task_complete"]:
            if event.node_duration:
                self.duration = event.node_duration
                self.end_time = event.timestamp
                self.update_status(NodeStatus.COMPLETED, "Explicit completion signal")
                return

        # Auto-completion for StrategyOrchestratorNode after LLM completion
        if (self.name == "StrategyOrchestratorNode" and
            event.event_type == "llm_call" and
            event.success and
            self.status == NodeStatus.RUNNING):
            self.end_time = event.timestamp
            self.duration = self.end_time - (self.start_time or event.timestamp)
            self.update_status(NodeStatus.COMPLETED, "Auto-detected: LLM call completed")
            return

        # Error-based completion detection
        if event.event_type == "error" or event.success is False:
            print(event.metadata, event.event_type, event.event_id)
            print("="*200)
            self.update_status(NodeStatus.FAILED, "Error detected", {
                "error": event.metadata.get("error", (event.tool_error if hasattr(event, 'tool_error') else "Unknown error") or "Unknown error"),
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
            if event.event_type == "error" or event.success is False:
                error_entry = {
                    "timestamp": event.timestamp,
                    "node": node_name,
                    "event_type": event.event_type,
                    "error": event.metadata.get("error", "Unknown error"),
                    "error_type": event.metadata.get("error_type", "UnknownError"),
                    "retry_count": node.retry_count
                }
                self.error_log.append(error_entry)

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


class ProgressiveTreePrinter:
    """Production-ready progressive tree printer with enhanced features"""

    def __init__(self, mode: VerbosityMode = VerbosityMode.STANDARD, use_rich: bool = True,
                 auto_refresh: bool = True, max_history: int = 1000):
        self.mode = mode
        self.use_rich = use_rich and RICH_AVAILABLE
        self.auto_refresh = auto_refresh
        self.max_history = max_history

        self.tree_builder = ExecutionTreeBuilder()
        self.print_history: List[Dict[str, Any]] = []

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

    def _should_print_update(self) -> bool:
        """Enhanced decision logic for when to print updates"""
        current_time = time.time()

        # Rate limiting - don't print too frequently
        if current_time - self._last_update_time < 2.5 and self.mode != VerbosityMode.REALTIME:
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
                # Only major state changes
                should_update = (current_hash != self._last_print_hash and
                                 (current_state["completed_nodes"] !=
                                  getattr(self, '_last_completed_count', 0) or
                                  current_state["failed_nodes"] !=
                                  getattr(self, '_last_failed_count', 0)))

                self._last_completed_count = current_state["completed_nodes"]
                self._last_failed_count = current_state["failed_nodes"]

            elif self.mode == VerbosityMode.REALTIME:
                # Always update in realtime mode
                should_update = True

            elif self.mode in [VerbosityMode.STANDARD, VerbosityMode.VERBOSE]:
                # Regular updates on significant changes
                should_update = current_hash != self._last_print_hash

            else:  # DEBUG mode
                # Update on every event
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
            if self.tree_builder.error_log and self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                self._add_error_log_branch(tree)

            # Performance metrics
            if self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                self._add_performance_branch(tree, summary)

            # Routing history
            if (self.tree_builder.routing_history and
                self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]):
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
                    node_text += " ⚡"
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
        """Add error log branch"""
        if not self.tree_builder.error_log:
            return

        error_branch = tree.add(f"❌ Error Log ({len(self.tree_builder.error_log)})", style="red bold")

        # Show recent errors
        recent_errors = self.tree_builder.error_log[-5:]  # Last 5 errors
        for error in recent_errors:
            timestamp = datetime.fromtimestamp(error["timestamp"]).strftime("%H:%M:%S")
            error_text = f"[{timestamp}] {error['node']}: {error['error']}"
            if error.get('retry_count', 0) > 0:
                error_text += f" (Retry #{error['retry_count']})"
            error_branch.add(error_text, style="red dim")

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
        """Print tree update with error handling"""
        try:
            if self._fallback_mode:
                self._print_fallback()
                return

            if not self.use_rich:
                self._print_fallback()
                return

            self._print_counter += 1
            summary = self.tree_builder.get_execution_summary()

            # Clear screen in realtime mode
            if self.mode == VerbosityMode.REALTIME and self._print_counter > 1:
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

        status_text = " | ".join(status_parts)

        # Progress info
        progress_text = (f"Progress: {session_info['completed_nodes']}/{session_info['total_nodes']} "
                         f"({health['completion_rate']:.1%})")

        # Timing info
        timing_text = f"Runtime: {timing['elapsed']:.1f}s"
        if timing["estimated_completion"]:
            eta = timing["estimated_completion"] - time.time()
            if eta > 0:
                timing_text += f" | ETA: {eta:.0f}s"

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
            print(f"Total events processed: {self.tree_builder.total_events}")
            print(f"Nodes: {len(self.tree_builder.nodes)}")
            print(f"Errors encountered: {len(self.tree_builder.error_log)}")
            if e:
                print(f"Print error: {e}")

    async def progress_callback(self, event: ProgressEvent):
        """Main progress callback with comprehensive error handling"""
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

            # Print debug info in debug mode
            if self.mode == VerbosityMode.DEBUG:
                self._print_debug_event(event)

            # Decide whether to print update
            if self._should_print_update():
                self._print_tree_update()

        except Exception as e:
            # Emergency error handling
            self._consecutive_errors += 1
            print(f"⚠️  Progress callback error #{self._consecutive_errors}: {e}")

            if self._consecutive_errors > self._error_threshold:
                print("🚨 Progress printing disabled due to excessive errors")
                # Disable further callbacks
                self.progress_callback = self._noop_callback

    def _print_debug_event(self, event: ProgressEvent):
        """Print individual event details in debug mode"""
        timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S.%f")[:-3]

        if self.use_rich:
            debug_text = f"[{timestamp}] {event.event_type.upper()} - {event.node_name}"
            if event.success is not None:
                success_icon = "✅" if event.success else "❌"
                debug_text += f" {success_icon}"
            self.console.print(debug_text, style="dim")
        else:
            print(f"[{timestamp}] {event.event_type.upper()} - {event.node_name}")

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
    """Comprehensive demo of the enhanced progress printer"""
    import asyncio

    print("🚀 Starting Enhanced Progress Printer Demo...")
    print("Choose verbosity mode:")
    print("1. MINIMAL - Only major updates")
    print("2. STANDARD - Regular updates")
    print("3. VERBOSE - Detailed information")
    print("4. DEBUG - Full debugging info")
    print("5. REALTIME - Live updates with progress")

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

    print(f"Demo running in {mode.value.upper()} mode...")
    await asyncio.sleep(1)

    # Simulate comprehensive agent execution
    events = [
        # Execution start
        ProgressEvent(
            event_type="execution_start",
            timestamp=time.time(),
            node_name="FlowAgent",
            session_id="demo_session",
            metadata={"query": "Analyze the current market trends", "user_id": "demo_user"}
        ),

        # Strategy selection
        ProgressEvent(
            event_type="node_enter",
            timestamp=time.time(),
            node_name="StrategyOrchestratorNode"
        ),
        ProgressEvent(
            event_type="llm_call",
            timestamp=time.time(),
            node_name="StrategyOrchestratorNode",
            llm_model="gpt-4",
            llm_total_tokens=1200,
            llm_cost=0.024,
            llm_duration=1.5,
            success=True,
            metadata={
                "strategy": "research_and_analyze",
                "reasoning": "Complex query requires multi-source research",
                "estimated_complexity": "high"
            }
        ),

        # Task planning
        ProgressEvent(
            event_type="node_enter",
            timestamp=time.time(),
            node_name="TaskPlannerNode"
        ),
        ProgressEvent(
            event_type="llm_call",
            timestamp=time.time(),
            node_name="TaskPlannerNode",
            llm_model="gpt-3.5-turbo",
            llm_total_tokens=800,
            llm_cost=0.0016,
            llm_duration=0.8,
            success=True
        ),

        # Task execution with tools
        ProgressEvent(
            event_type="node_enter",
            timestamp=time.time(),
            node_name="TaskExecutorNode"
        ),
        ProgressEvent(
            event_type="tool_call",
            timestamp=time.time(),
            node_name="TaskExecutorNode",
            tool_name="web_search",
            tool_args={"query": "current market trends 2024"},
            tool_duration=2.3,
            tool_success=True,
            tool_result="Found 15 relevant articles"
        ),

        # Simulated error
        ProgressEvent(
            event_type="tool_call",
            timestamp=time.time(),
            node_name="TaskExecutorNode",
            tool_name="financial_api",
            tool_args={"symbol": "SPY"},
            tool_duration=5.0,
            tool_success=False,
            tool_error="API rate limit exceeded",
            success=False,
            metadata={"error": "Rate limit exceeded", "error_type": "APIError"}
        ),

        # Recovery with alternative approach
        ProgressEvent(
            event_type="tool_call",
            timestamp=time.time(),
            node_name="TaskExecutorNode",
            tool_name="alternative_data_source",
            tool_args={"query": "market data"},
            tool_duration=1.2,
            tool_success=True,
            tool_result="Retrieved market data from backup source"
        ),

        # LLM analysis
        ProgressEvent(
            event_type="llm_call",
            timestamp=time.time(),
            node_name="LLMToolNode",
            llm_model="gpt-4",
            llm_total_tokens=2500,
            llm_cost=0.05,
            llm_duration=3.2,
            success=True
        ),

        # Response generation
        ProgressEvent(
            event_type="node_enter",
            timestamp=time.time(),
            node_name="ResponseGenerationNode"
        ),
        ProgressEvent(
            event_type="llm_call",
            timestamp=time.time(),
            node_name="ResponseGenerationNode",
            llm_model="gpt-4",
            llm_total_tokens=1800,
            llm_cost=0.036,
            llm_duration=2.1,
            success=True
        ),

        # Completion
        ProgressEvent(
            event_type="execution_complete",
            timestamp=time.time(),
            node_name="FlowAgent",
            node_duration=15.2,
            success=True,
            metadata={"result_length": 1247, "user_satisfaction": 0.95}
        )
    ]

    print(f"\n🎬 Simulating {len(events)} events...")

    for i, event in enumerate(events):
        await printer.progress_callback(event)

        # Variable delay to simulate real execution
        if mode == VerbosityMode.REALTIME:
            await asyncio.sleep(0.3)
        else:
            await asyncio.sleep(0.8)

        # Show progress
        if i % 3 == 0:
            print(f"Progress: {i + 1}/{len(events)} events processed")

    print("\n🏁 Demo completed!")

    # Print final summary
    printer.print_final_summary()

    # Export summary
    if input("\nExport execution summary? (y/n): ").lower().startswith('y'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"execution_summary_{timestamp}.json"
        summary = printer.export_summary(filename)
        print(f"📁 Summary exported to: {filename}")
        print(f"📊 Total nodes: {len(summary['detailed_nodes'])}")
        print(f"💰 Total cost: ${summary['performance_metrics']['total_cost']:.4f}")


if __name__ == "__main__":
    print("🔧 Enhanced CLI Progress Printing System")
    print("=" * 50)

    # Run the demo
    import asyncio

    try:
        asyncio.run(demo_enhanced_printer())
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
