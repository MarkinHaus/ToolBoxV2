import json
import time
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from typing import Any

try:
    from rich import box
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: Rich not available. Install with: pip install rich")

from toolboxv2.mods.isaa.base.Agent.types import (
    LLMTask,
    NodeStatus,
    ProgressEvent,
    TaskPlan,
    ToolTask, ChainMetadata,
)


class VerbosityMode(Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    VERBOSE = "verbose"
    DEBUG = "debug"
    REALTIME = "realtime"


class DualTrackState:
    """Manages the dual-track system: Progress Track + System Track"""

    def __init__(self):
        # Progress Track (What the Agent Does)
        self.semantic_progress = {
            'execution_phase': 'starting',  # starting, planning, executing, completing
            'current_outline': None,
            'outline_progress': {'current_step': 0, 'total_steps': 0, 'completed_steps': []},
            'current_reasoning_loop': 0,
            'active_meta_tools': [],
            'task_execution_state': {'total': 0, 'completed': 0, 'failed': 0, 'running': []},
            'llm_interactions': {'total_calls': 0, 'total_cost': 0.0, 'total_tokens': 0}
        }

        # System Track (Where the Agent Is)
        self.system_state = {
            'active_nodes': {},
            'node_flow': [],
            'current_node': None,
            'node_phases': {},  # node_name -> current phase
            'system_health': {'status': 'healthy', 'error_count': 0, 'warnings': []}
        }

        # Cross-track correlations
        self.correlations = {
            'semantic_to_system': {},  # semantic events -> system nodes
            'system_to_semantic': {},  # system nodes -> semantic events
            'timing_correlations': []
        }


class DualTrackEventProcessor:
    """Processes events for both tracking perspectives"""

    def __init__(self):
        self.state = DualTrackState()
        self.event_history = []
        self.start_time = None

    def process_event(self, event: ProgressEvent):
        """Route event to appropriate track processors"""
        if not self.start_time:
            self.start_time = event.timestamp

        self.event_history.append(event)

        # Route to progress track processor
        if self._is_progress_track_event(event):
            self._process_progress_event(event)

        # Route to system track processor
        if self._is_system_track_event(event):
            self._process_system_event(event)

        # Update cross-track correlations
        self._update_correlations(event)

    def _is_progress_track_event(self, event: ProgressEvent) -> bool:
        """Determine if event belongs to progress track"""
        progress_events = {
            'execution_start', 'execution_complete',
            'outline_created', 'plan_created',
            'reasoning_loop', 'meta_tool_analysis',
            'tool_call', 'task_start', 'task_complete', 'task_error',
            'llm_call'
        }
        return event.event_type in progress_events

    def _is_system_track_event(self, event: ProgressEvent) -> bool:
        """Determine if event belongs to system track"""
        system_events = {
            'node_enter', 'node_exit', 'node_phase', 'error'
        }
        return event.event_type in system_events

    def _process_progress_event(self, event: ProgressEvent):
        """Process events in the semantic progress track"""
        if event.event_type == 'execution_start':
            self.state.semantic_progress['execution_phase'] = 'initializing'

        elif event.event_type == 'outline_created':
            # NEW & IMPORTANT - extract outline structure
            outline_data = event.metadata.get('outline') if event.metadata else None
            if outline_data:
                self.state.semantic_progress['current_outline'] = outline_data
                steps = outline_data.get('steps', []) if isinstance(outline_data, dict) else []
                self.state.semantic_progress['outline_progress'] = {
                    'current_step': 1,
                    'total_steps': len(steps),
                    'completed_steps': [],
                    'step_details': steps
                }
            self.state.semantic_progress['execution_phase'] = 'planning'

        elif event.event_type == 'plan_created':
            self.state.semantic_progress['execution_phase'] = 'executing'

        elif event.event_type == 'reasoning_loop':
            loop_num = event.metadata.get('loop_number', 0) if event.metadata else 0
            self.state.semantic_progress['current_reasoning_loop'] = loop_num

        elif event.event_type == 'tool_call':
            is_meta = event.metadata.get('is_meta_tool', False) if event.metadata else False
            tool_name = event.tool_name or 'unknown'

            if is_meta:
                if event.status == 'RUNNING':
                    self.state.semantic_progress['active_meta_tools'].append(tool_name)
                elif event.status in ['COMPLETED', 'FAILED']:
                    if tool_name in self.state.semantic_progress['active_meta_tools']:
                        self.state.semantic_progress['active_meta_tools'].remove(tool_name)

        elif event.event_type in ['task_start', 'task_complete', 'task_error']:
            task_state = self.state.semantic_progress['task_execution_state']
            if event.event_type == 'task_start':
                task_state['running'].append(event.task_id)
                task_state['total'] += 1
            elif event.event_type == 'task_complete':
                if event.task_id in task_state['running']:
                    task_state['running'].remove(event.task_id)
                task_state['completed'] += 1
            elif event.event_type == 'task_error':
                if event.task_id in task_state['running']:
                    task_state['running'].remove(event.task_id)
                task_state['failed'] += 1

        elif event.event_type == 'llm_call':
            llm_state = self.state.semantic_progress['llm_interactions']
            llm_state['total_calls'] += 1
            if event.llm_cost:
                llm_state['total_cost'] += event.llm_cost
            if event.llm_total_tokens:
                llm_state['total_tokens'] += event.llm_total_tokens

        elif event.event_type == 'execution_complete':
            self.state.semantic_progress['execution_phase'] = 'completed'

    def _process_system_event(self, event: ProgressEvent):
        """Process events in the system track"""
        node_name = event.node_name or 'unknown'

        if event.event_type == 'node_enter':
            self.state.system_state['active_nodes'][node_name] = {
                'status': 'active',
                'start_time': event.timestamp,
                'current_phase': 'initializing'
            }
            if node_name not in self.state.system_state['node_flow']:
                self.state.system_state['node_flow'].append(node_name)
            self.state.system_state['current_node'] = node_name

        elif event.event_type == 'node_exit':
            if node_name in self.state.system_state['active_nodes']:
                node_info = self.state.system_state['active_nodes'][node_name]
                node_info['status'] = 'completed' if event.success else 'failed'
                node_info['end_time'] = event.timestamp
                node_info['duration'] = event.node_duration
                # Remove from active
                del self.state.system_state['active_nodes'][node_name]

        elif event.event_type == 'node_phase':
            if node_name in self.state.system_state['active_nodes']:
                self.state.system_state['active_nodes'][node_name]['current_phase'] = event.node_phase
            self.state.system_state['node_phases'][node_name] = event.node_phase

        elif event.event_type == 'error':
            self.state.system_state['system_health']['error_count'] += 1
            error_detail = {
                'timestamp': event.timestamp,
                'node': node_name,
                'error': event.error_details or 'Unknown error'
            }
            self.state.system_state['system_health']['warnings'].append(error_detail)
            if self.state.system_state['system_health']['error_count'] > 5:
                self.state.system_state['system_health']['status'] = 'degraded'

    def _update_correlations(self, event: ProgressEvent):
        """Update cross-track correlations"""
        # Correlate semantic events with system nodes
        if event.node_name and self._is_progress_track_event(event):
            semantic_key = f"{event.event_type}:{event.timestamp}"
            self.state.correlations['semantic_to_system'][semantic_key] = event.node_name

        # Track timing correlations for performance analysis
        if event.node_duration:
            self.state.correlations['timing_correlations'].append({
                'event_type': event.event_type,
                'node_name': event.node_name,
                'duration': event.node_duration,
                'timestamp': event.timestamp
            })

    def get_progress_summary(self) -> dict[str, Any]:
        """Get comprehensive progress summary across both tracks"""
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 0

        return {
            'dual_track_state': {
                'semantic_progress': self.state.semantic_progress.copy(),
                'system_state': self.state.system_state.copy(),
                'correlations_count': {
                    'semantic_to_system': len(self.state.correlations['semantic_to_system']),
                    'timing_data_points': len(self.state.correlations['timing_correlations'])
                }
            },
            'execution_metrics': {
                'total_events': len(self.event_history),
                'elapsed_time': elapsed,
                'events_per_second': len(self.event_history) / max(elapsed, 1),
                'system_health': self.state.system_state['system_health']['status'],
                'error_rate': self.state.system_state['system_health']['error_count'] / max(len(self.event_history), 1)
            },
            'current_activity': self._get_current_activity_summary()
        }

    def _get_current_activity_summary(self) -> dict[str, Any]:
        """Synthesize current activity from both tracks"""
        semantic = self.state.semantic_progress
        system = self.state.system_state

        return {
            'execution_phase': semantic['execution_phase'],
            'current_outline_step': semantic['outline_progress']['current_step'],
            'total_outline_steps': semantic['outline_progress']['total_steps'],
            'outline_completion_percent': (
                len(semantic['outline_progress']['completed_steps']) /
                max(semantic['outline_progress']['total_steps'], 1) * 100
            ),
            'active_reasoning_loop': semantic['current_reasoning_loop'],
            'active_meta_tools': semantic['active_meta_tools'].copy(),
            'running_tasks': len(semantic['task_execution_state']['running']),
            'current_system_node': system['current_node'],
            'active_system_nodes': len(system['active_nodes']),
            'system_health_status': system['system_health']['status']
        }


class EnhancedDisplayRenderer:
    """Renders dual-track information with intelligent display management"""

    def __init__(self, mode: VerbosityMode, use_rich: bool = True):
        self.mode = mode
        self.use_rich = use_rich and RICH_AVAILABLE
        self.console = Console() if self.use_rich else None
        self.last_display_hash = None

    def render_dual_track_display(self, processor: DualTrackEventProcessor) -> str:
        """Main rendering method for dual-track display"""
        summary = processor.get_progress_summary()

        if not self.use_rich:
            return self._render_fallback_display(summary)

        if self.mode == VerbosityMode.MINIMAL:
            return self._render_minimal_display(summary)
        elif self.mode == VerbosityMode.STANDARD:
            return self._render_standard_display(summary)
        elif self.mode == VerbosityMode.VERBOSE:
            return self._render_verbose_display(summary)
        elif self.mode == VerbosityMode.DEBUG:
            return self._render_debug_display(summary)
        elif self.mode == VerbosityMode.REALTIME:
            return self._render_realtime_display(summary)

        return ""

    def _render_minimal_display(self, summary: dict[str, Any]) -> str:
        """Minimal display - just essential progress"""
        activity = summary['current_activity']
        metrics = summary['execution_metrics']

        # Simple status line
        phase = activity['execution_phase'].title()
        if activity['total_outline_steps'] > 0:
            progress = f"{activity['outline_completion_percent']:.0f}%"
            status = f"🤖 {phase} | Step {activity['current_outline_step']}/{activity['total_outline_steps']} | {progress}"
        else:
            status = f"🤖 {phase}"

        if metrics['error_rate'] > 0.1:
            status += f" | ⚠️ {metrics['error_rate']:.1%} errors"

        self.console.print(status, style="cyan")
        return status

    def _render_standard_display(self, summary: dict[str, Any]) -> str:
        """Standard display - balanced detail"""
        activity = summary['current_activity']
        semantic = summary['dual_track_state']['semantic_progress']
        system = summary['dual_track_state']['system_state']

        # Main header
        self.console.print()
        header_content = self._build_standard_header(activity, summary['execution_metrics'])
        header_panel = Panel(header_content, title="🤖 Agent Execution Status", style="cyan", box=box.ROUNDED)
        self.console.print(header_panel)

        # Progress overview
        if semantic['current_outline']:
            progress_content = self._build_outline_progress_display(semantic['outline_progress'])
            progress_panel = Panel(progress_content, title="📋 Execution Outline", style="blue", box=box.ROUNDED)
            self.console.print(progress_panel)

        # Current activity
        current_activity = self._build_current_activity_display(activity, semantic, system)
        activity_panel = Panel(current_activity, title="🔄 Current Activity", style="green", box=box.ROUNDED)
        self.console.print(activity_panel)

        return "standard_display_rendered"

    def _render_verbose_display(self, summary: dict[str, Any]) -> str:
        """Verbose display - detailed dual-track view"""
        # Render standard display first
        self._render_standard_display(summary)

        # Add detailed system state
        system = summary['dual_track_state']['system_state']
        system_content = self._build_system_state_display(system)
        system_panel = Panel(system_content, title="🔧 System State", style="yellow", box=box.ROUNDED)
        self.console.print(system_panel)

        # Add performance metrics
        metrics_content = self._build_metrics_display(summary['execution_metrics'])
        metrics_panel = Panel(metrics_content, title="📊 Performance Metrics", style="magenta", box=box.ROUNDED)
        self.console.print(metrics_panel)

        return "verbose_display_rendered"

    def _render_debug_display(self, summary: dict[str, Any]) -> str:
        """Debug display - full dual-track details"""
        # Render verbose display first
        self._render_verbose_display(summary)

        # Add correlation data
        correlations = summary['dual_track_state']['correlations_count']
        correlation_content = f"Semantic↔System: {correlations['semantic_to_system']} mappings\n"
        correlation_content += f"Timing Data Points: {correlations['timing_data_points']}"

        correlation_panel = Panel(correlation_content, title="🔗 Track Correlations", style="red", box=box.ROUNDED)
        self.console.print(correlation_panel)

        return "debug_display_rendered"

    def _render_realtime_display(self, summary: dict[str, Any]) -> str:
        """Realtime display - live updates"""
        activity = summary['current_activity']

        # Single line live status
        phase = activity['execution_phase']
        step_info = f"step {activity['current_outline_step']}/{activity['total_outline_steps']}" if activity[
                                                                                                        'total_outline_steps'] > 0 else "no outline"

        # Animated spinner
        spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner_idx = int(time.time() * 2) % len(spinner_chars)
        spinner = spinner_chars[spinner_idx]

        status_line = f"\r{spinner} 🤖 {phase.title()} | {step_info} | {activity['outline_completion_percent']:.0f}%"

        if activity['active_meta_tools']:
            tools_str = ",".join(activity['active_meta_tools'][:2])
            status_line += f" | tools:{tools_str}"

        print(status_line, end="", flush=True)
        return status_line

    def _build_standard_header(self, activity: dict[str, Any], metrics: dict[str, Any]) -> str:
        """Build standard header content"""
        lines = []

        # Execution phase with progress
        phase_line = f"Phase: {activity['execution_phase'].title()}"
        if activity['total_outline_steps'] > 0:
            phase_line += f" | Progress: {activity['outline_completion_percent']:.1f}%"
        lines.append(phase_line)

        # Current activity
        activity_parts = []
        if activity['current_outline_step'] > 0:
            activity_parts.append(f"Step {activity['current_outline_step']}/{activity['total_outline_steps']}")
        if activity['active_reasoning_loop'] > 0:
            activity_parts.append(f"Reasoning Loop {activity['active_reasoning_loop']}")
        if activity['active_meta_tools']:
            activity_parts.append(f"Using: {', '.join(activity['active_meta_tools'][:3])}")

        if activity_parts:
            lines.append("Activity: " + " | ".join(activity_parts))

        # System health
        health_line = f"Health: {activity['system_health_status'].title()}"
        if metrics['error_rate'] > 0:
            health_line += f" | Error Rate: {metrics['error_rate']:.1%}"
        health_line += f" | Runtime: {metrics['elapsed_time']:.1f}s"
        lines.append(health_line)

        return "\n".join(lines)

    def _build_outline_progress_display(self, outline_progress: dict[str, Any]) -> str:
        """Build outline progress visualization"""
        if not outline_progress.get('step_details'):
            return "No outline available"

        lines = []
        current_step = outline_progress['current_step']
        completed_steps = set(outline_progress['completed_steps'])

        for i, step_detail in enumerate(outline_progress['step_details'], 1):
            if isinstance(step_detail, dict):
                description = step_detail.get('description', f'Step {i}')
            else:
                description = str(step_detail)

            # Status icon
            if i in completed_steps:
                icon = "✅"
                style = "completed"
            elif i == current_step:
                icon = "🔄"
                style = "current"
            else:
                icon = "⏸️"
                style = "pending"

            # Truncate long descriptions
            if len(description) > 60:
                description = description[:57] + "..."

            lines.append(f"{icon} Step {i}: {description}")

        return "\n".join(lines)

    def _build_current_activity_display(self, activity: dict[str, Any], semantic: dict[str, Any],
                                        system: dict[str, Any]) -> str:
        """Build current activity summary"""
        lines = []

        # Current focus
        if system['current_node']:
            lines.append(f"🎯 Current Node: {system['current_node']}")

        # Active operations
        active_ops = []
        if activity['active_meta_tools']:
            active_ops.extend(activity['active_meta_tools'])
        if activity['running_tasks'] > 0:
            active_ops.append(f"{activity['running_tasks']} running tasks")

        if active_ops:
            lines.append(f"⚙️ Active Operations: {', '.join(active_ops)}")

        # Resource usage
        llm_info = semantic['llm_interactions']
        if llm_info['total_calls'] > 0:
            resource_line = f"💰 LLM: {llm_info['total_calls']} calls"
            if llm_info['total_cost'] > 0:
                resource_line += f", ${llm_info['total_cost']:.4f}"
            if llm_info['total_tokens'] > 0:
                resource_line += f", {llm_info['total_tokens']:,} tokens"
            lines.append(resource_line)

        return "\n".join(lines) if lines else "System initializing..."

    def _build_system_state_display(self, system: dict[str, Any]) -> str:
        """Build detailed system state display"""
        lines = []

        # Active nodes
        if system['active_nodes']:
            lines.append(f"🔄 Active Nodes ({len(system['active_nodes'])}):")
            for node_name, node_info in list(system['active_nodes'].items())[:5]:
                phase = node_info.get('current_phase', 'unknown')
                elapsed = time.time() - node_info.get('start_time', time.time())
                lines.append(f"  • {node_name}: {phase} ({elapsed:.1f}s)")

        # Node execution flow
        if system['node_flow']:
            flow_display = " → ".join(system['node_flow'][-5:])  # Last 5 nodes
            lines.append(f"🔗 Execution Flow: {flow_display}")

        # System health details
        health = system['system_health']
        if health['error_count'] > 0:
            lines.append(f"⚠️ Errors: {health['error_count']}")
            if health['warnings']:
                latest_warning = health['warnings'][-1]
                warning_time = datetime.fromtimestamp(latest_warning['timestamp']).strftime("%H:%M:%S")
                lines.append(f"   Latest: [{warning_time}] {latest_warning['error']}")

        return "\n".join(lines) if lines else "System state nominal"

    def _build_metrics_display(self, metrics: dict[str, Any]) -> str:
        """Build performance metrics display"""
        lines = []

        lines.append(f"📊 Total Events: {metrics['total_events']}")
        lines.append(f"⚡ Processing Rate: {metrics['events_per_second']:.1f} events/sec")
        lines.append(f"⏱️ Runtime: {metrics['elapsed_time']:.2f}s")
        lines.append(f"🏥 System Health: {metrics['system_health']}")

        if metrics['error_rate'] > 0:
            lines.append(f"❌ Error Rate: {metrics['error_rate']:.2%}")

        return "\n".join(lines)

    def _render_fallback_display(self, summary: dict[str, Any]) -> str:
        """Fallback display without Rich"""
        activity = summary['current_activity']
        metrics = summary['execution_metrics']

        print(f"\n{'=' * 60}")
        print("🤖 AGENT EXECUTION STATUS")
        print(f"{'=' * 60}")
        print(f"Phase: {activity['execution_phase'].title()}")
        if activity['total_outline_steps'] > 0:
            print(
                f"Progress: {activity['outline_completion_percent']:.1f}% (Step {activity['current_outline_step']}/{activity['total_outline_steps']})")
        print(f"Health: {activity['system_health_status'].title()}")
        print(f"Runtime: {metrics['elapsed_time']:.1f}s")
        print(f"Events: {metrics['total_events']} ({metrics['events_per_second']:.1f}/sec)")
        if metrics['error_rate'] > 0:
            print(f"Error Rate: {metrics['error_rate']:.1%}")
        print(f"{'=' * 60}")

        return "fallback_display_rendered"


class ProgressiveTreePrinter:
    """Production-ready progressive tree printer with dual-track event processing"""

    def __init__(self, mode: VerbosityMode = VerbosityMode.STANDARD, use_rich: bool = True,
                 auto_refresh: bool = True, max_history: int = 1000, **kwargs):
        self.mode = mode
        self.use_rich = use_rich and RICH_AVAILABLE
        self.auto_refresh = auto_refresh
        self.max_history = max_history

        # Initialize dual-track processor
        self.event_processor = DualTrackEventProcessor()
        self.display_renderer = EnhancedDisplayRenderer(mode, use_rich)

        # Display management
        self._last_display_time = 0
        self._display_interval = self._get_display_interval()
        self._consecutive_errors = 0
        self._error_threshold = 5

        # Session tracking
        self.agent_name = "FlowAgent"
        self.session_id = None
        self._print_counter = 0

        # Accumulated runs tracking
        self._accumulated_runs = []
        self._current_run_id = 0
        self._global_start_time = time.time()

        # Rich console setup (if available)
        if self.use_rich:
            self.console = Console(record=True)

    def flush(self, run_name: str = None) -> dict[str, Any]:
        """Enhanced flush with dual-track state management"""
        try:
            current_time = time.time()
            if run_name is None:
                run_name = f"run_{self._current_run_id + 1}"

            # Generate comprehensive run data using dual-track system
            summary = self.event_processor.get_progress_summary()

            # Create comprehensive run data
            run_data = {
                "run_id": self._current_run_id + 1,
                "run_name": run_name,
                "flush_timestamp": current_time,
                "dual_track_summary": summary,
                "execution_events": self.event_processor.event_history.copy(),
                "semantic_progress": summary['dual_track_state']['semantic_progress'].copy(),
                "system_state": summary['dual_track_state']['system_state'].copy(),
                "execution_metrics": summary['execution_metrics'].copy(),
                "current_activity": summary['current_activity'].copy(),
                "print_counter": self._print_counter,
                "agent_name": self.agent_name,
                "session_id": self.session_id
            }

            # Add detailed execution flow analysis
            run_data["execution_analysis"] = {
                "outline_completion_rate": summary['current_activity']['outline_completion_percent'] / 100,
                "reasoning_loops_count": summary['current_activity']['active_reasoning_loop'],
                "system_node_count": len(summary['dual_track_state']['system_state']['node_flow']),
                "error_density": summary['execution_metrics']['error_rate'],
                "processing_efficiency": summary['execution_metrics']['events_per_second']
            }

            # Store in accumulated runs
            self._accumulated_runs.append(run_data)

            # Reset for fresh execution
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

    def print_final_summary(self):
        """Print comprehensive final summary with dual-track analysis"""
        try:
            if not self.use_rich:
                self._print_summary_fallback(self.event_processor.get_progress_summary())
                return

            summary = self.event_processor.get_progress_summary()

            # Clear display and show completion
            self.console.print()
            self.console.print("🎉 [bold green]EXECUTION COMPLETED[/bold green] 🎉")

            # Final dual-track display
            self.display_renderer.render_dual_track_display(self.event_processor)

            # Comprehensive summary table
            self._print_comprehensive_final_table(summary)

            # Performance analysis
            if self.mode in [VerbosityMode.VERBOSE, VerbosityMode.DEBUG]:
                self._print_dual_track_performance_analysis(summary)

        except Exception as e:
            print(f"⚠️ Error printing final summary: {e}")
            self._print_summary_fallback(self.event_processor.get_progress_summary())

    def get_accumulated_summary(self) -> dict[str, Any]:
        """Get comprehensive summary of all accumulated runs with dual-track metrics"""
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
            total_duration = 0.0
            total_outline_steps = 0
            total_reasoning_loops = 0

            run_summaries = []

            for run in self._accumulated_runs:
                # Handle both old and new run data formats
                if 'dual_track_summary' in run:
                    # New dual-track format
                    summary = run['dual_track_summary']
                    semantic = summary['dual_track_state']['semantic_progress']
                    metrics = summary['execution_metrics']

                    total_cost += semantic['llm_interactions']['total_cost']
                    total_tokens += semantic['llm_interactions']['total_tokens']
                    total_events += metrics['total_events']
                    total_errors += summary['dual_track_state']['system_state']['system_health']['error_count']
                    total_duration += metrics['elapsed_time']
                    total_outline_steps += semantic['outline_progress']['total_steps']
                    total_reasoning_loops += semantic['current_reasoning_loop']

                    run_summaries.append({
                        "run_id": run["run_id"],
                        "run_name": run["run_name"],
                        "duration": metrics['elapsed_time'],
                        "events": metrics['total_events'],
                        "cost": semantic['llm_interactions']['total_cost'],
                        "tokens": semantic['llm_interactions']['total_tokens'],
                        "errors": summary['dual_track_state']['system_state']['system_health']['error_count'],
                        "outline_completion": summary['current_activity']['outline_completion_percent'],
                        "reasoning_loops": semantic['current_reasoning_loop'],
                        "system_health": summary['current_activity']['system_health_status']
                    })
                else:
                    # Fallback for old format
                    exec_summary = run.get("execution_summary", {})
                    perf = exec_summary.get("performance_metrics", {})
                    timing = exec_summary.get("timing", {})

                    total_cost += perf.get("total_cost", 0)
                    total_tokens += perf.get("total_tokens", 0)
                    total_events += perf.get("total_events", 0)
                    total_errors += perf.get("error_count", 0)
                    total_duration += timing.get("elapsed", 0)

                    run_summaries.append({
                        "run_id": run["run_id"],
                        "run_name": run["run_name"],
                        "duration": timing.get("elapsed", 0),
                        "events": perf.get("total_events", 0),
                        "cost": perf.get("total_cost", 0),
                        "tokens": perf.get("total_tokens", 0),
                        "errors": perf.get("error_count", 0),
                        "outline_completion": 0,  # Not available in old format
                        "reasoning_loops": 0,  # Not available in old format
                        "system_health": "unknown"
                    })

            # Calculate averages
            num_runs = len(self._accumulated_runs)
            avg_duration = total_duration / num_runs
            avg_cost = total_cost / num_runs
            avg_tokens = total_tokens / num_runs
            avg_events = total_events / num_runs

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
                    "total_duration": total_duration,
                    "total_outline_steps": total_outline_steps,
                    "total_reasoning_loops": total_reasoning_loops,
                },

                "average_metrics": {
                    "avg_duration": avg_duration,
                    "avg_cost": avg_cost,
                    "avg_tokens": avg_tokens,
                    "avg_events": avg_events,
                    "avg_error_rate": total_errors / max(total_events, 1),
                    "avg_outline_completion": sum(r.get("outline_completion", 0) for r in run_summaries) / num_runs,
                    "avg_reasoning_loops": total_reasoning_loops / num_runs
                },

                "run_summaries": run_summaries,
                "performance_insights": self._generate_accumulated_insights(run_summaries)
            }

        except Exception as e:
            return {"error": f"Error generating accumulated summary: {e}"}

    def export_accumulated_data(self, filepath: str = None, extra_data: dict[str, Any] = None) -> str:
        """Export all accumulated run data to file with dual-track information"""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"accumulated_execution_data_{timestamp}.json"

            export_data = {
                "export_timestamp": time.time(),
                "export_version": "2.0",  # Updated version for dual-track
                "printer_config": {
                    "mode": self.mode.value,
                    "use_rich": self.use_rich,
                    "agent_name": self.agent_name
                },
                "accumulated_summary": self.get_accumulated_summary(),
                "all_runs": self._accumulated_runs,
                "dual_track_metadata": {
                    "total_semantic_events": sum(
                        len(run.get('execution_events', [])) for run in self._accumulated_runs
                    ),
                    "total_system_nodes": sum(
                        len(run.get('system_state', {}).get('node_flow', [])) for run in self._accumulated_runs
                    ),
                    "export_features": ["dual_track_processing", "semantic_progress", "system_state"]
                }
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

    def _format_cost(self, cost: float) -> str:
        """Enhanced cost formatting with better precision"""
        if cost < 0.0001:
            return f"${cost * 1000000:.1f}μ"
        elif cost < 0.001:
            return f"${cost * 1000:.2f}m"
        elif cost < 1:
            return f"${cost:.4f}"
        else:
            return f"${cost:.2f}"

    def reset_global_start_time(self):
        """Reset global start time for new session"""
        self._global_start_time = time.time()

    def _print_accumulated_summary_fallback(self, summary: dict[str, Any]):
        """Fallback accumulated summary without Rich"""
        try:
            print(f"\n{'=' * 80}")
            print("🗂️ ACCUMULATED EXECUTION SUMMARY")
            print(f"{'=' * 80}")

            agg = summary["aggregate_metrics"]
            avg = summary["average_metrics"]

            print(f"Total Runs: {summary['total_runs']}")
            print(f"Total Duration: {agg['total_duration']:.1f}s (avg: {avg['avg_duration']:.1f}s)")
            print(f"Total Events: {agg['total_events']} (avg: {avg['avg_events']:.1f})")

            if agg["total_cost"] > 0:
                print(f"Total Cost: {self._format_cost(agg['total_cost'])} (avg: {self._format_cost(avg['avg_cost'])})")

            if agg["total_tokens"] > 0:
                print(f"Total Tokens: {agg['total_tokens']:,} (avg: {avg['avg_tokens']:,.0f})")

            # Dual-track specific metrics
            if agg.get("total_outline_steps", 0) > 0:
                print(f"Total Outline Steps: {agg['total_outline_steps']}")
                print(f"Avg Outline Completion: {avg['avg_outline_completion']:.1f}%")

            if agg.get("total_reasoning_loops", 0) > 0:
                print(f"Total Reasoning Loops: {agg['total_reasoning_loops']} (avg: {avg['avg_reasoning_loops']:.1f})")

            print(f"Average Error Rate: {avg['avg_error_rate']:.1%}")

            print(f"\n{'=' * 80}")
            print("🏃 INDIVIDUAL RUNS:")
            print(f"{'=' * 80}")

            for run in summary["run_summaries"]:
                cost_str = self._format_cost(run["cost"]) if run["cost"] > 0 else "N/A"
                outline_str = f"{run['outline_completion']:.0f}%" if run.get('outline_completion') else "N/A"

                print(f"• {run['run_name']}: {run['duration']:.1f}s | "
                      f"{run['events']} events | Cost: {cost_str} | "
                      f"Outline: {outline_str} | Health: {run.get('system_health', 'unknown')}")

            # Insights
            if summary.get("performance_insights"):
                print("\n🔍 PERFORMANCE INSIGHTS:")
                print(f"{'-' * 40}")
                for insight in summary["performance_insights"]:
                    print(f"• {insight}")

            print(f"{'=' * 80}")

        except Exception as e:
            print(f"❌ Error printing fallback summary: {e}")

    def _generate_accumulated_insights(self, run_summaries: list[dict[str, Any]]) -> list[str]:
        """Generate insights from accumulated run data with dual-track awareness"""
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
            error_counts = [r["errors"] for r in run_summaries]
            avg_errors = sum(error_counts) / len(error_counts)

            if avg_errors == 0:
                insights.append("✨ Perfect reliability: Zero errors across all runs")
            elif avg_errors < 1:
                insights.append(f"✅ High reliability: {avg_errors:.1f} average errors per run")
            elif avg_errors > 5:
                insights.append(f"🔧 Reliability concerns: {avg_errors:.1f} average errors per run")

            # Cost efficiency
            costs = [r["cost"] for r in run_summaries if r["cost"] > 0]
            if costs:
                avg_cost = sum(costs) / len(costs)
                if avg_cost < 0.01:
                    insights.append(f"💚 Very cost efficient: {self._format_cost(avg_cost)} average per run")
                elif avg_cost > 0.1:
                    insights.append(f"💸 High cost per run: {self._format_cost(avg_cost)} average")

            # Dual-track specific insights
            outline_completions = [r.get("outline_completion", 0) for r in run_summaries if r.get("outline_completion")]
            if outline_completions:
                avg_completion = sum(outline_completions) / len(outline_completions)
                if avg_completion > 95:
                    insights.append(f"🎯 Excellent outline completion: {avg_completion:.1f}% average")
                elif avg_completion < 80:
                    insights.append(f"📋 Low outline completion: {avg_completion:.1f}% - investigate planning")

            reasoning_loops = [r.get("reasoning_loops", 0) for r in run_summaries if r.get("reasoning_loops")]
            if reasoning_loops:
                avg_loops = sum(reasoning_loops) / len(reasoning_loops)
                if avg_loops > 10:
                    insights.append(f"🧠 High reasoning activity: {avg_loops:.1f} loops average")
                elif avg_loops < 3:
                    insights.append(f"⚡ Efficient reasoning: {avg_loops:.1f} loops average")

            # System health patterns
            health_statuses = [r.get("system_health", "unknown") for r in run_summaries]
            healthy_count = sum(1 for h in health_statuses if h == "healthy")
            if healthy_count == len(health_statuses):
                insights.append("💚 Perfect system health across all runs")
            elif healthy_count / len(health_statuses) < 0.8:
                insights.append("⚠️ System health issues detected in multiple runs")

            # Consistency analysis
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

        except Exception as e:
            insights.append(f"⚠️ Error generating insights: {e}")

        return insights

    def _reset_for_fresh_execution(self):
        """Reset internal state for a completely fresh execution"""
        try:
            # Increment run counter
            self._current_run_id += 1

            # Reset dual-track processor
            self.event_processor = DualTrackEventProcessor()

            # Reset display management
            self._last_display_time = 0
            self._print_counter = 0
            self._consecutive_errors = 0

            # Reset session info
            self.session_id = None

        except Exception as e:
            print(f"⚠️ Error during reset: {e}")

    def _print_comprehensive_final_table(self, summary: dict[str, Any]):
        """Print comprehensive final summary table with dual-track metrics"""
        if not self.use_rich:
            return

        table = Table(title="📊 Final Execution Summary", box=box.ROUNDED)
        table.add_column("Category", style="cyan", min_width=15)
        table.add_column("Metric", style="white", min_width=20)
        table.add_column("Value", style="green", min_width=15)

        # Session info
        table.add_row("Session", "Agent Name", self.agent_name)
        table.add_row("", "Session ID", str(self.session_id or "N/A"))
        table.add_row("", "Total Runtime", f"{summary['execution_metrics']['elapsed_time']:.2f}s")

        # Progress track
        semantic = summary['dual_track_state']['semantic_progress']
        activity = summary['current_activity']

        table.add_row("Progress", "Final Phase", activity['execution_phase'].title())
        table.add_row("", "Outline Completion", f"{activity['outline_completion_percent']:.1f}%")
        table.add_row("", "Reasoning Loops", str(semantic['current_reasoning_loop']))

        # System track
        system = summary['dual_track_state']['system_state']
        table.add_row("System", "Nodes Processed", str(len(system['node_flow'])))
        table.add_row("", "System Health", system['system_health']['status'].title())
        table.add_row("", "Error Count", str(system['system_health']['error_count']))

        # Performance
        metrics = summary['execution_metrics']
        table.add_row("Performance", "Total Events", str(metrics['total_events']))
        table.add_row("", "Processing Rate", f"{metrics['events_per_second']:.1f} events/sec")
        table.add_row("", "Error Rate", f"{metrics['error_rate']:.1%}")

        # LLM metrics
        llm = semantic['llm_interactions']
        if llm['total_calls'] > 0:
            table.add_row("LLM", "Total Calls", str(llm['total_calls']))
            if llm['total_cost'] > 0:
                table.add_row("", "Total Cost", self._format_cost(llm['total_cost']))
            if llm['total_tokens'] > 0:
                table.add_row("", "Total Tokens", f"{llm['total_tokens']:,}")

        self.console.print()
        self.console.print(table)

    def _print_dual_track_performance_analysis(self, summary: dict[str, Any]):
        """Print performance analysis with dual-track insights"""
        if not self.use_rich:
            return

        insights = []

        # Progress track analysis
        activity = summary['current_activity']
        semantic = summary['dual_track_state']['semantic_progress']

        if activity['outline_completion_percent'] > 95:
            insights.append("✨ Excellent outline completion")
        elif activity['outline_completion_percent'] < 80:
            insights.append("⚠️ Low outline completion - planning may need improvement")

        if semantic['current_reasoning_loop'] > 10:
            insights.append("🧠 High reasoning activity - complex problem solving")
        elif semantic['current_reasoning_loop'] < 3:
            insights.append("⚡ Efficient reasoning - straightforward execution")

        # System track analysis
        system = summary['dual_track_state']['system_state']
        metrics = summary['execution_metrics']

        if metrics['events_per_second'] > 10:
            insights.append("🚀 High processing efficiency")
        elif metrics['events_per_second'] < 2:
            insights.append("🐌 Low processing rate - possible bottlenecks")

        if system['system_health']['status'] == 'healthy':
            insights.append("💚 Perfect system health")
        else:
            insights.append("🔧 System health issues detected")

        # Cross-track analysis
        if (activity['outline_completion_percent'] > 90 and
            system['system_health']['status'] == 'healthy' and
            metrics['error_rate'] < 0.1):
            insights.append("🏆 Optimal execution across all tracks")

        if insights:
            analysis_panel = Panel(
                "\n".join(f"• {insight}" for insight in insights),
                title="🔍 Dual-Track Performance Analysis",
                style="yellow"
            )
            self.console.print()
            self.console.print(analysis_panel)

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

            # Overview table with dual-track metrics
            overview_table = Table(title="📊 Aggregate Overview", box=box.ROUNDED)
            overview_table.add_column("Metric", style="cyan", min_width=25)
            overview_table.add_column("Total", style="green", min_width=15)
            overview_table.add_column("Average", style="blue", min_width=15)

            agg = summary["aggregate_metrics"]
            avg = summary["average_metrics"]

            overview_table.add_row("Runs", str(summary["total_runs"]), "")
            overview_table.add_row("Duration", f"{agg['total_duration']:.1f}s", f"{avg['avg_duration']:.1f}s")
            overview_table.add_row("Events", str(agg["total_events"]), f"{avg['avg_events']:.1f}")

            if agg["total_cost"] > 0:
                overview_table.add_row("Cost", self._format_cost(agg["total_cost"]), self._format_cost(avg["avg_cost"]))

            if agg["total_tokens"] > 0:
                overview_table.add_row("Tokens", f"{agg['total_tokens']:,}", f"{avg['avg_tokens']:,.0f}")

            # Dual-track specific metrics
            if agg.get("total_outline_steps", 0) > 0:
                overview_table.add_row("Outline Steps", str(agg["total_outline_steps"]), "")
                overview_table.add_row("Outline Completion", "", f"{avg['avg_outline_completion']:.1f}%")

            if agg.get("total_reasoning_loops", 0) > 0:
                overview_table.add_row("Reasoning Loops", str(agg["total_reasoning_loops"]),
                                       f"{avg['avg_reasoning_loops']:.1f}")

            overview_table.add_row("Error Rate", "", f"{avg['avg_error_rate']:.1%}")

            self.console.print(overview_table)

            # Individual runs table
            runs_table = Table(title="🏃 Individual Runs", box=box.ROUNDED)
            runs_table.add_column("Run", style="cyan")
            runs_table.add_column("Duration", style="blue")
            runs_table.add_column("Events", style="green")
            runs_table.add_column("Cost", style="yellow")
            runs_table.add_column("Outline", style="magenta")
            runs_table.add_column("Health", style="white")

            for run in summary["run_summaries"]:
                cost_str = self._format_cost(run["cost"]) if run["cost"] > 0 else "-"
                outline_str = f"{run.get('outline_completion', 0):.0f}%" if run.get('outline_completion') else "N/A"
                health_str = run.get('system_health', 'unknown')

                runs_table.add_row(
                    run["run_name"],
                    f"{run['duration']:.1f}s",
                    str(run['events']),
                    cost_str,
                    outline_str,
                    health_str
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

    def _get_display_interval(self) -> float:
        """Get appropriate display update interval based on mode"""
        intervals = {
            VerbosityMode.MINIMAL: 2.0,
            VerbosityMode.STANDARD: 1.0,
            VerbosityMode.VERBOSE: 0.5,
            VerbosityMode.DEBUG: 0.1,
            VerbosityMode.REALTIME: 0.2
        }
        return intervals.get(self.mode, 1.0)

    async def progress_callback(self, event: ProgressEvent):
        """Enhanced progress callback with dual-track processing"""
        try:
            # Update agent info
            if event.agent_name:
                self.agent_name = event.agent_name
            if event.session_id:
                self.session_id = event.session_id

            # Process through dual-track system
            self.event_processor.process_event(event)

            # Update display if enough time has passed or important event
            should_update = (
                time.time() - self._last_display_time >= self._display_interval or
                self._is_important_event(event)
            )

            if should_update and self.auto_refresh:
                self._update_display()
                self._last_display_time = time.time()

        except Exception as e:
            self._consecutive_errors += 1
            if self._consecutive_errors <= self._error_threshold:
                print(f"⚠️ Progress callback error #{self._consecutive_errors}: {e}")
            if self._consecutive_errors > self._error_threshold:
                print("🚨 Progress printing disabled due to excessive errors")
                self.progress_callback = self._noop_callback

    def _is_important_event(self, event: ProgressEvent) -> bool:
        """Determine if event requires immediate display update"""
        important_events = {
            'execution_start', 'execution_complete',
            'outline_created', 'plan_created',
            'error', 'task_error'
        }
        return event.event_type in important_events or event.success is False

    def _update_display(self):
        """Update the display using dual-track renderer"""
        try:
            self._print_counter += 1
            self.display_renderer.render_dual_track_display(self.event_processor)
            self._consecutive_errors = 0  # Reset error counter on success

        except Exception as e:
            self._consecutive_errors += 1
            print(f"⚠️ Display update error: {e}")

    def print_execution_summary(self):
        """Print comprehensive execution summary"""
        try:
            summary = self.event_processor.get_progress_summary()

            if not self.use_rich:
                self._print_summary_fallback(summary)
                return

            self.display_renderer.console.print()
            self.display_renderer.console.print("🎉 [bold green]EXECUTION SUMMARY[/bold green] 🎉")

            # Final status display
            self.display_renderer.render_dual_track_display(self.event_processor)

            # Detailed metrics table
            self._print_detailed_metrics_table(summary)

        except Exception as e:
            print(f"⚠️ Error printing execution summary: {e}")
            self._print_summary_fallback(self.event_processor.get_progress_summary())

    def _print_detailed_metrics_table(self, summary: dict[str, Any]):
        """Print detailed metrics table"""
        if not self.use_rich:
            return

        table = Table(title="📊 Execution Metrics", box=box.ROUNDED)
        table.add_column("Category", style="cyan")
        table.add_column("Metric", style="white")
        table.add_column("Value", style="green")

        # Progress track metrics
        semantic = summary['dual_track_state']['semantic_progress']
        table.add_row("Progress", "Execution Phase", semantic['execution_phase'].title())
        table.add_row("", "Outline Steps",
                      f"{len(semantic['outline_progress']['completed_steps'])}/{semantic['outline_progress']['total_steps']}")
        table.add_row("", "Reasoning Loops", str(semantic['current_reasoning_loop']))

        # System track metrics
        system = summary['dual_track_state']['system_state']
        table.add_row("System", "Node Flow Length", str(len(system['node_flow'])))
        table.add_row("", "System Health", system['system_health']['status'].title())
        table.add_row("", "Error Count", str(system['system_health']['error_count']))

        # Execution metrics
        metrics = summary['execution_metrics']
        table.add_row("Performance", "Total Events", str(metrics['total_events']))
        table.add_row("", "Runtime", f"{metrics['elapsed_time']:.2f}s")
        table.add_row("", "Events/sec", f"{metrics['events_per_second']:.1f}")

        # LLM metrics
        llm = semantic['llm_interactions']
        if llm['total_calls'] > 0:
            table.add_row("LLM", "Total Calls", str(llm['total_calls']))
            if llm['total_cost'] > 0:
                table.add_row("", "Total Cost", f"${llm['total_cost']:.4f}")
            if llm['total_tokens'] > 0:
                table.add_row("", "Total Tokens", f"{llm['total_tokens']:,}")

        self.display_renderer.console.print()
        self.display_renderer.console.print(table)

    def _print_summary_fallback(self, summary: dict[str, Any]):
        """Fallback summary without Rich"""
        activity = summary['current_activity']
        metrics = summary['execution_metrics']
        semantic = summary['dual_track_state']['semantic_progress']

        print(f"\n{'=' * 60}")
        print("🎉 EXECUTION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Agent: {self.agent_name}")
        print(f"Session: {self.session_id or 'N/A'}")
        print(f"Final Phase: {activity['execution_phase'].title()}")

        if semantic['outline_progress']['total_steps'] > 0:
            completed_steps = len(semantic['outline_progress']['completed_steps'])
            total_steps = semantic['outline_progress']['total_steps']
            print(
                f"Outline Progress: {completed_steps}/{total_steps} steps ({activity['outline_completion_percent']:.1f}%)")

        print(f"Total Runtime: {metrics['elapsed_time']:.2f}s")
        print(f"Total Events: {metrics['total_events']}")
        print(f"Processing Rate: {metrics['events_per_second']:.1f} events/sec")
        print(f"System Health: {activity['system_health_status'].title()}")

        if metrics['error_rate'] > 0:
            print(f"Error Rate: {metrics['error_rate']:.2%}")

        llm = semantic['llm_interactions']
        if llm['total_calls'] > 0:
            print(f"LLM Calls: {llm['total_calls']}")
            if llm['total_cost'] > 0:
                print(f"LLM Cost: ${llm['total_cost']:.4f}")

        print(f"{'=' * 60}")

    # [Keep all the existing methods that were listed as "methods to keep"]
    # flush, get_accumulated_summary, export_accumulated_data, etc.

    async def _noop_callback(self, event: ProgressEvent):
        """No-op callback when printing is disabled"""
        pass

    def get_current_execution_state(self) -> dict[str, Any]:
        """Get current execution state for external monitoring"""
        return self.event_processor.get_progress_summary()

    def force_display_update(self):
        """Force an immediate display update"""
        self._update_display()

    def set_display_mode(self, mode: VerbosityMode):
        """Change display mode at runtime"""
        self.mode = mode
        self.display_renderer = EnhancedDisplayRenderer(mode, self.use_rich)
        self._display_interval = self._get_display_interval()

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


class ChainProgressTracker:
    """Enhanced progress tracker for chain execution with live display"""

    def __init__(self, chain_printer: 'ChainPrinter'):
        self.events: list[ProgressEvent] = []
        self.start_time = time.time()
        self.chain_printer = chain_printer
        self.current_task = None
        self.task_count = 0
        self.completed_tasks = 0

    async def emit_event(self, event: ProgressEvent):
        """Emit progress event with live display updates"""
        self.events.append(event)

        if event.event_type == "chain_start":
            self.task_count = event.metadata.get("task_count", 0)
            self.chain_printer.print_progress_start(event.node_name)

        elif event.event_type == "task_start":
            self.current_task = event.node_name
            self.chain_printer.print_task_start(event.node_name, self.completed_tasks, self.task_count)

        elif event.event_type == "task_complete":
            if event.status == NodeStatus.COMPLETED:
                self.completed_tasks += 1
                self.chain_printer.print_task_complete(event.node_name, self.completed_tasks, self.task_count)
            elif event.status == NodeStatus.FAILED:
                self.chain_printer.print_task_error(event.node_name, event.metadata.get("error", "Unknown error"))

        elif event.event_type == "chain_end":
            duration = time.time() - self.start_time
            self.chain_printer.print_progress_end(event.node_name, duration, event.status == NodeStatus.COMPLETED)

        elif event.event_type == "tool_call" and event.success == False:
            self.chain_printer.print_tool_usage_error(event.tool_name, event.metadata.get("error",
                                                                                          event.metadata.get("message",
                                                                                                             event.error_details.get(
                                                                                                                 "error",
                                                                                                                 "Unknown error"))))

        elif event.event_type == "tool_call" and event.success == True:
            self.chain_printer.print_tool_usage_success(event.tool_name, event.duration, event.is_meta_tool)

        elif event.event_type == "outline_created":
            self.chain_printer.print_outline_created(event.metadata.get("outline", {}))

        elif event.event_type == "reasoning_loop":
            self.chain_printer.print_reasoning_loop(event.metadata)

        elif event.event_type == "task_error":
            self.chain_printer.print_task_error(event.node_name, event.metadata.get("error", "Unknown error"))


class ChainPrinter:
    """Custom printer for enhanced chain visualization and progress display"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.colors = {
            'success': '\033[92m',
            'error': '\033[91m',
            'warning': '\033[93m',
            'info': '\033[94m',
            'highlight': '\033[95m',
            'dim': '\033[2m',
            'bold': '\033[1m',
            'reset': '\033[0m'
        }

    def _colorize(self, text: str, color: str) -> str:
        return f"{self.colors.get(color, '')}{text}{self.colors['reset']}"

    def print_header(self, title: str, subtitle: str = None):
        """Print formatted header"""
        print(f"\n{self._colorize('═' * 60, 'highlight')}")
        print(f"{self._colorize(f'🔗 {title}', 'bold')}")
        if subtitle:
            print(f"{self._colorize(subtitle, 'dim')}")
        print(f"{self._colorize('═' * 60, 'highlight')}\n")

    def print_success(self, message: str):
        print(f"{self._colorize('✅ ', 'success')}{message}")

    def print_error(self, message: str):
        print(f"{self._colorize('❌ ', 'error')}{message}")

    def print_warning(self, message: str):
        print(f"{self._colorize('⚠️ ', 'warning')}{message}")

    def print_info(self, message: str):
        print(f"{self._colorize('ℹ️ ', 'info')}{message}")

    def print_progress_start(self, chain_name: str):
        print(f"\n{self._colorize('🚀 Starting chain execution:', 'info')} {self._colorize(chain_name, 'bold')}")

    def print_task_start(self, task_name: str, current: int, total: int):
        progress = f"[{current + 1}/{total}]" if total > 0 else ""
        print(f"  {self._colorize('▶️ ', 'info')}{progress} {task_name}")

    def print_task_complete(self, task_name: str, completed: int, total: int):
        progress = f"[{completed}/{total}]" if total > 0 else ""
        print(f"  {self._colorize('✅', 'success')} {progress} {task_name} completed")

    def print_task_error(self, task_name: str, error: str):
        print(f"  {self._colorize('❌', 'error')} {task_name} failed: {error}")

    def print_progress_end(self, chain_name: str, duration: float, success: bool):
        status = self._colorize('✅ COMPLETED', 'success') if success else self._colorize('❌ FAILED', 'error')
        print(f"\n{status} {chain_name} ({duration:.2f}s)\n")

    def print_tool_usage_success(self, tool_name: str, duration: float, is_meta_tool: bool = False):
        if is_meta_tool:
            print(f"  {self._colorize('🔧 ', 'info')}{tool_name} completed ({duration:.2f}s)")
        else:
            print(f"  {self._colorize('🔩 ', 'info')}{tool_name} completed ({duration:.2f}s)")

    def print_tool_usage_error(self, tool_name: str, error: str, is_meta_tool: bool = False):
        if is_meta_tool:
            print(f"  {self._colorize('🔧 ', 'error')}{tool_name} failed: {error}")
        else:
            print(f"  {self._colorize('🔩 ', 'error')}{tool_name} failed: {error}")

    def print_outline_created(self, outline: dict):
        for step in outline.get("steps", []):
            print(f"  {self._colorize('📖 ', 'info')}Step: {self._colorize(step.get('description', 'Unknown'), 'dim')}")

    def print_reasoning_loop(self, loop_data: dict):
        print(f"  {self._colorize('🧠 ', 'info')}Reasoning Loop #{loop_data.get('loop_number', '?')}")
        print(
            f"    {self._colorize('📖 ', 'info')}Outline Step: {loop_data.get('outline_step', 0)} of {loop_data.get('outline_total', 0)}")
        print(f"    {self._colorize('📚 ', 'info')}Context Size: {loop_data.get('context_size', 0)} entries")
        print(f"    {self._colorize('📋 ', 'info')}Task Stack: {loop_data.get('task_stack_size', 0)} items")
        print(f"    {self._colorize('🔄 ', 'info')}Recovery Attempts: {loop_data.get('auto_recovery_attempts', 0)}")
        print(f"    {self._colorize('📊 ', 'info')}Performance Metrics: {loop_data.get('performance_metrics', {})}")

    def print_chain_list(self, chains: list[tuple[str, ChainMetadata]]):
        """Print formatted list of available chains"""
        if not chains:
            self.print_info("No chains found. Use 'create' to build your first chain.")
            return

        self.print_header("Available Chains", f"Total: {len(chains)}")

        for name, meta in chains:
            # Status indicators
            indicators = []
            if meta.has_parallels:
                indicators.append(self._colorize("⚡", "highlight"))
            if meta.has_conditionals:
                indicators.append(self._colorize("🔀", "warning"))
            if meta.has_error_handling:
                indicators.append(self._colorize("🛡️", "info"))

            status_str = " ".join(indicators) if indicators else ""

            # Complexity color
            complexity_colors = {"simple": "success", "medium": "warning", "complex": "error"}
            complexity = self._colorize(meta.complexity, complexity_colors.get(meta.complexity, "info"))

            print(f"  {self._colorize(name, 'bold')} {status_str}")
            print(f"    {meta.description or 'No description'}")
            print(f"    {complexity} • {meta.agent_count} agents • {meta.version}")
            if meta.tags:
                tags_str = " ".join([f"#{tag}" for tag in meta.tags])
                print(f"    {self._colorize(tags_str, 'dim')}")
            print()

# Demo and testing functions
async def demo_enhanced_printer():
    """Comprehensive demo of the enhanced progress printer showcasing all modes"""

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
            input("\n⏸️  Press Enter to continue to next mode...")


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
        duration=1.1,
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
        duration=1.3,
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
        duration=2.1,
        success=not should_fail,
        tool_result="Search completed" if not should_fail else None,
        tool_error="Search failed" if should_fail else None,
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
            duration=2.3,
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
                duration=1.5 + i * 0.3,
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
