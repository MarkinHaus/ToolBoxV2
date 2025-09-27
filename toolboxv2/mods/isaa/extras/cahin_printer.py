import time
from typing import Any

from toolboxv2.mods.isaa.base.Agent.types import (
    NodeStatus,
    ProgressEvent, ChainMetadata,
)
from toolboxv2.mods.isaa.extras.terminal_progress import arguments_summary


class ChainProgressTracker:
    """Enhanced progress tracker for chain execution with live display"""

    def __init__(self, chain_printer: 'ChainPrinter' = None):
        self.events: list[ProgressEvent] = []
        self.start_time = time.time()
        self.chain_printer = chain_printer or ChainPrinter()
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
            self.chain_printer.print_tool_usage_success(event.tool_name, event.duration, event.is_meta_tool, event.tool_args)

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

    def print_tool_usage_success(self, tool_name: str, duration: float, is_meta_tool: bool = False, tool_args: dict[str, Any] = None):
        if is_meta_tool:
            print(f"  {self._colorize('🔧 ', 'info')}{tool_name} completed ({duration:.2f}s) {arguments_summary(tool_args)}")
        else:
            print(f"  {self._colorize('🔩 ', 'info')}{tool_name} completed ({duration:.2f}s) {arguments_summary(tool_args)}")

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
