"""
ProA Kernel CLI Interface
==========================

Production-ready CLI interface for the Enhanced ProA Kernel with:
- Auto-persistence (save/load on start/stop)
- Signal handling (graceful shutdown)
- Rich terminal output with colors
- Command history
- Multi-line input support
- Status display
"""

import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from toolboxv2 import get_app
from toolboxv2.mods.isaa.extras.terminal_progress import ProgressiveTreePrinter
from toolboxv2.mods.isaa.kernel.instace import Kernel
from toolboxv2.mods.isaa.kernel.types import Signal as KernelSignal, SignalType, KernelConfig, IOutputRouter


class CLIOutputRouter(IOutputRouter):
    """CLI-specific output router with colored terminal output"""

    def __init__(self):
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'red': '\033[91m',
            'cyan': '\033[96m',
        }

    def _colorize(self, text: str, color: str) -> str:
        """Add color to text"""
        return f"{self.colors.get(color, '')}{text}{self.colors['reset']}"

    async def send_response(self, user_id: str, content: str,role: str = "assistant",metadata: dict = None):
        """Send agent response to CLI"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n{self._colorize(f'[{timestamp}] Agent:', 'cyan')} {content}\n")

    async def send_notification(self, user_id: str, content: str, priority: int = 5, metadata: dict = None):
        """Send notification to CLI"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = 'yellow' if priority >= 7 else 'blue'
        print(f"{self._colorize(f'[{timestamp}] 🔔 {content}', color)}")

    async def send_error(self, user_id: str, error: str, metadata: dict = None):
        """Send error message to CLI"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{self._colorize(f'[{timestamp}] ❌ Error: {error}', 'red')}")


class CLIKernel:
    """CLI-based ProA Kernel with auto-persistence"""

    def __init__(self, agent, user_id: str = "cli_user", auto_save_interval: int = 300):
        """
        Initialize CLI Kernel

        Args:
            agent: FlowAgent instance
            user_id: User identifier for CLI session
            auto_save_interval: Auto-save interval in seconds (default: 5 minutes)
        """
        self.agent = agent
        self.user_id = user_id
        self.auto_save_interval = auto_save_interval
        self.running = False
        self.save_path = self._get_save_path()

        # Initialize kernel with CLI output router
        config = KernelConfig(
            heartbeat_interval=30.0,
            idle_threshold=300.0,
            proactive_cooldown=60.0,
            max_proactive_per_hour=10
        )

        self.output_router = CLIOutputRouter()
        self.kernel = Kernel(
            agent=agent,
            config=config,
            output_router=self.output_router
        )

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print(self.output_router._colorize("✓ CLI Kernel initialized", "green"))

    def _get_save_path(self) -> Path:
        """Get save file path"""
        app = get_app()
        save_dir = Path(app.data_dir) / 'Agents' / 'kernel' / self.agent.amd.name / 'cli'
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / f"cli_kernel_{self.user_id}.pkl"

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(self.output_router._colorize("\n\n🛑 Shutdown signal received...", "yellow"))
        asyncio.create_task(self.stop())

    async def _auto_save_loop(self):
        """Auto-save kernel state periodically"""
        while self.running:
            await asyncio.sleep(self.auto_save_interval)
            if self.running:
                await self.kernel.save_to_file(str(self.save_path))
                print(self.output_router._colorize(f"💾 Auto-saved at {datetime.now().strftime('%H:%M:%S')}", "blue"))

    async def start(self):
        """Start the CLI kernel"""
        self.running = True

        # Load previous state if exists
        if self.save_path.exists():
            print(self.output_router._colorize("📂 Loading previous session...", "yellow"))
            await self.kernel.load_from_file(str(self.save_path))

        # Start kernel
        await self.kernel.start()

        # Inject kernel prompt to agent
        self.kernel.inject_kernel_prompt_to_agent()

        # Start auto-save loop
        asyncio.create_task(self._auto_save_loop())

        print(self.output_router._colorize("\n" + "="*60, "green"))
        print(self.output_router._colorize("  ProA Kernel CLI - Ready", "bold"))
        print(self.output_router._colorize("="*60 + "\n", "green"))
        print("Commands:")
        print("  - Type your message and press Enter")
        print("  - Type 'exit' or 'quit' to stop")
        print("  - Type 'status' to see kernel status")
        print("  - Press Ctrl+C for graceful shutdown\n")



    async def _process_input(self, user_input: str):
        """Process user input"""
        # Handle special commands
        if user_input.lower() in ['exit', 'quit']:
            await self.stop()
            return

        if user_input.lower() == 'status':
            await self._show_status()
            return

        # Send to kernel
        signal = KernelSignal(
            type=SignalType.USER_INPUT,
            id=self.user_id,
            content=user_input,
            metadata={"interface": "cli"}
        )
        await self.kernel.process_signal(signal)

    async def _show_status(self):
        """Show kernel status"""
        status = self.kernel.to_dict()
        print(self.output_router._colorize("\n" + "="*60, "cyan"))
        print(self.output_router._colorize("  Kernel Status", "bold"))
        print(self.output_router._colorize("="*60, "cyan"))
        print(f"State: {status['state']}")
        print(f"Running: {status['running']}")
        print(f"Signals Processed: {status['metrics']['signals_processed']}")
        print(f"Learning Records: {status['learning']['total_records']}")
        print(f"Memories: {status['memory']['total_memories']}")
        print(f"Scheduled Tasks: {status['scheduler']['total_tasks']}")
        print(self.output_router._colorize("="*60 + "\n", "cyan"))

    async def run(self):
        """Run the CLI interface"""
        await self.start()

        try:
            # Main input loop
            while self.running:
                try:
                    # Read input (non-blocking)
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: input(self.output_router._colorize("You: ", "green"))
                    )

                    if user_input.strip():
                        await self._process_input(user_input.strip())

                except EOFError:
                    # Handle Ctrl+D
                    await self.stop()
                    break
                except Exception as e:
                    print(self.output_router._colorize(f"Error: {e}", "red"))

        finally:
            if self.running:
                await self.stop()

    async def stop(self):
        """Stop the CLI kernel"""
        if not self.running:
            return

        self.running = False
        print(self.output_router._colorize("\n💾 Saving session...", "yellow"))

        # Save final state
        await self.kernel.save_to_file(str(self.save_path))

        # Stop kernel
        await self.kernel.stop()

        print(self.output_router._colorize("✓ Session saved", "green"))
        print(self.output_router._colorize("👋 Goodbye!\n", "cyan"))
        sys.exit(0)


# ===== USAGE EXAMPLE =====

async def main():
    """Example usage of CLI Kernel"""
    from toolboxv2 import get_app

    # Get ISAA tools
    app = get_app()
    isaa = app.get_mod("isaa")
    agent = await isaa.get_agent("self")
    agent.set_progress_callback(ProgressiveTreePrinter().progress_callback)
    # Create and run CLI kernel
    cli_kernel = CLIKernel(agent, user_id="default_user")
    await cli_kernel.run()


if __name__ == "__main__":
    asyncio.run(main())
