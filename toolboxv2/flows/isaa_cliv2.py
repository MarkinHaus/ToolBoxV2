import asyncio
import base64
import datetime
import json
import mimetypes
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import pickle
import signal
import sys

import litellm
import psutil
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import (
    FuzzyCompleter,
    NestedCompleter,
    PathCompleter,
    WordCompleter,
)
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

# ToolboxV2-spezifische Imports
from toolboxv2 import get_app, __init_cwd__
from toolboxv2.mods.isaa.base.Agent.agent import FlowAgent
from toolboxv2.mods.isaa.base.Agent.chain import Chain, CF, IS, Function, ParallelChain, ConditionalChain, \
    ErrorHandlingChain
from toolboxv2.mods.isaa.extras.terminal_progress import ProgressiveTreePrinter, VerbosityMode
from toolboxv2.mods.isaa.extras.verbose_output import EnhancedVerboseOutput
from toolboxv2.mods.isaa.module import Tools as Isaatools
from toolboxv2.mods.isaa.module import detect_shell
from toolboxv2.utils.extras.Style import Style, remove_styles
from toolboxv2.utils.extras.notification import (
    NotificationSystem, NotificationType, NotificationAction,
    NotificationDetails, NotificationPosition,
    quick_success, quick_error, quick_warning, quick_info
)

NAME = "isaa_cliv2"


# ===== TASK MANAGEMENT SYSTEM =====
@dataclass
class TaskStatus:
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    task_id: str
    agent_name: str
    prompt: str
    status: TaskStatus
    created_at: datetime.datetime
    started_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None
    session_id: str = "default"
    is_background: bool = False
    progress_data: Optional[Dict] = None


class TaskManager:
    """Manages background and foreground tasks with full lifecycle support"""

    def __init__(self, notification_system: NotificationSystem):
        self.tasks: Dict[str, TaskInfo] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.next_id = 1
        self.notification_system = notification_system
        self._shutdown_event = threading.Event()

    def generate_task_id(self) -> str:
        """Generate unique task ID"""
        task_id = f"task_{self.next_id:04d}"
        self.next_id += 1
        return task_id

    async def start_task(self, agent: FlowAgent, prompt: str, session_id: str = "default",
                         is_background: bool = False, **kwargs) -> str:
        """Start a new task"""
        task_id = self.generate_task_id()

        task_info = TaskInfo(
            task_id=task_id,
            agent_name=agent.amd.name,
            prompt=prompt,
            status=TaskStatus.QUEUED,
            created_at=datetime.datetime.now(),
            session_id=session_id,
            is_background=is_background
        )

        self.tasks[task_id] = task_info

        # Create and start the actual asyncio task
        async_task = asyncio.create_task(self._execute_task(task_info, agent, **kwargs))
        self.running_tasks[task_id] = async_task

        return task_id

    async def _execute_task(self, task_info: TaskInfo, agent: FlowAgent, **kwargs):
        """Execute a single task with full lifecycle management"""
        try:
            task_info.status = TaskStatus.RUNNING
            task_info.started_at = datetime.datetime.now()

            # Execute the agent
            result = await agent.a_run(task_info.prompt, session_id=task_info.session_id, **kwargs)

            # Task completed successfully
            task_info.status = TaskStatus.COMPLETED
            task_info.completed_at = datetime.datetime.now()
            task_info.result = result

            # Notify completion for background tasks
            if task_info.is_background:
                duration = (task_info.completed_at - task_info.started_at).total_seconds()
                self.notification_system.show_notification(
                    title="Task Completed",
                    message=f"Background task {task_info.task_id} finished successfully",
                    notification_type=NotificationType.SUCCESS,
                    details=NotificationDetails(
                        title="Task Details",
                        content=f"Agent: {task_info.agent_name}\nDuration: {duration:.1f}s\nPrompt: {task_info.prompt[:100]}...",
                        data={"task_id": task_info.task_id, "result_length": len(str(result))}
                    )
                )

        except Exception as e:
            task_info.status = TaskStatus.FAILED
            task_info.completed_at = datetime.datetime.now()
            task_info.error = str(e)

            # Notify failure
            if task_info.is_background:
                self.notification_system.show_notification(
                    title="Task Failed",
                    message=f"Background task {task_info.task_id} encountered an error",
                    notification_type=NotificationType.ERROR,
                    details=NotificationDetails(
                        title="Error Details",
                        content=f"Agent: {task_info.agent_name}\nError: {str(e)}\nPrompt: {task_info.prompt[:100]}...",
                        data={"task_id": task_info.task_id, "error": str(e)}
                    )
                )

        finally:
            # Clean up running task reference
            if task_info.task_id in self.running_tasks:
                del self.running_tasks[task_info.task_id]

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Get task by ID"""
        return self.tasks.get(task_id)

    def list_tasks(self, status_filter: Optional[TaskStatus] = None) -> List[TaskInfo]:
        """List all tasks, optionally filtered by status"""
        tasks = list(self.tasks.values())
        if status_filter:
            tasks = [t for t in tasks if t.status == status_filter]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    async def pause_task(self, task_id: str) -> bool:
        """Pause a running task (if supported by agent)"""
        if task_id in self.running_tasks and task_id in self.tasks:
            task_info = self.tasks[task_id]
            if task_info.status == TaskStatus.RUNNING:
                # This would require agent support for pausing
                task_info.status = TaskStatus.PAUSED
                return True
        return False

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if task_id in self.running_tasks:
            async_task = self.running_tasks[task_id]
            async_task.cancel()

            if task_id in self.tasks:
                self.tasks[task_id].status = TaskStatus.CANCELLED
                self.tasks[task_id].completed_at = datetime.datetime.now()

            return True
        return False

    async def attach_to_task(self, task_id: str, output_callback: Callable[[str], None]):
        """Attach to a running background task to see live output"""
        if task_id in self.running_tasks and task_id in self.tasks:
            task_info = self.tasks[task_id]
            if task_info.status == TaskStatus.RUNNING:
                # Wait for task completion while showing live updates
                async_task = self.running_tasks[task_id]

                try:
                    output_callback(f"🔗 Attached to task {task_id} ({task_info.agent_name})")
                    output_callback(f"📝 Prompt: {task_info.prompt}")
                    output_callback("⏳ Waiting for completion... (Press Ctrl+C to detach)")
                    agent = await get_app().get_mod("isaa").get_agent(task_info.agent_name)
                    agent.set_progress_callback(ProgressiveTreePrinter().progress_callback)

                    await async_task

                    if task_info.status == TaskStatus.COMPLETED:
                        output_callback(f"✅ Task completed successfully")
                        if task_info.result:
                            output_callback(f"📋 Result: {task_info.result}")
                    elif task_info.status == TaskStatus.FAILED:
                        output_callback(f"❌ Task failed: {task_info.error}")

                except asyncio.CancelledError or KeyboardInterrupt:
                    output_callback(f"🔌 Detached from task {task_id}")
                    agent = await get_app().get_mod("isaa").get_agent(task_info.agent_name)
                    async def noop_callback(event: Any):
                        pass
                    agent.set_progress_callback(noop_callback)

                return True
        return False

    def save_state(self, filepath: Path):
        """Save task manager state to file"""
        state = {
            'tasks': {tid: asdict(task) for tid, task in self.tasks.items()},
            'next_id': self.next_id
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath: Path):
        """Load task manager state from file"""
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    state = pickle.load(f)

                # Reconstruct tasks
                for tid, task_data in state.get('tasks', {}).items():
                    # Convert datetime strings back to datetime objects
                    for field in ['created_at', 'started_at', 'completed_at']:
                        if task_data.get(field):
                            task_data[field] = datetime.datetime.fromisoformat(task_data[field]) if isinstance(
                                task_data[field], str) else task_data[field]

                    # Convert status back to enum
                    if 'status' in task_data:
                        task_data['status'] = TaskStatus(task_data['status'])

                    self.tasks[tid] = TaskInfo(**task_data)

                self.next_id = state.get('next_id', 1)

            except Exception as e:
                print(f"⚠️ Failed to load task state: {e}")


# ===== FLOW PARSER & MANAGER =====

class FlowParser:
    """Parses workflow syntax into executable Chain objects"""

    def __init__(self, isaa_tools: Isaatools):
        self.isaa_tools = isaa_tools
        self.saved_flows: Dict[str, str] = {}
        self.flow_definitions: Dict[str, Chain] = {}

    async def parse_flow_string(self, flow_string: str) -> Chain:
        """Parse a flow string like 'agent1 >> agent2 | CF(Model)' into a Chain"""
        # This is a simplified parser - a production version would use proper parsing

        # Replace agent names with actual agent instances
        flow_parts = flow_string.split()
        processed_parts = []

        i = 0
        while i < len(flow_parts):
            part = flow_parts[i]

            # Handle operators
            if part in ['>>', '|', '+', '&', '%']:
                processed_parts.append(part)

            # Handle CF syntax
            elif part.startswith('CF(') and part.endswith(')'):
                model_name = part[3:-1]  # Extract model name
                # This would require importing the pydantic model
                processed_parts.append(f"CF({model_name})")

            # Handle IS syntax
            elif part.startswith('IS('):
                # Extract condition parameters
                processed_parts.append(part)

            # Handle Function syntax
            elif part.startswith('Function('):
                processed_parts.append(part)

            # Assume it's an agent name
            else:
                try:
                    agent = await self.isaa_tools.get_agent(part)
                    processed_parts.append(agent)
                except:
                    # If agent doesn't exist, keep as string for error handling
                    processed_parts.append(part)

            i += 1

        # Build chain from processed parts
        return await self._build_chain_from_parts(processed_parts)

    async def _build_chain_from_parts(self, parts: List) -> Chain:
        """Build a Chain object from parsed parts"""
        if not parts:
            raise ValueError("Empty flow definition")

        if len(parts) == 1:
            return Chain(parts[0])

        # Simple left-to-right chain building
        # A full implementation would handle operator precedence
        current_chain = Chain(parts[0])

        i = 1
        while i < len(parts) - 1:
            operator = parts[i]
            next_component = parts[i + 1]

            if operator == '>>':
                current_chain = current_chain >> next_component
            elif operator == '+' or operator == '&':
                current_chain = current_chain + next_component
            elif operator == '|':
                current_chain = current_chain | next_component
            elif operator == '%':
                current_chain = current_chain % next_component

            i += 2

        return current_chain

    def save_flow(self, name: str, flow_string: str):
        """Save a flow definition"""
        self.saved_flows[name] = flow_string

        # Persist to file
        flow_dir = Path(get_app().data_dir) / "isaa_cli" / "flows"
        flow_dir.mkdir(parents=True, exist_ok=True)

        with open(flow_dir / f"{name}.json", 'w') as f:
            json.dump({'name': name, 'definition': flow_string, 'created_at': datetime.datetime.now().isoformat()}, f,
                      indent=2)

    def load_saved_flows(self):
        """Load all saved flow definitions"""
        flow_dir = Path(get_app().data_dir) / "isaa_cli" / "flows"
        if not flow_dir.exists():
            return

        for flow_file in flow_dir.glob("*.json"):
            try:
                with open(flow_file, 'r') as f:
                    flow_data = json.load(f)
                    self.saved_flows[flow_data['name']] = flow_data['definition']
            except Exception as e:
                print(f"⚠️ Failed to load flow {flow_file}: {e}")

    def list_flows(self) -> Dict[str, str]:
        """List all saved flows"""
        return self.saved_flows.copy()

    async def run_flow(self, name: str, input_data: str, **kwargs) -> str:
        """Execute a saved flow"""
        if name not in self.saved_flows:
            raise ValueError(f"Flow '{name}' not found")

        flow_string = self.saved_flows[name]
        chain = await self.parse_flow_string(flow_string)

        return await chain.a_run(input_data, **kwargs)


# ===== USER PROXY AGENT MANAGER =====

class UserProxyManager:
    """Manages the user-proxy agent and bindings"""

    def __init__(self, isaa_tools: Isaatools):
        self.isaa_tools = isaa_tools
        self.proxy_agent: Optional[FlowAgent] = None
        self.bound_agents: List[str] = []
        self.user_preferences: Dict[str, Any] = {}

    async def ensure_user_proxy(self) -> FlowAgent:
        """Ensure user-proxy agent exists and is loaded"""
        if self.proxy_agent is None:
            try:
                # Try to get existing user-proxy agent
                self.proxy_agent = await self.isaa_tools.get_agent("user-proxy")
            except:
                # Create user-proxy agent if it doesn't exist
                await self._create_user_proxy()

        return self.proxy_agent

    async def _create_user_proxy(self):
        """Create the user-proxy agent with default configuration"""
        builder = self.isaa_tools.get_agent_builder("user-proxy")

        system_message = """You are the User Proxy Agent - a persistent, learning companion that grows with the user.
Your role is to:
1. Store and remember user preferences, workflows, and context
2. Provide personalized assistance based on learned patterns
3. Coordinate with other agents to maintain consistency
4. Act as a bridge between the user's intent and agent capabilities

You have access to the user's complete context and can make intelligent decisions about tool usage and agent coordination."""

        builder.with_system_message(system_message)
        builder.with_models(os.environ.get("FAST_MODEL", "openrouter/anthropic/claude-3-haiku"), "openrouter/openai/gpt-5")
        builder.with_checkpointing(enabled=True, interval_seconds=60)

        await self.isaa_tools.register_agent(builder)
        self.proxy_agent = await self.isaa_tools.get_agent("user-proxy")

        # Initialize with user preferences
        self.proxy_agent.set_variable("user.cli_started", datetime.datetime.now().isoformat())
        self.proxy_agent.set_variable("user.preferences", self.user_preferences)

    async def bind_agent(self, agent_name: str, shared_scopes: List[str] = None) -> Dict[str, Any]:
        """Bind an agent to the user-proxy"""
        if shared_scopes is None:
            shared_scopes = ['world', 'results', 'system', 'user']

        proxy_agent = await self.ensure_user_proxy()
        target_agent = await self.isaa_tools.get_agent(agent_name)

        # Use the agent's built-in binding system
        binding_config = proxy_agent.bind(target_agent, shared_scopes=shared_scopes)

        if agent_name not in self.bound_agents:
            self.bound_agents.append(agent_name)

        return binding_config

    async def unbind_agent(self, agent_name: str) -> bool:
        """Unbind an agent from the user-proxy"""
        try:
            agent = await self.isaa_tools.get_agent(agent_name)
            agent.unbind()

            if agent_name in self.bound_agents:
                self.bound_agents.remove(agent_name)

            return True
        except:
            return False

    def set_preference(self, key: str, value: Any):
        """Set a user preference"""
        self.user_preferences[key] = value
        if self.proxy_agent:
            self.proxy_agent.set_variable(f"user.preferences.{key}", value)

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference"""
        return self.user_preferences.get(key, default)

    def get_bound_agents(self) -> List[str]:
        """Get list of bound agents"""
        return self.bound_agents.copy()


# ===== MAIN CLI CLASS =====

class AdvancedIsaaCli:
    """
    Production-ready ISAA CLI with full system integration
    """

    def __init__(self, app_instance: Any, mode=VerbosityMode.STANDARD):
        self.app = app_instance
        self.isaa_tools: Isaatools = app_instance.get_mod("isaa")

        # Core systems
        self.notification_system = NotificationSystem()
        self.task_manager = TaskManager(self.notification_system)
        self.flow_parser = FlowParser(self.isaa_tools)
        self.user_proxy_manager = UserProxyManager(self.isaa_tools)

        # UI and output
        self.printer = ProgressiveTreePrinter(mode=mode)
        self.formatter = EnhancedVerboseOutput(verbose=True, print_func=print)
        self._current_verbosity_mode = mode

        # CLI state
        self.active_agent_name = "user-proxy"
        self.session_id = "cli_session_default"  # Default session without timestamp
        self.workspace_path = __init_cwd__

        # Persistence
        self.state_dir = Path(self.app.data_dir) / "isaa_cli"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # History and completion
        self.history = FileHistory(self.state_dir / "history.txt")
        self.completion_dict = self._build_completer()
        self.prompt_session = PromptSession(
            history=self.history,
            completer=FuzzyCompleter(NestedCompleter.from_nested_dict(self.completion_dict)),
            complete_while_typing=True,
        )

        # Runtime flags
        self._shutdown_requested = False
        self._running = False

        # Load persisted state
        self._load_state()

    def _build_completer(self) -> Dict:
        """Build command completion dictionary"""
        return {
            "/agent": {
                "list": None,
                "create": None,
                "build": None,  # NEW: Interactive agent builder
                "switch": WordCompleter(self.isaa_tools.config.get("agents-name-list", [])),
                "inspect": WordCompleter(self.isaa_tools.config.get("agents-name-list", [])),
                "config": WordCompleter(self.isaa_tools.config.get("agents-name-list", [])),
                "bind": WordCompleter(self.isaa_tools.config.get("agents-name-list", [])),
                "unbind": WordCompleter(self.isaa_tools.config.get("agents-name-list", [])),
                "status": WordCompleter(self.isaa_tools.config.get("agents-name-list", [])),
                "clone": WordCompleter(self.isaa_tools.config.get("agents-name-list", [])),
                "checkpoint": WordCompleter(self.isaa_tools.config.get("agents-name-list", [])),
                "clear-context": WordCompleter(self.isaa_tools.config.get("agents-name-list", [])),  # NEW: Moved from /context
            },
            "/task": {
                "run": None,
                "list": None,
                "attach": None,
                "pause": None,
                "cancel": None,
                "inspect": None,
            },
            "/flow": {
                "design": None,
                "run": WordCompleter(list(self.flow_parser.saved_flows.keys())),
                "list": None,
                "inspect": WordCompleter(list(self.flow_parser.saved_flows.keys())),
                "save": None,
            },
            "/session": {  # NEW: Replaces /context for session management
                "list": None,
                "new": None,  # NEW: Create new session
                "switch": None,
                "inspect": None,
                "clear": None,
                "export": None,
                "import": None,
            },
            "/var": {
                "list": None,
                "get": None,
                "set": None,
                "delete": None,
            },
            "/system": {
                "status": None,
                "verbosity": WordCompleter(["MINIMAL", "STANDARD", "VERBOSE", "DEBUG", "REALTIME"]),
                "notifications": WordCompleter(["on", "off"]),
                "checkpoint": WordCompleter(["create", "load"]),  # Removed list and clean
            },
            "/help": None,
            "/quit": None,
            "/exit": None,
            "/clear": None,
        }

    async def init(self):
        """Initialize the CLI system"""
        self.formatter.print_progress_bar(0, 4, "🔄 Initializing ISAA CLI...")

        # Initialize user proxy
        await self.user_proxy_manager.ensure_user_proxy()
        self.formatter.print_progress_bar(1, 4, "✅ User Proxy Agent loaded")

        # Load saved flows
        self.flow_parser.load_saved_flows()
        self.formatter.print_progress_bar(2, 4, "✅ Workflows loaded")

        # Setup signal handlers
        self._setup_signal_handlers()
        self.formatter.print_progress_bar(3, 4, "✅ Signal handlers configured")

        # Complete
        self.formatter.print_progress_bar(4, 4, "✅ CLI Ready")
        print()

        # Show welcome
        await self.show_welcome()

        # Send startup notification
        self.notification_system.show_notification(
            title="ISAA CLI Started",
            message="Advanced agent interface ready for use",
            notification_type=NotificationType.SUCCESS,
            timeout=2000
        )

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""

        def signal_handler(signum, frame):
            self._shutdown_requested = True
            print("\n🛑 Graceful shutdown requested...")

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def show_welcome(self):
        """Show enhanced welcome message"""
        try:
            terminal_width = os.get_terminal_size().columns
        except OSError:
            terminal_width = 80

        welcome_text = "Advanced ISAA Command Line Interface"

        print(Style.CYAN("═" * terminal_width))
        print(Style.Bold(Style.BLUE(welcome_text.center(terminal_width))))
        print(Style.CYAN("═" * terminal_width))
        print()

        # Enhanced status info
        bound_agents = len(self.user_proxy_manager.bound_agents)
        active_tasks = len([t for t in self.task_manager.tasks.values() if t.status == TaskStatus.RUNNING])
        saved_flows = len(self.flow_parser.saved_flows)

        self.formatter.print_section(
            "System Status",
            f"🏠  Workspace: {self.workspace_path.name}\n"
            f"🤖  Active Agent: {self.active_agent_name}\n"
            f"🔗  Bound Agents: {bound_agents}\n"
            f"⚡  Active Tasks: {active_tasks}\n"
            f"🔄  Saved Flows: {saved_flows}\n"
            f"💬  Session: {self.session_id}"
        )

        # Quick tips
        tips = [
            f"💡 Type {Style.CYAN('/help')} to see all commands",
            f"🔨 Build custom agents with {Style.CYAN('/agent build')}",
            f"🔍 Inspect agents with {Style.CYAN('/agent inspect -d')} for detailed context",
            f"⚡ Use {Style.CYAN('--fast')} or {Style.CYAN('--callback')} flags for advanced execution",
            f"💬 Create new sessions with {Style.CYAN('/session new <name>')}",
        ]

        print()
        for tip in tips:
            print(f"  {tip}")
        print()

        self.formatter.print_info("🚀 Ready for advanced agent orchestration!")

    async def run(self):
        """Main CLI loop with full error handling and graceful shutdown"""
        await self.init()
        self._running = True

        try:
            while not self._shutdown_requested and self._running:
                try:
                    # Update completion
                    await self._update_completion()

                    # Get user input
                    user_input = await self.prompt_session.prompt_async(self._get_prompt_text())

                    if not user_input.strip():
                        continue

                    # Process input
                    if user_input.strip().startswith("!"):
                        await self._handle_shell_command(user_input.strip()[1:])
                    elif user_input.strip().startswith("/"):
                        await self._handle_command(user_input.strip())
                    else:
                        await self._handle_agent_request(user_input.strip())

                except (EOFError, KeyboardInterrupt):
                    if await self._confirm_exit():
                        break

                except Exception as e:
                    self.formatter.print_error(f"An error occurred: {e}")
                    import traceback
                    print(traceback.format_exc())

                    # Send error notification
                    self.notification_system.show_notification(
                        title="CLI Error",
                        message="An unexpected error occurred",
                        notification_type=NotificationType.ERROR,
                        details=NotificationDetails(
                            title="Error Details",
                            content=str(e),
                            data={"traceback": traceback.format_exc()}
                        )
                    )

        finally:
            await self._cleanup()

    async def _update_completion(self):
        """Update command completion with current state"""
        # Update agent names
        agent_names = self.isaa_tools.config.get("agents-name-list", [])
        self.completion_dict["/agent"]["switch"] = WordCompleter(agent_names)
        self.completion_dict["/agent"]["inspect"] = WordCompleter(agent_names)
        self.completion_dict["/agent"]["config"] = WordCompleter(agent_names)
        self.completion_dict["/agent"]["bind"] = WordCompleter(agent_names)
        self.completion_dict["/agent"]["unbind"] = WordCompleter(agent_names)
        self.completion_dict["/agent"]["status"] = WordCompleter(agent_names)

        # Update flow names
        flow_names = list(self.flow_parser.saved_flows.keys())
        self.completion_dict["/flow"]["run"] = WordCompleter(flow_names)
        self.completion_dict["/flow"]["inspect"] = WordCompleter(flow_names)

        # Update task IDs
        task_ids = list(self.task_manager.tasks.keys())
        self.completion_dict["/task"]["attach"] = WordCompleter(task_ids)
        self.completion_dict["/task"]["pause"] = WordCompleter(task_ids)
        self.completion_dict["/task"]["resume"] = WordCompleter(task_ids)
        self.completion_dict["/task"]["cancel"] = WordCompleter(task_ids)
        self.completion_dict["/task"]["inspect"] = WordCompleter(task_ids)

        # Update completer
        self.prompt_session.completer = FuzzyCompleter(
            NestedCompleter.from_nested_dict(self.completion_dict)
        )

    def _get_prompt_text(self) -> HTML:
        """Generate dynamic prompt with rich status information"""
        import html

        # Escape XML-invalid characters
        workspace_name = html.escape(self.workspace_path.name)
        agent_name = html.escape(self.active_agent_name)

        # Add status indicators
        active_tasks = len([t for t in self.task_manager.tasks.values() if t.status == TaskStatus.RUNNING])
        bound_agents = len(self.user_proxy_manager.bound_agents)

        status_indicators = []
        if active_tasks > 0:
            status_indicators.append(f"⚡{active_tasks}")
        if bound_agents > 0:
            status_indicators.append(f"🔗{bound_agents}")

        status_str = " " + " ".join(status_indicators) if status_indicators else ""

        return HTML(
            f"<ansicyan>[</ansicyan>"
            f"<ansigreen>{workspace_name}</ansigreen>"
            f"<ansicyan>]</ansicyan> "
            f"<ansiyellow>({agent_name})</ansiyellow>"
            f"<ansibrightblack>{status_str}</ansibrightblack>"
            f"\n<ansiblue>❯</ansiblue> "
        )

    async def _handle_command(self, command: str):
        """Handle slash commands with comprehensive routing"""
        parts = command.split()
        if not parts:
            return

        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        handlers = {
            "/agent": self._handle_agent_cmd,
            "/task": self._handle_task_cmd,
            "/flow": self._handle_flow_cmd,
            "/session": self._handle_session_cmd,  # NEW: Replaces /context
            "/var": self._handle_var_cmd,
            "/system": self._handle_system_cmd,
            "/help": self._handle_help_cmd,
            "/quit": self._handle_exit_cmd,
            "/exit": self._handle_exit_cmd,
            "/clear": self._handle_clear_cmd,
        }

        if cmd in handlers:
            await handlers[cmd](args)
        else:
            # Try partial matches
            matches = [k for k in handlers.keys() if k.startswith(cmd)]
            if len(matches) == 1:
                await handlers[matches[0]](args)
            elif len(matches) > 1:
                self.formatter.print_error(f"Ambiguous command '{cmd}'. Matches: {', '.join(matches)}")
            else:
                self.formatter.print_error(f"Unknown command: {cmd}")
                self.formatter.print_info("Type '/help' for available commands")

    async def _handle_agent_request(self, request: str):
        """Handle direct agent requests with fast mode and callback support"""
        # Parse inline flags (NEW)
        fast_mode = False
        use_callback = False

        if request.startswith("--fast "):
            fast_mode = True
            request = request[7:]
        elif request.startswith("--callback "):
            use_callback = True
            request = request[11:]

        try:
            # Get active agent
            agent = await self.isaa_tools.get_agent(self.active_agent_name)

            tools_interface = self.isaa_tools.get_tools_interface(self.active_agent_name)
            if tools_interface.vfs.base_dir != self.workspace_path:
                tools_interface.set_base_directory(self.workspace_path)

            # Set progress callback
            agent.set_progress_callback(self.printer.progress_callback)

            # Prepare run kwargs (NEW)
            run_kwargs = {}
            if fast_mode:
                run_kwargs["fast_run"] = True
                self.formatter.print_info("⚡ Fast mode enabled")

            if use_callback:
                # Define dynamic callback
                async def dynamic_callback(context_data):
                    """Interactive callback for context injection"""
                    print(f"\n🔄 Callback triggered")
                    print(f"   Context: {context_data}")
                    additional_context = await self.prompt_session.prompt_async(
                        "Add additional context (or press Enter to skip): "
                    )
                    return additional_context if additional_context else None

                run_kwargs["as_callback"] = dynamic_callback
                self.formatter.print_info("🔄 Callback mode enabled")

            # Start task
            task_id = await self.task_manager.start_task(
                agent=agent,
                prompt=request,
                session_id=self.session_id,
                is_background=False,
                **run_kwargs  # NEW: Pass additional kwargs
            )

            # Wait for completion and show result
            task_info = self.task_manager.get_task(task_id)
            if task_info:
                await self.task_manager.running_tasks[task_id]  # Wait for completion

                if task_info.status == TaskStatus.COMPLETED:
                    self.formatter.print_success("✅ Agent Response:")
                    print(task_info.result)
                elif task_info.status == TaskStatus.FAILED:
                    self.formatter.print_error(f"❌ Agent Error: {task_info.error}")

        except Exception as e:
            self.formatter.print_error(f"Failed to process request: {e}")

    # ===== COMMAND HANDLERS =====

    async def _handle_agent_cmd(self, args: List[str]):
        """Handle /agent commands"""
        if not args:
            await self._show_agent_help()
            return

        sub_cmd = args[0].lower()
        sub_args = args[1:] if len(args) > 1 else []

        if sub_cmd == "list":
            await self._agent_list(sub_args)
        elif sub_cmd == "create":
            await self._agent_create(sub_args)
        elif sub_cmd == "build":  # NEW: Interactive agent builder
            await self._agent_build_interactive(sub_args)
        elif sub_cmd == "switch":
            await self._agent_switch(sub_args)
        elif sub_cmd == "inspect":
            await self._agent_inspect(sub_args)
        elif sub_cmd == "config":
            await self._agent_config(sub_args)
        elif sub_cmd == "bind":
            await self._agent_bind(sub_args)
        elif sub_cmd == "unbind":
            await self._agent_unbind(sub_args)
        elif sub_cmd == "status":
            await (await self.isaa_tools.get_agent(self.active_agent_name)).status(pretty_print=True)
        elif sub_cmd == "clone":
            await self._agent_clone(sub_args)
        elif sub_cmd == "checkpoint":
            await self._agent_checkpoint(sub_args)
        elif sub_cmd == "clear-context":  # NEW: Moved from /context
            await self._agent_clear_context(sub_args)
        else:
            self.formatter.print_error(f"Unknown agent subcommand: {sub_cmd}")
            await self._show_agent_help()

    async def _agent_list(self, args: List[str]):
        """List all agents with enhanced information"""
        detailed = "--detailed" in args or "-d" in args
        bound_only = "--bound" in args

        agents = self.isaa_tools.config.get("agents-name-list", [])
        if not agents:
            self.formatter.print_info("No agents found")
            return

        # Get bound agents for filtering
        bound_agents = self.user_proxy_manager.get_bound_agents()

        if bound_only:
            agents = [a for a in agents if a in bound_agents]

        self.formatter.print_section("Available Agents", "")

        for name in agents:
            # Status indicators
            is_active = name == self.active_agent_name
            is_bound = name in bound_agents

            markers = []
            if is_active:
                markers.append("▶️")
            if is_bound:
                markers.append("🔗")

            marker_str = " ".join(markers) if markers else "  "

            print(f" {marker_str} {Style.GREEN(name)}")

            if detailed:
                try:
                    agent = await self.isaa_tools.get_agent(name)
                    print(f"     📝 System: {agent.amd.system_message[:100]}...")
                    print(f"     🧠 Models: {agent.amd.fast_llm_model} / {agent.amd.complex_llm_model}")
                    if hasattr(agent, 'shared') and 'available_tools' in agent.shared:
                        tool_count = len(agent.shared['available_tools'])
                        print(f"     🛠️  Tools: {tool_count}")
                except Exception:
                    print(f"     ❌ Could not load details")

        print(f"\nTotal: {len(agents)} agents")
        if bound_agents:
            print(f"Bound to user-proxy: {len(bound_agents)} agents")

    async def _agent_create(self, args: List[str]):
        """Interactive agent creation with templates"""
        try:
            # Get agent name
            if args:
                name = args[0]
            else:
                name = await self.prompt_session.prompt_async("Agent name: ")

            if not name:
                self.formatter.print_warning("Agent creation cancelled")
                return

            # Check if agent exists
            existing_agents = self.isaa_tools.config.get("agents-name-list", [])
            if name in existing_agents:
                self.formatter.print_error(f"Agent '{name}' already exists")
                return

            # Template selection
            templates = {
                "1": ("General Assistant", "You are a helpful AI assistant.", os.environ.get("FAST_MODEL", os.environ.get("FAST_MODEL", "openrouter/anthropic/claude-3-haiku")),
                      "openrouter/openai/gpt-5"),
                "2": ("Developer", "You are a senior software developer.", os.environ.get("FAST_MODEL", "openrouter/anthropic/claude-3-haiku"),
                      "openrouter/anthropic/gpt-5"),
                "3": ("Analyst", "You are a data analyst.", os.environ.get("FAST_MODEL", "openrouter/anthropic/claude-3-haiku"),
                      "openrouter/openai/gpt-5"),
                "4": ("Custom", "", "", "")
            }

            print("\n📋 Agent Templates:")
            for key, (title, _, _, _) in templates.items():
                print(f"  {key}. {title}")

            template_choice = await self.prompt_session.prompt_async("Select template (1-4): ")

            if template_choice in templates:
                template_name, system_msg, fast_model, complex_model = templates[template_choice]

                if template_choice == "4":  # Custom
                    system_msg = await self.prompt_session.prompt_async("System message: ")
                    fast_model = await self.prompt_session.prompt_async("Fast model: ",
                                                                        default=os.environ.get("FAST_MODEL", "openrouter/anthropic/claude-3-haiku"))
                    complex_model = await self.prompt_session.prompt_async("Complex model: ",
                                                                           default="openrouter/openai/gpt-5")

                # Create agent
                builder = self.isaa_tools.get_agent_builder(name, working_directory=self.workspace_path)
                builder.with_system_message(system_msg)
                builder.with_models(fast_model, complex_model)
                builder.with_checkpointing(enabled=True)

                await self.isaa_tools.register_agent(builder)

                self.formatter.print_success(f"✅ Agent '{name}' created successfully")

                # Ask to bind to user-proxy
                bind_choice = await self.prompt_session.prompt_async("Bind to user-proxy? (y/N): ")
                if bind_choice.lower() in ['y', 'yes']:
                    await self._agent_bind([name])

                # Ask to switch to new agent
                switch_choice = await self.prompt_session.prompt_async("Switch to new agent? (y/N): ")
                if switch_choice.lower() in ['y', 'yes']:
                    await self._agent_switch([name])

            else:
                self.formatter.print_warning("Invalid template selection")

        except (EOFError, KeyboardInterrupt):
            self.formatter.print_warning("\nAgent creation cancelled")

    async def _agent_build_interactive(self, args: List[str]):
        """Interactive agent builder with comprehensive configuration"""
        try:
            self.formatter.print_section("🔨 Interactive Agent Builder", "")
            print("Let's build your custom agent step by step.\n")

            # Step 1: Name
            name = args[0] if args else await self.prompt_session.prompt_async("Agent name: ")

            # Step 2: Choose base template or start from scratch
            print("\n📋 Base Template:")
            templates = {
                "1": ("Developer", "developer"),
                "2": ("Analyst", "analyst"),
                "3": ("Assistant", "assistant"),
                "4": ("Creative", "creative"),
                "5": ("Executive", "executive"),
                "6": ("From Scratch", "custom")
            }

            for key, (title, _) in templates.items():
                print(f"  {key}. {title}")

            template_choice = await self.prompt_session.prompt_async("Select base (1-6): ")

            if template_choice not in templates:
                self.formatter.print_error("Invalid selection")
                return

            template_name, template_type = templates[template_choice]

            # Initialize builder
            builder = self.isaa_tools.get_agent_builder(name, working_directory=self.workspace_path)

            # Apply template persona if not custom
            if template_type != "custom":
                if template_type == "developer":
                    builder.with_developer_persona()
                elif template_type == "analyst":
                    builder.with_analyst_persona()
                elif template_type == "assistant":
                    builder.with_assistant_persona()
                elif template_type == "creative":
                    builder.with_creative_persona()
                elif template_type == "executive":
                    builder.with_executive_persona()

                print(f"✅ Applied {template_name} persona")

            # Step 3: Models
            print("\n🧠 Models:")
            use_defaults = await self.prompt_session.prompt_async("Use default models? (Y/n): ")

            if use_defaults.lower() not in ['n', 'no']:
                fast_model = os.environ.get("FAST_MODEL", "openrouter/anthropic/claude-3-haiku")
                complex_model = "openrouter/openai/gpt-5"
            else:
                fast_model = await self.prompt_session.prompt_async("Fast model: ")
                complex_model = await self.prompt_session.prompt_async("Complex model: ")

            builder.with_models(fast_model, complex_model)
            print(f"✅ Models configured")

            # Step 4: System Message (if custom or override)
            if template_type == "custom":
                print("\n💬 System Message:")
                system_msg = await self.prompt_session.prompt_async("System message: ")
                builder.with_system_message(system_msg)
            else:
                override_msg = await self.prompt_session.prompt_async("Override system message? (y/N): ")
                if override_msg.lower() in ['y', 'yes']:
                    system_msg = await self.prompt_session.prompt_async("System message: ")
                    builder.with_system_message(system_msg)

            # Step 5: Features
            print("\n⚙️  Features:")

            # Checkpointing
            enable_checkpoint = await self.prompt_session.prompt_async("Enable checkpointing? (Y/n): ")
            if enable_checkpoint.lower() not in ['n', 'no']:
                builder.with_checkpointing(enabled=True, interval_seconds=300)
                print("  ✅ Checkpointing enabled")

            # MCP
            enable_mcp = await self.prompt_session.prompt_async("Enable MCP server? (y/N): ")
            if enable_mcp.lower() in ['y', 'yes']:
                mcp_port = await self.prompt_session.prompt_async("MCP port (default 8001): ", default="8001")
                builder.enable_mcp_server(port=int(mcp_port))
                print(f"  ✅ MCP server enabled on port {mcp_port}")

            # A2A
            enable_a2a = await self.prompt_session.prompt_async("Enable A2A (Agent-to-Agent)? (y/N): ")
            if enable_a2a.lower() in ['y', 'yes']:
                a2a_port = await self.prompt_session.prompt_async("A2A port (default 5001): ", default="5001")
                builder.enable_a2a_server(port=int(a2a_port))
                print(f"  ✅ A2A enabled on port {a2a_port}")

            # Telemetry
            enable_telemetry = await self.prompt_session.prompt_async("Enable telemetry? (y/N): ")
            if enable_telemetry.lower() in ['y', 'yes']:
                builder.enable_telemetry(console_export=True)
                print("  ✅ Telemetry enabled")

            # Step 6: Custom Variables
            print("\n📦 Custom Variables:")
            add_vars = await self.prompt_session.prompt_async("Add custom variables? (y/N): ")
            if add_vars.lower() in ['y', 'yes']:
                variables = {}
                while True:
                    var_name = await self.prompt_session.prompt_async("Variable name (or Enter to finish): ")
                    if not var_name:
                        break
                    var_value = await self.prompt_session.prompt_async(f"Value for '{var_name}': ")
                    variables[var_name] = var_value

                if variables:
                    builder.with_custom_variables(variables)
                    print(f"  ✅ Added {len(variables)} variables")

            # Step 7: Preview & Confirm
            print("\n📋 Agent Configuration Summary:")
            print(f"  Name: {name}")
            print(f"  Template: {template_name}")
            print(f"  Fast Model: {fast_model}")
            print(f"  Complex Model: {complex_model}")

            confirm = await self.prompt_session.prompt_async("\nBuild this agent? (Y/n): ")

            if confirm.lower() in ['n', 'no']:
                self.formatter.print_warning("Agent build cancelled")
                return

            # Build the agent
            print("\n🔨 Building agent...")
            await self.isaa_tools.register_agent(builder)

            self.formatter.print_success(f"✅ Agent '{name}' built successfully!")

            # Post-build options
            bind_choice = await self.prompt_session.prompt_async("Bind to user-proxy? (y/N): ")
            if bind_choice.lower() in ['y', 'yes']:
                await self._agent_bind([name])

            switch_choice = await self.prompt_session.prompt_async("Switch to new agent? (Y/n): ")
            if switch_choice.lower() not in ['n', 'no']:
                await self._agent_switch([name])

        except (EOFError, KeyboardInterrupt):
            self.formatter.print_warning("\nAgent build cancelled")
        except Exception as e:
            self.formatter.print_error(f"Failed to build agent: {e}")

    async def _agent_switch(self, args: List[str]):
        """Switch active agent"""
        if not args:
            self.formatter.print_error("Usage: /agent switch <agent_name>")
            return

        agent_name = args[0]
        agents = self.isaa_tools.config.get("agents-name-list", [])

        if agent_name in agents:
            self.active_agent_name = agent_name
            self.formatter.print_success(f"✅ Switched to agent: {agent_name}")

            # Send notification
            self.notification_system.show_notification(
                title="Agent Switched",
                message=f"Now using agent: {agent_name}",
                notification_type=NotificationType.INFO,
                timeout=1500
            )
        else:
            self.formatter.print_error(f"Agent '{agent_name}' not found")

    async def _agent_inspect(self, args: List[str]):
        """Inspect agent with comprehensive details including context distribution"""
        # Parse arguments
        agent_name = None
        detailed = False

        for arg in args:
            if arg in ["-d", "--detailed"]:
                detailed = True
            elif not agent_name:
                agent_name = arg

        if not agent_name:
            agent_name = self.active_agent_name

        try:
            agent = await self.isaa_tools.get_agent(agent_name)

            self.formatter.print_section(f"Agent Inspection: {agent_name}", "")

            # Basic info
            print(f"🤖 Name: {agent.amd.name}")
            print(f"🧠 Fast Model: {agent.amd.fast_llm_model}")
            print(f"🧠 Complex Model: {agent.amd.complex_llm_model}")

            # Token and cost statistics
            print(f"\n💰 Usage Statistics:")
            print(f"   Total Cost: ${agent.total_cost_accumulated:.4f}")
            print(f"   Total LLM Calls: {agent.total_llm_calls}")
            print(f"   Tokens In: {agent.total_tokens_in:,}")
            print(f"   Tokens Out: {agent.total_tokens_out:,}")
            print(f"   Total Tokens: {agent.total_tokens_in + agent.total_tokens_out:,}")

            # Binding status
            is_bound = agent_name in self.user_proxy_manager.get_bound_agents()
            print(f"🔗 Bound to User-Proxy: {'Yes' if is_bound else 'No'}")

            # Tools and capabilities
            if hasattr(agent, 'shared') and 'available_tools' in agent.shared:
                tool_count = len(agent.shared['available_tools'])
                print(f"🛠️  Available Tools: {tool_count}")
                if tool_count > 0:
                    tools_preview = ', '.join(agent.shared['available_tools'][:5])
                    if tool_count > 5:
                        tools_preview += f"... ({tool_count - 5} more)"
                    print(f"     {tools_preview}")

            # Recent tasks
            agent_tasks = [t for t in self.task_manager.tasks.values() if t.agent_name == agent_name]
            recent_tasks = sorted(agent_tasks, key=lambda t: t.created_at, reverse=True)[:3]

            if recent_tasks:
                print(f"📋 Recent Tasks: {len(recent_tasks)}")
                for task in recent_tasks:
                    status_emoji = {"completed": "✅", "failed": "❌", "running": "⏳", "queued": "📋", "paused": "⏸️"}.get(
                        task.status, "❓")
                    print(f"     {status_emoji} {task.task_id}: {task.prompt[:50]}...")

            # Context Distribution (NEW - Enhanced)
            print(f"\n📊 Context Distribution:")
            try:
                context_overview = await agent.get_context_overview(display=False)

                if context_overview and 'token_summary' in context_overview:
                    token_summary = context_overview['token_summary']
                    print(f"   Total: ~{token_summary['total_tokens']:,} tokens")

                    # Show breakdown with visual bars
                    breakdown = token_summary.get('breakdown', {})
                    percentages = token_summary.get('percentage_breakdown', {})

                    for component, token_count in breakdown.items():
                        if token_count > 0:
                            percentage = percentages.get(component, 0)
                            bar_length = int(percentage / 5)  # 20 chars max (100% / 5)
                            bar = "█" * bar_length
                            print(f"   {component:20s}: {bar:20s} {token_count:6,} ({percentage:5.1f}%)")
                else:
                    print(f"   No context data available")
            except Exception as e:
                print(f"   Error retrieving context: {e}")

            # Session Information (NEW)
            if hasattr(agent, 'context_manager') and agent.context_manager:
                sessions = agent.context_manager.session_managers
                print(f"\n💬 Active Sessions: {len(sessions)}")
                for session_id, session in list(sessions.items())[:5]:  # Show max 5
                    history_len = len(session.history) if hasattr(session, 'history') else 0
                    print(f"   {session_id}: {history_len} messages")
                if len(sessions) > 5:
                    print(f"   ... and {len(sessions) - 5} more")

            # Detailed Mode (-d flag) (NEW)
            if detailed:
                print(f"\n🔍 Detailed Information:")

                # Full context breakdown with display
                try:
                    await agent.get_context_overview(display=True)
                except:
                    pass

                # All tools with descriptions
                if hasattr(agent, 'shared') and 'available_tools' in agent.shared:
                    print(f"\n🛠️  All Tools:")
                    for tool_name in agent.shared['available_tools']:
                        print(f"   - {tool_name}")

                # Variable scopes
                if hasattr(agent, 'context_manager') and hasattr(agent.context_manager, 'variable_manager'):
                    var_manager = agent.context_manager.variable_manager
                    print(f"\n📦 Variables:")
                    for scope in ['global', 'agent', 'session', 'task']:
                        vars_in_scope = var_manager.get_all_variables(scope)
                        if vars_in_scope:
                            print(f"   {scope}: {len(vars_in_scope)} variables")
                            for key in list(vars_in_scope.keys())[:3]:
                                print(f"      - {key}")
                            if len(vars_in_scope) > 3:
                                print(f"      ... and {len(vars_in_scope) - 3} more")

        except Exception as e:
            self.formatter.print_error(f"Failed to inspect agent '{agent_name}': {e}")

    async def _agent_config(self, args: List[str]):
        """Configure agent interactively"""
        agent_name = args[0] if args else self.active_agent_name

        try:
            agent = await self.isaa_tools.get_agent(agent_name)

            self.formatter.print_section(f"Agent Configuration: {agent_name}", "")

            # Show current persona if exists
            if agent.amd.persona:
                print(f"Current persona: {agent.amd.persona.name}")
                print(f"Style: {agent.amd.persona.style}, Tone: {agent.amd.persona.tone}")

            # Configuration options
            print("\n📋 Configuration Options:")
            print("  1. Set persona")
            print("  2. Set response format")
            print("  3. Configure models")
            print("  4. Exit")

            choice = await self.prompt_session.prompt_async("Select option (1-4): ")

            if choice == "1":
                name = await self.prompt_session.prompt_async("Persona name: ", default="Assistant")
                style = await self.prompt_session.prompt_async("Style (professional/casual/technical): ",
                                                               default="professional")
                tone = await self.prompt_session.prompt_async("Tone (friendly/formal/helpful): ", default="friendly")

                agent.set_persona(name, style, tone)
                self.formatter.print_success("✅ Persona updated")

            elif choice == "2":
                formats = agent.get_available_formats()
                print("\nAvailable formats:", ", ".join(formats['formats']))
                print("Available lengths:", ", ".join(formats['lengths']))

                response_format = await self.prompt_session.prompt_async("Response format: ", default="text-only")
                text_length = await self.prompt_session.prompt_async("Text length: ", default="chat-conversation")

                agent.set_response_format(response_format, text_length)
                self.formatter.print_success("✅ Response format updated")

            elif choice == "3":
                current_fast = agent.amd.fast_llm_model
                current_complex = agent.amd.complex_llm_model

                fast_model = await self.prompt_session.prompt_async(f"Fast model [{current_fast}]: ",
                                                                    default=current_fast)
                complex_model = await self.prompt_session.prompt_async(f"Complex model [{current_complex}]: ",
                                                                       default=current_complex)

                # This would require rebuilding the agent - simplified for demo
                self.formatter.print_info("Model configuration requires agent recreation")

        except Exception as e:
            self.formatter.print_error(f"Failed to configure agent: {e}")

    async def _agent_bind(self, args: List[str]):
        """Bind agent to user-proxy"""
        if not args:
            self.formatter.print_error("Usage: /agent bind <agent_name> [--scopes scope1,scope2,...]")
            return

        agent_name = args[0]

        # Parse scopes
        shared_scopes = ['world', 'results', 'system', 'user']  # Default scopes
        if "--scopes" in args:
            scope_idx = args.index("--scopes")
            if scope_idx + 1 < len(args):
                scopes_str = args[scope_idx + 1]
                shared_scopes = [s.strip() for s in scopes_str.split(',')]

        try:
            binding_config = await self.user_proxy_manager.bind_agent(agent_name, shared_scopes)

            self.formatter.print_success(f"✅ Agent '{agent_name}' bound to user-proxy")
            print(f"   Binding ID: {binding_config.get('binding_id', 'unknown')}")
            print(f"   Shared Scopes: {', '.join(shared_scopes)}")

            # Send notification
            self.notification_system.show_notification(
                title="Agent Bound",
                message=f"Agent '{agent_name}' is now connected to your preferences",
                notification_type=NotificationType.SUCCESS,
                timeout=2000
            )

        except Exception as e:
            self.formatter.print_error(f"Failed to bind agent '{agent_name}': {e}")

    async def _agent_unbind(self, args: List[str]):
        """Unbind agent from user-proxy"""
        if not args:
            self.formatter.print_error("Usage: /agent unbind <agent_name>")
            return

        agent_name = args[0]

        success = await self.user_proxy_manager.unbind_agent(agent_name)

        if success:
            self.formatter.print_success(f"✅ Agent '{agent_name}' unbound from user-proxy")
        else:
            self.formatter.print_error(f"Failed to unbind agent '{agent_name}'")

    async def _agent_clone(self, args: List[str]):
        """Clone an existing agent"""
        if len(args) < 2:
            self.formatter.print_error("Usage: /agent clone <source_agent> <new_name>")
            return

        source_name, new_name = args[0], args[1]

        try:
            # Get source agent
            source_agent = await self.isaa_tools.get_agent(source_name)

            # Create new agent with same configuration
            builder = self.isaa_tools.get_agent_builder(new_name)
            builder.with_system_message(source_agent.amd.system_message)
            builder.with_models(source_agent.amd.fast_llm_model, source_agent.amd.complex_llm_model)
            builder.with_checkpointing(enabled=True)

            # Copy persona if exists
            if source_agent.amd.persona:
                builder.with_persona(source_agent.amd.persona)

            await self.isaa_tools.register_agent(builder)

            # Copy checkpoint/state if available
            try:
                checkpoint = await source_agent._create_checkpoint()
                new_agent = await self.isaa_tools.get_agent(new_name)
                await new_agent._restore_from_checkpoint_simplified(checkpoint, auto_restore_history=True)
            except:
                pass  # Clone without checkpoint if it fails

            self.formatter.print_success(f"✅ Agent '{new_name}' cloned from '{source_name}'")

        except Exception as e:
            self.formatter.print_error(f"Failed to clone agent: {e}")

    async def _agent_checkpoint(self, args: List[str]):
        """Create agent checkpoint"""
        agent_name = args[0] if args else self.active_agent_name

        try:
            agent = await self.isaa_tools.get_agent(agent_name)
            checkpoint = await agent._create_checkpoint()
            await agent._save_checkpoint(checkpoint)

            self.formatter.print_success(f"✅ Checkpoint created for agent '{agent_name}'")

        except Exception as e:
            self.formatter.print_error(f"Failed to create checkpoint: {e}")

    async def _agent_clear_context(self, args: List[str]):
        """Clear agent context with confirmation (moved from /context)"""
        agent_name = args[0] if args else self.active_agent_name
        session_id = None

        # Check for session-specific clear
        if "--session" in args:
            session_idx = args.index("--session")
            if session_idx + 1 < len(args):
                session_id = args[session_idx + 1]

        try:
            agent = await self.isaa_tools.get_agent(agent_name)

            # Confirm action
            confirm_msg = f"Clear context for agent '{agent_name}'"
            if session_id:
                confirm_msg += f" (session: {session_id})"
            confirm_msg += "? (y/N): "

            confirmation = await self.prompt_session.prompt_async(confirm_msg)

            if confirmation.lower() in ['y', 'yes']:
                success = agent.clear_context(session_id)

                if success:
                    self.formatter.print_success("✅ Context cleared successfully")

                    # Send notification
                    self.notification_system.show_notification(
                        title="Context Cleared",
                        message=f"Context cleared for {agent_name}",
                        notification_type=NotificationType.INFO,
                        timeout=1500
                    )
                else:
                    self.formatter.print_error("Failed to clear context")
            else:
                self.formatter.print_info("Context clear cancelled")

        except Exception as e:
            self.formatter.print_error(f"Failed to clear context: {e}")

    async def _handle_task_cmd(self, args: List[str]):
        """Handle /task commands"""
        if not args:
            await self._show_task_help()
            return

        sub_cmd = args[0].lower()
        sub_args = args[1:] if len(args) > 1 else []

        if sub_cmd == "run":
            await self._task_run(sub_args)
        elif sub_cmd == "list":
            await self._task_list(sub_args)
        elif sub_cmd == "attach":
            await self._task_attach(sub_args)
        elif sub_cmd == "pause":
            await self._task_pause(sub_args)
        elif sub_cmd == "cancel":
            await self._task_cancel(sub_args)
        elif sub_cmd == "inspect":
            await self._task_inspect(sub_args)
        else:
            self.formatter.print_error(f"Unknown task subcommand: {sub_cmd}")
            await self._show_task_help()

    async def _task_run(self, args: List[str]):
        """Run a task with full option support including fast mode and callback"""
        # Parse arguments
        is_background = "--bg" in args or "--background" in args
        is_watch = "--watch" in args
        fast_mode = "--fast" in args  # NEW: Fast mode flag
        use_callback = "--callback" in args  # NEW: Callback flag
        agent_name = None

        # Extract agent name
        if "--agent" in args:
            agent_idx = args.index("--agent")
            if agent_idx + 1 < len(args):
                agent_name = args[agent_idx + 1]

        # Get prompt (everything not starting with --)
        prompt_parts = []
        skip_next = False
        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if arg.startswith("--"):
                if arg == "--agent":
                    skip_next = True
                continue
            prompt_parts.append(arg)

        if not prompt_parts:
            self.formatter.print_error("Usage: /task run \"prompt\" [--agent name] [--bg] [--watch] [--fast] [--callback]")
            return

        prompt = " ".join(prompt_parts)
        agent_name = agent_name or self.active_agent_name

        try:
            # Get agent
            agent = await self.isaa_tools.get_agent(agent_name)
            tools_interface = self.isaa_tools.get_tools_interface(agent_name)
            if tools_interface.vfs.base_dir != self.workspace_path:
                tools_interface.set_base_directory(self.workspace_path)

            # Set progress callback if watching
            if is_watch or not is_background:
                agent.set_progress_callback(self.printer.progress_callback)

            # Prepare run kwargs (NEW)
            run_kwargs = {}
            if fast_mode:
                run_kwargs["fast_run"] = True
                self.formatter.print_info("⚡ Fast mode enabled")

            if use_callback:
                # Define dynamic callback
                async def dynamic_callback(context_data):
                    """Interactive callback for context injection"""
                    print(f"\n🔄 Callback triggered")
                    print(f"   Context: {context_data}")
                    additional_context = await self.prompt_session.prompt_async(
                        "Add additional context (or press Enter to skip): "
                    )
                    return additional_context if additional_context else None

                run_kwargs["as_callback"] = dynamic_callback
                self.formatter.print_info("🔄 Callback mode enabled")

            # Start task
            task_id = await self.task_manager.start_task(
                agent=agent,
                prompt=prompt,
                session_id=self.session_id,
                is_background=is_background,
                **run_kwargs  # NEW: Pass additional kwargs
            )

            if is_background:
                self.formatter.print_success(f"✅ Background task {task_id} started")
                print(f"   Agent: {agent_name}")
                print(f"   Prompt: {prompt[:100]}...")
                print(f"   Use '/task attach {task_id}' to monitor progress")
            else:
                # Wait for completion
                task_info = self.task_manager.get_task(task_id)
                if task_info:
                    await self.task_manager.running_tasks[task_id]

                    if task_info.status == TaskStatus.COMPLETED:
                        self.formatter.print_success("✅ Task completed:")
                        print(task_info.result)
                    elif task_info.status == TaskStatus.FAILED:
                        self.formatter.print_error(f"❌ Task failed: {task_info.error}")

        except Exception as e:
            self.formatter.print_error(f"Failed to run task: {e}")

    async def _task_list(self, args: List[str]):
        """List all tasks with filtering options"""
        # Parse filters
        status_filter = None
        agent_filter = None

        if "--status" in args:
            status_idx = args.index("--status")
            if status_idx + 1 < len(args):
                status_str = args[status_idx + 1]
                try:
                    status_filter = TaskStatus(status_str.lower())
                except ValueError:
                    self.formatter.print_error(f"Invalid status: {status_str}")
                    return

        if "--agent" in args:
            agent_idx = args.index("--agent")
            if agent_idx + 1 < len(args):
                agent_filter = args[agent_idx + 1]

        # Get tasks
        tasks = self.task_manager.list_tasks(status_filter)

        if agent_filter:
            tasks = [t for t in tasks if t.agent_name == agent_filter]

        if not tasks:
            self.formatter.print_info("No tasks found")
            return

        # Display tasks
        self.formatter.print_section("Task List", "")

        print(f"{'ID':<12} {'Agent':<15} {'Status':<10} {'Duration':<10} {'Prompt':<50}")
        print("-" * 100)

        for task in tasks:
            # Calculate duration
            if task.completed_at and task.started_at:
                duration = (task.completed_at - task.started_at).total_seconds()
                duration_str = f"{duration:.1f}s"
            elif task.started_at:
                duration = (datetime.datetime.now() - task.started_at).total_seconds()
                duration_str = f"{duration:.1f}s*"
            else:
                duration_str = "-"

            # Status emoji
            status_emoji = {
                "completed": "✅",
                "failed": "❌",
                "running": "⏳",
                "queued": "📋",
                "paused": "⏸️",
                "cancelled": "🚫"
            }.get(task.status.value, "❓")

            # Truncate prompt
            prompt_preview = task.prompt[:47] + "..." if len(task.prompt) > 50 else task.prompt

            print(
                f"{task.task_id:<12} {task.agent_name:<15} {status_emoji}{task.status.value:<9} {duration_str:<10} {prompt_preview:<50}")

        print(f"\nTotal: {len(tasks)} tasks")

    async def _task_attach(self, args: List[str]):
        """Attach to a running background task"""
        if not args:
            self.formatter.print_error("Usage: /task attach <task_id>")
            return

        task_id = args[0]

        def output_callback(message: str):
            print(message)

        success = await self.task_manager.attach_to_task(task_id, output_callback)

        if not success:
            self.formatter.print_error(f"Cannot attach to task '{task_id}' (not found or not running)")

    async def _task_pause(self, args: List[str]):
        """Pause a running task"""
        if not args:
            self.formatter.print_error("Usage: /task pause <task_id>")
            return

        task_id = args[0]
        success = await self.task_manager.pause_task(task_id)

        if success:
            self.formatter.print_success(f"✅ Task {task_id} paused")
        else:
            self.formatter.print_error(f"Cannot pause task '{task_id}'")

    async def _task_cancel(self, args: List[str]):
        """Cancel a running task"""
        if not args:
            self.formatter.print_error("Usage: /task cancel <task_id>")
            return

        task_id = args[0]
        success = await self.task_manager.cancel_task(task_id)

        if success:
            self.formatter.print_success(f"✅ Task {task_id} cancelled")
        else:
            self.formatter.print_error(f"Cannot cancel task '{task_id}'")

    async def _task_inspect(self, args: List[str]):
        """Inspect a task with full details"""
        if not args:
            self.formatter.print_error("Usage: /task inspect <task_id>")
            return

        task_id = args[0]
        task_info = self.task_manager.get_task(task_id)

        if not task_info:
            self.formatter.print_error(f"Task '{task_id}' not found")
            return

        self.formatter.print_section(f"Task Inspection: {task_id}", "")

        print(f"🆔 ID: {task_info.task_id}")
        print(f"🤖 Agent: {task_info.agent_name}")
        print(f"📅 Created: {task_info.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

        if task_info.started_at:
            print(f"▶️  Started: {task_info.started_at.strftime('%Y-%m-%d %H:%M:%S')}")

        if task_info.completed_at:
            print(f"⏹️  Completed: {task_info.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
            duration = (task_info.completed_at - task_info.started_at).total_seconds() if task_info.started_at else 0
            print(f"⏱️  Duration: {duration:.1f} seconds")

        status_emoji = {
            "completed": "✅",
            "failed": "❌",
            "running": "⏳",
            "queued": "📋",
            "paused": "⏸️",
            "cancelled": "🚫"
        }.get(task_info.status.value, "❓")

        print(f"📊 Status: {status_emoji} {task_info.status.value}")
        print(f"🎯 Background: {'Yes' if task_info.is_background else 'No'}")
        print(f"💬 Session: {task_info.session_id}")

        print(f"\n📝 Prompt:")
        print(f"   {task_info.prompt}")

        if task_info.result:
            print(f"\n✅ Result:")
            result_preview = task_info.result[:500] + "..." if len(task_info.result) > 500 else task_info.result
            print(f"   {result_preview}")

        if task_info.error:
            print(f"\n❌ Error:")
            print(f"   {task_info.error}")

    async def _handle_flow_cmd(self, args: List[str]):
        """Handle /flow commands"""
        if not args:
            await self._show_flow_help()
            return

        sub_cmd = args[0].lower()
        sub_args = args[1:] if len(args) > 1 else []

        if sub_cmd == "design":
            await self._flow_design(sub_args)
        elif sub_cmd == "run":
            await self._flow_run(sub_args)
        elif sub_cmd == "list":
            await self._flow_list(sub_args)
        elif sub_cmd == "inspect":
            await self._flow_inspect(sub_args)
        elif sub_cmd == "save":
            await self._flow_save(sub_args)
        else:
            self.formatter.print_error(f"Unknown flow subcommand: {sub_cmd}")
            await self._show_flow_help()

    async def _flow_design(self, args: List[str]):
        """Interactive flow design with syntax help"""
        try:
            self.formatter.print_section("Flow Designer", "Create complex agent workflows")

            print("🔄 Flow Syntax Guide:")
            print("   >> : Sequential execution (agent1 >> agent2)")
            print("   +  : Parallel execution (agent1 + agent2)")
            print("   &  : Parallel execution alias")
            print("   |  : Error handling/fallback (primary | fallback)")
            print("   %  : Conditional else branch")
            print("   CF(Model) : Format output using Pydantic model")
            print("   IS(key, value) : Condition check")
            print("   Function(func) : Custom function")
            print()

            # Get flow name
            if args:
                flow_name = args[0]
            else:
                flow_name = await self.prompt_session.prompt_async("Flow name: ")

            if not flow_name:
                self.formatter.print_warning("Flow design cancelled")
                return

            # Get flow definition
            print("Enter flow definition (examples below):")
            print("  writer >> reviewer")
            print("  (data_fetcher + analyzer) >> reporter")
            print("  validator >> IS('valid', True) >> processor % error_handler")
            print()

            flow_definition = await self.prompt_session.prompt_async("Flow definition: ")

            if not flow_definition:
                self.formatter.print_warning("Flow design cancelled")
                return

            # Validate flow syntax
            try:
                chain = await self.flow_parser.parse_flow_string(flow_definition)

                # Show flow graph
                print("\n🔍 Flow Structure:")
                chain.print_graph()

                # Ask to save
                save_choice = await self.prompt_session.prompt_async("Save this flow? (Y/n): ")
                if save_choice.lower() not in ['n', 'no']:
                    self.flow_parser.save_flow(flow_name, flow_definition)
                    self.formatter.print_success(f"✅ Flow '{flow_name}' saved")

            except Exception as e:
                self.formatter.print_error(f"Flow syntax error: {e}")

        except (EOFError, KeyboardInterrupt):
            self.formatter.print_warning("\nFlow design cancelled")

    async def _flow_run(self, args: List[str]):
        """Run a saved flow"""
        if not args:
            self.formatter.print_error("Usage: /flow run <flow_name> [--input \"input_text\"]")
            return

        flow_name = args[0]

        # Parse input
        input_text = "Execute workflow"  # Default
        if "--input" in args:
            input_idx = args.index("--input")
            if input_idx + 1 < len(args):
                input_text = args[input_idx + 1]

        try:
            result = await self.flow_parser.run_flow(flow_name, input_text, session_id=self.session_id)

            self.formatter.print_success(f"✅ Flow '{flow_name}' completed:")
            print(result)

            # Send completion notification
            self.notification_system.show_notification(
                title="Workflow Completed",
                message=f"Flow '{flow_name}' finished successfully",
                notification_type=NotificationType.SUCCESS,
                details=NotificationDetails(
                    title="Workflow Results",
                    content=str(result)[:200] + "..." if len(str(result)) > 200 else str(result),
                    data={"flow_name": flow_name, "input": input_text}
                )
            )

        except Exception as e:
            self.formatter.print_error(f"Flow execution failed: {e}")

            # Send error notification
            self.notification_system.show_notification(
                title="Workflow Failed",
                message=f"Flow '{flow_name}' encountered an error",
                notification_type=NotificationType.ERROR,
                details=NotificationDetails(
                    title="Error Details",
                    content=str(e),
                    data={"flow_name": flow_name, "error": str(e)}
                )
            )

    async def _flow_list(self, args: List[str]):
        """List all saved flows"""
        flows = self.flow_parser.list_flows()

        if not flows:
            self.formatter.print_info("No saved flows found")
            return

        self.formatter.print_section("Saved Workflows", "")

        for name, definition in flows.items():
            print(f"🔄 {Style.GREEN(name)}")
            print(f"   {definition}")

        print(f"\nTotal: {len(flows)} flows")

    async def _flow_inspect(self, args: List[str]):
        """Inspect a saved flow with visualization"""
        if not args:
            self.formatter.print_error("Usage: /flow inspect <flow_name>")
            return

        flow_name = args[0]
        flows = self.flow_parser.list_flows()

        if flow_name not in flows:
            self.formatter.print_error(f"Flow '{flow_name}' not found")
            return

        flow_definition = flows[flow_name]

        try:
            chain = await self.flow_parser.parse_flow_string(flow_definition)

            self.formatter.print_section(f"Flow Inspection: {flow_name}", "")

            print(f"📝 Definition: {flow_definition}")
            print(f"\n🔍 Structure:")

            chain.print_graph()

        except Exception as e:
            self.formatter.print_error(f"Failed to parse flow: {e}")

    async def _flow_save(self, args: List[str]):
        """Save a flow from command line"""
        if len(args) < 2:
            self.formatter.print_error("Usage: /flow save <name> <definition>")
            return

        flow_name = args[0]
        flow_definition = " ".join(args[1:])

        try:
            # Validate syntax
            await self.flow_parser.parse_flow_string(flow_definition)

            # Save flow
            self.flow_parser.save_flow(flow_name, flow_definition)
            self.formatter.print_success(f"✅ Flow '{flow_name}' saved")

        except Exception as e:
            self.formatter.print_error(f"Invalid flow syntax: {e}")

    async def _handle_session_cmd(self, args: List[str]):
        """Handle /session commands for session-specific management"""
        if not args:
            await self._show_session_help()
            return

        sub_cmd = args[0].lower()
        sub_args = args[1:] if len(args) > 1 else []

        if sub_cmd == "list":
            await self._session_list(sub_args)
        elif sub_cmd == "new":
            await self._session_new(sub_args)
        elif sub_cmd == "switch":
            await self._session_switch(sub_args)
        elif sub_cmd == "inspect":
            await self._session_inspect(sub_args)
        elif sub_cmd == "clear":
            await self._session_clear(sub_args)
        elif sub_cmd == "export":
            await self._session_export(sub_args)
        elif sub_cmd == "import":
            await self._session_import(sub_args)
        else:
            self.formatter.print_error(f"Unknown session subcommand: {sub_cmd}")
            await self._show_session_help()

    async def _session_list(self, args: List[str]):
        """List all sessions for active agent"""
        agent_name = args[0] if args else self.active_agent_name

        try:
            agent = await self.isaa_tools.get_agent(agent_name)

            if hasattr(agent, 'context_manager') and agent.context_manager:
                sessions = agent.context_manager.session_managers

                self.formatter.print_section(f"Sessions: {agent_name}", "")

                if not sessions:
                    self.formatter.print_info("No active sessions")
                    return

                for session_id, session in sessions.items():
                    history_len = len(session.history) if hasattr(session, 'history') else 0
                    is_active = session_id == self.session_id
                    marker = "▶️" if is_active else "  "

                    print(f"{marker} {session_id}: {history_len} messages")

                    # Show last message preview
                    if history_len > 0 and hasattr(session, 'history'):
                        last_msg = session.history[-1]
                        role = last_msg.get('role', 'unknown')
                        content = last_msg.get('content', '')[:50]
                        print(f"     Last: [{role}] {content}...")
            else:
                self.formatter.print_error("Agent has no session manager")

        except Exception as e:
            self.formatter.print_error(f"Failed to list sessions: {e}")

    async def _session_new(self, args: List[str]):
        """Create a new session with custom name"""
        if not args:
            self.formatter.print_error("Usage: /session new <session_name>")
            return

        session_name = args[0]
        agent_name = args[1] if len(args) > 1 else self.active_agent_name

        try:
            agent = await self.isaa_tools.get_agent(agent_name)

            if hasattr(agent, 'context_manager') and agent.context_manager:
                # Initialize new session
                await agent.initialize_session_context(session_name)

                self.formatter.print_success(f"✅ New session created: {session_name}")

                # Ask if user wants to switch to new session
                switch_choice = await self.prompt_session.prompt_async("Switch to new session? (Y/n): ")
                if switch_choice.lower() not in ['n', 'no']:
                    self.session_id = session_name
                    self.formatter.print_success(f"✅ Switched to session: {session_name}")

                    # Send notification
                    self.notification_system.show_notification(
                        title="Session Created",
                        message=f"New session '{session_name}' created and activated",
                        notification_type=NotificationType.SUCCESS,
                        timeout=2000
                    )
            else:
                self.formatter.print_error("Agent has no session manager")

        except Exception as e:
            self.formatter.print_error(f"Failed to create session: {e}")

    async def _session_switch(self, args: List[str]):
        """Switch to a different session"""
        if not args:
            self.formatter.print_error("Usage: /session switch <session_id>")
            return

        new_session_id = args[0]
        self.session_id = new_session_id
        self.formatter.print_success(f"✅ Switched to session: {new_session_id}")

    async def _session_inspect(self, args: List[str]):
        """Inspect a specific session"""
        if not args:
            self.formatter.print_error("Usage: /session inspect <session_id>")
            return

        session_id = args[0]
        agent_name = args[1] if len(args) > 1 else self.active_agent_name

        try:
            agent = await self.isaa_tools.get_agent(agent_name)

            if hasattr(agent, 'context_manager') and agent.context_manager:
                sessions = agent.context_manager.session_managers

                if session_id not in sessions:
                    self.formatter.print_error(f"Session '{session_id}' not found")
                    return

                session = sessions[session_id]
                history_len = len(session.history) if hasattr(session, 'history') else 0

                self.formatter.print_section(f"Session: {session_id}", "")
                print(f"Agent: {agent_name}")
                print(f"Messages: {history_len}")

                # Show recent messages
                if history_len > 0 and hasattr(session, 'history'):
                    print(f"\nRecent Messages:")
                    for msg in session.history[-5:]:  # Last 5 messages
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')[:100]
                        timestamp = msg.get('timestamp', 'N/A')
                        print(f"  [{role}] {content}...")
                        print(f"     Time: {timestamp}")
            else:
                self.formatter.print_error("Agent has no session manager")

        except Exception as e:
            self.formatter.print_error(f"Failed to inspect session: {e}")

    async def _session_clear(self, args: List[str]):
        """Clear a specific session"""
        if not args:
            self.formatter.print_error("Usage: /session clear <session_id>")
            return

        session_id = args[0]
        agent_name = args[1] if len(args) > 1 else self.active_agent_name

        try:
            agent = await self.isaa_tools.get_agent(agent_name)

            # Confirm action
            confirmation = await self.prompt_session.prompt_async(
                f"Clear session '{session_id}'? (y/N): "
            )

            if confirmation.lower() in ['y', 'yes']:
                success = agent.clear_context(session_id)

                if success:
                    self.formatter.print_success(f"✅ Session '{session_id}' cleared")
                else:
                    self.formatter.print_error("Failed to clear session")
            else:
                self.formatter.print_info("Session clear cancelled")

        except Exception as e:
            self.formatter.print_error(f"Failed to clear session: {e}")

    async def _session_export(self, args: List[str]):
        """Export session to file"""
        if len(args) < 2:
            self.formatter.print_error("Usage: /session export <session_id> <file_path>")
            return

        session_id = args[0]
        file_path = args[1]
        agent_name = args[2] if len(args) > 2 else self.active_agent_name

        try:
            agent = await self.isaa_tools.get_agent(agent_name)

            if hasattr(agent, 'context_manager') and agent.context_manager:
                sessions = agent.context_manager.session_managers

                if session_id not in sessions:
                    self.formatter.print_error(f"Session '{session_id}' not found")
                    return

                session = sessions[session_id]

                # Export session history
                import json
                export_data = {
                    "session_id": session_id,
                    "agent_name": agent_name,
                    "history": session.history if hasattr(session, 'history') else []
                }

                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)

                self.formatter.print_success(f"✅ Session exported to: {file_path}")
            else:
                self.formatter.print_error("Agent has no session manager")

        except Exception as e:
            self.formatter.print_error(f"Failed to export session: {e}")

    async def _session_import(self, args: List[str]):
        """Import session from file"""
        if not args:
            self.formatter.print_error("Usage: /session import <file_path>")
            return

        file_path = args[0]

        try:
            import json
            with open(file_path, 'r') as f:
                import_data = json.load(f)

            session_id = import_data.get('session_id')
            agent_name = import_data.get('agent_name', self.active_agent_name)
            history = import_data.get('history', [])

            agent = await self.isaa_tools.get_agent(agent_name)

            if hasattr(agent, 'context_manager') and agent.context_manager:
                # Initialize session if not exists
                await agent.initialize_session_context(session_id)

                # Add history
                session = agent.context_manager.session_managers.get(session_id)
                if session and hasattr(session, 'history'):
                    session.history.extend(history)

                self.formatter.print_success(f"✅ Session imported: {session_id} ({len(history)} messages)")
            else:
                self.formatter.print_error("Agent has no session manager")

        except Exception as e:
            self.formatter.print_error(f"Failed to import session: {e}")

    async def _handle_var_cmd(self, args: List[str]):
        """Handle /var commands for variable management"""
        if not args:
            await self._show_var_help()
            return

        sub_cmd = args[0].lower()
        sub_args = args[1:] if len(args) > 1 else []

        if sub_cmd == "list":
            await self._var_list(sub_args)
        elif sub_cmd == "get":
            await self._var_get(sub_args)
        elif sub_cmd == "set":
            await self._var_set(sub_args)
        elif sub_cmd == "delete" or sub_cmd == "del":
            await self._var_delete(sub_args)
        else:
            self.formatter.print_error(f"Unknown variable subcommand: {sub_cmd}")
            await self._show_var_help()

    async def _var_list(self, args: List[str]):
        """List all variables with scope filtering"""
        agent_name = self.active_agent_name
        scope_filter = None

        if "--agent" in args:
            agent_idx = args.index("--agent")
            if agent_idx + 1 < len(args):
                agent_name = args[agent_idx + 1]

        if "--scope" in args:
            scope_idx = args.index("--scope")
            if scope_idx + 1 < len(args):
                scope_filter = args[scope_idx + 1]

        try:
            agent = await self.isaa_tools.get_agent(agent_name)

            if hasattr(agent, 'variable_manager') and agent.variable_manager:
                available_vars = agent.variable_manager.get_available_variables()

                self.formatter.print_section(f"Variables: {agent_name}", "")

                for scope_name, scope_vars in available_vars.items():
                    if scope_filter and scope_filter != scope_name:
                        continue

                    if scope_vars:
                        print(f"\n📁 {Style.YELLOW(scope_name)}")
                        for var_name, var_info in scope_vars.items():
                            print(f"   {var_name}: {var_info}")

                if not available_vars:
                    self.formatter.print_info("No variables found")
            else:
                self.formatter.print_error("Agent has no variable manager")

        except Exception as e:
            self.formatter.print_error(f"Failed to list variables: {e}")

    async def _var_get(self, args: List[str]):
        """Get variable value"""
        if not args:
            self.formatter.print_error("Usage: /var get <path.to.variable> [--agent agent_name]")
            return

        var_path = args[0]
        agent_name = self.active_agent_name

        if "--agent" in args:
            agent_idx = args.index("--agent")
            if agent_idx + 1 < len(args):
                agent_name = args[agent_idx + 1]

        try:
            agent = await self.isaa_tools.get_agent(agent_name)

            if hasattr(agent, 'variable_manager') and agent.variable_manager:
                value = agent.variable_manager.get(var_path)

                if value is not None:
                    self.formatter.print_success(f"✅ {var_path}:")
                    print(f"   {value}")
                else:
                    self.formatter.print_error(f"Variable '{var_path}' not found")
            else:
                self.formatter.print_error("Agent has no variable manager")

        except Exception as e:
            self.formatter.print_error(f"Failed to get variable: {e}")

    async def _var_set(self, args: List[str]):
        """Set variable value"""
        if len(args) < 2:
            self.formatter.print_error("Usage: /var set <path.to.variable> <value> [--agent agent_name]")
            return

        var_path = args[0]
        var_value = args[1]
        agent_name = self.active_agent_name

        if "--agent" in args:
            agent_idx = args.index("--agent")
            if agent_idx + 1 < len(args):
                agent_name = args[agent_idx + 1]

        # Try to parse value as JSON, fallback to string
        try:
            parsed_value = json.loads(var_value)
        except:
            parsed_value = var_value

        try:
            agent = await self.isaa_tools.get_agent(agent_name)

            if hasattr(agent, 'variable_manager') and agent.variable_manager:
                agent.variable_manager.set(var_path, parsed_value)
                self.formatter.print_success(f"✅ Set {var_path} = {parsed_value}")

                # If setting user preference, also update user proxy
                if var_path.startswith('user.preferences.'):
                    pref_key = var_path[17:]  # Remove 'user.preferences.' prefix
                    self.user_proxy_manager.set_preference(pref_key, parsed_value)

            else:
                self.formatter.print_error("Agent has no variable manager")

        except Exception as e:
            self.formatter.print_error(f"Failed to set variable: {e}")

    async def _var_delete(self, args: List[str]):
        """Delete variable"""
        if not args:
            self.formatter.print_error("Usage: /var delete <path.to.variable> [--agent agent_name]")
            return

        var_path = args[0]
        agent_name = self.active_agent_name

        if "--agent" in args:
            agent_idx = args.index("--agent")
            if agent_idx + 1 < len(args):
                agent_name = args[agent_idx + 1]

        # Confirm deletion
        confirmation = await self.prompt_session.prompt_async(f"Delete variable '{var_path}'? (y/N): ")

        if confirmation.lower() not in ['y', 'yes']:
            self.formatter.print_info("Variable deletion cancelled")
            return

        try:
            agent = await self.isaa_tools.get_agent(agent_name)

            if hasattr(agent, 'variable_manager') and agent.variable_manager:
                # This would require implementing a delete method in VariableManager
                self.formatter.print_info("Variable deletion not yet fully implemented")
            else:
                self.formatter.print_error("Agent has no variable manager")

        except Exception as e:
            self.formatter.print_error(f"Failed to delete variable: {e}")

    async def _handle_system_cmd(self, args: List[str]):
        """Handle /system commands"""
        if not args:
            await self._show_system_help()
            return

        sub_cmd = args[0].lower()
        sub_args = args[1:] if len(args) > 1 else []

        if sub_cmd == "status":
            await self._system_status(sub_args)
        elif sub_cmd == "verbosity":
            await self._system_verbosity(sub_args)
        elif sub_cmd == "notifications":
            await self._system_notifications(sub_args)
        elif sub_cmd == "checkpoint":
            await self._system_checkpoint(sub_args)
        else:
            self.formatter.print_error(f"Unknown system subcommand: {sub_cmd}")
            await self._show_system_help()

    async def _system_status(self, args: List[str]):
        """Show comprehensive system status"""
        self.formatter.print_section("System Status", "")

        # CLI info
        print(f"🖥️  CLI Version: Advanced ISAA CLI")
        print(f"🏠  Workspace: {self.workspace_path}")
        print(f"🤖  Active Agent: {self.active_agent_name}")

        # User proxy info
        bound_agents = self.user_proxy_manager.get_bound_agents()
        print(f"🔗  Bound Agents: {len(bound_agents)}")
        if bound_agents:
            print(f"     {', '.join(bound_agents)}")

        # Task info
        all_tasks = self.task_manager.list_tasks()
        running_tasks = [t for t in all_tasks if t.status == TaskStatus.RUNNING]
        completed_tasks = [t for t in all_tasks if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in all_tasks if t.status == TaskStatus.FAILED]

        print(f"⚡  Active Tasks: {len(running_tasks)}")
        print(f"✅  Completed Tasks: {len(completed_tasks)}")
        print(f"❌  Failed Tasks: {len(failed_tasks)}")

        # Flow info
        flows = self.flow_parser.list_flows()
        print(f"🔄  Saved Flows: {len(flows)}")

        # System resources
        try:
            memory_usage = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            print(f"💾  Memory Usage: {memory_usage.percent}%")
            print(f"🔧  CPU Usage: {cpu_percent}%")
        except:
            pass

        # Notification status
        print(f"🔔  Notifications: {'Enabled' if self.notification_system else 'Disabled'}")

    async def _system_verbosity(self, args: List[str]):
        """Set system verbosity level"""
        if not args:
            self.formatter.print_info(f"Current verbosity: {self._current_verbosity_mode.name}")
            return

        try:
            new_mode = VerbosityMode[args[0].upper()]
            self.printer.mode = new_mode
            self._current_verbosity_mode = new_mode
            self.formatter.print_success(f"✅ Verbosity set to {new_mode.name}")
        except KeyError:
            self.formatter.print_error(f"Invalid verbosity mode '{args[0]}'")
            print("Available modes: MINIMAL, STANDARD, VERBOSE, DEBUG, REALTIME")

    async def _system_notifications(self, args: List[str]):
        """Toggle notifications on/off"""
        if not args:
            status = "enabled" if self.notification_system else "disabled"
            self.formatter.print_info(f"Notifications are {status}")
            return

        setting = args[0].lower()
        if setting in ['on', 'enable', 'enabled']:
            if not self.notification_system:
                self.notification_system = NotificationSystem()
            self.formatter.print_success("✅ Notifications enabled")
        elif setting in ['off', 'disable', 'disabled']:
            self.notification_system = None
            self.formatter.print_success("✅ Notifications disabled")
        else:
            self.formatter.print_error("Usage: /system notifications <on|off>")

    async def _system_checkpoint(self, args: List[str]):
        """System checkpoint management"""
        if not args:
            self.formatter.print_error("Usage: /system checkpoint <create|load|list|clean>")
            return

        sub_cmd = args[0].lower()

        if sub_cmd == "create":
            await self._save_state()
            self.formatter.print_success("✅ System checkpoint created")
        elif sub_cmd == "load":
            await self._load_state()
            self.formatter.print_success("✅ System checkpoint loaded")
        else:
            self.formatter.print_error(f"Unknown checkpoint subcommand: {sub_cmd}")
            self.formatter.print_info("Available: create, load")

    # ===== HELP COMMANDS =====

    async def _show_agent_help(self):
        """Show agent command help"""
        self.formatter.print_section("Agent Commands", "")
        commands = [
            ("/agent list [--detailed] [--bound]", "List all agents with optional details"),
            ("/agent create [name]", "Create a new agent with templates"),
            ("/agent build [name]", "Interactive agent builder (NEW)"),
            ("/agent switch <name>", "Switch to a different agent"),
            ("/agent inspect [name] [-d]", "Inspect agent with context distribution (NEW)"),
            ("/agent config [name]", "Configure agent settings"),
            ("/agent bind <name> [--scopes]", "Bind agent to user-proxy"),
            ("/agent unbind <name>", "Unbind agent from user-proxy"),
            ("/agent clone <source> <new_name>", "Clone an existing agent"),
            ("/agent checkpoint [name]", "Create agent checkpoint"),
            ("/agent clear-context [name]", "Clear agent context (NEW)"),
        ]

        for cmd, desc in commands:
            print(f"  {Style.CYAN(cmd.ljust(40))} {desc}")

    async def _show_task_help(self):
        """Show task command help"""
        self.formatter.print_section("Task Commands", "")
        commands = [
            ("/task run \"prompt\" [options]", "Run a task with options"),
            ("  --bg, --background", "Run task in background"),
            ("  --agent <name>", "Use specific agent"),
            ("  --watch", "Show live progress"),
            ("  --fast", "Enable fast mode (NEW)"),
            ("  --callback", "Enable interactive callback (NEW)"),
            ("/task list [--status] [--agent]", "List tasks with filters"),
            ("/task attach <id>", "Attach to background task"),
            ("/task pause <id>", "Pause running task"),
            ("/task cancel <id>", "Cancel running task"),
            ("/task inspect <id>", "Inspect task details"),
        ]

        for cmd, desc in commands:
            print(f"  {Style.CYAN(cmd.ljust(40))} {desc}")

    async def _show_flow_help(self):
        """Show flow command help"""
        self.formatter.print_section("Flow Commands", "")
        commands = [
            ("/flow design [name]", "Interactive flow designer"),
            ("/flow run <name> [--input]", "Run saved workflow"),
            ("/flow list", "List all saved workflows"),
            ("/flow inspect <name>", "Inspect flow structure"),
            ("/flow save <name> <definition>", "Save flow from command line"),
        ]

        for cmd, desc in commands:
            print(f"  {Style.CYAN(cmd.ljust(30))} {desc}")

        print(f"\n{Style.YELLOW('Flow Syntax:')}")
        syntax = [
            (">>", "Sequential execution"),
            ("+, &", "Parallel execution"),
            ("|", "Error handling/fallback"),
            ("%", "Conditional else branch"),
            ("CF(Model)", "Format with Pydantic model"),
            ("IS(key, value)", "Conditional check"),
            ("Function(func)", "Custom function"),
        ]

        for op, desc in syntax:
            print(f"  {Style.CYAN(op.ljust(15))} {desc}")

    async def _show_session_help(self):
        """Show session command help"""
        self.formatter.print_section("Session Commands", "")
        commands = [
            ("/session list [agent]", "List all sessions for agent"),
            ("/session new <name> [agent]", "Create new named session (NEW)"),
            ("/session switch <session_id>", "Switch to different session"),
            ("/session inspect <session_id>", "Inspect session details"),
            ("/session clear <session_id>", "Clear session history"),
            ("/session export <id> <file>", "Export session to file"),
            ("/session import <file>", "Import session from file"),
        ]

        for cmd, desc in commands:
            print(f"  {Style.CYAN(cmd.ljust(40))} {desc}")

    async def _show_var_help(self):
        """Show variable command help"""
        self.formatter.print_section("Variable Commands", "")
        commands = [
            ("/var list [--scope] [--agent]", "List variables with filtering"),
            ("/var get <path> [--agent]", "Get variable value"),
            ("/var set <path> <value> [--agent]", "Set variable value"),
            ("/var delete <path> [--agent]", "Delete variable"),
        ]

        for cmd, desc in commands:
            print(f"  {Style.CYAN(cmd.ljust(30))} {desc}")

    async def _show_system_help(self):
        """Show system command help"""
        self.formatter.print_section("System Commands", "")
        commands = [
            ("/system status", "Show comprehensive system status"),
            ("/system verbosity <level>", "Set output verbosity level"),
            ("/system notifications <on|off>", "Toggle notifications"),
            ("/system checkpoint <action>", "Manage system checkpoints"),
        ]

        for cmd, desc in commands:
            print(f"  {Style.CYAN(cmd.ljust(30))} {desc}")

    async def _handle_help_cmd(self, args: List[str]):
        """Show comprehensive help"""
        if args:
            # Show specific help
            topic = args[0].lower()
            if topic == "agent":
                await self._show_agent_help()
            elif topic == "task":
                await self._show_task_help()
            elif topic == "flow":
                await self._show_flow_help()
            elif topic == "session":  # NEW: Changed from context
                await self._show_session_help()
            elif topic == "var":
                await self._show_var_help()
            elif topic == "system":
                await self._show_system_help()
            else:
                self.formatter.print_error(f"Unknown help topic: {topic}")
        else:
            # Show general help
            self.formatter.log_header("ISAA CLI - Help")

            categories = [
                ("Agent Management", "/agent", "Create, manage, and bind agents"),
                ("Task Execution", "/task", "Run and manage tasks in background"),
                ("Workflow System", "/flow", "Design and execute complex workflows"),
                ("Session Management", "/session", "Manage agent sessions and history"),  # NEW: Changed from Context Control
                ("Variable System", "/var", "Set and get agent variables"),
                ("System Control", "/system", "CLI settings and status"),
            ]

            for category, command, description in categories:
                print(f"\n{Style.Bold(Style.YELLOW(category))}")
                print(f"  {Style.CYAN(command)} - {description}")
                print(f"  Use '{command} help' or '/help {command[1:]}' for details")

            print(f"\n{Style.Bold('Other Commands:')}")
            print(f"  {Style.CYAN('/help [topic]')} - Show this help or topic help")
            print(f"  {Style.CYAN('/clear')} - Clear screen")
            print(f"  {Style.CYAN('/quit, /exit')} - Exit CLI")
            print(f"  {Style.CYAN('!<command>')} - Execute shell command")

            print(f"\n{Style.Bold('Quick Tips:')}")
            print(f"  • Use Tab for command completion")
            print(f"  • Start with '/agent create' to create your first agent")
            print(f"  • Bind agents with '/agent bind <name>' for shared preferences")
            print(f"  • Run tasks in background with '/task run \"prompt\" --bg'")
            print(f"  • Create workflows with '/flow design'")

    async def _handle_exit_cmd(self, args: List[str]):
        """Handle exit command"""
        self._shutdown_requested = True

    async def _handle_clear_cmd(self, args: List[str]):
        """Handle clear screen command"""
        os.system('cls' if os.name == 'nt' else 'clear')
        await self.show_welcome()

    async def _handle_shell_command(self, command: str):
        """Execute shell command with enhanced output"""
        if not command.strip():
            self.formatter.print_error("Shell command cannot be empty")
            return

        self.formatter.print_info(f"🚀 Executing: `{command}`")

        try:
            shell_exe, cmd_flag = detect_shell()
            process = await asyncio.create_subprocess_shell(
                f'"{shell_exe}" {cmd_flag} "{command}"',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            async def stream_reader(stream, style_func):
                while not stream.at_eof():
                    line = await stream.readline()
                    if line:
                        style_func(line.decode(errors='ignore').strip())

            await asyncio.gather(
                stream_reader(process.stdout, print),
                stream_reader(process.stderr, lambda x: self.formatter.print_error(x))
            )

            return_code = await process.wait()

            if return_code == 0:
                self.formatter.print_success(f"✅ Command completed (Exit Code: {return_code})")
            else:
                self.formatter.print_warning(f"⚠️ Command finished with error (Exit Code: {return_code})")

        except Exception as e:
            self.formatter.print_error(f"❌ Command failed: {e}")

    # ===== STATE MANAGEMENT =====

    async def _save_state(self):
        """Save CLI state to disk"""
        try:
            # Save task manager state
            task_state_file = self.state_dir / "tasks.pkl"
            self.task_manager.save_state(task_state_file)

            # Save user preferences
            preferences_file = self.state_dir / "preferences.json"
            with open(preferences_file, 'w') as f:
                json.dump({
                    'user_preferences': self.user_proxy_manager.user_preferences,
                    'bound_agents': self.user_proxy_manager.bound_agents,
                    'active_agent': self.active_agent_name,
                    'session_id': self.session_id,
                    'workspace_path': str(self.workspace_path)
                }, f, indent=2)

        except Exception as e:
            print(f"⚠️ Failed to save state: {e}")

    def _load_state(self):
        """Load CLI state from disk"""
        try:
            # Load task manager state
            task_state_file = self.state_dir / "tasks.pkl"
            if task_state_file.exists():
                self.task_manager.load_state(task_state_file)

            # Load user preferences
            preferences_file = self.state_dir / "preferences.json"
            if preferences_file.exists():
                with open(preferences_file, 'r') as f:
                    data = json.load(f)

                self.user_proxy_manager.user_preferences = data.get('user_preferences', {})
                self.user_proxy_manager.bound_agents = data.get('bound_agents', [])
                self.active_agent_name = data.get('active_agent', 'user-proxy')

        except Exception as e:
            print(f"⚠️ Failed to load state: {e}")

    async def _confirm_exit(self) -> bool:
        """Confirm exit with running tasks check"""
        running_tasks = [t for t in self.task_manager.tasks.values() if t.status == TaskStatus.RUNNING]

        if running_tasks:
            self.formatter.print_warning(f"⚠️ {len(running_tasks)} tasks are still running!")
            for task in running_tasks[:3]:  # Show first 3
                print(f"   {task.task_id}: {task.prompt[:50]}...")

            if len(running_tasks) > 3:
                print(f"   ... and {len(running_tasks) - 3} more")

            choice = await self.prompt_session.prompt_async(
                "Exit anyway? Running tasks will be cancelled. (y/N): "
            )
            return choice.lower() in ['y', 'yes']

        return True

    async def _cleanup(self):
        """Cleanup resources and save state"""
        self.formatter.print_info("🔄 Shutting down CLI...")

        # Cancel running tasks
        running_tasks = [t for t in self.task_manager.tasks.values() if t.status == TaskStatus.RUNNING]
        if running_tasks:
            self.formatter.print_info(f"⏹️ Cancelling {len(running_tasks)} running tasks...")
            for task in running_tasks:
                await self.task_manager.cancel_task(task.task_id)

        # Save state
        await self._save_state()

        # Send shutdown notification
        if self.notification_system:
            self.notification_system.show_notification(
                title="ISAA CLI Shutdown",
                message="CLI session ended. State has been saved.",
                notification_type=NotificationType.INFO,
                timeout=1500
            )

        # Cleanup app
        try:
            await self.app.a_exit()
        except:
            pass

        self.formatter.print_success("✅ ISAA CLI shutdown complete. Goodbye!")


# ===== MAIN ENTRY POINT =====

async def run(app, *args):
    """Main entry point for the advanced ISAA CLI"""
    try:
        cli_app_instance = get_app("isaa_cli_instance")
        cli = AdvancedIsaaCli(cli_app_instance)
        await cli.run()

    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()

        # Try to send error notification
        try:
            notifier = NotificationSystem()
            notifier.show_notification(
                title="CLI Fatal Error",
                message="ISAA CLI encountered a fatal error",
                notification_type=NotificationType.ERROR,
                details=NotificationDetails(
                    title="Error Details",
                    content=str(e),
                    data={"traceback": traceback.format_exc()}
                )
            )
        except:
            pass

        return 1


if __name__ == "__main__":
    asyncio.run(run(None))
