import asyncio
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Optional, Dict, List
import glob

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter, WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import get_app as get_pt_app

from toolboxv2 import get_app
from toolboxv2.mods.isaa.CodingAgent.live import EnhancedVerboseOutput
from toolboxv2.mods.isaa.module import Tools as Isaatools
from toolboxv2.utils.extras.Style import Style, Spinner

NAME = "isaa_cli"


class WorkspaceIsaasCli:
    """Advanced ISAA CLI with comprehensive agent tools and enhanced formatting"""

    def __init__(self, app_instance: Any):
        self.app = app_instance
        self.isaa_tools: Isaatools = app_instance.get_mod("isaa")
        self.formatter = EnhancedVerboseOutput(verbose=True, print_func=app_instance.print)
        self.style = Style()
        self.active_agent_name = "workspace_supervisor"
        self.session_id = "workspace_session"
        self.history = FileHistory(Path(self.app.data_dir) / "isaa_cli_history.txt")
        self.completer = self.build_workspace_completer()
        self.key_bindings = self.create_key_bindings()
        self.prompt_session = PromptSession(
            history=self.history,
            completer=self.completer,
            complete_while_typing=True,
            key_bindings=self.key_bindings,
            multiline=True,
            wrap_lines=True,
        )
        self.background_tasks = {}
        self.interrupt_count = 0
        self.workspace_path = Path.cwd()

    def create_key_bindings(self):
        """Create custom key bindings for enhanced UX"""
        kb = KeyBindings()

        @kb.add('c-c')
        def _(event):
            """Handle Ctrl+C gracefully"""
            event.app.exit(exception=KeyboardInterrupt)

        @kb.add('c-d')
        def _(event):
            """Handle Ctrl+D for exit"""
            if not event.current_buffer.text:
                event.app.exit(exception=EOFError)

        return kb

    def build_workspace_completer(self):
        """Build workspace management focused autocompletion using WordCompleter."""
        commands_dict = {
            "/workspace": {
                "status": None,
                "cd": WordCompleter([]),  # Placeholder for directory completion later
                "ls": WordCompleter([]),  # Placeholder for directory completion later
                "info": None,
            },
            "/agent": {
                "list": None,
                "switch": None,
                "status": None,
            },
            "/tasks": {
                "list": None,
                "attach": None,
                "kill": None,
                "status": None,
                "monitor": None,
            },
            "/context": {
                "save": None,
                "load": None,
                "list": None,
                "clear": None,
                "delete": None,
            },
            "/monitor": {
                "start": None,
                "stop": None,
                "status": None,
                "logs": None,
            },
            "/system": {
                "status": None,
                "config": None,
                "backup": None,
                "restore": None,
                "performance": None,
            },
            "/help": None,
            "/exit": None,
            "/quit": None,
            "/clear": None,
        }

        # Helper function to flatten the nested dict into a list of full commands
        def flatten_commands(d, prefix=''):
            flat_list = []
            for key, value in d.items():
                current_command = f"{prefix}{key}"
                if isinstance(value, dict):
                    # Add the parent command itself (e.g., '/workspace')
                    # and then its children.
                    flat_list.append(current_command)
                    flat_list.extend(flatten_commands(value, f"{current_command} "))
                else:
                    flat_list.append(current_command)
            return flat_list

        all_possible_commands = flatten_commands(commands_dict)

        # Use WordCompleter with sentence=True to match against the whole line
        return WordCompleter(all_possible_commands, sentence=True)

    def get_prompt_text(self) -> HTML:
        """Generate workspace-focused prompt with status indicators"""
        try:
            # Git info
            git_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                cwd=self.workspace_path
            )
            git_info = git_result.stdout.strip() if git_result.returncode == 0 else None

            if git_info:
                status_result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    cwd=self.workspace_path
                )
                if status_result.stdout.strip():
                    git_info += "*"
        except:
            git_info = None

        workspace_name = self.workspace_path.name
        bg_count = len([t for t in self.background_tasks.values() if not t['task'].done()])

        # Build prompt components
        components = []

        # Workspace component
        components.append(f'<ansicyan>[</ansicyan><ansigreen>{workspace_name}</ansigreen>')

        # Git component
        if git_info:
            if '*' in git_info:
                components.append(f'<ansicyan> on </ansicyan><ansired>{git_info}</ansired>')
            else:
                components.append(f'<ansicyan> on </ansicyan><ansimagenta>{git_info}</ansimagenta>')

        components.append('<ansicyan>]</ansicyan>')

        # Agent and background tasks
        components.append(f' <ansiyellow>({self.active_agent_name})</ansiyellow>')
        if bg_count > 0:
            components.append(f' <ansiblue>[{bg_count} bg]</ansiblue>')

        # Prompt symbol
        components.append('\n<ansiblue>❯</ansiblue> ')

        return HTML(''.join(components))

    # ##################################################################
    # TOOL DEFINITIONS MOVED TO CLASS METHODS
    # ##################################################################

    async def replace_in_files_tool(self, old_str: str, new_str: str, file_patterns: str = "*", directory: str = ".",
                                    recursive: Optional[bool] = True, file_types: Optional[str] = None, dry_run: Optional[bool] = False):
        """Replace any string in any files matching patterns - the most powerful replace tool.

        Args:
            old_str: String to replace
            new_str: Replacement string
            file_patterns: File patterns (e.g., "*.py,*.txt,*.json" or "*")
            directory: Directory to search (default: current)
            recursive: Search subdirectories
            file_types: Additional file type filter (e.g., "text,code,config")
            dry_run: Preview changes without making them
        """
        results = {
            "files_modified": [],
            "total_replacements": 0,
            "errors": [],
            "skipped_files": [],
            "preview": dry_run
        }

        base_path = Path(directory).resolve()
        if not base_path.exists():
            results["errors"].append(f"Directory {directory} does not exist")
            return json.dumps(results, indent=2)

        patterns = [p.strip() for p in file_patterns.split(",")]
        if not patterns:
            patterns = ["*"]

        files_to_process = []
        for pattern in patterns:
            if recursive:
                files_to_process.extend(base_path.rglob(pattern))
            else:
                files_to_process.extend(base_path.glob(pattern))

        if file_types:
            type_filters = [t.strip().lower() for t in file_types.split(",")]
            filtered_files = []
            for file_path in files_to_process:
                if file_path.is_file():
                    suffix = file_path.suffix.lower()
                    if any(ftype in suffix or suffix.endswith(ftype) for ftype in type_filters):
                        filtered_files.append(file_path)
            files_to_process = filtered_files

        files_to_process = sorted(set(f for f in files_to_process if f.is_file()))

        for file_path in files_to_process:
            try:
                content = None
                used_encoding = None
                for encoding in ['utf-8', 'utf-16', 'latin-1', 'ascii']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                        used_encoding = encoding
                        break
                    except UnicodeDecodeError:
                        continue

                if content is None:
                    results["skipped_files"].append(f"{file_path} (binary/unreadable)")
                    continue

                if old_str in content:
                    replacements_count = content.count(old_str)
                    if not dry_run:
                        new_content = content.replace(old_str, new_str)
                        with open(file_path, 'w', encoding=used_encoding) as f:
                            f.write(new_content)
                    results["files_modified"].append({
                        "file": str(file_path.relative_to(base_path)),
                        "replacements": replacements_count,
                        "encoding": used_encoding
                    })
                    results["total_replacements"] += replacements_count
            except Exception as e:
                results["errors"].append(f"Error processing {file_path}: {str(e)}")

        return json.dumps(results, indent=2)

    async def read_file_tool(self, file_path: str, encoding: str = "utf-8", lines_range: Optional[str] = None):
        """Read file content with optional line range (e.g., '1-10' or '5-')"""
        try:
            path = Path(file_path)
            if not path.exists():
                return f"❌ File {file_path} does not exist"
            with open(path, 'r', encoding=encoding) as f:
                if lines_range:
                    lines = f.readlines()
                    if '-' in lines_range:
                        start, end = lines_range.split('-', 1)
                        start = int(start) - 1 if start else 0
                        end = int(end) if end else len(lines)
                        content = ''.join(lines[start:end])
                    else:
                        line_num = int(lines_range) - 1
                        content = lines[line_num] if line_num < len(lines) else ""
                else:
                    content = f.read()
            return f"📄 Content of {file_path}:\n\n{content}"
        except Exception as e:
            return f"❌ Error reading {file_path}: {str(e)}"

    async def write_file_tool(self, file_path: str, content: str, encoding: str = "utf-8", append: bool = False,
                              backup: bool = False):
        """Write content to file with optional backup"""
        try:
            path = Path(file_path)
            if backup and path.exists():
                backup_path = path.with_suffix(path.suffix + '.backup')
                path.rename(backup_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = 'a' if append else 'w'
            with open(path, mode, encoding=encoding) as f:
                f.write(content)
            action = "✅ Appended to" if append else "✅ Written to"
            backup_msg = " (backup created)" if backup else ""
            return f"{action} {file_path}{backup_msg}"
        except Exception as e:
            return f"❌ Error writing to {file_path}: {str(e)}"

    async def search_in_files_tool(self, search_term: str, file_patterns: str = "*", directory: str = ".",
                                   recursive: bool = True, context_lines: int = 0):
        """Search for text in files with context"""
        results = []
        base_path = Path(directory).resolve()
        patterns = [p.strip() for p in file_patterns.split(",")]
        files_to_search = []
        for pattern in patterns:
            if recursive:
                files_to_search.extend(base_path.rglob(pattern))
            else:
                files_to_search.extend(base_path.glob(pattern))
        for file_path in files_to_search:
            if not file_path.is_file():
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                for line_num, line in enumerate(lines, 1):
                    if search_term in line:
                        match_info = {
                            "file": str(file_path.relative_to(base_path)),
                            "line": line_num,
                            "content": line.strip()
                        }
                        if context_lines > 0:
                            start = max(0, line_num - context_lines - 1)
                            end = min(len(lines), line_num + context_lines)
                            match_info["context"] = [
                                f"{i + start + 1}: {lines[i + start].rstrip()}"
                                for i in range(end - start)
                            ]
                        results.append(match_info)
            except:
                continue
        return json.dumps(results, indent=2)

    async def list_directory_tool(self, directory: str = ".", recursive: bool = False, file_types: Optional[str] = None,
                                  show_hidden: bool = False):
        """List directory contents with advanced filtering"""
        try:
            path = Path(directory)
            if not path.exists():
                return f"❌ Directory {directory} does not exist"
            files, dirs = [], []
            if recursive:
                for item in path.rglob("*"):
                    if not show_hidden and item.name.startswith('.'): continue
                    if item.is_file(): files.append(item)
                    elif item.is_dir(): dirs.append(item)
            else:
                for item in path.iterdir():
                    if not show_hidden and item.name.startswith('.'): continue
                    if item.is_file(): files.append(item)
                    elif item.is_dir(): dirs.append(item)
            if file_types:
                type_filters = [t.strip() for t in file_types.split(",")]
                files = [f for f in files if any(t in f.suffix.lower() for t in type_filters)]
            result = f"📁 Contents of {directory}:\n\n"
            result += f"Directories ({len(dirs)}):\n"
            for d in sorted(dirs):
                result += f"  📁 {d.relative_to(path) if recursive else d.name}\n"
            result += f"\nFiles ({len(files)}):\n"
            for f in sorted(files):
                size = f.stat().st_size
                size_str = f"{size:,} bytes" if size < 1024 else f"{size / 1024:.1f} KB"
                result += f"  📄 {f.relative_to(path) if recursive else f.name} ({size_str})\n"
            return result
        except Exception as e:
            return f"❌ Error listing directory: {str(e)}"

    async def create_specialized_agent_tool(self, agent_name: str, specialization: str, system_prompt: Optional[str] = None,
                                            model: Optional[str] = None, tools: Optional[str] = None):
        """Create a specialized agent with predefined or custom capabilities"""
        try:
            new_builder = self.isaa_tools.get_agent_builder(agent_name)
            if system_prompt:
                new_builder.with_system_message(system_prompt)
            else:
                specialized_prompts = {
                    "coder": "You are a specialized coding agent...",
                    "writer": "You are a specialized writing agent...",
                    "analyzer": "You are a specialized analysis agent...",
                    # ... (rest of prompts unchanged)
                }
                prompt = specialized_prompts.get(specialization.lower(),
                                                 f"You are a specialized {specialization} agent.")
                new_builder.with_system_message(prompt)
            if model:
                new_builder.with_model(model)
            if tools and hasattr(new_builder, 'with_tools'):
                pass
            await self.isaa_tools.register_agent(new_builder)
            return f"✅ Specialized agent '{agent_name}' created for {specialization}"
        except Exception as e:
            return f"❌ Error creating agent: {str(e)}"

    async def remove_agent_tool(self, agent_name: str, confirm: bool = False):
        """Remove an agent from the system with confirmation"""
        try:
            if not confirm:
                return f"⚠️  Use confirm=True to actually remove agent '{agent_name}'"
            agents_list = self.isaa_tools.config.get("agents-name-list", [])
            if agent_name in agents_list:
                agents_list.remove(agent_name)
                return f"✅ Agent '{agent_name}' removed successfully"
            else:
                return f"❌ Agent '{agent_name}' not found"
        except Exception as e:
            return f"❌ Error removing agent: {str(e)}"

    async def list_agents_tool(self, detailed: bool = False):
        """List all available agents with optional details"""
        agents = self.isaa_tools.config.get("agents-name-list", [])
        if not agents: return "📝 No agents available"
        if not detailed:
            agent_list = "🤖 Available Agents:\n"
            for name in agents:
                marker = "🟢" if name == self.active_agent_name else "⚪"
                agent_list += f"{marker} {name}\n"
            return agent_list
        agent_info = "🤖 Detailed Agent Information:\n\n"
        for name in agents:
            try:
                agent = await self.isaa_tools.get_agent(name)
                status = "✅ Active" if agent else "❌ Inactive"
                marker = "🟢" if name == self.active_agent_name else "⚪"
                agent_info += f"{marker} {name}: {status}\n"
                if agent and hasattr(agent, 'system_message'):
                    msg = agent.system_message[:100] + "..." if len(agent.system_message) > 100 else agent.system_message
                    agent_info += f"   System: {msg}\n"
                agent_info += "\n"
            except Exception as e:
                agent_info += f"⚪ {name}: Error - {str(e)}\n\n"
        return agent_info

    async def run_agent_background_tool(self, agent_name: str, task_prompt: str, session_id: Optional[str] = None,
                                        priority: Optional[str] = "normal"):
        """Run a task with specified agent in background with priority"""
        if priority is None:
            priority = "normal"
        try:
            if not session_id:
                session_id = f"bg_{len(self.background_tasks)}_{agent_name}"
            task = asyncio.create_task(
                self.isaa_tools.run_agent(agent_name, task_prompt, session_id=session_id, progress_callback=self.progress_callback))
            task_id = len(self.background_tasks)
            self.background_tasks[task_id] = {'task': task, 'agent': agent_name, 'prompt': task_prompt[:100],
                                              'started': asyncio.get_event_loop().time(), 'session_id': session_id,
                                              'priority': priority, 'status': 'running'}
            return f"⧖ Background task {task_id} started with agent '{agent_name}' (priority: {priority})"
        except Exception as e:
            return f"❌ Error starting background task: {str(e)}"

    async def get_background_tasks_status_tool(self, show_completed: bool = True):
        """Get detailed status of all background tasks"""
        if not self.background_tasks: return "📝 No background tasks found"
        status_info, running, completed = "🔄 Background Tasks Status:\n\n", 0, 0
        for tid, tinfo in self.background_tasks.items():
            is_done = tinfo['task'].done()
            if is_done:
                completed += 1
                if not show_completed: continue
                status = "✅ Completed" if not tinfo['task'].cancelled() else "❌ Cancelled"
            else:
                running += 1
                status = "⧖ Running"
            elapsed = asyncio.get_event_loop().time() - tinfo['started']
            status_info += f"Task {tid}: {status}\n"
            status_info += f"  Agent: {tinfo['agent']}\n"
            status_info += f"  Priority: {tinfo.get('priority', 'normal')}\n"
            status_info += f"  Elapsed: {elapsed:.1f}s\n"
            status_info += f"  Prompt: {tinfo['prompt']}\n\n"
        return f"Summary: {running} running, {completed} completed\n\n" + status_info

    async def kill_background_task_tool(self, task_id: int, force: bool = False):
        """Kill a specific background task"""
        try:
            task_id = int(task_id)
            if task_id not in self.background_tasks: return f"❌ Task {task_id} not found"
            tinfo = self.background_tasks[task_id]
            if tinfo['task'].done(): return f"ℹ️  Task {task_id} already completed"
            tinfo['task'].cancel()
            tinfo['status'] = 'cancelled' if force else 'cancelling'
            return f"✅ Task {task_id} {'force ' if force else ''}cancelled"
        except ValueError:
            return "❌ Invalid task ID - must be a number"
        except Exception as e:
            return f"❌ Error killing task: {str(e)}"

    async def change_workspace_tool(self, directory: str, create_if_missing: bool = False):
        """Change workspace directory with optional creation"""
        try:
            new_path = Path(directory).resolve()
            if not new_path.exists():
                if create_if_missing:
                    new_path.mkdir(parents=True, exist_ok=True)
                else:
                    return f"❌ Directory {directory} does not exist (use create_if_missing=True to create)"
            if not new_path.is_dir(): return f"❌ {directory} is not a directory"
            old_path, self.workspace_path = self.workspace_path, new_path
            os.chdir(new_path)
            return f"✅ Workspace changed from {old_path} to {new_path}"
        except Exception as e:
            return f"❌ Error changing workspace: {str(e)}"

    async def workspace_status_tool(self, include_git: bool = True, include_files: bool = True):
        """Get comprehensive workspace status"""
        try:
            parts = [f"📁 Workspace: {self.workspace_path}", f"🤖 Active Agent: {self.active_agent_name}"]
            bg_running = len([t for t in self.background_tasks.values() if not t['task'].done()])
            parts.append(f"🔄 Background Tasks: {bg_running} running / {len(self.background_tasks)} total")
            if include_git:
                try:
                    git_res = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, cwd=self.workspace_path)
                    if git_res.returncode == 0:
                        status_res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=self.workspace_path)
                        changes = "clean" if not status_res.stdout.strip() else "modified"
                        parts.append(f"🔀 Git: {git_res.stdout.strip()} ({changes})")
                    else:
                        parts.append("🔀 Git: Not a git repository")
                except:
                    parts.append("🔀 Git: Not available")
            if include_files:
                try:
                    files = len([f for f in self.workspace_path.rglob("*") if f.is_file()])
                    dirs = len([d for d in self.workspace_path.rglob("*") if d.is_dir()])
                    parts.append(f"📊 Files: {files} files, {dirs} directories")
                except:
                    parts.append("📊 Files: Count unavailable")
            return "\n".join(parts)
        except Exception as e:
            return f"❌ Error getting workspace status: {str(e)}"

    async def show_welcome(self):
        """Display enhanced welcome with status overview"""
        welcome_text = "ISAA CLI Assistant"
        subtitle = "Intelligent System Automation & Analysis"

        # Calculate padding for centering
        terminal_width = os.get_terminal_size().columns
        welcome_len = len(welcome_text)
        subtitle_len = len(subtitle)

        welcome_padding = (terminal_width - welcome_len) // 2
        subtitle_padding = (terminal_width - subtitle_len) // 2

        print()
        print(self.style.CYAN("═" * terminal_width))
        print()
        print(" " * welcome_padding + self.style.Bold(self.style.BLUE(welcome_text)))
        print(" " * subtitle_padding + self.style.GREY(subtitle))
        print()
        print(self.style.CYAN("═" * terminal_width))
        print()

        # System status
        bg_count = len([t for t in self.background_tasks.values() if not t['task'].done()])
        agents_count = len(self.isaa_tools.config.get("agents-name-list", []))

        # Status overview
        self.formatter.print_section(
            "Workspace Overview",
            f"📁 Path: {self.workspace_path}\n"
            f"🤖 Active Agent: {self.active_agent_name}\n"
            f"⚙️  Total Agents: {agents_count}\n"
            f"🔄 Background Tasks: {bg_count}"
        )

        # Quick start tips
        tips = [
            f"{self.style.YELLOW('●')} Type naturally - the agent will use tools to help you",
            f"{self.style.YELLOW('●')} Type {self.style.CYAN('/help')} for available commands",
            f"{self.style.YELLOW('●')} Use {self.style.CYAN('Tab')} for autocompletion",
            f"{self.style.YELLOW('●')} {self.style.CYAN('Ctrl+C')} to interrupt, {self.style.CYAN('Ctrl+D')} to exit",
        ]

        for tip in tips:
            print(f"  {tip}")
        print()

        self.formatter.print_info("online")

    async def init(self):
        """Initialize workspace CLI with progress tracking"""

        # Initialization steps with progress
        steps = [
            ("Initializing ISAA framework", self.isaa_tools.init_isaa),
            ("Setting up workspace agent", self.setup_workspace_agent),
            ("Loading configurations", self.load_configurations),
            ("Preparing workspace tools", self.prepare_tools),
        ]

        for i, (step_name, step_func) in enumerate(steps):
            self.formatter.print_progress_bar(i, len(steps), f"Setup: {step_name}")
            if asyncio.iscoroutinefunction(step_func):
                await step_func()
            else:
                step_func()
            await asyncio.sleep(0.2)  # Brief pause for visual feedback

        self.formatter.print_progress_bar(len(steps), len(steps), "Setup: Complete")
        print()  # New line after progress

        await self.show_welcome()


    async def setup_workspace_agent(self):
        """Setup the main workspace supervisor agent"""
        if self.active_agent_name != "workspace_agent":
            self.active_agent_name = "workspace_agent"
        builder = self.isaa_tools.get_agent_builder(self.active_agent_name)
        builder.with_system_message(
            "You are a Workspace Supervisor Agent with comprehensive capabilities.\n\n"
            "YOUR ROLE:\n"
            "• Manage files and directories with precision\n"
            "• Create and coordinate specialized agents\n"
            "• Execute complex multi-step workflows\n"
            "• Handle background task orchestration\n\n"
            "YOUR TOOLS:\n"
            "• File operations: read, write, search, replace strings in any files\n"
            "• Agent management: create, configure, remove specialized agents\n"
            "• Task execution: background processing and coordination\n"
            "• Workspace management: directory operations, status monitoring\n\n"
            "GUIDELINES:\n"
            "• Always report actions clearly and completely\n"
            "• Use appropriate tools for each task\n"
            "• Maintain workspace organization\n"
            "• Provide detailed feedback on operations\n"
            "• When replacing strings, be thorough and report all changes"
        )
        builder = await self.add_comprehensive_tools_to_agent(builder)
        print("Registering workspace agent...")
        await self.isaa_tools.register_agent(builder)

    def load_configurations(self):
        """Load and validate workspace configurations"""
        try:
            workspace_config_file = self.workspace_path / ".isaa_workspace.json"
            if workspace_config_file.exists():
                with open(workspace_config_file, 'r') as f:
                    workspace_config = json.load(f)
                if 'default_agent' in workspace_config: self.active_agent_name = workspace_config['default_agent']
                if 'session_id' in workspace_config: self.session_id = workspace_config['session_id']

            agents_list = self.isaa_tools.config.get("agents-name-list", [])
            if self.active_agent_name not in agents_list and agents_list:
                self.active_agent_name = agents_list[0]
        except Exception:
            pass

    def prepare_tools(self):
        """Prepare additional tools and utilities"""
        pass

    async def add_comprehensive_tools_to_agent(self, builder):
        """Add comprehensive workspace and agent management tools"""

        # Note: All tool functions are now methods of this class.
        # We are referencing them with `self.tool_name`.

        builder.with_adk_tool_function(
            self.replace_in_files_tool,
            name="replace_in_files",
            description="🔁 Replace all occurrences of a string in matching files (supports dry-run, patterns, and recursion)."
        )
        builder.with_adk_tool_function(
            self.read_file_tool,
            name="read_file",
            description="📖 Read the content of a file, optionally by line range (e.g. '1-10')."
        )
        builder.with_adk_tool_function(
            self.write_file_tool,
            name="write_file",
            description="✍️ Write or append content to a file, with optional backup."
        )
        # Optional Tools: uncomment if needed
        # builder.with_adk_tool_function(
        #     self.search_in_files_tool,
        #     name="search_in_files",
        #     description="🔍 Search for a term in files, with optional surrounding context lines."
        # )
        # builder.with_adk_tool_function(
        #     self.list_directory_tool,
        #     name="list_directory",
        #     description="📂 List contents of a directory with filtering, recursion, and hidden file support."
        # )
        builder.with_adk_tool_function(
            self.create_specialized_agent_tool,
            name="create_specialized_agent",
            description="🤖 Create a new agent with a specialization like coder, writer, researcher, etc."
        )
        builder.with_adk_tool_function(
            self.remove_agent_tool,
            name="remove_agent",
            description="🗑️ Remove an existing agent (requires confirm=True)."
        )
        builder.with_adk_tool_function(
            self.list_agents_tool,
            name="list_agents",
            description="📋 List all agents in the system, optionally with details and system prompts."
        )
        builder.with_adk_tool_function(
            self.run_agent_background_tool,
            name="run_agent_background",
            description="⚙️ Run a task with a specific agent in the background (with priority)."
        )
        builder.with_adk_tool_function(
            self.get_background_tasks_status_tool,
            name="get_background_tasks_status",
            description="🔄 Show all background task statuses with agent, prompt, and runtime info."
        )
        builder.with_adk_tool_function(
            self.kill_background_task_tool,
            name="kill_background_task",
            description="⛔ Kill or cancel a background task by its ID (optionally force)."
        )
        # Optional: workspace tools
        # builder.with_adk_tool_function(
        #     self.change_workspace_tool,
        #     name="change_workspace",
        #     description="📁 Change the current workspace directory (create if missing)."
        # )
        builder.with_adk_tool_function(
            self.workspace_status_tool,
            name="workspace_status",
            description="📊 Get current workspace status, active agent, background task count, Git status, and file stats."
        )
        return builder

    async def run(self):
        """Main workspace CLI loop with enhanced error handling"""
        await self.init()
        while True:
            try:
                self.interrupt_count = 0
                user_input = await self.prompt_session.prompt_async(self.get_prompt_text(), multiline=False)
                if not user_input.strip():
                    continue
                if user_input.strip().startswith("/"):
                    await self.handle_workspace_command(user_input.strip())
                else:
                    try:
                        await self.handle_agent_request(user_input.strip())
                    except KeyboardInterrupt:
                        self.formatter.print_warning("Operation interrupted by user")
                        continue
            except (EOFError, KeyboardInterrupt) as e:
                if self.interrupt_count == 0 and not isinstance(e, EOFError):
                    self.interrupt_count += 1
                    self.formatter.print_info("Press Ctrl+D or type /exit to quit")
                    continue
                break
            except Exception as e:
                self.formatter.print_error(f"Unexpected error: {e}")
                continue
        await self.cleanup()

    async def cleanup(self):
        """Clean shutdown with progress tracking"""
        if self.background_tasks:
            active_tasks = [t for t in self.background_tasks.values() if not t['task'].done()]
            if active_tasks:
                await self.formatter.process("Cleaning up background tasks", self.cancel_background_tasks())
        self.formatter.print_success("ISAA Workspace Manager shutdown complete. Goodbye! 👋")

    async def cancel_background_tasks(self):
        """Cancel all background tasks"""
        for task_info in self.background_tasks.values():
            if not task_info['task'].done():
                task_info['task'].cancel()
        await asyncio.sleep(0.5)

    async def handle_agent_request(self, request: str):
        """Handle requests to the workspace supervisor agent with progress"""
        try:
            agent = await self.isaa_tools.get_agent(self.active_agent_name)
            agent.progress_callback = self.formatter.print_event
            response = await self.isaa_tools.run_agent(
                name=self.active_agent_name,
                text=request,
                session_id=self.session_id,
                progress_callback=self.progress_callback
            )
            await self.formatter.print_agent_response(response)
        except Exception as e:
            self.formatter.print_error(f"Error processing agent request: {e}")

    async def progress_callback(self, data: dict):
        """Enhanced progress callback with better formatting"""
        event_type = data.get("type", "unknown")
        if event_type == "tool_call":
            tool_name = data.get("tool_name", "Unknown")
            self.formatter.print_info(f"🔧 Using tool: {tool_name}")
        elif event_type == "thinking":
            content = data.get('content', 'Processing...')
            self.formatter.log_state("THINKING", {"process": content})
        else:
            self.formatter.print_event(data)

    async def handle_workspace_command(self, user_input: str):
        """Handle workspace management commands with enhanced formatting"""
        print(user_input)
        parts = user_input.split()
        command, args = parts[0].lower(), parts[1:]
        command_map = {
            "/workspace": self.handle_workspace_cmd,
            "/agent": self.handle_agent_cmd,
            "/tasks": self.handle_tasks_cmd,
            "/context": self.handle_context_cmd,
            "/monitor": self.handle_monitor_cmd,
            "/system": self.handle_system_cmd,
            "/help": self.handle_help_cmd,
            "/exit": self.handle_exit_cmd,
            "/quit": self.handle_exit_cmd,
            "/clear": self.handle_clear_cmd,
        }
        handler = command_map.get(command)
        if not handler:
            for cmd_ in command_map.keys():
                if cmd_.startswith(command):
                    handler = command_map.get(cmd_)
                    break
            else:
                self.formatter.print_error(f"Unknown command: {command} {args}")
                self.formatter.print_info("Type /help for available commands")
                return
        try:
            await handler(args)
        except Exception as e:
            import traceback
            self.formatter.print_error(f"Command failed: {e}\n{traceback.format_exc()}")

    async def handle_workspace_cmd(self, args: list[str]):
        """Handle workspace commands with enhanced formatting and direct calls."""
        if not args:
            self.formatter.print_error("Usage: /workspace <status|cd|ls|info>")
            return
        sub_command = args[0]
        if sub_command == "status":
            try:
                status_output = await self.formatter.process(
                    "Getting workspace status",
                    self.workspace_status_tool(include_git=True, include_files=True)
                )
                print(status_output)
            except Exception as e:
                self.formatter.print_error(f"Error getting workspace status: {e}")
        elif sub_command == "cd":
            if len(args) < 2:
                self.formatter.print_error("Usage: /workspace cd <directory>")
                return
            try:
                result = await self.change_workspace_tool(args[1])
                if "✅" in result:
                    self.formatter.print_success(result.replace("✅ ", ""))
                else:
                    self.formatter.print_error(result.replace("❌ ", ""))
            except Exception as e:
                self.formatter.print_error(f"Error changing directory: {e}")
        elif sub_command == "ls":
            directory = args[1] if len(args) > 1 else "."
            recursive = "--recursive" in args or "-r" in args
            show_hidden = "--all" in args or "-a" in args
            try:
                listing = await self.list_directory_tool(directory, recursive, show_hidden=show_hidden)
                print(listing)
            except Exception as e:
                self.formatter.print_error(f"Error listing directory: {e}")
        elif sub_command == "info":
            self.formatter.print_section(
                "Workspace Information",
                f"📁 Current Path: {self.workspace_path}\n"
                f"🤖 Active Agent: {self.active_agent_name}\n"
                f"📝 Session ID: {self.session_id}\n"
                f"💾 Data Directory: {self.app.data_dir}"
            )
        else:
            self.formatter.print_error(f"Unknown workspace command: {sub_command}")

    async def handle_agent_cmd(self, args: list[str]):
        """Handle agent control commands with direct calls."""
        if not args:
            self.formatter.print_error("Usage: /agent <list|switch|status>")
            return
        sub_command = args[0]
        if sub_command == "list":
            try:
                detailed = "--detailed" in args or "-d" in args
                agent_list = await self.list_agents_tool(detailed=detailed)
                print(agent_list)
            except Exception as e:
                self.formatter.print_error(f"Error listing agents: {e}")
        elif sub_command == "switch":
            if len(args) < 2:
                self.formatter.print_error("Usage: /agent switch <name>")
                return
            agent_name = args[1]
            agents = self.isaa_tools.config.get("agents-name-list", [])
            if agent_name in agents:
                old_agent, self.active_agent_name = self.active_agent_name, agent_name
                self.formatter.print_success(f"Switched from '{old_agent}' to '{agent_name}'")
            else:
                self.formatter.print_error(f"Agent '{agent_name}' not found")
                agents_list = "\n".join([f"  • {agent}" for agent in agents])
                self.formatter.print_section("Available Agents", agents_list)
        elif sub_command == "status":
            self.formatter.print_section(
                f"Agent Status: {self.active_agent_name}",
                f"🤖 Active Agent: {self.active_agent_name}\n"
                f"📝 Session: {self.session_id}\n"
                f"🔧 Tools: Available via agent capabilities"
            )
        else:
            self.formatter.print_error(f"Unknown agent command: {sub_command}")

    async def handle_tasks_cmd(self, args: list[str]):
        """Handle background task management with direct calls."""
        if not args:
            self.formatter.print_error("Usage: /tasks <list|attach|kill|monitor|status>")
            return
        sub_command = args[0]
        if sub_command in ["list", "status"]:
            try:
                show_completed = "--all" in args or "-a" in args
                status_output = await self.get_background_tasks_status_tool(show_completed=show_completed)
                print(status_output)
            except Exception as e:
                self.formatter.print_error(f"Error getting task status: {e}")
        elif sub_command == "attach":
            if len(args) < 2:
                self.formatter.print_error("Usage: /tasks attach <task_id>")
                return
            try:
                task_id = int(args[1])
                if task_id not in self.background_tasks:
                    self.formatter.print_error(f"Task {task_id} not found")
                    return
                task_info = self.background_tasks[task_id]
                task = task_info['task']
                if task.done():
                    try:
                        self.formatter.print_section(f"Task {task_id} Results", str(task.result()))
                    except Exception as e:
                        self.formatter.print_error(f"Task {task_id} failed: {e}")
                else:
                    result = await self.formatter.process(f"Attaching to task {task_id}", task)
                    self.formatter.print_section(f"Task {task_id} Results", str(result))
            except ValueError:
                self.formatter.print_error("Invalid task ID")
            except Exception as e:
                self.formatter.print_error(f"Error attaching to task: {e}")
        elif sub_command == "kill":
            if len(args) < 2:
                self.formatter.print_error("Usage: /tasks kill <task_id> [--force]")
                return
            try:
                task_id = int(args[1])
                force = "--force" in args
                result = await self.kill_background_task_tool(task_id, force)
                if "✅" in result:
                    self.formatter.print_success(result.replace("✅ ", ""))
                else:
                    self.formatter.print_warning(result.replace("⚠️", "").replace("ℹ️", ""))
            except ValueError:
                self.formatter.print_error("Invalid task ID")
            except Exception as e:
                self.formatter.print_error(f"Error killing task: {e}")
        elif sub_command == "monitor":
            if self.background_tasks:
                running_tasks = [(tid, info) for tid, info in self.background_tasks.items() if
                                 not info['task'].done()]
                if running_tasks:
                    headers = ["ID", "Agent", "Status", "Elapsed", "Prompt"]
                    rows = []
                    for tid, tinfo in running_tasks:
                        elapsed = asyncio.get_event_loop().time() - tinfo['started']
                        rows.append([str(tid), tinfo['agent'][:15], "Running", f"{elapsed:.1f}s",
                                     tinfo['prompt'][:30] + "..."])
                    self.formatter.print_table(headers, rows)
                else:
                    self.formatter.print_info("No running tasks to monitor")
            else:
                self.formatter.print_info("No background tasks available")
        else:
            self.formatter.print_error(f"Unknown tasks command: {sub_command}")

    async def handle_context_cmd(self, args: list[str]):
        """Handle context management with enhanced formatting"""
        if not args:
            self.formatter.print_error("Usage: /context <save|load|list|clear|delete>")
            return
        sub_command = args[0]
        try:
            agent = await self.isaa_tools.get_agent(self.active_agent_name)
        except Exception as e:
            self.formatter.print_error(f"Could not get active agent: {e}")
            return

        if sub_command == "save":
            session_name = args[1] if len(args) > 1 else self.session_id
            history = agent.message_history.get(self.session_id, [])
            context_file = Path(self.app.data_dir) / f"context_{session_name}.json"
            await self.formatter.process(f"Saving context '{session_name}'", self.save_context(context_file, history))
            self.formatter.print_success(f"Context saved as '{session_name}' ({len(history)} messages)")

        elif sub_command == "load":
            if len(args) < 2:
                self.formatter.print_error("Usage: /context load <session_name>")
                return
            session_name = args[1]
            context_file = Path(self.app.data_dir) / f"context_{session_name}.json"
            try:
                history = await self.formatter.process(f"Loading context '{session_name}'", self.load_context(context_file))
                agent.message_history[self.session_id] = history
                self.formatter.print_success(f"Context '{session_name}' loaded ({len(history)} messages)")
            except FileNotFoundError:
                self.formatter.print_error(f"Context '{session_name}' not found")
            except Exception as e:
                self.formatter.print_error(f"Error loading context: {e}")
        elif sub_command == "list":
            context_files = list(Path(self.app.data_dir).glob("context_*.json"))
            if context_files:
                contexts_data = []
                for f in context_files:
                    try:
                        with open(f, 'r') as file:
                            data = json.load(file)
                        contexts_data.append([f.stem.replace("context_", ""), f"{len(data)} messages", f.stat().st_mtime])
                    except:
                        contexts_data.append([f.stem.replace("context_", ""), "Error", 0])
                contexts_data.sort(key=lambda x: x[2], reverse=True)
                headers = ["Context", "Size", "Modified"]
                rows = [[ctx[0], ctx[1], "Recently"] for ctx in contexts_data] # Simplified time
                self.formatter.print_table(headers, rows)
            else:
                self.formatter.print_info("No saved contexts found")
        elif sub_command == "clear":
            message_count = len(agent.message_history.get(self.session_id, []))
            agent.message_history[self.session_id] = []
            self.formatter.print_success(f"Context cleared ({message_count} messages removed)")
        elif sub_command == "delete":
            if len(args) < 2:
                self.formatter.print_error("Usage: /context delete <session_name>")
                return
            session_name = args[1]
            context_file = Path(self.app.data_dir) / f"context_{session_name}.json"
            try:
                context_file.unlink()
                self.formatter.print_success(f"Context '{session_name}' deleted")
            except FileNotFoundError:
                self.formatter.print_error(f"Context '{session_name}' not found")
            except Exception as e:
                self.formatter.print_error(f"Error deleting context: {e}")
        else:
            self.formatter.print_error(f"Unknown context command: {sub_command}")

    async def save_context(self, file_path: Path, history: list):
        """Save context with async operation"""
        with open(file_path, "w") as f:
            json.dump(history, f, indent=2)
        await asyncio.sleep(0.1)

    async def load_context(self, file_path: Path):
        """Load context with async operation"""
        with open(file_path, "r") as f:
            history = json.load(f)
        await asyncio.sleep(0.1)
        return history

    async def handle_monitor_cmd(self, args: list[str]):
        """Handle monitoring commands"""
        self.formatter.print_warning("Advanced monitoring features coming soon")
        self.formatter.print_info("Use /tasks monitor for basic task monitoring")

    async def handle_system_cmd(self, args: list[str]):
        """Handle system commands with enhanced formatting"""
        if not args:
            self.formatter.print_error("Usage: /system <status|config|performance>")
            return
        sub_command = args[0]
        if sub_command == "status":
            agents_count = len(self.isaa_tools.config.get("agents-name-list", []))
            bg_running = len([t for t in self.background_tasks.values() if not t['task'].done()])
            bg_total = len(self.background_tasks)
            status_data = [
                ["Workspace", str(self.workspace_path)], ["Active Agent", self.active_agent_name],
                ["Total Agents", str(agents_count)], [ "Running Tasks", f"{bg_running}/{bg_total}"],
                ["Session ID", self.session_id], ["Data Directory", str(self.app.data_dir)]
            ]
            self.formatter.print_table(["Property", "Value"], status_data)
        elif sub_command == "config":
            config_preview = {k: v for k, v in self.isaa_tools.config.items() if "api_key" not in k}
            config_json = json.dumps(config_preview, indent=2)
            self.formatter.print_code_block(config_json)
        elif sub_command == "performance":
            import psutil, platform
            perf_data = [
                ["System", platform.system()], ["CPU Usage", f"{psutil.cpu_percent()}%"],
                ["Memory Usage", f"{psutil.virtual_memory().percent}%"],
                ["Python Version", platform.python_version()], ["Process PID", str(os.getpid())],
            ]
            self.formatter.print_table(["Metric", "Value"], perf_data)
        else:
            self.formatter.print_error(f"Unknown system command: {sub_command}")

    async def handle_help_cmd(self, args: list[str]):
        """Display comprehensive help with enhanced formatting"""
        self.formatter.log_header("ISAA Workspace Manager - Help & Reference")
        self.formatter.print_section("🗣️  Natural Language Usage", "...")
        command_data = [
            ["/workspace status", "Show workspace information"], ["/workspace cd <dir>", "Change workspace directory"],
            ["/workspace ls [dir]", "List directory contents"], ["/agent list", "Show all agents"],
            ["/agent switch <name>", "Switch active agent"], ["/tasks list", "Show background tasks"],
            ["/tasks attach <id>", "Attach to task output"], ["/tasks kill <id>", "Cancel background task"],
            ["/context save [name]", "Save conversation"], ["/context load <name>", "Load conversation"],
            ["/context list", "Show saved contexts"], ["/system status", "Show system status"],
            ["/clear", "Clear screen"], ["/exit", "Exit workspace manager"]
        ]
        self.formatter.print_table(["Command", "Description"], command_data)
        self.formatter.print_section("🤖 Agent Capabilities", "...")
        self.formatter.print_section("💡 Tips & Tricks", "...")

    async def handle_clear_cmd(self, args: list[str]):
        """Clear screen and show welcome"""
        os.system('clear' if os.name == 'posix' else 'cls')
        await self.show_welcome()

    async def handle_exit_cmd(self, args: list[str]):
        """Exit the workspace manager with enhanced feedback"""
        running_tasks = [t for t in self.background_tasks.values() if not t['task'].done()]
        if running_tasks:
            self.formatter.print_warning(f"You have {len(running_tasks)} running background tasks")
            try:
                confirm = await self.prompt_session.prompt_async("Exit anyway? (y/N): ")
                if confirm.lower() not in ['y', 'yes']:
                    self.formatter.print_info("Exit cancelled")
                    return
            except (KeyboardInterrupt, EOFError):
                self.formatter.print_info("Exit cancelled")
                return
        self.formatter.print_info("Shutting down ISAA Workspace Manager...")
        exit(0)


async def run(app, *args, **kwargs):
    """Entry point for the enhanced ISAA CLI"""
    app = get_app("isaa_cli_instance")
    cli = WorkspaceIsaasCli(app)
    try:
        await cli.run()
    except KeyboardInterrupt:
        print("\n🛑 Shutdown interrupted.")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(run(None))



