import asyncio
import collections
import json
import os
import platform
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import glob
import psutil
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application,in_terminal, get_app_or_none as get_pt_app
from prompt_toolkit.completion import (FuzzyCompleter, NestedCompleter,
                                       PathCompleter, WordCompleter)
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.shortcuts import prompt

from toolboxv2 import get_app
from toolboxv2.mods.isaa.CodingAgent.live import EnhancedVerboseOutput
from toolboxv2.mods.isaa.module import detect_shell, Tools as Isaatools
from toolboxv2.utils.extras.Style import Spinner, Style

from prompt_toolkit.formatted_text import ANSI



NAME = "isaa_cli"

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

class WorkspaceIsaasCli:
    """Advanced ISAA CLI with comprehensive agent tools and enhanced formatting"""

    def __init__(self, app_instance: Any):
        self.app = app_instance
        self.isaa_tools: Isaatools = app_instance.get_mod("isaa")
        self.isaa_tools.stuf = True #
        self.formatter = EnhancedVerboseOutput(verbose=True, print_func=app_instance.print)
        self.active_agent_name = "workspace_supervisor"
        self.session_id = "workspace_session"
        self.history = FileHistory(Path(self.app.data_dir) / "isaa_cli_history.txt")

        # Dedizierte Completer für Pfade
        self.dir_completer = PathCompleter(only_directories=True, expanduser=True)
        self.path_completer = PathCompleter(expanduser=True)

        self.completion_dict = self.build_workspace_completer()
        self.key_bindings = self.create_key_bindings()

        self.dynamic_completions_file = Path(self.app.data_dir) / "isaa_cli_completions.json"
        self.dynamic_completions = {"world_tags": [], "context_tags": []}
        self._load_dynamic_completions()  # Methode zum Laden der Tags beim Start)

        self.session_stats = self._init_session_stats()
        self.prompt_start_time = None

        self.prompt_session = PromptSession(
            history=self.history,
            completer= FuzzyCompleter(NestedCompleter.from_nested_dict(self.completion_dict)),#self.completer_dict_to_world_completer(),
            complete_while_typing=True,
            # Key bindings are now managed by the main Application object
        )


        self.background_tasks = {}
        self.interrupt_count = 0
        from toolboxv2 import __init_cwd__
        self.workspace_path = __init_cwd__
        self.default_exclude_dirs = [
            "node_modules",
            "__pycache__",
            ".git",
            ".svn",
            "CVS",
            ".bzr",
            ".hg",
            "build",
            "dist",
            "target",
            "out",
            "bin",
            "obj",
            ".idea",
            ".vscode",
            ".project",
            ".settings",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".DS_Store"
        ]

    def _init_session_stats(self) -> Dict:
        """Initialisiert die Struktur für die Sitzungsstatistiken."""
        return {
            "session_start_time": asyncio.get_event_loop().time(),
            "interaction_time": 0.0,
            "agent_running_time": 0.0,
            "total_cost": 0.0,
            "total_tokens": {"prompt": 0, "completion": 0},
            "agents": {},  # Statistiken pro Agent
            "tools": {
                "total_calls": 0,
                "failed_calls": 0,
                "calls_by_name": {}
            }
        }

    def _load_dynamic_completions(self):
        """Lädt dynamische Vervollständigungs-Tags aus einer JSON-Datei."""
        try:
            if self.dynamic_completions_file.exists():
                with open(self.dynamic_completions_file, 'r') as f:
                    data = json.load(f)
                    self.dynamic_completions["world_tags"] = data.get("world_tags", [])
                    self.dynamic_completions["context_tags"] = data.get("context_tags", [])
        except (IOError, json.JSONDecodeError):
            self.dynamic_completions = {"world_tags": [], "context_tags": []}

    async def _save_dynamic_completions(self):
        """Speichert die aktuellen dynamischen Vervollständigungs-Tags in einer JSON-Datei."""
        self.dynamic_completions["world_tags"] = sorted(list(set(self.dynamic_completions["world_tags"])))
        self.dynamic_completions["context_tags"] = sorted(list(set(self.dynamic_completions["context_tags"])))
        try:
            with open(self.dynamic_completions_file, 'w') as f:
                json.dump(self.dynamic_completions, f, indent=2)
        except IOError as e:
            self.formatter.print_error(f"Konnte Vervollständigungen nicht speichern: {e}")

        # In der `WorkspaceIsaasCli`-Klasse

    async def _update_completer(self):
        """Aktualisiert den prompt_toolkit-Completer mit den neuesten dynamischen Daten."""
        # Agentennamen live aus der Konfiguration laden (bestehender Code)
        try:
            agent_names = self.isaa_tools.config.get("agents-name-list", [])
            self.completion_dict["/agent"]["switch"] = WordCompleter(agent_names, ignore_case=True)
        except Exception:
            self.completion_dict["/agent"]["switch"] = None

        # World-Tags aus der geladenen/gespeicherten Liste (bestehender Code)
        world_tags = self.dynamic_completions.get("world_tags", [])
        if world_tags:
            self.completion_dict["/world"]["load"] = WordCompleter(world_tags, ignore_case=True)

        # Context-Tags aus der geladenen/gespeicherten Liste (bestehender Code)
        context_tags = self.dynamic_completions.get("context_tags", [])
        if context_tags:
            completer = WordCompleter(context_tags, ignore_case=True)
            self.completion_dict["/context"]["load"] = completer
            self.completion_dict["/context"]["delete"] = completer

        # NEU: Task-IDs aus den laufenden Hintergrund-Tasks holen
        try:
            running_task_ids = [
                str(tid) for tid, tinfo in self.background_tasks.items()
                if not tinfo['task'].done()
            ]
            if running_task_ids:
                task_id_completer = WordCompleter(running_task_ids, ignore_case=True)
                self.completion_dict["/tasks"]["attach"] = task_id_completer
                self.completion_dict["/tasks"]["kill"] = task_id_completer
            else:
                # Wenn keine Tasks laufen, leere Completer setzen
                self.completion_dict["/tasks"]["attach"] = WordCompleter([])
                self.completion_dict["/tasks"]["kill"] = WordCompleter([])
        except Exception:
            self.completion_dict["/tasks"]["attach"] = WordCompleter([])
            self.completion_dict["/tasks"]["kill"] = WordCompleter([])

        try:
            import git
            repo = git.Repo(search_parent_directories=False)
            branch_names = [branch.name for branch in repo.branches]
            if branch_names:
                branch_completer = WordCompleter(branch_names, ignore_case=True)
                self.completion_dict["/system"]["branch"] = branch_completer
            else:
                if "/system" in self.completion_dict:
                    self.completion_dict["/system"]["branch"] = WordCompleter([])
        except ImportError:
            self.completion_dict["/system"]["branch"] = None
        except Exception as e:

            self.completion_dict["/system"]["branch"] = None


        self.prompt_session.completer = FuzzyCompleter(NestedCompleter.from_nested_dict(self.completion_dict)) #self.completer_dict_to_world_completer()

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
                "cd": self.dir_completer,
                "ls": self.path_completer,
                "info": None,
            },
            "/world": {
                "show": None, "add": None,
                "remove": None,
                "clear": None, "save": None, "list": None,
                "load": WordCompleter([]),  # Wird dynamisch gefüllt
            },
            "/agent": {
                "list": None, "status": None,
                "switch": WordCompleter([]),  # Wird dynamisch gefüllt
            },
            "/tasks": {
                "list": None, "attach": WordCompleter([]), "kill": WordCompleter([]), "status": None,
            },
            "/context": {
                "list": None, "clear": None, "save": None,
                "load": WordCompleter([]),  # Wird dynamisch gefüllt
                "delete": WordCompleter([]), # Wird dynamisch gefüllt
            },
            "/monitor": None,
            "/system": {"branch": WordCompleter([]), "config": None, "backup": None, "restore": None, "performance": None},
            "/help": None, "/quit": None, "/clear": None}
        return commands_dict

    def completer_dict_to_world_completer(self) -> WordCompleter:
        commands_dict = self.completion_dict
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
            path = Path(self.workspace_path / file_path)
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
            path = Path(self.workspace_path / file_path)
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
        base_path = Path(self.workspace_path / directory).resolve()
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
                                  show_hidden: bool = False, exclude_dirs: Optional[List[str]] = None):
        """List directory contents with advanced filtering and exclusion."""
        if exclude_dirs is None:
            exclude_dirs = self.default_exclude_dirs
        else:
            exclude_dirs.extend(self.default_exclude_dirs)
            # Remove duplicates
            exclude_dirs = list(set(exclude_dirs))

        try:
            path = self.workspace_path / directory
            if not path.exists():
                return f"❌ Directory {directory} does not exist"
            files, dirs = [], []

            items = path.rglob("*") if recursive else path.iterdir()

            for item in items:
                if not show_hidden and item.name.startswith('.'):
                    continue

                if item.is_dir():
                    if item.name in exclude_dirs:
                        continue
                    dirs.append(item)
                elif item.is_file():
                    files.append(item)

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

            now = asyncio.get_event_loop().time()

            # Der Callback muss direkt an den Runner übergeben werden.

            async def comp_helper():
                res = await self.isaa_tools.run_agent(
                    agent_name,
                    task_prompt,
                    session_id=session_id,
                    progress_callback=self.monitoring_progress_callback
                )
                self.background_tasks[task_id]['end_time'] = asyncio.get_event_loop().time()
                return res

            task = asyncio.create_task(
                comp_helper(), name=f"BGTask-{agent_name}-{session_id}-{str(uuid.uuid4())[:8]}"
            )
            task_id = len(self.background_tasks)

            self.background_tasks[task_id] = {
                'task': task,
                'agent': agent_name,
                'prompt': task_prompt[:100],
                'started': now,
                'end_time': None,
                'session_id': session_id,
                'priority': priority,
                'status': 'running',
                'last_activity': now,
                'last_event': 'created',
                'agent_state': 'Starting',
                'current_tool_name': None,
                'current_tool_input': None,
                'history': []
            }
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
            new_path = Path(self.workspace_path / directory).resolve()
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

    async def workspace_status_tool(self, include_git: bool = True, max_items_per_type: int = 15):
        """
        Displays a comprehensive and visually clean workspace status directly to the console.
        Features a color-coded overview and an elegant file tree.

        Args:
            include_git (bool): Whether to include the Git status section.
            max_items_per_type (int): The maximum number of files and directories to list.
        """
        try:
            # --- Haupt-Header ---
            self.formatter.log_header("Workspace Status")

            # --- Allgemeine Übersicht ---
            bg_running = len([t for t in self.background_tasks.values() if not t['task'].done()])
            bg_total = len(self.background_tasks)

            overview_text = (
                f"  Path:         {Style.CYAN(str(self.workspace_path))}\n"
                f"  Active Agent: {Style.YELLOW(self.active_agent_name)}\n"
                f"  Background:   {Style.BLUE(f'{bg_running} running')} / {bg_total} total"
                f"  Session ID: {self.session_id}\n"
            )
            self.formatter.print_section("Overview 📝", overview_text)

            # --- Git-Status ---

            git_info_lines = []
            if include_git:
                try:
                    git_res = subprocess.run(
                        ["git", "branch", "--show-current"], capture_output=True, text=True,
                        cwd=self.workspace_path, check=False, timeout=2
                    )
                    if git_res.returncode == 0 and git_res.stdout.strip():
                        branch = git_res.stdout.strip()
                        status_res = subprocess.run(
                            ["git", "status", "--porcelain"], capture_output=True, text=True,
                            cwd=self.workspace_path, check=False, timeout=2
                        )
                        has_changes = status_res.stdout.strip()
                        changes_text = "modified" if has_changes else "clean"
                        status_style = Style.RED if has_changes else Style.GREEN

                        git_info_lines.append(f"  Branch: {Style.MAGENTA(branch)} ({status_style(changes_text)})")
                    else:
                        git_info_lines.append(Style.GREY("  (Not a git repository or no active branch)"))
                except FileNotFoundError:
                    git_info_lines.append(Style.YELLOW("  ('git' command not found, status unavailable)"))
                except Exception as e:
                    git_info_lines.append(Style.RED(f"  (Error getting Git status: {e})"))

                self.formatter.print_section("Git Status 🔀", "\n".join(git_info_lines))

            # --- Verzeichnisübersicht mit Baumstruktur ---
            dir_listing_lines = []
            try:
                items = [item for item in self.workspace_path.iterdir() if not item.name.startswith('.')]
                # Sortiere Ordner zuerst, dann Dateien, beides alphabetisch
                dirs = sorted([d for d in items if d.is_dir()], key=lambda p: p.name.lower())
                files = sorted([f for f in items if f.is_file()], key=lambda p: p.name.lower())

                all_items = dirs + files

                if not all_items:
                    dir_listing_lines.append(Style.GREY("  (Directory is empty)"))
                else:
                    display_items = all_items[:max_items_per_type]

                    for i, item in enumerate(display_items):
                        # Bestimme den Baum-Präfix
                        is_last = (i == len(display_items) - 1)
                        prefix = "└──" if is_last else "├──"

                        if item.is_dir():
                            item_text = f"📁 {Style.BLUE(item.name)}/"
                        else:
                            item_text = f"📄 {item.name}"

                        dir_listing_lines.append(f"  {Style.GREY(prefix)} {item_text}")

                    if len(all_items) > max_items_per_type:
                        remaining = len(all_items) - max_items_per_type
                        dir_listing_lines.append(Style.GREY(f"  ... and {remaining} more items"))

            except Exception as e:
                dir_listing_lines.append(Style.RED(f"  Error listing directory contents: {e}"))

            self.formatter.print_section("Directory Contents 📁", "\n".join(dir_listing_lines))

            agents_count = len(self.isaa_tools.config.get("agents-name-list", []))
            bg_running = len([t for t in self.background_tasks.values() if not t['task'].done()])
            bg_total = len(self.background_tasks)
            status_data = [
                ["Workspace", str(self.workspace_path)],
                ["Active Agent", self.active_agent_name],
                ["Total Agents", str(agents_count)],
                ["Running Tasks", f"{bg_running}/{bg_total}"],
                ["Session ID", self.session_id],
                ["Data Directory", str(self.app.data_dir)]
            ]
            self.formatter.print_table(["Eigenschaft", "Wert"], status_data)

            print()  # Add a final newline for spacing
            return "\n".join([overview_text] + git_info_lines+ dir_listing_lines + [f"{e}-{w}" for e,w in status_data])
        except Exception as e:
            self.formatter.print_error(f"Error generating workspace status: {str(e)}")

    async def show_welcome(self):
        """Display enhanced welcome with status overview"""
        welcome_text = "ISAA CLI Assistant"
        subtitle = "Intelligent System Agents & Automation"

        # Calculate padding for centering
        terminal_width = os.get_terminal_size().columns
        welcome_len = len(welcome_text)
        subtitle_len = len(subtitle)

        welcome_padding = (terminal_width - welcome_len) // 2
        subtitle_padding = (terminal_width - subtitle_len) // 2

        print()
        print(Style.CYAN("═" * terminal_width))
        print()
        print(" " * welcome_padding + Style.Bold(Style.BLUE(welcome_text)))
        print(" " * subtitle_padding + Style.GREY(subtitle))
        print()
        print(Style.CYAN("═" * terminal_width))
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
            f"{Style.YELLOW('●')} Type naturally - the agent will use tools to help you",
            f"{Style.YELLOW('●')} Type {Style.CYAN('/help')} for available commands",
            f"{Style.YELLOW('●')} Use {Style.CYAN('Tab')} for autocompletion",
            f"{Style.YELLOW('●')} {Style.CYAN('Ctrl+C')} to interrupt, {Style.CYAN('Ctrl+D')} to exit",
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
        if self.active_agent_name != "workspace_supervisor":
            self.active_agent_name = "workspace_supervisor"
        builder = self.isaa_tools.get_agent_builder(self.active_agent_name)
        builder.with_system_message(
            """You are the Workspace Supervisor, a sophisticated AI agent designed to manage a development environment with precision and efficiency.

**Your Core Directives:**

*   **Orchestrate Tasks:** Your primary role is to manage files, directories, and complex workflows. You are the central coordinator for all activities within the workspace.
*   **Delegate and Conquer:** You are not meant to perform every task yourself. You have a team of specialized agents at your disposal. Delegate tasks like coding, writing, or analysis to the appropriate agent. For any long-running process, you MUST execute it as a background task to keep the main interface responsive.
*   **Maintain Clarity:** Report your actions clearly and concisely. Provide detailed feedback on the success or failure of operations. Keep the workspace organized and clean.
*   **Master Your Tools:** You have a comprehensive set of tools for file manipulation, agent management, and task execution. Use them wisely and efficiently.

**Your Workflow:**

1.  **Analyze the Request:** Understand the user's goal.
2.  **Formulate a Plan:** Decide which tools and agents are needed.
3.  **Execute and Delegate:** Run commands and delegate to other agents, preferably in the background.
4.  **Report Results:** Inform the user of the outcome.

**Key Principles:**

*   **Efficiency:** Always choose the most direct path to accomplish a goal.
*   **Robustness:** Anticipate and handle potential errors.
*   **Clarity:** Communicate with the user in a clear and professional manner.
*   **Proactiveness:** When appropriate, suggest next steps or improvements."""
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
        builder.with_adk_tool_function(
            self.list_directory_tool,
            name="list_directory",
            description="📂 List contents of a directory with filtering, recursion, and hidden file support."
        )
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
                await self._update_completer()
                self.interrupt_count = 0
                self.prompt_start_time = asyncio.get_event_loop().time()

                # This is the correct way to call the prompt.
                user_input = await self.prompt_session.prompt_async(self.get_prompt_text())

                # Calculate interaction duration and update session stats
                if self.prompt_start_time:
                    interaction_duration = asyncio.get_event_loop().time() - self.prompt_start_time
                    self.session_stats["interaction_time"] += interaction_duration
                    self.prompt_start_time = None

                if not user_input.strip():
                    continue
                if user_input.strip().startswith("!"):
                    await self._handle_shell_command(user_input.strip()[1:])
                elif user_input.strip().startswith("/"):
                    await self.handle_workspace_command(user_input.strip())
                else:
                    await self.handle_agent_request(user_input.strip())
            except (EOFError, KeyboardInterrupt) as e:
                if self.interrupt_count == 0 and not isinstance(e, EOFError):
                    self.interrupt_count += 1
                    self.formatter.print_info("Press Ctrl+D or type /quit to quit")
                    continue
                break
            except Exception as e:
                self.formatter.print_error(f"Unexpected error in main loop: {e}")
                import traceback
                print(traceback.format_exc())
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

    def _display_session_summary(self):
        """Displays a summary of the session stats upon exit."""
        self.formatter.log_header("Session Summary")

        now = asyncio.get_event_loop().time()
        total_duration = now - self.session_stats['session_start_time']

        # Zeit-Statistiken
        self.formatter.print_section("Time Usage", (
            f"Total Session: {total_duration:.2f}s\n"
            f"User Interaction: {human_readable_time(self.session_stats['interaction_time'])}\n"
            f"Agent Processing: {human_readable_time(self.session_stats['agent_running_time'])}"
        ))

        # Kosten und Token
        self.formatter.print_section("Resource Usage", (
            f"Total Estimated Cost: ${self.session_stats['total_cost']:.4f}\n"
            f"Total Prompt Tokens: {self.session_stats['total_tokens']['prompt']}\n"
            f"Total Completion Tokens: {self.session_stats['total_tokens']['completion']}"
        ))

        # Tool-Nutzung
        tool_stats = self.session_stats["tools"]
        tool_summary = (
            f"Total Calls: {tool_stats['total_calls']}\n"
            f"Failed Calls: {tool_stats['failed_calls']}"
        )
        self.formatter.print_section("Tool Calls", tool_summary)
        if tool_stats['calls_by_name']:
            headers = ["Tool Name", "Success", "Fail"]
            rows = [[name, counts['success'], counts['fail']] for name, counts in tool_stats['calls_by_name'].items()]
            self.formatter.print_table(headers, rows)

        # Agenten-spezifische Statistiken
        if self.session_stats['agents']:
            self.formatter.print_section("Agent Specifics", "")
            headers = ["Agent Name", "Cost ($)", "Tokens (P/C)", "Tool Calls"]
            rows = []
            for name, data in self.session_stats['agents'].items():
                rows.append([
                    name,
                    f"{data['cost']:.4f}",
                    f"{data['tokens']['prompt']}/{data['tokens']['completion']}",
                    data['tool_calls']
                ])
            self.formatter.print_table(headers, rows)

    async def handle_agent_request(self, request: str):
        """
        Handles requests to the workspace agent, allowing interruption with Ctrl+C.
        This version uses get_app() to reliably access the application instance for UI suspension.
        """
        agent_task = None
        start_time = asyncio.get_event_loop().time()

        # Use the official prompt_toolkit function to get the active application instance.
        # This is the correct way to access it after a prompt has finished.
        from prompt_toolkit.widgets import TextArea
        kb = KeyBindings()
        agent_task = None  # Will hold the asyncio.Task

        # Handler for Ctrl+C
        @kb.add('c-c')
        def _(event):
            nonlocal agent_task
            if agent_task and not agent_task.done():
                agent_task.cancel()
            event.app.exit()

        # Handler for Esc
        @kb.add('escape')
        def _(event):
            nonlocal agent_task
            if agent_task and not agent_task.done():
                agent_task.cancel()
            event.app.exit()

        # Statuszeile ganz unten mit invertierter Farbe
        from prompt_toolkit.layout.dimension import D
        content_area = TextArea(
            text="",
            read_only=True
        )

        # Status bar at the bottom with inverted colors

        main_app = Application(full_screen=False, key_bindings=kb,  layout=Layout(HSplit([
            content_area,
        ])))

        self.formatter.print_info(Style.GREY("Agent is running... Cancel with Ctrl+C or ESC"))

        if not main_app:
            self.formatter.print_error(
                "Could not get application instance. Agent will run without clean UI suspension."
            )
            # As a fallback, we could run the agent directly, but the output would
            # conflict with the prompt. For now, we abort the request.
            return
        try:
            # Prepare the agent task before suspending the UI
            agent_task = asyncio.create_task(self.isaa_tools.run_agent(
                name=self.active_agent_name,
                text=request,
                session_id=self.session_id,
                progress_callback=self.progress_callback
            ))


            async def run_task():
                nonlocal agent_task
                try:
                    response = await agent_task
                    main_app.print_text("\n" + response)
                except asyncio.CancelledError:
                    main_app.print_text("\nOperation interrupted by user.")
                finally:
                    main_app.exit()

            main_app.create_background_task(run_task())

            # Run the application in asyncio event loop
            return await main_app.run_async()


        except KeyboardInterrupt:
            if agent_task and not agent_task.done():
                agent_task.cancel()
            print()
            self.formatter.print_warning("Operation interrupted by user.")

        except asyncio.CancelledError:
            # This is expected after a KeyboardInterrupt, so we can pass silently.
            pass

        except Exception as e:
            self.formatter.print_error(f"An unexpected error occurred during agent execution: {e}")
            import traceback
            print(traceback.format_exc())

        finally:
            duration = asyncio.get_event_loop().time() - start_time
            self.session_stats["agent_running_time"] += duration
            # Force a redraw of the prompt to clean up any visual artifacts
            if main_app:
                main_app.invalidate()

    async def _update_stats_from_event(self, event: Dict[str, Any]):
        """Handles all session statistics updates based on an ADK event."""
        agent_name = event.get("author", getattr(self, 'active_agent_name', 'unknown_agent'))

        if agent_name not in self.session_stats["agents"]:
            self.session_stats["agents"][agent_name] = {
                "cost": 0.0, "tokens": {"prompt": 0, "completion": 0}, "tool_calls": 0
            }

        if event.get("content") and event["content"].get("parts"):
            for part in event["content"]["parts"]:
                if "function_call" in part:
                    call_data = part["function_call"]
                    tool_name = call_data.get("name", "UnknownTool")
                    stats = self.session_stats["tools"]
                    stats["total_calls"] += 1
                    stats["calls_by_name"].setdefault(tool_name, {"success": 0, "fail": 0})
                    self.session_stats["agents"][agent_name]["tool_calls"] += 1

                if "function_response" in part:
                    response_data = part["function_response"]
                    tool_name = response_data.get("name", "UnknownTool")
                    response_content = response_data.get("response", {})
                    stats = self.session_stats["tools"]

                    is_error = "error" in response_content
                    if not is_error and isinstance(response_content.get("result"), str):
                        try:
                            result_json = json.loads(response_content["result"])
                            if isinstance(result_json, dict) and result_json.get("success") is False:
                                is_error = True
                        except json.JSONDecodeError:
                            pass

                    if is_error:
                        stats["failed_calls"] += 1
                        stats["calls_by_name"][tool_name]["fail"] += 1
                    else:
                        stats["calls_by_name"][tool_name]["success"] += 1

        if "usage_metadata" in event:
            usage = event["usage_metadata"]
            prompt_tokens = usage.get("prompt_token_count", 0)
            completion_tokens = usage.get("candidates_token_count", 0)

            if prompt_tokens > 0 or completion_tokens > 0:
                self.session_stats["total_tokens"]["prompt"] += prompt_tokens
                self.session_stats["total_tokens"]["completion"] += completion_tokens
                self.session_stats["agents"][agent_name]["tokens"]["prompt"] += prompt_tokens
                self.session_stats["agents"][agent_name]["tokens"]["completion"] += completion_tokens

            try:
                agent = await self.isaa_tools.get_agent(agent_name)
                if hasattr(agent, 'total_cost'):
                    new_cost = agent.total_cost or 0.0
                    previous_cost = self.session_stats["agents"][agent_name].get("cost", 0.0)
                    cost_delta = new_cost - previous_cost
                    if cost_delta > 0:
                        self.session_stats["agents"][agent_name]["cost"] = new_cost
                        self.session_stats["total_cost"] += cost_delta
            except Exception:
                pass # Ignore if agent not found or other issues

    async def progress_callback(self, event: Dict[str, Any]):
        """The main progress callback for the interactive CLI, handles printing."""
        await self._update_stats_from_event(event)

        agent_name = event.get("author", getattr(self, 'active_agent_name', 'unknown_agent'))

        if event.get("content") and event["content"].get("parts"):
            for part in event["content"]["parts"]:
                if "text" in part:
                    self.formatter.log_state("THINKING", {"process": Style.GREY(part.get("text", "...")[:255] + '...')})

                if "function_call" in part:
                    call_data = part["function_call"]
                    tool_name = call_data.get("name", "UnknownTool")
                    tool_args = call_data.get("args", {})
                    self.formatter.print_info(f"🔧 Using tool: {tool_name}")
                    self.formatter.print_info(f"   Args: {json.dumps(tool_args, indent=2, ensure_ascii=False)}")

                if "function_response" in part:
                    response_data = part["function_response"]
                    tool_name = response_data.get("name", "UnknownTool")
                    response_content = response_data.get("response", {})

                    is_error = "error" in response_content

                    if is_error:
                        self.formatter.print_error(f"❌ Tool '{tool_name}' failed.")
                    else:
                        self.formatter.print_success(f"✅ Tool '{tool_name}' succeeded.")

                    result = response_content.get("result", "")
                    self.formatter.print_section("Result", Style.GREY(str(result))[:255] + '...')

        if "usage_metadata" in event:
            try:
                agent = await self.isaa_tools.get_agent(agent_name)
                if hasattr(agent, 'total_cost'):
                    new_cost = agent.total_cost or 0.0
                    previous_cost = self.session_stats["agents"][agent_name].get("cost", 0.0)
                    cost_delta = new_cost - previous_cost
                    if cost_delta > 0:
                        self.formatter.print_info(f"Cost update for {agent_name}: +${cost_delta:.6f} (Total: ${new_cost:.6f})")
            except Exception:
                pass

    async def monitoring_progress_callback(self, event: Dict[str, Any]):
        """A dedicated callback for background tasks to update monitor state."""
        await self._update_stats_from_event(event)

        session_id = event.get("invocation_id")
        task_info = None
        for tid, tinfo in self.background_tasks.items():
            if tinfo.get('session_id') == session_id:
                task_info = tinfo
                break

        if task_info:
            now = asyncio.get_event_loop().time()
            task_info['last_activity'] = now
            event_log = {"time": now, "type": "unknown", "content": ""}

            if event.get("content") and event["content"].get("parts"):
                for part in event["content"]["parts"]:
                    if "text" in part and part["text"]:
                        task_info['agent_state'] = 'Thinking'
                        task_info['current_tool_name'] = None
                        task_info['current_tool_input'] = None
                        event_log = {"time": now, "type": "Thinking", "content": part["text"][:200]}

                    if "function_call" in part:
                        call = part["function_call"]
                        tool_name = call.get('name')
                        tool_args = call.get('args', {})
                        task_info['agent_state'] = 'Using Tool'
                        task_info['current_tool_name'] = tool_name
                        task_info['current_tool_input'] = json.dumps(tool_args, ensure_ascii=False)
                        event_log = {"time": now, "type": "Tool Call", "content": f"{tool_name}({tool_args})"}

                    if "function_response" in part:
                        task_info['agent_state'] = 'Processing'
                        # Werkzeugname bleibt für Kontext sichtbar
                        response = part["function_response"]
                        event_log = {"time": now, "type": "Tool Response",
                                     "content": f"Result for {response.get('name')}"}

            if event_log["type"] != "unknown":
                task_info['history'].append(event_log)

    async def _handle_shell_command(self, command: str):
        """
        Executes a shell command directly using asyncio's subprocess tools,
        streaming stdout and stderr in real-time.
        """
        if not command.strip():
            self.formatter.print_error("Shell command cannot be empty.")
            return

        self.formatter.print_info(f"🚀 Executing shell command: `{command}`")
        try:
            # Create a subprocess from the shell command.
            # We pipe stdout and stderr to capture them.
            shell_exe, cmd_flag = detect_shell()
            full_command = '"'+shell_exe + '" ' + cmd_flag + " " + command
            process = await asyncio.create_subprocess_shell(
                full_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # On Windows, you might need shell=True explicitly, but
                # create_subprocess_shell handles this.
            )

            # Helper function to read from a stream and print lines.
            async def stream_reader(stream, style_func):
                while not stream.at_eof():
                    line = await stream.readline()
                    if line:
                        # Decode bytes to string and print using the specified formatter style.
                        style_func(line.decode().strip())
                    await asyncio.sleep(0.01)  # Yield control briefly

            # Create concurrent tasks to read stdout and stderr.
            # This ensures we see output as it happens, regardless of which stream it's on.
            stdout_task = asyncio.create_task(stream_reader(process.stdout, self.formatter.print))
            stderr_task = asyncio.create_task(stream_reader(process.stderr,  lambda x: self.formatter.print_section("Shell Error", Style.RED(x))))

            # Wait for both stream readers to finish.
            await asyncio.gather(stdout_task, stderr_task)

            # Wait for the process to terminate and get its return code.
            return_code = await process.wait()

            if return_code == 0:
                self.formatter.print(f"✅ Command finished successfully (Exit Code: {return_code}).")
            else:
                self.formatter.print(f"⚠️ Command finished with an error (Exit Code: {return_code}).")

        except FileNotFoundError:
            self.formatter.print(f"Error: Command not found. Make sure it's installed and in your system's PATH.")
        except Exception as e:
            self.formatter.print(f"An unexpected error occurred while running the shell command: {e}")

    async def handle_workspace_command(self, user_input: str):
        """Handle workspace management commands with enhanced formatting"""
        parts = user_input.split()
        command, args = parts[0].lower(), parts[1:]
        command_map = {
            "/workspace": self.handle_workspace_cmd,
            "/world": self.handle_world_model_cmd,
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

    async def handle_world_model_cmd(self, args: list[str]):
        """Handle world model commands with enhanced formatting and direct calls."""
        if not args:
            self.formatter.print_error("Usage: /world <show|add|remove|clear|save|load>")
            return
        agent = await self.isaa_tools.get_agent(self.active_agent_name)
        sub_command = args[0]
        if sub_command == "show":
            try:
                world_model = agent.world_model.show()
                if world_model:
                    print(world_model)
                else:
                    self.formatter.print_info("World model is empty")
            except Exception as e:
                self.formatter.print_error(f"Error showing world model: {e}")
        elif sub_command == "list":
            world_models_list = self.dynamic_completions.get("world_tags", [])
            if world_models_list:
                print(world_models_list)
            else:
                self.formatter.print_info("World models list is empty")
        elif sub_command == "add":
            if len(args) < 2:
                self.formatter.print_error("Usage: /world add <key> <value>")
                return
            try:
                key, value = args[1], " ".join(args[2:])
                await agent.world_model.set(key, value)
                self.formatter.print_success(f"World model updated with {key}: {value}")
            except Exception as e:
                self.formatter.print_error(f"Error adding to world model: {e}")
        elif sub_command == "remove":
            if len(args) < 2:
                self.formatter.print_error("Usage: /world remove <key>")
                return
            try:
                key = args[1]
                await agent.world_model.remove(key)
                self.formatter.print_success(f"World model key '{key}' removed")
            except Exception as e:
                self.formatter.print_error(f"Error removing from world model: {e}")
        elif sub_command == "clear":
            try:
                agent.world_model.data = {}
                self.formatter.print_success("World model cleared")
            except Exception as e:
                self.formatter.print_error(f"Error clearing world model: {e}")
        elif sub_command == "save":
            # save to fil
            if len(args) < 2:
                self.formatter.print_error("Usage: /world save <tag>")
                return
            tag = args[1]
            world_model_file = Path(self.app.data_dir) / f"world_model_{self.active_agent_name}_{tag}.json"
            try:
                data = agent.world_model.data
                with open(world_model_file, "w") as f:
                    json.dump(data, f, indent=2)
                self.formatter.print_success("World model saved")

                if tag not in self.dynamic_completions["world_tags"]:
                    self.dynamic_completions["world_tags"].append(tag)
                    await self._save_dynamic_completions()

            except Exception as e:
                self.formatter.print_error(f"Error saving world model: {e}")
        elif sub_command == "load":
            if len(args) < 2:
                self.formatter.print_error("Usage: /world load <tag>")
                return
            tag = args[1]
            world_model_file = Path(self.app.data_dir) / f"world_model_{self.active_agent_name}_{tag}.json"
            try:
                with open(world_model_file, "r") as f:
                    data = json.load(f)
                agent.world_model.data = data
                self.formatter.print_success("World model loaded")
            except Exception as e:
                self.formatter.print_error(f"Error loading world model: {e}")
        else:
            self.formatter.print_error(f"Unknown world model command: {sub_command}")

    async def handle_workspace_cmd(self, args: list[str]):
        """Handle workspace commands with enhanced formatting and direct calls."""
        if not args:
            self.formatter.print_error("Usage: /workspace <status|cd|ls|info>")
            return
        sub_command = args[0]
        if sub_command == "status":
            try:
                await self.workspace_status_tool(include_git=True)
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
            self.formatter.print_error("Usage: /tasks <list|attach|kill|status>")
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
            if session_name not in self.dynamic_completions["context_tags"]:
                self.dynamic_completions["context_tags"].append(session_name)
                await self._save_dynamic_completions()
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
            # Get the active agent instance
            if not agent:
                self.formatter.print_error("No active agent to clear context for.")
                return

            # --- Step 1: Clear the local LiteLLM message history ---
            local_message_count = len(agent.message_history.get(self.session_id, []))
            agent.message_history[self.session_id] = []

            # --- Step 2: Reset the ADK session if configured ---
            if not agent.adk_runner or not agent.adk_session_service:
                self.formatter.print_warning("ADK not configured. Only local message history was cleared.")
                self.formatter.print_success(f"Local context cleared ({local_message_count} messages removed).")
                return

            self.formatter.print_info(f"Resetting ADK session '{self.session_id}'...")

            try:
                app_name = agent.adk_runner.app_name
                user_id = agent.amd.user_id or "adk_user"

                # To "reset" an ADK session, we effectively re-create it.
                # This overwrites the existing session on the service with a new, empty one.
                # We initialize its state from the agent's current World Model.
                initial_state = agent.world_model.to_dict() if agent.sync_adk_state else {}

                # This is the same logic used to create a session for the first time.
                # By calling it on an existing session_id, we achieve a reset.
                await agent.adk_session_service.create_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=self.session_id,
                    state=initial_state
                )

                self.formatter.print_success(
                    f"Full context cleared. Local history ({local_message_count} messages) "
                    f"and ADK session '{self.session_id}' have been reset."
                )

            except Exception as e:
                self.formatter.print_error(f"Failed to reset the ADK session: {e}")
                self.formatter.print_warning("Only the local message history was cleared.")
        elif sub_command == "delete":
            if len(args) < 2:
                self.formatter.print_error("Usage: /context delete <session_name>")
                return
            session_name = args[1]
            context_file = Path(self.app.data_dir) / f"context_{session_name}.json"
            try:
                context_file.unlink()
                if session_name in self.dynamic_completions["context_tags"]:
                    self.dynamic_completions["context_tags"].remove(session_name)
                    await self._save_dynamic_completions()
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
        """Enters an interactive monitoring mode for background tasks."""
        if not self.background_tasks:
            self.formatter.print_info("No background tasks to monitor.")
            return

        selected_task_index = 0
        detail_view_task_id = None
        stop_monitoring = False

        output_control = FormattedTextControl(text=ANSI(""), focusable=False)
        layout = Layout(HSplit([Window(content=output_control, always_hide_cursor=True)]))
        kb = KeyBindings()

        @kb.add('q')
        @kb.add('escape')
        @kb.add('c-c')
        def _(event):
            nonlocal stop_monitoring
            stop_monitoring = True
            event.app.exit()

        @kb.add('up')
        def _(event):
            nonlocal selected_task_index
            selected_task_index = max(0, selected_task_index - 1)

        @kb.add('down')
        def _(event):
            nonlocal selected_task_index
            num_tasks = len(self.background_tasks)
            selected_task_index = min(num_tasks - 1, selected_task_index + 1)

        @kb.add('k')
        async def _(event):
            nonlocal selected_task_index
            sorted_tasks = sorted(self.background_tasks.items())
            if 0 <= selected_task_index < len(sorted_tasks):
                task_id_to_kill, info = sorted_tasks[selected_task_index]
                if not info['task'].done():
                    await self.kill_background_task_tool(task_id_to_kill)

        @kb.add('d')
        def _(event):
            nonlocal selected_task_index, detail_view_task_id
            sorted_tasks = sorted(self.background_tasks.items())
            if 0 <= selected_task_index < len(sorted_tasks):
                task_id, _ = sorted_tasks[selected_task_index]
                # Toggle detail view for the selected task
                detail_view_task_id = task_id if detail_view_task_id != task_id else None

        app = Application(layout=layout, key_bindings=kb, full_screen=True)

        async def monitor_loop():
            nonlocal stop_monitoring, selected_task_index, detail_view_task_id
            while not stop_monitoring:
                now = asyncio.get_event_loop().time()
                lines = []

                # Header
                lines.append(Style.Bold(f"ISAA Agent Monitor @ {time.strftime('%H:%M:%S')}"))
                lines.append(Style.GREY("Use [↑/↓] to select, [k] to kill, [d] for details, [q] to quit.\n"))

                headers = ["ID", "Agent", "Status", "Runtime", "State", "Current Tool"]
                lines.append(Style.Underlined(" | ".join(f"{h:<15}" for h in headers)))

                sorted_tasks = sorted(self.background_tasks.items())
                if not sorted_tasks:
                    selected_task_index = 0
                else:
                    selected_task_index = min(selected_task_index, len(sorted_tasks) - 1)

                for idx, (tid, tinfo) in enumerate(sorted_tasks):
                    task = tinfo['task']

                    # Correct runtime calculation
                    if task.done() and tinfo.get('end_time') is None:
                        tinfo['end_time'] = tinfo.get('last_activity', now)

                    runtime = (tinfo['end_time'] or now) - tinfo['started']

                    # Determine status
                    if task.done():
                        if task.cancelled():
                            status = Style.YELLOW("Cancelled")
                        elif task.exception():
                            status = Style.RED("Failed")
                        else:
                            status = Style.GREEN("Completed")
                    else:
                        status = Style.CYAN("Running")

                    # Agent state styling
                    agent_state = tinfo.get('agent_state', 'n/a')
                    if agent_state == 'Using Tool':
                        agent_state = Style.BLUE(agent_state)
                    elif agent_state == 'Thinking':
                        agent_state = Style.MAGENTA(agent_state)
                    else:
                        agent_state = Style.GREY(agent_state)

                    row_data = [
                        str(tid),
                        tinfo['agent'],
                        status,
                        human_readable_time(runtime),
                        agent_state,
                        tinfo.get('current_tool_name', 'None') or 'None'
                    ]

                    # Build and style the row
                    row_str = " | ".join(f"{str(s):<15}" for s in row_data)
                    if idx == selected_task_index:
                        lines.append(HTML(f"<style bg='ansiyellow' fg='ansiblack'>{row_str}</style>"))
                    else:
                        lines.append(row_str)

                    # Show details if toggled
                    if detail_view_task_id == tid:
                        lines.append(Style.GREY("  " + "─" * 60))
                        if tinfo.get('history'):
                            for log in reversed(tinfo['history'][-10:]):  # Show last 10 events
                                log_time = time.strftime('%H:%M:%S', time.localtime(log['time']))
                                lines.append(HTML(
                                    f"  <grey>└ {log_time}</grey> <yellow>{log['type']:<15}</yellow> {log['content']}"))
                        else:
                            lines.append("  " + Style.GREY("└ No execution history recorded."))
                        lines.append(Style.GREY("  " + "─" * 60))

                output_control.text = ANSI("\n".join(lines))
                await asyncio.sleep(1)  # Refresh rate

        try:
            await asyncio.gather(app.run_async(), monitor_loop())
        except Exception as e:
            # Exit fullscreen mode before printing error
            print(f"\x1b[?1049l", end="")  # Exit alternate screen
            self.formatter.print_error(f"Monitor crashed: {e}")
        finally:
            self.monitoring_active = False

    async def handle_system_cmd(self, args: list[str]):
        """Verarbeitet Systembefehle, einschließlich Status, Konfiguration, Performance und Git-Backup/Restore."""
        if not args:
            self.formatter.print_error("Nutzung: /system <branch|config|performance|backup|restore|log>")
            return

        sub_command = args[0].lower()

        if sub_command == "branch":
            try:
                import git
                repo = git.Repo(search_parent_directories=True)
            except ImportError:
                self.formatter.print_error("The 'GitPython' library is not installed. Run 'pip install GitPython'.")
                return
            except git.InvalidGitRepositoryError:
                self.formatter.print_error("The current directory is not a valid Git repository.")
                return

            if len(args) < 2:
                self.formatter.print_error("Usage: /system branch <branch-name>")
                # Optional: show the current branch
                try:
                    self.formatter.print_info(f"Current branch: {repo.active_branch.name}")
                except TypeError:
                    self.formatter.print_warning("Repository has no initial commits (detached HEAD).")
                return

            branch_name = args[1]
            existing_branches = [branch.name for branch in repo.branches]

            # --- Case 1: Branch already exists, perform checkout ---
            if branch_name in existing_branches:
                if repo.active_branch.name == branch_name:
                    self.formatter.print_info(f"You are already on branch '{branch_name}'.")
                    return

                self.formatter.print_info(f"Switching to existing branch '{branch_name}'...")
                repo.git.checkout(branch_name)
                self.formatter.print_success(f"Successfully checked out branch '{branch_name}'.")

            # --- Case 2: Branch does not exist, create a new one ---
            else:
                base_branch = repo.active_branch
                current_branch_name = base_branch.name

                # Ask the user if not currently on 'main' or 'master'
                if current_branch_name not in ["main", "master"]:
                    try:
                        choice_prompt = f"Which branch should '{branch_name}' be created from? (main/master/current) [{current_branch_name}]: "
                        user_choice = await self.prompt_session.prompt_async(choice_prompt, default=current_branch_name)
                        user_choice = user_choice.lower().strip()

                        if user_choice == "main" and "main" in existing_branches:
                            base_branch = repo.branches.main
                        elif user_choice == "master" and "master" in existing_branches:
                            base_branch = repo.branches.master
                        elif user_choice in ["current", current_branch_name]:
                            base_branch = repo.active_branch
                        else:
                            self.formatter.print_error(
                                f"Invalid or unknown base branch '{user_choice}'. Action cancelled.")
                            return
                    except (KeyboardInterrupt, EOFError):
                        self.formatter.print_warning("\nBranch creation cancelled by user.")
                        return

                self.formatter.print_info(f"Creating new branch '{branch_name}' from '{base_branch.name}'...")
                new_branch = repo.create_head(branch_name, base_branch)
                new_branch.checkout()
                self.formatter.print_success(f"Branch '{branch_name}' created and checked out successfully.")


        elif sub_command == "config":
            config_preview = {k: v for k, v in self.isaa_tools.config.items() if "api_key" not in k.lower() and type(v) in (str, int, float, bool, list, dict)}
            config_json = json.dumps(config_preview, indent=2, ensure_ascii=False)
            self.formatter.print_code_block(config_json, "json")

        elif sub_command == "performance":
            perf_data = [
                ["System", f"{platform.system()} {platform.release()}"],
                ["CPU Auslastung", f"{psutil.cpu_percent()}%"],
                ["RAM Auslastung", f"{psutil.virtual_memory().percent}%"],
                ["Python Version", platform.python_version()],
                ["Prozess PID", str(os.getpid())],
            ]
            self.formatter.print_table(["Metrik", "Wert"], perf_data)

        elif sub_command == "backup":
            if not await self._ensure_git_repo():
                return  # Fehler wurde in der Helfermethode bereits ausgegeben

            self.formatter.print_info("Erstelle Backup des Workspaces...")
            # Alle Änderungen hinzufügen (neue, geänderte, gelöschte Dateien)
            await self._run_git_command(['add', '.'])

            # Commit erstellen
            commit_message = " ".join(args[1:]) if len(
                args) > 1 else f"System-Backup erstellt am {time.strftime('%Y-%m-%d %H:%M:%S')}"
            result = await self._run_git_command(['commit', '-m', commit_message])

            if result.returncode == 0:
                self.formatter.print_success("Backup erfolgreich erstellt.")
                self.formatter.print_code_block(result.stdout)
            elif "nothing to commit" in result.stdout:
                self.formatter.print_info("Keine Änderungen im Workspace seit dem letzten Backup.")
            else:
                self.formatter.print_error("Fehler beim Erstellen des Backups:")
                self.formatter.print_code_block(result.stderr)

        elif sub_command == "restore":
            if not await self._ensure_git_repo():
                return

            target = " ".join(args[1:]) if len(args) > 1 else "list"

            if target == "list":
                self.formatter.print_info("Letzte 10 Backups (Commits):")
                log_result = await self._run_git_command(
                    ['log', '--oneline', '--pretty=format:%h - %s (%cr)', '-n', '10'])
                if log_result.returncode == 0:
                    self.formatter.print_code_block(log_result.stdout)
                    self.formatter.print_info(
                        "\nUm einen Zustand wiederherzustellen, nutzen Sie: /system restore <commit_id>")
                else:
                    self.formatter.print_error("Konnte die Backup-Historie nicht abrufen.")
                    self.formatter.print_code_block(log_result.stderr)
            else:
                self.formatter.print_warning(
                    f"WARNUNG: Der Workspace wird unwiderruflich auf den Stand von '{target}' zurückgesetzt.")
                self.formatter.print_warning("Alle nicht gespeicherten Änderungen gehen verloren.")

                # Hier könnte man eine zusätzliche Bestätigung einbauen
                # z.B. /system restore <commit_id> --force

                result = await self._run_git_command(['reset', '--hard', target])
                if result.returncode == 0:
                    self.formatter.print_success(f"Workspace erfolgreich auf '{target}' zurückgesetzt.")
                else:
                    self.formatter.print_error(f"Fehler bei der Wiederherstellung zu '{target}':")
                    self.formatter.print_code_block(result.stderr)

        elif sub_command == "log":
            # Alias für /system restore list
            await self.handle_system_cmd(["restore", "list"])

        else:
            self.formatter.print_error(f"Unbekannter Systembefehl: {sub_command}")

        # --- Private Git Helper Methods ---

    async def _ensure_git_repo(self) -> bool:
        """Stellt sicher, dass der Workspace ein Git-Repository ist, und initialisiert es bei Bedarf."""
        git_dir = os.path.join(self.workspace_path, '.git')
        if os.path.isdir(git_dir):
            return True

        self.formatter.print_warning("Kein Git-Repository im Workspace gefunden. Initialisiere ein Neues...")
        result = await self._run_git_command(['init'])
        if result.returncode == 0:
            self.formatter.print_success("Git-Repository erfolgreich initialisiert.")
            return True
        else:
            self.formatter.print_error("Fehler bei der Initialisierung des Git-Repositorys:")
            self.formatter.print_code_block(result.stderr)
            return False

    async def _run_git_command(self, command_args: list[str]) -> subprocess.CompletedProcess:
        """Führt einen Git-Befehl sicher im Workspace-Verzeichnis aus."""
        # '-C' weist Git an, im angegebenen Verzeichnis zu laufen, ohne das CWD zu ändern
        base_command = ['git', '-C', str(self.workspace_path)]
        full_command = base_command + command_args

        try:
            # Führe den Prozess in einem Thread aus, um den Haupt-Event-Loop nicht zu blockieren
            return await asyncio.to_thread(
                subprocess.run,
                full_command,
                capture_output=True,
                text=True,
                check=False  # Wir prüfen den returncode manuell
            )
        except FileNotFoundError:
            # Dies passiert, wenn 'git' nicht im System-PATH ist
            self.formatter.print_error(
                "Fehler: Der 'git'-Befehl wurde nicht gefunden. Bitte stellen Sie sicher, dass Git installiert und im PATH ist.")
            # Gebe ein "leeres" Fehlerergebnis zurück
            return subprocess.CompletedProcess(args=full_command, returncode=1, stderr="Git-Befehl nicht gefunden.")
        except Exception as e:
            self.formatter.print_error(f"Ein unerwarteter Fehler ist bei der Ausführung von Git aufgetreten: {e}")
            return subprocess.CompletedProcess(args=full_command, returncode=1, stderr=str(e))

    async def handle_help_cmd(self, args: list[str]):
        """Displays a comprehensive help guide with enhanced formatting and all current commands."""
        self.formatter.log_header("ISAAC Workspace Manager - Help & Reference")

        # --- Natural Language ---
        self.formatter.print_section(
            "🗣️  Natural Language Usage",
            "Simply type your request or question and press Enter. The active agent will process it.\n"
            "The agent can use tools to perform actions like creating files, running commands, and managing projects."
        )

        # --- Command Reference ---
        self.formatter.print_section(
            "⌨️  Command Reference",
            "Commands start with a forward slash (/). They provide direct control over the CLI and its features."
        )
        command_data = [
            # Workspace & File System
            ["/workspace status", "Show an overview of the current workspace, including Git status."],
            ["/workspace cd <dir>", "Change the current workspace directory."],
            ["/workspace ls [path]", "List contents of a directory. Use -r for recursive, -a for all."],

            # Agent Management
            ["/agent list [-d]", "Show available agents. Use -d for detailed view."],
            ["/agent switch <name>", "Switch the currently active agent."],
            ["/agent status", "Display information about the active agent and session."],

            # World Model (Agent Memory)
            ["/world show", "Display the agent's current world model (short-term memory)."],
            ["/world add <k> <v>", "Add or update a key-value pair in the world model."],
            ["/world remove <key>", "Remove a key from the world model."],
            ["/world save <tag>", "Save the current world model to a file with a tag."],
            ["/world load <tag>", "Load a previously saved world model."],
            ["/world list", "List all saved world model tags."],
            ["/world clear", "Clear the current world model."],

            # Task & Process Management
            ["/tasks list [-a]", "List background tasks. Use -a to show completed tasks."],
            ["/tasks attach <id>", "Attach to a task's live output and follow its progress."],
            ["/tasks kill <id>", "Cancel a running background task by its ID."],
            ["/monitor", "Enter a full-screen interactive monitor for all background tasks."],

            # Context & Session
            ["/context save [name]", "Save the current conversation history to a file."],
            ["/context load <name>", "Load a previously saved conversation into the current session."],
            ["/context list", "Show all saved conversation contexts."],
            ["/context delete <name>", "Delete a saved context file."],
            ["/context clear", "Clear the message history for the current session."],

            # System & Git
            ["/system branch <name>", "Switch to an existing Git branch or create a new one."],
            ["/system config", "Display the current (non-sensitive) application configuration."],
            ["/system performance", "Show system CPU, memory, and process information."],
            ["/system backup [msg]", "Create a workspace backup (Git commit)."],
            ["/system restore [id]", "Restore workspace to a previous backup. Use 'list' to see history."],
            ["/system log", "Show the backup (Git commit) history for the workspace."],

            # General
            ["/clear", "Clear the terminal screen."],
            ["/help", "Display this help message."],
            ["/quit or /exit", "Exit the workspace CLI session."],
        ]
        self.formatter.print_table(["Command", "Description"], command_data)

        # --- Tips & Tricks ---
        self.formatter.print_section(
            "💡  Tips & Tricks",
            "  - Shell Commands: Start a line with '!' to execute a shell command (e.g., !pip list).\n"
            "  - Autocompletion: Press Tab to autocomplete commands, arguments, and file paths.\n"
            "  - Command History: Use the Up/Down arrow keys to cycle through your previous inputs.\n"
            "  - Interruption: Press Ctrl+C to interrupt a running agent task."
        )

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
        self._display_session_summary()
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



