# toolboxv2/flows/cli.py - The Ultimate Productive CLI Environment

import asyncio
import contextlib
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from prompt_toolkit import HTML, PromptSession
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.clipboard import InMemoryClipboard
from prompt_toolkit.completion import NestedCompleter, FuzzyCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.shortcuts import set_title
from prompt_toolkit.styles import Style as PTStyle

from toolboxv2 import App, Result, Code
from toolboxv2.tb_browser.install import detect_shell
from toolboxv2.utils.extras.Style import Style, cls
from toolboxv2.utils.extras.blobs import BlobFile

NAME = 'cli'


# =================== Data Structures ===================

@dataclass
class CLIContext:
    """Current CLI context and state"""
    mode: str = "command"  # command, task, workspace, learn, explore
    active_module: str = ""
    active_workspace: str = "default"
    quick_vars: Dict[str, Any] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)
    pinned_commands: List[str] = field(default_factory=list)
    watch_vars: List[str] = field(default_factory=list)
    last_result: Any = None
    result_count: int = 0


@dataclass
class QuickCommand:
    """Quick command definition"""
    trigger: str
    description: str
    action: Callable
    icon: str = "⚡"


@dataclass
class WorkspaceConfig:
    """Workspace configuration"""
    name: str
    modules: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    startup_commands: List[str] = field(default_factory=list)
    layout: str = "default"


# =================== Macros ===================

@dataclass
class Macro:
    """Macro definition"""
    name: str
    commands: List[str]
    description: str = ""
    variables: Dict[str, Any] = field(default_factory=dict)
    created: datetime.datetime = field(default_factory=datetime.datetime.now)
    usage_count: int = 0
    tags: List[str] = field(default_factory=list)
    conditional: bool = False
    loop_count: int = 1

@dataclass
class MacroContext:
    """Macro execution context"""
    variables: Dict[str, Any] = field(default_factory=dict)
    loop_vars: Dict[str, Any] = field(default_factory=dict)
    break_flag: bool = False
    continue_flag: bool = False
    return_value: Any = None

# =================== CLI Engine ===================

class BeastCLI:
    """The ultimate CLI environment"""

    def __init__(self, app: App):
        self.app = app
        self.context = CLIContext()
        self.session: Optional[PromptSession] = None
        self.bindings = KeyBindings()
        self.quick_commands: Dict[str, QuickCommand] = {}
        self.quick_commands_completer: Dict[str, dict|None] = {}
        self.workspaces: Dict[str, WorkspaceConfig] = {}
        self.running = True

        # Performance
        self.cache: Dict[str, Any] = {}
        self.last_update = datetime.datetime.now()

        # UI State
        self.show_status_bar = True
        self.show_help_bar = True
        self.compact_mode = False

        self.macros: Dict[str, Macro] = {}
        self.macro_context = MacroContext()
        self._load_macros()

        self._setup_quick_commands()
        self._setup_key_bindings()


    async def _handle_macro_command(self, command: str):
        """Execute saved macro with advanced features"""
        parts = command.split()
        macro_name = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        if macro_name not in self.macros:
            print(f"❌ Macro '{macro_name}' not found")
            similar = [m for m in self.macros.keys() if macro_name.lower() in m.lower()]
            if similar:
                print(f"💡 Similar macros: {', '.join(similar[:5])}")
            return

        macro = self.macros[macro_name]
        macro.usage_count += 1

        print(f"⚡ Executing macro: {macro_name}")
        if macro.description:
            print(f"📝 {macro.description}")

        # Setup macro context with arguments
        self.macro_context = MacroContext()
        for i, arg in enumerate(args):
            self.macro_context.variables[f"arg{i + 1}"] = arg
            self.macro_context.variables[f"${i + 1}"] = arg

        # Add macro variables
        self.macro_context.variables.update(macro.variables)

        # Execute commands
        await self._execute_macro_commands(macro.commands, macro.loop_count)

        # Store return value if any
        if self.macro_context.return_value:
            self.context.last_result = self.macro_context.return_value
            self.context.result_count += 1
            result_var = f"r{self.context.result_count}"
            self.context.quick_vars[result_var] = self.macro_context.return_value
            print(f"💾 Macro result saved as ${result_var}")

    async def _execute_macro_commands(self, commands: List[str], loop_count: int = 1):
        """Execute macro commands with control flow"""
        for loop_i in range(loop_count):
            self.macro_context.loop_vars['i'] = loop_i
            self.macro_context.loop_vars['loop'] = loop_i + 1

            for cmd in commands:
                if self.macro_context.break_flag:
                    break

                if self.macro_context.continue_flag:
                    self.macro_context.continue_flag = False
                    continue

                # Variable substitution in macro context
                cmd = self._substitute_macro_variables(cmd)

                # Control flow commands
                if cmd.startswith('#'):
                    continue  # Comment
                elif cmd.startswith('if '):
                    await self._handle_macro_if(cmd)
                elif cmd.startswith('for '):
                    await self._handle_macro_for(cmd)
                elif cmd.startswith('while '):
                    await self._handle_macro_while(cmd)
                elif cmd == 'break':
                    self.macro_context.break_flag = True
                elif cmd == 'continue':
                    self.macro_context.continue_flag = True
                elif cmd.startswith('return '):
                    value = cmd[7:].strip()
                    try:
                        self.macro_context.return_value = eval(value, {}, self.macro_context.variables)
                    except:
                        self.macro_context.return_value = value
                    return
                elif cmd.startswith('set '):
                    await self._handle_macro_set(cmd)
                elif cmd.startswith('echo '):
                    print(cmd[5:])
                elif cmd.startswith('sleep '):
                    import asyncio
                    await asyncio.sleep(float(cmd[6:]))
                else:
                    # Regular command execution
                    await self._process_command(cmd)

            if self.macro_context.break_flag:
                break

    def _substitute_variables(self, command: str) -> str:
        """Substitute $variables in command - unified with macro variables"""
        # CLI quick variables
        for var_name, var_value in self.context.quick_vars.items():
            command = command.replace(f"${var_name}", str(var_value))

        # If we're in a macro context, also substitute macro variables
        if hasattr(self, 'macro_context') and self.macro_context:
            # Macro context variables (highest priority)
            for var_name, var_value in self.macro_context.variables.items():
                command = command.replace(f"${var_name}", str(var_value))

            # Loop variables (highest priority)
            for var_name, var_value in self.macro_context.loop_vars.items():
                command = command.replace(f"${var_name}", str(var_value))

        return command

    def _substitute_macro_variables(self, command: str) -> str:
        """Substitute macro variables - unified with CLI variables"""
        # Start with CLI variables as base
        for var_name, var_value in self.context.quick_vars.items():
            command = command.replace(f"${var_name}", str(var_value))

        # Macro context variables (override CLI variables)
        for var_name, var_value in self.macro_context.variables.items():
            command = command.replace(f"${var_name}", str(var_value))

        # Loop variables (highest priority)
        for var_name, var_value in self.macro_context.loop_vars.items():
            command = command.replace(f"${var_name}", str(var_value))

        return command

    async def _handle_macro_set(self, cmd: str):
        """Handle variable assignment in macro"""
        # set var_name = expression
        parts = cmd[4:].split(' = ', 1)
        if len(parts) == 2:
            var_name, expression = parts
            var_name = var_name.strip()
            try:
                value = eval(expression, {}, self.macro_context.variables)
            except:
                value = expression
            self.macro_context.variables[var_name] = value

    async def _macro_manager(self, args=""):
        """Macro management interface"""
        if not args:
            await self._show_macro_list()
            return

        parts = args.split()
        action = parts[0]

        if action == "create":
            await self._create_macro_interactive()
        elif action == "edit":
            macro_name = parts[1] if len(parts) > 1 else ""
            await self._edit_macro(macro_name)
        elif action == "delete":
            macro_name = parts[1] if len(parts) > 1 else ""
            await self._delete_macro(macro_name)
        elif action == "export":
            await self._export_macros()
        elif action == "import":
            file_path = parts[1] if len(parts) > 1 else ""
            await self._import_macros(file_path)
        elif action == "help":
            args = parts[1] if len(parts) > 1 else ""
            await self._macro_help(args)
        else:
            print("❌ Unknown macro action")
            print("💡 Available: create, edit, delete, export, import")

    async def _show_macro_list(self):
        """Show all available macros"""
        if not self.macros:
            print("📝 No macros defined")
            return

        print("\n⚡ Available Macros:")
        print(f"{'─' * 80}")

        for name, macro in sorted(self.macros.items()):
            tags_str = f" [{', '.join(macro.tags)}]" if macro.tags else ""
            usage_str = f" (used {macro.usage_count}x)" if macro.usage_count > 0 else ""

            print(f"  📌 {name}{tags_str}{usage_str}")
            if macro.description:
                print(f"     {macro.description}")
            print(f"     Commands: {len(macro.commands)}")
            print()

    async def _create_macro_interactive(self):
        """Interactive macro creation"""
        try:
            name = await self.session.prompt_async("📝 Macro name: ")
            if not name or name in self.macros:
                print("❌ Invalid or existing name")
                return

            description = await self.session.prompt_async("📄 Description (optional): ")

            print("📝 Enter commands (empty line to finish):")
            commands = []
            while True:
                cmd = await self.session.prompt_async(f"  {len(commands) + 1}> ")
                if not cmd.strip():
                    break
                commands.append(cmd.strip())

            if not commands:
                print("❌ No commands entered")
                return

            # Optional settings
            tags_input = await self.session.prompt_async("🏷️  Tags (comma-separated): ")
            tags = [t.strip() for t in tags_input.split(',')] if tags_input else []

            loop_input = await self.session.prompt_async("🔄 Loop count (default 1): ")
            loop_count = int(loop_input) if loop_input.isdigit() else 1

            # Create macro
            macro = Macro(
                name=name,
                commands=commands,
                description=description,
                tags=tags,
                loop_count=loop_count
            )

            self.macros[name] = macro
            self._save_macros()

            print(f"✅ Macro '{name}' created with {len(commands)} commands")

        except (KeyboardInterrupt, EOFError):
            print("\n❌ Macro creation cancelled")

    async def _record_macro(self, args=""):
        """Record macro from live commands"""
        if not args:
            print("❌ Usage: :record <macro_name>")
            return

        macro_name = args.strip()
        if macro_name in self.macros:
            print(f"❌ Macro '{macro_name}' already exists")
            return

        print(f"🎬 Recording macro '{macro_name}' - type 'stop_recording' to finish")

        recorded_commands = []
        original_history_len = len(self.context.history)

        # Set recording flag
        self.context.recording_macro = macro_name

        try:
            while True:
                cmd = await self._get_input()
                if cmd == "stop_recording":
                    break

                # Process command normally
                await self._process_command(cmd)
                recorded_commands.append(cmd)

        except (KeyboardInterrupt, EOFError):
            print("\n❌ Recording cancelled")
            return
        finally:
            self.context.recording_macro = None

        if recorded_commands:
            macro = Macro(
                name=macro_name,
                commands=recorded_commands,
                description=f"Recorded macro with {len(recorded_commands)} commands"
            )

            self.macros[macro_name] = macro
            self._save_macros()

            print(f"✅ Recorded macro '{macro_name}' with {len(recorded_commands)} commands")
        else:
            print("❌ No commands recorded")

    def _save_macros(self):
        """Save macros to file"""
        try:
            with BlobFile("cli/macros.json", key=Code.DK()(), mode="w") as f:
                macro_data = {}
                for name, macro in self.macros.items():
                    macro_data[name] = {
                        'commands': macro.commands,
                        'description': macro.description,
                        'variables': macro.variables,
                        'created': macro.created.isoformat(),
                        'usage_count': macro.usage_count,
                        'tags': macro.tags,
                        'loop_count': macro.loop_count
                    }
                f.write_json(macro_data)
        except Exception as e:
            print(f"❌ Failed to save macros: {e}")

    def _load_macros(self):
        """Load macros from file"""
        try:
            with BlobFile("cli/macros.json", key=Code.DK()(), mode="r") as f:
                if f.exists():
                    macro_data = f.read_json()
                    for name, data in macro_data.items():
                        self.macros[name] = Macro(
                            name=name,
                            commands=data['commands'],
                            description=data.get('description', ''),
                            variables=data.get('variables', {}),
                            created=datetime.datetime.fromisoformat(
                                data.get('created', datetime.datetime.now().isoformat())),
                            usage_count=data.get('usage_count', 0),
                            tags=data.get('tags', []),
                            loop_count=data.get('loop_count', 1)
                        )
        except Exception as e:
            print(f"⚠️ Failed to load macros: {e}")

    def _setup_quick_commands(self):
        """Setup quick command shortcuts"""
        self.quick_commands = {
            ":q": QuickCommand(":q", "Quick exit", self._quick_exit, "🚪"),
            ":h": QuickCommand(":h", "Quick help", self._quick_help, "❓"),
            ":m": QuickCommand(":m", "Module browser", self._module_browser, "📦"),
            ":t": QuickCommand(":t", "Task manager", self._task_manager, "✅"),
            ":w": QuickCommand(":w", "Workspace switcher", self._workspace_switch, "🏗️"),
            ":s": QuickCommand(":s", "Quick search", self._quick_search, "🔍"),
            ":r": QuickCommand(":r", "Recent commands", self._recent_commands, "⏱️"),
            ":v": QuickCommand(":v", "View variables", self._view_vars, "📊"),
            ":c": QuickCommand(":c", "Clear screen", self._clear_screen, "🧹"),
            ":!": QuickCommand(":!", "System shell", self._system_shell, "💻"),

            ":macro": QuickCommand(":macro", "Macro manager", self._macro_manager, "⚡"),
            ":record": QuickCommand(":record", "Record macro", self._record_macro, "🎬"),
            ":play": QuickCommand(":play", "Play macro", self._play_macro, "▶️"),
        }


    def _setup_key_bindings(self):
        """Setup keyboard shortcuts"""

        # Quick command palette (Ctrl+P)
        @self.bindings.add('c-p')
        def quick_palette(event):
            run_in_terminal(lambda: self._show_command_palette())

        # Quick execute (Ctrl+_)
        @self.bindings.add('c-_')
        def quick_execute(event):
            buff = event.app.current_buffer
            res = self._execute_command_and_return(buff.text)
            if res:
                res.print(full_data=True)
            buff.text = ""

        # History search (Ctrl+R)
        @self.bindings.add('c-r')
        def history_search(event):
            run_in_terminal(lambda: self._search_history())

        # Module quick switch (Ctrl+Q)
        @self.bindings.add('c-q')
        async def module_switch(event):
            await run_in_terminal(self._quick_module_select)

        # Toggle compact mode (Ctrl+T)
        @self.bindings.add('c-t')
        def toggle_compact(event):
            self.compact_mode = not self.compact_mode
            print(f"\n{'Compact' if self.compact_mode else 'Full'} mode activated")

        # Smart help (Alt+H)
        @self.bindings.add('escape', 'h')
        def smart_help(event):
            buff = event.app.current_buffer
            run_in_terminal(lambda: self._smart_help(buff.text))

        # Quick save result (Ctrl+S)
        @self.bindings.add('c-s')
        def save_result(event):
            if self.context.last_result:
                var_name = f"r{self.context.result_count}"
                self.context.quick_vars[var_name] = self.context.last_result
                print(f"\n💾 Saved as ${var_name}")

        # Variable substitution helper (Alt+V)
        @self.bindings.add('escape', 'v')
        def var_helper(event):
            run_in_terminal(lambda: self._show_variables())

        # Workspace manager (Alt+W)
        @self.bindings.add('escape', 'w')
        def workspace_manager(event):
            run_in_terminal(lambda: self._workspace_manager())

    async def run(self):
        """Main CLI loop"""
        # Setup
        self._load_workspaces()
        await self._init_session()

        # Welcome
        self._show_welcome()

        # Main loop
        while self.running:
            try:
                # Get input
                command = await self._get_input()

                if not command or not command.strip():
                    continue

                # Process command
                await self._process_command(command)

            except KeyboardInterrupt:
                print("\n⚠️  Use :q to exit or Ctrl+D")
                continue
            except EOFError:
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                if self.app.debug:
                    import traceback
                    traceback.print_exc()

        # Cleanup
        await self._cleanup()

    async def _init_session(self):
        """Initialize prompt session"""
        # Build autocomplete
        completer = self._build_completer()

        # History
        history_file = Path(self.app.data_dir) / f"{self.app.id}-beast-cli.history"
        history = FileHistory(str(history_file))

        # Style
        style = self._build_style()

        # Create session
        self.session = PromptSession(
            history=history,
            auto_suggest=AutoSuggestFromHistory(),
            completer=FuzzyCompleter(completer),
            complete_while_typing=True,
            clipboard=InMemoryClipboard(),
            mouse_support=True,
            color_depth=ColorDepth.TRUE_COLOR,
            key_bindings=self.bindings,
            rprompt=self._get_rprompt,
            bottom_toolbar=self._get_bottom_toolbar,
            style=style,
            refresh_interval=2,
        )

    def _build_completer(self) -> NestedCompleter:
        """Build smart autocompleter"""
        completion_dict = {}

        self.quick_commands_completer ={
            ":macro": {
                "create": None,
                "edit": None,
                "delete": None,
                "export": None,
                "import": None,
                "help": None,
            },
            ":record": None,
            ":play": {
                "macro_name": {k: None for k in self.macros.keys()},
            },
        }
        # Quick commands
        for cmd, qc in self.quick_commands.items():
            completion_dict[cmd] = self.quick_commands_completer.get(cmd)

        # Modules and functions
        for module_name, module_data in self.app.functions.items():
            func_dict = {}

            if isinstance(module_data, dict):
                for func_name, func_data in module_data.items():
                    if isinstance(func_data, dict):
                        # Add function with parameter hints
                        params = func_data.get('params', [])
                        if params:
                            param_dict = {p: None for p in params if p not in ['app', 'self']}
                            func_dict[func_name] = param_dict
                        else:
                            func_dict[func_name] = None

            completion_dict[module_name] = func_dict

        # System commands
        if self.app.system_flag == "Windows":
            # Add common Windows commands
            for cmd in ['dir', 'cd', 'mkdir', 'del', 'copy', 'move', 'type', 'cls']:
                completion_dict[cmd] = None
        else:
            # Add common Unix commands
            for cmd in ['ls', 'cd', 'mkdir', 'rm', 'cp', 'mv', 'cat', 'clear', 'grep']:
                completion_dict[cmd] = None

        # Variables (with $ prefix)
        for var_name in self.context.quick_vars.keys():
            completion_dict[f"${var_name}"] = None

        return NestedCompleter.from_nested_dict(completion_dict)

    def _build_style(self) -> PTStyle:
        """Build prompt style"""
        return PTStyle.from_dict({
            'prompt': '#00d9ff bold',
            'rprompt': '#666666',
            'bottom-toolbar': 'bg:#1c1c1c #00d9ff',
            'completion-menu': 'bg:#1c1c1c #ffffff',
            'completion-menu.completion': 'bg:#1c1c1c #ffffff',
            'completion-menu.completion.current': 'bg:#00d9ff #000000 bold',
        })

    def _get_rprompt(self) -> HTML:
        """Right prompt with context info"""
        parts = []

        # Mode indicator
        mode_icons = {
            'command': '⚡',
            'task': '✅',
            'workspace': '🏗️',
            'learn': '📚',
            'explore': '🔍'
        }
        mode_icon = mode_icons.get(self.context.mode, '⚡')
        parts.append(f'<b>{mode_icon} {self.context.mode}</b>')

        # Active module
        if self.context.active_module:
            parts.append(f'<b>📦 {self.context.active_module}</b>')

        # Workspace
        if self.context.active_workspace != "default":
            parts.append(f'<b>🏗️ {self.context.active_workspace}</b>')

        # Time
        current_time = datetime.datetime.now().strftime("%H:%M")
        parts.append(f'<b>🕐 {current_time}  </b>')

        return HTML(' │ '.join(parts))

    def _get_bottom_toolbar(self) -> HTML:
        """Bottom toolbar with shortcuts"""
        if not self.show_help_bar:
            return HTML('')

        if self.compact_mode:
            shortcuts = [
                '<b>^P</b> Palette',
                '<b>^R</b> History',
                '<b>^M</b> Modules',
                '<b>:h</b> Help'
            ]
        else:
            shortcuts = [
                '<b><style bg="ansired">Ctrl+P</style></b> Command Palette',
                '<b><style bg="ansigreen">Ctrl+R</style></b> History Search',
                '<b><style bg="ansiblue">Ctrl+M</style></b> Module Switch',
                '<b><style bg="ansiyellow">Alt+H</style></b> Smart Help',
                '<b><style bg="ansimagenta">:h</style></b> Quick Help',
                '<b><style bg="ansicyan">:q</style></b> Exit'
            ]

        return HTML(' │ '.join(shortcuts))

    def _get_prompt_message(self) -> str:
        """Get dynamic prompt message"""
        # Base prompt with mode indicator
        mode_colors = {
            'command': 'ansibrightcyan',
            'task': 'ansibrightgreen',
            'workspace': 'ansibrightblue',
            'learn': 'ansibrightmagenta',
            'explore': 'ansibrightblue'
        }

        color = mode_colors.get(self.context.mode, '\033[96m')

        # Build prompt
        parts = []

        if self.context.active_workspace != "default":
            parts.append(f"({self.context.active_workspace})")

        if self.context.active_module:
            parts.append(f"[{self.context.active_module}]")

        parts.append(f"<{color}>❯</{color}> ")

        return HTML(''.join(parts))

    async def _get_input(self) -> str:
        """Get user input"""
        try:
            prompt_msg = self._get_prompt_message()

            # Update completer if cache is stale
            if (datetime.datetime.now() - self.last_update).seconds > 60:
                self.session.completer = FuzzyCompleter(self._build_completer())
                self.last_update = datetime.datetime.now()

            return await self.session.prompt_async(prompt_msg)

        except KeyboardInterrupt:
            raise
        except EOFError:
            raise
        except Exception as e:
            print(f"Input error: {e}")
            return ""

    async def _process_command(self, command: str):
        """Process and execute command"""
        command = command.strip()

        if not command:
            return

        # Add to history
        self.context.history.append(command)

        # Variable substitution
        command = self._substitute_variables(command)

        # Quick commands
        if command in self.quick_commands:
            await self.quick_commands[command].action()
            return

        print(f"Processing: {command}")

        # Command type detection
        if command.startswith(':'):
            await self._handle_special_command(command)
        elif command.startswith('!'):
            await self._handle_shell_command(command[1:])
        elif command.startswith('$'):
            await self._handle_variable_command(command)
        elif command.startswith('@'):
            await self._handle_macro_command(command[1:])
        elif ' = ' in command:
            await self._handle_assignment(command)
        else:
            if command.split()[0] in self.macros:
                await self._handle_macro_command(command)
                return
            await self._handle_normal_command(command)

    async def _handle_special_command(self, command: str):
        """Handle special commands starting with :"""
        parts = command.split(maxsplit=1)
        cmd = parts[0]
        args = parts[1] if len(parts) > 1 else ""

        if cmd in self.quick_commands:
            await self.quick_commands[cmd].action(args)
        else:
            print(f"❌ Unknown special command: {cmd}")
            print(f"💡 Try :h for help")

    async def _handle_shell_command(self, command: str):
        """Execute system shell command"""
        print(f"\n🖥️  Executing: {command}")
        import subprocess
        import signal
        import sys

        # Globale Variable für den Prozess
        current_process = None

        def signal_handler(sig, frame):
            """Handler für Ctrl+C - stoppt nur den Subprozess"""
            if current_process:
                print("\n[Stoppe Subprozess...]")
                current_process.terminate()
                try:
                    current_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    current_process.kill()
            # Hauptskript läuft weiter!

        # Signal Handler registrieren
        signal.signal(signal.SIGINT, signal_handler)

        try:
            # Prozess starten OHNE capture_output
            a,b = detect_shell()
            current_process = subprocess.Popen(
                [a, b, command],
                shell=True,
                # Keine Umleitung - I/O geht direkt durch
                stdin=sys.stdin,
                stdout=sys.stdout,
                stderr=sys.stderr
            )

            # Auf Prozessende warten
            returncode = current_process.wait()

            print(f"\nProzess beendet mit Code: {returncode}")

        except subprocess.TimeoutExpired:
            print("⏱️  Command timed out")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            current_process = None

    async def _handle_variable_command(self, command: str):
        """Handle variable operations - unified with macro system"""
        # $var = value
        if ' = ' in command:
            var_name, value = command.split(' = ', 1)
            var_name = var_name[1:]  # Remove $
            try:
                # Try to evaluate value with access to all variables
                eval_context = {}
                eval_context.update(self.context.quick_vars)
                if hasattr(self, 'macro_context') and self.macro_context:
                    eval_context.update(self.macro_context.variables)
                value = eval(value, {}, eval_context)
            except:
                pass  # Keep as string

            # Store in both CLI and macro context if available
            self.context.quick_vars[var_name] = value
            if hasattr(self, 'macro_context') and self.macro_context:
                self.macro_context.variables[var_name] = value

            print(f"💾 ${var_name} = {value}")
        else:
            # Show variable - check both contexts
            var_name = command[1:]
            value = None

            # Check macro context first (higher priority)
            if hasattr(self, 'macro_context') and self.macro_context:
                if var_name in self.macro_context.loop_vars:
                    value = self.macro_context.loop_vars[var_name]
                    print(f"${var_name} = {value} (loop variable)")
                    return
                elif var_name in self.macro_context.variables:
                    value = self.macro_context.variables[var_name]
                    print(f"${var_name} = {value} (macro variable)")
                    return

            # Check CLI variables
            if var_name in self.context.quick_vars:
                value = self.context.quick_vars[var_name]
                print(f"${var_name} = {value} (CLI variable)")
            else:
                print(f"❌ Variable ${var_name} not found")

    async def _handle_assignment(self, command: str):
        """Handle variable assignment"""
        var_name, expression = command.split(' = ', 1)
        var_name = var_name.strip()

        # Execute expression and store result
        result = await self._execute_expression(expression)
        self.context.quick_vars[var_name] = result

        print(f"💾 {var_name} = {result}")

    async def _handle_normal_command(self, command: str):
        """Handle normal toolbox commands"""
        # Parse command
        parts = command.split()

        if len(parts) < 1:
            return

        module_name = parts[0]
        function_name = parts[1] if len(parts) > 1 else None
        args = parts[2:] if len(parts) > 2 else []

        # Check if it's a module
        if module_name not in self.app.functions:
            # Suggest similar
            similar = [m for m in self.app.functions.keys()
                       if module_name.lower() in m.lower()]
            if similar:
                print(f"💡 Did you mean: {', '.join(similar[:5])}")
            else:
                await self._handle_shell_command(command)
            return

        # If no function, show functions
        if not function_name:
            self._show_module_functions(module_name)
            return

        # Check function exists
        module_data = self.app.functions.get(module_name, {})
        if function_name not in module_data:
            print(f"❌ Function '{function_name}' not found in {module_name}")

            # Show available
            funcs = [f for f in module_data.keys()
                     if isinstance(module_data[f], dict)]
            if funcs:
                print(f"💡 Available: {', '.join(funcs[:10])}")
            return

        # Execute
        await self._execute_function(module_name, function_name, args)

    async def _execute_expression(self, expression: str) -> Any:
        """Execute an expression and return result"""
        # Try to evaluate in context
        try:
            return eval(expression, {}, self.context.quick_vars)
        except:
            # Try as command
            return await self._execute_command_and_return(expression)

    async def _execute_command_and_return(self, command: str) -> Any:
        """Execute command and return result"""
        parts = command.split()
        if len(parts) >= 2:
            module_name, function_name = parts[0], parts[1]
            args = parts[2:]

            result = await self.app.a_run_any(
                (module_name, function_name),
                args_=args,
                get_results=True,
                tb_run_with_specification='app'
            )

            if asyncio.iscoroutine(result):
                result = await result

            return result

        return None

    async def _execute_function(self, module_name: str, function_name: str, args: List[str]):
        """Execute a toolbox function"""
        print(f"\n⚡ Executing: {module_name}.{function_name}")

        try:
            # Execute
            result = await self.app.a_run_any(
                (module_name, function_name),
                args_=args,
                get_results=True,
                tb_run_with_specification='app'
            )

            # Handle coroutine
            if asyncio.iscoroutine(result):
                result = await result

            # Store result
            self.context.last_result = result
            self.context.result_count += 1
            result_var = f"r{self.context.result_count}"
            self.context.quick_vars[result_var] = result

            # Display result
            print(f"\n{'─' * 60}")
            print(f"Result (saved as ${result_var}):")
            print(f"{'─' * 60}")

            if hasattr(result, 'print'):
                result.print(full_data=True)
            elif isinstance(result, dict):
                import json
                print(json.dumps(result, indent=2, default=str))
            else:
                print(result)

            print(f"{'─' * 60}\n")

        except Exception as e:
            print(f"\n❌ Execution failed: {e}")
            if self.app.debug:
                import traceback
                traceback.print_exc()

    def _show_module_functions(self, module_name: str):
        """Show functions in a module"""
        module_data = self.app.functions.get(module_name, {})

        print(f"\n📦 Module: {module_name}")
        print(f"{'─' * 60}")

        functions = []
        for func_name, func_data in module_data.items():
            if isinstance(func_data, dict) and 'func' in func_data:
                functions.append(func_name)

        if not functions:
            print("No functions available")
        else:
            for i, func in enumerate(functions, 1):
                print(f"  {i}. {func}")

        print(f"{'─' * 60}\n")

    # =================== Quick Commands Implementation ===================

    async def _quick_exit(self, args=""):
        """Quick exit"""
        self.running = False
        print("\n👋 Goodbye!")

    async def _quick_help(self, args=""):
        """Show quick help"""
        print("\n╔════════════════════════════════════════════════════════════════╗")
        print("║                    🚀       CLI - Quick Help                   ║")
        print("╠════════════════════════════════════════════════════════════════╣")
        print("║                                                                ║")
        print("║  Quick Commands:                                               ║")
        print("║    :q              Exit                                        ║")
        print("║    :h              This help                                   ║")
        print("║    :m              Module browser                              ║")
        print("║    :t              Task manager                                ║")
        print("║    :w              Workspace switcher                          ║")
        print("║    :s <query>      Quick search                                ║")
        print("║    :r              Recent commands                             ║")
        print("║    :v              View variables                              ║")
        print("║    :c              Clear screen                                ║")
        print("║    :! <cmd>        System shell                                ║")
        print("║    :macro <name>   Macro manager                               ║")
        print("║    :record <name>  Record macro                                ║")
        print("║    :play <name>     Play macro                                 ║")
        print("║                                                                ║")
        print("║  Keyboard Shortcuts:                                           ║")
        print("║    Ctrl+P          Command palette                             ║")
        print("║    Ctrl+R          History search                              ║")
        print("║    Ctrl+M          Quick module switch                         ║")
        print("║    Ctrl+S          Save last result                            ║")
        print("║    Ctrl+T          Toggle compact mode                         ║")
        print("║    Alt+H           Smart help for current input                ║")
        print("║    Alt+V           Show variables                              ║")
        print("║    Alt+W           Workspace manager                           ║")
        print("║                                                                ║")
        print("║  Command Syntax:                                               ║")
        print("║    module function [args...]     Execute function              ║")
        print("║    !command                      System command                ║")
        print("║    $var = value                  Set variable                  ║")
        print("║    $var                          Get variable                  ║")
        print("║    var = expression              Evaluate and store            ║")
        print("║    @macro                        Execute macro                 ║")
        print("║                                                                ║")
        print("║  Variables:                                                    ║")
        print("║    $r1, $r2, ...   Auto-saved results                          ║")
        print("║    $<name>         Custom variables                            ║")
        print("║                                                                ║")
        print("╚════════════════════════════════════════════════════════════════╝\n")

    async def _module_browser(self, args=""):
        """Interactive module browser"""
        # TODO: Implement with prompt_toolkit application
        modules = list(self.app.functions.keys())
        modules.sort()

        print("\n📦 Available Modules:")
        print(f"{'─' * 60}")

        for i, mod in enumerate(modules, 1):
            try:
                mod_obj = self.app.get_mod(mod)
                version = getattr(mod_obj, 'version', 'unknown')
            except:
                version = 'unknown'

            print(f"  {i:3d}. {mod:<30} v{version}")

        print(f"{'─' * 60}\n")

    async def _task_manager(self, args=""):
        """Task manager integration"""
        sm = self.app.get_mod("SchedulerManager")
        if sm:
            print(sm.get_tasks_table())
        else:
            print("⚠️  Task manager not available")

    async def _workspace_switch(self, args=""):
        """Switch workspace"""
        if args:
            self.context.active_workspace = args
            print(f"🏗️  Switched to workspace: {args}")
        else:
            # Show available workspaces
            print("\n🏗️  Available Workspaces:")
            for name in self.workspaces.keys():
                current = " (current)" if name == self.context.active_workspace else ""
                print(f"  • {name}{current}")
            print()

    async def _quick_search(self, args=""):
        """Quick search across modules and functions"""
        if not args:
            args = input("🔍 Search query: ")

        query = args.lower()
        results = []

        # Search modules
        for module in self.app.functions.keys():
            if query in module.lower():
                results.append(('module', module, None))

        # Search functions
        for module, funcs in self.app.functions.items():
            if isinstance(funcs, dict):
                for func in funcs.keys():
                    if query in func.lower():
                        results.append(('function', module, func))

        if not results:
            print(f"❌ No results for '{query}'")
        else:
            print(f"\n🔍 Search Results for '{query}':")
            print(f"{'─' * 60}")

            for rtype, module, func in results[:20]:
                if rtype == 'module':
                    print(f"  📦 {module}")
                else:
                    print(f"  ⚡ {module}.{func}")

            if len(results) > 20:
                print(f"\n  ... and {len(results) - 20} more results")

            print(f"{'─' * 60}\n")

    async def _recent_commands(self, args=""):
        """Show recent commands"""
        print("\n⏱️  Recent Commands:")
        print(f"{'─' * 60}")

        recent = self.context.history[-20:][::-1]  # Last 20, reversed

        for i, cmd in enumerate(recent, 1):
            print(f"  {i:2d}. {cmd}")

        print(f"{'─' * 60}\n")

    async def _view_vars(self, args=""):
        """View all variables - unified display"""
        print("\n📊 Variable Overview:")
        print(f"{'─' * 80}")

        all_vars = {}

        # Collect CLI variables
        for name, value in self.context.quick_vars.items():
            all_vars[name] = {
                'value': value,
                'type': 'CLI',
                'scope': 'Global'
            }

        # Collect macro variables if in macro context
        if hasattr(self, 'macro_context') and self.macro_context:
            for name, value in self.macro_context.variables.items():
                if name in all_vars:
                    all_vars[name]['type'] = 'CLI+Macro'
                else:
                    all_vars[name] = {
                        'value': value,
                        'type': 'Macro',
                        'scope': 'Macro'
                    }

            # Loop variables (temporary)
            for name, value in self.macro_context.loop_vars.items():
                all_vars[name] = {
                    'value': value,
                    'type': 'Loop',
                    'scope': 'Temporary'
                }

        if not all_vars:
            print("  No variables defined")
        else:
            # Display in organized format
            print(f"{'Variable':<20} {'Value':<30} {'Type':<12} {'Scope'}")
            print(f"{'─' * 20} {'─' * 30} {'─' * 12} {'─' * 10}")

            for name, info in sorted(all_vars.items()):
                value_str = str(info['value'])
                if len(value_str) > 28:
                    value_str = value_str[:25] + "..."

                type_color = {
                    'CLI': '🔵',
                    'Macro': '🟡',
                    'CLI+Macro': '🟢',
                    'Loop': '🔴'
                }.get(info['type'], '⚪')

                print(f"${name:<19} {value_str:<30} {type_color}{info['type']:<11} {info['scope']}")

        print(f"{'─' * 80}")
        print("🔵 CLI  🟡 Macro  🟢 Unified  🔴 Loop (temporary)")
        print()

    async def _clear_screen(self, args=""):
        """Clear screen"""
        cls()

    async def _system_shell(self, args=""):
        """Open system shell"""
        if args:
            await self._handle_shell_command(args)
        else:
            print("💡 Use !<command> or :! <command> to execute shell commands")

    # =================== Advanced Features ===================

    def _show_command_palette(self):
        """Show command palette"""
        print("\n🎨 Command Palette:")
        print(f"{'─' * 60}")

        for cmd, qc in self.quick_commands.items():
            print(f"  {qc.icon} {cmd:<10} - {qc.description}")

        print(f"{'─' * 60}\n")

    async def _search_history(self):
        """Search command history"""
        from prompt_toolkit import prompt

        try:
            query = await self.session.prompt_async("🔍 Search history: ")

            if not query:
                return

            matches = [cmd for cmd in self.context.history if query.lower() in cmd.lower()]

            if not matches:
                print("❌ No matches found")
            else:
                print(f"\n📜 History matches for '{query}':")
                for i, cmd in enumerate(matches[-10:], 1):
                    print(f"  {i}. {cmd}")
                print()
        except (KeyboardInterrupt, EOFError):
            print("\n❌ Cancelled")
    def _quick_module_select(self):
        """Quick module selection"""
        modules = list(self.app.functions.keys())

        # Show first 10
        print("\n📦 Quick Module Select:")
        for i, mod in enumerate(modules[:10], 1):
            print(f"  {i}. {mod}")

        # Use prompt_toolkit for input instead of input()
        from prompt_toolkit import prompt

        try:
            choice = prompt("❯ Select (number or name): ")

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(modules):
                    self.context.active_module = modules[idx]
                    print(f"✓ Active module: {self.context.active_module}")
            except ValueError:
                if choice in modules:
                    self.context.active_module = choice
                    print(f"✓ Active module: {self.context.active_module}")
        except (KeyboardInterrupt, EOFError):
            print("\n❌ Cancelled")

    def _smart_help(self, text: str):
        """Context-aware help"""
        if not text:
            print("💡 Type a command to get context-specific help")
            return

        parts = text.split()

        if len(parts) >= 1:
            module = parts[0]

            if module in self.app.functions:
                if len(parts) >= 2:
                    # Help for specific function
                    func = parts[1]
                    self._show_function_help(module, func)
                else:
                    # Help for module
                    self._show_module_functions(module)

    def _show_function_help(self, module: str, function: str):
        """Show help for specific function"""
        func_data = self.app.get_function((module, function), metadata=True)

        if not func_data:
            print(f"❌ Function not found: {module}.{function}")
            return

        print(f"\n📖 Help: {module}.{function}")
        print(f"{'─' * 60}")

        # Show parameters
        params = func_data.get('params', [])
        if params:
            print(f"\nParameters:")
            for param in params:
                if param not in ['app', 'self']:
                    print(f"  • {param}")

        # Show docstring if available
        func_obj = func_data.get('func')
        if func_obj and hasattr(func_obj, '__doc__') and func_obj.__doc__:
            print(f"\nDescription:")
            print(func_obj.__doc__)

        print(f"{'─' * 60}\n")

    def _show_variables(self):
        """Show variables in a formatted way"""
        asyncio.create_task(self._view_vars())

    def _workspace_manager(self):
        """Workspace management interface"""
        print("\n🏗️  Workspace Manager:")
        print(f"{'─' * 60}")
        print(f"  Current: {self.context.active_workspace}")
        print(f"\n  Available workspaces:")

        for name, config in self.workspaces.items():
            print(f"    • {name}")
            print(f"      Modules: {', '.join(config.modules[:5])}")

        print(f"{'─' * 60}\n")

    def _load_workspaces(self):
        """Load workspace configurations"""
        # Default workspace
        self.workspaces['default'] = WorkspaceConfig(
            name='default',
            modules=[],
            env_vars={},
            startup_commands=[]
        )

        # TODO: Load from config file

    def _show_welcome(self):
        """Show welcome message"""
        cls()

        print("╔════════════════════════════════════════════════════════════════╗")
        print("║              🚀 Beast CLI - Ultimate Productivity              ║")
        print("╠════════════════════════════════════════════════════════════════╣")
        print("║                                                                ║")
        print(f"║  Welcome back, {self.app.get_username():<46}  ║")
        print(f"║  Instance: {self.app.id:<50}  ║")
        print("║                                                                ║")
        print("║  Type :h for help or Ctrl+P for command palette                ║")
        print("║                                                                ║")
        print("╚════════════════════════════════════════════════════════════════╝\n")

    async def _cleanup(self):
        """Cleanup before exit"""
        # Save history, state, etc.
        print("\n🧹 Cleaning up...")

        # Save context
        with BlobFile("cli/context.c", key=Code.DK()(), mode="w") as f:
            f.write_json(self.context.__dict__)

        print("✓ Done")

    async def _edit_macro(self, macro_name: str):
        """Edit existing macro"""
        if not macro_name:
            # Show list and let user select
            if not self.macros:
                print("❌ No macros to edit")
                return

            print("\n📝 Select macro to edit:")
            macro_list = list(self.macros.keys())
            for i, name in enumerate(macro_list, 1):
                print(f"  {i}. {name}")

            from prompt_toolkit import prompt
            try:
                choice = await self.session.prompt_async("❯ Select macro (number or name): ")
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(macro_list):
                        macro_name = macro_list[idx]
                    else:
                        print("❌ Invalid selection")
                        return
                except ValueError:
                    if choice in self.macros:
                        macro_name = choice
                    else:
                        print("❌ Macro not found")
                        return
            except (KeyboardInterrupt, EOFError):
                print("\n❌ Cancelled")
                return

        if macro_name not in self.macros:
            print(f"❌ Macro '{macro_name}' not found")
            return

        macro = self.macros[macro_name]

        try:
            print(f"\n✏️  Editing macro: {macro_name}")
            print(f"Current description: {macro.description}")
            print(f"Current commands ({len(macro.commands)}):")
            for i, cmd in enumerate(macro.commands, 1):
                print(f"  {i}. {cmd}")
            print()

            # Edit description
            new_desc = await self.session.prompt_async(f"📄 Description [{macro.description}]: ")
            if new_desc.strip():
                macro.description = new_desc.strip()

            # Edit commands
            print(f"\n📝 Current commands ({len(macro.commands)}):")
            for i, cmd in enumerate(macro.commands, 1):
                print(f"  {i}. {cmd}")

            print("\n📝 Add new commands (empty line to finish):")
            new_commands = list(macro.commands)  # Copy existing
            while True:
                cmd = await self.session.prompt_async(f"  {len(new_commands)+1}> ")
                if not cmd.strip():
                    break
                new_commands.append(cmd.strip())

            # Edit tags
            current_tags = ', '.join(macro.tags)
            new_tags_input = await self.session.prompt_async(f"🏷️  Tags [{current_tags}]: ")
            if new_tags_input.strip():
                macro.tags = [t.strip() for t in new_tags_input.split(',')]

            # Edit loop count
            new_loop = await self.session.prompt_async(f"🔄 Loop count [{macro.loop_count}]: ")
            if new_loop.strip().isdigit():
                macro.loop_count = int(new_loop.strip())

            # Update macro
            macro.commands = new_commands
            self._save_macros()

            print(f"✅ Macro '{macro_name}' updated with {len(new_commands)} commands")

        except (KeyboardInterrupt, EOFError):
            print("\n❌ Edit cancelled")

    async def _delete_macro(self, macro_name: str = ""):
        """Delete macro with confirmation"""
        if not macro_name:
            if not self.macros:
                print("❌ No macros available")
                return

            print("\n🗑️  Select macro to delete:")
            macro_list = list(self.macros.keys())
            for i, name in enumerate(macro_list, 1):
                macro = self.macros[name]
                print(f"  {i}. {name} ({len(macro.commands)} commands)")

            try:
                choice = await self.session.prompt_async("❯ Select macro (number or name): ")
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(macro_list):
                        macro_name = macro_list[idx]
                    else:
                        print("❌ Invalid selection")
                        return
                except ValueError:
                    if choice in self.macros:
                        macro_name = choice
                    else:
                        print("❌ Macro not found")
                        return
            except (KeyboardInterrupt, EOFError):
                print("\n❌ Cancelled")
                return

        if macro_name not in self.macros:
            print(f"❌ Macro '{macro_name}' not found")
            return

        macro = self.macros[macro_name]

        # Show macro details
        print(f"\n🗑️  Delete macro: {macro_name}")
        print(f"Description: {macro.description}")
        print(f"Commands: {len(macro.commands)}")
        print(f"Usage count: {macro.usage_count}")
        print(f"Tags: {', '.join(macro.tags)}")

        try:
            confirm = await self.session.prompt_async("❯ Are you sure? (yes/no): ")
            if confirm.lower() in ['yes', 'y']:
                del self.macros[macro_name]
                self._save_macros()
                print(f"✅ Macro '{macro_name}' deleted")
            else:
                print("❌ Deletion cancelled")
        except (KeyboardInterrupt, EOFError):
            print("\n❌ Cancelled")

    async def _export_macros(self, file_path: str = ""):
        """Export macros to file"""
        if not self.macros:
            print("❌ No macros to export")
            return

        try:
            if not file_path:
                file_path = await self.session.prompt_async("📁 Export file path: ")
                if not file_path.strip():
                    print("❌ No file path provided")
                    return

            # Select macros to export
            print("\n📦 Select macros to export:")
            macro_list = list(self.macros.keys())
            for i, name in enumerate(macro_list, 1):
                macro = self.macros[name]
                print(f"  {i}. {name} ({len(macro.commands)} commands)")

            selection = await self.session.prompt_async("❯ Select macros (comma-separated numbers or 'all'): ")

            if selection.lower() == 'all':
                selected_macros = macro_list
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in selection.split(',')]
                    selected_macros = [macro_list[i] for i in indices if 0 <= i < len(macro_list)]
                except (ValueError, IndexError):
                    print("❌ Invalid selection")
                    return

            if not selected_macros:
                print("❌ No macros selected")
                return

            # Export data
            export_data = {
                'version': '1.0',
                'exported_at': datetime.datetime.now().isoformat(),
                'macros': {}
            }

            # Export selected macros
            for name in selected_macros:
                macro = self.macros[name]
                export_data['macros'][name] = {
                    'commands': macro.commands,
                    'description': macro.description,
                    'variables': macro.variables,
                    'created': macro.created.isoformat(),
                    'usage_count': macro.usage_count,
                    'tags': macro.tags,
                    'loop_count': macro.loop_count
                }

            # Write to file
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            print(f"✅ Exported {len(selected_macros)} macros to {file_path}")

        except (KeyboardInterrupt, EOFError):
            print("\n❌ Export cancelled")
        except Exception as e:
            print(f"❌ Export failed: {e}")

    async def _import_macros(self, file_path: str = ""):
        """Import macros from file"""
        try:
            if not file_path:
                file_path = await self.session.prompt_async("📁 Import file path: ")
                if not file_path.strip():
                    print("❌ No file path provided")
                    return

            # Check if file exists
            import os
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return

            # Load file
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            if 'macros' not in import_data:
                print("❌ Invalid macro file format")
                return

            imported_macros = import_data['macros']

            print(f"\n📦 Found {len(imported_macros)} macros to import:")
            for name, data in imported_macros.items():
                print(f"  • {name} ({len(data.get('commands', []))} commands)")

            confirm = await self.session.prompt_async("❯ Import all macros? (yes/no): ")
            if confirm.lower() not in ['yes', 'y']:
                print("❌ Import cancelled")
                return

            imported_count = 0
            for name, data in imported_macros.items():
                final_name = name

                # Handle name conflicts
                if name in self.macros:
                    conflict_choice = await self.session.prompt_async(f"❯ Macro '{name}' exists. (o)verwrite, (r)ename, (s)kip: ")
                    if conflict_choice.lower() == 's':
                        continue
                    elif conflict_choice.lower() == 'r':
                        new_name = await self.session.prompt_async(f"❯ New name for '{name}': ")
                        if new_name.strip():
                            final_name = new_name.strip()
                        else:
                            continue

                # Create macro
                try:
                    macro = Macro(
                        name=final_name,
                        commands=data.get('commands', []),
                        description=data.get('description', ''),
                        variables=data.get('variables', {}),
                        created=datetime.datetime.fromisoformat(
                            data.get('created', datetime.datetime.now().isoformat())
                        ),
                        usage_count=data.get('usage_count', 0),
                        tags=data.get('tags', []),
                        loop_count=data.get('loop_count', 1)
                    )

                    self.macros[final_name] = macro
                    imported_count += 1

                except Exception as e:
                    print(f"❌ Failed to import {name}: {e}")

            if imported_count > 0:
                self._save_macros()
                print(f"✅ Successfully imported {imported_count} macros")
            else:
                print("❌ No macros were imported")

        except (KeyboardInterrupt, EOFError):
            print("\n❌ Import cancelled")
        except Exception as e:
            print(f"❌ Import failed: {e}")

    async def _play_macro(self, args=""):
        """Quick macro execution"""
        if not args:
            if not self.macros:
                print("❌ No macros available")
                return

            print("\n▶️  Select macro to play:")
            macro_list = list(self.macros.keys())
            for i, name in enumerate(macro_list, 1):
                macro = self.macros[name]
                print(f"  {i}. {name} - {macro.description}")

            try:
                choice = await self.session.prompt_async("❯ Select macro: ")
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(macro_list):
                        args = macro_list[idx]
                    else:
                        print("❌ Invalid selection")
                        return
                except ValueError:
                    if choice in self.macros:
                        args = choice
                    else:
                        print("❌ Macro not found")
                        return
            except (KeyboardInterrupt, EOFError):
                print("\n❌ Cancelled")
                return

        # Execute macro
        await self._handle_macro_command(args)

    async def _handle_macro_if(self, cmd: str):
        """Handle if statements in macros"""
        # if condition: command
        # if $var == "value": echo "match"
        # if $counter > 5: break

        if ':' not in cmd:
            print(f"❌ Invalid if syntax: {cmd}")
            return
        condition_part, action_part = cmd[3:].split(':', 1)
        condition_part = condition_part.strip()
        action_part = action_part.strip()

        try:
            # Simple condition evaluation
            # Replace variables in condition
            condition_eval = condition_part
            for var_name, var_value in self.macro_context.variables.items():
                condition_eval = condition_eval.replace(f"${var_name}", repr(var_value))
            for var_name, var_value in self.macro_context.loop_vars.items():
                condition_eval = condition_eval.replace(f"${var_name}", repr(var_value))
            for var_name, var_value in self.context.quick_vars.items():
                condition_eval = condition_eval.replace(f"${var_name}", repr(var_value))

            # Evaluate condition
            result = eval(condition_eval)

            if result:
                # Execute action
                if action_part == 'break':
                    self.macro_context.break_flag = True
                elif action_part == 'continue':
                    self.macro_context.continue_flag = True
                elif action_part.startswith('return '):
                    value = action_part[7:].strip()
                    try:
                        self.macro_context.return_value = eval(value, {}, self.macro_context.variables)
                    except:
                        self.macro_context.return_value = value
                    return
                elif action_part.startswith('set '):
                    await self._handle_macro_set(action_part)
                elif action_part.startswith('echo '):
                    print(action_part[5:])
                else:
                    # Execute as regular command
                    await self._process_command(action_part)

        except Exception as e:
            print(f"❌ Error in if condition '{condition_part}': {e}")

    async def _handle_macro_for(self, cmd: str):
        """Handle for loops in macros"""
        # for i in range(5): echo $i
        # for item in $list: process $item
        # for file in ["a.txt", "b.txt"]: cat $file

        if ':' not in cmd:
            print(f"❌ Invalid for syntax: {cmd}")
            return

        loop_part, action_part = cmd[4:].split(':', 1)
        loop_part = loop_part.strip()
        action_part = action_part.strip()

        try:
            # Parse loop: "var in iterable"
            if ' in ' not in loop_part:
                print(f"❌ Invalid for syntax, missing 'in': {loop_part}")
                return

            var_name, iterable_expr = loop_part.split(' in ', 1)
            var_name = var_name.strip()
            iterable_expr = iterable_expr.strip()

            # Substitute variables in iterable expression
            iterable_eval = iterable_expr
            for v_name, v_value in self.macro_context.variables.items():
                iterable_eval = iterable_eval.replace(f"${v_name}", repr(v_value))
            for v_name, v_value in self.macro_context.loop_vars.items():
                iterable_eval = iterable_eval.replace(f"${v_name}", repr(v_value))
            for v_name, v_value in self.context.quick_vars.items():
                iterable_eval = iterable_eval.replace(f"${v_name}", repr(v_value))

            # Evaluate iterable
            iterable = eval(iterable_eval)

            # Execute loop
            for item in iterable:
                if self.macro_context.break_flag:
                    break

                if self.macro_context.continue_flag:
                    self.macro_context.continue_flag = False
                    continue

                # Set loop variable
                old_value = self.macro_context.variables.get(var_name)
                self.macro_context.variables[var_name] = item

                try:
                    # Execute action
                    if action_part == 'break':
                        self.macro_context.break_flag = True
                    elif action_part == 'continue':
                        self.macro_context.continue_flag = True
                    elif action_part.startswith('return '):
                        value = action_part[7:].strip()
                        try:
                            self.macro_context.return_value = eval(value, {}, self.macro_context.variables)
                        except:
                            self.macro_context.return_value = value
                        return
                    elif action_part.startswith('set '):
                        await self._handle_macro_set(action_part)
                    elif action_part.startswith('echo '):
                        print(action_part[5:])
                    else:
                        # Execute as regular command
                        await self._process_command(action_part)

                except Exception as e:
                    print(f"❌ Error in loop action '{action_part}': {e}")

        except Exception as e:
            print(f"❌ Error in for loop '{loop_part}': {e}")

    async def _handle_macro_while(self, cmd: str):
        """Handle while loops in macros"""
        # while condition: command
        # while $counter < 10: echo $counter

        if ':' not in cmd:
            print(f"❌ Invalid while syntax: {cmd}")
            return

        condition_part, action_part = cmd[6:].split(':', 1)
        condition_part = condition_part.strip()
        action_part = action_part.strip()

        try:
            # Simple condition evaluation
            # Replace variables in condition
            condition_eval = condition_part
            for var_name, var_value in self.macro_context.variables.items():
                condition_eval = condition_eval.replace(f"${var_name}", repr(var_value))
            for var_name, var_value in self.macro_context.loop_vars.items():
                condition_eval = condition_eval.replace(f"${var_name}", repr(var_value))
            for var_name, var_value in self.context.quick_vars.items():
                condition_eval = condition_eval.replace(f"${var_name}", repr(var_value))

            # Evaluate condition
            result = eval(condition_eval)

            # Limit while loop to 1000 iterations
            max_iterations = 1000
            iteration_count = 0

            while result and iteration_count < max_iterations:
                # Execute action
                if action_part == 'break':
                    self.macro_context.break_flag = True
                    break
                elif action_part == 'continue':
                    self.macro_context.continue_flag = True
                elif action_part.startswith('return '):
                    value = action_part[7:].strip()
                    try:
                        self.macro_context.return_value = eval(value, {}, self.macro_context.variables)
                    except:
                        self.macro_context.return_value = value
                    return
                elif action_part.startswith('set '):
                    await self._handle_macro_set(action_part)
                elif action_part.startswith('echo '):
                    print(action_part[5:])
                else:
                    # Execute as regular command
                    await self._process_command(action_part)

                # Update loop variables
                for var_name, var_value in self.macro_context.variables.items():
                    if var_name.startswith('!'):
                        self.macro_context.variables[var_name[1:]] = not var_value

                # Re-evaluate condition
                result = eval(condition_eval)

                # Increment iteration count
                iteration_count += 1

            if iteration_count >= max_iterations:
                print(f"⚠️  While loop exceeded maximum iterations ({max_iterations})")

        except Exception as e:
            print(f"❌ Error in while condition '{condition_part}': {e}")

    async def _macro_help(self, args=""):
        """Show comprehensive macro help"""
        if args:
            # Specific help topic
            topic = args.lower()
            if topic == "syntax":
                await self._show_macro_syntax_help()
            elif topic == "control":
                await self._show_macro_control_help()
            elif topic == "variables":
                await self._show_macro_variables_help()
            elif topic == "examples":
                await self._show_macro_examples_help()
            else:
                print(f"❌ Unknown help topic: {topic}")
                print("💡 Available topics: syntax, control, variables, examples")
        else:
            await self._show_macro_overview_help()

    async def _show_macro_overview_help(self):
        """Show macro system overview"""
        print("\n╔════════════════════════════════════════════════════════════════╗")
        print("║                    ⚡ Macro System Help                        ║")
        print("╠════════════════════════════════════════════════════════════════╣")
        print("║                                                                ║")
        print("║  📋 Management Commands:                                       ║")
        print("║    :macro                List all macros                       ║")
        print("║    :macro create         Interactive macro creation            ║")
        print("║    :macro edit <name>    Edit existing macro                   ║")
        print("║    :macro delete <name>  Delete macro with confirmation        ║")
        print("║    :macro export [file]  Export macros to JSON file            ║")
        print("║    :macro import [file]  Import macros from JSON file          ║")
        print("║                                                                ║")
        print("║  ▶️  Execution Commands:                                       ║")
        print("║    :play <name>          Quick macro execution                 ║")
        print("║    :record <name>        Record live commands as macro         ║")
        print("║    @<name> [args]        Execute macro with arguments          ║")
        print("║                                                                ║")
        print("║  📚 Help Topics:                                               ║")
        print("║    :macro help syntax    Command syntax and structure          ║")
        print("║    :macro help control   Control flow (if, for, while)         ║")
        print("║    :macro help variables Variable system and substitution      ║")
        print("║    :macro help examples  Practical macro examples              ║")
        print("║                                                                ║")
        print("╚════════════════════════════════════════════════════════════════╝")

    async def _show_macro_syntax_help(self):
        """Show macro syntax help"""
        print("\n╔════════════════════════════════════════════════════════════════╗")
        print("║                    📝 Macro Syntax Help                        ║")
        print("╠════════════════════════════════════════════════════════════════╣")
        print("║                                                                ║")
        print("║  🔧 Basic Commands:                                            ║")
        print("║    # comment                    Comment line (ignored)         ║")
        print("║    echo <message>               Print message to console       ║")
        print("║    sleep <seconds>              Pause execution                ║")
        print("║    set <var> = <value>          Set variable                   ║")
        print("║    return <value>               Return value and exit          ║")
        print("║    break                        Exit current loop              ║")
        print("║    continue                     Skip to next loop iteration    ║")
        print("║                                                                ║")
        print("║  📊 Variable Usage:                                            ║")
        print("║    $var                         Access variable value          ║")
        print("║    $arg1, $arg2, ...            Macro arguments ($1, $2, ...)  ║")
        print("║    $r1, $r2, ...                Auto-saved command results     ║")
        print("║                                                                ║")
        print("║  🎯 Command Execution:                                         ║")
        print("║    module function args         Execute module function        ║")
        print("║    !system_command              Execute system command         ║")
        print("║    :quick_command               Execute CLI quick command      ║")
        print("║                                                                ║")
        print("║  ⚙️  Macro Properties:                                         ║")
        print("║    • Name: Unique identifier                                   ║")
        print("║    • Description: Optional documentation                       ║")
        print("║    • Tags: Categorization labels                               ║")
        print("║    • Loop Count: Repeat entire macro N times                   ║")
        print("║    • Variables: Persistent macro-specific data                 ║")
        print("║                                                                ║")
        print("╚════════════════════════════════════════════════════════════════╝")

    async def _show_macro_control_help(self):
        """Show control flow help"""
        print("\n╔════════════════════════════════════════════════════════════════╗")
        print("║                   🔄 Control Flow Help                         ║")
        print("╠════════════════════════════════════════════════════════════════╣")
        print("║                                                                ║")
        print("║  🔀 IF Statements:                                             ║")
        print("║    if <condition>: <action>                                    ║")
        print("║                                                                ║")
        print("║    Examples:                                                   ║")
        print("║      if $counter > 5: echo 'High count'                        ║")
        print("║      if $status == 'ready': start_process                      ║")
        print("║      if len($queue) == 0: break                                ║")
        print("║                                                                ║")
        print("║  🔁 FOR Loops:                                                 ║")
        print("║    for <var> in <iterable>: <action>                           ║")
        print("║                                                                ║")
        print("║    Examples:                                                   ║")
        print("║      for i in range(10): echo 'Item $i'                        ║")
        print("║      for file in ['a.txt', 'b.txt']: cat $file                 ║")
        print("║      for item in $my_list: process_item $item                  ║")
        print("║                                                                ║")
        print("║  ⏳ WHILE Loops:                                               ║")
        print("║    while <condition>: <action>                                 ║")
        print("║                                                                ║")
        print("║    Examples:                                                   ║")
        print("║      while $running == True: check_status                      ║")
        print("║      while $counter < 100: increment_counter                   ║")
        print("║      while len($queue) > 0: process_next                       ║")
        print("║                                                                ║")
        print("║  🛡️  Safety Features:                                          ║")
        print("║    • While loops limited to 1000 iterations                    ║")
        print("║    • Break/continue work in all loop types                     ║")
        print("║    • Variable scoping preserved in for loops                   ║")
        print("║    • Error handling for invalid conditions                     ║")
        print("║                                                                ║")
        print("╚════════════════════════════════════════════════════════════════╝")

    async def _show_macro_variables_help(self):
        """Show variable system help"""
        print("\n╔════════════════════════════════════════════════════════════════╗")
        print("║                   📊 Variable System Help                      ║")
        print("╠════════════════════════════════════════════════════════════════╣")
        print("║                                                                ║")
        print("║  🎯 Variable Types:                                            ║")
        print("║    • Macro Arguments: $arg1, $arg2, $1, $2, ...                ║")
        print("║    • Macro Variables: Custom variables set in macro            ║")
        print("║    • Quick Variables: $r1, $r2, ... (command results)          ║")
        print("║    • Loop Variables: Temporary variables in for loops          ║")
        print("║                                                                ║")
        print("║  ⚙️  Variable Operations:                                      ║")
        print("║    set var_name = value         Set variable                   ║")
        print("║    set counter = $counter + 1   Increment counter              ║")
        print("║    set result = len($my_list)   Use functions                  ║")
        print("║                                                                ║")
        print("║  🔄 Variable Substitution:                                     ║")
        print("║    • Variables replaced before command execution               ║")
        print("║    • Works in conditions, actions, and regular commands        ║")
        print("║    • Supports nested variable references                       ║")
        print("║                                                                ║")
        print("║  📋 Variable Scope Priority:                                   ║")
        print("║    1. Loop variables (for loop iteration vars)                 ║")
        print("║    2. Macro variables (set within macro)                       ║")
        print("║    3. Quick variables (global $r1, $r2, etc.)                  ║")
        print("║                                                                ║")
        print("║  💡 Tips:                                                      ║")
        print("║    • Use descriptive variable names                            ║")
        print("║    • Variables persist throughout macro execution              ║")
        print("║    • Return values can be captured as variables                ║")
        print("║    • Use :v to view current variables                          ║")
        print("║                                                                ║")
        print("╚════════════════════════════════════════════════════════════════╝")

    async def _show_macro_examples_help(self):
        """Show practical macro examples"""
        print("\n╔════════════════════════════════════════════════════════════════╗")
        print("║                   💡 Macro Examples Help                       ║")
        print("╠════════════════════════════════════════════════════════════════╣")
        print("║                                                                ║")
        print("║  🔧 Simple Automation:                                         ║")
        print("║    # Daily startup routine                                     ║")
        print("║    echo 'Starting daily tasks...'                              ║")
        print("║    mymodule check_status                                       ║")
        print("║    mymodule update_data                                        ║")
        print("║    echo 'Daily tasks completed!'                               ║")
        print("║                                                                ║")
        print("║  🔄 Loop Processing:                                           ║")
        print("║    # Process multiple files                                    ║")
        print("║    set counter = 0                                             ║")
        print("║    for file in ['data1.txt', 'data2.txt', 'data3.txt']:        ║")
        print("║      echo 'Processing $file...'                                ║")
        print("║      filemod process $file                                     ║")
        print("║      set counter = $counter + 1                                ║")
        print("║    echo 'Processed $counter files'                             ║")
        print("║                                                                ║")
        print("║  🎯 Conditional Logic:                                         ║")
        print("║    # Smart backup routine                                      ║")
        print("║    dbmod get_size                                              ║")
        print("║    if $r1 > 1000000: echo 'Large database detected'            ║")
        print("║    if $r1 > 1000000: dbmod compress_backup                     ║")
        print("║    if $r1 <= 1000000: dbmod quick_backup                       ║")
        print("║    echo 'Backup completed'                                     ║")
        print("║                                                                ║")
        print("║  ⏳ Monitoring Loop:                                           ║")
        print("║    # Service monitoring                                        ║")
        print("║    set max_checks = 10                                         ║")
        print("║    set check_count = 0                                         ║")
        print("║    while $check_count < $max_checks:                           ║")
        print("║      sysmod check_service web_server                           ║")
        print("║      if $r1 == 'running': break                                ║")
        print("║      sleep 5                                                   ║")
        print("║      set check_count = $check_count + 1                        ║")
        print("║    echo 'Service check completed'                              ║")
        print("║                                                                ║")
        print("║  🎪 Advanced Example:                                          ║")
        print("║    # Data processing pipeline with error handling              ║")
        print("║    set errors = 0                                              ║")
        print("║    for dataset in $arg1:                                       ║")
        print("║      echo 'Processing dataset: $dataset'                       ║")
        print("║      datamod validate $dataset                                 ║")
        print("║      if $r1 == 'invalid': set errors = $errors + 1             ║")
        print("║      if $r1 == 'invalid': continue                             ║")
        print("║      datamod transform $dataset                                ║")
        print("║      datamod save_result $dataset                              ║")
        print("║    if $errors > 0: echo 'Warning: $errors datasets failed'     ║")
        print("║    return 'Pipeline completed with $errors errors'             ║")
        print("║                                                                ║")
        print("╚════════════════════════════════════════════════════════════════╝")

# =================== Flow Entry Point ===================

async def run(app: App, args):
    """Main entry point for Beast CLI"""

    with contextlib.suppress(Exception):
        set_title(f"CLI - ToolBoxV2 {app.version}")

    # Create and run CLI
    cli = BeastCLI(app)
    with BlobFile("cli/context.c", key=Code.DK()(), mode="r") as f:
        if f.exists() and f.read():
            cli.context.__dict__ = f.read_json()
    await cli.run()

    # Cleanup
    with contextlib.suppress(Exception):
        set_title("")

    await app.a_exit()

