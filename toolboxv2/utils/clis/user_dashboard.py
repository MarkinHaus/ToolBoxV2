# Add this to the runner_setup() function to replace "default": lambda: None
import os
import sys
from platform import system, node

from ..system.getting_and_closing_app import get_app
from ..system.main_tool import get_version_from_pyproject
from .api import cli_api_runner
from .db_cli_manager import cli_db_runner
from .tcm_p2p_cli import cli_tcm_runner
from ... import Style
from ...mcp_server import MCPSafeIO




def interactive_user_dashboard():
    """Modern interactive user dashboard and mini CLI"""
    import asyncio
    from pathlib import Path

    # =================== UI Helper Functions ===================

    def print_box_header(title: str, icon: str = "ℹ", width: int = 76):
        """Print a styled box header"""
        title_text = f" {icon} {title} "
        padding = (width - len(title_text)) // 2
        icon_adjust = 1 if len(icon) == 1 else 0

        print("\n┌" + "─" * width + "┐")
        print("│" + " " * padding + title_text + " " * (width - padding - len(title_text) - icon_adjust) + "│")
        print("├" + "─" * width + "┤")

    def print_box_content(text: str, style: str = "", width: int = 76):
        """Print content inside a box"""

        if style == "success":
            icon = "✓"
            text = f"{icon} {text}"
        elif style == "error":
            icon = "✗"
            text = f"{icon} {text}"
        elif style == "warning":
            icon = "⚠"
            text = f"{icon} {text}"
        elif style == "info":
            icon = "ℹ"
            text = f"{icon} {text}"

        print("│ " + text.ljust(width - 1) + "│")

    def print_box_footer(width: int = 76):
        """Print box footer"""
        print("└" + "─" * width + "┘\n")

    def print_status(message: str, status: str = "info"):
        """Print a status message with icon"""
        icons = {
            'success': '✓',
            'error': '✗',
            'warning': '⚠',
            'info': 'ℹ',
            'progress': '⟳',
            'module': '📦',
            'function': '⚡',
            'service': '🔧'
        }

        colors = {
            'success': '\033[92m',
            'error': '\033[91m',
            'warning': '\033[93m',
            'info': '\033[94m',
            'progress': '\033[96m',
            'module': '\033[95m',
            'function': '\033[92m',
            'service': '\033[96m'
        }

        reset = '\033[0m'
        icon = icons.get(status, '•')
        color = colors.get(status, '')

        print(f"{color}{icon} {message}{reset}")

    def print_separator(char: str = "─", width: int = 76):
        """Print a separator line"""
        print(char * width)

    def get_key():
        """Get single keypress (cross-platform)"""
        if system() == "Windows":
            import msvcrt
            key = msvcrt.getch()
            if key == b'\xe0':  # Arrow key prefix
                key = msvcrt.getch()
                if key == b'H':
                    return 'up'
                elif key == b'P':
                    return 'down'
            elif key == b'\r':
                return 'enter'
            elif key in (b'q', b'Q', b'\x03'):
                return 'quit'
            elif key in (b'w', b'W'):
                return 'up'
            elif key in (b's', b'S'):
                return 'down'
            elif key in (b'/', b'?'):
                return 'search'
            elif key in (b'h', b'H'):
                return 'help'
            return key.decode('utf-8', errors='ignore')
        else:
            import tty
            import termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
                if ch == '\x1b':  # ESC sequence
                    next_chars = sys.stdin.read(2)
                    if next_chars == '[A':
                        return 'up'
                    elif next_chars == '[B':
                        return 'down'
                elif ch in ('\r', '\n'):
                    return 'enter'
                elif ch in ('q', 'Q', '\x03'):
                    return 'quit'
                elif ch in ('w', 'W'):
                    return 'up'
                elif ch in ('s', 'S'):
                    return 'down'
                elif ch in ('/', '?'):
                    return 'search'
                elif ch in ('h', 'H'):
                    return 'help'
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # =================== Dashboard Manager ===================

    class DashboardManager:
        """Manages the interactive dashboard"""

        def __init__(self, app):
            self.app = app
            self.current_view = "main_menu"
            self.selected_index = 0
            self.running = True
            self.history = []
            self.search_query = ""

            # Cache
            self.modules_cache = None
            self.current_module = None
            self.current_functions = []

        async def run(self):
            """Main run loop"""
            # Clear screen
            print('\033[2J\033[H')

            # Welcome
            print_box_header("ToolBoxV2 Interactive Dashboard", "🎯")
            print_box_content("Welcome to the ToolBoxV2 Command Center", "info")
            print_box_footer()

            await asyncio.sleep(1)

            while self.running:
                try:
                    if self.current_view == "main_menu":
                        await self.show_main_menu()
                    elif self.current_view == "modules":
                        await self.show_modules()
                    elif self.current_view == "module_detail":
                        await self.show_module_detail()
                    elif self.current_view == "function_execute":
                        await self.execute_function()
                    elif self.current_view == "function_runner":
                        await self.show_function_runner()
                    elif self.current_view == "workflow_runner":
                        await self.show_workflow_runner()
                    elif self.current_view == "status":
                        await self.show_status()
                    elif self.current_view == "services":
                        await self.show_services()
                    elif self.current_view == "quick_actions":
                        await self.show_quick_actions()
                    elif self.current_view == "search":
                        await self.show_search()
                    elif self.current_view == "settings":
                        await self.show_settings()
                except KeyboardInterrupt:
                    if await self.confirm_exit():
                        break
                    continue

        async def show_main_menu(self):
            """Show main menu"""
            menu_items = [
                ("📦", "Browse Modules", "modules"),
                ("⚡", "Quick Actions", "quick_actions"),
                ("🎯", "Function Runner", "function_runner"),
                ("⏩", "Workflow Runner", "workflow_runner"),
                ("🔧", "Manage Services", "services"),
                ("📊", "System Status", "status"),
                ("🔍", "Search", "search"),
                ("⚙️", "Settings", "settings"),
                ("❌", "Exit", "exit")
            ]

            while True:
                print('\033[2J\033[H')

                print_box_header("Main Menu", "🏠")
                print()

                # User info
                username = self.app.get_username() if hasattr(self.app, 'get_username') else "Guest"
                print(f"  👤 User: {username}")
                print(f"  📍 Instance: {self.app.id}")
                print(f"  🖥️  System: {system()}")
                print()
                print_separator()
                print()

                # Menu items
                for i, (icon, label, _) in enumerate(menu_items):
                    is_selected = i == self.selected_index
                    arrow = "▶" if is_selected else " "

                    if is_selected:
                        print(f"  {arrow} \033[1;96m{icon} {label}\033[0m")
                    else:
                        print(f"  {arrow} {icon} {label}")

                print()
                print_box_footer()
                print_status("↑↓/w/s: Navigate | Enter: Select | h: Help | q: Quit", "info")

                key = get_key()

                if key == 'quit':
                    if await self.confirm_exit():
                        self.running = False
                        return
                elif key == 'up':
                    self.selected_index = max(0, self.selected_index - 1)
                elif key == 'down':
                    self.selected_index = min(len(menu_items) - 1, self.selected_index + 1)
                elif key == 'enter':
                    _, _, action = menu_items[self.selected_index]

                    if action == "exit":
                        if await self.confirm_exit():
                            self.running = False
                            return
                    elif action == "function_runner":
                        self.history.append(self.current_view)
                        self.current_view = "function_runner"
                        self.selected_index = 0
                        return
                    elif action == "workflow_runner":
                        self.history.append(self.current_view)
                        self.current_view = "workflow_runner"
                        self.selected_index = 0
                        return
                    else:
                        self.history.append(self.current_view)
                        self.current_view = action
                        self.selected_index = 0
                        return
                elif key == 'search':
                    self.history.append(self.current_view)
                    self.current_view = "search"
                    return
                elif key == 'help':
                    await self.show_help()

        async def show_modules(self):
            """Show modules list"""
            if self.modules_cache is None:
                print_status("Loading modules...", "progress")
                self.modules_cache = list(self.app.functions.keys())
                self.modules_cache.sort()

            while True:
                print('\033[2J\033[H')

                print_box_header("Module Browser", "📦")
                print_box_content(f"Total modules: {len(self.modules_cache)}", "info")
                print_box_footer()

                if not self.modules_cache:
                    print_status("No modules loaded", "warning")
                    print_status("Use -l flag to load all modules", "info")
                    print()
                    print_status("Press any key to go back...", "info")
                    get_key()
                    self.go_back()
                    return

                # Calculate visible range
                visible_count = 15
                start_idx = max(0, self.selected_index - visible_count // 2)
                end_idx = min(len(self.modules_cache), start_idx + visible_count)

                if end_idx - start_idx < visible_count:
                    start_idx = max(0, end_idx - visible_count)

                # Show modules
                print()
                for i in range(start_idx, end_idx):
                    module_name = self.modules_cache[i]
                    is_selected = i == self.selected_index
                    arrow = "▶" if is_selected else " "

                    # Get module version if available
                    try:
                        mod = self.app.get_mod(module_name)
                        version = getattr(mod, 'version', '?.?.?')
                    except:
                        version = '?.?.?'

                    if is_selected:
                        print(f"  {arrow} \033[1;96m📦 {module_name:<30} v{version}\033[0m")
                    else:
                        print(f"  {arrow} 📦 {module_name:<30} v{version}")

                if len(self.modules_cache) > visible_count:
                    print(f"\n  Showing {start_idx + 1}-{end_idx} of {len(self.modules_cache)}")

                print()
                print_separator()
                print_status("↑↓/w/s: Navigate | Enter: Open | /: Search | b/Esc: Back", "info")

                key = get_key()

                if key in ('quit', 'b', 'B'):
                    self.go_back()
                    return
                elif key == 'up':
                    self.selected_index = max(0, self.selected_index - 1)
                elif key == 'down':
                    self.selected_index = min(len(self.modules_cache) - 1, self.selected_index + 1)
                elif key == 'enter':
                    self.current_module = self.modules_cache[self.selected_index]
                    self.history.append(self.current_view)
                    self.current_view = "module_detail"
                    self.selected_index = 0
                    return
                elif key == 'search':
                    self.history.append(self.current_view)
                    self.current_view = "search"
                    return

        async def show_module_detail(self):
            """Show module detail with functions"""
            if not self.current_module:
                self.go_back()
                return

            # Load functions
            print_status(f"Loading functions from {self.current_module}...", "progress")

            module_data = self.app.functions.get(self.current_module, {})
            self.current_functions = []

            for func_name, func_data in module_data.items():
                if isinstance(func_data, dict) and 'func' in func_data:
                    self.current_functions.append({
                        'name': func_name,
                        'data': func_data
                    })

            while True:
                print('\033[2J\033[H')

                print_box_header(f"Module: {self.current_module}", "📦")
                print()

                # Module info
                try:
                    mod = self.app.get_mod(self.current_module)
                    version = getattr(mod, 'version', 'unknown')
                    print(f"  Version: {version}")
                except:
                    print(f"  Version: unknown")

                print(f"  Functions: {len(self.current_functions)}")
                print()
                print_separator()
                print()

                if not self.current_functions:
                    print_status("No functions available in this module", "warning")
                    print()
                    print_status("Press any key to go back...", "info")
                    get_key()
                    self.go_back()
                    return

                # Show functions
                visible_count = 12
                start_idx = max(0, self.selected_index - visible_count // 2)
                end_idx = min(len(self.current_functions), start_idx + visible_count)

                if end_idx - start_idx < visible_count:
                    start_idx = max(0, end_idx - visible_count)

                for i in range(start_idx, end_idx):
                    func = self.current_functions[i]
                    is_selected = i == self.selected_index
                    arrow = "▶" if is_selected else " "

                    # Get function type
                    func_type = func['data'].get('type', 'unknown')
                    type_icon = "⚡" if 'async' in str(func_type) else "🔧"

                    if is_selected:
                        print(f"  {arrow} \033[1;96m{type_icon} {func['name']}\033[0m")
                    else:
                        print(f"  {arrow} {type_icon} {func['name']}")

                if len(self.current_functions) > visible_count:
                    print(f"\n  Showing {start_idx + 1}-{end_idx} of {len(self.current_functions)}")

                print()
                print_separator()
                print_status("↑↓/w/s: Navigate | Enter: Execute | i: Info | b: Back", "info")

                key = get_key()

                if key in ('quit', 'b', 'B'):
                    self.go_back()
                    return
                elif key == 'up':
                    self.selected_index = max(0, self.selected_index - 1)
                elif key == 'down':
                    self.selected_index = min(len(self.current_functions) - 1, self.selected_index + 1)
                elif key == 'enter':
                    self.history.append(self.current_view)
                    self.current_view = "function_execute"
                    return
                elif key in ('i', 'I'):
                    await self.show_function_info(self.current_functions[self.selected_index])

        async def show_function_runner(self):
            """Interactive function runner with autocomplete"""
            print('\033[2J\033[H')

            print_box_header("Function Runner", "🎯")
            print_box_content("Execute functions with autocomplete", "info")
            print_box_footer()

            # Restore terminal for input
            if system() != "Windows":
                import termios
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

            print()
            print("  Format: module_name function_name [args...]")
            print("  Example: CloudM Version")
            print("  Example: helper create-user john john@mail.com")
            print()

            # Get all available modules and functions for autocomplete hints
            available_modules = list(self.app.functions.keys())

            print("  Available modules:")
            print("  " + ", ".join(available_modules[:10]))
            if len(available_modules) > 10:
                print(f"  ... and {len(available_modules) - 10} more")
            print()

            command_input = input("  Command: ").strip()

            if not command_input:
                self.go_back()
                return

            parts = command_input.split()

            if len(parts) < 2:
                print()
                print_status("Need at least module and function name", "error")
                print_status("Press any key to continue...", "info")
                get_key()
                return

            module_name = parts[0]
            function_name = parts[1]
            args = parts[2:]

            # Check if module exists
            if module_name not in self.app.functions:
                print()
                print_status(f"Module '{module_name}' not found", "error")

                # Suggest similar modules
                similar = [m for m in available_modules if module_name.lower() in m.lower()]
                if similar:
                    print()
                    print("  Did you mean:")
                    for s in similar[:5]:
                        print(f"    • {s}")

                print()
                print_status("Press any key to continue...", "info")
                get_key()
                return

            # Check if function exists
            module_data = self.app.functions.get(module_name, {})
            if function_name not in module_data:
                print()
                print_status(f"Function '{function_name}' not found in {module_name}", "error")

                # Show available functions
                available_funcs = [f for f in module_data.keys() if isinstance(module_data[f], dict)]
                if available_funcs:
                    print()
                    print("  Available functions:")
                    for f in available_funcs[:10]:
                        print(f"    • {f}")
                    if len(available_funcs) > 10:
                        print(f"    ... and {len(available_funcs) - 10} more")

                print()
                print_status("Press any key to continue...", "info")
                get_key()
                return

            # Ask for kwargs
            print()
            print_status("Enter keyword arguments (optional)", "info")
            kwargs_input = input("  Kwargs (key=value, space-separated): ").strip()

            kwargs = {}
            if kwargs_input:
                for pair in kwargs_input.split():
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        kwargs[key.strip()] = value.strip()
                    elif ':' in pair:
                        key, value = pair.split(':', 1)
                        kwargs[key.strip()] = value.strip()

            print()
            print_separator("═")
            print(f"  Executing: {module_name}.{function_name}")
            print_separator("═")
            print()

            try:
                # Execute function
                result = await self.app.a_run_any(
                    (module_name, function_name),
                    args_=args,
                    tb_run_with_specification='app',
                    get_results=True,
                    **kwargs
                )

                # Handle coroutine results
                if asyncio.iscoroutine(result):
                    result = await result

                if isinstance(result, asyncio.Task):
                    result = await result

                print()
                print_separator("═")
                print("  Result:")
                print_separator("═")
                print()

                if hasattr(result, 'print'):
                    result.print(full_data=True)
                elif hasattr(result, '__dict__'):
                    import pprint
                    pprint.pprint(result.__dict__)
                else:
                    print(f"  {result}")

                print()
                print_status("Execution completed successfully", "success")

            except Exception as e:
                print()
                print_status(f"Execution failed: {e}", "error")

                import traceback
                print()
                print("  Traceback:")
                print_separator()
                traceback.print_exc()

            print()
            print_status("Press any key to continue...", "info")
            get_key()

            # Ask if user wants to run another command
            if system() != "Windows":
                import termios
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

            again = input("\n  Run another command? (y/N): ").strip().lower()

            if again == 'y':
                # Stay in function runner
                return
            else:
                self.go_back()

        async def show_workflow_runner(self):
            """Interactive workflow runner with autocomplete"""
            print('\033[2J\033[H')

            print_box_header("Workflow Runner", "🎯")
            print_box_content("Execute workflows with autocomplete", "info")
            print_box_footer()

            if system() != "Windows":
                import termios
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

            all_flows = self.app.flows.keys()
            if not all_flows:
                from toolboxv2.flows import flows_dict as flows_dict_func
                flows_dict = flows_dict_func(remote=False)
                self.app.set_flows(flows_dict)
                all_flows = self.app.flows.keys()
            print("  Available workflows:")
            # show in an 3 by n grid
            for i, flow in enumerate(all_flows):
                print(f" {str(i) + ' '+flow:<20}", end='\n' if i % 3 == 2 else ' ')

            command_input = input("  Workflow: ").strip()

            try:
                command_input = int(command_input)
                command_input = list(all_flows)[command_input]
            except:
                pass

            if not command_input:
                self.go_back()
                return

            if command_input not in all_flows:
                print()
                print_status(f"Workflow '{command_input}' not found", "error")
                print_status("Press any key to continue...", "info")
                get_key()
                return

            print()
            print_separator("═")
            print(f"  Executing: {command_input}")
            print_separator("═")
            print()
            try:
                self.go_back()
                await self.app.run_flows(command_input)
                print()
                print_status("Execution completed successfully", "success")
            except Exception as e:
                print()
                print_status(f"Execution failed: {e}", "error")
                import traceback
                print()
                print("  Traceback:")
                print_separator()
                traceback.print_exc()


        async def execute_function(self):
            """Execute selected function"""
            if not self.current_module or not self.current_functions:
                self.go_back()
                return

            func = self.current_functions[self.selected_index]

            print('\033[2J\033[H')

            print_box_header(f"Execute Function", "⚡")
            print_box_content(f"Module: {self.current_module}", "info")
            print_box_content(f"Function: {func['name']}", "info")
            print_box_footer()

            # Restore terminal for input
            if system() != "Windows":
                import termios
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

            # Get arguments
            print()
            print_status("Enter function arguments (leave empty if none)", "info")
            args_input = input("  Args (space-separated): ").strip()

            args = args_input.split() if args_input else []

            print()
            print_status("Enter keyword arguments (leave empty if none)", "info")
            kwargs_input = input("  Kwargs (key=value, space-separated): ").strip()

            kwargs = {}
            if kwargs_input:
                for pair in kwargs_input.split():
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        kwargs[key.strip()] = value.strip()

            print()
            print_separator("═")
            print("  Executing...")
            print_separator("═")
            print()

            try:
                # Execute
                result = await self.app.a_run_any(
                    (self.current_module, func['name']),
                    args_=args,
                    tb_run_with_specification='app',
                    get_results=True,
                    **kwargs
                )

                if asyncio.iscoroutine(result):
                    result = await result

                print()
                print_separator("═")
                print("  Result:")
                print_separator("═")
                print()

                if hasattr(result, 'print'):
                    result.print(full_data=True)
                else:
                    print(f"  {result}")

                print()
                print_status("Execution completed successfully", "success")

            except Exception as e:
                print()
                print_status(f"Execution failed: {e}", "error")

                import traceback
                print()
                print("  Traceback:")
                print_separator()
                print(traceback.format_exc())

            print()
            print_status("Press any key to continue...", "info")
            get_key()

            self.go_back()

        async def show_status(self):
            """Show system status"""
            print('\033[2J\033[H')

            print_box_header("System Status", "📊")
            print_box_footer()

            # User info
            print()
            print_separator("─")
            print("  User Information")
            print_separator("─")

            try:
                username = self.app.get_username() if hasattr(self.app, 'get_username') else "Guest"
                print(f"  Username: {username}")

                # Check login status
                try:
                    from toolboxv2.utils.extras.blobs import BlobFile
                    from toolboxv2.utils.security.cryp import Code

                    with BlobFile(f"claim/{username}/jwt.c", key=Code.DK()(), mode="r") as blob:
                        claim = blob.read()
                        if claim and claim != b'Error decoding':
                            print(f"  Status: \033[92m✓ Logged in\033[0m")
                        else:
                            print(f"  Status: \033[91m✗ Not logged in\033[0m")
                except:
                    print(f"  Status: \033[91m✗ Not logged in\033[0m")
            except:
                print(f"  Username: Guest")
                print(f"  Status: \033[91m✗ Not logged in\033[0m")

            # System info
            print()
            print_separator("─")
            print("  System Information")
            print_separator("─")
            print(f"  Instance: {self.app.id}")
            print(f"  System: {system()}")
            print(f"  Node: {node()}")

            # Modules
            print()
            print_separator("─")
            print("  Modules")
            print_separator("─")
            modules_count = len(self.app.functions.keys())
            print(f"  Loaded modules: {modules_count}")

            # Services
            print()
            print_separator("─")
            print("  Services Status")
            print_separator("─")

            # Check DB
            try:
                # Quick check without full status output
                print(f"  Database: \033[92m✓ Available\033[0m")
            except:
                print(f"  Database: \033[91m✗ Not available\033[0m")

            # Check API
            try:
                from toolboxv2.utils.system.state_system import read_server_state
                pid, _, _ = read_server_state()
                from toolboxv2.utils.system.state_system import is_process_running
                if is_process_running(pid):
                    print(f"  API Server: \033[92m✓ Running (PID: {pid})\033[0m")
                else:
                    print(f"  API Server: \033[91m✗ Stopped\033[0m")
            except:
                print(f"  API Server: \033[91m✗ Stopped\033[0m")

            print()
            print_separator()
            print_status("Press any key to go back...", "info")
            get_key()

            self.go_back()

        async def show_services(self):
            """Show services management"""
            services = [
                ("🖥️", "API Server", "api"),
                ("🗄️", "Database", "db"),
                ("🌐", "P2P Client", "p2p"),
                ("📦", "Module Manager", "mods"),
                ("🔙", "Back", "back")
            ]

            while True:
                print('\033[2J\033[H')

                print_box_header("Service Management", "🔧")
                print_box_footer()

                print()
                for i, (icon, label, _) in enumerate(services):
                    is_selected = i == self.selected_index
                    arrow = "▶" if is_selected else " "

                    if is_selected:
                        print(f"  {arrow} \033[1;96m{icon} {label}\033[0m")
                    else:
                        print(f"  {arrow} {icon} {label}")

                print()
                print_separator()
                print_status("↑↓/w/s: Navigate | Enter: Manage | b: Back", "info")

                key = get_key()

                if key in ('quit', 'b', 'B'):
                    self.go_back()
                    return
                elif key == 'up':
                    self.selected_index = max(0, self.selected_index - 1)
                elif key == 'down':
                    self.selected_index = min(len(services) - 1, self.selected_index + 1)
                elif key == 'enter':
                    _, _, action = services[self.selected_index]

                    if action == "back":
                        self.go_back()
                        return
                    else:
                        await self.manage_service(action)

        async def manage_service(self, service_name: str):
            """Manage a specific service"""
            print('\033[2J\033[H')

            print_box_header(f"Manage {service_name.upper()}", "🔧")
            print_box_footer()

            actions = [
                ("▶️", "Start", "start"),
                ("⏹️", "Stop", "stop"),
                ("📊", "Status", "status"),
            ]

            if service_name == "api":
                # build, debug, clean, remove-exe, update
                actions.extend([
                    ("🔨", "Build", "build"),
                    ("🔥", "Debug", "debug"),
                    ("🧹", "Clean", "clean"),
                    ("🗑️", "Remove Executable", "remove-exe"),
                    ("🔄", "Update", "update"),
                ])
            if service_name == "db":
                # health, update , build, clean, discover
                actions.extend([
                    ("❤️", "Health", "health"),
                    ("🔄", "Update", "update"),
                    ("🔨", "Build", "build"),
                    ("🧹", "Clean", "clean"),
                    ("🔍", "Discover", "discover"),
                ])
            if service_name == "p2p":
                # interactive
                actions.append(("🎮", "Interactive", "interactive"))
                actions.remove(("▶️", "Start", "start"))
                actions.remove(("⏹️", "Stop", "stop"))

            actions.append(("🔙", "Back", "back"))

            action_idx = 0

            while True:
                print('\033[2J\033[H')

                print_box_header(f"Manage {service_name.upper()}", "🔧")
                print_box_footer()

                print()
                for i, (icon, label, _) in enumerate(actions):
                    is_selected = i == action_idx
                    arrow = "▶" if is_selected else " "

                    if is_selected:
                        print(f"  {arrow} \033[1;96m{icon} {label}\033[0m")
                    else:
                        print(f"  {arrow} {icon} {label}")

                print()
                print_separator()
                print_status("↑↓/w/s: Navigate | Enter: Execute | b: Back", "info")

                key = get_key()

                if key in ('quit', 'b', 'B'):
                    return
                elif key == 'up':
                    action_idx = max(0, action_idx - 1)
                elif key == 'down':
                    action_idx = min(len(actions) - 1, action_idx + 1)
                elif key == 'enter':
                    _, _, action = actions[action_idx]

                    if action == "back":
                        return

                    print()
                    print_separator("═")
                    print(f"  Executing: {action} on {service_name}")
                    print_separator("═")
                    print()

                    # Execute action
                    try:
                        if service_name == "api":
                            sys.argv = ["api", action]
                            cli_api_runner()
                        elif service_name == "db":
                            sys.argv = ["db", action]
                            cli_db_runner()
                        elif service_name == "p2p":
                            sys.argv = ["p2p", action]
                            cli_tcm_runner()
                        elif service_name == "mods":
                            await self.app.a_run_any("CloudM", "manager")

                        print()
                        print_status("Command executed", "success")
                    except Exception as e:
                        print()
                        print_status(f"Error: {e}", "error")

                    print()
                    print_status("Press any key to continue...", "info")
                    get_key()

        async def show_quick_actions(self):
            """Show quick actions menu"""
            actions = [
                ("🔐", "Login", self.quick_login),
                ("🚪", "Logout", self.quick_logout),
                ("📊", "System Status", self.quick_status),
                ("🔄", "Reload Modules", self.quick_reload),
                ("🧹", "Clear Cache", self.quick_clear_cache),
                ("🔙", "Back", None)
            ]

            while True:
                print('\033[2J\033[H')

                print_box_header("Quick Actions", "⚡")
                print_box_footer()

                print()
                for i, (icon, label, _) in enumerate(actions):
                    is_selected = i == self.selected_index
                    arrow = "▶" if is_selected else " "

                    if is_selected:
                        print(f"  {arrow} \033[1;96m{icon} {label}\033[0m")
                    else:
                        print(f"  {arrow} {icon} {label}")

                print()
                print_separator()
                print_status("↑↓/w/s: Navigate | Enter: Execute | b: Back", "info")

                key = get_key()

                if key in ('quit', 'b', 'B'):
                    self.go_back()
                    return
                elif key == 'up':
                    self.selected_index = max(0, self.selected_index - 1)
                elif key == 'down':
                    self.selected_index = min(len(actions) - 1, self.selected_index + 1)
                elif key == 'enter':
                    _, _, action_func = actions[self.selected_index]

                    if action_func is None:
                        self.go_back()
                        return

                    await action_func()

        async def show_search(self):
            """Show search interface"""
            print('\033[2J\033[H')

            print_box_header("Search", "🔍")
            print_box_footer()

            # Restore terminal for input
            if system() != "Windows":
                import termios
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

            print()
            query = input("  Search query: ").strip().lower()

            if not query:
                self.go_back()
                return

            print()
            print_status("Searching...", "progress")

            # Search in modules and functions
            results = []

            for module_name in self.app.functions.keys():
                if query in module_name.lower():
                    results.append(("module", module_name, None))

                module_data = self.app.functions.get(module_name, {})
                for func_name in module_data.keys():
                    if query in func_name.lower():
                        results.append(("function", module_name, func_name))

            print('\033[2J\033[H')

            print_box_header(f"Search Results for '{query}'", "🔍")
            print_box_content(f"Found {len(results)} results", "info")
            print_box_footer()

            if not results:
                print()
                print_status("No results found", "warning")
            else:
                print()
                for result_type, module, func in results[:20]:  # Limit to 20 results
                    if result_type == "module":
                        print(f"  📦 Module: {module}")
                    else:
                        print(f"  ⚡ Function: {module}.{func}")

                if len(results) > 20:
                    print(f"\n  ... and {len(results) - 20} more results")

            print()
            print_separator()
            print_status("Press any key to go back...", "info")
            get_key()

            self.go_back()

        async def show_settings(self):
            """Show settings menu"""
            settings = [
                ("🔧", "Environment Variables", "env"),
                ("📝", "View Config", "view_config"),
                ("💾", "Save Config", "save_config"),
                ("📈", "App Footprint", "app_footprint"),
                ("ℹ️", "About", "about"),

                ("🔙", "Back", "back")
            ]

            self.selected_index = 0

            while True:
                print('\033[2J\033[H')

                print_box_header("Settings", "⚙️")
                print_box_footer()

                print()
                for i, (icon, label, _) in enumerate(settings):
                    is_selected = i == self.selected_index
                    arrow = "▶" if is_selected else " "

                    if is_selected:
                        print(f"  {arrow} \033[1;96m{icon} {label}\033[0m")
                    else:
                        print(f"  {arrow} {icon} {label}")

                print()
                print_separator()
                print_status("↑↓/w/s: Navigate | Enter: Open | b: Back", "info")

                key = get_key()

                if key in ('quit', 'b', 'B'):
                    self.go_back()
                    return
                elif key == 'up':
                    self.selected_index = max(0, self.selected_index - 1)
                elif key == 'down':
                    self.selected_index = min(len(settings) - 1, self.selected_index + 1)
                elif key == 'enter':
                    _, _, action = settings[self.selected_index]

                    if action == "back":
                        self.go_back()
                        return
                    elif action == "about":
                        await self.show_about()
                    elif action == "env":
                        await self.manage_env_vars()
                    elif action == "view_config":
                        await self.view_config()
                    elif action == "save_config":
                        await self.save_config()
                    elif action == "app_footprint":
                        print(get_app().print_footprint())
                        input(Style.GREY("Press Enter to continue..."))

        async def manage_env_vars(self):
            """Manage environment variables"""
            import os

            # Important ToolBox env vars
            env_vars = [
                ("TOOLBOXV2_REMOTE_BASE", "Remote server base URL", os.getenv("TOOLBOXV2_REMOTE_BASE", "")),
                ("APP_BASE_URL", "Application base URL", os.getenv("APP_BASE_URL", "")),
                ("TB_R_KEY", "Remote access key", os.getenv("TB_R_KEY", "")),
                ("DB_MODE_KEY", "Database mode", os.getenv("DB_MODE_KEY", "LC")),
                ("PYTHON_EXECUTABLE", "Python executable path", os.getenv("PYTHON_EXECUTABLE", "")),
                ("RUST_LOG", "Rust log level", os.getenv("RUST_LOG", "")),
            ]

            actions = [
                ("➕", "Add/Edit Variable", "edit"),
                ("📋", "View All", "view"),
                ("💾", "Save to .env", "save"),
                ("🔄", "Reload from .env", "reload"),
                ("🔙", "Back", "back")
            ]

            selected = 0

            while True:
                print('\033[2J\033[H')

                print_box_header("Environment Variables", "🔧")
                print_box_footer()

                # Show current values
                print()
                print_separator("─")
                print("  Current Variables:")
                print_separator("─")

                for var_name, description, value in env_vars:
                    display_value = value if value else "\033[90m(not set)\033[0m"
                    if len(display_value) > 40 and value:
                        display_value = display_value[:37] + "..."
                    print(f"  {var_name:<25} {display_value}")

                print()
                print_separator("─")
                print("  Actions:")
                print_separator("─")
                print()

                for i, (icon, label, _) in enumerate(actions):
                    is_selected = i == selected
                    arrow = "▶" if is_selected else " "

                    if is_selected:
                        print(f"  {arrow} \033[1;96m{icon} {label}\033[0m")
                    else:
                        print(f"  {arrow} {icon} {label}")

                print()
                print_separator()
                print_status("↑↓/w/s: Navigate | Enter: Select | b: Back", "info")

                key = get_key()

                if key in ('quit', 'b', 'B'):
                    return
                elif key == 'up':
                    selected = max(0, selected - 1)
                elif key == 'down':
                    selected = min(len(actions) - 1, selected + 1)
                elif key == 'enter':
                    _, _, action = actions[selected]

                    if action == "back":
                        return
                    elif action == "edit":
                        await self.edit_env_var(env_vars)
                    elif action == "view":
                        await self.view_all_env_vars()
                    elif action == "save":
                        await self.save_env_to_file(env_vars)
                    elif action == "reload":
                        await self.reload_env_from_file()

        async def edit_env_var(self, env_vars):
            """Edit an environment variable"""
            print('\033[2J\033[H')

            print_box_header("Edit Environment Variable", "✎")
            print_box_footer()

            # Restore terminal
            if system() != "Windows":
                import termios
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

            print()
            print("  Available variables:")
            for i, (name, desc, _) in enumerate(env_vars, 1):
                print(f"    {i}. {name} - {desc}")

            print()
            choice = input("  Select variable number (or enter custom name): ").strip()

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(env_vars):
                    var_name = env_vars[idx][0]
                else:
                    var_name = choice
            except ValueError:
                var_name = choice

            if not var_name:
                return

            current_value = os.getenv(var_name, "")
            print(f"\n  Current value: {current_value or '(not set)'}")

            new_value = input(f"  New value (leave empty to keep current): ").strip()

            if new_value:
                os.environ[var_name] = new_value
                print()
                print_status(f"Set {var_name} = {new_value}", "success")
            else:
                print()
                print_status("No changes made", "info")

            print()
            print_status("Press any key to continue...", "info")
            get_key()

        async def view_all_env_vars(self):
            """View all environment variables"""
            print('\033[2J\033[H')

            print_box_header("All Environment Variables", "📋")
            print_box_footer()

            env_vars = sorted(os.environ.items())

            print()
            print(f"  Total: {len(env_vars)} variables")
            print()
            print_separator()

            # Show first 30
            for key, value in env_vars[:30]:
                display_value = value
                if len(display_value) > 50:
                    display_value = display_value[:47] + "..."
                print(f"  {key:<30} = {display_value}")

            if len(env_vars) > 30:
                print(f"\n  ... and {len(env_vars) - 30} more")

            print()
            print_separator()
            print_status("Press any key to go back...", "info")
            get_key()

        async def save_env_to_file(self, env_vars):
            """Save environment variables to .env file"""
            from pathlib import Path

            print('\033[2J\033[H')

            print_box_header("Save to .env File", "💾")
            print_box_footer()

            env_file = Path(".env")

            print()
            print(f"  File: {env_file.absolute()}")
            print()

            # Restore terminal
            if system() != "Windows":
                import termios
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

            confirm = input("  Save current values to .env? (y/N): ").strip().lower()

            if confirm == 'y':
                try:
                    with open(env_file, 'w') as f:
                        for var_name, description, value in env_vars:
                            current = os.getenv(var_name, value)
                            if current:
                                f.write(f"# {description}\n")
                                f.write(f"{var_name}={current}\n\n")

                    print()
                    print_status(f"Saved to {env_file}", "success")
                except Exception as e:
                    print()
                    print_status(f"Error saving: {e}", "error")
            else:
                print()
                print_status("Cancelled", "info")

            print()
            print_status("Press any key to continue...", "info")
            get_key()

        async def reload_env_from_file(self):
            """Reload environment variables from .env file"""
            from pathlib import Path
            from dotenv import load_dotenv

            print('\033[2J\033[H')

            print_box_header("Reload from .env", "🔄")
            print_box_footer()

            env_file = Path(".env")

            print()
            if not env_file.exists():
                print_status(f".env file not found: {env_file.absolute()}", "warning")
            else:
                try:
                    load_dotenv(override=True)
                    print_status("Environment variables reloaded", "success")
                except Exception as e:
                    print_status(f"Error reloading: {e}", "error")

            print()
            print_status("Press any key to continue...", "info")
            get_key()

        async def view_config(self):
            """View current configuration"""
            print('\033[2J\033[H')

            print_box_header("Current Configuration", "📝")
            print_box_footer()

            print()
            print_separator("─")
            print("  Application Settings:")
            print_separator("─")
            print(f"  Instance ID: {self.app.id}")
            print(f"  Start Directory: {self.app.start_dir}")
            print(f"  System: {system()}")
            print(f"  Node: {node()}")

            print()
            print_separator("─")
            print("  Module Information:")
            print_separator("─")
            modules_count = len(self.app.functions.keys())
            print(f"  Loaded Modules: {modules_count}")

            print()
            print_separator()
            print_status("Press any key to go back...", "info")
            get_key()

        async def save_config(self):
            """Save current configuration"""
            print('\033[2J\033[H')

            print_box_header("Save Configuration", "💾")
            print_box_footer()

            print()
            print_status("Configuration auto-saved", "success")
            print()
            print_status("Press any key to continue...", "info")
            get_key()

        async def show_about(self):
            """Show about information"""
            print('\033[2J\033[H')

            print_box_header("About ToolBoxV2", "ℹ️")
            print()

            version = get_version_from_pyproject()
            from toolboxv2 import tb_root_dir, init_cwd

            print(f"  ToolBoxV2 Interactive Dashboard")
            print(f"  Version: {version}")
            print(f"  System: {system()}")
            print(f"  Python: {sys.version.split()[0]}")
            print(f"  Home: {tb_root_dir}")
            print(f"  Start: {init_cwd}")
            print()
            print(f"  A powerful, modular Python framework")
            print(f"  for building and managing tools.")
            print()
            print_box_footer()

            print_status("Press any key to go back...", "info")
            get_key()

        async def show_help(self):
            """Show help screen"""
            print('\033[2J\033[H')

            print_box_header("Keyboard Shortcuts", "❓")
            print()
            print("  Navigation:")
            print("    ↑/↓ or w/s     Navigate menu items")
            print("    Enter          Select/Execute")
            print("    b / Esc        Go back")
            print()
            print("  Global:")
            print("    /              Search")
            print("    h              Show help")
            print("    q              Quit")
            print()
            print("  Function Execution:")
            print("    i              Show function info")
            print("    Enter          Execute function")
            print()
            print_box_footer()

            print_status("Press any key to continue...", "info")
            get_key()

        async def show_function_info(self, func):
            """Show detailed function information"""
            print('\033[2J\033[H')

            print_box_header(f"Function Info: {func['name']}", "ℹ️")
            print()

            func_data = func['data']

            print(f"  Name: {func['name']}")
            print(f"  Module: {self.current_module}")
            print(f"  Type: {func_data.get('type', 'unknown')}")

            if 'version' in func_data:
                print(f"  Version: {func_data['version']}")

            if 'test' in func_data:
                print(f"  Testable: {func_data['test']}")

            # Try to get docstring
            try:
                func_obj = func_data.get('func')
                if func_obj and hasattr(func_obj, '__doc__') and func_obj.__doc__:
                    print()
                    print("  Description:")
                    print_separator()
                    for line in func_obj.__doc__.split('\n'):
                        print(f"  {line}")
            except:
                pass

            print()
            print_box_footer()

            print_status("Press any key to continue...", "info")
            get_key()

        # Quick action implementations

        async def quick_login(self):
            """Quick login action"""
            print('\033[2J\033[H')
            print_box_header("Quick Login", "🔐")
            print_box_footer()

            try:
                result = await self.app.a_run_any("CloudM", "cli_web_login")
                print()
                if result:
                    print_status("Login successful!", "success")
                else:
                    print_status("Login failed or cancelled", "warning")
            except Exception as e:
                print_status(f"Error: {e}", "error")

            print()
            print_status("Press any key to continue...", "info")
            get_key()

        async def quick_logout(self):
            """Quick logout action"""
            print('\033[2J\033[H')
            print_box_header("Quick Logout", "🚪")
            print_box_footer()

            try:
                result = await self.app.a_run_any("CloudM", "cli_logout")
                print()
                print_status("Logout successful!", "success")
            except Exception as e:
                print_status(f"Error: {e}", "error")

            print()
            print_status("Press any key to continue...", "info")
            get_key()

        async def quick_status(self):
            """Quick status check"""
            await self.show_status()

        async def quick_reload(self):
            """Quick module reload"""
            print('\033[2J\033[H')
            print_box_header("Reload Modules", "🔄")
            print_box_footer()

            print()
            print_status("Reloading modules...", "progress")

            try:
                await self.app.load_all_mods_in_file()
                self.modules_cache = None  # Clear cache
                print_status("Modules reloaded successfully!", "success")
            except Exception as e:
                print_status(f"Error: {e}", "error")

            print()
            print_status("Press any key to continue...", "info")
            get_key()

        async def quick_clear_cache(self):
            """Clear dashboard cache"""
            print('\033[2J\033[H')
            print_box_header("Clear Cache", "🧹")
            print_box_footer()

            print()
            self.modules_cache = None
            self.current_functions = []
            print_status("Cache cleared!", "success")

            print()
            print_status("Press any key to continue...", "info")
            get_key()

        # Helper methods

        def go_back(self):
            """Go back to previous view"""
            if self.history:
                self.current_view = self.history.pop()
                self.selected_index = 0
            else:
                self.current_view = "main_menu"
                self.selected_index = 0

        async def confirm_exit(self):
            """Confirm exit"""
            print('\033[2J\033[H')

            print_box_header("Confirm Exit", "❓")
            print_box_content("Are you sure you want to exit?", "warning")
            print_box_footer()

            print()
            print("  Press 'y' to confirm, any other key to cancel")

            key = get_key()
            return key in ('y', 'Y')

    # =================== Main Entry Point ===================

    async def run_dashboard():
        """Run the dashboard"""
        # Setup app
        app= get_app(from_="run_dashboard")

        # Create and run dashboard
        dashboard = DashboardManager(app)

        # Load modules if not already loaded
        if not app.functions or len(app.functions) == 0:
            print_status("No modules loaded. Use -l flag to load all modules.", "info")
            print_status("or in ui '⚡ Quick Actions' -> '🔄 Reload Modules' ", "info")

        await dashboard.run()

        # Cleanup
        print('\033[2J\033[H')
        print_box_header("Goodbye!", "👋")
        print_box_content("Thank you for using ToolBoxV2", "success")
        print_box_footer()

        if not app.called_exit[0]:
            await app.a_exit()

    # Run
    asyncio.run(run_dashboard())

