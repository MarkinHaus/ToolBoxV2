

import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from functools import wraps

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import (
    FuzzyCompleter,
    NestedCompleter,
    WordCompleter,
)
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

# ToolboxV2 Imports
from toolboxv2 import get_app, __init_cwd__
from toolboxv2.mods.isaa.module import Tools as Isaatools
from toolboxv2.mods.isaa.module import detect_shell
from toolboxv2.utils.extras.Style import remove_styles, Style

NAME = "isaa_cli"


# --- HELPER DECORATOR FOR STYLE CLASS ---
def text_save(func):
    """Dummy decorator to support the provided Style class implementation."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper




# --- UTILITY FUNCTIONS ---

def print_columns(c1: str, c2: str, c3: str):
    """
    Prints three columns of text, dynamically adjusted to terminal width.
    Text is truncated if it exceeds column width.
    """
    try:
        width = shutil.get_terminal_size().columns
    except:
        width = 80

    # Calculate column width (subtracting 2 for spacing)
    col_w = (width // 3) - 2
    if col_w < 10: col_w = 10  # Minimum width

    def truncate_pad(text, length):
        # Remove ANSI codes for length calculation
        clean = remove_styles(text)
        real_len = len(clean)

        if real_len > length:
            # If plain text is too long, we need a simple approach.
            # Handling colored truncation is hard, so we just return trimmed styled text if simple
            return text.replace(clean, clean[:length - 3] + "...")
        else:
            padding = " " * (length - real_len)
            return text + padding

    row = f"{truncate_pad(c1, col_w)}  {truncate_pad(c2, col_w)}  {truncate_pad(c3, col_w)}"
    print(row)


def print_tree(data: Dict, level: int = 0, max_depth: int = 3):
    """Recursively prints a VFS directory structure."""
    if level > max_depth:
        return

    indent = "  " * level
    folder_icon = Style.BLUE("📂")
    file_icon = Style.GREY("📄")

    # Sorted: Folders first
    items = sorted(data.items(), key=lambda x: (not isinstance(x[1], dict), x[0]))

    for name, content in items:
        if name.startswith('.'): continue  # Skip hidden

        if isinstance(content, dict):
            # Directory
            print(f"{indent}{folder_icon} {Style.Bold(Style.CYAN(name))}")
            print_tree(content, level + 1, max_depth)
        else:
            # File (content is usually file content or metadata string)
            size_hint = ""
            if isinstance(content, str):
                size_hint = f"{Style.GREY(f'({len(content)}b)')}"
            print(f"{indent}{file_icon} {name} {size_hint}")


def format_tokens(count: int, limit: int = 128000) -> str:
    """Formats token usage with color warnings."""
    usage_pct = (count / limit) * 100
    color = Style.GREEN
    if usage_pct > 50: color = Style.YELLOW
    if usage_pct > 80: color = Style.RED
    return color(f"{count:,}")


# --- CLI CLASS ---

class SimpleIsaaCLI:
    """
    Simplified, lightweight ISAA CLI V3 with focus on visual insights and low overhead.
    """

    def __init__(self, app_instance: Any):
        self.app = app_instance
        self.isaa_tools: Isaatools = app_instance.get_mod("isaa")

        self.state_file = Path(self.app.data_dir) / "isaa_v3_state.json"
        self.history_file = Path(self.app.data_dir) / "isaa_v3_history.txt"

        # Default State
        self.active_agent_name = "default"
        self.active_session_id = "default"

        # Load persisted state
        self._load_state()

        # Initialize Prompt Toolkit components
        self.history = FileHistory(str(self.history_file))
        self.completer_dict = self._build_completer()
        self.prompt_session = PromptSession(
            history=self.history,
            completer=FuzzyCompleter(NestedCompleter.from_nested_dict(self.completer_dict)),
            complete_while_typing=True
        )

    def _build_completer(self) -> Dict:
        """Builds the autocomplete dictionary."""
        agent_names = self.isaa_tools.config.get("agents-name-list", [])
        features = ["lsp", "docker"]
        toggles = ["on", "off"]

        return {
            "/agent": {
                "switch": WordCompleter(agent_names),
                "list": None,
            },
            "/session": {
                "list": None,
                "new": None,
                "switch": None,  # Dynamic, handled by runtime logic strictly speaking
            },
            "/feature": {
                f: WordCompleter(toggles) for f in features
            },
            "/vfs": {
                "show": None
            },
            "/context": {
                "stats": None
            },
            "/status": None,
            "/clear": None,
            "/help": None,
            "/quit": None,
            "/exit": None
        }

    def _load_state(self):
        """Loads last active agent and session."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.active_agent_name = data.get("agent", "default")
                    self.active_session_id = data.get("session", "default")
            except Exception:
                pass  # Fail silently, use defaults

    def _save_state(self):
        """Persists current agent and session."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump({
                    "agent": self.active_agent_name,
                    "session": self.active_session_id
                }, f)
        except Exception:
            pass

    async def get_active_agent(self):
        """Helper to get the actual agent instance."""
        try:
            return await self.isaa_tools.get_agent(self.active_agent_name)
        except Exception:
            print(Style.RED(f"Error: Active agent '{self.active_agent_name}' not found. Switching to default."))
            self.active_agent_name = "default"
            return await self.isaa_tools.get_agent("default")

    def get_prompt_text(self) -> HTML:
        """Returns the prompt string with HTML formatting."""
        # Clean naming for display
        cwd_name = Path.cwd().name
        return HTML(
            f"<style fg='ansicyan'>[</style>"
            f"<style fg='ansigreen'>{cwd_name}</style>"
            f"<style fg='ansicyan'>]</style> "
            f"<style fg='ansiyellow'>({self.active_agent_name})</style>"
            f"<style fg='grey'>@{self.active_session_id}</style>"
            f"\n<style fg='ansiblue'>❯</style> "
        )

    # --- MAIN LOOP ---

    async def run(self):
        """Main execution loop."""
        print(Style.Bold(Style.BLUE("\n=== ISAA CLI V3: Lightweight Agent Interface ===\n")))
        await self._print_status_dashboard()

        while True:
            try:
                # Update completer if agent list changed (simplified: rebuilds on every loop)
                self.prompt_session.completer = FuzzyCompleter(
                    NestedCompleter.from_nested_dict(self._build_completer())
                )

                with patch_stdout():
                    user_input = await self.prompt_session.prompt_async(self.get_prompt_text())

                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.startswith("!"):
                    await self._handle_shell(user_input[1:])
                elif user_input.startswith("/"):
                    await self._handle_command(user_input)
                else:
                    await self._handle_agent_interaction(user_input)

            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            except Exception as e:
                print(Style.RED(f"Unexpected Error: {e}"))
                import traceback
                traceback.print_exc()

        self._save_state()
        print(Style.GREEN("\nGoodbye."))

    # --- COMMAND HANDLERS ---

    async def _handle_command(self, cmd_str: str):
        parts = cmd_str.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd == "/quit" or cmd == "/exit":
            raise EOFError

        elif cmd == "/clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            await self._print_status_dashboard()

        elif cmd == "/agent":
            await self._cmd_agent(args)

        elif cmd == "/session":
            await self._cmd_session(args)

        elif cmd == "/feature":
            await self._cmd_feature(args)

        elif cmd == "/vfs":
            if args and args[0] == "show":
                await self._cmd_vfs_show()
            else:
                print(Style.RED("Usage: /vfs show"))

        elif cmd == "/context":
            if args and args[0] == "stats":
                await self._cmd_context_stats()
            else:
                print(Style.RED("Usage: /context stats"))

        elif cmd == "/status":
            await self._print_status_dashboard()

        elif cmd == "/help":
            self._print_help()

        else:
            print(Style.RED(f"Unknown command: {cmd}"))

    async def _cmd_agent(self, args: List[str]):
        if not args:
            print(Style.YELLOW("Usage: /agent <switch|list> [name]"))
            return

        action = args[0]
        if action == "list":
            agents = self.isaa_tools.config.get("agents-name-list", [])
            print(Style.Bold(Style.CYAN("\nAvailable Agents:")))
            for a in agents:
                prefix = "*" if a == self.active_agent_name else " "
                print(f" {prefix} {a}")
            print()

        elif action == "switch":
            if len(args) < 2:
                print(Style.RED("Usage: /agent switch <name>"))
                return
            target = args[1]
            if target in self.isaa_tools.config.get("agents-name-list", []):
                self.active_agent_name = target
                # Reset session to default when switching agent to avoid ID mismatch
                self.active_session_id = "default"
                self._save_state()
                print(Style.GREEN(f"Switched to agent '{target}'"))
                await self._print_status_dashboard()
            else:
                print(Style.RED(f"Agent '{target}' not found."))

    async def _cmd_session(self, args: List[str]):
        agent = await self.get_active_agent()
        if not hasattr(agent, "session_manager"):
            print(Style.RED("This agent does not support session management."))
            return

        if not args:
            print(Style.YELLOW("Usage: /session <list|new|switch> [args]"))
            return

        action = args[0]

        if action == "list":
            sessions = agent.session_manager.sessions
            print(Style.Bold(Style.CYAN(f"\nSessions for {self.active_agent_name}:")))
            for sid, sess in sessions.items():
                active_mark = "*" if sid == self.active_session_id else " "
                # Try to get message count
                msg_count = len(sess.history) if hasattr(sess, "history") else 0
                print(f" {active_mark} {Style.YELLOW(sid):<20} {Style.GREY(f'({msg_count} msgs)')}")
            print()

        elif action == "new":
            if len(args) < 2:
                print(Style.RED("Usage: /session new <name>"))
                return
            name = args[1]
            await agent.session_manager.get_or_create(name)
            self.active_session_id = name
            print(Style.GREEN(f"Created and switched to session '{name}'"))

        elif action == "switch":
            if len(args) < 2:
                print(Style.RED("Usage: /session switch <id>"))
                return
            sid = args[1]
            if sid in agent.session_manager.sessions:
                self.active_session_id = sid
                print(Style.GREEN(f"Switched to session '{sid}'"))
            else:
                print(Style.RED(f"Session '{sid}' not found. Use '/session list'."))

    async def _cmd_feature(self, args: List[str]):
        if len(args) < 2:
            print(Style.YELLOW("Usage: /feature <lsp|docker> <on|off>"))
            return

        feature = args[0]
        state = args[1].lower() == "on"

        agent = await self.get_active_agent()

        changed = False
        if feature == "lsp":
            agent.amd.enable_lsp = state
            changed = True
        elif feature == "docker":
            agent.amd.enable_docker = state
            changed = True

        if changed:
            # Need to update the active session's config potentially
            session = agent.session_manager.get(self.active_session_id)
            if session:
                if feature == "lsp": session.enable_lsp = state
                if feature == "docker": session.enable_docker = state

            status_color = Style.GREEN if state else Style.GREY
            print(f"Feature {Style.CYAN(feature)} is now {status_color(args[1].upper())}")
        else:
            print(Style.RED(f"Unknown feature: {feature}"))

    async def _cmd_vfs_show(self):
        agent = await self.get_active_agent()
        session = agent.session_manager.get(self.active_session_id)
        if not session or not hasattr(session, "vfs"):
            print(Style.RED("No VFS available in current session."))
            return

        print(Style.Bold(Style.BLUE(f"\n📂 VFS Structure (Session: {self.active_session_id})")))
        print(Style.GREY("------------------------------------------------"))
        # Accessing private _structure or using ls tool logic
        try:
            # We construct a tree from the files dict
            tree = {}
            for path in session.vfs.files.keys():
                parts = path.strip("/").split("/")
                current = tree
                for part in parts:
                    current = current.setdefault(part, {})
            print_tree(tree)
        except Exception as e:
            print(Style.RED(f"Error reading VFS: {e}"))
        print()

    async def _cmd_context_stats(self):
        agent = await self.get_active_agent()
        print(Style.Bold(Style.BLUE("\n📊 Context Overview")))

        # Use helper from flow agent
        try:
            metrics = await agent.context_overview(self.active_session_id, print_visual=False)
            if not metrics:
                print(Style.RED("Could not calculate metrics."))
                return

            print_columns(
                Style.Underline("Component"),
                Style.Underline("Tokens"),
                Style.Underline("% of Window")
            )

            total = metrics.get('total', 0)
            limit = metrics.get('limit', 128000)

            def print_row(name, key):
                val = metrics.get(key, 0)
                pct = (val / limit) * 100
                print_columns(
                    Style.CYAN(name),
                    format_tokens(val),
                    f"{pct:.1f}%"
                )

            print_row("System Prompt", "system_prompt")
            print_row("Tools (Defs)", "tool_definitions")
            print_row("VFS Context", "vfs_context")
            print_row("Chat History", "history")
            print(Style.GREY("-" * 60))
            print_columns(
                Style.Bold("TOTAL"),
                format_tokens(total, limit),
                Style.Bold(f"{(total / limit) * 100:.1f}%")
            )

        except Exception as e:
            print(Style.RED(f"Error fetching stats: {e}"))

    async def _print_status_dashboard(self):
        """Prints the 3-column dashboard."""
        agent = await self.get_active_agent()
        session = agent.session_manager.get(self.active_session_id)

        # Column 1: Agent Info
        features = []
        if agent.amd.enable_docker: features.append("🐳 Docker")
        if agent.amd.enable_lsp: features.append("🧠 LSP")
        feat_str = ", ".join(features) if features else "Basic"

        c1 = f"{Style.Bold('🤖 AGENT'):<20}\n" \
             f"Name: {Style.CYAN(agent.amd.name):<20}\n" \
             f"Model: {Style.GREY(agent.amd.fast_llm_model.split('/')[-1]):<20}\n" \
             f"Feat: {feat_str:<20}"

        # Column 2: Session Info
        msg_count = len(session._chat_session.history) if session else 0
        c2 = f"{Style.Bold('💬 SESSION'):<20}\n" \
             f"ID: {Style.YELLOW(self.active_session_id):<20}\n" \
             f"Msgs: {msg_count:<20}\n" \
             f"Cost: ${agent.total_cost:.4f}"

        # Column 3: Environment
        cwd_trunc = str(__init_cwd__)
        if len(cwd_trunc) > 20: cwd_trunc = "..." + cwd_trunc[-17:]

        c3 = f"{Style.Bold('🌍 ENV'):<20}\n" \
             f"Path: {cwd_trunc:<20}\n" \
             f"VFS: {'Active' if session else 'Inactive':<20}\n" \
             f"Time: {time.strftime('%H:%M'):<20}"

        print(Style.GREY("=" * shutil.get_terminal_size().columns))
        # Split lines to print properly
        lines1 = c1.split('\n')
        lines2 = c2.split('\n')
        lines3 = c3.split('\n')

        for i in range(max(len(lines1), len(lines2), len(lines3))):
            l1 = lines1[i] if i < len(lines1) else ""
            l2 = lines2[i] if i < len(lines2) else ""
            l3 = lines3[i] if i < len(lines3) else ""
            print_columns(l1, l2, l3)
        print(Style.GREY("=" * shutil.get_terminal_size().columns))

    def _print_help(self):
        print(Style.Bold("\nISAA V3 Commands:"))
        print_columns("/agent switch <name>", "Switch active agent", "")
        print_columns("/session list", "List sessions", "/session new <name>")
        print_columns("/feature <feat> <on/off>", "Toggle LSP/Docker", "")
        print_columns("/vfs show", "Show VFS Tree", "")
        print_columns("/context stats", "Show Token Usage", "")
        print_columns("/status", "Show Dashboard", "")
        print_columns("/clear", "Clear screen", "/quit")
        print()

    async def _handle_shell(self, cmd):
        print(Style.GREY(f"executing: {cmd}..."))
        os.system(cmd)

    # --- EXECUTION LOGIC ---

    async def _handle_agent_interaction(self, user_prompt: str):
        agent = await self.get_active_agent()

        # Callback for "Fading Out" effect
        async def progress_handler(event):
            # Check event type for intermediate steps
            if hasattr(event, "event_type"):
                msg = ""
                if event.event_type == "tool_call":
                    msg = f"🔧 Tool: {event.tool_name}"
                elif event.event_type == "reasoning_loop":
                    msg = f"🤔 Thinking (Step {event.metadata.get('loop_number', '?')})"

                if msg:
                    # Print in grey/dimmed immediately
                    print(Style.GREY(f"  {msg}"))

        try:
            # Using progress_callback to intercept steps
            # FlowAgent V2 usually supports this via kwargs or set_progress_callback

            # Temporary override callback just for this run if possible,
            # or pass it if a_run supports it. Based on provided file, a_run has progress_callback in __init__
            # or the printer attaches to it. We'll rely on the agent instance having `progress_tracker`.

            # Since we can't easily inject the callback into a_run specifically for one call
            # without modifying the agent, we will attach it globally then detach.
            original_cb = agent.progress_tracker.progress_callback
            agent.progress_tracker.progress_callback = progress_handler

            response = await agent.a_run(
                query=user_prompt,
                session_id=self.active_session_id,
                # intermediate_callback isn't exactly the progress event, but can be used for streaming
            )

            # Restore
            agent.progress_tracker.progress_callback = original_cb

            print(Style.GREEN("assistant") + ": " + str(response))
            print(Style.GREY("_" * 40))

        except Exception as e:
            print(Style.RED(f"Agent Error: {e}"))


# --- ENTRY POINT ---

async def run(app, *args):
    cli_app = get_app("isaa_cli_instance")
    cli = SimpleIsaaCLI(cli_app)
    await cli.run()


if __name__ == "__main__":
    asyncio.run(run(None))
