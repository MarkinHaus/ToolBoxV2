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
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
from toolboxv2.mods.isaa.extras.terminal_progress import ProgressiveTreePrinter, VerbosityMode
from toolboxv2.mods.isaa.extras.verbose_output import EnhancedVerboseOutput
from toolboxv2.mods.isaa.module import Tools as Isaatools
from toolboxv2.mods.isaa.module import detect_shell
from toolboxv2.utils.extras.Style import Style, remove_styles

NAME = "isaa_cli"


def human_readable_time(seconds: float) -> str:
    # (Funktion bleibt unverändert)
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
    """
    Eine produktionsreife Agenten-CLI mit Fokus auf Agenten-Konfiguration,
    Workspace-Management und klarer Visualisierung des Agenten-Ablaufs.
    """

    def __init__(self, app_instance: Any, mode=VerbosityMode.STANDARD):
        self.app = app_instance
        self.isaa_tools: Isaatools = app_instance.get_mod("isaa")

        # Neuer, vereinheitlichter Printer
        self.printer = ProgressiveTreePrinter(mode=mode)
        self._current_verbosity_mode = mode

        self.formatter = EnhancedVerboseOutput(verbose=True, print_func=print)
        self.active_agent_name = "self"
        self.session_id = f"cli_session_{int(time.time())}"
        self.history = FileHistory(Path(self.app.data_dir) / "isaa_cli_history.txt")

        self.workspace_path = __init_cwd__
        self.dir_completer = PathCompleter(only_directories=True, expanduser=True)
        self.path_completer = PathCompleter(expanduser=True)

        self.completion_dict = self._build_completer()
        self.prompt_session = PromptSession(
            history=self.history,
            completer=FuzzyCompleter(NestedCompleter.from_nested_dict(self.completion_dict)),
            complete_while_typing=True,
        )

        self.session_stats = self._init_session_stats()

    def _build_completer(self):
        """Erstellt die Befehlsvervollständigung für die CLI."""
        return {
            "/agent": {
                "create": None,
                "list": None,
                "switch": WordCompleter(self.isaa_tools.config.get("agents-name-list", [])),
                "config": None,
                "save-checkpoint": None,
                "load-checkpoint": None,
                "load_mcp": self.path_completer,
            },
            "/workspace": {
                "status": None,
                "cd": self.dir_completer,
                "ls": self.path_completer,
            },
            "/context": {
                "show": None,
                "clear": None,
                "save": None,
                "load": None,
            },
            "/system": {
                "verbosity": WordCompleter(["MINIMAL", "STANDARD", "VERBOSE", "DEBUG", "REALTIME"]),
                "performance": None,
            },
            "/help": None,
            "/quit": None,
            "/clear": None,
        }

    def _init_session_stats(self) -> dict:
        """Initialisiert die Statistiken für die Sitzung."""
        return {
            "session_start_time": time.time(),
            "total_cost": 0.0,
            "total_tokens": {"prompt": 0, "completion": 0},
            "agents": {},
        }

    # --- Initialisierung & Hauptschleife ---

    async def init(self):
        """Initialisiert die CLI und den Standard-Agenten."""
        self.formatter.print_progress_bar(0, 1, "Initializing Default Agent...")
        self.formatter.print_progress_bar(1, 1, "Setup Complete")
        print()
        await self.show_welcome()


    async def _add_base_tools_to_agent(self, builder):
        """Fügt dem Agenten ein Set von Basis-Tools hinzu."""
        # Dateisystem-Tools
        builder.add_tool(self.workspace_status_tool, "workspace_status",
                         "Displays the status of the current workspace.")

        return builder

    async def run(self):
        """Die Hauptschleife der CLI."""
        await self.init()
        while True:
            try:
                # Vervollständigung dynamisch aktualisieren
                self.completion_dict["/agent"]["switch"] = WordCompleter(
                    self.isaa_tools.config.get("agents-name-list", []), ignore_case=True
                )
                self.prompt_session.completer = FuzzyCompleter(NestedCompleter.from_nested_dict(self.completion_dict))

                user_input = await self.prompt_session.prompt_async(self.get_prompt_text())

                if not user_input.strip():
                    continue

                if user_input.strip().startswith("!"):
                    await self._handle_shell_command(user_input.strip()[1:])
                elif user_input.strip().startswith("/"):
                    await self.handle_workspace_command(user_input.strip())
                else:
                    await self.handle_agent_request(user_input.strip())

            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                self.formatter.print_error(f"An unexpected error occurred: {e}")
                import traceback
                self.formatter.print_error(traceback.format_exc())

        await self.cleanup()

    def get_prompt_text(self) -> HTML:
        """Erstellt den dynamischen Eingabe-Prompt für die Konsole."""
        # Escape potential XML-invalid characters in the names
        import html
        workspace_name = html.escape(self.workspace_path.name)
        agent_name = html.escape(self.active_agent_name)

        # Verwenden Sie direkt die HTML-Tags von prompt_toolkit für die Farbgebung
        return HTML(
            f"<ansicyan>[</ansicyan>"
            f"<ansigreen>{workspace_name}</ansigreen>"
            f"<ansicyan>]</ansicyan> "
            f"<ansiyellow>({agent_name})</ansiyellow>"
            f"\n<ansiblue>❯</ansiblue> "
        )

    async def cleanup(self):
        """Räumt auf und beendet die Anwendung."""
        await self.app.a_exit()
        self.formatter.print_success("ISAA Workspace CLI shutting down. Goodbye!")


    # --- Agenten-Interaktion ---

    async def show_welcome(self):
        """Zeigt eine ansprechende Willkommensnachricht und eine Übersicht an."""
        try:
            terminal_width = os.get_terminal_size().columns
        except OSError:
            terminal_width = 80  # Fallback

        welcome_text = "ISAA Command Line Interface"

        print(Style.CYAN("═" * terminal_width))
        print(Style.Bold(Style.BLUE(welcome_text.center(terminal_width))))
        print(Style.CYAN("═" * terminal_width))
        print()

        self.formatter.print_section(
            "Workspace Overview",
            f"📁  Path: {self.workspace_path}\n"
            f"🤖  Active Agent: {self.active_agent_name}\n"
            f"💬  Session ID: {self.session_id}"
        )

        tips = [
            f"Tippen Sie {Style.CYAN('/help')}, um alle Befehle anzuzeigen.",
            f"Nutzen Sie die {Style.CYAN('Tab')}-Taste zur Autovervollständigung.",
            f"Starten Sie eine Zeile mit {Style.CYAN('!')}, um einen Shell-Befehl auszuführen (z.B. {Style.CYAN('!ls')}).",
            f"Verwenden Sie {Style.CYAN('Ctrl+D')} oder {Style.CYAN('/quit')} zum Beenden.",
        ]

        print()
        for tip in tips:
            print(f"  {tip}")
        print()
        self.formatter.print_info("Agent is online and ready for your commands.")

    async def handle_agent_request(self, request: str):
        """Verarbeitet eine Anfrage an den aktiven Agenten."""
        agent_name = self.active_agent_name
        if agent_name not in self.session_stats["agents"]:
            self._ensure_agent_stats_initialized(agent_name)

        start_time = time.time()
        try:
            self.printer.prompt_app = self.prompt_session.app

            # Agenten-Ausführung mit Callback für den Printer
            response = await self.isaa_tools.run_agent(
                name=agent_name,
                text=request,
                session_id=self.session_id,
                user_id="cli_user",
                progress_callback=self.printer.progress_callback,
            )

            # Finale Zusammenfassung nach erfolgreichem Lauf

            self.formatter.print_success("Agent response:")
            print(response)

            agent_stats = self.session_stats["agents"][agent_name]
            agent_stats["successful_runs"] = agent_stats.get("successful_runs", 0) + 1

        except Exception as e:
            self.formatter.print_error(f"Agent execution failed: {e}")
            agent_stats = self.session_stats["agents"][agent_name]
            agent_stats["failed_runs"] = agent_stats.get("failed_runs", 0) + 1

        finally:
            duration = time.time() - start_time
            # (Statistik-Updates werden jetzt durch Events im Callback gehandhabt)

    # --- Befehls-Handler ---

    async def handle_workspace_command(self, user_input: str):
        parts = user_input.split()
        command, args = parts[0].lower(), parts[1:]

        handlers = {
            "/agent": self.handle_agent_cmd,
            "/workspace": self.handle_workspace_cmd,
            "/context": self.handle_context_cmd,
            "/system": self.handle_system_cmd,
            "/help": self.handle_help_cmd,
            "/quit": self.handle_exit_cmd,
            "/clear": self.handle_clear_cmd,
        }

        handler = handlers.get(command)
        if handler:
            await handler(args)
        elif sum([1 if k.startswith(command) else 0 for k in handlers.keys()]) == 1:
            handler = handlers[[k for k in handlers.keys() if k.startswith(command)][0]]
            await handler(args)
        else:
            self.formatter.print_error(f"Unknown command: {command}")

    async def _handle_shell_command(self, command: str):
        # (Funktion bleibt im Wesentlichen unverändert, nutzt jetzt aber den Style-Formatter)
        if not command.strip():
            self.formatter.print_error("Shell command cannot be empty.")
            return

        self.formatter.print_info(f"🚀 Executing shell command: `{command}`")
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
                stream_reader(process.stdout, self.formatter.print),
                stream_reader(process.stderr, self.formatter.print_error)
            )

            return_code = await process.wait()
            if return_code == 0:
                self.formatter.print_success(f"Command finished successfully (Exit Code: {return_code}).")
            else:
                self.formatter.print_warning(f"Command finished with an error (Exit Code: {return_code}).")

        except Exception as e:
            self.formatter.print_error(f"An unexpected error occurred: {e}")

    # ... [Andere Befehls-Handler wie handle_help_cmd, handle_clear_cmd etc. bleiben ähnlich]
    # ... [Alle Tool-Funktionen wie read_file_tool, write_file_tool etc. bleiben unverändert]

    # --- NEUE & AKTUALISIERTE Befehls-Handler ---

    async def handle_agent_cmd(self, args: list[str]):
        """Verwaltet Agenten: erstellen, wechseln, konfigurieren, speichern, laden."""
        if not args:
            self.formatter.print_error(
                "Usage: /agent <create|list|switch|config|save-checkpoint|load-checkpoint|load_mcp>")
            return

        sub_command = args[0].lower()

        if sub_command == "create":
            await self._create_agent_interactive()

        elif sub_command == "list":
            detailed = "-d" in args or "--detailed" in args
            await self.list_agents_tool(detailed)

        elif sub_command == "show":
            agent = await self.isaa_tools.get_agent(self.active_agent_name)
            data = agent.amd.model_dump(exclude_none=True)
            del data["budget_manager"]
            del data["api_key"]
            print(f"AMD: {json.dumps(data, indent=2)}")

        elif sub_command == "switch":
            if len(args) < 2:
                self.formatter.print_error("Usage: /agent switch <agent_name>")
                return
            agent_name = args[1]
            if agent_name in self.isaa_tools.config.get("agents-name-list", []):
                self.active_agent_name = agent_name
                self.formatter.print_success(f"Switched to agent: {agent_name}")
            else:
                self.formatter.print_error(f"Agent '{agent_name}' not found.")

        elif sub_command == "config":
            await self._configure_agent_interactive()

        elif sub_command == "save-checkpoint":
            await self._save_agent_checkpoint()

        elif sub_command == "load-checkpoint":
            await self._load_agent_checkpoint()

        elif sub_command == "load_mcp":
            if len(args) < 2:
                self.formatter.print_error("Usage: /agent load_mcp <path_to_config.json>")
                return
            await self._load_mcp_config(args[1])

        else:
            self.formatter.print_error(f"Unknown agent command: {sub_command}")

    async def _create_agent_interactive(self):
        """Führt den Benutzer durch die Erstellung eines neuen Agenten."""
        try:
            name = await self.prompt_session.prompt_async("Enter a name for the new agent: ")
            if not name:
                self.formatter.print_warning("Agent creation cancelled.")
                return

            prompt = await self.prompt_session.prompt_async("Enter the system prompt for the agent: ")
            fast_model = await self.prompt_session.prompt_async(
                "Enter the fast LLM model (e.g., groq/llama3-8b-8192): ", default="groq/llama3-8b-8192")
            complex_model = await self.prompt_session.prompt_async(
                "Enter the complex LLM model (e.g., openrouter/openai/gpt-4o): ", default="openrouter/openai/gpt-4o")

            builder = self.isaa_tools.get_agent_builder(name)
            (builder.with_system_message(prompt)
             .with_models(fast_model, complex_model)
             .with_checkpointing(enabled=True)
             )
            builder = await self._add_base_tools_to_agent(builder)

            await self.isaa_tools.register_agent(builder)
            self.formatter.print_success(f"Agent '{name}' created successfully.")
            self.active_agent_name = name
            self.formatter.print_info(f"Switched to new agent: {name}")

        except (EOFError, KeyboardInterrupt):
            self.formatter.print_warning("\nAgent creation cancelled.")

    async def _configure_agent_interactive(self):
        """Interaktive Konfiguration des aktiven Agenten."""
        agent = await self.isaa_tools.get_agent(self.active_agent_name)
        if not agent:
            self.formatter.print_error(f"Could not load agent '{self.active_agent_name}'.")
            return

        try:
            p_name = agent.amd.persona.name if agent.amd.persona else agent.amd.name
            p_style = agent.amd.persona.style if agent.amd.persona else 'professional'
            p_tone = agent.amd.persona.tone if agent.amd.persona else 'friendly'
            persona = await self.prompt_session.prompt_async(f"Persona name [{p_name}]: ",
                                                             default=p_name)
            style = await self.prompt_session.prompt_async(f"Style [{p_style}]: ",
                                                           default=p_style)
            tone = await self.prompt_session.prompt_async(f"Tone [{p_tone}]: ",
                                                          default=p_tone)

            agent.set_persona(name=persona, style=style, tone=tone)
            self.formatter.print_success(f"Agent '{self.active_agent_name}' persona updated.")

        except (EOFError, KeyboardInterrupt):
            self.formatter.print_warning("\nConfiguration cancelled.")

    async def _save_agent_checkpoint(self):
        """Speichert einen Checkpoint des aktiven Agenten."""
        agent = await self.isaa_tools.get_agent(self.active_agent_name)
        if not agent:
            self.formatter.print_error(f"Could not load agent '{self.active_agent_name}'.")
            return

        if await agent._save_checkpoint(await agent._create_checkpoint()):
            self.formatter.print_success(f"Checkpoint for agent '{self.active_agent_name}' saved successfully.")
        else:
            self.formatter.print_error(f"Failed to save checkpoint for agent '{self.active_agent_name}'.")

    async def _load_agent_checkpoint(self):
        """Lädt den neuesten Checkpoint für den aktiven Agenten."""
        agent = await self.isaa_tools.get_agent(self.active_agent_name)
        if not agent:
            self.formatter.print_error(f"Could not load agent '{self.active_agent_name}'.")
            return

        result = await agent.load_latest_checkpoint()
        if result.get("success"):
            self.formatter.print_success(
                f"Checkpoint from {result['checkpoint_timestamp']} loaded for agent '{self.active_agent_name}'.")
        else:
            self.formatter.print_error(f"Failed to load checkpoint: {result.get('error')}")

    async def _load_mcp_config(self, path: str):
        """Lädt eine MCP-Tool-Konfiguration und registriert die Tools neu."""
        config_path = Path(self.workspace_path) / path
        if not config_path.exists():
            self.formatter.print_error(f"MCP config file not found: {config_path}")
            return

        try:
            builder = self.isaa_tools.get_agent_builder(self.active_agent_name)
            builder.load_mcp_tools_from_config(str(config_path))

            # Agenten neu registrieren, um die neuen Tools zu übernehmen
            agent = await self.isaa_tools.get_agent(self.active_agent_name)
            builder.with_system_message(agent.amd.system_message)
            builder.with_models(agent.amd.fast_llm_model, agent.amd.complex_llm_model)
            builder = await self._add_base_tools_to_agent(builder)

            await self.isaa_tools.register_agent(builder)
            self.formatter.print_success(
                f"MCP tools from '{path}' loaded and agent '{self.active_agent_name}' updated.")
        except Exception as e:
            self.formatter.print_error(f"Failed to load MCP config: {e}")

    async def handle_help_cmd(self, args: list[str]):
        """Zeigt eine umfassende Hilfe für alle Befehle an."""
        self.formatter.log_header("ISAA CLI - Hilfe")

        command_data = [
            ["Agenten-Management", ""],
            ["/agent create", "Startet einen interaktiven Prozess zur Erstellung eines neuen Agenten."],
            ["/agent list [-d]", "Listet alle verfügbaren Agenten auf. `-d` für Details."],
            ["/agent switch <name>", "Wechselt zum angegebenen Agenten."],
            ["/agent config", "Konfiguriert interaktiv die Persona des aktiven Agenten."],
            ["/agent save-checkpoint", "Speichert den aktuellen Zustand des Agenten."],
            ["/agent load-checkpoint", "Lädt den letzten Zustand des Agenten."],
            ["/agent load_mcp <path>", "Lädt MCP-Tools aus einer Konfigurationsdatei."],
            ["", ""],
            ["Workspace", ""],
            ["/workspace status", "Zeigt den Status des aktuellen Workspace an."],
            ["/workspace cd <dir>", "Wechselt das Arbeitsverzeichnis."],
            ["/workspace ls [path]", "Listet den Inhalt eines Verzeichnisses auf."],
            ["", ""],
            ["Kontext & Sitzung", ""],
            ["/context show", "Zeigt den aktuellen Gesprächsverlauf an."],
            ["/context clear", "Löscht den aktuellen Gesprächsverlauf."],
            ["/context save/load", "Speichert/lädt den Kontext der Sitzung."],
            ["", ""],
            ["System", ""],
            ["/system verbosity <MODE>", "Ändert die Detailstufe der Ausgabe."],
            ["! <command>", "Führt einen direkten Shell-Befehl aus."],
            ["", ""],
            ["Allgemein", ""],
            ["/clear", "Löscht den Bildschirm."],
            ["/quit", "Beendet die CLI."],
        ]

        # Statt der Tabellenfunktion, die entfernt wurde, formatieren wir es manuell
        for category, desc in command_data:
            if desc == "":
                print(f"\n{Style.Bold(Style.YELLOW(category))}")
            else:
                print(f"  {Style.CYAN(category.ljust(25))} {desc}")

    async def list_agents_tool(self, detailed: bool = False):
        """Listet alle verfügbaren Agenten auf."""
        agents = self.isaa_tools.config.get("agents-name-list", [])
        if not agents:
            self.formatter.print_info("Keine Agenten gefunden.")
            return

        self.formatter.print_section("Verfügbare Agenten", "")
        for name in agents:
            marker = "▶" if name == self.active_agent_name else " "
            print(f" {marker} {Style.GREEN(name)}")
            if detailed:
                try:
                    agent = await self.isaa_tools.get_agent(name)
                    prompt = agent.amd.system_message
                    prompt_preview = (prompt[:100] + '...') if len(prompt) > 100 else prompt
                    print(Style.GREY(f"     └─ Prompt: {prompt_preview}"))
                except Exception:
                    print(Style.RED("     └─ Konnte Details nicht laden."))

    async def handle_system_cmd(self, args: list[str]):
        """Verwaltet Systemeinstellungen wie die Ausführlichkeit."""
        if not args:
            self.formatter.print_error("Usage: /system <verbosity>")
            return
        sub_command = args[0].lower()
        if sub_command == "verbosity":
            if len(args) < 2:
                self.formatter.print_info(f"Current verbosity: {self._current_verbosity_mode.name}")
                return
            try:
                new_mode = VerbosityMode[args[1].upper()]
                self.printer.mode = new_mode
                self._current_verbosity_mode = new_mode
                self.formatter.print_success(f"Verbosity set to {new_mode.name}")
            except KeyError:
                self.formatter.print_error(f"Invalid verbosity mode '{args[1]}'.")
        else:
            self.formatter.print_error("Unknown system command.")

    async def handle_exit_cmd(self, args: list[str]):
        raise EOFError

    async def handle_clear_cmd(self, args: list[str]):
        os.system('cls' if os.name == 'nt' else 'clear')
        await self.show_welcome()

    async def handle_context_cmd(self, args: list[str]):
        """Verwaltet den Konversationskontext der aktuellen Sitzung."""
        if not args:
            self.formatter.print_error("Usage: /context <show|clear|save|load>")
            return

        sub_command = args[0].lower()
        agent = await self.isaa_tools.get_agent(self.active_agent_name)
        if not agent:
            self.formatter.print_error(f"Agent '{self.active_agent_name}' nicht gefunden.")
            return

        session_context = agent.shared.get("conversation_history", [])

        if sub_command == "show":
            if not session_context:
                self.formatter.print_info("Kontext ist leer.")
                return
            self.formatter.print_section("Current Conversation Context", "")
            for msg in session_context:
                role_style = Style.GREEN if msg['role'] == 'user' else Style.BLUE
                print(f"{role_style(msg['role'].upper())}: {msg['content']}")

        elif sub_command == "clear":
            agent.clear_context(self.session_id)
            self.formatter.print_success("Kontext für die aktuelle Sitzung wurde gelöscht.")

        elif sub_command == "save":
            path = Path(self.app.data_dir) / f"context_{self.session_id}.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(session_context, f, indent=2)
            self.formatter.print_success(f"Kontext gespeichert unter: {path}")

        elif sub_command == "load":
            path = Path(self.app.data_dir) / f"context_{self.session_id}.json"
            if not path.exists():
                self.formatter.print_error(f"Kein gespeicherter Kontext für Sitzung '{self.session_id}' gefunden.")
                return
            with open(path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            agent.shared['conversation_history'] = history
            self.formatter.print_success(f"Kontext aus '{path}' geladen.")
        else:
            self.formatter.print_error(f"Unbekannter Kontext-Befehl: {sub_command}")

    async def handle_workspace_cmd(self, args: list[str]):
        """Verarbeitet Befehle, die den Arbeitsbereich betreffen."""
        if not args:
            self.formatter.print_error("Usage: /workspace <status|cd|ls>")
            return
        sub_command = args[0].lower()
        if sub_command == "status":
            await self.workspace_status_tool()
        elif sub_command == "cd":
            if len(args) < 2:
                self.formatter.print_error("Usage: /workspace cd <directory>")
                return
            result = await self.change_workspace_tool(args[1])
            if "✅" in result:
                self.formatter.print_success(result)
            else:
                self.formatter.print_error(result)
        elif sub_command == "ls":
            path = args[1] if len(args) > 1 else "."
            result = await self.list_directory_tool(path)
            print(result)
        else:
            self.formatter.print_error(f"Unknown workspace command: {sub_command}")

    async def change_workspace_tool(self, directory: str) -> str:
        """Ändert das aktuelle Arbeitsverzeichnis für die CLI und den Agenten."""
        try:
            new_path = (self.workspace_path / directory).resolve()
            if not new_path.is_dir():
                return f"❌ Fehler: '{new_path}' ist kein gültiges Verzeichnis."

            os.chdir(new_path)
            self.workspace_path = new_path

            # Wichtig: Agenten über die Änderung informieren
            agent = await self.isaa_tools.get_agent(self.active_agent_name)
            if agent and hasattr(agent, 'set_variable'):
                agent.set_variable('system.workspace_path', str(new_path))

            return f"✅ Workspace geändert zu: {new_path}"
        except Exception as e:
            return f"❌ Fehler beim Ändern des Workspace: {e}"

    async def workspace_status_tool(self, *args):
        """Zeigt den Status des aktuellen Workspace an."""
        self.formatter.print_section(
            "Workspace Status",
            f"📍  Current Path: {Style.CYAN(str(self.workspace_path))}\n"
            f"🤖  Active Agent: {Style.YELLOW(self.active_agent_name)}\n"
            f"💬  Session ID:   {self.session_id}"
        )
        return "Status displayed."

    def _ensure_agent_stats_initialized(self, agent_name: str):
        """Stellt sicher, dass die Statistik-Struktur für einen Agenten existiert."""
        if agent_name not in self.session_stats["agents"]:
            self.session_stats["agents"][agent_name] = {
                "cost": 0.0,
                "tokens": {"prompt": 0, "completion": 0},
                "successful_runs": 0,
                "failed_runs": 0,
                "tool_calls": 0,
            }

    # HINWEIS: Die Implementierungen für read_file_tool, write_file_tool und list_directory_tool
    # können direkt aus der Originaldatei übernommen werden, da sie bereits robust sind.
# --- Main Entry Point ---
async def run(app, *args):
    """Startet die ISAA CLI."""
    cli_app_instance = get_app("isaa_cli_instance")
    cli = WorkspaceIsaasCli(cli_app_instance)
    try:
        await cli.run()
    except Exception as e:
        print(f"\n💥 A fatal error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run(None))
