"""
ISAA Host CLI v4 - The Multi-Agent Host System
===============================================

A production-ready CLI that acts as a host system controlled by a "Self Agent".
Features:
- Global Rate Limiter configuration shared across all agents
- Self Agent with exclusive shell access and system management tools
- Multi-Agent registry with background task support
- Audio interface with F4 keybinding for voice input
- Skill sharing and agent binding capabilities
- Professional terminal UI using cli_printing

Author: ISAA Team
Version: 4.0.0
"""

import asyncio
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import FuzzyCompleter, NestedCompleter, WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout

# ToolBoxV2 Imports
from toolboxv2 import get_app

# ISAA Agent Imports
from toolboxv2.mods.isaa.base.Agent.builder import (
    AgentConfig,
    FlowAgentBuilder,
    RateLimiterConfig,
)
from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
from toolboxv2.mods.isaa.base.Agent.instant_data_vis import (
    visualize_data_terminal,
)
from toolboxv2.mods.isaa.base.AgentUtils import detect_shell
from toolboxv2.mods.isaa.module import Tools as IsaaTool

import html
import json
import sys
from prompt_toolkit import print_formatted_text, HTML
from prompt_toolkit.styles import Style


# =================== Helpers & Setup ===================

class PTColors:
    """Farb-Mapping für Prompt Toolkit HTML"""
    GREY = 'gray'
    WHITE = 'ansiwhite'
    GREEN = 'ansigreen'
    YELLOW = 'ansiyellow'
    CYAN = 'ansicyan'
    BLUE = 'ansiblue'
    RED = 'ansired'
    MAGENTA = 'ansimagenta'
    BRIGHT_WHITE = '#ffffff'
    BRIGHT_CYAN = '#00ffff'


class Colors:
    """Legacy Support falls anderer Code direkt Colors.RED aufruft"""
    RED = 'ansired'
    GREEN = 'ansigreen'
    YELLOW = 'ansiyellow'
    BLUE = 'ansiblue'
    CYAN = 'ansicyan'
    GREY = 'gray'
    RESET = ''
    BOLD = 'bold'


def esc(text: Any) -> str:
    """Escaped Text für HTML-Tags, verhindert Crash bei < oder > im Text"""
    return html.escape(str(text))


def c_print(*args, **kwargs):
    """Drop-in Replacement für print, nutzt prompt_toolkit"""
    # Konvertiert alles zu Strings und escaped es
    text = " ".join(str(a) for a in args)

    # Wenn bereits HTML Objekt, direkt drucken, sonst wrappen
    if len(text) == len(args) == 0:
        print()
    elif isinstance(args[0], HTML):
        print_formatted_text(*args, **kwargs)
    else:
        print_formatted_text(HTML(esc(text)), **kwargs)


# =================== Die 8 gewünschten Funktionen ===================

def print_box_header(title: str, icon: str = "ℹ", width: int = 76):
    """1. Header mit Icon und Titel"""
    print_formatted_text(HTML(""))  # Leere Zeile
    print_formatted_text(HTML(f"<style font-weight='bold'>{icon} {esc(title)}</style>"))
    print_formatted_text(HTML(f"<style fg='{PTColors.GREY}'>{'─' * width}</style>"))


def print_box_footer(width: int = 76):
    """2. Footer (einfacher Abschluss)"""
    print_formatted_text(HTML(""))


def print_box_content(text: str, style: str = "", width: int = 76, auto_wrap: bool = True):
    """3. Inhalt mit Icon-Mapping für Status"""
    style_config = {
        'success': {'icon': '✓', 'color': PTColors.GREEN},
        'error': {'icon': '✗', 'color': PTColors.RED},
        'warning': {'icon': '⚠', 'color': PTColors.YELLOW},
        'info': {'icon': 'ℹ', 'color': PTColors.BLUE},
    }

    safe_text = esc(text)
    if style in style_config:
        config = style_config[style]
        # Icon farbig, Text normal
        print_formatted_text(HTML(f"  <style fg='{config['color']}'>{config['icon']}</style> {safe_text}"))
    else:
        print_formatted_text(HTML(f"  {safe_text}"))


def print_status(message: str, status: str = "info"):
    """4. Statusmeldungen (für Logs, Progress, etc.)"""
    status_config = {
        'success': {'icon': '✓', 'color': PTColors.GREEN},
        'error': {'icon': '✗', 'color': PTColors.RED},
        'warning': {'icon': '⚠', 'color': PTColors.YELLOW},
        'info': {'icon': 'ℹ', 'color': PTColors.BLUE},
        'progress': {'icon': '⟳', 'color': PTColors.CYAN},
        'data': {'icon': '💾', 'color': PTColors.YELLOW},
        'configure': {'icon': '🔧', 'color': PTColors.YELLOW},
        'launch': {'icon': '🚀', 'color': PTColors.GREEN},
    }

    config = status_config.get(status, {'icon': '•', 'color': PTColors.WHITE})
    color_attr = f"fg='{config['color']}'" if config['color'] else ""

    print_formatted_text(HTML(f"<style {color_attr}>{config['icon']}</style> {esc(message)}"))


def print_separator(char: str = "─", width: int = 76):
    """5. Trennlinie"""
    print_formatted_text(HTML(f"<style fg='{PTColors.GREY}'>{char * width}</style>"))


def print_table_header(columns: list, widths: list):
    """6. Tabellenkopf"""
    header_parts = []
    for (name, _), width in zip(columns, widths):
        # Text fett und hellweiß
        header_parts.append(f"<style font-weight='bold' fg='{PTColors.BRIGHT_WHITE}'>{esc(name):<{width}}</style>")

    # Trenner in Cyan
    sep_parts = [f"<style fg='{PTColors.BRIGHT_CYAN}'>{'─' * w}</style>" for w in widths]

    joined_headers = " │ ".join(header_parts)
    joined_seps = f"<style fg='{PTColors.BRIGHT_CYAN}'>─┼─</style>".join(sep_parts)

    print_formatted_text(HTML(f"  {joined_headers}"))
    print_formatted_text(HTML(f"  {joined_seps}"))


def print_table_row(values: list, widths: list, styles: list = None):
    """7. Tabellenzeile mit spaltenweisen Farben"""
    if styles is None:
        styles = [""] * len(values)

    color_map = {
        'grey': PTColors.GREY,
        'white': PTColors.WHITE,
        'green': PTColors.GREEN,
        'yellow': PTColors.YELLOW,
        'cyan': PTColors.CYAN,
        'blue': PTColors.BLUE,
        'red': PTColors.RED,
    }

    row_parts = []
    for value, width, style in zip(values, widths, styles):
        safe_val = esc(str(value))
        color = color_map.get(style.lower(), '')

        if color:
            # Wir berechnen das Padding manuell, damit die Farbe nicht den Leerraum füllt
            padding = width - len(safe_val)
            padding_str = " " * max(0, padding)
            row_parts.append(f"<style fg='{color}'>{safe_val}</style>{padding_str}")
        else:
            row_parts.append(f"{safe_val:<{width}}")

    # Vertikale Linien in Grau
    sep = f" <style fg='{PTColors.GREY}'>│</style> "
    print_formatted_text(HTML(f"  {sep.join(row_parts)}"))


def print_code_block(code: str, language: str = "text", width: int = 76, show_line_numbers: bool = False):
    """8. Code Block mit Basic Syntax Highlighting"""
    lines = []

    # JSON Highlighting Logic
    if language.lower() == 'json':
        try:
            parsed = json.loads(code) if isinstance(code, str) else code
            formatted = json.dumps(parsed, indent=2)
            raw_lines = formatted.split('\n')
            for line in raw_lines:
                # Key-Highlighting (Cyan für Keys, Grün für Strings)
                safe_line = esc(line)
                if ':' in safe_line:
                    k, v = safe_line.split(':', 1)
                    lines.append(f"<style fg='{PTColors.CYAN}'>{k}</style>:{v}")
                else:
                    lines.append(safe_line)
        except:
            lines = [esc(l) for l in code.split('\n')]

    # YAML/ENV Highlighting Logic
    elif language.lower() in ['yaml', 'yml', 'env']:
        for line in code.split('\n'):
            safe_line = esc(line)
            if safe_line.strip().startswith('#'):
                lines.append(f"<style fg='{PTColors.GREY}'>{safe_line}</style>")
            elif ':' in safe_line:
                k, v = safe_line.split(':', 1)
                lines.append(f"<style fg='{PTColors.CYAN}'>{k}</style>:{v}")
            elif '=' in safe_line:
                k, v = safe_line.split('=', 1)
                lines.append(f"<style fg='{PTColors.CYAN}'>{k}</style>={v}")
            else:
                lines.append(safe_line)

    # Fallback / Markdown
    else:
        lines = [esc(l) for l in code.split('\n')]

    # Ausgabe
    for i, line in enumerate(lines, 1):
        if show_line_numbers:
            print_formatted_text(HTML(f"  <style fg='{PTColors.GREY}'>{i:3d}</style> {line}"))
        else:
            print_formatted_text(HTML(f"  {line}"))

# =============================================================================
# CONSTANTS & VERSION
# =============================================================================

VERSION = "4.0.0"
CLI_NAME = "ISAA Host"
NAME = "icli"
# Default Rate Limiter Configuration (shared across all agents)
DEFAULT_RATE_LIMITER_CONFIG = {
    "features": {
        "rate_limiting": True,
        "model_fallback": True,
        "key_rotation": True,
        "key_rotation_mode": "balance",
    },
    "api_keys": {},
    "fallback_chains": {
        "zglm/glm-4.7": [
            "zglm/glm-4.7-flash",
            "zglm/glm-4.7-flashx",
            "zai/glm-4.7-flash",
            "zai/glm-4.7-flashx",
        ],
    },
    "limits": {},
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class BackgroundTask:
    """Represents a background task running an agent."""

    task_id: str
    agent_name: str
    query: str
    task: asyncio.Task
    started_at: datetime = field(default_factory=datetime.now)
    status: str = "running"


@dataclass
class AgentInfo:
    """Information about a registered agent."""

    name: str
    created_at: datetime = field(default_factory=datetime.now)
    persona: str = "default"
    is_self_agent: bool = False
    has_shell_access: bool = False
    mcp_servers: list[str] = field(default_factory=list)
    bound_agents: list[str] = field(default_factory=list)


# =============================================================================
# ISAA HOST - MAIN CLASS
# =============================================================================


class ISAA_Host:
    """
    The ISAA Host System - A multi-agent host controlled by a Self Agent.

    Features:
    - Global rate limiter configuration shared across all agents
    - Self Agent with exclusive shell access
    - Background task management
    - Agent registry and lifecycle management
    - Audio input via F4 keybinding
    - Skill sharing between agents
    """

    def __init__(self, app_instance: Any = None):
        """Initialize the ISAA Host system."""
        self.app = app_instance or get_app("isaa-host")

        # Get ISAA Tools module - THE source of truth for agent management
        self.isaa_tools: 'IsaaTools' = self.app.get_mod("isaa")

        # Host state
        self.host_id = str(uuid.uuid4())[:8]
        self.started_at = datetime.now()

        # Global Rate Limiter Config (shared across all agents)
        self._rate_limiter_config = DEFAULT_RATE_LIMITER_CONFIG.copy()

        # Agent Registry (metadata only - actual instances via isaa_tools)
        self.agent_registry: dict[str, AgentInfo] = {}

        # Background Task Manager
        self.background_tasks: dict[str, BackgroundTask] = {}
        self._task_counter = 0

        # Session state
        self.active_agent_name = "self"
        self.active_session_id = "default"

        # File paths
        self.state_file = Path(self.app.appdata) / "icli" / "isaa_host_state.json"
        self.history_file = Path(self.app.appdata) / "icli" / "isaa_host_history.txt"
        self.rate_limiter_config_file = (
            Path(self.app.appdata) / "icli" / "rate_limiter_config.json"
        )

        if not (Path(self.app.appdata) / "icli" ).exists():
            (Path(self.app.appdata) / "icli").mkdir(parents=True, exist_ok=True)
        if not self.state_file.exists():
            self.state_file.touch(exist_ok=True)
        if not self.history_file.exists():
            self.history_file.touch(exist_ok=True)
        if not self.rate_limiter_config_file.exists():
            self.rate_limiter_config_file.touch(exist_ok=True)

        # Audio state
        self._audio_recording = False
        self._audio_buffer: list[bytes] = []
        self._last_transcription: str | None = None

        # Prompt Toolkit setup
        self.history = FileHistory(str(self.history_file))
        self.key_bindings = self._create_key_bindings()
        self.prompt_session: PromptSession | None = None

        # Self Agent initialization flag
        self._self_agent_initialized = False

        # Load persisted state
        self._load_rate_limiter_config()
        self._load_state()

    # =========================================================================
    # RATE LIMITER CONFIG MANAGEMENT
    # =========================================================================

    def _load_rate_limiter_config(self):
        """Load rate limiter config from file if exists."""
        if self.rate_limiter_config_file.exists():
            try:
                with open(self.rate_limiter_config_file, encoding="utf-8") as f:
                    loaded = json.load(f)
                    for key in DEFAULT_RATE_LIMITER_CONFIG:
                        if key not in loaded:
                            loaded[key] = DEFAULT_RATE_LIMITER_CONFIG[key]
                    self._rate_limiter_config = loaded
            except Exception as e:
                print_status(f"Failed to load rate limiter config: {e}", "warning")

    def _save_rate_limiter_config(self):
        """Save rate limiter config to file."""
        try:
            self.rate_limiter_config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.rate_limiter_config_file, "w", encoding="utf-8") as f:
                json.dump(self._rate_limiter_config, f, indent=2)
        except Exception as e:
            print_status(f"Failed to save rate limiter config: {e}", "error")

    def get_rate_limiter_config(self) -> dict:
        """Get the global rate limiter configuration."""
        return self._rate_limiter_config.copy()

    def update_rate_limiter_config(self, updates: dict):
        """Update the global rate limiter configuration."""
        for key, value in updates.items():
            if key in self._rate_limiter_config:
                if isinstance(self._rate_limiter_config[key], dict) and isinstance(
                    value, dict
                ):
                    self._rate_limiter_config[key].update(value)
                else:
                    self._rate_limiter_config[key] = value
        self._save_rate_limiter_config()

    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================

    def _load_state(self):
        """Load persisted host state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, encoding="utf-8") as f:
                    state = json.load(f)
                    self.active_agent_name = state.get("active_agent", "self")
                    self.active_session_id = state.get("active_session", "default")
                    for name, info in state.get("agent_registry", {}).items():
                        self.agent_registry[name] = AgentInfo(
                            name=name,
                            persona=info.get("persona", "default"),
                            is_self_agent=info.get("is_self_agent", False),
                            has_shell_access=info.get("has_shell_access", False),
                            mcp_servers=info.get("mcp_servers", []),
                            bound_agents=info.get("bound_agents", []),
                        )
            except Exception as e:
                print_status(f"Failed to load state: {e}", "warning")

    def _save_state(self):
        """Save host state to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "active_agent": self.active_agent_name,
                "active_session": self.active_session_id,
                "agent_registry": {
                    name: {
                        "persona": info.persona,
                        "is_self_agent": info.is_self_agent,
                        "has_shell_access": info.has_shell_access,
                        "mcp_servers": info.mcp_servers,
                        "bound_agents": info.bound_agents,
                    }
                    for name, info in self.agent_registry.items()
                },
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print_status(f"Failed to save state: {e}", "error")

    # =========================================================================
    # KEY BINDINGS (AUDIO)
    # =========================================================================

    def _create_key_bindings(self) -> KeyBindings:
        """Create prompt_toolkit key bindings."""
        kb = KeyBindings()

        @kb.add("f4")
        def _(event):
            """Toggle audio recording with F4."""
            asyncio.create_task(self._toggle_audio_recording())

        @kb.add("f5")
        def _(event):
            """Show status dashboard with F5."""
            asyncio.create_task(self._print_status_dashboard())

        return kb

    async def _toggle_audio_recording(self):
        """Toggle audio recording state."""
        if self._audio_recording:
            self._audio_recording = False
            print_status("Processing audio...", "progress")
            await self._process_recorded_audio()
        else:
            self._audio_recording = True
            self._audio_buffer = []
            print_status("🎤 Recording... Press F4 to stop", "info")
            asyncio.create_task(self._record_audio())

    async def _record_audio(self):
        """Record audio from microphone."""
        try:
            import numpy as np
            import sounddevice as sd

            sample_rate = 16000
            channels = 1

            def callback(indata, frames, time, status):
                if self._audio_recording:
                    self._audio_buffer.append(indata.copy())

            with sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="int16",
                callback=callback,
            ):
                while self._audio_recording:
                    await asyncio.sleep(0.1)

        except ImportError:
            print_status(
                "Audio requires 'sounddevice'. Install: pip install sounddevice", "error"
            )
            self._audio_recording = False
        except Exception as e:
            print_status(f"Audio recording error: {e}", "error")
            self._audio_recording = False

    async def _process_recorded_audio(self):
        """Process recorded audio and transcribe."""
        if not self._audio_buffer:
            print_status("No audio recorded", "warning")
            return

        try:
            import numpy as np

            from toolboxv2.mods.isaa.base.audio_io.Stt import STTConfig, transcribe

            audio_data = np.concatenate(self._audio_buffer)
            audio_bytes = audio_data.tobytes()

            result = transcribe(audio_bytes, config=STTConfig(language="de"))

            if result.text:
                print_status(f"Transcribed: {result.text}", "success")
                self._last_transcription = result.text
            else:
                print_status("No speech detected", "warning")

        except ImportError as e:
            print_status(f"Audio processing requires additional packages: {e}", "error")
        except Exception as e:
            print_status(f"Audio processing error: {e}", "error")
        finally:
            self._audio_buffer = []

    # =========================================================================
    # SELF AGENT INITIALIZATION
    # =========================================================================

    async def _init_self_agent(self):
        """Initialize the Self Agent with exclusive capabilities."""
        if self._self_agent_initialized:
            return

        print_status("Initializing Self Agent...", "progress")

        builder = self.isaa_tools.get_agent_builder(
            name="self", add_base_tools=True, with_dangerous_shell=True
        )

        self._apply_rate_limiter_to_builder(builder)
        self._register_self_agent_tools(builder)

        await self.isaa_tools.register_agent(builder)

        self.agent_registry["self"] = AgentInfo(
            name="self",
            persona="Host Administrator",
            is_self_agent=True,
            has_shell_access=True,
        )

        self._self_agent_initialized = True
        print_status("Self Agent initialized", "success")

    def _apply_rate_limiter_to_builder(self, builder: FlowAgentBuilder):
        """Apply global rate limiter config to a builder."""
        features = self._rate_limiter_config.get("features", {})

        builder.with_rate_limiter(
            enable_rate_limiting=features.get("rate_limiting", True),
            enable_model_fallback=features.get("model_fallback", True),
            enable_key_rotation=features.get("key_rotation", True),
            key_rotation_mode=features.get("key_rotation_mode", "balance"),
        )

        for provider, keys in self._rate_limiter_config.get("api_keys", {}).items():
            for key in keys:
                builder.add_api_key(provider, key)

        for primary, fallbacks in self._rate_limiter_config.get(
            "fallback_chains", {}
        ).items():
            builder.add_fallback_chain(primary, fallbacks)

        for model, limits in self._rate_limiter_config.get("limits", {}).items():
            builder.set_model_limits(model, **limits)

    def _register_self_agent_tools(self, builder: FlowAgentBuilder):
        """Register exclusive tools for the Self Agent."""

        host_ref = self

        # ===== AGENT MANAGEMENT TOOLS =====

        async def cli_spawn_agent(
            name: str,
            persona: str = "general assistant",
            model: str | None = None,
            background: bool = False,
        ) -> str:
            """
            Spawn a new agent with the given name and persona.

            Args:
                name: Unique name for the agent
                persona: Description of the agent's role/personality
                model: Optional model override
                background: If True, agent runs in background mode

            Returns:
                Status message about agent creation
            """
            return await host_ref._tool_spawn_agent(name, persona, model, background)

        async def cli_list_agents() -> str:
            """
            List all registered agents and their status.

            Returns:
                Formatted list of all agents with their status
            """
            return await host_ref._tool_list_agents()

        # ===== TASK MANAGEMENT TOOLS =====

        async def cli_delegate(
            agent_name: str,
            task: str,
            wait: bool = True,
            session_id: str = "default",
        ) -> str:
            """
            Delegate a task to another agent.

            Args:
                agent_name: Name of the agent to delegate to
                task: The task/query to execute
                wait: If True, wait for result. If False, run in background
                session_id: Session ID for the task

            Returns:
                Result of the task or background task ID
            """
            return await host_ref._tool_delegate(agent_name, task, wait, session_id)

        async def cli_stop_agent(agent_name: str) -> str:
            """
            Stop all running tasks for an agent.

            Args:
                agent_name: Name of the agent to stop

            Returns:
                Status message
            """
            return await host_ref._tool_stop_agent(agent_name)

        async def cli_task_status(task_id: str | None = None) -> str:
            """
            Check status of background tasks.

            Args:
                task_id: Specific task ID or None for all tasks

            Returns:
                Task status information
            """
            return await host_ref._tool_task_status(task_id)

        # ===== SKILL SHARING TOOLS =====

        async def cli_teach_skill(
            target_agent: str,
            skill_name: str,
            instruction: str,
            triggers: list[str],
        ) -> str:
            """
            Teach a skill to an agent.

            Args:
                target_agent: Name of the agent to teach
                skill_name: Name for the skill
                instruction: Step-by-step instructions for the skill
                triggers: Keywords that activate this skill

            Returns:
                Status message
            """
            return await host_ref._tool_teach_skill(
                target_agent, skill_name, instruction, triggers
            )

        async def cli_bind_agents(
            agent_a: str, agent_b: str, mode: str = "public"
        ) -> str:
            """
            Bind two agents for data sharing.

            Args:
                agent_a: First agent name
                agent_b: Second agent name
                mode: Binding mode ('public' or 'private')

            Returns:
                Status message
            """
            return await host_ref._tool_bind_agents(agent_a, agent_b, mode)

        # ===== SYSTEM TOOLS =====

        def cli_shell(command: str) -> str:
            """
            Execute a shell command. CAUTION: Use responsibly.

            Args:
                command: Shell command to execute

            Returns:
                Command output as JSON
            """
            return host_ref._tool_shell(command)

        async def cli_mcp_connect(
            server_name: str,
            command: str,
            args: list[str],
            target_agent: str | None = None,
        ) -> str:
            """
            Connect an MCP server to an agent.

            Args:
                server_name: Name for the MCP server
                command: Command to start the MCP server
                args: Arguments for the MCP server
                target_agent: Agent to add MCP to (None = create new agent)

            Returns:
                Status message
            """
            return await host_ref._tool_mcp_connect(
                server_name, command, args, target_agent
            )

        async def cli_update_agent_config(agent_name: str, config_updates: dict) -> str:
            """
            Update an agent's configuration (saved for next restart).

            Args:
                agent_name: Agent to update
                config_updates: Configuration updates (dict)

            Returns:
                Status message
            """
            return await host_ref._tool_update_agent_config(agent_name, config_updates)

        # ===== REGISTER ALL TOOLS =====
        builder.add_tool(
            cli_spawn_agent,
            "spawnAgent",
            "Create a new agent with specified name and persona",
            category=["agent_management"],
        )
        builder.add_tool(
            cli_list_agents,
            "listAgents",
            "List all registered agents and their status",
            category=["agent_management"],
        )
        builder.add_tool(
            cli_delegate,
            "delegate",
            "Delegate a task to another agent",
            category=["task_management"],
        )
        builder.add_tool(
            cli_stop_agent,
            "stopAgent",
            "Stop running tasks for an agent",
            category=["task_management"],
        )
        builder.add_tool(
            cli_task_status,
            "taskStatus",
            "Check status of background tasks",
            category=["task_management"],
        )
        builder.add_tool(
            cli_teach_skill,
            "teachSkill",
            "Teach a skill to an agent",
            category=["skill_sharing"],
        )
        builder.add_tool(
            cli_bind_agents,
            "bindAgents",
            "Bind two agents for data sharing",
            category=["agent_binding"],
        )
        builder.add_tool(
            cli_shell,
            "shell",
            f"Execute shell command in {detect_shell()[0]}",
            category=["system"],
        )
        builder.add_tool(
            cli_mcp_connect,
            "mcpConnect",
            "Connect an MCP server to an agent",
            category=["mcp"],
        )
        builder.add_tool(
            cli_update_agent_config,
            "updateAgentConfig",
            "Update agent configuration for next restart",
            category=["agent_management"],
        )

    # =========================================================================
    # TOOL IMPLEMENTATIONS
    # =========================================================================

    async def _tool_spawn_agent(
        self, name: str, persona: str, model: str | None = None, background: bool = False
    ) -> str:
        """Implementation: Spawn a new agent."""
        try:
            if name in self.agent_registry:
                return f"Agent '{name}' already exists. Use a different name."

            builder = self.isaa_tools.get_agent_builder(
                name=name, add_base_tools=True, with_dangerous_shell=False
            )

            self._apply_rate_limiter_to_builder(builder)
            builder.config.system_message = (
                f"You are {persona}. Act according to this role."
            )

            if model:
                builder.with_models(model, model)

            await self.isaa_tools.register_agent(builder)

            self.agent_registry[name] = AgentInfo(
                name=name, persona=persona, is_self_agent=False, has_shell_access=False
            )
            self._save_state()

            return f"✓ Agent '{name}' spawned with persona: {persona}"

        except Exception as e:
            return f"✗ Failed to spawn agent: {e}"

    async def _tool_list_agents(self) -> str:
        """Implementation: List all agents."""
        isaa_agents: list[str] = self.isaa_tools.config.get("agents-name-list", [])

        result = ["=== Registered Agents ===\n"]

        for agent_name in isaa_agents:
            info = self.agent_registry.get(agent_name, AgentInfo(name=agent_name))

            instance_key = f"agent-instance-{agent_name}"
            is_active = instance_key in self.isaa_tools.config

            bg_tasks = sum(
                1
                for t in self.background_tasks.values()
                if t.agent_name == agent_name and t.status == "running"
            )

            status = "🟢 Active" if is_active else "⚪ Idle"
            shell_icon = "🔓" if info.has_shell_access else ""
            self_icon = "👑" if info.is_self_agent else ""

            result.append(
                f"  {self_icon}{agent_name} {shell_icon}\n"
                f"    Status: {status}\n"
                f"    Persona: {info.persona}\n"
                f"    Background Tasks: {bg_tasks}\n"
                f"    Bound To: {', '.join(info.bound_agents) or 'None'}\n"
            )

        if not isaa_agents:
            result.append("  No agents registered.\n")

        return "\n".join(result)

    async def _tool_delegate(
        self, agent_name: str, task: str, wait: bool, session_id: str
    ) -> str:
        """Implementation: Delegate task to agent."""
        try:
            agent = await self.isaa_tools.get_agent(agent_name)

            if wait:
                result = await agent.a_run(query=task, session_id=session_id)
                return str(result)
            else:
                self._task_counter += 1
                task_id = f"task_{self._task_counter}_{agent_name}"

                async def run_task():
                    try:
                        result = await agent.a_run(query=task, session_id=session_id)
                        self.background_tasks[task_id].status = "completed"
                        return result
                    except asyncio.CancelledError:
                        self.background_tasks[task_id].status = "cancelled"
                        raise
                    except Exception:
                        self.background_tasks[task_id].status = "failed"
                        raise

                async_task = asyncio.create_task(run_task())

                self.background_tasks[task_id] = BackgroundTask(
                    task_id=task_id,
                    agent_name=agent_name,
                    query=task[:100],
                    task=async_task,
                )

                return f"✓ Background task started: {task_id}"

        except Exception as e:
            return f"✗ Delegation failed: {e}"

    async def _tool_stop_agent(self, agent_name: str) -> str:
        """Implementation: Stop agent tasks."""
        stopped = 0

        for _, bg_task in list(self.background_tasks.items()):
            if bg_task.agent_name == agent_name and bg_task.status == "running":
                bg_task.task.cancel()
                bg_task.status = "cancelled"
                stopped += 1

        try:
            agent = await self.isaa_tools.get_agent(agent_name)
            # Cancel all active executions for this agent
            for exec_info in agent.list_executions():
                exec_id = exec_info.get("run_id")
                if exec_id:
                    await agent.cancel_execution(exec_id)
        except Exception:
            pass

        return f"✓ Stopped {stopped} task(s) for agent '{agent_name}'"

    async def _tool_task_status(self, task_id: str | None = None) -> str:
        """Implementation: Check task status."""
        if task_id and task_id in self.background_tasks:
            t = self.background_tasks[task_id]
            return (
                f"Task: {t.task_id}\n"
                f"Agent: {t.agent_name}\n"
                f"Query: {t.query}\n"
                f"Status: {t.status}\n"
                f"Started: {t.started_at.isoformat()}"
            )

        result = ["=== Background Tasks ===\n"]

        for tid, t in self.background_tasks.items():
            result.append(
                f"  [{t.status.upper()}] {tid}\n"
                f"    Agent: {t.agent_name}\n"
                f"    Query: {t.query[:50]}...\n"
            )

        if not self.background_tasks:
            result.append("  No background tasks.\n")

        return "\n".join(result)

    async def _tool_teach_skill(
        self, target_agent: str, skill_name: str, instruction: str, triggers: list[str]
    ) -> str:
        """Implementation: Teach skill to agent."""
        try:
            agent = await self.isaa_tools.get_agent(target_agent)
            session_id = "default"

            skill_id = f"custom_{skill_name}_{uuid.uuid4().hex[:6]}"
            skill_data = {
                "id": skill_id,
                "name": skill_name,
                "triggers": triggers,
                "instruction": instruction,
                "tools_used": [],
                "tool_groups": [],
                "source": "taught",
                "confidence": 0.8,
                "activation_threshold": 0.6,
                "success_count": 0,
                "failure_count": 0,
                "created_at": datetime.now().isoformat(),
                "last_used": None,
            }

            exec_engine = agent._get_execution_engine()
            success = exec_engine.skills_manager.import_skill(skill_data, overwrite=True)

            if success:
                return f"✓ Skill '{skill_name}' taught to agent '{target_agent}'"
            else:
                return f"✗ Failed to import skill to agent '{target_agent}'"

        except Exception as e:
            return f"✗ Failed to teach skill: {e}"

    async def _tool_bind_agents(
        self, agent_a: str, agent_b: str, mode: str = "public"
    ) -> str:
        """Implementation: Bind two agents."""
        try:
            agent_a_instance = await self.isaa_tools.get_agent(agent_a)
            agent_b_instance = await self.isaa_tools.get_agent(agent_b)

            await agent_a_instance.bind_manager.bind(partner=agent_b_instance, mode=mode)

            if (
                agent_a in self.agent_registry
                and agent_b not in self.agent_registry[agent_a].bound_agents
            ):
                self.agent_registry[agent_a].bound_agents.append(agent_b)
            if (
                agent_b in self.agent_registry
                and agent_a not in self.agent_registry[agent_b].bound_agents
            ):
                self.agent_registry[agent_b].bound_agents.append(agent_a)

            self._save_state()

            return f"✓ Agents '{agent_a}' and '{agent_b}' bound in '{mode}' mode"

        except Exception as e:
            return f"✗ Failed to bind agents: {e}"

    def _tool_shell(self, command: str) -> str:
        """Implementation: Execute shell command."""
        import shlex

        result = {"success": False, "output": "", "error": ""}

        try:
            tokens = shlex.split(command)

            for i, tok in enumerate(tokens):
                if tok in ("python", "python3"):
                    tokens[i] = sys.executable

            cmd_str = " ".join(shlex.quote(t) for t in tokens)
            shell_exe, cmd_flag = detect_shell()

            process = subprocess.run(
                [shell_exe, cmd_flag, cmd_str],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )

            if process.returncode == 0:
                result.update(
                    {"success": True, "output": process.stdout, "error": process.stderr}
                )
            else:
                result.update(
                    {
                        "success": False,
                        "output": process.stdout,
                        "error": process.stderr or f"Exit code: {process.returncode}",
                    }
                )

        except subprocess.TimeoutExpired:
            result.update({"error": "Timeout after 120 seconds"})
        except Exception as e:
            result.update({"error": f"{type(e).__name__}: {e}"})

        return json.dumps(result, ensure_ascii=False)

    async def _tool_mcp_connect(
        self,
        server_name: str,
        command: str,
        args: list[str],
        target_agent: str | None = None,
    ) -> str:
        """Implementation: Connect MCP server."""
        try:
            mcp_config = {"name": server_name, "command": command, "args": args}

            if target_agent:
                # Update existing agent's config for next restart
                agent_config_path = Path(
                    f"{get_app().data_dir}/Agents/{target_agent}/agent.json"
                )

                if agent_config_path.exists():
                    with open(agent_config_path, encoding="utf-8") as f:
                        config_data = json.load(f)

                    if "mcp" not in config_data:
                        config_data["mcp"] = {"enabled": True, "servers": []}
                    if "servers" not in config_data["mcp"]:
                        config_data["mcp"]["servers"] = []

                    config_data["mcp"]["servers"].append(mcp_config)

                    with open(agent_config_path, "w", encoding="utf-8") as f:
                        json.dump(config_data, f, indent=2)

                    if target_agent in self.agent_registry:
                        self.agent_registry[target_agent].mcp_servers.append(server_name)

                    return f"✓ MCP server '{server_name}' added to '{target_agent}' config. Restart agent to activate."
                else:
                    return f"✗ Agent '{target_agent}' config not found"

            else:
                # Create new agent with MCP
                new_agent_name = f"mcp_{server_name}"

                builder = self.isaa_tools.get_agent_builder(
                    name=new_agent_name, add_base_tools=True, with_dangerous_shell=False
                )

                self._apply_rate_limiter_to_builder(builder)

                # Enable MCP and add server config
                builder.config.mcp.enabled = True
                builder._mcp_config_data = {server_name: mcp_config}
                builder._mcp_needs_loading = True

                await self.isaa_tools.register_agent(builder)

                self.agent_registry[new_agent_name] = AgentInfo(
                    name=new_agent_name,
                    persona=f"MCP Agent: {server_name}",
                    mcp_servers=[server_name],
                )

                self._save_state()

                return (
                    f"✓ Created agent '{new_agent_name}' with MCP server '{server_name}'"
                )

        except Exception as e:
            return f"✗ Failed to connect MCP: {e}"

    async def _tool_update_agent_config(
        self, agent_name: str, config_updates: dict
    ) -> str:
        """Implementation: Update agent config."""
        try:
            agent_config_path = Path(
                f"{get_app().data_dir}/Agents/{agent_name}/agent.json"
            )

            if not agent_config_path.exists():
                return f"✗ Agent '{agent_name}' config not found"

            with open(agent_config_path, encoding="utf-8") as f:
                config_data = json.load(f)

            # Deep merge updates
            def deep_merge(base: dict, updates: dict):
                for key, value in updates.items():
                    if (
                        key in base
                        and isinstance(base[key], dict)
                        and isinstance(value, dict)
                    ):
                        deep_merge(base[key], value)
                    else:
                        base[key] = value

            deep_merge(config_data, config_updates)

            with open(agent_config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2)

            return f"✓ Config updated for '{agent_name}'. Restart agent to apply changes."

        except Exception as e:
            return f"✗ Failed to update config: {e}"

    # =========================================================================
    # CLI INTERFACE
    # =========================================================================

    def _build_completer(self) -> dict:
        """Build nested completer dictionary."""
        agents = self.isaa_tools.config.get("agents-name-list", ["self"])

        # Try to get VFS files for autocomplete
        vfs_files: dict = {}
        try:
            instance_key = f"agent-instance-{self.active_agent_name}"
            if instance_key in self.isaa_tools.config:
                agent = self.isaa_tools.config[instance_key]
                session = agent.session_manager.get(self.active_session_id)
                if session and hasattr(session, "vfs"):
                    vfs_files = {f: None for f in session.vfs.files}
        except Exception:
            pass

        return {
            "/help": None,
            "/quit": None,
            "/exit": None,
            "/clear": None,
            "/status": None,
            "/agent": {
                "switch": {a: None for a in agents},
                "list": None,
                "spawn": None,
                "stop": {a: None for a in agents},
            },
            "/session": {
                "switch": {},
                "list": None,
                "new": None,
            },
            "/task": {
                "list": None,
                "status": None,
                "cancel": None,
            },
            "/vfs": vfs_files
            if vfs_files
            else None,  # /vfs shows tree, /vfs <file> shows content
            "/context": {
                "stats": None,
            },
            "/bind": {a: None for a in agents},
            "/teach": {a: None for a in agents},
            "/rate-limiter": {
                "status": None,
                "config": None,
            },
        }

    def get_prompt_text(self) -> HTML:
        """Generate prompt text with status indicators."""
        cwd_name = Path.cwd().name
        bg_count = sum(1 for t in self.background_tasks.values() if t.status == "running")
        bg_indicator = (
            f" <style fg='ansiyellow'>[{bg_count}bg]</style>" if bg_count > 0 else ""
        )

        audio_indicator = (
            " <style fg='ansired'>🎤</style>" if self._audio_recording else ""
        )

        return HTML(
            f"<style fg='ansicyan'>[</style>"
            f"<style fg='ansigreen'>{cwd_name}</style>"
            f"<style fg='ansicyan'>]</style> "
            f"<style fg='ansiyellow'>({self.active_agent_name})</style>"
            f"<style fg='grey'>@{self.active_session_id}</style>"
            f"{bg_indicator}{audio_indicator}"
            f"\n<style fg='ansiblue'>❯</style> "
        )

    async def _print_status_dashboard(self):
        """Print comprehensive status dashboard."""
        c_print()
        print_box_header(f"{CLI_NAME} v{VERSION}", "🤖")

        # Host Info
        print_box_content(f"Host ID: {self.host_id}", "info")
        print_box_content(f"Uptime: {datetime.now() - self.started_at}", "info")

        print_separator()

        # Agents
        agents = self.isaa_tools.config.get("agents-name-list", [])
        print_status(f"Agents: {len(agents)} registered", "data")

        columns = [("Name", 15), ("Status", 10), ("Persona", 25), ("Tasks", 8)]
        widths = [15, 10, 25, 8]
        print_table_header(columns, widths)

        for name in agents[:10]:  # Limit display
            info = self.agent_registry.get(name, AgentInfo(name=name))
            instance_key = f"agent-instance-{name}"
            status = "Active" if instance_key in self.isaa_tools.config else "Idle"
            status_style = "green" if status == "Active" else "grey"

            bg_count = sum(
                1
                for t in self.background_tasks.values()
                if t.agent_name == name and t.status == "running"
            )

            persona = info.persona[:23] + ".." if len(info.persona) > 25 else info.persona

            name_style = "cyan" if info.is_self_agent else "white"

            print_table_row(
                [name, status, persona, str(bg_count)],
                widths,
                [name_style, status_style, "grey", "yellow"],
            )

        print_separator()

        # Background Tasks
        running_tasks = [
            t for t in self.background_tasks.values() if t.status == "running"
        ]
        print_status(f"Background Tasks: {len(running_tasks)} running", "progress")

        # Rate Limiter
        print_separator()
        rl = self._rate_limiter_config.get("features", {})
        print_status("Rate Limiter Config:", "configure")
        print_box_content(
            f"  Rate Limiting: {'✓' if rl.get('rate_limiting') else '✗'}", ""
        )
        print_box_content(
            f"  Model Fallback: {'✓' if rl.get('model_fallback') else '✗'}", ""
        )
        print_box_content(f"  Key Rotation: {rl.get('key_rotation_mode', 'N/A')}", "")

        print_box_footer()
        c_print()

    def _print_help(self):
        """Print help information."""
        print_box_header("ISAA Host Commands", "❓")

        print_status("Navigation", "info")
        print_box_content("/help - Show this help", "")
        print_box_content("/status - Show status dashboard (or F5)", "")
        print_box_content("/clear - Clear screen", "")
        print_box_content("/quit, /exit - Exit CLI", "")

        print_separator()

        print_status("Agent Management", "info")
        print_box_content("/agent list - List all agents", "")
        print_box_content("/agent switch <name> - Switch active agent", "")
        print_box_content("/agent spawn <name> <persona> - Create new agent", "")
        print_box_content("/agent stop <name> - Stop agent tasks", "")

        print_separator()

        print_status("Session Management", "info")
        print_box_content("/session list - List sessions", "")
        print_box_content("/session switch <id> - Switch session", "")
        print_box_content("/session new - Create new session", "")

        print_separator()

        print_status("Task Management", "info")
        print_box_content("/task list - List background tasks", "")
        print_box_content("/task cancel <id> - Cancel a task", "")

        print_separator()

        print_status("Advanced", "info")
        print_box_content("/bind <agent_a> <agent_b> - Bind agents", "")
        print_box_content("/teach <agent> <skill_name> - Teach skill", "")
        print_box_content("/vfs show - Show VFS structure", "")
        print_box_content("/context stats - Show context stats", "")

        print_separator()

        print_status("Shortcuts", "info")
        print_box_content("F4 - Toggle audio recording", "")
        print_box_content("F5 - Show status dashboard", "")
        print_box_content("!<cmd> - Execute shell command", "")

        print_box_footer()

    async def _handle_command(self, cmd_str: str):
        """Handle slash commands."""
        parts = cmd_str.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("/quit", "/exit", "/q", "/x" ,"/e"):
            raise EOFError

        elif cmd == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            await self._print_status_dashboard()

        elif cmd == "/help":
            self._print_help()

        elif cmd == "/status":
            await self._print_status_dashboard()

        elif cmd == "/agent":
            await self._cmd_agent(args)

        elif cmd == "/session":
            await self._cmd_session(args)

        elif cmd == "/task":
            await self._cmd_task(args)

        elif cmd == "/vfs":
            await self._cmd_vfs(args)

        elif cmd == "/context":
            await self._cmd_context(args)

        elif cmd == "/bind":
            if len(args) >= 2:
                result = await self._tool_bind_agents(args[0], args[1])
                print_status(result, "success" if "✓" in result else "error")
            else:
                print_status("Usage: /bind <agent_a> <agent_b>", "warning")

        elif cmd == "/teach":
            if len(args) >= 2:
                agent = args[0]
                skill_name = args[1]
                print_status("Enter skill instruction (end with empty line):", "info")
                lines: list[str] = []
                if self.prompt_session is not None:
                    while True:
                        line = await self.prompt_session.prompt_async(
                            HTML("<style fg='grey'>... </style>")
                        )
                        if not line.strip():
                            break
                        lines.append(line)
                instruction = "\n".join(lines)
                triggers = input("Enter triggers (comma-separated): ").split(",")
                triggers = [t.strip() for t in triggers if t.strip()]
                result = await self._tool_teach_skill(
                    agent, skill_name, instruction, triggers
                )
                print_status(result, "success" if "✓" in result else "error")
            else:
                print_status("Usage: /teach <agent> <skill_name>", "warning")

        elif cmd == "/rate-limiter":
            if args and args[0] == "status":
                print_code_block(json.dumps(self._rate_limiter_config, indent=2), "json")
            else:
                print_status("Usage: /rate-limiter status", "warning")

        else:
            print_status(f"Unknown command: {cmd}. Type /help for help.", "error")

    async def _cmd_agent(self, args: list[str]):
        """Handle /agent commands."""
        if not args:
            print_status("Usage: /agent <list|switch|spawn|stop> [args]", "warning")
            return

        action = args[0]

        if action == "list":
            result = await self._tool_list_agents()
            c_print(result)

        elif action == "switch":
            if len(args) < 2:
                print_status("Usage: /agent switch <name>", "warning")
                return
            target = args[1]
            if target in self.isaa_tools.config.get("agents-name-list", []):
                self.active_agent_name = target
                self.active_session_id = "default"
                self._save_state()
                print_status(f"Switched to agent: {target}", "success")
            else:
                print_status(f"Agent '{target}' not found", "error")

        elif action == "spawn":
            if len(args) < 2:
                print_status("Usage: /agent spawn <name> [persona]", "warning")
                return
            name = args[1]
            persona = " ".join(args[2:]) if len(args) > 2 else "general assistant"
            result = await self._tool_spawn_agent(name, persona)
            print_status(result, "success" if "✓" in result else "error")

        elif action == "stop":
            if len(args) < 2:
                print_status("Usage: /agent stop <name>", "warning")
                return
            result = await self._tool_stop_agent(args[1])
            print_status(result, "success" if "✓" in result else "error")

        else:
            print_status(f"Unknown agent action: {action}", "error")

    async def _cmd_session(self, args: list[str]):
        """Handle /session commands."""
        if not args:
            print_status("Usage: /session <list|switch|new>", "warning")
            return

        action = args[0]

        if action == "list":
            try:
                agent = await self.isaa_tools.get_agent(self.active_agent_name)
                sessions = list(agent.session_manager.sessions.keys())
                print_box_header("Sessions", "📁")
                for sid in sessions:
                    prefix = "* " if sid == self.active_session_id else "  "
                    print_box_content(f"{prefix}{sid}", "")
                print_box_footer()
            except Exception as e:
                print_status(f"Error: {e}", "error")

        elif action == "switch":
            if len(args) < 2:
                print_status("Usage: /session switch <id>", "warning")
                return
            self.active_session_id = args[1]
            self._save_state()
            print_status(f"Switched to session: {args[1]}", "success")

        elif action == "new":
            new_id = args[-1] if args[-1] != "new" else f"session_{uuid.uuid4().hex[:8]}"
            self.active_session_id = new_id
            self._save_state()
            print_status(f"Created new session: {new_id}", "success")

        else:
            print_status(f"Unknown session action: {action}", "error")

    async def _cmd_task(self, args: list[str]):
        """Handle /task commands."""
        if not args:
            print_status("Usage: /task <list|cancel> [id]", "warning")
            return

        action = args[0]

        if action == "list":
            result = await self._tool_task_status()
            c_print(result)

        elif action == "cancel":
            if len(args) < 2:
                print_status("Usage: /task cancel <id>", "warning")
                return
            task_id = args[1]
            if task_id in self.background_tasks:
                self.background_tasks[task_id].task.cancel()
                self.background_tasks[task_id].status = "cancelled"
                print_status(f"Task {task_id} cancelled", "success")
            else:
                print_status(f"Task {task_id} not found", "error")

        else:
            print_status(f"Unknown task action: {action}", "error")

    def _print_vfs_tree(self, tree: dict, level: int = 0, max_depth: int = 4):
        """Recursively print VFS directory structure (HTML version)."""
        if level > max_depth:
            c_print(HTML(f"{'  ' * level}<style fg='{PTColors.GREY}'>...</style>"))
            return

        indent = "  " * level
        # Icons als HTML Strings vorbereiten
        folder_icon = f"<style fg='{PTColors.BLUE}'>📂</style>"
        file_icon = f"<style fg='{PTColors.GREY}'>📄</style>"

        # Sort: folders first, then files
        items = sorted(tree.items(), key=lambda x: (not isinstance(x[1], dict), x[0]))

        for name, content in items:
            if name.startswith("."):
                continue  # Skip hidden

            safe_name = html.escape(name)

            if isinstance(content, dict) and content:
                # Directory (non-empty dict) -> Bold Cyan Name
                c_print(HTML(
                    f"{indent}{folder_icon} <style font-weight='bold' fg='{PTColors.CYAN}'>{safe_name}</style>"
                ))
                self._print_vfs_tree(content, level + 1, max_depth)
            else:
                # File -> Grey Size Hint
                size_hint = ""
                if isinstance(content, str) or hasattr(content, "__len__"):
                    size_hint = f" <style fg='{PTColors.GREY}'>({len(content)}b)</style>"

                c_print(HTML(f"{indent}{file_icon} {safe_name}{size_hint}"))

    def _detect_file_type(self, filename: str) -> str:
        """Detect file type from extension."""
        ext = Path(filename).suffix.lower()
        type_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".md": "markdown",
            ".html": "html",
            ".css": "css",
            ".sh": "bash",
            ".bash": "bash",
            ".sql": "sql",
            ".xml": "xml",
            ".env": "env",
            ".txt": "text",
            ".log": "text",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
        }
        return type_map.get(ext, "markdown")  # Default to markdown

    async def _cmd_vfs(self, args: list[str]):
        """Handle /vfs commands - show tree or file content."""
        try:
            agent = await self.isaa_tools.get_agent(self.active_agent_name)
            session = agent.session_manager.get(self.active_session_id)

            if not session or not hasattr(session, "vfs"):
                print_status("No VFS available in current session", "warning")
                return

            # If filename provided, show file content
            if args:
                filename = " ".join(args)  # Handle filenames with spaces

                # Try to read the file
                try:
                    result = session.vfs.read(filename)

                    if isinstance(result, dict):
                        # VFS returns dict with 'success' and 'content'
                        if result.get("success"):
                            content = result.get("content", "")
                        else:
                            print_status(f"File not found: {filename}", "error")
                            return
                    else:
                        content = str(result)

                    # Detect file type and display
                    file_type = self._detect_file_type(filename)

                    print_box_header(f"📄 {filename}", "")
                    print_box_content(
                        f"Type: {file_type} | Size: {len(content)} bytes", "info"
                    )
                    print_separator()

                    # Format based on type
                    if file_type == "json":
                        try:
                            parsed = json.loads(content)
                            content = json.dumps(parsed, indent=2, ensure_ascii=False)
                        except json.JSONDecodeError:
                            pass
                        print_code_block(content, "json", show_line_numbers=True)
                    elif file_type in ("yaml", "yml"):
                        print_code_block(content, "yaml", show_line_numbers=True)
                    elif file_type == "toml":
                        print_code_block(content, "toml", show_line_numbers=True)
                    elif file_type == "env":
                        print_code_block(content, "env", show_line_numbers=False)
                    elif file_type == "markdown":
                        # Print markdown with basic formatting (HTML version)
                        for line in content.split("\n"):
                            safe_line = html.escape(line)

                            if line.startswith("# "):
                                # H1: Bold + Cyan
                                c_print(HTML(f"<style font-weight='bold' fg='{PTColors.CYAN}'>{safe_line}</style>"))
                            elif line.startswith("## "):
                                # H2: Bold + Blue
                                c_print(HTML(f"<style font-weight='bold' fg='{PTColors.BLUE}'>{safe_line}</style>"))
                            elif line.startswith("### "):
                                # H3: Bold
                                c_print(HTML(f"<style font-weight='bold'>{safe_line}</style>"))
                            elif line.startswith("```"):
                                # Code fence: Grey
                                c_print(HTML(f"<style fg='{PTColors.GREY}'>{safe_line}</style>"))
                            elif line.startswith("- ") or line.startswith("* "):
                                # List item: Cyan Bullet
                                c_print(HTML(f"  <style fg='{PTColors.CYAN}'>•</style> {safe_line[2:]}"))
                            elif line.startswith("> "):
                                # Quote: Grey Bar + Italic
                                c_print(HTML(
                                    f"  <style fg='{PTColors.GREY}'>│</style> <style italic='true'>{safe_line[2:]}</style>"))
                            else:
                                # Normal text
                                c_print(HTML(f"  {safe_line}"))
                    else:
                        # Code files with line numbers
                        print_code_block(content, "text", show_line_numbers=True)

                    print_box_footer()

                except Exception as e:
                    print_status(f"Error reading file '{filename}': {e}", "error")
                return

            # No args: show VFS tree structure
            print_box_header(
                f"VFS Structure: {self.active_agent_name}@{self.active_session_id}", "📂"
            )

            # Build tree from flat file list
            tree: dict = {}
            for filepath in session.vfs.files:
                parts = filepath.strip("/").split("/")
                current = tree
                for i, part in enumerate(parts):
                    if i == len(parts) - 1:
                        # Last part is filename - store content length
                        file_content = session.vfs.files.get(filepath, "")
                        current[part] = str(file_content) if file_content else ""
                    else:
                        # Directory
                        current = current.setdefault(part, {})

            if tree:
                c_print()
                self._print_vfs_tree(tree)
                c_print()

                # Summary
                total_files = len(session.vfs.files)
                total_size = sum(len(str(c)) for c in session.vfs.files.values() if c)
                print_separator()
                print_box_content(
                    f"Total: {total_files} files, {total_size:,} bytes", "info"
                )
            else:
                print_box_content("VFS is empty", "warning")

            print_box_footer()

        except Exception as e:
            print_status(f"Error: {e}", "error")

    async def _cmd_context(self, args: list[str]):
        """Handle /context commands."""
        try:
            agent = await self.isaa_tools.get_agent(self.active_agent_name)
            overview = await agent.context_overview(self.active_session_id)
        except Exception as e:
            print_status(f"Error: {e}", "error")

    async def _handle_agent_interaction(self, user_input: str):
        """Handle regular agent interaction."""
        try:
            agent = await self.isaa_tools.get_agent(self.active_agent_name)

            print_status(f"Processing with {self.active_agent_name}...", "progress")

            # Stream response
            final_response = ""
            full_response = ""
            start_data = False
            async for chunk in agent.a_stream_verbose(
                query=user_input,
                session_id=self.active_session_id,
            ):
                full_response += chunk
                print(chunk, end="", flush=True)
                if not start_data and "```json" in full_response:
                    start_data = True
                if start_data:
                    final_response += chunk

            c_print()  # Newline after streaming

            # Try to visualize JSON responses
            try:
                if final_response.strip().startswith("```json"):
                    final_response = final_response.strip()[7:-3]
                if final_response.strip().startswith("```"):
                    final_response = final_response.strip()[3:-3]
                if final_response.strip().startswith("{"):
                    data = json.loads(final_response.strip())
                    if isinstance(data, dict):
                        print_separator()
                        await visualize_data_terminal(
                            data, agent, max_preview_chars=max(len(final_response), 8000)
                        )
            except (json.JSONDecodeError, Exception):
                pass

        except Exception as e:
            print_status(f"Error: {e}", "error")
            import traceback

            traceback.print_exc()

    async def _handle_shell(self, command: str):
        """Handle shell command (! prefix)."""
        result = self._tool_shell(command)
        try:
            data = json.loads(result)
            if data.get("success"):
                if data.get("output"):
                    c_print(data["output"])
                if data.get("error"):
                    print_status(data["error"], "warning")
            else:
                print_status(data.get("error", "Command failed"), "error")
                if data.get("output"):
                    c_print(data["output"])
        except json.JSONDecodeError:
            c_print(result)

    # =========================================================================
    # MAIN RUN LOOP
    # =========================================================================

    async def run(self):
        """Main CLI execution loop."""
        # Print banner
        c_print()
        print_box_header(f"{CLI_NAME} v{VERSION}", "🤖")
        print_box_content("Multi-Agent Host System", "info")
        print_box_content("Type /help for commands, F4 for voice input", "")
        print_box_footer()
        c_print()

        # Initialize Self Agent
        await self._init_self_agent()

        # Print status
        await self._print_status_dashboard()

        # Create prompt session
        self.prompt_session = PromptSession(
            history=self.history,
            completer=FuzzyCompleter(
                NestedCompleter.from_nested_dict(self._build_completer())
            ),
            complete_while_typing=True,
            key_bindings=self.key_bindings,
        )

        # Main loop
        while True:
            try:
                # Update completer
                self.prompt_session.completer = FuzzyCompleter(
                    NestedCompleter.from_nested_dict(self._build_completer())
                )

                # Get input
                with patch_stdout():
                    user_input = await self.prompt_session.prompt_async(
                        self.get_prompt_text()
                    )

                # Check for transcription
                if hasattr(self, "_last_transcription") and self._last_transcription:
                    user_input = self._last_transcription
                    self._last_transcription = None
                    print_status(f"Using transcription: {user_input}", "info")

                user_input = user_input.strip()
                if not user_input:
                    continue

                # Route input
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
                print_status(f"Unexpected Error: {e}", "error")
                import traceback

                traceback.print_exc()

        # Cleanup
        self._save_state()

        # Cancel background tasks
        for _, bg_task in self.background_tasks.items():
            if bg_task.status == "running":
                bg_task.task.cancel()

        print_status("Goodbye!", "success")


# =============================================================================
# ENTRY POINT
# =============================================================================


async def run(app=None, *args):
    """Entry point for ISAA Host CLI."""
    app = app or get_app("isaa-host")
    host = ISAA_Host(app)
    await host.run()


def main():
    """Synchronous entry point."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
