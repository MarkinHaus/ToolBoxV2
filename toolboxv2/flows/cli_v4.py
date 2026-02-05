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
from prompt_toolkit.completion import FuzzyCompleter, NestedCompleter, WordCompleter, PathCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout

# ToolBoxV2 Imports
from toolboxv2 import get_app, remove_styles

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
from toolboxv2.mods.isaa.base.Agent.vfs_v2 import FileBackingType
from toolboxv2.mods.isaa.base.AgentUtils import detect_shell, anything_from_str_to_dict
from toolboxv2.mods.isaa.base.audio_io.audioIo import AudioStreamPlayer
from toolboxv2.mods.isaa.module import Tools as IsaaTool

import html
import json
import sys
from prompt_toolkit import print_formatted_text, HTML
from prompt_toolkit.styles import Style


# =================== Helpers & Setup ===================
MODEL_MAPPING = {
    "gemini-3-flash": "openrouter/google/gemini-3-flash-preview",
    "deepseek-v3.2": "openrouter/deepseek/deepseek-v3.2",
    "kimi-k2.5": "openrouter/moonshotai/kimi-k2.5",
    "gpt-oss-120bF": "openrouter/openai/gpt-oss-120b:free",
    "gpt-oss-120b": "openrouter/openai/gpt-oss-120b",
    "mistral-14b": "openrouter/mistralai/ministral-14b-2512",
    "devstral": "openrouter/mistralai/devstral-2512",
    "gpt-oss-safeguard-20b": "openrouter/openai/gpt-oss-safeguard-20b",
    "nemotron-3": "openrouter/nvidia/nemotron-3-nano-30b-a3b",
    "minimax-m2.1": "openrouter/minimax/minimax-m2.1",
    "glm-4.7-flash": "openrouter/z-ai/glm-4.7-flash",
    "glm-4.7": "openrouter/z-ai/glm-4.7",
    "glm-4.6v": "openrouter/z-ai/glm-4.6v",
    "glm-4.5f": "openrouter/z-ai/glm-4.5-air:free",
    "step-3.5-flash": "openrouter/stepfun/step-3.5-flash:free",
    "trinity-large": "openrouter/arcee-ai/trinity-large-preview:free",
}
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


# =========================== Simple Feature Manager ==========================
class SimpleFeatureManager:
    def __init__(self):
        self.agent_ref = None
        self.features = { }

    def list_features(self):
        return list(self.features.keys())

    def enable(self, feature):
        if feature in self.features:
            self.features[feature]["is_enabled"] = True
            if self.features[feature]["activation_f"]:
                self.features[feature]["activation_f"](self.agent_ref)

    def is_enabled(self, feature):
        return self.features.get(feature, {"is_enabled": False})["is_enabled"]

    def disable(self, feature):
        if feature in self.features:
            self.features[feature]["is_enabled"] = False
            if self.features[feature]["deactivation_f"]:
                self.features[feature]["deactivation_f"](self.agent_ref)

    def add_feature(self, feature, activation_f=None, deactivation_f=None):
        self.features[feature] = {
            "is_enabled": False,
            "activation_f": activation_f,
            "deactivation_f": deactivation_f,
        }

    def set_agent(self, agent):
        self.agent_ref = agent


# ============================ Feature Definitions ============================

def load_desktop_auto_feature(fm: SimpleFeatureManager):
    def enable_desktop_auto(agent):
        from toolboxv2.mods.isaa.extras.destop_auto import register_enhanced_tools
        kit, tools = register_enhanced_tools()
        agent.add_tools(**tools)
        print_status("Desktop Automation enabled.", "success")

    def disable_desktop_auto(agent):
        from toolboxv2.mods.isaa.extras.destop_auto import register_enhanced_tools
        kit, tools = register_enhanced_tools()
        agent.remove_tools(tools)
        print_status("Desktop Automation enabled.", "success")
    fm.add_feature("desktop_auto", activation_f=enable_desktop_auto, deactivation_f=disable_desktop_auto)

def load_web_auto_feature(fm: SimpleFeatureManager):
    def enable_web_auto(agent):
        from toolboxv2.mods.isaa.extras.web_helper.web_agent import minimal_web_agent_integration
        agent.add_tools(minimal_web_agent_integration())
        print_status("Web Automation enabled.", "success")

    def disable_web_auto(agent):
        from toolboxv2.mods.isaa.extras.web_helper.web_agent import minimal_web_agent_integration
        agent.remove_tools(minimal_web_agent_integration())
        print_status("Web Automation disabled.", "success")

    fm.add_feature("mini_web_auto", activation_f=enable_web_auto, deactivation_f=disable_web_auto)

def load_full_web_auto_feature(fm: SimpleFeatureManager):
    def enable_full_web_auto(agent):
        from toolboxv2.mods.isaa.extras.web_helper.tooklit import get_full_tools
        agent.add_tools(get_full_tools()[1])
        print_status("Full Web Automation enabled.", "success")

    def disable_full_web_auto(agent):
        from toolboxv2.mods.isaa.extras.web_helper.tooklit import get_full_tools
        agent.remove_tools(get_full_tools()[1])
        print_status("Full Web Automation disabled.", "success")

    fm.add_feature("full_web_auto", activation_f=enable_full_web_auto, deactivation_f=disable_full_web_auto)


ALL_FEATURES = {
    "desktop_auto": load_desktop_auto_feature,
    "mini_web_auto": load_web_auto_feature,
    "full_web_auto": load_full_web_auto_feature,
}

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

    version = VERSION

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

        # audio
        self.audio_device_index = 0

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

        self.audio_player = AudioStreamPlayer()
        self.verbose_audio = False  # /audio on aktiviert das

        # Prompt Toolkit setup
        self.history = FileHistory(str(self.history_file))
        self.key_bindings = self._create_key_bindings()
        self.prompt_session: PromptSession | None = None

        # Self Agent initialization flag
        self._self_agent_initialized = False

        # Load persisted state
        self._load_rate_limiter_config()
        self._load_state()


        self.feature_manager = SimpleFeatureManager()
        for feature in ALL_FEATURES.values():
            feature(self.feature_manager)

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
                    self.audio_device_index = state.get("audio_device_index", 0)
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
                "audio_device_index": self.audio_device_index,
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
            async def __():
                await self._print_status_dashboard()
                # Rate Limiter
                await self._cmd_vfs(["list"])
                await self._cmd_skill(["list"])
                await self._cmd_mcp(["list"])
                await self._cmd_session(["list"])
                await self._cmd_session(["show"])
            asyncio.create_task(__())

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

    def _select_audio_device(self):
        """Select audio input device."""
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            devices_names = []
            for i, device in enumerate(devices):
                if device["name"] in devices_names:
                    continue
                devices_names.append(device["name"])
                print(f"[{i}] {device['name']}")
            device_index = int(input("Select device index: "))
            self.audio_device_index = device_index
        except ImportError:
            print_status(
                "Audio requires 'sounddevice'. Install: pip install sounddevice", "error"
            )

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
                device=self.audio_device_index
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
            import traceback
            traceback.print_exc()
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
        """
        Führt einen Shell-Befehl LIVE und INTERAKTIV aus.
        Unterstützt Windows (CMD/PowerShell) und Unix (Bash/Zsh).
        """
        import subprocess
        import shlex
        import sys

        # Shell-Erkennung (Windows/Unix)
        shell_exe, cmd_flag = detect_shell()

        # Vorbereitung für Windows "Charm" / ANSI Support
        # Wir übergeben stdin/stdout/stderr direkt (None), damit der Prozess
        # das Terminal "besitzt".
        try:
            # Wir nutzen subprocess.run OHNE capture_output,
            # damit das Terminal direkt interagieren kann.
            print_separator(char="═")

            # Ausführung im Vordergrund
            process = subprocess.run(
                [shell_exe, cmd_flag, command],
                stdin=None,  # Direktes Terminal-Input
                stdout=None,  # Direktes Terminal-Output (Live!)
                stderr=None,  # Direktes Terminal-Error
                check=False
            )

            print_separator(char="═")

            result = {
                "success": process.returncode == 0,
                "exit_code": process.returncode
            }
            return json.dumps(result)

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    async def _handle_shell(self, command: str):
        """Wrapper für den Shell-Befehl mit Prompt-Toolkit Suspension."""
        from prompt_toolkit.eventloop import run_in_executor_with_context

        # WICHTIG: Wir müssen die prompt_toolkit UI pausieren,
        # damit die Shell das Terminal sauber übernehmen kann.
        try:
            # Wir führen die blockierende Shell-Aktion in einem Thread aus,
            # aber geben ihr vollen Zugriff auf das Terminal.
            await run_in_executor_with_context(
                lambda: self._tool_shell(command)
            )
        except Exception as e:
            print_status(f"Shell Error: {e}", "error")

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

        # Try to get VFS files and mounts for autocomplete
        vfs_files: dict = {}
        vfs_mounts: dict = {}
        vfs_dirty: dict = {}
        model_options: dict = {}
        current_skills: dict = {}
        features: dict = {_:None for _ in self.feature_manager.list_features()}
        try:
            instance_key = f"agent-instance-{self.active_agent_name}"
            if instance_key in self.isaa_tools.config:
                agent = self.isaa_tools.config[instance_key]
                session = agent.session_manager.get(self.active_session_id)
                if session and hasattr(session, "vfs"):
                    vfs_files = {f: None for f in session.vfs.files}
                    vfs_mounts = {m: None for m in session.vfs.mounts} if hasattr(session.vfs, 'mounts') else {}
                    vfs_dirty = {
                        f: None for f, file in session.vfs.files.items()
                        if hasattr(file, 'is_dirty') and file.is_dirty
                    }

                engine = agent._get_execution_engine()
                if hasattr(engine, 'skills_manager'):
                    current_skills = {s_id: None for s_id in engine.skills_manager.skills.keys()}
            model_options = {m: None for m in MODEL_MAPPING.keys()}
        except Exception:
            pass

        path_compl = PathCompleter(expanduser=True)



        return {
            "/help": None,
            "/quit": None,
            "/exit": None,
            "/clear": None,
            "/status": None,
            "/audio": {
                "on": None,
                "off": None,
                "stop": None,
                "voice": None,
                "backend": {
                    "groq": None,
                    "piper": None,
                    "elevenlabs": None,
                },
                "lang": None,
            },
            "/agent": {
                "switch": {a: None for a in agents},
                "list": None,
                "spawn": None,
                "stop": {a: None for a in agents},
                "model": {
                    "fast": model_options,
                    "complex": model_options
                },
                "checkpoint": {
                    "save": {a: None for a in agents},
                    "load": {a: None for a in agents},
                },
                "load-all": None,
                "save-all": None,
                "stats": {a: None for a in agents},
                "delete": {a: None for a in agents},
                "config": {a: None for a in agents},
            },
            "/mcp": {
                "list": None,
                "add": None,  # /mcp add <name> <cmd> <args>
                "remove": {s: None for s in getattr(self, 'current_mcp_servers', [])},
                "reload": None,  # Re-connectet alle Server
                "info": None,  # Details zu einem Server
            },
            "/session": {
                "switch": {},
                "list": None,
                "new": None,
                "show": None,
                "clear": None,
            },
            "/task": {
                "status": {t: None for t in self.background_tasks.keys()},
                "cancel": {t: None for t in self.background_tasks.keys()},
            },
            "/vfs": {
                "mount": path_compl, # /vfs mount <local_path> [vfs_path] [--readonly] [--no-sync]
                "unmount": vfs_mounts if vfs_mounts else None,
                "sync": vfs_dirty if vfs_dirty else vfs_files if vfs_files else None,
                "refresh": vfs_mounts if vfs_mounts else None,
                "pull": vfs_files if vfs_files else None,
                "save": vfs_files if vfs_files else None,
                "mounts": None,
                "dirty": None,
                **vfs_files,  # Direct file access: /vfs <filename>
            } if vfs_files or vfs_mounts else None,
            "/context": {
                "stats": None,
            },
            "/skill": {
                "list": None,
                "show": current_skills if current_skills else None,
                "edit": current_skills if current_skills else None,
                "delete": current_skills if current_skills else None,
                "boost": current_skills if current_skills else None,
                "merge": current_skills if current_skills else None,
                "import": {},
                "export": {s_id: path_compl for s_id in ["all"]+list(current_skills.keys())} if current_skills else None,
            },
            "/bind": {a: None for a in agents},
            "/teach": {a: None for a in agents},
            "/feature": {
                "list": None,
                "enable": features,
                "disable": features,
            },
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

        max_name_length = max(len(name) for name in agents)
        columns = [("Name", max_name_length), ("Status", 10), ("Persona", 25), ("Tasks", 8)]
        widths = [max_name_length, 10, 25, 8]
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
        print_box_content("/agent model <fast|complex> <name> - Change LLM model on the fly", "")
        print_box_content("/agent checkpoint <save|load> [name] - Manage state persistence", "")
        print_box_content("/agent load-all               - Initialize all agents from disk", "")
        print_box_content("/agent save-all               - Save checkpoints for all active agents", "")
        print_box_content("/agent stats [name]           - Show token usage and cost metrics", "")
        print_box_content("/agent delete <name>          - Remove agent and its data", "")
        print_box_content("/agent config <name>          - View raw JSON configuration", "")

        print_separator()

        print_status("Session Management", "info")
        print_box_content("/session list - List sessions", "")
        print_box_content("/session switch <id> - Switch session", "")
        print_box_content("/session new - Create new session", "")
        print_box_content("/session show [n]       - Show last n messages (default 10)", "")
        print_box_content("/session clear          - Clear current session history", "")

        print_separator()

        print_status("MCP Management (Live)", "info")
        print_box_content("/mcp list                    - Zeige aktive MCP Verbindungen", "")
        print_box_content("/mcp add <n> <cmd> [args]    - Server hinzufügen & Tools laden", "")
        print_box_content("/mcp remove <name>           - Server trennen & Tools löschen", "")
        print_box_content("/mcp reload                  - Alle MCP Tools neu indizieren", "")

        print_separator()

        print_status("Task Management", "info")
        print_box_content("/task list - List background tasks", "")
        print_box_content("/task cancel <id> - Cancel a task", "")

        print_separator()

        print_status("Advanced", "info")
        print_box_content("/bind <agent_a> <agent_b> - Bind agents", "")
        print_box_content("/teach <agent> <skill_name> - Teach skill", "")
        print_box_content("/context stats - Show context stats", "")
        print_separator()
        print_status("History Management", "info")
        print_box_content("/history show [n]       - Show last n messages (default 10)", "")
        print_box_content("/history clear          - Clear current session history", "")
        print_separator()
        print_status("VFS Management", "info")
        print_box_content("/vfs                         - Show VFS tree", "")
        print_box_content("/vfs <file>                  - Show file content", "")
        print_box_content("/vfs mount <path> [vfs_path] - Mount local folder", "")
        print_box_content("/vfs unmount <vfs_path>      - Unmount folder", "")
        print_box_content("/vfs sync [file]             - Sync changes to disk", "")
        print_box_content("/vfs save <vfs_path> [file]  - Save vfs file to disk", "")
        print_box_content("/vfs refresh <mount>         - Re-scan mount for changes", "")
        print_box_content("/vfs pull <file>             - Reload file from disk", "")
        print_box_content("/vfs mounts                  - List active mounts", "")
        print_box_content("/vfs dirty                   - Show modified files", "")
        print_separator()
        print_status("Mount Options", "info")
        print_box_content("  --readonly                 - No write operations", "")
        print_box_content("  --no-sync                  - Manual sync only", "")
        print_separator()
        print_status("Skill Management", "info")
        print_box_content("/skill list             - List skills of active agent", "")
        print_box_content("/skill list --inactive  - List inactive skills ", "")
        print_box_content("/skill show <id>        - Show details/instruction", "")
        print_box_content("/skill edit <id>        - Edit skill instruction", "")
        print_box_content("/skill delete <id>      - Delete a skill", "")
        print_box_content("/skill merge <keep_id> <remove_id>", "")
        print_box_content("/skill boost <skill_id> 0.3  - Delete a skill", "")
        print_box_content("/skill import <path>         - import skills from directory/skill file", "")
        print_box_content("/skill export <id> <path>    - id=all Extprt skill or all skills", "")
        print_separator()

        print_status("Additional Features", "info")
        print_box_content("/feature list                     - List all features", "")
        print_box_content("/feature disable <feature>        - Disable a feature", "")
        print_box_content("/feature enable <feature>         - Enable a feature", "")
        print_box_content("/feature enable desktop           - Enable Desktop Automation", "")
        print_box_content("/feature enable web <headless>    - Enable Desktop Web Automation", "")
        print_separator()

        print_status("Audio Settings", "info")
        print_box_content("/audio on       - Enable verbose audio", "")
        print_box_content("/audio off      - Disable verbose audio", "")
        print_box_content("/audio voice <v>- Set voice", "")
        print_box_content("/audio backend <b> - Set backend (groq/piper/elevenlabs)", "")
        print_box_content("/audio stop     - Stop current playback", "")
        print_box_content("", "")
        print_box_content("Tip: Add #audio to any message for one-time audio response", "info")
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

        elif cmd == "/mcp":
            await self._cmd_mcp(args)

        elif cmd == "/vfs":
            await self._cmd_vfs(args)

        elif cmd == "/skill":
            await self._cmd_skill(args)

        elif cmd == "/audio":
            await self._handle_audio_command(args)

        elif cmd == "/context":
            await self._cmd_context(args)
        elif cmd == "/feature":
            await self._cmd_feature(args)

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
            print_status("Usage: /agent <list|switch|spawn|stop|model|...> [args]", "warning")
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

        elif action == "model":
            if len(args) < 3:
                print_status("Usage: /agent model <fast|complex> <model_name>", "warning")
                return

            target_type = args[1].lower()  # "fast" oder "complex"
            model_alias = args[2]

            if model_alias not in MODEL_MAPPING:
                print_status(f"Model '{model_alias}' not in registry. Using raw name.", "info")
                full_model_name = model_alias
            else:
                full_model_name = MODEL_MAPPING[model_alias]

            try:
                agent = await self.isaa_tools.get_agent(self.active_agent_name)
                if target_type == "fast":
                    agent.amd.fast_llm_model = full_model_name
                elif target_type == "complex":
                    agent.amd.complex_llm_model = full_model_name
                else:
                    print_status("Type must be 'fast' or 'complex'", "error")
                    return

                print_status(f"Updated {target_type} model for {self.active_agent_name} to: {full_model_name}",
                             "success")

                # Optional: Config persistent speichern
                await self._tool_update_agent_config(self.active_agent_name, {
                    f"{target_type}_llm_model": full_model_name
                })

            except Exception as e:
                print_status(f"Failed to update model: {e}", "error")

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

        elif action == "checkpoint":
            if len(args) < 2:
                print_status("Usage: /agent checkpoint <save|load> [name]", "warning")
                return
            sub = args[1].lower()
            target = args[2] if len(args) > 2 else self.active_agent_name
            try:
                agent = await self.isaa_tools.get_agent(target)
                if sub == "save":
                    path = await agent.save()
                    print_status(f"Checkpoint saved: {path}", "success")
                else:
                    await agent.restore()
                    print_status(f"State restored for {target}", "success")
            except Exception as e:
                print_status(f"Checkpoint error: {e}", "error")

        elif action == "load-all":
            print_status("Scanning agent directory...", "progress")
            agent_dir = Path(self.app.data_dir) / "Agents"
            loaded = 0
            if agent_dir.exists():
                for d in agent_dir.iterdir():
                    if d.is_dir() and (d / "agent.json").exists():
                        try:
                            await self.isaa_tools.get_agent(d.name)
                            loaded += 1
                        except:
                            pass
            print_status(f"Loaded {loaded} agents from disk", "success")

        elif action == "save-all":
            print_status("Saving all active agent states...", "progress")
            for name in self.isaa_tools.config.get("agents-name-list", []):
                instance_key = f"agent-instance-{name}"
                if instance_key in self.isaa_tools.config:
                    agent = self.isaa_tools.config[instance_key]
                    await agent.save()
            print_status("All agents checkpointed", "success")

        elif action == "stats":
            target = args[1] if len(args) > 1 else self.active_agent_name
            try:
                agent = await self.isaa_tools.get_agent(target)
                stats = agent.get_stats()
                print_box_header(f"Stats: {target}", "📊")
                # Modells
                print_table_row(["Fast Model", agent.amd.fast_llm_model], [20, 15], ["white", "blue"])
                print_table_row(["Complex Model", agent.amd.complex_llm_model], [20, 15], ["white", "blue"])
                print_table_row(["Input Tokens", f"{stats['total_tokens_in']:,}"], [20, 15], ["white", "cyan"])
                print_table_row(["Output Tokens", f"{stats['total_tokens_out']:,}"], [20, 15], ["white", "cyan"])
                print_table_row(["Total Cost", f"${stats['total_cost']:.4f}"], [20, 15], ["white", "green"])
                print_table_row(["LLM Calls", str(stats['total_llm_calls'])], [20, 15], ["white", "yellow"])
                # Session data {
                #             'version': 2,
                #             'agent_name': self.agent_name,
                #             'total_sessions': len(self.sessions),
                #             'active_sessions': active_count,
                #             'docker_enabled_sessions': docker_count,
                #             'running_containers': running_containers,
                #             'total_sessions_created': self._total_sessions_created,
                #             'total_history_messages': total_history,
                #             'memory_loaded': self._memory_instance is not None,
                #             'default_lsp_enabled': self.enable_lsp,
                #             'default_docker_enabled': self.enable_docker,
                #             'session_ids': list(self.sessions.keys())
                #         }
                print_table_row(["Total Sessions", str(stats['sessions']['total_sessions'])], [20, 15], ["white", "blue"])
                print_table_row(["Active Sessions", str(stats['sessions']['active_sessions'])], [20, 15], ["white", "blue"])
                print_table_row(["Running Containers", str(stats['sessions']['running_containers'])], [20, 15], ["white", "blue"])
                print_table_row(["Total Sessions", str(stats['sessions']['total_sessions_created'])], [20, 15], ["white", "blue"])
                print_table_row(["Total History", str(stats['sessions']['total_history_messages'])], [20, 15], ["white", "blue"])
                print_table_row(["Memory Loaded", str(stats['sessions']['memory_loaded'])], [20, 15], ["white", "blue"])
                print_table_row(["Default LSP", str(stats['sessions']['default_lsp_enabled'])], [20, 15], ["white", "blue"])
                print_table_row(["Default Docker", str(stats['sessions']['default_docker_enabled'])], [20, 15], ["white", "blue"])
                # Tools section 'total_tools': len(self._registry),
                #             'by_source': {
                #                 source: len(names)
                #                 for source, names in self._source_index.items()
                #             },
                #             'categories': list(self._category_index.keys()),
                #             'total_calls'
                print_table_row(["Total Tools", str(stats['tools']['total_tools'])], [20, 15], ["white", "blue"])
                print_table_row(["Total Calls", str(stats['tools']['total_calls'])], [20, 15], ["white", "blue"])
                # Binding data {
                #             'agent_name': self.agent_name,
                #             'total_bindings': len(self.bindings),
                #             'public_bindings': sum(1 for b in self.bindings.values() if b.mode == 'public'),
                #             'private_bindings': sum(1 for b in self.bindings.values() if b.mode == 'private'),
                #             'total_messages_sent': total_sent,
                #             'total_messages_received': total_received,
                #             'partners': list(self.bindings.keys())
                #         }
                print_table_row(["Total Bindings", str(stats['bindings']['total_bindings'])], [20, 15], ["white", "blue"])
                print_table_row(["Public Bindings", str(stats['bindings']['public_bindings'])], [20, 15], ["white", "blue"])
                print_table_row(["Private Bindings", str(stats['bindings']['private_bindings'])], [20, 15], ["white", "blue"])
                print_table_row(["Total I/O Messages", f"{str(stats['bindings']['total_messages_received'])}/{str(stats['bindings']['total_messages_sent'])}"], [20, 15], ["white", "blue"])
                print_box_footer()
                # tools categories
                print_code_block(json.dumps({"Tools Categories":stats['tools']['categories']}, indent=2), "json")
            except Exception as e:
                print_status(f"Could not get stats: {e}", "error")

        elif action == "delete":
            if len(args) < 2:
                print_status("Usage: /agent delete <name>", "warning")
                return
            target = args[1]
            confirm = input(f"Really delete agent '{target}' and all its data? (y/N): ")
            if confirm.lower() == 'y':
                # Registry cleanup
                self.agent_registry.pop(target, None)
                # Disk cleanup
                import shutil
                agent_path = Path(self.app.data_dir) / "Agents" / target
                if agent_path.exists(): shutil.rmtree(agent_path)
                print_status(f"Agent '{target}' deleted", "success")

        elif action == "config":
            target = args[1] if len(args) > 1 else self.active_agent_name
            agent_path = Path(self.app.data_dir) / "Agents" / target / "agent.json"
            if agent_path.exists():
                with open(agent_path, 'r', encoding='utf-8') as f:
                    print_code_block(f.read(), "json")
            else:
                print_status("Config not found on disk", "error")

        else:
            print_status(f"Unknown agent action: {action}", "error")

    async def _cmd_session(self, args: list[str]):
        """Handle /session commands."""
        if not args:
            print_status("Usage: /session <list|switch|new|clear|show>", "warning")
            return

        action = args[0]
        if action == "clear":
            try:
                agent = await self.isaa_tools.get_agent(self.active_agent_name)
                session = agent.session_manager.get(self.active_session_id)

                if not session:
                    print_status("No active session found.", "error")
                    return
            except Exception as e:
                print_status(f"Error accessing session: {e}", "error")
                return
            session.clear_history()
            # If the session has a persistence layer, ensure it saves
            self._save_state()
            print_status(f"History cleared for session '{self.active_session_id}'.", "success")

        elif action == "show":
            limit = 10
            if len(args) > 1 and args[1].isdigit():
                limit = int(args[1])
            try:
                agent = await self.isaa_tools.get_agent(self.active_agent_name)
                session = agent.session_manager.get(self.active_session_id)

                if not session:
                    print_status("No active session found.", "error")
                    return
            except Exception as e:
                print_status(f"Error accessing session: {e}", "error")
                return
            history = session.get_history(last_n=limit)

            if not history:
                print_status("History is empty.", "info")
                return

            print_box_header(f"History: {self.active_agent_name}@{self.active_session_id} (Last {len(history)})", "💬")

            for msg in history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                # Format based on role
                if role == "user":
                    c_print(HTML(f"  <style font-weight='bold' fg='ansigreen'>User 👤</style>"))
                    c_print(HTML(f"  {esc(content)}"))
                    c_print(HTML(""))  # Spacing

                elif role == "assistant":
                    c_print(HTML(f"  <style font-weight='bold' fg='ansicyan'>{self.active_agent_name} 🤖</style>"))

                    # Check for Tool Calls
                    if "tool_calls" in msg and msg["tool_calls"]:
                        for tc in msg["tool_calls"]:
                            fn = tc.get("function", {})
                            name = fn.get("name", "unknown")
                            c_print(HTML(f"  <style fg='ansiyellow'>🔧 Calls: {name}(...)</style>"))

                    if content:
                        c_print(HTML(f"  {esc(content)}"))
                    c_print(HTML(""))  # Spacing

                elif role == "tool":
                    # Tool Output - usually verbose, show summary
                    call_id = msg.get("tool_call_id", "unknown")
                    preview = content[:10000] + "..." if len(content) > 10000 else content
                    c_print(HTML(f"  <style fg='ansimagenta'>⚙️ Tool Result ({call_id})</style>"))
                    c_print(HTML(f"  <style fg='gray'>{esc(preview)}</style>"))
                    c_print(HTML(""))

                elif role == "system":
                    c_print(HTML(f"  <style fg='ansired'>System ⚠️</style>"))
                    c_print(HTML(f"  <style fg='gray'>{esc(content[:10000])}...</style>"))
                    c_print(HTML(""))

            print_box_footer()

        elif action == "list":
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
            print_status("Usage: /task <cancel|status> [id]", "warning")
            return

        action = args[0]

        if action == "cancel":
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

        elif action == "status":
            if len(args) < 2:
                result = await self._tool_task_status()
                c_print(result)
                return
            task_id = args[1]
            print_status(await self._tool_task_status(task_id), "info")

        else:
            print_status(f"Unknown task action: {action}", "error")

    async def _cmd_mcp(self, args: list[str]):
        """Handle live MCP management commands."""
        if not args:
            print_status("Usage: /mcp <list|add|remove|reload|info> [args]", "warning")
            return

        action = args[0].lower()
        agent = await self.isaa_tools.get_agent(self.active_agent_name)

        # Sicherstellen, dass der Agent einen MCPSessionManager hat
        if not hasattr(agent, "_mcp_session_manager") or agent._mcp_session_manager is None:
            from toolboxv2.mods.isaa.extras.mcp_session_manager import MCPSessionManager
            agent._mcp_session_manager = MCPSessionManager()

        if action == "list":
            print_box_header(f"MCP Servers: {self.active_agent_name}", "🔌")
            active_sessions = agent._mcp_session_manager.sessions
            if not active_sessions:
                print_box_content("Keine aktiven MCP Server.", "info")
            for name, session in active_sessions.items():
                print_box_content(f"{name} (Status: Connected)", "success")
            print_box_footer()

        elif action == "add":
            if len(args) < 3:
                print_status("Usage: /mcp add <name> <command> [args...]", "warning")
                return

            name = args[1]
            cmd = args[2]
            cmd_args = args[3:] if len(args) > 3 else []

            server_config = {"command": cmd, "args": cmd_args}

            print_status(f"Connecting to MCP '{name}'...", "progress")
            try:
                # 1. Verbindung herstellen
                session = await agent._mcp_session_manager.get_session(name, server_config)
                # 2. Tools extrahieren
                caps = await agent._mcp_session_manager.extract_capabilities(session, name)

                # 3. Tools im ToolManager des Agenten registrieren (Live-Rebuild)
                count = 0
                for t_name, t_info in caps.get('tools', {}).items():
                    wrapper_name = f"{name}_{t_name}"

                    # Nutze die Builder-Logik für den Wrapper (simuliert)
                    from toolboxv2.mods.isaa.base.Agent.builder import FlowAgentBuilder
                    builder_tmp = FlowAgentBuilder()
                    wrapper = builder_tmp._create_mcp_tool_wrapper(name, t_name, t_info, session)

                    agent.add_tool(
                        wrapper,
                        name=wrapper_name,
                        description=t_info.get('description'),
                        category=[f"mcp_{name}", "mcp"]
                    )
                    count += 1

                # 4. In agent.json persistent speichern
                await self._tool_mcp_connect(name, cmd, cmd_args, self.active_agent_name)

                print_status(f"Successfully added {count} tools from '{name}'", "success")

            except Exception as e:
                print_status(f"Failed to add MCP server: {e}", "error")

        elif action == "remove":
            if len(args) < 2:
                print_status("Usage: /mcp remove <name>", "warning")
                return

            name = args[1]
            # 1. Session schließen
            if name in agent._mcp_session_manager.sessions:
                # Wir löschen die Session (Shutdown erfolgt im Manager)
                del agent._mcp_session_manager.sessions[name]

            # 2. Tools aus Registry entfernen
            tools_to_remove = [t for t in agent.tool_manager.tools.keys() if t.startswith(f"{name}_")]
            for t in tools_to_remove:
                del agent.tool_manager.tools[t]

            # 3. Config bereinigen
            agent_config_path = Path(self.app.data_dir) / "Agents" / self.active_agent_name / "agent.json"
            if agent_config_path.exists():
                with open(agent_config_path, 'r+') as f:
                    cfg = json.load(f)
                    if "mcp" in cfg and "servers" in cfg["mcp"]:
                        cfg["mcp"]["servers"] = [s for s in cfg["mcp"]["servers"] if s['name'] != name]
                        f.seek(0)
                        json.dump(cfg, f, indent=2)
                        f.truncate()

            print_status(f"MCP server '{name}' and its {len(tools_to_remove)} tools removed.", "success")

        elif action == "reload":
            print_status("Reloading all MCP configurations...", "progress")
            # Wir triggern den FlowAgentBuilder Re-Process
            from toolboxv2.mods.isaa.base.Agent.builder import FlowAgentBuilder
            agent_config_path = Path(self.app.data_dir) / "Agents" / self.active_agent_name / "agent.json"

            if agent_config_path.exists():
                builder = FlowAgentBuilder(config_path=str(agent_config_path))
                # Extrahiere Config-Teil
                mcp_data = {"mcpServers": {s['name']: s for s in builder.config.mcp.model_dump().get('servers', [])}}
                builder.load_mcp_tools_from_config(mcp_data)
                await builder._process_mcp_config(agent)
                print_status("MCP Rebuild complete.", "success")
            else:
                print_status("No config found to reload.", "error")


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
        """Handle /vfs commands - mount, unmount, sync, tree, file content."""
        try:
            agent = await self.isaa_tools.get_agent(self.active_agent_name)
            session = agent.session_manager.get(self.active_session_id)

            if not session or not hasattr(session, "vfs"):
                print_status("No VFS available in current session", "warning")
                return

            if not args:
                # No args: show VFS tree structure
                await self._vfs_show_tree(session)
                return

            cmd = args[0].lower()

            # /vfs mount <local_path> [vfs_path] [--readonly] [--no-sync]
            if cmd == "mount":
                if len(args) < 2:
                    print_status("Usage: /vfs mount <local_path> [vfs_path] [--readonly] [--no-sync]", "warning")
                    return

                local_path = args[1]
                vfs_path = "/project"
                readonly = False
                auto_sync = True

                for i, arg in enumerate(args[2:], start=2):
                    if arg == "--readonly":
                        readonly = True
                    elif arg == "--no-sync":
                        auto_sync = False
                    elif not arg.startswith("--") and i == 2:
                        vfs_path = arg

                print_status(f"Mounting {local_path} → {vfs_path}...", "info")
                result = session.vfs.mount(
                    local_path=local_path,
                    vfs_path=vfs_path,
                    readonly=readonly,
                    auto_sync=auto_sync
                )

                if result.get("success"):
                    print_status(f"Mounted: {result['files_indexed']} files, {result['dirs_indexed']} dirs", "success")
                    print_box_content(f"Scan time: {result['scan_time_ms']:.1f}ms", "info")
                else:
                    print_status(f"Mount failed: {result.get('error')}", "error")
            elif cmd == "obsidian":
                if len(args) < 2:
                    print_status("Usage: /vfs obsidian <mount|unmount|sync> <local_path> [vfs_path]", "warning")
                    return
                action = args[1].lower()
                if action == "mount":
                    if len(args) < 3:
                        print_status("Usage: /vfs obsidian mount <local_path> [vfs_path]", "warning")
                        return
                    local_path = args[2]
                    vfs_path = args[3] if len(args) > 3 else "/obsidian"
                    from toolboxv2.mods.isaa.base.Agent.vfs_v2 import sync_obsidian_vault
                    result = sync_obsidian_vault(session.vfs, local_path, vfs_path)
                    if result.get("success"):
                        print_status(f"Obsidian vault mounted: {local_path} → {vfs_path}", "success")
                    else:
                        print_status(f"Mount failed: {result.get('error')}", "error")
                elif action == "unmount":
                    if len(args) < 3:
                        print_status("Usage: /vfs obsidian unmount <vfs_path>", "warning")
                        return
                    vfs_path = args[2]
                    result = session.vfs.unmount(vfs_path, save_changes=False)
                    if result.get("success"):
                        print_status(f"Obsidian vault unmounted: {vfs_path}", "success")
                    else:
                        print_status(f"Unmount failed: {result.get('error')}", "error")
                elif action == "sync":
                    if len(args) < 3:
                        print_status("Usage: /vfs obsidian sync <vfs_path>", "warning")
                        return
                    vfs_path = args[2]
                    result = session.vfs.refresh_mount(vfs_path)
                    if result.get("success"):
                        print_status(f"Obsidian vault synced: {vfs_path}", "success")
                    else:
                        print_status(f"Sync failed: {result.get('error')}", "error")
                else:
                    print_status(f"Unknown obsidian action: {action} available: mount, unmount, sync", "error")

            # /vfs unmount <vfs_path> [--no-save]
            elif cmd == "unmount":
                if len(args) < 2:
                    print_status("Usage: /vfs unmount <vfs_path> [--no-save]", "warning")
                    return

                vfs_path = args[1]
                save_changes = "--no-save" not in args

                result = session.vfs.unmount(vfs_path, save_changes=save_changes)

                if result.get("success"):
                    saved = result.get("files_saved", [])
                    print_status(f"Unmounted: {vfs_path}", "success")
                    if saved:
                        print_box_content(f"Saved {len(saved)} modified files", "info")
                else:
                    print_status(f"Unmount failed: {result.get('error')}", "error")

            # /vfs sync [vfs_path]
            elif cmd == "sync":
                if len(args) > 1:
                    # Sync specific file
                    path = args[1]
                    result = session.vfs._sync_to_local(path)
                    if result.get("success"):
                        print_status(f"Synced: {path} → {result['synced_to']}", "success")
                    else:
                        print_status(f"Sync failed: {result.get('error')}", "error")
                else:
                    # Sync all
                    result = session.vfs.sync_all()
                    if result.get("success"):
                        print_status(f"Synced {len(result['synced'])} files", "success")
                    else:
                        for err in result.get("errors", []):
                            print_status(err, "error")

            elif cmd == "save":
                if len(args) < 2:
                    print_status("Usage: /vfs save <vfs_path> <local_path>", "warning")
                    return
                vfs_path = args[1]
                local_path = args[2]
                result = session.vfs.save_to_local(vfs_path, local_path, overwrite=True, create_dirs=True)
                if result.get("success"):
                    print_status(f"Saved: {vfs_path} → {local_path}", "success")
                else:
                    print_status(f"Save failed: {result.get('error')}", "error")

            # /vfs refresh <vfs_path>
            elif cmd == "refresh":
                if len(args) < 2:
                    print_status("Usage: /vfs refresh <mount_path>", "warning")
                    return

                vfs_path = args[1]
                result = session.vfs.refresh_mount(vfs_path)

                if result.get("success"):
                    print_status(f"Refreshed: {result['files_indexed']} files", "success")
                    if result.get("modified_preserved", 0) > 0:
                        print_box_content(f"Preserved {result['modified_preserved']} modified files", "info")
                else:
                    print_status(f"Refresh failed: {result.get('error')}", "error")

            # /vfs pull <vfs_path> - reload from disk (discard local changes)
            elif cmd == "pull":
                if len(args) < 2:
                    print_status("Usage: /vfs pull <file_path>", "warning")
                    return

                path = session.vfs._normalize_path(args[1])
                f = session.vfs.files.get(path)

                if not f:
                    print_status(f"File not found: {path}", "error")
                    return

                if hasattr(f, 'local_path') and f.local_path:
                    result = session.vfs._load_shadow_content(path)
                    if result.get("success"):
                        f.is_dirty = False
                        f.backing_type = FileBackingType.SHADOW
                        print_status(f"Pulled: {path} ({result['loaded_bytes']} bytes)", "success")
                    else:
                        print_status(f"Pull failed: {result.get('error')}", "error")
                else:
                    print_status("Not a shadow file", "warning")

            # /vfs mounts - list all mounts
            elif cmd == "mounts":
                if not session.vfs.mounts:
                    print_status("No active mounts", "info")
                    return

                print_box_header("Active Mounts", "📂")
                for vfs_path, mount in session.vfs.mounts.items():
                    flags = []
                    if mount.readonly:
                        flags.append("readonly")
                    if mount.auto_sync:
                        flags.append("auto-sync")
                    flags_str = f" [{', '.join(flags)}]" if flags else ""
                    print_box_content(f"{vfs_path} → {mount.local_path}{flags_str}", "")
                print_box_footer()

            # /vfs dirty - show modified files
            elif cmd == "dirty":
                dirty_files = [
                    (path, f) for path, f in session.vfs.files.items()
                    if hasattr(f, 'is_dirty') and f.is_dirty
                ]

                if not dirty_files:
                    print_status("No modified files", "info")
                    return

                print_box_header("Modified Files", "✏️")
                for path, f in dirty_files:
                    local = f.local_path if hasattr(f, 'local_path') else "memory"
                    print_box_content(f"{path} → {local}", "")
                print_box_footer()

            # /vfs <filename> - show file content (original behavior)
            else:
                filename = " ".join(args)
                await self._vfs_show_file(session, filename)

        except Exception as e:
            print_status(f"Error: {e}", "error")

    async def _vfs_show_tree(self, session):
        """Show VFS tree structure."""
        print_box_header(
            f"VFS Structure: {self.active_agent_name}@{self.active_session_id}", "📂"
        )
        print_code_block(session.vfs.build_context_string(), "markdown")


    async def _vfs_show_file(self, session, filename: str):
        """Show file content."""
        try:
            result = session.vfs.read(filename)

            if isinstance(result, dict):
                if result.get("success"):
                    content = result.get("content", "")
                else:
                    print_status(f"File not found: {filename}", "error")
                    return
            else:
                content = str(result)

            file_type = self._detect_file_type(filename)

            # Check if shadow/dirty
            f = session.vfs.files.get(session.vfs._normalize_path(filename))
            status_parts = [f"Type: {file_type}", f"Size: {len(content)} bytes"]
            if f and hasattr(f, 'is_dirty') and f.is_dirty:
                status_parts.append("MODIFIED")
            if f and hasattr(f, 'local_path') and f.local_path:
                status_parts.append(f"→ {f.local_path}")

            print_box_header(f"📄 {filename}", "")
            print_box_content(" | ".join(status_parts), "info")
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
                for line in content.split("\n"):
                    safe_line = html.escape(line)
                    if line.startswith("# "):
                        c_print(HTML(f"<style font-weight='bold' fg='{PTColors.CYAN}'>{safe_line}</style>"))
                    elif line.startswith("## "):
                        c_print(HTML(f"<style font-weight='bold' fg='{PTColors.BLUE}'>{safe_line}</style>"))
                    elif line.startswith("### "):
                        c_print(HTML(f"<style font-weight='bold'>{safe_line}</style>"))
                    elif line.startswith("```"):
                        c_print(HTML(f"<style fg='{PTColors.GREY}'>{safe_line}</style>"))
                    elif line.startswith("- ") or line.startswith("* "):
                        c_print(HTML(f"  <style fg='{PTColors.CYAN}'>•</style> {safe_line[2:]}"))
                    elif line.startswith("> "):
                        c_print(HTML(
                            f"  <style fg='{PTColors.GREY}'>│</style> <style italic='true'>{safe_line[2:]}</style>"))
                    else:
                        c_print(HTML(f"  {safe_line}"))
            else:
                print_code_block(content, "text", show_line_numbers=True)

            print_box_footer()

        except Exception as e:
            print_status(f"Error reading file '{filename}': {e}", "error")

    async def _handle_audio_command(self, args: list[str]):
        """Handle /audio commands"""

        if not args:
            # Status zeigen
            verbose = getattr(self, 'verbose_audio', False)
            print_box_header("Audio Settings", "🔊")
            print_box_content(f"Verbose Audio: {'ON' if verbose else 'OFF'}", "")
            print_box_content(f"TTS Backend: {self.audio_player.tts_backend}", "")
            print_box_content(f"Voice: {self.audio_player.tts_voice}", "")
            print_box_content(f"Language: {self.audio_player.language}", "")
            print_separator()
            print_box_content("Commands:", "bold")
            print_box_content("/audio on           - Enable verbose audio", "")
            print_box_content("/audio off          - Disable verbose audio", "")
            print_box_content("/audio voice <v>    - Set voice", "")
            print_box_content("/audio backend <b>  - Set backend (groq/piper/elevenlabs)", "")
            print_box_content("/audio stop         - Stop current playback", "")
            print_box_content("/audio device <d>   - Set audio input device", "")
            print_box_content("", "")
            print_box_content("Tip: Add #audio to any message for one-time audio response", "info")
            print_box_footer()
            return

        cmd = args[0].lower()

        if cmd == "on":
            self.verbose_audio = True
            print_status("Verbose audio enabled - all responses will be spoken", "success")

        elif cmd == "off":
            self.verbose_audio = False
            print_status("Verbose audio disabled", "success")

        elif cmd == "stop":
            if hasattr(self, 'audio_player'):
                await self.audio_player.stop()
                print_status("Audio stopped", "success")

        elif cmd == "voice" and len(args) > 1:
            self.audio_player.tts_voice = args[1]
            print_status(f"Voice set to: {args[1]}", "success")

        elif cmd == "backend" and len(args) > 1:
            backend = args[1].lower()
            if backend in ["groq", "piper", "elevenlabs"]:
                self.audio_player.tts_backend = backend
                print_status(f"Backend set to: {backend}", "success")
            else:
                print_status("Valid backends: groq, piper, elevenlabs", "error")

        elif cmd == "lang" and len(args) > 1:
            self.audio_player.language = args[1]
            print_status(f"Language set to: {args[1]}", "success")

        elif cmd == "device":
            if len(args) > 1:
                self.audio_device_index = int(args[1])
            else:
                self._select_audio_device()

        else:
            print_status(f"Unknown audio command: {cmd} add args", "error")

    async def _cmd_skill(self, args: list[str]):
        """Handle /skill commands."""
        if not args:
            print_status("Usage: /skill <list|show|edit|delete|boost|merge|import|export> [id]", "warning")
            return

        action = args[0].lower()

        try:
            agent = await self.isaa_tools.get_agent(self.active_agent_name)
            # Access the SkillsManager via the ExecutionEngine
            engine = agent._get_execution_engine()
            sm = engine.skills_manager
            if sm.export_skills is None:
                from toolboxv2.mods.isaa.base.Agent.skills import add_anthropic_skill_io
                add_anthropic_skill_io(sm)
        except Exception as e:
            print_status(f"Could not access skills for agent '{self.active_agent_name}'", "error")
            import traceback
            traceback.print_exc()
            return


        if action == "show":
            if len(args) < 2:
                print_status("Usage: /skill show <skill_id>", "warning")
                return

            skill_id = args[1]
            skill = sm.skills.get(skill_id)

            if not skill:
                print_status(f"Skill '{skill_id}' not found.", "error")
                return

            print_box_header(f"Skill: {skill.name}", "🧠")
            print_box_content(f"ID: {skill.id}", "info")
            print_box_content(f"Source: {skill.source} | Confidence: {skill.confidence:.2f}", "info")
            print_box_content(f"Triggers: {', '.join(skill.triggers)}", "info")
            print_separator()
            print_status("Instruction:", "info")
            print_code_block(skill.instruction, "markdown")
            print_box_footer()

        elif action == "export":
            if len(args) < 3:
                print_status("Usage: /skill export <skill_id/all> <output_path>", "warning")
                return
            skill_id = args[1]
            output_path = args[2]
            if skill_id == "all":
                sm.export_skills(output_path)
            else:
                sm.export_to_skill_file(skill_id, output_path)
            print_status(f"Skill '{skill_id}' exported to '{output_path}'", "success")

        elif action == "import":
            if len(args) < 2:
                print_status("Usage: /skill import <input_path>", "warning")
                return
            input_path = args[1]
            results = sm.import_skills(input_path, True)
            for name, success in results.items():
                print(f"{'✅' if success else '❌'} {name}")
            print_status(f"Skill '{input_path}' imported", "success")

        elif action == "delete":
            if len(args) < 2:
                print_status("Usage: /skill delete <skill_id>", "warning")
                return

            skill_id = args[1]
            if skill_id in sm.skills:
                # Prevent deleting predefined skills unless forced (optional safety)
                if sm.skills[skill_id].source == "predefined":
                    print_status("Cannot delete predefined skills.", "warning")
                    return

                del sm.skills[skill_id]
                # Trigger save via checkpoint manager implicitly later or set dirty flag
                sm._skill_embeddings_dirty = True
                print_status(f"Skill '{skill_id}' deleted.", "success")
            else:
                print_status(f"Skill '{skill_id}' not found.", "error")
        elif action == "list":
            show_inactive_only = "--inactive" in args

            print_box_header(f"Skills: {self.active_agent_name}", "🧠")

            max_id_length = max(len(skill.id) for skill in sm.skills.values())
            max_name_length = max(len(skill.name) for skill in sm.skills.values())
            columns = [("ID", max_id_length), ("Name", max_name_length), ("Src", 10), ("Conf", 6), ("Active", 8)]
            widths = [max_id_length, max_name_length, 10, 6, 8]
            print_table_header(columns, widths)

            skills = sm.skills.values()

            if show_inactive_only:
                skills = [
                    s for s in skills
                    if s.source == "learned" and not s.is_active()
                ]

            sorted_skills = sorted(
                skills,
                key=lambda s: (s.source != "learned", s.confidence),
                reverse=False
            )

            for skill in sorted_skills:
                disp_id = skill.id if len(skill.id) < 24 else skill.id[:22] + ".."

                source_style = "green" if skill.source == "learned" else "grey"
                conf_style = "green" if skill.confidence > 0.8 else "yellow"
                active_style = "green" if skill.is_active() else "grey"

                print_table_row(
                    [
                        disp_id,
                        skill.name,
                        skill.source,
                        f"{skill.confidence:.2f}",
                        "YES" if skill.is_active() else "NO"
                    ],
                    widths,
                    ["cyan", "white", source_style, conf_style, active_style]
                )

            print_box_footer()

        elif action == "merge":
            if len(args) < 3:
                print_status("Usage: /skill merge <keep_id> <remove_id>", "warning")
                return

            keep_id, remove_id = args[1], args[2]

            keep_skill = sm.skills.get(keep_id)
            remove_skill = sm.skills.get(remove_id)

            if not keep_skill or not remove_skill:
                print_status("One or both skills not found.", "error")
                return

            if keep_id == remove_id:
                print_status("Cannot merge a skill into itself.", "warning")
                return

            # Merge logic
            keep_skill.merge_with(remove_skill)

            del sm.skills[remove_id]
            sm._skill_embeddings_dirty = True

            print_status(
                f"Merged skill '{remove_skill.name}' into '{keep_skill.name}'.",
                "success"
            )

        elif action == "boost":
            if len(args) < 3:
                print_status("Usage: /skill boost <skill_id> <amount>", "warning")
                return

            skill_id = args[1]

            try:
                amount = float(args[2])
            except ValueError:
                print_status("Boost amount must be a float (e.g. 0.3).", "error")
                return

            skill = sm.skills.get(skill_id)
            if not skill:
                print_status(f"Skill '{skill_id}' not found.", "error")
                return

            old_conf = skill.confidence
            skill.confidence = min(1.0, skill.confidence + amount)

            sm._skill_embeddings_dirty = True

            print_status(
                f"Skill '{skill.name}' boosted: {old_conf:.2f} → {skill.confidence:.2f}",
                "success"
            )


        elif action == "edit":
            if len(args) < 2:
                print_status("Usage: /skill edit <skill_id>", "warning")
                return

            skill_id = args[1]
            skill = sm.skills.get(skill_id)

            if not skill:
                print_status(f"Skill '{skill_id}' not found.", "error")
                return

            print_box_header(f"Editing: {skill.name}", "✏️")
            print_status("Current Instruction:", "info")
            print_code_block(skill.instruction, "markdown")
            print_separator()

            print_status("Enter NEW instruction (end with empty line) or type 'CANCEL':", "configure")

            lines: list[str] = []
            if self.prompt_session is not None:
                while True:
                    line = await self.prompt_session.prompt_async(
                        HTML("<style fg='grey'>... </style>")
                    )
                    if not line.strip():
                        break
                    if line.strip().upper() == "CANCEL":
                        print_status("Edit cancelled.", "warning")
                        return
                    lines.append(line)

            new_instruction = "\n".join(lines)
            if new_instruction:
                skill.instruction = new_instruction
                sm._skill_embeddings_dirty = True  # Force re-embedding
                print_status(f"Skill '{skill.name}' updated.", "success")

                # Try to save agent state
                try:
                    await agent.save()
                    print_status("Agent state saved.", "data")
                except Exception as e:
                    print_status(f"Warning: Could not save to disk immediately: {e}", "warning")

        else:
            print_status(f"Unknown skill action: {action}", "error")

    async def _cmd_feature(self, args: list[str]):
        """Handle /feature commands."""
        if len(args) < 2:
            print_status("Usage: /feature <action> [options]", "warning")
            return

        try:
            agent = await self.isaa_tools.get_agent(self.active_agent_name)
        except Exception as e:
            print_status(f"Could not access agent '{self.active_agent_name}'", "error")
            import traceback
            traceback.print_exc()
            return

        self.feature_manager.set_agent(agent)

        action = args[0].lower()
        if action == "list":
            print_box_header("Available Features", "📦")
            for feature in self.feature_manager.list_features():
                print_status(f"{feature}", "info")
        elif action == "enable":
            if len(args) < 2:
                print_status("Usage: /feature enable <feature> [options]", "warning")
                return
            feature = args[1].lower()
            if feature not in self.feature_manager.list_features():
                print_status(f"Feature '{feature}' not found.", "error")
                return
            self.feature_manager.enable(feature)
            print_status(f"Feature '{feature}' enabled.", "success")
        elif action == "disable":
            if len(args) < 2:
                print_status("Usage: /feature disable <feature> [options]", "warning")
                return
            feature = args[1].lower()
            if feature not in self.feature_manager.list_features():
                print_status(f"Feature '{feature}' not found.", "error")
                return
            # test active
            if self.feature_manager.is_enabled(feature):
                print_status(f"Feature '{feature}' is active. Please disable it first.", "warning")
                return
            self.feature_manager.disable(feature)
            print_status(f"Feature '{feature}' disabled.", "success")
        else:
            print_status(f"Unknown feature action: {action}", "error")


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

            # Check für #audio Flag
            wants_audio = user_input.strip().endswith("#audio")
            if wants_audio:
                user_input = user_input.rsplit("#audio", 1)[0].strip()

            # Kombiniere mit verbose_audio Setting
            should_speak = wants_audio or getattr(self, 'verbose_audio', False)

            # Audio Player starten wenn nötig
            if should_speak and hasattr(self, 'audio_player'):
                await self.audio_player.start()

            print_status(f"Processing with {self.active_agent_name}...", "progress")

            # Stream response
            final_response = ""
            full_response = ""
            current_sentence = ""
            already_played = []
            stop_for_speech = False
            async for chunk in agent.a_stream_verbose(
                query=user_input,
                session_id=self.active_session_id,
            ):

                if not full_response.endswith(chunk):
                    print(chunk, end="", flush=True)
                full_response += chunk

                if '```' in chunk:
                    stop_for_speech = True

                # Für Audio: Sammle bis Satzende
                if should_speak and hasattr(self, 'audio_player'):
                    if chunk == remove_styles(chunk) and not stop_for_speech:
                        current_sentence += chunk

                    # Check ob Satzende erreicht
                    if any(current_sentence.rstrip().endswith(p) for p in ['.', '!', '?', ':', '\n\n']):
                        get_app("ci.audio.bg.task").run_bg_task_advanced(self.audio_player.queue_text,remove_styles(current_sentence.strip()))
                        current_sentence = ""

            c_print()  # Newline after streaming

            # Try to visualize JSON responses
            try:
                if "```json" in full_response or "```" in full_response:
                    if "```json" in full_response:
                        final_response = full_response.split("```json")[1].split("```")[0].strip()
                    else:
                        final_response = full_response.split("```")[1]

                    data = json.loads(final_response)
                    if isinstance(data, list) and len(data) == 1:
                        data = data[0]
                    if isinstance(data, dict):
                        print_separator()
                        await visualize_data_terminal(
                            data, agent, max_preview_chars=max(len(final_response), 8000)
                        )
            except (json.JSONDecodeError, Exception):
                import traceback
                traceback.print_exc()
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
                if hasattr(self, "_last_transcription") and self._last_transcription is not None:
                    user_input = self._last_transcription + " " + user_input
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
    try:
        from toolboxv2.mods.isaa.extras.discord_interface.integration_example import patch_cli_for_discord
        patch_cli_for_discord(host)
        print("Discord integration enabled.")
    except ImportError as e:
        import traceback
        traceback.print_exc()
        print(e)
        print("⚠️ Discord integration not available.")
        pass
    await host.run()


def main():
    """Synchronous entry point."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
