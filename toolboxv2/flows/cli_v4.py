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
import logging
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from prompt_toolkit.document import Document

from toolboxv2.mods.isaa.extras.dream_graph import dream_with_viz_v2
from toolboxv2.mods.isaa.extras.zen.zen_plus import ZenPlus

# Suppress noisy loggers
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from prompt_toolkit import PromptSession, ANSI
from prompt_toolkit.completion import FuzzyCompleter, NestedCompleter, PathCompleter, Completer, \
    Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout

# ToolBoxV2 Imports
from toolboxv2 import get_app, remove_styles, get_logger

# ISAA Agent Imports
from toolboxv2.mods.isaa.base.Agent.builder import (
    FlowAgentBuilder,
)
from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
from toolboxv2.mods.isaa.base.Agent.instant_data_vis import (
    visualize_data_terminal,
)
from toolboxv2.mods.isaa.base.Agent.vfs_v2 import FileBackingType, VFSFile
from toolboxv2.mods.isaa.base.AgentUtils import detect_shell
from toolboxv2.mods.isaa.base.audio_io.audioIo import AudioStreamPlayer

from toolboxv2.mods.isaa.extras.zen.zen_renderer import ZenRendererV2
from toolboxv2.mods.isaa.extras.jobs import JobDefinition, TriggerConfig, JobScheduler

import html
from pathlib import Path
from toolboxv2.utils.extras.mkdocs import DocsSystem
from toolboxv2 import init_cwd, tb_root_dir
import json
from prompt_toolkit import print_formatted_text, HTML
from toolboxv2.mods.isaa.CodingAgent.coder import CoderAgent

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


    # Zen Colors
    ZEN_DIM = '#6b7280'
    ZEN_CYAN = '#67e8f9'
    ZEN_AMBER = '#fbbf24'
    ZEN_GREEN = '#4ade80'
    ZEN_RED = '#fb7185'


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
    return html.escape(str(text).encode().decode(encoding="utf-8", errors="replace"))


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
        try:
            print_formatted_text(HTML(esc(text)), **kwargs)
        except:
            print(text)


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
    run_id: str
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


def load_web_auto_feature(fm):
    from toolboxv2.mods.isaa.extras.web_helper.tooklit import PlaywrightProxy
    proxy = PlaywrightProxy(full=False, headless=True)
    tools_set = [None]
    def enable(agent):
        proxy.start()
        tools_set[0] = proxy.build_agent_tools()
        agent.add_tools(tools_set[0])
        print_status("Mini Web Automation enabled.", "success")

    def disable(agent):
        proxy.shutdown()
        agent.remove_tools(tools_set[0])
        print_status("Mini Web Automation disabled.", "success")

    fm.add_feature("mini_web_auto", activation_f=enable, deactivation_f=disable)


def load_full_web_auto_feature(fm):
    from toolboxv2.mods.isaa.extras.web_helper.tooklit import PlaywrightProxy
    proxy = PlaywrightProxy(full=True, headless=True)
    tools_set = [None]
    def enable(agent):
        proxy.start()
        tools_set[0] = proxy.build_agent_tools()
        agent.add_tools(tools_set[0])
        print_status("Full Web Automation enabled.", "success")

    def disable(agent):
        proxy.shutdown()
        agent.remove_tools(tools_set[0])
        print_status("Full Web Automation disabled.", "success")

    fm.add_feature("full_web_auto", activation_f=enable, deactivation_f=disable)

def load_coder_toolkit(fm):
    from toolboxv2.mods.isaa.CodingAgent.coder_toolset import coder_register_flow_tools
    from toolboxv2 import init_cwd
    pool = [None]

    def enable(agent):
        c_print(f"Starting coder from: {str(init_cwd)}")
        _pool, tools = coder_register_flow_tools(agent, str(init_cwd))
        pool[0] = _pool
        agent.add_tools(tools)
        print_status("Coder enabled.", "success")

    def disable(agent):
        _pool, tools = coder_register_flow_tools(agent, init_cwd)
        agent.remove_tools(tools)
        print_status("Coder disabled.", "success")

    fm.add_feature("coder", activation_f=enable, deactivation_f=disable)

def load_chain_toolkit(fm):
    from toolboxv2.mods.isaa.base.chain.chain_tools import create_chain_tools
    tools_set = [None]

    agent_registry = {}
    coder_registry = {}
    format_registry = {}

    def enable(agent):
        tools_set[0] = create_chain_tools(agent, agent_registry=agent_registry, coder_registry=coder_registry, format_registry=format_registry)
        agent.add_tools(tools_set[0])
        print_status("Chains enabled.", "success")

    def disable(agent):
        agent.remove_tools(tools_set[0])
        print_status("Chains disabled.", "success")

    fm.add_feature("chain", activation_f=enable, deactivation_f=disable)

def load_execute(fm):
    from toolboxv2.mods.isaa.base.Agent.executors import register_code_exec_tools
    tools_set = [None]

    def enable(agent):
        tools_set[0] = register_code_exec_tools(agent)[0]
        print_status("exec_code enabled.", "success")

    def disable(agent):
        agent.remove_tools(tools_set[0])
        print_status("exec_code disabled.", "success")

    fm.add_feature("chain", activation_f=enable, deactivation_f=disable)

def load_docs_feature(fm):
    """
    Documentation Feature - Integrates mkdocs-based docs system.

    When active in toolboxv2 directory: Uses tb_root_dir.parent/docs as doc dir.
    When active in another directory: Prompts user to select docs directory.
    """
    docs_system = [None]  # Mutable container for the docs system instance
    docs_tools = [None]   # Mutable container for the tool list

    def enable(agent):
        """Enable documentation system and add tools to agent."""
        try:
            # Determine docs directory based on current working directory
            current_dir = init_cwd
            project_root = tb_root_dir

            # Check if we're in toolboxv2 or its parent
            if current_dir == project_root or current_dir == project_root.parent or project_root in current_dir.parents:
                docs_dir = project_root.parent / "docs"
                c_print(f"<style fg='ansicyan'>📚 Auto-detected docs dir:</style> <style fg='ansigreen'>{docs_dir}</style>")
            else:
                # Prompt user for docs directory
                docs_input = input("Enter documentation directory path: ")
                docs_dir = Path(docs_input).expanduser().resolve()

            if not docs_dir.exists():
                c_print(f"<style fg='ansired'>⚠️ Docs directory not found: {docs_dir}</style>")
                c_print("<style fg='ansiyellow'>Creating minimal docs structure...</style>")
                docs_dir.mkdir(parents=True, exist_ok=True)

            # Create docs system instance
            system = DocsSystem(
                project_root=project_root,
                docs_root=docs_dir,
                include_dirs=["toolboxv2", "flows", "mods", "utils", "docs"]
            )

            # Initialize (load existing index or build new one)
            import asyncio
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(system.initialize())
            c_print(f"<style fg='ansigreen'>✓ Docs initialized:</style> {result['sections']} sections, {result['elements']} elements")

            docs_system[0] = system

            # Prepare tool list with proper descriptions
            tools = [
                {
                    "tool_func": system.read,
                    "name": "docs_reader",
                    "description": "Durchsucht die Dokumentation nach relevanten Abschnitten basierend auf einer Suchanfrage. Nützlich um schnell Informationen aus der Dokumentation zu finden ohne manuell zu suchen. Gibt strukturierte Ergebnisse mit Relevanz-Scores zurück.",
                    "category": ["docs", "read", "search"],
                    "flags": {}
                },
                {
                    "tool_func": system.write,
                    "name": "docs_writer",
                    "description": "Schreibt neue Dokumentations-Dateien oder aktualisiert existierende. Unterstützt Markdown-Formatierung und speichert im docs-Verzeichnis. Erstellt automatisch die nötige Verzeichnisstruktur wenn nötig. Ideal um Ergebnisse und Erkenntnisse zu dokumentieren.",
                    "category": ["docs", "write"],
                    "flags": {}
                },
                {
                    "tool_func": system.lookup_code,
                    "name": "docs_lookup",
                    "description": "Sucht nach Code-Elementen (Klassen, Funktionen, Module, etc.) im gesamten Codebase. Gibt Definitionen, Signaturen und Docstrings zurück. Nützlich um Implementierungsdetails schnell zu verstehen ohne durch Dateien navigieren zu müssen.",
                    "category": ["docs", "code", "search"],
                    "flags": {}
                },
                {
                    "tool_func": system.sync,
                    "name": "docs_sync",
                    "description": "Synchronisiert Änderungen aus der VFS zurück zum Dateisystem. Stellt sicher dass Änderungen persistent gespeichert werden. Sollte regelmäßig aufgerufen werden nachdem Änderungen vorgenommen wurden.",
                    "category": ["docs", "sync"],
                    "flags": {}
                },
                {
                    "tool_func": system.initialize,
                    "name": "docs_init",
                    "description": "Baut den Dokumentations-Index neu auf. Nützlich wenn neue Dateien hinzugefügt wurden oder der Index veraltet ist. Indiziert Markdown-, Python- und JavaScript/TypeScript-Dateien aus den konfigurierten include_dirs.",
                    "category": ["docs", "index"],
                    "flags": {}
                },
                {
                    "tool_func": system.get_task_context,
                    "name": "get_task_context",
                    "description": "Generiert Kontext-Informationen für Aufgaben wie relevante Dateien, Klassen und Dokumentation basierend auf einer Aufgabenbeschreibung. Hilft dem Agent die Aufgabe besser zu verstehen und die richtigen Ressourcen zu finden.",
                    "category": ["docs", "context"],
                    "flags": {}
                }
            ]

            docs_tools[0] = tools
            agent.add_tools(tools)

            print_status("Documentation feature enabled.", "success")

        except Exception as e:
            c_print(f"<style fg='ansired'>✗ Failed to enable docs: {e}</style>")
            import traceback
            traceback.print_exc()

    def disable(agent):
        """Disable documentation system."""
        try:
            if docs_tools[0]:
                agent.remove_tools(docs_tools[0])
            docs_system[0] = None
            docs_tools[0] = None
            print_status("Documentation feature disabled.", "success")
        except Exception as e:
            c_print(f"<style fg='ansired'>✗ Failed to disable docs: {e}</style>")

    fm.add_feature("docs", activation_f=enable, deactivation_f=disable)


ALL_FEATURES = {
    "desktop_auto": load_desktop_auto_feature,
    "mini_web_auto": load_web_auto_feature,
    "full_web_auto": load_full_web_auto_feature,
    "coder": load_coder_toolkit,
    "chain": load_chain_toolkit,
    "execute": load_execute,
}


# ─── Subcommand Definitions ─────────────────────────────────────────────────

# Maps subcommand → argument spec
# Spec: list of (arg_type, required) tuples
#   arg_type: "vfs_path" | "vfs_file" | "vfs_dir" | "local_path" | "mount"
#             | "dirty" | "subcmd:<options>" | None (no completion)

SUBCOMMANDS: dict[str, dict] = {
    "mount":       {"args": ["local_path", "vfs_path"], "flags": ["--readonly", "--no-sync"]},
    "unmount":     {"args": ["mount"],                  "flags": ["--no-save"]},
    "sync":        {"args": ["vfs_dirty_or_all"],       "flags": []},
    "refresh":     {"args": ["mount"],                  "flags": []},
    "pull":        {"args": ["vfs_path"],               "flags": []},
    "save":        {"args": ["vfs_path", "local_path"], "flags": []},
    "mounts":      {"args": [],                         "flags": []},
    "dirty":       {"args": [],                         "flags": []},
    "rm":          {"args": ["vfs_path"],               "flags": []},
    "remove":      {"args": ["vfs_path"],               "flags": []},
    "sys-add":     {"args": ["local_path", "vfs_path"], "flags": ["--refresh"]},
    "sys-remove":  {"args": ["vfs_path"],               "flags": []},
    "sys-refresh": {"args": ["vfs_path"],               "flags": []},
    "sys-list":    {"args": [],                         "flags": []},
    "obsidian":    {"args": ["subcmd:mount|unmount|sync", "local_path", "vfs_path"], "flags": []},
}


class VFSCompleter(Completer):
    """
    Hierarchischer VFS-Completer.

    Wird als Sub-Completer eingehängt — empfängt nur den Text nach '/vfs '.
    Beispiel: User tippt '/vfs mount /ho' → dieser Completer sieht 'mount /ho'.
    """

    def __init__(self, vfs: 'VirtualFileSystemV2'):
        self._vfs = vfs

    # ─── Entry Point ─────────────────────────────────────────────────────

    def get_completions(
        self, document: 'Document', complete_event: 'CompleteEvent'
    ):
        text = document.text_before_cursor
        stripped = text.lstrip()

        # ── Case 1: Kein Text oder kein Space → Subcommand + Top-Level VFS Pfade
        if " " not in stripped:
            yield from self._complete_subcommand_or_path(stripped)
            return

        # ── Case 2: Subcommand erkannt → Argumente completieren
        parts = stripped.split(None, 1)  # maxsplit=1
        subcmd = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

        if subcmd in SUBCOMMANDS:
            yield from self._complete_subcmd_args(subcmd, rest)
        else:
            # Kein bekannter Subcommand → als VFS-Pfad interpretieren
            yield from self._complete_vfs_path(stripped)

    # ─── Subcommand / Top-Level Path Completion ──────────────────────────

    def _complete_subcommand_or_path(self, partial: str):
        """Complete subcommand names AND top-level VFS paths."""
        partial_lower = partial.lower()

        # Subcommands
        for cmd in SUBCOMMANDS:
            if cmd.startswith(partial_lower):
                yield Completion(cmd, start_position=-len(partial), display=cmd)

        # Top-level VFS paths (direct access: /vfs <path>)
        yield from self._complete_vfs_path(partial)

    # ─── Subcommand Argument Completion ──────────────────────────────────

    def _complete_subcmd_args(self, subcmd: str, rest: str):
        """Complete arguments for a known subcommand."""
        spec = SUBCOMMANDS[subcmd]
        arg_defs = spec["args"]
        flags = spec["flags"]

        # Split rest into tokens, but keep track of current partial
        tokens = rest.split()
        # If rest ends with space → completing NEW token, else completing last token
        if rest.endswith(" ") or not rest:
            completed_args = tokens
            current_partial = ""
        else:
            completed_args = tokens[:-1]
            current_partial = tokens[-1]

        # Filter out already-used flags from completed args
        used_flags = {t for t in completed_args if t.startswith("--")}
        non_flag_args = [t for t in completed_args if not t.startswith("--")]

        # ── Flag completion (if partial starts with -)
        if current_partial.startswith("-"):
            for flag in flags:
                if flag not in used_flags and flag.startswith(current_partial):
                    yield Completion(
                        flag, start_position=-len(current_partial), display=flag
                    )
            return

        # ── Determine which positional arg we're on
        arg_index = len(non_flag_args)

        if arg_index < len(arg_defs):
            arg_type = arg_defs[arg_index]
            yield from self._complete_arg_type(arg_type, current_partial)

        # ── Always offer remaining flags
        if not current_partial or current_partial.startswith("-"):
            for flag in flags:
                if flag not in used_flags and flag.startswith(current_partial):
                    yield Completion(
                        flag, start_position=-len(current_partial), display=flag
                    )

    def _complete_arg_type(self, arg_type: str, partial: str):
        """Dispatch completion based on argument type."""
        if arg_type == "vfs_path":
            yield from self._complete_vfs_path(partial)
        elif arg_type == "vfs_dirty_or_all":
            yield from self._complete_vfs_dirty_or_all(partial)
        elif arg_type == "local_path":
            yield from self._complete_local_path(partial)
        elif arg_type == "mount":
            yield from self._complete_mount_points(partial)
        elif arg_type.startswith("subcmd:"):
            options = arg_type.split(":", 1)[1].split("|")
            for opt in options:
                if opt.startswith(partial):
                    yield Completion(opt, start_position=-len(partial), display=opt)

    # ─── VFS Path Completion (hierarchisch!) ─────────────────────────────

    def _complete_vfs_path(self, partial: str):
        """
        Hierarchische VFS-Pfad-Completion.

        Löst nur direkte Kinder des aktuellen Parent-Dirs auf.
        '/project/sr' → listet Kinder von /project die mit 'sr' anfangen.
        """
        # Normalize: ensure starts with /
        if not partial:
            partial_path = "/"
        elif not partial.startswith("/"):
            partial_path = "/" + partial
        else:
            partial_path = partial

        # Determine parent dir and search prefix
        if partial_path.endswith("/") and self._vfs._is_directory(partial_path.rstrip("/") or "/"):
            # User typed full dir with trailing slash → list children
            parent = partial_path.rstrip("/") or "/"
            search = ""
        elif "/" in partial_path[1:]:
            # Has path separator → split into parent + partial name
            parent = partial_path.rsplit("/", 1)[0] or "/"
            search = partial_path.rsplit("/", 1)[1].lower()
        else:
            # Top-level: /something
            parent = "/"
            search = partial_path.lstrip("/").lower()

        # Verify parent exists
        if not self._vfs._is_directory(parent):
            return

        # Get direct children
        contents = self._vfs._list_directory_contents(parent)

        for item in contents:
            name = item["name"]
            if search and not name.lower().startswith(search):
                continue

            is_dir = item["type"] == "directory"
            suffix = "/" if is_dir else ""
            full_path = f"{parent.rstrip('/')}/{name}{suffix}"

            # Display: just the name + type indicator
            if is_dir:
                display = f"📁 {name}/"
            else:
                f = self._vfs.files.get(item["path"])
                icon = f.file_type.icon if f and f.file_type else "📄"
                state = " ●" if f and f.state == "open" else ""
                dirty = " ✱" if f and hasattr(f, "is_dirty") and f.is_dirty else ""
                display = f"{icon} {name}{state}{dirty}"

            yield Completion(
                full_path,
                start_position=-len(partial),
                display=display,
                display_meta=item.get("file_type", "") if not is_dir else "",
            )

    # ─── Dirty / All VFS Paths ───────────────────────────────────────────

    def _complete_vfs_dirty_or_all(self, partial: str):
        """Complete with dirty files first, then fall back to all VFS paths."""
        partial_lower = partial.lower() if partial else ""

        # Dirty files zuerst (Priorität für sync)
        dirty_yielded = set()
        for path, f in self._vfs.files.items():
            if hasattr(f, "is_dirty") and f.is_dirty:
                if not partial or path.lower().startswith(partial_lower) or (
                    not partial.startswith("/") and path.lower().startswith("/" + partial_lower)
                ):
                    dirty_yielded.add(path)
                    yield Completion(
                        path,
                        start_position=-len(partial),
                        display=f"✱ {path}",
                        display_meta="modified",
                    )

        # Dann normale Pfad-Completion (ohne bereits gezeigte)
        for completion in self._complete_vfs_path(partial):
            if completion.text not in dirty_yielded:
                yield completion

    # ─── Mount Point Completion ──────────────────────────────────────────

    def _complete_mount_points(self, partial: str):
        """Complete with active mount points."""
        partial_lower = partial.lower() if partial else ""

        if not hasattr(self._vfs, "mounts"):
            return

        for mount_path, mount in self._vfs.mounts.items():
            if not partial or mount_path.lower().startswith(partial_lower) or (
                not partial.startswith("/") and mount_path.lower().startswith("/" + partial_lower)
            ):
                local = getattr(mount, "local_path", "")
                yield Completion(
                    mount_path,
                    start_position=-len(partial),
                    display=f"📂 {mount_path}",
                    display_meta=local,
                )

    # ─── Local Path Completion ───────────────────────────────────────────

    def _complete_local_path(self, partial: str):
        """
        Complete local filesystem paths.

        Handles ~ expansion and hierarchical directory traversal.
        """
        if not partial:
            partial = "./"

        # Expand user home
        expanded = os.path.expanduser(partial)

        # Determine dir to scan and prefix to match
        if os.path.isdir(expanded):
            scan_dir = expanded
            name_prefix = ""
            # Ensure partial ends with separator for correct start_position
            if not partial.endswith(os.sep) and not partial.endswith("/"):
                partial += os.sep
        else:
            scan_dir = os.path.dirname(expanded) or "."
            name_prefix = os.path.basename(expanded).lower()

        if not os.path.isdir(scan_dir):
            return

        try:
            entries = os.listdir(scan_dir)
        except PermissionError:
            return

        entries.sort()

        for entry in entries:
            # Skip hidden unless user explicitly typed dot
            if entry.startswith(".") and not name_prefix.startswith("."):
                continue

            if name_prefix and not entry.lower().startswith(name_prefix):
                continue

            full = os.path.join(scan_dir, entry)
            is_dir = os.path.isdir(full)

            # Build the completion text preserving user's original format
            if partial.startswith("~"):
                # Keep ~ prefix
                rel_to_home = os.path.relpath(full, os.path.expanduser("~"))
                completion_text = f"~/{rel_to_home}"
            elif os.path.isabs(partial) or os.path.isabs(expanded):
                completion_text = full
            else:
                try:
                    completion_text = os.path.relpath(full)
                except ValueError:
                    completion_text = full

            if is_dir:
                completion_text += os.sep
                display = f"📁 {entry}/"
            else:
                display = f"  {entry}"

            yield Completion(
                completion_text,
                start_position=-len(partial),
                display=display,
            )
class SmartCompleter(Completer):
    """FuzzyCompleter für alles AUSSER /vfs — dort direkt, damit Tab akzeptiert."""

    def __init__(self, nested_dict: dict, vfs_completer: VFSCompleter | None = None):
        self._fuzzy = FuzzyCompleter(NestedCompleter.from_nested_dict(nested_dict))
        self._vfs = vfs_completer

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        if text.startswith("/vfs ") and self._vfs:
            sub_doc = Document(text[5:], len(text[5:]))
            yield from self._vfs.get_completions(sub_doc, complete_event)
        else:
            yield from self._fuzzy.get_completions(document, complete_event)

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
        self.zen_plus_mode = False
        self.app = app_instance or get_app("isaa-host")
        def _(*args, **k):
            text = " ".join(str(a) for a in args)
            try:
                print_formatted_text(ANSI(text), **k)
            except:
                print_formatted_text(ANSI(text))
        self.app._print = _

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

        self.active_coder: CoderAgent | None = None
        from toolboxv2 import init_cwd
        self.init_dir = init_cwd
        self.active_coder_path: str | None = init_cwd

        # Prompt Toolkit setup
        self.history = FileHistory(str(self.history_file))
        self.key_bindings = self._create_key_bindings()
        self.prompt_session: PromptSession | None = None

        # Self Agent initialization flag
        self._self_agent_initialized = False
        self._active_renderer: ZenRendererV2 | None = None

        # Job Scheduler
        self.jobs_file = Path(self.app.appdata) / "icli" / "isaa_host_jobs.json"
        self.job_scheduler: JobScheduler | None = None

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
                await self._cmd_vfs(["list"])
                await self._cmd_skill(["list"])
                await self._cmd_mcp(["list"])
                await self._cmd_session(["list"])
                await self._cmd_session(["show"])
            asyncio.create_task(__())

        @kb.add("f6")
        def _(event):
            """Toggle renderer minimize with F6."""
            if hasattr(self, '_active_renderer') and self._active_renderer:
                self._active_renderer.toggle_minimize()
            else:
                print("Not active")

        @kb.add("f2")
        def _(event):
            self.zen_plus_mode = not self.zen_plus_mode
            mode = "ZEN+" if self.zen_plus_mode else "ZEN"
            from prompt_toolkit import print_formatted_text, HTML
            print_formatted_text(HTML(
                f"<style fg='#67e8f9'>  ◎ Mode: {mode}</style>"
            ))

        @kb.add("tab")
        def handle_tab(event):
            buf = event.app.current_buffer
            if buf.complete_state:
                # Completion offen: aktuelle oder erste akzeptieren
                if buf.complete_state.current_completion is None:
                    buf.complete_state.go_to_index(0)
                buf.apply_completion(buf.complete_state.current_completion)
                # Directory → nächste Ebene
                if buf.text.rstrip().endswith("/"):
                    buf.start_completion()
                    if buf.complete_state:
                        buf.complete_state.go_to_index(0)
            else:
                # Completion starten + erste vorauswählen
                buf.start_completion()
                if buf.complete_state:
                    buf.complete_state.go_to_index(0)

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

        # ===== JOB MANAGEMENT TOOLS =====

        async def cli_create_job(
            name: str,
            agent_name: str,
            query: str,
            trigger_type: str,
            trigger_config: dict | None = None,
            timeout_seconds: int = 300,
            session_id: str = "default",
            # Trigger-specific parameters (for backward compatibility)
            cron_expression: str | None = None,
            interval_seconds: int | None = None,
            at_datetime: str | None = None,
            watch_job_id: str | None = None,
            watch_path: str | None = None,
            watch_patterns: list[str] | None = None,
            webhook_path: str | None = None,
            idle_seconds: int | None = None,
            agent_idle_seconds: int | None = None,
            # Allow additional trigger parameters via **kwargs
            **extra_trigger_kwargs
        ) -> str:
            """
            Create a persistent scheduled job that fires an agent query on a trigger.

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            ⚠️  IMPORTANT: TRIGGER PARAMETER USAGE
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            Trigger-specific parameters (cron_expression, interval_seconds, etc.)
            can be passed in THREE ways:

            1. DIRECTLY as function parameters (recommended):
               createJob(name="my-job", trigger_type="on_cron", cron_expression="0 2 * * 0")

            2. Via trigger_config dict:
               createJob(name="my-job", trigger_type="on_cron",
                        trigger_config={"cron_expression": "0 2 * * 0"})

            3. Mixed - parameters in trigger_config override direct parameters

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            Args:
                name: Human-readable job name
                agent_name: Which agent runs this job (e.g. 'self', 'researcher')
                query: The prompt/query to send to the agent
                trigger_type: Trigger type (on_time, on_interval, on_cron, on_cli_start,
                               on_cli_exit, on_job_completed, on_job_failed, on_file_changed,
                               on_network_available, on_system_idle, on_webhook_received, on_agent_idle, on_dream_start, on_dream_end,
                               on_dream_budget_hit, on_dream_skill_evolved, etc.)
                trigger_config: Optional trigger parameters dict - overrides direct params
                timeout_seconds: Max execution time in seconds (default 300)
                session_id: Session ID for the agent (default 'default')

            Trigger-Specific Parameters (for on_cron, on_interval, etc.):
                cron_expression: Cron schedule string (e.g. "0 2 * * 0" for Sunday 2am)
                interval_seconds: Fire every N seconds
                at_datetime: ISO datetime string for one-time execution
                watch_job_id: Job ID to watch for job_completion/failed/timeout triggers
                watch_path: File/directory path to watch for on_file_changed
                watch_patterns: Glob patterns for file watching
                webhook_path: HTTP path for on_webhook_received
                idle_seconds: Idle threshold for on_system_idle

            Returns:
                Job ID or error message

            Examples:
                # Cron job (weekly Sunday 2am)
                createJob(name="weekly-update", trigger_type="on_cron",
                         agent_name="self", query="run updates",
                         cron_expression="0 2 * * 0")

                # Interval job (every 5 minutes)
                createJob(name="heartbeat", trigger_type="on_interval",
                         agent_name="self", query="ping server",
                         interval_seconds=300)

                # Using trigger_config dict
                createJob(name="daily-backup", trigger_type="on_cron",
                         agent_name="self", query="backup database",
                         trigger_config={"cron_expression": "0 3 * * *"})
            """
            if not host_ref.job_scheduler:
                return "✗ Job scheduler not initialized"

            try:
                import traceback

                # Build trigger config from multiple sources
                # Priority: trigger_config > direct parameters > None
                trigger_params = {}

                # Add direct parameters if provided
                if cron_expression is not None:
                    trigger_params["cron_expression"] = cron_expression
                if interval_seconds is not None:
                    trigger_params["interval_seconds"] = interval_seconds
                if at_datetime is not None:
                    trigger_params["at_datetime"] = at_datetime
                if watch_job_id is not None:
                    trigger_params["watch_job_id"] = watch_job_id
                if watch_path is not None:
                    trigger_params["watch_path"] = watch_path
                if watch_patterns is not None:
                    trigger_params["watch_patterns"] = watch_patterns
                if webhook_path is not None:
                    trigger_params["webhook_path"] = webhook_path
                if idle_seconds is not None:
                    trigger_params["idle_seconds"] = idle_seconds
                if agent_idle_seconds is not None:
                    trigger_params["agent_idle_seconds"] = agent_idle_seconds

                # Add extra kwargs (for extensibility)
                trigger_params.update(extra_trigger_kwargs)

                # Override with trigger_config if provided
                if trigger_config:
                    trigger_params.update(trigger_config)

                # Create TriggerConfig
                tc = TriggerConfig(trigger_type=trigger_type)

                # Apply all trigger parameters
                for k, v in trigger_params.items():
                    if hasattr(tc, k):
                        setattr(tc, k, v)
                    else:
                        # Warn about unknown parameters but don't fail
                        c_print(f"Unknown trigger parameter: {k}={v}")

                # Create job definition
                job = JobDefinition(
                    job_id=JobDefinition.generate_id(),
                    name=name,
                    agent_name=agent_name,
                    query=query,
                    trigger=tc,
                    timeout_seconds=timeout_seconds,
                    session_id=session_id,
                )

                # Add job to scheduler
                job_id = host_ref.job_scheduler.add_job(job)

                return (
                    f"✓ Job created successfully!\n"
                    f"  Job ID: {job_id}\n"
                    f"  Name: {name}\n"
                    f"  Trigger: {trigger_type}\n"
                    f"  Config: {trigger_params or '(default)'}\n"
                    f"  Agent: {agent_name}\n"
                    f"  Verify with: listJobs()"
                )

            except Exception as e:
                # LÖSUNG 2: Bessere Fehlerbehandlung mit Traceback
                import traceback
                error_details = traceback.format_exc()

                get_logger().error(f"Failed to create job '{name}': {e}\n{error_details}")

                return (
                    f"✗ Failed to create job '{name}'\n"
                    f"  Error: {e}\n"
                    f"  Trigger Type: {trigger_type}\n"
                    f"  Parameters: {locals().get('trigger_params', {})}\n\n"
                    f"  Debug Info:\n"
                    f"  {error_details}"
                )

        async def cli_delete_job(job_id: str) -> str:
            """
            Delete a scheduled job by ID.

            Args:
                job_id: The job ID to delete

            Returns:
                Status message
            """
            if not host_ref.job_scheduler:
                return "Job scheduler not initialized"
            if host_ref.job_scheduler.remove_job(job_id):
                return f"✓ Job {job_id} deleted"
            return f"✗ Job {job_id} not found"

        async def cli_list_jobs() -> str:
            """
            List all scheduled jobs with their status.

            Returns:
                Formatted list of all jobs
            """
            if not host_ref.job_scheduler:
                return "Job scheduler not initialized"
            jobs = host_ref.job_scheduler.list_jobs()
            if not jobs:
                return "No scheduled jobs."
            lines = ["=== Scheduled Jobs ===\n"]
            for j in jobs:
                lines.append(
                    f"  {j.job_id} | {j.name} | {j.trigger.trigger_type} | "
                    f"{j.status} | runs:{j.run_count} fails:{j.fail_count}"
                )
                if j.last_result:
                    lines.append(f"    Last: {j.last_result} at {j.last_run_at}")
            return "\n".join(lines)

        async def cli_create_dream_job(
            agent_name: str = "self",
            trigger_type: str = "on_cron",
            cron_expression: str = "0 3 * * *",
            agent_idle_seconds: int | None = None,
            max_budget: int = 3000,
            do_skill_split: bool = True,
            do_skill_evolve: bool = True,
            do_persona_evolve: bool = True,
            do_create_new: bool = True,
            hard_stop: bool = False,
        ) -> str:
            """
            Create a dream job (async meta-learning cycle).

            Args:
                agent_name: Agent to dream (default: self)
                trigger_type: on_cron (default), on_agent_idle, on_job_completed, etc.
                cron_expression: Cron schedule (default: nightly 03:00)
                agent_idle_seconds: Idle threshold for on_agent_idle trigger
                max_budget: Max tokens for LLM calls during dream
                do_skill_split: Split bloated skills into sub-skills
                do_skill_evolve: Refine instructions from failure patterns
                do_persona_evolve: Adjust persona profiles
                do_create_new: Allow creation of new skills/personas
                hard_stop: Abort on first error (False = skip & continue)

            Examples:
                createDreamJob()                                          # Nightly at 03:00
                createDreamJob(trigger_type="on_agent_idle", agent_idle_seconds=600)  # After 10min idle
                createDreamJob(trigger_type="on_job_completed")           # After every successful job
            """
            dream_config = {
                "max_budget": max_budget,
                "do_skill_split": do_skill_split,
                "do_skill_evolve": do_skill_evolve,
                "do_persona_evolve": do_persona_evolve,
                "do_create_new": do_create_new,
                "hard_stop": hard_stop,
            }

            return await cli_create_job(
                name=f"dream-{agent_name}",
                agent_name=agent_name,
                query="__dream__",
                trigger_type=trigger_type,
                cron_expression=cron_expression if trigger_type == "on_cron" else None,
                agent_idle_seconds=agent_idle_seconds if trigger_type == "on_agent_idle" else None,
                trigger_config={"extra": {"dream_config": dream_config}},
                timeout_seconds=600,
            )

        builder.add_tool(
            cli_create_dream_job,
            "createDreamJob",
            "Create a dream job (async meta-learning) with configurable triggers",
            category=["job_management"],
        )

        builder.add_tool(
            cli_create_job,
            "createJob",
            "Create a persistent scheduled job that fires an agent on a trigger",
            category=["job_management"],
        )
        builder.add_tool(
            cli_delete_job,
            "deleteJob",
            "Delete a scheduled job",
            category=["job_management"],
        )
        builder.add_tool(
            cli_list_jobs,
            "listJobs",
            "List all scheduled jobs with status",
            category=["job_management"],
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

            run_id = uuid.uuid4().hex[:8]

            if wait:
                # For synchronous wait, we still might want to see it in stats if possible,
                # but usually CLI blocks here.
                result = await agent.a_run(query=task, session_id=session_id, execution_id=run_id)
                return str(result)
            else:
                self._task_counter += 1
                task_id = f"task_{self._task_counter}_{agent_name}"

                async def run_task():
                    try:
                        # Pass execution_id explicitly to link with engine
                        result = await agent.a_run(query=task, session_id=session_id, execution_id=run_id)
                        if task_id in self.background_tasks:
                            self.background_tasks[task_id].status = "completed"
                        return result
                    except asyncio.CancelledError:
                        if task_id in self.background_tasks:
                            self.background_tasks[task_id].status = "cancelled"
                        raise
                    except Exception as e:
                        if task_id in self.background_tasks:
                            self.background_tasks[task_id].status = "failed"
                        print_status(f"Task {task_id} failed: {e}", "error")
                        raise

                async_task = asyncio.create_task(run_task())

                self.background_tasks[task_id] = BackgroundTask(
                    task_id=task_id,
                    run_id=run_id,  # Store run_id
                    agent_name=agent_name,
                    query=task[:100],
                    task=async_task,
                )
                zp = ZenPlus.get()
                if zp.active:
                    zp.inject_job(task_id, agent_name, task[:60], "running",
                                  run_id=run_id, kind="delegate")

                return f"✓ Background task started: {task_id} (RunID: {run_id})"

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

    def _build_completer(self) -> tuple[dict[
        str | Any, None | dict[str, dict[str, None] | None] | dict[str, PathCompleter | None | dict[str, None]] | dict[
            str | Any, dict[Any, None] | None | dict[str, dict] | dict[str, dict[Any, None]] | Any] | dict[
            str, dict[Any, None] | None] | dict[str, dict[Any, Any] | None] | Any], VFSCompleter | dict[str, None]]:
        """Build nested completer dictionary."""
        agents = self.isaa_tools.config.get("agents-name-list", ["self"])

        # Try to get VFS files, dirs, and mounts for autocomplete
        session = None
        is_vfs = False
        model_options: dict = {}
        current_skills: dict = {}
        checkpoint_structure: dict = {
                    "save": {a: None for a in agents},
                    "load": {a: None for a in agents},
                }
        features: dict = {_:None for _ in self.feature_manager.list_features()}
        try:
            instance_key = f"agent-instance-{self.active_agent_name}"
            if instance_key in self.isaa_tools.config:
                agent = self.isaa_tools.config[instance_key]
                session = agent.session_manager.get(self.active_session_id)
                if session and hasattr(session, "vfs"):
                    is_vfs = True

                engine = agent._get_execution_engine()
                if hasattr(engine, 'skills_manager'):
                    current_skills = {s_id: None for s_id in engine.skills_manager.skills.keys()}
            model_options = {m: None for m in MODEL_MAPPING.keys()}

            # Tools-Optionen
            tools_flag = {
                "true": None,
                "false": None,
                "t": None,
                "f": None,
            }

            # Agent-Unterstruktur für save/load
            checkpoint_structure = {}

            for action in ["save", "load"]:
                checkpoint_structure[action] = {
                    # Agent auswählen
                    **{
                        agent: {
                            # optionaler Checkpoint-Name (frei)
                            # danach Pfad
                            None: PathCompleter(only_directories=True, expanduser=True)
                        }
                        for agent in agents
                    }
                }
        except Exception as e:
            print(e)
            pass

        path_compl = PathCompleter(expanduser=True)



        return {
            "/help": None,
            "/quit": None,
            "/exit": None,
            "/clear": None,
            "/status": None,
            "/vfs": {"init":None},
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
            "/coder": {
                "start": PathCompleter(only_directories=True, expanduser=True),
                "stop": None,
                "task": None,
                "test": None,  # Freitext Befehl
                "accept": None,
                "reject": None,
                "info": None,
                "stream": {"on":None,"off":None},
                "diff": None,
                "files": None,
            },
            "/zenplus": None,
            "/agent": {
                "switch": {a: None for a in agents},
                "list": None,
                "spawn": None,
                "stop": {a: None for a in agents},
                "model": {
                    "fast": model_options,
                    "complex": model_options
                },
                "checkpoint": checkpoint_structure,
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
                "list": None,
                "view": {t: None for t in self.background_tasks.keys()},
                "cancel": {t: None for t in self.background_tasks.keys()},
                "clean": None,
                "status": {t: None for t in self.background_tasks.keys()},
            },
            "/job": {
                "list": None,
                "add": None,
                "remove": {j.job_id: None for j in (self.job_scheduler.list_jobs() if self.job_scheduler else [])},
                "pause": {j.job_id: None for j in (self.job_scheduler.list_jobs() if self.job_scheduler else []) if j.status == "active"},
                "resume": {j.job_id: None for j in (self.job_scheduler.list_jobs() if self.job_scheduler else []) if j.status == "paused"},
                "fire": {j.job_id: None for j in (self.job_scheduler.list_jobs() if self.job_scheduler else [])},
                "detail": {j.job_id: None for j in (self.job_scheduler.list_jobs() if self.job_scheduler else [])},
                "autowake": {"install": None, "remove": None, "status": None},
                "dream": {"create": None,"status": None, "live": None},
            },

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
                "import": PathCompleter(only_directories=True, expanduser=True),
                "export": {
                    "id":
                    {s_id: path_compl for s_id in ["all"] + list(current_skills.keys())} if current_skills else None},
                    "path": PathCompleter(only_directories=True, expanduser=True)
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

        }, VFSCompleter(session.vfs) if is_vfs else None

    def get_prompt_text(self) -> HTML:
        """Generate prompt text with status indicators."""
        cwd_name = Path.cwd().name
        bg_count = sum(1 for t in self.background_tasks.values() if t.status == "running")
        bg_indicator = (
            f" <style fg='ansiyellow'>[{bg_count}bg]</style>" if bg_count > 0 else ""
        )

        audio_indicator = (
            " <style fg='ansired'>REC</style>" if self._audio_recording else ""
        )

        # Coder Mode Indicator
        if self.active_coder:
            mode_indicator = f"<style fg='ansimagenta'>[CODER:{Path(self.active_coder_path).name}]</style>"
            agent_indicator = ""
        else:
            mode_indicator = ""
            agent_indicator = f"<style fg='ansiyellow'>({self.active_agent_name})</style>"

        # Active features - compact tag line
        active_feats = [f for f in self.feature_manager.list_features() if self.feature_manager.is_enabled(f)]
        feat_indicator = ""
        if active_feats:
            tags = " ".join(f"<style fg='{PTColors.ZEN_DIM}'>{f}</style>" for f in active_feats)
            feat_indicator = f" {tags}"

        return HTML(
            f"<style fg='ansicyan'>[</style>"
            f"<style fg='ansigreen'>{cwd_name}</style>"
            f"<style fg='ansicyan'>]</style> "
            f"{agent_indicator}"
            f"{mode_indicator}"
            f"<style fg='grey'>@{self.active_session_id}</style>"
            f"{bg_indicator}{audio_indicator}{feat_indicator}"
            f"\n<style fg='ansiblue'>></style> "
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

        # Background Tasks - read from engine.live for real-time state
        running_tasks = [t for t in self.background_tasks.values() if t.status == "running"]
        print_status(f"Background Tasks: {len(running_tasks)} running", "progress")
        if running_tasks:
            print_table_header(
                [("ID/Agent", 18), ("Progress", 25), ("Phase", 10), ("Thought/Tool", 20)],
                [18, 25, 10, 20]
            )

            for t in running_tasks:
                phase_str = "-"
                bar_str = ""
                focus_str = "-"
                try:
                    agent = await self.isaa_tools.get_agent(t.agent_name)
                    engine = agent._get_execution_engine()
                    live = engine.live

                    # Progress bar from live state
                    it, mx = live.iteration, live.max_iterations
                    if mx > 0:
                        filled = int(20 * it / mx)
                        bar_str = f"{'━' * filled}{'─' * (20 - filled)} {it}/{mx}"
                    else:
                        bar_str = f"{'─' * 20} {it}/?"

                    phase_str = live.phase.value[:10]

                    # Show thought or tool (whichever is most recent)
                    if live.tool.name:
                        focus_str = f"◇ {live.tool.name[:18]}"
                    elif live.thought:
                        focus_str = f"◎ {live.thought[:18]}"
                    elif live.status_msg:
                        focus_str = live.status_msg[:20]

                except Exception:
                    elapsed = (datetime.now() - t.started_at).total_seconds()
                    bar_str = f"{'─' * 20} {elapsed:.0f}s"

                print_table_row(
                    [
                        f"{t.task_id[:8]}.. ({t.agent_name})",
                        bar_str,
                        phase_str,
                        focus_str,
                    ],
                    [18, 25, 10, 20],
                    ["cyan", "green", "white", "grey"]
                )

        # Jobs summary
        if self.job_scheduler and self.job_scheduler.total_count > 0:
            print_status(
                f"Scheduled Jobs: {self.job_scheduler.active_count} active / {self.job_scheduler.total_count} total",
                "data"
            )

        c_print()

    def _print_help(self):
        """Print help information."""
        print_box_header("ISAA Host Commands", "❓")

        print_status("Navigation", "info")
        print_box_content("/help   - Show this help", "")
        print_box_content("/status - Show status dashboard (or F5)", "")
        print_box_content("/clear   - Clear screen", "")
        print_box_content("/zenplus - Togged full Screen agent live vier", "")
        print_box_content("/quit, /exit - Exit CLI", "")
        print_box_content("F6 - Toggle minimize/expand agent output", "")
        print_box_content("Ctrl+C - Safe stop agent (continue/fresh/quit)", "")

        print_separator()

        print_status("Agent Management", "info")
        print_box_content("/agent list - List all agents", "")
        print_box_content("/agent switch <name> - Switch active agent", "")
        print_box_content("/agent spawn <name> <persona> - Create new agent", "")
        print_box_content("/agent stop <name> - Stop agent tasks", "")
        print_box_content("/agent model <fast|complex> <name> - Change LLM model on the fly", "")
        print_box_content("/agent checkpoint <save|load> [name]   - Manage state persistence", "")
        print_box_content("/agent checkpoint help        - list information's about addition args like path", "")
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
        print_box_content("/task                    - Show all background tasks", "")
        print_box_content("/task view [id]          - Live view of task (auto-selects if 1)", "")
        print_box_content("/task cancel <id>        - Cancel a running task", "")
        print_box_content("/task clean              - Remove finished tasks", "")
        print_box_content("F6 during execution      - Move agent to background", "")

        print_separator()

        print_status("Job Scheduler", "info")
        print_box_content("/job list                - List all scheduled jobs", "")
        print_box_content("/job add                 - Add a new job (interactive)", "")
        print_box_content("/job remove <id>         - Remove a job", "")
        print_box_content("/job pause <id>          - Pause a job", "")
        print_box_content("/job resume <id>         - Resume a paused job", "")
        print_box_content("/job fire <id>           - Manually fire a job now", "")
        print_box_content("/job detail <id>         - Show job details", "")
        print_box_content("/job autowake <cmd>      - Manage OS auto-wake (install/remove/status)", "")
        print_status("Dreamer Job", "info")
        print_box_content("/job dream create [agent]    - Create nightly dream job (default: self, 03:00)", "")
        print_box_content("/job dream status            - Show all configured dream jobs", "")
        print_box_content("/job dream live              - Run dream process now with visualization", "")

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
        print_box_content("/vfs <path>                  - Show file content or dir listing", "")
        print_box_content("/vfs mount <path> [vfs_path] - Mount local folder", "")
        print_box_content("/vfs unmount <vfs_path>      - Unmount folder", "")
        print_box_content("/vfs sync [path]             - Sync file/dir to disk", "")
        print_box_content("/vfs save <vfs_path> <local> - Save file/dir to local path", "")
        print_box_content("/vfs refresh <mount>         - Re-scan mount for changes", "")
        print_box_content("/vfs pull <path>             - Reload file/dir from disk", "")
        print_box_content("/vfs mounts                  - List active mounts", "")
        print_box_content("/vfs dirty                   - Show modified files", "")
        print_box_content("/vfs rm/remove               - Remove Folder or File", "")
        print_separator()
        print_status("System Files (Read-Only)", "info")
        print_box_content("/vfs sys-add <local> [path]  - Add file as read-only system file", "")
        print_box_content("/vfs sys-remove <vfs_path>    - Remove a system file", "")
        print_box_content("/vfs sys-refresh <vfs_path>  - Reload system file from disk", "")
        print_box_content("/vfs sys-list                 - List all system files", "")
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

        elif cmd == "/zenplus":
            await self._cmd_zenplus(args)

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

        elif cmd == "/coder":
            await self._cmd_coder(args)

        elif cmd == "/audio":
            await self._handle_audio_command(args)

        elif cmd == "/job":
            await self._cmd_job(args)

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

    async def _cmd_zenplus(self, args: list):
        """Toggle Zen+ mode. Usage: /zenplus"""
        self.zen_plus_mode = not self.zen_plus_mode
        mode = "ZEN+" if self.zen_plus_mode else "ZEN"
        from prompt_toolkit import print_formatted_text, HTML
        print_formatted_text(HTML(
            f"<style fg='#67e8f9'>  ◎ Renderer: {mode}</style>"
        ))

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
                w = ( "info" if (args[1] if len(args) > 1 else "warning") == "help" else "warning")
                print_status("Usage: /agent checkpoint <save|load> [name] <path> <tools[t/f]>", w)
                return
            sub = args[1].lower()
            target = args[2] if len(args) > 2 else self.active_agent_name
            path = args[3] if len(args) > 3 else None
            with_tools = args[4] if len(args) > 4 else None
            if path is None:
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

                return
            data = None
            if sub == "save":
                sucsess, data = await self.isaa_tools.save_agent(
                    agent_name=target,
                    path=path,
                    include_checkpoint=True,
                    include_tools=with_tools,
                    notes="cli-export"
                )
                if not sucsess:
                    print_status(f"Agent saved: {target} Failed {data}", "error")
                    return

            elif path:
                warnings: list[str]
                _, data, warnings = await self.isaa_tools.load_agent(
                    path=path, override_name=target, load_tools=with_tools, register=True
                )
                if _ is None:
                    print_status(f"Agent loading: {target}", "error")

                    if warnings:
                        print_box_header("Load Warnings", icon="⚠")
                        for w in warnings:
                            print_box_content(w, style="warning")
                        print_box_footer()

                    return

            agent_version = data.agent_version if hasattr(data, "agent_version") else None
            has_checkpoint = data.has_checkpoint if hasattr(data, "has_checkpoint") else None
            has_tools = data.has_tools if hasattr(data, "has_tools") else None
            tool_count = data.tool_count if hasattr(data, "tool_count") else None
            serializable_tools = data.serializable_tools if hasattr(data, "serializable_tools") else None
            non_serializable_tools = data.non_serializable_tools if hasattr(data, "non_serializable_tools") else None
            bindings = data.bindings if hasattr(data, "bindings") else None
            # ─────────────────────────────────────────────
            # Header
            # ─────────────────────────────────────────────
            print_box_header("Agent Overview", icon="🤖")

            # ─────────────────────────────────────────────
            # Basisinformationen
            # ─────────────────────────────────────────────
            print_box_content(f"Version: {agent_version or 'N/A'}", style="info")

            if has_checkpoint is True:
                print_box_content("Checkpoint verfügbar", style="success")
            elif has_checkpoint is False:
                print_box_content("Kein Checkpoint vorhanden", style="warning")

            if has_tools is True:
                print_box_content(f"Tools aktiviert ({tool_count or 0})", style="success")
            elif has_tools is False:
                print_box_content("Keine Tools registriert", style="warning")

            print_separator()

            # ─────────────────────────────────────────────
            # Tool-Details (Tabelle)
            # ─────────────────────────────────────────────
            print_table_header(
                columns=[
                    ("Kategorie", None),
                    ("Anzahl", None)
                ],
                widths=[30, 10]
            )

            print_table_row(
                ["Serializable Tools", serializable_tools or 0],
                widths=[30, 10],
                styles=["cyan", "green"]
            )

            print_table_row(
                ["Non-Serializable Tools", non_serializable_tools or 0],
                widths=[30, 10],
                styles=["cyan", "yellow"]
            )

            print_separator()

            # ─────────────────────────────────────────────
            # Bindings
            # ─────────────────────────────────────────────
            if bindings:
                print_box_content("Bindings registriert:", style="info")
                print_code_block(
                    code=str(bindings),
                    language="json",
                    show_line_numbers=False
                )
            else:
                print_box_content("Keine Bindings vorhanden", style="warning")

            # ─────────────────────────────────────────────
            # Footer
            # ─────────────────────────────────────────────
            print_box_footer()
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
            # Default: show task overview
            await self._task_show_overview()
            return

        action = args[0]

        if action == "cancel":
            if len(args) < 2:
                print_status("Usage: /task cancel <id>", "warning")
                return
            task_id = args[1]
            # Allow partial match
            matched = [t for t in self.background_tasks if t.startswith(task_id) or t == task_id]
            if not matched:
                print_status(f"Task '{task_id}' not found", "error")
                return
            for tid in matched:
                self.background_tasks[tid].task.cancel()
                self.background_tasks[tid].status = "cancelled"
                print_status(f"Task {tid} cancelled", "success")

        elif action in ("status", "list"):
            await self._task_show_overview()

        elif action == "view":
            # /task view [id] - show live state or result of a task
            if len(args) < 2:
                # Auto-select: if only 1 running, show that one
                running = [t for t in self.background_tasks.values() if t.status == "running"]
                if len(running) == 1:
                    await self._task_view_detail(running[0].task_id)
                elif not running:
                    # Show most recent completed
                    completed = [t for t in self.background_tasks.values() if t.status == "completed"]
                    if completed:
                        await self._task_view_detail(completed[-1].task_id)
                    else:
                        print_status("No tasks to view", "info")
                else:
                    print_status(f"{len(running)} tasks running. Specify: /task view <id>", "warning")
                    await self._task_show_overview()
                return
            task_id = args[1]
            # Partial match
            matched = [t for t in self.background_tasks if t.startswith(task_id) or t == task_id]
            if matched:
                await self._task_view_detail(matched[0])
            else:
                print_status(f"Task '{task_id}' not found", "error")

        elif action == "clean":
            # Remove completed/failed/cancelled tasks
            to_remove = [tid for tid, t in self.background_tasks.items()
                         if t.status in ("completed", "failed", "cancelled")]
            for tid in to_remove:
                del self.background_tasks[tid]
            print_status(f"Cleaned {len(to_remove)} finished tasks", "success")

        else:
            print_status(f"Unknown task action: {action}. Use: list, view, cancel, clean", "error")

    async def _task_show_overview(self):
        """Show compact task overview table."""
        if not self.background_tasks:
            print_status("No background tasks", "info")
            return

        print_box_header("Background Tasks", "◈")
        columns = [("ID", 22), ("Agent", 12), ("Status", 10), ("Elapsed", 8), ("Query", 22)]
        widths = [22, 12, 10, 8, 22]
        print_table_header(columns, widths)

        for tid, t in self.background_tasks.items():
            elapsed = (datetime.now() - t.started_at).total_seconds()
            elapsed_str = f"{elapsed:.0f}s"
            status_style = {
                "running": "green", "completed": "cyan",
                "failed": "red", "cancelled": "yellow"
            }.get(t.status, "grey")

            query_short = t.query[:20] + ".." if len(t.query) > 22 else t.query
            print_table_row(
                [tid[:22], t.agent_name[:12], t.status, elapsed_str, query_short],
                widths,
                ["cyan", "white", status_style, "grey", "grey"],
            )
        print_box_footer()

    async def _task_view_detail(self, task_id: str):
        """Show detailed view of a specific task (live state or result)."""
        t = self.background_tasks.get(task_id)
        if not t:
            print_status(f"Task {task_id} not found", "error")
            return

        elapsed = (datetime.now() - t.started_at).total_seconds()
        print_box_header(f"Task: {task_id}", "◈")
        print_box_content(f"Agent: {t.agent_name}  Status: {t.status}  Elapsed: {elapsed:.1f}s", "info")
        print_box_content(f"Query: {t.query}", "")

        if t.status == "running":
            # Show live engine state
            try:
                agent = await self.isaa_tools.get_agent(t.agent_name)
                engine = agent._get_execution_engine()
                live = engine.live

                it, mx = live.iteration, live.max_iterations
                if mx > 0:
                    filled = int(20 * it / mx)
                    bar = f"{'━' * filled}{'─' * (20 - filled)} {it}/{mx}"
                else:
                    bar = f"{'─' * 20} {it}/?"
                print_box_content(f"Progress: {bar}", "")

                if live.phase:
                    print_box_content(f"Phase: {live.phase.value}", "info")
                if live.thought:
                    print_box_content(f"Thought: {live.thought[:80]}", "")
                if live.tool.name:
                    print_box_content(f"Tool: {live.tool.name} {live.tool.args_summary[:40]}", "")
                if live.status_msg:
                    print_box_content(f"Status: {live.status_msg}", "")
            except Exception:
                print_box_content("(live state unavailable)", "warning")

        elif t.status == "completed":
            # Show result
            try:
                result = t.task.result()
                if result:
                    result_str = str(result)
                    if len(result_str) > 500:
                        print_box_content(f"Result ({len(result_str)} chars):", "success")
                        print_code_block(result_str[:500] + "\n... (truncated)")
                    else:
                        print_box_content("Result:", "success")
                        c_print(result_str)
            except Exception as e:
                print_box_content(f"Result error: {e}", "error")

        elif t.status == "failed":
            try:
                t.task.result()
            except Exception as e:
                print_box_content(f"Error: {e}", "error")

        print_box_footer()

    # =========================================================================
    # JOB SCHEDULER INTEGRATION
    # =========================================================================

    async def _fire_job_from_scheduler(self, job: JobDefinition) -> str:
        """Callback for JobScheduler: run an agent query as a BackgroundTask."""
        self._task_counter += 1
        task_id = f"job_{self._task_counter}_{job.agent_name}"
        run_id = uuid.uuid4().hex[:8]
        host_ref = self

        async def _run_job():
            try:
                agent = await host_ref.isaa_tools.get_agent(job.agent_name)
                result = await agent.a_run(
                    job.query,
                    session_id=job.session_id,
                    execution_id=run_id,
                )
                if task_id in host_ref.background_tasks:
                    host_ref.background_tasks[task_id].status = "completed"
                return result or ""
            except Exception as e:
                if task_id in host_ref.background_tasks:
                    host_ref.background_tasks[task_id].status = "failed"
                raise

        async_task = asyncio.create_task(_run_job())

        def _on_done(fut):
            try:
                r = fut.result()
                preview = (r[:60] + "..") if len(r) > 62 else r
                zp = ZenPlus.get()
                if zp.active:
                    zp.update_job(task_id, preview)
                c_print(HTML(f"\n<style fg='{PTColors.ZEN_GREEN}'>✓ {task_id}</style>"
                             f"  <style fg='{PTColors.ZEN_DIM}'>{html.escape(preview)}</style>\n"))
            except (asyncio.CancelledError, Exception):
                zp = ZenPlus.get()
                if zp.active:
                    zp.update_job(task_id, "failed")
                pass

        async_task.add_done_callback(_on_done)
        self.background_tasks[task_id] = BackgroundTask(
            task_id=task_id, agent_name=job.agent_name,
            run_id=run_id, query=job.query, task=async_task,
        )

        zp = ZenPlus.get()
        if zp.active:
            zp.inject_job(task_id, job.agent_name, job.query[:60], "running", kind="job")

        # Wait for it (scheduler handles timeout externally)
        return await async_task

    async def _cmd_job(self, args: list[str]):
        """Handle /job commands."""
        if not self.job_scheduler:
            print_status("Job scheduler not initialized", "error")
            return

        if not args:
            args = ["list"]

        action = args[0]

        if action == "list":
            jobs = self.job_scheduler.list_jobs()
            if not jobs:
                print_status("No scheduled jobs", "info")
                return
            print_box_header(f"Scheduled Jobs ({len(jobs)})", "◎")
            columns = [("ID", 14), ("Name", 18), ("Trigger", 16), ("Status", 8), ("Runs", 5), ("Last", 12)]
            widths = [14, 18, 16, 8, 5, 12]
            print_table_header(columns, widths)
            for j in jobs:
                status_style = {
                    "active": "green", "paused": "yellow",
                    "expired": "grey", "disabled": "red",
                }.get(j.status, "white")
                last = j.last_run_at[:10] if j.last_run_at else "-"
                print_table_row(
                    [j.job_id[:14], j.name[:18], j.trigger.trigger_type[:16],
                     j.status, str(j.run_count), last],
                    widths,
                    ["cyan", "white", "magenta", status_style, "grey", "grey"],
                )
            print_box_footer()

        elif action == "add":
            await self._job_add_interactive()

        elif action == "remove":
            if len(args) < 2:
                print_status("Usage: /job remove <id>", "warning")
                return
            if self.job_scheduler.remove_job(args[1]):
                print_status(f"Job {args[1]} removed", "success")
            else:
                print_status(f"Job {args[1]} not found", "error")

        elif action == "pause":
            if len(args) < 2:
                print_status("Usage: /job pause <id>", "warning")
                return
            if self.job_scheduler.pause_job(args[1]):
                print_status(f"Job {args[1]} paused", "success")
            else:
                print_status(f"Job {args[1]} not found or not active", "error")

        elif action == "resume":
            if len(args) < 2:
                print_status("Usage: /job resume <id>", "warning")
                return
            if self.job_scheduler.resume_job(args[1]):
                print_status(f"Job {args[1]} resumed", "success")
            else:
                print_status(f"Job {args[1]} not found or not paused", "error")

        elif action == "fire":
            if len(args) < 2:
                print_status("Usage: /job fire <id>", "warning")
                return
            job = self.job_scheduler.get_job(args[1])
            if not job:
                print_status(f"Job {args[1]} not found", "error")
                return
            print_status(f"Firing job {job.job_id} ({job.name})...", "info")
            asyncio.create_task(self.job_scheduler._fire_job(job))

        elif action == "detail":
            if len(args) < 2:
                print_status("Usage: /job detail <id>", "warning")
                return
            job = self.job_scheduler.get_job(args[1])
            if not job:
                # Try partial match
                matches = self.job_scheduler.find_jobs_by_name(args[1])
                if matches:
                    job = matches[0]
                else:
                    print_status(f"Job {args[1]} not found", "error")
                    return
            print_box_header(f"Job: {job.name}", "◎")
            print_box_content(f"ID: {job.job_id}", "")
            print_box_content(f"Agent: {job.agent_name}", "")
            print_box_content(f"Query: {job.query[:80]}", "")
            print_box_content(f"Trigger: {job.trigger.trigger_type}", "info")
            if job.trigger.at_datetime:
                print_box_content(f"  At: {job.trigger.at_datetime}", "")
            if job.trigger.interval_seconds:
                print_box_content(f"  Interval: {job.trigger.interval_seconds}s", "")
            if job.trigger.cron_expression:
                print_box_content(f"  Cron: {job.trigger.cron_expression}", "")
            if job.trigger.watch_job_id:
                print_box_content(f"  Watch Job: {job.trigger.watch_job_id}", "")
            if job.trigger.watch_path:
                print_box_content(f"  Watch Path: {job.trigger.watch_path}", "")
            print_box_content(f"Status: {job.status}", "")
            print_box_content(f"Session: {job.session_id}", "")
            print_box_content(f"Timeout: {job.timeout_seconds}s", "")
            print_box_content(f"Runs: {job.run_count}  Fails: {job.fail_count}", "")
            if job.last_run_at:
                print_box_content(f"Last Run: {job.last_run_at}", "")
            if job.last_result:
                print_box_content(f"Last Result: {job.last_result}", "")
            print_box_content(f"Created: {job.created_at}", "")
            print_box_footer()

        elif action == "autowake":
            await self._job_autowake(args[1:])

        elif action == "dream":
            sub = args[1] if len(args) > 1 else "status"
            if sub == "create":
                agent_name = args[2] if len(args) > 2 else "self"
                if not hasattr(self, '_current_agent') or not self._current_agent:
                    print_status("No active agent", "error")
                    return
                from toolboxv2.mods.isaa.base.Agent.dreamer import a_dream
                agent = self._current_agent
                if not hasattr(agent, 'a_dream'):
                    agent.a_dream = a_dream.__get__(agent, type(agent))
                self.job_scheduler.add_dream_job(agent_name)
                print_status(f"Dream job created for {agent_name} (nightly 03:00)", "success")
            elif sub == "status":
                dream_jobs = [j for j in self.job_scheduler.list_jobs() if j.query == "__dream__"]
                if not dream_jobs:
                    print_status("No dream jobs configured", "info")
                else:
                    for j in dream_jobs:
                        print_status(
                            f"{j.job_id} | {j.name} | {j.trigger.trigger_type} | "
                            f"{j.status} | runs:{j.run_count}",
                            "info"
                        )
            elif sub == "live":
                agent: FlowAgent= await self.isaa_tools.get_agent(self.active_agent_name)
                agent.active_session = self.active_session_id
                await dream_with_viz_v2(self.isaa_tools, self.active_agent_name)

        else:
            print_status(f"Unknown job action: {action}. Use: list, add, remove, pause, resume, fire, detail, autowake", "error")

    async def _job_add_interactive(self):
        """Interactive job creation."""
        if not self.prompt_session:
            print_status("Prompt session not available", "error")
            return

        agents = self.isaa_tools.config.get("agents-name-list", ["self"])
        available_triggers = self.job_scheduler.trigger_registry.available_types() if self.job_scheduler else []

        print_box_header("Add New Job", "◎")
        print_box_content(f"Agents: {', '.join(agents)}", "info")
        print_box_content(f"Triggers: {', '.join(available_triggers)}", "info")
        print_box_footer()

        try:
            name = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Name: </style>"))
            if not name.strip():
                print_status("Cancelled", "warning")
                return

            agent_name = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Agent: </style>"))
            if not agent_name.strip():
                agent_name = "self"

            query = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Query: </style>"))
            if not query.strip():
                print_status("Query is required", "error")
                return

            trigger_type = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Trigger type: </style>"))
            if not trigger_type.strip():
                print_status("Trigger type is required", "error")
                return

            trigger_cfg = TriggerConfig(trigger_type=trigger_type.strip())

            if trigger_type.strip() == "on_time":
                dt = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Datetime (ISO): </style>"))
                trigger_cfg.at_datetime = dt.strip()
            elif trigger_type.strip() == "on_interval":
                secs = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Interval (seconds): </style>"))
                trigger_cfg.interval_seconds = int(secs.strip())
            elif trigger_type.strip() == "on_cron":
                expr = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Cron expression: </style>"))
                trigger_cfg.cron_expression = expr.strip()
            elif trigger_type.strip() in ("on_job_completed", "on_job_failed", "on_job_timeout"):
                jid = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Watch job ID: </style>"))
                trigger_cfg.watch_job_id = jid.strip()
            elif trigger_type.strip() == "on_file_changed":
                path = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Watch path: </style>"))
                trigger_cfg.watch_path = path.strip()
                pats = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Patterns (comma-sep, empty=all): </style>"))
                if pats.strip():
                    trigger_cfg.watch_patterns = [p.strip() for p in pats.split(",")]
            elif trigger_type.strip() == "on_system_idle":
                idle = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Idle seconds threshold [300]: </style>"))
                trigger_cfg.idle_seconds = int(idle.strip()) if idle.strip() else 300

            timeout = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Timeout seconds [300]: </style>"))
            timeout_s = int(timeout.strip()) if timeout.strip() else 300

            job = JobDefinition(
                job_id=JobDefinition.generate_id(),
                name=name.strip(),
                agent_name=agent_name.strip(),
                query=query.strip(),
                trigger=trigger_cfg,
                timeout_seconds=timeout_s,
            )
            job_id = self.job_scheduler.add_job(job)
            print_status(f"Job created: {job_id} ({name.strip()})", "success")

        except (EOFError, KeyboardInterrupt):
            print_status("Cancelled", "warning")

    async def _job_autowake(self, args: list[str]):
        """Handle /job autowake install/remove/status."""
        if not args:
            print_status("Usage: /job autowake <install|remove|status>", "warning")
            return

        try:
            from toolboxv2.mods.isaa.extras.jobs.os_scheduler import install_autowake, remove_autowake, autowake_status
        except ImportError:
            print_status("OS scheduler module not available", "error")
            return

        action = args[0]
        if action == "install":
            result = install_autowake(self.jobs_file)
            print_status(result, "success" if "installed" in result.lower() else "error")
        elif action == "remove":
            result = remove_autowake()
            print_status(result, "success" if "removed" in result.lower() else "error")
        elif action == "status":
            result = autowake_status()
            print_status(result, "info")
        else:
            print_status("Usage: /job autowake <install|remove|status>", "warning")

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
            if cmd == "init":
                print_status("VFS Online")
                return
            elif cmd == "mount":
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

            # /vfs sync [vfs_path]  - file or directory
            elif cmd == "sync":
                if len(args) > 1:
                    path = session.vfs._normalize_path(args[1])

                    if session.vfs._is_directory(path):
                        # Sync all dirty files under this directory
                        prefix = path if path == "/" else path + "/"
                        synced, errors = [], []
                        for fpath, f in session.vfs.files.items():
                            if fpath.startswith(prefix) and isinstance(f, VFSFile) and f.is_dirty and f.local_path:
                                r = session.vfs._sync_to_local(fpath)
                                if r.get("success"):
                                    synced.append(fpath)
                                else:
                                    errors.append(f"{fpath}: {r.get('error')}")
                        print_status(f"Synced {len(synced)} files in {path}", "success")
                        for err in errors:
                            print_status(err, "error")
                    elif session.vfs._is_file(path):
                        result = session.vfs._sync_to_local(path)
                        if result.get("success"):
                            print_status(f"Synced: {path} → {result['synced_to']}", "success")
                        else:
                            print_status(f"Sync failed: {result.get('error')}", "error")
                    else:
                        print_status(f"Not found: {path}", "error")
                else:
                    # Sync all dirty files
                    result = session.vfs.sync_all()
                    if result.get("success"):
                        print_status(f"Synced {len(result['synced'])} files", "success")
                    else:
                        for err in result.get("errors", []):
                            print_status(err, "error")

            # /vfs save <vfs_path> <local_path>  - file or directory
            elif cmd == "save":
                if len(args) < 3:
                    print_status("Usage: /vfs save <vfs_path> <local_path>", "warning")
                    return
                vfs_path = session.vfs._normalize_path(args[1])
                local_path = args[2]

                if session.vfs._is_directory(vfs_path):
                    # Save entire directory to local path
                    prefix = vfs_path if vfs_path == "/" else vfs_path + "/"
                    saved, errors = 0, 0
                    local_base = os.path.abspath(os.path.expanduser(local_path))
                    os.makedirs(local_base, exist_ok=True)

                    for fpath, f in session.vfs.files.items():
                        if fpath.startswith(prefix):
                            relative = fpath[len(prefix):]
                            target = os.path.join(local_base, relative.replace("/", os.sep))
                            result = session.vfs.save_to_local(
                                fpath, target, overwrite=True, create_dirs=True
                            )
                            if result.get("success"):
                                saved += 1
                            else:
                                errors += 1
                    print_status(f"Saved {saved} files from {vfs_path} → {local_path}", "success")
                    if errors:
                        print_status(f"{errors} files failed", "warning")
                else:
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

            # /vfs pull <vfs_path> - reload from disk (file or directory)
            elif cmd == "pull":
                if len(args) < 2:
                    print_status("Usage: /vfs pull <path>", "warning")
                    return

                path = session.vfs._normalize_path(args[1])

                if session.vfs._is_directory(path):
                    # Pull all shadow files under this directory
                    prefix = path if path == "/" else path + "/"
                    pulled, skipped = 0, 0
                    for fpath, f in session.vfs.files.items():
                        if fpath.startswith(prefix) and hasattr(f, 'local_path') and f.local_path:
                            result = session.vfs._load_shadow_content(fpath)
                            if result.get("success"):
                                f.is_dirty = False
                                f.backing_type = FileBackingType.SHADOW
                                pulled += 1
                            else:
                                skipped += 1
                    print_status(f"Pulled {pulled} files in {path}", "success")
                    if skipped:
                        print_status(f"{skipped} files skipped/failed", "warning")
                elif session.vfs._is_file(path):
                    f = session.vfs.files.get(path)
                    if f and hasattr(f, 'local_path') and f.local_path:
                        result = session.vfs._load_shadow_content(path)
                        if result.get("success"):
                            f.is_dirty = False
                            f.backing_type = FileBackingType.SHADOW
                            print_status(f"Pulled: {path} ({result['loaded_bytes']} bytes)", "success")
                        else:
                            print_status(f"Pull failed: {result.get('error')}", "error")
                    else:
                        print_status("Not a shadow file", "warning")
                else:
                    print_status(f"Not found: {path}", "error")

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

            # /vfs sys-add <local_path> [vfs_path] [--refresh]
            elif cmd == "sys-add":
                if len(args) < 2:
                    print_status("Usage: /vfs sys-add <local_path> [vfs_path] [--refresh]", "warning")
                    return

                local_path = args[1]
                vfs_path = args[2] if len(args) > 2 and not args[2].startswith("--") else None
                auto_refresh = "--refresh" in args

                print_status(f"Adding system file: {local_path}...", "info")
                result = session.vfs.add_system_file(
                    local_path=local_path,
                    vfs_path=vfs_path,
                    auto_refresh=auto_refresh
                )

                if result.get("success"):
                    print_status(f"✓ {result['message']}", "success")
                    print_box_content(f"Size: {result['size_bytes']} bytes, Lines: {result['lines']}", "info")
                    if result.get('auto_refresh'):
                        print_box_content("Auto-refresh: enabled", "info")
                else:
                    print_status(f"✗ {result.get('error')}", "error")

            # /vfs sys-remove <vfs_path>
            elif cmd == "sys-remove":
                if len(args) < 2:
                    print_status("Usage: /vfs sys-remove <vfs_path>", "warning")
                    return

                vfs_path = args[1]
                result = session.vfs.remove_system_file(vfs_path)

                if result.get("success"):
                    print_status(f"✓ {result['message']}", "success")
                else:
                    print_status(f"✗ {result.get('error')}", "error")

            elif cmd in ["remove", "rm"]:
                if len(args) < 2:
                    print_status("Usage: /vfs remove <vfs_path>", "warning")
                    return

                # Unterstützt Pfade mit Leerzeichen
                vfs_path = " ".join(args[1:])
                norm_path = session.vfs._normalize_path(vfs_path)

                # 1. Existenz und Typ prüfen
                is_file = session.vfs._is_file(norm_path)
                is_dir = session.vfs._is_directory(norm_path)

                if not is_file and not is_dir:
                    print_status(f"✗ Path not found: {norm_path}", "error")
                    return

                # Systemdateien (Read-Only) vorher abfangen (Verhindert Absturz beim Bestätigen)
                if is_file and session.vfs.files[norm_path].readonly:
                    print_status(f"✗ Cannot delete system file: {norm_path} (Use sys-remove instead)", "error")
                    return
                if is_dir and session.vfs.directories[norm_path].readonly:
                    print_status(f"✗ Cannot delete readonly directory: {norm_path}", "error")
                    return

                # 2. Statistiken sammeln
                total_size = 0
                file_count = 0
                dir_count = 0

                if is_file:
                    f = session.vfs.files[norm_path]
                    file_count = 1
                    total_size = f.size
                    target_desc = f"File: 📄 {f.filename}"
                else:
                    # Ordner rekursiv analysieren
                    ls_result = session.vfs.ls(norm_path, recursive=True)
                    if ls_result.get("success"):
                        for item in ls_result["contents"]:
                            if item["type"] == "directory":
                                dir_count += 1
                            else:
                                file_count += 1
                                total_size += item.get("size", 0)
                    dir_count += 1  # Den Hauptordner selbst mitzählen
                    target_desc = f"Directory: 📁 {norm_path}"

                # Größe formatieren
                size_str = f"{total_size} bytes" if total_size < 1024 else f"{total_size / 1024:.2f} KB"

                # 3. Bestätigungs-Dialog anzeigen
                print_box_header("Confirm Deletion", "⚠️")
                print_box_content(target_desc, "")

                if is_dir:
                    print_box_content(f"  Includes: {file_count} files, {dir_count - 1} subdirectories", "warning")

                print_box_content(f"  Total size: {size_str}", "warning")
                print_box_footer()

                # Nutzer um Erlaubnis fragen
                try:
                    confirm = input("\nAre you sure you want to permanently delete this? (y/N): ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    confirm = "n"

                if confirm not in ['y', 'yes']:
                    print_status("Deletion cancelled.", "info")
                    return

                # 4. Löschen ausführen
                if is_file:
                    result = session.vfs.delete(norm_path)
                else:
                    # force=True zwingt das VFS, auch gefüllte Ordner rekursiv zu löschen
                    result = session.vfs.rmdir(norm_path, force=True)

                # 5. Ergebnis ausgeben
                if result.get("success"):
                    print_status(f"✓ {result['message']}", "success")
                else:
                    print_status(f"✗ Failed to delete: {result.get('error')}", "error")

            # /vfs sys-refresh <vfs_path>
            elif cmd == "sys-refresh":
                if len(args) < 2:
                    print_status("Usage: /vfs sys-refresh <vfs_path>", "warning")
                    return

                vfs_path = args[1]
                result = session.vfs.refresh_system_file(vfs_path)

                if result.get("success"):
                    print_status(f"✓ {result['message']}", "success")
                    print_box_content(f"Size: {result['size_bytes']} bytes, Lines: {result['lines']}", "info")
                else:
                    print_status(f"✗ {result.get('error')}", "error")

            # /vfs sys-list - list all system files
            elif cmd == "sys-list":
                result = session.vfs.list_system_files()

                if not result.get("system_files"):
                    print_status("No system files", "info")
                    return

                print_box_header("System Files (Read-Only)", "📄")
                for info in result["system_files"]:
                    path = info["path"]
                    local = info.get("local_path") or "memory"
                    refresh = " [auto-refresh]" if info.get("auto_refresh") else ""
                    print_box_content(f"{path} ← {local}{refresh}", "")
                    print_box_content(f"  {info['lines']} lines, {info['file_type']}", "dim")
                print_box_footer()

            # /vfs <path> - show file content or directory listing
            else:
                path_str = " ".join(args)
                norm = session.vfs._normalize_path(path_str)

                if session.vfs._is_directory(norm):
                    # Directory: show listing
                    contents = session.vfs._list_directory_contents(norm)
                    print_box_header(f"VFS: {norm}", "📂")
                    if not contents:
                        print_box_content("(empty directory)", "")
                    else:
                        for item in contents:
                            if item["type"] == "directory":
                                print_box_content(f"  📁 {item['name']}/", "")
                            else:
                                size = item.get("size", 0)
                                state = item.get("state", "")
                                ftype = item.get("file_type", "")
                                meta = f"{size}b" if size < 1024 else f"{size / 1024:.1f}kb"
                                dirty = " ●" if session.vfs.files.get(item["path"], None) and getattr(session.vfs.files[item["path"]], 'is_dirty', False) else ""
                                print_box_content(f"  {item['name']:<30} {meta:>8}  {ftype}{dirty}", "")
                    print_box_footer()
                else:
                    await self._vfs_show_file(session, path_str)

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

    async def _cmd_coder(self, args: list[str]):
        """Handle /coder commands for native code generation."""
        if not args:
            print_status("Usage: /coder <start|stop|stream|info|task|diff|accept|reject|test|files> [args]", "warning")
            return

        action = args[0].lower()

        # --- START ---
        if action == "start":
            target_path = os.path.abspath(os.path.expanduser(args[1])) if len(args) >= 2 else self.init_dir
            if not os.path.isdir(target_path):
                print_status(f"Directory not found: {target_path}", "error")
                return
            try:
                agent = await self.isaa_tools.get_agent(self.active_agent_name)
                print_status(f"Initializing Coder on {target_path}...", "progress")
                agent.verbose = True
                self.active_coder = CoderAgent(agent, target_path)
                self.active_coder_path = target_path

                print_box_header("Coder Mode Activated", "👨‍💻")
                print_box_content(f"Target: {target_path}", "info")
                print_box_content(f"Agent: {self.active_agent_name}", "info")
                print_box_content("Commands:", "bold")
                for line in [
                    "  /coder task <instruction>     - Auto-implement",
                    "  <instruction>                 - Auto-implement (shortcut)",
                    "  @<instruction>                - Normal agent / next line",
                    "  /coder diff [file]            - Show changes",
                    "  /coder accept [file ...]      - Apply all or cherry-pick files",
                    "  /coder reject                 - Discard all changes",
                    "  /coder rollback [file ...]    - Reset all or specific files",
                    "  /coder test [cmd]             - Run tests in worktree",
                    "  /coder files                  - List worktree contents",
                    "  /coder info                   - Show paths + status",
                    "  /coder stream [on/off]        - Show paths + status",
                    "  /coder stop                   - Exit (accept changes first!)",
                ]:
                    print_box_content(line, "")
                print_box_footer()
            except Exception as e:
                print_status(f"Failed to start coder: {e}", "error")
        elif action == "stream":
            if len(args) < 2:
                c_print("/coder stream on or off")
                return
            do = args[1]
            if do == "on":
                self.active_coder.stream_enabled = True
            else:
                self.active_coder.stream_enabled = False
            c_print(f"Coder streaming {'enabled' if self.active_coder.stream_enabled else 'disabled'}")

        elif action == "info":

            wt = self.active_coder.worktree
            wt_path = wt.worktree_path
            print_box_header("Coder Info", "ℹ")
            print_box_content(f"Origin (dein Repo):  {wt.origin_root}", "info")
            print_box_content(f"Worktree (Coder):    {wt_path}", "info")
            print_box_content(
                f"Git-Modus:           {'Ja (Branch: ' + wt._branch + ')' if wt._is_git else 'Nein (Kopie)'}", "info")
            print_box_content(f"Agent:               {self.active_agent_name}", "info")
            print_box_content(f"Model:               {self.active_coder.model}", "info")
            print_box_content(f"Stream:              {self.active_coder.stream_enabled}", "info")
            print_separator()

            try:
                changed = await wt.changed_files()
                if changed:
                    print_box_content(f"Geänderte Dateien ({len(changed)}):", "bold")
                    for f in changed:
                        src = wt_path / f
                        size = src.stat().st_size if src.exists() else 0
                        c_print(f"  ● {f}  ({size:,} bytes)")
                else:
                    print_box_content("Keine Änderungen im Worktree.", "info")
            except Exception as e:
                print_box_content(f"Fehler beim Lesen: {e}", "error")

            # Letzte Edits aus Memory
            if self.active_coder.memory.reports:
                print_separator()
                print_box_content("Letzte Tasks:", "bold")
                for r in self.active_coder.memory.reports[-3:]:
                    status = "✓" if r.get("success") else "✗"
                    files = ", ".join(r.get("changed_files", [])[:5])
                    c_print(f"  {status} {r.get('task', '?')[:60]}")
                    if files:
                        c_print(f"    → {files}")

            print_box_footer()
        # --- STOP ---
        elif action == "stop":
            if not self.active_coder:
                print_status("Coder Mode is not active.", "warning")
                return
            # Warn about pending changes
            try:
                pending = await self.active_coder.worktree.changed_files()
                if pending:
                    print_status(f"⚠ {len(pending)} uncommitted file(s) will be lost:", "warning")
                    for f in pending[:10]:
                        c_print(f"  - {f}")
                    if len(pending) > 10:
                        c_print(f"  ... and {len(pending)-10} more")
            except Exception:
                pass
            try:
                self.active_coder.worktree.cleanup()
            except Exception:
                pass
            self.active_coder = None
            self.active_coder_path = self.init_dir
            print_status("Coder Mode deactivated.", "success")

        # --- ACTIONS (require active coder) ---
        else:
            if not self.active_coder:
                print_status("Coder not active. Use '/coder start <path>' first.", "error")
                return

            wt = self.active_coder.worktree

            if action == "task":
                if len(args) < 2:
                    print_status("Usage: /coder task <instruction>", "warning")
                    return
                task_prompt = " ".join(args[1:])
                try:
                    print_status(f"Coder working on: {task_prompt}", "progress")
                except UnicodeEncodeError:
                    task_prompt = task_prompt.encode('utf-8').decode('utf-8', errors="replace")

                c_print(HTML("<style fg='ansimagenta'>⚡ Code Generation Loop...</style>"))
                try:
                    result = await self.active_coder.execute(task_prompt)
                    if result.success:
                        print_box_header("Done", "✅")
                        print_box_content(result.message, "success")
                        if result.changed_files:
                            print_status("Modified:", "info")
                            for f in result.changed_files:
                                c_print(f"  → {f}")
                        print_separator()
                        print_box_content("'/coder diff' to review, '/coder accept' to apply.", "warning")
                        print_box_footer()
                    else:
                        print_status(f"Failed: {result.message}", "error")
                except Exception as e:
                    print_status(f"Critical: {e}", "error")
                    import traceback; traceback.print_exc()

            elif action == "diff":
                try:
                    wt_path = wt.worktree_path
                    if wt._is_git:
                        # Windows: asyncio.create_subprocess_shell wirft NotImplementedError
                        # → subprocess.run stattdessen
                        subprocess.run(["git", "add", "-A"], cwd=str(wt_path),
                                       capture_output=True, encoding="utf-8", errors="replace")
                        target = args[1] if len(args) > 1 else ""
                        cmd = ["git", "diff", "--cached", "--color"]
                        if target:
                            cmd.append(target)
                        result = subprocess.run(cmd, cwd=str(wt_path),
                                                capture_output=True, encoding="utf-8", errors="replace")
                        if result.stdout.strip():
                            print(result.stdout)
                        else:
                            print_status("No changes.", "info")
                    else:
                        changed = await wt.changed_files()
                        if not changed:
                            print_status("No changes.", "info")
                        else:
                            import difflib
                            filter_file = args[1] if len(args) > 1 else None
                            for rel in changed:
                                if filter_file and rel != filter_file: continue
                                orig = wt.origin_root / rel
                                curr = wt.path / rel
                                old = orig.read_text(encoding="utf-8",
                                                     errors="replace").splitlines() if orig.exists() else []
                                new = curr.read_text(encoding="utf-8",
                                                     errors="replace").splitlines() if curr.exists() else []
                                diff = difflib.unified_diff(old, new, fromfile=f"a/{rel}", tofile=f"b/{rel}",
                                                            lineterm="")
                                for line in diff:
                                    c_print(line)
                except Exception as e:
                    print_status(f"Diff error: {e}", "error")
                    import traceback
                    traceback.print_exc()

            elif action == "accept":
                # /coder accept              → apply all (git merge or copy)
                # /coder accept f1.py f2.py  → cherry-pick specific files
                try:
                    if len(args) > 1:
                        # Cherry-pick mode
                        files = args[1:]
                        available = await wt.changed_files()
                        invalid = [f for f in files if f not in available]
                        if invalid:
                            print_status(f"Not changed in worktree: {', '.join(invalid)}", "error")
                            if available:
                                print_status("Available:", "info")
                                for f in available: c_print(f"  {f}")
                            return

                        applied = await wt.apply_files(files)
                        for f in applied:
                            c_print(f"  ✓ {f}")
                        print_status(f"Cherry-picked {len(applied)} file(s).", "success")
                    else:
                        # Full apply
                        n = await wt.apply_back()
                        if n == -1:
                            print_status("Merged via git.", "success")
                        else:
                            print_status(f"Applied {n} file(s).", "success")

                    # Reset worktree for next task
                    wt.cleanup()
                    wt.setup()
                    print_status("Worktree reset for next task.", "info")
                except subprocess.CalledProcessError as e:
                    print_status(f"Merge conflict! Resolve manually in {wt.origin_root}", "error")
                    if e.stderr:
                        for line in e.stderr.splitlines()[:10]:
                            c_print(f"  {line}")
                except Exception as e:
                    print_status(f"Accept failed: {e}", "error")

            elif action == "reject":
                print_status("Discarding all changes...", "warning")
                wt.cleanup()
                wt.setup()
                print_status("Worktree reset.", "success")

            elif action == "rollback":
                # /coder rollback              → reset entire worktree
                # /coder rollback f1.py f2.py  → reset specific files
                try:
                    if len(args) > 1:
                        files = args[1:]
                        await wt.rollback(files)
                        for f in files:
                            c_print(f"  ↩ {f}")
                        print_status(f"Rolled back {len(files)} file(s).", "success")
                    else:
                        changed = await wt.changed_files()
                        if not changed:
                            print_status("Nothing to rollback.", "info")
                            return
                        print_status(f"Rolling back {len(changed)} file(s)...", "warning")
                        await wt.rollback()
                        print_status("Full rollback complete.", "success")
                except Exception as e:
                    print_status(f"Rollback failed: {e}", "error")

            elif action == "test":
                cmd = " ".join(args[1:]) if len(args) > 1 else "pytest"
                print_status(f"Running in worktree: {cmd}", "progress")
                try:
                    print_separator(char=".")
                    proc = await asyncio.create_subprocess_shell(
                        cmd, cwd=str(wt.worktree_path), stdout=None, stderr=None)
                    await proc.wait()
                    print_separator(char=".")
                    if proc.returncode == 0:
                        print_status("Tests passed.", "success")
                    else:
                        print_status(f"Tests failed (exit {proc.returncode})", "error")
                except Exception as e:
                    print_status(f"Test error: {e}", "error")

            elif action == "files":
                wt_path = wt.worktree_path
                changed = set(await wt.changed_files())
                for root, dirs, files in os.walk(wt_path):
                    if ".git" in root: continue
                    level = root.replace(str(wt_path), "").count(os.sep)
                    indent = "  " * level
                    c_print(f"{indent}{os.path.basename(root)}/")
                    for f in files:
                        rel = str((Path(root) / f).relative_to(wt_path))
                        marker = " ●" if rel in changed else ""
                        c_print(f"{indent}  {f}{marker}")

            else:
                print_status(f"Unknown: {action}. Commands: task|diff|accept|reject|rollback|test|files|stop", "error")

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
            overview = await agent.context_overview(self.active_session_id, f_print=c_print)
        except Exception as e:
            print_status(f"Error: {e}", "error")
            import traceback
            traceback.print_exc()

    def _move_stream_to_background(self, stream_gen, agent, query: str):
        """Move a running stream to background, continuing to drain chunks silently."""
        self._task_counter += 1
        agent_name = self.active_agent_name
        task_id = f"bg_{self._task_counter}_{agent_name}"
        run_id = uuid.uuid4().hex[:8]
        host_ref = self

        async def bg_drain():
            result_text = ""
            try:
                while True:
                    try:
                        chunk = await stream_gen.__anext__()
                    except StopAsyncIteration:
                        break
                    if chunk.get("type") == "content":
                        result_text += chunk.get("chunk", "")
                    elif chunk.get("type") == "final_answer":
                        result_text = chunk.get("answer", result_text)

                if task_id in host_ref.background_tasks:
                    host_ref.background_tasks[task_id].status = "completed"
                return result_text
            except asyncio.CancelledError:
                try:
                    await stream_gen.aclose()
                except Exception:
                    pass
                if task_id in host_ref.background_tasks:
                    host_ref.background_tasks[task_id].status = "cancelled"
                raise
            except Exception:
                if task_id in host_ref.background_tasks:
                    host_ref.background_tasks[task_id].status = "failed"
                raise

        async_task = asyncio.create_task(bg_drain())

        # Auto-notify on completion
        def _on_bg_done(fut):
            running = [t for t in host_ref.background_tasks.values() if t.status == "running"]
            zp = ZenPlus.get()
            try:
                result = fut.result()
                if zp.active:
                    zp.update_job(task_id, f"completed of {len(running)}")
                result_preview = (result[:80] + "..") if len(result) > 82 else result
                c_print(HTML(
                    f"\n<style fg='{PTColors.ZEN_GREEN}'>✓ {task_id} complete</style>"
                    f"  <style fg='{PTColors.ZEN_DIM}'>{esc(result_preview)}</style>\n"
                ))
            except (asyncio.CancelledError, Exception):
                if zp.active:
                    zp.update_job(task_id, "failed")

        async_task.add_done_callback(_on_bg_done)

        self.background_tasks[task_id] = BackgroundTask(
            task_id=task_id,
            run_id=run_id,
            agent_name=agent_name,
            query=query[:100],
            task=async_task,
        )
        zp = ZenPlus.get()
        if zp.active:
            zp.inject_job(task_id, agent_name, query[:60], "running", kind="bg")
        c_print(HTML(
            f"<style fg='{PTColors.ZEN_DIM}'>  ▾ moved to background: {task_id}  "
            f"(/task status to view)</style>"
        ))
        return task_id

    async def _handle_agent_interaction(self, user_input: str):
        """Handle regular agent interaction with ZEN Mode streaming."""
        if self.active_coder and not user_input.startswith("@"):
            await self._cmd_coder(["task", user_input])
            return
        elif self.active_coder and user_input.startswith("@"):
            user_input = user_input[1:]
        try:
            agent = await self.isaa_tools.get_agent(self.active_agent_name)
            engine = agent._get_execution_engine()

            # Audio setup
            wants_audio = user_input.strip().endswith("#audio")
            if wants_audio:
                user_input = user_input.rsplit("#audio", 1)[0].strip()
            should_speak = wants_audio or getattr(self, 'verbose_audio', False)
            if should_speak and hasattr(self, 'audio_player'):
                await self.audio_player.start()

            # ZenRendererV2 with live state access
            renderer = ZenRendererV2(engine)
            self._active_renderer = renderer
            c_print()  # Spacing

            # State for final processing
            final_response_text = ""
            current_sentence = ""
            stop_for_speech = False
            moved_to_bg = False

            # Manual stream iteration so we can hand off the generator on minimize
            stream = agent.a_stream(
                query=user_input,
                session_id=self.active_session_id,
            )

            if self.zen_plus_mode:
                from toolboxv2.mods.isaa.extras.zen.zen_plus import ZenPlus  # adjust import path

                zp = ZenPlus.get()
                zp.clear_panes()
                renderer.set_zen_plus(zp)

                # Replay any chunks already buffered (falls mid-toggle)
                for c in renderer._chunk_buffer:
                    zp.feed_chunk(c)

                async def _drain_stream():
                    """Consume stream, push through renderer (→ bridge → ZenPlus)."""
                    try:
                        while True:
                            try:
                                chunk = await stream.__anext__()
                            except StopAsyncIteration:
                                break
                            renderer.process_chunk(chunk)

                            if chunk.get("type") == "content":
                                final_response_text_parts.append(chunk.get("chunk", ""))
                    except Exception:
                        pass
                    finally:
                        zp.signal_stream_done()

                final_response_text_parts = []

                # Run stream consumer + ZenPlus UI concurrently
                stream_task = asyncio.create_task(_drain_stream())

                def _on_zen_plus_exit():
                    """Called when Esc at Grid level → deactivate Zen+."""
                    self.zen_plus_mode = False
                    renderer.set_zen_plus(None)
                await zp.start(on_exit=_on_zen_plus_exit)

                # ZenPlus exited (user pressed Esc)
                if not stream_task.done():
                    # Stream still running → let it finish in background or cancel
                    stream_task.cancel()
                    try:
                        await stream_task
                    except asyncio.CancelledError:
                        pass

                try:
                    await stream.aclose()
                except Exception:
                    pass

            try:
                    while True:
                        try:
                            chunk = await stream.__anext__()
                        except StopAsyncIteration:
                            break

                        # F6 minimize check: hand off stream to background task
                        if renderer.minimized:
                            moved_to_bg = True
                            break

                        renderer.process_chunk(chunk)

                        if chunk.get("type") == "content":
                            text = chunk.get("chunk", "")
                            final_response_text += text

                            if should_speak and hasattr(self, 'audio_player'):
                                if '```' in text: stop_for_speech = not stop_for_speech
                                if not stop_for_speech:
                                    current_sentence += text
                                    if any(current_sentence.rstrip().endswith(p) for p in ['.', '!', '?', ':', '\n\n']):
                                        clean_text = remove_styles(current_sentence.strip())
                                        if clean_text:
                                            get_app("ci.audio.bg.task").run_bg_task_advanced(
                                                self.audio_player.queue_text, clean_text
                                            )
                                        current_sentence = ""

            except KeyboardInterrupt:
                # --- Safe Ctrl+C: stop agent, don't exit program ---
                try:
                    await stream.aclose()
                except Exception:
                    pass

                c_print()
                c_print(HTML(f"\n<style fg='ansiyellow'>⚠ Interrupted</style>"))

                for exec_info in agent.list_executions():
                    if exec_info.get("session_id") == self.active_session_id:
                        await agent.cancel_execution(exec_info.get("run_id"))

                renderer.print_live_summary()
                c_print()
                c_print(HTML(
                    f"<style fg='{PTColors.ZEN_DIM}'>"
                    f"  [c] continue (add context)  "
                    f"  [Enter/q] back to prompt"
                    f"</style>"
                ))

                try:
                    with patch_stdout():
                        choice = await self.prompt_session.prompt_async(
                            HTML(f"<style fg='{PTColors.ZEN_CYAN}'>▸ </style>")
                        )
                    choice = choice.strip().lower()

                    if choice.startswith("c"):
                        extra = choice[1:].strip() if len(choice) > 1 else ""
                        if not extra:
                            c_print(HTML(f"<style fg='{PTColors.ZEN_DIM}'>  (optional context, Enter to skip)</style>"))
                            with patch_stdout():
                                extra = await self.prompt_session.prompt_async(
                                    HTML(f"<style fg='{PTColors.ZEN_CYAN}'>+ </style>")
                                )
                            extra = extra.strip()

                        continue_query = user_input
                        if extra:
                            continue_query = f"{user_input}\n\n[User added after interruption]: {extra}"
                        await self._handle_agent_interaction(continue_query)
                        return

                    elif choice and choice != "q":
                        await self._handle_agent_interaction(choice)
                        return

                except (KeyboardInterrupt, EOFError):
                    pass

                self._active_renderer = None
                return

            # --- Post-stream handling ---
            if moved_to_bg:
                # Hand off the still-open stream generator to a background task
                self._move_stream_to_background(stream, agent, user_input)
                self._active_renderer = None
                return

            # Normal completion - close the generator
            try:
                await stream.aclose()
            except Exception:
                pass

            self._active_renderer = None
            c_print()

            # Post-Processing: Visualize JSON
            try:
                async def _():
                    try:
                        if "```json" in final_response_text or "```" in final_response_text:
                            if "```json" in final_response_text:
                                json_str = final_response_text.split("```json")[1].split("```")[0].strip()
                            else:
                                json_str = final_response_text.split("```")[1]

                            data = json.loads(json_str)
                            if isinstance(data, list) and len(data) == 1:
                                data = data[0]
                            if isinstance(data, dict):
                                print_separator()
                                await visualize_data_terminal(
                                    data, agent, max_preview_chars=max(len(final_response_text), 8000)
                                )
                    except:
                        pass

                get_app().run_bg_task_advanced(_)
            except:
                pass

        except Exception as e:
            print_status(f"System Error: {e}", "error")
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
        c_print(HTML(f"<style fg='{PTColors.ZEN_CYAN}'>{CLI_NAME}</style> <style fg='{PTColors.ZEN_DIM}'>v{VERSION}</style>"))
        c_print(HTML(f"<style fg='{PTColors.ZEN_DIM}'>/help  F4 voice  F5 status  F6 minimize  Ctrl+C safe stop</style>"))
        c_print()

        # Initialize Self Agent
        await self._init_self_agent()

        # Start Job Scheduler
        self.job_scheduler = JobScheduler(self.jobs_file, self._fire_job_from_scheduler)
        await self.job_scheduler.start()
        await self.job_scheduler.fire_lifecycle("on_cli_start")

        # Show active features
        all_feats = self.feature_manager.list_features()
        if all_feats:
            active = [f for f in all_feats if self.feature_manager.is_enabled(f)]
            inactive = [f for f in all_feats if not self.feature_manager.is_enabled(f)]
            parts = []
            for f in active:
                parts.append(f"<style fg='{PTColors.ZEN_GREEN}'>{f}</style>")
            for f in inactive:
                parts.append(f"<style fg='{PTColors.ZEN_DIM}'>{f}</style>")
            c_print(HTML(f"<style fg='{PTColors.ZEN_DIM}'>features:</style> {' '.join(parts)}"))
            c_print()

        # Print status
        await self._print_status_dashboard()

        # Create prompt session
        dict_coplet, vfs_cplet = self._build_completer()
        self.prompt_session = PromptSession(
            history=self.history,
            completer=SmartCompleter(
                nested_dict=dict_coplet, vfs_completer=vfs_cplet
            ),
            complete_while_typing=True,
            key_bindings=self.key_bindings,
        )

        # Main loop
        while True:
            try:
                # Update completer

                dict_coplet, vfs_cplet = self._build_completer()
                self.prompt_session.completer = SmartCompleter(
                    nested_dict=dict_coplet, vfs_completer=vfs_cplet
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

        # Stop job scheduler
        if self.job_scheduler:
            await self.job_scheduler.fire_lifecycle("on_cli_exit")
            await self.job_scheduler.stop()

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
