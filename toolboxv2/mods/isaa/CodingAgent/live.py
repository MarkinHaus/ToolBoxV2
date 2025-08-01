import ast
import asyncio
import importlib
import io
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout, contextmanager
from copy import deepcopy
from dataclasses import dataclass

### ---- Styles ------- ###
from enum import Enum, auto
from inspect import (
    Signature,
    currentframe,
    getdoc,
    isclass,
    isfunction,
    ismethod,
    signature,
)
from pathlib import Path
from typing import Any, Optional, Dict

import nest_asyncio
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

import toolboxv2
from toolboxv2 import Spinner, Style, get_app
from toolboxv2.mods.isaa.extras.session import ChatSession


@dataclass
class JSExecutionRecord:
    """Records JavaScript execution details"""
    code: str
    result: Any
    error: str | None = None
    page_state: dict | None = None
    extracted_data: dict | None = None


class DynamicVerboseFormatter:
    """Unified, dynamic formatter that adapts to screen size"""

    def __init__(self, print_func=None, min_width: int = 40, max_width: int = 240):
        self.style = Style()
        self.print = print_func or print
        self.min_width = min_width
        self.max_width = max_width
        self._terminal_width = self._get_terminal_width()


    def get_git_info(self):
        """Checks for a git repo and returns its name and branch, or None."""
        try:
            # Check if we are in a git repository
            subprocess.check_output(['git', 'rev-parse', '--is-inside-work-tree'], stderr=subprocess.DEVNULL)

            # Get the repo name (root folder name)
            repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'],
                                                stderr=subprocess.DEVNULL).strip().decode('utf-8')
            repo_name = os.path.basename(repo_root)

            # Get the current branch name
            branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                             stderr=subprocess.DEVNULL).strip().decode('utf-8')

            return repo_name, branch
        except (subprocess.CalledProcessError, FileNotFoundError):
            # This handles cases where 'git' is not installed or it's not a git repo
            return None

    def _get_terminal_width(self) -> int:
        """Get current terminal width with fallback"""
        try:
            width = shutil.get_terminal_size().columns
            return max(self.min_width, min(width - 2, self.max_width))
        except (OSError, AttributeError):
            return 80

    def _wrap_text(self, text: str, width: int = None) -> list[str]:
        """Wrap text to fit terminal width"""
        if width is None:
            width = self._terminal_width - 4  # Account for borders

        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + len(current_line) <= width:
                current_line.append(word)
                current_length += len(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def _create_border(self, char: str = "─", width: int = None) -> str:
        """Create a border line that fits the terminal"""
        if width is None:
            width = self._terminal_width
        return char * width

    def _center_text(self, text: str, width: int = None) -> str:
        """Center text within the given width"""
        if width is None:
            width = self._terminal_width

        # Remove ANSI codes for length calculation
        clean_text = self._strip_ansi(text)
        padding = max(0, (width - len(clean_text)) // 2)
        return " " * padding + text

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape codes for length calculation"""
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def print_header(self, text: str):
        """Print a dynamic header that adapts to screen size"""
        self._terminal_width = self._get_terminal_width()

        if self._terminal_width < 60:  # Tiny screen
            self.print()
            self.print(self.style.CYAN("=" * self._terminal_width))
            self.print(self.style.CYAN(self.style.Bold(text)))
            self.print(self.style.CYAN("=" * self._terminal_width))
        else:  # Regular/large screen
            border_width = min(len(text) + 2, self._terminal_width - 2)
            border = "─" * border_width

            self.print()
            self.print(self.style.CYAN(f"┌{border}┐"))
            self.print(self.style.CYAN(f"│ {self.style.Bold(text).center(border_width - 2)} │"))
            self.print(self.style.CYAN(f"└{border}┘"))
        self.print()

    def print_section(self, title: str, content: str):
        """Print a clean section with adaptive formatting"""
        self._terminal_width = self._get_terminal_width()

        # Title
        if self._terminal_width < 60:
            self.print(f"\n{self.style.BLUE('●')} {self.style.Bold(title)}")
        else:
            self.print(f"\n{self.style.BLUE('●')} {self.style.Bold(self.style.BLUE(title))}")

        # Content with proper wrapping
        for line in content.split('\n'):
            if line.strip():
                wrapped_lines = self._wrap_text(line.strip())
                for wrapped_line in wrapped_lines:
                    if self._terminal_width < 60:
                        self.print(f"  {wrapped_line}")
                    else:
                        self.print(f"  {self.style.GREY('│')} {wrapped_line}")
        self.print()

    def print_progress_bar(self, current: int, maximum: int, title: str = "Progress"):
        """Dynamic progress bar that adapts to screen size"""
        self._terminal_width = self._get_terminal_width()

        # Calculate bar width based on screen size
        if self._terminal_width < 60:
            bar_width = 10
            template = f"\r{title}: [{{}}] {current}/{maximum}"
        else:
            bar_width = min(30, self._terminal_width - 30)
            template = f"\r{self.style.CYAN(title)}: [{{}}] {current}/{maximum} ({current / maximum * 100:.1f}%)"

        progress = int((current / maximum) * bar_width)
        bar = "█" * progress + "░" * (bar_width - progress)

        self.print(template.format(bar), end='', flush=True)

    def print_state(self, state: str, details: Dict[str, Any] = None) -> str:
        """Print current state with adaptive formatting"""
        self._terminal_width = self._get_terminal_width()

        state_colors = {
            'ACTION': self.style.GREEN2,
            'PROCESSING': self.style.YELLOW2,
            'BRAKE': self.style.RED2,
            'DONE': self.style.BLUE2,
            'ERROR': self.style.RED,
            'SUCCESS': self.style.GREEN,
            'INFO': self.style.CYAN
        }

        color_func = state_colors.get(state.upper(), self.style.WHITE2)

        if self._terminal_width < 60:
            # Compact format for small screens
            self.print(f"\n[{color_func(state)}]")
            result = f"\n[{state}]"
        else:
            # Full format for larger screens
            self.print(f"\n{self.style.Bold('State:')} {color_func(state)}")
            result = f"\nState: {state}"

        if details:
            for key, value in details.items():
                # Truncate long values on small screens
                if self._terminal_width < 60 and len(str(value)) > 30:
                    display_value = str(value)[:27] + "..."
                else:
                    display_value = str(value)

                if self._terminal_width < 60:
                    self.print(f"  {key}: {display_value}")
                    result += f"\n  {key}: {display_value}"
                else:
                    self.print(f"  {self.style.GREY('├─')} {self.style.CYAN(key)}: {display_value}")
                    result += f"\n  ├─ {key}: {display_value}"

        return result

    def print_code_block(self, code: str, language: str = "python"):
        """Print code with syntax awareness and proper formatting"""
        self._terminal_width = self._get_terminal_width()

        if self._terminal_width < 60:
            # Simple format for small screens
            self.print(f"\n{self.style.GREY('Code:')}")
            for line in code.split('\n'):
                self.print(f"  {line}")
        else:
            # Detailed format for larger screens
            self.print(f"\n{self.style.BLUE('┌─')} {self.style.YELLOW2(f'{language.upper()} Code')}")

            lines = code.split('\n')
            for i, line in enumerate(lines):
                if i == len(lines) - 1 and not line.strip():
                    continue

                # Wrap long lines
                if len(line) > self._terminal_width - 6:
                    wrapped = self._wrap_text(line, self._terminal_width - 6)
                    for j, wrapped_line in enumerate(wrapped):
                        prefix = "│" if j == 0 else "│"
                        self.print(f"{self.style.BLUE(prefix)} {wrapped_line}")
                else:
                    self.print(f"{self.style.BLUE('│')} {line}")

            self.print(f"{self.style.BLUE('└─')} {self.style.GREY('End of code block')}")

    def print_table(self, headers: list[str], rows: list[list[str]]):
        """Print a dynamic table that adapts to screen size"""
        self._terminal_width = self._get_terminal_width()

        if not rows:
            return

        # Calculate column widths
        all_data = [headers] + rows
        col_widths = []

        for col in range(len(headers)):
            max_width = max(len(str(row[col])) for row in all_data if col < len(row))
            col_widths.append(min(max_width, self._terminal_width // len(headers) - 2))

        # Adjust if total width exceeds terminal
        total_width = sum(col_widths) + len(headers) * 3 + 1
        if total_width > self._terminal_width:
            # Proportionally reduce column widths
            scale_factor = (self._terminal_width - len(headers) * 3 - 1) / sum(col_widths)
            col_widths = [max(8, int(w * scale_factor)) for w in col_widths]

        # Print table
        self._print_table_row(headers, col_widths, is_header=True)
        self._print_table_separator(col_widths)

        for row in rows:
            self._print_table_row(row, col_widths)

    def _print_table_row(self, row: list[str], widths: list[int], is_header: bool = False):
        """Helper method to print a table row"""
        formatted_cells = []
        for i, (cell, width) in enumerate(zip(row, widths)):
            cell_str = str(cell)
            if len(cell_str) > width:
                cell_str = cell_str[:width - 3] + "..."

            if is_header:
                formatted_cells.append(self.style.Bold(self.style.CYAN(cell_str.ljust(width))))
            else:
                formatted_cells.append(cell_str.ljust(width))

        self.print(f"│ {' │ '.join(formatted_cells)} │")

    def _print_table_separator(self, widths: list[int]):
        """Helper method to print table separator"""
        parts = ['─' * w for w in widths]
        self.print(f"├─{'─┼─'.join(parts)}─┤")

    async def process_with_spinner(self, message: str, coroutine):
        """Execute coroutine with adaptive spinner"""
        self._terminal_width = self._get_terminal_width()

        if self._terminal_width < 60:
            # Simple spinner for small screens
            spinner_symbols = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        else:
            # Detailed spinner for larger screens
            spinner_symbols = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

        # Truncate message if too long
        if len(message) > self._terminal_width - 10:
            display_message = message[:self._terminal_width - 13] + "..."
        else:
            display_message = message

        with Spinner(f"{self.style.CYAN('●')} {display_message}", symbols=spinner_symbols):
            return await coroutine

    def print_git_info(self) -> Optional[str]:
        """Get current git branch with error handling"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                branch = result.stdout.strip()

                # Check for uncommitted changes
                status_result = subprocess.run(
                    ['git', 'status', '--porcelain'],
                    capture_output=True, text=True, timeout=1
                )
                dirty = "*" if status_result.stdout.strip() else ""

                git_info = f"{branch}{dirty}"
                self.print_info(f"Git: {git_info}")
                return git_info
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        return None

    # Convenience methods with consistent styling
    def print_error(self, message: str):
        """Print error message with consistent formatting"""
        self.print(f"{self.style.RED('✗')} {self.style.RED(message)}")

    def print_success(self, message: str):
        """Print success message with consistent formatting"""
        self.print(f"{self.style.GREEN('✓')} {self.style.GREEN(message)}")

    def print_warning(self, message: str):
        """Print warning message with consistent formatting"""
        self.print(f"{self.style.YELLOW('⚠')} {self.style.YELLOW(message)}")

    def print_info(self, message: str):
        """Print info message with consistent formatting"""
        self.print(f"{self.style.CYAN('ℹ')} {self.style.CYAN(message)}")

    def print_debug(self, message: str):
        """Print debug message with consistent formatting"""
        self.print(f"{self.style.GREY('🐛')} {self.style.GREY(message)}")


class EnhancedVerboseOutput:
    """Main interface for verbose output with full functionality"""

    def __init__(self, verbose: bool = True, print_func=None, **formatter_kwargs):
        self.verbose = verbose
        self.print = print_func or print
        self.formatter = DynamicVerboseFormatter(self.print, **formatter_kwargs)
        self._start_time = time.time()

    def __getattr__(self, name):
        """Delegate to formatter for convenience"""
        return getattr(self.formatter, name)

    async def print_agent_response(self, response: str):
        await self.log_message("assistant", response)

    async def print_thought(self, thought: str):
        await self.log_message("assistant", f"Thought: {thought}")

    async def log_message(self, role: str, content: str):
        """Log chat messages with role-based formatting"""
        if not self.verbose:
            return

        role_formats = {
            'user': (self.formatter.style.GREEN, "👤"),
            'assistant': (self.formatter.style.BLUE, "🤖"),
            'system': (self.formatter.style.YELLOW, "⚙️"),
            'error': (self.formatter.style.RED, "❌"),
            'debug': (self.formatter.style.GREY, "🐛")
        }

        color_func, icon = role_formats.get(role.lower(), (self.formatter.style.WHITE, "•"))

        # Adapt formatting based on screen size
        if self.formatter._terminal_width < 60:
            self.print(f"\n{icon} [{role.upper()}]")
            # Wrap content for small screens
            wrapped_content = self.formatter._wrap_text(content, self.formatter._terminal_width - 2)
            for line in wrapped_content:
                self.print(f"  {line}")
        else:
            self.print(f"\n{icon} {color_func(f'[{role.upper()}]')}")
            self.print(f"{self.formatter.style.GREY('└─')} {content}")
        self.print()

    async def log_process_result(self, result: Dict[str, Any]):
        """Log processing results with structured formatting"""
        if not self.verbose:
            return

        content_parts = []

        if 'action' in result:
            content_parts.append(f"Action: {result['action']}")
        if 'is_completed' in result:
            content_parts.append(f"Completed: {result['is_completed']}")
        if 'effectiveness' in result:
            content_parts.append(f"Effectiveness: {result['effectiveness']}")
        if 'recommendations' in result:
            content_parts.append(f"Recommendations:\n{result['recommendations']}")
        if 'workflow' in result:
            content_parts.append(f"Workflow:\n{result['workflow']}")
        if 'errors' in result and result['errors']:
            content_parts.append(f"Errors: {result['errors']}")
        if 'content' in result:
            content_parts.append(f"Content:\n{result['content']}")

        self.formatter.print_section("Process Result", '\n'.join(content_parts))

    def log_header(self, text: str):
        """Log header with timing information"""
        if not self.verbose:
            return

        elapsed = time.time() - self._start_time
        if elapsed > 60:
            timing = f" ({elapsed / 60:.1f}m)"
        else:
            timing = f" ({elapsed:.1f}s)"

        self.formatter.print_header(f"{text}{timing}")

    def log_state(self, state: str, user_ns: Dict = None, override: bool = False):
        """Log state with optional override"""
        if not self.verbose and not override:
            return

        return self.formatter.print_state(state, user_ns)

    async def process(self, message: str, coroutine):
        """Process with optional spinner"""
        if not self.verbose:
            return await coroutine

        if message.lower() in ["code", "silent"]:
            return await coroutine

        return await self.formatter.process_with_spinner(message, coroutine)

    def print_tool_call(self, tool_name: str, tool_args: Dict, result: Optional[str] = None):
        """
        Gibt Informationen zum Tool-Aufruf aus.
        Versucht, das Ergebnis als JSON zu formatieren, wenn möglich.
        """
        if not self.verbose:
            return

        # Argumente wie zuvor formatieren
        args_str = json.dumps(tool_args, indent=2, ensure_ascii=False) if tool_args else "None"
        content = f"Tool: {tool_name}\nArguments:\n{args_str}"

        if result:
            result_output = ""
            try:
                # 1. Versuch, den String als JSON zu parsen
                data = json.loads(result)

                # 2. Prüfen, ob das Ergebnis ein Dictionary ist (der häufigste Fall)
                if isinstance(data, dict):
                    # Eine Kopie für die Anzeige erstellen, um den 'output'-Wert zu ersetzen
                    display_data = data.copy()
                    output_preview = ""

                    # Spezielle Handhabung für einen langen 'output'-String, falls vorhanden
                    if 'output' in display_data and isinstance(display_data['output'], str):
                        full_output = display_data['output']
                        # Den langen String im JSON durch einen Platzhalter ersetzen
                        display_data['output'] = "<-- [Inhalt wird separat formatiert]"

                        # Vorschau mit den ersten 3 Zeilen erstellen
                        lines = full_output.strip().split('\n')[:3]
                        preview_text = '\n'.join(lines)
                        output_preview = f"\n\n--- Vorschau für 'output' ---\n\x1b[90m{preview_text}\n...\x1b[0m"  # Hellgrauer Text
                        # display_data['output'] = output_preview
                    # Das formatierte JSON (mit Platzhalter) zum Inhalt hinzufügen
                    formatted_json = json.dumps(display_data, indent=2, ensure_ascii=False)
                    result_output = f"Geparstes Dictionary:\n{formatted_json}{output_preview}"

                else:
                    # Falls es valides JSON, aber kein Dictionary ist (z.B. eine Liste)
                    result_output = f"Gepastes JSON (kein Dictionary):\n{json.dumps(data, indent=2, ensure_ascii=False)}"

            except json.JSONDecodeError:
                # 3. Wenn Parsen fehlschlägt, den String als Rohtext behandeln
                result_output = f"{result}"

            content += f"\nResult:\n{result_output}"

        else:
            # Fall, wenn der Task noch läuft
            content += "\nResult: In progress..."

        # Den gesamten Inhalt an den Formatter übergeben
        self.formatter.print_section("Tool Call", content)

    def print_event(self, event: Dict):
        """Print event information"""
        if not self.verbose:
            return

        if event.get("content") and event["content"].get("parts"):
            for part in event["content"]["parts"]:
                if part.get("text"):
                    self.formatter.print_info(f"Thought: {part['text']}")
                if part.get("function_call"):
                    self.print_tool_call(
                        part["function_call"]["name"],
                        part["function_call"]["args"]
                    )
                if part.get("function_response"):
                    result = part["function_response"]["response"].get("result", "")
                    self.print_tool_call(
                        part["function_response"]["name"],
                        {},
                        str(result)
                    )

        if event.get("usage_metadata"):
            self.formatter.print_info(f"Token usage: {event['usage_metadata']}")

    @contextmanager
    def section_context(self, title: str):
        """Context manager for sections"""
        if self.verbose:
            self.formatter.print_section(title, "Starting...")
        try:
            yield
        finally:
            if self.verbose:
                self.formatter.print_success(f"Completed: {title}")

    def clear_line(self):
        """Clear current line"""
        self.print('\r' + ' ' * self.formatter._terminal_width + '\r', end='')

    def print_separator(self, char: str = "─"):
        """Print a separator line"""
        self.print(self.formatter.style.GREY(char * self.formatter._terminal_width))

    def print_warning(self, message: str):
        """Print a warning message with yellow style"""
        if self.verbose:
            self.print(self.formatter.style.YELLOW(f"⚠️  WARNING: {message}"))

    def print_error(self, message: str):
        """Print an error message with red style"""
        if self.verbose:
            self.print(self.formatter.style.RED(f"❌ ERROR: {message}"))

    def print_success(self, message: str):
        """Print a success message with green style"""
        if self.verbose:
            self.print(self.formatter.style.GREEN(f"✅ SUCCESS: {message}"))



### -- TYPESs --- ###

class ThinkState(Enum):
    ACTION = auto()
    PROCESSING = auto()
    BRAKE = auto()
    DONE = auto()


class MethodUpdate(BaseModel):
    class_name: str = Field(..., description="Name of the class to update")
    method_name: str = Field(..., description="Name of the method to update")
    code: str = Field(..., description="Python code for the method implementation")
    description: str | None = Field(None, description="Description of what the method does")


class ThinkResult(BaseModel):
    action: str = Field(..., description="Next action to take: 'code', 'brake', 'done'")
    content: str = Field(..., description="Content related to the action")
    context: dict[str, str | int | float | bool | dict[str, str | int | float | bool]] | None = Field(default_factory=dict, description="Additional context for the action")

class ThinkResults(BaseModel):
    actions:list[ThinkResult]

@dataclass
class ExecutionRecord:
    code: str
    result: Any
    error: str | None = None

    def __str__(self):
        return  '' if self.result is None and self.error is None else f"Output -> {self.result if self.result else ''}{'(error: '+self.error+')' if self.error else ''}"


@dataclass
class PipelineResult:
    variables: dict[str, Any]
    result: Any
    execution_history: list[ExecutionRecord]
    message: list[dict[str, str]]



class CargoRustInterface:
    '''Usage :
# Create interface
cargo_interface = CargoRustInterface()

# Set up new project
await cargo_interface.setup_project("hello_rust")

# Add a dependency
await cargo_interface.add_dependency("serde", "1.0")

# Write and run some code
code = """
fn main() {
    println!("Hello, Rust!");
}
"""
result = await cargo_interface.run_code(code)

# Modify code
new_function = """
fn main() {
    println!("Modified Hello, Rust!");
}
"""
await cargo_interface.modify_code(new_function, "main()")

# Build and test
await cargo_interface.build()
await cargo_interface.test()

    '''
    def __init__(self, session_dir=None, auto_remove=True):
        """Initialize the Rust/Cargo interface"""
        self.auto_remove = auto_remove
        self._session_dir = session_dir or Path.home() / '.cargo_sessions'
        self._session_dir.mkdir(exist_ok=True)
        self.vfs = VirtualFileSystem(self._session_dir / 'virtual_fs')
        self.output_history = {}
        self._execution_count = 0
        self.current_project = None

    def reset(self):
        """Reset the interface state"""
        if self.auto_remove and self.current_project:
            shutil.rmtree(self.current_project, ignore_errors=True)
        self.output_history.clear()
        self._execution_count = 0
        self.current_project = None

    async def setup_project(self, name: str) -> str:
        """Set up a new Cargo project"""
        try:
            project_path = self.vfs.base_dir / name
            if project_path.exists():
                shutil.rmtree(project_path)

            result = subprocess.run(
                ['cargo', 'new', str(project_path)],
                capture_output=True,
                text=True, check=True
            )

            if result.returncode != 0:
                return f"Error creating project: {result.stderr}"

            self.current_project = project_path
            return f"Created new project at {project_path}"

        except Exception as e:
            return f"Failed to create project: {str(e)}"

    async def add_dependency(self, name: str, version: str | None = None) -> str:
        """Add a dependency to Cargo.toml"""
        if not self.current_project:
            return "No active project"

        try:
            cargo_toml = self.current_project / "Cargo.toml"
            if not cargo_toml.exists():
                return "Cargo.toml not found"

            cmd = ['cargo', 'add', name]
            if version:
                cmd.extend(['--vers', version])

            result = subprocess.run(
                cmd,
                cwd=self.current_project,
                capture_output=True,
                text=True,check=True
            )

            return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"

        except Exception as e:
            return f"Failed to add dependency: {str(e)}"

    async def build(self, release: bool = False) -> str:
        """Build the project"""
        if not self.current_project:
            return "No active project"

        try:
            cmd = ['cargo', 'build']
            if release:
                cmd.append('--release')

            result = subprocess.run(
                cmd,
                cwd=self.current_project,
                capture_output=True,
                text=True
            )

            return result.stdout if result.returncode == 0 else f"Build error: {result.stderr}"

        except Exception as e:
            return f"Build failed: {str(e)}"

    async def test(self) -> str:
        """Run project tests"""
        if not self.current_project:
            return "No active project"

        try:
            result = subprocess.run(
                ['cargo', 'test'],
                cwd=self.current_project,
                capture_output=True,
                text=True, check=True
            )

            return result.stdout if result.returncode == 0 else f"Test error: {result.stderr}"

        except Exception as e:
            return f"Tests failed: {str(e)}"

    async def run_code(self, code: str) -> str:
        """Run Rust code"""
        if not self.current_project:
            return "No active project"

        try:
            # Write code to main.rs
            main_rs = self.current_project / "src" / "main.rs"
            with open(main_rs, 'w') as f:
                f.write(code)

            # Build and run
            build_result = subprocess.run(
                ['cargo', 'build'],
                cwd=self.current_project,
                capture_output=True,
                text=True
            )

            if build_result.returncode != 0:
                return f"Compilation error: {build_result.stderr}"

            run_result = subprocess.run(
                ['cargo', 'run'],
                cwd=self.current_project,
                capture_output=True,
                text=True
            )

            self._execution_count += 1
            output = {
                'code': code,
                'stdout': run_result.stdout,
                'stderr': run_result.stderr,
                'result': run_result.returncode == 0
            }
            self.output_history[self._execution_count] = output

            return run_result.stdout if run_result.returncode == 0 else f"Runtime error: {run_result.stderr}"

        except Exception as e:
            return f"Execution failed: {str(e)}"

    async def modify_code(self, code: str, object_name: str, file: str = "src/main.rs") -> str:
        """Modify existing Rust code"""
        if not self.current_project:
            return "No active project"

        try:
            file_path = self.current_project / file
            if not file_path.exists():
                return f"File {file} not found"

            with open(file_path) as f:
                content = f.read()

            # Handle function modification
            if object_name.endswith("()"):
                func_name = object_name[:-2]
                # Find and replace function definition
                pattern = f"fn {func_name}.*?}}(?=\n|$)"
                updated_content = re.sub(pattern, code.strip(), content, flags=re.DOTALL)
            else:
                # Handle other modifications (structs, constants, etc.)
                pattern = f"{object_name}.*?(?=\n|$)"
                updated_content = re.sub(pattern, code.strip(), content)

            with open(file_path, 'w') as f:
                f.write(updated_content)

            return f"Modified {object_name} in {file}"

        except Exception as e:
            return f"Modification failed: {str(e)}"

    def save_session(self, name: str):
        """Save current session state"""
        session_file = self._session_dir / f"{name}.json"
        state = {
            'output_history': self.output_history,
            'current_project': str(self.current_project) if self.current_project else None
        }

        with open(session_file, 'w') as f:
            json.dump(state, f)

    def load_session(self, name: str):
        """Load saved session state"""
        session_file = self._session_dir / f"{name}.json"
        if session_file.exists():
            with open(session_file) as f:
                state = json.load(f)
                self.output_history = state['output_history']
                self.current_project = Path(state['current_project']) if state['current_project'] else None
### ---- logic ---- ###

class VirtualFileSystem:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.current_dir = base_dir
        self.virtual_files: dict[str, str] = {}
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_file(self, filepath: str | Path, content: str) -> Path:
        """Write content to a virtual file and persist to disk using UTF-8"""
        try:
            abs_path = self._resolve_path(filepath)
        except ValueError:
            print("invalid :", filepath)
            filepath = "src/temp_js/_temp_fix.py"
            abs_path = self._resolve_path(filepath)
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        # Store in virtual filesystem
        rel_path = str(abs_path.relative_to(self.base_dir))
        self.virtual_files[rel_path] = content

        # Write to actual filesystem with UTF-8 encoding
        with open(abs_path, 'w', encoding='utf-8', errors='replace') as f:
            f.write(content)

        return abs_path

    def read_file(self, filepath: str | Path) -> str:
        """Read content from a virtual file using UTF-8"""
        abs_path = self._resolve_path(filepath)
        if not abs_path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        rel_path = str(abs_path.relative_to(self.base_dir))

        # Check virtual filesystem first
        if rel_path in self.virtual_files:
            return self.virtual_files[rel_path]

        # Fall back to reading from disk with UTF-8 encoding
        with open(abs_path, encoding='utf-8', errors='replace') as f:
            content = f.read()
            self.virtual_files[rel_path] = content
            return content

    def delete_file(self, filepath: str | Path):
        """Delete a virtual file"""
        abs_path = self._resolve_path(filepath)
        rel_path = str(abs_path.relative_to(self.base_dir))

        if rel_path in self.virtual_files:
            del self.virtual_files[rel_path]

        if abs_path.exists():
            abs_path.unlink()

    def create_directory(self, dirpath: str | Path):
        """Create a new directory"""
        abs_path = self._resolve_path(dirpath)
        abs_path.mkdir(parents=True, exist_ok=True)
        return abs_path


    def list_directory(self, dirpath: str | Path = '.') -> list:
        """List contents of a directory"""
        abs_path = self._resolve_path(dirpath)
        if not abs_path.exists():
            raise FileNotFoundError(f"Directory not found: {dirpath}")
        return [p.name for p in abs_path.iterdir()]

    def change_directory(self, dirpath: str | Path):
        """Change current working directory"""
        new_dir = self._resolve_path(dirpath)
        if not new_dir.exists() or not new_dir.is_dir():
            raise NotADirectoryError(f"Directory not found: {dirpath}")
        self.current_dir = new_dir

    def _resolve_path(self, filepath: str | Path) -> Path:
        """Convert relative path to absolute path"""
        filepath = Path(filepath)
        if filepath.is_absolute():
            if not str(filepath).startswith(str(self.base_dir)):
                raise ValueError("Path must be within base directory")
            return filepath
        return (self.current_dir / filepath).resolve()

    def save_state(self, state_file: Path):
        """Save virtual filesystem state to disk"""
        state = {
            'current_dir': str(self.current_dir.relative_to(self.base_dir)),
            'virtual_files': self.virtual_files
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

    def load_state(self, state_file: Path):
        """Load virtual filesystem state from disk"""
        if not state_file.exists():
            return

        with open(state_file) as f:
            state = json.load(f)
            self.current_dir = self.base_dir / state['current_dir']
            self.virtual_files = state['virtual_files']

    def print_file_structure(self, start_path: str | Path = '.', indent: str = ''):
        """Print the file structure starting from the given path"""
        start_path = self._resolve_path(start_path)
        if not start_path.exists():
            s = f"Path not found: {start_path}"
            return s

        s = f"{indent}{start_path.name}/"
        for item in sorted(start_path.iterdir()):
            if item.is_dir():
               s+= self.print_file_structure(item, indent + '  ')
            else:
                s = f"{indent}  {item.name}"
        return s






class VirtualEnvContext:
    """Context manager for temporary virtual environment activation"""

    def __init__(self, venv_path: Path):
        self.venv_path = venv_path
        self._original_path = None
        self._original_sys_path = None
        self._original_prefix = None
        self._original_virtual_env = None

    def _get_venv_paths(self):
        """Get virtual environment paths based on platform"""
        if sys.platform == 'win32':
            site_packages = self.venv_path / 'Lib' / 'site-packages'
            scripts_dir = self.venv_path / 'Scripts'
            python_path = scripts_dir / 'python.exe'
        else:
            python_version = f'python{sys.version_info.major}.{sys.version_info.minor}'
            site_packages = self.venv_path / 'lib' / python_version / 'site-packages'
            scripts_dir = self.venv_path / 'bin'
            python_path = scripts_dir / 'python'

        return site_packages, scripts_dir, python_path

    def __enter__(self):
        # Save original state
        self._original_path = os.environ.get('PATH', '')
        self._original_sys_path = sys.path.copy()
        self._original_prefix = sys.prefix
        self._original_virtual_env = os.environ.get('VIRTUAL_ENV')

        # Get venv paths
        site_packages, scripts_dir, python_path = self._get_venv_paths()

        # Modify environment for venv
        if scripts_dir.exists():
            new_path = os.pathsep.join([str(scripts_dir), self._original_path])
            os.environ['PATH'] = new_path

        if site_packages.exists():
            sys.path.insert(0, str(site_packages))

        os.environ['VIRTUAL_ENV'] = str(self.venv_path)

        # Return the python executable path for potential subprocess calls
        return str(python_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        os.environ['PATH'] = self._original_path
        sys.path = self._original_sys_path

        if self._original_virtual_env is None:
            os.environ.pop('VIRTUAL_ENV', None)
        else:
            os.environ['VIRTUAL_ENV'] = self._original_virtual_env

class TeeStream:
    """Stream that writes to both console and buffer"""
    def __init__(self, console_stream, buffer_stream):
        self.console_stream = console_stream
        self.buffer_stream = buffer_stream

    def write(self, data):
        self.console_stream.write(data)
        self.buffer_stream.write(data)
        self.console_stream.flush()  # Ensure immediate console output

    def flush(self):
        self.console_stream.flush()
        self.buffer_stream.flush()


class ParentNodeTransformer(ast.NodeTransformer):
    """Add parent references to AST nodes"""
    def visit(self, node):
        for child in ast.iter_child_nodes(node):
            child.parent = node
        return super().visit(node)

class AsyncCodeDetector(ast.NodeVisitor):
    """Detect async code and top-level await"""
    def __init__(self):
        self.has_async = False
        self.has_top_level_await = False
        self.await_nodes = []

    def visit_AsyncFunctionDef(self, node):
        self.has_async = True
        self.generic_visit(node)

    def visit_Await(self, node):
        self.has_async = True
        # Track all await nodes
        self.await_nodes.append(node)
        # Check if this await is at top level
        parent = node
        while hasattr(parent, 'parent'):
            parent = parent.parent
            if isinstance(parent, ast.AsyncFunctionDef | ast.FunctionDef):
                break
        else:
            self.has_top_level_await = True
        self.generic_visit(node)

def auto_install(package_name, install_method='pip', upgrade=False, quiet=False, version=None, extra_args=None):
    '''
    Enhanced auto-save import with version and extra arguments support
    '''
    try:
        # Attempt to import the package
        return importlib.import_module(package_name)
    except ImportError:
        # Package not found, prepare for installation
        print(f"Package '{package_name}' not found. Attempting to install...")
        try:
            # Determine Python executable based on virtual environment
            venv_path = os.environ.get('VIRTUAL_ENV')
            if venv_path:
                venv_path = Path(venv_path)
                if sys.platform == 'win32':
                    python_exec = str(venv_path / 'Scripts' / 'python.exe')
                else:
                    python_exec = str(venv_path / 'bin' / 'python')
                # Check if the Python executable exists
                if not Path(python_exec).exists():
                    python_exec = sys.executable
            else:
                python_exec = sys.executable

            # Construct installation command with more flexibility
            install_cmd = [python_exec, "-m", install_method, "install"]
            if upgrade:
                install_cmd.append("--upgrade")
            # Support specific version installation
            if version:
                install_cmd.append(f"{package_name}=={version}")
            else:
                install_cmd.append(package_name)
            # Add extra arguments if provided
            if extra_args:
                install_cmd.extend(extra_args)
            # Run installation with appropriate verbosity
            installation_output = subprocess.run(
                install_cmd,
                capture_output=quiet,
                text=True
            )
            # Check installation status
            if installation_output.returncode == 0:
                print(f"Successfully installed {package_name}")
                return importlib.import_module(package_name)
            else:
                raise Exception(f"Installation failed: {installation_output.stderr}")
        except Exception as install_error:
            print(f"Error installing {package_name}: {install_error}")
            return None

class MockIPython:
    def __init__(self, _session_dir=None, auto_remove=True):
        self.auto_remove = auto_remove
        self.output_history = {}
        self._execution_count = 0
        self._session_dir = _session_dir or Path(get_app().appdata) / '.pipeline_sessions'
        self._session_dir.mkdir(exist_ok=True)
        self.vfs = VirtualFileSystem(self._session_dir / 'virtual_fs')
        self._venv_path = self._session_dir / 'venv'
        self.user_ns: dict[str, Any] = {}
        nest_asyncio.apply()
        # Set up virtual environment if it doesn't exist
        with Spinner("Starting virtual environment"):
            self._setup_venv()
        self.reset()

    def _setup_venv(self):
        """Create virtual environment if it doesn't exist"""
        if not self._venv_path.exists():
            try:
                subprocess.run([sys.executable, "-m", "venv", str(self._venv_path)], check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to create virtual environment: {str(e)}")

    def _virtual_open(self, filepath, mode='r', *args, **kwargs):
        """Custom open function that uses virtual filesystem"""
        abs_path = self.vfs._resolve_path(filepath)

        if 'w' in mode or 'a' in mode:
            # Ensure parent directory exists
            abs_path.parent.mkdir(parents=True, exist_ok=True)

        # Use actual filesystem but track in virtual fs
        real_file = open(abs_path, mode, *args, **kwargs)

        if 'r' in mode:
            # Track file content in virtual filesystem when reading
            rel_path = str(abs_path.relative_to(self.vfs.base_dir))
            if rel_path not in self.vfs.virtual_files:
                try:
                    self.vfs.virtual_files[rel_path] = real_file.read()
                    real_file.seek(0)
                except UnicodeDecodeError:
                    # Handle binary files
                    pass

        return real_file

    def reset(self):
        """Reset the interpreter state"""
        self.user_ns = {
            '__name__': '__main__',
            '__builtins__': __builtins__,
            'toolboxv2': toolboxv2,
            '__file__': None,
            '__path__': [str(self.vfs.current_dir)],
            'auto_install': auto_install,
            'modify_code': self.modify_code,
        }
        self.output_history.clear()
        self._execution_count = 0
        if self.auto_remove:
            shutil.rmtree(self.vfs.base_dir, ignore_errors=True)

    def get_namespace(self) -> dict[str, Any]:
        """Get current namespace"""
        return self.user_ns.copy()

    def update_namespace(self, variables: dict[str, Any]):
        """Update namespace with new variables"""
        self.user_ns.update(variables)

    @staticmethod
    def _parse_code(code: str) -> tuple[Any, Any | None, bool, bool]:
        """Parse code and handle top-level await"""
        code_ = ""
        for line in code.split('\n'):
            if line.strip().startswith('#'):
                continue
            if line.strip().startswith('asyncio.run('):
                line = (' ' *(len(line) - len(line.strip()))) + 'await ' + line.strip()[len('asyncio.run('):-1]
            code_ += line + '\n'
        try:
            tree = ast.parse(code)
            # Add parent references
            ParentNodeTransformer().visit(tree)

            # Detect async features
            detector = AsyncCodeDetector()
            detector.visit(tree)

            if detector.has_top_level_await:
                # Wrap code in async function
                wrapped_code = "async def __wrapper():\n"
                wrapped_code += "    global result\n"  # Allow writing to global scope
                wrapped_code += "    result = None\n"
                # add try:
                wrapped_code +="    try:\n"
                # Indent the original code
                wrapped_code += "\n".join(f"        {line}" for line in code.splitlines())
                # Add return statement for last expression
                wrapped_code +="\n    except Exception as e:\n"
                wrapped_code +="        import traceback\n"
                wrapped_code +="        print(traceback.format_exc())\n"
                wrapped_code +="        raise e\n"
                if isinstance(tree.body[-1], ast.Expr):
                    wrapped_code += "\n    return result"

                # Parse and compile wrapped code
                wrapped_tree = ast.parse(wrapped_code)
                return (
                    compile(wrapped_tree, '<exec>', 'exec'),
                    None,
                    True,
                    True
                )

            # Handle regular code
            if isinstance(tree.body[-1], ast.Expr):
                exec_code = ast.Module(
                    body=tree.body[:-1],
                    type_ignores=[]
                )
                eval_code = ast.Expression(
                    body=tree.body[-1].value
                )
                return (
                    compile(exec_code, '<exec>', 'exec'),
                    compile(eval_code, '<eval>', 'eval'),
                    detector.has_async,
                    False
                )

            return (
                compile(tree, '<exec>', 'exec'),
                None,
                detector.has_async,
                False
            )

        except SyntaxError as e:
            lines = code.splitlines()
            if e.lineno and e.lineno <= len(lines):
                line = lines[e.lineno - 1]
                arrow = ' ' * (e.offset - 1) + '^' if e.offset else ''
                error_msg = (
                    f"Syntax error at line {e.lineno}:\n"
                    f"{line}\n"
                    f"{arrow}\n"
                    f"{e.msg}"
                )
            else:
                error_msg = str(e)

            error_msg += traceback.format_exc()

            raise SyntaxError(error_msg) from e

    async def run_cell(self, code: str, live_output: bool = True) -> Any:
        """Async version of run_cell that handles both sync and async code"""
        result = None
        error = None
        tb = None
        original_dir = os.getcwd()

        if live_output:
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            stdout = TeeStream(sys.__stdout__, stdout_buffer)
            stderr = TeeStream(sys.__stderr__, stderr_buffer)
        else:
            stdout = io.StringIO()
            stderr = io.StringIO()

        try:
            # Check if a file is already specified
            original_file = self.user_ns.get('__file__')
            if original_file is None:
                # Create temp file if no file specified
                temp_file = self.vfs.write_file(
                    f'src/temp/_temp_{self._execution_count}.py',
                    code
                )
                # work_ns = self.user_ns.copy()
                self.user_ns['__file__'] = str(temp_file)
            else:
                # Use existing file
                temp_file = Path(original_file)
                # Write code to the existing file
                self.vfs.write_file(temp_file, code)
                #work_ns = self.user_ns.copy()

            self.user_ns['__builtins__'] = __builtins__
            with VirtualEnvContext(self._venv_path) as python_exec:
                try:
                    exec_code, eval_code, is_async, has_top_level_await = self._parse_code(
                        code.encode('utf-8', errors='replace').decode('utf-8')
                    )
                    if exec_code is None:
                        return "No executable code"
                    os.makedirs(str(temp_file.parent.absolute()), exist_ok=True)
                    os.chdir(str(temp_file.parent.absolute()))
                    self.user_ns['PYTHON_EXEC'] = python_exec

                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        if has_top_level_await:
                            try:
                                # Execute wrapped code and await it
                                exec(exec_code, self.user_ns)
                                result = self.user_ns['__wrapper']()
                                if asyncio.iscoroutine(result):
                                    result = await result
                            finally:
                                self.user_ns.pop('__wrapper', None)
                        elif is_async:
                            # Execute async code
                            exec(exec_code, self.user_ns)
                            if eval_code:
                                result = eval(eval_code, self.user_ns)
                                if asyncio.iscoroutine(result):
                                    result = await result
                        else:
                            # Execute sync code
                            exec(exec_code, self.user_ns)
                            if eval_code:
                                result = eval(eval_code, self.user_ns)

                        if result is not None:
                            self.user_ns['_'] = result
                except KeyboardInterrupt:
                    print("Stop execution manuel!")

                except Exception as e:
                    error = str(e)
                    tb = traceback.format_exc()
                    if live_output:
                        sys.__stderr__.write(f"{error}\n{tb}")
                    stderr.write(f"{error}\n{tb}")

                finally:
                    os.chdir(original_dir)
                    self._execution_count += 1
                    # self.user_ns = work_ns.copy()
                    if live_output:
                        stdout_value = stdout_buffer.getvalue()
                        stderr_value = stderr_buffer.getvalue()
                    else:
                        stdout_value = stdout.getvalue()
                        stderr_value = stderr.getvalue()

                    output = {
                        'code': code,
                        'stdout': stdout_value,
                        'stderr': stderr_value,
                        'result': result if result else "stdout"
                    }
                    self.output_history[self._execution_count] = output

        except Exception as e:
            error_msg = f"Error executing code: {str(e)}\n{traceback.format_exc()}"
            if live_output:
                sys.__stderr__.write(error_msg)
            return error_msg

        if not result:
            result = ""
        if output['stdout']:
            result = f"{result}\nstdout:{output['stdout']}"
        if output['stderr']:
            result = f"{result}\nstderr:{output['stderr']}"

        if self.auto_remove and original_file is None:
            # Only remove temp files, not user-specified files
            self.vfs.delete_file(temp_file)

        return result

    async def modify_code(self, code: str = None, object_name: str = None, file: str = None) -> str:
        '''
        Modify existing code in memory (user namespace) and optionally in the corresponding file.

        This method updates variables, functions, or methods in the current Python session and can
        also update the corresponding source file if specified.

        Args:
            code: New value or implementation for the object
            object_name: Name of the object to modify (variable, function, or method)
            file: Path to the file to update (if None, only updates in memory)

        Returns:
            String describing the modification result

        Examples:

        # 1. Update a variable in memory
        await ipython.modify_code(code="5", object_name="x")

    # 2. Change a method implementation
    await ipython.modify_code(
        code='"""def sound(self):\n        return "Woof""""',
        object_name="Dog.sound"
    )

    # 3. Modify a function
    await ipython.modify_code(
        code='"""def calculate_age():\n    return 25"""',
        object_name="calculate_age"
    )

    # 4. Update variable in memory and file
    await ipython.modify_code(
        code="100",
        object_name="MAX_SIZE",
        file="config.py"
    )

    # 5. Modifying an attribute in __init__
    await ipython.modify_code(
        code='"""def __init__(self):\n        self.name = "Buddy""""',
        object_name="Dog.__init__"
    )
        '''
        try:
            if not object_name:
                raise ValueError("Object name must be specified")
            if code is None:
                raise ValueError("New code or value must be provided")

            # Process object name (handle methods with parentheses)
            clean_object_name = object_name.replace("()", "")

            # Step 1: Update in memory (user namespace)
            result_message = []

            # Handle different types of objects
            if "." in clean_object_name:
                # For methods or class attributes
                parts = clean_object_name.split(".")
                base_obj_name = parts[0]
                attr_name = parts[1]

                if base_obj_name not in self.user_ns:
                    raise ValueError(f"Object '{base_obj_name}' not found in namespace")

                base_obj = self.user_ns[base_obj_name]

                # Handle method definitions which are passed as docstrings
                if code.split('\n'):
                    method_code = code

                    # Parse the method code to extract its body
                    method_ast = ast.parse(method_code).body[0]
                    method_name = method_ast.name

                    # Create a new function object from the code
                    method_locals = {}
                    exec(
                        f"def _temp_func{signature(getattr(base_obj.__class__, attr_name, None))}: {method_ast.body[0].value.s}",
                        globals(), method_locals)
                    new_method = method_locals['_temp_func']

                    # Set the method on the class
                    setattr(base_obj.__class__, attr_name, new_method)
                    result_message.append(f"Updated method '{clean_object_name}' in memory")
                else:
                    # For simple attributes
                    setattr(base_obj, attr_name, eval(code, self.user_ns))
                    result_message.append(f"Updated attribute '{clean_object_name}' in memory")
            else:
                # For variables and functions
                if code.startswith('"""') and code.endswith('"""'):
                    # Handle function definitions
                    func_code = code.strip('"""')
                    func_ast = ast.parse(func_code).body[0]
                    func_name = func_ast.name

                    # Create a new function object from the code
                    func_locals = {}
                    exec(f"{func_code}", globals(), func_locals)
                    self.user_ns[clean_object_name] = func_locals[func_name]
                    result_message.append(f"Updated function '{clean_object_name}' in memory")
                else:
                    # Simple variable assignment
                    self.user_ns[clean_object_name] = eval(code, self.user_ns)
                    result_message.append(f"Updated variable '{clean_object_name}' in memory")

            # Step 2: Update in file if specified
            if file is not None:
                file_path = self.vfs._resolve_path(file)

                if not file_path.exists():
                    self.user_ns['__file__'] = str(file_path)
                    return await self.run_cell(code)

                # Read original content
                original_content = self.vfs.read_file(file_path)
                updated_content = original_content

                # Handle different object types for file updates
                if "." in clean_object_name:
                    # For methods
                    parts = clean_object_name.split(".")
                    class_name = parts[0]
                    method_name = parts[1]

                    if code.startswith('"""') and code.endswith('"""'):
                        method_code = code.strip('"""')

                        # Use ast to parse the file and find the method to replace
                        file_ast = ast.parse(original_content)
                        for node in ast.walk(file_ast):
                            if isinstance(node, ast.ClassDef) and node.name == class_name:
                                for method in node.body:
                                    if isinstance(method, ast.FunctionDef) and method.name == method_name:
                                        # Find the method in the source code
                                        method_pattern = fr"def {method_name}.*?:(.*?)(?=\n    \w|\n\w|\Z)"
                                        method_match = re.search(method_pattern, original_content, re.DOTALL)

                                        if method_match:
                                            indentation = re.match(r"^(\s*)", method_match.group(0)).group(1)
                                            method_indented = textwrap.indent(method_code, indentation)
                                            updated_content = original_content.replace(
                                                method_match.group(0),
                                                method_indented
                                            )
                                            self.vfs.write_file(file_path, updated_content)
                                            result_message.append(
                                                f"Updated method '{clean_object_name}' in file '{file}'")
                else:
                    # For variables and functions
                    if code.startswith('"""') and code.endswith('"""'):
                        # Handle function updates
                        func_code = code.strip('"""')
                        func_pattern = fr"def {clean_object_name}.*?:(.*?)(?=\n\w|\Z)"
                        func_match = re.search(func_pattern, original_content, re.DOTALL)

                        if func_match:
                            indentation = re.match(r"^(\s*)", func_match.group(0)).group(1)
                            func_indented = textwrap.indent(func_code, indentation)
                            updated_content = original_content.replace(
                                func_match.group(0),
                                func_indented
                            )
                            self.vfs.write_file(file_path, updated_content)
                            result_message.append(f"Updated function '{clean_object_name}' in file '{file}'")
                    else:
                        # Handle variable updates
                        var_pattern = fr"{clean_object_name}\s*=.*"
                        var_replacement = f"{clean_object_name} = {code}"
                        updated_content = re.sub(var_pattern, var_replacement, original_content)

                        if updated_content != original_content:
                            self.vfs.write_file(file_path, updated_content)
                            result_message.append(f"Updated variable '{clean_object_name}' in file '{file}'")
                        else:
                            result_message.append(f"Could not find variable '{clean_object_name}' in file '{file}'")

            return "\n".join(result_message)

        except Exception as e:
            return f"Error during code modification: {str(e)}\n{traceback.format_exc()}"


    def save_session(self, name: str):
        """Save session with UTF-8 encoding"""
        session_file = self._session_dir / f"{name}.pkl"
        user_ns = self.user_ns.copy()
        output_history = self.output_history.copy()

        # Ensure all strings are properly encoded
        for key, value in user_ns.items():
            try:
                if isinstance(value, str):
                    value = value.encode('utf-8').decode('utf-8')
                pickle.dumps(value)
            except Exception:
                user_ns[key] = f"not serializable: {str(value)}"

        for key, value in output_history.items():
            try:
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, str):
                            value[k] = v.encode('utf-8').decode('utf-8')
                pickle.dumps(value)
            except Exception:
                output_history[key] = f"not serializable: {str(value)}"


        session_data = {
            'user_ns': user_ns,
            'output_history': output_history,

        }

        with open(session_file, 'wb') as f:
            pickle.dump(session_data, f)

        # Save VFS state with UTF-8 encoding
        vfs_state_file = self._session_dir / f"{name}_vfs.json"
        with open(vfs_state_file, 'w', encoding='utf-8') as f:
            json.dump(self.vfs.virtual_files, f, ensure_ascii=False)

    def load_session(self, name: str):
        """Load session with UTF-8 encoding"""
        session_file = self._session_dir / f"{name}.pkl"
        if session_file.exists():
            with open(session_file, 'rb') as f:
                session_data = pickle.load(f)
                # self.user_ns.update(session_data['user_ns'])
                self.output_history.update(session_data['output_history'])

        # Load VFS state with UTF-8 encoding
        vfs_state_file = self._session_dir / f"{name}_vfs.json"
        if vfs_state_file.exists():
            with open(vfs_state_file, encoding='utf-8') as f:
                self.vfs.virtual_files = json.load(f)

    def __str__(self):
        """String representation of current session"""
        output = []
        for count, data in self.output_history.items():
            output.append(f"In [{count}]: {data['code']}")
            if data['stdout']:
                output.append(data['stdout'])
            if data['stderr']:
                output.append(f"Error: {data['stderr']}")
            if data['result'] is not None:
                output.append(f"Out[{count}]: {data['result']}")
        return "\n".join(output)


def super_strip(s: str) -> str:
    # Remove ANSI escape sequences (e.g. "\x1b[K", "\x1b[...m", etc.)
    s = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', s)

    # Remove any header text before the first "Episode"
    episode_index = s.find("Episode")
    if episode_index != -1:
        s = s[episode_index:]

    # Split the string into lines (split on newline characters)
    lines = s.splitlines()
    processed_lines = []
    for line in lines:
        # If the line contains carriage returns,
        # only keep the text after the last one.
        if "\r" in line:
            line = line.split("\r")[-1]
        processed_lines.append(line)

    # Rejoin the processed lines with newline characters.
    return "\n".join(processed_lines)

async def default_python_execute_function(files):
    # Create a temporary directory to store the files
    temp_dir = Path("./temp_project")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Write files to the temporary directory
        for file_path, content in files.items():
            full_path = temp_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)

        # Check if main.py exists
        main_file = temp_dir / "main.py"
        if main_file.exists():
            # Run main.py
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(main_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            return f"Main execution result:\nStdout:\n{stdout.decode()}\nStderr:\n{stderr.decode()}"

        # If main.py doesn't exist, look for files with __main__ block
        main_files = []
        for file_path in temp_dir.glob("**/*.py"):
            if "__main__" in file_path.read_text():
                main_files.append(file_path)

        if main_files:
            results = []
            for file in main_files:
                process = await asyncio.create_subprocess_exec(
                    sys.executable, str(file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                results.append(f"Execution of {file.name}:\nStdout:\n{stdout.decode()}\nStderr:\n{stderr.decode()}")
            return "\n\n".join(results)

        # If no main files found, run pytest
        pytest_output = subprocess.run(
            [sys.executable, "-m", "pytest", str(temp_dir)],
            capture_output=True,
            text=True
        )
        return f"Pytest execution result:\n{pytest_output.stdout}\n{pytest_output.stderr}"

    finally:
        # Clean up temporary directory
        for file in temp_dir.glob("**/*"):
            if file.is_file():
                file.unlink()
        for dir in reversed(list(temp_dir.glob("**/*"))):
            if dir.is_dir():
                dir.rmdir()
        temp_dir.rmdir()


async def default_rust_execute_function(files):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write files to the temporary directory
        for file_path, content in files.items():
            full_path = os.path.join(temp_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)

        # Check if there's a Cargo.toml file
        if "Cargo.toml" in files:
            # Run cargo check for syntax and compiler errors
            process = await asyncio.create_subprocess_exec(
                "cargo", "check",
                cwd=temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            check_stdout, check_stderr = await process.communicate()

            # Run cargo run for execution
            process = await asyncio.create_subprocess_exec(
                "cargo", "run",
                cwd=temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            run_stdout, run_stderr = await process.communicate()

            return f"""Rust project execution result:

Cargo check (syntax and compiler hints):
{check_stdout.decode()}
{check_stderr.decode()}

Cargo run (execution result):
{run_stdout.decode()}
{run_stderr.decode()}
"""
        else:
            # Assume it's a single file project
            main_file = next((f for f in files if f.endswith('.rs')), None)
            if main_file:
                file_path = os.path.join(temp_dir, main_file)

                # Run rustc for compilation and syntax check
                process = await asyncio.create_subprocess_exec(
                    "rustc", file_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                compile_stdout, compile_stderr = await process.communicate()

                if process.returncode == 0:
                    # Run the compiled executable
                    executable = os.path.join(temp_dir, os.path.splitext(main_file)[0])
                    process = await asyncio.create_subprocess_exec(
                        executable,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    run_stdout, run_stderr = await process.communicate()

                    return f"""Rust file execution result:

Compilation:
{compile_stdout.decode()}
{compile_stderr.decode()}

Execution:
{run_stdout.decode()}
{run_stderr.decode()}
"""
                else:
                    return f"""Rust file compilation failed:

{compile_stdout.decode()}
{compile_stderr.decode()}
"""
            else:
                return "No Rust files found in the project."


from typing import Any

from browser_use import Agent as BrowserAgent
from browser_use import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from langchain_community.chat_models import ChatLiteLLM


class WebContentParser:
    """
    Parser for extracting content from web pages in various formats.

    Provides methods to extract content as markdown, plain text,
    structured data, and take screenshots with scrolling support.
    """

    def __init__(self, browser_wrapper):
        """Initialize the parser with a browser wrapper instance"""
        self.browser = browser_wrapper

    async def to_markdown(self, page=None, selector="main, article, #content, .content, body",
                          include_images=True):
        """
        Convert webpage content to markdown format

        Args:
            page: The page to parse (uses current page if None)
            selector: CSS selector for the content to extract
            include_images: Whether to include image references

        Returns:
            str: Markdown content
        """
        return await self.browser.extract_markdown(page, selector, include_images)

    async def to_text(self, page=None, selector="body"):
        """Extract plain text from webpage"""
        return await self.browser.extract_text(page, selector)

    async def to_structured(self, page=None, config=None):
        """Extract structured data from webpage using selector configuration"""
        return await self.browser.extract_structured_content(page, config)

    async def to_screenshot(self, page=None, full_page=True, path=None,
                            initial_delay=1000, scroll_delay=500, format='png'):
        """
        Take a screenshot with scrolling functionality

        Args:
            page: The page to screenshot
            full_page: Whether to capture the full page
            path: Path to save the screenshot
            initial_delay: Delay in ms before starting screenshot
            scroll_delay: Delay in ms between scrolls
            format: Image format ('png' or 'jpeg')
        """
        return await self.browser.take_scrolling_screenshot(
            page, full_page, path, initial_delay, scroll_delay, format
        )

    async def extract_all(self, page=None, selector="body", include_images=True,
                          screenshot=True, screenshot_path=None):
        """Extract all content types (markdown, text, structured data, screenshot)"""
        result = {
            'markdown': await self.to_markdown(page, selector, include_images),
            'text': await self.to_text(page, selector),
            'structured': await self.to_structured(page)
        }

        if screenshot:
            result['screenshot'] = await self.to_screenshot(
                page, path=screenshot_path, initial_delay=1000
            )

        return result

class BrowserWrapper:
    """
    A wrapper for browser agent functionality that allows seamless interaction with web browsers.

    This class provides a system-agnostic interface to control browsers through the browser_use
    library, supporting both local and remote browser connections.

    Attributes:
        browser: The Browser instance for web automation
        agent: The BrowserAgent instance for intelligent browsing
        is_initialized (bool): Whether the browser has been initialized
        config (Dict): Configuration for the browser
        remote_url (Optional[str]): URL for remote browser connection if applicable
    """

    def __init__(self,
                 llm: Any = None,
                 headless: bool = False,
                 chrome_path: str | None = None,
                 remote_url: str | None = None,
                 api_key: str | None=None,
                 config: dict[str, Any] | None = None):
        """
        Initialize the browser wrapper.

        Args:
            llm: Language model to use for the browser agent
            headless: Whether to run the browser in headless mode
            chrome_path: Path to local Chrome executable
            remote_url: URL for remote browser connection (wss or cdp)
            config: Additional browser configuration
        """
        self.is_initialized = False
        self.agent = None
        self.browser = None
        self.context = None
        import os

        from pydantic import SecretStr
        def pars(x):
            return x.split('/')[-1] if '/' in x else x
        if llm is None:
            llm = 'google/gemini-2.0-flash-exp'
        if not isinstance(llm, str):
            llm = llm
        elif 'deepseek' in llm:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(base_url='https://api.deepseek.com/v1', model=pars(llm), api_key=SecretStr(api_key or os.getenv('DEEPSEEK_API_KEY')))
        elif 'google' in llm:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model=pars(llm), api_key=SecretStr(api_key or os.getenv('GEMINI_API_KEY')))
        elif 'claude' in llm:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model_name=pars(llm),
                temperature=0.0,
                timeout=400,  # Increase for complex tasks
                api_key=SecretStr(api_key or os.getenv('ANTHROPIC_API_KEY')))
        elif isinstance(llm, str):
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=pars(llm),
                temperature=0.0,api_key=SecretStr(api_key or os.getenv('OPENAI_API_KEY'))
            )



        self.llm = ChatLiteLLM(model=llm) if isinstance(llm,str) else llm
        self.parser = None

        browser_config = {
            'headless': headless,
            'disable_security': True
        }

        if config:
            browser_config.update(config)

        self.config = browser_config

        # Set up remote connection if specified
        if remote_url:
            if remote_url.startswith('wss://'):
                self.config['wss_url'] = remote_url
            elif remote_url.startswith('http'):
                self.config['cdp_url'] = remote_url
            self.remote_url = remote_url
        else:
            self.remote_url = None

        # Set up local Chrome path if specified
        if not headless and remote_url is None and chrome_path is None:
            import os
            import platform

            def get_chrome_path():
                """
                Returns the correct path to the Chrome executable based on the OS.
                If Chrome is not found, returns None.
                """
                chrome_paths = {
                    "Darwin": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",  # macOS
                    "Windows": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",  # Windows
                    "Linux": "/usr/bin/google-chrome"  # Linux
                }

                system = platform.system()
                chrome_path_ = chrome_paths.get(system)

                if chrome_path_ and os.path.isfile(chrome_path_):
                    return chrome_path_

                return None

            chrome_path = get_chrome_path()
        if chrome_path:
            self.config['chrome_instance_path'] = chrome_path


    async def initialize(self):
        """Initialize the browser and context"""
        if self.is_initialized:
            return

        try:
            # Create browser instance
            self.browser = Browser(
                config=BrowserConfig(**self.config)
            )

            # Create context configuration with better settings for scraping
            context_config = BrowserContextConfig(
                wait_for_network_idle_page_load_time=3.0,
                highlight_elements=True,
                viewport_expansion=500,
                wait_between_actions=0.5  # Add a small delay between actions
            )

            # Initialize context
            self.context = await self.browser.new_context(config=context_config)

            # Create an initial page
            browser_state = self.context
            if not browser_state or not browser_state.tabs:
                # If no tabs exist, create a new page
                await self.browser.get_playwright_browser()
                browser_context = await self.context.get_playwright_context()
                self.page = await browser_context.new_page()
            else:
                # Use the existing active tab
                self.page = await self.context.get_current_page()

            self.is_initialized = True

        except Exception as e:
            # Clean up resources in case of initialization error
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            raise Exception(f"Failed to initialize browser: {str(e)}")

    async def create_agent(self, task: str, initial_actions=None):
        """Create a browser agent with the specified task"""
        #if not self.is_initialized:
        #    await self.initialize()

        self.agent = BrowserAgent(
            task=task,
            llm=self.llm,
            #browser_context=self.context,
            initial_actions=initial_actions,
            #browser=self.browser,
        )
        return self.agent

    async def run(self, task: str):
        """Run the browser agent with the specified task"""
        agent = await self.create_agent(task)
        result = await agent.run()
        return result

    async def navigate(self, url: str):
        """Navigate to a URL"""
        if not self.is_initialized:
            await self.initialize()

        # Get the current active page or create a new one if needed
        try:
            page = await self.context.get_current_page()
            if not page:
                browser_context = await self.context.get_playwright_context()
                page = await browser_context.new_page()

            # Navigate to the URL
            await page.goto(url)
            self.page = page
            return page
        except Exception as e:
            raise Exception(f"Failed to navigate to {url}: {str(e)}")

    async def get_tabs(self):
        """Get all open tabs/pages"""
        if not self.is_initialized:
            await self.initialize()

        browser_state = await self.context.get_state()
        return browser_state.tabs if browser_state else []

    async def switch_to_tab(self, tab_index: int):
        """Switch to a specific tab by index"""
        if not self.is_initialized:
            await self.initialize()

        browser_state = await self.context.get_state()
        if not browser_state or not browser_state.tabs or tab_index >= len(browser_state.tabs):
            raise ValueError(f"Tab index {tab_index} is out of range")

        tab_id = browser_state.tabs[tab_index].id
        await self.context.switch_to_tab(tab_id)
        self.page = await self.context.get_current_page()
        return self.page

    async def create_new_tab(self):
        """Create a new tab/page"""
        if not self.is_initialized:
            await self.initialize()

        browser_context = await self.context.get_playwright_context()
        new_page = await browser_context.new_page()
        self.page = new_page
        return new_page

    async def close_current_tab(self):
        """Close the current tab/page"""
        if not self.is_initialized:
            return

        page = await self.context.get_current_page()
        if page:
            await page.close()

        # Update the current page reference
        browser_state = await self.context.get_state()
        if browser_state and browser_state.tabs:
            await self.switch_to_tab(0)

    async def execute_js(self, code: str, page=None):
        """Execute JavaScript code in the browser context"""
        if not self.is_initialized:
            await self.initialize()

        if page is None:
            pages = await self.context.pages()
            if not pages:
                page = await self.context.new_page()
            else:
                page = pages[0]

        result = await page.evaluate(code)
        return result

    async def save_context(self):
        """Save browser context state"""
        if not self.is_initialized:
            return None

        return await self.browser.export_context(self.context)

    async def restore_context(self, context_data):
        """Restore browser context from saved state"""
        if not self.is_initialized:
            await self.initialize()

        await self.browser.import_context(context_data)

    async def close(self):
        """Close the browser"""
        if self.is_initialized and self.browser:
            await self.browser.close()
            self.is_initialized = False

    # Add these methods to the BrowserWrapper class

    def get_parser(self):
        """Get a content parser for the browser"""
        if self.parser is None:
            self.parser = WebContentParser(self)
        return self.parser

    async def extract_markdown(self, page=None, selector="body", include_images=True):
        """
        Extract content from a webpage and convert it to markdown.
        """
        if not self.is_initialized:
            await self.initialize()

        if page is None:
            pages = await self.context.pages()
            if not pages:
                page = await self.context.new_page()
            else:
                page = pages[0]

        # JavaScript to convert HTML to markdown
        script = """
        (selector, includeImages) => {
            const element = document.querySelector(selector);
            if (!element) return '';

            // Simple HTML to Markdown conversion function
            const htmlToMarkdown = (node) => {
                let result = '';

                // Process text nodes
                if (node.nodeType === Node.TEXT_NODE) {
                    return node.textContent;
                }

                // Process element nodes
                if (node.nodeType === Node.ELEMENT_NODE) {
                    const tagName = node.tagName.toLowerCase();

                    // Process by tag type
                    switch(tagName) {
                        case 'h1': return '# ' + getInnerText(node) + '\\n\\n';
                        case 'h2': return '## ' + getInnerText(node) + '\\n\\n';
                        case 'h3': return '### ' + getInnerText(node) + '\\n\\n';
                        case 'h4': return '#### ' + getInnerText(node) + '\\n\\n';
                        case 'h5': return '##### ' + getInnerText(node) + '\\n\\n';
                        case 'h6': return '###### ' + getInnerText(node) + '\\n\\n';
                        case 'p': return getInnerText(node) + '\\n\\n';
                        case 'br': return '\\n';
                        case 'hr': return '---\\n\\n';
                        case 'b':
                        case 'strong': return '**' + getInnerText(node) + '**';
                        case 'i':
                        case 'em': return '*' + getInnerText(node) + '*';
                        case 'a': {
                            const href = node.getAttribute('href');
                            return '[' + getInnerText(node) + '](' + href + ')';
                        }
                        case 'img': {
                            if (!includeImages) return '';
                            const src = node.getAttribute('src');
                            const alt = node.getAttribute('alt') || 'image';
                            return '![' + alt + '](' + src + ')\\n\\n';
                        }
                        case 'code':
                        case 'pre': return '`' + getInnerText(node) + '`';
                        case 'ul': {
                            let listResult = '\\n';
                            Array.from(node.children).forEach(li => {
                                if (li.tagName.toLowerCase() === 'li') {
                                    listResult += '- ' + getInnerText(li) + '\\n';
                                }
                            });
                            return listResult + '\\n';
                        }
                        case 'ol': {
                            let listResult = '\\n';
                            Array.from(node.children).forEach((li, index) => {
                                if (li.tagName.toLowerCase() === 'li') {
                                    listResult += (index + 1) + '. ' + getInnerText(li) + '\\n';
                                }
                            });
                            return listResult + '\\n';
                        }
                        case 'blockquote': return '> ' + getInnerText(node) + '\\n\\n';
                        default: {
                            // Process child nodes for other elements
                            for (const child of node.childNodes) {
                                result += htmlToMarkdown(child);
                            }
                            return result;
                        }
                    }
                }

                return '';
            };

            // Helper function to get inner text with special handling
            const getInnerText = (node) => {
                let text = '';
                for (const child of node.childNodes) {
                    text += htmlToMarkdown(child);
                }
                return text;
            };

            return htmlToMarkdown(element);
        }
        """

        try:
            # Try to convert to markdown using our script
            markdown = await page.evaluate(script, selector, include_images)

            # Add a title if we have one
            title = await page.title()
            if title and not markdown.startswith("# "):
                markdown = f"# {title}\n\n{markdown}"

            return markdown
        except Exception:
            # Fallback to basic extraction if script fails
            content = await self.extract_text(page, selector)
            title = await page.title()
            return f"# {title}\n\n{content}"

    async def take_scrolling_screenshot(self, page=None, full_page=True, path=None,
                                        initial_delay=1000, scroll_delay=500, format='png'):
        """
        Take a screenshot with scrolling functionality and delay.
        """
        if not self.is_initialized:
            await self.initialize()

        if page is None:
            pages = await self.context.pages()
            if not pages:
                page = await self.context.new_page()
            else:
                page = pages[0]

        # Wait for the initial delay to let content load
        if initial_delay > 0:
            await page.wait_for_timeout(initial_delay)

        if full_page and scroll_delay > 0:
            # Get page dimensions
            dimensions = await page.evaluate("""
                () => {
                    return {
                        width: document.documentElement.scrollWidth,
                        height: document.documentElement.scrollHeight,
                        windowHeight: window.innerHeight
                    }
                }
            """)

            # Scroll down the page gradually to trigger lazy loading
            current_position = 0
            while current_position < dimensions['height']:
                await page.evaluate(f"window.scrollTo(0, {current_position})")
                await page.wait_for_timeout(scroll_delay)
                current_position += dimensions['windowHeight'] // 2  # Scroll by half viewport

        # Reset scroll position to top
        await page.evaluate("window.scrollTo(0, 0)")

        # Take the screenshot
        screenshot_params = {
            'full_page': full_page,
            'type': format
        }

        if path:
            screenshot_params['path'] = path

        return await page.screenshot(**screenshot_params)

    async def extract_text(self, page=None, selector="body"):
        """
        Extract plain text from a webpage.
        """
        if not self.is_initialized:
            await self.initialize()

        if page is None:
            pages = await self.context.pages()
            if not pages:
                page = await self.context.new_page()
            else:
                page = pages[0]

        text = await page.evaluate("""
            (selector) => {
                const element = document.querySelector(selector);
                return element ? element.innerText : '';
            }
        """, selector)

        return text

    async def extract_structured_content(self, page=None, config=None):
        """
        Extract structured content from a webpage based on a configuration.
        """
        if not self.is_initialized:
            await self.initialize()

        if page is None:
            pages = await self.context.pages()
            if not pages:
                page = await self.context.new_page()
            else:
                page = pages[0]

        if not config:
            # Default configuration if none provided
            config = {
                'title': 'h1',
                'headings': 'h2, h3, h4, h5, h6',
                'paragraphs': 'p',
                'links': 'a',
                'images': 'img'
            }

        result = {}

        for key, selector in config.items():
            if key == 'links':
                # Extract links with their href and text
                result[key] = await page.evaluate("""
                    (selector) => {
                        return Array.from(document.querySelectorAll(selector))
                            .map(el => ({
                                text: el.innerText.trim(),
                                href: el.href
                            }))
                            .filter(item => item.text && item.href);
                    }
                """, selector)
            elif key == 'images':
                # Extract images with their src and alt
                result[key] = await page.evaluate("""
                    (selector) => {
                        return Array.from(document.querySelectorAll(selector))
                            .map(el => ({
                                src: el.src,
                                alt: el.alt || ''
                            }))
                            .filter(item => item.src);
                    }
                """, selector)
            else:
                # Extract text content for other elements
                result[key] = await page.evaluate("""
                    (selector) => {
                        return Array.from(document.querySelectorAll(selector))
                            .map(el => el.innerText.trim())
                            .filter(text => text);
                    }
                """, selector)

        return result


class Pipeline:
    """
        A pipeline for executing AI agent-driven tasks with interactive code execution and variable management.

        The Pipeline class provides a structured environment for AI agents to:
        1. Execute code in a controlled environment
        2. Manage and track variables
        3. Update methods dynamically
        4. Save and load session states
        5. Generate detailed variable descriptions

        Attributes:
            agent: The AI agent instance used for task execution
            task (str): The task to be performed
            mas_iter (int): Maximum number of iterations allowed (default: 12)
            variables (Dict[str, Any]): Dictionary of variables available to the pipeline
            top_n (Optional[int]): Limit variable descriptions to top N most used
            execution_history (List[ExecutionRecord]): History of executed code and results
            session_name (Optional[str]): Name of the current session if saved
            ipython: IPython or MockIPython instance for code execution

        Example:
            >>> agent = get_free_agent("demo", "anthropic/claude-3-haiku-20240307")
            >>> pipeline = Pipeline(
            ...     agent=agent,
            ...     variables={"n": 10}
            ... )
            >>> result = pipeline.run("Calculate fibonacci sequence")
            >>> print(result.result)

        Notes:
            - The pipeline uses either IPython if available or a MockIPython implementation
            - Variables can be provided as either a dictionary or list
            - Session state can be saved and loaded
            - Method updates are handled through a structured BaseModel approach
        """
    def __init__(
        self,
        agent: Any,
        verbose: bool=False,
        max_iter: int= 12,
        variables: dict[str, Any] | list[Any] | None = None,
        top_n: bool | None = None,
        restore: bool | None = None,
        max_think_after_think = None,
        print_f=None,
        web_js=False,
        timeout_timer=25,
        v_agent=None,
        web_llm=None,
    ):
        """
        Initialize the Pipeline.

        Args:
            agent: AI agent instance to use for task execution
            verbose: print internal results
            max_iter: Maximum number of iterations (default: 12)
            variables: Dictionary or list of variables to make available
            top_n: Limit variable descriptions to top N most used
            web_js: if the agent is allow to use the web
        """

        self.timeout_timer = timeout_timer
        self.top_n = top_n
        self.max_iter = max_iter
        self.max_think_after_think = max_think_after_think or max_iter // 2
        self.agent = agent
        self.v_agent = v_agent or agent
        # self.agent.verbose = verbose
        self.task = None
        self.web_js = web_js
        self.print_f = print_f
        self.verbose_output = EnhancedVerboseOutput(verbose=verbose, print_func=self.print_f)
        self.variables = self._process_variables(variables or {})
        self.variables['auto_install'] = auto_install
        self.execution_history = []
        self.session_name = None

        self.browser_session: BrowserWrapper | None = BrowserWrapper(llm=web_llm or agent.amd.model)
        self.js_history: list[JSExecutionRecord] = []

        self._session_dir = Path(get_app().appdata) / 'ChatSession' / agent.amd.name
        self.ipython = MockIPython(self._session_dir, auto_remove=False)
        self.chat_session = ChatSession(get_app().get_mod("isaa").get_memory(), space_name=f"ChatSession/{agent.amd.name}/Pipeline.session", max_length=max_iter)
        self.process_memory = ChatSession(get_app().get_mod("isaa").get_memory(), space_name=f"ChatSession/{agent.amd.name}/Process.session", max_length=max_iter)

        # Initialize interpreter with variables
        self.init_keys = list(self.ipython.user_ns.keys()).copy()
        if self.web_js:
            self.variables['web_actions'] = self.browser_session.run
            self.variables['browser_session'] = self.browser_session
        self.ipython.user_ns.update(self.variables)

        self.restore_var = restore

        if restore:
            self.restore()

    def on_exit(self):
        self.chat_session.on_exit()
        self.process_memory.on_exit()
        self.save_session(f"Pipeline_Session_{self.agent.amd.name}")

    def restore(self):
        self.load_session(f"Pipeline_Session_{self.agent.amd.name}")

    def save_session(self, name: str):
        """Save current session"""
        self.session_name = name
        self.ipython.save_session(name)

    def load_session(self, name: str):
        """Load saved session"""
        self.ipython.load_session(name)
        self.variables.update(self.ipython.user_ns)


    def show_graph_html(self, output_file=None, get_output_html=False, get_output_net=False):

        if output_file is None:
            chat_graph = self.ipython._session_dir / 'chat_graph.html'
            process_graph = self.ipython._session_dir / 'process_graph.html'
            output_file = str(chat_graph.absolute())
            p_output_file = str(process_graph.absolute())
        else:
            output_file = output_file + '_chat_graph.html'
            p_output_file = output_file + '_process_graph.html'

        return (self.chat_session.mem.memories.get(
            self.chat_session.mem._sanitize_name(
                self.chat_session.space_name)).vis(output_file=output_file,
        get_output_html=get_output_html, get_output_net=get_output_net)  ,
                self.process_memory.mem.memories.get(
            self.process_memory.mem._sanitize_name(
                self.process_memory.space_name)).vis(output_file=p_output_file,
        get_output_html=get_output_html, get_output_net=get_output_net))

    @staticmethod
    def _process_variables(variables: dict[str, Any] | list[Any]) -> dict[str, Any]:
        """
        Process variables to generate meaningful names, using actual variable names where possible.
        Instances get lowercase names based on their class names.

        Args:
            variables: Dictionary of variables or list of variables to process

        Returns:
            Dict[str, Any]: Processed variables with meaningful names
        """
        if isinstance(variables, dict):
            return variables

        processed = {}
        name_counts = defaultdict(int)

        # Get caller's frame to find variable names
        caller_frame = currentframe().f_back
        caller_locals = {**caller_frame.f_locals, **caller_frame.f_globals}

        def find_var_name(obj: Any) -> str:
            # Find original variable name if exists
            var_names = [name for name, val in caller_locals.items()
                         if val is obj and not name.startswith('_')]
            if var_names:
                return var_names[0]

            # Special handling for functions
            if isfunction(obj) or isclass(obj):
                return obj.__name__
            # Handle instances
            elif hasattr(obj, '__class__'):
                base_name = obj.__class__.__name__.lower()  # Lowercase for instances
                count = name_counts[base_name]
                name_counts[base_name] += 1
                return f"{base_name}_{count + 1}" if count > 0 else base_name

            return type(obj).__name__

        # Process each variable
        for var in variables:
            name = find_var_name(var)
            while name in processed:
                if name.rpartition('_')[0]:
                    base, _, num = name.rpartition('_')
                    try:
                        num = int(num) + 1
                        name = f"{base}_{num}"
                    except ValueError:
                        name = f"{name}"
                else:
                    name = f"{name}"

            processed[name] = var

        return processed

    def _generate_variable_descriptions(
        self,
        top_n: int | None = None
    ) -> str:
        """
        Generate detailed descriptions of variables, showing args, kwargs, docstrings, and return values.

        Args:
            top_n: Optional limit to show only top N variables

        Returns:
            str: Formatted variable descriptions in Markdown
        """
        if top_n is None:
            top_n = self.top_n

        def format_value_preview(var: Any) -> str:
            """Format preview of variable contents"""
            try:
                if isinstance(var, int | float | bool | str):
                    return f"`{repr(var)}`"
                elif isinstance(var, list | tuple | set):
                    preview = str(list(var)[:3])[:-1] + ", ...]"
                    return f"{len(var)} items: {preview}"
                elif isinstance(var, dict):
                    preview_items = [f"{repr(k)}: {repr(v)}" for k, v in list(var.items())[:3]]
                    return f"{len(var)} pairs: {{{', '.join(preview_items)}, ...}}"
                return f"<{type(var).__name__}>"
            except:
                return "<error getting value>"

        def get_instance_state(var: Any) -> dict[str, Any]:
            """Get current instance state"""
            state = {}
            if hasattr(var, '__dict__'):
                for name, value in var.__dict__.items():
                    if not name.startswith('_') and not callable(value):
                        state[name] = format_value_preview(value)
            return state

        # Process variables
        variables = self.variables.items()
        if top_n:
            variables = list(variables)[:top_n]

        descriptions = []
        for name, var in variables:
            if name in ["PYTHON_EXEC", "__name__", "__builtins__", "__path__", "asyncio"]:
                continue

            desc_parts = [f"### {name}"]

            # Handle different types
            if isinstance(var, type):  # Class
                desc_parts.append(f"**Type:** `class '{var.__name__}'`")
                if var.__doc__:
                    desc_parts.append(f"**Documentation:**\n{var.__doc__.strip()}")

                # Show methods
                methods = []
                for attr_name, attr in var.__dict__.items():
                    if (not attr_name.startswith('_') or attr_name == "__init__") and (isfunction(attr) or ismethod(attr)):
                        try:
                            sig = signature(attr)
                            is_a = asyncio.iscoroutinefunction(var)
                            methods.append(f"- `{attr_name}{sig}` Async: `{is_a}")
                            if attr.__doc__:
                                r = attr.__doc__.split('\n')[0]
                                methods.append(f"  {r}")
                        except:
                            methods.append(f"- `{attr_name}()`")
                if methods:
                    desc_parts.append("**Methods:**\n" + "\n".join(methods))

            elif isfunction(var) or ismethod(var):  # Function
                try:
                    sig = signature(var)
                    desc_parts.append(f"**Signature:** `{var.__name__}{sig}`")
                    is_a = asyncio.iscoroutinefunction(var)
                    desc_parts.append(f"**IS Async:** `{is_a}`")
                    if var.__doc__:
                        desc_parts.append(f"**Documentation:**\n{var.__doc__.strip()}")
                    ret_anno = sig.return_annotation
                    if ret_anno != Signature.empty:
                        desc_parts.append(f"**Returns:** `{ret_anno}`")
                except:
                    desc_parts.append(f"**Function:** `{var.__name__}()`")

            elif isinstance(var, BaseModel):  # Pydantic model
                desc_parts.append(f"**Type:** Pydantic model '{var.__class__.__name__}'")
                fields = []
                for field_name, field in var.model_fields.items():
                    value = getattr(var, field_name, None)
                    fields.append(f"- `{field_name}: {field.annotation.__name__}` = {repr(value)}")
                if fields:
                    desc_parts.append("**Fields:**\n" + "\n".join(fields))

            else:  # Instance
                class_type = var.__class__
                desc_parts.append(f"**Type:** `{class_type.__module__}.{class_type.__name__}`")

                # Instance initialization details
                try:
                    init = class_type.__init__
                    sig = signature(init)
                    params = list(sig.parameters.items())[1:]  # Skip self
                    if params:
                        args = []
                        for name, param in params:
                            if param.default == param.empty:
                                args.append(name)
                            else:
                                args.append(f"{name}={param.default}")
                        desc_parts.append(f"**Init Args:** `{', '.join(args)}`")
                except:
                    pass

                # Instance state
                state = get_instance_state(var)
                if state:
                    desc_parts.append("**Current instance State:**")
                    for attr_name, attr_value in state.items():
                        desc_parts.append(f"- `{attr_name}` = {attr_value}")

                # Documentation
                doc = getdoc(var) or getdoc(class_type)
                if doc:
                    desc_parts.append(f"**Documentation:**\n{doc.strip()}")

            descriptions.append("\n".join(desc_parts))

        return "\n\n".join(descriptions)

    async def _execute_code(self, code: str, context:dict) -> ExecutionRecord:
        """Execute code and track results"""
        lang = context.get('lang', 'py')
        try:

            if'py' in lang:

                return await self._execute_py(code)

            elif self.web_js and 'js' in lang:
                return await self._execute_js(code, context)

        except Exception as e:
            record = ExecutionRecord(code=code, result=None, error=str(e))
            self.execution_history.append(record)
            return record
        record = ExecutionRecord(code=code, result=None, error=f"Invalid lang {lang} valid is, {'js' if self.web_js else 'py'}]")
        self.execution_history.append(record)
        return record

    async def _execute_py(self, code) -> ExecutionRecord:
        show = True #len(code) > 450 and code.count('while') > 1 and code.count('print') >= 1
        result = await self.ipython.run_cell(code, show)

        all_keys = list(self.ipython.user_ns.keys())

        new_keys = [key for key in all_keys if key not in self.init_keys]
        # Update pipeline variables from IPython namespace

        for var_name in new_keys:
            if var_name.startswith('_'):
                continue
            self.variables[var_name] = self.ipython.user_ns[var_name]

        record = ExecutionRecord(code=code, result=result, error=None)
        self.execution_history.append(record)
        return record

    async def _execute_js(self, code: str, context: dict) -> ExecutionRecord:
        """Execute JavaScript code in browser context"""

        if '<script>' in code:
            code = code.split('<script>')[1]
        if '</script>' in code:
            code = code.split('</script>')[0]
        def _format_error_markdown(error: str) -> str:
            """Format error as Markdown"""
            return f"""
# Execution Error
{error}
"""

        def _format_result_markdown(result_: dict) -> str:
            """Format execution result as Markdown"""

            def _clean_html_content(html: str) -> str:
                """Clean HTML content and convert to Markdown-like format"""
                soup = BeautifulSoup(html, 'html.parser')

                # Remove scripts and styles
                for script in soup(["script", "style"]):
                    script.decompose()

                # Extract text
                text = soup.get_text()

                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)

                # Add Markdown formatting
                text = re.sub(r'^(.+)$', r'> \1', text, flags=re.MULTILINE)

                return text

            md_parts = []

            # Add title
            md_parts.append("# Page Analysis Results\n")

            # Format JavaScript result
            if result_.get('js_result'):
                md_parts.append("## JavaScript Execution Result")
                md_parts.append("```")
                md_parts.append(str(result_['js_result']))
                md_parts.append("```\n")

            # Format page state
            if 'page_state' in result_:
                md_parts.append("## Page Information")
                md_parts.append(f"- **URL**: {result_['page_state']['url']}")
                md_parts.append(f"- **Title**: {result_['page_state']['title']}\n")

                # Clean and format content
                if 'content' in result_['page_state']:
                    content = _clean_html_content(result_['page_state']['content'])
                    if content:
                        md_parts.append("### Page Content")
                        md_parts.append(content + "\n")

            # Format extracted data
            if result_.get('extracted_data'):
                md_parts.append("## Extracted Data")
                for key, value in result_['extracted_data'].items():
                    if value:
                        md_parts.append(f"### {key.replace('_', ' ').title()}")
                        if isinstance(value, list):
                            for item in value:
                                md_parts.append(f"- {item}")
                        else:
                            md_parts.append(str(value))
                        md_parts.append("")

            return "\n".join(md_parts)

        try:
            # Prepare execution context
            url = context.get('url')
            page = None
            result = None
            page_state = {}

            extracted_data = None
            if url:
                page = await self.browser_session.navigate(url)
                parser = self.browser_session.get_parser()
                markdown = await parser.to_markdown(page)

                if 'patterns' in context:
                    extracted_data = await parser.to_structured(page, context['patterns'])

                page_state = {
                    'url': page.url,
                    'title': await page.title(),
                    'content': markdown,
                }

            if code:
                result = await self.browser_session.execute_js(code, page)

                if isinstance(result, dict) and 'success' in result:
                    if not result['success']:
                        raise Exception(f"JavaScript Error: {result.get('error')}\nStack: {result.get('stack')}")
                    result = result.get('result')

            # Capture page state after execution


            # Extract data using patterns if specified

            # Create execution record
            record = JSExecutionRecord(
                code=code,
                result=result,
                page_state=page_state,
                extracted_data=extracted_data
            )

            self.js_history.append(record)

            # Convert to standard ExecutionRecord for pipeline
            return ExecutionRecord(
                code=code,
                result=_format_result_markdown({
                    'js_result': result,
                    'page_state': page_state,
                    'extracted_data': extracted_data
                }),
                error=None
            )

        except Exception as e:
            error_md = _format_error_markdown(str(e))
            return ExecutionRecord(code=code, result=None, error=error_md)


    def __str__(self):
        """String representation of pipeline session"""
        return str(self.ipython)

    async def _process_think_result(self, think_result: ThinkResult, task:str) -> tuple[ThinkState,  ExecutionRecord | str | None]:
        """Process the result of agent thinking"""
        if think_result.action == 'brake':
            return ThinkState.BRAKE, think_result.content

        elif think_result.action == 'update':
            if think_result.context.get('object_name') is None:
                return ThinkState.ACTION, "no object_name specified in context!"
            if think_result.context.get('file') is not None:
                self.ipython.user_ns['__file__'] = think_result.context.get('file')
            result = await self.verbose_output.process(think_result.action,
                                                       self.ipython.modify_code(code=think_result.content,
                                                    object_name=think_result.context.get('object_name'),))
            return ThinkState.PROCESSING, result

        elif think_result.action == 'code':
            if think_result.context.get('file') is not None:
                self.ipython.user_ns['__file__'] = think_result.context.get('file')
            result = await self._execute_code(think_result.content, think_result.context)
            return ThinkState.PROCESSING, result

        elif think_result.action == 'done':
            return ThinkState.DONE, think_result.content

        elif think_result.action == 'infos':
            infos = await self.chat_session.get_reference(think_result.content, to_str=True)
            return ThinkState.ACTION, infos

        elif think_result.action == 'guide':
            details = await self.process_memory.get_reference(think_result.content, to_str=True)
            plan = await self.agent.a_run(f"""You are an AI guidance system designed to help determine the next step in a task and provide instructions on how to proceed. Your role is to analyze the given information and offer clear, actionable guidance for the next steps.

First, carefully read and understand the main task:
<main_task>
{task}
</main_task>

Next, review the last thought of the agent, if available:
<last_thought>
{think_result.content}
{think_result.context}
</last_thought>

Then, examine the processing history, if provided:
<processing_history>
{details}
</processing_history>

To determine the next step and provide guidance, follow these instructions:

1. Analyze the main task, breaking it down into smaller, manageable steps if necessary.
2. Consider the last thought and processing history to understand the current progress and context.
3. Identify any gaps, challenges, or areas that need further attention.
4. Determine the most logical and efficient next step to move the task forward.
5. Provide clear, concise instructions on how to complete this next step.

When formulating your response, follow this structure:

1. Begin with a brief summary of the current situation, referencing the main task and any relevant information from the last thought or processing history.
2. Clearly state the next step that should be taken.
3. Provide detailed instructions on how to complete this step, including any specific techniques, methods, or considerations to keep in mind.
4. If applicable, mention any potential challenges or pitfalls to be aware of during this step.
5. Conclude with a brief statement on how this step contributes to the overall progress of the main task.

Format your response using the following sections:
<summary>
(Include your summary of the current situation here)
</summary>

<next_step>
(State the next step to be taken here)
</next_step>

<instructions>
(Provide detailed instructions for completing the next step here)
</instructions>

<challenges>
(If applicable, mention potential challenges or pitfalls here)
</challenges>

<conclusion>
(Briefly state how this step contributes to overall progress)
</conclusion>

Remember to be clear, concise, and specific in your guidance. Avoid vague or ambiguous instructions, and provide concrete examples or explanations where necessary.""",persist_history=False, strategy_override="direct_llm")
            return ThinkState.ACTION, plan

        return ThinkState.ACTION, None

    async def execute(self, code:str):
        return str(await self._execute_code(code))

    def clear(self):
        self.chat_session.history = []
        self.process_memory.history = []
        self.execution_history = []
        self.variables = {}
        self.ipython.reset()
        self.js_history = []

    async def get_process_hint(self, task):
        return await self.process_memory.get_reference(task, to_str=True), await self.chat_session.get_reference(task, to_str=True)

    def show_vars(self):
        return self.verbose_output.log_state("VARS", self.variables, override=True)

    def set_file(self, full_file_path_and_name):
        if not os.path.exists(full_file_path_and_name):
            print("Invalid file")
            return
        self.ipython.user_ns["__file__"] = full_file_path_and_name

    async def run(self, task, do_continue=False) -> PipelineResult:
        """Run the pipeline with separated thinking and processing phases"""
        state = ThinkState.ACTION
        result = None
        original_task = task
        if not do_continue:
            task = await self.agent.a_run(f"""You are an AI assistant tasked with refactoring a user-provided task description into a more structured format with context learning and examples. Your goal is to create a comprehensive and well-organized task description that incorporates model flows and potential code fixes.

First, I will provide you with a task description and some example tasks. Please read them carefully:

<existing_globals>
{self._generate_variable_descriptions()}
</existing_globals>

<example_tasks>
Task: Create a simple analysis of a list of numbers
- Generate a list of 100 random numbers between 1-1000
- Calculate the mean, median, and standard deviation
- Create a histogram of the distribution
- Print all results and display the plot

Task: Create a reinforcement learning (RL) agent to play a simple game
- Set up an OpenAI Gym environment (e.g., CartPole)
- Implement a Q-learning or Deep Q-Network (DQN) agent
- Train the model and optimize hyperparameters
- Visualize learning progress with reward graphs
- Save and reload trained models for inference
- Provide an option to let the trained agent play in real time

Task: Perform edge detection on an image
- Load an image from a URL or local file
- Convert the image to grayscale
- Apply Gaussian blur to reduce noise
- Use Canny edge detection to extract edges
- Display the original and processed images side by side
- Save the output image

Task: Build a basic sentiment analysis system
- Load a dataset of movie reviews (you can use a small sample)
- Preprocess the text (remove punctuation, lowercase, etc.)
- Create a TF-IDF vectorizer
- Split data into training and testing sets
- Train a classifier (e.g., Naive Bayes or LogisticRegression)
- Evaluate performance with accuracy, precision, recall
- Create a confusion matrix visualization
- Make predictions on new sample texts
</example_tasks>

Now, please refactor the given task description using the following guidelines:

1. Analyze the task description and identify the main components and objectives.

2. Structure the refactored task in a similar format to the example tasks, including:
   - A clear title that summarizes the task
   - A difficulty level (Easy, Intermediate, Hard, or Super Hard)
   - A brief introduction to the task's context and purpose
   - A code block containing step-by-step instructions
   - A list of required skills, libraries, or technologies

3. Incorporate model flows by breaking down the task into logical steps and explaining the process flow.

4. Include potential code fixes or common pitfalls that users might encounter while working on the task.

5. Add context learning elements by providing brief explanations or resources for key concepts related to the task.

6. Ensure that the refactored task is comprehensive and can stand alone as a learning exercise.

Please provide your refactored task description within <refactored_task> tags. Use appropriate subheadings and formatting to make the description clear and easy to read.

Additional tips:
- Mention any prerequisites or assumed knowledge
- Suggest potential extensions or variations of the task for further learning

Remember to maintain the original intent and complexity of the task while improving its structure and clarity. the task is: {task}""", persist_history=False, strategy_override="direct_llm")
            if '<refactored_task>' in task:
                task = task.split('<refactored_task>')[1]
            if '</refactored_task>' in task:
                task = task.split('</refactored_task>')[0]
        code_follow_up_prompt = f"""
You are an AI assistant responsible for evaluating task completion and providing feedback on the execution process. Your goal is to determine if a given task has been completed based on the execution result, and to offer insights for future improvements.

You will be provided with two inputs:
<task_description>
{original_task}
{f'<refactored_task_description_from_ai>{task}</refactored_task_description_from_ai>' if not do_continue else ''}
</task_description>

<code>
#CODE#
</code>

<execution_result>
#EXECUTION_RESULT#
</execution_result>

First, carefully analyze the task description and the execution result. Determine whether the task has been completed successfully based on the information provided.

If the task is completed:
1. Prepare a brief statement indicating that the task is done.
2. Summarize the output for the user in a clear and concise manner.

If the task is not completed:
1. Prepare a brief statement indicating that the task is not done.
2. Identify the specific aspects of the task that remain incomplete.

Regardless of task completion status, evaluate the procedure and effectiveness of the execution:
1. Analyze the workflow: Describe the steps taken in the execution process.
2. Assess effectiveness: Determine how well the procedure achieved the desired outcome.
3. Identify errors: Pinpoint any mistakes or inefficiencies in the execution.
4. Provide recommendations: Suggest improvements for future task executions.

tip: Enclose mutil line strings property for python eval to function!
tip: Set is_completed True if all requirements are completed from <task_description>.
tip: Help the Agent with your analyses to finalize the <task_description>.
{'tip: Prefer new informations from <execution_result> over <refactored_task_description_from_ai> based of <code>' if not do_continue else ''}
note : for the final result only toke information from the <execution_result>. if the relevant informations is not avalabel try string withe tips in the recommendations. else set is_completed True and return the teh Task failed!
Ensure that your evaluation is thorough, constructive, and provides actionable insights for improving future task executions.
Add guidance based on the the last execution result"""
        code_follow_up_prompt_ = [code_follow_up_prompt]
        initial_prompt = f"""
You are an AI py coding agent specializing in iterative development and code refinement, designed to perform tasks that involve thinking. Your goal is to complete the given task while demonstrating a clear thought process throughout the execution.
SYSTEM STATE:
<current_state>
Iteration: #ITER#
Status: #STATE#
Last EXECUTION: #EXECUTION#
</current_state>

ENVIRONMENT: {'current file :'+self.ipython.user_ns.get("__file__")  if self.ipython.user_ns.get("__file__") is not None else ''}

'''<global_variables>
#LOCALS#
</global_variables>'''

MEMORY:
<process_memory>
#PHINT#
</process_memory>

<chat_memory>
#CHINT#
</chat_memory>

VALIDATION CHECKLIST (Must verify before each action):
1. ✓ Check existing variables in ENVIRONMENT <global_variables>
2. ✓ Verify existing functions and classes
3. ✓ Review current imports
4. ✓ Confirm method signatures
5. ✓ Validate state preservation

WORKFLOW STEPS:
1. Analyze Current State:
   - Reason and use all avalabel context
   - Do not repeat the same errors
   - Review existing implementations
   - Check variable values
   - Verify import statements
   - Document dependencies

2. Plan Change:
   - NO example/simulation/simulate
   - No demo er moc Data no Simulations Allowed or u will die!!
   - Use existing variables and code when possible
   - Prefer updates over rewrites

3. Execute Change:
   - Use appropriate action
   - Maintain existing state
   - Document modifications
   - Verify results

You will use a structure called ThinkResult to organize your thoughts and actions.
For each step of your task, follow this process:

ACTIONS:
1. 'code':
    - MUST check <global_variables> first
    - NEVER create demo functions
    - Include 'reason'
    - lang default 'py'
    - Required: code in content
    - code MUST call a function or display the row variabel / value at the end!
    - Required: {{'context':{{'lang':'py',  'reason': ... }}...}}
    - Optional file key in context example {{'context':{{'lang':'py',  'file': 'main.py' ,  'reason': ... }}...}}
    - py code allows for toplevel await !!! use it !!! like
:file-start:
print("using toplevel await")
await abc()
:file-end:

    - Tip: use comments to reason with in the code
3. 'infos': Request specific details
4. 'guide': Get step clarification use on complex task and ery 5 step for staying on trak!
5. 'brake': Pause for assessment
6. 'done': Summarize changes

CODE CONSTRAINTS:
1. State Preservation:
   - ALL variables ar persist
   - ALL functions remain
   - ALL classes ar maintained

2. Import Management:
   - Check <global_variables> for modules
   - Use absolute imports
   - Document new dependencies

3. Function Handling:
   - NEVER overwrite existing
   - Use update for changes
   - Preserve signatures

4. Variable Scope:
   - Maintain existing scope
   - Check for conflicts
   - Document state changes

EXECUTION RULES:
1. VERIFY before create
2. UPDATE don't replace
3. TEST after each change

Next Action Required:
1. Review current state
2. Check existing code
3. Execute with state preservation

!!CRITICAL!!
- NO demo functions
- NO placeholder functions
- USE existing code
- FOR Implementations prefer writing large production redy code chunks.
- FOR reasoning and validation write small code blocks.
- THE CODE must call something or end the code with an value!
- NO INFINIT LOOPS! none breakable while loops ar not allowed, exception ui (closed by user)
- NO 'python' top level return, only write the variabel or value itself!
- 'code is run using exec! do not use !pip ...'
'- instead use auto_install(package_name, install_method="pip", upgrade=False, quiet=False, version=None, extra_args=None)'
# Example usage first time
│ auto_install('pandas', version='1.3.0')
│ import pandas
│ auto_install('pygame')
│ import pygame
│ auto_install('numpy')
│ import numpy as np
!TIPS!
- '<global_variables> can contain instances and functions you can use in your python' code
- if the function is async you can use top level await
- if their is missing of informations try running code to get the infos
- if you got stuck or need assistance break with a question to the user.
'- run functions from <global_variables> using name(*args, **kwargs) or await name(*args, **kwargs)'
'- <global_variables> ar global accessible!'
'- if an <global_variables> name is lower lists an redy to use instance'
"""
        p_hint, c_hint = await self.get_process_hint(task)
        initial_prompt = initial_prompt.replace('#PHINT#', p_hint)
        initial_prompt = initial_prompt.replace('#CHINT#', c_hint)
        initial_prompt_ = initial_prompt
        iter_i = 0
        iter_p = 0
        iter_tat = 0
        next_infos = ""
        if not do_continue:
            await self.chat_session.add_message({'role': 'user', 'content': task})
        else:
            self.restore()
            await self.chat_session.add_message({'role': 'user', 'content': task})

        if self.web_js and self.browser_session is None:
            self.browser_session = BrowserWrapper(llm=self.agent.amd.modle)

        # await self.verbose_output.log_message('user', task)
        self.verbose_output.log_header(task)
        while state != ThinkState.DONE:
            iter_i += 1
            t0 = time.perf_counter()
            prompt = initial_prompt.replace('#ITER#', f'{iter_i} max {self.max_iter}')
            prompt = prompt.replace('#STATE#', f'{state.name}')
            prompt = prompt.replace('#EXECUTION#', f'{next_infos}')  if next_infos else prompt.replace('Last EXECUTION: #EXECUTION#', '')
            prompt = prompt.replace('#LOCALS#', f'{self._generate_variable_descriptions()}')
            self.verbose_output.log_state(state.name, {})
            self.verbose_output.print(Style.GREY(f"{iter_i}/{self.max_iter}"))
            if state == ThinkState.ACTION:
                iter_tat +=1
                if iter_tat > self.max_think_after_think:
                    state = ThinkState.BRAKE
            else:
                iter_tat = 0

            if state == ThinkState.ACTION:
                # Get agent's thoughts
                think_dicts = await self.verbose_output.process(state.name, self.agent.a_format_class(
                    ThinkResults,
                    prompt,
                    message_context=self.chat_session.get_past_x(self.max_iter*2, last_u=not do_continue).copy()+([self.process_memory.history[-1]] if self.process_memory.history else []) ,
                ))
                think_dicts = think_dicts.get("actions")
                if think_dicts is None:
                    think_dicts = [await self.verbose_output.process(state.name, self.agent.a_format_class(
                        ThinkResult,
                        prompt,
                        message_context=self.chat_session.get_past_x(self.max_iter * 2, last_u=not do_continue).copy() + (
                            [self.process_memory.history[-1]] if self.process_memory.history else []),
                    ))]
                if len(think_dicts) == 1:
                    think_dict = think_dicts[0]
                else:
                    for think_dict in think_dicts[:-1]:
                        if think_dict.get('context') is None:
                            think_dict['context'] = {'context': 'N/A'}
                        if not isinstance(think_dict.get('context'), dict):
                            think_dict['context'] = {'context': think_dict.get('context')}
                        think_result = ThinkResult(**think_dict)
                        await self.chat_session.add_message(
                            {'role': 'assistant', 'content': think_result.content + str(think_result.context)})
                        state, result = await self.verbose_output.process(think_dict.get("action"),
                                                                          self._process_think_result(think_result,
                                                                                                     task=task))
                        if result:
                            await self.chat_session.add_message(
                                {'role': 'system', 'content': 'Evaluation: ' + str(result)})
                            await self.verbose_output.log_message('system', str(result))
                    think_dict = think_dicts[-1]
                self.verbose_output.formatter.print_code_block(think_dict.get("content"), 'python' if think_dict.get("context", {'lang': 'py'}).get('lang') == 'py' else 'javascript')
                if think_dict.get('context') is None:
                    think_dict['context'] = {'context': 'N/A'}
                if not isinstance(think_dict.get('context'), dict):
                    think_dict['context'] = {'context': think_dict.get('context')}
                think_result = ThinkResult(**think_dict)
                state, result = await self.verbose_output.process(think_dict.get("action"), self._process_think_result(think_result, task=task))
                await self.chat_session.add_message({'role': 'assistant', 'content': think_result.content + str(think_result.context)})
                if result:
                    await self.chat_session.add_message({'role': 'system', 'content': 'Evaluation: '+str(result)})
                    await self.verbose_output.log_message('system', str(result))
                    code_follow_up_prompt_[0] = code_follow_up_prompt.replace("#EXECUTION_RESULT#", str(result))
                    if isinstance(result ,ExecutionRecord):
                        code_follow_up_prompt_[0] = code_follow_up_prompt_[0].replace("#CODE#", result.code)
                    else:
                        code_follow_up_prompt_[0] = code_follow_up_prompt_[0].replace("#CODE#", self._generate_variable_descriptions())
                else:
                    code_follow_up_prompt_[0] = code_follow_up_prompt.replace("#EXECUTION_RESULT#", str(think_result))
                    code_follow_up_prompt_[0] = code_follow_up_prompt_[0].replace("#CODE#",
                                                                              self._generate_variable_descriptions())


            elif state == ThinkState.PROCESSING:
                # Get agent's thoughts
                class Next(BaseModel):
                    is_completed: bool
                    recommendations: str
                    errors: str
                    effectiveness: str
                    workflow: str
                    text: str
                # Format the agent's thoughts into a structured response
                _agent = self.v_agent if self.v_agent is not None else self.agent
                next_dict = await self.verbose_output.process(state.name, _agent.a_format_class(
                    Next,
                    code_follow_up_prompt_[0],
                    message_context=self.chat_session.get_past_x(self.max_iter*2, last_u=not do_continue).copy(),
                ))
                next_infos = json.dumps(next_dict)
                await self.verbose_output.log_process_result(next_dict)
                await self.process_memory.add_message({'role': 'assistant', 'content': next_infos.replace('workflow:', 'past-workflow:')})
                iter_p += 1
                code_follow_up_prompt_[0] = code_follow_up_prompt
                if not next_dict.get('is_completed', True):
                    state = ThinkState.ACTION
                    initial_prompt = initial_prompt_.replace('#ITER#',f'#ITER#\nReasoning assist result: {next_dict}')
                    continue
                elif next_dict.get('is_completed', False):
                    result = next_dict.get('text', '')
                    state = ThinkState.DONE
                    continue
                else:
                    result = next_dict.get('text', '')
                    break

            elif state == ThinkState.BRAKE:
                break

            if iter_i < self.max_iter:
                await asyncio.sleep(1)
                # if time.perf_counter() -t0 < self.timeout_timer*2.5:
                #     with Spinner(f"Prevent rate limit posing for {self.timeout_timer}s", symbols='+', time_in_s=self.timeout_timer, count_down=True):
                #         await asyncio.sleep(self.timeout_timer)
            else:
                state = ThinkState.BRAKE
                if isinstance(result, ExecutionRecord):
                    result = result.result
                elif isinstance(result, str):
                    pass
                else:
                    result = "Max iterations"
                break

        self.verbose_output.log_state(state.name, {})

        return PipelineResult(
            variables=self.variables,
            result=result,
            execution_history=self.execution_history,
            message=self.chat_session.get_past_x(iter_i*2, last_u=not do_continue),
        )

    async def run_project(self, task, lang='py', execute_function=None):
        if execute_function is None:
            if lang == 'py':
                execute_function = default_python_execute_function
            elif lang == 'rust':
                execute_function = default_rust_execute_function
            else:
                raise ValueError(f"Unsupported language: {lang}")
        class FileAction(BaseModel):
            action: str
            path: str
            content: str | None = None

        class ProjectThinkResult(BaseModel):
            action: str
            file_actions: list[FileAction]
            reasoning: str

        class ProjectPipelineResult(BaseModel):
            result: str
            execution_history: list[str]
            files: dict[str, str]
        state = ThinkState.ACTION
        result = None
        vfs = VirtualFileSystem(self._session_dir / f"project_{lang}")

        project_prompt = f"""
    You are an AI coding agent specializing in {lang} project development. Your task is to create, modify, and manage files within a project structure to complete the given task. Use the VirtualFileSystem to interact with files.

    TASK DESCRIPTION:
    {task}
    CURRENT FILES:
    #files#

    WORKFLOW STEPS:
    1. Analyze the current project state
    2. Plan necessary changes or additions
    3. Execute changes using file actions
    4. Evaluate the project's progress

    Use the ProjectThinkResult structure to organize your thoughts and actions:

    class ProjectThinkResult(BaseModel):
        action: str  # 'code', 'evaluate', 'done'
        file_actions: List[FileAction]
        reasoning: str

    class FileAction(BaseModel):
        action: str  # 'write', 'read', 'delete', 'list'
        path: str
        content: Optional[str] = None

    EXECUTION RULES:
    1. Use absolute paths for all file operations
    2. Maintain a clear project structure
    3. Document your code and reasoning
    4. Ensure all necessary files are created and properly linked
    5. Use the appropriate language syntax and best practices for {lang}

    Next Action Required:
    1. Review the current project state
    2. Plan the next step in project development
    3. Execute file actions to implement changes
    """

        execution_history = []
        files = {}

        iter_i = 0
        self.verbose_output.log_header(task)

        while state != ThinkState.DONE:
            iter_i += 1
            self.verbose_output.print(Style.GREY(f"{iter_i}/{self.max_iter}"))
            if iter_i>self.max_iter:
                break
            if state == ThinkState.ACTION:
                think_result = await self.agent.a_format_class(
                    ProjectThinkResult,
                    project_prompt.replace('#files#', vfs.print_file_structure()),
                    message_context=execution_history
                )
                self.verbose_output.log_state(state.name, think_result)
                think_result = ProjectThinkResult(**think_result)
                for file_action in think_result.file_actions:
                    path = file_action.path
                    Path(file_action.path).mkdir(exist_ok=True)
                    if file_action.action == 'write':
                        vfs.write_file(path, file_action.content)
                        files[path] = file_action.content
                    elif file_action.action == 'read':
                        content = vfs.read_file(path)
                        files[path] = content
                    elif file_action.action == 'delete':
                        vfs.delete_file(path)
                        files.pop(path, None)
                    elif file_action.action == 'list':
                        dir_contents = vfs.list_directory(path)
                        files[path] = str(dir_contents)

                if think_result.action == 'evaluate':
                    state = ThinkState.PROCESSING
                elif think_result.action == 'done':
                    state = ThinkState.DONE

                execution_history.append(f"Action: {think_result.action}\nReasoning: {think_result.reasoning}")

            elif state == ThinkState.PROCESSING:
                if execute_function:
                    execution_result = await execute_function(files)
                    execution_history.append(f"Execution Result: {execution_result}")

                    evaluation_prompt = f"""
    Evaluate the current state of the project based on the execution result:

    {execution_result}

    Determine if the project is complete or if further modifications are needed.
    """
                    evaluation = await self.agent.a_format_class(
                        ProjectThinkResult,
                        evaluation_prompt,
                        message_context=execution_history
                    )
                    self.verbose_output.log_state(state.name, evaluation)
                    evaluation = ProjectThinkResult(**evaluation)
                    if evaluation.action == 'done':
                        state = ThinkState.DONE
                        result = execution_result
                    else:
                        state = ThinkState.ACTION
                else:
                    state = ThinkState.ACTION
            else:
                break

        return ProjectPipelineResult(
            result=result,
            execution_history=execution_history,
            files=files
        )

    async def __aenter__(self):
        self.clear()
        return self

    async def configure(self, verbose=None, print_function=None, with_js=False, agent=None, variables=None, web_kwargs=None):
        if verbose is not None and (print_function is not None or verbose != self.verbose_output.verbose):
            if agent is None:
                agent = self.agent
            else:
                self.agent = agent
            agent.verbose = verbose
            self.verbose_output = EnhancedVerboseOutput(verbose=verbose, print_f=print_function)

            if print_function is not None:
                agent.print_verbose = print_function
        if variables:
            self.variables = {**self.variables, **self._process_variables(variables)}
        if with_js and web_kwargs:
            self.browser_session: BrowserWrapper | None = BrowserWrapper(**web_kwargs)
        self.web_js = with_js
        if self.restore_var:
            self.restore()

        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self.web_js:
            await self.browser_session.close()
            if self.restore_var:
                self.save_session(f"Pipeline_Session_{self.agent.amd.name}")
        if exc_type is not None:
            print(f"Exception occurred: {exc_value}")
        else:
            print("Pipe Exit")

### -- extra -- ###

@dataclass
class SyncReport:
    """Report of variables synced from namespace to pipeline"""
    added: dict[str, str]
    skipped: dict[str, str]  # var_name -> reason
    errors: dict[str, str]  # var_name -> error message

    def __str__(self) -> str:
        parts = []
        if self.added:
            parts.append("Added variables:")
            for name, type_ in self.added.items():
                parts.append(f"  - {name}: {type_}")
        if self.skipped:
            parts.append("\nSkipped variables:")
            for name, reason in self.skipped.items():
                parts.append(f"  - {name}: {reason}")
        if self.errors:
            parts.append("\nErrors:")
            for name, error in self.errors.items():
                parts.append(f"  - {name}: {error}")
        return "\n".join(parts)


def sync_globals_to_vars(
    pipeline: Any,
    namespace: dict[str, Any] | None = None,
    prefix: str | None = None,
    include_types: type | list[type] | None = None,
    exclude_patterns: list[str] | None = None,
    exclude_private: bool = True,
    deep_copy: bool = False,
    only_serializable: bool = False
) -> SyncReport:
    """
    Sync global variables or a specific namespace to pipeline variables.

    Args:
        pipeline: Pipeline instance to sync variables to
        namespace: Optional dictionary of variables (defaults to globals())
        prefix: Optional prefix for variable names (e.g., 'global_')
        include_types: Only include variables of these types
        exclude_patterns: List of regex patterns to exclude
        exclude_private: Exclude variables starting with underscore
        deep_copy: Create deep copies of variables instead of references
        only_serializable: Only include variables that can be serialized

    Returns:
        SyncReport with details about added, skipped and error variables

    Usage example:
# Basic usage - sync all globals
report = sync_globals_to_vars(pipeline)

# Sync only numeric types with prefix
report = sync_globals_to_vars(
    pipeline,
    include_types=[int, float],
    prefix="global_"
)

# Sync from specific namespace
import numpy as np
namespace = {"arr": np.array([1,2,3])}
report = sync_globals_to_vars(pipeline, namespace=namespace)

# Sync with deep copy and serialization check
report = sync_globals_to_vars(
    pipeline,
    deep_copy=True,
    only_serializable=True
)
    """
    # Initialize report
    report = SyncReport(
        added={},
        skipped={},
        errors={}
    )

    # Get namespace
    if namespace is None:
        # Get caller's globals
        namespace = currentframe().f_back.f_globals

    # Compile exclude patterns
    if exclude_patterns:
        patterns = [re.compile(pattern) for pattern in exclude_patterns]
    else:
        patterns = []

    # Normalize include_types
    if include_types and not isinstance(include_types, list | tuple | set):
        include_types = [include_types]
    def get_type_info(var: Any) -> str:
        """Helper to get detailed type information"""
        if isinstance(var, type):
            return f"class '{var.__name__}'"
        elif isinstance(var, BaseModel):
            return f"Pydantic model '{var.__class__.__name__}'"
        elif hasattr(var, '__class__'):
            type_name = var.__class__.__name__
            module_name = var.__class__.__module__
            if module_name != 'builtins':
                return f"{module_name}.{type_name}"
            return type_name
        return type(var).__name__
    # Process each variable
    for name, value in namespace.items():
        try:
            # Skip if matches exclude criteria
            if exclude_private and name.startswith('_'):
                report.skipped[name] = "private variable"
                continue

            if any(pattern.match(name) for pattern in patterns):
                report.skipped[name] = "matched exclude pattern"
                continue

            if include_types and not isinstance(value, tuple(include_types)):
                report.skipped[name] = f"type {type(value).__name__} not in include_types"
                continue

            # Test serialization if required
            if only_serializable:
                try:
                    import pickle
                    pickle.dumps(value)
                except Exception as e:
                    report.skipped[name] = f"not serializable: {str(e)}"
                    continue

            # Prepare variable
            var_value = deepcopy(value) if deep_copy else value
            var_name = f"{prefix}{name}" if prefix else name

            # Add to pipeline variables
            pipeline.variables[var_name] = var_value
            report.added[var_name] = get_type_info(value)

        except Exception as e:
            report.errors[name] = str(e)

    return report


if __name__ == '__main__':
    # agent = get_free_agent("demo", "anthropic/claude-3-haiku-20240307")
    async def run_code():
        mock_ipy = MockIPython()
        mock_ipy.user_ns['VirtualFileSystem'] = VirtualFileSystem
        # Run async code with top-level await
        result = await mock_ipy.run_cell("""
if __name__ == '__main__':
    x = 1
        """, live_output=True)

        print("Result:", result)


    # Run the async function
    asyncio.run(run_code())

    #asd = "Evaluation: Output -> \nstdout:Episode 0: Total Reward = 35.0, Epsilon = 0.98\n\r⣾ code | 0.04\x1b[K\r⣽ code | 0.15\x1b[K\r⣻ code | 0.26\x1b[K\r⢿ code | 0.37\x1b[KEpisode 50: Total Reward = 10.0, Epsilon = 0.06\n\r⡿ code | 0.48\x1b[K\r⣟ code | 0.59\x1b[K\r⣯ code | 0.70\x1b[K\r⣷ code | 0.81\x1b[K\r⣾ code | 0.92\x1b[K\r⣽ code | 1.03\x1b[K\r⣻ code | 1.14\x1b[K\r⢿ code | 1.25\x1b[K\r⡿ code | 1.37\x1b[K\r⣟ code | 1.47\x1b[K\r⣯ code | 1.58\x1b[K\r⣷ code | 1.70\x1b[K\r⣾ code | 1.81\x1b[K\r⣽ code | 1.92\x1b[K\r⣻ code | 2.03\x1b[KEpisode 100: Total Reward = 58.0, Epsilon = 0.01\n\r⢿ code | 2.14\x1b[K\r⡿ code | 2.25\x1b[K\r⣟ code | 2.36\x1b[K\r⣯ code | 2.47\x1b[K\r⣷ code | 2.58\x1b[K\r⣾ code | 2.69\x1b[K\r⣽ code | 2.80\x1b[K\r⣻ code | 2.91\x1b[K\r⢿ code | 3.02\x1b[K\r⡿ code | 3.13\x1b[K\r⣟ code | 3.24\x1b[K\r⣯ code | 3.35\x1b[K\r⣷ code | 3.46\x1b[K\r⣾ code | 3.57\x1b[K\r⣽ code | 3.68\x1b[K\r⣻ code | 3.79\x1b[K\r⢿ code | 3.90\x1b[K\r⡿ code | 4.01\x1b[K\r⣟ code | 4.12\x1b[K\r⣯ code | 4.23\x1b[K\r⣷ code | 4.34\x1b[K\r⣾ code | 4.45\x1b[K\r⣽ code | 4.56\x1b[K\r⣻ code | 4.67\x1b[K\r⢿ code | 4.79\x1b[K\r⡿ code | 4.90\x1b[K\r⣟ code | 5.01\x1b[K\r⣯ code | 5.12\x1b[K\r⣷ code | 5.23\x1b[K\r⣾ code | 5.34\x1b[K\r⣽ code | 5.45\x1b[K\r⣻ code | 5.56\x1b[K\r⢿ code | 5.67\x1b[K\r⡿ code | 5.78\x1b[K\r⣟ code | 5.89\x1b[K\r⣯ code | 6.00\x1b[K\r⣷ code | 6.11\x1b[K\r⣾ code | 6.22\x1b[K\r⣽ code | 6.32\x1b[K\r⣻ code | 6.42\x1b[K\r⢿ code | 6.53\x1b[K\r⡿ code | 6.64\x1b[K\r⣟ code | 6.75\x1b[K\r⣯ code | 6.86\x1b[K\r⣷ code | 6.97\x1b[K\r⣾ code | 7.08\x1b[K\r⣽ code | 7.19\x1b[K\r⣻ code | 7.30\x1b[K\r⢿ code | 7.41\x1b[K\r⡿ code | 7.52\x1b[K\r⣟ code | 7.63\x1b[K\r⣯ code | 7.74\x1b[K\r⣷ code | 7.85\x1b[KEpisode 150: Total Reward = 200.0, Epsilon = 0.01\n\r⣾ code | 7.96\x1b[K\r⣽ code | 8.07\x1b[K\r⣻ code | 8.18\x1b[K\r⢿ code | 8.29\x1b[K\r⡿ code | 8.40\x1b[K\r⣟ code | 8.51\x1b[K\r⣯ code | 8.62\x1b[K\r⣷ code | 8.73\x1b[K\r⣾ code | 8.84\x1b[K\r⣽ code | 8.95\x1b[K\r⣻ code | 9.07\x1b[K\r⢿ code | 9.18\x1b[K\r⡿ code | 9.29\x1b[K\r⣟ code | 9.40\x1b[K\r⣯ code | 9.51\x1b[K\r⣷ code | 9.61\x1b[K\r⣾ code | 9.72\x1b[K\r⣽ code | 9.84\x1b[K\r⣻ code | 9.94\x1b[K\r⢿ code | 10.04\x1b[K\r⡿ code | 10.15\x1b[K\r⣟ code | 10.26\x1b[K\r⣯ code | 10.37\x1b[K\r⣷ code | 10.48\x1b[K\r⣾ code | 10.59\x1b[K\r⣽ code | 10.71\x1b[K\r⣻ code | 10.81\x1b[K\r⢿ code | 10.92\x1b[K\r⡿ code | 11.03\x1b[K\r⣟ code | 11.15\x1b[K\r⣯ code | 11.26\x1b[K\r⣷ code | 11.37\x1b[K\r⣾ code | 11.48\x1b[K\r⣽ code | 11.59\x1b[K\r⣻ code | 11.70\x1b[K\r⢿ code | 11.81\x1b[K\r⡿ code | 11.92\x1b[K\r⣟ code | 12.03\x1b[K\r⣯ code | 12.14\x1b[K\r⣷ code | 12.25\x1b[K\r⣾ code | 12.36\x1b[K\r⣽ code | 12.47\x1b[K\r⣻ code | 12.57\x1b[K\r⢿ code | 12.67\x1b[K\r⡿ code | 12.77\x1b[K\r⣟ code | 12.87\x1b[K\r⣯ code | 12.98\x1b[K\r⣷ code | 13.08\x1b[K\r⣾ code | 13.19\x1b[K\r⣽ code | 13.29\x1b[K\r⣻ code | 13.39\x1b[K\r⢿ code | 13.51\x1b[K\r⡿ code | 13.61\x1b[K\r⣟ code | 13.71\x1b[K\r⣯ code | 13.83\x1b[K\r⣷ code | 13.93\x1b[K\r⣾ code | 14.04\x1b[K\r⣽ code | 14.14\x1b[K\r⣻ code | 14.24\x1b[K\r⢿ code | 14.36\x1b[K\r⡿ code | 14.46\x1b[K\r⣟ code | 14.56\x1b[K\r⣯ code | 14.66\x1b[K\r⣷ code | 14.78\x1b[K\r⣾ code | 14.88\x1b[K\r⣽ code | 14.98\x1b[K\r⣻ code | 15.08\x1b[K\r⢿ code | 15.18\x1b[K\r⡿ code | 15.28\x1b[K\r⣟ code | 15.39\x1b[K\r⣯ code | 15.50\x1b[K\r⣷ code | 15.60\x1b[K\r⣾ code | 15.70\x1b[K\r⣽ code | 15.80\x1b[K\r⣻ code | 15.90\x1b[K\r⢿ code | 16.01\x1b[K\r⡿ code | 16.11\x1b[K\r⣟ code | 16.21\x1b[K\r⣯ code | 16.33\x1b[K\r⣷ code | 16.43\x1b[K\r⣾ code | 16.53\x1b[K\r⣽ code | 16.65\x1b[K\r⣻ code | 16.75\x1b[KEpisode 200: Total Reward = 200.0, Epsilon = 0.01\n\r⢿ code | 16.86\x1b[K\r⡿ code | 16.96\x1b[K\r⣟ code | 17.06\x1b[K\r⣯ code | 17.16\x1b[K\r⣷ code | 17.28\x1b[K\r⣾ code | 17.38\x1b[K\r⣽ code | 17.50\x1b[K\r⣻ code | 17.60\x1b[K\r⢿ code | 17.70\x1b[K\r⡿ code | 17.80\x1b[K\r⣟ code | 17.90\x1b[K\r⣯ code | 18.01\x1b[K\r⣷ code | 18.11\x1b[K\r⣾ code | 18.21\x1b[K\r⣽ code | 18.33\x1b[K\r⣻ code | 18.43\x1b[K\r⢿ code | 18.53\x1b[K\r⡿ code | 18.63\x1b[K\r⣟ code | 18.73\x1b[K\r⣯ code | 18.85\x1b[K\r⣷ code | 18.95\x1b[K\r⣾ code | 19.06\x1b[K\r⣽ code | 19.16\x1b[K\r⣻ code | 19.27\x1b[K\r⢿ code | 19.38\x1b[K\r⡿ code | 19.48\x1b[K\r⣟ code | 19.58\x1b[K\r⣯ code | 19.70\x1b[K\r⣷ code | 19.80\x1b[K\r⣾ code | 19.90\x1b[K\r⣽ code | 20.00\x1b[K\r⣻ code | 20.12\x1b[K\r⢿ code | 20.22\x1b[K\r⡿ code | 20.32\x1b[K\r⣟ code | 20.42\x1b[K\r⣯ code | 20.53\x1b[K\r⣷ code | 20.63\x1b[K\r⣾ code | 20.75\x1b[K\r⣽ code | 20.85\x1b[K\r⣻ code | 20.95\x1b[K\r⢿ code | 21.07\x1b[K\r⡿ code | 21.17\x1b[K\r⣟ code | 21.27\x1b[K\r⣯ code | 21.38\x1b[K\r⣷ code | 21.48\x1b[K\r⣾ code | 21.59\x1b[K\r⣽ code | 21.70\x1b[K\r⣻ code | 21.80\x1b[K\r⢿ code | 21.90\x1b[K\r⡿ code | 22.00\x1b[K\r⣟ code | 22.12\x1b[K\r⣯ code | 22.22\x1b[K\r⣷ code | 22.32\x1b[K\r⣾ code | 22.43\x1b[K\r⣽ code | 22.53\x1b[K\r⣻ code | 22.64\x1b[K\r⢿ code | 22.74\x1b[K\r⡿ code | 22.84\x1b[K\r⣟ code | 22.95\x1b[K\r⣯ code | 23.05\x1b[K\r⣷ code | 23.15\x1b[K\r⣾ code | 23.25\x1b[K\r⣽ code | 23.37\x1b[K\r⣻ code | 23.47\x1b[K\r⢿ code | 23.57\x1b[K\r⡿ code | 23.67\x1b[K\r⣟ code | 23.79\x1b[K\r⣯ code | 23.89\x1b[K\r⣷ code | 24.00\x1b[K\r⣾ code | 24.10\x1b[K\r⣽ code | 24.20\x1b[K\r⣻ code | 24.32\x1b[K\r⢿ code | 24.42\x1b[K\r⡿ code | 24.54\x1b[K\r⣟ code | 24.64\x1b[K\r⣯ code | 24.74\x1b[K\r⣷ code | 24.85\x1b[K\r⣾ code | 24.97\x1b[K\r⣽ code | 25.07\x1b[K\r⣻ code | 25.17\x1b[K\r⢿ code | 25.29\x1b[K\r⡿ code | 25.39\x1b[K\r⣟ code | 25.49\x1b[K\r⣯ code | 25.60\x1b[K\r⣷ code | 25.71\x1b[K\r⣾ code | 25.81\x1b[K\r⣽ code | 25.91\x1b[K\r⣻ code | 26.02\x1b[K\r⢿ code | 26.12\x1b[K\r⡿ code | 26.24\x1b[K\r⣟ code | 26.34\x1b[K\r⣯ code | 26.44\x1b[K\r⣷ code | 26.54\x1b[K\r⣾ code | 26.66\x1b[KEpisode 250: Total Reward = 200.0, Epsilon = 0.01\n\r⣽ code | 26.76\x1b[K\r⣻ code | 26.87\x1b[K\r⢿ code | 26.97\x1b[K\r⡿ code | 27.07\x1b[K\r⣟ code | 27.17\x1b[K\r⣯ code | 27.27\x1b[K\r⣷ code | 27.37\x1b[K\r⣾ code | 27.49\x1b[K\r⣽ code | 27.59\x1b[K\r⣻ code | 27.71\x1b[K\r⢿ code | 27.81\x1b[K\r⡿ code | 27.92\x1b[K\r⣟ code | 28.02\x1b[K\r⣯ code | 28.12\x1b[K\r⣷ code | 28.22\x1b[K\r⣾ code | 28.32\x1b[K\r⣽ code | 28.42\x1b[K\r⣻ code | 28.53\x1b[K\r⢿ code | 28.64\x1b[K\r⡿ code | 28.74\x1b[K\r⣟ code | 28.86\x1b[K\r⣯ code | 28.96\x1b[K\r⣷ code | 29.06\x1b[K\r⣾ code | 29.17\x1b[K\r⣽ code | 29.28\x1b[K\r⣻ code | 29.38\x1b[K\r⢿ code | 29.48\x1b[K\r⡿ code | 29.58\x1b[K\r⣟ code | 29.69\x1b[K\r⣯ code | 29.79\x1b[K\r⣷ code | 29.91\x1b[K\r⣾ code | 30.01\x1b[K\r⣽ code | 30.11\x1b[K\r⣻ code | 30.22\x1b[K\r⢿ code | 30.33\x1b[K\r⡿ code | 30.44\x1b[K\r⣟ code | 30.54\x1b[K\r⣯ code | 30.64\x1b[K\r⣷ code | 30.76\x1b[K\r⣾ code | 30.86\x1b[K\r⣽ code | 30.96\x1b[K\r⣻ code | 31.06\x1b[K\r⢿ code | 31.16\x1b[K\r⡿ code | 31.28\x1b[K\r⣟ code | 31.39\x1b[K\r⣯ code | 31.49\x1b[K\r⣷ code | 31.60\x1b[K\r⣾ code | 31.70\x1b[K\r⣽ code | 31.80\x1b[K\r⣻ code | 31.90\x1b[K\r⢿ code | 32.01\x1b[K\r⡿ code | 32.13\x1b[K\r⣟ code | 32.24\x1b[K\r⣯ code | 32.34\x1b[K\r⣷ code | 32.44\x1b[K\r⣾ code | 32.55\x1b[K\r⣽ code | 32.66\x1b[K\r⣻ code | 32.77\x1b[K\r⢿ code | 32.88\x1b[K\r⡿ code | 32.99\x1b[K\r⣟ code | 33.10\x1b[K\r⣯ code | 33.21\x1b[K\r⣷ code | 33.32\x1b[K\r⣾ code | 33.43\x1b[K\r⣽ code | 33.54\x1b[K\r⣻ code | 33.65\x1b[K\r⢿ code | 33.75\x1b[K\r⡿ code | 33.86\x1b[K\r⣟ code | 33.96\x1b[K\r⣯ code | 34.06\x1b[K\r⣷ code | 34.18\x1b[K\r⣾ code | 34.28\x1b[K\r⣽ code | 34.40\x1b[K\r⣻ code | 34.51\x1b[K\r⢿ code | 34.61\x1b[K\r⡿ code | 34.71\x1b[K\r⣟ code | 34.82\x1b[K\r⣯ code | 34.92\x1b[K\r⣷ code | 35.03\x1b[K\r⣾ code | 35.14\x1b[K\r⣽ code | 35.25\x1b[K\r⣻ code | 35.36\x1b[K\r⢿ code | 35.47\x1b[K\r⡿ code | 35.58\x1b[K\r⣟ code | 35.69\x1b[K\r⣯ code | 35.80\x1b[K\r⣷ code | 35.90\x1b[K\r⣾ code | 36.01\x1b[K\r⣽ code | 36.12\x1b[K\r⣻ code | 36.22\x1b[K\r⢿ code | 36.33\x1b[K\r⡿ code | 36.43\x1b[K\r⣟ code | 36.53\x1b[K\r⣯ code | 36.65\x1b[K\r⣷ code | 36.75\x1b[K\r⣾ code | 36.85\x1b[K\r⣽ code | 36.95\x1b[K\r⣻ code | 37.07\x1b[K\r⢿ code | 37.18\x1b[K\r⡿ code | 37.28\x1b[K\r⣟ code | 37.38\x1b[K\r⣯ code | 37.50\x1b[K\r⣷ code | 37.60\x1b[K\r⣾ code | 37.72\x1b[KEpisode 300: Total Reward = 200.0, Epsilon = 0.01\n\r⣽ code | 37.82\x1b[K\r⣻ code | 37.93\x1b[K\r⢿ code | 38.05\x1b[K\r⡿ code | 38.15\x1b[K\r⣟ code | 38.27\x1b[K\r⣯ code | 38.37\x1b[K\r⣷ code | 38.47\x1b[K\r⣾ code | 38.57\x1b[K\r⣽ code | 38.67\x1b[K\r⣻ code | 38.77\x1b[K\r⢿ code | 38.87\x1b[K\r⡿ code | 38.98\x1b[K\r⣟ code | 39.09\x1b[K\r⣯ code | 39.20\x1b[K\r⣷ code | 39.32\x1b[K\r⣾ code | 39.42\x1b[K\r⣽ code | 39.52\x1b[K\r⣻ code | 39.64\x1b[K\r⢿ code | 39.74\x1b[K\r⡿ code | 39.84\x1b[K\r⣟ code | 39.94\x1b[K\r⣯ code | 40.04\x1b[K\r⣷ code | 40.14\x1b[K\r⣾ code | 40.24\x1b[K\r⣽ code | 40.35\x1b[K\r⣻ code | 40.45\x1b[K\r⢿ code | 40.55\x1b[K\r⡿ code | 40.65\x1b[K\r⣟ code | 40.77\x1b[K\r⣯ code | 40.87\x1b[K\r⣷ code | 40.99\x1b[K\r⣾ code | 41.09\x1b[K\r⣽ code | 41.19\x1b[K\r⣻ code | 41.30\x1b[K\r⢿ code | 41.41\x1b[K\r⡿ code | 41.52\x1b[K\r⣟ code | 41.62\x1b[K\r⣯ code | 41.72\x1b[K\r⣷ code | 41.84\x1b[K\r⣾ code | 41.94\x1b[K\r⣽ code | 42.04\x1b[K\r⣻ code | 42.15\x1b[K\r⢿ code | 42.26\x1b[K\r⡿ code | 42.36\x1b[K\r⣟ code | 42.47\x1b[K\r⣯ code | 42.57\x1b[K\r⣷ code | 42.67\x1b[K\r⣾ code | 42.77\x1b[K\r⣽ code | 42.89\x1b[K\r⣻ code | 42.99\x1b[K\r⢿ code | 43.09\x1b[K\r⡿ code | 43.21\x1b[K\r⣟ code | 43.31\x1b[K\r⣯ code | 43.41\x1b[K\r⣷ code | 43.52\x1b[K\r⣾ code | 43.62\x1b[K\r⣽ code | 43.74\x1b[K\r⣻ code | 43.84\x1b[K\r⢿ code | 43.95\x1b[K\r⡿ code | 44.06\x1b[K\r⣟ code | 44.16\x1b[K\r⣯ code | 44.27\x1b[K\r⣷ code | 44.37\x1b[K\r⣾ code | 44.49\x1b[K\r⣽ code | 44.59\x1b[K\r⣻ code | 44.69\x1b[K\r⢿ code | 44.81\x1b[K\r⡿ code | 44.91\x1b[K\r⣟ code | 45.02\x1b[K\r⣯ code | 45.13\x1b[K\r⣷ code | 45.24\x1b[K\r⣾ code | 45.36\x1b[K\r⣽ code | 45.47\x1b[K\r⣻ code | 45.58\x1b[K\r⢿ code | 45.69\x1b[K\r⡿ code | 45.80\x1b[K\r⣟ code | 45.91\x1b[K\r⣯ code | 46.01\x1b[K\r⣷ code | 46.11\x1b[K\r⣾ code | 46.23\x1b[K\r⣽ code | 46.33\x1b[K\r⣻ code | 46.44\x1b[K\r⢿ code | 46.56\x1b[K\r⡿ code | 46.66\x1b[K\r⣟ code | 46.76\x1b[K\r⣯ code | 46.88\x1b[K\r⣷ code | 46.98\x1b[K\r⣾ code | 47.09\x1b[K\r⣽ code | 47.21\x1b[K\r⣻ code | 47.31\x1b[K\r⢿ code | 47.41\x1b[K\r⡿ code | 47.53\x1b[K\r⣟ code | 47.63\x1b[K\r⣯ code | 47.74\x1b[K\r⣷ code | 47.85\x1b[K\r⣾ code | 47.96\x1b[K\r⣽ code | 48.06\x1b[K\r⣻ code | 48.16\x1b[K\r⢿ code | 48.26\x1b[K\r⡿ code | 48.37\x1b[K\r⣟ code | 48.48\x1b[K\r⣯ code | 48.59\x1b[K\r⣷ code | 48.70\x1b[K\r⣾ code | 48.81\x1b[K\r⣽ code | 48.92\x1b[K\r⣻ code | 49.03\x1b[K\r⢿ code | 49.13\x1b[K\r⡿ code | 49.24\x1b[K\r⣟ code | 49.35\x1b[K\r⣯ code | 49.45\x1b[K\r⣷ code | 49.56\x1b[K\r⣾ code | 49.66\x1b[K\r⣽ code | 49.76\x1b[K\r⣻ code | 49.88\x1b[K\r⢿ code | 49.99\x1b[K\r⡿ code | 50.09\x1b[K\r⣟ code | 50.21\x1b[K\r⣯ code | 50.32\x1b[KEpisode 350: Total Reward = 200.0, Epsilon = 0.01\n\r⣷ code | 50.43\x1b[K\r⣾ code | 50.54\x1b[K\r⣽ code | 50.65\x1b[K\r⣻ code | 50.75\x1b[K\r⢿ code | 50.86\x1b[K\r⡿ code | 50.96\x1b[K\r⣟ code | 51.08\x1b[K\r⣯ code | 51.18\x1b[K\r⣷ code | 51.28\x1b[K\r⣾ code | 51.38\x1b[K\r⣽ code | 51.50\x1b[K\r⣻ code | 51.60\x1b[K\r⢿ code | 51.71\x1b[K\r⡿ code | 51.81\x1b[K\r⣟ code | 51.92\x1b[K\r⣯ code | 52.02\x1b[K\r⣷ code | 52.13\x1b[K\r⣾ code | 52.23\x1b[K\r⣽ code | 52.33\x1b[K\r⣻ code | 52.43\x1b[K\r⢿ code | 52.53\x1b[K\r⡿ code | 52.65\x1b[K\r⣟ code | 52.75\x1b[K\r⣯ code | 52.85\x1b[K\r⣷ code | 52.97\x1b[K\r⣾ code | 53.07\x1b[K\r⣽ code | 53.18\x1b[K\r⣻ code | 53.29\x1b[K\r⢿ code | 53.40\x1b[K\r⡿ code | 53.51\x1b[K\r⣟ code | 53.62\x1b[K\r⣯ code | 53.73\x1b[K\r⣷ code | 53.84\x1b[K\r⣾ code | 53.95\x1b[K\r⣽ code | 54.05\x1b[K\r⣻ code | 54.15\x1b[K\r⢿ code | 54.27\x1b[K\r⡿ code | 54.37\x1b[K\r⣟ code | 54.48\x1b[K\r⣯ code | 54.58\x1b[K\r⣷ code | 54.69\x1b[K\r⣾ code | 54.80\x1b[K\r⣽ code | 54.90\x1b[K\r⣻ code | 55.02\x1b[K\r⢿ code | 55.12\x1b[K\r⡿ code | 55.22\x1b[K\r⣟ code | 55.33\x1b[K\r⣯ code | 55.44\x1b[K\r⣷ code | 55.54\x1b[K\r⣾ code | 55.64\x1b[K\r⣽ code | 55.74\x1b[K\r⣻ code | 55.84\x1b[K\r⢿ code | 55.94\x1b[K\r⡿ code | 56.04\x1b[K\r⣟ code | 56.15\x1b[K\r⣯ code | 56.25\x1b[K\r⣷ code | 56.35\x1b[K\r⣾ code | 56.47\x1b[K\r⣽ code | 56.57\x1b[K\r⣻ code | 56.67\x1b[K\r⢿ code | 56.79\x1b[K\r⡿ code | 56.89\x1b[K\r⣟ code | 56.99\x1b[K\r⣯ code | 57.09\x1b[K\r⣷ code | 57.20\x1b[K\r⣾ code | 57.30\x1b[K\r⣽ code | 57.42\x1b[K\r⣻ code | 57.52\x1b[K\r⢿ code | 57.64\x1b[K\r⡿ code | 57.74\x1b[K\r⣟ code | 57.84\x1b[K\r⣯ code | 57.95\x1b[K\r⣷ code | 58.05\x1b[K\r⣾ code | 58.16\x1b[K\r⣽ code | 58.27\x1b[K\r⣻ code | 58.37\x1b[K\r⢿ code | 58.47\x1b[K\r⡿ code | 58.57\x1b[K\r⣟ code | 58.67\x1b[K\r⣯ code | 58.79\x1b[K\r⣷ code | 58.89\x1b[K\r⣾ code | 59.01\x1b[K\r⣽ code | 59.11\x1b[K\r⣻ code | 59.21\x1b[K\r⢿ code | 59.36\x1b[K\r⡿ code | 59.48\x1b[K\r⣟ code | 59.59\x1b[K\r⣯ code | 59.69\x1b[K\r⣷ code | 59.81\x1b[K\r⣾ code | 59.91\x1b[K\r⣽ code | 60.02\x1b[K\r⣻ code | 60.14\x1b[K\r⢿ code | 60.24\x1b[K\r⡿ code | 60.36\x1b[K\r⣟ code | 60.46\x1b[K\r⣯ code | 60.56\x1b[K\r⣷ code | 60.67\x1b[K\r⣾ code | 60.78\x1b[K\r⣽ code | 60.89\x1b[K\r⣻ code | 60.99\x1b[K\r⢿ code | 61.11\x1b[K\r⡿ code | 61.22\x1b[K\r⣟ code | 61.33\x1b[K\r⣯ code | 61.44\x1b[K\r⣷ code | 61.54\x1b[K\r⣾ code | 61.66\x1b[K\r⣽ code | 61.76\x1b[K\r⣻ code | 61.86\x1b[K\r⢿ code | 61.98\x1b[K\r⡿ code | 62.08\x1b[K\r⣟ code | 62.19\x1b[K\r⣯ code | 62.31\x1b[K\r⣷ code | 62.41\x1b[K\r⣾ code | 62.51\x1b[K\r⣽ code | 62.63\x1b[K\r⣻ code | 62.74\x1b[K\r⢿ code | 62.86\x1b[K\r⡿ code | 62.96\x1b[K\r⣟ code | 63.07\x1b[K\r⣯ code | 63.18\x1b[K\r⣷ code | 63.29\x1b[K\r⣾ code | 63.41\x1b[K\r⣽ code | 63.51\x1b[K\r⣻ code | 63.63\x1b[K\r⢿ code | 63.73\x1b[K\r⡿ code | 63.83\x1b[K\r⣟ code | 63.93\x1b[K\r⣯ code | 64.04\x1b[K\r⣷ code | 64.15\x1b[K\r⣾ code | 64.26\x1b[K\r⣽ code | 64.36\x1b[K\r⣻ code | 64.48\x1b[K\r⢿ code | 64.58\x1b[K\r⡿ code | 64.68\x1b[K\r⣟ code | 64.80\x1b[K\r⣯ code | 64.91\x1b[K\r⣷ code | 65.01\x1b[K\r⣾ code | 65.11\x1b[KEpisode 400: Total Reward = 200.0, Epsilon = 0.01\n\r⣽ code | 65.21\x1b[K\r⣻ code | 65.34\x1b[K\r⢿ code | 65.45\x1b[K\r⡿ code | 65.57\x1b[K\r⣟ code | 65.68\x1b[K\r⣯ code | 65.80\x1b[K\r⣷ code | 65.90\x1b[K\r⣾ code | 66.00\x1b[K\r⣽ code | 66.10\x1b[K\r⣻ code | 66.23\x1b[K\r⢿ code | 66.33\x1b[K\r⡿ code | 66.45\x1b[K\r⣟ code | 66.55\x1b[K\r⣯ code | 66.65\x1b[K\r⣷ code | 66.76\x1b[K\r⣾ code | 66.88\x1b[K\r⣽ code | 66.98\x1b[K\r⣻ code | 67.08\x1b[K\r⢿ code | 67.20\x1b[K\r⡿ code | 67.30\x1b[K\r⣟ code | 67.41\x1b[K\r⣯ code | 67.52\x1b[K\r⣷ code | 67.63\x1b[K\r⣾ code | 67.73\x1b[K\r⣽ code | 67.83\x1b[K\r⣻ code | 67.95\x1b[K\r⢿ code | 68.05\x1b[K\r⡿ code | 68.16\x1b[K\r⣟ code | 68.27\x1b[K\r⣯ code | 68.38\x1b[K\r⣷ code | 68.48\x1b[K\r⣾ code | 68.60\x1b[K\r⣽ code | 68.70\x1b[K\r⣻ code | 68.82\x1b[K\r⢿ code | 68.92\x1b[K\r⡿ code | 69.03\x1b[K\r⣟ code | 69.13\x1b[K\r⣯ code | 69.23\x1b[K\r⣷ code | 69.34\x1b[K\r⣾ code | 69.46\x1b[K\r⣽ code | 69.60\x1b[K\r⣻ code | 69.70\x1b[K\r⢿ code | 69.82\x1b[K\r⡿ code | 69.93\x1b[K\r⣟ code | 70.03\x1b[K\r⣯ code | 70.13\x1b[K\r⣷ code | 70.23\x1b[K\r⣾ code | 70.35\x1b[K\r⣽ code | 70.45\x1b[K\r⣻ code | 70.57\x1b[K\r⢿ code | 70.67\x1b[K\r⡿ code | 70.77\x1b[K\r⣟ code | 70.87\x1b[K\r⣯ code | 70.98\x1b[K\r⣷ code | 71.09\x1b[K\r⣾ code | 71.23\x1b[K\r⣽ code | 71.33\x1b[K\r⣻ code | 71.44\x1b[K\r⢿ code | 71.55\x1b[K\r⡿ code | 71.65\x1b[K\r⣟ code | 71.77\x1b[K\r⣯ code | 71.89\x1b[K\r⣷ code | 72.05\x1b[K\r⣾ code | 72.15\x1b[K\r⣽ code | 72.27\x1b[K\r⣻ code | 72.37\x1b[K\r⢿ code | 72.49\x1b[K\r⡿ code | 72.59\x1b[K\r⣟ code | 72.70\x1b[K\r⣯ code | 72.80\x1b[K\r⣷ code | 72.92\x1b[K\r⣾ code | 73.02\x1b[K\r⣽ code | 73.14\x1b[K\r⣻ code | 73.24\x1b[K\r⢿ code | 73.34\x1b[K\r⡿ code | 73.45\x1b[K\r⣟ code | 73.55\x1b[K\r⣯ code | 73.66\x1b[K\r⣷ code | 73.77\x1b[K\r⣾ code | 73.87\x1b[K\r⣽ code | 73.99\x1b[K\r⣻ code | 74.09\x1b[K\r⢿ code | 74.19\x1b[K\r⡿ code | 74.30\x1b[K\r⣟ code | 74.40\x1b[K\r⣯ code | 74.51\x1b[K\r⣷ code | 74.62\x1b[K\r⣾ code | 74.72\x1b[K\r⣽ code | 74.86\x1b[K\r⣻ code | 74.98\x1b[K\r⢿ code | 75.09\x1b[K\r⡿ code | 75.19\x1b[K\r⣟ code | 75.29\x1b[K\r⣯ code | 75.40\x1b[K\r⣷ code | 75.50\x1b[K\r⣾ code | 75.61\x1b[K\r⣽ code | 75.72\x1b[K\r⣻ code | 75.84\x1b[K\r⢿ code | 75.95\x1b[K\r⡿ code | 76.08\x1b[K\r⣟ code | 76.19\x1b[K\r⣯ code | 76.30\x1b[K\r⣷ code | 76.41\x1b[K\r⣾ code | 76.52\x1b[K\r⣽ code | 76.64\x1b[K\r⣻ code | 76.75\x1b[K\r⢿ code | 76.86\x1b[K\r⡿ code | 76.97\x1b[K\r⣟ code | 77.08\x1b[K\r⣯ code | 77.19\x1b[K\r⣷ code | 77.31\x1b[K\r⣾ code | 77.43\x1b[K\r⣽ code | 77.54\x1b[K\r⣻ code | 77.65\x1b[K\r⢿ code | 77.76\x1b[K\r⡿ code | 77.87\x1b[K\r⣟ code | 77.98\x1b[K\r⣯ code | 78.09\x1b[K\r⣷ code | 78.20\x1b[K\r⣾ code | 78.31\x1b[K\r⣽ code | 78.42\x1b[K\r⣻ code | 78.53\x1b[K\r⢿ code | 78.64\x1b[K\r⡿ code | 78.75\x1b[K\r⣟ code | 78.86\x1b[K\r⣯ code | 78.97\x1b[K\r⣷ code | 79.08\x1b[K\r⣾ code | 79.19\x1b[K\r⣽ code | 79.30\x1b[K\r⣻ code | 79.41\x1b[K\r⢿ code | 79.52\x1b[K\r⡿ code | 79.63\x1b[K\r⣟ code | 79.75\x1b[K\r⣯ code | 79.88\x1b[K\r⣷ code | 79.98\x1b[K\r⣾ code | 80.09\x1b[K\r⣽ code | 80.20\x1b[K\r⣻ code | 80.31\x1b[K\r⢿ code | 80.44\x1b[K\r⡿ code | 80.56\x1b[K\r⣟ code | 80.66\x1b[K\r⣯ code | 80.77\x1b[K\r⣷ code | 80.88\x1b[K\r⣾ code | 80.99\x1b[K\r⣽ code | 81.10\x1b[K\r⣻ code | 81.21\x1b[K\r⢿ code | 81.32\x1b[K\r⡿ code | 81.43\x1b[K\r⣟ code | 81.54\x1b[K\r⣯ code | 81.65\x1b[K\r⣷ code | 81.76\x1b[K\r⣾ code | 81.87\x1b[K\r⣽ code | 81.98\x1b[K\r⣻ code | 82.09\x1b[K\r⢿ code | 82.20\x1b[K\r⡿ code | 82.31\x1b[K\r⣟ code | 82.42\x1b[K\r⣯ code | 82.53\x1b[K\r⣷ code | 82.65\x1b[KEpisode 450: Total Reward = 200.0, Epsilon = 0.01\n\r⣾ code | 82.78\x1b[K\r⣽ code | 82.90\x1b[K\r⣻ code | 83.01\x1b[K\r⢿ code | 83.12\x1b[K\r⡿ code | 83.23\x1b[K\r⣟ code | 83.34\x1b[K\r⣯ code | 83.45\x1b[K\r⣷ code | 83.56\x1b[K\r⣾ code | 83.72\x1b[K\r⣽ code | 83.83\x1b[K\r⣻ code | 83.94\x1b[K\r⢿ code | 84.05\x1b[K\r⡿ code | 84.16\x1b[K\r⣟ code | 84.27\x1b[K\r⣯ code | 84.38\x1b[K\r⣷ code | 84.49\x1b[K\r⣾ code | 84.60\x1b[K\r⣽ code | 84.71\x1b[K\r⣻ code | 84.82\x1b[K\r⢿ code | 84.93\x1b[K\r⡿ code | 85.08\x1b[K\r⣟ code | 85.19\x1b[K\r⣯ code | 85.34\x1b[K\r⣷ code | 85.45\x1b[K\r⣾ code | 85.56\x1b[K\r⣽ code | 85.67\x1b[K\r⣻ code | 85.78\x1b[K\r⢿ code | 85.89\x1b[K\r⡿ code | 86.00\x1b[K\r⣟ code | 86.10\x1b[K\r⣯ code | 86.20\x1b[K\r⣷ code | 86.30\x1b[K\r⣾ code | 86.40\x1b[K\r⣽ code | 86.52\x1b[K\r⣻ code | 86.62\x1b[K\r⢿ code | 86.72\x1b[K\r⡿ code | 86.83\x1b[K\r⣟ code | 86.93\x1b[K\r⣯ code | 87.04\x1b[K\r⣷ code | 87.15\x1b[K\r⣾ code | 87.25\x1b[K\r⣽ code | 87.35\x1b[K\r⣻ code | 87.47\x1b[K\r⢿ code | 87.57\x1b[K\r⡿ code | 87.68\x1b[K\r⣟ code | 87.79\x1b[K\r⣯ code | 87.90\x1b[K\r⣷ code | 88.00\x1b[K\r⣾ code | 88.12\x1b[K\r⣽ code | 88.22\x1b[K\r⣻ code | 88.34\x1b[K\r⢿ code | 88.44\x1b[K\r⡿ code | 88.55\x1b[K\r⣟ code | 88.65\x1b[K\r⣯ code | 88.77\x1b[K\r⣷ code | 88.87\x1b[K\r⣾ code | 88.99\x1b[K\r⣽ code | 89.09\x1b[K\r⣻ code | 89.19\x1b[K\r⢿ code | 89.29\x1b[K\r⡿ code | 89.40\x1b[K\r⣟ code | 89.52\x1b[K\r⣯ code | 89.62\x1b[K\r⣷ code | 89.74\x1b[K\r⣾ code | 89.84\x1b[K\r⣽ code | 89.95\x1b[K\r⣻ code | 90.06\x1b[K\r⢿ code | 90.16\x1b[K\r⡿ code | 90.27\x1b[K\r⣟ code | 90.37\x1b[K\r⣯ code | 90.48\x1b[K\r⣷ code | 90.60\x1b[K\r⣾ code | 90.71\x1b[K\r⣽ code | 90.82\x1b[K\r⣻ code | 90.92\x1b[K\r⢿ code | 91.04\x1b[K\r⡿ code | 91.14\x1b[K\r⣟ code | 91.24\x1b[K\r⣯ code | 91.36\x1b[K\r⣷ code | 91.46\x1b[K\r⣾ code | 91.56\x1b[K\r⣽ code | 91.67\x1b[K\r⣻ code | 91.77\x1b[K\r⢿ code | 91.87\x1b[K\r⡿ code | 91.97\x1b[K\r⣟ code | 92.09\x1b[K\r⣯ code | 92.19\x1b[K\r⣷ code | 92.29\x1b[K\r⣾ code | 92.41\x1b[K\r⣽ code | 92.51\x1b[K\r⣻ code | 92.61\x1b[K\r⢿ code | 92.71\x1b[K\r⡿ code | 92.82\x1b[K\r⣟ code | 92.92\x1b[K\r⣯ code | 93.05\x1b[K\r⣷ code | 93.16\x1b[K\r⣾ code | 93.28\x1b[K\r⣽ code | 93.39\x1b[K\r⣻ code | 93.51\x1b[K\r⢿ code | 93.61\x1b[K\r⡿ code | 93.71\x1b[K\r⣟ code | 93.82\x1b[K\r⣯ code | 93.93\x1b[K\r⣷ code | 94.03\x1b[K\r⣾ code | 94.13\x1b[K\r⣽ code | 94.24\x1b[K\r⣻ code | 94.34\x1b[K\r⢿ code | 94.46\x1b[K\r⡿ code | 94.56\x1b[K\r⣟ code | 94.68\x1b[K\r⣯ code | 94.78\x1b[K\r⣷ code | 94.89\x1b[K\r⣾ code | 94.99\x1b[K\r⣽ code | 95.11\x1b[K\r⣻ code | 95.21\x1b[K\r⢿ code | 95.31\x1b[K\r⡿ code | 95.43\x1b[K\r⣟ code | 95.53\x1b[K\r⣯ code | 95.63\x1b[K\r⣷ code | 95.74\x1b[K\r⣾ code | 95.84\x1b[K\r⣽ code | 95.94\x1b[K\r⣻ code | 96.04\x1b[K\r⢿ code | 96.16\x1b[K\r⡿ code | 96.26\x1b[K\r⣟ code | 96.36\x1b[K\r⣯ code | 96.46\x1b[K\r⣷ code | 96.58\x1b[K\r⣾ code | 96.68\x1b[K\r⣽ code | 96.78\x1b[K\r⣻ code | 96.88\x1b[K\r⢿ code | 96.99\x1b[K\r⡿ code | 97.10\x1b[K\r⣟ code | 97.20\x1b[K\r⣯ code | 97.31\x1b[K\r⣷ code | 97.41\x1b[K\r⣾ code | 97.51\x1b[K\r⣽ code | 97.63\x1b[K\r⣻ code | 97.73\x1b[K\r⢿ code | 97.85\x1b[K\r⡿ code | 97.95\x1b[K\r⣟ code | 98.06\x1b[K\r⣯ code | 98.16\x1b[K\r⣷ code | 98.28\x1b[K\r⣾ code | 98.38\x1b[K\r⣽ code | 98.48\x1b[K\r⣻ code | 98.60\x1b[K\r⢿ code | 98.70\x1b[K\r⡿ code | 98.80\x1b[K\r⣟ code | 98.90\x1b[K\r⣯ code | 99.01\x1b[K\r⣷ code | 99.11\x1b[K\r⣾ code | 99.23\x1b[K\r⣽ code | 99.33\x1b[K\r⣻ code | 99.45\x1b[K\r⢿ code | 99.55\x1b[K\r⡿ code | 99.65\x1b[K\r⣟ code | 99.76\x1b[K\r⣯ code | 99.87\x1b[K\r⣷ code | 99.98\x1b[K\r⣾ code | 100.08\x1b[K\r⣽ code | 100.18\x1b[K\r⣻ code | 100.30\x1b[K\r⢿ code | 100.40\x1b[K\r⡿ code | 100.50\x1b[K\r⣟ code | 100.60\x1b[K\r⣯ code | 100.72\x1b[K\r⣷ code | 100.82\x1b[K\r⣾ code | 100.92\x1b[K\r⣽ code | 101.03\x1b[K\r⣻ code | 101.13\x1b[K\r⢿ code | 101.23\x1b[K\r⡿ code | 101.33\x1b[K\r⣟ code | 101.45\x1b[K\r⣯ code | 101.55\x1b[K\r⣷ code | 101.65\x1b[K\r⣾ code | 101.76\x1b[K\r⣽ code | 101.87\x1b[K\r⣻ code | 101.98\x1b[K\r⢿ code | 102.08\x1b[K\r⡿ code | 102.18\x1b[K\r⣟ code | 102.30\x1b[K\r⣯ code | 102.40\x1b[K\r⣷ code | 102.50\x1b[K\r⣾ code | 102.62\x1b[K\r⣽ code | 102.72\x1b[K\r⣻ code | 102.83\x1b[K\r⢿ code | 102.93\x1b[K\r⡿ code | 103.04\x1b[K\r⣟ code | 103.15\x1b[K\r⣯ code | 103.25\x1b[K\r⣷ code | 103.35\x1b[K\r⣾ code | 103.47\x1b[K\r⣽ code | 103.57\x1b[K\r⣻ code | 103.67\x1b[K\r⢿ code | 103.78\x1b[K\r⡿ code | 103.89\x1b[K\r⣟ code | 103.99\x1b[K\r⣯ code | 104.10\x1b[K\r⣷ code | 104.20\x1b[K\r⣾ code | 104.30\x1b[K\r⣽ code | 104.40\x1b[K\r⣻ code | 104.50\x1b[K\r⢿ code | 104.62\x1b[K\r⡿ code | 104.72\x1b[K\r⣟ code | 104.82\x1b[K\r⣯ code | 104.92\x1b[K\r⣷ code | 105.02\x1b[K\r⣾ code | 105.13\x1b[K\r⣽ code | 105.24\x1b[K\r⣻ code | 105.35\x1b[K\r⢿ code | 105.45\x1b[K\r⡿ code | 105.55\x1b[K\r⣟ code | 105.65\x1b[K\r⣯ code | 105.75\x1b[K\r⣷ code | 105.85\x1b[K\r⣾ code | 105.95\x1b[K\r⣽ code | 106.05\x1b[K\r⣻ code | 106.17\x1b[K\r⢿ code | 106.27\x1b[K\r⡿ code | 106.39\x1b[K\r⣟ code | 106.49\x1b[K\r⣯ code | 106.59\x1b[K\r⣷ code | 106.69\x1b[K\r⣾ code | 106.80\x1b[K\r⣽ code | 106.90\x1b[K\r⣻ code | 107.01\x1b[K\r⢿ code | 107.11\x1b[K\r⡿ code | 107.22\x1b[K\r⣟ code | 107.32\x1b[K\r⣯ code | 107.42\x1b[K\r⣷ code | 107.52\x1b[K\r⣾ code | 107.64\x1b[K\r⣽ code | 107.74\x1b[K\r⣻ code | 107.85\x1b[K\r⢿ code | 107.96\x1b[K\r⡿ code | 108.06\x1b[K\r⣟ code | 108.17\x1b[K\r⣯ code | 108.27\x1b[K\r⣷ code | 108.37\x1b[K\r⣾ code | 108.49\x1b[K\r⣽ code | 108.59\x1b[K\r⣻ code | 108.69\x1b[K\r⢿ code | 108.79\x1b[K\r⡿ code | 108.91\x1b[K\r⣟ code | 109.01\x1b[K\r⣯ code | 109.11\x1b[K\r⣷ code | 109.22\x1b[K\r⣾ code | 109.33\x1b[K\r⣽ code | 109.44\x1b[K\r⣻ code | 109.54\x1b[K\r⢿ code | 109.64\x1b[K\r⡿ code | 109.74\x1b[K\r⣟ code | 109.84\x1b[K\r⣯ code | 109.94\x1b[K\r⣷ code | 110.06\x1b[K\r⣾ code | 110.16\x1b[K\r⣽ code | 110.27\x1b[K\r⣻ code | 110.38\x1b[K\r⢿ code | 110.48\x1b[K\r⡿ code | 110.58\x1b[K\r⣟ code | 110.69\x1b[K\nstderr:<string>:51: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at C:\\actions-runner\\_work\\pytorch\\pytorch\\builder\\windows\\pytorch\\torch\\csrc\\utils\\tensor_new.cpp:281.)\n"
    #cleaned_result = super_strip(asd)
    #print(type(cleaned_result), len(asd), len(cleaned_result))
    #print(cleaned_result)

if __name__ == "__main__":
    async def main():
        agent = get_app("demo").get_mod("isaa").get_agent("self")
        agent = await agent
        pipeline = Pipeline(
        agent = agent,
        variables = {"n": 10})
        result = await pipeline.run(task = "Calculate fibonacci sequence to n")
        print(result.result)


    asyncio.run(main())
    # print(asyncio.run(BrowserWrapper().run("Finde eine Lösung für mein Problem. ich habe ine rust aplikation die beim ausführen der exe zurückgibt : returned non-zero exit status 3221225781. oder spezifischer : (exit code: 0xc0000135, STATUS_DLL_NOT_FOUND) ich nutze pyo3 damit rust python verwenden kann in einer venv so wiet habe ich nur das gefunde : https://github.com/PyO3/pyo3/issues/3589 wie fixe ich mein probelm?")))


# Example usage and testing
if __name__ == "__main__2":
    async def demo():
        # Create formatter instance
        formatter = EnhancedVerboseOutput(verbose=True)

        # Demo header
        formatter.log_header("Dynamic Formatter Demo")

        # Demo different message types
        formatter.print_success("Formatter initialized successfully")
        formatter.print_info("Adapting to terminal width...")
        formatter.print_warning("This is a warning message")

        # Demo section
        formatter.print_section(
            "Features",
            "- Dynamic width adaptation\n- Clean formatting\n- Multiple output types\n- Progress indicators"
        )

        # Demo progress bar
        for i in range(11):
            formatter.print_progress_bar(i, 10, "Loading")
            await asyncio.sleep(0.1)
        print()  # New line after progress

        # Demo state logging
        formatter.log_state("PROCESSING", {
            "current_task": "Data processing",
            "items_processed": 150,
            "items_remaining": 50
        })

        # Demo code block
        formatter.print_code_block("""
def example_function():
    \"\"\"This is an example function\"\"\"
    return "Hello, World!"
        """.strip())

        # Demo table
        formatter.print_table(
            ["Name", "Status", "Progress"],
            [
                ["Task 1", "Complete", "100%"],
                ["Task 2", "In Progress", "65%"],
                ["Task 3", "Pending", "0%"]
            ]
        )

        # Demo spinner
        await formatter.process("Processing data...", asyncio.sleep(2))

        formatter.print_success("Demo completed!")


    # Run demo
    asyncio.run(demo())
