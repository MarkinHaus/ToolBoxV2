import json
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from typing import Any

from toolboxv2 import Spinner, Style


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

    def print_state(self, state: str, details: dict[str, Any] = None) -> str:
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
        for _i, (cell, width) in enumerate(zip(row, widths, strict=False)):
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

    def print_git_info(self) -> str | None:
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

        if content.startswith("```"):
            self.formatter.print_code_block(content)
            return

        if content.startswith("{") or content.startswith("[") and content.endswith("}") or content.endswith("]"):
            content = json.dumps(json.loads(content), indent=2)

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

    async def log_process_result(self, result: dict[str, Any]):
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
        timing = f" ({elapsed / 60:.1f}m)" if elapsed > 60 else f" ({elapsed:.1f}s)"

        self.formatter.print_header(f"{text}{timing}")

    def log_state(self, state: str, user_ns: dict = None, override: bool = False):
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

    def print_tool_call(self, tool_name: str, tool_args: dict, result: str | None = None):
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

    def print_event(self, event: dict):
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

