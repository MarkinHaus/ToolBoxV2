# file: toolboxv2/utils/clis/cli_printing.py
# Shared printing utilities for all CLI tools in ToolBoxV2
# Provides consistent UI components: headers, status messages, tables, separators

import unicodedata

# --- Enhanced UI Imports ---
try:
    from toolboxv2.utils.extras.Style import Style
except ImportError:
    try:
        from toolboxv2.extras.Style import Style
    except ImportError:
        # Fallback if Style is not available
        class Style:
            @staticmethod
            def GREY(x): return x

            @staticmethod
            def WHITE(x): return x

            @staticmethod
            def GREEN(x): return x

            @staticmethod
            def YELLOW(x): return x

            @staticmethod
            def CYAN(x): return x

            @staticmethod
            def BLUE(x): return x


# =================== ANSI Color Codes ===================

class Colors:
    """ANSI color codes for terminal styling"""
    # Basic colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    GREY = '\033[90m'

    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'

    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

    # Reset
    RESET = '\033[0m'


# =================== Visual Width Calculation ===================

def _get_visual_width(text: str) -> int:
    """Calculate visual width of text, accounting for emojis and wide characters"""
    single_width_symbols = {
        '✓', '✗', '⚠', 'ℹ', '•', '⟳', '⏳'
    }

    visual_width = 0
    i = 0
    while i < len(text):
        char = text[i]
        cp = ord(char)

        has_variation_selector = (i + 1 < len(text) and 0xFE00 <= ord(text[i + 1]) <= 0xFE0F)

        if 0xFE00 <= cp <= 0xFE0F:
            i += 1
            continue

        if char in single_width_symbols:
            visual_width += 1
        elif (
            unicodedata.east_asian_width(char) in ('F', 'W') or
            (0x1F300 <= cp <= 0x1F9FF) or
            (0x1F000 <= cp <= 0x1FAFF) or
            has_variation_selector
        ):
            visual_width += 2
            if has_variation_selector:
                i += 1
        else:
            visual_width += 1

        i += 1

    return visual_width


def _strip_ansi(text: str) -> str:
    """Remove ANSI color codes from text for length calculation"""
    import re
    ansi_escape = re.compile(r'\033\[[0-9;]*m')
    return ansi_escape.sub('', text)


# =================== Header Functions ===================

def print_box_header(title: str, icon: str = "ℹ", width: int = 76):
    """Print a minimal styled header"""
    print()
    print(f"{Colors.BOLD}{icon} {title}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * width}{Colors.RESET}")


def print_box_footer(width: int = 76):
    """Print a minimal footer"""
    print()


# =================== Content Display Functions ===================

def print_box_content(text: str, style: str = "", width: int = 76, auto_wrap: bool = True):
    """Print content with minimal styled prefix"""
    style_config = {
        'success': {'icon': '✓', 'color': Colors.GREEN},
        'error': {'icon': '✗', 'color': Colors.RED},
        'warning': {'icon': '⚠', 'color': Colors.YELLOW},
        'info': {'icon': 'ℹ', 'color': Colors.BLUE},
    }

    if style in style_config:
        config = style_config[style]
        print(f"  {config['color']}{config['icon']}{Colors.RESET} {text}")
    else:
        print(f"  {text}")


def print_code_block(code: str, language: str = "text", width: int = 76, show_line_numbers: bool = False):
    """Print code block with minimal syntax highlighting"""
    import json

    if language.lower() in ['json']:
        try:
            parsed = json.loads(code) if isinstance(code, str) else code
            formatted = json.dumps(parsed, indent=2)
            lines = formatted.split('\n')
        except:
            lines = code.split('\n')
    elif language.lower() in ['yaml', 'yml']:
        lines = code.split('\n')
        formatted_lines = []
        for line in lines:
            if ':' in line and not line.strip().startswith('#'):
                key, value = line.split(':', 1)
                formatted_lines.append(f"{Colors.CYAN}{key}{Colors.RESET}:{value}")
            elif line.strip().startswith('#'):
                formatted_lines.append(f"{Colors.DIM}{line}{Colors.RESET}")
            else:
                formatted_lines.append(line)
        lines = formatted_lines
    elif language.lower() in ['toml']:
        lines = code.split('\n')
        formatted_lines = []
        for line in lines:
            if line.strip().startswith('[') and line.strip().endswith(']'):
                formatted_lines.append(f"{Colors.BOLD}{line}{Colors.RESET}")
            elif '=' in line and not line.strip().startswith('#'):
                key, value = line.split('=', 1)
                formatted_lines.append(f"{Colors.CYAN}{key}{Colors.RESET}={value}")
            elif line.strip().startswith('#'):
                formatted_lines.append(f"{Colors.DIM}{line}{Colors.RESET}")
            else:
                formatted_lines.append(line)
        lines = formatted_lines
    elif language.lower() in ['env', 'dotenv']:
        lines = code.split('\n')
        formatted_lines = []
        for line in lines:
            if '=' in line and not line.strip().startswith('#'):
                key, value = line.split('=', 1)
                formatted_lines.append(f"{Colors.CYAN}{key}{Colors.RESET}={value}")
            elif line.strip().startswith('#'):
                formatted_lines.append(f"{Colors.DIM}{line}{Colors.RESET}")
            else:
                formatted_lines.append(line)
        lines = formatted_lines
    else:
        lines = code.split('\n')

    for i, line in enumerate(lines, 1):
        if show_line_numbers:
            print(f"  {Colors.DIM}{i:3d}{Colors.RESET} {line}")
        else:
            print(f"  {line}")


# =================== Status Messages ===================

def print_status(message: str, status: str = "info"):
    """Print a minimal status message with icon and color"""
    status_config = {
        'success': {'icon': '✓', 'color': Colors.GREEN},
        'error': {'icon': '✗', 'color': Colors.RED},
        'warning': {'icon': '⚠', 'color': Colors.YELLOW},
        'info': {'icon': 'ℹ', 'color': Colors.BLUE},
        'progress': {'icon': '⟳', 'color': Colors.CYAN},
        'waiting': {'icon': '⏳', 'color': Colors.MAGENTA},
        'launch': {'icon': '🚀', 'color': Colors.GREEN},
        'install': {'icon': '📦', 'color': Colors.CYAN},
        'download': {'icon': '⬇️', 'color': Colors.BLUE},
        'upload': {'icon': '⬆️', 'color': Colors.MAGENTA},
        'connect': {'icon': '🔗', 'color': Colors.GREEN},
        'disconnect': {'icon': '🔌', 'color': Colors.RED},
        'configure': {'icon': '🔧', 'color': Colors.YELLOW},
        'debug': {'icon': '🐞', 'color': Colors.MAGENTA},
        'test': {'icon': '🧪', 'color': Colors.GREEN},
        'analyze': {'icon': '🔍', 'color': Colors.BLUE},
        'data': {'icon': '💾', 'color': Colors.YELLOW},
        'database': {'icon': '🗃️', 'color': Colors.MAGENTA},
        'server': {'icon': '🖥️', 'color': Colors.GREEN},
        'network': {'icon': '🌐', 'color': Colors.BLUE},
        'build': {'icon': '🔨', 'color': Colors.CYAN},
        'update': {'icon': '🔄', 'color': Colors.MAGENTA}
    }

    config = status_config.get(status, {'icon': '•', 'color': ''})

    print(f"{config['color']}{config['icon']}{Colors.RESET} {message}")


# =================== Separators ===================

def print_separator(char: str = "─", width: int = 76):
    """Print a minimal separator line"""
    print(f"{Colors.DIM}{char * width}{Colors.RESET}")


# =================== Table Printing ===================

def print_table_header(columns: list, widths: list):
    """Print a table header with columns"""
    header_parts = []
    for (name, _), width in zip(columns, widths):
        header_parts.append(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}{name:<{width}}{Colors.RESET}")

    print(f"  {' │ '.join(header_parts)}")

    sep_parts = [f"{Colors.BRIGHT_CYAN}{'─' * w}{Colors.RESET}" for w in widths]
    print(f"  {f'{Colors.BRIGHT_CYAN}─┼─{Colors.RESET}'.join(sep_parts)}")


def print_table_row(values: list, widths: list, styles: list = None):
    """Print a table row"""
    if styles is None:
        styles = [""] * len(values)

    color_map = {
        'grey': Colors.GREY,
        'white': Colors.WHITE,
        'green': Colors.BRIGHT_GREEN,
        'yellow': Colors.BRIGHT_YELLOW,
        'cyan': Colors.BRIGHT_CYAN,
        'blue': Colors.BRIGHT_BLUE,
        'red': Colors.BRIGHT_RED,
        'magenta': Colors.BRIGHT_MAGENTA,
    }

    row_parts = []
    for value, width, style in zip(values, widths, styles):
        color = color_map.get(style.lower(), '')
        if color:
            colored_value = f"{color}{value}{Colors.RESET}"
            padding = width - len(value)
            row_parts.append(colored_value + " " * padding)
        else:
            row_parts.append(f"{value:<{width}}")

    print(f"  {f' {Colors.DIM}│{Colors.RESET} '.join(row_parts)}")


# =================== Visual Test ===================

def run_visual_test():
    """Visual test for all UI components - for alignment and testing"""
    print("\n" + f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE} VISUAL TEST - CLI UI COMPONENTS ".center(80, '='))
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 80}{Colors.RESET}\n")

    # Test 1: Headers with different icons
    print(f"{Colors.BOLD}TEST 1: Headers with Different Icons{Colors.RESET}")
    print(f"{Colors.DIM}{'-' * 80}{Colors.RESET}")

    print_box_header("Standard Info Header", "ℹ")
    print_box_footer()

    print_box_header("Success Header", "✓")
    print_box_footer()

    print_box_header("Server Header", "🖥️")
    print_box_footer()

    # Test 2: Content with different styles
    print(f"\n{Colors.BOLD}TEST 2: Content with Different Styles{Colors.RESET}")
    print(f"{Colors.DIM}{'-' * 80}{Colors.RESET}")
    print_box_header("Content Styles Test", "🎨")
    print_box_content("This is a plain text without style")
    print_box_content("This is a success message", "success")
    print_box_content("This is an error message", "error")
    print_box_content("This is a warning message", "warning")
    print_box_content("This is an info message", "info")
    print_box_footer()

    # Test 3: Combined content
    print(f"\n{Colors.BOLD}TEST 3: Combined Content{Colors.RESET}")
    print(f"{Colors.DIM}{'-' * 80}{Colors.RESET}")
    print_box_header("Server Status Example", "🖥️")
    print_box_content("Server Name: ToolBoxV2 API Server", "info")
    print_box_content("Status: Running", "success")
    print_box_content("Port: 8080", "info")
    print_box_content("Warning: High memory usage detected", "warning")
    print_box_content("Error: Connection timeout on endpoint /api/test", "error")
    print_box_content("Plain text information line")
    print_box_footer()

    # Test 4: Status messages
    print(f"\n{Colors.BOLD}TEST 4: Status Messages{Colors.RESET}")
    print(f"{Colors.DIM}{'-' * 80}{Colors.RESET}")
    print_status("Success status message", "success")
    print_status("Error status message", "error")
    print_status("Warning status message", "warning")
    print_status("Info status message", "info")
    print_status("Progress status message", "progress")
    print_status("Server status message", "server")
    print_status("Build status message", "build")
    print_status("Update status message", "update")

    # Test 5: Separators
    print(f"\n{Colors.BOLD}TEST 5: Separators{Colors.RESET}")
    print(f"{Colors.DIM}{'-' * 80}{Colors.RESET}")
    print_separator("─")
    print_separator("═")
    print_separator("━")

    # Test 6: Tables
    print(f"\n{Colors.BOLD}TEST 6: Table Display{Colors.RESET}")
    print(f"{Colors.DIM}{'-' * 80}{Colors.RESET}")
    columns = [
        ("Property", 20),
        ("Value", 30),
        ("Status", 20)
    ]
    widths = [w for _, w in columns]

    print_table_header(columns, widths)
    print_table_row(["Server Name", "ToolBoxV2 API", "Active"], widths, ["white", "cyan", "green"])
    print_table_row(["PID", "12345", "Running"], widths, ["white", "grey", "green"])
    print_table_row(["Version", "1.0.0", "Latest"], widths, ["white", "yellow", "green"])
    print_table_row(["Port", "8080", "Open"], widths, ["white", "blue", "green"])

    # Test 7: Code blocks
    print(f"\n\n{Colors.BOLD}TEST 7: Code & Config File Display{Colors.RESET}")
    print(f"{Colors.DIM}{'-' * 80}{Colors.RESET}")

    print_box_header("JSON Configuration", "📄")
    json_example = '''{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "debug": true
  },
  "database": {
    "url": "postgresql://localhost/mydb",
    "pool_size": 10
  }
}'''
    print_code_block(json_example, "json", show_line_numbers=True)
    print_box_footer()

    print_box_header("Environment Variables", "📄")
    env_example = '''# Application Settings
APP_NAME=ToolBoxV2
APP_ENV=production
DEBUG=false

# Database
DATABASE_URL=postgresql://localhost/mydb'''
    print_code_block(env_example, "env")
    print_box_footer()

    # Test 8: Real-world example
    print(f"\n{Colors.BOLD}TEST 8: Real-World Server Start Example{Colors.RESET}")
    print(f"{Colors.DIM}{'-' * 80}{Colors.RESET}")
    print_box_header("Starting Server v1.2.3", "🚀")
    print_box_content("Executable: /usr/local/bin/simple-core-server", "info")
    print_box_content("Host: 0.0.0.0:8080", "info")
    print_box_content("Mode: POSIX Zero-Downtime", "info")
    print_box_footer()

    print_status("Launching server", "progress")
    print_status("Socket created - FD 3 saved to server_socket.fd", "success")
    print()

    print_box_header("Server Started", "✓")
    print_box_content("Version: 1.2.3", "success")
    print_box_content("PID: 54321", "success")
    print_box_content("Port: 8080", "success")
    print_box_footer()

    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE} END OF VISUAL TEST ".center(80, '='))
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 80}{Colors.RESET}\n")


# =================== CLI Entry Point for Testing ===================

def main():
    """Entry point for running visual test directly"""
    run_visual_test()


if __name__ == "__main__":
    main()
