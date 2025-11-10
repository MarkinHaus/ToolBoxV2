import itertools
import os
import re
import sys
import threading
import time
from json import JSONDecoder
from platform import system
from random import uniform
from time import sleep

from ..singelton_class import Singleton


def stram_print(text):
    min_typing_speed, max_typing_speed = 0.0009, 0.0005
    print_to_console(
        "",
        "",
        text,
        min_typing_speed=min_typing_speed,
        max_typing_speed=max_typing_speed, auto_new_line=False)


def print_prompt(msg_data):
    messages = msg_data.get('massages', msg_data.get('messages', [])) if isinstance(msg_data, dict) else msg_data
    if len(messages) == 0:
        print(Style.YELLOW("NO PROMPT to print"))
        return
    print(Style.GREEN2("PROMPT START "))
    for message in messages:
        caller = Style.WHITE(message.get('role', 'NONE').upper()) if message.get('role',
                                                                                 'NONE') == 'user' else 'NONE'
        caller = Style.CYAN(message.get('role', 'NONE').upper()) if message.get('role',
                                                                                'NONE') == 'system' else caller
        caller = Style.VIOLET2(message.get('role', 'NONE').upper()) if message.get('role',
                                                                                   'NONE') == 'assistant' else caller
        print(f"\n{caller}\n{Style.GREY(str(message.get('content', '--#--')))}\n")
    print(Style.GREEN("PROMPT END -- "))



def cls():
    if system() == "Windows":
        os.system("cls")
    if system() == "Linux":
        os.system("clear")


def text_save(function):
    def deco(text):
        text = str(text).encode(sys.stdout.encoding or 'utf-8', 'replace').decode(sys.stdout.encoding or 'utf-8')
        return function(text)
    return deco


class Style:
    _END = '\33[0m'
    _BLACK = '\33[30m'
    _RED = '\33[31m'
    _GREEN = '\33[32m'
    _YELLOW = '\33[33m'
    _BLUE = '\33[34m'
    _MAGENTA = '\33[35m'
    _CYAN = '\33[36m'
    _WHITE = '\33[37m'

    _Bold = '\33[1m'
    _ITALIC = '\33[3m'
    _Underline = '\33[4m'
    _BLINK = '\33[5m'
    _BLINK2 = '\33[6m'
    _Reversed = '\33[7m'

    _BLACKBG = '\33[40m'
    _REDBG = '\33[41m'
    _GREENBG = '\33[42m'
    _YELLOWBG = '\33[43m'
    _BLUEBG = '\33[44m'
    _VIOLETBG = '\33[45m'
    _BEIGEBG = '\33[46m'
    _WHITEBG = '\33[47m'

    _GREY = '\33[90m'
    _RED2 = '\33[91m'
    _GREEN2 = '\33[92m'
    _YELLOW2 = '\33[93m'
    _BLUE2 = '\33[94m'
    _VIOLET2 = '\33[95m'
    _BEIGE2 = '\33[96m'
    _WHITE2 = '\33[97m'

    _GREYBG = '\33[100m'
    _REDBG2 = '\33[101m'
    _GREENBG2 = '\33[102m'
    _YELLOWBG2 = '\33[103m'
    _BLUEBG2 = '\33[104m'
    _VIOLETBG2 = '\33[105m'
    _BEIGEBG2 = '\33[106m'
    _WHITEBG2 = '\33[107m'

    style_dic = {
        "END": _END,
        "BLACK": _BLACK,
        "RED": _RED,
        "GREEN": _GREEN,
        "YELLOW": _YELLOW,
        "BLUE": _BLUE,
        "MAGENTA": _MAGENTA,
        "CYAN": _CYAN,
        "WHITE": _WHITE,
        "Bold": _Bold,
        "Underline": _Underline,
        "Reversed": _Reversed,

        "ITALIC": _ITALIC,
        "BLINK": _BLINK,
        "BLINK2": _BLINK2,
        "BLACKBG": _BLACKBG,
        "REDBG": _REDBG,
        "GREENBG": _GREENBG,
        "YELLOWBG": _YELLOWBG,
        "BLUEBG": _BLUEBG,
        "VIOLETBG": _VIOLETBG,
        "BEIGEBG": _BEIGEBG,
        "WHITEBG": _WHITEBG,
        "GRAY": _GREY,
        "GREY": _GREY,
        "RED2": _RED2,
        "GREEN2": _GREEN2,
        "YELLOW2": _YELLOW2,
        "BLUE2": _BLUE2,
        "VIOLET2": _VIOLET2,
        "BEIGE2": _BEIGE2,
        "WHITE2": _WHITE2,
        "GREYBG": _GREYBG,
        "REDBG2": _REDBG2,
        "GREENBG2": _GREENBG2,
        "YELLOWBG2": _YELLOWBG2,
        "BLUEBG2": _BLUEBG2,
        "VIOLETBG2": _VIOLETBG2,
        "BEIGEBG2": _BEIGEBG2,
        "WHITEBG2": _WHITEBG2,

    }

    @staticmethod
    @text_save
    def END_():
        print(Style._END)

    @staticmethod
    @text_save
    def GREEN_():
        print(Style._GREEN)

    @staticmethod
    @text_save
    def BLUE(text: str):
        return Style._BLUE + text + Style._END

    @staticmethod
    @text_save
    def BLACK(text: str):
        return Style._BLACK + text + Style._END

    @staticmethod
    @text_save
    def RED(text: str):
        return Style._RED + text + Style._END

    @staticmethod
    @text_save
    def GREEN(text: str):
        return Style._GREEN + text + Style._END

    @staticmethod
    @text_save
    def YELLOW(text: str):
        return Style._YELLOW + text + Style._END

    @staticmethod
    @text_save
    def MAGENTA(text: str):
        return Style._MAGENTA + text + Style._END

    @staticmethod
    @text_save
    def CYAN(text: str):
        return Style._CYAN + text + Style._END

    @staticmethod
    @text_save
    def WHITE(text: str):
        return Style._WHITE + text + Style._END

    @staticmethod
    @text_save
    def Bold(text: str):
        return Style._Bold + text + Style._END

    @staticmethod
    @text_save
    def Underline(text: str):
        return Style._Underline + text + Style._END

    @staticmethod
    @text_save
    def Underlined(text: str):
        return Style._Underline + text + Style._END

    @staticmethod
    @text_save
    def Reversed(text: str):
        return Style._Reversed + text + Style._END

    @staticmethod
    @text_save
    def ITALIC(text: str):
        return Style._ITALIC + text + Style._END

    @staticmethod
    @text_save
    def BLINK(text: str):
        return Style._BLINK + text + Style._END

    @staticmethod
    @text_save
    def BLINK2(text: str):
        return Style._BLINK2 + text + Style._END

    @staticmethod
    @text_save
    def BLACKBG(text: str):
        return Style._BLACKBG + text + Style._END

    @staticmethod
    @text_save
    def REDBG(text: str):
        return Style._REDBG + text + Style._END

    @staticmethod
    @text_save
    def GREENBG(text: str):
        return Style._GREENBG + text + Style._END

    @staticmethod
    @text_save
    def YELLOWBG(text: str):
        return Style._YELLOWBG + text + Style._END

    @staticmethod
    @text_save
    def BLUEBG(text: str):
        return Style._BLUEBG + text + Style._END

    @staticmethod
    @text_save
    def VIOLETBG(text: str):
        return Style._VIOLETBG + text + Style._END

    @staticmethod
    @text_save
    def BEIGEBG(text: str):
        return Style._BEIGEBG + text + Style._END

    @staticmethod
    @text_save
    def WHITEBG(text: str):
        return Style._WHITEBG + text + Style._END

    @staticmethod
    @text_save
    def GREY(text: str):
        return Style._GREY + str(text) + Style._END

    @staticmethod
    @text_save
    def RED2(text: str):
        return Style._RED2 + text + Style._END

    @staticmethod
    @text_save
    def GREEN2(text: str):
        return Style._GREEN2 + text + Style._END

    @staticmethod
    @text_save
    def YELLOW2(text: str):
        return Style._YELLOW2 + text + Style._END

    @staticmethod
    @text_save
    def BLUE2(text: str):
        return Style._BLUE2 + text + Style._END

    @staticmethod
    @text_save
    def VIOLET2(text: str):
        return Style._VIOLET2 + text + Style._END

    @staticmethod
    @text_save
    def BEIGE2(text: str):
        return Style._BEIGE2 + text + Style._END

    @staticmethod
    @text_save
    def WHITE2(text: str):
        return Style._WHITE2 + text + Style._END

    @staticmethod
    @text_save
    def GREYBG(text: str):
        return Style._GREYBG + text + Style._END

    @staticmethod
    @text_save
    def REDBG2(text: str):
        return Style._REDBG2 + text + Style._END

    @staticmethod
    @text_save
    def GREENBG2(text: str):
        return Style._GREENBG2 + text + Style._END

    @staticmethod
    @text_save
    def YELLOWBG2(text: str):
        return Style._YELLOWBG2 + text + Style._END

    @staticmethod
    @text_save
    def BLUEBG2(text: str):
        return Style._BLUEBG2 + text + Style._END

    @staticmethod
    @text_save
    def VIOLETBG2(text: str):
        return Style._VIOLETBG2 + text + Style._END

    @staticmethod
    @text_save
    def BEIGEBG2(text: str):
        return Style._BEIGEBG2 + text + Style._END

    @staticmethod
    @text_save
    def WHITEBG2(text: str):
        return Style._WHITEBG2 + text + Style._END

    @staticmethod
    @text_save
    def loading_al(text: str):
        b = f"{text} /"
        print(b)
        sleep(0.05)
        cls()
        b = f"{text} -"
        print(b)
        sleep(0.05)
        cls()
        b = f"{text} \\"
        print(b)
        sleep(0.05)
        cls()
        b = f"{text} |"
        print(b)
        sleep(0.05)
        cls()

    @property
    def END(self):
        return self._END

    def color_demo(self):
        for color in self.style_dic:
            print(f"{color} -> {self.style_dic[color]}Effect{self._END}")

    @property
    def Underline2(self):
        return self._Underline

    def style_text(self, text, color, bold=False):
        text = self.style_dic.get(color, 'WHITE') + text + self._END
        if bold:
            text = self._Bold + text + self._END
        return text

def remove_styles(text: str, infos=False):
    in_ = []
    for key, style in Style.style_dic.items():
        if style in text:
            text = text.replace(style, '')
            if infos:
                in_.append([key for key, st in Style.style_dic.items() if style == st][0])
    if infos:
        if "END" in in_:
            in_.remove('END')
        return text, in_
    return text


def print_to_console(
    title,
    title_color,
    content,
    min_typing_speed=0.05,
    max_typing_speed=0.01, auto_new_line=True):
    print(title_color + title + Style.BLUE("") + " ", end="")
    if content:
        if isinstance(content, list):
            content = " ".join(content)
        if not isinstance(content, str):
            print(f"SYSTEM NO STR type : {type(content)}")
            print(content)
            return
        words = content.split()
        if len(words) > 5000:
            words = words[:5000]

        min_typing_speed = min_typing_speed * 0.01
        max_typing_speed = max_typing_speed * 0.01
        for i, word in enumerate(words):
            print(word, end="", flush=True)
            if i < len(words) - 1:
                print(" ", end="", flush=True)
            typing_speed = uniform(min_typing_speed, max_typing_speed)
            time.sleep(typing_speed)
            # type faster after each word
            min_typing_speed = min_typing_speed * 0.95
            max_typing_speed = max_typing_speed * 0.95
    if auto_new_line:
        print()


class JSONExtractor(JSONDecoder):
    def decode(self, s, _w=None):
        self.raw_decode(s)
        return s

    def raw_decode(self, s, _w=None):
        try:
            obj, end = super().raw_decode(s)
            return obj, end
        except ValueError:
            return None, 0


def extract_json_strings(text):
    json_strings = []
    extractor = JSONExtractor()

    # Ersetzt einfache Anführungszeichen durch doppelte Anführungszeichen
    text = re.sub(r"(?<!\\)'", "\"", text)

    start = 0
    while True:
        match = re.search(r'\{', text[start:])
        if not match:
            break

        start += match.start()
        decoded, end = extractor.raw_decode(text[start:])
        if decoded is not None:
            json_strings.append(text[start:start + end])
            start += end
        else:
            start += 1

    return json_strings


def extract_python_code(text):
    python_code_blocks = []

    # Finden Sie alle Codeblöcke, die mit ```python beginnen und mit ``` enden
    code_blocks = re.findall(r'```python(.*?)```', text, re.DOTALL)

    for block in code_blocks:
        # Entfernen Sie führende und nachfolgende Leerzeichen
        block = block.strip()
        python_code_blocks.append(block)

    return python_code_blocks




import signal


class SpinnerManager(metaclass=Singleton):
    """
    Manages multiple spinners to ensure tqdm-like line rendering.
    Automatically captures SIGINT (Ctrl+C) to stop all spinners.
    """
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._init_manager()
        return cls._instance

    def _init_manager(self):
        """Initialize spinner management resources and register SIGINT handler."""
        self._spinners = []
        self._lock = threading.Lock()
        self._render_thread = None
        self._should_run = False
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
        except ValueError:
            print("Spinner Manager not in the min Thread no signal possible")
            pass

    def _signal_handler(self, signum, frame):
        """Handle SIGINT by stopping all spinners gracefully."""
        with self._lock:
            for spinner in self._spinners:
                spinner.running = False
            self._spinners.clear()
        self._should_run = False
        sys.stdout.write("\r\033[K")  # Clear the spinner's line.
        sys.stdout.flush()
        sys.exit(0)

    def register_spinner(self, spinner):
        """Register a new spinner."""
        with self._lock:
            # First spinner defines the rendering line.
            if not self._spinners:
                spinner._is_primary = True
            self._spinners.append(spinner)
            # Start rendering if not already running.
            if not self._should_run:
                self._should_run = True
                self._render_thread = threading.Thread(
                    target=self._render_loop,
                    daemon=True
                )
                self._render_thread.start()

    def unregister_spinner(self, spinner):
        """Unregister a completed spinner."""
        with self._lock:
            if spinner in self._spinners:
                self._spinners.remove(spinner)

    def _render_loop(self):
        """Continuous rendering loop for all active spinners."""
        while self._should_run:
            if not self._spinners:
                self._should_run = False
                break

            with self._lock:
                # Find primary spinner (first registered).
                primary_spinner = next((s for s in self._spinners if s._is_primary), None)

                if primary_spinner and primary_spinner.running:
                    # Render in the same line.
                    render_line = primary_spinner._generate_render_line()

                    # Append additional spinner info if multiple exist.
                    if len(self._spinners) > 1:
                        secondary_info = " | ".join(
                            s._generate_secondary_info()
                            for s in self._spinners
                            if s is not primary_spinner and s.running
                        )
                        render_line += f" [{secondary_info}]"

                    # Clear line and write.
                    try:
                        sys.stdout.write("\r" + render_line + "\033[K")
                        sys.stdout.flush()
                    except Exception:
                        self._should_run = False

            time.sleep(0.1)  # Render interval.

class Spinner:
    """
    Enhanced Spinner with tqdm-like line rendering.
    """
    SYMBOL_SETS = {
        "c": ["◐", "◓", "◑", "◒"],
        "b": ["▁", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃"],
        "d": ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"],
        "w": ["🌍", "🌎", "🌏"],
        "s": ["🌀   ", " 🌀  ", "  🌀 ", "   🌀", "  🌀 ", " 🌀  "],
        "+": ["+", "x"],
        "t": ["✶", "✸", "✹", "✺", "✹", "✷"]
    }

    def __init__(
        self,
        message: str = "Loading...",
        delay: float = 0.1,
        symbols=None,
        count_down: bool = False,
        time_in_s: float = 0
    ):
        """Initialize spinner with flexible configuration."""
        # Resolve symbol set.
        if isinstance(symbols, str):
            symbols = self.SYMBOL_SETS.get(symbols, None)

        # Default symbols if not provided.
        if symbols is None:
            symbols = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        # Test mode symbol set.
        if 'unittest' in sys.argv[0]:
            symbols = ['#', '=', '-']

        self.spinner = itertools.cycle(symbols)
        self.delay = delay
        self.message = message
        self.running = False
        self.spinner_thread = None
        self.max_t = time_in_s
        self.contd = count_down

        # Rendering management.
        self._is_primary = False
        self._start_time = 0

        # Central manager.
        self.manager = SpinnerManager()

    def _generate_render_line(self):
        """Generate the primary render line."""
        current_time = time.time()
        if self.contd:
            remaining = max(0, self.max_t - (current_time - self._start_time))
            time_display = f"{remaining:.2f}"
        else:
            time_display = f"{current_time - self._start_time:.2f}"

        symbol = next(self.spinner)
        return f"{symbol} {self.message} | {time_display}"

    def _generate_secondary_info(self):
        """Generate secondary spinner info for additional spinners."""
        return f"{self.message}"

    def __enter__(self):
        """Start the spinner."""
        self.running = True
        self._start_time = time.time()
        self.manager.register_spinner(self)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Stop the spinner."""
        self.running = False
        self.manager.unregister_spinner(self)
        # Clear the spinner's line if it was the primary spinner.
        if self._is_primary:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
