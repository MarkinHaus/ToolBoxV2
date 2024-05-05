import os
from json.decoder import WHITESPACE
from random import uniform
import re
import time
from json import JSONDecoder
from platform import system
from time import sleep
import itertools
import sys
import threading


def stram_print(text):
    min_typing_speed, max_typing_speed = 0.0009, 0.0005
    print_to_console(
        "",
        "",
        text,
        min_typing_speed=min_typing_speed,
        max_typing_speed=max_typing_speed, auto_new_line=False)


def cls():
    if system() == "Windows":
        os.system("cls")
    if system() == "Linux":
        os.system("clear")


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
    def END_():
        print(Style._END)

    @staticmethod
    def GREEN_():
        print(Style._GREEN)

    @staticmethod
    def BLUE(text: str):
        return Style._BLUE + text + Style._END

    @staticmethod
    def BLACK(text: str):
        return Style._BLACK + text + Style._END

    @staticmethod
    def RED(text: str):
        return Style._RED + text + Style._END

    @staticmethod
    def GREEN(text: str):
        return Style._GREEN + text + Style._END

    @staticmethod
    def YELLOW(text: str):
        return Style._YELLOW + text + Style._END

    @staticmethod
    def MAGENTA(text: str):
        return Style._MAGENTA + text + Style._END

    @staticmethod
    def CYAN(text: str):
        return Style._CYAN + text + Style._END

    @staticmethod
    def WHITE(text: str):
        return Style._WHITE + text + Style._END

    @staticmethod
    def Bold(text: str):
        return Style._Bold + text + Style._END

    @staticmethod
    def Underline(text: str):
        return Style._Underline + text + Style._END

    @staticmethod
    def Reversed(text: str):
        return Style._Reversed + text + Style._END

    @staticmethod
    def ITALIC(text: str):
        return Style._ITALIC + text + Style._END

    @staticmethod
    def BLINK(text: str):
        return Style._BLINK + text + Style._END

    @staticmethod
    def BLINK2(text: str):
        return Style._BLINK2 + text + Style._END

    @staticmethod
    def BLACKBG(text: str):
        return Style._BLACKBG + text + Style._END

    @staticmethod
    def REDBG(text: str):
        return Style._REDBG + text + Style._END

    @staticmethod
    def GREENBG(text: str):
        return Style._GREENBG + text + Style._END

    @staticmethod
    def YELLOWBG(text: str):
        return Style._YELLOWBG + text + Style._END

    @staticmethod
    def BLUEBG(text: str):
        return Style._BLUEBG + text + Style._END

    @staticmethod
    def VIOLETBG(text: str):
        return Style._VIOLETBG + text + Style._END

    @staticmethod
    def BEIGEBG(text: str):
        return Style._BEIGEBG + text + Style._END

    @staticmethod
    def WHITEBG(text: str):
        return Style._WHITEBG + text + Style._END

    @staticmethod
    def GREY(text: str):
        return Style._GREY + text + Style._END

    @staticmethod
    def RED2(text: str):
        return Style._RED2 + text + Style._END

    @staticmethod
    def GREEN2(text: str):
        return Style._GREEN2 + text + Style._END

    @staticmethod
    def YELLOW2(text: str):
        return Style._YELLOW2 + text + Style._END

    @staticmethod
    def BLUE2(text: str):
        return Style._BLUE2 + text + Style._END

    @staticmethod
    def VIOLET2(text: str):
        return Style._VIOLET2 + text + Style._END

    @staticmethod
    def BEIGE2(text: str):
        return Style._BEIGE2 + text + Style._END

    @staticmethod
    def WHITE2(text: str):
        return Style._WHITE2 + text + Style._END

    @staticmethod
    def GREYBG(text: str):
        return Style._GREYBG + text + Style._END

    @staticmethod
    def REDBG2(text: str):
        return Style._REDBG2 + text + Style._END

    @staticmethod
    def GREENBG2(text: str):
        return Style._GREENBG2 + text + Style._END

    @staticmethod
    def YELLOWBG2(text: str):
        return Style._YELLOWBG2 + text + Style._END

    @staticmethod
    def BLUEBG2(text: str):
        return Style._BLUEBG2 + text + Style._END

    @staticmethod
    def VIOLETBG2(text: str):
        return Style._VIOLETBG2 + text + Style._END

    @staticmethod
    def BEIGEBG2(text: str):
        return Style._BEIGEBG2 + text + Style._END

    @staticmethod
    def WHITEBG2(text: str):
        return Style._WHITEBG2 + text + Style._END

    @staticmethod
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
        for color in self.style_dic.keys():
            print(f"{color} -> {self.style_dic[color]}Effect{self._END}")

    @property
    def Underline2(self):
        return self._Underline


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
    def decode(self, s, _w=WHITESPACE.match):
        self.raw_decode(s)
        return s

    def raw_decode(self, s, _w=WHITESPACE.match):
        try:
            obj, end = super().raw_decode(s)
            return obj, end
        except ValueError as e:
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


class Spinner:
    """A simple spinner class"""

    def __init__(self, message: str = "Loading...", delay: float = 0.1, symbols=None, count_down=False,
                 time_in_s=0) -> None:
        """Initialize the spinner class
        Args:
            message (str): The message to display.
            delay (float): The delay between each spinner update.
        """

        if isinstance(symbols, str):
            if symbols == "c":
                symbols = ["◐", "◓", "◑", "◒"]
            elif symbols == "b":
                symbols = ["▁", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃"]
            elif symbols == "d":
                symbols = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
            elif symbols == "w":
                symbols = ["🌍", "🌎", "🌏"]
            elif symbols == "s":
                symbols = ["🌀   ", " 🌀  ", "  🌀 ", "   🌀", "  🌀 ", " 🌀  "]
            elif symbols == "+":
                symbols = ["+", "x"]
            elif symbols == "t":
                symbols = ["✶", "✸", "✹", "✺", "✹", "✷"]
            else:
                symbols = None

        if symbols is None:
            symbols = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        self.spinner = itertools.cycle(symbols)
        self.delay = delay
        self.message = message
        self.running = False
        self.spinner_thread = None
        self.max_t = time_in_s
        self.contd = count_down

    def __enter__(self) -> None:
        """Start the spinner"""
        self.running = True
        self.spinner_thread = threading.Thread(target=self.spin, daemon=True)
        self.spinner_thread.start()

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        """Stop the spinner"""
        self.running = False
        if self.spinner_thread is not None:
            self.spinner_thread.join()
        sys.stdout.write(f"\r{' ' * len(self.message)}\r")
        sys.stdout.flush()

    def spin(self) -> None:
        """Spin the spinner"""
        t0 = time.time()
        while self.running:
            if self.contd:
                _ = self.max_t - (time.time() - t0)
                extra = f"{_:.2f}"
                if _ < 0:
                    self.contd = False
                    t0 = time.time()
            else:
                extra = f"{time.time() - t0:.2f}"
            sys.stdout.write(f"\r{next(self.spinner)} {self.message} | {extra} ")
            sys.stdout.flush()
            time.sleep(self.delay)
