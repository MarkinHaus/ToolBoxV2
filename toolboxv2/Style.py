import os
from platform import system
from time import sleep


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
        "BLACK": _BLACK,
        "RED": _RED,
        "GREEN": _GREEN,
        "YELLOW": _YELLOW,
        "BLUE": _BLUE,
        "MAGENTA": _MAGENTA,
        "CYAN": _CYAN,
        "WHITE": _WHITE,
        "END": _END,
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

# print(Style().color_demo())
