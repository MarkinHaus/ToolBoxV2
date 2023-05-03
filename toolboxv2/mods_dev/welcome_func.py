import sys

from toolboxv2 import MainTool, App, Style
from time import sleep
from platform import system
import os


class Tools(MainTool):

    def __init__(self, app=None):
        self.version = "0.0.1"
        self.name = "welcome"
        self.color = "YELLOW"
        self.logs = app.logger if app else None
        self.tools = {
            "all": [["Version", "Shows current Version "],
                    ["printT", "print TOOL BOX"]],
            "name": "welcome",
            "Version": show_version,
            "printT": print_t}

        MainTool.__init__(self, load=print_t, v=self.version, tool=self.tools,
                          name=self.name, logs=self.logs, color=self.color, on_exit=lambda: "")


def get_tool(app: App):
    return app.AC_MOD


def show_version(c, app: App):
    welcome_f: Tools = get_tool(app)
    welcome_f.print(f"Version: {welcome_f.version}")
    return welcome_f.version


def cls():
    if system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


def print_t(_, app: App):
    print()
    welcome_f = get_tool(app)
    printc("**************************************************************************")
    printc("***████████╗*██████╗***██████╗**██╗*********██████╗***██████╗*██╗***██╗***")
    printc("***╚══██╔══╝██╔═══██╗*██╔═══██╗*██║*********██╔══██╗*██╔═══██╗*╚██╗██╔╝***")
    printc("******██║***██║***██║*██║***██║*██║*********██████╔╝*██║***██║**╚███╔╝****")
    printc("******██║***██║***██║*██║***██║*██║*********██╔══██╗*██║***██║**██╔██╗****")
    printc("******██║***╚██████╔╝*╚██████╔╝*███████╗****██████╔╝*╚██████╔╝*██╔╝*██╗***")
    printc("******╚═╝****╚═════╝***╚═════╝**╚══════╝****╚═════╝***╚═════╝**╚═╝**╚═╝***")
    printc("**************************************************************************")
    welcome_f.print(f"Version: {welcome_f.version} : {system()}")
    return "TOOL BOX"


def printc(str_):
    if 'unittest' in sys.argv[0]:
        print(f"{__name__=} {sys.argv=}")
        print("unsupported chars unittest")
        return
    try:
        print(Style.GREEN(str(str_, 'Utf-8')))
    except TypeError:
        try:
            print(Style.GREEN(str(str_, 'ISO-88591')))
        except TypeError:
            print(str_)
