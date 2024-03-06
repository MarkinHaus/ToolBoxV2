from toolboxv2.utils.Style import Style, remove_styles, Spinner
from toolboxv2.utils.file_handler import FileHandler
from toolboxv2.utils.toolbox import App, get_app
from toolboxv2.utils.tb_logger import setup_logging, get_logger
from toolboxv2.utils.main_tool import MainTool
from toolboxv2.utils import all_functions_enums as tbef
from toolboxv2.utils.types import Result, AppArgs
from toolboxv2.utils.cryp import Code

__all__ = [
    "App",
    "MainTool",
    "FileHandler",
    "Style",
    "Spinner",
    "remove_styles",
    "AppArgs",
    "setup_logging",
    "get_logger",
    "get_app",
    "tbef",
    "Result",
    "Code",
]
import ctypes


class Singleton(type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    _instances = {}
    _kwargs = {}
    _args = {}

    def __call__(cls, *args, **kwargs):
        # print("Calling Singleton instances :", cls._instances)
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
            cls._args[cls] = args
            cls._kwargs[cls] = kwargs
        return cls._instances[cls]


TBRUNNER_console_viabel = True


def show_console(show=True):
    global TBRUNNER_console_viabel
    """Brings up the Console Window."""
    try:
        if show and not TBRUNNER_console_viabel:
            # Show console
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 4)
            TBRUNNER_console_viabel = True
        elif not show and TBRUNNER_console_viabel:
            # Hide console
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
            TBRUNNER_console_viabel = False
    except:
        print(f"Could not show_console {show=}", )
