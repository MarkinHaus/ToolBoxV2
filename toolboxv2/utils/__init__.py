import os

from yaml import safe_load

from .extras.show_and_hide_console import show_console
from .extras.Style import Spinner, Style, remove_styles
from .security.cryp import Code
from .singelton_class import Singleton
from .system import all_functions_enums as TBEF
from .system.file_handler import FileHandler
from .system.getting_and_closing_app import get_app
from .system.main_tool import MainTool
from .system.tb_logger import get_logger, setup_logging
from .system.types import ApiResult, AppArgs, Result
from .toolbox import App

__all__ = [
    "App",
    "Singleton",
    "MainTool",
    "FileHandler",
    "Style",
    "Spinner",
    "remove_styles",
    "AppArgs",
    "show_console",
    "setup_logging",
    "get_logger",
    "get_app",
    "TBEF",
    "Result",
    "ApiResult",
    "Code",
]
