"""Top-level package for ToolBox."""
from toolboxv2.toolbox import App, AppArgs
from toolboxv2.main_tool import MainTool
from toolboxv2.util import setup_logging, get_logger
from toolboxv2.file_handler import FileHandler
from toolboxv2.Style import Style, remove_styles
from toolboxv2.readchar_buldin_style_cli import run_cli
from toolboxv2.app.serve_app import AppServerHandler

__author__ = """Markin Hausmanns"""
__email__ = 'Markinhausmanns@gmail.com'
__version__ = '0.0.3'
__all__ = [
    __version__,
    App,
    MainTool,
    FileHandler,
    Style,
    remove_styles,
    run_cli,
    AppServerHandler,
    AppArgs,
    setup_logging,
    get_logger,
    ]
ToolBox_ovner = "root"
