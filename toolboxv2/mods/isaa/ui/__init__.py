"""ISAA UI — FastTB web interface."""
# Lazy imports — `app` pulls FastTB which needs the full ToolBoxV2 install.
# Tests for chat_store / stream_bridge work without that.
try:
    from .app import build_app, main  # noqa: F401
    __all__ = ["build_app", "main"]
except ImportError:
    __all__ = []
