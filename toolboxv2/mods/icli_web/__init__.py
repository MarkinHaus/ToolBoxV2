"""
icli_web — ToolBoxV2 module.

This module exposes the icli-side WebSocket client. The FastAPI server
runs separately via `python -m toolboxv2.mods.icli_web.server` (or via
ServiceManager / systemd / docker — whatever you prefer).

To connect icli to the server, add to ICli.__init__:

    from toolboxv2.mods.icli_web.client import IcliWebClient
    IcliWebClient.get().attach(self)

Environment:
    ICLI_WEB_HOST       default 127.0.0.1
    ICLI_WEB_PORT       default 5055
    ICLI_WEB_API_KEY    from ~/.toolbox/icli_web.key if unset
"""
from __future__ import annotations

from toolboxv2 import get_app, MainTool

from .client import IcliWebClient

MOD_NAME = "icli_web"
VERSION = "0.5.0"


class Tools(MainTool):
    version = VERSION
    name = MOD_NAME
    color = "CYAN"

    def __init__(self, app=None):
        self.app = app or get_app()
        self.tools = {"name": MOD_NAME, "Version": self.show_version}
        MainTool.__init__(self, v=self.version, tool=self.tools,
                          name=self.name, color=self.color)

    def show_version(self):
        return self.version


__all__ = ["IcliWebClient"]
