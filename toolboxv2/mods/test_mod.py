from toolboxv2 import MainTool
from time import sleep
from platform import system
import os

from toolboxv2.Style import Style


class Tools(MainTool):
    def __init__(self, app=None):
        self.version = "0.3.2"
        self.name = "welcome"
        self.logs = app.logs_ if app else None
        self.color = "YELLOW"
        self.tools = {
            "all": [["Version", "Shows current Version "], ["tool_tip", "tips"], ["Animation", "TOOL BOX 0 8s"],
                    ["Animation1", "TOOL BOX 1 8s"], ["printT", "print TOOL BOX"]],
            "name": "print_main",
            "Version": self.version_}

        MainTool.__init__(self, load=None, v=self.version, tool=self.tools,
                          name=self.name, logs=self.logs, color=self.color, on_exit=lambda: "")

    def version_(self):
        return self.version
