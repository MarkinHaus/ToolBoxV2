from toolboxv2 import MainTool, FileHandler, App
from toolboxv2.Style import Style


def decorator_x(function):
    def fx(c, a):
        return function(c, a)

    return fx


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.2"
        self.name = "DecoTP"
        self.logs = app.logger if app else None
        self.color = "WHITE"
        # ~ self.keys = {}
        self.tools = {
            "all": [["Version", "Shows current Version"]],
            "name": "DecoTP",
            "Version": self.show_version,
        }
        # ~ FileHandler.__init__(self, "File name", app.id if app else __name__, keys=self.keys, defaults={})
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                        name=self.name, logs=self.logs, color=self.color, on_exit=self.on_exit)

    def show_version(self):
        self.print("Version: ", self.version)

    def on_start(self):
        # ~ self.load_file_handler()
        pass

    def on_exit(self):
        # ~ self.save_file_handler()
        pass

    @decorator_x
    def deco_test(self, command, app):
        return "x"


