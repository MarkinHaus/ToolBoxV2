from toolboxv2 import MainTool


class Tools(MainTool):
    def __init__(self, app=None):
        self.version = "0.3.2"
        self.name = "welcome"
        self.logs = app.logs_ if app else None
        self.color = "YELLOW"
        self.tools = {
            "all": [["Version", "Shows current Version "]],
            "name": "print_main",
            "Version": self.show_version}

        MainTool.__init__(self, load=None, v=self.version, tool=self.tools,
                          name=self.name, logs=self.logs, color=self.color, on_exit=lambda: "")

    def show_version(self):
        return self.version
