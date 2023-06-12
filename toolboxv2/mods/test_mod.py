from toolboxv2 import MainTool, FileHandler


class Tools(MainTool, FileHandler):
    def __init__(self, app=None):
        self.version = "0.0.1"
        self.name = "welcome"
        self.logs = app.logger if app else None
        self.color = "YELLOW"
        self.tools = {
            "all": [["Version", "Shows current Version "]],
            "name": "print_main",
            "Version": self.show_version}
        default_config = {
            "url": '"http://127.0.0.1:500/api"',
            "TOKEN": '"~tok~"',
        }
        default_keys = {
            "URL": "comm-vcd~~",
            "TOKEN": "comm-tok~~",
        }
        FileHandler.__init__(self, "modules.config", app.id if app else __name__, default_keys, default_config)

        MainTool.__init__(self, load=None, v=self.version, tool=self.tools,
                          name=self.name, logs=self.logs, color=self.color, on_exit=lambda: "")

    def show_version(self):
        return self.version
