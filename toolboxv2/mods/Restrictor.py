import inspect
import logging
from toolboxv2 import MainTool, FileHandler, App, Style
from toolboxv2 import ToolBox_over


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.2"
        self.name = "Restrictor"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "WHITE"
        # ~ self.keys = {}
        self.tools = {
            "all": [["Version", "Shows current Version"]],
            "name": "Restrictor",
            "Version": self.show_version,
        }
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                        name=self.name, logs=self.logger, color=self.color, on_exit=self.on_exit)
        self.seves = {}

    def on_start(self):
        self.logger.info(f"Starting Restrictor")

    def on_exit(self):
        self.logger.info(f"Closing Restrictor")

    def show_version(self):
        self.print("Version: ", self.version)
        return self.version

    def restricted_generator(self, by, name):
        text = Style.WHITE(Style.Bold(f"Function {name} is restricted ") + "by" )+by
        def restricted(*args, **kwargs):
            self.print(text)
            curframe = inspect.currentframe()

            calframe = inspect.getouterframes(curframe, 2)

            self.print('ty access from:', calframe)
            print(f"{args=}"
                  f"{kwargs=}")

        return restricted

    def restrict(self, mod, function_name, by, resid, real_name: str or None = None):
        if real_name is None:
            real_name = function_name
        self.seves[f"{by}-{function_name}"] = {"id": resid, "function": getattr(mod, real_name), 'mod': mod}
        restriction = self.restricted_generator(by, function_name)
        mod.tools[function_name] = restriction
        setattr(mod, real_name, restriction)

    def un_lock(self, by, resid, function_name):

        if ToolBox_over != 'root':
            self.print(Style.RED("Permission dined"))
            return

        if f"{by}-{function_name}" not in self.seves.keys():
            self.print(Style.YELLOW("Function not found"))
            return

        data = self.seves[f"{by}-{function_name}"]

        if data['id'] != resid:
            self.print("Wrong access id")

        data['mod'].tools[function_name] = data['function']
