import logging
from importlib import import_module

from toolboxv2 import MainTool, FileHandler, App, Style
from toolboxv2.utils.toolbox import get_app


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.2"
        self.name = "VirtualizationTool"
        self.logger: logging.Logger or None = app.logger if app else None
        if app is None:
            app = get_app()

        self.app = app
        self.instances = {}
        self.color = "BLUE"
        self.keys = {
            "tools": "v-tools~~~"
        }
        self.tools = {
            "all": [["Version", "Shows current Version"],
                    ["create", "Crate an new instance"],
                    ["set-ac", "set an existing instance"],
                    ["list", "list all instances"],
                    ["shear", "shear functions withe an v- instance"]],
            "name": "VirtualizationTool",
            "Version": self.show_version,
            "create":self.create_instance,
            "set-ac":self.set_ac_instances,
           "list":self.list_instances,
            "shear":self.shear_function,
        }
        FileHandler.__init__(self, "VirtualizationTool.config", app.id if app else __name__, keys=self.keys, defaults={})
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                        name=self.name, logs=self.logger, color=self.color, on_exit=self.on_exit)

    def on_start(self):
        self.logger.info(f"Starting VirtualizationTool")
        self.load_file_handler()
        pass

    def on_exit(self):
        self.logger.info(f"Closing VirtualizationTool")
        self.save_file_handler()
        pass


    def show_version(self):
        self.print("Version: ", self.version)
        return self.version

    def set_ac_instances(self, name):
        if name not in self.instances.keys():
            self.print(Style.RED("Pleas Create an instance before calling it!"))
            return None
        self.app.AC_MOD = self.instances[name]

    def get_instance(self, name):
        if name not in self.instances.keys():
            self.print(Style.RED("Pleas Create an instance before calling it!"))
            return None
        return self.instances[name]

    def create_instance(self, name, mod_name):
        loc = "toolboxv2.mods."
        mod = import_module(loc + mod_name)
        mod = getattr(mod, "Tools")
        mod = mod(app=get_app(f"Virtual-{name}"))
        if not mod:
            self.logger.errow(Style.RED("No Mod found to virtualize"))
        self.instances[name] = mod
        self.set_ac_instances(name)
        self.shear_function(name, mod_name, 'show_version')

    def list_instances(self):
        for name, instance in self.instances.items():
            self.print(f"{name}: {instance.name}")

    def shear_function(self, name, mod_name, function_name):
        self.app.new_ac_mod(mod_name)
        function = getattr(self.app.AC_MOD, function_name)
        setattr(self.instances[name], function_name, function)
