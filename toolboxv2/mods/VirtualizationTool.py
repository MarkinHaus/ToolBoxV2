import logging
from importlib import import_module

from toolboxv2 import MainTool, FileHandler, Style
from toolboxv2.utils.toolbox import get_app


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.2"
        self.name = "VirtualizationTool"
        self.logger: logging.Logger or None = app.logger if app else None
        if app is None:
            app = get_app()

        self.app_ = app
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

    def set_ac_instances(self, command):
        name = command
        if isinstance(command, list):
            if len(command) == 1:
                name = command[0]
            if len(command) == 2:
                name = command[1]
        if name not in self.instances.keys():
            self.print(Style.RED(f"Pleas Create an instance before calling it! : {name}"))

            self.logger.warning(Style.RED("set_ac_instances - Pleas Create an instance before calling it!"))
            return False
        self.app_.AC_MOD = self.instances[name]
        self.print(Style.WHITE(Style.Bold(f"New ac instances: {self.app_.AC_MOD.name}:{name}")))
        return True

    def get_instance(self, name):
        if name not in self.instances.keys():
            self.list_instances()
            self.print(Style.YELLOW("Pleas Create an instance before calling it!"))
            self.logger.warning(Style.YELLOW("get_instance - Pleas Create an instance before calling it!"))
            return None
        return self.instances[name]

    def create_instance(self, name, mod_name):
        mod_sto = self.app_.AC_MOD
        loc = "toolboxv2.mods."
        self.print(Style.CYAN(f"Create an instance {mod_name}"))
        try:
            mod = import_module(loc + mod_name)
            mod = getattr(mod, "Tools")
            mod.toolID = name
            mod_init = mod(app=get_app(f"Virtual-{name}"))
            mod.toolID = ""
            if not mod_init:
                self.logger.errow(Style.RED("No Mod found to virtualize"))
            if issubclass(mod, FileHandler):
                mod_init.file_handler_file_prefix += f"Virtualize/{name}/"

            self.print(f"Virtualizing source {mod_init.name} to -> {name}")

            mod_init.name = name
            self.instances[name] = mod_init
            self.app_.save_init_mod(name, mod_init)
            self.set_ac_instances(name)
            self.shear_function(mod_name, name, 'show_version')
            self.app_.AC_MOD = mod_sto
            return mod_init
        except ImportError as e:
            self.app_.AC_MOD = mod_sto
            self.print(Style.Bold(f"{Style.RED('Error')} Virtualizing {mod_name} in {name} ; {e}"))
            return None

    def list_instances(self):
        for name, instance in self.instances.items():
            self.print(f"{name}: {instance.name}")

    def shear_function(self, name, instance_name, function_name):
        print(name)
        self.app_.new_ac_mod(name)
        function = getattr(self.app_.AC_MOD, function_name)
        setattr(self.instances[instance_name], function_name, function)

