import logging
from toolboxv2 import MainTool, FileHandler, App, Style


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.2"
        self.name = "user_manager"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "WHITE"
        self.keys = {}
        self.tools = {
            "all": [["Version", "Shows current Version"]],
            "name": "user_manager",
            "Version": self.show_version
        }
        FileHandler.__init__(self, "File name", app.id if app else __name__, keys=self.keys, defaults={})
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                        name=self.name, logs=self.logger, color=self.color, on_exit=self.on_exit)

    def on_start(self):
        self.logger.info(f"Starting user_manager")
        self.load_file_handler()

    def on_exit(self):
        self.logger.info(f"Closing user_manager")
        self.save_file_handler()


    def show_version(self):
        self.print("Version: ", self.version)
        return self.version

    def run_for_user(self, command, app):

        uid, err = self.get_uid(command, app)

        if err:
            raise ValueError("Invalid user token")

        # nutzer möchte isaa ausführen

        data = command[0].data # {'mod':'isaa', command:[]}

        if 'mod' not in data.keys():
            raise ValueError("No Mod to run Serifed data : {mod:name:str,command:commands:{data}}")

        if 'command' not in data.keys():
            raise ValueError("No command Serifed data : {mod:name:str,command:commands:list[str]}")

        command_reverse = data['command'][::-1].appaend(command[0])

        mod_command = command_reverse[::-1]

