from toolboxv2 import MainTool, FileHandler, App
from toolboxv2.Style import Style


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.2"
        self.name = "Shopmate"
        self.color = "WHITE"
        self.keys = {}
        self.tools = {
            "all": [["Version", "Shows current Version"]],
            "name": "Shopmate",
            "Version": self.show_version,
        }
        FileHandler.__init__(self, "File name", app.id if app else __name__, keys=self.keys, defaults={})
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=None, color=self.color, on_exit=self.on_exit)

    def show_version(self):
        self.print("Version: ", self.version)

    def on_start(self):
        self.load_file_handler()

    def on_exit(self):
        self.save_file_handler()

    def create_grop(self, command, app: App):

        # TODO: die gruppe existiert bereits meldung

        uid, err = self.get_uid(command, app)

        if err:
            app.logger.error(uid)
            return uid

        data = command[0].data  # {"name": "Gruppe", "Password": "password"}

        key = f"Shopmte::Grop::{data.name}::{uid}"

        value = {
            'owner': uid,
            'members': [uid, ],
            'types': [],
            'items': [],
        }

        app.run_any('DB', 'set', ["", key, str(value)])

        # TODO: zum profil hinzufügen

        key = f"Shopmte::Grops::{uid}"

        value = {
            data.name: {""}
        }

        app.run_any('DB', 'set', ["", key, str(value)])

    def share_grop(self, command, app: App):

        uid, err = self.get_uid(command, app)

        if err:
            app.logger.error(uid)
            return uid

        data = command[0].data  # {"name": "Gruppe", "Password": "password"}

    def join_grop(self):
        pass
