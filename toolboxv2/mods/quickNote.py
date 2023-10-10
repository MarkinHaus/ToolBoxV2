import json

import requests

from toolboxv2 import MainTool, FileHandler, App, Style


class Tools(MainTool, FileHandler):  # FileHandler

    def __init__(self, app=None):
        self.edit_note = None
        self.remove_note = None
        self.version = "0.0.1"
        self.name = "quickNote"
        self.logs = app.logger if app else None
        self.color = "GREEN"
        self.keys = {
            "inbox": "INBOX@ADD~",
            "token": "comm-tok~~",
        }
        self.inbox = []
        self.inbox_sto = []
        self.tools = {
            "all": [["Version", "Shows current Version"],
                    ["ADD", "Add a new note to inbox"],
                    ["VIEW", "View notes"],
                    ["Save_inbox", "Save notes to inbox"],
                    ["Get_inbox", "Get notes from inbox"],
                    ["save_inbox_api", ""],
                    ["get_inbox_api", ""],
                    ["save_types_api", ""],
                    ["get_types_api", ""],
                    ],
            "name": "quickNote",
            "Version": self.show_version,
            "ADD": self.add_note,
            "VIEW": self.view_note,
            "Save_inbox": self.save_data,
            "Get_inbox": self.get_inbox,
            "save_inbox_api": self.save_inbox_api,
            "get_inbox_api": self.get_inbox_api,
            "save_types_api": self.save_types_api,
            "get_types_api": self.get_types_api,
        }

        FileHandler.__init__(self, "quickNote.data", app.id if app else __name__, self.keys, {"inbox": [],
                                                                                              "token": "#TOKEN#"})
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=self.logs, color=self.color, on_exit=self.on_exit)

    def show_version(self):
        self.print("Version: ", self.version)
        return self.version

    def on_start(self):
        self.load_file_handler()
        self.open_inbox()

    def on_exit(self):
        self.add_to_save_file_handler(self.keys["inbox"], str(self.inbox))
        self.save_file_handler()

    def open_inbox(self):
        self.print("Loading inbox | ", end="")
        self.logs.info("quickNote try access inbox")
        inbox = self.get_file_handler(self.keys["inbox"])

        if inbox is not None:
            self.inbox = inbox
            self.print(Style.GREEN("load inbox | "), end="")
            self.logs.info("quickNote loaded inbox")
        else:
            self.print(Style.YELLOW("No inbox found | "), end="")
            self.logs.info(Style.RED("No inbox found"))

    def add_note(self, note):

        self.print(f"Adding note... {len(note)}")

        if len(note) <= 1:
            self.print("Write note")
            note = input(": ")
            id_ = sum([ord(c) for c in note])
            if note:
                self.inbox.append({"note": note, "type": "quickNote/inbox", "id": id_})

    def view_note(self):

        for pos, note in enumerate(self.inbox):
            self.print(f"NOTE : {note['note']}\n\t{note['type']=}\n\t{note['id']=}\n")

        return self.inbox

    def save_data(self):
        # get user Token
        token_sto = FileHandler("modules.config").load_file_handler()
        token = token_sto.get_file_handler(self.keys["token"])
        token_sto.file_handler_storage.close()

        if token:
            data = {"notes": self.inbox,
                    "token": token}
            url = "http://172.19.0.1:8081/quick_note/save_inbox"

            r = requests.post(url, json=data)
            self.print(r.status_code)
            self.print(r.content)
        else:
            print(f"No token found in modules.config")

    def get_inbox(self):

        token_sto = FileHandler("modules.config").open_l_file_handler().load_file_handler()
        token = token_sto.get_file_handler(self.keys["token"])
        token_sto.file_handler_storage.close()

        if token:

            url = "http://172.19.0.1:8081/quick_note/get_inbox/" + token

            r = requests.get(url)
            self.print(r.status_code)
            self.print(r.content)
            print(r.json())
            message = r.json()["message"]
            status = r.json()["status"]

            if status:
                print(f"Message: {message}")
                self.inbox_sto = self.inbox
                self.inbox = json.loads(message)
                self.print("Saved")
            #
            else:
                self.print(Style.RED(f"ERROR: {message}"))
        else:
            print(f"No token found in modules.config")

    def get_uid(self, command, app: App):

        if "cloudm" not in list(app.MOD_LIST.keys()):
            return f"Server has no cloudM module", True

        if "db" not in list(app.MOD_LIST.keys()):
            return "Server has no database module", True

        res = app.run_any('cloudm', "validate_jwt", command)

        if type(res) is str:
            return res, True

        return res["uid"], False

    def save_inbox_api(self, command, app: App):

        data = command[0].data
        uid, err = self.get_uid(command, app)

        if err:
            return uid

        return app.run_any('db', 'set', ["", f"quickNote::inbox::{uid}", str(data["notes"])])

    def get_inbox_api(self, command, app: App):
        self.print(command)
        uid, err = self.get_uid(command, app)

        if err:
            return uid
        inbox = app.run_any('db', 'get', [f"quickNote::inbox::{uid}"])
        if len(str(inbox)) > 2:
            return inbox
        return [{'id': '0', 'note': '', 'type': 'quickNotes/Inbox'}]

    def save_types_api(self, command, app: App):

        data = command[0].data
        uid, err = self.get_uid(command, app)

        if err:
            return uid

        return app.run_any('db', 'set', ["", f"quickNote::types::{uid}", str(data["types"])])

    def get_types_api(self, command, app: App):

        uid, err = self.get_uid(command, app)

        if err:
            return uid

        types = app.run_any('db', 'get', [f"quickNote::types::{uid}"])

        if len(str(types)) > 2:
            return types
        return [{'color': '', 'name': 'root', 'sub_type_of': ''}]
