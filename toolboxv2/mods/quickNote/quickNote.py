import json
from dataclasses import asdict

from toolboxv2 import MainTool, FileHandler, Style, Result, tbef
from .types import *
from ..CloudM import User
from ...utils.extras.blobs import BlobFile


class Tools(MainTool, FileHandler):  # FileHandler

    def __init__(self, app=None):
        self.version = "0.0.2"
        self.name = "quickNote"
        self.logs = app.logger if app else None
        self.color = "GREEN"
        self.keys = {
            "inbox": "INBOX@ADD~",
            "token": "quNo-tok~~",
        }
        self.user: User or None = None
        self.inbox = {
            "Notes": {},
            "Tags": {},
        }
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
            "REMOVE": self.add_note,
            "VIEW": self.view_note,
            "Find": self.view_note,
            "init": self.init,
        }
        FileHandler.__init__(self, "quickNote.data", app.id if app else __name__, self.keys, {"inbox": [],
                                                                                              "token": "#TOKEN#"})
        MainTool.__init__(self, v=self.version, tool=self.tools,
                          name=self.name, logs=self.logs, color=self.color, on_exit=self.on_exit)

    def show_version(self):
        self.print("Version: ", self.version)
        return self.version

    def set_user(self, user):
        if self.user is not None:
            return
        self.user = user

    def init(self, username: str, sign: str or None = None, jwt: str or None = None):

        result = Result.default_internal_error(info="Error Log in Module")
        if jwt is not None:
            result = self.app.run_any(tbef.CLOUDM_AUTHMANAGER.JWT_CHECK_CLAIM_SERVER_SIDE, username=username,
                                      jwt_claim=jwt,
                                      get_results=True)
            if result.is_data() and not result.is_error() and result.get() is True:
                self.user = self.app.run_any(tbef.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=username)
            else:
                return result.lazy_return('intern')

        elif sign is not None:
            result = self.app.run_any(tbef.CLOUDM_AUTHMANAGER.AUTHENTICATE_USER_GET_SYNC_KEY, username=username,
                                      signature=sign, get_user=True,
                                      get_results=True)
            if result.is_data() and not result.is_error():
                s_key, user_data = result.get()
                self.user = User(**user_data)
            else:
                return result.lazy_return('intern')

        else:
            result = self.app.run_any(tbef.CLOUDM_AUTHMANAGER.LOCAL_LOGIN, username=username, get_results=True)
            if result.is_data() and not result.is_error():
                self.user = self.app.run_any(tbef.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=username)
            else:
                return result.lazy_return('intern')

        return result

    def on_exit(self):
        self.print(f'exit QN')
        if self.user is not None:
            self.print(f"SAVED DATAT {self.spec[10:]}")
            self.app.run_any(tbef.DB.SET, query=self.keys["inbox"] + self.user.uid, data=self.inbox)
            with BlobFile(f"users/{self.spec}/{self.name}/index.yaml", 'w') as f:
                f.write_yaml(self.inbox)
        self.save_file_handler()

    def open_inbox(self, uid=None):
        self.print("Loading inbox | ", end="")
        self.logs.info("quickNote try access inbox")
        if uid is None and self.user is not None:
            uid = self.user.uid
        if uid is None:
            uid = "public"
        inbox = self.app.run_any(tbef.DB.GET, query=self.keys["inbox"] + uid, get_results=True)

        if not inbox.is_error() and inbox.is_data():
            inbox = inbox.get()[0]
            if isinstance(inbox, bytes):
                self.inbox = json.loads(inbox.decode("utf-8").replace("'", '"'))
            self.print(Style.GREEN("load inbox | "))
            self.logs.info("quickNote loaded inbox")
        else:
            self.print(Style.YELLOW("No inbox found | "), end="")
            with BlobFile(f"users/{self.spec}/{self.name}/index.yaml", 'r') as f:
                inbox = f.read_yaml()
            if inbox:
                if isinstance(inbox, bytes):
                    inbox = json.loads(inbox.decode("utf-8").replace("'", '"'))
                self.inbox = inbox
                self.logs.info(Style.Bold("inbox recovered"))
            else:
                self.logs.info(Style.RED("No inbox found"))

    @staticmethod
    def get_by_id_ny_name(name: str, t="note"):
        return get_id(name=name + t)

    def add_note(self, note: Note):

        if note is None:
            return

        self.inbox["Notes"][note.id] = asdict(note)

        self.print(f"Adding note... {note.name}")

    def add_tag(self, tag: Tag):

        if tag is None:
            return

        self.inbox["Tags"][tag.id] = asdict(tag)

        self.print(f"Adding tag... {tag.name}")

    def get_note(self, note_id: str):

        if note_id is None:
            return Result.default_user_error(info="Note not found no id provided")
        note = self.inbox["Notes"].get(note_id, None)
        if note is not None:
            return Result.ok(data=Note(**note), info="Noted data")
        return Result.default_user_error(info="Note not found", exec_code=404)

    def get_tag(self, tag_id: str):

        if tag_id is None:
            return Result.default_user_error(info="Tag not found", exec_code=404)
        tag = self.inbox["Tags"].get(tag_id, None)
        if tag is None:
            return Result.default_user_error(info="Note not found", exec_code=404)

        return Result.ok(data=Tag(**tag), info="Tag data")

    def get_note_by_name(self, note_name: str):
        return self.get_note(self.get_by_id_ny_name(note_name, t=":Note"))

    def get_tag_by_name(self, tag_name: str):
        return self.get_tag(self.get_by_id_ny_name(tag_name, t=":Tag"))

    def remove_note(self, note_id: str):

        if note_id is None:
            return

        note_name: str = self.get_note(note_id).get().name

        del self.inbox["Notes"][note_id]

        self.print(f"Removed note... {note_name}")

    def remove_tag(self, tag_id: str):

        if tag_id is None:
            return

        tag_name: str = self.get_tag(tag_id).get().name

        del self.inbox["Tags"][tag_id]

        self.print(f"Adding tag... {tag_name}")

    def remove_note_by_name(self, note_name: str):
        self.remove_note(self.get_by_id_ny_name(note_name, t=":Note"))

    def remove_tag_by_name(self, tag_name: str):
        self.remove_tag(self.get_by_id_ny_name(tag_name, t=":Tag"))

    def view_note(self, show=False, data=False):

        notes = []
        for note in self.inbox["Notes"].values():
            res = note.print(show=show, debug=self.app.debug, data=data)
            notes.append(res)

        return notes

    def view_tags(self, show=False, data=False):

        tags = []
        for tag in self.inbox["Tags"].values():
            res = tag.print(show=show, debug=self.app.debug, data=data)
            tags.append(res)

        return tags
