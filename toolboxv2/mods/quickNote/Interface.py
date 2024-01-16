from toolboxv2 import get_app, App, tbef, Result
from .quickNote import Tools
from .types import *

Name = 'quickNote'
export = get_app(f"{Name}.Export").tb
default_export = export(mod_name=Name)
test_only = export(mod_name=Name, test_only=True, state=False)
version = '0.0.1'


def mInput():
    t = ""
    while True:
        s = input("")
        if not s:
            break
        t += s
    return s


@export(mod_name=Name, initial=True)
def log_in_to_instance(state: Tools, username: str):
    return state.init(username).print()


def add_node(state: Tools):
    print("multi line input add empty lien to exit")
    data = mInput()
    name = input("Add a name")
    note = Note.crate_new_text(name, data, state.tag, "", state.tag)
    state.add_note(note)


def add(self, tag_data: Tag, note_data: Note or None = None, is_tag: bool = False, is_note: bool = True):
    pass
