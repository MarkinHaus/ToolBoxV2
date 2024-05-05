from toolboxv2 import get_app
from .quickNote import Tools
from .types import *
from ..CloudM import User

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


def test_quickNote():
    app = get_app("quicknote_test", "test")
    quicknote: Tools = app.get_mod(Name)
    quicknote.set_user(User())
    quicknote.open_inbox()
    note = quicknote.get_note_by_name("404")
    assert note.is_error()
    quicknote.add_note(Note.crate_root())
    root_n = quicknote.get_note_by_name("root")
    assert not root_n.is_error()
    quicknote.on_exit()
    quicknote: Tools = app.get_mod(Name)
    quicknote.open_inbox()
    root_n = quicknote.get_note_by_name("root")
    assert not root_n.is_error()
    quicknote.remove_note_by_name("root")
    root_n = quicknote.get_note_by_name("root")
    assert root_n.is_error()
    quicknote.on_exit()
    quicknote: Tools = app.get_mod(Name)
    quicknote.open_inbox()
    root_n = quicknote.get_note_by_name("root")
    assert root_n.is_error()


@export(mod_name=Name, initial=True)
def log_in_to_instance(state: Tools, username: str):
    return state.init(username).print()


def add_node(state: Tools):
    print("multi line input add empty lien to exit")
    data = mInput()
    name = input("Add a name")
    note = Note.crate_new_text(name, data, [state.tag])
    state.add_note(note)


def add(self, tag_data: Tag, note_data: Note or None = None, is_tag: bool = False, is_note: bool = True):
    pass
