import datetime

from prompt_toolkit import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import set_title, yes_no_dialog

from toolboxv2 import App, Result, tbef
from toolboxv2.utils.extras.Style import cls
from toolboxv2.utils.system.types import CallingObject

NAME = 'qncli'


def run(app: App, _):
    from toolboxv2.mods.quickNote.quickNote import Tools
    from toolboxv2.mods.quickNote.types import Tag, Note

    set_title(f"ToolBox : {app.version} quickNote")
    threaded = False
    qn_instance: Tools = app.get_mod('quickNote', 'app')
    if len(qn_instance.inbox.get("Notes", {}).keys()) == 0:
        qn_instance.add_note(qn_instance.note)
    if len(qn_instance.inbox.get("Tags", {}).keys()) == 0:
        qn_instance.add_tag(qn_instance.tag)

    # init
    # crate node tag
    # add note
    # remove
    # vue singel / all reladeed
    # switch note

    def init(call_: CallingObject):
        if not call_.function_name:
            return Result.default_user_error(info="add username").set_origin("quickNoteCli::build-in")
        if call_.function_name == "":
            return Result.default_user_error(info="No data provided")
        res = qn_instance.init(username=call_.function_name)
        if res.is_error():
            return res
        qn_instance.open_inbox()
        return res.ok()

    def crate_tag(call_: CallingObject):
        if not call_.function_name:
            return Result.default_user_error(info="add username").set_origin("quickNoteCli::build-in")
        if call_.function_name == "":
            return Result.default_user_error(info="No data provided")

        qn_instance.tag = Tag.crate(call_.function_name)
        return add_tag(call_)

    def crate_note(call_: CallingObject):
        if not call_.function_name:
            return Result.default_user_error(info="add username").set_origin("quickNoteCli::build-in")

        name = call_.args[0]
        data = call_.args[1:]
        if call_.function_name == "":
            return Result.default_user_error(info="No data provided")
        else:  # call_.function_name.lower() in ["t", "text"]:
            qn_instance.note = Note.crate_new_text(name=call_.function_name, data=' '.join(call_.args),
                                                   tag=qn_instance.tag, parent=qn_instance.note)
        # elif call_.function_name.lower() in ["m", "md"]:
        #     qn_instance.note = Note.crate_new_md(name=name, data=data, tag=qn_instance.tag, parent=qn_instance.note)
        # elif call_.function_name.lower() in ["h", "html"]:
        #     qn_instance.note = Note.crate_new_html(name=name, data=data, tag=qn_instance.tag, parent=qn_instance.note)
        # elif call_.function_name.lower() in ["c", "custom"]:
        #     qn_instance.note = Note.crate_new_custom(name=name, data=data, tag=qn_instance.tag, parent=qn_instance.note)
        # else:
        #     return Result.default_user_error(info=f"invalid Mode {call_.function_name} not t,m,h or c")
        qn_instance.note.print()
        return add_note(call_)

    def add_note(_):
        qn_instance.add_note(qn_instance.note)
        return Result.ok()

    def add_tag(_):
        qn_instance.add_tag(qn_instance.tag)
        return Result.ok()

    def save(_):
        qn_instance.on_exit()
        qn_instance.open_inbox()
        return Result.ok()

    def set_tag(call_):
        if not call_.function_name:
            return Result.default_user_error(info="add tag name").set_origin("quickNoteCli::build-in")
        if call_.function_name == "":
            return Result.default_user_error(info="No data provided")

        tag_result = qn_instance.get_tag_by_name(call_.function_name)
        if tag_result.is_error() and not tag_result.is_data():
            return tag_result
        qn_instance.tag = tag_result.get()
        qn_instance.tag.print()
        return Result.ok(data=qn_instance.tag)

    def set_note(call_):
        if not call_.function_name:
            return Result.default_user_error(info="add note name").set_origin("quickNoteCli::build-in")
        if call_.function_name == "":
            return Result.default_user_error(info="No data provided")

        note_result = qn_instance.get_note_by_name(call_.function_name)
        if note_result.is_error() and not note_result.is_data():
            return note_result
        qn_instance.note = note_result.get()
        qn_instance.note.print()
        return Result.ok(data=qn_instance.note)

    def remove_tag(call_):
        if qn_instance.tag is not None:
            qn_instance.remove_tag(qn_instance.tag.id)
            print("removed", qn_instance.tag.name)
            return Result.ok()
        return Result.default_user_error(info="Tag not found 404")

    def remove_note(call_):
        if qn_instance.note is not None:
            qn_instance.remove_note(qn_instance.note.id)
            print("removed")
            return Result.ok()
        return Result.default_user_error(info="Note not found 404")

    def vue_tag(call_):
        if not call_.function_name:
            return Result.default_user_error(info="add note name").set_origin("quickNoteCli::build-in")
        if call_.function_name == "":
            return Result.default_user_error(info="No data provided")

        tag_result = qn_instance.get_tag_by_name(call_.function_name)
        if tag_result.is_error() and not tag_result.is_data():
            return tag_result
        tag_result.get().print()
        return Result.ok()

    def vue_tag_r(call_):
        if not call_.function_name:
            return Result.default_user_error(info="add note name").set_origin("quickNoteCli::build-in")
        if call_.function_name == "":
            return Result.default_user_error(info="No data provided")

        tag_result = qn_instance.get_tag_by_name(call_.function_name)
        if tag_result.is_error() and not tag_result.is_data():
            return tag_result
        tag: Tag = tag_result.get()
        for r_tag_id in tag.related:
            r_tag = qn_instance.get_tag(r_tag_id)
            r_tag.print()
        return Result.ok()

    def vue_note(call_):
        if not call_.function_name:
            return Result.default_user_error(info="add note name").set_origin("quickNoteCli::build-in")
        if call_.function_name == "":
            return Result.default_user_error(info="No data provided")

        note_result = qn_instance.get_note_by_name(call_.function_name)
        if note_result.is_error() and not note_result.is_data():
            return note_result
        note_result.get().print()
        return Result.ok()

    def vue_note_r(call_):
        if not call_.function_name:
            return Result.default_user_error(info="add note name").set_origin("quickNoteCli::build-in")
        if call_.function_name == "":
            return Result.default_user_error(info="No data provided")

        note_result = qn_instance.get_note_by_name(call_.function_name)
        if note_result.is_error() and not note_result.is_data():
            return note_result
        note: Note = note_result.get()
        for r_id in note.links:
            r_ob = qn_instance.get_tag(r_id)
            if r_ob.name == 'root':
                r_ob = qn_instance.get_note(r_id)
            r_ob.print()
        return Result.ok()

    def bottom_toolbar():
        return HTML(
            f'<div><style fg="lightgreen">Hotkeys shift:s control:c  <b><style bg="ansired">s+left</style></b> helper info '
            f'<b><style bg="ansired">c+space</style></b> Autocompletion tips '
            f'<b><style bg="ansired">s+up</style></b> run in shell</style></div>')

    def exit_(_):
        if 'main' in app.id:
            res = yes_no_dialog(
                title='Exit ToolBox',
                text='Do you want to Close the ToolBox?').run()
            app.alive = not res
        else:
            app.alive = False
        return Result.ok().set_origin("minicli::build-in")

    helper_exequtor = [None]

    def cls_(_):
        cls()
        return Result.ok(info="cls").set_origin("minicli::build-in")

    def run_(call_: CallingObject) -> Result:
        if not call_.function_name:
            return (Result.default_user_error(info=f"Avalabel are : {list(app.runnable.keys())}")
                    .set_origin("minicli::build-in"))
        if call_.function_name in app.runnable:
            app.run_runnable(call_.function_name)
            return Result.ok().set_origin("minicli::build-in")
        return Result.default_user_error("404").set_origin("minicli::build-in")

    bindings = KeyBindings()

    @bindings.add('down')
    def show_ac_note(event):
        qn_instance.note.print(debug=app.debug)

    bic = {
        "exit": exit_,
        "save": save,
        "cls": cls_,
        "init": init,
        "vNote": vue_note,
        "vTag": vue_tag,
        "vrNote": vue_note_r,
        "vrNag": vue_tag_r,
        "setN": set_note,
        "setT": set_tag,
        "removeN": remove_note,
        "removeT": remove_tag,
        "NN": crate_note,
        "NT": crate_tag,
        "run": run_,
    }

    # set up Autocompletion
    autocompletion_dict = {}
    autocompletion_dict = app.run_any(tbef.CLI_FUNCTIONS.UPDATE_AUTOCOMPLETION_LIST_OR_KEY, list_or_key=bic,
                                      autocompletion_dict=autocompletion_dict)

    # autocompletion_dict["slm:set_load_mode"] = {arg: None for arg in ['i', 'c']}
    # autocompletion_dict["sdm:set_debug_mode"] = {arg: None for arg in ['on', 'off']}
    autocompletion_dict["init"] = {'name': {'root'}, }
    autocompletion_dict["vNote"] = {'name': {'root'}, }
    autocompletion_dict["vTag"] = {'name': {'root'}, }
    autocompletion_dict["vrNote"] = {'name': {'root'}, }
    autocompletion_dict["vrNag"] = {'name': {'root'}, }
    autocompletion_dict["setN"] = {'name': {'root'}, }
    autocompletion_dict["setT"] = {'name': {'root'}, }
    autocompletion_dict["removeN"] = {'name': {'root'}, }
    autocompletion_dict["removeT"] = {'name': {'root'}, }
    autocompletion_dict["NT"] = {'name': {'root'}, }
    autocompletion_dict["NN"] = {'name': {'root'}, 'text': None}
    autocompletion_dict["run"] = {arg: None for arg in list(app.runnable.keys())}

    autocompletion_dict = app.run_any(tbef.CLI_FUNCTIONS.UPDATE_AUTOCOMPLETION_MODS,
                                      autocompletion_dict=autocompletion_dict)

    active_modular = ""

    running_instance = None

    while app.alive:

        def get_rprompt():
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Get current time
            return HTML(
                f'''<b> AC Tag : {qn_instance.tag.print(show=False, debug=app.debug)}</b>''')

        call = app.run_any(tbef.CLI_FUNCTIONS.USER_INPUT, completer_dict=autocompletion_dict,
                           get_rprompt=get_rprompt, bottom_toolbar=bottom_toolbar, active_modul=active_modular,
                           message=f"~Note:{qn_instance.note.name}@>")

        print("", end="" + "start ->>\r")

        if call is None:
            continue

        if call.module_name == "open":
            autocompletion_dict = app.run_any(tbef.CLI_FUNCTIONS.UPDATE_AUTOCOMPLETION_MODS,
                                              autocompletion_dict=autocompletion_dict)

        running_instance = app.run_any(tbef.CLI_FUNCTIONS.CO_EVALUATE,
                                       obj=call,
                                       build_in_commands=bic,
                                       threaded=threaded,
                                       helper=helper_exequtor[0])

        print("", end="" + "done ->>\r")

    if running_instance is not None:
        print("Closing running instance")
        running_instance.join()
        print("Done")

    set_title("")
