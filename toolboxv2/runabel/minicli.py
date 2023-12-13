import datetime
import psutil

from prompt_toolkit import HTML
from prompt_toolkit.shortcuts import set_title, yes_no_dialog

from toolboxv2.utils.Style import cls
from toolboxv2 import App, Result, tbef
from toolboxv2.utils import CallingObject

NAME = 'minicli'


def run(app: App, args):
    set_title(f"ToolBox : {app.version}")

    def bottom_toolbar():
        return HTML(f'Hotkeys shift:s control:c  <b><style bg="ansired">s+left</style></b> helper info '
                    f'<b><style bg="ansired">c+space</style></b> Autocompletion tips '
                    f'<b><style bg="ansired">s+up</style></b> run in shell')

    def exit_(_):
        if 'main' in app.id:
            res = yes_no_dialog(
                title='Exit ToolBox',
                text='Do you want to Close the ToolBox?').run()
            app.alive = not res
        else:
            app.alive = False
        return Result.ok()

    def set_load_mode(call_: CallingObject) -> Result:
        if not call_.function_name:
            return Result.default_user_error(info=f"slm (Set Load Mode) needs at least one argument I or C\napp is in"
                                                  f" {'Inplace-loading' if app.mlm == 'I' else 'Coppy-loading'} mode")
        if call_.function_name.lower() == "i":
            app.mlm = 'I'
        elif call_.function_name.lower() == "c":
            app.mlm = 'C'
        else:
            return Result.default_user_error(info=f"{call_.function_name} != I or C")
        return Result.ok(info=f"New Load Mode {app.mlm}")

    def set_debug_mode(call_: CallingObject) -> Result:
        if not call_.function_name:
            return Result.default_user_error(info=f"sdm (Set Debug Mode) needs at least one argument on or off\napp is"
                                                  f" {'' if app.debug else 'NOT'} in debug mode")
        if call_.function_name.lower() == "on":
            app.debug = True
        elif call_.function_name.lower() == "off":
            app.debug = False
        else:
            return Result.default_user_error(info=f"{call_.function_name} != on or off")
        return Result.ok(info=f"New Load Mode {app.mlm}")

    def hr(call_: CallingObject) -> Result:
        if not call_.function_name:
            app.remove_all_modules()
            app.load_all_mods_in_file()
        if call_.function_name.lower() in app.functions:
            app.remove_mod(call_.function_name.lower())
            if not app.save_load(call_.function_name.lower()):
                return Result.default_internal_error()
        return Result.ok()

    def open_(call_: CallingObject) -> Result:
        if not call_.function_name:
            app.load_all_mods_in_file()
            return Result.default_user_error(info="No module specified")
        if not app.save_load(call_.function_name.lower()):
            return Result.default_internal_error()
        return Result.ok()

    def close_(call_: CallingObject) -> Result:
        if not call_.function_name:
            app.remove_all_modules()
            return Result.default_user_error(info="No module specified")
        if not app.remove_mod(call_.function_name.lower()):
            return Result.default_internal_error()
        return Result.ok()

    def run_(call_: CallingObject) -> Result:
        if not call_.function_name:
            return Result.default_user_error(info=f"Avalabel are : {list(app.runnable.keys())}")
        if call_.function_name in app.runnable:
            app.run_runnable(call_.function_name)
            return Result.ok()
        return Result.default_user_error("404")

    helper_exequtor = [None]

    def remote(call_: CallingObject) -> Result:
        if not call_.function_name:
            return Result.default_user_error(info="add keyword local or port and host")
        if call_.function_name != "local":
            app.args_sto.host = call_.function_name
        if call_.kwargs:
            print("Adding", call_.kwargs)
        status, sender, receiver_que = app.run_runnable("demon", as_server=False, programmabel_interface=True)
        if status == -1:
            return Result.default_internal_error(info="Failed to connect, No service available")

        def remote_exex_helper(calling_obj: CallingObject):

            kwargs = {
                "mod_function_name": (calling_obj.module_name, calling_obj.function_name)
            }
            if calling_obj.kwargs:
                kwargs = kwargs.update(calling_obj.kwargs)

            if calling_obj.module_name == "exit":
                helper_exequtor[0] = None
                sender({'exit': True})
            sender(kwargs)
            while receiver_que.not_empty:
                print(receiver_que.get())

        helper_exequtor[0] = remote_exex_helper

    bic = {
        "exit": exit_,
        "cls": lambda x: cls(),
        "slm:set_load_mode": set_load_mode,
        "sdm:set_debug_mode": set_debug_mode,
        "open": open_,
        "close": close_,
        "run": run_,
        "reload": hr,
        "remote": remote,
        "..": lambda x: Result.ok(x),
    }

    all_modes = app.get_all_mods()

    # set up Autocompletion
    autocompletion_dict = {}
    autocompletion_dict = app.run_any(tbef.CLI_FUNCTIONS.UPDATE_AUTOCOMPLETION_LIST_OR_KEY, list_or_key=bic,
                                      autocompletion_dict=autocompletion_dict)
    autocompletion_dict["slm:set_load_mode"] = {arg: None for arg in ['i', 'c']}
    autocompletion_dict["sdm:set_debug_mode"] = {arg: None for arg in ['on', 'off']}
    autocompletion_dict["open"] = autocompletion_dict["close"] = autocompletion_dict["reload"] = \
        {arg: None for arg in all_modes}
    autocompletion_dict["run"] = {arg: None for arg in list(app.runnable.keys())}
    autocompletion_dict = app.run_any(tbef.CLI_FUNCTIONS.UPDATE_AUTOCOMPLETION_MODS,
                                      autocompletion_dict=autocompletion_dict)

    active_modular = ""

    running_instance = None

    while app.alive:
        # Get CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)

        # Get memory usage
        memory_usage = psutil.virtual_memory().percent

        # Get disk usage
        disk_usage = psutil.disk_usage('/').percent

        def get_rprompt():
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Get current time
            return HTML(
                f'''<b> App Infos: {app.id} \nCPU: {cpu_usage}% Memory: {memory_usage}% Disk :{disk_usage}%\nTime: {current_time}</b>''')

        call = app.run_any(tbef.CLI_FUNCTIONS.USER_INPUT, completer_dict=autocompletion_dict,
                           get_rprompt=get_rprompt, bottom_toolbar=bottom_toolbar, active_modul=active_modular)

        print("", end="" + "start ->>\r")

        if call.module_name == "open":
            autocompletion_dict = app.run_any(tbef.CLI_FUNCTIONS.UPDATE_AUTOCOMPLETION_MODS,
                                              autocompletion_dict=autocompletion_dict)

        running_instance = app.run_any(tbef.CLI_FUNCTIONS.CO_EVALUATE,
                                       obj=call,
                                       build_in_commands=bic,
                                       threaded=True,
                                       helper=helper_exequtor[0])

        print("", end="" + "done ->>\r")

    if running_instance is not None:
        print("Closing running instance")
        running_instance.join()
        print("Done")

    set_title("")

    # example_style = StylePt.from_dict({
    #    'dialog': 'bg:#88ff88',
    #    'dialog frame.label': 'bg:#ffffff #000000',
    #    'dialog.body': 'bg:#000000 #00ff00',
    #    'dialog shadow': 'bg:#00aa00',
    # })
    #
    # result = radiolist_dialog(
    #    title="RadioList dialog",
    #    text="Which breakfast would you like ?",
    #    values=[
    #        ("breakfast1", "Eggs and beacon"),
    #        ("breakfast2", "French breakfast"),
    #        ("breakfast3", "Equestrian breakfast")
    #    ]
    # ).run()
    #
    # message_dialog(
    #    title=HTML('<style bg="blue" fg="white">Styled</style> '
    #               '<style fg="ansired">dialog</style> window'),
    #    text='Do you want to continue?\nPress ENTER to quit.',
    #    style=example_style).run()
    #
    # results = checkboxlist_dialog(
    #    title="CheckboxList dialog",
    #    text="What would you like in your breakfast ?",
    #    values=[
    #        ("eggs", "Eggs"),
    #        ("bacon", "Bacon"),
    #        ("croissants", "20 Croissants"),
    #        ("daily", "The breakfast of the day")
    #    ],
    #    style=StylePt.from_dict({
    #        'dialog': 'bg:#cdbbb3',
    #        'button': 'bg:#bf99a4',
    #        'checkbox': '#e8612c',
    #        'dialog.body': 'bg:#a9cfd0',
    #        'dialog shadow': 'bg:#c98982',
    #        'frame.label': '#fcaca3',
    #        'dialog.body label': '#fd8bb6',
    #    })
    # ).run()
    #
    # message_dialog(
    #    title='running dialog window',
    #    text='Do you want to continue?\nPress ENTER to quit.').run()

    # from prompt_toolkit import Application
    # from prompt_toolkit.buffer import Buffer
    # from prompt_toolkit.layout.containers import VSplit, Window
    # from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    # from prompt_toolkit.layout.layout import Layout

    # buffer1 = Buffer()  # Editable buffer.
    #
    # root_container = VSplit([
    #     # One window that holds the BufferControl with the default buffer on
    #     # the left.
    #     Window(content=BufferControl(buffer=buffer1)),
    #
    #     # A vertical line in the middle. We explicitly specify the width, to
    #     # make sure that the layout engine will not try to divide the whole
    #     # width by three for all these windows. The window will simply fill its
    #     # content by repeating this character.
    #     Window(width=1, char='|'),
    #
    #     # Display the text 'Hello world' on the right.
    #     Window(content=FormattedTextControl(text='Hello world')),
    # ])
    #
    # layout = Layout(root_container)
    #
    # app = Application(layout=layout, full_screen=True, key_bindings=bindings)
    # app.run()  # You won't be able to Exit this app
