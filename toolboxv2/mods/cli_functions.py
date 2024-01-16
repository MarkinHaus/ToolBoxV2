import os
from dataclasses import dataclass, field
from threading import Thread

from toolboxv2.utils.types import CallingObject

try:
    from readchar import key as readchar_key
    from readchar import readkey

    READCHAR = True
    READCHAR_error = None
except ImportError and ModuleNotFoundError as READCHAR_error:
    READCHAR = False

from toolboxv2 import get_app, App, Result
from platform import node

try:
    from prompt_toolkit import PromptSession, HTML
    from prompt_toolkit.completion import NestedCompleter
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.clipboard import InMemoryClipboard
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.output import ColorDepth
    from prompt_toolkit.application import run_in_terminal

    PROMPT_TOOLKIT = True
    PROMPT_TOOLKIT_error = None
except ImportError and ModuleNotFoundError as PROMPT_TOOLKIT_error:
    PROMPT_TOOLKIT = False

Name = 'cli_functions'
export = get_app("cli_functions.Export").tb
default_export = export(mod_name=Name)
version = '0.0.1'


@export(mod_name=Name, name='Version', version=version)
def get_version():
    return version


class UserInputObject:
    char: chr or str or None = None
    word: str or None = None
    offset_x: int or None = None
    offset_y: int or None = None

    @classmethod
    def default(cls,
                char: chr or str or None = None,
                word: str or None = None,
                offset_x: int or None = None,
                offset_y: int or None = None):
        cls.char = char
        cls.word = word
        cls.offset_x = offset_x
        cls.offset_y = offset_y
        return cls

    @classmethod
    def final(cls):
        cls.char = "LAST"
        cls.word = "LAST"
        cls.offset_x = 0
        cls.offset_y = 0
        return cls

    @classmethod
    def ve(cls):
        cls.char = "ValueError"
        cls.word = "ValueError"
        cls.offset_x = 0
        cls.offset_y = 0
        return cls


@default_export
def get_character():
    get_input = True

    offset_x = 0
    offset_y = 0
    word = ""
    char = ''

    # session_history += [c for c in app.command_history]

    while get_input:

        key = readkey()

        if key == b'\x05' or key == '\x05':
            print('\033', end="")
            get_input = False
            word = "EXIT"

        elif key == readchar_key.LEFT:
            offset_x -= 1

        elif key == readchar_key.RIGHT:
            offset_x += 1

        elif key == readchar_key.UP:
            offset_y -= 1

        elif key == readchar_key.DOWN:
            offset_y += 1

        elif key == b'\x08' or key == b'\x7f' or key == '\x08' or key == '\x7f':
            word = word[:-1]
            char = ''
        elif key == b' ' or key == ' ':
            word = ""
            char = ' '
        elif key == readchar_key.ENTER:
            word = ""
            char = '\n'
        elif key == b'\t' or key == '\t':
            word = "\t"
            char = '\t'
        else:
            if isinstance(key, str):
                word += key
            else:
                try:
                    word += str(key, "ISO-8859-1")
                except ValueError:
                    yield UserInputObject.ve()

            char = key

        yield UserInputObject.default(char, word, offset_x, offset_y)

    return UserInputObject.final()


@default_export
def get_generator():
    def helper():
        return get_character()

    return helper


@default_export
def update_autocompletion_mods(app: App, autocompletion_dict=None):
    if app is None:
        app = get_app(from_="cliF.update_autocompletion_mods")
    if autocompletion_dict is None:
        autocompletion_dict = {}

    for module_name, module in app.functions.items():
        data = {}
        for function_name, function_data in app.functions[module_name].items():
            if not isinstance(function_data, dict):
                continue
            data[function_name] = {arg: None for arg in function_data.get("params", [])}  # TODO get default from sig
        autocompletion_dict[module_name] = data

    return autocompletion_dict


@default_export
def update_autocompletion_list_or_key(list_or_key: iter or None = None, autocompletion_dict=None, raise_e=True, do_lower=False):
    if list_or_key is None:
        list_or_key = []
    if autocompletion_dict is None:
        autocompletion_dict = {}

    for key in list_or_key:
        if raise_e and key in autocompletion_dict:
            raise ValueError(f"Naming Collision {key}")
        autocompletion_dict[key if do_lower else key] = None

    return autocompletion_dict


@export(mod_name=Name, test=False)
def user_input(app,
               completer_dict=None,
               get_rprompt=None,
               bottom_toolbar=None,
               active_modul="",
               password=False,
               bindings=None,
               message=f"~{node()}@>") -> CallingObject:
    if app is None:
        app = get_app(from_="cliF.user_input")
    if completer_dict is None:
        completer_dict = {}
    if not PROMPT_TOOLKIT:
        raise ImportError("prompt toolkit is not available install via 'pip install prompt-toolkit'")
    if app is None:
        app = get_app("cli_functions.user_input")
    if get_rprompt is None:
        get_rprompt = lambda: ""
    if bottom_toolbar is None:
        def bottom_toolbar_helper():
            return HTML(f'Hotkeys shift:s control:c  <b><style bg="ansired">s+left</style></b> helper info '
                        f'<b><style bg="ansired">c+space</style></b> Autocompletion tips '
                        f'<b><style bg="ansired">s+up</style></b> run in shell')

        bottom_toolbar = bottom_toolbar_helper

    if bindings is None:
        bindings = KeyBindings()

    completer = NestedCompleter.from_nested_dict(completer_dict)
    fh = FileHistory(f'{app.data_dir}/{app.args_sto.modi}-cli.txt')
    auto_suggest = AutoSuggestFromHistory()

    @bindings.add('s-up')
    def run_in_shell(event):
        buff = event.app.current_buffer.text

        def run_in_console():
            if buff.startswith('cd'):
                print("CD not available")
                return
            fh.append_string(buff)
            os.system(buff)

        run_in_terminal(run_in_console)
        event.app.current_buffer.text = ""

    @bindings.add('s-left')
    def user_helper(event):

        buff = event.app.current_buffer.text.strip()

        def print_help():
            if buff == "":
                print("All commands: ", completer_dict)
            user_input_buffer_info = buff.split(" ")
            if len(user_input_buffer_info) == 1:
                if user_input_buffer_info[0] in completer_dict:
                    print("Avalabel functions:", completer_dict[user_input_buffer_info[0]])
                else:
                    print("Module is not available")
            if len(user_input_buffer_info) > 1:
                if user_input_buffer_info[0] in completer_dict:
                    if user_input_buffer_info[1] in completer_dict[user_input_buffer_info[0]]:
                        print("Avalabel args:", completer_dict[user_input_buffer_info[0]][user_input_buffer_info[1]])
                else:
                    print("Module is not available")

        run_in_terminal(print_help)

    @bindings.add('c-space')
    def state_completion(event):
        " Initialize autocompletion, or select the next completion. "
        buff = event.app.current_buffer
        if buff.complete_state:
            buff.complete_next()
        else:
            buff.start_completion(select_first=False)

    if not os.path.exists(f'{app.data_dir}/{app.args_sto.modi}-cli.txt'):
        open(f'{app.data_dir}/{app.args_sto.modi}-cli.txt', "a")

    session = PromptSession(message=message,
                            history=fh,
                            color_depth=ColorDepth.TRUE_COLOR,
                            # lexer=PygmentsLexer(l),
                            clipboard=InMemoryClipboard(),
                            auto_suggest=auto_suggest,
                            # prompt_continuation=0,
                            rprompt=get_rprompt,
                            bottom_toolbar=bottom_toolbar,
                            mouse_support=True,
                            key_bindings=bindings,
                            completer=completer,
                            refresh_interval=60,
                            reserve_space_for_menu=4,
                            complete_in_thread=True,
                            is_password=password,
                            )
    call_obj = CallingObject.empty()
    try:
        text = session.prompt(default=active_modul,
                              mouse_support=True)
    except KeyboardInterrupt:

        return user_input(app, completer_dict, get_rprompt, bottom_toolbar, active_modul)
    except EOFError:

        return user_input(app, completer_dict, get_rprompt, bottom_toolbar, active_modul)
    else:
        infos = text.split(" ")

        if len(infos) >= 1:
            call_obj.module_name = infos[0]
        if len(infos) >= 2:
            call_obj.function_name = infos[1]
        if len(infos) > 2:
            call_obj.kwargs = {}
            call_obj.args = infos[2:]
            if call_obj.module_name not in completer_dict:
                return call_obj
            if call_obj.function_name not in completer_dict[call_obj.module_name]:
                return call_obj
            kwargs_name = completer_dict[call_obj.module_name][call_obj.function_name].get(
                'params')  # TODO FIX parsm ist type list
            if kwargs_name is None:
                return call_obj
            kwargs_name = kwargs_name.remove('app').remove('self')
            call_obj.kwargs = dict(zip(kwargs_name, infos[2:]))
        return call_obj


@export(mod_name=Name, samples=[{
                                 "obj": CallingObject(
                                     module_name="welcome",
                                     function_name="show_version",
                                 ),
                                 "build_in_commands": {},
                                 }])
def co_evaluate(app: App,
                obj: CallingObject or None,
                build_in_commands: dict,
                threaded=False,
                helper=None,
                return_parm=False
                ):
    if obj is None:
        return Result.default_user_error(info="No object specified")

    if app is None:
        app = get_app(from_="cliF.co_evaluate")
    command = obj.module_name

    if not command:
        return Result.default_user_error(info="No module Provided").set_origin("cli_functions.co_evaluate").print()

    if command in build_in_commands:
        return build_in_commands[command](obj).print()

    function_name = obj.function_name

    if not function_name:
        return Result.default_user_error(info="No function Provided").set_origin("cli_functions.co_evaluate").print()

    if obj.kwargs is None:
        obj.kwargs = {}

    if helper is None:
        def helper_function(obj_):

            # obj_.print()
            result = app.run_any((obj_.module_name, obj_.function_name), get_results=True,
                                 args_=obj_.args,
                                 kwargs_=obj_.kwargs)

            result.print()

            if isinstance(return_parm, list):
                return_parm[0] = result
            elif return_parm:
                return return_parm
            else:
                return None

        helper = helper_function

    if threaded:
        t = Thread(target=helper, args=(obj,))
        if return_parm:
            return_parm = [Result.default_internal_error(info="No Data"), 0]
        t.start()
        return t

    return helper(obj)
