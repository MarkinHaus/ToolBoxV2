import asyncio
import os
import re
import subprocess
from dataclasses import dataclass, field
from threading import Thread

from toolboxv2.utils.system.types import CallingObject, ApiResult

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


def parse_linux_command_output():
    # Führe den Linux-Befehl aus und erhalte die Ausgabe
    result = subprocess.run(["bash", "-c", "compgen -A function -abck"], capture_output=True, text=True)
    output = result.stdout

    # Trennen der Daten in Zeilen
    lines = output.strip().split('\n')

    # Hier könnten weitere Filter- oder Verarbeitungsschritte hinzugefügt werden,
    # falls die Daten weitergehend bereinigt oder analysiert werden müssten.

    return lines


def parse_command_output():
    # Führe den PowerShell-Befehl aus und erhalte die Ausgabe
    result = subprocess.run(["powershell", "-Command", "Get-Command *"], capture_output=True, text=True)
    output = result.stdout

    # Trennen der Daten in Zeilen
    lines = output.split('\n')

    # Erstellen einer Liste für die Namen der Dateien und ein Dictionary für die gesamten Daten
    exe_names = []
    applications = []

    # Schleife über die Zeilen der Ausgabe, um die Daten zu extrahieren
    for line in lines:
        parts = line.split()

        if len(parts) >= 2 and parts[0] == "Alias":
            # Nehme an, dass der Dateiname und die Version immer vorhanden sind
            # und der Rest der Zeile ist der Pfad
            app_type = parts[0]
            exe_name = parts[1]
            if len(exe_name) < 2:
                continue

            location = ' '.join(parts[3:])

            # Füge den Dateinamen zur Liste hinzu
            exe_names.append(exe_name)

            # Erstelle ein Dictionary für die aktuelle Anwendung und füge es zur Liste hinzu
            application_info = {
                'type': app_type,
                'name': exe_name,
                'infos': location
            }
            applications.append(application_info)

        if len(parts) >= 2 and parts[0] == "Cmdlet":
            # Nehme an, dass der Dateiname und die Version immer vorhanden sind
            # und der Rest der Zeile ist der Pfad
            app_type = parts[0]
            exe_name = parts[1]
            if len(exe_name) < 2:
                continue

            location = ' '.join(parts[2:])

            # Füge den Dateinamen zur Liste hinzu
            exe_names.append(exe_name)

            # Erstelle ein Dictionary für die aktuelle Anwendung und füge es zur Liste hinzu
            application_info = {
                'type': app_type,
                'name': exe_name,
                'infos': location
            }
            applications.append(application_info)

        if len(parts) >= 3 and parts[0] == "ExternalScript":
            # Nehme an, dass der Dateiname und die Version immer vorhanden sind
            # und der Rest der Zeile ist der Pfad
            app_type = parts[0]
            exe_name = parts[1]
            if len(exe_name) < 2:
                continue

            location = ' '.join(parts[2:])

            # Füge den Dateinamen zur Liste hinzu
            exe_names.append(exe_name)

            # Erstelle ein Dictionary für die aktuelle Anwendung und füge es zur Liste hinzu
            application_info = {
                'type': app_type,
                'name': exe_name,
                'infos': location
            }
            applications.append(application_info)
        if len(parts) >= 2 and parts[0] == "Function":
            # Nehme an, dass der Dateiname und die Version immer vorhanden sind
            # und der Rest der Zeile ist der Pfad
            app_type = parts[0]
            exe_name = parts[1]
            if len(exe_name) < 2 and not exe_name.endswith(':'):
                continue

            location = ' '.join(parts[2:])

            # Füge den Dateinamen zur Liste hinzu
            exe_names.append(exe_name)

            # Erstelle ein Dictionary für die aktuelle Anwendung und füge es zur Liste hinzu
            application_info = {
                'type': app_type,
                'name': exe_name,
                'infos': location
            }
            applications.append(application_info)

        if len(parts) >= 4 and parts[0] == "Application":
            # Nehme an, dass der Dateiname und die Version immer vorhanden sind
            # und der Rest der Zeile ist der Pfad
            app_type = parts[0]
            exe_name = parts[1]
            version = parts[2]
            location = ' '.join(parts[3:])

            # Füge den Dateinamen zur Liste hinzu
            exe_names.append(exe_name)

            # Erstelle ein Dictionary für die aktuelle Anwendung und füge es zur Liste hinzu
            application_info = {
                'type': app_type,
                'name': exe_name,
                'version': version,
                'location': location
            }
            applications.append(application_info)

    return exe_names, applications


def replace_bracketed_content(text, replacements, inlist=False):
    """
    Ersetzt Inhalte in eckigen Klammern mit entsprechenden Werten aus einem Wörterbuch.

    :param text: Der zu verarbeitende Text als String.
    :param replacements: Ein Wörterbuch mit Schlüssel-Wert-Paaren für die Ersetzung.
    :return: Den modifizierten Text.
    """
    # Finde alle Vorkommen von Texten in eckigen Klammern
    matches = re.findall(r'\[([^\]]+)\]', text)

    # Ersetze jeden gefundenen Text durch den entsprechenden Wert aus dem Wörterbuch
    as_list = text.split(' ')
    i = 0
    for key in matches:
        if key in replacements:
            if not inlist:
                text = text.replace(f'[{key}]', str(replacements[key]))
            else:
                as_list[i] = replacements[key]
        i += 1
    if not inlist:
        return text
    return as_list


@export(mod_name=Name, name='Version', version=version)
def get_version():
    return version


@dataclass
class UserInputObject:
    char: chr or str or None = field(default=None)
    word: str or None = field(default=None)
    offset_x: int or None = field(default=None)
    offset_y: int or None = field(default=None)

    def is_last(self) -> bool:
        return "LAST" == self.char

    def is_v_error(self) -> bool:
        return "ValueError" == self.char

    @classmethod
    def default(cls,
                char: chr or str or None = None,
                word: str or None = None,
                offset_x: int or None = None,
                offset_y: int or None = None):
        return cls(
            char=char,
            word=word,
            offset_x=offset_x,
            offset_y=offset_y,
        )

    @classmethod
    def final(cls):
        return cls(char="LAST",
                   word="LAST",
                   offset_x=0,
                   offset_y=0)

    @classmethod
    def ve(cls):
        return cls(char="ValueError",
                   word="ValueError",
                   offset_x=0,
                   offset_y=0, )


@default_export
def get_character():
    get_input = True

    offset_x = 0
    offset_y = 0
    word = ""
    char = ''

    # session_history += [c for c in app.command_history]

    print("-->", end='\r')
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

    app.save_autocompletion_dict()
    autocompletion_dict_ = app.get_autocompletion_dict()

    if autocompletion_dict_ is not None:
        autocompletion_dict = {**autocompletion_dict_, **autocompletion_dict}

    return autocompletion_dict


@default_export
def update_autocompletion_list_or_key(list_or_key: iter or None = None, autocompletion_dict=None, raise_e=True,
                                      do_lower=False):
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
               message=f"~{node()}@>", fh=None) -> CallingObject:
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
    if fh is None:
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

    @bindings.add('c-up')
    def run_in_shell(event):
        buff = event.app.current_buffer.text

        def run_in_console():
            if buff.startswith('cd'):
                print("CD not available")
                return
            fh.append_string(buff)
            if app.locals['user'].get('counts') is None:
                app.locals['user']['counts'] = 0

            try:
                result = eval(buff, app.globals['root'], app.locals['user'])
                if result is not None:
                    print(f"+ {buff}\n#{app.locals['user']['counts']}>", result)
                else:
                    print(f"- {buff}\n#{app.locals['user']['counts']}>")
            except SyntaxError:
                exec(buff, app.globals['root'], app.locals['user'])
                print(f"* {buff}\n#{app.locals['user']['counts']}> Statement executed")
            except Exception as e:
                print(f"Error: {e}")

            app.locals['user']['counts'] += 1

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
                    os.system(f"{user_input_buffer_info[0]} --help")
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
        text = session.prompt(default=active_modul, mouse_support=True, in_thread=True)
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
            if call_obj.function_name not in (
                completer_dict[call_obj.module_name] if completer_dict[call_obj.module_name] is not None else {}):
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
    ),
    "build_in_commands": {},
}])
async def co_evaluate(app: App,
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
        if command == "exit":
            await build_in_commands[command](None)
            return
        if asyncio.iscoroutinefunction(build_in_commands[command]):
            res = await build_in_commands[command](obj)
        else:
            res = build_in_commands[command](obj)
        return res.print()

    function_name = obj.function_name

    if not function_name:
        return Result.default_user_error(info="No function Provided").set_origin("cli_functions.co_evaluate").print()

    if obj.kwargs is None:
        obj.kwargs = {}

    if app.locals['user'].get('res_id') is None:
        app.locals['user']['res_id'] = 0

    if helper is None:
        def helper_function(obj_):

            # obj_.print()
            result = app.run_any((obj_.module_name, obj_.function_name), get_results=True,
                                 args_=obj_.args,
                                 kwargs_=obj_.kwargs)

            app.locals['user'][f"result{app.locals['user']['res_id']}"] = result
            print(f"\n~> result{app.locals['user']['res_id']} = ")
            app.locals['user']['res_id'] += 1

            if isinstance(result, Result) or isinstance(result, ApiResult):
                result.print()
            else:
                print(result)

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
