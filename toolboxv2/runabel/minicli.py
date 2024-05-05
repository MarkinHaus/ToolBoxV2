import asyncio
import datetime
import os
import threading

import psutil
from prompt_toolkit import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import set_title

from toolboxv2 import App, Result, tbef
from toolboxv2.mods.cli_functions import parse_command_output, parse_linux_command_output, replace_bracketed_content
from toolboxv2.utils import show_console
from toolboxv2.utils.extras.Style import cls, Spinner, Style
from toolboxv2.utils.system.types import CallingObject

NAME = 'cli'


def run_in_console(buff, fh, pw=False):
    # if buff.startswith('cd'):
    #     print("CD not available")
    #     return
    fh.append_string(buff)
    print(Style.BEIGE2('## ') + buff)
    _ = ""
    if pw:
        _ = "powershell -Command "
    else:
        _ = "bash -c "
    os.system(_ + buff)


def run_in_terminal(app, buff, fh):
    if app.locals['user'].get('counts') is None:
        app.locals['user']['counts'] = 0

    try:
        result = eval(buff, app.globals['root'], app.locals['user'])
        if result is not None:
            print(f"+ #{app.locals['user']['counts']}>", result)
        else:
            print(f"- #{app.locals['user']['counts']}>")
    except SyntaxError:
        try:
            exec(buff, app.globals['root'], app.locals['user'])
            print(f"* #{app.locals['user']['counts']}> Statement executed")
        except Exception as e:
            print(f"Error: {e}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

    fh.append_string(buff)
    app.locals['user']['counts'] += 1
    return True


async def run(app: App, args):
    try:
        set_title(f"ToolBox : {app.version}")
    except:
        pass
    threaded = False

    def bottom_toolbar():
        return HTML(f'Hotkeys shift:s control:c  <b><style bg="ansired">s+left</style></b> helper info '
                    f'<b><style bg="ansired">c+space</style></b> Autocompletion tips '
                    f'<b><style bg="ansired">s+up</style></b> run in shell')

    async def exit_(_):
        print("EXITING")
        if app.debug:
            await app.hide_console()
        app.alive = False
        return Result.ok().set_origin("minicli::build-in")

    def set_debug_mode(call_: CallingObject) -> Result:
        if not call_.function_name:
            return (Result.default_user_error(info=f"sdm (Set Debug Mode) needs at least one argument on or off\napp is"
                                                   f" {'' if app.debug else 'NOT'} in debug mode")
                    .set_origin("minicli::build-in"))
        if call_.function_name.lower() == "on":
            app.debug = True
        elif call_.function_name.lower() == "off":
            app.debug = False
        else:
            return Result.default_user_error(info=f"{call_.function_name} != on or off").set_origin("minicli::build-in")
        return Result.ok(info=f"New Debug Mode {app.debug}").set_origin("minicli::build-in")

    def hr(call_: CallingObject) -> Result:
        if not call_.function_name:
            app.remove_all_modules()
            app.load_all_mods_in_file()
        if call_.function_name in app.functions:
            app.remove_mod(call_.function_name)
            if not app.save_load(call_.function_name):
                return Result.default_internal_error().set_origin("minicli::build-in")
        return Result.ok().set_origin("minicli::build-in")

    def open_(call_: CallingObject) -> Result:
        if not call_.function_name:
            app.load_all_mods_in_file()
            return Result.default_user_error(info="No module specified").set_origin("minicli::build-in")
        if not app.save_load(call_.function_name):
            return Result.default_internal_error().set_origin("minicli::build-in")
        return Result.ok().set_origin("minicli::build-in")

    def close_(call_: CallingObject) -> Result:
        if not call_.function_name:
            app.remove_all_modules()
            return Result.default_user_error(info="No module specified").set_origin("minicli::build-in")
        if not app.remove_mod(call_.function_name):
            return Result.default_internal_error().set_origin("minicli::build-in")
        return Result.ok().set_origin("minicli::build-in")

    async def run_(call_: CallingObject) -> Result:
        if not call_.function_name:
            return (Result.default_user_error(info=f"Avalabel are : {list(app.runnable.keys())}")
                    .set_origin("minicli::build-in"))
        if call_.function_name in app.runnable:
            await app.run_runnable(call_.function_name)
            return Result.ok().set_origin("minicli::build-in")
        return Result.default_user_error("404").set_origin("minicli::build-in")

    helper_exequtor = [None]

    def remote(call_: CallingObject) -> Result:
        if not call_.function_name:
            return Result.default_user_error(info="add keyword local or port and host").set_origin("minicli::build-in")
        if call_.function_name != "local":
            app.args_sto.host = call_.function_name
        if call_.kwargs:
            print("Adding", call_.kwargs)
        status, sender, receiver_que = app.run_runnable("daemon", as_server=False, programmabel_interface=True)
        if status == -1:
            return (Result.default_internal_error(info="Failed to connect, No service available")
                    .set_origin("minicli::build-in"))

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

        return Result.ok().set_origin("minicli::build-in")

    def cls_(_):
        cls()
        return Result.ok(info="cls").set_origin("minicli::build-in")

    def toggle_threaded(_):
        global threaded
        threaded = not threaded
        return Result.ok(info=f"in threaded mode {threaded}").set_origin("minicli::build-in").print()

    def infos(_):
        app.print_functions()
        return Result.ok(info=f"").set_origin("minicli::build-in")

    def colose_console(_):
        show_console(False)
        return Result.ok(info=f"").set_origin("minicli::build-in")

    def open_console(_):
        app.show_console(True)
        return Result.ok(info=f"").set_origin("minicli::build-in")

    bic = {
        "exit": exit_,
        "cls": cls_,
        "sdm:set_debug_mode": set_debug_mode,
        "openM": open_,
        "closeM": close_,
        "runM": run_,
        "infos": infos,
        "reload": hr,
        "remote": remote,
        "hide_console": colose_console,
        "show_console": open_console,
        "toggle_threaded": toggle_threaded,
        "..": lambda x: Result.ok(x),
    }

    all_modes = app.get_all_mods()

    # set up Autocompletion

    autocompletion_dict = {}
    autocompletion_dict = app.run_any(tbef.CLI_FUNCTIONS.UPDATE_AUTOCOMPLETION_LIST_OR_KEY, list_or_key=bic,
                                      autocompletion_dict=autocompletion_dict)

    autocompletion_dict_ = app.get_autocompletion_dict()
    if autocompletion_dict is None:
        autocompletion_dict = {}

    if autocompletion_dict_ is not None:
        while asyncio.iscoroutine(autocompletion_dict_):
            autocompletion_dict_ = await autocompletion_dict_
        while asyncio.iscoroutine(autocompletion_dict):
            autocompletion_dict = await autocompletion_dict
        if isinstance(autocompletion_dict, dict) and isinstance(autocompletion_dict_, dict):
            autocompletion_dict = {**autocompletion_dict, **autocompletion_dict_}

    autocompletion_dict["sdm:set_debug_mode"] = {arg: None for arg in ['on', 'off']}
    autocompletion_dict["openM"] = autocompletion_dict["closeM"] = autocompletion_dict["reload"] = \
        {arg: None for arg in all_modes}
    autocompletion_dict["runM"] = {arg: None for arg in list(app.runnable.keys())}

    active_modular = ""

    with Spinner("importing System Commands"):

        if app.system_flag == "Windows":
            exe_names, _ = parse_command_output()
        else:
            exe_names = parse_linux_command_output()

        for exe in exe_names:
            autocompletion_dict[exe] = None

    running_instance = None
    call = CallingObject.empty()
    running = True
    fh = FileHistory(f'{app.data_dir}/{app.args_sto.modi}-cli.txt')
    print("", end="" + "start ->>\r")
    while running:
        # Get CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)

        # Get memory usage
        memory_usage = psutil.virtual_memory().percent

        # Get disk usage
        disk_usage = psutil.disk_usage('/').percent

        def get_rprompt():
            current_time: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return HTML(
                f'<b> App Infos: '
                f'{app.id} \nCPU: {cpu_usage}% Memory: {memory_usage}% Disk :{disk_usage}%\nTime: {current_time}</b>')

        call = app.run_any(tbef.CLI_FUNCTIONS.USER_INPUT, completer_dict=autocompletion_dict,
                           get_rprompt=get_rprompt, bottom_toolbar=bottom_toolbar, active_modul=active_modular, fh=fh)

        if asyncio.iscoroutine(call):
            call = await call
            print("D", call, call.module_name.split('.')[0])

        print("", end="" + "eval ->>\r")

        if call is None:
            continue
        if call.module_name == "openM":
            autocompletion_dict = app.run_any(tbef.CLI_FUNCTIONS.UPDATE_AUTOCOMPLETION_MODS,
                                              autocompletion_dict=autocompletion_dict)
        elif call.module_name.split('.')[0] in all_modes or call.module_name in bic.keys():
            if call.function_name.strip() == '' and call.module_name not in bic.keys():
                app.print_functions(call.module_name)
            else:
                if call.args is not None:
                    call.args = replace_bracketed_content(' '.join(call.args), app.locals['user'], inlist=True)
                running_instance = await app.run_any(tbef.CLI_FUNCTIONS.CO_EVALUATE,
                                                     obj=call,
                                                     build_in_commands=bic,
                                                     threaded=threaded,
                                                     helper=helper_exequtor[0])
        elif call.module_name in exe_names:
            buff = str(call)
            buff = replace_bracketed_content(buff, app.locals['user'])
            run_in_console(buff, fh, app.system_flag == "Windows" and "powershell.exe" in exe_names)
            running_instance = None
        elif len(str(call).strip()) == 0:
            app.print_functions()
        else:
            buff = str(call)
            buff = replace_bracketed_content(buff, app.locals['user'])
            res_ = run_in_terminal(app, buff, fh)
            running_instance = None
            if not res_:
                pass  # shell ginei

        if isinstance(running_instance, asyncio.Task):
            v = await running_instance
            running_instance = None

        print("", end="" + "done ->>\r")
        running = app.alive

    if hasattr(app, 'timeout'):
        app.timeout = 2
    if running_instance is not None:
        print("Closing running instance")
        if isinstance(running_instance, Result):
            running_instance.print()
        elif isinstance(running_instance, threading.Thread):
            running_instance.join()
        elif isinstance(running_instance, asyncio.Task):
            await running_instance
        else:
            print(running_instance)
        print("Done")

    try:
        set_title("")
    except:
        pass
    await app.a_exit()
