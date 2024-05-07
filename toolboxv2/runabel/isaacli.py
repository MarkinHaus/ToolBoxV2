import os

from prompt_toolkit import HTML
from prompt_toolkit.shortcuts import set_title, yes_no_dialog

from toolboxv2 import App, Result, tbef
from toolboxv2.utils import Style
from toolboxv2.utils.extras.Style import cls, Spinner
from toolboxv2.utils.system.types import CallingObject

NAME = 'isaacli'


def run(app: App, args):
    from toolboxv2.mods.isaa.Agents import Agent, AgentBuilder, LLMMode
    from toolboxv2.mods.isaa import Tools
    from toolboxv2.mods.isaa.AgentFramwork import CodingCapability

    set_title(f"ToolBox : {app.version}")
    threaded = False

    def bottom_toolbar():
        return HTML(f'Hotkeys shift:s control:c  <b><style bg="violet">s+left</style></b> helper info '
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

    def run_(call_: CallingObject) -> Result:
        if not call_.function_name:
            return (Result.default_user_error(info=f"Avalabel are : {list(app.runnable.keys())}")
                    .set_origin("minicli::build-in"))
        if call_.function_name in app.runnable:
            app.run_runnable(call_.function_name)
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

    bic = {
        "exit": exit_,
        "cls": cls_,
        "sdm:set_debug_mode": set_debug_mode,
        "open": open_,
        "close": close_,
        "run": run_,
        "infos": infos,
        "reload": hr,
        "remote": remote,
        "toggle_threaded": toggle_threaded,
        "..": lambda x: Result.ok(x),
    }

    all_modes = app.get_all_mods()

    # set up Autocompletion
    autocompletion_dict = {}
    autocompletion_dict = app.run_any(tbef.CLI_FUNCTIONS.UPDATE_AUTOCOMPLETION_LIST_OR_KEY, list_or_key=bic,
                                      autocompletion_dict=autocompletion_dict)

    autocompletion_dict["sdm:set_debug_mode"] = {arg: None for arg in ['on', 'off']}
    autocompletion_dict["open"] = autocompletion_dict["close"] = autocompletion_dict["reload"] = \
        {arg: None for arg in all_modes}
    autocompletion_dict["run"] = {arg: None for arg in list(app.runnable.keys())}
    autocompletion_dict = app.run_any(tbef.CLI_FUNCTIONS.UPDATE_AUTOCOMPLETION_MODS,
                                      autocompletion_dict=autocompletion_dict)

    active_modular = ""

    running_instance = None
    call = CallingObject.empty()

    isaa: Tools = app.get_mod("isaa")

    def print_prompt(msg_data):

        messages = msg_data.get('messages', {})
        print(Style.GREEN2("PROMPT START "))
        for message in messages:
            caller = Style.WHITE(message.get('role', 'NONE').upper()) if message.get('role',
                                                                                     'NONE') == 'user' else 'NONE'
            caller = Style.CYAN(message.get('role', 'NONE').upper()) if message.get('role',
                                                                                    'NONE') == 'system' else caller
            caller = Style.VIOLET2(message.get('role', 'NONE').upper()) if message.get('role',
                                                                                       'NONE') == 'assistent' else caller
            print(f"\n{caller}\n{Style.GREY(message.get('content', '--#--'))}\n")
        print(Style.GREEN("PROMPT END -- "))

    isaa.register_agents_setter(lambda x: x
                                .set_amd_model("ollama/llama2")
                                .set_stream(True)
                                .set_logging_callback(print_prompt)
                                # .set_logging_callback(isaa.print)
                                .set_verbose(True)
                                .set_max_tokens(1200)
                                .set_amd_stop_sequence(['\n\n\n'])
                                )
    isaa.init_isaa()
    with Spinner("Start pipline zero-shot classification"):
        isaa.init_pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    if not isaa.controller.registered("Greetings"):
        isaa.controller.add("Greetings", LLMMode(name="Greetings",
                                                 description="geet the user and get him day redy",
                                                 system_msg="Greet the user and give 1 useful tip"
                                                 ))
    if not isaa.controller.registered("CommandRunner"):
        isaa.controller.add("CommandRunner", LLMMode(name="CommandRunner",
                                                     description="Convert Natural Language to shell commands",
                                                     system_msg="Convert the input to shell command and run the command "
                                                                "withe FUNCTION: {'Action': 'shell', 'Input': str}"
                                                     ))

    app.alive = True

    labels = ["statement", "question", "coding-task", "command"]
    # statement -> true / false - 1 to 0
    # question -> time / personenbezogen
    q_labels = ["time/location", "person", "system"]
    # call -> fuction python / eval ece
    c_labels = ["python", "shell", "custom"]

    # command -> shell
    user_text = ""
    _agent_m: Agent = isaa.get_agent_class("self")
    _agent_m.mode = isaa.controller.get("Greetings")

    _agent_c_builder: AgentBuilder = isaa.get_default_agent_builder("code")
    _agent_c_builder.set_capabilities(CodingCapability)
    _agent_c = isaa.register_agent(_agent_c_builder)

    _agent_s_builder: AgentBuilder = isaa.get_default_agent_builder("shell")
    (_agent_s_builder.set_max_tokens(60)
     .set_amd_system_message("U ar Shell agent only return the fitting shell command else None"))
    _agent_s = isaa.register_agent(_agent_s_builder)
    _agent_s.mode = isaa.controller.get("CommandRunner")

    # isaa.run_agent("self",
    #                "Hi Isaa.",
    #                running_mode='once',
    #                persist=False,
    #                fetch_memory=True,
    #                persist_mem=False,
    #                persist_local=True)

    for tool in isaa.get_agent_class("self").functions:
        app.locals['user'][tool.name] = tool.function

    _agent_m.mode = None

    while app.alive:

        def get_rprompt():
            return HTML(f'ISAA')

        next_fuction = isaa.get_agent_class("self").next_fuction
        if next_fuction is not None:
            print(f"Next fuction isaa whats to call {next_fuction}")
            if isaa.get_agent_class("self").llm_function_runner is None:
                print("No llm function runner available")
            elif 'y' in input('Run function Yes / No').lower():
                out = isaa.get_agent_class("self").execute_fuction()
                print(f"Eval:", out)
            else:
                pass

        call = app.run_any(tbef.CLI_FUNCTIONS.USER_INPUT, completer_dict=autocompletion_dict,
                           get_rprompt=get_rprompt, bottom_toolbar=bottom_toolbar, active_modul=active_modular)

        print("", end="" + "start ->>\r")

        if call is None:
            continue

        if call.module_name == "open":
            autocompletion_dict = app.run_any(tbef.CLI_FUNCTIONS.UPDATE_AUTOCOMPLETION_MODS,
                                              autocompletion_dict=autocompletion_dict)

        if call.module_name in autocompletion_dict.keys():
            running_instance_res = app.run_any(tbef.CLI_FUNCTIONS.CO_EVALUATE,
                                               obj=call,
                                               build_in_commands=bic,
                                               threaded=threaded,
                                               helper=helper_exequtor[0], get_results=True)
            running_instance = running_instance_res.get()
            if not running_instance_res.is_error():
                continue

        user_text = str(call).strip()

        if user_text == '':
            continue

        if user_text == 'exit':
            do_next = 2
            return_parm = [Result.ok()]
        else:
            # print("Processioning :", user_text)
            with Spinner(message=f"Analysing input len-{len(user_text)}...", symbols='d'):
                a0 = isaa.run_any_hf_pipline("zero-shot-classification", model="facebook/bart-large-mnli",
                                             sequences=user_text,
                                             candidate_labels=labels)
            do_next = labels.index(a0.get('labels', ["statement"])[0])
            print(f"Next Running:{a0.get('labels', [' Not Clear'])[0]} with accuracy :{a0.get('scores', [-1])[0]:.2f}")
            return_parm = [Result.default_user_error(exec_code=-999)]
        out = ""
        if do_next == 0:  # statement                if user_text.startswith('cd'):
            if app.locals['user'].get('counts') is None:
                app.locals['user']['counts'] = 0
            er = False
            try:
                result = eval(user_text, app.globals['root'], app.locals['user'])
                if result is not None:
                    print(f"#{app.locals['user']['counts']}>", result)
                else:
                    print(f"#{app.locals['user']['counts']}>")
                return_parm = [Result.ok(data=out)]
            except SyntaxError:
                try:
                    exec(user_text, app.globals['root'], app.locals['user'])
                    print(f"#{app.locals['user']['counts']}> Statement executed")
                    return_parm = [Result.ok(data=out)]
                except:
                    er = True
                    pass
            except Exception as e:
                er = True
                print("Running Isaa on statement failed to exec", e)
            if er:
                out = isaa.run_agent("self", f"User Locals Variables : {app.locals['user']}\nUsr Input:" + user_text,
                                     running_mode='oncex', persist=False)
                return_parm = [Result.ok(data=out)]

        if do_next == 1:  # question
            with Spinner(message=f"Analysing question type...", symbols='d'):
                a1 = isaa.run_any_hf_pipline("zero-shot-classification", model="facebook/bart-large-mnli",
                                             sequences=user_text,
                                             candidate_labels=q_labels)
                return_parm = [Result.ok(data=a1)]
            print(
                f"Question is {a1.get('labels', ['None'])[0]} Relevant with accuracy :{a1.get('scores', [-1])[0]:.2f} ")
            do_next_ = q_labels.index(a1.get('labels', ["time"])[0])

            if do_next_ == 0:  # time
                out = isaa.run_agent("self", user_text, max_iterations=2, running_mode="lineIs", persist=False,
                                     persist_local=False)
                return_parm = [Result.ok(data=out)]

            if do_next_ == 1:  # person
                print("Retrieving Memory Date")
                memdata = isaa.get_context_memory().get_context_for(user_text)
                if memdata:
                    isaa.get_agent_class("self").messages.append({"content": memdata, "role": "system"})
                return_parm = [Result.ok(data=memdata)]
                isaa.run_agent("self", user_text, running_mode="once")
            if do_next_ == 2:  # system
                out = isaa.run_agent("self", user_text, max_iterations=2, running_mode="lineIs", persist=False)
                return_parm = [Result.ok(data=out)]

        if do_next == 2:  # coding-task
            pass

        if return_parm[0].is_error():
            isaa.run_agent("self", user_text + ' ' + out, persist=False)

        if do_next == 3:  # command
            with Spinner(message=f"Analysing command type...", symbols='d'):
                a2 = isaa.run_any_hf_pipline("zero-shot-classification", model="facebook/bart-large-mnli",
                                             sequences=user_text,
                                             candidate_labels=c_labels)
            print(
                f"Question is {a2.get('labels', ['None'])[0]} Relevant with accuracy :{a2.get('scores', [-1])[0]:.2f}")
            do_next_ = c_labels.index(a2.get('labels', ["python"])[0])

            if do_next_ == 0:  # python
                if user_text.startswith('cd'):
                    print("CD not available")
                    return
                if app.locals['user'].get('counts') is None:
                    app.locals['user']['counts'] = 0
                try:
                    result = eval(user_text, app.globals['root'], app.locals['user'])
                    if result is not None:
                        print(f"#{app.locals['user']['counts']}>", result)
                    else:
                        print(f"#{app.locals['user']['counts']}>")
                except SyntaxError:
                    exec(user_text, app.globals['root'], app.locals['user'])
                    print(f"#{app.locals['user']['counts']}> Statement executed")
                except Exception as e:
                    print(f"Error: {e}")

            if do_next_ == 1:  # shell
                os.system(user_text)

            if do_next_ == 2:  # custom
                isaa.run_agent("shell", user_text, running_mode="oncex", persist=False)

        print("", end="" + "done ->>\r")

    if running_instance is not None:
        print("Closing running instance")
        running_instance.join()
        print("Done")

    set_title("")
