"""Console script for toolboxv2. Isaa CMD Tool"""
import os
import random
import time

from langchain.agents import load_tools

from toolboxv2 import Style, get_logger, App
from toolboxv2.mods.isaa_extars.isaa_modi import init_isaa, generate_exi_dict, \
    idea_enhancer, startage_task_aproche, free_run_in_cmd, get_code_files, \
    run_chain_in_cmd_auto_observation_que, sys_print

NAME = "isaa-l-auto"


def run(app: App, args):
    isaa, self_agent_config, chains = init_isaa(app, speak_mode=args.speak, calendar=False, ide=True, create=True,
                                                isaa_print=False, python_test=True, init_mem=True, init_pipe=True,
                                                join_now=False,
                                                global_stream_override=False, chain_runner=True)

    isaa.get_context_memory().load_all()
    isaa.agent_collective_senses = True
    isaa.summarization_mode = 1

    self_agent_config.stream = True
    self_agent_config.set_completion_mode('chat')
    self_agent_config.set_model_name('gpt-4')

    self_agent_config.stop_sequence = ['\n\n\n', "Execute:", "Observation:", "User:"]

    from langchain.tools import ShellTool
    from langchain.tools import AIPluginTool
    shell_tool = ShellTool()

    plugins = [
        # SceneXplain
        # "https://scenex.jina.ai/.well-known/ai-plugin.json",
        # Weather Plugin for getting current weather information.
        #    "https://gptweather.skirano.repl.co/.well-known/ai-plugin.json",
        # Transvribe Plugin that allows you to ask any YouTube video a question.
        #    "https://www.transvribe.com/.well-known/ai-plugin.json",
        # ASCII Art Convert any text to ASCII art.
        #    "https://chatgpt-plugin-ts.transitive-bullshit.workers.dev/.well-known/ai-plugin.json",
        # DomainsGPT Check the availability of a domain and compare prices across different registrars.
        # "https://domainsg.pt/.well-known/ai-plugin.json",
        # PlugSugar Search for information from the internet
        #    "https://websearch.plugsugar.com/.well-known/ai-plugin.json",
        # FreeTV App Plugin for getting the latest news, include breaking news and local news
        #    "https://www.freetv-app.com/.well-known/ai-plugin.json",
        # Screenshot (Urlbox) Render HTML to an image or ask to see the web page of any URL or organisation.
        # "https://www.urlbox.io/.well-known/ai-plugin.json",
        # OneLook Thesaurus Plugin for searching for words by describing their meaning, sound, or spelling.
        # "https://datamuse.com/.well-known/ai-plugin.json", -> long loading time
        # Shop Search for millions of products from the world's greatest brands.
        # "https://server.shop.app/.well-known/ai-plugin.json",
        # Zapier Interact with over 5,000+ apps like Google Sheets, Gmail, HubSpot, Salesforce, and thousands more.
        "https://nla.zapier.com/.well-known/ai-plugin.json",
        # Remote Ambition Search millions of jobs near you
        # "https://remoteambition.com/.well-known/ai-plugin.json",
        # Kyuda Interact with over 1,000+ apps like Google Sheets, Gmail, HubSpot, Salesforce, and more.
        # "https://www.kyuda.io/.well-known/ai-plugin.json",
        # GitHub (unofficial) Plugin for interacting with GitHub repositories, accessing file structures, and modifying code. @albfresco for support.
        #     "https://gh-plugin.teammait.com/.well-known/ai-plugin.json",
        # getit Finds new plugins for you
        "https://api.getit.ai/.well_known/ai-plugin.json",
        # WOXO VidGPT Plugin for create video from prompt
        "https://woxo.tech/.well-known/ai-plugin.json",
        # Semgrep Plugin for Semgrep. A plugin for scanning your code with Semgrep for security, correctness, and performance issues.
        "https://semgrep.dev/.well-known/ai-plugin.json",
    ]

    isaa.lang_chain_tools_dict = {
        "ShellTool": shell_tool,
        # "ReadFileTool": read_file_tool,
        # "CopyFileTool": copy_file_tool,
        # "DeleteFileTool": delete_file_tool,
        # "MoveFileTool": move_file_tool,
        # "WriteFileTool": write_file_tool,
        # "ListDirectoryTool": list_directory_tool,
    }

    for plugin_url in plugins:
        get_logger().info(Style.BLUE(f"Try opening plugin from : {plugin_url}"))
        try:
            plugin_tool = AIPluginTool.from_plugin_url(plugin_url)
            get_logger().info(Style.GREEN(f"Plugin : {plugin_tool.name} loaded successfully"))
            isaa.lang_chain_tools_dict[plugin_tool.name + "-usage-information"] = plugin_tool
        except Exception as e:
            get_logger().error(Style.RED(f"Could not load : {plugin_url}"))
            get_logger().error(Style.GREEN(f"{e}"))

    isaa.get_agent_config_class("think")
    execution_agent = isaa.get_agent_config_class("execution")

    for tool in load_tools(["requests_all", 'wikipedia', 'human']):
        isaa.lang_chain_tools_dict[tool.name] = tool

    execution_agent.tools = dict(execution_agent.tools, **self_agent_config.tools)

    isaa.add_lang_chain_tools_to_agent(execution_agent, execution_agent.tools)

    self_agent_config.tools = execution_agent.tools.copy()

    def sum_dir(dir_):

        code_and_md_files = get_code_files(dir_)

        def do_on_file(filename):
            time.sleep(random.randint(1, 100) / 100)
            description, file_doc = isaa.execute_thought_chain(filename,
                                                               chains.get("Generate_docs")
                                                               , self_agent_config)
            print("=" * 20)
            print(file_doc)
            print("Description:\n", Style.Bold(Style.BLUE(description)))
            print("=" * 20)

            isaa.get_context_memory().add_data(app.id, file_doc[-1][1])
            return file_doc[-1][1]

        deta = []
        for file in code_and_md_files:
            deta.append(do_on_file(file))

        return deta

    isaa.add_tool("ReadDir", sum_dir, "get and save dir information to memory", "ReadDir(path)", self_agent_config)
    state_save_file = f".data/{app.id}/StateSave.state"
    print(Style.CYAN(f"Save File : {state_save_file}"))
    state_save = {}

    def save_state(state_name, content):
        state_save[state_name] = content
        app.pretty_print_dict(state_save)
        try:
            with open(state_save_file, "w") as f1:
                f1.write(str(state_save))
        except Exception as e:
            print(Style.YELLOW("Saving not possible ")+str(e))

    sys_print("----------------------------Starting-----------------------")
    if os.path.exists(state_save_file):
        try:
            with open(state_save_file, "r") as f:
                state_save = eval(f.read())
            app.pretty_print_dict(state_save)
        except Exception as e:
            print(Style.RED("Loading Error ")+str(e))
    else:
        with open(state_save_file, 'a'):
            pass

    mem = isaa.get_context_memory()
    mem.init_store(app.id)

    if state_save:
        sys_print("Enter y or yes to resume")
        if 'y' not in input("Resume on Task ?").lower():
            state_save = {}

        else:
            sys_print(Style.GREY("Loading Agents..."))
            idea_enhancer(isaa, '', self_agent_config, chains, create_agent=True)
            startage_task_aproche(isaa, '', self_agent_config, chains, create_agent=True)
            generate_exi_dict(isaa, '', create_agent=True, tools=self_agent_config.tools, retrys=0)

    if "task" not in state_save.keys():
        sys_print("Enter the task")
        task = input(":")
        sys_print("Analysing Task")
        task = idea_enhancer(isaa, task, self_agent_config, chains, create_agent=True)
        sys_print(f"Final Task : {task}")
        save_state("task", str(task))
    else:
        sys_print(Style.GREY("Receiving task from sto.."))
        task = state_save["task"]

    if "approach" not in state_save.keys():
        sys_print("Generating approach to save the Task")
        approach = startage_task_aproche(isaa, task, self_agent_config, chains, create_agent=True)
        sys_print(f"Final approach : {approach}")
        save_state("approach", str(approach))
    else:
        sys_print(Style.GREY("Receiving approach from sto.."))
        approach = state_save["approach"]

    if "expyd" not in state_save.keys():
        sys_print(f"Generating Executable dict")  # test if on existing is fitting and alings with the approach
        expyd = generate_exi_dict(isaa, task + '\n' + approach, create_agent=True, tools=self_agent_config.tools, retrys=3)
        if isinstance(expyd, dict):
            sys_print(f"Dict Generation don")
            save_state("expyd", expyd['name'])
        else:
            sys_print(Style.YELLOW(f"Got an String as Response"))
            save_state("expyd", expyd)
    else:
        sys_print(Style.GREY("Receiving dict from sto.."))
        expyd = state_save["expyd"]
        if isinstance(expyd, str):
            print(expyd in chains.chains.keys(), expyd , chains.chains.keys())
            if expyd in chains.chains.keys():
                sys_print(Style.GREY(f"Got chain name : {expyd}"))
            else:
                sys_print(Style.CYAN(f"Try extracting dict"))
                if expyd.startswith('{') and expyd.endswith('}'):
                    expyd = eval(expyd)
                    sys_print(Style.GREEN(f"Extracted dict"))
                else:
                    sys_print(Style.GREY(f"sticking to sting instructions"))

    task_in_progress = True

    step = 0
    out = ''

    if "eval:out" in state_save.keys():
        out = state_save["eval:out"]

    self_agent_config.short_mem.clear_to_collective()

    while task_in_progress:
        sys_print(Style.GREY(f"------------- IN Processioning at step : {step} ------------------------------"))
        execution_agent.short_mem.clear_to_collective()
        execution_agent.get_messages(create=True)
        infos_c = f"List of Avalabel Agents : {isaa.config['agents-name-list']}\n Tools : {list(execution_agent.tools.keys())}\n" \
                  f" Functions : {isaa.scripts.get_scripts_list()}\n executable python dicts : {str(isaa.get_chain())} "
        execution_agent.add_message("user", task)
        execution_agent.add_message("system", infos_c)

        app.pretty_print_dict(expyd)

        user_input = input("0=Chat, 1=ausführen, 2=dict-Anpassen, 3=NeueStrategie, 4=Task-Anpassen exit\n:")

        if len(user_input) > 5:
            execution_agent.add_message("user", user_input[:-1])

        if user_input.endswith("0"):
            perfect = False
            data_execution_agent_0_run_ret = []

            sto = execution_agent.stop_sequence
            mode_sto = execution_agent.mode
            execution_agent.mode = 'free'
            execution_agent.stop_sequence = ["\n\n\n\n"]

            execution_agent.add_message("assistant", "Take action in the real world!")
            while not perfect:

                execution_agent_0_run_ret = isaa.run_agent(execution_agent, '')

                data_execution_agent_0_run_ret.append(execution_agent_0_run_ret)

                execution_agent.add_message("assistant", execution_agent_0_run_ret)

                u = input("enter don to leve:")

                if u == 'x':
                    task_in_progress = False
                    perfect = True
                elif u == 'don':
                    perfect = True
                else:
                    execution_agent.add_message("user", u)

            execution_agent.stop_sequence = sto
            execution_agent.mode = mode_sto

            if data_execution_agent_0_run_ret:

                out = isaa.run_agent(execution_agent, f"Crate a summary of Data to check :"
                                                      f"{data_execution_agent_0_run_ret}"
                                                      f"The task to be processed {task}"
                                                      f"The chosen approach {approach}"
                                                      f"Gives an evaluation and suggestions for improvement.")
                save_state(f"eval:out", out)
        if user_input.endswith("1"):
            if isinstance(expyd, dict) or expyd in chains.chains.keys():
                execution_agent.add_message("assistant",
                                         "Planning is complete and I will now begin executing the plan by taking "
                                         "action.")
                execution_agent.add_message("system", """
You can't tell what happened. everything that happened is in the text. give concrete information about the plan in order to fulfill the plan. if you don't know what to do next, you can make
1. look in your memories
2. another agent.
3. switch to the planning mode with the task to subdivide the current step until it matches your skills.
4. ask the user.

                """)
                ret, chain_infos = run_chain_in_cmd_auto_observation_que(isaa, task, chains, expyd, execution_agent)
                summary = isaa.summarize_ret_list(chain_infos)

                out = isaa.run_agent(execution_agent, f"Crate a summary of  Data to check :"
                                                        f"{ret} {summary}"
                                                        f"The task to be processed {task}"
                                                        f"Gives an evaluation and suggestions for improvement.")
                save_state(f"eval:out", out)
                save_state(f"chain_infos", chain_infos)
            else:
                free_run_in_cmd(isaa, task, execution_agent)
        if user_input.endswith("2"):
            expyd = generate_exi_dict(isaa, f"Optimise the dict : {expyd} based on this outcome : {out}"
                                            f" the approach {approach} and the task {task}\nOnly return the dict\nDict:", create_agent=False,
                                      tools=execution_agent.tools, retrys=3)

            save_state("expyd", expyd)
        if user_input.endswith("3"):
            approach = startage_task_aproche(isaa, f"Optimise the approach : {approach} based on this outcome"
                                                   f" : {out} and the task {task}",
                                             execution_agent, chains, create_agent=False)
            save_state("approach", approach)
        if user_input.endswith("4"):
            new_infos = input(":")
            task = idea_enhancer(isaa, f"Optimise the task now infos = {new_infos} old task"
                                       f" = {task} based on this outcome : {out}", execution_agent,
                                 chains, create_agent=False)
            save_state("task", task)
        if user_input == 'exit':
            task_in_progress = False

        step += 1



if __name__ == '__main__':
    import asyncio
    import websockets
    import json

    render_data = {
        "render": {
            "content": '<link rel="stylesheet" href="/app/test.css"><h1>Hello, World!</h1>',
            "place": "#content",
            "id": "content",
            "externals": ["test.css"],
            "placeholderContent": "<h1>Loading...</h1>"
        }
    }


    async def websocket_client():
        uri = "ws://localhost:5000/ws/app-live-test-DESKTOP-CI57V1L1"
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps({"message": "Hello from Python client!"}))
            print("FirstData")
            await websocket.send(json.dumps(render_data))
            print("SecodData")
            while True:
                response = await websocket.recv()
                if 'exit' in str(response):
                    await websocket.close()
                    break
                print("WebSocket message received:", response)
                await websocket.send(json.dumps({"message": input()}))


    asyncio.get_event_loop().run_until_complete(websocket_client())

