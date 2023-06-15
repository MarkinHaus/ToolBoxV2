"""Console script for toolboxv2. Isaa CMD Tool"""
import json
import os
import random
import time

from langchain.agents import load_tools

from toolboxv2 import Style, Spinner, get_logger, App
from toolboxv2.mods.isaa import AgentChain, IsaaQuestionBinaryTree, AgentConfig
from toolboxv2.utils.isaa_util import init_isaa, generate_exi_dict, \
    idea_enhancer, startage_task_aproche, free_run_in_cmd, get_code_files, \
    run_chain_in_cmd_auto_observation_que

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
    from langchain.tools.file_management import (
        ReadFileTool,
        CopyFileTool,
        DeleteFileTool,
        MoveFileTool,
        WriteFileTool,
        ListDirectoryTool,
    )
    from langchain.tools import AIPluginTool
    shell_tool = ShellTool()
    read_file_tool = ReadFileTool()
    copy_file_tool = CopyFileTool()
    delete_file_tool = DeleteFileTool()
    move_file_tool = MoveFileTool()
    write_file_tool = WriteFileTool()
    list_directory_tool = ListDirectoryTool()

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
        # "https://semgrep.dev/.well-known/ai-plugin.json",
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

    state_save = {}

    state_save_file = "StateSave.state"
    if os.path.exists(state_save_file):
        try:
            with open(state_save_file, "r") as f:
                state_save = eval(f.read())
            app.pretty_print_dict(state_save)
        except Exception:
            print(Style.RED("Loading Error"))

    def save_state(state_name, content):
        state_save[state_name] = content
        app.pretty_print_dict(state_save)
        try:
            with open(state_save_file, "w") as f1:
                f1.write(str(state_save))
        except Exception:
            print(Style.YELLOW("Saving not possible"))

    mem = isaa.get_context_memory()
    mem.init_store(app.id)

    if state_save:
        app.pretty_print_dict(state_save)
        if 'y' not in input("Resume on Task ?"):
            state_save = {}

        else:
            idea_enhancer(isaa, '', self_agent_config, chains, create_agent=True)
            startage_task_aproche(isaa, '', self_agent_config, chains, create_agent=True)
            generate_exi_dict(isaa, '', create_agent=True, tools=self_agent_config.tools, retrys=0)

    if "task" not in state_save.keys():
        task = input(":")
        task = idea_enhancer(isaa, task, self_agent_config, chains, create_agent=True)
        save_state("task", str(task))
    else:
        task = state_save["task"]

    if "approach" not in state_save.keys():
        approach = startage_task_aproche(isaa, task, self_agent_config, chains, create_agent=True)
        save_state("approach", str(approach))
    else:
        approach = state_save["approach"]

    if "expyd" not in state_save.keys():
        expyd = generate_exi_dict(isaa, task + '\n' + approach, create_agent=True, tools=self_agent_config.tools, retrys=3)
        if isinstance(expyd, dict):
            save_state("expyd", expyd['name'])
        else:
            save_state("expyd", expyd)
    else:
        expyd = state_save["expyd"]
        if isinstance(expyd, str):
            if expyd.startswith('{') and expyd.endswith('}'):
                expyd = eval(expyd)

    task_in_progress = True

    step = 0
    iteration = 0
    chain_infos = []
    out = ''

    if "eval:out" in state_save.keys():
        out = state_save["eval:out"]

    while task_in_progress:
        print("At step: ", step)
        app.pretty_print_dict(expyd)
        print("iteration: ", iteration)

        user_input = input("1=ausführen, 2=dict-Anpassen, 3=NeueStrategie, 4=Task-Anpassen exit\n:")

        if user_input == "1":
            if isinstance(expyd, dict) or expyd in chains.chains.keys():
                infos_c = f"List of Avalabel Agents : {isaa.config['agents-name-list']}\n Tools : {list(execution_agent.tools.keys())}\n" \
                          f" Functions : {isaa.scripts.get_scripts_list()}\n executable python dicts : {str(isaa.get_chain())} "
                execution_agent.get_messages(create=True)
                execution_agent.add_message("user", task)
                execution_agent.add_message("system", infos_c)
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

                out = isaa.run_agent(self_agent_config, f"ate a summary of  Data to check :"
                                                        f"{ret} {summary}"
                                                        f"The task to be processed {task}"
                                                        f"The chosen approach {approach}"
                                                        f"Gives an evaluation and suggestions for improvement.")
                save_state(f"eval:out", out)
            else:
                free_run_in_cmd(isaa, task, self_agent_config)
        if user_input == "2":
            expyd = generate_exi_dict(isaa, f"Optimise the dict : {expyd} based on this outcome : {out}"
                                            f" the approach {approach} and the task {task}\nOnly return the dict\nDict:", create_agent=False,
                                      tools=self_agent_config.tools, retrys=3)
            save_state("expyd", expyd)


        if user_input == "3":
            approach = startage_task_aproche(isaa, f"Optimise the approach : {approach} based on this outcome"
                                                   f" : {out} and the task {task}",
                                             self_agent_config, chains, create_agent=False)
            save_state("approach", approach)
        if user_input == "4":
            new_infos = input(":")
            task = idea_enhancer(isaa, f"Optimise the task now infos = {new_infos} old task"
                                       f" = {task} based on this outcome : {out}", self_agent_config,
                                 chains, create_agent=False)
            save_state("task", task)

        if user_input == 'exit':
            task_in_progress = False

        step += 1
