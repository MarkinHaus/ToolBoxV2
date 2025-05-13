import fnmatch
import json
import os
import re
import subprocess
import threading
import uuid

import requests
from bs4 import BeautifulSoup
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import (
    AIPluginTool,
    CopyFileTool,
    DeleteFileTool,
    ListDirectoryTool,
    MoveFileTool,
    ReadFileTool,
    ShellTool,
    WriteFileTool,
)
from tqdm import tqdm

from toolboxv2 import App, get_logger
from toolboxv2.mods import BROWSER
from toolboxv2.mods.isaa.subtools.web_loder import route_url_to_function
from toolboxv2.utils.toolbox import get_app

PIPLINE = None

try:
    from toolboxv2.mods.isaa_audio import (
        get_audio_transcribe,
        s30sek_mean,
        speech_stream,
        text_to_speech3,
    )

    SPEAK = True
except ImportError:
    SPEAK = False

try:
    import inquirer

    INQUIRER = True
except ImportError:
    INQUIRER = False

import networkx as nx

from toolboxv2.utils.extras.Style import Spinner, Style, print_to_console


def visualize_tree(tree, graph=None, parent_name=None, node_name=''):
    if graph is None:
        graph = nx.DiGraph()

    if 'start' in tree:
        if parent_name:
            graph.add_edge(parent_name, tree['start'])
        parent_name = tree['start']

    if 'tree' in tree:
        for sub_key in tree['tree']:
            visualize_tree(tree['tree'][sub_key], graph, parent_name, node_name + sub_key)

    return graph


def hydrate(params):
    def helper(name):
        return params[name]

    return helper


def speak(x, speak_text=SPEAK, vi=0, **kwargs):
    global PIPLINE
    if PIPLINE is None:
        from transformers import pipeline as PIPLINE
    if len(x) > 2401:
        print(f"text len to log : {len(x)}")
        return

    if len(x) > 1200:
        speak(x[:1200])
        x = x[1200:]

    cls_lang = PIPLINE("text-classification", model="papluca/xlm-roberta-base-language-detection")
    ln = cls_lang(x)

    if len(x) > 400:
        app: App = get_app()
        app.new_ac_mod("isaa")
        x = app.AC_MOD.mas_text_summaries(x, min_length=50)

    if ln[0]["label"] == 'de' and ln[0]["score"] > 0.2:
        text_to_speech3(x)

    elif ln[0]["label"] == 'en' and ln[0]["score"] > 0.5:
        speech_stream(x, voice_index=vi)
    else:
        sys_print(f"SPEEK SCORE TO LOW : {ln[0]['score']}")


def sys_print(x, **kwargs):
    print_to_console("SYSTEM:", Style.style_dic['BLUE'], x, max_typing_speed=0.04, min_typing_speed=0.08)


def run_agent_cmd(isaa, user_text, self_agent_config, step, spek):
    print("\nAGENT section\n")
    response = isaa.run_agent(self_agent_config, user_text)  ##code
    print("\nAGENT section END\n")

    task_done = isaa.test_task_done(response)

    sys_print(f"\n{'=' * 20}STEP:{step}{'=' * 20}\n")
    sys_print(f"\tMODE               : {self_agent_config.mode}\n")
    sys_print(f"\tObservationMemory  : {self_agent_config.observe_mem.tokens}\n")
    sys_print(f"\tShortTermMemory    : {self_agent_config.short_mem.tokens}\n\n")
    if "Answer: " in response:
        sys_print("AGENT: " + response.split('Answer:')[1] + "\n")
        spek(response.split('Answer:')[1])
    else:
        sys_print("AGENT: " + "\n".join(response.split(':')) + "\n")

    return response, task_done


def stop_helper(imp):
    if "Question:" in imp:
        return True
    return "User:" in imp


def split_todo_list(todo_string):
    # Regex-Muster, um verschiedene Variationen von Nummerierungen zu erkennen
    patterns = [
        r"^\d+[\.\)]",  # 1., 1), 2., 2), ...
        r"^\d+\)",  # 1), 2), 3), ...
        r"^\d+",  # 1, 2, 3, ...
        r"^[\d:]+\s*-\s*",  # 1: -, 2: -, 3: -, ...
        r"^\d+\s*-\s*",  # 1 -, 2 -, 3 -, ...
        r"^-\s*",  # - -, - -, - -, ...
    ]

    # Durchsuchen der Zeichenkette nach passenden Mustern und Aufteilen in To-Do-Elemente
    todo_list = []
    for pattern in patterns:
        todos = re.split(pattern, todo_string, flags=re.MULTILINE)[1:]  # Erste Position leeren
        if todos:
            todo_list.extend(todos)

    # Entfernen von Leerzeichen am Anfang und Ende der To-Do-Elemente
    todo_list = [todo.strip() for todo in todo_list]

    return todo_list


def extract_dict_from_string(string):
    start_index = string.find("{")
    end_index = string.rfind("}")
    if start_index != -1 and end_index != -1 and start_index < end_index:
        dict_str = string[start_index:end_index + 1]
        print("Found - dictionary :")
        try:
            dictionary = json.loads(dict_str)
            if isinstance(dictionary, dict):
                return dictionary
        except json.JSONDecodeError as e:
            print("Found - error :", e)
            return e
    return None


def test_amplitude_for_talk_mode(sek=10):
    if not SPEAK:
        return -1
    print(f"Pleas stay silent for {sek}s")
    mean_0 = s30sek_mean(sek)
    return mean_0


def get_code_files(git_project_dir, code_extensions: None or list = None):
    result = []
    if code_extensions is None:
        code_extensions = ['*.py', '*.js', '*.java', '*.c', '*.cpp', '*.css', '*.rb', '*.go', '*.php', '*.html',
                           '*.json']

    for root, _, files in os.walk(git_project_dir):
        for file in files:
            for ext in code_extensions:
                if fnmatch.fnmatch(file, ext):
                    result.append("/app/" + os.path.join(root, file).replace('isaa_work/', ''))
                    break

    return result


def download_github_project(repo_url, branch, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    command = f"git clone --branch {branch} {repo_url} {destination_folder}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error occurred while downloading the project: {stderr.decode('utf-8')}")
        return False

    print(f"Project downloaded successfully to {destination_folder}")
    return True


def validate_dictionary(dictionary, valid_agents, valid_tools, valid_functions, valid_chians):
    errors = []
    logger = get_logger()
    logger.info("testing")
    if not isinstance(dictionary, dict):
        logger.info(Style.RED("# type error"))
        errors.append("The provided object is not a dictionary.")
    elif "name" not in dictionary or "tasks" not in dictionary:
        logger.info(Style.RED("# no name &| tasks"))
        errors.append("The dictionary does not have the required keys 'name' and 'tasks'.")
    elif not isinstance(dictionary["name"], str):
        logger.info(Style.RED("# name not str"))
        errors.append("The value of the 'name' key must be a string.")
    elif not isinstance(dictionary["tasks"], list):
        logger.info(Style.RED("# tasks not list"))
        errors.append("The value of the 'tasks' key must be a list.")
    logger.info(Style.BLUE("Testing next tasks"))
    if "The dictionary does not have the required keys 'name' and 'tasks'." not in errors:
        i = 0
        for task in dictionary["tasks"]:
            i += 1
            if not isinstance(task, dict):
                logger.info(Style.RED("# task not dict"))
                errors.append(f"An entry in the tasks list is not a valid dictionary. in task : {i}")
                continue
            if "use" not in task or "name" not in task or "args" not in task or "return" not in task:
                errors.append(
                    f"A task entry is missing the required keys 'use', 'name', 'args', or 'return'. in task : {i}")
                continue
            use_type = task["use"]
            logger.info(Style.GREY(f"Task {i} is using : {use_type}"))
            if use_type == "agent":
                if task["name"] not in valid_agents:
                    logger.info(Style.RED(f"# wrong name {task['name']} valid ar {valid_agents}"))
                    errors.append(f"The agent name '{task['name']}' is not valid. in task : {i}")
            elif use_type == "tool":
                if task["name"] not in valid_tools:
                    logger.info(Style.RED(f"# wrong name {task['name']} valid ar {valid_tools}"))
                    errors.append(f"The tool name '{task['name']}' is not valid. in task : {i}")
            elif use_type == "function":
                if task["name"] not in valid_functions:
                    logger.info(Style.RED(f"# wrong name {task['name']} valid ar {valid_functions}"))
                    errors.append(f"The function name '{task['name']}' is not valid. in task :  {i}")
            elif use_type == "expyd" or use_type == "chain":
                if task["name"] not in valid_chians:
                    logger.info(Style.RED(f"# wrong name {task['name']} valid ar {valid_chians}"))
                    errors.append(f"The chain name '{task['name']}' is not valid. in task :  {i}")
            else:
                errors.append(
                    f"Invalid 'use' type '{use_type}' in a task. It should be 'agent', 'chain', 'tool', or 'function'. {i}")
            if not (isinstance(task["args"], str) or isinstance(task["args"], dict)):
                errors.append(f"The value of the 'args' key in a task must be a string. in task : {i}")
            if not isinstance(task["return"], str):
                errors.append(f"The value of the 'return' key in a task must be a string. in task : {i}")

    return errors


def generate_exi_dict(isaa, task, create_agent=False, tools=None, retrys=3):
    if tools is None:
        tools = []
    if create_agent:
        agent_config = isaa.get_agent_config_class("generate_exi_dict") \
            .set_completion_mode('chat') \
            .set_model_name('gpt-4-0613').set_mode('free')
        agent_config.stop_sequence = ['\n\n\n', "Execute:", "Observation:", "User:"]
        agent_config.stream = True
        if not task:
            return
    else:
        agent_config = isaa.get_agent_config_class("generate_exi_dict")

    if isinstance(tools, dict):
        tools = list(tools.keys())

    infos_c = f"List of Avalabel Agents : {isaa.config['agents-name-list']}\n Tools : {tools}\n" \
              f" Functions : {isaa.scripts.get_scripts_list()}\n executable python dicts : {str(isaa.get_chain())} "

    extracted_dict = {}

    agent_config.get_messages(create=True)
    agent_config.add_message("user", task)
    agent_config.add_message("system", infos_c)
    agent_config.add_message("assistant",
                             "I now coordinate the work of the various task handlers to ensure that the "
                             "right tasks are routed to the right handlers. This ensures that the tasks "
                             "are executed efficiently and effectively. For this I return a valid executable python "
                             "dict with the individual steps to be taken.")
    agent_config.add_message("system", """Stricht Syntax for an executable python dict :

        {
        "name": "<title of the task>",
        "tasks": [
        {
        "use": "<type of usage>", // tool, agent, expyd, or function
        "name": "<name of the tool, agent, or function>",
        "args": "<arguments>", // or including previous return value
        "return": "<return value>"

        // Optional
        "infos": "<additional-infos>"
        "short-mem": "<way-to-use-memory>" // 'summary' 'full' 'clear'
        "to-edit-text": True

        // Optional keys when dealing with large amounts of text
        "text-splitter": <maximum text size>,
        "chunk-run": "<operation on return value>"

        },
        // Additional tasks can be added here...

        ]
        }

        Example Tasks :

        { # I want to search for information's so i use the search agent, the information is stored in the $infos variable
               "use": "tool",
              "name": "search",
              "args": "search information on $user-input",
              "return":"$infos"
            },

        { # I want to call other task chain
             "use": "expyd",
              "name": "ai_assisted_task_processing",
              "args": "write_python : $requirements",
              "return":"$infos"
                },


        { # I want to do an action
          "use": "agent",
          "name": "execution",
          "args": "action-description $infos",
          "return": "$valur"
        }

        Examples Dictionary :

        {
            "name": "Generate_docs",
            "tasks": [
              {
                "use": "tool",
                "name": "read",
                "args": "$user-input",
                "return": "$file-content",
                "separators": "py",
                "text-splitter": 4000
              },
              {
                "use": "agent",
                "name": "thinkm",
                "args": "Act as a Programming expert your specialties are writing documentation. Your task : write an compleat documentation about '''\n $file-content \n'''",
                "return": "$docs",
                "chuck-run-all": "$file-content",
                "short-mem": "summary",
                "text-splitter": 16000
              },
              {
                "use": "agent",
                "name": "execution",
                "args": "Speichere die informationen in einer datei $docs",
                "return": "$file-name"
              }
            ]
        }

        {
            "name": "search_infos",
            "tasks": [
              {
                "use": "tool",
                "name": "search",
                "args": "Suche Information zu $user-input",
                "return": "$infos0"
              },
              {
                "use": "agent",
                "name": "think",
                "args": "Gebe einen Kompletten und diversen überblick der information zu Thema $infos0",
                "return": "$infos1"
              }
              {
                "use": "agent",
                "name": "execution",
                "args": "Speichere die informationen in einer datei infos0 $infos1",
                "return": "$file-name"
              }
            ]
        }
""")
    logger = get_logger()
    valid_dict = False
    coordination = "NO Data"
    for _ in range(retrys):
        logger.info(f"Retrying at {_}")
        coordination = isaa.run_agent(agent_config, f"Generate an executable python dict for "
                                                    f"{task}\n"
                                                    f" Brain storm deep and detailed about the conversion then start.")

        agent_config.add_message("assistant", coordination)
        extracted_dict = extract_dict_from_string(coordination)
        if isinstance(extracted_dict, dict):
            logger.info(Style.GREEN("Dictionary extracted"))
            logger.info(Style.GREY("Validate dictionary"))
            errors = validate_dictionary(extracted_dict, isaa.config['agents-name-list'], tools,
                                         list(isaa.scripts.scripts.keys()), list(isaa.get_chain().chains.keys()))
            print(f"Errors: {len(errors)} : {errors[:1]}")
            if errors:
                agent_config.add_message("system",
                                         "Errors : " + ', '.join(errors) + f" Fix them by using {infos_c} refer to your"
                                                                           f" last output oly change the error and return the full dict")
                if retrys == 2:
                    purpes = isaa.run_agent(agent_config, f"What is the purpes of the magent listed {errors}")
                    isaa.run_agent(agent_config, f"Crate the missing agent : {errors} {purpes}", mode_over_lode='tools')
            else:
                keys_d = list(extracted_dict.keys())
                if 'name' in keys_d and 'tasks' in keys_d:
                    print("Valid")
                    isaa.get_chain().add(extracted_dict['name'], extracted_dict["tasks"])
                    valid_dict = True
                    break
        if extracted_dict is not None:
            agent_config.add_message("system", 'Validation: The dictionary is not valid ' + str(extracted_dict))

    input(f"VALIDATION: {valid_dict=}")
    if valid_dict:
        isaa.get_chain().init_chain(extracted_dict['name'])
        return extracted_dict
    return coordination


def run_chain_in_cmd(isaa, task, chains, extracted_dict: str or dict, self_agent_config):
    response = ''
    task_done = False
    chain_ret = []
    chain_name = extracted_dict
    if isinstance(extracted_dict, dict):
        get_logger().info("Getting dict")
        chain_name = extracted_dict['name']
        dicht = chains.get(chain_name)
        if dicht != extracted_dict:
            get_logger().info("Getting not found start init")
            chains.add(chain_name, extracted_dict['tasks'])
            chains.init_chain(chain_name)
            get_logger().info(f"added {chain_name}")
        else:
            get_logger().info(f"{chain_name} is valid")

    while not task_done:
        # try:
        evaluation, chain_ret = isaa.execute_thought_chain(task, chains.get(chain_name), self_agent_config)
        # except Exception as e:
        #    print(e, '🔴')
        #    return "ERROR", chain_ret
        evaluation = evaluation[::-1][:300][::-1]
        pipe_res = isaa.text_classification(evaluation)
        print(chain_ret)
        print(pipe_res)
        if pipe_res[0]['label'] == "NEGATIVE":
            print('🟡')
            task_done = True
            if "y" in input("retry ? : "):
                task_done = False
            response = chain_ret[-1][1]
        else:
            print(pipe_res[0]['score'])
            print('🟢')
            task_done = True
            response = chain_ret[-1][1]

    return response, chain_ret


def run_chain_in_cmd_auto_observation_que(isaa, task, chains, extracted_dict: str or dict,
                                          self_agent_config):
    response = ''
    pressing = True

    def get_chain_name(extracted_dict_data):
        chain_name_ = extracted_dict_data
        if isinstance(extracted_dict_data, dict):
            get_logger().info("Getting dict")
            chain_name_ = extracted_dict_data['name']
            dicht = chains.get(chain_name_)
            if dicht != extracted_dict_data:
                get_logger().info("Getting not found start init")
                chains.add(chain_name_, extracted_dict_data['tasks'])
                chains.init_chain(chain_name_)
                get_logger().info(f"added {chain_name_}")
            else:
                get_logger().info(f"{chain_name_} is valid")
        return chain_name_

    chain_name = get_chain_name(extracted_dict)
    task_que = chains.get(chain_name)

    # user_text: str, agent_tasks, config: AgentConfig, speak = lambda x: x, start = 0,
    # end = None, chain_ret = None, chain_data = None, uesd_mem = None, chain_data_infos = False)
    uesd_mem = {}
    chain_data = {}
    chain_ret = []
    step = 0
    RETRYS = 4
    while pressing:

        if len(task_que) - step <= 0:
            pressing = False

        # do task get
        # evaluate, data...
        sys_print("---------------------- Start --------------------")
        pipe_res_label = "POSITIVE"
        try:
            chain_ret, chain_data, uesd_mem = isaa.execute_thought_chain(task, chains.get(chain_name),
                                                                         self_agent_config, start=step, end=step + 1,
                                                                         chain_ret=chain_ret, chain_data=chain_data,
                                                                         uesd_mem=uesd_mem, chain_data_infos=True)
            evaluation = chain_ret[-1][-1]
            # print(self_agent_config.last_prompt)
            evaluation_ = evaluation[::-1][:300][::-1]
            pipe_res = isaa.text_classification(evaluation_)
            pipe_res_label = pipe_res[0]['label']
            print(evaluation_)

        except Exception as e:
            sys_print(f"🔴 {e}")
            evaluation = e

        sys_print("---------------------- End execute_thought_chain step(s) --------------------")

        sys_print(f"Progress Main Chain at step : {step} from :{len(task_que)}")

        if pipe_res_label == "NEGATIVE":
            sys_print('🟡')
            if chain_ret:
                get_app().pretty_print_dict({"Last-task": chain_ret[-1]})
            print("Y -> to generate task adjustment\nR (text for infos)-> Retry on Task\nE -> return current state\n"
                  "lev black for next task")
            ui = input("optimise ? : ").lower()
            if ui == "y":
                data = generate_exi_dict(isaa,
                                         f"Optimise the task: {task_que[step]} based on this outcome : {chain_ret[-1]}"
                                         f" the evaluation {evaluation} and the task {task}\nOnly return the dict\nWitch The Corrent Task updated:",
                                         create_agent=False,
                                         tools=self_agent_config.tools, retrys=3)
                if isinstance(data, dict):
                    try:
                        task_que[step] = data['task'][0]
                        sys_print('🟡🟢')
                    except KeyError:
                        sys_print('🟡🔴')
                        step += 1
            elif ui == 'r':
                print("RETRY")
                sys_print('🟡🟡')
                if RETRYS == 0:
                    sys_print('🟡🟡🔴')
                    break
                RETRYS -= 1
            elif len(ui) > 3:
                self_agent_config.add_message("user", ui)
                sys_print('🟡🟢🟢')
            elif ui == 'e':
                chain_sum_data = isaa.summarize_ret_list(chain_ret)
                response = isaa.run_agent("think",
                                          f"Produce a summarization of what happened "
                                          f"(max 1 paragraph) using the given information {chain_sum_data}"
                                          f"and validate if the task was executed successfully")

                return response, chain_ret
            else:
                sys_print('🟢🟡')
                step += 1

        else:
            sys_print('🟢')
            step += 1

    chain_sum_data = isaa.summarize_ret_list(chain_ret)
    response = isaa.run_agent("think",
                              f"Produce a summarization of what happened "
                              f"(max 1 paragraph) using the given information {chain_sum_data}"
                              f"and validate if the task was executed successfully")

    return response, chain_ret


def free_run_in_cmd(isaa, task, self_agent_config):
    agents = isaa.config['agents-name-list']
    new_agent = isaa.config["agents-name-list"][-1]
    if len(agents) != 3:
        for agent_name in agents[3:]:
            isaa.get_agent_config_class(agent_name)
    free_run = True
    strp = 0
    self_agent_config.get_messages(create=True)
    self_agent_config.add_message("user", task)
    env_text = f"""Welcome, you are in a AI environment, your name is isaa.
    you have several basic skills 1. creating agents 2. creating some agents 3. using skills, agents and tools

    you have created {len(agents)}agents so far these are : {agents}.

    use your basic functions with the agent and skills to complete a task.

    for your further support you have a python environment at your disposal. write python code to access it.
    if you have no ather wy then to ask for help write Question: 'your question'\nUser:

    Task : {task}"""
    self_agent_config.add_message("system", env_text)
    data = []
    while free_run:

        sys_print("-------------------- Start Agent (free text mode) -----------------")
        sim = isaa.run_agent(self_agent_config, env_text, mode_over_lode='execution')
        sys_print("-------------------- End Agent -----------------")

        self_agent_config.add_message("assistant", sim)

        strp += 1

        sys_print(f"-------------------- in free exiqution ----------------- STEP : {strp}")

        if "user:" in sim.lower():
            sys_print("-------------------- USER QUESTION -----------------")
            self_agent_config.add_message("user", input("User: "))

        if new_agent != isaa.config["agents-name-list"][-1]:
            new_agent = isaa.config["agents-name-list"][-1]
            isaa.get_agent_config_class(new_agent).save_to_file()
        do_list = split_todo_list(sim)
        if do_list:
            self_agent_config.todo_list = do_list

        user_val = input("User (exit with n): ")

        data.append([sim, user_val])

        if user_val == "n":
            free_run = False

        self_agent_config.add_message("user", user_val)

    return data


def startage_task_aproche(isaa, task, self_agent_config, chains, create_agent=False):
    sto_agent_ = isaa.agent_collective_senses
    sto_summar = isaa.summarization_mode

    if create_agent:
        isaa.agent_collective_senses = True
        isaa.summarization_mode = 2
        isaa.get_chain().save_to_file()

        think_agent = isaa.get_agent_config_class("think") \
            .set_completion_mode('chat') \
            .set_model_name('gpt-4').set_mode('free')
        think_agent.stop_sequence = ['\n\n\n']
        think_agent.stream = True
        if not task:
            return
        # new env isaa withs chains
    else:
        think_agent = isaa.get_agent_config_class("think")

    agents = isaa.config["agents-name-list"]
    infos_c = f"List of Avalabel Agents : {agents}\n Tools : {list(self_agent_config.tools.keys())}\n" \
              f" Functions : {isaa.scripts.get_scripts_list()}\n Chains : {str(chains)} "

    think_agent.get_messages(create=True)
    think_agent.add_message("user", task)
    think_agent.add_message("system", infos_c)
    think_agent.add_message("system", "Process help Start by gathering relevant information. Then coordinate the next "
                                      "steps based on the information.When the task is simple enough, proceed with "
                                      "the execution. Then help yourself by creating an expert agent that can solve "
                                      "the task. Also use existing solution methods to solve the task more "
                                      "effectively.")
    think_agent.add_message("system", "Create 4 strategies (add a Describing name) "
                                      "with which you can solve this problem."
                                      "Specify the required agent tools and scripts in each strategie."
                                      " For each stratagem you should specify a success probability from 0% to 100%."
                                      "For each stratagem you should specify a deviation from the task"
                                      "from 100 to 0. -> 0 = no deviation is in perfect alignment to the task."
                                      " 100 = full deviation not related to the task ")
    think_agent.add_message("assistant", "After long and detailed step by step thinking and evaluating,"
                                         " brainstormen of the task I have created the following strategies."
                                         "Strategies :")

    strategies = isaa.run_agent(think_agent, 'Exec the Task as best as you can.')

    think_agent.add_message("assistant", strategies)

    think_agent.add_message("system", "Think about 3 further strategies with an lower Deviation then the best strategy."
                                      "Brainstorm new ideas and add old knowledge by extracted and or combined with "
                                      "new ideas."
                                      "Consider your stills,"
                                      " Reflect the successes "
                                      "as well as the deviation from the task at hand. Give both numbers.")

    perfact = False
    strategies_final = ""
    while not perfact:
        strategies_final = isaa.run_agent(think_agent, 'Exec the Task as best as you can.')
        think_agent.add_message("assistant", strategies_final)
        u = input(":")
        if u == 'x':
            exit(0)
        if u == 'y':
            think_agent.add_message("system", "Return an Elaborate of the effective strategie for the next agent"
                                              " consider what the user ask and the best variant.")
            strategies_final = isaa.run_agent(think_agent, 'Exec the Task as best as you can.')
            perfact = True
        think_agent.add_message("user", u)

    isaa.agent_collective_senses = sto_agent_
    isaa.summarization_mode = sto_summar

    return strategies_final


def idea_enhancer(isaa, task, self_agent_config, chains, create_agent=False):
    sto_agent_ = isaa.agent_collective_senses
    sto_summar = isaa.summarization_mode

    if create_agent:
        isaa.agent_collective_senses = True
        isaa.summarization_mode = 2
        isaa.get_chain().save_to_file()

        clarification_agent = isaa.get_agent_config_class("user_input_helper") \
            .set_completion_mode('chat') \
            .set_model_name('gpt-4').set_mode('free')
        clarification_agent.stop_sequence = ['\n\n\n']
        clarification_agent.stream = True
        if not task:
            return
        # new env isaa withs chains
    else:
        clarification_agent = isaa.get_agent_config_class("user_input_helper")

    # new env isaa withs chains
    agents = isaa.config["agents-name-list"]
    infos_c = f"List of Avalabel Agents : {agents}\n Tools : {list(self_agent_config.tools.keys())}\n" \
              f" Functions : {isaa.scripts.get_scripts_list()}\n Chains : {str(chains)} "
    clarification_agent.get_messages(create=True)
    clarification_agent.add_message("user", task)
    clarification_agent.add_message("system", infos_c)
    clarification_agent.add_message("system", "Reproduce the task four times in your own words and"
                                              " think about Possible Solution approaches ."
                                              " with which you can understand this problem better."
                                              " For each variant you should specify a Understanding from 0% to 100%."
                                              " For each variant you should specify a Complexity"
                                              "  approximate the numbers of step taken to compleat"
                                              "For each variant you should specify a deviation from the task probability "
                                              "from 100% to 0%. -> 0 = no deviation is in perfect alignment to the task."
                                              " 100 = full deviation not related to the task ")
    clarification_agent.add_message("assistant", "After long and detailed step by step thinking and evaluating,"
                                                 " brainstormen of the task I have created the following variant."
                                                 "variant :")
    perfact = False
    new_task = ""
    while not perfact:
        new_task = isaa.run_agent(clarification_agent, 'Exec the Task as best as you can.')
        clarification_agent.add_message("assistant", new_task)
        u = input(":")
        if u == 'x':
            exit(0)
        if u == 'y':
            clarification_agent.add_message("system", "Return an Elaborate task for the next agent"
                                                      " consider what the user ask and the best variant.")
            new_task = isaa.run_agent(clarification_agent, 'Exec the Task as best as you can.')
            perfact = True
        clarification_agent.add_message("user", u)

    isaa.agent_collective_senses = sto_agent_
    isaa.summarization_mode = sto_summar

    return str(new_task)


def add_skills(isaa, self_agent_config):
    shell_tool = ShellTool()
    read_file_tool = ReadFileTool()
    copy_file_tool = CopyFileTool()
    delete_file_tool = DeleteFileTool()
    move_file_tool = MoveFileTool()
    WriteFileTool()
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
        "ReadFileTool": read_file_tool,
        "CopyFileTool": copy_file_tool,
        "DeleteFileTool": delete_file_tool,
        "MoveFileTool": move_file_tool,
        "ListDirectoryTool": list_directory_tool,
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
    isaa.get_agent_config_class("execution")
    for tool in load_tools(["requests_all"]):
        isaa.lang_chain_tools_dict[tool.name] = tool
    isaa.add_lang_chain_tools_to_agent(self_agent_config, self_agent_config.tools)


def get_multiline_input(init_text="", line_starter=""):
    lines = []
    if init_text:
        print(init_text, end='')
    while True:
        line = input(line_starter)
        if line:
            lines.append(line)
        else:
            break
    return "\n".join(lines)


try:
    import inquirer

    INQUIRER = True
except ImportError:
    INQUIRER = False


def choiceList(all_chains, print_=print, input_=input, do_INQUIRER=True):
    all_chains += ['None']
    if INQUIRER and do_INQUIRER:

        questions = [
            inquirer.List('chain',
                          message="Choose a chain?",
                          choices=all_chains,
                          ),
        ]
        choice = inquirer.prompt(questions)['chain']

    else:
        choice = input_(f"{all_chains} select one (q) to quit:")
        while choice not in all_chains:
            if choice.lower() == 'q':
                return "None"
            print_("Invalid Chain name")
            choice = input_(f"{all_chains} select one (q) to quit:")
    return choice


def show_image_in_internet(images_url, browser=BROWSER):
    if isinstance(images_url, str):
        images_url = [images_url]
    for image_url in images_url:
        os.system(f'start {browser} {image_url}')


##########################################################################################################
def browse_website(url, question, summ):
    with Spinner(f"Processioning Web date for {url}"):
        summary = get_text_summary(url, question, summ)
    links = get_hyperlinks(url)

    # Limit links to 5
    if len(links) > 5:
        links = links[:5]

    result = f"""Website Content Summary: {summary}\n\nLinks: {links}"""

    return result


def get_text_summary(url, question, summarize):
    _, docs = route_url_to_function(url)
    summary = f"Scraping url: {url} with question {question[:250]}"
    l = docs()
    for d in tqdm(l, desc="Reding web page", total=len(l)):
        summary_ = summarize(f"Context ###{d.page_content}### Question ###{question}###")
        if isinstance(summary_, list):
            summary += '\n'.join(summary_)
        else:
            summary += summary_

    return """Result: """ + summary


def get_hyperlinks(url):
    link_list = scrape_links(url)
    return link_list


def scrape_text(url):
    response = requests.get(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"})

    # Check if the response contains an HTTP error
    if response.status_code >= 400:
        return "Error: HTTP " + str(response.status_code) + " error"

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


def extract_hyperlinks(soup):
    hyperlinks = []
    for link in soup.find_all('a', href=True):
        hyperlinks.append((link.text, link['href']))
    return hyperlinks


def format_hyperlinks(hyperlinks):
    formatted_links = []
    for link_text, link_url in hyperlinks:
        formatted_links.append(f"{link_text} ({link_url})")
    return formatted_links


def scrape_links(url):
    response = requests.get(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"})

    # Check if the response contains an HTTP error
    if response.status_code >= 400:
        return "error"

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    hyperlinks = extract_hyperlinks(soup)

    return format_hyperlinks(hyperlinks)

def adsadsadsadasd():

    def describe_all_chains(self):

        for chain_name in self.agent_chain.chains:
            if self.agent_chain.get_discr(chain_name):
                continue
            self.describe_chain(chain_name)

    def run_describe_chains(self, command):

        if len(command) == 2:
            self.describe_chain(command[1])

        else:
            self.describe_all_chains()

    def get_best_fitting(self, subject):

        all_description = ""

        for key in self.agent_chain.chains:
            if "Task Generator" in key or "Task-Generator" in key:
                continue
            des = self.agent_chain.get_discr(key)
            if des is None:
                des = key
            all_description += f"NAME:{key} \nUse case:{des}"

        mini_task0 = f"""Bitte durchsuchen Sie eine Liste von Aufgabenketten oder Lösungsansätzen und identifizieren
        Sie die beste Option für ein bestimmtes Thema oder Problem. Berücksichtigen Sie dabei die spezifischen
        Anforderungen und Ziele des Themas. Ihre Analyse sollte gründlich und detailliert sein, um die Stärken und
        Schwächen jeder Option zu beleuchten und zu begründen, warum die von Ihnen gewählte Option die beste ist.
        Stellen Sie sicher, dass Ihre Antwort klar, präzise und gut begründet ist.
        geben sie den Namen des Lösungsansatz mit an!
        Problem :
        {subject}
        Lösungsansätze:
        {all_description}
        """
        mini_task0_res = self.stream_read_llm(mini_task0, self.get_agent_config_class("thinkm"))

        mini_task1 = f""" "{mini_task0_res}"\n welcher Lösungsansatz wurde ausgewählt von diesen ausgewählt {list(self.agent_chain.chains.keys())}
        gebe nur den namen zurück
        wenn keiner der ansätze passt das gebe None zurück.
        name:"""

        mini_task1_res = self.mini_task_completion(mini_task1)

        chain_name = mini_task1_res
        for task_name in list(self.agent_chain.chains.keys()):
            if mini_task1_res.lower() in task_name.lower():
                chain_name = task_name

        self.print(f"Das system schlägt {chain_name} vor")
        self.print(f"mit der beründung : {mini_task0_res}")

        return chain_name, mini_task0_res

    def run_create_task_cli(self):
        self.print("Enter your text (empty line to finish):")
        task = get_multiline_input()
        name = self.create_task(task=task)
        self.print(f"New Task Crated name : {name}")

    def remove_chain_cli(self):
        all_chains = list(self.agent_chain.chains.keys())
        if not all_chains:
            return "No Cains Installed or loaded"

        chain_name = choiceList(all_chains, self.print)
        if chain_name == "None":
            return
        self.agent_chain.remove(chain_name)

    def run_chain_cli(self):
        all_chains = list(self.agent_chain.chains.keys())
        if not all_chains:
            return "No Cains Installed or loaded"

        chain_name = choiceList(all_chains, self.print)
        if chain_name == "None":
            return
        self.print("Enter your text (empty line to finish):")
        task = get_multiline_input()
        self.print(f"Starting Chin : {chain_name}")
        run_chain = self.agent_chain.get(chain_name)
        self.print(f"Chin len : {len(run_chain)}")
        if run_chain:
            res = self.execute_thought_chain(task, run_chain, self.get_agent_config_class("self"))
            self.print(f"Chain return \n{self.st_router.pretty_print(list(res))}")
        else:
            res = ["No chain found"]

        return res

    def run_auto_chain_cli(self):
        all_chains = list(self.agent_chain.chains.keys())
        if not all_chains:
            return "No Cains Installed or loaded"

        self.print("Enter your text (empty line to finish):")
        task = get_multiline_input()

        return self.run_auto_chain(task)

    def run_auto_chain(self, task):

        chain_name, begründung = self.get_best_fitting(task)

        if "y" not in input("Validate (y/n)"):
            return "Presses Stopped"

        self.print(f"Starting Chin : {chain_name}")
        return self.run_chain_on_name(chain_name, task)

    def run_chain_on_name(self, name, task):
        run_chain = self.agent_chain.get(name)
        self.print(f"Chin len : {len(run_chain)}")

        if run_chain:
            res = self.execute_thought_chain(task, run_chain, self.get_agent_config_class("self"))
            self.print(f"Chain return \n{self.st_router.pretty_print(list(res))}")
        else:
            res = "No chain found", []

        return res

    def init_cli(self):
        self.load_keys_from_env()
        if "augment" in self.config:
            self.init_from_augment(self.config['augment'],
                                   exclude=['task_list', 'task_list_done', 'step_between', 'pre_task', 'task_index'])
            self.print("Initialized from config augment")
            if 'tools' in self.config['augment']:
                if self.config['augment']['tools']:
                    return
        else:
            self.init_from_augment({'tools':
                                        {'lagChinTools': ['python_repl', 'requests_all', 'terminal', 'sleep',
                                                          'google-search',
                                                          'ddg-search', 'wikipedia', 'llm-math', 'requests_get',
                                                          'requests_post',
                                                          'requests_patch', 'requests_put', 'requests_delete', 'human'],
                                         'huggingTools': [],
                                         'Plugins': [], 'Custom': []}}, 'tools',
                                   exclude=['task_list', 'task_list_done', 'step_between', 'pre_task', 'task_index'])

    def create_task_cli(self):
        self.print("Enter your text (empty line to finish):")
        task = get_multiline_input()
        self.create_task(task)

    def optimise_task_cli(self):
        all_chains = list(self.agent_chain.chains.keys())
        chain_name = choiceList(all_chains, self.print)
        if chain_name == "None":
            return
        new_chain = self.optimise_task(chain_name)
        self.print(new_chain)

    def describe_chain(self, name):
        run_chain = self.agent_chain.get(name)
        if not len(run_chain):
            return "invalid Chain Namen"

        task = (f"Bitte analysieren und interpretieren Sie das gegebene JSON-Objekt, das eine Aufgabenkette "
                "repräsentiert. Identifizieren Sie das übergeordnete Ziel, die Anwendungsfälle und die Strategie, "
                "die durch diese Aufgabenkette dargestellt werden. Stellen Sie sicher,"
                " dass Ihre Analyse detailliert und präzise ist. Ihre Antwort sollte klar und präzise sein,"
                " um ein vollständiges Verständnis der Aufgabenkette und ihrer "
                "möglichen Einschränkungen zu ermöglichen. Deine Antwort soll kurtz und pregnant sein Maximal 2 sätze"
                f"zu analysierende Aufgabenkette: {run_chain}")

        discription = self.stream_read_llm(task, self.get_agent_config_class("think"))

        if len(discription) > 1000:
            discription = self.mas_text_summaries(discription, min_length=1000)

        self.print(f"Infos : {discription}")
        self.agent_chain.add_discr(name, discription)
        return discription


    def generate_task(self, subject, variables=None, context=None):

        if context is None:
            context = []
        if variables is None:
            variables = []

        self_agent = self.get_agent_config_class('self')

        f"""
Handle als Entscheidungsagenten du sollst, basierend auf einer Auswahl an Aufgaben und dem Kontext entscheiden, ob und welche Aufgabe für das Subjekt X angewendet werden soll. Wenn keine Aufgabe eine Erfolgswahrscheinlichkeit von über 80% für die beste Aufgabe aufweist, soll der Agent angeben, dass keine Aufgabe das Ziel erreicht, und das System wird eine passende Aufgabe erstellen.
Befehl: Entscheide, welche Aufgabe für {subject} basierend auf dem Kontext {context} {variables} angewendet werden soll. Wenn keine Aufgabe eine Erfolgswahrscheinlichkeit von über 80% für die beste Aufgabe aufweist, gib an, dass keine Aufgabe das Ziel erreicht, und erstelle eine passende Aufgabe.
Verfügbare aufgaben : {str(self.agent_chain)}
Aufgaben Name oder None:"""

        # task_name = self.mini_task_completion(agent_context_de)
        # task_name_l = task_name.lower()
        # if not (task_name_l != "None".lower() or len(task_name) > 1):
        #    self.init_config_var_initialise('chains-keys', [l.lower() for l in self.agent_chain.chains.keys()])
        #    if task_name_l in self.config['chains-keys']:
        #        return task_name  # Agent selected a valid task
        #
        # self.print_stream(f"Agent Evaluation: System cant detect valid task : {task_name}")
        self.print_stream("Pleas Open The Task editor or the isaa task creator")
        tools, names = self_agent.generate_tools_and_names_compact()
        ta_c = self.mini_task_completion(
            f"Handle als Entscheidungsagenten Überlege, wie komplex die Aufgabe ist und welche Fähigkeiten dafür "
            f"benötigt werden. Es gibt verschiedene Tools, Die du zu auswahl hast"
            f", wie zum Beispiel ein Text2Text Taschenrechner. Wähle zwischen einem "
            f"Tool oder einem Agenten für diese Aufgabe. Die verfügbaren Tools sind "
            f"{names}. Hier sind ihre Beschreibungen: {tools}. Es stehen auch folgende "
            f"Agenten zur Verfügung: {self.config['agents-name-list']}. Wenn weder ein "
            f"Agent noch ein Tool zum Thema '{subject}' passen, hast du noch eine weitere Option: "
            f"Gib 'Create-Agent' ein. Bitte beachte den Kontext: {context} {variables}. "
            f"Was ist dein gewählter Tool- oder Agentenname oder möchtest du einen "
            f"Agenten erstellen?"
            f"Ausgabe:")

        if not ta_c:
            ta_c = 'crate-task'

        self.print(ta_c)

        return {'isaa': ta_c}  ## TODO test

    def start_widget(self, command, app):

        uid, err = self.get_uid(command, app)

        if err:
            return "Invalid Token"

        self.logger.debug("Instance get_user_instance")

        user_instance = self.get_user_instance(uid, app)

        self.logger.debug("Instace Recived")

        sender, receiver = self.st_router.run_any("WebSocketManager", "srqw",
                                                  ["ws://localhost:5000/ws", user_instance["webSocketID"]])

        widget_id = str(uuid.uuid4())[25:]

        def print_ws(x):
            sender.put(json.dumps({"Isaa": x}))

        self.print_stream = print_ws

        group_name = user_instance["webSocketID"] + "-IsaaSWidget"
        collection_name = user_instance["webSocketID"] + '-' + widget_id + "-IsaaSWidget"

        self.st_router.run_any("MinimalHtml", "add_group", [group_name])

        widget_data = {'name': collection_name, 'group': [
            {'name': 'nav', 'file_path': './app/1/simpchat/simpchat.html',
             'kwargs': {'chatID': widget_id}}]}

        self.st_router.run_any("MinimalHtml", "add_collection_to_group", [group_name, widget_data])

        isaa_widget_html_element = self.st_router.run_any("MinimalHtml", "generate_html", [group_name, collection_name])

        print(isaa_widget_html_element)

        # Initialize the widget ui
        ui_html_content = self.st_router.run_any("WebSocketManager", "construct_render",
                                                 command=isaa_widget_html_element[0]['html_element'],
                                                 element_id="widgetChat",
                                                 externals=["/app/1/simpchat/simpchat.js"])

        # Initial the widget backend
        # on receiver { task: '', IChain': {
        #             "args": "Present the final report $final_report",
        #             "name": "execution",
        #             "return": "$presentation",
        #             "use": "agent"
        #         } }

        def runner():

            uesd_mem = {}
            chain_data = {}
            chain_ret = []

            running = True
            while running:
                while not receiver.empty():
                    data = receiver.get()

                    if 'exit' in data:
                        running = False
                    self.logger.info(f'Received Data {data}')

                    # if 'widgetID' not in data.keys():
                    #    continue
                    #
                    # self.logger.info(f'widgetID found in Data keys Valid:{data["widgetID"] != widget_id}')
                    #
                    # if data['widgetID'] != widget_id:
                    #    continue

                    try:
                        if "type" in data:
                            if 'id' not in data:
                                continue
                            # if data['id'] != widget_id:
                            #    continue
                            if data["type"] == "textWidgetData":
                                chain_data[data["context"]] = data["text"]
                                sender.put({"ChairData": True, "data": {'res': f"Text in {data['context']}"}})
                        elif 'task' in data and 'IChain' in data:
                            chain_ret, chain_data, uesd_mem = self.execute_thought_chain(data['task'], [data["IChain"]],
                                                                                         chain_ret=chain_ret,
                                                                                         chain_data=chain_data,
                                                                                         uesd_mem=uesd_mem,
                                                                                         chain_data_infos=True,
                                                                                         config=self.get_agent_config_class(
                                                                                             "self"))

                            sender.put({"ChairData": True, "data": {'res': chain_ret[-1][-1]}})
                        elif 'subject' in data:
                            context = self.get_memory().query(data['subject'])
                            res = self.generate_task(data['subject'], str(chain_data), context)
                            sender.put({"ChairData": True, "data": {'res': res}})

                    except Exception as e:
                        sender.put({'error': "Error e", 'res': str(e)})
                        sender.put('exit')
            sender.put('exit')

        widget_runner = threading.Thread(target=runner, daemon=True)
        widget_runner.start()

        self.print(ui_html_content)

        return ui_html_content
