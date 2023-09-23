import json
import os
import sys
import threading
import fnmatch
import subprocess
import re
from pathlib import Path

from langchain.agents import load_tools
from langchain.tools import ShellTool
from transformers import pipeline
from langchain.tools.file_management import (
    ReadFileTool,
    CopyFileTool,
    DeleteFileTool,
    MoveFileTool,
    ListDirectoryTool,
)

from toolboxv2 import App, get_logger
from toolboxv2.utils.toolbox import get_app
from toolboxv2.mods.isaa.isaa import AgentConfig, Tools as Isaa, AgentChain

try:
    from toolboxv2.mods.isaa_audio import s30sek_mean, text_to_speech3, speech_stream, get_audio_transcribe

    SPEAK = True
except ImportError:
    SPEAK = False

from toolboxv2.utils.Style import print_to_console, Style
from langchain.utilities import PythonREPL

import networkx as nx


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
    if len(x) > 2401:
        print(f"text len to log : {len(x)}")
        return

    if len(x) > 1200:
        speak(x[:1200])
        x = x[1200:]

    cls_lang = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    ln = cls_lang(x)

    if len(x) > 400:
        app: App = get_app()
        app.new_ac_mod("isaa")
        x = app.AC_MOD.mas_text_summaries(x, min_length=50)

    if 'de' == ln[0]["label"] and ln[0]["score"] > 0.2:
        text_to_speech3(x)

    elif 'en' == ln[0]["label"] and ln[0]["score"] > 0.5:
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
    if "User:" in imp:
        return True

    return False


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
                    result.append("/app/"+os.path.join(root, file).replace('isaa_work/', ''))
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
        get_logger().info(f"Getting dict")
        chain_name = extracted_dict['name']
        dicht = chains.get(chain_name)
        if dicht != extracted_dict:
            get_logger().info(f"Getting not found start init")
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
            print(f'🟢')
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
            get_logger().info(f"Getting dict")
            chain_name_ = extracted_dict_data['name']
            dicht = chains.get(chain_name_)
            if dicht != extracted_dict_data:
                get_logger().info(f"Getting not found start init")
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
        sys_print(f"---------------------- Start --------------------")
        pipe_res_label = "POSITIVE"
        try:
            chain_ret, chain_data, uesd_mem = isaa.execute_thought_chain(task,chains.get(chain_name),
                 self_agent_config,start=step,end=step + 1,chain_ret=chain_ret, chain_data=chain_data,
                  uesd_mem=uesd_mem,chain_data_infos=True)
            evaluation = chain_ret[-1][-1]
            #print(self_agent_config.last_prompt)
            evaluation_ = evaluation[::-1][:300][::-1]
            pipe_res = isaa.text_classification(evaluation_)
            pipe_res_label = pipe_res[0]['label']
            print(evaluation_)

        except Exception as e:
            sys_print(f"🔴 {e}")
            evaluation = e

        sys_print(f"---------------------- End execute_thought_chain step(s) --------------------")

        sys_print(f"Progress Main Chain at step : {step} from :{len(task_que)}")

        if pipe_res_label == "NEGATIVE":
            sys_print('🟡')
            if chain_ret:
                get_app().pretty_print_dict({"Last-task":chain_ret[-1]})
            print("Y -> to generate task adjustment\nR (text for infos)-> Retry on Task\nE -> return current state\n"
                  "lev black for next task")
            ui = input("optimise ? : ").lower()
            if "y" == ui:
                data = generate_exi_dict(isaa,
                                         f"Optimise the task: {task_que[step]} based on this outcome : {chain_ret[-1]}"
                                         f" the evaluation {evaluation} and the task {task}\nOnly return the dict\nWitch The Corrent Task updated:",
                                         create_agent=False,
                                         tools=self_agent_config.tools, retrys=3)
                if isinstance(data, dict):
                    try:
                        task_que[step] = data['task'][0]
                        sys_print(f'🟡🟢')
                    except KeyError:
                        sys_print(f'🟡🔴')
                        step += 1
            elif 'r' == ui:
                print("RETRY")
                sys_print(f'🟡🟡')
                if RETRYS == 0:
                    sys_print(f'🟡🟡🔴')
                    break
                RETRYS -= 1
            elif len(ui) > 3:
                self_agent_config.add_message("user", ui)
                sys_print(f'🟡🟢🟢')
            elif ui == 'e':
                chain_sum_data = isaa.summarize_ret_list(chain_ret)
                response = isaa.run_agent("think",
                                          f"Produce a summarization of what happened "
                                          f"(max 1 paragraph) using the given information {chain_sum_data}"
                                          f"and validate if the task was executed successfully")

                return response, chain_ret
            else:
                sys_print(f'🟢🟡')
                step += 1

        else:
            sys_print(f'🟢')
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
    env_text = f"""Welcome, you are in a hostile environment, your name is isaa.
    you have several basic skills 1. creating agents 2. creating some agents 3. using skills, agents and tools

    you have created {len(agents)}agents so far these are : {agents}.

    use your basic functions with the agent and skills to complete a task.

    for your further support you have a python environment at your disposal. write python code to access it.
    if you have no ather wy then to ask for help write Question: 'your question'\nUser:

    Task : {task}"""
    self_agent_config.add_message("system", env_text)
    data = []
    while free_run:

        sys_print(f"-------------------- Start Agent (free text mode) -----------------")
        sim = isaa.run_agent(self_agent_config, env_text, mode_over_lode='execution')
        sys_print(f"-------------------- End Agent -----------------")

        self_agent_config.add_message("assistant", sim)


        strp += 1

        sys_print(f"-------------------- in free exiqution ----------------- STEP : {strp}")

        if "user:" in sim.lower():
            sys_print(f"-------------------- USER QUESTION -----------------")
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
        isaa.get_context_memory().load_all()
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

    strategies = isaa.run_agent(think_agent, 'Exquisite the Task as best as you can.')

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
        strategies_final = isaa.run_agent(think_agent, 'Exquisite the Task as best as you can.')
        think_agent.add_message("assistant", strategies_final)
        u = input(":")
        if u == 'x':
            exit(0)
        if u == 'y':
            think_agent.add_message("system", "Return an Elaborate of the effective strategie for the next agent"
                                              " consider what the user ask and the best variant.")
            strategies_final = isaa.run_agent(think_agent, 'Exquisite the Task as best as you can.')
            perfact = True
        think_agent.add_message("user", u)

    isaa.agent_collective_senses = sto_agent_
    isaa.summarization_mode = sto_summar

    return strategies_final


def idea_enhancer(isaa, task, self_agent_config, chains, create_agent=False):
    sto_agent_ = isaa.agent_collective_senses
    sto_summar = isaa.summarization_mode

    if create_agent:
        isaa.get_context_memory().load_all()
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
        new_task = isaa.run_agent(clarification_agent, 'Exquisite the Task as best as you can.')
        clarification_agent.add_message("assistant", new_task)
        u = input(":")
        if u == 'x':
            exit(0)
        if u == 'y':
            clarification_agent.add_message("system", "Return an Elaborate task for the next agent"
                                                      " consider what the user ask and the best variant.")
            new_task = isaa.run_agent(clarification_agent, 'Exquisite the Task as best as you can.')
            perfact = True
        clarification_agent.add_message("user", u)

    isaa.agent_collective_senses = sto_agent_
    isaa.summarization_mode = sto_summar

    return str(new_task)


def add_skills(isaa, self_agent_config):
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


def init_isaa(app, speak_mode=False, calendar=False, ide=False, create=False,
              isaa_print=False, python_test=False, init_mem=False, init_pipe=False, join_now=False,
              global_stream_override=False, chain_runner=False, agents_default=None):

    if agents_default is None:
        agents_default = ["tools", "think", "search", "todolist"]
    chain_h = {}

    if calendar:
        app.save_load("isaa_calendar")
        app.logger.info("Isaa audio is running")
        app.new_ac_mod("isaa_calendar")
        try:
            calender_run = app.AC_MOD.get_llm_tool("markinhausmanns@gmail.com")
        except Exception:
            if os.path.exists("token.pickle"):
                os.remove("token.pickle")
            calender_run = app.AC_MOD.get_llm_tool("markinhausmanns@gmail.com")
        append_calender_agent = app.AC_MOD.append_agent

    if speak_mode:
        # min_ = test_amplitude_for_talk_mode(sek=5)
        # print("Done Testing : " + str(min_)) ##chad
        min_ = 0
        # init setup

        app.logger.info("Init audio")
        app.save_load("isaa_audio")
        app.logger.info("Isaa audio is running")
        app.new_ac_mod("isaa_audio")

        # speech = app.AC_MOD.speech
        # app.AC_MOD.generate_cache_from_history()

    app.logger.info("Init Isaa")
    app.save_load("isaa")
    app.logger.info("Isaa is running")

    sys.setrecursionlimit(1500)

    app.new_ac_mod('isaa')
    isaa: Isaa = app.get_mod('isaa')
    isaa.load_keys_from_env()

    if global_stream_override:
        isaa.global_stream_override = True

    if init_pipe:
        qu_init_t = threading.Thread(target=isaa.init_all_pipes_default)
        qu_init_t.start()

    if init_mem:
        mem_init_t = threading.Thread(target=isaa.get_context_memory().load_all)
        mem_init_t.start()

    self_agent_config: AgentConfig = isaa.get_agent_config_class("self")

    if isaa_print:
        # def helper(x):
        #    print_to_console("ISAA", x, max_typing_speed=0.06, min_typing_speed=0.12)
        isaa.print_stream = lambda x: print_to_console("ISAA:", title_color=Style.style_dic['_MAGENTA'], content=x,
                                                       max_typing_speed=0.06, min_typing_speed=0.12)

    if calendar:
        calender_agent_config: AgentConfig = isaa.get_agent_config_class("Calender-Agent")

        def run_agent_think_wrapper(x):
            if not x:
                return "Provide input"
            return isaa.run_agent(calender_agent_config, x, mode_over_lode='talk')

        append_calender_agent(calender_agent_config, calender_run, run_agent_think_wrapper)

        def run_calender_agent(x):
            if not x:
                return "Provide input"
            return isaa.run_agent(calender_agent_config, x)

        isaa.add_tool("Calender", run_calender_agent,
                      "a tool to use the calender and mange user task todos und meetings", "Calender(<task>)",
                      self_agent_config)

    if ide:
        def extract_code(x):
            data = x.split('```')
            if len(data) == 3:
                text = data[1].split('\n')
                code_type = text[0]
                code = '\n'.join(text[1:])
                return code, code_type
            if len(data) > 3:
                print(x)
            return '', ''

        def save_file(name, text):

            if not os.path.exists("./data/isaa_data/work"):
                Path("./data/isaa_data/work").mkdir(parents=True, exist_ok=True)

            open('./data/isaa_data/work/' + name, 'a').close()
            with open('./data/isaa_data/work/' + name, 'w') as f:
                f.write(text)

        def helper(x):
            code, type_ = extract_code(x)

            if code:
                save_file("test." + type_, code)

        def modify_file(file_path, content):
            """Modify the file contents file name content"""
            save_file(file_path, content)

        chain_h['save_code'] = helper

        rft = ReadFileTool()
        cft = CopyFileTool()
        dft = DeleteFileTool()
        mft = MoveFileTool()
        lft = ListDirectoryTool()

        isaa.add_tool("Read", rft, f"Read({rft.args})", rft.description, self_agent_config, lagchaintool=True)
        isaa.add_tool("Copy", cft, f"Copy({cft.args})", cft.description, self_agent_config, lagchaintool=True)
        isaa.add_tool("Delete", dft, f"Delete({dft.args})", dft.description, self_agent_config, lagchaintool=True)
        isaa.add_tool("Move", mft, f"Move({mft.args})", mft.description, self_agent_config, lagchaintool=True)
        isaa.add_tool("Write-first-code-block-to-file", helper, f"Write-first-code-block-to-file(<code>)", "extract first code block"
                                                                                                           "and w"
                                                                                                           "rite code "
                                                                                                           "to file",
                      self_agent_config)
        isaa.add_tool("ListDirectory", lft, f"ListDirectory({lft.args})", lft.description, self_agent_config,
                      lagchaintool=True)
        isaa.add_tool("modify_file", modify_file, f"modify_file(file_path, content)", modify_file.__doc__, self_agent_config)

    if speak_mode:
        isaa.speak = speak

    for a_n in agents_default:
        isaa.get_agent_config_class(a_n)

    chains = isaa.get_chain(None, hydrate(chain_h))
    if chain_runner:
        chains.load_from_file()

    if join_now:
        if init_pipe:
            qu_init_t.join()
        if init_mem:
            mem_init_t.join()

    return isaa, self_agent_config, chains
