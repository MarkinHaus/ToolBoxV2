import json
import os
import sys
import threading
import fnmatch
import subprocess
import re

from langchain.tools import ShellTool
from transformers import pipeline
from langchain.tools.file_management import (
    ReadFileTool,
    CopyFileTool,
    DeleteFileTool,
    MoveFileTool,
    WriteFileTool,
    ListDirectoryTool,
)

from toolboxv2 import App
from toolboxv2.utils.toolbox import get_app
from toolboxv2.mods.isaa import CollectiveMemory, AgentConfig, Tools as Isaa, AgentChain

try:
    from toolboxv2.mods.isaa_audio import s30sek_mean, text_to_speech3, speech_stream, get_audio_transcribe

    SPEAK = True
except ImportError:
    SPEAK = False

from toolboxv2.utils.Style import print_to_console, Style, Spinner
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

    task_done = isaa.test_task_done(response, self_agent_config)

    sys_print(f"\n{'=' * 20}STEP:{step}{'=' * 20}\n")
    sys_print(f"\tMODE               : {self_agent_config.mode}\n")
    sys_print(f"\tCollectiveMemory   : {CollectiveMemory().token_in_use} | total vec num : "
              f"{CollectiveMemory().memory.get_stats()['total_vector_count']}\n")
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
        dict_str = string[start_index:end_index+1]
        try:
            dictionary = json.loads(dict_str)
            if isinstance(dictionary, dict):
                return dictionary
        except json.JSONDecodeError as e:
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
        code_extensions = ['*.py', '*.js', '*.java', '*.c', '*.cpp', '*.cs', '*.rb', '*.go', '*.php']

    for root, _, files in os.walk(git_project_dir):
        for file in files:
            for ext in code_extensions:
                if fnmatch.fnmatch(file, ext):
                    result.append(os.path.join(root, file).replace('isaa_work/', ''))
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


def init_isaa(app, speak_mode=False, calendar=False, ide=False, create=False,
              isaa_print=False, python_test=False, init_mem=False, init_pipe=False, join_now=False,
              global_stream_override=False, chain_runner=False):
    chain_h = {}

    if calendar:
        app.save_load("isaa_calendar")
        app.logger.info("Isaa audio is running")
        app.new_ac_mod("isaa_calendar")
        try:
            calender_run = app.AC_MOD.get_llm_tool("markinhausmanns@gmail.com")
        except Exception:
            os.remove("token.pickle")
            calender_run = app.AC_MOD.get_llm_tool("markinhausmanns@gmail.com")
        append_calender_agent = app.AC_MOD.append_agent

    if speak_mode:
        # min_ = test_amplitude_for_talk_mode(sek=5)
        # print("Done Testing : " + str(min_)) ##chad
        min_ = 0
        app.logger.info("Init calendar")
        # init setup

        app.logger.info("Init audio")
        app.save_load("isaa_audio")
        app.logger.info("Isaa audio is running")
        app.new_ac_mod("isaa_audio")

        # speech = app.AC_MOD.speech
        app.AC_MOD.generate_cache_from_history()

    app.logger.info("Init Isaa")
    app.save_load("isaa")
    app.logger.info("Isaa is running")

    sys.setrecursionlimit(1500)

    app.new_ac_mod('isaa')
    isaa: Isaa = app.AC_MOD
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

    if create:
        def text_to_dict(text: str) -> dict:
            data_dict = {
                'Personal': None,
                'Goals': None,
                'Task': None,
                'Name': None,
                'Capabilities': None,
                'Mode': None,
            }
            text = text.split('\n')
            for line_ in text:
                line_ = line_.split(',')
                for line in line_:
                    key_value = line.strip().split(':')
                    if len(key_value) >= 2:
                        key = key_value[-2].strip()
                        value = key_value[-1].strip()
                        if key in 'Name' or key.endswith("Name") or key.startswith("Name"):
                            data_dict['Name'] = value
                        elif key == 'Mode':
                            data_dict['Mode'] = value
                        elif key == 'Task':
                            data_dict['Task'] = value
                        elif key == 'Personal':
                            data_dict['Personal'] = value
                        elif key == 'Goals':
                            data_dict['Goals'] = value
                        elif key == 'Capabilities':
                            data_dict['Capabilities'] = value
            return data_dict

        def create_agent(x: str) -> str:

            """
    The create_agent function takes a single string argument x, which is expected to contain a set of key-value pairs separated by colons (:). These pairs specify various attributes of an agent that is to be created and run.

    The function parses the input string x and extracts the values associated with the following keys:

        Name: The name of the agent to be created. This key is required and must be present in the input string.
        Mode: The mode in which the agent is to be run. This is an optional key. available ar [free, execution]
        Task: The task that the agent is to perform. This is an optional key.
        Personal: The personality of the agent. This is an optional key.
        Goals: The goals of the agent. This is an optional key.
        Capabilities: The capabilities of the agent. This is an optional key.

    The function then creates an AgentConfig object with the specified name and sets its personality, goals, and capabilities attributes to the values associated with the corresponding keys, if those keys were present in the input string."""
            # personal: Optional[str] = None
            # goals: Optional[str] = None
            # task: Optional[str] = None
            # name: Optional[str] = None
            # capabilities: Optional[str] = None
            # mode: Optional[str] = None
            data = text_to_dict(x)

            if not data['Name']:
                return "ValueError('Agent name must be specified.')"

            agent_config: AgentConfig = isaa.get_agent_config_class(data['Name'])
            agent_config.tools = self_agent_config.tools

            if data['Personal'] is not None:
                agent_config.personality = data['Personal']

            if data['Goals'] is not None:
                agent_config.goals = data['Goals']

            if data['Capabilities'] is not None:
                agent_config.capabilities = data['Capabilities']

            return isaa.run_agent(agent_config, data['Task'], mode_over_lode=data['Mode'])

        isaa.add_tool("spawn_agent", create_agent,
                      "The create_agent function takes a single string argument x, which is expected to contain a set of key-value pairs separated by colons (:). These pairs specify various attributes of an agent that is to be created and run. use agent to divde taskes"
                      , """The function parses the input string x and extracts the values associated with the following keys:

               Name: The name of the agent to be created. This key is required and must be present in the input string.
               Mode: The mode in which the agent is to be run. This is an optional key. available ar [free, tools, talk]
               Task: The task that the agent is to perform. This is an optional key.
               Personal: The personality of the agent. This is an optional key.
               Goals: The goals of the agent. This is an optional key.
               Capabilities: The capabilities of the agent. This is an optional key.

           The function then creates an AgentConfig object with the specified name and sets its personality, goals, and capabilities attributes to the values associated with the corresponding keys, if those keys were present in the input string.""",
                      self_agent_config)
        isaa.add_tool("talk_to_agent", create_agent,
                      "The talk_to_agent function takes a single string argument x, which is expected to contain a set of key-value pairs separated by colons (:). These pairs specify various attributes of an agent that is to be created and run. use agent to divde taskes"
                      , """The function parses the input string x and extracts the values associated with the following keys:

               Name: The name of the agent to be created. This key is required and must be present in the input string.
               Mode: The mode in which the agent is to be run. This is an optional key. available ar [free, tools, talk]
               Task: The task that the agent is to perform. This is an optional key.

           The function then runs the Agent with the specified name.""",
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
            open('./data/isaa_data/work/' + name, 'a').close()
            with open('./data/isaa_data/work/' + name, 'w') as f:
                f.write(text)

        def helper(x):
            code, type_ = extract_code(x)

            if code:
                save_file("test." + type_, code)

        chain_h['save_code'] = helper

        rft = ReadFileTool()
        cft = CopyFileTool()
        dft = DeleteFileTool()
        mft = MoveFileTool()
        wft = WriteFileTool()
        lft = ListDirectoryTool()

        isaa.add_tool("Read", rft, f"Read({rft.args})", rft.description, self_agent_config, lagchaintool=True)
        isaa.add_tool("Copy", cft, f"Copy({cft.args})", cft.description, self_agent_config, lagchaintool=True)
        isaa.add_tool("Delete", dft, f"Delete({dft.args})", dft.description, self_agent_config, lagchaintool=True)
        isaa.add_tool("Move", mft, f"Move({mft.args})", mft.description, self_agent_config, lagchaintool=True)
        isaa.add_tool("Write", wft, f"Write({wft.args})", wft.description, self_agent_config, lagchaintool=True)
        isaa.add_tool("ListDirectory", lft, f"ListDirectory({lft.args})", lft.description, self_agent_config,
                      lagchaintool=True)

    if chain_runner:
        chain_instance: AgentChain = isaa.get_chain()
        agent_categorize_config: AgentConfig = isaa.get_agent_config_class("categorize")
        agent_categorize_config \
            .set_mode('free') \
            .set_completion_mode('text') \
            .stream = False

        def chain_helper(x):
            chain_instance.load_from_file()

            res = isaa.run_agent(agent_categorize_config, f"What chain '{str(chain_instance)}'"
                                                          f" \nis fitting for this input '{x}'\n"
                                                          f"Only return the correct name or None\nName: ")
            if res.lower() == 'none':
                res = "I cant find a fitting chain"

            infos = '\n'.join([f'{item[0]} ID: {item[1]}' for item in list(zip(chain_instance.chains.keys(),
                                                                               range(
                                                                                   len(chain_instance.chains.keys()))))])
            user_vlidation = input(f"Isaa whats tu use : '{res}'\n"
                                   f"if its the chain is wrong type the corresponding number {infos}\n"
                                   f"other wise live black\nInput: ")
            do_task = False
            chain = ''
            evaluation = 'evaluation error'
            if user_vlidation in ['y', '']:
                do_task = True
                chain = res
            elif user_vlidation in [str(z) for z in range(len(chain_instance.chains.keys()))]:
                user_vlidation = int(user_vlidation)
                chain = list(chain_instance.chains.keys())[user_vlidation]
                do_task = True
            else:
                do_task = False
                evaluation = "Invalid user input"

            if do_task:
                evaluation, chain_ret = isaa.execute_thought_chain(
                    x, chain_instance.get(chain), self_agent_config)

            return evaluation

        isaa.add_tool("execute-chain", chain_helper, "input and details for the chain", '', self_agent_config)

    if python_test:
        st = ShellTool()
        python_repl = PythonREPL()
        isaa.add_tool("Shell", st, f"Read({st.args})", st.description, self_agent_config, lagchaintool=True)
        isaa.add_tool("eval-python-code", python_repl.run, "PythonREPL to eval python code", '', self_agent_config)
    if speak_mode:
        isaa.speak = speak

    mem = isaa.get_context_memory()

    def get_relevant_informations(x):
        ress = mem.get_context_for(x)

        task = f"Act as an summary expert your specialties are writing summary. you are known to think in small and " \
               f"detailed steps to get the right result. Your task : write a summary reladet to {x}\n\n{ress}"
        res = isaa.run_agent(isaa.get_default_agent_config('think').set_model_name('gpt-3.5-turbo'), task)

        if res:
            return res

        return ress

    def ad_data(x):
        mem.add_data('main', x)

        return 'added to memory'

    isaa.add_tool("memory", get_relevant_informations, "a tool to get similar information from your memories."
                                                       " useful to get similar data. ", "memory(<related_information>)",
                  self_agent_config)

    isaa.add_tool("save_data_to_memory", ad_data, "tool to save data to memory,"
                                                  " write the data as specific"
                                                  " and accurate as possible.",
                  "save_data_to_memory(<store_information>)",
                  self_agent_config)

    chains = isaa.get_chain(None, hydrate(chain_h))
    if chain_runner:
        chains.load_from_file()

    if join_now:
        if init_pipe:
            qu_init_t.join()
        if init_mem:
            mem_init_t.join()

    return isaa, self_agent_config, chains
