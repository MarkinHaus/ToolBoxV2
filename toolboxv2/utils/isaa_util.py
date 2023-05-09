import sys
import time

from transformers import pipeline

from toolboxv2 import App
from toolboxv2.mods.isaa import CollectiveMemory, AgentConfig
from toolboxv2.utils.toolbox import get_app
from toolboxv2.mods.isaa import Tools as Isaa

try:
    from toolboxv2.mods.isaa_audio import s30sek_mean, text_to_speech3, speech_stream, get_audio_transcribe

    SPEAK = True
except ImportError:
    SPEAK = False

from toolboxv2.utils.Style import print_to_console, Style, Spinner
from langchain.utilities import PythonREPL


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
    sys_print(f"\tObservationMemory  : {self_agent_config.obser_mem.tokens}\n")
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


def test_amplitude_for_talk_mode(sek=10):
    if not SPEAK:
        return -1
    print(f"Pleas stay silent for {sek}s")
    mean_0 = s30sek_mean(sek)
    return mean_0


def init_isaa(app, speak_mode=False, calendar=False, ide=False, create=False, isaa_print=False, python_test=False):
    if calendar:
        app.save_load("isaa_calendar")
        app.logger.info("Isaa audio is running")
        app.new_ac_mod("isaa_calendar")

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

    if ide:
        app.logger.info("Init IDE")
        app.save_load("isaa_ide")
        app.new_ac_mod("isaa_ide")
        app.logger.info("Isaa IDE is running")
        file_functions_ = app.AC_MOD.process_input

    app.logger.info("Init Isaa")
    app.save_load("isaa")
    app.logger.info("Isaa is running")

    sys.setrecursionlimit(1500)

    app.new_ac_mod('isaa')
    isaa: Isaa = app.AC_MOD
    isaa.loade_keys_from_env()

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
        Mode: The mode in which the agent is to be run. This is an optional key. available ar [free, tools, talk]
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

    if ide:
        file_functions_dis = """
function for file operation
syntax for function call : <function_name> <arguments>

"""

        def file_functions(x, from_="list"):
            try:
                if input(x + " - ACCEPT? :").lower() in ["y", "yes"]:
                    return file_functions_(from_ + ' ' + x.strip())
                return "Not authorised by user"
            except Exception as e:
                return "Error in file_functions : " + str(e)

        isaa.add_tool("create", lambda x: file_functions(x, from_='create'), "format for singel input functions ["
                                                                             "create] arguments is <path>",
                      file_functions_dis, self_agent_config)
        isaa.add_tool("delete", lambda x: file_functions(x, from_='delete'), "format for singel input functions ["
                                                                             "delete] arguments is <path>",
                      file_functions_dis, self_agent_config)
        isaa.add_tool("list", lambda x: file_functions(x, from_='list'), "format for singel input functions list] "
                                                                         "arguments is <path>", file_functions_dis,
                      self_agent_config)
        isaa.add_tool("read", lambda x: file_functions(x, from_='read'), "format for singel input functions [read] "
                                                                         "arguments is <path>", file_functions_dis,
                      self_agent_config)
        isaa.add_tool("move", lambda x: file_functions(x, from_='move'), "format for 2 input functions [move] "
                                                                         "arguments ar <source> <destination>",
                      file_functions_dis, self_agent_config)
        isaa.add_tool("insert_edit", lambda x: file_functions(x, from_='insert_edit'), "format for 2 input functions "
                                                                                       "[copy] arguments ar <source> "
                                                                                       "<destination>",
                      file_functions_dis, self_agent_config)
        isaa.add_tool("search", lambda x: file_functions(x, from_='search'), "format for 2 input functions [search] "
                                                                             "arguments ar <path> <keyword>",
                      file_functions_dis, self_agent_config)
        isaa.add_tool("copy", lambda x: file_functions(x, from_='copy'), "format for 2 input functions [copy] "
                                                                         "arguments ar <source> <destination>",
                      file_functions_dis, self_agent_config)

    if python_test:
        python_repl = PythonREPL()
        isaa.add_tool("eval-python-code", python_repl.run, "PythonREPL to eval python code", '', self_agent_config)
    if speak_mode:
        isaa.speak = speak

    def momory_wraper(x):
        momoey_ = CollectiveMemory().text(context=x)
        if momoey_ == "[]":
            return "No data found, Try entering other data related to your task"
        return momoey_

    mem = isaa.get_context_memory()

    def get_relevant_informations(x):
        ress = mem.get_context_for(x)

        last = []
        final = ''
        for res in ress:
            if last != res[1]:
                final += res[0].page_content + '\n\n'
            else:
                print("WARNING- same")

        task = f"Act as an summary expert your specialties are writing summary. you are known to think in small and " \
               f"detailed steps to get the right result. Your task : write a summary reladet to {x}\n\n{final}"
        res = isaa.run_agent(isaa.get_default_agent_config('think').set_model_name('gpt-3.5-turbo'), task)

        if res:
            return res

        return final

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

    return isaa, self_agent_config
