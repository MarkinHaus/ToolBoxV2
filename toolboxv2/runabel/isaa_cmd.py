"""Console script for toolboxv2. Isaa CMD Tool"""
from pebble import concurrent
import random
import sys
import time
from typing import Optional

from transformers import pipeline

from toolboxv2 import Style
from toolboxv2.mods.isaa_audio import get_audio_transcribe, text_to_speech3, s30sek_mean
from toolboxv2.util.isaa_util import print_to_console
from toolboxv2.util.toolbox import App
from toolboxv2.mods.isaa import AgentConfig, CollectiveMemory
from toolboxv2.mods.isaa import Tools as Isaa


def print_(x, **kwargs):
    print_to_console("SYSTEM:", Style.style_dic['BLUE'], x, max_typing_speed=0.04, min_typing_speed=0.08)


def run_agent_cmd(isaa, user_text, self_agent_config, step, spek):
    print("\nAGENT section\n")
    response = isaa.run_agent("self", user_text)
    print("\nAGENT section END\n")

    task_done = isaa.test_task_done(response, self_agent_config)

    print_(f"\n{'=' * 20}STEP:{step}{'=' * 20}\n")
    print_(f"\tMODE               : {self_agent_config.mode}\n")
    print_(f"\tCollectiveMemory   : {CollectiveMemory().token_in_use} | total vec num : "
           f"{CollectiveMemory().memory.get_stats()['total_vector_count']}\n")
    print_(f"\tObservationMemory  : {self_agent_config.obser_mem.tokens}\n")
    print_(f"\tShortTermMemory    : {self_agent_config.short_mem.tokens}\n\n")
    if "Answer: " in response:
        print_("AGENT: " + response.split('Answer:')[1] + "\n")
        spek(response.split('Answer:')[1])
    else:
        print_("AGENT: " + "\n".join(response.split(':')) + "\n")

    return response, task_done


def test_amplitude_for_talk_mode(sek=10):
    print(f"Pleas stay silent for {sek}s")
    mean_0 = s30sek_mean(sek)

    return mean_0  # , mean_1


def stop_helper(imp):
    if "Question:" in imp:
        return True
    if "User:" in imp:
        return True

    return False


def run_isaa_verb(app: App, speek_mode=True, task=""):
    min_ = 105
    speech_stream = lambda x, **kwargs: x
    ide_mode = False

    app.save_load("isaa_calendar")
    app.logger.info("Isaa audio is running")
    app.new_ac_mod("isaa_calendar")

    calender_run = app.AC_MOD.get_llm_tool("markinhausmanns@gmail.com")
    append_calender_agent = app.AC_MOD.append_agent

    if speek_mode:
        min_ = test_amplitude_for_talk_mode(sek=5)
        print("Done Testing : " + str(min_))
        app.logger.info("Init calendar")
        # init setup

        app.logger.info("Init audio")
        app.save_load("isaa_audio")
        app.logger.info("Isaa audio is running")
        app.new_ac_mod("isaa_audio")

        # speech = app.AC_MOD.speech
        speech_stream = app.AC_MOD.speech_stream
        app.AC_MOD.generate_cache_from_history()

    try:
        app.logger.info("Init IDE")
        app.save_load("isaa_ide")
        app.logger.info("Isaa IDE is running")
        ide_mode = True
    except Exception:
        pass

    if ide_mode:
        app.new_ac_mod("isaa_ide")
        file_functions_ = app.AC_MOD.process_input
        file_functions_dis = """
witch file_functions you have access to 8 file reladead functions
format for singel input functions [create, delete, list, read] arguments is <path>
format for 2 input functions [move, copy] arguments ar <source> <destination>
format for 2 input functions [search] arguments ar <path> <keyword>
format for 2 input functions [write] arguments ar <path> <text>

syntax for function call : <function_name> <arguments>
    """

        def file_functions(x):
            try:
                if input(x + "\n\nACCEPT?\n:").lower() in ["y", "yes"]:
                    return file_functions_(x)
                return "Not authorised by user"
            except Exception as e:
                return "Error in file_functions : " + str(e)

    app.logger.info("Init Isaa")
    app.save_load("isaa")
    app.logger.info("Isaa is running")

    sys.setrecursionlimit(1500)

    app.new_ac_mod('isaa')
    isaa: Isaa = app.AC_MOD
    isaa.loade_keys_from_env()

    self_agent_config: AgentConfig = isaa.get_agent_config_class("self")
    calender_agent_config: AgentConfig = isaa.get_agent_config_class("calender_agent")

    def run_agent_think_wrapper(x):
        if not x:
            return "Provide input"
        return isaa.run_agent("calender_agent", x, mode_over_lode='talk')

    append_calender_agent(calender_agent_config, calender_run, run_agent_think_wrapper)

    def run_calender_agent(x):
        if not x:
            return "Provide input"
        return isaa.run_agent("calender_agent", x)

    def spown_agent(x: str) -> str:

        """
The spown_agent function takes a single string argument x, which is expected to contain a set of key-value pairs separated by colons (:). These pairs specify various attributes of an agent that is to be created and run.

The function parses the input string x and extracts the values associated with the following keys:

    Name: The name of the agent to be created. This key is required and must be present in the input string.
    Mode: The mode in which the agent is to be run. This is an optional key. available ar [free, tools, talk]
    Task: The task that the agent is to perform. This is an optional key.
    Personal: The personality of the agent. This is an optional key.
    Goals: The goals of the agent. This is an optional key.
    Capabilities: The capabilities of the agent. This is an optional key.

The function then creates an AgentConfig object with the specified name and sets its personality, goals, and capabilities attributes to the values associated with the corresponding keys, if those keys were present in the input string."""
        personal: Optional[str] = None
        goals: Optional[str] = None
        task: Optional[str] = None
        name: Optional[str] = None
        capabilities: Optional[str] = None
        mode: Optional[str] = None

        for line in x.splitlines():
            key, value = map(str.strip, line.split(':', 1))
            if key == 'Name':
                name = value
            elif key == 'Mode':
                mode = value
            elif key == 'Task':
                task = value
            elif key == 'Personal':
                personal = value
            elif key == 'Goals':
                goals = value
            elif key == 'Capabilities':
                capabilities = value

        if not name:
            return "ValueError('Agent name must be specified.')"

        agent_config: AgentConfig = isaa.get_agent_config_class(name)

        if personal is not None:
            agent_config.personality = personal

        if goals is not None:
            agent_config.goals = goals

        if capabilities is not None:
            agent_config.capabilities = capabilities

        return isaa.run_agent(agent_config, task, mode_over_lode=mode)

    def momory_wraper(x):
        momoey_ = CollectiveMemory().text(context=x)
        if momoey_ == "[]":
            return "No data found, Try entering other data related to your task"
        return momoey_

    # Adding Tools
    app.AC_MOD.add_tool("spawn_agent ", spown_agent,
                        "The spown_agent function takes a single string argument x, which is expected to contain a set of key-value pairs separated by colons (:). These pairs specify various attributes of an agent that is to be created and run. use agent to divde taskes"
                        , """The function parses the input string x and extracts the values associated with the following keys:

    Name: The name of the agent to be created. This key is required and must be present in the input string.
    Mode: The mode in which the agent is to be run. This is an optional key. available ar [free, tools, talk]
    Task: The task that the agent is to perform. This is an optional key.
    Personal: The personality of the agent. This is an optional key.
    Goals: The goals of the agent. This is an optional key.
    Capabilities: The capabilities of the agent. This is an optional key.

The function then creates an AgentConfig object with the specified name and sets its personality, goals, and capabilities attributes to the values associated with the corresponding keys, if those keys were present in the input string.""",
                        self_agent_config.tools)

    app.AC_MOD.add_tool("memory", momory_wraper, "a tool to get similar information from your memories."
                                                 " useful to get similar data. ", "memory(<related_information>)",
                        self_agent_config.tools)

    app.AC_MOD.add_tool("Calender", run_calender_agent,
                        "a tool to use the calender and mange user task todos und meetings", "Calender(<task>)",
                        self_agent_config.tools)

    app.AC_MOD.add_tool("save_data_to_memory", CollectiveMemory().text_add, "tool to save data to memory,"
                                                                            " write the data as specific"
                                                                            " and accurate as possible.",
                        "save_data_to_memory(<store_information>)",
                        self_agent_config.tools)

    if ide_mode:
        app.AC_MOD.add_tool("file_functions", file_functions, file_functions_dis,
                            " ",
                            self_agent_config.tools)

    def get_input():
        if speek_mode:
            input("Start listig ->")
            return get_audio_transcribe(amplitude_min=min_, s_duration_max=2)
        return input("User:")

    def spek(x, speak_text=speek_mode, vi=0, **kwargs):
        if x.startswith("Action:"):
            return

        if not speak_text:
            return

        if ":" in x:
            x = "".join(x.split(":")[1:])

        cls_lang = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
        ln = cls_lang(x)

        if len(x) > 400:
            x = app.AC_MOD.mas_text_summaries(x, min_length=50)

        if 'de' == ln[0]["label"] and ln[0]["score"] > 0.2:
            text_to_speech3(x)

        elif 'en' == ln[0]["label"] and ln[0]["score"] > 0.5:
            speech_stream(x, voice_index=vi)
        else:
            print_(f"SPEEK SCORE TO LOW : {ln[0]['score']}")

    app.AC_MOD.speek = spek

    response = ""
    print_("Welcome")
    spek("Welcome")
    # app.AC_MOD.genrate_image("Blue 4k Sky", app)
    user_text = task if task else get_input()
    print_("\n================================Starting-Agent================================")
    print_("USER0: " + user_text)
    step = 0
    task_done = False
    #while app.alive:

    self_agent_config.mode = 'planning'
    user_text = """
    Create a life plan for the next 2 years.
I am 20 years old and am in 4 months with my trial study finished. I do not necessarily want to continue studying. and if so then not in berlin to collect new impressions. what should I do the next 2 years my interests are almost all Technich programming but I go from time to time aucg like bouldering
    """
    response, task_done = run_agent_cmd(app.AC_MOD, user_text, self_agent_config, step, spek)

    self_agent_config.task_list = response.split('\n')
    self_agent_config.mode = 'execution'
    user_text = ''
    while not task_done:
        response, task_done = run_agent_cmd(app.AC_MOD, user_text, self_agent_config, step, spek)
        print(self_agent_config.last_prompt)
        print(response)
        print(task_done)
        input("-")


    a = """
    if stop_helper(response):
        text_to_speech3("Ich habe eine Frage")
        spek(response, vi=1)
        user_text = get_input()
        if user_text.lower() in "exit":
            app.alive = False
        if user_text.lower() in "sleep":
            awake = False
        print_(f"User: {user_text}")
    if task_done:  # TODO CLEar St mem to storage -- new task
        if speek_mode:
            text_to_speech3("Ist die Aufgabe abgeschlossen?")
        else:
            print("Ist die Aufgabe abgeschlossen?")
        print("Type:\n\texit\n\tstop or sleep\n\tyes or y\n\tno or n\n\tand for prompt end with -p")
        p = input("User Input: ").lower()
        if p.endswith("p"):
            print(self_agent_config.last_prompt)
            input("User Input: .. ")
        if p.startswith("e") or p.startswith("y"):
            app.alive = False
        elif p.startswith("n"):
            user_text = get_input()
            if not user_text:
                user_text = input("Sorry got no input ... :")
        else:
            pass
        print_(f"User: {user_text}")
    else:
        if input("Agent continues - ") in ['e', 'exit']:
            app.alive = False
"""
    step += 1


