import sys

# pyperclip.copy('The text to be copied to the clipboard.')
# pyperclip.paste()

import keyboard
import pyperclip

from transformers import pipeline

from toolboxv2 import Style, App
from toolboxv2.mods.isaa import CollectiveMemory, AgentConfig
from toolboxv2.mods.isaa_audio import text_to_speech3, get_audio_transcribe
from toolboxv2.runabel.isaa_cmd import stop_helper, print_to_console, test_amplitude_for_talk_mode


def set_up(app, speek_mode, trigger_word):
    min_ = 105
    speech_stream = lambda x, **kwargs: x
    ide_mode = False

    if speek_mode:
        min_ = test_amplitude_for_talk_mode(sek=2)
        print("Done Testing : " + str(min_))
        app.logger.info("Init audio")
        # init setup
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

    app.save_load("isaa_calendar")
    app.logger.info("Isaa audio is running")
    app.new_ac_mod("isaa_calendar")

    calender_run = app.AC_MOD.get_llm_tool("markinhausmanns@gmail.com")
    append_calender_agent = app.AC_MOD.append_agent

    app.logger.info("Init Isaa")
    app.save_load("isaa")
    app.logger.info("Isaa is running")


    sys.setrecursionlimit(1500)

    # short_term_mem_file = "isaa_data/shortMemory.txt"
    observation_term_mem_file = "isaa_data/observationMemory/"

    app.new_ac_mod('isaa')
    app.AC_MOD.observation_term_mem_file = observation_term_mem_file
    app.AC_MOD.loade_keys_from_env()

    calender_agent_config: AgentConfig = app.AC_MOD.get_agent_config_class("calender_agent")
    def run_agent_think_wrapper(x):
        if not x:
            return "Provide input"
        return app.AC_MOD.run_agent("calender_agent", x, mode_over_lode='talk')

    append_calender_agent(calender_agent_config, calender_run, run_agent_think_wrapper)

    calender_agent_config.name = "calender_agent"

    calender_agent_config.short_mem.max_length = 3000

    def momory_wraper(x):
        momoey_ = CollectiveMemory().text(context=x)
        if momoey_ == "[]":
            return "No data found, Try entering other data related to your task"
        return momoey_

    # Adding Tools
    app.AC_MOD.add_tool("memory", momory_wraper, "a tool to get similar information from your memories."
                                                 " useful to get similar data. ", "memory(<related_information>)",
                        calender_agent_config.tools)

    app.AC_MOD.add_tool("save_data_to_memory", CollectiveMemory().text_add, "tool to save data to memory,"
                                                                            " write the data as specific"
                                                                            " and accurate as possible.",
                        "save_data_to_memory(<store_information>)",
                        calender_agent_config.tools)

    if ide_mode:
        app.AC_MOD.add_tool("file_functions", file_functions, file_functions_dis,
                            " ",
                            calender_agent_config.tools)

    calender_agent_config.mode = 'free'

    def get_input():
        if speek_mode:
            input("Start listig ->")
            return get_audio_transcribe(amplitude_min=min_, s_duration_max=2)
        return input("User:")

    def print_(x, **kwargs):
        print_to_console("SYSTEM:", Style.style_dic['BLUE'], x, max_typing_speed=0.04, min_typing_speed=0.08)

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

    print_(f"Welcome the trigger word is : '{trigger_word}'")
    spek("welcome i am waiting for your task activate me with the triggerword and the context in your clipboard.")

    return calender_agent_config, print_, spek, get_input, speek_mode


def run_agent_clip(app, user_text, self_agent_config, print_, spek, get_input, speek_mode=False):
    alive_isaa = True
    step = 0
    print_("\n================================Starting-Agent================================")
    print_("USER0: " + user_text)
    response = ""
    final = ""
    while alive_isaa:
        print("\nAGENT section\n")
        response = app.AC_MOD.run_agent(self_agent_config, user_text)
        print("\nAGENT section END\n")

        task_done = app.AC_MOD.test_task_done(response, self_agent_config)

        print_(f"\n{'=' * 20}STEP:{step}{'=' * 20}\n")
        print_(f"\tMODE               : {self_agent_config.mode}\n")
        print_(f"\tCollectiveMemory   : {CollectiveMemory().token_in_use} | total vec num : "
               f"{CollectiveMemory().memory.get_stats()['total_vector_count']}\n")
        print_(f"\tObservationMemory  : {self_agent_config.obser_mem.tokens}\n")
        print_(f"\tShortTermMemory    : {self_agent_config.short_mem.tokens}\n\n")
        if "Answer: " in response:
            print_("AGENT: " + response.split('Answer:')[1] + "\n")
            spek(response.split('Answer:')[1])
            final = response.split('Answer:')[1]
        else:
            print_("AGENT: " + "\n".join(response.split(':')) + "\n")
        if stop_helper(response):

            text_to_speech3("Ich habe eine Frage")
            spek(response, vi=1)

            user_text = get_input()
            if user_text.lower() in "exit":
                alive_isaa = False

            print_(f"User: {user_text}")

        if task_done:
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
                alive_isaa = False
            elif p.startswith("n"):
                user_text = get_input()
                if not user_text:
                    user_text = input("Sorry got no input ... :")
            else:
                pass

            print_(f"User: {user_text}")

        else:
            if input("Agent continues - ") in ['e', 'exit']:
                alive_isaa = False

        step += 1

        response = ""
    return final


def run_appointment_isaa(app, speak_mode=False):
    # Trigger word to process the text
    trigger_word = '##ic'

    print(f"Script running in the background")

    agent_config, print_, spek, get_input, speak_mode = set_up(app, speak_mode, trigger_word)
    buffer = ' ' * 9
    alive = True
    print('Running ', agent_config.name)
    while alive:
        event = keyboard.read_event()

        if event.event_type == keyboard.KEY_DOWN:
            key_name = event.name

            if key_name in ['backspace', 'enter', 'space']:
                buffer = ' ' * 9
            else:
                buffer += str(key_name).lower()
                buffer = buffer[1:]

            print(f"{buffer}, {buffer.endswith('isaa-exit')}, {buffer.endswith(trigger_word)}", end='\r')

            if buffer.endswith("isaa-exit"):
                alive = False

            if buffer.endswith("isaa-clear"):
                agent_config.short_mem.memory_data = []
                agent_config.obser_mem.memory_data = []

            if buffer.endswith(trigger_word):
                print("Isaa Running\n")# asdasdwdsadwdas
                context = pyperclip.paste()

                if context:
                    agent_config.short_mem.text = context

                user_text = get_input()

                res = run_agent_clip(app, user_text, agent_config, print_, spek, get_input, speak_mode)

                if res:
                    print("Agent-final:", res)
                    pyperclip.copy(res)

                buffer = ' ' * 8

    print("\nExiting...")


if __name__ == '__main__':

    app = App("Calender-debug")

    run_appointment_isaa(app)

    exit(0)

