import sys

# pyperclip.copy('The text to be copied to the clipboard.')
# pyperclip.paste()

import keyboard
import pyperclip

from transformers import pipeline

from toolboxv2 import Style, Spinner
from toolboxv2.mods.isaa import CollectiveMemory, AgentConfig, AgentChain, Tools as Isaa
from toolboxv2.mods.isaa_audio import text_to_speech3, get_audio_transcribe, get_speak_input
from toolboxv2.utils.isaa_util import init_isaa, sys_print, speak, run_agent_cmd, stop_helper

NAME = "isaa-clip"


def get_input(speek_mode=False, min_=30):
    if speek_mode:
        input("Start listig ->")
        return get_speak_input()
    return input("User:")


def run_agent_clip(app, user_text, self_agent_config, print_, spek, get_input, speek_mode=False):
    alive_isaa = True
    step = 0
    print_("\n================================Starting-Agent================================")
    final = ""

    self_agent_config.model_name = "gpt-4"
    self_agent_config.mode = "free"
    self_agent_config.completion_mode = "chat"
    self_agent_config.stop_sequence = ["\n\n\n", "Observation:", "Execute:"]

    while alive_isaa:
        print("\nAGENT section\n")
        response = app.AC_MOD.run_agent("self", user_text)
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
                final = response
                alive_isaa = False

        step += 1

        response = ""
    return final


def run_editor(isaa, user_text, self_agent_config):
    sys_print("Starting Isaa editor")
    response = ""
    sys_print("\n================================Starting-Agent================================")
    step = 0
    task_done = False

    self_agent_config.model_name = "code-davinci-edit-001"  # "gpt-3.5-turbo"
    self_agent_config.completion_mode = "edit"

    sys_print("USER0: " + user_text)

    while not task_done:
        try:
            response, task_done = run_agent_cmd(isaa, user_text, self_agent_config, step, speak)
        except Exception as e:
            print(e, '🔴')

        if task_done:
            print(f'🟢')
        if not task_done:
            print('🔵')
            user_text = input("Update Part: ")

        if user_text in ['-exit', '-don']:
            return response

        step += 1

    return response


def run_chad(isaa: Isaa, user_text, self_agent_config):
    sys_print("Starting Isaa Chad")
    response = ""
    sys_print("\n================================Starting-Agent================================")
    step = 0
    task_done = False

    chain_instance: AgentChain = isaa.get_chain()
    chain_instance.load_from_file()

    # chain_instance.add("calendar_entry", [
    #     {
    #         "use": 'agent',
    #         "name": "categorize",
    #         "args": "Bestimme den Typ des Kalendereintrags basierend auf $user-input",
    #         "return": "$entry-type",
    #     },
    #     {
    #         "use": 'tool',
    #         "name": "Calender",
    #         "args": "Speichere den Eintrag Typ: $entry-type \nUser: $user-input",
    #         "infos": f"$Date"
    #     }
    # ])

    self_agent_config.model_name = "gpt-3.5-turbo"
    self_agent_config.mode = "free"
    self_agent_config.completion_mode = "chat"

    agent_categorize_config: AgentConfig = isaa.get_agent_config_class("categorize")

    agent_categorize_config \
        .set_mode('free') \
        .set_completion_mode('text') \
        .set_model_name('gpt-3.5-turbo') \
        .stream = True

    sys_print("USER0: " + user_text)

    res = isaa.run_agent(agent_categorize_config, f"What chain '{str(chain_instance)}'"
                                                  f" \nis fitting for this input '{user_text}'")
    chain = ''
    for chain_key in chain_instance.chains.keys():
        if res in chain_key:
            chain = chain_key
            break
    print("chain: ", chain)
    user_vlidation = input(
        f"if its the chain is wrong type the corresponding number {list(zip(chain_instance.chains.keys(), range(len(chain_instance.chains.keys()))))} :")
    if user_vlidation:
        user_vlidation = int(user_vlidation) - 1
        chain = list(chain_instance.chains.keys())[user_vlidation]

    pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")


    while not task_done:
        try:
            evauation, chain_ret = isaa.execute_thought_chain(user_text, chain_instance.get(chain),
                                                              self_agent_config.set_model_name("gpt-4"), speak=speak)
        except Exception as e:
            print(e, '🔴')
            return "ERROR"

        pipe_res = pipe("evauation")
        speak(f"The evaluation of the chain is {evauation} i am {int(pipe_res[0]['score'])*10} Peasant sure")

        print(pipe_res[0]['score'])
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

    return response


def run(app, args):
    speak_mode = args.speak
    # Trigger word to process the text
    trigger_word = '##isaa'
    trigger_word_editor = '##code'
    trigger_word_chad = '##chad'

    print(f"Script running in the background")

    isaa, self_agent_config = init_isaa(app, speak_mode=speak_mode, calendar=True, ide=True, create=True)
    print("init completed waiting for trigger word: ")
    buffer = ' ' * 8
    alive = True
    while alive:
        event = keyboard.read_event()

        if event.event_type == keyboard.KEY_DOWN:
            key_name = event.name

            if key_name in ['backspace', 'enter', 'space']:
                buffer = ' ' * 8
            else:
                buffer += str(key_name).lower()

            print(buffer[len(buffer) - 8:], end='\r')

            if buffer.endswith("isaa-exit"):
                alive = False

            if buffer.endswith("isaa-clear"):
                self_agent_config.short_mem.memory_data = []
                self_agent_config.obser_mem.memory_data = []
                self_agent_config.edit_text.memory_data = []
                isaa.get_chain().load_from_file()
                print("Memory cleared")

            if buffer.endswith(trigger_word):
                print("Isaa Running N\n")
                context = pyperclip.paste()

                if context:
                    self_agent_config.short_mem.text = context

                user_text = get_input(speek_mode=speak_mode)

                res = run_agent_clip(app, user_text, self_agent_config, sys_print, speak, get_input, speak_mode)

                if res:
                    print("Agent:", res)
                    pyperclip.copy(res)

                buffer = ' ' * 8
                print("waiting for trigger word:: ")

            if buffer.endswith(trigger_word_editor):

                print("Isaa Running E\n")
                context = pyperclip.paste()

                if context:
                    self_agent_config.edit_text.text = context

                user_text = get_input(speek_mode=speak_mode)

                res = run_editor(isaa, user_text, self_agent_config)

                if res:
                    print("Agent:", res)
                    pyperclip.copy(res)

                buffer = ' ' * 8
                print("waiting for trigger word:: ")

            if buffer.endswith(trigger_word_chad):

                print("Isaa Running C\n")
                context = pyperclip.paste()

                if context:
                    self_agent_config.edit_text.text = context

                user_text = get_input(speek_mode=speak_mode)

                res = run_chad(isaa, user_text, self_agent_config)

                if res:
                    speak(res)
                    print("Agent:", res)
                    pyperclip.copy(res)

                buffer = ' ' * 8
                print("waiting for trigger word:: ")

    print("\nExiting...")
