import platform

import keyboard

# pyperclip.copy('The text to be copied to the clipboard.')
# pyperclip.paste()

import pyperclip

from transformers import pipeline

from toolboxv2 import Style, Spinner
from toolboxv2.mods.isaa import AgentConfig
from toolboxv2.mods.isaa import AgentChain
from toolboxv2.mods.isaa import Tools as Isaa
from toolboxv2.mods.isaa_audio import text_to_speech3, get_audio_transcribe
from toolboxv2.utils.isaa_util import init_isaa, sys_print, run_agent_cmd, stop_helper, idea_enhancer, generate_exi_dict

NAME = "isaa-clip"


def get_speak_input():
    pass


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
    self_agent_config.stream = True
    self_agent_config.completion_mode = "chat"
    self_agent_config.stop_sequence = ["\n\n\n", "Observation:", "Execute:"]

    while alive_isaa:
        print("\nAGENT section\n")
        response = app.AC_MOD.run_agent("self", user_text)
        print("\nAGENT section END\n")

        task_done = app.AC_MOD.test_task_done(response)

        print_(f"\n{'=' * 20}STEP:{step}{'=' * 20}\n")
        print_(f"\tMODE               : {self_agent_config.mode}\n")
        print_(f"\tObservationMemory  : {self_agent_config.observe_mem.tokens}\n")
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

    self_agent_config.model_name = "code-davinci-edit-001"  # "gpt-3.5-turbo-0613"
    self_agent_config.completion_mode = "edit"

    sys_print("USER0: " + user_text)

    while not task_done:
        try:
            response, task_done = run_agent_cmd(isaa, user_text, self_agent_config, step, isaa.speak)
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
    task_done = True

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

    self_agent_config.model_name = "gpt-3.5-turbo-0613"
    self_agent_config.mode = "free"
    self_agent_config.completion_mode = "chat"

    agent_categorize_config: AgentConfig = isaa.get_agent_config_class("categorize")

    agent_categorize_config \
        .set_mode('free') \
        .set_completion_mode('text') \
        .set_model_name('gpt-3.5-turbo-0613') \
        .stream = False
    # user_text = 'cloudM.py'
    sys_print("USER0: " + user_text)

    isaa.get_agent_config_class("think").set_model_name('gpt-4').stream = True

    res = isaa.run_agent(agent_categorize_config, f"What chain '{str(chain_instance)}'"
                                                  f" \nis fitting for this input '{user_text}'\n")
    chain = ''
    if res in list(chain_instance.chains.keys()):
        chain = res

    print("chain: ", chain)
    print()
    infos = '\n'.join([f'{item[0]} ID: {item[1]}' for item in list(zip(chain_instance.chains.keys(),
                                                                       range(len(chain_instance.chains.keys()))))])
    user_vlidation = input(
        f"if its the chain is wrong type the corresponding number {infos}\nInput: ")
    if user_vlidation in ['y']:
        task_done = False
    elif user_vlidation:
        try:
            user_vlidation = int(user_vlidation)
            chain = list(chain_instance.chains.keys())[user_vlidation]
            task_done = False
        except ValueError:
            print("Invalid user input please type y or the chain number")

    pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

    # self_agent_config.short_mem.max_length = 2676*4.3

    while not task_done:
        # try:
        evaluation, chain_ret = isaa.execute_thought_chain(user_text, chain_instance.get(chain),
                                                           self_agent_config)
        # except Exception as e:
        #    print(e, '🔴')
        #    return "ERROR"
        evaluation = evaluation[::-1][:300][::-1]
        pipe_res = pipe(evaluation)
        # aweeeewedspeak(f"The evaluation of the chain is {evaluation} i am {int(pipe_res[0]['score'])*10} Peasant sure")
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

    return response


def run_idea_development(isaa: Isaa, user_text, chains, self_agent_config):
    sys_print("\n================================Starting-Agent================================")
    res = idea_enhancer(isaa, user_text, self_agent_config, chains, create_agent=True)
    sys_print("\n================================Exit-Agent================================")
    return res


def run_generate_exi_dict(isaa: Isaa, user_text, self_agent_config):
    sys_print("\n================================Starting-Agent================================")
    res = generate_exi_dict(isaa, user_text, create_agent=False, tools=list(self_agent_config.tools.keys()), retrys=3)
    sys_print("\n================================Exit-Agent================================")
    return res


def get_buffer(buffer):
    if platform.system() != "Darwin":

        event = keyboard.read_event()

        if event.event_type == keyboard.KEY_DOWN:
            key_name = event.name

            if key_name in ['backspace', 'enter', 'space']:
                buffer = ' ' * 8
            else:
                buffer += str(key_name).lower()

            print(buffer[len(buffer) - 8:], end='\r')
    else:
        buffer = input(":")

    return buffer


def run(app, args):
    speak_mode = args.speak
    # Trigger word to process the text
    trigger_word = '##i'
    trigger_word_editor = '##e'
    trigger_word_chad = '##c'
    trigger_idea_dev = '##h'
    trigger_gen_dic = '##g'

    print(f"Script running in the background")

    isaa, self_agent_config, chains = init_isaa(app, speak_mode=speak_mode, calendar=True, ide=True, create=True)

    idea_enhancer(isaa, '', self_agent_config, chains, create_agent=True)
    generate_exi_dict(isaa, '', create_agent=True, tools=self_agent_config.tools, retrys=0)
    print("init completed waiting for trigger word: ")
    buffer = ' ' * 8
    alive = True
    while alive:

        buffer = get_buffer(buffer)

        if buffer.endswith("isaa-exit"):
            alive = False

        if buffer.endswith("isaa-clear"):
            self_agent_config.short_mem.memory_data = []
            self_agent_config.observe_mem.memory_data = []
            self_agent_config.edit_text.memory_data = []
            isaa.get_chain().load_from_file()
            print("Memory cleared")

        if buffer.endswith(trigger_word):
            print("Isaa Running simple exiqution\n")
            context = pyperclip.paste()

            if context:
                self_agent_config.short_mem.text = context

            user_text = get_input(speek_mode=speak_mode)

            res = run_agent_clip(app, user_text, self_agent_config, sys_print, isaa.speak, get_input, speak_mode)

            if res:
                print("Agent:", res)
                pyperclip.copy(res)

            buffer = ' ' * 8
            print("waiting for trigger word:: ")

        if buffer.endswith(trigger_word_editor):

            print("Isaa Running Editor\n")
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

            print("Isaa Running Chain runner\n")
            context = pyperclip.paste()

            if context:
                self_agent_config.edit_text.text = context

            user_text = get_input(speek_mode=speak_mode)

            res = run_chad(isaa, user_text, self_agent_config)

            if res:
                isaa.speak(res)
                print("Agent:", res)
                pyperclip.copy(res)

            buffer = ' ' * 8
            print("waiting for trigger word:: ")

        if buffer.endswith(trigger_idea_dev):

            print("Isaa Running Idea development\n")
            context = pyperclip.paste()

            if context:
                self_agent_config.edit_text.text = context

            user_text = get_input(speek_mode=speak_mode)

            res = run_idea_development(isaa, user_text, chains, self_agent_config)

            if res:
                isaa.speak(res)
                print("Agent:", res)
                pyperclip.copy(res)

            buffer = ' ' * 8
            print("waiting for trigger word:: ")

        if buffer.endswith(trigger_gen_dic):

            print("Isaa Running Plan generation\n")
            context = pyperclip.paste()

            if context:
                self_agent_config.edit_text.text = context

            user_text = get_input(speek_mode=speak_mode)

            res = run_generate_exi_dict(isaa, context + '\n' + user_text, self_agent_config)

            if res:
                isaa.speak(res)
                print("Agent:", res)
                pyperclip.copy(str(res))

            buffer = ' ' * 8
            print("waiting for trigger word:: ")

    print("\nExiting...")
