import random

import time

import pyperclip

from toolboxv2 import Spinner
from toolboxv2.utils.toolbox import get_app

"""Console script for toolboxv2. Isaa Conversation Tool"""
from toolboxv2.mods.isaa_audio import init_live_transcript, speech_stream, s30sek_mean
from toolboxv2.mods.isaa.isaa_modi import sys_print, init_isaa
from toolboxv2.mods.isaa.AgentUtils import AgentConfig
from toolboxv2.mods.isaa import Tools as Isaa

NAME = "isaa-talk"


def run(app, args):
    isaa: Isaa
    self_agent_config: AgentConfig
    isaa, self_agent_config, chains = init_isaa(app, speak_mode=True, calendar=False, ide=False, create=False,
                                                isaa_print=False, python_test=True, init_mem=False, init_pipe=True,
                                                join_now=False, global_stream_override=False, chain_runner=True
                                                )

    mean = 72.06#s30sek_mean(seconds=20, p=True)
    comm, que = init_live_transcript(chunk_duration=2.6, amplitude_min=mean, model="whisper-1")

    alive = True

    # isaa.init_from_augment(isaa.config['augment'], self_agent_config,
    #                        exclude=['task_list', 'task_list_done', 'step_between', 'pre_task', 'task_index'])

    def ask_user(question:str):
        print("")
        print("================================")
        print(f"Agent is Asking user : {question}")
        with Spinner("Generating audio..."):
            speech_stream(question, voice_index=0)

        user_response = ""

        getting_infos = True

        while getting_infos:

            while not que.empty():
                parte = que.get()
                print(f'\n{parte}\n')
                user_response += parte + ' '
                if 'stop' in parte:
                    comm('stop')
                    user_response = ''

            user_i = input(f"Continue with : '{user_text}' with y")

            if 'r' in user_i:
                user_response = ''

            if 'y' not in user_i:
                continue

            getting_infos = False

        return user_response

    def spek_to_user(text: str):
        with Spinner("Generating audio..."):
            speech_stream(text, voice_index=0)

    isaa.add_tool("ask_user", ask_user, "ask user for permission or information only use if ur unsertant or"
                                        " ther is a lack of informations usage provide a concrete question fo the user", "ask_user(question)", self_agent_config)
    isaa.add_tool("spek_to_user", spek_to_user, "use to speak the input text to the user. use to comunicate ur thoghts and prosess",
                  "spek_to_user(text)", self_agent_config)

    isaa.speak = spek_to_user

    """
    Based on the information provided, Markin is currently studying MINT green, an orientation program for computer science studies at TU Berlin. However, he is not satisfied with the quality of education and is considering studying abroad or taking a gap year. To create a life plan for the next 2 years, we need more information about Markin's goals, interests, and priorities.

To help Markin, the following agents and tools can be utilized:

1. Career counselor: To identify career goals, interests, and suggest suitable paths and educational programs.
2. Education consultant: To evaluate the current educational program and suggest alternatives or universities that align with Markin's goals.
3. Financial advisor: To evaluate financial situations and suggest ways to finance education and living expenses.
4. Language learning tools: To learn a new language if studying abroad, such as Duolingo or Rosetta Stone.
5. Travel booking tools: To make travel arrangements, such as Expedia or Kayak.
6. Time management tools: To balance studies and other activities, such as calendars, to-do lists, and productivity apps.
7. Communication tools: To stay in touch with family and friends, such as Skype, WhatsApp, or Zoom.

The following skills would be helpful for this task:

- Get a differentiated point of view: To understand Markin's perspective and priorities.
- Search for information: To gather information about potential universities or programs.
- Calendar entry: To organize and schedule important dates and deadlines.
- Generate documents: To create and organize documents related to education and travel plans.
- Utilize tools: To identify and use tools and resources for organizing and planning.
- Read, analyze, and write summaries: To summarize and organize information about potential universities or programs.

It is essential to gather more information about Markin's situation and preferences before making any decisions or recommendations.

Ask questions to help to find a decisions or recommendations.
    """

    def quickAudioCommandMapping(text, options=None):
        if options is None:
            options = [
                "Talk With Isaa",
                "Give Isaa a Mission",
                "Exit the system",
            ]
        if len(text) > 500:
            text = " ".join(text[:500].split(" ")[:-1])
        #tasks = ""
        #i = 0
        #for t in options:
        #    tasks += f"{t} return the value : {i}\n"
        #    i += 1

        if len(text) < 500:
            i = 0
        if len(text) < 360:
            i = 1
        if len(text) < 60:
            i = 2

        tasks = options[i]

        return i, tasks

    input("Start listening: ")

    comm('start')
    issa_res = ""
    user_text = ""
    time.sleep(4)
    try:
        while alive:
            while not que.empty():
                data = que.get()
                print(f'\n{data}\n')
                user_text += data + ' '
                if 'stop' in data:
                    comm('stop')
                    user_text = ''
                if 'ende' in data:
                    comm('exit')
                    alive = False

            user_text = user_text.strip()

            user_i = input(f"Continue with : '{user_text}' with y")

            if 'x' in user_i:
                alive = False
                user_text = ''
                continue

            if 'r' in user_i:
                user_text = ''
                continue

            if 's' in user_i:
                comm('stop')
                user_text = ''
                if 's' in input(f"Start or exit").lower():
                    comm('start')
                    continue
                alive = False

            if 'y' not in user_i:
                continue

            comm('stop')

            do, options_return = quickAudioCommandMapping(user_text)

            if do == 3:
                continue

            if 'y' not in input(f"Accept : '{options_return}' with y"):
                do = int(input(f"enet nuber or -1  to retry or -2 to exit\n0 self runner 1 missions runner :"))

                if do == -1:
                    user_text = ""
                    continue

                if do == -2:
                    user_text = ""
                    comm('exit')
                    alive = False

            context = pyperclip.paste()
            if context:
                self_agent_config.short_mem.text = context

            if do == 0:
                issa_res, _ = isaa.run_chain_on_name("SelfRunner", user_text)
            if do == 1:
                issa_res, _ = isaa.run_chain_on_name("liveRunnerMission", user_text)
            if do == 2:
                comm('exit')
                alive = False

            if issa_res:

                isaa.print(issa_res)

                with Spinner("Generating audio..."):
                    speech_stream(issa_res, voice_index=0)

            #
            self_agent_config.save_to_permanent_mem()
            user_text = ''
            input("Start listening: ")
            comm('start')
    except Exception as e:
        print('Error :', e)

    comm('exit')
    print("Auf wieder sehen")
    isaa.save_to_mem()
    return "Done!"
    # qa = isaa.init_db_questions('main', self_agent_config)
    # if qa is None:
    #     return
    # chat_history = []
    # while True:
    #     q = input('Question:')
    #     if q == 'quit':
    #         break
    #     result = qa({"question": q, "chat_history": chat_history})
    #     chat_history.append((q, result['answer']))
    #     print(f"-> **Question**: {q} \n")
    #     print(f"**Answer**: {result['answer']} \n")
#
    # print("================================")
