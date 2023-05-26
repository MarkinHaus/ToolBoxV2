import msvcrt
import queue
import threading

import keyboard
import pyperclip

from toolboxv2 import Spinner
from toolboxv2.mods.isaa import IsaaQuestionBinaryTree

"""Console script for toolboxv2. Isaa Conversation Tool"""
from toolboxv2.mods.isaa_audio import init_live_transcript, s30sek_mean
from toolboxv2.utils.isaa_util import sys_print, init_isaa

from transformers.tools.text_classification import TextClassificationTool

NAME = "isaa-talk"


def run(app, args):
    isaa, self_agent_config, chains = init_isaa(app, speak_mode=True, calendar=True, ide=True, create=True,
                                                isaa_print=False, python_test=True, init_mem=True, init_pipe=True,
                                                join_now=True, global_stream_override=True, chain_runner=True
                                                )

    is_task_list = TextClassificationTool(model='facebook/bart-large-mnli')

    mean = 82#s30sek_mean(seconds=.5, p=True)
    comm, que = init_live_transcript(chunk_duration=1.55, amplitude_min=mean)

    mode = 'free'
    alive = True

    user_text = ''
    while alive:
        while not que.empty():
            data = que.get()
            print(f'\n{data}\n')
            user_text += data + ' '
            if 'stop' in data:
                comm('stop')
                user_text = ''
            if 'ende' in data:
                comm('stop')
                comm('exit')
                alive = False

        buffer = input(f"\ncurrent input : {user_text}\ncurrent mode : {mode}\ncommands ar : exit\n\tstart #s"
                       "\n\treset input #r"
                       "\n\treset and stop #rs"
                       "\n\trun isaa #"
                       "\n\tclear memory #cl"
                       "\n"
                       "\nswitch mode commands:"
                       "\n\texecution #e"
                       "\n\tfree #f"
                       "\n\tconversation #c \n\n:")

        if buffer.endswith("exit"):
            alive = False
            que.put('stop')
        if buffer.endswith("#e"):
            mode = 'execution'
            print(f"Switch to {mode=}")
        if buffer.endswith("#f"):
            mode = 'free'
            print(f"Switch to {mode=}")
        if buffer.endswith("#c"):
            mode = 'conversation'
            print(f"Switch to {mode=}")
        if buffer.endswith("#s"):
            comm('start')
            print(f"Start")
        if buffer.endswith("#rs"):
            comm('stop')
            print(f"stop")
            user_text = ''
        if buffer.endswith("#r"):
            user_text = ''
        if buffer.endswith("#cl"):
            self_agent_config.short_mem.memory_data = []
            self_agent_config.observe_mem.memory_data = []
            self_agent_config.edit_text.memory_data = []
            isaa.get_chain().load_from_file()
            print("Memory cleared")
        if buffer.endswith("#"):
            if buffer.endswith("##"):
                user_text = buffer.replace("##", "")
            print("You pressed enter!")
            comm('stop')
            # self_agent_config.set_completion_mode('chat').set_model_name('gpt-3.5-turbo')
            while not que.empty():
                user_text += que.get() + ' '
            sys_print("\n================================Starting-Agent================================")
            print("User text :", user_text)
            context = pyperclip.paste()
            if context:
                self_agent_config.edit_text.text = context

            # Get information about the task.

            chain_instance = isaa.get_chain()

            # Create tree
            tree = IsaaQuestionBinaryTree().deserialize({
                'question': 'Can you complete this task?',
                'left': {
                    'question': 'What are the specific steps needed to complete this task?',
                    'left': {
                        'question': 'Where can you find the information and resources needed for each step?',
                        'left': {
                            'question': 'In which order dos the function need to be called to complete this task?',
                            'left': None,
                            'right': None
                        },
                        'right': {
                            'question': 'Identify potential obstacles in obtaining the necessary resources and plan how to overcome them?',
                            'left': None,
                            'right': None
                        }
                    },
                    'right': {
                        'question': 'How would you classify this task to successfully complete it?',
                        'left': {
                            'question': 'What similar tasks have you successfully completed before?',
                            'left': None,
                            'right': None
                        },
                        'right': {
                            'question': 'Are there any particular strategies that you would apply to this task based on its classification?',
                            'left': None,
                            'right': None
                        }
                    }
                },
                'right': {
                    'question': 'What skills are you lacking to complete this task?',
                    'left': {
                        'question': 'What would you need to do to acquire these skills?',
                        'left': {
                            'question': 'How much time would it take for you to acquire these skills?',
                            'left': None,
                            'right': None
                        },
                        'right': {
                            'question': 'What assistance or resources would you need to acquire these skills?',
                            'left': None,
                            'right': None
                        }
                    },
                    'right': {
                        'question': 'Are there any alternative strategies or resources that you could use to complete the task despite lacking these skills?',
                        'left': None,
                        'right': None
                    }
                }
            })
            self_agent_config.binary_tree = tree
            self_agent_config.set_completion_mode('chat').set_model_name('gpt-3.5-turbo')
            sys_print("\n===============Analysing user Input============")

            res = isaa.run_agent(self_agent_config, user_text, mode_over_lode='q2tree')
            is_task_list(f"Question:{user_text}\nResponse:{res}", labels=["positive answer", "positive completion plan"])
            don, next_on, speak = False, 0, res
            for line in res.split("\n"):
                if line.startswith("+1"):
                    line = line.strip().replace("+1", "")
                    speak = line
                    if '0' in line:
                        don = True
                        break
                    for char in line:
                        try:
                            next_on = int(char)
                        except ValueError:
                            pass
            print()
            print(don, next_on, speak)

            res = isaa.run_agent(self_agent_config, user_text
                                 , mode_over_lode=mode)


            print(res)
            user_text = ''
            print("\n")

    comm('stop')
    comm('exit')
    print("Auf wieder sehen")

    qa = isaa.init_db_questions('main', self_agent_config)
    if qa is None:
        return
    chat_history = []
    while True:
        question = input('Question:')
        if question == 'quit':
            break
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result['answer']))
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")

    print("================================")
