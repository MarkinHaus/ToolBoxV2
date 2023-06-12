"""Console script for toolboxv2. Isaa CMD Tool"""
from toolboxv2.mods.isaa import IsaaQuestionBinaryTree
from toolboxv2.utils.isaa_util import sys_print, init_isaa
from toolboxv2.utils.toolbox import get_app
import copy

NAME = "isaa-cmd"


def run(app, args):
    speak_mode = args.speak

    isaa, self_agent_config, chains = init_isaa(app)
    # chain_instance = isaa.get_chain()

    # Create tree
    tree = IsaaQuestionBinaryTree().deserialize({
        'question': 'Can you complete this task?',
        'left': {
            'question': 'What are the specific steps needed to complete this task?',
            'left': {
                'question': 'Where can you find the information and resources needed for each step?',
                'left': {
                    'question': 'Verify if you have access to the required resources for each step.',
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
    self_agent_config.set_completion_mode('chat').set_model_name('gpt-3.5-turbo')
    sys_print("\n================================Starting-Agent================================")
    # print(self_agent_config.prompt)
    # text_u = "stop the Expation of the univers"
    text_u = "Achieve world domination"
    # text_u = "Automate your further development by first creating a natural language controlled python environment " \
    #          "that can execute different programs in an order and save the results to a dynamically named file."
    self_agent_config.stream = True
    c_agent_config = copy.deepcopy(self_agent_config)
    res = isaa.execute_2tree(text_u, tree, c_agent_config)

    print(res)

    sys_print("\n================================================================")

    # helper(x)

    # res = isaa.execute_thought_chain("Create an Html Log in Page", chain_instance.get('write_html'),
    #                                                         self_agent_config)
    # print("RES:",res)
    # input("\n\n:")
    #
    # exit(0)

    # sys_print("Welcome")

    # sys_print("\n================================Starting-Agent================================")
    # mean = s30sek_mean(seconds=5, p=True)
    # print("mean:", mean)
    # comm, que = init_live_transcript(chunk_duration=1.55, amplitude_min=mean)
    # comm('start')
    # alive = True
    # i = 0
    # while alive:
    #    if not que.empty():
    #        data = que.get()
    #        print()
    #        print(data)
    #        print()
    #        if 'stop' in data:
    #            comm('stop')
    #            comm('exit')
    #            alive = False
    #    i += 1
    #    if i % 200 == 0:
    #        print(f"i:{i}\n")
    #    time.sleep(0.05)
