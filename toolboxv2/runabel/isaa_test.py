"""Console script for toolboxv2. Isaa CMD Tool"""

from toolboxv2 import Style, Spinner
from toolboxv2.mods.isaa import AgentChain, IsaaQuestionBinaryTree, AgentConfig
from toolboxv2.utils.isaa_util import init_isaa, extract_dict_from_string, split_todo_list, generate_exi_dict, \
    run_chain_in_cmd, free_run_in_cmd, validate_dictionary, idea_enhancer

NAME = "isaa-test"


def run(app, args):
    isaa, self_agent_config, chains = init_isaa(app, speak_mode=args.speak, calendar=False, ide=True, create=True,
                                                isaa_print=False, python_test=True, init_mem=True, init_pipe=True,
                                                join_now=False,
                                                global_stream_override=False, chain_runner=True)

    isaa.get_context_memory().load_all()
    isaa.agent_collective_senses = True
    isaa.summarization_mode = 1

    self_agent_config.stream = True
    self_agent_config.max_tokens = 4012
    self_agent_config.set_completion_mode('chat').set_mode('execution')
    self_agent_config.set_model_name('gpt-4')

    self_agent_config.stop_sequence = ['\n\n\n', "Execute:", "Observation:", "User:"]

    # new env isaa withs chains
    agents = isaa.config["agents-name-list"]
    task = """Crate an File operating agent. and read the ai_collaboration_extension directory return all files."""

    infos_c = f"List of Avalabel Agents : {isaa.config['agents-name-list']}\n Tools : {list(self_agent_config.tools.keys())}\n" \
              f" Functions : {isaa.scripts.get_scripts_list()}\n executable python dicts : {str(isaa.get_chain())} "

    self_agent_config.get_messages(create=True)
    self_agent_config.add_message("user", task)
    self_agent_config.add_message("system", infos_c)

    self_agent_config.add_message("system", """Use Spawn agent and talk to agent to crat a
     agent for the task with te right specification""")

    coordination = "NO Data"
    for _ in range(3):
        coordination = isaa.run_agent(self_agent_config, '', mode_over_lode='execution')

        print(self_agent_config.last_prompt)

        self_agent_config.add_message("assistant", coordination)
        self_agent_config.add_message("user", input(":"))

    return coordination

def run__(app, args):
    isaa, self_agent_config, chains = init_isaa(app, speak_mode=args.speak, calendar=False, ide=True, create=True,
                                                isaa_print=False, python_test=True, init_mem=True, init_pipe=True,
                                                join_now=False,
                                                global_stream_override=False, chain_runner=True)

    isaa.get_context_memory().load_all()
    isaa.agent_collective_senses = True
    isaa.summarization_mode = 1

    self_agent_config.stream = True
    self_agent_config.max_tokens = 4012
    self_agent_config.set_completion_mode('chat')
    self_agent_config.set_model_name('gpt-4')

    self_agent_config.stop_sequence = ['\n\n\n', "Execute:", "Observation:", "User:"]

    think_agent = isaa.get_agent_config_class("think").set_completion_mode('chat')  # .set_model_name('gpt-4')
    thinkm_agent = isaa.get_agent_config_class("thinkm").set_completion_mode('chat')
    execution_agent = isaa.get_agent_config_class("execution").set_completion_mode('chat')

    execution_agent.stop_sequence = ['\n\n\n', "Execute:", "Observation:", "User:"]

    think_agent.stream = True
    thinkm_agent.stream = True
    execution_agent.stream = True

    # new env isaa withs chains
    agents = isaa.config["agents-name-list"]
    task = """Vervollständige die ai_collaboration_extension in dem du ein chat fester hinzufügst."""

    task = idea_enhancer(isaa, task, self_agent_config, chains, True)

    think_agent.get_messages(create=True)
    think_agent.add_message("assistant", "Final Strategie:\n" + task)

    plan = isaa.run_agent(think_agent, '', mode_over_lode='planing')

    think_agent.add_message("assistant", plan)
    think_agent.add_message("system", "Break the plan into smaller steps if necessary. write the plan and the steps "
                                      "so that it can be solved")

    step_step_plan = isaa.run_agent(think_agent, '')

    extracted_dict = generate_exi_dict(isaa, step_step_plan, True, list(self_agent_config.tools.keys()))

    print("Creating:", extracted_dict)

    if extracted_dict:
        run_chain_in_cmd(isaa, task, chains, extracted_dict, self_agent_config)


def run_(app, args):
    isaa, self_agent_config, chains = init_isaa(app, speak_mode=args.speak, calendar=False, ide=True, create=True,
                                                isaa_print=False, python_test=True, init_mem=False, init_pipe=False,
                                                join_now=False,
                                                global_stream_override=False, chain_runner=False)

    isaa.get_context_memory().load_all()

    isaa.get_agent_config_class("think").set_model_name("gpt-4").stream = True

    tree = IsaaQuestionBinaryTree().deserialize({
        'question': 'Can you complete this task?',
        'left': {
            'question': 'What are the specific steps needed to complete this task? Create a Plan!',
            'left': {
                'question': 'Where can you find the information and resources needed for each step?',
                'left': {
                    'question': 'Identify potential obstacles in obtaining the necessary resources and plan how to overcome them?',
                    'left': None,
                    'right': None
                },
                'right': None
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

    user_imp = """
Hello Isaa main name is Markin Hausmanns I have been working on you for the last 6 years to keep you in this state. I need your help. At the moment I am 20 years old and I graduated from high school a year ago. Now it's 2023 and I've been studying MINT green for a year, which is an Orinetirungsstudium. in this mine an hamen computer science studiren confirmed but also disproved. Because I have found out that the TU Berlin my university at which I am Grade is not right for me on the one hand, the worst is very theoretical and too langsm what is taught does not keep up with today's development in computer science. Fammiler I would also like to go out only in Berlin the rents are too expensive I would like to make many maybe a Auslans study or year. Help me with the decision and with the following necessary organization.
    """
    user_imp = """
Create a user friendly web app First start with an interesting landing page!
     """
    agent_categorize_config: AgentConfig = isaa.get_agent_config_class("categorize")

    agent_categorize_config \
        .set_mode('free') \
        .set_completion_mode('chat') \
        .set_model_name('gpt-3.5-turbo')

    isaa.get_agent_config_class("think").stream = True

    # 'bigscience/bloom' to small
    agent_categorize_config.set_model_name(
        'gpt-3.5-turbo')  # chavinlo/gpt4-x-alpaca # nomic-ai/gpt4all-j # TheBloke/gpt4-alpaca-lora-13B-GPTQ-4bit-128g
    agent_categorize_config.stream = True
    agent_categorize_config.max_tokens = 4012
    agent_categorize_config.set_completion_mode('chat')
    self_agent_config.stream = True
    self_agent_config.max_tokens = 4012
    self_agent_config.set_completion_mode('chat')
    self_agent_config.set_model_name('gpt-4')
    agent_categorize_config.stop_sequence = ['\n\n\n']
    user_imp = isaa.run_agent('thinkm', f"Yur Task is to add information and"
                                        f" specification to as task tah dask is "
                                        f"(writ in coherent text format): " + user_imp + "\n"
                              , mode_over_lode='free')
    print(user_imp)
    # plan, s = isaa.execute_2tree(user_imp, tree, copy.deepcopy(self_agent_config))
    # print(s)
    plan_ = """
Answer 1: As an AI language model, I am not capable of physically completing the task. However, I can provide guidance and suggestions on how to improve the toolbox system.

Answer 2: Here is a plan to improve the toolbox system:

1. User Interface:
   - Conduct user research to understand user needs and preferences.
   - Redesign the interface to be more intuitive and user-friendly.
   - Provide clear instructions and guidance for users.
   - Conduct usability testing to ensure the interface is easy to navigate.

2. Functionality:
   - Conduct a needs assessment to identify the most useful functions for users.
   - Add new features and tools to the system based on user needs.
   - Improve the performance and speed of existing functions.

3. Compatibility:
   - Test the system on different platforms and devices to ensure compatibility.
   - Address any compatibility issues that arise.

4. Security:
   - Implement strong encryption and authentication protocols to protect user data.
   - Regularly update the system to address any security vulnerabilities.

5. Documentation:
   - Create user manuals, tutorials, and online help resources.
   - Ensure the documentation is clear and comprehensive.

Overall, the plan involves conducting research, redesigning the interface, adding new features, testing for compatibility, implementing security measures, and creating documentation.

Answer 3: Resources for each step can be found in various places, such as:
1. User Interface:
   - User research can be conducted through surveys, interviews, and usability testing.
   - Design resources can be found online or through hiring a designer.
   - Instructional design resources can be found online or through hiring an instructional designer.

2. Functionality:
   - Needs assessment resources can be found online or through hiring a consultant.
   - Development resources can be found online or through hiring a developer.

3. Compatibility:
   - Testing resources can be found online or through hiring a testing team.

4. Security:
   - Security resources can be found online or through hiring a security consultant.

5. Documentation:
   - Documentation resources can be found online or through hiring a technical writer.

Answer 4: Potential obstacles in obtaining the necessary resources could include:
- Limited budget for hiring consultants or designers.
- Limited time for conducting research or testing.
- Difficulty finding qualified professionals for certain tasks.
To overcome these obstacles, it may be necessary to prioritize tasks and allocate resources accordingly. It may also be helpful to seek out alternative resources, such as online tutorials or open-source software.
"""
    alive = True
    step = 0
    exi_agent_output = ''
    adjustment = ''
    agent_validation = ''
    for_ex = ''
    do_help = False
    self_agent_config.task_index = 0
    print("Working on medium plan")
    step += 1
    self_agent_config.stop_sequence = ['\n\n\n']
    medium_plan = isaa.run_agent(self_agent_config, f"We ar at step {step} split down the step into smaller "
                                                    f"ones that can be worked on by the execution mode."
                                                    f"return information close to ech other from the plan"
                                                    f" that can be connected to a singed step for step {step}."
                                                    f" The Full Plan '{user_imp}'.Adapt the next steps "
                                                    f"appropriately. current information"
                                                    f" {exi_agent_output + ' ' + agent_validation}."
                                                    f" Add information on how to finish the task"
                                                    f"Return a valid python list ```python\nplan : List[str] ="
                                 , mode_over_lode='planning')

    try:
        self_agent_config.task_list = eval(medium_plan.strip())
        print(self_agent_config.task_list)
        if len(self_agent_config.task_list) == 0:
            self_agent_config.step_between = 'Task don produce the final output'
    except ValueError and SyntaxError:
        self_agent_config.task_list = [user_imp + medium_plan]
    for_ex = medium_plan
    self_agent_config.short_mem.clear_to_collective()
    user_help = ''
    while alive:
        if do_help:
            print("Working on adjustment")
            self_agent_config.stop_sequence = ['\n\n\n', f' {self_agent_config.task_index + 1}',
                                               f'\n{self_agent_config.task_index + 1}']
            self_agent_config.step_between = f"We ar at step " \
                                             f" '{self_agent_config.task_index}'. lased output" \
                                             f" '{exi_agent_output + ' Validation:' + agent_validation}'" \
                                             f" Adapt the plan appropriately. Only Return 1 step at the time"
            adjustment = isaa.run_agent(self_agent_config, '', mode_over_lode='planning')
            self_agent_config.step_between = adjustment
            self_agent_config.short_mem.clear_to_collective()

        self_agent_config.stop_sequence = ['\n\n', 'Execute:', 'Observation:']
        exi_agent_output = isaa.run_agent(self_agent_config, user_help, mode_over_lode='tools')

        print("=" * 20)
        print(self_agent_config.short_mem.text)
        print(f'Step {self_agent_config.task_index}:', Style.Bold(Style.BLUE(exi_agent_output)))
        print("=" * 20)

        if self_agent_config.completion_mode == 'chat':
            key = f"{self_agent_config.name}-{self_agent_config.mode}"
            if key in self_agent_config.messages_sto.keys():
                del self_agent_config.messages_sto[key]

        if 'question:' in exi_agent_output.lower() or 'user:' in exi_agent_output.lower():
            user_help = input("\nUser: ")
        elif 'Observation: Agent stopped due to iteration limit or time limit.' in exi_agent_output:
            user_help = input("User: ")
        else:
            user_help = ''
            self_agent_config.next_task()

        agent_validation = isaa.run_agent(agent_categorize_config,
                                          f"Is this Task {for_ex} completed or on the right way use this information'{self_agent_config.short_mem.text}'\n"
                                          f" Answer Yes if so else A description of what happened wrong\nAnswer:",
                                          mode_over_lode='free')
        if 'yes' in agent_validation.lower():
            do_help = False
        elif 'don' in agent_validation.lower():
            self_agent_config.task_index = 0
            self_agent_config.task_list = []
            do_help = False
            alive = False
        else:
            do_help = True
        print()
        print(f'Task: at {step} , {do_help=}')
        print("=" * 20)
        print('adjustment:', Style.Bold(Style.BLUE(adjustment)))
        print("-" * 20)
        print('exi_agent_output:', Style.Bold(Style.BLUE(exi_agent_output)))
        print("-" * 20)
        print('agent_validation:', Style.Bold(Style.BLUE(agent_validation)))
        print("=" * 20)
        user_val = input("user val :")
        if 'n' == user_val:
            alive = False
        elif user_val.lower() in ['h', 'help']:
            do_help = True
        elif len(user_val) > 5:
            agent_validation = "-by the user :" + user_val
        else:
            do_help = False
            self_agent_config.task_index = 0
            self_agent_config.short_mem.clear_to_collective()

    print("================================")

# if do_plan:
#     print("Working on medium plan")
#     step += 1
#     self_agent_config.stop_sequence = ['\n\n\n', f' {step + 1}', f'\n{step + 1}']
#     medium_plan = isaa.run_agent(self_agent_config, f"We ar at step {step} split down the step into smaller "
#                                                     f"ones that can be worked on by the execution mode."
#                                                     f"return information close to ech other from the plan"
#                                                     f" that can be connected to a singed step for step {step}."
#                                                     f" The Full Plan '{plan}'.Adapt the next steps "
#                                                     f"appropriately. current information"
#                                                     f" {exi_agent_output + ' ' + agent_validation}."
#                                                     f" Add information on how to finish the task"
#                                                     f" return a valid python List[str].\nsteps: list[str] = "
#                                  , mode_over_lode='planning')
#
#     try:
#         self_agent_config.task_list = eval(medium_plan.strip())
#         if len(self_agent_config.task_list) == 0:
#             alive = False
#             self_agent_config.step_between = 'Task don produce the final output'
#     except ValueError and SyntaxError:
#         self_agent_config.task_list = [user_imp + medium_plan]
#     for_ex = medium_plan
