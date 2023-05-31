"""Console script for toolboxv2. Isaa CMD Tool"""

from toolboxv2 import Style, Spinner
from toolboxv2.mods.isaa import AgentChain, IsaaQuestionBinaryTree, AgentConfig
from toolboxv2.utils.isaa_util import init_isaa, extract_dict_from_string, split_todo_list

NAME = "isaa-test"


def run(app, args):
    isaa, self_agent_config, chains = init_isaa(app, speak_mode=args.speak, calendar=False, ide=True, create=True,
                                                isaa_print=False, python_test=True, init_mem=False, init_pipe=False,
                                                join_now=False,
                                                global_stream_override=False, chain_runner=False)

    isaa.get_context_memory().load_all()

    self_agent_config.stream = True
    self_agent_config.max_tokens = 4012
    self_agent_config.set_completion_mode('chat')
    self_agent_config.set_model_name('gpt-4')

    self_agent_config.stop_sequence = ['\n\n\n', "Execute:", "Observation:", "User:"]

    isaa.get_agent_config_class("think").stream = True
    isaa.get_agent_config_class("thinkm").stream = True

    # new env isaa withs chains
    print(isaa.config["agents-name-lsit"])
    skills = []
    agents = isaa.config["agents-name-lsit"]
    task = """To create a comprehensive life plan for the next two years, including a gap year in Berlin, we need to consider several factors such as academic goals, personal development, and financial planning. Here is a suggested life plan:

1. Academic Goals:
- Research and identify the specific computer science courses and programs offered at TU Berlin or other universities in Berlin.
- Determine the prerequisites and admission requirements for these programs.
- Plan to complete any necessary prerequisite courses or exams during the first year.
- Apply for the chosen computer science program at TU Berlin or another university in Berlin during the second year.

2. Personal Development:
- Utilize the gap year to gain practical experience in the field of computer science through internships, part-time jobs, or volunteering.
- Attend workshops, conferences, and networking events related to computer science to expand your knowledge and connections in the industry.
- Develop soft skills such as communication, teamwork, and problem-solving through participation in clubs, organizations, or online courses.
- Learn German to enhance your cultural experience in Berlin and improve your chances of securing internships or job opportunities.

3. Financial Planning:
- Research and apply for scholarships, grants, or financial aid programs available for international students studying in Berlin.
- Create a budget for living expenses, tuition fees, and other costs associated with studying and living in Berlin.
- Explore part-time job opportunities or freelance work to supplement your income during your gap year and while studying.
- Save money during the first year to cover expenses during the second year of the life plan.

4. Travel Plans:
- Research and plan trips to explore Germany and other European countries during your gap year and study breaks.
- Consider joining a travel group or connecting with other students to share travel experiences and reduce costs.

5. Health and Well-being:
- Maintain a balanced lifestyle by incorporating regular exercise, a healthy diet, and sufficient sleep.
- Seek out social opportunities to make friends and build a support network in Berlin.
- Utilize university resources such as counseling services or mental health support if needed.

By following this life plan, you can make the most of your gap year in Berlin and set yourself up for success in your computer science studies at TU Berlin or another university in the city."""
    alive = True
    new_agent = isaa.config["agents-name-lsit"][-1]
    if len(agents) != 3:
        for agent_name in agents[3:]:
            isaa.get_agent_config_class(agent_name)
    while alive:
        env_text = f"""Welcome, you are in a hostile environment, your name is isaa.
you have several basic skills 1. creating agents 2. creating some agents 3. knowing agnts tools and functions to learn skills 4. using skills.

you have created {len(agents)}agents so far these are : {agents}.
you have learned {len(skills)}skills so far these are : {skills}.

use your basic functions with the agent and skills to complete a task.

for your further support you have a python environment at your disposal. write python code to access it.
if you have no ather wy then to ask for help write Question: 'your question'\nUser:

Task : {task}"""

        sim = isaa.run_agent(self_agent_config, env_text, mode_over_lode='execution')
        task += str(sim)

        if "user:" in sim.lower():
            print("USER QUESTION")
            task += input()

        print(isaa.config["agents-name-lsit"])
        agents = isaa.config["agents-name-lsit"]

        if new_agent != isaa.config["agents-name-lsit"][-1]:
            new_agent = isaa.config["agents-name-lsit"][-1]
            isaa.get_agent_config_class(new_agent).save_to_file()

        print(split_todo_list(sim))

        if input(""):
            alive = False




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


"""

The task was to create a user-friendly web app with an interesting email box page. The agent was instructed to consider design, functionality, security, compatibility, and user feedback. The agent followed a step-by-step approach to design a visually appealing and easy-to-navigate email box page, implement functionality for composing, sending, and receiving emails, ensure security using encryption, test the page on different browsers and devices, and gather user feedback to improve the user experience. It is unclear from the given information whether the task was executed successfully. [[{'use': 'agent', 'name': 'think', 'args': 'Design a visually appealing and easy to navigate email box page with a color scheme that is easy on the eyes and a clean, organized layout.', 'return': '$design'}, "Sure, let's break it down into steps to design a visually appealing and easy to navigate email box page:\n\n1. Choose a color scheme that is easy on the eyes: Select colors that are not too bright or too dark, and that complement each other well. For example, a light blue background with dark blue text and white accents can be a good choice.\n\n2. Create a clean and organized layout: Use a grid system to organize the different elements of the email box page, such as the inbox, sent messages, and compose button. Make sure there is enough white space between elements to avoid clutter.\n\n3. Use clear and easy-to-read fonts: Choose a font that is easy to read, such as Arial or Helvetica, and use a consistent font size throughout the page.\n\n4. Add visual elements to enhance the design: Use icons or images to help users quickly identify different sections of the email box page, such as the inbox or compose button.\n\n5. Make sure the page is easy to navigate: Use clear and descriptive labels for buttons and links, and make sure the navigation is intuitive and easy to use.\n\nBy following these steps, you can design a visually appealing and easy to navigate email box page that is easy on the eyes and has a clean, organized layout."], [{'use': 'agent', 'name': 'think', 'args': 'Implement functionality for composing, sending, and receiving emails, as well as features such as search, filters, and labels to help users organize their emails.', 'return': '$functionality'}, 'Sure, here are some steps to implement the functionality for composing, sending, and receiving emails, as well as search, filters, and labels:\n\n1. Create a form for composing emails that includes fields for the recipient, subject, and message body.\n2. Implement a send button that sends the email to the recipient.\n3. Create a page for displaying received emails, with a list of emails and a preview of each email.\n4. Implement a feature to filter emails by sender, recipient, subject, or date.\n5. Create a system for adding labels to emails, such as "important" or "unread", and allow users to filter emails by label.\n6. Implement a search feature that allows users to search for emails by keyword or phrase.\n\nThese steps should help you implement the necessary functionality for a user-friendly email box page.'], [{'use': 'agent', 'name': 'think', 'args': 'Ensure the email box page is secure by using encryption to protect user data and implementing measures to prevent unauthorized access.', 'return': '$security'}, "Here are the steps to ensure the email box page is secure:\n\n1. Use HTTPS: Use HTTPS instead of HTTP to encrypt data in transit between the user's browser and the server.\n\n2. Use encryption: Use encryption to protect user data at rest on the server. This can be achieved by encrypting the database or using file-level encryption.\n\n3. Implement authentication: Implement authentication to ensure that only authorized users can access the email box page. This can be achieved by requiring users to log in with a username and password.\n\n4. Implement access controls: Implement access controls to ensure that users can only access the emails they are authorized to see. This can be achieved by assigning roles and permissions to users.\n\n5. Implement logging and monitoring: Implement logging and monitoring to detect and respond to security incidents. This can be achieved by logging user activity and monitoring the system for suspicious behavior.\n\n6. Regularly update software: Regularly update the software used to run the email box page to ensure that security vulnerabilities are patched."], [{'use': 'agent', 'name': 'think', 'args': 'Test the email box page on different browsers and devices to ensure compatibility and a smooth user experience for all users.', 'return': '$compatibility'}, '1. Identify the different browsers and devices that the email box page needs to be tested on.\n2. Set up testing environments for each of the identified browsers and devices.\n3. Test the email box page on each of the identified browsers and devices to ensure compatibility and a smooth user experience.\n4. Record any issues or bugs encountered during testing.\n5. Address any issues or bugs found during testing to ensure the email box page works well for all users.'], [{'use': 'agent', 'name': 'think', 'args': 'Gather user feedback and make changes based on their suggestions to create a better user experience.', 'return': '$user_feedback'}, 'Sure, here are the step-by-step instructions for gathering user feedback and making changes based on their suggestions to create a better user experience:\n\n1. Provide a feedback form: Create a feedback form that users can fill out to provide their suggestions and feedback. The form should be easy to use and accessible from the email box page.\n\n2. Encourage feedback: Encourage users to provide feedback by promoting the feedback form on the email box page and through other channels such as social media or email newsletters.\n\n3. Analyze feedback: Review the feedback received and identify common themes or issues that users are experiencing. Categorize the feedback into different areas such as design, functionality, or security.\n\n4. Prioritize changes: Prioritize the changes that need to be made based on the feedback received. Focus on the most critical issues first and work your way down the list.\n\n5. Implement changes: Make the necessary changes to the email box page based on the feedback received. Test the changes thoroughly to ensure they are working as intended.\n\n6. Communicate changes: Communicate the changes made to the email box page to users. Let them know that their feedback was heard and that changes were made based on their suggestions.\n\n7. Continue to gather feedback: Keep the feedback form accessible and continue to encourage users to provide feedback. Use the feedback received to make ongoing improvements to the email box page.']]
"""
