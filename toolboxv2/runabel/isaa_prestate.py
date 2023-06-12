"""Console script for toolboxv2. Isaa CMD Tool"""

from toolboxv2 import Style, Spinner
from toolboxv2.mods.isaa import AgentChain, IsaaQuestionBinaryTree, AgentConfig
from toolboxv2.utils.isaa_util import init_isaa, extract_dict_from_string, split_todo_list

NAME = "isaa-s-auto"


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

    isaa.get_agent_config_class("think").stream = True
    isaa.get_agent_config_class("thinkm").stream = True

    task_ = """Hello Isaa main name is Markin Hausmanns I have been working on you for the last 6 years to keep you in
    this state. I need your help. At the moment I am 20 years old and I graduated from high school a year ago. Now
    it's 2023 and I've been studying MINT green for a year, which is an Orinetirungsstudium. in this mine an hamen
    computer science studiren confirmed but also disproved. Because I have found out that the TU Berlin my university
    at which I am Grade is not right for me on the one hand, the worst is very theoretical and too langsm what is
    taught does not keep up with today's development in computer science. Fammiler I would also like to go out only
    in Berlin the rents are too expensive I would like to make many maybe a Auslans study or year. Help me with the
    decision and with the following necessary organization. Crate a live plan for the next 2 years with all nessesery infortion """

    task = """
Create a plan for a fully automatic cigarette rolling machine that uses a standerd draher as the core. Create a functional and explanatory plan
"""

    construct_context = isaa.run_agent("thinkm",
                                       f"Based on the task entered by the user, I will now gather additional context and collect "
                                       f"relevant information from my memory. and I will finde information that could help"
                                       f" complete the task. Task: {task} Helpfully information's:", mode_over_lode='construct_context')

    sim = isaa.run_agent("search",
                         f"Find out how similar the task is to the task you have already completed. if so, create a "
                         f"outline to successfully complete this task.{task} , Context : {construct_context}",
                         mode_over_lode='execution')

    komplex = isaa.run_agent("think",
                             f"Find out how komplex the task is. Task:{task} , Context : {construct_context},"
                             f" Addition informations {sim}",
                             mode_over_lode='komplex')

    agent_neadet = isaa.run_agent("think",
                                  f"Find out what agent you need and what tools dise should"
                                  f" have to successfully complete the task. Task:{task} , Context : {construct_context},"
                                  f" Addition informations {sim} {komplex}",
                                  mode_over_lode='free')

    chains_neadet = isaa.run_agent("think",
                                   f"Find out which combination of these tasks evaluation skills is helpful for this task.. Task:{task} , Context : {construct_context},"
                                   f" Addition informations {sim} {komplex}\n tasks evaluation skills: {str(chains)}",
                                   mode_over_lode='free')
    self_agent_config.stop_sequence, sto = ['\n\n\n'], self_agent_config.stop_sequence
    analysis = isaa.run_agent(self_agent_config,
                              f"I will now analyze the task received to understand the intentions and goals. This "
                              f"includes understanding the details of the task as well as all relevant contextual "
                              f"information. Task: {task}, Context : {construct_context},"
                              f" Application reports {agent_neadet} {chains_neadet}"
                              , mode_over_lode='Analysis')

    validation = ''
    extracted_dict = {}
    free_run = False
    coordination = ''
    for _ in range(2):
        coordination = isaa.run_agent(self_agent_config,
                                      """ I now coordinate the work of the various task handlers to ensure that the right tasks are routed to the right handlers. This ensures that the tasks are executed efficiently and effectively.
For this I return a valid python dict with the individual steps to be taken.

Structure :
{
"name":'<overword for the task>
"tasks":[
<list of tasks>
]
}
<caption for the task> : str
<list of tasks> : List[dict]
task: dict = {
      "use": str, tool | agent | fuction
      "name": str, |name of the tool agent or function|
      "args":str, |argse $user-input
      "return": str, | return key

      optionals use when dealing with large amount of text
"text-splitter": int | text max size
"chuck-run": str, | on return key
}

Example Tasks :

{ # I want to search for information's so i use the search agent, the information is stored in the $infos variable
      }, "use": "tool",
      "name": "search",
      "args": "search information on $user-input",
      "return":"$infos"
    },

{ # I want to call other task chain
      }, "use": "tool",
      "name": "execute-chain",
      "args": "write_python : $requirements", # i can run """+str(chains)+"""
      "return":"$infos"
    },

{ # Now i have mutabel information so i want to summarize and get a report, $infos-0 .. and return a report in
# Notice i used the '$' variable in the args section.
      }, "use": "agent",
      }, { "name": "think",
      { "args": "summarize and report information on topic $infos-0 \nPro page $infos-1 \nCon page $infos-2 \n\ndiffernzirte",
      "return":"$report"
    }

{ # We need to get information from a file so i use.. the file is large so i can use text-splitter.
# ! Notice the $file-name variable must be returned before thes call!
# a safer way of action is tho direct wirt the file name
      }, "use": { "tool",
      }, { "name": "read",
      { "args": "$file-name",
      "return": "$file-content",
      "text-splitter": 10000
    },
    {
      }, "use": { "agent",
      }, { "name": "think",
      { "args": "Act as an summary expert your specialties are writing summary. you are known to think in small and detailed steps to get the right result. Your task : $file-content",
      "chuck-run": "$file-content",
      "return": "$summary"
    }
""" + f"Create a task list for : Task: {task} \nAnalysis :{analysis}\n{validation}", mode_over_lode='Coordination')
        extracted_dict = extract_dict_from_string(coordination)
        if isinstance(extracted_dict, dict):
            if 'name' in extracted_dict.keys() and 'tasks' in extracted_dict.keys():
                break
        if extracted_dict is not None:
            validation = 'Validation: The dictionary is not valid ' + str(extracted_dict)
        print(extracted_dict)
    self_agent_config.stop_sequence = sto
    task = coordination
    if not extracted_dict:
        free_run = True

    if not free_run:

        if 'name' in extracted_dict.keys() and 'tasks' in extracted_dict.keys():

            chains.add(extracted_dict['name'], extracted_dict['tasks'])

            evaluation, chain_ret = isaa.execute_thought_chain(task, chains.get(extracted_dict['name']),
                                                               self_agent_config)

            print(evaluation, len(chain_ret), len(extracted_dict['tasks']) == len(chain_ret))

            for task in chain_ret:
                print(task[1])

            task = chain_ret[int(input("Chuse outut to work on :"))][1]

    isaa.summarization_mode = 2
    self_agent_config.stream = True
    self_agent_config.max_tokens = 4012
    self_agent_config.set_completion_mode('chat')
    self_agent_config.set_model_name('gpt-4')
    self_agent_config.stop_sequence = ['\n\n\n', "Execute:", "Observation:", "User:"]
    isaa.get_agent_config_class("think").stream = True
    isaa.get_agent_config_class("thinkm").stream = True
    self_agent_config.short_mem.memory_data = []
    self_agent_config.observe_mem.memory_data = []
    self_agent_config.edit_text.memory_data = []
    chains.load_from_file()
    # new env isaa withs chains

    agents = isaa.config["agents-name-lsit"]
    task = """
    Create a beautiful html index page thats welcoms the user wit an cool animation to simple"""
    alive = True
    new_agent = isaa.config["agents-name-lsit"][-1]
    if len(agents) != 3:
        for agent_name in agents[3:]:
            isaa.get_agent_config_class(agent_name)
    while alive:
        env_text = f"""Welcome, you are in a hostile environment, your name is isaa.
    you have several basic skills 1. creating agents 2. creating some agents 3. using skills, agents and tools

    you have created {len(agents)}agents so far these are : {agents}.
    you have learned {len(chains.chains.keys())}skills so far these are : {str(chains)}.

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
        user_val = input("")
        if user_val == "n":
            alive = False
        elif len(user_val) > 3:
            task += "User: " + user_val

"""
It seems like you're looking for information on creating a virtual agent development lab and developing tools for agent testing. Here's a summary of the process and the features needed for a virtual agent development lab:

1. Integrated Development Environment (IDE): The lab should have an IDE that allows developers to write, test, and debug code for their virtual agents.
2. Agent Creation Tools: The lab should have tools that allow developers to create and customize virtual agents, including their appearance, behavior, and communication capabilities.
3. Agent Testing and Debugging Tools: The lab should have tools that allow developers to test and debug their virtual agents, including simulating different scenarios and environments.
4. Agent Deployment and Management Tools: The lab should have tools that allow developers to deploy and manage their virtual agents, including loading and saving agents, and monitoring their performance.
5. Collaboration and Sharing Tools: The lab should have tools that allow developers to collaborate and share their virtual agents with others, including version control and sharing features.
6. User-Friendly Interface: The lab should have a user-friendly interface that is easy to navigate and use, even for non-technical users.
7. Documentation and Support: The lab should have comprehensive documentation and support resources, including tutorials, FAQs, and user forums.

To create a virtual agent development lab, follow these steps:

1. Define the purpose and scope of the lab.
2. Choose the development platform.
3. Develop the IDE.
4. Create agent creation tools.
5. Develop agent testing and debugging tools.
6. Create agent deployment and management tools.
7. Develop collaboration and sharing tools.
8. Design a user-friendly interface.
9. Develop documentation and support resources.
10. Test and refine the lab.

Developing tools for agent testing involves creating a system that can simulate different scenarios and environments, identify and fix errors and bugs, and provide feedback to developers. This will help ensure that the virtual agents are functioning correctly and efficiently.
"""



