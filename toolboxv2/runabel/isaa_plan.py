"""Console script for toolboxv2. Isaa CMD Tool"""

from toolboxv2 import Style, Spinner
from toolboxv2.mods.isaa import AgentChain, IsaaQuestionBinaryTree, AgentConfig
from toolboxv2.utils.isaa_util import init_isaa, extract_dict_from_string

NAME = "isaa-plan"


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

    task = """Hello Isaa main name is Markin Hausmanns I have been working on you for the last 6 years to keep you in
    this state. I need your help. At the moment I am 20 years old and I graduated from high school a year ago. Now
    it's 2023 and I've been studying MINT green for a year, which is an Orinetirungsstudium. in this mine an hamen
    computer science studiren confirmed but also disproved. Because I have found out that the TU Berlin my university
    at which I am Grade is not right for me on the one hand, the worst is very theoretical and too langsm what is
    taught does not keep up with today's development in computer science. Fammiler I would also like to go out only
    in Berlin the rents are too expensive I would like to make many maybe a Auslans study or year. Help me with the
    decision and with the following necessary organization. Crate a live plan for the next 2 years with all nessesery infortion """

    construct_context = isaa.run_agent("thinkm",
                                       f"Based on the task entered by the user, I will now gather additional context and collect "
                                       f"relevant information from my memory. and I will finde information that could help"
                                       f" complete the task. Task: {task}", mode_over_lode='construct_context')

    sim = isaa.run_agent("search",
                         f"Find out how similar the task is to the task you have already completed. if so, create a "
                         f"outline to successfully complete this task.{task} , Context : {construct_context}")

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
    for _ in range(3):
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
    if not extracted_dict:
        return
    if not ('name' in extracted_dict.keys() and 'tasks' in extracted_dict.keys()):
        return

    chain_instance: AgentChain = isaa.get_chain()
    chain_instance.load_from_file()

    chain_instance.add(extracted_dict['name'], extracted_dict['tasks'])

    evaluation, chain_ret = isaa.execute_thought_chain(task, chain_instance.get(extracted_dict['name']),
                                                       self_agent_config)

    print(evaluation, len(chain_ret), len(extracted_dict['tasks']) == len(chain_ret))

    for task in chain_ret:
        print(task[1])


