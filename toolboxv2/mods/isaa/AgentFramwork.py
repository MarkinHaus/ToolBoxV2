from typing import Optional, List

from langchain.agents import load_tools, load_huggingface_tool
from langchain.tools import BaseTool

from .Agents import Agent, get_free_agent_data_factory, LLMMode, Capabilities, LLMFunction, ModeController, \
    flatten_reduce_lambda


def get_free_agent(name: str) -> Agent:
    return Agent(
        amd=get_free_agent_data_factory(name, ),  # model="GPT4All-13B-snoozy.ggmlv3.q4_0.bin"
        stream=True
    )


def run_agent_0shot(agent: Agent, user_input: str, persist: Optional[bool] = None, fetch_memory: Optional[bool] = None,
                    isaa=None, **kwargs):
    message = agent.get_llm_message(
        user_input,
        persist=persist,
        fetch_memory=fetch_memory,
        isaa=isaa
    )

    print(message)

    result = agent.run_model(message, **kwargs)

    print(result)

    return result


def run_0shot_to_multi_shot():
    pass


"""Introducing the "Function Caller," a new type of agent that specializes in making function calls to other agents within the system. This agent is highly skilled at navigating complex systems and has an uncanny ability to find the right function call for any given task.
The Function Caller's primary function is to make function calls to other agents within the system, allowing them to work together seamlessly and efficiently. They are experts in understanding the different functions and their capabilities, and can quickly identify the best function call for a given task.
In addition to their technical skills, the Function Caller also has excellent communication and collaboration skills. They are able to effectively communicate with other agents within the system, ensuring that they are working towards the same goal. This makes them an invaluable asset in any team or system.
Overall, the Function Caller is a highly valuable agent who can help teams work more efficiently and effectively. Their ability to make function calls quickly and accurately.



The Prompt Generation Agent is a software application that generates prompts based on user inputs. It takes into account the context of the conversation, the preferences of the user, and any relevant information about the topic at hand to generate prompts that are tailored to the specific needs of the user.
The agent uses natural language processing (NLP) techniques to analyze the text of the conversation and identify key phrases or keywords that can be used as prompts. It also takes into account the context of the conversation, including any previous responses from the user, to generate prompts that are relevant and helpful for the user's needs.
The agent is designed to be flexible and adaptable, allowing it to work with a wide range of topics and applications. It can be used in a variety of settings, such as chatbots, virtual assistants, or other conversational interfaces.
Overall, the Prompt Generation Agent is a powerful tool for generating prompts that are tailored to the specific needs of the


Sure, I can help you with that!

Here's an example of how you might generate instructions for a prompt generation agent:

Step 1: Identify the purpose of the agent
The purpose of the prompt generation agent is to assist users in generating prompts for natural language generation applications.

Step 2: Gather information on the type of prompts that will be generated
The agent will generate prompts for a specific type of application, such as chatbots or virtual assistants.

Step 3: Identify the target audience for the prompts
The agent will generate prompts that are appropriate for the intended audience, whether it's children, teenagers, adults, or seniors.

Step 4: Determine the level of complexity of the prompts
The agent will generate prompts at different levels of complexity, ranging from simple commands to more complex sentences with proper grammar and syntax.

Step 5: Organize the prompts into categories
The agent will organize the prompts into


To extract important information about a subject from text, follow these steps:

1. Read the text carefully and identify the main topic or subject matter of the text.
2. Look for specific keywords or phrases that indicate the importance of the information being presented.
3. Take note of any key points or arguments made in the text.
4. Organize the information into a logical structure, grouping it by topic or theme.
5. Use this organized information to create a summary or overview of the main points of the text.
6. Check for accuracy and completeness of the summary/overview.


To extract important information about a subject from text, you can follow these steps:

1. Read the text carefully and identify the main topic or theme of the article/book/document.
2. Look for headings, subheadings, and other organizational structures that may provide clues to the main points of the text.
3. Skim through the text to get a general overview of the information presented in the document.
4. Use specific keywords or phrases from the text to search for relevant information on the subject.
5. Take note of any statistics, facts, or figures mentioned in the text that may be useful in further research.
6. Organize your notes and extract only the most important information from the text.
7. Double-check your work by verifying the accuracy of the information you have extracted.
"""


def craft_mode(objective, additional_informations: str = "", agent: Optional[Agent] = None, isaa=None,
               mode_prompt: Optional[LLMMode or ModeController] = None,
               mode_describe: Optional[LLMMode or ModeController] = None,
               mode_naming: Optional[LLMMode or ModeController] = None,
               mode_clean_infos: Optional[LLMMode or ModeController] = None):
    system_msg_infos = create_prompt(objective + additional_informations, agent=agent, mode_controller=mode_prompt,
                                     isaa=isaa)

    system_msg = create_prompt(system_msg_infos + "\nReformulate into a concrete statement, instruction", agent=agent,
                               mode_controller=mode_prompt, isaa=isaa)

    description = create_description(f"\nMain objective: {objective}\n Additional informations: {system_msg_infos}",
                                     agent=agent, mode_controller=mode_describe,
                                     isaa=isaa)

    temp_sto_agent_max_token = None
    if agent:
        temp_sto_agent_max_token = agent.max_tokens
        agent.max_tokens = 4

    name = create_naming(objective + additional_informations + description, agent=agent, mode_controller=mode_naming,
                         isaa=isaa)
    if agent:
        agent.max_tokens = temp_sto_agent_max_token

    # name = create_clean_informations(name, agent=agent, mode_controller=mode_clean_infos,
    #                                  isaa=isaa)
    # description = create_clean_informations(description, agent=agent, mode_controller=mode_clean_infos,
    #                                         isaa=isaa)
    # system_msg = create_clean_informations(system_msg, agent=agent, mode_controller=mode_clean_infos,
    #                                        isaa=isaa)

    return LLMMode(
        name=name,
        description=description,
        system_msg=system_msg,
        post_msg="Assistant:"
    )


def crate_llm_function_from_langchain_tools(tool: str or BaseTool or List[str]) -> List[LLMFunction]:
    if isinstance(tool, BaseTool):
        return [LLMFunction(name=tool.name, description=tool.description, parameters=tool.args, function=tool)]

    if isinstance(tool, list):
        pass

    if isinstance(tool, str):
        tool = [tool]

    returning_llm_function_list = []

    for langchain_tool in load_tools(tool):
        returning_llm_function_list.append(LLMFunction(name=langchain_tool.name, description=langchain_tool.description,
                                                       parameters=langchain_tool.args, function=langchain_tool))

    return returning_llm_function_list


def crate_trait(capability: Capabilities, agent: Optional[Agent] = None,
                mode_controller: Optional[LLMMode or ModeController] = None,
                isaa=None):
    if agent is None:
        agent = get_free_agent("Temp-Agent")

    functions_infos = "\n".join([str(functions) for functions in capability.functions])

    return create_clean_informations(f"function descriptions:{functions_infos}\nSummers all function descriptions to a "
                                     f"single Agent trait",
                                     agent=agent, mode_controller=mode_controller,
                                     isaa=isaa)


def create_capability(name, informations, functions: List[LLMFunction],
                      mode_describe: Optional[LLMMode or ModeController] = None, agent=None, isaa=None):
    description = create_description(f"\nMain objective: {name}\n Additional informations: {informations}",
                                     agent=agent, mode_controller=mode_describe,
                                     isaa=isaa)

    trait = crate_trait(Capabilities(name="", description="", trait="", functions=functions))

    return Capabilities(name=name, description=description, trait=trait, functions=functions)


def combine_capabilities(capabilities: List[Capabilities], mode_describe: Optional[LLMMode or ModeController] = None,
                         agent=None, isaa=None):
    all_func = flatten_reduce_lambda([[f for f in fuc.functions] for fuc in capabilities])
    all_trait = [fuc.trait for fuc in capabilities]
    all_description = [fuc.description for fuc in capabilities]
    all_name = [fuc.name for fuc in capabilities]

    temp_capability = create_capability(" ".join(all_name), " ".join(all_description), all_func,
                                        mode_describe=mode_describe, agent=agent, isaa=isaa)

    print(temp_capability)
    print(all_trait)

    return temp_capability


def prefixes_mode_asapt(task: str, agent: Optional[Agent] = None, isaa=None,
                        mode_controller: Optional[LLMMode or ModeController] = None):
    if mode_controller is None:
        mode_controller = LLMMode(
            name="ASAPT-Model",
            description="use a reactive framework to solve a problem",
            system_msg="The Assistant, act in a certain prefix structure. I can use the following "
                       "prefixes:\n======\nASK: In this line the following text should contain a"
                       "question for the user. ask the user only in necessary special situations.\nSPEAK: The "
                       "following text will be spoken.\nTHINK: This text remains hidden. The THINK prefix should be "
                       "used regularly to reflect.\nPLAN: To reflect a plan.\nACTION: The agent has tools that it can "
                       "access. Actions should be described in JSON format.{'Action':'name','Inputs':args}\n======\nNow"
                       " start using the reactive prefix framework on this Task.\n\n",
            post_msg="\nAssistant: ",
            examples=["User: What is the wetter in Berlin?\nAssistant:\n THINK: I need to searcher for live "
                      "informations\nPLAN: first i need to searcher for informations\nACTION: {'Action':'requests',"
                      "'Inputs':'https://www.berlin.de/wetter/'}"]
        )

    result = run_on_mode(task=task, mode_controller=mode_controller, agent=agent, isaa=isaa)

    return result


def create_description(task: str, agent: Optional[Agent] = None, isaa=None,
                       mode_controller: Optional[LLMMode or ModeController] = None):
    if mode_controller is None:
        mode_controller = LLMMode(
            name="DescribeMode",
            description="This LLM mode is designed to generate descriptions for Capabilities based on the input name.",
            system_msg="You are a descriptions creation engine. Create a detailed and concrete description for a given"
                       "user input. User input: ",
            post_msg="\nAssistant:"
        )

    result = run_on_mode(task=task, mode_controller=mode_controller, agent=agent, isaa=isaa)

    return result


def create_prompt(task: str, agent: Optional[Agent] = None, isaa=None,
                  mode_controller: Optional[LLMMode or ModeController] = None):
    if mode_controller is None:
        mode_controller = LLMMode(
            name="CreatePrompt",
            description="This LLM mode is designed to generate Prompts for other Agents based on a Subject.",
            system_msg="You are a specialized instruction-Prompt generator, trained to craft clear and precise "
                       "instructions-Prompts"
                       "based on given information formulate one clear stance!"
                       " Generate instruction for this subject :\n",
            post_msg="\nAssistant:"
        )

    result = run_on_mode(task=task, mode_controller=mode_controller, agent=agent, isaa=isaa)

    return result


def create_clean_informations(task: str, agent: Optional[Agent] = None, isaa=None,
                              mode_controller: Optional[LLMMode or ModeController] = None):
    if mode_controller is None:
        mode_controller = LLMMode(name='Text Extractor', description="Extracting the main information from a text",
                                  system_msg='\n\nTo extract the main information from a text, you can follow these '
                                             'steps:\n\n1. Read through the text carefully: Take your time to read '
                                             'through the text thoroughly. This will help you identify any important '
                                             'information that may be relevant to your analysis.\n2. Look for '
                                             'organizational structures: Check if the text has any organizational '
                                             'structures like headings or subheadings. These can help you organize '
                                             'the main points of the text into a clear and concise summary.\n3. Use '
                                             'knowledge of grammar and syntax: Look for sentences or phrases that are '
                                             'likely to be important or relevant to your analysis. This may include '
                                             'using knowledge of grammar and syntax to identify key phrases or '
                                             'sentences.', post_msg='Assistant:', examples=None)

        LLMMode(name='Text Extractor',
                description='\n\nMain Objective: Extracting the main information from a text involves using various '
                            'techniques to identify and organize the most relevant information within the text. This '
                            'can include reading through the text carefully, looking for organizational structures '
                            'like headings and subheadings, using',
                system_msg='To extract the main information from a text, follow these steps:\n\n1. Read through the '
                           'text carefully\n2. Look for organizational structures\n3. Use knowledge of the text to '
                           'organize the main points\n4. Identify any relevant subtopics\n5. Organize the text into a '
                           'summary\n6. Check for accuracy and completeness\n7. Revise the summary for '
                           'clarity\n',
                post_msg='Assistant:', examples=None)

    result = run_on_mode(task=task, mode_controller=mode_controller, agent=agent, isaa=isaa)

    return result


def create_naming(task: str, agent: Optional[Agent] = None, isaa=None,
                  mode_controller: Optional[LLMMode or ModeController] = None):
    if mode_controller is None:
        mode_controller = LLMMode(name='NamingGenerator',
                                  description='To generate a descriptive name for the given text',
                                  system_msg='You ar a naming Generator To find a name for a given input text, you can follow these steps:\n\n1. Grasp the Main '
                                             'Ideea of the Text\n2. Combine and schorten them\n3. Write the fineal '
                                             'Name\n\nExample:\n\nLet\'s say you have a text that says "The quick brown fox jumps over '
                                             'the lazy dog". To rename it as "Jumpiung Fox"\ninput text to name : "',
                                  post_msg='"\nAssistant:', examples=None)

    result = run_on_mode(task=task, mode_controller=mode_controller, agent=agent, isaa=isaa)

    return result


def create_pkcloudtest(task: str, agent: Optional[Agent] = None, isaa=None,
                       mode_controller: Optional[LLMMode or ModeController] = None):
    if mode_controller is None:
        mode_controller = LLMMode(
            name='Divide and Conquer Evaluator',
            description='Plan and implement a divide and conquer approach to evaluate a complex problem',
            system_msg='Here are the steps to plan a divide and conquer evaluation loop:\n\n1. Break down the problem into smaller sub-problems that can be solved independently.\n2. Solve each sub-problem recursively using the same divide and conquer approach.\n3. Combine the solutions to the sub-problems to form the solution for the original problem.\n4. Analyze the performance and correctness of the overall solution.\n5. Identify any improvements that can be made through additional decomposition and recursion.',
            post_msg='Assistant:',
            examples=[
                'Break down handwriting recognition into character recognition, segmentation, and language modeling.',
                'Divide a sorting algorithm by breaking the data into chunks that can be sorted recursively.'
            ]
        )

    result = run_on_mode(task=task, mode_controller=mode_controller, agent=agent, isaa=isaa)

    return result


def create_llamatp(task: str, agent: Optional[Agent] = None, isaa=None,
                   mode_controller: Optional[LLMMode or ModeController] = None):
    if mode_controller is None:
        mode_controller = LLMMode(name='Task Planner',
                                  description='Plan a task for a divide and conquer evaluation loop',
                                  system_msg='To plan a task for a divide and conquer evaluation loop, follow these steps:\n\n1. Define the task: Clearly define the task you want to evaluate using a divide and conquer approach.\n2. Divide the task into smaller subtasks: Break down the task into smaller, more manageable subtasks that can be evaluated independently.\n3. Conquer the subtasks: Evaluate each subtask and determine if it is feasible to solve. If it is not feasible, break it down further into smaller subsubtasks.\n4. Repeat the process: Continue dividing and conquering subtasks until you have a feasible solution for the entire task.\n5. Evaluate the solution: Evaluate the solution to ensure it meets the desired criteria. If it does not, refine the solution and repeat the evaluation process.\n6. Summarize the results: Summarize the results of the evaluation, including the feasible solution and any refinements made to the solution.',
                                  post_msg='Assistant:',
                                  examples=None)

    result = run_on_mode(task=task, mode_controller=mode_controller, agent=agent, isaa=isaa)

    return result


def run_on_mode(task: str, mode_controller: LLMMode or ModeController, agent: Optional[Agent] = None, isaa=None,
                persist=False, fetch_memory=True, **kwargs):
    if agent is None:
        agent = get_free_agent("Temp-Agent")

    if isinstance(mode_controller, LLMMode):
        agent.mode = mode_controller

    if isinstance(mode_controller, ModeController):
        result = mode_controller.run_agent_on_mode(task, agent, persist=persist, fetch_memory=fetch_memory,
                                                   isaa=isaa, **kwargs)
    else:
        llm_message = agent.get_llm_message(task, persist=persist, fetch_memory=fetch_memory, isaa=isaa)
        result = agent.run_model(llm_message, **kwargs)

    return result
