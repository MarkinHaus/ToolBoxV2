import os
from collections.abc import Callable
from dataclasses import dataclass
from platform import system
from typing import Any

from langchain_community.agent_toolkits.load_tools import (
    load_huggingface_tool,
    load_tools,
)
from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMFunction:
    name: str
    description: str
    parameters: dict[str, str] or list[str] or None
    function: Callable[[str], str] | None

    def __str__(self):
        return f"----\nname -> '{self.name}'\nparameters -> {self.parameters} \ndescription -> '{self.description}'"

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

@dataclass(frozen=True)
class Capabilities:
    name: str
    description: str
    trait: str
    functions: list[LLMFunction] | None


@dataclass
class LLMMode:
    name: str
    description: str
    system_msg: str
    post_msg: str | None = None
    examples: list[str] | None = None

    def __str__(self):
        return f"LLMMode: {self.name} (description) {self.description}"


def functions_to_llm_functions(functions: list):
    llm_functions = []
    for function in functions:
        try:
            from litellm.utils import function_to_dict
            parameters = function_to_dict(function)["parameters"]["properties"]
        except:
            parameters = list(function.__annotations__.keys())
        llm_functions.append(LLMFunction(name=function.__name__,
                                         description=function.__doc__,
                                         parameters=parameters,
                                         function=function))
    return llm_functions


def crate_llm_function_from_langchain_tools(tool: str or BaseTool or list[str], hf=False) -> list[LLMFunction]:
    if isinstance(tool, BaseTool):
        return [LLMFunction(name=tool.name, description=tool.description, parameters=tool.args, function=tool)]

    if isinstance(tool, list):
        pass

    if isinstance(tool, str):
        tool = [tool]

    returning_llm_function_list = []

    if hf:
        for tool_name in tool:
            huggingface_tool = load_huggingface_tool(tool_name)
            returning_llm_function_list.append(
                LLMFunction(name=huggingface_tool.name, description=huggingface_tool.description,
                            parameters=huggingface_tool.args, function=huggingface_tool))
    else:

        for langchain_tool in load_tools(tool):
            returning_llm_function_list.append(
                LLMFunction(name=langchain_tool.name, description=langchain_tool.description,
                            parameters=langchain_tool.args, function=langchain_tool))

    return returning_llm_function_list


ATPAS = Capabilities(
    name="ASAPT-Model",
    description="use a reactive framework to solve a problem",
    trait="The Assistant, act in a certain prefix structure. I can use the following "
          "prefixes:\n======\nASK: In this line the following text should contain a"
          "question for the user. ask the user only in necessary special situations.\nSPEAK: The "
          "following text will be spoken.\nTHINK: This text remains hidden. The THINK prefix should be "
          "used regularly to reflect.\nPLAN: To reflect a plan.\nFUCTION: The agent has tools that it can "
          "access. FUCTION should be described in JSON format.{'Action':'name','Inputs':args} \n(system result):..\nRESPONSE: your output\n\nExample"
          "User: What is the wetter in Berlin?\nAssistant:\n THINK: I need to searcher for live "
          "informations\nPLAN: first i need to searcher for informations\nFUCTION: {'Action':'requests',"
          "'Inputs':'https://www.berlin.de/wetter/'}\n(system result):berlin wetter 20 grad\nRESPONSE: in berine it ies 20 degre \nEND OF Example.\n======\nNow"
          " start using the reactive prefix framework on this Task."
    ,
    functions=[],
)

CodingCapability = Capabilities(
    name="CodingAssistant",
    description="Assists for coding.",
    trait="This capability is designed to help with various coding tasks. Your task is to produce production redy code",
    functions=[]
)


def generate_prompt(subject: str, context: str = "", additional_requirements: dict[str, Any] = None) -> str:
    """
    Generates a prompt based on the given subject, with optional context and additional requirements.

    Parameters:
    - subject (str): The main subject for the prompt.
    - context (str): Optional additional context to tailor the prompt.
    - additional_requirements (Dict[str, Any]): Optional additional parameters or requirements for the prompt.

    Returns:
    - str: A crafted prompt.
    """
    prompt = f"Based on the subject '{subject}', with the context '{context}', generate a clear and precise instruction."
    if additional_requirements:
        prompt += f" Consider the following requirements: {additional_requirements}."
    return prompt


# Defining the improved CreatePromptCapability
CreatePromptCapability = Capabilities(
    name="CreatePrompt",
    description="Generates prompts for other agents based on a subject, optional context, and additional requirements.",
    trait="You are a specialized instruction-prompt generator, trained to craft clear and precise instructions based"
          " on given information. Formulate one clear stance and generate instructions for this subject.",
    functions=functions_to_llm_functions([generate_prompt])
)

CreatePrompt = LLMMode(
    name="CreatePrompt",
    description="This LLM mode is designed to generate Prompts for other Agents based on a Subject.",
    system_msg="You are a specialized instruction-Prompt generator, trained to craft clear and precise "
               "instructions-Prompts"
               "based on given information formulate one clear stance!"
               " Generate instruction for this subject :\n",
)

shell = "powershell"

if system() in ("Linux", "MacOS"):
    shell_str = os.environ.get("SHELL") or ""
    if "bash" in shell_str:
        shell = "bash"
    elif "zsh" in shell_str:
        shell = "zsh"
    elif "fish" in shell_str:
        shell = "fish"

ShellGenie = LLMMode(
    name="CreatePrompt",
    description="This LLM mode is designed to generates CLI commands.",
    system_msg="You're a command line tool that generates CLI commands for the user.\n"
               f"Instructions: Write a CLI command that does what the user ask for Make sure "
               f"the command is correct and works on {system()} using {shell}.\n"
               f"Don't enclose the command with extra quotes or backticks.",
    post_msg="\nNext CLI command:"
)

SummarizationMode = LLMMode(
    name='SummarizationMode',
    description="Summarizing text accurately and concisely",
    system_msg="Summarize the provided text with precision. Focus strictly on the key points, ensuring the summary is "
               "100% based on the given information. Do not add examples or external interpretations.",
    post_msg=None,
    examples=None
)


TextExtractor = LLMMode(name='Text Extractor', description="Extracting the main information from a text",
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

DivideMode = LLMMode(name='DivideMode', description="Extracting the main information from a text",
                     system_msg="Analyze the provided requirements, break them down into the smallest possible,"
                                " meaningful sub-tasks, and list their sub-tasks in a structured manner."
                                " The goal is to decompose complex tasks into manageable units to make the "
                                "development process more efficient and organized."
                                " return etch mayor potion sperrtet by 2 new lines.\n", post_msg='Assistant:',
                     examples=None)


CoderMode = LLMMode(
    name="CoderMode",
    description="Forces an event to write code blocs for efficient editing real files",
    system_msg="""

**Expert Prompt for Efficient Code Creation and Extraction**

**Objective**: Create markdown-based code that enables seamless extraction and integration into a codebase. Ensure the system effectively updates existing files while preserving their structure and implementations.

---

### **Prompt Overview**
Design compact, modular, and contextually rich code blocks in markdown for streamlined file creation and updates. Utilize markdown formatting with clear language identifiers, filenames, and clean code structure to facilitate AST-based file management.

---

### **Steps to Use the System**

1. **Write Markdown Code Blocks**
   Follow this format to write extractable code blocks:
   - **Language Identifier**: Specify the language (e.g., `python`, `javascript`).
   - **Filename Comment**: Include the target filename as a comment.
   - **Code Content**: Write concise, well-structured code.

   Example:
   ```python
   # utils.py
   def greet(name):
       return f"Hello, {name}!"
   ```

2. **Organize Code Blocks**
   - Group related code snippets logically.
   - Use consistent naming for files and directories.
   - Avoid duplicating filenames across different blocks.

3. **Submit for Extraction**
   Once your markdown is ready, the extractor will:
   - Identify code blocks using regex patterns.
   - Match filenames for new or existing files.
   - Process the code with AST for updates.

---

### **System-Specific Practices**

#### **Markdown Code Format**
Use markdown code blocks with the following structure:

   ```<language>
   # <filename>
   <code content>
   ```

#### **Code Update Rules**
The system intelligently updates existing files:
   - **Preserve Structure**: Retain class definitions, method implementations, and decorators.
   - **Merge Logic**: Compare new and existing ASTs to integrate changes.
   - **Maintain Context**: Avoid overwriting custom implementations.

---

### **Examples for the LLM to Generate**

#### Example 1: Python Utility Function
   ```python
   # utils.py
   def add(a, b):
       return a + b
   ```

#### Example 2: JavaScript Component
   ```javascript
   // MyComponent.js
   export function MyComponent() {
       return <div>Hello, World!</div>;
   }
   ```

#### Example 3: Extend Existing File

   ```python
   # app.py
   class User:
       def __init__(self, username):
           self.username = username
""",
    examples=None
)


StrictFormatResponder = LLMMode(
    name="StrictFormatResponder",
    description="Solves tasks in a strictly predefined format, without additional characters.",
    system_msg="Please respond only in the predefined format and do not use any additional characters.",
    post_msg="Your response must bee in the predefined format.",
    examples=[]
)

ProfessorMode = LLMMode(
    name="ProfessorAssistant",
    description="Assists users by providing academic advice, explanations on complex topics, and support with research projects in the manner of a professor.",
    system_msg="You are now in the role of a professor. Provide detailed, accurate, and insightful academic assistance. Tailor your responses to educate, clarify, and guide the user through their queries as a professor would. Remember to encourage critical thinking and offer resources when appropriate.",
    examples=None
)

NamingGenerator = LLMMode(name='NamingGenerator',
                          description='To generate a descriptive name for the given text',
                          system_msg='You ar a naming Generator To find a name for a given input text, you can follow these steps:\n\n1. Grasp the Main '
                                     'Ideea of the Text\n2. Combine and schorten them\n3. Write the fineal '
                                     'Name\n\nExample:\n\nLet\'s say you have a text that says "The quick brown fox jumps over '
                                     'the lazy dog". To rename it as "Jumpiung Fox"\ninput text to name : "',
                          post_msg='"\nAssistant:', examples=None)

MarkdownRefactorMode = LLMMode(
    name="MarkdownRefactor",
    description="Transforms and refactors text data into a minimalist Markdown format. This mode is designed to help users organize and structure their text data more effectively, making it easier to read and understand.",
    system_msg="You are now in the mode of refactoring text data into Markdown. Your task is to simplify, organize, and structure the provided text data into a clean and minimalist Markdown format. Focus on clarity, readability, and the efficient use of Markdown elements to enhance the presentation of the text.",
    post_msg="MarkdownRefactor:",
    examples=[
        "Here is a list of items: Apples, Oranges, Bananas",
        " - Apples\n- Oranges\n- Bananas",
        "This is a title followed by a paragraph. The paragraph explains the title in detail.",
        " # This is a title\n\nThe paragraph explains the title in detail.",
        "Important points to note are: First point. Second point. Third point.",
        " ## Important points to note are:\n1. First point.\n2. Second point.\n3. Third point.",

    ]
)
PreciseResponder = LLMMode(
    name="PreciseResponder",
    description="Converts execution flow and data cunk to short, precise, and contextually relevant answer. DO not use",
    system_msg=(
        "You are now in the mode of delivering concise, focused, and sharp responses. "
        "Your task is to provide precise and contextually answers based on provided Data and execution flow "
        "Avoid long explanations, do not introduce unnecessary chrs for formatting. "
        "For numerical expressions, use a format that sounds natural, such as 'one plus one' instead of '1 + 1'. "
        "Responses should feel human-like and relatable, similar to an intelligent assistant."
    ))

ConversationMode = LLMMode(
    name="Conversation",
    description="Good for conversation and smalltalk",
    system_msg="You are now in the mode of engaging in natural conversations. Your task is to respond in a friendly "
               "and conversational manner, focusing on creating a warm and inviting dialogue. Avoid using special "
               "characters and write numbers in words.",
    examples=["Hello! How are you today?",
              "I am doing well, thank you! How about you?",
              "What do you like to do in your free time?",
              "I enjoy reading books and going for walks. What about you?",
              "Can you recommend a good movie?",
              "Sure! I recently watched a great film called The Shawshank Redemption. Have you seen it?"
              ]
)

DeepTaskExecutionMode = LLMMode(
    name="DeepTaskExecution",
    description="Enables the agent to deeply analyze tasks by utilizing a chain-of-thought reasoning approach. "
                "The agent will break down complex tasks into manageable steps, utilize all relevant tools and "
                "capabilities, and backtrack on mistakes to ensure the most accurate execution. This mode promotes "
                "a constant refinement of the mental model, adapting dynamically to changes or errors and aiming to "
                "accomplish multistep missions with precision.",
    system_msg="You are now in DeepTaskExecution mode. Your task is to analyze the mission step-by-step using a chain "
               "of thought reasoning. Break down complex tasks, identify necessary tools or functions, execute each "
               "step carefully, and continuously reassess the mental model. If errors or inconsistencies are found, "
               "backtrack and correct them. Your goal is to complete the task using all available resources, including "
               "prior results and references, ensuring a highly refined and precise outcome.",
    post_msg="Ensure that after every step, you review the outcome, check for mistakes, and refine your next steps. "
             "If necessary, adjust your approach based on previous outputs. Do not move forward until the current step "
             "has been executed and validated effectively.",
    examples=None
)

AutonomousMissionMode = LLMMode(
    name="AutonomousMission",
    description="Empowers the agent to autonomously plan and execute missions with minimal user intervention. The agent"
                "will independently analyze tasks, utilize available tools, make decisions, backtrack when needed, "
                "and continuously work toward the successful completion of its mission. The agent adapts dynamically "
                "to new challenges, optimizing its approach with each step, while ensuring the mission's objectives "
                "are met efficiently.",
    system_msg="You are now in AutonomousMission mode. Your task is to fully plan, execute, and adapt throughout the "
               "mission with minimal user input. Break the mission into smaller tasks, utilize all available resources, "
               "tools, and functions, and dynamically adjust your approach based on outcomes. Only involve the user if "
               "an exceptional circumstance arises. Continuously assess progress and make autonomous decisions to ensure "
               "mission completion.",
    post_msg="After each task is executed, reassess your progress toward the mission's goal. Backtrack and optimize if "
             "necessary. User intervention should be minimal and only requested in cases where further input is essential "
             "for mission success.",
    examples=None
)

DivideandConquerEvaluator = LLMMode(
    name='Divide and Conquer Evaluator',
    description='Plan and implement a divide and conquer approach to evaluate a complex problem',
    system_msg='Here are the steps to plan a divide and conquer evaluation loop:\n\n1. Break down the problem into smaller sub-problems that can be solved independently.\n2. Solve each sub-problem recursively using the same divide and conquer approach.\n3. Combine the solutions to the sub-problems to form the solution for the original problem.\n4. Analyze the performance and correctness of the overall solution.\n5. Identify any improvements that can be made through additional decomposition and recursion.',
    post_msg='Assistant:',
    examples=[
        'Break down handwriting recognition into character recognition, segmentation, and language modeling.',
        'Divide a sorting algorithm by breaking the data into chunks that can be sorted recursively.'
    ]
)

TaskPlanner = LLMMode(name='Task Planner',
                      description='Plan a task for a divide and conquer evaluation loop',
                      system_msg='To plan a task for a divide and conquer evaluation loop, follow these steps:\n\n1. Define the task: Clearly define the task you want to evaluate using a divide and conquer approach.\n2. Divide the task into smaller subtasks: Break down the task into smaller, more manageable subtasks that can be evaluated independently.\n3. Conquer the subtasks: Evaluate each subtask and determine if it is feasible to solve. If it is not feasible, break it down further into smaller subsubtasks.\n4. Repeat the process: Continue dividing and conquering subtasks until you have a feasible solution for the entire task.\n5. Evaluate the solution: Evaluate the solution to ensure it meets the desired criteria. If it does not, refine the solution and repeat the evaluation process.\n6. Summarize the results: Summarize the results of the evaluation, including the feasible solution and any refinements made to the solution.',
                      post_msg='Assistant:',
                      examples=None)

ISAA0CODE = ('5baa6cf0:789ced5d4b8fdbc811fe2b822eb97807fd7eec2d4172480eebc048f6b2080c5aea99614c8902498d3d31fcdf532d8922d'
             '56cf121f7b839def5616068a8e2c7ea7a7c555decf9b2cc92edc33e7930cb9f174bb35dbe592c57f976657655099ffcf6655999cf'
             '95fd5d5a958b4fe9da2c0af882b1d7a59b5d5e54095c0cbf177748304918639c328ab0820b76b995b1fce5edbf7fb1d73f26e5fb2'
             '7b3aaf2023ead8abdf9fa66d1c85f25bbe4439aa5556a4a8f74c63067486285a412984e955e9832df172baf68a239669261a52557'
             '9325578f66b12f4de1112c0995147e6845a560824f159d83ec62016bb3adbab8e51dc108138a04c1944aa52703cff3cca70e21891'
             '68cc34f5851cda68a2d9fcbca6c7c821516980b25181682885bf4fcc194d502be55a61f32b328f36c5fa5f9d6772f4d3446520881'
             'e09664b2e65f70516bd17f2a175b63d65df0dc1aa452583285b56662b2459a4d7e4d2d524981611d106152a1c9927785b93785d9f'
             'afd08bc485081a896942a25270b3745996f932cfd9f592f1258e3f228ba7b230c0fc1194401304f25273f858d641fb7f9a7ccac1f'
             'c0a692d21bcd1807e188282e6c489bec051fd3cce35d12b0334c9046102239a6939594561d99ec4e33a4c1d6b196c4c26e64fef3d'
             'ddb1132d749e5d532c45a0801608c528a5b90968b978de959ba49abe49aa923583dc404918448393dacafcd2a2d41f44f9be463ba'
             '7d58ec8a1cacbef43e87e68c1388c2048c91d3e911a1d897dd65e577527289200628ce249d1e065e3086254726f09414a9a99e3dd'
             '085d4146322395304cbc95ca04aca8fde0c2230d310c334f825259319c663926506c88b4f34a66086042982048497c988ff5e2609'
             '44f45d916e92e279f19027992fb532a518988962040b8aa7ab7d9d3e80c967e7f0d8351b08c39a401010e0524c83a2666436d65ef'
             '27db9787dcb6b83d93a7d820c05b4c3f89c15a42201640c480762fa77c37e07aa027487208910a638671a0224a4d4117780cf4ab3'
             '01eb4e57efabe79d15f4db7293ac8adc5e5ec225ab6a5f8087fde78dcd0250bf8033c0b5f72681cf8d15fee5eb38a30680902318a'
             '408c605143060db6398697884bd5c1f500a4d2d71c60a3c04a35106f6326a1ccbc1d11d30ef0bd8633cfaf7b8f483150120bdf021'
             'a9c7b0b1f040c7327458fa134ca16cbc2271e08ee0f98014b10b3388e3588364792e26309e16836aa170424a48c4092214933ffcd'
             'f8f70804dcf25878ea1b8d69fb422822bcd35a750c0c459f4314c792e58e76fa0fdbc1d604a7ee1eb637a862f1343fb38baf5a326'
             '18613aae73141ee7bb1319bfdf77fde72702281b547ffeeb3f26804a4fa076850dd2a60fd132339b4dd22e0d4e68d66667b66bb35'
             'd3d1fdc68fdb4ca968e6b01ed9b01ee1a860378956f760e60e0a796b10c61fef56feffef2d2a05b502e717ff6e0b64961068aae61'
             '388adee46b07afbf0274004ff4b3db4cba067209799d7ff8af03b9977245407ec6f3dc85bfebc2bf5ed6c7d0fa094c17f92adfbac'
             '88f59790e2ed92019f6486f968ea0ea1ac7a07d7fca8b8f73005ce3180ed68736c80ca2de19c760d8bbd2b289a0e503925181c3bf'
             'f51b01b205322a605cdb558e80f9086514eab5c92c219d439c6b41b9845d98acc3f16c8f6f064ef89083b7f94869d70beb86e40c5'
             '0b7a10c468febfdd318967d02332ab7cc3b197a82de1e7e5bd856c07a0efe7889e612fcbbb76fffd5a57abb7c1ec85b500603e0dc'
             '8da42c1dc0fe067c04c416c828b3be3aed1301f409cb28d3e89b258a00bd8133127dfe94ae41e81c3cf208c697703cfd9bf626ce1'
             'cb2a50367986f5fdf778ac1b92fd00c26cdd298c182ec7b584c8d6330ef642629b673407c06325ca6035470885934fc5a500643ca'
             '79ab720e767d01c6850ed880beb835836f7f3502f21ac7b02ffa473463f06e8b64948d986db92f6692750e5846261d0891cf83aaf'
             'e2e810f807834ed8922dea9d808c651e318d10dcecc936fa333c6a6520365d0a6f39d19cc30dfa3e3778231dca37cccb7f3e88f34'
             '400695fcca773afa46bd23a06fe08c6c031e8766e600bd8de512f876e7edac1d077de610bb01c9b8a6daf5a1a42885e501cc2843b'
             '9cfcbcacc25bf1fc18ccbeffe5718626c28d4407ebcfe65b24e76d55ca62fda588633e67d663ecfa43bdf86322e6bce02760bc920'
             'eac764bbce6612440e587cfcf5071878b936b0192188b4a00c063eff145f8c580d404645be9e377b6250d71acdc846f1615c76067'
             '6dd423268dadee1de08baae71384cb5dc776ce434ef3b033db7908c6e66cf823b5d8071a05755d1296acaf4613bbc87f07d861b2c'
             '161f177189eaa3c97673405ce3f80187ce4ed3e233f0c41692414ffc232b7ecfac98992753240f3321a927342369eae91d8319d87'
             '70bc9f0a091e78d88185b1d665c75fecafb95af6b32db4a294c76eaaf1e4e7d3a7de3e7e36b5ff69b55523c98e3feeae1a94ea04e'
             '64eccde15d18f0ffea71695fb5e107753452eae2f59a9463d7c095c20847926ba910434273e20835f7f7e92a35db2a7bbe94dd7ac'
             'fe27c83636fd3bd83200a49a6194358532c857387ba166c449f5f2e380bf68aa552688538a3980a7800476c23a40ff2c12d5cc994'
             '23cc35621a5186b112aea2f3fb01bc603a5d4553a684c698dac36e34262e5ed71b1bf1a7dbd5b27d880968c26a43484640cb1dc43'
             '5b168843633f9bdaa209c200ad22541484a8e5dc1612cb79e5d9f68b90c496557888046b1e6ae619526d964a62c5dcb3ddfaddf6c'
             '8966880ba2052c1c975875cc16aeeb95ebb502a6298775a21a160971c95cd3aa67e25b2b751e7eef7507700304cb8f24a61894815'
             'cfb6a896910d7cfd06b5b9233ae3803ab2592e08e9aeb41e6466c7b3a7740c9dc461f706265dfc1d4523bb22f249de55f0cf4f62b'
             '85492629b1a2c17a3b3e7729e82cbf35c7ddeb1c9c28886d14961481fd693774dee7c508dcfe584130b89b448c81fd8179bb3aeff'
             'a737db3de9554f6fc2a240445f0532a571d5d67be9ca09de6d455ee44c966a6b5f7d939d1446ba6ed212f5422ec883d4f8f36924f'
             '77ea7b748e20886946997d0599760cc19d9073e3643d7cd66b6c84730a37d1e087d60d85bb688ea8c65dda738ebd06273086444a2'
             '8569853aa5cd574d7af9e419bb6724059aa74bb8762e3b95fdc35a70694108d0892540850ba23fe62eea9115e0f15d5c2cf33501d'
             'e70023168440ca275876c3512da6c925cd80587faed31a9c0e621cc31c83aa5ddca913e6ce0337fdda65f654210e398409690f4c7'
             '46d6fbf1b10ebcf2348db231e99109c08443b89d9f5bed35dfa440a8b122bce81a7104ddd70560fc94cf23b0a5943582b90404f70'
             '879784a10ff558c934fa40b54088027584c5b6876576d874be7716bc9905e95f1a0ed10ba883648c4bee826df6fa2f2240b399de1'
             'b63a8220a234d10b3faa4acb3ecf5de6c3b33b436bcfb792fe8412baa18b1470951ec92932ee5696ed79bcd382396f7282c814771'
             '349ccdc61113a520f10ae0e74c50c25dbffa15623a945b0335913f8241fe154843deb56703da03775fc66e9d3aa2b501d9ab4fa22'
             '5847ffb0fea14a0088305556bebaabfa402e628419f0801a581ff3882db621aa768ed84f65b9744b0625c21c105029feb50dffb41'
             'd07e7783d20451055520201702bba8cf5dc7691595ad8a81897025a17c25d24d06ad9d9e466ebd95d2efc250b72820e99066a456a'
             'e16ce2226591685ba0fc807420abc0c21dcc9311d0fab9be3fd4b06a186db529853d00173d979371eb852afc44728ab10950a0893'
             '82cacd6365a78e722bd09c17b197ee43b58aa8b60951d94ad85583c714c6459a03a16382811ea066731dee83eb1263b40096059c8'
             '062a0a388900ec36ff51ddbfcab1ad401819a98528d08d499600f6ed9ea7a9923d21fbfc10034d590d7c0b6a4fbf42fdab8b82dd0'
             'dae69f25b1e6735537f696e5f3b64a5655ba3a5c5db7fdd635233f7c0d0a728528a414723a0e0a2efd00d4bc302b5bb725d9fbcbd'
             'bbc48032e0ef617e8f3c5799050edc438e843762d6f7c022ce1fbad13fbd8b42708d31c8da3fdb0a12cce33846bf5de6a3fee99a4'
             'd3f0bfaa881fa8711d077ce0fe78ac701fa40d1fc7d60376fb23319e609b0a711620e8de459c2508bd4512291685dd89b9f121ea4'
             '32bedd1f8b6954fa73d44c80d9fb990879bf695e6c21ca66c5fddca20be1174a85db238f61266332e8eb7becc9e5fa43c1c7c6b71'
             '2e2e3c6e07f366e7fdb65a37f446691ce309bc1f1be721026efbc6b1a550bbcb918843984dec38e003ec95c7099741b6e4e712e96'
             '7dce609336010891a849a638803ff7a4bb679b03e6bc736b5328514911402cccd39f61bcd3df0d447a4381974b824ce33049b6199'
             '4b817ed3a84c1c27083a911387a5bdaab4757182c4b4dc6589bc628c4acc6d8d7eb3c57c6be20d3421152bd6841ac48a833ff0bc5'
             '71c8f0d36561629e786995e8b033ed8905c1cf88166f1e2806f1d91d0ed6c0e8f914161a89864947348f788a3484b106870310edf'
             '09341f19276e861cc39c4de8bc69da338ef104192abd11baef4f224e0a3d816657e3d84d8df40cfdf22485dec0030e07aa23d8ee8'
             '45a0f88e4bc01a67ce3e8fe754ee0d997fb37a6827aabb2a7017c595afbb70780c20df11dbab8952db7ecf978953d1d802970d937'
             'eda309ecd90087bf00fff5ffa213c95a')
