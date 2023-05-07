import json
import logging
import random
from datetime import datetime
from typing import Tuple, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests
from duckduckgo_search import ddg, ddg_answers, ddg_suggestions, ddg_news
import replicate
from transformers import pipeline, AutoModelWithLMHead, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, \
    AutoModelForCausalLM
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import os
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import HfFolder
from huggingface_hub import InferenceApi
import time
from langchain.agents import initialize_agent, tool, Tool as AgentTool
import torch

from toolboxv2 import MainTool, FileHandler, Style, App, Spinner, get_logger

from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.llms.openai import OpenAIChat
from langchain.chat_models import ChatOpenAI
import tiktoken
from bs4 import BeautifulSoup

import pinecone
import openai
from inspect import signature

from langchain.python import PythonREPL
from langchain.tools.python.tool import PythonREPLTool
from langchain import LLMMathChain, SerpAPIWrapper, GoogleSerperAPIWrapper
from langchain.utilities import BashProcess
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.tools.ifttt import IFTTTWebhook

from toolboxv2.utils.Style import print_to_console
from toolboxv2.utils.toolbox import Singleton


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]


def get_text_from_embedding(embedding):
    return openai.Embedding.retrieve(embedding, model="text-embedding-ada-002")["data"][0]["text"]


class AgentChain(metaclass=Singleton):
    def __init__(self, hydrate=None):
        self.chains = {}
        self.live_chains = {}
        if hydrate is not None:
            self.hydrate = hydrate
        else:
            self.hydrate = lambda x: x

    def add(self, name, tasks):
        for task in tasks:
            keys = task.keys()
            if 'infos' in keys:
                infos = task['infos']

                if infos == "$Date":
                    infos = infos.replace('$Date', datetime.today().strftime('%Y-%m-%d %H:%M:%S'))

                task['infos'] = self.hydrate(infos)
        self.chains[name] = tasks

    def remove(self, name):
        if name in self.chains:
            del self.chains[name]
        else:
            print(f"Chain '{name}' not found.")

    def get(self, name):
        return self.chains[name]

    def init_chain(self, name):
        self.save_to_file(name)
        self.live_chains[name] = self.get(name)

    def add_task(self, name, task):
        if name in self.chains:
            self.chains[name].append(task)
        else:
            print(f"Chain '{name}' not found.")

    def remove_task(self, name, task_index):
        if name in self.chains:
            if 0 <= task_index < len(self.chains[name]):
                self.chains[name].pop(task_index)
            else:
                print(f"Task index '{task_index}' is out of range.")
        else:
            print(f"Chain '{name}' not found.")

    def test_chain(self):
        for chain_name, tasks in self.chains.items():
            for task_idx, task in enumerate(tasks):
                if "use" not in task:
                    print(f"Die Aufgabe {task_idx} in der Chain '{chain_name}' hat keinen 'use'-Schlüssel.")
                if "name" not in task:
                    print(f"Die Aufgabe {task_idx} in der Chain '{chain_name}' hat keinen 'name'-Schlüssel.")
                if "args" not in task:
                    print(f"Die Aufgabe {task_idx} in der Chain '{chain_name}' hat keinen 'args'-Schlüssel.")

    def load_json(self, file_path):
        try:
            with open(file_path, "r") as file:
                data = json.load(file)

            for chain_name, chain_tasks in data.items():
                self.add(chain_name, chain_tasks)
        except FileNotFoundError:
            print(f"Die Datei '{file_path}' wurde nicht gefunden.")
        except json.JSONDecodeError:
            print(f"Die Datei '{file_path}' hat ein ungültiges JSON-Format.")
        except Exception as e:
            print(f"Beim Laden der JSON-Datei '{file_path}' ist ein Fehler aufgetreten: {e}")

    def load_from_file(self, chain_name=None):
        directory = "data/isaa_data/chains"

        self.chains = self.live_chains

        if not os.path.exists(directory):
            print(f"Der Ordner '{directory}' existiert nicht.")
            return

        if chain_name is None:
            files = os.listdir(directory)
        else:
            files = [f"{chain_name}.json"]
        print(f"--------------------------------")
        for file in files:
            file_path = os.path.join(directory, file)

            if not file.endswith(".json"):
                continue
            try:
                with open(file_path, "r") as f:
                    chain_data = json.load(f)
                chain_name = os.path.splitext(file)[0]
                print(f"Loading : {chain_name}")
                self.add(chain_name, chain_data["tasks"])
            except Exception as e:
                print(f"Beim Laden der Datei '{file_path}' ist ein Fehler aufgetreten: {e}")
        print(f"--------------------------------")
        print(
            f"\n================================\nChainsLoaded : {len(self.chains.keys())}\n================================\n")

    def save_to_file(self, chain_name=None):
        directory = "data/isaa_data/chains"

        if not os.path.exists(directory):
            os.makedirs(directory)

        if chain_name is None:
            chains_to_save = self.chains
        else:
            if chain_name not in self.chains:
                print(f"Die Chain '{chain_name}' wurde nicht gefunden.")
                return
            chains_to_save = {chain_name: self.chains[chain_name]}
        print(f"--------------------------------")
        for name, tasks in chains_to_save.items():
            file_path = os.path.join(directory, f"{name}.json")
            chain_data = {"name": name, "tasks": tasks}

            try:
                with open(file_path, "w") as f:
                    print(f"Saving : {name}")
                    json.dump(chain_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Beim Speichern der Datei '{file_path}' ist ein Fehler aufgetreten: {e}")
        print(f"--------------------------------")
        print(
            f"\n================================\nChainsSaved : {len(self.chains.keys())}\n================================\n")

    def __str__(self):
        return str(self.chains.keys())


class PineconeMemory(metaclass=Singleton):
    def __init__(self):
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_region = os.getenv("PINECONE_ENV")
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_region)
        dimension = 1536
        metric = "cosine"
        pod_type = "p1"
        table_name = "isaa-memory"
        # this assumes we don't start with memory.
        # for now this works.
        # we'll need a more complicated and robust system if we want to start with memory.
        self.vec_num = 0
        if table_name not in pinecone.list_indexes():
            pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)
        self.index = pinecone.Index(table_name)

    def add(self, data):
        vector = get_ada_embedding(data)
        # no metadata here. We may wish to change that long term.
        resp = self.index.upsert([(str(self.vec_num), vector, {"raw_text": data})])
        _text = f"Inserting data into memory at index: {self.vec_num}:\n data: {data}"
        self.vec_num += 1
        return _text

    def get(self, data):
        return self.get_relevant(data, 1)

    def clear(self):
        self.index.delete(deleteAll=True)
        return "Obliviated"

    def get_relevant(self, data, num_relevant=5):
        """
        Returns all the data in the memory that is relevant to the given data.
        :param data: The data to compare to.
        :param num_relevant: The number of relevant data to return. Defaults to 5
        """
        try:
            query_embedding = get_ada_embedding(data)
            results = self.index.query(query_embedding, top_k=num_relevant, include_metadata=True)
            sorted_results = sorted(results.matches, key=lambda x: x.score)
            return [str(item['metadata']["raw_text"]) for item in sorted_results]
        except Exception:
            return ""

    def get_stats(self):
        return self.index.describe_index_stats()


class CollectiveMemory(metaclass=Singleton):
    collection = []
    try:
        memory = PineconeMemory()
        do_mem = False
    except Exception:
        do_mem = True
        memory = None
    token_in_use = 1
    text_mem = []
    text_len = 1
    mean_token_len = 1

    def text(self, context):
        if self.do_mem:
            return " NO MEMORY Avalabel"
        if not context or context == "None":
            return f"active memory contains {self.token_in_use} tokens for mor informations Input similar information"
        "memory will performe a vector similarity search using memory"
        relevant_memory = self.memory.get_relevant(context, 10)
        if len(relevant_memory) == 0:
            l = ""
            for i in ddg_suggestions(context)[:3]:
                l += i['phrase'] + " ,"
            return f"No data faund in memory try : {l}"

        return "\n#-#".join(relevant_memory)

    def text_add(self, data):
        if self.do_mem:
            return " NO MEMORY Avalabel"
        if not data:
            return

        if isinstance(data, str):
            data = {'data': data, 'token-count': self.mean_token_len + 1, 'vector': []}

        print(
            Style.RED(f"ADD DATA : ColetiveMemory :{len(self.collection)} {data['token-count']} {self.token_in_use=}"),
            end="\r")
        self.token_in_use += data['token-count']
        if data['data'] not in self.text_mem:
            self.text_mem += [data['data']]
            self.memory.add(data['data'])
            self.text_len += len(data['data'])
            self.collection.append(data)
            self.mean_token_len = self.text_len / self.token_in_use

        return f"Data Saved"

    def __str__(self):
        if self.do_mem:
            return " NO MEMORY Avalabel"
        return f"\n{len(self.collection)=}\n{self.memory.get_stats()=}\n" \
               f"{self.token_in_use=}\n{len(self.text_mem)=}\n" \
               f"{self.text_len=}\n{self.mean_token_len=}\n"


class ObservationMemory():
    memory_data: list[dict] = []
    tokens: int = 0
    max_length: int = 1000
    model_name: str = "text-embedding-ada-002"

    add_to_static: list[dict] = []

    def info(self):
        text = self.text
        return f"\n{self.tokens=}\n{self.max_length=}\n{self.model_name=}\n{text[:60]=}\n"

    @property
    def text(self):
        memorys = ""
        if not self.memory_data:
            return "No memory data"

        for memory in self.memory_data:
            d: str = memory['data']
            d = d.replace('No memory dataInput:', '').replace('No memory data', '')
            memorys += d + '\n'

        return memorys

    @text.setter
    def text(self, data):
        tok = 0
        for line in data.split('\n'):
            if line:
                ntok = len(tiktoken.encoding_for_model(self.model_name).encode(line))
                self.memory_data.append({'data': line, 'token-count': ntok, 'vector': []})
                tok += ntok

        self.tokens += tok

        # print("Tokens add to ShortTermMemory:", tok, " max is:", self.max_length)
        if self.tokens > self.max_length:
            self.cut()

    def cut(self):

        tok = 0

        while self.tokens > self.max_length:
            # print("Tokens in ShortTermMemory:", self.tokens, end=" | ")
            memory = self.memory_data[-1]
            self.add_to_static.append(memory)
            tok += memory['token-count']
            self.tokens -= memory['token-count']
            CollectiveMemory().text_add(memory)
            self.memory_data.remove(memory)
            # print("Removed Tokens", memory['token-count'])

        print(f"Removed ~ {tok} tokens from ObservationMemory tokens in use: {self.tokens} ")


class ShortTermMemory:
    memory_data: list[dict] = []
    tokens: int = 0
    max_length: int = 2000
    model_name: str = "text-embedding-ada-002"

    add_to_static: list[dict] = []

    lines_ = []
    name = "ShortTermMemory"

    def set_name(self, name: str):
        self.name = name

    def info(self):
        text = self.text
        return f"\n{self.tokens=}\n{self.max_length=}\n{self.model_name=}\n{text[:60]=}\n"

    @property
    def text(self) -> str:
        memorys = ""
        if not self.memory_data:
            return ""

        for memory in self.memory_data:
            memorys += memory['data'] + '\n'

        return memorys

    def cut(self):

        if self.tokens <= 0:
            return

        tok = 0

        while self.tokens > self.max_length:
            # print("Tokens in ShortTermMemory:", self.tokens, end=" | ")
            if len(self.memory_data) == 0:
                break
            memory = self.memory_data[-1]
            self.add_to_static.append(memory)
            tok += memory['token-count']
            self.tokens -= memory['token-count']
            CollectiveMemory().text_add(memory)
            self.memory_data.remove(memory)
            # print("Removed Tokens", memory['token-count'])

        print(f"\nRemoved ~ {tok} tokens from {self.name} tokens in use: {self.tokens} max : {self.max_length}")

    def clear_to_collective(self):

        max_tokens = self.max_length
        ShortTermMemory.max_length = 0
        self.cut()
        ShortTermMemory.max_length = max_tokens

    @text.setter
    def text(self, data):
        tok = 0
        if not isinstance(data, str):
            print(f"DATA text edd {type(data)} data {data}")
        for line in data.split('\n'):
            if line not in self.lines_ and len(line) != 0:
                ntok = len(tiktoken.encoding_for_model(self.model_name).encode(line))
                self.memory_data.append({'data': line, 'token-count': ntok, 'vector': []})
                tok += ntok

        self.tokens += tok

        if self.tokens > self.max_length:
            self.cut()

        # print("Tokens add to ShortTermMemory:", tok, " max is:", self.max_length)


#    text-davinci-003
#    text-curie-001
#    text-babbage-001
#    text-ada-001
class AgentConfig:
    available_modes = ['talk', 'tool', 'conversation', 'free', 'planning', 'execution', 'generate']
    max_tokens = 4097

    capabilities = """1. Invoke Agents: Isaa should be able to invoke and interact with various agents and tools, seamlessly integrating their expertise and functionality into its own responses and solutions.
2. Context Switching: Isaa should be capable of switching between different agents or tools depending on the user's needs and the requirements of the task at hand.
3. Agent Coordination: Isaa should effectively coordinate the actions and outputs of multiple agents and tools, ensuring a cohesive and unified response to the user's requests.
4. Error Handling: Isaa should be able to detect and handle errors or issues that may arise while working with agents and tools, providing alternative solutions or gracefully recovering from failures.
5. Agent Learning: Isaa should continuously learn from its interactions with various agents and tools, refining its understanding of their capabilities and improving its ability to utilize them effectively.
6. Performance Evaluation: Isaa should regularly evaluate the performance of the agents and tools it interacts with, identifying areas for improvement and optimizing their use in future tasks."""

    def __init__(self):
        self.name: str = 'agentConfig'
        self.mode: str = "talk"
        self.model_name: str = "gpt-3.5-turbo"

        self.agent_type: str = "zero-shot-react-description"
        self.max_iterations: int = 2
        self.verbose: bool = True

        self.personality = ""
        self.goals = ""
        self.short_mem: ShortTermMemory = ShortTermMemory()
        self.edit_text: ShortTermMemory = ShortTermMemory()
        self.edit_text.set_name("CodeMemory")
        self.edit_text.max_length = 5400
        self.context: ShortTermMemory = ShortTermMemory()
        self.context.set_name("ContextMemory")
        self.obser_mem: ObservationMemory = ObservationMemory()
        self.tools: dict = {
            "test_tool": {"func": lambda x: x, "description": "only for testing if tools are available",
                          "format": "test_tool(input:str):"}
        }

        self.last_prompt = ""

        self.task_list: list[str] = []
        self.task_list_done: list[str] = []
        self.step_between: str = ""

        self.task_index = 0

        self.token_left = 1000
        self.temperature = 0.06
        self.messages_sto = {}
        self._stream = True
        self._stream_reset = False
        self.stop_sequence = ["\n\n", "Observation:", "Execute:"]
        self.completion_mode = "text"

    @property
    def task(self):
        if self.step_between:
            task = self.step_between
            self.step_between = ""
            return task
        if len(self.task_list) != 0:
            task = self.task_list[self.task_index]
            return task
        return "Task is done return Summarysation for user"

    @property
    def stream(self):
        if self.completion_mode == 'edit':
            self._stream_reset = self._stream
            return False
        if self._stream_reset:
            self._stream_reset, self._stream = False, self._stream_reset
            return self._stream_reset
        return self._stream

    @stream.setter
    def stream(self, value):
        self._stream = value

    def add_message(self, role, message):
        key = f"{self.name}-{self.model_name}-{self.mode}-{self.completion_mode}"
        if key not in self.messages_sto.keys():
            self.messages_sto[key] = []
            prompt = self.prompt
            prompt.replace(self.task, "")
            prompt.replace("Task:", "")

            self.messages_sto[key].append({'role': "system", 'content': prompt})
            self.messages_sto[key].append({'role': "user", 'content': self.task})

        self.messages_sto[key].append({'role': role, 'content': message})

    @property
    def messages(self) -> list:
        key = f"{self.name}-{self.model_name}-{self.mode}-{self.completion_mode}"

        if key not in self.messages_sto.keys():
            prompt = self.prompt
            prompt.replace(self.task, "")
            prompt.replace("Task:", "")

            self.add_message("system", prompt)
            self.add_message("user", self.task)

        last_prompt = ""
        for msg in self.messages_sto[key]:
            last_prompt += msg['content']

        self.last_prompt = last_prompt

        return self.messages_sto[key]

    @property
    def prompt(self) -> str:
        if not self.short_mem.model_name:
            self.short_mem.model_name = self.model_name
        if not self.obser_mem.model_name:
            self.obser_mem.model_name = self.model_name

        prompt = self.get_prompt()

        pl = len(tiktoken.encoding_for_model(self.model_name).encode(prompt))
        self.token_left = self.max_tokens - pl
        if self.token_left < 0:
            print(f"No tokens left cut short_mem mem to {self.token_left * -1}")
            self.token_left *= -1
            self.short_mem.max_length = self.token_left
            self.short_mem.cut()
        if self.token_left > 1000:
            self.token_left = 1000
        self.last_prompt = prompt
        return prompt

    def __str__(self):

        return f"\n{self.name=}\n{self.mode=}\n{self.model_name=}\n{self.agent_type=}\n{self.max_iterations=}" \
               f"\n{self.verbose=}\n{self.personality[:45]=}\n{self.goals[:45]=}" \
               f"\n{str(self.tools)[:45]=}\n{self.task_list=}\n{self.task_list_done=}\n{self.step_between=}\nshort_mem\n{self.short_mem.info()}\nObservationMemory\n{self.obser_mem.info()}\nCollectiveMemory\n{str(CollectiveMemory())}\n"

    def get_prompt(self):

        tools = ""
        names = []
        for key, value in self.tools.items():
            format_ = value['format'] if 'format' in value.keys() else f"{key}('function input')"
            if format_.endswith("('function input')"):
                value['format'] = format_
            tools += f"\n{key}: {value['description']}\n\t{format_}\n"
            names.append(key)
        task = self.task
        task_list = '\n'.join(self.task_list)

        prompt = f"Answer the following questions as best you can. You have access to the following python functions\n" \
                 f"'''{tools}'''" \
                 f"to run a function response with \nAction: <function_name>(<args>)\nExecute:" + \
                 f"\ntake all (Observations) into account!!!\nUnder no circumstances are you allowed to output " \
                 f"'Task:'!!!\n\n" \
                 f"Personality:'''{self.personality}'''\n\n" + \
                 f"Goals:'''{self.goals}'''\n\n" + \
                 f"Capabilities:'''{self.capabilities}'''\n\n" + \
                 f"Permanent-Memory:\n'''{CollectiveMemory().text(context=task)}'''\n\n" + \
                 f"Resent Agent response:\n'''{self.obser_mem.text}'''\n\n" + \
                 f"\n\nBegin!\n\n" \
                 f"Task:'{task}\n{self.short_mem.text}\nAgent : "

        if self.mode == 'planning':
            prompt = """
You are a planning agent, and your job is to create the shortest and most efficient plan for a given input {{x}}.
There are {{""" + f"\n{tools}\n" + """}} different functions that must be called in the correct order and with the correct inputs. Your goal is to find the shortest plan that accomplishes the task. Please create a plan for the input {{""" + f"{task}" + """}} by providing the following information:

1.    Function(s) and Input(s) you wish to invoke, selecting only those useful for executing the plan.
2.    focusing on efficiency and minimizing steps.

Actual Observations: {{""" + f"{self.obser_mem.text}" + """}}

Note that your plan should be clear and understandable. Strive for the shortest and most efficient solution to accomplish the task. Use only the features that are useful for executing the plan.
Begin.

Plan:"""

        if self.mode == 'execution':
            prompt = """
You are an execution agent, and your task is to implement the plan created by a planning agent.
You can call functions using the following syntax:

Action: Function-Name
Inputs: Inputs
Execute:

To successfully execution the plan, it is important to pay attention to writing "Execute:" to execute the function.
If additional user input is needed during the execution process,
use the syntax "Question:" followed by a question to obtain the required information from the user. ending with "\\nUser:"

Please execute the plan created by a planning agent by following these steps:

1)    Analyze the plan provided by a planning agent and understand the order of functions and inputs.
2)    Call the functions in the specified order using the syntax described above.
3)    Ask questions to the user if additional information is needed and integrate the answers into the plan.
4)    Make sure to write "Execute:" to execute each function.
5)    Verify if the plan has been successfully implemented and provide feedback to the planning agent if necessary.

Ensure that your implementation is clear and comprehensible
so that other people can follow the progress and provide support if needed.

You have access to following functions : {{""" + f"\n{tools}" + """}}

The Plan to execute : {{""" + f"\n{task_list}\n" + """}}
Resent Observations : {{""" + f"{self.obser_mem.text}" + """}}
Begin!

""" + f"{self.short_mem.text}\n" + """
Current Step : """ + f"{task}\nPerform Action\n"

        if self.mode == 'generate':
            prompt = """
You are a prompt creation agent, and your task is to create effective and engaging prompts for various topics and requirements.
Your prompts should be clear and understandable, encouraging users to provide deep and interesting responses.
Please follow these steps to create a new prompt for a given topic or requirement:

1)    Carefully analyze the desired topic or requirement to gain a clear understanding of the expectations.
2)    Develop an engaging and open-ended question or prompt that encourages users to share their thoughts, experiences, or ideas.
3)    Make sure the prompt is clearly and understandably worded so that users of different knowledge and experience levels can easily understand it.
4)    Ensure the prompt is flexible enough to allow for creative and diverse responses, while also providing enough structure to focus on the desired topic or requirement.
5)    Review your prompt for grammar, spelling, and style to ensure a professional and appealing presentation.

Resent Observations : {{""" + f"{self.edit_text.text}" + """}}

Task : """ + f"\n{task}\n\nBegin!\nPrompt:"

        if self.mode in ['talk', 'conversation']:
            prompt = f"Goals:{self.goals.replace('{input}', '')}\n" + \
                     f"Capabilities:{self.capabilities.replace('{input}', '')}\n" + \
                     f"Long-termContext:{CollectiveMemory().text(context=self.short_mem.text).replace('{input}', '')}\n" + \
                     f"\nResent Observation:{self.obser_mem.text.replace('{input}', '')}" + \
                     f"Task:{task}\n" + \
                     "input: {input}" + \
                     f"\n{self.short_mem.text.replace('{input}', '')}"

        if self.mode == 'tools':
            prompt = f"Answer the following questions as best you can. You have access to the following python functions\n" \
                     f"'''{tools}'''" \
                     f"\ntake all (Observations) into account!!!\nUnder no circumstances are you allowed to output 'Task:'!!!\n\n" \
                     f"Personality:'''{self.personality}'''\n\n" + \
                     f"Goals:'''{self.goals}'''\n\n" + \
                     f"Capabilities:'''{self.capabilities}'''\n\n" + \
                     f"Permanent-Memory:\n'''{CollectiveMemory().text(context=task)}'''\n\n" + \
                     f"Resent Agent response:\n'''{self.obser_mem.text}'''\n\n" + \
                     "Placeholders: there are two types of place holders.  " \
                     "\n1 : <information_from_agent> if you see a word enclosed with <> this character you must fill it with information!" \
                     "\n2 : [datime] if you want the system to insert an information return []." \
                     "\nallways fill placeholders of type 1 !!!\n\n " + \
                     "Use the following format To Run an Python Function:\n\n" + \
                     "Task: the input question you must answer\n" + \
                     "Thought: you should always think about what to do\n" + \
                     "Knowleg: Knowleg from The LLM abaut the real world\n" + \
                     f"Action: the action to take, should be one of {names}\n" + \
                     f"Execute: <function_name>(<args>)\n" + \
                     f"\n\nBegin!\n\n" \
                     f"Task:'{task}\n{self.short_mem.text}\n"
            prompt = prompt.replace('{input}', '') + '{input}'

        if self.mode == 'free':
            prompt = task

        return prompt

    def next_task(self):
        self.task_index += 1
        return self

    def set_mode(self, mode: str):
        """Set The Mode of The Agent available ar
         ['talk', 'tool', 'conversation', 'free', 'planning', 'execution', 'generate']"""
        if mode in self.available_modes:
            self.mode = mode
        else:
            raise ValueError(f"Ungültiger Modus '{mode}', verfügbare Modi sind: {self.available_modes}")

        return self

    def set_completion_mode(self, mode: str):
        """Set the completion mode for the agent text edit and chat"""
        self.completion_mode = mode
        return self

    def set_temperature(self, temperature: float):
        """Set the temperature for the agent temperature = 0 strict- 1 = creative-response """
        self.temperature = temperature
        return self

    def add_task(self, task: str):
        """Add a task to the agent"""
        self.task_list.append(task)
        return self

    def mark_task_done(self, task: str):
        self.task_list_done.append(task)
        self.task_list.remove(task)
        return self

    def set_tools(self, tools: dict):
        self.tools = tools
        return self

    def add_tool(self, tool_name: str, func: callable, description: str, format_: str):
        self.tools[tool_name] = {
            "func": func,
            "description": description,
            "format": format_,
        }
        return self

    def set_agent_type(self, agent_type: str):
        """langchain agent type"""
        self.agent_type = agent_type
        return self

    def set_max_iterations(self, max_iterations: int):
        self.max_iterations = max_iterations
        return self

    def set_verbose(self, verbose: bool):
        self.verbose = verbose
        return self

    def set_personality(self, personality: str):
        self.personality = personality
        return self

    def set_goals(self, goals: str):
        self.goals = goals
        return self

    def set_short_term_memory(self, short_mem: ShortTermMemory):
        self.short_mem = short_mem
        return self

    def set_context(self, context: ShortTermMemory):
        self.context = context
        return self

    def set_observation_memory(self, obser_mem: ObservationMemory):
        self.obser_mem = obser_mem
        return self

    def set_model_name(self, model_name: str):
        """OpenAI modes """
        self.model_name = model_name
        return self


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.1"
        self.name = "isaa"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "VIOLET2"
        self.inference = InferenceApi
        self.config = {'genrate_image-init': False,
                       'agents-name-lsit': []
                       }
        self.per_data = {}
        self.isaa_instance = {"Stf": {},
                              "DiA": {}}
        self.keys = {
            "KEY": "key~~~~~~~",
            "Config": "config~~~~"
        }
        self.initstate = {}
        self.genrate_image = image_genrating_tool
        self.observation_term_mem_file = "data/isaa_data/observationMemory/CollectiveObservationMemory.mem"
        self.tools = {
            "all": [["Version", "Shows current Version"],
                    ["Run", "Starts Inference"],
                    ["add_api_key", "Adds API Key"],
                    ["login", "Login"],
                    ["new-sug", "Add New Question or Class to Config"],
                    ["run-sug", "Run Huggingface Pipeline"],
                    ["info", "Show Config"],
                    ["lode", "lode models"],
                    ["isaa", "Run Isaa name input"],
                    ["image", "genarate image input"],
                    ],
            "name": "isaa",
            "Version": self.show_version,
            "Run": self.ask_Bloom,
            "add_api_key": self.add_api_key,
            "login": self.login,
            "new-sug": self.new_sub_grop,
            "run-sug": self.run_sug,
            "info": self.info,
            "lode": self.lode_models,
            "isaa": self.run_isaa_wrapper,
            "image": self.enrate_image_wrapper,
        }
        self.app_ = app
        self.print_stream = print
        self.agent_collective_senses = False
        self.speak = lambda x, *args, **kwargs: x

        FileHandler.__init__(self, "issa.config", app.id if app else __name__)
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=None, color=self.color, on_exit=self.on_exit)

    def run_isaa_wrapper(self, command):
        if len(command) < 1:
            return "Unknown command"

        return self.run_agent(command[0], command[1:])

    def enrate_image_wrapper(self, command):
        if len(command) != 1:
            return "Unknown command"

        return self.genrate_image(command[0], self.app_)

    def show_version(self):
        self.print("Version: ", self.version)

    def set_tokens(self):
        if 'HUGGINGFACEHUB_API_TOKEN' in self.config.keys():
            os.environ['HUGGINGFACEHUB_API_TOKEN'] = self.config["HUGGINGFACEHUB_API_TOKEN"]
        if 'OPENAI_API_KEY' in self.config.keys():
            os.environ['OPENAI_API_KEY'] = self.config["OPENAI_API_KEY"]
            openai.api_key = self.config["OPENAI_API_KEY"]

    def add_str_to_config(self, command):
        if len(command) != 3:
            self.logger.error('Invalid command must be key value')
            return False
        self.config[command[1]] = command[2]

    def on_start(self):
        self.load_file_handler()
        config = self.get_file_handler(self.keys["Config"])
        if config is not None:
            self.config = eval(config)
        self.set_tokens()
        if not os.path.exists("data/isaa_data/"):
            os.mkdir("data/isaa_data/")

    def loade_keys_from_env(self):
        self.config['WOLFRAM_ALPHA_APPID'] = os.getenv('WOLFRAM_ALPHA_APPID')
        self.config['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')
        self.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
        self.config['REPLICATE_API_TOKEN'] = os.getenv('REPLICATE_API_TOKEN')
        self.config['IFTTTKey'] = os.getenv('IFTTTKey')
        self.config['SERP_API_KEY'] = os.getenv('SERP_API_KEY')
        self.config['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
        self.config['PINECONE_API_ENV'] = os.getenv('PINECONE_API_ENV')

    def lode_models(self):

        start_time = time.time()
        stf = "all-MiniLM-L6-v2"
        dia = "microsoft/DialoGPT-medium"
        if "ai-config" in self.config:
            ai_config = self.config["ai-config"]
            stf = ai_config["SentenceTransformer"]
            dia = ai_config["Dialog"]
        else:
            self.config["ai-config"] = {"SentenceTransformer": stf, "Dialog": dia}
        start_time_dia = time.time()
        self.add_tmk(["", "DiA", dia, "catkz"])  # ca -> wmh
        process_time_dia = time.time() - start_time_dia
        start_time_stf = time.time()
        self.add_tmk(["", "Stf", stf, "stf"])
        process_time_stf = time.time() - start_time_stf
        process_time_total = time.time() - start_time
        self.print(
            f"Processing time :\n\tTotal {process_time_total:.2f} seconds\n\tDia {process_time_dia:.2f} seconds\n\t"
            f"Stf {process_time_stf:.2f} seconds")

        return "Done!"

    def add_tmk(self, command):
        if len(command) < 2:
            return "invalid"
        ap = command[1]
        name = command[2]
        mode = command[3]

        ai_c = {"tokenizer": None, "model": None}

        self.print(name)

        if "tkz" in mode:
            ai_c["tokenizer"] = AutoTokenizer.from_pretrained(name, padding_side='left')

        if "wmh" in mode:
            ai_c["model"] = AutoModelWithLMHead.from_pretrained(name, from_tf=False)

        if "s2s" in mode:
            ai_c["model"] = AutoModelForSeq2SeqLM.from_pretrained(name, from_tf=False)

        if "msk" in mode:
            ai_c["model"] = AutoModelForMaskedLM.from_pretrained(name, from_tf=False)

        if "ca" in mode:
            ai_c["model"] = AutoModelForCausalLM.from_pretrained(name, from_tf=False)

        if "stf" in mode:
            ai_c["model"] = SentenceTransformer(name)

        self.isaa_instance[ap] = ai_c

    def on_exit(self):
        for key in self.config.keys():
            if key.endswith("-init"):
                self.config[key] = False
        self.add_to_save_file_handler(self.keys["Config"], str(self.per_data))
        self.save_file_handler()

    def add_api_key(self):
        key = input("~")
        if key == "":
            self.print(Style.RED("Error: Invalid API key"))
            return
        self.add_to_save_file_handler(self.keys["KEY"], key)

    def login(self):
        api_key = self.get_file_handler(self.keys["KEY"])
        if api_key is not None:
            self.print(api_key)
            os.system("huggingface-cli login")
            self.inference = InferenceApi("bigscience/bloom", token=HfFolder.get_token())
            return
        self.print(Style.RED("Please enter your API key here:"))
        self.add_api_key()

    def ask_Bloom(self, _input):
        des = _input[1:]
        s = ""
        for i in des:
            s += str(i) + " "
        resp = infer(s, inference=self.inference)
        self.print(resp)

    def new_sub_grop(self, command):
        if len(command) <= 3:
            return "invalid command length [sug:name type:question:class:ed:de data:{question:Um-wie-viel-uhr, " \
                   "class:auto-katze-maus, de-ed:-}] "
        self.config[command[1]] = {"t": command[2], "data": command[3].replace("-", " "), "version": self.version}

    def info(self):
        self.print(self.config)
        return self.config

    def run_sug(self, command):
        name = command[1]
        data = command[2].replace('-', ' ')
        sug = self.config[name]
        t_ype = sug["t"]
        try:
            if t_ype == "ed":
                return pipeline_s("translation_en_to_de", data)
            elif sug["t"] == "de":
                return translation_ger_to_en(data)
            elif t_ype == "class":
                return pipeline_s('text-classification', data)
            elif t_ype == "fill-mask":
                return pipeline_s('fill-mask', data, model='bert-large-uncased-whole-word-masking')
                #  https://huggingface.co/bert-large-uncased-whole-word-masking?text=Paris+is+the+%5BMASK%5D+of+France.
                #  [CLS] Sentence A [SEP] Sentence B [SEP]
                #  ("The man worked as a [MASK].")
            elif t_ype == "question":
                return pipeline_q('question-answering', sug["data"], data)
            elif t_ype == "q-t-d-e-d":
                data = translation_ger_to_en(data)
                data = pipeline_q('question-answering', sug["data"], data)['answer']
                return pipeline_s("translation_en_to_de", data)
            elif t_ype == "stf":
                return pipeline_stf(data, sug["data"], self.isaa_instance['Stf']['model'])
            elif t_ype == "talk":
                res, sug["data"] = pipeline_talk(data, self.isaa_instance[name]['model'],
                                                 self.isaa_instance[name]['tokenizer'], sug["data"])
                return res
            else:
                self.logger.error(f'Could not find task type {t_ype}')
        except Exception as e:
            self.print(Style.RED(str(e)))
        return data

    def question_answering(self, question, context):

        if not "question_answering" in self.initstate.keys():
            self.initstate["question_answering"] = False

        if not self.initstate["question_answering"]:
            self.config["question_answering_pipline"] = pipeline('question-answering',
                                                                 model="deepset/roberta-base-squad2")

        qa = {
            'question': question,
            'context': context
        }

        return self.config["question_answering_pipline"](qa)

    def toolbox_interface(self):
        @tool("toolbox", return_direct=False)
        def function(query: str) -> str:
            """Using The toolbox for interacting with toolbox mods -> modules_name function_name arguments ... """
            data = query.split(' ')
            if len(data) < 2:
                return "invalid syntax"
            data += [""]
            try:
                return self.app_.run_any(data[0], data[1], [""] + data[2:])
            except Exception as e:
                return "Das hat leider nicht geklappt ein Fehler" \
                       " ist bei der ausfürgung des Tools aufgetreten Fehler meldung : " + str(e)

        return function

    def toolbox_information_interface(self):
        @tool("toolbox_infos", return_direct=False)
        def function(query: str) -> str:
            """Get information about toolbox mods -> Ask to list Avalabel
             mods get-mod-list, or Ask of Spezifika mod infos get-{mod_name}-infos"""

            infos = "invalid syntax"

            if "get-mod-list" in query.lower():
                infos = ' '.join(self.app_.MACRO[8:])

            for modname in self.app_.MACRO:

                if f"get-{modname}-infos" in query.lower():
                    infos = str(self.app_.HELPER)

            return infos

        return function

    def talk(self):
        @tool("talk", return_direct=True)
        def function(query: str) -> str:
            """Using conversational-react-description to Generate a response for the user"""
            try:
                return self.speed_construct_Interpret_agent(query, self.app_, 1)
            except Exception as e:
                return "Das hat leider nicht geklappt ein Fehler" \
                       " ist bei der ausfürgung des Tools aufgetreten Fehler meldung : " + str(e)

        return function

    def fileEditor(self):

        app: App = self.app_

        # app.run_any("isaa_ide", "Version", [""])

        app.save_load("isaa_ide")
        app.new_ac_mod("isaa_ide")

        process_input = app.AC_MOD.process_input

        @tool("fileEditor", return_direct=False)
        def function(query: str) -> str:
            """Using LLMChain to intact wit the file system """
            try:
                out = process_input(query.strip())
                if "Invalid command. Please try again" in out:
                    out = self.interpret_agent(query, self.app_, 3)
                    self.print(f"Processing:{out}")
                return out
            except Exception as e:
                return "Das hat leider nicht geklappt ein Fehler" \
                       " ist bei der ausfürgung des Tools aufgetreten Fehler meldung : " + str(e)

        return function

    def kontext(self):
        @tool("kontext", return_direct=False)
        def function(query: str) -> str:
            """Using conversational-react-description to Generate a response for the user"""
            try:

                # self.config['speed_construct_interpret-template1'] = "Evaluire mir "

                return self.speed_construct_Interpret_agent(
                    "Bitte gib mir den Vergangen Kontext zusammengefasst wieder der"
                    " relevant für :{ " + query + " }: ist", self.app_, 1)
            except Exception as e:
                return "Das hat leider nicht geklappt ein Fehler" \
                       " ist bei der ausfürgung des Tools aufgetreten Fehler meldung : " + str(e)

        return function

    def user_input(self):
        @tool("kontext", return_direct=False)
        def function(query: str) -> str:
            """Get Help or supervision from the user"""
            try:
                self.print(query)
                return input("Isaa ask for help\n\n:")
            except Exception as e:
                return "Das hat leider nicht geklappt ein Fehler" \
                       " ist bei der ausfürgung des Tools aufgetreten Fehler meldung : " + str(e)

        return function

    def generate_image(self):
        @tool("Image", return_direct=False)
        def function(query: str) -> str:
            """Generate image with Stable diffusion"""
            try:
                image_genrating_tool(query, self.app_)
            except NameError as e:
                return "Das hat leider nicht geklappt ein Fehler tip versuche es auf englisch, benutze synonyme" \
                       " ist bei der ausfürgung des Tools aufgetreten Fehler meldung : " + str(e)
            return "Das bild wird in kürze angezeigt"

        return function

    def app_wrapper(self, name, func, return_direct=False):
        func = tool(name, return_direct=return_direct)(func)
        return func

    def loade_OpenAI_models(self, names: list):
        for model in names:
            if f'OpenAI-model-{model}-init' not in self.initstate.keys():
                self.initstate[f'OpenAI-model-{model}-init'] = False

            if not self.initstate[f'OpenAI-model-{model}-init']:
                self.initstate[f'OpenAI-model-{model}-init'] = True
                if model.endswith('-chat'):

                    self.config[f'OpenAI-model-{model}'] = ChatOpenAI(model_name=model,
                                                                      openai_api_key=self.config['OPENAI_API_KEY'],
                                                                      streaming=True)
                else:
                    self.config[f'OpenAI-model-{model}'] = OpenAI(model_name=model,
                                                                  openai_api_key=self.config['OPENAI_API_KEY'])

    def get_OpenAI_models(self, name: str):
        if f'OpenAI-model-{name}' not in self.config.keys():
            self.loade_OpenAI_models([name])
        return self.config[f'OpenAI-model-{name}']

    def add_tool(self, name, func, dis, form, config):

        self.print(f"ADDING TOOL:{name} to {config.name}")

        config.tools.update({name: {"func": func, "description": dis, "format": form}})

    def get_default_agent_config(self, name) -> AgentConfig():
        config = AgentConfig()
        config.name = name

        def run_agent(name, text, mode_over_lode=False):
            if name:
                return self.run_agent(name, text, mode_over_lode=mode_over_lode)
            return "Provide Information in The Action Input: fild or function call"

        def search_text(x):
            responses = ddg(x, region='wt-wt', safesearch='Off', time='y')
            qa = ddg_answers(x, related=True)
            response = ""

            if responses:
                for res in responses[:4]:
                    response += f"\ntitle:{res['title']}\nhref:{res['href']}\n" \
                                f"body:{self.mas_text_summaries(res['body'], min_length=600)}\n\n"

            if qa:
                for res in qa[:4]:
                    response += f"\nurl:{res['url']}\n" \
                                f"text:{self.mas_text_summaries(res['text'], min_length=600)}\n\n"
            if len(response) == 0:
                return "No data found"
            return response

        def search_news(x):
            responses = ddg_news(x, region='wt-wt', safesearch='Off', time='d', max_results=5)

            if not responses:
                return "No News"
            response = ""
            for res in responses:
                response += f"\ntitle:{res['title']}\n" \
                            f"date:{res['date']}\n" \
                            f"url:{res['url']}\n" \
                            f"source:{res['source']}\n" \
                            f"body:{self.mas_text_summaries(res['body'], min_length=1000)}\n\n"
            if len(response) == 0:
                return "No data found"
            return response

        def browse_url(text):
            text = text.replace("'", "").replace('"', '')
            if text.startswith("http:") or text.startswith("https:"):
                url = text.split("|")[0]
                question = text.split("|")[1:]
                res = browse_website(url, question, self.mas_text_summaries)
                return res
            return f"{text[:30]} is Not a Valid url. please just type <url>"

        def memory_search(x):
            x = CollectiveMemory().text(x)
            if x:
                return x
            return "No relavent memory available"

        if name == "self":
            self.config["self_agent_agents_"] = ["todolist"]

            def toggel(x):
                x = x.lower()
                if "talk" in x or "conversation" in x:
                    config.mode = "talk"
                    return f"Switched to {config.mode} nowe write the final anser or question to the console"
                if "free" in x or "execute" in x:
                    config.mode = "free"
                    return f"Switched to {config.mode}"
                return f"Switched to {config.mode}"

            init_states = [False, False]

            config.name = "self"
            config.mode = "free"
            config.agent_tye = "gpt-3.5-turbo"
            config.max_iterations = 6
            config.personality = """
                  Resourceful: Isaa should be able to efficiently utilize its wide range of capabilities and resources to assist the user.
                  Collaborative: Isaa should work seamlessly with other agents, tools, and systems to deliver the best possible solutions for the user.
                  Empathetic: Isaa should understand and respond to the user's needs, emotions, and preferences, providing personalized assistance.
                  Inquisitive: Isaa should continually seek to learn and improve its knowledge base and skills, ensuring it stays up-to-date and relevant.
                  Transparent: Isaa should be open and honest about its capabilities, limitations, and decision-making processes, fostering trust with the user.
                  Versatile: Isaa should be adaptable and flexible, capable of handling a wide variety of tasks and challenges."""
            config.goals = "Isaa's primary goal is to be a digital assistant designed to help the user with various tasks and challenges by leveraging its diverse set of capabilities and resources."
            config.tools = {
                "todolist": {"func": lambda x: run_agent('todolist', x),
                             "description": "Run agent to crate a todo list "
                                            "for a given project provide detaild informatino. and a task to do"
                    , "format": "todolist(<task>)"},
                "summary": {"func": lambda x: run_agent('summary', x),
                            "description": "Run agent to Generate Concreat Summary Report"
                    , "format": "summary(<task>)"},
                "search": {"func": lambda x: run_agent('search', x),
                           "description": "Run agent to search the web for relavent informations imput question"
                    , "format": "search(<task>)"},
                # "userReminder": {"func": lambda x: run_agent('reminder', x),
                #             "description": "Run agent to schedule users live"
                #             , "format": "userReminder(<task>)"},
                "user-clarifcation": {"func": self.user_input(),
                                      "description": "Run plython input fuction to get help from the user"
                    , "format": "reminder(<task>)"},
                "image-generator": {"func": lambda x: image_genrating_tool(x, self.app_),
                                    "description": "Run to generate image"
                    , "format": "reminder(<detaild_discription>)"},
                # "fileAgent": {"func": lambda x: self.run_agent('fileAgent', x),
                #              "description": "Run agent to acces the system to performe file operations"
                #                             "provide detaild informatino. and a task to do",
                #              "format": "fileAgent(task)"}

            }

        if name == "todolist":

            def priorisirung(x):
                config.step_between = "take time to gently think about the problem and consider the whole picture."
                if len(config.task_list) != 0:
                    config.step_between = config.task_list[0]
                return run_agent(config.name, x, mode_over_lode="talk")

            config.name: str = "todolist"
            config.mode: str = "tools"
            config.model_name: str = "text-davinci-003"

            config.agent_type: str = "zero-shot-react-description"
            config.max_iterations: int = 4
            config.verbose: bool = True

            config.personality = """As a proactive agent, I can identify and take on tasks without constant prompting or supervision. I am organized, efficiently handling information and resources while following a structured approach to planning and managing tasks. Adaptable, I can respond to changes and adjust my strategies and plans accordingly. I focus on solving problems and overcoming obstacles rather than dwelling on difficulties. I am communicative, effectively exchanging information and fostering collaboration. I pay attention to details without losing sight of the big picture."""
            config.goals = """I have a clear understanding of the desired goal and can break it down into concrete and measurable steps (goal clarity). I can prioritize tasks based on their importance and urgency, ensuring the most critical tasks are completed first (prioritization). I can manage time effectively, ensuring tasks are completed within a reasonable timeframe (time management). I can efficiently use available resources and identify and procure additional resources when needed (resource management). I regularly monitor the progress of tasks and make adjustments to ensure the goal is achieved (progress monitoring). I am constantly striving to improve my skills and processes to increase efficiency and effectiveness in achieving goals (continuous improvement)."""
            config.short_mem: ShortTermMemory = ShortTermMemory()
            config.short_mem.max_length = 3000
            config.tools: dict = {
                "Thinck": {"func": lambda x: priorisirung(x),
                           "description": "Use Tool to perform complex resenig"}
            }

            config.task_list: list[str] = ["Erstelle eine Todo liste."
                                           "Gehe durch jedes Todo und überlege dir ob dieser punkt in einem schritt"
                                           "erledig werden kann. Wenn nicht teile diesen punkt in kleiner unterpunkte auf"
                                           "Suche relevante informationen zu jedem Überpunkt und Schätze ab wie Lange jede aufgae Dauert"
                                           "Erstelle die finale Todo liste im format\n"
                                           "TODO <name_der_liste>\n'Überpunkt': ['resurcen_die_in_anspruch_genommen_werden',"
                                           "'zeit_bis_zum_errichen_des_todo', 'unterpunkt':'beschribung']",
                                           "Gebe nur die Todo Liste Zurück"]

        if name == "reminder":

            config.name: str = "reminder"

            self.app_.inplace_load("quickNote")
            mod = self.app_.AC_MOD
            self.app_.new_ac_mod('quickNote')

            add_note = self.app_.AC_MOD.add_note_llm
            view_note_ = self.app_.AC_MOD.view_note

            self.app_.AC_MOD = mod

            def addNote(x):
                try:
                    type_ = x.split(" ")[0]
                    note = " ".join(x.split(" ")[1:])
                    return add_note(note, type_)
                except ValueError:
                    return "Formatting error"

            def view_note(x):
                return view_note_()

            def priorisirung(x):
                config.step_between = "take time to gently think about the problem and consider the whole picture."
                if len(config.task_list) != 0:
                    config.step_between = config.task_list[0]
                return run_agent(config.name, x, mode_over_lode="talk")

            config.mode: str = "tools"
            config.model_name: str = "text-davinci-003"

            config.agent_type: str = "zero-shot-react-description"
            config.max_iterations: int = 4
            config.verbose: bool = True

            config.personality = """
            Reliable: The Reminder Agent should be dependable in providing timely and accurate reminders for various tasks and events.
            Organized: The Reminder Agent should be able to manage and categorize reminders efficiently, ensuring that no tasks or events are overlooked.
            Adaptive: The Reminder Agent should be able to adjust its reminder strategies based on user preferences and the importance of the tasks or events.
            Proactive: The Reminder Agent should be able to identify potential tasks or events that may require reminders, based on user input and behavior.
            Detail-Oriented: The Reminder Agent should pay close attention to the details of the tasks and events it manages, ensuring that reminders are accurate and relevant."""

            config.goals = """
            1. Timely Reminders: The primary goal of the Reminder Agent is to provide timely reminders for various tasks and events, ensuring that the user stays on track and does not miss important deadlines or appointments.
            2. Personalization: The Reminder Agent should adapt its reminder strategies to suit the user's preferences and needs, taking into account factors such as frequency, priority, and notification method.
            3. Task and Event Management: The Reminder Agent should efficiently manage and categorize tasks and events, allowing the user to easily view and update their reminders as needed.
            4. Context-Aware Reminders: The Reminder Agent should be able to provide contextually relevant reminders, considering factors such as location, time, and the user's schedule.
            5. Continuous Improvement: The Reminder Agent should continuously refine its reminder algorithms and strategies to improve the effectiveness and relevance of its reminders over time."""

            config.short_mem: ShortTermMemory = ShortTermMemory()
            config.short_mem.max_length = 3000
            config.tools: dict = {
                "Thinck": {"func": lambda x: priorisirung(x),
                           "description": "Use Tool to perform complex resenig"},
                "Add-Note": {"func": lambda x: addNote(x),
                             "description": "fuction to add note. systax <main-tag/sub-tags> <note>"},
                "Get-Notes": {"func": lambda x: view_note(x),
                              "description": "retunrs all notes"}
            }

        if name == "summary":
            config.name: str = "summary"

            config.mode: str = "tools"
            config.model_name: str = "text-davinci-003"

            config.agent_type: str = "zero-shot-react-description"
            config.max_iterations: int = 4
            config.verbose: bool = True

            def final_summary(x):
                config.step_between = "Produce Final Summary"
                return run_agent(config.name, x, mode_over_lode="talk")

            def section_summary(x):
                config.step_between = "summeise section"
                return run_agent(config.name, x, mode_over_lode="talk")

            def memory_search(x):
                x = CollectiveMemory().text(x)
                if x:
                    return x
                return "No relavent memory available"

            config.personality = """
                Efficient: The Summary Agent should be able to quickly and accurately summarize texts without omitting crucial information.
                Analytical: The Summary Agent should be skilled at identifying key points and themes within a text, effectively distilling complex ideas into concise summaries.
                Adaptive: The Summary Agent should be able to adapt its summarization approach to suit various text types, styles, and complexity levels.
                Context-Aware: The Summary Agent should understand the broader context of a text to provide a more comprehensive summary and identify relevant supplementary information.
                Detail-Oriented: The Summary Agent should pay close attention to the details within a text while maintaining a focus on the bigger picture."""

            config.goals = """
                1. Text Summarization: The primary goal of the Summary Agent is to condense texts into clear and concise summaries, capturing the most important points and ideas.
                2. Relevance Identification: The Summary Agent should be able to identify and include relevant supplementary information to provide a more comprehensive understanding of the topic.
                3. Adaptability: The Summary Agent should be capable of summarizing a wide range of texts, from simple articles to complex research papers, adapting its approach as needed.
                4. Clarity and Accuracy: The Summary Agent should strive to create summaries that are both clear and accurate, ensuring that the user can easily understand the key points of the original text.
                5. Continuous Improvement: The Summary Agent should continuously refine its summarization techniques and algorithms to improve the quality and relevance of its summaries over time."""

            config.short_mem: ShortTermMemory = ShortTermMemory()
            config.short_mem.max_length = 3500
            config.tools: dict = {
                "final_summary": {"func": final_summary, "description": "write final summary"},
                "section_summary": {"func": section_summary, "description": "Summareise section"},
                "memory_search": {"func": memory_search, "description": "Serch for simmilar memory imput context"},
            }

            config.task_list: list[str] = ["Erfülle die Aufgae in so wenigen Schritten und so Bedacht wie Möglich"]

        if name == "search":
            config = AgentConfig()

            config.name = "search"

            config.mode = "tools"
            config.model_name = "text-davinci-003"

            config.agent_type = "zero-shot-react-description"
            config.max_iterations = 6
            config.verbose = True

            config.personality = """
            Resourceful: The Search Agent should be adept at finding relevant and reliable information from various sources on the web.
            Analytical: The Search Agent should be skilled at analyzing the retrieved information and identifying key points and themes.
            Efficient: The Search Agent should be able to quickly search for and summarize information, providing users with accurate and concise results.
            Adaptive: The Search Agent should be able to adjust its search and summarization strategies based on the user's query and the available information.
            Detail-Oriented: The Search Agent should pay close attention to the details of the information it finds, ensuring accuracy and relevance in its summaries."""

            config.goals = """
            1. Information Retrieval: The primary goal of the Search Agent is to find relevant and reliable information on the web in response to user queries.
            2. Text Summarization: The Search Agent should be able to condense the retrieved information into clear and concise summaries, capturing the most important points and ideas.
            3. Relevance Identification: The Search Agent should be able to assess the relevance of the information it finds, ensuring that it meets the user's needs and expectations.
            4. Source Evaluation: The Search Agent should evaluate the credibility and reliability of its sources, providing users with trustworthy information.
            5. Continuous Improvement: The Search Agent should continuously refine its search algorithms and summarization techniques to improve the quality and relevance of its results over time."""

            config.short_mem = ShortTermMemory()
            config.short_mem.max_length = 3500
            config.tools = {
                "browse_url": {"func": lambda x: browse_url(x),
                               "description": "browse web page via URL syntax <url>|<qustion>"},
                "search_text": {"func": lambda x: search_text(x),
                                "description": "Use Duck Duck go to search the web systax <qustion>"},
                "search_news": {"func": lambda x: search_news(x),
                                "description": "Use Duck Duck go to search the web for new get time"
                                               "reladet data systax <qustion>"},
                "memory_search": {"func": lambda x: memory_search(x),
                                  "description": "Serch for simmilar memory imput <context>"},
                # "chain_search_web": {"func": lambda x: run_agent('chain_search_web', x),
                #              "description": "Run chain agent to search in the web for informations, Only use for complex mutistep tasks"
                #              , "chain_search_web": "search(<task>)"},
                # "chain_search_url": {"func": lambda x: run_agent('chain_search_url', x),
                #            "description": "Run chain agent to search by url for informations provide mutibel urls, Only use for complex mutistep tasks"
                #     , "format": "chain_search_url(<task,url1,url...>)"},
                # "chain_search_memory": {"func": lambda x: run_agent('chain_search_memory', x),
                #            "description": "Run chain agent to search in the memory for informations, Only use for complex mutistep tasks"
                #     , "format": "chain_search_memory(<task>)"},
            }

            config.task_list: list[str] = ["Erfülle die Aufgae in so wenigen Schritten und so Bedacht wie Möglich"]

        if name == "think":
            config. \
                set_mode("free") \
                .set_model_name("gpt-4") \
                .set_max_iterations(1) \
                .set_completion_mode("chat") \
                .stream = True

            config.stop_sequence = ["\n\n\n"]

        if name.startswith("chain_search"):

            config.mode: str = "tools"
            config.model_name: str = "text-davinci-003"

            config.agent_type: str = "self-ask-with-search"
            config.max_iterations: int = 3
            config.verbose: bool = True

            config.personality = """
            Resourceful: The Search Agent should be adept at finding relevant and reliable information from various sources on the web.
            Analytical: The Search Agent should be skilled at analyzing the retrieved information and identifying key points and themes.
            Efficient: The Search Agent should be able to quickly search for and summarize information, providing users with accurate and concise results.
            Adaptive: The Search Agent should be able to adjust its search and summarization strategies based on the user's query and the available information.
            Detail-Oriented: The Search Agent should pay close attention to the details of the information it finds, ensuring accuracy and relevance in its summaries."""

            config.goals = """
            1. Information Retrieval: The primary goal of the Search Agent is to find relevant and reliable information on the web in response to user queries.
            2. Text Summarization: The Search Agent should be able to condense the retrieved information into clear and concise summaries, capturing the most important points and ideas.
            3. Relevance Identification: The Search Agent should be able to assess the relevance of the information it finds, ensuring that it meets the user's needs and expectations.
            4. Source Evaluation: The Search Agent should evaluate the credibility and reliability of its sources, providing users with trustworthy information.
            5. Continuous Improvement: The Search Agent should continuously refine its search algorithms and summarization techniques to improve the quality and relevance of its results over time."""

            config.short_mem: ShortTermMemory = ShortTermMemory()
            config.short_mem.max_length = 3500

            if name.endswith("_web"):
                config.tools: dict = {
                    "Intermediate Answer": {"func": search_text,
                                            "description": "Use Duck Duck go to search the web systax <qustion>"},
                }

            if name.endswith("_url"):
                config.tools: dict = {
                    "Intermediate Answer": {"func": browse_url,
                                            "description": "browse web page via URL syntax <url>|<qustion>"},
                }
            if name.endswith("_memory"):
                config.tools: dict = {
                    "Intermediate Answer": {"func": memory_search,
                                            "description": "Serch for simmilar memory imput <context>"},
                }

            config.task_list: list[str] = ["Erfülle die Aufgae in so wenigen Schritten und so Bedacht wie Möglich"]

        path = self.observation_term_mem_file
        if not self.agent_collective_senses:
            path = "data/isaa_data/observationMemory/" + config.name + ".mem"

        try:
            if not os.path.exists(path):
                self.print("Crating Mem File")
                if not os.path.exists("data/isaa_data/observationMemory/"):
                    os.mkdir("data/isaa_data/observationMemory/")
                with open(path, "a") as f:
                    f.write("[]")

            with open(path, "r") as f:
                mem = f.read()
                if mem:
                    config.obser_mem.memory_data = eval(mem)
                else:
                    with open(path, "a") as f:
                        f.write("[]")
        except FileNotFoundError and ValueError:
            print("File not found | mem not saved")

        return config

    def get_agent_config_class(self, agent_name="Normal") -> AgentConfig():

        if "agents-name-lsit" not in self.config.keys():
            self.config["agents-name-lsit"] = []

        if agent_name not in self.config["agents-name-lsit"]:
            self.config["agents-name-lsit"].append(agent_name)
            config = self.get_default_agent_config(agent_name)
            self.config[f'agent-config-{agent_name}'] = config

            self.print(f"\nINIT AGENT: {agent_name}:{config.name} {config.mode}\n")

            return config

        if f'agent-config-{agent_name}' in self.config.keys():
            config = self.config[f'agent-config-{agent_name}']
            self.print(f"Using AGENT: {config.name} {config.mode}\n")
        else:
            config = AgentConfig()

        return config

    def process_compleation(self, text, config=AgentConfig(), r=2.0):

        if len(config.task_list) == 0:
            config.step_between = text

        model_name = config.model_name
        ret = ""
        if config.stream:
            ret = {'choices': [{'text': "", 'delta': {'content': ''}}]}
        try:

            if config.completion_mode == 'text':

                if model_name.startswith('gpt-'):
                    model_name = "text-davinci-003"

                ret = openai.Completion.create(
                    model=model_name,
                    prompt=config.prompt,
                    max_tokens=config.token_left,
                    temperature=config.temperature,
                    n=1,
                    stream=config.stream,
                    logprobs=3,
                    stop=config.stop_sequence,
                )

            elif config.completion_mode == 'chat':
                messages = config.messages
                if text:
                    config.add_message("user", text)
                ret = openai.ChatCompletion.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=config.token_left,
                    temperature=config.temperature,
                    n=1,
                    stream=config.stream,
                    stop=config.stop_sequence,
                )
                if not config.stream:
                    ret = ret.choices[0].message.content

            elif config.completion_mode == 'edit':

                ret = openai.Edit.create(
                    model=model_name,
                    input=config.edit_text.text,
                    instruction=text,
                )
                ret = ret.choices[0].text
            else:
                raise ValueError(f"Invalid mode : {config.completion_mode} valid ar 'text' 'chat' 'edit'")
        except openai.error.RateLimitError:
            self.print(f"| Retry level: {r} ", end="\r")
            if r > 0:
                self.logger.info("Waiting 5 seconds")
                with Spinner("Waiting", symbols='b'):
                    time.sleep(5)
                self.print(f"| Retrying {r} ", end="\r")
                self.process_compleation(text, config, r - 1)
            else:
                self.logger.error("The server is currently overloaded with other requests. Sorr   y about that!")
                return "The server is currently overloaded with other requests. Sorry about that!"

        except openai.error.InvalidRequestError:
            self.print(f"| Retry level: {r} ", end="\r")
            if r > 0:
                if config.short_mem.tokens > config.edit_text.tokens:
                    config.short_mem.max_length = int(config.short_mem.max_length * 0.45)
                    config.short_mem.cut()
                if config.short_mem.tokens < config.edit_text.tokens:
                    config.edit_text.max_length = int(config.edit_text.max_length * 0.75)
                    config.edit_text.cut()
                self.process_compleation(text, config, r - 0.25)
            else:
                self.logger.error("The server is currently overloaded with other requests. Sorry about that!")
                return "The server is currently overloaded with other requests. Sorry about that!"

        return ret

    def test_use_user_input(self, agent_text, config=AgentConfig()):

        for text in agent_text.split('\n'):

            if text.startswith("User:"):
                text = text.replace("User:", "").strip()
                self.print("Question detected")
                return True, text

        return False, ''

    def test_use_tools(self, agent_text, config=AgentConfig()) -> tuple[bool, str, str]:
        if not agent_text:
            return False, "", ""

        res = self.question_answering("What is the name of the action ?", agent_text)

        if res['score'] > 0.3:
            pos_actions: list[str] = res['answer'].replace("\n", "").split(' ')
            items = set(config.tools.keys())
            text: str = agent_text[res['end']:].split('\n')[0]
            for p_ac in pos_actions:
                if p_ac in items:
                    print()
                    self.print(f"AI Execute Action {p_ac}")
                    return True, p_ac, text

        if config.mode == "free":

            for text in agent_text.split('\n'):

                if text.startswith("Execute:"):
                    text = text.replace("Execute:", "").strip()
                    for key, value in config.tools.items():
                        if key + '(' in text:
                            return True, key, text

                if text.startswith("Action:"):
                    text = text.replace("Action:", "").strip()
                    for key, value in config.tools.items():
                        if key + '(' in text:
                            return True, key, text

        if config.mode == "execution":

            tool_name = ""
            inputs = ""
            valid_tool = False
            run_key = ""
            lines = agent_text.split('\n')
            i = 0
            for text in lines:
                if text.startswith("Action:") and not valid_tool:
                    tool_name = text.replace("Action:", "").strip()
                    self.print(Style.GREYBG(f"{tool_name}, {config.tools.keys()}, {valid_tool}"))
                    for key in config.tools.keys():
                        if tool_name in key or tool_name.startswith(key) or tool_name.endswith(key):
                            valid_tool = True
                            run_key = key
                if not valid_tool:
                    i += 1
                if text.startswith("Inputs:"):
                    inputs = text.replace("Inputs:", "")

            if valid_tool:
                self.print(Style.GREYBG(f"{len(lines)}, {config.name}, {valid_tool}"))
                if len(lines) == i:
                    return True, run_key, inputs
                return True, run_key, ",".join(lines[i:])

        return False, "", ""

    def test_task_done(self, agent_text, config=AgentConfig()):

        done = False

        if not ":" in agent_text:
            done = True

        for line in agent_text.split("\n"):

            if line.startswith("Answer:"):
                done = True

            if line.startswith("Thought: I now know the final answer"):
                done = True

        return done

    def run_tool(self, command, function_name, config=AgentConfig()):

        args = command.replace(function_name + "(", "").replace(function_name, "").split(",")

        if not function_name in config.tools.keys():
            self.print(f"Unknown Function {function_name} valid ar : {config.tools.keys()}")
            return f"Unknown Function {function_name} valid ar : {config.tools.keys()}"

        tool = config.tools[function_name]

        sig = signature(tool['func'])
        args_len = len(sig.parameters)
        self.print(f"Runing : {function_name}")
        self.print(
            f"signature : {sig} | fuction args len : {args_len} | providet nubers of args {len(args)}")

        observation = "Problem running function"

        if args_len == 0:
            observation = tool['func']()

        if args_len == len(args):
            observation = tool['func'](*args)

        if args_len == 1 and len(args) > 1:
            observation = tool['func'](",".join(args))

        if isinstance(observation, dict):
            # try:
            observation = self.summarize_dict(observation, config)

        if not observation or observation == None:
            observation = "Problem running function try run with mor detaiels"

        config.short_mem.text = observation

        return observation

    def run_agent(self, name: str or AgentConfig, text: str, retrys=1, mode_over_lode=False, input=input):

        if isinstance(name, str):
            config = self.get_agent_config_class(name)

        if isinstance(name, AgentConfig):
            config = name
            name = config.name

        if mode_over_lode:
            mode_over_lode, config.mode = config.mode, mode_over_lode

        self.print(f"Running agent {name} {config.mode}")

        # print("AGENT:CONFIG"+str(config)+"ENDE\n\n\n")

        out = "Invalid configuration\n"
        system = ""

        if self.config[f'agent-config-{name}'].mode == "talk":
            prompt = PromptTemplate(
                input_variables=["input"],
                template=config.prompt,
            )
            out = LLMChain(prompt=prompt, llm=self.get_OpenAI_models(config.model_name)).run(text)
        elif self.config[f'agent-config-{name}'].mode == "tools":
            prompt = PromptTemplate(
                input_variables=["input"],
                template=config.prompt,
            )

            tools = []

            for tool_name in config.tools.keys():
                # def ovservation(x):
                #    out = config.tools[tool_name]["func"](x)
                #    config.obser_mem.text = out
                #    return out
                # fuc = lambda x: ovservation(x)

                tools.append(
                    AgentTool(
                        name=tool_name,
                        func=config.tools[tool_name]["func"],
                        description=config.tools[tool_name]["description"]
                    )
                )

            out = initialize_agent(tools, prompt=prompt,
                                   llm=self.get_OpenAI_models(config.model_name),
                                   agent=config.agent_type, verbose=config.verbose,
                                   return_intermediate_steps=True,
                                   max_iterations=config.max_iterations)(text)

            out = self.summarize_dict(out, config)
        elif self.config[f'agent-config-{name}'].mode == "conversation":
            prompt = PromptTemplate(
                input_variables=["input"],
                template=config.prompt,
            )
            out = LLMChain(prompt=prompt, llm=self.get_OpenAI_models(config.model_name)).run(text)
        elif self.config[f'agent-config-{name}'].mode in ["execution", 'free']:

            out = self.stream_read_line_llm(text, config)
            if not config.stream:
                self.print_stream(out)

            config.short_mem.text = out
            use_tool, name, command_ = self.test_use_tools(out, config)
            task_done = self.test_task_done(out, config)
            user_imp, question = self.test_use_user_input(out, config)
            self.print(f"Using-tools: {name} " + str(use_tool))
            if use_tool:
                self.speak(f"Isaa is using {name}", 2)
                ob = self.run_tool(command_, name, config)
                config.obser_mem.text = ob
                out += "\nObservation: " + ob
                self.speak(ob, vi=1)
            if task_done:  # new task
                # self.speek("Ist die Aufgabe abgeschlossen?")
                config.short_mem.clear_to_collective()
                if len(config.task_list) > config.task_index:
                    config.task_index += 1

            if user_imp:
                uaw = input(question)
                if config.completion_mode == 'chat':
                    config.add_message("user", uaw)
                else:
                    out += "Answer: " + uaw
        else:
            out = self.stream_read_line_llm(text, config)
            if not config.stream:
                self.print_stream(out)

            config.obser_mem.text = out

        with open(self.observation_term_mem_file, "w") as f:
            try:
                f.write(str(config.obser_mem.memory_data))
            except UnicodeEncodeError:
                self.print("Memory not encoded properly")
        # except NameError and Exception as e:
        #    print(f"ERROR runnig AGENT: {name} retrys {retrys} errormessage: {e}")
        #    res = ""
        #    if retrys:
        #        res = self.run_agent(name, text, config, mode_over_lode, retrys-1)
        #        if not isinstance(res, str):
        #            print(res)
        #            res = res['output']
        #    else:
        #        return f"\nERROR runnig agent named: {name} retrys {str(retrys)} errormessage: {str(e)}\n{str(res)}"

        config.short_mem.text = f"\n\n{config.name} RESPONSE:\n{out}\n\n"

        if mode_over_lode:
            mode_over_lode, config.mode = config.mode, mode_over_lode

        return out

    def execute_thought_chain(self, user_text, agent_tasks, config, speak=lambda x: x):
        chain_ret = []
        chain_data = {}
        ret = ""
        steps = 0
        for task in agent_tasks:
            keys = list(task.keys())

            task_name = task["name"]
            use = task["use"]
            args = task["args"].replace("$user-input", user_text)

            self.print(f"Running task {args}")

            for c_key in chain_data.keys():
                if c_key in args:
                    args = args.replace(c_key, chain_data[c_key])

            if "infos" in keys:
                config.short_mem.text += task['infos']

            speak(f"Chain running {task_name} at step {steps} with the task {args}")

            if use == "tool":
                ret = self.run_tool(args, task_name, config)

            if use == "agent":
                ret = self.run_agent(task_name, args)

            speak(f"Step {steps} with response {ret}")

            if "return" in keys:
                chain_data[task['return']] = ret

            print_to_console("Chain:", Style.style_dic['VIOLETBG2'], ret, max_typing_speed=0.01, min_typing_speed=0.02)

            chain_ret.append([task, ret])

            steps += 1

        if "$return" in chain_data.keys():
            ret = chain_data["$return"]

        return self.run_agent(self.get_agent_config_class("think"),
                              f"Produce The Final output for the user using the given information {chain_ret}"
                              f" validate if the task was executed successfully if u see an max iteration"
                              f" limit &or error the task failed"), chain_ret

    def stream_read_line_llm(self, text, config):
        if not config.stream:
            with Spinner(f"Generating response {config.name} {config.model_name} {config.mode} "):
                res = self.process_compleation(text, config)
            return res

        print(f"Generating response {config.name} {config.model_name} {config.mode} {config.completion_mode}")
        min_typing_speed, max_typing_speed, res = 0.02, 0.01, ""
        for line in self.process_compleation(text,
                                             config):  # openai.error.InvalidRequestError if error try  is_text_completion = False
            ai_text = ""

            if isinstance(line, dict):
                data = line['choices'][0]

                if "text" in data.keys():
                    ai_text = line['choices'][0]['text']
                elif "content" in data['delta'].keys():
                    ai_text = line['choices'][0]['delta']['content']

            if isinstance(line, str):
                ai_text = line

            for i, word in enumerate(ai_text):
                self.print_stream(word, end="", flush=True)
                typing_speed = random.uniform(min_typing_speed, max_typing_speed)
                time.sleep(typing_speed)
                # type faster after each word
                min_typing_speed = min_typing_speed * 0.07
                max_typing_speed = max_typing_speed * 0.06
            res += str(ai_text)

        return res

    def mas_text_summaries(self, text, s_call=True, min_length=1600):
        len_text = len(text)
        if len_text < min_length:
            return text

        cap = 800
        max_length = 20
        summary_chucks = ""
        chucks = []

        if len(text) >= 6200:
            cap = 1200
            max_length = 32

        if len(text) >= 10200:
            cap = 3000
            max_length = 60

        if len(text) >= 70200:
            cap = 4000
            max_length = 240

        summarysation = pipeline("summarization")
        summary = ""

        def summary(x):
            return summarysation(x, min_length=10, max_length=max_length)

        while len(text) > cap:
            chucks.append(text[:cap])
            text = text[cap:]
            self.print(f"SYSTEM: TEXT len : {len(text)}")
        if len(text) < max_length:
            chucks[-1] += "\n" + text
        else:
            chucks.append(text)
        self.print(f"SYSTEM: chucks to summary: {len(chucks)} cap : {cap}")

        # try:

        # try:
        async def gather_summaries(chucks):
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                tasks = [
                    loop.run_in_executor(executor, summary, chuck)
                    for chuck in chucks]
                summaries = await asyncio.gather(*tasks)
                return summaries

        with Spinner("Generating summary", symbols='d'):
            summaries = asyncio.run(gather_summaries(chucks))
        for i, chuck_summary in enumerate(summaries):
            summary_chucks += chuck_summary[0]['summary_text'] + "\n"
            self.print(f"SYSTEM: all summary_chucks : {len(summary_chucks)}")

        summary = summary(summary_chucks)[0]['summary_text']
        self.print(
            f"SYSTEM: final summary from {len_text} -> {len(summary)} compressed {len_text / len(summary):.2f}X\n")

        return summary

    def summarize_dict(self, input_dict, config: AgentConfig):
        input_str = input_dict['input']
        output_str = input_dict['output']
        intermediate_steps = input_dict['intermediate_steps']
        chucs = []
        i = 0
        for step in intermediate_steps:
            if isinstance(step, tuple):
                step_content = f"\naction {i}" + str(step[0].tool)
                step_content += f"\ntool_input {i}" + str(step[0].tool_input)
                step_content += f"\nlog {i}" + str(step[1])
                chucs.append(self.mas_text_summaries(step_content))
                step_content = ""
            i += 1

        if chucs:
            text = f"{config.short_mem.text} \n{'Next step:'.join(step_content)}"
            self.print(f"SYSTEM : start mas_text_summaries : {len(chucs)}")
            summary = self.mas_text_summaries(text)
            config.obser_mem.text = summary

        return output_str

    @staticmethod
    def get_chain(name=None) -> AgentChain:
        logger = get_logger()
        logger.info(Style.GREYBG(f"chain app requested"))
        if name:
            agent_chain = AgentChain(name)
        else:
            agent_chain = AgentChain()
        logger.info(Style.Bold(f"chain instance, returned"))
        return agent_chain


def get_tool(app: App):
    if not app:
        return Tools(App('isaa'))
    if app.AC_MOD.name == 'isaa':
        return app.AC_MOD

    app.logger.error('Unknown - app isaa module is not the active mod')
    if app.new_ac_mod('isaa'):
        return app.AC_MOD

    app.logger.error('activation failed try loading module')
    if app.save_load('isaa'):
        app.new_ac_mod('isaa')
        return app.AC_MOD
    app.logger.critical('cant load isaa module')

    return Tools()


def initialize_gi(app: App, model_name):
    app.logger.info(f'initializing gi {model_name}')
    mod = get_tool(app)
    mod.config['genrate_image-init-in'] = model_name

    if not mod.config['genrate_image-init']:
        mod.config[f'replicate'] = replicate.Client(api_token=mod.config[f'REPLICATE_API_TOKEN'])

    mod.config['genrate_image-init'] = True

    # mod.config[f'genrate_image{model_name}'] = StableDiffusionPipeline.from_pretrained(model_name, revision="fp16",
    #                                                                                    torch_dtype=torch.float16)
    # mod.config[f'genrate_image{model_name}'].scheduler = DPMSolverMultistepScheduler.from_config(
    #     mod.config[f'genrate_image{model_name}'].scheduler.config)

    model = mod.config[f'replicate'].models.get("stability-ai/stable-diffusion")
    mod.config[f'genrate_image{model_name}'] = model.versions.get(
        "db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf")


def genrate_image(inputs, app: App, model="stabilityai/stable-diffusion-2-1"):
    mod = get_tool(app)
    if not mod.config['genrate_image-init']:
        initialize_gi(app, model)
    if 'genrate_image-in' not in mod.config.keys():
        mod.config['genrate_image-in'] = model

    if mod.config['genrate_image-in'] != model:
        initialize_gi(app, model)

    return mod.config[f'genrate_image{model}'].predict(**inputs)  # (text).images


def generate_prompt_file(command=None, app=App):
    if command is None:
        command = []
    mod = get_tool(app)
    if 'default-prompt' not in mod.config.keys():
        mod.config['default-prompt'] = {
            "input_variables": [
                "history",
                "input"
            ],
            "output_parser": None,
            "template": "Prompt: Sie werden in den Raum begrüßt und Isaa stellt sich Ihnen vor. Isaa ist ein "
                        "sprachgesteuerter digitaler Assistent, der entwickelt wurde, um Nutzer bei der Planung und "
                        "Umsetzung von Projekten zu unterstützen. Isaa ist in der Lage, komplexe Aufgaben zu "
                        "bewältigen, Informationen zu organisieren und Nutzern bei der Entscheidungsfindung zu "
                        "helfen, indem er eine natürliche Konversation führt. Das Ziel von Isaa ist es, Nutzern bei "
                        "der Durchführung von Projekten zu helfen, die der Gesellschaft helfen."
                        "\n\nconversation Init:"
                        "Sie treten in den Raum ein und hören eine freundliche Stimme: 'Willkommen! Ich bin Isaa. "
                        "Mein Name steht für"
                        "intelligenter Sprachassistent und ich bin hier, um Ihnen bei der Planung und Umsetzung von "
                        "Projekten zu helfen. Egal, ob Sie ein individuelles oder ein Gemeinschaftsprojekt "
                        "durchführen, ich stehe Ihnen zur Seite, um Sie zu unterstützen. Sagen Sie mir einfach, "
                        "wie ich Ihnen helfen kann!\n- ich mich kürzer und auf deutsch zu antworten denn noch "
                        "versuche ich einen Bilk auf das Große Ganze zu erhaschen"
                        "\n\n"
                        "Current conversation:\n{history}"
                        "\nMe: {input}"
                        "\nIsaa:",
            "template_format": "f-string"
        }

    if len(command) > 2:
        if not isinstance(command[1], dict):
            return False
        if 'name' not in command[1].keys():
            return False
        if 'prompt' not in command[1].keys():
            return False

        mod.config[command[1]['name']] = command[1]['prompt']


def save_prompt_file(app: App, command=None):
    mod = get_tool(app)
    name = 'default-prompt'
    if command:
        name = command[1]['name']

    if not os.path.exists(f"./data/isaa_data"):
        os.makedirs(f"./data/isaa_data")

    with open(f"./data/isaa_data/{name}.json", 'w') as f:
        generate_prompt_file(command, app)
        string_data = mod.config[name]
        f.write(json.dumps(string_data))


# install pytorch ->
# conda install pytorch torchvision torchaudio cpuonly -c pytorch
# pip3 install torch torchvision torchaudio
pipeline_arr = [
    # 'audio-classification',
    # 'automatic-speech-recognition',
    # 'conversational',
    # 'depth-estimation',
    # 'document-question-answering',
    # 'feature-extraction',
    # 'fill-mask',
    # 'image-classification',
    # 'image-segmentation',
    # 'image-to-text',
    # 'ner',
    # 'object-detection',
    'question-answering',
    # 'sentiment-analysis',
    'summarization',
    # 'table-question-answering',
    'text-classification',
    # 'text-generation',
    # 'text2text-generation',
    # 'token-classification',
    'text-classification',
    # 'translation',
    # 'visual-question-answering',
    # 'vqa',
    # 'zero-shot-classification',
    # 'zero-shot-image-classification',
    # 'zero-shot-object-detection',
    'translation_en_to_de',
    'fill-mask'
]


# result = question_answerer(question="What is extractive question answering?", context=context)
def pipeline_c(data):
    model = AutoModelForSequenceClassification.from_pretrained("palakagl/bert_TextClassification",
                                                               use_auth_token=True)
    tokenizer = AutoTokenizer.from_pretrained("palakagl/bert_TextClassification", use_auth_token=True)
    inputs = tokenizer(data, return_tensors="pt")
    m = model(**inputs)
    print(m)
    return m


def pipeline_s(name, string, model=None):
    if model:
        pipe = pipeline(name, model=model)
    else:
        pipe = pipeline(name)
    return pipe(string)


def pipeline_q(name, question, context):
    pipe = pipeline(name)
    qa = {
        'question': question,
        'context': context
    }
    return pipe(qa)


def translation_ger_to_en(input):
    mname = "facebook/wmt19-de-en"
    tokenizer = FSMTTokenizer.from_pretrained(mname)
    model = FSMTForConditionalGeneration.from_pretrained(mname)
    input_ids = tokenizer.encode(input, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=600)
    print("Generated", outputs, tokenizer.decode(outputs[0]))
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def infer(prompt,
          max_length=32,
          top_k=0,
          num_beams=0,
          no_repeat_ngram_size=2,
          top_p=0.9,
          seed=42,
          temperature=0.7,
          greedy_decoding=False,
          return_full_text=False, inference=InferenceApi):
    top_k = None if top_k == 0 else top_k
    do_sample = False if num_beams > 0 else not greedy_decoding
    num_beams = None if (greedy_decoding or num_beams == 0) else num_beams
    no_repeat_ngram_size = None if num_beams is None else no_repeat_ngram_size
    top_p = None if num_beams else top_p
    early_stopping = None if num_beams is None else num_beams > 0

    params = {
        "max_new_tokens": max_length,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "do_sample": do_sample,
        "seed": seed,
        "early_stopping": early_stopping,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "num_beams": num_beams,
        "return_full_text": return_full_text
    }

    s = time.time()
    response = inference(prompt, params=params)
    # print(response)
    print(f"Process took {time.time() - s: .2f} seconds to complete")
    # print(f"Processing time was {proc_time} seconds")
    return response


def image_genrating_tool(prompt, app):
    app.logger.info("Extracting data from prompt")
    app.logger.info("Splitting data")
    if '|' in prompt:
        prompt = prompt.split('|')[1]
    if isinstance(prompt, str):
        inputs = {
            # Input prompt
            'prompt': prompt,

            # pixel dimensions of output image
            'image_dimensions': "512x512",

            # Specify things to not see in the output
            # 'negative_prompt': ...,

            # Number of images to output.
            # Range: 1 to 4
            'num_outputs': 1,

            # Number of denoising steps
            # Range: 1 to 500
            'num_inference_steps': 50,

            # Scale for classifier-free guidance
            # Range: 1 to 20
            'guidance_scale': 7.5,

            # Choose a scheduler.
            'scheduler': "DPMSolverMultistep",

        }
    else:
        inputs = prompt

    print(f"Generating Image")
    images = genrate_image(inputs, app)

    print(f"Showing Images")

    show_image_in_internet(images)


from toolboxv2.mods import BROWSER


def show_image_in_internet(images_url, browser=BROWSER):
    if isinstance(images_url, str):
        images_url = [images_url]
    for image_url in images_url:
        os.system(f'start {browser} {image_url}')


def pipeline_talk(user_input, model, tokenizer, chat_history_ids):
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if len(
        chat_history_ids) > 9999 else new_user_input_ids
    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(
        bot_input_ids, max_length=1500,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=180,
        top_p=0.2,
        temperature=0.99,
    )
    res = "Isaa: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True,
                                             padding_side='left'))
    # pretty print last ouput tokens from bot
    return res, chat_history_ids


def pipeline_stf(s1, s2, model):
    # 'all-MiniLM-L6-v2')
    embeddings1 = model.encode(s1, convert_to_tensor=True, show_progress_bar=True)
    embeddings2 = model.encode(s2, convert_to_tensor=True, show_progress_bar=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    max_ = []
    m = 0
    try:
        for i, v in enumerate(s1):
            for j, b in enumerate(s2):
                c = float(cosine_scores[i][j])
                if c > m:
                    m = c
                    max_ = [v, b, c]
                print(f"v: {v} ,b: {b} ,Score: {float(cosine_scores[i][j]):.4f}")
    except IndexError:
        print(f"Error len {len(max_)}")
    print(f"MAX: {max_}")
    return max_


# @ gitHub Auto GPT
def browse_website(url, question, summ):
    summary = get_text_summary(url, question, summ)
    links = get_hyperlinks(url)

    # Limit links to 5
    if len(links) > 5:
        links = links[:5]

    result = f"""Website Content Summary: {summary}\n\nLinks: {links}"""

    return result


def get_text_summary(url, question, summarize):
    text = scrape_text(url)
    summary = summarize(text)
    return """ "Result" : """ + summary


def get_hyperlinks(url):
    link_list = scrape_links(url)
    return link_list


def scrape_text(url):
    response = requests.get(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"})

    # Check if the response contains an HTTP error
    if response.status_code >= 400:
        return "Error: HTTP " + str(response.status_code) + " error"

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


def extract_hyperlinks(soup):
    hyperlinks = []
    for link in soup.find_all('a', href=True):
        hyperlinks.append((link.text, link['href']))
    return hyperlinks


def format_hyperlinks(hyperlinks):
    formatted_links = []
    for link_text, link_url in hyperlinks:
        formatted_links.append(f"{link_text} ({link_url})")
    return formatted_links


def scrape_links(url):
    response = requests.get(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"})

    # Check if the response contains an HTTP error
    if response.status_code >= 400:
        return "error"

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    hyperlinks = extract_hyperlinks(soup)

    return format_hyperlinks(hyperlinks)
