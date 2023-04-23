import json
import logging
import math

import random
import requests
from duckduckgo_search import ddg, ddg_answers, ddg_suggestions, ddg_news
import replicate
import openai
from langchain.agents.agent_toolkits import create_python_agent
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import load_prompt
from langchain.python import PythonREPL
from langchain.tools.python.tool import PythonREPLTool
from transformers import pipeline, AutoModelWithLMHead, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, \
    AutoModelForCausalLM
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import os
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import HfFolder
from huggingface_hub import InferenceApi
import time
from langchain.agents import load_tools, initialize_agent, tool, create_csv_agent, Tool
from langchain import LLMMathChain, SerpAPIWrapper, GoogleSerperAPIWrapper
from langchain.utilities import BashProcess, RequestsWrapper
import torch
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.tools.ifttt import IFTTTWebhook

from toolboxv2 import MainTool, FileHandler, Style, App
from langchain.chains import load_chain
# from langchain.llms import HuggingFacePipeline
# from langchain.llms import HuggingFaceHub, OpenAI, OpenAIChat
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.embeddings import HuggingFaceHubEmbeddings
# from langchain.text_splitter import CharacterTextSplitter

from langchain import PromptTemplate, HuggingFaceHub, LLMChain, ConversationChain, OpenAI
from langchain.llms.openai import OpenAIChat
import tiktoken
from bs4 import BeautifulSoup

# CharacterTextSplitter.from_huggingface_tokenizer(...)
class Singleton(type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls]


import pinecone
import openai
from inspect import signature


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]


def get_text_from_embedding(embedding):
    return openai.Embedding.retrieve(embedding, model="text-embedding-ada-002")["data"][0]["text"]


class PineconeMemory_(metaclass=Singleton):
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
        memory = PineconeMemory_()
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
        if not context or context=="None":
            return f"active memory contains {self.token_in_use} tokens for mor informations Input similar information"
        "memory will performe a vector similarity search using memory"
        relevant_memory = self.memory.get_relevant(context, 10)
        if len(relevant_memory) == 0:
            l = ""
            for i in ddg_suggestions(context)[:3]:
                l += i['phrase']+" ,"
            return f"No data faund in memory try : {l}"

        return "\n".join(relevant_memory)

    def text_add(self, data):
        if self.do_mem:
            return " NO MEMORY Avalabel"
        if not data:
            return

        if isinstance(data, str):
            data = {'data': data, 'token-count': self.mean_token_len+1, 'vector': []}

        print(Style.RED(f"ADD DATA : ColetiveMemory :{len(self.collection)} {data['token-count']} {self.token_in_use=}"))
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

        print(f"Removed ~ {tok} tokens from ObservationMemory tokens in use: {self.tokens}")


class ShortTermMemory:

    memory_data:list[dict] = []
    tokens: int = 0
    max_length: int = 2000
    model_name: str = "text-embedding-ada-002"

    add_to_static: list[dict] = []

    lines_ = []

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

        print(f"Removed ~ {tok} tokens from ShortTermMemory tokens in use: {self.tokens}")


    def clear_to_collective(self):

        max_tokens = self.max_length
        self.max_length = 0
        self.cut()
        self.max_length = max_tokens




    @text.setter
    def text(self, data):
        tok = 0
        for line in data.split('\n'):
            if line not in self.lines_ and len(line) != 0:
                ntok = len(tiktoken.encoding_for_model(self.model_name).encode(line))
                self.memory_data.append({'data':line, 'token-count':ntok, 'vector':[]})
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

    avalabel_modes = ['talk', 'tool', 'conversation']
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
        self.obser_mem: ObservationMemory = ObservationMemory()
        self.tools: dict = {
            "test_tool": {"func": lambda x: x, "description": "only for testing if tools are available", "format": "test_tool(input:str):"}
        }

        self.last_prompt = ""

        self.task_list: list[str] = []
        self.task_list_done: list[str] = []
        self.step_between: str = ""

        self.task_index = 0

        self.token_left = 1000
        self.temperature = 0.16
        self.stop_sequence = ["\n\n"]

    @property
    def task(self):
        if self.step_between:
            task = self.step_between
            self.step_between = ""
            return task
        if len(self.task_list) != 0:
            task = self.task_list[self.task_index]
            return task
        return "Task is done Retorn Summarysation for user"

    @property
    def prompt(self):
        if not self.short_mem.model_name:
            self.short_mem.model_name = self.model_name
        if not self.obser_mem.model_name:
            self.obser_mem.model_name = self.model_name

        tools = ""
        names = []
        for key, value in self.tools.items():
            format_ = value['format'] if 'format' in value.keys() else f"{key}('function input')"
            if format_.endswith("('function input')"):
                value['format'] = format_
            tools += f"\n{key}: {value['description']}\n\t{format_}\n"
            names.append(key)

        task = self.task

        prompt = f"Answer the following questions as best you can. You have access to the following python functions:\n" \
                 f"{tools}\n" \
                 f"\n\nUnder no circumstances are you allowed to output 'Task:'!!!\n\n\n" \
                 f"Personality:'{self.personality}'\n\n" + \
                 f"Goals:'{self.goals}'\n\n" + \
                 f"Capabilities:'{self.capabilities}'\n\n" + \
                 f"Permanet-Memory:\n'{CollectiveMemory().text(context=task)}'\n\n" + \
                 f"Resent Agent response:\n'{self.obser_mem.text}'\n\n"  + \
                 "Placeholders: there are two types of place holders.  " \
                 "\n1 : <information_from_agent> if you see a word enclosed with <> this character you must fill it with information!" \
                 "\n2 : [datime] if you want the system to insert an information return []." \
                 "\nallways fill placeholders of type 1 !!!\n\n " +\
                 "Use the following format:\n\n" +\
                 "Task: the input question you must answer\n" +\
                 "Thought: you should always think about what to do\n" +\
                 "Context: context abaut the task you about to acomplisch\n" +\
                 f"Action: the action to take, should be one of {names}\n" +\
                 f"Exiqute: write to exique action!\n" +\
                 "Observation: the result of the action\n" +\
                 f"... (this Thought/Context/Action/Exiqute/Observation can repeat N times) \n type Question: for user help\n" +\
                 "Thought: I now know the final answer\n"+\
                 "Answer: the final answer to the original input question\n"+\
                 f"\n\nBegin!\n\n" \
                 f"Task:'{task}'\n{self.short_mem.text}\n{'CONTINUE: ' if self.short_mem.tokens > 600 else ''}" \
                 f"\n{self.step_between}" \

        # print("\nprompting:\n\n"+ prompt+ "\n\n----------------------------------------------------------------")
        pl = len(tiktoken.encoding_for_model(self.model_name).encode(prompt))
        self.token_left = self.max_tokens - pl
        if self.token_left < 0:
            print(f"\n----------------------------------------------------prompt token:[{pl}]-------\n{self.token_left}")
            print("Token --")
            self.token_left *= -1
            print(f"Tokens left: {self.token_left}\n----------------------------------------------------------------")
            self.short_mem.max_length = self.token_left
            self.short_mem.cut()
        if self.token_left > 1000:
            self.token_left = 1000
        self.last_prompt = prompt
        return prompt

    @property
    def promptt(self):
        if not self.short_mem.model_name:
            self.short_mem.model_name = self.model_name
        if not self.obser_mem.model_name:
            self.obser_mem.model_name = self.model_name

        short_mem = self.short_mem.text
        obser_mem = self.obser_mem.text

        prompt = f"Goals:{self.goals.replace('{input}', '')}\n" + \
                 f"Capabilities:{self.capabilities.replace('{input}', '')}\n" + \
                 f"Long-termContext:{CollectiveMemory().text(context=self.short_mem.text).replace('{input}', '')}\n" + \
                 f"\nResent Observation:{obser_mem.replace('{input}', '')}"  + \
                 f"Task:{self.task.replace('{input}', '')} \n" \
                 f"SystemTask: when you're done, issue the user's request as a task to yourself. syntax -> 'TASK-FROM-{self.name}: task'\n" + \
                 "input: {input} \n"+f"{self.short_mem.text.replace('{input}', '')}"
        return prompt

    @property
    def prompta(self):
        if not self.short_mem.model_name:
            self.short_mem.model_name = self.model_name
        if not self.obser_mem.model_name:
            self.obser_mem.model_name = self.model_name
        prompt = f"Goals:{self.goals.replace('{input}', '')}\n" + \
                 f"Capabilities:{self.capabilities.replace('{input}', '')}\n" + \
                 f"Long-termContext:{CollectiveMemory().text(context=self.short_mem.text).replace('{input}', '')}\n" + \
                 f"\nResent Observation:{self.obser_mem.text.replace('{input}', '')}"+ \
                 f"Task:{self.task}\n" + \
                 "input: {input}"+ \
                 f"\n{self.short_mem.text.replace('{input}', '')}"
        return prompt


    def __str__(self):

        return f"\n{self.name=}\n{self.mode=}\n{self.model_name=}\n{self.agent_type=}\n{self.max_iterations=}" \
               f"\n{self.verbose=}\n{self.personality[:45]=}\n{self.goals[:45]=}" \
               f"\n{str(self.tools)[:45]=}\n{self.task_list=}\n{self.task_list_done=}\n{self.step_between=}\nshort_mem\n{self.short_mem.info()}\nObservationMemory\n{self.obser_mem.info()}\nCollectiveMemory\n{str(CollectiveMemory())}\n"



class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.1"
        self.name = "isaa"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "VIOLET2"
        self.inference = InferenceApi
        self.config = {'speed_construct_interpret-init': False,
                       'speed_construct_interpret-history': '',
                       'agent_tools-init': False,
                       'genrate_image-init': False,
                       'interpret_agent-init': False,
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
        self.speed_construct_Interpret_agent = speed_construct_Interpret_agent
        self.agent_tools = agent_tools
        self.genrate_image = genrate_image
        self.interpret_agent = interpret_agent
        self.tools = {
            "all": [["Version", "Shows current Version"],
                    ["Run", "Starts Inference"],
                    ["add_api_key", "Adds API Key"],
                    ["login", "Login"],
                    ["new-sug", "Add New Question or Class to Config"],
                    ["run-sug", "Run Huggingface Pipeline"],
                    ["info", "Show Config"],
                    ["lode", "lode models"],
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
        }
        self.app_ = app
        self.print_stream = print
        self.speek = lambda x, *args, **kwargs: x

        FileHandler.__init__(self, "issa.config", app.id if app else __name__)
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=None, color=self.color, on_exit=self.on_exit)

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

        app.inplace_load("isaa_ide", "toolboxv2.mods_dev.")
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

    def search(self):
        @tool("search", return_direct=True)
        def function(query: str) -> str:
            """Using self-ask-with-search to Finde specific answer to komplex questions"""
            try:
                return str(self.agent_tools(query, self.app_, 7))
            except Exception as e:
                return "Das hat leider nicht geklappt ein Fehler" \
                       " ist bei der ausfürgung des Tools aufgetreten Fehler meldung : " + str(e)

        return function

    def bugfix(self):

        def d(s):
            if '```' in s:
                return s.split('```')[1]
            return False

        @tool("bugfix", return_direct=False)
        def function(query: str) -> str:
            """Using 3 steps to find and fix bug in code if necessary"""
            # try:
            # step one find bugs config['speed_construct_interpret-template1'] = "Find bug in code and writ an rport
            # for etch bug code:{input}" speed_construct_Interpret_agent
            # bugs

            # step tow find bugs config['speed_construct_interpret-template1'] = "Find ways to optimise the code
            # and fix the bugs in:{input}" speed_construct_Interpret_agent
            # optimised

            self.config['speed_construct_interpret-template1'] = "Identify any existing bugs in the code and " \
                                                                 "develop a report detailing each issue. " \
                                                                 "Additionally, propose strategies for optimizing " \
                                                                 "the code to improve its efficiency and " \
                                                                 "functionality.\n" \
                                                                 "Implement solutions to fix the " \
                                                                 "identified bugs and improve the overall quality " \
                                                                 "of the code\nCode:{text}"
            code = self.speed_construct_Interpret_agent(query, self.app_, -1)

            self.print("Code: " + code)
            self.print("<" + "====" * 20 + "Code" + "====" * 20 + ">")
            if d(code):
                self.print("return:")
                return d(code)

            # step 3 find bugs config['speed_construct_interpret-template1'] = "produce the final version of the
            # improved code using following information : bug reports {bugs}\n\n ways to fix the code {optimised}
            # \n\n extra data {extra_data}
            # {input}" speed_construct_Interpret_agent
            # final_code

            prompt = "produce the final version of the code " \
                     "using following information : " \
                     f"bug reports \n{code}\n\n " \
                     f"the code: \n{query}"
            final_code = "TRY0"
            try:
                final_code = self.agent_tools(prompt, self.app_, 1)
                self.print("<" + "====" * 20 + "Final_code" + "====" * 20 + ">")
                self.print("Final_code: " + final_code)
                self.print("<" + "====" * 20 + "Final_code" + "====" * 20 + ">")
                if d(final_code):
                    self.print("return:")
                    return d(final_code)
            except ValueError as e:
                for i in range(3):
                    try:
                        final_code = self.agent_tools(prompt + "\n\nError:\n" + str(e), self.app_, 1)
                        self.print("<" + "====" * 20 + "" + str(i) + " Final_code" + "====" * 20 + ">")
                        self.print("Final_code: " + final_code + "\n\nError:\n" + str(e))
                        self.print("<" + "====" * 20 + "" + str(i) + "Final_code" + "====" * 20 + ">")
                        if d(final_code):
                            self.print("return:")
                            return d(final_code)
                    except ValueError as e2:
                        print("Invalid", e2)
            return "Somthing is off \n\n" + code

        # except Exception as e:
        #    return "Das hat leider nicht geklappt ein Fehler" \
        #           " ist bei der ausfürgung des Tools aufgetreten Fehler meldung : " + str(e)

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

                    self.config[f'OpenAI-model-{model}'] = OpenAIChat(model_name=model,
                                                                      openai_api_key=self.config['OPENAI_API_KEY'],
                                                                      streaming=True)
                else:
                    self.config[f'OpenAI-model-{model}'] = OpenAI(model_name=model,
                                                                  openai_api_key=self.config['OPENAI_API_KEY'])

    def get_OpenAI_models(self, name: str):
        if f'OpenAI-model-{name}' not in self.config.keys():
            self.loade_OpenAI_models([name])
        return self.config[f'OpenAI-model-{name}']

    def add_tool(self, name, func, dis, form, tools: dict):

        print(f"\nADDING TOOL:{name} \n__doc__:\n {dis}\n")

        tools.update({name: {"func": func, "description": dis, "format": form}})

    def get_default_agent_config(self, name) -> AgentConfig():
        config = AgentConfig()

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
                return "No New faund"
            response = ""
            for res in responses:
                AgentConfig().step_between
                self.get_agent_config_class("summary").step_between = f"Summaryse the input relatet to {x}"
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
            self.config["self_agent_agents_"] = ["todolist", "fileAgent"]

            def toggel(x):
                x = x.lower()
                if "talk" in x or "conversation" in x:
                    config.mode = "talk"
                    return f"Switched to {config.mode} nowe write the final anser or question to the console"
                if "tools" in x or "execute" in x:
                    config.mode = "tools"
                    return f"Switched to {config.mode}"
                return f"Switched to {config.mode}"

            init_states = [False, False]

            config.name = "isaa"
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
                "reminder": {"func": lambda x: run_agent('reminder', x),
                             "description": "Run agent to reminde user and add, vue notes"
                             , "format": "reminder(<task>)"},
                #"fileAgent": {"func": lambda x: self.run_agent('fileAgent', x),
                #              "description": "Run agent to acces the system to performe file operations"
                #                             "provide detaild informatino. and a task to do",
                #              "format": "fileAgent(task)"}

            }

        if name == "todolist":

            config.name: str = "Planing-Agent"


            def priorisirung(x):
                config.step_between = "take time to gently think about the problem and consider the whole picture."
                if len(config.task_list) != 0:
                    config.step_between = config.task_list[0]
                return run_agent(config.name, x, mode_over_lode="talk")


            config.name: str = "Planing-Agent"
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

            config.name: str = "Reminder-Agent"

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


            config.name: str = "Planing-Agent"
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

        if name == "fileAgent":

            config.name: str = "fileAgent"

            def write(x):
                try:
                    name = x.split(" ")[0]
                except ValueError:
                    return "Invalid format"
                return write_to_file(name, " ".join(x.split(" ")[1:])).replace("isaa-directory\\", "")

            def writea(x):
                try:
                    name = x.split(" ")[0]
                except ValueError:
                    return "Invalid format"
                return append_to_file(name, " ".join(x.split(" ")[1:])).replace("isaa-directory\\", "")

            config.mode: str = "tools"
            config.model_name: str = "text-davinci-003"

            config.agent_type: str = "zero-shot-react-description"
            config.max_iterations: int = 4
            config.verbose: bool = True

            config.personality = """
                  Proactive: The File Agent should be able to identify and handle file-related tasks without constant prompting or supervision.
                  Organized: The File Agent should be efficient in managing files and folders, maintaining a structured approach to file organization and categorization.
                  Adaptable: The File Agent should be able to respond to changes in file structures, formats, and requirements, and adjust its strategies and plans accordingly.
                  Solution-oriented: The File Agent should focus on resolving file-related issues and overcoming obstacles rather than dwelling on difficulties.
                  Communicative: The File Agent should effectively communicate its progress, status, and any issues encountered while managing files.
                  Detail-oriented: The File Agent should pay attention to details, such as file metadata and version history, without losing sight of the overall goals."""

            config.goals = """
                  File Organization: The File Agent should maintain a clear and logical organization of files and folders, ensuring easy access and retrieval.
                  File Management: The File Agent should be able to create, rename, move, delete, and manipulate files and folders as required.
                  Version Control: The File Agent should maintain version history for files, allowing users to revert to previous versions if needed.
                  File Integrity: The File Agent should ensure the integrity of files by preventing unauthorized access, corruption, or loss of data.
                  Compatibility: The File Agent should be able to handle various file formats and ensure compatibility across different platforms and applications.
                  Efficiency: The File Agent should continuously improve its processes and techniques to increase efficiency in managing and organizing files."""

            config.short_mem: ShortTermMemory = ShortTermMemory()
            config.short_mem.max_length = 3500
            config.tools: dict = {
                "read_file": {"func": read_file, "description": "Red file context"},
                "delete_file": {"func": delete_file, "description": "deleat file"},
                "search_file": {"func": search_files,
                                "description": f"search file in directory, currend dir : {search_files('.')}"},
                "write_file": {"func": write, "description": "write to file SYNTAX: file.ending text"},
                "appant_to_files": {"func": writea, "description": "appnd to file to file SYNTAX: file.ending text"},
            }

            config.task_list: list[str] = ["Erfülle die Aufgae in so wenigen Schritten und so Bedacht wie Möglich"]

        if name == "summary":
            config.name: str = "Summary-Agent"


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

            config.name: str = "Search-agent"

            config.mode: str = "tools"
            config.model_name: str = "text-davinci-003"

            config.agent_type: str = "zero-shot-react-description"
            config.max_iterations: int = 6
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
            config.tools: dict = {
                "browse_url": {"func": lambda x: browse_url(x), "description": "browse web page via URL syntax <url>|<qustion>"},
                "search_text": {"func": lambda x: search_text(x), "description": "Use Duck Duck go to search the web systax <qustion>"},
                "search_news": {"func": lambda x: search_news(x), "description": "Use Duck Duck go to search the web for new get time"
                                                                    "reladet data systax <qustion>"},
                "memory_search": {"func": lambda x: memory_search(x), "description": "Serch for simmilar memory imput <context>"},
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

            return config

        if name.startswith("chain_search"):

            config.name: str = name

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
                    "Intermediate Answer": {"func": search_text, "description": "Use Duck Duck go to search the web systax <qustion>"},
                }

            if name.endswith("_url"):

                config.tools: dict = {
                    "Intermediate Answer": {"func": browse_url, "description": "browse web page via URL syntax <url>|<qustion>"},
                }
            if name.endswith("_memory"):

                config.tools: dict = {
                    "Intermediate Answer": {"func": memory_search, "description": "Serch for simmilar memory imput <context>"},
                }

            config.task_list: list[str] = ["Erfülle die Aufgae in so wenigen Schritten und so Bedacht wie Möglich"]

        try:
            if not os.path.exists(self.observation_term_mem_file+config.name):
                with open(self.observation_term_mem_file+config.name, "a") as f:
                    f.write("[]")

            with open(self.observation_term_mem_file+config.name, "r") as f:
                mem = f.read()

                if mem:

                        config.obser_mem.memory_data = eval(mem)
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

            self.print(f"\nINIT AGENT: {config.name} {config.mode}\n")

            return config



        if f'agent-config-{agent_name}' in self.config.keys():
            config = self.config[f'agent-config-{agent_name}']
            self.print(f"\nUsing AGENT: {config.name} {config.mode}\n")
        else:
            config = AgentConfig()

        return config

    def process_compleation(self, text, config=AgentConfig()):

        if len(config.task_list) == 0:
            config.step_between = text

        model_name = config.model_name

        if config.model_name.startswith('gpt-3'):
            model_name = "text-davinci-003"

        ret = openai.Completion.create(
            model=model_name,
            prompt=config.prompt,
            max_tokens=config.token_left,
            temperature=config.temperature,
            n=1,
            stream=True,
            logprobs=3,
            stop=config.stop_sequence
        )

        return ret

    def use_tools(self, agent_text, config=AgentConfig()):


        if agent_text.startswith("Exiqute:"):
            print("Use Exiq")
            agent_text = agent_text.replace("Exiqute:", "").strip()
            for key, value in config.tools.items():
                if key+'(' in agent_text:
                    return True, key, agent_text

        #if line.startswith('Action: '):
        #        if key in line and not use_tool:
        #            use_tool = True
        #            name = key
        #            return use_tool, name, f"{name}({config.short_mem.text})"

        return False, "", ""

    def test_task_done(self, agent_text, config=AgentConfig()):

        done = False

        for line in agent_text.split("\n"):

            if line.startswith("Answer:"):
                done = True

            if line.startswith("Thought: I now know the final answer"):
                done = True

        return done

    def run_tool(self, command, config=AgentConfig()):

        # cleanup command

        function_name = command.split("(")[0].strip()

        args = command.split("(")[1]
        args = "".join(args)
        args = args[:-1]

        tool = config.tools[function_name]

        sig = signature(tool['func'])
        args_len = len(sig.parameters)
        self.print(f"Runing : {function_name}")
        self.print(f"signature : {sig} | fuction args len : {args_len} | providet nubers of args {len(args.split(','))}")

        args = args.split(",")

        observation = "Problem running function"

        if args_len==0:
            observation = tool['func']()

        if args_len==len(args):
            observation = tool['func'](*args)

        if args_len==1 and len(args) > 1:
            observation = tool['func'](" ".join(args))

        if isinstance(observation, dict):
            #try:
            observation = self.summarize_dict(observation, config)

        config.short_mem.text = observation

        if not observation:
            observation = "Problem running function try run with mor detaiels"


        return observation

    def run_agent(self, name: str, text: str, retrys=1, mode_over_lode=False):

        config = self.get_agent_config_class(name)

        if mode_over_lode:
            mode_over_lode, config.mode = config.mode, mode_over_lode

        # print("AGENT:CONFIG"+str(config)+"ENDE\n\n\n")

        if not text:
            text = "View the Momory section to get more information about your task." \
                   "If you still got no information use the memory function."

        out = "Invalid configuration\n"
        system = ""

        if self.config[f'agent-config-{name}'].mode == "talk":
            prompt = PromptTemplate(
                input_variables=["input"],
                template=config.prompta,
            )
            out = LLMChain(prompt=prompt, llm=self.get_OpenAI_models(config.model_name)).run(text)
        elif self.config[f'agent-config-{name}'].mode == "tools":
            prompt = PromptTemplate(
                input_variables=["input"],
                template=config.prompta,
            )

            tools = []

            for tool_name in config.tools.keys():

                def ovservation(x):
                    out = config.tools[tool_name]["func"](x)
                    config.obser_mem.text = out
                    return out

                fuc = lambda x: ovservation(x)

                tools.append(
                    Tool(
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
                template=config.prompta,
            )
            out = LLMChain(prompt=prompt,llm=self.get_OpenAI_models(config.model_name)).run(text)
        elif self.config[f'agent-config-{name}'].mode == "free":
            min_typing_speed, max_typing_speed, sant = 0.02,0.01,1
            for line in self.process_compleation(text, config):

                text = line['choices'][0]['text']
                for i, word in enumerate(text):
                    self.print_stream(word, end="", flush=True)
                    typing_speed = random.uniform(min_typing_speed, max_typing_speed)
                    time.sleep(typing_speed)
                    # type faster after each word
                    min_typing_speed = min_typing_speed * 0.07
                    max_typing_speed = max_typing_speed * 0.06

                if text:
                    out += str(text)

                if len(out.split('\n')) == sant:
                    sant += 1
                    if len(out.split('\n')) == 1:
                        continue
                    sp_santance = out.split('\n')[-2]

                    # print(sp_santance)

                    self.speek(sp_santance)

                    config.short_mem.text = sp_santance

                    use_tool, name, command_ = self.use_tools(sp_santance, config)
                    task_done = self.test_task_done(sp_santance, config)

                    if use_tool:
                        self.print(f"Using-tools: {name} " + str(use_tool))
                        self.print(f"command: {command_} ")
                        self.speek(f"Isaa is using {name}", 2)
                        ob = self.run_tool(command_, config)
                        config.short_mem.text = "Function response: " + ob
                        config.obser_mem.text = "Observation: " + ob
                        out += "\nObservation: " + ob

                        with open(self.observation_term_mem_file+config.name, "w") as f:
                            try:
                                f.write(str(config.obser_mem.memory_data))
                            except UnicodeEncodeError:
                                print_("Memory not encoded properly")

                        self.speek(ob, speak_text=True, vi=1)
                        use_tool = False

                    if task_done:  # new task

                        self.speek("Ist die Aufgabe abgeschlossen?")
                        config.short_mem.clear_to_collective()
                        if len(config.task_list) > config.task_index:
                            config.task_index += 1

                        return out

        #except NameError and Exception as e:
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


    def mas_text_summaries(self, text, s_call=True, min_length=1600):
        len_text= len(text)
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
            cap = 5000
            max_length = 80

        summarysation = pipeline("summarization")
        summary = ""

        while len(text) > cap:
            chucks.append(text[:cap])
            text = text[cap:]
            self.print(f"SYSTEM: TEXT lne : {len(text)}")
        if len(text) < max_length:
            chucks[-1] += "\n" + text
        else:
            chucks.append(text)
        self.print(f"SYSTEM: chucks to summary: {len(chucks)} cap : {cap}")
        #try:
        for chck in chucks:
            summary_chucks += summarysation(chck, min_length=10, max_length=max_length)[0]['summary_text'] + "\n"
            self.print(f"SYSTEM: all summary_chucks : {len(summary_chucks)}")
        summary = summarysation(summary_chucks, min_length=10, max_length=max_length)[0]['summary_text']
        self.print(f"SYSTEM: final summary from {len_text} -> {len(summary)} compressed {len_text/len(summary):.2f}X\n")
        #except Exception as e:
        #    self.print(f"SYSTEM: error in summary {e}")
        #    if s_call:
        #        self.print("SYSTEM: retrying")
        #        summary = self.mas_text_summaries(summary_chucks, s_call=False)
        #    else:
        #        summary = ""
        #        if max_length <= 20:
        #            summary = text

        return summary


    def summarize_dict(self, input_dict, config: AgentConfig):
        input_str = input_dict['input']
        output_str = input_dict['output']
        intermediate_steps = input_dict['intermediate_steps']

        summary = f"Output: {output_str}\nIntermediate steps:\n"

        action = ""
        tool_input = ""
        log = ""
        result = ""
        i = 0
        for step in intermediate_steps:
            if isinstance(step, tuple):
                action += f"\naction {i}" + str(step[0].tool)
                tool_input += f"\ntool_input {i}" + str(step[0].tool_input)
                result  += f"\nlog {i}" + str(step[1])
                summary += f"- {i} Action: {action}\n  Tool Input: {tool_input}\n  Result: {result}\n"
            else:
                summary += f"-{i} {step}\n"
            i += 1


        if summary:
            st = config.short_mem.text
            text = f"{config.short_mem.text} \n{summary}"

            if len(text) > 4000:

                self.print(f"SYSTEM : start mas_text_summaries : {len(summary)}")

                self.print(f"\taction : {len(action)}" )
                action_summary = self.mas_text_summaries(st+"\nResent Actions\n"+action)
                self.print(f"\taction_summary : {len(action_summary)}")

                self.print(f"\ttool_input : {len(tool_input)}")
                tool_input_summary = self.mas_text_summaries(st+"\nResent Agent-Inputs\n"+tool_input)
                self.print(f"\ttool_input_summary : {len(tool_input_summary)}")

                self.print(f"\tresult : {len(result)}")
                result_summary = self.mas_text_summaries(st+"\nResent Agent-Results\n"+result)
                self.print(f"\tresult_summary : {len(result_summary)}")

                summary = self.mas_text_summaries(action_summary+"\n"+
                                                  tool_input_summary+"\n"+
                                                  result_summary)
            else:
                summary = self.mas_text_summaries(text)

            config.obser_mem.text = summary

        return output_str


def buld_disiontree(app):
    pass

def execute_disiontree(app):
    pass


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




def informatin_agent(input, role, prompt):
    """an Angent Thet Ansers User qustens : The Agent Can Serve The web, and loock up infrmatins abaut the users
    (Calender, TodoLists, Goals, Messages).
    The Agent Helps The user with informations. If The Agent wont to do an action The Agent will Wirte <ACTION>"""

    # Constructing the prompt
    prompt = PromptConfig()


def generate_new_tool_for_agent(app, tool_discription):
    """An Agent that will generate a new tool for the given description"""

    # 1. Bild A Detalt step by step Plan on what functions features the tool has
    prompt = "U ar an Planig Expert" \
           "Bild A Tool for given description {text}. First Bild A Detalt step by step Plan on what" \
           f" functions features the tool has."

    prompt = role+" {text}"
    mod.config['speed_construct_interpret-template1'] = prompt
    plan = speed_construct_Interpret_agent(tool_discription, app, -1)


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


def initialize_scia(app: App, task):
    app.logger.info(f'initializing scia {task}')
    mod = get_tool(app)
    mod.config['speed_construct_interpret-init-in'] = task

    if not mod.config['speed_construct_interpret-init']:
        #    text-davinci-003
        #    text-curie-001
        #    text-babbage-001
        #    text-ada-001
        mod.config['speed_construct_interpret-model'] = OpenAI(model_name='text-babbage-001',
                                                               openai_api_key=mod.config['OPENAI_API_KEY'])
        mod.config['speed_construct_interpret-model-davinci'] = OpenAI(model_name='text-davinci-003',
                                                                       openai_api_key=mod.config['OPENAI_API_KEY'])
        mod.config['speed_construct_interpret-model-turbo'] = OpenAI(model_name='gpt-3.5-turbo',
                                                                     openai_api_key=mod.config['OPENAI_API_KEY'])

    mod.config['speed_construct_interpret-init'] = True

    if 'speed_construct_interpret-template0' not in mod.config.keys():
        mod.config['speed_construct_interpret-template0'] = """Isaa: Hallo! Ich bin Isaa, der digitale Assistent,
        der entwickelt wurde, um dem nutzer helfen.

        Aufgabe: {Aufgabe}

        {text}
        """

    if task == 0:
        prompt = PromptTemplate(
            input_variables=["text"],
            template=mod.config['speed_construct_interpret-template0'],
        )
        mod.config['speed_construct_interpret-llm_chain'] = LLMChain(prompt=prompt,
                                                                     llm=mod.config['speed_construct_interpret-model'])
    if task == -1:
        prompt = PromptTemplate(
            input_variables=["text"],
            template=mod.config['speed_construct_interpret-template1'],
        )
        mod.config['speed_construct_interpret-llm_chain'] = LLMChain(prompt=prompt,
                                                                     llm=mod.config['speed_construct_interpret-model'
                                                                                    '-turbo'])

    if task == 1:
        if not os.path.exists("./isaa_data/default-prompt.json"):
            save_prompt_file(app=app, command=None)

        prompt = load_prompt("./isaa_data/default-prompt.json")
        mod.config['speed_construct_interpret-llm_chain'] = ConversationChain(prompt=prompt,
                                                                              verbose=True,
                                                                              memory=mod.config[
                                                                                  'speed_construct_interpret-memory'],
                                                                              llm=mod.config[
                                                                                  'speed_construct_interpret-model-turbo'])
    if task == 2:
        if not os.path.exists("./isaa_data/tools-prompt.json"):
            save_prompt_file(app=app, command=None)

        prompt = load_prompt("./isaa_data/tools-prompt.json")
        mod.config['speed_construct_interpret-llm_chain'] = ConversationChain(prompt=prompt,
                                                                              verbose=True,
                                                                              llm=mod.config[
                                                                                  'speed_construct_interpret-model-davinci'],
                                                                              memory=mod.config[
                                                                                  'speed_construct_interpret-memory']
                                                                              )

    if task == 3:
        mod.config['speed_construct_interpret-llm_chain'] = load_summarize_chain(llm=mod.config[
            'speed_construct_interpret-model-davinci'],
                                                                                 chain_type='map_reduce',
                                                                                 verbose=True)

    if task == 4:
        if not os.path.exists("./isaa_data/generator-prompt.json"):
            save_prompt_file(app=app, command=[{"name": "generator-prompt"}])

        prompt = load_prompt("./isaa_data/generator-prompt.json")
        mod.config['speed_construct_interpret-llm_chain'] = LLMChain(prompt=prompt,
                                                                     llm=mod.config['speed_construct_interpret-model'
                                                                                    '-turbo'])
    if task == 5:
        if not os.path.exists("./isaa_data/file-prompt.json"):
            save_prompt_file(app=app, command=[{"name": "file-prompt"}])

        prompt = load_prompt("./isaa_data/file-prompt.json")
        mod.config['speed_construct_interpret-llm_chain'] = LLMChain(prompt=prompt,
                                                                     llm=mod.config['speed_construct_interpret-model'
                                                                                    '-davinci'])


def speed_construct_Interpret_agent(text, app: App, task=0, memory=False):
    mod = get_tool(app)
    if memory:
        mod.config['speed_construct_interpret-memory'] = memory
    if not mod.config['speed_construct_interpret-init']:
        initialize_scia(app, task)
    if mod.config['speed_construct_interpret-init-in'] != task:
        initialize_scia(app, task)

    return mod.config['speed_construct_interpret-llm_chain'].run(text)


def initialize_ia(app: App, task):
    app.logger.info(f'initializing ia {task}')
    mod = get_tool(app)
    mod.config['interpret_agent-init-in'] = task

    if not mod.config['interpret_agent-init']:
        #    text-davinci-003
        #    text-curie-001
        #    text-babbage-001
        #    text-ada-001
        mod.config['interpret_agent-model'] = OpenAI(model_name='text-babbage-001',
                                                     openai_api_key=mod.config['OPENAI_API_KEY'])
        mod.config['interpret_agent-davinci'] = OpenAI(model_name='text-davinci-003',
                                                       openai_api_key=mod.config['OPENAI_API_KEY'])
        mod.config['interpret_agent-turbo'] = OpenAI(model_name='gpt-3.5-turbo',
                                                     openai_api_key=mod.config['OPENAI_API_KEY'])

    mod.config['interpret_agent-init'] = True

    if task == 0:
        bash = BashProcess()
        search = SerpAPIWrapper(serpapi_api_key=mod.config['SERP_API_KEY'])
        requests = RequestsWrapper()
        wolfram = WolframAlphaAPIWrapper(wolfram_alpha_appid=mod.config['WOLFRAM_ALPHA_APPID'])
        python_repl = PythonREPL()
        key = mod.config["IFTTTKey"]
        command = 'Isaa-spotify-start'
        url = f"https://maker.ifttt.com/trigger/{command}/json/with/key/{key}"
        tool_play_s = IFTTTWebhook(name="Spotify-play", description="spotify play musik", url=url)

        prompt = PromptTemplate(
            input_variables=["text"],
            template="""beziehe zum lösen der aufgabe den vergangen Kontext mit ein!\n
            Aufgabe: {text} """,
        )

        # Construct the agent. We will use the default agent type here.
        # See documentation for a full list of options.

        #     toolbox_interface toolbox_information_interface generate_image
        tools = [
            Tool(
                name="Talk",
                func=mod.talk(),
                description="Tool zum bilden von Antworten die der Nutzer erhalten Soll "
                            "-> Bitte antworte uns auf de"
            ),
            Tool(
                name="Ask user",
                func=mod.user_input(),
                description="Tool zum fragen des Users"
            ),
            Tool(
                name="kontext",
                func=mod.kontext(),
                description="Tool zum Besseren Beantworten der Frage Holt Kontext aus vorherigen Fragen"
            ),
            Tool(
                name="web-search",
                func=search.run,
                description="Tool zum finden von Information, Das Tool Braucht Kontext"
                            " -> Das tool Kommuniziert in "
                            "natürlicher Sprache, ausschließlich in Englisch"
            ),
            # Tool(
            #     name="multi-web-search",
            #     func=mod.search(),
            #     description="Tool zum finden von Information wenn diese aufeinander aufbauen, Das Tool Braucht Kontext"
            #                 " -> Das tool Kommuniziert in "
            #                 "natürlicher Sprache, ausschließlich in Englisch"
            # ),
            # Tool(
            #     name="Spotify Musik abspielen",
            #     func=tool_play_s.run,
            #     description="Spielt nur dann musik ab wenn Spotify offen ist"
            # ),
            # Tool(
            #     name="toolbox_interface",
            #     func=mod.toolbox_interface(),
            #     description="Tool zum Interagieren mit der ToolBox Platform -> module-name function-name args"
            #                 "ist valide Input. weitere information findest du unter toolbox_information_interface"
            # ),
            # Tool(
            #     name="toolbox_information_interface",
            #     func=mod.toolbox_information_interface(),
            #     description="Tool zum Interagieren mit der ToolBox Platform zum finden"
            #                 " verschiedener Module und deren Funktionen -> get-mod-list "
            # ),
            Tool(
                name="Image",
                func=mod.generate_image(),
                description="Tool zum generiren und anzeigen von bildern. Beschreibe das bild bezhie dich auf diese "
                            "9 punkte bei der beseeching. Subject, Medium, Style, ArtStyle, Perspective, Resolution, "
                            "Additional details, Color, Lighting, Remarks. auf englisch :"
            ),
            # Tool(
            #     name="PythonREPL",
            #     func=python_repl.run,
            #     description="Tool zum Evaluation von Python code -> This interface will only return things that are "
            #                 "printed - therefor, if you want to use it to calculate an answer, make sure to have it "
            #                 "print out the answer."
            # ),
            # Tool(
            #     name="wolfram",
            #     func=wolfram.run,
            #     description="Tool zum Lösen von Mathematics, Physics and co Aufgeben und Beschaffung"
            #                 " von Wissenschaftlichen informationen"
            #                 " -> Das tool Kommuniziert in "
            #                 "natürlicher Sprache, ausschließlich in Englisch Z.B waht is 2/x = 15"
            # ),
            # Tool(
            #     name="fileEditor",
            #     func=mod.fileEditor(),
            #     description="Tool zum Interagieren mit der AI-IDE, System Nützlich zum bearbeiten von dateien / "
            #                 "Application"
            #                 "-> Stelle eine Anfrage zum Arbeiten mir dateien"
            # ),
            # Tool(
            #    name="Bash",
            #    func=bash.run,
            #    description="Tool zum Interagieren mit der Konsole, zum finden von System-informationen"
            #                "-> bash commands sind valide Input."
            # ),
            # Tool(
            #     name="Request",
            #     func=requests.get,
            #     description="Tool zum Beschaffen von Website Spezifischen Information"
            #                 "Nützlich Zum finden von Website Spezifischen Information"
            #                 " -> Die Genaue URL der Website "
            #                 "ist der Input."
            # ),
        ]
        mod.config['interpret_agent-agent0'] = initialize_agent(tools, prompt=prompt,
                                                                llm=mod.config['interpret_agent-turbo'],
                                                                agent="zero-shot-react-description", verbose=True,
                                                                return_intermediate_steps=True, max_iterations=6)

    if task == 1:
        bash = BashProcess()
        search = SerpAPIWrapper(serpapi_api_key=mod.config['SERP_API_KEY'])
        requests = RequestsWrapper()
        wolfram = WolframAlphaAPIWrapper(wolfram_alpha_appid=mod.config['WOLFRAM_ALPHA_APPID'])
        python_repl = PythonREPL()
        key = mod.config["IFTTTKey"]
        command = 'Isaa-spotify-start'
        url = f"https://maker.ifttt.com/trigger/{command}/json/with/key/{key}"
        tool_play_s = IFTTTWebhook(name="Spotify-play", description="spotify play musik", url=url)

        prompt = PromptTemplate(
            input_variables=["text"],
            template="""beziehe zum lösen der aufgabe den vergangen Kontext mit ein!\n
            Aufgabe: {text} """,
        )

        # Construct the agent. We will use the default agent type here.
        # See documentation for a full list of options.

        #     toolbox_interface toolbox_information_interface generate_image
        tools = [
            Tool(
                name="Talk",
                func=mod.talk(),
                description="Tool zum bilden von Antworten die der Nutzer erhalten Soll "
                            "-> Die frage des Nutzers ist der Input."
            ),
            Tool(
                name="kontext",
                func=mod.kontext(),
                description="Tool zum Besseren Beantworten der Frage Holt Kontext aus vorherigen Fragen"
            ),
            Tool(
                name="fileEditor",
                func=mod.fileEditor(),
                description="Tool zum Interagieren mit der AI-IDE, System Nützlich zum bearbeiten von dateien / "
                            "Application"
                            "-> Stelle eine Anfrage zum Arbeiten mir dateien"
            ),
        ]
        mod.config['interpret_agent-agent1'] = initialize_agent(tools, prompt=prompt,
                                                                llm=mod.config['interpret_agent-turbo'],
                                                                agent="zero-shot-react-description", verbose=True,
                                                                return_intermediate_steps=True, max_iterations=6)

    if task == 2:
        # Steps:
        #   1. -> Get data
        #   2. -> get relevant information's
        #   3. -> Think about optimisations staps
        #       3.1 -> Fehler fangen
        #       3.2 -> Code schneller machen
        #       3.3 -> Besser lesbar
        #       3.4 -> Effizienter
        #       3.5 -> Fehler behebung
        #   4. -> Optimise code
        #   5. -> save code
        #   6. -> repeat

        prompt = PromptTemplate(
            input_variables=["text"],
            template="""Follow these steps 5 to optimize code and fix bugs.
            fix bugs. Start collecting data and relevant information before you start thinking about
            optimizations. Break the optimization step into subtasks to find bugs,
            make code faster, more readable, more efficient, and fix bugs.
            Finally, optimize your code and save it. Please make sure that each step is explained clearly and
            explained clearly and precisely. first use fileEditor to read the file | compute one task after the other \n Aufgabe: {text} """,
        )

        # Construct the agent. We will use the default agent type here.
        # See documentation for a full list of options.

        #     toolbox_interface toolbox_information_interface generate_image
        tools = [
            Tool(
                name="Talk",
                func=mod.talk(),
                description="Tool zum bilden von Antworten die der Nutzer erhalten Soll "
                            "-> Die frage des Nutzers ist der Input."
            ),
            Tool(
                name="kontext",
                func=mod.kontext(),
                description="Tool zum Besseren Beantworten der Frage Holt Kontext aus vorherigen Fragen"
            ),
            Tool(
                name="User-help",
                func=mod.user_input(),
                description="Tool zum Schreiben mit dem User nützlich wenn du weitere informationen brauchst"
            ),
            Tool(
                name="fileEditor",
                func=mod.fileEditor(),
                description="Tool zum Interagieren mit der AI-IDE, System Nützlich zum bearbeiten von dateien / "
                            "Application"
                            "-> Stelle eine Anfrage zum Arbeiten mir dateien"
            ),
        ]
        mod.config['interpret_agent-agent2'] = initialize_agent(tools, prompt=prompt,
                                                                llm=mod.config['interpret_agent-turbo'],
                                                                agent="zero-shot-react-description", verbose=True,
                                                                return_intermediate_steps=True, max_iterations=10)

    if task == 3:
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""Deine aufgabe ist es code zu verbessern in 5 schritte lese die angefragte datei frage wen du
            welche hast, Bugfix, neue datei erstellen isaa-improved-file und zum schluß den code speichern\nAufgabe: {text} """,
        )

        app.inplace_load("isaa_ide", "toolboxv2.mods_dev.")
        app.new_ac_mod("isaa_ide")
        create = app.AC_MOD.create
        read = app.AC_MOD.read
        process_input = app.AC_MOD.process_input  # search copy insert
        app.new_ac_mod("isaa")

        tools = [
            Tool(
                name="Bugfix",
                func=mod.bugfix(),
                description="Tool zum bugfixes."
            ),
            Tool(
                name="Create",
                func=mod.app_wrapper("Create", create, False),
                description="erstellt eine datei oder ordner in einem gewünschtem path -> nur den path eingeben"
            ),
            Tool(
                name="Write",
                func=mod.app_wrapper("Write", process_input, True),
                description="zum schreiben in eine datei. dafür beachte folgende syntax insert start end text"
            ),
            Tool(
                name="Read",
                func=mod.app_wrapper("Read", read, False),
                description="zum schreiben in eine datei. dafür beachte folgende syntax insert start end text"
            ),
            Tool(
                name="User-help",
                func=mod.user_input(),
                description="Tool zum Schreiben mit dem User nützlich wenn du weitere informationen brauchst oder "
                            "nicht weiter weißt"
            ),
        ]
        mod.config['interpret_agent-agent3'] = initialize_agent(tools, prompt=prompt,
                                                                llm=mod.config['interpret_agent-turbo'],
                                                                agent="zero-shot-react-description", verbose=True,
                                                                return_intermediate_steps=True, max_iterations=8)


def interpret_agent(text, app: App, task=0, memory=False):
    mod = get_tool(app)
    if memory:
        mod.config['interpret_agent-memory'] = memory
    if not mod.config['interpret_agent-init']:
        initialize_ia(app, task)
    if mod.config['interpret_agent-init-in'] != task:
        initialize_ia(app, task)

    return mod.config[f'interpret_agent-agent{task}']({'input': text})


def initialize_at(app: App, task):
    app.logger.info(f'initializing at {task}')
    mod = get_tool(app)
    mod.config['agent_tools-init-in'] = task

    if not mod.config['agent_tools-init']:
        #    text-davinci-003
        #    text-curie-001
        #    text-babbage-001
        #    text-ada-001
        mod.config['agent_tools-model'] = OpenAI(temperature=0, model_name='text-babbage-001',
                                                 openai_api_key=mod.config['OPENAI_API_KEY'])

    if task >= 1:
        #    text-davinci-003
        #    text-curie-001
        #    text-babbage-001
        #    text-ada-001
        mod.config['agent_tools-davinci'] = OpenAI(model_name='text-davinci-003',
                                                   openai_api_key=mod.config['OPENAI_API_KEY'])

    mod.config['agent_tools-init'] = True

    if task == 0:
        # `zero - shot - react - description`
        # `react - docstore`
        # `self - ask -
        # with-search`
        # `conversational - react - description`
        # If
        # None and agent_path is also
        # None, will
        # default
        # to
        # `zero - shot - react - description`.

        mod.config['agent_tools-tools0'] = load_tools([
            "terminal",
        ], llm=mod.config['agent_tools-model'])
        mod.config['agent_tools-agent0'] = initialize_agent(tools=mod.config['agent_tools-tools0'],
                                                            llm=mod.config['agent_tools-model'],
                                                            agent="zero-shot-react-description",
                                                            verbose=True)
    if task == 1:
        mod.config['agent_tools-tools1'] = PythonREPLTool()
        mod.config['agent_tools-agent1'] = create_python_agent(tool=mod.config['agent_tools-tools1'],
                                                               llm=mod.config['agent_tools-model'],
                                                               agent="zero-shot-react-description",
                                                               verbose=True)
    if task == 2:
        tools = [

        ]
        # Construct the agent. We will use the default agent type here.
        # See documentation for a full list of options.
        mod.config['agent_tools-agent2'] = initialize_agent(tools, llm=mod.config['agent_tools-davinci'],
                                                            agent="zero-shot-react-description", verbose=True)

    if task == 3:  # _LLM_TOOLS
        mod.config['agent_tools-tools3'] = load_tools(["pal-math",
                                                       "pal-colored-objects"
                                                       "llm-math",
                                                       "open-meteo-api",
                                                       ], llm=mod.config['agent_tools-model'])
        mod.config['agent_tools-agent3'] = initialize_agent(tools=mod.config['agent_tools-tools3'],
                                                            llm=mod.config['agent_tools-davinci'],
                                                            agent="zero-shot-react-description",
                                                            verbose=True)

    if task == 4:  # _EXTRA_LLM_TOOLS
        mod.config['agent_tools-tools4'] = load_tools(["wolfram-alpha",
                                                       "serpapi",
                                                       ], llm=mod.config['agent_tools-turbo'],
                                                      wolfram_alpha_appid=mod.config['WOLFRAM_ALPHA_APPID'],
                                                      serpapi_api_key=mod.config['SERP_API_KEY']
                                                      )
        mod.config['agent_tools-agent4'] = initialize_agent(tools=mod.config['agent_tools-tools4'],
                                                            llm=mod.config['agent_tools-model'],
                                                            agent="zero-shot-react-description",
                                                            verbose=True)

    if task == 5:  # CV"
        mod.config['agent_tools-agent5'] = create_csv_agent(mod.config['agent_tools-model'],
                                                            './isaa_data/data.csv',
                                                            verbose=True)
    if task == 6:
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""Formuliere die Frage auf Englisch aus : {text} """,
        )
        tools = load_tools(["wolfram"], llm=mod.config['agent_tools-davinci'])
        mod.config['agent_tools-agent6'] = initialize_agent(tools=tools,
                                                            llm=mod.config['agent_tools-davinci'],
                                                            agent="self-ask-with-search",
                                                            verbose=True, prompt=prompt)

    if task == 7:
        search = SerpAPIWrapper(serpapi_api_key=mod.config['SERP_API_KEY'])

        tools = [
            Tool(
                name="Intermediate Answer",
                func=search.run,
                description="Tool zum suchen von informationen"
            ),
        ]
        mod.config['agent_tools-agent7'] = initialize_agent(tools,
                                                            llm=mod.config['interpret_agent-davinci'],
                                                            agent="self-ask-with-search", verbose=True,
                                                            max_iterations=4, return_intermediate_steps=True)


@tool("Cmd", return_direct=False)
def cmd(query: str):
    """Using Windows Command Line Interface"""
    print(query)
    os.system(query)


def agent_tools(text, app: App, task=0):
    mod = get_tool(app)
    if not mod.config['agent_tools-init']:
        initialize_at(app, task)
    if mod.config['agent_tools-init-in'] != task:
        initialize_at(app, task)

    return mod.config[f'agent_tools-agent{task}'].run(text)


def generate_prompt_file(command=None, app: App = App()):
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

    if not os.path.exists(f"./isaa_data"):
        os.makedirs(f"./isaa_data")

    with open(f"./isaa_data/{name}.json", 'w') as f:
        generate_prompt_file(command, app)
        string_data = mod.config[name]
        f.write(json.dumps(string_data))


def conversational_prompt_chain(app):
    from langchain.prompts import load_prompt
    from langchain.chains import ConversationChain
    mod = get_tool(app)
    llm = mod.config['speed_construct_interpret-model']
    prompt = load_prompt('lc://prompts/conversation/<file-name>')
    chain = ConversationChain(llm=llm, prompt=prompt)


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
    try:
        inputs = eval(prompt)
    except ValueError and SyntaxError:
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

    print(f"Generating Image")
    images = genrate_image(inputs, app)

    print(f"Showing Images")

    show_image_in_internet(images)


from toolboxv2.mods_dev import BROWSER  # TODO: live -> from toolboxv2.mods import BROWSER
def show_image_in_internet(images, browser=BROWSER):
    if isinstance(images, str):
        images = [images]
    for image in images:
        os.system(f'start {browser} {image}')


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

# "What time is the appointment?"
# "Appointment location?"

def get_command(response):
    from toolboxv2.util.agent.scripts.json_parser import fix_and_parse_json
    try:
        response_json = fix_and_parse_json(response)

        if "command" not in response_json:
            return "Error:" , "Missing 'command' object in JSON"

        command = response_json["command"]

        if "name" not in command:
            return "Error:", "Missing 'name' field in 'command' object"

        command_name = command["name"]

        # Use an empty dictionary if 'args' field is not present in 'command' object
        arguments = command.get("args", {})

        if not arguments:
            arguments = {}

        return command_name, arguments
    except json.decoder.JSONDecodeError:
        return "Error:", "Invalid JSON"
    # All other errors, return "Error: + error message"
    except Exception as e:
        return "Error:", str(e)



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
    response = requests.get(url, headers={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"})

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
    response = requests.get(url, headers={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"})

    # Check if the response contains an HTTP error
    if response.status_code >= 400:
        return "error"

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    hyperlinks = extract_hyperlinks(soup)

    return format_hyperlinks(hyperlinks)




import os
import os.path

# Set a dedicated folder for file I/O
working_directory = "isaa-directory"

if not os.path.exists(working_directory):
    os.makedirs(working_directory)


def safe_join(base, *paths):
    new_path = os.path.join(base, *paths)
    norm_new_path = os.path.normpath(new_path)

    if os.path.commonprefix([base, norm_new_path]) != base:
        return "Agent is not allowed to use .. to get current directory contet input /"

    return norm_new_path


def read_file(filename):
    try:
        filepath = safe_join(working_directory, filename)
        with open(filepath, "r") as f:
            content = f.read()
        print(filepath)
        return content
    except Exception as e:
        return "Error: " + str(e).replace("isaa-directory\\", "")


def write_to_file(filename, text):
    try:
        filepath = safe_join(working_directory, filename)
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filepath, "w") as f:
            f.write(text)
        print(filepath)
        return "File written to successfully."
    except Exception as e:
        return "Error: " + str(e).replace("isaa-directory\\", "")


def append_to_file(filename, text):
    try:
        filepath = safe_join(working_directory, filename)
        with open(filepath, "a") as f:
            f.write(text)
        return "Text appended successfully."
    except Exception as e:
        return "Error: " + str(e).replace("isaa-directory\\", "")


def delete_file(filename):
    if input(f"Delete file by Agent file name : {filename} \nto confirm type:y") in ["yes","y"]:
        try:
            filepath = safe_join(working_directory, filename)
            os.remove(filepath)
            return "File deleted successfully."
        except Exception as e:
            return "Error: " + str(e)
    return "Not authorized to delete"


def search_files(directory):
    found_files = []

    if directory == "" or directory == "/":
        search_directory = working_directory
    else:
        search_directory = safe_join(working_directory, directory)

    for root, _, files in os.walk(search_directory):
        for file in files:
            if file.startswith('.'):
                continue
            relative_path = os.path.relpath(os.path.join(root, file), working_directory)
            found_files.append(relative_path.replace("isaa-directory\\", ""))

    return found_files
