import copy
import os
import threading
import time
from collections.abc import Callable
from dataclasses import field
from enum import Enum
from inspect import signature

import requests
import torch
from langchain_community.agent_toolkits.load_tools import (
    load_huggingface_tool,
    load_tools,
)
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceHub, OpenAI
from langchain_community.tools import AIPluginTool
from pebble import concurrent
from pydantic import BaseModel

from toolboxv2.mods.isaa.base.KnowledgeBase import TextSplitter
from toolboxv2.mods.isaa.extras.filter import filter_relevant_texts
from toolboxv2.mods.isaa.types import TaskChain

# from toolboxv2.mods.isaa.ui.nice import IsaaWebSocketUI
from toolboxv2.utils.system import FileCache

from ...utils.toolbox import stram_print

try:
    import gpt4all
except Exception:
    def gpt4all():
        return None
    gpt4all.GPT4All = None

import json
import locale
import platform
import shlex
import subprocess
import sys
from typing import Any

from toolboxv2 import FileHandler, MainTool, Spinner, Style, get_app, get_logger
from toolboxv2.mods.isaa.base.Agents import (
    Agent,
    AgentBuilder,
    AgentVirtualEnv,
    ControllerManager,
    LLMFunction,
)
from toolboxv2.mods.isaa.base.AgentUtils import (
    AgentChain,
    AISemanticMemory,
    Scripts,
    dilate_string,
)
from toolboxv2.mods.isaa.CodingAgent.live import Pipeline
from toolboxv2.mods.isaa.extras.modes import (
    ISAA0CODE,
    ChainTreeExecutor,
    StrictFormatResponder,
    SummarizationMode,
    TaskChainMode,
    crate_llm_function_from_langchain_tools,
)

from .SearchAgentCluster.search_tool import web_search

PIPLINE = None
Name = 'isaa'
version = "0.1.5"

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
    'text-to-speech',
    # 'text-generation',
    # 'text2text-generation',
    # 'token-classification',
    # 'translation',
    # 'visual-question-answering',
    # 'vqa',
    # 'zero-shot-classification',
    # 'zero-shot-image-classification',
    # 'zero-shot-object-detection',
    # 'translation_en_to_de',
    # 'fill-mask'
]


def get_ip():
    response = requests.get('https://api64.ipify.org?format=json').json()
    return response["ip"]


@concurrent.process(timeout=12)
def get_location():
    ip_address = get_ip()
    response = requests.get(f'https://ipapi.co/{ip_address}/json/').json()
    location_data = f"city: {response.get('city')},region: {response.get('region')},country: {response.get('country_name')},"

    return location_data


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):

        self.run_callback = None
        self.coding_projects: dict[str, ProjectManager] = {}
        self.pipes: dict[str, Pipeline] = {}
        if app is None:
            app = get_app("isaa-mod")
        self.version = version
        self.name = "isaa"
        self.Name = "isaa"
        self.color = "VIOLET2"
        self.config = {'controller-init': False,
                       'agents-name-list': [],

                       "DEFAULTMODEL0": "ollama/llama3.1",
                       "DEFAULT_AUDIO_MODEL": "groq/whisper-large-v3-turbo",
                       "DEFAULTMODEL1": "ollama/llama3.1",
                       "DEFAULTMODELST": "ollama/llama3.1",
                       "DEFAULTMODEL2": "ollama/llama3.1",
                       "DEFAULTMODELCODE": "ollama/llama3.1",
                       "DEFAULTMODELSUMMERY": "ollama/llama3.1",
                       "DEFAULTMODEL_LF_TOOLS": "ollama/llama3.1",
                       }
        self.per_data = {}
        self.agent_data = {}
        self.keys = {
            "KEY": "key~~~~~~~",
            "Config": "config~~~~"
        }
        self.initstate = {}

        extra_path = ""
        if self.toolID:
            extra_path = f"/{self.toolID}"
        self.observation_term_mem_file = f".data/{app.id}/Memory{extra_path}/observationMemory/"
        self.config['controller_file'] = f".data/{app.id}{extra_path}/controller.json"
        self.mas_text_summaries_dict = FileCache(folder=f".data/{app.id}/Memory{extra_path}/summaries/")
        self.tools = {
            "name": "isaa",
            "Version": self.show_version,
            "add_task": self.add_task,
            "save_task": self.save_task,
            "load_task": self.load_task,
            "get_task": self.get_task,
            "list_task": self.list_task,
            "save_to_mem": self.save_to_mem,
            # "mini_task": self.mini_task_completion,
            "get_agent": self.get_agent,
            # "run_agent": self.run_agent,
            "run_task": self.run_task,
            "crate_task_chain": self.crate_task_chain,
            "format_class": self.format_class,
            "get_memory": self.get_memory,
            "get_pipe": self.get_pipe,
            "run_pipe": self.run_pipe,
            "rget_mode": lambda mode: self.controller.rget(mode),
            "set_local_files_tools": self.set_local_files_tools,
        }
        self.working_directory = os.getenv('ISAA_WORKING_PATH')
        self.print_stream = stram_print
        self.agent_collective_senses = False
        self.global_stream_override = False
        self.pipes_device = 1
        self.lang_chain_tools_dict: dict[str, str] = {}
        self.agent_chain = AgentChain(directory=f".data/{app.id}{extra_path}/chains")
        self.agent_chain_executor = ChainTreeExecutor()
        self.agent_chain_executor.function_runner = lambda name, **b: self.get_agent("self").function_invoke(name,
                                                                                                                   **b)
        self.agent_chain_executor.agent_runner = lambda name, task, **k: self.run_agent(name, task, **k)
        self.agent_memory: AISemanticMemory = f"{app.id}{extra_path}/Memory"
        self.controller = ControllerManager({})
        self.summarization_mode = 1  # 0 to 3  0 huggingface 1 text 2 opnai 3 gpt
        self.summarization_limiter = 102000
        self.speak = lambda x, *args, **kwargs: x
        self.scripts = Scripts(f".data/{app.id}{extra_path}/ScriptFile")
        self.ac_task = None
        self.default_setter = None
        self.local_files_tools = True
        self.initialized = False

        self.personality_code = ISAA0CODE

        FileHandler.__init__(self, f"isaa{extra_path.replace('/', '-')}.config", app.id if app else __name__)
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=None, color=self.color, on_exit=self.on_exit)

        self.fc_generators = {}
        self.toolID = ""
        MainTool.toolID = ""
        self.web_search = web_search
        self.shell_tool_function = shell_tool_function

        self.print(f"Start {self.spec}.isaa")
        # IsaaWebSocketUI(self)
        # init_isaaflow_ui(self.app)
        with Spinner(message="Starting module", symbols='c'):
            self.load_file_handler()
            config = self.get_file_handler(self.keys["Config"])
            if config is not None:
                if isinstance(config, str):
                    config = json.loads(config)
                if isinstance(config, dict):
                    self.config = {**config, **self.config}

            if self.spec == 'app':
                self.load_keys_from_env()

            if not os.path.exists(f".data/{get_app('isaa-initIsaa').id}/Agents/"):
                os.mkdir(f".data/{get_app('isaa-initIsaa').id}/Agents/")
            if not os.path.exists(f".data/{get_app().id}/Memory/"):
                os.mkdir(f".data/{get_app('isaa-initIsaa').id}/Memory/")

    def add_task(self, name, task):
        self.agent_chain.add_task(name, task)

    def list_task(self):
        return str(self.agent_chain)

    def remove_task(self, name):
        return self.agent_chain.remove(name)

    def save_task(self, name=None):
        self.agent_chain.save_to_file(name)

    def load_task(self, name=None):
        self.agent_chain.load_from_file(name)

    def get_task(self, name=None):
        return self.agent_chain.get(name)

    def run_task(self, task, name, sum_up=True):
        self.agent_chain_executor.reset()
        return self.agent_chain_executor.execute(task, self.agent_chain.get(name), sum_up=sum_up)

    def crun_task(self, prompt):
        chain_name = self.crate_task_chain(prompt)

        out = self.run_task(prompt, chain_name) if chain_name else "No chain generated"

        return out, chain_name

    def crate_task_chain(self, prompt):

        prompt += f"\n\nAvalabel Agents: {self.config.get('agents-name-list', ['self', 'isaa'])}"
        prompt += f"\n\nAvalabel Tools: {[f.function for f in self.get_agent('self').functions]}"
        prompt += f"\n\nAvalabel Chains: {self.list_task()}"

        if 'TaskChainAgent' not in self.config['agents-name-list']:
            task_chain_agent = self.get_default_agent_builder("code")
            task_chain_agent.set_amd_name("TaskChainAgent")
            tcm = self.controller.rget(TaskChainMode)
            task_chain_agent.set_mode(tcm)
            self.register_agent(task_chain_agent)

        task_chain: TaskChain = TaskChain(**self.format_class(TaskChain, prompt, agent_name="TaskChainAgent"))

        self.print(f"New TaskChain {task_chain.name} len:{len(task_chain.tasks)}")

        if task_chain and len(task_chain.tasks):
            self.print(f"adding : {task_chain.name}")
            self.agent_chain.add(task_chain.name, task_chain.model_dump().get("tasks"))
            self.agent_chain.add_discr(task_chain.name, task_chain.dis)
        return task_chain.name

    def get_augment(self, task_name=None, exclude=None):
        return {
            "tools": {},
            "Agents": self.serialize_all(exclude=exclude),
            "customFunctions": json.dumps(self.scripts.scripts),
            "tasks": self.agent_chain.save_to_dict(task_name)
        }

    def init_from_augment(self, augment, agent_name: str or AgentBuilder = 'self', exclude=None):
        if isinstance(agent_name, str):
            agent = self.get_agent(agent_name)
        elif isinstance(agent_name, AgentBuilder):
            agent = agent_name
        else:
            return ValueError(f"Invalid Type {type(agent_name)} accept ar : str and AgentProvider")
        a_keys = augment.keys()

        if "tools" in a_keys:
            tools = augment['tools']
            print("tools:", tools)
            self.init_tools(tools, tools.get("tools.model", self.config['DEFAULTMODEL_LF_TOOLS'], agent))
            self.print("tools initialized")

        if "Agents" in a_keys:
            agents = augment['Agents']
            self.deserialize_all(agents)
            self.print("Agents crated")

        if "customFunctions" in a_keys:
            custom_functions = augment['customFunctions']
            if isinstance(custom_functions, str):
                custom_functions = json.loads(custom_functions)
            if custom_functions:
                self.scripts.scripts = custom_functions
                self.print("customFunctions saved")

        if "tasks" in a_keys:
            tasks = augment['tasks']
            if isinstance(tasks, str):
                tasks = json.loads(tasks)
            if tasks:
                self.agent_chain.load_from_dict(tasks)
                self.print("tasks chains restored")

    def init_tools(self, tools, model_name: str, agent: Agent | None = None):  # not  in unit test


        # tools = {  # Todo save tools to file and loade from usaage data format : and isaa_extras
        #    "lagChinTools": ["ShellTool", "ReadFileTool", "CopyFileTool",
        #                     "DeleteFileTool", "MoveFileTool", "ListDirectoryTool"],
        #    "huggingTools": [],
        #    "Plugins": ["https://nla.zapier.com/.well-known/ai-plugin.json"],
        #    "Custom": [],
        # }

        if agent is None:
            agent = self.get_agent("self")

        if 'Plugins' not in tools:
            tools['Plugins'] = []
        if 'lagChinTools' not in tools:
            tools['lagChinTools'] = []
        if 'huggingTools' not in tools:
            tools['huggingTools'] = []

        llm_fuctions = []

        for plugin_url in set(tools['Plugins']):
            get_logger().info(Style.BLUE(f"Try opening plugin from : {plugin_url}"))
            try:
                plugin_tool = AIPluginTool.from_plugin_url(plugin_url)
                get_logger().info(Style.GREEN(f"Plugin : {plugin_tool.name} loaded successfully"))
                plugin_tool.description += "API Tool use request; infos :" + plugin_tool.api_spec + "." + str(
                    plugin_tool.args_schema)
                llm_fuctions += crate_llm_function_from_langchain_tools(plugin_tool)
                self.lang_chain_tools_dict[plugin_tool.name + "-usage-information"] = plugin_tool
            except Exception as e:
                get_logger().error(Style.RED(f"Could not load : {plugin_url}"))
                get_logger().error(Style.GREEN(f"{e}"))

        for tool in load_tools(list(set(tools['lagChinTools'])),
                               self.get_llm_models(model_name)):
            llm_fuctions += crate_llm_function_from_langchain_tools(tool)
        for tool in set(tools['huggingTools']):
            llm_fuctions += crate_llm_function_from_langchain_tools(
                load_huggingface_tool(tool, self.config['HUGGINGFACEHUB_API_TOKEN']))
        agent.functions += llm_fuctions

    def serialize_all(self, exclude=None):
        if exclude is None:
            exclude = []
        data = copy.deepcopy(self.agent_data)
        for agent_name, agent_data in data.items():
            for e in exclude:
                del agent_data[e]
            if 'taskstack' in agent_data:
                del agent_data['taskstack']
            if 'amd' in agent_data and 'provider' in agent_data['amd']:
                if isinstance(agent_data['amd'].get('provider'), Enum):
                    agent_data['amd']['provider'] = str(agent_data['amd'].get('provider').name).upper()
            data[agent_name] = agent_data
        return data

    def deserialize_all(self, data):
        for key, _agent_data in data.items():
            _ = self.get_agent(key)

    def init_isaa(self, name='self', build=False, only_v=False, **kwargs):
        if self.initialized:
            self.print(f"Already initialized returning agent / builder name : {name}")
            if build:
                return self.get_default_agent_builder(name)
            return self.get_agent(name)

        self.initialized = True
        sys.setrecursionlimit(1500)

        self.load_keys_from_env()

        def helper():
            self.agent_chain.load_from_file()
            self.scripts.load_scripts()
            self.config["controller-init"] = True
            return True

        threading.Thread(target=helper, daemon=True).start()

        with Spinner(message="Building Controller", symbols='c'):
            self.controller.init(self.config['controller_file'])

        if build:
            return self.get_agent(name)

        with Spinner(message=f"Preparing default config for Agent {name}", symbols='c'):
            return self.get_default_agent_builder(name)

    def show_version(self):
        self.print("Version: ", self.version)
        return self.version

    async def on_start(self):
        pass
        # init_isaaflow_ui(self.app)

    def load_secrit_keys_from_env(self):
        self.config['WOLFRAM_ALPHA_APPID'] = os.getenv('WOLFRAM_ALPHA_APPID')
        self.config['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')
        self.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
        self.config['REPLICATE_API_TOKEN'] = os.getenv('REPLICATE_API_TOKEN')
        self.config['IFTTTKey'] = os.getenv('IFTTTKey')
        self.config['SERP_API_KEY'] = os.getenv('SERP_API_KEY')
        self.config['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
        self.config['PINECONE_API_ENV'] = os.getenv('PINECONE_API_ENV')

    def load_keys_from_env(self):
        self.config['DEFAULTMODELST'] = os.getenv("DEFAULTMODELST", "ollama/llama3.1")
        self.config['DEFAULTMODEL0'] = os.getenv("DEFAULTMODEL0", "ollama/llama3.1")
        self.config['DEFAULTMODEL1'] = os.getenv("DEFAULTMODEL1", "ollama/llama3.1")
        self.config['DEFAULTMODEL2'] = os.getenv("DEFAULTMODEL2", "ollama/llama3.1")
        self.config['DEFAULTMODELCODE'] = os.getenv("DEFAULTMODELCODE", "ollama/llama3.1")
        self.config['DEFAULTMODELSUMMERY'] = os.getenv("DEFAULTMODELSUMMERY", "ollama/llama3.1")
        self.config['DEFAULTMODEL_LF_TOOLS'] = os.getenv("DEFAULTMODEL_LF_TOOLS", "ollama/llama3.1")
        self.config['VAULTS'] = os.getenv("VAULTS")

    def webInstall(self, user_instance, construct_render) -> str:
        self.print('Installing')
        return construct_render(content="./app/0/isaa_installer/ii.html",
                                element_id="Installation",
                                externals=["/app/0/isaa_installer/ii.js"],
                                from_file=True)

    def on_exit(self):

        threading.Thread(target=self.save_to_mem, daemon=True).start()

        for v in self.pipes.values():
            v.on_exit()

        self.config['augment'] = self.get_augment(exclude=['amd'])
        del self.config['augment']['tasks']

        if self.config["controller-init"]:
            self.controller.save(self.config['controller_file'])
            self.config["controller-init"] = False

        for key in list(self.config.keys()):
            if key.startswith("LLM-model-"):
                del self.config[key]
            if key.startswith("agent-config-"):
                del self.config[key]
            if key.endswith("_pipeline"):
                del self.config[key]
            if key.endswith("-init"):
                self.config[key] = False
            if key == 'agents-name-list':
                for agent_name in self.config[key]:
                    self.config[f"agent-world_model-{agent_name}"] = self.config[f'agent-config-{agent_name}'].world_model
                self.config[key] = []
        # print(self.config)
        self.add_to_save_file_handler(self.keys["Config"], json.dumps(self.config))
        self.save_file_handler()
        self.agent_chain.save_to_file()
        self.scripts.save_scripts()

    def init_pipeline(self, p_type, model, **kwargs):
        global PIPLINE
        if PIPLINE is None:
            from transformers import pipeline as PIPLINE
        if p_type not in self.initstate:
            self.initstate[p_type + model] = False

        if not self.initstate[p_type + model]:
            self.app.logger.info(f"init {p_type} pipeline")
            if self.pipes_device >= 1 and torch.cuda.is_available():
                if torch.cuda.device_count() < self.pipes_device:
                    self.print("device count exceeded ava-label ar")
                    for i in range(1, torch.cuda.device_count()):
                        self.print(torch.cuda.get_device_name(i - 1))

                self.config[f"{p_type + model}_pipeline"] = PIPLINE(p_type, model=model, device=self.pipes_device - 1,
                                                                    **kwargs)
            else:
                self.app.logger.warning("Cuda is not available")
                self.config[f"{p_type + model}_pipeline"] = PIPLINE(p_type, model=model, **kwargs)
            self.app.logger.info("Done")
            self.initstate[p_type + model] = True

    def free_llm_model(self, names: list[str]):
        for model in names:
            self.initstate[f'LLM-model-{model}-init'] = False
            del self.config[f'LLM-model-{model}']

    def load_llm_models(self, names: list[str]):
        for model in names:
            if f'LLM-model-{model}-init' not in self.initstate:
                self.initstate[f'LLM-model-{model}-init'] = False

            if not self.initstate[f'LLM-model-{model}-init']:
                self.initstate[f'LLM-model-{model}-init'] = True
                if '/' in model:
                    self.config[f'LLM-model-{model}'] = HuggingFaceHub(repo_id=model,
                                                                       huggingfacehub_api_token=self.config[
                                                                           'HUGGINGFACEHUB_API_TOKEN'])
                    self.print(f'Initialized HF model : {model}')
                elif model.startswith('gpt4all#'):
                    m = gpt4all.GPT4All(model.replace('gpt4all#', ''))
                    self.config[f'LLM-model-{model}'] = m
                    self.print(f'Initialized gpt4all model : {model}')
                elif model.startswith('gpt'):
                    self.config[f'LLM-model-{model}'] = ChatOpenAI(model_name=model,
                                                                   openai_api_key=self.config['OPENAI_API_KEY'],
                                                                   streaming=True)
                    self.print(f'Initialized OpenAi model : {model}')
                else:
                    self.config[f'LLM-model-{model}'] = OpenAI(model_name=model,
                                                               openai_api_key=self.config['OPENAI_API_KEY'])
                    self.print(f'Initialized OpenAi : {model}')

    def get_llm_models(self, name: str):
        if f'LLM-model-{name}' not in self.config:
            self.load_llm_models([name])
        return self.config[f'LLM-model-{name}']

    def add_lang_chain_tools_to_agent(self, agent: Agent, tools: list[str] | None = None):

        if tools is None:
            tools = []
        for key in self.lang_chain_tools_dict:
            self.print(f"Adding tool for loading : {key}")
            tools += [key]

        self.lang_chain_tools_dict = {}

        ll_functions = crate_llm_function_from_langchain_tools(tools)

        agent.functions += ll_functions

    def tools_to_llm_functions(self, tools: dict):
        llm_functions = []
        for tool_name, tool in tools.items():
            if isinstance(tool, dict):
                func = tool.get('func', None)
            if isinstance(tool, Callable):
                func = tool
                tool = {'func': func}
            if func is None:
                self.app.logger.warning(f'No function found for {tool_name}')
                continue

            parameters = tool.get('parameters')
            if parameters is None:

                try:
                    from litellm.utils import function_to_dict
                    parameters = function_to_dict(func)["parameters"]["properties"]
                except:
                    parameters = {}
                    for _1, _ in signature(func).parameters.items():
                        if hasattr(_.annotation, '__name__'):
                            parameters[_1] = _.annotation.__name__
                        else:
                            parameters[_1] = _.annotation
            llm_functions.append(
                LLMFunction(name=tool_name,
                            description=tool.get('description'),
                            parameters=parameters,
                            function=func)
            )
        return llm_functions

    def get_agent_builder(self, name="BP") -> AgentBuilder:
        return AgentBuilder(Agent).set_isaa_reference(self).set_amd_name(name)

    def register_agents_setter(self, setter):
        self.default_setter = setter

    def register_agent(self, agent_builder):
        if f'agent-config-{agent_builder.agent.amd.name}' in self.config:
            print(f"{agent_builder.agent.amd.name} Agent already registered")
            return

        agent_builder.save_to_json(f".data/{get_app('isaa.register_agent').id}/Agents/{agent_builder.agent.amd.name}.agent")
        self.config[f'agent-config-{agent_builder.agent.amd.name}'] = agent_builder.build()
        self.config["agents-name-list"].append(agent_builder.agent.amd.name)
        self.agent_data[agent_builder.amd_attributes['name']] = agent_builder.get_dict()
        self.print(f"Agent:{agent_builder.agent.amd.name} Registered")
        return self.config[f'agent-config-{agent_builder.agent.amd.name}']

    def get_default_agent_builder(self, name="self") -> AgentBuilder:
        if name == 'None':
            return self.get_default_agent_builder()
        self.print(f"Default AgentBuilder::{name}")
        agent_builder: AgentBuilder = self.get_agent_builder(name)

        if name != "":
            if os.path.exists(f".data/{get_app('isaa.get_default_agent_builder').id}/Memory/{name}.agent"):
                agent_builder = agent_builder.load_from_json_file(f".data/{get_app('isaa.get_default_agent_builder').id}/Memory/{name}.agent", Agent)
                agent_builder.set_isaa_reference(self)

        if self.global_stream_override:
            agent_builder.set_stream(True)

        mem = self.get_memory()
        tools = {}

        agent_builder.set_memory(mem).set_amd_stop_sequence(["QUERY:", "...\n"])  # .set_trim(Trims.isaa)

        if self.default_setter is not None:
            agent_builder = self.default_setter(agent_builder)

        if self.local_files_tools:
            pass
            # if name in ['liveInterpretation', 'tools']:
            #    toolkit = FileManagementToolkit(
            #        root_dir=str(self.working_directory)
            #    )  # If you don't provide a root_dir, operations will default to the current working directory
            #    for file_tool in toolkit.get_tools():
            #        # print("adding file tool", file_tool.name)
            #        tools[file_tool.name] = file_tool
            # if name in ['self', 'liveInterpretation'] or 'ide' in name:
            #     isaa_ide_online = self.app.mod_online("isaa_ide", installed=True)
            #     if isaa_ide_online:
            #         isaa_ide = self.app.get_mod("isaa_ide")
            #         isaa_ide.scope = self.working_directory
            #         isaa_ide.add_tools(tools)

        agent_builder.init_agent_memory(name)

        def run_agent(agent_name: str, text: str, **kwargs):
            text = text.replace("'", "").replace('"', '')
            if agent_name:
                return self.run_agent(agent_name, text, **kwargs)
            return "Provide Information in The Action Input: fild or function call"

        async def run_pipe(task: str, do_continue:bool=False):
            task = task.replace("'", "").replace('"', '')
            return await self.run_pipe(name, task, do_continue)

        def memory_search(query: str):
            ress = self.get_memory().query(query,to_str=True)

            if not ress:
                return "no informations found for :" + query

            return ress

        async def ad_data(data: str):
            await mem.add_data(name, str(data))

            return 'added to memory'

        def get_agents(*a,**k):
            agents_name_list = self.config['agents-name-list'].copy()
            if 'TaskCompletion' in agents_name_list:
                agents_name_list.remove('TaskCompletion')
            if 'create_task' in agents_name_list:
                agents_name_list.remove('create_task')
            if 'summary' in agents_name_list:
                agents_name_list.remove('summary')
            return agents_name_list

        if name == "self":
            # config.mode = "free"
            # config.model_name = self.config['DEFAULTMODEL0']  # "gpt-4"
            # config.max_iterations = 6
            agent_builder.set_amd_model(self.config['DEFAULTMODEL0'])

            tools["runAgent"] = {
                "func": lambda agent_name, instructions: self.run_agent(agent_name, instructions),
                "description": "The run_agent function takes a 2 arguments agent_name, instructions"
                               + f"""The function parses the input string x and extracts the values associated with the following keys:

                       agent_name: The name of the agent to be run. : {get_agents()}
                       instructions: The task that the agent is to perform. (do not enter the name of a task_chain!) give clear Instructions

                   The function then runs the Agent with the specified name and Instructions."""}

            tools["getAvailableAgents"] = {
                "func": get_agents,
                "description": "Use to get list of all agents avalabel"}

            tools["saveDataToMemory"] = {"func": ad_data, "description": "tool to save data to memory,"
                                                                         " write the data as specific"
                                                                         " and accurate as possible."}

            tools = {**tools, **{
                "memorySearch": {"func": memory_search,
                                 "description": "must input reference context to search"},
                "searchWeb": {"func": self.web_search,
                              "description": "search the web (online) for information's input a query"
                    , "format": "search(<task>)"},
                "think": {"func": lambda x: run_agent('think', x),
                          "description": "Run agent to solve a text based problem"
                    , "format": "think(<task>)"},
                "shell": {"func": shell_tool_function,
                          "description": "Run shell command"
                    , "format": "shell(command: str)"},
                "miniTask": {"func": lambda x: self.mini_task_completion(x),
                             "description": "programmable pattern completion engin. use text args:str only"
                    , "format": "miniTask(<detaild_discription>)"},
                "Coder": {"func": self.code,
                          "description": "to write code basd from description"
                    , "format": "coding_step(task<detaild_discription>: str, project_name<data/$examplename>: str)"},
                "run_pipe": {"func": run_pipe,
                          "description": "to perform complex multi step task in an inactive coding env"
                    , "format": "run_pipe(task<detaild_discription>: str, do_continue<if continue on last run or start fresh>: bool)"},

            }}

        if "STT" in name:

            agent_builder.set_amd_model(self.config['DEFAULT_AUDIO_MODEL'])

        if "tool" in name:
            tools = {}
            for key, _tool in self.lang_chain_tools_dict.items():
                tools[key] = {"func": _tool, "description": _tool.description, "format": f"{key}({_tool.args})"}
            agent_builder.set_amd_model(self.config['DEFAULTMODEL0'])

        if "search" in name:

            # config.mode = "tools"
            # config.model_name = self.config['DEFAULTMODEL1']
            # config.completion_mode = "chat"
            # config.set_agent_type("structured-chat-zero-shot-react-description")
            # config.max_iterations = 6
            # config.verbose = True
            agent_builder.set_amd_model(self.config['DEFAULTMODEL0'])
            agent_builder.set_content_memory_max_length(3500)
            tools.update({"memorySearch": {"func": lambda context: memory_search(context),
                                           "description": "Search for memory  <context>"}})
            tools.update({"WebSearch": {"func": self.web_search,
                                            "description": "Search the web"}})

            tools["saveDataToMemory"] = {"func": ad_data, "description": "tool to save data to memory,"
                                                                         " write the data as specific"
                                                                         " and accurate as possible."}

        if name == "think":
            agent_builder.set_amd_model(self.config['DEFAULTMODELST'])
            # .stop_sequence = ["\n\n\n"]

        if "shell" in name:
            (agent_builder.set_amd_model(self.config['DEFAULTMODEL1'])
             .set_amd_system_message("Act as an Command Shell Agent. You can run shell commandants by writing "
                                     "\nFUCTION: {'Action','shell','Input':[shell_command]}"))
            tools["shell"] = {"func": shell_tool_function,
                              "description": "Run shell command"
                , "parameters": {"type": "string"}}
            pass
            # .set_model_name(self.config['DEFAULTMODEL1'])
            # .add_system_information = False
            # .stop_sequence = ["\n"]

        if name == "liveInterpretation":
            pass
            # .set_model_name(self.config['DEFAULTMODEL0']).stream = True
            # config.stop_sequence = ["!X!"]

        if name == "summary":
            agent_builder.set_amd_model(self.config['DEFAULTMODELSUMMERY'])

        if name == "thinkm":
            agent_builder.set_amd_model(self.config['DEFAULTMODEL1'])

        if name == "TaskCompletion":
            agent_builder.set_amd_model(self.config['DEFAULTMODEL1'])

        if name == "code":
            agent_builder.set_amd_model(self.config['DEFAULTMODELCODE'])

        tools = {**tools, **{

            "saveDataToMemory": {"func": ad_data, "description": "tool to save data to memory,"
                                                                 " write the data as specific"
                                                                 " and accurate as possible."},
            "memorySearch": {"func": lambda x: memory_search(x),
                             "description": "Search for similar memory input <context>"}, }}

        if agent_builder.amd_attributes.get('model') is None:
            agent_builder.set_amd_model(self.config['DEFAULTMODEL2'])
        llm_functions = self.tools_to_llm_functions(tools)
        agent_builder.set_functions(llm_functions)
        os.makedirs(f".data/{get_app('isaa-get-agent').id}/Agents/", exist_ok=True)
        agent_builder_dict = agent_builder.save_to_json(f".data/{get_app('isaa-get-agent').id}/Agents/{name}.agent")
        self.agent_data[agent_builder.amd_attributes['name']] = agent_builder_dict
        # agent_builder.set_verbose(True)
        return agent_builder

    def remove_agent_config(self, name):
        del self.config[f'agent-config-{name}']
        self.config["agents-name-list"].remove(name)

    def get_agent(self, agent_name="Normal", model=None) -> Agent:

        if "agents-name-list" not in self.config:
            self.config["agents-name-list"] = []

        # self.config["agents-name-list"] = [k.replace('agent-config-', '') for k in self.config.keys() if k.startswith('agent-config-')])
        if f'agent-config-{agent_name}' in self.config:
            agent = self.config[f'agent-config-{agent_name}']
            if model:
                agent.amd.model = model
            self.print(f"collecting AGENT: {agent_name} "
                       f"{'Mode:' + str(agent.mode) if agent.mode is not None else ''} "
                       f"{'Cape:' + agent.capabilities.name if agent.capabilities is not None else ''}")
        else:
            with Spinner(message=f"Building Agent {agent_name}", symbols='c'):
                agent_builder = self.get_default_agent_builder(agent_name)
                if model:
                    agent_builder.set_amd_model(model)
                if agent_builder.amd_attributes.get('model', '').startswith('ollama'):
                    try:
                        agent = agent_builder.build()
                    except Exception:
                        subprocess.Popen("wsl -e ollama serve", shell=True, stdout=subprocess.PIPE,
                                                   stderr=subprocess.PIPE)
                        time.sleep(5)
                        agent = agent_builder.build()
                else:
                    agent = agent_builder.build()
            del agent_builder
            agent.world_model = self.config.get(f"agent-world_model-{agent_name}", {})
            self.config[f'agent-config-{agent_name}'] = agent
            self.print(f"Init:Agent::{agent_name}{' -' + str(agent.mode) if agent.mode is not None else ''}")
        if agent_name not in self.config["agents-name-list"]:
            self.config["agents-name-list"].append(agent_name)
        return agent


    def mini_task_completion(self, mini_task:str, user_task=None, mode=None,
                             max_tokens=None, task_from="system", stream_function=None, message=None):
        if mini_task is None:
            return None
        self.print(f"running mini task Volumen {len(mini_task)}")
        agent: Agent = self.get_agent("TaskCompletion")
        agent.mode = mode
        _stream_function = agent.stream_function
        if stream_function is not None:
            agent.stream_function = stream_function
        sto_add_function_to_prompt = agent.add_function_to_prompt
        agent.add_function_to_prompt = False
        m = agent.max_tokens
        if max_tokens is not None:
            agent.max_tokens = max_tokens

        if user_task is not None:
            user_task, mini_task = mini_task, user_task

        res = agent.mini_task(mini_task, task_from, user_task)

        agent.mode = None
        agent.add_function_to_prompt = sto_add_function_to_prompt
        agent.max_tokens = m
        agent.stream_function = _stream_function
        agent.verbose = True

        return res

    def mini_task_completion_format(self, mini_task, format_, max_tokens=None, agent_name="TaskCompletion",
                                    task_from="system", mode_overload=None, user_task=None):
        if mini_task is None:
            return None
        self.print(f"running f mini task Volumen {len(mini_task)}, format Volumen {len(mini_task)}")
        agent: Agent = self.get_agent(agent_name)
        if mode_overload is None:
            mode_overload = self.controller.rget(StrictFormatResponder)
        # if not isinstance(format_, dict):
        #     format_ = {'text':format_}
        agent.set_rformat(format_)
        res: str or list = self.mini_task_completion(mini_task=mini_task,
                                                     mode=mode_overload,
                                                     max_tokens=max_tokens,
                                                     task_from=task_from,
                                                     user_task=user_task)
        agent.reset_rformat()
        if isinstance(res, str):
            res = res.strip()

        if format_ == bool:
            return agent.fuzzy_string_match(res, ['true', 'Treue', 'false', 'False']).lower() == 'true'

        # if '{' in res and '}' in res:
        #     res_ = anything_from_str_to_dict(res)
        #     if len(res_) > 0:
        #         return res_[0]
        return res


    def format_class(self, format_class, task, agent_name="TaskCompletion"):
        if format_class is None:
            return None
        if not task:
            return None
        if isinstance(agent_name, str):
            agent: Agent = self.get_agent(agent_name)
        elif isinstance(agent_name, Agent):
            agent = agent_name

        return agent.format_class(format_class, task)

    def get_pipe(self, agent_name, *args, **kwargs) -> Pipeline:
        if isinstance(agent_name, str):
            agent: Agent = self.get_agent(agent_name)
        else:
            agent = agent_name

        if agent.amd.name in self.pipes:
            return self.pipes[agent.amd.name]

        else:
            self.pipes[agent.amd.name] = Pipeline(agent, *args, **kwargs)
        return self.pipes[agent.amd.name]

    async def run_pipe(self, agent_name, task,do_continue=False):
        return await self.get_pipe(agent_name).run(task, do_continue=do_continue)


    def short_prompt_messages(self, messages, get_tokens, max_tokens, prompt_token_margin=20):
        prompt_len = get_tokens(messages)
        max_tokens *= 0.985
        if prompt_len <= max_tokens - prompt_token_margin:
            return messages

        self.print(f"Context length: {prompt_len}, Max tokens: {max_tokens} ")

        # Pre-process first and last messages if they're too long
        first_message = messages[0]
        if len(messages) == 1:
            first_message['content'] = self.mas_text_summaries(first_message['content'])
            return [first_message]

        last_message = messages[-1]

        first_message_tokens = get_tokens([first_message])
        last_message_tokens = get_tokens([last_message])

        if first_message_tokens > max_tokens // 2:
            first_message['content'] = self.mas_text_summaries(first_message['content'])

        if last_message_tokens > max_tokens // 2:
            last_message['content'] = self.mas_text_summaries(last_message['content'],
                                                              ref=first_message['content'][:260])
        if len(messages) == 2:
            return [first_message] + [last_message]

        # Keep first and last messages intact
        middle_messages = messages[1:-1]

        all_content = "\n".join([msg['content'] for msg in middle_messages])

        dilated_content = self.mas_text_summaries(all_content, ref=first_message.get('content', '')+last_message.get('content', ''))
        new_middle_messages = {'role': "system", 'content': "History -> "+dilated_content}

        # Check if we're within token limit
        if get_tokens([first_message]+ [new_middle_messages] + [last_message]) <= max_tokens - prompt_token_margin:
            return [first_message]+ [new_middle_messages] + [last_message]

        # Final attempt: Use summarization
        new_middle_messages['content'] = dilate_string(new_middle_messages['content'], "\n", 2, 1)

        # Ensure we're within token limit
        final_messages = [first_message] + [new_middle_messages] + [last_message]
        if get_tokens(final_messages) > max_tokens - prompt_token_margin:
            # If still too long, truncate the summary
            allowed_length = max_tokens - prompt_token_margin - get_tokens([first_message, last_message])
            if 0 < allowed_length < max_tokens // 10:
                final_messages[1]['content'] = final_messages[1]['content'][:allowed_length]
            elif allowed_length < 0:
                allowed_length *= -.5
                allowed_length = int(allowed_length)
                final_messages[0]['content'] = final_messages[0]['content'][:allowed_length]
                final_messages[-1]['content'] = final_messages[-1]['content'][allowed_length:]

        return final_messages


    async def run_agent_in_environment(self, task,
                                 agent_or_name: (str or Agent) | None = None,
                                 agent_env: (str or AgentVirtualEnv) | None = None,
                                 persist=False,
                                 persist_ref=False,
                                 max_iterations=10,
                                 verbose=False,
                                 message=None,
                                 task_from='user',
                                 get_final_code=False):

        if isinstance(agent_or_name, str):
            agent = self.get_agent(agent_or_name)
        elif isinstance(agent_or_name, Agent):
            agent = agent_or_name
            name = agent.amd.name
            if name not in self.config["agents-name-list"]:
                self.config[f'agent-config-{name}'] = agent
        else:
            agent = self.get_agent("self")

        def default_env():
            env = AgentVirtualEnv()

            @env.register_prefix("THINK",
                                 "This text remains hidden. The THINK prefix should be used regularly to reason.")
            def process_think(content: str):
                return self.run_agent('think', content, verbose=verbose)

            @env.register_prefix("PLAN", "To reflect a plan.")
            def process_plan(content: str):
                return self.run_agent('self', content, running_mode='pegasus', verbose=verbose)

            @env.register_prefix("RESPONSE", "THE Final output! must write a response. in the final Turn!")
            def process_response(content: str):
                return content

            env.set_brake_test(lambda r: "RESPONSE" in r or r.rstrip().endswith('?') or r.rstrip().endswith('.'))
            return env

        if agent_env is None:
            agent_env = default_env()

        # save_state
        tso_c = agent.capabilities
        sto_add_function_to_prompt = agent.add_function_to_prompt
        sto_verbose = agent.verbose

        agent.add_function_to_prompt = True
        agent.verbose = verbose

        agent.capabilities = agent_env.get_llm_mode()

        async def fuction_exec_helper(x):
            return await agent.execute_fuction(persist, persist_ref) if x else None

        out = ""
        main_task = task
        if not persist:
            agent.reset_context()
        turns = 0
        for turn in range(max_iterations):
            turns += 1
            print()
            self.print(f"=================== Enter Turn : {turn + 1} of {max_iterations} =================\n")
            self.print(f"Task : {task[:60]}")

            agent_env.results_reset()

            with Spinner(message="Fetching llm_message...", symbols='+'):
                message = agent.get_llm_message(task, persist=True, task_from=task_from, message=message)

            out_ = await agent.a_run_model(message, persist_local=True, persist_mem=persist_ref)
            #print_prompt(message + [{'role': 'assistant', 'content': out_},
            #                        {'role': 'system', 'content': agent_env.results()}])
            out += out_

            with Spinner(message="Processioning Env step", symbols='+'):
                [await fuction_exec_helper(agent.if_for_fuction_use(line)) for line in out_.split('\n')]
                out += agent_env.results()

            if agent_env.break_test(out):
                break

            task = f"MAIN TASK: {main_task}\nMAIN TASK END\n in Turn {turn + 1}from{max_iterations}\nLast Turn Results: {out_}\n\n{agent_env.results()}\n"

        if not persist:
            agent.reset_context()
        agent.add_function_to_prompt = sto_add_function_to_prompt
        agent.capabilities = tso_c
        agent.verbose = sto_verbose

        self.print(f"DONE RUNNING ENV FOR {agent.amd.name} in Turns {turns}")

        if get_final_code:
            return out, ""
        return out

    # @get_app('isaa-run-agent').tb(name=Name, test=False)
    async def run_agent(self, name: str or Agent,
                  text: str,
                  verbose: bool = False,
                  **kwargs):
        if text is None:
            return ""
        agent = None
        if isinstance(name, str):
            # self.load_keys_from_env()
            agent = self.get_agent(name)

        elif isinstance(name, Agent):
            agent = name
            name = agent.amd.name

            if name not in self.config["agents-name-list"]:
                self.config[f'agent-config-{name}'] = agent
                self.print(f"Register:Agent::{name}:{agent.amd.name} {str(agent.mode)}\n")

        else:
            raise ValueError(f"Invalid arguments agent is not str or Agent {type(agent)}")

        agent.verbose = verbose
        agent.print_verbose(f"Running task {text[:200]}")
        # self.print(f"Running agent {name}")

        stream = agent.stream
        self.app.logger.info(f"stream: {stream}")

        if agent.mode is not None and not self.controller.registered(agent.mode.name):
            self.controller.add(agent.mode.name, agent.mode)

        if stream and agent.stream_function is None:
            agent.stream_function = self.print_stream

        return await agent.run(text, **kwargs)

    def mas_text_summaries(self, text, min_length=3600, ref=None):
        """text to summarises and ref is wit focus to summarise for, example text abut Plains, ref = Plains and Engines -> to gent a summary basd of text about Plain Engines"""
        len_text = len(text)
        if len_text < min_length:
            return text
        key = self.one_way_hash(text, 'summaries', 'isaa')
        value = self.mas_text_summaries_dict.get(key)
        self.print(f"len input : {len_text}")
        if value is not None:
            self.print("summ return vom chash")
            return value

        if ref is None:
            ref = text
        with Spinner("Processioning Summarization"):

            texts = TextSplitter(chunk_size=min_length*10, chunk_overlap=min_length // 10).split_text(text)
            relevant_texts = filter_relevant_texts(ref, texts=texts, fuzzy_threshold=51, semantic_threshold=0.52)
            self.print(f"Summary Volume from {len(text)} to {len(relevant_texts)}")
            if len(relevant_texts) == 0:
                relevant_texts = texts
            relevant_text_len = len(''.join(relevant_texts))
            self.print(f"Relevant texts Volume {len(relevant_texts)} "
                       f"average cuk len {sum([len(r) for r in relevant_texts]) / len(relevant_texts)}"
                       f" in mode "
                       f": {self.summarization_mode} ratio {len(text)}/{relevant_text_len}")

            if min_length > relevant_text_len:

                class Segments(BaseModel):
                    """Important and relevant information segment in itself complete"""
                    information: str

                class SummarizationSegments(BaseModel):
                    f"""importance for: {ref if ref != text else 'key details and concrete informations'}"""
                    segments: list[Segments] = field(default_factory=list)

                if len(relevant_texts) > 26:
                    bf = self.mas_text_summaries(' '.join(relevant_texts[20:]), min_length=min_length + 100, ref=ref)
                    relevant_texts = relevant_texts[:20] + [bf]
                segments = self.format_class(SummarizationSegments,
                                  '\n'.join(relevant_texts))["segments"]
                if sum([len(segment['information']) for segment in segments ]) > min_length*2:
                    summary = self.mini_task_completion(mini_task="Create a Summary" +
                                                                  (
                                                                      "" if ref == text else " for this reference: " + ref),
                                                        user_task='Segments:\n'.join(
                                                            [x['information'] for x in segments
                                                             ]),
                                                        mode=self.controller.rget(SummarizationMode))
                else:
                    summary = '\n'.join([x['information']for x in segments])
                if summary is None:
                    return relevant_texts[:18] + [f"Information chunks lost : {len(relevant_texts) - 18}"]
            else:
                summary = '\n'.join(relevant_texts)

            if not isinstance(summary, str):
                bf = self.mas_text_summaries(' '.join(relevant_texts[10:]), min_length=min_length + 100, ref=ref)
                relevant_texts = relevant_texts[:10] + [bf]
                summary = '\n'.join(relevant_texts)

        self.mas_text_summaries_dict.set(key, summary)

        return summary

    async def mass_text_summaries(self, text: str, min_length: int = 1600, ref: str | None = None) -> str:
        """
        Efficient large-text summarization using semantic memory retrieval
        Features:
        - Chunk-based parallel processing
        - LightRAG-powered relevance scoring
        - LiteLLM-optimized summarization
        - Multi-level caching
        """

        # 1. Text Length Check and Caching
        len_text = len(text)
        if len_text < min_length:
            return text

        cache_key = self.one_way_hash(text, 'summaries', 'isaa')
        if cached := self.mas_text_summaries_dict.get(cache_key):
            self.print("Returning cached summary")
            return cached

        # 2. Memory Initialization
        semantic_memory = self.get_memory()
        ref_query = ref or text

        def _chunk_text(text: str, chunk_size: int = 4000) -> list[str]:
            """Optimized text chunking with overlap"""
            return [text[i:i + chunk_size]
                    for i in range(0, len(text), chunk_size - 200)]

        chunks = _chunk_text(text, chunk_size=4000)
        await semantic_memory.add_data(
                                   "summary_cache",
                                   chunks,
                                   {"source": "mass_summary"})


        # 4. LightRAG-Powered Relevance Extraction
        query_params = {}

        # Hybrid search for key concepts
        results = await semantic_memory.query(
            query=ref_query,
            memory_names="summary_cache",
            query_params=query_params
        )

        summary = self._generate_llm_summary(results, ref_query, min_length)

        # 6. Cache Management
        self.mas_text_summaries_dict.set(cache_key, summary)
        return summary


    def _generate_llm_summary(self, chunks: list[dict[str, str]], query: str, min_length: int) -> str:
        """LiteLLM-optimized summarization pipeline"""
        summary_prompt = f"""Generate a concise summary focusing on {query} from these key excerpts:
        {chunks}

        Requirements:
        - Length between {min_length} and {min_length * 1.2} characters
        - Maintain technical details and numerical data
        - Use clear section headings
        - Highlight relationships between concepts"""

        return self.get_agent("self").mini_task(summary_prompt)


    def get_memory(self, name: str | None=None) -> AISemanticMemory:
        logger = get_logger()
        if isinstance(self.agent_memory, str):
            logger.info(Style.GREYBG("AISemanticMemory Initialized"))
            self.agent_memory = AISemanticMemory(base_path=self.agent_memory)
        logger.info(Style.GREYBG("AIContextMemory requested"))
        cm = self.agent_memory
        if name is not None:
            r = cm.get(name)
            if len(r) == 1:
                return r[0]
            return r
        logger.info(Style.Bold("AISemanticMemory instance, returned"))
        return cm

    def save_to_mem(self):
        for name in self.config['agents-name-list']:
            self.get_agent(agent_name=name).save_memory()

    def set_local_files_tools(self, local_files_tools):
        try:
            self.local_files_tools = bool(local_files_tools)
        except ValueError as e:
            return f"Invalid boolean value True or False not {local_files_tools} \n{str(e)}"
        return f"set to {self.local_files_tools=}"


def detect_shell() -> str:
    """Detect system-appropriate shell with fallbacks"""
    if platform.system() == "Windows":
        return "cmd.exe"

    # For Unix-like systems
    return os.environ.get("SHELL", "/bin/sh")


def safe_decode(data: bytes) -> str:
    """Handle encoding detection with multiple fallbacks"""
    encodings = [
        sys.stdout.encoding,
        locale.getpreferredencoding(),
        'utf-8',
        'latin-1',
        'iso-8859-1'
    ]

    for enc in encodings:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode('utf-8', errors='replace')


def shell_tool_function(command: str) -> str:
    """
    Robust system-agnostic command execution
    Handles encoding issues and shell detection automatically
    """
    result: dict[str, Any] = {"success": False, "output": "", "error": ""}
    shell = detect_shell()

    try:
        # Windows command formatting
        if platform.system() == "Windows":
            if "powershell" in shell.lower():
                full_cmd = f"{shell} -Command {shlex.quote(command)}"
            else:
                full_cmd = f'{shell} /c "{command}"'
        else:
            # Unix-style command formatting
            full_cmd = f"{shell} -c {shlex.quote(command)}"

        # Execute command
        process = subprocess.run(
            full_cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            text=False  # Handle decoding ourselves
        )

        result.update({
            "success": True,
            "output": safe_decode(process.stdout).split("EndOfString")[-1],
            "error": ""
        })

    except subprocess.CalledProcessError as e:
        result.update({
            "error": f"Process error [{e.returncode}]",
            "output": safe_decode(e.output)
        })
    except subprocess.TimeoutExpired:
        result.update({
            "error": "Timeout",
            "output": f"Command timed out: {command}"
        })
    except Exception as e:
        result.update({
            "error": f"Unexpected error: {type(e).__name__}",
            "output": str(e)
        })

    return json.dumps(result, ensure_ascii=False)




if __name__ == "__main__":
    print(shell_tool_function("ls"))
