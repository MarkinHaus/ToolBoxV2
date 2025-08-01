import copy
import os
import threading
import time
from collections.abc import Callable
from dataclasses import field
from enum import Enum
from inspect import signature
import asyncio  # Added for async operations
from pathlib import Path

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

from .base.KnowledgeBase import TextSplitter
from .extras.filter import filter_relevant_texts
from .types import TaskChain

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
import shutil
import subprocess
import sys
from typing import Any, Optional, Awaitable

from toolboxv2 import FileHandler, MainTool, Spinner, Style, get_app, get_logger

# Updated imports for EnhancedAgent
from .base.Agent.agent import (
    EnhancedAgent,
    AgentModelData,  # For type hinting if needed
    WorldModel, ProcessingStrategy,  # For type hinting if needed
)
from .base.Agent.builder import (
    EnhancedAgentBuilder,
    BuilderConfig,
)
# AgentVirtualEnv and LLMFunction might be deprecated or need adaptation
# For now, keeping them if they are used by other parts not being refactored.
# from t.base.Agents import AgentVirtualEnv, LLMFunction


from .base.AgentUtils import (
    AgentChain,
    AISemanticMemory,
    Scripts,
    dilate_string, ControllerManager
)
from .CodingAgent.live import Pipeline
from .extras.modes import (
    ISAA0CODE,  # Assuming this is a constant string
    ChainTreeExecutor,
    StrictFormatResponder,
    SummarizationMode,
    TaskChainMode,
    # crate_llm_function_from_langchain_tools, # This will need to adapt to ADK tools
)

from .SearchAgentCluster.search_tool import web_search
from .chainUi import initialize_module as initialize_isaa_chains
from .ui import initialize_isaa_webui_module
PIPLINE = None  # This seems unused or related to old pipeline
Name = 'isaa'
version = "0.2.0"  # Version bump for significant changes

pipeline_arr = [  # This seems to be for HuggingFace pipeline, keep as is for now
    'question-answering',
    'summarization',
    'text-classification',
    'text-to-speech',
]

row_agent_builder_sto = {}

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
        # self.coding_projects: dict[str, ProjectManager] = {} # Assuming ProjectManager is defined elsewhere or removed
        self.pipes: dict[str, Pipeline] = {}
        if app is None:
            app = get_app("isaa-mod")
        self.version = version
        self.name = "isaa"
        self.Name = "isaa"
        self.color = "VIOLET2"
        self.config = {'controller-init': False,
                       'agents-name-list': [],
                       "DEFAULTMODEL0": os.getenv("DEFAULTMODEL0", "ollama/llama3.1"),
                       "DEFAULT_AUDIO_MODEL": os.getenv("DEFAULT_AUDIO_MODEL", "groq/whisper-large-v3-turbo"),
                       "DEFAULTMODEL1": os.getenv("DEFAULTMODEL1", "ollama/llama3.1"),
                       "DEFAULTMODELST": os.getenv("DEFAULTMODELST", "ollama/llama3.1"),
                       "DEFAULTMODEL2": os.getenv("DEFAULTMODEL2", "ollama/llama3.1"),
                       "DEFAULTMODELCODE": os.getenv("DEFAULTMODELCODE", "ollama/llama3.1"),
                       "DEFAULTMODELSUMMERY": os.getenv("DEFAULTMODELSUMMERY", "ollama/llama3.1"),
                       "DEFAULTMODEL_LF_TOOLS": os.getenv("DEFAULTMODEL_LF_TOOLS", "ollama/llama3.1"),
                       }
        self.per_data = {}
        self.agent_data: dict[str, dict] = {}  # Will store BuilderConfig dicts
        self.keys = {
            "KEY": "key~~~~~~~",
            "Config": "config~~~~"
        }
        self.initstate = {}

        extra_path = ""
        if self.toolID:  # MainTool attribute
            extra_path = f"/{self.toolID}"
        self.observation_term_mem_file = f"{app.data_dir}/Memory{extra_path}/observationMemory/"
        self.config['controller_file'] = f"{app.data_dir}{extra_path}/controller.json"
        self.mas_text_summaries_dict = FileCache(folder=f"{app.data_dir}/Memory{extra_path}/summaries/")
        self.tools = {
            "name": "isaa",
            "Version": self.show_version,
            "add_task": self.add_task,
            "save_task": self.save_task,
            "load_task": self.load_task,
            "get_task": self.get_task,
            "list_task": self.list_task,
            "mini_task_completion": self.mini_task_completion,
            "run_agent": self.run_agent,
            "save_to_mem": self.save_to_mem_sync,
            "get_agent": self.get_agent,  # Now async
            "run_task": self.run_task,  # Now async
            "crate_task_chain": self.crate_task_chain,  # Now async
            "format_class": self.format_class,  # Now async
            "get_memory": self.get_memory,
            "get_pipe": self.get_pipe,  # Now async
            "run_pipe": self.run_pipe,  # Now async
            "rget_mode": lambda mode: self.controller.rget(mode),
            "set_local_files_tools": self.set_local_files_tools,
        }
        self.working_directory = os.getenv('ISAA_WORKING_PATH', os.getcwd())
        self.print_stream = stram_print
        self.agent_collective_senses = False  # This might be obsolete with EnhancedAgent's design
        self.global_stream_override = False  # Handled by EnhancedAgentBuilder
        self.pipes_device = 1  # For HuggingFace pipelines
        self.lang_chain_tools_dict: dict[str, Any] = {}  # Store actual tool objects for wrapping
        self.agent_chain = AgentChain(directory=f"{app.data_dir}{extra_path}/chains")
        self.agent_chain_executor = ChainTreeExecutor()
        # These runners will become async due to get_agent being async
        self.agent_chain_executor.function_runner = self._async_function_runner
        self.agent_chain_executor.agent_runner = self._async_agent_runner

        self.agent_memory: AISemanticMemory = f"{app.id}{extra_path}/Memory"  # Path for AISemanticMemory
        self.controller = ControllerManager({})
        self.summarization_mode = 1
        self.summarization_limiter = 102000
        self.speak = lambda x, *args, **kwargs: x  # Placeholder
        self.scripts = Scripts(f"{app.data_dir}{extra_path}/ScriptFile")
        self.ac_task = None  # Unused?
        self.default_setter = None  # For agent builder customization
        self.local_files_tools = True  # Related to old FileManagementToolkit
        self.initialized = False

        self.personality_code = ISAA0CODE  # Constant string

        FileHandler.__init__(self, f"isaa{extra_path.replace('/', '-')}.config", app.id if app else __name__)
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=None, color=self.color, on_exit=self.on_exit)

        self.fc_generators = {}  # Unused?
        self.toolID = ""  # MainTool attribute
        MainTool.toolID = ""  # Static attribute?
        self.web_search = web_search
        self.shell_tool_function = shell_tool_function
        self.tools["shell"] = shell_tool_function

        self.print(f"Start {self.spec}.isaa")
        with Spinner(message="Starting module", symbols='c'):
            self.load_file_handler()
            config_fh = self.get_file_handler(self.keys["Config"])
            if config_fh is not None:
                if isinstance(config_fh, str):
                    try:
                        config_fh = json.loads(config_fh)
                    except json.JSONDecodeError:
                        self.print(f"Warning: Could not parse config from file handler: {config_fh[:100]}...")
                        config_fh = {}

                if isinstance(config_fh, dict):
                    # Merge, prioritizing existing self.config for defaults not in file
                    loaded_config = config_fh
                    for key, value in self.config.items():
                        if key not in loaded_config:
                            loaded_config[key] = value
                    self.config = loaded_config

            if self.spec == 'app':  # MainTool attribute
                self.load_keys_from_env()

            # Ensure directories exist
            Path(f"{get_app('isaa-initIsaa').data_dir}/Agents/").mkdir(parents=True, exist_ok=True)
            Path(f"{get_app('isaa-initIsaa').data_dir}/Memory/").mkdir(parents=True, exist_ok=True)

        #initialize_isaa_chains(self.app)
        #initialize_isaa_webui_module(self.app, self)
        #self.print("ISAA module started. fallback")

    async def _async_function_runner(self, name, **kwargs):
        agent = await self.get_agent("self")  # Get self agent for its tools
        # EnhancedAgent doesn't have function_invoke. Need to find tool and run.
        # This is a simplified version. Real ADK tool execution is more complex.
        for tool in agent.tools:  # Assuming agent.tools is populated for EnhancedAgent
            if tool.name == name:
                # This is a placeholder. ADK tools expect ToolContext.
                # For simple FunctionTool, calling tool.func might work.
                if hasattr(tool, 'func') and callable(tool.func):
                    if asyncio.iscoroutinefunction(tool.func):
                        return await tool.func(**kwargs)
                    else:
                        return tool.func(**kwargs)
                else:  # Fallback to trying to run the tool directly if it's a BaseTool instance
                    # This is highly dependent on the tool's implementation
                    try:
                        if hasattr(tool, 'run_async'):
                            return await tool.run_async(args=kwargs, tool_context=None)
                        elif hasattr(tool, 'run'):
                            return tool.run(args=kwargs, tool_context=None)
                    except Exception as e:
                        self.print(f"Error running tool {name} via fallback: {e}")
                        return f"Error running tool {name}"
        return f"Tool {name} not found on self agent."

    async def _async_agent_runner(self, name, task, **kwargs):
        return await self.run_agent(name, task, **kwargs)

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

    async def run_task(self, task_input: str, chain_name: str, sum_up=True):
        self.agent_chain_executor.reset()

        return self.agent_chain_executor.execute(task_input,
                                          self.agent_chain.get(chain_name), sum_up)

    async def crate_task_chain(self, prompt):
        agents_list = self.config.get('agents-name-list', ['self', 'isaa'])
        # Tools list needs to be adapted for EnhancedAgent/ADK
        self_agent = await self.get_agent("self")
        tools_list_str = ", ".join(
            [tool.name for tool in self_agent.tools]) if self_agent.tools else "No tools available"

        prompt += f"\n\nAvailable Agents: {agents_list}"
        prompt += f"\n\nAvailable Tools on 'self' agent: {tools_list_str}"
        prompt += f"\n\nAvailable Chains: {self.list_task()}"

        if 'TaskChainAgent' not in self.config['agents-name-list']:
            task_chain_builder = self.get_agent_builder("code")
            task_chain_builder.with_agent_name("TaskChainAgent")
            tcm = self.controller.rget(TaskChainMode)
            task_chain_builder.with_system_message(tcm.system_msg)  # Use system_msg from LLMMode
            await self.register_agent(task_chain_builder)  # await here

        task_chain_dict = await self.format_class(TaskChain, prompt, agent_name="TaskChainAgent")
        task_chain = TaskChain(**task_chain_dict)

        self.print(f"New TaskChain {task_chain.name} len:{len(task_chain.tasks)}")

        if task_chain and len(task_chain.tasks):
            self.print(f"adding : {task_chain.name}")
            self.agent_chain.add(task_chain.name, task_chain.model_dump().get("tasks"))
            self.agent_chain.add_discr(task_chain.name, task_chain.description)
        return task_chain.name

    def get_augment(self, task_name=None, exclude=None):
        # This needs to be adapted. Serialization of EnhancedAgent is through BuilderConfig.
        return {
            "tools": {},  # Tool configurations might be part of BuilderConfig now
            "Agents": self.serialize_all(exclude=exclude),  # Returns dict of BuilderConfig dicts
            "customFunctions": json.dumps(self.scripts.scripts),  # Remains same
            "tasks": self.agent_chain.save_to_dict(task_name)  # Remains same
        }

    async def init_from_augment(self, augment, agent_name: str or EnhancedAgentBuilder = 'self', exclude=None):
        builder_instance = None
        if isinstance(agent_name, str):
            builder_instance = self.get_agent_builder(agent_name)  # Gets a builder
        elif isinstance(agent_name, EnhancedAgentBuilder):
            builder_instance = agent_name
        else:
            raise ValueError(f"Invalid Type {type(agent_name)} accept ar: str and EnhancedAgentBuilder")

        a_keys = augment.keys()

        if "tools" in a_keys:
            tools_config = augment['tools']  # This config needs to define how tools are added to builder
            # model_for_tools = tools_config.get("tools.model", self.config['DEFAULTMODEL_LF_TOOLS']) # model is per agent
            # Await self.init_tools(tools_config, builder_instance) # init_tools needs to work with builder
            pass  # Tool init needs redesign for builder

        if "Agents" in a_keys:
            agents_configs_dict = augment['Agents']  # This is dict of BuilderConfig dicts
            self.deserialize_all(agents_configs_dict)  # Populates self.agent_data
            self.print("Agent configurations loaded.")

        if "customFunctions" in a_keys:
            custom_functions = augment['customFunctions']
            if isinstance(custom_functions, str):
                custom_functions = json.loads(custom_functions)
            if custom_functions:
                self.scripts.scripts = custom_functions
                self.print("customFunctions saved")

        if "tasks" in a_keys:
            tasks = augment['tasks']  # This is AgentChain data
            if isinstance(tasks, str):
                tasks = json.loads(tasks)
            if tasks:
                self.agent_chain.load_from_dict(tasks)  # AgentChain handles its own format
                self.print("tasks chains restored")

    async def init_tools(self, tools_config: dict, agent_builder: EnhancedAgentBuilder):
        # This function needs to be adapted to add tools to the EnhancedAgentBuilder
        # For LangChain tools, they need to be wrapped as callables or ADK BaseTool instances.
        lc_tools_names = tools_config.get('lagChinTools', [])
        # hf_tools_names = tools_config.get('huggingTools', []) # HuggingFace tools are also LangChain tools
        # plugin_urls = tools_config.get('Plugins', [])

        all_lc_tool_names = list(set(lc_tools_names))  # + hf_tools_names

        for tool_name in all_lc_tool_names:
            try:
                # Load tool instance (LangChain's load_tools might return a list)
                loaded_tools = load_tools([tool_name], llm=None)  # LLM not always needed for tool definition
                for lc_tool_instance in loaded_tools:
                    # Wrap and add to builder
                    # Simple case: wrap lc_tool_instance.run or lc_tool_instance._run
                    if hasattr(lc_tool_instance, 'run') and callable(lc_tool_instance.run):
                        # ADK FunctionTool needs a schema, or infers it.
                        # We might need to manually create Pydantic models for args.
                        # For simplicity, assume ADK can infer or the tool takes simple args.
                        agent_builder.with_adk_tool_function(lc_tool_instance.run, name=lc_tool_instance.name,
                                                             description=lc_tool_instance.description)
                        self.print(f"Added LangChain tool '{lc_tool_instance.name}' to builder.")
                        self.lang_chain_tools_dict[lc_tool_instance.name] = lc_tool_instance  # Store for reference
            except Exception as e:
                self.print(f"Failed to load/add LangChain tool '{tool_name}': {e}")

        # AIPluginTool needs more complex handling as it's a class
        # for url in plugin_urls:
        #     try:
        #         plugin = AIPluginTool.from_plugin_url(url)
        #         # Exposing AIPluginTool methods might require creating individual FunctionTools
        #         # Or creating a custom ADK BaseTool wrapper for AIPluginTool
        #         self.print(f"AIPluginTool {plugin.name} loaded. Manual ADK wrapping needed.")
        #     except Exception as e:
        #         self.print(f"Failed to load AIPlugin from {url}: {e}")

    def serialize_all(self, exclude=None):
        # Returns a copy of agent_data, which contains BuilderConfig dicts
        # The exclude logic might be different if it was excluding fields from old AgentBuilder
        # For BuilderConfig, exclusion happens during model_dump if needed.
        return copy.deepcopy(self.agent_data)

    def deserialize_all(self, data: dict[str, dict]):
        # Data is a dict of {agent_name: builder_config_dict}
        self.agent_data.update(data)
        # Clear instances from self.config so they are rebuilt with new configs
        for agent_name in data.keys():
            self.config.pop(f'agent-instance-{agent_name}', None)

    async def init_isaa(self, name='self', build=False, only_v=False, **kwargs):  # build/only_v seem unused
        if self.initialized:
            self.print(f"Already initialized. Getting agent/builder: {name}")
            # build=True implies getting the builder, build=False (default) implies getting agent instance
            return self.get_agent_builder(name) if build else await self.get_agent(name)

        self.initialized = True
        sys.setrecursionlimit(1500)
        self.load_keys_from_env()

        # Background loading
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, self.agent_chain.load_from_file)
        loop.run_in_executor(None, self.scripts.load_scripts)

        with Spinner(message="Building Controller", symbols='c'):
            self.controller.init(self.config['controller_file'])
        self.config["controller-init"] = True


        return self.get_agent_builder(name) if build else await self.get_agent(name)

    def show_version(self):
        self.print("Version: ", self.version)
        return self.version

    def on_start(self):

        initialize_isaa_chains(self.app)
        initialize_isaa_webui_module(self.app, self)

        threading.Thread(target=self.load_to_mem_sync, daemon=True).start()
        self.print("ISAA module started.")

    def load_secrit_keys_from_env(self):
        # These are often used by LiteLLM if not passed directly or set as env vars for LiteLLM
        pass  # Keeping this empty as API keys are better handled by direct env vars or builder config

    def load_keys_from_env(self):
        # Update default model names from environment variables
        for key in self.config:
            if key.startswith("DEFAULTMODEL"):
                self.config[key] = os.getenv(key, self.config[key])
        self.config['VAULTS'] = os.getenv("VAULTS")

    def on_exit(self):
        # Save agent configurations
        for agent_name, agent_instance in self.config.items():
            if agent_name.startswith('agent-instance-') and isinstance(agent_instance, EnhancedAgent):
                # If agent instance has its own save logic (e.g. cost tracker)
                # asyncio.run(agent_instance.close()) # This might block, consider task group
                # The BuilderConfig is already in self.agent_data, which should be saved.
                pass  # Agent instances are not directly saved, their configs are.

        threading.Thread(target=self.save_to_mem_sync, daemon=True).start()  # Sync wrapper for save_to_mem

        # Save controller if initialized
        if self.config.get("controller-init"):
            self.controller.save(self.config['controller_file'])

        # Clean up self.config for saving
        clean_config = {}
        for key, value in self.config.items():
            if key.startswith('agent-instance-'): continue  # Don't save instances
            if key.startswith('LLM-model-'): continue  # Don't save langchain models
            clean_config[key] = value
        self.add_to_save_file_handler(self.keys["Config"], json.dumps(clean_config))

        # Save other persistent data
        self.save_file_handler()
        self.agent_chain.save_to_file()
        self.scripts.save_scripts()

    def save_to_mem_sync(self):
        # This used to call agent.save_memory(). EnhancedAgent does not have this.
        # If AISemanticMemory needs global saving, it should be handled by AISemanticMemory itself.
        # For now, this can be a no-op or save AISemanticMemory instances if managed by Tools.
        memory_instance = self.get_memory()  # Assuming this returns AISemanticMemory
        if hasattr(memory_instance, 'save_all_memories'):  # Hypothetical method
            memory_instance.save_all_memories(f"{get_app().data_dir}/Memory/")
        self.print("Memory saving process initiated")
    def load_to_mem_sync(self):
        # This used to call agent.save_memory(). EnhancedAgent does not have this.
        # If AISemanticMemory needs global saving, it should be handled by AISemanticMemory itself.
        # For now, this can be a no-op or save AISemanticMemory instances if managed by Tools.
        memory_instance = self.get_memory()  # Assuming this returns AISemanticMemory
        if hasattr(memory_instance, 'load_all_memories'):  # Hypothetical method
            memory_instance.load_all_memories(f"{get_app().data_dir}/Memory/")
        self.print("Memory loading process initiated")

    def get_agent_builder(self, name="self") -> EnhancedAgentBuilder:
        if name == 'None': name = "self"  # Default name
        self.print(f"Default EnhancedAgentBuilder::{name}")

        agent_builder = EnhancedAgentBuilder(agent_name=name)
        agent_builder._isaa_ref = self  # Store ISAA reference if needed by builder logic or tools

        # Load from file if exists
        agent_config_path = Path(f"{get_app().data_dir}/Agents/{name}.agent.json")  # Builder saves as .json
        if agent_config_path.exists():
            try:
                agent_builder.load_config(agent_config_path)
                self.print(f"Loaded existing configuration for builder {name} from {agent_config_path}")
            except Exception as e:
                self.print(f"Failed to load config for {name}: {e}. Using defaults.")
                # Fallback to default settings if load fails
                agent_builder = EnhancedAgentBuilder(agent_name=name)
                agent_builder._isaa_ref = self

        if self.global_stream_override:  # This is a global flag for ISAA
            agent_builder.enable_streaming(True)

        # Set default model (can be overridden by loaded config)
        if not agent_builder._config.model_identifier:
            agent_builder.with_model(self.config.get(f'DEFAULTMODEL{name.upper()}', self.config['DEFAULTMODEL0']))

        # Apply ISAA specific setter if available
        if self.default_setter:
            agent_builder = self.default_setter(agent_builder, name)  # Pass name for context

        # Add common ISAA tools
        async def run_isaa_agent_tool(target_agent_name: str, instructions: str, **kwargs_):
            # Ensure this function is awaitable as ADK tools might be async
            return await self.run_agent(target_agent_name, instructions, **kwargs_)

        from typing import Optional, Dict

        async def memory_search_tool(
            query: str,
            search_mode: Optional[str] = "wide",
            context_name: Optional[str] = None
        ) -> str:
            """
            Führt eine flexible Suche im Speicher durch, deren Genauigkeit über einen Parameter gesteuert werden kann.

            Args:
                query: Der Suchbegriff, nach dem im Speicher gesucht werden soll.
                search_mode: Optional. Bestimmt die Art der Suche. Muss einer der folgenden Werte sein:
                             - "wide": Für eine breite, umfassende Suche mit vielen Ergebnissen.
                             - "balanced": (Standard) Für eine ausgewogene Suche mit einem guten Kompromiss
                                           zwischen Relevanz und Anzahl der Ergebnisse.
                             - "narrow": Für eine sehr präzise und enge Suche, die nur die relevantesten
                                         Ergebnisse liefert.
                context_name: Optional. Gibt an, welcher Speicher durchsucht werden soll.
                              Für mehrere Speicher, übergeben Sie die Namen als einen einzigen,
                              komma-separierten String (z.B. "projekt_alpha,allgemeine_notizen").
                              Wird nichts angegeben, werden alle Speicher durchsucht.
            """
            # Dies muss den *eigenen* Speicher des Agenten oder einen gemeinsamen ISAA-Speicher verwenden
            # Vorerst wird ein globaler ISAA-Speicher angenommen
            mem_instance = self.get_memory()

            # Verarbeite den optionalen, komma-separierten context_name String in eine Liste
            memory_names_list = [name.strip() for name in context_name.split(',')] if context_name else None

            # Setze die Suchparameter basierend auf dem gewählten Modus
            if search_mode == "wide":
                # Parameter für eine breite Suche: mehr Ergebnisse, geringere Ähnlichkeitsschwelle
                search_params = {
                    "k": 7, "min_similarity": 0.1, "cross_ref_depth": 3,
                    "max_cross_refs": 4, "max_sentences": 8
                }
            elif search_mode == "narrow":
                # Parameter für eine enge, präzise Suche: wenige Ergebnisse, hohe Ähnlichkeitsschwelle
                search_params = {
                    "k": 2, "min_similarity": 0.75, "cross_ref_depth": 1,
                    "max_cross_refs": 1, "max_sentences": 3
                }
            else:  # "balanced" ist der Standard
                # Standardparameter für eine ausgewogene Suche
                search_params = {
                    "k": 3, "min_similarity": 0.2, "cross_ref_depth": 2,
                    "max_cross_refs": 2, "max_sentences": 5
                }

            # Führe die eigentliche Abfrage mit den festgelegten Parametern durch
            return await mem_instance.query(
                query=query,
                memory_names=memory_names_list,
                query_params=search_params,
                to_str=True
            )

        async def save_to_memory_tool(data_to_save: str, context_name: str = name):
            mem_instance = self.get_memory()
            ist = await mem_instance.add_data(context_name, str(data_to_save), direct=True)
            if ist:
                return 'Data added to memory.'
            raise ValueError('Error adding data to memory.')

        # agent_builder.with_adk_tool_function(run_isaa_agent_tool, name="runAgent",
        #                                      description=f"Run another ISAA agent. Available: {self.config.get('agents-name-list', [])}")
        agent_builder.with_adk_tool_function(memory_search_tool, name="memorySearch",
                                             description="Search ISAA's semantic memory.")
        agent_builder.with_adk_tool_function(save_to_memory_tool, name="saveDataToMemory",
                                             description="Save data to ISAA's semantic memory for the current agent's context.")
        #agent_builder.with_adk_tool_function(self.web_search, name="searchWeb",
        #                                     description="Search the web for information.")
        agent_builder.with_adk_tool_function(self.shell_tool_function, name="shell", description=f"Run a shell command. in {detect_shell()}")

        # Add more tools based on agent 'name' or type
        if name == "self" or "code" in name.lower() or "pipe" in name.lower():  # Example condition
            async def code_pipeline_tool(task: str, do_continue: bool = False):
                pipe_agent_name = name  # Use current agent's context for the pipe
                return await self.run_pipe(pipe_agent_name, task, do_continue=do_continue)

            agent_builder.with_adk_tool_function(code_pipeline_tool, name="runCodePipeline",
                                                 description="Run a multi-step code generation/execution task.")

        # Configure cost tracking persistency if not already set by loaded config
        if agent_builder._config.cost_tracker_config is None or agent_builder._config.cost_tracker_config.get(
            'type') != 'json':
            cost_file = Path(f"{get_app().data_dir}/Agents/{name}.costs.json")
            agent_builder.with_json_cost_tracker(cost_file)

        return agent_builder

    async def register_agent(self, agent_builder: EnhancedAgentBuilder):
        agent_name = agent_builder._config.agent_name
        if f'agent-instance-{agent_name}' in self.config:
            self.print(f"Agent '{agent_name}' instance already exists. Overwriting config and rebuilding on next get.")
            self.config.pop(f'agent-instance-{agent_name}', None)  # Remove old instance

        # Save the builder's configuration
        config_path = Path(f"{get_app().data_dir}/Agents/{agent_name}.agent.json")
        agent_builder.save_config(config_path)
        self.print(f"Saved EnhancedAgentBuilder config for '{agent_name}' to {config_path}")

        # Store the serializable config dict in self.agent_data
        self.agent_data[agent_name] = agent_builder._config.model_dump()

        if agent_name not in self.config.get("agents-name-list", []):
            if "agents-name-list" not in self.config: self.config["agents-name-list"] = []
            self.config["agents-name-list"].append(agent_name)

        self.print(f"EnhancedAgent '{agent_name}' configuration registered. Will be built on first use.")
        row_agent_builder_sto[agent_name] = agent_builder
        # Agent is built on demand by get_agent to handle async build

    async def get_agent(self, agent_name="Normal", model_override: str | None = None) -> EnhancedAgent:
        if "agents-name-list" not in self.config: self.config["agents-name-list"] = []

        instance_key = f'agent-instance-{agent_name}'
        if instance_key in self.config:
            agent_instance = self.config[instance_key]
            if model_override and agent_instance.amd.model != model_override:
                self.print(f"Model override for {agent_name}: {model_override}. Rebuilding.")
                self.config.pop(instance_key, None)  # Remove old instance
                # Fall through to build with new model
            else:
                self.print(f"Returning existing EnhancedAgent instance: {agent_name}")
                return agent_instance

        builder_to_use = None
        if agent_name in self.agent_data:
            self.print(f"Loading configuration for EnhancedAgent: {agent_name}")
            try:
                builder_to_use = row_agent_builder_sto.get(agent_name, EnhancedAgentBuilder(config=BuilderConfig(**self.agent_data[agent_name])))
            except Exception as e:
                self.print(f"Error loading BuilderConfig for {agent_name}: {e}. Falling back to default.")

        if builder_to_use is None:
            self.print(f"No existing config for {agent_name} or load failed. Creating default builder.")
            builder_to_use = self.get_agent_builder(agent_name)  # This returns a configured builder

        # Apply ISAARef and model override if any
        builder_to_use._isaa_ref = self
        if model_override:
            builder_to_use.with_model(model_override)

        # Ensure agent name in builder config matches requested name, important if default builder was fetched
        if builder_to_use._config.agent_name != agent_name:
            builder_to_use.with_agent_name(agent_name)

        self.print(f"Building EnhancedAgent: {agent_name} with model {builder_to_use._config.model_identifier}")
        agent_instance = await builder_to_use.build()

        self.config[instance_key] = agent_instance
        # Ensure its config is in agent_data (if built from default)
        if agent_name not in self.agent_data:
            self.agent_data[agent_name] = builder_to_use._config.model_dump()

        if agent_name not in self.config["agents-name-list"]:
            self.config["agents-name-list"].append(agent_name)

        self.print(f"Built and cached EnhancedAgent instance: {agent_name}")
        return agent_instance

    async def mini_task_completion(self, mini_task: str, user_task: str | None = None, mode: Any = None,  # LLMMode
                                   max_tokens_override: int | None = None, task_from="system",
                                   stream_function: Callable | None = None, message_history: list | None = None, agent_name="TaskCompletion"):
        if mini_task is None: return None
        if mini_task == "test": return "test"
        self.print(f"Running mini task, volume {len(mini_task)}")

        agent = await self.get_agent(agent_name)  # Ensure agent is retrieved (and built if needed)

        effective_system_message = agent.amd.system_message
        if mode and hasattr(mode, 'system_msg') and mode.system_msg:
            effective_system_message = mode.system_msg

        messages = []
        if effective_system_message:
            messages.append({"role": "system", "content": effective_system_message})
        if message_history:
            messages.extend(message_history)

        current_prompt = mini_task
        if user_task:  # If user_task is provided, it becomes the main prompt, mini_task is context
            messages.append({"role": task_from, "content": mini_task})  # mini_task as prior context
            current_prompt = user_task  # user_task as the current prompt

        messages.append({"role": "user", "content": current_prompt})

        # Prepare params for a_run_llm_completion
        llm_params = {"model": agent.amd.model, "llm_messages": messages}
        if max_tokens_override:
            llm_params['max_tokens'] = max_tokens_override
        else:
            llm_params['max_tokens'] = agent.amd.max_tokens

        if stream_function:
            llm_params['stream'] = True
            # EnhancedAgent a_run_llm_completion handles stream_callback via agent.stream_callback
            # For a one-off, we might need a temporary override or pass it if supported.
            # For now, assume stream_callback is set on agent instance if needed globally.
            # If stream_function is for this call only, agent.a_run_llm_completion needs modification
            # or we use a temporary agent instance. This part is tricky.
            # Let's assume for now that if stream_function is passed, it's a global override for this agent type.
            original_stream_cb = agent.stream_callback
            original_stream_val = agent.stream
            agent.stream_callback = stream_function
            agent.stream = True
            try:
                response_content = await agent.a_run_llm_completion(**llm_params)
            finally:
                agent.stream_callback = original_stream_cb
                agent.stream = original_stream_val  # Reset to builder's config
            return response_content  # Streaming output handled by callback

        llm_params['stream'] = False
        response_content = await agent.a_run_llm_completion(**llm_params)
        return response_content

    async def mini_task_completion_format(self, mini_task, format_schema: type[BaseModel],
                                          max_tokens_override: int | None = None, agent_name="TaskCompletion",
                                          task_from="system", mode_overload: Any = None, user_task: str | None = None):
        if mini_task is None: return None
        self.print(f"Running formatted mini task, volume {len(mini_task)}")

        agent = await self.get_agent(agent_name)

        effective_system_message = None
        if mode_overload and hasattr(mode_overload, 'system_msg') and mode_overload.system_msg:
            effective_system_message = mode_overload.system_msg

        message_context = []
        if effective_system_message:
            message_context.append({"role": "system", "content": effective_system_message})

        current_prompt = mini_task
        if user_task:
            message_context.append({"role": task_from, "content": mini_task})
            current_prompt = user_task

        # Use agent.a_format_class
        try:
            result_dict = await agent.a_format_class(
                pydantic_model=format_schema,
                prompt=current_prompt,
                message_context=message_context,
                # max_tokens can be part of agent's model config or passed if a_format_class supports it
            )
            if format_schema == bool:  # Special handling for boolean schema
                # a_format_class returns a dict, e.g. {"value": True}. Extract the bool.
                # This depends on how bool schema is defined. A common way: class BoolResponse(BaseModel): value: bool
                return result_dict.get("value", False) if isinstance(result_dict, dict) else False
            return result_dict
        except Exception as e:
            self.print(f"Error in mini_task_completion_format: {e}")
            return None  # Or raise

    async def format_class(self, format_schema: type[BaseModel], task: str, agent_name="TaskCompletion"):
        if format_schema is None or not task: return None

        agent = None
        if isinstance(agent_name, str):
            agent = await self.get_agent(agent_name)
        elif isinstance(agent_name, EnhancedAgent):
            agent = agent_name
        else:
            raise TypeError("agent_name must be str or EnhancedAgent instance")

        return await agent.a_format_class(format_schema, task)

    async def get_pipe(self, agent_name_or_instance: str | EnhancedAgent, *args, **kwargs) -> Pipeline:
        agent_instance = None
        agent_name_str = ""

        if isinstance(agent_name_or_instance, str):
            agent_name_str = agent_name_or_instance
            agent_instance = await self.get_agent(agent_name_str)
        elif isinstance(agent_name_or_instance, EnhancedAgent):
            agent_instance = agent_name_or_instance
            agent_name_str = agent_instance.amd.name  # amd is AgentModelData
        else:
            return self.return_result().default_internal_error(f"agent_name_or_instance must be str or EnhancedAgent is {type(agent_name_or_instance)}")

        if agent_name_str in self.pipes:
            # Optionally reconfigure if args/kwargs are different
            # For simplicity, returning existing pipe.
            return self.pipes[agent_name_str]
        else:
            # Pass the already fetched/validated agent_instance to Pipeline
            self.pipes[agent_name_str] = Pipeline(agent_instance, *args, **kwargs)
            return self.pipes[agent_name_str]

    async def run_pipe(self, agent_name_or_instance: str | EnhancedAgent, task: str, do_continue=False):
        pipe = await self.get_pipe(agent_name_or_instance)
        return await pipe.run(task, do_continue=do_continue)  # pipeline.run is async

    async def run_agent(self, name: str | EnhancedAgent,
                        text: str,
                        verbose: bool = False,  # Handled by agent's own config mostly
                        session_id: Optional[str] = None,
                        persist_history: bool = True,
                        strategy_override: str | None = None,  # Maps to ProcessingStrategy enum
                        progress_callback: Callable[[Any], None | Awaitable[None]] | None = None,
                        **kwargs):  # Other kwargs for a_run
        if text is None: return ""
        if text == "test": return ""

        agent_instance = None
        if isinstance(name, str):
            agent_instance = await self.get_agent(name)
        elif isinstance(name, EnhancedAgent):
            agent_instance = name
        else:
            return self.return_result().default_internal_error(
                f"Invalid agent identifier type: {type(name)}")

        self.print(f"Running agent {agent_instance.amd.name} for task: {text[:100]}...")
        save_p = None
        if progress_callback:
            save_p = agent_instance.progress_callback
            agent_instance.progress_callback = progress_callback

        # Map strategy_override string to Enum if needed
        # from toolboxv2.mods.isaa.base.Agent.agent import ProcessingStrategy (example)
        strategy_enum = ProcessingStrategy[strategy_override.upper()] if strategy_override else None

        # Call EnhancedAgent's a_run method
        response = await agent_instance.a_run(
            user_input=text,
            session_id=session_id,
            persist_history=persist_history,
            strategy_override=strategy_enum,
            kwargs_override=kwargs  # Pass other kwargs if a_run supports them
        )
        if save_p:
            agent_instance.progress_callback = save_p

        return response

    # mass_text_summaries and related methods remain complex and depend on AISemanticMemory
    # and specific summarization strategies. For now, keeping their structure,
    # but calls to self.format_class or self.mini_task_completion will become async.

    async def mas_text_summaries(self, text, min_length=3600, ref=None):
        len_text = len(text)
        if len_text < min_length: return text
        key = self.one_way_hash(text, 'summaries', 'isaa')
        value = self.mas_text_summaries_dict.get(key)
        if value is not None: return value

        # This part needs to become async due to format_class
        # Simplified version:
        summary = await self.mini_task_completion(
            mini_task=f"Summarize this text, focusing on aspects related to '{ref if ref else 'key details'}'. The text is: {text}",
            mode=self.controller.rget(SummarizationMode))

        if summary is None or not isinstance(summary, str):
            # Fallback or error handling
            summary = text[:min_length] + "... (summarization failed)"

        self.mas_text_summaries_dict.set(key, summary)
        return summary

    def get_memory(self, name: str | None = None) -> AISemanticMemory:
        # This method's logic seems okay, AISemanticMemory is a separate system.
        logger_ = get_logger()  # Renamed to avoid conflict with self.logger
        if isinstance(self.agent_memory, str):  # Path string
            logger_.info(Style.GREYBG("AISemanticMemory Initialized from path"))
            self.agent_memory = AISemanticMemory(base_path=self.agent_memory)

        cm = self.agent_memory
        if name is not None:
            # Assuming AISemanticMemory.get is synchronous or you handle async appropriately
            # If AISemanticMemory methods become async, this needs adjustment
            mem_kb = cm.get(name)  # This might return a list of KnowledgeBase or single one
            return mem_kb
        return cm

    # set_local_files_tools seems related to old FileManagementToolkit, may need removal or ADK equivalent
    def set_local_files_tools(self, local_files_tools_enabled: bool):
        self.local_files_tools = local_files_tools_enabled
        self.print(f"Local file tools (old system) set to: {self.local_files_tools}")
        # If using ADK, file tools would be added to agents via builder.
        # This flag might control whether default builders get default file tools.
        return f"Local file tools (old system) set to: {self.local_files_tools}"


def detect_shell() -> tuple[str, str]:
    """
    Detects the best available shell and the argument to execute a command.
    Returns:
        A tuple of (shell_executable, command_argument).
        e.g., ('/bin/bash', '-c') or ('powershell.exe', '-Command')
    """
    if platform.system() == "Windows":
        if shell_path := shutil.which("pwsh"):
            return shell_path, "-Command"
        if shell_path := shutil.which("powershell"):
            return shell_path, "-Command"
        return "cmd.exe", "/c"

    shell_env = os.environ.get("SHELL")
    if shell_env and shutil.which(shell_env):
        return shell_env, "-c"

    for shell in ["bash", "zsh", "sh"]:
        if shell_path := shutil.which(shell):
            return shell_path, "-c"

    return "/bin/sh", "-c"


def safe_decode(data: bytes) -> str:
    encodings = [sys.stdout.encoding, locale.getpreferredencoding(), 'utf-8', 'latin-1', 'iso-8859-1']
    for enc in encodings:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode('utf-8', errors='replace')


def shell_tool_function(command: str) -> str:
    result: dict[str, Any] = {"success": False, "output": "", "error": ""}
    try:
        shell_exe, cmd_flag = detect_shell()

        process = subprocess.run(
            [shell_exe, cmd_flag, command],
            capture_output=True,
            text=False,
            timeout=120,
            check=False
        )

        stdout = safe_decode(process.stdout)
        stderr = safe_decode(process.stderr)

        if process.returncode == 0:
            result.update({"success": True, "output": stdout, "error": stderr if stderr else ""})
        else:
            error_output = (f"Stdout:\n{stdout}\nStderr:\n{stderr}" if stdout else stderr).strip()
            result.update({
                "success": False,
                "output": stdout,
                "error": error_output if error_output else f"Command failed with exit code {process.returncode}"
            })

    except subprocess.TimeoutExpired:
        result.update({"error": "Timeout", "output": f"Command '{command}' timed out after 120 seconds."})
    except Exception as e:
        result.update({"error": f"Unexpected error: {type(e).__name__}", "output": str(e)})

    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    # Example of running an async method from Tools if needed for testing
    async def test_isaa_tools():
        app_instance = get_app("isaa_test_app")
        isaa_tool_instance = Tools(app=app_instance)
        await isaa_tool_instance.init_isaa()

        # Test get_agent
        self_agent = await isaa_tool_instance.get_agent("self")
        print(f"Got agent: {self_agent.amd.name} with model {self_agent.amd.model}")

        # Test run_agent
        # response = await isaa_tool_instance.run_agent("self", "Hello, world!")
        # print(f"Response from self agent: {response}")

        # Test format_class (example Pydantic model)
        class MyData(BaseModel):
            name: str
            value: int

        # formatted_data = await isaa_tool_instance.format_class(MyData, "The item is 'test' and its count is 5.")
        # print(f"Formatted data: {formatted_data}")


    asyncio.run(test_isaa_tools())
