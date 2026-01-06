import asyncio
import copy
import os
import secrets
import shlex
import threading
import time
from collections.abc import Callable
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import requests
from langchain_community.agent_toolkits.load_tools import (
    load_tools,
)
from pydantic import BaseModel

from toolboxv2.mods.isaa.base.Agent.types import ProgressEvent
from toolboxv2.mods.isaa.CodingAgent.live import ToolsInterface

from toolboxv2.utils.system import FileCache
from toolboxv2.utils.toolbox import stram_print


import json
import subprocess
import sys
from collections.abc import Awaitable
from typing import Any

from toolboxv2 import (
    FileHandler,
    MainTool,
    RequestData,
    Result,
    Spinner,
    Style,
    get_app,
    get_logger,
    remove_styles,
)

# Updated imports for FlowAgent
from .base.Agent.flow_agent import (
    FlowAgent,
)
from .base.Agent.builder import (
    AgentConfig,
    FlowAgentBuilder,
)
from .base.AgentUtils import (
    AISemanticMemory,
    ControllerManager,
    detect_shell,
    safe_decode,
)


PIPLINE = None  # This seems unused or related to old pipeline
Name = 'isaa'
version = "0.2.0"  # Version bump for significant changes
export = get_app("isaa.Export").tb
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


def get_location():
    ip_address = get_ip()
    response = requests.get(f'https://ipapi.co/{ip_address}/json/').json()
    location_data = f"city: {response.get('city')},region: {response.get('region')},country: {response.get('country_name')},"
    return location_data

class Tools(MainTool, FileHandler):

    def __init__(self, app=None):

        self.run_callback = None
        # self.coding_projects: dict[str, ProjectManager] = {} # Assuming ProjectManager is defined elsewhere or removed
        if app is None:
            app = get_app("isaa-mod")
        self.version = version
        self.name = "isaa"
        self.Name = "isaa"
        self.color = "VIOLET2"
        self.config = {'controller-init': False,
                       'agents-name-list': [], # TODO Remain ComplexModel FastModel BlitzModel, AudioModel, (ImageModel[i/o], VideoModel[i/o]), SummaryModel
                       "FASTMODEL": os.getenv("FASTMODEL", "ollama/llama3.1"),
                       "AUDIOMODEL": os.getenv("AUDIOMODEL", "groq/whisper-large-v3-turbo"),
                       "BLITZMODEL": os.getenv("BLITZMODEL", "ollama/llama3.1"),
                       "COMPLEXMODEL": os.getenv("COMPLEXMODEL", "ollama/llama3.1"),
                       "SUMMARYMODEL": os.getenv("SUMMARYMODEL", "ollama/llama3.1"),
                       "IMAGEMODEL": os.getenv("IMAGEMODEL", "ollama/llama3.1"),
                       "DEFAULTMODELEMBEDDING": os.getenv("DEFAULTMODELEMBEDDING", "gemini/text-embedding-004"),
                       }
        self.per_data = {}
        self.agent_data: dict[str, dict] = {}  # Will store AgentConfig dicts
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
        from .kernel.kernelin.run_unified_kernels import main as kernel_start
        self.tools = {
            "name": "isaa",
            "Version": self.show_version,
            "mini_task_completion": self.mini_task_completion,
            "run_agent": self.run_agent,
            "save_to_mem": self.save_to_mem_sync,
            "get_agent": self.get_agent,
            "format_class": self.format_class,  # Now async
            "get_memory": self.get_memory,
            "save_all_memory_vis": self.save_all_memory_vis,
            "rget_mode": lambda mode: self.controller.rget(mode),
            "kernel_start": kernel_start,
        }
        self.tools_interfaces: dict[str, ToolsInterface] = {}
        self.working_directory = os.getenv('ISAA_WORKING_PATH', os.getcwd())
        self.print_stream = stram_print
        self.global_stream_override = False  # Handled by FlowAgentBuilder
        self.lang_chain_tools_dict: dict[str, Any] = {}  # Store actual tool objects for wrapping

        self.agent_memory: AISemanticMemory = f"{app.id}{extra_path}/Memory"  # Path for AISemanticMemory
        self.controller = ControllerManager({})
        self.summarization_mode = 1
        self.summarization_limiter = 102000
        self.speak = lambda x, *args, **kwargs: x  # Placeholder

        self.default_setter = None  # For agent builder customization
        self.initialized = False

        FileHandler.__init__(self, f"isaa{extra_path.replace('/', '-')}.config", app.id if app else __name__)
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=None, color=self.color, on_exit=self.on_exit)

        from .extras.web_search import web_search
        async def web_search_tool(query: str) -> str:
            res = web_search(query)
            return await self.mas_text_summaries(str(res), min_length=12000, ref=query)
        self.web_search = web_search_tool
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
                from .extras.agent_ui import initialize

                initialize(self.app)

                # Oder in CloudM
                self.app.run_any(
                    ("CloudM", "add_ui"),
                    name="AgentUI",
                    title="FlowAgent Chat",
                    description="Chat with your FlowAgents",
                    path="/api/Minu/render?view=agent_ui&ssr=true",
                )

            # Ensure directories exist
            Path(f"{get_app('isaa-initIsaa').data_dir}/Agents/").mkdir(parents=True, exist_ok=True)
            Path(f"{get_app('isaa-initIsaa').data_dir}/Memory/").mkdir(parents=True, exist_ok=True)


    def get_augment(self):
        # This needs to be adapted. Serialization of FlowAgent is through AgentConfig.
        return {
            "Agents": self.serialize_all(),  # Returns dict of AgentConfig dicts
        }

    async def init_from_augment(self, augment, agent_name: str = 'self'):
        """Initialize from augmented data using new builder system"""

        # Handle agent_name parameter
        if isinstance(agent_name, str):
            pass  # Use string name
        elif hasattr(agent_name, 'config'):  # FlowAgentBuilder
            agent_name = agent_name.config.name
        else:
            raise ValueError(f"Invalid agent_name type: {type(agent_name)}")

        a_keys = augment.keys()

        # Load agent configurations
        if "Agents" in a_keys:
            agents_configs_dict = augment['Agents']
            self.deserialize_all(agents_configs_dict)
            self.print("Agent configurations loaded.")

        # Tools are now handled by the builder system during agent creation
        if "tools" in a_keys:
            self.print("Tool configurations noted - will be applied during agent building")

    async def init_tools(self, tools_config: dict, agent_builder: FlowAgentBuilder):
        # This function needs to be adapted to add tools to the FlowAgentBuilder
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
                        agent_builder.add_tool(lc_tool_instance.run, name=lc_tool_instance.name,
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

    def serialize_all(self):
        # Returns a copy of agent_data, which contains AgentConfig dicts
        # The exclude logic might be different if it was excluding fields from old AgentBuilder
        # For AgentConfig, exclusion happens during model_dump if needed.
        return copy.deepcopy(self.agent_data)

    def deserialize_all(self, data: dict[str, dict]):
        # Data is a dict of {agent_name: builder_config_dict}
        self.agent_data.update(data)
        # Clear instances from self.config so they are rebuilt with new configs
        for agent_name in data:
            self.config.pop(f'agent-instance-{agent_name}', None)

    async def init_isaa(self, name='self', build=False, **kwargs):
        if self.initialized:
            self.print(f"Already initialized. Getting agent/builder: {name}")
            # build=True implies getting the builder, build=False (default) implies getting agent instance
            return self.get_agent_builder(name) if build else await self.get_agent(name)

        self.initialized = True
        sys.setrecursionlimit(1500)
        self.load_keys_from_env()

        with Spinner(message="Building Controller", symbols='c'):
            self.controller.init(self.config['controller_file'])
        self.config["controller-init"] = True


        return self.get_agent_builder(name) if build else await self.get_agent(name)

    def show_version(self):
        self.print("Version: ", self.version)
        return self.version

    def on_start(self):

        threading.Thread(target=self.load_to_mem_sync, daemon=True).start()
        self.print("ISAA module started.")

    def load_keys_from_env(self):
        # Update default model names from environment variables
        for key in self.config:
            if key.startswith("DEFAULTMODEL"):
                self.config[key] = os.getenv(key, self.config[key])
        self.config['VAULTS'] = os.getenv("VAULTS")

    def on_exit(self):
        self.app.run_bg_task_advanced(self.cleanup_tools_interfaces)
        # Save agent configurations
        for agent_name, agent_instance in self.config.items():
            if agent_name.startswith('agent-instance-') and agent_instance and isinstance(agent_instance, list) and isinstance(agent_instance[0], FlowAgent):
                self.app.run_bg_task_advanced(asyncio.gather(*[agent_instance.close() for agent_instance in agent_instance]))
                # If agent instance has its own save logic (e.g. cost tracker)
                # asyncio.run(agent_instance.save()) # This might block, consider task group
                # The AgentConfig is already in self.agent_data, which should be saved.
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

    def save_to_mem_sync(self):
        # This used to call agent.save_memory(). FlowAgent does not have this.
        # If AISemanticMemory needs global saving, it should be handled by AISemanticMemory itself.
        # For now, this can be a no-op or save AISemanticMemory instances if managed by Tools.
        memory_instance = self.get_memory()  # Assuming this returns AISemanticMemory
        if hasattr(memory_instance, 'save_all_memories'):  # Hypothetical method
            memory_instance.save_all_memories(f"{get_app().data_dir}/Memory/")
        self.print("Memory saving process initiated")

    def load_to_mem_sync(self):
        # This used to call agent.save_memory(). FlowAgent does not have this.
        # If AISemanticMemory needs global saving, it should be handled by AISemanticMemory itself.
        # For now, this can be a no-op or save AISemanticMemory instances if managed by Tools.
        memory_instance = self.get_memory()  # Assuming this returns AISemanticMemory
        if hasattr(memory_instance, 'load_all_memories'):  # Hypothetical method
            memory_instance.load_all_memories(f"{get_app().data_dir}/Memory/")
        self.print("Memory loading process initiated")

    def get_agent_builder(self, name="self", extra_tools=None, add_tools=True, add_base_tools=True, working_directory=None) -> FlowAgentBuilder:
        if name == 'None':
            name = "self"

        if extra_tools is None:
            extra_tools = []

        self.print(f"Creating FlowAgentBuilder: {name}")

        # Create builder with agent-specific configuration
        config = AgentConfig(
            name=name,
            fast_llm_model=self.config.get(f'{name.upper()}MODEL', self.config['FASTMODEL']),
            complex_llm_model=self.config.get(f'{name.upper()}MODEL', self.config['COMPLEXMODEL']),
            system_message="You are a production-ready autonomous agent.",
            temperature=0.7,
            max_tokens_output=2048,
            max_tokens_input=32768,
            use_fast_response=True,
            max_parallel_tasks=3,
            verbose_logging=False
        )

        builder = FlowAgentBuilder(config=config)
        builder._isaa_ref = self  # Store ISAA reference

        # Load existing configuration if available
        agent_config_path = Path(f"{get_app().data_dir}/Agents/{name}/agent.json")
        if agent_config_path.exists():
            try:
                builder = FlowAgentBuilder.from_config_file(str(agent_config_path))
                builder._isaa_ref = self
                self.print(f"Loaded existing configuration for builder {name}")
            except Exception as e:
                self.print(f"Failed to load config for {name}: {e}. Using defaults.")

        # Apply global settings
        if self.global_stream_override:
            builder.verbose(True)

        # Apply custom setter if available
        if self.default_setter:
            builder = self.default_setter(builder, name)

        # Initialize ToolsInterface for this agent
        if not hasattr(self, 'tools_interfaces'):
            self.tools_interfaces = {}

        # Create or get existing ToolsInterface for this agent
        if name not in self.tools_interfaces:
            try:
                # Initialize ToolsInterface
                p = Path(get_app().data_dir) / "Agents" / name / "tools_session"
                p.mkdir(parents=True, exist_ok=True)
                tools_interface = ToolsInterface(
                    session_dir=str(Path(get_app().data_dir) / "Agents" / name / "tools_session"),
                    auto_remove=False,  # Keep session data for agents
                    variables={
                        'agent_name': name,
                        'isaa_instance': self
                    },
                    variable_manager=getattr(self, 'variable_manager', None),
                )
                if working_directory:
                    tools_interface.set_base_directory(working_directory)

                self.tools_interfaces[name] = tools_interface
                self.print(f"Created ToolsInterface for agent: {name}")

            except Exception as e:
                self.print(f"Failed to create ToolsInterface for {name}: {e}")
                self.tools_interfaces[name] = None

        tools_interface = self.tools_interfaces[name]

        # Add ISAA core tools
        async def run_isaa_agent_tool(target_agent_name: str, instructions: str, **kwargs_):
            if not instructions:
                return "No instructions provided."
            if target_agent_name.startswith('"') and target_agent_name.endswith('"') or target_agent_name.startswith(
                "'") and target_agent_name.endswith("'"):
                target_agent_name = target_agent_name[1:-1]
            return await self.run_agent(target_agent_name, text=instructions, **kwargs_)

        async def memory_search_tool(
            query: str,
            search_mode: str | None = "balanced",
            context_name: str | None = None
        ) -> str:
            """Memory search with configurable precision"""
            mem_instance = self.get_memory()
            memory_names_list = [name.strip() for name in context_name.split(',')] if context_name else None

            search_params = {
                "wide": {"k": 7, "min_similarity": 0.1, "cross_ref_depth": 3, "max_cross_refs": 4, "max_sentences": 8},
                "narrow": {"k": 2, "min_similarity": 0.75, "cross_ref_depth": 1, "max_cross_refs": 1,
                           "max_sentences": 3},
                "balanced": {"k": 3, "min_similarity": 0.2, "cross_ref_depth": 2, "max_cross_refs": 2,
                             "max_sentences": 5}
            }.get(search_mode,
                  {"k": 3, "min_similarity": 0.2, "cross_ref_depth": 2, "max_cross_refs": 2, "max_sentences": 5})

            return await mem_instance.query(
                query=query, memory_names=memory_names_list,
                query_params=search_params, to_str=True
            )

        async def save_to_memory_tool(data_to_save: str, context_name: str = name):
            mem_instance = self.get_memory()
            result = await mem_instance.add_data(context_name, str(data_to_save), direct=True)
            return 'Data added to memory.' if result else 'Error adding data to memory.'

        # Add ISAA core tools


        if add_base_tools:
            builder.add_tool(memory_search_tool, "memorySearch", "Search ISAA's semantic memory")
            builder.add_tool(save_to_memory_tool, "saveDataToMemory", "Save data to ISAA's semantic memory")
            builder.add_tool(self.web_search, "searchWeb", "Search the web for information")
            builder.add_tool(self.shell_tool_function, "shell", f"Run shell command in {detect_shell()}")

        # Add ToolsInterface tools dynamically
        if add_tools and tools_interface:
            try:
                # Get all tools from ToolsInterface
                interface_tools = tools_interface.get_tools()

                # Determine which tools to add based on agent name/type
                tool_categories = {
                    'code': ['execute_python', 'install_package'],
                    'file': ['write_file', 'replace_in_file', 'read_file', 'list_directory', 'create_directory'],
                    'session': ['get_execution_history', 'clear_session', 'get_variables'],
                    'config': ['set_base_directory', 'set_current_file']
                }

                # Determine which categories to include
                include_categories = set()
                name_lower = name.lower()

                # Code execution for development/coding agents
                if any(keyword in name_lower for keyword in ["dev", "code", "program", "script", "python", "rust", "worker"]):
                    include_categories.update(['code', 'file', 'session', 'config'])

                # Web tools for web-focused agents
                if any(keyword in name_lower for keyword in ["web", "browser", "scrape", "crawl", "extract"]):
                    include_categories.update(['file', 'session'])

                # File tools for file management agents
                if any(keyword in name_lower for keyword in ["file", "fs", "document", "write", "read"]):
                    include_categories.update(['file', 'session', 'config'])

                # Default: add core tools for general agents
                if not include_categories or name == "self":
                    include_categories.update(['code', 'file', 'session', 'config'])

                # Add selected tools
                tools_added = 0
                for tool_func, tool_name, tool_description in interface_tools:
                    # Check if this tool should be included
                    should_include = tool_name in extra_tools

                    if not should_include:
                        for category, tool_names in tool_categories.items():
                            if category in include_categories and tool_name in tool_names:
                                should_include = True
                                break

                    # Always include session management tools
                    if tool_name in ['get_execution_history', 'get_variables']:
                        should_include = True

                    if should_include:
                        try:
                            builder.add_tool(tool_func, tool_name, tool_description)
                            tools_added += 1
                        except Exception as e:
                            self.print(f"Failed to add tool {tool_name}: {e}")

                self.print(f"Added {tools_added} ToolsInterface tools to agent {name}")

            except Exception as e:
                self.print(f"Error adding ToolsInterface tools to {name}: {e}")

        # Configure cost tracking
        builder.with_budget_manager(max_cost=100.0)

        # Store agent configuration
        try:
            agent_dir = Path(f"{get_app().data_dir}/Agents/{name}")
            agent_dir.mkdir(parents=True, exist_ok=True)

            # Save agent metadata
            metadata = {
                'name': name,
                'created_at': time.time(),
                'tools_interface_available': tools_interface is not None,
                'session_dir': str(agent_dir / "tools_session")
            }

            metadata_file = agent_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            self.print(f"Failed to save agent metadata for {name}: {e}")

        return builder

    def get_tools_interface(self, agent_name: str = "self") -> ToolsInterface | None:
        """
        Get the ToolsInterface instance for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            ToolsInterface instance or None if not found
        """
        if not hasattr(self, 'tools_interfaces'):
            return None

        return self.tools_interfaces.get(agent_name)

    async def configure_tools_interface(self, agent_name: str, **kwargs) -> bool:
        """
        Configure the ToolsInterface for a specific agent.

        Args:
            agent_name: Name of the agent
            **kwargs: Configuration parameters

        Returns:
            True if successful, False otherwise
        """
        tools_interface = self.get_tools_interface(agent_name)
        if not tools_interface:
            self.print(f"No ToolsInterface found for agent {agent_name}")
            return False

        try:
            # Configure based on provided parameters
            if 'base_directory' in kwargs:
                await tools_interface.set_base_directory(kwargs['base_directory'])

            if 'current_file' in kwargs:
                await tools_interface.set_current_file(kwargs['current_file'])

            if 'variables' in kwargs:
                tools_interface.ipython.user_ns.update(kwargs['variables'])

            self.print(f"Configured ToolsInterface for agent {agent_name}")
            return True

        except Exception as e:
            self.print(f"Failed to configure ToolsInterface for {agent_name}: {e}")
            return False

    async def cleanup_tools_interfaces(self):
        """
        Cleanup all ToolsInterface instances.
        """
        if not hasattr(self, 'tools_interfaces'):
            return

        async def cleanup_async():
            for name, tools_interface in self.tools_interfaces.items():
                if tools_interface:
                    try:
                        await tools_interface.__aexit__(None, None, None)
                    except Exception as e:
                        self.print(f"Error cleaning up ToolsInterface for {name}: {e}")

        # Run cleanup
        try:
            await cleanup_async()
            self.tools_interfaces.clear()
            self.print("Cleaned up all ToolsInterface instances")
        except Exception as e:
            self.print(f"Error during ToolsInterface cleanup: {e}")

    async def register_agent(self, agent_builder: FlowAgentBuilder):
        agent_name = agent_builder.config.name

        if f'agent-instance-{agent_name}' in self.config:
            self.print(f"Agent '{agent_name}' instance already exists. Overwriting config and rebuilding on next get.")
            self.config.pop(f'agent-instance-{agent_name}', None)

        # Save the builder's configuration
        config_path = Path(f"{get_app().data_dir}/Agents/{agent_name}/agent.json")
        agent_builder.save_config(str(config_path), format='json')
        self.print(f"Saved FlowAgentBuilder config for '{agent_name}' to {config_path}")

        # Store serializable config in agent_data
        self.agent_data[agent_name] = agent_builder.config.model_dump()

        if agent_name not in self.config.get("agents-name-list", []):
            if "agents-name-list" not in self.config:
                self.config["agents-name-list"] = []
            self.config["agents-name-list"].append(agent_name)

        self.print(f"FlowAgent '{agent_name}' configuration registered. Will be built on first use.")
        row_agent_builder_sto[agent_name] = agent_builder  # Cache builder

    async def get_agent(self, agent_name="Normal", model_override: str | None = None) -> 'FlowAgent':
        if "agents-name-list" not in self.config:
            self.config["agents-name-list"] = []

        instance_key = f'agent-instance-{agent_name}'
        if instance_key in self.config:
            agent_instance = self.config[instance_key]
            if model_override and agent_instance.amd.fast_llm_model != model_override:
                self.print(f"Model override for {agent_name}: {model_override}. Rebuilding.")
                self.config.pop(instance_key, None)
            else:
                self.print(f"Returning existing FlowAgent instance: {agent_name}")
                return agent_instance

        builder_to_use = None

        # Try to get cached builder first
        if agent_name in row_agent_builder_sto:
            builder_to_use = row_agent_builder_sto[agent_name]
            self.print(f"Using cached builder for {agent_name}")

        # Try to load from stored config
        elif agent_name in self.agent_data:
            self.print(f"Loading configuration for FlowAgent: {agent_name}")
            try:
                config = AgentConfig(**self.agent_data[agent_name])
                builder_to_use = FlowAgentBuilder(config=config)
            except Exception as e:
                self.print(f"Error loading config for {agent_name}: {e}. Falling back to default.")

        # Create default builder if none found
        if builder_to_use is None:
            self.print(f"No existing config for {agent_name}. Creating default builder.")
            builder_to_use = self.get_agent_builder(agent_name)

        # Apply overrides and ensure correct name
        builder_to_use._isaa_ref = self
        if model_override:
            builder_to_use.with_models(model_override, model_override)

        if builder_to_use.config.name != agent_name:
            builder_to_use.with_name(agent_name)

        self.print(
            f"Building FlowAgent: {agent_name} with models {builder_to_use.config.fast_llm_model} - {builder_to_use.config.complex_llm_model}")

        # Build the agent
        agent_instance: FlowAgent = await builder_to_use.build()


        # if interface := self.get_tools_interface(agent_name):
        #     interface.variable_manager = agent_instance.variable_manager
#
        # # colletive cabability cahring for reduched reduanda analysis _tool_capabilities
        # agent_tool_nams = set(agent_instance.tool_registry.keys())
#
        # tools_data = {}
        # for _agent_name in self.config["agents-name-list"]:
        #     _instance_key = f'agent-instance-{_agent_name}'
        #     if _instance_key not in self.config:
        #         if agent_name != "self" and _agent_name == "self":
        #             await self.get_agent("self")
#
        #     if _instance_key not in self.config:
        #         continue
        #     _agent_instance = self.config[_instance_key]
        #     _agent_tool_nams = set(_agent_instance._tool_capabilities.keys())
        #     # extract the tool names that are in both agents_registry
        #     overlap_tool_nams = agent_tool_nams.intersection(_agent_tool_nams)
        #     _tc = _agent_instance._tool_capabilities
        #     for tool_name in overlap_tool_nams:
        #         if tool_name not in _tc:
        #             continue
        #         tools_data[tool_name] = _tc[tool_name]
#
        # agent_instance._tool_capabilities.update(tools_data)
        # Cache the instance and update tracking
        self.config[instance_key] = agent_instance
        if agent_name not in self.agent_data:
            self.agent_data[agent_name] = builder_to_use.config.model_dump()
        if agent_name not in self.config["agents-name-list"]:
            self.config["agents-name-list"].append(agent_name)

        self.print(f"Built and cached FlowAgent instance: {agent_name}")
        return agent_instance

    @export(api=True, version=version, request_as_kwarg=True, mod_name="isaa")
    async def mini_task_completion(self, mini_task: str | None = None, user_task: str | None = None, mode: Any = None,  # LLMMode
                                   max_tokens_override: int | None = None, task_from="system",
                                   stream_function: Callable | None = None, message_history: list | None = None, agent_name="TaskCompletion", use_complex: bool = False, request: RequestData | None = None, form_data: dict | None = None, data: dict | None = None, **kwargs):
        if request is not None or form_data is not None or data is not None:
            data_dict = (request.request.body if request else None) or form_data or data
            mini_task = mini_task or  data_dict.get("mini_task")
            user_task = user_task or data_dict.get("user_task")
            mode = mode or data_dict.get("mode")
            max_tokens_override = max_tokens_override or data_dict.get("max_tokens_override")
            task_from = data_dict.get("task_from") or task_from
            agent_name = data_dict.get("agent_name") or agent_name
            use_complex = use_complex or data_dict.get("use_complex")
            kwargs = kwargs or data_dict.get("kwargs")
            message_history = message_history or data_dict.get("message_history")
            if isinstance(message_history, str):
                message_history = json.loads(message_history)
        print(mini_task, agent_name, use_complex, kwargs, message_history, form_data or data)
        if mini_task is None: return None
        if agent_name is None: return None
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
        if use_complex:
            llm_params = {"model": agent.amd.complex_llm_model, "messages": messages}
        else:
            llm_params = {"model": agent.amd.fast_llm_model if agent.amd.use_fast_response else agent.amd.complex_llm_model, "messages": messages}
        if max_tokens_override:
            llm_params['max_tokens'] = max_tokens_override
        else:
            llm_params['max_tokens'] = agent.amd.max_tokens
        if kwargs:
            llm_params.update(kwargs)  # Add any additional kwargs
        if stream_function:
            llm_params['stream'] = True
            # FlowAgent a_run_llm_completion handles stream_callback via agent.stream_callback
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
                                          task_from="system", mode_overload: Any = None, user_task: str | None = None, auto_context=False, **kwargs):
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
                auto_context=auto_context
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

    @export(api=True, version=version, name="version")
    async def get_version(self, *a,**k):
        return self.version

    @export(api=True, version=version, request_as_kwarg=True, mod_name="isaa")
    async def format_class(self, format_schema: type[BaseModel] | None = None, task: str | None = None, agent_name="TaskCompletion", auto_context=False, request: RequestData | None = None, form_data: dict | None = None, data: dict | None = None, **kwargs):
        if request is not None or form_data is not None or data is not None:
            data_dict = (request.request.body if request else None) or form_data or data
            format_schema = format_schema or data_dict.get("format_schema")
            task = task or data_dict.get("task")
            agent_name = data_dict.get("agent_name") or agent_name
            auto_context = auto_context or data_dict.get("auto_context")
            kwargs = kwargs or data_dict.get("kwargs")
        if format_schema is None or not task: return None
        agent = None
        if isinstance(agent_name, str):
            agent = await self.get_agent(agent_name)
        elif isinstance(agent_name, FlowAgent):
            agent = agent_name
        else:
            raise TypeError("agent_name must be str or FlowAgent instance")

        return await agent.a_format_class(format_schema, task, auto_context=auto_context)

    async def run_agent(self, name: str | FlowAgent,
                        text: str,
                        verbose: bool = False,  # Handled by agent's own config mostly
                        session_id: str | None = None,
                        progress_callback: Callable[[Any], None | Awaitable[None]] | None = None,
                        **kwargs):  # Other kwargs for a_run
        if text is None: return ""
        if name is None: return ""
        if text == "test": return ""

        agent_instance = None
        if isinstance(name, str):
            agent_instance = await self.get_agent(name)
        elif isinstance(name, FlowAgent):
            agent_instance = name
        else:
            return self.return_result().default_internal_error(
                f"Invalid agent identifier type: {type(name)}")

        self.print(f"Running agent {agent_instance.amd.name} for task: {text[:100]}...")
        save_p = None
        if progress_callback:
            save_p = agent_instance.progress_callback
            agent_instance.progress_callback = progress_callback

        if verbose:
            agent_instance.verbose = True

        # Call FlowAgent's a_run method
        response = await agent_instance.a_run(
            query=text,
            session_id=session_id,
            user_id=None,
            stream_callback=None

        )
        if save_p:
            agent_instance.progress_callback = save_p

        return response

    # mass_text_summaries and related methods remain complex and depend on AISemanticMemory
    # and specific summarization strategies. For now, keeping their structure,
    # but calls to self.format_class or self.mini_task_completion will become async.

    async def mas_text_summaries(self, text, min_length=36000, ref=None, max_tokens_override=None):
        len_text = len(text)
        if len_text < min_length: return text
        key = self.one_way_hash(text, 'summaries', 'isaa')
        value = self.mas_text_summaries_dict.get(key)
        if value is not None: return value

        # This part needs to become async due to format_class
        # Simplified version:
        from .extras.modes import (
            SummarizationMode,
            # crate_llm_function_from_langchain_tools,
        )
        summary = await self.mini_task_completion(
            mini_task=f"Summarize this text, focusing on aspects related to '{ref if ref else 'key details'}'. The text is: {text}",
            mode=self.controller.rget(SummarizationMode), max_tokens_override=max_tokens_override, agent_name="self")

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

    async def save_all_memory_vis(self, dir_path=None):
        if dir_path is None:
            dir_path = f"{get_app().data_dir}/Memory/vis"
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        self.load_to_mem_sync()
        for name, kb in self.get_memory().memories.items():
            self.print(f"Saving to {name}.html with {len(kb.concept_extractor.concept_graph.concepts)} concepts")
            await kb.vis(output_file=f"{dir_path}/{name}.html")
        return dir_path


def shell_tool_function(command: str) -> str:
    result: dict[str, Any] = {"success": False, "output": "", "error": ""}
    # auto python
    tokens = shlex.split(command)

    # Replace "python" or "python3" only if it’s a standalone command
    for i, tok in enumerate(tokens):
        if tok in ("python", "python3"):
            tokens[i] = sys.executable

    # Rebuild the command string
    command = " ".join(shlex.quote(t) for t in tokens)
    try:
        shell_exe, cmd_flag = detect_shell()

        process = subprocess.run(
            [shell_exe, cmd_flag, command],
            capture_output=True,
            text=False,
            timeout=120,
            check=False
        )

        stdout = remove_styles(safe_decode(process.stdout))
        stderr = remove_styles(safe_decode(process.stderr))

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

@export(mod_name="isaa", name="listAllAgents", api=True, request_as_kwarg=True)
async def list_all_agents(self, request: RequestData | None = None):
    return self.config.get("agents-name-list", [])


if __name__ == "__main__":
    # Example of running an async method from Tools if needed for testing
    async def test_isaa_tools():
        app_instance = get_app("isaa_test_app")
        isaa_tool_instance = Tools(app=app_instance)
        await isaa_tool_instance.init_isaa()

        # Test get_agent
        self_agent = await isaa_tool_instance.get_agent("self")
        print(f"Got agent: {self_agent.amd.name} with model {self_agent.amd.fast_llm_model} and {self_agent.amd.complex_llm_model}")

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
