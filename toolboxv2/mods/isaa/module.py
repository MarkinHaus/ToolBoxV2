"""
ISAA Module - Refactored V2

Changes from V1:
- Removed ToolsInterface (obsolete)
- Added native Chain support with helper methods
- Added Agent Export/Import system (.tar.gz with dill serialization)
- Cleaned up unused code
- Preserved all existing APIs (mini_task_completion, format_class, etc.)

Author: FlowAgent V2
"""

import asyncio
import copy
import io
import json
import os
import platform
import queue
import secrets
import shlex
import tarfile
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Dict
from collections.abc import Awaitable

import requests
import uuid
from langchain_community.agent_toolkits.load_tools import load_tools
from pydantic import BaseModel

from toolboxv2.mods.isaa.base.Agent.types import ProgressEvent
from toolboxv2.utils.system import FileCache
from toolboxv2.utils.toolbox import stram_print

import subprocess
import sys

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

# FlowAgent imports
from .base.Agent.flow_agent import FlowAgent
from .base.Agent.builder import AgentConfig, FlowAgentBuilder

# Chain imports - native support
from .base.Agent.chain import (
    Chain,
    ParallelChain,
    ConditionalChain,
    ErrorHandlingChain,
    ChainBase,
    CF,
    IS,
    Function,
)

from .base.AgentUtils import (
    AISemanticMemory,
    ControllerManager,
    detect_shell,
    safe_decode,
)

# Optional dill import for tool serialization
try:
    import dill

    DILL_AVAILABLE = True
except ImportError:
    DILL_AVAILABLE = False
    dill = None

# Optional cloudpickle as fallback
try:
    import cloudpickle

    CLOUDPICKLE_AVAILABLE = True
except ImportError:
    CLOUDPICKLE_AVAILABLE = False
    cloudpickle = None

PIPLINE = None
Name = 'isaa'
version = "0.3.0"  # Version bump for refactoring
export = get_app("isaa.Export").tb
pipeline_arr = [
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


# =============================================================================
# TOOL SERIALIZATION HELPERS
# =============================================================================

class ToolSerializationError(Exception):
    """Raised when a tool cannot be serialized"""
    pass


class ToolSerializationInfo(BaseModel):
    """Information about a tool's serialization status"""
    name: str
    serializable: bool
    error_message: str | None = None
    source_hint: str | None = None  # Hint for manual recreation
    module_path: str | None = None
    function_name: str | None = None


def _get_serializer():
    """Get the best available serializer"""
    if DILL_AVAILABLE:
        return dill
    if CLOUDPICKLE_AVAILABLE:
        return cloudpickle
    return None


def _serialize_tool(func: Callable, name: str) -> tuple[bytes | None, ToolSerializationInfo]:
    """
    Attempt to serialize a tool function.

    Returns:
        Tuple of (serialized_bytes or None, ToolSerializationInfo)
    """
    serializer = _get_serializer()

    info = ToolSerializationInfo(
        name=name,
        serializable=False,
        module_path=getattr(func, '__module__', None),
        function_name=getattr(func, '__name__', name)
    )

    if serializer is None:
        info.error_message = "No serializer available. Install 'dill' or 'cloudpickle': pip install dill"
        info.source_hint = f"Recreate manually: from {info.module_path} import {info.function_name}"
        return None, info

    try:
        # Try to serialize
        serialized = serializer.dumps(func)
        info.serializable = True
        return serialized, info
    except Exception as e:
        info.error_message = f"Serialization failed: {str(e)}"

        # Provide helpful hints based on common issues
        if "closure" in str(e).lower():
            info.source_hint = f"Tool '{name}' is a closure. Define it at module level or recreate after loading."
        elif "lambda" in str(e).lower():
            info.source_hint = f"Tool '{name}' is a lambda. Convert to named function or recreate after loading."
        elif info.module_path:
            info.source_hint = f"Try: from {info.module_path} import {info.function_name}"
        else:
            info.source_hint = f"Recreate the tool '{name}' manually after loading the agent."

        return None, info


def _deserialize_tool(data: bytes, name: str) -> tuple[Callable | None, str | None]:
    """
    Attempt to deserialize a tool function.

    Returns:
        Tuple of (function or None, error_message or None)
    """
    serializer = _get_serializer()

    if serializer is None:
        return None, "No serializer available. Install 'dill' or 'cloudpickle'"

    try:
        func = serializer.loads(data)
        return func, None
    except Exception as e:
        return None, f"Deserialization failed for '{name}': {str(e)}"


# =============================================================================
# AGENT EXPORT/IMPORT CLASSES
# =============================================================================

class AgentExportManifest(BaseModel):
    """Manifest file for exported agent archive"""
    version: str = "1.0"
    export_date: str
    agent_name: str
    agent_version: str
    has_checkpoint: bool
    has_tools: bool
    tool_count: int
    serializable_tools: list[str]
    non_serializable_tools: list[ToolSerializationInfo]
    bindings: list[str]  # Names of bound agents
    notes: str | None = None


class AgentNetworkManifest(BaseModel):
    """Manifest for multi-agent network export"""
    version: str = "1.0"
    export_date: str
    agents: list[str]
    bindings: dict[str, list[str]]  # agent_name -> [bound_agent_names]
    entry_agent: str | None = None


# =============================================================================
# MAIN TOOLS CLASS
# =============================================================================

class Tools(MainTool):

    def __init__(self, app=None):
        self.run_callback = None

        if app is None:
            app = get_app("isaa-mod")

        self.version = version
        self.name = "isaa"
        self.Name = "isaa"
        self.color = "VIOLET2"
        self.config = {
            'controller-init': False,
            'agents-name-list': [],
            "FASTMODEL": os.getenv("FASTMODEL", "ollama/llama3.1"),
            "AUDIOMODEL": os.getenv("AUDIOMODEL", "groq/whisper-large-v3-turbo"),
            "BLITZMODEL": os.getenv("BLITZMODEL", "ollama/llama3.1"),
            "COMPLEXMODEL": os.getenv("COMPLEXMODEL", "ollama/llama3.1"),
            "SUMMARYMODEL": os.getenv("SUMMARYMODEL", "ollama/llama3.1"),
            "IMAGEMODEL": os.getenv("IMAGEMODEL", "ollama/llama3.1"),
            "DEFAULTMODELEMBEDDING": os.getenv("DEFAULTMODELEMBEDDING", "gemini/text-embedding-004"),
        }
        self.per_data = {}
        self.agent_data: dict[str, dict] = {}
        self.keys = {
            "KEY": "key~~~~~~~",
            "Config": "config~~~~"
        }
        self.initstate = {}

        extra_path = ""
        if self.toolID:
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
            "format_class": self.format_class,
            "get_memory": self.get_memory,
            "save_all_memory_vis": self.save_all_memory_vis,
            "rget_mode": lambda mode: self.controller.rget(mode),
            "kernel_start": kernel_start,
            # Chain helpers
            "create_chain": self.create_chain,
            "run_chain": self.run_chain,
            # Agent export/import
            "save_agent": self.save_agent,
            "load_agent": self.load_agent,
            "export_agent_network": self.export_agent_network,
            "import_agent_network": self.import_agent_network,
        }

        self.working_directory = os.getenv('ISAA_WORKING_PATH', os.getcwd())
        self.print_stream = stram_print
        self.global_stream_override = False
        self.global_verbose_override = False

        self.agent_memory: AISemanticMemory = f"{app.id}{extra_path}/Memory"
        self.controller = ControllerManager({})
        self.summarization_mode = 1
        self.summarization_limiter = 102000
        self.speak = lambda x, *args, **kwargs: x

        self.default_setter = None
        self.initialized = False

        self.file_handler = FileHandler(f"isaa{extra_path.replace('/', '-')}.config", app.id if app else __name__)
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=None, color=self.color, on_exit=self.on_exit)

        from .extras.web_search import web_search

        async def web_search_tool(query: str) -> str:
            res = web_search(query)
            return await self.mas_text_summaries(str(res), min_length=12000, ref=query)

        self.web_search = web_search_tool
        self.shell_tool_function = shell_tool_function

        self.print(f"Start {self.spec}.isaa")
        with Spinner(message="Starting module", symbols='c'):
            self.file_handler.load_file_handler()
            config_fh = self.file_handler.get_file_handler(self.keys["Config"])
            if config_fh is not None:
                if isinstance(config_fh, str):
                    try:
                        config_fh = json.loads(config_fh)
                    except json.JSONDecodeError:
                        self.print(f"Warning: Could not parse config from file handler: {config_fh[:100]}...")
                        config_fh = {}

                if isinstance(config_fh, dict):
                    loaded_config = config_fh
                    for key, value in self.config.items():
                        if key not in loaded_config:
                            loaded_config[key] = value
                    self.config = loaded_config

            if self.spec == 'app':
                self.load_keys_from_env()
                from .extras.agent_ui import initialize
                initialize(self.app)

                self.app.run_any(
                    ("CloudM", "add_ui"),
                    name="AgentUI",
                    title="FlowAgent Chat",
                    description="Chat with your FlowAgents",
                    path="/api/Minu/render?view=agent_ui&ssr=true",
                )

            Path(f"{get_app('isaa-initIsaa').data_dir}/Agents/").mkdir(parents=True, exist_ok=True)
            Path(f"{get_app('isaa-initIsaa').data_dir}/Memory/").mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # CHAIN SUPPORT - Helper Methods
    # =========================================================================

    def create_chain(self, *agents_or_components) -> Chain:
        """
        Create a Chain from agents and/or components.

        Usage:
            # Simple sequential chain
            chain = isaa.create_chain(agent1, agent2, agent3)

            # With formatting
            chain = isaa.create_chain(agent1, CF(MyModel), agent2)

            # With conditions
            chain = isaa.create_chain(agent1, IS("key", "value"), agent2)

            # Mixed with functions
            chain = isaa.create_chain(agent1, lambda x: x.upper(), agent2)

        Returns:
            Chain object ready for execution
        """
        if len(agents_or_components) == 0:
            return Chain()

        if len(agents_or_components) == 1:
            comp = agents_or_components[0]
            if isinstance(comp, Chain):
                return comp
            return Chain(comp) if hasattr(comp, 'a_run') else Chain._create_chain([comp])

        # Build chain from components
        chain = Chain()
        chain.tasks = list(agents_or_components)
        return chain

    async def run_chain(
        self,
        chain: Chain | list,
        query: str,
        session_id: str = "default",
        **kwargs
    ) -> Any:
        """
        Execute a chain with the given query.

        Args:
            chain: Chain object or list of components
            query: Initial query/data
            session_id: Session ID for all agents in the chain
            **kwargs: Additional arguments passed to chain execution

        Returns:
            Final result from chain execution
        """
        if isinstance(chain, list):
            chain = self.create_chain(*chain)

        if chain is None:
            print("Chain is None")
            return None

        return await chain.a_run(query, session_id=session_id, **kwargs)

    def chain_from_agents(self, *agent_names: str) -> Chain:
        """
        Create a chain from agent names (will be resolved on execution).

        Usage:
            chain = isaa.chain_from_agents("analyzer", "summarizer", "formatter")
        """

        async def get_agents():
            agents = []
            for name in agent_names:
                agent = await self.get_agent(name)
                agents.append(agent)
            return agents

        # Return a lazy chain that resolves agents on first run
        class LazyAgentChain(Chain):
            def __init__(self, isaa_ref, names):
                super().__init__()
                self._isaa_ref = isaa_ref
                self._agent_names = names
                self._resolved = False

            async def a_run(self, query, **kwargs):
                if not self._resolved:
                    for name in self._agent_names:
                        agent = await self._isaa_ref.get_agent(name)
                        self.tasks.append(agent)
                    self._resolved = True
                return await super().a_run(query, **kwargs)

        return LazyAgentChain(self, agent_names)

    # =========================================================================
    # AGENT EXPORT/IMPORT SYSTEM
    # =========================================================================

    async def save_agent(
        self,
        agent_name: str,
        path: str,
        include_checkpoint: bool = True,
        include_tools: bool = True,
        notes: str | None = None
    ) -> tuple[bool, AgentExportManifest | str]:
        """
        Export an agent to a .tar.gz archive.

        Archive structure:
            agent_name.tar.gz/
            ├── manifest.json        # Export metadata
            ├── config.json          # AgentConfig
            ├── checkpoint.json      # Optional: Agent state
            ├── tools.dill           # Optional: Serialized tools
            └── tools_manifest.json  # Tool serialization info

        Args:
            agent_name: Name of the agent to export
            path: Output path (will add .tar.gz if not present)
            include_checkpoint: Include agent checkpoint/state
            include_tools: Attempt to serialize tools
            notes: Optional notes to include in manifest

        Returns:
            Tuple of (success: bool, manifest or error_message)
        """
        if agent_name is None:
            return False, "Agent name is required"
        if path is None:
            path = f"{agent_name}.tar.gz"
        if not path.endswith('.tar.gz'):
            path = f"{path}.tar.gz"

        try:
            # Get agent instance and builder config
            agent = await self.get_agent(agent_name)
            builder_config = self.agent_data.get(agent_name, {})

            if not builder_config:
                # Try to get from builder
                if agent_name in row_agent_builder_sto:
                    builder_config = row_agent_builder_sto[agent_name].config.model_dump()
                else:
                    self.print(f"No builder config found for {agent_name}. Creating default. to save")
                    builder_config = AgentConfig(name=agent_name).model_dump()

            # Prepare tool serialization
            serializable_tools = []
            non_serializable_tools = []
            tools_data = {}

            if include_tools and hasattr(agent, 'tool_manager'):
                for tool_name, tool_info in agent.tool_manager.tools.items():
                    func = tool_info.get('function') or tool_info.get('func')
                    if func:
                        serialized, info = _serialize_tool(func, tool_name)
                        if serialized:
                            serializable_tools.append(tool_name)
                            tools_data[tool_name] = {
                                'data': serialized,
                                'description': tool_info.get('description', ''),
                                'category': tool_info.get('category', []),
                            }
                        else:
                            non_serializable_tools.append(info)

            # Prepare checkpoint
            checkpoint_data = None
            if include_checkpoint:
                try:
                    checkpoint_path = await agent.checkpoint_manager.save_current()
                    if checkpoint_path and Path(checkpoint_path).exists():
                        with open(checkpoint_path, 'r') as f:
                            checkpoint_data = json.load(f)
                except Exception as e:
                    self.print(f"Warning: Could not save checkpoint: {e}")

            # Get bindings
            bindings = []
            if hasattr(agent, 'bind_manager'):
                bindings = list(agent.bind_manager.bindings.keys())

            # Create manifest
            manifest = AgentExportManifest(
                export_date=datetime.now().isoformat(),
                agent_name=agent_name,
                agent_version=builder_config.get('version', '1.0.0'),
                has_checkpoint=checkpoint_data is not None,
                has_tools=len(tools_data) > 0,
                tool_count=len(serializable_tools) + len(non_serializable_tools),
                serializable_tools=serializable_tools,
                non_serializable_tools=non_serializable_tools,
                bindings=bindings,
                notes=notes
            )

            # Create tar.gz archive
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            with tarfile.open(path, 'w:gz') as tar:
                # Add manifest
                manifest_bytes = manifest.model_dump_json(indent=2).encode('utf-8')
                self._add_bytes_to_tar(tar, 'manifest.json', manifest_bytes)

                # Add config
                config_bytes = json.dumps(builder_config, indent=2).encode('utf-8')
                self._add_bytes_to_tar(tar, 'config.json', config_bytes)

                # Add checkpoint
                if checkpoint_data:
                    checkpoint_bytes = json.dumps(checkpoint_data, indent=2).encode('utf-8')
                    self._add_bytes_to_tar(tar, 'checkpoint.json', checkpoint_bytes)

                # Add serialized tools
                if tools_data:
                    serializer = _get_serializer()
                    if serializer:
                        tools_bytes = serializer.dumps(tools_data)
                        self._add_bytes_to_tar(tar, 'tools.dill', tools_bytes)

                # Add tools manifest (human readable)
                tools_manifest = {
                    'serializable': serializable_tools,
                    'non_serializable': [t.model_dump() for t in non_serializable_tools]
                }
                tools_manifest_bytes = json.dumps(tools_manifest, indent=2).encode('utf-8')
                self._add_bytes_to_tar(tar, 'tools_manifest.json', tools_manifest_bytes)

            self.print(f"Agent '{agent_name}' exported to {path}")
            self.print(f"  - Tools: {len(serializable_tools)} serialized, {len(non_serializable_tools)} manual")

            return True, manifest

        except Exception as e:
            error_msg = f"Failed to export agent '{agent_name}': {str(e)}"
            self.print(error_msg)
            return False, error_msg

    async def load_agent(
        self,
        path: str,
        override_name: str | None = None,
        load_tools: bool = True,
        register: bool = True
    ) -> tuple[FlowAgent | None, AgentExportManifest | None, list[str]]:
        """
        Import an agent from a .tar.gz archive.

        Args:
            path: Path to the archive
            override_name: Optional new name for the agent
            load_tools: Attempt to deserialize and register tools
            register: Register the agent in ISAA

        Returns:
            Tuple of (agent or None, manifest or None, list of warnings)
        """
        warnings = []

        if override_name is None:
            return None, None, ["No agent name specified"]
        if path is None:
            path = f"{override_name}.tar.gz"

        if not Path(path).exists():
            return None, None, [f"Archive not found: {path}"]

        try:
            with tarfile.open(path, 'r:gz') as tar:
                # Read manifest
                manifest_data = self._read_from_tar(tar, 'manifest.json')
                if not manifest_data:
                    return None, None, ["Invalid archive: missing manifest.json"]
                manifest = AgentExportManifest(**json.loads(manifest_data))

                # Read config
                config_data = self._read_from_tar(tar, 'config.json')
                if not config_data:
                    return None, None, ["Invalid archive: missing config.json"]
                config_dict = json.loads(config_data)

                # Override name if requested
                agent_name = override_name or manifest.agent_name
                config_dict['name'] = agent_name

                # Create builder and agent
                config = AgentConfig(**config_dict)
                builder = FlowAgentBuilder(config=config)
                builder._isaa_ref = self

                # Load tools
                if load_tools and manifest.has_tools:
                    tools_bytes = self._read_from_tar(tar, 'tools.dill', binary=True)
                    if tools_bytes:
                        serializer = _get_serializer()
                        if serializer:
                            try:
                                tools_data = serializer.loads(tools_bytes)
                                for tool_name, tool_info in tools_data.items():
                                    func_data = tool_info.get('data')
                                    if func_data:
                                        func, error = _deserialize_tool(func_data, tool_name)
                                        if func:
                                            builder.add_tool(
                                                func,
                                                name=tool_name,
                                                description=tool_info.get('description', ''),
                                                category=tool_info.get('category')
                                            )
                                        else:
                                            warnings.append(f"Tool '{tool_name}': {error}")
                            except Exception as e:
                                warnings.append(f"Failed to load tools: {str(e)}")
                        else:
                            warnings.append("No serializer available for tools. Install 'dill'.")

                # Report non-serializable tools
                for tool_info in manifest.non_serializable_tools:
                    hint = tool_info.source_hint or f"Recreate tool '{tool_info.name}' manually"
                    warnings.append(f"Tool '{tool_info.name}' not loaded: {hint}")

                # Build agent
                agent = await builder.build()

                # Load checkpoint
                if manifest.has_checkpoint:
                    checkpoint_data = self._read_from_tar(tar, 'checkpoint.json')
                    if checkpoint_data:
                        try:
                            checkpoint = json.loads(checkpoint_data)
                            # Apply checkpoint to agent
                            if hasattr(agent, 'checkpoint_manager'):
                                await agent.checkpoint_manager.restore_from_dict(checkpoint)
                        except Exception as e:
                            warnings.append(f"Failed to restore checkpoint: {str(e)}")

                # Register agent
                if register:
                    self.agent_data[agent_name] = config_dict
                    self.config[f'agent-instance-{agent_name}'] = agent
                    if agent_name not in self.config.get("agents-name-list", []):
                        self.config.setdefault("agents-name-list", []).append(agent_name)

                self.print(f"Agent '{agent_name}' loaded from {path}")
                if warnings:
                    self.print(f"  Warnings: {len(warnings)}")
                    for w in warnings[:5]:  # Show first 5
                        self.print(f"    - {w}")

                return agent, manifest, warnings

        except Exception as e:
            return None, None, [f"Failed to load agent: {str(e)}"]

    async def export_agent_network(
        self,
        agent_names: list[str],
        path: str,
        entry_agent: str | None = None,
        include_checkpoints: bool = True,
        include_tools: bool = True
    ) -> tuple[bool, str]:
        """
        Export multiple connected agents as a network archive.

        Args:
            agent_names: List of agent names to export
            path: Output path for the network archive
            entry_agent: Optional entry point agent name
            include_checkpoints: Include checkpoints for all agents
            include_tools: Include tool serialization

        Returns:
            Tuple of (success, message)
        """

        if path is None and agent_names is None:
            return False, "No path or agent names specified"

        if path is None and agent_names is not None:
            path = f"network_{agent_names[0]}.tar.gz"

        if not path.endswith('.tar.gz'):
            path = f"{path}.tar.gz"

        try:
            # Collect binding information
            bindings = {}
            for name in agent_names:
                agent = await self.get_agent(name)
                if hasattr(agent, 'bind_manager'):
                    bound_names = [n for n in agent.bind_manager.bindings.keys() if n in agent_names]
                    if bound_names:
                        bindings[name] = bound_names

            # Create network manifest
            network_manifest = AgentNetworkManifest(
                export_date=datetime.now().isoformat(),
                agents=agent_names,
                bindings=bindings,
                entry_agent=entry_agent or (agent_names[0] if agent_names else None)
            )

            # Create temporary directory for individual exports
            with tempfile.TemporaryDirectory() as tmpdir:
                # Export each agent
                for name in agent_names:
                    agent_path = Path(tmpdir) / f"{name}.tar.gz"
                    success, result = await self.save_agent(
                        name,
                        str(agent_path),
                        include_checkpoint=include_checkpoints,
                        include_tools=include_tools
                    )
                    if not success:
                        return False, f"Failed to export agent '{name}': {result}"

                # Create network archive
                Path(path).parent.mkdir(parents=True, exist_ok=True)

                with tarfile.open(path, 'w:gz') as tar:
                    # Add network manifest
                    manifest_bytes = network_manifest.model_dump_json(indent=2).encode('utf-8')
                    self._add_bytes_to_tar(tar, 'network_manifest.json', manifest_bytes)

                    # Add individual agent archives
                    for name in agent_names:
                        agent_archive = Path(tmpdir) / f"{name}.tar.gz"
                        tar.add(str(agent_archive), arcname=f"agents/{name}.tar.gz")

            self.print(f"Agent network exported to {path}")
            self.print(f"  - Agents: {len(agent_names)}")
            self.print(f"  - Entry point: {network_manifest.entry_agent}")

            return True, f"Network with {len(agent_names)} agents exported successfully"

        except Exception as e:
            return False, f"Failed to export network: {str(e)}"

    async def import_agent_network(
        self,
        path: str,
        name_prefix: str = "",
        restore_bindings: bool = True
    ) -> tuple[dict[str, FlowAgent], list[str]]:
        """
        Import a network of agents from an archive.

        Args:
            path: Path to the network archive
            name_prefix: Optional prefix for all agent names
            restore_bindings: Restore agent-to-agent bindings

        Returns:
            Tuple of (dict of name->agent, list of warnings)
        """

        if path is None:
            return {}, ["No path specified"]
        agents = {}
        all_warnings = []

        if not Path(path).exists():
            return {}, [f"Archive not found: {path}"]

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Extract network archive
                with tarfile.open(path, 'r:gz') as tar:
                    tar.extractall(tmpdir)

                # Read network manifest
                manifest_path = Path(tmpdir) / 'network_manifest.json'
                if not manifest_path.exists():
                    return {}, ["Invalid network archive: missing network_manifest.json"]

                with open(manifest_path) as f:
                    network_manifest = AgentNetworkManifest(**json.load(f))

                # Import each agent
                for agent_name in network_manifest.agents:
                    agent_archive = Path(tmpdir) / "agents" / f"{agent_name}.tar.gz"
                    if not agent_archive.exists():
                        all_warnings.append(f"Agent archive missing: {agent_name}")
                        continue

                    new_name = f"{name_prefix}{agent_name}" if name_prefix else agent_name
                    agent, manifest, warnings = await self.load_agent(
                        str(agent_archive),
                        override_name=new_name,
                        load_tools=True,
                        register=True
                    )

                    if agent:
                        agents[new_name] = agent
                    all_warnings.extend(warnings)

                # Restore bindings
                if restore_bindings:
                    for source_name, bound_names in network_manifest.bindings.items():
                        source_full = f"{name_prefix}{source_name}" if name_prefix else source_name
                        if source_full not in agents:
                            continue

                        source_agent = agents[source_full]
                        for target_name in bound_names:
                            target_full = f"{name_prefix}{target_name}" if name_prefix else target_name
                            if target_full in agents:
                                try:
                                    await source_agent.bind(agents[target_full])
                                except Exception as e:
                                    all_warnings.append(f"Failed to bind {source_full} -> {target_full}: {e}")

                self.print(f"Agent network loaded from {path}")
                self.print(f"  - Agents: {len(agents)}/{len(network_manifest.agents)}")

                return agents, all_warnings

        except Exception as e:
            return {}, [f"Failed to import network: {str(e)}"]

    def _add_bytes_to_tar(self, tar: tarfile.TarFile, name: str, data: bytes):
        """Helper to add bytes to a tar archive"""
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mtime = time.time()
        tar.addfile(info, io.BytesIO(data))

    def _read_from_tar(self, tar: tarfile.TarFile, name: str, binary: bool = False) -> bytes | str | None:
        """Helper to read a file from tar archive"""
        try:
            member = tar.getmember(name)
            f = tar.extractfile(member)
            if f:
                data = f.read()
                return data if binary else data.decode('utf-8')
        except KeyError:
            pass
        return None

    # =========================================================================
    # AUGMENT SYSTEM (Simplified - delegates to save_agent/load_agent)
    # =========================================================================

    def get_augment(self):
        """Get augmented data for serialization (legacy compatibility)"""
        return {
            "Agents": self.serialize_all(),
        }

    async def init_from_augment(self, augment, agent_name: str = 'self'):
        """Initialize from augmented data (legacy compatibility)"""
        if isinstance(agent_name, str):
            pass
        elif hasattr(agent_name, 'config'):
            agent_name = agent_name.config.name
        else:
            raise ValueError(f"Invalid agent_name type: {type(agent_name)}")

        a_keys = augment.keys()

        if "Agents" in a_keys:
            agents_configs_dict = augment['Agents']
            self.deserialize_all(agents_configs_dict)
            self.print("Agent configurations loaded.")

        if "tools" in a_keys:
            self.print("Tool configurations noted - will be applied during agent building")

    async def add_langchain_tools_to_builder(self, tools_config: dict, agent_builder: FlowAgentBuilder):
        """Initialize tools from config (legacy compatibility)"""
        lc_tools_names = tools_config.get('lagChinTools', [])
        all_lc_tool_names = list(set(lc_tools_names))

        for tool_name in all_lc_tool_names:
            try:
                loaded_tools = load_tools([tool_name], llm=None)
                for lc_tool_instance in loaded_tools:
                    if hasattr(lc_tool_instance, 'run') and callable(lc_tool_instance.run):
                        agent_builder.add_tool(
                            lc_tool_instance.run,
                            name=lc_tool_instance.name,
                            description=lc_tool_instance.description
                        )
                        self.print(f"Added LangChain tool '{lc_tool_instance.name}' to builder.")
            except Exception as e:
                self.print(f"Failed to load/add LangChain tool '{tool_name}': {e}")

    def serialize_all(self):
        """Returns a copy of agent_data"""
        return copy.deepcopy(self.agent_data)

    def deserialize_all(self, data: dict[str, dict]):
        """Load agent configurations"""
        self.agent_data.update(data)
        for agent_name in data:
            self.config.pop(f'agent-instance-{agent_name}', None)

    # =========================================================================
    # CORE METHODS (Preserved from original)
    # =========================================================================

    async def init_isaa(self, name='self', build=False, **kwargs):
        if self.initialized:
            self.print(f"Already initialized. Getting agent/builder: {name}")
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
        for key in self.config:
            if key.startswith("DEFAULTMODEL"):
                self.config[key] = os.getenv(key, self.config[key])
        self.config['VAULTS'] = os.getenv("VAULTS")

    async def on_exit(self):
        tasks = []
        for agent_name, agent_instance in self.config.items():
            if agent_name.startswith('agent-instance-') and agent_instance:
                if isinstance(agent_instance, FlowAgent):
                     tasks.append(agent_instance.close())

        threading.Thread(target=self.save_to_mem_sync, daemon=True).start()
        await asyncio.gather(*tasks)
        if self.config.get("controller-init"):
            self.controller.save(self.config['controller_file'])
        cleanup_sessions()
        clean_config = {}
        for key, value in self.config.items():
            if key.startswith('agent-instance-'):
                continue
            if key.startswith('LLM-model-'):
                continue
            clean_config[key] = value

        self.file_handler.add_to_save_file_handler(self.keys["Config"], json.dumps(clean_config))
        self.file_handler.save_file_handler()

    def save_to_mem_sync(self):
        memory_instance = self.get_memory()
        if hasattr(memory_instance, 'save_all_memories'):
            memory_instance.save_all_memories(f"{get_app().data_dir}/Memory/")
        self.print("Memory saving process initiated")

    def load_to_mem_sync(self):
        memory_instance = self.get_memory()
        if hasattr(memory_instance, 'load_all_memories'):
            memory_instance.load_all_memories(f"{get_app().data_dir}/Memory/")
        self.print("Memory loading process initiated")

    def get_agent_builder(
        self,
        name="self",
        extra_tools=None,
        add_base_tools=True,
        with_dangerous_shell=False
    ) -> FlowAgentBuilder:
        if name == 'None':
            name = "self"

        if extra_tools is None:
            extra_tools = []

        self.print(f"Creating FlowAgentBuilder: {name}")

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
        builder._isaa_ref = self

        agent_config_path = Path(f"{get_app().data_dir}/Agents/{name}/agent.json")
        if agent_config_path.exists():
            try:
                builder = FlowAgentBuilder.from_config_file(str(agent_config_path))
                builder._isaa_ref = self
                self.print(f"Loaded existing configuration for builder {name}")
            except Exception as e:
                self.print(f"Failed to load config for {name}: {e}. Using defaults.")

        if self.global_stream_override:
            builder.with_stream(True)
        if self.global_verbose_override:
            builder.verbose(True)

        if self.default_setter:
            builder = self.default_setter(builder, name)

        # ISAA core tools
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

        if add_base_tools:
            builder.add_tool(memory_search_tool, "memorySearch", "Search ISAA's semantic memory")
            builder.add_tool(save_to_memory_tool, "saveDataToMemory", "Save data to ISAA's semantic memory")
            builder.add_tool(self.web_search, "searchWeb", "Search the web for information")
            if with_dangerous_shell:
                builder.add_tool(self.shell_tool_function, "shell", f"Run shell command in {detect_shell()}")

        builder.with_budget_manager(max_cost=100.0)
        builder.save_config(str(agent_config_path), format='json')
        return builder

    async def register_agent(self, agent_builder: FlowAgentBuilder):
        agent_name = agent_builder.config.name

        if f'agent-instance-{agent_name}' in self.config:
            self.print(f"Agent '{agent_name}' instance already exists. Overwriting config and rebuilding on next get.")
            self.config.pop(f'agent-instance-{agent_name}', None)

        config_path = Path(f"{get_app().data_dir}/Agents/{agent_name}/agent.json")
        agent_builder.save_config(str(config_path), format='json')
        self.print(f"Saved FlowAgentBuilder config for '{agent_name}' to {config_path}")

        self.agent_data[agent_name] = agent_builder.config.model_dump()

        if agent_name not in self.config.get("agents-name-list", []):
            if "agents-name-list" not in self.config:
                self.config["agents-name-list"] = []
            self.config["agents-name-list"].append(agent_name)

        self.print(f"FlowAgent '{agent_name}' configuration registered. Will be built on first use.")
        row_agent_builder_sto[agent_name] = agent_builder

    async def get_agent(self, agent_name="Normal", model_override: str | None = None) -> FlowAgent:
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

        if agent_name in row_agent_builder_sto:
            builder_to_use = row_agent_builder_sto[agent_name]
            self.print(f"Using cached builder for {agent_name}")

        elif agent_name in self.agent_data:
            self.print(f"Loading configuration for FlowAgent: {agent_name}")
            try:
                config = AgentConfig(**self.agent_data[agent_name])
                builder_to_use = FlowAgentBuilder(config=config)
            except Exception as e:
                self.print(f"Error loading config for {agent_name}: {e}. Falling back to default.")

        if builder_to_use is None:
            self.print(f"get builder for FlowAgent: {agent_name}.")
            builder_to_use = self.get_agent_builder(agent_name)
            await self.register_agent(builder_to_use)

        builder_to_use._isaa_ref = self
        if model_override:
            builder_to_use.with_models(model_override, model_override)

        if builder_to_use.config.name != agent_name:
            builder_to_use.with_name(agent_name)

        self.print(
            f"Building FlowAgent: {agent_name} with models {builder_to_use.config.fast_llm_model} - {builder_to_use.config.complex_llm_model}")

        agent_instance: FlowAgent = await builder_to_use.build()

        # agent_instance.

        self.config[instance_key] = agent_instance
        if agent_name not in self.agent_data:
            self.agent_data[agent_name] = builder_to_use.config.model_dump()
        if agent_name not in self.config["agents-name-list"]:
            self.config["agents-name-list"].append(agent_name)

        self.print(f"Built and cached FlowAgent instance: {agent_name}")
        return agent_instance

    @export(api=True, version=version, request_as_kwarg=True, mod_name="isaa")
    async def mini_task_completion(
        self,
        mini_task: str | None = None,
        user_task: str | None = None,
        mode: Any = None,
        max_tokens_override: int | None = None,
        task_from="system",
        stream_function: Callable | None = None,
        message_history: list | None = None,
        agent_name="TaskCompletion",
        use_complex: bool = False,
        request: RequestData | None = None,
        form_data: dict | None = None,
        data: dict | None = None,
        **kwargs
    ):
        if request is not None or form_data is not None or data is not None:
            data_dict = (request.request.body if request else None) or form_data or data
            mini_task = mini_task or data_dict.get("mini_task")
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

        if mini_task is None:
            return None
        if agent_name is None:
            return None
        if mini_task == "test":
            return "test"

        self.print(f"Running mini task, volume {len(mini_task)}")

        agent = await self.get_agent(agent_name)

        effective_system_message = agent.amd.system_message
        if mode and hasattr(mode, 'system_msg') and mode.system_msg:
            effective_system_message = mode.system_msg

        messages = []
        if effective_system_message:
            messages.append({"role": "system", "content": effective_system_message})
        if message_history:
            messages.extend(message_history)

        current_prompt = mini_task
        if user_task:
            messages.append({"role": task_from, "content": mini_task})
            current_prompt = user_task

        messages.append({"role": "user", "content": current_prompt})

        if use_complex:
            llm_params = {"model": agent.amd.complex_llm_model, "messages": messages}
        else:
            llm_params = {
                "model": agent.amd.fast_llm_model if agent.amd.use_fast_response else agent.amd.complex_llm_model,
                "messages": messages
            }

        if max_tokens_override:
            llm_params['max_tokens'] = max_tokens_override
        else:
            llm_params['max_tokens'] = agent.amd.max_tokens

        if kwargs:
            llm_params.update(kwargs)

        if stream_function:
            llm_params['stream'] = True
            original_stream_cb = agent.stream_callback
            original_stream_val = agent.stream
            agent.stream_callback = stream_function
            agent.stream = True
            try:
                response_content = await agent.a_run_llm_completion(**llm_params)
            finally:
                agent.stream_callback = original_stream_cb
                agent.stream = original_stream_val
            return response_content

        llm_params['stream'] = False
        response_content = await agent.a_run_llm_completion(**llm_params)
        return response_content

    async def mini_task_completion_format(
        self,
        mini_task,
        format_schema: type[BaseModel],
        max_tokens_override: int | None = None,
        agent_name="TaskCompletion",
        task_from="system",
        mode_overload: Any = None,
        user_task: str | None = None,
        auto_context=False,
        **kwargs
    ):
        if mini_task is None:
            return None
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

        try:
            result_dict = await agent.a_format_class(
                pydantic_model=format_schema,
                prompt=current_prompt,
                message_context=message_context,
                auto_context=auto_context
            )
            if format_schema == bool:
                return result_dict.get("value", False) if isinstance(result_dict, dict) else False
            return result_dict
        except Exception as e:
            self.print(f"Error in mini_task_completion_format: {e}")
            return None

    @export(api=True, version=version, name="version")
    async def get_version(self, *a, **k):
        return self.version

    @export(api=True, version=version, request_as_kwarg=True, mod_name="isaa")
    async def format_class(
        self,
        format_schema: type[BaseModel] | None = None,
        task: str | None = None,
        agent_name="TaskCompletion",
        auto_context=False,
        request: RequestData | None = None,
        form_data: dict | None = None,
        data: dict | None = None,
        **kwargs
    ):
        if request is not None or form_data is not None or data is not None:
            data_dict = (request.request.body if request else None) or form_data or data
            format_schema = format_schema or data_dict.get("format_schema")
            task = task or data_dict.get("task")
            agent_name = data_dict.get("agent_name") or agent_name
            auto_context = auto_context or data_dict.get("auto_context")
            kwargs = kwargs or data_dict.get("kwargs")

        if format_schema is None or not task:
            return None

        agent = None
        if isinstance(agent_name, str):
            agent = await self.get_agent(agent_name)
        elif isinstance(agent_name, FlowAgent):
            agent = agent_name
        else:
            raise TypeError("agent_name must be str or FlowAgent instance")

        return await agent.a_format_class(format_schema, task, auto_context=auto_context)

    async def run_agent(
        self,
        name: str | FlowAgent,
        text: str,
        verbose: bool = False,
        session_id: str | None = 'default',
        progress_callback: Callable[[Any], None | Awaitable[None]] | None = None,
        **kwargs
    ):
        if text is None:
            return ""
        if name is None:
            return ""
        if text == "test":
            return ""

        agent_instance = None
        if isinstance(name, str):
            agent_instance = await self.get_agent(name)
        elif isinstance(name, FlowAgent):
            agent_instance = name
        else:
            return self.return_result().default_internal_error(f"Invalid agent identifier type: {type(name)}")

        self.print(f"Running agent {agent_instance.amd.name} for task: {text[:100]}...")
        save_p = None
        if progress_callback:
            save_p = agent_instance.progress_callback
            agent_instance.progress_callback = progress_callback

        if verbose:
            agent_instance.verbose = True

        response = await agent_instance.a_run(
            query=text,
            session_id=session_id,
            user_id=None,
            stream_callback=None
        )

        if save_p:
            agent_instance.progress_callback = save_p

        return response

    async def mas_text_summaries(self, text, min_length=36000, ref=None, max_tokens_override=None):
        len_text = len(text)
        if len_text < min_length:
            return text

        key = self.one_way_hash(text, 'summaries', 'isaa')
        value = self.mas_text_summaries_dict.get(key)
        if value is not None:
            return value

        from .extras.modes import SummarizationMode

        summary = await self.mini_task_completion(
            mini_task=f"Summarize this text, focusing on aspects related to '{ref if ref else 'key details'}'. The text is: {text}",
            mode=self.controller.rget(SummarizationMode),
            max_tokens_override=max_tokens_override,
            agent_name="self"
        )

        if summary is None or not isinstance(summary, str):
            summary = text[:min_length] + "... (summarization failed)"

        self.mas_text_summaries_dict.set(key, summary)
        return summary

    def get_memory(self, name: str | None = None) -> AISemanticMemory:
        logger_ = get_logger()
        if isinstance(self.agent_memory, str):
            logger_.info(Style.GREYBG("AISemanticMemory Initialized from path"))
            self.agent_memory = AISemanticMemory(base_path=self.agent_memory)

        cm = self.agent_memory
        if name is not None:
            mem_kb = cm.get(name)
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


# =============================================================================
# SHELL TOOL
# =============================================================================

import subprocess
import threading
import queue
import time
import uuid
import json
import os
import platform
import shutil
import base64
from typing import Dict, Optional, Any, Tuple

# =============================================================================
# GLOBALE SESSION-VERWALTUNG
# =============================================================================
_session_store: Dict[str, 'ShellSession'] = {}


def is_admin() -> bool:
    """Prüft, ob der aktuelle Prozess bereits Admin-Rechte hat."""
    try:
        if platform.system() == "Windows":
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        else:
            return os.getuid() == 0
    except:
        return False


# =============================================================================
# SHELL SESSION KLASSE
# =============================================================================

class ShellSession:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.system = platform.system()
        self.is_windows = self.system == "Windows"

        shell_exe, _ = detect_shell()

        # Environment mit UTF-8 Support
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        # PowerShell/CMD ohne zusätzliche Parameter starten
        start_args = [shell_exe]

        if "powershell" in shell_exe.lower() or "pwsh" in shell_exe.lower():
            # PowerShell: NoLogo, NoExit für persistente Session
            start_args.extend([
                "-NoLogo",
                "-NoExit",
                "-NoProfile",  # Schnellerer Start
                "-ExecutionPolicy", "Bypass"
            ])

        self.process = subprocess.Popen(
            start_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
            env=env,
            shell=False,
            # Windows-spezifisch: Verhindert Popup-Fenster
            creationflags=subprocess.CREATE_NO_WINDOW if self.is_windows else 0
        )

        self.out_queue = queue.Queue()
        self.err_queue = queue.Queue()
        self.active = True

        self._start_reader(self.process.stdout, self.out_queue)
        self._start_reader(self.process.stderr, self.err_queue)

    def _start_reader(self, pipe, q):
        """Startet async Reader für non-blocking I/O"""

        def reader():
            try:
                while True:
                    chunk = pipe.read(1)
                    if not chunk:
                        break
                    q.put(chunk)
            except Exception:
                pass

        threading.Thread(target=reader, daemon=True).start()

    def write(self, command: str):
        """
        Sendet Command 1:1 an Shell - KEINE Manipulation!
        """
        if not self.process.stdin:
            return

        try:
            cmd_str = command.strip()
            line_ending = "\r\n" if self.is_windows else "\n"

            self.process.stdin.write((cmd_str + line_ending).encode('utf-8'))
            self.process.stdin.flush()
        except (IOError, BrokenPipeError):
            self.active = False

    def read_output(self, timeout: float = 2.0) -> Dict[str, Any]:
        """
        Liest Output mit intelligenterem Timeout-Handling
        """
        stdout_acc = bytearray()
        stderr_acc = bytearray()

        start_time = time.time()
        last_data_time = time.time()

        # Initial wait für Command-Processing
        time.sleep(0.15)

        while True:
            got_data = False

            try:
                while not self.out_queue.empty():
                    stdout_acc.extend(self.out_queue.get_nowait())
                    got_data = True
                    last_data_time = time.time()

                while not self.err_queue.empty():
                    stderr_acc.extend(self.err_queue.get_nowait())
                    got_data = True
                    last_data_time = time.time()
            except queue.Empty:
                pass

            # Timeout wenn keine neuen Daten mehr kommen
            if (time.time() - last_data_time) > timeout:
                break

            # Absolutes Timeout
            if (time.time() - start_time) > (timeout * 3):
                break

            if self.process.poll() is not None:
                self.active = False
                break

            time.sleep(0.05)

        return {
            "stdout": self._safe_decode(stdout_acc),
            "stderr": self._safe_decode(stderr_acc),
            "is_alive": self.process.poll() is None
        }

    def _safe_decode(self, data: bytearray) -> str:
        """Multi-Codec Decoding mit Fallbacks"""
        for encoding in ['utf-8', 'cp1252', 'latin-1']:
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode('latin-1', errors='replace')

    def terminate(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.active = False


# =============================================================================
# TOOL FUNCTION - Verbesserte API
# =============================================================================

def shell_tool_function(
    command: Optional[str] = None,
    session_id: Optional[str] = None,
    user_input: Optional[str] = None,
    new_session: bool = False,
    timeout: float = 2.0
) -> str:
    r"""
    Shell-Tool mit Session-Support und verbessertem Error-Handling.

    Features:
    - Persistente Shell-Sessions
    - Pipes & Operators (|, ;, &&, ||)
    - Backslashes in Pfaden (C:\Users\...)
    - Variable Evaluation ($env:VAR, $PSVersionTable)
    - Interactive Input Support
    - Multi-line Commands
    - Cross-Platform (Windows/Linux/Mac)

    - special command: list-sessions, cleanup-sessions

    Args:
        command: Shell command to execute
        session_id: Continue existing session
        user_input: Send input to running process
        new_session: Force new session creation
        timeout: Output read timeout (default: 2.0s)

    Returns:
        JSON string with execution result
    """

    if command == "list-sessions":
        return list_sessions()
    if command == "cleanup-sessions":
        return cleanup_sessions()

    session = None
    msg_info = ""

    # Session Management
    if session_id and session_id in _session_store and not new_session:
        session = _session_store[session_id]
        msg_info = "Resumed session"
    else:
        session = ShellSession()
        _session_store[session.id] = session
        msg_info = "New session started"

    # Command/Input Execution
    input_to_send = user_input if user_input else command

    if input_to_send:
        session.write(input_to_send)

    # Output Collection
    wait_time = timeout if command else (timeout / 2)
    output = session.read_output(timeout=wait_time)

    # Status Detection
    status = "running" if output["is_alive"] else "finished"

    stdout_str = output["stdout"]
    if status == "running" and stdout_str and not stdout_str.endswith(("\n", ">")):
        status = "waiting_for_input"

    # Result Assembly
    result = {
        "success": True,
        "session_id": session.id,
        "stdout": stdout_str.strip(),
        "stderr": output["stderr"].strip(),
        "status": status,
        "info": msg_info,
        "system": session.system
    }

    # Auto-Cleanup für One-Shot Commands
    if not session_id and not new_session and status != "waiting_for_input":
        session.terminate()
        del _session_store[session.id]

    return json.dumps(result, ensure_ascii=False, indent=2)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_sessions() -> str:
    """Liste alle aktiven Sessions"""
    sessions = [
        {
            "session_id": sid,
            "system": session.system,
            "active": session.active,
            "pid": session.process.pid if session.process else None
        }
        for sid, session in _session_store.items()
    ]
    return json.dumps({"sessions": sessions}, indent=2)


def cleanup_sessions() -> str:
    """Beende alle Sessions"""
    count = len(_session_store)
    for session in _session_store.values():
        session.terminate()
    _session_store.clear()
    return json.dumps({"cleaned": count})


# =============================================================================
# TOOL FUNCTION (Interface)
# =============================================================================

# =============================================================================
# EXPORTS
# =============================================================================

@export(mod_name="isaa", name="listAllAgents", api=True, request_as_kwarg=True)
async def list_all_agents(self, request: RequestData | None = None):
    return self.config.get("agents-name-list", [])


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    async def test_isaa_tools():
        app_instance = get_app("isaa_test_app")
        isaa_tool_instance = Tools(app=app_instance)
        await isaa_tool_instance.init_isaa()

        # Test get_agent
        self_agent = await isaa_tool_instance.get_agent("self")
        print(f"Got agent: {self_agent.amd.name}")

        # Test chain creation
        chain = isaa_tool_instance.create_chain(self_agent)
        print(f"Created chain: {chain}")

        # Test save_agent
        success, manifest = await isaa_tool_instance.save_agent("self", "/tmp/test_agent.tar.gz")
        print(f"Save agent: {success}")
        if success:
            print(f"Manifest: {manifest}")


    asyncio.run(test_isaa_tools())
