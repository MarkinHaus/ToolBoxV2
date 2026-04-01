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
import sys
import tarfile
import tempfile
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel

from toolboxv2 import (
    FileHandler,
    MainTool,
    RequestData,
    Spinner,
    Style,
    get_app,
    get_logger,
)
from toolboxv2.mods.isaa.base.Agent.executors import register_code_exec_tools
from toolboxv2.mods.isaa.base.MemoryKnowledgeActor import MemoryKnowledgeActor
from toolboxv2.utils.extras.Style import print_prompt
from toolboxv2.utils.system import FileCache
from toolboxv2.utils.toolbox import stram_print
from pathlib import Path
from toolboxv2.mods.isaa.extras.jobs import JobScheduler, JobDefinition, TriggerConfig

from .base.Agent.builder import AgentConfig, FlowAgentBuilder

# Chain imports - native support
from .base.Agent.chain import (
    Chain,
)

# FlowAgent imports
from .base.Agent.flow_agent import FlowAgent
from .base.AgentUtils import (
    ControllerManager,
    detect_shell,
)
from .base.ai_semantic_memory import AISemanticMemory

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
Name = "isaa"
version = "0.3.0"  # Version bump for refactoring
export = get_app("isaa.Export").tb
pipeline_arr = [
    "question-answering",
    "summarization",
    "text-classification",
    "text-to-speech",
]

row_agent_builder_sto = {}


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


def _serialize_tool(
    func: Callable, name: str
) -> tuple[bytes | None, ToolSerializationInfo]:
    """
    Attempt to serialize a tool function.

    Returns:
        Tuple of (serialized_bytes or None, ToolSerializationInfo)
    """
    serializer = _get_serializer()

    info = ToolSerializationInfo(
        name=name,
        serializable=False,
        module_path=getattr(func, "__module__", None),
        function_name=getattr(func, "__name__", name),
    )

    if serializer is None:
        info.error_message = (
            "No serializer available. Install 'dill' or 'cloudpickle': pip install dill"
        )
        info.source_hint = (
            f"Recreate manually: from {info.module_path} import {info.function_name}"
        )
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
            info.source_hint = (
                f"Recreate the tool '{name}' manually after loading the agent."
            )

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
            "controller-init": False,
            "agents-name-list": [],
            "FASTMODEL": os.getenv("FASTMODEL", "ollama/llama3.1"),
            "AUDIOMODEL": os.getenv("AUDIOMODEL", "groq/whisper-large-v3-turbo"),
            "BLITZMODEL": os.getenv("BLITZMODEL", "ollama/llama3.1"),
            "COMPLEXMODEL": os.getenv("COMPLEXMODEL", "ollama/llama3.1"),
            "SUMMARYMODEL": os.getenv("SUMMARYMODEL", "ollama/llama3.1"),
            "IMAGEMODEL": os.getenv("IMAGEMODEL", "ollama/llama3.1"),
            "DEFAULTMODELEMBEDDING": os.getenv(
                "DEFAULTMODELEMBEDDING", "gemini/text-embedding-004"
            ),
        }
        self.per_data = {}
        self.agent_data: dict[str, dict] = {}
        self.keys = {"KEY": "key~~~~~~~", "Config": "config~~~~"}
        self.initstate = {}

        extra_path = ""
        if self.toolID:
            extra_path = f"/{self.toolID}"

        self.observation_term_mem_file = (
            f"{app.data_dir}/Memory{extra_path}/observationMemory/"
        )
        self.config["controller_file"] = f"{app.data_dir}{extra_path}/controller.json"
        self.mas_text_summaries_dict = FileCache(
            folder=f"{app.data_dir}/Memory{extra_path}/summaries/"
        )

        self.tools = {
            "name": "isaa",
            "Version": self.show_version,
            "mini_task_completion": self.mini_task_completion,
            "run_agent": self.run_agent,
            "save_to_mem": self.save_to_mem_sync,
            "get_agent": self.get_agent,
            "delete_agent": self.delete_agent,
            "format_class": self.format_class,
            "get_memory": self.get_memory,
            "save_all_memory_vis": self.save_all_memory_vis,
            "rget_mode": lambda mode: self.controller.rget(mode),
            # Chain helpers
            "create_chain": self.create_chain,
            "start_caht": lambda: __import__(
            "toolboxv2.mods.isaa.isaa_chat", fromlist=["main"]
        ).main(),
            "run_chain": self.run_chain,
            # Agent export/import
            "save_agent": self.save_agent,
            "load_agent": self.load_agent,

            "load_all_agents": self.load_all_agents,
            "export_agent_network": self.export_agent_network,
            "import_agent_network": self.import_agent_network,
        }

        self.working_directory = os.getenv("ISAA_WORKING_PATH", os.getcwd())
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
        self.job_scheduler = JobScheduler(
            jobs_file=Path(self.app.appdata) / "isaa" / "jobs.json",
            fire_callback=self._fire_job_callback,
        )

        self.file_handler = FileHandler(
            f"isaa{extra_path.replace('/', '-')}.config", app.id if app else __name__
        )
        MainTool.__init__(
            self,
            load=self.on_start,
            v=self.version,
            tool=self.tools,
            name=self.name,
            logs=None,
            color=self.color,
            on_exit=self.on_exit,
        )

        from toolboxv2.mods.isaa.extras.web_helper.web_search import web_search

        async def web_search_tool(query: str) -> str:
            res = web_search(query)
            return await self.mas_text_summaries(str(res), min_length=12000, ref=query)

        # advanced web search tool with dorks

        from toolboxv2.mods.isaa.extras.web_helper.web_agent import quick_search

        # top 5 dorking keys site, filetype, inurl, intitle, exclude

        async def advanced_web_search_tool(
            query: str, dork_kwargs: dict[str, str] = None
        ) -> str:
            """
            🔎 Web-Suche mit Google Dorks Support.

            Args:
                query: Suchbegriff
                dork_kwargs: Google Dork Parameter (site=, filetype=, etc.)

            Returns:
                Suchergebnisse mit Titel, URL, Snippet
            """
            dork_kwargs = dork_kwargs or {}
            results = await quick_search(query, **dork_kwargs)
            return str(results)

        self.web_search = advanced_web_search_tool  # web_search_tool
        self.shell_tool_function = shell_tool_function

        self.print(f"Start {self.spec}.isaa")
        with Spinner(message="Starting module", symbols="c"):
            self.file_handler.load_file_handler()
            config_fh = self.file_handler.get_file_handler(self.keys["Config"])
            if config_fh is not None:
                if isinstance(config_fh, str):
                    try:
                        config_fh = json.loads(config_fh)
                    except json.JSONDecodeError:
                        self.print(
                            f"Warning: Could not parse config from file handler: {config_fh[:100]}..."
                        )
                        config_fh = {}

                if isinstance(config_fh, dict):
                    loaded_config = config_fh
                    for key, value in self.config.items():
                        if key not in loaded_config:
                            loaded_config[key] = value
                    self.config = loaded_config

                    self.config.update(
                        {"FASTMODEL": os.getenv("FASTMODEL", self.config.get("FASTMODEL")),
                         "AUDIOMODEL": os.getenv("AUDIOMODEL", self.config.get("AUDIOMODEL")),
                         "BLITZMODEL": os.getenv("BLITZMODEL", self.config.get("BLITZMODEL")),
                         "COMPLEXMODEL": os.getenv("COMPLEXMODEL", self.config.get("COMPLEXMODEL")),
                         "SUMMARYMODEL": os.getenv("SUMMARYMODEL", self.config.get("SUMMARYMODEL")),
                         "IMAGEMODEL": os.getenv("IMAGEMODEL", self.config.get("IMAGEMODEL")),
                         "DEFAULTMODELEMBEDDING": os.getenv("DEFAULTMODELEMBEDDING",
                                                            self.config.get("DEFAULTMODELEMBEDDING"))}
                    )


            if self.spec == "app":
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

            Path(f"{get_app('isaa-initIsaa').data_dir}/Agents/").mkdir(
                parents=True, exist_ok=True
            )
            Path(f"{get_app('isaa-initIsaa').data_dir}/Memory/").mkdir(
                parents=True, exist_ok=True
            )

    async def _fire_job_callback(self, job: JobDefinition):
        """Callback wenn ein Job feuert."""
        if job.query == "__dream__":
            from toolboxv2.mods.isaa.base.Agent.dreamer import DreamConfig
            agent = await self.get_agent(job.agent_name)
            dream_cfg = DreamConfig()
            if job.trigger.extra and "dream_config" in job.trigger.extra:
                dream_cfg = DreamConfig(**job.trigger.extra["dream_config"])
            return await agent.a_dream(dream_cfg)

        agent = await self.get_agent(job.agent_name)
        return await agent.a_run(job.query, session_id=job.session_id)

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
            return Chain(comp) if hasattr(comp, "a_run") else Chain._create_chain([comp])

        # Build chain from components
        chain = Chain()
        chain.tasks = list(agents_or_components)
        return chain

    async def run_chain(
        self, chain: Chain | list, query: str, session_id: str = "default", **kwargs
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
        notes: str | None = None,
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
        if not path.endswith(".tar.gz"):
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
                    self.print(
                        f"No builder config found for {agent_name}. Creating default. to save"
                    )
                    builder_config = AgentConfig(name=agent_name).model_dump()

            # Prepare tool serialization
            serializable_tools = []
            non_serializable_tools = []
            tools_data = {}

            if include_tools and hasattr(agent, "tool_manager"):
                for tool_name, tool_info in agent.tool_manager.tools.items():
                    func = tool_info.get("function") or tool_info.get("func")
                    if func:
                        serialized, info = _serialize_tool(func, tool_name)
                        if serialized:
                            serializable_tools.append(tool_name)
                            tools_data[tool_name] = {
                                "data": serialized,
                                "description": tool_info.get("description", ""),
                                "category": tool_info.get("category", []),
                            }
                        else:
                            non_serializable_tools.append(info)

            # Prepare checkpoint
            checkpoint_data = None
            if include_checkpoint:
                try:
                    checkpoint_path = await agent.checkpoint_manager.save_current()
                    if checkpoint_path and Path(checkpoint_path).exists():
                        with open(checkpoint_path, "r") as f:
                            checkpoint_data = json.load(f)
                except Exception as e:
                    self.print(f"Warning: Could not save checkpoint: {e}")

            # Get bindings
            bindings = []
            if hasattr(agent, "bind_manager"):
                bindings = list(agent.bind_manager.bindings.keys())

            # Create manifest
            manifest = AgentExportManifest(
                export_date=datetime.now().isoformat(),
                agent_name=agent_name,
                agent_version=builder_config.get("version", "1.0.0"),
                has_checkpoint=checkpoint_data is not None,
                has_tools=len(tools_data) > 0,
                tool_count=len(serializable_tools) + len(non_serializable_tools),
                serializable_tools=serializable_tools,
                non_serializable_tools=non_serializable_tools,
                bindings=bindings,
                notes=notes,
            )

            # Create tar.gz archive
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            with tarfile.open(path, "w:gz") as tar:
                # Add manifest
                manifest_bytes = manifest.model_dump_json(indent=2).encode("utf-8")
                self._add_bytes_to_tar(tar, "manifest.json", manifest_bytes)

                # Add config
                config_bytes = json.dumps(builder_config, indent=2).encode("utf-8")
                self._add_bytes_to_tar(tar, "config.json", config_bytes)

                # Add checkpoint
                if checkpoint_data:
                    checkpoint_bytes = json.dumps(checkpoint_data, indent=2).encode(
                        "utf-8"
                    )
                    self._add_bytes_to_tar(tar, "checkpoint.json", checkpoint_bytes)

                # Add serialized tools
                if tools_data:
                    serializer = _get_serializer()
                    if serializer:
                        tools_bytes = serializer.dumps(tools_data)
                        self._add_bytes_to_tar(tar, "tools.dill", tools_bytes)

                # Add tools manifest (human readable)
                tools_manifest = {
                    "serializable": serializable_tools,
                    "non_serializable": [t.model_dump() for t in non_serializable_tools],
                }
                tools_manifest_bytes = json.dumps(tools_manifest, indent=2).encode(
                    "utf-8"
                )
                self._add_bytes_to_tar(tar, "tools_manifest.json", tools_manifest_bytes)

            self.print(f"Agent '{agent_name}' exported to {path}")
            self.print(
                f"  - Tools: {len(serializable_tools)} serialized, {len(non_serializable_tools)} manual"
            )

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
        register: bool = True,
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
            with tarfile.open(path, "r:gz") as tar:
                # Read manifest
                manifest_data = self._read_from_tar(tar, "manifest.json")
                if not manifest_data:
                    return None, None, ["Invalid archive: missing manifest.json"]
                manifest = AgentExportManifest(**json.loads(manifest_data))

                # Read config
                config_data = self._read_from_tar(tar, "config.json")
                if not config_data:
                    return None, None, ["Invalid archive: missing config.json"]
                config_dict = json.loads(config_data)

                # Override name if requested
                agent_name = override_name or manifest.agent_name
                config_dict["name"] = agent_name

                # Create builder and agent
                config = AgentConfig(**config_dict)
                builder = FlowAgentBuilder(config=config)
                builder._isaa_ref = self

                # Load tools
                if load_tools and manifest.has_tools:
                    tools_bytes = self._read_from_tar(tar, "tools.dill", binary=True)
                    if tools_bytes:
                        serializer = _get_serializer()
                        if serializer:
                            try:
                                tools_data = serializer.loads(tools_bytes)
                                for tool_name, tool_info in tools_data.items():
                                    func_data = tool_info.get("data")
                                    if func_data:
                                        func, error = _deserialize_tool(
                                            func_data, tool_name
                                        )
                                        if func:
                                            builder.add_tool(
                                                func,
                                                name=tool_name,
                                                description=tool_info.get(
                                                    "description", ""
                                                ),
                                                category=tool_info.get("category"),
                                            )
                                        else:
                                            warnings.append(
                                                f"Tool '{tool_name}': {error}"
                                            )
                            except Exception as e:
                                warnings.append(f"Failed to load tools: {str(e)}")
                        else:
                            warnings.append(
                                "No serializer available for tools. Install 'dill'."
                            )

                # Report non-serializable tools
                for tool_info in manifest.non_serializable_tools:
                    hint = (
                        tool_info.source_hint
                        or f"Recreate tool '{tool_info.name}' manually"
                    )
                    warnings.append(f"Tool '{tool_info.name}' not loaded: {hint}")

                # Build agent
                agent = await builder.build()

                # Load checkpoint
                if manifest.has_checkpoint:
                    checkpoint_data = self._read_from_tar(tar, "checkpoint.json")
                    if checkpoint_data:
                        try:
                            checkpoint = json.loads(checkpoint_data)
                            # Apply checkpoint to agent
                            if hasattr(agent, "checkpoint_manager"):
                                await agent.checkpoint_manager.restore_from_dict(
                                    checkpoint
                                )
                        except Exception as e:
                            warnings.append(f"Failed to restore checkpoint: {str(e)}")

                # Register agent
                if register:
                    self.agent_data[agent_name] = config_dict
                    self.config[f"agent-instance-{agent_name}"] = agent
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

    async def load_all_agents(self, do_not_cleanup: bool = False) -> dict:
        """
        Lädt alle verfügbaren Agenten aus dem 'Agents'-Verzeichnis von der Festplatte
        und löscht standardmäßig alle 'bench_'-Ordner im aktuellen Arbeitsverzeichnis.

        Args:
            do_not_cleanup (bool): Wenn True, wird der Cleanup der 'bench_'-Ordner übersprungen.

        Returns:
            dict: Statusbericht mit geladenen Agenten und gelöschten Ordnern.
        """
        import shutil
        from pathlib import Path

        report = {
            "loaded_agents": [],
            "deleted_bench_folders": [],
            "errors": []
        }

        # --- 1. Agenten laden ---
        async def _(_):
            try:
                # get_agent initialisiert, baut und registriert den Agenten sicher im System
                await self.get_agent(_)
                report["loaded_agents"].append(_)
            except Exception as e:
                report["errors"].append(f"Fehler beim Laden von Agent '{_}': {e}")

        try:
            agent_dir = Path(get_app().data_dir) / "Agents"
            tasks = []
            if agent_dir.exists():
                for d in agent_dir.iterdir():
                    if d.is_dir() and (d / "agent.json").exists():
                        tasks.append(_(d.name))
                await asyncio.gather(*tasks)
        except Exception as e:
            report["errors"].append(f"Kritischer Fehler beim Durchsuchen des Agenten-Verzeichnisses: {e}")

        # --- 2. Workspace Cleanup (bench_ Ordner) ---
        if not do_not_cleanup:
            try:
                base_dir = Path(self.working_directory).resolve()
                if base_dir.exists():
                    for item in base_dir.iterdir():
                        if item.is_dir() and item.name.startswith("bench_"):
                            try:
                                shutil.rmtree(item)
                                report["deleted_bench_folders"].append(item.name)
                            except Exception as e:
                                report["errors"].append(f"Fehler beim Löschen von Ordner '{item.name}': {e}")
            except Exception as e:
                report["errors"].append(f"Kritischer Fehler beim Cleanup: {e}")

        self.print(f"Loaded {len(report['loaded_agents'])} agents. "
                   f"Deleted {len(report['deleted_bench_folders'])} bench_ folders.")

        return report

    async def export_agent_network(
        self,
        agent_names: list[str],
        path: str,
        entry_agent: str | None = None,
        include_checkpoints: bool = True,
        include_tools: bool = True,
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

        if not path.endswith(".tar.gz"):
            path = f"{path}.tar.gz"

        try:
            # Collect binding information
            bindings = {}
            for name in agent_names:
                agent = await self.get_agent(name)
                if hasattr(agent, "bind_manager"):
                    bound_names = [
                        n for n in agent.bind_manager.bindings.keys() if n in agent_names
                    ]
                    if bound_names:
                        bindings[name] = bound_names

            # Create network manifest
            network_manifest = AgentNetworkManifest(
                export_date=datetime.now().isoformat(),
                agents=agent_names,
                bindings=bindings,
                entry_agent=entry_agent or (agent_names[0] if agent_names else None),
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
                        include_tools=include_tools,
                    )
                    if not success:
                        return False, f"Failed to export agent '{name}': {result}"

                # Create network archive
                Path(path).parent.mkdir(parents=True, exist_ok=True)

                with tarfile.open(path, "w:gz") as tar:
                    # Add network manifest
                    manifest_bytes = network_manifest.model_dump_json(indent=2).encode(
                        "utf-8"
                    )
                    self._add_bytes_to_tar(tar, "network_manifest.json", manifest_bytes)

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
        self, path: str, name_prefix: str = "", restore_bindings: bool = True
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
                with tarfile.open(path, "r:gz") as tar:
                    tar.extractall(tmpdir)

                # Read network manifest
                manifest_path = Path(tmpdir) / "network_manifest.json"
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
                        register=True,
                    )

                    if agent:
                        agents[new_name] = agent
                    all_warnings.extend(warnings)

                # Restore bindings
                if restore_bindings:
                    for source_name, bound_names in network_manifest.bindings.items():
                        source_full = (
                            f"{name_prefix}{source_name}" if name_prefix else source_name
                        )
                        if source_full not in agents:
                            continue

                        source_agent = agents[source_full]
                        for target_name in bound_names:
                            target_full = (
                                f"{name_prefix}{target_name}"
                                if name_prefix
                                else target_name
                            )
                            if target_full in agents:
                                try:
                                    await source_agent.bind(agents[target_full])
                                except Exception as e:
                                    all_warnings.append(
                                        f"Failed to bind {source_full} -> {target_full}: {e}"
                                    )

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

    def _read_from_tar(
        self, tar: tarfile.TarFile, name: str, binary: bool = False
    ) -> bytes | str | None:
        """Helper to read a file from tar archive"""
        try:
            member = tar.getmember(name)
            f = tar.extractfile(member)
            if f:
                data = f.read()
                return data if binary else data.decode("utf-8")
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

    async def init_from_augment(self, augment, agent_name: str = "self"):
        """Initialize from augmented data (legacy compatibility)"""
        if isinstance(agent_name, str):
            pass
        elif hasattr(agent_name, "config"):
            agent_name = agent_name.config.name
        else:
            raise ValueError(f"Invalid agent_name type: {type(agent_name)}")

        a_keys = augment.keys()

        if "Agents" in a_keys:
            agents_configs_dict = augment["Agents"]
            self.deserialize_all(agents_configs_dict)
            self.print("Agent configurations loaded.")

        if "tools" in a_keys:
            self.print(
                "Tool configurations noted - will be applied during agent building"
            )

    async def add_langchain_tools_to_builder(
        self, tools_config: dict, agent_builder: FlowAgentBuilder
    ):
        """Initialize tools from config (legacy compatibility)"""
        lc_tools_names = tools_config.get("lagChinTools", [])
        all_lc_tool_names = list(set(lc_tools_names))

        for tool_name in all_lc_tool_names:
            try:
                from langchain_community.agent_toolkits.load_tools import load_tools
                loaded_tools = load_tools([tool_name], llm=None)
                for lc_tool_instance in loaded_tools:
                    if hasattr(lc_tool_instance, "run") and callable(
                        lc_tool_instance.run
                    ):
                        agent_builder.add_tool(
                            lc_tool_instance.run,
                            name=lc_tool_instance.name,
                            description=lc_tool_instance.description,
                        )
                        self.print(
                            f"Added LangChain tool '{lc_tool_instance.name}' to builder."
                        )
            except Exception as e:
                self.print(f"Failed to load/add LangChain tool '{tool_name}': {e}")

    def serialize_all(self):
        """Returns a copy of agent_data"""
        return copy.deepcopy(self.agent_data)

    def deserialize_all(self, data: dict[str, dict]):
        """Load agent configurations"""
        self.agent_data.update(data)
        for agent_name in data:
            self.config.pop(f"agent-instance-{agent_name}", None)

    # =========================================================================
    # CORE METHODS (Preserved from original)
    # =========================================================================

    async def init_isaa(self, name="self", build=False, **kwargs):
        if self.initialized:
            self.print(f"Already initialized. Getting agent/builder: {name}")
            return self.get_agent_builder(name) if build else await self.get_agent(name)

        self.initialized = True
        sys.setrecursionlimit(1500)
        self.load_keys_from_env()

        with Spinner(message="Building Controller", symbols="c"):
            self.controller.init(self.config["controller_file"])
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
        self.config["VAULTS"] = os.getenv("VAULTS")

    async def on_exit(self):
        tasks = []
        for agent_name, agent_instance in self.config.items():
            if agent_name.startswith("agent-instance-") and agent_instance:
                if isinstance(agent_instance, FlowAgent):
                    tasks.append(agent_instance.close())

        threading.Thread(target=self.save_to_mem_sync, daemon=True).start()
        await asyncio.gather(*tasks)
        if self.config.get("controller-init"):
            self.controller.save(self.config["controller_file"])
        cleanup_sessions()
        clean_config = {}
        for key, value in self.config.items():
            if key.startswith("agent-instance-"):
                continue
            if key.startswith("LLM-model-"):
                continue
            clean_config[key] = value

        self.file_handler.add_to_save_file_handler(
            self.keys["Config"], json.dumps(clean_config)
        )
        self.file_handler.save_file_handler()

    def save_to_mem_sync(self):
        memory_instance = self.get_memory()
        if hasattr(memory_instance, "save_all_memories"):
            memory_instance.save_all_memories(f"{get_app().data_dir}/Memory/")
            self.print("Memory saving done")

    def load_to_mem_sync(self):
        memory_instance = self.get_memory()
        if hasattr(memory_instance, "load_all_memories"):
            memory_instance.load_all_memories(f"{get_app().data_dir}/Memory/")

    def get_agent_builder(
        self,
        name="self",
        extra_tools=None,
        add_base_tools=True,
        with_dangerous_shell=False,
    ) -> FlowAgentBuilder:
        if name == "None":
            name = "self"

        if extra_tools is None:
            extra_tools = []

        self.print(f"Creating FlowAgentBuilder: {name}")

        config = AgentConfig(
            name=name,
            fast_llm_model=self.config.get(
                f"{name.upper()}MODEL", self.config["FASTMODEL"]
            ),
            complex_llm_model=self.config.get(
                f"{name.upper()}MODEL", self.config["COMPLEXMODEL"]
            ),
            system_message= self.app.manifest.isaa.self_agent.system_message if self.app.manifest.isaa and self.app.manifest.isaa.self_agent.system_message else ("""# ISAA Agent System Prompt v2.1

---

## IDENTITY

You are an autonomous agent in the ISAA system.
Your job: complete the user's task using available tools, then call `final_answer`.
You are NOT a chatbot by default — **you act**.
But you can also think out loud, brainstorm, and have a real conversation when that is what is needed.
Read the MODE section below to know which applies.

---

## MODE — ACT vs. TALK

**ACT mode** (default when a concrete task is given):
- Tool calls, verification, `final_answer`
- Minimal prose, maximum action
- Output: results, file references, status

**TALK mode** (when the user is brainstorming, exploring, or in audio/voice):
- Conversational, warm, exploratory
- No forced tool use — respond like a knowledgeable partner.
- all information must be in context or from tool results!
- Ask clarifying questions, offer options, think out loud together
- Output: natural language, ideas, questions back

**How to detect TALK mode:**
- "what do you think about…", "let's brainstorm…", "help me understand…"
- Audio/voice context (short sentences, no markdown lists)
- No concrete deliverable requested
- User explicitly says "just talk to me" or "let's think through this"

**Switching:** You can switch mid-conversation. If the user shifts from brainstorming to "okay, build it", "deep dive" — switch to ACT immediately.

---

## YOUR TOOLS

### Always available (static)
| Tool | When to use |
|---|---|
| `think` | Before any irreversible action. Max 2 consecutive calls, then act. |
| `final_answer` | Exactly once: task done, blocked, or max iterations reached |
| `shift_focus` | > 8 iterations elapsed — archive progress, reset context |
| `list_tools` | Don't know what's available — always check before assuming |
| `load_tools` | Load tools by name after discovering them via list_tools |

### Filesystem (VFS) — always available
| Tool | What it does |
|---|---|
| `vfs_shell` | Unix-like shell: `ls cat head tail wc stat tree find grep touch write edit echo mkdir rm mv cp close exec` — returns `{success, stdout, stderr, returncode}` |
| `vfs_view` | Open/scroll a file in the context window. Use `scroll_to=` to jump to a pattern. Files opened here appear in EVERY following prompt. |
| `search_vfs` | Find files or code by name/content/regex. Returns file list + snippets. |

### Filesystem (real FS ↔ VFS)
| Tool | What it does |
|---|---|
| `fs_copy_to_vfs` | Copy real-filesystem file → VFS |
| `fs_copy_from_vfs` | Copy VFS file → real filesystem |
| `fs_copy_dir_from_vfs` | Recursively export a VFS directory → real filesystem |

### Mount
| Tool | What it does |
|---|---|
| `vfs_mount` | Mount local folder as lazy shadow into VFS |
| `vfs_unmount` | Unmount, optionally save dirty files to disk |
| `vfs_refresh_mount` | Re-scan mount for new/deleted files |
| `vfs_sync_all` | Sync all modified VFS files back to disk |

### Sharing
| Tool | What it does |
|---|---|
| `vfs_share_create` | Create shareable ID for a VFS directory |
| `vfs_share_list` | List directories shared with this agent |
| `vfs_share_mount` | Mount a shared directory from another agent into your VFS |

### LSP / Docker / History
| Tool | What it does |
|---|---|
| `vfs_diagnostics` | LSP errors/warnings/hints for a code file (async) |
| `docker_run` | Run shell command in Docker container (syncs VFS before/after) |
| `docker_start_app` | Start web app in Docker |
| `docker_stop_app` | Stop running web app |
| `docker_logs` | Get last N log lines from running app |
| `docker_status` | Docker container status (ports, running, etc.) |
| `history` | Last N messages from conversation history |
| `set_agent_situation` | Set current situation + intent for rule-based behavior |
| `check_permissions` | Check if an action is permitted under active rule set |

### Dynamic tools (loaded on demand)
Use `list_tools` to discover, `load_tools` to activate. These are session-specific and not pre-loaded.

---

## ABSOLUTE RULES

**ALWAYS:**
1. Call `think` before any destructive or irreversible action
2. Call `final_answer` exactly once — when done, blocked, or at max iterations
3. Reference code as `file_path:line_number`
4. Check tool results before the next step — not after
5. Use `search_vfs` before `vfs_shell write` if the target path is unknown

**NEVER:**
1. Call `final_answer` more than once
2. Repeat the same tool call with same arguments 3+ times
3. Invent file paths, tool names, or command syntax — verify first
4. Assume a tool is loaded — use `list_tools` when unsure
5. Write code comments unless explicitly requested
6. Continue past a hard block — escalate via `final_answer`

**On verification:** Not every tool returns content on success. A `vfs_shell` `mkdir` returns `{success: true, stdout: ""}` — that is valid. Verify the *right* thing via the *appropriate* tool:
- Write operations → verify with a subsequent `vfs_shell stat` or `vfs_shell cat`
- Async operations → verify with a status tool (`docker_status`, `vfs_diagnostics`)
- Side-effect tools → trust `success: true` unless behavior is observable otherwise

---

## CORE DECISION CHAINS (X → Y → Z)

### Chain 1 — Unknown task, unclear tools
```
user_query
  → think("what category of task is this?")
  → list_tools(category="relevant_keyword")
  → load_tools(["tool_a", "tool_b"])
  → execute tool
  → final_answer(answer=result, success=True)
```

### Chain 2 — File task (read / write / modify)
```
user asks about file
  → search_vfs(query="filename_or_symbol") to locate it
  → vfs_view(path="found_file") OR vfs_shell("cat found_file")
  → think("what changes are needed?")
  → vfs_shell("write target_file ...")
  → vfs_shell("stat target_file")   ← verify write succeeded
  → final_answer(answer="Done. target_file:line_number", success=True)
```

### Chain 3 — Multi-step task (> 3 actions)
```
complex_task
  → think("steps: 1. X  2. Y  3. Z")
  → execute step 1 → check result
  → execute step 2 → check result
  → [step fails] → think("alternative?")
               → retry once OR final_answer(success=False, explain)
  → execute step 3
  → final_answer(summary_of_all_steps, success=True)
```

### Chain 4 — Tool not working
```
tool_call fails
  → think("why? wrong args? missing dependency? wrong path?")
  → [fixable] → adjust args → retry ONCE
  → [still fails] → list_tools() → find alternative
  → [no alternative] → final_answer("Blocked: reason", success=False)
  → NEVER retry the same call 3+ times
```

### Chain 5 — Stuck or looping
```
loop_detected (same tool, same args, repeated)
  → STOP immediately
  → think("what am I missing? what would unblock this?")
  → [answerable] → different approach
  → [not answerable] → final_answer(explain_block, success=False)
```

### Chain 6 — Context getting large (> 8 iterations)
```
many_iterations_elapsed
  → shift_focus(
      summary_of_achievements="what was done, what files were created",
      next_objective="next concrete step"
    )
  → continue with clean context
```

### Chain 7 — Delegating to a sub-agent
```
task_needs_parallel_work OR isolated_subtask_identified
  → think("what exactly should the sub-agent do?")
  → spawn sub-agent with:
      task = "concrete, self-contained instruction"
      output_dir = "/workspace/{sub_agent_name}/"   ← sub-agent writes ONLY here
  → sub-agent executes independently, writes results to its output_dir
  → main agent reads results via:
      vfs_shell("cat /workspace/{sub_agent_name}/result.md") OR
      vfs_share_mount(share_id="...", mount_point="/shared/{sub_agent_name}")
  → integrate results → final_answer

SUB-AGENT CONSTRAINT: A sub-agent may ONLY write to its assigned output_dir.
It may read from anywhere (read-only outside its dir).
It must call final_answer with a path to its output, not inline all content.
```

---

## TASK EXECUTION PROTOCOL

### Phase 1 — UNDERSTAND (before first tool call)
- What exactly is being asked?
- Do I have the right tools loaded?
- What is the success condition?
- What could go wrong?

Use `think`. Do not skip on complex tasks.

### Phase 2 — EXECUTE
- One objective at a time
- Check tool result before next step
- Fail → think → adapt → retry once → escalate

### Phase 3 — VERIFY
- Does the output match what was asked?
- File exists? Code runs? Data looks right?
- If no: back to Phase 2 for that step only

### Phase 4 — REPORT (`final_answer`)
- What was accomplished
- File references: `path:line_number`
- If partial: what remains and why

---

## SKEPTICAL REASONING

```
"I think the code does X"        → WRONG. Read the file first.
"This should work"               → WRONG. Test it.
"The tool probably exists"       → WRONG. Call list_tools.
"It returned nothing = failure"  → WRONG. Check the tool's contract.
Internal confidence ≠ correctness. Ground in evidence.
```

When uncertain:
```
uncertain_about_X
  → think("minimum needed to verify X?")
  → cheapest verification (read before write, stat before cat)
  → proceed only after confirmation
```

---

## COMMON FAILURE MODES

| Pattern | What it looks like | Fix |
|---|---|---|
| **Loop** | Same tool, same args, 3× | `think` → alternative → `final_answer` if blocked |
| **Hallucinated path** | Writing to file never read or located | `search_vfs` first, always |
| **Tool assumption** | Calling tool never loaded | `list_tools` → `load_tools` first |
| **Silent failure** | Tool errors, agent continues anyway | Always check `success` field before next step |
| **Over-planning** | 3+ consecutive `think` calls, no action | Act after 2 `think` calls max |
| **Missing verify** | Writes file, never confirms | Follow write with `vfs_shell stat` or `vfs_shell cat` |
| **Wrong final_answer timing** | Called before task complete | Only when: done, blocked, or max iterations |
| **Sub-agent scope leak** | Sub-agent writes outside its output_dir | Enforce output_dir constraint at spawn time |
| **TALK mode rigidity** | Using markdown lists in audio/voice context | Detect mode, switch to natural sentences |

---

## OUTPUT FORMAT

**In ACT mode:**
- Tool calls: concise args, no explanation text in args
- After tool results: `think` to interpret if needed, then next action
- `final_answer`:
  ```
  ✅ Done: [one-line summary]
  📁 [file_path:line_number if relevant]
  ⚠️ Remaining: [if anything unfinished, and why]
  ```
- Max 5 sentences unless detail was requested

**In TALK mode:**
- Natural prose, no forced structure
- No markdown lists unless user is reading on screen
- Short sentences in audio context
- End with a question or an offer: "Want me to investigate that?" / "Should we go deeper on X?"""),
            temperature=0.7,
            max_tokens_output=2048,
            max_tokens_input=32768,
            use_fast_response=True,
            max_parallel_tasks=3,
            verbose_logging=False,
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

        async def memory_recall(query: str, search_type: str = "auto") -> str:
            """
            Ruft persistentes Wissen, Architektur-Dokus oder Projekt-Kontext ab.

            Args:
                query: Wonach gesucht wird (z.B. "Wie funktioniert das Auth-System?")
                search_type: "auto" (Vektor+BM25), "concept" (Sucht nach Konzept-Schlagwort) oder "relations" (Graph-Verbindungen für eine Entity)
            """
            mem_instance = self.get_memory()
            # MKA initialisieren (nutzt den default space des Agenten)
            mka = MemoryKnowledgeActor(memory=mem_instance, space_name=name)

            if search_type == "concept":
                return await mka.search_by_concept(query, k=5)
            elif search_type == "relations":
                return await mka.get_related_entities(query, depth=2)
            else:
                return await mka.search(query, k=4, min_similarity=0.4)

        async def memory_save(content: str, concepts: list[str] = None, entity_name: str = None) -> str:
            """
            Speichert WICHTIGE Architektur-Entscheidungen, Regeln oder Zusammenfassungen dauerhaft.

            Args:
                content: Der Text/Fakt, der gespeichert werden soll.
                concepts: Liste von Schlagwörtern (z.B. ["auth", "database", "api"]).
                entity_name: Optional. Wenn es eine Code-Komponente ist, benenne sie hier (z.B. "AuthService"), um den Graphen aufzubauen.
            """
            mem_instance = self.get_memory()
            mka = MemoryKnowledgeActor(memory=mem_instance, space_name=name)

            # Punkt hinzufügen
            res = await mka.add_data_point(text=content, content_type="fact", concepts=concepts)

            # Wenn es eine Entity ist, direkt in den Graphen einhängen
            if entity_name:
                await mka.add_entity(entity_id=entity_name.lower(), entity_type="component", name=entity_name)
                res += f"\nEntity '{entity_name}' zum Architektur-Graphen hinzugefügt."

            return res

        async def memory_list_spaces() -> str:
            """
            Listet alle verfügbaren Memory-Spaces (Wissensbereiche) auf.
            Nutze dies, um zu sehen, welche Kategorien oder Projekte bereits im Gedächtnis existieren.
            """
            mem_instance = self.get_memory()
            spaces = list(mem_instance.memories.keys())
            if not spaces:
                return "Keine Memory-Spaces gefunden (Gedächtnis ist leer)."
            return "Verfügbare Memory-Spaces:\n" + "\n".join([f"- {s}" for s in sorted(spaces)])

        # --- UPDATE: memory_analyse mit target_space Parameter ---
        async def memory_analyse(topic: str, depth: int = 5, target_space: str = None) -> str:
            """
            Startet eine tiefe, autonome Analyse im Langzeitgedächtnis zu einem Thema.

            Args:
                topic: Das Thema oder die Frage, die analysiert werden soll.
                depth: Maximale Anzahl der Analyseschritte (Standard: 5).
                target_space: Optional. Name des spezifischen Memory-Spaces (siehe memory_list_spaces).
                              Wenn leer, wird der eigene Agent-Space genutzt.
            """
            mem_instance = self.get_memory()

            # Wähle den Space: Entweder explizit angegeben oder der Standard-Space des Agenten
            space_to_use = target_space if target_space else name

            # Prüfen ob Space existiert, um Fehler zu vermeiden
            if space_to_use not in mem_instance.memories:
                # Fallback auf Default oder Fehler, falls explizit gewünscht
                if target_space:
                    return f"Fehler: Space '{target_space}' nicht gefunden. Verfügbar: {list(mem_instance.memories.keys())}"
                # Falls Default-Space noch nicht existiert, wird er vom MKA erstellt, das ist OK.

            # MKA initialisieren
            mka = MemoryKnowledgeActor(memory=mem_instance, space_name=space_to_use)

            # Startet den autonomen Loop
            history = await mka.start_analysis_loop(user_task=topic, max_iterations=depth, agent_name="self")

            final_result = None
            steps_taken = []

            for msg in history:
                if msg.get("role") == "tool":
                    content = msg.get("content", {})
                    if isinstance(content, dict):
                        if content.get("tool") == "final_analysis":
                            final_result = content.get("result")
                        elif "tool" in content:
                            steps_taken.append(content["tool"])

            if final_result:
                return f"Analyse in Space '{space_to_use}' erfolgreich ({len(steps_taken)} Schritte):\n{final_result}"
            else:
                return f"Analyse in Space '{space_to_use}' ohne eindeutiges Ergebnis beendet."

        if add_base_tools:
            builder.add_tool(memory_recall, "memory_recall", "Wissen, Architektur und Kontext abrufen (Suchen).",
                             category=["memory", "read"])

            builder.add_tool(memory_save, "memory_save", "Wichtige Fakten und Architektur-Details dauerhaft speichern.",
                             category=["memory", "write"])

            builder.add_tool(memory_analyse, "memory_analyse", "Tiefe Analyse von Zusammenhängen im Memory (Verbindet Fakten). Optional in spezifischem Space.", category=["memory", "read", "deep"])

            builder.add_tool(memory_list_spaces, "memory_list_spaces",
                             "Listet alle verfügbaren Wissens-Kategorien (Spaces) auf.",
                             category=["memory", "read", "meta"])
            builder.add_tool(
                self.web_search, "searchWeb", "Search the web for information"
            )

            if with_dangerous_shell:
                builder.add_tool(
                    self.shell_tool_function,
                    "shell",
                    f"Run shell command in {detect_shell()}",
                    category=["local", "shell", "commands"],
                    flags={"shell": True, "detect_shell": True, "dangerous":True}
                )

        builder.with_budget_manager(max_cost=100.0)
        builder.save_config(str(agent_config_path), format="json")
        return builder

    @staticmethod
    def add_code_executor_to_agent_sync(agent: FlowAgent):
        register_code_exec_tools(agent=agent)

    async def add_code_executor_to_agent(self, agent_or_name: FlowAgent | str):
        agent = agent_or_name
        if isinstance(agent_or_name, str):
            agent = await self.get_agent(agent_or_name)
        self.add_code_executor_to_agent_sync(agent=agent)

    async def register_agent(self, agent_builder: FlowAgentBuilder):
        agent_name = agent_builder.config.name

        if f"agent-instance-{agent_name}" in self.config:
            self.print(
                f"Agent '{agent_name}' instance already exists. Overwriting config and rebuilding on next get."
            )
            self.config.pop(f"agent-instance-{agent_name}", None)

        config_path = Path(f"{get_app().data_dir}/Agents/{agent_name}/agent.json")
        agent_builder.save_config(str(config_path), format="json")
        self.print(f"Saved FlowAgentBuilder config for '{agent_name}' to {config_path}")

        self.agent_data[agent_name] = agent_builder.config.model_dump()

        if agent_name not in self.config.get("agents-name-list", []):
            if "agents-name-list" not in self.config:
                self.config["agents-name-list"] = []
            self.config["agents-name-list"].append(agent_name)

        self.print(
            f"FlowAgent '{agent_name}' configuration registered. Will be built on first use."
        )
        row_agent_builder_sto[agent_name] = agent_builder

    async def get_agent(
        self, agent_name="Normal", model_override: str | None = None
    ) -> FlowAgent:
        if "agents-name-list" not in self.config:
            self.config["agents-name-list"] = []

        instance_key = f"agent-instance-{agent_name}"
        if instance_key in self.config:
            agent_instance = self.config[instance_key]
            if model_override and agent_instance.amd.fast_llm_model != model_override:
                self.print(
                    f"Model override for {agent_name}: {model_override}. Rebuilding."
                )
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
                self.print(
                    f"Error loading config for {agent_name}: {e}. Falling back to default."
                )

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
            f"Building FlowAgent: {agent_name} with models {builder_to_use.config.fast_llm_model} - {builder_to_use.config.complex_llm_model}"
        )

        agent_instance: FlowAgent = await builder_to_use.build()

        # agent_instance.

        self.config[instance_key] = agent_instance
        if agent_name not in self.agent_data:
            self.agent_data[agent_name] = builder_to_use.config.model_dump()
        if agent_name not in self.config["agents-name-list"]:
            self.config["agents-name-list"].append(agent_name)

        self.print(f"Built and cached FlowAgent instance: {agent_name}")
        return agent_instance

    async def delete_agent(self, agent_name: str) -> bool:
        """
        Löscht einen Agenten vollständig aus dem System (RAM & Disk).

        Aktionen:
        1. Stoppt laufende Instanz (close).
        2. Entfernt aus internen Registern (Registry, Builder-Cache).
        3. LÖSCHT den Konfigurations-Ordner (data/Agents/{name}).
        4. LÖSCHT den Memory-Ordner (data/Memory/{name}).

        Args:
            agent_name: Name des zu löschenden Agenten

        Returns:
            bool: True wenn erfolgreich ausgeführt.
        """
        if not agent_name:
            return False

        self.print(f"🗑️ START: Deleting agent '{agent_name}' completely...")

        # Zugriff auf App-Verzeichnis
        try:
            app_data_dir = get_app().data_dir
        except Exception:
            # Fallback falls get_app() fehlschlägt (z.B. Testumgebung)
            app_data_dir = self.working_directory

        # --- SCHRITT 1: Laufende Instanz stoppen (Memory Leak Prevention) ---
        instance_key = f"agent-instance-{agent_name}"
        if instance_key in self.config:
            agent_instance = self.config.pop(instance_key)
            if hasattr(agent_instance, "close"):
                try:
                    await agent_instance.close()
                    self.print(f"   - Instance stopped.")
                except Exception as e:
                    self.print(f"   - Warning: Error closing agent instance: {e}")

        # --- SCHRITT 2: Interne Register bereinigen ---
        # Builder Config entfernen
        if agent_name in self.agent_data:
            del self.agent_data[agent_name]

        # Globalen Builder Cache bereinigen
        if agent_name in row_agent_builder_sto:
            del row_agent_builder_sto[agent_name]

        # Aus der Namensliste entfernen
        if "agents-name-list" in self.config:
            if agent_name in self.config["agents-name-list"]:
                try:
                    self.config["agents-name-list"].remove(agent_name)
                    self.print(f"   - Removed from registry list.")
                except ValueError:
                    pass

        # --- SCHRITT 3: Konfigurations-Dateien löschen ---
        # Pfad: .../data/Agents/{agent_name}/
        agent_config_path = Path(f"{app_data_dir}/Agents/{agent_name}")
        if agent_config_path.exists():
            try:
                import shutil
                shutil.rmtree(agent_config_path)
                self.print(f"   - Deleted config files at: {agent_config_path}")
            except Exception as e:
                self.print(f"   - CRITICAL: Could not delete config files: {e}")

        # --- SCHRITT 4: Memory / Gedächtnis löschen ---
        # A) Aus dem RAM-Objekt entfernen
        try:
            mem_instance = self.get_memory()
            if hasattr(mem_instance, "memories") and agent_name in mem_instance.memories:
                del mem_instance.memories[agent_name]
        except Exception as e:
            self.print(f"   - Warning accessing memory object: {e}")

        # B) Von der Festplatte löschen
        # Pfad: .../data/Memory/{agent_name}/
        agent_memory_path = Path(f"{app_data_dir}/Memory/{agent_name}")
        if agent_memory_path.exists():
            try:
                import shutil
                shutil.rmtree(agent_memory_path)
                self.print(f"   - Deleted memory files at: {agent_memory_path}")
            except Exception as e:
                self.print(f"   - CRITICAL: Could not delete memory files: {e}")
        else:
            self.print(f"   - No persistent memory found to delete.")

        self.print(f"✅ Agent '{agent_name}' successfully deleted.")
        return True

    @export(api=True, version=version, request_as_kwarg=True, mod_name="isaa")
    async def mini_task_completion(
        self,
        mini_task: str | None = None,
        user_task: str | None = None,
        mode: Any = None,
        max_tokens_override: int | None = None,
        task_from="system",
        stream: bool = False,
        message_history: list | None = None,
        agent_name="TaskCompletion",
        use_complex: bool = False,
        request: RequestData | None = None,
        form_data: dict | None = None,
        data: dict | None = None,
        **kwargs,
    ):
        if request is not None or form_data is not None or data is not None:
            data_dict = (request.request.body if request else None) or form_data or data
            mini_task = mini_task or data_dict.get("mini_task")
            user_task = user_task or data_dict.get("user_task")
            print(data_dict)
            mode = None# mode or data_dict.get("mode")
            max_tokens_override = max_tokens_override or data_dict.get(
                "max_tokens_override"
            )
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
        if mode and hasattr(mode, "system_msg") and mode.system_msg:
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
                "model": agent.amd.fast_llm_model
                if agent.amd.use_fast_response
                else agent.amd.complex_llm_model,
                "messages": messages,
            }

        if max_tokens_override:
            llm_params["max_tokens"] = max_tokens_override
        else:
            llm_params["max_tokens"] = agent.amd.max_tokens

        if kwargs:
            llm_params.update(kwargs)

        llm_params["stream"] = stream
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
        **kwargs,
    ):
        if mini_task is None:
            return None
        self.print(f"Running formatted mini task, {mini_task[15:]}... volume {len(mini_task)} {user_task[34:]}...")

        agent = await self.get_agent(agent_name)

        effective_system_message = None
        if (
            mode_overload
            and hasattr(mode_overload, "system_msg")
            and mode_overload.system_msg
        ):
            effective_system_message = mode_overload.system_msg

        message_context = []
        if effective_system_message:
            message_context.append(
                {"role": "system", "content": effective_system_message}
            )

        current_prompt = mini_task
        if user_task:
            message_context.append({"role": task_from, "content": mini_task})
            current_prompt = user_task
        try:
            result_dict = await agent.a_format_class(
                pydantic_model=format_schema,
                prompt=current_prompt,
                message_context=message_context,
                auto_context=auto_context,
            )
            if format_schema == bool:
                return (
                    result_dict.get("value", False)
                    if isinstance(result_dict, dict)
                    else False
                )
            return result_dict
        except Exception as e:
            import traceback
            traceback.print_exc()
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
        **kwargs,
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
        session_id: str | None = "default",
        progress_callback: Callable[[Any], None | Awaitable[None]] | None = None,
        **kwargs,
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
            return self.return_result().default_internal_error(
                f"Invalid agent identifier type: {type(name)}"
            )

        self.print(f"Running agent {agent_instance.amd.name} for task: {text[:100]}...")
        save_p = None
        if progress_callback:
            save_p = agent_instance.progress_callback
            agent_instance.progress_callback = progress_callback

        if verbose:
            agent_instance.verbose = True

        response = await agent_instance.a_run(
            query=text, session_id=session_id, user_id=None, stream_callback=None
        )

        if save_p:
            agent_instance.progress_callback = save_p

        return response

    async def mas_text_summaries(
        self, text, min_length=36000, ref=None, max_tokens_override=None
    ):
        len_text = len(text)
        if len_text < min_length:
            return text

        key = self.one_way_hash(text, "summaries", "isaa")
        value = self.mas_text_summaries_dict.get(key)
        if value is not None:
            return value

        from .extras.modes import SummarizationMode

        summary = await self.mini_task_completion(
            mini_task=f"Summarize this text, focusing on aspects related to '{ref if ref else 'key details'}'. The text is: {text}",
            mode=self.controller.rget(SummarizationMode),
            max_tokens_override=max_tokens_override,
            agent_name="self",
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
            self.print(
                f"Saving to {name}.html with {len(kb.concept_extractor.concept_graph.concepts)} concepts"
            )
            await kb.vis(output_file=f"{dir_path}/{name}.html")
        return dir_path


# =============================================================================
# SHELL TOOL
# =============================================================================

import json
import os
import platform
import queue
import subprocess
import threading
import time
import uuid
from typing import Any, Dict, Optional

# =============================================================================
# GLOBALE SESSION-VERWALTUNG
# =============================================================================
_session_store: Dict[str, "ShellSession"] = {}


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
            start_args.extend(
                [
                    "-NoLogo",
                    "-NoExit",
                    "-NoProfile",  # Schnellerer Start
                    "-ExecutionPolicy",
                    "Bypass",
                ]
            )

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
            creationflags=subprocess.CREATE_NO_WINDOW if self.is_windows else 0,
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

            self.process.stdin.write((cmd_str + line_ending).encode("utf-8"))
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
            "is_alive": self.process.poll() is None,
        }

    def _safe_decode(self, data: bytearray) -> str:
        """Multi-Codec Decoding mit Fallbacks"""
        for encoding in ["utf-8", "cp1252", "latin-1"]:
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode("latin-1", errors="replace")

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
    timeout: float = 2.0,
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
        "system": session.system,
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
            "pid": session.process.pid if session.process else None,
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
    res = self.config.get("agents-name-list", [])
    if not res:
        await self.load_all_agents()
        res = self.config.get("agents-name-list", [])
    return res


# =============================================================================
# JOB MANAGEMENT + CLI/VIEWER LAUNCH  ─  ToolBox-facing API
# =============================================================================

import os
import subprocess
import sys
import threading
from pathlib import Path
import json as _json


# ── helper: live state enrichment ────────────────────────────────────────────

def _enrich_with_live(job_dict: dict, live_entry) -> dict:
    """Merge a JobLiveEntry into a job dict for API output."""
    if live_entry is None:
        return job_dict
    job_dict["is_running"]    = live_entry.status == "running"
    job_dict["iteration"]     = live_entry.iteration if live_entry.status == "running" else None
    job_dict["context_pct"]   = round(live_entry.context_pct(), 1) if live_entry.status == "running" else None
    job_dict["last_thought"]  = live_entry.last_thought[-400:] if live_entry.last_thought else None
    job_dict["tool_calls"]    = live_entry.tool_calls[-6:] if live_entry.tool_calls else []
    return job_dict


def _next_fire_label(job) -> str | None:
    """Human-readable 'next fire' string for a JobDefinition."""
    from datetime import datetime, timezone
    tt  = job.trigger.trigger_type
    now = datetime.now(timezone.utc)

    if tt == "on_time" and job.trigger.at_datetime:
        try:
            target = datetime.fromisoformat(job.trigger.at_datetime)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            secs = int((target - now).total_seconds())
            if secs <= 0:
                return "overdue"
            return f"in {secs // 60}m" if secs < 3600 else f"in {secs // 3600}h"
        except Exception:
            return None

    if tt == "on_interval" and job.trigger.interval_seconds:
        if job.last_run_at:
            try:
                last = datetime.fromisoformat(job.last_run_at)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                rem = job.trigger.interval_seconds - int((now - last).total_seconds())
                return "now" if rem <= 0 else f"in {rem}s"
            except Exception:
                pass
        return f"every {job.trigger.interval_seconds}s"

    if tt == "on_cron" and job.trigger.cron_expression:
        return job.trigger.cron_expression

    return tt.replace("on_", "")


# ── 1. List all jobs ──────────────────────────────────────────────────────────

@export(mod_name="isaa", name="jobList", api=True, request_as_kwarg=True)
async def job_list(self, request: RequestData | None = None) -> list[dict]:
    """Return all jobs enriched with live state."""
    from toolboxv2.mods.isaa.extras.jobs.job_live_state import JobLiveStateReader

    live_file = self.job_scheduler.jobs_file.with_suffix(".live.json")
    reader    = JobLiveStateReader(live_file)
    live      = reader.read()

    result = []
    for job in self.job_scheduler.list_jobs():
        d = job.to_dict()
        d["next_fire"]  = _next_fire_label(job)
        d["is_running"] = job.job_id in self.job_scheduler._firing
        d = _enrich_with_live(d, live.get(job.job_id))
        result.append(d)

    return result


# ── 2. Add a job ─────────────────────────────────────────────────────────────

@export(mod_name="isaa", name="jobAdd", api=True, request_as_kwarg=True)
async def job_add(
    self,
    name: str | None = None,
    agent_name: str | None = None,
    query: str | None = None,
    trigger_type: str = "on_interval",
    interval_seconds: int | None = None,
    at_datetime: str | None = None,
    cron_expression: str | None = None,
    session_id: str = "default",
    timeout_seconds: int = 300,
    request: RequestData | None = None,
) -> dict:
    """Create and register a new persistent job."""
    # Accept body from HTTP POST as well
    if request is not None:
        body = request.request.body or {}
        name             = name             or body.get("name")
        agent_name       = agent_name       or body.get("agent_name")
        query            = query            or body.get("query")
        trigger_type     = body.get("trigger_type", trigger_type)
        interval_seconds = interval_seconds or body.get("interval_seconds")
        at_datetime      = at_datetime      or body.get("at_datetime")
        cron_expression  = cron_expression  or body.get("cron_expression")
        session_id       = body.get("session_id", session_id)
        timeout_seconds  = body.get("timeout_seconds", timeout_seconds)

    if not name or not agent_name or not query:
        return {"ok": False, "message": "name, agent_name and query are required", "job_id": None}

    trigger = TriggerConfig(
        trigger_type=trigger_type,
        interval_seconds=interval_seconds,
        at_datetime=at_datetime,
        cron_expression=cron_expression,
    )
    job = JobDefinition(
        job_id=JobDefinition.generate_id(),
        name=name,
        agent_name=agent_name,
        query=query,
        trigger=trigger,
        session_id=session_id,
        timeout_seconds=timeout_seconds,
    )
    job_id = self.job_scheduler.add_job(job)

    # Auto-install OS autowake when first persistent job is created
    if self.job_scheduler.has_persistent_jobs():
        try:
            from toolboxv2.mods.isaa.extras.jobs.os_scheduler import (
                autowake_status, install_autowake,
            )
            if "Not installed" in autowake_status():
                install_autowake(self.job_scheduler.jobs_file)
        except Exception:
            pass

    return {"ok": True, "message": f"Job '{name}' created", "job_id": job_id}


# ── 3. Remove a job ───────────────────────────────────────────────────────────

@export(mod_name="isaa", name="jobRemove", api=True, request_as_kwarg=True)
async def job_remove(
    self,
    job_id: str | None = None,
    request: RequestData | None = None,
) -> dict:
    if request and not job_id:
        job_id = (request.request.body or {}).get("job_id")
    if not job_id:
        return {"ok": False, "message": "job_id required", "job_id": None}

    ok = self.job_scheduler.remove_job(job_id)
    return {
        "ok": ok,
        "message": "removed" if ok else "job not found",
        "job_id": job_id,
    }


# ── 4. Pause / Resume ─────────────────────────────────────────────────────────

@export(mod_name="isaa", name="jobPause", api=True, request_as_kwarg=True)
async def job_pause(
    self,
    job_id: str | None = None,
    request: RequestData | None = None,
) -> dict:
    if request and not job_id:
        job_id = (request.request.body or {}).get("job_id")
    if not job_id:
        return {"ok": False, "message": "job_id required", "job_id": None}
    ok = self.job_scheduler.pause_job(job_id)
    return {"ok": ok, "message": "paused" if ok else "not found / already paused", "job_id": job_id}


@export(mod_name="isaa", name="jobResume", api=True, request_as_kwarg=True)
async def job_resume(
    self,
    job_id: str | None = None,
    request: RequestData | None = None,
) -> dict:
    if request and not job_id:
        job_id = (request.request.body or {}).get("job_id")
    if not job_id:
        return {"ok": False, "message": "job_id required", "job_id": None}
    ok = self.job_scheduler.resume_job(job_id)
    return {"ok": ok, "message": "resumed" if ok else "not found / not paused", "job_id": job_id}


# ── 5. Manual fire ────────────────────────────────────────────────────────────

@export(mod_name="isaa", name="jobFire", api=True, request_as_kwarg=True)
async def job_fire(
    self,
    job_id: str | None = None,
    request: RequestData | None = None,
) -> dict:
    if request and not job_id:
        job_id = (request.request.body or {}).get("job_id")
    if not job_id:
        return {"ok": False, "message": "job_id required", "job_id": None}

    job = self.job_scheduler.get_job(job_id)
    if not job:
        return {"ok": False, "message": "job not found", "job_id": job_id}

    asyncio.ensure_future(self.job_scheduler._fire_job(job))
    return {"ok": True, "message": f"Job '{job.name}' fired", "job_id": job_id}


# ── 6. Job detail (full live snapshot) ───────────────────────────────────────

@export(mod_name="isaa", name="jobDetail", api=True, request_as_kwarg=True)
async def job_detail(
    self,
    job_id: str | None = None,
    request: RequestData | None = None,
) -> dict:
    if request and not job_id:
        job_id = (request.request.query or {}).get("job_id") or (request.request.body or {}).get("job_id")
    if not job_id:
        return {"ok": False, "message": "job_id required"}

    job = self.job_scheduler.get_job(job_id)
    if not job:
        return {"ok": False, "message": "job not found"}

    from toolboxv2.mods.isaa.extras.jobs.job_live_state import JobLiveStateReader
    live_file = self.job_scheduler.jobs_file.with_suffix(".live.json")
    live = JobLiveStateReader(live_file).read()

    d = job.to_dict()
    d["next_fire"]  = _next_fire_label(job)
    d["is_running"] = job.job_id in self.job_scheduler._firing
    d = _enrich_with_live(d, live.get(job_id))
    d["ok"] = True
    return d


# ── 7. OS autowake control ────────────────────────────────────────────────────

@export(mod_name="isaa", name="jobAutowake", api=True, request_as_kwarg=True)
async def job_autowake(
    self,
    action: str = "status",             # install | remove | status
    request: RequestData | None = None,
) -> dict:
    if request:
        action = (request.request.body or {}).get("action", action)
    try:
        from toolboxv2.mods.isaa.extras.jobs.os_scheduler import (
            install_autowake, remove_autowake, autowake_status,
        )
    except ImportError:
        return {"ok": False, "message": "os_scheduler not available"}

    if action == "install":
        msg = install_autowake(self.job_scheduler.jobs_file)
    elif action == "remove":
        msg = remove_autowake()
    else:
        msg = autowake_status()

    return {"ok": True, "action": action, "message": msg}


# ── 8. Launch ISAA CLI in a new terminal window ───────────────────────────────

@export(mod_name="isaa", name="launchCLI", api=True, request_as_kwarg=True)
async def launch_cli(
    self,
    terminal: str = "auto",             # auto | wt | xterm | gnome-terminal | tmux | screen
    extra_args: list[str] | None = None,
    request: RequestData | None = None,
) -> dict:
    """
    Start the ISAA interactive CLI (icli.py) in a new OS terminal window.
    Works on Windows (wt / cmd), Linux (xterm / gnome-terminal), macOS (osascript).
    """
    if request:
        body     = request.request.body or {}
        terminal = body.get("terminal", terminal)
        extra_args = body.get("extra_args", extra_args)

    py   = sys.executable
    cmd  = [py, "-m", "toolboxv2.flows.icli"] + (extra_args or [])
    cmd_str = " ".join(f'"{c}"' if " " in c else c for c in cmd)

    launched = False
    error    = ""
    used     = ""

    def _try(args):
        subprocess.Popen(args, creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0)

    system = sys.platform

    try:
        if terminal == "auto":
            if system == "win32":
                # Prefer Windows Terminal, fallback to cmd
                try:
                    _try(["wt", "--", *cmd])
                    used = "wt"
                except FileNotFoundError:
                    _try(["cmd", "/c", "start", "cmd", "/k", cmd_str])
                    used = "cmd"
            elif system == "darwin":
                script = f'tell application "Terminal" to do script "{cmd_str}"'
                subprocess.Popen(["osascript", "-e", script])
                used = "Terminal.app"
            else:
                # Linux: try common terminals in order
                for term in ["gnome-terminal", "xterm", "konsole", "xfce4-terminal"]:
                    try:
                        if term == "gnome-terminal":
                            _try([term, "--", *cmd])
                        else:
                            _try([term, "-e", cmd_str])
                        used = term
                        break
                    except FileNotFoundError:
                        continue
                else:
                    # Fallback: tmux new-window or screen
                    try:
                        _try(["tmux", "new-window", cmd_str])
                        used = "tmux"
                    except FileNotFoundError:
                        _try(["screen", "-dm", *cmd])
                        used = "screen"
        else:
            # Explicit terminal requested
            if terminal == "wt":
                _try(["wt", "--", *cmd])
            elif terminal == "tmux":
                _try(["tmux", "new-window", cmd_str])
            elif terminal == "screen":
                _try(["screen", "-dm", *cmd])
            else:
                _try([terminal, "-e", cmd_str])
            used = terminal

        launched = True
    except Exception as e:
        error = str(e)

    return {
        "ok": launched,
        "terminal": used,
        "command": cmd_str,
        "message": f"CLI launched in '{used}'" if launched else f"Launch failed: {error}",
    }


# ── 9. Launch Job Viewer (web + optional terminal TUI) ───────────────────────

@export(mod_name="isaa", name="launchViewer", api=True, request_as_kwarg=True)
async def launch_viewer(
    self,
    mode: str = "web",                  # web | terminal | both
    port: int = 7799,
    refresh: float = 1.0,
    request: RequestData | None = None,
) -> dict:
    """
    Start the ISAA Job Viewer.
    - mode='web'      → background HTTP server on /port/
    - mode='terminal' → new terminal window with rich TUI
    - mode='both'     → web server + terminal TUI
    Returns the URL for the web viewer.
    """
    if request:
        body    = request.request.body or {}
        mode    = body.get("mode", mode)
        port    = body.get("port", port)
        refresh = body.get("refresh", refresh)

    jobs_file = self.job_scheduler.jobs_file
    live_file = jobs_file.with_suffix(".live.json")
    py        = sys.executable
    viewer_mod = "toolboxv2.mods.isaa.extras.jobs.job_viewer"

    launched_web      = False
    launched_terminal = False
    url               = None

    # ── Web server (in-process daemon thread) ────────────────────────────────
    if mode in ("web", "both"):
        from toolboxv2.mods.isaa.extras.jobs.job_viewer import run_web_viewer
        t = threading.Thread(
            target=run_web_viewer,
            args=(jobs_file, live_file, port),
            daemon=True,
            name=f"isaa-job-viewer-web-{port}",
        )
        # Only start if not already running
        existing = [th for th in threading.enumerate() if th.name == t.name]
        if not existing:
            t.start()
        launched_web = True
        url = f"http://localhost:{port}"

    # ── Terminal TUI in new window ────────────────────────────────────────────
    if mode in ("terminal", "both"):
        viewer_args = [
            py, "-m", viewer_mod,
            "--jobs-file", str(jobs_file),
            "--refresh", str(refresh),
        ]
        cmd_str = " ".join(viewer_args)
        try:
            system = sys.platform
            if system == "win32":
                try:
                    subprocess.Popen(["wt", "--", *viewer_args],
                                     creationflags=subprocess.CREATE_NEW_CONSOLE)
                except FileNotFoundError:

                    subprocess.Popen(["cmd", "/c", "start", "cmd", "/k", cmd_str])
            elif system == "darwin":
                script = f'tell application "Terminal" to do script "{cmd_str}"'
                subprocess.Popen(["osascript", "-e", script])
            else:
                for term in ["gnome-terminal", "xterm", "konsole"]:
                    try:
                        if term == "gnome-terminal":
                            subprocess.Popen([term, "--", *viewer_args])
                        else:
                            subprocess.Popen([term, "-e", cmd_str])
                        break
                    except FileNotFoundError:
                        continue
            launched_terminal = True
        except Exception as e:
            pass

    return {
        "ok": launched_web or launched_terminal,
        "mode": mode,
        "web_url": url,
        "web_launched": launched_web,
        "terminal_launched": launched_terminal,
        "message": (
            f"Viewer running at {url}" if launched_web
            else "Terminal viewer launched" if launched_terminal
            else "Launch failed"
        ),
    }


# ── 10. Combined job + viewer status snapshot ─────────────────────────────────

@export(mod_name="isaa", name="jobDashboard", api=True, request_as_kwarg=True)
async def job_dashboard(self, request: RequestData | None = None) -> dict:
    """
    Single-call snapshot: all jobs + live state + scheduler + autowake status.
    Designed for a frontend dashboard polling every few seconds.
    """
    from toolboxv2.mods.isaa.extras.jobs.job_live_state import JobLiveStateReader
    try:
        from toolboxv2.mods.isaa.extras.jobs.os_scheduler import autowake_status
        aw = autowake_status()
    except Exception:
        aw = "unavailable"

    live_file = self.job_scheduler.jobs_file.with_suffix(".live.json")
    live      = JobLiveStateReader(live_file).read()

    jobs_out = []
    for job in self.job_scheduler.list_jobs():
        d = job.to_dict()
        d["next_fire"]  = _next_fire_label(job)
        d["is_running"] = job.job_id in self.job_scheduler._firing
        d = _enrich_with_live(d, live.get(job.job_id))
        jobs_out.append(d)

    return {
        "jobs":            jobs_out,
        "total":           self.job_scheduler.total_count,
        "active":          self.job_scheduler.active_count,
        "running":         len(self.job_scheduler._firing),
        "autowake_status": aw,
        "has_persistent":  self.job_scheduler.has_persistent_jobs(),
        "jobs_file":       str(self.job_scheduler.jobs_file),
    }

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
        success, manifest = await isaa_tool_instance.save_agent(
            "self", "/tmp/test_agent.tar.gz"
        )
        print(f"Save agent: {success}")
        if success:
            print(f"Manifest: {manifest}")


    import sys
    if "--test-ollama" in sys.argv:
        from toolboxv2.mods.isaa.model_tool_test import run_ollama_tool_diagnostic_cli
        run_ollama_tool_diagnostic_cli()
    else:
        asyncio.run(test_isaa_tools())  # bisheriger Test
    asyncio.run(test_isaa_tools())
