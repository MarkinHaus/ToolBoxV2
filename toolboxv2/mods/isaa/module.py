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
from .base.Agent.agent import (
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


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            # For enum members, return their value (e.g., "pending")
            return obj.value
        # Let the base class default method raise the TypeError for other types
        return super().default(obj)

class EnhancedAgentRequestHandler(BaseHTTPRequestHandler):
    """Enhanced HTTP request handler for standalone server with comprehensive UI support."""

    def __init__(self, isaa_mod, agent_id: str, agent, *args, **kwargs):
        self.isaa_mod = isaa_mod
        self.agent_id = agent_id
        self.agent = agent
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests for enhanced UI and status."""
        parsed_path = urlparse(self.path)

        if parsed_path.path in ['/', '/ui']:
            self._serve_enhanced_ui()
        elif parsed_path.path in ['/api/status', '/api/agent_ui/status', '/status']:
            self._serve_status()
        else:
            self._send_404()

    def do_POST(self):
        """Handle POST requests for enhanced API endpoints."""
        parsed_path = urlparse(self.path)

        if parsed_path.path in ['/api/run', '/api/agent_ui/run_agent']:
            self._handle_run_request()
        elif parsed_path.path in ['/api/reset', '/api/agent_ui/reset_context']:
            self._handle_reset_request()
        else:
            self._send_404()

    def _serve_enhanced_ui(self):
        """Serve the enhanced UI HTML."""
        try:
            from .extras.agent_ui import get_agent_ui_html
            html_content = get_agent_ui_html()

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Content-Length', str(len(html_content.encode('utf-8'))))
            self.end_headers()
            self.wfile.write(html_content.encode('utf-8'))

        except Exception as e:
            self._send_error_response(500, f"Error serving UI: {str(e)}")

    def _serve_status(self):
        """Serve enhanced status information."""
        try:
            status_info = {
                'agent_id': self.agent_id,
                'agent_name': getattr(self.agent, 'name', 'Unknown'),
                'agent_type': self.agent.__class__.__name__,
                'status': 'active',
                'server_type': 'standalone',
                'timestamp': time.time()
            }

            if hasattr(self.agent, 'status'):
                try:
                    agent_status = self.agent.status()
                    if isinstance(agent_status, dict):
                        status_info['agent_status'] = agent_status
                except:
                    pass

            response_data = json.dumps(status_info).encode('utf-8')

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Content-Length', str(len(response_data)))
            self.end_headers()
            self.wfile.write(response_data)

        except Exception as e:
            self._send_error_response(500, f"Error getting status: {str(e)}")

    def _handle_run_request(self):
        """Handle enhanced run requests with comprehensive progress tracking."""
        try:
            content_length = int(self.headers['Content-Length'])
            request_body = self.rfile.read(content_length)
            request_data = json.loads(request_body.decode('utf-8'))

            query = request_data.get('query', '')
            session_id = request_data.get('session_id', f'standalone_{secrets.token_hex(8)}')
            include_progress = request_data.get('include_progress', False)

            if not query:
                self._send_error_response(400, "Missing 'query' field")
                return

            # Run agent with enhanced progress tracking
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                progress_tracker = EnhancedProgressTracker()
                progress_events = []
                enhanced_progress = {}

                async def standalone_progress_callback(event: ProgressEvent):
                    if include_progress:
                        progress_data = progress_tracker.extract_progress_data(event)
                        progress_events.append({
                            'timestamp': event.timestamp,
                            'event_type': event.event_type,
                            'status': getattr(event, 'status', 'unknown').value if hasattr(event, 'status') and event.status else 'unknown',
                            'data': event.to_dict()
                        })
                        enhanced_progress.update(progress_data)

                # Set progress callback
                original_callback = getattr(self.agent, 'progress_callback', None)

                if hasattr(self.agent, 'set_progress_callback'):
                    self.agent.set_progress_callback(standalone_progress_callback)
                elif hasattr(self.agent, 'progress_callback'):
                    self.agent.progress_callback = standalone_progress_callback

                # Execute agent
                result = loop.run_until_complete(
                    self.agent.a_run(query=query, session_id=session_id)
                )

                # Restore callback
                if hasattr(self.agent, 'set_progress_callback'):
                    self.agent.set_progress_callback(original_callback)
                elif hasattr(self.agent, 'progress_callback'):
                    self.agent.progress_callback = original_callback

                # Create enhanced response
                response_data = {
                    'success': True,
                    'result': result,
                    'session_id': session_id,
                    'agent_id': self.agent_id,
                    'server_type': 'standalone',
                    'timestamp': time.time()
                }

                if include_progress:
                    response_data.update({
                        'progress_events': progress_events,
                        'enhanced_progress': enhanced_progress,
                        'final_summary': progress_tracker.get_final_summary()
                    })
                self._send_json_response(response_data)

            finally:
                loop.close()

        except Exception as e:
            self._send_error_response(500, f"Execution error: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def _handle_reset_request(self):
        """Handle enhanced reset requests."""
        try:
            success = False
            message = "Reset not supported"

            if hasattr(self.agent, 'clear_context'):
                self.agent.clear_context()
                success = True
                message = "Context reset successfully"
            elif hasattr(self.agent, 'reset'):
                self.agent.reset()
                success = True
                message = "Agent reset successfully"

            response_data = {
                'success': success,
                'message': message,
                'agent_id': self.agent_id,
                'timestamp': time.time()
            }

            self._send_json_response(response_data)

        except Exception as e:
            self._send_error_response(500, f"Reset error: {str(e)}")

    def _send_json_response(self, data: dict):
        """Send JSON response with CORS headers."""
        response_body = json.dumps(data, cls=CustomJSONEncoder).encode('utf-8')

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-Length', str(len(response_body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(response_body)

    def _send_error_response(self, code: int, message: str):
        """Send error response."""
        error_data = {'success': False, 'error': message, 'code': code}
        response_body = json.dumps(error_data).encode('utf-8')

        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-Length', str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def _send_404(self):
        """Send 404 response."""
        self._send_error_response(404, "Not Found")

    def log_message(self, format, *args):
        """Override to reduce logging noise."""
        pass

    def do_OPTIONS(self):
        """Handle preflight CORS requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

class EnhancedProgressTracker:
    """Enhanced progress tracker for detailed UI updates."""

    def __init__(self):
        self.session_state = {}
        self.last_outline_update = None
        self.last_activity_update = None

    def extract_progress_data(self, event: ProgressEvent) -> dict[str, Any]:
        """Extract comprehensive progress data from event."""
        progress_data = {}

        # Outline progress
        if hasattr(event, 'outline_data') or 'outline' in event.metadata:
            outline_info = getattr(event, 'outline_data', event.metadata.get('outline', {}))
            progress_data['outline'] = {
                'current_step': outline_info.get('current_step', 'Unknown'),
                'total_steps': outline_info.get('total_steps', 0),
                'step_name': outline_info.get('step_name', 'Processing'),
                'progress_percentage': outline_info.get('progress_percentage', 0),
                'substeps': outline_info.get('substeps', []),
                'estimated_completion': outline_info.get('estimated_completion')
            }

        # Activity information
        if hasattr(event, 'activity_data') or 'activity' in event.metadata:
            activity_info = getattr(event, 'activity_data', event.metadata.get('activity', {}))
            progress_data['activity'] = {
                'current_action': activity_info.get('current_action', 'Processing'),
                'action_details': activity_info.get('action_details', ''),
                'start_time': activity_info.get('start_time'),
                'elapsed_time': activity_info.get('elapsed_time'),
                'expected_duration': activity_info.get('expected_duration')
            }

        # Meta tool information
        if hasattr(event, 'meta_tool_data') or 'meta_tool' in event.metadata:
            meta_tool_info = getattr(event, 'meta_tool_data', event.metadata.get('meta_tool', {}))
            progress_data['meta_tool'] = {
                'tool_name': meta_tool_info.get('tool_name', 'Unknown'),
                'tool_status': meta_tool_info.get('tool_status', 'active'),
                'tool_input': meta_tool_info.get('tool_input', ''),
                'tool_output': meta_tool_info.get('tool_output', ''),
                'execution_time': meta_tool_info.get('execution_time')
            }

        # System status
        if hasattr(event, 'system_data') or 'system' in event.metadata:
            system_info = getattr(event, 'system_data', event.metadata.get('system', {}))
            progress_data['system'] = {
                'memory_usage': system_info.get('memory_usage', 0),
                'cpu_usage': system_info.get('cpu_usage', 0),
                'active_threads': system_info.get('active_threads', 1),
                'queue_size': system_info.get('queue_size', 0)
            }

        # Graph/workflow information
        if hasattr(event, 'graph_data') or 'graph' in event.metadata:
            graph_info = getattr(event, 'graph_data', event.metadata.get('graph', {}))
            progress_data['graph'] = {
                'current_node': graph_info.get('current_node', 'Unknown'),
                'completed_nodes': graph_info.get('completed_nodes', []),
                'remaining_nodes': graph_info.get('remaining_nodes', []),
                'node_connections': graph_info.get('node_connections', []),
                'execution_path': graph_info.get('execution_path', [])
            }

        return progress_data

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
                # asyncio.run(agent_instance.close()) # This might block, consider task group
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

        if agent_instance.amd.name == "self":
            self.app.run_bg_task_advanced(agent_instance.initialize_context_awareness)

        if interface := self.get_tools_interface(agent_name):
            interface.variable_manager = agent_instance.variable_manager

        # colletive cabability cahring for reduched reduanda analysis _tool_capabilities
        agent_tool_nams = set(agent_instance.tool_registry.keys())

        tools_data = {}
        for _agent_name in self.config["agents-name-list"]:
            _instance_key = f'agent-instance-{_agent_name}'
            if _instance_key not in self.config:
                if agent_name != "self" and _agent_name == "self":
                    await self.get_agent("self")

            if _instance_key not in self.config:
                continue
            _agent_instance = self.config[_instance_key]
            _agent_tool_nams = set(_agent_instance._tool_capabilities.keys())
            # extract the tool names that are in both agents_registry
            overlap_tool_nams = agent_tool_nams.intersection(_agent_tool_nams)
            _tc = _agent_instance._tool_capabilities
            for tool_name in overlap_tool_nams:
                if tool_name not in _tc:
                    continue
                tools_data[tool_name] = _tc[tool_name]

        agent_instance._tool_capabilities.update(tools_data)
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

    async def host_agent_ui(
        self,
        agent,
        host: str = "0.0.0.0",
        port: int | None = None,
        access: str = 'local',
        registry_server: str | None = None,
        public_name: str | None = None,
        description: str | None = None,
        use_builtin_server: bool = None
    ) -> dict[str, str]:
        """
        Unified agent hosting with WebSocket-enabled UI and optional registry publishing.

        Args:
            agent: Agent or Chain instance to host
            host: Host address (default: 0.0.0.0 for remote access)
            port: Port number (auto-assigned if None)
            access: 'local', 'remote', or 'registry'
            registry_server: Registry server URL for publishing (e.g., "ws://localhost:8080/ws/registry/connect")
            public_name: Public name for registry publishing
            description: Description for registry publishing
            use_builtin_server: Use toolbox built-in server vs standalone Python server

        Returns:
            Dictionary with access URLs and configuration
        """
        use_builtin_server = use_builtin_server or self.app.is_server
        if not hasattr(self, '_hosted_agents'):
            self._hosted_agents = {}

        agent_id = f"agent_{secrets.token_urlsafe(8)}"

        # Generate unique port if not specified
        if not port:
            port = 8765 + len(self._hosted_agents)

        # Store agent reference
        self._hosted_agents[agent_id] = {
            'agent': agent,
            'port': port,
            'host': host,
            'access': access,
            'public_name': public_name or f"Agent_{agent_id}",
            'description': description
        }

        result = {
            'agent_id': agent_id,
            'local_url': f"http://{host}:{port}",
            'status': 'starting'
        }

        if use_builtin_server:
            # Use toolbox built-in server
            result.update(await self._setup_builtin_server_hosting(agent_id, agent, host, port))
        else:
            # Use standalone Python server
            result.update(await self._setup_standalone_server_hosting(agent_id, agent, host, port))

        # Handle registry publishing if requested
        if access in ['remote', 'registry'] and registry_server:
            if not public_name:
                raise ValueError("public_name required for registry publishing")

            registry_result = await self._publish_to_registry(
                agent=agent,
                public_name=public_name,
                registry_server=registry_server,
                description=description,
                agent_id=agent_id
            )
            result.update(registry_result)

        self.app.print(f"🚀 Agent '{result.get('public_name', agent_id)}' hosted successfully!")
        self.app.print(f"   Local UI: {result['local_url']}")
        if 'public_url' in result:
            self.app.print(f"   Public URL: {result['public_url']}")
            self.app.print(f"   API Key: {result.get('api_key', 'N/A')}")

        return result

    # toolboxv2/mods/isaa/__init__.py - Missing Methods

    import asyncio
    import json
    import secrets
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse


    async def _handle_reset_context(self, agent_id: str, agent, conn_id: str):
        """Handle context reset requests from WebSocket UI."""

        try:
            # Reset agent context if supported
            if hasattr(agent, 'clear_context'):
                agent.clear_context()
                message = "Context reset successfully"
                success = True
            else:
                message = "Agent does not support context reset"
                success = False

            # Send response back to UI
            await self._broadcast_to_agent_ui(agent_id, {
                'event': 'reset_response',
                'data': {
                    'success': success,
                    'message': message,
                    'timestamp': time.time()
                }
            })

            self.app.print(f"Context reset requested for agent {agent_id}: {message}")

        except Exception as e:
            error_message = f"Context reset failed: {str(e)}"
            self.app.print(f"Context reset error for agent {agent_id}: {e}")

            await self._broadcast_to_agent_ui(agent_id, {
                'event': 'error',
                'data': {
                    'error': error_message,
                    'timestamp': time.time()
                }
            })

    async def _handle_get_status(self, agent_id: str, agent, conn_id: str):
        """Handle status requests from WebSocket UI."""

        try:
            # Collect agent status information
            status_info = {
                'agent_id': agent_id,
                'agent_name': getattr(agent, 'name', 'Unknown'),
                'agent_type': agent.__class__.__name__,
                'status': 'active',
                'timestamp': time.time(),
                'server_type': 'builtin'
            }

            # Add additional status if available
            if hasattr(agent, 'status'):
                try:
                    agent_status = agent.status()
                    if isinstance(agent_status, dict):
                        status_info.update(agent_status)
                except:
                    pass

            # Add hosted agent info
            if hasattr(self, '_hosted_agents') and agent_id in self._hosted_agents:
                hosted_info = self._hosted_agents[agent_id]
                status_info.update({
                    'host': hosted_info.get('host'),
                    'port': hosted_info.get('port'),
                    'access': hosted_info.get('access'),
                    'public_name': hosted_info.get('public_name')
                })

            # Send status back to UI
            await self._broadcast_to_agent_ui(agent_id, {
                'event': 'status_response',
                'data': status_info
            })

            self.app.print(f"Status requested for agent {agent_id}")

        except Exception as e:
            error_message = f"Status retrieval failed: {str(e)}"
            self.app.print(f"Status error for agent {agent_id}: {e}")

            await self._broadcast_to_agent_ui(agent_id, {
                'event': 'error',
                'data': {
                    'error': error_message,
                    'timestamp': time.time()
                }
            })


    async def stop_hosted_agent(self, agent_id: str = None, port: int = None):
        """Stop a hosted agent by agent_id or port."""

        if not hasattr(self, '_hosted_agents') and not hasattr(self, '_standalone_servers'):
            self.app.print("No hosted agents found")
            return False

        # Stop by agent_id
        if agent_id:
            if hasattr(self, '_hosted_agents') and agent_id in self._hosted_agents:
                agent_info = self._hosted_agents[agent_id]
                agent_port = agent_info.get('port')

                # Stop standalone server if exists
                if hasattr(self, '_standalone_servers') and agent_port in self._standalone_servers:
                    server_info = self._standalone_servers[agent_port]
                    try:
                        server_info['server'].shutdown()
                        self.app.print(f"Stopped standalone server for agent {agent_id}")
                    except:
                        pass

                # Clean up hosted agent info
                del self._hosted_agents[agent_id]
                self.app.print(f"Stopped hosted agent {agent_id}")
                return True

        # Stop by port
        if port:
            if hasattr(self, '_standalone_servers') and port in self._standalone_servers:
                server_info = self._standalone_servers[port]
                try:
                    server_info['server'].shutdown()
                    self.app.print(f"Stopped server on port {port}")
                    return True
                except Exception as e:
                    self.app.print(f"Failed to stop server on port {port}: {e}")
                    return False

        self.app.print("Agent or port not found")
        return False

    async def list_hosted_agents(self) -> dict[str, Any]:
        """List all currently hosted agents."""

        hosted_info = {
            'builtin_agents': {},
            'standalone_agents': {},
            'total_count': 0
        }

        # Built-in server agents
        if hasattr(self, '_hosted_agents'):
            for agent_id, info in self._hosted_agents.items():
                hosted_info['builtin_agents'][agent_id] = {
                    'public_name': info.get('public_name'),
                    'host': info.get('host'),
                    'port': info.get('port'),
                    'access': info.get('access'),
                    'description': info.get('description')
                }

        # Standalone server agents
        if hasattr(self, '_standalone_servers'):
            for port, info in self._standalone_servers.items():
                hosted_info['standalone_agents'][info['agent_id']] = {
                    'port': port,
                    'thread_alive': info['thread'].is_alive(),
                    'server_type': 'standalone'
                }

        hosted_info['total_count'] = len(hosted_info['builtin_agents']) + len(hosted_info['standalone_agents'])

        return hosted_info

    def _create_agent_ws_connect_handler(self, agent_id: str):
        """Create WebSocket connect handler for specific agent."""

        async def on_connect(app, conn_id: str, session: dict):
            if not hasattr(self, '_agent_connections'):
                self._agent_connections = {}

            if agent_id not in self._agent_connections:
                self._agent_connections[agent_id] = set()

            self._agent_connections[agent_id].add(conn_id)

            # Send initial status
            await app.ws_send(conn_id, {
                'event': 'agent_connected',
                'data': {
                    'agent_id': agent_id,
                    'status': 'ready',
                    'capabilities': ['chat', 'progress_tracking', 'real_time_updates']
                }
            })

            self.app.print(f"UI client connected to agent {agent_id}: {conn_id}")

        return on_connect

    def _create_agent_ws_message_handler(self, agent_id: str, agent):
        """Create WebSocket message handler for specific agent."""

        async def on_message(app, conn_id: str, session: dict, payload: dict):
            event = payload.get('event')
            data = payload.get('data', {})

            if event == 'chat_message':
                await self._handle_chat_message(agent_id, agent, conn_id, data)
            elif event == 'reset_context':
                await self._handle_reset_context(agent_id, agent, conn_id)
            elif event == 'get_status':
                await self._handle_get_status(agent_id, agent, conn_id)
            else:
                self.app.print(f"Unknown event from UI: {event}")

        return on_message

    def _create_agent_ws_disconnect_handler(self, agent_id: str):
        """Create WebSocket disconnect handler for specific agent."""

        async def on_disconnect(app, conn_id: str, session: dict = None):
            if hasattr(self, '_agent_connections') and agent_id in self._agent_connections:
                self._agent_connections[agent_id].discard(conn_id)

            self.app.print(f"UI client disconnected from agent {agent_id}: {conn_id}")

        return on_disconnect


    async def _broadcast_to_agent_ui(self, agent_id: str, message: dict):
        """Broadcast message to all UI clients connected to specific agent."""
        if not hasattr(self, '_agent_connections') or agent_id not in self._agent_connections:
            return

        for conn_id in self._agent_connections[agent_id].copy():
            try:
                await self.app.ws_send(conn_id, message)
            except Exception as e:
                self.app.print(f"Failed to send to UI client {conn_id}: {e}")
                self._agent_connections[agent_id].discard(conn_id)

    async def _publish_to_registry(
        self,
        agent,
        public_name: str,
        registry_server: str,
        description: str | None = None,
        agent_id: str | None = None
    ) -> dict[str, str]:
        """Publish agent to registry server."""
        try:
            # Import registry client dynamically to avoid circular imports
            registry_client_module = __import__("toolboxv2.mods.registry.client", fromlist=["get_registry_client"])
            get_registry_client = registry_client_module.get_registry_client

            client = get_registry_client(self.app)

            # Connect if not already connected
            if not client.ws or not client.ws.open:
                await client.connect(registry_server)

            if not client.ws or not client.ws.open:
                raise Exception("Failed to connect to registry server")

            # Register the agent
            reg_info = await client.register(agent, public_name, description)

            if reg_info:
                return {
                    'public_url': reg_info.public_url,
                    'api_key': reg_info.public_api_key,
                    'public_agent_id': reg_info.public_agent_id,
                    'registry_status': 'published'
                }
            else:
                raise Exception("Registration failed")

        except Exception as e:
            self.app.print(f"Registry publishing failed: {e}")
            return {'registry_status': 'failed', 'registry_error': str(e)}

    def _get_enhanced_agent_ui_html(self, agent_id: str) -> str:
        """Get production-ready enhanced UI HTML with comprehensive progress visualization."""
        agent_info = self._hosted_agents.get(agent_id, {})
        server_info = {
            'server_type': 'standalone' if not hasattr(self.app, 'tb') else 'builtin',
            'agent_id': agent_id
        }

        # Update the JavaScript section in the HTML template:
        js_config = f"""
                window.SERVER_CONFIG = {json.dumps(server_info)};
            """
        html_template = """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{agent_name}</title>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            :root {
                --bg-primary: #0d1117;
                --bg-secondary: #161b22;
                --bg-tertiary: #21262d;
                --text-primary: #f0f6fc;
                --text-secondary: #8b949e;
                --text-muted: #6e7681;
                --accent-blue: #58a6ff;
                --accent-green: #3fb950;
                --accent-red: #f85149;
                --accent-orange: #d29922;
                --accent-purple: #a5a5f5;
                --accent-cyan: #39d0d8;
                --border-color: #30363d;
                --shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            }

            * { margin: 0; padding: 0; box-sizing: border-box; }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
                background: var(--bg-primary);
                color: var(--text-primary);
                height: 100vh;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }

            .header {
                background: var(--bg-tertiary);
                padding: 12px 20px;
                border-bottom: 1px solid var(--border-color);
                display: flex;
                align-items: center;
                justify-content: space-between;
                box-shadow: var(--shadow);
                z-index: 100;
            }

            .agent-info {
                display: flex;
                align-items: center;
                gap: 16px;
            }

            .agent-title {
                font-size: 18px;
                font-weight: 600;
                color: var(--accent-blue);
            }

            .agent-status {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 14px;
            }

            .status-dot {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: var(--accent-red);
                animation: pulse 2s infinite;
            }

            .status-dot.connected {
                background: var(--accent-green);
                animation: none;
            }

            .status-dot.processing {
                background: var(--accent-orange);
                animation: pulse 1s infinite;
            }

            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }

            .main-container {
                display: grid;
                grid-template-columns: 2fr 1.5fr 1fr;
                grid-template-rows: 1fr 1fr;
                grid-template-areas:
                    "chat outline activity"
                    "chat system graph";
                flex: 1;
                gap: 1px;
                background: var(--border-color);
                overflow: hidden;
            }

            .panel {
                background: var(--bg-secondary);
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }

            .chat-panel { grid-area: chat; }
            .outline-panel { grid-area: outline; }
            .activity-panel { grid-area: activity; }
            .system-panel { grid-area: system; }
            .graph-panel { grid-area: graph; }

            .panel-header {
                padding: 12px 16px;
                background: var(--bg-tertiary);
                border-bottom: 1px solid var(--border-color);
                font-weight: 600;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .panel-content {
                flex: 1;
                overflow-y: auto;
                padding: 12px;
            }

            /* Chat Panel Styles */
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 16px;
                display: flex;
                flex-direction: column;
                gap: 16px;
            }

            .message {
                display: flex;
                align-items: flex-start;
                gap: 12px;
                max-width: 85%;
            }

            .message.user {
                flex-direction: row-reverse;
                margin-left: auto;
            }

            .message-avatar {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                font-weight: 600;
                flex-shrink: 0;
            }

            .message.user .message-avatar {
                background: var(--accent-blue);
            }

            .message.agent .message-avatar {
                background: var(--accent-green);
            }

            .message-content {
                padding: 12px 16px;
                border-radius: 12px;
                line-height: 1.5;
                font-size: 14px;
            }

            .message.user .message-content {
                background: var(--accent-blue);
                color: white;
            }

            .message.agent .message-content {
                background: var(--bg-tertiary);
                border: 1px solid var(--border-color);
            }

            .chat-input-area {
                border-top: 1px solid var(--border-color);
                padding: 16px;
                display: flex;
                gap: 12px;
            }

            .chat-input {
                flex: 1;
                background: var(--bg-primary);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 12px;
                color: var(--text-primary);
                font-size: 14px;
            }

            .chat-input:focus {
                outline: none;
                border-color: var(--accent-blue);
            }

            .send-button {
                background: var(--accent-blue);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.2s;
            }

            .send-button:hover:not(:disabled) {
                background: #4493f8;
                transform: translateY(-1px);
            }

            .send-button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: none;
            }

            /* Progress Indicator */
            .progress-indicator {
                display: none;
                align-items: center;
                gap: 12px;
                padding: 12px 16px;
                background: var(--bg-tertiary);
                border-top: 1px solid var(--border-color);
                font-size: 14px;
            }

            .progress-indicator.active { display: flex; }

            .spinner {
                width: 16px;
                height: 16px;
                border: 2px solid var(--border-color);
                border-top: 2px solid var(--accent-blue);
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            /* Outline Panel Styles */
            .outline-progress {
                margin-bottom: 16px;
            }

            .outline-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 12px;
            }

            .outline-title {
                font-weight: 600;
                color: var(--accent-cyan);
            }

            .outline-stats {
                font-size: 12px;
                color: var(--text-muted);
            }

            .progress-bar {
                width: 100%;
                height: 6px;
                background: var(--bg-primary);
                border-radius: 3px;
                overflow: hidden;
                margin-bottom: 16px;
            }

            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
                width: 0%;
                transition: width 0.5s ease;
            }

            .outline-steps {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }

            .outline-step {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 8px 12px;
                border-radius: 6px;
                background: var(--bg-primary);
                border-left: 3px solid var(--border-color);
                transition: all 0.3s;
            }

            .outline-step.active {
                border-left-color: var(--accent-orange);
                background: rgba(217, 153, 34, 0.1);
            }

            .outline-step.completed {
                border-left-color: var(--accent-green);
                background: rgba(63, 185, 80, 0.1);
            }

            .step-icon {
                font-size: 14px;
                width: 16px;
            }

            .step-text {
                flex: 1;
                font-size: 13px;
            }

            .step-method {
                font-size: 11px;
                color: var(--text-muted);
                background: var(--bg-tertiary);
                padding: 2px 6px;
                border-radius: 4px;
            }

            /* Activity Panel Styles */
            .current-activity {
                background: var(--bg-primary);
                border: 1px solid var(--border-color);
                border-radius: 6px;
                padding: 12px;
                margin-bottom: 12px;
            }

            .activity-header {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 8px;
            }

            .activity-title {
                font-weight: 600;
                color: var(--accent-orange);
            }

            .activity-duration {
                font-size: 11px;
                color: var(--text-muted);
                background: var(--bg-tertiary);
                padding: 2px 6px;
                border-radius: 4px;
            }

            .activity-description {
                font-size: 13px;
                line-height: 1.4;
                color: var(--text-secondary);
            }

            .meta-tools-list {
                display: flex;
                flex-direction: column;
                gap: 6px;
            }

            .meta-tool {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 6px 10px;
                background: var(--bg-primary);
                border-radius: 4px;
                font-size: 12px;
            }

            .tool-icon {
                width: 12px;
                text-align: center;
            }

            .tool-name {
                flex: 1;
                color: var(--text-secondary);
            }

            .tool-status {
                font-size: 10px;
                padding: 2px 6px;
                border-radius: 3px;
            }

            .tool-status.running {
                background: var(--accent-orange);
                color: white;
            }

            .tool-status.completed {
                background: var(--accent-green);
                color: white;
            }

            .tool-status.error {
                background: var(--accent-red);
                color: white;
            }

            /* System Panel Styles */
            .system-grid {
                display: grid;
                grid-template-columns: 1fr 2fr;
                gap: 8px 12px;
                font-size: 12px;
            }

            .system-key {
                color: var(--text-muted);
                font-weight: 500;
            }

            .system-value {
                color: var(--text-primary);
                font-family: 'SF Mono', Monaco, monospace;
                word-break: break-word;
            }

            .current-node {
                background: var(--bg-primary);
                padding: 8px 10px;
                border-radius: 6px;
                margin-bottom: 12px;
                border: 1px solid var(--border-color);
            }

            .node-name {
                font-weight: 600;
                color: var(--accent-purple);
                margin-bottom: 4px;
            }

            .node-operation {
                font-size: 11px;
                color: var(--text-muted);
            }

            /* Graph Panel Styles */
            .agent-graph {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 8px;
                padding: 8px;
            }

            .graph-node {
                padding: 6px 12px;
                background: var(--bg-primary);
                border: 1px solid var(--border-color);
                border-radius: 6px;
                font-size: 11px;
                text-align: center;
                min-width: 80px;
            }

            .graph-node.active {
                border-color: var(--accent-orange);
                background: rgba(217, 153, 34, 0.1);
            }

            .graph-node.completed {
                border-color: var(--accent-green);
                background: rgba(63, 185, 80, 0.1);
            }

            .graph-arrow {
                color: var(--text-muted);
                font-size: 12px;
            }

            /* Connection Error Styles */
            .connection-error {
                background: var(--accent-red);
                color: white;
                padding: 8px 12px;
                margin: 8px;
                border-radius: 6px;
                font-size: 12px;
                text-align: center;
            }

            .fallback-mode {
                background: var(--accent-orange);
                color: white;
                padding: 8px 12px;
                margin: 8px;
                border-radius: 6px;
                font-size: 12px;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="agent-info">
                <div class="agent-title">{agent_name}</div>
                <div class="text-secondary">{agent_description}</div>
            </div>
            <div class="agent-status">
                <div class="status-dot" id="status-dot"></div>
                <span id="status-text">Initializing...</span>
            </div>
        </div>

        <div class="main-container">
            <!-- Chat Panel -->
            <div class="panel chat-panel">
                <div class="panel-header">💬 Conversation</div>
                <div class="chat-messages" id="chat-messages">
                    <div class="message agent">
                        <div class="message-avatar">AI</div>
                        <div class="message-content">Hello! I'm ready to help you. What would you like to know?</div>
                    </div>
                </div>
                <div class="progress-indicator" id="progress-indicator">
                    <div class="spinner"></div>
                    <span id="progress-text">Processing...</span>
                </div>
                <div class="chat-input-area">
                    <input type="text" id="chat-input" class="chat-input" placeholder="Type your message...">
                    <button id="send-button" class="send-button">Send</button>
                </div>
            </div>

            <!-- Outline & Progress Panel -->
            <div class="panel outline-panel">
                <div class="panel-header">📋 Execution Outline</div>
                <div class="panel-content">
                    <div class="outline-progress">
                        <div class="outline-header">
                            <div class="outline-title" id="outline-title">Ready</div>
                            <div class="outline-stats" id="outline-stats">0/0 steps</div>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="outline-progress-fill"></div>
                        </div>
                    </div>
                    <div class="outline-steps" id="outline-steps">
                        <div class="outline-step">
                            <div class="step-icon">⏳</div>
                            <div class="step-text">Waiting for query...</div>
                        </div>
                    </div>
                    <div class="current-activity" id="current-activity" style="display: none;">
                        <div class="activity-header">
                            <div class="activity-title" id="activity-title">Current Activity</div>
                            <div class="activity-duration" id="activity-duration">0s</div>
                        </div>
                        <div class="activity-description" id="activity-description"></div>
                    </div>
                </div>
            </div>

            <!-- Activity & Meta-Tools Panel -->
            <div class="panel activity-panel">
                <div class="panel-header">⚙️ Meta-Tool Activity</div>
                <div class="panel-content">
                    <div class="meta-tools-list" id="meta-tools-list">
                        <div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 20px;">
                            No activity yet
                        </div>
                    </div>
                </div>
            </div>

            <!-- System Status Panel -->
            <div class="panel system-panel">
                <div class="panel-header">🔧 System Status</div>
                <div class="panel-content">
                    <div class="current-node" id="current-node">
                        <div class="node-name" id="node-name">System</div>
                        <div class="node-operation" id="node-operation">Idle</div>
                    </div>
                    <div class="system-grid" id="system-grid">
                        <div class="system-key">Status</div>
                        <div class="system-value">Ready</div>
                        <div class="system-key">Runtime</div>
                        <div class="system-value">0s</div>
                        <div class="system-key">Events</div>
                        <div class="system-value">0</div>
                        <div class="system-key">Errors</div>
                        <div class="system-value">0</div>
                    </div>
                </div>
            </div>

            <!-- Agent Graph Panel -->
            <div class="panel graph-panel">
                <div class="panel-header">🌐 Agent Flow</div>
                <div class="panel-content">
                    <div class="agent-graph" id="agent-graph">
                        <div class="graph-node">LLMReasonerNode</div>
                        <div class="graph-arrow">↓</div>
                        <div class="graph-node">Ready</div>
                    </div>
                </div>
            </div>
        </div>

        <script unSave="true">
            __SERVER_CONFIG__
            class ProductionAgentUI {
                constructor() {
                    this.ws = null;
                    this.isProcessing = false;
                    this.sessionId = 'ui_session_' + Math.random().toString(36).substr(2, 9);
                    this.startTime = null;
                    this.reconnectAttempts = 0;
                    this.maxReconnectAttempts = 10;
                    this.reconnectDelay = 1000;
                    this.useWebSocket = true;
                    this.fallbackMode = false;

                    // Progress tracking
                    this.currentOutline = null;
                    this.currentActivity = null;
                    this.metaTools = new Map();
                    this.systemStatus = {};
                    this.agentGraph = [];
                    this.progressEvents = [];

                    this.elements = {
                        statusDot: document.getElementById('status-dot'),
                        statusText: document.getElementById('status-text'),
                        chatMessages: document.getElementById('chat-messages'),
                        chatInput: document.getElementById('chat-input'),
                        sendButton: document.getElementById('send-button'),
                        progressIndicator: document.getElementById('progress-indicator'),
                        progressText: document.getElementById('progress-text'),

                        // Outline elements
                        outlineTitle: document.getElementById('outline-title'),
                        outlineStats: document.getElementById('outline-stats'),
                        outlineProgressFill: document.getElementById('outline-progress-fill'),
                        outlineSteps: document.getElementById('outline-steps'),
                        currentActivity: document.getElementById('current-activity'),
                        activityTitle: document.getElementById('activity-title'),
                        activityDuration: document.getElementById('activity-duration'),
                        activityDescription: document.getElementById('activity-description'),

                        // Meta-tools elements
                        metaToolsList: document.getElementById('meta-tools-list'),

                        // System elements
                        currentNode: document.getElementById('current-node'),
                        nodeName: document.getElementById('node-name'),
                        nodeOperation: document.getElementById('node-operation'),
                        systemGrid: document.getElementById('system-grid'),

                        // Graph elements
                        agentGraph: document.getElementById('agent-graph')
                    };
                    this.init();
                }


                init() {

                    this.configureAPIPaths();
                    this.setupEventListeners();
                    this.detectServerMode();
                    this.startStatusUpdates();
                }

                configureAPIPaths() {
                    const serverType = window.SERVER_CONFIG?.server_type || 'standalone';

                    if (serverType === 'builtin') {
                        this.apiPaths = {
                            status: '/api/agent_ui/status',
                            run: '/api/agent_ui/run_agent',
                            reset: '/api/agent_ui/reset_context'
                        };
                        this.useWebSocket = true;
                    } else {
                        this.apiPaths = {
                            status: '/api/status',
                            run: '/api/run',
                            reset: '/api/reset'
                        };
                        this.useWebSocket = false;
                        this.enableFallbackMode();
                    }
                }

                setupEventListeners() {
                    this.elements.sendButton.addEventListener('click', () => this.sendMessage());
                    this.elements.chatInput.addEventListener('keypress', (e) => {
                        if (e.key === 'Enter' && !this.isProcessing) {
                            this.sendMessage();
                        }
                    });

                    // Handle page visibility for reconnection
                    document.addEventListener('visibilitychange', () => {
                        if (!document.hidden && (!this.ws || this.ws.readyState === WebSocket.CLOSED)) {
                            this.connectWebSocket();
                        }
                    });
                }

                detectServerMode() {
                    // Use configured paths instead of hardcoded ones
                    fetch(this.apiPaths.status)
                        .then(response => response.json())
                        .then(data => {
                            this.addLogEntry(`Server detected: ${data.server_type || 'standalone'}`, 'info');
                            if (data.server_type === 'builtin' && this.useWebSocket) {
                                this.connectWebSocket();
                            }
                        })
                        .catch(() => {
                            this.addLogEntry('Server detection failed, using fallback mode', 'error');
                            this.enableFallbackMode();
                        });
                }

                connectWebSocket() {
                    if (!this.useWebSocket) return;

                    try {
                        // Construct WebSocket URL more robustly
                        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                        const wsUrl = `${protocol}//${window.location.host}/ws/agent_ui/connect`;

                        this.addLogEntry(`Attempting WebSocket connection to: ${wsUrl}`);
                        this.ws = new WebSocket(wsUrl);

                        this.ws.onopen = () => {
                            this.reconnectAttempts = 0;
                            this.fallbackMode = false;
                            this.setStatus('connected', 'Connected');
                            this.addLogEntry('WebSocket connected successfully', 'success');
                            this.removeFallbackIndicators();
                        };

                        this.ws.onmessage = (event) => {
                            try {
                                const message = JSON.parse(event.data);
                                this.handleWebSocketMessage(message);
                            } catch (error) {
                                this.addLogEntry(`WebSocket message parse error: ${error.message}`, 'error');
                            }
                        };

                        this.ws.onclose = (event) => {
                            this.setStatus('disconnected', 'Disconnected');
                            this.addLogEntry(`WebSocket disconnected (code: ${event.code})`, 'error');
                            this.scheduleReconnection();
                        };

                        this.ws.onerror = (error) => {
                            this.setStatus('error', 'Connection Error');
                            this.addLogEntry('WebSocket connection error', 'error');
                            this.scheduleReconnection();
                        };

                    } catch (error) {
                        this.addLogEntry(`WebSocket setup error: ${error.message}`, 'error');
                        this.enableFallbackMode();
                    }
                }

                scheduleReconnection() {
                    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                        this.addLogEntry('Max reconnection attempts reached, enabling fallback mode', 'error');
                        this.enableFallbackMode();
                        return;
                    }

                    this.reconnectAttempts++;
                    const delay = Math.min(this.reconnectDelay * this.reconnectAttempts, 30000);

                    this.setStatus('error', `Reconnecting in ${delay/1000}s (attempt ${this.reconnectAttempts})`);

                    setTimeout(() => {
                        if (!this.ws || this.ws.readyState === WebSocket.CLOSED) {
                            this.connectWebSocket();
                        }
                    }, delay);
                }

                enableFallbackMode() {
                    this.fallbackMode = true;
                    this.useWebSocket = false;
                    this.setStatus('disconnected', 'Fallback Mode (API Only)');
                    this.showFallbackIndicator();
                    this.addLogEntry('WebSocket unavailable - using API fallback mode', 'info');
                }

                showFallbackIndicator() {
                    const indicator = document.createElement('div');
                    indicator.className = 'fallback-mode';
                    indicator.textContent = 'Using API fallback mode - limited real-time updates';
                    indicator.id = 'fallback-indicator';
                    document.body.appendChild(indicator);
                }

                removeFallbackIndicators() {
                    const indicator = document.getElementById('fallback-indicator');
                    if (indicator) {
                        indicator.remove();
                    }
                }

                handleWebSocketMessage(message) {
                    try {
                        switch (message.event) {
                            case 'agent_connected':
                                this.addLogEntry('Agent ready for interaction', 'success');
                                this.updateSystemStatus({
                                    status: 'Connected',
                                    capabilities: message.data.capabilities
                                });
                                break;

                            case 'processing_start':
                                this.setProcessing(true);
                                this.startTime = Date.now();
                                this.addLogEntry(`Processing: ${message.data.query}`, 'progress');
                                this.resetProgressTracking();
                                break;

                            case 'progress_update':
                                this.handleProgressUpdate(message.data);
                                break;

                            case 'outline_update':
                                this.handleOutlineUpdate(message.data);
                                break;

                            case 'meta_tool_update':
                                this.handleMetaToolUpdate(message.data);
                                break;

                            case 'activity_update':
                                this.handleActivityUpdate(message.data);
                                break;

                            case 'system_update':
                                this.handleSystemUpdate(message.data);
                                break;

                            case 'graph_update':
                                this.handleGraphUpdate(message.data);
                                break;

                            case 'chat_response':
                                this.addMessage('agent', message.data.response);
                                this.setProcessing(false);
                                this.addLogEntry('Response completed', 'success');
                                this.showFinalSummary(message.data);
                                break;

                            case 'error':
                                this.addMessage('agent', `Error: ${message.data.error}`);
                                this.setProcessing(false);
                                this.addLogEntry(`Error: ${message.data.error}`, 'error');
                                break;

                            default:
                                console.log('Unhandled WebSocket message:', message);
                        }
                    } catch (error) {
                        this.addLogEntry(`Message handling error: ${error.message}`, 'error');
                    }
                }

                handleProgressUpdate(data) {
                    this.progressEvents.push(data);

                    const progressText = `${data.event_type}: ${data.status || 'processing'}`;
                    this.elements.progressText.textContent = progressText;

                    // Update based on event type
                    if (data.event_type === 'reasoning_loop') {
                        this.addLogEntry(`🧠 Reasoning loop #${data.loop_number || '?'}`, 'reasoning');
                        this.updateCurrentActivity({
                            title: 'Reasoning',
                            description: data.current_focus || 'Deep thinking in progress',
                            duration: data.time_in_activity || 0
                        });
                    } else if (data.event_type === 'meta_tool_call') {
                        this.addLogEntry(`⚙️ Meta-tool: ${data.meta_tool_name || 'unknown'}`, 'meta-tool');
                    } else {
                        this.addLogEntry(`Progress - ${progressText}`, 'progress');
                    }

                    // Update system status
                    this.updateSystemStatus({
                        current_node: data.node_name,
                        current_operation: data.event_type,
                        runtime: this.getRuntime(),
                        events: this.progressEvents.length
                    });
                }

                handleOutlineUpdate(data) {
                    this.currentOutline = data;

                    if (data.outline_created && data.steps) {
                        this.elements.outlineTitle.textContent = 'Execution Outline';

                        const completedCount = (data.completed_steps || []).length;
                        const totalCount = data.total_steps || data.steps.length;

                        this.elements.outlineStats.textContent = `${completedCount}/${totalCount} steps`;

                        // Update progress bar
                        const progress = totalCount > 0 ? (completedCount / totalCount) * 100 : 0;
                        this.elements.outlineProgressFill.style.width = `${progress}%`;

                        // Update steps
                        this.updateOutlineSteps(data.steps, data.current_step, data.completed_steps || []);

                        this.addLogEntry(`Outline progress: ${completedCount}/${totalCount} steps completed`, 'outline');
                    }
                }

                updateOutlineSteps(steps, currentStep, completedSteps) {
                    this.elements.outlineSteps.innerHTML = '';

                    steps.forEach((step, index) => {
                        const stepEl = document.createElement('div');
                        stepEl.className = 'outline-step';

                        const stepId = step.id || (index + 1);
                        let icon = '⏳';

                        if (completedSteps.includes(stepId)) {
                            stepEl.classList.add('completed');
                            icon = '✅';
                        } else if (stepId === currentStep) {
                            stepEl.classList.add('active');
                            icon = '🔄';
                        }

                        stepEl.innerHTML = `
                            <div class="step-icon">${icon}</div>
                            <div class="step-text">${step.description || `Step ${stepId}`}</div>
                            <div class="step-method">${step.method || 'unknown'}</div>
                        `;

                        this.elements.outlineSteps.appendChild(stepEl);
                    });
                }

                handleMetaToolUpdate(data) {
                    const toolId = `${data.meta_tool_name}_${Date.now()}`;
                    const toolData = {
                        name: data.meta_tool_name,
                        status: data.status || 'running',
                        timestamp: Date.now(),
                        phase: data.execution_phase,
                        data: data
                    };

                    this.metaTools.set(toolId, toolData);
                    this.updateMetaToolsList();

                    // Add to log with appropriate icon
                    const statusIcon = data.status === 'completed' ? '✅' :
                                     data.status === 'error' ? '❌' : '⚙️';
                    this.addLogEntry(`${statusIcon} ${data.meta_tool_name}: ${data.status || 'running'}`, 'meta-tool');
                }

                updateMetaToolsList() {
                    this.elements.metaToolsList.innerHTML = '';

                    if (this.metaTools.size === 0) {
                        this.elements.metaToolsList.innerHTML = `
                            <div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 20px;">
                                No meta-tool activity yet
                            </div>
                        `;
                        return;
                    }

                    // Show recent meta-tools (last 8)
                    const recentTools = Array.from(this.metaTools.values())
                        .sort((a, b) => b.timestamp - a.timestamp)
                        .slice(0, 8);

                    recentTools.forEach(tool => {
                        const toolEl = document.createElement('div');
                        toolEl.className = 'meta-tool';

                        const icons = {
                            internal_reasoning: '🧠',
                            delegate_to_llm_tool_node: '🎯',
                            create_and_execute_plan: '📋',
                            manage_internal_task_stack: '📚',
                            advance_outline_step: '➡️',
                            write_to_variables: '💾',
                            read_from_variables: '📖',
                            direct_response: '✨'
                        };

                        const icon = icons[tool.name] || '⚙️';
                        const displayName = tool.name.replace(/_/g, ' ');
                        const age = Math.floor((Date.now() - tool.timestamp) / 1000);

                        toolEl.innerHTML = `
                            <div class="tool-icon">${icon}</div>
                            <div class="tool-name">${displayName} (${age}s ago)</div>
                            <div class="tool-status ${tool.status}">${tool.status}</div>
                        `;

                        this.elements.metaToolsList.appendChild(toolEl);
                    });
                }

                handleActivityUpdate(data) {
                    this.currentActivity = data;
                    this.updateCurrentActivity(data);
                }

                updateCurrentActivity(data) {
                    if (data.primary_activity && data.primary_activity !== 'Unknown') {
                        this.elements.currentActivity.style.display = 'block';
                        this.elements.activityTitle.textContent = data.primary_activity || data.title;

                        const duration = data.time_in_current_activity || data.duration || 0;
                        if (duration > 0) {
                            this.elements.activityDuration.textContent = this.formatDuration(duration);
                        }

                        this.elements.activityDescription.textContent =
                            data.detailed_description || data.description || '';
                    } else {
                        this.elements.currentActivity.style.display = 'none';
                    }
                }

                handleSystemUpdate(data) {
                    this.systemStatus = { ...this.systemStatus, ...data };
                    this.updateSystemStatus(data);
                }

                updateSystemStatus(data) {
                    // Update current node
                    if (data.current_node) {
                        this.elements.nodeName.textContent = data.current_node;
                        this.elements.nodeOperation.textContent = data.current_operation || 'Processing';
                    }

                    // Update system grid
                    const gridData = [
                        ['Status', data.status || this.systemStatus.status || 'Running'],
                        ['Runtime', this.formatDuration(data.runtime || this.getRuntime())],
                        ['Events', data.events || this.progressEvents.length],
                        ['Errors', data.error_count || this.systemStatus.error_count || 0],
                        ['Node', data.current_node || this.systemStatus.current_node || 'Unknown']
                    ];

                    if (data.total_cost !== undefined) {
                        gridData.push(['Cost', `$${data.total_cost.toFixed(4)}`]);
                    }

                    if (data.total_tokens !== undefined) {
                        gridData.push(['Tokens', data.total_tokens.toLocaleString()]);
                    }

                    this.elements.systemGrid.innerHTML = '';
                    gridData.forEach(([key, value]) => {
                        this.elements.systemGrid.innerHTML += `
                            <div class="system-key">${key}</div>
                            <div class="system-value">${value}</div>
                        `;
                    });
                }

                handleGraphUpdate(data) {
                    this.agentGraph = data.nodes || [];
                    this.updateAgentGraph();
                }

                updateAgentGraph() {
                    this.elements.agentGraph.innerHTML = '';

                    if (this.agentGraph.length === 0) {
                        const currentNode = this.systemStatus.current_node || 'LLMReasonerNode';
                        this.elements.agentGraph.innerHTML = `
                            <div class="graph-node active">${currentNode}</div>
                            <div class="graph-arrow">↓</div>
                            <div class="graph-node">Processing</div>
                        `;
                        return;
                    }

                    this.agentGraph.forEach((node, index) => {
                        const nodeEl = document.createElement('div');
                        nodeEl.className = 'graph-node';

                        if (node.active) nodeEl.classList.add('active');
                        if (node.completed) nodeEl.classList.add('completed');

                        nodeEl.textContent = node.name || `Node ${index + 1}`;
                        this.elements.agentGraph.appendChild(nodeEl);

                        if (index < this.agentGraph.length - 1) {
                            const arrow = document.createElement('div');
                            arrow.className = 'graph-arrow';
                            arrow.textContent = '↓';
                            this.elements.agentGraph.appendChild(arrow);
                        }
                    });
                }

                async sendMessage() {
                    const message = this.elements.chatInput.value.trim();
                    if (!message || this.isProcessing) return;

                    this.addMessage('user', message);
                    this.elements.chatInput.value = '';

                    if (this.useWebSocket && this.ws && this.ws.readyState === WebSocket.OPEN) {
                        // Send via WebSocket
                        this.ws.send(JSON.stringify({
                            event: 'chat_message',
                            data: {
                                message: message,
                                session_id: this.sessionId
                            }
                        }));
                    } else {
                        // Fallback to API
                        await this.sendMessageViaAPI(message);
                    }
                }

                async sendMessageViaAPI(message) {
                    this.setProcessing(true);
                    this.startTime = Date.now();
                    this.resetProgressTracking();

                    try {
                        const response = await fetch(this.apiPaths.run, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                query: message,
                                session_id: this.sessionId,
                                include_progress: true
                            })
                        });

                        const result = await response.json();

                        if (result.success) {
                            this.addMessage('agent', result.result);
                            this.addLogEntry(`Request completed via API`, 'success');

                            // Process progress events if available
                            if (result.progress_events) {
                                this.processAPIProgressEvents(result.progress_events);
                            }

                            // Process enhanced progress if available
                            if (result.enhanced_progress) {
                                this.processEnhancedProgress(result.enhanced_progress);
                            }
                        } else {
                            this.addMessage('agent', `Error: ${result.error}`);
                            this.addLogEntry(`API request failed: ${result.error}`, 'error');
                        }

                    } catch (error) {
                        this.addMessage('agent', `Network error: ${error.message}`);
                        this.addLogEntry(`Network error: ${error.message}`, 'error');
                    } finally {
                        this.setProcessing(false);
                    }
                }

                processAPIProgressEvents(events) {
                    events.forEach(event => {
                        this.handleProgressUpdate(event);
                    });
                }

                processEnhancedProgress(progress) {
                    if (progress.outline) {
                        this.handleOutlineUpdate(progress.outline);
                    }
                    if (progress.activity) {
                        this.handleActivityUpdate(progress.activity);
                    }
                    if (progress.system) {
                        this.handleSystemUpdate(progress.system);
                    }
                    if (progress.graph) {
                        this.handleGraphUpdate(progress.graph);
                    }
                }

                resetProgressTracking() {
                    this.progressEvents = [];
                    this.metaTools.clear();
                    this.updateSystemStatus({ status: 'Processing', events: 0 });
                }

                showFinalSummary(data) {
                    if (data.final_summary) {
                        const summary = data.final_summary;
                        this.addLogEntry(`Final Summary - Outline: ${summary.outline_completed ? 'Complete' : 'Partial'}, Meta-tools: ${summary.total_meta_tools}, Nodes: ${summary.total_nodes}`, 'success');
                    }
                }

                addMessage(sender, content) {
                    const messageEl = document.createElement('div');
                    messageEl.classList.add('message', sender);

                    const avatarEl = document.createElement('div');
                    avatarEl.classList.add('message-avatar');
                    avatarEl.textContent = sender === 'user' ? 'You' : 'AI';

                    const contentEl = document.createElement('div');
                    contentEl.classList.add('message-content');

                    if (sender === 'agent' && window.marked) {
                        try {
                            contentEl.innerHTML = marked.parse(content);
                        } catch (error) {
                            contentEl.textContent = content;
                        }
                    } else {
                        contentEl.textContent = content;
                    }

                    messageEl.appendChild(avatarEl);
                    messageEl.appendChild(contentEl);

                    this.elements.chatMessages.appendChild(messageEl);
                    this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
                }

                addLogEntry(message, type = 'info') {
                    // For debugging - could show in a log panel
                    const timestamp = new Date().toLocaleTimeString();
                    console.log(`[${timestamp}] [${type.toUpperCase()}] ${message}`);
                }

                setStatus(status, text) {
                    this.elements.statusDot.className = `status-dot ${status}`;
                    this.elements.statusText.textContent = text;
                }

                setProcessing(processing) {
                    this.isProcessing = processing;
                    this.elements.sendButton.disabled = processing;
                    this.elements.chatInput.disabled = processing;

                    if (processing) {
                        this.elements.progressIndicator.classList.add('active');
                        this.setStatus('processing', 'Processing');
                    } else {
                        this.elements.progressIndicator.classList.remove('active');
                        this.setStatus(this.ws && this.ws.readyState === WebSocket.OPEN ? 'connected' : 'disconnected',
                                      this.ws && this.ws.readyState === WebSocket.OPEN ? 'Connected' : 'Disconnected');
                        this.startTime = null;
                    }
                }

                formatDuration(seconds) {
                    if (typeof seconds !== 'number') return '0s';
                    if (seconds < 60) return `${seconds.toFixed(1)}s`;
                    if (seconds < 3600) return `${Math.floor(seconds/60)}m${Math.floor(seconds%60)}s`;
                    return `${Math.floor(seconds/3600)}h${Math.floor((seconds%3600)/60)}m`;
                }

                getRuntime() {
                    return this.startTime ? (Date.now() - this.startTime) / 1000 : 0;
                }

                startStatusUpdates() {
                    setInterval(() => {
                        if (this.isProcessing) {
                            this.updateSystemStatus({ runtime: this.getRuntime() });
                        }
                    }, 1000);
                }
            }

            // Initialize the production UI
            if (!window.TB) {

                document.addEventListener('DOMContentLoaded', () => {
                    window.agentUI = new ProductionAgentUI();
                });
            } else {
                TB.once(() => {
                    window.agentUI = new ProductionAgentUI();
                });
            }
        </script>
    </body>
    </html>"""

        return (html_template.
                replace("{agent_name}", agent_info.get('public_name', 'Agent Interface')).
                replace("{agent_description}", agent_info.get('description', '')).
                replace("__SERVER_CONFIG__", js_config)
                )

    async def _handle_chat_message_with_progress_integration(self, agent_id: str, agent, conn_id: str, data: dict):
        """Enhanced chat message handler with ProgressiveTreePrinter integration."""
        query = data.get('message', '')
        session_id = data.get('session_id', f"ui_session_{conn_id}")

        if not query:
            return

        # Create ProgressiveTreePrinter for real-time UI updates
        from toolboxv2.mods.isaa.extras.terminal_progress import (
            ProgressiveTreePrinter,
            VerbosityMode,
        )
        progress_printer = ProgressiveTreePrinter(
            mode=VerbosityMode.STANDARD,
            use_rich=False,
            auto_refresh=False
        )

        # Enhanced progress callback that extracts all UI data
        async def comprehensive_progress_callback(event):
            try:
                # Add event to progress printer for processing
                progress_printer.tree_builder.add_event(event)

                # Get comprehensive summary from the printer
                summary = progress_printer.tree_builder.get_execution_summary()

                # Extract outline information
                outline_info = progress_printer._get_current_outline_info()

                # Extract current activity
                activity_info = progress_printer._get_detailed_current_activity()

                # Extract tool usage
                tool_usage = progress_printer._get_tool_usage_summary()

                # Extract task progress
                task_progress = progress_printer._get_task_executor_progress()

                # Send basic progress update
                await self._broadcast_to_agent_ui(agent_id, {
                    'event': 'progress_update',
                    'data': {
                        'event_type': event.event_type,
                        'status': getattr(event, 'status', 'processing').value if hasattr(event, 'status') and event.status else 'unknown',
                        'node_name': getattr(event, 'node_name', 'Unknown'),
                        'timestamp': event.timestamp,
                        'loop_number': getattr(event.metadata, {}).get('reasoning_loop', 0),
                        'meta_tool_name': getattr(event.metadata, {}).get('meta_tool_name'),
                        'current_focus': getattr(event.metadata, {}).get('current_focus', ''),
                        'time_in_activity': activity_info.get('time_in_current_activity', 0)
                    }
                })

                # Send outline updates
                if outline_info.get('outline_created'):
                    await self._broadcast_to_agent_ui(agent_id, {
                        'event': 'outline_update',
                        'data': outline_info
                    })

                # Send meta-tool updates
                if event.metadata and event.metadata.get('meta_tool_name'):
                    await self._broadcast_to_agent_ui(agent_id, {
                        'event': 'meta_tool_update',
                        'data': {
                            'meta_tool_name': event.metadata['meta_tool_name'],
                            'status': 'completed' if event.success else (
                                'error' if event.success is False else 'running'),
                            'execution_phase': event.metadata.get('execution_phase', 'unknown'),
                            'reasoning_loop': event.metadata.get('reasoning_loop', 0),
                            'timestamp': event.timestamp
                        }
                    })

                # Send activity updates
                if activity_info['primary_activity'] != 'Unknown':
                    await self._broadcast_to_agent_ui(agent_id, {
                        'event': 'activity_update',
                        'data': activity_info
                    })

                # Send system updates
                await self._broadcast_to_agent_ui(agent_id, {
                    'event': 'system_update',
                    'data': {
                        'current_node': summary['execution_flow']['current_node'],
                        'current_operation': activity_info.get('primary_activity', 'Processing'),
                        'status': 'Processing',
                        'runtime': summary['timing']['elapsed'],
                        'total_events': summary['performance_metrics']['total_events'],
                        'error_count': summary['performance_metrics']['error_count'],
                        'total_cost': summary['performance_metrics']['total_cost'],
                        'total_tokens': summary['performance_metrics']['total_tokens'],
                        'completed_nodes': summary['session_info']['completed_nodes'],
                        'total_nodes': summary['session_info']['total_nodes'],
                        'tool_usage': {
                            'tools_used': list(tool_usage.get('tools_used', set())),
                            'tools_active': list(tool_usage.get('tools_active', set())),
                            'current_tool_operation': tool_usage.get('current_tool_operation')
                        }
                    }
                })

                # Send graph updates
                flow_nodes = []
                for node_name in summary['execution_flow']['flow']:
                    if node_name in progress_printer.tree_builder.nodes:
                        node = progress_printer.tree_builder.nodes[node_name]
                        flow_nodes.append({
                            'name': node_name,
                            'active': node_name in summary['execution_flow']['active_nodes'],
                            'completed': (node.status.value == 'completed') if node.status else False,
                            'status': node.status.value if node.status else 'unknown'
                        })

                if flow_nodes:
                    await self._broadcast_to_agent_ui(agent_id, {
                        'event': 'graph_update',
                        'data': {'nodes': flow_nodes}
                    })

            except Exception as e:
                self.app.print(f"Comprehensive progress callback error: {e}")

        # Set progress callback
        original_callback = getattr(agent, 'progress_callback', None)

        try:
            if hasattr(agent, 'set_progress_callback'):
                agent.set_progress_callback(comprehensive_progress_callback)
            elif hasattr(agent, 'progress_callback'):
                agent.progress_callback = comprehensive_progress_callback

            # Send processing start notification
            await self._broadcast_to_agent_ui(agent_id, {
                'event': 'processing_start',
                'data': {'query': query, 'session_id': session_id}
            })

            # Execute agent
            result = await agent.a_run(query=query, session_id=session_id)

            # Get final summary
            final_summary = progress_printer.tree_builder.get_execution_summary()

            # Extract outline information
            outline_info = progress_printer._get_current_outline_info()

            # Initialize outline_info if empty
            if not outline_info or not outline_info.get('steps'):
                outline_info = {
                    'steps': [],
                    'current_step': 1,
                    'completed_steps': [],
                    'total_steps': 0,
                    'step_descriptions': {},
                    'current_step_progress': "",
                    'outline_raw_data': None,
                    'outline_created': False,
                    'actual_step_completions': []
                }

            # Try to infer outline from execution pattern if not found
            if not outline_info.get('outline_created'):
                outline_info = progress_printer._infer_outline_from_execution_pattern(outline_info)

            # Send final result with summary
            await self._broadcast_to_agent_ui(agent_id, {
                'event': 'chat_response',
                'data': {
                    'response': result,
                    'query': query,
                    'session_id': session_id,
                    'completed_at': asyncio.get_event_loop().time(),
                    'final_summary': {
                        'outline_completed': len(outline_info.get('completed_steps', [])) == outline_info.get(
                            'total_steps', 0),
                        'total_meta_tools': len([e for e in progress_printer.tree_builder.nodes.values()
                                                 for event in e.llm_calls + e.sub_events
                                                 if event.metadata and event.metadata.get('meta_tool_name')]),
                        'total_nodes': final_summary['session_info']['total_nodes'],
                        'execution_time': final_summary['timing']['elapsed'],
                        'total_cost': final_summary['performance_metrics']['total_cost']
                    }
                }
            })

        except Exception as e:
            await self._broadcast_to_agent_ui(agent_id, {
                'event': 'error',
                'data': {'error': str(e), 'query': query}
            })
        finally:
            # Restore original callback
            if hasattr(agent, 'set_progress_callback'):
                agent.set_progress_callback(original_callback)
            elif hasattr(agent, 'progress_callback'):
                agent.progress_callback = original_callback

    # Replace the existing method
    async def _handle_chat_message(self, agent_id: str, agent, conn_id: str, data: dict):
        """Delegate to enhanced handler."""
        await self._handle_chat_message_with_progress_integration(agent_id, agent, conn_id, data)

    # Unified publish and host method
    # toolboxv2/mods/isaa/Tools.py

    async def publish_and_host_agent(
        self,
        agent,
        public_name: str,
        registry_server: str = "ws://localhost:8080/ws/registry/connect",
        description: str | None = None,
        access_level: str = "public"
    ) -> dict[str, Any]:
        """FIXED: Mit Debug-Ausgaben für Troubleshooting."""

        if hasattr(agent, 'name') and not hasattr(agent, 'amd') and hasattr(agent, 'a_run'):
            agent.amd = lambda :None
            agent.amd.name = agent.name

        try:
            # Registry Client initialisieren
            from toolboxv2.mods.registry.client import get_registry_client
            registry_client = get_registry_client(self.app)

            self.app.print(f"Connecting to registry server: {registry_server}")
            await registry_client.connect(registry_server)

            # Progress Callback für Live-Updates einrichten
            callback_success = await self.setup_live_progress_callback(agent, registry_client, f"agent_{agent.amd.name}")
            if not callback_success:
                self.app.print("Warning: Progress callback setup failed")
            else:
                self.app.print("✅ Progress callback setup successful")

            # Agent beim Registry registrieren
            self.app.print(f"Registering agent: {public_name}")
            registration_info = await registry_client.register(
                agent_instance=agent,
                public_name=public_name,
                description=description or f"Agent: {public_name}"
            )

            if not registration_info:
                return {"error": "Registration failed", "success": False}

            self.app.print(f"✅ Agent registration successful: {registration_info.public_agent_id}")

            result = {
                "success": True,
                "agent_name": public_name,
                "public_agent_id": registration_info.public_agent_id,
                "public_api_key": registration_info.public_api_key,
                "public_url": registration_info.public_url,
                "registry_server": registry_server,
                "access_level": access_level,
                "ui_url": registration_info.public_url.replace("/api/registry/run", "/api/registry/ui"),
                "websocket_url": registry_server.replace("/connect", "/ui_connect"),
                "status": "registered"
            }

            return result

        except Exception as e:
            self.app.print(f"Failed to publish agent: {e}")
            return {"error": str(e), "success": False}

    # toolboxv2/mods/isaa/Tools.py

    async def setup_live_progress_callback(self, agent, registry_client, agent_id: str = None):
        """Enhanced setup for live progress callback with proper error handling."""

        if not registry_client:
            self.app.print("Warning: No registry client provided for progress updates")
            return False

        if not registry_client.is_connected:
            self.app.print("Warning: Registry client is not connected")
            return False

        progress_tracker = EnhancedProgressTracker()

        # Generate agent ID if not provided
        if not agent_id:
            agent_id = getattr(agent, 'name', f'agent_{id(agent)}')

        async def enhanced_live_progress_callback(event: ProgressEvent):
            """Enhanced progress callback with comprehensive data extraction."""
            try:
                # Validate event
                if not event:
                    self.app.print("Warning: Received null progress event")
                    return

                # Debug output for local development
                event_type = getattr(event, 'event_type', 'unknown')
                status = getattr(event, 'status', 'unknown')
                agent_name = getattr(event, 'agent_name', 'Unknown Agent')

                self.app.print(f"📊 Progress Event: {event_type} | {status} | {agent_name}")

                # Extract comprehensive progress data
                progress_data = progress_tracker.extract_progress_data(event)

                # Prepare enhanced progress message
                ui_progress_data = {
                    "agent_id": agent_id,
                    "event_type": event_type,
                    "status": status.value if hasattr(status, 'value') else str(status),
                    "timestamp": getattr(event, 'timestamp', asyncio.get_event_loop().time()),
                    "agent_name": agent_name,
                    "node_name": getattr(event, 'node_name', 'Unknown'),
                    "session_id": getattr(event, 'session_id', None),

                    # Core event metadata
                    "metadata": {
                        **getattr(event, 'metadata', {}),
                        "event_id": getattr(event, 'event_id', f"evt_{asyncio.get_event_loop().time()}"),
                        "sequence_number": getattr(event, 'sequence_number', 0),
                        "parent_event_id": getattr(event, 'parent_event_id', None)
                    },

                    # Detailed progress data for UI panels
                    "progress_data": progress_data,

                    # UI-specific flags for selective updates
                    "ui_flags": {
                        "should_update_outline": bool(progress_data.get('outline')),
                        "should_update_activity": bool(progress_data.get('activity')),
                        "should_update_meta_tools": bool(progress_data.get('meta_tool')),
                        "should_update_system": bool(progress_data.get('system')),
                        "should_update_graph": bool(progress_data.get('graph')),
                        "is_error": event_type.lower() in ['error', 'exception', 'failed'],
                        "is_completion": event_type.lower() in ['complete', 'finished', 'success'],
                        "requires_user_input": getattr(event, 'requires_user_input', False)
                    },

                    # Performance metrics
                    "performance": {
                        "execution_time": getattr(event, 'execution_time', None),
                        "memory_delta": getattr(event, 'memory_delta', None),
                        "tokens_used": getattr(event, 'tokens_used', None),
                        "api_calls_made": getattr(event, 'api_calls_made', None)
                    }
                }

                # Send live update to registry server
                await registry_client.send_ui_progress(ui_progress_data)

                # Also send agent status update if this is a significant event
                if event_type in ['started', 'completed', 'error', 'paused', 'resumed']:
                    agent_status = 'processing'
                    if event_type == 'completed':
                        agent_status = 'idle'
                    elif event_type == 'error':
                        agent_status = 'error'
                    elif event_type == 'paused':
                        agent_status = 'paused'

                    await registry_client.send_agent_status(
                        agent_id=agent_id,
                        status=agent_status,
                        details={
                            "last_event": event_type,
                            "last_update": ui_progress_data["timestamp"],
                            "current_node": progress_data.get('graph', {}).get('current_node', 'Unknown')
                        }
                    )

                # Log successful progress update
                self.app.print(f"✅ Sent progress update: {event_type} -> Registry Server")

            except Exception as e:
                self.app.print(f"❌ Progress callback error: {e}")
                # Send error notification to UI
                try:
                    await registry_client.send_ui_progress({
                        "agent_id": agent_id,
                        "event_type": "progress_callback_error",
                        "status": "error",
                        "timestamp": asyncio.get_event_loop().time(),
                        "agent_name": getattr(agent, 'name', 'Unknown'),
                        "metadata": {"error": str(e)},
                        "ui_flags": {"is_error": True}
                    })
                except Exception as nested_error:
                    self.app.print(f"Failed to send error notification: {nested_error}")

        # Set up progress callback with enhanced error handling
        callback_set = False

        if hasattr(agent, 'set_progress_callback'):
            try:
                self.app.print(f"🔧 Setting progress callback via set_progress_callback for agent: {agent_id}")
                agent.set_progress_callback(enhanced_live_progress_callback)
                callback_set = True
            except Exception as e:
                self.app.print(f"Failed to set progress callback via set_progress_callback: {e}")

        if not callback_set and hasattr(agent, 'progress_callback'):
            try:
                self.app.print(f"🔧 Setting progress callback via direct assignment for agent: {agent_id}")
                agent.progress_callback = enhanced_live_progress_callback
                callback_set = True
            except Exception as e:
                self.app.print(f"Failed to set progress callback via direct assignment: {e}")

        if not callback_set:
            self.app.print(f"⚠️ Warning: Agent {agent_id} doesn't support progress callbacks")
            return False

        # Send initial agent status
        try:
            await registry_client.send_agent_status(
                agent_id=agent_id,
                status='online',
                details={
                    "progress_callback_enabled": True,
                    "callback_setup_time": asyncio.get_event_loop().time(),
                    "agent_type": type(agent).__name__
                }
            )
            self.app.print(f"✅ Progress callback successfully set up for agent: {agent_id}")
        except Exception as e:
            self.app.print(f"Failed to send initial agent status: {e}")

        return True


    async def _setup_builtin_server_hosting(self, agent_id: str, agent, host, port) -> dict[str, str]:
        """Setup agent hosting using toolbox built-in server with enhanced WebSocket support."""

        # Register WebSocket handlers for this agent
        @self.app.tb(mod_name="agent_ui", websocket_handler="connect")
        def register_agent_ws_handlers(_):
            return {
                "on_connect": self._create_agent_ws_connect_handler(agent_id),
                "on_message": self._create_agent_ws_message_handler(agent_id, agent),
                "on_disconnect": self._create_agent_ws_disconnect_handler(agent_id),
            }

        # Register UI endpoint - now uses enhanced UI
        @self.app.tb(mod_name="agent_ui", api=True, version="1", api_methods=['GET'])
        async def ui():
            return Result.html(
                self._get_enhanced_agent_ui_html(agent_id), row=True
            )

        # Register API endpoint for direct agent interaction
        @self.app.tb(mod_name="agent_ui", api=True, version="1", request_as_kwarg=True, api_methods=['POST'])
        async def run_agent(request: RequestData):
            return await self._handle_direct_agent_run(agent_id, agent, request)

        # Register additional API endpoints for enhanced features
        @self.app.tb(mod_name="agent_ui", api=True, version="1", request_as_kwarg=True, api_methods=['POST'])
        async def reset_context(request: RequestData):
            return await self._handle_api_reset_context(agent_id, agent, request)

        @self.app.tb(mod_name="agent_ui", api=True, version="1", request_as_kwarg=True, api_methods=['GET'])
        async def status(request: RequestData):
            return await self._handle_api_get_status(agent_id, agent, request)

        # WebSocket endpoint URL
        uri = f"{host}:{port}" if port else f"{host}"
        ws_url = f"ws://{uri}/ws/agent_ui/connect"
        ui_url = f"http://{uri}/api/agent_ui/ui"
        api_url = f"http://{uri}/api/agent_ui/run_agent"

        return {
            'ui_url': ui_url,
            'ws_url': ws_url,
            'api_url': api_url,
            'reset_url': f"http://localhost:{self.app.args_sto.port}/api/agent_ui/reset_context",
            'status_url': f"http://localhost:{self.app.args_sto.port}/api/agent_ui/status",
            'server_type': 'builtin',
            'status': 'running'
        }

    async def _setup_standalone_server_hosting(self, agent_id: str, agent, host: str, port: int) -> dict[str, str]:
        """Setup agent hosting using standalone Python HTTP server with enhanced UI support."""

        if not hasattr(self, '_standalone_servers'):
            self._standalone_servers = {}

        if port in self._standalone_servers:
            self.app.print(f"Port {port} is already in use by another agent")
            return {'status': 'error', 'error': f'Port {port} already in use'}

        # Store server info for the handler
        server_info = {
            'agent_id': agent_id,
            'server_type': 'standalone',
            'api_paths': {
                'ui': '/ui',
                'status': '/api/status',
                'run': '/api/run',
                'reset': '/api/reset'
            }
        }

        # Create handler factory with agent reference and server info
        def handler_factory(*args, **kwargs):
            handler = EnhancedAgentRequestHandler(self, agent_id, agent, *args, **kwargs)
            handler.server_info = server_info
            return handler

        # Start HTTP server in separate thread
        def run_server():
            try:
                httpd = HTTPServer((host, port), handler_factory)
                self._standalone_servers[port] = {
                    'server': httpd,
                    'agent_id': agent_id,
                    'thread': threading.current_thread(),
                    'server_info': server_info
                }

                self.app.print(f"Enhanced standalone server for agent '{agent_id}' running on http://{host}:{port}")
                self.app.print(f"  UI: http://{host}:{port}/ui")
                self.app.print(f"  API: http://{host}:{port}/api/run")
                self.app.print(f"  Status: http://{host}:{port}/api/status")

                httpd.serve_forever()

            except Exception as e:
                self.app.print(f"Standalone server failed: {e}")
            finally:
                if port in self._standalone_servers:
                    del self._standalone_servers[port]

        # Start server in daemon thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        # Wait a moment to ensure server starts
        await asyncio.sleep(0.5)

        return {
            'server_type': 'standalone',
            'local_url': f"http://{host}:{port}",
            'ui_url': f"http://{host}:{port}/ui",
            'api_url': f"http://{host}:{port}/api/run",
            'reset_url': f"http://{host}:{port}/api/reset",
            'status_url': f"http://{host}:{port}/api/status",
            'status': 'running',
            'port': port
        }

    async def _handle_direct_agent_run(self, agent_id: str, agent, request_data) -> Result:
        """Handle direct agent API calls with enhanced progress tracking."""

        try:
            # Parse request body
            body = request_data.body if hasattr(request_data, 'body') else {}

            if not isinstance(body, dict):
                return Result.default_user_error("Request body must be JSON object", exec_code=400)

            query = body.get('query', '')
            session_id = body.get('session_id', f'api_{secrets.token_hex(8)}')
            kwargs = body.get('kwargs', {})
            include_progress = body.get('include_progress', True)

            if not query:
                return Result.default_user_error("Missing 'query' field in request body", exec_code=400)

            # Enhanced progress tracking for API
            progress_events = []
            enhanced_progress = {}

            async def enhanced_api_progress_callback(event):
                if include_progress:
                    progress_tracker = EnhancedProgressTracker()
                    progress_data = progress_tracker.extract_progress_data(event)

                    progress_events.append({
                        'timestamp': event.timestamp,
                        'event_type': event.event_type,
                        'status': event.status.value if event.status else 'unknown',
                        'agent_name': event.agent_name,
                        'metadata': event.metadata
                    })

                    # Store enhanced progress data
                    enhanced_progress.update(progress_data)

            # Set progress callback
            original_callback = getattr(agent, 'progress_callback', None)

            try:
                if hasattr(agent, 'set_progress_callback'):
                    agent.set_progress_callback(enhanced_api_progress_callback)
                elif hasattr(agent, 'progress_callback'):
                    agent.progress_callback = enhanced_api_progress_callback

                # Execute agent
                result = await agent.a_run(query=query, session_id=session_id, **kwargs)

                # Return enhanced structured response
                response_data = {
                    'success': True,
                    'result': result,
                    'session_id': session_id,
                    'agent_id': agent_id,
                    'execution_time': time.time()
                }

                if include_progress:
                    response_data.update({
                        'progress_events': progress_events,
                        'enhanced_progress': enhanced_progress,
                        'outline_info': enhanced_progress.get('outline', {}),
                        'system_info': enhanced_progress.get('system', {}),
                        'meta_tools_used': enhanced_progress.get('meta_tools', [])
                    })

                return Result.json(data=response_data)

            except Exception as e:
                self.app.print(f"Agent execution error: {e}")
                return Result.default_internal_error(
                    info=f"Agent execution failed: {str(e)}",
                    exec_code=500
                )
            finally:
                # Restore original callback
                if hasattr(agent, 'set_progress_callback'):
                    agent.set_progress_callback(original_callback)
                elif hasattr(agent, 'progress_callback'):
                    agent.progress_callback = original_callback

        except Exception as e:
            self.app.print(f"Direct agent run error: {e}")
            return Result.default_internal_error(
                info=f"Request processing failed: {str(e)}",
                exec_code=500
            )

    async def _handle_api_reset_context(self, agent_id: str, agent, request_data) -> Result:
        """Handle API context reset requests."""
        try:
            if hasattr(agent, 'clear_context'):
                agent.clear_context()
                message = "Context reset successfully"
                success = True
            elif hasattr(agent, 'reset'):
                agent.reset()
                message = "Agent reset successfully"
                success = True
            else:
                message = "Agent does not support context reset"
                success = False

            return Result.json(data={
                'success': success,
                'message': message,
                'agent_id': agent_id,
                'timestamp': time.time()
            })

        except Exception as e:
            return Result.default_internal_error(
                info=f"Context reset failed: {str(e)}",
                exec_code=500
            )

    async def _handle_api_get_status(self, agent_id: str, agent, request_data) -> Result:
        """Handle API status requests."""
        try:
            # Collect comprehensive agent status
            status_info = {
                'agent_id': agent_id,
                'agent_name': getattr(agent, 'name', 'Unknown'),
                'agent_type': agent.__class__.__name__,
                'status': 'active',
                'timestamp': time.time(),
                'server_type': 'api'
            }

            # Add agent-specific status
            if hasattr(agent, 'status'):
                try:
                    agent_status = agent.status()
                    if isinstance(agent_status, dict):
                        status_info['agent_status'] = agent_status
                except:
                    pass

            # Add hosted agent info
            if hasattr(self, '_hosted_agents') and agent_id in self._hosted_agents:
                hosted_info = self._hosted_agents[agent_id]
                status_info.update({
                    'host': hosted_info.get('host'),
                    'port': hosted_info.get('port'),
                    'access': hosted_info.get('access'),
                    'public_name': hosted_info.get('public_name'),
                    'description': hosted_info.get('description')
                })

            # Add connection info
            connection_count = 0
            if hasattr(self, '_agent_connections') and agent_id in self._agent_connections:
                connection_count = len(self._agent_connections[agent_id])

            status_info['active_connections'] = connection_count

            return Result.json(data=status_info)

        except Exception as e:
            return Result.default_internal_error(
                info=f"Status retrieval failed: {str(e)}",
                exec_code=500
            )

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
