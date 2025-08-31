# toolboxv2/mods/isaa/ui.py
import asyncio
import json
import time  # Keep for now, might be useful elsewhere
import uuid
from typing import Dict, Optional, List, Any, AsyncGenerator

from pydantic import BaseModel, Field

from toolboxv2 import get_app, App, RequestData, Result, ToolBoxError, ToolBoxResult, ToolBoxInfo, ToolBoxInterfaces

# Moduldefinition
MOD_NAME = "isaa.ui"
VERSION = "0.1.0"
export = get_app(f"{MOD_NAME}.API").tb  # Assuming this sets up the export correctly
Name = MOD_NAME


# --- Helper to get ISAA instance ---
def get_isaa_instance(app: App):
    # Ensure isaa module is loaded if it has an explicit init or load mechanism
    # This might be handled by app.get_mod if 'isaa' is a known module alias
    isaa_mod = app.get_mod("isaa")
    if not isaa_mod:
        raise ValueError("ISAA module not found or loaded.")
    # Assuming the main ISAA class instance is an attribute or accessible via a function
    # For example, if isaa_mod is the module itself and has an 'agent_manager' instance:
    # return isaa_mod.agent_manager
    # Or if get_mod returns the primary class instance:
    return isaa_mod


# --- API Endpunkte ---
@export(mod_name=MOD_NAME, version=VERSION)
async def version(app: App):
    return VERSION


class RunAgentStreamParams(BaseModel):  # For GET query parameters
    agent_name: str = "self"
    prompt: str
    session_id: Optional[str] = None


# Note: The 'export' decorator and your app framework must support
# mapping GET query parameters to the Pydantic model (RunAgentStreamParams).
# If not, you'd change the signature to:
# async def run_agent_stream(app: App, request: RequestData, agent_name: str, prompt: str, session_id: Optional[str] = None):
# and extract params from request.query_params or however your framework provides them.
# For this example, we assume Pydantic model binding from query params works.
@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['GET'])
async def run_agent_stream(app: App, session_id, agent_name, prompt, request: RequestData=None,  **kwargs):

    isaa = get_isaa_instance(app)
    # Params are already parsed into the 'params' Pydantic model by the framework (assumed)
    session_id_val = session_id or f"webui-session-{request.session.SiID[:8] if request.session_id else uuid.uuid4().hex[:8]}"

    try:
        agent = await isaa.get_agent(agent_name)  # isaa.get_agent is async

        async def sse_event_generator() -> AsyncGenerator[Dict[str, Any], None]:
            # This generator yields dictionaries that SSEGenerator will format.
            # SSEGenerator will add 'stream_start' and 'stream_end' events.

            original_stream_state = agent.stream
            original_callback = agent.stream_callback

            agent.stream = True  # Force streaming for this call

            # Buffer for SSE yields from the agent's stream_callback
            event_queue = asyncio.Queue()
            done_marker = object()  # Sentinel

            async def temp_agent_stream_callback(chunk_str: str):
                # This callback is called by the agent with raw chunks
                # We package it as an SSE event dictionary
                await event_queue.put({'event': 'token', 'data': {'content': chunk_str}})

            agent.stream_callback = temp_agent_stream_callback

            # Run the agent in a separate task so we can consume from the event_queue
            # and allow a_run to complete to get its final response.
            agent_processing_task = asyncio.create_task(
                agent.a_run(user_input=prompt, session_id=session_id_val)
            )

            try:
                # Consume from the queue until the agent task is done processing tokens
                while not agent_processing_task.done() or not event_queue.empty():
                    try:
                        event_dict = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                        if event_dict is done_marker:  # Should not happen with this logic
                            break
                        yield event_dict
                        event_queue.task_done()
                    except asyncio.TimeoutError:
                        if agent_processing_task.done() and event_queue.empty():
                            break  # Agent finished and queue is empty
                        continue  # Agent still running or queue has items, just timed out waiting

                final_response_text = await agent_processing_task  # Get the full response

                yield {'event': 'final_response', 'data': {'content': final_response_text}}
                yield {'event': 'status', 'data': {'message': 'Agent processing complete.'}}

            except Exception as e_inner:
                app.logger.error(f"Error during agent streaming for SSE: {e_inner}", exc_info=True)
                yield {'event': 'error', 'data': {'message': f"Streaming error: {str(e_inner)}"}}
            finally:
                # Restore original agent stream state
                agent.stream = original_stream_state
                agent.stream_callback = original_callback

        # The cleanup_func for Result.sse is for the SSE stream itself, not the agent.
        return Result.sse(stream_generator=sse_event_generator())

    except Exception as e_outer:
        app.logger.error(f"Error setting up run_agent_stream: {e_outer}", exc_info=True)

        # For setup errors, we also need to yield through an async generator for Result.sse
        async def error_event_generator():
            yield {'event': 'error', 'data': {'message': str(e_outer)}}

        return Result.sse(stream_generator=error_event_generator())


class RunAgentRequest(BaseModel):  # For POST body of run_agent_once
    agent_name: str = "self"
    prompt: str
    session_id: Optional[str] = None


@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['POST'])
async def run_agent_once(app: App, request: RequestData, data: RunAgentRequest):
    isaa = get_isaa_instance(app)
    # Assuming ISAA might have an init method if not auto-initialized
    # if hasattr(isaa, 'initialized') and not isaa.initialized:
    #     if hasattr(isaa, 'init_isaa'):
    #         await isaa.init_isaa(build=True) # Or however ISAA is initialized
    if request is None or data is None:
        return Result.default_user_error(info=f"Failed to run agent: No request provided.")
    if isinstance(data, dict):  # Should be automatically handled by Pydantic if type hint is RunAgentRequest
        data = RunAgentRequest(**data)

    session_id_val = data.session_id or f"webui-session-{request.session.SiID[:8] if request.session_id else uuid.uuid4().hex[:8]}"

    try:
        # Ensure agent.stream is False for a single response
        agent = await isaa.get_agent(data.agent_name)
        original_stream_state = agent.stream
        agent.stream = False  # Explicitly set for non-streaming run

        result_text = await agent.a_run(user_input=data.prompt, session_id=session_id_val)

        agent.stream = original_stream_state  # Restore
        return Result.json(data={"response": result_text})
    except Exception as e:
        app.logger.error(f"Error running agent {data.agent_name}: {e}", exc_info=True)
        return Result.custom_error(info=f"Failed to run agent: {str(e)}", exec_code=500)


@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['GET'])
async def list_agents(app: App, request: Optional[RequestData] = None):
    isaa = get_isaa_instance(app)
    agent_names = []
    if hasattr(isaa, 'config') and isaa.config:  # Check if isaa has config and it's not None
        agent_names = isaa.config.get('agents-name-list', [])

    detailed_agents = []
    for name in agent_names:
        agent_data = None
        if hasattr(isaa, 'agent_data') and name in isaa.agent_data:
            agent_data = isaa.agent_data[name]

        if agent_data and isinstance(agent_data, dict):  # Assuming agent_data stores dicts (BuilderConfig)
            detailed_agents.append({
                "name": name,
                "description": agent_data.get("system_message",
                                              agent_data.get("description", "No description available.")),
                "model": agent_data.get("model_identifier", "N/A")
            })
        elif agent_data and hasattr(agent_data, 'description') and hasattr(agent_data,
                                                                           'model_identifier'):  # If it's an object
            detailed_agents.append({
                "name": name,
                "description": agent_data.description or "No description available.",
                "model": agent_data.model_identifier or "N/A"
            })
        else:
            detailed_agents.append({
                "name": name,
                "description": "No detailed configuration found.",
                "model": "N/A"
            })
    return Result.json(data=detailed_agents)  # Result.json expects the data directly


# --- Hauptseite ---
@export(mod_name=MOD_NAME, api=True, version=VERSION, name="main", api_methods=['GET'])
async def get_isaa_webui_page(app: App, request: Optional[RequestData] = None):
    if app is None:  # Should not happen if called via export
        app = get_app()
    # HTML content (truncated for brevity, only script part shown)
    html_content = """
                                       <div class="main-content frosted-glass">
                                           <title>ISAA Web UI</title>
                                           <style>
                                               body {
                                                   transition: background-color 0.3s, color 0.3s;
                                               }

                                               #chat-output p {
                                                   margin-bottom: 0.5em;
                                               }

                                               .user-message { color: #3b82f6; /* Blue */ }
                                               .agent-message { color: #10b981; /* Green */ }
                                               .system-message { color: #f59e0b; /* Amber */ }
                                               .error-message { color: #ef4444; /* Red */ }
                                               .thinking-indicator {
                                                   display: inline-block; width: 20px; height: 20px;
                                                   border: 3px solid rgba(0, 0, 0, .3);
                                                   border-radius: 50%; border-top-color: #fff;
                                                   animation: spin 1s ease-in-out infinite; margin-left: 10px;
                                               }
                                               @keyframes spin { to { transform: rotate(360deg); } }
                                           </style>
                                           <div id="app-root" class="tb-container tb-mx-auto tb-p-4 tb-flex tb-flex-col tb-h-screen">
                                               <header class="tb-flex tb-justify-between tb-items-center tb-mb-4 tb-pb-2 tb-border-b">
                                                   <h1 class="tb-text-3xl tb-font-bold">ISAA Interactive</h1>
                                                   <div><div id="darkModeToggleContainer" style="display: inline-block;"></div></div>
                                               </header>
                                               <div class="tb-flex tb-flex-grow tb-overflow-hidden tb-space-x-4">
                                                   <aside class="tb-w-1/4 tb-p-4 tb-bg-gray-100 dark:tb-bg-gray-800 tb-rounded-lg tb-overflow-y-auto">
                                                       <h2 class="tb-text-xl tb-font-semibold tb-mb-3">Agents</h2>
                                                       <div id="agent-list" class="tb-space-y-2"><p class="tb-text-sm tb-text-gray-500">Lade Agenten...</p></div>
                                                       <hr class="tb-my-4">
                                                       <h2 class="tb-text-xl tb-font-semibold tb-mb-3">Settings</h2>
                                                       <div class="tb-form-group">
                                                           <label for="session-id-input" class="tb-label">Session ID:</label>
                                                           <input type="text" id="session-id-input" class="tb-input tb-w-full tb-mb-2" placeholder="Optional, auto-generiert">
                                                       </div>
                                                       <div class="tb-form-group">
                                                           <label for="streaming-toggle" class="tb-label tb-flex tb-items-center">
                                                               <input type="checkbox" id="streaming-toggle" class="tb-checkbox tb-mr-2" checked> Enable Streaming
                                                           </label>
                                                       </div>
                                                   </aside>
                                                   <main class="tb-w-3/4 tb-flex tb-flex-col tb-bg-white dark:tb-bg-gray-700 tb-rounded-lg tb-shadow-lg tb-overflow-hidden">
                                                       <div id="chat-output" class="tb-flex-grow tb-p-4 tb-overflow-y-auto tb-prose dark:tb-prose-invert tb-max-w-none">
                                                           <p class="system-message">Willkommen bei ISAA Interactive! Wählen Sie einen Agenten und starten Sie den Chat.</p>
                                                       </div>
                                                       <div class="tb-p-4 tb-border-t dark:tb-border-gray-600">
                                                           <form id="chat-form" class="tb-flex tb-space-x-2">
                                                               <input type="text" id="chat-input" class="tb-input tb-flex-grow" placeholder="Nachricht an Agenten..." autocomplete="off">
                                                               <button type="submit" id="send-button" class="tb-btn tb-btn-primary">
                                                                   <span class="material-symbols-outlined">send</span>
                                                               </button>
                                                           </form>
                                                       </div>
                                                   </main>
                                               </div>
                                           </div>

                                           <script defer type="module">
                                               // Warten bis DOM geladen ist und tbjs initialisiert wurde
                                               if (window.TB?.events) {
                                                   if (window.TB.config?.get('appRootId')) {
                                                       initializeAppISAA();
                                                   } else {
                                                       window.TB.events.on('tbjs:initialized', initializeAppISAA, {once: true});
                                                   }
                                               } else {
                                                   document.addEventListener('tbjs:initialized', initializeAppISAA, {once: true});
                                               }

                                               let currentAgentName = 'self'; // Default agent
                                               let currentSessionId = '';
                                               let sseConnection = null; // Stores the EventSource object
                                               let currentSseUrl = null; // Stores the URL of the current SSE connection

                                               function initializeAppISAA() {
                                                   TB.ui.Toast.showInfo("ISAA UI Initialized!");
                                                   loadAgentList();

                                                   const chatForm = document.getElementById('chat-form');
                                                   const chatInput = document.getElementById('chat-input');
                                                   const sendButton = document.getElementById('send-button');
                                                   const sessionIdInput = document.getElementById('session-id-input');
                                                   const chatOutput = document.getElementById('chat-output'); // Added

                                                   chatForm.addEventListener('submit', async (e) => {
                                                       e.preventDefault();
                                                       const prompt = chatInput.value.trim();
                                                       if (!prompt) return;

                                                       currentSessionId = sessionIdInput.value.trim() || `webui-session-${Date.now()}${Math.random().toString(36).substring(2,6)}`;
                                                       sessionIdInput.value = currentSessionId;

                                                       addMessageToChat('user', prompt);
                                                       chatInput.value = '';
                                                       sendButton.disabled = true;
                                                       addThinkingIndicator();

                                                       const useStreaming = document.getElementById('streaming-toggle').checked;

                                                       if (useStreaming) {
                                                           handleStreamedAgentRequest(currentAgentName, prompt, currentSessionId);
                                                       } else {
                                                           // Disconnect any active SSE stream if switching to non-streaming
                                                           if (sseConnection && currentSseUrl) {
                                                               TB.sse.disconnect(currentSseUrl);
                                                               sseConnection = null;
                                                               currentSseUrl = null;
                                                           }
                                                           try {
                                                               const response = await TB.api.request('isaa.ui', 'run_agent_once', {
                                                                   agent_name: currentAgentName,
                                                                   prompt: prompt,
                                                                   session_id: currentSessionId
                                                               }, 'POST'); // This remains POST

                                                               removeThinkingIndicator();
                                                               if (response.error === TB.ToolBoxError.none && response.get()?.response) {
                                                                   addMessageToChat('agent', response.get().response);
                                                               } else {
                                                                   addMessageToChat('error', 'Fehler: ' + (response.info?.help_text || response.error?.message || 'Unbekannter Fehler'));
                                                               }
                                                           } catch (error) {
                                                               removeThinkingIndicator();
                                                               addMessageToChat('error', 'Netzwerkfehler oder serverseitiger Fehler: ' + error.message);
                                                               console.error(error);
                                                           } finally {
                                                               sendButton.disabled = false;
                                                           }
                                                       }
                                                   });
                                               }

                                               function handleStreamedAgentRequest(agentName, prompt, sessionId) {
                                                   const chatOutput = document.getElementById('chat-output');
                                                   const sendButton = document.getElementById('send-button');
                                                   let agentMessageElement = null;

                                                   // Disconnect previous SSE connection if exists
                                                   if (sseConnection && currentSseUrl) {
                                                       TB.sse.disconnect(currentSseUrl);
                                                       sseConnection = null;
                                                       currentSseUrl = null;
                                                   }

                                                   // Construct the SSE URL with query parameters
                                                   // The Python endpoint 'run_agent_stream' needs to be GET and handle these.
                                                   const queryParams = new URLSearchParams({
                                                       agent_name: agentName,
                                                       prompt: prompt,
                                                       session_id: sessionId
                                                   });
                                                   const sseEndpointUrl = `/sse/isaa.ui/run_agent_stream?${queryParams.toString()}`;
                                                   currentSseUrl = sseEndpointUrl; // Store for disconnect

                                                   TB.logger.info(`SSE: Connecting to ${sseEndpointUrl}`);

                                                   sseConnection = TB.sse.connect(sseEndpointUrl, {
                                                       onOpen: (event) => {
                                                           TB.logger.log(`SSE: Connection opened to ${sseEndpointUrl}`, event);
                                                           // The 'stream_start' event from Python will provide more app-specific status
                                                       },
                                                       onError: (error) => { // This is for EventSource level errors
                                                           TB.logger.error(`SSE: Connection error with ${sseEndpointUrl}`, error);
                                                           addMessageToChat('error', 'Streaming connection error. Please try again.');
                                                           removeThinkingIndicator();
                                                           sendButton.disabled = false;
                                                           agentMessageElement = null;
                                                           sseConnection = null; // Clear connection object
                                                           currentSseUrl = null;
                                                       },
                                                       // onMessage: (data, event) => { // Generic message, less useful if using named events
                                                       //     TB.logger.log('SSE generic message:', data);
                                                       // },
                                                       listeners: {
                                                           'stream_start': (eventPayload, event) => { // eventPayload is data from 'data:' line, parsed
                                                               TB.logger.log('SSE Event (stream_start):', eventPayload);
                                                               // eventPayload should contain {'id': '0'}
                                                               addMessageToChat('system', `Agent ${agentName} started streaming... (ID: ${eventPayload?.id})`);
                                                           },
                                                           'token': (eventPayload, event) => {
                                                               TB.logger.debug('SSE Event (token):', eventPayload);
                                                               if (!agentMessageElement) {
                                                                   agentMessageElement = addMessageToChat('agent', '', true); // Create empty, return element
                                                               }
                                                               if (eventPayload && typeof eventPayload.content === 'string') {
                                                                  agentMessageElement.textContent += eventPayload.content;
                                                                  chatOutput.scrollTop = chatOutput.scrollHeight;
                                                               } else {
                                                                  TB.logger.warn('SSE: Received token event without valid data.content', eventPayload);
                                                               }
                                                           },
                                                           'final_response': (eventPayload, event) => {
                                                               TB.logger.log('SSE Event (final_response):', eventPayload);
                                                               if (agentMessageElement && eventPayload && typeof eventPayload.content === 'string') {
                                                                   agentMessageElement.textContent = eventPayload.content; // Overwrite if partial was different
                                                               } else if (eventPayload && typeof eventPayload.content === 'string') {
                                                                   addMessageToChat('agent', eventPayload.content);
                                                               }
                                                               // Usually stream_end will handle UI finalization
                                                           },
                                                           'status': (eventPayload, event) => {
                                                               TB.logger.log('SSE Event (status):', eventPayload);
                                                               if (eventPayload && typeof eventPayload.message === 'string') {
                                                                  addMessageToChat('system', eventPayload.message);
                                                               }
                                                           },
                                                           'error': (eventPayload, event) => { // Application-level errors from the stream
                                                               TB.logger.error('SSE Event (error):', eventPayload);
                                                               if (eventPayload && typeof eventPayload.message === 'string') {
                                                                  addMessageToChat('error', 'Stream error: ' + eventPayload.message);
                                                               } else {
                                                                  addMessageToChat('error', 'An unknown error occurred during streaming.');
                                                               }
                                                               removeThinkingIndicator();
                                                               sendButton.disabled = false;
                                                               agentMessageElement = null;
                                                               if (sseConnection && currentSseUrl) TB.sse.disconnect(currentSseUrl); // Disconnect on app error
                                                               sseConnection = null;
                                                               currentSseUrl = null;
                                                           },
                                                           'stream_end': (eventPayload, event) => {
                                                               TB.logger.log('SSE Event (stream_end):', eventPayload);
                                                               // eventPayload should contain {'id': 'final'}
                                                               addMessageToChat('system', `Stream finished. (ID: ${eventPayload?.id})`);
                                                               removeThinkingIndicator();
                                                               sendButton.disabled = false;
                                                               agentMessageElement = null;
                                                               // EventSource keeps connection alive for potential retries.
                                                               // If the stream is definitively over, explicitly close.
                                                               if (sseConnection && currentSseUrl) TB.sse.disconnect(currentSseUrl);
                                                               sseConnection = null;
                                                               currentSseUrl = null;
                                                           }
                                                           // You can also listen for 'binary' events if your Python side sends them
                                                       }
                                                   });
                                               }

                                               async function loadAgentList() {
                                                   const agentListDiv = document.getElementById('agent-list');
                                                   try {
                                                       const response = await TB.api.request('isaa.ui', 'list_agents', null, 'GET');
                                                       // Assuming TB.api.request returns a similar structure to Result.to_api_result()
                                                       // and TB.ToolBoxError.none is available
                                                       if (response.error === TB.ToolBoxError.none && response.result?.data) {
                                                           const agents = response.result.data;
                                                           agentListDiv.innerHTML = '';
                                                           if (agents.length === 0) {
                                                               agentListDiv.innerHTML = '<p class="tb-text-sm tb-text-gray-500">Keine Agenten verfügbar.</p>';
                                                               return;
                                                           }
                                                           agents.forEach(agent => {
                                                               const agentButton = document.createElement('button');
                                                               agentButton.className = 'tb-btn tb-btn-secondary tb-w-full tb-text-left tb-mb-1';
                                                               agentButton.textContent = agent.name;
                                                               agentButton.title = `${agent.description || 'N/A'}\\nModel: ${agent.model || 'N/A'}`;
                                                               if (agent.name === currentAgentName) {
                                                                   agentButton.classList.add('tb-btn-primary');
                                                               }
                                                               agentButton.addEventListener('click', () => {
                                                                   currentAgentName = agent.name;
                                                                   TB.ui.Toast.showInfo(`Agent auf ${agent.name} gewechselt.`);
                                                                   document.querySelectorAll('#agent-list button').forEach(btn => {
                                                                       btn.classList.remove('tb-btn-primary');
                                                                       btn.classList.add('tb-btn-secondary');
                                                                   });
                                                                   agentButton.classList.add('tb-btn-primary');
                                                                   agentButton.classList.remove('tb-btn-secondary');
                                                                   addMessageToChat('system', `Agent auf ${agent.name} (${agent.model || 'N/A'}) gewechselt.`);
                                                               });
                                                               agentListDiv.appendChild(agentButton);
                                                           });
                                                       } else {
                                                           agentListDiv.innerHTML = '<p class="tb-text-sm tb-text-red-500">Fehler beim Laden der Agenten.</p>';
                                                           TB.logger.error("Failed to load agents:", response.info?.help_text || response.error);
                                                       }
                                                   } catch (error) {
                                                       agentListDiv.innerHTML = '<p class="tb-text-sm tb-text-red-500">Netzwerkfehler beim Laden der Agenten.</p>';
                                                       console.error(error);
                                                        TB.logger.error("Network error loading agents:", error);
                                                   }
                                               }

                                               function addMessageToChat(role, text, returnElement = false) {
                                                   const chatOutput = document.getElementById('chat-output');
                                                   const messageElement = document.createElement('p');
                                                   messageElement.className = `${role}-message`; // Ensure Tailwind classes are separate if needed
                                                   messageElement.textContent = text;
                                                   chatOutput.appendChild(messageElement);
                                                   chatOutput.scrollTop = chatOutput.scrollHeight;
                                                   if (returnElement) return messageElement;
                                               }

                                               let thinkingIndicatorDiv = null;
                                               function addThinkingIndicator() {
                                                   if (thinkingIndicatorDiv) return;
                                                   const chatForm = document.getElementById('chat-form');
                                                   thinkingIndicatorDiv = document.createElement('div');
                                                   thinkingIndicatorDiv.className = 'thinking-indicator';
                                                   // Append after the send button or within the form
                                                   const sendButton = document.getElementById('send-button');
                                                   sendButton.parentNode.insertBefore(thinkingIndicatorDiv, sendButton.nextSibling);
                                               }

                                               function removeThinkingIndicator() {
                                                   if (thinkingIndicatorDiv) {
                                                       thinkingIndicatorDiv.remove();
                                                       thinkingIndicatorDiv = null;
                                                   }
                                               }
                                           </script>
                                       </div>
                                       """
    return Result.html(data=html_content)  # Assuming row=True means don't add extra wrappers


# Initialisierungsfunktion für das Modul (optional)
@export(mod_name=MOD_NAME, version=VERSION)
def initialize_isaa_webui_module(app: App, isaa_instance=None):  # isaa_instance might be passed if main app manages it
    if app is None:
        app = get_app()

    # Ensure the ISAA module itself is initialized if it has specific setup
    if isaa_instance is None:
        isaa_instance = get_isaa_instance(app)  # Get or load the main ISAA module/class

    # Example: if ISAA has an init method that needs to be called
    # if hasattr(isaa_instance, 'init_isaa') and callable(isaa_instance.init_isaa):
    #     app.run_async(isaa_instance.init_isaa()) # if it's async

    # Assuming CloudM module is available and add_ui is a known function
    try:
        # If add_ui is async, you might need app.run_async or similar
        app.run_any(("CloudM", "add_ui"),  # Or ("CloudM", "add_ui") if get_mod returns the module object
                    name=Name,
                    title="ISAA UI",  # More user-friendly title
                    path=f"/api/{MOD_NAME}/main",  # Use MOD_NAME for consistency
                    description="Interactive Web UI for ISAA",auth=True
                    )
    except Exception as e:
        app.logger.error(f"Failed to register ISAA UI with CloudM: {e}")

    return Result.ok(info="ISAA WebUI Modul bereit.")


# isaa/ui.py
#@export(mod_name=MOD_NAME, api=True, version=VERSION, api_methods=['GET'])
# isaa/ui.py

# isaa/ui.py

# isaa/ui.py
# isaa/ui.py

import asyncio
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import time


def get_agent_ui_html() -> str:
    """
    Returns the full HTML, CSS, and JavaScript for the interactive agent UI.
    """
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ISAA FlowAgent Interface</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --background-color: #1a1a1a;
            --panel-background: #242424;
            --text-color: #e0e0e0;
            --muted-text-color: #888;
            --border-color: #3a3a3a;
            --accent-color: #007acc;
            --error-color: #e53935;
            --success-color: #43a047;
            --warning-color: #fdd835;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        body {
            font-family: var(--font-family);
            background-color: var(--background-color);
            color: var(--text-color);
            margin: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }
        .main-container {
            display: flex;
            flex: 1;
            overflow: hidden;
        }
        .panel {
            background-color: var(--panel-background);
            border-left: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        #chat-container { flex: 2; border-left: none; }
        #log-container { flex: 1.5; }
        #status-container { flex: 1; }
        .panel-header {
            padding: 8px 12px;
            background-color: #2c2c2c;
            border-bottom: 1px solid var(--border-color);
            font-weight: bold;
        }
        .panel-content {
            padding: 12px;
            overflow-y: auto;
            flex-grow: 1;
        }
        .message {
            margin-bottom: 15px;
            display: flex;
        }
        .message .avatar {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background-color: #444;
            margin-right: 10px;
            text-align: center;
            line-height: 30px;
            font-size: 12px;
        }
        .message.user {
            flex-direction: row-reverse;
        }
        .message.user .avatar {
            margin-left: 10px;
            margin-right: 0;
            background-color: var(--accent-color);
        }
        .message .content {
            max-width: 80%;
        }
        .message .text {
            background-color: #333;
            padding: 10px;
            border-radius: 8px;
        }
        .message.user .text {
            background-color: #004a7c;
        }
        #chat-input-area {
            border-top: 1px solid var(--border-color);
            padding: 10px;
            display: flex;
        }
        #chat-input {
            flex-grow: 1;
            background-color: #333;
            border: 1px solid #444;
            color: var(--text-color);
            padding: 8px;
            border-radius: 5px;
        }
        #send-button {
            background-color: var(--accent-color);
            color: white;
            border: none;
            padding: 8px 15px;
            margin-left: 10px;
            border-radius: 5px;
            cursor: pointer;
        }
        #send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .main-controls {
            padding: 10px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            gap: 10px;
        }
        .main-controls button {
            background-color: #333;
            border: 1px solid #555;
            color: var(--text-color);
            padding: 8px 12px;
            border-radius: 5px;
            cursor: pointer;
        }
        .main-controls button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .status-grid {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 8px 15px;
            align-items: center;
        }
        .status-key {
            font-weight: bold;
            text-align: right;
            color: var(--muted-text-color);
        }
        .status-value {
            font-family: monospace;
        }
        .loading {
            color: var(--accent-color);
        }
    </style>
</head>
<body>
    <div class="main-controls">
        <button id="reset-btn">Reset Context</button>
        <span id="connection-status" style="margin-left: auto; align-self: center;">Ready</span>
    </div>
    <div class="main-container">
        <div id="chat-container" class="panel">
            <div class="panel-header">Chat</div>
            <div class="panel-content" id="chat-messages"></div>
            <div id="chat-input-area">
                <input type="text" id="chat-input" placeholder="Send a message to the agent..." autocomplete="off">
                <button id="send-button">Send</button>
            </div>
        </div>
        <div id="log-container" class="panel">
            <div class="panel-header">Execution Log</div>
            <div class="panel-content" id="log-content">No activity yet.</div>
        </div>
        <div id="status-container" class="panel">
            <div class="panel-header">System Status</div>
            <div class="panel-content" id="status-grid"></div>
        </div>
    </div>

    <script>
        const app = {
            isProcessing: false,

            elements: {
                chatMessages: document.getElementById('chat-messages'),
                chatInput: document.getElementById('chat-input'),
                sendButton: document.getElementById('send-button'),
                logContent: document.getElementById('log-content'),
                statusGrid: document.getElementById('status-grid'),
                connectionStatus: document.getElementById('connection-status'),
                resetBtn: document.getElementById('reset-btn'),
            },

            init() {
                this.initEventListeners();
                this.updateStatus();
            },

            initEventListeners() {
                this.elements.sendButton.addEventListener('click', () => this.sendMessage());
                this.elements.chatInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        this.sendMessage();
                    }
                });
                this.elements.resetBtn.addEventListener('click', () => this.resetContext());
            },

            async sendMessage() {
                const message = this.elements.chatInput.value.trim();
                if (message && !this.isProcessing) {
                    this.addChatMessage('user', message);
                    this.elements.chatInput.value = '';
                    this.setProcessing(true);

                    try {
                        const response = await fetch('/api/run', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ query: message })
                        });
                        const result = await response.json();

                        if (result.success) {
                            this.addChatMessage('agent', result.response);
                            this.updateLog(result.log);
                            this.updateStatus(result.status);
                        } else {
                            this.addChatMessage('agent', 'Error: ' + result.error);
                        }
                    } catch (error) {
                        this.addChatMessage('agent', 'Network error: ' + error.message);
                    }

                    this.setProcessing(false);
                }
            },

            async resetContext() {
                if (this.isProcessing) return;

                try {
                    const response = await fetch('/api/reset', { method: 'POST' });
                    const result = await response.json();

                    if (result.success) {
                        this.elements.chatMessages.innerHTML = '';
                        this.elements.logContent.textContent = 'Context reset.';
                        this.updateStatus();
                    }
                } catch (error) {
                    console.error('Reset failed:', error);
                }
            },

            addChatMessage(sender, text) {
                const messageEl = document.createElement('div');
                messageEl.classList.add('message', sender);

                const avatarEl = document.createElement('div');
                avatarEl.classList.add('avatar');
                avatarEl.textContent = sender === 'user' ? 'You' : 'AI';

                const contentEl = document.createElement('div');
                contentEl.classList.add('content');

                const textEl = document.createElement('div');
                textEl.classList.add('text');

                if (sender === 'agent' && window.marked) {
                    textEl.innerHTML = marked.parse(text);
                } else {
                    textEl.textContent = text;
                }

                contentEl.appendChild(textEl);
                messageEl.appendChild(sender === 'user' ? contentEl : avatarEl);
                messageEl.appendChild(sender === 'user' ? avatarEl : contentEl);

                this.elements.chatMessages.appendChild(messageEl);
                this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
            },

            setProcessing(processing) {
                this.isProcessing = processing;
                this.elements.sendButton.disabled = processing;
                this.elements.chatInput.disabled = processing;
                this.elements.resetBtn.disabled = processing;
                this.elements.connectionStatus.textContent = processing ? 'Processing...' : 'Ready';
                this.elements.connectionStatus.className = processing ? 'loading' : '';
            },

            updateLog(logData) {
                if (logData) {
                    this.elements.logContent.innerHTML = logData.map(entry =>
                        `<div>[${entry.timestamp}] ${entry.message}</div>`
                    ).join('');
                }
            },

            updateStatus(statusData) {
                const defaultStatus = {
                    'Agent Status': 'Ready',
                    'Last Update': new Date().toLocaleString()
                };

                const status = statusData || defaultStatus;
                let html = '';
                for (const [key, value] of Object.entries(status)) {
                    html += `<div class="status-key">${key}</div><div class="status-value">${value}</div>`;
                }
                this.elements.statusGrid.innerHTML = html;
            }
        };

        document.addEventListener('DOMContentLoaded', () => app.init());
    </script>
</body>
</html>
"""


class AgentRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, agent, module, *args, **kwargs):
        self.agent = agent
        self.module = module
        self.log_entries = []
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(get_agent_ui_html().encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Not Found')

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        if self.path == '/api/run':
            try:
                data = json.loads(post_data.decode('utf-8'))
                query = data.get('query', '')

                # Simple progress tracking
                self.log_entries = [
                    {'timestamp': time.strftime('%H:%M:%S'), 'message': f'Processing query: {query}'},
                    {'timestamp': time.strftime('%H:%M:%S'), 'message': 'Agent started...'}
                ]

                # Run the agent synchronously (simplified)
                try:
                    # Create a simple event loop for async agent
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self.agent.a_run(query))
                    loop.close()

                    self.log_entries.append({
                        'timestamp': time.strftime('%H:%M:%S'),
                        'message': 'Agent completed successfully'
                    })

                    response_data = {
                        'success': True,
                        'response': result or 'Task completed',
                        'log': self.log_entries,
                        'status': {
                            'Agent Status': 'Ready',
                            'Last Query': query,
                            'Last Update': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                    }
                except Exception as e:
                    self.log_entries.append({
                        'timestamp': time.strftime('%H:%M:%S'),
                        'message': f'Agent failed: {str(e)}'
                    })

                    response_data = {
                        'success': False,
                        'error': str(e),
                        'log': self.log_entries
                    }

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))

            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = {'success': False, 'error': str(e)}
                self.wfile.write(json.dumps(error_response).encode('utf-8'))

        elif self.path == '/api/reset':
            try:
                self.agent.clear_context()
                self.log_entries = []

                response_data = {'success': True}
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))

            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = {'success': False, 'error': str(e)}
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Not Found')

    def log_message(self, format, *args):
        # Suppress default HTTP server logs
        pass







