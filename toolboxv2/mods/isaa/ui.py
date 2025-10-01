# toolboxv2/mods/isaa/ui.py
import asyncio
import json
import secrets
import time  # Keep for now, might be useful elsewhere
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel

from toolboxv2 import (
    App,
    RequestData,
    Result,
    get_app,
)

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
    session_id: str | None = None


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

        async def sse_event_generator() -> AsyncGenerator[dict[str, Any], None]:
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
                    except TimeoutError:
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
    session_id: str | None = None


@export(mod_name=MOD_NAME, api=True, version=VERSION, request_as_kwarg=True, api_methods=['POST'])
async def run_agent_once(app: App, request: RequestData, data: RunAgentRequest):
    isaa = get_isaa_instance(app)
    # Assuming ISAA might have an init method if not auto-initialized
    # if hasattr(isaa, 'initialized') and not isaa.initialized:
    #     if hasattr(isaa, 'init_isaa'):
    #         await isaa.init_isaa(build=True) # Or however ISAA is initialized
    if request is None or data is None:
        return Result.default_user_error(info="Failed to run agent: No request provided.")
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
async def list_agents(app: App, request: RequestData | None = None):
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
async def get_isaa_webui_page(app: App, request: RequestData | None = None):
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

from http.server import BaseHTTPRequestHandler

# toolboxv2/mods/registry/ui.py

def get_agent_ui_html() -> str:
    """Produktionsfertige UI mit Live-Progress-Tracking."""

    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Registry - Live Interface</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        /* Modernes Dark Theme UI */
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

            --sidebar-width: 300px;
            --progress-width: 660px;
            --sidebar-collapsed: 60px;
            --progress-collapsed: 60px;
        }
        /* Enhanced Progress Panel Styles */
        .progress-section {
            margin-bottom: 16px;
        }

        /* ADD to existing CSS */
        .event-status-badge {
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: 500;
        }

        .event-status-badge.completed {
            background: var(--accent-green);
            color: white;
        }

        .event-status-badge.running {
            background: var(--accent-orange);
            color: white;
        }

        .event-status-badge.failed, .event-status-badge.error {
            background: var(--accent-red);
            color: white;
        }

        .event-status-badge.starting {
            background: var(--accent-cyan);
            color: white;
        }

        .progress-item.expandable[data-event-id*="tool_call"] {
            border-left-color: var(--accent-orange);
        }

        .progress-item.expandable[data-event-id*="llm_call"] {
            border-left-color: var(--accent-purple);
        }

        .progress-item.expandable[data-event-id*="meta_tool"] {
            border-left-color: var(--accent-cyan);
        }

        .progress-item.expandable[data-event-id*="error"] {
            border-left-color: var(--accent-red);
            background: rgba(248, 81, 73, 0.02);
        }

        .section-title.expandable-section {
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            transition: all 0.2s;
        }

        .section-title.expandable-section:hover {
            background: var(--bg-tertiary);
        }

        .section-toggle {
            transition: transform 0.2s;
            font-size: 12px;
        }

        .section-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
            background: var(--bg-primary);
            border-radius: 0 0 6px 6px;
        }

        .section-content.expanded {
            max-height: 900px;
            padding: 12px;
            border: 1px solid var(--border-color);
            border-top: none;
            overflow-y: auto;
        }

        .no-data {
            color: var(--text-muted);
            font-size: 12px;
            text-align: center;
            padding: 12px;
            font-style: italic;
        }

        /* Expandable Progress Items */
        .progress-item.expandable {
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 8px;
        }

        .progress-item.expandable:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }

        .progress-item.expandable.expanded {
            border-color: var(--accent-blue);
        }

        .progress-item.expandable.latest {
            border-left: 3px solid var(--accent-green);
            background: rgba(63, 185, 80, 0.05);
        }

        .progress-item-header {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
        }

        .progress-meta {
            margin-left: auto;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .expand-indicator {
            transition: transform 0.2s;
            font-size: 12px;
            color: var(--text-muted);
        }

        .progress-item.expanded .expand-indicator {
            transform: rotate(180deg);
        }

        .progress-summary {
            padding: 0 12px 8px 36px;
            font-size: 11px;
            color: var(--text-secondary);
        }

        .progress-item-expanded {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border-color);
        }

        .progress-item-expanded.active {
            max-height: 400px;
            padding: 12px;
            overflow-y: auto;
        }

        .expanded-section {
            margin-bottom: 12px;
        }

        .expanded-section-title {
            font-size: 12px;
            font-weight: 600;
            color: var(--accent-blue);
            margin-bottom: 6px;
            padding-bottom: 4px;
            border-bottom: 1px solid var(--border-color);
        }

        .event-field {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 4px 0;
            font-size: 11px;
        }

        .event-field-label {
            font-weight: 500;
            color: var(--text-secondary);
            min-width: 80px;
        }

        .event-field-value {
            color: var(--text-primary);
            text-align: right;
            flex: 1;
        }

        .event-field-value.json {
            background: var(--bg-primary);
            border-radius: 4px;
            padding: 6px;
            font-family: monospace;
            font-size: 10px;
            text-align: left;
            white-space: pre-wrap;
            max-height: 100px;
            overflow-y: auto;
        }

        /* ADD to existing CSS */
.thinking-step.outline-step {
    border-color: var(--accent-cyan);
    background: rgba(57, 208, 216, 0.05);
}

.thinking-step.outline-step.completed {
    border-color: var(--accent-green);
    background: rgba(63, 185, 80, 0.05);
}

.thinking-step.outline-step.running {
    border-color: var(--accent-orange);
    background: rgba(210, 153, 34, 0.05);
}

.outline-progress {
    margin: 8px 0;
}

.progress-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.progress-text {
    font-size: 11px;
    color: var(--text-secondary);
    font-weight: 500;
}

.progress-percentage {
    font-size: 11px;
    color: var(--accent-blue);
    font-weight: 600;
}

.progress-bar-container {
    margin-bottom: 8px;
}

.progress-bar {
    height: 6px;
    background: var(--bg-primary);
    border-radius: 3px;
    overflow: hidden;
    border: 1px solid var(--border-color);
}

.progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue));
    transition: width 0.5s ease-out;
    position: relative;
}

.progress-bar-fill::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    animation: shimmer 2s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.step-completed {
    color: var(--accent-green);
    font-size: 10px;
    text-align: center;
    font-weight: 500;
}

.step-working {
    color: var(--accent-orange);
    font-size: 10px;
    text-align: center;
    font-style: italic;
}

.context-info {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--border-color);
}

.context-item {
    font-size: 10px;
    color: var(--text-muted);
    background: var(--bg-primary);
    padding: 2px 6px;
    border-radius: 3px;
    border: 1px solid var(--border-color);
}

.thinking-step.plan-created {
    border-color: var(--accent-blue);
    background: rgba(88, 166, 255, 0.05);
}

.plan-details {
    text-align: center;
}

.plan-info {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-bottom: 8px;
}

.plan-item {
    font-size: 11px;
    color: var(--text-secondary);
    background: var(--bg-primary);
    padding: 4px 8px;
    border-radius: 4px;
    border: 1px solid var(--border-color);
}

.plan-ready, .outline-ready {
    margin-top: 8px;
    color: var(--accent-green);
    font-size: 10px;
    text-align: center;
}

.step-status {
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 500;
    margin-left: auto;
}

.step-status.completed {
    background: var(--accent-green);
    color: white;
}

.step-status.running {
    background: var(--accent-orange);
    color: white;
}

.step-status.ready {
    background: var(--accent-blue);
    color: white;
}

        /* Enhanced Chat Integration Styles */
        .thinking-step {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px 12px;
            margin: 8px 0;
            font-size: 13px;
            transition: all 0.2s;
        }

        .thinking-step:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        .thinking-step.reasoning-loop {
            border-color: var(--accent-purple);
            background: rgba(165, 165, 245, 0.05);
        }

        .thinking-step.outline-created {
            border-color: var(--accent-cyan);
            background: rgba(57, 208, 216, 0.05);
        }

        .thinking-step.task-progress.starting {
            border-color: var(--accent-orange);
            background: rgba(210, 153, 34, 0.05);
        }

        .thinking-step.task-progress.completed {
            border-color: var(--accent-green);
            background: rgba(63, 185, 80, 0.05);
        }

        .thinking-step.task-progress.error {
            border-color: var(--accent-red);
            background: rgba(248, 81, 73, 0.05);
        }

        .thinking-step-header {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--text-primary);
        }

        .step-progress, .step-info, .step-status {
            margin-left: auto;
            font-size: 10px;
            font-weight: normal;
            color: var(--text-muted);
            background: var(--bg-primary);
            padding: 2px 6px;
            border-radius: 3px;
        }

        .priority-badge {
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: 500;
        }

        .priority-badge.high {
            background: var(--accent-red);
            color: white;
        }

        .priority-badge.normal {
            background: var(--accent-blue);
            color: white;
        }

        .priority-badge.low {
            background: var(--text-muted);
            color: white;
        }

        /* Performance Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 8px;
        }

        .metric-card {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 8px;
            text-align: center;
        }

        .metric-label {
            font-size: 10px;
            color: var(--text-muted);
            margin-bottom: 4px;
        }

        .metric-value {
            font-size: 14px;
            font-weight: 600;
            color: var(--accent-blue);
        }

        .reasoning-metrics .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            margin-bottom: 8px;
        }

        .metric-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 11px;
        }

        .metric-label {
            color: var(--text-muted);
        }

        .metric-value {
            color: var(--text-primary);
            font-weight: 500;
        }

        /* Progress Bar */
        .progress-bar-container {
            margin: 8px 0;
        }

        .progress-bar-info {
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: var(--text-muted);
            margin-bottom: 4px;
        }

        .progress-bar {
            height: 4px;
            background: var(--bg-primary);
            border-radius: 2px;
            overflow: hidden;
        }

        .progress-bar-fill {
            height: 100%;
            background: var(--accent-blue);
            transition: width 0.5s ease-out;
        }

        /* Outline Display */
        .outline-steps {
            margin: 8px 0;
        }

        .outline-step {
            display: flex;
            align-items: flex-start;
            gap: 8px;
            margin-bottom: 4px;
            font-size: 11px;
        }

        .step-number {
            color: var(--accent-blue);
            font-weight: 600;
            min-width: 20px;
        }

        .step-text {
            color: var(--text-primary);
            line-height: 1.3;
        }

        .context-metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            margin-bottom: 8px;
        }

        .context-metric {
            display: flex;
            flex-direction: column;
            align-items: center;
            font-size: 10px;
            padding: 6px;
            background: var(--bg-primary);
            border-radius: 4px;
        }

        .context-label {
            color: var(--text-muted);
            margin-bottom: 2px;
        }

        .context-value {
            color: var(--text-primary);
            font-weight: 600;
        }

        .task-description {
            margin-bottom: 6px;
            font-weight: 500;
        }

        .task-timing {
            font-size: 10px;
            color: var(--accent-green);
        }

        .task-error {
            font-size: 10px;
            color: var(--accent-red);
            background: rgba(248, 81, 73, 0.1);
            padding: 4px;
            border-radius: 3px;
            margin-top: 4px;
        }

        .reasoning-insight {
            margin-top: 8px;
            font-size: 11px;
            color: var(--accent-purple);
            text-align: center;
            font-style: italic;
        }

        .idle-status {
            border-color: var(--accent-green);
            background: rgba(63, 185, 80, 0.02);
        }

        @media (max-width: 1200px) {
            :root {
                --sidebar-width: 250px;
                --progress-width: 580px;
            }
        }

        @media (max-width: 1024px) {
            :root {
                --sidebar-width: 220px;
                --progress-width: 460px;
            }
        }

        .sidebar.collapsed::before {
            content: '📋';
            font-size: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px 0;
            border-bottom: 1px solid var(--border-color);
        }

        .progress-panel.collapsed::before {
            content: '📊';
            font-size: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px 0;
            border-bottom: 1px solid var(--border-color);
            writing-mode: vertical-lr;
        }

        .sidebar, .progress-panel {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .main-container {
            transition: grid-template-columns 0.3s cubic-bezier(0.4, 0, 0.2, 1);
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

        html, body {
            height: 100%;
            overflow: hidden;
        }

        .api-key-modal {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }

        .api-key-content {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            max-width: 500px;
            width: 90%;
            text-align: center;
        }

        .api-key-title {
            font-size: 20px;
            font-weight: 600;
            color: var(--accent-blue);
            margin-bottom: 16px;
        }

        .api-key-description {
            color: var(--text-secondary);
            margin-bottom: 20px;
            line-height: 1.5;
        }

        .api-key-input {
            width: 100%;
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
            color: var(--text-primary);
            font-size: 14px;
            margin-bottom: 16px;
        }

        .api-key-button {
            background: var(--accent-blue);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            cursor: pointer;
            font-weight: 600;
        }

        /* Updated Header */
        .header {
            background: var(--bg-tertiary);
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: var(--shadow);
            flex-shrink: 0;
        }

        .header-controls {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .panel-toggle {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }

        .panel-toggle:hover {
            background: var(--bg-primary);
        }

        .panel-toggle.active {
            background: var(--accent-blue);
            color: white;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 20px;
            font-weight: 700;
            color: var(--accent-blue);
        }

        .connection-status {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
        }

        .status-indicator.connected {
            background: rgba(63, 185, 80, 0.1);
            color: var(--accent-green);
            border: 1px solid var(--accent-green);
        }

        .status-indicator.disconnected {
            background: rgba(248, 81, 73, 0.1);
            color: var(--accent-red);
            border: 1px solid var(--accent-red);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
            animation: pulse 2s infinite;
        }

        .status-dot.connected { animation: none; }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        /* FIXED: Better grid layout that properly handles collapsing */
        .main-container {
            display: grid;
            grid-template-areas: "sidebar chat progress";
            grid-template-columns: var(--sidebar-width) 1fr var(--progress-width);
            flex: 1;
            overflow: hidden;
            min-height: 0;
            height: 100%;
        }

        .main-container.sidebar-collapsed {
            grid-template-columns: var(--sidebar-collapsed) 1fr var(--progress-width);
        }

        .main-container.progress-collapsed {
            grid-template-columns: var(--sidebar-width) 1fr var(--progress-collapsed);
        }

        .main-container.both-collapsed {
            grid-template-columns: var(--sidebar-collapsed) 1fr var(--progress-collapsed);
        }

        .sidebar {
            grid-area: sidebar;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            height: 100%;
        }

        .sidebar.collapsed .agents-list,
        .sidebar.collapsed .system-info {
            display: none;
        }

        .sidebar.collapsed .sidebar-header {
            padding: 12px 8px;
            justify-content: center;
        }

        .sidebar.collapsed .sidebar-title {
            display: none;
        }

        .sidebar.collapsed .collapse-btn {
            writing-mode: vertical-lr;
            text-orientation: mixed;
        }

        .progress-panel.collapsed .collapse-btn {
            writing-mode: vertical-lr;
            text-orientation: mixed;
            transform: rotate(180deg);
        }

        .sidebar-header {
            padding: 12px 16px;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            min-height: 48px;
        }

        .sidebar-title {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
        }

        .collapse-btn {
            background: none;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            padding: 4px;
            border-radius: 4px;
            transition: all 0.2s;
        }

        .collapse-btn:hover {
            background: var(--bg-primary);
            color: var(--text-primary);
        }

        /* FIXED: Chat area properly uses grid area and expands */
        .chat-area {
            grid-area: chat;
            display: flex;
            flex-direction: column;
            background: var(--bg-primary);
            min-height: 0;
            height: 100%;
            overflow: hidden;
        }

        /* Updated Progress Panel */
        .progress-panel {
            grid-area: progress;
            background: var(--bg-secondary);
            border-left: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            height: 100%;
        }

        .progress-panel.collapsed .panel-content {
            display: none;
        }

        .progress-panel.collapsed .progress-header {
            padding: 12px 8px;
            justify-content: center;
            writing-mode: vertical-lr;
            text-orientation: mixed;
        }

        .progress-panel.collapsed .progress-header span {
            transform: rotate(180deg);
        }

        .progress-header {
            padding: 12px 16px;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-weight: 600;
            font-size: 14px;
            min-height: 48px;
        }

        /* ADD to existing CSS */
        .progress-item.llm_call {
            border-left: 3px solid var(--accent-purple);
        }

        .progress-item.llm_call.latest {
            border-left: 3px solid var(--accent-purple);
            background: rgba(165, 165, 245, 0.03);
        }

        .progress-item.llm_call .progress-icon {
            color: var(--accent-purple);
        }

        .progress-summary {
            padding: 0 12px 8px 36px;
            font-size: 10px;
            color: var(--text-secondary);
            line-height: 1.3;
        }

        .event-field-value.json {
            background: var(--bg-primary);
            border-radius: 4px;
            padding: 8px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 10px;
            text-align: left;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid var(--border-color);
            word-break: break-all;
        }

        .expanded-section {
            margin-bottom: 12px;
            border-bottom: 1px solid rgba(48, 54, 61, 0.3);
            padding-bottom: 8px;
        }

        .expanded-section:last-child {
            border-bottom: none;
            margin-bottom: 0;
        }

        /* FIXED: Hide mobile tabs on desktop by default */
        .mobile-tabs {
            display: none;
        }

        /* Mobile Responsive */
        @media (max-width: 768px) {
            .main-container {
                display: flex !important;
                flex-direction: column;
                height: 100%;
                grid-template-areas: none;
                grid-template-columns: none;
            }

            .mobile-tabs {
                display: flex;
                background: var(--bg-tertiary);
                border-bottom: 1px solid var(--border-color);
                flex-shrink: 0;
            }

            .header-controls {
                display: none;
            }

            .mobile-tab {
                flex: 1;
                padding: 12px;
                text-align: center;
                background: var(--bg-secondary);
                border-right: 1px solid var(--border-color);
                cursor: pointer;
                transition: all 0.2s;
                font-size: 14px;
            }

            .mobile-tab:last-child {
                border-right: none;
            }

            .mobile-tab.active {
                background: var(--accent-blue);
                color: white;
            }

            .sidebar,
            .progress-panel {
                flex: 1;
                border-right: none;
                border-left: none;
                border-bottom: 1px solid var(--border-color);
                min-height: 0;
                max-height: none;
            }

            .chat-area {
                flex: 1;
                min-height: 0;
            }

            .sidebar,
            .chat-area,
            .progress-panel {
                display: none;
            }
        }

        @media (min-width: 769px) {
            .main-container {
                display: grid !important;
            }

            .sidebar,
            .chat-area,
            .progress-panel {
                display: flex !important;
                height: 100%;
            }
        }

        .agents-list {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            min-height: 0;
        }

        .agents-header {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .agent-item {
            padding: 12px;
            margin-bottom: 8px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .agent-item:hover {
            border-color: var(--accent-blue);
            transform: translateY(-1px);
        }

        .agent-item.active {
            border-color: var(--accent-blue);
            background: rgba(88, 166, 255, 0.1);
        }

        .agent-name {
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 4px;
        }

        .agent-description {
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 6px;
        }

        .agent-status {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
        }

        .agent-status.online { color: var(--accent-green); }
        .agent-status.offline { color: var(--accent-red); }

        .chat-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
            background: var(--bg-tertiary);
            flex-shrink: 0;
        }

        .chat-title {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 4px;
        }

        .chat-subtitle {
            font-size: 12px;
            color: var(--text-muted);
        }

        .messages-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            min-height: 0;
        }

        .message {
            display: flex;
            gap: 12px;
            max-width: 85%;
        }

        .message.user {
            flex-direction: row-reverse;
            margin-left: auto;
        }

        .message-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: 600;
            flex-shrink: 0;
        }

        .message.user .message-avatar {
            background: var(--accent-blue);
            color: white;
        }

        .message.agent .message-avatar {
            background: var(--accent-green);
            color: white;
        }

        .message-content {
            padding: 12px 16px;
            border-radius: 16px;
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
            color: var(--text-primary);
        }

        /* NEW: Thinking step styles */
        .thinking-step {
            background: var(--bg-secondary);
            border: 1px solid var(--accent-purple);
            border-radius: 12px;
            padding: 12px 16px;
            margin: 8px 0;
            font-size: 13px;
            color: var(--text-secondary);
        }

        .thinking-step.outline-step {
            border-color: var(--accent-cyan);
            background: rgba(57, 208, 216, 0.05);
        }

        .thinking-step-header {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 600;
            margin-bottom: 6px;
            color: var(--text-primary);
        }

        .thinking-step-content {
            line-height: 1.4;
        }

        .message-input {
            border-top: 1px solid var(--border-color);
            padding: 16px 20px;
            display: flex;
            gap: 12px;
            flex-shrink: 0;
            background: var(--bg-secondary);
        }

        .input-field {
            flex: 1;
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
            color: var(--text-primary);
            font-size: 14px;
        }

        .input-field:focus {
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

        .panel-header {
            padding: 16px;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-color);
            font-weight: 600;
            font-size: 14px;
        }

        .panel-content {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            min-height: 0;
        }

        .progress-section {
            margin-bottom: 20px;
        }

        .section-title {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 8px;
            letter-spacing: 0.5px;
        }

        /* NEW: Enhanced progress item styles */
        .progress-item {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 8px;
            font-size: 12px;
            transition: all 0.2s;
        }

        .progress-item:hover {
            border-color: var(--accent-blue);
            transform: translateY(-1px);
        }

        .progress-item-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
        }

        .progress-icon {
            width: 16px;
            text-align: center;
            font-size: 14px;
        }

        .progress-title {
            font-weight: 500;
            color: var(--text-primary);
            flex: 1;
        }

        .progress-status {
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: 500;
        }

        .progress-status.running {
            background: var(--accent-orange);
            color: white;
        }

        .progress-status.completed {
            background: var(--accent-green);
            color: white;
        }

        .progress-status.error {
            background: var(--accent-red);
            color: white;
        }

        .progress-status.starting {
            background: var(--accent-cyan);
            color: white;
        }

        .progress-details {
            color: var(--text-secondary);
            font-size: 11px;
            line-height: 1.3;
        }

        .performance-metrics {
            background: rgba(88, 166, 255, 0.05);
            border: 1px solid rgba(88, 166, 255, 0.2);
            border-radius: 6px;
            padding: 8px;
            margin-top: 6px;
            font-size: 10px;
        }

        .performance-metrics .metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 2px;
        }

        .no-agent-selected {
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            gap: 16px;
            height: 100%;
            color: var(--text-muted);
            text-align: center;
        }

        .no-agent-selected .icon {
            font-size: 48px;
            opacity: 0.5;
        }

        .typing-indicator {
            display: none;
            align-items: center;
            gap: 8px;
            padding: 12px 16px;
            background: var(--bg-tertiary);
            margin: 12px 20px;
            border-radius: 16px;
            font-size: 14px;
            color: var(--text-muted);
            flex-shrink: 0;
        }

        .typing-indicator.active { display: flex; }

        .typing-dots {
            display: flex;
            gap: 4px;
        }

        .typing-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--text-muted);
            animation: typing 1.4s infinite;
        }

        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typing {
            0%, 60%, 100% { opacity: 0.3; }
            30% { opacity: 1; }
        }

        .system-info {
            margin-top: auto;
            padding: 12px;
            border-top: 1px solid var(--border-color);
            font-size: 11px;
            color: var(--text-muted);
            flex-shrink: 0;
        }

        .error-message {
            background: rgba(248, 81, 73, 0.1);
            border: 1px solid var(--accent-red);
            color: var(--accent-red);
            padding: 12px;
            border-radius: 6px;
            margin: 12px;
            font-size: 14px;
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 2000;
            max-width: 300px;
        }

        .event-detail-modal {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 2000;
            padding: 20px;
        }

        .event-detail-modal.active {
            display: flex;
        }

        .event-detail-content {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            max-width: 800px;
            max-height: 80vh;
            width: 100%;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: var(--shadow);
        }

        .event-detail-header {
            padding: 20px 24px;
            border-bottom: 1px solid var(--border-color);
            background: var(--bg-tertiary);
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-shrink: 0;
        }

        .event-detail-title {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .event-detail-close {
            background: none;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            padding: 8px;
            border-radius: 6px;
            font-size: 20px;
            transition: all 0.2s;
        }

        .event-detail-close:hover {
            background: var(--bg-primary);
            color: var(--text-primary);
        }

        .event-detail-body {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            min-height: 0;
        }

        .event-section {
            margin-bottom: 24px;
        }

        .event-section-title {
            font-size: 14px;
            font-weight: 600;
            color: var(--accent-blue);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
            padding-bottom: 6px;
            border-bottom: 1px solid var(--border-color);
        }

        .event-field {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 8px 0;
            border-bottom: 1px solid rgba(48, 54, 61, 0.5);
            font-size: 14px;
        }

        .event-field:last-child {
            border-bottom: none;
        }

        .event-field-label {
            font-weight: 500;
            color: var(--text-secondary);
            min-width: 140px;
            flex-shrink: 0;
        }

        .event-field-value {
            color: var(--text-primary);
            flex: 1;
            text-align: right;
            word-break: break-word;
        }

        .event-field-value.json {
            background: var(--bg-primary);
            border-radius: 6px;
            padding: 8px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            text-align: left;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
        }

        .event-status-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }

        .event-status-badge.completed {
            background: var(--accent-green);
            color: white;
        }

        .event-status-badge.running {
            background: var(--accent-orange);
            color: white;
        }

        .event-status-badge.failed {
            background: var(--accent-red);
            color: white;
        }

        .progress-item {
            cursor: pointer;
            transition: all 0.2s;
        }

        .progress-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }

        .thinking-step {
            cursor: pointer;
            transition: all 0.2s;
        }

        .thinking-step:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        }
    </style>
</head>
<body>

<div class="api-key-modal" id="api-key-modal">
    <div class="api-key-content">
        <div class="api-key-title">🔐 Enter API Key</div>
        <div class="api-key-description">
            Please enter your API key to access the agent. You can find this key in your agent registration details.
        </div>
        <input type="text" class="api-key-input" id="api-key-input"
               placeholder="tbk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx">
        <button class="api-key-button" id="api-key-submit">Connect</button>
    </div>
</div>

<div class="header">
    <div class="logo">
        <span>🤖</span>
        <span>Agent Registry</span>
    </div>
    <div class="header-controls">
        <button class="panel-toggle active" id="sidebar-toggle">📋 Agents</button>
        <button class="panel-toggle active" id="progress-toggle">📊 Progress</button>
        <div class="status-indicator disconnected" id="connection-status">
            <div class="status-dot"></div>
            <span>Connecting...</span>
        </div>
    </div>
</div>

<div class="mobile-tabs">
    <div class="mobile-tab active" data-tab="chat">💬 Chat</div>
    <div class="mobile-tab" data-tab="agents">📋 Agents</div>
    <div class="mobile-tab" data-tab="progress">📊 Progress</div>
</div>

<div class="main-container">
    <!-- Agents Sidebar -->
    <div class="sidebar" id="sidebar">
        <div class="sidebar-header">
            <div class="sidebar-title">Available Agents</div>
            <button class="collapse-btn" id="sidebar-collapse">◀</button>
        </div>
        <div class="agents-list">
            <div id="agents-container">
                <div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 20px;">
                    Loading agents...
                </div>
            </div>
        </div>
        <div class="system-info">
            <div>Registry Server</div>
            <div id="server-info">ws://localhost:8080</div>
        </div>
    </div>

    <!-- Chat Area -->
    <div class="chat-area">
        <div class="chat-header">
            <div class="chat-title" id="chat-title">Select an Agent</div>
            <div class="chat-subtitle" id="chat-subtitle">Choose an agent from the sidebar to start chatting</div>
        </div>

        <div class="messages-container" id="messages-container">
            <div class="no-agent-selected">
                <div class="icon">💬</div>
                <div>Select an agent to start a conversation</div>
            </div>
        </div>

        <div class="typing-indicator" id="typing-indicator">
            <span>Agent is thinking</span>
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>

        <div class="message-input">
            <input type="text" class="input-field" id="message-input"
                   placeholder="Type your message..." disabled>
            <button class="send-button" id="send-button" disabled>Send</button>
        </div>
    </div>
    <!-- Progress Panel -->
    <div class="progress-panel" id="progress-panel">
        <div class="progress-header">
            <span>Live Progress</span>
            <button class="collapse-btn" id="progress-collapse">▶</button>
        </div>
        <div class="panel-content" id="progress-content">
            <div class="progress-section">
                <div class="section-title">Current Status</div>
                <div id="current-status">
                    <div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 20px;">
                        No active execution
                    </div>
                </div>
            </div>

            <div class="progress-section">
                <div class="section-title">Performance Metrics</div>
                <div id="performance-metrics">
                    <div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 10px;">
                        No metrics available
                    </div>
                </div>
            </div>

            <div class="progress-section">
                <div class="section-title">Meta Tools History</div>
                <div id="meta-tools-history">
                    <div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 10px;">
                        No meta-tool activity
                    </div>
                </div>
            </div>

            <div class="progress-section">
                <div class="section-title">System Events</div>
                <div id="system-events">
                    <div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 10px;">
                        System idle
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script unSave="true">



    class AgentRegistryUI {
        constructor() {
            this.ws = null;
            this.currentAgent = null;
            this.sessionId = 'ui_session_' + Math.random().toString(36).substr(2, 9);
            this.isConnected = false;
            this.reconnectAttempts = 0;
            this.apiKey = null;
            this.maxReconnectAttempts = 10;
            this.reconnectDelay = 1000;

            this.panelStates = {
                sidebar: true,
                progress: true,
                mobile: 'chat'
            };

            this.agents = new Map();
            this.currentExecution = null;

            // NEW: Enhanced progress tracking
            this.progressHistory = [];
            this.maxProgressHistory = 200;
            this.expandedProgressItem = null;
            this.currentPerformanceMetrics = null;
            this.currentOutline = null;

            this.elements = {
                connectionStatus: document.getElementById('connection-status'),
                agentsContainer: document.getElementById('agents-container'),
                chatTitle: document.getElementById('chat-title'),
                chatSubtitle: document.getElementById('chat-subtitle'),
                messagesContainer: document.getElementById('messages-container'),
                messageInput: document.getElementById('message-input'),
                sendButton: document.getElementById('send-button'),
                typingIndicator: document.getElementById('typing-indicator'),
                serverInfo: document.getElementById('server-info'),

                // API Key elements
                apiKeyModal: document.getElementById('api-key-modal'),
                apiKeyInput: document.getElementById('api-key-input'),
                apiKeySubmit: document.getElementById('api-key-submit'),

                // Panel control elements
                sidebarToggle: document.getElementById('sidebar-toggle'),
                progressToggle: document.getElementById('progress-toggle'),
                sidebarCollapse: document.getElementById('sidebar-collapse'),
                progressCollapse: document.getElementById('progress-collapse'),
                mainContainer: document.querySelector('.main-container'),
                sidebar: document.getElementById('sidebar'),
                progressPanel: document.getElementById('progress-panel'),
                progressContent: document.getElementById('progress-content')
            };

            // Enhanced cleanup timer
            setInterval(() => {
                if (this.isTyping && this.currentExecution) {
                    const timeSinceLastUpdate = Date.now() - this.currentExecution.lastUpdate;
                    if (timeSinceLastUpdate > 30000) {
                        console.log('🧹 Cleanup: Hiding stuck typing indicator');
                        this.showTypingIndicator(false);
                        this.currentExecution = null;
                        this.updateCurrentStatusToIdle();
                    }
                }
            }, 5000);

            this.init();
        }

        init() {
            this.setupEventListeners();
            this.setupPanelControls();
            this.initializeProgressPanel();
            this.showApiKeyModal();
        }

        // NEW: Initialize the refactored progress panel
        initializeProgressPanel() {
            if (this.elements.progressContent) {
                this.elements.progressContent.innerHTML = `
                <div class="progress-section metrics-section">
                    <div class="section-title expandable-section" onclick="window.agentUI.toggleSection('metrics')">
                        <span>📊 Performance Metrics</span>
                        <span class="section-toggle">▼</span>
                    </div>
                    <div class="section-content" id="performance-metrics">
                        <div class="no-data">No metrics available</div>
                    </div>
                </div>

                <div class="progress-section outline-section">
                    <div class="section-title expandable-section" onclick="window.agentUI.toggleSection('outline')">
                        <span>🗺️ Execution Outline & Context</span>
                        <span class="section-toggle">▼</span>
                    </div>
                    <div class="section-content" id="execution-outline">
                        <div class="no-data">No outline available</div>
                    </div>
                </div>

                <div class="progress-section status-history-section">
                    <div class="section-title expandable-section" onclick="window.agentUI.toggleSection('status')">
                        <span>⚡ Status & History</span>
                        <span class="section-toggle">▼</span>
                    </div>
                    <div class="section-content expanded" id="status-history">
                        <div class="no-data">No active execution</div>
                    </div>
                </div>
            `;
            }
        }

        // NEW: Toggle progress panel sections
        toggleSection(sectionName) {
            const section = document.querySelector(`.${sectionName}-section .section-content`);
            const toggle = document.querySelector(`.${sectionName}-section .section-toggle`);

            if (!section || !toggle) return;

            const isExpanded = section.classList.contains('expanded');

            if (isExpanded) {
                section.classList.remove('expanded');
                toggle.textContent = '▼';
            } else {
                section.classList.add('expanded');
                toggle.textContent = '▲';
            }
        }

        // REFACTORED: Main message handler with unified progress system
        handleWebSocketMessage(data) {
            console.log('WebSocket message received:', data);

            if (data.event === 'execution_progress') {
                const executionData = data.data;
                if (executionData && executionData.payload) {
                    this.handleUnifiedProgressEvent(executionData);
                }
                return;
            }

            if (data.request_id && data.payload) {
                this.handleUnifiedProgressEvent(data);
                return;
            }

            if (data.event) {
                this.handleRegistryEvent(data);
                return;
            }

            console.log('Unhandled message format:', data);
        }

        // NEW: Unified progress event handler
        // REPLACE the existing handleUnifiedProgressEvent method
        handleUnifiedProgressEvent(eventData) {
            const payload = eventData.payload;
            const eventType = payload.event_type;
            const isFinal = eventData.is_final;
            const requestId = eventData.request_id;

            console.log(`🎯 Processing Event: ${eventType}`, payload);

            // Handle final events
            if (isFinal || eventType === 'execution_complete' || payload.status === 'completed') {
                this.showTypingIndicator(false);

                const result = payload.metadata?.result || payload.result || payload.response || payload.output;
                if (result && typeof result === 'string' && result.trim()) {
                    this.addMessage('agent', result);
                }

                this.currentExecution = null;
                this.updateCurrentStatusToIdle();
                return;
            }

            // Initialize execution tracking
            if (!this.currentExecution) {
                this.currentExecution = {
                    requestId,
                    startTime: Date.now(),
                    events: [],
                    lastUpdate: Date.now()
                };
                this.showTypingIndicator(true);
            }

            // ADD: Store ALL events in progress history
            this.addToProgressHistory(payload);

            // Handle chat integration for important events
            this.handleChatIntegration(payload);

            // Update performance metrics
            this.updatePerformanceMetricsFromEvent(payload);

            // Update execution outline
            this.updateExecutionOutlineFromEvent(payload);

            // Refresh status history (shows all events)
            this.refreshStatusHistory();

            // Update current execution
            if (this.currentExecution) {
                this.currentExecution.events.push({...payload, timestamp: Date.now()});
                this.currentExecution.lastUpdate = Date.now();
            }
        }

        // NEW: Add event to progress history
        // UPDATE the addToProgressHistory method to ensure all events are captured
        addToProgressHistory(payload) {
        const irrelevantEventTypes = ['node_phase', 'node_enter']; // Fügen Sie hier weitere Typen hinzu, falls nötig

    // Prüfen, ob der Event-Typ in der Liste der irrelevanten Typen ist
    if (irrelevantEventTypes.includes(payload.event_type)) {
        // Optional: Hier könnte man das Event kurz an anderer Stelle anzeigen,
        // aber wir speichern es nicht im langfristigen Verlauf.
        console.log(`📝 Skipping storage for irrelevant event: ${payload.event_type}`);
        return; // Die Funktion hier beenden, um das Speichern zu verhindern
    }

            // Generate consistent ID for events
            const eventId = payload.event_id || `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

            const historyItem = {
                ...payload,
                timestamp: payload.timestamp || Date.now(),
                id: eventId
            };

            // Remove any existing event with same ID to avoid duplicates
            this.progressHistory = this.progressHistory.filter(item => item.id !== eventId);

            this.progressHistory.unshift(historyItem);

            if (this.progressHistory.length - 10 > this.maxProgressHistory) {
                this.progressHistory = this.progressHistory.slice(0, this.maxProgressHistory-50);
            }

            console.log(`📝 Added to progress history: ${payload.event_type}`, historyItem);
        }

        // NEW: Refresh unified status history display
        refreshStatusHistory() {
            const container = document.getElementById('status-history');
            if (!container) return;

            if (this.progressHistory.length === 0) {
                container.innerHTML = '<div class="no-data">No events recorded</div>';
                return;
            }

            container.innerHTML = '';

            this.progressHistory.forEach((event, index) => {
                const eventElement = this.createExpandableProgressItem(event, index === 0);
                container.appendChild(eventElement);
            });
        }

        // NEW: Create expandable progress item (only one expandable at a time)
// UPDATE the createExpandableProgressItem method to show more LLM details
        createExpandableProgressItem(event, isLatest = false) {
            const div = document.createElement('div');
            div.className = `progress-item expandable ${isLatest ? 'latest' : ''} ${event.event_type}`;
            div.setAttribute('data-event-id', event.id);

            const icon = this.getEventIcon(event.event_type, event.status);
            const title = this.getDisplayAction(event.event_type, event);
            const timestamp = new Date((event.timestamp || Date.now()) * (event.timestamp > 10000000000 ? 1 : 1000)).toLocaleTimeString();
            const status = event.status || 'unknown';

            // ADD: Special summary for LLM calls
            let summaryDetails = '';
            if (event.node_name) summaryDetails += `${event.node_name} • `;
            summaryDetails += timestamp;

            if (event.event_type === 'llm_call') {
                if (event.llm_temperature !== undefined) summaryDetails += ` • Temp: ${event.llm_temperature}`;
                if (event.llm_total_tokens) summaryDetails += ` • ${event.llm_total_tokens} tokens`;
                if (event.llm_cost) summaryDetails += ` • $${event.llm_cost.toFixed(4)}`;
                if (event.duration) summaryDetails += ` • ${event.duration.toFixed(2)}s`;
            } else {
                if (event.duration) summaryDetails += ` • ${event.duration.toFixed(2)}s`;
            }

            div.innerHTML = `
        <div class="progress-item-header" onclick="window.agentUI.toggleProgressItem('${event.id}')">
            <div class="progress-icon">${icon}</div>
            <div class="progress-title">${title}</div>
            <div class="progress-meta">
                <span class="progress-status ${status}">${status}</span>
                <span class="expand-indicator">▼</span>
            </div>
        </div>
        <div class="progress-summary">
            ${summaryDetails}
        </div>
        <div class="progress-item-expanded" id="expanded-${event.id}">
            ${this.createExpandedEventContent(event)}
        </div>
    `;

            return div;
        }
        // NEW: Toggle progress item (only one at a time)
        toggleProgressItem(eventId) {
            if (this.expandedProgressItem && this.expandedProgressItem !== eventId) {
                this.closeProgressItem(this.expandedProgressItem);
            }

            const expandedContent = document.getElementById(`expanded-${eventId}`);
            const progressItem = document.querySelector(`[data-event-id="${eventId}"]`);
            const indicator = progressItem?.querySelector('.expand-indicator');

            if (!expandedContent || !progressItem) return;

            const isExpanded = expandedContent.classList.contains('active');

            if (isExpanded) {
                expandedContent.classList.remove('active');
                progressItem.classList.remove('expanded');
                if (indicator) indicator.textContent = '▼';
                this.expandedProgressItem = null;
            } else {
                expandedContent.classList.add('active');
                progressItem.classList.add('expanded');
                if (indicator) indicator.textContent = '▲';
                this.expandedProgressItem = eventId;
            }
        }

        // NEW: Close progress item
        closeProgressItem(eventId) {
            const expandedContent = document.getElementById(`expanded-${eventId}`);
            const progressItem = document.querySelector(`[data-event-id="${eventId}"]`);
            const indicator = progressItem?.querySelector('.expand-indicator');

            if (expandedContent) expandedContent.classList.remove('active');
            if (progressItem) progressItem.classList.remove('expanded');
            if (indicator) indicator.textContent = '▼';
        }

        // NEW: Create detailed expanded content
// ADD this method to create comprehensive event details
        createExpandedEventContent(event) {
            const sections = [];

            // Core Information
            const coreInfo = this.extractCoreFields(event);
            if (Object.keys(coreInfo).length > 0) {
                sections.push(this.createEventSection('Core Information', coreInfo));
            }

            // Timing Information
            const timingInfo = this.extractTimingFields(event);
            if (Object.keys(timingInfo).length > 0) {
                sections.push(this.createEventSection('Timing & Status', timingInfo));
            }

            // LLM Information
            const llmInfo = this.extractLLMFields(event);
            if (Object.keys(llmInfo).length > 0) {
                sections.push(this.createEventSection('LLM Details', llmInfo));
            }

            // Tool Information
            const toolInfo = this.extractToolFields(event);
            if (Object.keys(toolInfo).length > 0) {
                sections.push(this.createEventSection('Tool Details', toolInfo));
            }

            // Performance Information
            const perfInfo = this.extractPerformanceFields(event);
            if (Object.keys(perfInfo).length > 0) {
                sections.push(this.createEventSection('Performance', perfInfo));
            }

            // Reasoning Context
            const reasoningInfo = this.extractReasoningFields(event);
            if (Object.keys(reasoningInfo).length > 0) {
                sections.push(this.createEventSection('Reasoning Context', reasoningInfo));
            }

            // Error Information
            const errorInfo = this.extractErrorFields(event);
            if (Object.keys(errorInfo).length > 0) {
                sections.push(this.createEventSection('Error Details', errorInfo));
            }

            // Raw Data
            const rawData = this.extractRawDataFields(event);
            if (Object.keys(rawData).length > 0) {
                sections.push(this.createEventSection('Raw Data', rawData));
            }

            return sections.join('') || '<div class="no-expanded-data">No detailed information available</div>';
        }

        // NEW: Create event section for expanded view
        createEventSection(title, fields) {
            const fieldsHtml = Object.entries(fields)
                .map(([key, value]) => {
                    if (typeof value === 'object' && value.type === 'json') {
                        return `
                        <div class="event-field">
                            <div class="event-field-label">${key}:</div>
                            <div class="event-field-value json">${value.value}</div>
                        </div>
                    `;
                    } else {
                        return `
                        <div class="event-field">
                            <div class="event-field-label">${key}:</div>
                            <div class="event-field-value">${value}</div>
                        </div>
                    `;
                    }
                })
                .join('');

            return `
            <div class="expanded-section">
                <div class="expanded-section-title">${title}</div>
                ${fieldsHtml}
            </div>
        `;
        }

        // ENHANCED: Chat integration for reasoning loops and task execution
        handleChatIntegration(payload) {
            const eventType = payload.event_type;
            const metadata = payload.metadata || {};
            switch (eventType) {
                case 'reasoning_loop':
                    if (metadata.outline_step && metadata.outline_total) {
                        this.handleOutlineStepInChat(payload);
                    }
                    break;
                case 'outline_created':
                    this.handleOutlineCreatedInChat(payload);
                    break;
                case 'task_start':
                case 'task_complete':
                case 'task_error':
                    this.handleTaskProgressInChat(payload);
                    break;
                case 'plan_created':
                    this.handlePlanCreatedInChat(payload);
                    break;
                case 'tool_call':
                    // Only show important tool calls in chat
                    if (payload.tool_name && !payload.tool_name.includes('internal')) {
                        this.handleToolCallInChat(payload);
                    }
                    break;
            }
        }

        // ADD this method for plan creation
handlePlanCreatedInChat(payload) {
    const metadata = payload.metadata || {};
    const planName = metadata.plan_name || 'Execution Plan';
    const taskCount = metadata.task_count || 0;
    const strategy = metadata.strategy || 'sequential';

    const planDiv = document.createElement('div');
    planDiv.className = 'thinking-step plan-created';
    planDiv.innerHTML = `
        <div class="thinking-step-header">
            <span>📋</span>
            <span>${planName} Created</span>
            <span class="step-status completed">Ready</span>
        </div>
        <div class="thinking-step-content">
            <div class="plan-details">
                <div class="plan-info">
                    <span class="plan-item">Tasks: ${taskCount}</span>
                    <span class="plan-item">Strategy: ${strategy}</span>
                </div>
                <div class="plan-ready">
                    <em>🚀 Plan ready for execution</em>
                </div>
            </div>
        </div>
    `;

    this.elements.messagesContainer.appendChild(planDiv);
    this.scrollToBottom();
}

        // ADD this new method for outline step progress
handleOutlineStepInChat(payload) {
    const metadata = payload.metadata || {};
    const outlineStep = metadata.outline_step || 0;
    const outlineTotal = metadata.outline_total || 0;
    const loopNumber = metadata.loop_number || 0;
    const status = payload.status || 'running';

    if (outlineStep === 0 || outlineTotal === 0) return;

    const progressPercentage = Math.round((outlineStep / outlineTotal) * 100);
    const isCompleted = status === 'completed';

    const stepDiv = document.createElement('div');
    stepDiv.className = `thinking-step outline-step ${isCompleted ? 'completed' : 'running'}`;

    let stepTitle = `Outline Step ${outlineStep} of ${outlineTotal}`;
    let stepIcon = isCompleted ? '✅' : '🗺️';
    let stepStatus = isCompleted ? 'Completed' : 'In Progress';

    stepDiv.innerHTML = `
        <div class="thinking-step-header">
            <span>${stepIcon}</span>
            <span>${stepTitle}</span>
            <span class="step-status ${status}">${stepStatus}</span>
        </div>
        <div class="thinking-step-content">
            <div class="outline-progress">
                <div class="progress-info">
                    <span class="progress-text">Execution Progress</span>
                    <span class="progress-percentage">${progressPercentage}%</span>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar">
                        <div class="progress-bar-fill" style="width: ${progressPercentage}%"></div>
                    </div>
                </div>
                ${isCompleted ?
                    '<div class="step-completed">This execution step is now complete</div>' :
                    '<div class="step-working">Working on this step...</div>'
                }
            </div>

            ${metadata.context_size || metadata.task_stack_size ? `
                <div class="context-info">
                    ${metadata.context_size ? `<span class="context-item">Context: ${metadata.context_size}</span>` : ''}
                    ${metadata.task_stack_size ? `<span class="context-item">Tasks: ${metadata.task_stack_size}</span>` : ''}
                </div>
            ` : ''}
        </div>
    `;

    this.elements.messagesContainer.appendChild(stepDiv);
    this.scrollToBottom();
}


        // NEW: Handle outline creation with detailed information
// REPLACE the existing handleOutlineCreatedInChat method
handleOutlineCreatedInChat(payload) {
    const metadata = payload.metadata || {};
    const outline = metadata.outline;

    if (!outline) return;

    const outlineDiv = document.createElement('div');
    outlineDiv.className = 'thinking-step outline-created';
    outlineDiv.innerHTML = `
        <div class="thinking-step-header">
            <span>📋</span>
            <span>Execution Plan Created</span>
            <span class="step-status completed">Ready</span>
        </div>
        <div class="thinking-step-content">
            <div class="outline-content">
                ${this.formatOutlineForChat(outline)}
            </div>
            <div class="outline-ready">
                <em>✨ Ready to execute plan step by step</em>
            </div>
        </div>
    `;

    this.elements.messagesContainer.appendChild(outlineDiv);
    this.scrollToBottom();
}

        // NEW: Handle task execution progress cleanly
        handleTaskProgressInChat(payload) {
            const eventType = payload.event_type;
            const taskId = payload.task_id;
            const metadata = payload.metadata || {};
            const description = metadata.description || 'Task execution';
            const taskType = metadata.type || 'Task';
            const priority = metadata.priority || 'normal';

            let icon = '📋';
            let status = '';
            let statusClass = 'running';

            if (eventType === 'task_start') {
                icon = '▶️';
                status = 'Starting';
                statusClass = 'starting';
            } else if (eventType === 'task_complete') {
                icon = '✅';
                status = 'Completed';
                statusClass = 'completed';
            } else if (eventType === 'task_error') {
                icon = '❌';
                status = 'Failed';
                statusClass = 'error';
            }

            const taskDiv = document.createElement('div');
            taskDiv.className = `thinking-step task-progress ${statusClass}`;
            taskDiv.innerHTML = `
            <div class="thinking-step-header">
                <span>${icon}</span>
                <span>${taskType} ${status}</span>
                <span class="priority-badge ${priority}">${priority}</span>
            </div>
            <div class="thinking-step-content">
                <div class="task-description">${description}</div>
                ${payload.duration ? `<div class="task-timing">Duration: ${payload.duration.toFixed(2)}s</div>` : ''}
                ${eventType === 'task_error' && payload.error_details?.message ?
                `<div class="task-error">Error: ${payload.error_details.message}</div>` : ''}
            </div>
        `;

            this.elements.messagesContainer.appendChild(taskDiv);
            this.scrollToBottom();
        }

        // NEW: Handle tool calls in chat
        handleToolCallInChat(payload) {
            const toolName = payload.tool_name;
            const status = payload.status;

            if (status === 'running') return; // Only show completed tool calls

            const toolDiv = document.createElement('div');
            toolDiv.className = `thinking-step tool-call ${status}`;
            toolDiv.innerHTML = `
            <div class="thinking-step-header">
                <span>🔧</span>
                <span>Used ${toolName}</span>
                <span class="tool-status ${status}">${status}</span>
            </div>
            <div class="thinking-step-content">
                <div class="tool-result">
                    ${status === 'completed' ? 'Tool executed successfully' : 'Tool execution failed'}
                    ${payload.duration ? ` in ${payload.duration.toFixed(2)}s` : ''}
                </div>
            </div>
        `;

            this.elements.messagesContainer.appendChild(toolDiv);
            this.scrollToBottom();
        }

        // NEW: Format outline for chat display
        formatOutlineForChat(outline) {
            if (typeof outline === 'string') {
                return `<div class="outline-text">${outline}</div>`;
            }

            if (Array.isArray(outline)) {
                return `
                <div class="outline-steps">
                    ${outline.map((step, index) =>
                    `<div class="outline-step">
                            <span class="step-number">${index + 1}.</span>
                            <span class="step-text">${step}</span>
                        </div>`
                ).join('')}
                </div>
            `;
            }

            return '<div class="outline-text">Execution plan created</div>';
        }

        // NEW: Create progress bar
        createProgressBar(current, total) {
            if (!total || total === 0) return '';

            const percentage = Math.round((current / total) * 100);

            return `
            <div class="progress-bar-container">
                <div class="progress-bar-info">
                    <span>Progress</span>
                    <span>${percentage}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-bar-fill" style="width: ${percentage}%"></div>
                </div>
            </div>
        `;
        }

        // ENHANCED: Update performance metrics
        updatePerformanceMetricsFromEvent(payload) {
            const metadata = payload.metadata || {};
            const performance = metadata.performance_metrics;

            if (performance && Object.keys(performance).length > 0) {
                this.currentPerformanceMetrics = performance;
                this.refreshPerformanceMetrics();
            }
        }

        // NEW: Refresh performance metrics display
        refreshPerformanceMetrics() {
            const container = document.getElementById('performance-metrics');
            if (!container || !this.currentPerformanceMetrics) return;

            const metrics = {
                'Action Efficiency': `${Math.round((this.currentPerformanceMetrics.action_efficiency || 0) * 100)}%`,
                'Avg Loop Time': `${(this.currentPerformanceMetrics.avg_loop_time || 0).toFixed(1)}s`,
                'Progress Rate': `${Math.round((this.currentPerformanceMetrics.progress_rate || 0) * 100)}%`,
                'Total Loops': this.currentPerformanceMetrics.total_loops || 0,
                'Progress Loops': this.currentPerformanceMetrics.progress_loops || 0
            };

            container.innerHTML = `
            <div class="metrics-grid">
                ${Object.entries(metrics).map(([key, value]) => `
                    <div class="metric-card">
                        <div class="metric-label">${key}</div>
                        <div class="metric-value">${value}</div>
                    </div>
                `).join('')}
            </div>
        `;
        }

        // NEW: Update execution outline
        updateExecutionOutlineFromEvent(payload) {
            const eventType = payload.event_type;
            const metadata = payload.metadata || {};

            if (eventType === 'outline_created' || eventType === 'reasoning_loop') {
                const outlineContainer = document.getElementById('execution-outline');
                if (!outlineContainer) return;

                const outline = metadata.outline;
                const outlineStep = metadata.outline_step || 0;
                const outlineTotal = metadata.outline_total || 0;
                const contextSize = metadata.context_size || 0;
                const taskStackSize = metadata.task_stack_size || 0;

                outlineContainer.innerHTML = `
                <div class="outline-info">
                    <div class="context-metrics">
                        <div class="context-metric">
                            <span class="context-label">Context Size:</span>
                            <span class="context-value">${contextSize}</span>
                        </div>
                        <div class="context-metric">
                            <span class="context-label">Task Stack:</span>
                            <span class="context-value">${taskStackSize}</span>
                        </div>
                        <div class="context-metric">
                            <span class="context-label">Progress:</span>
                            <span class="context-value">${outlineStep}/${outlineTotal}</span>
                        </div>
                    </div>

                    ${outlineTotal > 0 ? this.createProgressBar(outlineStep, outlineTotal) : ''}
                </div>

                ${outline ? `
                    <div class="outline-details">
                        <div class="outline-title">Current Plan</div>
                        ${this.formatOutlineForChat(outline)}
                    </div>
                ` : ''}
            `;
            }
        }

        // Helper methods for field extraction (using existing implementations)
        extractCoreFields(event) {
            const fields = {};
            if (event.event_type) fields['Event Type'] = event.event_type.replace(/_/g, ' ').toUpperCase();
            if (event.node_name) fields['Node'] = event.node_name;
            if (event.agent_name) fields['Agent'] = event.agent_name;
            if (event.task_id) fields['Task ID'] = event.task_id;
            if (event.plan_id) fields['Plan ID'] = event.plan_id;
            if (event.timestamp) fields['Timestamp'] = new Date((event.timestamp > 10000000000 ? event.timestamp : event.timestamp * 1000)).toLocaleString();
            return fields;
        }

// REPLACE the existing extractLLMFields method
        extractLLMFields(event) {
            const fields = {};
            const metadata = event.metadata || {};

            if (event.llm_model) fields['Model'] = event.llm_model;
            if (event.llm_temperature !== undefined) fields['Temperature'] = event.llm_temperature;
            if (event.llm_prompt_tokens) fields['Prompt Tokens'] = event.llm_prompt_tokens.toLocaleString();
            if (event.llm_completion_tokens) fields['Completion Tokens'] = event.llm_completion_tokens.toLocaleString();
            if (event.llm_total_tokens) fields['Total Tokens'] = event.llm_total_tokens.toLocaleString();
            if (event.llm_cost) fields['Cost'] = `$${event.llm_cost.toFixed(4)}`;

            // ADD: Model preferences and metadata
            if (metadata.model_preference) fields['Model Preference'] = metadata.model_preference;

            return fields;
        }

        extractToolFields(event) {
            const fields = {};
            const metadata = event.metadata || {};

            if (event.tool_name) fields['Tool Name'] = event.tool_name;
            if (metadata.meta_tool_name) fields['Meta Tool Name'] = metadata.meta_tool_name;

            if (event.is_meta_tool !== null && event.is_meta_tool !== undefined) {
                fields['Is Meta Tool'] = event.is_meta_tool ? '✅ Yes' : '❌ No';
            }

            // ADD: Tool execution details
            if (metadata.execution_phase) fields['Execution Phase'] = metadata.execution_phase;
            if (metadata.reasoning_loop) fields['Reasoning Loop'] = metadata.reasoning_loop;
            if (metadata.parsed_args && metadata.parsed_args.confidence_level) {
                fields['Confidence Level'] = `${Math.round(metadata.parsed_args.confidence_level * 100)}%`;
            }

            return fields;
        }

// ADD these helper methods for comprehensive data extraction
        extractTimingFields(event) {
            const fields = {};

            if (event.status) {
                fields['Status'] = `<span class="event-status-badge ${event.status}">${event.status.toUpperCase()}</span>`;
            }
            if (event.success !== null && event.success !== undefined) {
                fields['Success'] = event.success ? '✅ Yes' : '❌ No';
            }
            if (event.timestamp) {
                fields['Timestamp'] = new Date((event.timestamp > 10000000000 ? event.timestamp : event.timestamp * 1000)).toLocaleString();
            }
            if (event.duration) {
                fields['Duration'] = `${event.duration.toFixed(3)}s`;
            }
            if (event.node_duration) {
                fields['Node Duration'] = `${event.node_duration.toFixed(3)}s`;
            }
            if (event.routing_decision) {
                fields['Next Step'] = event.routing_decision;
            }

            return fields;
        }

        extractErrorFields(event) {
            const fields = {};

            if (event.error_details) {
                const errorDetails = event.error_details;
                if (errorDetails.message) fields['Error Message'] = errorDetails.message;
                if (errorDetails.type) fields['Error Type'] = errorDetails.type;
                if (errorDetails.traceback) {
                    fields['Traceback'] = {
                        type: 'json',
                        value: errorDetails.traceback
                    };
                }
            }

            if (event.tool_error) {
                fields['Tool Error'] = event.tool_error;
            }

            return fields;
        }
// REPLACE the existing extractRawDataFields method
        extractRawDataFields(event) {
            const fields = {};

            // ADD: Full LLM Input/Output for LLM calls
            if (event.event_type === 'llm_call') {
                if (event.llm_input) {
                    fields['LLM Input (Full Prompt)'] = {
                        type: 'json',
                        value: event.llm_input
                    };
                }

                if (event.llm_output) {
                    fields['LLM Output (Response)'] = {
                        type: 'json',
                        value: event.llm_output
                    };
                }
            }

            // Show other raw data for tool calls
            if (event.tool_args && typeof event.tool_args === 'object') {
                fields['Tool Arguments'] = {
                    type: 'json',
                    value: JSON.stringify(event.tool_args, null, 2)
                };
            }

            if (event.tool_result) {
                const resultStr = typeof event.tool_result === 'string' ?
                    event.tool_result :
                    JSON.stringify(event.tool_result, null, 2);

                fields['Tool Result'] = {
                    type: 'json',
                    value: resultStr.length > 1000 ?
                        resultStr.substring(0, 1000) + '\\n\\n... [truncated]' :
                        resultStr
                };
            }

            return fields;
        }

        extractPerformanceFields(event) {
            const fields = {};
            const metadata = event.metadata || {};
            const performance = metadata.performance_metrics || {};

            if (performance.action_efficiency) fields['Action Efficiency'] = `${Math.round(performance.action_efficiency * 100)}%`;
            if (performance.avg_loop_time) fields['Avg Loop Time'] = `${performance.avg_loop_time.toFixed(2)}s`;
            if (performance.progress_rate) fields['Progress Rate'] = `${Math.round(performance.progress_rate * 100)}%`;

            return fields;
        }

        extractReasoningFields(event) {
            const fields = {};
            const metadata = event.metadata || {};

            if (metadata.outline_step && metadata.outline_total) {
                fields['Outline Progress'] = `${metadata.outline_step}/${metadata.outline_total}`;
            }
            if (metadata.loop_number) fields['Loop Number'] = metadata.loop_number;
            if (metadata.context_size) fields['Context Size'] = metadata.context_size.toLocaleString();
            if (metadata.task_stack_size) fields['Task Stack Size'] = metadata.task_stack_size;

            return fields;
        }

        extractMetadata(event) {
            const fields = {};
            const metadata = event.metadata || {};

            // Show complex data as JSON
            const complexFields = ['tool_args', 'tool_result', 'llm_input', 'llm_output', 'error_details'];

            for (const field of complexFields) {
                if (event[field] && typeof event[field] === 'object') {
                    fields[field.replace(/_/g, ' ').toUpperCase()] = {
                        type: 'json',
                        value: JSON.stringify(event[field], null, 2)
                    };
                }
            }

            return fields;
        }

        // Enhanced helper methods
        // REPLACE the existing getDisplayAction method
        getDisplayAction(eventType, payload) {
            const metadata = payload.metadata || {};
            switch (eventType) {
                case 'reasoning_loop':
                    const step = metadata.outline_step || 0;
                    const total = metadata.outline_total || 0;
                    return step > 0 ? `Reasoning Step ${step}/${total}` : 'Deep Reasoning';
                case 'task_start':
                    return `Starting: ${metadata.description || 'Task'}`;
                case 'task_complete':
                    return `Completed: ${metadata.description || 'Task'}`;
                case 'task_error':
                    return `Failed: ${metadata.description || 'Task'}`;
                case 'tool_call':
                    const status = payload.status || 'running';
                    const toolName = payload.tool_name || 'Unknown Tool';
                    return `${status === 'running' ? 'Calling' : 'Called'} ${toolName}`;

                case 'llm_call':
                    const llmStatus = payload.status || 'running';
                    const model = payload.llm_model || 'LLM';
                    const taskId = payload.task_id || '';

                    // Show more context for LLM calls
                    let displayText = `${llmStatus === 'running' ? '🔄 Calling' : '✅ Called'} ${model}`;
                    if (taskId && taskId !== 'unknown') {
                        displayText += ` (${taskId})`;
                    }
                    return displayText;
                case 'plan_created':
                    return `Plan: ${metadata.plan_name || 'Execution Plan'}`;
                case 'outline_created':
                    return 'Execution Outline Created';
                case 'node_enter':
                    return `Started: ${payload.node_name || 'Processing'}`;
                case 'node_exit':
                    return `Finished: ${payload.node_name || 'Processing'}`;
                case 'node_phase':
                    return `${payload.node_name || 'Node'}: ${payload.node_phase || 'Processing'}`;
                case 'execution_start':
                    return 'Execution Started';
                case 'execution_complete':
                    return 'Execution Complete';
                // ADD: Meta tool events
                case 'meta_tool_call':
                    const metaToolName = metadata.meta_tool_name || payload.tool_name || 'Meta Tool';
                    const metaStatus = payload.status || 'running';
                    return `${metaStatus === 'running' ? 'Using' : 'Used'} ${metaToolName.replace(/_/g, ' ')}`;
                // ADD: Error events
                case 'error':
                    return `Error in ${payload.node_name || 'System'}`;
                default:
                    return eventType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            }
        }

        // REPLACE the existing getEventIcon method
        getEventIcon(eventType, status) {
            if (status === 'error' || status === 'failed') return '❌';
            if (status === 'completed') return '✅';

            switch (eventType) {
                case 'reasoning_loop': return '🧠';
                case 'task_start': return '▶️';
                case 'task_complete': return '✅';
                case 'task_error': return '❌';
                case 'tool_call': return '🔧';
                case 'llm_call': return '💭';
                case 'plan_created': return '📋';
                case 'outline_created': return '🗺️';
                case 'node_enter': return '🚀';
                case 'node_exit': return '🏁';
                case 'node_phase': return '⚙️';
                case 'execution_start': return '🎬';
                case 'execution_complete': return '🎉';
                case 'meta_tool_call': return '🛠️';
                case 'error': return '🚨';
                default: return '⚡';
            }
        }

        updateCurrentStatusToIdle() {
            const container = document.getElementById('status-history');
            if (container && container.children.length === 0) {
                container.innerHTML = `
                <div class="progress-item idle-status">
                    <div class="progress-item-header">
                        <div class="progress-icon">💤</div>
                        <div class="progress-title">Ready & Waiting</div>
                        <div class="progress-meta">
                            <span class="progress-status idle">idle</span>
                        </div>
                    </div>
                    <div class="progress-summary">
                        Agent ready for next message • ${new Date().toLocaleTimeString()}
                    </div>
                </div>
            `;
            }
        }

        showTypingIndicator(show) {
            console.log(`💭 ${show ? 'Showing' : 'Hiding'} typing indicator`);
            this.elements.typingIndicator.classList.toggle('active', show);
            if (show) {
                this.elements.typingIndicator.scrollIntoView({ behavior: 'smooth', block: 'end' });
            }
            this.isTyping = show;
        }

        scrollToBottom() {
            if (this.elements.messagesContainer) {
                this.elements.messagesContainer.scrollTop = this.elements.messagesContainer.scrollHeight;
            }
        }

        showApiKeyModal() {
            const storedKey = localStorage.getItem('agent_registry_api_key');
            if (storedKey) {
                this.apiKey = storedKey;
                this.elements.apiKeyModal.style.display = 'none';
                this.connect();
            } else {
                this.elements.apiKeyModal.style.display = 'flex';
            }
        }

        async validateAndStoreApiKey() {
            const apiKey = this.elements.apiKeyInput.value.trim();
            if (!apiKey) {
                this.showError('Please enter an API key');
                return;
            }

            if (!apiKey.startsWith('tbk_')) {
                this.showError('Invalid API key format (should start with tbk_)');
                return;
            }

            this.apiKey = apiKey;
            this.elements.apiKeyModal.style.display = 'none';
            this.connect();
        }

        setupPanelControls() {
            this.elements.sidebarToggle?.addEventListener('click', () => this.togglePanel('sidebar'));
            this.elements.progressToggle?.addEventListener('click', () => this.togglePanel('progress'));
            this.elements.sidebarCollapse?.addEventListener('click', () => this.togglePanel('sidebar'));
            this.elements.progressCollapse?.addEventListener('click', () => this.togglePanel('progress'));

            const mobileTabs = document.querySelectorAll('.mobile-tab');
            if (mobileTabs.length > 0) {
                mobileTabs.forEach(tab => {
                    tab.addEventListener('click', () => this.switchMobileTab(tab.dataset.tab));
                });
            }

            this.setupResponsiveHandlers();
        }

        togglePanel(panel) {
            this.panelStates[panel] = !this.panelStates[panel];
            this.updatePanelStates();
        }

        updatePanelStates() {
            const { sidebar, progress } = this.panelStates;

            if (this.elements.mainContainer) {
                this.elements.mainContainer.classList.remove('sidebar-collapsed', 'progress-collapsed', 'both-collapsed');
                if (!sidebar && !progress) {
                    this.elements.mainContainer.classList.add('both-collapsed');
                } else if (!sidebar) {
                    this.elements.mainContainer.classList.add('sidebar-collapsed');
                } else if (!progress) {
                    this.elements.mainContainer.classList.add('progress-collapsed');
                }
            }

            if (this.elements.sidebar) this.elements.sidebar.classList.toggle('collapsed', !sidebar);
            if (this.elements.progressPanel) this.elements.progressPanel.classList.toggle('collapsed', !progress);

            if (this.elements.sidebarToggle) {
                this.elements.sidebarToggle.classList.toggle('active', sidebar);
                this.elements.sidebarToggle.textContent = sidebar ? '📋 Agents' : '📋';
            }

            if (this.elements.progressToggle) {
                this.elements.progressToggle.classList.toggle('active', progress);
                this.elements.progressToggle.textContent = progress ? '📊 Progress' : '📊';
            }

            if (this.elements.sidebarCollapse) this.elements.sidebarCollapse.textContent = sidebar ? '◀' : '▶';
            if (this.elements.progressCollapse) this.elements.progressCollapse.textContent = progress ? '▶' : '◀';

            if (this.elements.mainContainer) this.elements.mainContainer.offsetHeight;
        }

        handleWindowResize() {
            const chatArea = document.querySelector('.chat-area');
            const mainContainer = this.elements.mainContainer;

            if (chatArea && mainContainer) {
                const currentDisplay = mainContainer.style.display;
                mainContainer.style.display = 'none';
                mainContainer.offsetHeight;
                mainContainer.style.display = currentDisplay || '';
            }
        }

        switchMobileTab(tab) {
            this.panelStates.mobile = tab;

            const mobileTabs = document.querySelectorAll('.mobile-tab');
            if (mobileTabs.length > 0) {
                mobileTabs.forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
            }

            const sidebarEl = document.querySelector('.sidebar');
            const chatAreaEl = document.querySelector('.chat-area');
            const progressPanelEl = document.querySelector('.progress-panel');

            if (sidebarEl) sidebarEl.style.display = tab === 'agents' ? 'flex' : 'none';
            if (chatAreaEl) chatAreaEl.style.display = tab === 'chat' ? 'flex' : 'none';
            if (progressPanelEl) progressPanelEl.style.display = tab === 'progress' ? 'flex' : 'none';
        }

        setupResponsiveHandlers() {
            const mediaQuery = window.matchMedia('(max-width: 768px)');
            const handleResponsive = (e) => {
                if (e.matches) {
                    this.switchMobileTab(this.panelStates.mobile);
                } else {
                    const panels = document.querySelectorAll('.sidebar, .chat-area, .progress-panel');
                    panels.forEach(panel => { if (panel) panel.style.display = ''; });
                }
            };

            if (mediaQuery.addEventListener) {
                mediaQuery.addEventListener('change', handleResponsive);
            } else {
                mediaQuery.addListener(handleResponsive);
            }
            handleResponsive(mediaQuery);
        }

        setupEventListeners() {
            this.elements.apiKeySubmit?.addEventListener('click', () => this.validateAndStoreApiKey());
            window.addEventListener('resize', () => this.handleWindowResize());
            this.elements.apiKeyInput?.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.validateAndStoreApiKey();
            });
            this.elements.sendButton.addEventListener('click', () => this.sendMessage());
            this.elements.messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey && this.currentAgent) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });

            document.addEventListener('visibilitychange', () => {
                if (!document.hidden && (!this.ws || this.ws.readyState === WebSocket.CLOSED)) {
                    this.connect();
                }
            });
        }

        connect() {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) return;

            this.updateConnectionStatus('connecting', 'Connecting...');

            try {
                const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
                const wsProtocol = isLocal ? 'ws' : 'wss';
                const wsUrl = `${wsProtocol}://${window.location.host}/ws/registry/ui_connect`;
                this.ws = new WebSocket(wsUrl);

                // Heartbeat mechanism
                this.heartbeatInterval = setInterval(() => {
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        this.ws.send(JSON.stringify({event: 'ping', data: {}}));
                    }
                }, 25000); // Send ping every 25 seconds

                this.ws.onopen = () => {
                    this.isConnected = true;
                    this.reconnectAttempts = 0;
                    this.updateConnectionStatus('connected', 'Connected');
                    console.log('Connected to Registry Server');
                };

                this.ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.event === 'pong') {
                            // Handle pong response
                            return;
                        }
                        this.handleWebSocketMessage(data);
                    } catch (error) {
                        console.error('Message parse error:', error);
                    }
                };

                this.ws.onclose = (event) => {
                    this.isConnected = false;
                    if (this.heartbeatInterval) {
                        clearInterval(this.heartbeatInterval);
                    }
                    this.updateConnectionStatus('disconnected', 'Disconnected');
                    this.scheduleReconnection();
                };

                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.updateConnectionStatus('error', 'Connection Error');
                };

            } catch (error) {
                console.error('Connection error:', error);
                this.updateConnectionStatus('error', 'Connection Failed');
                this.scheduleReconnection();
            }
        }

        scheduleReconnection() {
            if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                this.updateConnectionStatus('error', 'Connection Failed (Max attempts reached)');
                return;
            }

            this.reconnectAttempts++;
            const delay = Math.min(this.reconnectDelay * this.reconnectAttempts, 30000);

            this.updateConnectionStatus('connecting', `Reconnecting in ${delay/1000}s (attempt ${this.reconnectAttempts})`);

            setTimeout(() => {
                if (!this.isConnected) this.connect();
            }, delay);
        }

        updateConnectionStatus(status, text) {
            this.elements.connectionStatus.className = `status-indicator ${status}`;
            this.elements.connectionStatus.querySelector('span').textContent = text;
        }

        handleRegistryEvent(data) {
            const event = data.event;
            const payload = data.data || data;

            console.log(`📋 Registry Event: ${event}`, payload);

            switch (event) {
                case 'api_key_validation':
                    if (payload.valid) {
                        console.log('✅ API key validated successfully');
                    } else {
                        this.showError('❌ Invalid API key for this agent');
                        this.currentAgent = null;
                        this.elements.messageInput.disabled = true;
                        this.elements.sendButton.disabled = true;
                    }
                    break;
                case 'agents_list':
                    console.log('📝 Updating agents list:', payload.agents);
                    this.updateAgentsList(payload.agents);
                    break;
                case 'agent_registered':
                    console.log('🆕 Agent registered:', payload);
                    this.addAgent(payload);
                    break;
                case 'error':
                    console.error('❌ WebSocket error:', payload);
                    this.showError(payload.error || payload.message || 'Unknown error');
                    break;
                default:
                    console.log('❓ Unhandled registry event:', event, payload);
            }
        }

        updateAgentsList(agents) {
            this.elements.agentsContainer.innerHTML = '';

            if (!agents || agents.length === 0) {
                this.elements.agentsContainer.innerHTML = '<div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 20px;">No agents available</div>';
                return;
            }

            agents.forEach(agent => {
                this.agents.set(agent.public_agent_id, agent);
                const agentEl = this.createAgentElement(agent);
                this.elements.agentsContainer.appendChild(agentEl);
            });
        }

        createAgentElement(agent) {
            const div = document.createElement('div');
            div.className = 'agent-item';
            div.dataset.agentId = agent.public_agent_id;

            div.innerHTML = `
            <div class="agent-name">${agent.public_name}</div>
            <div class="agent-description">${agent.description || 'No description'}</div>
            <div class="agent-status ${agent.status}">
                <div class="status-dot"></div>
                <span>${agent.status.toUpperCase()}</span>
            </div>
        `;

            div.addEventListener('click', () => this.selectAgent(agent));
            return div;
        }

        selectAgent(agent) {
            if (!this.apiKey) {
                this.showError('Please set your API key first');
                return;
            }

            this.sendWebSocketMessage({
                event: 'validate_api_key',
                data: { public_agent_id: agent.public_agent_id, api_key: this.apiKey }
            });

            document.querySelectorAll('.agent-item').forEach(el => el.classList.remove('active'));
            document.querySelector(`[data-agent-id="${agent.public_agent_id}"]`)?.classList.add('active');

            this.currentAgent = agent;
            this.elements.chatTitle.textContent = agent.public_name;
            this.elements.chatSubtitle.textContent = agent.description || 'Ready for conversation';

            this.elements.messageInput.disabled = false;
            this.elements.sendButton.disabled = false;

            this.elements.messagesContainer.innerHTML = '';
            this.addMessage('agent', `Hello! I'm ${agent.public_name}. How can I help you?`);

            this.sendWebSocketMessage({
                event: 'subscribe_agent',
                data: { public_agent_id: agent.public_agent_id }
            });

            this.sendWebSocketMessage({
                event: 'get_agent_status',
                data: { public_agent_id: agent.public_agent_id }
            });

            // Reset progress panels
            this.progressHistory = [];
            this.refreshStatusHistory();
            const metricsContainer = document.getElementById('performance-metrics');
            if (metricsContainer) metricsContainer.innerHTML = '<div class="no-data">No metrics available</div>';
            const outlineContainer = document.getElementById('execution-outline');
            if (outlineContainer) outlineContainer.innerHTML = '<div class="no-data">No outline available</div>';
        }

        sendMessage() {
            if (!this.currentAgent || !this.elements.messageInput.value.trim()) return;

            const message = this.elements.messageInput.value.trim();
            this.addMessage('user', message);

            this.sendWebSocketMessage({
                event: 'chat_message',
                data: {
                    public_agent_id: this.currentAgent.public_agent_id,
                    message: message,
                    session_id: this.sessionId,
                    api_key: this.apiKey
                }
            });

            this.elements.messageInput.value = '';

            // Reset progress state
            this.progressHistory = [];
            this.expandedProgressItem = null;
            this.refreshStatusHistory();

            // Failsafe timeout
            setTimeout(() => {
                if (this.currentExecution) {
                    console.log('⏰ Timeout: Hiding typing indicator and resetting execution state');
                    this.showTypingIndicator(false);
                    this.currentExecution = null;
                    this.updateCurrentStatusToIdle();
                    this.showError('Agent response timeout - please try again');
                }
            }, 60000);
        }

        addMessage(sender, content) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', sender);

            const avatar = document.createElement('div');
            avatar.classList.add('message-avatar');
            avatar.textContent = sender === 'user' ? 'U' : 'AI';

            const contentDiv = document.createElement('div');
            contentDiv.classList.add('message-content');

            if (sender === 'agent' && window.marked) {
                try {
                    contentDiv.innerHTML = marked.parse(content);
                } catch (error) {
                    contentDiv.textContent = content;
                }
            } else {
                contentDiv.textContent = content;
            }

            messageDiv.appendChild(avatar);
            messageDiv.appendChild(contentDiv);

            this.elements.messagesContainer.appendChild(messageDiv);
            this.elements.messagesContainer.scrollTop = this.elements.messagesContainer.scrollHeight;

            if (sender === 'agent') {
                this.showTypingIndicator(false);
                setTimeout(() => {
                    if (this.currentExecution) {
                        this.currentExecution = null;
                        this.updateCurrentStatusToIdle();
                    }
                }, 1000);
            }
        }

        showError(message) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.textContent = message;

            document.body.appendChild(errorDiv);
            setTimeout(() => {
                if (errorDiv.parentNode) {
                    errorDiv.parentNode.removeChild(errorDiv);
                }
            }, 5000);
        }

        sendWebSocketMessage(data) {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify(data));
            } else {
                console.warn('WebSocket not connected, cannot send message');
            }
        }

    }

    // Initialize UI when DOM is ready
    if (!window.TB) {
        document.addEventListener('DOMContentLoaded', () => {
            window.agentUI = new AgentRegistryUI();
        });
    } else {
        TB.once(() => {
            window.agentUI = new AgentRegistryUI();
        });
    }
</script>
</body>
</html>"""


class AgentRequestHandlerV0(BaseHTTPRequestHandler):
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


# Create custom request handler class with agent reference
class AgentRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, isaa_module, agent_id,agent, *args, **kwargs):
        self.agent_id = agent_id
        self.agent = agent
        self.isaa_module = isaa_module
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/' or self.path == '/ui':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            html_content = self.isaa_module._get_standalone_agent_ui_html_0(self.agent_id)
            self.wfile.write(html_content.encode('utf-8'))

        elif self.path.startswith('/ws'):
            # WebSocket upgrade handling (simplified)
            self.send_response(101)
            self.send_header('Upgrade', 'websocket')
            self.send_header('Connection', 'Upgrade')
            self.end_headers()
            # Note: Full WebSocket implementation would require additional libraries

        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Not Found')

    def do_POST(self):
        if self.path == '/api/run':
            self._handle_api_run()
        elif self.path == '/api/reset':
            self._handle_api_reset()
        elif self.path == '/api/status':
            self._handle_api_status()
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"error": "Endpoint not found"}')

    def _handle_api_run(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            query = data.get('query', '')
            session_id = data.get('session_id', f'standalone_{secrets.token_hex(8)}')

            # Run agent synchronously in thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result = loop.run_until_complete(self.agent.a_run(query, session_id=session_id))
                response = {
                    'success': True,
                    'result': result,
                    'session_id': session_id,
                    'agent_id': self.agent_id
                }
                self._send_json_response(200, response)

            except Exception as e:
                error_response = {
                    'success': False,
                    'error': str(e),
                    'session_id': session_id,
                    'agent_id': self.agent_id
                }
                self._send_json_response(500, error_response)
            finally:
                loop.close()

        except Exception as e:
            error_response = {'success': False, 'error': f'Request processing error: {str(e)}'}
            self._send_json_response(400, error_response)

    def _handle_api_reset(self):
        try:
            if hasattr(self.agent, 'clear_context'):
                self.agent.clear_context()
                response = {'success': True, 'message': 'Context reset successfully'}
            else:
                response = {'success': False, 'error': 'Agent does not support context reset'}

            self._send_json_response(200, response)

        except Exception as e:
            error_response = {'success': False, 'error': str(e)}
            self._send_json_response(500, error_response)

    def _handle_api_status(self):
        try:
            status_info = {
                'agent_id': self.agent_id,
                'agent_name': getattr(self.agent, 'name', 'Unknown'),
                'status': 'active',
                'uptime': time.time(),
                'server_type': 'standalone'
            }
            self._send_json_response(200, status_info)

        except Exception as e:
            error_response = {'success': False, 'error': str(e)}
            self._send_json_response(500, error_response)

    def _send_json_response(self, status_code: int, data: dict):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_OPTIONS(self):
        # Handle CORS preflight
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

    def log_message(self, format, *args):
        # Suppress default HTTP server logs or redirect to app logger
        self.isaa_module.app.print(f"HTTP {self.address_string()}: {format % args}")





