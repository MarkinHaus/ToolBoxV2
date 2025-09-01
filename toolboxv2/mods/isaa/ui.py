# toolboxv2/mods/isaa/ui.py
import secrets

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
            --progress-width: 400px;
            --sidebar-collapsed: 60px;
            --progress-collapsed: 60px;
        }

        @media (max-width: 1200px) {
            :root {
                --sidebar-width: 250px;
                --progress-width: 350px;
            }
        }

        @media (max-width: 1024px) {
            :root {
                --sidebar-width: 220px;
                --progress-width: 300px;
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
                this.progressData = new Map();
                this.currentExecution = null;

                this.elements = {
                    connectionStatus: document.getElementById('connection-status'),
                    agentsContainer: document.getElementById('agents-container'),
                    chatTitle: document.getElementById('chat-title'),
                    chatSubtitle: document.getElementById('chat-subtitle'),
                    messagesContainer: document.getElementById('messages-container'),
                    messageInput: document.getElementById('message-input'),
                    sendButton: document.getElementById('send-button'),
                    typingIndicator: document.getElementById('typing-indicator'),
                    currentStatus: document.getElementById('current-status'),
                    performanceMetrics: document.getElementById('performance-metrics'),
                    metaToolsHistory: document.getElementById('meta-tools-history'),
                    systemEvents: document.getElementById('system-events'),
                    serverInfo: document.getElementById('server-info'),

                    // API Key elements
                    apiKeyModal: document.getElementById('api-key-modal'),
                    apiKeyInput: document.getElementById('api-key-input'),
                    apiKeySubmit: document.getElementById('api-key-submit'),

                    // Panel control elements with fallbacks
                    sidebarToggle: document.getElementById('sidebar-toggle'),
                    progressToggle: document.getElementById('progress-toggle'),
                    sidebarCollapse: document.getElementById('sidebar-collapse'),
                    progressCollapse: document.getElementById('progress-collapse'),
                    mainContainer: document.querySelector('.main-container'),
                    sidebar: document.getElementById('sidebar'),
                    progressPanel: document.getElementById('progress-panel')
                };

                this.init();
            }

            init() {
                this.setupEventListeners();
                this.setupPanelControls();
                this.showApiKeyModal();
                // this.connect(); // TODO: Remove for production
            }

            showApiKeyModal() {
                // Check if API key is stored
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
                // localStorage.setItem('agent_registry_api_key', apiKey);
                this.elements.apiKeyModal.style.display = 'none';
                this.connect();
            }

            // Panel Controls Setup
            setupPanelControls() {
                this.elements.sidebarToggle?.addEventListener('click', () => {
                    this.togglePanel('sidebar');
                });

                this.elements.progressToggle?.addEventListener('click', () => {
                    this.togglePanel('progress');
                });

                this.elements.sidebarCollapse?.addEventListener('click', () => {
                    this.togglePanel('sidebar');
                });

                this.elements.progressCollapse?.addEventListener('click', () => {
                    this.togglePanel('progress');
                });

                const mobileTabs = document.querySelectorAll('.mobile-tab');
                if (mobileTabs.length > 0) {
                    mobileTabs.forEach(tab => {
                        tab.addEventListener('click', () => {
                            this.switchMobileTab(tab.dataset.tab);
                        });
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
                    this.elements.mainContainer.classList.remove(
                        'sidebar-collapsed',
                        'progress-collapsed',
                        'both-collapsed'
                    );

                    if (!sidebar && !progress) {
                        this.elements.mainContainer.classList.add('both-collapsed');
                    } else if (!sidebar) {
                        this.elements.mainContainer.classList.add('sidebar-collapsed');
                    } else if (!progress) {
                        this.elements.mainContainer.classList.add('progress-collapsed');
                    }
                }

                if (this.elements.sidebar) {
                    this.elements.sidebar.classList.toggle('collapsed', !sidebar);
                }

                if (this.elements.progressPanel) {
                    this.elements.progressPanel.classList.toggle('collapsed', !progress);
                }

                // Update toggle button states
                if (this.elements.sidebarToggle) {
                    this.elements.sidebarToggle.classList.toggle('active', sidebar);
                    this.elements.sidebarToggle.textContent = sidebar ? '📋 Agents' : '📋';
                }

                if (this.elements.progressToggle) {
                    this.elements.progressToggle.classList.toggle('active', progress);
                    this.elements.progressToggle.textContent = progress ? '📊 Progress' : '📊';
                }

                if (this.elements.sidebarCollapse) {
                    this.elements.sidebarCollapse.textContent = sidebar ? '◀' : '▶';
                }
                if (this.elements.progressCollapse) {
                    this.elements.progressCollapse.textContent = progress ? '▶' : '◀';
                }

                // Force layout recalculation
                if (this.elements.mainContainer) {
                    this.elements.mainContainer.offsetHeight; // Trigger reflow
                }
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
                    mobileTabs.forEach(t => {
                        t.classList.toggle('active', t.dataset.tab === tab);
                    });
                }

                const sidebarEl = document.querySelector('.sidebar');
                const chatAreaEl = document.querySelector('.chat-area');
                const progressPanelEl = document.querySelector('.progress-panel');

                if (sidebarEl) {
                    sidebarEl.style.display = tab === 'agents' ? 'flex' : 'none';
                }
                if (chatAreaEl) {
                    chatAreaEl.style.display = tab === 'chat' ? 'flex' : 'none';
                }
                if (progressPanelEl) {
                    progressPanelEl.style.display = tab === 'progress' ? 'flex' : 'none';
                }
            }

            setupResponsiveHandlers() {
                const mediaQuery = window.matchMedia('(max-width: 768px)');

                const handleResponsive = (e) => {
                    if (e.matches) {
                        this.switchMobileTab(this.panelStates.mobile);
                    } else {
                        const panels = document.querySelectorAll('.sidebar, .chat-area, .progress-panel');
                        panels.forEach(panel => {
                            if (panel) {
                                panel.style.display = '';
                            }
                        });
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
                this.elements.apiKeySubmit?.addEventListener('click', () => {
                    this.validateAndStoreApiKey();
                });

                window.addEventListener('resize', () => {
                    this.handleWindowResize();
                });

                this.elements.apiKeyInput?.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        this.validateAndStoreApiKey();
                    }
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
                    const wsUrl = `ws://${window.location.host}/ws/registry/ui_connect`;
                    this.ws = new WebSocket(wsUrl);

                    this.ws.onopen = () => {
                        this.isConnected = true;
                        this.reconnectAttempts = 0;
                        this.updateConnectionStatus('connected', 'Connected');
                        console.log('Connected to Registry Server');
                    };

                    this.ws.onmessage = (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            this.handleWebSocketMessage(data);
                        } catch (error) {
                            console.error('Message parse error:', error);
                        }
                    };

                    this.ws.onclose = () => {
                        this.isConnected = false;
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
                    if (!this.isConnected) {
                        this.connect();
                    }
                }, delay);
            }

            updateConnectionStatus(status, text) {
                this.elements.connectionStatus.className = `status-indicator ${status}`;
                this.elements.connectionStatus.querySelector('span').textContent = text;
            }

            // Replace the entire handleWebSocketMessage method and related handlers
handleWebSocketMessage(data) {
    console.log('WebSocket message received:', data);

    // FIXED: Handle execution_progress events properly
    if (data.event === 'execution_progress') {
        // Extract the nested execution data
        const executionData = data.data;
        if (executionData && executionData.payload) {
            this.handleAgentExecutionEvent(executionData);
        }
        return;
    }

    // Handle direct execution events (when data has request_id and payload directly)
    if (data.request_id && data.payload) {
        this.handleAgentExecutionEvent(data);
        return;
    }

    // Handle other registry events
    if (data.event) {
        this.handleRegistryEvent(data);
        return;
    }

    console.log('Unhandled message format:', data);
}

handleAgentExecutionEvent(eventData) {
    const payload = eventData.payload;
    const eventType = payload.event_type;
    const isFinal = eventData.is_final;
    const requestId = eventData.request_id;

    console.log(`🎯 Processing Event: ${eventType}`, payload);

    // FIXED: Handle final execution complete with proper result extraction
    if (isFinal) {
        // Extract result from metadata or other possible locations
        const result = payload.metadata?.result ||
                      payload.result ||
                      payload.response ||
                      payload.output;

        if (result && typeof result === 'string' && result.trim()) {
            this.addMessage('agent', result);
        }
        this.showTypingIndicator(false);
        this.currentExecution = null;
        this.updateCurrentStatusToIdle();
        return;
    }

    // Start execution tracking
    if (!this.currentExecution) {
        this.currentExecution = {
            requestId,
            startTime: Date.now(),
            events: [],
            lastUpdate: Date.now()
        };
        this.showTypingIndicator(true);
    }

    // Store event if we have active execution
    if (this.currentExecution) {
        this.currentExecution.events.push({
            ...payload,
            timestamp: Date.now()
        });
        this.currentExecution.lastUpdate = Date.now();
    }

    // FIXED: Route events to specific handlers with better error handling
    try {
        switch (eventType) {
            case 'reasoning_loop':
                this.handleReasoningLoop(payload);
                this.updateCurrentStatus(payload, '🧠 Reasoning');
                break;
            case 'meta_tool_call':
                this.handleMetaToolCall(payload);
                this.updateCurrentStatus(payload, '⚙️ Using Tool');
                break;
            case 'llm_call':
                this.handleLLMCall(payload);
                this.updateCurrentStatus(payload, '💭 AI Thinking');
                break;
            case 'node_phase':
                this.handleNodeEvent(payload);
                this.updateCurrentStatus(payload, '🔧 Processing Phase');
                break;
            case 'node_exit':
                this.handleNodeEvent(payload);
                this.updateCurrentStatus(payload, '✅ Completed Phase');
                break;
            case 'execution_start':
                this.updateCurrentStatus(payload, '🚀 Starting');
                break;
            case 'execution_complete':
                this.updateCurrentStatus(payload, '✅ Complete');
                break;
            default:
                // Still update status for unknown events
                this.updateCurrentStatus(payload, `⚡ ${eventType.replace(/_/g, ' ')}`);
                console.log('📝 Unhandled event type:', eventType, payload);
        }
    } catch (error) {
        console.error('❌ Error handling event:', error, payload);
        this.showError(`Event processing error: ${error.message}`);
    }
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
        case 'execution_progress':
            // This shouldn't happen anymore with the fixed routing above
            console.log('🔄 Legacy execution progress event:', payload);
            if (payload.payload) {
                this.handleAgentExecutionEvent(payload);
            }
            break;
        default:
            console.log('❓ Unhandled registry event:', event, payload);
    }
}

// FIXED: Enhanced reasoning loop handler
handleReasoningLoop(payload) {
    const metadata = payload.metadata || {};
    const loopNumber = metadata.loop_number || 0;
    const outlineStep = metadata.outline_step || 0;
    const outlineTotal = metadata.outline_total || 0;
    const performance = metadata.performance_metrics || {};
    const status = payload.status || 'running';

    console.log(`🧠 Reasoning Loop ${loopNumber}:`, metadata);

    // Show outline progress in chat
    if (outlineStep > 0 && status === 'running') {
        const stepDiv = document.createElement('div');
        stepDiv.className = 'thinking-step outline-step';
        stepDiv.innerHTML = `
            <div class="thinking-step-header">
                <span>🗺️</span>
                <span>Planning Step ${outlineStep} of ${outlineTotal}</span>
            </div>
            <div class="thinking-step-content">
                ${JSON.stringify(metadata, null, 2)}
            </div>
        `;
        this.elements.messagesContainer.appendChild(stepDiv);
        this.scrollToBottom();
    }

    // Update performance metrics in progress panel
    if (performance && Object.keys(performance).length > 0) {
        this.updatePerformanceMetrics(performance);
    }
}

// FIXED: Enhanced meta tool handler
handleMetaToolCall(payload) {
    const metadata = payload.metadata || {};
    const metaToolName = metadata.meta_tool_name || 'unknown_tool';
    const status = payload.status || 'running';
    const phase = metadata.execution_phase || 'unknown';

    console.log(`⚙️ Meta Tool: ${metaToolName} (${status})`, metadata);

    // FIXED: Show internal reasoning in chat with better formatting
    if (metaToolName === 'internal_reasoning' && status === 'completed') {
        const args = metadata.parsed_args || {};
        const thought = args.thought;
        const thoughtNumber = args.thought_number;
        const confidence = args.confidence_level || 0;

        if (thought) {
            const thinkingDiv = document.createElement('div');
            thinkingDiv.className = 'thinking-step';
            thinkingDiv.innerHTML = `
                <div class="thinking-step-header">
                    <span>🧠</span>
                    <span>Internal Reasoning ${thoughtNumber ? `#${thoughtNumber}` : ''}</span>
                    <span style="font-size: 10px; color: var(--text-muted);">${Math.round(confidence * 100)}% confidence</span>
                </div>
                <div class="thinking-step-content">
                    ${this.formatThinkingContent(thought)}
                </div>
            `;
            this.elements.messagesContainer.appendChild(thinkingDiv);
            this.scrollToBottom();
        }
    }

    // FIXED: Always update meta tools history
    this.updateMetaToolsHistory(payload);
}

// NEW: Format thinking content nicely
formatThinkingContent(thought) {
    if (!thought) return '';

    // Simple formatting for better readability
    return thought
        .replace(/\\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>');
}

// FIXED: Enhanced LLM call handler
handleLLMCall(payload) {
    const model = payload.llm_model || 'Unknown Model';
    const tokens = payload.llm_total_tokens || 0;
    const cost = payload.llm_cost || 0;
    const duration = payload.llm_duration || 0;
    const status = payload.status || 'running';

    console.log(`💭 LLM Call: ${model} (${status})`, { tokens, cost, duration });

    this.updateSystemEvents({
        type: 'llm_call',
        model,
        tokens,
        cost,
        duration,
        status,
        timestamp: Date.now()
    });
}

// FIXED: Enhanced node event handler
handleNodeEvent(payload) {
    const nodeName = payload.node_name || 'Unknown Node';
    const phase = payload.node_phase || 'processing';
    const status = payload.status || 'running';
    const duration = payload.node_duration || 0;
    const routingDecision = payload.routing_decision;

    console.log(`🔧 Node Event: ${nodeName}`, { phase, status, duration, routingDecision });

    this.updateSystemEvents({
        type: 'node_event',
        nodeName,
        phase,
        status,
        duration,
        routingDecision,
        timestamp: Date.now()
    });
}

// FIXED: Enhanced current status updates
updateCurrentStatus(payload, userFriendlyAction = null) {
    const eventType = payload.event_type || 'unknown';
    const status = payload.status || 'processing';
    const nodeName = payload.node_name || '';
    const timestamp = new Date().toLocaleTimeString();
    const agentName = payload.agent_name || 'Agent';

    // FIXED: Use user-friendly action or derive from event type
    const displayAction = userFriendlyAction || this.getDisplayAction(eventType, payload);
    const icon = this.getEventIcon(eventType, status);

    const statusDiv = document.createElement('div');
    statusDiv.className = 'progress-item';
    statusDiv.innerHTML = `
        <div class="progress-item-header">
            <div class="progress-icon">${icon}</div>
            <div class="progress-title">${displayAction}</div>
            <div class="progress-status ${status}">${status}</div>
        </div>
        <div class="progress-details">
            ${agentName} • ${timestamp}
            ${nodeName ? ` • ${nodeName}` : ''}
            ${payload.routing_decision ? ` • Route: ${payload.routing_decision}` : ''}
        </div>
        ${this.createProgressBar(payload)}
    `;

    // FIXED: Clear and update status
    this.elements.currentStatus.innerHTML = '';
    this.elements.currentStatus.appendChild(statusDiv);
}

// NEW: Create progress bars for certain events
createProgressBar(payload) {
    const metadata = payload.metadata || {};

    // For reasoning loops, show progress
    if (payload.event_type === 'reasoning_loop') {
        const outlineStep = metadata.outline_step || 0;
        const outlineTotal = metadata.outline_total || 0;

        if (outlineTotal > 0) {
            const percentage = Math.round((outlineStep / outlineTotal) * 100);
            return `
                <div style="margin-top: 8px;">
                    <div style="display: flex; justify-content: space-between; font-size: 10px; color: var(--text-muted); margin-bottom: 2px;">
                        <span>Progress</span>
                        <span>${percentage}%</span>
                    </div>
                    <div style="background: var(--bg-primary); border-radius: 4px; overflow: hidden; height: 4px;">
                        <div style="background: var(--accent-blue); height: 100%; width: ${percentage}%; transition: width 0.3s;"></div>
                    </div>
                </div>
            `;
        }
    }

    return '';
}

// FIXED: Enhanced performance metrics
updatePerformanceMetrics(performance) {
    console.log('📊 Updating Performance Metrics:', performance);

    const metricsDiv = document.createElement('div');
    metricsDiv.className = 'progress-item';

    const metrics = {
        'Action Efficiency': `${Math.round((performance.action_efficiency || 0) * 100)}%`,
        'Avg Loop Time': `${(performance.avg_loop_time || 0).toFixed(1)}s`,
        'Progress Rate': `${Math.round((performance.progress_rate || 0) * 100)}%`,
        'Total Loops': performance.total_loops || 0,
        'Progress Loops': performance.progress_loops || 0
    };

    const metricsHtml = Object.entries(metrics)
        .map(([key, value]) => `
            <div class="metric">
                <span>${key}:</span>
                <span style="font-weight: 600;">${value}</span>
            </div>
        `).join('');

    metricsDiv.innerHTML = `
        <div class="progress-item-header">
            <div class="progress-icon">📊</div>
            <div class="progress-title">Performance Metrics</div>
            <div class="progress-status running">live</div>
        </div>
        <div class="performance-metrics">
            ${metricsHtml}
        </div>
    `;

    this.elements.performanceMetrics.innerHTML = '';
    this.elements.performanceMetrics.appendChild(metricsDiv);
}

// FIXED: Enhanced meta tools history
updateMetaToolsHistory(payload) {
    const metadata = payload.metadata || {};
    const toolName = metadata.meta_tool_name || 'unknown_tool';
    const status = payload.status || 'running';
    const phase = metadata.execution_phase || 'unknown';
    const timestamp = new Date().toLocaleTimeString();

    console.log('🛠️ Updating Meta Tools History:', { toolName, status, phase });

    let icon = '⚙️';
    if (toolName.includes('reasoning')) icon = '🧠';
    else if (toolName.includes('delegate')) icon = '🎯';
    else if (toolName.includes('plan')) icon = '📋';
    else if (toolName.includes('variables')) icon = '💾';
    else if (toolName.includes('internal')) icon = '🔍';
    else if (status === 'completed') icon = '✅';
    else if (status === 'error') icon = '❌';

    const toolDiv = document.createElement('div');
    toolDiv.className = 'progress-item';

    // Add different styling based on tool type
    if (status === 'error') {
        toolDiv.style.borderColor = 'var(--accent-red)';
        toolDiv.style.backgroundColor = 'rgba(248, 81, 73, 0.05)';
    } else if (status === 'completed') {
        toolDiv.style.borderColor = 'var(--accent-green)';
    }

    toolDiv.innerHTML = `
        <div class="progress-item-header">
            <div class="progress-icon">${icon}</div>
            <div class="progress-title">${toolName.replace(/_/g, ' ')}</div>
            <div class="progress-status ${status}">${status}</div>
        </div>
        <div class="progress-details">
            Phase: ${phase} • ${timestamp}
            ${metadata.reasoning_loop ? ` • Loop: ${metadata.reasoning_loop}` : ''}
            ${metadata.parsed_args?.confidence_level ? ` • ${Math.round(metadata.parsed_args.confidence_level * 100)}% confidence` : ''}
        </div>
    `;

    this.elements.metaToolsHistory.insertBefore(toolDiv, this.elements.metaToolsHistory.firstChild);

    // Keep only last 10 items
    const items = this.elements.metaToolsHistory.children;
    while (items.length > 10) {
        this.elements.metaToolsHistory.removeChild(items[items.length - 1]);
    }
}

// FIXED: Enhanced system events
updateSystemEvents(eventData) {
    const timestamp = new Date(eventData.timestamp).toLocaleTimeString();

    console.log('🔔 Updating System Events:', eventData);

    let icon = '🔧';
    let title = 'System Event';
    let details = '';
    let statusClass = 'running';

    if (eventData.type === 'llm_call') {
        icon = '💬';
        title = `LLM: ${eventData.model}`;
        details = `${eventData.tokens} tokens`;
        if (eventData.cost > 0) {
            details += ` • $${eventData.cost.toFixed(4)}`;
        }
        if (eventData.duration > 0) {
            details += ` • ${eventData.duration.toFixed(2)}s`;
        }
        statusClass = eventData.status || 'running';
    } else if (eventData.type === 'node_event') {
        icon = eventData.status === 'completed' ? '✅' : '🔧';
        title = `${eventData.nodeName}`;
        details = eventData.phase;
        if (eventData.duration > 0) {
            details += ` • ${eventData.duration.toFixed(2)}s`;
        }
        if (eventData.routingDecision) {
            details += ` • → ${eventData.routingDecision}`;
        }
        statusClass = eventData.status || 'running';
    }

    const eventDiv = document.createElement('div');
    eventDiv.className = 'progress-item';
    eventDiv.innerHTML = `
        <div class="progress-item-header">
            <div class="progress-icon">${icon}</div>
            <div class="progress-title">${title}</div>
            <div class="progress-status ${statusClass}">${statusClass}</div>
        </div>
        <div class="progress-details">
            ${details}<br>
            ${timestamp}
        </div>
    `;

    this.elements.systemEvents.insertBefore(eventDiv, this.elements.systemEvents.firstChild);

    // Keep only last 8 events
    const items = this.elements.systemEvents.children;
    while (items.length > 8) {
        this.elements.systemEvents.removeChild(items[items.length - 1]);
    }
}

// NEW: Helper method to scroll messages to bottom
scrollToBottom() {
    if (this.elements.messagesContainer) {
        this.elements.messagesContainer.scrollTop = this.elements.messagesContainer.scrollHeight;
    }
}

// FIXED: Better display action generation
getDisplayAction(eventType, payload) {
    const metadata = payload.metadata || {};

    switch (eventType) {
        case 'reasoning_loop':
            const step = metadata.outline_step || 0;
            const total = metadata.outline_total || 0;
            return step > 0 ? `Planning Step ${step}/${total}` : 'Deep Reasoning';
        case 'meta_tool_call':
            const toolName = metadata.meta_tool_name || 'tool';
            return `${toolName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}`;
        case 'llm_call':
            const model = payload.llm_model || 'AI';
            return `${model} Thinking`;
        case 'node_phase':
            return `${payload.node_name || 'Processing'} • ${payload.node_phase || 'Phase'}`;
        case 'node_exit':
            return `${payload.node_name || 'Processing'} • Complete`;
        case 'execution_start':
            return 'Starting Execution';
        case 'execution_complete':
            return 'Execution Complete';
        default:
            return eventType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
}

// Enhanced idle status
updateCurrentStatusToIdle() {
    const timestamp = new Date().toLocaleTimeString();
    this.elements.currentStatus.innerHTML = `
        <div class="progress-item">
            <div class="progress-item-header">
                <div class="progress-icon">💤</div>
                <div class="progress-title">Ready & Waiting</div>
                <div class="progress-status completed">idle</div>
            </div>
            <div class="progress-details">
                Agent is ready for your next message • ${timestamp}
            </div>
        </div>
    `;
}

getEventIcon(eventType, status) {
    if (status === 'error') return '❌';
    if (status === 'completed') return '✅';

    switch (eventType) {
        case 'reasoning_loop': return '🧠';
        case 'meta_tool_call': return '⚙️';
        case 'llm_call': return '💭';
        case 'node_phase':
        case 'node_exit': return '🔧';
        case 'execution_complete': return '✅';
        default: return '⚡';
    }
}


            updateAgentsList(agents) {
                this.elements.agentsContainer.innerHTML = '';

                if (!agents || agents.length === 0) {
                    this.elements.agentsContainer.innerHTML = `
                        <div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 20px;">
                            No agents available
                        </div>
                    `;
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
                    data: {
                        public_agent_id: agent.public_agent_id,
                        api_key: this.apiKey
                    }
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

                // Reset progress panels
                this.elements.currentStatus.innerHTML = '<div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 20px;">Processing...</div>';
                this.elements.performanceMetrics.innerHTML = '<div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 10px;">Waiting for metrics...</div>';
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
            }

            showTypingIndicator(show) {
    this.elements.typingIndicator.classList.toggle('active', show);
    if (show) {
     this.elements.typingIndicator.display = 'flex';
        this.elements.typingIndicator.scrollIntoView({ behavior: 'smooth', block: 'end' });

    } else {
        this.elements.messagesContainer.scrollTop = this.elements.messagesContainer.scrollHeight;
        this.elements.typingIndicator.display = 'none';

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





