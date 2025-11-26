# talk.py
import asyncio
import base64
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field

from toolboxv2 import TBEF, App, MainTool, RequestData, Result, get_app
from toolboxv2.mods.isaa.extras.session import ChatSession
from toolboxv2.utils.extras.base_widget import get_current_user_from_request

# The ChatSession is central to maintaining conversation context with the agent.


# --- Constants ---
MOD_NAME = "talk"
VERSION = "1.0.0"
export = get_app(f"widgets.{MOD_NAME}").tb


# --- Session State Model ---
class TalkSession(BaseModel):
    """Represents the state of a single voice conversation session."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    chat_session: ChatSession
    event_queue: asyncio.Queue = Field(default_factory=asyncio.Queue, exclude=True)
    # Task to track the running agent process, preventing concurrent requests
    agent_task: asyncio.Task | None = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True


# --- Main Module Class ---
class Tools(MainTool):
    """
    The main class for the Talk module, handling initialization,
    session management, and dependency loading.
    """

    def __init__(self, app: App):
        # Initialize the MainTool with module-specific information
        self.version = VERSION
        self.name = MOD_NAME
        self.color = "CYAN"
        self.sessions: dict[str, TalkSession] = {}
        self.stt_func = None
        self.tts_func = None
        self.isaa_mod = None
        super().__init__(load=self.on_start, v=VERSION, name=MOD_NAME, tool={}, on_exit=self.on_exit)

    def on_start(self):
        """Initializes the Talk module, its dependencies (ISAA, AUDIO), and UI registration."""
        self.app.logger.info(f"Starting {self.name} v{self.version}...")

        # Get the ISAA module instance, which is a critical dependency
        self.isaa_mod = self.app.get_mod("isaa")
        if not self.isaa_mod:
            self.app.logger.error(
                f"{self.name}: ISAA module not found or failed to load. Voice assistant will not be functional.")
            return

        # Initialize STT and TTS services from the AUDIO module
        if hasattr(TBEF, "AUDIO") and self.app.get_mod("AUDIO"):
            self.stt_func = self.app.run_any(TBEF.AUDIO.STT_GENERATE, model="openai/whisper-small", row=True, device=0)
            self.tts_func = self.app.get_function(TBEF.AUDIO.SPEECH, state=False)[0]

            if self.stt_func and self.stt_func != "404":
                self.app.logger.info("Talk STT (whisper-small) is Online.")
            else:
                self.app.logger.warning("Talk STT function not available.")
                self.stt_func = None

            if self.tts_func and self.tts_func != "404":
                self.app.logger.info("Talk TTS function is Online.")
            else:
                self.app.logger.warning("Talk TTS function not available.")
                self.tts_func = None
        else:
            self.app.logger.warning("Talk module: AUDIO module features are not available or the module is not loaded.")

        if not all([self.stt_func, self.tts_func]):
            self.app.logger.error("Talk module cannot function without both STT and TTS services.")

        # Register the UI component with CloudM
        self.app.run_any(("CloudM", "add_ui"),
                         name=MOD_NAME, title="Voice Assistant", path=f"/api/{MOD_NAME}/ui",
                         description="Natural conversation with an AI assistant.", auth=True)
        self.app.logger.info(f"{self.name} UI registered with CloudM.")

    def on_exit(self):
        """Clean up resources, especially cancelling any active agent tasks."""
        for session in self.sessions.values():
            if session.agent_task and not session.agent_task.done():
                session.agent_task.cancel()
        self.app.logger.info(f"Closing {self.name} and cleaning up sessions.")


# --- Helper Function ---
async def _get_user_uid(app: App, request: RequestData) -> str | None:
    """Securely retrieves the user ID from the request context."""
    user = await get_current_user_from_request(app, request)
    return user.uid if user and hasattr(user, 'uid') and user.uid else None


# --- Core Agent Logic (Background Task) ---
async def _run_agent_and_respond(self: Tools, session: TalkSession, text: str, voice_params: dict):
    """
    The core logic for running the agent, handling callbacks, and generating responses.
    This function is designed to run as a background asyncio.Task.
    """
    queue = session.event_queue
    try:
        # Get the main agent from ISAA
        agent = await self.isaa_mod.get_agent("self")
        if not agent:
            raise RuntimeError("Could not retrieve 'self' agent from ISAA.")

        # Define callbacks to push live feedback to the client via the event queue
        async def tool_start_callback(tool_name: str, tool_input: Any):
            await queue.put({"event": "agent_thought", "data": f"Executing tool: {tool_name}..."})

        async def tool_end_callback(tool_output: Any):
            await queue.put({"event": "agent_thought", "data": "Tool execution finished."})

        # Set callbacks on the agent's tool executor if it exists
        if hasattr(agent, 'tool_executor') and agent.tool_executor:
            agent.tool_executor.start_callback = tool_start_callback
            agent.tool_executor.end_callback = tool_end_callback

        # Stream the LLM's text response chunk by chunk
        full_response = ""
        async for chunk in agent.a_stream(text, session_id=session.session_id):
            await queue.put({"event": "agent_response_chunk", "data": chunk})
            full_response += chunk

        # Generate audio from the complete response text
        if self.tts_func and full_response.strip():
            await queue.put({"event": "agent_thought", "data": "Generating audio..."})
            audio_data: bytes = self.tts_func(
                text=full_response,
                voice_index=voice_params.get('voice_index', 0),
                provider=voice_params.get('provider', 'piper'),
                config={'play_local': False, 'model_name': voice_params.get('model_name', 'ryan')},
                local=False, save=False
            )
            if audio_data:
                # Send audio as a base64 encoded string within a JSON payload
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                await queue.put({"event": "audio_playback", "data": {"format": "audio/mpeg", "content": audio_base64}})
            else:
                await queue.put({"event": "error", "data": "Failed to generate audio for the response."})

    except Exception as e:
        self.app.logger.error(f"Error in agent task for session {session.session_id}: {e}", exc_info=True)
        await queue.put({"event": "error", "data": f"An internal error occurred: {str(e)}"})
    finally:
        # Signal to the client that processing is complete
        await queue.put({"event": "processing_complete", "data": "Ready for next input."})
        session.agent_task = None  # Clear the task reference to allow new requests


# --- API Endpoints ---

@export(mod_name=MOD_NAME, api=True, name="start_session", api_methods=['POST'], request_as_kwarg=True)
async def api_start_session(self: Tools, request: RequestData) -> Result:
    """Creates a new talk session for an authenticated user."""
    user_id = await _get_user_uid(self.app, request)
    if not user_id:
        return Result.default_user_error(info="User authentication required.", exec_code=401)

    if not self.isaa_mod:
        return Result.default_internal_error(info="ISAA module is not available.")

    # Create a new ISAA ChatSession for conversation history
    chat_session = ChatSession(mem=self.isaa_mod.get_memory())
    session = TalkSession(user_id=user_id, chat_session=chat_session)
    self.sessions[session.session_id] = session

    self.app.logger.info(f"Started new talk session {session.session_id} for user {user_id}")
    return Result.json(data={"session_id": session.session_id})


@export(mod_name=MOD_NAME, api=True, name="stream", api_methods=['GET'], request_as_kwarg=True)
async def api_open_stream(self: Tools, request: RequestData, session_id: str) -> Result:
    """Opens a Server-Sent Events (SSE) stream for a given session ID."""
    if not session_id or session_id not in self.sessions:
        return Result.default_user_error(info="Invalid or expired session ID.", exec_code=404)

    session = self.sessions[session_id]
    queue = session.event_queue

    async def event_generator() -> AsyncGenerator[dict[str, Any], None]:
        self.app.logger.info(f"SSE stream opened for session {session_id}")
        await queue.put({"event": "connection_ready", "data": "Stream connected successfully."})
        try:
            while True:
                event_data = await queue.get()
                yield event_data
                queue.task_done()
        except asyncio.CancelledError:
            self.app.logger.info(f"SSE stream for session {session_id} cancelled by client.")
        finally:
            if session_id in self.sessions:
                if self.sessions[session_id].agent_task and not self.sessions[session_id].agent_task.done():
                    self.sessions[session_id].agent_task.cancel()
                del self.sessions[session_id]
                self.app.logger.info(f"Cleaned up and closed session {session_id}.")

    return Result.sse(stream_generator=event_generator())


@export(mod_name=MOD_NAME, api=True, name="process_audio", api_methods=['POST'], request_as_kwarg=True)
async def api_process_audio(self: Tools, request: RequestData, form_data: dict) -> Result:
    """Receives audio, transcribes it, and starts the agent processing task."""
    if not self.stt_func:
        return Result.default_internal_error(info="Speech-to-text service is not available.")

    session_id = form_data.get('session_id')
    audio_file_data = form_data.get('audio_blob')

    if not session_id or session_id not in self.sessions:
        return Result.default_user_error(info="Invalid or missing session_id.", exec_code=400)

    session = self.sessions[session_id]

    if session.agent_task and not session.agent_task.done():
        return Result.default_user_error(info="Already processing a previous request.", exec_code=429)

    if not audio_file_data or 'content_base64' not in audio_file_data:
        return Result.default_user_error(info="Audio data is missing or in the wrong format.", exec_code=400)

    try:
        audio_bytes = base64.b64decode(audio_file_data['content_base64'])
        transcription_result = self.stt_func(audio_bytes)
        transcribed_text = transcription_result.get('text', '').strip()

        if not transcribed_text:
            await session.event_queue.put({"event": "error", "data": "Could not understand audio. Please try again."})
            return Result.ok(data={"message": "Transcription was empty."})

        await session.event_queue.put({"event": "transcription_update", "data": transcribed_text})

        voice_params = {
            "voice_index": int(form_data.get('voice_index', '0')),
            "provider": form_data.get('provider', 'piper'),
            "model_name": form_data.get('model_name', 'ryan')
        }

        # Start the background task; the request returns immediately.
        session.agent_task = asyncio.create_task(
            _run_agent_and_respond(self, session, transcribed_text, voice_params)
        )
        return Result.ok(data={"message": "Audio received and processing started."})

    except Exception as e:
        self.app.logger.error(f"Error processing audio for session {session_id}: {e}", exc_info=True)
        return Result.default_internal_error(info=f"Failed to process audio: {str(e)}")


@export(mod_name=MOD_NAME, name="ui", api=True, api_methods=['GET'], request_as_kwarg=True)
def get_main_ui(self: Tools, request: RequestData) -> Result:
    """Serves the main HTML and JavaScript UI for the Talk widget."""
    html_content = """
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ToolBoxV2 - Voice Assistant</title>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
    <style>
        body { font-family: sans-serif; background-color: var(--theme-bg); color: var(--theme-text); display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
        .container { display: flex; flex-direction: column; align-items: center; justify-content: center; width: 100%; max-width: 600px; padding: 20px; text-align: center; }
        .visualizer { width: 250px; height: 250px; background-color: var(--glass-bg); border-radius: 50%; position: relative; overflow: hidden; border: 3px solid var(--theme-border); box-shadow: inset 0 0 15px rgba(0,0,0,0.2); transition: border-color 0.3s, box-shadow 0.3s; }
        .visualizer.recording { border-color: #ef4444; }
        .visualizer.thinking { border-color: #3b82f6; animation: pulse 2s infinite; }
        .visualizer.speaking { border-color: #22c55e; }
        .particle { position: absolute; width: 8px; height: 8px; background-color: var(--theme-primary); border-radius: 50%; pointer-events: none; transition: all 0.1s; }
        #micButton { margin-top: 30px; width: 80px; height: 80px; border-radius: 50%; border: none; background-color: var(--theme-primary); color: white; cursor: pointer; display: flex; justify-content: center; align-items: center; box-shadow: 0 4px 10px rgba(0,0,0,0.2); transition: background-color 0.2s, transform 0.1s; }
        #micButton:active { transform: scale(0.95); }
        #micButton:disabled { background-color: #9ca3af; cursor: not-allowed; }
        #micButton .material-symbols-outlined { font-size: 40px; }
        #statusText { margin-top: 20px; min-height: 50px; font-size: 1.2em; color: var(--theme-text-muted); line-height: 1.5; }
        @keyframes pulse { 0% { box-shadow: inset 0 0 15px rgba(0,0,0,0.2), 0 0 0 0 rgba(59, 130, 246, 0.7); } 70% { box-shadow: inset 0 0 15px rgba(0,0,0,0.2), 0 0 0 15px rgba(59, 130, 246, 0); } 100% { box-shadow: inset 0 0 15px rgba(0,0,0,0.2), 0 0 0 0 rgba(59, 130, 246, 0); } }
    </style>
</head>
<body>
    <div class="container">
        <div class="visualizer" id="visualizer"></div>
        <p id="statusText">Press the microphone to start</p>
        <button id="micButton"><span class="material-symbols-outlined">hourglass_empty</span></button>
        <div class="options" style="margin-top: 20px;">
            <label for="voiceSelect">Voice:</label>
            <select id="voiceSelect">
                <option value='{"provider": "piper", "model_name": "ryan", "voice_index": 0}'>Ryan (EN)</option>
                <option value='{"provider": "piper", "model_name": "kathleen", "voice_index": 0}'>Kathleen (EN)</option>
                <option value='{"provider": "piper", "model_name": "karlsson", "voice_index": 0}'>Karlsson (DE)</option>
            </select>
        </div>
    </div>
    <script unSave="true">
    function initTalk() {
        const visualizer = document.getElementById('visualizer');
        const micButton = document.getElementById('micButton');
        const statusText = document.getElementById('statusText');
        const voiceSelect = document.getElementById('voiceSelect');

        const state = { sessionId: null, sseConnection: null, mediaRecorder: null, audioChunks: [], isRecording: false, isProcessing: false, currentAudio: null };
        let audioContext, analyser, particles = [];

        function setStatus(text, mode = 'idle') {
            statusText.textContent = text;
            visualizer.className = 'visualizer ' + mode;
        }

        function createParticles(num = 50) {
            visualizer.innerHTML = ''; particles = [];
            for (let i = 0; i < num; i++) {
                const p = document.createElement('div'); p.classList.add('particle');
                visualizer.appendChild(p);
                particles.push({ element: p, angle: Math.random() * Math.PI * 2, radius: 50 + Math.random() * 50, speed: 0.01 + Math.random() * 0.02 });
            }
        }

        function animateVisualizer() {
            if (analyser) {
                const dataArray = new Uint8Array(analyser.frequencyBinCount);
                analyser.getByteFrequencyData(dataArray);
                let average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
                particles.forEach(p => {
                    p.angle += p.speed;
                    const scale = 1 + (average / 128);
                    p.element.style.transform = `translate(${Math.cos(p.angle) * p.radius * scale}px, ${Math.sin(p.angle) * p.radius * scale}px)`;
                });
            }
            requestAnimationFrame(animateVisualizer);
        }

        async function startSession() {
            if (state.sessionId) return;
            setStatus("Connecting...", 'thinking');
            micButton.disabled = true;
            try {
                const response = await TB.api.request('talk', 'start_session', {}, 'POST');
                if (response.error === 'none' && response.get()?.session_id) {
                    state.sessionId = response.get().session_id;
                    connectSse();
                } else {
                    setStatus(response.info?.help_text || "Failed to start session.", 'error');
                }
            } catch (e) {
                setStatus("Connection error.", 'error');
            }
        }

        function connectSse() {
            if (!state.sessionId) return;
            state.sseConnection = TB.sse.connect(`/sse/talk/stream?session_id=${state.sessionId}`, {
                onOpen: () => console.log("SSE Stream Open"),
                onError: () => setStatus("Connection lost.", 'error'),
                listeners: {
                    'connection_ready': (data) => { setStatus("Press the microphone to start"); micButton.disabled = false; micButton.innerHTML = '<span class="material-symbols-outlined">mic</span>'; },
                    'transcription_update': (data) => { setStatus(`“${data}”`, 'thinking'); state.isProcessing = true; },
                    'agent_thought': (data) => setStatus(data, 'thinking'),
                    'agent_response_chunk': (data) => { if (statusText.textContent.startsWith('“')) statusText.textContent = ""; statusText.textContent += data; },
                    'audio_playback': (data) => playAudio(data.content, data.format),
                    'processing_complete': (data) => { state.isProcessing = false; setStatus(data); micButton.disabled = false; micButton.innerHTML = '<span class="material-symbols-outlined">mic</span>'; },
                    'error': (data) => { state.isProcessing = false; setStatus(data, 'error'); micButton.disabled = false; micButton.innerHTML = '<span class="material-symbols-outlined">mic</span>'; }
                }
            });
        }

        async function playAudio(base64, format) {
            setStatus("...", 'speaking');
            const blob = await (await fetch(`data:${format};base64,${base64}`)).blob();
            const url = URL.createObjectURL(blob);
            if (state.currentAudio) state.currentAudio.pause();
            state.currentAudio = new Audio(url);

            if (!audioContext) audioContext = new AudioContext();
            const source = audioContext.createMediaElementSource(state.currentAudio);
            if (!analyser) { analyser = audioContext.createAnalyser(); analyser.fftSize = 64; }
            source.connect(analyser);
            analyser.connect(audioContext.destination);

            state.currentAudio.play();
            state.currentAudio.onended = () => { setStatus("Finished speaking."); URL.revokeObjectURL(url); };
        }

        async function toggleRecording() {
            if (state.isProcessing) return;
            if (!state.sessionId) { await startSession(); return; }

            if (state.isRecording) {
                state.mediaRecorder.stop();
                micButton.disabled = true;
                micButton.innerHTML = '<span class="material-symbols-outlined">hourglass_top</span>';
                setStatus("Processing...", 'thinking');
            } else {
                if (!state.mediaRecorder) {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });
                        if (!audioContext) audioContext = new AudioContext();
                        const source = audioContext.createMediaStreamSource(stream);
                        if (!analyser) { analyser = audioContext.createAnalyser(); analyser.fftSize = 64; }
                        source.connect(analyser);

                        state.mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
                        state.mediaRecorder.ondataavailable = e => state.audioChunks.push(e.data);
                        state.mediaRecorder.onstop = uploadAudio;
                    } catch (e) { setStatus("Could not access microphone.", 'error'); return; }
                }
                state.audioChunks = []; state.mediaRecorder.start(); state.isRecording = true;
                setStatus("Listening...", 'recording');
                micButton.innerHTML = '<span class="material-symbols-outlined">stop_circle</span>';
            }
        }

        async function uploadAudio() {
            state.isRecording = false; state.isProcessing = true;
            if (state.audioChunks.length === 0) { setStatus("No audio recorded."); state.isProcessing = false; micButton.disabled = false; micButton.innerHTML = '<span class="material-symbols-outlined">mic</span>'; return; }
            const audioBlob = new Blob(state.audioChunks, { type: 'audio/webm;codecs=opus' });

            const formData = new FormData();
            formData.append('session_id', state.sessionId);
            formData.append('audio_blob', audioBlob, 'recording.webm');

            const voiceParams = JSON.parse(voiceSelect.value);
            for (const key in voiceParams) formData.append(key, voiceParams[key]);

            try {
                const response = await TB.api.request('talk', 'process_audio', formData, 'POST');
                if (response.error !== 'none') {
                    setStatus(response.info?.help_text || "Failed to process audio.", 'error');
                    state.isProcessing = false; micButton.disabled = false; micButton.innerHTML = '<span class="material-symbols-outlined">mic</span>';
                }
            } catch(e) {
                 setStatus("Error sending audio.", 'error'); state.isProcessing = false; micButton.disabled = false; micButton.innerHTML = '<span class="material-symbols-outlined">mic</span>';
            }
        }

        micButton.addEventListener('click', toggleRecording);
        createParticles(); animateVisualizer();
        if (window.TB.isInitialized) startSession(); else window.TB.events.on('tbjs:initialized', startSession, { once: true });
    }
if (window.TB?.events) {
    if (window.TB.config?.get('appRootId')) { // A sign that TB.init might have run
         initTalk();
    } else {
        window.TB.events.on('tbjs:initialized', initTalk, { once: true });
    }
} else {
    // Fallback if TB is not even an object yet, very early load
    document.addEventListener('tbjs:initialized', initTalk, { once: true }); // Custom event dispatch from TB.init
}

    </script>
</body>
</html>"""
    return Result.html(data=html_content)
