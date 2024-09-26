import json
import os
import io
import subprocess
import tempfile
import threading
import wave

from starlette.responses import HTMLResponse

from toolboxv2 import get_app, App, TBEF, Spinner
from toolboxv2.mods.CloudM.AuthManager import on_start
from toolboxv2.tests.a_util import async_test
from toolboxv2.utils.extras.base_widget import get_spec, get_user_from_request
from functools import partial

from fastapi import WebSocket, Request
import asyncio

Name = 'talk'
export = get_app("cli_functions.Export").tb
default_export = export(mod_name=Name)
version = '0.0.1'
talk_generate, talk_tts = None, None


@export(mod_name=Name, version=version, initial=True)
def start(app=None):
    global talk_generate, talk_tts
    if app is None:
        app = get_app("Starting Talk interface")
    if not app.mod_online("audio"):
        app.print(f"TalK is starting AUDIO")
        audio_mod = app.load_mod("audio")
        if audio_mod:
            audio_mod.on_start()
    if not app.mod_online("audio"):
        print("Talk Offline install isaa")
        return
    app.load_mod("isaa").init_isaa(build=False)
    talk_generate = app.run_any(TBEF.AUDIO.STT_GENERATE,
                                model="openai/whisper-small",
                                row=True, device=1)

    talk_tts = partial(app.get_function(TBEF.AUDIO.SPEECH)[0], voice_index=0,
                       use_cache=False,
                       provider='piper',
                       config={'play_local': False},
                       save=False)

    if talk_generate is not None:
        app.print('talk_generate Online')
    else:
        app.print("ERROR talk_generate")
    if talk_tts is not None:
        app.print('talk_tts Online')
    else:
        app.print("ERROR talk_tts")


# WebSocket-Endpunkt zum Senden der Audio-Chunks
@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True,
        name="talk_websocket_echo", row=True)
async def upload_audio(websocket: WebSocket):
    if websocket is None:
        return
    await websocket.accept()

    try:
        while True:
            # Empfangen des Audio-Blobs vom Client
            audio_data = await websocket.receive_bytes()
            await asyncio.sleep(0.6)
            await websocket.send_bytes(audio_data)

    except Exception as e:
        print(f"Fehler beim Empfangen der Audiodaten: {e}")


async def stream_response(app, input_text, websocket: WebSocket,
                          provider='piper', voice_index=0, fetch_memory=False, all_meme=False, model_name='ryan',
                          f=False):
    llm_text = [""]

    async def stream_text(text):
        llm_text[0] += text
        if text.endswith('\n\n') or text.endswith('\n') or text.endswith('.') or text.endswith('?'):
            if llm_text[0] == "":
                return
            await websocket.send_json({"type": "response", "text": llm_text[0]})
            await asyncio.sleep(0.25)
            audio_data: bytes = app.run_any(TBEF.AUDIO.SPEECH, text=llm_text[0], voice_index=voice_index,
                                            use_cache=False,
                                            provider=provider,
                                            config={'play_local': False, 'model_name': model_name},
                                            local=False,
                                            save=False)
            llm_text[0] = ""
            if not audio_data:
                return
            # await websocket.send_json(audio_data)
            await websocket.send_bytes(audio_data)
            await asyncio.sleep(0.25)

    agent = app.run_any(TBEF.ISAA.GET_AGENT_CLASS, agent_name='self')
    agent.stream = True

    from toolboxv2.mods.isaa.AgentFramwork import ConversationMode
    agent.mode = app.get_mod('isaa').controller.rget(ConversationMode)
    agent.stream_function = stream_text

    with Spinner(message="Fetching llm_message...", symbols='+'):
        llm_message = agent.get_llm_message(input_text, persist=True, fetch_memory=fetch_memory,
                                            isaa=app.get_mod("isaa"),
                                            task_from="user", all_meme=all_meme)
    print(llm_message)
    out = await agent.a_run_model(llm_message=llm_message, persist_local=True, persist_mem=fetch_memory)
    f_out = ''
    if f and agent.if_for_fuction_use(out):
        f_out = agent.execute_fuction(persist=True, persist_mem=fetch_memory)
        await stream_text(f_out)
    print("Success", out, f_out)
    return out, f_out


# WebSocket-Endpunkt zum Senden der Audio-Chunks
@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True,
        name="talk_websocket_context", row=True)
async def upload_audio_isaa_context(websocket: WebSocket):
    if websocket is None:
        return
    await websocket.accept()
    app = get_app("Talk Transcribe Audio")
    try:
        while True:
            # Empfangen des Audio-Blobs vom Client
            audio_data: bytes = await websocket.receive_bytes()
            text = talk_generate(audio_data)['text']
            print(text)
            text = get_app().run_any(TBEF.ISAA.MINI_TASK, mini_task=f"{text}",
                                     mode=None, fetch_memory=True, all_mem=True)
            print(text)
            audio_data: bytes = app.run_any(TBEF.AUDIO.SPEECH, voice_index=0,
                                            use_cache=False,
                                            provider='piper',
                                            config={'play_local': False},
                                            save=False, text=text)
            print(f"AUDIO Data : {len(audio_data)}")
            await websocket.send_bytes(audio_data)
    except Exception as e:
        print(f"Fehler beim Empfangen der Audiodaten: {e}")
    await websocket.close()



@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True,
        name="talk_websocket", row=True)
async def upload_audio_isaa(websocket: WebSocket, context="F", all_c="F", v_name="ryan", v_index="0", provider='piper',
                            f="F"):
    if websocket is None:
        return
    await websocket.accept()
    app = get_app("Talk Transcribe Audio")
    stt = app.run_any(TBEF.AUDIO.STT_GENERATE,
                      model="openai/whisper-small",
                      row=True, device=1)

    if v_index not in [str(x) for x in range(len(v_index) - 1, 10 * len(v_index))]:
        await websocket.close()
        return

    accumulated_text = [""]
    workers = [0]
    format_bytes = [b""]
    lock = threading.Lock()

    async def worker_transcribe(audio_data):
        WEBM_START_BYTES_FORMAT = 162
        try:
            if workers[0] > 1:
                text = stt(format_bytes[0][:WEBM_START_BYTES_FORMAT]+audio_data)['text']
            else:
                text = stt(audio_data)['text']
            if text:
                lock.acquire(True)
                accumulated_text[0] += text + " "
                # Send transcription update to client
                await websocket.send_json({"type": "transcription", "text": text})
                workers[0] -= 1
                lock.release()
            print("Done transcribe", workers[0])
        except Exception as e:
            workers[0] -= 1
            app.debug_rains(e)

    try:
        while True:
            data = await websocket.receive()
            if 'bytes' in data.keys():
                print("received audio data ", workers[0])
                # Perform real-time transcription
                if workers[0] == 0:
                    if lock.locked():
                        lock.release()
                    format_bytes[0] = data['bytes']
                workers[0] += 1
                threading.Thread(target=async_test(worker_transcribe), args=(data['bytes'],), daemon=True).start()

            elif 'text' in data.keys():
                print("s", workers[0])
                message = json.loads(data.get('text'))
                if message.get("action") == "process":
                    # Process accumulated text
                    max_itter = 0
                    while workers[0] > 1 and max_itter < 500000:
                        await asyncio.sleep(0.2)
                        max_itter += 1
                    print(accumulated_text[0])
                    response, f_r = await stream_response(app, accumulated_text[0], websocket, provider=provider,
                                                          voice_index=[str(x) for x in
                                                                       range(len(v_index) - 1,
                                                                             10 * len(v_index))].index(
                                                              v_index),
                                                          fetch_memory=context == "T",
                                                          all_meme=all_c == "T", model_name=v_name, f=f == "T")
                    # Send processed response to client
                    accumulated_text = [""]  # Reset accumulated text
                    workers[0] = 0

    except Exception as e:
        app.logger.error(f"Error in Talk receiving audio data: {str(e)}")
        await websocket.send_json({"type": "error", "message": str(e)})

    finally:
        await websocket.close()

    # try:
    #     while True:
    #         # Empfangen des Audio-Blobs vom Client
    #         try:
    #             audio_data: bytes = await websocket.receive_bytes()
    #             text = stt(audio_data)['text']
    #             await stream_response(app, text, websocket, provider=provider,
    #                               voice_index=[str(x) for x in range(len(v_index) - 1, 10 * len(v_index))].index(
    #                                   v_index),
    #                               fetch_memory=context == "T",
    #                               all_meme=all_c == "T", model_name=v_name, f=f == "T")
    #         except Exception as e:
    #             app.logger.error(f"Error in Talk reciving audio data: {str(e)}")
    #             # text = "Something went wrong. I couldn understand your message!"


#
# except Exception as e:
#     print(f"Fehler beim Empfangen der Audiodaten: {e}")
# await websocket.close()


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True,
        name="main_web_talk_entry", row=True)
async def main_web_talk_entry(app: App = None, request: Request or None = None, modi=None):
    if request is None:
        return
    spec_ = get_spec(request).get()
    user = await get_user_from_request(app, request)
    if user.name == "":
        return HTMLResponse(content="<p>Invalid User Pleas Log In <a href='/'>Home</a></p>")

    return HTMLResponse(content='''<script src="/index.js" type="module" defer></script><div>
    <style>
        #container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
        }

        #audioVisualizer {
            width: 300px;
            height: 300px;
            background-color: rgba(var(--background-color), 0.8);
            border-radius: 50%;
            position: relative;
            overflow: hidden;
            border: 2.5px dashed rgba(255, 255, 255, 0.4);
            box-shadow: inset -9px -11px 10px 0px var(--background-color);
        }

        .particle {
            position: absolute;
            width: 10px;
            height: 10px;
            background-color: rgba(var(--text-color), 0.8);
            border-radius: 50%;
            pointer-events: none;
        }

        #microphoneButton {
            margin-top: 20px;
            font-size: 30px;
            padding: 10px 20px;
        }
    </style>
    <div id="container">
        <div id="audioVisualizer"></div>
        <p id="infos"> Infos </p>
        <button id="microphoneButton">
            <span class="material-symbols-outlined">mic</span>
        </button>
    </div>

    <script unSave="true">
        const audioVisualizer = document.getElementById('audioVisualizer');
        const microphoneButton = document.getElementById('microphoneButton');
        const infoPtag = document.getElementById('infos');

        let audioContext, analyser, audioSource, mediaRecorder, webSocket, currentAudio;
        let particles = [];
        let isRecording = false;
        let isPlaying = false;
        let isError = false;
        let audioChunks = [];
        let audioQue = [];

        audioVisualizer.style.borderColor = 'black';

        if (window.history.state && !window.history.state.url.includes('?')) {
        const modi = window.history.state.url + '?modi=echo | context| translate?lang=en | ..';
        const rest_def = window.history.state.url + '?v_name=ryan&context=F&all_c=F';
        infoPtag.innerText = modi+" OR "+rest_def+" Options in url : context={F,T},all_c={F,T}, v_name={karlsson[DE],pavoque[DE],hfc_female[EN],kathleen[EN],lessac[EN],ryan[EN]}, v_index=0, provider={piper,eleven_labs}";
        }

        // Initialisiere den AudioContext und AnalyserNode
        async function initAudioAnalysis() {
            try {
                audioContext = new AudioContext();
                const stream = await navigator.mediaDevices.getUserMedia({ audio: {
                    channelCount: 1,
                    sampleRate: 16000
                } });
                audioSource = audioContext.createMediaStreamSource(stream);
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 64;
                audioSource.connect(analyser);

                // MediaRecorder initialisieren (new Blob([event.data], { type: 'audio/webm;codecs=opus' }));
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus', });

                 let isFirstChunk = true;


                mediaRecorder.ondataavailable = async (event) => {
                    if (event.data.size > 0 && webSocket && webSocket.readyState === WebSocket.OPEN) {
                        // Erstelle eine neue Blob für jeden Chunk
                        const chunk = new Blob([event.data], { type: 'audio/webm;codecs=opus' });
                        webSocket.send(chunk);
                    }
                };

                //mediaRecorder.ondataavailable = event => {
                //    if (event.data.size > 0) {
                //        audioChunks.push(event.data);
//
                //        // Sende Audiodaten erst nach dem ersten Chunk
                //        if (!isFirstChunk && webSocket && webSocket.readyState === WebSocket.OPEN) {
                //            const blob = new Blob(audioChunks, { type: 'audio/webm;codecs=opus' });
                //            webSocket.send(blob);
                //            audioChunks = []; // Leere den Puffer nach dem Senden
                //        }
                //        isFirstChunk = false;
                //    }
                //};
                mediaRecorder.onstart = () => {
            console.log("MediaRecorder gestartet");
            audioChunks = [];
            isFirstChunk = true;
            infoPtag.innerText = "Recording...";
        };

        mediaRecorder.onstop = async () => {
            console.log("MediaRecorder gestoppt");
            infoPtag.innerText = "Processing...";

            // Sende verbleibende Audiodaten
            if (audioChunks.length > 0 && webSocket && webSocket.readyState === WebSocket.OPEN) {
                const blob = new Blob(audioChunks, { type: 'audio/webm;codecs=opus' });
                webSocket.send(blob);
            }

            // Sende ein Signal an den Server, dass die Aufnahme beendet wurde
            if (webSocket && webSocket.readyState === WebSocket.OPEN) {
                webSocket.send(JSON.stringify({ action: 'process' }));
            }

            audioChunks = [];
        };

                createParticles(100);
                visualizeAudio();
            } catch (error) {
                console.error('Error initializing audio analysis:', error);
                errorRedParticles();
            }
        }

        // Erstelle die Partikel für den Visualizer
        function createParticles(num) {
            for (let i = 0; i < num; i++) {
                const particle = document.createElement('div');
                particle.classList.add('particle');

                // Positioniere die Partikel entlang eines Kreises
                const angle = Math.random() * Math.PI * 2;  // Zufälliger Winkel im Kreis
                const radius = Math.random() * 50 + 100;    // Zufälliger Radius im Kreis
                particle.x = 150 + Math.cos(angle) * radius;
                particle.y = 150 + Math.sin(angle) * radius;

                particle.style.transform = `translate(${particle.x}px, ${particle.y}px)`;
                particle.speedX = (Math.random() - 0.5) * 2;  // zufällige Bewegungsrichtung X
                particle.speedY = (Math.random() - 0.5) * 2;  // zufällige Bewegungsrichtung Y
                audioVisualizer.appendChild(particle);
                particles.push(particle);
            }
        }

        // Visualisiere die Audiodaten mit Partikeln
        function visualizeAudio() {
            const frequencyData = new Uint8Array(analyser.frequencyBinCount);

            function renderFrame() {
                requestAnimationFrame(renderFrame);
                analyser.getByteFrequencyData(frequencyData);
                if (isRecording || isPlaying) {
                    particles.forEach((particle, index) => {
                        let intensity = frequencyData[index % frequencyData.length] / 255;

                        const size = Math.max(5, intensity * 30); // Partikelgröße basierend auf der Intensität
                        particle.style.width = `${size}px`;
                        particle.style.height = `${size}px`;

                        // Partikelbewegung basierend auf Lautstärke (wenn aufgenommen wird)

                        const speedMultiplier = intensity * 10;
                        particle.x += particle.speedX * speedMultiplier;
                        particle.y += particle.speedY * speedMultiplier;

                        // Begrenzung der Partikel innerhalb des Containers (Kreis)
                        const distanceFromCenter = Math.sqrt(
                            Math.pow(particle.x - 150, 2) + Math.pow(particle.y - 150, 2)
                        );
                        if (distanceFromCenter > 150) {
                            particle.speedX *= -1;
                            particle.speedY *= -1;
                        }

                        particle.style.transform = `translate(${particle.x}px, ${particle.y}px)`;
                        particle.style.backgroundColor = `rgba(0, 0, 255, ${intensity})`;

                    });
                }else if (audioChunks.length > 0) {
                    particles.forEach((particle, index) => {
                        particle.x += particle.speedX;
                        particle.y += particle.speedY;

                        // Begrenzung der Partikel innerhalb des Containers (Kreis)
                        const distanceFromCenter = Math.sqrt(
                            Math.pow(particle.x - 150, 2) + Math.pow(particle.y - 150, 2)
                        );
                        if (distanceFromCenter > 150) {
                            particle.speedX *= -1;
                            particle.speedY *= -1;
                        }

                        particle.style.transform = `translate(${particle.x}px, ${particle.y}px)`;

                    });
                }else if (!isRecording && !isError){
                    particles.forEach((particle, index) => {
                        particle.x += particle.speedX / 2;
                        particle.y += particle.speedY / 2;

                        // Begrenzung der Partikel innerhalb des Containers (Kreis)
                        const distanceFromCenter = Math.sqrt(
                            Math.pow(particle.x - 150, 2) + Math.pow(particle.y - 150, 2)
                        );
                        if (distanceFromCenter > 150) {
                            particle.speedX *= -1;
                            particle.speedY *= -1;
                        }

                        particle.style.transform = `translate(${particle.x}px, ${particle.y}px)`;

                    });
                }
            }

            renderFrame();
        }

        // Initialisiere die WebSocket-Verbindung
        function playNextAudio() {
            if (audioQue.length) {
                isPlaying = false;
                playAudio(audioQue.shift());
            }
        }
        function playAudio(audioUrl) {
            if (isPlaying) {
                    audioQue.push(audioUrl);
                    return
                }
            if (currentAudio) {
                    currentAudio.pause();
                    currentAudio.currentTime = 0; // Setze den Audiowiederholungszeitpunkt auf den Anfang
                }

                currentAudio = new Audio(audioUrl);
                currentAudio.volume = 0.6;

                const audioContext2 = new AudioContext();
                const audioSource2 = audioContext2.createMediaElementSource(currentAudio);
                analyser = audioContext2.createAnalyser();
                audioSource2.connect(analyser);
                analyser.connect(audioContext2.destination);

                currentAudio.play();
                isPlaying = true;

                currentAudio.onended = () => {
                    isPlaying = false;
                    if (audioQue.length) {
                        playNextAudio();
                        infoPtag.innerText = "Playing next audio "+audioQue.length+" in list";
                    }else {
                        microphoneButton.innerHTML = `<span class="material-symbols-outlined">mic</span>`;
                        resetParticles();
                        infoPtag.innerText = "Press enter to continue...";
                    }
                };

        }
        function initWebSocket() {
            if (!window.history.state.TB){
                audioVisualizer.style.borderColor = 'red';
                audioVisualizer.innerHtml = `<h2>Refresh the page (F5)</h2>`;
                // microphoneButton.innerHTML = `<span class="material-symbols-outlined">stop_circle</span>`;
                return
            }
            const url = window.history.state.TB.base.replace(/http/s, 'ws')+'/api/talk/talk_websocket''' + (
        '_' + modi if modi else '') + ''''
            webSocket = new WebSocket(url);
            audioVisualizer.style.borderColor = 'violet';
            webSocket.onopen = () => {
                console.log('WebSocket connection opened');
                resetParticles();
                audioVisualizer.style.borderColor = 'white';
            };

            webSocket.onmessage = (event) => {
                if (event.data instanceof Blob) {
            // Handle audio data
            microphoneButton.innerHTML = `<span class="material-symbols-outlined">stop_circle</span>`;
            lilaParticles();

                    const audioBlob = new Blob([event.data], { type: 'audio/mpeg' });
                    const audioUrl = URL.createObjectURL(audioBlob);

                    playAudio(audioUrl);
                } else {
                console.log(event.data)
                    const message = JSON.parse(event.data);
                    if (message.type === 'transcription') {
                        updateSubtitles(message.text);
                    } else if (message.type === 'response') {
                        displayResponse(message.text);
                    } else if (message.type === 'error') {
                        handleError(message.message);
                    }
                }
            };

            webSocket.onclose = () => {
                console.log('WebSocket connection closed');
                audioVisualizer.style.borderColor = 'red';
                infoPtag.innerText = "Connection closed";
                errorRedParticles();
            };
        }

        // Ereignishandler für Mikrofonaufnahme
        microphoneButton.addEventListener('click', async () => {
            infoPtag.innerText = "";
            if (isPlaying) {
                stopPlayback();
                return;
            }
            if (!isRecording) {
                await startRecording();
            } else {
                stopRecording();
            }
        });

        async function startRecording() {
            if (!audioContext) {
                await initAudioAnalysis();
            }else {
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 64;
                audioSource.connect(analyser);
            }
            // microphoneButton.innerHTML = `<span class="material-symbols-outlined">send</span>`;
            audioChunks = [];  // Reset audioChunks before starting recording

            mediaRecorder.start(1200);  // Sicherstellen, dass mediaRecorder initialisiert ist
            isRecording = true;
            updateUI();
            // infoPtag.innerText = "Listening";
        }

        function stopRecording() {
            stopPlayback();
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                updateUI();
                // Partikel in Neon-Grün und konstante Bewegung
                particles.forEach(particle => {
                    particle.style.backgroundColor = 'rgba(0, 255, 0, 0.5)';
                });
            }
            // microphoneButton.innerHTML = `<span class="material-symbols-outlined">mic</span>`;

        }

        function updateSubtitles(text) {
            infoPtag.innerHTML = text;
        }

        function displayResponse(text) {
            infoPtag.innerHTML = text;
        }

        function handleError(message) {
            console.error('Error:', message);
            errorRedParticles();
        }

        function updateUI() {
            const micButton = document.getElementById('microphoneButton');
            micButton.innerHTML = isRecording ?
                '<span class="material-symbols-outlined">stop_circle</span>' :
                '<span class="material-symbols-outlined">mic</span>';
        }

        // Audio-Daten über WebSocket senden
        function sendAudioData() {
            microphoneButton.innerHTML = `<span class="material-symbols-outlined">cancel</span>`;
            const audioBlob = new Blob(audioChunks, { type: 'audio/mpeg' });
            if (webSocket && webSocket.readyState === WebSocket.OPEN) {
                webSocket.send(audioBlob);  // Sicherstellen, dass die WebSocket-Verbindung geöffnet ist
                console.log("Audio data sent to the server");
            } else {
                console.error("WebSocket is not open. Cannot send data.");
                errorRedParticles();
            }
            audioChunks = [];
        }

        // Stoppe das Abspielen und setze den Zustand zurück
        function stopPlayback() {
             if (isPlaying) {
                isPlaying = false;
                if (currentAudio) {
                    currentAudio.pause();
                    currentAudio.currentTime = 0; // Setze den Audiowiederholungszeitpunkt auf den Anfang
                }
                resetParticles();
                microphoneButton.innerHTML = `<span class="material-symbols-outlined">mic</span>`;
                audioQue = [];
            }
        }

        function resetParticles() {
            particles.forEach(particle => {
                particle.style.backgroundColor = 'rgba(255,255,255, 0.3)';
            });
        }
        function blueParticles() {
            particles.forEach(particle => {
                particle.style.backgroundColor = 'rgba(0, 0, 255, 0.6)';
            });
        }
        function lilaParticles() {
            particles.forEach(particle => {
                particle.style.backgroundColor = 'rgba(0, 255, 0, 0.6)';
            });
        }
        function errorRedParticles() {
            isError = true;
            particles.forEach(particle => {
                particle.style.backgroundColor = 'rgba(255, 0, 0, 0.8)';
            });
        }

        document.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                microphoneButton.click(); // Simuliere einen Klick auf den Button
            }
        });

        // WebSocket beim Start initialisieren
        initWebSocket();
    </script>
</div>''')
