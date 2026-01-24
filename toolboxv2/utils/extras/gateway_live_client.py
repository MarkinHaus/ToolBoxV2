"""
LLM Gateway Live Voice Client (Python)

Minimalist async client for the Live Voice API.

Usage:
    from live_client import LiveClient

    async with LiveClient(api_key="sk-...") as client:
        session = await client.create_session(model="qwen3-4b")

        async for event in session.stream():
            if event["type"] == "audio":
                play_audio(event["audio"])
            elif event["type"] == "text":
                print(event["text"], end="")

        await session.send_audio(audio_chunk, transcript="Hello")
        await session.end_turn("Hello, how are you?")
"""

import asyncio
import base64
import json
import os
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, AsyncGenerator, Callable
from contextlib import asynccontextmanager

try:
    import httpx
    import websockets
except ImportError as e:
    raise ImportError(
        "Required packages not installed. Run: pip install httpx websockets"
    ) from e

try:
    import pyaudio
    import webrtcvad
except ImportError as e:
    raise ImportError(
        "Für Audio-Streaming werden zusätzliche Pakete benötigt. "
        "Run: pip install pyaudio webrtcvad"
    ) from e

# === Configuration ===

@dataclass
class AudioConfig:
    input_format: str = "webm"
    output_format: str = "opus"
    sample_rate: int = 24000
    allow_interrupt: bool = True

    def to_dict(self) -> Dict:
        return {
            "input_format": self.input_format,
            "output_format": self.output_format,
            "sample_rate": self.sample_rate,
            "allow_interrupt": self.allow_interrupt
        }


@dataclass
class WakeWordConfig:
    enabled: bool = False
    words: List[str] = field(default_factory=list)
    mode: str = "pre"  # pre, post, mid

    def to_dict(self) -> Dict:
        return {
            "enabled": self.enabled,
            "words": self.words,
            "mode": self.mode
        }


@dataclass
class VoiceConfig:
    voice_id: str = "default"
    speed: float = 1.0
    language: str = "auto"

    def to_dict(self) -> Dict:
        return {
            "voice_id": self.voice_id,
            "speed": self.speed,
            "language": self.language
        }


@dataclass
class LLMConfig:
    model: str
    system_prompt: str = "Du bist ein hilfreicher Assistent."
    tools: List[Dict] = field(default_factory=list)
    history_length: int = 20
    temperature: float = 0.7
    max_tokens: int = 1024

    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "history_length": self.history_length,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


# === Events ===

@dataclass
class LiveEvent:
    """Base event from live session"""
    type: str
    data: Dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict) -> "LiveEvent":
        return cls(type=d.get("type", "unknown"), data=d)

    @property
    def is_audio(self) -> bool:
        return self.type == "audio"

    @property
    def is_text(self) -> bool:
        return self.type == "text"

    @property
    def is_complete(self) -> bool:
        return self.type == "turn_complete"

    @property
    def is_interrupted(self) -> bool:
        return self.type == "interrupted"

    @property
    def is_error(self) -> bool:
        return self.type == "error"

    def get_audio_bytes(self) -> Optional[bytes]:
        """Get decoded audio bytes if this is an audio event"""
        if self.type == "audio" and "audio" in self.data:
            return base64.b64decode(self.data["audio"])
        return None

    def get_text(self) -> Optional[str]:
        """Get text content"""
        return self.data.get("text")

    def get_error(self) -> Optional[str]:
        """Get error message"""
        return self.data.get("message")


# --- NEUE KLASSEN FÜR AUDIO I/O & VAD ---

class AudioPlayer:
    """Verwaltet asynchrone Audio-Wiedergabe mit Interrupt-Fähigkeit"""

    def __init__(self, sample_rate: int = 24000):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
            start=False
        )
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self._thread = None

    def _play_loop(self):
        while self.is_playing:
            try:
                # Timeout erlaubt sanftes Beenden
                chunk = self.audio_queue.get(timeout=0.1)
                if chunk is None:  # Stop-Signal
                    break
                self.stream.write(chunk)
            except queue.Empty:
                continue

    def start(self):
        self.is_playing = True
        self.stream.start_stream()
        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()

    def add_chunk(self, audio_bytes: bytes):
        if self.is_playing:
            self.audio_queue.put(audio_bytes)

    def stop(self):
        """Sofortiger Interrupt der Wiedergabe"""
        self.is_playing = False
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        self.audio_queue.put(None)  # Unblock
        if self._thread:
            self._thread.join(timeout=0.5)
        self.stream.stop_stream()

    def close(self):
        self.stop()
        self.stream.close()
        self.p.terminate()


class VADRecorder:
    """Nimmt Audio auf und erkennt Sprache (VAD) für automatisches Turn-Handling"""

    def __init__(self, sample_rate: int = 16000, frame_duration_ms: int = 30):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.chunk_size = int(sample_rate * frame_duration_ms / 1000)

        self.vad = webrtcvad.Vad(3)  # Aggressivität 3 (höchste)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

    def read_frame(self) -> tuple[bytes, bool]:
        """Liest einen Frame und prüft auf Sprache"""
        frame = self.stream.read(self.chunk_size, exception_on_overflow=False)
        is_speech = self.vad.is_speech(frame, self.sample_rate)
        return frame, is_speech

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


# === Session ===

class LiveSession:
    """
    Active live voice session.

    Handles WebSocket communication with the server.
    """

    def __init__(
        self,
        token: str,
        websocket_url: str,
        expires_at: str,
        api_key: str
    ):
        self.token = token
        self.websocket_url = websocket_url
        self.expires_at = expires_at
        self._api_key = api_key
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._receive_task: Optional[asyncio.Task] = None
        self._event_queue: asyncio.Queue[LiveEvent] = asyncio.Queue()

        # --- METHODE ZU LiveSession HINZUFÜGEN ---

    async def run_voice_loop(
        self,
        on_text: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[Dict], None]] = None,
        silence_timeout_ms: int = 1500
    ):
        """
        Führt den vollständigen Voice-Loop aus:
        Mikrofon -> VAD -> Server -> Playback.
        Unterstützt automatische Unterbrechung, wenn der User zu sprechen beginnt.
        """
        if not self._connected:
            raise RuntimeError("Nicht verbunden")

        # Für einfache Wiedergabe fordern wir PCM vom Server an
        self.audio_config.output_format = "pcm"

        recorder = VADRecorder(sample_rate=16000)  # WebRTC VAD benötigt 16kHz
        player = AudioPlayer(sample_rate=self.audio_config.sample_rate)
        player.start()

        # VAD State
        is_speaking = False
        silence_frames = 0
        max_silence_frames = int((silence_timeout_ms / 1000) * (16000 / recorder.chunk_size))

        print("🎤 Höre zu... (Sprich, um die Interaktion zu starten)")

        try:
            while self._connected:
                # 1. Lokales Audio aufnehmen und VAD prüfen
                frame, is_speech = recorder.read_frame()

                # INTERRUPT LOGIK: User spricht, während Assistent noch spielt
                if is_speech and player.is_playing:
                    if self.audio_config.allow_interrupt:
                        print("\n[INTERRUPT ERKANNT] Stoppe Wiedergabe...")
                        player.stop()
                        await self.interrupt()
                        # Buffer leeren und neu starten für neue Aufnahme
                        player.start()

                if is_speech:
                    if not is_speaking:
                        print("\n🗣️ Sprache erkannt...")
                        is_speaking = True
                    silence_frames = 0
                    # Sende Audio-Chunk an Server (für potenzielles Server-STT)
                    await self.send_audio(frame, is_final=False)

                elif is_speaking:
                    silence_frames += 1
                    await self.send_audio(frame, is_final=False)  # Ausklingphase mitsenden

                    # TURN-ENDE ERKANNT
                    if silence_frames > max_silence_frames:
                        is_speaking = False
                        print("✅ Spracheingabe beendet, verarbeite...")
                        # Signalisiere Server: Turn beendet, starte LLM+TTS
                        await self.send_audio(b"", is_final=True)
                        await self.end_turn("")  # Leerer Text signalisiert: Nutze gesendetes Audio

                # 2. Eingehende Server-Events verarbeiten (Nicht-blockierend)
                try:
                    event = self._event_queue.get_nowait()

                    if event.is_audio:
                        audio_bytes = event.get_audio_bytes()
                        if audio_bytes:
                            player.add_chunk(audio_bytes)

                    elif event.is_text and on_text:
                        on_text(event.get_text())

                    elif event.type == "turn_complete":
                        if "tools_called" in event.data and event.data["tools_called"]:
                            if on_tool_call:
                                on_tool_call(event.data)
                        print(f"\n[Turn komplett | Kosten: €{event.data.get('cost', 0):.4f}]")

                    elif event.is_error:
                        print(f"❌ Server Fehler: {event.get_error()}")

                except asyncio.QueueEmpty:
                    pass

                await asyncio.sleep(0.001)  # Event-Loop atmen lassen

        finally:
            recorder.close()
            player.close()

    async def connect(self):
        """Connect to WebSocket"""
        if self._connected:
            return

        # WebSocket URL should be absolute
        ws_url = self.websocket_url
        if not ws_url.startswith("ws"):
            # Convert http to ws
            ws_url = ws_url.replace("http://", "ws://").replace("https://", "wss://")

        self._ws = await websockets.connect(
            ws_url,
            extra_headers={"Authorization": f"Bearer {self._api_key}"}
        )
        self._connected = True

        # Start receive task
        self._receive_task = asyncio.create_task(self._receive_loop())

        # Wait for ready event
        event = await self._event_queue.get()
        if event.type != "ready":
            raise ConnectionError(f"Expected 'ready', got '{event.type}'")

    async def disconnect(self):
        """Disconnect from WebSocket"""
        if not self._connected:
            return

        self._connected = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            await self._ws.close()
            self._ws = None

    async def _receive_loop(self):
        """Background task to receive messages"""
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    event = LiveEvent.from_dict(data)
                    await self._event_queue.put(event)
                except json.JSONDecodeError:
                    pass
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            await self._event_queue.put(LiveEvent(
                type="error",
                data={"message": str(e)}
            ))

    async def send_audio(
        self,
        audio: bytes,
        transcript: str = "",
        is_final: bool = False
    ):
        """
        Send audio chunk with optional transcript.

        Args:
            audio: Raw audio bytes
            transcript: Transcribed text (from client-side STT)
            is_final: Whether this is the final chunk of the utterance
        """
        if not self._connected:
            raise RuntimeError("Not connected")

        await self._ws.send(json.dumps({
            "type": "audio",
            "audio": base64.b64encode(audio).decode() if audio else "",
            "transcript": transcript,
            "is_final": is_final
        }))

    async def end_turn(self, transcript: str):
        """
        Signal end of user turn with full transcript.

        This triggers LLM processing and TTS response.

        Args:
            transcript: Full transcribed text of user's speech
        """
        if not self._connected:
            raise RuntimeError("Not connected")

        await self._ws.send(json.dumps({
            "type": "end_turn",
            "transcript": transcript
        }))

    async def interrupt(self):
        """
        Interrupt current assistant response.

        Call this when user starts speaking during assistant output.
        """
        if not self._connected:
            raise RuntimeError("Not connected")

        await self._ws.send(json.dumps({
            "type": "interrupt"
        }))

    async def ping(self):
        """Send keep-alive ping"""
        if not self._connected:
            return

        await self._ws.send(json.dumps({"type": "ping"}))

    async def receive(self, timeout: Optional[float] = None) -> LiveEvent:
        """
        Receive next event.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            LiveEvent from server
        """
        if timeout:
            return await asyncio.wait_for(
                self._event_queue.get(),
                timeout=timeout
            )
        return await self._event_queue.get()

    async def stream(self) -> AsyncGenerator[LiveEvent, None]:
        """
        Stream events from server.

        Yields events until session ends or error occurs.

        Usage:
            async for event in session.stream():
                if event.is_audio:
                    play_audio(event.get_audio_bytes())
        """
        while self._connected:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                yield event

                if event.is_error:
                    break

            except asyncio.TimeoutError:
                continue

    async def converse(
        self,
        text: str,
        on_audio: Optional[Callable[[bytes], None]] = None,
        on_text: Optional[Callable[[str], None]] = None
    ) -> Dict:
        """
        High-level conversation turn.

        Sends text, waits for response, and returns completion info.

        Args:
            text: User's text input
            on_audio: Callback for audio chunks
            on_text: Callback for text chunks

        Returns:
            turn_complete event data
        """
        # Send user turn
        await self.end_turn(text)

        # Collect response
        full_text = ""

        while True:
            event = await self.receive(timeout=60.0)

            if event.is_audio and on_audio:
                audio = event.get_audio_bytes()
                if audio:
                    on_audio(audio)

            if event.is_text:
                chunk = event.get_text() or ""
                full_text += chunk
                if on_text:
                    on_text(chunk)

            if event.is_complete:
                return event.data

            if event.is_interrupted:
                return {**event.data, "interrupted": True}

            if event.is_error:
                raise RuntimeError(event.get_error())

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


# === Client ===

class LiveClient:
    """
    LLM Gateway Live Voice Client.

    Usage:
        async with LiveClient(api_key="sk-...") as client:
            session = await client.create_session(model="qwen3-4b")
            async with session:
                result = await session.converse("Hello!")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:4000"
    ):
        self.api_key = api_key or os.getenv("TB_LLM_GATEWAY_KEY", "")
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30.0
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def create_session(
        self,
        model: str,
        system_prompt: str = "Du bist ein hilfreicher Assistent.",
        audio_config: Optional[AudioConfig] = None,
        wake_word_config: Optional[WakeWordConfig] = None,
        voice_config: Optional[VoiceConfig] = None,
        tools: Optional[List[Dict]] = None,
        history_length: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        session_ttl_minutes: int = 15
    ) -> LiveSession:
        """
        Create a new live voice session.

        Args:
            model: LLM model name
            system_prompt: System prompt for conversation
            audio_config: Audio input/output configuration
            wake_word_config: Wake word detection settings
            voice_config: TTS voice settings
            tools: Function calling tools
            history_length: Number of turns to keep in context
            temperature: LLM temperature
            max_tokens: Max response tokens
            session_ttl_minutes: Session timeout (5-20 minutes)

        Returns:
            LiveSession ready to connect
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context.")

        payload = {
            "audio_config": (audio_config or AudioConfig()).to_dict(),
            "wake_word_config": (wake_word_config or WakeWordConfig()).to_dict(),
            "voice_config": (voice_config or VoiceConfig()).to_dict(),
            "llm_config": LLMConfig(
                model=model,
                system_prompt=system_prompt,
                tools=tools or [],
                history_length=history_length,
                temperature=temperature,
                max_tokens=max_tokens
            ).to_dict(),
            "session_ttl_minutes": session_ttl_minutes
        }

        resp = await self._client.post("/v1/audio/live", json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Build full WebSocket URL
        ws_url = data["websocket_url"]
        if not ws_url.startswith("ws"):
            ws_url = self.base_url.replace("http", "ws") + ws_url

        return LiveSession(
            token=data["session_token"],
            websocket_url=ws_url,
            expires_at=data["expires_at"],
            api_key=self.api_key
        )

    async def close_session(self, session_token: str):
        """Close a session by token"""
        if not self._client:
            raise RuntimeError("Client not initialized")

        resp = await self._client.delete(f"/v1/audio/live/{session_token}")
        resp.raise_for_status()
        return resp.json()

    async def get_session_info(self, session_token: str) -> Dict:
        """Get session info"""
        if not self._client:
            raise RuntimeError("Client not initialized")

        resp = await self._client.get(f"/v1/audio/live/{session_token}")
        resp.raise_for_status()
        return resp.json()

    async def text_to_speech(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        response_format: str = "opus"
    ) -> bytes:
        """
        One-shot text-to-speech.

        Args:
            text: Text to synthesize
            voice: Voice ID
            speed: Speech speed
            response_format: Output format (opus, mp3, wav)

        Returns:
            Audio bytes
        """
        if not self._client:
            raise RuntimeError("Client not initialized")

        resp = await self._client.post(
            "/v1/audio/speech",
            json={
                "model": "tts",
                "input": text,
                "voice": voice,
                "speed": speed,
                "response_format": response_format
            }
        )
        resp.raise_for_status()
        return resp.content


# === Example Usage ===

async def main1():
    """Example usage of LiveClient"""
    import os

    api_key = os.getenv("TB_LLM_GATEWAY_KEY", "sk-admin-xxx")

    print("🎙️ LLM Gateway Live Voice Client Demo\n")

    async with LiveClient(api_key=api_key) as client:
        # Create session
        print("Creating session...")
        session = await client.create_session(
            model="qwen3-4b",
            system_prompt="Du bist ein freundlicher Assistent. Antworte kurz und prägnant.",
            session_ttl_minutes=10
        )
        print(f"Session created: {session.token}")
        print(f"Expires at: {session.expires_at}\n")

        async with session:
            print("Connected! Starting conversation...\n")

            # Simple conversation loop
            while True:
                try:
                    user_input = input("You: ").strip()
                    if not user_input:
                        continue
                    if user_input.lower() in ("quit", "exit", "bye"):
                        break

                    print("Assistant: ", end="", flush=True)

                    result = await session.converse(
                        user_input,
                        on_text=lambda t: print(t, end="", flush=True)
                    )

                    print()  # Newline after response
                    print(f"  [Tokens: {result.get('usage', {}).get('tokens_in', 0)} in, "
                          f"{result.get('usage', {}).get('tokens_out', 0)} out | "
                          f"Cost: €{result.get('cost', 0):.6f}]\n")

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"\nError: {e}\n")

        print("\nSession closed. Goodbye!")

# --- AKTUALISIERTE main() FUNKTION ---

async def main():
    import os

    api_key = os.getenv("TB_LLM_GATEWAY_KEY", "sk-admin-change-me-on-first-run")

    print("🎙️ LLM Gateway Live Voice Client Demo")
    print("--------------------------------------")

    async with LiveClient(api_key=api_key) as client:
        # Konfiguration für direkte PCM Wiedergabe
        audio_cfg = AudioConfig(output_format="pcm", sample_rate=24000, allow_interrupt=True)

        session = await client.create_session(
            model="qwen3-4b", # Oder "omni" für Audio-Input-Modelle
            system_prompt="Du bist ein freundlicher Assistent. Antworte kurz und prägnant auf Deutsch.",
            audio_config=audio_cfg,
            session_ttl_minutes=15
        )
        print(f"Session verbunden: {session.token}")

        async with session:
            try:
                # Callback für LLM Text-Stream
                def on_text(text):
                    print(text, end="", flush=True)

                # Callback für Tool Calls
                def on_tools(data):
                    tools = data.get("tools_called", [])
                    print(f"\n🔧 Tools aufgerufen: {tools}")

                # Startet den echten Live Voice Loop
                await session.run_voice_loop(
                    on_text=on_text,
                    on_tool_call=on_tools,
                    silence_timeout_ms=1000 # 1 Sekunde Stille = Turn-Ende
                )

            except KeyboardInterrupt:
                print("\nSitzung durch Benutzer beendet.")

if __name__ == "__main__":
    asyncio.run(main())

if __name__ == "__main__":
    asyncio.run(main())
