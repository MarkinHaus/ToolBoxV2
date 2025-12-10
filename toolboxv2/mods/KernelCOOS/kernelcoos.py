"""
KernelCOOS - Co-OS Kernel Web Interface
=========================================

Complete implementation based on coos.py specification with:
- Full WebSocket-based communication (Toolbox Websockets)
- Voice-to-Voice support with VAD (Voice Activity Detection)
- Wake Word Activation
- Real-time chat interface
- Session management & configuration
- Memory store (JSONL backend)
- Task scheduler
- Signal bus with priority handling
- Learning engine integration

Version: 1.0.0
Author: Co-OS Team
"""

import asyncio
import json
import os
import time
import uuid
import heapq
import tempfile
import base64
import wave
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict
import traceback

from pydantic import BaseModel, Field

# Framework imports
from toolboxv2 import App, Result, get_app, RequestData, MainTool
from toolboxv2.mods.isaa.kernel.instace import Kernel
from toolboxv2.mods.isaa.kernel.types import (
    Signal as KernelSignal, SignalType, KernelConfig, IOutputRouter,
    UserState, KernelState, KernelMetrics
)

# Try to import audio processing libraries
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️ NumPy not installed. Audio processing may be limited.")

try:
    from groq import Groq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq not installed. Voice transcription disabled.")

try:
    from elevenlabs import ElevenLabs, Voice, VoiceSettings

    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    print("⚠️ ElevenLabs not installed. TTS disabled.")

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = False # No apenai for now but is fully implementet jest set to Ture
except ImportError:
    OPENAI_AVAILABLE = False

# ===== CONSTANTS =====

Name = "KernelCOOS"
VERSION = "1.0.0"

# Wake word configuration
DEFAULT_WAKE_WORDS = ["hey coos", "hey assistant", "ok coos", "coos", "kernel"]

# VAD Configuration
VAD_SILENCE_THRESHOLD = 0.02  # RMS threshold for silence detection
VAD_SPEECH_MIN_DURATION = 0.3  # Minimum speech duration in seconds
VAD_SILENCE_DURATION = 1.5  # Silence duration to end speech in seconds


# ===== SIGNAL TYPES =====

class COOSSignalType(Enum):
    """Extended signal types for COOS"""
    USER_INPUT = "user_input"
    VOICE_INPUT = "voice_input"
    SYSTEM_EVENT = "system_event"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    TOOL_RESULT = "tool_result"
    WAKE_WORD = "wake_word"
    VAD_START = "vad_start"
    VAD_END = "vad_end"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CONFIG_CHANGE = "config_change"


@dataclass
class COOSSignal:
    """Signal structure for COOS kernel"""
    id: str
    type: COOSSignalType
    content: Any
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    priority: int = 5
    metadata: dict = field(default_factory=dict)

    def __lt__(self, other):
        return self.priority > other.priority


# ===== SESSION CONFIGURATION =====

class VoiceConfig(BaseModel):
    """Voice configuration for a session"""
    enabled: bool = True
    wake_word_enabled: bool = True
    wake_words: List[str] = Field(default_factory=lambda: DEFAULT_WAKE_WORDS.copy())
    vad_enabled: bool = True
    vad_sensitivity: float = 0.5  # 0.0 - 1.0
    auto_speak_response: bool = True
    tts_voice: str = "alloy"  # Voice ID for TTS
    tts_provider: str = "openai"  # openai, elevenlabs, browser
    language: str = "de"  # Primary language
    transcription_model: str = "whisper-large-v3-turbo"


class SessionConfig(BaseModel):
    """Complete session configuration"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "anonymous"
    user_name: str = "User"
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    theme: str = "dark"  # dark, light, auto
    response_style: str = "balanced"  # concise, detailed, balanced
    proactivity_level: str = "medium"  # low, medium, high
    notifications_enabled: bool = True
    created_at: float = Field(default_factory=time.time)
    last_active: float = Field(default_factory=time.time)


# ===== VAD (VOICE ACTIVITY DETECTION) =====

class VADProcessor:
    """Voice Activity Detection processor"""

    def __init__(self, config: VoiceConfig = None):
        self.config = config or VoiceConfig()
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.audio_buffer: List[bytes] = []
        self.rms_history: List[float] = []
        self.sample_rate = 16000
        self.channels = 1

        # Dynamic threshold based on sensitivity
        self.silence_threshold = VAD_SILENCE_THRESHOLD * (1 - self.config.vad_sensitivity * 0.5)

    def calculate_rms(self, audio_data: bytes) -> float:
        """Calculate RMS (Root Mean Square) of audio data"""
        if not NUMPY_AVAILABLE:
            return 0.0

        try:
            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_array = audio_array / 32768.0  # Normalize to [-1, 1]

            if len(audio_array) == 0:
                return 0.0

            rms = np.sqrt(np.mean(audio_array ** 2))
            return float(rms)
        except Exception as e:
            print(f"RMS calculation error: {e}")
            return 0.0

    def process_audio_chunk(self, audio_data: bytes) -> Tuple[bool, Optional[str]]:
        """
        Process audio chunk and detect voice activity

        Returns:
            Tuple of (is_speech_detected, event_type)
            event_type can be: "speech_start", "speech_end", or None
        """
        rms = self.calculate_rms(audio_data)
        self.rms_history.append(rms)

        # Keep only last 50 samples for smoothing
        if len(self.rms_history) > 50:
            self.rms_history = self.rms_history[-50:]

        # Smoothed RMS
        avg_rms = sum(self.rms_history[-10:]) / min(len(self.rms_history), 10)

        current_time = time.time()
        event = None

        if avg_rms > self.silence_threshold:
            # Speech detected
            self.audio_buffer.append(audio_data)
            self.silence_start_time = None

            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = current_time
                event = "speech_start"

        else:
            # Silence detected
            if self.is_speaking:
                self.audio_buffer.append(audio_data)

                if self.silence_start_time is None:
                    self.silence_start_time = current_time

                # Check if silence duration exceeded threshold
                silence_duration = current_time - self.silence_start_time
                if silence_duration >= VAD_SILENCE_DURATION:
                    # Speech ended
                    speech_duration = current_time - self.speech_start_time if self.speech_start_time else 0

                    if speech_duration >= VAD_SPEECH_MIN_DURATION:
                        event = "speech_end"

                    self.is_speaking = False
                    self.speech_start_time = None
                    self.silence_start_time = None

        return self.is_speaking, event

    def get_audio_buffer(self) -> bytes:
        """Get the accumulated audio buffer and clear it"""
        if not self.audio_buffer:
            return b""

        audio_data = b"".join(self.audio_buffer)
        self.audio_buffer = []
        return audio_data

    def reset(self):
        """Reset VAD state"""
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.audio_buffer = []
        self.rms_history = []


# ===== WAKE WORD DETECTOR =====

class WakeWordDetector:
    """Simple wake word detection using transcription"""

    def __init__(self, wake_words: List[str] = None):
        self.wake_words = wake_words or DEFAULT_WAKE_WORDS.copy()
        self.is_activated = False
        self.activation_time = None
        self.activation_timeout = 30.0  # Seconds before deactivation

    def check_wake_word(self, transcription: str) -> Tuple[bool, Optional[str]]:
        """
        Check if transcription contains a wake word

        Returns:
            Tuple of (wake_word_detected, matched_wake_word)
        """
        if not transcription:
            return False, None

        transcription_lower = transcription.lower().strip()

        for wake_word in self.wake_words:
            if wake_word.lower() in transcription_lower:
                self.is_activated = True
                self.activation_time = time.time()
                return True, wake_word

        return False, None

    def is_active(self) -> bool:
        """Check if wake word is currently active"""
        if not self.is_activated:
            return False

        # Check timeout
        if self.activation_time and time.time() - self.activation_time > self.activation_timeout:
            self.is_activated = False
            return False

        return True

    def deactivate(self):
        """Manually deactivate wake word"""
        self.is_activated = False
        self.activation_time = None

    def reset_timeout(self):
        """Reset the activation timeout"""
        if self.is_activated:
            self.activation_time = time.time()


# ===== TRANSCRIPTION SERVICE =====

class TranscriptionService:
    """Audio transcription service using Groq or OpenAI"""

    def __init__(self, provider: str = "groq"):
        self.provider = provider
        self.groq_client = None
        self.openai_client = None

        if provider == "groq" and GROQ_AVAILABLE:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self.groq_client = Groq(api_key=api_key)
        elif provider == "openai" and OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)

    async def transcribe(self, audio_data: bytes, language: str = "de") -> Optional[str]:
        """Transcribe audio data to text"""
        if not audio_data:
            return None

        try:
            # Create a WAV file from PCM data
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_data)
            wav_buffer.seek(0)

            if self.provider == "groq" and self.groq_client:
                # Use Groq Whisper
                transcription = await asyncio.to_thread(
                    self._groq_transcribe,
                    wav_buffer,
                    language
                )
                return transcription

            elif self.provider == "openai" and self.openai_client:
                # Use OpenAI Whisper
                transcription = await asyncio.to_thread(
                    self._openai_transcribe,
                    wav_buffer,
                    language
                )
                return transcription

        except Exception as e:
            print(f"Transcription error: {e}")
            traceback.print_exc()
            return None

    def _groq_transcribe(self, wav_buffer: io.BytesIO, language: str) -> Optional[str]:
        """Groq transcription (blocking)"""
        if not self.groq_client:
            return None

        try:
            result = self.groq_client.audio.transcriptions.create(
                file=("audio.wav", wav_buffer, "audio/wav"),
                model="whisper-large-v3-turbo",
                language=language[:2] if len(language) > 2 else language,
                response_format="text"
            )
            return result.strip() if result else None
        except Exception as e:
            print(f"Groq transcription error: {e}")
            return None

    def _openai_transcribe(self, wav_buffer: io.BytesIO, language: str) -> Optional[str]:
        """OpenAI transcription (blocking)"""
        if not self.openai_client:
            return None

        try:
            result = self.openai_client.audio.transcriptions.create(
                file=("audio.wav", wav_buffer, "audio/wav"),
                model="whisper-1",
                language=language[:2] if len(language) > 2 else language,
                response_format="text"
            )
            return result.strip() if result else None
        except Exception as e:
            print(f"OpenAI transcription error: {e}")
            return None


# ===== TTS SERVICE =====

class TTSService:
    """Text-to-Speech service"""

    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.openai_client = None
        self.elevenlabs_client = None

        if provider == "openai" and OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
        elif provider == "elevenlabs" and ELEVENLABS_AVAILABLE:
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if api_key:
                self.elevenlabs_client = ElevenLabs(api_key=api_key)

    async def synthesize(self, text: str, voice: str = "alloy") -> Optional[bytes]:
        """Synthesize text to speech audio"""
        if not text:
            return None

        try:
            if self.provider == "openai" and self.openai_client:
                audio_data = await asyncio.to_thread(
                    self._openai_synthesize,
                    text,
                    voice
                )
                return audio_data

            elif self.provider == "elevenlabs" and self.elevenlabs_client:
                audio_data = await asyncio.to_thread(
                    self._elevenlabs_synthesize,
                    text,
                    voice
                )
                return audio_data

        except Exception as e:
            print(f"TTS error: {e}")
            return None

        return None

    def _openai_synthesize(self, text: str, voice: str) -> Optional[bytes]:
        """OpenAI TTS (blocking)"""
        if not self.openai_client:
            return None

        try:
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                response_format="mp3"
            )
            return response.content
        except Exception as e:
            print(f"OpenAI TTS error: {e}")
            return None

    def _elevenlabs_synthesize(self, text: str, voice: str) -> Optional[bytes]:
        """ElevenLabs TTS (blocking)"""
        if not self.elevenlabs_client:
            return None

        try:
            audio = self.elevenlabs_client.generate(
                text=text,
                voice=voice,
                model="eleven_multilingual_v2"
            )
            return b"".join(audio)
        except Exception as e:
            print(f"ElevenLabs TTS error: {e}")
            return None


# ===== WEBSOCKET OUTPUT ROUTER =====

class COOSWebSocketRouter(IOutputRouter):
    """WebSocket-based output router for COOS kernel"""

    def __init__(self, app: App, channel_id: str):
        self.app = app
        self.channel_id = channel_id
        self.connections: Dict[str, dict] = {}  # conn_id -> session info
        self.user_sessions: Dict[str, str] = {}  # user_id -> conn_id
        self.tts_service: Optional[TTSService] = None

    def set_tts_service(self, tts_service: TTSService):
        """Set TTS service for voice responses"""
        self.tts_service = tts_service

    def register_connection(self, conn_id: str, session: dict):
        """Register a new WebSocket connection"""
        user_id = session.get("user_name", session.get("user_id", "Anonymous"))

        self.connections[conn_id] = {
            "session": session,
            "user_id": user_id,
            "connected_at": datetime.now().isoformat(),
            "config": session.get("config", SessionConfig().model_dump())
        }
        self.user_sessions[user_id] = conn_id
        print(f"✓ Registered connection {conn_id} for user {user_id}")

    def unregister_connection(self, conn_id: str):
        """Unregister a WebSocket connection"""
        if conn_id in self.connections:
            user_id = self.connections[conn_id].get("user_id")
            if user_id and user_id in self.user_sessions:
                del self.user_sessions[user_id]
            del self.connections[conn_id]
            print(f"✓ Unregistered connection {conn_id}")

    async def send_response(self, user_id: str, content: str, role: str = "assistant", metadata: dict = None):
        """Send agent response to user"""
        conn_id = self.user_sessions.get(user_id)
        if not conn_id:
            # Try to find by connection info
            for cid, info in self.connections.items():
                if info.get("user_id") == user_id:
                    conn_id = cid
                    break

        if conn_id:
            message = {
                "event": "agent_response",
                "data": {
                    "content": content,
                    "role": role,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata or {}
                }
            }

            await self.app.ws_send(conn_id, message)

            # Check if we should generate TTS
            config = self.connections.get(conn_id, {}).get("config", {})
            voice_config = config.get("voice", {})

            if voice_config.get("auto_speak_response", True) and self.tts_service:
                # Generate TTS audio
                audio_data = await self.tts_service.synthesize(
                    content,
                    voice_config.get("tts_voice", "alloy")
                )

                if audio_data:
                    # Send audio as base64
                    await self.app.ws_send(conn_id, {
                        "event": "tts_audio",
                        "data": {
                            "audio": base64.b64encode(audio_data).decode('utf-8'),
                            "format": "mp3",
                            "timestamp": datetime.now().isoformat()
                        }
                    })

    async def send_notification(self, user_id: str, content: str, priority: int = 5, metadata: dict = None):
        """Send notification to user"""
        conn_id = self.user_sessions.get(user_id)
        if not conn_id:
            for cid, info in self.connections.items():
                if info.get("user_id") == user_id:
                    conn_id = cid
                    break

        if conn_id:
            await self.app.ws_send(conn_id, {
                "event": "notification",
                "data": {
                    "content": content,
                    "priority": priority,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata or {}
                }
            })

    async def send_error(self, user_id: str, error: str, metadata: dict = None):
        """Send error to user"""
        conn_id = self.user_sessions.get(user_id)
        if conn_id:
            await self.app.ws_send(conn_id, {
                "event": "error",
                "data": {
                    "error": error,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata or {}
                }
            })

    async def send_intermediate(self, user_id: str, content: str, stage: str = "processing"):
        """Send intermediate response during processing"""
        conn_id = self.user_sessions.get(user_id)
        if conn_id:
            await self.app.ws_send(conn_id, {
                "event": "intermediate",
                "data": {
                    "content": content,
                    "stage": stage,
                    "timestamp": datetime.now().isoformat()
                }
            })

    async def send_vad_event(self, user_id: str, event_type: str, metadata: dict = None):
        """Send VAD event to user"""
        conn_id = self.user_sessions.get(user_id)
        if conn_id:
            await self.app.ws_send(conn_id, {
                "event": f"vad_{event_type}",
                "data": {
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata or {}
                }
            })

    async def send_wake_word_event(self, user_id: str, wake_word: str, activated: bool):
        """Send wake word event to user"""
        conn_id = self.user_sessions.get(user_id)
        if conn_id:
            await self.app.ws_send(conn_id, {
                "event": "wake_word",
                "data": {
                    "wake_word": wake_word,
                    "activated": activated,
                    "timestamp": datetime.now().isoformat()
                }
            })

    async def send_transcription(self, user_id: str, text: str, is_final: bool = True):
        """Send transcription result to user"""
        conn_id = self.user_sessions.get(user_id)
        if conn_id:
            await self.app.ws_send(conn_id, {
                "event": "transcription",
                "data": {
                    "text": text,
                    "is_final": is_final,
                    "timestamp": datetime.now().isoformat()
                }
            })

    async def broadcast(self, content: str, event_type: str = "broadcast", exclude_user: str = None):
        """Broadcast to all connections"""
        await self.app.ws_broadcast(
            channel_id=self.channel_id,
            payload={
                "event": event_type,
                "data": {
                    "content": content,
                    "timestamp": datetime.now().isoformat()
                }
            },
            source_conn_id=self.user_sessions.get(exclude_user) if exclude_user else None
        )


# ===== COOS WEB KERNEL =====

class COOSWebKernel:
    """Complete COOS Web Kernel with voice support"""

    def __init__(
        self,
        agent,
        app: App,
        channel_id: str = "coos_kernel",
        auto_save_interval: int = 300
    ):
        self.agent = agent
        self.app = app
        self.channel_id = channel_id
        self.auto_save_interval = auto_save_interval
        self.running = False
        self.save_path = self._get_save_path() if agent else None

        # Initialize kernel config
        config = KernelConfig(
            heartbeat_interval=30.0,
            idle_threshold=300.0,
            proactive_cooldown=60.0,
            max_proactive_per_hour=10
        )

        # Initialize output router
        self.output_router = COOSWebSocketRouter(app, channel_id)

        # Initialize kernel
        self.kernel = Kernel(
            agent=agent,
            config=config,
            output_router=self.output_router
        )

        # Initialize services
        self.transcription_service = TranscriptionService(
            provider="groq" if GROQ_AVAILABLE else "openai"
        )

        tts_provider = "openai" if OPENAI_AVAILABLE else "elevenlabs"
        self.tts_service = TTSService(provider=tts_provider)
        self.output_router.set_tts_service(self.tts_service)

        # Session management
        self.sessions: Dict[str, SessionConfig] = {}
        self.vad_processors: Dict[str, VADProcessor] = {}
        self.wake_word_detectors: Dict[str, WakeWordDetector] = {}

        print(f"✓ COOS Web Kernel initialized")
        print(f"  - Transcription: {'Groq' if GROQ_AVAILABLE else 'OpenAI' if OPENAI_AVAILABLE else 'Disabled'}")
        print(f"  - TTS: {tts_provider if OPENAI_AVAILABLE or ELEVENLABS_AVAILABLE else 'Browser'}")

    async def init(self):
        if self.agent:
            return
        isaa = app.get_mod("isaa")
        builder = isaa.get_agent_builder("COOSKernelAssistant")
        builder.with_system_message(
            """You are COOS, a helpful voice-first AI assistant. You provide clear, engaging responses optimized for both text and voice interaction.

Key behaviors:
- Keep voice responses concise and natural
- Use clear language without complex formatting for voice
- Be proactive and anticipate user needs
- Remember user preferences and context
- Support both German and English fluently"""
        )

        await isaa.register_agent(builder)
        self.agent = await isaa.get_agent("COOSKernelAssistant")
        self.save_path = self._get_save_path()
        self.kernel.agent = self.agent
        self.kernel.learning_engine.agent = self.agent

    def _get_save_path(self) -> Path:
        """Get save file path"""
        save_dir = Path(self.app.data_dir) / 'Agents' / 'kernel' / self.agent.amd.name / 'coos'
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / f"coos_kernel_{self.channel_id}.pkl"

    async def _auto_save_loop(self):
        """Auto-save loop"""
        while self.running:
            await asyncio.sleep(self.auto_save_interval)
            if self.running:
                await self.kernel.save_to_file(str(self.save_path))
                print(f"💾 Auto-saved COOS kernel at {datetime.now().strftime('%H:%M:%S')}")

    async def start(self):
        """Start the kernel"""
        self.running = True
        await self.init()
        # Load previous state
        if self.save_path.exists():
            print("📂 Loading previous COOS session...")
            await self.kernel.load_from_file(str(self.save_path))

        # Start kernel
        await self.kernel.start()

        # Inject kernel prompt
        self.kernel.inject_kernel_prompt_to_agent()

        # Start auto-save
        asyncio.create_task(self._auto_save_loop())

        print(f"✓ COOS Web Kernel started on channel: {self.channel_id}")

    async def stop(self):
        """Stop the kernel"""
        if not self.running:
            return

        self.running = False
        print("💾 Saving COOS session...")

        await self.kernel.save_to_file(str(self.save_path))
        await self.kernel.stop()

        print("✓ COOS Web Kernel stopped")

    def get_or_create_session(self, session_id: str, user_data: dict = None) -> SessionConfig:
        """Get or create session configuration"""
        if session_id not in self.sessions:
            config = SessionConfig(
                session_id=session_id,
                user_id=user_data.get("user_id", "anonymous") if user_data else "anonymous",
                user_name=user_data.get("user_name", "User") if user_data else "User"
            )
            self.sessions[session_id] = config

            # Initialize VAD and wake word for this session
            self.vad_processors[session_id] = VADProcessor(config.voice)
            self.wake_word_detectors[session_id] = WakeWordDetector(config.voice.wake_words)

        return self.sessions[session_id]

    def update_session_config(self, session_id: str, config_updates: dict):
        """Update session configuration"""
        if session_id in self.sessions:
            session = self.sessions[session_id]

            # Update voice config
            if "voice" in config_updates:
                for key, value in config_updates["voice"].items():
                    if hasattr(session.voice, key):
                        setattr(session.voice, key, value)

                # Update VAD processor with new config
                self.vad_processors[session_id] = VADProcessor(session.voice)

                # Update wake word detector
                if "wake_words" in config_updates["voice"]:
                    self.wake_word_detectors[session_id] = WakeWordDetector(session.voice.wake_words)

            # Update other config
            for key, value in config_updates.items():
                if key != "voice" and hasattr(session, key):
                    setattr(session, key, value)

            session.last_active = time.time()

    async def handle_connect(self, conn_id: str, session_data: dict):
        """Handle WebSocket connection"""
        user_id = session_data.get("user_name", session_data.get("user_id", "Anonymous"))
        session_id = session_data.get("session_id", conn_id)

        # Get or create session config
        config = self.get_or_create_session(session_id, session_data)
        session_data["config"] = config.model_dump()

        # Register connection
        self.output_router.register_connection(conn_id, session_data)

        # Send welcome message
        await self.app.ws_send(conn_id, {
            "event": "welcome",
            "data": {
                "message": f"Welcome to COOS Kernel, {user_id}!",
                "session_id": session_id,
                "config": config.model_dump(),
                "kernel_status": self.kernel.to_dict(),
                "capabilities": {
                    "voice_enabled": GROQ_AVAILABLE or OPENAI_AVAILABLE,
                    "tts_enabled": OPENAI_AVAILABLE or ELEVENLABS_AVAILABLE,
                    "vad_enabled": NUMPY_AVAILABLE,
                    "transcription_provider": "groq" if GROQ_AVAILABLE else "openai" if OPENAI_AVAILABLE else "browser",
                    "tts_provider": "openai" if OPENAI_AVAILABLE else "elevenlabs" if ELEVENLABS_AVAILABLE else "browser"
                }
            }
        })

        # Send kernel signal
        signal = KernelSignal(
            type=SignalType.SYSTEM_EVENT,
            id="websocket",
            content=f"User {user_id} connected",
            metadata={"event": "user_connect", "conn_id": conn_id, "session_id": session_id}
        )
        await self.kernel.process_signal(signal)

    async def handle_disconnect(self, conn_id: str, session_data: dict = None):
        """Handle WebSocket disconnection"""
        if session_data is None:
            session_data = {}

        user_id = session_data.get("user_name", "Anonymous")

        # Unregister connection
        self.output_router.unregister_connection(conn_id)

        # Send kernel signal
        signal = KernelSignal(
            type=SignalType.SYSTEM_EVENT,
            id="websocket",
            content=f"User {user_id} disconnected",
            metadata={"event": "user_disconnect", "conn_id": conn_id}
        )
        await self.kernel.process_signal(signal)

    async def handle_message(self, conn_id: str, session_data: dict, payload: dict):
        """Handle incoming WebSocket message"""
        user_id = session_data.get("user_name", "Anonymous")
        session_id = session_data.get("session_id", conn_id)
        event = payload.get("event", "message")
        data = payload.get("data", {})

        try:
            if event == "chat":
                # Text chat message
                await self._handle_chat_message(user_id, session_id, data)

            elif event == "audio_data":
                # Audio data for voice input
                await self._handle_audio_data(user_id, session_id, conn_id, data)

            elif event == "config_update":
                # Update session configuration
                await self._handle_config_update(session_id, conn_id, data)

            elif event == "get_config":
                # Get current session configuration
                await self._handle_get_config(session_id, conn_id)

            elif event == "tts_request":
                # Request TTS synthesis
                await self._handle_tts_request(user_id, conn_id, data)

            elif event == "wake_word_activate":
                # Manually activate wake word
                await self._handle_wake_word_activate(session_id, conn_id)

            elif event == "wake_word_deactivate":
                # Manually deactivate wake word
                await self._handle_wake_word_deactivate(session_id, conn_id)

            elif event == "ping":
                # Heartbeat
                await self.app.ws_send(conn_id, {"event": "pong", "data": {"timestamp": time.time()}})

        except Exception as e:
            print(f"Error handling message: {e}")
            traceback.print_exc()
            await self.output_router.send_error(user_id, str(e))

    async def _handle_chat_message(self, user_id: str, session_id: str, data: dict):
        """Handle text chat message"""
        message = data.get("message", "").strip()
        if not message:
            return

        # Update session activity
        if session_id in self.sessions:
            self.sessions[session_id].last_active = time.time()

        # Send to kernel
        signal = KernelSignal(
            type=SignalType.USER_INPUT,
            id=user_id,
            content=message,
            metadata={
                "interface": "websocket",
                "session_id": session_id,
                "input_type": "text"
            }
        )
        await self.kernel.process_signal(signal)

    async def _handle_audio_data(self, user_id: str, session_id: str, conn_id: str, data: dict):
        """Handle incoming audio data"""
        audio_b64 = data.get("audio", "")
        if not audio_b64:
            return

        # Decode audio
        try:
            audio_data = base64.b64decode(audio_b64)
        except Exception as e:
            print(f"Error decoding audio: {e}")
            return

        # Get session config
        config = self.sessions.get(session_id)
        if not config or not config.voice.enabled:
            return

        # Process with VAD
        vad = self.vad_processors.get(session_id)
        if not vad:
            return

        is_speaking, event = vad.process_audio_chunk(audio_data)

        # Send VAD events
        if event == "speech_start":
            await self.output_router.send_vad_event(user_id, "start")

        elif event == "speech_end":
            await self.output_router.send_vad_event(user_id, "end")

            # Get buffered audio and transcribe
            buffered_audio = vad.get_audio_buffer()
            if buffered_audio:
                await self._process_voice_input(user_id, session_id, conn_id, buffered_audio)

    async def _process_voice_input(self, user_id: str, session_id: str, conn_id: str, audio_data: bytes):
        """Process voice input - transcribe and handle"""
        config = self.sessions.get(session_id)
        if not config:
            return

        # Transcribe
        transcription = await self.transcription_service.transcribe(
            audio_data,
            config.voice.language
        )

        if not transcription:
            return

        # Send transcription to client
        await self.output_router.send_transcription(user_id, transcription)

        # Check wake word if enabled
        if config.voice.wake_word_enabled:
            detector = self.wake_word_detectors.get(session_id)
            if detector:
                is_wake_word, matched_word = detector.check_wake_word(transcription)

                if is_wake_word:
                    await self.output_router.send_wake_word_event(user_id, matched_word, True)
                    # Remove wake word from transcription for processing
                    for ww in detector.wake_words:
                        transcription = transcription.lower().replace(ww.lower(), "").strip()

                    if not transcription:
                        # Only wake word, no actual command
                        return

                elif not detector.is_active():
                    # Wake word not active, ignore input
                    return
                else:
                    # Reset timeout since we're processing
                    detector.reset_timeout()

        # Send to kernel as voice input
        signal = KernelSignal(
            type=SignalType.USER_INPUT,
            id=user_id,
            content=transcription,
            metadata={
                "interface": "websocket",
                "session_id": session_id,
                "input_type": "voice",
                "fast_response": True,  # Enable fast response mode for voice
                "formatting_instructions": "Keep your response concise and natural for voice output. Avoid markdown formatting."
            }
        )
        await self.kernel.process_signal(signal)

    async def _handle_config_update(self, session_id: str, conn_id: str, data: dict):
        """Handle configuration update"""
        self.update_session_config(session_id, data)

        # Send updated config back
        config = self.sessions.get(session_id)
        if config:
            await self.app.ws_send(conn_id, {
                "event": "config_updated",
                "data": config.model_dump()
            })

    async def _handle_get_config(self, session_id: str, conn_id: str):
        """Handle get configuration request"""
        config = self.sessions.get(session_id)
        if config:
            await self.app.ws_send(conn_id, {
                "event": "config",
                "data": config.model_dump()
            })

    async def _handle_tts_request(self, user_id: str, conn_id: str, data: dict):
        """Handle TTS synthesis request"""
        text = data.get("text", "")
        voice = data.get("voice", "alloy")

        if not text:
            return

        audio_data = await self.tts_service.synthesize(text, voice)

        if audio_data:
            await self.app.ws_send(conn_id, {
                "event": "tts_audio",
                "data": {
                    "audio": base64.b64encode(audio_data).decode('utf-8'),
                    "format": "mp3",
                    "timestamp": datetime.now().isoformat()
                }
            })

    async def _handle_wake_word_activate(self, session_id: str, conn_id: str):
        """Handle manual wake word activation"""
        detector = self.wake_word_detectors.get(session_id)
        if detector:
            detector.is_activated = True
            detector.activation_time = time.time()

            await self.app.ws_send(conn_id, {
                "event": "wake_word",
                "data": {
                    "wake_word": "manual",
                    "activated": True,
                    "timestamp": datetime.now().isoformat()
                }
            })

    async def _handle_wake_word_deactivate(self, session_id: str, conn_id: str):
        """Handle manual wake word deactivation"""
        detector = self.wake_word_detectors.get(session_id)
        if detector:
            detector.deactivate()

            await self.app.ws_send(conn_id, {
                "event": "wake_word",
                "data": {
                    "wake_word": None,
                    "activated": False,
                    "timestamp": datetime.now().isoformat()
                }
            })


app = get_app(Name)


export = app.tb
_kernel_instance: Optional[COOSWebKernel] = None

# Global kernel instance
# ===== MODULE REGISTRATION =====


class Tools(MainTool):
    """DirCut Module Tools"""

    def __init__(self, app: App):
        self.name = Name
        self.version = VERSION
        self.tools = {
            "all": [["version", "Zeigt Modul-Version"]],
            "name": self.name,
            "version": self.show_version,
        }

        super().__init__(
            load=init_kernel_coos,
            v=self.version,
            tool=self.tools,
            name=self.name,
            on_exit=self.on_exit
        )


    def on_exit(self):
        """Cleanup beim Beenden"""
        self.app.logger.info(f"{self.name} wird beendet...")

    def show_version(self):
        """Zeigt Version"""
        return self.version


@export(mod_name=Name, version=VERSION, api=True, name="ui", row=True)
def get_kernel_ui(app: App) -> Result:
    """Deliver the COOS Kernel Web UI"""

    # Load UI from file or return inline
    ui_path = Path(__file__).parent / "kernelcoos_ui.html"
    if ui_path.exists():
        with open(ui_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    else:
        # Inline minimal UI as fallback
        html_content = f"""
        {app.web_context()}
        <style>
            body {{ margin: 0; padding: 20px; font-family: system-ui; background: #0a0a0a; color: #fff; }}
            h1 {{ color: #10b981; }}
        </style>
        <h1>COOS Kernel</h1>
        <p>UI file not found. Please ensure kernelcoos_ui.html is in the same directory.</p>
        """

    return Result.html(data=html_content)


@export(mod_name=Name, version=VERSION, initial=True)
def init_kernel_coos(app: App = None):
    """Initialize the COOS Kernel module"""
    if app is None:
        app = get_app()
    app.run_any(("CloudM", "add_ui"),
                name=Name,
                title="COOS Kernel",
                path=f"/api/{Name}/ui",
                description="AI-powered voice assistant with COOS Kernel")
    return {"success": True, "info": "KernelCOOS initialized"}


@export(mod_name=Name, version=VERSION, websocket_handler="kernel")
def register_kernel_handlers(app: App) -> dict:
    """Register WebSocket handlers for COOS kernel"""
    global _kernel_instance

    # Create kernel instance on first registration
    if _kernel_instance is None:
        # Get ISAA and create agent

        # Create kernel
        _kernel_instance = COOSWebKernel(None, app, channel_id=f"{Name}/kernel")
        app.run_bg_task_advanced(_kernel_instance.start)

    return {
        "on_connect": _kernel_instance.handle_connect,
        "on_message": _kernel_instance.handle_message,
        "on_disconnect": _kernel_instance.handle_disconnect
    }


@export(mod_name=Name, version=VERSION, api=True, name="status")
async def get_kernel_status(app: App) -> Result:
    """Get COOS kernel status"""
    global _kernel_instance

    if _kernel_instance is None:
        return Result.json(data={"status": "not_initialized"})

    return Result.json(data={
        "status": "running" if _kernel_instance.running else "stopped",
        "kernel": _kernel_instance.kernel.to_dict(),
        "sessions": len(_kernel_instance.sessions),
        "connections": len(_kernel_instance.output_router.connections),
        "capabilities": {
            "voice_enabled": GROQ_AVAILABLE or OPENAI_AVAILABLE,
            "tts_enabled": OPENAI_AVAILABLE or ELEVENLABS_AVAILABLE,
            "vad_enabled": NUMPY_AVAILABLE
        }
    })


@export(mod_name=Name, version=VERSION, api=True, name="config", api_methods=["GET", "POST"], request_as_kwarg=True)
async def handle_config(app: App, request: RequestData = None) -> Result:
    """Get or update session configuration"""
    global _kernel_instance

    if _kernel_instance is None:
        return Result.default_internal_error(info="Kernel not initialized")

    if request and request.method == "POST":
        # Update config
        body = request.json() if hasattr(request, 'json') else {}
        session_id = body.get("session_id")
        config_updates = body.get("config", {})

        if session_id:
            _kernel_instance.update_session_config(session_id, config_updates)
            config = _kernel_instance.sessions.get(session_id)
            if config:
                return Result.json(data=config.model_dump())

        return Result.default_user_error(info="Session not found")
    else:
        # Get default config
        return Result.json(data=SessionConfig().model_dump())


if __name__ == "__main__":
    print("STARTED START FROM CLI")

