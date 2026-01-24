"""
Live Voice API Handler - Real-time voice conversation via WebSocket

Features:
- Bidirectional audio streaming (WebSocket)
- Wake word detection (pre/post/mid modes)
- Interrupt handling with [INTERRUPTED BY USER] marker
- Parallel LLM + TTS streaming for minimal latency
- Session management with SQLite persistence
"""

import asyncio
import base64
import json
import secrets
import time
import io
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, AsyncGenerator
from dataclasses import dataclass, field, asdict
from enum import Enum

import aiosqlite
import httpx
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field


# === Enums & Models ===

class WakeWordMode(str, Enum):
    PRE = "pre"  # Wake word at start, discard everything before
    POST = "post"  # Everything from speech start to wake word goes to LLM
    MID = "mid"  # Wake word must appear somewhere in utterance


class AudioFormat(str, Enum):
    WEBM = "webm"
    OPUS = "opus"
    WAV = "wav"
    PCM = "pcm"
    MP3 = "mp3"


class AudioConfig(BaseModel):
    input_format: AudioFormat = AudioFormat.WEBM
    output_format: AudioFormat = AudioFormat.OPUS
    sample_rate: int = 24000
    allow_interrupt: bool = True


class WakeWordConfig(BaseModel):
    enabled: bool = False
    words: List[str] = Field(default_factory=list)
    mode: WakeWordMode = WakeWordMode.PRE


class VoiceConfig(BaseModel):
    voice_id: str = "default"
    speed: float = 1.0
    language: str = "auto"


class LLMConfig(BaseModel):
    model: str
    system_prompt: str = "Du bist ein hilfreicher Assistent."
    tools: List[Dict] = Field(default_factory=list)
    history_length: int = 20
    temperature: float = 0.7
    max_tokens: int = 1024


class LiveSessionRequest(BaseModel):
    audio_config: AudioConfig = Field(default_factory=AudioConfig)
    wake_word_config: WakeWordConfig = Field(default_factory=WakeWordConfig)
    voice_config: VoiceConfig = Field(default_factory=VoiceConfig)
    llm_config: LLMConfig
    session_ttl_minutes: int = Field(default=15, ge=5, le=20)


class LiveSessionResponse(BaseModel):
    session_token: str
    websocket_url: str
    expires_at: str


# === Session State ===

@dataclass
class ConversationTurn:
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    interrupted: bool = False


@dataclass
class LiveSession:
    token: str
    user_id: int
    audio_config: AudioConfig
    wake_word_config: WakeWordConfig
    voice_config: VoiceConfig
    llm_config: LLMConfig
    history: List[ConversationTurn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    expires_at: float = 0.0
    is_active: bool = True

    # Runtime state (not persisted)
    is_generating: bool = False
    current_output: str = ""

    def to_dict(self) -> Dict:
        return {
            "token": self.token,
            "user_id": self.user_id,
            "audio_config": self.audio_config.model_dump(),
            "wake_word_config": self.wake_word_config.model_dump(),
            "voice_config": self.voice_config.model_dump(),
            "llm_config": self.llm_config.model_dump(),
            "history": [asdict(h) for h in self.history],
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "expires_at": self.expires_at,
            "is_active": self.is_active
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "LiveSession":
        return cls(
            token=data["token"],
            user_id=data["user_id"],
            audio_config=AudioConfig(**data["audio_config"]),
            wake_word_config=WakeWordConfig(**data["wake_word_config"]),
            voice_config=VoiceConfig(**data["voice_config"]),
            llm_config=LLMConfig(**data["llm_config"]),
            history=[ConversationTurn(**h) for h in data.get("history", [])],
            created_at=data["created_at"],
            last_activity=data["last_activity"],
            expires_at=data["expires_at"],
            is_active=data.get("is_active", True)
        )

    def get_messages_for_llm(self) -> List[Dict]:
        """Build message list for LLM, respecting history_length"""
        messages = []

        # System prompt
        if self.llm_config.system_prompt:
            messages.append({
                "role": "system",
                "content": self.llm_config.system_prompt
            })

        # Conversation history (last N turns)
        history_slice = self.history[-self.llm_config.history_length:]
        for turn in history_slice:
            content = turn.content
            if turn.interrupted:
                content += " [INTERRUPTED BY USER]"
            messages.append({
                "role": turn.role,
                "content": content
            })

        return messages

    def add_user_turn(self, text: str):
        """Add user message to history"""
        self.history.append(ConversationTurn(role="user", content=text))
        self.last_activity = time.time()

    def add_assistant_turn(self, text: str, interrupted: bool = False):
        """Add assistant message to history"""
        self.history.append(ConversationTurn(
            role="assistant",
            content=text,
            interrupted=interrupted
        ))
        self.last_activity = time.time()


# === Live Handler ===

class LiveHandler:
    """
    Manages live voice sessions with WebSocket connections.

    Architecture:
    - Sessions stored in SQLite for persistence
    - In-memory cache for active sessions
    - Lazy cleanup on new session creation
    """

    def __init__(self, db_path: str, model_manager):
        self.db_path = db_path
        self.model_manager = model_manager
        self.active_sessions: Dict[str, LiveSession] = {}
        self.active_websockets: Dict[str, WebSocket] = {}

    async def init_db(self):
        """Initialize live_sessions table"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                             CREATE TABLE IF NOT EXISTS live_sessions
                             (
                                 token
                                 TEXT
                                 PRIMARY
                                 KEY,
                                 user_id
                                 INTEGER
                                 NOT
                                 NULL,
                                 session_data
                                 TEXT
                                 NOT
                                 NULL,
                                 created_at
                                 TEXT
                                 DEFAULT
                                 CURRENT_TIMESTAMP,
                                 expires_at
                                 TEXT
                                 NOT
                                 NULL,
                                 is_active
                                 INTEGER
                                 DEFAULT
                                 1,
                                 FOREIGN
                                 KEY
                             (
                                 user_id
                             ) REFERENCES users
                             (
                                 id
                             )
                                 )
                             """)
            await db.execute("""
                             CREATE INDEX IF NOT EXISTS idx_live_sessions_user
                                 ON live_sessions(user_id)
                             """)
            await db.execute("""
                             CREATE INDEX IF NOT EXISTS idx_live_sessions_expires
                                 ON live_sessions(expires_at)
                             """)
            await db.commit()

    async def _cleanup_expired_sessions(self, user_id: int):
        """Remove expired sessions for user (lazy cleanup)"""
        now = datetime.utcnow().isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            # Get expired session tokens
            cursor = await db.execute(
                "SELECT token FROM live_sessions WHERE user_id = ? AND expires_at < ?",
                (user_id, now)
            )
            expired = [row[0] for row in await cursor.fetchall()]

            # Remove from memory
            for token in expired:
                self.active_sessions.pop(token, None)
                ws = self.active_websockets.pop(token, None)
                if ws:
                    try:
                        await ws.close(code=1000, reason="Session expired")
                    except:
                        pass

            # Remove from DB
            await db.execute(
                "DELETE FROM live_sessions WHERE user_id = ? AND expires_at < ?",
                (user_id, now)
            )
            await db.commit()

    async def create_session(
        self,
        request: LiveSessionRequest,
        user_id: int
    ) -> LiveSessionResponse:
        """Create new live session"""

        # Verify TTS model is available
        tts_slot = self.model_manager.find_tts_slot()
        if not tts_slot:
            raise HTTPException(
                503,
                "No TTS model loaded. Load a TTS model (type='tts') first."
            )

        # Verify LLM model is available
        llm_slot = self.model_manager.find_model_slot(request.llm_config.model)
        if not llm_slot:
            llm_slot = self.model_manager.find_text_slot()
        if not llm_slot:
            raise HTTPException(
                503,
                f"No LLM model available. Load model '{request.llm_config.model}' or any text model."
            )

        # Cleanup expired sessions for this user
        await self._cleanup_expired_sessions(user_id)

        # Generate session token
        token = f"live-{secrets.token_hex(16)}"

        # Calculate expiry
        ttl_minutes = max(5, min(20, request.session_ttl_minutes))
        expires_at = time.time() + (ttl_minutes * 60)

        # Create session
        session = LiveSession(
            token=token,
            user_id=user_id,
            audio_config=request.audio_config,
            wake_word_config=request.wake_word_config,
            voice_config=request.voice_config,
            llm_config=request.llm_config,
            expires_at=expires_at
        )

        # Store in memory and DB
        self.active_sessions[token] = session

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO live_sessions
                       (token, user_id, session_data, expires_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    token,
                    user_id,
                    json.dumps(session.to_dict()),
                    datetime.fromtimestamp(expires_at).isoformat()
                )
            )
            await db.commit()

        return LiveSessionResponse(
            session_token=token,
            websocket_url=f"/v1/audio/live/ws/{token}",
            expires_at=datetime.fromtimestamp(expires_at).isoformat()
        )

    async def get_session(self, token: str) -> Optional[LiveSession]:
        """Get session by token"""

        # Check memory first
        if token in self.active_sessions:
            session = self.active_sessions[token]
            if time.time() < session.expires_at:
                return session
            else:
                # Expired, remove
                del self.active_sessions[token]
                return None

        # Load from DB
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM live_sessions WHERE token = ? AND is_active = 1",
                (token,)
            )
            row = await cursor.fetchone()

            if not row:
                return None

            # Check expiry
            expires_at = datetime.fromisoformat(row["expires_at"])
            if datetime.utcnow() > expires_at:
                return None

            # Reconstruct session
            session_data = json.loads(row["session_data"])
            session = LiveSession.from_dict(session_data)

            # Cache in memory
            self.active_sessions[token] = session
            return session

    async def update_session(self, session: LiveSession):
        """Persist session changes to DB"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE live_sessions SET session_data = ? WHERE token = ?",
                (json.dumps(session.to_dict()), session.token)
            )
            await db.commit()

    async def close_session(self, token: str):
        """Close and cleanup session"""
        self.active_sessions.pop(token, None)

        ws = self.active_websockets.pop(token, None)
        if ws:
            try:
                await ws.close(code=1000, reason="Session closed")
            except:
                pass

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE live_sessions SET is_active = 0 WHERE token = ?",
                (token,)
            )
            await db.commit()

    # === WebSocket Handler ===

    async def handle_websocket(self, websocket: WebSocket, token: str):
        """Main WebSocket handler for live session"""

        # Validate session
        session = await self.get_session(token)
        if not session:
            await websocket.close(code=4004, reason="Invalid or expired session")
            return

        # Accept connection
        await websocket.accept()
        self.active_websockets[token] = websocket

        # Send ready message
        await websocket.send_json({
            "type": "ready",
            "session_token": token,
            "config": {
                "audio": session.audio_config.model_dump(),
                "wake_word": session.wake_word_config.model_dump(),
                "voice": session.voice_config.model_dump()
            }
        })

        try:
            while True:
                # Receive message
                message = await websocket.receive_json()
                msg_type = message.get("type")

                if msg_type == "audio":
                    # Client sends audio chunk with transcription
                    await self._handle_audio_message(websocket, session, message)

                elif msg_type == "end_turn":
                    # Client signals end of speech
                    await self._handle_end_turn(websocket, session, message)

                elif msg_type == "interrupt":
                    # Client interrupts current output
                    await self._handle_interrupt(websocket, session)

                elif msg_type == "ping":
                    # Keep-alive
                    session.last_activity = time.time()
                    await websocket.send_json({"type": "pong"})

                elif msg_type == "close":
                    # Client requests close
                    break

        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
            except:
                pass
        finally:
            self.active_websockets.pop(token, None)
            await self.update_session(session)

    async def _handle_audio_message(
        self,
        websocket: WebSocket,
        session: LiveSession,
        message: Dict
    ):
        """Handle incoming audio chunk with transcription from client"""
        # Client does VAD, so we receive transcribed text
        # Audio is optional (for potential server-side processing)

        transcript = message.get("transcript", "")
        is_final = message.get("is_final", False)

        if transcript:
            # Send acknowledgment
            await websocket.send_json({
                "type": "transcript",
                "text": transcript,
                "is_input": True,
                "is_final": is_final
            })

    async def _handle_end_turn(
        self,
        websocket: WebSocket,
        session: LiveSession,
        message: Dict
    ):
        """Handle end of user turn - process with LLM and TTS"""

        transcript = message.get("transcript", "")

        if not transcript.strip():
            await websocket.send_json({
                "type": "error",
                "message": "Empty transcript"
            })
            return

        # Process wake word if enabled
        processed_text = self._process_wake_word(transcript, session.wake_word_config)

        if processed_text is None:
            # Wake word not found when required
            await websocket.send_json({
                "type": "wake_word_required",
                "message": "Wake word not detected"
            })
            return

        if not processed_text.strip():
            await websocket.send_json({
                "type": "error",
                "message": "No content after wake word processing"
            })
            return

        # Add to history
        session.add_user_turn(processed_text)
        session.is_generating = True
        session.current_output = ""

        # Track usage
        start_time = time.time()
        tokens_in = 0
        tokens_out = 0
        tools_called = []

        try:
            # Get LLM slot
            llm_slot = self.model_manager.find_model_slot(session.llm_config.model)
            if not llm_slot:
                llm_slot = self.model_manager.find_text_slot()

            if not llm_slot:
                raise Exception("No LLM model available")

            # Get TTS slot
            tts_slot = self.model_manager.find_tts_slot()
            if not tts_slot:
                raise Exception("No TTS model available")

            # Build LLM request
            messages = session.get_messages_for_llm()

            llm_url = f"http://127.0.0.1:{llm_slot['port']}/v1/chat/completions"
            tts_url = f"http://127.0.0.1:{tts_slot['port']}/v1/audio/speech"

            # Stream LLM response
            full_response = ""
            sentence_buffer = ""
            interrupted = False

            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "model": session.llm_config.model,
                    "messages": messages,
                    "temperature": session.llm_config.temperature,
                    "max_tokens": session.llm_config.max_tokens,
                    "stream": True
                }

                if session.llm_config.tools:
                    payload["tools"] = session.llm_config.tools

                async with client.stream("POST", llm_url, json=payload) as resp:
                    async for line in resp.aiter_lines():
                        # Check for interrupt
                        if not session.is_generating:
                            interrupted = True
                            break

                        if not line.startswith("data: "):
                            continue

                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)

                            # Handle tool calls
                            if chunk.get("choices", [{}])[0].get("delta", {}).get("tool_calls"):
                                tool_call = chunk["choices"][0]["delta"]["tool_calls"][0]
                                if tool_call.get("function", {}).get("name"):
                                    tools_called.append(tool_call["function"]["name"])

                            # Get content
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                full_response += content
                                session.current_output = full_response
                                sentence_buffer += content
                                tokens_out += 1

                                # Send text chunk to client
                                await websocket.send_json({
                                    "type": "text",
                                    "text": content,
                                    "is_final": False
                                })

                                # Check for complete sentence for TTS
                                sentences = self._extract_sentences(sentence_buffer)
                                for sentence in sentences[:-1]:  # All complete sentences
                                    if sentence.strip():
                                        # Generate and stream TTS
                                        await self._stream_tts(
                                            websocket,
                                            tts_url,
                                            sentence.strip(),
                                            session
                                        )

                                # Keep incomplete sentence in buffer
                                sentence_buffer = sentences[-1] if sentences else ""

                            # Usage info
                            if chunk.get("usage"):
                                tokens_in = chunk["usage"].get("prompt_tokens", tokens_in)
                                tokens_out = chunk["usage"].get("completion_tokens", tokens_out)

                        except json.JSONDecodeError:
                            continue

                # Process remaining buffer
                if sentence_buffer.strip() and session.is_generating:
                    await self._stream_tts(
                        websocket,
                        tts_url,
                        sentence_buffer.strip(),
                        session
                    )

            # Add to history
            session.add_assistant_turn(full_response, interrupted=interrupted)

            # Calculate cost
            pricing = {"input_per_1k": 0.0001, "output_per_1k": 0.0002}
            cost = (tokens_in / 1000 * pricing["input_per_1k"] +
                    tokens_out / 1000 * pricing["output_per_1k"])

            # Send turn complete
            await websocket.send_json({
                "type": "turn_complete",
                "input_transcript": processed_text,
                "output_transcript": full_response,
                "tools_called": tools_called,
                "interrupted": interrupted,
                "usage": {
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out
                },
                "cost": cost,
                "latency_ms": int((time.time() - start_time) * 1000)
            })

        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })

        finally:
            session.is_generating = False
            session.current_output = ""
            await self.update_session(session)

    async def _handle_interrupt(self, websocket: WebSocket, session: LiveSession):
        """Handle interrupt from client"""

        if not session.is_generating:
            return

        # Stop generation
        session.is_generating = False
        partial_output = session.current_output

        # Send interrupted acknowledgment
        await websocket.send_json({
            "type": "interrupted",
            "partial_output": partial_output
        })

    def _process_wake_word(
        self,
        text: str,
        config: WakeWordConfig
    ) -> Optional[str]:
        """
        Process wake word based on mode.
        Returns processed text or None if wake word required but not found.
        """

        if not config.enabled or not config.words:
            return text

        text_lower = text.lower()
        found_word = None
        found_pos = -1

        # Find wake word
        for word in config.words:
            pos = text_lower.find(word.lower())
            if pos != -1:
                found_word = word
                found_pos = pos
                break

        if found_word is None:
            return None  # Wake word required but not found

        word_end = found_pos + len(found_word)

        if config.mode == WakeWordMode.PRE:
            # Wake word must be at start (with some tolerance)
            # Return everything after the wake word
            if found_pos <= 10:  # Allow up to 10 chars before (filler words)
                return text[word_end:].strip()
            return None

        elif config.mode == WakeWordMode.POST:
            # Return everything from start to wake word
            return text[:found_pos].strip()

        elif config.mode == WakeWordMode.MID:
            # Wake word just needs to be present
            # Return full text without the wake word
            result = text[:found_pos] + text[word_end:]
            # Clean up double spaces
            return ' '.join(result.split())

        return text

    def _extract_sentences(self, text: str) -> List[str]:
        """Split text into sentences for TTS streaming"""
        # Split on sentence boundaries
        pattern = r'(?<=[.!?])\s+'
        parts = re.split(pattern, text)
        return parts if parts else [text]

    async def _stream_tts(
        self,
        websocket: WebSocket,
        tts_url: str,
        text: str,
        session: LiveSession
    ):
        """Generate TTS and stream audio to client"""

        if not session.is_generating:
            return

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # TTS request (OpenAI compatible)
                payload = {
                    "model": "tts",
                    "input": text,
                    "voice": session.voice_config.voice_id,
                    "speed": session.voice_config.speed,
                    "response_format": session.audio_config.output_format.value
                }

                # Stream TTS response
                async with client.stream("POST", tts_url, json=payload) as resp:
                    if resp.status_code != 200:
                        return

                    async for chunk in resp.aiter_bytes(chunk_size=4096):
                        if not session.is_generating:
                            break

                        # Send audio chunk
                        await websocket.send_json({
                            "type": "audio",
                            "audio": base64.b64encode(chunk).decode(),
                            "format": session.audio_config.output_format.value,
                            "text": text
                        })

        except Exception as e:
            # TTS error - continue without audio
            print(f"TTS error: {e}")


# === Factory Function ===

def create_live_handler(db_path: str, model_manager) -> LiveHandler:
    """Create and return LiveHandler instance"""
    return LiveHandler(db_path, model_manager)
