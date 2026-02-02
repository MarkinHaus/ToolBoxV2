"""
Discord Voice Mode - Voice Channel TTS und Listening
=====================================================

Erweitert das Discord Interface um Voice Channel Support:
- Voice Channel Join/Leave
- TTS Streaming in Voice (Satz für Satz)
- Voice Receive (benötigt discord-ext-voice-recv oder py-cord)
- Multi-Speaker Tracking
- Letzte 5 Minuten Conversation History

WICHTIG:
- discord.py hat KEIN built-in Voice Receive
- Für Voice Listening: pip install discord-ext-voice-recv
- Alternativ: py-cord statt discord.py (hat built-in Voice Receive)

Dependencies:
- discord.py[voice]>=2.6.4
- FFmpeg (muss im PATH sein)
- Optional: discord-ext-voice-recv für Listening

Author: Markin / ToolBoxV2
Version: 2.0.0
"""

import asyncio
import io
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

try:
    import discord
    from discord import VoiceClient
    from discord.ext import commands
except ImportError:
    raise ImportError("pip install discord.py[voice]>=2.6.4")

# TTS Integration (aus discord_interface.py)
from .discord_interface import MediaHandler, _load_tts

# Voice Receive Library (optional)
try:
    from discord.ext import voice_recv

    VOICE_RECV_AVAILABLE = True
except ImportError:
    VOICE_RECV_AVAILABLE = False
    voice_recv = None


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VoiceMessage:
    """Eine Sprachnachricht von einem User"""
    user_id: int
    user_name: str
    text: str  # Transkribierter Text
    timestamp: datetime
    audio_path: Optional[str] = None  # Temp Audio File
    duration_seconds: float = 0.0


@dataclass
class VoiceConversation:
    """Speichert die letzten 5 Minuten Conversation in einem Voice Channel"""
    channel_id: int
    channel_name: str
    guild_id: int
    messages: deque = field(default_factory=lambda: deque(maxlen=100))
    max_history_seconds: int = 300  # 5 Minuten

    def add_message(self, msg: VoiceMessage):
        """Fügt Message hinzu und entfernt alte"""
        self.messages.append(msg)
        self._cleanup_old()

    def _cleanup_old(self):
        """Entfernt Messages älter als max_history_seconds"""
        cutoff = datetime.now().timestamp() - self.max_history_seconds
        while self.messages and self.messages[0].timestamp.timestamp() < cutoff:
            self.messages.popleft()

    def get_context(self) -> str:
        """Gibt formatierte Conversation History zurück"""
        if not self.messages:
            return "[No recent voice conversation]"

        lines = []
        for msg in self.messages:
            time_str = msg.timestamp.strftime("%H:%M:%S")
            lines.append(f"[{time_str}] {msg.user_name}: {msg.text}")

        return "\n".join(lines)

    def get_participants(self) -> list[str]:
        """Gibt Liste der aktiven Teilnehmer zurück"""
        participants = set()
        for msg in self.messages:
            participants.add(msg.user_name)
        return list(participants)


# =============================================================================
# VOICE RECEIVE SINK
# =============================================================================

class UserAudioSink(discord.sinks.Sink if hasattr(discord, 'sinks') else object):
    """
    Custom Sink für discord-ext-voice-recv.
    Sammelt Audio per User und triggert Transcription nach Stille.
    """

    def __init__(self, handler: "VoiceHandler"):
        if hasattr(discord, 'sinks'):
            super().__init__()
        self.handler = handler
        self.audio_data: dict[int, bytearray] = {}  # user_id -> audio bytes
        self.last_packet_time: dict[int, float] = {}

    def write(self, data, user):
        """Called for each audio packet received"""
        if user is None:
            return

        user_id = user.id
        now = time.time()

        # Initialize buffer if needed
        if user_id not in self.audio_data:
            self.audio_data[user_id] = bytearray()

        # Add audio data
        self.audio_data[user_id].extend(data)
        self.last_packet_time[user_id] = now

        # Update handler timestamps
        self.handler._last_audio_time[user_id] = now

        # Cancel existing silence task
        if user_id in self.handler._silence_tasks:
            self.handler._silence_tasks[user_id].cancel()

        # Start new silence detection
        self.handler._silence_tasks[user_id] = asyncio.create_task(
            self._check_user_silence(user_id, user)
        )

    async def _check_user_silence(self, user_id: int, user):
        """Check if user stopped speaking"""
        await asyncio.sleep(self.handler.silence_threshold_ms / 1000.0)

        now = time.time()
        last = self.last_packet_time.get(user_id, 0)

        if (now - last) * 1000 >= self.handler.silence_threshold_ms:
            # User stopped speaking - process their audio
            await self._process_user_audio(user_id, user)

    async def _process_user_audio(self, user_id: int, user):
        """Process accumulated audio from a user"""
        if user_id not in self.audio_data:
            return

        audio_bytes = bytes(self.audio_data[user_id])
        self.audio_data[user_id] = bytearray()  # Clear buffer

        # Check minimum length (48kHz stereo 16-bit = 192000 bytes/sec)
        # Discord sends 48kHz stereo, so ~192KB = 1 second
        min_bytes = int(192000 * self.handler.min_audio_length_ms / 1000)

        if len(audio_bytes) < min_bytes:
            return

        # Save to temp file and transcribe
        temp_path = Path(tempfile.mktemp(suffix=".wav"))
        try:
            # Convert Discord audio format to WAV
            self._save_discord_audio_as_wav(audio_bytes, temp_path)

            # Transcribe
            transcription = await self.handler.media_handler.transcribe_audio(str(temp_path))

            if transcription and transcription.strip():
                user_name = user.display_name if hasattr(user, 'display_name') else f"User_{user_id}"

                # Create VoiceMessage
                msg = VoiceMessage(
                    user_id=user_id,
                    user_name=user_name,
                    text=transcription,
                    timestamp=datetime.now(),
                    duration_seconds=len(audio_bytes) / 192000,
                )

                print(f"[Voice] 🎤 {user_name}: {transcription}")

                # Add to conversations and trigger callback
                for conv in self.handler._conversations.values():
                    conv.add_message(msg)

                    if self.handler.on_voice_message:
                        try:
                            result = self.handler.on_voice_message(msg, conv)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception as e:
                            print(f"[Voice] Callback error: {e}")
                    break

        except Exception as e:
            print(f"[Voice] Audio processing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                temp_path.unlink()
            except:
                pass

    def _save_discord_audio_as_wav(self, audio_bytes: bytes, path: Path):
        """
        Convert Discord audio (48kHz stereo 16-bit PCM) to WAV.
        Resamples to 16kHz mono for better STT compatibility.
        """
        import wave

        try:
            import numpy as np

            # Discord: 48kHz, stereo, 16-bit signed
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

            # Convert stereo to mono (average channels)
            if len(audio_array) % 2 == 0:
                stereo = audio_array.reshape(-1, 2)
                mono = stereo.mean(axis=1).astype(np.int16)
            else:
                mono = audio_array

            # Resample 48kHz -> 16kHz (factor of 3)
            # Simple decimation - für bessere Qualität könnte man scipy.signal.resample nutzen
            resampled = mono[::3]

            # Save as WAV
            with wave.open(str(path), 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(resampled.tobytes())

        except ImportError:
            # Fallback ohne numpy - speichere direkt (weniger optimal)
            with wave.open(str(path), 'wb') as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(48000)
                wav.writeframes(audio_bytes)

    def cleanup(self):
        """Cleanup when sink is destroyed"""
        self.audio_data.clear()
        self.last_packet_time.clear()


# =============================================================================
# VOICE HANDLER
# =============================================================================

class VoiceHandler:
    """
    Handles Discord Voice Channel Interactions.

    Features:
    - Voice Channel Join/Leave
    - TTS Output (Sentence-by-Sentence Streaming)
    - Voice Receive (wenn discord-ext-voice-recv installiert)
    - Multi-Speaker Tracking
    - Conversation History (letzte 5 Minuten)
    """

    def __init__(
        self,
        bot: commands.Bot,
        media_handler: MediaHandler,
        on_voice_message: Optional[Callable[[VoiceMessage, VoiceConversation], Any]] = None,
    ):
        self.bot = bot
        self.media_handler = media_handler
        self.on_voice_message = on_voice_message

        # Voice Clients per Guild
        self._voice_clients: dict[int, VoiceClient] = {}  # guild_id -> VoiceClient

        # Conversation History per Channel
        self._conversations: dict[int, VoiceConversation] = {}  # channel_id -> VoiceConversation

        # Voice Receive State (wenn verfügbar)
        self._audio_buffers: dict[int, bytearray] = {}  # user_id -> audio buffer
        self._last_audio_time: dict[int, float] = {}  # user_id -> timestamp
        self._silence_tasks: dict[int, asyncio.Task] = {}

        # TTS Queue für sequential playback
        self._tts_queues: dict[int, asyncio.Queue] = {}  # guild_id -> Queue
        self._tts_tasks: dict[int, asyncio.Task] = {}

        # Settings
        self.silence_threshold_ms = 1500  # Stille = Ende der Aussage
        self.min_audio_length_ms = 500  # Mindestlänge für Processing

    # =========================================================================
    # VOICE CHANNEL MANAGEMENT
    # =========================================================================

    async def join_channel(self, channel: discord.VoiceChannel) -> dict:
        """
        Join a voice channel.

        Args:
            channel: Discord VoiceChannel to join

        Returns:
            Result dict with success/error
        """
        guild_id = channel.guild.id

        try:
            # Disconnect from existing if connected
            if guild_id in self._voice_clients:
                await self.leave_channel(guild_id)

            # Connect to channel
            vc = await channel.connect()
            self._voice_clients[guild_id] = vc

            # Initialize conversation
            self._conversations[channel.id] = VoiceConversation(
                channel_id=channel.id,
                channel_name=channel.name,
                guild_id=guild_id,
            )

            # Initialize TTS queue
            self._tts_queues[guild_id] = asyncio.Queue()
            self._tts_tasks[guild_id] = asyncio.create_task(
                self._tts_worker(guild_id)
            )

            # Start listening if available
            if VOICE_RECV_AVAILABLE:
                await self._start_listening(vc)
                return {
                    "success": True,
                    "channel_id": channel.id,
                    "channel_name": channel.name,
                    "listening": True,
                }
            else:
                return {
                    "success": True,
                    "channel_id": channel.id,
                    "channel_name": channel.name,
                    "listening": False,
                    "warning": "Voice receive not available. Install discord-ext-voice-recv for listening.",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def leave_channel(self, guild_id: int) -> dict:
        """
        Leave voice channel in a guild.

        Args:
            guild_id: Guild ID to leave

        Returns:
            Result dict
        """
        if guild_id not in self._voice_clients:
            return {"success": False, "error": "Not in voice channel"}

        try:
            vc = self._voice_clients[guild_id]
            channel_id = vc.channel.id if vc.channel else None

            # Stop TTS worker
            if guild_id in self._tts_tasks:
                self._tts_tasks[guild_id].cancel()
                del self._tts_tasks[guild_id]

            if guild_id in self._tts_queues:
                del self._tts_queues[guild_id]

            # Disconnect
            await vc.disconnect()
            del self._voice_clients[guild_id]

            # Clear conversation
            if channel_id and channel_id in self._conversations:
                del self._conversations[channel_id]

            return {"success": True, "guild_id": guild_id}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_voice_status(self, guild_id: int) -> dict:
        """Get voice connection status for a guild"""
        if guild_id not in self._voice_clients:
            return {
                "connected": False,
                "guild_id": guild_id,
            }

        vc = self._voice_clients[guild_id]
        channel = vc.channel
        conversation = self._conversations.get(channel.id) if channel else None

        return {
            "connected": vc.is_connected(),
            "guild_id": guild_id,
            "channel_id": channel.id if channel else None,
            "channel_name": channel.name if channel else None,
            "is_playing": vc.is_playing(),
            "is_paused": vc.is_paused(),
            "listening": VOICE_RECV_AVAILABLE,
            "participants": conversation.get_participants() if conversation else [],
            "conversation_messages": len(conversation.messages) if conversation else 0,
            "latency": vc.latency,
        }

    def is_connected(self, guild_id: int) -> bool:
        """Check if connected to voice in guild"""
        vc = self._voice_clients.get(guild_id)
        return vc is not None and vc.is_connected()

    def get_connected_channel(self, guild_id: int) -> Optional[discord.VoiceChannel]:
        """Get currently connected voice channel"""
        vc = self._voice_clients.get(guild_id)
        if vc and vc.is_connected():
            return vc.channel
        return None

    # =========================================================================
    # TTS OUTPUT
    # =========================================================================

    async def speak(self, guild_id: int, text: str, priority: int = 5):
        """
        Queue text to be spoken in voice channel.

        Args:
            guild_id: Guild ID
            text: Text to speak
            priority: Lower = higher priority (0-10)
        """
        if guild_id not in self._tts_queues:
            print(f"[Voice] Not connected to voice in guild {guild_id}")
            return

        # Queue for TTS worker
        await self._tts_queues[guild_id].put((priority, text))

    async def speak_streaming(self, guild_id: int, text: str):
        """
        Speak text sentence-by-sentence for more natural streaming.

        Args:
            guild_id: Guild ID
            text: Full text to speak
        """
        # Split into sentences
        sentences = self._split_sentences(text)

        for sentence in sentences:
            if sentence.strip():
                await self.speak(guild_id, sentence)

    async def _tts_worker(self, guild_id: int):
        """Background worker that processes TTS queue"""
        queue = self._tts_queues.get(guild_id)
        if not queue:
            return

        while True:
            try:
                # Get next text (with priority)
                priority, text = await queue.get()

                # Check if still connected
                if guild_id not in self._voice_clients:
                    break

                vc = self._voice_clients[guild_id]
                if not vc.is_connected():
                    break

                # Wait if currently playing
                while vc.is_playing():
                    await asyncio.sleep(0.1)

                # Generate and play TTS
                await self._play_tts(vc, text)

                queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Voice] TTS worker error: {e}")

    async def _play_tts(self, vc: VoiceClient, text: str):
        """Generate TTS and play in voice channel"""
        try:
            # Generate TTS audio
            audio_bytes = await self.media_handler.synthesize_speech(text)

            if not audio_bytes:
                print(f"[Voice] TTS generation failed for: {text[:50]}...")
                return

            # Save to temp file (FFmpeg needs file)
            temp_path = Path(tempfile.mktemp(suffix=".wav"))
            temp_path.write_bytes(audio_bytes)

            try:
                # Play audio
                source = discord.FFmpegPCMAudio(str(temp_path))
                vc.play(source)

                # Wait for playback to finish
                while vc.is_playing():
                    await asyncio.sleep(0.1)

            finally:
                # Cleanup temp file
                try:
                    temp_path.unlink()
                except:
                    pass

        except Exception as e:
            print(f"[Voice] TTS playback error: {e}")

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences for streaming"""
        import re

        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Merge very short sentences
        merged = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) < 100:
                current = f"{current} {sentence}".strip()
            else:
                if current:
                    merged.append(current)
                current = sentence

        if current:
            merged.append(current)

        return merged if merged else [text]

    async def stop_speaking(self, guild_id: int):
        """Stop current TTS playback"""
        if guild_id in self._voice_clients:
            vc = self._voice_clients[guild_id]
            if vc.is_playing():
                vc.stop()

        # Clear queue
        if guild_id in self._tts_queues:
            queue = self._tts_queues[guild_id]
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except:
                    break

    # =========================================================================
    # VOICE RECEIVE (erfordert discord-ext-voice-recv)
    # =========================================================================

    async def _start_listening(self, vc: VoiceClient):
        """Start listening for voice input mit discord-ext-voice-recv"""
        if not VOICE_RECV_AVAILABLE:
            print("[Voice] Voice receive not available - install discord-ext-voice-recv")
            return

        # Create custom sink that processes audio per user
        sink = UserAudioSink(self)

        # Start listening
        vc.start_recording(sink, self._on_recording_stopped)
        print(f"[Voice] Started listening in {vc.channel.name}")

    def _on_recording_stopped(self, sink, *args):
        """Callback when recording stops"""
        print("[Voice] Recording stopped")

    async def stop_listening(self, guild_id: int):
        """Stop listening in a guild"""
        if guild_id in self._voice_clients:
            vc = self._voice_clients[guild_id]
            if vc.is_connected():
                vc.stop_recording()
                print(f"[Voice] Stopped listening in guild {guild_id}")

    def _on_audio_packet(self, user_id: int, pcm_data: bytes):
        """Handle incoming audio packet from a user"""
        now = time.time()

        # Initialize buffer if needed
        if user_id not in self._audio_buffers:
            self._audio_buffers[user_id] = bytearray()

        # Add to buffer
        self._audio_buffers[user_id].extend(pcm_data)
        self._last_audio_time[user_id] = now

        # Cancel existing silence task
        if user_id in self._silence_tasks:
            self._silence_tasks[user_id].cancel()

        # Start new silence detection task
        self._silence_tasks[user_id] = asyncio.create_task(
            self._check_silence(user_id)
        )

    async def _check_silence(self, user_id: int):
        """Check if user has stopped speaking"""
        await asyncio.sleep(self.silence_threshold_ms / 1000.0)

        # Check if still silent
        now = time.time()
        last = self._last_audio_time.get(user_id, 0)

        if (now - last) * 1000 >= self.silence_threshold_ms:
            # User has stopped speaking - process audio
            await self._process_audio_buffer(user_id)

    async def _process_audio_buffer(self, user_id: int):
        """Process accumulated audio buffer"""
        if user_id not in self._audio_buffers:
            return

        buffer = self._audio_buffers[user_id]

        # Check minimum length
        # Assuming 16kHz, 16-bit mono: 32000 bytes = 1 second
        min_bytes = int(32000 * self.min_audio_length_ms / 1000)

        if len(buffer) < min_bytes:
            self._audio_buffers[user_id] = bytearray()
            return

        # Save to temp file
        temp_path = Path(tempfile.mktemp(suffix=".wav"))
        try:
            self._save_pcm_as_wav(bytes(buffer), temp_path)

            # Transcribe
            transcription = await self.media_handler.transcribe_audio(str(temp_path))

            if transcription and transcription.strip():
                # Get user info
                user = self.bot.get_user(user_id)
                user_name = user.display_name if user else f"User_{user_id}"

                # Create VoiceMessage
                msg = VoiceMessage(
                    user_id=user_id,
                    user_name=user_name,
                    text=transcription,
                    timestamp=datetime.now(),
                    duration_seconds=len(buffer) / 32000,  # Approximate
                )

                # Find conversation (from any connected channel)
                for conv in self._conversations.values():
                    conv.add_message(msg)

                    # Callback
                    if self.on_voice_message:
                        await self.on_voice_message(msg, conv)

                    break  # Only one conversation per user

        finally:
            # Cleanup
            self._audio_buffers[user_id] = bytearray()
            try:
                temp_path.unlink()
            except:
                pass

    def _save_pcm_as_wav(self, pcm_data: bytes, path: Path):
        """Save raw PCM data as WAV file"""
        import wave

        with wave.open(str(path), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(16000)  # Discord uses 48kHz but we resample
            wav.writeframes(pcm_data)

    # =========================================================================
    # CONVERSATION MANAGEMENT
    # =========================================================================

    def get_conversation(self, channel_id: int) -> Optional[VoiceConversation]:
        """Get conversation for a channel"""
        return self._conversations.get(channel_id)

    def get_conversation_context(self, guild_id: int) -> str:
        """Get formatted conversation context for a guild"""
        # Find conversation for this guild
        for conv in self._conversations.values():
            if conv.guild_id == guild_id:
                return conv.get_context()

        return "[No active voice conversation]"


# =============================================================================
# VOICE MODE INTEGRATION
# =============================================================================

class VoiceModeExtension:
    """
    Extension für DiscordInterface die Voice Mode hinzufügt.

    Usage:
        interface = DiscordInterface(agent, token)
        voice_ext = VoiceModeExtension(interface)

        # In CLI:
        /discord voice join <channel_id>
        /discord voice leave
        /discord voice status
    """

    def __init__(self, discord_interface):
        from .discord_interface import DiscordInterface

        self.interface: DiscordInterface = discord_interface

        # Create VoiceHandler
        self.voice_handler = VoiceHandler(
            bot=self.interface.bot,
            media_handler=self.interface.media_handler,
            on_voice_message=self._on_voice_message,
        )

        # Register voice events
        self._register_events()

        # Register voice tools
        self._register_tools()

    def _register_events(self):
        """Register voice-related Discord events"""

        @self.interface.bot.event
        async def on_voice_state_update(
            member: discord.Member,
            before: discord.VoiceState,
            after: discord.VoiceState,
        ):
            await self._handle_voice_state(member, before, after)

    async def _handle_voice_state(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ):
        """Handle voice state changes (user join/leave)"""
        if member.bot:
            return

        # User joined voice channel where bot is
        if after.channel:
            vc = self.voice_handler._voice_clients.get(member.guild.id)
            if vc and vc.channel and vc.channel.id == after.channel.id:
                print(f"[Voice] {member.display_name} joined {after.channel.name}")

        # User left voice channel where bot is
        if before.channel:
            vc = self.voice_handler._voice_clients.get(member.guild.id)
            if vc and vc.channel and vc.channel.id == before.channel.id:
                print(f"[Voice] {member.display_name} left {before.channel.name}")

    async def _on_voice_message(self, msg: VoiceMessage, conv: VoiceConversation):
        """Called when a voice message is transcribed"""
        print(f"[Voice] 🎤 {msg.user_name}: {msg.text}")

        # Optionen für automatische Agent-Antwort:
        # 1. Immer antworten wenn engaged
        # 2. Nur bei Keywords/Wake Words
        # 3. Nur bei direkter Ansprache

        # Für jetzt: Check ob Bot Name erwähnt wird
        bot_name = self.interface.bot.user.name.lower() if self.interface.bot.user else "bot"
        text_lower = msg.text.lower()

        # Wake words
        wake_words = [bot_name, "hey bot", "ok bot", "assistant", "agent"]
        should_respond = any(word in text_lower for word in wake_words)

        if should_respond:
            # Build context mit Voice Conversation History
            context = f"""[Voice Channel Context]
Channel: {conv.channel_name}
Participants: {', '.join(conv.get_participants())}
Recent conversation:
{conv.get_context()}

[Current Message]
{msg.user_name}: {msg.text}
"""
            try:
                # Call Agent
                response = await self.interface.agent.a_run(
                    query=context,
                    session_id=f"discord_voice_{conv.guild_id}",
                )

                if response:
                    # Speak response in voice channel
                    await self.voice_handler.speak_streaming(conv.guild_id, response)

            except Exception as e:
                print(f"[Voice] Agent error: {e}")
                import traceback
                traceback.print_exc()

    def _register_tools(self):
        """Register voice-related agent tools"""

        handler = self.voice_handler
        interface = self.interface

        async def discord_voice_join(channel_id: int) -> str:
            """
            Join a voice channel.

            Args:
                channel_id: Voice channel ID to join

            Returns:
                Result JSON
            """
            import json

            try:
                channel = interface.bot.get_channel(channel_id)
                if not channel:
                    channel = await interface.bot.fetch_channel(channel_id)

                if not isinstance(channel, discord.VoiceChannel):
                    return json.dumps({"error": "Not a voice channel"})

                result = await handler.join_channel(channel)
                return json.dumps(result)

            except Exception as e:
                return json.dumps({"error": str(e)})

        async def discord_voice_leave(guild_id: int) -> str:
            """
            Leave voice channel in a guild.

            Args:
                guild_id: Guild ID

            Returns:
                Result JSON
            """
            import json
            result = await handler.leave_channel(guild_id)
            return json.dumps(result)

        async def discord_voice_speak(guild_id: int, text: str, streaming: bool = True) -> str:
            """
            Speak text in voice channel via TTS.

            Args:
                guild_id: Guild ID
                text: Text to speak
                streaming: If True, speak sentence-by-sentence

            Returns:
                Result JSON
            """
            import json

            if not handler.is_connected(guild_id):
                return json.dumps({"error": "Not in voice channel"})

            if streaming:
                await handler.speak_streaming(guild_id, text)
            else:
                await handler.speak(guild_id, text)

            return json.dumps({"success": True, "text_length": len(text)})

        async def discord_voice_stop(guild_id: int) -> str:
            """
            Stop current TTS playback.

            Args:
                guild_id: Guild ID

            Returns:
                Result JSON
            """
            import json
            await handler.stop_speaking(guild_id)
            return json.dumps({"success": True})

        async def discord_voice_status(guild_id: int) -> str:
            """
            Get voice connection status.

            Args:
                guild_id: Guild ID

            Returns:
                Status JSON
            """
            import json
            return json.dumps(handler.get_voice_status(guild_id))

        async def discord_voice_context(guild_id: int) -> str:
            """
            Get recent voice conversation context (last 5 minutes).

            Args:
                guild_id: Guild ID

            Returns:
                Formatted conversation history
            """
            return handler.get_conversation_context(guild_id)

        # Register tools
        interface.agent.add_tool(
            discord_voice_join,
            "discord_voice_join",
            description="Join a Discord voice channel by ID. Required before you can speak in voice.",
            category=["discord", "voice"],
        )

        interface.agent.add_tool(
            discord_voice_leave,
            "discord_voice_leave",
            description="Leave the voice channel in a guild.",
            category=["discord", "voice"],
        )

        interface.agent.add_tool(
            discord_voice_speak,
            "discord_voice_speak",
            description="Speak text in voice channel using TTS. Set streaming=True for natural sentence-by-sentence output.",
            category=["discord", "voice"],
        )

        interface.agent.add_tool(
            discord_voice_stop,
            "discord_voice_stop",
            description="Stop current TTS playback in voice channel.",
            category=["discord", "voice"],
        )

        interface.agent.add_tool(
            discord_voice_status,
            "discord_voice_status",
            description="Get voice connection status including participants and playback state.",
            category=["discord", "voice"],
        )

        interface.agent.add_tool(
            discord_voice_context,
            "discord_voice_context",
            description="Get recent voice conversation context (last 5 minutes of transcribed speech).",
            category=["discord", "voice"],
        )

        print(
            "[Voice] Voice tools registered: discord_voice_join, discord_voice_leave, discord_voice_speak, discord_voice_stop, discord_voice_status, discord_voice_context")


# =============================================================================
# CLI COMMANDS
# =============================================================================

def get_voice_cli_commands() -> dict:
    """
    CLI Commands für Voice Mode.

    Returns:
        Dict für Completer: {subcommand: help}
    """
    return {
        "join": "Join voice channel: /discord voice join <channel_id>",
        "leave": "Leave voice channel: /discord voice leave",
        "status": "Voice status: /discord voice status",
        "speak": "Speak text: /discord voice speak <text>",
        "stop": "Stop speaking: /discord voice stop",
        "context": "Get conversation: /discord voice context",
        "listen": "Toggle listening: /discord voice listen [on|off]",
    }


async def handle_voice_cli_command(
    voice_ext: VoiceModeExtension,
    args: list[str],
    guild_id: Optional[int] = None,
) -> str:
    """
    Handle /discord voice <subcommand> from CLI.

    Args:
        voice_ext: VoiceModeExtension instance
        args: Command arguments after "voice"
        guild_id: Current guild context (if known)

    Returns:
        Response string
    """
    if not args:
        return "Voice commands: join, leave, status, speak, stop, context, listen"

    cmd = args[0].lower()
    handler = voice_ext.voice_handler

    if cmd == "join":
        if len(args) < 2:
            return "Usage: /discord voice join <channel_id>"

        try:
            channel_id = int(args[1])
            channel = voice_ext.interface.bot.get_channel(channel_id)

            if not channel:
                channel = await voice_ext.interface.bot.fetch_channel(channel_id)

            if not isinstance(channel, discord.VoiceChannel):
                return f"Error: {channel_id} is not a voice channel"

            result = await handler.join_channel(channel)

            if result.get("success"):
                listening = "✅ Listening" if result.get("listening") else "❌ Not listening"
                return f"✅ Joined {result.get('channel_name')} | {listening}"
            else:
                return f"❌ Error: {result.get('error')}"

        except ValueError:
            return "Error: Invalid channel ID"
        except Exception as e:
            return f"Error: {e}"

    elif cmd == "leave":
        if not guild_id:
            # Try to find any connected guild
            for gid in handler._voice_clients.keys():
                guild_id = gid
                break

        if not guild_id:
            return "Not in any voice channel"

        result = await handler.leave_channel(guild_id)

        if result.get("success"):
            return "✅ Left voice channel"
        else:
            return f"❌ Error: {result.get('error')}"

    elif cmd == "status":
        if not guild_id:
            # List all connections
            if not handler._voice_clients:
                return "Not connected to any voice channels"

            lines = ["Voice connections:"]
            for gid, vc in handler._voice_clients.items():
                status = handler.get_voice_status(gid)
                channel = status.get("channel_name", "Unknown")
                playing = "🔊" if status.get("is_playing") else "🔇"
                listening = "👂" if status.get("listening") else ""
                participants = status.get("participants", [])
                lines.append(f"  • {channel} {playing}{listening} ({len(participants)} users)")

            return "\n".join(lines)
        else:
            status = handler.get_voice_status(guild_id)
            if not status.get("connected"):
                return "Not connected"

            return f"""Voice Status:
  Channel: {status.get('channel_name')}
  Playing: {status.get('is_playing')}
  Listening: {status.get('listening')}
  Participants: {', '.join(status.get('participants', []))}
  Messages: {status.get('conversation_messages')}
  Latency: {status.get('latency', 0) * 1000:.0f}ms"""

    elif cmd == "speak":
        if len(args) < 2:
            return "Usage: /discord voice speak <text>"

        text = " ".join(args[1:])

        if not guild_id:
            for gid in handler._voice_clients.keys():
                guild_id = gid
                break

        if not guild_id:
            return "Not in any voice channel"

        await handler.speak_streaming(guild_id, text)
        return f"🔊 Speaking: {text[:50]}..."

    elif cmd == "stop":
        if not guild_id:
            for gid in handler._voice_clients.keys():
                guild_id = gid
                break

        if guild_id:
            await handler.stop_speaking(guild_id)
            return "⏹️ Stopped speaking"
        else:
            return "Not in any voice channel"

    elif cmd == "context":
        if not guild_id:
            for gid in handler._voice_clients.keys():
                guild_id = gid
                break

        if guild_id:
            context = handler.get_conversation_context(guild_id)
            return f"Voice Conversation (last 5 min):\n{context}"
        else:
            return "Not in any voice channel"

    elif cmd == "listen":
        if len(args) > 1:
            mode = args[1].lower()
            if mode == "on":
                return "Listening is automatic when joining voice channels"
            elif mode == "off":
                if guild_id:
                    await handler.stop_listening(guild_id)
                    return "🔇 Stopped listening"

        return f"Voice receive available: {VOICE_RECV_AVAILABLE}"

    else:
        return f"Unknown voice command: {cmd}\nAvailable: join, leave, status, speak, stop, context, listen"


# =============================================================================
# FACTORY
# =============================================================================

def create_voice_mode(discord_interface) -> VoiceModeExtension:
    """
    Factory function to add Voice Mode to a DiscordInterface.

    Args:
        discord_interface: DiscordInterface instance

    Returns:
        VoiceModeExtension instance

    Example:
        interface = create_discord_interface(agent, token)
        voice_mode = create_voice_mode(interface)

        # Now voice tools are available to the agent
        await interface.start()
    """
    return VoiceModeExtension(discord_interface)
