"""
ProA Kernel Discord Interface
===============================

Production-ready Discord interface for the ProA Kernel with:
- Auto-persistence (save/load on start/stop)
- Full media support (attachments, embeds, images)
- Rich embeds with colors and fields
- Reaction support
- Thread support
- Voice channel support (requires PyNaCl)
- Voice input/transcription (requires discord-ext-voice-recv + Groq)
- Voice state tracking
- Slash commands integration

Installation:
-------------
1. Basic voice support (join/leave channels):
    pip install discord.py[voice]

2. Voice input/transcription support:
    pip install discord-ext-voice-recv groq

3. Set environment variable:
    export GROQ_API_KEY="your_groq_api_key"

Voice Commands:
---------------
- !join - Join your current voice channel
- !leave - Leave the voice channel
- !voice_status - Show voice connection status
- !listen - Start listening and transcribing voice input (requires Groq)
- !stop_listening - Stop listening to voice input

Voice Features:
---------------
- Real-time voice transcription using Groq Whisper (whisper-large-v3-turbo)
- Automatic language detection
- Transcriptions sent directly to kernel as user input
- Multi-user support (tracks each speaker separately)
- Configurable transcription interval (default: 3 seconds)

Voice Events:
-------------
- Tracks when users join/leave/move between voice channels
- Sends signals to kernel for voice state changes

Limitations:
------------
- Discord bots CANNOT initiate private calls (Discord API limitation)
- Bots can only join guild voice channels
- Bots can join DM voice channels only if invited by a user
"""

import asyncio
import os
import random
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
from collections import defaultdict

try:
    import discord
    from discord.ext import commands
    from discord.ui import View, Button, Select
    # Check for voice support
    try:
        import nacl
        VOICE_SUPPORT = True
    except ImportError:
        VOICE_SUPPORT = False
        print("⚠️ PyNaCl not installed. Voice support disabled. Install with: pip install discord.py[voice]")

    # Check for voice receive support
    try:
        from discord.ext import voice_recv
        VOICE_RECEIVE_SUPPORT = True
    except ImportError:
        VOICE_RECEIVE_SUPPORT = False
        print("⚠️ discord-ext-voice-recv not installed. Voice input disabled. Install with: pip install discord-ext-voice-recv")

except ImportError:
    print("⚠️ discord.py not installed. Install with: pip install discord.py[voice]")
    discord = None
    commands = None
    VOICE_SUPPORT = False
    VOICE_RECEIVE_SUPPORT = False

# Check for Groq API
try:
    from groq import Groq
    GROQ_SUPPORT = True
except ImportError:
    GROQ_SUPPORT = False
    print("⚠️ Groq not installed. Voice transcription disabled. Install with: pip install groq")

# Check for ElevenLabs
try:
    from elevenlabs import ElevenLabs
    ELEVENLABS_SUPPORT = True
except ImportError:
    ELEVENLABS_SUPPORT = False
    print("⚠️ ElevenLabs not installed. TTS disabled. Install with: pip install elevenlabs")

PIPER_SUPPORT = False


from toolboxv2 import App, get_app
from toolboxv2.mods.isaa.kernel.instace import Kernel
from toolboxv2.mods.isaa.kernel.types import Signal as KernelSignal, SignalType, KernelConfig, IOutputRouter
from toolboxv2.mods.isaa.kernel.kernelin.tools.discord_tools import DiscordKernelTools
from toolboxv2.mods.isaa.base.Agent.types import ProgressEvent, NodeStatus
import io
import wave
import tempfile
import os
import subprocess
import time


class WhisperAudioSink(voice_recv.AudioSink if VOICE_RECEIVE_SUPPORT else object):
    """Audio sink for receiving and transcribing voice input with Groq Whisper + VAD"""

    def __init__(self, kernel: Kernel, user_id: str, groq_client: 'Groq' = None, output_router=None, discord_kernel=None):
        print(f"🎤 [DEBUG] Initializing WhisperAudioSink for user {user_id}")

        if VOICE_RECEIVE_SUPPORT:
            super().__init__()
            print(f"🎤 [DEBUG] Voice receive support enabled")
        else:
            print(f"🎤 [DEBUG] WARNING: Voice receive support NOT enabled!")

        self.kernel = kernel
        self.user_id = user_id
        self.groq_client = groq_client
        self.output_router = output_router
        self.discord_kernel = discord_kernel  # Reference to DiscordKernel for context
        self.audio_buffer: Dict[str, List[bytes]] = {}  # user_id -> audio chunks
        self.transcription_interval = 3.0  # Transcribe every 3 seconds
        self.last_transcription: Dict[str, float] = {}  # user_id -> timestamp
        self.speaking_state: Dict[str, bool] = {}  # user_id -> is_speaking
        self.last_audio_time: Dict[str, float] = {}  # user_id -> last audio timestamp
        self.silence_threshold = 1.0  # 1 second of silence before stopping transcription

        # Voice channel history for group calls (15 minute window)
        self.voice_channel_history: Dict[str, List[dict]] = {}  # channel_id -> list of history entries
        self.history_max_age = 900  # 15 minutes in seconds

        print(f"🎤 [DEBUG] WhisperAudioSink initialized successfully")
        print(f"🎤 [DEBUG] - Groq client: {'✅' if groq_client else '❌'}")
        print(f"🎤 [DEBUG] - Transcription interval: {self.transcription_interval}s")
        print(f"🎤 [DEBUG] - Voice channel history: 15 minute window")

    def wants_opus(self) -> bool:
        """We want decoded PCM audio, not Opus"""
        return False

    def write(self, user, data):
        """Receive audio data from Discord"""
        if not user:
            print(f"🎤 [DEBUG] write() called with no user")
            return

        user_id = str(user.id)

        # Debug: Print data attributes
        if user_id not in self.audio_buffer:
            print(f"🎤 [DEBUG] First audio packet from {user.display_name} (ID: {user_id})")
            print(f"🎤 [DEBUG] Data type: {type(data)}")
            print(f"🎤 [DEBUG] Data attributes: {dir(data)}")
            if hasattr(data, 'pcm'):
                print(f"🎤 [DEBUG] PCM data size: {len(data.pcm)} bytes")

        # Buffer audio data
        if user_id not in self.audio_buffer:
            self.audio_buffer[user_id] = []
            self.last_transcription[user_id] = time.time()
            print(f"🎤 [DEBUG] Created new audio buffer for {user.display_name} (ID: {user_id})")

        # Append PCM audio data
        if hasattr(data, 'pcm'):
            self.audio_buffer[user_id].append(data.pcm)
        else:
            print(f"🎤 [DEBUG] WARNING: No PCM data in packet from {user.display_name}")
            return

        buffer_size = len(self.audio_buffer[user_id])

        # Only print every 10 chunks to avoid spam
        if buffer_size % 10 == 0:
            print(f"🎤 [DEBUG] Audio buffer for {user.display_name}: {buffer_size} chunks")

        # Check if we should transcribe
        current_time = time.time()
        if current_time - self.last_transcription[user_id] >= self.transcription_interval:
            time_since_last = current_time - self.last_transcription[user_id]
            print(f"🎤 [DEBUG] Triggering transcription for {user.display_name} (buffer: {buffer_size} chunks, time since last: {time_since_last:.2f}s)")

            # Schedule transcription in the event loop (write() is called from a different thread)
            try:
                from toolboxv2 import get_app
                get_app().run_bg_task_advanced(self._transcribe_buffer, user_id, user)
                # loop = asyncio.get_event_loop()
                # asyncio.run_coroutine_threadsafe(self._transcribe_buffer(user_id, user), loop)
            except Exception as e:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.run_coroutine_threadsafe(self._transcribe_buffer(user_id, user), loop)
                except Exception as e:
                    print(f"❌ [DEBUG] Error scheduling transcription: {e}")

            self.last_transcription[user_id] = current_time

    def _cleanup_old_history(self, channel_id: str):
        """Remove history entries older than max_age (15 minutes)"""
        if channel_id not in self.voice_channel_history:
            return

        current_time = time.time()
        cutoff_time = current_time - self.history_max_age

        # Filter out old entries
        original_count = len(self.voice_channel_history[channel_id])
        self.voice_channel_history[channel_id] = [
            entry for entry in self.voice_channel_history[channel_id]
            if entry["timestamp"] > cutoff_time
        ]

        removed_count = original_count - len(self.voice_channel_history[channel_id])
        if removed_count > 0:
            print(f"🗑️ [HISTORY] Cleaned up {removed_count} old entries from channel {channel_id}")

    def _add_to_history(self, channel_id: str, user_name: str, user_id: str, text: str, language: str):
        """Add a transcription to voice channel history"""
        if channel_id not in self.voice_channel_history:
            self.voice_channel_history[channel_id] = []

        entry = {
            "user": user_name,
            "user_id": user_id,
            "text": text,
            "timestamp": time.time(),
            "language": language
        }

        self.voice_channel_history[channel_id].append(entry)
        print(f"📝 [HISTORY] Added to channel {channel_id}: [{user_name}] {text}")

        # Cleanup old entries
        self._cleanup_old_history(channel_id)

    def _format_history(self, channel_id: str) -> str:
        """Format voice channel history for agent context"""
        if channel_id not in self.voice_channel_history or not self.voice_channel_history[channel_id]:
            return ""

        # Cleanup before formatting
        self._cleanup_old_history(channel_id)

        history_entries = self.voice_channel_history[channel_id]
        if not history_entries:
            return ""

        # Format as readable history
        lines = ["Voice Channel Recent History (last 15 minutes):"]
        for entry in history_entries:
            timestamp = entry["timestamp"]
            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            user = entry["user"]
            text = entry["text"]
            lines.append(f"[{time_str}] {user}: {text}")

        formatted = "\n".join(lines)
        print(f"📋 [HISTORY] Formatted {len(history_entries)} history entries for channel {channel_id}")
        return formatted

    async def _transcribe_buffer(self, user_id: str, user):
        """Transcribe buffered audio for a user"""
        print(f"🎤 [DEBUG] _transcribe_buffer called for user {user.display_name} (ID: {user_id})")

        if user_id not in self.audio_buffer or not self.audio_buffer[user_id]:
            print(f"🎤 [DEBUG] No audio buffer found for user {user_id}")
            return

        if not GROQ_SUPPORT or not self.groq_client:
            print("⚠️ [DEBUG] Groq not available for transcription")
            return

        try:
            print(f"🎤 [DEBUG] Processing audio for {user.display_name}")

            # Combine audio chunks
            audio_data = b''.join(self.audio_buffer[user_id])
            chunk_count = len(self.audio_buffer[user_id])
            self.audio_buffer[user_id] = []  # Clear buffer

            print(f"🎤 [DEBUG] Combined {chunk_count} audio chunks, total size: {len(audio_data)} bytes")

            # Calculate audio duration (48kHz stereo, 16-bit = 192000 bytes/second)
            duration_seconds = len(audio_data) / 192000
            print(f"🎤 [DEBUG] Audio duration: {duration_seconds:.2f} seconds")

            # Skip if too short (less than 0.5 seconds - likely just noise)
            if duration_seconds < 0.5:
                print(f"🎤 [DEBUG] Audio too short ({duration_seconds:.2f}s), skipping transcription")
                return

            # Skip if too few chunks (less than 5 chunks - likely just background noise)
            if chunk_count < 5:
                print(f"🎤 [DEBUG] Too few audio chunks ({chunk_count}), likely background noise, skipping")
                return

            # Create WAV file in memory
            print(f"🎤 [DEBUG] Creating WAV file (48kHz, stereo, 16-bit)")
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(2)  # Stereo
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(48000)  # Discord uses 48kHz
                wav_file.writeframes(audio_data)

            wav_buffer.seek(0)
            wav_size = len(wav_buffer.getvalue())
            print(f"🎤 [DEBUG] WAV file created, size: {wav_size} bytes")

            # Save to temporary file (Groq API needs file path)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(wav_buffer.read())
                temp_path = temp_file.name

            print(f"🎤 [DEBUG] Saved to temp file: {temp_path}")

            try:
                # Transcribe with Groq Whisper
                print(f"🎤 [DEBUG] Sending to Groq Whisper API (model: whisper-large-v3-turbo)...")
                with open(temp_path, 'rb') as audio_file:
                    transcription = self.groq_client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-large-v3-turbo",
                        response_format="json",
                        temperature=0.0
                    )

                print(f"🎤 [DEBUG] Groq API response received")

                text = transcription.text.strip()
                language = getattr(transcription, 'language', 'unknown')

                print(f"🎤 [DEBUG] Transcription result: '{text}' (language: {language})")

                # Filter out common Whisper hallucinations for background noise
                hallucinations = [
                    "thank you", "thanks for watching", "thank you for watching",
                    "bye", "goodbye", "see you", "see you next time",
                    "subscribe", "like and subscribe",
                    ".", "..", "...",
                    "you", "uh", "um", "hmm", "mhm",
                    "music", "[music]", "(music)",
                    "applause", "[applause]", "(applause)",
                    "laughter", "[laughter]", "(laughter)"
                ]

                text_lower = text.lower()
                is_hallucination = any(text_lower == h or text_lower.strip('.,!? ') == h for h in hallucinations)

                if is_hallucination:
                    print(f"🎤 [DEBUG] Detected hallucination/noise: '{text}', skipping")
                    return

                if text and len(text) > 2:  # At least 3 characters
                    print(f"🎤 [DEBUG] Text is not empty, processing...")

                    # ===== STOP COMMAND DETECTION =====
                    # Check if user said "stop" to stop playback
                    text_lower = text.lower().strip()
                    if text_lower in ["stop", "stopp", "halt", "pause"]:
                        print(f"🛑 [VOICE] Stop command detected from {user.display_name}")

                        # Stop playback if active
                        guild_id = user.guild.id if hasattr(user, 'guild') else None
                        if guild_id and guild_id in self.output_router.voice_clients:
                            voice_client = self.output_router.voice_clients[guild_id]
                            if voice_client.is_playing():
                                voice_client.stop()
                                print(f"🛑 [VOICE] Stopped playback in guild {guild_id}")
                            else:
                                print(f"🛑 [VOICE] No active playback to stop")

                        # Don't process this as a regular input
                        return

                    # ===== VOICE CHANNEL HISTORY TRACKING =====
                    # Get voice channel ID for history tracking
                    voice_channel_id = None
                    guild_id = user.guild.id if hasattr(user, 'guild') else None
                    if guild_id and guild_id in self.output_router.voice_clients:
                        voice_client = self.output_router.voice_clients[guild_id]
                        if voice_client and voice_client.channel:
                            voice_channel_id = str(voice_client.channel.id)

                    # ALWAYS add transcription to history (even without wake word)
                    # This allows agent to reference previous context when called
                    if voice_channel_id:
                        self._add_to_history(
                            channel_id=voice_channel_id,
                            user_name=user.display_name,
                            user_id=str(user.id),
                            text=text,
                            language=language
                        )

                    # ===== WAKE WORD DETECTION FOR GROUP CALLS =====
                    # Check if multiple users are in the voice channel
                    should_process = True  # Default: process in single-user calls
                    voice_channel_history = ""  # Will be populated if wake word detected

                    if guild_id and guild_id in self.output_router.voice_clients:
                        voice_client = self.output_router.voice_clients[guild_id]
                        if voice_client and voice_client.channel:
                            # Count non-bot members in voice channel
                            members_in_voice = [m for m in voice_client.channel.members if not m.bot]
                            is_group_call = len(members_in_voice) > 1

                            print(f"🎤 [DEBUG] Voice channel members: {len(members_in_voice)} (group call: {is_group_call})")

                            if is_group_call:
                                # Check for wake words (case-insensitive)
                                wake_words = [
                                    "agent", "toolbox", "isaa", "bot", "isabot", "isa", "issa",
                                    # German variants
                                    "assistent", "assistant"
                                ]

                                # Check if any wake word is in the text
                                has_wake_word = any(wake_word in text_lower for wake_word in wake_words)

                                if not has_wake_word:
                                    print(f"🎤 [DEBUG] Group call detected but no wake word found, storing in history only: '{text}'")
                                    should_process = False  # Don't send to agent
                                else:
                                    print(f"🎤 [DEBUG] Wake word detected in group call: '{text}'")

                                    # Get voice channel history for context
                                    if voice_channel_id:
                                        voice_channel_history = self._format_history(voice_channel_id)
                                        if voice_channel_history:
                                            print(f"📋 [HISTORY] Including {len(self.voice_channel_history[voice_channel_id])} history entries in context")

                                    # Remove wake word from text for cleaner processing
                                    for wake_word in wake_words:
                                        text = text.replace(wake_word, "").replace(wake_word.capitalize(), "")
                                    text = text.strip()
                                    print(f"🎤 [DEBUG] Text after wake word removal: '{text}'")

                    # If we shouldn't process (group call without wake word), return early
                    if not should_process:
                        return

                    # Get Discord context if available
                    discord_context = None
                    if self.discord_kernel and hasattr(user, 'guild'):
                        print(f"🎤 [DEBUG] Getting Discord context for {user.display_name}")

                        # Create a mock message object for context gathering
                        class MockMessage:
                            def __init__(self, author, guild, channel):
                                self.author = author
                                self.guild = guild
                                self.channel = channel
                                self.id = 0
                                self.attachments = []

                        # Get voice channel
                        if hasattr(user, 'voice') and user.voice:
                            voice_channel = user.voice.channel if user.voice else None
                            if voice_channel:
                                print(f"🎤 [DEBUG] User is in voice channel: {voice_channel.name}")
                                mock_msg = MockMessage(user, user.guild, voice_channel)
                                discord_context = self.discord_kernel._get_discord_context(mock_msg)

                                # Inject context into agent's variable system
                                if hasattr(self.kernel.agent, 'variable_manager'):
                                    self.kernel.agent.variable_manager.set(
                                        f'discord.current_context.{str(user.id)}',
                                        discord_context
                                    )
                                    print(f"🎤 [DEBUG] Discord context injected into agent")

                    # Register user channel for responses (use voice channel's text channel)
                    if hasattr(user, 'voice') and user.voice and user.voice.channel:
                        # Find the guild's text channels and use the first one, or system channel
                        guild = user.guild
                        text_channel = None

                        # Try to find a general/main text channel
                        if guild.system_channel:
                            text_channel = guild.system_channel
                        else:
                            # Use first available text channel
                            text_channels = [ch for ch in guild.text_channels if ch.permissions_for(guild.me).send_messages]
                            if text_channels:
                                text_channel = text_channels[0]

                        if text_channel:
                            self.output_router.user_channels[str(user.id)] = text_channel
                            print(f"🎤 [DEBUG] Registered text channel '{text_channel.name}' for user {user.display_name}")
                        else:
                            print(f"🎤 [DEBUG] WARNING: No text channel found for responses")

                    # Determine output mode (TTS or Text)
                    guild_id = user.guild.id if hasattr(user, 'guild') else None
                    tts_enabled = guild_id and guild_id in self.output_router.tts_enabled and self.output_router.tts_enabled[guild_id]
                    in_voice = guild_id and guild_id in self.output_router.voice_clients and self.output_router.voice_clients[guild_id].is_connected()
                    output_mode = "tts" if (tts_enabled and in_voice) else "text"

                    print(f"🎤 [DEBUG] Output mode: {output_mode} (TTS: {tts_enabled}, In Voice: {in_voice})")

                    # Inject output mode into agent's variable system
                    if hasattr(self.kernel.agent, 'variable_manager'):
                        self.kernel.agent.variable_manager.set(
                            f'discord.output_mode.{str(user.id)}',
                            output_mode
                        )

                        # Set formatting instructions based on output mode
                        if output_mode == "tts":
                            formatting_instructions = (
                                "IMPORTANT: You are responding via Text-to-Speech (TTS). "
                                "Use ONLY plain text. NO emojis, NO formatting, NO abbreviations like 'etc.', 'usw.', 'z.B.'. "
                                "Write out everything fully. Keep responses natural and conversational for speech."
                            )
                        else:
                            formatting_instructions = (
                                "You are responding via Discord text chat. "
                                "Use Discord markdown formatting, emojis, code blocks, and rich formatting to enhance readability. "
                                "Make your responses visually appealing and well-structured."
                            )

                        self.kernel.agent.variable_manager.set(
                            f'discord.formatting_instructions.{str(user.id)}',
                            formatting_instructions
                        )
                        print(f"🎤 [DEBUG] Output mode and formatting instructions injected into agent")

                    # Send transcription to kernel with enhanced metadata
                    print(f"🎤 [DEBUG] Creating kernel signal for user {user.id}")

                    # Build metadata with voice channel history if available
                    signal_metadata = {
                        "interface": "discord_voice",
                        "user_name": str(user),
                        "user_display_name": user.display_name,
                        "transcription": True,
                        "language": language,
                        "discord_context": discord_context,
                        "output_mode": output_mode,
                        "formatting_instructions": formatting_instructions,
                        "fast_response": True,  # Enable fast response mode for voice
                        "user_id": str(user.id)  # Ensure user_id is in metadata
                    }

                    # Add voice channel history if available (from wake word detection)
                    if voice_channel_history:
                        signal_metadata["voice_channel_history"] = voice_channel_history
                        print(f"📋 [HISTORY] Including voice channel history in signal metadata")

                    signal = KernelSignal(
                        type=SignalType.USER_INPUT,
                        id=str(user.id),
                        content=text,
                        metadata=signal_metadata
                    )
                    print(f"🎤 [DEBUG] Sending signal to kernel with fast_response=True...")
                    await self.kernel.process_signal(signal)
                    print(f"🎤 ✅ Voice input from {user.display_name}: {text}")
                else:
                    print(f"🎤 [DEBUG] Transcription text is empty, skipping")

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    print(f"🎤 [DEBUG] Cleaned up temp file: {temp_path}")

        except Exception as e:
            print(f"❌ [DEBUG] Error transcribing audio: {e}")
            import traceback
            traceback.print_exc()

    def cleanup(self):
        """Cleanup when sink is stopped"""
        print(f"🎤 [DEBUG] Cleaning up WhisperAudioSink")
        self.audio_buffer.clear()
        self.last_transcription.clear()
        self.speaking_state.clear()
        self.last_audio_time.clear()

    @voice_recv.AudioSink.listener() if VOICE_RECEIVE_SUPPORT else lambda f: f
    def on_voice_member_connect(self, member: discord.Member):
        """Handle member connect"""
        print(f"🎤 [DEBUG] {member.display_name} connected to voice")

    @voice_recv.AudioSink.listener() if VOICE_RECEIVE_SUPPORT else lambda f: f
    def on_voice_member_disconnect(self, member: discord.Member, ssrc: int):
        """Handle member disconnect"""
        user_id = str(member.id)
        print(f"🎤 [DEBUG] {member.display_name} disconnected from voice")

        if user_id in self.audio_buffer:
            del self.audio_buffer[user_id]
        if user_id in self.last_transcription:
            del self.last_transcription[user_id]
        if user_id in self.speaking_state:
            del self.speaking_state[user_id]
        if user_id in self.last_audio_time:
            del self.last_audio_time[user_id]

    @voice_recv.AudioSink.listener() if VOICE_RECEIVE_SUPPORT else lambda f: f
    def on_voice_member_speaking_start(self, member: discord.Member):
        """Handle speaking start (VAD)"""
        user_id = str(member.id)
        print(f"🎤 [DEBUG] {member.display_name} started speaking")
        self.speaking_state[user_id] = True

    @voice_recv.AudioSink.listener() if VOICE_RECEIVE_SUPPORT else lambda f: f
    def on_voice_member_speaking_stop(self, member: discord.Member):
        """Handle speaking stop (VAD)"""
        user_id = str(member.id)
        print(f"🔇 {member.display_name} stopped speaking")
        self.speaking_state[user_id] = False

        # Trigger final transcription if there's buffered audio
        if user_id in self.audio_buffer and self.audio_buffer[user_id]:
            print(f"🎤 [DEBUG] Triggering final transcription for {member.display_name}")

            # Schedule transcription in the event loop (listener is called from a different thread)
            try:
                from toolboxv2 import get_app
                get_app().run_bg_task_advanced(self._transcribe_buffer, user_id, member)
            except Exception as e:
                print(f"❌ [DEBUG] Error scheduling final transcription: {e}")


class DiscordProgressPrinter:
    """
    Discord-specific progress printer that updates a single master message
    instead of spamming multiple messages.

    Features:
    - Single master message that gets updated
    - Discord Embeds for structured display
    - Buttons for expandable sub-sections
    - Rate-limiting to avoid Discord API limits
    - Toggleable with !progress command
    """

    def __init__(self, channel: discord.TextChannel, user_id: str):
        self.channel = channel
        self.user_id = user_id
        self.master_message: Optional[discord.Message] = None
        self.enabled = False

        # State tracking (similar to terminal version)
        self.agent_name = "Agent"
        self.execution_phase = 'initializing'
        self.start_time = time.time()
        self.error_count = 0
        self.llm_calls = 0
        self.llm_cost = 0.0
        self.llm_tokens = 0
        self.tool_history = []
        self.active_nodes = set()
        self.current_task = None

        # Rate limiting
        self.last_update_time = 0
        self.update_interval = 2.0  # Update at most every 2 seconds
        self.pending_update = False

        # Expandable sections state
        self.show_tools = False
        self.show_llm = False
        self.show_system = False

    async def progress_callback(self, event: ProgressEvent):
        """Main entry point for progress events"""
        if not self.enabled:
            return

        # Process event
        await self._process_event(event)

        # Schedule update (rate-limited)
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            await self._update_display()
            self.last_update_time = current_time
            self.pending_update = False
        else:
            self.pending_update = True

    async def _process_event(self, event: ProgressEvent):
        """Process progress event and update state"""
        if event.agent_name:
            self.agent_name = event.agent_name

        # Track execution phase
        if event.event_type == 'execution_start':
            self.execution_phase = 'running'
            self.start_time = time.time()
        elif event.event_type == 'execution_complete':
            self.execution_phase = 'completed'
        elif event.event_type == 'error':
            self.error_count += 1

        # Track nodes
        if event.event_type == 'node_enter' and event.node_name:
            self.active_nodes.add(event.node_name)
        elif event.event_type == 'node_exit' and event.node_name:
            self.active_nodes.discard(event.node_name)

        # Track LLM calls
        if event.event_type == 'llm_call' and event.success:
            self.llm_calls += 1
            self.llm_cost += event.llm_cost or 0
            self.llm_tokens += event.llm_total_tokens or 0

        # Track tools
        if event.event_type == 'tool_call' and event.status in [NodeStatus.COMPLETED, NodeStatus.FAILED]:
            self.tool_history.append({
                'name': event.tool_name,
                'success': event.success,
                'duration': event.duration,
                'is_meta': event.is_meta_tool
            })
            if len(self.tool_history) > 5:
                self.tool_history.pop(0)

        # Track current task
        if event.event_type == 'task_start':
            self.current_task = event.metadata.get('task_description', 'Unknown task') if event.metadata else 'Unknown task'
        elif event.event_type == 'task_complete':
            self.current_task = None

    async def _update_display(self):
        """Update the Discord master message"""
        try:
            embed = self._create_embed()
            view = self._create_view()

            if self.master_message is None:
                # Create new master message
                self.master_message = await self.channel.send(
                    content=f"🤖 **Agent Progress** (User: <@{self.user_id}>)",
                    embed=embed,
                    view=view
                )
            else:
                # Update existing message
                await self.master_message.edit(embed=embed, view=view)

        except discord.HTTPException as e:
            # Handle rate limits gracefully
            if e.status == 429:  # Too Many Requests
                print(f"⚠️ Discord rate limit hit, skipping update")
            else:
                print(f"❌ Error updating progress message: {e}")
        except Exception as e:
            print(f"❌ Error updating progress display: {e}")

    def _create_embed(self) -> discord.Embed:
        """Create Discord embed with current state"""
        # Determine color based on phase
        color_map = {
            'initializing': discord.Color.blue(),
            'running': discord.Color.gold(),
            'completed': discord.Color.green(),
            'error': discord.Color.red()
        }
        color = color_map.get(self.execution_phase, discord.Color.blue())

        # Create embed
        embed = discord.Embed(
            title=f"🤖 {self.agent_name}",
            description=f"**Phase:** {self.execution_phase.upper()}",
            color=color,
            timestamp=datetime.utcnow()
        )

        # Runtime
        runtime = time.time() - self.start_time
        runtime_str = self._format_duration(runtime)
        embed.add_field(name="⏱️ Runtime", value=runtime_str, inline=True)

        # Errors
        error_emoji = "✅" if self.error_count == 0 else "⚠️"
        embed.add_field(name=f"{error_emoji} Errors", value=str(self.error_count), inline=True)

        # Active nodes
        active_count = len(self.active_nodes)
        embed.add_field(name="🔄 Active Nodes", value=str(active_count), inline=True)

        # Current task
        if self.current_task:
            task_preview = self.current_task[:100] + "..." if len(self.current_task) > 100 else self.current_task
            embed.add_field(name="📋 Current Task", value=task_preview, inline=False)

        # LLM Stats (always visible)
        llm_stats = f"**Calls:** {self.llm_calls}\n**Cost:** ${self.llm_cost:.4f}\n**Tokens:** {self.llm_tokens:,}"
        embed.add_field(name="🤖 LLM Statistics", value=llm_stats, inline=True)

        # Tool History (if expanded)
        if self.show_tools and self.tool_history:
            tool_text = ""
            for tool in self.tool_history[-3:]:  # Last 3 tools
                icon = "✅" if tool['success'] else "❌"
                duration = self._format_duration(tool['duration']) if tool['duration'] else "N/A"
                tool_text += f"{icon} `{tool['name']}` ({duration})\n"
            embed.add_field(name="🛠️ Recent Tools", value=tool_text or "No tools yet", inline=False)

        # System Flow (if expanded)
        if self.show_system and self.active_nodes:
            nodes_text = "\n".join([f"🔄 `{node[:30]}`" for node in list(self.active_nodes)[-3:]])
            embed.add_field(name="🔧 Active Nodes", value=nodes_text or "No active nodes", inline=False)

        embed.set_footer(text=f"Updates every {self.update_interval}s • Toggle sections with buttons")

        return embed

    def _create_view(self) -> discord.ui.View:
        """Create Discord view with buttons"""
        view = discord.ui.View(timeout=None)

        # Toggle Tools button
        tools_button = discord.ui.Button(
            label="Tools" if not self.show_tools else "Hide Tools",
            style=discord.ButtonStyle.primary if self.show_tools else discord.ButtonStyle.secondary,
            custom_id=f"progress_tools_{self.user_id}"
        )
        tools_button.callback = self._toggle_tools
        view.add_item(tools_button)

        # Toggle System button
        system_button = discord.ui.Button(
            label="System" if not self.show_system else "Hide System",
            style=discord.ButtonStyle.primary if self.show_system else discord.ButtonStyle.secondary,
            custom_id=f"progress_system_{self.user_id}"
        )
        system_button.callback = self._toggle_system
        view.add_item(system_button)

        # Stop button
        stop_button = discord.ui.Button(
            label="Stop Updates",
            style=discord.ButtonStyle.danger,
            custom_id=f"progress_stop_{self.user_id}"
        )
        stop_button.callback = self._stop_updates
        view.add_item(stop_button)

        return view

    async def _toggle_tools(self, interaction: discord.Interaction):
        """Toggle tools section"""
        self.show_tools = not self.show_tools
        await interaction.response.defer()
        await self._update_display()

    async def _toggle_system(self, interaction: discord.Interaction):
        """Toggle system section"""
        self.show_system = not self.show_system
        await interaction.response.defer()
        await self._update_display()

    async def _stop_updates(self, interaction: discord.Interaction):
        """Stop progress updates"""
        self.enabled = False
        await interaction.response.send_message("✅ Progress updates stopped", ephemeral=True)
        if self.master_message:
            await self.master_message.edit(view=None)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds is None:
            return "N/A"
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        minutes, seconds = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {seconds}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes}m"

    async def enable(self):
        """Enable progress updates"""
        self.enabled = True
        self.start_time = time.time()
        await self._update_display()

    async def disable(self):
        """Disable progress updates"""
        self.enabled = False
        if self.master_message:
            await self.master_message.edit(view=None)

    async def finalize(self):
        """Finalize progress display (called when execution completes)"""
        if self.pending_update:
            await self._update_display()
        if self.master_message:
            # Remove buttons when done
            await self.master_message.edit(view=None)


class DiscordOutputRouter(IOutputRouter):
    """Discord-specific output router with embed, media, voice, and TTS support"""

    def __init__(self, bot: commands.Bot, groq_client: 'Groq' = None, elevenlabs_client: 'ElevenLabs' = None, piper_path: str = None, piper_model: str = None):
        self.bot = bot
        self.active_channels: Dict[int, discord.TextChannel] = {}
        self.user_channels: Dict[str, discord.TextChannel] = {}  # user_id -> channel object
        self.voice_clients: Dict[int, discord.VoiceClient] = {}  # guild_id -> voice client
        self.audio_sinks: Dict[int, WhisperAudioSink] = {}  # guild_id -> audio sink
        self.groq_client = groq_client
        self.elevenlabs_client = elevenlabs_client
        self.piper_path = piper_path
        self.piper_model = piper_model  # Path to .onnx model file
        self.tts_enabled: Dict[int, bool] = {}  # guild_id -> tts enabled
        self.tts_mode: Dict[int, str] = {}  # guild_id -> "elevenlabs" or "piper"

    def _split_message(self, content: str, max_length: int = 1900) -> List[str]:
        """
        Split a long message into chunks that fit Discord's limits.
        Uses smart splitting at sentence/paragraph boundaries.

        Args:
            content: The message to split
            max_length: Maximum length per chunk (default 1900 to leave room for formatting)

        Returns:
            List of message chunks
        """
        if len(content) <= max_length:
            return [content]

        chunks = []
        current_chunk = ""

        # Try to split at paragraph boundaries first
        paragraphs = content.split('\n\n')

        for para in paragraphs:
            # If paragraph itself is too long, split at sentence boundaries
            if len(para) > max_length:
                sentences = para.replace('. ', '.|').replace('! ', '!|').replace('? ', '?|').split('|')

                for sentence in sentences:
                    # If sentence itself is too long, split at word boundaries
                    if len(sentence) > max_length:
                        words = sentence.split(' ')
                        for word in words:
                            if len(current_chunk) + len(word) + 1 > max_length:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = word + ' '
                            else:
                                current_chunk += word + ' '
                    else:
                        if len(current_chunk) + len(sentence) + 1 > max_length:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + ' '
                        else:
                            current_chunk += sentence + ' '
            else:
                if len(current_chunk) + len(para) + 2 > max_length:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + '\n\n'
                else:
                    current_chunk += para + '\n\n'

        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _create_embed(
        self,
        content: str,
        title: str = None,
        color: discord.Color = discord.Color.blue(),
        fields: List[dict] = None
    ) -> discord.Embed:
        """Create a Discord embed"""
        # Discord embed description limit is 4096 characters
        if len(content) > 4096:
            content = content[:4093] + "..."

        embed = discord.Embed(
            title=title,
            description=content,
            color=color,
            timestamp=datetime.now()
        )

        if fields:
            for field in fields:
                embed.add_field(
                    name=field.get("name", "Field"),
                    value=field.get("value", ""),
                    inline=field.get("inline", False)
                )

        embed.set_footer(text="ProA Kernel")
        return embed

    async def send_response(self, user_id: str, content: str, role: str = "assistant", metadata: dict = None):
        """Send agent response to Discord user (with optional TTS)"""
        try:
            channel = self.user_channels.get(user_id)
            if not channel:
                print(f"⚠️ No channel found for user {user_id}")
                return

            # Fix emoji and umlaut encoding issues
            import codecs
            try:
                # First, try to fix UTF-8 encoding issues (e.g., "fÃ¼r" -> "für")
                # This happens when UTF-8 bytes are incorrectly interpreted as Latin-1
                if any(char in content for char in ['Ã', 'â', 'Â']):
                    # Encode as Latin-1 and decode as UTF-8
                    content = content.encode('latin-1').decode('utf-8')
            except Exception as e:
                # If UTF-8 fix fails, try unicode escape sequences
                try:
                    # Decode unicode escape sequences like \u2764 to actual emojis
                    if '\\u' in content:
                        content = codecs.decode(content, 'unicode_escape')
                except Exception as e2:
                    # If all decoding fails, use original content
                    print(f"⚠️ Could not decode text: {e}, {e2}")

            # Check if TTS is enabled and bot is in voice channel with user
            guild_id = channel.guild.id if channel.guild else None
            tts_enabled = guild_id and guild_id in self.tts_enabled and self.tts_enabled[guild_id]
            in_voice = guild_id and guild_id in self.voice_clients and self.voice_clients[guild_id].is_connected()

            print(f"🔊 [DEBUG] Response mode - TTS: {tts_enabled}, In Voice: {in_voice}")

            if tts_enabled and in_voice:
                # TTS Mode: Only voice output, no text message
                print(f"🔊 [DEBUG] TTS Mode: Sending voice response only")
                await self._speak_text(guild_id, content)
            else:
                # Text Mode: Send text message (no TTS)
                print(f"💬 [DEBUG] Text Mode: Sending text response")
                use_embed = metadata and metadata.get("use_embed", True)

                if use_embed:
                    # Embed description limit is 4096, but we use _create_embed which handles truncation
                    embed = self._create_embed(
                        content=content,
                        title=metadata.get("title") if metadata else None,
                        color=discord.Color.green()
                    )
                    await channel.send(embed=embed)
                else:
                    # Plain text mode - split if too long (2000 char limit)
                    if len(content) > 2000:
                        print(f"💬 [DEBUG] Message too long ({len(content)} chars), splitting into chunks")
                        chunks = self._split_message(content, max_length=1900)

                        for i, chunk in enumerate(chunks, 1):
                            if i == 1:
                                # First message
                                await channel.send(chunk)
                            else:
                                # Subsequent messages with continuation indicator
                                await channel.send(f"*...continued ({i}/{len(chunks)})*\n\n{chunk}")

                            # Small delay between messages to avoid rate limiting
                            if i < len(chunks):
                                await asyncio.sleep(0.5)

                        print(f"💬 [DEBUG] Sent message in {len(chunks)} chunks")
                    else:
                        await channel.send(content)

        except Exception as e:
            print(f"❌ Error sending Discord response to user {user_id}: {e}")

    async def send_notification(self, user_id: str, content: str, priority: int = 5, metadata: dict = None):
        """Send notification to Discord user"""
        try:
            channel = self.user_channels.get(user_id)
            if not channel:
                print(f"⚠️ No channel found for user {user_id}")
                return

            # Color based on priority
            color = discord.Color.red() if priority >= 7 else discord.Color.orange()

            embed = self._create_embed(
                content=content,
                title="🔔 Notification",
                color=color
            )
            await channel.send(embed=embed)

        except Exception as e:
            print(f"❌ Error sending Discord notification to user {user_id}: {e}")

    async def send_error(self, user_id: str, error: str, metadata: dict = None):
        """Send error message to Discord user"""
        try:
            channel = self.user_channels.get(user_id)
            if not channel:
                print(f"⚠️ No channel found for user {user_id}")
                return

            embed = self._create_embed(
                content=error,
                title="❌ Error",
                color=discord.Color.red()
            )
            await channel.send(embed=embed)

        except Exception as e:
            print(f"❌ Error sending Discord error to user {user_id}: {e}")

    async def _speak_text(self, guild_id: int, text: str):
        """Speak text in voice channel using TTS"""
        if guild_id not in self.voice_clients:
            return

        voice_client = self.voice_clients[guild_id]
        if not voice_client or not voice_client.is_connected():
            return

        # Don't interrupt current playback
        if voice_client.is_playing():
            return

        try:
            tts_mode = self.tts_mode.get(guild_id, "piper")

            if tts_mode == "elevenlabs" and self.elevenlabs_client and ELEVENLABS_SUPPORT:
                await self._speak_elevenlabs(voice_client, text)
            elif tts_mode == "piper" and self.piper_path:
                await self._speak_piper(voice_client, text)
            else:
                print(f"⚠️ TTS mode '{tts_mode}' not available")
        except Exception as e:
            print(f"❌ Error speaking text: {e}")

    async def _speak_elevenlabs(self, voice_client: discord.VoiceClient, text: str):
        """Speak using ElevenLabs TTS"""
        try:
            # Generate audio stream
            audio_stream = self.elevenlabs_client.text_to_speech.stream(
                voice_id="21m00Tcm4TlvDq8ikWAM",  # Default voice (Rachel)
                text=text,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128"
            )

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                for chunk in audio_stream:
                    temp_file.write(chunk)
                temp_path = temp_file.name

            # Play audio
            audio_source = discord.FFmpegPCMAudio(temp_path)
            voice_client.play(audio_source, after=lambda e: os.unlink(temp_path) if e is None else print(f"Error: {e}"))

        except Exception as e:
            print(f"❌ ElevenLabs TTS error: {e}")

    async def _speak_piper(self, voice_client: discord.VoiceClient, text: str):
        """Speak using Piper TTS (local)"""
        try:
            print(f"🔊 [DEBUG] Piper TTS: Starting synthesis for text: '{text[:50]}...'")

            # Create temporary output file
            output_path = tempfile.mktemp(suffix=".wav")
            print(f"🔊 [DEBUG] Piper TTS: Output file: {output_path}")

            # Build Piper command
            # Piper reads text from stdin and requires --model and --output_file
            cmd = [
                self.piper_path,
                "--model", self.piper_model,
                "--output_file", output_path
            ]

            print(f"🔊 [DEBUG] Piper TTS: Command: {' '.join(cmd)}")
            print(f"🔊 [DEBUG] Piper TTS: Model: {self.piper_model}")

            # Run Piper (reads from stdin)
            result = subprocess.run(
                cmd,
                input=text.encode('utf-8'),
                capture_output=True,
                check=False  # Don't raise exception, we'll check returncode
            )

            print(f"🔊 [DEBUG] Piper TTS: Return code: {result.returncode}")

            if result.returncode != 0:
                print(f"❌ [DEBUG] Piper TTS stderr: {result.stderr.decode('utf-8', errors='ignore')}")
                print(f"❌ [DEBUG] Piper TTS stdout: {result.stdout.decode('utf-8', errors='ignore')}")
                raise Exception(f"Piper failed with return code {result.returncode}")

            print(f"🔊 [DEBUG] Piper TTS: Audio file created successfully")

            # Check if file exists and has content
            if not os.path.exists(output_path):
                raise Exception(f"Output file not created: {output_path}")

            file_size = os.path.getsize(output_path)
            print(f"🔊 [DEBUG] Piper TTS: Audio file size: {file_size} bytes")

            if file_size == 0:
                raise Exception("Output file is empty")

            # Play audio
            print(f"🔊 [DEBUG] Piper TTS: Starting playback...")
            audio_source = discord.FFmpegPCMAudio(output_path)

            def cleanup(error):
                try:
                    os.unlink(output_path)
                    print(f"🔊 [DEBUG] Piper TTS: Cleaned up output file")
                except Exception as e:
                    print(f"⚠️ [DEBUG] Piper TTS: Cleanup error: {e}")
                if error:
                    print(f"❌ [DEBUG] Piper TTS: Playback error: {error}")
                else:
                    print(f"🔊 [DEBUG] Piper TTS: Playback completed successfully")

            voice_client.play(audio_source, after=cleanup)
            print(f"🔊 [DEBUG] Piper TTS: Audio source playing")

        except Exception as e:
            print(f"❌ [DEBUG] Piper TTS error: {e}")
            import traceback
            traceback.print_exc()

    async def send_media(
        self,
        user_id: str,
        file_path: str = None,
        url: str = None,
        caption: str = None
    ) -> Dict[str, Any]:
        """Send media to Discord user"""
        try:
            channel = self.user_channels.get(user_id)
            if not channel:
                print(f"⚠️ No channel found for user {user_id}")
                return {
                    "success": False,
                    "error": "No channel found for user"
                }

            if file_path:
                # Send file attachment
                file = discord.File(file_path)
                message = await channel.send(content=caption, file=file)
                return {
                    "success": True,
                    "message_id": message.id,
                    "type": "file",
                    "file_path": file_path,
                    "caption": caption
                }
            elif url:
                # Send embed with image
                embed = discord.Embed(description=caption or "")
                embed.set_image(url=url)
                message = await channel.send(embed=embed)
                return {
                    "success": True,
                    "message_id": message.id,
                    "type": "url",
                    "url": url,
                    "caption": caption
                }
            else:
                return {
                    "success": False,
                    "error": "Either file_path or url must be provided"
                }

        except Exception as e:
            print(f"❌ Error sending Discord media to user {user_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class VariableExplorerView(View):
    """Interactive UI for exploring variable scopes with tree navigation"""

    def __init__(self, var_manager, user_id: str, current_path: str = "", timeout: int = 300):
        super().__init__(timeout=timeout)
        self.var_manager = var_manager
        self.user_id = user_id
        self.current_path = current_path
        self.page = 0
        self.items_per_page = 10

        self._build_ui()

    def _build_ui(self):
        """Build the UI components based on current state"""
        self.clear_items()

        # Add scope selector if at root
        if not self.current_path:
            self._add_scope_selector()

        # Add navigation buttons
        if self.current_path:
            self._add_back_button()

        self._add_refresh_button()
        self._add_export_button()

        # Add pagination if needed
        current_data = self._get_current_data()

        # Check if it's a top-level scope
        if self.current_path in self.var_manager.scopes:
            current_data = self.var_manager.scopes[self.current_path]

        if isinstance(current_data, dict) and len(current_data) > self.items_per_page:
            self._add_pagination_buttons()

    def _add_scope_selector(self):
        """Add dropdown for scope selection"""
        scopes = list(self.var_manager.scopes.keys())

        if len(scopes) > 0:
            options = [
                discord.SelectOption(
                    label=scope,
                    value=scope,
                    description=f"Explore {scope} scope",
                    emoji="📁"
                )
                for scope in scopes[:25]  # Discord limit
            ]

            select = Select(
                placeholder="Select a scope to explore...",
                options=options,
                custom_id=f"scope_select_{self.user_id}"
            )
            select.callback = self._scope_select_callback
            self.add_item(select)

    def _add_back_button(self):
        """Add back navigation button"""
        button = Button(
            label="⬅️ Back",
            style=discord.ButtonStyle.secondary,
            custom_id=f"back_{self.user_id}"
        )
        button.callback = self._back_callback
        self.add_item(button)

    def _add_refresh_button(self):
        """Add refresh button"""
        button = Button(
            label="🔄 Refresh",
            style=discord.ButtonStyle.primary,
            custom_id=f"refresh_{self.user_id}"
        )
        button.callback = self._refresh_callback
        self.add_item(button)

    def _add_export_button(self):
        """Add export button"""
        button = Button(
            label="💾 Export",
            style=discord.ButtonStyle.success,
            custom_id=f"export_{self.user_id}"
        )
        button.callback = self._export_callback
        self.add_item(button)

    def _add_pagination_buttons(self):
        """Add pagination controls"""
        # Get actual data for pagination calculation
        if self.current_path in self.var_manager.scopes:
            current_data = self.var_manager.scopes[self.current_path]
        else:
            current_data = self._get_current_data()

        total_pages = (len(current_data) + self.items_per_page - 1) // self.items_per_page

        # Previous page button
        prev_button = Button(
            label="◀️ Previous",
            style=discord.ButtonStyle.secondary,
            custom_id=f"prev_{self.user_id}",
            disabled=self.page == 0
        )
        prev_button.callback = self._prev_page_callback
        self.add_item(prev_button)

        # Page indicator button (disabled, just for display)
        page_button = Button(
            label=f"Page {self.page + 1}/{total_pages}",
            style=discord.ButtonStyle.secondary,
            disabled=True
        )
        self.add_item(page_button)

        # Next page button
        next_button = Button(
            label="Next ▶️",
            style=discord.ButtonStyle.secondary,
            custom_id=f"next_{self.user_id}",
            disabled=self.page >= total_pages - 1
        )
        next_button.callback = self._next_page_callback
        self.add_item(next_button)

    def _get_current_data(self) -> Any:
        """Get data for current path"""
        if not self.current_path:
            return self.var_manager.scopes

        # For top-level scopes, get directly from scopes dict
        if self.current_path in self.var_manager.scopes:
            return self.var_manager.scopes[self.current_path]

        # For nested paths, use the get method
        data = self.var_manager.get(self.current_path)

        # Handle None returns
        if data is None:
            data = {}

        return data

    def _get_child_items(self) -> List[Tuple[str, Any]]:
        """Get child items for current path with pagination"""
        # Get the actual data
        if self.current_path in self.var_manager.scopes:
            current_data = self.var_manager.scopes[self.current_path]
        else:
            current_data = self._get_current_data()

        if isinstance(current_data, dict):
            items = list(current_data.items())
        elif isinstance(current_data, list):
            items = [(str(i), item) for i, item in enumerate(current_data)]
        else:
            return []

        # Apply pagination
        start_idx = self.page * self.items_per_page
        end_idx = start_idx + self.items_per_page
        return items[start_idx:end_idx]

    def _format_value_preview(self, value: Any, max_length: int = 100) -> str:
        """Format a value preview with type information"""
        value_type = type(value).__name__

        if isinstance(value, dict):
            keys = list(value.keys())[:3]
            preview = f"Dict with {len(value)} keys: {keys}"
            if len(value) > 3:
                preview += "..."
        elif isinstance(value, list):
            preview = f"List with {len(value)} items"
            if len(value) > 0:
                preview += f" (first: {str(value[0])[:30]}...)"
        elif isinstance(value, str):
            preview = value[:max_length]
            if len(value) > max_length:
                preview += "..."
        else:
            preview = str(value)[:max_length]

        return f"[{value_type}] {preview}"

    def _calculate_scope_sizes(self) -> Dict[str, Dict[str, int]]:
        """Calculate size statistics for each scope"""
        sizes = {}

        for scope_name, scope_data in self.var_manager.scopes.items():
            try:
                # Calculate approximate size in characters
                json_str = json.dumps(scope_data, default=str)
                sizes[scope_name] = {
                    'chars': len(json_str),
                    'keys': len(scope_data) if isinstance(scope_data, dict) else 1,
                    'kb': len(json_str) / 1024
                }
            except:
                sizes[scope_name] = {'chars': 0, 'keys': 0, 'kb': 0}

        return sizes

    def create_embed(self) -> discord.Embed:
        """Create the main embed for current view"""
        if not self.current_path:
            return self._create_root_embed()
        else:
            return self._create_path_embed()

    def _create_root_embed(self) -> discord.Embed:
        """Create embed for root scope overview"""
        embed = discord.Embed(
            title="🗂️ Variable Explorer - Root",
            description="Select a scope to explore its contents",
            color=discord.Color.blue(),
            timestamp=datetime.now(UTC)
        )

        # Calculate scope sizes
        sizes = self._calculate_scope_sizes()

        # Add scope overview
        scope_info = []
        for scope_name, scope_data in self.var_manager.scopes.items():
            size_info = sizes.get(scope_name, {})

            # Count items
            if isinstance(scope_data, dict):
                item_count = len(scope_data)
                icon = "📁"
            elif isinstance(scope_data, list):
                item_count = len(scope_data)
                icon = "📋"
            else:
                item_count = 1
                icon = "📄"

            scope_info.append(
                f"{icon} **{scope_name}**\n"
                f"  └─ {item_count} items | {size_info.get('kb', 0):.2f} KB"
            )

        # Split into multiple fields if needed
        field_text = "\n\n".join(scope_info)
        if len(field_text) > 1024:
            # Split into multiple fields
            chunks = self._split_text(field_text, 1024)
            for i, chunk in enumerate(chunks[:25]):  # Max 25 fields
                embed.add_field(
                    name=f"Scopes (Part {i + 1})" if i > 0 else "Available Scopes",
                    value=chunk,
                    inline=False
                )
        else:
            embed.add_field(
                name="Available Scopes",
                value=field_text or "No scopes available",
                inline=False
            )

        # Add total stats
        total_size = sum(s.get('kb', 0) for s in sizes.values())
        total_items = sum(s.get('keys', 0) for s in sizes.values())

        embed.set_footer(text=f"Total: {total_items} items | {total_size:.2f} KB")

        return embed

    def _create_path_embed(self) -> discord.Embed:
        """Create embed for specific path view"""
        # Get actual data
        if self.current_path in self.var_manager.scopes:
            current_data = self.var_manager.scopes[self.current_path]
        else:
            current_data = self._get_current_data()

        # Determine breadcrumb trail
        path_parts = self.current_path.split('.')
        breadcrumb = ' > '.join(path_parts)

        embed = discord.Embed(
            title=f"📂 {path_parts[-1]}",
            description=f"Path: `{breadcrumb}`",
            color=discord.Color.green(),
            timestamp=datetime.now(UTC)
        )

        # Handle different data types
        if isinstance(current_data, dict):
            embed = self._add_dict_fields(embed, current_data)
        elif isinstance(current_data, list):
            embed = self._add_list_fields(embed, current_data)
        else:
            # Leaf value
            embed = self._add_value_field(embed, current_data)

        return embed

    def _add_dict_fields(self, embed: discord.Embed, data: dict) -> discord.Embed:
        """Add dictionary contents to embed"""
        # Ensure we have the actual data
        if not data or len(data) == 0:
            if self.current_path in self.var_manager.scopes:
                data = self.var_manager.scopes[self.current_path]

            if not data or len(data) == 0:
                embed.add_field(name="Empty", value="This dictionary is empty", inline=False)
                return embed

        child_items = self._get_child_items()

        if not child_items:
            embed.add_field(name="Empty", value="This dictionary is empty", inline=False)
            return embed

        # Group by type for better organization
        grouped = defaultdict(list)
        for key, value in child_items:
            value_type = 'dict' if isinstance(value, dict) else \
                'list' if isinstance(value, list) else 'value'
            grouped[value_type].append((key, value))

        # Add fields for each group
        for type_name in ['dict', 'list', 'value']:
            if type_name not in grouped:
                continue

            items = grouped[type_name]
            icon = "📁" if type_name == 'dict' else "📋" if type_name == 'list' else "📄"

            field_text = ""
            for key, value in items:
                preview = self._format_value_preview(value, 80)
                field_text += f"{icon} **{key}**\n  └─ {preview}\n\n"

            # Handle field length limit
            if len(field_text) > 1024:
                chunks = self._split_text(field_text, 1024)
                for i, chunk in enumerate(chunks[:3]):  # Max 3 chunks per type
                    field_name = f"{type_name.title()}s (Part {i + 1})" if i > 0 else f"{type_name.title()}s"
                    embed.add_field(name=field_name, value=chunk, inline=False)
            else:
                embed.add_field(
                    name=f"{type_name.title()}s",
                    value=field_text,
                    inline=False
                )

        # Add pagination info
        total_items = len(data)
        start_idx = self.page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, total_items)

        embed.set_footer(text=f"Showing {start_idx + 1}-{end_idx} of {total_items} items")

        return embed

    def _add_list_fields(self, embed: discord.Embed, data: list) -> discord.Embed:
        """Add list contents to embed"""
        child_items = self._get_child_items()

        if not child_items:
            embed.add_field(name="Empty", value="This list is empty", inline=False)
            return embed

        field_text = ""
        for idx, value in child_items:
            preview = self._format_value_preview(value, 80)
            field_text += f"**[{idx}]**\n  └─ {preview}\n\n"

        # Handle field length limit
        if len(field_text) > 1024:
            chunks = self._split_text(field_text, 1024)
            for i, chunk in enumerate(chunks[:25]):
                field_name = f"Items (Part {i + 1})" if i > 0 else "Items"
                embed.add_field(name=field_name, value=chunk, inline=False)
        else:
            embed.add_field(name="Items", value=field_text, inline=False)

        # Add pagination info
        total_items = len(data)
        start_idx = self.page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, total_items)

        embed.set_footer(text=f"Showing {start_idx + 1}-{end_idx} of {total_items} items")

        return embed

    def _add_value_field(self, embed: discord.Embed, value: Any) -> discord.Embed:
        """Add leaf value to embed"""
        value_type = type(value).__name__

        # Format value based on type
        if isinstance(value, str):
            formatted = value
        else:
            try:
                formatted = json.dumps(value, indent=2, default=str)
            except:
                formatted = str(value)

        # Split if too long
        if len(formatted) > 1024:
            chunks = self._split_text(formatted, 1024)
            for i, chunk in enumerate(chunks[:25]):
                field_name = f"Value (Part {i + 1})" if i > 0 else f"Value [{value_type}]"
                embed.add_field(name=field_name, value=f"```\n{chunk}\n```", inline=False)
        else:
            embed.add_field(
                name=f"Value [{value_type}]",
                value=f"```\n{formatted}\n```",
                inline=False
            )

        return embed

    @staticmethod
    def _split_text(text: str, max_length: int) -> List[str]:
        """Split text into chunks of max_length, preserving line breaks"""
        chunks = []
        current_chunk = ""

        for line in text.split('\n'):
            if len(current_chunk) + len(line) + 1 > max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += ('\n' if current_chunk else '') + line

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    # Callback methods
    async def _scope_select_callback(self, interaction: discord.Interaction):
        """Handle scope selection"""
        if str(interaction.user.id) != self.user_id:
            await interaction.response.send_message("❌ This is not your explorer!", ephemeral=True)
            return

        scope_name = interaction.data['values'][0]
        self.current_path = scope_name
        self.page = 0
        self._build_ui()

        embed = self.create_embed()
        await interaction.response.edit_message(embed=embed, view=self)

    async def _back_callback(self, interaction: discord.Interaction):
        """Handle back navigation"""
        if str(interaction.user.id) != self.user_id:
            await interaction.response.send_message("❌ This is not your explorer!", ephemeral=True)
            return

        path_parts = self.current_path.split('.')
        if len(path_parts) > 1:
            self.current_path = '.'.join(path_parts[:-1])
        else:
            self.current_path = ""

        self.page = 0
        self._build_ui()

        embed = self.create_embed()
        await interaction.response.edit_message(embed=embed, view=self)

    async def _refresh_callback(self, interaction: discord.Interaction):
        """Handle refresh"""
        if str(interaction.user.id) != self.user_id:
            await interaction.response.send_message("❌ This is not your explorer!", ephemeral=True)
            return

        self._build_ui()
        embed = self.create_embed()
        await interaction.response.edit_message(embed=embed, view=self)

    async def _export_callback(self, interaction: discord.Interaction):
        """Handle export to JSON file"""
        if str(interaction.user.id) != self.user_id:
            await interaction.response.send_message("❌ This is not your explorer!", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)

        try:
            # Get actual data
            if self.current_path in self.var_manager.scopes:
                current_data = self.var_manager.scopes[self.current_path]
            else:
                current_data = self._get_current_data()

            json_str = json.dumps(current_data, indent=2, default=str)

            # Create file
            import io
            file_content = io.BytesIO(json_str.encode('utf-8'))
            file_name = f"variables_{self.current_path or 'root'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            file = discord.File(file_content, filename=file_name)

            await interaction.followup.send(
                f"📥 Exported: `{self.current_path or 'root'}`",
                file=file,
                ephemeral=True
            )
        except Exception as e:
            await interaction.followup.send(f"❌ Export failed: {e}", ephemeral=True)

    async def _prev_page_callback(self, interaction: discord.Interaction):
        """Handle previous page"""
        if str(interaction.user.id) != self.user_id:
            await interaction.response.send_message("❌ This is not your explorer!", ephemeral=True)
            return

        if self.page > 0:
            self.page -= 1
            self._build_ui()
            embed = self.create_embed()
            await interaction.response.edit_message(embed=embed, view=self)
        else:
            await interaction.response.defer()

    async def _next_page_callback(self, interaction: discord.Interaction):
        """Handle next page"""
        if str(interaction.user.id) != self.user_id:
            await interaction.response.send_message("❌ This is not your explorer!", ephemeral=True)
            return

        # Get actual data for page count
        if self.current_path in self.var_manager.scopes:
            current_data = self.var_manager.scopes[self.current_path]
        else:
            current_data = self._get_current_data()

        total_pages = (len(current_data) + self.items_per_page - 1) // self.items_per_page

        if self.page < total_pages - 1:
            self.page += 1
            self._build_ui()
            embed = self.create_embed()
            await interaction.response.edit_message(embed=embed, view=self)
        else:
            await interaction.response.defer()


class ContextPaginationView(discord.ui.View):
    """Paginated view for context data with navigation buttons"""

    def __init__(self, user_id: str, data_type: str, items: list, formatter_func, timeout: int = 300):
        super().__init__(timeout=timeout)
        self.user_id = user_id
        self.data_type = data_type
        self.items = items
        self.formatter_func = formatter_func
        self.current_page = 0
        self.items_per_page = 5
        self.total_pages = (len(items) + self.items_per_page - 1) // self.items_per_page if items else 1

        self._build_buttons()

    def _build_buttons(self):
        """Build navigation buttons"""
        self.clear_items()

        # Previous page button
        prev_button = discord.ui.Button(
            label="◀️ Previous",
            style=discord.ButtonStyle.secondary,
            custom_id=f"prev_{self.user_id}",
            disabled=self.current_page == 0
        )
        prev_button.callback = self._prev_callback
        self.add_item(prev_button)

        # Page indicator
        page_button = discord.ui.Button(
            label=f"Page {self.current_page + 1}/{self.total_pages}",
            style=discord.ButtonStyle.secondary,
            disabled=True
        )
        self.add_item(page_button)

        # Next page button
        next_button = discord.ui.Button(
            label="Next ▶️",
            style=discord.ButtonStyle.secondary,
            custom_id=f"next_{self.user_id}",
            disabled=self.current_page >= self.total_pages - 1
        )
        next_button.callback = self._next_callback
        self.add_item(next_button)

        # Jump to first page button
        if self.total_pages > 2:
            first_button = discord.ui.Button(
                label="⏮️ First",
                style=discord.ButtonStyle.primary,
                custom_id=f"first_{self.user_id}",
                disabled=self.current_page == 0
            )
            first_button.callback = self._first_callback
            self.add_item(first_button)

        # Jump to last page button
        if self.total_pages > 2:
            last_button = discord.ui.Button(
                label="Last ⏭️",
                style=discord.ButtonStyle.primary,
                custom_id=f"last_{self.user_id}",
                disabled=self.current_page >= self.total_pages - 1
            )
            last_button.callback = self._last_callback
            self.add_item(last_button)

    def get_current_page_items(self) -> list:
        """Get items for current page"""
        start_idx = self.current_page * self.items_per_page
        end_idx = start_idx + self.items_per_page
        return self.items[start_idx:end_idx]

    def create_embed(self) -> discord.Embed:
        """Create embed for current page"""
        page_items = self.get_current_page_items()

        # Create base embed
        embed = discord.Embed(
            title=self._get_title(),
            description=self._get_description(),
            color=self._get_color(),
            timestamp=datetime.now(UTC)
        )

        # Add items using formatter function
        for item in page_items:
            field_data = self.formatter_func(item)
            if field_data:
                embed.add_field(
                    name=field_data.get('name', 'Item'),
                    value=field_data.get('value', 'No data'),
                    inline=field_data.get('inline', False)
                )

        # Add footer with page info
        start_idx = self.current_page * self.items_per_page + 1
        end_idx = min(start_idx + len(page_items) - 1, len(self.items))
        embed.set_footer(
            text=f"Showing {start_idx}-{end_idx} of {len(self.items)} items • Page {self.current_page + 1}/{self.total_pages}")

        return embed

    def _get_title(self) -> str:
        """Get embed title based on data type"""
        titles = {
            'memories': '📝 Your Memories',
            'learning': '📚 Learning Records',
            'history': '📜 Conversation History',
            'tasks': '📅 Scheduled Tasks'
        }
        return titles.get(self.data_type, '📋 Data')

    def _get_description(self) -> str:
        """Get embed description"""
        if not self.items:
            descriptions = {
                'memories': 'No memories stored yet. I\'ll learn about you as we interact!',
                'learning': 'No learning records yet. I\'ll learn from our interactions!',
                'history': 'No history records yet. Start chatting to build history!',
                'tasks': 'No scheduled tasks. Use kernel tools to schedule tasks!'
            }
            return descriptions.get(self.data_type, 'No data available')

        return f"Total {self.data_type}: {len(self.items)}"

    def _get_color(self) -> discord.Color:
        """Get embed color based on data type"""
        colors = {
            'memories': discord.Color.green(),
            'learning': discord.Color.purple(),
            'history': discord.Color.blue(),
            'tasks': discord.Color.gold()
        }
        return colors.get(self.data_type, discord.Color.blue())

    async def _check_permission(self, interaction: discord.Interaction) -> bool:
        """Check if user has permission to interact"""
        if str(interaction.user.id) != self.user_id:
            await interaction.response.send_message("❌ This is not your context!", ephemeral=True)
            return False
        return True

    async def _prev_callback(self, interaction: discord.Interaction):
        """Handle previous page"""
        if not await self._check_permission(interaction):
            return

        if self.current_page > 0:
            self.current_page -= 1
            self._build_buttons()
            embed = self.create_embed()
            await interaction.response.edit_message(embed=embed, view=self)
        else:
            await interaction.response.defer()

    async def _next_callback(self, interaction: discord.Interaction):
        """Handle next page"""
        if not await self._check_permission(interaction):
            return

        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self._build_buttons()
            embed = self.create_embed()
            await interaction.response.edit_message(embed=embed, view=self)
        else:
            await interaction.response.defer()

    async def _first_callback(self, interaction: discord.Interaction):
        """Handle jump to first page"""
        if not await self._check_permission(interaction):
            return

        if self.current_page != 0:
            self.current_page = 0
            self._build_buttons()
            embed = self.create_embed()
            await interaction.response.edit_message(embed=embed, view=self)
        else:
            await interaction.response.defer()

    async def _last_callback(self, interaction: discord.Interaction):
        """Handle jump to last page"""
        if not await self._check_permission(interaction):
            return

        if self.current_page != self.total_pages - 1:
            self.current_page = self.total_pages - 1
            self._build_buttons()
            embed = self.create_embed()
            await interaction.response.edit_message(embed=embed, view=self)
        else:
            await interaction.response.defer()


class DiscordKernel:
    """Discord-based ProA Kernel with auto-persistence and rich features"""

    def __init__(
        self,
        agent,
        app: App,
        bot_token: str,
        command_prefix: str = "!",
        instance_id: str = "default",
        auto_save_interval: int = 300
    ):
        """
        Initialize Discord Kernel

        Args:
            agent: FlowAgent instance
            app: ToolBoxV2 App instance
            bot_token: Discord bot token
            command_prefix: Command prefix for bot commands
            instance_id: Instance identifier
            auto_save_interval: Auto-save interval in seconds (default: 5 minutes)
        """
        if discord is None or commands is None:
            raise ImportError("discord.py not installed")

        self.agent = agent
        self.app = app
        self.instance_id = instance_id
        self.auto_save_interval = auto_save_interval
        self.running = False
        self.save_path = self._get_save_path()

        # Initialize Discord bot
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.guilds = True

        # Bot description for help command
        bot_description = (
            "🤖 **ToolBox Isaa Agent** - Your intelligent AI assistant\n\n"
            "I can help you with various tasks, answer questions, and interact via voice or text.\n"
            "Use commands to control my behavior and access advanced features."
        )

        self.bot = commands.Bot(
            command_prefix=command_prefix,
            intents=intents,
            description=bot_description,
            help_command=commands.DefaultHelpCommand(),  # Enable default help command
            strip_after_prefix=True
        )
        self.bot_token = bot_token

        # Admin whitelist - only these users can use admin commands (!vars, !reset, !exit)
        # Default: "Kinr3" and bot owner
        self.admin_whitelist = {"kinr3"}  # Lowercase for case-insensitive comparison
        print(f"🔒 [SECURITY] Admin whitelist initialized: {self.admin_whitelist}")

        # Initialize kernel with Discord output router
        config = KernelConfig(
            heartbeat_interval=30.0,
            idle_threshold=600.0,  # 10 minutes
            proactive_cooldown=120.0,  # 2 minutes
            max_proactive_per_hour=8
        )

        # Initialize Groq client if available
        groq_client = None
        if GROQ_SUPPORT:
            groq_api_key = os.getenv('GROQ_API_KEY')
            if groq_api_key:
                groq_client = Groq(api_key=groq_api_key)
                print("✓ Groq Whisper enabled for voice transcription")
            else:
                print("⚠️ GROQ_API_KEY not set. Voice transcription disabled.")

        # Initialize ElevenLabs client if available
        elevenlabs_client = None
        if ELEVENLABS_SUPPORT:
            elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
            if elevenlabs_api_key:
                elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)
                print("✓ ElevenLabs TTS enabled")
            else:
                print("⚠️ ELEVENLABS_API_KEY not set. ElevenLabs TTS disabled.")

        # Check for Piper TTS
        piper_path = os.getenv('PIPER_PATH', r'C:\Users\Markin\Workspace\piper_w\piper.exe')
        piper_model = os.getenv('PIPER_MODEL', r'C:\Users\Markin\Workspace\piper_w\models\de_DE-thorsten-high.onnx')

        global PIPER_SUPPORT
        if os.path.exists(piper_path):
            print(f"✓ Piper TTS enabled at {piper_path}")

            # Check if model exists
            if os.path.exists(piper_model):
                print(f"✓ Piper model found: {piper_model}")
                PIPER_SUPPORT = True
            else:
                print(f"⚠️ Piper model not found at {piper_model}")
                print(f"⚠️ Set PIPER_MODEL environment variable or place model at default location")
                print(f"⚠️ Available models should be in: C:\\Users\\Markin\\Workspace\\piper_w\\models\\")
                piper_path = None
                piper_model = None
                PIPER_SUPPORT = False
        else:
            print(f"⚠️ Piper not found at {piper_path}. Local TTS disabled.")
            piper_path = None
            piper_model = None
            PIPER_SUPPORT = False

        # Print support status
        print("\n" + "=" * 60)
        print("🎤 VOICE SYSTEM SUPPORT STATUS")
        print("=" * 60)
        print(f"VOICE_SUPPORT:         {'✅' if VOICE_SUPPORT else '❌'}")
        print(f"VOICE_RECEIVE_SUPPORT: {'✅' if VOICE_RECEIVE_SUPPORT else '❌'}")
        print(f"GROQ_SUPPORT:          {'✅' if GROQ_SUPPORT else '❌'}")
        print(f"ELEVENLABS_SUPPORT:    {'✅' if ELEVENLABS_SUPPORT else '❌'}")
        print(f"PIPER_SUPPORT:         {'✅' if PIPER_SUPPORT else '❌'}")
        print("=" * 60 + "\n")
        self.output_router = DiscordOutputRouter(
            self.bot,
            groq_client=groq_client,
            elevenlabs_client=elevenlabs_client,
            piper_path=piper_path,
            piper_model=piper_model
        )
        self.kernel = Kernel(
            agent=agent,
            config=config,
            output_router=self.output_router
        )

        # Initialize Discord-specific tools
        self.discord_tools = DiscordKernelTools(
            bot=self.bot,
            kernel=self.kernel,
            output_router=self.output_router
        )

        # Progress printers per user
        self.progress_printers: Dict[str, DiscordProgressPrinter] = {}

        # Setup bot events
        self._setup_bot_events()
        self._setup_bot_commands()

        # Print registered commands
        print(f"\n🎮 Registered Discord Commands:")
        for cmd in self.bot.commands:
            print(f"   • !{cmd.name}")
        print()

        print(f"✓ Discord Kernel initialized (instance: {instance_id})")

    def _is_admin(self, ctx: commands.Context) -> bool:
        """Check if user is in admin whitelist or is bot owner"""
        # Check if user is bot owner
        if ctx.author.id == self.bot.owner_id:
            return True

        # Check if username is in whitelist (case-insensitive)
        username_lower = ctx.author.name.lower()
        if username_lower in self.admin_whitelist:
            return True

        # Check if user ID is in whitelist (for ID-based whitelist entries)
        user_id_str = str(ctx.author.id)
        if user_id_str in self.admin_whitelist:
            return True

        return False

    async def _check_admin_permission(self, ctx: commands.Context) -> bool:
        """Check admin permission and send error message if denied"""
        if not self._is_admin(ctx):
            embed = discord.Embed(
                title="🔒 Access Denied",
                description="This command is restricted to administrators only.",
                color=discord.Color.red()
            )
            embed.add_field(
                name="Your Access Level",
                value=f"User: {ctx.author.name}\nAdmin: ❌",
                inline=False
            )
            await ctx.send(embed=embed, ephemeral=True)
            print(f"🔒 [SECURITY] Admin command denied for user {ctx.author.name} (ID: {ctx.author.id})")
            return False
        return True

    def _get_save_path(self) -> Path:
        """Get save file path"""
        save_dir = Path(self.app.data_dir) / 'Agents' / 'kernel' / self.agent.amd.name / 'discord'
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / f"discord_kernel_{self.instance_id}.pkl"

    def _setup_bot_events(self):
        """Setup Discord bot events"""

        @self.bot.event
        async def on_ready():
            print(f"✓ Discord bot logged in as {self.bot.user}")

            # Set bot status
            await self.bot.change_presence(
                activity=discord.Activity(
                    type=discord.ActivityType.listening,
                    name="your messages | !help"
                )
            )

        @self.bot.event
        async def on_message(message: discord.Message):
            # Ignore bot messages
            if message.author.bot:
                return

            # Check if message is a command (starts with command prefix)
            ctx = await self.bot.get_context(message)
            if ctx.valid:
                # This is a valid command, process it and DON'T send to agent
                await self.bot.process_commands(message)
                return

            # Handle direct messages or mentions (only non-command messages)
            if isinstance(message.channel, discord.DMChannel) or self.bot.user in message.mentions:
                await self.handle_message(message)

        @self.bot.event
        async def on_message_edit(before: discord.Message, after: discord.Message):
            # Handle edited messages
            if not after.author.bot and after.content != before.content:
                signal = KernelSignal(
                    type=SignalType.SYSTEM_EVENT,
                    id=str(after.author.id),
                    content=f"Message edited: {before.content} -> {after.content}",
                    metadata={"event": "message_edit"}
                )
                await self.kernel.process_signal(signal)

        @self.bot.event
        async def on_reaction_add(reaction: discord.Reaction, user: discord.User):
            # Handle reactions
            if not user.bot:
                signal = KernelSignal(
                    type=SignalType.SYSTEM_EVENT,
                    id=str(user.id),
                    content=f"Reaction added: {reaction.emoji}",
                    metadata={"event": "reaction_add", "emoji": str(reaction.emoji)}
                )
                await self.kernel.process_signal(signal)

        @self.bot.event
        async def on_voice_state_update(member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
            # Track voice state changes
            if member.bot:
                return

            # User joined a voice channel
            if before.channel is None and after.channel is not None:
                signal = KernelSignal(
                    type=SignalType.SYSTEM_EVENT,
                    id=str(member.id),
                    content=f"{member.display_name} joined voice channel {after.channel.name}",
                    metadata={
                        "event": "voice_join",
                        "channel_id": after.channel.id,
                        "channel_name": after.channel.name
                    }
                )
                await self.kernel.process_signal(signal)

            # User left a voice channel
            elif before.channel is not None and after.channel is None:
                signal = KernelSignal(
                    type=SignalType.SYSTEM_EVENT,
                    id=str(member.id),
                    content=f"{member.display_name} left voice channel {before.channel.name}",
                    metadata={
                        "event": "voice_leave",
                        "channel_id": before.channel.id,
                        "channel_name": before.channel.name
                    }
                )
                await self.kernel.process_signal(signal)

            # User moved between voice channels
            elif before.channel != after.channel:
                signal = KernelSignal(
                    type=SignalType.SYSTEM_EVENT,
                    id=str(member.id),
                    content=f"{member.display_name} moved from {before.channel.name} to {after.channel.name}",
                    metadata={
                        "event": "voice_move",
                        "from_channel_id": before.channel.id,
                        "to_channel_id": after.channel.id
                    }
                )
                await self.kernel.process_signal(signal)

    def _setup_bot_commands(self):
        """Setup Discord bot commands"""

        @self.bot.command(name="status")
        async def status_command(ctx: commands.Context):
            """Show comprehensive kernel status"""
            status = self.kernel.to_dict()

            embed = discord.Embed(
                title="🤖 ProA Kernel Status",
                description=f"State: **{status['state']}** | Running: {'✅' if status['running'] else '❌'}",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )

            # Core Metrics
            embed.add_field(
                name="📊 Core Metrics",
                value=(
                    f"**Signals Processed:** {status['metrics']['signals_processed']}\n"
                    f"**Learning Records:** {status['learning']['total_records']}\n"
                    f"**Memories:** {status['memory']['total_memories']}\n"
                    f"**Scheduled Tasks:** {status['scheduler']['total_tasks']}"
                ),
                inline=False
            )

            # Discord Integration
            guild_count = len(self.bot.guilds)
            total_members = sum(g.member_count for g in self.bot.guilds)
            embed.add_field(
                name="🌐 Discord Integration",
                value=(
                    f"**Servers:** {guild_count}\n"
                    f"**Total Members:** {total_members}\n"
                    f"**Latency:** {round(self.bot.latency * 1000)}ms\n"
                    f"**Discord Tools:** 21 tools exported"
                ),
                inline=False
            )

            # Voice Status (if available)
            if VOICE_SUPPORT:
                voice_connections = len(self.bot.voice_clients)
                listening_count = sum(1 for vc in self.bot.voice_clients if vc.is_listening())
                tts_enabled_count = sum(1 for enabled in self.output_router.tts_enabled.values() if enabled)

                embed.add_field(
                    name="🎤 Voice Status",
                    value=(
                        f"**Voice Connections:** {voice_connections}\n"
                        f"**Listening:** {listening_count}\n"
                        f"**TTS Enabled:** {tts_enabled_count}\n"
                        f"**Voice Support:** {'✅ Full' if VOICE_RECEIVE_SUPPORT and GROQ_SUPPORT else '⚠️ Partial'}"
                    ),
                    inline=False
                )

            # Agent Status
            agent_tools_count = len(self.kernel.agent.tools) if hasattr(self.kernel.agent, 'tools') else "N/A"
            embed.add_field(
                name="🧠 Agent Status",
                value=(
                    f"**Total Tools:** {agent_tools_count}\n"
                    f"**Learning:** {'✅ Active' if status['learning']['total_records'] > 0 else '⚠️ No data'}\n"
                    f"**Memory:** {'✅ Active' if status['memory']['total_memories'] > 0 else '⚠️ Empty'}"
                ),
                inline=False
            )

            embed.set_footer(text=f"ProA Kernel v2.0 | Uptime: {status.get('uptime', 'N/A')}")

            await ctx.send(embed=embed)

        @self.bot.command(name="exit")
        async def exit_command(ctx: commands.Context):
            """Exit the kernel (Admin only)"""
            # Check admin permission
            if not await self._check_admin_permission(ctx):
                return

            await ctx.send("👋 Goodbye!")
            await self.stop()
            sys.exit(0)

        @self.bot.command(name="info")
        async def help_command(ctx: commands.Context):
            """Show comprehensive help message"""
            embed = discord.Embed(
                title="🤖 ProA Kernel - AI Assistant",
                description="Advanced AI-powered assistant with learning, memory, voice support, and Discord integration",
                color=discord.Color.green()
            )

            # Basic Commands
            basic_commands = (
                "• `!status` - Show kernel status and metrics\n"
                "• `!info` - Show this help message\n"
                "• `!progress [on|off|toggle]` - Toggle agent progress tracking\n"
                "• `!context` - Show agent context and user profile\n"
                "• `!reset` - Reset user data (memories, preferences, tasks)"
            )
            embed.add_field(
                name="📋 Basic Commands",
                value=basic_commands,
                inline=False
            )

            # Voice Commands (if available)
            if VOICE_SUPPORT:
                voice_commands = (
                    "• `!join` - Join your voice channel\n"
                    "• `!leave` - Leave voice channel\n"
                    "• `!voice_status` - Show voice connection status"
                )
                if VOICE_RECEIVE_SUPPORT and GROQ_SUPPORT:
                    voice_commands += (
                        "\n• `!listen` - Start voice transcription (Groq Whisper)\n"
                        "• `!stop_listening` - Stop voice transcription"
                    )
                voice_commands += "\n• `!tts [elevenlabs|piper|off]` - Toggle Text-to-Speech"

                embed.add_field(
                    name="🎤 Voice Commands",
                    value=voice_commands,
                    inline=False
                )

            # Agent Capabilities
            agent_capabilities = (
                "• **21 Discord Tools** - Server, message, voice, role management\n"
                "• **Learning System** - Learns from interactions and feedback\n"
                "• **Memory System** - Remembers important information\n"
                "• **Task Scheduling** - Can schedule reminders and tasks\n"
                "• **Multi-Speaker Support** - Tracks individual users in voice"
            )
            embed.add_field(
                name="🧠 Agent Capabilities",
                value=agent_capabilities,
                inline=False
            )

            # Usage
            usage = (
                "• **Mention me** or **DM me** to chat\n"
                "• I can manage messages, roles, and server settings\n"
                "• I can join voice channels and transcribe speech\n"
                "• I learn from feedback and improve over time"
            )
            embed.add_field(
                name="💡 How to Use",
                value=usage,
                inline=False
            )

            # Voice Features (if available)
            if VOICE_SUPPORT:
                voice_features = (
                    "• **Voice Input** - Real-time transcription with Groq Whisper\n"
                    "• **Voice Output** - TTS with ElevenLabs or Piper\n"
                    "• **Voice Activity Detection** - Automatic speech detection\n"
                    "• **DM Voice Channels** - Works in private calls too"
                )
                embed.add_field(
                    name="🔊 Voice Features",
                    value=voice_features,
                    inline=False
                )

            embed.set_footer(text="ProA Kernel v2.0 | Powered by Augment AI")

            await ctx.send(embed=embed)

        # Voice Commands (only if voice support is available)
        if VOICE_SUPPORT:
            @self.bot.command(name="join")
            async def join_voice(ctx: commands.Context):
                """Join the user's voice channel (Guild or DM)"""
                print(f"🎤 [DEBUG] !join command called by {ctx.author.display_name}")

                # Check if user is in a voice channel
                if not ctx.author.voice:
                    print(f"🎤 [DEBUG] User is not in a voice channel")
                    await ctx.send("❌ You need to be in a voice channel!")
                    return

                channel = ctx.author.voice.channel
                channel_name = getattr(channel, 'name', 'DM Voice Channel')
                print(f"🎤 [DEBUG] User is in voice channel: {channel_name}")

                try:
                    if ctx.voice_client:
                        print(f"🎤 [DEBUG] Bot already in voice, moving to {channel_name}")
                        await ctx.voice_client.move_to(channel)
                        await ctx.send(f"🔊 Moved to {channel_name}")
                        print(f"🎤 [DEBUG] Successfully moved to {channel_name}")
                    else:
                        print(f"🎤 [DEBUG] Connecting to voice channel {channel_name}...")

                        # Use VoiceRecvClient if voice receive support is available
                        if VOICE_RECEIVE_SUPPORT:
                            print(f"🎤 [DEBUG] Using VoiceRecvClient for voice receive support")
                            voice_client = await channel.connect(cls=voice_recv.VoiceRecvClient)
                        else:
                            print(f"🎤 [DEBUG] Using standard VoiceClient (no voice receive)")
                            voice_client = await channel.connect()

                        print(f"🎤 [DEBUG] Connected successfully")
                        print(f"🎤 [DEBUG] VoiceClient type: {type(voice_client).__name__}")
                        print(f"🎤 [DEBUG] Has listen method: {hasattr(voice_client, 'listen')}")
                        print(f"🎤 [DEBUG] Has is_listening method: {hasattr(voice_client, 'is_listening')}")

                        # Store voice client (use guild_id or user_id for DMs)
                        if ctx.guild:
                            self.output_router.voice_clients[ctx.guild.id] = voice_client
                            print(f"🎤 [DEBUG] Stored voice client for guild {ctx.guild.id}")
                            await ctx.send(f"🔊 Joined {channel.name}")
                        else:
                            # DM Voice Channel
                            self.output_router.voice_clients[ctx.author.id] = voice_client
                            print(f"🎤 [DEBUG] Stored voice client for user {ctx.author.id}")
                            await ctx.send(f"🔊 Joined DM voice channel")

                        print(f"🎤 [DEBUG] !join command completed successfully")
                except Exception as e:
                    print(f"❌ [DEBUG] Error in !join command: {e}")
                    import traceback
                    traceback.print_exc()
                    await ctx.send(f"❌ Error joining voice channel: {e}")

            @self.bot.command(name="leave")
            async def leave_voice(ctx: commands.Context):
                """Leave the voice channel (Guild or DM)"""
                if not ctx.voice_client:
                    await ctx.send("❌ I'm not in a voice channel!")
                    return

                try:
                    # Determine client ID (guild or user)
                    client_id = ctx.guild.id if ctx.guild else ctx.author.id

                    await ctx.voice_client.disconnect()

                    if client_id in self.output_router.voice_clients:
                        del self.output_router.voice_clients[client_id]
                    if client_id in self.output_router.audio_sinks:
                        del self.output_router.audio_sinks[client_id]
                    if client_id in self.output_router.tts_enabled:
                        del self.output_router.tts_enabled[client_id]

                    await ctx.send("👋 Left the voice channel")
                except Exception as e:
                    await ctx.send(f"❌ Error leaving voice channel: {e}")

            @self.bot.command(name="voice_status")
            async def voice_status(ctx: commands.Context):
                """Show voice connection status"""
                if not ctx.voice_client:
                    await ctx.send("❌ Not connected to any voice channel")
                    return

                vc = ctx.voice_client
                embed = discord.Embed(
                    title="🔊 Voice Status",
                    color=discord.Color.blue()
                )

                embed.add_field(name="Channel", value=vc.channel.name, inline=True)
                embed.add_field(name="Connected", value="✅" if vc.is_connected() else "❌", inline=True)
                embed.add_field(name="Playing", value="✅" if vc.is_playing() else "❌", inline=True)
                embed.add_field(name="Paused", value="✅" if vc.is_paused() else "❌", inline=True)
                embed.add_field(name="Latency", value=f"{vc.latency * 1000:.2f}ms", inline=True)

                # Check if listening
                if VOICE_RECEIVE_SUPPORT and hasattr(vc, 'is_listening'):
                    is_listening = vc.is_listening()
                    embed.add_field(name="Listening", value="✅" if is_listening else "❌", inline=True)

                await ctx.send(embed=embed)

            # Voice input commands (only if voice receive support is available)
            print(f"🎤 [DEBUG] Checking voice input command registration...")
            print(f"🎤 [DEBUG] VOICE_RECEIVE_SUPPORT: {VOICE_RECEIVE_SUPPORT}")
            print(f"🎤 [DEBUG] GROQ_SUPPORT: {GROQ_SUPPORT}")

            if VOICE_RECEIVE_SUPPORT and GROQ_SUPPORT:
                print(f"🎤 [DEBUG] ✅ Registering !listen and !stop_listening commands")

                @self.bot.command(name="listen")
                async def start_listening(ctx: commands.Context):
                    """Start listening to voice input and transcribing with Groq Whisper"""
                    print(f"🎤 [DEBUG] !listen command called by {ctx.author.display_name}")

                    if not ctx.voice_client:
                        print(f"🎤 [DEBUG] Bot is not in a voice channel")
                        await ctx.send("❌ I'm not in a voice channel! Use `!join` first.")
                        return

                    # Check if already listening (only if voice_recv is available)
                    if hasattr(ctx.voice_client, 'is_listening') and ctx.voice_client.is_listening():
                        print(f"🎤 [DEBUG] Already listening")
                        await ctx.send("⚠️ Already listening!")
                        return

                    try:
                        guild_id = ctx.guild.id
                        print(f"🎤 [DEBUG] Guild ID: {guild_id}")

                        # Create audio sink with Discord context
                        print(f"🎤 [DEBUG] Creating WhisperAudioSink...")
                        sink = WhisperAudioSink(
                            kernel=self.kernel,
                            user_id=str(ctx.author.id),
                            groq_client=self.output_router.groq_client,
                            output_router=self.output_router,
                            discord_kernel=self  # Pass Discord kernel for context
                        )
                        print(f"🎤 [DEBUG] WhisperAudioSink created successfully")

                        # Start listening
                        print(f"🎤 [DEBUG] Starting voice client listening...")

                        # Check if listen method exists
                        if not hasattr(ctx.voice_client, 'listen'):
                            print(f"🎤 [DEBUG] ERROR: listen() method not available on VoiceClient")
                            print(f"🎤 [DEBUG] This means discord-ext-voice-recv is NOT installed!")
                            await ctx.send("❌ Voice receive not supported! Install: `pip install discord-ext-voice-recv`")
                            return

                        ctx.voice_client.listen(sink)
                        self.output_router.audio_sinks[guild_id] = sink
                        print(f"🎤 [DEBUG] Voice client is now listening")

                        await ctx.send("🎤 Started listening! Speak and I'll transcribe your voice in real-time.")
                        print(f"🎤 [DEBUG] !listen command completed successfully")
                    except Exception as e:
                        print(f"❌ [DEBUG] Error in !listen command: {e}")
                        import traceback
                        traceback.print_exc()
                        await ctx.send(f"❌ Error starting voice input: {e}")

                @self.bot.command(name="stop_listening")
                async def stop_listening(ctx: commands.Context):
                    """Stop listening to voice input"""
                    print(f"🎤 [DEBUG] !stop_listening command called by {ctx.author.display_name}")

                    if not ctx.voice_client:
                        print(f"🎤 [DEBUG] Bot is not in a voice channel")
                        await ctx.send("❌ I'm not in a voice channel!")
                        return

                    # Check if listening (only if voice_recv is available)
                    if not hasattr(ctx.voice_client, 'is_listening') or not ctx.voice_client.is_listening():
                        print(f"🎤 [DEBUG] Not currently listening")
                        await ctx.send("⚠️ Not currently listening!")
                        return

                    try:
                        guild_id = ctx.guild.id
                        print(f"🎤 [DEBUG] Stopping voice client listening...")

                        # Stop listening (only if method exists)
                        if hasattr(ctx.voice_client, 'stop_listening'):
                            ctx.voice_client.stop_listening()
                        else:
                            print(f"🎤 [DEBUG] WARNING: stop_listening method not available")
                            await ctx.send("❌ Voice receive not supported!")
                            return

                        if guild_id in self.output_router.audio_sinks:
                            print(f"🎤 [DEBUG] Removing audio sink for guild {guild_id}")
                            del self.output_router.audio_sinks[guild_id]

                        await ctx.send("🔇 Stopped listening to voice input.")
                        print(f"🎤 [DEBUG] !stop_listening command completed successfully")
                    except Exception as e:
                        print(f"❌ [DEBUG] Error in !stop_listening command: {e}")
                        import traceback
                        traceback.print_exc()
                        await ctx.send(f"❌ Error stopping voice input: {e}")
            else:
                print(f"🎤 [DEBUG] ❌ Voice input commands NOT registered!")
                print(f"🎤 [DEBUG] Reason: VOICE_RECEIVE_SUPPORT={VOICE_RECEIVE_SUPPORT}, GROQ_SUPPORT={GROQ_SUPPORT}")

            # TTS Commands
            @self.bot.command(name="tts")
            async def toggle_tts(ctx: commands.Context, mode: str = None):
                """Toggle TTS (Text-to-Speech) on/off. Usage: !tts [elevenlabs|piper|off]"""
                if not ctx.guild:
                    await ctx.send("❌ TTS only works in servers!")
                    return

                guild_id = ctx.guild.id

                if mode is None:
                    # Show current status
                    enabled = self.output_router.tts_enabled.get(guild_id, False)
                    current_mode = self.output_router.tts_mode.get(guild_id, "piper")
                    status = f"🔊 TTS is {'enabled' if enabled else 'disabled'}"
                    if enabled:
                        status += f" (mode: {current_mode})"
                    await ctx.send(status)
                    return

                mode = mode.lower()

                if mode == "off":
                    self.output_router.tts_enabled[guild_id] = False
                    await ctx.send("🔇 TTS disabled")
                elif mode in ["elevenlabs", "piper"]:
                    # Check if mode is available
                    if mode == "elevenlabs" and not (ELEVENLABS_SUPPORT and self.output_router.elevenlabs_client):
                        await ctx.send("❌ ElevenLabs not available. Set ELEVENLABS_API_KEY.")
                        return
                    if mode == "piper" and not self.output_router.piper_path:
                        await ctx.send("❌ Piper not available. Check PIPER_PATH.")
                        return

                    self.output_router.tts_enabled[guild_id] = True
                    self.output_router.tts_mode[guild_id] = mode
                    await ctx.send(f"🔊 TTS enabled with {mode}")
                else:
                    await ctx.send("❌ Invalid mode. Use: !tts [elevenlabs|piper|off]")

        else:

            @self.bot.command(name="join")
            async def join_voice_disabled(ctx: commands.Context):
                """Voice support not available"""
                await ctx.send("❌ Voice support is not available. Install PyNaCl: `pip install discord.py[voice]`")

        # Progress tracking command
        @self.bot.command(name="progress")
        async def progress_command(ctx: commands.Context, action: str = "toggle"):
            """Toggle agent progress tracking. Usage: !progress [on|off|toggle]"""
            user_id = str(ctx.author.id)

            if action.lower() == "on":
                # Enable progress tracking
                if user_id not in self.progress_printers:
                    printer = DiscordProgressPrinter(ctx.channel, user_id)
                    self.progress_printers[user_id] = printer
                    # Register global progress callback if not already registered
                    if not hasattr(self, '_progress_callback_registered'):
                        self.kernel.agent.set_progress_callback(self._dispatch_progress_event)
                        self._progress_callback_registered = True
                    await printer.enable()
                    await ctx.send("✅ Progress tracking enabled!")
                else:
                    await self.progress_printers[user_id].enable()
                    await ctx.send("✅ Progress tracking re-enabled!")

            elif action.lower() == "off":
                # Disable progress tracking
                if user_id in self.progress_printers:
                    await self.progress_printers[user_id].disable()
                    await ctx.send("✅ Progress tracking disabled!")
                else:
                    await ctx.send("⚠️ Progress tracking is not active!")

            elif action.lower() == "toggle":
                # Toggle progress tracking
                if user_id in self.progress_printers:
                    printer = self.progress_printers[user_id]
                    if printer.enabled:
                        await printer.disable()
                        await ctx.send("✅ Progress tracking disabled!")
                    else:
                        await printer.enable()
                        await ctx.send("✅ Progress tracking enabled!")
                else:
                    # Create new printer
                    printer = DiscordProgressPrinter(ctx.channel, user_id)
                    self.progress_printers[user_id] = printer
                    # Register global progress callback if not already registered
                    if not hasattr(self, '_progress_callback_registered'):
                        self.kernel.agent.set_progress_callback(self._dispatch_progress_event)
                        self._progress_callback_registered = True
                    await printer.enable()
                    await ctx.send("✅ Progress tracking enabled!")
            else:
                await ctx.send("❌ Invalid action. Use: !progress [on|off|toggle]")

        # Reset command
        @self.bot.command(name="reset")
        async def reset_command(ctx: commands.Context):
            """Reset user data (memories, context, preferences, scheduled tasks, history)"""

            user_id = str(ctx.author.id)

            embed = discord.Embed(
                title="🔄 Reset User Data",
                description="Choose what you want to reset. **Warning:** This action cannot be undone!",
                color=discord.Color.orange()
            )

            # Show current data counts
            user_memories = self.kernel.memory_store.user_memories.get(user_id, [])
            user_prefs = self.kernel.learning_engine.preferences.get(user_id)
            user_tasks = self.kernel.scheduler.get_user_tasks(user_id)

            data_summary = (
                f"**Memories:** {len(user_memories)}\n"
                f"**Preferences:** {'✅ Set' if user_prefs else '❌ None'}\n"
                f"**Scheduled Tasks:** {len(user_tasks)}\n"
            )
            embed.add_field(name="📊 Current Data", value=data_summary, inline=False)

            # Create interactive view with reset buttons
            view = discord.ui.View(timeout=60)  # 1 minute timeout

            # Button: Reset Memories
            reset_memories_btn = discord.ui.Button(
                label=f"🗑️ Reset Memories ({len(user_memories)})",
                style=discord.ButtonStyle.danger,
                custom_id=f"reset_memories_{user_id}"
            )

            async def reset_memories_callback(interaction: discord.Interaction):
                if str(interaction.user.id) != user_id:
                    await interaction.response.send_message("❌ This is not your reset menu!", ephemeral=True)
                    return

                # Delete all memories
                if user_id in self.kernel.memory_store.user_memories:
                    count = len(self.kernel.memory_store.user_memories[user_id])
                    self.kernel.memory_store.user_memories[user_id] = []
                    await interaction.response.send_message(
                        f"✅ Deleted {count} memories!",
                        ephemeral=True
                    )
                else:
                    await interaction.response.send_message("⚠️ No memories to delete!", ephemeral=True)

            reset_memories_btn.callback = reset_memories_callback
            view.add_item(reset_memories_btn)

            # Button: Reset Preferences
            reset_prefs_btn = discord.ui.Button(
                label="⚙️ Reset Preferences",
                style=discord.ButtonStyle.danger,
                custom_id=f"reset_prefs_{user_id}"
            )

            async def reset_prefs_callback(interaction: discord.Interaction):
                if str(interaction.user.id) != user_id:
                    await interaction.response.send_message("❌ This is not your reset menu!", ephemeral=True)
                    return

                # Delete preferences
                if user_id in self.kernel.learning_engine.preferences:
                    del self.kernel.learning_engine.preferences[user_id]
                    await interaction.response.send_message("✅ Preferences reset!", ephemeral=True)
                else:
                    await interaction.response.send_message("⚠️ No preferences to reset!", ephemeral=True)

            reset_prefs_btn.callback = reset_prefs_callback
            view.add_item(reset_prefs_btn)

            # Button: Reset Scheduled Tasks
            reset_tasks_btn = discord.ui.Button(
                label=f"📅 Reset Tasks ({len(user_tasks)})",
                style=discord.ButtonStyle.danger,
                custom_id=f"reset_tasks_{user_id}"
            )

            async def reset_tasks_callback(interaction: discord.Interaction):
                if str(interaction.user.id) != user_id:
                    await interaction.response.send_message("❌ This is not your reset menu!", ephemeral=True)
                    return

                # Cancel all user tasks
                user_tasks = self.kernel.scheduler.get_user_tasks(user_id)
                cancelled_count = 0
                for task in user_tasks:
                    if await self.kernel.scheduler.cancel_task(task.id):
                        cancelled_count += 1

                await interaction.response.send_message(
                    f"✅ Cancelled {cancelled_count} scheduled tasks!",
                    ephemeral=True
                )

            reset_tasks_btn.callback = reset_tasks_callback
            view.add_item(reset_tasks_btn)

            session = self.kernel.agent.context_manager.session_managers.get(user_id, {"history": []})
            if hasattr(session, 'history'):
                len_his = len(session.history)
            elif isinstance(session, dict) and 'history' in session:
                len_his = len(session['history'])

            # Button: Reset History Tasks
            reset_history_btn = discord.ui.Button(
                label=f"📜 Reset History ({len_his})",
                style=discord.ButtonStyle.danger,
                custom_id=f"reset_history_{user_id}"
            )

            async def reset_history_callback(interaction: discord.Interaction):
                if str(interaction.user.id) != user_id:
                    await interaction.response.send_message("❌ This is not your reset menu!", ephemeral=True)
                    return

                # Clear history
                self.kernel.agent.clear_context(user_id)

                await interaction.response.send_message(
                    f"✅ History reset!",
                    ephemeral=True
                )

            reset_history_btn.callback = reset_history_callback
            view.add_item(reset_history_btn)

            # Button: Reset ALL
            reset_all_btn = discord.ui.Button(
                label="🔥 Reset ALL",
                style=discord.ButtonStyle.danger,
                custom_id=f"reset_all_{user_id}"
            )

            async def reset_all_callback(interaction: discord.Interaction):
                if str(interaction.user.id) != user_id:
                    await interaction.response.send_message("❌ This is not your reset menu!", ephemeral=True)
                    return

                # Reset everything
                mem_count = 0
                if user_id in self.kernel.memory_store.user_memories:
                    mem_count = len(self.kernel.memory_store.user_memories[user_id])
                    self.kernel.memory_store.user_memories[user_id] = []

                prefs_reset = False
                if user_id in self.kernel.learning_engine.preferences:
                    del self.kernel.learning_engine.preferences[user_id]
                    prefs_reset = True

                user_tasks = self.kernel.scheduler.get_user_tasks(user_id)
                task_count = 0
                for task in user_tasks:
                    if await self.kernel.scheduler.cancel_task(task.id):
                        task_count += 1

                summary = (
                    f"✅ **Reset Complete!**\n"
                    f"• Deleted {mem_count} memories\n"
                    f"• Reset preferences: {'✅' if prefs_reset else '❌'}\n"
                    f"• Cancelled {task_count} tasks"
                )
                await interaction.response.send_message(summary, ephemeral=True)

            reset_all_btn.callback = reset_all_callback
            view.add_item(reset_all_btn)

            await ctx.send(embed=embed, view=view)

        # Context overview command
        @self.bot.command(name="context")
        async def context_command(ctx: commands.Context):
            """Show agent context, user profile, and usage statistics"""
            user_id = str(ctx.author.id)

            try:
                # Get context overview from agent
                context_overview = await self.kernel.agent.get_context_overview(display=False)

                # Create embed
                embed = discord.Embed(
                    title="🧠 Agent Context & User Profile",
                    description=f"Context information for <@{user_id}>",
                    color=discord.Color.blue(),
                    timestamp= datetime.now(UTC)
                )

                # Usage Statistics
                total_tokens = self.kernel.agent.total_tokens_in + self.kernel.agent.total_tokens_out
                usage_stats = (
                    f"**Total Cost:** ${self.kernel.agent.total_cost_accumulated:.4f}\n"
                    f"**Total LLM Calls:** {self.kernel.agent.total_llm_calls}\n"
                    f"**Tokens In:** {self.kernel.agent.total_tokens_in:,}\n"
                    f"**Tokens Out:** {self.kernel.agent.total_tokens_out:,}\n"
                    f"**Total Tokens:** {total_tokens:,}"
                )
                embed.add_field(name="💰 Usage Statistics", value=usage_stats, inline=False)

                # Discord Context (if available)
                if hasattr(self.kernel.agent, 'variable_manager'):
                    discord_context = self.kernel.agent.variable_manager.get(f'discord.current_context.{user_id}')
                    if discord_context:
                        location_info = (
                            f"**Channel Type:** {discord_context.get('channel_type', 'Unknown')}\n"
                            f"**Channel:** {discord_context.get('channel_name', 'Unknown')}\n"
                        )
                        if discord_context.get('guild_name'):
                            location_info += f"**Server:** {discord_context['guild_name']}\n"

                        embed.add_field(name="📍 Current Location", value=location_info, inline=False)

                        # Voice Status
                        bot_voice = discord_context.get('bot_voice_status', {})
                        if bot_voice.get('in_voice'):
                            voice_info = (
                                f"**In Voice:** ✅\n"
                                f"**Channel:** {bot_voice.get('channel_name', 'Unknown')}\n"
                                f"**Listening:** {'✅' if bot_voice.get('listening') else '❌'}\n"
                                f"**TTS:** {'✅' if bot_voice.get('tts_enabled') else '❌'}"
                            )
                            embed.add_field(name="🎤 Voice Status", value=voice_info, inline=False)

                # Kernel Status
                kernel_status = self.kernel.to_dict()
                kernel_info = (
                    f"**State:** {kernel_status['state']}\n"
                    f"**Signals Processed:** {kernel_status['metrics']['signals_processed']}\n"
                    f"**Memories:** {kernel_status['memory']['total_memories']}\n"
                    f"**Learning Records:** {kernel_status['learning']['total_records']}"
                )
                embed.add_field(name="🤖 Kernel Status", value=kernel_info, inline=False)

                # Context Overview (if available)
                if context_overview and 'token_summary' in context_overview:
                    token_summary = context_overview['token_summary']
                    total_tokens = token_summary.get('total_tokens', 0)
                    breakdown = token_summary.get('breakdown', {})
                    percentages = token_summary.get('percentage_breakdown', {})

                    # Get max tokens for models
                    try:
                        max_tokens_fast = self.kernel.agent.amd.max_input_tokens
                        max_tokens_complex = self.kernel.agent.amd.max_input_tokens
                    except:
                        max_tokens_fast = self.kernel.agent.amd.max_tokens if hasattr(self.kernel.agent.amd, 'max_tokens') else 128000
                        max_tokens_complex = max_tokens_fast

                    # Context Distribution with visual bars
                    context_text = f"**Total Context:** ~{total_tokens:,} tokens\n\n"

                    # Components with visual bars (Discord-friendly)
                    components = [
                        ("System prompt", "system_prompt", "🔧"),
                        ("Agent tools", "agent_tools", "🛠️"),
                        ("Meta tools", "meta_tools", "⚡"),
                        ("Variables", "variables", "📝"),
                        ("History", "system_history", "📚"),
                        ("Unified ctx", "unified_context", "🔗"),
                        ("Reasoning", "reasoning_context", "🧠"),
                        ("LLM Tools", "llm_tool_context", "🤖"),
                    ]

                    for name, key, icon in components:
                        token_count = breakdown.get(key, 0)
                        if token_count > 0:
                            percentage = percentages.get(key, 0)
                            # Create visual bar (Discord-friendly, max 10 chars)
                            bar_length = int(percentage / 10)  # 10 chars max (100% / 10)
                            bar = "█" * bar_length + "░" * (10 - bar_length)
                            context_text += f"{icon} `{name:12s}` {bar} {percentage:4.1f}% ({token_count:,})\n"

                    # Add free space info
                    usage_fast = (total_tokens / max_tokens_fast * 100) if max_tokens_fast > 0 else 0
                    usage_complex = (total_tokens / max_tokens_complex * 100) if max_tokens_complex > 0 else 0
                    context_text += f"\n⬜ `Fast Model  ` {max_tokens_fast:,} tokens | Used: {usage_fast:.1f}%\n"
                    context_text += f"⬜ `Complex Mdl ` {max_tokens_complex:,} tokens | Used: {usage_complex:.1f}%"

                    embed.add_field(name="📊 Context Distribution", value=context_text, inline=False)

                # Get user-specific data counts
                # user_memories contains memory IDs, not Memory objects - need to fetch the actual objects
                user_memory_ids = self.kernel.memory_store.user_memories.get(user_id, [])
                user_memories = [
                    self.kernel.memory_store.memories[mid]
                    for mid in user_memory_ids
                    if mid in self.kernel.memory_store.memories
                ]
                user_learning = [r for r in self.kernel.learning_engine.records if r.user_id == user_id]
                user_prefs = self.kernel.learning_engine.preferences.get(user_id)
                user_tasks = self.kernel.scheduler.get_user_tasks(user_id)

                # Add user data summary
                user_data_summary = (
                    f"**Memories:** {len(user_memories)}\n"
                    f"**Learning Records:** {len(user_learning)}\n"
                    f"**Preferences:** {'✅ Learned' if user_prefs else '❌ Not yet'}\n"
                    f"**Scheduled Tasks:** {len(user_tasks)}"
                )
                embed.add_field(name="🧑 What I Know About You", value=user_data_summary, inline=False)

                embed.set_footer(text="ProA Kernel Context System • Use buttons below for details")

                # Create interactive view with buttons
                view = discord.ui.View(timeout=300)  # 5 minutes timeout

                # Button: Show Memories
                memories_button = discord.ui.Button(
                    label=f"📝 Memories ({len(user_memories)})",
                    style=discord.ButtonStyle.primary,
                    custom_id=f"context_memories_{user_id}"
                )

                async def memories_callback(interaction: discord.Interaction):
                    if str(interaction.user.id) != user_id:
                        await interaction.response.send_message("❌ This is not your context!", ephemeral=True)
                        return

                    # Formatter function for memories
                    def format_memory(mem):
                        importance_bar = "⭐" * int(mem.importance * 5)
                        tags_str = f" `[{', '.join(mem.tags[:3])}]`" if mem.tags else ""
                        content = mem.content[:200] + "..." if len(mem.content) > 200 else mem.content

                        return {
                            'name': f"{importance_bar} {mem.memory_type.value.upper()}{tags_str}",
                            'value': content,
                            'inline': False
                        }

                    # Create paginated view
                    view = ContextPaginationView(user_id, 'memories', user_memories, format_memory)
                    embed = view.create_embed()

                    await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

                memories_button.callback = memories_callback
                view.add_item(memories_button)

                # Button: Show Preferences
                prefs_button = discord.ui.Button(
                    label="⚙️ Preferences",
                    style=discord.ButtonStyle.primary,
                    custom_id=f"context_prefs_{user_id}"
                )

                async def prefs_callback(interaction: discord.Interaction):
                    if str(interaction.user.id) != user_id:
                        await interaction.response.send_message("❌ This is not your context!", ephemeral=True)
                        return

                    prefs_embed = discord.Embed(
                        title="⚙️ Your Preferences",
                        color=discord.Color.blue()
                    )

                    if user_prefs:
                        prefs_text = (
                            f"**Communication Style:** {user_prefs.communication_style}\n"
                            f"**Response Format:** {user_prefs.response_format}\n"
                            f"**Proactivity Level:** {user_prefs.proactivity_level}\n"
                            f"**Preferred Tools:** {', '.join(user_prefs.preferred_tools) if user_prefs.preferred_tools else 'None yet'}\n"
                            f"**Topic Interests:** {', '.join(user_prefs.topic_interests) if user_prefs.topic_interests else 'None yet'}\n"
                            f"**Time Preferences:** {user_prefs.time_preferences or 'Not learned yet'}"
                        )
                        prefs_embed.description = prefs_text
                        prefs_embed.set_footer(text=f"Last updated: {datetime.fromtimestamp(user_prefs.last_updated).strftime('%Y-%m-%d %H:%M:%S')}")
                    else:
                        prefs_embed.description = "No preferences learned yet. I'll adapt to your style as we interact!"

                    await interaction.response.send_message(embed=prefs_embed, ephemeral=True)

                prefs_button.callback = prefs_callback
                view.add_item(prefs_button)

                # Button: Show Learning Records
                learning_button = discord.ui.Button(
                    label=f"📚 Learning ({len(user_learning)})",
                    style=discord.ButtonStyle.primary,
                    custom_id=f"context_learning_{user_id}"
                )

                async def learning_callback(interaction: discord.Interaction):
                    if str(interaction.user.id) != user_id:
                        await interaction.response.send_message("❌ This is not your context!", ephemeral=True)
                        return

                    # Sort by timestamp (newest first)
                    sorted_learning = sorted(user_learning, key=lambda r: r.timestamp, reverse=True)

                    # Formatter function for learning records
                    def format_learning(record):
                        if record.feedback_score is not None:
                            feedback_emoji = "👍" if record.feedback_score > 0 else "👎"
                        else:
                            feedback_emoji = "➖"


                        time_str = datetime.fromtimestamp(record.timestamp).strftime('%Y-%m-%d %H:%M')
                        content = record.content or record.outcome or "No content"
                        content_preview = content[:200] + "..." if len(content) > 200 else content

                        return {
                            'name': f"{record.interaction_type.value} - {time_str} {feedback_emoji}",
                            'value': content_preview,
                            'inline': False
                        }

                    # Create paginated view
                    view = ContextPaginationView(user_id, 'learning', sorted_learning, format_learning)
                    embed = view.create_embed()

                    await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

                learning_button.callback = learning_callback
                view.add_item(learning_button)

                session = self.kernel.agent.context_manager.session_managers.get(user_id, {"history": []})
                if hasattr(session, 'history'):
                    user_history = session.history
                elif isinstance(session, dict) and 'history' in session:
                    user_history = session['history']
                else:
                    user_history = []

                # Button: Show History Records
                history_button = discord.ui.Button(
                    label=f"📚 History ({len(user_history)})",
                    style=discord.ButtonStyle.primary,
                    custom_id=f"context_history_{user_id}"
                )

                async def history_callback(interaction: discord.Interaction):
                    if str(interaction.user.id) != user_id:
                        await interaction.response.send_message("❌ This is not your context!", ephemeral=True)
                        return

                    # Reverse to show newest first
                    reversed_history = list(user_history)

                    # Formatter function for history records
                    def format_history(record):
                        if isinstance(record, dict):
                            role = record.get('role', 'unknown')
                            content = record.get('content', 'unknown')
                        elif hasattr(record, 'role') and hasattr(record, 'content'):
                            role = record.role
                            content = record.content
                        else:
                            role = 'unknown'
                            content = str(record)

                        # Truncate long content
                        content_preview = content[:500] + "..." if len(content) > 500 else content

                        # Role emoji mapping
                        role_emoji = {
                            'user': '👤',
                            'assistant': '🤖',
                            'system': '⚙️',
                            'tool': '🛠️'
                        }
                        emoji = role_emoji.get(role.lower(), '❓')

                        return {
                            'name': f"{emoji} {role.upper()}",
                            'value': f"```\n{content_preview}\n```",
                            'inline': False
                        }

                    # Create paginated view
                    view = ContextPaginationView(user_id, 'history', reversed_history, format_history)
                    embed = view.create_embed()

                    await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

                history_button.callback = history_callback
                view.add_item(history_button)

                # Button: Show All Memories (Full List)
                all_memories_button = discord.ui.Button(
                    label="📋 All Memories",
                    style=discord.ButtonStyle.secondary,
                    custom_id=f"context_all_memories_{user_id}"
                )

                async def all_memories_callback(interaction: discord.Interaction):
                    if str(interaction.user.id) != user_id:
                        await interaction.response.send_message("❌ This is not your context!", ephemeral=True)
                        return

                    if not user_memories:
                        await interaction.response.send_message("📝 No memories stored yet!", ephemeral=True)
                        return

                    # Sort by importance (highest first)
                    sorted_memories = sorted(user_memories, key=lambda m: m.importance, reverse=True)

                    # Formatter function for all memories (detailed view)
                    def format_detailed_memory(mem):
                        importance_bar = "⭐" * int(mem.importance * 5)
                        tags_str = f"\n**Tags:** {mem.importance:.2f} {', '.join(mem.tags)}" if mem.tags else ""

                        # Add metadata
                        metadata_lines = []
                        if hasattr(mem, 'created_at'):
                            created = datetime.fromtimestamp(mem.created_at).strftime('%Y-%m-%d %H:%M')
                            metadata_lines.append(f"**Created:** {created}")
                        if hasattr(mem, 'last_accessed'):
                            accessed = datetime.fromtimestamp(mem.last_accessed).strftime('%Y-%m-%d %H:%M')
                            metadata_lines.append(f"**Last Accessed:** {accessed}")
                        if hasattr(mem, 'access_count'):
                            metadata_lines.append(f"**Access Count:** {mem.access_count}")

                        metadata = "\n".join(metadata_lines) if metadata_lines else ""

                        return {
                            'name': f"{importance_bar} {mem.memory_type.value.upper()}",
                            'value': f"{mem.content}{tags_str}\n{metadata}",
                            'inline': False
                        }

                    # Create paginated view with detailed formatting
                    view = ContextPaginationView(user_id, 'memories', sorted_memories, format_detailed_memory)
                    view.items_per_page = 3  # Fewer items per page for detailed view
                    view.total_pages = (len(sorted_memories) + view.items_per_page - 1) // view.items_per_page
                    view._build_buttons()

                    embed = view.create_embed()
                    embed.title = "📋 All Memories (Detailed)"

                    await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

                all_memories_button.callback = all_memories_callback
                view.add_item(all_memories_button)

                # Button: Show Scheduled Tasks
                tasks_button = discord.ui.Button(
                    label=f"📅 Scheduled Tasks ({len(user_tasks)})",
                    style=discord.ButtonStyle.secondary,
                    custom_id=f"context_tasks_{user_id}"
                )

                async def tasks_callback(interaction: discord.Interaction):
                    if str(interaction.user.id) != user_id:
                        await interaction.response.send_message("❌ This is not your context!", ephemeral=True)
                        return

                    # Sort by scheduled time (nearest first)
                    sorted_tasks = sorted(user_tasks, key=lambda t: t.scheduled_time)

                    # Formatter function for tasks
                    def format_task(task):
                        scheduled_dt = datetime.fromtimestamp(task.scheduled_time).strftime('%Y-%m-%d %H:%M')
                        priority_stars = "⭐" * task.priority
                        status_emoji = {
                            'pending': '⏳',
                            'completed': '✅',
                            'failed': '❌',
                            'cancelled': '🚫'
                        }
                        emoji = status_emoji.get(task.status.value.lower(), '❓')

                        content = task.content[:200] + "..." if len(task.content) > 200 else task.content

                        return {
                            'name': f"{emoji} {priority_stars} {task.task_type} - {scheduled_dt}",
                            'value': f"**Status:** {task.status.value}\n{content}",
                            'inline': False
                        }

                    # Create paginated view
                    view = ContextPaginationView(user_id, 'tasks', sorted_tasks, format_task)
                    embed = view.create_embed()

                    await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

                tasks_button.callback = tasks_callback
                view.add_item(tasks_button)

                await ctx.send(embed=embed, view=view)

            except Exception as e:
                await ctx.send(f"❌ Error retrieving context: {e}")

        @self.bot.command(name="restrict")
        async def restrict_command(ctx: commands.Context, action: str = None, *, args: str = None):
            """
            Manage tool restrictions for sessions (Admin only).

            Usage:
                !restrict list                           - List all tool restrictions
                !restrict sessions                       - List all known sessions/users
                !restrict tools                          - List all available tools
                !restrict set <tool> <session> <allow>   - Set restriction for specific session
                !restrict default <tool> <allow>         - Set default restriction for tool
                !restrict reset [tool]                   - Reset restrictions (all or specific tool)
                !restrict check <tool> <session>         - Check if tool is allowed in session

            Examples:
                !restrict list
                !restrict sessions
                !restrict set execute_python 123456789 false
                !restrict default dangerous_tool false
                !restrict reset execute_python
                !restrict check execute_python 123456789
            """
            # Check admin permission
            if not await self._check_admin_permission(ctx):
                return

            if not hasattr(self.kernel.agent, 'session_tool_restrictions'):
                await ctx.send("❌ Tool restrictions not available on this agent!")
                return

            user_id = str(ctx.author.id)

            # Default action is list
            if action is None:
                action = "list"

            action = action.lower()

            try:
                if action == "list":
                    # List all current restrictions
                    restrictions = self.kernel.agent.list_tool_restrictions()

                    if not restrictions:
                        await ctx.send("📋 No tool restrictions configured. All tools are allowed by default.")
                        return

                    embed = discord.Embed(
                        title="🔒 Tool Restrictions",
                        description=f"Total tools with restrictions: {len(restrictions)}",
                        color=discord.Color.orange(),
                        timestamp=datetime.now(UTC)
                    )

                    # Group by tool
                    for tool_name, sessions in restrictions.items():
                        # Format session restrictions
                        session_lines = []

                        # Show default first if exists
                        if '*' in sessions:
                            default_status = "✅ Allowed" if sessions['*'] else "❌ Restricted"
                            session_lines.append(f"**Default:** {default_status}")

                        # Show specific sessions
                        for session_id, allowed in sessions.items():
                            if session_id == '*':
                                continue

                            status = "✅" if allowed else "❌"

                            # Try to get user display name
                            try:
                                user = self.bot.get_user(int(session_id))
                                display_name = f"{user.display_name} ({session_id})" if user else session_id
                            except:
                                display_name = session_id

                            session_lines.append(f"{status} `{display_name}`")

                        session_text = "\n".join(session_lines) if session_lines else "No restrictions"

                        # Add field (max 1024 chars)
                        if len(session_text) > 1024:
                            session_text = session_text[:1020] + "..."

                        embed.add_field(
                            name=f"🛠️ {tool_name}",
                            value=session_text,
                            inline=False
                        )

                    embed.set_footer(text="Use !restrict set <tool> <session> <true/false> to modify")

                    await ctx.send(embed=embed)

                elif action == "sessions":
                    # List all known sessions/users
                    await ctx.send("🔍 Gathering session information...")

                    sessions_info = await self.list_all_known_sessions()

                    if not sessions_info:
                        await ctx.send("📋 No sessions found.")
                        return

                    embed = discord.Embed(
                        title="👥 Known Sessions/Users",
                        description=f"Total sessions: {len(sessions_info)}",
                        color=discord.Color.blue(),
                        timestamp=datetime.now(UTC)
                    )

                    # Group by source
                    sources = {
                        'active': [],
                        'memory': [],
                        'learning': [],
                        'context': []
                    }

                    for user_info in sessions_info:
                        user_id = user_info['user_id']
                        display = f"`{user_id}` - **{user_info['display_name']}** (@{user_info['username']})"

                        # Check which sources have this user
                        has_active = user_id in self.output_router.user_channels
                        has_memory = user_id in self.kernel.memory_store.user_memories
                        has_learning = user_id in self.kernel.learning_engine.preferences
                        has_context = (hasattr(self.kernel.agent, 'context_manager') and
                                       user_id in self.kernel.agent.context_manager.session_managers)

                        status_icons = []
                        if has_active: status_icons.append("💬")
                        if has_context: status_icons.append("🧠")
                        if has_memory: status_icons.append("💾")
                        if has_learning: status_icons.append("📚")

                        display += f" {' '.join(status_icons)}"

                        if has_active:
                            sources['active'].append(display)
                        elif has_context:
                            sources['context'].append(display)
                        elif has_memory:
                            sources['memory'].append(display)
                        else:
                            sources['learning'].append(display)

                    # Add fields for each source
                    if sources['active']:
                        text = "\n".join(sources['active'][:10])
                        if len(sources['active']) > 10:
                            text += f"\n... and {len(sources['active']) - 10} more"
                        embed.add_field(
                            name="💬 Active Sessions",
                            value=text,
                            inline=False
                        )

                    if sources['context']:
                        text = "\n".join(sources['context'][:10])
                        if len(sources['context']) > 10:
                            text += f"\n... and {len(sources['context']) - 10} more"
                        embed.add_field(
                            name="🧠 Context Sessions",
                            value=text,
                            inline=False
                        )

                    if sources['memory']:
                        text = "\n".join(sources['memory'][:10])
                        if len(sources['memory']) > 10:
                            text += f"\n... and {len(sources['memory']) - 10} more"
                        embed.add_field(
                            name="💾 Memory Only",
                            value=text,
                            inline=False
                        )

                    if sources['learning']:
                        text = "\n".join(sources['learning'][:10])
                        if len(sources['learning']) > 10:
                            text += f"\n... and {len(sources['learning']) - 10} more"
                        embed.add_field(
                            name="📚 Learning Only",
                            value=text,
                            inline=False
                        )

                    embed.set_footer(text="Icons: 💬 Active | 🧠 Context | 💾 Memory | 📚 Learning")

                    await ctx.send(embed=embed)

                elif action == "tools":
                    # List all available tools
                    if not hasattr(self.kernel.agent, '_tool_registry'):
                        await ctx.send("❌ No tools available!")
                        return

                    tools = self.kernel.agent._tool_registry

                    embed = discord.Embed(
                        title="🛠️ Available Tools",
                        description=f"Total tools: {len(tools)}",
                        color=discord.Color.green(),
                        timestamp=datetime.now(UTC)
                    )

                    # Group tools by category (if available)
                    categorized = {}
                    for tool in tools.keys():
                        category = tool.split('_')[0]
                        if category not in categorized:
                            categorized[category] = []

                        tool_name = tool
                        # Check if tool has any restrictions
                        has_restrictions = tool_name in self.kernel.agent.session_tool_restrictions
                        restriction_icon = "🔒" if has_restrictions else "🔓"

                        categorized[category].append(f"{restriction_icon} `{tool_name}`")

                    # Add fields for each category
                    for category, tool_list in sorted(categorized.items()):
                        tool_text = "\n".join(tool_list[:20])  # Max 20 per category
                        if len(tool_list) > 20:
                            tool_text += f"\n... and {len(tool_list) - 20} more"

                        embed.add_field(
                            name=f"📁 {category}",
                            value=tool_text,
                            inline=False
                        )

                    embed.set_footer(text="🔓 No restrictions | 🔒 Has restrictions")

                    await ctx.send(embed=embed)

                elif action == "set":
                    # Set restriction for specific session
                    if not args:
                        await ctx.send("❌ Usage: `!restrict set <tool> <session> <true/false>`")
                        return

                    parts = args.split()
                    if len(parts) < 3:
                        await ctx.send("❌ Usage: `!restrict set <tool> <session> <true/false>`")
                        return

                    tool_name = parts[0]
                    session_id = parts[1]
                    allowed_str = parts[2].lower()

                    # Parse allowed value
                    if allowed_str in ['true', 'yes', '1', 'allow', 'allowed']:
                        allowed = True
                    elif allowed_str in ['false', 'no', '0', 'deny', 'restrict', 'restricted']:
                        allowed = False
                    else:
                        await ctx.send(f"❌ Invalid value: `{allowed_str}`. Use true/false, yes/no, allow/deny")
                        return

                    # Check if tool exists
                    if hasattr(self.kernel.agent, 'agent_tools'):
                        tool_names = [t.name for t in self.kernel.agent.agent_tools]
                        if tool_name not in tool_names:
                            await ctx.send(
                                f"⚠️ Warning: Tool `{tool_name}` not found in available tools. Setting restriction anyway.")

                    # Set restriction
                    self.kernel.agent.set_tool_restriction(tool_name, session_id, allowed)

                    # Get user display name
                    try:
                        user = self.bot.get_user(int(session_id))
                        display_name = f"{user.display_name} ({session_id})" if user else session_id
                    except:
                        display_name = session_id

                    status_text = "✅ Allowed" if allowed else "❌ Restricted"

                    embed = discord.Embed(
                        title="✅ Restriction Set",
                        description=f"Tool restriction updated successfully",
                        color=discord.Color.green() if allowed else discord.Color.red(),
                        timestamp=datetime.now(UTC)
                    )

                    embed.add_field(name="Tool", value=f"`{tool_name}`", inline=True)
                    embed.add_field(name="Session", value=display_name, inline=True)
                    embed.add_field(name="Status", value=status_text, inline=True)

                    await ctx.send(embed=embed)

                elif action == "default":
                    # Set default restriction for tool
                    if not args:
                        await ctx.send("❌ Usage: `!restrict default <tool> <true/false>`")
                        return

                    parts = args.split()
                    if len(parts) < 2:
                        await ctx.send("❌ Usage: `!restrict default <tool> <true/false>`")
                        return

                    tool_name = parts[0]
                    allowed_str = parts[1].lower()

                    # Parse allowed value
                    if allowed_str in ['true', 'yes', '1', 'allow', 'allowed']:
                        allowed = True
                    elif allowed_str in ['false', 'no', '0', 'deny', 'restrict', 'restricted']:
                        allowed = False
                    else:
                        await ctx.send(f"❌ Invalid value: `{allowed_str}`. Use true/false, yes/no, allow/deny")
                        return

                    # Set default restriction
                    self.kernel.agent.set_tool_restriction(tool_name, '*', allowed)

                    status_text = "✅ Allowed by default" if allowed else "❌ Restricted by default"

                    embed = discord.Embed(
                        title="✅ Default Restriction Set",
                        description=f"Default restriction for `{tool_name}` updated",
                        color=discord.Color.green() if allowed else discord.Color.red(),
                        timestamp=datetime.now(UTC)
                    )

                    embed.add_field(name="Tool", value=f"`{tool_name}`", inline=True)
                    embed.add_field(name="Default Status", value=status_text, inline=True)

                    embed.set_footer(text="This applies to all sessions unless overridden")

                    await ctx.send(embed=embed)

                elif action == "reset":
                    # Reset restrictions
                    tool_name = args.strip() if args else None

                    # Confirmation
                    if tool_name:
                        confirm_text = f"reset restrictions for tool `{tool_name}`"
                    else:
                        confirm_text = "reset **ALL** tool restrictions"

                    embed = discord.Embed(
                        title="⚠️ Confirm Reset",
                        description=f"Are you sure you want to {confirm_text}?",
                        color=discord.Color.orange(),
                        timestamp=datetime.now(UTC)
                    )

                    embed.set_footer(text="React with ✅ to confirm or ❌ to cancel (30s timeout)")

                    msg = await ctx.send(embed=embed)
                    await msg.add_reaction("✅")
                    await msg.add_reaction("❌")

                    # Wait for reaction
                    def check(reaction, user):
                        return user == ctx.author and str(reaction.emoji) in ["✅",
                                                                              "❌"] and reaction.message.id == msg.id

                    try:
                        reaction, user = await self.bot.wait_for('reaction_add', timeout=30.0, check=check)

                        if str(reaction.emoji) == "❌":
                            await msg.edit(embed=discord.Embed(
                                title="❌ Reset Cancelled",
                                color=discord.Color.red()
                            ))
                            await msg.clear_reactions()
                            return

                        # Perform reset
                        self.kernel.agent.reset_tool_restrictions(tool_name)

                        embed = discord.Embed(
                            title="✅ Restrictions Reset",
                            description=f"Successfully reset {confirm_text}",
                            color=discord.Color.green(),
                            timestamp=datetime.now(UTC)
                        )

                        await msg.edit(embed=embed)
                        await msg.clear_reactions()

                    except asyncio.TimeoutError:
                        await msg.edit(embed=discord.Embed(
                            title="⏱️ Reset Timeout",
                            description="Confirmation timed out. Reset cancelled.",
                            color=discord.Color.red()
                        ))
                        await msg.clear_reactions()

                elif action == "check":
                    # Check if tool is allowed in session
                    if not args:
                        await ctx.send("❌ Usage: `!restrict check <tool> <session>`")
                        return

                    parts = args.split()
                    if len(parts) < 2:
                        await ctx.send("❌ Usage: `!restrict check <tool> <session>`")
                        return

                    tool_name = parts[0]
                    session_id = parts[1]

                    # Check restriction
                    is_allowed = self.kernel.agent.get_tool_restriction(tool_name, session_id)

                    # Get user display name
                    try:
                        user = self.bot.get_user(int(session_id))
                        display_name = f"{user.display_name} ({session_id})" if user else session_id
                    except:
                        display_name = session_id

                    # Check what rule applies
                    restrictions = self.kernel.agent.session_tool_restrictions.get(tool_name, {})

                    if session_id in restrictions:
                        rule = f"Specific session rule: `{session_id}`"
                    elif '*' in restrictions:
                        rule = "Default rule: `*`"
                    else:
                        rule = "No restrictions (allowed by default)"

                    status_text = "✅ Allowed" if is_allowed else "❌ Restricted"
                    color = discord.Color.green() if is_allowed else discord.Color.red()

                    embed = discord.Embed(
                        title="🔍 Restriction Check",
                        description=f"Checking tool access for session",
                        color=color,
                        timestamp=datetime.now(UTC)
                    )

                    embed.add_field(name="Tool", value=f"`{tool_name}`", inline=True)
                    embed.add_field(name="Session", value=display_name, inline=True)
                    embed.add_field(name="Status", value=status_text, inline=True)
                    embed.add_field(name="Applied Rule", value=rule, inline=False)

                    await ctx.send(embed=embed)

                else:
                    await ctx.send(
                        f"❌ Unknown action: `{action}`\n\nValid actions: list, sessions, tools, set, default, reset, check")

            except Exception as e:
                await ctx.send(f"❌ Error managing restrictions: {e}")
                import traceback
                traceback.print_exc()

        # Variables management command
        @self.bot.command(name="vars")
        async def vars_command(ctx: commands.Context, action: str = None, *, args: str = None):
            """
            Interactive variable explorer and manager (Admin only).

            Usage:
                !vars                 - Open interactive explorer
                !vars explore [path]  - Explore specific path
                !vars get <path>      - Get value at path
                !vars set <path> <value> - Set value at path
                !vars delete <path>   - Delete value at path
                !vars search <query>  - Search for variables

            Examples:
                !vars
                !vars explore discord
                !vars get discord.output_mode.123456789
                !vars set user.theme dark
                !vars delete temp.cache
                !vars search user
            """
            if not await self._check_admin_permission(ctx):
                return

            if not hasattr(self.kernel.agent, 'variable_manager'):
                await ctx.send("❌ Variable manager not available!")
                return

            var_manager = self.kernel.agent.variable_manager
            user_id = str(ctx.author.id)

            # Default action is explore
            if action is None:
                action = "explore"

            action = action.lower()

            try:
                if action == "explore":
                    # Open interactive explorer
                    start_path = args.strip() if args else ""

                    view = VariableExplorerView(var_manager, user_id, start_path)
                    embed = view.create_embed()

                    await ctx.send(embed=embed, view=view)

                elif action == "get":
                    if not args:
                        await ctx.send("❌ Usage: `!vars get <path>`")
                        return

                    path = args.strip()
                    value = var_manager.get(path)

                    if value is None:
                        await ctx.send(f"❌ Variable not found: `{path}`")
                        return

                    # Format value
                    try:
                        if isinstance(value, (dict, list)):
                            formatted = json.dumps(value, indent=2, default=str)
                        else:
                            formatted = str(value)
                    except:
                        formatted = str(value)

                    # Split if too long
                    if len(formatted) > 1900:
                        # Send as file
                        import io
                        file_content = io.BytesIO(formatted.encode('utf-8'))
                        file = discord.File(file_content, filename=f"{path.replace('.', '_')}.json")

                        await ctx.send(f"📄 Value at `{path}` (sent as file):", file=file)
                    else:
                        embed = discord.Embed(
                            title=f"🔍 Variable: {path}",
                            description=f"```json\n{formatted}\n```",
                            color=discord.Color.blue()
                        )
                        await ctx.send(embed=embed)

                elif action == "set":
                    if not args or ' ' not in args:
                        await ctx.send("❌ Usage: `!vars set <path> <value>`")
                        return

                    # Split path and value
                    parts = args.split(' ', 1)
                    path = parts[0].strip()
                    value_str = parts[1].strip()

                    # Try to parse value as JSON
                    try:
                        value = json.loads(value_str)
                    except:
                        value = value_str

                    var_manager.set(path, value)

                    embed = discord.Embed(
                        title="✅ Variable Set",
                        description=f"**Path:** `{path}`\n**Value:** `{value}`",
                        color=discord.Color.green()
                    )
                    await ctx.send(embed=embed)

                elif action == "delete":
                    if not args:
                        await ctx.send("❌ Usage: `!vars delete <path>`")
                        return

                    path = args.strip()

                    if var_manager.get(path) is None:
                        await ctx.send(f"❌ Variable not found: `{path}`")
                        return

                    # Delete variable
                    if hasattr(var_manager, 'delete'):
                        var_manager.delete(path)
                    else:
                        var_manager.set(path, None)

                    embed = discord.Embed(
                        title="✅ Variable Deleted",
                        description=f"**Path:** `{path}`",
                        color=discord.Color.orange()
                    )
                    await ctx.send(embed=embed)

                elif action == "search":
                    if not args:
                        await ctx.send("❌ Usage: `!vars search <query>`")
                        return

                    query = args.strip().lower()
                    results = []

                    # Search through all scopes
                    def search_recursive(data, path_prefix=""):
                        if isinstance(data, dict):
                            for key, value in data.items():
                                current_path = f"{path_prefix}.{key}" if path_prefix else key

                                # Check if key matches
                                if query in key.lower():
                                    results.append((current_path, value))

                                # Check if value matches (for strings)
                                if isinstance(value, str) and query in value.lower():
                                    results.append((current_path, value))

                                # Recurse
                                if isinstance(value, (dict, list)):
                                    search_recursive(value, current_path)

                        elif isinstance(data, list):
                            for i, item in enumerate(data):
                                current_path = f"{path_prefix}.{i}"

                                if isinstance(item, str) and query in item.lower():
                                    results.append((current_path, item))

                                if isinstance(item, (dict, list)):
                                    search_recursive(item, current_path)

                    # Search all scopes
                    for scope_name, scope_data in var_manager.scopes.items():
                        search_recursive(scope_data, scope_name)

                    if not results:
                        await ctx.send(f"🔍 No results found for: `{query}`")
                        return

                    # Create result embed
                    embed = discord.Embed(
                        title=f"🔍 Search Results: {query}",
                        description=f"Found {len(results)} match(es)",
                        color=discord.Color.blue()
                    )

                    # Add results (limit to first 10)
                    result_text = ""
                    for path, value in results[:10]:
                        preview = str(value)[:100]
                        if len(str(value)) > 100:
                            preview += "..."
                        result_text += f"📍 `{path}`\n  └─ {preview}\n\n"

                    if len(results) > 10:
                        result_text += f"... and {len(results) - 10} more results"

                    embed.add_field(name="Results", value=result_text, inline=False)

                    await ctx.send(embed=embed)

                else:
                    await ctx.send(
                        f"❌ Unknown action: `{action}`\n\nValid actions: explore, get, set, delete, search")

            except Exception as e:
                await ctx.send(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()

        @self.bot.command(name="varsreset")
        async def vars_reset_command(ctx: commands.Context, scope: str = None):
            """
            Reset variables - clear specific scope or all variables (Admin only).

            Usage:
                !varsreset              - Reset ALL variables (requires confirmation)
                !varsreset <scope>      - Reset specific scope (requires confirmation)
                !varsreset <scope> --force - Reset without confirmation

            Examples:
                !varsreset
                !varsreset shared
                !varsreset results --force
            """
            # Check admin permission
            if not await self._check_admin_permission(ctx):
                return

            if not hasattr(self.kernel.agent, 'variable_manager'):
                await ctx.send("❌ Variable manager not available!")
                return

            var_manager = self.kernel.agent.variable_manager
            user_id = str(ctx.author.id)

            # Check for --force flag
            force = False
            if scope and scope.endswith("--force"):
                force = True
                scope = scope.replace("--force", "").strip()

            try:
                # Determine what to reset
                if scope is None:
                    # Reset ALL variables
                    target = "ALL VARIABLES"
                    scopes_to_reset = list(var_manager.scopes.keys())
                elif scope in var_manager.scopes:
                    # Reset specific scope
                    target = f"scope '{scope}'"
                    scopes_to_reset = [scope]
                else:
                    await ctx.send(
                        f"❌ Scope not found: `{scope}`\n\nAvailable scopes: {', '.join(var_manager.scopes.keys())}")
                    return

                # Confirmation prompt if not forced
                if not force:
                    embed = discord.Embed(
                        title="⚠️ Confirm Reset",
                        description=f"Are you sure you want to reset **{target}**?\n\nThis action cannot be undone!",
                        color=discord.Color.orange(),
                        timestamp=datetime.now(UTC)
                    )

                    # Show what will be affected
                    affected_info = []
                    for scope_name in scopes_to_reset:
                        scope_data = var_manager.scopes[scope_name]
                        if isinstance(scope_data, dict):
                            count = len(scope_data)
                        elif isinstance(scope_data, list):
                            count = len(scope_data)
                        else:
                            count = 1
                        affected_info.append(f"📁 **{scope_name}**: {count} items")

                    embed.add_field(
                        name="Affected Scopes",
                        value="\n".join(affected_info),
                        inline=False
                    )

                    embed.set_footer(text="React with ✅ to confirm or ❌ to cancel (60s timeout)")

                    msg = await ctx.send(embed=embed)

                    # Add reactions
                    await msg.add_reaction("✅")
                    await msg.add_reaction("❌")

                    # Wait for reaction
                    def check(reaction, user):
                        return user == ctx.author and str(reaction.emoji) in ["✅",
                                                                              "❌"] and reaction.message.id == msg.id

                    try:
                        reaction, user = await self.bot.wait_for('reaction_add', timeout=60.0, check=check)

                        if str(reaction.emoji) == "❌":
                            await msg.edit(embed=discord.Embed(
                                title="❌ Reset Cancelled",
                                description=f"Reset of {target} was cancelled.",
                                color=discord.Color.red()
                            ))
                            await msg.clear_reactions()
                            return

                    except asyncio.TimeoutError:
                        await msg.edit(embed=discord.Embed(
                            title="⏱️ Reset Timeout",
                            description="Confirmation timed out. Reset cancelled.",
                            color=discord.Color.red()
                        ))
                        await msg.clear_reactions()
                        return

                # Perform the reset
                reset_stats = {
                    'scopes_reset': 0,
                    'items_cleared': 0,
                    'backup_created': False
                }

                # Create backup before reset
                backup = {}
                for scope_name in scopes_to_reset:
                    backup[scope_name] = var_manager.scopes[scope_name]

                # Store backup in session_archive
                backup_key = f"reset_backup_{datetime.now().isoformat()}"
                if 'session_archive' in var_manager.scopes:
                    var_manager.scopes['session_archive'][backup_key] = {
                        'type': 'reset_backup',
                        'timestamp': datetime.now().isoformat(),
                        'user_id': user_id,
                        'scopes': backup
                    }
                    reset_stats['backup_created'] = True

                # Reset the scopes
                for scope_name in scopes_to_reset:
                    scope_data = var_manager.scopes[scope_name]

                    # Count items before clearing
                    if isinstance(scope_data, dict):
                        reset_stats['items_cleared'] += len(scope_data)
                        var_manager.scopes[scope_name] = {}
                    elif isinstance(scope_data, list):
                        reset_stats['items_cleared'] += len(scope_data)
                        var_manager.scopes[scope_name] = []
                    else:
                        reset_stats['items_cleared'] += 1
                        var_manager.scopes[scope_name] = None

                    reset_stats['scopes_reset'] += 1

                # Clear cache
                if hasattr(var_manager, '_cache'):
                    var_manager._cache.clear()

                # Success embed
                embed = discord.Embed(
                    title="✅ Variables Reset",
                    description=f"Successfully reset {target}",
                    color=discord.Color.green(),
                    timestamp=datetime.now(UTC)
                )

                embed.add_field(
                    name="Statistics",
                    value=f"🗑️ Scopes reset: {reset_stats['scopes_reset']}\n"
                          f"📦 Items cleared: {reset_stats['items_cleared']}\n"
                          f"💾 Backup created: {'Yes' if reset_stats['backup_created'] else 'No'}",
                    inline=False
                )

                if reset_stats['backup_created']:
                    embed.add_field(
                        name="Backup Info",
                        value=f"A backup was created: `{backup_key}`\n"
                              f"You can restore it using: `!varsrestore {backup_key}`",
                        inline=False
                    )

                if force:
                    await ctx.send(embed=embed)
                else:
                    await msg.edit(embed=embed)
                    try:
                        await msg.clear_reactions()
                    except Exception as e:
                        print(f"❌ Error clearing reactions: {e}")
                        try:
                            await msg.delete()
                        except Exception as e:
                            print(f"❌ Error clearing reactions: {e}")

            except Exception as e:
                await ctx.send(f"❌ Error resetting variables: {e}")
                import traceback
                traceback.print_exc()

        # Whitelist management command
        @self.bot.command(name="whitelist")
        async def whitelist_command(ctx: commands.Context, action: str = None, *, user: str = None):
            """
            Manage admin whitelist (Admin only).

            Usage:
                !whitelist              - List all whitelisted users
                !whitelist add <user>   - Add user to whitelist (username or ID)
                !whitelist remove <user> - Remove user from whitelist

            Examples:
                !whitelist
                !whitelist add Kinr3
                !whitelist add 268830485889810432
                !whitelist remove SomeUser
            """
            # Check admin permission
            if not await self._check_admin_permission(ctx):
                return

            try:
                # List action (default)
                if action is None or action.lower() == "list":
                    embed = discord.Embed(
                        title="🔒 Admin Whitelist",
                        description="Users with admin access to restricted commands",
                        color=discord.Color.blue(),
                        timestamp=datetime.now(UTC)
                    )

                    if self.admin_whitelist:
                        whitelist_text = "\n".join([f"• `{user}`" for user in sorted(self.admin_whitelist)])
                        embed.add_field(
                            name=f"Whitelisted Users ({len(self.admin_whitelist)})",
                            value=whitelist_text,
                            inline=False
                        )
                    else:
                        embed.add_field(
                            name="Whitelisted Users",
                            value="*No users in whitelist*",
                            inline=False
                        )

                    embed.add_field(
                        name="ℹ️ Note",
                        value="Bot owner always has admin access, even if not in whitelist.",
                        inline=False
                    )

                    await ctx.send(embed=embed)

                # Add action
                elif action.lower() == "add":
                    if not user:
                        await ctx.send("❌ Please specify a user to add.\n\nUsage: `!whitelist add <username or ID>`")
                        return

                    # Normalize to lowercase for case-insensitive comparison
                    user_normalized = user.lower()

                    if user_normalized in self.admin_whitelist:
                        await ctx.send(f"⚠️ User `{user}` is already in the whitelist.")
                        return

                    self.admin_whitelist.add(user_normalized)
                    print(f"🔒 [SECURITY] Added {user} to admin whitelist by {ctx.author.name}")

                    embed = discord.Embed(
                        title="✅ User Added to Whitelist",
                        description=f"**User:** `{user}`\n\nThis user now has admin access.",
                        color=discord.Color.green(),
                        timestamp=datetime.now(UTC)
                    )

                    await ctx.send(embed=embed)

                # Remove action
                elif action.lower() == "remove":
                    if not user:
                        await ctx.send("❌ Please specify a user to remove.\n\nUsage: `!whitelist remove <username or ID>`")
                        return

                    # Normalize to lowercase
                    user_normalized = user.lower()

                    if user_normalized not in self.admin_whitelist:
                        await ctx.send(f"⚠️ User `{user}` is not in the whitelist.")
                        return

                    self.admin_whitelist.remove(user_normalized)
                    print(f"🔒 [SECURITY] Removed {user} from admin whitelist by {ctx.author.name}")

                    embed = discord.Embed(
                        title="✅ User Removed from Whitelist",
                        description=f"**User:** `{user}`\n\nThis user no longer has admin access.",
                        color=discord.Color.orange(),
                        timestamp=datetime.now(UTC)
                    )

                    await ctx.send(embed=embed)

                else:
                    await ctx.send(f"❌ Unknown action: `{action}`\n\nValid actions: list, add, remove")

            except Exception as e:
                await ctx.send(f"❌ Error managing whitelist: {e}")
                import traceback
                traceback.print_exc()

    async def _dispatch_progress_event(self, event: ProgressEvent):
        """Dispatch progress events to all enabled progress printers"""
        # Send event to all enabled printers
        for user_id, printer in self.progress_printers.items():
            if printer.enabled:
                try:
                    await printer.progress_callback(event)
                except Exception as e:
                    print(f"⚠️ Error dispatching progress event to user {user_id}: {e}")

    async def _auto_save_loop(self):
        """Auto-save kernel state periodically"""
        while self.running:
            await asyncio.sleep(self.auto_save_interval)
            if self.running:
                await self.kernel.save_to_file(str(self.save_path))
                print(f"💾 Auto-saved Discord kernel at {datetime.now().strftime('%H:%M:%S')}")

    def _inject_discord_context_to_agent(self):
        """
        Inject Discord-specific context awareness into agent's system prompt

        This makes the agent aware of:
        - Its Discord environment and capabilities
        - Voice status and multi-instance awareness
        - Available Discord tools and commands
        """
        try:
            discord_context_prompt = """

# ========== DISCORD CONTEXT AWARENESS ==========

## Your Discord Environment

You are operating in a Discord environment with full context awareness. You have access to detailed information about your current location and status through the variable system.

### Current Context Variables

You can access the following context information:
- `discord.current_context.{user_id}` - Full context for the current conversation
- `discord.location` - Simplified location info (type, name, guild, voice status)

### Context Information Available

**Location Context:**
- Channel type (DM, Guild Text Channel, Thread)
- Channel name and ID
- Guild name and ID (if in a server)
- Guild member count

**Voice Context:**
- Are you in a voice channel? (bot_voice_status.connected)
- Which voice channel? (bot_voice_status.channel_name)
- Are you listening to voice input? (bot_voice_status.listening)
- Is TTS enabled? (bot_voice_status.tts_enabled, tts_mode)
- Who else is in the voice channel? (bot_voice_status.users_in_channel)

**User Voice Context:**
- Is the user in a voice channel? (user_voice_status.in_voice)
- Are you in the same voice channel as the user? (user_voice_status.same_channel_as_bot)

**Multi-Instance Awareness:**
- Total active conversations (active_conversations.total_active_channels)
- Total active users (active_conversations.total_active_users)
- Voice connections (active_conversations.voice_connections)
- Is this a DM? (active_conversations.this_is_dm)

**Capabilities:**
- Can manage messages, roles, channels (bot_capabilities)
- Can join voice, transcribe, use TTS (bot_capabilities)
- 21 Discord tools available (bot_capabilities.has_discord_tools)

### Important Context Rules

1. **Location Awareness**: Always know where you are (DM vs Server, Voice vs Text)
2. **Voice Awareness**: Know if you're in voice and with whom
3. **Multi-Instance**: You may have multiple text conversations but only ONE voice connection
4. **User Awareness**: Know if the user is in voice and if you're together
5. **Capability Awareness**: Know what you can do in the current context

### Example Context Usage

When responding, consider:
- "I'm currently in voice with you in {channel_name}" (if in same voice channel)
- "I see you're in {voice_channel}, would you like me to join?" (if user in voice, you're not)
- "I'm already in a voice channel in {guild_name}, I can only be in one voice channel at a time" (multi-instance awareness)
- "I'm in a DM with you, so I have limited server management capabilities" (capability awareness)

### Discord Tools Available

You have 21 Discord-specific tools for:
- **Server Management**: Get server/channel/user info, list channels
- **Message Management**: Send, edit, delete, react to messages, pin/unpin
- **Voice Control**: Join, leave, get status, toggle TTS
- **Role Management**: Get roles, add/remove roles
- **Lifetime Management**: Get bot status, kernel metrics

Use these tools to interact with Discord based on your current context!

# ========== END DISCORD CONTEXT ==========
"""

            if hasattr(self.kernel.agent, 'amd'):
                current_prompt = self.kernel.agent.amd.system_message or ""

                # Check if already injected
                if "DISCORD CONTEXT AWARENESS" not in current_prompt:
                    self.kernel.agent.amd.system_message = current_prompt + "\n" + discord_context_prompt
                    print("✓ Discord context awareness injected into agent system prompt")
                else:
                    # Update existing section
                    parts = current_prompt.split("# ========== DISCORD CONTEXT AWARENESS ==========")
                    if len(parts) >= 2:
                        # Keep everything before the Discord context section
                        self.kernel.agent.amd.system_message = parts[0] + discord_context_prompt
                        print("✓ Discord context awareness updated in agent system prompt")
            else:
                print("⚠️  Agent does not have AMD - cannot inject Discord context")

        except Exception as e:
            print(f"❌ Failed to inject Discord context to agent: {e}")

    async def start(self):
        """Start the Discord kernel"""
        self.running = True

        # Load previous state if exists
        if self.save_path.exists():
            print("📂 Loading previous Discord session...")
            await self.kernel.load_from_file(str(self.save_path))

        # Start kernel
        await self.kernel.start()

        # Inject kernel prompt to agent
        self.kernel.inject_kernel_prompt_to_agent()

        # Inject Discord-specific context awareness
        self._inject_discord_context_to_agent()

        # Export Discord-specific tools to agent
        print("🔧 Exporting Discord tools to agent...")
        await self.discord_tools.export_to_agent()

        # Start auto-save loop
        asyncio.create_task(self._auto_save_loop())

        # Start Discord bot
        asyncio.create_task(self.bot.start(self.bot_token))

        print(f"✓ Discord Kernel started (instance: {self.instance_id})")

    async def stop(self):
        """Stop the Discord kernel"""
        if not self.running:
            return

        self.running = False
        print("💾 Saving Discord session...")

        # Save final state
        await self.kernel.save_to_file(str(self.save_path))

        # Stop kernel
        await self.kernel.stop()

        # Stop Discord bot
        await self.bot.close()

        print("✓ Discord Kernel stopped")

    def _get_discord_context(self, message: discord.Message) -> dict:
        """
        Gather comprehensive Discord context for the agent

        Returns detailed information about:
        - Current location (guild, channel, DM)
        - User information
        - Voice status (is bot in voice? is user in voice?)
        - Active conversations
        - Bot capabilities in this context
        """
        user_id = str(message.author.id)
        channel_id = message.channel.id

        # Basic context
        context = {
            "user_id": user_id,
            "user_name": str(message.author),
            "user_display_name": message.author.display_name,
            "channel_id": channel_id,
            "message_id": message.id,
        }

        # Channel type and location
        if isinstance(message.channel, discord.DMChannel):
            context["channel_type"] = "DM"
            context["channel_name"] = f"DM with {message.author.display_name}"
            context["guild_id"] = None
            context["guild_name"] = None
        elif isinstance(message.channel, discord.TextChannel):
            context["channel_type"] = "Guild Text Channel"
            context["channel_name"] = message.channel.name
            context["guild_id"] = message.guild.id
            context["guild_name"] = message.guild.name
            context["guild_member_count"] = message.guild.member_count
        elif isinstance(message.channel, discord.Thread):
            context["channel_type"] = "Thread"
            context["channel_name"] = message.channel.name
            context["parent_channel_name"] = message.channel.parent.name if message.channel.parent else None
            context["guild_id"] = message.guild.id
            context["guild_name"] = message.guild.name
        else:
            context["channel_type"] = "Unknown"
            context["channel_name"] = getattr(message.channel, 'name', 'Unknown')
            context["guild_id"] = message.guild.id if message.guild else None
            context["guild_name"] = message.guild.name if message.guild else None

        # Voice status - Is the bot in a voice channel?
        context["bot_voice_status"] = {
            "connected": False,
            "channel_id": None,
            "channel_name": None,
            "listening": False,
            "tts_enabled": False,
            "tts_mode": None,
            "users_in_channel": []
        }

        if message.guild:
            # Check if bot is in voice in this guild
            voice_client = message.guild.voice_client
            if voice_client and voice_client.is_connected():
                context["bot_voice_status"]["connected"] = True
                context["bot_voice_status"]["channel_id"] = voice_client.channel.id
                context["bot_voice_status"]["channel_name"] = voice_client.channel.name
                context["bot_voice_status"]["listening"] = voice_client.is_listening() if hasattr(voice_client, 'is_listening') else False

                # TTS status
                guild_id = message.guild.id
                context["bot_voice_status"]["tts_enabled"] = self.output_router.tts_enabled.get(guild_id, False)
                context["bot_voice_status"]["tts_mode"] = self.output_router.tts_mode.get(guild_id, "piper")

                # Users in voice channel
                context["bot_voice_status"]["users_in_channel"] = [
                    {
                        "id": str(member.id),
                        "name": member.display_name,
                        "is_self": member.id == message.author.id
                    }
                    for member in voice_client.channel.members
                    if not member.bot
                ]
        elif isinstance(message.channel, discord.DMChannel):
            # Check if bot is in DM voice channel
            voice_client = self.output_router.voice_clients.get(message.author.id)
            if voice_client and voice_client.is_connected():
                context["bot_voice_status"]["connected"] = True
                context["bot_voice_status"]["channel_id"] = voice_client.channel.id
                context["bot_voice_status"]["channel_name"] = "DM Voice Channel"
                context["bot_voice_status"]["listening"] = voice_client.is_listening() if hasattr(voice_client, 'is_listening') else False
                context["bot_voice_status"]["tts_enabled"] = self.output_router.tts_enabled.get(message.author.id, False)
                context["bot_voice_status"]["tts_mode"] = self.output_router.tts_mode.get(message.author.id, "piper")

        # User voice status - Is the user in a voice channel?
        context["user_voice_status"] = {
            "in_voice": False,
            "channel_id": None,
            "channel_name": None,
            "same_channel_as_bot": False
        }

        if hasattr(message.author, 'voice') and message.author.voice and message.author.voice.channel:
            context["user_voice_status"]["in_voice"] = True
            context["user_voice_status"]["channel_id"] = message.author.voice.channel.id
            context["user_voice_status"]["channel_name"] = getattr(message.author.voice.channel, 'name', 'Voice Channel')

            # Check if user is in same voice channel as bot
            if context["bot_voice_status"]["connected"]:
                context["user_voice_status"]["same_channel_as_bot"] = (
                    message.author.voice.channel.id == context["bot_voice_status"]["channel_id"]
                )
        else:
            context["user_voice_status"]["in_voice"] = False

        # Active conversations - Track multi-instance awareness
        context["active_conversations"] = {
            "total_active_channels": len(self.output_router.active_channels),
            "total_active_users": len(self.output_router.user_channels),
            "voice_connections": len(self.bot.voice_clients),
            "this_is_dm": isinstance(message.channel, discord.DMChannel)
        }

        # Bot capabilities in this context
        context["bot_capabilities"] = {
            "can_manage_messages": message.channel.permissions_for(message.guild.me).manage_messages if message.guild else False,
            "can_manage_roles": message.channel.permissions_for(message.guild.me).manage_roles if message.guild else False,
            "can_manage_channels": message.channel.permissions_for(message.guild.me).manage_channels if message.guild else False,
            "can_join_voice": VOICE_SUPPORT,
            "can_transcribe_voice": VOICE_RECEIVE_SUPPORT and GROQ_SUPPORT,
            "can_use_tts": VOICE_SUPPORT and (ELEVENLABS_SUPPORT or PIPER_SUPPORT),
            "has_discord_tools": True,  # 21 Discord tools available
        }

        return context

    async def handle_message(self, message: discord.Message):
        """Handle incoming Discord message with full context awareness"""
        try:
            user_id = str(message.author.id)
            channel_id = message.channel.id

            # Register user channel (store channel object directly for this user)
            self.output_router.user_channels[user_id] = message.channel
            self.output_router.active_channels[channel_id] = message.channel

            # Gather comprehensive Discord context
            discord_context = self._get_discord_context(message)

            # Inject context into agent's variable system
            if hasattr(self.kernel.agent, 'variable_manager'):
                self.kernel.agent.variable_manager.set(
                    f'discord.current_context.{user_id}',
                    discord_context
                )

                # Also set a simplified version for easy access
                self.kernel.agent.variable_manager.set(
                    'discord.location',
                    {
                        "type": discord_context["channel_type"],
                        "name": discord_context["channel_name"],
                        "guild": discord_context.get("guild_name"),
                        "in_voice": discord_context["bot_voice_status"]["connected"],
                        "voice_channel": discord_context["bot_voice_status"]["channel_name"]
                    }
                )

            # Extract content
            content = message.content

            # Remove bot mention from content
            if self.bot.user in message.mentions:
                content = content.replace(f"<@{self.bot.user.id}>", "").strip()

            # Handle attachments - add them as [media:url] to content
            attachments_info = []
            if message.attachments:
                media_links = []
                for attachment in message.attachments:
                    attachments_info.append({
                        "filename": attachment.filename,
                        "url": attachment.url,
                        "content_type": attachment.content_type
                    })
                    # Add media link to content
                    media_type = "image" if attachment.content_type and attachment.content_type.startswith("image") else "file"
                    media_links.append(f"[{media_type}:{attachment.url}]")

                # Append media links to content
                if media_links:
                    if content:
                        content += "\n\n" + "\n".join(media_links)
                    else:
                        content = "\n".join(media_links)

            # Send typing indicator
            async with message.channel.typing():
                # Send signal to kernel with enhanced metadata
                signal = KernelSignal(
                    type=SignalType.USER_INPUT,
                    id=user_id,
                    content=content,
                    metadata={
                        "interface": "discord",
                        "channel_id": channel_id,
                        "message_id": message.id,
                        "attachments": attachments_info,
                        "guild_id": message.guild.id if message.guild else None,
                        "user_name": str(message.author),
                        "user_display_name": message.author.display_name,
                        # Enhanced context
                        "discord_context": discord_context
                    }
                )
                await self.kernel.process_signal(signal)

        except Exception as e:
            print(f"❌ Error handling Discord message from {message.author}: {e}")

    # Methode 1: Über user_channels (alle User, die Nachrichten gesendet haben)
    async def list_all_users_with_nicknames(self):
        """Liste alle bekannten User mit ihren Nicknames auf"""
        users_info = []

        # Durchlaufe alle user_ids in user_channels
        for user_id in self.output_router.user_channels.keys():
            # Hole das Discord User Objekt
            user = self.bot.get_user(int(user_id))

            if user:
                users_info.append({
                    'user_id': user_id,
                    'username': user.name,
                    'display_name': user.display_name,
                    'discriminator': user.discriminator if hasattr(user, 'discriminator') else None
                })
            else:
                # Falls User nicht im Cache ist, versuche ihn zu fetchen
                try:
                    user = await self.bot.fetch_user(int(user_id))
                    users_info.append({
                        'user_id': user_id,
                        'username': user.name,
                        'display_name': user.display_name,
                        'discriminator': user.discriminator if hasattr(user, 'discriminator') else None
                    })
                except:
                    users_info.append({
                        'user_id': user_id,
                        'username': 'Unknown',
                        'display_name': 'Unknown'
                    })

        return users_info

    # Methode 2: Kombiniere mehrere Quellen für vollständige Liste
    async def list_all_known_sessions(self):
        """Liste alle bekannten Sessions aus verschiedenen Quellen"""
        all_user_ids = set()

        # User aus user_channels
        all_user_ids.update(self.output_router.user_channels.keys())

        # User aus session_managers
        if hasattr(self.kernel.agent, 'context_manager'):
            all_user_ids.update(self.kernel.agent.context_manager.session_managers.keys())

        # User aus memory_store
        all_user_ids.update(self.kernel.memory_store.user_memories.keys())

        # User aus learning_engine
        all_user_ids.update(self.kernel.learning_engine.preferences.keys())

        # Hole Nicknames für alle User
        users_info = []
        for user_id in all_user_ids:
            try:
                user_id = int(user_id)
            except:
                users_info.append({
                    'user_id': user_id,
                    'display_name': 'Unknown',
                    'username': 'Unknown'
                })

            user = self.bot.get_user(user_id)
            if not user:
                try:
                    user = await self.bot.fetch_user(user_id)
                except:
                    users_info.append({
                        'user_id': user_id,
                        'display_name': 'Unknown',
                        'username': 'Unknown'
                    })
            if not user:
                users_info.append({
                    'user_id': user_id,
                    'display_name': 'Unknown',
                    'username': 'Unknown'
                })
            else:
                users_info.append({
                    'user_id': user_id,
                    'display_name': user.display_name,
                    'username': user.name
                })


        return users_info


# ===== MODULE REGISTRATION =====

Name = "isaa.KernelDiscord"
version = "1.0.0"
app = get_app(Name)
export = app.tb

# Global kernel instance
_kernel_instance: Optional[DiscordKernel] = None


@export(mod_name=Name, version=version, initial=True)
async def init_kernel_discord(app: App):
    """Initialize the Discord Kernel module"""
    global _kernel_instance

    # Get Discord configuration from environment
    bot_token = os.getenv("DISCORD_BOT_TOKEN")

    if not bot_token:
        return {
            "success": False,
            "error": "Discord bot token not configured. Set DISCORD_BOT_TOKEN environment variable"
        }

    # Get ISAA and create agent
    isaa = app.get_mod("isaa")
    builder = isaa.get_agent_builder("DiscordKernelAssistant")
    builder.with_system_message(
        "You are a helpful Discord assistant. Provide clear, engaging responses. "
        "Use Discord formatting when appropriate (bold, italic, code blocks)."
    )
    # builder.with_models(
    #     fast_llm_model="openrouter/anthropic/claude-3-haiku",
    #     complex_llm_model="openrouter/openai/gpt-4o"
    # )

    await isaa.register_agent(builder)
    _ = await isaa.get_agent("self")
    agent = await isaa.get_agent("DiscordKernelAssistant")
    #agent.set_progress_callback(ProgressiveTreePrinter().progress_callback)
    # Create and start kernel
    _kernel_instance = DiscordKernel(agent, app, bot_token=bot_token)
    await _kernel_instance.start()

    return {"success": True, "info": "KernelDiscord initialized"}


@export(mod_name=Name, version=version)
async def stop_kernel_discord():
    """Stop the Discord kernel"""
    global _kernel_instance

    if _kernel_instance:
        await _kernel_instance.stop()
        _kernel_instance = None

    return {"success": True, "info": "KernelDiscord stopped"}


async def main():
    await init_kernel_discord(get_app())
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
