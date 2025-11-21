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
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional, Dict, List, Any

from toolboxv2.mods.isaa.extras.terminal_progress import ProgressiveTreePrinter

try:
    import discord
    from discord.ext import commands

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

        print(f"🎤 [DEBUG] WhisperAudioSink initialized successfully")
        print(f"🎤 [DEBUG] - Groq client: {'✅' if groq_client else '❌'}")
        print(f"🎤 [DEBUG] - Transcription interval: {self.transcription_interval}s")

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
                    signal = KernelSignal(
                        type=SignalType.USER_INPUT,
                        id=str(user.id),
                        content=text,
                        metadata={
                            "interface": "discord_voice",
                            "user_name": str(user),
                            "user_display_name": user.display_name,
                            "transcription": True,
                            "language": language,
                            "discord_context": discord_context,
                            "output_mode": output_mode,
                            "formatting_instructions": formatting_instructions
                        }
                    )
                    print(f"🎤 [DEBUG] Sending signal to kernel...")
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

        self.bot = commands.Bot(command_prefix=command_prefix, intents=intents)
        self.bot_token = bot_token

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
        async def help_command(ctx: commands.Context):
            """Exit the kernel"""

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
            """Reset user data (memories, context, preferences, scheduled tasks)"""
            user_id = str(ctx.author.id)

            embed = discord.Embed(
                title="🔄 Reset User Data",
                description="Choose what you want to reset. **Warning:** This action cannot be undone!",
                color=discord.Color.orange()
            )

            # Show current data counts
            user_memories = self.kernel.memory_store.user_memories.get(user_id, [])
            user_prefs = self.kernel.learning_engine.user_preferences.get(user_id)
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
                if user_id in self.kernel.learning_engine.user_preferences:
                    del self.kernel.learning_engine.user_preferences[user_id]
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
                if user_id in self.kernel.learning_engine.user_preferences:
                    del self.kernel.learning_engine.user_preferences[user_id]
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

                    # Create memories embed
                    mem_embed = discord.Embed(
                        title="📝 Your Memories",
                        description=f"I have {len(user_memories)} memories about you",
                        color=discord.Color.green()
                    )

                    if user_memories:
                        # Group by type
                        from collections import defaultdict
                        by_type = defaultdict(list)
                        for mem in user_memories:
                            by_type[mem.memory_type.value].append(mem)

                        for mem_type, mems in sorted(by_type.items()):
                            # Show top 5 most important
                            sorted_mems = sorted(mems, key=lambda m: m.importance, reverse=True)[:5]
                            mem_text = ""
                            for mem in sorted_mems:
                                importance_bar = "⭐" * int(mem.importance * 5)
                                mem_text += f"{importance_bar} {mem.content[:100]}\n"

                            mem_embed.add_field(
                                name=f"{mem_type.upper()} ({len(mems)} total)",
                                value=mem_text or "None",
                                inline=False
                            )
                    else:
                        mem_embed.description = "No memories stored yet. I'll learn about you as we interact!"

                    await interaction.response.send_message(embed=mem_embed, ephemeral=True)

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

                    learn_embed = discord.Embed(
                        title="📚 Learning Records",
                        description=f"I have {len(user_learning)} learning records from our interactions",
                        color=discord.Color.purple()
                    )

                    if user_learning:
                        # Show recent records
                        recent = sorted(user_learning, key=lambda r: r.timestamp, reverse=True)[:10]

                        for record in recent:
                            time_str = datetime.fromtimestamp(record.timestamp).strftime('%Y-%m-%d %H:%M')
                            learn_embed.add_field(
                                name=f"{record.interaction_type.value} - {time_str}",
                                value=f"Context: {record.outcome if record.outcome else record.context} {record.feedback_score}",
                                inline=False
                            )
                    else:
                        learn_embed.description = "No learning records yet. I'll learn from our interactions!"

                    await interaction.response.send_message(embed=learn_embed, ephemeral=True)

                learning_button.callback = learning_callback
                view.add_item(learning_button)

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

                    # Create paginated view of all memories
                    all_mem_text = "**All Memories:**\n\n"
                    for i, mem in enumerate(sorted(user_memories, key=lambda m: m.importance, reverse=True), 1):
                        importance_bar = "⭐" * int(mem.importance * 5)
                        tags_str = f" [{', '.join(mem.tags)}]" if mem.tags else ""
                        all_mem_text += f"{i}. {importance_bar} **{mem.memory_type.value}**{tags_str}\n   {mem.content}\n\n"

                        # Discord message limit
                        if len(all_mem_text) > 1800:
                            all_mem_text += f"... and {len(user_memories) - i} more"
                            break

                    await interaction.response.send_message(all_mem_text, ephemeral=True)

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

                    tasks_embed = discord.Embed(
                        title="📅 Scheduled Tasks",
                        description=f"You have {len(user_tasks)} scheduled tasks",
                        color=discord.Color.gold()
                    )

                    if user_tasks:
                        # Group by status
                        from collections import defaultdict
                        by_status = defaultdict(list)
                        for task in user_tasks:
                            by_status[task.status.value].append(task)

                        for status, tasks in sorted(by_status.items()):
                            task_text = ""
                            for task in tasks[:5]:  # Show max 5 per status
                                scheduled_dt = datetime.fromtimestamp(task.scheduled_time).strftime('%Y-%m-%d %H:%M')
                                priority_stars = "⭐" * task.priority
                                task_text += f"{priority_stars} **{task.task_type}** - {scheduled_dt}\n   {task.content[:80]}\n\n"

                            if len(tasks) > 5:
                                task_text += f"... and {len(tasks) - 5} more\n"

                            tasks_embed.add_field(
                                name=f"{status.upper()} ({len(tasks)} total)",
                                value=task_text or "None",
                                inline=False
                            )
                    else:
                        tasks_embed.description = "No scheduled tasks. Use kernel tools to schedule tasks!"

                    await interaction.response.send_message(embed=tasks_embed, ephemeral=True)

                tasks_button.callback = tasks_callback
                view.add_item(tasks_button)

                await ctx.send(embed=embed, view=view)

            except Exception as e:
                await ctx.send(f"❌ Error retrieving context: {e}")

        # Variables management command
        @self.bot.command(name="vars")
        async def vars_command(ctx: commands.Context, action: str = None, path: str = None, *, value: str = None):
            """
            Manage agent variables interactively.

            Usage:
                !vars                    - List all variables
                !vars list [path]        - List variables (optionally filtered by path)
                !vars get <path>         - Get a specific variable
                !vars set <path> <value> - Set a variable
                !vars delete <path>      - Delete a variable

            Examples:
                !vars
                !vars list discord
                !vars get discord.output_mode.268830485889810432
                !vars set user.preferences.theme dark
                !vars delete user.temp.data
            """
            user_id = str(ctx.author.id)

            if not hasattr(self.kernel.agent, 'variable_manager'):
                await ctx.send("❌ Variable manager not available!")
                return

            var_manager = self.kernel.agent.variable_manager

            # Default action is list
            if action is None:
                action = "list"

            action = action.lower()

            try:
                if action == "list":
                    # List all variables or filter by path
                    # Try to get all variables - check if get_all() exists
                    if hasattr(var_manager, 'get_all'):
                        all_vars = var_manager.get_all()
                    elif hasattr(var_manager, 'variables'):
                        # If variables are stored in a dict attribute
                        all_vars = var_manager.variables
                    elif hasattr(var_manager, '_variables'):
                        all_vars = var_manager._variables
                    else:
                        # Fallback: try to access internal storage
                        await ctx.send("❌ Cannot list variables - variable manager doesn't support listing!")
                        return

                    if not all_vars:
                        await ctx.send("📝 No variables stored yet!")
                        return

                    # Filter by path if provided
                    if path:
                        filtered_vars = {k: v for k, v in all_vars.items() if k.startswith(path)}
                        if not filtered_vars:
                            await ctx.send(f"📝 No variables found matching path: `{path}`")
                            return
                        all_vars = filtered_vars

                    # Create interactive view
                    embed = discord.Embed(
                        title="🔧 Agent Variables",
                        description=f"Total: {len(all_vars)} variable(s)" + (f" (filtered by `{path}`)" if path else ""),
                        color=discord.Color.blue(),
                        timestamp=datetime.now(UTC)
                    )

                    # Group variables by prefix
                    from collections import defaultdict
                    grouped = defaultdict(list)
                    for var_path, var_value in all_vars.items():
                        prefix = var_path.split('.')[0] if '.' in var_path else 'root'
                        grouped[prefix].append((var_path, var_value))

                    # Add fields for each group (max 25 fields per embed)
                    field_count = 0
                    for prefix, vars_list in sorted(grouped.items()):
                        if field_count >= 25:
                            break

                        # Show first 5 variables in each group
                        var_text = ""
                        for var_path, var_value in sorted(vars_list)[:5]:
                            # Truncate long values
                            value_str = str(var_value)
                            if len(value_str) > 100:
                                value_str = value_str[:97] + "..."
                            var_text += f"`{var_path}`\n└─ {value_str}\n\n"

                        if len(vars_list) > 5:
                            var_text += f"... and {len(vars_list) - 5} more\n"

                        embed.add_field(
                            name=f"📁 {prefix.upper()} ({len(vars_list)})",
                            value=var_text or "Empty",
                            inline=False
                        )
                        field_count += 1

                    # Create interactive buttons
                    view = discord.ui.View(timeout=300)

                    # Button: Refresh
                    refresh_button = discord.ui.Button(
                        label="🔄 Refresh",
                        style=discord.ButtonStyle.primary,
                        custom_id=f"vars_refresh_{user_id}"
                    )

                    async def refresh_callback(interaction: discord.Interaction):
                        if str(interaction.user.id) != user_id:
                            await interaction.response.send_message("❌ This is not your command!", ephemeral=True)
                            return

                        # Re-run the list command
                        await interaction.response.defer()
                        await vars_command.callback(ctx, "list", path)

                    refresh_button.callback = refresh_callback
                    view.add_item(refresh_button)

                    await ctx.send(embed=embed, view=view)

                elif action == "get":
                    if not path:
                        await ctx.send("❌ Usage: `!vars get <path>`")
                        return

                    var_value = var_manager.get(path)

                    if var_value is None:
                        await ctx.send(f"❌ Variable not found: `{path}`")
                        return

                    # Format value nicely
                    import json
                    try:
                        if isinstance(var_value, (dict, list)):
                            value_str = json.dumps(var_value, indent=2)
                        else:
                            value_str = str(var_value)
                    except:
                        value_str = str(var_value)

                    embed = discord.Embed(
                        title=f"🔧 Variable: {path}",
                        description=f"```json\n{value_str[:4000]}\n```" if len(value_str) < 4000 else f"```\n{value_str[:4000]}...\n```",
                        color=discord.Color.green(),
                        timestamp=datetime.now(UTC)
                    )

                    await ctx.send(embed=embed)

                elif action == "set":
                    if not path or value is None:
                        await ctx.send("❌ Usage: `!vars set <path> <value>`")
                        return

                    # Try to parse value as JSON first
                    import json
                    try:
                        parsed_value = json.loads(value)
                    except:
                        # If not JSON, use as string
                        parsed_value = value

                    var_manager.set(path, parsed_value)

                    embed = discord.Embed(
                        title="✅ Variable Set",
                        description=f"**Path:** `{path}`\n**Value:** `{parsed_value}`",
                        color=discord.Color.green(),
                        timestamp=datetime.now(UTC)
                    )

                    await ctx.send(embed=embed)

                elif action == "delete":
                    if not path:
                        await ctx.send("❌ Usage: `!vars delete <path>`")
                        return

                    # Check if variable exists
                    if var_manager.get(path) is None:
                        await ctx.send(f"❌ Variable not found: `{path}`")
                        return

                    # Delete variable - check if delete() exists
                    if hasattr(var_manager, 'delete'):
                        var_manager.delete(path)
                    elif hasattr(var_manager, 'remove'):
                        var_manager.remove(path)
                    else:
                        # Fallback: set to None
                        var_manager.set(path, None)
                        await ctx.send(f"⚠️ Variable set to None (delete not supported): `{path}`")
                        return

                    embed = discord.Embed(
                        title="✅ Variable Deleted",
                        description=f"**Path:** `{path}`",
                        color=discord.Color.orange(),
                        timestamp=datetime.now(UTC)
                    )

                    await ctx.send(embed=embed)

                else:
                    await ctx.send(f"❌ Unknown action: `{action}`\n\nValid actions: list, get, set, delete")

            except Exception as e:
                await ctx.send(f"❌ Error managing variables: {e}")
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
