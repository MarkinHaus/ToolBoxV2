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
from typing import Optional, Dict, List

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

PIPE_SUPPORT = False

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
        if VOICE_RECEIVE_SUPPORT:
            super().__init__()
        self.kernel = kernel
        self.user_id = user_id
        self.groq_client = groq_client
        self.output_router = output_router
        self.discord_kernel = discord_kernel  # Reference to DiscordKernel for context
        self.audio_buffer: Dict[int, List[bytes]] = {}  # ssrc -> audio chunks
        self.user_ssrc_map: Dict[int, discord.Member] = {}  # ssrc -> member
        self.transcription_interval = 3.0  # Transcribe every 3 seconds
        self.last_transcription: Dict[int, float] = {}  # ssrc -> timestamp
        self.speaking_state: Dict[int, bool] = {}  # ssrc -> is_speaking
        self.last_audio_time: Dict[int, float] = {}  # ssrc -> last audio timestamp
        self.silence_threshold = 1.0  # 1 second of silence before stopping transcription

    def wants_opus(self) -> bool:
        """We want decoded PCM audio, not Opus"""
        return False

    def write(self, user, data):
        """Receive audio data from Discord"""
        if not user:
            return

        # Get SSRC (Synchronization Source identifier)
        ssrc = data.ssrc if hasattr(data, 'ssrc') else None
        if not ssrc:
            return

        # Map SSRC to user
        if ssrc not in self.user_ssrc_map:
            self.user_ssrc_map[ssrc] = user

        # Buffer audio data
        if ssrc not in self.audio_buffer:
            self.audio_buffer[ssrc] = []
            self.last_transcription[ssrc] = time.time()

        self.audio_buffer[ssrc].append(data.pcm)

        # Check if we should transcribe
        current_time = time.time()
        if current_time - self.last_transcription[ssrc] >= self.transcription_interval:
            asyncio.create_task(self._transcribe_buffer(ssrc))
            self.last_transcription[ssrc] = current_time

    async def _transcribe_buffer(self, ssrc: int):
        """Transcribe buffered audio for a user"""
        if ssrc not in self.audio_buffer or not self.audio_buffer[ssrc]:
            return

        if not GROQ_SUPPORT or not self.groq_client:
            print("⚠️ Groq not available for transcription")
            return

        try:
            # Get user
            user = self.user_ssrc_map.get(ssrc)
            if not user:
                return

            # Combine audio chunks
            audio_data = b''.join(self.audio_buffer[ssrc])
            self.audio_buffer[ssrc] = []  # Clear buffer

            if len(audio_data) < 1600:  # Too short (< 0.1 seconds at 16kHz)
                return

            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(2)  # Stereo
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(48000)  # Discord uses 48kHz
                wav_file.writeframes(audio_data)

            wav_buffer.seek(0)

            # Save to temporary file (Groq API needs file path)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(wav_buffer.read())
                temp_path = temp_file.name

            try:
                # Transcribe with Groq Whisper
                with open(temp_path, 'rb') as audio_file:
                    transcription = self.groq_client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-large-v3-turbo",
                        response_format="json",
                        temperature=0.0
                    )

                text = transcription.text.strip()

                if text:
                    # Get Discord context if available
                    discord_context = None
                    if self.discord_kernel and hasattr(user, 'guild'):
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
                                mock_msg = MockMessage(user, user.guild, voice_channel)
                                discord_context = self.discord_kernel._get_discord_context(mock_msg)

                                # Inject context into agent's variable system
                                if hasattr(self.kernel.agent, 'variable_manager'):
                                    self.kernel.agent.variable_manager.set(
                                        f'discord.current_context.{str(user.id)}',
                                        discord_context
                                    )

                    # Send transcription to kernel with enhanced metadata
                    signal = KernelSignal(
                        type=SignalType.USER_INPUT,
                        id=str(user.id),
                        content=text,
                        metadata={
                            "interface": "discord_voice",
                            "user_name": str(user),
                            "user_display_name": user.display_name,
                            "transcription": True,
                            "language": getattr(transcription, 'language', 'unknown'),
                            "discord_context": discord_context
                        }
                    )
                    await self.kernel.process_signal(signal)
                    print(f"🎤 Voice input from {user.display_name}: {text}")

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            print(f"❌ Error transcribing audio: {e}")

    def cleanup(self):
        """Cleanup when sink is stopped"""
        self.audio_buffer.clear()
        self.user_ssrc_map.clear()
        self.last_transcription.clear()

    @voice_recv.AudioSink.listener() if VOICE_RECEIVE_SUPPORT else lambda f: f
    def on_voice_member_disconnect(self, member: discord.Member, ssrc: int):
        """Handle member disconnect"""
        if ssrc in self.audio_buffer:
            del self.audio_buffer[ssrc]
        if ssrc in self.user_ssrc_map:
            del self.user_ssrc_map[ssrc]
        if ssrc in self.last_transcription:
            del self.last_transcription[ssrc]
        if ssrc in self.speaking_state:
            del self.speaking_state[ssrc]
        if ssrc in self.last_audio_time:
            del self.last_audio_time[ssrc]

    @voice_recv.AudioSink.listener() if VOICE_RECEIVE_SUPPORT else lambda f: f
    def on_voice_member_speaking_start(self, member: discord.Member):
        """Handle speaking start (VAD)"""
        print(f"🎤 {member.display_name} started speaking")
        # Find SSRC for this member
        for ssrc, m in self.user_ssrc_map.items():
            if m.id == member.id:
                self.speaking_state[ssrc] = True
                break

    @voice_recv.AudioSink.listener() if VOICE_RECEIVE_SUPPORT else lambda f: f
    def on_voice_member_speaking_stop(self, member: discord.Member):
        """Handle speaking stop (VAD)"""
        print(f"🔇 {member.display_name} stopped speaking")
        # Find SSRC for this member
        for ssrc, m in self.user_ssrc_map.items():
            if m.id == member.id:
                self.speaking_state[ssrc] = False
                # Trigger final transcription
                asyncio.create_task(self._transcribe_buffer(ssrc))
                break


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

    def __init__(self, bot: commands.Bot, groq_client: 'Groq' = None, elevenlabs_client: 'ElevenLabs' = None, piper_path: str = None):
        self.bot = bot
        self.active_channels: Dict[int, discord.TextChannel] = {}
        self.user_channels: Dict[str, discord.TextChannel] = {}  # user_id -> channel object
        self.voice_clients: Dict[int, discord.VoiceClient] = {}  # guild_id -> voice client
        self.audio_sinks: Dict[int, WhisperAudioSink] = {}  # guild_id -> audio sink
        self.groq_client = groq_client
        self.elevenlabs_client = elevenlabs_client
        self.piper_path = piper_path
        self.tts_enabled: Dict[int, bool] = {}  # guild_id -> tts enabled
        self.tts_mode: Dict[int, str] = {}  # guild_id -> "elevenlabs" or "piper"

    def _create_embed(
        self,
        content: str,
        title: str = None,
        color: discord.Color = discord.Color.blue(),
        fields: List[dict] = None
    ) -> discord.Embed:
        """Create a Discord embed"""
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

            # Use embeds for rich formatting
            use_embed = metadata and metadata.get("use_embed", True)

            if use_embed:
                embed = self._create_embed(
                    content=content,
                    title=metadata.get("title") if metadata else None,
                    color=discord.Color.green()
                )
                await channel.send(embed=embed)
            else:
                await channel.send(content)

            # TTS if enabled and in voice channel
            if channel.guild and channel.guild.id in self.tts_enabled and self.tts_enabled[channel.guild.id]:
                await self._speak_text(channel.guild.id, content)

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
            # Create temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as text_file:
                text_file.write(text)
                text_path = text_file.name

            output_path = tempfile.mktemp(suffix=".wav")

            # Run Piper (reads from stdin)
            with open(text_path, 'r', encoding='utf-8') as input_file:
                subprocess.run(
                    [self.piper_path, "--output_file", output_path],
                    stdin=input_file,
                    check=True,
                    capture_output=True
                )

            # Play audio
            audio_source = discord.FFmpegPCMAudio(output_path)

            def cleanup(error):
                try:
                    os.unlink(output_path)
                    os.unlink(text_path)
                except:
                    pass
                if error:
                    print(f"Error playing audio: {error}")

            voice_client.play(audio_source, after=cleanup)

        except Exception as e:
            print(f"❌ Piper TTS error: {e}")

    async def send_media(
        self,
        user_id: str,
        file_path: str = None,
        url: str = None,
        caption: str = None
    ):
        """Send media to Discord user"""
        try:
            channel = self.user_channels.get(user_id)
            if not channel:
                print(f"⚠️ No channel found for user {user_id}")
                return

            if file_path:
                # Send file attachment
                file = discord.File(file_path)
                await channel.send(content=caption, file=file)
            elif url:
                # Send embed with image
                embed = discord.Embed(description=caption or "")
                embed.set_image(url=url)
                await channel.send(embed=embed)

        except Exception as e:
            print(f"❌ Error sending Discord media to user {user_id}: {e}")


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
        global PIPER_SUPPORT
        if os.path.exists(piper_path):
            print(f"✓ Piper TTS enabled at {piper_path}")
            PIPER_SUPPORT = True
        else:
            print(f"⚠️ Piper not found at {piper_path}. Local TTS disabled.")
            piper_path = None
            PIPER_SUPPORT = False

        self.output_router = DiscordOutputRouter(
            self.bot,
            groq_client=groq_client,
            elevenlabs_client=elevenlabs_client,
            piper_path=piper_path
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
            await self.stop()
            await ctx.send("👋 Goodbye!")
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
                "• `!context` - Show agent context and user profile"
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
                # Check if user is in a voice channel
                if not ctx.author.voice:
                    await ctx.send("❌ You need to be in a voice channel!")
                    return

                channel = ctx.author.voice.channel

                try:
                    if ctx.voice_client:
                        await ctx.voice_client.move_to(channel)
                        channel_name = getattr(channel, 'name', 'DM Voice Channel')
                        await ctx.send(f"🔊 Moved to {channel_name}")
                    else:
                        voice_client = await channel.connect()

                        # Store voice client (use guild_id or user_id for DMs)
                        if ctx.guild:
                            self.output_router.voice_clients[ctx.guild.id] = voice_client
                            await ctx.send(f"🔊 Joined {channel.name}")
                        else:
                            # DM Voice Channel
                            self.output_router.voice_clients[ctx.author.id] = voice_client
                            await ctx.send(f"🔊 Joined DM voice channel")
                except Exception as e:
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
            if VOICE_RECEIVE_SUPPORT and GROQ_SUPPORT:
                @self.bot.command(name="listen")
                async def start_listening(ctx: commands.Context):
                    """Start listening to voice input and transcribing with Groq Whisper"""
                    if not ctx.voice_client:
                        await ctx.send("❌ I'm not in a voice channel! Use `!join` first.")
                        return

                    if ctx.voice_client.is_listening():
                        await ctx.send("⚠️ Already listening!")
                        return

                    try:
                        guild_id = ctx.guild.id

                        # Create audio sink with Discord context
                        sink = WhisperAudioSink(
                            kernel=self.kernel,
                            user_id=str(ctx.author.id),
                            groq_client=self.output_router.groq_client,
                            output_router=self.output_router,
                            discord_kernel=self  # Pass Discord kernel for context
                        )

                        # Start listening
                        ctx.voice_client.listen(sink)
                        self.output_router.audio_sinks[guild_id] = sink

                        await ctx.send("🎤 Started listening! Speak and I'll transcribe your voice in real-time.")
                    except Exception as e:
                        await ctx.send(f"❌ Error starting voice input: {e}")

                @self.bot.command(name="stop_listening")
                async def stop_listening(ctx: commands.Context):
                    """Stop listening to voice input"""
                    if not ctx.voice_client:
                        await ctx.send("❌ I'm not in a voice channel!")
                        return

                    if not ctx.voice_client.is_listening():
                        await ctx.send("⚠️ Not currently listening!")
                        return

                    try:
                        guild_id = ctx.guild.id
                        ctx.voice_client.stop_listening()

                        if guild_id in self.output_router.audio_sinks:
                            del self.output_router.audio_sinks[guild_id]

                        await ctx.send("🔇 Stopped listening to voice input.")
                    except Exception as e:
                        await ctx.send(f"❌ Error stopping voice input: {e}")

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
                    # Register with agent
                    self.kernel.agent.set_progress_callback(printer.progress_callback)
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
                    self.kernel.agent.set_progress_callback(printer.progress_callback)
                    await printer.enable()
                    await ctx.send("✅ Progress tracking enabled!")
            else:
                await ctx.send("❌ Invalid action. Use: !progress [on|off|toggle]")

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
                        from toolboxv2.mods.isaa.base.Agent.utils import get_max_tokens
                        fast_model = self.kernel.agent.amd.fast_llm_model.split('/')[-1]
                        complex_model = self.kernel.agent.amd.complex_llm_model.split('/')[-1]
                        max_tokens_fast = get_max_tokens(fast_model)
                        max_tokens_complex = get_max_tokens(complex_model)
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

                embed.set_footer(text="ProA Kernel Context System")

                await ctx.send(embed=embed)

            except Exception as e:
                await ctx.send(f"❌ Error retrieving context: {e}")

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

            # Handle attachments
            attachments_info = []
            if message.attachments:
                for attachment in message.attachments:
                    attachments_info.append({
                        "filename": attachment.filename,
                        "url": attachment.url,
                        "content_type": attachment.content_type
                    })

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
