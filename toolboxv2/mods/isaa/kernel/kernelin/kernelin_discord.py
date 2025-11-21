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
import time
from datetime import datetime
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

from toolboxv2 import App, get_app
from toolboxv2.mods.isaa.kernel.instace import Kernel
from toolboxv2.mods.isaa.kernel.types import Signal as KernelSignal, SignalType, KernelConfig, IOutputRouter
from toolboxv2.mods.isaa.kernel.kernelin.tools.discord_tools import DiscordKernelTools
import io
import wave
import tempfile
import os
import subprocess
import time


class WhisperAudioSink(voice_recv.AudioSink if VOICE_RECEIVE_SUPPORT else object):
    """Audio sink for receiving and transcribing voice input with Groq Whisper + VAD"""

    def __init__(self, kernel: Kernel, user_id: str, groq_client: 'Groq' = None, output_router=None):
        if VOICE_RECEIVE_SUPPORT:
            super().__init__()
        self.kernel = kernel
        self.user_id = user_id
        self.groq_client = groq_client
        self.output_router = output_router
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
                    # Send transcription to kernel
                    signal = KernelSignal(
                        type=SignalType.USER_INPUT,
                        id=str(user.id),
                        content=text,
                        metadata={
                            "interface": "discord_voice",
                            "user_name": str(user),
                            "user_display_name": user.display_name,
                            "transcription": True,
                            "language": getattr(transcription, 'language', 'unknown')
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
        if os.path.exists(piper_path):
            print(f"✓ Piper TTS enabled at {piper_path}")
        else:
            print(f"⚠️ Piper not found at {piper_path}. Local TTS disabled.")
            piper_path = None

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

        @self.bot.command(name="info")
        async def help_command(ctx: commands.Context):
            """Show comprehensive help message"""
            embed = discord.Embed(
                title="🤖 ProA Kernel - AI Assistant",
                description="Advanced AI-powered assistant with learning, memory, voice support, and Discord integration",
                color=discord.Color.green()
            )

            # Basic Commands
            basic_commands = "• `!status` - Show kernel status and metrics\n• `!info` - Show this help message"
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

                        # Create audio sink
                        sink = WhisperAudioSink(
                            kernel=self.kernel,
                            user_id=str(ctx.author.id),
                            groq_client=self.output_router.groq_client,
                            output_router=self.output_router
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

    async def _auto_save_loop(self):
        """Auto-save kernel state periodically"""
        while self.running:
            await asyncio.sleep(self.auto_save_interval)
            if self.running:
                await self.kernel.save_to_file(str(self.save_path))
                print(f"💾 Auto-saved Discord kernel at {datetime.now().strftime('%H:%M:%S')}")

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

    async def handle_message(self, message: discord.Message):
        """Handle incoming Discord message"""
        try:
            user_id = str(message.author.id)
            channel_id = message.channel.id

            # Register user channel (store channel object directly for this user)
            self.output_router.user_channels[user_id] = message.channel
            self.output_router.active_channels[channel_id] = message.channel

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
                # Send signal to kernel
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
                        "user_display_name": message.author.display_name
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
    agent.set_progress_callback(ProgressiveTreePrinter().progress_callback)
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
