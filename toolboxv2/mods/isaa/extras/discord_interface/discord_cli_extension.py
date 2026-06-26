"""
CLI Discord Extension - Integration für cli_v4
===============================================

Fügt /discord Befehle zur CLI hinzu:
- /discord connect    - Verbindet zum Discord Bot
- /discord disconnect - Trennt Discord
- /discord status     - Zeigt Status
- /discord channels   - Listet aktive Channels
- /discord voice      - Voice Channel Befehle

Usage in cli_v4.py:
    from discord_cli_extension import DiscordCLIExtension

    # In ISAA_Host.__init__:
    self.discord_ext = DiscordCLIExtension(self)

    # In _handle_command:
    if command == "/discord":
        await self.discord_ext.handle_command(args)
"""

import asyncio
import os
from threading import Thread
from typing import TYPE_CHECKING, Optional

from toolboxv2 import get_app

if TYPE_CHECKING:
    from toolboxv2.flows.isaa.icli import ISAA_Host

# Import print functions (angepasst für cli_v4)
try:
    from cli_v4 import (
        print_box_header,
        print_box_footer,
        print_box_content,
        print_status,
        print_separator,
        print_table_header,
        print_table_row,
        c_print,
    )
except ImportError:
    # Fallback
    def print_status(msg, status="info"):
        print(f"[{status}] {msg}")
    def print_box_header(title, icon=""):
        print(f"\n{icon} {title}")
        print("-" * 40)
    def print_box_footer():
        print()
    def print_box_content(text, style=""):
        print(f"  {text}")
    def print_separator():
        print("-" * 40)
    def print_table_header(cols, widths):
        print(" | ".join([c[0].ljust(w) for (c, w) in zip(cols, widths)]))
    def print_table_row(vals, widths, styles=None):
        print(" | ".join([str(v).ljust(w) for v, w in zip(vals, widths)]))
    def c_print(*args):
        print(*args)


class DiscordCLIExtension:
    """CLI Extension für Discord Integration mit Voice Support"""

    def __init__(self, host: "ISAA_Host"):
        self.host = host
        self.interface = None
        self.voice_mode = None  # VoiceModeExtension
        self._discord_task: Optional[Thread] = None
        self._connected = False
        # Voice backend mode: "omni_cloud" (streaming S2S) | "pipeline" (classic turn)
        self._omni_mode = "omni_cloud"

    async def handle_command(self, args: list[str]):
        """
        Haupthandler für /discord Befehle.

        Commands:
            /discord connect [token]  - Verbindet zum Discord Bot
            /discord disconnect       - Trennt Discord
            /discord status           - Zeigt Verbindungsstatus
            /discord channels         - Listet aktive Channels
            /discord send <address> <message> - Sendet Nachricht
            /discord search <query>   - Sucht im Adressbuch
            /discord voice <cmd>      - Voice Channel Befehle
        """
        if not args:
            await self._show_help()
            return

        action = args[0].lower()
        sub_args = args[1:]

        if action == "connect":
            await self._connect(sub_args)
        elif action == "disconnect":
            await self._disconnect()
        elif action == "status":
            await self._show_status()
        elif action == "channels":
            await self._list_channels()
        elif action == "send":
            await self._send_message(sub_args)
        elif action == "search":
            await self._search_address(sub_args)
        elif action == "active":
            await self._show_active()
        elif action == "whitelist":
            await self._handle_whitelist(sub_args)
        elif action == "voice":
            await self._handle_voice(sub_args)
        else:
            print_status(f"Unknown discord command: {action}", "error")
            await self._show_help()

    async def _show_help(self):
        """Zeigt Hilfe an"""
        print_box_header("Discord Commands", "💬")
        print_box_content("/discord connect [token] - Connect to Discord", "")
        print_box_content("/discord disconnect      - Disconnect from Discord", "")
        print_box_content("/discord status          - Show connection status", "")
        print_box_content("/discord whitelist        - List whitelisted users", "")
        print_box_content("/discord whitelist add <id> - Add user ID to whitelist", "")
        print_box_content("/discord whitelist remove <id> - Remove user ID", "")
        print_box_content("/discord channels         - List active channels", "")
        print_box_content("/discord channels        - List active channels", "")
        print_box_content("/discord active          - Show active conversations", "")
        print_box_content("/discord send <addr> <msg> - Send message to address", "")
        print_box_content("/discord search <query>  - Search address book", "")
        print_separator()
        print_box_content("Voice Commands:", "bold")
        print_box_content("/discord voice join <id> - Join voice channel", "")
        print_box_content("/discord voice leave     - Leave voice channel", "")
        print_box_content("/discord voice status    - Voice connection status", "")
        print_box_content("/discord voice speak <text> - Speak in voice", "")
        print_box_content("/discord voice stop      - Stop speaking", "")
        print_box_content("/discord voice set <voice> <backend> - Stop speaking", "")
        print_box_content("/discord voice context   - Show voice conversation", "")
        print_box_footer()

    async def _connect(self, args: list[str]):
        """Verbindet zum Discord Bot"""
        if self._connected:
            print_status("Already connected to Discord", "warning")
            return

        # Token holen
        token = args[0] if args else os.getenv("DISCORD_BOT_TOKEN")
        if not token:
            print_status("Discord token required. Usage: /discord connect <token>", "error")
            print_status("Or set DISCORD_TOKEN environment variable", "info")
            return

        print_status("Connecting to Discord...", "progress")

        try:
            from .discord_interface import create_discord_interface
            from .voice_mode import create_voice_mode, VOICE_RECV_AVAILABLE

            # Self-Agent (operator's full agent) — admins can switch to it
            self_agent = await self.host.isaa_tools.get_agent(self.host.active_agent_name)

            # Dedicated public moderator agent (discord-only tools, safe to expose)
            moderator_agent = await self.host.isaa_tools.get_agent("discord_moderator")

            # _connect runs on the icli event loop; capture it so Discord (own
            # thread/loop) can bridge agent runs onto the dashboard runner.
            runner_loop = asyncio.get_running_loop()

            # Interface erstellen
            self.interface = create_discord_interface(
                agent=moderator_agent,
                self_agent=self_agent,
                token=token,
                respond_to_mentions_only=True,
                tts_backend="groq",
                tts_voice="autumn",
                stt_backend="groq",
                language="de",
                host=self.host,
                runner_loop=runner_loop,
            )

            # Voice Mode hinzufügen
            self.voice_mode = create_voice_mode(self.interface, omni_mode=self._omni_mode)

            # In Background Task starten
            self._discord_task = get_app().run_bg_task_advanced(self._run_discord)
            self._connected = True

            print_status("Discord interface started in background", "success")
            print_status("Bot will respond to @mentions in channels and all DMs", "info")

            if VOICE_RECV_AVAILABLE:
                print_status("Voice receive available - use /discord voice join", "info")
            else:
                print_status("Voice receive not available - install discord-ext-voice-recv", "warning")

        except ImportError as e:
            print_status(f"Import error: {e}", "error")
            print_status("Make sure discord_interface package is available.", "info")
        except Exception as e:
            print_status(f"Failed to connect: {e}", "error")
            import traceback
            traceback.print_exc()

    async def _run_discord(self):
        """Läuft den Discord Bot im Background"""
        try:
            await self.interface.start()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print_status(f"Discord error: {e}", "error")
        finally:
            self._connected = False

    async def _disconnect(self):
        """Trennt Discord"""
        if not self._connected:
            print_status("Not connected to Discord", "warning")
            return

        print_status("Disconnecting from Discord...", "progress")

        # Voice zuerst beenden
        if self.voice_mode:
            for guild_id in list(self.voice_mode.voice_handler._voice_clients.keys()):
                await self.voice_mode.voice_handler.leave_channel(guild_id)

        if self._discord_task:
            del self._discord_task

        if self.interface:
            await self.interface.stop()
            self.interface = None
            self.voice_mode = None

        self._connected = False
        print_status("Disconnected from Discord", "success")

    async def _show_status(self):
        """Zeigt Verbindungsstatus"""
        print_box_header("Discord Status", "💬")

        if not self._connected:
            print_box_content("Not connected", "warning")
            print_box_footer()
            return

        print_box_content("Connected", "success")

        if self.interface and self.interface.bot.user:
            print_box_content(f"Bot: {self.interface.bot.user.name}", "info")
            print_box_content(f"ID: {self.interface.bot.user.id}", "info")
            print_box_content(f"Guilds: {len(self.interface.bot.guilds)}", "info")

            for guild in self.interface.bot.guilds:
                print_box_content(f"  - {guild.name} ({guild.member_count} members)", "")

        # Voice Status
        if self.voice_mode:
            voice_clients = self.voice_mode.voice_handler._voice_clients
            if voice_clients:
                print_separator()
                print_box_content("Voice Connections:", "bold")
                for guild_id, vc in voice_clients.items():
                    if vc.is_connected():
                        channel = vc.channel
                        playing = "🔊" if vc.is_playing() else "🔇"
                        print_box_content(f"  {playing} {channel.name} ({channel.guild.name})", "")

        print_box_footer()

    async def _list_channels(self):
        """Listet verfügbare Channels"""
        if not self._connected or not self.interface:
            print_status("Not connected to Discord", "warning")
            return

        print_box_header("Discord Channels", "📺")

        columns = [("Server", ""), ("Channel", ""), ("Type", "")]
        widths = [20, 25, 10]
        print_table_header(columns, widths)

        for guild in self.interface.bot.guilds:
            for channel in guild.channels:
                if hasattr(channel, 'send') or hasattr(channel, 'connect'):  # Text or Voice
                    ch_type = "voice" if hasattr(channel, 'connect') and not hasattr(channel, 'send') else str(channel.type)
                    print_table_row(
                        [guild.name[:18], f"#{channel.name}"[:23], ch_type[:8]],
                        widths,
                        ["white", "cyan", "grey"]
                    )

        print_box_footer()

    async def _handle_whitelist(self, args: list[str]):
        """Verwaltet die Whitelist von Discord User-IDs über die CLI [3]"""
        if not self._connected or not self.interface:
            print_status("Not connected to Discord", "warning")
            return

        if not args:
            print_box_header("Discord Whitelist", "🔒")
            if not self.interface.admin_ids:
                print_box_content("Whitelist is empty (everyone is ignored)", "warning")
            else:
                for uid in self.interface.admin_ids:
                    # Benutzernamen asynchron und thread-sicher vom Bot-Loop auflösen [3]
                    try:
                        async def _fetch_username(u_id=uid):
                            user = self.interface.bot.get_user(u_id)
                            if not user:
                                try:
                                    user = await self.interface.bot.fetch_user(u_id)
                                except Exception:
                                    return "Unknown User"
                            return f"@{user.name}" if user else "Unknown User"

                        future = asyncio.run_coroutine_threadsafe(_fetch_username(), self.interface.bot.loop)
                        username = await asyncio.wrap_future(future)
                    except Exception:
                        username = "Unknown User"

                    print_box_content(f"  • {uid} ({username})", "")

            # Mini-Docs: Anleitung zum Finden der ID [3]
            print_separator()
            print_box_content("How to get a Discord User ID:", "bold")
            print_box_content("1. Open Discord -> User Settings -> Advanced.", "")
            print_box_content("2. Enable 'Developer Mode'.", "")
            print_box_content("3. Right-click any user's avatar/profile -> Click 'Copy User ID'.", "")
            print_box_footer()
            return

        sub_action = args[0].lower()
        if sub_action == "add":
            if len(args) < 2:
                print_status("Usage: /discord whitelist add <user_id>", "error")
                return
            try:
                user_id = int(args[1])
                if user_id not in self.interface.admin_ids:
                    self.interface.admin_ids.append(user_id)
                    print_status(f"Added {user_id} to whitelist", "success")
                else:
                    print_status(f"User {user_id} is already whitelisted", "info")
            except ValueError:
                print_status("Invalid user ID format", "error")

        elif sub_action == "remove":
            if len(args) < 2:
                print_status("Usage: /discord whitelist remove <user_id>", "error")
                return
            try:
                user_id = int(args[1])
                if user_id in self.interface.admin_ids:
                    self.interface.admin_ids.remove(user_id)
                    print_status(f"Removed {user_id} from whitelist", "success")
                else:
                    print_status(f"User {user_id} not found in whitelist", "warning")
            except ValueError:
                print_status("Invalid user ID format", "error")
        else:
            print_status(f"Unknown whitelist action: {sub_action}", "error")

    async def _show_active(self):
        """Zeigt aktive Conversations"""
        if not self._connected or not self.interface:
            print_status("Not connected to Discord", "warning")
            return

        if not self.interface.address_book:
            print_status("Address book not initialized", "warning")
            return

        print_box_header("Active Conversations", "💬")

        active = self.interface.address_book.get_active_conversations()
        convs = active.get("conversations", {})

        if not convs:
            print_box_content("No active conversations", "info")
            print_box_footer()
            return

        columns = [("User", ""), ("Channel", ""), ("Last Message", "")]
        widths = [18, 20, 20]
        print_table_header(columns, widths)

        for addr, info in convs.items():
            print_table_row(
                [
                    info.get("user_name", "?")[:16],
                    (info.get("channel_name") or "DM")[:18],
                    info.get("last_message", "?")[:18],
                ],
                widths,
                ["cyan", "white", "grey"]
            )

        print_box_footer()

    async def _send_message(self, args: list[str]):
        """Sendet eine Nachricht"""
        if not self._connected or not self.interface:
            print_status("Not connected to Discord", "warning")
            return

        if len(args) < 2:
            print_status("Usage: /discord send <address> <message>", "error")
            print_status("Example: /discord send discord://dm:123456789 Hello!", "info")
            return

        address = args[0]
        message = " ".join(args[1:])

        print_status(f"Sending to {address}...", "progress")

        result = await self.interface.router.route_response(
            content=message,
            target_address=address,
        )

        if result.get("success"):
            print_status("Message sent", "success")
        else:
            print_status(f"Failed: {result.get('error')}", "error")

    async def _search_address(self, args: list[str]):
        """Sucht im Adressbuch"""
        if not self._connected or not self.interface:
            print_status("Not connected to Discord", "warning")
            return

        if not args:
            print_status("Usage: /discord search <query>", "error")
            return

        query = " ".join(args)

        if not self.interface.address_book:
            print_status("Address book not initialized", "warning")
            return

        results = self.interface.address_book.search(query)

        if not results:
            print_status(f"No results for '{query}'", "info")
            return

        print_box_header(f"Search Results: {query}", "🔍")

        columns = [("Type", ""), ("Name", ""), ("Address", "")]
        widths = [10, 20, 40]
        print_table_header(columns, widths)

        for r in results:
            print_table_row(
                [r["type"][:8], r["name"][:18], r["address"][:38]],
                widths,
                ["cyan", "white", "grey"]
            )

        print_box_footer()

    # =========================================================================
    # VOICE COMMANDS
    # =========================================================================

    async def _handle_voice(self, args: list[str]):
        """Handle /discord voice <subcommand>"""
        if not self._connected or not self.interface:
            print_status("Not connected to Discord", "warning")
            return

        if not self.voice_mode:
            print_status("Voice mode not initialized", "error")
            return

        if not args:
            await self._show_voice_help()
            return

        cmd = args[0].lower()
        sub_args = args[1:]
        handler = self.voice_mode.voice_handler

        if cmd == "join":
            await self._voice_join(sub_args)
        elif cmd == "leave":
            await self._voice_leave()
        elif cmd == "status":
            await self._voice_status()
        elif cmd == "speak":
            await self._voice_speak(sub_args)
        elif cmd == "stop":
            await self._voice_stop()
        elif cmd == "context":
            await self._voice_context()
        elif cmd == "set":
            await self._voice_set(sub_args)
        elif cmd == "channels":
            await self._voice_channels()
        elif cmd == "backend":
            await self._voice_backend(sub_args)
        else:
            print_status(f"Unknown voice command: {cmd}", "error")
            await self._show_voice_help()

    async def _voice_backend(self, args: list[str]):
        """/discord voice backend <omni|pipeline> — select the voice mode.

        omni     -> omni_cloud streaming S2S (server VAD ends turns)
        pipeline -> classic STT->LLM->TTS turn pipeline (own VAD, needs silence)
        """
        _MAP = {
            "omni": "omni_cloud", "cloud": "omni_cloud", "streaming": "omni_cloud",
            "pipeline": "pipeline", "classic": "pipeline", "turn": "pipeline",
        }
        if not args:
            print_status(f"Current voice backend: {self._omni_mode}", "info")
            print_status("Usage: /discord voice backend <omni|pipeline>", "info")
            return

        choice = _MAP.get(args[0].lower())
        if choice is None:
            print_status(f"Unknown backend: {args[0]} (use omni|pipeline)", "error")
            return

        ctrl = getattr(self.voice_mode, "omni_controller", None)
        if ctrl is not None and getattr(ctrl, "_sessions", None):
            print_status("Active voice session — leave and rejoin to apply.", "warning")

        self._omni_mode = choice
        if ctrl is not None:
            ctrl.mode = choice  # next attach() builds this backend
        print_status(f"Voice backend -> {choice} (effective on next join)", "success")

    async def _show_voice_help(self):
        """Zeigt Voice-Hilfe"""
        print_box_header("Voice Commands", "🎤")
        print_box_content("/discord voice join <channel_id> - Join voice channel", "")
        print_box_content("/discord voice leave             - Leave voice channel", "")
        print_box_content("/discord voice status            - Show voice status", "")
        print_box_content("/discord voice speak <text>      - Speak text via TTS", "")
        print_box_content("/discord voice stop              - Stop speaking", "")
        print_box_content("/discord voice context           - Show conversation", "")
        print_box_content("/discord voice set <voice> <backend> - Set voice and backend", "")
        print_box_content("/discord voice channels          - List voice channels", "")
        print_box_footer()

    async def _voice_join(self, args: list[str]):
        """Join voice channel (Thread-Safe)"""
        if not args:
            print_status("Usage: /discord voice join <channel_id>", "error")
            print_status("Use /discord voice channels to see available channels", "info")
            return

        try:
            channel_id = int(args[0])
        except ValueError:
            print_status("Invalid channel ID", "error")
            return

        # Sicherstellen, dass wir Zugriff auf den Bot-Loop haben
        if not self.interface or not self.interface.bot.loop:
            print_status("Discord not connected or loop not available", "error")
            return

        # Logik, die im Bot-Thread ausgeführt werden muss
        async def _join_task():
            import discord
            # Channel im Kontext des Bot-Loops abrufen
            channel = self.interface.bot.get_channel(channel_id)
            if not channel:
                try:
                    channel = await self.interface.bot.fetch_channel(channel_id)
                except:
                    return {"success": False, "error": f"Channel {channel_id} not found"}

            if not isinstance(channel, discord.VoiceChannel):
                return {"success": False, "error": f"{channel.name} is not a voice channel"}

            # Join ausführen (im richtigen Loop)
            return await self.voice_mode.voice_handler.join_channel(channel)

        print_status(f"Joining channel {channel_id}...", "progress")

        try:
            # Task sicher an den Bot-Loop übergeben und im Haupt-Loop auf Ergebnis warten
            future = asyncio.run_coroutine_threadsafe(_join_task(), self.interface.bot.loop)
            result = await asyncio.wrap_future(future)

            if result.get("success"):
                listening = "✅ Listening enabled" if result.get("listening") else "❌ Listening disabled"
                print_status(f"Joined {result.get('channel_name')}", "success")
                print_status(listening, "info")

                if result.get("warning"):
                    print_status(result["warning"], "warning")
            else:
                print_status(f"Failed: {result.get('error')}", "error")

        except Exception as e:
            print_status(f"Error joining channel: {e}", "error")

    async def _voice_set(self, args: list[str]):
        """Set voice and backend (Thread-Safe)"""
        if not args or len(args) != 2:
            print_status("Usage: /discord voice set <voice> <backend>", "error")
            return

        voice = args[0]
        backend = args[1]

        if backend not in ["groq", "piper", "elevenlabs"]:
            print_status("Invalid backend. Valid: groq, piper, elevenlabs", "error")
            return

        # Sicherstellen, dass wir Zugriff auf den Bot-Loop haben
        if not self.interface or not self.interface.bot.loop:
            print_status("Discord not connected or loop not available", "error")
            return

        # Logik, die im Bot-Thread ausgeführt werden muss
        async def _set_task():
            self.interface.media_handler.tts_voice = voice
            self.interface.media_handler.tts_backend = backend
            return True

        print_status(f"Setting voice to {voice} with {backend}...", "progress")

        try:
            future = asyncio.run_coroutine_threadsafe(_set_task(), self.interface.bot.loop)
            await asyncio.wrap_future(future)
            print_status("Voice settings updated", "success")
        except Exception as e:
            print_status(f"Error setting voice: {e}", "error")

    async def _voice_leave(self):
        """Leave voice channel (Thread-Safe)"""
        if not self.interface or not self.interface.bot.loop:
            return

        handler = self.voice_mode.voice_handler

        async def _leave_task():
            # Guild ID finden
            guild_id = None
            for gid in handler._voice_clients.keys():
                guild_id = gid
                break

            if not guild_id:
                return {"success": False, "error": "Not in any voice channel"}

            return await handler.leave_channel(guild_id)

        try:
            future = asyncio.run_coroutine_threadsafe(_leave_task(), self.interface.bot.loop)
            result = await asyncio.wrap_future(future)

            if result.get("success"):
                print_status("Left voice channel", "success")
            else:
                print_status(f"Failed: {result.get('error')}", "error")

        except Exception as e:
            print_status(f"Error leaving: {e}", "error")

    async def _voice_status(self):
        """Show voice status"""
        handler = self.voice_mode.voice_handler

        if not handler._voice_clients:
            print_status("Not in any voice channel", "info")
            return

        print_box_header("Voice Status", "🎤")

        for guild_id, vc in handler._voice_clients.items():
            status = handler.get_voice_status(guild_id)

            print_box_content(f"Channel: {status.get('channel_name')}", "")
            print_box_content(f"Playing: {'Yes' if status.get('is_playing') else 'No'}", "")
            print_box_content(f"Listening: {'Yes' if status.get('listening') else 'No'}", "")
            print_box_content(f"Participants: {', '.join(status.get('participants', []))}", "")
            print_box_content(f"Messages: {status.get('conversation_messages', 0)}", "")
            print_box_content(f"Latency: {status.get('latency', 0)*1000:.0f}ms", "")
            print_box_content(f"Voice: {status.get('tts_voice', 'unknown')}", "")
            print_box_content(f"Backend: {status.get('tts_backend', 'unknown')}", "")

        print_box_footer()

    async def _voice_speak(self, args: list[str]):
        """Speak text in voice channel (Thread-Safe)"""
        if not args:
            print_status("Usage: /discord voice speak <text>", "error")
            return

        if not self.interface or not self.interface.bot.loop:
            return

        text = " ".join(args)
        handler = self.voice_mode.voice_handler

        async def _speak_task():
            # Guild ID finden
            guild_id = None
            for gid in handler._voice_clients.keys():
                guild_id = gid
                break

            if not guild_id:
                return False

            await handler.speak_streaming(guild_id, text)
            return True

        print_status(f"Speaking: {text[:50]}...", "progress")

        try:
            future = asyncio.run_coroutine_threadsafe(_speak_task(), self.interface.bot.loop)
            success = await asyncio.wrap_future(future)

            if success:
                print_status("Queued for playback", "success")
            else:
                print_status("Not in any voice channel", "warning")

        except Exception as e:
            print_status(f"Error speaking: {e}", "error")

    async def _voice_stop(self):
        """Stop speaking"""
        handler = self.voice_mode.voice_handler

        for guild_id in handler._voice_clients.keys():
            await handler.stop_speaking(guild_id)

        print_status("Stopped speaking", "success")

    async def _voice_context(self):
        """Show voice conversation context"""
        handler = self.voice_mode.voice_handler

        # Find connected guild
        guild_id = None
        for gid in handler._voice_clients.keys():
            guild_id = gid
            break

        if not guild_id:
            print_status("Not in any voice channel", "warning")
            return

        context = handler.get_conversation_context(guild_id)

        print_box_header("Voice Conversation (last 5 min)", "💬")
        print(context)
        print_box_footer()

    async def _voice_channels(self):
        """List voice channels"""
        if not self.interface:
            return

        import discord

        print_box_header("Voice Channels", "🔊")

        columns = [("Server", ""), ("Channel", ""), ("ID", ""), ("Users", "")]
        widths = [18, 20, 20, 8]
        print_table_header(columns, widths)

        for guild in self.interface.bot.guilds:
            for channel in guild.channels:
                if isinstance(channel, discord.VoiceChannel):
                    members = len([m for m in channel.members if not m.bot])
                    print_table_row(
                        [guild.name[:16], channel.name[:18], str(channel.id), str(members)],
                        widths,
                        ["white", "cyan", "grey", "green"]
                    )

        print_box_footer()


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def get_completer_dict() -> dict:
    """
    Returns completer dict für CLI Integration.

    Usage in cli_v4.py _build_completer():
        from discord_cli_extension import get_completer_dict
        completer_dict["/discord"] = get_completer_dict()
    """
    return {
        "connect": None,
        "disconnect": None,
        "status": None,
        "whitelist": {
            "add": None,
            "remove": None,
        },
        "channels": None,
        "active": None,
        "send": None,
        "search": None,
        "voice": {
            "join": None,
            "leave": None,
            "status": None,
            "speak": None,
            "stop": None,
            "context": None,
            "channels": None,
            "set": {
                "groq": {"autumn": None, "diana": None, "hannah": None, "austin": None, "daniel": None, "troy": None},
                "piper": None,
                "elevenlabs": None,
            },
        },
    }
