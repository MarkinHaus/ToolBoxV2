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
from typing import TYPE_CHECKING, Optional

from toolboxv2 import get_app

if TYPE_CHECKING:
    from toolboxv2.flows.cli_v4 import ISAA_Host

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
        self._discord_task: Optional[asyncio.Task] = None
        self._connected = False

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

            # Agent holen
            agent = await self.host.isaa_tools.get_agent(self.host.active_agent_name)

            # Interface erstellen
            self.interface = create_discord_interface(
                agent=agent,
                token=token,
                respond_to_mentions_only=True,
                tts_backend="groq",
                tts_voice="Fritz-PlayAI",
                stt_backend="groq",
                language="de",
            )

            # Voice Mode hinzufügen
            self.voice_mode = create_voice_mode(self.interface)

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
            self._discord_task.cancel()
            try:
                await self._discord_task
            except asyncio.CancelledError:
                pass

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
        elif cmd == "channels":
            await self._voice_channels()
        else:
            print_status(f"Unknown voice command: {cmd}", "error")
            await self._show_voice_help()

    async def _show_voice_help(self):
        """Zeigt Voice-Hilfe"""
        print_box_header("Voice Commands", "🎤")
        print_box_content("/discord voice join <channel_id> - Join voice channel", "")
        print_box_content("/discord voice leave             - Leave voice channel", "")
        print_box_content("/discord voice status            - Show voice status", "")
        print_box_content("/discord voice speak <text>      - Speak text via TTS", "")
        print_box_content("/discord voice stop              - Stop speaking", "")
        print_box_content("/discord voice context           - Show conversation", "")
        print_box_content("/discord voice channels          - List voice channels", "")
        print_box_footer()

    async def _voice_join(self, args: list[str]):
        """Join voice channel"""
        if not args:
            print_status("Usage: /discord voice join <channel_id>", "error")
            print_status("Use /discord voice channels to see available channels", "info")
            return

        try:
            channel_id = int(args[0])
        except ValueError:
            print_status("Invalid channel ID", "error")
            return

        import discord

        channel = self.interface.bot.get_channel(channel_id)
        if not channel:
            try:
                channel = await self.interface.bot.fetch_channel(channel_id)
            except:
                print_status(f"Channel {channel_id} not found", "error")
                return

        if not isinstance(channel, discord.VoiceChannel):
            print_status(f"{channel.name} is not a voice channel", "error")
            return

        print_status(f"Joining {channel.name}...", "progress")

        result = await self.voice_mode.voice_handler.join_channel(channel)

        if result.get("success"):
            listening = "✅ Listening enabled" if result.get("listening") else "❌ Listening disabled"
            print_status(f"Joined {result.get('channel_name')}", "success")
            print_status(listening, "info")

            if result.get("warning"):
                print_status(result["warning"], "warning")
        else:
            print_status(f"Failed: {result.get('error')}", "error")

    async def _voice_leave(self):
        """Leave voice channel"""
        handler = self.voice_mode.voice_handler

        # Find connected guild
        guild_id = None
        for gid in handler._voice_clients.keys():
            guild_id = gid
            break

        if not guild_id:
            print_status("Not in any voice channel", "warning")
            return

        result = await handler.leave_channel(guild_id)

        if result.get("success"):
            print_status("Left voice channel", "success")
        else:
            print_status(f"Failed: {result.get('error')}", "error")

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

        print_box_footer()

    async def _voice_speak(self, args: list[str]):
        """Speak text in voice channel"""
        if not args:
            print_status("Usage: /discord voice speak <text>", "error")
            return

        handler = self.voice_mode.voice_handler

        # Find connected guild
        guild_id = None
        for gid in handler._voice_clients.keys():
            guild_id = gid
            break

        if not guild_id:
            print_status("Not in any voice channel", "warning")
            return

        text = " ".join(args)

        print_status(f"Speaking: {text[:50]}...", "progress")

        await handler.speak_streaming(guild_id, text)

        print_status("Queued for playback", "success")

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
        },
    }
