"""
Discord-Specific Tools for ProA Kernel
Version: 1.0.0

Provides Discord-specific tools for server management, user management,
voice control, and lifetime management that are exported to the agent.
"""


from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

try:
    import discord
except ImportError:
    print("pip install discord.py")
    discord = None

class DiscordKernelTools:
    """Discord-specific tools for kernel integration"""

    def __init__(self, bot: 'discord.discord.ext.commands.Bot', kernel, output_router):
        self.bot = bot
        self.kernel = kernel
        self.output_router = output_router

    # ===== SERVER MANAGEMENT =====

    async def get_server_info(self, guild_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get information about a Discord server (guild).

        Args:
            guild_id: Optional guild ID. If None, returns info for all guilds.

        Returns:
            Dict with server information including name, member count, channels, roles, etc.
        """
        if guild_id:
            guild = self.bot.get_guild(guild_id)
            if not guild:
                return {"error": f"Guild {guild_id} not found"}

            return {
                "id": guild.id,
                "name": guild.name,
                "member_count": guild.member_count,
                "owner_id": guild.owner_id,
                "created_at": guild.created_at.isoformat(),
                "text_channels": len(guild.text_channels),
                "voice_channels": len(guild.voice_channels),
                "roles": len(guild.roles),
                "emojis": len(guild.emojis),
                "boost_level": guild.premium_tier,
                "boost_count": guild.premium_subscription_count
            }
        else:
            # Return info for all guilds
            return {
                "guilds": [
                    {
                        "id": g.id,
                        "name": g.name,
                        "member_count": g.member_count
                    }
                    for g in self.bot.guilds
                ],
                "total_guilds": len(self.bot.guilds)
            }

    async def get_channel_info(self, channel_id: int) -> Dict[str, Any]:
        """
        Get information about a Discord channel.

        Args:
            channel_id: Channel ID

        Returns:
            Dict with channel information
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return {"error": f"Channel {channel_id} not found"}

        info = {
            "id": channel.id,
            "name": getattr(channel, 'name', 'DM Channel'),
            "type": str(channel.type),
            "created_at": channel.created_at.isoformat()
        }

        # Add guild-specific info
        if hasattr(channel, 'guild') and channel.guild:
            info["guild_id"] = channel.guild.id
            info["guild_name"] = channel.guild.name

        # Add text channel specific info
        if isinstance(channel, discord.TextChannel):
            info["topic"] = channel.topic
            info["slowmode_delay"] = channel.slowmode_delay
            info["nsfw"] = channel.nsfw

        # Add voice channel specific info
        if isinstance(channel, discord.VoiceChannel):
            info["bitrate"] = channel.bitrate
            info["user_limit"] = channel.user_limit
            info["members"] = [m.display_name for m in channel.members]

        return info

    async def list_channels(self, guild_id: int, channel_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all channels in a guild.

        Args:
            guild_id: Guild ID
            channel_type: Optional filter by type ('text', 'voice', 'category', 'stage')

        Returns:
            List of channel info dicts
        """
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return []

        channels = []
        for channel in guild.channels:
            if channel_type:
                if channel_type == 'text' and not isinstance(channel, discord.TextChannel):
                    continue
                if channel_type == 'voice' and not isinstance(channel, discord.VoiceChannel):
                    continue
                if channel_type == 'category' and not isinstance(channel, discord.CategoryChannel):
                    continue
                if channel_type == 'stage' and not isinstance(channel, discord.StageChannel):
                    continue

            channels.append({
                "id": channel.id,
                "name": channel.name,
                "type": str(channel.type),
                "position": channel.position
            })

        return channels

    async def get_user_info(self, user_id: int, guild_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get information about a Discord user.

        Args:
            user_id: User ID
            guild_id: Optional guild ID for member-specific info

        Returns:
            Dict with user information
        """
        user = self.bot.get_user(user_id)
        if not user:
            return {"error": f"User {user_id} not found"}

        info = {
            "id": user.id,
            "name": user.name,
            "display_name": user.display_name,
            "bot": user.bot,
            "created_at": user.created_at.isoformat()
        }

        # Add member-specific info if guild provided
        if guild_id:
            guild = self.bot.get_guild(guild_id)
            if guild:
                member = guild.get_member(user_id)
                if member:
                    info["nickname"] = member.nick
                    info["joined_at"] = member.joined_at.isoformat() if member.joined_at else None
                    info["roles"] = [role.name for role in member.roles if role.name != "@everyone"]
                    info["top_role"] = member.top_role.name
                    info["voice_channel"] = member.voice.channel.name if member.voice else None

        return info

    # ===== MESSAGE MANAGEMENT =====

    async def send_message(
        self,
        channel_id: int,
        content: str,
        embed: Optional[Dict[str, Any]] = None,
        reply_to: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Send a message to a Discord channel.

        Args:
            channel_id: Channel ID to send message to
            content: Message content (text)
            embed: Optional embed dict with title, description, color, fields
            reply_to: Optional message ID to reply to

        Returns:
            Dict with sent message info (id, channel_id, timestamp)
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return {"error": f"Channel {channel_id} not found"}

        try:
            # Create embed if provided
            discord_embed = None
            if embed:
                discord_embed = discord.Embed(
                    title=embed.get("title"),
                    description=embed.get("description"),
                    color=discord.Color(embed.get("color", 0x3498db))
                )

                # Add fields
                for field in embed.get("fields", []):
                    discord_embed.add_field(
                        name=field.get("name", "Field"),
                        value=field.get("value", ""),
                        inline=field.get("inline", False)
                    )

            # Get reference message if replying
            reference = None
            if reply_to:
                try:
                    ref_msg = await channel.fetch_message(reply_to)
                    reference = ref_msg
                except:
                    pass

            # Send message
            message = await channel.send(
                content=content,
                embed=discord_embed,
                reference=reference
            )

            return {
                "success": True,
                "message_id": message.id,
                "channel_id": message.channel.id,
                "timestamp": message.created_at.isoformat()
            }
        except Exception as e:
            return {"error": str(e)}

    async def edit_message(
        self,
        channel_id: int,
        message_id: int,
        new_content: Optional[str] = None,
        new_embed: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Edit an existing message.

        Args:
            channel_id: Channel ID where message is located
            message_id: Message ID to edit
            new_content: New message content (optional)
            new_embed: New embed dict (optional)

        Returns:
            Dict with success status and edited message info
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return {"error": f"Channel {channel_id} not found"}

        try:
            message = await channel.fetch_message(message_id)

            # Create new embed if provided
            discord_embed = None
            if new_embed:
                discord_embed = discord.Embed(
                    title=new_embed.get("title"),
                    description=new_embed.get("description"),
                    color=discord.Color(new_embed.get("color", 0x3498db))
                )

                for field in new_embed.get("fields", []):
                    discord_embed.add_field(
                        name=field.get("name", "Field"),
                        value=field.get("value", ""),
                        inline=field.get("inline", False)
                    )

            # Edit message
            await message.edit(content=new_content, embed=discord_embed)

            return {
                "success": True,
                "message_id": message.id,
                "edited_at": datetime.now().isoformat()
            }
        except discord.NotFound:
            return {"error": f"Message {message_id} not found"}
        except discord.Forbidden:
            return {"error": "No permission to edit this message"}
        except Exception as e:
            return {"error": str(e)}

    async def delete_message(self, channel_id: int, message_id: int, delay: float = 0) -> Dict[str, Any]:
        """
        Delete a message.

        Args:
            channel_id: Channel ID where message is located
            message_id: Message ID to delete
            delay: Optional delay in seconds before deletion

        Returns:
            Dict with success status
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return {"error": f"Channel {channel_id} not found"}

        try:
            message = await channel.fetch_message(message_id)
            await message.delete(delay=delay)

            return {
                "success": True,
                "message_id": message_id,
                "deleted_at": datetime.now().isoformat()
            }
        except discord.NotFound:
            return {"error": f"Message {message_id} not found"}
        except discord.Forbidden:
            return {"error": "No permission to delete this message"}
        except Exception as e:
            return {"error": str(e)}

    async def get_message(self, channel_id: int, message_id: int) -> Dict[str, Any]:
        """
        Get information about a specific message.

        Args:
            channel_id: Channel ID where message is located
            message_id: Message ID to fetch

        Returns:
            Dict with message information
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return {"error": f"Channel {channel_id} not found"}

        try:
            message = await channel.fetch_message(message_id)

            return {
                "id": message.id,
                "content": message.content,
                "author": {
                    "id": message.author.id,
                    "name": message.author.name,
                    "display_name": message.author.display_name
                },
                "channel_id": message.channel.id,
                "created_at": message.created_at.isoformat(),
                "edited_at": message.edited_at.isoformat() if message.edited_at else None,
                "embeds": len(message.embeds),
                "attachments": [
                    {
                        "filename": att.filename,
                        "url": att.url,
                        "size": att.size
                    }
                    for att in message.attachments
                ],
                "reactions": [
                    {
                        "emoji": str(reaction.emoji),
                        "count": reaction.count
                    }
                    for reaction in message.reactions
                ]
            }
        except discord.NotFound:
            return {"error": f"Message {message_id} not found"}
        except Exception as e:
            return {"error": str(e)}

    async def get_recent_messages(
        self,
        channel_id: int,
        limit: int = 10,
        before: Optional[int] = None,
        after: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent messages from a channel.

        Args:
            channel_id: Channel ID to fetch messages from
            limit: Maximum number of messages to fetch (default 10, max 100)
            before: Fetch messages before this message ID
            after: Fetch messages after this message ID

        Returns:
            List of message info dicts
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return []

        try:
            limit = min(limit, 100)  # Discord API limit

            # Fetch messages
            messages = []
            async for message in channel.history(limit=limit, before=before, after=after):
                messages.append({
                    "id": message.id,
                    "content": message.content,
                    "author": {
                        "id": message.author.id,
                        "name": message.author.name
                    },
                    "created_at": message.created_at.isoformat(),
                    "has_embeds": len(message.embeds) > 0,
                    "has_attachments": len(message.attachments) > 0
                })

            return messages
        except Exception as e:
            return []


    #  ===== Message Reaction Tools =====
    async def get_message_reactions(
        self,
        channel_id: int,
        message_id: int,
        emoji: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get reactions from a message.

        Args:
            channel_id: Channel ID where the message is
            message_id: Message ID
            emoji: Optional specific emoji to get reactions for (e.g., "👍", "custom_emoji_name")

        Returns:
            Dict with reaction data
        """
        try:
            channel = self.bot.get_channel(channel_id)
            if not channel:
                return {"error": f"Channel {channel_id} not found"}

            message = await channel.fetch_message(message_id)

            if not message.reactions:
                return {
                    "success": True,
                    "message_id": message_id,
                    "channel_id": channel_id,
                    "reactions": []
                }

            reactions_data = []

            for reaction in message.reactions:
                # Filter by emoji if specified
                if emoji:
                    # Handle custom emojis
                    if isinstance(reaction.emoji, str):
                        if reaction.emoji != emoji:
                            continue
                    else:  # discord.PartialEmoji or discord.Emoji
                        if reaction.emoji.name != emoji and str(reaction.emoji) != emoji:
                            continue

                # Get users who reacted
                users = []
                async for user in reaction.users():
                    users.append({
                        "id": user.id,
                        "name": user.name,
                        "display_name": user.display_name,
                        "bot": user.bot
                    })

                reaction_info = {
                    "emoji": str(reaction.emoji),
                    "count": reaction.count,
                    "me": reaction.me,  # Whether the bot reacted
                    "users": users
                }

                # Add custom emoji details if applicable
                if isinstance(reaction.emoji, (discord.PartialEmoji, discord.Emoji)):
                    reaction_info["custom"] = True
                    reaction_info["emoji_id"] = reaction.emoji.id
                    reaction_info["emoji_name"] = reaction.emoji.name
                    reaction_info["animated"] = reaction.emoji.animated
                else:
                    reaction_info["custom"] = False

                reactions_data.append(reaction_info)

            return {
                "success": True,
                "message_id": message_id,
                "channel_id": channel_id,
                "message_content": message.content[:100] + "..." if len(message.content) > 100 else message.content,
                "author": {
                    "id": message.author.id,
                    "name": message.author.name
                },
                "reactions": reactions_data,
                "total_reactions": sum(r["count"] for r in reactions_data)
            }

        except discord.NotFound:
            return {"error": f"Message {message_id} not found in channel {channel_id}"}
        except discord.Forbidden:
            return {"error": "Missing permissions to access this channel or message"}
        except Exception as e:
            return {"error": str(e)}

    async def add_reaction(self, channel_id: int, message_id: int, emoji: str) -> Dict[str, Any]:
        """
        Add a reaction to a message.

        Args:
            channel_id: Channel ID where message is located
            message_id: Message ID to react to
            emoji: Emoji to add (unicode or custom emoji name)

        Returns:
            Dict with success status
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return {"error": f"Channel {channel_id} not found"}

        try:
            message = await channel.fetch_message(message_id)
            await message.add_reaction(emoji)

            return {
                "success": True,
                "message_id": message_id,
                "emoji": emoji
            }
        except discord.NotFound:
            return {"error": f"Message {message_id} not found"}
        except discord.HTTPException as e:
            return {"error": f"Invalid emoji or HTTP error: {e}"}
        except Exception as e:
            return {"error": str(e)}

    async def remove_reaction(
        self,
        channel_id: int,
        message_id: int,
        emoji: str,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Remove a reaction from a message.

        Args:
            channel_id: Channel ID where message is located
            message_id: Message ID to remove reaction from
            emoji: Emoji to remove
            user_id: Optional user ID (if None, removes bot's reaction)

        Returns:
            Dict with success status
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return {"error": f"Channel {channel_id} not found"}

        try:
            message = await channel.fetch_message(message_id)

            if user_id:
                user = self.bot.get_user(user_id)
                if user:
                    await message.remove_reaction(emoji, user)
            else:
                await message.remove_reaction(emoji, self.bot.user)

            return {
                "success": True,
                "message_id": message_id,
                "emoji": emoji
            }
        except discord.NotFound:
            return {"error": f"Message {message_id} not found"}
        except Exception as e:
            return {"error": str(e)}

    # ===== VOICE CONTROL =====

    async def join_voice_channel(self, channel_id: int) -> Dict[str, Any]:
        """
        Join a voice channel.

        Args:
            channel_id: Voice channel ID to join

        Returns:
            Dict with success status and voice client info
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return {"error": f"Channel {channel_id} not found"}

        if not isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
            return {"error": "Channel is not a voice channel"}

        try:
            # Check if already in a voice channel in this guild
            if channel.guild:
                existing_vc = channel.guild.voice_client
                if existing_vc:
                    await existing_vc.move_to(channel)
                    return {
                        "success": True,
                        "action": "moved",
                        "channel_id": channel.id,
                        "channel_name": channel.name
                    }

            # Connect to voice channel
            voice_client = await channel.connect()

            # Store voice client
            if channel.guild:
                self.output_router.voice_clients[channel.guild.id] = voice_client

            return {
                "success": True,
                "action": "joined",
                "channel_id": channel.id,
                "channel_name": channel.name
            }
        except Exception as e:
            return {"error": str(e)}

    async def leave_voice_channel(self, guild_id: int) -> Dict[str, Any]:
        """
        Leave the current voice channel in a guild.

        Args:
            guild_id: Guild ID to leave voice channel from

        Returns:
            Dict with success status
        """
        if guild_id not in self.output_router.voice_clients:
            return {"error": "Not in a voice channel in this guild"}

        try:
            voice_client = self.output_router.voice_clients[guild_id]
            await voice_client.disconnect()

            # Cleanup
            del self.output_router.voice_clients[guild_id]
            if guild_id in self.output_router.audio_sinks:
                del self.output_router.audio_sinks[guild_id]
            if guild_id in self.output_router.tts_enabled:
                del self.output_router.tts_enabled[guild_id]

            return {
                "success": True,
                "guild_id": guild_id
            }
        except Exception as e:
            return {"error": str(e)}

    async def get_voice_status(self, guild_id: int) -> Dict[str, Any]:
        """
        Get voice connection status for a guild.

        Args:
            guild_id: Guild ID to check

        Returns:
            Dict with voice status information
        """
        if guild_id not in self.output_router.voice_clients:
            return {
                "connected": False,
                "guild_id": guild_id
            }

        voice_client = self.output_router.voice_clients[guild_id]

        return {
            "connected": voice_client.is_connected(),
            "channel_id": voice_client.channel.id if voice_client.channel else None,
            "channel_name": voice_client.channel.name if voice_client.channel else None,
            "playing": voice_client.is_playing(),
            "paused": voice_client.is_paused(),
            "listening": voice_client.is_listening() if hasattr(voice_client, 'is_listening') else False,
            "tts_enabled": self.output_router.tts_enabled.get(guild_id, False),
            "tts_mode": self.output_router.tts_mode.get(guild_id, "piper"),
            "latency": voice_client.latency,
            "guild_id": guild_id
        }

    async def toggle_tts(self, guild_id: int, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Toggle TTS (Text-to-Speech) on/off.

        Args:
            guild_id: Guild ID
            mode: TTS mode ('elevenlabs', 'piper', 'off', or None to toggle)

        Returns:
            Dict with TTS status
        """
        if mode == "off":
            self.output_router.tts_enabled[guild_id] = False
            return {
                "success": True,
                "tts_enabled": False,
                "guild_id": guild_id
            }
        elif mode in ["elevenlabs", "piper"]:
            self.output_router.tts_enabled[guild_id] = True
            self.output_router.tts_mode[guild_id] = mode
            return {
                "success": True,
                "tts_enabled": True,
                "tts_mode": mode,
                "guild_id": guild_id
            }
        elif mode is None:
            # Toggle
            current = self.output_router.tts_enabled.get(guild_id, False)
            self.output_router.tts_enabled[guild_id] = not current
            return {
                "success": True,
                "tts_enabled": not current,
                "tts_mode": self.output_router.tts_mode.get(guild_id, "piper"),
                "guild_id": guild_id
            }
        else:
            return {"error": f"Invalid TTS mode: {mode}"}

    async def send_tts_message(self, guild_id: int, text: str, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a TTS (Text-to-Speech) message in the current voice channel.

        Args:
            guild_id: Guild ID where the bot is in a voice channel
            text: Text to speak via TTS
            mode: TTS mode ('elevenlabs' or 'piper', defaults to current mode)

        Returns:
            Dict with success status and TTS info
        """
        # Check if bot is in voice channel
        if guild_id not in self.output_router.voice_clients:
            return {"error": "Not in a voice channel in this guild. Use discord_join_voice first."}

        voice_client = self.output_router.voice_clients[guild_id]
        if not voice_client.is_connected():
            return {"error": "Voice client is not connected"}

        # Determine TTS mode
        tts_mode = mode or self.output_router.tts_mode.get(guild_id, "piper")
        if tts_mode not in ["elevenlabs", "piper"]:
            return {"error": f"Invalid TTS mode: {tts_mode}. Use 'elevenlabs' or 'piper'."}

        try:
            # Enable TTS temporarily if not enabled
            was_enabled = self.output_router.tts_enabled.get(guild_id, False)
            original_mode = self.output_router.tts_mode.get(guild_id, "piper")

            self.output_router.tts_enabled[guild_id] = True
            self.output_router.tts_mode[guild_id] = tts_mode

            # Send TTS message via output router
            await self.output_router.send_tts(guild_id, text)

            # Restore original TTS settings
            if not was_enabled:
                self.output_router.tts_enabled[guild_id] = False
            self.output_router.tts_mode[guild_id] = original_mode

            return {
                "success": True,
                "text": text,
                "tts_mode": tts_mode,
                "guild_id": guild_id,
                "channel_id": voice_client.channel.id,
                "channel_name": voice_client.channel.name
            }
        except Exception as e:
            return {"error": f"Failed to send TTS message: {str(e)}"}

    async def can_hear_user(self, guild_id: int, user_id: int) -> Dict[str, Any]:
        """
        Check if the bot can hear a specific user (voice listening status).

        Args:
            guild_id: Guild ID
            user_id: User ID to check

        Returns:
            Dict with hearing status and details
        """
        # Check if bot is in voice channel
        if guild_id not in self.output_router.voice_clients:
            return {
                "can_hear": False,
                "reason": "Not in a voice channel",
                "guild_id": guild_id,
                "user_id": user_id
            }

        voice_client = self.output_router.voice_clients[guild_id]
        if not voice_client.is_connected():
            return {
                "can_hear": False,
                "reason": "Voice client not connected",
                "guild_id": guild_id,
                "user_id": user_id
            }

        # Check if listening is enabled
        is_listening = voice_client.is_listening() if hasattr(voice_client, 'is_listening') else False
        if not is_listening:
            return {
                "can_hear": False,
                "reason": "Voice listening is not enabled. Use !listen command to start listening.",
                "guild_id": guild_id,
                "user_id": user_id,
                "voice_channel": voice_client.channel.name
            }

        # Get guild and user
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return {
                "can_hear": False,
                "reason": "Guild not found",
                "guild_id": guild_id,
                "user_id": user_id
            }

        member = guild.get_member(user_id)
        if not member:
            return {
                "can_hear": False,
                "reason": "User not found in guild",
                "guild_id": guild_id,
                "user_id": user_id
            }

        # Check if user is in the same voice channel
        if not member.voice or not member.voice.channel:
            return {
                "can_hear": False,
                "reason": "User is not in a voice channel",
                "guild_id": guild_id,
                "user_id": user_id,
                "bot_voice_channel": voice_client.channel.name
            }

        if member.voice.channel.id != voice_client.channel.id:
            return {
                "can_hear": False,
                "reason": "User is in a different voice channel",
                "guild_id": guild_id,
                "user_id": user_id,
                "bot_voice_channel": voice_client.channel.name,
                "user_voice_channel": member.voice.channel.name
            }

        # Check if user is muted
        if member.voice.self_mute or member.voice.mute:
            return {
                "can_hear": False,
                "reason": "User is muted",
                "guild_id": guild_id,
                "user_id": user_id,
                "voice_channel": voice_client.channel.name,
                "self_mute": member.voice.self_mute,
                "server_mute": member.voice.mute
            }

        # All checks passed - can hear user!
        return {
            "can_hear": True,
            "guild_id": guild_id,
            "user_id": user_id,
            "user_name": member.display_name,
            "voice_channel": voice_client.channel.name,
            "voice_channel_id": voice_client.channel.id,
            "listening": True,
            "users_in_channel": [m.display_name for m in voice_client.channel.members if not m.bot]
        }

    # ===== ROLE & PERMISSION MANAGEMENT =====

    async def get_member_roles(self, guild_id: int, user_id: int) -> List[Dict[str, Any]]:
        """
        Get all roles of a member in a guild.

        Args:
            guild_id: Guild ID
            user_id: User ID

        Returns:
            List of role info dicts
        """
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return []

        member = guild.get_member(user_id)
        if not member:
            return []

        return [
            {
                "id": role.id,
                "name": role.name,
                "color": role.color.value,
                "position": role.position,
                "permissions": role.permissions.value
            }
            for role in member.roles
            if role.name != "@everyone"
        ]

    async def add_role(self, guild_id: int, user_id: int, role_id: int, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a role to a member.

        Args:
            guild_id: Guild ID
            user_id: User ID
            role_id: Role ID to add
            reason: Optional reason for audit log

        Returns:
            Dict with success status
        """
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return {"error": f"Guild {guild_id} not found"}

        member = guild.get_member(user_id)
        if not member:
            return {"error": f"Member {user_id} not found"}

        role = guild.get_role(role_id)
        if not role:
            return {"error": f"Role {role_id} not found"}

        try:
            await member.add_roles(role, reason=reason)
            return {
                "success": True,
                "user_id": user_id,
                "role_id": role_id,
                "role_name": role.name
            }
        except discord.Forbidden:
            return {"error": "No permission to add this role"}
        except Exception as e:
            return {"error": str(e)}

    async def remove_role(self, guild_id: int, user_id: int, role_id: int, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Remove a role from a member.

        Args:
            guild_id: Guild ID
            user_id: User ID
            role_id: Role ID to remove
            reason: Optional reason for audit log

        Returns:
            Dict with success status
        """
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return {"error": f"Guild {guild_id} not found"}

        member = guild.get_member(user_id)
        if not member:
            return {"error": f"Member {user_id} not found"}

        role = guild.get_role(role_id)
        if not role:
            return {"error": f"Role {role_id} not found"}

        try:
            await member.remove_roles(role, reason=reason)
            return {
                "success": True,
                "user_id": user_id,
                "role_id": role_id,
                "role_name": role.name
            }
        except discord.Forbidden:
            return {"error": "No permission to remove this role"}
        except Exception as e:
            return {"error": str(e)}

    # ===== LIFETIME MANAGEMENT =====

    async def get_bot_status(self) -> Dict[str, Any]:
        """
        Get current bot status and statistics.

        Returns:
            Dict with bot status information
        """
        return {
            "bot_id": self.bot.user.id,
            "bot_name": self.bot.user.name,
            "latency": round(self.bot.latency * 1000, 2),  # ms
            "guilds": len(self.bot.guilds),
            "users": sum(g.member_count for g in self.bot.guilds),
            "voice_connections": len(self.output_router.voice_clients),
            "uptime": "N/A",  # Would need to track start time
            "kernel_state": str(self.kernel.state)
        }

    async def set_bot_status(
        self,
        status: str = "online",
        activity_type: str = "playing",
        activity_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Set bot's Discord status and activity.

        Args:
            status: Status ('online', 'idle', 'dnd', 'invisible')
            activity_type: Activity type ('playing', 'watching', 'listening', 'streaming')
            activity_name: Activity name/text

        Returns:
            Dict with success status
        """
        try:
            # Map status string to discord.Status
            status_map = {
                "online": discord.Status.online,
                "idle": discord.Status.idle,
                "dnd": discord.Status.dnd,
                "invisible": discord.Status.invisible
            }

            discord_status = status_map.get(status, discord.Status.online)

            # Create activity
            activity = None
            if activity_name:
                if activity_type == "playing":
                    activity = discord.Game(name=activity_name)
                elif activity_type == "watching":
                    activity = discord.Activity(type=discord.ActivityType.watching, name=activity_name)
                elif activity_type == "listening":
                    activity = discord.Activity(type=discord.ActivityType.listening, name=activity_name)
                elif activity_type == "streaming":
                    activity = discord.Streaming(name=activity_name, url="https://twitch.tv/placeholder")

            # Update presence
            await self.bot.change_presence(status=discord_status, activity=activity)

            return {
                "success": True,
                "status": status,
                "activity_type": activity_type,
                "activity_name": activity_name
            }
        except Exception as e:
            return {"error": str(e)}

    async def get_kernel_metrics(self) -> Dict[str, Any]:
        """
        Get kernel performance metrics.

        Returns:
            Dict with kernel metrics
        """
        metrics = self.kernel.metrics
        return {
            "total_signals": metrics.total_signals,
            "user_inputs": metrics.user_inputs,
            "agent_responses": metrics.agent_responses,
            "proactive_actions": metrics.proactive_actions,
            "scheduled_tasks": metrics.scheduled_tasks,
            "errors": metrics.errors,
            "avg_response_time": round(metrics.avg_response_time, 3) if metrics.avg_response_time else 0
        }

    # ===== SERVER SETUP & MANAGEMENT =====

    async def create_server(
        self,
        name: str,
        icon: Optional[str] = None,
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new Discord server (guild).

        Args:
            name: Server name
            icon: Optional base64 encoded icon
            region: Optional voice region

        Returns:
            Dict with server info
        """
        try:
            guild = await self.bot.create_guild(name=name, icon=icon, region=region)
            return {
                "success": True,
                "guild_id": guild.id,
                "guild_name": guild.name,
                "created_at": guild.created_at.isoformat()
            }
        except Exception as e:
            return {"error": str(e)}

    async def delete_server(self, guild_id: int) -> Dict[str, Any]:
        """
        Delete a Discord server (only if bot is owner).

        Args:
            guild_id: Guild ID to delete

        Returns:
            Dict with success status
        """
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return {"error": f"Guild {guild_id} not found"}

        try:
            await guild.delete()
            return {
                "success": True,
                "guild_id": guild_id
            }
        except discord.Forbidden:
            return {"error": "Bot must be server owner to delete"}
        except Exception as e:
            return {"error": str(e)}

    async def edit_server(
        self,
        guild_id: int,
        name: Optional[str] = None,
        icon: Optional[str] = None,
        description: Optional[str] = None,
        verification_level: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Edit server settings.

        Args:
            guild_id: Guild ID
            name: New server name
            icon: New icon (base64)
            description: New description
            verification_level: Verification level (0-4)

        Returns:
            Dict with success status
        """
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return {"error": f"Guild {guild_id} not found"}

        try:
            kwargs = {}
            if name: kwargs['name'] = name
            if icon: kwargs['icon'] = icon
            if description: kwargs['description'] = description
            if verification_level is not None:
                kwargs['verification_level'] = discord.VerificationLevel(str(verification_level))

            await guild.edit(**kwargs)
            return {
                "success": True,
                "guild_id": guild_id
            }
        except Exception as e:
            return {"error": str(e)}

    # ===== CHANNEL MANAGEMENT =====

    async def create_channel(
        self,
        guild_id: int,
        name: str,
        channel_type: str = "text",
        category_id: Optional[int] = None,
        topic: Optional[str] = None,
        slowmode_delay: int = 0,
        nsfw: bool = False
    ) -> Dict[str, Any]:
        """
        Create a new channel.

        Args:
            guild_id: Guild ID
            name: Channel name
            channel_type: 'text', 'voice', 'category', 'stage'
            category_id: Parent category ID
            topic: Channel topic (text channels)
            slowmode_delay: Slowmode in seconds
            nsfw: NSFW flag

        Returns:
            Dict with channel info
        """
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return {"error": f"Guild {guild_id} not found"}

        try:
            category = guild.get_channel(category_id) if category_id else None

            if channel_type == "text":
                channel = await guild.create_text_channel(
                    name=name,
                    category=category,
                    topic=topic,
                    slowmode_delay=slowmode_delay,
                    nsfw=nsfw
                )
            elif channel_type == "voice":
                channel = await guild.create_voice_channel(
                    name=name,
                    category=category
                )
            elif channel_type == "category":
                channel = await guild.create_category(name=name)
            elif channel_type == "stage":
                channel = await guild.create_stage_channel(
                    name=name,
                    category=category
                )
            else:
                return {"error": f"Invalid channel type: {channel_type}"}

            return {
                "success": True,
                "channel_id": channel.id,
                "channel_name": channel.name,
                "channel_type": str(channel.type)
            }
        except Exception as e:
            return {"error": str(e)}

    async def delete_channel(self, channel_id: int, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete a channel.

        Args:
            channel_id: Channel ID
            reason: Audit log reason

        Returns:
            Dict with success status
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return {"error": f"Channel {channel_id} not found"}

        try:
            await channel.delete(reason=reason)
            return {
                "success": True,
                "channel_id": channel_id
            }
        except Exception as e:
            return {"error": str(e)}

    async def edit_channel(
        self,
        channel_id: int,
        name: Optional[str] = None,
        topic: Optional[str] = None,
        slowmode_delay: Optional[int] = None,
        nsfw: Optional[bool] = None,
        position: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Edit channel settings.

        Args:
            channel_id: Channel ID
            name: New name
            topic: New topic
            slowmode_delay: Slowmode seconds
            nsfw: NSFW flag
            position: Channel position

        Returns:
            Dict with success status
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return {"error": f"Channel {channel_id} not found"}

        try:
            kwargs = {}
            if name: kwargs['name'] = name
            if position is not None: kwargs['position'] = position

            if isinstance(channel, discord.TextChannel):
                if topic is not None: kwargs['topic'] = topic
                if slowmode_delay is not None: kwargs['slowmode_delay'] = slowmode_delay
                if nsfw is not None: kwargs['nsfw'] = nsfw

            await channel.edit(**kwargs)
            return {
                "success": True,
                "channel_id": channel_id
            }
        except Exception as e:
            return {"error": str(e)}

    # ===== THREAD MANAGEMENT =====

    async def create_thread(
        self,
        channel_id: int,
        name: str,
        message_id: Optional[int] = None,
        auto_archive_duration: int = 1440
    ) -> Dict[str, Any]:
        """
        Create a thread in a channel.

        Args:
            channel_id: Channel ID
            name: Thread name
            message_id: Message to create thread from (optional)
            auto_archive_duration: Auto-archive in minutes (60, 1440, 4320, 10080)

        Returns:
            Dict with thread info
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return {"error": f"Channel {channel_id} not found"}

        try:
            if message_id:
                message = await channel.fetch_message(message_id)
                thread = await message.create_thread(
                    name=name,
                    auto_archive_duration=auto_archive_duration
                )
            else:
                thread = await channel.create_thread(
                    name=name,
                    auto_archive_duration=auto_archive_duration
                )

            return {
                "success": True,
                "thread_id": thread.id,
                "thread_name": thread.name
            }
        except Exception as e:
            return {"error": str(e)}

    async def join_thread(self, thread_id: int) -> Dict[str, Any]:
        """Join a thread."""
        thread = self.bot.get_channel(thread_id)
        if not thread or not isinstance(thread, discord.Thread):
            return {"error": "Thread not found"}

        try:
            await thread.join()
            return {"success": True, "thread_id": thread_id}
        except Exception as e:
            return {"error": str(e)}

    async def leave_thread(self, thread_id: int) -> Dict[str, Any]:
        """Leave a thread."""
        thread = self.bot.get_channel(thread_id)
        if not thread or not isinstance(thread, discord.Thread):
            return {"error": "Thread not found"}

        try:
            await thread.leave()
            return {"success": True, "thread_id": thread_id}
        except Exception as e:
            return {"error": str(e)}

    # ===== MODERATION =====

    async def kick_member(self, guild_id: int, user_id: int, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Kick a member from the server.

        Args:
            guild_id: Guild ID
            user_id: User ID to kick
            reason: Audit log reason

        Returns:
            Dict with success status
        """
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return {"error": f"Guild {guild_id} not found"}

        member = guild.get_member(user_id)
        if not member:
            return {"error": f"Member {user_id} not found"}

        try:
            await member.kick(reason=reason)
            return {
                "success": True,
                "user_id": user_id,
                "action": "kicked"
            }
        except discord.Forbidden:
            return {"error": "No permission to kick"}
        except Exception as e:
            return {"error": str(e)}

    async def ban_member(
        self,
        guild_id: int,
        user_id: int,
        reason: Optional[str] = None,
        delete_message_days: int = 0
    ) -> Dict[str, Any]:
        """
        Ban a member from the server.

        Args:
            guild_id: Guild ID
            user_id: User ID to ban
            reason: Audit log reason
            delete_message_days: Days of messages to delete (0-7)

        Returns:
            Dict with success status
        """
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return {"error": f"Guild {guild_id} not found"}

        try:
            user = await self.bot.fetch_user(user_id)
            await guild.ban(user, reason=reason, delete_message_days=delete_message_days)
            return {
                "success": True,
                "user_id": user_id,
                "action": "banned"
            }
        except discord.Forbidden:
            return {"error": "No permission to ban"}
        except Exception as e:
            return {"error": str(e)}

    async def unban_member(self, guild_id: int, user_id: int, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Unban a member.

        Args:
            guild_id: Guild ID
            user_id: User ID to unban
            reason: Audit log reason

        Returns:
            Dict with success status
        """
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return {"error": f"Guild {guild_id} not found"}

        try:
            user = await self.bot.fetch_user(user_id)
            await guild.unban(user, reason=reason)
            return {
                "success": True,
                "user_id": user_id,
                "action": "unbanned"
            }
        except Exception as e:
            return {"error": str(e)}

    async def timeout_member(
        self,
        guild_id: int,
        user_id: int,
        duration_minutes: int,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Timeout (mute) a member.

        Args:
            guild_id: Guild ID
            user_id: User ID
            duration_minutes: Timeout duration in minutes (max 40320 = 28 days)
            reason: Audit log reason

        Returns:
            Dict with success status
        """
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return {"error": f"Guild {guild_id} not found"}

        member = guild.get_member(user_id)
        if not member:
            return {"error": f"Member {user_id} not found"}

        try:
            duration = timedelta(minutes=duration_minutes)
            await member.timeout(duration, reason=reason)
            return {
                "success": True,
                "user_id": user_id,
                "timeout_until": (datetime.now() + duration).isoformat()
            }
        except Exception as e:
            return {"error": str(e)}

    async def remove_timeout(self, guild_id: int, user_id: int, reason: Optional[str] = None) -> Dict[str, Any]:
        """Remove timeout from member."""
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return {"error": f"Guild {guild_id} not found"}

        member = guild.get_member(user_id)
        if not member:
            return {"error": f"Member {user_id} not found"}

        try:
            await member.timeout(None, reason=reason)
            return {
                "success": True,
                "user_id": user_id,
                "action": "timeout_removed"
            }
        except Exception as e:
            return {"error": str(e)}

    async def change_nickname(
        self,
        guild_id: int,
        user_id: int,
        nickname: Optional[str],
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Change a member's nickname.

        Args:
            guild_id: Guild ID
            user_id: User ID
            nickname: New nickname (None to remove)
            reason: Audit log reason

        Returns:
            Dict with success status
        """
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return {"error": f"Guild {guild_id} not found"}

        member = guild.get_member(user_id)
        if not member:
            return {"error": f"Member {user_id} not found"}

        try:
            await member.edit(nick=nickname, reason=reason)
            return {
                "success": True,
                "user_id": user_id,
                "nickname": nickname
            }
        except Exception as e:
            return {"error": str(e)}

    async def move_member(self, guild_id: int, user_id: int, channel_id: int) -> Dict[str, Any]:
        """
        Move member to different voice channel.

        Args:
            guild_id: Guild ID
            user_id: User ID
            channel_id: Target voice channel ID

        Returns:
            Dict with success status
        """
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return {"error": f"Guild {guild_id} not found"}

        member = guild.get_member(user_id)
        if not member:
            return {"error": f"Member {user_id} not found"}

        channel = guild.get_channel(channel_id)
        if not channel or not isinstance(channel, discord.VoiceChannel):
            return {"error": "Invalid voice channel"}

        try:
            await member.move_to(channel)
            return {
                "success": True,
                "user_id": user_id,
                "channel_id": channel_id
            }
        except Exception as e:
            return {"error": str(e)}

    async def disconnect_member(self, guild_id: int, user_id: int) -> Dict[str, Any]:
        """Disconnect member from voice channel."""
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return {"error": f"Guild {guild_id} not found"}

        member = guild.get_member(user_id)
        if not member:
            return {"error": f"Member {user_id} not found"}

        try:
            await member.move_to(None)
            return {
                "success": True,
                "user_id": user_id,
                "action": "disconnected"
            }
        except Exception as e:
            return {"error": str(e)}

    # ===== FILE & EMBED MANAGEMENT =====

    async def send_file(
        self,
        channel_id: int,
        file_path: str,
        filename: Optional[str] = None,
        content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a file to a channel.

        Args:
            channel_id: Channel ID
            file_path: Path to file
            filename: Optional filename override
            content: Optional message content

        Returns:
            Dict with message info
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return {"error": f"Channel {channel_id} not found"}

        try:
            file = discord.File(file_path, filename=filename)
            message = await channel.send(content=content, file=file)
            return {
                "success": True,
                "message_id": message.id,
                "channel_id": channel_id
            }
        except Exception as e:
            return {"error": str(e)}

    # ===== PERMISSIONS =====

    async def set_channel_permissions(
        self,
        channel_id: int,
        target_id: int,
        target_type: str,
        allow: Optional[int] = None,
        deny: Optional[int] = None,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Set channel permissions for role or member.

        Args:
            channel_id: Channel ID
            target_id: Role or member ID
            target_type: 'role' or 'member'
            allow: Permissions to allow (bitfield)
            deny: Permissions to deny (bitfield)
            reason: Audit log reason

        Returns:
            Dict with success status
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return {"error": f"Channel {channel_id} not found"}

        try:
            if target_type == "role":
                target = channel.guild.get_role(target_id)
            elif target_type == "member":
                target = channel.guild.get_member(target_id)
            else:
                return {"error": "target_type must be 'role' or 'member'"}

            if not target:
                return {"error": f"Target {target_id} not found"}

            overwrite = discord.PermissionOverwrite()
            if allow:
                overwrite.update(**{p: True for p, v in discord.Permissions(allow) if v})
            if deny:
                overwrite.update(**{p: False for p, v in discord.Permissions(deny) if v})

            await channel.set_permissions(target, overwrite=overwrite, reason=reason)
            return {
                "success": True,
                "channel_id": channel_id,
                "target_id": target_id
            }
        except Exception as e:
            return {"error": str(e)}

    # ===== DM SUPPORT =====

    async def send_dm(
        self,
        user_id: int,
        content: str,
        embed: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a DM to a user.

        Args:
            user_id: User ID
            content: Message content
            embed: Optional embed dict

        Returns:
            Dict with success status
        """
        try:
            user = await self.bot.fetch_user(user_id)

            discord_embed = None
            if embed:
                discord_embed = discord.Embed(
                    title=embed.get("title"),
                    description=embed.get("description"),
                    color=discord.Color(embed.get("color", 0x3498db))
                )

            message = await user.send(content=content, embed=discord_embed)
            return {
                "success": True,
                "message_id": message.id,
                "user_id": user_id
            }
        except discord.Forbidden:
            return {"error": "Cannot send DM to this user (blocked or privacy settings)"}
        except Exception as e:
            return {"error": str(e)}

    # ===== WEBHOOK MANAGEMENT =====

    async def create_webhook(
        self,
        channel_id: int,
        name: str,
        avatar: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Create a webhook.

        Args:
            channel_id: Channel ID
            name: Webhook name
            avatar: Optional avatar bytes

        Returns:
            Dict with webhook info
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return {"error": f"Channel {channel_id} not found"}

        try:
            webhook = await channel.create_webhook(name=name, avatar=avatar)
            return {
                "success": True,
                "webhook_id": webhook.id,
                "webhook_url": webhook.url,
                "webhook_name": webhook.name
            }
        except Exception as e:
            return {"error": str(e)}

    # ===== INVITATION MANAGEMENT =====

    async def create_invite(
        self,
        channel_id: int,
        max_age: int = 86400,
        max_uses: int = 0,
        temporary: bool = False,
        unique: bool = True,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an invitation link for a channel/server.

        Args:
            channel_id: Channel ID to create invite for
            max_age: Time in seconds until invite expires (0 = never, default 86400 = 24h)
            max_uses: Max number of uses (0 = unlimited)
            temporary: Whether members get temporary membership
            unique: Create a unique invite (if False, may return existing similar invite)
            reason: Audit log reason

        Returns:
            Dict with invite code, URL, and settings
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return {"error": f"Channel {channel_id} not found"}

        try:
            invite = await channel.create_invite(
                max_age=max_age,
                max_uses=max_uses,
                temporary=temporary,
                unique=unique,
                reason=reason
            )

            return {
                "success": True,
                "invite_code": invite.code,
                "invite_url": invite.url,
                "channel_id": channel_id,
                "channel_name": channel.name,
                "guild_id": channel.guild.id if hasattr(channel, 'guild') else None,
                "guild_name": channel.guild.name if hasattr(channel, 'guild') else None,
                "max_age": max_age,
                "max_uses": max_uses,
                "temporary": temporary,
                "created_at": invite.created_at.isoformat() if invite.created_at else None,
                "expires_at": (invite.created_at + timedelta(
                    seconds=max_age)).isoformat() if invite.created_at and max_age > 0 else None
            }
        except discord.Forbidden:
            return {"error": "No permission to create invites"}
        except Exception as e:
            return {"error": str(e)}

    async def get_invites(self, guild_id: int) -> List[Dict[str, Any]]:
        """
        Get all invites for a server.

        Args:
            guild_id: Guild ID

        Returns:
            List of invite info dicts
        """
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return []

        try:
            invites = await guild.invites()

            return [
                {
                    "code": invite.code,
                    "url": invite.url,
                    "channel_id": invite.channel.id if invite.channel else None,
                    "channel_name": invite.channel.name if invite.channel else None,
                    "inviter_id": invite.inviter.id if invite.inviter else None,
                    "inviter_name": invite.inviter.name if invite.inviter else None,
                    "uses": invite.uses,
                    "max_uses": invite.max_uses,
                    "max_age": invite.max_age,
                    "temporary": invite.temporary,
                    "created_at": invite.created_at.isoformat() if invite.created_at else None,
                    "expires_at": invite.expires_at.isoformat() if invite.expires_at else None
                }
                for invite in invites
            ]
        except discord.Forbidden:
            return []
        except Exception as e:
            return []

    async def delete_invite(self, invite_code: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete/revoke an invite.

        Args:
            invite_code: Invite code (not full URL, just the code part)
            reason: Audit log reason

        Returns:
            Dict with success status
        """
        try:
            invite = await self.bot.fetch_invite(invite_code)
            await invite.delete(reason=reason)

            return {
                "success": True,
                "invite_code": invite_code,
                "action": "deleted"
            }
        except discord.NotFound:
            return {"error": f"Invite {invite_code} not found"}
        except discord.Forbidden:
            return {"error": "No permission to delete this invite"}
        except Exception as e:
            return {"error": str(e)}

    async def get_invite_info(self, invite_code: str) -> Dict[str, Any]:
        """
        Get information about an invite without joining.

        Args:
            invite_code: Invite code

        Returns:
            Dict with invite information
        """
        try:
            invite = await self.bot.fetch_invite(invite_code, with_counts=True, with_expiration=True)

            return {
                "code": invite.code,
                "url": invite.url,
                "guild_id": invite.guild.id if invite.guild else None,
                "guild_name": invite.guild.name if invite.guild else None,
                "channel_id": invite.channel.id if invite.channel else None,
                "channel_name": invite.channel.name if invite.channel else None,
                "inviter_id": invite.inviter.id if invite.inviter else None,
                "inviter_name": invite.inviter.name if invite.inviter else None,
                "approximate_member_count": invite.approximate_member_count,
                "approximate_presence_count": invite.approximate_presence_count,
                "expires_at": invite.expires_at.isoformat() if invite.expires_at else None,
                "created_at": invite.created_at.isoformat() if invite.created_at else None
            }
        except discord.NotFound:
            return {"error": f"Invite {invite_code} not found or expired"}
        except Exception as e:
            return {"error": str(e)}

    # ===== TEMPLATE MESSAGE MANAGEMENT =====

    async def create_message_template(
        self,
        template_name: str,
        content: Optional[str] = None,
        embed: Optional[Dict[str, Any]] = None,
        components: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Create a reusable message template.

        Args:
            template_name: Unique name for the template
            content: Message text content
            embed: Embed configuration dict
            components: List of components (buttons, select menus)

        Returns:
            Dict with template info
        """
        # Store templates in kernel memory or local storage
        if not hasattr(self, 'message_templates'):
            self.message_templates = {}

        template = {
            "name": template_name,
            "content": content,
            "embed": embed,
            "components": components,
            "created_at": datetime.now().isoformat()
        }

        self.message_templates[template_name] = template

        return {
            "success": True,
            "template_name": template_name,
            "has_content": content is not None,
            "has_embed": embed is not None,
            "has_components": components is not None and len(components) > 0
        }

    async def get_message_template(self, template_name: str) -> Dict[str, Any]:
        """
        Get a message template by name.

        Args:
            template_name: Template name

        Returns:
            Dict with template data
        """
        if not hasattr(self, 'message_templates'):
            self.message_templates = {}

        if template_name not in self.message_templates:
            return {"error": f"Template '{template_name}' not found"}

        return {
            "success": True,
            "template": self.message_templates[template_name]
        }

    async def list_message_templates(self) -> List[Dict[str, Any]]:
        """
        List all available message templates.

        Returns:
            List of template names and info
        """
        if not hasattr(self, 'message_templates'):
            self.message_templates = {}

        return [
            {
                "name": name,
                "has_content": template.get("content") is not None,
                "has_embed": template.get("embed") is not None,
                "has_components": template.get("components") is not None,
                "created_at": template.get("created_at")
            }
            for name, template in self.message_templates.items()
        ]

    async def delete_message_template(self, template_name: str) -> Dict[str, Any]:
        """
        Delete a message template.

        Args:
            template_name: Template name

        Returns:
            Dict with success status
        """
        if not hasattr(self, 'message_templates'):
            self.message_templates = {}

        if template_name not in self.message_templates:
            return {"error": f"Template '{template_name}' not found"}

        del self.message_templates[template_name]

        return {
            "success": True,
            "template_name": template_name,
            "action": "deleted"
        }

    async def send_template_message(
        self,
        channel_id: int,
        template_name: str,
        variables: Optional[Dict[str, str]] = None,
        reply_to: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Send a message using a template with variable substitution.

        Args:
            channel_id: Channel ID to send to
            template_name: Template name
            variables: Dict of variables to substitute (e.g., {"username": "John", "points": "100"})
            reply_to: Optional message ID to reply to

        Returns:
            Dict with sent message info
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return {"error": f"Channel {channel_id} not found"}

        if not hasattr(self, 'message_templates'):
            self.message_templates = {}

        if template_name not in self.message_templates:
            return {"error": f"Template '{template_name}' not found"}

        template = self.message_templates[template_name]

        try:
            # Substitute variables in content
            content = template.get("content")
            if content and variables:
                for key, value in variables.items():
                    content = content.replace(f"{{{key}}}", str(value))

            # Create embed with variable substitution
            discord_embed = None
            if template.get("embed"):
                embed_data = template["embed"].copy()

                # Substitute variables in embed fields
                if variables:
                    for key, value in variables.items():
                        if embed_data.get("title"):
                            embed_data["title"] = embed_data["title"].replace(f"{{{key}}}", str(value))
                        if embed_data.get("description"):
                            embed_data["description"] = embed_data["description"].replace(f"{{{key}}}", str(value))

                        # Substitute in fields
                        if embed_data.get("fields"):
                            for field in embed_data["fields"]:
                                if field.get("name"):
                                    field["name"] = field["name"].replace(f"{{{key}}}", str(value))
                                if field.get("value"):
                                    field["value"] = field["value"].replace(f"{{{key}}}", str(value))

                discord_embed = discord.Embed(
                    title=embed_data.get("title"),
                    description=embed_data.get("description"),
                    color=discord.Color(embed_data.get("color", 0x3498db))
                )

                # Add fields
                for field in embed_data.get("fields", []):
                    discord_embed.add_field(
                        name=field.get("name", "Field"),
                        value=field.get("value", ""),
                        inline=field.get("inline", False)
                    )

                # Add footer, author, thumbnail, image if present
                if embed_data.get("footer"):
                    discord_embed.set_footer(text=embed_data["footer"].get("text"))
                if embed_data.get("author"):
                    discord_embed.set_author(name=embed_data["author"].get("name"))
                if embed_data.get("thumbnail"):
                    discord_embed.set_thumbnail(url=embed_data["thumbnail"])
                if embed_data.get("image"):
                    discord_embed.set_image(url=embed_data["image"])

            # Create components (buttons, select menus)
            view = None
            if template.get("components"):
                view = discord.ui.View(timeout=None)

                for component in template["components"]:
                    comp_type = component.get("type")

                    if comp_type == "button":
                        button = discord.ui.Button(
                            label=component.get("label", "Button"),
                            style=discord.ButtonStyle[component.get("style", "primary")],
                            custom_id=component.get("custom_id"),
                            emoji=component.get("emoji"),
                            url=component.get("url"),
                            disabled=component.get("disabled", False)
                        )
                        view.add_item(button)

                    elif comp_type == "select":
                        options = [
                            discord.SelectOption(
                                label=opt.get("label"),
                                value=opt.get("value"),
                                description=opt.get("description"),
                                emoji=opt.get("emoji")
                            )
                            for opt in component.get("options", [])
                        ]

                        select = discord.ui.Select(
                            placeholder=component.get("placeholder", "Select an option"),
                            options=options,
                            custom_id=component.get("custom_id"),
                            min_values=component.get("min_values", 1),
                            max_values=component.get("max_values", 1)
                        )
                        view.add_item(select)

            # Get reference message if replying
            reference = None
            if reply_to:
                try:
                    ref_msg = await channel.fetch_message(reply_to)
                    reference = ref_msg
                except:
                    pass

            # Send message
            message = await channel.send(
                content=content,
                embed=discord_embed,
                view=view,
                reference=reference
            )

            return {
                "success": True,
                "message_id": message.id,
                "channel_id": channel_id,
                "template_name": template_name,
                "timestamp": message.created_at.isoformat()
            }
        except Exception as e:
            return {"error": str(e)}

    async def create_welcome_template(
        self,
        template_name: str = "welcome",
        title: str = "Welcome to {server_name}!",
        description: str = "Hey {username}, welcome to our server! We're glad to have you here.",
        color: int = 0x00ff00,
        thumbnail: Optional[str] = None,
        image: Optional[str] = None,
        fields: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Create a welcome message template with common variables.

        Args:
            template_name: Template name
            title: Title with variables like {username}, {server_name}, {member_count}
            description: Description text with variables
            color: Embed color (hex)
            thumbnail: Thumbnail URL
            image: Image URL
            fields: List of embed fields

        Returns:
            Dict with template info
        """
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "fields": fields or [],
            "thumbnail": thumbnail,
            "image": image,
            "footer": {"text": "Member #{member_count}"}
        }

        return await self.create_message_template(
            template_name=template_name,
            embed=embed
        )

    async def create_announcement_template(
        self,
        template_name: str = "announcement",
        title: str = "📢 Announcement",
        description: str = "{message}",
        color: int = 0xff9900,
        mention_role: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an announcement message template.

        Args:
            template_name: Template name
            title: Announcement title
            description: Description with {message} variable
            color: Embed color
            mention_role: Role mention (e.g., "@everyone", "@here")

        Returns:
            Dict with template info
        """
        content = mention_role if mention_role else None

        embed = {
            "title": title,
            "description": description,
            "color": color,
            "footer": {"text": "Posted on {date}"}
        }

        return await self.create_message_template(
            template_name=template_name,
            content=content,
            embed=embed
        )

    async def create_poll_template(
        self,
        template_name: str = "poll",
        question: str = "{question}",
        options: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a poll template with reaction options.

        Args:
            template_name: Template name
            question: Poll question with variables
            options: List of poll options (max 10)

        Returns:
            Dict with template info
        """
        if not options:
            options = ["{option1}", "{option2}", "{option3}"]

        # Emoji numbers for reactions
        emoji_numbers = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]

        description = question + "\n\n"
        for i, option in enumerate(options[:10]):
            description += f"{emoji_numbers[i]} {option}\n"

        embed = {
            "title": "📊 Poll",
            "description": description,
            "color": 0x3498db,
            "footer": {"text": "React to vote!"}
        }

        return await self.create_message_template(
            template_name=template_name,
            embed=embed
        )

    async def create_embed_template(
        self,
        template_name: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        color: int = 0x3498db,
        fields: Optional[List[Dict[str, Any]]] = None,
        footer: Optional[str] = None,
        author: Optional[str] = None,
        thumbnail: Optional[str] = None,
        image: Optional[str] = None,
        url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a custom embed template with all options.

        Args:
            template_name: Template name
            title: Embed title (supports variables)
            description: Embed description (supports variables)
            color: Color as hex integer
            fields: List of {"name": str, "value": str, "inline": bool}
            footer: Footer text
            author: Author name
            thumbnail: Thumbnail URL
            image: Image URL
            url: Title URL

        Returns:
            Dict with template info
        """
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "fields": fields or [],
            "url": url
        }

        if footer:
            embed["footer"] = [{"text": footer}]
        if author:
            embed["author"] = [{"name": author}]
        if thumbnail:
            embed["thumbnail"] = thumbnail
        if image:
            embed["image"] = image

        return await self.create_message_template(
            template_name=template_name,
            embed=embed
        )

    async def create_button_template(
        self,
        template_name: str,
        content: Optional[str] = None,
        buttons: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Create a message template with buttons.

        Args:
            template_name: Template name
            content: Message content
            buttons: List of button configs with keys:
                     - label: Button text
                     - style: "primary"/"secondary"/"success"/"danger"/"link"
                     - custom_id: Unique ID for the button
                     - emoji: Optional emoji
                     - url: URL for link buttons
                     - disabled: Boolean

        Returns:
            Dict with template info
        """
        components = []

        if buttons:
            for button in buttons:
                components.append({
                    "type": "button",
                    "label": button.get("label", "Button"),
                    "style": button.get("style", "primary"),
                    "custom_id": button.get("custom_id"),
                    "emoji": button.get("emoji"),
                    "url": button.get("url"),
                    "disabled": button.get("disabled", False)
                })

        return await self.create_message_template(
            template_name=template_name,
            content=content,
            components=components
        )

    async def create_select_menu_template(
        self,
        template_name: str,
        content: Optional[str] = None,
        placeholder: str = "Select an option",
        options: Optional[List[Dict[str, Any]]] = None,
        min_values: int = 1,
        max_values: int = 1
    ) -> Dict[str, Any]:
        """
        Create a message template with a select menu.

        Args:
            template_name: Template name
            content: Message content
            placeholder: Placeholder text
            options: List of option configs with keys:
                     - label: Option label
                     - value: Option value
                     - description: Optional description
                     - emoji: Optional emoji
            min_values: Minimum selections
            max_values: Maximum selections

        Returns:
            Dict with template info
        """
        if not options:
            options = []

        components = [{
            "type": "select",
            "placeholder": placeholder,
            "options": options,
            "custom_id": f"select_{template_name}",
            "min_values": min_values,
            "max_values": max_values
        }]

        return await self.create_message_template(
            template_name=template_name,
            content=content,
            components=components
        )

    # ===== INFORMATION & HELP TOOLS =====

    async def get_template_help(self) -> Dict[str, Any]:
        """
        Get comprehensive help on creating and using message templates.

        Returns:
            Dict with detailed template documentation and examples
        """
        help_text = {
            "overview": "Message templates allow you to create reusable messages with variable substitution, embeds, buttons, and select menus.",

            "variable_substitution": {
                "description": "Use {variable_name} syntax in templates. Variables are replaced when sending.",
                "common_variables": {
                    "username": "User's display name",
                    "user_id": "User's ID",
                    "server_name": "Server/guild name",
                    "member_count": "Total member count",
                    "channel_name": "Channel name",
                    "date": "Current date",
                    "time": "Current time",
                    "message": "Custom message content"
                },
                "example": "Title: 'Welcome {username}!' → Becomes: 'Welcome John!'"
            },

            "template_types": {
                "basic_text": {
                    "description": "Simple text message with variables",
                    "example": {
                        "function": "discord_create_message_template",
                        "args": {
                            "template_name": "greeting",
                            "content": "Hello {username}, welcome to {server_name}!"
                        }
                    }
                },

                "embed": {
                    "description": "Rich embed messages with title, description, fields, colors, images",
                    "structure": {
                        "title": "Embed title (supports variables)",
                        "description": "Main content (supports variables)",
                        "color": "Hex color code (e.g., 0xff0000 for red)",
                        "fields": "List of {name, value, inline} dicts",
                        "footer": "Footer text",
                        "thumbnail": "Small image URL (top right)",
                        "image": "Large image URL (bottom)",
                        "author": "Author name (top)"
                    },
                    "example": {
                        "function": "discord_create_embed_template",
                        "args": {
                            "template_name": "user_info",
                            "title": "User: {username}",
                            "description": "Member since {join_date}",
                            "color": 0x00ff00,
                            "fields": [
                                {"name": "User ID", "value": "{user_id}", "inline": True},
                                {"name": "Roles", "value": "{roles}", "inline": True}
                            ],
                            "footer": "Server: {server_name}"
                        }
                    }
                },

                "welcome": {
                    "description": "Pre-configured welcome message template",
                    "variables": ["username", "server_name", "member_count"],
                    "example": {
                        "function": "discord_create_welcome_template",
                        "args": {
                            "template_name": "new_member",
                            "title": "Welcome {username}!",
                            "description": "Welcome to {server_name}! You are member #{member_count}",
                            "color": 0x00ff00,
                            "thumbnail": "https://example.com/welcome.png"
                        }
                    }
                },

                "announcement": {
                    "description": "Announcement message with optional role mentions",
                    "variables": ["message", "date"],
                    "example": {
                        "function": "discord_create_announcement_template",
                        "args": {
                            "template_name": "server_update",
                            "title": "📢 Server Update",
                            "description": "{message}",
                            "color": 0xff9900,
                            "mention_role": "@everyone"
                        }
                    }
                },

                "poll": {
                    "description": "Poll with numbered reaction options",
                    "variables": ["question", "option1", "option2", "option3", "..."],
                    "example": {
                        "function": "discord_create_poll_template",
                        "args": {
                            "template_name": "vote",
                            "question": "What should we do next?",
                            "options": ["Add new features", "Fix bugs", "Improve performance"]
                        }
                    }
                },

                "buttons": {
                    "description": "Interactive buttons for user actions",
                    "button_styles": {
                        "primary": "Blurple/blue button",
                        "secondary": "Gray button",
                        "success": "Green button",
                        "danger": "Red button",
                        "link": "Link button (requires url)"
                    },
                    "example": {
                        "function": "discord_create_button_template",
                        "args": {
                            "template_name": "verify",
                            "content": "Click to verify your account",
                            "buttons": [
                                {
                                    "label": "✅ Verify",
                                    "style": "success",
                                    "custom_id": "verify_button"
                                },
                                {
                                    "label": "Help",
                                    "style": "link",
                                    "url": "https://example.com/help"
                                }
                            ]
                        }
                    }
                },

                "select_menu": {
                    "description": "Dropdown menu for multiple choice selection",
                    "example": {
                        "function": "discord_create_select_menu_template",
                        "args": {
                            "template_name": "role_select",
                            "content": "Choose your roles:",
                            "placeholder": "Select roles...",
                            "options": [
                                {
                                    "label": "Developer",
                                    "value": "dev",
                                    "description": "Programming role",
                                    "emoji": "💻"
                                },
                                {
                                    "label": "Designer",
                                    "value": "design",
                                    "description": "Design role",
                                    "emoji": "🎨"
                                }
                            ],
                            "min_values": 1,
                            "max_values": 2
                        }
                    }
                }
            },

            "workflow": {
                "step_1": {
                    "action": "Create template",
                    "description": "Use one of the create_*_template functions",
                    "example": "discord_create_welcome_template('welcome', title='Hi {username}!')"
                },
                "step_2": {
                    "action": "List templates",
                    "description": "View all available templates",
                    "example": "discord_list_message_templates()"
                },
                "step_3": {
                    "action": "Send template",
                    "description": "Send template with variable values",
                    "example": "discord_send_template_message(channel_id=123, template_name='welcome', variables={'username': 'John', 'member_count': '500'})"
                },
                "step_4": {
                    "action": "Manage templates",
                    "description": "Get, update, or delete templates as needed",
                    "example": "discord_delete_message_template('old_template')"
                }
            },

            "color_codes": {
                "description": "Common color hex codes for embeds",
                "colors": {
                    "blue": 0x3498db,
                    "green": 0x00ff00,
                    "red": 0xff0000,
                    "yellow": 0xffff00,
                    "purple": 0x9b59b6,
                    "orange": 0xff9900,
                    "pink": 0xff69b4,
                    "black": 0x000000,
                    "white": 0xffffff,
                    "discord_blurple": 0x5865F2,
                    "discord_green": 0x57F287,
                    "discord_yellow": 0xFEE75C,
                    "discord_fuchsia": 0xEB459E,
                    "discord_red": 0xED4245
                }
            },

            "best_practices": [
                "Use clear, descriptive template names",
                "Include all necessary variables in template documentation",
                "Test templates before using in production",
                "Use appropriate colors for message type (green=success, red=error, blue=info)",
                "Keep embed descriptions concise (max 4096 characters)",
                "Limit fields to 25 per embed",
                "Use inline fields for compact layouts",
                "Add emojis for visual appeal",
                "Include footers for timestamps or additional context",
                "Use buttons/selects for interactive experiences"
            ],

            "common_use_cases": {
                "welcome_messages": "Greet new members with server info",
                "announcements": "Notify members of updates or events",
                "polls": "Gather community feedback",
                "role_selection": "Let users choose their roles",
                "verification": "Button-based verification system",
                "help_menus": "Interactive help with buttons/selects",
                "moderation_logs": "Formatted mod action logs",
                "status_updates": "Bot or server status messages",
                "leaderboards": "Display rankings and scores",
                "ticket_systems": "User support ticket creation"
            },

            "tips": [
                "Variables are case-sensitive: {username} ≠ {Username}",
                "Use preview mode: Get template first, check structure",
                "Combine content + embed for rich messages",
                "Custom IDs for buttons/selects must be unique",
                "Link buttons don't need custom_id",
                "Select menus can have 1-25 options",
                "Button rows have max 5 buttons each",
                "Embeds support markdown formatting",
                "Use \\n for line breaks in descriptions",
                "Thumbnails show small (top-right), images show large (bottom)"
            ]
        }

        return {
            "success": True,
            "help": help_text
        }

    async def get_tools_overview(self) -> Dict[str, Any]:
        """
        Get overview of all available Discord tools organized by category.

        Returns:
            Dict with categorized tool information
        """
        tools_overview = {
            "total_tools": 56,

            "categories": {
                "server_management": {
                    "description": "Tools for creating and managing Discord servers",
                    "tools": [
                        {
                            "name": "discord_create_server",
                            "description": "Create a new Discord server",
                            "usage": "discord_create_server(name='My Server')"
                        },
                        {
                            "name": "discord_delete_server",
                            "description": "Delete a server (bot must be owner)",
                            "usage": "discord_delete_server(guild_id=123)"
                        },
                        {
                            "name": "discord_edit_server",
                            "description": "Edit server settings",
                            "usage": "discord_edit_server(guild_id=123, name='New Name')"
                        },
                        {
                            "name": "discord_get_server_info",
                            "description": "Get server information",
                            "usage": "discord_get_server_info(guild_id=123)"
                        }
                    ]
                },

                "channel_management": {
                    "description": "Tools for creating and managing channels",
                    "tools": [
                        {
                            "name": "discord_create_channel",
                            "description": "Create a new channel",
                            "usage": "discord_create_channel(guild_id=123, name='general', channel_type='text')"
                        },
                        {
                            "name": "discord_delete_channel",
                            "description": "Delete a channel",
                            "usage": "discord_delete_channel(channel_id=456)"
                        },
                        {
                            "name": "discord_edit_channel",
                            "description": "Edit channel settings",
                            "usage": "discord_edit_channel(channel_id=456, name='new-name', topic='New topic')"
                        },
                        {
                            "name": "discord_list_channels",
                            "description": "List all channels in a server",
                            "usage": "discord_list_channels(guild_id=123, channel_type='text')"
                        },
                        {
                            "name": "discord_get_channel_info",
                            "description": "Get channel information",
                            "usage": "discord_get_channel_info(channel_id=456)"
                        }
                    ]
                },

                "message_management": {
                    "description": "Tools for sending and managing messages",
                    "tools": [
                        {
                            "name": "discord_send_message",
                            "description": "Send a message",
                            "usage": "discord_send_message(channel_id=456, content='Hello!')"
                        },
                        {
                            "name": "discord_edit_message",
                            "description": "Edit a message",
                            "usage": "discord_edit_message(channel_id=456, message_id=789, new_content='Updated')"
                        },
                        {
                            "name": "discord_delete_message",
                            "description": "Delete a message",
                            "usage": "discord_delete_message(channel_id=456, message_id=789)"
                        },
                        {
                            "name": "discord_get_message",
                            "description": "Get message information",
                            "usage": "discord_get_message(channel_id=456, message_id=789)"
                        },
                        {
                            "name": "discord_get_recent_messages",
                            "description": "Get recent messages from channel",
                            "usage": "discord_get_recent_messages(channel_id=456, limit=10)"
                        },
                        {
                            "name": "discord_send_file",
                            "description": "Send a file",
                            "usage": "discord_send_file(channel_id=456, file_path='/path/to/file.png')"
                        }
                    ]
                },

                "template_management": {
                    "description": "Tools for creating and using message templates",
                    "tools": [
                        {
                            "name": "discord_create_message_template",
                            "description": "Create a custom template",
                            "usage": "discord_create_message_template('greeting', content='Hello {username}!')"
                        },
                        {
                            "name": "discord_create_welcome_template",
                            "description": "Create a welcome template",
                            "usage": "discord_create_welcome_template(title='Welcome {username}!')"
                        },
                        {
                            "name": "discord_create_announcement_template",
                            "description": "Create an announcement template",
                            "usage": "discord_create_announcement_template(description='{message}')"
                        },
                        {
                            "name": "discord_create_poll_template",
                            "description": "Create a poll template",
                            "usage": "discord_create_poll_template(question='Favorite?', options=['A', 'B'])"
                        },
                        {
                            "name": "discord_create_embed_template",
                            "description": "Create a custom embed template",
                            "usage": "discord_create_embed_template('info', title='{title}', color=0xff0000)"
                        },
                        {
                            "name": "discord_create_button_template",
                            "description": "Create a template with buttons",
                            "usage": "discord_create_button_template('menu', buttons=[{'label': 'Click', 'style': 'primary'}])"
                        },
                        {
                            "name": "discord_create_select_menu_template",
                            "description": "Create a template with dropdown",
                            "usage": "discord_create_select_menu_template('roles', options=[{'label': 'Role', 'value': 'role1'}])"
                        },
                        {
                            "name": "discord_send_template_message",
                            "description": "Send a template with variables",
                            "usage": "discord_send_template_message(channel_id=456, template_name='welcome', variables={'username': 'John'})"
                        },
                        {
                            "name": "discord_list_message_templates",
                            "description": "List all templates",
                            "usage": "discord_list_message_templates()"
                        },
                        {
                            "name": "discord_get_message_template",
                            "description": "Get a specific template",
                            "usage": "discord_get_message_template('welcome')"
                        },
                        {
                            "name": "discord_delete_message_template",
                            "description": "Delete a template",
                            "usage": "discord_delete_message_template('old_template')"
                        }
                    ]
                },

                "moderation": {
                    "description": "Tools for moderating users and content",
                    "tools": [
                        {
                            "name": "discord_kick_member",
                            "description": "Kick a member",
                            "usage": "discord_kick_member(guild_id=123, user_id=789, reason='Spam')"
                        },
                        {
                            "name": "discord_ban_member",
                            "description": "Ban a member",
                            "usage": "discord_ban_member(guild_id=123, user_id=789, reason='Rule violation')"
                        },
                        {
                            "name": "discord_unban_member",
                            "description": "Unban a member",
                            "usage": "discord_unban_member(guild_id=123, user_id=789)"
                        },
                        {
                            "name": "discord_timeout_member",
                            "description": "Timeout a member",
                            "usage": "discord_timeout_member(guild_id=123, user_id=789, duration_minutes=60)"
                        },
                        {
                            "name": "discord_remove_timeout",
                            "description": "Remove timeout",
                            "usage": "discord_remove_timeout(guild_id=123, user_id=789)"
                        },
                        {
                            "name": "discord_change_nickname",
                            "description": "Change member nickname",
                            "usage": "discord_change_nickname(guild_id=123, user_id=789, nickname='NewName')"
                        }
                    ]
                },

                "role_management": {
                    "description": "Tools for managing roles",
                    "tools": [
                        {
                            "name": "discord_add_role",
                            "description": "Add role to member",
                            "usage": "discord_add_role(guild_id=123, user_id=789, role_id=456)"
                        },
                        {
                            "name": "discord_remove_role",
                            "description": "Remove role from member",
                            "usage": "discord_remove_role(guild_id=123, user_id=789, role_id=456)"
                        },
                        {
                            "name": "discord_get_member_roles",
                            "description": "Get member's roles",
                            "usage": "discord_get_member_roles(guild_id=123, user_id=789)"
                        }
                    ]
                },

                "voice_management": {
                    "description": "Tools for voice channels and audio",
                    "tools": [
                        {
                            "name": "discord_join_voice",
                            "description": "Join a voice channel",
                            "usage": "discord_join_voice(channel_id=456)"
                        },
                        {
                            "name": "discord_leave_voice",
                            "description": "Leave voice channel",
                            "usage": "discord_leave_voice(guild_id=123)"
                        },
                        {
                            "name": "discord_get_voice_status",
                            "description": "Get voice status",
                            "usage": "discord_get_voice_status(guild_id=123)"
                        },
                        {
                            "name": "discord_toggle_tts",
                            "description": "Toggle text-to-speech",
                            "usage": "discord_toggle_tts(guild_id=123, mode='piper')"
                        },
                        {
                            "name": "discord_move_member",
                            "description": "Move member to voice channel",
                            "usage": "discord_move_member(guild_id=123, user_id=789, channel_id=456)"
                        },
                        {
                            "name": "discord_disconnect_member",
                            "description": "Disconnect member from voice",
                            "usage": "discord_disconnect_member(guild_id=123, user_id=789)"
                        }
                    ]
                },

                "threads": {
                    "description": "Tools for managing threads",
                    "tools": [
                        {
                            "name": "discord_create_thread",
                            "description": "Create a thread",
                            "usage": "discord_create_thread(channel_id=456, name='Discussion')"
                        },
                        {
                            "name": "discord_join_thread",
                            "description": "Join a thread",
                            "usage": "discord_join_thread(thread_id=789)"
                        },
                        {
                            "name": "discord_leave_thread",
                            "description": "Leave a thread",
                            "usage": "discord_leave_thread(thread_id=789)"
                        }
                    ]
                },

                "invitations": {
                    "description": "Tools for managing server invites",
                    "tools": [
                        {
                            "name": "discord_create_invite",
                            "description": "Create an invite link",
                            "usage": "discord_create_invite(channel_id=456, max_age=3600, max_uses=10)"
                        },
                        {
                            "name": "discord_get_invites",
                            "description": "Get all server invites",
                            "usage": "discord_get_invites(guild_id=123)"
                        },
                        {
                            "name": "discord_delete_invite",
                            "description": "Delete an invite",
                            "usage": "discord_delete_invite(invite_code='abc123')"
                        },
                        {
                            "name": "discord_get_invite_info",
                            "description": "Get invite information",
                            "usage": "discord_get_invite_info(invite_code='abc123')"
                        }
                    ]
                },

                "reactions": {
                    "description": "Tools for managing reactions",
                    "tools": [
                        {
                            "name": "discord_add_reaction",
                            "description": "Add reaction to message",
                            "usage": "discord_add_reaction(channel_id=456, message_id=789, emoji='👍')"
                        },
                        {
                            "name": "discord_remove_reaction",
                            "description": "Remove reaction",
                            "usage": "discord_remove_reaction(channel_id=456, message_id=789, emoji='👍')"
                        }
                    ]
                },

                "permissions": {
                    "description": "Tools for managing permissions",
                    "tools": [
                        {
                            "name": "discord_set_channel_permissions",
                            "description": "Set channel permissions",
                            "usage": "discord_set_channel_permissions(channel_id=456, target_id=789, target_type='role')"
                        }
                    ]
                },

                "direct_messages": {
                    "description": "Tools for DMs",
                    "tools": [
                        {
                            "name": "discord_send_dm",
                            "description": "Send a DM to user",
                            "usage": "discord_send_dm(user_id=789, content='Hello!')"
                        }
                    ]
                },

                "webhooks": {
                    "description": "Tools for webhook management",
                    "tools": [
                        {
                            "name": "discord_create_webhook",
                            "description": "Create a webhook",
                            "usage": "discord_create_webhook(channel_id=456, name='My Webhook')"
                        }
                    ]
                },

                "bot_status": {
                    "description": "Tools for bot management",
                    "tools": [
                        {
                            "name": "discord_get_bot_status",
                            "description": "Get bot status",
                            "usage": "discord_get_bot_status()"
                        },
                        {
                            "name": "discord_set_bot_status",
                            "description": "Set bot status",
                            "usage": "discord_set_bot_status(status='online', activity_type='playing', activity_name='with AI')"
                        },
                        {
                            "name": "discord_get_kernel_metrics",
                            "description": "Get kernel metrics",
                            "usage": "discord_get_kernel_metrics()"
                        }
                    ]
                },

                "user_info": {
                    "description": "Tools for getting user information",
                    "tools": [
                        {
                            "name": "discord_get_user_info",
                            "description": "Get user information",
                            "usage": "discord_get_user_info(user_id=789, guild_id=123)"
                        }
                    ]
                }
            },

            "quick_start_examples": {
                "setup_new_server": [
                    "1. Create server: discord_create_server(name='My Server')",
                    "2. Create channels: discord_create_channel(guild_id=X, name='general', channel_type='text')",
                    "3. Create invite: discord_create_invite(channel_id=Y, max_age=0)",
                    "4. Create welcome template: discord_create_welcome_template()",
                    "5. Send welcome: discord_send_template_message(channel_id=Y, template_name='welcome', variables={'username': 'User'})"
                ],

                "moderation_workflow": [
                    "1. Get user info: discord_get_user_info(user_id=X, guild_id=Y)",
                    "2. Timeout user: discord_timeout_member(guild_id=Y, user_id=X, duration_minutes=60)",
                    "3. Or kick: discord_kick_member(guild_id=Y, user_id=X, reason='Spam')",
                    "4. Or ban: discord_ban_member(guild_id=Y, user_id=X, reason='Violation')"
                ],

                "announcement_workflow": [
                    "1. Create template: discord_create_announcement_template()",
                    "2. Send announcement: discord_send_template_message(channel_id=X, template_name='announcement', variables={'message': 'Server update!', 'date': '2024-01-01'})"
                ]
            }
        }

        return {
            "success": True,
            "overview": tools_overview
        }

    async def get_template_examples(self) -> Dict[str, Any]:
        """
        Get practical template examples for common scenarios.

        Returns:
            Dict with ready-to-use template examples showing tool usage
        """
        examples = {
            "welcome_member": {
                "description": "Welcome new members with server info",
                "workflow": [
                    {
                        "step": 1,
                        "action": "Get server info",
                        "tool": "discord_get_server_info",
                        "args": {"guild_id": 123456789}
                    },
                    {
                        "step": 2,
                        "action": "Send welcome message with embed",
                        "tool": "discord_send_message",
                        "args": {
                            "channel_id": 987654321,
                            "content": "Welcome to the server!",
                            "embed": {
                                "title": "Welcome {username}! 🎉",
                                "description": "We're excited to have you here! You are member #{member_count}",
                                "color": 65280,
                                "fields": [
                                    {"name": "📜 Read the Rules", "value": "Check out <#rules_channel_id>", "inline": False},
                                    {"name": "👋 Say Hi", "value": "Introduce yourself in <#intro_channel_id>", "inline": False}
                                ]
                            }
                        }
                    }
                ],
                "result": "Rich welcome message with server info and helpful links"
            },

            "moderation_log": {
                "description": "Log moderation actions",
                "workflow": [
                    {
                        "step": 1,
                        "action": "Get user info",
                        "tool": "discord_get_user_info",
                        "args": {"user_id": 111111, "guild_id": 123456789}
                    },
                    {
                        "step": 2,
                        "action": "Send moderation log",
                        "tool": "discord_send_message",
                        "args": {
                            "channel_id": 555555,
                            "embed": {
                                "title": "🔨 Moderation Action",
                                "description": "**Action:** Ban\n**User:** Username (111111)\n**Moderator:** ModName\n**Reason:** Repeated rule violations",
                                "color": 16711680
                            }
                        }
                    }
                ],
                "result": "Formatted moderation log entry"
            },

            "verification_system": {
                "description": "Button-based verification (requires interaction handling)",
                "workflow": [
                    {
                        "step": 1,
                        "action": "Send verification message",
                        "tool": "discord_send_message",
                        "args": {
                            "channel_id": 999999,
                            "content": "Welcome! Please verify to access the server.",
                            "embed": {
                                "title": "✅ Verification Required",
                                "description": "Click the button below to verify and gain access to all channels.",
                                "color": 3066993
                            }
                        }
                    },
                    {
                        "step": 2,
                        "action": "Add reaction for manual verification",
                        "tool": "discord_add_reaction",
                        "args": {
                            "channel_id": 999999,
                            "message_id": 777777,
                            "emoji": "✅"
                        }
                    }
                ],
                "result": "Verification message (button interactions require bot event handlers)"
            },

            "role_assignment": {
                "description": "Assign role to user",
                "workflow": [
                    {
                        "step": 1,
                        "action": "Get member's current roles",
                        "tool": "discord_get_member_roles",
                        "args": {"guild_id": 123456789, "user_id": 111111}
                    },
                    {
                        "step": 2,
                        "action": "Add new role",
                        "tool": "discord_add_role",
                        "args": {
                            "guild_id": 123456789,
                            "user_id": 111111,
                            "role_id": 888888,
                            "reason": "Verified member"
                        }
                    },
                    {
                        "step": 3,
                        "action": "Notify user via DM",
                        "tool": "discord_send_message",
                        "args": {
                            "channel_id": 111111,
                            "content": "You've been assigned the Verified role! 🎉"
                        }
                    }
                ],
                "result": "Role assigned and user notified"
            },

            "server_announcement": {
                "description": "Create and send server announcement",
                "workflow": [
                    {
                        "step": 1,
                        "action": "Send announcement with embed",
                        "tool": "discord_send_message",
                        "args": {
                            "channel_id": 123456,
                            "content": "@everyone",
                            "embed": {
                                "title": "📢 Server Announcement",
                                "description": "Important update for all members!",
                                "color": 15844367,
                                "fields": [
                                    {"name": "What's New", "value": "New features added", "inline": False},
                                    {"name": "When", "value": "Effective immediately", "inline": False}
                                ]
                            }
                        }
                    },
                    {
                        "step": 2,
                        "action": "Pin the announcement",
                        "tool": "discord_pin_message",
                        "args": {"channel_id": 123456, "message_id": 999999}
                    }
                ],
                "result": "Pinned announcement visible to all members"
            },

            "poll_with_reactions": {
                "description": "Create a poll using reactions",
                "workflow": [
                    {
                        "step": 1,
                        "action": "Send poll message",
                        "tool": "discord_send_message",
                        "args": {
                            "channel_id": 123456,
                            "embed": {
                                "title": "📊 Poll: What feature should we add next?",
                                "description": "1️⃣ New game modes\n2️⃣ More channels\n3️⃣ Bot improvements\n4️⃣ Events and contests",
                                "color": 3447003
                            }
                        }
                    },
                    {
                        "step": 2,
                        "action": "Add reaction options",
                        "tool": "discord_add_reaction",
                        "args": {"channel_id": 123456, "message_id": 999999, "emoji": "1️⃣"}
                    },
                    {
                        "step": 3,
                        "action": "Add more reactions",
                        "tool": "discord_add_reaction",
                        "args": {"channel_id": 123456, "message_id": 999999, "emoji": "2️⃣"}
                    }
                ],
                "result": "Poll with numbered reactions for voting",
                "note": "Repeat step 3 for each option (3️⃣, 4️⃣, etc.)"
            },

            "event_announcement": {
                "description": "Announce server events",
                "workflow": [
                    {
                        "step": 1,
                        "action": "Send event announcement",
                        "tool": "discord_send_message",
                        "args": {
                            "channel_id": 789012,
                            "embed": {
                                "title": "🎉 Movie Night",
                                "description": "Join us for a community movie night!",
                                "color": 16738740,
                                "fields": [
                                    {"name": "📅 Date", "value": "Saturday, Jan 15", "inline": True},
                                    {"name": "🕐 Time", "value": "8:00 PM EST", "inline": True},
                                    {"name": "📍 Location", "value": "Voice Channel #1", "inline": True},
                                    {"name": "ℹ️ Details", "value": "We'll be watching a community-voted movie. Bring snacks!", "inline": False}
                                ]
                            }
                        }
                    },
                    {
                        "step": 2,
                        "action": "Add RSVP reaction",
                        "tool": "discord_add_reaction",
                        "args": {"channel_id": 789012, "message_id": 888888, "emoji": "✅"}
                    }
                ],
                "result": "Rich event announcement with all details and RSVP option"
            },

            "leaderboard_display": {
                "description": "Display rankings and scores",
                "workflow": [
                    {
                        "step": 1,
                        "action": "Send leaderboard",
                        "tool": "discord_send_message",
                        "args": {
                            "channel_id": 345678,
                            "embed": {
                                "title": "🏆 Weekly Top Contributors",
                                "description": "Top members this week",
                                "color": 16766720,
                                "fields": [
                                    {"name": "🥇 1st Place", "value": "**@User1** - 1,250 points", "inline": False},
                                    {"name": "🥈 2nd Place", "value": "**@User2** - 980 points", "inline": False},
                                    {"name": "🥉 3rd Place", "value": "**@User3** - 875 points", "inline": False},
                                    {"name": "Others", "value": "4. @User4 - 720\n5. @User5 - 650", "inline": False}
                                ]
                            }
                        }
                    }
                ],
                "result": "Formatted leaderboard with rankings"
            },

            "voice_session_management": {
                "description": "Manage voice channel sessions",
                "workflow": [
                    {
                        "step": 1,
                        "action": "Join voice channel",
                        "tool": "discord_join_voice",
                        "args": {"channel_id": 555555}
                    },
                    {
                        "step": 2,
                        "action": "Enable TTS",
                        "tool": "discord_toggle_tts",
                        "args": {"guild_id": 123456789, "mode": "piper"}
                    },
                    {
                        "step": 3,
                        "action": "Check voice status",
                        "tool": "discord_get_voice_status",
                        "args": {"guild_id": 123456789}
                    },
                    {
                        "step": 4,
                        "action": "Leave when done",
                        "tool": "discord_leave_voice",
                        "args": {"guild_id": 123456789}
                    }
                ],
                "result": "Complete voice session with TTS enabled"
            },

            "member_info_check": {
                "description": "Get comprehensive member information",
                "workflow": [
                    {
                        "step": 1,
                        "action": "Get user info",
                        "tool": "discord_get_user_info",
                        "args": {"user_id": 111111, "guild_id": 123456789}
                    },
                    {
                        "step": 2,
                        "action": "Get member roles",
                        "tool": "discord_get_member_roles",
                        "args": {"guild_id": 123456789, "user_id": 111111}
                    },
                    {
                        "step": 3,
                        "action": "Get recent messages",
                        "tool": "discord_get_recent_messages",
                        "args": {"channel_id": 987654, "limit": 10}
                    }
                ],
                "result": "Complete member profile with roles and activity"
            },

            "bot_status_update": {
                "description": "Display bot status and metrics",
                "workflow": [
                    {
                        "step": 1,
                        "action": "Get bot status",
                        "tool": "discord_get_bot_status",
                        "args": {}
                    },
                    {
                        "step": 2,
                        "action": "Get kernel metrics",
                        "tool": "discord_get_kernel_metrics",
                        "args": {}
                    },
                    {
                        "step": 3,
                        "action": "Send status message",
                        "tool": "discord_send_message",
                        "args": {
                            "channel_id": 123456,
                            "embed": {
                                "title": "📊 Bot Status",
                                "description": "All systems operational",
                                "color": 3447003,
                                "fields": [
                                    {"name": "Status", "value": "🟢 Online", "inline": True},
                                    {"name": "Latency", "value": "45ms", "inline": True},
                                    {"name": "Guilds", "value": "10", "inline": True},
                                    {"name": "Users", "value": "1,234", "inline": True}
                                ]
                            }
                        }
                    }
                ],
                "result": "Comprehensive status dashboard with live metrics"
            },

            "message_cleanup": {
                "description": "Clean up old messages",
                "workflow": [
                    {
                        "step": 1,
                        "action": "Get recent messages",
                        "tool": "discord_get_recent_messages",
                        "args": {"channel_id": 123456, "limit": 50}
                    },
                    {
                        "step": 2,
                        "action": "Delete specific message",
                        "tool": "discord_delete_message",
                        "args": {"channel_id": 123456, "message_id": 999999, "delay": 0}
                    }
                ],
                "result": "Messages cleaned up",
                "note": "Repeat step 2 for each message to delete"
            }
        }

        return {
            "success": True,
            "examples": examples,
            "total_examples": len(examples),
            "usage_note": "Each example shows a workflow with specific tool calls and arguments. Use these as templates for common Discord tasks."
        }

    # ===== EXPORT TO AGENT =====

    async def export_to_agent(self):
        """Export all Discord tools to the agent"""
        agent = self.kernel.agent

        # Server Management Tools
        await agent.add_tool(
            self.get_server_info,
            "discord_get_server_info",
            description="Get information about Discord server(s). "
                       "Args: guild_id (int, optional). If None, returns all servers. "
                       "Returns: Dict with server info (name, member_count, channels, roles, etc.). "
                       "Example: discord_get_server_info(guild_id=123456789)"
        )

        await agent.add_tool(
            self.get_channel_info,
            "discord_get_channel_info",
            description="Get information about a Discord channel. "
                       "Args: channel_id (int). "
                       "Returns: Dict with channel info (name, type, topic, members, etc.). "
                       "Example: discord_get_channel_info(channel_id=987654321)"
        )

        await agent.add_tool(
            self.list_channels,
            "discord_list_channels",
            description="List all channels in a guild. "
                       "Args: guild_id (int), channel_type (str, optional: 'text'/'voice'/'category'/'stage'). "
                       "Returns: List of channel dicts. "
                       "Example: discord_list_channels(guild_id=123, channel_type='text')"
        )

        await agent.add_tool(
            self.get_user_info,
            "discord_get_user_info",
            description="Get information about a Discord user. "
                       "Args: user_id (int), guild_id (int, optional for member-specific info). "
                       "Returns: Dict with user info (name, roles, voice_channel, etc.). "
                       "Example: discord_get_user_info(user_id=111, guild_id=222)"
        )

        # Message Management Tools
        await agent.add_tool(
            self.send_message,
            "discord_send_message",
            description="Send a message to a Discord channel. "
                       "Args: channel_id (int), content (str), embed (dict, optional), reply_to (int, optional). "
                       "Embed format: {'title': str, 'description': str, 'color': int, 'fields': [{'name': str, 'value': str, 'inline': bool}]}. "
                       "Returns: Dict with message_id and timestamp. "
                       "Example: discord_send_message(channel_id=123, content='Hello!', reply_to=456)"
        )

        await agent.add_tool(
            self.output_router.send_media,
            "discord_send_media",
            description="Send media (images, files) to a Discord user. "
                       "Args: user_id (str), file_path (str, optional), url (str, optional), caption (str, optional). "
                       "Either file_path or url must be provided. "
                       "Returns: Dict with success status. "
                       "Example: discord_send_media(user_id='123456789', url='https://example.com/image.png', caption='Check this out!')"
        )

        await agent.add_tool(
            self.edit_message,
            "discord_edit_message",
            description="Edit an existing message. "
                       "Args: channel_id (int), message_id (int), new_content (str, optional), new_embed (dict, optional). "
                       "Returns: Dict with success status. "
                       "Example: discord_edit_message(channel_id=123, message_id=456, new_content='Updated!')"
        )

        await agent.add_tool(
            self.delete_message,
            "discord_delete_message",
            description="Delete a message. "
                       "Args: channel_id (int), message_id (int), delay (float, optional seconds). "
                       "Returns: Dict with success status. "
                       "Example: discord_delete_message(channel_id=123, message_id=456, delay=5.0)"
        )

        await agent.add_tool(
            self.get_message,
            "discord_get_message",
            description="Get information about a specific message. "
                       "Args: channel_id (int), message_id (int). "
                       "Returns: Dict with message info (content, author, embeds, reactions, etc.). "
                       "Example: discord_get_message(channel_id=123, message_id=456)"
        )

        await agent.add_tool(
            self.get_recent_messages,
            "discord_get_recent_messages",
            description="Get recent messages from a channel. "
                       "Args: channel_id (int), limit (int, default 10, max 100), before (int, optional), after (int, optional). "
                       "Returns: List of message dicts. "
                       "Example: discord_get_recent_messages(channel_id=123, limit=20)"
        )

        await agent.add_tool(
            self.get_message_reactions,
            "discord_get_message_reactions",
            description="Get reactions from a Discord message. "
                        "Args: channel_id (int), message_id (int), emoji (str, optional). "
                        "If emoji is specified, only returns data for that specific reaction. "
                        "Returns: Dict with reaction data including emoji, count, and users who reacted. "
                        "Example: discord_get_message_reactions(channel_id=123456789, message_id=987654321) "
                        "or discord_get_message_reactions(channel_id=123456789, message_id=987654321, emoji='👍')"
        )

        await agent.add_tool(
            self.add_reaction,
            "discord_add_reaction",
            description="Add a reaction emoji to a message. "
                       "Args: channel_id (int), message_id (int), emoji (str). "
                       "Returns: Dict with success status. "
                       "Example: discord_add_reaction(channel_id=123, message_id=456, emoji='👍')"
        )

        await agent.add_tool(
            self.remove_reaction,
            "discord_remove_reaction",
            description="Remove a reaction from a message. "
                       "Args: channel_id (int), message_id (int), emoji (str), user_id (int, optional). "
                       "Returns: Dict with success status. "
                       "Example: discord_remove_reaction(channel_id=123, message_id=456, emoji='👍')"
        )

        # Voice Control Tools
        await agent.add_tool(
            self.join_voice_channel,
            "discord_join_voice",
            description="Join a voice channel. "
                       "Args: channel_id (int). "
                       "Returns: Dict with success status and channel info. "
                       "Example: discord_join_voice(channel_id=123456789)"
        )

        await agent.add_tool(
            self.leave_voice_channel,
            "discord_leave_voice",
            description="Leave the current voice channel in a guild. "
                       "Args: guild_id (int). "
                       "Returns: Dict with success status. "
                       "Example: discord_leave_voice(guild_id=123456789)"
        )

        await agent.add_tool(
            self.get_voice_status,
            "discord_get_voice_status",
            description="Get voice connection status for a guild. "
                       "Args: guild_id (int). "
                       "Returns: Dict with voice status (connected, channel, playing, listening, tts_enabled, etc.). "
                       "Example: discord_get_voice_status(guild_id=123456789)"
        )

        await agent.add_tool(
            self.toggle_tts,
            "discord_toggle_tts",
            description="Toggle TTS (Text-to-Speech) on/off. "
                       "Args: guild_id (int), mode (str, optional: 'elevenlabs'/'piper'/'off'/None to toggle). "
                       "Returns: Dict with TTS status. "
                       "Example: discord_toggle_tts(guild_id=123, mode='piper')"
        )

        await agent.add_tool(
            self.send_tts_message,
            "discord_send_tts_message",
            description="Send a TTS (Text-to-Speech) message in the current voice channel. "
                       "Args: guild_id (int), text (str), mode (str, optional: 'elevenlabs'/'piper'). "
                       "Returns: Dict with success status and TTS info. "
                       "Example: discord_send_tts_message(guild_id=123, text='Hello from voice!', mode='piper')"
        )

        await agent.add_tool(
            self.can_hear_user,
            "discord_can_hear_user",
            description="Check if the bot can hear a specific user (voice listening status). "
                       "Verifies: bot in voice, listening enabled, user in same channel, user not muted. "
                       "Args: guild_id (int), user_id (int). "
                       "Returns: Dict with can_hear (bool), reason, voice_channel, users_in_channel. "
                       "Example: discord_can_hear_user(guild_id=123, user_id=456)"
        )

        # Role & Permission Tools
        await agent.add_tool(
            self.get_member_roles,
            "discord_get_member_roles",
            description="Get all roles of a member in a guild. "
                       "Args: guild_id (int), user_id (int). "
                       "Returns: List of role dicts with id, name, color, position, permissions. "
                       "Example: discord_get_member_roles(guild_id=123, user_id=456)"
        )

        await agent.add_tool(
            self.add_role,
            "discord_add_role",
            description="Add a role to a member. "
                       "Args: guild_id (int), user_id (int), role_id (int), reason (str, optional). "
                       "Returns: Dict with success status. "
                       "Example: discord_add_role(guild_id=123, user_id=456, role_id=789, reason='Promotion')"
        )

        await agent.add_tool(
            self.remove_role,
            "discord_remove_role",
            description="Remove a role from a member. "
                       "Args: guild_id (int), user_id (int), role_id (int), reason (str, optional). "
                       "Returns: Dict with success status. "
                       "Example: discord_remove_role(guild_id=123, user_id=456, role_id=789)"
        )

        # Lifetime Management Tools
        await agent.add_tool(
            self.get_bot_status,
            "discord_get_bot_status",
            description="Get current bot status and statistics. "
                       "Returns: Dict with bot info (latency, guilds, users, voice_connections, kernel_state, etc.). "
                       "Example: discord_get_bot_status()"
        )

        await agent.add_tool(
            self.set_bot_status,
            "discord_set_bot_status",
            description="Set bot's Discord status and activity. "
                       "Args: status (str: 'online'/'idle'/'dnd'/'invisible'), "
                       "activity_type (str: 'playing'/'watching'/'listening'/'streaming'), "
                       "activity_name (str, optional). "
                       "Returns: Dict with success status. "
                       "Example: discord_set_bot_status(status='online', activity_type='playing', activity_name='with AI')"
        )

        await agent.add_tool(
            self.get_kernel_metrics,
            "discord_get_kernel_metrics",
            description="Get kernel performance metrics. "
                       "Returns: Dict with metrics (total_signals, user_inputs, agent_responses, proactive_actions, errors, avg_response_time). "
                       "Example: discord_get_kernel_metrics()"
        )

        # Server Management
        await agent.add_tool(
            self.create_server,
            "discord_create_server",
            description="Create a new Discord server. Args: name (str), icon (str, optional base64), region (str, optional). Returns: Dict with guild_id and info."
        )

        await agent.add_tool(
            self.delete_server,
            "discord_delete_server",
            description="Delete a Discord server (bot must be owner). Args: guild_id (int). Returns: Dict with success status."
        )

        await agent.add_tool(
            self.edit_server,
            "discord_edit_server",
            description="Edit server settings. Args: guild_id (int), name (str, optional), icon (str, optional), description (str, optional), verification_level (int 0-4, optional). Returns: Dict with success status."
        )

        # Channel Management
        await agent.add_tool(
            self.create_channel,
            "discord_create_channel",
            description="Create a channel. Args: guild_id (int), name (str), channel_type (str: 'text'/'voice'/'category'/'stage'), category_id (int, optional), topic (str, optional), slowmode_delay (int, optional), nsfw (bool, optional). Returns: Dict with channel info."
        )

        await agent.add_tool(
            self.delete_channel,
            "discord_delete_channel",
            description="Delete a channel. Args: channel_id (int), reason (str, optional). Returns: Dict with success status."
        )

        await agent.add_tool(
            self.edit_channel,
            "discord_edit_channel",
            description="Edit channel settings. Args: channel_id (int), name (str, optional), topic (str, optional), slowmode_delay (int, optional), nsfw (bool, optional), position (int, optional). Returns: Dict with success status."
        )

        # Thread Management
        await agent.add_tool(
            self.create_thread,
            "discord_create_thread",
            description="Create a thread. Args: channel_id (int), name (str), message_id (int, optional), auto_archive_duration (int: 60/1440/4320/10080 minutes). Returns: Dict with thread info."
        )

        await agent.add_tool(
            self.join_thread,
            "discord_join_thread",
            description="Join a thread. Args: thread_id (int). Returns: Dict with success status."
        )

        await agent.add_tool(
            self.leave_thread,
            "discord_leave_thread",
            description="Leave a thread. Args: thread_id (int). Returns: Dict with success status."
        )

        # Moderation
        await agent.add_tool(
            self.kick_member,
            "discord_kick_member",
            description="Kick a member. Args: guild_id (int), user_id (int), reason (str, optional). Returns: Dict with success status."
        )

        await agent.add_tool(
            self.ban_member,
            "discord_ban_member",
            description="Ban a member. Args: guild_id (int), user_id (int), reason (str, optional), delete_message_days (int 0-7, optional). Returns: Dict with success status."
        )

        await agent.add_tool(
            self.unban_member,
            "discord_unban_member",
            description="Unban a member. Args: guild_id (int), user_id (int), reason (str, optional). Returns: Dict with success status."
        )

        await agent.add_tool(
            self.timeout_member,
            "discord_timeout_member",
            description="Timeout (mute) a member. Args: guild_id (int), user_id (int), duration_minutes (int, max 40320), reason (str, optional). Returns: Dict with timeout info."
        )

        await agent.add_tool(
            self.remove_timeout,
            "discord_remove_timeout",
            description="Remove timeout from member. Args: guild_id (int), user_id (int), reason (str, optional). Returns: Dict with success status."
        )

        await agent.add_tool(
            self.change_nickname,
            "discord_change_nickname",
            description="Change member nickname. Args: guild_id (int), user_id (int), nickname (str or None), reason (str, optional). Returns: Dict with success status."
        )

        await agent.add_tool(
            self.move_member,
            "discord_move_member",
            description="Move member to voice channel. Args: guild_id (int), user_id (int), channel_id (int). Returns: Dict with success status."
        )

        await agent.add_tool(
            self.disconnect_member,
            "discord_disconnect_member",
            description="Disconnect member from voice. Args: guild_id (int), user_id (int). Returns: Dict with success status."
        )

        # Files & Permissions
        await agent.add_tool(
            self.send_file,
            "discord_send_file",
            description="Send a file. Args: channel_id (int), file_path (str), filename (str, optional), content (str, optional). Returns: Dict with message info."
        )

        await agent.add_tool(
            self.set_channel_permissions,
            "discord_set_channel_permissions",
            description="Set channel permissions. Args: channel_id (int), target_id (int), target_type (str: 'role'/'member'), allow (int bitfield, optional), deny (int bitfield, optional), reason (str, optional). Returns: Dict with success status."
        )

        # DM & Webhooks
        await agent.add_tool(
            self.send_dm,
            "discord_send_dm",
            description="Send a DM to user. Args: user_id (int), content (str), embed (dict, optional). Returns: Dict with message info."
        )

        await agent.add_tool(
            self.create_webhook,
            "discord_create_webhook",
            description="Create a webhook. Args: channel_id (int), name (str), avatar (bytes, optional). Returns: Dict with webhook URL and info."
        )


        # Add these to the export_to_agent() method:

        # Invitation Management
        await agent.add_tool(
            self.create_invite,
            "discord_create_invite",
            description="Create a server invitation link. "
                        "Args: channel_id (int), max_age (int, seconds until expiry, 0=never, default 86400=24h), "
                        "max_uses (int, 0=unlimited), temporary (bool, temporary membership), unique (bool, create unique invite), "
                        "reason (str, optional). "
                        "Returns: Dict with invite_code, invite_url, expiration info. "
                        "Example: discord_create_invite(channel_id=123, max_age=3600, max_uses=10)"
        )

        await agent.add_tool(
            self.get_invites,
            "discord_get_invites",
            description="Get all invites for a server. "
                        "Args: guild_id (int). "
                        "Returns: List of invite dicts with code, URL, uses, max_uses, expiration. "
                        "Example: discord_get_invites(guild_id=123456789)"
        )

        await agent.add_tool(
            self.delete_invite,
            "discord_delete_invite",
            description="Delete/revoke an invite. "
                        "Args: invite_code (str, just the code not full URL), reason (str, optional). "
                        "Returns: Dict with success status. "
                        "Example: discord_delete_invite(invite_code='abc123XYZ')"
        )

        await agent.add_tool(
            self.get_invite_info,
            "discord_get_invite_info",
            description="Get information about an invite without joining. "
                        "Args: invite_code (str). "
                        "Returns: Dict with guild info, member counts, expiration. "
                        "Example: discord_get_invite_info(invite_code='abc123XYZ')"
        )

        # Add these to the export_to_agent() method:

        # Template Message Management
        await agent.add_tool(
            self.create_message_template,
            "discord_create_message_template",
            description="Create a reusable message template. "
                        "Args: template_name (str), content (str, optional), embed (dict, optional), components (list, optional). "
                        "Supports variable substitution with {variable_name} syntax. "
                        "Returns: Dict with template info. "
                        "Example: discord_create_message_template('welcome', content='Hello {username}!', embed={'title': 'Welcome', 'description': '{username} joined'})"
        )

        await agent.add_tool(
            self.get_message_template,
            "discord_get_message_template",
            description="Get a message template by name. "
                        "Args: template_name (str). "
                        "Returns: Dict with template data. "
                        "Example: discord_get_message_template('welcome')"
        )

        await agent.add_tool(
            self.list_message_templates,
            "discord_list_message_templates",
            description="List all available message templates. "
                        "Returns: List of template info dicts. "
                        "Example: discord_list_message_templates()"
        )

        await agent.add_tool(
            self.delete_message_template,
            "discord_delete_message_template",
            description="Delete a message template. "
                        "Args: template_name (str). "
                        "Returns: Dict with success status. "
                        "Example: discord_delete_message_template('old_template')"
        )

        await agent.add_tool(
            self.send_template_message,
            "discord_send_template_message",
            description="Send a message using a template with variable substitution. "
                        "Args: channel_id (int), template_name (str), variables (dict, optional), reply_to (int, optional). "
                        "Variables replace {key} in template with values. "
                        "Returns: Dict with message info. "
                        "Example: discord_send_template_message(channel_id=123, template_name='welcome', variables={'username': 'John', 'member_count': '100'})"
        )

        await agent.add_tool(
            self.create_welcome_template,
            "discord_create_welcome_template",
            description="Create a welcome message template. "
                        "Args: template_name (str, default 'welcome'), title (str), description (str), color (int hex), "
                        "thumbnail (str, optional), image (str, optional), fields (list, optional). "
                        "Supports variables: {username}, {server_name}, {member_count}. "
                        "Returns: Dict with template info. "
                        "Example: discord_create_welcome_template(title='Welcome {username}!', description='Welcome to {server_name}')"
        )

        await agent.add_tool(
            self.create_announcement_template,
            "discord_create_announcement_template",
            description="Create an announcement template. "
                        "Args: template_name (str, default 'announcement'), title (str), description (str), "
                        "color (int hex), mention_role (str, optional like '@everyone'). "
                        "Supports variables: {message}, {date}. "
                        "Returns: Dict with template info. "
                        "Example: discord_create_announcement_template(title='Update', description='{message}')"
        )

        await agent.add_tool(
            self.create_poll_template,
            "discord_create_poll_template",
            description="Create a poll template with reaction options. "
                        "Args: template_name (str, default 'poll'), question (str), options (list of str, max 10). "
                        "Supports variables: {question}, {option1}, {option2}, etc. "
                        "Returns: Dict with template info. "
                        "Example: discord_create_poll_template(question='Favorite color?', options=['Red', 'Blue', 'Green'])"
        )

        await agent.add_tool(
            self.create_embed_template,
            "discord_create_embed_template",
            description="Create a custom embed template with all options. "
                        "Args: template_name (str), title (str, optional), description (str, optional), "
                        "color (int hex, default 0x3498db), fields (list, optional), footer (str, optional), "
                        "author (str, optional), thumbnail (str URL, optional), image (str URL, optional), url (str, optional). "
                        "All text fields support variable substitution. "
                        "Returns: Dict with template info. "
                        "Example: discord_create_embed_template('info', title='{title}', description='{content}', color=0xff0000)"
        )

        await agent.add_tool(
            self.create_button_template,
            "discord_create_button_template",
            description="Create a message template with buttons. "
                        "Args: template_name (str), content (str, optional), "
                        "buttons (list of dicts with keys: label, style='primary'/'secondary'/'success'/'danger'/'link', "
                        "custom_id, emoji, url, disabled). "
                        "Returns: Dict with template info. "
                        "Example: discord_create_button_template('menu', buttons=[{'label': 'Click me', 'style': 'primary', 'custom_id': 'btn1'}])"
        )

        await agent.add_tool(
            self.create_select_menu_template,
            "discord_create_select_menu_template",
            description="Create a message template with a select menu (dropdown). "
                        "Args: template_name (str), content (str, optional), placeholder (str), "
                        "options (list of dicts with keys: label, value, description, emoji), "
                        "min_values (int, default 1), max_values (int, default 1). "
                        "Returns: Dict with template info. "
                        "Example: discord_create_select_menu_template('roles', options=[{'label': 'Admin', 'value': 'admin'}, {'label': 'User', 'value': 'user'}])"
        )

        # Information & Help Tools
        await agent.add_tool(
            self.get_template_help,
            "discord_get_template_help",
            description="Get comprehensive help on creating and using message templates. "
                        "Returns detailed documentation including: variable substitution, template types, "
                        "workflow examples, color codes, best practices, common use cases, and tips. "
                        "No arguments required. "
                        "Returns: Dict with complete template documentation. "
                        "Example: discord_get_template_help()"
        )

        await agent.add_tool(
            self.get_tools_overview,
            "discord_get_tools_overview",
            description="Get overview of all 56+ available Discord tools organized by category. "
                        "Includes: server management, channels, messages, templates, moderation, roles, "
                        "voice, threads, invites, reactions, permissions, DMs, webhooks, bot status. "
                        "Each category includes tool names, descriptions, and usage examples. "
                        "No arguments required. "
                        "Returns: Dict with categorized tool information and quick-start workflows. "
                        "Example: discord_get_tools_overview()"
        )

        await agent.add_tool(
            self.get_template_examples,
            "discord_get_template_examples",
            description="Get practical, ready-to-use template examples for common scenarios. "
                        "Includes complete code examples for: welcome messages, moderation logs, "
                        "verification systems, role selection, polls, event announcements, leaderboards, "
                        "ticket systems, status updates, help menus, giveaways, server rules, level-up notifications. "
                        "Each example includes full implementation code and expected results. "
                        "No arguments required. "
                        "Returns: Dict with 12+ complete template examples. "
                        "Example: discord_get_template_examples()"
        )

        print("✓ Discord tools exported to agent (59 tools total)")


