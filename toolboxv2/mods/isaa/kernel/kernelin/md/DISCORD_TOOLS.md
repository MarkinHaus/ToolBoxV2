# Discord Agent Tools Documentation

## Overview

The Discord Kernel now provides **21 Discord-specific tools** that are automatically exported to the agent. These tools enable the agent to interact with Discord servers, channels, users, messages, voice channels, and manage the bot's lifetime.

## Tool Categories

### 1. Server Management (4 tools)

#### `discord_get_server_info(guild_id: Optional[int] = None)`
Get information about Discord server(s).
- **Args**: `guild_id` (int, optional) - If None, returns all servers
- **Returns**: Dict with server info (name, member_count, channels, roles, etc.)
- **Example**: `info = await discord_get_server_info(guild_id=123456789)`

#### `discord_get_channel_info(channel_id: int)`
Get information about a Discord channel.
- **Args**: `channel_id` (int)
- **Returns**: Dict with channel info (name, type, topic, members, etc.)
- **Example**: `info = await discord_get_channel_info(channel_id=987654321)`

#### `discord_list_channels(guild_id: int, channel_type: Optional[str] = None)`
List all channels in a guild.
- **Args**: 
  - `guild_id` (int)
  - `channel_type` (str, optional): 'text', 'voice', 'category', 'stage'
- **Returns**: List of channel dicts
- **Example**: `channels = await discord_list_channels(guild_id=123, channel_type='text')`

#### `discord_get_user_info(user_id: int, guild_id: Optional[int] = None)`
Get information about a Discord user.
- **Args**: 
  - `user_id` (int)
  - `guild_id` (int, optional) - For member-specific info
- **Returns**: Dict with user info (name, roles, voice_channel, etc.)
- **Example**: `info = await discord_get_user_info(user_id=111, guild_id=222)`

---

### 2. Message Management (7 tools)

#### `discord_send_message(channel_id: int, content: str, embed: Optional[Dict] = None, reply_to: Optional[int] = None)`
Send a message to a Discord channel.
- **Args**: 
  - `channel_id` (int)
  - `content` (str)
  - `embed` (dict, optional): `{'title': str, 'description': str, 'color': int, 'fields': [...]}`
  - `reply_to` (int, optional) - Message ID to reply to
- **Returns**: Dict with message_id and timestamp
- **Example**: `result = await discord_send_message(channel_id=123, content='Hello!', reply_to=456)`

#### `discord_edit_message(channel_id: int, message_id: int, new_content: Optional[str] = None, new_embed: Optional[Dict] = None)`
Edit an existing message.
- **Example**: `result = await discord_edit_message(channel_id=123, message_id=456, new_content='Updated!')`

#### `discord_delete_message(channel_id: int, message_id: int, delay: float = 0)`
Delete a message.
- **Args**: `delay` (float, optional) - Delay in seconds before deletion
- **Example**: `result = await discord_delete_message(channel_id=123, message_id=456, delay=5.0)`

#### `discord_get_message(channel_id: int, message_id: int)`
Get information about a specific message.
- **Returns**: Dict with message info (content, author, embeds, reactions, etc.)
- **Example**: `msg = await discord_get_message(channel_id=123, message_id=456)`

#### `discord_get_recent_messages(channel_id: int, limit: int = 10, before: Optional[int] = None, after: Optional[int] = None)`
Get recent messages from a channel.
- **Args**: 
  - `limit` (int, default 10, max 100)
  - `before` (int, optional) - Message ID
  - `after` (int, optional) - Message ID
- **Returns**: List of message dicts
- **Example**: `messages = await discord_get_recent_messages(channel_id=123, limit=20)`

#### `discord_add_reaction(channel_id: int, message_id: int, emoji: str)`
Add a reaction emoji to a message.
- **Example**: `result = await discord_add_reaction(channel_id=123, message_id=456, emoji='👍')`

#### `discord_remove_reaction(channel_id: int, message_id: int, emoji: str, user_id: Optional[int] = None)`
Remove a reaction from a message.
- **Example**: `result = await discord_remove_reaction(channel_id=123, message_id=456, emoji='👍')`

---

### 3. Voice Control (4 tools)

#### `discord_join_voice(channel_id: int)`
Join a voice channel.
- **Returns**: Dict with success status and channel info
- **Example**: `result = await discord_join_voice(channel_id=123456789)`

#### `discord_leave_voice(guild_id: int)`
Leave the current voice channel in a guild.
- **Example**: `result = await discord_leave_voice(guild_id=123456789)`

#### `discord_get_voice_status(guild_id: int)`
Get voice connection status for a guild.
- **Returns**: Dict with voice status (connected, channel, playing, listening, tts_enabled, etc.)
- **Example**: `status = await discord_get_voice_status(guild_id=123456789)`

#### `discord_toggle_tts(guild_id: int, mode: Optional[str] = None)`
Toggle TTS (Text-to-Speech) on/off.
- **Args**: `mode` (str, optional): 'elevenlabs', 'piper', 'off', or None to toggle
- **Returns**: Dict with TTS status
- **Example**: `result = await discord_toggle_tts(guild_id=123, mode='piper')`

---

### 4. Role & Permission Management (3 tools)

#### `discord_get_member_roles(guild_id: int, user_id: int)`
Get all roles of a member in a guild.
- **Returns**: List of role dicts with id, name, color, position, permissions
- **Example**: `roles = await discord_get_member_roles(guild_id=123, user_id=456)`

#### `discord_add_role(guild_id: int, user_id: int, role_id: int, reason: Optional[str] = None)`
Add a role to a member.
- **Example**: `result = await discord_add_role(guild_id=123, user_id=456, role_id=789, reason='Promotion')`

#### `discord_remove_role(guild_id: int, user_id: int, role_id: int, reason: Optional[str] = None)`
Remove a role from a member.
- **Example**: `result = await discord_remove_role(guild_id=123, user_id=456, role_id=789)`

---

### 5. Lifetime Management (3 tools)

#### `discord_get_bot_status()`
Get current bot status and statistics.
- **Returns**: Dict with bot info (latency, guilds, users, voice_connections, kernel_state, etc.)
- **Example**: `status = await discord_get_bot_status()`

#### `discord_set_bot_status(status: str = "online", activity_type: str = "playing", activity_name: Optional[str] = None)`
Set bot's Discord status and activity.
- **Args**: 
  - `status` (str): 'online', 'idle', 'dnd', 'invisible'
  - `activity_type` (str): 'playing', 'watching', 'listening', 'streaming'
  - `activity_name` (str, optional)
- **Example**: `result = await discord_set_bot_status(status='online', activity_type='playing', activity_name='with AI')`

#### `discord_get_kernel_metrics()`
Get kernel performance metrics.
- **Returns**: Dict with metrics (total_signals, user_inputs, agent_responses, proactive_actions, errors, avg_response_time)
- **Example**: `metrics = await discord_get_kernel_metrics()`

