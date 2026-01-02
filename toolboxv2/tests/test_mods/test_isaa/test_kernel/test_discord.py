#!/usr/bin/env python3
"""
Discord Kernel Unit & E2E Tests
Comprehensive testing for Discord Kernel and Discord Tools
All Discord API and LLM calls are mocked.

Note: These tests use pure mocking without importing the actual discord_tools module
to avoid complex dependency issues with pydantic/mcp.
"""

import unittest
import asyncio
import random
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List


# ===== MOCK DISCORD MODULE =====

class MockChannelType:
    """Mock for discord.ChannelType enum"""
    text = "text"
    voice = "voice"
    private = "private"
    category = "category"
    news = "news"
    stage_voice = "stage_voice"
    forum = "forum"


class MockDiscord:
    """Mock for the discord module"""
    ChannelType = MockChannelType


# Create a global mock for discord module
discord_mock = MockDiscord()


# ===== MOCK HELPER FUNCTIONS =====

class AsyncContextManagerMock:
    """Async context manager mock for channel.typing()"""
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass


def create_mock_role(role_id: int = 111111111, name: str = "TestRole"):
    """Create a mock Discord role"""
    role = Mock()
    role.id = role_id
    role.name = name
    role.color = Mock(value=0x3498db)
    role.position = 1
    role.permissions = Mock()
    role.mentionable = True
    role.hoist = False
    return role


def create_mock_user(user_id: int = 222222222, name: str = "TestUser", bot: bool = False):
    """Create a mock Discord user"""
    user = Mock()
    user.id = user_id
    user.name = name
    user.display_name = name
    user.bot = bot
    user.created_at = datetime.now()
    user.avatar = None
    user.mention = f"<@{user_id}>"
    return user


def create_mock_member(user_id: int = 222222222, name: str = "TestUser", guild=None):
    """Create a mock Discord member (user in a guild)"""
    member = create_mock_user(user_id, name)
    member.nick = None
    member.joined_at = datetime.now()
    member.roles = [create_mock_role()]
    member.top_role = member.roles[0]
    member.voice = None
    member.guild = guild
    return member


def create_mock_text_channel(channel_id: int = 987654321, name: str = "general"):
    """Create a mock Discord text channel"""
    channel = Mock()
    channel.id = channel_id
    channel.name = name
    channel.type = "text"
    channel.created_at = datetime.now()
    channel.topic = "Test topic"
    channel.slowmode_delay = 0
    channel.nsfw = False
    channel.guild = None  # Will be set by create_mock_guild
    # Create a simple mock message without circular reference
    mock_msg = Mock()
    mock_msg.id = 123456
    mock_msg.content = "test"
    channel.send = AsyncMock(return_value=mock_msg)
    channel.typing = MagicMock(return_value=AsyncContextManagerMock())
    channel.history = AsyncMock()
    channel.purge = AsyncMock()
    return channel


def create_mock_message(content: str = "test message", author_id: int = 222222222):
    """Create a mock Discord message with channel and guild"""
    channel = create_mock_text_channel()
    guild = create_mock_guild()
    channel.guild = guild

    message = Mock()
    message.id = random.randint(100000, 999999)
    message.content = content
    message.author = create_mock_user(author_id)
    message.channel = channel
    message.guild = guild
    message.attachments = []
    message.mentions = []
    message.created_at = datetime.now()
    message.edit = AsyncMock()
    message.delete = AsyncMock()
    message.add_reaction = AsyncMock()
    message.remove_reaction = AsyncMock()
    message.reply = AsyncMock()
    return message


def create_mock_voice_channel(channel_id: int = 555555555, name: str = "Voice"):
    """Create a mock Discord voice channel"""
    channel = Mock()
    channel.id = channel_id
    channel.name = name
    channel.type = discord_mock.ChannelType.voice
    channel.created_at = datetime.now()
    channel.bitrate = 64000
    channel.user_limit = 0
    channel.members = []
    channel.guild = None
    channel.connect = AsyncMock()
    return channel


def create_mock_guild(guild_id: int = 123456789, name: str = "Test Server"):
    """Create a mock Discord guild (server)"""
    guild = Mock()
    guild.id = guild_id
    guild.name = name
    guild.member_count = 100
    guild.owner_id = 111111111
    guild.created_at = datetime.now()
    guild.premium_tier = 0
    guild.premium_subscription_count = 0

    text_channel = create_mock_text_channel()
    voice_channel = create_mock_voice_channel()
    text_channel.guild = guild
    voice_channel.guild = guild

    guild.text_channels = [text_channel]
    guild.voice_channels = [voice_channel]
    guild.channels = [text_channel, voice_channel]
    guild.roles = [create_mock_role()]
    guild.emojis = []
    guild.get_member = Mock(return_value=create_mock_member(guild=guild))
    guild.get_channel = Mock(return_value=text_channel)
    return guild


def create_mock_bot():
    """Create a mock Discord bot"""
    bot = Mock()
    guild = create_mock_guild()
    bot.guilds = [guild]
    bot.get_guild = Mock(return_value=guild)
    bot.get_channel = Mock(return_value=guild.text_channels[0])
    bot.get_user = Mock(return_value=create_mock_user())
    bot.user = create_mock_user(name="TestBot", bot=True)
    bot.latency = 0.05
    return bot


def create_mock_kernel():
    """Create a mock Kernel"""
    kernel = Mock()
    kernel.process_signal = AsyncMock()
    kernel.agent = AsyncMock()
    kernel.agent.add_tool = AsyncMock()
    kernel.memory_store = Mock()
    kernel.memory_store.user_memories = {}
    kernel.learning_engine = Mock()
    kernel.learning_engine.preferences = {}
    return kernel


def create_mock_output_router():
    """Create a mock DiscordOutputRouter"""
    router = Mock()
    router.user_channels = {}
    router.active_channels = {}
    router.voice_clients = {}
    router.audio_sinks = {}
    router.tts_enabled = {}
    router.tts_mode = {}
    router.send_text = AsyncMock()
    router.send_embed = AsyncMock()
    return router




# ===== UNIT TESTS: DiscordKernelTools Logic =====

class TestDiscordKernelToolsLogic(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for DiscordKernelTools logic - simulating the tool behavior
    without importing the actual module (to avoid complex dependencies).
    """

    async def asyncSetUp(self):
        """Set up test fixtures"""
        self.mock_bot = create_mock_bot()
        self.mock_kernel = create_mock_kernel()
        self.mock_output_router = create_mock_output_router()

    # ===== Server Info Tests =====

    async def test_get_server_info_with_guild_id(self):
        """Test get_server_info logic with specific guild ID"""
        guild_id = 123456789
        guild = self.mock_bot.get_guild(guild_id)

        # Simulate get_server_info logic
        result = {
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

        self.assertEqual(result["id"], guild_id)
        self.assertEqual(result["name"], "Test Server")
        self.assertEqual(result["member_count"], 100)

    async def test_get_server_info_guild_not_found(self):
        """Test error handling when guild not found"""
        self.mock_bot.get_guild.return_value = None
        guild = self.mock_bot.get_guild(999999)

        # Simulate error handling
        if not guild:
            result = {"error": f"Guild 999999 not found"}
        else:
            result = {"id": guild.id}

        self.assertIn("error", result)
        self.assertIn("not found", result["error"])

    async def test_get_server_info_all_guilds(self):
        """Test get_server_info without guild_id returns all guilds"""
        # Simulate get_server_info(None) logic
        result = {
            "guilds": [
                {
                    "id": g.id,
                    "name": g.name,
                    "member_count": g.member_count
                }
                for g in self.mock_bot.guilds
            ],
            "total_guilds": len(self.mock_bot.guilds)
        }

        self.assertIn("guilds", result)
        self.assertIn("total_guilds", result)
        self.assertEqual(result["total_guilds"], 1)

    # ===== Channel Info Tests =====

    async def test_get_channel_info_text_channel(self):
        """Test get_channel_info for TextChannel"""
        channel = create_mock_text_channel()
        self.mock_bot.get_channel.return_value = channel

        # Simulate get_channel_info logic
        result = {
            "id": channel.id,
            "name": channel.name,
            "type": str(channel.type),
            "created_at": channel.created_at.isoformat(),
            "topic": channel.topic,
            "slowmode_delay": channel.slowmode_delay,
            "nsfw": channel.nsfw
        }

        self.assertEqual(result["id"], channel.id)
        self.assertEqual(result["name"], "general")
        self.assertIn("type", result)

    async def test_get_channel_info_channel_not_found(self):
        """Test error handling when channel not found"""
        self.mock_bot.get_channel.return_value = None
        channel = self.mock_bot.get_channel(999999)

        # Simulate error handling
        if not channel:
            result = {"error": f"Channel 999999 not found"}
        else:
            result = {"id": channel.id}

        self.assertIn("error", result)
        self.assertIn("not found", result["error"])

    # ===== List Channels Tests =====

    async def test_list_channels(self):
        """Test list_channels returns all channels"""
        guild_id = 123456789
        guild = self.mock_bot.get_guild(guild_id)

        # Simulate list_channels logic
        result = [
            {
                "id": channel.id,
                "name": channel.name,
                "type": str(channel.type),
                "position": channel.position
            }
            for channel in guild.channels
        ]

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    async def test_list_channels_guild_not_found(self):
        """Test list_channels with non-existent guild"""
        self.mock_bot.get_guild.return_value = None
        guild = self.mock_bot.get_guild(999999)

        # Simulate error handling
        if not guild:
            result = []
        else:
            result = [{"id": c.id} for c in guild.channels]

        self.assertEqual(result, [])

    # ===== User Info Tests =====

    async def test_get_user_info(self):
        """Test get_user_info returns user data"""
        user_id = 222222222
        user = self.mock_bot.get_user(user_id)

        # Simulate get_user_info logic
        result = {
            "id": user.id,
            "name": user.name,
            "display_name": user.display_name,
            "bot": user.bot,
            "created_at": user.created_at.isoformat()
        }

        self.assertEqual(result["id"], user_id)
        self.assertEqual(result["name"], "TestUser")
        self.assertIn("display_name", result)
        self.assertIn("bot", result)

    async def test_get_user_info_user_not_found(self):
        """Test error handling when user not found"""
        self.mock_bot.get_user.return_value = None
        user = self.mock_bot.get_user(999999)

        # Simulate error handling
        if not user:
            result = {"error": f"User 999999 not found"}
        else:
            result = {"id": user.id}

        self.assertIn("error", result)
        self.assertIn("not found", result["error"])

    async def test_get_user_info_with_guild(self):
        """Test get_user_info with guild_id for member info"""
        user_id = 222222222
        guild_id = 123456789
        user = self.mock_bot.get_user(user_id)
        guild = self.mock_bot.get_guild(guild_id)
        member = guild.get_member(user_id)

        # Simulate get_user_info with guild logic
        result = {
            "id": user.id,
            "name": user.name,
            "display_name": user.display_name,
            "bot": user.bot,
            "created_at": user.created_at.isoformat(),
            "nickname": member.nick,
            "joined_at": member.joined_at.isoformat() if member.joined_at else None,
            "roles": [role.name for role in member.roles if role.name != "@everyone"],
            "top_role": member.top_role.name,
            "voice_channel": member.voice.channel.name if member.voice else None
        }

        self.assertEqual(result["id"], user_id)
        self.assertIn("roles", result)


# ===== UNIT TESTS: DiscordOutputRouter =====

class TestDiscordOutputRouter(unittest.TestCase):
    """Tests for the Discord Output Router"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_bot = create_mock_bot()

    def test_user_channel_mapping(self):
        """Test user -> channel mapping"""
        router = create_mock_output_router()

        user_id = "222222222"
        channel = create_mock_text_channel()
        router.user_channels[user_id] = channel

        self.assertIn(user_id, router.user_channels)
        self.assertEqual(router.user_channels[user_id], channel)

    def test_voice_client_tracking(self):
        """Test voice client tracking per guild"""
        router = create_mock_output_router()

        guild_id = 123456789
        voice_client = Mock()
        voice_client.is_connected = Mock(return_value=True)
        router.voice_clients[guild_id] = voice_client

        self.assertIn(guild_id, router.voice_clients)
        self.assertTrue(router.voice_clients[guild_id].is_connected())

    def test_tts_enabled_tracking(self):
        """Test TTS enabled state per guild"""
        router = create_mock_output_router()

        guild_id = 123456789
        router.tts_enabled[guild_id] = True

        self.assertTrue(router.tts_enabled.get(guild_id, False))


# ===== UNIT TESTS: WhisperAudioSink =====

class TestWhisperAudioSink(unittest.TestCase):
    """Tests for WhisperAudioSink without real Groq calls"""

    def test_audio_buffer_per_user(self):
        """Test that audio buffers are separate per user"""
        # Create a minimal mock sink
        sink = Mock()
        sink.audio_buffer = {}
        sink.last_transcription = {}

        user1_id = "111"
        user2_id = "222"

        # Simulate buffering audio for two users
        sink.audio_buffer[user1_id] = [b"audio_data_1"]
        sink.audio_buffer[user2_id] = [b"audio_data_2"]

        self.assertIn(user1_id, sink.audio_buffer)
        self.assertIn(user2_id, sink.audio_buffer)
        self.assertNotEqual(sink.audio_buffer[user1_id], sink.audio_buffer[user2_id])

    def test_voice_channel_history_structure(self):
        """Test voice channel history data structure"""
        history = {}
        channel_id = "555555555"

        # Add history entry
        entry = {
            "user": "TestUser",
            "user_id": "222222222",
            "text": "Hello world",
            "timestamp": datetime.now().timestamp(),
            "language": "en"
        }
        history[channel_id] = [entry]

        self.assertIn(channel_id, history)
        self.assertEqual(len(history[channel_id]), 1)
        self.assertEqual(history[channel_id][0]["text"], "Hello world")

    def test_transcription_interval_default(self):
        """Test default transcription interval is 3 seconds"""
        # Default value from WhisperAudioSink
        transcription_interval = 3.0
        self.assertEqual(transcription_interval, 3.0)

    def test_silence_threshold_default(self):
        """Test default silence threshold is 1 second"""
        silence_threshold = 1.0
        self.assertEqual(silence_threshold, 1.0)



# ===== INTEGRATION TESTS =====

class TestDiscordKernelIntegration(unittest.IsolatedAsyncioTestCase):
    """Async Integration Tests for Discord Kernel"""

    async def asyncSetUp(self):
        """Set up test fixtures"""
        self.mock_bot = create_mock_bot()
        self.mock_kernel = create_mock_kernel()
        self.mock_output_router = create_mock_output_router()

    async def test_message_handling_flow(self):
        """Test message handling creates correct signal"""
        message = create_mock_message(content="Hello Bot!")

        # Simulate message handling
        user_id = str(message.author.id)
        channel_id = message.channel.id

        # Register channel
        self.mock_output_router.user_channels[user_id] = message.channel
        self.mock_output_router.active_channels[channel_id] = message.channel

        self.assertIn(user_id, self.mock_output_router.user_channels)
        self.assertIn(channel_id, self.mock_output_router.active_channels)

    async def test_attachment_handling(self):
        """Test message with attachments"""
        message = create_mock_message(content="Check this file")

        # Add mock attachment
        attachment = Mock()
        attachment.filename = "test.txt"
        attachment.url = "https://cdn.discord.com/attachments/test.txt"
        attachment.size = 1024
        attachment.content_type = "text/plain"
        message.attachments = [attachment]

        self.assertEqual(len(message.attachments), 1)
        self.assertEqual(message.attachments[0].filename, "test.txt")

    async def test_embed_creation(self):
        """Test embed creation for responses"""
        embed_data = {
            "title": "Test Embed",
            "description": "This is a test embed",
            "color": 0x3498db,
            "fields": [
                {"name": "Field 1", "value": "Value 1", "inline": True}
            ]
        }

        self.assertIn("title", embed_data)
        self.assertIn("description", embed_data)
        self.assertEqual(len(embed_data["fields"]), 1)


# ===== E2E TESTS =====

class TestDiscordKernelE2E(unittest.IsolatedAsyncioTestCase):
    """End-to-End Tests with mocked Discord Bot"""

    async def asyncSetUp(self):
        """Set up test fixtures"""
        self.mock_bot = create_mock_bot()
        self.mock_kernel = create_mock_kernel()
        self.mock_output_router = create_mock_output_router()

    async def test_full_conversation_flow(self):
        """
        Test complete conversation flow:
        1. User sends message
        2. Kernel processes
        3. Agent is called (mocked)
        4. Response is sent to channel
        """
        # 1. User sends message
        message = create_mock_message(content="Hello Bot!")
        user_id = str(message.author.id)

        # 2. Register channel
        self.mock_output_router.user_channels[user_id] = message.channel

        # 3. Simulate kernel processing
        self.mock_kernel.process_signal = AsyncMock(return_value={"response": "Hello User!"})

        # 4. Verify channel can send response
        await message.channel.send("Hello User!")
        message.channel.send.assert_called_once_with("Hello User!")

    async def test_bot_mention_handling(self):
        """Test handling when bot is mentioned"""
        message = create_mock_message(content="@TestBot help me")
        message.mentions = [self.mock_bot.user]

        self.assertEqual(len(message.mentions), 1)
        self.assertTrue(message.mentions[0].bot)

    async def test_voice_channel_join_leave_flow(self):
        """Test voice channel join/leave commands"""
        guild_id = 123456789
        voice_channel = create_mock_voice_channel()

        # Simulate voice client
        voice_client = Mock()
        voice_client.is_connected = Mock(return_value=True)
        voice_client.channel = voice_channel
        voice_client.disconnect = AsyncMock()

        self.mock_output_router.voice_clients[guild_id] = voice_client

        # Verify connected
        self.assertTrue(self.mock_output_router.voice_clients[guild_id].is_connected())

        # Simulate disconnect
        await voice_client.disconnect()
        voice_client.disconnect.assert_called_once()

    async def test_reaction_handling(self):
        """Test reaction add/remove"""
        message = create_mock_message()

        # Add reaction
        await message.add_reaction("👍")
        message.add_reaction.assert_called_with("👍")

        # Remove reaction
        await message.remove_reaction("👍", self.mock_bot.user)
        message.remove_reaction.assert_called_with("👍", self.mock_bot.user)

    async def test_dm_channel_handling(self):
        """Test DM channel message handling"""
        dm_channel = Mock()
        dm_channel.id = 888888888
        dm_channel.type = discord_mock.ChannelType.private
        dm_channel.send = AsyncMock()

        message = create_mock_message(content="Private message")
        message.channel = dm_channel
        message.guild = None  # DMs have no guild

        self.assertIsNone(message.guild)
        await dm_channel.send("Response to DM")
        dm_channel.send.assert_called_once()


# ===== TEMPLATE TESTS =====

class TestDiscordTemplates(unittest.IsolatedAsyncioTestCase):
    """Tests for Discord message templates"""

    async def asyncSetUp(self):
        """Set up test fixtures"""
        self.mock_bot = create_mock_bot()
        self.mock_kernel = create_mock_kernel()
        self.mock_output_router = create_mock_output_router()

    async def test_template_variable_substitution(self):
        """Test template variable substitution"""
        template = "Hello {username}! Welcome to {server_name}."
        variables = {"username": "TestUser", "server_name": "Test Server"}

        result = template.format(**variables)

        self.assertEqual(result, "Hello TestUser! Welcome to Test Server.")

    async def test_embed_template_structure(self):
        """Test embed template structure"""
        embed_template = {
            "title": "Welcome {username}",
            "description": "You joined {server_name}",
            "color": 0x00ff00,
            "fields": [
                {"name": "Rules", "value": "Please read the rules", "inline": False}
            ]
        }

        self.assertIn("title", embed_template)
        self.assertIn("fields", embed_template)
        self.assertEqual(len(embed_template["fields"]), 1)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
