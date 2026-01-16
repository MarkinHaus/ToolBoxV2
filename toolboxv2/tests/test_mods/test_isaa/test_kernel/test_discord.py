#!/usr/bin/env python3
"""
Discord Kernel Unit Tests - Real Code Import Tests
Comprehensive testing for Discord Kernel and Discord Tools
Imports the real DiscordKernelTools code - only agent responses are mocked.

Tests:
- DiscordKernelTools methods with mocked bot/kernel/router
- Template system
- Voice status tracking
- Channel/User/Server info retrieval
- Message operations
- Moderation tools
- export_to_agent with categories and flags
"""

import unittest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, Any, List, Optional

# Import the real DiscordKernelTools
from toolboxv2.mods.isaa.kernel.kernelin.tools.discord_tools import DiscordKernelTools


# ===== MOCK DISCORD OBJECTS =====

class MockColor:
    """Mock for discord.Color"""
    def __init__(self, value: int = 0x3498db):
        self.value = value


class MockEmbed:
    """Mock for discord.Embed"""
    def __init__(self, title=None, description=None, color=None):
        self.title = title
        self.description = description
        self.color = color
        self.fields = []

    def add_field(self, name: str, value: str, inline: bool = False):
        self.fields.append({"name": name, "value": value, "inline": inline})


class MockChannelType:
    """Mock for discord.ChannelType"""
    text = "text"
    voice = "voice"
    private = "private"
    category = "category"
    news = "news"
    stage_voice = "stage_voice"


class MockRole:
    """Mock Discord Role"""
    def __init__(self, role_id: int = 111111111, name: str = "TestRole"):
        self.id = role_id
        self.name = name
        self.color = MockColor(0x3498db)
        self.position = 1
        self.permissions = Mock()
        self.permissions.value = 8
        self.mentionable = True
        self.hoist = False


class MockUser:
    """Mock Discord User"""
    def __init__(self, user_id: int = 222222222, name: str = "TestUser", bot: bool = False):
        self.id = user_id
        self.name = name
        self.display_name = name
        self.bot = bot
        self.created_at = datetime.now()
        self.avatar = None
        self.mention = f"<@{user_id}>"


class MockMember(MockUser):
    """Mock Discord Member (User in a Guild)"""
    def __init__(self, user_id: int = 222222222, name: str = "TestUser", guild=None):
        super().__init__(user_id, name)
        self.nick = None
        self.joined_at = datetime.now()
        self.roles = [MockRole()]
        self.top_role = self.roles[0]
        self.voice = None
        self.guild = guild
        self.add_roles = AsyncMock()
        self.remove_roles = AsyncMock()
        self.kick = AsyncMock()
        self.ban = AsyncMock()
        self.edit = AsyncMock()
        self.timeout = AsyncMock()


class MockMessage:
    """Mock Discord Message"""
    def __init__(self, content: str = "test", author: MockUser = None, channel=None):
        self.id = 123456789
        self.content = content
        self.author = author or MockUser()
        self.channel = channel
        self.guild = channel.guild if channel else None
        self.created_at = datetime.now()
        self.edited_at = None
        self.embeds = []
        self.attachments = []
        self.reactions = []
        self.edit = AsyncMock()
        self.delete = AsyncMock()
        self.add_reaction = AsyncMock()
        self.remove_reaction = AsyncMock()


class MockTextChannel:
    """Mock Discord TextChannel"""
    def __init__(self, channel_id: int = 987654321, name: str = "general", guild=None):
        self.id = channel_id
        self.name = name
        self.type = MockChannelType.text
        self.created_at = datetime.now()
        self.topic = "Test channel topic"
        self.slowmode_delay = 0
        self.nsfw = False
        self.position = 0
        self.guild = guild

        # Mock message for send
        self._mock_message = MockMessage(channel=self)
        self._mock_message.id = 999888777
        self._mock_message.created_at = datetime.now()

        self.send = AsyncMock(return_value=self._mock_message)
        self.fetch_message = AsyncMock(return_value=self._mock_message)
        self.history = self._create_history_mock()
        self.purge = AsyncMock()

    def _create_history_mock(self):
        """Create async iterator for history"""
        async def history_iter(*args, **kwargs):
            messages = [
                MockMessage(content=f"Message {i}", channel=self)
                for i in range(3)
            ]
            for msg in messages:
                yield msg
        return MagicMock(return_value=history_iter())


class MockVoiceChannel:
    """Mock Discord VoiceChannel"""
    def __init__(self, channel_id: int = 555555555, name: str = "Voice", guild=None):
        self.id = channel_id
        self.name = name
        self.type = MockChannelType.voice
        self.created_at = datetime.now()
        self.bitrate = 64000
        self.user_limit = 0
        self.members = []
        self.position = 0
        self.guild = guild
        self.connect = AsyncMock()


class MockVoiceClient:
    """Mock Discord VoiceClient"""
    def __init__(self, channel: MockVoiceChannel = None):
        self.channel = channel or MockVoiceChannel()
        self._connected = True
        self.disconnect = AsyncMock()
        self.move_to = AsyncMock()
        self.play = MagicMock()
        self.stop = MagicMock()
        self.is_playing = MagicMock(return_value=False)
        self.is_paused = MagicMock(return_value=False)
        self.latency = 0.05  # 50ms latency

    def is_connected(self):
        return self._connected


class MockGuild:
    """Mock Discord Guild (Server)"""
    def __init__(self, guild_id: int = 123456789, name: str = "Test Server"):
        self.id = guild_id
        self.name = name
        self.member_count = 100
        self.owner_id = 111111111
        self.created_at = datetime.now()
        self.premium_tier = 0
        self.premium_subscription_count = 0
        self.icon = None
        self.description = "Test server description"

        # Create channels
        self.text_channels = [MockTextChannel(guild=self)]
        self.voice_channels = [MockVoiceChannel(guild=self)]
        self.channels = self.text_channels + self.voice_channels
        self.roles = [MockRole()]
        self.emojis = []

        # Members
        self._members = {222222222: MockMember(222222222, "TestUser", guild=self)}

        self.get_member = Mock(side_effect=lambda uid: self._members.get(uid))
        self.get_channel = Mock(return_value=self.text_channels[0])
        self.get_role = Mock(return_value=self.roles[0])
        self.fetch_invites = AsyncMock(return_value=[])
        self.ban = AsyncMock()
        self.unban = AsyncMock()
        self.kick = AsyncMock()


class MockBot:
    """Mock Discord Bot"""
    def __init__(self):
        self.guild = MockGuild()
        self.guilds = [self.guild]
        self.user = MockUser(user_id=111111111, name="TestBot", bot=True)
        self.latency = 0.05

        self.get_guild = Mock(return_value=self.guild)
        self.get_channel = Mock(return_value=self.guild.text_channels[0])
        self.get_user = Mock(return_value=MockUser())
        self.fetch_user = AsyncMock(return_value=MockUser())


class MockKernel:
    """Mock Kernel with agent"""
    def __init__(self):
        self.agent = Mock()
        self.agent.add_tool = AsyncMock()
        self.state = "running"
        self.metrics = Mock()
        self.metrics.signals_processed = 100
        self.metrics.user_inputs_handled = 50
        self.metrics.proactive_actions = 10
        self.metrics.errors = 2
        self.metrics.average_response_time = 0.5
        self.metrics.avg_response_time = 0.5  # Alternative name used in some methods
        self.memory_store = Mock()
        self.memory_store.user_memories = {}
        self.learning_engine = Mock()
        self.learning_engine.preferences = {}


class MockOutputRouter:
    """Mock DiscordOutputRouter"""
    def __init__(self):
        self.user_channels = {}
        self.active_channels = {}
        self.voice_clients: Dict[int, MockVoiceClient] = {}
        self.audio_sinks = {}
        self.tts_enabled: Dict[int, bool] = {}
        self.tts_mode: Dict[int, str] = {}
        self.send_text = AsyncMock()
        self.send_embed = AsyncMock()
        self.send_media = AsyncMock(return_value={"success": True})


# ===== DISCORD MODULE MOCK =====

def create_discord_mock():
    """Create a mock for the discord module"""
    discord_mock = Mock()
    discord_mock.TextChannel = MockTextChannel
    discord_mock.VoiceChannel = MockVoiceChannel
    discord_mock.CategoryChannel = Mock
    discord_mock.StageChannel = Mock
    discord_mock.Embed = MockEmbed
    discord_mock.Color = MockColor
    discord_mock.ChannelType = MockChannelType
    discord_mock.NotFound = Exception
    discord_mock.Forbidden = Exception
    return discord_mock


# ===== UNIT TESTS: DiscordKernelTools with Real Code =====

class TestDiscordKernelToolsReal(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests using the real DiscordKernelTools class
    with mocked bot, kernel, and output_router.
    """

    async def asyncSetUp(self):
        """Set up test fixtures with real DiscordKernelTools"""
        self.mock_bot = MockBot()
        self.mock_kernel = MockKernel()
        self.mock_output_router = MockOutputRouter()

        # Create real DiscordKernelTools instance
        self.tools = DiscordKernelTools(
            bot=self.mock_bot,
            kernel=self.mock_kernel,
            output_router=self.mock_output_router
        )

    # ===== Server Info Tests =====

    async def test_get_server_info_with_guild_id(self):
        """Test get_server_info with specific guild ID"""
        result = await self.tools.get_server_info(guild_id=123456789)

        self.assertEqual(result["id"], 123456789)
        self.assertEqual(result["name"], "Test Server")
        self.assertEqual(result["member_count"], 100)
        self.assertIn("text_channels", result)
        self.assertIn("voice_channels", result)

    async def test_get_server_info_guild_not_found(self):
        """Test get_server_info when guild not found"""
        self.mock_bot.get_guild = Mock(return_value=None)

        result = await self.tools.get_server_info(guild_id=999999)

        self.assertIn("error", result)
        self.assertIn("not found", result["error"])

    async def test_get_server_info_all_guilds(self):
        """Test get_server_info without guild_id returns all guilds"""
        result = await self.tools.get_server_info()

        self.assertIn("guilds", result)
        self.assertIn("total_guilds", result)
        self.assertEqual(result["total_guilds"], 1)
        self.assertEqual(result["guilds"][0]["name"], "Test Server")

    # ===== Channel Info Tests =====

    async def test_get_channel_info(self):
        """Test get_channel_info for a text channel"""
        result = await self.tools.get_channel_info(channel_id=987654321)

        self.assertEqual(result["id"], 987654321)
        self.assertEqual(result["name"], "general")
        self.assertIn("type", result)
        self.assertIn("created_at", result)

    async def test_get_channel_info_not_found(self):
        """Test get_channel_info when channel not found"""
        self.mock_bot.get_channel = Mock(return_value=None)

        result = await self.tools.get_channel_info(channel_id=999999)

        self.assertIn("error", result)
        self.assertIn("not found", result["error"])

    # ===== List Channels Tests =====

    async def test_list_channels(self):
        """Test list_channels returns all channels"""
        result = await self.tools.list_channels(guild_id=123456789)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("id", result[0])
        self.assertIn("name", result[0])
        self.assertIn("type", result[0])

    async def test_list_channels_guild_not_found(self):
        """Test list_channels when guild not found"""
        self.mock_bot.get_guild = Mock(return_value=None)

        result = await self.tools.list_channels(guild_id=999999)

        self.assertEqual(result, [])

    # ===== User Info Tests =====

    async def test_get_user_info(self):
        """Test get_user_info returns user data"""
        result = await self.tools.get_user_info(user_id=222222222)

        self.assertEqual(result["id"], 222222222)
        self.assertEqual(result["name"], "TestUser")
        self.assertIn("display_name", result)
        self.assertIn("bot", result)
        self.assertFalse(result["bot"])

    async def test_get_user_info_not_found(self):
        """Test get_user_info when user not found"""
        self.mock_bot.get_user = Mock(return_value=None)

        result = await self.tools.get_user_info(user_id=999999)

        self.assertIn("error", result)
        self.assertIn("not found", result["error"])

    async def test_get_user_info_with_guild(self):
        """Test get_user_info with guild_id includes member info"""
        result = await self.tools.get_user_info(user_id=222222222, guild_id=123456789)

        self.assertEqual(result["id"], 222222222)
        self.assertIn("roles", result)
        self.assertIn("top_role", result)

    # ===== Message Tests =====

    async def test_send_message(self):
        """Test send_message sends to channel"""
        result = await self.tools.send_message(
            channel_id=987654321,
            content="Hello World!"
        )

        self.assertTrue(result.get("success"))
        self.assertIn("message_id", result)
        self.assertIn("timestamp", result)

    async def test_send_message_with_embed(self):
        """Test send_message with embed"""
        embed = {
            "title": "Test Title",
            "description": "Test Description",
            "color": 0x00ff00
        }

        result = await self.tools.send_message(
            channel_id=987654321,
            content="Check this embed",
            embed=embed
        )

        self.assertTrue(result.get("success"))

    async def test_send_message_channel_not_found(self):
        """Test send_message when channel not found"""
        self.mock_bot.get_channel = Mock(return_value=None)

        result = await self.tools.send_message(
            channel_id=999999,
            content="Hello"
        )

        self.assertIn("error", result)

    # ===== Bot Status Tests =====

    async def test_get_bot_status(self):
        """Test get_bot_status returns bot info"""
        result = await self.tools.get_bot_status()

        # The real method returns 'latency' not 'latency_ms'
        self.assertIn("latency", result)
        self.assertIn("guilds", result)
        self.assertIn("kernel_state", result)

    async def test_get_kernel_metrics(self):
        """Test get_kernel_metrics returns metrics"""
        # Set up proper mock values that can be rounded
        self.mock_kernel.metrics.avg_response_time = 0.5

        result = await self.tools.get_kernel_metrics()

        self.assertIn("total_signals", result)
        self.assertIn("user_inputs", result)
        self.assertIn("errors", result)
        self.assertIn("avg_response_time", result)

    # ===== Voice Tests =====

    async def test_get_voice_status_not_connected(self):
        """Test get_voice_status when not in voice"""
        result = await self.tools.get_voice_status(guild_id=123456789)

        self.assertFalse(result["connected"])

    async def test_get_voice_status_connected(self):
        """Test get_voice_status when connected"""
        voice_client = MockVoiceClient()
        self.mock_output_router.voice_clients[123456789] = voice_client

        result = await self.tools.get_voice_status(guild_id=123456789)

        self.assertTrue(result["connected"])
        # The method returns 'channel_id' and 'channel_name', not 'channel'
        self.assertIn("channel_id", result)
        self.assertIn("channel_name", result)

    async def test_toggle_tts(self):
        """Test toggle_tts enables/disables TTS"""
        # First, connect to voice
        voice_client = MockVoiceClient()
        self.mock_output_router.voice_clients[123456789] = voice_client

        # Enable TTS
        result = await self.tools.toggle_tts(guild_id=123456789, mode="piper")

        self.assertTrue(result.get("tts_enabled"))
        # The method returns 'tts_mode' not 'mode'
        self.assertEqual(result.get("tts_mode"), "piper")

    async def test_can_hear_user_not_in_voice(self):
        """Test can_hear_user when bot not in voice"""
        result = await self.tools.can_hear_user(guild_id=123456789, user_id=222222222)

        self.assertFalse(result["can_hear"])
        self.assertIn("reason", result)


# ===== UNIT TESTS: Template System =====

class TestDiscordTemplates(unittest.IsolatedAsyncioTestCase):
    """Tests for Discord message template system"""

    async def asyncSetUp(self):
        """Set up test fixtures"""
        self.mock_bot = MockBot()
        self.mock_kernel = MockKernel()
        self.mock_output_router = MockOutputRouter()

        self.tools = DiscordKernelTools(
            bot=self.mock_bot,
            kernel=self.mock_kernel,
            output_router=self.mock_output_router
        )

    async def test_create_message_template(self):
        """Test creating a message template"""
        result = await self.tools.create_message_template(
            template_name="welcome",
            content="Hello {username}!",
            embed={"title": "Welcome", "description": "Welcome to {server_name}"}
        )

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("template_name"), "welcome")

    async def test_get_message_template(self):
        """Test getting a message template"""
        # First create a template
        await self.tools.create_message_template(
            template_name="test_template",
            content="Test content"
        )

        result = await self.tools.get_message_template("test_template")

        self.assertTrue(result.get("success") or "content" in result)

    async def test_list_message_templates(self):
        """Test listing all templates"""
        # Create some templates
        await self.tools.create_message_template("template1", content="Test 1")
        await self.tools.create_message_template("template2", content="Test 2")

        result = await self.tools.list_message_templates()

        # list_message_templates returns a list directly, not a dict with 'templates'
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 2)

    async def test_delete_message_template(self):
        """Test deleting a template"""
        await self.tools.create_message_template("to_delete", content="Delete me")

        result = await self.tools.delete_message_template("to_delete")

        self.assertTrue(result.get("success"))

    async def test_create_welcome_template(self):
        """Test creating a welcome template"""
        result = await self.tools.create_welcome_template(
            title="Welcome {username}!",
            description="You are member #{member_count}",
            color=0x00ff00
        )

        self.assertTrue(result.get("success"))
        self.assertIn("template_name", result)

    async def test_create_announcement_template(self):
        """Test creating an announcement template"""
        result = await self.tools.create_announcement_template(
            title="Important Update",
            description="{message}",
            color=0xff0000
        )

        self.assertTrue(result.get("success"))

    async def test_create_poll_template(self):
        """Test creating a poll template"""
        result = await self.tools.create_poll_template(
            question="What's your favorite color?",
            options=["Red", "Blue", "Green"]
        )

        self.assertTrue(result.get("success"))

    async def test_get_template_help(self):
        """Test getting template help documentation"""
        result = await self.tools.get_template_help()

        # get_template_help returns {"success": True, "help": {...}}
        self.assertTrue(result.get("success"))
        self.assertIn("help", result)
        self.assertIn("variable_substitution", result["help"])
        self.assertIn("template_types", result["help"])

    async def test_get_tools_overview(self):
        """Test getting tools overview"""
        result = await self.tools.get_tools_overview()

        # get_tools_overview returns {"success": True, "overview": {...}}
        self.assertTrue(result.get("success"))
        self.assertIn("overview", result)
        self.assertIn("categories", result["overview"])
        self.assertIn("total_tools", result["overview"])

    async def test_get_template_examples(self):
        """Test getting template examples"""
        result = await self.tools.get_template_examples()

        self.assertIn("examples", result)


# ===== UNIT TESTS: Export to Agent =====

class TestExportToAgent(unittest.IsolatedAsyncioTestCase):
    """Tests for export_to_agent with categories and flags"""

    async def asyncSetUp(self):
        """Set up test fixtures"""
        self.mock_bot = MockBot()
        self.mock_kernel = MockKernel()
        self.mock_output_router = MockOutputRouter()

        self.tools = DiscordKernelTools(
            bot=self.mock_bot,
            kernel=self.mock_kernel,
            output_router=self.mock_output_router
        )

    async def test_export_to_agent_calls_add_tool(self):
        """Test export_to_agent calls agent.add_tool multiple times"""
        await self.tools.export_to_agent()

        # Verify add_tool was called multiple times
        self.assertGreater(self.mock_kernel.agent.add_tool.call_count, 50)

    async def test_export_to_agent_with_categories(self):
        """Test export_to_agent passes categories"""
        await self.tools.export_to_agent()

        # Check that add_tool was called with category parameter
        calls = self.mock_kernel.agent.add_tool.call_args_list
        categories_found = set()

        for call in calls:
            kwargs = call.kwargs if hasattr(call, 'kwargs') else (call[1] if len(call) > 1 else {})
            if 'category' in kwargs:
                for cat in kwargs['category']:
                    categories_found.add(cat)

        # Verify expected categories exist
        expected_categories = {"discord", "discord_read", "discord_write", "discord_voice", "discord_admin", "discord_moderation"}
        self.assertTrue(expected_categories.issubset(categories_found))

    async def test_export_to_agent_with_flags(self):
        """Test export_to_agent passes flags"""
        await self.tools.export_to_agent()

        calls = self.mock_kernel.agent.add_tool.call_args_list
        flags_found = []

        for call in calls:
            kwargs = call.kwargs if hasattr(call, 'kwargs') else (call[1] if len(call) > 1 else {})
            if 'flags' in kwargs:
                flags_found.append(kwargs['flags'])

        # Verify flags are present
        self.assertGreater(len(flags_found), 0)

        # Check for expected flag keys
        for flags in flags_found:
            self.assertIn("read", flags)
            self.assertIn("write", flags)
            self.assertIn("dangerous", flags)

    async def test_export_read_tools_have_correct_flags(self):
        """Test read tools have read=True, write=False"""
        await self.tools.export_to_agent()

        calls = self.mock_kernel.agent.add_tool.call_args_list

        for call in calls:
            args = call.args if hasattr(call, 'args') else call[0]
            kwargs = call.kwargs if hasattr(call, 'kwargs') else (call[1] if len(call) > 1 else {})

            if 'category' in kwargs and 'discord_read' in kwargs['category']:
                flags = kwargs.get('flags', {})
                self.assertTrue(flags.get("read", False))
                self.assertFalse(flags.get("write", True))
                self.assertFalse(flags.get("dangerous", True))

    async def test_export_moderation_tools_have_dangerous_flag(self):
        """Test moderation tools have dangerous=True"""
        await self.tools.export_to_agent()

        calls = self.mock_kernel.agent.add_tool.call_args_list

        for call in calls:
            kwargs = call.kwargs if hasattr(call, 'kwargs') else (call[1] if len(call) > 1 else {})

            if 'category' in kwargs and 'discord_moderation' in kwargs['category']:
                flags = kwargs.get('flags', {})
                self.assertTrue(flags.get("dangerous", False))


# ===== UNIT TESTS: Member Roles =====

class TestMemberRoles(unittest.IsolatedAsyncioTestCase):
    """Tests for role management"""

    async def asyncSetUp(self):
        """Set up test fixtures"""
        self.mock_bot = MockBot()
        self.mock_kernel = MockKernel()
        self.mock_output_router = MockOutputRouter()

        self.tools = DiscordKernelTools(
            bot=self.mock_bot,
            kernel=self.mock_kernel,
            output_router=self.mock_output_router
        )

    async def test_get_member_roles(self):
        """Test getting member roles"""
        result = await self.tools.get_member_roles(
            guild_id=123456789,
            user_id=222222222
        )

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("id", result[0])
        self.assertIn("name", result[0])

    async def test_get_member_roles_member_not_found(self):
        """Test get_member_roles when member not found"""
        self.mock_bot.guild._members = {}

        result = await self.tools.get_member_roles(
            guild_id=123456789,
            user_id=999999
        )

        # get_member_roles returns an empty list when member not found, not a dict with 'error'
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)


# ===== UNIT TESTS: Invite Management =====

class TestInviteManagement(unittest.IsolatedAsyncioTestCase):
    """Tests for invite management"""

    async def asyncSetUp(self):
        """Set up test fixtures"""
        self.mock_bot = MockBot()
        self.mock_kernel = MockKernel()
        self.mock_output_router = MockOutputRouter()

        self.tools = DiscordKernelTools(
            bot=self.mock_bot,
            kernel=self.mock_kernel,
            output_router=self.mock_output_router
        )

    async def test_get_invites(self):
        """Test getting server invites"""
        result = await self.tools.get_invites(guild_id=123456789)

        # get_invites returns a list directly, not a dict with 'invites'
        self.assertIsInstance(result, list)


# ===== UNIT TESTS: Output Router State =====

class TestOutputRouterState(unittest.TestCase):
    """Tests for output router state tracking"""

    def test_voice_clients_tracking(self):
        """Test voice clients are tracked per guild"""
        router = MockOutputRouter()

        voice_client = MockVoiceClient()
        router.voice_clients[123456789] = voice_client

        self.assertIn(123456789, router.voice_clients)
        self.assertTrue(router.voice_clients[123456789].is_connected())

    def test_tts_enabled_tracking(self):
        """Test TTS enabled state per guild"""
        router = MockOutputRouter()

        router.tts_enabled[123456789] = True
        router.tts_mode[123456789] = "piper"

        self.assertTrue(router.tts_enabled[123456789])
        self.assertEqual(router.tts_mode[123456789], "piper")

    def test_user_channel_mapping(self):
        """Test user to channel mapping"""
        router = MockOutputRouter()

        channel = MockTextChannel()
        router.user_channels["222222222"] = channel

        self.assertIn("222222222", router.user_channels)
        self.assertEqual(router.user_channels["222222222"].id, channel.id)


# ===== UNIT TESTS: Message Reactions =====

class TestMessageReactions(unittest.IsolatedAsyncioTestCase):
    """Tests for message reaction handling"""

    async def asyncSetUp(self):
        """Set up test fixtures"""
        self.mock_bot = MockBot()
        self.mock_kernel = MockKernel()
        self.mock_output_router = MockOutputRouter()

        self.tools = DiscordKernelTools(
            bot=self.mock_bot,
            kernel=self.mock_kernel,
            output_router=self.mock_output_router
        )

    async def test_add_reaction(self):
        """Test adding a reaction"""
        result = await self.tools.add_reaction(
            channel_id=987654321,
            message_id=123456789,
            emoji="👍"
        )

        self.assertTrue(result.get("success"))

    async def test_remove_reaction(self):
        """Test removing a reaction"""
        result = await self.tools.remove_reaction(
            channel_id=987654321,
            message_id=123456789,
            emoji="👍"
        )

        self.assertTrue(result.get("success"))

    async def test_get_message_reactions_empty(self):
        """Test getting reactions from message with no reactions"""
        result = await self.tools.get_message_reactions(
            channel_id=987654321,
            message_id=123456789
        )

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("reactions"), [])


# ===== RUN TESTS =====

if __name__ == '__main__':
    unittest.main(verbosity=2)
