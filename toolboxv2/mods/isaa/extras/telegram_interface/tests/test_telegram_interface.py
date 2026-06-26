"""Tests for TelegramInterface — unit tests without real Telegram connection."""

import asyncio
import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

from toolboxv2.mods.isaa.extras.telegram_interface.telegram_interface import (
    TelegramInterface,
    MessageContext,
    MessageSource,
    AddressBook,
    create_telegram_interface,
)
from toolboxv2.mods.isaa.extras.telegram_interface.telegram_cli_extension import (
    TelegramCliExtension,
)


def make_ctx(user_id=123, user_name="testuser", chat_id=456, chat_name="TestChat",
             content="hello", source=MessageSource.PRIVATE, mentioned_bot=True):
    return MessageContext(
        user_id=user_id, user_name=user_name, user_display_name="Test User",
        chat_id=chat_id, chat_name=chat_name, content=content, source=source,
        mentioned_bot=mentioned_bot, source_address=f"telegram://chat:{chat_id}",
    )

def make_mock_icli():
    icli = MagicMock()
    icli.isaa_tools = MagicMock()
    icli.active_session_id = "test_session"
    icli.max_iteration = 10
    icli._active_skill_context = {}
    icli._create_execution = MagicMock()
    icli._drain_agent_stream = AsyncMock()
    icli._on_agent_task_done = MagicMock()
    icli._ensure_world_model = AsyncMock()
    async def mock_monitored(*args, **kwargs):
        yield {"type": "text", "content": "Hello from agent!"}
        yield {"type": "final", "content": ""}
    icli.run_agent_monitored = mock_monitored
    return icli


class TestMessageContext(unittest.TestCase):
    def test_creation(self):
        ctx = make_ctx()
        self.assertEqual(ctx.user_id, 123)
        self.assertEqual(ctx.source, MessageSource.PRIVATE)

    def test_to_agent_context(self):
        ctx = make_ctx(content="Hallo Welt")
        result = ctx.to_agent_context()
        self.assertIn("[Telegram]", result)
        self.assertIn("Test User", result)

    def test_to_dict(self):
        ctx = make_ctx()
        d = ctx.to_dict()
        self.assertEqual(d["user_id"], 123)
        self.assertEqual(d["source"], "private")

    def test_message_source_enum(self):
        self.assertEqual(MessageSource.PRIVATE.value, "private")
        self.assertEqual(MessageSource.GROUP.value, "group")


class TestAddressBook(unittest.TestCase):
    def test_register_user(self):
        ab = AddressBook()
        ab.register_user(123, "testuser", "Test User")
        self.assertIn(123, ab._contacts)

    def test_register_chat(self):
        ab = AddressBook()
        ab.register_chat(456, "TestChat", "group")
        self.assertIn(456, ab._chats)

    def test_update_active_conversation(self):
        ab = AddressBook()
        ctx = make_ctx()
        ab.update_active_conversation(ctx)
        self.assertEqual(len(ab.get_active_conversations()), 1)

    def test_search_user(self):
        ab = AddressBook()
        ab.register_user(123, "markin", "Markin Hausmanns")
        self.assertEqual(len(ab.search("markin")), 1)

    def test_search_chat(self):
        ab = AddressBook()
        ab.register_chat(456, "DevChat", "group")
        self.assertEqual(len(ab.search("dev")), 1)

    def test_search_no_results(self):
        ab = AddressBook()
        self.assertEqual(len(ab.search("nonexistent")), 0)


class TestTelegramInterface(unittest.TestCase):
    def setUp(self):
        self.icli = make_mock_icli()
        self.interface = TelegramInterface(icli_host=self.icli, token="test-token-123", admin_ids=[999])

    def test_init(self):
        self.assertEqual(self.interface.admin_ids, [999])
        self.assertEqual(self.interface.token, "test-token-123")

    def test_should_respond_private(self):
        self.assertTrue(self.interface._should_respond(make_ctx(source=MessageSource.PRIVATE)))

    def test_should_respond_group_mentioned(self):
        self.assertTrue(self.interface._should_respond(make_ctx(source=MessageSource.GROUP, mentioned_bot=True)))

    def test_should_not_respond_group_unmentioned(self):
        self.assertFalse(self.interface._should_respond(make_ctx(source=MessageSource.GROUP, mentioned_bot=False)))

    def test_should_respond_group_all_allowed(self):
        self.interface.respond_to_groups_only_when_mentioned = False
        self.assertTrue(self.interface._should_respond(make_ctx(source=MessageSource.GROUP, mentioned_bot=False)))

    def test_isolated_session_id(self):
        ctx = make_ctx(user_id=123, chat_id=456)
        self.assertEqual(self.interface._isolated_session_id(ctx), "tg_456_123")

    def test_resolve_route_moderator_default(self):
        ctx = make_ctx(user_id=111)
        agent_name, session_id = self.interface._resolve_route(ctx)
        self.assertEqual(agent_name, "moderator")

    def test_resolve_route_admin_default_session(self):
        self.interface.user_prefs[999] = {"session": "default"}
        ctx = make_ctx(user_id=999)
        _, session_id = self.interface._resolve_route(ctx)
        self.assertEqual(session_id, "default")

    def test_resolve_route_admin_self_agent(self):
        self.interface.user_prefs[999] = {"agent": "self"}
        ctx = make_ctx(user_id=999)
        agent_name, _ = self.interface._resolve_route(ctx)
        self.assertEqual(agent_name, "self")

    def test_resolve_route_non_admin_cannot_use_self(self):
        self.interface.user_prefs[111] = {"agent": "self"}
        ctx = make_ctx(user_id=111)
        agent_name, _ = self.interface._resolve_route(ctx)
        self.assertEqual(agent_name, "moderator")


class TestAdminCommands(unittest.TestCase):
    def setUp(self):
        self.icli = make_mock_icli()
        self.interface = TelegramInterface(icli_host=self.icli, token="test-token", admin_ids=[999])

    def test_agent_switch_to_self(self):
        ctx = make_ctx(user_id=999, content="/agent self")
        self.assertIn("self", self.interface._handle_admin_command(ctx).lower())

    def test_agent_switch_to_moderator(self):
        ctx = make_ctx(user_id=999, content="/agent moderator")
        self.assertIn("moderator", self.interface._handle_admin_command(ctx).lower())

    def test_agent_non_admin_cannot_switch(self):
        ctx = make_ctx(user_id=111, content="/agent self")
        self.assertIn("Invalid", self.interface._handle_admin_command(ctx))

    def test_session_switch(self):
        ctx = make_ctx(user_id=999, content="/session default")
        self.assertIn("default", self.interface._handle_admin_command(ctx).lower())

    def test_whoami_admin(self):
        ctx = make_ctx(user_id=999, content="/whoami")
        result = self.interface._handle_admin_command(ctx)
        self.assertIn("999", result)
        self.assertIn("Admin: yes", result)

    def test_whoami_non_admin(self):
        ctx = make_ctx(user_id=111, content="/whoami")
        self.assertIn("Admin: no", self.interface._handle_admin_command(ctx))

    def test_unknown_command_returns_none(self):
        ctx = make_ctx(user_id=999, content="/unknown_cmd")
        self.assertIsNone(self.interface._handle_admin_command(ctx))


class TestFactoryFunction(unittest.TestCase):
    def test_factory_with_token(self):
        icli = make_mock_icli()
        interface = create_telegram_interface(icli_host=icli, token="my-token", admin_ids=[123])
        self.assertIsInstance(interface, TelegramInterface)
        self.assertEqual(interface.token, "my-token")

    def test_factory_no_token_raises(self):
        icli = make_mock_icli()
        with self.assertRaises(ValueError):
            create_telegram_interface(icli_host=icli, token=None)

    def test_factory_env_token(self):
        icli = make_mock_icli()
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "env-token"}):
            interface = create_telegram_interface(icli_host=icli)
            self.assertEqual(interface.token, "env-token")

    def test_factory_env_admin_ids(self):
        icli = make_mock_icli()
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "token", "TELEGRAM_ADMIN_IDS": "111,222,333"}):
            interface = create_telegram_interface(icli_host=icli)
            self.assertEqual(interface.admin_ids, [111, 222, 333])


class TestMessageSplitting(unittest.TestCase):
    def test_short_message_no_split(self):
        chunks = ["Hello World"[i:i+4096] for i in range(0, len("Hello World"), 4096)]
        self.assertEqual(len(chunks), 1)

    def test_long_message_splits(self):
        text = "A" * 5000
        chunks = [text[i:i+4096] for i in range(0, len(text), 4096)]
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 4096)
        self.assertEqual(len(chunks[1]), 904)

    def test_exact_boundary(self):
        text = "A" * 4096
        chunks = [text[i:i+4096] for i in range(0, len(text), 4096)]
        self.assertEqual(len(chunks), 1)


class TestModeratorSafeMode(unittest.TestCase):
    def test_default_safe_tools(self):
        self.assertEqual(TelegramInterface.MODERATOR_SAFE_TOOLS, {"search_web"})

    def test_custom_safelist(self):
        icli = make_mock_icli()
        interface = TelegramInterface(icli_host=icli, token="test", moderator_safelist={"search_web", "memory_recall"})
        self.assertEqual(interface.moderator_safelist, {"search_web", "memory_recall"})


class TestCliExtension(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.icli = make_mock_icli()
        self.ext = TelegramCliExtension(self.icli)

    async def test_help_no_args(self):
        result = await self.ext.handle_command("")
        self.assertIn("connect", result)

    async def test_status_not_connected(self):
        result = await self.ext.handle_command("status")
        self.assertIn("Not connected", result)

    async def test_disconnect_not_connected(self):
        result = await self.ext.handle_command("disconnect")
        self.assertIn("Not connected", result)

    async def test_unknown_command(self):
        result = await self.ext.handle_command("foobar")
        self.assertIn("Unknown", result)

    async def test_send_missing_args(self):
        result = await self.ext.handle_command("send 123")
        self.assertIn("Usage", result)

    async def test_search_missing_args(self):
        result = await self.ext.handle_command("search")
        self.assertIn("Usage", result)

    async def test_admin_missing_args(self):
        result = await self.ext.handle_command("admin add")
        self.assertIn("Usage", result)


if __name__ == "__main__":
    unittest.main()
