#!/usr/bin/env python3
"""
Telegram Kernel Tests
=====================

Unit tests, integration tests, and E2E tests for the Telegram Kernel.
All tests mock external dependencies (Telegram API, LLM calls, Groq).

Run with:
    pytest toolboxv2/tests/test_mods/test_telegram_kernel.py -v
"""

import unittest
import asyncio
import time
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from dataclasses import dataclass

# Mock telegram module before imports
import sys
sys.modules['telegram'] = MagicMock()
sys.modules['telegram.ext'] = MagicMock()
sys.modules['telegram.constants'] = MagicMock()


class TestUserAgentMapping(unittest.TestCase):
    """Test UserAgentMapping data structure"""

    def test_mapping_creation(self):
        """Test creating a user mapping"""
        from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import UserAgentMapping

        mapping = UserAgentMapping(
            telegram_id="123456789",
            display_name="Markin"
        )

        self.assertEqual(mapping.telegram_id, "123456789")
        self.assertEqual(mapping.display_name, "Markin")
        self.assertEqual(mapping.agent_name, "self-markin")
        self.assertIsNone(mapping.discord_id)

    def test_mapping_with_discord(self):
        """Test mapping with Discord ID"""
        from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import UserAgentMapping

        mapping = UserAgentMapping(
            telegram_id="123456789",
            discord_id="987654321",
            display_name="Markin"
        )

        self.assertEqual(mapping.discord_id, "987654321")
        self.assertEqual(mapping.agent_name, "self-markin")

    def test_agent_name_generation(self):
        """Test automatic agent name generation"""
        from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import UserAgentMapping

        # Test with spaces
        mapping = UserAgentMapping(
            telegram_id="111",
            display_name="Max Mustermann"
        )
        self.assertEqual(mapping.agent_name, "self-max_mustermann")

        # Test with uppercase
        mapping2 = UserAgentMapping(
            telegram_id="222",
            display_name="JOHN"
        )
        self.assertEqual(mapping2.agent_name, "self-john")


class TestUserMappingStore(unittest.TestCase):
    """Test UserMappingStore persistence"""

    def setUp(self):
        """Create temp directory for tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.save_path = Path(self.temp_dir) / "mappings.json"

    def tearDown(self):
        """Cleanup temp files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_register_new_user(self):
        """Test registering a new user"""
        from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import UserMappingStore

        store = UserMappingStore(self.save_path)

        mapping = store.register_user(
            telegram_id="123",
            display_name="TestUser"
        )

        self.assertEqual(mapping.telegram_id, "123")
        self.assertEqual(mapping.agent_name, "self-testuser")

        # Verify persistence
        self.assertTrue(self.save_path.exists())

    def test_get_by_telegram(self):
        """Test retrieving by Telegram ID"""
        from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import UserMappingStore

        store = UserMappingStore(self.save_path)
        store.register_user("123", "TestUser")

        mapping = store.get_by_telegram("123")
        self.assertIsNotNone(mapping)
        self.assertEqual(mapping.display_name, "TestUser")

        # Non-existent user
        self.assertIsNone(store.get_by_telegram("999"))

    def test_link_discord(self):
        """Test linking Discord account"""
        from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import UserMappingStore

        store = UserMappingStore(self.save_path)
        store.register_user("123", "TestUser")

        result = store.link_discord("123", "discord_456")
        self.assertTrue(result)

        # Verify link
        mapping = store.get_by_telegram("123")
        self.assertEqual(mapping.discord_id, "discord_456")

        # Get by Discord
        mapping_by_discord = store.get_by_discord("discord_456")
        self.assertEqual(mapping_by_discord.telegram_id, "123")

    def test_persistence_reload(self):
        """Test that data persists across store instances"""
        from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import UserMappingStore

        # Create and save
        store1 = UserMappingStore(self.save_path)
        store1.register_user("123", "TestUser", discord_id="456")

        # Load in new instance
        store2 = UserMappingStore(self.save_path)

        mapping = store2.get_by_telegram("123")
        self.assertIsNotNone(mapping)
        self.assertEqual(mapping.display_name, "TestUser")
        self.assertEqual(mapping.discord_id, "456")


class TestTelegramOutputRouter(unittest.TestCase):
    """Test TelegramOutputRouter"""

    def setUp(self):
        """Setup mock bot application"""
        self.mock_bot = MagicMock()
        self.mock_bot.bot = MagicMock()
        self.mock_bot.bot.send_message = AsyncMock(return_value=MagicMock(message_id=123))
        self.mock_bot.bot.send_photo = AsyncMock(return_value=MagicMock(message_id=124))
        self.mock_bot.bot.send_document = AsyncMock(return_value=MagicMock(message_id=125))

    def test_message_splitting(self):
        """Test long message splitting"""
        from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import TelegramOutputRouter

        router = TelegramOutputRouter(self.mock_bot)

        # Short message
        chunks = router._split_message("Hello world", max_length=100)
        self.assertEqual(len(chunks), 1)

        # Long message
        long_text = "A" * 5000
        chunks = router._split_message(long_text, max_length=1000)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 1000)

    def test_send_response(self):
        """Test sending response"""
        from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import TelegramOutputRouter

        router = TelegramOutputRouter(self.mock_bot)
        router.user_chats["user_123"] = 456

        # Run async test
        asyncio.run(router.send_response("user_123", "Hello!"))

        self.mock_bot.bot.send_message.assert_called()

    def test_send_notification_priority(self):
        """Test notification with different priorities"""
        from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import TelegramOutputRouter

        router = TelegramOutputRouter(self.mock_bot)
        router.user_chats["user_123"] = 456

        # High priority
        asyncio.run(router.send_notification("user_123", "Urgent!", priority=9))
        call_args = self.mock_bot.bot.send_message.call_args
        self.assertIn("URGENT", call_args[1]['text'])


class TestProactiveScheduler(unittest.TestCase):
    """Test ProactiveScheduler"""

    def setUp(self):
        """Setup mock kernel"""
        self.mock_kernel = MagicMock()
        self.mock_kernel.bot_app = MagicMock()
        self.mock_kernel.bot_app.job_queue = MagicMock()
        self.mock_kernel.bot_app.job_queue.get_jobs_by_name = MagicMock(return_value=[])
        self.mock_kernel.bot_app.job_queue.run_daily = MagicMock()
        self.mock_kernel.bot_app.job_queue.run_once = MagicMock()
        self.mock_kernel.output_router = MagicMock()
        self.mock_kernel.output_router.send_notification = AsyncMock()

    def test_set_user_schedule(self):
        """Test setting user schedule"""
        from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import ProactiveScheduler

        scheduler = ProactiveScheduler(self.mock_kernel)
        scheduler.set_user_schedule("user_123", morning_time="07:00", evening_time="21:00")

        self.assertEqual(scheduler.user_schedules["user_123"]["morning_time"], "07:00")
        self.assertEqual(scheduler.user_schedules["user_123"]["evening_time"], "21:00")

    def test_schedule_morning_brief(self):
        """Test scheduling morning brief"""
        from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import ProactiveScheduler

        scheduler = ProactiveScheduler(self.mock_kernel)

        asyncio.run(scheduler.schedule_morning_brief("user_123"))

        self.mock_kernel.bot_app.job_queue.run_daily.assert_called()

    def test_schedule_reminder(self):
        """Test scheduling a reminder"""
        from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import ProactiveScheduler

        scheduler = ProactiveScheduler(self.mock_kernel)

        remind_at = datetime.now() + timedelta(hours=1)
        reminder_id = asyncio.run(scheduler.schedule_reminder(
            user_id="user_123",
            content="Test reminder",
            remind_at=remind_at
        ))

        self.assertIn("reminder_", reminder_id)
        self.assertIn(reminder_id, scheduler.tasks)
        self.mock_kernel.bot_app.job_queue.run_once.assert_called()

    def test_get_user_tasks(self):
        """Test getting user tasks"""
        from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import ProactiveScheduler, ScheduledTask

        scheduler = ProactiveScheduler(self.mock_kernel)

        # Add tasks for different users
        scheduler.tasks["task1"] = ScheduledTask(
            id="task1", user_id="user_123", task_type="reminder",
            scheduled_time=datetime.now(), content="Task 1"
        )
        scheduler.tasks["task2"] = ScheduledTask(
            id="task2", user_id="user_456", task_type="reminder",
            scheduled_time=datetime.now(), content="Task 2"
        )
        scheduler.tasks["task3"] = ScheduledTask(
            id="task3", user_id="user_123", task_type="reminder",
            scheduled_time=datetime.now(), content="Task 3"
        )

        user_123_tasks = scheduler.get_user_tasks("user_123")
        self.assertEqual(len(user_123_tasks), 2)

    def test_cancel_task(self):
        """Test cancelling a task"""
        from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import ProactiveScheduler, ScheduledTask

        scheduler = ProactiveScheduler(self.mock_kernel)

        scheduler.tasks["task1"] = ScheduledTask(
            id="task1", user_id="user_123", task_type="reminder",
            scheduled_time=datetime.now(), content="Task 1"
        )

        result = asyncio.run(scheduler.cancel_task("task1"))
        self.assertTrue(result)
        self.assertNotIn("task1", scheduler.tasks)

        # Cancel non-existent
        result = asyncio.run(scheduler.cancel_task("nonexistent"))
        self.assertFalse(result)


class TestTimeParser(unittest.TestCase):
    """Test time parsing functionality"""

    def test_parse_relative_minutes(self):
        """Test parsing relative minutes"""
        # We need to test the _parse_time method
        # Import and test
        pass  # Will be implemented when we have access to the method

    def test_parse_relative_hours(self):
        """Test parsing relative hours"""
        pass

    def test_parse_absolute_time(self):
        """Test parsing HH:MM format"""
        pass


class TestTelegramKernelIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for TelegramKernel"""

    async def asyncSetUp(self):
        """Setup mocks"""
        self.mock_agent = MagicMock()
        self.mock_agent.amd = MagicMock()
        self.mock_agent.amd.name = "TestAgent"
        self.mock_agent.variable_manager = MagicMock()

        self.mock_app = MagicMock()
        self.mock_app.data_dir = tempfile.mkdtemp()

    async def asyncTearDown(self):
        """Cleanup"""
        import shutil
        shutil.rmtree(self.mock_app.data_dir, ignore_errors=True)

    @patch('toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram.Application')
    @patch('toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram.Kernel')
    async def test_kernel_initialization(self, mock_kernel_class, mock_app_class):
        """Test kernel initialization"""
        # Setup mocks
        mock_app_class.builder.return_value.token.return_value.build.return_value = MagicMock()
        mock_kernel_class.return_value = MagicMock()

        from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import TelegramKernel

        kernel = TelegramKernel(
            agent=self.mock_agent,
            app=self.mock_app,
            bot_token="test_token_123"
        )

        self.assertIsNotNone(kernel)
        self.assertEqual(kernel.instance_id, "telegram")


class TestTelegramKernelCommands(unittest.IsolatedAsyncioTestCase):
    """Test individual command handlers"""

    def _create_mock_update(self, user_id=123, chat_id=456, text="/start"):
        """Create mock Update object"""
        update = MagicMock()
        update.effective_user = MagicMock()
        update.effective_user.id = user_id
        update.effective_user.first_name = "TestUser"
        update.effective_user.username = "testuser"
        update.effective_chat = MagicMock()
        update.effective_chat.id = chat_id
        update.effective_chat.type = "private"
        update.message = MagicMock()
        update.message.text = text
        update.message.reply_text = AsyncMock()
        update.message.chat = MagicMock()
        update.message.chat.send_action = AsyncMock()
        return update

    def _create_mock_context(self, args=None):
        """Create mock Context object"""
        context = MagicMock()
        context.args = args or []
        return context

    async def test_capture_command_no_args(self):
        """Test /capture without arguments"""
        # This would test the capture command
        pass

    async def test_capture_command_with_text(self):
        """Test /capture with text"""
        pass

    async def test_focus_command(self):
        """Test /focus command"""
        pass


# ===== MOCK HELPERS =====

def create_mock_telegram_user(user_id=123, first_name="Test", username="testuser"):
    """Create mock Telegram user"""
    user = MagicMock()
    user.id = user_id
    user.first_name = first_name
    user.username = username
    return user


def create_mock_telegram_message(text="Hello", user_id=123, chat_id=456):
    """Create mock Telegram message"""
    message = MagicMock()
    message.text = text
    message.message_id = 789
    message.chat = MagicMock()
    message.chat.id = chat_id
    message.chat.type = "private"
    message.chat.send_action = AsyncMock()
    message.reply_text = AsyncMock()
    message.from_user = create_mock_telegram_user(user_id)
    return message


def create_mock_voice_message():
    """Create mock voice message"""
    voice = MagicMock()
    voice.file_id = "voice_123"
    voice.duration = 5
    voice.get_file = AsyncMock(return_value=MagicMock())
    return voice


if __name__ == '__main__':
    unittest.main(verbosity=2)
