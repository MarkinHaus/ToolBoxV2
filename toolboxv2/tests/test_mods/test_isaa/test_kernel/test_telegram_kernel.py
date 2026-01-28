"""
Updated Telegram Kernel Tests
Compatible with Telegram Transport Layer v1.0.0
"""

import unittest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch, mock_open

# Importiere die zu testenden Klassen
# HINWEIS: Passe den Import-Pfad an deine Struktur an!
try:
    from toolboxv2.mods.isaa.kernel.kernelin.kernelin_telegram import (
        TelegramConfig,
        TelegramOutputRouter,
        TelegramTransport,
        TelegramMediaHandler,
        MarkdownV2Escaper
    )
    import importlib.util
    TELEGRAM_AVAILABLE = importlib.util.find_spec("telegram") is not None
except ImportError:
    print("⚠️ Telegram not installed. Telegram kernel disabled.")
    TelegramConfig = None
    TelegramOutputRouter = None
    TelegramTransport = None
    TelegramMediaHandler = None
    MarkdownV2Escaper = None
    TELEGRAM_AVAILABLE = False

@unittest.skipUnless(TELEGRAM_AVAILABLE, "telegram not installed")
class TestMarkdownEscaper(unittest.TestCase):
    """Test MarkdownV2 escaping logic"""

    def test_basic_escape(self):
        text = "Hello_World*!"
        escaped = MarkdownV2Escaper.escape(text)
        # _ * und ! müssen escaped sein
        self.assertEqual(escaped, "Hello\\_World\\*\\!")

    def test_format_response_with_code(self):
        text = "Hier ist Code:\n```python\nprint('Hello')\n```\nUnd Ende."
        formatted = MarkdownV2Escaper.format_response(text)

        # Der Code-Block selbst sollte nicht die Python-Syntax escapen,
        # aber der Text davor und danach muss escaped sein.
        self.assertIn("Hier ist Code", formatted.replace('\\', ''))  # Check content roughly
        self.assertIn("```python", formatted)

    def test_inline_code(self):
        text = "Nutze `print()` funktion"
        formatted = MarkdownV2Escaper.format_response(text)
        # ` darf nicht doppelt escaped werden, text davor schon
        self.assertIn("`print()`", formatted) # Backticks werden escaped innerhalb der Logik für Code-Blöcke

@unittest.skipUnless(TELEGRAM_AVAILABLE, "telegram not installed")
class TestTelegramOutputRouter(unittest.IsolatedAsyncioTestCase):
    """Test TelegramOutputRouter with correct Config injection"""

    def setUp(self):
        self.mock_bot = MagicMock()
        # Wichtig: AsyncMock für asynchrone Methoden
        self.mock_bot.send_message = AsyncMock()
        self.mock_bot.send_photo = AsyncMock()
        self.mock_bot.send_document = AsyncMock()
        self.mock_bot.send_chat_action = AsyncMock()

        self.config = TelegramConfig(token="123:test", temp_dir="/tmp/test")
        self.router = TelegramOutputRouter(self.mock_bot, self.config)
        self.router.register_user_chat("user_1", 1001)

    async def test_send_response_simple(self):
        await self.router.send_response("user_1", "Hello World")

        self.mock_bot.send_message.assert_called_once()
        call_args = self.mock_bot.send_message.call_args
        self.assertEqual(call_args.kwargs['chat_id'], 1001)
        # Check if text is escaped (MarkdownV2 is default)
        self.assertEqual(call_args.kwargs['text'], "Hello World")

    async def test_send_notification_priority(self):
        await self.router.send_notification("user_1", "Alert", priority=9)

        call_args = self.mock_bot.send_message.call_args
        text = call_args.kwargs['text']
        # Check for Red Circle (Priority 9)
        self.assertIn("🔴", text)

    async def test_send_response_unknown_user(self):
        await self.router.send_response("unknown_user", "Test")
        self.mock_bot.send_message.assert_not_called()

    async def test_message_splitting(self):
        # Create a message longer than 4096 chars
        long_text = "a" * 5000
        await self.router._send_text(1001, long_text)

        # Should be called twice (4000 + 1000)
        self.assertEqual(self.mock_bot.send_message.call_count, 2)

@unittest.skipUnless(TELEGRAM_AVAILABLE, "telegram not installed")
class TestTelegramMediaHandler(unittest.IsolatedAsyncioTestCase):
    """Test Media Handler logic"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = TelegramConfig(token="test", temp_dir=self.temp_dir)
        self.mock_bot = MagicMock()
        self.handler = TelegramMediaHandler(self.config, self.mock_bot)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    async def test_download_photo(self):
        # Mock Telegram File Object
        mock_file = MagicMock()
        mock_file.download_to_drive = AsyncMock()
        self.mock_bot.get_file = AsyncMock(return_value=mock_file)

        path = await self.handler.download_photo("file_id_123")

        self.assertIsNotNone(path)
        self.assertTrue(path.startswith(self.temp_dir))
        self.assertTrue(path.endswith(".jpg"))
        self.mock_bot.get_file.assert_called_with("file_id_123")
        mock_file.download_to_drive.assert_called_once()

    async def test_download_document_too_large(self):
        # Mock large document
        mock_doc = MagicMock()
        mock_doc.file_size = 50 * 1024 * 1024  # 50MB

        path = await self.handler.download_document(mock_doc)
        self.assertIsNone(path)  # Should fail due to size limit (20MB default)

@unittest.skipUnless(TELEGRAM_AVAILABLE, "telegram not installed")
class TestTelegramTransport(unittest.IsolatedAsyncioTestCase):
    """Test the main Transport Layer"""

    async def asyncSetUp(self):
        self.mock_kernel = MagicMock()
        self.mock_kernel.handle_user_input = AsyncMock()
        self.config = TelegramConfig(token="123:test", admin_whitelist=[999])

        # Patch ApplicationBuilder to avoid real network calls during init
        with patch('telegram.ext.ApplicationBuilder') as MockBuilder:
            mock_app = MagicMock()
            mock_app.bot = MagicMock()
            MockBuilder.return_value.token.return_value.build.return_value = mock_app

            self.transport = TelegramTransport(self.config, self.mock_kernel)
            # Inject output router mock to verify typing status
            self.transport.output_router = MagicMock()
            self.transport.output_router.start_typing = AsyncMock()
            self.transport.output_router._cancel_typing = MagicMock()

    async def test_handle_text_authorized(self):
        # Mock Update object
        update = MagicMock()
        update.message.text = "Hello Kernel"
        update.message.from_user.id = 999  # Authorized ID
        update.message.from_user.full_name = "Admin"
        update.message.chat_id = 100

        # Call handler manually
        await self.transport._handle_text(update, None)

        # Verify Kernel interaction
        self.mock_kernel.handle_user_input.assert_called_once()
        call_args = self.mock_kernel.handle_user_input.call_args
        self.assertEqual(call_args.kwargs['content'], "Hello Kernel")
        self.assertEqual(call_args.kwargs['user_id'], "telegram_999")

    async def test_handle_text_unauthorized(self):
        update = MagicMock()
        update.message.text = "Hacker"
        update.message.from_user.id = 666  # Not in whitelist

        await self.transport._handle_text(update, None)

        self.mock_kernel.handle_user_input.assert_not_called()

    async def test_resolve_user_id_mapping(self):
        # Test Identity Mapping feature
        identity_map = {"telegram:999": "master_user"}

        with patch('telegram.ext.ApplicationBuilder'):  # Suppress build
            transport = TelegramTransport(self.config, self.mock_kernel, identity_map)
            resolved = transport._resolve_user_id(999)
            self.assertEqual(resolved, "master_user")

            resolved_default = transport._resolve_user_id(123)
            self.assertEqual(resolved_default, "telegram_123")


if __name__ == '__main__':
    unittest.main()
