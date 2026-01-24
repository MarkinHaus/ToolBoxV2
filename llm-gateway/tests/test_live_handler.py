"""
Unit Tests für LLM Gateway Live Voice API

Verwendet unittest (kein pytest!)
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ..live_handler import (
    LiveHandler,
    LiveSession,
    LiveSessionRequest,
    AudioConfig,
    WakeWordConfig,
    WakeWordMode,
    VoiceConfig,
    LLMConfig,
    ConversationTurn,
    create_live_handler,
)


class TestWakeWordProcessing(unittest.TestCase):
    """Test wake word detection and processing"""

    def setUp(self):
        self.handler = LiveHandler(":memory:", MagicMock())

    def test_wake_word_disabled(self):
        """Without wake word enabled, text passes through"""
        config = WakeWordConfig(enabled=False, words=[], mode=WakeWordMode.PRE)
        result = self.handler._process_wake_word("hello world", config)
        self.assertEqual(result, "hello world")

    def test_wake_word_pre_mode_found_at_start(self):
        """PRE mode: wake word at start returns text after it"""
        config = WakeWordConfig(enabled=True, words=["hey claude"], mode=WakeWordMode.PRE)
        result = self.handler._process_wake_word("hey claude what time is it", config)
        self.assertEqual(result, "what time is it")

    def test_wake_word_pre_mode_not_at_start(self):
        """PRE mode: returns text after wake word"""
        config = WakeWordConfig(enabled=True, words=["hey claude"], mode=WakeWordMode.PRE)
        result = self.handler._process_wake_word("I said hey claude what time is it", config)
        self.assertEqual(result, "what time is it")

    def test_wake_word_post_mode(self):
        """POST mode: returns text before wake word"""
        config = WakeWordConfig(enabled=True, words=["over"], mode=WakeWordMode.POST)
        result = self.handler._process_wake_word("please help me over", config)
        self.assertEqual(result, "please help me")

    def test_wake_word_mid_mode(self):
        """MID mode: wake word anywhere, removed from text"""
        config = WakeWordConfig(enabled=True, words=["okay"], mode=WakeWordMode.MID)
        result = self.handler._process_wake_word("so okay what do you think", config)
        self.assertEqual(result, "so what do you think")

    def test_wake_word_not_found(self):
        """Returns None when wake word not found"""
        config = WakeWordConfig(enabled=True, words=["hey claude"], mode=WakeWordMode.PRE)
        result = self.handler._process_wake_word("hello world", config)
        self.assertIsNone(result)

    def test_wake_word_case_insensitive(self):
        """Wake word matching is case insensitive"""
        config = WakeWordConfig(enabled=True, words=["hey claude"], mode=WakeWordMode.PRE)
        result = self.handler._process_wake_word("HEY CLAUDE hello", config)
        self.assertEqual(result, "hello")

    def test_multiple_wake_words(self):
        """First matching wake word is used"""
        config = WakeWordConfig(
            enabled=True,
            words=["hey claude", "ok computer"],
            mode=WakeWordMode.PRE
        )
        result = self.handler._process_wake_word("ok computer help me", config)
        self.assertEqual(result, "help me")


class TestSentenceExtraction(unittest.TestCase):
    """Test sentence splitting for TTS streaming"""

    def setUp(self):
        self.handler = LiveHandler(":memory:", MagicMock())

    def test_single_sentence(self):
        """Single sentence returns as-is"""
        result = self.handler._extract_sentences("Hello world")
        self.assertEqual(result, ["Hello world"])

    def test_multiple_sentences(self):
        """Multiple sentences are split"""
        result = self.handler._extract_sentences("Hello. How are you? I am fine!")
        self.assertEqual(len(result), 3)

    def test_sentence_with_period(self):
        """Sentences split on period"""
        result = self.handler._extract_sentences("First sentence. Second sentence.")
        self.assertEqual(len(result), 2)


class TestLiveSession(unittest.TestCase):
    """Test LiveSession state management"""

    def test_session_creation(self):
        """Session creates with correct defaults"""
        session = LiveSession(
            token="test-123",
            user_id=1,
            audio_config=AudioConfig(),
            wake_word_config=WakeWordConfig(),
            voice_config=VoiceConfig(),
            llm_config=LLMConfig(model="test-model")
        )

        self.assertEqual(session.token, "test-123")
        self.assertEqual(session.user_id, 1)
        self.assertEqual(len(session.history), 0)
        self.assertFalse(session.is_generating)

    def test_add_user_turn(self):
        """Adding user turn updates history"""
        session = LiveSession(
            token="test-123",
            user_id=1,
            audio_config=AudioConfig(),
            wake_word_config=WakeWordConfig(),
            voice_config=VoiceConfig(),
            llm_config=LLMConfig(model="test-model")
        )

        session.add_user_turn("Hello")

        self.assertEqual(len(session.history), 1)
        self.assertEqual(session.history[0].role, "user")
        self.assertEqual(session.history[0].content, "Hello")

    def test_add_assistant_turn_interrupted(self):
        """Interrupted assistant turn marked correctly"""
        session = LiveSession(
            token="test-123",
            user_id=1,
            audio_config=AudioConfig(),
            wake_word_config=WakeWordConfig(),
            voice_config=VoiceConfig(),
            llm_config=LLMConfig(model="test-model")
        )

        session.add_assistant_turn("Hello, I was saying...", interrupted=True)

        self.assertEqual(len(session.history), 1)
        self.assertTrue(session.history[0].interrupted)

    def test_get_messages_for_llm(self):
        """LLM messages built correctly"""
        session = LiveSession(
            token="test-123",
            user_id=1,
            audio_config=AudioConfig(),
            wake_word_config=WakeWordConfig(),
            voice_config=VoiceConfig(),
            llm_config=LLMConfig(
                model="test-model",
                system_prompt="You are helpful.",
                history_length=10
            )
        )

        session.add_user_turn("Hello")
        session.add_assistant_turn("Hi there!")
        session.add_user_turn("How are you?")

        messages = session.get_messages_for_llm()

        # System + 3 conversation turns
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[3]["role"], "user")

    def test_history_length_limit(self):
        """History respects length limit"""
        session = LiveSession(
            token="test-123",
            user_id=1,
            audio_config=AudioConfig(),
            wake_word_config=WakeWordConfig(),
            voice_config=VoiceConfig(),
            llm_config=LLMConfig(
                model="test-model",
                system_prompt="",
                history_length=2
            )
        )

        session.add_user_turn("Message 1")
        session.add_assistant_turn("Response 1")
        session.add_user_turn("Message 2")
        session.add_assistant_turn("Response 2")
        session.add_user_turn("Message 3")

        messages = session.get_messages_for_llm()

        # Only last 2 turns (history_length=2)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["content"], "Response 2")
        self.assertEqual(messages[1]["content"], "Message 3")

    def test_interrupted_marker_in_messages(self):
        """Interrupted turns have marker in messages"""
        session = LiveSession(
            token="test-123",
            user_id=1,
            audio_config=AudioConfig(),
            wake_word_config=WakeWordConfig(),
            voice_config=VoiceConfig(),
            llm_config=LLMConfig(model="test-model", system_prompt="")
        )

        session.add_assistant_turn("I was saying...", interrupted=True)

        messages = session.get_messages_for_llm()

        self.assertIn("[INTERRUPTED BY USER]", messages[0]["content"])

    def test_session_serialization(self):
        """Session can be serialized and deserialized"""
        session = LiveSession(
            token="test-123",
            user_id=1,
            audio_config=AudioConfig(allow_interrupt=False),
            wake_word_config=WakeWordConfig(enabled=True, words=["hey"]),
            voice_config=VoiceConfig(speed=1.5),
            llm_config=LLMConfig(model="test-model")
        )

        session.add_user_turn("Hello")

        # Serialize
        data = session.to_dict()

        # Deserialize
        restored = LiveSession.from_dict(data)

        self.assertEqual(restored.token, "test-123")
        self.assertEqual(restored.user_id, 1)
        self.assertFalse(restored.audio_config.allow_interrupt)
        self.assertTrue(restored.wake_word_config.enabled)
        self.assertEqual(restored.voice_config.speed, 1.5)
        self.assertEqual(len(restored.history), 1)


class TestAudioConfig(unittest.TestCase):
    """Test audio configuration"""

    def test_default_values(self):
        """Default audio config values"""
        config = AudioConfig()

        self.assertEqual(config.input_format.value, "webm")
        self.assertEqual(config.output_format.value, "opus")
        self.assertEqual(config.sample_rate, 24000)
        self.assertTrue(config.allow_interrupt)


class TestLLMConfig(unittest.TestCase):
    """Test LLM configuration"""

    def test_default_values(self):
        """Default LLM config values"""
        config = LLMConfig(model="test-model")

        self.assertEqual(config.model, "test-model")
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_tokens, 1024)
        self.assertEqual(config.history_length, 20)


class AsyncTestCase(unittest.TestCase):
    """Base class for async tests"""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def run_async(self, coro):
        return self.loop.run_until_complete(coro)


class TestLiveHandlerDatabase(AsyncTestCase):
    """Test LiveHandler database operations"""

    def test_init_db(self):
        """Database initialization creates tables"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            handler = LiveHandler(db_path, MagicMock())
            self.run_async(handler.init_db())

            # Verify table exists by trying to query it
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='live_sessions'"
            )
            self.assertIsNotNone(cursor.fetchone())
            conn.close()
        finally:
            os.unlink(db_path)

    def test_create_session_requires_models(self):
        """Session creation fails without required models"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            # Mock model manager without TTS
            model_manager = MagicMock()
            model_manager.find_tts_slot.return_value = None

            handler = LiveHandler(db_path, model_manager)
            self.run_async(handler.init_db())

            request = LiveSessionRequest(
                llm_config=LLMConfig(model="test-model")
            )

            with self.assertRaises(Exception) as context:
                self.run_async(handler.create_session(request, user_id=1))

            self.assertIn("TTS", str(context.exception))
        finally:
            os.unlink(db_path)


class TestConversationTurn(unittest.TestCase):
    """Test ConversationTurn dataclass"""

    def test_creation(self):
        """Turn creates with correct values"""
        turn = ConversationTurn(role="user", content="Hello")

        self.assertEqual(turn.role, "user")
        self.assertEqual(turn.content, "Hello")
        self.assertFalse(turn.interrupted)
        self.assertIsInstance(turn.timestamp, float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
