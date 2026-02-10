"""
Unit Tests for LLM Gateway Server - FastAPI Endpoints

Uses unittest (NOT pytest)
Tests endpoints using httpx.AsyncClient with app
"""

import asyncio
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiosqlite
from fastapi.testclient import TestClient

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path setup
from server import (
    app,
    RateLimiter,
    ChatMessage,
    detect_content_requirements,
    hash_key,
)


TEST_SCHEMA = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        balance REAL DEFAULT 0.0,
        tier TEXT DEFAULT 'payg',
        active INTEGER DEFAULT 1,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS api_keys (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        key_hash TEXT UNIQUE NOT NULL,
        key_prefix TEXT NOT NULL DEFAULT '',
        name TEXT DEFAULT 'default',
        active INTEGER DEFAULT 1,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS usage (
        id INTEGER PRIMARY KEY,
        api_key_id INTEGER NOT NULL,
        model TEXT NOT NULL,
        tokens_in INTEGER DEFAULT 0,
        tokens_out INTEGER DEFAULT 0,
        cost REAL DEFAULT 0.0,
        latency_ms INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
    );
    CREATE TABLE IF NOT EXISTS signup_requests (
        id INTEGER PRIMARY KEY,
        email TEXT NOT NULL,
        tier TEXT DEFAULT 'payg',
        message TEXT,
        status TEXT DEFAULT 'pending',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
"""


class AsyncTestCase(unittest.TestCase):
    """Base class for async tests"""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def run_async(self, coro):
        return self.loop.run_until_complete(coro)


class TestRateLimiter(unittest.TestCase):
    """Test RateLimiter class"""

    def setUp(self):
        self.limiter = RateLimiter()
        self.config = {"rate_limits": {"payg": 5, "sub": 10}}

    def test_admin_always_allowed(self):
        """Admin tier has unlimited requests"""
        for _ in range(100):
            result = self.limiter.is_allowed(1, "admin", self.config)
            self.assertTrue(result)

    def test_payg_rate_limit(self):
        """PAYG tier respects rate limit"""
        user_id = 1
        # Should allow 5 requests
        for i in range(5):
            result = self.limiter.is_allowed(user_id, "payg", self.config)
            self.assertTrue(result, f"Request {i+1} should be allowed")

        # 6th request should be denied
        result = self.limiter.is_allowed(user_id, "payg", self.config)
        self.assertFalse(result)

    def test_sub_rate_limit(self):
        """Subscription tier has higher limit"""
        user_id = 2
        # Should allow 10 requests
        for i in range(10):
            result = self.limiter.is_allowed(user_id, "sub", self.config)
            self.assertTrue(result)

        # 11th request should be denied
        result = self.limiter.is_allowed(user_id, "sub", self.config)
        self.assertFalse(result)

    def test_different_users_independent(self):
        """Rate limits are independent per user"""
        user1 = 1
        user2 = 2

        # Max out user1
        for _ in range(5):
            self.limiter.is_allowed(user1, "payg", self.config)

        # User1 should be blocked
        self.assertFalse(self.limiter.is_allowed(user1, "payg", self.config))

        # User2 should still be allowed
        self.assertTrue(self.limiter.is_allowed(user2, "payg", self.config))

    def test_get_remaining(self):
        """get_remaining returns correct count"""
        user_id = 1
        remaining = self.limiter.get_remaining(user_id, "payg", self.config)
        self.assertEqual(remaining, 5)

        # Use 2 requests
        self.limiter.is_allowed(user_id, "payg", self.config)
        self.limiter.is_allowed(user_id, "payg", self.config)

        remaining = self.limiter.get_remaining(user_id, "payg", self.config)
        self.assertEqual(remaining, 3)

    def test_get_remaining_admin(self):
        """Admin tier shows 999 remaining"""
        remaining = self.limiter.get_remaining(1, "admin", self.config)
        self.assertEqual(remaining, 999)


class TestDetectContentRequirements(unittest.TestCase):
    """Test detect_content_requirements function"""

    def test_text_only_message(self):
        """Text-only message needs no special capabilities"""
        messages = [ChatMessage(role="user", content="Hello, world!")]

        needs_audio, needs_vision = detect_content_requirements(messages)

        self.assertFalse(needs_audio)
        self.assertFalse(needs_vision)

    def test_image_url_in_message(self):
        """Message with image_url needs vision"""
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            )
        ]

        needs_audio, needs_vision = detect_content_requirements(messages)

        self.assertFalse(needs_audio)
        self.assertTrue(needs_vision)

    def test_audio_url_in_message(self):
        """Message with audio_url needs audio"""
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Transcribe this"},
                    {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,xyz"}},
                ],
            )
        ]

        needs_audio, needs_vision = detect_content_requirements(messages)

        self.assertTrue(needs_audio)
        self.assertFalse(needs_vision)

    def test_image_type_in_message(self):
        """Message with image type needs vision"""
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "image", "image": "base64data"},
                ],
            )
        ]

        needs_audio, needs_vision = detect_content_requirements(messages)

        self.assertFalse(needs_audio)
        self.assertTrue(needs_vision)

    def test_audio_type_in_message(self):
        """Message with audio type needs audio"""
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "audio", "audio": "base64data"},
                ],
            )
        ]

        needs_audio, needs_vision = detect_content_requirements(messages)

        self.assertTrue(needs_audio)
        self.assertFalse(needs_vision)

    def test_input_audio_type(self):
        """Message with input_audio type needs audio"""
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "input_audio", "input_audio": "data"},
                ],
            )
        ]

        needs_audio, needs_vision = detect_content_requirements(messages)

        self.assertTrue(needs_audio)

    def test_data_uri_in_text_content(self):
        """Data URIs in text content are detected"""
        messages = [
            ChatMessage(role="user", content="Look at data:image/jpeg;base64,abc123")
        ]

        needs_audio, needs_vision = detect_content_requirements(messages)

        self.assertTrue(needs_vision)

    def test_audio_data_uri_in_text(self):
        """Audio data URIs in text are detected"""
        messages = [
            ChatMessage(role="user", content="Process data:audio/mp3;base64,xyz789")
        ]

        needs_audio, needs_vision = detect_content_requirements(messages)

        self.assertTrue(needs_audio)

    def test_audio_in_image_url_field(self):
        """Audio data in image_url field is detected as audio"""
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:audio/wav;base64,xyz"},
                    },
                ],
            )
        ]

        needs_audio, needs_vision = detect_content_requirements(messages)

        self.assertTrue(needs_audio)
        self.assertFalse(needs_vision)

    def test_multiple_messages(self):
        """Requirements detected across multiple messages"""
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(
                role="user",
                content=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}],
            ),
        ]

        needs_audio, needs_vision = detect_content_requirements(messages)

        self.assertTrue(needs_vision)

    def test_both_audio_and_vision(self):
        """Both audio and vision requirements detected"""
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,xyz"}},
                ],
            )
        ]

        needs_audio, needs_vision = detect_content_requirements(messages)

        self.assertTrue(needs_audio)
        self.assertTrue(needs_vision)


class TestHealthEndpoint(AsyncTestCase):
    """Test /health endpoint"""

    def test_health_endpoint(self):
        """Health endpoint returns status"""
        with patch("server.model_manager") as mock_manager:
            mock_manager._ollama_health = AsyncMock(return_value=True)
            mock_manager.get_active_models.return_value = [{"name": "test"}]

            client = TestClient(app)
            response = client.get("/health")

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("status", data)
            self.assertIn("timestamp", data)
            self.assertIn("active_models", data)

    def test_health_endpoint_degraded(self):
        """Health endpoint shows degraded when Ollama down"""
        with patch("server.model_manager") as mock_manager:
            mock_manager._ollama_health = AsyncMock(return_value=False)
            mock_manager.get_active_models.return_value = []

            client = TestClient(app)
            response = client.get("/health")

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "degraded")


class TestPublicEndpoints(AsyncTestCase):
    """Test public endpoints (no auth required)"""

    def test_public_models_endpoint(self):
        """Public models endpoint returns active models"""
        with patch("server.model_manager") as mock_manager:
            mock_manager.get_active_models.return_value = [
                {"name": "llama3.2", "type": "text"},
                {"name": "llava", "type": "vision"},
            ]

            client = TestClient(app)
            response = client.get("/api/models")

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("models", data)
            self.assertEqual(len(data["models"]), 2)
            self.assertEqual(data["models"][0]["id"], "llama3.2")

    def test_public_signup_endpoint(self):
        """Public signup endpoint accepts requests"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Initialize database
            async def init_test_db():
                async with aiosqlite.connect(db_path) as db:
                    await db.executescript(TEST_SCHEMA)
                    await db.commit()

            self.run_async(init_test_db())

            with patch("server.DB_PATH", db_path):
                client = TestClient(app)
                response = client.post(
                    "/api/signup",
                    json={"email": "test@example.com", "tier": "payg", "message": "Test"},
                )

                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["status"], "pending")
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_public_signup_duplicate_email(self):
        """Public signup rejects duplicate emails"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            async def init_and_add_user():
                async with aiosqlite.connect(db_path) as db:
                    await db.executescript(TEST_SCHEMA)
                    await db.execute(
                        "INSERT INTO users (email) VALUES (?)", ("test@example.com",)
                    )
                    await db.commit()

            self.run_async(init_and_add_user())

            with patch("server.DB_PATH", db_path):
                client = TestClient(app)
                response = client.post(
                    "/api/signup", json={"email": "test@example.com"}
                )

                self.assertEqual(response.status_code, 400)
                self.assertIn("already registered", response.json()["detail"])
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestAuthEndpoints(AsyncTestCase):
    """Test authentication flow"""

    def test_missing_authorization_header(self):
        """Endpoints reject requests without Authorization header"""
        client = TestClient(app)
        response = client.get("/v1/models")

        self.assertEqual(response.status_code, 401)
        self.assertIn("Missing Authorization", response.json()["detail"])

    def test_invalid_authorization_format(self):
        """Endpoints reject invalid Authorization format"""
        client = TestClient(app)
        response = client.get("/v1/models", headers={"Authorization": "InvalidFormat"})

        self.assertEqual(response.status_code, 401)
        self.assertIn("Invalid Authorization", response.json()["detail"])

    def test_invalid_api_key(self):
        """Endpoints reject invalid API keys"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            async def init_test_db():
                async with aiosqlite.connect(db_path) as db:
                    await db.executescript(TEST_SCHEMA)
                    await db.commit()

            self.run_async(init_test_db())

            with patch("server.DB_PATH", db_path):
                client = TestClient(app)
                response = client.get(
                    "/v1/models", headers={"Authorization": "Bearer sk-invalid-key"}
                )

                self.assertEqual(response.status_code, 401)
                self.assertIn("Invalid API key", response.json()["detail"])
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_valid_api_key(self):
        """Endpoints accept valid API keys"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            test_key = "sk-test-valid-key-12345678"
            key_hash = hash_key(test_key)

            async def setup_valid_user():
                async with aiosqlite.connect(db_path) as db:
                    await db.executescript(TEST_SCHEMA)
                    await db.execute(
                        "INSERT INTO users (email, tier) VALUES (?, ?)",
                        ("test@example.com", "admin"),
                    )
                    await db.execute(
                        "INSERT INTO api_keys (user_id, key_hash) VALUES (?, ?)",
                        (1, key_hash),
                    )
                    await db.commit()

            self.run_async(setup_valid_user())

            with patch("server.DB_PATH", db_path):
                with patch("server.model_manager") as mock_manager:
                    mock_manager.get_active_models.return_value = []

                    client = TestClient(app)
                    response = client.get(
                        "/v1/models", headers={"Authorization": f"Bearer {test_key}"}
                    )

                    self.assertEqual(response.status_code, 200)
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestAdminEndpoints(AsyncTestCase):
    """Test admin-only endpoints"""

    def test_admin_endpoint_requires_admin_key(self):
        """Admin endpoints require admin tier"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            test_key = "sk-test-user-key"
            key_hash = hash_key(test_key)

            async def setup_regular_user():
                async with aiosqlite.connect(db_path) as db:
                    await db.executescript(TEST_SCHEMA)
                    await db.execute(
                        "INSERT INTO users (email, tier) VALUES (?, ?)",
                        ("user@example.com", "payg"),
                    )
                    await db.execute(
                        "INSERT INTO api_keys (user_id, key_hash) VALUES (?, ?)",
                        (1, key_hash),
                    )
                    await db.commit()

            self.run_async(setup_regular_user())

            with patch("server.DB_PATH", db_path):
                client = TestClient(app)
                response = client.get(
                    "/admin/api/users", headers={"Authorization": f"Bearer {test_key}"}
                )

                self.assertEqual(response.status_code, 403)
                self.assertIn("Admin access required", response.json()["detail"])
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_admin_endpoint_with_admin_key(self):
        """Admin endpoints work with admin tier"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            admin_key = "sk-admin-key"
            key_hash = hash_key(admin_key)

            async def setup_admin_user():
                async with aiosqlite.connect(db_path) as db:
                    await db.executescript(TEST_SCHEMA)
                    await db.execute(
                        "INSERT INTO users (email, tier) VALUES (?, ?)",
                        ("admin@example.com", "admin"),
                    )
                    await db.execute(
                        "INSERT INTO api_keys (user_id, key_hash) VALUES (?, ?)",
                        (1, key_hash),
                    )
                    await db.commit()

            self.run_async(setup_admin_user())

            with patch("server.DB_PATH", db_path):
                client = TestClient(app)
                response = client.get(
                    "/admin/api/users", headers={"Authorization": f"Bearer {admin_key}"}
                )

                self.assertEqual(response.status_code, 200)
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestV1ModelsEndpoint(AsyncTestCase):
    """Test /v1/models endpoint"""

    def test_v1_models_returns_active_models(self):
        """Returns OpenAI-compatible model list"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            test_key = "sk-test-key"
            key_hash = hash_key(test_key)

            async def setup_user():
                async with aiosqlite.connect(db_path) as db:
                    await db.executescript(TEST_SCHEMA)
                    await db.execute(
                        "INSERT INTO users (email, tier) VALUES (?, ?)",
                        ("test@example.com", "admin"),
                    )
                    await db.execute(
                        "INSERT INTO api_keys (user_id, key_hash) VALUES (?, ?)",
                        (1, key_hash),
                    )
                    await db.commit()

            self.run_async(setup_user())

            with patch("server.DB_PATH", db_path):
                with patch("server.model_manager") as mock_manager:
                    mock_manager.get_active_models.return_value = [
                        {"name": "llama3.2", "type": "text"},
                    ]

                    client = TestClient(app)
                    response = client.get(
                        "/v1/models", headers={"Authorization": f"Bearer {test_key}"}
                    )

                    self.assertEqual(response.status_code, 200)
                    data = response.json()
                    self.assertEqual(data["object"], "list")
                    self.assertEqual(len(data["data"]), 1)
                    self.assertEqual(data["data"][0]["id"], "llama3.2")
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestChatCompletionsEndpoint(AsyncTestCase):
    """Test /v1/chat/completions endpoint"""

    def test_chat_completions_model_not_found(self):
        """Returns 404 when requested model not loaded"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            test_key = "sk-test-key"
            key_hash = hash_key(test_key)

            async def setup_user():
                async with aiosqlite.connect(db_path) as db:
                    await db.executescript(TEST_SCHEMA)
                    await db.execute(
                        "INSERT INTO users (email, tier, balance) VALUES (?, ?, ?)",
                        ("test@example.com", "admin", 999.0),
                    )
                    await db.execute(
                        "INSERT INTO api_keys (user_id, key_hash) VALUES (?, ?)",
                        (1, key_hash),
                    )
                    await db.commit()

            self.run_async(setup_user())

            with patch("server.DB_PATH", db_path):
                with patch("server.CONFIG_PATH", Path(db_path).parent / "config.json"):
                    with patch("server.model_manager") as mock_manager:
                        with patch("server.rate_limiter") as mock_limiter:
                            mock_limiter.is_allowed.return_value = True
                            mock_manager.find_model_for_request.return_value = None

                            client = TestClient(app)
                            response = client.post(
                                "/v1/chat/completions",
                                headers={"Authorization": f"Bearer {test_key}"},
                                json={
                                    "model": "nonexistent-model",
                                    "messages": [{"role": "user", "content": "Hello"}],
                                },
                            )

                            self.assertEqual(response.status_code, 404)
                            self.assertIn("not loaded", response.json()["detail"])
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_chat_completions_vision_requirement(self):
        """Returns 404 with helpful message when vision model needed"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            test_key = "sk-test-key"
            key_hash = hash_key(test_key)

            async def setup_user():
                async with aiosqlite.connect(db_path) as db:
                    await db.executescript(TEST_SCHEMA)
                    await db.execute(
                        "INSERT INTO users (email, tier, balance) VALUES (?, ?, ?)",
                        ("test@example.com", "admin", 999.0),
                    )
                    await db.execute(
                        "INSERT INTO api_keys (user_id, key_hash) VALUES (?, ?)",
                        (1, key_hash),
                    )
                    await db.commit()

            self.run_async(setup_user())

            with patch("server.DB_PATH", db_path):
                with patch("server.CONFIG_PATH", Path(db_path).parent / "config.json"):
                    with patch("server.model_manager") as mock_manager:
                        with patch("server.rate_limiter") as mock_limiter:
                            mock_limiter.is_allowed.return_value = True
                            mock_manager.find_model_for_request.return_value = None

                            client = TestClient(app)
                            response = client.post(
                                "/v1/chat/completions",
                                headers={"Authorization": f"Bearer {test_key}"},
                                json={
                                    "model": "gpt-4-vision",
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "text", "text": "What's this?"},
                                                {
                                                    "type": "image_url",
                                                    "image_url": {
                                                        "url": "data:image/png;base64,abc"
                                                    },
                                                },
                                            ],
                                        }
                                    ],
                                },
                            )

                            self.assertEqual(response.status_code, 404)
                            self.assertIn("vision", response.json()["detail"].lower())
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_chat_completions_rate_limit_exceeded(self):
        """Returns 429 when rate limit exceeded"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            test_key = "sk-test-key"
            key_hash = hash_key(test_key)

            async def setup_user():
                async with aiosqlite.connect(db_path) as db:
                    await db.executescript(TEST_SCHEMA)
                    await db.execute(
                        "INSERT INTO users (email, tier, balance) VALUES (?, ?, ?)",
                        ("test@example.com", "payg", 10.0),
                    )
                    await db.execute(
                        "INSERT INTO api_keys (user_id, key_hash) VALUES (?, ?)",
                        (1, key_hash),
                    )
                    await db.commit()

            self.run_async(setup_user())

            with patch("server.DB_PATH", db_path):
                with patch("server.CONFIG_PATH", Path(db_path).parent / "config.json"):
                    with patch("server.rate_limiter") as mock_limiter:
                        mock_limiter.is_allowed.return_value = False

                        client = TestClient(app)
                        response = client.post(
                            "/v1/chat/completions",
                            headers={"Authorization": f"Bearer {test_key}"},
                            json={
                                "model": "llama3.2",
                                "messages": [{"role": "user", "content": "Hello"}],
                            },
                        )

                        self.assertEqual(response.status_code, 429)
                        self.assertIn("Rate limit", response.json()["detail"])
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_chat_completions_insufficient_balance(self):
        """Returns 402 when PAYG user has no balance"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            test_key = "sk-test-key"
            key_hash = hash_key(test_key)

            async def setup_user():
                async with aiosqlite.connect(db_path) as db:
                    await db.executescript(TEST_SCHEMA)
                    await db.execute(
                        "INSERT INTO users (email, tier, balance) VALUES (?, ?, ?)",
                        ("test@example.com", "payg", 0.0),
                    )
                    await db.execute(
                        "INSERT INTO api_keys (user_id, key_hash) VALUES (?, ?)",
                        (1, key_hash),
                    )
                    await db.commit()

            self.run_async(setup_user())

            with patch("server.DB_PATH", db_path):
                with patch("server.CONFIG_PATH", Path(db_path).parent / "config.json"):
                    with patch("server.rate_limiter") as mock_limiter:
                        mock_limiter.is_allowed.return_value = True

                        client = TestClient(app)
                        response = client.post(
                            "/v1/chat/completions",
                            headers={"Authorization": f"Bearer {test_key}"},
                            json={
                                "model": "llama3.2",
                                "messages": [{"role": "user", "content": "Hello"}],
                            },
                        )

                        self.assertEqual(response.status_code, 402)
                        self.assertIn("Insufficient balance", response.json()["detail"])
        finally:
            Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
