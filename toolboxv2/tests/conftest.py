"""
Pytest configuration and fixtures for ToolBoxV2 test suite.

This module provides automatic mocking for external services (MinIO, LiteLLM/Ollama)
when running in CI environments where these services are not available.

Environment Variables:
    CI: Set to 'true' in GitHub Actions, enables automatic mocking
    MOCK_EXTERNAL_SERVICES: Set to 'true' to force mocking regardless of CI
    MINIO_ENDPOINT: MinIO endpoint (default: 127.0.0.1:9000)
    OLLAMA_API_BASE: Ollama/LiteLLM endpoint (default: http://localhost:11434)
"""

import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Any, Dict, Optional

import pytest

# Determine if we should mock external services
IS_CI = os.getenv("CI", "false").lower() == "true"
FORCE_MOCK = os.getenv("MOCK_EXTERNAL_SERVICES", "false").lower() == "true"
SHOULD_MOCK = IS_CI or FORCE_MOCK


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "requires_minio: mark test as requiring MinIO")
    config.addinivalue_line("markers", "requires_llm: mark test as requiring LLM API")
    config.addinivalue_line("markers", "integration: mark test as integration test")


# ============================================================================
# MinIO Mocking
# ============================================================================

class MockMinioClient:
    """Mock MinIO client for testing without actual MinIO server."""

    def __init__(self, *args, **kwargs):
        self._buckets: Dict[str, Dict[str, bytes]] = {}
        self._endpoint = kwargs.get("endpoint", "127.0.0.1:9000")

    def bucket_exists(self, bucket_name: str) -> bool:
        return bucket_name in self._buckets

    def make_bucket(self, bucket_name: str, *args, **kwargs):
        if bucket_name not in self._buckets:
            self._buckets[bucket_name] = {}

    def put_object(self, bucket_name: str, object_name: str, data, length: int, *args, **kwargs):
        if bucket_name not in self._buckets:
            self._buckets[bucket_name] = {}
        if hasattr(data, 'read'):
            self._buckets[bucket_name][object_name] = data.read()
        else:
            self._buckets[bucket_name][object_name] = data

    def get_object(self, bucket_name: str, object_name: str, *args, **kwargs):
        if bucket_name in self._buckets and object_name in self._buckets[bucket_name]:
            from io import BytesIO
            mock_response = MagicMock()
            mock_response.read.return_value = self._buckets[bucket_name][object_name]
            mock_response.data = self._buckets[bucket_name][object_name]
            return mock_response
        raise Exception(f"Object {object_name} not found in bucket {bucket_name}")

    def remove_object(self, bucket_name: str, object_name: str, *args, **kwargs):
        if bucket_name in self._buckets:
            self._buckets[bucket_name].pop(object_name, None)

    def list_objects(self, bucket_name: str, prefix: str = "", *args, **kwargs):
        if bucket_name not in self._buckets:
            return []
        objects = []
        for key in self._buckets[bucket_name]:
            if key.startswith(prefix):
                obj = MagicMock()
                obj.object_name = key
                obj.size = len(self._buckets[bucket_name][key])
                objects.append(obj)
        return objects

    def fget_object(self, bucket_name: str, object_name: str, file_path: str, *args, **kwargs):
        data = self._buckets.get(bucket_name, {}).get(object_name, b"")
        with open(file_path, "wb") as f:
            f.write(data)

    def fput_object(self, bucket_name: str, object_name: str, file_path: str, *args, **kwargs):
        with open(file_path, "rb") as f:
            self.put_object(bucket_name, object_name, f.read(), 0)


# ============================================================================
# LiteLLM/Ollama Mocking
# ============================================================================

def create_mock_llm_response(content: str = "This is a mock LLM response for testing."):
    """Create a mock LiteLLM response object."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = content
    mock_response.choices[0].message.role = "assistant"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30
    mock_response.model = "mock-model"
    mock_response.id = "mock-response-id"
    return mock_response


async def mock_acompletion(*args, **kwargs):
    """Async mock for litellm.acompletion."""
    return create_mock_llm_response()


def mock_completion(*args, **kwargs):
    """Sync mock for litellm.completion."""
    return create_mock_llm_response()


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def mock_external_services(monkeypatch):
    """Automatically mock external services in CI environment."""
    if not SHOULD_MOCK:
        yield
        return

    # Set environment variables for offline/mock mode
    monkeypatch.setenv("IS_OFFLINE_DB", "true")
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "true")

    # Mock MinIO
    try:
        monkeypatch.setattr("minio.Minio", MockMinioClient)
    except (ImportError, AttributeError):
        pass

    # Mock LiteLLM
    try:
        import litellm
        monkeypatch.setattr(litellm, "completion", mock_completion)
        monkeypatch.setattr(litellm, "acompletion", mock_acompletion)
    except (ImportError, AttributeError):
        pass

    yield

