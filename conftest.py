"""
Root conftest.py for ToolBoxV2 test suite.

Provides:
- Custom pytest markers
- pytest_plugins declaration (must live here, not in sub-conftest)
- Automatic mocking for external services (MinIO, LiteLLM/Ollama) in CI

Environment Variables:
    CI: Set to 'true' in GitHub Actions → enables automatic mocking
    MOCK_EXTERNAL_SERVICES: Set to 'true' to force mocking regardless of CI
"""

import os
from typing import Dict
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Plugin declarations — MUST be in root conftest, nowhere else
# ---------------------------------------------------------------------------

pytest_plugins = ("pytest_asyncio",)

# ---------------------------------------------------------------------------
# CI / mock detection
# ---------------------------------------------------------------------------

IS_CI = os.getenv("CI", "false").lower() == "true"
FORCE_MOCK = os.getenv("MOCK_EXTERNAL_SERVICES", "false").lower() == "true"
SHOULD_MOCK = IS_CI or FORCE_MOCK


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register all custom markers used across the test suite."""
    config.addinivalue_line("markers", "requires_minio: mark test as requiring MinIO")
    config.addinivalue_line("markers", "requires_llm: mark test as requiring LLM API")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "real_app: mark test as requiring a real App instance")


# ---------------------------------------------------------------------------
# MinIO mock
# ---------------------------------------------------------------------------

class MockMinioClient:
    """In-memory MinIO replacement — no real server needed."""

    def __init__(self, *args, **kwargs):
        self._buckets: Dict[str, Dict[str, bytes]] = {}

    def bucket_exists(self, bucket_name: str) -> bool:
        return bucket_name in self._buckets

    def make_bucket(self, bucket_name: str, *args, **kwargs):
        self._buckets.setdefault(bucket_name, {})

    def put_object(self, bucket_name: str, object_name: str, data, length: int,
                   *args, **kwargs):
        self._buckets.setdefault(bucket_name, {})
        self._buckets[bucket_name][object_name] = (
            data.read() if hasattr(data, "read") else data
        )

    def get_object(self, bucket_name: str, object_name: str, *args, **kwargs):
        if bucket_name in self._buckets and object_name in self._buckets[bucket_name]:
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
        result = []
        for key, data in self._buckets[bucket_name].items():
            if key.startswith(prefix):
                obj = MagicMock()
                obj.object_name = key
                obj.size = len(data)
                result.append(obj)
        return result

    def fget_object(self, bucket_name: str, object_name: str, file_path: str,
                    *args, **kwargs):
        data = self._buckets.get(bucket_name, {}).get(object_name, b"")
        with open(file_path, "wb") as f:
            f.write(data)

    def fput_object(self, bucket_name: str, object_name: str, file_path: str,
                    *args, **kwargs):
        with open(file_path, "rb") as f:
            self.put_object(bucket_name, object_name, f.read(), 0)


# ---------------------------------------------------------------------------
# LiteLLM / Ollama mocks
# ---------------------------------------------------------------------------

def _mock_llm_response(content: str = "Mock LLM response for testing."):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.content = content
    response.choices[0].message.role = "assistant"
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    response.model = "mock-model"
    response.id = "mock-response-id"
    return response


async def _mock_acompletion(*args, **kwargs):
    return _mock_llm_response()


def _mock_completion(*args, **kwargs):
    return _mock_llm_response()


# ---------------------------------------------------------------------------
# Fixture: auto-mock external services in CI
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_external_services(monkeypatch):
    """Automatically mock MinIO and LiteLLM when running in CI."""
    if not SHOULD_MOCK:
        yield
        return

    monkeypatch.setenv("IS_OFFLINE_DB", "true")
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "true")

    try:
        monkeypatch.setattr("minio.Minio", MockMinioClient)
    except (ImportError, AttributeError):
        pass

    try:
        import litellm
        monkeypatch.setattr(litellm, "completion", _mock_completion)
        monkeypatch.setattr(litellm, "acompletion", _mock_acompletion)
    except (ImportError, AttributeError):
        pass

    yield
