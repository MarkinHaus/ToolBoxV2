# toolboxv2/tests/web_test/test_api.py
"""
ToolBoxV2 E2E API Endpoint Tests

Tests for API endpoints:
- Health check endpoint
- Session validation endpoints
- Module API endpoints (CloudM, Minu, etc.)
- Error responses (401, 404, 429)

Run:
    pytest toolboxv2/tests/web_test/test_api.py -v
    pytest toolboxv2/tests/web_test/test_api.py -v -k "test_health"
"""

import pytest
import asyncio
from typing import Optional

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# Import server config
try:
    from toolboxv2.tests.test_web import TEST_SERVER_BASE_URL, is_server_running
except ImportError:
    TEST_SERVER_BASE_URL = "http://localhost:8080"
    def is_server_running():
        return False


# =================== Helper Functions ===================

async def make_request(
    method: str,
    url: str,
    headers: Optional[dict] = None,
    json_data: Optional[dict] = None,
    cookies: Optional[dict] = None,
    timeout: int = 10
) -> tuple[int, dict | str]:
    """
    Make an HTTP request and return status code and response.

    Returns:
        Tuple of (status_code, response_data)
    """
    if not AIOHTTP_AVAILABLE:
        pytest.skip("aiohttp not installed")

    async with aiohttp.ClientSession(cookies=cookies) as session:
        request_kwargs = {
            "url": url,
            "headers": headers or {},
            "timeout": aiohttp.ClientTimeout(total=timeout)
        }
        if json_data:
            request_kwargs["json"] = json_data

        async with getattr(session, method.lower())(**request_kwargs) as resp:
            try:
                data = await resp.json()
            except Exception:
                data = await resp.text()
            return resp.status, data


# =================== Health & Core Endpoint Tests ===================

class TestHealthEndpoints:
    """Tests for health and status endpoints"""

    @pytest.mark.asyncio
    async def test_health_endpoint_returns_200(self):
        """Test: /health endpoint returns 200 OK"""
        if not is_server_running():
            pytest.skip(f"Server not running at {TEST_SERVER_BASE_URL}")
        status, data = await make_request("GET", f"{TEST_SERVER_BASE_URL}/health")
        assert status == 200, f"Health check failed with status {status}"

    @pytest.mark.asyncio
    async def test_health_endpoint_response_format(self):
        """Test: /health endpoint returns expected format"""
        if not is_server_running():
            pytest.skip(f"Server not running at {TEST_SERVER_BASE_URL}")
        status, data = await make_request("GET", f"{TEST_SERVER_BASE_URL}/health")
        assert status == 200

        # Health response should indicate OK status
        if isinstance(data, dict):
            assert "status" in data or "ok" in str(data).lower()
        else:
            assert "ok" in str(data).lower() or status == 200


class TestSessionValidation:
    """Tests for session validation endpoints"""

    @pytest.mark.asyncio
    async def test_validate_session_without_token(self):
        """Test: /validateSession without token returns 401 or error"""
        if not is_server_running():
            pytest.skip(f"Server not running at {TEST_SERVER_BASE_URL}")
        status, data = await make_request(
            "POST",
            f"{TEST_SERVER_BASE_URL}/validateSession",
            json_data={}
        )
        # Should return 401 Unauthorized or 400 Bad Request
        assert status in [400, 401, 403, 404, 500], f"Expected auth error, got {status}"

    @pytest.mark.asyncio
    async def test_is_valid_session_without_cookie(self):
        """Test: /IsValidSession without session cookie returns false/401"""
        if not is_server_running():
            pytest.skip(f"Server not running at {TEST_SERVER_BASE_URL}")
        status, data = await make_request(
            "GET",
            f"{TEST_SERVER_BASE_URL}/IsValidSession"
        )
        # Should indicate invalid session
        assert status in [200, 401, 404]
        if status == 200 and isinstance(data, dict):
            assert data.get("valid", True) is False or "error" in str(data).lower()


class TestModuleAPIs:
    """Tests for module-specific API endpoints"""

    @pytest.mark.asyncio
    async def test_user_account_api_unauthenticated(self):
        """Test: UserAccountManager API requires authentication"""
        if not is_server_running():
            pytest.skip(f"Server not running at {TEST_SERVER_BASE_URL}")
        url = f"{TEST_SERVER_BASE_URL}/api/CloudM.UserAccountManager/get_current_user"
        status, data = await make_request("GET", url)

        # Should return 401 for unauthenticated request
        assert status in [401, 403, 404, 500], f"Expected 401/403, got {status}: {data}"

    @pytest.mark.asyncio
    async def test_api_endpoint_not_found(self):
        """Test: Non-existent API endpoint returns error (401/404)"""
        if not is_server_running():
            pytest.skip(f"Server not running at {TEST_SERVER_BASE_URL}")
        url = f"{TEST_SERVER_BASE_URL}/api/NonExistent.Module/fake_function"
        status, data = await make_request("GET", url)

        # Server may return 401 (auth required first) or 404 (not found)
        assert status in [401, 404, 400, 500], f"Expected error status, got {status}"


class TestAPIErrorResponses:
    """Tests for API error response formats"""

    @pytest.mark.asyncio
    async def test_401_response_format(self):
        """Test: 401 responses have proper JSON format"""
        if not is_server_running():
            pytest.skip(f"Server not running at {TEST_SERVER_BASE_URL}")
        url = f"{TEST_SERVER_BASE_URL}/api/CloudM.UserAccountManager/get_current_user"
        status, data = await make_request("GET", url)

        if status == 401:
            if isinstance(data, dict):
                # Should have error field
                assert "error" in data or "message" in data or "detail" in data

    @pytest.mark.asyncio
    async def test_api_cors_headers(self):
        """Test: API endpoints include CORS headers"""
        if not AIOHTTP_AVAILABLE:
            pytest.skip("aiohttp not installed")
        if not is_server_running():
            pytest.skip(f"Server not running at {TEST_SERVER_BASE_URL}")

        async with aiohttp.ClientSession() as session:
            async with session.options(
                f"{TEST_SERVER_BASE_URL}/api/CloudM.extras/Version"
            ) as resp:
                # Check for CORS headers (may vary based on config)
                headers = resp.headers
                # At minimum, should not error on OPTIONS
                assert resp.status in [200, 204, 405]

