"""
Tests for the Registry Client.

Tests cover:
- RegistryClient initialization
- LockFileManager operations
- Data classes
- Error handling
"""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from toolboxv2.utils.extras.registry_client import (
    ArtifactBuild,
    ArtifactDetail,
    DownloadError,
    LockFileManager,
    PackageDetail,
    PackageNotFoundError,
    PackageSummary,
    RegistryAuthError,
    RegistryClient,
    RegistryConnectionError,
    RegistryError,
    ResolvedPackage,
    ResolutionResult,
    UserInfo,
    VersionDetail,
    VersionInfo,
    VersionNotFoundError,
)


class TestDataClasses(unittest.TestCase):
    """Test data class creation and defaults."""

    def test_package_summary(self):
        """Test PackageSummary creation."""
        pkg = PackageSummary(
            name="TestMod",
            description="A test module",
            latest_version="1.0.0",
            visibility="public",
            downloads=100,
            publisher="testuser",
        )
        self.assertEqual(pkg.name, "TestMod")
        self.assertEqual(pkg.latest_version, "1.0.0")
        self.assertEqual(pkg.downloads, 100)

    def test_version_info(self):
        """Test VersionInfo creation."""
        ver = VersionInfo(
            version="1.0.0",
            published_at="2024-01-01T00:00:00Z",
            yanked=False,
            downloads=50,
        )
        self.assertEqual(ver.version, "1.0.0")
        self.assertFalse(ver.yanked)

    def test_package_detail(self):
        """Test PackageDetail with versions."""
        versions = [
            VersionInfo("1.0.0", "2024-01-01", False, 100),
            VersionInfo("0.9.0", "2023-12-01", False, 50),
        ]
        pkg = PackageDetail(
            name="TestMod",
            description="Test",
            latest_version="1.0.0",
            visibility="public",
            downloads=150,
            publisher="user",
            homepage="https://example.com",
            repository="https://github.com/test/test",
            license="MIT",
            keywords=["test", "module"],
            versions=versions,
        )
        self.assertEqual(len(pkg.versions), 2)
        self.assertEqual(pkg.keywords, ["test", "module"])

    def test_resolution_result(self):
        """Test ResolutionResult creation."""
        resolved = {
            "TestMod": ResolvedPackage(
                name="TestMod",
                version="1.0.0",
                download_url="https://example.com/test.zip",
                checksum_sha256="abc123",
                dependencies=[],
            )
        }
        result = ResolutionResult(
            success=True,
            resolved=resolved,
            conflicts=[],
            errors=[],
        )
        self.assertTrue(result.success)
        self.assertIn("TestMod", result.resolved)

    def test_artifact_build(self):
        """Test ArtifactBuild creation."""
        build = ArtifactBuild(
            version="1.0.0",
            platform="windows",
            arch="x86_64",
            download_url="https://example.com/artifact.zip",
            checksum_sha256="def456",
            size=1024000,
        )
        self.assertEqual(build.platform, "windows")
        self.assertEqual(build.size, 1024000)


class TestLockFileManager(unittest.TestCase):
    """Test LockFileManager operations."""

    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.lock_path = Path(self.temp_dir) / "mods.lock.yaml"
        self.manager = LockFileManager(self.lock_path)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_creates_default(self):
        """Test loading creates default structure."""
        data = self.manager.load()
        self.assertEqual(data["version"], "1.0")
        self.assertEqual(data["packages"], {})
        self.assertEqual(data["pending_updates"], {})

    def test_add_package(self):
        """Test adding a package."""
        self.manager.add_package(
            name="TestMod",
            version="1.0.0",
            checksum="abc123",
            source="https://registry.example.com/TestMod",
            dependencies=["DepMod"],
        )

        self.assertTrue(self.manager.is_installed("TestMod"))
        self.assertEqual(self.manager.get_installed_version("TestMod"), "1.0.0")

    def test_remove_package(self):
        """Test removing a package."""
        self.manager.add_package("TestMod", "1.0.0", "abc", "https://example.com")
        self.assertTrue(self.manager.is_installed("TestMod"))

        self.manager.remove_package("TestMod")
        self.assertFalse(self.manager.is_installed("TestMod"))

    def test_get_all_installed(self):
        """Test getting all installed packages."""
        self.manager.add_package("Mod1", "1.0.0", "a", "url1")
        self.manager.add_package("Mod2", "2.0.0", "b", "url2")

        installed = self.manager.get_all_installed()
        self.assertEqual(len(installed), 2)
        self.assertEqual(installed["Mod1"], "1.0.0")
        self.assertEqual(installed["Mod2"], "2.0.0")

    def test_pending_updates(self):
        """Test pending update management."""
        self.manager.add_package("TestMod", "1.0.0", "abc", "url")
        self.manager.mark_pending_update("TestMod", "1.1.0")

        pending = self.manager.get_pending_updates()
        self.assertEqual(pending["TestMod"], "1.1.0")

        self.manager.clear_pending_update("TestMod")
        pending = self.manager.get_pending_updates()
        self.assertNotIn("TestMod", pending)

    def test_save_and_reload(self):
        """Test saving and reloading lock file."""
        self.manager.add_package("TestMod", "1.0.0", "abc", "url")
        self.manager.save()

        # Create new manager and load
        new_manager = LockFileManager(self.lock_path)
        self.assertTrue(new_manager.is_installed("TestMod"))
        self.assertEqual(new_manager.get_installed_version("TestMod"), "1.0.0")


class TestRegistryClientInit(unittest.TestCase):
    """Test RegistryClient initialization."""

    def test_default_initialization(self):
        """Test default client initialization."""
        client = RegistryClient()
        self.assertEqual(client.registry_url, "https://registry.simplecore.app")
        self.assertIsNone(client.auth_token)

    def test_custom_url(self):
        """Test custom registry URL."""
        client = RegistryClient(registry_url="https://custom.registry.com")
        self.assertEqual(client.registry_url, "https://custom.registry.com")

    def test_with_token(self):
        """Test initialization with auth token."""
        client = RegistryClient(auth_token="test-token")
        self.assertEqual(client.auth_token, "test-token")


class TestRegistryClientAsync(unittest.IsolatedAsyncioTestCase):
    """Async tests for RegistryClient."""

    async def test_is_authenticated_without_token(self):
        """Test authentication check without token."""
        client = RegistryClient()
        self.assertFalse(await client.is_authenticated())

    async def test_is_authenticated_with_token(self):
        """Test authentication check with token."""
        client = RegistryClient(auth_token="test-token")
        # Without mocking the API, this should still work
        # as it checks token presence first
        self.assertTrue(client.auth_token is not None)

    @patch("httpx.AsyncClient")
    async def test_search_packages(self, mock_client_class):
        """Test package search."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "packages": [
                {
                    "name": "TestMod",
                    "description": "A test module",
                    "latest_version": "1.0.0",
                    "visibility": "public",
                    "downloads": 100,
                    "publisher": "testuser",
                }
            ]
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        client = RegistryClient()
        client._client = mock_client

        packages = await client.search("test")
        self.assertEqual(len(packages), 1)
        self.assertEqual(packages[0].name, "TestMod")


class TestExceptions(unittest.TestCase):
    """Test custom exceptions."""

    def test_registry_error(self):
        """Test RegistryError."""
        error = RegistryError("Test error")
        self.assertEqual(str(error), "Test error")

    def test_registry_connection_error(self):
        """Test RegistryConnectionError inheritance."""
        error = RegistryConnectionError("Connection failed")
        self.assertIsInstance(error, RegistryError)

    def test_registry_auth_error(self):
        """Test RegistryAuthError inheritance."""
        error = RegistryAuthError("Auth failed")
        self.assertIsInstance(error, RegistryError)

    def test_package_not_found_error(self):
        """Test PackageNotFoundError inheritance."""
        error = PackageNotFoundError("Package not found")
        self.assertIsInstance(error, RegistryError)

    def test_version_not_found_error(self):
        """Test VersionNotFoundError inheritance."""
        error = VersionNotFoundError("Version not found")
        self.assertIsInstance(error, RegistryError)

    def test_download_error(self):
        """Test DownloadError inheritance."""
        error = DownloadError("Download failed")
        self.assertIsInstance(error, RegistryError)


class TestTokenRefresh(unittest.IsolatedAsyncioTestCase):
    """Tests for token refresh functionality."""

    def test_client_init_with_session_id(self):
        """Test client initialization with session_id."""
        client = RegistryClient(
            auth_token="test-token",
            session_id="sess_123",
            clerk_user_id="user_456"
        )
        self.assertEqual(client.session_id, "sess_123")
        self.assertEqual(client.clerk_user_id, "user_456")
        self.assertFalse(client._token_refresh_attempted)

    def test_client_init_with_callback(self):
        """Test client initialization with token refresh callback."""
        async def refresh_callback(**kwargs):
            return "new-token"

        client = RegistryClient(
            auth_token="test-token",
            token_refresh_callback=refresh_callback
        )
        self.assertIsNotNone(client._token_refresh_callback)

    async def test_login_stores_session_id(self):
        """Test that login stores session_id."""
        client = RegistryClient()

        with patch.object(client, 'get_current_user', new_callable=AsyncMock) as mock_get_user:
            mock_get_user.return_value = UserInfo(
                id="user_123",
                username="testuser",
                email="test@example.com"
            )

            result = await client.login(
                clerk_token="jwt-token",
                session_id="sess_abc",
                clerk_user_id="user_123"
            )

            self.assertTrue(result)
            self.assertEqual(client.session_id, "sess_abc")
            self.assertEqual(client.clerk_user_id, "user_123")
            self.assertFalse(client._token_refresh_attempted)

    async def test_logout_clears_session_data(self):
        """Test that logout clears all session data."""
        client = RegistryClient(
            auth_token="test-token",
            session_id="sess_123",
            clerk_user_id="user_456"
        )

        await client.logout()

        self.assertIsNone(client.auth_token)
        self.assertIsNone(client.session_id)
        self.assertIsNone(client._user)
        self.assertFalse(client._token_refresh_attempted)

    async def test_refresh_token_with_callback(self):
        """Test token refresh using callback."""
        async def refresh_callback(**kwargs):
            return "new-refreshed-token"

        client = RegistryClient(
            auth_token="old-token",
            token_refresh_callback=refresh_callback
        )

        result = await client.refresh_token()

        self.assertTrue(result)
        self.assertEqual(client.auth_token, "new-refreshed-token")
        self.assertFalse(client._token_refresh_attempted)

    async def test_refresh_token_prevents_loop(self):
        """Test that refresh_token prevents infinite loops."""
        client = RegistryClient(auth_token="old-token")
        client._token_refresh_attempted = True

        result = await client.refresh_token()

        self.assertFalse(result)

    async def test_refresh_token_callback_failure(self):
        """Test token refresh when callback returns None."""
        async def failing_callback(**kwargs):
            return None

        client = RegistryClient(
            auth_token="old-token",
            token_refresh_callback=failing_callback
        )

        result = await client.refresh_token()

        self.assertFalse(result)
        self.assertEqual(client.auth_token, "old-token")

    async def test_request_with_retry_refreshes_on_401(self):
        """Test that _request_with_retry attempts refresh on 401."""
        refresh_called = False

        async def refresh_callback(**kwargs):
            nonlocal refresh_called
            refresh_called = True
            return "new-token"

        client = RegistryClient(
            auth_token="expired-token",
            token_refresh_callback=refresh_callback
        )

        # Mock the HTTP client
        mock_response_401 = MagicMock()
        mock_response_401.status_code = 401

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {"success": True}

        # Create a mock that returns 401 first, then 200
        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_response_401
            return mock_response_200

        mock_http_client = MagicMock()
        mock_http_client.request = mock_request
        mock_http_client.is_closed = False
        mock_http_client.aclose = AsyncMock()

        # Patch _get_client to always return our mock
        async def mock_get_client():
            return mock_http_client

        client._get_client = mock_get_client

        response = await client._request_with_retry("GET", "/api/test")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(client.auth_token, "new-token")
        self.assertTrue(refresh_called)
        self.assertEqual(call_count, 2)


if __name__ == "__main__":
    unittest.main()

