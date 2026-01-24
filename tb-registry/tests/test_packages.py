"""Tests for package endpoints and services.

Note: Tests that require storage (list_packages, get_package, search)
are skipped when running without MinIO. These tests require integration
testing with a running MinIO instance.
"""

import pytest
from fastapi.testclient import TestClient

from registry.models.package import PackageType, Visibility


def test_create_package_unauthorized(client: TestClient) -> None:
    """Test creating package without auth.

    Args:
        client: Test client.
    """
    response = client.post(
        "/api/v1/packages",
        json={
            "name": "test-package",
            "display_name": "Test Package",
            "package_type": "mod",
        },
    )
    assert response.status_code == 401


def test_resolve_empty_requirements(client: TestClient) -> None:
    """Test resolving empty requirements.

    Args:
        client: Test client.
    """
    response = client.post(
        "/api/v1/resolve",
        json={"requirements": []},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["resolved"] == {}


def test_resolve_nonexistent_package(client: TestClient) -> None:
    """Test resolving non-existent package.

    Args:
        client: Test client.
    """
    response = client.post(
        "/api/v1/resolve",
        json={"requirements": ["nonexistent>=1.0.0"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert len(data["conflicts"]) > 0


# =================== Download Visibility Tests ===================


class TestDownloadVisibility:
    """Tests for download visibility permissions.

    Download permissions based on visibility:
    - PUBLIC: Anyone can download
    - UNLISTED: Only authenticated users can download
    - PRIVATE: Only owner can download

    Note: Integration tests with actual client require storage setup.
    Unit tests below test the service layer directly.
    """

    def test_download_unlisted_package_anonymous_denied(
        self, client: TestClient
    ) -> None:
        """Test that anonymous users cannot download unlisted packages.

        The endpoint should return 403 Forbidden for unlisted packages
        when accessed without authentication.
        """
        # This test validates the visibility check is in place
        # Actual test requires a mock package with UNLISTED visibility
        pass  # Placeholder - requires integration test setup

    def test_download_private_package_anonymous_denied(
        self, client: TestClient
    ) -> None:
        """Test that anonymous users cannot download private packages.

        The endpoint should return 403 Forbidden for private packages
        when accessed without authentication.
        """
        # This test validates the visibility check is in place
        # Actual test requires a mock package with PRIVATE visibility
        pass  # Placeholder - requires integration test setup


class TestDownloadVisibilityService:
    """Unit tests for PackageService download visibility logic."""

    @pytest.mark.asyncio
    async def test_get_download_url_public_no_auth(self) -> None:
        """Test that public packages can be downloaded without auth."""
        from unittest.mock import AsyncMock, MagicMock

        from registry.models.package import Visibility
        from registry.services.package_service import PackageService

        # Create mock service with all required arguments
        mock_repo = MagicMock()
        mock_user_repo = MagicMock()
        mock_storage = MagicMock()

        # Mock package with PUBLIC visibility
        mock_package = MagicMock()
        mock_package.visibility = Visibility.PUBLIC
        mock_package.owner_id = "owner_123"
        mock_repo.get_by_name = AsyncMock(return_value=mock_package)

        # Mock version with storage location
        mock_version = MagicMock()
        mock_version.storage_locations = [MagicMock(path="packages/test/1.0.0.zip")]
        mock_repo.get_version = AsyncMock(return_value=mock_version)

        # Mock storage URL
        mock_storage.get_download_url = AsyncMock(
            return_value="https://storage.example.com/presigned-url"
        )

        service = PackageService(mock_repo, mock_user_repo, mock_storage)

        # Should succeed without viewer_id (anonymous)
        url = await service.get_download_url("test-package", "1.0.0", viewer_id=None)
        assert url is not None
        assert "presigned-url" in url

    @pytest.mark.asyncio
    async def test_get_download_url_unlisted_requires_auth(self) -> None:
        """Test that unlisted packages require authentication."""
        from unittest.mock import AsyncMock, MagicMock

        import pytest

        from registry.exceptions import PermissionDeniedError
        from registry.models.package import Visibility
        from registry.services.package_service import PackageService

        # Create mock service with all required arguments
        mock_repo = MagicMock()
        mock_user_repo = MagicMock()
        mock_storage = MagicMock()

        # Mock package with UNLISTED visibility
        mock_package = MagicMock()
        mock_package.visibility = Visibility.UNLISTED
        mock_package.owner_id = "owner_123"
        mock_repo.get_by_name = AsyncMock(return_value=mock_package)

        service = PackageService(mock_repo, mock_user_repo, mock_storage)

        # Should fail without viewer_id (anonymous)
        with pytest.raises(PermissionDeniedError):
            await service.get_download_url("test-package", "1.0.0", viewer_id=None)

    @pytest.mark.asyncio
    async def test_get_download_url_unlisted_with_auth(self) -> None:
        """Test that authenticated users can download unlisted packages."""
        from unittest.mock import AsyncMock, MagicMock

        from registry.models.package import Visibility
        from registry.services.package_service import PackageService

        # Create mock service with all required arguments
        mock_repo = MagicMock()
        mock_user_repo = MagicMock()
        mock_storage = MagicMock()

        # Mock package with UNLISTED visibility
        mock_package = MagicMock()
        mock_package.visibility = Visibility.UNLISTED
        mock_package.owner_id = "owner_123"
        mock_repo.get_by_name = AsyncMock(return_value=mock_package)

        # Mock version with storage location
        mock_version = MagicMock()
        mock_version.storage_locations = [MagicMock(path="packages/test/1.0.0.zip")]
        mock_repo.get_version = AsyncMock(return_value=mock_version)

        # Mock storage URL
        mock_storage.get_download_url = AsyncMock(
            return_value="https://storage.example.com/presigned-url"
        )

        service = PackageService(mock_repo, mock_user_repo, mock_storage)

        # Should succeed with any authenticated user
        url = await service.get_download_url(
            "test-package", "1.0.0", viewer_id="any_user_123"
        )
        assert url is not None

    @pytest.mark.asyncio
    async def test_get_download_url_private_owner_only(self) -> None:
        """Test that private packages can only be downloaded by owner."""
        from unittest.mock import AsyncMock, MagicMock

        import pytest

        from registry.exceptions import PermissionDeniedError
        from registry.models.package import Visibility
        from registry.services.package_service import PackageService

        # Create mock service with all required arguments
        mock_repo = MagicMock()
        mock_user_repo = MagicMock()
        mock_storage = MagicMock()

        # Mock package with PRIVATE visibility
        mock_package = MagicMock()
        mock_package.visibility = Visibility.PRIVATE
        mock_package.owner_id = "owner_123"
        mock_repo.get_by_name = AsyncMock(return_value=mock_package)

        service = PackageService(mock_repo, mock_user_repo, mock_storage)

        # Should fail for non-owner
        with pytest.raises(PermissionDeniedError):
            await service.get_download_url(
                "test-package", "1.0.0", viewer_id="other_user_456"
            )

    @pytest.mark.asyncio
    async def test_get_download_url_private_owner_allowed(self) -> None:
        """Test that owner can download their private packages."""
        from unittest.mock import AsyncMock, MagicMock

        from registry.models.package import Visibility
        from registry.services.package_service import PackageService

        # Create mock service with all required arguments
        mock_repo = MagicMock()
        mock_user_repo = MagicMock()
        mock_storage = MagicMock()

        # Mock package with PRIVATE visibility
        mock_package = MagicMock()
        mock_package.visibility = Visibility.PRIVATE
        mock_package.owner_id = "owner_123"
        mock_repo.get_by_name = AsyncMock(return_value=mock_package)

        # Mock version with storage location
        mock_version = MagicMock()
        mock_version.storage_locations = [MagicMock(path="packages/test/1.0.0.zip")]
        mock_repo.get_version = AsyncMock(return_value=mock_version)

        # Mock storage URL
        mock_storage.get_download_url = AsyncMock(
            return_value="https://storage.example.com/presigned-url"
        )

        service = PackageService(mock_repo, mock_user_repo, mock_storage)

        # Should succeed for owner
        url = await service.get_download_url(
            "test-package", "1.0.0", viewer_id="owner_123"
        )
        assert url is not None
