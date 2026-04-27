"""
Tests for DiffUploader using TDD approach.

Test-Driven Development: Write tests first, then implement.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import AsyncMock, MagicMock

from toolboxv2.utils.extras.install.diff_uploader import DiffUploader
from toolboxv2.utils.extras.install.upload_cache import UploadCache
from toolboxv2.utils.extras.registry_client import RegistryClient


@pytest.fixture
def temp_cache():
    """Create a temporary cache directory."""
    temp_dir = Path(tempfile.mkdtemp())
    cache = UploadCache(temp_dir)
    yield cache
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_zip(tmp_path):
    """Create a sample ZIP file."""
    zip_path = tmp_path / "test.zip"
    import zipfile
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("test.txt", "Hello, World!")
        zf.writestr("data.json", '{"version": "1.0.0"}')
    return zip_path


@pytest.fixture
def mock_registry_client():
    """Create a mock RegistryClient."""
    client = MagicMock(spec=RegistryClient)
    client.auth_token = "test_token"
    client.upload_version = AsyncMock(return_value=True)
    return client


class TestDiffUploader:
    """Test suite for DiffUploader."""

    @pytest.mark.asyncio
    async def test_upload_without_previous_version(self, mock_registry_client, temp_cache, sample_zip):
        """
        Test: Upload when no previous version exists.

        GIVEN: DiffUploader with empty cache
        WHEN: upload_with_diff() is called
        THEN: Should upload as full (not diff)
        """
        # Arrange
        uploader = DiffUploader(mock_registry_client, temp_cache)

        # Act
        result = await uploader.upload_with_diff("test-mod", "1.0.0", sample_zip)

        # Assert
        assert result.success is True
        assert result.upload_type == "full"
        assert result.uploaded_bytes > 0
        mock_registry_client.upload_version.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_with_small_diff_uses_diff(self, mock_registry_client, temp_cache, sample_zip):
        """
        Test: Upload with small diff uses diff upload.

        GIVEN: DiffUploader with v1.0.0 cached and small change in v1.0.1
        WHEN: upload_with_diff() is called
        THEN: Should use diff upload
        """
        # Arrange
        # Create v1.0.0
        v1_zip = temp_cache.cache_dir / "v1.zip"
        shutil.copy(sample_zip, v1_zip)
        temp_cache.store_uploaded_package("test-mod", "1.0.0", v1_zip)

        # Create v1.0.1 with small change
        v2_zip = temp_cache.cache_dir / "v2.zip"
        shutil.copy(sample_zip, v2_zip)
        # Modify slightly (simulate small change)
        import zipfile
        with zipfile.ZipFile(v2_zip, 'a') as zf:
            zf.writestr("new_file.txt", "small addition")

        uploader = DiffUploader(mock_registry_client, temp_cache)

        # Act
        result = await uploader.upload_with_diff("test-mod", "1.0.1", v2_zip, max_diff_ratio=0.5)

        # Assert
        assert result.success is True
        assert result.upload_type == "diff"

    @pytest.mark.asyncio
    async def test_upload_with_large_diff_uses_full(self, mock_registry_client, temp_cache, sample_zip):
        """
        Test: Upload with large diff falls back to full upload.

        GIVEN: DiffUploader with v1.0.0 cached and large change in v2.0.0
        WHEN: upload_with_diff() is called with low threshold
        THEN: Should use full upload
        """
        # Arrange
        # Create v1.0.0 (small file)
        v1_zip = temp_cache.cache_dir / "v1.zip"
        shutil.copy(sample_zip, v1_zip)
        temp_cache.store_uploaded_package("test-mod", "1.0.0", v1_zip)

        # Create v2.0.0 (very different - new large content)
        v2_zip = temp_cache.cache_dir / "v2.zip"
        import zipfile
        with zipfile.ZipFile(v2_zip, 'w') as zf:
            # Add lots of different content
            for i in range(100):
                zf.writestr(f"file_{i}.txt", f"Large content {i}" * 100)

        uploader = DiffUploader(mock_registry_client, temp_cache)

        # Act - Set very low threshold to force full upload
        result = await uploader.upload_with_diff(
            "test-mod", "2.0.0", v2_zip, max_diff_ratio=0.01
        )

        # Assert
        assert result.success is True
        assert result.upload_type == "full"

    @pytest.mark.asyncio
    async def test_upload_stores_in_cache_after_success(self, mock_registry_client, temp_cache, sample_zip):
        """
        Test: Successful upload stores package in cache.

        GIVEN: DiffUploader
        WHEN: upload_with_diff() succeeds
        THEN: Package should be stored in cache
        """
        # Arrange
        uploader = DiffUploader(mock_registry_client, temp_cache)

        # Act
        await uploader.upload_with_diff("test-mod", "1.0.0", sample_zip)

        # Assert
        assert temp_cache.has_package("test-mod", "1.0.0") is True

    @pytest.mark.asyncio
    async def test_upload_with_changelog(self, mock_registry_client, temp_cache, sample_zip):
        """
        Test: Upload includes changelog.

        GIVEN: DiffUploader
        WHEN: upload_with_diff() is called with changelog
        THEN: Changelog should be passed to upload_version
        """
        # Arrange
        uploader = DiffUploader(mock_registry_client, temp_cache)
        changelog = "Fixed bug in feature X"

        # Act
        await uploader.upload_with_diff("test-mod", "1.0.0", sample_zip, changelog=changelog)

        # Assert
        call_args = mock_registry_client.upload_version.call_args
        assert call_args[1]["changelog"] == changelog

    @pytest.mark.asyncio
    async def test_upload_failure_propagates(self, mock_registry_client, temp_cache, sample_zip):
        """
        Test: Upload failure is propagated.

        GIVEN: DiffUploader with failing upload_version
        WHEN: upload_with_diff() is called
        THEN: Should return UploadResult with success=False
        """
        # Arrange
        mock_registry_client.upload_version = AsyncMock(side_effect=Exception("Upload failed"))
        uploader = DiffUploader(mock_registry_client, temp_cache)

        # Act
        result = await uploader.upload_with_diff("test-mod", "1.0.0", sample_zip)

        # Assert
        assert result.success is False
        assert "Upload failed" in result.error_message
