"""
Tests for UploadCache using TDD approach.

Test-Driven Development: Write tests first, then implement.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from toolboxv2.utils.extras.install.upload_cache import UploadCache


@pytest.fixture
def temp_cache():
    """Create a temporary cache directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    cache = UploadCache(temp_dir)
    yield cache
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_zip(tmp_path):
    """Create a sample ZIP file for testing."""
    zip_path = tmp_path / "test.zip"
    # Create a minimal ZIP file
    import zipfile
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("test.txt", "Hello, World!")
    return zip_path


class TestUploadCache:
    """Test suite for UploadCache."""

    def test_store_and_retrieve_package(self, temp_cache, sample_zip):
        """
        Test: Store a package and retrieve it.

        GIVEN: An UploadCache instance
        WHEN: A package is stored with store_uploaded_package()
        THEN: The package can be retrieved with get_uploaded_package()
        """
        # Arrange
        name = "test-mod"
        version = "1.0.0"

        # Act
        temp_cache.store_uploaded_package(name, version, sample_zip)
        result = temp_cache.get_uploaded_package(name, version)

        # Assert
        assert result is not None, "Package should be retrieved"
        assert result.exists(), "Package file should exist"
        assert result.stat().st_size > 0, "Package should have content"

    def test_store_nonexistent_file(self, temp_cache):
        """
        Test: Storing a nonexistent file raises an error.

        GIVEN: An UploadCache instance
        WHEN: A nonexistent file is stored
        THEN: Should raise FileNotFoundError
        """
        # Arrange
        nonexistent = Path("nonexistent.zip")

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            temp_cache.store_uploaded_package("test", "1.0.0", nonexistent)

    def test_has_package_exists(self, temp_cache, sample_zip):
        """
        Test: Check if package exists in cache.

        GIVEN: An UploadCache with a stored package
        WHEN: has_package() is called with existing package
        THEN: Should return True
        """
        # Arrange
        temp_cache.store_uploaded_package("test-mod", "1.0.0", sample_zip)

        # Act & Assert
        assert temp_cache.has_package("test-mod", "1.0.0") is True

    def test_has_package_not_exists(self, temp_cache):
        """
        Test: Check if package exists when it doesn't.

        GIVEN: An empty UploadCache
        WHEN: has_package() is called with non-existing package
        THEN: Should return False
        """
        # Act & Assert
        assert temp_cache.has_package("test-mod", "1.0.0") is False

    def test_list_versions_empty(self, temp_cache):
        """
        Test: List versions when no versions stored.

        GIVEN: An empty UploadCache
        WHEN: list_versions() is called
        THEN: Should return empty list
        """
        # Act
        versions = temp_cache.list_versions("test-mod")

        # Assert
        assert versions == []

    def test_list_versions_multiple(self, temp_cache, sample_zip):
        """
        Test: List all versions of a package.

        GIVEN: An UploadCache with multiple versions of a package
        WHEN: list_versions() is called
        THEN: Should return all versions in sorted order
        """
        # Arrange
        versions = ["1.0.0", "1.1.0", "2.0.0"]
        for v in versions:
            # Create a copy of sample_zip for each version
            import shutil
            zip_copy = temp_cache.cache_dir / f"temp_{v}.zip"
            shutil.copy(sample_zip, zip_copy)
            temp_cache.store_uploaded_package("test-mod", v, zip_copy)

        # Act
        result = temp_cache.list_versions("test-mod")

        # Assert
        assert len(result) == 3
        assert set(result) == set(versions)

    def test_list_different_packages(self, temp_cache, sample_zip):
        """
        Test: Versions are isolated per package.

        GIVEN: An UploadCache with multiple packages
        WHEN: list_versions() is called for different packages
        THEN: Should only return versions for that package
        """
        # Arrange
        import shutil

        # Store mod-a versions
        for v in ["1.0.0", "1.1.0"]:
            zip_copy = temp_cache.cache_dir / f"mod_a_{v}.zip"
            shutil.copy(sample_zip, zip_copy)
            temp_cache.store_uploaded_package("mod-a", v, zip_copy)

        # Store mod-b versions
        for v in ["2.0.0"]:
            zip_copy = temp_cache.cache_dir / f"mod_b_{v}.zip"
            shutil.copy(sample_zip, zip_copy)
            temp_cache.store_uploaded_package("mod-b", v, zip_copy)

        # Act & Assert
        assert len(temp_cache.list_versions("mod-a")) == 2
        assert len(temp_cache.list_versions("mod-b")) == 1

    def test_get_latest_version(self, temp_cache, sample_zip):
        """
        Test: Get the latest version of a package.

        GIVEN: An UploadCache with multiple versions
        WHEN: get_latest_version() is called
        THEN: Should return the highest version
        """
        # Arrange

        import shutil
        versions = ["1.0.0", "1.5.0", "1.2.0", "2.0.0"]
        for v in versions:
            zip_copy = temp_cache.cache_dir / f"temp_{v}.zip"
            shutil.copy(sample_zip, zip_copy)
            temp_cache.store_uploaded_package("test-mod", v, zip_copy)

        # Act
        latest = temp_cache.get_latest_version("test-mod")

        # Assert
        assert latest == "2.0.0"

    def test_delete_package(self, temp_cache, sample_zip):
        """
        Test: Delete a package from cache.

        GIVEN: An UploadCache with a stored package
        WHEN: delete_package() is called
        THEN: Package should be removed from cache
        """
        # Arrange
        temp_cache.store_uploaded_package("test-mod", "1.0.0", sample_zip)
        assert temp_cache.has_package("test-mod", "1.0.0") is True

        # Act
        temp_cache.delete_package("test-mod", "1.0.0")

        # Assert
        assert temp_cache.has_package("test-mod", "1.0.0") is False

    def test_clear_package(self, temp_cache, sample_zip):
        """
        Test: Clear all versions of a package.

        GIVEN: An UploadCache with multiple versions
        WHEN: clear_package() is called
        THEN: All versions should be removed
        """
        # Arrange
        import shutil
        for v in ["1.0.0", "1.1.0", "2.0.0"]:
            zip_copy = temp_cache.cache_dir / f"temp_{v}.zip"
            shutil.copy(sample_zip, zip_copy)
            temp_cache.store_uploaded_package("test-mod", v, zip_copy)

        # Act
        temp_cache.clear_package("test-mod")

        # Assert
        assert temp_cache.list_versions("test-mod") == []

    def test_cache_size(self, temp_cache, sample_zip):
        """
        Test: Get total cache size.

        GIVEN: An UploadCache with stored packages
        WHEN: get_cache_size() is called
        THEN: Should return total size in bytes
        """
        # Arrange
        temp_cache.store_uploaded_package("mod-a", "1.0.0", sample_zip)
        temp_cache.store_uploaded_package("mod-b", "1.0.0", sample_zip)

        # Act
        size = temp_cache.get_cache_size()

        # Assert
        assert size > 0
        # Should be approximately 2x the zip file size
        expected = sample_zip.stat().st_size * 2
        assert abs(size - expected) < 1000  # Allow small margin
