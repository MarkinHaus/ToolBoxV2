"""Tests for storage backends."""

import tempfile
from pathlib import Path

import pytest

from registry.storage.manager import StorageManager


class TestStorageManager:
    """Tests for StorageManager."""

    def test_calculate_checksum(self, tmp_path: Path) -> None:
        """Test checksum calculation.

        Args:
            tmp_path: Temporary directory.
        """
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        # Calculate checksum
        import asyncio
        checksum = asyncio.run(StorageManager.calculate_checksum(test_file))

        # Verify it's a valid SHA256 hex string
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_calculate_checksum_empty_file(self, tmp_path: Path) -> None:
        """Test checksum of empty file.

        Args:
            tmp_path: Temporary directory.
        """
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        import asyncio
        checksum = asyncio.run(StorageManager.calculate_checksum(test_file))

        # SHA256 of empty string
        assert checksum == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_calculate_checksum_binary(self, tmp_path: Path) -> None:
        """Test checksum of binary file.

        Args:
            tmp_path: Temporary directory.
        """
        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(bytes(range(256)))

        import asyncio
        checksum = asyncio.run(StorageManager.calculate_checksum(test_file))

        assert len(checksum) == 64

