"""Tests for Rollback Manager."""

import pytest

from toolboxv2.utils.extras.install.rollback_manager import (
    BackupMetadata,
    RollbackManager,
    calculate_checksum,
    calculate_directory_size,
)


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for tests."""
    test_dir = tmp_path / "test_rollback"
    test_dir.mkdir()
    yield test_dir
    # Cleanup is handled by tmp_path fixture


@pytest.fixture
def sample_package(temp_dir):
    """Create a sample package directory."""
    pkg_dir = temp_dir / "test_package"
    pkg_dir.mkdir()

    # Create some files
    (pkg_dir / "__init__.py").write_text("# Test package")
    (pkg_dir / "main.py").write_text("print('hello')")
    (pkg_dir / "config.json").write_text('{"version": "1.0.0"}')

    # Create a subdirectory
    subdir = pkg_dir / "utils"
    subdir.mkdir()
    (subdir / "helper.py").write_text("def helper(): pass")

    return pkg_dir


@pytest.fixture
def rollback_manager(temp_dir):
    """Create a RollbackManager instance for testing."""
    backup_root = temp_dir / "backups"
    return RollbackManager(backup_root)


# =================== BackupMetadata Tests ===================


def test_backup_metadata_to_dict():
    """Test BackupMetadata serialization to dict."""
    metadata = BackupMetadata(
        backup_id="test_123",
        package_name="test_pkg",
        version="1.0.0",
        source_path="/path/to/source",
        backup_path="/path/to/backup",
        checksum="abc123",
        dependencies=["dep1", "dep2"],
        config={"key": "value"},
    )

    data = metadata.to_dict()

    assert data["backup_id"] == "test_123"
    assert data["package_name"] == "test_pkg"
    assert data["version"] == "1.0.0"
    assert data["checksum"] == "abc123"
    assert data["dependencies"] == ["dep1", "dep2"]
    assert data["config"]["key"] == "value"


def test_backup_metadata_from_dict():
    """Test BackupMetadata deserialization from dict."""
    data = {
        "backup_id": "test_456",
        "package_name": "test_pkg",
        "version": "2.0.0",
        "source_path": "/path/to/source",
        "backup_path": "/path/to/backup",
        "checksum": "def456",
        "dependencies": [],
        "config": {},
        "created_at": 1234567890.0,
        "size_bytes": 1024,
    }

    metadata = BackupMetadata.from_dict(data)

    assert metadata.backup_id == "test_456"
    assert metadata.package_name == "test_pkg"
    assert metadata.version == "2.0.0"
    assert metadata.checksum == "def456"


def test_backup_metadata_file_io(temp_dir):
    """Test BackupMetadata file save/load."""
    metadata = BackupMetadata(
        backup_id="test_file",
        package_name="test_pkg",
        version="1.0.0",
        source_path="/source",
        backup_path="/backup",
        checksum="checksum123",
    )

    meta_file = temp_dir / "metadata.json"

    # Save
    metadata.to_file(meta_file)

    # Verify file exists
    assert meta_file.exists()

    # Load
    loaded = BackupMetadata.from_file(meta_file)

    assert loaded.backup_id == "test_file"
    assert loaded.package_name == "test_pkg"
    assert loaded.checksum == "checksum123"


# =================== Checksum Tests ===================


def test_calculate_checksum_file(temp_dir):
    """Test checksum calculation for a file."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("hello world")

    checksum = calculate_checksum(test_file)

    # SHA256 of "hello world"
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert checksum == expected


def test_calculate_checksum_directory(sample_package):
    """Test checksum calculation for a directory."""
    checksum1 = calculate_checksum(sample_package)
    checksum2 = calculate_checksum(sample_package)

    # Same directory should produce same checksum
    assert checksum1 == checksum2
    assert len(checksum1) == 64  # SHA256 hex length


def test_calculate_checksum_change_detection(sample_package):
    """Test that checksum detects file changes."""
    checksum1 = calculate_checksum(sample_package)

    # Modify a file
    (sample_package / "main.py").write_text("print('changed')")

    checksum2 = calculate_checksum(sample_package)

    assert checksum1 != checksum2


def test_calculate_directory_size(sample_package):
    """Test directory size calculation."""
    size = calculate_directory_size(sample_package)

    assert size > 0
    assert isinstance(size, int)


def test_calculate_directory_size_nonexistent(temp_dir):
    """Test size calculation for nonexistent path."""
    size = calculate_directory_size(temp_dir / "nonexistent")
    assert size == 0


# =================== RollbackManager Tests ===================


def test_rollback_manager_initialization(temp_dir):
    """Test RollbackManager initialization."""
    backup_root = temp_dir / "backups"
    manager = RollbackManager(backup_root)

    assert manager.backup_root == backup_root
    assert manager.metadata_dir.exists()
    assert manager.packages_dir.exists()


def test_create_backup(rollback_manager, sample_package):
    """Test creating a backup."""
    metadata = rollback_manager.create_backup(
        package_name="test_package",
        version="1.0.0",
        source_path=sample_package,
        dependencies=["dep1"],
        config={"key": "value"},
    )

    assert metadata.package_name == "test_package"
    assert metadata.version == "1.0.0"
    assert metadata.dependencies == ["dep1"]
    assert metadata.config["key"] == "value"
    assert metadata.checksum
    assert metadata.size_bytes > 0

    # Verify backup exists
    backup_path = rollback_manager._get_backup_path(metadata.backup_id)
    assert backup_path.exists()
    assert (backup_path / "test_package").exists()


def test_create_backup_nonexistent_path(rollback_manager, temp_dir):
    """Test backup creation with nonexistent path."""
    with pytest.raises(FileNotFoundError):
        rollback_manager.create_backup(
            package_name="nonexistent",
            version="1.0.0",
            source_path=temp_dir / "does_not_exist",
        )


def test_restore_backup(rollback_manager, sample_package, temp_dir):
    """Test restoring a backup."""
    # Create backup
    metadata = rollback_manager.create_backup(
        package_name="test_package",
        version="1.0.0",
        source_path=sample_package,
    )

    # Modify original
    (sample_package / "main.py").write_text("modified content")

    # Restore to different location
    restore_path = temp_dir / "restored"
    result = rollback_manager.restore_backup(
        backup_id=metadata.backup_id,
        target_path=restore_path,
    )

    assert result is True
    assert restore_path.exists()
    assert (restore_path / "main.py").read_text() == "print('hello')"


def test_restore_backup_with_checksum_verification(rollback_manager, sample_package, temp_dir):
    """Test restore with checksum verification."""
    metadata = rollback_manager.create_backup(
        package_name="test_package",
        version="1.0.0",
        source_path=sample_package,
    )

    # Corrupt the backup
    backup_path = rollback_manager._get_backup_path(metadata.backup_id)
    corrupted_file = backup_path / "test_package" / "main.py"
    if corrupted_file.exists():
        corrupted_file.write_text("corrupted data")

    # Restore should fail due to checksum mismatch
    restore_path = temp_dir / "restored"
    result = rollback_manager.restore_backup(
        backup_id=metadata.backup_id,
        target_path=restore_path,
        verify_checksum=True,
    )

    assert result is False


def test_restore_nonexistent_backup(rollback_manager, temp_dir):
    """Test restoring a nonexistent backup."""
    result = rollback_manager.restore_backup("nonexistent_id", temp_dir / "target")
    assert result is False


def test_get_latest_backup(rollback_manager, sample_package):
    """Test getting the latest backup."""
    # Create multiple backups
    metadata1 = rollback_manager.create_backup(
        package_name="test_package",
        version="1.0.0",
        source_path=sample_package,
    )

    # Modify and create another backup
    (sample_package / "main.py").write_text("v2")
    metadata2 = rollback_manager.create_backup(
        package_name="test_package",
        version="2.0.0",
        source_path=sample_package,
    )

    latest = rollback_manager.get_latest_backup("test_package")

    assert latest is not None
    assert latest.backup_id == metadata2.backup_id
    assert latest.version == "2.0.0"


def test_get_latest_backup_no_backups(rollback_manager):
    """Test getting latest backup when none exist."""
    latest = rollback_manager.get_latest_backup("nonexistent")
    assert latest is None


def test_list_backups(rollback_manager, sample_package):
    """Test listing all backups."""
    # Create backups for different packages
    metadata1 = rollback_manager.create_backup(
        package_name="package1",
        version="1.0.0",
        source_path=sample_package,
    )

    metadata2 = rollback_manager.create_backup(
        package_name="package2",
        version="1.0.0",
        source_path=sample_package,
    )

    # List all
    all_backups = rollback_manager.list_backups()
    assert len(all_backups) >= 2

    # List filtered
    pkg1_backups = rollback_manager.list_backups("package1")
    assert len(pkg1_backups) == 1
    assert pkg1_backups[0].package_name == "package1"


def test_delete_backup(rollback_manager, sample_package):
    """Test deleting a backup."""
    metadata = rollback_manager.create_backup(
        package_name="test_package",
        version="1.0.0",
        source_path=sample_package,
    )

    backup_id = metadata.backup_id
    backup_path = rollback_manager._get_backup_path(backup_id)
    metadata_path = rollback_manager._get_metadata_path(backup_id)

    assert backup_path.exists()
    assert metadata_path.exists()

    # Delete
    result = rollback_manager.delete_backup(backup_id)

    assert result is True
    assert not backup_path.exists()
    assert not metadata_path.exists()


def test_cleanup_old_backups(rollback_manager, sample_package):
    """Test cleanup of old backups."""
    # Create multiple backups
    for i in range(7):
        rollback_manager.create_backup(
            package_name="test_package",
            version=f"1.{i}.0",
            source_path=sample_package,
        )

    initial_count = rollback_manager.get_backup_count()
    assert initial_count == 7

    # Cleanup keeping only 5 most recent
    deleted = rollback_manager.cleanup_old_backups(
        max_age_days=0,
        keep_per_package=5,
    )

    # Should delete 2 oldest
    assert deleted == 2
    assert rollback_manager.get_backup_count() == 5


def test_get_total_backup_size(rollback_manager, sample_package):
    """Test getting total backup size."""
    rollback_manager.create_backup(
        package_name="test_package",
        version="1.0.0",
        source_path=sample_package,
    )

    total_size = rollback_manager.get_total_backup_size()
    assert total_size > 0


def test_get_backup_count(rollback_manager, sample_package):
    """Test getting backup count."""
    assert rollback_manager.get_backup_count() == 0

    rollback_manager.create_backup(
        package_name="test_package",
        version="1.0.0",
        source_path=sample_package,
    )

    assert rollback_manager.get_backup_count() == 1


def test_backup_preserves_file_structure(sample_package, rollback_manager, temp_dir):
    """Test that backup preserves directory structure."""
    # Create nested structure
    nested_dir = sample_package / "deep" / "nested" / "dir"
    nested_dir.mkdir(parents=True)
    (nested_dir / "file.txt").write_text("nested content")

    metadata = rollback_manager.create_backup(
        package_name="test_package",
        version="1.0.0",
        source_path=sample_package,
    )

    # Restore and verify structure
    restore_path = temp_dir / "restored"
    rollback_manager.restore_backup(metadata.backup_id, restore_path)

    assert (restore_path / "deep" / "nested" / "dir" / "file.txt").exists()
    assert (restore_path / "deep" / "nested" / "dir" / "file.txt").read_text() == "nested content"


def test_create_backup_single_file(temp_dir, rollback_manager):
    """Test creating backup of a single file."""
    test_file = temp_dir / "single_file.txt"
    test_file.write_text("single file content")

    metadata = rollback_manager.create_backup(
        package_name="single_file",
        version="1.0.0",
        source_path=test_file,
    )

    assert metadata.checksum
    backup_path = rollback_manager._get_backup_path(metadata.backup_id)
    assert backup_path.exists()
