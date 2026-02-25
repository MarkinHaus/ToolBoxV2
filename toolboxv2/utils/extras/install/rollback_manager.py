"""
Rollback Manager for Package Updates.

This module provides backup and rollback functionality for package updates.
It creates backups before updates and can restore them on failure.

Classes:
    BackupMetadata: Dataclass for backup information
    RollbackManager: Main manager class for backups

Functions:
    calculate_checksum(path: Path) -> str
        Calculate SHA256 checksum of a file or directory
"""

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BackupMetadata:
    """Metadata for a package backup.

    Attributes:
        backup_id: Unique backup identifier (timestamp-based)
        package_name: Name of the backed up package
        version: Version of the backed up package
        source_path: Original path of the package
        backup_path: Path where backup is stored
        checksum: SHA256 checksum for verification
        dependencies: List of dependency package names
        config: Package configuration dict
        created_at: Unix timestamp of creation
        size_bytes: Size of backup in bytes
    """
    backup_id: str
    package_name: str
    version: str
    source_path: str
    backup_path: str
    checksum: str
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackupMetadata":
        """Create from dictionary."""
        return cls(**data)

    def to_file(self, path: Path) -> None:
        """Save metadata to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_file(cls, path: Path) -> "BackupMetadata":
        """Load metadata from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def calculate_checksum(path: Path) -> str:
    """Calculate SHA256 checksum of a file or directory.

    Args:
        path: Path to file or directory

    Returns:
        Hexadecimal SHA256 checksum string
    """
    hasher = hashlib.sha256()

    if path.is_file():
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    # For directories, hash all files recursively
    if path.is_dir():
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file():
                # Include relative path in hash
                rel_path = file_path.relative_to(path)
                hasher.update(str(rel_path).encode())
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)
        return hasher.hexdigest()

    return ""


def calculate_directory_size(path: Path) -> int:
    """Calculate total size of a directory in bytes.

    Args:
        path: Path to directory

    Returns:
        Total size in bytes
    """
    if not path.exists():
        return 0

    if path.is_file():
        return path.stat().st_size

    total_size = 0
    for item in path.rglob("*"):
        if item.is_file():
            total_size += item.stat().st_size
    return total_size


class RollbackManager:
    """Manager for creating and restoring package backups.

    Provides:
    - Backup creation with checksums
    - Backup restoration with verification
    - Backup listing and querying
    - Automatic cleanup of old backups
    """

    def __init__(self, backup_root: Path):
        """Initialize rollback manager.

        Args:
            backup_root: Root directory for storing backups
        """
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(parents=True, exist_ok=True)

        # Metadata storage
        self.metadata_dir = self.backup_root / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Package backups storage
        self.packages_dir = self.backup_root / "packages"
        self.packages_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

    def _generate_backup_id(self, package_name: str) -> str:
        """Generate unique backup ID.

        Args:
            package_name: Name of the package

        Returns:
            Unique backup ID string
        """
        timestamp = int(time.time() * 1000)
        return f"{package_name}_{timestamp}"

    def _get_backup_path(self, backup_id: str) -> Path:
        """Get storage path for a backup.

        Args:
            backup_id: Backup identifier

        Returns:
            Path to backup directory
        """
        return self.packages_dir / backup_id

    def _get_metadata_path(self, backup_id: str) -> Path:
        """Get path to backup metadata file.

        Args:
            backup_id: Backup identifier

        Returns:
            Path to metadata JSON file
        """
        return self.metadata_dir / f"{backup_id}.json"

    def create_backup(
        self,
        package_name: str,
        version: str,
        source_path: Path,
        dependencies: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> BackupMetadata:
        """Create a backup of a package.

        Args:
            package_name: Name of the package
            version: Package version
            source_path: Path to the package to backup
            dependencies: List of dependency names
            config: Package configuration

        Returns:
            BackupMetadata with backup information

        Raises:
            FileNotFoundError: If source path doesn't exist
            IOError: If backup creation fails
        """
        source_path = Path(source_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")

        backup_id = self._generate_backup_id(package_name)
        backup_path = self._get_backup_path(backup_id)

        self.logger.info(f"Creating backup {backup_id} for {package_name}@{version}")

        try:
            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)

            # Copy package to backup location
            if source_path.is_dir():
                shutil.copytree(source_path, backup_path / package_name, dirs_exist_ok=True)
                checksum = calculate_checksum(backup_path / package_name)
                size_bytes = calculate_directory_size(backup_path / package_name)
            else:
                shutil.copy2(source_path, backup_path / source_path.name)
                checksum = calculate_checksum(backup_path / source_path.name)
                size_bytes = (backup_path / source_path.name).stat().st_size

            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                package_name=package_name,
                version=version,
                source_path=str(source_path),
                backup_path=str(backup_path),
                checksum=checksum,
                dependencies=dependencies or [],
                config=config or {},
                size_bytes=size_bytes,
            )

            # Save metadata
            metadata.to_file(self._get_metadata_path(backup_id))

            self.logger.info(f"Backup created successfully: {backup_id}")
            return metadata

        except Exception as e:
            # Clean up failed backup
            if backup_path.exists():
                shutil.rmtree(backup_path, ignore_errors=True)
            metadata_path = self._get_metadata_path(backup_id)
            if metadata_path.exists():
                metadata_path.unlink()

            self.logger.error(f"Failed to create backup: {e}")
            raise IOError(f"Backup creation failed: {e}") from e

    def restore_backup(
        self,
        backup_id: str,
        target_path: Optional[Path] = None,
        verify_checksum: bool = True,
    ) -> bool:
        """Restore a package from backup.

        Args:
            backup_id: Backup identifier to restore
            target_path: Target path for restoration (default: original source path)
            verify_checksum: Whether to verify checksum before restoration

        Returns:
            True if restoration successful, False otherwise
        """
        metadata_path = self._get_metadata_path(backup_id)

        if not metadata_path.exists():
            self.logger.error(f"Backup metadata not found: {backup_id}")
            return False

        try:
            # Load metadata
            metadata = BackupMetadata.from_file(metadata_path)
            backup_path = Path(metadata.backup_path)

            if not backup_path.exists():
                self.logger.error(f"Backup data not found: {backup_id}")
                return False

            # Verify checksum if requested
            if verify_checksum:
                if metadata.package_name in [p.name for p in backup_path.iterdir() if p.is_dir()]:
                    package_backup_path = backup_path / metadata.package_name
                else:
                    package_backup_path = backup_path

                current_checksum = calculate_checksum(package_backup_path)
                if current_checksum != metadata.checksum:
                    self.logger.error(
                        f"Checksum mismatch for {backup_id}: "
                        f"expected {metadata.checksum}, got {current_checksum}"
                    )
                    return False

            # Determine target path
            if target_path is None:
                target_path = Path(metadata.source_path)
            else:
                target_path = Path(target_path)

            # Remove existing target if it exists
            if target_path.exists():
                if target_path.is_dir():
                    shutil.rmtree(target_path)
                else:
                    target_path.unlink()

            # Restore from backup
            target_path.parent.mkdir(parents=True, exist_ok=True)

            if metadata.package_name in [p.name for p in backup_path.iterdir() if p.is_dir()]:
                # Directory backup
                shutil.copytree(backup_path / metadata.package_name, target_path)
            else:
                # File backup
                for file in backup_path.iterdir():
                    if file.is_file():
                        shutil.copy2(file, target_path / file.name)

            self.logger.info(f"Backup restored successfully: {backup_id} -> {target_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to restore backup {backup_id}: {e}")
            return False

    def get_latest_backup(self, package_name: str) -> Optional[BackupMetadata]:
        """Get the latest backup for a package.

        Args:
            package_name: Name of the package

        Returns:
            Latest BackupMetadata or None if no backups found
        """
        backups = self.list_backups(package_name)

        if not backups:
            return None

        # Sort by created_at descending
        backups.sort(key=lambda b: b.created_at, reverse=True)
        return backups[0]

    def list_backups(self, package_name: Optional[str] = None) -> List[BackupMetadata]:
        """List all backups, optionally filtered by package.

        Args:
            package_name: Optional package name filter

        Returns:
            List of BackupMetadata objects
        """
        backups = []

        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                metadata = BackupMetadata.from_file(metadata_file)

                if package_name is None or metadata.package_name == package_name:
                    backups.append(metadata)
            except Exception as e:
                self.logger.warning(f"Failed to load metadata from {metadata_file}: {e}")

        return backups

    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup.

        Args:
            backup_id: Backup identifier to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Delete backup data
            backup_path = self._get_backup_path(backup_id)
            if backup_path.exists():
                shutil.rmtree(backup_path)

            # Delete metadata
            metadata_path = self._get_metadata_path(backup_id)
            if metadata_path.exists():
                metadata_path.unlink()

            self.logger.info(f"Backup deleted: {backup_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False

    def cleanup_old_backups(
        self,
        max_age_days: int = 30,
        keep_per_package: int = 5,
        package_name: Optional[str] = None,
    ) -> int:
        """Clean up old backups.

        Removes backups older than max_age_days, but keeps at least
        keep_per_package most recent backups for each package.

        Args:
            max_age_days: Maximum age in days before cleanup
            keep_per_package: Minimum number of recent backups to keep per package
            package_name: Optional package name to limit cleanup

        Returns:
            Number of backups deleted
        """
        now = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        deleted_count = 0

        # Get all backups grouped by package
        backups_by_package: Dict[str, List[BackupMetadata]] = {}

        for backup in self.list_backups(package_name):
            if backup.package_name not in backups_by_package:
                backups_by_package[backup.package_name] = []
            backups_by_package[backup.package_name].append(backup)

        # Process each package
        for pkg_name, backups in backups_by_package.items():
            # Sort by creation time (newest first)
            backups.sort(key=lambda b: b.created_at, reverse=True)

            # Always keep the N most recent
            to_keep = set(b.backup_id for b in backups[:keep_per_package])

            # Also delete old ones beyond max_age
            for backup in backups:
                if backup.backup_id in to_keep:
                    continue

                age = now - backup.created_at
                if age > max_age_seconds:
                    if self.delete_backup(backup.backup_id):
                        deleted_count += 1

        self.logger.info(f"Cleaned up {deleted_count} old backups")
        return deleted_count

    def get_total_backup_size(self) -> int:
        """Get total size of all backups in bytes.

        Returns:
            Total size in bytes
        """
        return calculate_directory_size(self.backup_root)

    def get_backup_count(self) -> int:
        """Get total number of backups.

        Returns:
            Number of backups
        """
        return len(list(self.metadata_dir.glob("*.json")))
