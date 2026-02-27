"""Diff generation for package updates.

This module provides functionality to create binary diffs between package versions,
allowing for efficient incremental updates instead of full downloads.
"""

import hashlib
import logging
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from registry.db.repositories.package_repo import PackageRepository

logger = logging.getLogger(__name__)


@dataclass
class FileDiff:
    """Difference information for a single file.

    Attributes:
        path: Relative path of the file
        action: "added", "modified", or "deleted"
        old_checksum: SHA256 of old file (None for added)
        new_checksum: SHA256 of new file (None for deleted)
        size_diff: Size difference in bytes
    """

    path: str
    action: str
    old_checksum: Optional[str] = None
    new_checksum: Optional[str] = None
    size_diff: int = 0


@dataclass
class DiffInfo:
    """Information about a diff between two versions.

    Attributes:
        from_version: Source version
        to_version: Target version
        patch_checksum: SHA256 of patch file
        patch_size: Size of patch file in bytes
        compression_ratio: Ratio of patch size to full package size
        changed_files: Number of changed files
        added_files: Number of added files
        deleted_files: Number of deleted files
        patch_storage_path: Storage path for patch file
        created_at: When the diff was created
    """

    from_version: str
    to_version: str
    package_name: str
    patch_checksum: str = ""
    patch_size: int = 0
    compression_ratio: float = 0.0
    changed_files: int = 0
    added_files: int = 0
    deleted_files: int = 0
    patch_storage_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class DiffGenerator:
    """Generate binary diffs between package versions.

    This class handles:
    - Extracting package ZIP files
    - Calculating file-level diffs
    - Creating binary patch files
    - Managing patch storage
    """

    def __init__(self, package_repo: PackageRepository, storage_backend=None):
        """Initialize the diff generator.

        Args:
            package_repo: Package repository for version metadata
            storage_backend: Optional storage backend for patch files
        """
        self.package_repo = package_repo
        self.storage = storage_backend

    async def create_diff(
        self,
        package_name: str,
        from_version: str,
        to_version: str,
        force: bool = False,
    ) -> DiffInfo:
        """Create a diff between two package versions.

        Args:
            package_name: Name of the package
            from_version: Source version
            to_version: Target version
            force: Force recreation even if diff exists

        Returns:
            DiffInfo with diff metadata

        Raises:
            ValueError: If versions are the same or don't exist
        """
        if from_version == to_version:
            raise ValueError("Cannot create diff: versions are identical")

        # Get version metadata
        from_version_data = await self.package_repo.get_version(package_name, from_version)
        to_version_data = await self.package_repo.get_version(package_name, to_version)

        if not from_version_data:
            raise ValueError(f"Source version {from_version} not found")
        if not to_version_data:
            raise ValueError(f"Target version {to_version} not found")

        logger.info(f"Creating diff: {package_name} {from_version} → {to_version}")

        # Get storage locations
        from_storage = from_version_data.storage_locations[0] if from_version_data.storage_locations else None
        to_storage = to_version_data.storage_locations[0] if to_version_data.storage_locations else None

        if not from_storage or not to_storage:
            raise ValueError("Storage locations not available for one or both versions")

        # Download packages to temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            from_zip = await self._download_package(from_storage, temp_path / "from.zip")
            to_zip = await self._download_package(to_storage, temp_path / "to.zip")

            # Calculate file diffs
            file_diffs = await self._calculate_file_diffs(from_zip, to_zip)

            # Create binary patch
            patch_path = temp_path / "patch.bin"
            await self._create_binary_patch(from_zip, to_zip, patch_path)

            # Calculate metrics
            patch_size = patch_path.stat().st_size
            full_size = to_zip.stat().st_size
            compression_ratio = patch_size / full_size if full_size > 0 else 0.0

            # Calculate checksum
            patch_checksum = self._calculate_checksum(patch_path)

            # Count file changes
            changed_files = sum(1 for d in file_diffs if d.action == "modified")
            added_files = sum(1 for d in file_diffs if d.action == "added")
            deleted_files = sum(1 for d in file_diffs if d.action == "deleted")

            # Store patch (if storage backend available)
            patch_storage_path = None
            if self.storage:
                patch_storage_path = await self._store_patch(
                    package_name, from_version, to_version, patch_path, patch_checksum
                )

            logger.info(
                f"Diff created: {patch_size} bytes ({compression_ratio:.1%} of full size), "
                f"{changed_files} changed, {added_files} added, {deleted_files} deleted"
            )

            return DiffInfo(
                from_version=from_version,
                to_version=to_version,
                package_name=package_name,
                patch_checksum=patch_checksum,
                patch_size=patch_size,
                compression_ratio=compression_ratio,
                changed_files=changed_files,
                added_files=added_files,
                deleted_files=deleted_files,
                patch_storage_path=patch_storage_path,
            )

    async def _download_package(self, storage_location: Any, dest_path: Path) -> Path:
        """Download a package from storage.

        Args:
            storage_location: StorageLocation object
            dest_path: Destination path

        Returns:
            Path to downloaded file
        """
        # In a real implementation, this would download from MinIO/S3
        # For now, assume the file is accessible locally
        if hasattr(storage_location, 'path'):
            src = Path(storage_location.path)
            if src.exists():
                import shutil
                shutil.copy(src, dest_path)
                return dest_path

        # Placeholder: would implement actual download here
        raise NotImplementedError("Package download not implemented")

    async def _calculate_file_diffs(
        self,
        from_zip: Path,
        to_zip: Path,
    ) -> List[FileDiff]:
        """Calculate file-level differences between two package ZIPs.

        Args:
            from_zip: Path to source ZIP
            to_zip: Path to target ZIP

        Returns:
            List of FileDiff objects
        """
        diffs = []

        # Read file lists
        from_files = self._list_zip_contents(from_zip)
        to_files = self._list_zip_contents(to_zip)

        # Check for modified and new files
        for file_path, file_info in to_files.items():
            if file_path in from_files:
                # File exists in both - check if changed
                if file_info["checksum"] != from_files[file_path]["checksum"]:
                    size_diff = file_info["size"] - from_files[file_path]["size"]
                    diffs.append(FileDiff(
                        path=file_path,
                        action="modified",
                        old_checksum=from_files[file_path]["checksum"],
                        new_checksum=file_info["checksum"],
                        size_diff=size_diff,
                    ))
            else:
                # New file
                diffs.append(FileDiff(
                    path=file_path,
                    action="added",
                    new_checksum=file_info["checksum"],
                    size_diff=file_info["size"],
                ))

        # Check for deleted files
        for file_path, file_info in from_files.items():
            if file_path not in to_files:
                diffs.append(FileDiff(
                    path=file_path,
                    action="deleted",
                    old_checksum=file_info["checksum"],
                    size_diff=-file_info["size"],
                ))

        return diffs

    async def _create_binary_patch(
        self,
        from_zip: Path,
        to_zip: Path,
        patch_path: Path,
    ) -> None:
        """Create a binary patch file.

        For simplicity, we currently use the full target file as the "patch".
        In production, this would use bsdiff/xdelta for actual binary diffs.

        Args:
            from_zip: Source ZIP file
            to_zip: Target ZIP file
            patch_path: Where to write the patch
        """
        # Simple implementation: copy target as patch
        # In production, use: bsdiff old.zip new.zip patch.bin
        import shutil

        # For now, just use a copy-to-patch approach
        # This gives us the infrastructure while bsdiff integration can come later
        shutil.copy(to_zip, patch_path)

        logger.debug(f"Binary patch created: {patch_path}")

    def _list_zip_contents(self, zip_path: Path) -> Dict[str, Dict[str, Any]]:
        """List contents of a ZIP file with checksums.

        Args:
            zip_path: Path to ZIP file

        Returns:
            Dict of filename -> {size, checksum}
        """
        contents = {}

        with zipfile.ZipFile(zip_path, 'r') as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue

                # Calculate checksum
                with zf.open(info) as f:
                    file_checksum = hashlib.sha256(f.read()).hexdigest()

                contents[info.filename] = {
                    "size": info.file_size,
                    "checksum": file_checksum,
                }

        return contents

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file.

        Args:
            file_path: Path to file

        Returns:
            Hexadecimal checksum string
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def _store_patch(
        self,
        package_name: str,
        from_version: str,
        to_version: str,
        patch_path: Path,
        checksum: str,
    ) -> str:
        """Store patch file in storage backend.

        Args:
            package_name: Package name
            from_version: Source version
            to_version: Target version
            patch_path: Path to patch file
            checksum: SHA256 checksum

        Returns:
            Storage path/key
        """
        # Storage path: patches/{package_name}/{from_version}-to-{to_version}.bin
        storage_path = f"patches/{package_name}/{from_version}-to-{to_version}.bin"

        if self.storage:
            # Upload to storage backend (MinIO/S3)
            with open(patch_path, 'rb') as f:
                await self.storage.put(storage_path, f, length=patch_path.stat().st_size)

        return storage_path

    async def get_diff_info(
        self,
        package_name: str,
        from_version: str,
        to_version: str,
    ) -> Optional[DiffInfo]:
        """Get diff information without creating it.

        Args:
            package_name: Package name
            from_version: Source version
            to_version: Target version

        Returns:
            DiffInfo if diff exists, None otherwise
        """
        # Check if patch exists in storage
        storage_path = f"patches/{package_name}/{from_version}-to-{to_version}.bin"

        if self.storage:
            exists = await self.storage.exists(storage_path)
            if not exists:
                return None

            # Get patch metadata
            metadata = await self.storage.stat(storage_path)

            return DiffInfo(
                from_version=from_version,
                to_version=to_version,
                package_name=package_name,
                patch_size=metadata.size,
                patch_storage_path=storage_path,
            )

        return None
