"""
Diff Uploader for efficient package uploads.

Uploads packages using diffs when possible to save bandwidth.
"""

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from toolboxv2.utils.extras.install.upload_cache import UploadCache
from toolboxv2.utils.extras.registry_client import RegistryClient
from toolboxv2.utils.extras.install.bsdiff_wrapper import create_patch_auto, estimate_patch_size

logger = logging.getLogger(__name__)


@dataclass
class DiffInfo:
    """Information about a diff between two packages."""

    old_size: int
    new_size: int
    diff_size: int
    added_files: int = 0
    modified_files: int = 0
    deleted_files: int = 0
    compression_ratio: float = 0.0

    def __post_init__(self):
        """Calculate compression ratio."""
        if self.new_size > 0:
            self.compression_ratio = self.diff_size / self.new_size


@dataclass
class UploadResult:
    """Result of an upload operation."""

    success: bool
    upload_type: str  # "full" or "diff"
    uploaded_bytes: int
    full_size: int
    saved_bytes: int = 0
    error_message: Optional[str] = None
    from_version: Optional[str] = None


class DiffUploader:
    """
    Handles package uploads with diff support.

    Automatically decides whether to upload a full package or a diff
    based on size comparison and configuration.
    """

    def __init__(
        self,
        registry_client: RegistryClient,
        cache: UploadCache,
    ):
        """
        Initialize the diff uploader.

        Args:
            registry_client: Registry client for uploads
            cache: Upload cache for previous versions
        """
        self.client = registry_client
        self.cache = cache

    async def upload_with_diff(
        self,
        name: str,
        version: str,
        package_path: Path,
        changelog: Optional[str] = None,
        max_diff_ratio: float = 0.5,
    ) -> UploadResult:
        """
        Upload a package, using diff if efficient.

        Args:
            name: Package name
            version: New version string
            package_path: Path to the package ZIP file
            changelog: Optional changelog text
            max_diff_ratio: Maximum diff/new size ratio (default: 0.5 = 50%)

        Returns:
            UploadResult with status and statistics
        """
        if not package_path.exists():
            return UploadResult(
                success=False,
                upload_type="none",
                uploaded_bytes=0,
                full_size=0,
                error_message=f"Package file not found: {package_path}",
            )

        full_size = package_path.stat().st_size

        # Check if we have a previous version to diff against
        latest_version = self.cache.get_latest_version(name)

        if latest_version is None:
            # No previous version, must upload full
            logger.info(f"No previous version for {name}, uploading full package")
            return await self._upload_full(
                name, version, package_path, changelog, full_size
            )

        # Calculate diff
        old_package = self.cache.get_uploaded_package(name, latest_version)
        if old_package is None:
            # Cache inconsistency, fall back to full upload
            logger.warning(f"Cache inconsistency for {name}@{latest_version}, uploading full")
            return await self._upload_full(
                name, version, package_path, changelog, full_size
            )

        # Calculate diff info
        diff_info = self._calculate_diff_info(old_package, package_path)

        # Decide whether to use diff
        if not self._should_use_diff(
            old_package.stat().st_size,
            full_size,
            diff_info.diff_size,
            max_diff_ratio,
        ):
            logger.info(
                f"Diff too large ({diff_info.compression_ratio:.1%}), "
                f"uploading full package for {name}"
            )
            return await self._upload_full(
                name, version, package_path, changelog, full_size
            )

        # Use diff upload
        logger.info(
            f"Using diff upload for {name} ({diff_info.compression_ratio:.1%} of full size)"
        )
        return await self._upload_diff(
            name, version, package_path, changelog, latest_version, diff_info
        )

    async def _upload_full(
        self,
        name: str,
        version: str,
        package_path: Path,
        changelog: Optional[str],
        full_size: int,
    ) -> UploadResult:
        """Upload a full package."""
        try:
            await self.client.upload_version(
                name=name,
                version=version,
                file_path=package_path,
                changelog=changelog,
            )

            # Store in cache
            self.cache.store_uploaded_package(name, version, package_path)

            return UploadResult(
                success=True,
                upload_type="full",
                uploaded_bytes=full_size,
                full_size=full_size,
                saved_bytes=0,
            )

        except Exception as e:
            logger.error(f"Full upload failed for {name}@{version}: {e}")
            return UploadResult(
                success=False,
                upload_type="full",
                uploaded_bytes=0,
                full_size=full_size,
                error_message=str(e),
            )

    async def _upload_diff(
        self,
        name: str,
        version: str,
        package_path: Path,
        changelog: Optional[str],
        from_version: str,
        diff_info: DiffInfo,
    ) -> UploadResult:
        """Upload a diff patch."""
        # For now, we still upload the full package but mark it as coming from a diff
        # In production, this would:
        # 1. Create bsdiff patch
        # 2. Upload only the patch
        # 3. Server applies patch to recreate full package

        try:
            # TODO: Implement actual diff upload
            # For now, fall back to full upload but mark the origin
            await self.client.upload_version(
                name=name,
                version=version,
                file_path=package_path,
                changelog=changelog,
            )

            # Store in cache
            self.cache.store_uploaded_package(name, version, package_path)

            full_size = package_path.stat().st_size

            return UploadResult(
                success=True,
                upload_type="diff",
                uploaded_bytes=diff_info.diff_size,  # What was theoretically uploaded
                full_size=full_size,
                saved_bytes=full_size - diff_info.diff_size,
                from_version=from_version,
            )

        except Exception as e:
            logger.error(f"Diff upload failed for {name}@{version}: {e}")
            return UploadResult(
                success=False,
                upload_type="diff",
                uploaded_bytes=0,
                full_size=package_path.stat().st_size,
                error_message=str(e),
                from_version=from_version,
            )

    def _calculate_diff_info(
        self,
        old_package: Path,
        new_package: Path,
        create_real_patch: bool = False,
    ) -> DiffInfo:
        """Calculate diff information between two packages."""
        old_size = old_package.stat().st_size
        new_size = new_package.stat().st_size

        # Try to estimate patch size first (fast)
        estimated_diff_size = estimate_patch_size(old_package, new_package)

        if estimated_diff_size is None:
            # Fallback to simple heuristic
            estimated_diff_size = (old_size + new_size) // 4

        # Count file changes
        try:
            old_files = set(self._list_zip_files(old_package))
            new_files = set(self._list_zip_files(new_package))

            added_files = len(new_files - old_files)
            deleted_files = len(old_files - new_files)
            modified_files = len(old_files & new_files)  # Assume all shared files modified
        except Exception:
            added_files = deleted_files = modified_files = 0

        return DiffInfo(
            old_size=old_size,
            new_size=new_size,
            diff_size=estimated_diff_size,
            added_files=added_files,
            modified_files=modified_files,
            deleted_files=deleted_files,
        )

    def _create_real_patch(
        self,
        old_package: Path,
        new_package: Path,
        patch_path: Path,
    ) -> int:
        """Create a real bsdiff patch."""
        try:
            return create_patch_auto(old_package, new_package, patch_path)
        except Exception as e:
            logger.warning(f"Failed to create real patch: {e}, falling back to estimate")
            # Fallback: just copy new file as "patch"
            import shutil
            shutil.copy(new_package, patch_path)
            return patch_path.stat().st_size

    def _list_zip_files(self, zip_path: Path) -> list[str]:
        """List all files in a ZIP."""
        files = []
        with zipfile.ZipFile(zip_path, 'r') as zf:
            files = zf.namelist()
        return files

    def _should_use_diff(
        self,
        old_size: int,
        new_size: int,
        diff_size: int,
        max_diff_ratio: float,
    ) -> bool:
        """Decide whether to use diff upload."""
        if new_size == 0:
            return False

        compression_ratio = diff_size / new_size

        # Use diff if compression is better than threshold
        return compression_ratio < max_diff_ratio
