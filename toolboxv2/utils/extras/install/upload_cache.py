"""
Upload Cache for package management.

Caches uploaded packages locally to enable diff uploads.
"""

import logging
import shutil
from pathlib import Path
from typing import List, Optional

from packaging.version import Version

logger = logging.getLogger(__name__)


class UploadCache:
    """
    Manages locally cached uploaded packages.

    This cache stores packages that have been successfully uploaded,
    enabling future uploads to use diffs instead of full uploads.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the upload cache.

        Args:
            cache_dir: Root cache directory. Defaults to ~/.toolboxv2/uploads
        """
        if cache_dir is None:
            home = Path.home()
            cache_dir = home / ".toolboxv2" / "uploads"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"UploadCache initialized at: {self.cache_dir}")

    def _get_package_path(self, name: str, version: str) -> Path:
        """
        Get the cache path for a specific package version.

        Args:
            name: Package name
            version: Version string

        Returns:
            Path where the package should be stored
        """
        # Normalize package name (lowercase, replace spaces with dashes)
        normalized_name = name.lower().replace(" ", "-")
        return self.cache_dir / normalized_name / f"{version}.zip"

    def store_uploaded_package(
        self,
        name: str,
        version: str,
        zip_path: Path,
    ) -> None:
        """
        Store an uploaded package in the cache.

        Args:
            name: Package name
            version: Version string
            zip_path: Path to the ZIP file to cache

        Raises:
            FileNotFoundError: If zip_path doesn't exist
        """
        if not zip_path.exists():
            raise FileNotFoundError(f"Package file not found: {zip_path}")

        dest_path = self._get_package_path(name, version)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the file to cache
        shutil.copy2(zip_path, dest_path)

        logger.debug(f"Cached package: {name}@{version} at {dest_path}")

    def get_uploaded_package(
        self,
        name: str,
        version: str,
    ) -> Optional[Path]:
        """
        Get the cached path for a package version.

        Args:
            name: Package name
            version: Version string

        Returns:
            Path to cached package, or None if not found
        """
        cached_path = self._get_package_path(name, version)

        if cached_path.exists():
            return cached_path

        return None

    def has_package(self, name: str, version: str) -> bool:
        """
        Check if a package version is cached.

        Args:
            name: Package name
            version: Version string

        Returns:
            True if package is cached
        """
        return self._get_package_path(name, version).exists()

    def list_versions(self, name: str) -> List[str]:
        """
        List all cached versions of a package.

        Args:
            name: Package name

        Returns:
            List of version strings (sorted)
        """
        normalized_name = name.lower().replace(" ", "-")
        package_dir = self.cache_dir / normalized_name

        if not package_dir.exists():
            return []

        versions = []
        for file_path in package_dir.glob("*.zip"):
            # Remove .zip extension to get version
            version = file_path.stem
            versions.append(version)

        # Sort versions
        try:
            versions.sort(key=Version)
        except Exception:
            # Fallback to string sort if version parsing fails
            versions.sort()

        return versions

    def get_latest_version(self, name: str) -> Optional[str]:
        """
        Get the latest cached version of a package.

        Args:
            name: Package name

        Returns:
            Latest version string, or None if no versions cached
        """
        versions = self.list_versions(name)

        if not versions:
            return None

        # Versions are already sorted
        return versions[-1]

    def delete_package(self, name: str, version: str) -> None:
        """
        Delete a specific package version from cache.

        Args:
            name: Package name
            version: Version string
        """
        cached_path = self._get_package_path(name, version)

        if cached_path.exists():
            cached_path.unlink()
            logger.debug(f"Deleted cached package: {name}@{version}")

            # Remove package directory if empty
            package_dir = cached_path.parent
            if not any(package_dir.iterdir()):
                package_dir.rmdir()

    def clear_package(self, name: str) -> None:
        """
        Delete all versions of a package from cache.

        Args:
            name: Package name
        """
        normalized_name = name.lower().replace(" ", "-")
        package_dir = self.cache_dir / normalized_name

        if package_dir.exists():
            shutil.rmtree(package_dir)
            logger.debug(f"Cleared all versions of: {name}")

    def get_cache_size(self) -> int:
        """
        Get total cache size in bytes.

        Returns:
            Total size of all cached packages in bytes
        """
        total_size = 0

        for package_dir in self.cache_dir.iterdir():
            if package_dir.is_dir():
                for zip_file in package_dir.glob("*.zip"):
                    total_size += zip_file.stat().st_size

        return total_size

    def list_packages(self) -> List[str]:
        """
        List all package names in the cache.

        Returns:
            List of package names
        """
        packages = []

        for package_dir in self.cache_dir.iterdir():
            if package_dir.is_dir() and any(package_dir.glob("*.zip")):
                # Convert back from normalized name
                packages.append(package_dir.name)

        return sorted(packages)
