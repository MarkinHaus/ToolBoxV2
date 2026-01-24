"""Package service for business logic."""

import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from registry.db.repositories.package_repo import PackageRepository
from registry.db.repositories.user_repo import UserRepository
from registry.exceptions import (
    DuplicatePackageError,
    DuplicateVersionError,
    PackageNotFoundError,
    PermissionDeniedError,
    VersionNotFoundError,
)
from registry.models.package import (
    Package,
    PackageCreate,
    PackageSummary,
    PackageType,
    PackageUpdate,
    PackageVersion,
    StorageLocation,
    Visibility,
)
from registry.storage.manager import StorageManager

logger = logging.getLogger(__name__)


class PackageService:
    """Service for package operations.

    Handles business logic for package management.

    Attributes:
        repo: Package repository.
        user_repo: User repository.
        storage: Storage manager.
    """

    def __init__(
        self,
        repo: PackageRepository,
        user_repo: UserRepository,
        storage: StorageManager,
    ) -> None:
        """Initialize the service.

        Args:
            repo: Package repository.
            user_repo: User repository.
            storage: Storage manager.
        """
        self.repo = repo
        self.user_repo = user_repo
        self.storage = storage

    async def create_package(
        self,
        data: PackageCreate,
        publisher_id: str,
        owner_id: str,
    ) -> Package:
        """Create a new package.

        Args:
            data: Package creation data.
            publisher_id: Publisher ID.
            owner_id: Owner user ID.

        Returns:
            Created package.

        Raises:
            DuplicatePackageError: If package already exists.
        """
        # Check if package exists
        existing = await self.repo.get_by_name(data.name)
        if existing:
            raise DuplicatePackageError(data.name)

        package = Package(
            name=data.name,
            display_name=data.display_name,
            package_type=data.package_type,
            owner_id=owner_id,
            publisher_id=publisher_id,
            visibility=data.visibility,
            description=data.description,
            readme=data.readme,
            homepage=data.homepage,
            repository=data.repository,
            license=data.license,
            keywords=data.keywords,
        )

        created = await self.repo.create(package)

        # Update publisher package count
        await self.user_repo.increment_publisher_packages(publisher_id)

        logger.info(f"Created package: {data.name}")
        return created

    async def get_package(
        self,
        name: str,
        viewer_id: Optional[str] = None,
    ) -> Package:
        """Get a package by name.

        Args:
            name: Package name.
            viewer_id: Optional viewer user ID for visibility check.

        Returns:
            Package.

        Raises:
            PackageNotFoundError: If package not found.
            PermissionDeniedError: If viewer cannot access package.
        """
        package = await self.repo.get_by_name(name)
        if not package:
            raise PackageNotFoundError(name)

        # Check visibility
        if package.visibility == Visibility.PRIVATE:
            if not viewer_id or viewer_id != package.owner_id:
                raise PermissionDeniedError("Cannot access private package")

        return package

    async def list_packages(
        self,
        page: int = 1,
        per_page: int = 20,
        package_type: Optional[PackageType] = None,
        viewer_id: Optional[str] = None,
    ) -> tuple[list[PackageSummary], int]:
        """List packages with pagination.

        Args:
            page: Page number.
            per_page: Items per page.
            package_type: Filter by package type.
            viewer_id: Optional viewer user ID.

        Returns:
            Tuple of (list of package summaries, total count).
        """
        # Only show public and unlisted packages
        visibility = Visibility.PUBLIC if not viewer_id else None

        packages = await self.repo.list_all(
            page=page,
            per_page=per_page,
            package_type=package_type,
            visibility=visibility,
        )

        total = await self.repo.count_all(
            package_type=package_type,
            visibility=visibility,
        )

        summaries = [
            PackageSummary(
                name=p.name,
                display_name=p.display_name,
                package_type=p.package_type,
                visibility=p.visibility,
                description=p.description,
                latest_version=p.latest_version,
                total_downloads=p.total_downloads,
                updated_at=p.updated_at,
            )
            for p in packages
        ]

        return summaries, total

    async def update_package(
        self,
        name: str,
        data: PackageUpdate,
        user_id: str,
    ) -> Package:
        """Update a package.

        Args:
            name: Package name.
            data: Update data.
            user_id: User ID making the update.

        Returns:
            Updated package.

        Raises:
            PackageNotFoundError: If package not found.
            PermissionDeniedError: If user is not owner.
        """
        package = await self.repo.get_by_name(name)
        if not package:
            raise PackageNotFoundError(name)

        if package.owner_id != user_id:
            raise PermissionDeniedError("Only owner can update package")

        update_data = data.model_dump(exclude_unset=True)
        updated = await self.repo.update(name, update_data)
        return updated

    async def delete_package(
        self,
        name: str,
        user_id: str,
        is_admin: bool = False,
    ) -> bool:
        """Delete a package.

        Args:
            name: Package name.
            user_id: User ID making the deletion.
            is_admin: Whether user is admin.

        Returns:
            True if deleted.

        Raises:
            PackageNotFoundError: If package not found.
            PermissionDeniedError: If user is not owner or admin.
        """
        package = await self.repo.get_by_name(name)
        if not package:
            raise PackageNotFoundError(name)

        if not is_admin and package.owner_id != user_id:
            raise PermissionDeniedError("Only owner or admin can delete package")

        # Delete from storage
        versions = await self.repo.get_versions(name)
        for version in versions:
            for loc in version.storage_locations:
                await self.storage.delete(loc.path)

        return await self.repo.delete(name)

    async def upload_version(
        self,
        name: str,
        version: str,
        file: UploadFile,
        changelog: str,
        publisher_id: str,
        toolbox_version: Optional[str] = None,
        python_version: Optional[str] = None,
    ) -> PackageVersion:
        """Upload a new version.

        Args:
            name: Package name.
            version: Version string.
            file: Uploaded file.
            changelog: Version changelog.
            publisher_id: Publisher ID.
            toolbox_version: Required ToolBox version.
            python_version: Required Python version.

        Returns:
            Created version.

        Raises:
            PackageNotFoundError: If package not found.
            PermissionDeniedError: If publisher doesn't own package.
            DuplicateVersionError: If version already exists.
        """
        package = await self.repo.get_by_name(name)
        if not package:
            raise PackageNotFoundError(name)

        if package.publisher_id != publisher_id:
            raise PermissionDeniedError("Only package owner can upload versions")

        # Check if version exists
        existing = await self.repo.get_version(name, version)
        if existing:
            raise DuplicateVersionError(name, version)

        # Save file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tbmod") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            # Upload to storage
            remote_path = f"packages/{name}/{version}/{file.filename}"
            locations = await self.storage.upload(tmp_path, remote_path)

            # Create version
            pkg_version = PackageVersion(
                version=version,
                released_at=datetime.utcnow(),
                changelog=changelog,
                toolbox_version=toolbox_version,
                python_version=python_version,
                storage_locations=locations,
            )

            created = await self.repo.add_version(name, pkg_version)
            logger.info(f"Uploaded version {version} for package {name}")
            return created
        finally:
            tmp_path.unlink(missing_ok=True)

    async def get_download_url(
        self,
        name: str,
        version: str,
        prefer_mirror: bool = False,
        viewer_id: Optional[str] = None,
    ) -> Optional[str]:
        """Get download URL for a version.

        Download permissions based on visibility:
        - PUBLIC: Anyone can download
        - UNLISTED: Only authenticated users can download
        - PRIVATE: Only owner can download

        Args:
            name: Package name.
            version: Version string.
            prefer_mirror: Prefer mirror URL.
            viewer_id: Optional viewer user ID for visibility check.

        Returns:
            Presigned download URL.

        Raises:
            PackageNotFoundError: If package not found.
            VersionNotFoundError: If version not found.
            PermissionDeniedError: If viewer cannot download package.
        """
        package = await self.repo.get_by_name(name)
        if not package:
            raise PackageNotFoundError(name)

        # Check visibility permissions for download
        if package.visibility == Visibility.PRIVATE:
            # Private: Only owner can download
            if not viewer_id or viewer_id != package.owner_id:
                raise PermissionDeniedError("Cannot download private package")
        elif package.visibility == Visibility.UNLISTED:
            # Unlisted: Only authenticated users can download
            if not viewer_id:
                raise PermissionDeniedError(
                    "Authentication required to download unlisted package"
                )
        # PUBLIC: Anyone can download (no check needed)

        pkg_version = await self.repo.get_version(name, version)
        if not pkg_version:
            raise VersionNotFoundError(name, version)

        if not pkg_version.storage_locations:
            return None

        loc = pkg_version.storage_locations[0]
        return await self.storage.get_download_url(
            loc.path,
            expires_in=3600,
            prefer_mirror=prefer_mirror,
        )

    async def increment_downloads(self, name: str, version: str) -> None:
        """Increment download count.

        Args:
            name: Package name.
            version: Version string.
        """
        await self.repo.increment_downloads(name, version)

        # Also increment publisher downloads
        package = await self.repo.get_by_name(name)
        if package:
            await self.user_repo.increment_publisher_downloads(package.publisher_id)

    async def search(
        self,
        query: str,
        package_type: Optional[PackageType] = None,
        page: int = 1,
        per_page: int = 20,
    ) -> list[PackageSummary]:
        """Search packages.

        Args:
            query: Search query.
            package_type: Filter by package type.
            page: Page number.
            per_page: Items per page.

        Returns:
            List of matching package summaries.
        """
        packages = await self.repo.search(query, package_type, page, per_page)
        return [
            PackageSummary(
                name=p.name,
                display_name=p.display_name,
                package_type=p.package_type,
                visibility=p.visibility,
                description=p.description,
                latest_version=p.latest_version,
                total_downloads=p.total_downloads,
                updated_at=p.updated_at,
            )
            for p in packages
        ]

