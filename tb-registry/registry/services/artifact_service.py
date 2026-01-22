"""Artifact service for business logic."""

import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from registry.db.repositories.artifact_repo import ArtifactRepository
from registry.db.repositories.user_repo import UserRepository
from registry.exceptions import (
    PackageNotFoundError,
    PermissionDeniedError,
    VersionNotFoundError,
)
from registry.models.artifact import (
    Artifact,
    ArtifactBuild,
    ArtifactType,
    ArtifactVersion,
)
from registry.models.package import Architecture, Platform
from registry.storage.manager import StorageManager

logger = logging.getLogger(__name__)


class ArtifactService:
    """Service for artifact operations.

    Handles business logic for artifact management.

    Attributes:
        repo: Artifact repository.
        user_repo: User repository.
        storage: Storage manager.
    """

    def __init__(
        self,
        repo: ArtifactRepository,
        user_repo: UserRepository,
        storage: StorageManager,
    ) -> None:
        """Initialize the service.

        Args:
            repo: Artifact repository.
            user_repo: User repository.
            storage: Storage manager.
        """
        self.repo = repo
        self.user_repo = user_repo
        self.storage = storage

    async def create_artifact(
        self,
        name: str,
        artifact_type: ArtifactType,
        publisher_id: str,
        description: str = "",
        homepage: Optional[str] = None,
        repository: Optional[str] = None,
    ) -> Artifact:
        """Create a new artifact.

        Args:
            name: Artifact name.
            artifact_type: Type of artifact.
            publisher_id: Publisher ID.
            description: Artifact description.
            homepage: Homepage URL.
            repository: Repository URL.

        Returns:
            Created artifact.
        """
        artifact = Artifact(
            id="",  # Will be generated
            name=name,
            artifact_type=artifact_type,
            publisher_id=publisher_id,
            description=description,
            homepage=homepage,
            repository=repository,
        )

        created = await self.repo.create(artifact)
        logger.info(f"Created artifact: {name}")
        return created

    async def get_artifact(self, name: str) -> Artifact:
        """Get an artifact by name.

        Args:
            name: Artifact name.

        Returns:
            Artifact.

        Raises:
            PackageNotFoundError: If artifact not found.
        """
        artifact = await self.repo.get_by_name(name)
        if not artifact:
            raise PackageNotFoundError(name)
        return artifact

    async def list_artifacts(
        self,
        page: int = 1,
        per_page: int = 20,
        artifact_type: Optional[ArtifactType] = None,
    ) -> list[Artifact]:
        """List artifacts with pagination.

        Args:
            page: Page number.
            per_page: Items per page.
            artifact_type: Filter by artifact type.

        Returns:
            List of artifacts.
        """
        return await self.repo.list_all(
            page=page,
            per_page=per_page,
            artifact_type=artifact_type,
        )

    async def upload_build(
        self,
        artifact_name: str,
        version: str,
        file: UploadFile,
        platform: Platform,
        architecture: Architecture,
        publisher_id: str,
        changelog: str = "",
        installer_type: Optional[str] = None,
        min_os_version: Optional[str] = None,
    ) -> ArtifactVersion:
        """Upload a build for an artifact version.

        Args:
            artifact_name: Artifact name.
            version: Version string.
            file: Uploaded file.
            platform: Target platform.
            architecture: Target architecture.
            publisher_id: Publisher ID.
            changelog: Version changelog.
            installer_type: Type of installer.
            min_os_version: Minimum OS version.

        Returns:
            Created or updated version.
        """
        artifact = await self.repo.get_by_name(artifact_name)
        if not artifact:
            raise PackageNotFoundError(artifact_name)

        if artifact.publisher_id != publisher_id:
            raise PermissionDeniedError("Only artifact owner can upload builds")

        # Save file to temp location
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            # Calculate checksum
            checksum = await self.storage.calculate_checksum(tmp_path)
            size = tmp_path.stat().st_size

            # Upload to storage
            remote_path = f"artifacts/{artifact_name}/{version}/{platform.value}_{architecture.value}/{file.filename}"
            locations = await self.storage.upload(tmp_path, remote_path)

            # Create build
            build = ArtifactBuild(
                platform=platform,
                architecture=architecture,
                filename=file.filename,
                checksum_sha256=checksum,
                size_bytes=size,
                installer_type=installer_type,
                storage_locations=locations,
            )

            # Check if version exists
            existing_version = await self.repo.get_version(artifact.id, version)
            if existing_version:
                # Add build to existing version
                # For now, we'll create a new version entry
                pass

            # Create new version
            artifact_version = ArtifactVersion(
                version=version,
                released_at=datetime.utcnow(),
                changelog=changelog,
                builds=[build],
                min_os_version=min_os_version,
            )

            created = await self.repo.add_version(artifact.id, artifact_version)
            logger.info(f"Uploaded build for {artifact_name}@{version}")
            return created
        finally:
            tmp_path.unlink(missing_ok=True)

    async def get_latest_for_platform(
        self,
        artifact_name: str,
        platform: Platform,
        architecture: Architecture,
    ) -> Optional[tuple[ArtifactVersion, ArtifactBuild]]:
        """Get latest version with build for platform.

        Args:
            artifact_name: Artifact name.
            platform: Target platform.
            architecture: Target architecture.

        Returns:
            Tuple of (version, build) or None.
        """
        return await self.repo.get_latest_for_platform(
            artifact_name,
            platform,
            architecture,
        )

    async def get_download_url(
        self,
        artifact_name: str,
        version: str,
        platform: Platform,
        architecture: Architecture,
    ) -> Optional[str]:
        """Get download URL for a build.

        Args:
            artifact_name: Artifact name.
            version: Version string.
            platform: Target platform.
            architecture: Target architecture.

        Returns:
            Presigned download URL.
        """
        artifact = await self.repo.get_by_name(artifact_name)
        if not artifact:
            raise PackageNotFoundError(artifact_name)

        artifact_version = await self.repo.get_version(artifact.id, version)
        if not artifact_version:
            raise VersionNotFoundError(artifact_name, version)

        # Find matching build
        for build in artifact_version.builds:
            if (
                (build.platform == platform or build.platform == Platform.ALL)
                and (build.architecture == architecture or build.architecture == Architecture.ALL)
            ):
                if build.storage_locations:
                    loc = build.storage_locations[0]
                    return await self.storage.get_download_url(loc.path)

        return None

    async def increment_downloads(
        self,
        artifact_name: str,
        version: str,
        platform: Platform,
        architecture: Architecture,
    ) -> None:
        """Increment download count for a build.

        Args:
            artifact_name: Artifact name.
            version: Version string.
            platform: Target platform.
            architecture: Target architecture.
        """
        artifact = await self.repo.get_by_name(artifact_name)
        if artifact:
            await self.repo.increment_downloads(
                artifact.id,
                version,
                platform,
                architecture,
            )

