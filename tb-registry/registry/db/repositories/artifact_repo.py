"""Artifact repository for database operations."""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional

from registry.db.database import Database
from registry.models.artifact import (
    Artifact,
    ArtifactBuild,
    ArtifactType,
    ArtifactVersion,
)
from registry.models.package import Architecture, Platform, StorageLocation

logger = logging.getLogger(__name__)


class ArtifactRepository:
    """Repository for artifact CRUD operations.

    Attributes:
        db: Database instance.
    """

    def __init__(self, db: Database) -> None:
        """Initialize the repository.

        Args:
            db: Database instance.
        """
        self.db = db

    async def create(self, artifact: Artifact) -> Artifact:
        """Create a new artifact.

        Args:
            artifact: Artifact to create.

        Returns:
            Created artifact.
        """
        if not artifact.id:
            artifact.id = str(uuid.uuid4())

        query = """
        INSERT INTO artifacts (
            id, name, artifact_type, publisher_id, description,
            homepage, repository, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        await self.db.execute(
            query,
            (
                artifact.id,
                artifact.name,
                artifact.artifact_type.value,
                artifact.publisher_id,
                artifact.description,
                artifact.homepage,
                artifact.repository,
                artifact.created_at.isoformat(),
                artifact.updated_at.isoformat(),
            ),
        )
        await self.db.commit()
        logger.info(f"Created artifact: {artifact.name}")
        return artifact

    async def get_by_name(self, name: str) -> Optional[Artifact]:
        """Get an artifact by name.

        Args:
            name: Artifact name.

        Returns:
            Artifact or None if not found.
        """
        row = await self.db.fetch_one(
            "SELECT * FROM artifacts WHERE name = ?",
            (name,),
        )

        if not row:
            return None

        return self._row_to_artifact(row)

    async def get_by_id(self, artifact_id: str) -> Optional[Artifact]:
        """Get an artifact by ID.

        Args:
            artifact_id: Artifact ID.

        Returns:
            Artifact or None if not found.
        """
        row = await self.db.fetch_one(
            "SELECT * FROM artifacts WHERE id = ?",
            (artifact_id,),
        )

        if not row:
            return None

        return self._row_to_artifact(row)

    def _row_to_artifact(self, row) -> Artifact:
        """Convert a database row to an Artifact object.

        Args:
            row: Database row.

        Returns:
            Artifact object.
        """
        return Artifact(
            id=row["id"],
            name=row["name"],
            artifact_type=ArtifactType(row["artifact_type"]),
            publisher_id=row["publisher_id"],
            description=row["description"],
            homepage=row["homepage"],
            repository=row["repository"],
            latest_version=row["latest_version"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    async def list_all(
        self,
        page: int = 1,
        per_page: int = 20,
        artifact_type: Optional[ArtifactType] = None,
    ) -> list[Artifact]:
        """List all artifacts with pagination.

        Args:
            page: Page number.
            per_page: Items per page.
            artifact_type: Filter by artifact type.

        Returns:
            List of artifacts.
        """
        query = "SELECT * FROM artifacts WHERE 1=1"
        params: list = []

        if artifact_type:
            query += " AND artifact_type = ?"
            params.append(artifact_type.value)

        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([per_page, (page - 1) * per_page])

        rows = await self.db.fetch_all(query, tuple(params))
        return [self._row_to_artifact(row) for row in rows]

    async def add_version(
        self,
        artifact_id: str,
        version: ArtifactVersion,
    ) -> ArtifactVersion:
        """Add a version to an artifact.

        Args:
            artifact_id: Artifact ID.
            version: Version to add.

        Returns:
            Created version.
        """
        query = """
        INSERT INTO artifact_versions (
            artifact_id, version, released_at, changelog, min_os_version
        ) VALUES (?, ?, ?, ?, ?)
        """
        cursor = await self.db.execute(
            query,
            (
                artifact_id,
                version.version,
                version.released_at.isoformat(),
                version.changelog,
                version.min_os_version,
            ),
        )
        version_id = cursor.lastrowid
        await self.db.commit()

        # Add builds
        for build in version.builds:
            await self._add_build(version_id, build)

        # Update latest version
        await self.db.execute(
            "UPDATE artifacts SET latest_version = ?, updated_at = ? WHERE id = ?",
            (version.version, datetime.utcnow().isoformat(), artifact_id),
        )
        await self.db.commit()

        logger.info(f"Added version {version.version} to artifact {artifact_id}")
        return version

    async def _add_build(self, version_id: int, build: ArtifactBuild) -> None:
        """Add a build to a version.

        Args:
            version_id: Version ID.
            build: Build to add.
        """
        await self.db.execute(
            """
            INSERT INTO artifact_builds (
                version_id, platform, architecture, filename,
                checksum_sha256, size_bytes, installer_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                build.platform.value,
                build.architecture.value,
                build.filename,
                build.checksum_sha256,
                build.size_bytes,
                build.installer_type,
            ),
        )
        await self.db.commit()

    async def get_version(
        self,
        artifact_id: str,
        version: str,
    ) -> Optional[ArtifactVersion]:
        """Get a specific version of an artifact.

        Args:
            artifact_id: Artifact ID.
            version: Version string.

        Returns:
            ArtifactVersion or None.
        """
        row = await self.db.fetch_one(
            "SELECT * FROM artifact_versions WHERE artifact_id = ? AND version = ?",
            (artifact_id, version),
        )

        if not row:
            return None

        return await self._row_to_version(row)

    async def _row_to_version(self, row) -> ArtifactVersion:
        """Convert a database row to an ArtifactVersion object.

        Args:
            row: Database row.

        Returns:
            ArtifactVersion object.
        """
        # Get builds
        builds_rows = await self.db.fetch_all(
            "SELECT * FROM artifact_builds WHERE version_id = ?",
            (row["id"],),
        )

        builds = [
            ArtifactBuild(
                platform=Platform(b["platform"]),
                architecture=Architecture(b["architecture"]),
                filename=b["filename"],
                checksum_sha256=b["checksum_sha256"],
                size_bytes=b["size_bytes"],
                installer_type=b["installer_type"],
                downloads=b["downloads"],
            )
            for b in builds_rows
        ]

        return ArtifactVersion(
            version=row["version"],
            released_at=datetime.fromisoformat(row["released_at"]),
            changelog=row["changelog"],
            builds=builds,
            min_os_version=row["min_os_version"],
        )

    async def get_latest_for_platform(
        self,
        artifact_name: str,
        platform: Platform,
        architecture: Architecture,
    ) -> Optional[tuple[ArtifactVersion, ArtifactBuild]]:
        """Get the latest version with a build for the specified platform.

        Args:
            artifact_name: Artifact name.
            platform: Target platform.
            architecture: Target architecture.

        Returns:
            Tuple of (version, build) or None.
        """
        artifact = await self.get_by_name(artifact_name)
        if not artifact or not artifact.latest_version:
            return None

        version = await self.get_version(artifact.id, artifact.latest_version)
        if not version:
            return None

        # Find matching build
        for build in version.builds:
            if (
                (build.platform == platform or build.platform == Platform.ALL)
                and (build.architecture == architecture or build.architecture == Architecture.ALL)
            ):
                return (version, build)

        return None

    async def increment_downloads(
        self,
        artifact_id: str,
        version: str,
        platform: Platform,
        architecture: Architecture,
    ) -> None:
        """Increment download count for a build.

        Args:
            artifact_id: Artifact ID.
            version: Version string.
            platform: Platform.
            architecture: Architecture.
        """
        await self.db.execute(
            """
            UPDATE artifact_builds SET downloads = downloads + 1
            WHERE version_id = (
                SELECT id FROM artifact_versions
                WHERE artifact_id = ? AND version = ?
            ) AND platform = ? AND architecture = ?
            """,
            (artifact_id, version, platform.value, architecture.value),
        )
        await self.db.commit()

