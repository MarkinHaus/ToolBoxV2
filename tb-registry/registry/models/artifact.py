"""Artifact data models."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from registry.models.package import Architecture, Platform, StorageLocation


class ArtifactType(str, Enum):
    """Types of artifacts."""

    TAURI_APP = "tauri_app"
    CLI_EXECUTABLE = "cli_executable"
    BROWSER_EXTENSION = "browser_extension"
    MOBILE_APP = "mobile_app"
    LIBRARY = "library"


@dataclass
class ArtifactBuild:
    """Platform-specific artifact build.

    Attributes:
        platform: Target platform.
        architecture: Target architecture.
        filename: Build filename.
        checksum_sha256: SHA256 checksum.
        size_bytes: File size in bytes.
        installer_type: Type of installer (msi, exe, dmg, etc.).
        storage_locations: Storage locations.
        downloads: Download count.
    """

    platform: Platform
    architecture: Architecture
    filename: str
    checksum_sha256: str
    size_bytes: int
    installer_type: Optional[str] = None
    storage_locations: list[StorageLocation] = field(default_factory=list)
    downloads: int = 0


@dataclass
class ArtifactVersion:
    """Artifact version information.

    Attributes:
        version: Semantic version string.
        released_at: Release timestamp.
        changelog: Version changelog.
        builds: Platform-specific builds.
        min_os_version: Minimum OS version required.
    """

    version: str
    released_at: datetime = field(default_factory=datetime.utcnow)
    changelog: str = ""
    builds: list[ArtifactBuild] = field(default_factory=list)
    min_os_version: Optional[str] = None


@dataclass
class Artifact:
    """Complete artifact information.

    Attributes:
        id: Unique artifact ID.
        name: Artifact name.
        artifact_type: Type of artifact.
        publisher_id: Publisher's ID.
        description: Short description.
        homepage: Homepage URL.
        repository: Repository URL.
        versions: List of versions.
        latest_version: Latest version string.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
    """

    id: str
    name: str
    artifact_type: ArtifactType
    publisher_id: str
    description: str = ""
    homepage: Optional[str] = None
    repository: Optional[str] = None
    versions: list[ArtifactVersion] = field(default_factory=list)
    latest_version: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

