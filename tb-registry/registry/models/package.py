"""Package-related data models."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Visibility(str, Enum):
    """Package visibility levels."""

    PUBLIC = "public"
    PRIVATE = "private"
    UNLISTED = "unlisted"


class PackageType(str, Enum):
    """Types of packages in the registry."""

    MOD = "mod"
    ARTIFACT = "artifact"
    LIBRARY = "library"
    THEME = "theme"
    PLUGIN = "plugin"


class Platform(str, Enum):
    """Supported platforms."""

    ALL = "all"
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    ANDROID = "android"
    IOS = "ios"


class Architecture(str, Enum):
    """Supported architectures."""

    ALL = "all"
    X64 = "x64"
    X86 = "x86"
    ARM64 = "arm64"
    ARM32 = "arm32"


@dataclass
class Dependency:
    """Package dependency specification.

    Attributes:
        name: Name of the dependency package.
        version_spec: Version specifier (e.g., ">=1.0.0,<2.0.0").
        optional: Whether this dependency is optional.
        features: List of required features from the dependency.
    """

    name: str
    version_spec: str
    optional: bool = False
    features: list[str] = field(default_factory=list)


@dataclass
class StorageLocation:
    """Storage location for package files.

    Attributes:
        backend: Storage backend name (e.g., "minio-primary").
        bucket: Bucket name.
        path: Path within the bucket.
        checksum_sha256: SHA256 checksum of the file.
        size_bytes: File size in bytes.
        uploaded_at: Upload timestamp.
    """

    backend: str
    bucket: str
    path: str
    checksum_sha256: str
    size_bytes: int
    uploaded_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PackageAsset:
    """Platform-specific package asset.

    Attributes:
        filename: Original filename.
        platform: Target platform.
        architecture: Target architecture.
        checksum_sha256: SHA256 checksum.
        size_bytes: File size in bytes.
        storage_locations: List of storage locations.
    """

    filename: str
    platform: Platform
    architecture: Architecture
    checksum_sha256: str
    size_bytes: int
    storage_locations: list[StorageLocation] = field(default_factory=list)


@dataclass
class PackageVersion:
    """Package version information.

    Attributes:
        version: Semantic version string.
        released_at: Release timestamp.
        changelog: Version changelog/release notes.
        dependencies: List of dependencies.
        toolbox_version: Required ToolBox version.
        python_version: Required Python version.
        assets: Platform-specific assets.
        storage_locations: Storage locations for main package file.
        downloads: Download count.
        yanked: Whether this version is yanked.
        yank_reason: Reason for yanking.
    """

    version: str
    released_at: datetime = field(default_factory=datetime.utcnow)
    changelog: str = ""
    dependencies: list[Dependency] = field(default_factory=list)
    toolbox_version: Optional[str] = None
    python_version: Optional[str] = None
    assets: list[PackageAsset] = field(default_factory=list)
    storage_locations: list[StorageLocation] = field(default_factory=list)
    downloads: int = 0
    yanked: bool = False
    yank_reason: Optional[str] = None


@dataclass
class Package:
    """Complete package information.

    Attributes:
        name: Unique package name.
        display_name: Human-readable display name.
        package_type: Type of package.
        owner_id: Owner's user ID.
        publisher_id: Publisher's ID.
        visibility: Package visibility.
        description: Short description.
        readme: Full README content.
        homepage: Homepage URL.
        repository: Repository URL.
        license: License identifier.
        keywords: Search keywords.
        versions: List of versions.
        latest_version: Latest version string.
        total_downloads: Total download count.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
    """

    name: str
    display_name: str
    package_type: PackageType
    owner_id: str
    publisher_id: str
    visibility: Visibility = Visibility.PUBLIC
    description: str = ""
    readme: str = ""
    homepage: Optional[str] = None
    repository: Optional[str] = None
    license: Optional[str] = None
    keywords: list[str] = field(default_factory=list)
    versions: list[PackageVersion] = field(default_factory=list)
    latest_version: Optional[str] = None
    total_downloads: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


# Pydantic Models for API


class PackageCreate(BaseModel):
    """Request model for creating a package.

    Attributes:
        name: Unique package name.
        display_name: Human-readable display name.
        package_type: Type of package.
        visibility: Package visibility.
        description: Short description.
        readme: Full README content.
        homepage: Homepage URL.
        repository: Repository URL.
        license: License identifier.
        keywords: Search keywords.
    """

    name: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-z0-9_-]+$")
    display_name: str = Field(..., min_length=1, max_length=200)
    package_type: PackageType
    visibility: Visibility = Visibility.PUBLIC
    description: str = Field(default="", max_length=500)
    readme: str = ""
    homepage: Optional[str] = None
    repository: Optional[str] = None
    license: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)


class PackageUpdate(BaseModel):
    """Request model for updating a package.

    Attributes:
        display_name: Human-readable display name.
        visibility: Package visibility.
        description: Short description.
        readme: Full README content.
        homepage: Homepage URL.
        repository: Repository URL.
        license: License identifier.
        keywords: Search keywords.
    """

    display_name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    visibility: Optional[Visibility] = None
    description: Optional[str] = Field(default=None, max_length=500)
    readme: Optional[str] = None
    homepage: Optional[str] = None
    repository: Optional[str] = None
    license: Optional[str] = None
    keywords: Optional[list[str]] = None


class PackageSummary(BaseModel):
    """Summary model for package listings.

    Attributes:
        name: Unique package name.
        display_name: Human-readable display name.
        package_type: Type of package.
        visibility: Package visibility.
        description: Short description.
        latest_version: Latest version string.
        total_downloads: Total download count.
        updated_at: Last update timestamp.
    """

    name: str
    display_name: str
    package_type: PackageType
    visibility: Visibility
    description: str
    latest_version: Optional[str]
    total_downloads: int
    updated_at: datetime


class PackageDetail(BaseModel):
    """Detailed model for single package view.

    Attributes:
        name: Unique package name.
        display_name: Human-readable display name.
        package_type: Type of package.
        owner_id: Owner's user ID.
        publisher_id: Publisher's ID.
        visibility: Package visibility.
        description: Short description.
        readme: Full README content.
        homepage: Homepage URL.
        repository: Repository URL.
        license: License identifier.
        keywords: Search keywords.
        latest_version: Latest version string.
        total_downloads: Total download count.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
    """

    name: str
    display_name: str
    package_type: PackageType
    owner_id: str
    publisher_id: str
    visibility: Visibility
    description: str
    readme: str
    homepage: Optional[str]
    repository: Optional[str]
    license: Optional[str]
    keywords: list[str]
    latest_version: Optional[str]
    total_downloads: int
    created_at: datetime
    updated_at: datetime


class VersionCreate(BaseModel):
    """Request model for creating a version.

    Attributes:
        version: Semantic version string.
        changelog: Version changelog/release notes.
        toolbox_version: Required ToolBox version.
        python_version: Required Python version.
        dependencies: List of dependencies.
    """

    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$")
    changelog: str = ""
    toolbox_version: Optional[str] = None
    python_version: Optional[str] = None
    dependencies: list[dict] = Field(default_factory=list)


class VersionSummary(BaseModel):
    """Summary model for version listings.

    Attributes:
        version: Semantic version string.
        released_at: Release timestamp.
        downloads: Download count.
        yanked: Whether this version is yanked.
    """

    version: str
    released_at: datetime
    downloads: int
    yanked: bool


class VersionDetail(BaseModel):
    """Detailed model for single version view.

    Attributes:
        version: Semantic version string.
        released_at: Release timestamp.
        changelog: Version changelog/release notes.
        toolbox_version: Required ToolBox version.
        python_version: Required Python version.
        downloads: Download count.
        yanked: Whether this version is yanked.
        yank_reason: Reason for yanking.
        checksum_sha256: SHA256 checksum.
        size_bytes: File size in bytes.
    """

    version: str
    released_at: datetime
    changelog: str
    toolbox_version: Optional[str]
    python_version: Optional[str]
    downloads: int
    yanked: bool
    yank_reason: Optional[str]
    checksum_sha256: Optional[str]
    size_bytes: Optional[int]

