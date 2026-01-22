"""Data models for TB Registry."""

from registry.models.package import (
    Architecture,
    Dependency,
    Package,
    PackageAsset,
    PackageCreate,
    PackageDetail,
    PackageSummary,
    PackageType,
    PackageUpdate,
    PackageVersion,
    Platform,
    StorageLocation,
    VersionCreate,
    VersionDetail,
    VersionSummary,
    Visibility,
)
from registry.models.user import (
    Publisher,
    PublisherCreate,
    PublisherDetail,
    PublisherSummary,
    User,
    UserSummary,
    VerificationStatus,
)
from registry.models.artifact import (
    Artifact,
    ArtifactBuild,
    ArtifactType,
    ArtifactVersion,
)

__all__ = [
    # Package models
    "Architecture",
    "Dependency",
    "Package",
    "PackageAsset",
    "PackageCreate",
    "PackageDetail",
    "PackageSummary",
    "PackageType",
    "PackageUpdate",
    "PackageVersion",
    "Platform",
    "StorageLocation",
    "VersionCreate",
    "VersionDetail",
    "VersionSummary",
    "Visibility",
    # User models
    "Publisher",
    "PublisherCreate",
    "PublisherDetail",
    "PublisherSummary",
    "User",
    "UserSummary",
    "VerificationStatus",
    # Artifact models
    "Artifact",
    "ArtifactBuild",
    "ArtifactType",
    "ArtifactVersion",
]

