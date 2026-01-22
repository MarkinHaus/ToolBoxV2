"""User and Publisher data models."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    """Publisher verification status."""

    UNVERIFIED = "unverified"
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    SUSPENDED = "suspended"


@dataclass
class Publisher:
    """Publisher information.

    Attributes:
        id: Unique publisher ID.
        clerk_user_id: Associated Clerk user ID.
        name: Publisher display name.
        slug: URL-friendly slug.
        email: Contact email.
        website: Publisher website.
        github: GitHub username.
        status: Verification status.
        verified_at: Verification timestamp.
        verified_by: Admin who verified.
        verification_notes: Notes from verification.
        can_publish_public: Can publish public packages.
        can_publish_artifacts: Can publish artifacts.
        max_package_size_mb: Maximum package size in MB.
        packages_count: Number of packages.
        total_downloads: Total downloads across all packages.
        created_at: Creation timestamp.
    """

    id: str
    clerk_user_id: str
    name: str
    slug: str
    email: str
    website: Optional[str] = None
    github: Optional[str] = None
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    verified_at: Optional[datetime] = None
    verified_by: Optional[str] = None
    verification_notes: Optional[str] = None
    can_publish_public: bool = False
    can_publish_artifacts: bool = False
    max_package_size_mb: int = 100
    packages_count: int = 0
    total_downloads: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class User:
    """User information.

    Attributes:
        clerk_user_id: Clerk user ID.
        email: User email.
        username: Username.
        publisher_id: Associated publisher ID.
        is_admin: Whether user is admin.
        last_login: Last login timestamp.
        created_at: Creation timestamp.
    """

    clerk_user_id: str
    email: str
    username: str
    publisher_id: Optional[str] = None
    is_admin: bool = False
    last_login: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


# Pydantic Models for API


class PublisherCreate(BaseModel):
    """Request model for creating a publisher.

    Attributes:
        name: Publisher display name.
        slug: URL-friendly slug.
        email: Contact email.
        website: Publisher website.
        github: GitHub username.
    """

    name: str = Field(..., min_length=1, max_length=100)
    slug: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-z0-9_-]+$")
    email: str = Field(..., max_length=255)
    website: Optional[str] = None
    github: Optional[str] = None


class PublisherSummary(BaseModel):
    """Summary model for publisher listings.

    Attributes:
        id: Unique publisher ID.
        name: Publisher display name.
        slug: URL-friendly slug.
        status: Verification status.
        packages_count: Number of packages.
        total_downloads: Total downloads.
    """

    id: str
    name: str
    slug: str
    status: VerificationStatus
    packages_count: int
    total_downloads: int


class PublisherDetail(BaseModel):
    """Detailed model for single publisher view.

    Attributes:
        id: Unique publisher ID.
        name: Publisher display name.
        slug: URL-friendly slug.
        email: Contact email.
        website: Publisher website.
        github: GitHub username.
        status: Verification status.
        verified_at: Verification timestamp.
        can_publish_public: Can publish public packages.
        can_publish_artifacts: Can publish artifacts.
        packages_count: Number of packages.
        total_downloads: Total downloads.
        created_at: Creation timestamp.
    """

    id: str
    name: str
    slug: str
    email: str
    website: Optional[str]
    github: Optional[str]
    status: VerificationStatus
    verified_at: Optional[datetime]
    can_publish_public: bool
    can_publish_artifacts: bool
    packages_count: int
    total_downloads: int
    created_at: datetime


class UserSummary(BaseModel):
    """Summary model for user information.

    Attributes:
        clerk_user_id: Clerk user ID.
        email: User email.
        username: Username.
        is_admin: Whether user is admin.
        publisher_id: Associated publisher ID.
    """

    clerk_user_id: str
    email: str
    username: str
    is_admin: bool
    publisher_id: Optional[str]

