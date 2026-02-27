"""User and Publisher repository for database operations."""

import logging
import uuid
from datetime import datetime
from typing import Optional

from registry.db.database import Database
from registry.models.user import Publisher, User, VerificationStatus

logger = logging.getLogger(__name__)


class UserRepository:
    """Repository for user and publisher CRUD operations.

    Attributes:
        db: Database instance.
    """

    def __init__(self, db: Database) -> None:
        """Initialize the repository.

        Args:
            db: Database instance.
        """
        self.db = db

    async def create_user(self, user: User) -> User:
        """Create a new user.

        Args:
            user: User to create.

        Returns:
            Created user.
        """
        query = """
        INSERT INTO users (
            cloudm_user_id, email, username, publisher_id,
            is_admin, last_login, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        await self.db.execute(
            query,
            (
                user.cloudm_user_id,
                user.email,
                user.username,
                user.publisher_id,
                user.is_admin,
                user.last_login.isoformat() if user.last_login else None,
                user.created_at.isoformat(),
            ),
        )
        await self.db.commit()
        logger.info(f"Created user: {user.cloudm_user_id}")
        return user

    # Alias for backward compatibility during migration
    async def create(self, user: User) -> User:
        """Alias for create_user for backward compatibility."""
        return await self.create_user(user)

    async def get_by_cloudm_id(self, cloudm_user_id: str) -> Optional[User]:
        """Get a user by CloudM.Auth user ID.

        Args:
            cloudm_user_id: CloudM.Auth user ID.

        Returns:
            User or None if not found.
        """
        row = await self.db.fetch_one(
            "SELECT * FROM users WHERE cloudm_user_id = ?",
            (cloudm_user_id,),
        )

        if not row:
            return None

        return User(
            cloudm_user_id=row["cloudm_user_id"],
            email=row["email"],
            username=row["username"],
            publisher_id=row["publisher_id"],
            is_admin=bool(row["is_admin"]),
            last_login=datetime.fromisoformat(row["last_login"]) if row["last_login"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    async def get_by_clerk_id(self, cloudm_user_id: str) -> Optional[User]:
        """Get a user by Clerk user ID.

        DEPRECATED: Use get_by_cloudm_id instead.
        Kept for migration compatibility.

        Args:
            cloudm_user_id: Clerk user ID.

        Returns:
            User or None if not found.
        """
        row = await self.db.fetch_one(
            "SELECT * FROM users WHERE cloudm_user_id = ?",
            (cloudm_user_id,),
        )

        if not row:
            return None

        return User(
            cloudm_user_id=row.get("cloudm_user_id") or row["cloudm_user_id"],
            email=row["email"],
            username=row["username"],
            publisher_id=row["publisher_id"],
            is_admin=bool(row["is_admin"]),
            last_login=datetime.fromisoformat(row["last_login"]) if row["last_login"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    async def update_last_login(self, cloudm_user_id: str) -> None:
        """Update user's last login timestamp.

        Args:
            cloudm_user_id: Clerk user ID.
        """
        await self.db.execute(
            "UPDATE users SET last_login = ? WHERE cloudm_user_id = ?",
            (datetime.utcnow().isoformat(), cloudm_user_id),
        )
        await self.db.commit()

    async def create_publisher(self, publisher: Publisher) -> Publisher:
        """Create a new publisher.

        Args:
            publisher: Publisher to create.

        Returns:
            Created publisher.
        """
        if not publisher.id:
            publisher.id = str(uuid.uuid4())

        query = """
        INSERT INTO publishers (
            id, cloudm_user_id, name, slug, email, website, github,
            status, can_publish_public, can_publish_artifacts,
            max_package_size_mb, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        await self.db.execute(
            query,
            (
                publisher.id,
                publisher.cloudm_user_id,
                publisher.name,
                publisher.slug,
                publisher.email,
                publisher.website,
                publisher.github,
                publisher.status.value,
                publisher.can_publish_public,
                publisher.can_publish_artifacts,
                publisher.max_package_size_mb,
                publisher.created_at.isoformat(),
            ),
        )
        await self.db.commit()

        # Link user to publisher
        await self.db.execute(
            "UPDATE users SET publisher_id = ? WHERE cloudm_user_id = ?",
            (publisher.id, publisher.cloudm_user_id),
        )
        await self.db.commit()

        logger.info(f"Created publisher: {publisher.slug}")
        return publisher

    async def get_publisher(self, publisher_id: str) -> Optional[Publisher]:
        """Get a publisher by ID.

        Args:
            publisher_id: Publisher ID.

        Returns:
            Publisher or None if not found.
        """
        row = await self.db.fetch_one(
            "SELECT * FROM publishers WHERE id = ?",
            (publisher_id,),
        )

        if not row:
            return None

        return self._row_to_publisher(row)

    def _row_to_publisher(self, row) -> Publisher:
        """Convert a database row to a Publisher object.

        Args:
            row: Database row.

        Returns:
            Publisher object.
        """
        return Publisher(
            id=row["id"],
            cloudm_user_id=row.get("cloudm_user_id") or row["cloudm_user_id"],
            name=row["name"],
            slug=row["slug"],
            email=row["email"],
            website=row["website"],
            github=row["github"],
            status=VerificationStatus(row["status"]),
            verified_at=datetime.fromisoformat(row["verified_at"]) if row["verified_at"] else None,
            verified_by=row["verified_by"],
            verification_notes=row["verification_notes"],
            can_publish_public=bool(row["can_publish_public"]),
            can_publish_artifacts=bool(row["can_publish_artifacts"]),
            max_package_size_mb=row["max_package_size_mb"],
            packages_count=row["packages_count"],
            total_downloads=row["total_downloads"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    async def get_publisher_by_slug(self, slug: str) -> Optional[Publisher]:
        """Get a publisher by slug.

        Args:
            slug: Publisher slug.

        Returns:
            Publisher or None if not found.
        """
        row = await self.db.fetch_one(
            "SELECT * FROM publishers WHERE slug = ?",
            (slug,),
        )

        if not row:
            return None

        return self._row_to_publisher(row)

    async def get_publisher_by_clerk_id(self, cloudm_user_id: str) -> Optional[Publisher]:
        """Get a publisher by Clerk user ID.

        Args:
            cloudm_user_id: Clerk user ID.

        Returns:
            Publisher or None if not found.
        """
        row = await self.db.fetch_one(
            "SELECT * FROM publishers WHERE cloudm_user_id = ?",
            (cloudm_user_id,),
        )

        if not row:
            return None

        return self._row_to_publisher(row)

    async def update_publisher_status(
        self,
        publisher_id: str,
        status: VerificationStatus,
        verified_by: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Update publisher verification status.

        Args:
            publisher_id: Publisher ID.
            status: New verification status.
            verified_by: Admin who verified.
            notes: Verification notes.

        Returns:
            True if updated, False if not found.
        """
        verified_at = datetime.utcnow().isoformat() if status == VerificationStatus.VERIFIED else None
        can_publish = status == VerificationStatus.VERIFIED

        result = await self.db.execute(
            """
            UPDATE publishers SET
                status = ?,
                verified_at = ?,
                verified_by = ?,
                verification_notes = ?,
                can_publish_public = ?
            WHERE id = ?
            """,
            (status.value, verified_at, verified_by, notes, can_publish, publisher_id),
        )
        await self.db.commit()

        return result.rowcount > 0

    async def list_publishers(
        self,
        status: Optional[VerificationStatus] = None,
        page: int = 1,
        per_page: int = 20,
    ) -> list[Publisher]:
        """List publishers with optional status filter.

        Args:
            status: Filter by verification status.
            page: Page number.
            per_page: Items per page.

        Returns:
            List of publishers.
        """
        query = "SELECT * FROM publishers WHERE 1=1"
        params: list = []

        if status:
            query += " AND status = ?"
            params.append(status.value)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([per_page, (page - 1) * per_page])

        rows = await self.db.fetch_all(query, tuple(params))
        return [self._row_to_publisher(row) for row in rows]

    async def increment_publisher_packages(self, publisher_id: str) -> None:
        """Increment publisher's package count.

        Args:
            publisher_id: Publisher ID.
        """
        await self.db.execute(
            "UPDATE publishers SET packages_count = packages_count + 1 WHERE id = ?",
            (publisher_id,),
        )
        await self.db.commit()

    async def increment_publisher_downloads(self, publisher_id: str) -> None:
        """Increment publisher's total downloads.

        Args:
            publisher_id: Publisher ID.
        """
        await self.db.execute(
            "UPDATE publishers SET total_downloads = total_downloads + 1 WHERE id = ?",
            (publisher_id,),
        )
        await self.db.commit()


