"""Package repository for database operations."""

import json
import logging
from datetime import datetime
from typing import Optional

from registry.db.database import Database
from registry.models.package import (
    Dependency,
    Package,
    PackageType,
    PackageVersion,
    StorageLocation,
    Visibility,
)

logger = logging.getLogger(__name__)


class PackageRepository:
    """Repository for package CRUD operations.

    Attributes:
        db: Database instance.
    """

    def __init__(self, db: Database) -> None:
        """Initialize the repository.

        Args:
            db: Database instance.
        """
        self.db = db

    async def create(self, package: Package) -> Package:
        """Create a new package.

        Args:
            package: Package to create.

        Returns:
            Created package.
        """
        query = """
        INSERT INTO packages (
            name, display_name, package_type, owner_id, publisher_id,
            visibility, description, readme, homepage, repository,
            license, keywords, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        await self.db.execute(
            query,
            (
                package.name,
                package.display_name,
                package.package_type.value,
                package.owner_id,
                package.publisher_id,
                package.visibility.value,
                package.description,
                package.readme,
                package.homepage,
                package.repository,
                package.license,
                json.dumps(package.keywords),
                package.created_at.isoformat(),
                package.updated_at.isoformat(),
            ),
        )
        await self.db.commit()

        # Update FTS index
        await self._update_fts(package.name)

        logger.info(f"Created package: {package.name}")
        return package

    async def _update_fts(self, package_name: str) -> None:
        """Update full-text search index for a package.

        Args:
            package_name: Name of the package to index.
        """
        # Delete existing FTS entry
        await self.db.execute(
            "DELETE FROM packages_fts WHERE name = ?",
            (package_name,),
        )

        # Get package data
        row = await self.db.fetch_one(
            "SELECT name, display_name, description, keywords FROM packages WHERE name = ?",
            (package_name,),
        )

        if row:
            await self.db.execute(
                "INSERT INTO packages_fts (name, display_name, description, keywords) VALUES (?, ?, ?, ?)",
                (row["name"], row["display_name"], row["description"], row["keywords"]),
            )
            await self.db.commit()

    async def get_by_name(self, name: str) -> Optional[Package]:
        """Get a package by name.

        Args:
            name: Package name.

        Returns:
            Package or None if not found.
        """
        row = await self.db.fetch_one(
            "SELECT * FROM packages WHERE name = ?",
            (name,),
        )

        if not row:
            return None

        return self._row_to_package(row)

    def _row_to_package(self, row) -> Package:
        """Convert a database row to a Package object.

        Args:
            row: Database row.

        Returns:
            Package object.
        """
        return Package(
            name=row["name"],
            display_name=row["display_name"],
            package_type=PackageType(row["package_type"]),
            owner_id=row["owner_id"],
            publisher_id=row["publisher_id"],
            visibility=Visibility(row["visibility"]),
            description=row["description"],
            readme=row["readme"],
            homepage=row["homepage"],
            repository=row["repository"],
            license=row["license"],
            keywords=json.loads(row["keywords"]) if row["keywords"] else [],
            latest_version=row["latest_version"],
            total_downloads=row["total_downloads"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    async def list_all(
        self,
        page: int = 1,
        per_page: int = 20,
        package_type: Optional[PackageType] = None,
        visibility: Optional[Visibility] = None,
    ) -> list[Package]:
        """List all packages with pagination and filters.

        Args:
            page: Page number (1-indexed).
            per_page: Items per page.
            package_type: Filter by package type.
            visibility: Filter by visibility.

        Returns:
            List of packages.
        """
        query = "SELECT * FROM packages WHERE 1=1"
        params: list = []

        if package_type:
            query += " AND package_type = ?"
            params.append(package_type.value)

        if visibility:
            query += " AND visibility = ?"
            params.append(visibility.value)

        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([per_page, (page - 1) * per_page])

        rows = await self.db.fetch_all(query, tuple(params))
        return [self._row_to_package(row) for row in rows]

    async def count_all(
        self,
        package_type: Optional[PackageType] = None,
        visibility: Optional[Visibility] = None,
    ) -> int:
        """Count all packages with filters.

        Args:
            package_type: Filter by package type.
            visibility: Filter by visibility.

        Returns:
            Total count of packages matching filters.
        """
        query = "SELECT COUNT(*) FROM packages WHERE 1=1"
        params: list = []

        if package_type:
            query += " AND package_type = ?"
            params.append(package_type.value)

        if visibility:
            query += " AND visibility = ?"
            params.append(visibility.value)

        row = await self.db.fetch_one(query, tuple(params))
        return row[0] if row else 0

    async def update(self, name: str, data: dict) -> Optional[Package]:
        """Update a package.

        Args:
            name: Package name.
            data: Fields to update.

        Returns:
            Updated package or None.
        """
        if not data:
            return await self.get_by_name(name)

        # Build update query
        set_clauses = []
        params = []
        for key, value in data.items():
            if key == "keywords":
                value = json.dumps(value)
            set_clauses.append(f"{key} = ?")
            params.append(value)

        set_clauses.append("updated_at = ?")
        params.append(datetime.utcnow().isoformat())
        params.append(name)

        query = f"UPDATE packages SET {', '.join(set_clauses)} WHERE name = ?"
        await self.db.execute(query, tuple(params))
        await self.db.commit()

        # Update FTS index
        await self._update_fts(name)

        return await self.get_by_name(name)

    async def delete(self, name: str) -> bool:
        """Delete a package.

        Args:
            name: Package name.

        Returns:
            True if deleted, False if not found.
        """
        result = await self.db.execute(
            "DELETE FROM packages WHERE name = ?",
            (name,),
        )
        await self.db.commit()

        # Remove from FTS
        await self.db.execute(
            "DELETE FROM packages_fts WHERE name = ?",
            (name,),
        )
        await self.db.commit()

        return result.rowcount > 0

    async def add_version(
        self,
        package_name: str,
        version: PackageVersion,
    ) -> PackageVersion:
        """Add a version to a package.

        Args:
            package_name: Package name.
            version: Version to add.

        Returns:
            Created version.
        """
        query = """
        INSERT INTO package_versions (
            package_name, version, released_at, changelog,
            dependencies, toolbox_version, python_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        cursor = await self.db.execute(
            query,
            (
                package_name,
                version.version,
                version.released_at.isoformat(),
                version.changelog,
                json.dumps([
                    {
                        "name": d.name,
                        "version_spec": d.version_spec,
                        "optional": d.optional,
                        "features": d.features,
                    }
                    for d in version.dependencies
                ]),
                version.toolbox_version,
                version.python_version,
            ),
        )
        version_id = cursor.lastrowid
        await self.db.commit()

        # Add storage locations
        for loc in version.storage_locations:
            await self.db.execute(
                """
                INSERT INTO storage_locations (
                    version_id, backend, bucket, path,
                    checksum_sha256, size_bytes, uploaded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    loc.backend,
                    loc.bucket,
                    loc.path,
                    loc.checksum_sha256,
                    loc.size_bytes,
                    loc.uploaded_at.isoformat(),
                ),
            )
        await self.db.commit()

        # Update latest version
        await self.db.execute(
            "UPDATE packages SET latest_version = ?, updated_at = ? WHERE name = ?",
            (version.version, datetime.utcnow().isoformat(), package_name),
        )
        await self.db.commit()

        logger.info(f"Added version {version.version} to package {package_name}")
        return version

    async def get_version(
        self,
        package_name: str,
        version: str,
    ) -> Optional[PackageVersion]:
        """Get a specific version of a package.

        Args:
            package_name: Package name.
            version: Version string.

        Returns:
            PackageVersion or None.
        """
        row = await self.db.fetch_one(
            "SELECT * FROM package_versions WHERE package_name = ? AND version = ?",
            (package_name, version),
        )

        if not row:
            return None

        return await self._row_to_version(row)

    async def _row_to_version(self, row) -> PackageVersion:
        """Convert a database row to a PackageVersion object.

        Args:
            row: Database row.

        Returns:
            PackageVersion object.
        """
        # Get storage locations
        locations = await self.db.fetch_all(
            "SELECT * FROM storage_locations WHERE version_id = ?",
            (row["id"],),
        )

        deps_data = json.loads(row["dependencies"]) if row["dependencies"] else []
        dependencies = [
            Dependency(
                name=d["name"],
                version_spec=d["version_spec"],
                optional=d.get("optional", False),
                features=d.get("features", []),
            )
            for d in deps_data
        ]

        storage_locs = [
            StorageLocation(
                backend=loc["backend"],
                bucket=loc["bucket"],
                path=loc["path"],
                checksum_sha256=loc["checksum_sha256"],
                size_bytes=loc["size_bytes"],
                uploaded_at=datetime.fromisoformat(loc["uploaded_at"]),
            )
            for loc in locations
        ]

        return PackageVersion(
            version=row["version"],
            released_at=datetime.fromisoformat(row["released_at"]),
            changelog=row["changelog"],
            dependencies=dependencies,
            toolbox_version=row["toolbox_version"],
            python_version=row["python_version"],
            storage_locations=storage_locs,
            downloads=row["downloads"],
            yanked=bool(row["yanked"]),
            yank_reason=row["yank_reason"],
        )

    async def get_versions(self, package_name: str) -> list[PackageVersion]:
        """Get all versions of a package.

        Args:
            package_name: Package name.

        Returns:
            List of versions.
        """
        rows = await self.db.fetch_all(
            "SELECT * FROM package_versions WHERE package_name = ? ORDER BY released_at DESC",
            (package_name,),
        )
        return [await self._row_to_version(row) for row in rows]

    async def increment_downloads(
        self,
        package_name: str,
        version: str,
    ) -> None:
        """Increment download count for a version.

        Args:
            package_name: Package name.
            version: Version string.
        """
        await self.db.execute(
            "UPDATE package_versions SET downloads = downloads + 1 WHERE package_name = ? AND version = ?",
            (package_name, version),
        )
        await self.db.execute(
            "UPDATE packages SET total_downloads = total_downloads + 1 WHERE name = ?",
            (package_name,),
        )
        await self.db.commit()

    async def search(
        self,
        query: str,
        package_type: Optional[PackageType] = None,
        page: int = 1,
        per_page: int = 20,
    ) -> list[Package]:
        """Search packages using full-text search.

        Args:
            query: Search query.
            package_type: Filter by package type.
            page: Page number.
            per_page: Items per page.

        Returns:
            List of matching packages.
        """
        sql = """
        SELECT p.* FROM packages p
        JOIN packages_fts fts ON p.name = fts.name
        WHERE packages_fts MATCH ?
        """
        params: list = [query]

        if package_type:
            sql += " AND p.package_type = ?"
            params.append(package_type.value)

        sql += " ORDER BY rank LIMIT ? OFFSET ?"
        params.extend([per_page, (page - 1) * per_page])

        rows = await self.db.fetch_all(sql, tuple(params))
        return [self._row_to_package(row) for row in rows]

