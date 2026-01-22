"""Async SQLite database management."""

import logging
from pathlib import Path
from typing import Optional

import aiosqlite

from registry.config import get_settings

logger = logging.getLogger(__name__)


class Database:
    """Async SQLite database manager.

    Handles connection management, schema creation, and migrations.

    Attributes:
        db_path: Path to the SQLite database file.
        connection: Active database connection.
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        """Initialize the database manager.

        Args:
            db_url: Database URL (default: from settings).
        """
        url = db_url or get_settings().database_url
        # Extract path from sqlite:/// URL
        if url.startswith("sqlite:///"):
            self.db_path = Path(url[10:])
        else:
            self.db_path = Path(url)
        self.connection: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Initialize the database connection and schema.

        Creates the database directory if needed and sets up tables.
        """
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.connection = await aiosqlite.connect(str(self.db_path))
        self.connection.row_factory = aiosqlite.Row

        # Enable foreign keys
        await self.connection.execute("PRAGMA foreign_keys = ON")

        # Create schema
        await self._create_schema()

        logger.info(f"Database initialized at {self.db_path}")

    async def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            await self.connection.close()
            self.connection = None
            logger.info("Database connection closed")

    async def _create_schema(self) -> None:
        """Create database tables if they don't exist."""
        schema = """
        -- Schema version tracking
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Publishers table
        CREATE TABLE IF NOT EXISTS publishers (
            id TEXT PRIMARY KEY,
            clerk_user_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            slug TEXT UNIQUE NOT NULL,
            email TEXT NOT NULL,
            website TEXT,
            github TEXT,
            status TEXT DEFAULT 'unverified',
            verified_at TIMESTAMP,
            verified_by TEXT,
            verification_notes TEXT,
            can_publish_public BOOLEAN DEFAULT 0,
            can_publish_artifacts BOOLEAN DEFAULT 0,
            max_package_size_mb INTEGER DEFAULT 100,
            packages_count INTEGER DEFAULT 0,
            total_downloads INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Users table
        CREATE TABLE IF NOT EXISTS users (
            clerk_user_id TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            username TEXT NOT NULL,
            publisher_id TEXT REFERENCES publishers(id),
            is_admin BOOLEAN DEFAULT 0,
            last_login TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Packages table
        CREATE TABLE IF NOT EXISTS packages (
            name TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            package_type TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            publisher_id TEXT NOT NULL REFERENCES publishers(id),
            visibility TEXT DEFAULT 'public',
            description TEXT DEFAULT '',
            readme TEXT DEFAULT '',
            homepage TEXT,
            repository TEXT,
            license TEXT,
            keywords TEXT DEFAULT '[]',
            latest_version TEXT,
            total_downloads INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Package versions table
        CREATE TABLE IF NOT EXISTS package_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            package_name TEXT NOT NULL REFERENCES packages(name) ON DELETE CASCADE,
            version TEXT NOT NULL,
            released_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            changelog TEXT DEFAULT '',
            dependencies TEXT DEFAULT '[]',
            toolbox_version TEXT,
            python_version TEXT,
            downloads INTEGER DEFAULT 0,
            yanked BOOLEAN DEFAULT 0,
            yank_reason TEXT,
            UNIQUE(package_name, version)
        );

        -- Storage locations table
        CREATE TABLE IF NOT EXISTS storage_locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version_id INTEGER REFERENCES package_versions(id) ON DELETE CASCADE,
            backend TEXT NOT NULL,
            bucket TEXT NOT NULL,
            path TEXT NOT NULL,
            checksum_sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Package assets table
        CREATE TABLE IF NOT EXISTS package_assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version_id INTEGER REFERENCES package_versions(id) ON DELETE CASCADE,
            filename TEXT NOT NULL,
            platform TEXT NOT NULL,
            architecture TEXT NOT NULL,
            checksum_sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL
        );

        -- Verification requests table
        CREATE TABLE IF NOT EXISTS verification_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            publisher_id TEXT NOT NULL REFERENCES publishers(id),
            method TEXT NOT NULL,
            data TEXT DEFAULT '{}',
            status TEXT DEFAULT 'pending',
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reviewed_at TIMESTAMP,
            reviewed_by TEXT,
            notes TEXT
        );

        -- Artifacts table
        CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            artifact_type TEXT NOT NULL,
            publisher_id TEXT NOT NULL REFERENCES publishers(id),
            description TEXT DEFAULT '',
            homepage TEXT,
            repository TEXT,
            latest_version TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Artifact versions table
        CREATE TABLE IF NOT EXISTS artifact_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
            version TEXT NOT NULL,
            released_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            changelog TEXT DEFAULT '',
            min_os_version TEXT,
            UNIQUE(artifact_id, version)
        );

        -- Artifact builds table
        CREATE TABLE IF NOT EXISTS artifact_builds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version_id INTEGER REFERENCES artifact_versions(id) ON DELETE CASCADE,
            platform TEXT NOT NULL,
            architecture TEXT NOT NULL,
            filename TEXT NOT NULL,
            checksum_sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            installer_type TEXT,
            downloads INTEGER DEFAULT 0
        );

        -- Download stats table
        CREATE TABLE IF NOT EXISTS download_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            package_name TEXT,
            artifact_id TEXT,
            version TEXT NOT NULL,
            platform TEXT,
            architecture TEXT,
            downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ip_hash TEXT,
            user_agent TEXT
        );

        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_packages_publisher ON packages(publisher_id);
        CREATE INDEX IF NOT EXISTS idx_packages_type ON packages(package_type);
        CREATE INDEX IF NOT EXISTS idx_packages_visibility ON packages(visibility);
        CREATE INDEX IF NOT EXISTS idx_versions_package ON package_versions(package_name);
        CREATE INDEX IF NOT EXISTS idx_artifacts_publisher ON artifacts(publisher_id);
        CREATE INDEX IF NOT EXISTS idx_download_stats_package ON download_stats(package_name);
        CREATE INDEX IF NOT EXISTS idx_download_stats_artifact ON download_stats(artifact_id);

        -- Full-text search for packages
        CREATE VIRTUAL TABLE IF NOT EXISTS packages_fts USING fts5(
            name,
            display_name,
            description,
            keywords,
            content='packages',
            content_rowid='rowid'
        );
        """
        await self.connection.executescript(schema)
        await self.connection.commit()

    async def execute(
        self,
        query: str,
        params: tuple = (),
    ) -> aiosqlite.Cursor:
        """Execute a query and return the cursor.

        Args:
            query: SQL query string.
            params: Query parameters.

        Returns:
            Database cursor.
        """
        return await self.connection.execute(query, params)

    async def execute_many(
        self,
        query: str,
        params_list: list[tuple],
    ) -> None:
        """Execute a query with multiple parameter sets.

        Args:
            query: SQL query string.
            params_list: List of parameter tuples.
        """
        await self.connection.executemany(query, params_list)
        await self.connection.commit()

    async def fetch_one(
        self,
        query: str,
        params: tuple = (),
    ) -> Optional[aiosqlite.Row]:
        """Fetch a single row.

        Args:
            query: SQL query string.
            params: Query parameters.

        Returns:
            Single row or None.
        """
        cursor = await self.connection.execute(query, params)
        return await cursor.fetchone()

    async def fetch_all(
        self,
        query: str,
        params: tuple = (),
    ) -> list[aiosqlite.Row]:
        """Fetch all rows.

        Args:
            query: SQL query string.
            params: Query parameters.

        Returns:
            List of rows.
        """
        cursor = await self.connection.execute(query, params)
        return await cursor.fetchall()

    async def commit(self) -> None:
        """Commit the current transaction."""
        await self.connection.commit()

    async def health_check(self) -> bool:
        """Check database connectivity.

        Returns:
            True if database is healthy, False otherwise.
        """
        try:
            if not self.connection:
                return False
            cursor = await self.connection.execute("SELECT 1")
            result = await cursor.fetchone()
            return result is not None and result[0] == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

