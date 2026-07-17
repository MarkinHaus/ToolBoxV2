"""Pytest configuration and fixtures for registry tests."""

collect_ignore_glob = ["global/*"]

import asyncio
import sys
from pathlib import Path

import pytest
import pytest_asyncio

# Ensure registry package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from registry.db.database import Database
from registry.db.repositories.package_repo import PackageRepository
from registry.db.repositories.artifact_repo import ArtifactRepository
from registry.db.repositories.user_repo import UserRepository
from registry.models.user import User


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def db(tmp_path):
    """Provide a fresh in-memory or temp DB for each test."""
    db_path = tmp_path / "test_registry.db"
    database = Database(f"sqlite:///{db_path}")
    await database.initialize()
    yield database
    await database.close()


@pytest_asyncio.fixture
async def package_repo(db):
    """Package repository with fresh DB."""
    return PackageRepository(db)


@pytest_asyncio.fixture
async def artifact_repo(db):
    """Artifact repository with fresh DB."""
    return ArtifactRepository(db)


@pytest_asyncio.fixture
async def user_repo(db):
    """User repository with fresh DB."""
    return UserRepository(db)


@pytest_asyncio.fixture
async def test_user(user_repo):
    """Create a test user."""
    user = User(
        cloudm_user_id="test-user-001",
        email="test@toolbox.dev",
        username="testuser",
    )
    return await user_repo.create(user)


@pytest_asyncio.fixture
async def test_publisher(user_repo, test_user):
    """Ensure test user has a publisher_id."""
    from registry.models.user import Publisher  # noqa: F401
    # Directly create publisher via DB
    await user_repo.db.execute(
        """INSERT OR IGNORE INTO publishers (id, cloudm_user_id, name, slug, email, status)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("pub-001", "test-user-001", "Test Publisher", "test-pub", "test@toolbox.dev", "verified"),
    )
    await user_repo.db.execute(
        "UPDATE users SET publisher_id = ? WHERE cloudm_user_id = ?",
        ("pub-001", "test-user-001"),
    )
    await user_repo.db.commit()
    return "pub-001"
