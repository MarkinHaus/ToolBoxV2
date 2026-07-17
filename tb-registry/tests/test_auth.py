"""Tests for user/publisher/auth repository operations."""
import pytest

from registry.models.user import User


@pytest.mark.asyncio
async def test_create_user(user_repo):
    """User can be created and retrieved."""
    user = User(
        cloudm_user_id="auth-user-001",
        email="auth@toolbox.dev",
        username="authuser",
    )
    created = await user_repo.create(user)
    assert created.cloudm_user_id == "auth-user-001"

    retrieved = await user_repo.get_by_cloudm_id("auth-user-001")
    assert retrieved is not None
    assert retrieved.email == "auth@toolbox.dev"
    assert retrieved.username == "authuser"


@pytest.mark.asyncio
async def test_get_nonexistent_user(user_repo):
    """Getting a nonexistent user returns None."""
    assert await user_repo.get_by_cloudm_id("nope") is None


@pytest.mark.asyncio
async def test_create_duplicate_user_fails(user_repo):
    """Duplicate cloudm_user_id should fail."""
    user = User(
        cloudm_user_id="dup-user",
        email="dup@toolbox.dev",
        username="dupuser",
    )
    await user_repo.create(user)

    # Second insert should raise (PRIMARY KEY constraint)
    with pytest.raises(Exception):
        await user_repo.create(user)


@pytest.mark.asyncio
async def test_user_with_publisher(user_repo, test_publisher):
    """User with publisher_id set can be retrieved."""
    user = await user_repo.get_by_cloudm_id("test-user-001")
    assert user is not None
    assert user.publisher_id == "pub-001"


@pytest.mark.asyncio
async def test_admin_flag(user_repo):
    """Admin flag can be set."""
    await user_repo.create(User(
        cloudm_user_id="admin-user",
        email="admin@toolbox.dev",
        username="admin",
    ))
    # Set admin via direct SQL
    await user_repo.db.execute(
        "UPDATE users SET is_admin = 1 WHERE cloudm_user_id = ?",
        ("admin-user",),
    )
    await user_repo.db.commit()

    user = await user_repo.get_by_cloudm_id("admin-user")
    assert user.is_admin is True


@pytest.mark.asyncio
async def test_update_last_login(user_repo):
    """Last login timestamp updates."""
    await user_repo.create(User(
        cloudm_user_id="login-user",
        email="login@toolbox.dev",
        username="loginuser",
    ))
    await user_repo.db.execute(
        "UPDATE users SET last_login = ? WHERE cloudm_user_id = ?",
        ("2026-01-01T00:00:00", "login-user"),
    )
    await user_repo.db.commit()

    user = await user_repo.get_by_cloudm_id("login-user")
    assert user.last_login is not None


@pytest.mark.asyncio
async def test_publisher_creation(db):
    """Publisher can be created via direct DB."""
    await db.execute(
        """INSERT INTO publishers (id, cloudm_user_id, name, slug, email, status)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("pub-test", "pub-user-001", "Test Pub", "test-pub", "pub@toolbox.dev", "verified"),
    )
    await db.commit()

    row = await db.fetch_one("SELECT * FROM publishers WHERE id = ?", ("pub-test",))
    assert row is not None
    assert row["name"] == "Test Pub"
    assert row["status"] == "verified"


@pytest.mark.asyncio
async def test_publisher_user_link(user_repo, test_publisher):
    """User → Publisher link works."""
    user = await user_repo.get_by_cloudm_id("test-user-001")
    assert user.publisher_id == "pub-001"

    row = await user_repo.db.fetch_one(
        "SELECT * FROM publishers WHERE id = ?", ("pub-001",),
    )
    assert row is not None
    assert row["status"] == "verified"
