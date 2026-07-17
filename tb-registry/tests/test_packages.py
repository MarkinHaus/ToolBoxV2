"""Tests for package repository — CRUD, versions, search, FTS."""
import pytest

from registry.models.package import (
    Dependency,
    Package,
    PackageType,
    PackageVersion,
    Visibility,
)


@pytest.mark.asyncio
async def test_create_package(package_repo, test_publisher):
    """Package can be created and retrieved."""
    pkg = Package(
        name="test-mod",
        display_name="Test Mod",
        package_type=PackageType.MOD,
        owner_id="test-user-001",
        publisher_id=test_publisher,
        visibility=Visibility.PUBLIC,
        description="A test mod",
        keywords=["test", "mod"],
    )
    created = await package_repo.create(pkg)
    assert created.name == "test-mod"

    retrieved = await package_repo.get_by_name("test-mod")
    assert retrieved is not None
    assert retrieved.display_name == "Test Mod"
    assert retrieved.keywords == ["test", "mod"]


@pytest.mark.asyncio
async def test_get_nonexistent_package(package_repo):
    """Getting a nonexistent package returns None."""
    assert await package_repo.get_by_name("nope") is None


@pytest.mark.asyncio
async def test_list_packages(package_repo, test_publisher):
    """List packages with pagination."""
    for i in range(5):
        await package_repo.create(Package(
            name=f"mod-{i}",
            display_name=f"Mod {i}",
            package_type=PackageType.MOD,
            owner_id="test-user-001",
            publisher_id=test_publisher,
            description=f"Mod number {i}",
        ))
    all_pkgs = await package_repo.list_all(page=1, per_page=10)
    assert len(all_pkgs) == 5

    page1 = await package_repo.list_all(page=1, per_page=3)
    assert len(page1) == 3

    page2 = await package_repo.list_all(page=2, per_page=3)
    assert len(page2) == 2


@pytest.mark.asyncio
async def test_list_packages_filter_by_type(package_repo, test_publisher):
    """Filter packages by type."""
    await package_repo.create(Package(
        name="mod-1", display_name="M1",
        package_type=PackageType.MOD,
        owner_id="u", publisher_id=test_publisher,
    ))
    await package_repo.create(Package(
        name="theme-1", display_name="T1",
        package_type=PackageType.THEME,
        owner_id="u", publisher_id=test_publisher,
    ))
    mods = await package_repo.list_all(package_type=PackageType.MOD)
    assert len(mods) == 1
    assert mods[0].name == "mod-1"


@pytest.mark.asyncio
async def test_update_package(package_repo, test_publisher):
    """Update package fields."""
    await package_repo.create(Package(
        name="update-me",
        display_name="Old Name",
        package_type=PackageType.MOD,
        owner_id="u", publisher_id=test_publisher,
        description="Old description",
    ))
    updated = await package_repo.update("update-me", {
        "display_name": "New Name",
        "description": "New description",
    })
    assert updated.display_name == "New Name"
    assert updated.description == "New description"


@pytest.mark.asyncio
async def test_delete_package(package_repo, test_publisher):
    """Delete a package."""
    await package_repo.create(Package(
        name="delete-me",
        display_name="Delete Me",
        package_type=PackageType.MOD,
        owner_id="u", publisher_id=test_publisher,
    ))
    assert await package_repo.delete("delete-me") is True
    assert await package_repo.get_by_name("delete-me") is None


@pytest.mark.asyncio
async def test_add_version(package_repo, test_publisher):
    """Add a version to a package."""
    await package_repo.create(Package(
        name="versioned-mod",
        display_name="Versioned",
        package_type=PackageType.MOD,
        owner_id="u", publisher_id=test_publisher,
    ))
    version = PackageVersion(
        version="1.0.0",
        changelog="Initial",
        dependencies=[
            Dependency(name="dep-a", version_spec=">=1.0.0"),
        ],
    )
    created = await package_repo.add_version("versioned-mod", version)
    assert created.version == "1.0.0"

    pkg = await package_repo.get_by_name("versioned-mod")
    assert pkg.latest_version == "1.0.0"

    retrieved = await package_repo.get_version("versioned-mod", "1.0.0")
    assert retrieved is not None
    assert len(retrieved.dependencies) == 1


@pytest.mark.asyncio
async def test_get_versions(package_repo, test_publisher):
    """Get all versions of a package."""
    await package_repo.create(Package(
        name="multi-version",
        display_name="Multi",
        package_type=PackageType.MOD,
        owner_id="u", publisher_id=test_publisher,
    ))
    await package_repo.add_version("multi-version", PackageVersion(version="1.0.0"))
    await package_repo.add_version("multi-version", PackageVersion(version="2.0.0"))

    versions = await package_repo.get_versions("multi-version")
    assert len(versions) == 2


@pytest.mark.asyncio
async def test_search_packages_fts(package_repo, test_publisher):
    """FTS search returns matching packages."""
    await package_repo.create(Package(
        name="searchable-mod",
        display_name="Searchable Mod",
        package_type=PackageType.MOD,
        owner_id="u", publisher_id=test_publisher,
        description="A highly searchable mod for testing",
        keywords=["searchable", "mod"],
    ))
    await package_repo.create(Package(
        name="other-thing",
        display_name="Other",
        package_type=PackageType.MOD,
        owner_id="u", publisher_id=test_publisher,
        description="Completely unrelated",
    ))

    results = await package_repo.search("searchable")
    assert len(results) >= 1
    names = [r.name for r in results]
    assert "searchable-mod" in names
    assert "other-thing" not in names


@pytest.mark.asyncio
async def test_search_fts_no_results(package_repo, test_publisher):
    """Search with no matches returns empty list."""
    await package_repo.create(Package(
        name="exists",
        display_name="Exists",
        package_type=PackageType.MOD,
        owner_id="u", publisher_id=test_publisher,
    ))
    results = await package_repo.search("nonexistent")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_increment_downloads(package_repo, test_publisher):
    """Download counter increments."""
    await package_repo.create(Package(
        name="downloaded",
        display_name="Downloaded",
        package_type=PackageType.MOD,
        owner_id="u", publisher_id=test_publisher,
    ))
    await package_repo.add_version("downloaded", PackageVersion(version="1.0.0"))

    await package_repo.increment_downloads("downloaded", "1.0.0")
    await package_repo.increment_downloads("downloaded", "1.0.0")

    version = await package_repo.get_version("downloaded", "1.0.0")
    assert version.downloads == 2


@pytest.mark.asyncio
async def test_count_all(package_repo, test_publisher):
    """Count all packages."""
    for i in range(3):
        await package_repo.create(Package(
            name=f"count-{i}", display_name=f"C{i}",
            package_type=PackageType.MOD,
            owner_id="u", publisher_id=test_publisher,
        ))
    assert await package_repo.count_all() == 3
