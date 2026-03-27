"""Versions endpoint — latest version per package (update-ping)."""

from fastapi import APIRouter, Depends, Query

from registry.api.deps import get_package_repo
from registry.db.repositories.package_repo import PackageRepository

router = APIRouter()


@router.get("")
async def get_latest_versions(
    names: list[str] = Query(default=[]),
    repo: PackageRepository = Depends(get_package_repo),
) -> dict:
    """Return latest version for each requested package name.

    Used by ToolBoxV2 update-ping to check for new releases.
    Empty `names` → returns all packages.

    Args:
        names: List of package names to query (repeatable: ?names=a&names=b).
        repo: Package repository.

    Returns:
        {"versions": {"pkg-name": "1.2.3" | null, ...}}
    """
    filter_names = names if names else None
    results = await repo.get_latest_versions(filter_names)
    return {"versions": results}
