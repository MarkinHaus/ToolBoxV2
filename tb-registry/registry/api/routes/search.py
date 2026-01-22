"""Search routes."""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from registry.api.deps import get_package_service
from registry.models.package import PackageSummary, PackageType
from registry.services.package_service import PackageService

router = APIRouter()


class SearchResponse(BaseModel):
    """Response for search.

    Attributes:
        results: List of matching packages.
        total: Total count.
        query: Search query.
        page: Current page.
        per_page: Items per page.
    """

    results: list[PackageSummary]
    total: int
    query: str
    page: int
    per_page: int


@router.get("", response_model=SearchResponse)
async def search_packages(
    q: str = Query(..., min_length=1, max_length=100),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    package_type: Optional[PackageType] = None,
    service: PackageService = Depends(get_package_service),
) -> SearchResponse:
    """Search packages.

    Args:
        q: Search query.
        page: Page number.
        per_page: Items per page.
        package_type: Filter by package type.
        service: Package service.

    Returns:
        Search results.
    """
    results = await service.search(
        query=q,
        package_type=package_type,
        page=page,
        per_page=per_page,
    )

    return SearchResponse(
        results=results,
        total=len(results),
        query=q,
        page=page,
        per_page=per_page,
    )


@router.get("/suggest")
async def suggest_packages(
    q: str = Query(..., min_length=1, max_length=50),
    limit: int = Query(5, ge=1, le=10),
    service: PackageService = Depends(get_package_service),
) -> list[str]:
    """Get package name suggestions.

    Args:
        q: Partial query.
        limit: Maximum suggestions.
        service: Package service.

    Returns:
        List of suggested package names.
    """
    results = await service.search(query=q, page=1, per_page=limit)
    return [r.name for r in results]

