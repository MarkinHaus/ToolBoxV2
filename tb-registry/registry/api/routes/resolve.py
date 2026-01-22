"""Dependency resolution routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from registry.api.deps import get_dependency_resolver
from registry.resolver.dependency import DependencyResolver, Requirement

router = APIRouter()


class ResolveRequest(BaseModel):
    """Request to resolve dependencies.

    Attributes:
        requirements: List of requirement strings.
        toolbox_version: Optional ToolBox version.
    """

    requirements: list[str]
    toolbox_version: Optional[str] = None


class ResolvedPackageResponse(BaseModel):
    """Resolved package response.

    Attributes:
        name: Package name.
        version: Resolved version.
        download_url: Download URL.
        checksum: SHA256 checksum.
    """

    name: str
    version: str
    download_url: Optional[str] = None
    checksum: Optional[str] = None


class ResolveResponse(BaseModel):
    """Response for dependency resolution.

    Attributes:
        success: Whether resolution succeeded.
        resolved: Dict of resolved packages.
        conflicts: List of conflicts.
        warnings: List of warnings.
    """

    success: bool
    resolved: dict[str, ResolvedPackageResponse]
    conflicts: list[str]
    warnings: list[str]


@router.post("", response_model=ResolveResponse)
async def resolve_dependencies(
    request: ResolveRequest,
    resolver: DependencyResolver = Depends(get_dependency_resolver),
) -> ResolveResponse:
    """Resolve package dependencies.

    Args:
        request: Resolution request.
        resolver: Dependency resolver.

    Returns:
        Resolution result.
    """
    # Parse requirements
    requirements = []
    for req_str in request.requirements:
        try:
            req = DependencyResolver.parse_requirement_string(req_str)
            requirements.append(req)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid requirement: {req_str} - {e}",
            )

    # Resolve
    result = await resolver.resolve(
        requirements=requirements,
        toolbox_version=request.toolbox_version,
    )

    # Convert to response
    resolved = {}
    for name, pkg in result.resolved.items():
        resolved[name] = ResolvedPackageResponse(
            name=pkg.name,
            version=pkg.version,
            download_url=pkg.download_url,
            checksum=pkg.checksum,
        )

    return ResolveResponse(
        success=result.success,
        resolved=resolved,
        conflicts=result.conflicts,
        warnings=result.warnings,
    )


@router.get("/check")
async def check_compatibility(
    package: str,
    version: str,
    toolbox_version: str,
    resolver: DependencyResolver = Depends(get_dependency_resolver),
) -> dict:
    """Check if a package version is compatible with ToolBox version.

    Args:
        package: Package name.
        version: Package version.
        toolbox_version: ToolBox version.
        resolver: Dependency resolver.

    Returns:
        Compatibility check result.
    """
    req = Requirement(name=package, specifier=f"=={version}")
    result = await resolver.resolve(
        requirements=[req],
        toolbox_version=toolbox_version,
    )

    return {
        "compatible": result.success and not result.warnings,
        "warnings": result.warnings,
        "conflicts": result.conflicts,
    }

