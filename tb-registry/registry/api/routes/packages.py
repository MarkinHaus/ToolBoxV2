"""Package routes."""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel

from registry.api.deps import (
    get_current_user,
    get_optional_user,
    get_package_service,
)
from registry.exceptions import (
    DuplicatePackageError,
    DuplicateVersionError,
    PackageNotFoundError,
    PermissionDeniedError,
    VersionNotFoundError,
)
from registry.models.package import (
    Package,
    PackageCreate,
    PackageSummary,
    PackageType,
    PackageUpdate,
    PackageVersion,
)
from registry.models.user import User
from registry.services.package_service import PackageService

router = APIRouter()


class PackageListResponse(BaseModel):
    """Response for package list.

    Attributes:
        packages: List of package summaries.
        total: Total count.
        page: Current page.
        per_page: Items per page.
    """

    packages: list[PackageSummary]
    total: int
    page: int
    per_page: int


class DownloadUrlResponse(BaseModel):
    """Response for download URL.

    Attributes:
        url: Presigned download URL.
        expires_in: Expiration time in seconds.
    """

    url: str
    expires_in: int


@router.get("", response_model=PackageListResponse)
async def list_packages(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    package_type: Optional[PackageType] = None,
    user: Optional[User] = Depends(get_optional_user),
    service: PackageService = Depends(get_package_service),
) -> PackageListResponse:
    """List packages with pagination.

    Args:
        page: Page number.
        per_page: Items per page.
        package_type: Filter by package type.
        user: Optional current user.
        service: Package service.

    Returns:
        Paginated package list.
    """
    viewer_id = user.id if user else None
    packages, total = await service.list_packages(
        page=page,
        per_page=per_page,
        package_type=package_type,
        viewer_id=viewer_id,
    )

    return PackageListResponse(
        packages=packages,
        total=total,
        page=page,
        per_page=per_page,
    )


@router.post("", response_model=Package, status_code=status.HTTP_201_CREATED)
async def create_package(
    data: PackageCreate,
    user: User = Depends(get_current_user),
    service: PackageService = Depends(get_package_service),
) -> Package:
    """Create a new package.

    Args:
        data: Package creation data.
        user: Current authenticated user.
        service: Package service.

    Returns:
        Created package.

    Raises:
        HTTPException: If not a publisher or package exists.
    """
    if not user.publisher_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Must be a registered publisher to create packages",
        )

    try:
        return await service.create_package(
            data=data,
            publisher_id=user.publisher_id,
            owner_id=user.id,
        )
    except DuplicatePackageError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )


@router.get("/{name}", response_model=Package)
async def get_package(
    name: str,
    user: Optional[User] = Depends(get_optional_user),
    service: PackageService = Depends(get_package_service),
) -> Package:
    """Get a package by name.

    Args:
        name: Package name.
        user: Optional current user.
        service: Package service.

    Returns:
        Package.

    Raises:
        HTTPException: If package not found or not accessible.
    """
    viewer_id = user.id if user else None
    try:
        return await service.get_package(name, viewer_id)
    except PackageNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.patch("/{name}", response_model=Package)
async def update_package(
    name: str,
    data: PackageUpdate,
    user: User = Depends(get_current_user),
    service: PackageService = Depends(get_package_service),
) -> Package:
    """Update a package.

    Args:
        name: Package name.
        data: Update data.
        user: Current authenticated user.
        service: Package service.

    Returns:
        Updated package.
    """
    try:
        return await service.update_package(name, data, user.id)
    except PackageNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_package(
    name: str,
    user: User = Depends(get_current_user),
    service: PackageService = Depends(get_package_service),
) -> None:
    """Delete a package.

    Args:
        name: Package name.
        user: Current authenticated user.
        service: Package service.
    """
    try:
        await service.delete_package(name, user.id, user.is_admin)
    except PackageNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.post("/{name}/versions", response_model=PackageVersion, status_code=status.HTTP_201_CREATED)
async def upload_version(
    name: str,
    version: str = Form(...),
    changelog: str = Form(""),
    toolbox_version: Optional[str] = Form(None),
    python_version: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    service: PackageService = Depends(get_package_service),
) -> PackageVersion:
    """Upload a new version.

    Args:
        name: Package name.
        version: Version string.
        changelog: Version changelog.
        toolbox_version: Required ToolBox version.
        python_version: Required Python version.
        file: Package file.
        user: Current authenticated user.
        service: Package service.

    Returns:
        Created version.
    """
    if not user.publisher_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Must be a registered publisher",
        )

    try:
        return await service.upload_version(
            name=name,
            version=version,
            file=file,
            changelog=changelog,
            publisher_id=user.publisher_id,
            toolbox_version=toolbox_version,
            python_version=python_version,
        )
    except PackageNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )
    except DuplicateVersionError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )


@router.get("/{name}/versions/{version}/download", response_model=DownloadUrlResponse)
async def get_download_url(
    name: str,
    version: str,
    prefer_mirror: bool = Query(False),
    service: PackageService = Depends(get_package_service),
) -> DownloadUrlResponse:
    """Get download URL for a version.

    Args:
        name: Package name.
        version: Version string.
        prefer_mirror: Prefer mirror URL.
        service: Package service.

    Returns:
        Download URL response.
    """
    try:
        url = await service.get_download_url(name, version, prefer_mirror)
        if not url:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No download available",
            )

        # Increment download count
        await service.increment_downloads(name, version)

        return DownloadUrlResponse(url=url, expires_in=3600)
    except PackageNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except VersionNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

