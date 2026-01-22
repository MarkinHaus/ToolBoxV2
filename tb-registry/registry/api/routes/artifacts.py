"""Artifact routes."""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel

from registry.api.deps import get_artifact_service, get_current_user
from registry.exceptions import PackageNotFoundError, PermissionDeniedError, VersionNotFoundError
from registry.models.artifact import Artifact, ArtifactType, ArtifactVersion
from registry.models.package import Architecture, Platform
from registry.models.user import User
from registry.services.artifact_service import ArtifactService

router = APIRouter()


class CreateArtifactRequest(BaseModel):
    """Request to create an artifact.

    Attributes:
        name: Artifact name.
        artifact_type: Type of artifact.
        description: Artifact description.
        homepage: Homepage URL.
        repository: Repository URL.
    """

    name: str
    artifact_type: ArtifactType
    description: str = ""
    homepage: Optional[str] = None
    repository: Optional[str] = None


class ArtifactListResponse(BaseModel):
    """Response for artifact list.

    Attributes:
        artifacts: List of artifacts.
        total: Total count.
        page: Current page.
        per_page: Items per page.
    """

    artifacts: list[Artifact]
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


@router.get("", response_model=ArtifactListResponse)
async def list_artifacts(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    artifact_type: Optional[ArtifactType] = None,
    service: ArtifactService = Depends(get_artifact_service),
) -> ArtifactListResponse:
    """List artifacts with pagination.

    Args:
        page: Page number.
        per_page: Items per page.
        artifact_type: Filter by artifact type.
        service: Artifact service.

    Returns:
        Paginated artifact list.
    """
    artifacts = await service.list_artifacts(
        page=page,
        per_page=per_page,
        artifact_type=artifact_type,
    )

    return ArtifactListResponse(
        artifacts=artifacts,
        total=len(artifacts),
        page=page,
        per_page=per_page,
    )


@router.post("", response_model=Artifact, status_code=status.HTTP_201_CREATED)
async def create_artifact(
    data: CreateArtifactRequest,
    user: User = Depends(get_current_user),
    service: ArtifactService = Depends(get_artifact_service),
) -> Artifact:
    """Create a new artifact.

    Args:
        data: Artifact creation data.
        user: Current authenticated user.
        service: Artifact service.

    Returns:
        Created artifact.
    """
    if not user.publisher_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Must be a registered publisher",
        )

    return await service.create_artifact(
        name=data.name,
        artifact_type=data.artifact_type,
        publisher_id=user.publisher_id,
        description=data.description,
        homepage=data.homepage,
        repository=data.repository,
    )


@router.get("/{name}", response_model=Artifact)
async def get_artifact(
    name: str,
    service: ArtifactService = Depends(get_artifact_service),
) -> Artifact:
    """Get an artifact by name.

    Args:
        name: Artifact name.
        service: Artifact service.

    Returns:
        Artifact.
    """
    try:
        return await service.get_artifact(name)
    except PackageNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post("/{name}/builds", response_model=ArtifactVersion, status_code=status.HTTP_201_CREATED)
async def upload_build(
    name: str,
    version: str = Form(...),
    platform: Platform = Form(...),
    architecture: Architecture = Form(...),
    changelog: str = Form(""),
    installer_type: Optional[str] = Form(None),
    min_os_version: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    service: ArtifactService = Depends(get_artifact_service),
) -> ArtifactVersion:
    """Upload a build for an artifact.

    Args:
        name: Artifact name.
        version: Version string.
        platform: Target platform.
        architecture: Target architecture.
        changelog: Version changelog.
        installer_type: Type of installer.
        min_os_version: Minimum OS version.
        file: Build file.
        user: Current authenticated user.
        service: Artifact service.

    Returns:
        Created version.
    """
    if not user.publisher_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Must be a registered publisher",
        )

    try:
        return await service.upload_build(
            artifact_name=name,
            version=version,
            file=file,
            platform=platform,
            architecture=architecture,
            publisher_id=user.publisher_id,
            changelog=changelog,
            installer_type=installer_type,
            min_os_version=min_os_version,
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


@router.get("/{name}/latest")
async def get_latest_for_platform(
    name: str,
    platform: Platform = Query(...),
    architecture: Architecture = Query(...),
    service: ArtifactService = Depends(get_artifact_service),
) -> dict:
    """Get latest version for a platform.

    Args:
        name: Artifact name.
        platform: Target platform.
        architecture: Target architecture.
        service: Artifact service.

    Returns:
        Latest version and build info.
    """
    result = await service.get_latest_for_platform(name, platform, architecture)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No build found for platform",
        )

    version, build = result
    return {
        "version": version.version,
        "released_at": version.released_at,
        "changelog": version.changelog,
        "build": {
            "platform": build.platform.value,
            "architecture": build.architecture.value,
            "filename": build.filename,
            "size_bytes": build.size_bytes,
            "checksum_sha256": build.checksum_sha256,
        },
    }


@router.get("/{name}/versions/{version}/download", response_model=DownloadUrlResponse)
async def get_download_url(
    name: str,
    version: str,
    platform: Platform = Query(...),
    architecture: Architecture = Query(...),
    service: ArtifactService = Depends(get_artifact_service),
) -> DownloadUrlResponse:
    """Get download URL for a build.

    Args:
        name: Artifact name.
        version: Version string.
        platform: Target platform.
        architecture: Target architecture.
        service: Artifact service.

    Returns:
        Download URL response.
    """
    try:
        url = await service.get_download_url(name, version, platform, architecture)
        if not url:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No download available for platform",
            )

        await service.increment_downloads(name, version, platform, architecture)
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

