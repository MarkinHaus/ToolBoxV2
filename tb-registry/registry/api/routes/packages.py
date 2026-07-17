"""Package routes."""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel

from registry.api.deps import (
    get_current_user,
    get_package_repo,
    get_package_service,
    get_user_repo,
    require_admin,
)
from registry.db.repositories.package_repo import PackageRepository
from registry.db.repositories.user_repo import UserRepository
from registry.exceptions import DuplicateVersionError, PackageNotFoundError, PermissionDeniedError, VersionNotFoundError
from registry.models.package import PackageCreate, PackageType
from registry.models.user import User, VerificationStatus
from registry.services.package_service import PackageService

router = APIRouter()


class CreatePackageRequest(BaseModel):
    name: str
    display_name: str = ""          # optional — falls back to name
    package_type: PackageType
    description: str = ""
    homepage: Optional[str] = None
    repository: Optional[str] = None


class PackageSummaryResponse(BaseModel):
    name: str
    display_name: str
    package_type: str
    description: str
    latest_version: Optional[str]
    total_downloads: int


class PackageListResponse(BaseModel):
    packages: list[PackageSummaryResponse]
    total: int
    page: int
    per_page: int


# ── Publisher admin responses (kept for admin sub-routes) ──────────────────────

class PublisherResponse(BaseModel):
    id: str
    name: str
    display_name: str
    verification_status: str
    package_count: int
    total_downloads: int


class PublisherListResponse(BaseModel):
    publishers: list[PublisherResponse]
    total: int
    page: int
    per_page: int


def _pub_to_response(p) -> PublisherResponse:
    return PublisherResponse(
        id=p.id,
        name=p.slug,
        display_name=p.name,
        verification_status=p.status.value,
        package_count=p.packages_count,
        total_downloads=p.total_downloads,
    )


# ── Package endpoints ──────────────────────────────────────────────────────────

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_package(
    data: CreatePackageRequest,
    user: User = Depends(get_current_user),
    service: PackageService = Depends(get_package_service),
) -> dict:
    if not user.publisher_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Must be a registered publisher")

    package_data = PackageCreate(
        name=data.name,
        display_name=data.display_name or data.name,
        package_type=data.package_type,
        description=data.description,
        homepage=data.homepage,
        repository=data.repository,
    )
    package = await service.create_package(
        data=package_data,
        publisher_id=user.publisher_id,
        owner_id=user.cloudm_user_id,
    )
    return {"id": package.name, "name": package.name}


@router.get("", response_model=PackageListResponse)
async def list_packages(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    package_type: Optional[PackageType] = None,
    repo: PackageRepository = Depends(get_package_repo),
) -> PackageListResponse:
    packages = await repo.list_all(page=page, per_page=per_page, package_type=package_type)
    return PackageListResponse(
        packages=[
            PackageSummaryResponse(
                name=p.name,
                display_name=p.display_name,
                package_type=p.package_type.value,
                description=p.description,
                latest_version=p.latest_version,
                total_downloads=p.total_downloads,
            )
            for p in packages
        ],
        total=len(packages),
        page=page,
        per_page=per_page,
    )


@router.get("/{name}")
async def get_package(
    name: str,
    repo: PackageRepository = Depends(get_package_repo),
) -> dict:
    package = await repo.get_by_name(name)
    if not package:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Package not found")
    return {
        "name": package.name,
        "display_name": package.display_name,
        "package_type": package.package_type.value,
        "description": package.description,
        "latest_version": package.latest_version,
        "total_downloads": package.total_downloads,
        "homepage": package.homepage,
        "repository": package.repository,
    }


# ÄÄ Version & lifecycle endpoints ÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄ

@router.get("/{name}/versions")
async def list_package_versions(
    name: str,
    repo: PackageRepository = Depends(get_package_repo),
) -> dict:
    """List all versions of a package."""
    versions = await repo.get_versions(name)
    return {
        "package": name,
        "versions": [
            {
                "version": v.version,
                "released_at": v.released_at.isoformat() if v.released_at else None,
                "changelog": v.changelog,
                "downloads": v.downloads,
                "yanked": v.yanked,
                "toolbox_version": v.toolbox_version,
                "python_version": v.python_version,
            }
            for v in versions
        ],
    }


@router.post("/{name}/versions", status_code=status.HTTP_201_CREATED)
async def upload_package_version(
    name: str,
    version: str = Form(...),
    file: UploadFile = File(...),
    changelog: str = Form(""),
    toolbox_version: Optional[str] = Form(None),
    python_version: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
    service: PackageService = Depends(get_package_service),
) -> dict:
    """Upload a new version for a package."""
    if not user.publisher_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Must be a registered publisher")
    try:
        created = await service.upload_version(
            name=name,
            version=version,
            file=file,
            changelog=changelog,
            publisher_id=user.publisher_id,
            toolbox_version=toolbox_version,
            python_version=python_version,
        )
    except PackageNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Package not found")
    except PermissionDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except DuplicateVersionError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Version {version} already exists")
    return {"status": "uploaded", "package": name, "version": created.version}


@router.get("/{name}/versions/{version}/download")
async def download_package_version(
    name: str,
    version: str,
    user: User = Depends(get_current_user),
    service: PackageService = Depends(get_package_service),
) -> dict:
    """Get a presigned download URL for a package version."""
    try:
        url = await service.get_download_url(
            name, version, viewer_id=user.cloudm_user_id
        )
    except PackageNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Package not found")
    except VersionNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")
    except PermissionDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    if not url:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No download available")
    await service.increment_downloads(name, version)
    return {"url": url, "package": name, "version": version}


@router.delete("/{name}")
async def delete_package(
    name: str,
    user: User = Depends(get_current_user),
    service: PackageService = Depends(get_package_service),
) -> dict:
    """Delete a package (owner or admin only)."""
    try:
        await service.delete_package(
            name=name,
            user_id=user.cloudm_user_id,
            is_admin=getattr(user, "is_admin", False),
        )
    except PackageNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Package not found")
    except PermissionDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    return {"status": "deleted", "package": name}


# ── Admin endpoints ────────────────────────────────────────────────────────────

class AdminAction(BaseModel):
    notes: str = ""


@router.get("/admin/pending", response_model=PublisherListResponse)
async def admin_list_pending(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100),
    user: User = Depends(require_admin),
    user_repo: UserRepository = Depends(get_user_repo),
) -> PublisherListResponse:
    publishers = await user_repo.list_publishers(page=page, per_page=per_page, status=VerificationStatus.PENDING)
    return PublisherListResponse(
        publishers=[_pub_to_response(p) for p in publishers],
        total=len(publishers), page=page, per_page=per_page,
    )


@router.post("/admin/{publisher_id}/verify")
async def admin_verify_publisher(
    publisher_id: str, body: AdminAction = AdminAction(),
    user: User = Depends(require_admin),
    user_repo: UserRepository = Depends(get_user_repo),
) -> dict:
    publisher = await user_repo.get_publisher(publisher_id)
    if not publisher:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Publisher not found")
    await user_repo.update_publisher_status(
        publisher_id=publisher_id, status=VerificationStatus.VERIFIED,
        verified_by=user.id, notes=body.notes or f"Verified by {user.username}",
    )
    return {"status": "verified", "publisher_id": publisher_id}


@router.post("/admin/{publisher_id}/reject")
async def admin_reject_publisher(
    publisher_id: str, body: AdminAction = AdminAction(),
    user: User = Depends(require_admin),
    user_repo: UserRepository = Depends(get_user_repo),
) -> dict:
    publisher = await user_repo.get_publisher(publisher_id)
    if not publisher:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Publisher not found")
    await user_repo.update_publisher_status(
        publisher_id=publisher_id, status=VerificationStatus.REJECTED,
        verified_by=user.id, notes=body.notes or f"Rejected by {user.username}",
    )
    return {"status": "rejected", "publisher_id": publisher_id}


@router.post("/admin/{publisher_id}/revoke")
async def admin_revoke_publisher(
    publisher_id: str, body: AdminAction = AdminAction(),
    user: User = Depends(require_admin),
    user_repo: UserRepository = Depends(get_user_repo),
) -> dict:
    publisher = await user_repo.get_publisher(publisher_id)
    if not publisher:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Publisher not found")
    if publisher.status != VerificationStatus.VERIFIED:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Publisher is not verified (current: {publisher.status.value})")
    await user_repo.update_publisher_status(
        publisher_id=publisher_id, status=VerificationStatus.UNVERIFIED,
        verified_by=user.id, notes=body.notes or f"Revoked by {user.username}",
    )
    return {"status": "revoked", "publisher_id": publisher_id}
