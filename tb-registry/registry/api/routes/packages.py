"""Publisher routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from registry.api.deps import get_current_user, get_user_repo, get_verification_service, require_admin
from registry.db.repositories.user_repo import UserRepository
from registry.models.user import Publisher, User, VerificationStatus
from registry.services.verification import VerificationMethod, VerificationService

router = APIRouter()


class PublisherResponse(BaseModel):
    """Publisher response.

    Attributes:
        id: Publisher ID.
        name: Publisher name.
        display_name: Display name.
        verification_status: Verification status.
        package_count: Number of packages.
        total_downloads: Total downloads.
    """

    id: str
    name: str
    display_name: str
    verification_status: str
    package_count: int
    total_downloads: int


class PublisherListResponse(BaseModel):
    """Response for publisher list.

    Attributes:
        publishers: List of publishers.
        total: Total count.
        page: Current page.
        per_page: Items per page.
    """

    publishers: list[PublisherResponse]
    total: int
    page: int
    per_page: int


class VerificationRequest(BaseModel):
    """Request to submit verification.

    Attributes:
        method: Verification method.
        data: Method-specific data.
    """

    method: VerificationMethod
    data: dict


@router.get("", response_model=PublisherListResponse)
async def list_publishers(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    verified_only: bool = Query(False),
    status_filter: Optional[str] = Query(None, alias="status"),
    user_repo: UserRepository = Depends(get_user_repo),
) -> PublisherListResponse:
    """List publishers with pagination.

    Args:
        page: Page number.
        per_page: Items per page.
        verified_only: Only show verified publishers (legacy).
        status_filter: Filter by status (unverified, pending, verified, rejected).
        user_repo: User repository.

    Returns:
        Paginated publisher list.
    """
    if status_filter:
        try:
            resolved_status = VerificationStatus(status_filter)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}. "
                f"Valid: {', '.join(s.value for s in VerificationStatus)}",
            )
    elif verified_only:
        resolved_status = VerificationStatus.VERIFIED
    else:
        resolved_status = None

    publishers = await user_repo.list_publishers(
        page=page,
        per_page=per_page,
        status=resolved_status,
    )

    return PublisherListResponse(
        publishers=[
            PublisherResponse(
                id=p.id,
                name=p.slug,
                display_name=p.name,
                verification_status=p.status.value,
                package_count=p.packages_count,
                total_downloads=p.total_downloads,
            )
            for p in publishers
        ],
        total=len(publishers),
        page=page,
        per_page=per_page,
    )


@router.get("/{name}", response_model=PublisherResponse)
async def get_publisher(
    name: str,
    user_repo: UserRepository = Depends(get_user_repo),
) -> PublisherResponse:
    """Get a publisher by slug.

    Args:
        name: Publisher slug.
        user_repo: User repository.

    Returns:
        Publisher.
    """
    publisher = await user_repo.get_publisher_by_slug(name)
    if not publisher:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Publisher not found",
        )

    return PublisherResponse(
        id=publisher.id,
        name=publisher.slug,
        display_name=publisher.name,
        verification_status=publisher.status.value,
        package_count=publisher.packages_count,
        total_downloads=publisher.total_downloads,
    )


@router.post("/verify", status_code=status.HTTP_202_ACCEPTED)
async def submit_verification(
    request: VerificationRequest,
    user: User = Depends(get_current_user),
    verification_service: VerificationService = Depends(get_verification_service),
) -> dict:
    """Submit a verification request.

    Args:
        request: Verification request.
        user: Current authenticated user.
        verification_service: Verification service.

    Returns:
        Submission status.
    """
    if not user.publisher_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must be a registered publisher",
        )

    await verification_service.submit_request(
        cloudm_user_id=user.cloudm_user_id,
        publisher_id=user.publisher_id,
        method=request.method,
        data=request.data,
    )

    return {"status": "submitted", "message": "Verification request submitted"}


# ==================== Admin Endpoints ====================


class AdminPublisherAction(BaseModel):
    """Request body for admin publisher actions.

    Attributes:
        notes: Optional notes for audit trail.
    """

    notes: str = ""


@router.get("/admin/pending", response_model=PublisherListResponse)
async def admin_list_pending(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100),
    user: User = Depends(require_admin),
    user_repo: UserRepository = Depends(get_user_repo),
) -> PublisherListResponse:
    """List publishers with pending verification requests (admin only).

    Args:
        page: Page number.
        per_page: Items per page.
        user: Current admin user.
        user_repo: User repository.

    Returns:
        Paginated list of pending publishers.
    """
    publishers = await user_repo.list_publishers(
        page=page,
        per_page=per_page,
        status=VerificationStatus.PENDING,
    )

    return PublisherListResponse(
        publishers=[
            PublisherResponse(
                id=p.id,
                name=p.slug,
                display_name=p.name,
                verification_status=p.status.value,
                package_count=p.packages_count,
                total_downloads=p.total_downloads,
            )
            for p in publishers
        ],
        total=len(publishers),
        page=page,
        per_page=per_page,
    )


@router.post("/admin/{publisher_id}/verify")
async def admin_verify_publisher(
    publisher_id: str,
    body: AdminPublisherAction = AdminPublisherAction(),
    user: User = Depends(require_admin),
    user_repo: UserRepository = Depends(get_user_repo),
) -> dict:
    """Verify a publisher (admin only).

    Args:
        publisher_id: Publisher ID to verify.
        body: Optional notes.
        user: Current admin user.
        user_repo: User repository.

    Returns:
        Action result.
    """
    publisher = await user_repo.get_publisher(publisher_id)
    if not publisher:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Publisher not found",
        )

    await user_repo.update_publisher_status(
        publisher_id=publisher_id,
        status=VerificationStatus.VERIFIED,
        verified_by=user.id,
        notes=body.notes or f"Verified by {user.username}",
    )

    return {
        "status": "verified",
        "publisher_id": publisher_id,
        "message": f"Publisher '{publisher.slug}' is now verified",
    }


@router.post("/admin/{publisher_id}/reject")
async def admin_reject_publisher(
    publisher_id: str,
    body: AdminPublisherAction = AdminPublisherAction(),
    user: User = Depends(require_admin),
    user_repo: UserRepository = Depends(get_user_repo),
) -> dict:
    """Reject a publisher verification (admin only).

    Args:
        publisher_id: Publisher ID to reject.
        body: Optional notes with reason.
        user: Current admin user.
        user_repo: User repository.

    Returns:
        Action result.
    """
    publisher = await user_repo.get_publisher(publisher_id)
    if not publisher:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Publisher not found",
        )

    await user_repo.update_publisher_status(
        publisher_id=publisher_id,
        status=VerificationStatus.REJECTED,
        verified_by=user.id,
        notes=body.notes or f"Rejected by {user.username}",
    )

    return {
        "status": "rejected",
        "publisher_id": publisher_id,
        "message": f"Publisher '{publisher.slug}' verification rejected",
    }


@router.post("/admin/{publisher_id}/revoke")
async def admin_revoke_publisher(
    publisher_id: str,
    body: AdminPublisherAction = AdminPublisherAction(),
    user: User = Depends(require_admin),
    user_repo: UserRepository = Depends(get_user_repo),
) -> dict:
    """Revoke a publisher's verified status (admin only).

    Sets status back to unverified.

    Args:
        publisher_id: Publisher ID to revoke.
        body: Optional notes with reason.
        user: Current admin user.
        user_repo: User repository.

    Returns:
        Action result.
    """
    publisher = await user_repo.get_publisher(publisher_id)
    if not publisher:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Publisher not found",
        )

    if publisher.status != VerificationStatus.VERIFIED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Publisher is not verified (current: {publisher.status.value})",
        )

    await user_repo.update_publisher_status(
        publisher_id=publisher_id,
        status=VerificationStatus.UNVERIFIED,
        verified_by=user.id,
        notes=body.notes or f"Revoked by {user.username}",
    )

    return {
        "status": "revoked",
        "publisher_id": publisher_id,
        "message": f"Publisher '{publisher.slug}' verification revoked",
    }
