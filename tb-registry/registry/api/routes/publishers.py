"""Publisher routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from registry.api.deps import get_current_user, get_user_repo, get_verification_service
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
    user_repo: UserRepository = Depends(get_user_repo),
) -> PublisherListResponse:
    """List publishers with pagination.

    Args:
        page: Page number.
        per_page: Items per page.
        verified_only: Only show verified publishers.
        user_repo: User repository.

    Returns:
        Paginated publisher list.
    """
    status_filter = VerificationStatus.VERIFIED if verified_only else None
    publishers = await user_repo.list_publishers(
        page=page,
        per_page=per_page,
        status=status_filter,
    )

    return PublisherListResponse(
        publishers=[
            PublisherResponse(
                id=p.id,
                name=p.name,
                display_name=p.display_name,
                verification_status=p.verification_status.value,
                package_count=p.package_count,
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
    """Get a publisher by name.

    Args:
        name: Publisher name.
        user_repo: User repository.

    Returns:
        Publisher.
    """
    publisher = await user_repo.get_publisher_by_name(name)
    if not publisher:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Publisher not found",
        )

    return PublisherResponse(
        id=publisher.id,
        name=publisher.name,
        display_name=publisher.display_name,
        verification_status=publisher.verification_status.value,
        package_count=publisher.package_count,
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

