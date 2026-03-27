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
    id: str
    name: str          # = slug  (URL-friendly, used by client as .name)
    display_name: str  # = publisher.name (human display name)
    verification_status: str
    package_count: int
    total_downloads: int


class PublisherListResponse(BaseModel):
    publishers: list[PublisherResponse]
    total: int
    page: int
    per_page: int


class VerificationRequest(BaseModel):
    method: VerificationMethod
    data: dict


def _to_response(p: Publisher) -> PublisherResponse:
    return PublisherResponse(
        id=p.id,
        name=p.slug,            # client reads .name → slug is the URL key
        display_name=p.name,    # client reads .display_name → human name
        verification_status=p.status.value,
        package_count=p.packages_count,
        total_downloads=p.total_downloads,
    )


@router.get("", response_model=PublisherListResponse)
async def list_publishers(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    verified_only: bool = Query(False),
    user_repo: UserRepository = Depends(get_user_repo),
) -> PublisherListResponse:
    status_filter = VerificationStatus.VERIFIED if verified_only else None
    publishers = await user_repo.list_publishers(page=page, per_page=per_page, status=status_filter)
    return PublisherListResponse(
        publishers=[_to_response(p) for p in publishers],
        total=len(publishers),
        page=page,
        per_page=per_page,
    )


@router.get("/{name}", response_model=PublisherResponse)
async def get_publisher(
    name: str,
    user_repo: UserRepository = Depends(get_user_repo),
) -> PublisherResponse:
    publisher = await user_repo.get_publisher_by_slug(name)
    if not publisher:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Publisher not found")
    return _to_response(publisher)


@router.post("/verify", status_code=status.HTTP_202_ACCEPTED)
async def submit_verification(
    request: VerificationRequest,
    user: User = Depends(get_current_user),
    verification_service: VerificationService = Depends(get_verification_service),
) -> dict:
    if not user.publisher_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Must be a registered publisher")
    await verification_service.submit_request(
        cloudm_user_id=user.cloudm_user_id,
        publisher_id=user.publisher_id,
        method=request.method,
        data=request.data,
    )
    return {"status": "submitted", "message": "Verification request submitted"}
