"""Authentication routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from registry.api.deps import get_current_user, get_user_repo
from registry.db.repositories.user_repo import UserRepository
from registry.models.user import Publisher, User

router = APIRouter()


class RegisterPublisherRequest(BaseModel):
    """Request to register as a publisher.

    Attributes:
        name: Publisher name.
        display_name: Display name.
        email: Contact email.
        homepage: Optional homepage URL.
    """

    name: str
    display_name: str
    email: str
    homepage: str | None = None


class PublisherResponse(BaseModel):
    """Publisher response.

    Attributes:
        id: Publisher ID.
        name: Publisher name.
        display_name: Display name.
        verification_status: Verification status.
    """

    id: str
    name: str
    display_name: str
    verification_status: str


@router.get("/me")
async def get_current_user_info(
    user: User = Depends(get_current_user),
) -> dict:
    """Get current user information.

    Args:
        user: Current authenticated user.

    Returns:
        User information.
    """
    return {
        "id": user.id,
        "cloudm_user_id": user.cloudm_user_id,
        "email": user.email,
        "username": user.username,
        "is_admin": user.is_admin,
        "publisher_id": user.publisher_id,
    }


@router.post("/register-publisher", response_model=PublisherResponse)
async def register_publisher(
    request: RegisterPublisherRequest,
    user: User = Depends(get_current_user),
    user_repo: UserRepository = Depends(get_user_repo),
) -> PublisherResponse:
    """Register as a publisher.

    Args:
        request: Registration request.
        user: Current authenticated user.
        user_repo: User repository.

    Returns:
        Created publisher.

    Raises:
        HTTPException: If already a publisher or name taken.
    """
    if user.publisher_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Already registered as a publisher",
        )

    # Check if slug is taken
    existing = await user_repo.get_publisher_by_slug(request.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Publisher name already taken",
        )

    publisher = Publisher(
        id="",  # Will be generated
        cloudm_user_id=user.cloudm_user_id,
        name=request.display_name,
        slug=request.name,
        email=request.email,
        website=request.homepage,
    )

    created = await user_repo.create_publisher(publisher)

    return PublisherResponse(
        id=created.id,
        name=created.slug,
        display_name=created.name,
        verification_status=created.status.value,
    )


@router.get("/publisher")
async def get_my_publisher(
    user: User = Depends(get_current_user),
    user_repo: UserRepository = Depends(get_user_repo),
) -> PublisherResponse | None:
    """Get current user's publisher profile.

    Args:
        user: Current authenticated user.
        user_repo: User repository.

    Returns:
        Publisher profile or None.
    """
    if not user.publisher_id:
        return None

    publisher = await user_repo.get_publisher(user.publisher_id)
    if not publisher:
        return None

    return PublisherResponse(
        id=publisher.id,
        name=publisher.slug,
        display_name=publisher.name,
        verification_status=publisher.status.value,
    )
