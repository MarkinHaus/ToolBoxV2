"""FastAPI dependencies for dependency injection."""

import logging
from functools import lru_cache
from typing import Optional

import httpx
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from registry.config import Settings, get_settings
from registry.db.repositories.artifact_repo import ArtifactRepository
from registry.db.repositories.package_repo import PackageRepository
from registry.db.repositories.user_repo import UserRepository
from registry.models.user import User
from registry.resolver.dependency import DependencyResolver
from registry.services.artifact_service import ArtifactService
from registry.services.package_service import PackageService
from registry.services.verification import VerificationService
from registry.storage.manager import StorageManager

# Clerk SDK imports for JWT verification
try:
    from clerk_backend_api import Clerk
    from clerk_backend_api.security import authenticate_request
    from clerk_backend_api.security.types import AuthenticateRequestOptions, AuthStatus

    CLERK_SDK_AVAILABLE = True
except ImportError:
    CLERK_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


async def get_db(request: Request):
    """Get database session from request state.

    Args:
        request: FastAPI request.

    Returns:
        Database session.
    """
    return request.state.db


async def get_storage(request: Request) -> StorageManager:
    """Get storage manager from request state.

    Args:
        request: FastAPI request.

    Returns:
        Storage manager.
    """
    return request.state.storage


async def get_user_repo(db=Depends(get_db)) -> UserRepository:
    """Get user repository.

    Args:
        db: Database session.

    Returns:
        User repository.
    """
    return UserRepository(db)


async def get_package_repo(db=Depends(get_db)) -> PackageRepository:
    """Get package repository.

    Args:
        db: Database session.

    Returns:
        Package repository.
    """
    return PackageRepository(db)


async def get_artifact_repo(db=Depends(get_db)) -> ArtifactRepository:
    """Get artifact repository.

    Args:
        db: Database session.

    Returns:
        Artifact repository.
    """
    return ArtifactRepository(db)


async def get_package_service(
    repo: PackageRepository = Depends(get_package_repo),
    user_repo: UserRepository = Depends(get_user_repo),
    storage: StorageManager = Depends(get_storage),
) -> PackageService:
    """Get package service.

    Args:
        repo: Package repository.
        user_repo: User repository.
        storage: Storage manager.

    Returns:
        Package service.
    """
    return PackageService(repo, user_repo, storage)


async def get_artifact_service(
    repo: ArtifactRepository = Depends(get_artifact_repo),
    user_repo: UserRepository = Depends(get_user_repo),
    storage: StorageManager = Depends(get_storage),
) -> ArtifactService:
    """Get artifact service.

    Args:
        repo: Artifact repository.
        user_repo: User repository.
        storage: Storage manager.

    Returns:
        Artifact service.
    """
    return ArtifactService(repo, user_repo, storage)


async def get_verification_service(
    user_repo: UserRepository = Depends(get_user_repo),
) -> VerificationService:
    """Get verification service.

    Args:
        user_repo: User repository.

    Returns:
        Verification service.
    """
    return VerificationService(user_repo)


async def get_dependency_resolver(
    repo: PackageRepository = Depends(get_package_repo),
) -> DependencyResolver:
    """Get dependency resolver.

    Args:
        repo: Package repository.

    Returns:
        Dependency resolver.
    """
    return DependencyResolver(repo)


@lru_cache(maxsize=1)
def _get_clerk_client(secret_key: str) -> "Clerk":
    """Get cached Clerk client instance.

    Args:
        secret_key: Clerk secret key.

    Returns:
        Clerk client instance.
    """
    return Clerk(bearer_auth=secret_key)


# Authorized parties for JWT verification (origins that can use the API)
AUTHORIZED_PARTIES = [
    "https://simplecore.app",
    "https://registry.simplecore.app",
    "https://tauri.localhost",
    "http://tauri.localhost",
    "tauri://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://localhost:4025",
    "http://127.0.0.1:8080",
]


async def verify_clerk_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    settings: Settings = Depends(get_settings),
) -> Optional[dict]:
    """Verify Clerk JWT token using Clerk SDK.

    This function verifies JWT tokens issued by Clerk for CLI/API authentication.
    The JWT must be generated using the 'cli' template with audience 'tb-registry'.

    Args:
        credentials: HTTP authorization credentials (Bearer token).
        settings: Application settings containing Clerk secret key.

    Returns:
        Decoded token claims (payload) if valid, None if no token provided.

    Raises:
        HTTPException: If Clerk is not configured or token verification fails.
    """
    if not credentials:
        return None

    token = credentials.credentials

    # Check if Clerk is configured
    if not settings.clerk_secret_key:
        logger.warning("Clerk secret key not configured")
        if settings.debug:
            # In debug mode, return mock data for development
            logger.warning("Debug mode: returning mock user data")
            return {
                "sub": "user_debug",
                "email": "debug@example.com",
            }
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service not configured",
        )

    # Check if Clerk SDK is available
    if not CLERK_SDK_AVAILABLE:
        logger.error("Clerk SDK not installed. Install with: pip install clerk-backend-api")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable",
        )

    try:
        # Create httpx.Request for authenticate_request
        # The Clerk SDK expects an httpx.Request object
        fake_request = httpx.Request(
            method="GET",
            url="http://localhost/verify",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Get Clerk client (cached)
        clerk = _get_clerk_client(settings.clerk_secret_key)

        # Authenticate the request using Clerk SDK
        # This verifies the JWT signature using JWKS and validates claims
        request_state = clerk.authenticate_request(
            fake_request,
            AuthenticateRequestOptions(
                authorized_parties=AUTHORIZED_PARTIES,
                # audience="tb-registry",  # Uncomment when JWT template has audience set
            ),
        )

        if request_state.status == AuthStatus.SIGNED_IN:
            payload = request_state.payload or {}
            logger.debug(f"Token verified for user: {payload.get('sub')}")
            return payload
        else:
            # Token verification failed
            reason = getattr(request_state, "reason", "Unknown reason")
            logger.warning(f"Token verification failed: {reason}")
            return None

    except Exception as e:
        logger.error(f"Error verifying Clerk token: {e}")
        # Don't expose internal errors to client
        return None


async def get_current_user(
    token_data: Optional[dict] = Depends(verify_clerk_token),
    user_repo: UserRepository = Depends(get_user_repo),
) -> User:
    """Get current authenticated user.

    Args:
        token_data: Decoded token claims.
        user_repo: User repository.

    Returns:
        Current user.

    Raises:
        HTTPException: If not authenticated.
    """
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    clerk_user_id = token_data.get("sub")
    if not clerk_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    user = await user_repo.get_by_clerk_id(clerk_user_id)
    if not user:
        # Create user on first login
        user = User(
            id="",
            clerk_user_id=clerk_user_id,
            email=token_data.get("email", ""),
            username=token_data.get("username"),
        )
        user = await user_repo.create(user)

    return user


async def get_optional_user(
    token_data: Optional[dict] = Depends(verify_clerk_token),
    user_repo: UserRepository = Depends(get_user_repo),
) -> Optional[User]:
    """Get current user if authenticated.

    Args:
        token_data: Decoded token claims.
        user_repo: User repository.

    Returns:
        Current user or None.
    """
    if not token_data:
        return None

    clerk_user_id = token_data.get("sub")
    if not clerk_user_id:
        return None

    return await user_repo.get_by_clerk_id(clerk_user_id)


async def require_admin(
    user: User = Depends(get_current_user),
) -> User:
    """Require admin user.

    Args:
        user: Current user.

    Returns:
        Admin user.

    Raises:
        HTTPException: If not admin.
    """
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


async def require_publisher(
    user: User = Depends(get_current_user),
) -> User:
    """Require publisher user.

    Args:
        user: Current user.

    Returns:
        Publisher user.

    Raises:
        HTTPException: If not a publisher.
    """
    if not user.publisher_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Publisher registration required",
        )
    return user

