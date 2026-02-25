"""FastAPI dependencies for dependency injection."""

import logging
import time
from functools import lru_cache
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from registry.auth.cloudm_client import CloudMAuthClient, TokenPayload
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


async def get_storage(request: Request) -> Optional[StorageManager]:
    """Get storage manager from request state.

    Args:
        request: FastAPI request.

    Returns:
        Storage manager or None (e.g., in test mode without storage).
    """
    return getattr(request.state, "storage", None)


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
    storage: Optional[StorageManager] = Depends(get_storage),
) -> PackageService:
    """Get package service.

    Args:
        repo: Package repository.
        user_repo: User repository.
        storage: Storage manager (optional, for test mode).

    Returns:
        Package service.
    """
    return PackageService(repo, user_repo, storage)


async def get_artifact_service(
    repo: ArtifactRepository = Depends(get_artifact_repo),
    user_repo: UserRepository = Depends(get_user_repo),
    storage: Optional[StorageManager] = Depends(get_storage),
) -> ArtifactService:
    """Get artifact service.

    Args:
        repo: Artifact repository.
        user_repo: User repository.
        storage: Storage manager (optional, for test mode).

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


async def get_diff_generator(
    repo: PackageRepository = Depends(get_package_repo),
    storage: Optional[StorageManager] = Depends(get_storage),
) -> "registry.diff.DiffGenerator":
    """Get diff generator for creating package diffs.

    Args:
        repo: Package repository.
        storage: Optional storage manager for patch files.

    Returns:
        Diff generator instance.
    """
    from registry.diff import DiffGenerator

    return DiffGenerator(repo, storage)


@lru_cache(maxsize=1)
def _get_cloudm_client(jwt_secret: str) -> CloudMAuthClient:
    """Get cached CloudM.Auth client instance.

    Args:
        jwt_secret: JWT secret for validation.

    Returns:
        CloudM.Auth client instance.
    """
    return CloudMAuthClient(jwt_secret=jwt_secret)


async def verify_cloudm_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    settings: Settings = Depends(get_settings),
) -> Optional[TokenPayload]:
    """Verify CloudM.Auth JWT token.

    This function verifies JWT tokens issued by CloudM.Auth module.
    Tokens are signed with HS256 algorithm using shared secret.

    Args:
        credentials: HTTP authorization credentials (Bearer token).
        settings: Application settings containing JWT secret.

    Returns:
        TokenPayload if valid, None if no token provided.

    Raises:
        HTTPException: If token verification fails and not in debug mode.
    """
    if not credentials:
        return None

    token = credentials.credentials

    # Check if JWT secret is configured
    if not settings.cloudm_jwt_secret:
        logger.warning("CloudM JWT secret not configured")
        if settings.debug:
            # In debug mode, return mock data for development
            logger.warning("Debug mode: returning mock user data")
            return TokenPayload(
                user_id="user_debug",
                username="debug_user",
                email="debug@example.com",
                level=1,
                provider="debug",
                exp=int(time.time()) + 3600,
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service not configured",
        )

    try:
        # Get CloudM.Auth client (cached)
        client = _get_cloudm_client(settings.cloudm_jwt_secret)

        # Validate token
        payload = client.validate_token(token)
        if payload:
            logger.debug(f"Token verified for user: {payload.user_id}")
            return payload
        else:
            # Token verification failed
            logger.warning("Token verification failed")
            return None

    except Exception as e:
        logger.error(f"Error validating CloudM.Auth token: {e}")
        # Don't expose internal errors to client
        return None


async def get_current_user(
    token_data: Optional[TokenPayload] = Depends(verify_cloudm_token),
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

    cloudm_user_id = token_data.user_id  # Changed from token_data.get("sub")
    if not cloudm_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    user = await user_repo.get_by_cloudm_id(cloudm_user_id)
    if not user:
        # Create user on first login
        user = User(
            cloudm_user_id=cloudm_user_id,  # Changed
            email=token_data.email,
            username=token_data.username,
        )
        user = await user_repo.create(user)

    return user


async def get_optional_user(
    token_data: Optional[TokenPayload] = Depends(verify_cloudm_token),
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

    cloudm_user_id = token_data.user_id
    if not cloudm_user_id:
        return None

    return await user_repo.get_by_cloudm_id(cloudm_user_id)


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

