"""Health check routes."""

from typing import Optional

from fastapi import APIRouter, Depends, Request, Response

from registry.db.database import Database
from registry.storage.manager import StorageManager

router = APIRouter()


async def get_db_from_request(request: Request) -> Database:
    """Get database from request state.

    Args:
        request: FastAPI request.

    Returns:
        Database instance.
    """
    return request.state.db


async def get_storage_from_request(request: Request) -> Optional[StorageManager]:
    """Get storage from request state (optional).

    Args:
        request: FastAPI request.

    Returns:
        Storage manager instance or None if not available.
    """
    return getattr(request.state, "storage", None)


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint.

    Returns:
        Health status.
    """
    return {"status": "healthy"}


@router.get("/ready")
async def readiness_check(
    db: Database = Depends(get_db_from_request),
    storage: Optional[StorageManager] = Depends(get_storage_from_request),
) -> dict:
    """Readiness check endpoint.

    Checks database and storage connectivity.

    Args:
        db: Database instance.
        storage: Storage manager instance (optional).

    Returns:
        Readiness status with component health details.
    """
    errors = []

    # Check database connectivity
    db_healthy = await db.health_check()
    if not db_healthy:
        errors.append("database")

    # Check storage connectivity (if available)
    storage_status = None
    if storage is not None:
        storage_status = storage.health_check()
        if not storage_status["healthy"]:
            errors.append("storage")
    else:
        # Storage not configured (e.g., in test mode)
        storage_status = {"healthy": True, "note": "storage not configured"}

    if errors:
        return {
            "status": "not_ready",
            "errors": errors,
            "details": {
                "database": db_healthy,
                "storage": storage_status,
            },
        }

    return {
        "status": "ready",
        "details": {
            "database": db_healthy,
            "storage": storage_status,
        },
    }


@router.head("/health")
async def health_check_head() -> Response:
    """Health check HEAD endpoint.

    Returns:
        Empty response with 200 status.
    """
    return Response(status_code=200)

