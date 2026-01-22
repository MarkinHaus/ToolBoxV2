"""Main API router configuration."""

from fastapi import APIRouter

from registry.api.routes import (
    artifacts,
    auth,
    health,
    packages,
    publishers,
    resolve,
    search,
)


def create_router() -> APIRouter:
    """Create the main API router with all routes.

    Returns:
        Configured APIRouter.
    """
    router = APIRouter()

    # Health check routes
    router.include_router(
        health.router,
        tags=["health"],
    )

    # Auth routes
    router.include_router(
        auth.router,
        prefix="/auth",
        tags=["auth"],
    )

    # Package routes
    router.include_router(
        packages.router,
        prefix="/packages",
        tags=["packages"],
    )

    # Artifact routes
    router.include_router(
        artifacts.router,
        prefix="/artifacts",
        tags=["artifacts"],
    )

    # Publisher routes
    router.include_router(
        publishers.router,
        prefix="/publishers",
        tags=["publishers"],
    )

    # Search routes
    router.include_router(
        search.router,
        prefix="/search",
        tags=["search"],
    )

    # Resolve routes
    router.include_router(
        resolve.router,
        prefix="/resolve",
        tags=["resolve"],
    )

    return router

