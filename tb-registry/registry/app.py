"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import create_router
from .config import Settings, get_settings
from .db.database import Database
from .exceptions import RegistryException
from .storage.manager import StorageManager

# Web assets paths
WEB_DIR = Path(__file__).parent / "web"
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

logger = logging.getLogger(__name__)


def create_lifespan(skip_storage: bool = False):
    """Create a lifespan context manager.

    Args:
        skip_storage: Whether to skip storage initialization (for tests).

    Returns:
        Lifespan context manager.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Application lifespan manager.

        Handles startup and shutdown events.

        Args:
            app: FastAPI application.

        Yields:
            None.
        """
        settings = get_settings()

        # Initialize database
        db = Database(settings.database_url)
        await db.initialize()
        app.state.db = db

        # Initialize storage manager (skip in test mode)
        storage = None
        if not skip_storage:
            storage = StorageManager(settings)
            await storage.start()
            app.state.storage = storage

        logger.info("TB Registry started")

        yield

        # Cleanup
        if storage:
            await storage.stop()
        await db.close()
        logger.info("TB Registry stopped")

    return lifespan


def create_app(
    settings: Settings | None = None,
    skip_storage: bool = False,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings override.
        skip_storage: Whether to skip storage initialization (for tests).

    Returns:
        Configured FastAPI application.
    """
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="TB Registry",
        description="Package registry for ToolBox Framework",
        version="0.1.0",
        lifespan=create_lifespan(skip_storage=skip_storage),
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add exception handlers
    @app.exception_handler(RegistryException)
    async def registry_error_handler(
        request: Request,
        exc: RegistryException,
    ) -> JSONResponse:
        """Handle registry errors.

        Args:
            request: FastAPI request.
            exc: Registry error.

        Returns:
            JSON error response.
        """
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": str(exc)},
        )

    # Add middleware for request state
    @app.middleware("http")
    async def add_request_state(request: Request, call_next):
        """Add storage and database to request state.

        Args:
            request: FastAPI request.
            call_next: Next middleware.

        Returns:
            Response.
        """
        # Set database if it exists
        if hasattr(app.state, "db"):
            request.state.db = app.state.db
        # Set storage if it exists
        if hasattr(app.state, "storage"):
            request.state.storage = app.state.storage
        response = await call_next(request)
        return response

    # Include API router
    router = create_router()
    app.include_router(router, prefix="/api/v1")

    # Mount static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Web UI routes
    @app.get("/", include_in_schema=False)
    async def index():
        """Serve the main page."""
        return FileResponse(TEMPLATES_DIR / "index.html")

    @app.get("/docs", include_in_schema=False)
    async def docs_page():
        """Serve the docs page."""
        return FileResponse(TEMPLATES_DIR / "docs.html")

    @app.get("/package/{name}", include_in_schema=False)
    async def package_page(name: str):
        """Serve package detail page (reuses index for now)."""
        return FileResponse(TEMPLATES_DIR / "index.html")

    return app


# Create default application instance
app = create_app()

