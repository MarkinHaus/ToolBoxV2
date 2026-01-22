"""Entry point for running the registry server."""

import uvicorn

from registry.config import get_settings


def main() -> None:
    """Run the registry server."""
    settings = get_settings()

    uvicorn.run(
        "registry.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    main()

