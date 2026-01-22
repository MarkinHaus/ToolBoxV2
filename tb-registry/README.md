# TB Registry

Package registry for ToolBox Framework.

## Features

- Package management (mods, libraries, widgets)
- Artifact distribution (installers, binaries)
- Semantic versioning with dependency resolution
- Multi-backend storage (MinIO primary + mirrors)
- Publisher verification system
- Clerk authentication integration

## Quick Start

```bash
# Install dependencies
uv sync

# Run development server
uv run tb-registry

# Run tests
uv run pytest
```

## API Endpoints

- `GET /api/v1/health` - Health check
- `GET /api/v1/packages` - List packages
- `POST /api/v1/packages` - Create package
- `GET /api/v1/packages/{name}` - Get package
- `POST /api/v1/packages/{name}/versions` - Upload version
- `GET /api/v1/artifacts` - List artifacts
- `POST /api/v1/resolve` - Resolve dependencies

## Configuration

Copy `.env.example` to `.env` and configure:

- `CLERK_SECRET_KEY` - Clerk authentication
- `MINIO_PRIMARY_*` - Primary storage
- `MINIO_MIRROR_*` - Optional mirror storage

## License

MIT

