# TB Registry

> **Location**: `tb-registry/` (separate repo/directory from ToolBoxV2 core)
> **Default Port**: 4025
> **Default URL**: `https://registry.simplecore.app`

The TB Registry is a **self-hostable package registry** for ToolBoxV2 mods, feature packs, widgets, and binary artifacts. It provides semantic versioning, dependency resolution, and multi-backend MinIO storage.

## Features

- Package management (mods, libraries, widgets, artifacts)
- Semantic versioning with dependency resolution (`POST /api/v1/resolve`)
- Multi-backend storage: MinIO primary + optional mirrors
- Publisher verification system
- **CloudM.Auth** JWT authentication (migrated from Clerk, 2026-02-25)
- ZIP security: path traversal protection, symlink validation, size + compression ratio limits

## Quick Start

```bash
# Install dependencies
uv sync

# Configure
cp .env.example .env
# Set: CLOUDM_JWT_SECRET, MINIO_PRIMARY_*, etc.

# Run development server
uv run tb-registry

# Or via ToolBoxV2:
tb -c CloudM.RegistryServer cli start --port 4025
```

## Authentication

The Registry uses **CloudM.Auth** JWT tokens (HS256). The same `CLOUDM_JWT_SECRET` must be set in both the ToolBoxV2 instance and the Registry server.

```bash
# Required
CLOUDM_JWT_SECRET=<same_secret_as_toolboxv2>

# Optional
CLOUDM_AUTH_URL=http://localhost:4025
DEBUG=False
```

Public packages are accessible without authentication. Private/unlisted packages and publishing require a valid CloudM.Auth token.

## API Reference

### Packages

```
GET    /api/v1/health                              → Health check
GET    /api/v1/packages                            → List packages
POST   /api/v1/packages                            → Create package (auth required)
GET    /api/v1/packages/{name}                     → Get package metadata
POST   /api/v1/packages/{name}/versions            → Upload new version (auth required)
GET    /api/v1/packages/{name}/versions/{version}  → Download version
DEL    /api/v1/packages/{name}                     → Delete package (admin)
```

### Artifacts

```
GET    /api/v1/artifacts                           → List artifacts
POST   /api/v1/artifacts                           → Upload artifact (auth required)
GET    /api/v1/artifacts/{name}/{version}          → Download artifact
```

### Resolution

```
POST   /api/v1/resolve                             → Resolve dependency tree
```

Request body:
```json
{
  "packages": [
    {"name": "my-mod", "version": "^1.0.0"}
  ]
}
```

## Registry Server Management (via ToolBoxV2)

The `CloudM.RegistryServer` module manages the Registry process from within ToolBoxV2:

```bash
tb -c CloudM.RegistryServer cli start           # Foreground
tb -c CloudM.RegistryServer cli start --bg      # Background (daemonized)
tb -c CloudM.RegistryServer cli stop            # Stop background process
tb -c CloudM.RegistryServer cli status          # PID + running status
tb -c CloudM.RegistryServer cli restart         # Restart background process
```

The server process is located via:
1. `TB_REGISTRY_PATH` environment variable
2. Default: `../tb-registry/` relative to ToolBoxV2 root
3. `~/tb-registry/`, `/opt/tb-registry/`

## Docker Deployment

```bash
docker build -t tb-registry .
docker run -p 4025:4025 --env-file .env tb-registry
```

## Self-Hosting Checklist

- [ ] Set `CLOUDM_JWT_SECRET` (same value as ToolBoxV2 instance)
- [ ] Configure MinIO primary storage
- [ ] Set `CLOUDM_AUTH_URL` if auth server is on a different host
- [ ] (Optional) Configure mirror storage
- [ ] Run `tb manifest validate` to check ToolBoxV2 side
- [ ] Point `cloudm.registry_url` in `tb-manifest.yaml` to your instance

## Related

- [Mod Manager](../mods/CloudM/mod_manager.md) — Client that talks to this registry
- [CloudM Auth](../mods/CloudM/auth.md) — Auth system used by the registry
- [Dev Registry Guide](../devdocs/dev_registry.md) — How to publish packages
