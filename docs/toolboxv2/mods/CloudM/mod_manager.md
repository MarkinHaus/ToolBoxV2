# CloudM — Mod Manager

> **File**: `ModManager.py`

Handles the full lifecycle of ToolBoxV2 packages (mods, feature packs, widgets): discovery, installation, updates, and removal — with optional integration into the [TB Registry](../../registry/index.md).

## Package Types

| Type | Description |
|------|-------------|
| `mod` | Core module (Python, adds `tb` commands) |
| `feature_pack` | ZIP-based feature bundle (lazy-loaded) |
| `widget` | tbjs frontend component |
| `artifact` | Binary / installer file |

## Core Operations

### Install a Package

```bash
# From registry
tb -c CloudM ModManager install my-mod

# Specific version
tb -c CloudM ModManager install my-mod --version 1.2.0

# From local ZIP
tb -c CloudM ModManager install ./my-mod.zip --local
```

### Update

```bash
tb -c CloudM ModManager update my-mod
tb -c CloudM ModManager update --all
```

### Remove

```bash
tb -c CloudM ModManager remove my-mod
```

### List / Search

```bash
tb -c CloudM ModManager list                  # Installed packages
tb -c CloudM ModManager search "calendar"     # Search registry
tb -c CloudM ModManager info my-mod           # Package details
```

## Registry Integration

ModManager uses the `RegistryClient` (via `module.py`) to interact with the [TB Registry](../../registry/index.md):

```python
# Lazy initialized on first use
client = cloudm_instance.registry   # RegistryClient
```

Registry URL is configurable:

```bash
tb manifest set cloudm.registry_url https://registry.simplecore.app
```

### Authentication

Registry auth uses the current session token (from `CloudM.Auth`). Without auth, only public packages are accessible.

```python
await cloudm_instance.ensure_registry_auth()
```

## Snapshot System

Each installed mod has a version snapshot stored via `find_highest_zip_version`. Used for rollback and integrity checks.

```python
snapshot = cloudm_instance.get_mod_snapshot("my-mod")
# Returns: "1.2.0" or None
```

## Publisher Verification

Package publishers can be verified against the registry. Unverified publishers trigger a warning during install. Admin-only publishers require elevated session permissions.

## Configuration

Stored in `modules.config` (via `FileHandler`):

| Key | Default | Description |
|-----|---------|-------------|
| `URL` | `https://simpelm.com/api` | Legacy API URL |
| `REGISTRY_URL` | `https://registry.simplecore.app` | TB Registry URL |
| `TOKEN` | `~tok~` | Auth token (auto-populated from session) |

## Related

- [Registry](../../registry/index.md) — The TB Registry server ModManager talks to
- [Feature Packs](../../devdocs/dev_features.md) — ZIP-based feature bundles
- [Registry Publishing](../../devdocs/dev_registry.md) — How to publish packages
