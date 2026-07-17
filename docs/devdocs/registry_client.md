# RegistryClient (`utils/extras/registry_client.py`)

> **File:** `toolboxv2/utils/extras/registry_client.py` (~1904 Zeilen)
> **Typ:** Reference + Explanation
> TB-Registry Client — Package-Management (install, update, search, publish).

## Why This Matters

Der RegistryClient ist das Rückgrat des Mod-Systems. Wenn ein User `tb -c CloudM ModManager install mymod` ausführt, ist der RegistryClient derjenige, der:
1. Den Registry-Server kontaktiert
2. Paket-Metadaten validiert
3. Den Download durchführt
4. Die Integrität verifiziert (SHA-256 Checksum)
5. Das Paket entpackt und installiert

```mermaid
sequenceDiagram
    participant U as User (tb CLI)
    participant MM as ModManager
    participant RC as RegistryClient
    participant R as Registry Server
    participant FS as Local Filesystem

    U→>MM: install "mymod"
    MM→>RC: get_package_info("mymod")
    RC→>R: GET /api/package/mymod
    R-->>RC: PackageDetail (versions, checksum)
    RC→>R: GET /download/mymod/1.2.3
    R-->>RC: .tbz2 archive
    RC→>RC: verify_checksum(SHA-256)
    RC→>FS: extract to toolboxv2/mods/mymod/
    RC-->>MM: Success + install_path
    MM-->>U: "mymod 1.2.3 installed"
```

## Architecture

### Key Classes

| Class | Lines | Responsibility |
|-------|-------|----------------|
| `RegistryClient` | 209–1904 | Main client: auth, search, download, publish, version-mgmt |
| `PackageDetail` | dataclass | Package metadata (name, versions, description, author) |
| `VersionDetail` | dataclass | Single version info (version, checksum, size, deps) |
| `UserInfo` | dataclass | Publisher identity |

### Exceptions

| Exception | When |
|-----------|------|
| `RegistryError` | Base error for all registry operations |
| `RegistryConnectionError` | Server unreachable / timeout |
| `DownloadError` | Download failed or checksum mismatch |
| `VersionNotFoundError` | Requested version doesn't exist |

## API Reference

### Connection & Auth

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(registry_url, auth_token=None, timeout=30)` | Initialize client |
| `set_auth_token` | `(token: str)` | Set bearer token |
| `login` | `(username, password) → token` | Authenticate and cache token |
| `health_check` | `() → bool` | Check if registry is reachable |

### Package Discovery

| Method | Signature | Description |
|--------|-----------|-------------|
| `search` | `(query: str, limit=20) → List[PackageDetail]` | Full-text search |
| `get_package_info` | `(name: str) → PackageDetail` | Get all versions + metadata |
| `get_latest_version` | `(name: str) → VersionDetail` | Get newest version |
| `list_versions` | `(name: str) → List[str]` | List all version strings |
| `get_dependencies` | `(name, version) → List[str]` | Resolve dependency tree |

### Install & Download

| Method | Signature | Description |
|--------|-----------|-------------|
| `download_package` | `(name, version, dest_dir) → Path` | Download + verify + extract |
| `install_package` | `(name, version=None) → str` | Download + install to mods/ |
| `verify_checksum` | `(filepath, expected_sha256) → bool` | SHA-256 verification |
| `extract_archive` | `(archive_path, dest) → Path` | Extract .tbz2 archive |

### Publish & Manage

| Method | Signature | Description |
|--------|-----------|-------------|
| `publish_package` | `(package_path, metadata) → str` | Upload new package version |
| `unpublish_version` | `(name, version) → bool` | Remove a version |
| `get_my_packages` | `() → List[PackageDetail]` | List own packages |
| `update_package_info` | `(name, description, tags) → bool` | Edit metadata |

### Version Management

| Method | Signature | Description |
|--------|-----------|-------------|
| `resolve_version` | `(name, version_str) → str` | Resolve "latest", "^1.0" etc. |
| `compare_versions` | `(v1, v2) → int` | Semantic version compare |

## How-to: Install a Mod

```python
from toolboxv2.utils.extras.registry_client import RegistryClient

client = RegistryClient(registry_url="https://registry.toolbox.dev")
client.health_check()  # → True

# Install latest version
path = client.install_package("MyMod")
# → "toolboxv2/mods/MyMod/"

# Install specific version
path = client.install_package("MyMod", version="1.2.0")
```

## How-to: Publish a Mod

```python
client = RegistryClient("https://registry.toolbox.dev", auth_token="my-token")

result = client.publish_package(
    package_path="./MyMod-1.0.0.tbz2",
    metadata={
        "name": "MyMod",
        "version": "1.0.0",
        "description": "Does cool things",
        "author": "dev@example.com",
    }
)
# → returns version_url
```

## Common Pitfalls

- **Checksum mismatch**: If download is corrupted, `DownloadError` is raised. Registry caches are versioned — clearing local cache (`~/.local/share/ToolBoxV2/.cache/registry/`) fixes stale-state issues.
- **Auth token expiry**: Tokens expire. Re-login or use `set_auth_token` with a fresh token.
- **Circular dependencies**: `get_dependencies` detects cycles and raises `RegistryError`.

## Used By

- [Mod Manager](../mods/CloudM/mod_manager.md) — `tb -c CloudM ModManager install/update/search`
- [CLI Registry](../services/cli.md) — `tb registry info/search`

## Related

- [Core Types](types.md) — `Result` return type
- [CloudM ModManager](../mods/CloudM/mod_manager.md)
