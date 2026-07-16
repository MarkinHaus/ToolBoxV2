# CloudM — User Data API

> **File**: `UserDataAPI.py`

Unified interface for mod-to-mod data access with scoped, permission-controlled storage. Every piece of user data has an explicit scope that determines who can read and write it.

## Storage Scopes

| Scope | Read | Write | Storage |
|-------|------|-------|---------|
| `PUBLIC_READ` | Everyone | Admin only | TBEF.DB + cache |
| `PUBLIC_RW` | Everyone | Everyone | TBEF.DB + cache |
| `USER_PUBLIC` | Everyone | Owner only (own prefix) | TBEF.DB + cache |
| `USER_PRIVATE` | Owner only | Owner only | Local filesystem (encrypted) |
| `SERVER_SCOPE` | Server processes | Server processes | TBEF.DB |
| `MOD_DATA` | Mod-specific | Mod-specific | TBEF.DB |

### USER_PRIVATE

Private data is stored **locally** (never uploaded to cloud in plaintext). If cloud sync is enabled (via [FolderSync](folder_sync.md)), it is encrypted client-side before upload.

## Permission Levels

| Level | Value | Access |
|-------|-------|--------|
| Guest | 0 | `PUBLIC_READ`, `PUBLIC_RW` only |
| User | 1 | Own `USER_PUBLIC` + `USER_PRIVATE` |
| Moderator | 5 | Other users' `USER_PUBLIC` |
| Admin | 10 | All scopes |

## Core API

### Reading Data

```python
from toolboxv2.mods.CloudM.UserDataAPI import get_user_data

result = await get_user_data(
    app=app,
    requesting_user_id="user_abc",
    requesting_user_level=1,
    target_user_id="user_abc",
    scope="USER_PRIVATE",
    key="preferences",
)
```

### Writing Data

```python
from toolboxv2.mods.CloudM.UserDataAPI import set_user_data

result = await set_user_data(
    app=app,
    requesting_user_id="user_abc",
    requesting_user_level=1,
    target_user_id="user_abc",
    scope="USER_PRIVATE",
    key="preferences",
    value={"theme": "dark"},
)
```

### Mod-to-Mod Access

Mods can request access to data from other mods by declaring a permission:

```python
result = await get_mod_data(
    app=app,
    requesting_mod="MyMod",
    target_mod="CloudM",
    scope="MOD_DATA",
    key="shared_config",
)
```

## HTTP Endpoints

```
GET  /CloudM/userdata/{user_id}/{scope}/{key}     → Read data
POST /CloudM/userdata/{user_id}/{scope}/{key}     → Write data
DEL  /CloudM/userdata/{user_id}/{scope}/{key}     → Delete key
GET  /CloudM/userdata/{user_id}/{scope}           → List keys in scope
```

## Audit Log

All cross-user data accesses are logged to `TBEF.DB` under:
```
AUDIT::{requesting_user_id}::{target_user_id}::{timestamp}
```

Contains: accessor, target, scope, key, action (read/write/delete), and result.

## MinIO Policy Integration

For `USER_PRIVATE` data with cloud sync enabled, per-user MinIO policies are enforced via `CredentialBroker` (from `auth/minio_policy.py`). Each user gets scoped credentials with read/write access only to their own MinIO prefix.

## Caching

Non-private scopes use an in-process LRU cache. Cache is invalidated on write. Cache keys include `user_id + scope + key`.

## Related

- [Auth System](auth.md) — Provides user identity + permission level
- [Folder Sync](folder_sync.md) — Cloud sync for USER_PRIVATE data
- [BlobDB Reference](../../storage/ref_blobdb.md) — Underlying storage backend
