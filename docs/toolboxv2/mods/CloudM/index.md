# CloudM — Module Overview

CloudM is the **user management, authentication, and data layer** of ToolBoxV2. It handles everything from identity (OAuth2, Passkeys, Magic Links) to per-user storage, module management, and folder synchronization.

## Sub-Modules

| Module | File | Purpose |
|--------|------|---------|
| [Auth System](auth.md) | `Auth.py` + `auth/` | OAuth2, Passkeys, JWT, Magic Links |
| [Login System](login_system.md) | `LogInSystem.py` | CLI + Web session management |
| [User Data API](user_data.md) | `UserDataAPI.py` | Scoped per-user data storage |
| [Mod Manager](mod_manager.md) | `ModManager.py` | Package install, update, registry |
| [Folder Sync](folder_sync.md) | `FolderSync.py` | Encrypted bidirectional MinIO sync |
| Registry Server | `RegistryServer.py` | TB-Registry process management |
| Admin Dashboard | `AdminDashboard.py` | Server admin UI |
| User Dashboard | `UserDashboard.py` | Per-user dashboard API |
| Dashboard API | `DashboardAPI.py` | Dashboard data endpoints |
| User Account Manager | `UserAccountManager.py` | Account lifecycle |
| User Instances | `UserInstances.py` | Multi-instance user sessions |
| Email Services | `email_services.py` | Magic link + notification emails |
| Extras | `extras.py` | Utility functions |

## Architecture

```
CloudM
├── Auth (auth/)           ← Identity layer (OAuth2, Passkeys, JWT, Magic Link)
│   ├── api_oauth.py       ← Discord, Google OAuth callbacks
│   ├── api_passkey.py     ← WebAuthn registration + login
│   ├── api_magic_link.py  ← Email magic links + device invite codes
│   ├── api_session.py     ← Session validate, refresh, logout
│   ├── jwt_tokens.py      ← HS256 JWT generation + validation
│   └── user_store.py      ← User CRUD via TBEF.DB
│
├── LogInSystem.py         ← CLI session management (BlobFile + TBEF.DB)
├── UserDataAPI.py         ← Scoped storage (PUBLIC_READ / USER_PRIVATE / MOD_DATA)
├── ModManager.py          ← Package management + registry integration
├── FolderSync.py          ← Bidirectional encrypted MinIO sync (Watchdog)
│
├── AdminDashboard.py      ← Admin endpoints + UI
├── UserDashboard.py       ← User-facing dashboard
└── module.py              ← CloudM root (registry client, FileHandler init)
```

## DB Namespaces

All state is stored in `TBEF.DB` — no in-memory globals, multi-worker safe:

| Namespace | Content | TTL |
|-----------|---------|-----|
| `AUTH_USER::{user_id}` | User profile | Permanent |
| `AUTH_USER_PROVIDER::{provider}::{id}` | Provider → User index | Permanent |
| `AUTH_USER_EMAIL::{email}` | Email → User index | Permanent |
| `AUTH_STATE::{state}` | OAuth CSRF state | 10 min |
| `AUTH_CHALLENGE::{challenge}` | WebAuthn challenge | 5 min |
| `AUTH_BLACKLIST::{jti}` | Token blacklist | Until expiry |
| `AUTH_MAGIC_LINK::{token}` | Magic link token | 10 min |
| `AUTH_DEVICE_INVITE::{code}` | Device invite code | 5 min |

## Configuration

CloudM uses `manifest_config.yaml` for module-level config and respects the global `tb-manifest.yaml` for service settings (database mode, MinIO endpoint, worker ports).

See [Manifest Reference](../../manifest/ref_manifest.md) for full schema.

## Related

- [TB Registry](../../registry/index.md) — Package registry CloudM integrates with
- [Storage Reference](../../storage/ref_blobdb.md) — BlobDB backing UserDataAPI
- [Auth Migration Report](auth.md#migration-clerk-to-cloudmauth) — Clerk → CloudM.Auth
