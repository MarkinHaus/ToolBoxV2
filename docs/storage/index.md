# Storage

ToolBoxV2 storage layer: four database modes, encrypted blob sharing, offline fallback.

## Database Modes

| Mode | Enum | Backend | Requires Config |
|------|------|---------|-----------------|
| `LC` | `LOCAL_DICT` | Local JSON file (`data.json`) | Nothing |
| `LR` | `LOCAL_REDIS` | Local Redis server | `redis.url` |
| `RR` | `REMOTE_REDIS` | Remote Redis server | `redis.url` or credentials |
| `CB` | `CLUSTER_BLOB` | Encrypted MinIO blob storage | `minio.*` config |

Set via manifest:

```bash
tb manifest set database.mode CB
```

## DB Module API (`toolboxv2/mods/DB/`)

The DB module (`mod_name = "DB"`) exposes these operations via the ToolBox dispatch system:

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `get` | `(query)` | `Result` | Retrieve data by key/query |
| `set` | `(query, data)` | `Result` | Store data |
| `if_exist` | `(query)` | `Result` | Check existence |
| `delete` | `(query, matching)` | `Result` | Remove entry |
| `append_on_set` | `(query, data)` | `Result` | Append instead of overwrite |
| `edit_programmable` | `(mode)` | — | Switch edit mode |
| `edit_cli` | `(mode)` | — | CLI edit mode |
| `edit_dev_web_ui` | `(mode)` | — | Web UI edit mode |
| `close_db` | `()` | `Result` | Close DB connection |

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/db/status` | GET | DB status (mode, size, health) |
| `/api/db/keys` | GET | List all keys |
| `/api/db/value/{key}` | GET | Get value by key |
| `/api/db/value` | POST | Set value |
| `/api/db/key/{key}` | DELETE | Delete key |
| `/api/db/mode` | POST | Change DB mode |

## BlobStorage (`toolboxv2/utils/extras/blobs.py`)

High-level encrypted blob storage with offline fallback and multi-server support.

### Key Classes

| Class | Description |
|-------|-------------|
| `BlobStorage` | Main storage facade — multi-server, encrypted, offline-capable |
| `BlobFile` | File-like interface for reading/writing blobs (`with BlobFile(...) as f:`) |
| `WatchManager` | Background thread polling for blob changes, dispatching callbacks |

### Blob Sharing

See [Blob Sharing API](blob_sharing_api.md) for share/revoke/list operations.

### Offline Fallback

BlobStorage caches blobs locally. When remote servers are unreachable:
1. Reads served from local cache
2. Writes queued and retried
3. `WatchManager` uses exponential backoff (max 60s)

---

## Sub-Pages

- [Blob Sharing API](blob_sharing_api.md) — share/revoke, access levels, Public User ID
- [BlobDB Reference](ref_blobdb.md) — internal BlobDB layer

## Related

- [Auth / MinIO Policy](../mods/CloudM/auth.md) — per-user MinIO credentials
- [Manifest Config](../runtime/config.md) — `database.*` keys
- [CloudM FolderSync](../mods/CloudM/folder_sync.md) — encrypted folder sync (deprecated)
