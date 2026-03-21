# CloudM — Folder Sync

> **File**: `FolderSync.py`
> **Requires**: `pip install watchdog`

Bidirectional, encrypted folder synchronization via MinIO. Designed for syncing `USER_PRIVATE` data across devices without exposing plaintext to the server.

## Features

- **Live sync** via Watchdog filesystem events
- **Batch processing** for performance (event debouncing)
- **Client-side encryption** — AES + zlib compression before upload
- **Local SQLite index** for fast diff calculation (no full scan on start)
- **Share-token system** for pairing devices without exposing credentials

## Architecture

```
Local Folder
    │
    ├── Watchdog Observer       ← Detects file changes in real time
    │       │
    │       ▼
    ├── Batch Queue             ← Debounces rapid changes
    │       │
    │       ▼
    ├── Diff Engine             ← Compares against SQLite local index
    │       │
    │       ▼
    ├── Encrypt + Compress      ← zlib → AES (client-side, key never leaves device)
    │       │
    │       ▼
    └── MinIO Upload/Download   ← MinIOManager (MinIOConfig)
```

## Quick Start

```python
from toolboxv2.mods.CloudM.FolderSync import FolderSync

sync = FolderSync(
    local_path="/home/user/notes",
    minio_bucket="user-private",
    minio_prefix="user_abc/notes",
    encryption_key=b"<32-byte-key>",   # Never sent to server
)

# Start live sync
await sync.start()

# Stop
await sync.stop()
```

## CLI Commands

```bash
tb -c CloudM FolderSync start --path ~/notes --bucket user-private
tb -c CloudM FolderSync stop
tb -c CloudM FolderSync status
tb -c CloudM FolderSync force-sync        # Manual full sync
tb -c CloudM FolderSync generate-token    # Create share token for pairing
```

## Share Token Pairing

```bash
# Device A (already syncing):
tb -c CloudM FolderSync generate-token
# → Token: ABCD-1234-EFGH (valid 5 min)

# Device B:
tb -c CloudM FolderSync pair --token ABCD-1234-EFGH
# → Downloads encryption key + MinIO config, starts syncing
```

The share token **includes** the encryption key — share it only over a secure channel.

## Encryption Details

- Algorithm: AES (via `toolboxv2.utils.security.cryp.Code`)
- Compression: zlib applied before encryption (reduces bandwidth)
- Key storage: local only, never persisted on server
- Each file is encrypted independently

## Local Index

A SQLite database tracks the local state:

```
~/.local/share/ToolBoxV2/.data/foldersync/{bucket_hash}.db
```

Contains: file path, hash, size, last_modified, sync_status.

## Configuration

| Parameter | Description |
|-----------|-------------|
| `local_path` | Local directory to sync |
| `minio_bucket` | Target MinIO bucket |
| `minio_prefix` | Key prefix within bucket (usually `user_id/folder_name`) |
| `encryption_key` | 32-byte AES key |
| `batch_interval` | Event debounce interval in ms (default: 500) |
| `max_workers` | ThreadPoolExecutor size (default: 4) |

## Dependencies

```bash
pip install watchdog
```

Watchdog is optional — without it, FolderSync falls back to **polling mode** (less efficient, checks every 30s).

## Related

- [User Data API](user_data.md) — `USER_PRIVATE` scope uses FolderSync for cloud backup
- [BlobDB Reference](../../storage/ref_blobdb.md) — MinIO backend
- [Auth / MinIO Policy](auth.md) — Per-user MinIO credentials
