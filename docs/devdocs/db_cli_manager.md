# DB CLI Manager (`utils/clis/db_cli_manager.py`)

> **File:** `toolboxv2/utils/clis/db_cli_manager.py` (~1080 Zeilen)
> **Typ:** Reference
> MinIO CLI-Management: Bucket-Policies, User-Credentials, Lifecycle-Rules.

## Why This Matters

Wenn `tb db` aufgerufen wird, interagiert `MinIOCLIManager` mit dem MinIO-Server um:
- Buckets zu erstellen/konfigurieren
- Pro-User MinIO-Credentials zu verwalten
- Lifecycle-Rules für Auto-Cleanup zu setzen
- Health-Checks durchzuführen

Dies ist die CLI-Schicht über dem [BlobStorage](../storage/ref_blobdb.md) Backend.

## API Reference

### MinIOCLIManager

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(endpoint, access_key, secret_key, secure=True)` | Connect to MinIO |
| `ensure_bucket` | `(bucket, region="us-east-1") → bool` | Create if missing |
| `set_bucket_policy` | `(bucket, policy_json)` | Set access policy |
| `create_user_credentials` | `(username) → (access_key, secret_key)` | Pro-user MinIO account |
| `delete_user_credentials` | `(username)` | Remove user |
| `list_users` | `() → List[Dict]` | All MinIO users |
| `set_lifecycle_rule` | `(bucket, days, prefix)` | Auto-expire objects |
| `health_check` | `() → bool` | Is MinIO reachable? |
| `get_bucket_info` | `(bucket) → Dict` | Size, object count, policy |

### How-to: Configure Per-User Buckets

```python
from toolboxv2.utils.clis.db_cli_manager import MinIOCLIManager

mgr = MinIOCLIManager("localhost:9000", "admin", "secret123")
mgr.ensure_bucket("user-private")
mgr.set_bucket_policy("user-private", policy_read_write)
creds = mgr.create_user_credentials("alice")
# → ("alice_AK...", "alice_SK...")
```

## Used By

- `tb db` CLI command
- [CloudM Auth](../mods/CloudM/auth.md) — per-user MinIO credentials
- [Storage Overview](../storage/index.md) — CB mode backend

## Related

- [Storage Overview](../storage/index.md) — DB modes
- [BlobDB Reference](../storage/ref_blobdb.md) — MobileDB layer
- [WorkerManager](cli_worker_manager.md) — similar CLI pattern
