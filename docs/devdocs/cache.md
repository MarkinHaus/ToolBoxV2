# Cache (`utils/system/cache.py`)

> **File:** `toolboxv2/utils/system/cache.py` (~59 Zeilen)
> **Typ:** Reference
> Simple persistente Caches: `FileCache` (shelve) + `MemoryCache` (dict).

## API Reference

### FileCache

Disk-based cache using Python's `shelve` module. Persists across restarts.

```python
from toolboxv2.utils.system.cache import FileCache

cache = FileCache(folder="./.cache/", filename="mydata")
cache.set("user_count", 42)
val = cache.get("user_count")  # → 42
cache.cleanup()  # Delete cache files
```

| Method | Description |
|--------|-------------|
| `get(key)` | Get value (or `None`) |
| `set(key, value)` | Store value |
| `cleanup()` | Delete cache files + folder |

### MemoryCache

In-memory cache with TTL. Lost on restart.

```python
from toolboxv2.utils.system.cache import MemoryCache

mc = MemoryCache(max_size=100, ttl=300)  # 100 items, 5min TTL
mc.set("temp", "data")
val = mc.get("temp")  # → "data" (or None after TTL)
```

| Method | Description |
|--------|-------------|
| `get(key)` | Get if not expired |
| `set(key, value)` | Store with TTL |
| `clear()` | Wipe all entries |
| `size()` | Current item count |

## Used By

- `@app.tb(memory_cache=True)` decorator parameter
- BlobStorage local fallback caching

## Related

- [Core Types](types.md) — `@tb(memory_cache=True)` parameter
- [BlobStorage](../storage/ref_blobdb.md) — local blob cache
