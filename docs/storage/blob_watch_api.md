# Blob Watch API Documentation

## Overview

The Blob Watch API enables real-time monitoring of blob changes with automatic callback execution. It provides a non-blocking, thread-safe way to react to blob updates.

## Features

✅ **Non-Blocking**: Runs in background thread, doesn't block your application  
✅ **Batch Updates**: Multiple watches run in a single thread for efficiency  
✅ **Auto-Cleanup**: Callbacks auto-remove after configurable idle timeout  
✅ **Thread-Safe**: Safe to use from multiple threads  
✅ **Easy to Use**: Simple callback-based API  

---

## API Reference

### BlobStorage.watch()

Register a callback for blob changes.

```python
def watch(
    blob_id: str,
    callback: Callable[[BlobFile], None],
    max_idle_timeout: int = 600,
    threaded: bool = True
)
```

**Parameters:**
- `blob_id` (str): The blob to watch for changes
- `callback` (Callable): Function called when blob changes, receives `BlobFile` object
- `max_idle_timeout` (int): Seconds without updates before auto-removing callback (default: 600 = 10 min)
- `threaded` (bool): If True, runs in background thread (default: True)

**Example:**
```python
def on_change(blob_file: BlobFile):
    data = blob_file.read_json()
    print(f"Updated: {data}")

storage.watch('app/config.json', on_change, max_idle_timeout=600)
```

---

### BlobStorage.stop_watch()

Stop watching a blob.

```python
def stop_watch(
    blob_id: str,
    callback: Optional[Callable] = None
)
```

**Parameters:**
- `blob_id` (str): The blob to stop watching
- `callback` (Optional[Callable]): Specific callback to remove, or None to remove all

**Examples:**
```python
# Stop specific callback
storage.stop_watch('app/config.json', on_change)

# Stop all callbacks for a blob
storage.stop_watch('app/config.json')
```

---

### BlobStorage.stop_all_watches()

Stop all active watches and shutdown the watch thread.

```python
def stop_all_watches()
```

**Example:**
```python
# Cleanup on application shutdown
storage.stop_all_watches()
```

---

### BlobFile.watch()

Convenience method to watch a specific BlobFile instance.

```python
def watch(
    callback: Callable[[BlobFile], None],
    max_idle_timeout: int = 600,
    threaded: bool = True
)
```

**Example:**
```python
config_file = BlobFile('app/config.json', 'r')

def on_change(blob_file: BlobFile):
    print("Config changed!")

config_file.watch(on_change)
```

---

### BlobFile.stop_watch()

Stop watching this BlobFile instance.

```python
def stop_watch(callback: Optional[Callable] = None)
```

**Example:**
```python
config_file.stop_watch(on_change)
```

---

## How It Works

### Architecture

1. **WatchManager**: Manages all watch operations in a background thread
2. **Long-Polling**: Uses server's `/watch` endpoint (60s timeout)
3. **Batch Processing**: One thread handles all watches efficiently
4. **Callback Dispatch**: When blob changes, all registered callbacks are called
5. **Auto-Cleanup**: Callbacks without updates for `max_idle_timeout` are removed

### Flow Diagram

```
┌─────────────┐
│ Application │
└──────┬──────┘
       │ watch(blob_id, callback)
       ▼
┌─────────────────┐
│  WatchManager   │◄─── Background Thread
└────────┬────────┘
         │ Long-Polling /watch
         ▼
┌─────────────────┐
│  Blob Server    │
└────────┬────────┘
         │ blob_id changed
         ▼
┌─────────────────┐
│ Dispatch        │
│ Callbacks       │
└────────┬────────┘
         │ callback(BlobFile)
         ▼
┌─────────────────┐
│ Your Callback   │
└─────────────────┘
```

---

## Best Practices

### 1. Use Appropriate Timeouts

```python
# Short-lived watches (e.g., UI updates)
storage.watch('ui/state.json', on_ui_change, max_idle_timeout=60)

# Long-lived watches (e.g., config monitoring)
storage.watch('app/config.json', on_config_change, max_idle_timeout=3600)
```

### 2. Handle Errors in Callbacks

```python
def safe_callback(blob_file: BlobFile):
    try:
        data = blob_file.read_json()
        process_data(data)
    except Exception as e:
        logger.error(f"Callback error: {e}")
```

### 3. Cleanup on Shutdown

```python
import atexit

storage = BlobStorage(servers=['http://localhost:3000'])
atexit.register(storage.stop_all_watches)
```

### 4. Multiple Callbacks per Blob

```python
# You can register multiple callbacks for the same blob
storage.watch('data.json', callback_1, max_idle_timeout=300)
storage.watch('data.json', callback_2, max_idle_timeout=600)

# Each callback has independent timeout
```

---

## Performance Considerations

- **Single Thread**: All watches run in one thread, very efficient
- **Batch Updates**: Server sends one notification, all callbacks triggered
- **Memory**: Minimal overhead, only stores callback references
- **Network**: One long-polling connection per server

---

## Troubleshooting

### Callback Not Called

1. Check if blob is actually changing on server
2. Verify API key is valid
3. Check logs for errors
4. Ensure watch wasn't auto-removed due to timeout

### High CPU Usage

- Should not happen - watch thread sleeps during long-polling
- Check for errors in callbacks causing rapid retries

### Memory Leaks

- Always call `stop_watch()` or `stop_all_watches()` when done
- Use `max_idle_timeout` to auto-cleanup stale watches

