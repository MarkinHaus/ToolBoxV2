# Rust Database Manager

A high-performance, persistent database manager with Redis-JSON and RocksDB integration. This library provides efficient data storage capabilities with built-in memory optimization, redundant storage, and Python bindings.

## Features

- Redis-JSON for structured data storage
- RocksDB for efficient blob data storage
- Memory usage optimization with LRU caching
- Disk space management
- Reed-Solomon error correction for data integrity
- Distributed blob storage across devices
- Python bindings for easy integration

## Usage from Python

### Basic Database Operations

```python
from toolboxv2 import get_app

# Initialize the database
db = get_app().get_mod("DB")
db.edit_cli("CB")

# Set a value
db.set("my_key", "my_value")

# Get a value
value = db.get("my_key")

# Check if a key exists
exists = db.if_exist("my_key")

# Delete a key
db.delete("my_key")

# Clean up resources
db.exit()
```

### Blob Storage

```python
from toolboxv2.utils.extras.blobs import BlobFile, BlobStorage

# Initialize blob storage
storage = BlobStorage()

# Create a blob
blob_id = storage.create_blob(b"This is my blob data")

# Read a blob
data = storage.read_blob(blob_id)

# Update a blob
storage.update_blob(blob_id, b"Updated data")

# Delete a blob
storage.delete_blob(blob_id)

# Link blobs for redundancy
storage.chair_link([blob_id1, blob_id2, blob_id3])
```

### File-like Interface

```python
from toolboxv2.utils.extras.blobs import BlobFile, BlobStorage

# Write to a file
with BlobFile("blob_id/folder/file.txt", "w") as f:
    f.write("Hello, world!")

# Read from a file
with BlobFile("blob_id/folder/file.txt", "r") as f:
    content = f.read()

# JSON operations
with BlobFile("blob_id/folder/data.json", "w") as f:
    f.write_json({"name": "Test", "values": [1, 2, 3]})

with BlobFile("blob_id/folder/data.json", "r") as f:
    data = f.read_json()
```

## Performance Considerations

- The library automatically optimizes memory usage based on configured limits
- Least recently used data is compressed or offloaded to disk when memory limits are reached
- Data distribution across devices happens automatically in the background
- Reed-Solomon error correction ensures data integrity even with partial data loss

## License
