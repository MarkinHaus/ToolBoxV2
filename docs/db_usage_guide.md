# ToolBoxV2 Database (DB) Module Guide

This guide provides a comprehensive overview of how to use the ToolBoxV2 `DB` module. It is designed for both human developers and AI agents to understand how to perform persistent key-value storage within the framework.

## 1. Core Concepts

The `DB` module acts as a standardized abstraction layer for various key-value storage backends. This design allows you to write your application logic once, using a consistent API (`get`, `set`, `delete`, etc.), while the underlying storage engine can be swapped out with minimal configuration changes.

### Key Features:

*   **Unified API**: A simple, consistent set of functions for all database operations.
*   **Multiple Backends (Modes)**: Supports different storage mechanisms, from simple local files to distributed blobs and Redis caches.
*   **Automatic Encoding**: Data is automatically handled, allowing you to work with standard Python types (strings, dicts, lists) without manual serialization.
*   **Environment-Driven Configuration**: Database modes and credentials can be configured via environment variables, allowing for easy switching between development, testing, and production setups.

### Database Modes

The module can operate in several modes. The default mode is **`CLUSTER_BLOB`**, which is the recommended, most robust option for production environments within the ToolBoxV2 ecosystem.

*   **`CLUSTER_BLOB` (CB)**: The default and recommended mode. It uses the `BlobDB` backend, which stores data as encrypted blobs in the configured root storage. This is ideal for user-specific, secure, and distributed data storage.
*   **`LOCAL_DICT` (LC)**: Uses `MiniDictDB`. A simple, file-based dictionary stored locally. It's excellent for local development, testing, or storing non-critical application state.
*   **`LOCAL_REDIS` (LR)**: Connects to a local Redis instance. Use this for high-performance caching or when you need the advanced data structures offered by Redis.
*   **`REMOTE_REDIS` (RR)**: Connects to a remote Redis server. Suitable for shared state and caching in a distributed environment.

## 2. Basic Usage

To use the DB module, you first need to get a handle to it from the main `app` instance. All interactions then happen through this instance.

```python
# Assuming 'app' is your ToolBoxV2 App instance
db = app.get_mod("DB")

# Or, if you have a specific instance of the DB module
db_spec = app.get_mod("DB", spec="MySpecialDB")
```

### Storing Data (`set`)

The `set` function stores a value associated with a key. It overwrites any existing value.

```python
# Store a simple string
db.set("user:123:name", "Alice")

# Store a dictionary (will be automatically JSON-serialized)
user_profile = {"email": "alice@example.com", "level": 10}
db.set("user:123:profile", user_profile)

# Store a list
db.set("user:123:roles", ["admin", "editor"])
```

### Retrieving Data (`get`)

The `get` function retrieves a value by its key. The data is returned as a `Result` object.

```python
result = db.get("user:123:name")

if result.is_ok():
    user_name = result.get() # .get() extracts the data from the Result
    print(f"User name: {user_name}") # Output: User name: Alice

profile_result = db.get("user:123:profile")
if profile_result.is_ok():
    # The DB module automatically deserializes the JSON string back into a dict
    profile_data = profile_result.get()
    print(f"User email: {profile_data['email']}") # Output: User email: alice@example.com
```

#### Special `get` Queries: `all` and `all-k`

The `get` method supports special query strings to retrieve all keys or all key-value pairs from the database.

*   `get('all-k')`: Returns a list of all keys in the database.
*   `get('all')`: Returns a list of all key-value pairs (as tuples) in the database.

**Example:**

```python
# Get all keys
all_keys_result = db.get("all-k")
if all_keys_result.is_ok():
    keys = all_keys_result.get()
    print(f"All keys in the database: {keys}")
    # Output: All keys in the database: [\'user:123:name\', \'user:123:profile\', \'user:123:logs\']

# Get all items (key-value pairs)
all_items_result = db.get("all")
if all_items_result.is_ok():
    items = all_items_result.get()
    print(f"All items: {items}")
    # Output: All items: [(\'user:123:name\', \'Alice\'), (\'user:123:profile\', {\'email\': \'alice@example.com\', \'level\': 10}), ...]
```

### Checking for Existence (`if_exist`)

To check if a key exists without retrieving its value, use `if_exist`.

```python
if db.if_exist("user:123:name").get():
    print("User 123 exists!")
else:
    print("User 123 not found.")
```

### Deleting Data (`delete`)

```python
# Delete a single key
delete_result = db.delete("user:123:roles")
if delete_result.is_ok():
    print("Roles deleted.")

# Delete multiple keys using a matching prefix (supported in Redis/Dict modes)
# This would delete all keys starting with "user:123:"
db.delete("user:123:", matching=True)
```

### Appending to a List (`append_on_set`)

This function is useful for adding items to a key that stores a list. If the key doesn't exist, it's created as a new list.

```python
# Assuming "user:123:logs" doesn't exist yet
db.append_on_set("user:123:logs", "User logged in.")

# Append another log
db.append_on_set("user:123:logs", "User updated profile.")

# Retrieve the list
logs_result = db.get("user:123:logs")
# logs_result.get() will be ["User logged in.", "User updated profile."]
```

## 3. Configuration and Mode Switching

While the default mode is `CLUSTER_BLOB`, you can change it for development or other specific needs.

### Configuration via Environment Variables

The easiest way to configure the DB module is through environment variables in your `.env` file. The module will read these on startup.

| Mode            | `DB_MODE_KEY` | Required Environment Variables                                   |
|-----------------|---------------|------------------------------------------------------------------|
| **Cluster Blob**| `CB`          | (None - Uses application's internal blob storage and encryption) |
| Local Dictionary| `LC`          | (None - Uses local file storage)                                 |
| Local Redis     | `LR`          | `DB_CONNECTION_URI` (e.g., `redis://localhost:6379`)             |
| Remote Redis    | `RR`          | `DB_CONNECTION_URI` or `DB_USERNAME` & `DB_PASSWORD`             |

**Example `.env` file for using Local Redis:**
```
DB_MODE_KEY=LR
DB_CONNECTION_URI=redis://localhost:6379
```

### Switching Modes Programmatically

You can switch the database mode at runtime, which is useful for testing or dynamic configuration.

```python
from toolboxv2.mods.DB.types import DatabaseModes

db = app.get_mod("DB")

# Switch to Local Dictionary mode
result = db.edit_programmable(mode=DatabaseModes.LC)

if result.is_ok():
    print("Successfully switched DB mode to LOCAL_DICT")

# The module will automatically close the old connection 
# and initialize the new one.
```

This guide covers the primary functionalities of the ToolBoxV2 `DB` module. By leveraging its abstraction and flexible configuration, you can build robust applications with persistent data storage tailored to your specific needs.
