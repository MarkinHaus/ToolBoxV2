# Blob Sharing API Documentation

## Overview

The Blob Sharing API enables secure sharing of blobs between users with granular access control. Each user has a unique Public User ID generated from their device name, ensuring that one party cannot have both keys.

## Key Concepts

### Public User ID
- **Generated from device name** using a hash function
- Format: `user_<hash>` (e.g., `user_abc123def456`)
- Unique per device/user
- Used for sharing instead of exposing API keys

### Access Levels
- **`read_only`**: User can only read the blob
- **`read_write`**: User can read and modify the blob

### Security Features
✅ **One party cannot have both keys** - Enforced by user_id system  
✅ **Hash ring consistency** - All operations use the same server  
✅ **Server-side enforcement** - Access control managed by Rust server  
✅ **Cannot share with yourself** - Prevented by server  

---

## API Reference

### BlobStorage.get_user_id()

Get your Public User ID.

```python
def get_user_id(server: Optional[str] = None) -> Optional[str]
```

**Parameters:**
- `server` (Optional[str]): Specific server to get user_id from, or None for first available

**Returns:**
- User ID string (e.g., `'user_abc123'`) or None

**Example:**
```python
storage = BlobStorage(servers=['http://localhost:3000'])
my_user_id = storage.get_user_id()
print(f"My User ID: {my_user_id}")
```

---

### BlobStorage.share_blob()

Share a blob with another user.

```python
def share_blob(
    blob_id: str,
    target_user_id: str,
    access_level: str = 'read_only'
) -> Dict
```

**Parameters:**
- `blob_id` (str): The blob to share
- `target_user_id` (str): Public User ID of the target user
- `access_level` (str): `'read_only'` or `'read_write'`

**Returns:**
- Dict with share information:
  - `blob_id`: The shared blob
  - `user_id`: Target user ID
  - `access_level`: Access level granted
  - `granted_at`: Unix timestamp

**Raises:**
- `ValueError`: Invalid access_level
- `HTTPError`: Server error (e.g., cannot share with yourself, target user doesn't exist)

**Example:**
```python
# Share config file (read-only)
storage.share_blob('app/config.json', 'user_abc123', 'read_only')

# Share data file (read-write)
storage.share_blob('app/data.json', 'user_xyz789', 'read_write')
```

---

### BlobStorage.revoke_share()

Revoke share access for a user.

```python
def revoke_share(blob_id: str, target_user_id: str) -> bool
```

**Parameters:**
- `blob_id` (str): The blob to revoke access from
- `target_user_id` (str): Public User ID to revoke

**Returns:**
- `True` if successful, `False` otherwise

**Example:**
```python
storage.revoke_share('app/config.json', 'user_abc123')
```

---

### BlobStorage.list_shares()

List all users who have access to a blob.

```python
def list_shares(blob_id: str) -> List[Dict]
```

**Parameters:**
- `blob_id` (str): The blob to list shares for

**Returns:**
- List of dicts with:
  - `user_id`: Public User ID
  - `access_level`: `'read_only'` or `'read_write'`
  - `granted_by`: User ID who granted access
  - `granted_at`: Unix timestamp

**Example:**
```python
shares = storage.list_shares('app/config.json')
for share in shares:
    print(f"{share['user_id']}: {share['access_level']}")
```

---

## Server API Endpoints

### POST /keys
Create API key with device name.

**Request:**
```json
{
  "device_name": "my-laptop"
}
```

**Response:**
```json
{
  "key": "uuid-api-key",
  "user_id": "user_abc123def456"
}
```

---

### POST /share/:blob_id
Share a blob with a user.

**Headers:**
- `x-api-key`: Your API key

**Request:**
```json
{
  "user_id": "user_xyz789",
  "access_level": "read_only"
}
```

**Response:**
```json
{
  "blob_id": "app/config.json",
  "user_id": "user_xyz789",
  "access_level": "read_only",
  "granted_at": 1234567890
}
```

---

### GET /share/:blob_id
List all shares for a blob.

**Headers:**
- `x-api-key`: Your API key

**Response:**
```json
{
  "blob_id": "app/config.json",
  "shares": [
    {
      "user_id": "user_xyz789",
      "access_level": "read_only",
      "granted_by": "user_abc123",
      "granted_at": 1234567890
    }
  ]
}
```

---

### DELETE /share/:blob_id/:user_id
Revoke share for a user.

**Headers:**
- `x-api-key`: Your API key

**Response:**
- `204 No Content` on success

---

## Usage Examples

### Example 1: Share Config File (Read-Only)

```python
# User A
storage_a = BlobStorage(servers=['http://localhost:3000'])
user_a_id = storage_a.get_user_id()

# Create config
with BlobFile('app/config.json', 'w', storage=storage_a) as f:
    f.write_json({'theme': 'dark'})

# User B
storage_b = BlobStorage(servers=['http://localhost:3000'])
user_b_id = storage_b.get_user_id()

# Share with User B
storage_a.share_blob('app/config.json', user_b_id, 'read_only')

# User B can read
with BlobFile('app/config.json', 'r', storage=storage_b) as f:
    config = f.read_json()  # ✅ Works

# User B cannot write
with BlobFile('app/config.json', 'w', storage=storage_b) as f:
    f.write_json({'modified': True})  # ❌ 403 Forbidden
```

### Example 2: Collaborative Editing (Read-Write)

```python
# User A creates document
with BlobFile('collab/doc.json', 'w', storage=storage_a) as f:
    f.write_json({'content': 'Initial'})

# Share with User B (read-write)
storage_a.share_blob('collab/doc.json', user_b_id, 'read_write')

# User B can edit
with BlobFile('collab/doc.json', 'r', storage=storage_b) as f:
    doc = f.read_json()

doc['content'] += '\nEdited by B'

with BlobFile('collab/doc.json', 'w', storage=storage_b) as f:
    f.write_json(doc)  # ✅ Works
```

---

## Security Considerations

1. **User ID Generation**: Based on device hostname, ensure unique device names
2. **API Key Storage**: Keys are encrypted with device key
3. **Access Control**: Enforced server-side, cannot be bypassed
4. **Hash Ring**: Ensures all operations for a blob use the same server
5. **One Party Rule**: User IDs prevent one party from having both keys

---

## Error Handling

| Error | Status | Meaning |
|-------|--------|---------|
| `401 Unauthorized` | 401 | Invalid or missing API key |
| `403 Forbidden` | 403 | Insufficient permissions (e.g., read-only trying to write) |
| `404 Not Found` | 404 | Blob or user doesn't exist |
| `400 Bad Request` | 400 | Invalid request (e.g., sharing with yourself) |

