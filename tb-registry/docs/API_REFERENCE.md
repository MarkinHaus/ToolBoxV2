# API Reference - ToolBoxV2 Registry

**Base URL**: `https://registry.simplecore.app/api/v1`
**Version**: 1.0
**Authentication**: Bearer Token (JWT)

---

## Inhaltsverzeichnis

1. [Authentication](#authentication)
2. [Packages](#packages)
3. [Artifacts](#artifacts)
4. [Publishers](#publishers)
5. [Search](#search)
6. [Health](#health)
7. [Error Codes](#error-codes)

---

## Authentication

### Get Current User

```http
GET /api/v1/auth/me
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "id": "usr_abc123",
  "cloudm_user_id": "usr_abc123",
  "email": "user@example.com",
  "username": "username",
  "is_admin": false,
  "publisher_id": "pub_xyz789"
}
```

### Get Publisher Info

```http
GET /api/v1/auth/publisher
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "id": "pub_xyz789",
  "name": "My Publisher",
  "slug": "my-publisher",
  "status": "verified",
  "packages_count": 5
}
```

### Create Publisher

```http
POST /api/v1/auth/publisher
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "My Publisher",
  "slug": "my-publisher",
  "email": "contact@example.com",
  "website": "https://example.com",
  "github": "mygithub"
}
```

**Response (201 Created):**
```json
{
  "id": "pub_new123",
  "name": "My Publisher",
  "slug": "my-publisher",
  "status": "unverified"
}
```

---

## Packages

### List Packages

```http
GET /api/v1/packages?page=1&per_page=20&package_type=mod
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | integer | 1 | Page number |
| per_page | integer | 20 | Items per page (1-100) |
| package_type | string | - | Filter: `mod`, `library`, `artifact` |

**Response (200 OK):**
```json
{
  "packages": [
    {
      "name": "CloudM",
      "display_name": "CloudM Module",
      "version": "2.0.0",
      "package_type": "mod",
      "description": "Cloud management module",
      "author": "ToolBoxV2",
      "license": "MIT",
      "homepage": "https://github.com/toolboxv2/cloudm",
      "downloads": 1234,
      "visibility": "public",
      "created_at": "2025-01-15T10:00:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "per_page": 20
}
```

### Get Package

```http
GET /api/v1/packages/{name}
Authorization: Bearer <token> (optional)
```

**Response (200 OK):**
```json
{
  "name": "CloudM",
  "display_name": "CloudM Module",
  "version": "2.0.0",
  "package_type": "mod",
  "description": "Full description here...",
  "readme": "# CloudM\n\n...",
  "author": "ToolBoxV2",
  "license": "MIT",
  "homepage": "https://github.com/toolboxv2/cloudm",
  "repository": "https://github.com/toolboxv2/cloudm.git",
  "keywords": ["cloud", "management", "automation"],
  "latest_version": "2.0.0",
  "downloads": 1234,
  "visibility": "public",
  "versions": ["1.0.0", "1.5.0", "2.0.0"],
  "created_at": "2025-01-15T10:00:00Z",
  "updated_at": "2025-02-01T15:30:00Z"
}
```

### Create Package

```http
POST /api/v1/packages
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "my-mod",
  "display_name": "My Awesome Mod",
  "description": "This mod does awesome things",
  "package_type": "mod",
  "visibility": "unlisted",
  "homepage": "https://github.com/user/my-mod",
  "repository": "https://github.com/user/my-mod.git",
  "license": "MIT",
  "keywords": ["utility", "automation"]
}
```

**Response (201 Created):**
```json
{
  "name": "my-mod",
  "display_name": "My Awesome Mod",
  "version": null,
  "visibility": "unlisted",
  "created_at": "2025-02-25T10:00:00Z"
}
```

### Update Package

```http
PATCH /api/v1/packages/{name}
Authorization: Bearer <token>
Content-Type: application/json

{
  "description": "Updated description",
  "readme": "# Updated readme\n\n...",
  "visibility": "public"
}
```

**Response (200 OK):** Updated package object

### Delete Package

```http
DELETE /api/v1/packages/{name}
Authorization: Bearer <token>
```

**Response (204 No Content):** Package deleted

### Upload Package Version

```http
POST /api/v1/packages/{name}/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <binary>
version: "1.0.0"
changelog: "Initial release"
```

**cURL Example:**
```bash
curl -X POST https://registry.simplecore.app/api/v1/packages/my-mod/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@my-mod-1.0.0.tbx" \
  -F "version=1.0.0" \
  -F "changelog=Initial release"
```

**Response (201 Created):**
```json
{
  "name": "my-mod",
  "version": "1.0.0",
  "released_at": "2025-02-25T10:00:00Z",
  "checksum": "abc123...",
  "size_bytes": 12345
}
```

### Get Download URL

```http
GET /api/v1/packages/{name}/versions/{version}/download
Authorization: Bearer <token> (optional, required for private)
```

**Response (200 OK):**
```json
{
  "url": "https://minio.simplecore.app/bucket/my-mod-1.0.0.tbx?signature=...",
  "expires_in": 3600
}
```

### Yank Version

```http
POST /api/v1/packages/{name}/versions/{version}/yank
Authorization: Bearer <token>
Content-Type: application/json

{
  "reason": "Critical security bug"
}
```

---

## Artifacts

### List Artifacts

```http
GET /api/v1/artifacts?page=1&per_page=20
```

**Response (200 OK):**
```json
{
  "artifacts": [
    {
      "id": "art_abc123",
      "name": "TBCompiler",
      "artifact_type": "compiler",
      "description": "ToolBox compiler",
      "version": "1.0.0",
      "publisher": "ToolBoxV2"
    }
  ],
  "total": 1,
  "page": 1,
  "per_page": 20
}
```

### Get Artifact

```http
GET /api/v1/artifacts/{name}
```

### Get Latest Build

```http
GET /api/v1/artifacts/{name}/latest?platform=windows&architecture=x86_64
```

**Query Parameters:**
- `platform`: `windows`, `linux`, `macos`
- `architecture`: `x86_64`, `arm64`, `universal`

**Response (200 OK):**
```json
{
  "artifact_name": "TBCompiler",
  "version": "1.0.0",
  "platform": "windows",
  "architecture": "x86_64",
  "filename": "tb-compiler-1.0.0-win-x64.exe",
  "size_bytes": 25600000,
  "checksum": "def456...",
  "downloads": 0,
  "released_at": "2025-02-25T10:00:00Z"
}
```

---

## Publishers

### List Publishers

```http
GET /api/v1/publishers?page=1&per_page=20
```

### Get Publisher

```http
GET /api/v1/publishers/{slug}
```

**Response (200 OK):**
```json
{
  "id": "pub_xyz789",
  "name": "My Publisher",
  "slug": "my-publisher",
  "email": "contact@example.com",
  "website": "https://example.com",
  "github": "mygithub",
  "status": "verified",
  "packages_count": 5,
  "total_downloads": 50000,
  "created_at": "2025-01-01T00:00:00Z"
}
```

---

## Search

### Search Packages

```http
GET /api/v1/search?q=discord&page=1&per_page=20
```

**Query Parameters:**
- `q`: Search query
- `page`: Page number
- `per_page`: Items per page

**Response (200 OK):**
```json
{
  "results": [
    {
      "name": "discord-mod",
      "display_name": "Discord Integration",
      "description": "Integrate with Discord",
      "version": "1.0.0",
      "author": "ToolBoxV2",
      "relevance": 0.95
    }
  ],
  "total": 1,
  "page": 1,
  "per_page": 20
}
```

### Search Suggestions

```http
GET /api/v1/search/suggest?q=disc&limit=5
```

**Response (200 OK):**
```json
{
  "suggestions": [
    "discord-mod",
    "discord-utils",
    "discord-bot"
  ]
}
```

---

## Health

### Health Check

```http
GET /api/v1/health
```

**Response (200 OK):**
```json
{
  "status": "healthy"
}
```

### Readiness Check

```http
GET /api/v1/ready
```

**Response (200 OK):**
```json
{
  "status": "ready",
  "database": "connected",
  "storage": "connected"
}
```

---

## Error Codes

| Status | Code | Description |
|--------|------|-------------|
| 400 | BAD_REQUEST | Invalid request parameters |
| 401 | UNAUTHORIZED | Missing or invalid authentication |
| 403 | FORBIDDEN | Insufficient permissions |
| 404 | NOT_FOUND | Resource not found |
| 409 | CONFLICT | Resource already exists |
| 422 | UNPROCESSABLE_ENTITY | Validation error |
| 500 | INTERNAL_ERROR | Server error |

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Rate Limiting

**Current Limits:**
- Anonymous: 100 requests/hour
- Authenticated: 1000 requests/hour
- Admin: No limit

**Rate Limit Headers:**
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1740447600
```

---

## Webhooks (Coming Soon)

### Webhook Events

| Event | Description |
|-------|-------------|
| `package.created` | New package created |
| `package.updated` | Package updated |
| `package.deleted` | Package deleted |
| `version.published` | New version published |
| `version.yanked` | Version yanked |

### Webhook Delivery

Webhooks are delivered as POST requests:

```http
POST https://your-webhook-url
X-Webhook-Signature: sha256=signature
X-Webhook-Event: package.created
Content-Type: application/json

{
  "event": "package.created",
  "timestamp": "2025-02-25T10:00:00Z",
  "data": {
    "name": "my-mod",
    "version": "1.0.0",
    "publisher": "my-publisher"
  }
}
```

---

## Examples

### Python (httpx)

```python
import httpx

async def get_packages():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://registry.simplecore.app/api/v1/packages")
        return response.json()

async def upload_package(token: str, file_path: str):
    async with httpx.AsyncClient() as client:
        with open(file_path, "rb") as f:
            response = await client.post(
                "https://registry.simplecore.app/api/v1/packages/my-mod/upload",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": f},
                data={"version": "1.0.0"}
            )
        return response.json()
```

### JavaScript (fetch)

```javascript
// Get packages
async function getPackages() {
  const response = await fetch('https://registry.simplecore.app/api/v1/packages');
  return await response.json();
}

// Upload package
async function uploadPackage(token, file) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('version', '1.0.0');

  const response = await fetch(
    'https://registry.simplecore.app/api/v1/packages/my-mod/upload',
    {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`
      },
      body: formData
    }
  );
  return await response.json();
}
```

### cURL

```bash
# Get packages
curl https://registry.simplecore.app/api/v1/packages

# Get package
curl https://registry.simplecore.app/api/v1/packages/CloudM

# Upload package
curl -X POST https://registry.simplecore.app/api/v1/packages/my-mod/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@my-mod.tbx" \
  -F "version=1.0.0"

# Download package
curl -O -J $(curl -s https://registry.simplecore.app/api/v1/packages/my-mod/versions/1.0.0/download \
  -H "Authorization: Bearer $TOKEN" \
  | jq -r '.url')
```

---

**Letzte Aktualisierung**: 2026-02-25
**API Version**: 1.0
