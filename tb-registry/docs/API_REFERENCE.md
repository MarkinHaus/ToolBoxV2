# API Reference - ToolBoxV2 Registry

**Base URL**: `https://registry.simplecore.app/api/v1`
**Version**: 1.1
**Authentication**: Bearer Token (JWT via CloudM.Auth)

---

## Inhaltsverzeichnis

1. [Authentication](#authentication)
2. [Packages](#packages)
3. [Artifacts](#artifacts)
4. [Publishers](#publishers)
5. [Search](#search)
6. [Versions (Update-Ping)](#versions-update-ping)
7. [Dependency Resolution](#dependency-resolution)
8. [Diff (Incremental Updates)](#diff-incremental-updates)
9. [Admin](#admin)
10. [Health](#health)
11. [Error Codes](#error-codes)

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

**Hinweis**: Users werden automatisch beim ersten Login erstellt.

### Get Publisher Info

```http
GET /api/v1/auth/publisher
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "id": "pub_xyz789",
  "name": "my-publisher",
  "display_name": "My Publisher",
  "verification_status": "verified"
}
```

### Register as Publisher

```http
POST /api/v1/auth/register-publisher
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "my-publisher",
  "display_name": "My Publisher",
  "email": "contact@example.com",
  "homepage": "https://example.com"
}
```

**Response (201 Created):**
```json
{
  "id": "pub_new123",
  "name": "my-publisher",
  "display_name": "My Publisher",
  "verification_status": "unverified"
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
| package_type | string | - | Filter: `mod`, `library`, `artifact`, `theme`, `plugin` |

**Response (200 OK):**
```json
{
  "packages": [
    {
      "name": "CloudM",
      "display_name": "CloudM Module",
      "package_type": "mod",
      "description": "Cloud management module",
      "latest_version": "2.0.0",
      "total_downloads": 1234
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
```

**Response (200 OK):**
```json
{
  "name": "CloudM",
  "display_name": "CloudM Module",
  "package_type": "mod",
  "description": "Cloud management module",
  "latest_version": "2.0.0",
  "total_downloads": 1234,
  "homepage": "https://github.com/MarkinHaus/ToolBoxV2",
  "repository": "https://github.com/MarkinHaus/ToolBoxV2.git"
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
  "package_type": "mod"
}
```

**Response (201 Created):**
```json
{
  "id": "my-mod",
  "name": "my-mod"
}
```

### Update Package

```http
PATCH /api/v1/packages/{name}
Authorization: Bearer <token>
Content-Type: application/json

{
  "description": "Updated description",
  "visibility": "public"
}
```

### Delete Package

```http
DELETE /api/v1/packages/{name}
Authorization: Bearer <token>
```

**Response (204 No Content)**

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
  -F "file=@my-mod-1.0.0.zip" \
  -F "version=1.0.0" \
  -F "changelog=Initial release"
```

### Get Download URL

```http
GET /api/v1/packages/{name}/versions/{version}/download
Authorization: Bearer <token> (required for private/unlisted)
```

**Response (200 OK):**
```json
{
  "url": "https://minio.simplecore.app/bucket/my-mod-1.0.0.zip?signature=...",
  "expires_in": 3600
}
```

**Download-Berechtigungen:**
- **Public**: Jeder kann downloaden
- **Unlisted**: Nur authentifizierte User
- **Private**: Nur der Owner

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

Artifacts sind kompilierte Binaries und Apps (z.B. SimpleCore Desktop, TB CLI).

### Artifact Types

| Type | Beschreibung |
|------|-------------|
| `tauri_app` | Tauri Desktop Application |
| `cli_executable` | CLI Binary |
| `browser_extension` | Browser Extension |
| `mobile_app` | Mobile App |
| `library` | Compiled Library |

### List Artifacts

```http
GET /api/v1/artifacts?page=1&per_page=20&artifact_type=tauri_app
```

### Create Artifact

```http
POST /api/v1/artifacts
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "SimpleCore",
  "artifact_type": "tauri_app",
  "description": "SimpleCore Desktop Application",
  "homepage": "https://simplecore.app"
}
```

### Get Artifact

```http
GET /api/v1/artifacts/{name}
```

### Upload Build

```http
POST /api/v1/artifacts/{name}/builds
Authorization: Bearer <token>
Content-Type: multipart/form-data

version: "1.0.0"
platform: "windows"
architecture: "x64"
changelog: "Initial release"
installer_type: "msi"
min_os_version: "10.0"
file: <binary>
```

**cURL Example:**
```bash
curl -X POST https://registry.simplecore.app/api/v1/artifacts/SimpleCore/builds \
  -H "Authorization: Bearer $TOKEN" \
  -F "version=1.0.0" \
  -F "platform=windows" \
  -F "architecture=x64" \
  -F "installer_type=msi" \
  -F "file=@SimpleCore-1.0.0-win-x64.msi"
```

**Platforms**: `all`, `windows`, `linux`, `macos`, `android`, `ios`
**Architectures**: `all`, `x64`, `x86`, `arm64`, `arm32`

### Get Latest Build for Platform

```http
GET /api/v1/artifacts/{name}/latest?platform=windows&architecture=x64
```

**Response (200 OK):**
```json
{
  "version": "1.0.0",
  "released_at": "2025-02-25T10:00:00Z",
  "changelog": "Initial release",
  "build": {
    "platform": "windows",
    "architecture": "x64",
    "filename": "SimpleCore-1.0.0-win-x64.msi",
    "size_bytes": 25600000,
    "checksum_sha256": "def456..."
  }
}
```

### Get Build Download URL

```http
GET /api/v1/artifacts/{name}/versions/{version}/download?platform=windows&architecture=x64
```

**Response (200 OK):**
```json
{
  "url": "https://minio.simplecore.app/bucket/SimpleCore-1.0.0.msi?signature=...",
  "expires_in": 3600
}
```

---

## Publishers

### List Publishers

```http
GET /api/v1/publishers?page=1&per_page=20&verified_only=true
```

### Get Publisher

```http
GET /api/v1/publishers/{slug}
```

**Response (200 OK):**
```json
{
  "id": "pub_xyz789",
  "name": "my-publisher",
  "display_name": "My Publisher",
  "verification_status": "verified",
  "package_count": 5,
  "total_downloads": 50000
}
```

### Submit Verification Request

```http
POST /api/v1/publishers/verify
Authorization: Bearer <token>
Content-Type: application/json

{
  "method": "github",
  "data": {
    "username": "mygithub"
  }
}
```

**Response (202 Accepted):**
```json
{
  "status": "submitted",
  "message": "Verification request submitted"
}
```

---

## Search

### Search Packages

```http
GET /api/v1/search?q=discord&page=1&per_page=20
```

**Response (200 OK):**
```json
{
  "results": [
    {
      "name": "discord-mod",
      "display_name": "Discord Integration",
      "description": "Integrate with Discord",
      "latest_version": "1.0.0",
      "total_downloads": 100
    }
  ],
  "total": 1,
  "query": "discord",
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
["discord-mod", "discord-utils"]
```

---

## Versions (Update-Ping)

Batch-Abfrage der neuesten Versionen — genutzt von ToolBoxV2 für Update-Checks.

```http
GET /api/v1/versions?names=CloudM&names=discord-mod
```

**Response (200 OK):**
```json
{
  "versions": {
    "CloudM": "2.0.0",
    "discord-mod": "1.0.0"
  }
}
```

Leere `names` → gibt alle Packages zurück.

---

## Dependency Resolution

### Resolve Dependencies

```http
POST /api/v1/resolve
Content-Type: application/json

{
  "requirements": ["CloudM>=2.0.0", "discord-mod>=1.0.0"],
  "toolbox_version": "0.1.25"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "resolved": {
    "CloudM": {
      "name": "CloudM",
      "version": "2.0.0",
      "download_url": "https://...",
      "checksum": "abc123..."
    }
  },
  "conflicts": [],
  "warnings": []
}
```

### Check Compatibility

```http
GET /api/v1/resolve/check?package=CloudM&version=2.0.0&toolbox_version=0.1.25
```

**Response (200 OK):**
```json
{
  "compatible": true,
  "warnings": [],
  "conflicts": []
}
```

---

## Diff (Incremental Updates)

Diffs ermöglichen inkrementelle Updates — nur die Änderungen zwischen zwei Versionen werden übertragen.

### Get Diff Info

```http
GET /api/v1/packages/{name}/diff/{from_version}/{to_version}
```

**Response (200 OK):**
```json
{
  "package_name": "CloudM",
  "from_version": "1.0.0",
  "to_version": "2.0.0",
  "patch_size": 12345,
  "full_size": 98765,
  "compression_ratio": 0.125,
  "patch_storage_path": "diffs/CloudM/1.0.0-2.0.0.patch",
  "patch_checksum": "abc123..."
}
```

### Download Diff Patch

```http
GET /api/v1/packages/{name}/diff/{from_version}/{to_version}/download
```

### Create Diff (Pre-generation)

```http
POST /api/v1/packages/{name}/diff/create?from_version=1.0.0&to_version=2.0.0&force=false
```

---

## Admin

Alle Admin-Endpoints erfordern `is_admin = true`.

### List Pending Publishers

```http
GET /api/v1/packages/admin/pending?page=1&per_page=50
Authorization: Bearer <admin-token>
```

### Verify Publisher

```http
POST /api/v1/packages/admin/{publisher_id}/verify
Authorization: Bearer <admin-token>
Content-Type: application/json

{"notes": "Verified via API"}
```

### Reject Publisher

```http
POST /api/v1/packages/admin/{publisher_id}/reject
Authorization: Bearer <admin-token>
Content-Type: application/json

{"notes": "Reason for rejection"}
```

### Revoke Verification

```http
POST /api/v1/packages/admin/{publisher_id}/revoke
Authorization: Bearer <admin-token>
Content-Type: application/json

{"notes": "Reason for revocation"}
```

---

## Health

### Health Check

```http
GET /api/v1/health
```

**Response (200 OK):**
```json
{"status": "healthy"}
```

### Readiness Check

```http
GET /api/v1/ready
```

**Response (200 OK):**
```json
{
  "status": "ready",
  "details": {
    "database": true,
    "storage": {"healthy": true}
  }
}
```

---

## Error Codes

| Status | Code | Description |
|--------|------|-------------|
| 400 | BAD_REQUEST | Invalid request parameters |
| 401 | UNAUTHORIZED | Missing or invalid authentication |
| 403 | FORBIDDEN | Insufficient permissions / Admin required |
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

### cURL

```bash
# List packages
curl https://registry.simplecore.app/api/v1/packages

# Get package
curl https://registry.simplecore.app/api/v1/packages/CloudM

# Upload package version
curl -X POST https://registry.simplecore.app/api/v1/packages/my-mod/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@my-mod.zip" \
  -F "version=1.0.0"

# Check for updates (batch)
curl "https://registry.simplecore.app/api/v1/versions?names=CloudM&names=discord-mod"

# Upload artifact build
curl -X POST https://registry.simplecore.app/api/v1/artifacts/SimpleCore/builds \
  -H "Authorization: Bearer $TOKEN" \
  -F "version=1.0.0" \
  -F "platform=linux" \
  -F "architecture=x64" \
  -F "file=@simplecore-linux-x64.deb"
```

---

**Letzte Aktualisierung**: 2026-04-28
**API Version**: 1.1
