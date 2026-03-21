# MCP HTTP Transport Reference

## Overview

The MCP HTTP Transport provides a REST API layer for the ToolBoxV2 MCP server with API key authentication, CORS support, and session management.

**Requirements:**
- `aiohttp`
- `aiohttp-cors`

**Install:** `pip install aiohttp aiohttp-cors --break-system-packages`
<!-- verified: http_transport.py::HTTPTransport -->

---

## HTTP Endpoints

### MCP Protocol Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/mcp/initialize` | Required | Initialize session |
| POST | `/mcp/tools/list` | Required | List available tools |
| POST | `/mcp/tools/call` | Required | Execute a tool |
| POST | `/mcp/resources/list` | Required | List resources |
| POST | `/mcp/resources/read` | Required | Read a resource |

### Management Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/health` | None | Health check (no auth) |
| GET | `/status` | Required | Detailed server status |
| GET | `/api/keys` | Admin | List API keys |
| POST | `/api/keys` | Admin | Create new API key |
| DELETE | `/api/keys/{name}` | Admin | Revoke API key |

<!-- verified: http_transport.py::_setup_routes -->

---

## Tool Execution (Request/Response Format)

### POST /mcp/tools/call

**Request:**
```json
{
  "name": "tool_name",
  "arguments": {
    "key": "value"
  }
}
```

**Response:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "result content"
    }
  ],
  "isError": false
}
```

### POST /mcp/tools/list

**Response:**
```json
{
  "tools": [
    {
      "name": "tool_name",
      "description": "...",
      "inputSchema": { ... }
    }
  ]
}
```

<!-- verified: http_transport.py::_handle_call_tool -->

---

## Authentication

### API Key Methods

1. **Authorization Header:** `Authorization: Bearer <api_key>`
2. **X-API-Key Header:** `X-API-Key: <api_key>`

### Permission Levels

| Permission | Description |
|------------|-------------|
| `read` | Read resources |
| `write` | Write operations (e.g., docs_writer) |
| `execute` | Execute code (e.g., python execution) |
| `admin` | API key management |

### Permission-Based Tool Filtering

Tools are filtered based on API key permissions:
- `python*` tools → require `execute` permission
- `docs_writer*` tools → require `write` permission
- `admin*` tools → require `admin` permission

<!-- verified: http_transport.py::_authenticate -->

---

## Additional Auth Endpoints (server_worker.py)

The HTTP worker (`server_worker.py`) provides additional auth endpoints:

| Endpoint | Method | Handler | Description |
|----------|--------|---------|-------------|
| `/validateSession` | POST | `validate_session` | JWT token validation |
| `/IsValidSession` | GET | `is_valid_session` | Session check |
| `/web/logoutS` | POST | `logout` | Logout with token blacklist |
| `/api_user_data` | GET | `get_user_data` | User data retrieval |
| `/auth/discord/url` | GET | `get_discord_auth_url` | Discord OAuth URL |
| `/auth/discord/callback` | GET | `discord_callback` | Discord OAuth callback |
| `/auth/google/url` | GET | `get_google_auth_url` | Google OAuth URL |
| `/auth/google/callback` | GET | `google_callback` | Google OAuth callback |
| `/auth/magic/verify` | GET | `magic_link_verify` | Magic link verification |

<!-- verified: server_worker.py::HTTPWorker.AUTH_ENDPOINTS -->

---

## Access Levels (server_worker.py)

| Level | Constant | Description |
|-------|----------|-------------|
| -1 | `AccessLevel.ADMIN` | Full access |
| 0 | `AccessLevel.NOT_LOGGED_IN` | No access |
| 1 | `AccessLevel.LOGGED_IN` | Basic logged in |
| 2 | `AccessLevel.TRUSTED` | Trusted user |

Public endpoints (no auth required):
- Modules in `open_modules` config
- Functions starting with `open`

<!-- verified: server_worker.py::AccessLevel -->
