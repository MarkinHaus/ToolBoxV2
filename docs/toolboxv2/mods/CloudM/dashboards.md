# CloudM — Dashboards

CloudM provides two server-rendered dashboards and a shared event API layer.

| Module | File | Purpose |
|--------|------|---------|
| Admin Dashboard | `AdminDashboard.py` | Full system control (admin-only) |
| User Dashboard | `UserDashboard.py` | Per-user module + file management |
| Dashboard API | `DashboardAPI.py` | Shared routing: login, render, events |

---

## Admin Dashboard

> **Module**: `CloudM.AdminDashboard` · **Version**: 0.1.1  
> **Access**: Level `-1` OR username `root` / `loot`

The Admin Dashboard is a server-rendered tbjs UI for managing users, modules, and infrastructure. All endpoints require admin authentication via `_is_admin()`.

### API Endpoints

#### System

```
GET  /CloudM.AdminDashboard/main          → Render admin dashboard HTML
GET  /CloudM.AdminDashboard/get_system_status  → CPU, RAM, worker status, service health
```

#### User Management

```
GET   /CloudM.AdminDashboard/list_users_admin            → All users (paginated)
POST  /CloudM.AdminDashboard/update_user_admin           → Edit user (level, username, email)
POST  /CloudM.AdminDashboard/delete_user_admin           → Delete user by UID
GET   /CloudM.AdminDashboard/get_waiting_list_users_admin → View signup waiting list
POST  /CloudM.AdminDashboard/remove_from_waiting_list_admin  → Remove email from waiting list
POST  /CloudM.AdminDashboard/send_invite_to_waiting_list_user_admin → Send signup invitation email
```

#### Module Management

```
GET   /CloudM.AdminDashboard/list_modules_admin   → All installed modules + status
POST  /CloudM.AdminDashboard/reload_module_admin  → Hot-reload a module by name
```

#### MinIO Credential Management

```
GET   /CloudM.AdminDashboard/get_minio_users_admin            → All users with MinIO credentials
POST  /CloudM.AdminDashboard/ensure_minio_credentials_admin   → Provision MinIO creds for user
POST  /CloudM.AdminDashboard/rotate_minio_credentials_admin   → Rotate user's MinIO credentials
POST  /CloudM.AdminDashboard/revoke_minio_credentials_admin   → Revoke user's MinIO credentials
GET   /CloudM.AdminDashboard/list_spps_admin                  → List SPPs (Service Principal Policies)
```

### Admin Check

```python
# Admin = level -1 OR username "root" or "loot"
admin = await _is_admin(app, request)
if not admin:
    return Result.html("<h1>Access Denied</h1>", status=403)
```

---

## User Dashboard

> **Module**: `CloudM.UserDashboard` · **Version**: 0.0.x  
> **Access**: Any authenticated user (own data only)

The User Dashboard gives each user control over their module instances, files, settings, and MinIO storage credentials.

### API Endpoints

#### Dashboard

```
GET  /CloudM.UserDashboard/main                  → Render user dashboard HTML
```

#### Module Instances

```
GET   /CloudM.UserDashboard/get_all_available_modules  → All installable modules
GET   /CloudM.UserDashboard/get_my_active_instances    → User's currently active instances
POST  /CloudM.UserDashboard/add_module_to_instance     → Add module to active instance
POST  /CloudM.UserDashboard/remove_module_from_instance → Remove module from instance
POST  /CloudM.UserDashboard/add_module_to_saved        → Save module to favorites
POST  /CloudM.UserDashboard/remove_module_from_saved   → Remove from saved
GET   /CloudM.UserDashboard/get_all_mod_data           → All module data for current user
```

#### File Storage

```
GET   /CloudM.UserDashboard/list_user_files      → List user's stored files
POST  /CloudM.UserDashboard/upload_user_file     → Upload file (multipart)
GET   /CloudM.UserDashboard/download_user_file   → Download file by path
POST  /CloudM.UserDashboard/delete_user_file     → Delete file by path
```

#### Settings & Security

```
POST  /CloudM.UserDashboard/update_my_settings      → Update user preferences
GET   /CloudM.UserDashboard/get_security_data        → Active sessions, 2FA status
POST  /CloudM.UserDashboard/request_my_magic_link    → Request new magic link
POST  /CloudM.UserDashboard/close_cli_session        → Revoke a CLI session token
```

#### MinIO Credentials (per user)

```
GET   /CloudM.UserDashboard/get_minio_credentials    → Current MinIO credentials
POST  /CloudM.UserDashboard/create_minio_credentials → Provision new credentials
POST  /CloudM.UserDashboard/rotate_minio_credentials → Rotate existing credentials
```

---

## Dashboard API (Routing Layer)

> **Module**: `CloudM.DashboardAPI`  
> Shared routing layer that dispatches to Admin or User dashboard based on session level, and handles WebSocket-style dashboard events.

### Endpoints

```
POST  /CloudM.DashboardAPI/logout               → Logout current session
GET   /CloudM.DashboardAPI/render_user_dashboard  → Route to user dashboard HTML
GET   /CloudM.DashboardAPI/render_admin_dashboard → Route to admin dashboard HTML (admin required)
POST  /CloudM.DashboardAPI/handle_dashboard_event → Dispatch named event
```

### Event System

`handle_dashboard_event` accepts a JSON body with an `event` field and optional `payload`:

```json
{"event": "load_module", "payload": {"module_name": "MyMod"}}
```

Supported events:

| Event | Handler | Access |
|-------|---------|--------|
| `logout` | `_handle_logout` | Any |
| `load_module` | `_handle_load_module` | Any |
| `unload_module` | `_handle_unload_module` | Any |
| `save_module` | `_handle_save_module` | Any |
| `remove_saved_module` | `_handle_remove_saved_module` | Any |
| `update_setting` | `_handle_update_setting` | Any |
| `request_magic_link` | `_handle_request_magic_link` | Any |
| `refresh_status` | `_handle_refresh_status` | Admin |
| `restart_service` | `_handle_restart_service` | Admin |
| `delete_user` | `_handle_delete_user` | Admin |
| `send_invite` | `_handle_send_invite` | Admin |
| `remove_from_waiting` | `_handle_remove_from_waiting` | Admin |
| `reload_module` | `_handle_reload_module` | Admin |

---

## UI Styling

Both dashboards use **tbjs v2 CSS variables** for theming (dark slate scheme). The HTML is server-rendered and injected into the tbjs shell. No client-side framework required — all interactivity is via tbjs event bindings.

## Related

- [Auth System](auth.md) — Session validation powering `_is_admin()` and user identity
- [User Data API](user_data.md) — Storage backend for user settings and mod data
- [Folder Sync](folder_sync.md) — MinIO integration for user file storage
- [TBjs Framework](../../tbjs.md) — Frontend shell the dashboards render into
