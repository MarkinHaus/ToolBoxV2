# Admin Dashboard

> **File:** `AdminDashboard.py`
> Server administration UI — manage users, modules, services, and system settings.

## Features

- **User Management**: List, delete users, manage invitations
- **Module Management**: Load/unload/reload modules at runtime
- **Service Management**: Restart services, view status
- **System Settings**: Configure server-wide settings

## Events (via Dashboard API)

| Event | Access | Description |
|-------|--------|-------------|
| `refresh_status` | Admin | Refresh system status |
| `restart_service` | Admin | Restart a service |
| `delete_user` | Admin | Delete user account |
| `send_invite` | Admin | Send invitation |
| `remove_from_waiting` | Admin | Remove from waiting list |
| `reload_module` | Admin | Hot-reload a module |

## Related

- [Dashboard API](dashboard_api.md) — data endpoints
- [User Dashboard](user_dashboard.md) — per-user dashboard
- [Auth](../CloudM/auth.md) — admin access levels
