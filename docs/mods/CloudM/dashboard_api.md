# Dashboard API

> **File:** `DashboardAPI.py`
> REST endpoints for dashboard data — powers both admin and user dashboards.

## Key Methods

| Method | Description |
|--------|-------------|
| `get_system_status()` | System health, uptime, module count |
| `get_user_list()` | List all users (admin) |
| `get_user_data(user_id)` | Per-user data |
| `get_module_list()` | Loaded modules and their states |

## Related

- [Admin Dashboard](admin_dashboard.md)
- [User Dashboard](user_dashboard.md)
- [Dashboards Overview](dashboards.md)
