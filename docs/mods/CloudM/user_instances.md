# User Instances

> **File:** `UserInstances.py`
> Multi-instance user sessions — manages WebSocket/SI/VT identifiers per user.

## API

| Function | Input | Output |
|----------|-------|--------|
| `get_si_id` | `uid` | Session instance ID |
| `get_vt_id` | `uid` | Virtual terminal ID |
| `get_web_socket_id` | `uid` | WebSocket connection ID |
| `close_user_instance` | `uid` | — |
| `validate_ws_id` | `ws_id` | — |
| `delete_user_instance` | `uid` | — |
| `save_user_instances` | `instance` | — |
| `hydrate_instance` | `instance` | — |
| `save_close_user_instance` | `ws_id` | — |

## Related

- [Auth System](auth.md)
- [User Account Manager](user_account_manager.md)
- [WebSocket Manager](../../runtime/event_manager.md)
