# RegistryServer

> **File:** `RegistryServer.py`
> TB-Registry process management — manages the ToolBox registry server process lifecycle.

## API

| Method | Description |
|--------|-------------|
| `start_registry(config)` | Start the registry server process |
| `stop_registry()` | Gracefully stop registry |
| `restart_registry()` | Restart with current config |
| `registry_status() → dict` | Check if running, uptime, port |

## Related

- [Mod Manager](mod_manager.md) — package install/update via registry
