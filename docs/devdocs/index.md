# Developer Docs — Core Internals

Reference documentation for ToolBoxV2 core utilities and infrastructure.

## Core Types

| Doc | Source File | Description |
|-----|------------|-------------|
| [Core Types](types.md) | `utils/system/types.py` | `AppType`, `Result`, `Request`, `@tb` decorator |
| [All Functions Enums](all_functions_enums.md) | `utils/system/all_functions_enums.py` | Auto-generated dispatch table (`tb -l -sfe`) |
| [FileHandlerV2](file_handler.md) | `utils/system/file_handler.py` | Storage handler, scope detection, backends |
| [Crypto Utilities](cryp.md) | `utils/security/cryp.py` | AES/RSA encryption, signatures, hashing |

## Infrastructure

| Doc | Source File | Description |
|-----|------------|-------------|
| [RegistryClient](registry_client.md) | `utils/extras/registry_client.py` | TB-Registry client: install, publish, search |
| [WorkerManager](cli_worker_manager.md) | `utils/clis/cli_worker_manager.py` | Worker process + Nginx management |
| [WSWorker](ws_worker.md) | `utils/workers/ws_worker.py` | WebSocket worker: connections, heartbeat |
| [Toolbox Integration](toolbox_integration.md) | `utils/workers/toolbox_integration.py` | AccessController, ModuleRouter, ZMQEventBridge |
| [ManifestServiceManager](manifest_service_manager.md) | `utils/manifest/service_manager.py` | Manifest-driven service sync |
| [Utils Analysis](utils_analysis.md) | — | Coverage matrix, loose ends, priorities |

## User-Facing Utilities

| Doc | Source File | Description |
|-----|------------|-------------|
| [Style](style.md) | `utils/extras/Style.py` | ANSI colors, SpinnerManager, JSON extraction |
| [Notifications](notification.md) | `utils/extras/notification.py` | Desktop toasts, messageboxes, progress bars |
| [P2P CLI](p2p_cli.md) | `utils/clis/tcm_p2p_cli.py` | Encrypted chat, voice, file transfer |
| [DB CLI Manager](db_cli_manager.md) | `utils/clis/db_cli_manager.py` | MinIO bucket + credential management |
| [User Manager](user_manager.md) | `utils/clis/user_manager.py` | Unified auth + MinIO user provisioning |

## Helpers & Misc

| Doc | Source File | Description |
|-----|------------|-------------|
| [State System](state_system.md) | `utils/system/state_system.py` | Auto-update: download new tb executable |
| [Cache](cache.md) | `utils/system/cache.py` | FileCache (shelve) + MemoryCache (dict+TTL) |
| [Requirements Builder](reqbuilder.md) | `utils/extras/reqbuilder.py` | Stub — requirements.txt generator (not implemented) |
| [Live Debugger](live_debugger.md) | `utils/extras/live_debugger.py` | Thread + async task stack dumps |
| [Fallback Tray](fallback_tray.md) | `utils/extras/fallback_tray.py` | System tray icon fallback |

## How-to Guides

| Doc | Description |
|-----|-------------|
| [Mod erstellen & verlinken](howto-create-mod.md) | Tutorial: Neues Mod, Terminal + Code, extern verlinken |
| [Navbar-Konzepte](navbar_concepts.md) | 3 Navigations-Varianten für die Doku |

## Docs Toolchain

| Doc | Source File | Description |
|-----|------------|-------------|
| [Docs Toolchain](docs_toolchain.md) | `utils/extras/mkdocs.py` | DocsSystem: indexing, parsing, search |
