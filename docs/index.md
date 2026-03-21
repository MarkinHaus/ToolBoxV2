# ToolBoxV2

A flexible modular framework for tools, functions, and complete applications — deployable locally, on the web, or as a desktop/mobile app.

[![PyPI Version](https://img.shields.io/pypi/v/ToolBoxV2.svg)](https://pypi.python.org/pypi/ToolBoxV2)
[![GitHub](https://img.shields.io/badge/GitHub-ToolBoxV2-181717?logo=github)](https://github.com/MarkinHaus/ToolBoxV2)

---

## Architecture

```
                    ┌─────────────┐
                    │    Nginx    │
                    │ (Load Bal., │
                    │ Rate Limit) │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
  │ HTTP Worker │   │ HTTP Worker │   │ WS Worker   │
  │  (WSGI)     │   │  (WSGI)     │   │ (asyncio)   │
  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
         └────────────┬────┴─────────────────┘
                      │
               ┌──────▼──────┐
               │  ZeroMQ     │
               │ Event Broker│
               └──────┬──────┘
                      │
               ┌──────▼──────┐
               │ ToolBoxV2   │
               │ App Instance│
               └─────────────┘
```

## Key Components

| Component | Description |
|-----------|-------------|
| **Python Backend** | Core library with modular 4-layer architecture |
| **Worker System** | WSGI/async workers via ZeroMQ IPC |
| **CloudM** | Auth, user data, mod management, folder sync |
| **ISAA** | Multi-agent AI framework (FlowAgent, ExecutionEngine) |
| **tbjs + Tauri** | Cross-platform web/desktop UI |
| **TB Registry** | Package registry for mods and artifacts |
| **LLM Gateway** | OpenAI-compatible self-hosted gateway (Ollama) |

## Quick Navigation

- **New here?** → [Getting Started](foundations/quickstart.md)
- **Developer?** → [4-Layer Architecture](devdocs/dev_architecture.md) · [Creating a Mod](devdocs/dev_mod_creation.md)
- **Auth & Users?** → [CloudM Overview](mods/CloudM/index.md)
- **AI Agents?** → [ISAA Framework](mods/isaa/README.md)
- **Self-hosting?** → [Server Guide](guides/howto_server.md)

---

© 2022–2025 Markin Hausmanns — [GitHub](https://github.com/MarkinHaus/ToolBoxV2) · [Issues](https://github.com/MarkinHaus/ToolBoxV2/issues) · [PyPI](https://pypi.org/project/ToolBoxV2)
