---
hide:
  - navigation
  - toc
---

# ToolBoxV2

<p class="tb-tagline">
Ein modulares Python-Framework für Tools, Funktionen und komplette Anwendungen —
lokal, im Web, als Desktop- oder Mobile-App.
</p>

<p class="tb-badges">
<a href="https://pypi.python.org/pypi/ToolBoxV2"><img alt="PyPI" src="https://img.shields.io/pypi/v/ToolBoxV2.svg?style=flat-square&color=1a4fd0"></a>
<a href="https://github.com/MarkinHaus/ToolBoxV2"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-ToolBoxV2-181717?style=flat-square&logo=github"></a>
<img alt="Python" src="https://img.shields.io/badge/python-%E2%89%A53.10-1a4fd0?style=flat-square">
</p>

```bash
pip install toolboxv2 && tb --guide
```

---

###### NAVIGATE

<div class="grid cards" markdown>

-   __Loslegen__

    ---

    Installation, Profilwahl, Config-Wizard — von `pip install` bis zur
    laufenden Instanz.

    [:octicons-arrow-right-24: Onboarding](foundations/onboarding.md)

-   __ISAA · Agents__

    ---

    FlowAgent, ExecutionEngine V3, Chains, Skills, Dreamer. Das größte
    Subsystem im Repo.

    [:octicons-arrow-right-24: ISAA Overview](mods/isaa/index.md)

-   __CloudM__

    ---

    Auth (OAuth2, Passkeys, JWT), User-Daten, Mod-Manager, verschlüsselter
    Folder-Sync.

    [:octicons-arrow-right-24: CloudM](mods/CloudM/index.md)

-   __DB & Minu__

    ---

    Datenbank-Modul (4 Backends: JSON, Redis, MinIO) und Mini web UI
    mit WebSocket Live-Updates.

    [:octicons-arrow-right-24: DB Mod](mods/DB/index.md) · [:octicons-arrow-right-24: Minu](mods/Minu/index.md)

-   __CLIs & Flows__

    ---

    `tb`, MiniCLI + Macros, icli, Chains, Background-Flows.

    [:octicons-arrow-right-24: icli](flows/icli.md)

-   __CLI Reference__

    ---

    `tb` Runner-Übersicht, Profile, Worker- & Service-Befehle.

    [:octicons-arrow-right-24: CLI Reference](services/cli.md)

-   __Runtime__

    ---

    WSGI/ASGI-Worker, ZeroMQ-Broker, FastTB, Sessions, ServiceManager.

    [:octicons-arrow-right-24: Worker System](runtime/index.md)

-   __Storage__

    ---

    DB-Modi (LC/LR/RR/CB), BlobDB mit Offline-Fallback, Blob-Sharing & Watch-API.

    [:octicons-arrow-right-24: Database Modes](storage/ref_database.md)

-   __Frontend__

    ---

    tbjs-Framework, Tauri-App-Architektur, P2P-RPC, Browser-Extension.

    [:octicons-arrow-right-24: tbjs](frontend/tbjs.md)

-   __Infrastruktur__

    ---

    TB Registry für Mods & Artefakte, self-hosted OpenAI-kompatibler LLM Gateway.

    [:octicons-arrow-right-24: Registry](registry/index.md)

</div>

---

###### ARCHITECTURE

```mermaid
flowchart TD
    Nginx["Nginx<br/>Load Balancing · Rate Limit"]
    Nginx --> W1["HTTP Worker<br/>WSGI"]
    Nginx --> W2["HTTP Worker<br/>WSGI"]
    Nginx --> W3["WS Worker<br/>asyncio"]
    W1 & W2 & W3 --> ZMQ["ZeroMQ Event Broker"]
    ZMQ --> TB["ToolBoxV2 App"]
    TB --> MODS["Mods<br/>CloudM · ISAA · DB · …"]
    TB --> FLOWS["Flows<br/>cli · mini · isaa · desktop"]
```

###### BUILDING BLOCKS

| Baustein | Beschreibung | Docs |
|---|---|---|
| **Python Core** | 4-Layer-Architektur: `App`, `MainTool`, `Result` | [Core Types](devdocs/types.md), [Core Internals](devdocs/index.md) |
| **Worker System** | WSGI/async Worker über ZeroMQ IPC | [Runtime](runtime/index.md) |
| **DB Mod** | 4 Backend-Modi: Local JSON, Redis, MinIO Blob | [DB](mods/DB/index.md) |
| **Minu** | Mini web UI: Server-Side Rendering + WebSocket Live-Updates | [Minu](mods/Minu/index.md) |
| **CloudM** | Auth, User-Daten, Mod-Management, Folder-Sync | [CloudM](mods/CloudM/index.md) |
| **ISAA** | Multi-Agent-Framework (FlowAgent, Chains, Skills) | [ISAA](mods/isaa/index.md) |
| **tbjs + Tauri** | Cross-Platform Web-/Desktop-UI | [tbjs](frontend/tbjs.md), [Browser Extension](frontend/tb_browser.md) |
| **TB Registry** | Package-Registry für Mods und Artefakte | [Registry](registry/index.md) |
| **LLM Gateway** | OpenAI-kompatibler self-hosted Gateway | [Gateway](llm_gateway/index.md) |

!!! note "Doku-Stand"

    Core-Module, Runtime und Storage sind vollständig dokumentiert.
    Seiten mit `<!-- verified: … -->`-Markern sind gegen den Code geprüft.
    Fehler bitte als [Issue](https://github.com/MarkinHaus/ToolBoxV2/issues) melden.

<p class="tb-footer">
© 2022–2026 Markin Hausmanns · <a href="https://github.com/MarkinHaus/ToolBoxV2">GitHub</a> · <a href="https://pypi.org/project/ToolBoxV2">PyPI</a>
</p>
