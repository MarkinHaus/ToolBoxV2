# Changelog

Alle nennenswerten Änderen an ToolBoxV2 werden hier dokumentiert.
Format basiert auf [Keep a Changelog](https://keepachangelog.com/de/), Versionierung nach [SemVer](https://semver.org/lang/de/).

## [0.1.28] — 2026-07

### Added
- ISAA Dreamer V3: Standalone FlowAgent mit TaskMap-basiertem Meta-Learning
- Session History Access für Dreamer (`get_session_histories` Action)
- Mod-Architektur Doku (`devdocs/mod_architecture.md`)
- Manifest-basierte Feature-Verwaltung (`tb manifest feature`)
- LLM Gateway mit Live-Handler und Model-Manager

### Changed
- Worker-System: `cli_worker_manager.py` nutzt `tb-manifest.yaml` (YAML, nicht mehr TOML)
- Web-UI Port standardisiert auf `9005` (via Manifest `nginx.upstream_*`)
- Docs-Struktur bereinigt: alte `toolboxv2/` Pfade → neue Sektionen (`runtime/`, `storage/`, `frontend/`)
- CloudM LiveSync Doku konsolidiert (FolderSync → LiveSync)

### Removed
- Legacy `.md`-Log- Harvesting (`harvest.py`) — TaskMap ist jetzt einzige Wahrheit
- Veraltete Phantom-Klassen aus ISAA Reference (`UnifiedContextManager`, `VariableManager`, `Pipeline`)
- Leere Orphan-Ordner (`docs/tb_browser/`, `docs/tbjs_styles/`, `docs/workers/`)

## [0.1.27] — 2026-06

### Added
- FlowAgent Builder Pattern (`FlowAgentBuilder`)
- Chain DSL mit Operatoren (`>>`, `|`, `&`, `+`)
- Agent Skills/Rules/Personas System
- Session-Persistenz via `AgentSessionV2`
- CloudM Auth 2.0 (JWT-basiert)

### Changed
- ExecutionEngine überarbeitet: Streaming-Output, Tool-Registration via `tool_mgr`
- Mod-Manager: Registry-Integration, automatische Versionsprüfung

## [0.1.26] — 2026-05

### Added
- Worker-System (HTTP/WS/Broker) mit ZeroMQ
- Signed-Cookie Sessions (HMAC-SHA256)
- Nginx-Integration mit automatischer Config-Generierung
- BlobDB mit Offline-Fallback

### Changed
- Datenbank-Backends: LC/LR/RR/CB Modi implementiert
- Service Manager für Background-Services

## [0.1.0] — 2025

- Initiale ToolBoxV2-Architektur
- Mod-System (`@export` Decorator, `MainTool` Basisklasse)
- CLI (`tb`) mit Runner-System
