# CLI Reference

> **Runner**: `toolboxv2/utils/clis/` · **Entry**: `tb` binary

ToolBoxV2 CLI (`tb`) ist das zentrale Kommando-Center. Jeder Sub-Command (Runner) steuert einen Systembereich.

## Quick Start

```bash
tb                  # Interactive Dashboard (default)
tb --version        # Version anzeigen
tb status           # System-Status
tb --help           # Alle Runner anzeigen
tb <runner> --help  # Hilfe für spezifischen Runner
```

## Profile

Beim ersten Start wählt `tb` ein Profil:

| Profil | Verhalten | Use Case |
|--------|-----------|----------|
| `consumer` | Startet GUI | App-Nutzer |
| `homelab` | Interactive Dashboard | Lokale Multi-Mod-Nutzung |
| `server` | ASCII-Status, dann Exit | Infrastruktur |
| `business` | 3-Zeilen Health-Summary, dann Exit | Quick Health Check |
| `developer` | Interactive Dashboard + Dev-Hints | Entwicklung |

Default: `homelab`.

## Runner Übersicht

| Runner | Beschreibung |
|--------|-------------|
| `tb` (default) | Interactive Dashboard |
| `tb run` | Modul-Funktion direkt ausführen |
| `tb db` | Datenbank-Verwaltung |
| `tb workers` | Worker-System starten/stoppen |
| `tb services` | Background-Services verwalten |
| `tb mods` | Mod-Installation & Verwaltung |
| `tb flow` | Flows auflisten/ausführen |
| `tb manifest` | Manifest-Konfiguration |
| `tb status` | System-Health-Check |
| `tb user` | User-Management |
| `tb login` / `tb logout` | CloudM-Authentifizierung |
| `tb session` | Session-Verwaltung |
| `tb build` | Build & Packaging |
| `tb gui` | Desktop/Web GUI starten |
| `tb browser` | Browser-Client starten |
| `tb venv` | Virtual Environment verwalten |
| `tb mcp` | MCP-Server starten |
| `tb access` | Zugriffsrechte verwalten |
| `tb broker` | ZeroMQ Event Broker |
| `tb http_worker` | HTTP Worker direkt starten |
| `tb ws_worker` | WebSocket Worker direkt starten |

## Häufige Befehle

### System

```bash
tb status              # Vollständiger System-Status
tb manifest validate   # Manifest validieren
tb manifest show       # Aktuelle Konfiguration anzeigen
tb manifest init       | Initialisiere Manifest
```

### Module

```bash
tb mods                # Alle installierten Module auflisten
tb mods -i <mod>       # Modul installieren
tb mods -u <mod>       # Modul deinstallieren
```

### Worker & Services

```bash
tb workers             # Worker-Status
tb workers start       # Alle Worker starten
tb workers stop        # Alle Worker stoppen
tb services            # Services auflisten
```

### Ausführung

```bash
tb run -c <mod> <func> [args]   # Mod-Funktion ausführen
tb run --test                   # Tests ausführen
```

## Konfiguration

Konfiguration läuft über das **Manifest** (`tb-manifest.yaml`):

```bash
tb manifest show           # Gesamtes Manifest anzeigen
tb manifest show database  # Nur DB-Sektion
tb manifest set <key> <v>  # Wert setzen (synct .env automatisch)
tb manifest validate       # Schema-Validierung
```

Siehe auch: [Runtime Config](../runtime/config.md), [Worker System](../runtime/index.md)
