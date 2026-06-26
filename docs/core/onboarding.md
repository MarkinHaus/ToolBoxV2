# ToolBoxV2 — Umfassendes Onboarding Guide

> Dies ist das komplette Onboarding für ToolBoxV2 von 0 bis produktiv.

---

## 📑 Inhaltsverzeichnis

1. [🚀 Schnellstart für Anfänger (0 Code-Kenntnisse)](#1--schnellstart-für-anfänger-0-code-kenntnisse)
2. [🏠 Homelab Setup & Management](#2--homelab-setup--management)
3. [🖥️ Server Admin Guide](#3--server-admin-guide)
4. [📚 Migration: Von alten zu neuen Commands](#4--migration-von-alten-zu-neuen-commands)
5. [🔧 Cheat Sheet](#-cheat-sheet)

---

# 1. 🚀 Schnellstart für Anfänger (0 Code-Kenntnisse)

## Was ist ToolBoxV2?

ToolBoxV2 ist eine flexible Plattform für Apps, Tools und komplette Anwendungen. Sie können sie lokal auf Ihrem Computer, als Web-App oder als Desktop-App nutzen.

## Installation

### Methode 1: Automatischer Installer (Empfohlen)

**Windows:**
```powershell
irm \"https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/refs/heads/master/installer.ps1\" | iex
```

**Linux/macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/refs/heads/master/installer.sh | bash
```

### Methode 2: Via Python (pip)

Wenn Sie bereits Python installiert haben:

```bash
pip install toolboxv2
```

# ToolBoxV2 — Onboarding

Von `pip install` zu *angemeldet + ISAA-Chat* in einem Aufruf — auf Desktop,
Server und in Colab/Notebooks. Gute Defaults, alles überschreibbar.

```python
from toolboxv2 import init
app = init(profile="mini", headless=True)   # fertig: login + chat, kein CLI/Web nötig
```

---

## TL;DR

| Ziel | Befehl |
|------|--------|
| Programmatisch / Colab / CI | `init(profile="colab", headless=True)` |
| Desktop (Web/CLI + Tray) | `tb` (nutzt Profil `consumer`) |
| Server (Services + Autostart) | `tb manifest init` → Profil `server`, dann `tb --sm` |
| Manifest von Hand anlegen | `tb manifest init` **oder** `tb -init manifest` |
| Env/Manifest später ändern | `tb manifest set app.profile server` · `tb manifest get …` |

`init()` garantiert beim ersten Aufruf:

1. **Persistentes JWT/Cookie-Secret** wird auto-generiert → Login crasht nie.
2. **Fehlende Env-Vars** werden aus `env-template` + sinnvollen Defaults gefüllt.
3. **Profil-Manifest** wird angelegt (mini / colab / desktop / server).
4. **Offline-DB by default** für headless-Profile → kein MinIO-Retry-Sturm.

---

## Installation

```bash
# Aus PyPI
pip install toolboxv2

# Oder direkt von Git (out of the box)
pip install "git+https://github.com/MarkinHaus/ToolBoxV2.git"
```

Empfohlen in einem venv:

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install toolboxv2
```

---

## Die 4 Profile

`init(profile=…)` nimmt freundliche Namen; sie mappen auf die echten internen
Profile und setzen passende Infra-Defaults:

| Profil    | internes Profil | DB    | offline | Services            | Einsatz |
|-----------|-----------------|-------|---------|---------------------|---------|
| `mini`    | local           | LC    | ✅      | –                   | minimal, eingebettet |
| `colab`   | local           | LC    | ✅      | –                   | Notebooks / CI |
| `desktop` | consumer        | LC    | ✅      | workers             | GUI + CLI + Tray |
| `server`  | server          | RR    | ❌      | workers, db         | 24/7 Stack |

`LC` = lokales Dict (JSON-Datei, keine externe DB). `RR` = Remote Redis.
Offline ⇒ nur SQLite, kein MinIO.

---

## Szenario 1 — Programmatisch / Colab / Notebook (headless)

Kein CLI, kein Web. Nur ein lauffähiges `App`.

```python
from toolboxv2 import init

app = init(profile="colab", headless=True)
```

### Anmelden (lokaler Root, ohne Passwort)

Im headless-Modus gibt es einen impliziten lokalen Root-Admin. Token wird
in-process gemintet — kein HTTP, kein Browser.

```python
import asyncio
from toolboxv2.mods.CloudM.auth.local_admin import ensure_local_admin
from toolboxv2.mods.CloudM.auth.jwt_tokens import _generate_tokens

async def login():
    user   = await ensure_local_admin(app)     # legt Anon-Root an (idempotent)
    tokens = _generate_tokens(user, "local_admin")
    return user, tokens

user, tokens = asyncio.run(login())            # im Notebook: `user, tokens = await login()`
print(user.username, user.level)               # -> root -1
```

### Mit ISAA chatten

```python
isaa = app.get_mod("isaa")

async def chat(prompt: str) -> str:
    return await isaa.mini_task_completion(user_task=prompt)

print(asyncio.run(chat("Fasse ToolBoxV2 in einem Satz zusammen.")))
```

> ISAA braucht einen Modell-Key für echte Antworten (`OPENAI_API_KEY`,
> `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, …) oder ein lokales Ollama. Ohne Key
> lädt ISAA trotzdem; der Fehler kommt erst beim Call, nicht beim Import.

---

## Szenario 2 — Desktop (Web/CLI als Main-Entry + Tray)

```bash
tb manifest init            # einmalig (oder beim ersten `tb` automatisch)
tb manifest set app.profile consumer
tb                          # startet GUI/CLI passend zum Profil
```

- `consumer`/`homelab` → `tb` öffnet die GUI.
- `developer`/`local`   → `tb` öffnet das CLI-Dashboard.
- Tray-Icon begleitet die Session (System-Tray).

CLI-Login (interaktiv, bietet auch Magic-Link / OAuth / Passkey):

```bash
tb login
```

---

## Szenario 3 — Server (Services + Autostart)

```bash
tb manifest init
tb manifest set app.profile server
tb manifest set services.enabled '["workers","db"]'
tb manifest apply           # generiert Sub-Configs aus dem Manifest
tb --sm                     # Service-Manager: startet Worker
tb status                   # Übersicht (DB / API / Worker)
```

Bei `profile=server` zeigt ein nacktes `tb` eine ASCII-Status-Übersicht und
beendet sich — kein interaktives Dashboard.

> Autostart on boot (systemd-Unit aus dem Manifest) ist noch nicht verdrahtet.
> Bis dahin: vorhandene `tb-workers.service` / `setup_tb_server.sh` nutzen.

---

## Konfiguration

### Vars beim Init mitgeben

`env=` überschreibt explizit (gewinnt gegen Defaults und Template):

```python
init(profile="mini", env={
    "OPENAI_API_KEY": "sk-…",
    "APP_BASE_URL":   "http://localhost:8080",
    "FASTMODEL":      "openai/gpt-4o-mini",
})
```

### Eigenes Manifest mitgeben

Als Pfad oder als Dict:

```python
init(profile="server", manifest="/etc/toolbox/tb-manifest.yaml")

init(profile="server", manifest={
    "manifest_version": "1.0.0",
    "app":      {"name": "MyStack", "profile": "server"},
    "database": {"mode": "RR"},
    "services": {"enabled": ["workers", "db"]},
})
```

### Nur vorbereiten, App später bauen

```python
init(profile="mini", create_app=False)     # Secret + env + Manifest, kein App-Boot
# … später:
from toolboxv2 import get_app
app = get_app("my.app")
```

### Manifest nachträglich ändern (programmatisch oder CLI)

```bash
tb manifest get app.profile
tb manifest set database.mode LC          # synct auch .env
tb manifest set app.profile desktop
```

```python
from toolboxv2 import build_manifest
build_manifest("server", save=True)        # überschreibt das aktive Manifest
```

---

## `init()` — Referenz

```python
init(
    profile="mini",        # mini | colab | desktop | server
    headless=True,         # kein CLI/Web; gibt App zurück (oder None bei create_app=False)
    env=None,              # dict: zusätzliche/überschreibende Env-Vars (gewinnt)
    manifest=None,         # Pfad | dict: eigenes Manifest statt Profil-Default
    create_app=True,       # False = nur Env+Secret+Manifest vorbereiten
    persist_secret=True,   # False = Secret nur im Prozess (z. B. ephemeres CI)
) -> App | None
```

Weitere Helfer:

- `ensure_secret(persist=True) -> str` — stabiles JWT/Cookie-Secret, generiert+persistiert beim ersten Aufruf.
- `build_manifest(profile="mini", save=True, path=None) -> TBManifest` — Profil-Manifest erstellen.

### Wo landet das Secret?

In der ersten beschreibbaren Stelle, in dieser Reihenfolge:

1. `$TB_DATA_DIR/.env`
2. `<repo>/.env` (editable install)
3. `./.env` (cwd-Fallback, z. B. bei read-only site-packages)

`TB_JWT_SECRET` aus der Umgebung gewinnt immer. In `TB_ENV=production` wird
**nicht** auto-generiert — dort muss das Secret explizit gesetzt sein.

---

## Gute Env-Defaults (Auszug)

Werden beim Init via `setdefault` gesetzt — schon gesetzte Werte bleiben:

| Var | Default | Zweck |
|-----|---------|-------|
| `TB_ENV` | `development` | dev vs. production |
| `IS_OFFLINE_DB` | `true` (mini/colab/desktop) | nur SQLite, kein MinIO |
| `DB_MODE_KEY` | profilabhängig (`LC`/`RR`) | DB-Backend |
| `APP_BASE_URL` | `http://localhost:8000` | Basis-URL |
| `FASTMODEL` / `COMPLEXMODEL` | `ollama/llama3.1` | ISAA-Fallback-Modelle |
| `DEFAULTMODELEMBEDDING` | `gemini/text-embedding-004` | Embeddings |

Security-Keys (`TB_R_KEY`, `TB_JWT_SECRET`, `TOKEN_SECRET`, …) werden **nicht**
aus dem Template injiziert — sie bleiben user-kontrolliert.

---

## Troubleshooting

| Symptom | Ursache / Fix |
|---------|---------------|
| `TB_JWT_SECRET … not set` | Nur in `TB_ENV=production`. Secret explizit setzen oder `TB_ENV=development`. |
| MinIO-Connection-refused-Spam | `IS_OFFLINE_DB=true` setzen (für mini/colab/desktop default). |
| `Invalid Device Key` | Behoben: stale `device.enc` wird automatisch neu erzeugt. Sonst `device.enc` löschen. |
| `tb -init manifest` Fehler | Behoben. Alternativ `tb manifest init`. |
| Notebook: *event loop already running* | Im Notebook `await coro` statt `asyncio.run(coro)`. |
| ISAA antwortet nicht | Modell-Key setzen (`OPENAI_API_KEY` etc.) oder lokales Ollama. |



## Erster Start

Führen Sie den Befehl `tb` aus. Sie werden gefragt, welches Profil Sie verwenden möchten:

| # | Profil   | Für wen geeignet |
|---|-----------|-----------------|
| 1 | Consumer  | Sie wollen eine App nutzen |
| 2 | Homelab   | Sie möchten mehrere Apps/Features lokal betreiben |
| 3 | Server    | Sie verwalten IT-Infrastruktur |
| 4 | Business  | Sie wollen nur ein schnelles System-Health |
| 5 | Developer | Sie möchten Mods/Features entwickeln |

**Beispiel:** Wählen Sie `1` für Consumer.

## Erste Schritte als Consumer

Nach der Installation und Profilauswahl:

```bash
tb              # Startet Ihre App (öffnet GUI oder Dashboard)
```

### Mods und Features verstehen

- **Mods:** Eigenständige Module/Apps (z.B. CloudM für Cloud-Management)
- **Features:** Pakete von Mods und Konfigurationen (z.B. \"web\" für Web-Interface)

### Status prüfen und Updates

```bash
tb status           # System-Status anzeigen
tb fl status        # Feature-Status prüfen
```

### App starten

```bash
tb gui              # Startet die grafische Oberfläche
```

---

# 2. 🏠 Homelab Setup & Management

## Für wen ist dieses Profil?

Das **Homelab-Profil** ist ideal, wenn Sie:
- Mehrere Apps/Mods lokal betreiben möchten
- Ein privates Dashboard für Ihre Dienste haben wollen
- Features installieren und konfigurieren möchten

## Installation und Setup

### Installation (wie oben beschrieben)
Wählen Sie beim First-Run das Profil **Homelab**.

### Erster Start

```bash
tb                  # Öffnet das interaktive Dashboard
```

Das Dashboard zeigt Ihnen:
- ✅ Aktive Services
- 🔴 Gestoppte Services
- 📊 System-Ressourcen
- 📦 Installierte Mods/Features

## Features installieren

### Verfügbare Features auflisten

```bash
tb fl list          # Listet alle verfügbaren Features
```

### Feature installieren

```bash
tb fl unpack web    # Installiert das Web-Feature Pack
tb fl unpack isaa   # Installiert ISAA (Agent System)
```

## User erstellen und verwalten

### Ersten User erstellen (Lokal)

**Option A: Über Web UI (Empfohlen)**
```bash
tb gui              # Startet die Web UI
```
1. Öffnen Sie die angezeigte URL im Browser
2. Klicken Sie auf \"Registrieren\" oder \"Anmelden\"
3. Wählen Sie einen Auth-Provider (Discord, Google, oder E-Mail)

**Option B: Über CLI (für Admins)**
```bash
tb user list                           # Alle User auflisten
tb user info --username admin          # User-Infos anzeigen
tb user set-level admin root           # User-Level setzen
```

### User Levels

| Level | Beschreibung | Rechte |
|-------|-------------|--------|
| guest | Gast | Lesen nur |
| user | Normaler User | Standard-Zugriff |
| moderator | Moderator | User verwalten |
| admin | Administrator | Alle Admin-Funktionen |
| root | Root-Admin | Volle Kontrolle |

### User löschen

```bash
tb user delete --username john --force
```

## Services verwalten

### Services starten/stoppen

```bash
tb services                     # Service-Manager CLI starten
tb services start               # Alle Services starten
tb services stop                # Alle Services stoppen
```

### Konfiguration anzeigen

```bash
tb manifest show                # Vollständige Konfiguration anzeigen
```

## System \"up and running\" bekommen

### 1. Basis-Setup

```bash
tb                          # Prüft den Status
tb manifest init            # Config-Wizard starten
```

### 2. Services starten

```bash
tb workers start            # Worker-System starten
```

### 3. Web-Interface aktivieren

```bash
tb manifest set nginx.enabled true
tb manifest apply           # Nginx-Konfiguration generieren
```

### 4. Status prüfen

```bash
tb status                   # Detaillierter Service-Status
```

---

# 3. 🖥️ Server Admin Guide

## Für wen ist dieses Profil?

Das **Server-Profil** ist für IT-Administratoren gedacht, die:
- Verteilte Systeme verwalten
- Multi-Node Deployments betreiben
- Remote User Management benötigen
- Produktionssysteme überwachen

## Installation

### Server-Installation

```bash
pip install toolboxv2[web]   # Web-Features inkludieren
```

Oder für Produktion:

```bash
curl -fsSL https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/refs/heads/master/installer.sh | bash
# Wählen Sie Server-Profil
```

## Server Setup

### Konfigurations-Checkliste

```bash
tb manifest init            # Config-Wizard
tb manifest show            # Konfiguration prüfen
tb manifest validate        # Validieren
```

### Produktionseinstellungen

In `tb-manifest.yaml`:

```yaml
app:
  environment: production
  debug: false
  log_level: INFO

database:
  mode: RR                  # Remote Redis (Produktion)
  # mode: CB               # Cluster Blob (für MinIO)

nginx:
  enabled: true
  server_name: \"your-domain.com\"
  ssl_enabled: true
  rate_limit_enabled: true
```

### Konfiguration anwenden

```bash
tb manifest apply           # Generiert alle Sub-Configs (Nginx, etc.)
tb manifest sync            # Sync Services
```

## Remote User Management

### User von Remote Server verwalten

```bash
tb user list                           # Alle User auflisten
tb user info --username john            # User-Details
tb user set-level --username john admin   # User-Level ändern
tb user delete --username john --force    # User löschen
```

### MinIO Credentials verwalten

```bash
tb user rotate-minio --username john    # MinIO-Schlüssel rotieren
tb user revoke-minio --username john    # MinIO-Zugriff entziehen
```

## Deployment

### Worker-System starten

```bash
tb workers start            # Startet alle Worker (HTTP, WebSocket)
tb workers status           # Worker-Status
```

### Service-Manager (Auto-Start)

```bash
tb --sm                     # Service-Manager aktivieren
```

### Distributed Management

```bash
tb manifest set nginx.server_name node1.example.com
tb manifest set app.environment production
tb manifest apply
```

## Monitoring

### System-Health

```bash
tb                          # Zeigt ASCII-Übersicht (Nodes, Services, Load)
tb status                   # Detaillierter Status
```

### Logs

```bash
tb --lm                     # Log-Manager
```
___
### Moderne Implementierung

#### 1. Unified User Manager CLI

```bash
tb user list                           # Alle User auflisten
tb user info --username john            # User-Details
tb user set-level --username john admin   # Level setzen
tb user rotate-minio --username john    # MinIO rotieren
tb user revoke-minio --username john    # MinIO entziehen
tb user delete --username john --force    # Löschen
```

#### 2. CloudM.Auth (OAuth & Passkeys)

```bash
tb manifest set auth.discord.enabled true
tb manifest apply
```

**Features:**
- Discord OAuth2
- Google OAuth2
- WebAuthn Passkeys
- JWT Token Management
- Magic Link Authentication
- Device Invite Codes

#### 3. Web Dashboard

```bash
tb gui                              # Startet Web UI
```

**Features:**
- User-Registrierung/-Login über Browser
- OAuth Anbindung (Discord, Google)
- Passkey-Registrierung
- Magic Links per E-Mail

### Warum die Änderung?

- **Modernere Auth:** OAuth2, Passkeys statt CLI-basierter Auth
- **Bessere UI:** Web Dashboard statt CLI-Wizard
- **Skalierbarkeit:** Multi-worker safe, Cloud-ready
- **Security:** JWT Tokens, Token Blacklisting

---

# 5. 🔧 Cheat Sheet

## Profile-Befehle

```bash
tb                          → Profil-spezifische Aktion
tb manifest set app.profile <profil>  → Profil wechseln
```

## Manifest / Konfiguration

```bash
tb manifest init            → Config-Wizard starten
tb manifest show            → Vollständige Konfiguration anzeigen
tb manifest get <key>       → Wert lesen
tb manifest set <key> <val> → Wert setzen
tb manifest validate        → Fehler prüfen
tb manifest apply           → Nginx + Sub-Configs generieren
tb manifest sync            → Services synchronisieren
```

## Features

```bash
tb fl status                → Feature-Loader Status
tb fl list                  → Installierte + verfügbare Features
tb fl unpack <name>         → Feature installieren
tb fl pack <name>           → Feature packen
```

## Mods

```bash
tb mods                     → Mod-Manager
tb -i <mod>                 → Mod installieren
tb -u <mod>                 → Mod aktualisieren
tb -r <mod>                 → Mod entfernen
```

## Services

```bash
tb status                   → Service-Übersicht
tb services                 → Service-Manager CLI
tb workers                  → Worker-Manager
tb workers start            → Alle Worker starten
tb workers status           → Worker-Status
```

## User Management

```bash
tb user list                           # Alle User auflisten
tb user info --username <name>         # User-Details
tb user set-level --username <name> <level>  # Level setzen
tb user rotate-minio --username <name> # MinIO rotieren
tb user revoke-minio --username <name> # MinIO entziehen
tb user delete --username <name> --force    # Löschen
```

## GUI / Web

```bash
tb gui                      → GUI starten
tb --guide                  → Interaktiver Guide
```

## Docker

```bash
tb --docker                 → In Docker laufen
tb --build                  → Docker Image bauen
```

## Service-Manager

```bash
tb --sm                     → Service-Manager aktivieren
tb --lm                     → Log-Manager
```

## Daten (VORSICHT!)

```bash
tb --delete-config NAME     → Config löschen
tb --delete-data NAME       → Daten löschen
tb --delete-config-all      ⚠️  ALLE Configs löschen
tb --delete-data-all        ⚠️  ALLE Daten löschen
```

---

## Weiterführende Dokumentation

- [Installation Guide](../new/analysis/installation/installation.md)
- [Feature Management](../foundations/feature-management.md)
- [Registry Guide](../toolboxv2/services/registry/README.md)
- [Worker System](../../toolboxv2/utils/workers/README.md)
- [CloudM.Auth Modul](../../toolboxv2/mods/CloudM/Auth.py)

---

**Viel Erfolg mit ToolBoxV2! 🚀**
