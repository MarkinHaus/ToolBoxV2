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
- [Registry Guide](../services/registry/README.md)
- [Worker System](../../toolboxv2/utils/workers/README.md)
- [CloudM.Auth Modul](../../toolboxv2/mods/CloudM/Auth.py)

---

**Viel Erfolg mit ToolBoxV2! 🚀**
