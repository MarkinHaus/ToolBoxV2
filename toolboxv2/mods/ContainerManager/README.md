# ContainerManager

Docker Container Management für ToolBoxV2 mit User-Isolierung, Persistenz, HTTP-Exposition und SSH-Zugriff.

## Features

- ✅ **User-spezifische Container**: Jeder User bekommt isolierte Container
- ✅ **Persistente Speicherung**: Docker Volumes für Datenpersistenz
- ✅ **Auto-Restart**: Container starten automatisch nach Absturz neu
- ✅ **HTTP Exposition**: nginx Reverse Proxy für Container-Zugriff
- ✅ **SSH-Zugriff**: Integration mit Docksh für sicheren CLI-Zugriff
- ✅ **TBEF.DB Integration**: Persistente Speicherung in der ToolBox-internen DB
- ✅ **CloudM Auth**: Integration mit dem bestehenden Auth-System
- ✅ **Port-Pool**: Automatische Port-Zuweisung (9000-9500)

## Installation

```bash
# Admin Key generieren
python -m toolboxv2.mods.ContainerManager.cli generate-key

# Admin Key setzen
export CONTAINER_ADMIN_KEY=generated-key-here
```

## Container-Typen

| Typ | Beschreibung | Interner Port | SSH | Default Image |
|-----|--------------|---------------|-----|---------------|
| `cli_v4` | Persistente CLI v4 | 8080 | ✅ 2222 | toolboxv2:latest |
| `project_dev` | Streamlit Dev UI | 8501 | ❌ | toolboxv2:dev |
| `preview_server` | HTTP Preview Server | 8600 | ❌ | toolboxv2:latest |
| `custom` | Benutzerdefiniert | 8080 | Optional | toolboxv2:latest |

## CLI Nutzung

```bash
# Container erstellen
python -m toolboxv2.mods.ContainerManager.cli create usr_123 cli_v4

# Container auflisten
python -m toolboxv2.mods.ContainerManager.cli list usr_123

# Alle Container auflisten (Admin)
python -m toolboxv2.mods.ContainerManager.cli list --all

# Container stoppen
python -m toolboxv2.mods.ContainerManager.cli stop abc123def456

# Container starten
python -m toolboxv2.mods.ContainerManager.cli start abc123def456

# Container neustarten
python -m toolboxv2.mods.ContainerManager.cli restart abc123def456

# Logs anzeigen
python -m toolboxv2.mods.ContainerManager.cli logs abc123def456

# Command ausführen
python -m toolboxv2.mods.ContainerManager.cli exec abc123def456 "ls -la /data"

# Container löschen
python -m toolboxv2.mods.ContainerManager.cli delete abc123def456 --force

# ===== SSH BEFEHLE (Docksh Integration) =====

# SSH-aktive Container auflisten
python -m toolboxv2.mods.ContainerManager.cli list-ssh

# SSH-Zugriffsinfos für Container anzeigen
python -m toolboxv2.mods.ContainerManager.cli ssh abc123def456

# SSH Public Key zu Container hinzufügen (für User-Zugriff)
python -m toolboxv2.mods.ContainerManager.cli add-ssh-key abc123def456 "ssh-ed25519 AAAAC3Nza..."
```

## Web UI

```bash
# Streamlit UI starten
streamlit run toolboxv2/mods/ContainerManager/ui.py
```

Die UI bietet:
- **Dashboard**: Übersicht aller Container mit Filter
- **Create**: Container erstellen mit Formular
- **Users**: User-Management mit Container-Zuordnung
- **SSH Keys**: SSH Public Keys zu Containern hinzufügen 🔑
- **Settings**: Admin Key, Container Types, Port Pool, Nginx Integration

### SSH Key Management in der UI

Die "SSH Keys" Seite ermöglicht:

1. **Container-Auswahl**: Wähle den Ziel-Container aus einer Dropdown-Liste
2. **Key-Eingabe**: Paste den SSH Public Key des Users
3. **Key-Validierung**: Automatische Prüfung des Key-Formats
4. **Verbindungs-Infos**: Zeigt Connection String für den User an
5. **Container-Übersicht**: Liste aller SSH-fähigen Container

**Workflow:**
1. User führt `python -m toolboxv2.Docksh.docksh setup` aus
2. User sendet Public Key an Admin
3. Admin fügt Key in UI hinzu
4. User erhält Connection Info und verbindet sich

## REST API

Alle Container-Management-Funktionen sind automatisch als REST API verfügbar, wenn der ToolBox Worker läuft:

```bash
# POST /api/ContainerManager/create_container
{
    "container_type": "cli_v4",
    "user_id": "usr_123",
    "admin_key": "your-admin-key"
}

# GET /api/ContainerManager/list_containers?user_id=usr_123&admin_key=your-admin-key

# GET /api/ContainerManager/get_container?container_id=abc123&admin_key=your-admin-key

# POST /api/ContainerManager/delete_container
{
    "container_id": "abc123",
    "admin_key": "your-admin-key",
    "force": true
}

# GET /api/ContainerManager/container_logs?container_id=abc123&admin_key=your-admin-key

# SSH API Endpoints
# POST /api/ContainerManager/add_ssh_key_to_container
{
    "container_id": "abc123",
    "ssh_public_key": "ssh-ed25519 AAAAC3Nza...",
    "admin_key": "your-admin-key"
}

# GET /api/ContainerManager/get_container_ssh_info?container_id=abc123&admin_key=your-admin-key

# GET /api/ContainerManager/list_ssh_containers?user_id=usr_123&admin_key=your-admin-key
```

## SSH/Docksh Integration

Der ContainerManager ist vollständig mit dem Docksh-System integriert. Container vom Typ `cli_v4` unterstützen SSH-Zugriff für autorisierte User.

### SSH-Zugriff einrichten

1. **User erstellt SSH-Key:**
```bash
# Auf dem Client-Rechner des Users
python -m toolboxv2.Docksh.docksh setup
# Zeigt den Public Key an, den der User dem Admin schicken muss
```

2. **Admin fügt Key zum Container hinzu:**
```bash
# Admin fügt den SSH Public Key zum Container des Users hinzu
python -m toolboxv2.mods.ContainerManager.cli add-ssh-key abc123def456 "ssh-ed25519 AAAAC3Nza..."
```

3. **User verbindet sich via SSH:**
```bash
# User verbindet sich direkt
python -m toolboxv2.Docksh.docksh connect <server-ip> <ssh-port>

# Oder direkt mit SSH
ssh -p 2222 cli@<server-ip>
```

### SSH-Befehle

| Befehl | Beschreibung |
|--------|-------------|
| `list-ssh` | Liste alle Container mit SSH-Zugriff |
| `ssh <container_id>` | Zeige SSH-Infos und öffne Verbindung |
| `add-ssh-key <container_id> <key>` | Füge SSH Public Key hinzu |

### Docksh-Features

- **Persistente tmux-Session:** Die CLI läuft 24/7 im Hintergrund
- **Auto-Recovery:** Bei Absturz startet die CLI automatisch neu
- **Key-basierte Auth:** Keine Passwörter, nur Ed25519 Keys
- **Isolierte Umgebung:** Jeder User hat seinen eigenen Container

## Nginx Integration

Container werden automatisch unter folgenden URLs erreichbar gemacht:

```
http://your-server/container/{user_id}/{container_type}/
```

Beispiele:
- `http://your-server/container/usr_123/cli_v4/`
- `http://your-server/container/usr_456/project_dev/`

Die nginx Configs werden unter `/etc/nginx/box-available/` erstellt und nach `/etc/nginx/box-enabled/` verlinkt.

## Datenstruktur in TBEF.DB

```
CONTAINER::{container_id} → ContainerSpec JSON
CONTAINER_USER::{user_id} → [container_id1, container_id2, ...]
CONTAINER_PORT_POOL → [9001, 9002, 9005, ...]
```

## Sicherheit

- Alle API-Endpunkte erfordern einen Admin Key (außer User-eigene Container)
- Container-Isolation via Docker
- User-spezifische Zuordnung über Labels
- nginx Reverse Proxy mit WS-Support

## Anforderungen

- Docker (laufend)
- nginx (optional, für HTTP Exposition)
- Python 3.10+
- docker-py (`pip install docker`)
