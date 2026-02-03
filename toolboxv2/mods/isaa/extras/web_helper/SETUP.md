# SearXNG Setup Guide 🔍

**Lokale SearXNG Installation für Windows, Linux, Mac**

## Quick Start (Empfohlen)

```bash
# 1. Docker muss installiert sein
# 2. Starten:
python searxng_quick.py

# Fertig! SearXNG läuft auf http://localhost:8080
```

## Setup-Optionen

| Script | Methode | Voraussetzung | Komplexität |
|--------|---------|---------------|-------------|
| `searxng_quick.py` | Docker (single container) | Docker | ⭐ Minimal |
| `searxng_setup.py` | Docker Compose | Docker + Compose | ⭐⭐ Standard |
| `searxng_native_setup.py` | Python venv | Python 3.10+ | ⭐⭐⭐ Ohne Docker |

---

## Option 1: Quick Start (Docker)

**Einfachste Methode - ein Container, ein Befehl.**

```bash
python searxng_quick.py          # Start
python searxng_quick.py --stop   # Stop
python searxng_quick.py --test   # Test API
python searxng_quick.py --status # Status
```

Oder manuell:
```bash
docker run -d --name searxng -p 8080:8080 searxng/searxng:latest
```

---

## Option 2: Docker Compose Setup

**Mehr Kontrolle, persistente Konfiguration.**

```bash
python searxng_setup.py          # Vollständiges Setup
python searxng_setup.py --start  # Start
python searxng_setup.py --stop   # Stop
python searxng_setup.py --test   # Test API
```

Erstellt:
```
~/.searxng/
├── docker-compose.yml
└── searxng/
    └── settings.yml    # Mit JSON API aktiviert
```

---

## Option 3: Native (ohne Docker)

**Für Systeme ohne Docker.**

```bash
python searxng_native_setup.py          # Setup (erstellt venv)
python searxng_native_setup.py --start  # Start
python searxng_native_setup.py --stop   # Stop
```

Erstellt:
```
~/.searxng-native/
├── venv/           # Python Virtual Environment
├── settings.yml
└── searxng.log
```

---

## Docker Installation

### Windows

1. **Docker Desktop** (empfohlen):
   - Download: https://www.docker.com/products/docker-desktop/
   - Installieren → Neustarten → Docker Desktop öffnen

2. **WSL2 Alternative**:
   ```powershell
   wsl --install
   # Dann Docker in WSL2 installieren
   ```

### macOS

```bash
# Option 1: Docker Desktop
# Download: https://www.docker.com/products/docker-desktop/

# Option 2: Homebrew
brew install --cask docker

# Option 3: Colima (lightweight)
brew install colima docker
colima start
```

### Linux

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Ausloggen und wieder einloggen

# Arch
sudo pacman -S docker docker-compose
sudo systemctl enable --now docker

# Fedora
sudo dnf install docker docker-compose
sudo systemctl enable --now docker
```

---

## Konfiguration

Die `settings.yml` wird automatisch erstellt mit:

```yaml
# Wichtig für API-Nutzung:
server:
  limiter: false    # Rate-Limiting deaktiviert

search:
  formats:
    - html
    - json         # ← Erforderlich für API!
```

### Engines aktivieren/deaktivieren

```yaml
engines:
  - name: google
    disabled: false

  - name: bing
    disabled: false

  - name: duckduckgo
    disabled: false

  - name: brave
    disabled: true   # ← Deaktiviert
```

---

## API Testen

```bash
# Browser oder curl:
curl "http://localhost:8080/search?q=test&format=json"

# Python:
python -c "
import httpx
r = httpx.get('http://localhost:8080/search', params={'q': 'test', 'format': 'json'})
print(f'Results: {len(r.json().get(\"results\", []))}')
"
```

---

## Mit WebAgent verwenden

```python
from web_agent import WebAgent

async with WebAgent(
    searxng_url="http://localhost:8080",  # ← Lokale Instanz
    headless=True
) as agent:
    # Suchen
    results = await agent.search.search("Python tutorial")
    print(f"Found {results.total_results} results")

    # Mit Google Dorks
    results = await agent.search.search(
        agent.search.build_dork("API docs", site="github.com")
    )
```

---

## Troubleshooting

### "Docker not found"
```bash
# Prüfen ob Docker läuft:
docker info

# Windows: Docker Desktop starten
# Mac: Docker.app starten oder colima start
# Linux: sudo systemctl start docker
```

### "Connection refused"
```bash
# Container läuft?
docker ps

# Logs prüfen:
docker logs searxng
```

### "No results" oder leere Antwort
```bash
# settings.yml prüfen - json muss in formats sein:
cat ~/.searxng/searxng/settings.yml | grep -A3 formats

# Manuell testen:
curl "http://localhost:8080/search?q=test&format=json"
```

### Port bereits belegt
```bash
# Anderen Port verwenden:
python searxng_quick.py --port 8888
```

---

## Verzeichnisse

| Setup | Verzeichnis |
|-------|-------------|
| Quick | `~/.searxng-quick/` |
| Compose | `~/.searxng/` |
| Native | `~/.searxng-native/` |

---

## Ressourcen

- [SearXNG Dokumentation](https://docs.searxng.org/)
- [SearXNG GitHub](https://github.com/searxng/searxng)
- [SearXNG Docker](https://github.com/searxng/searxng-docker)
- [Öffentliche Instanzen](https://searx.space/) (falls lokal nicht möglich)
