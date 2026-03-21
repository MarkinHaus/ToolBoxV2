# How-To: ToolBoxV2 als Consumer nutzen

## TL;DR
Als Consumer installieren und einfach `tb` ausführen.

<!-- verified: __main__.py::_get_profile -->
<!-- verified: __main__.py::ProfileType -->

## Profil: Consumer
Das Consumer-Profil ist für Nutzer, die eine App oder ein Mod verwenden möchten.

<!-- verified: utils/clis/first_run.py::PROFILES -->

## Erste Schritte

### 1. Installation
```bash
pip install toolboxv2
```

### 2. Erster Start
Beim ersten Start wird automatisch das First-Run Onboarding aufgerufen:

<!-- verified: utils/clis/first_run.py::run_first_run -->

Das System erkennt ob ein `app.profile` im Manifest gesetzt ist:
- **Kein Profile** → First-Run wird gestartet
- **Consumer gewählt** → `tb` startet die GUI

### 3. Profil wählen
```
1) Consumer → Ich nutze eine App / ein Mod. Einfach starten.
```

### 4. Normaler Betrieb
```bash
tb              # GUI starten
tb status       # Status anzeigen
```

## Verfügbare Extras

| Extra | Beschreibung |
|-------|---------------|
| `cli` | CLI-Tools |
| `web` | Web-Interface |
| `desktop` | Desktop-Integration |
| `all` | Alle Features |

```bash
pip install toolboxv2[cli]      # Nur CLI
pip install toolboxv2[web]      # Web-Interface
pip install toolboxv2[all]      # Alles
```

## Profile ändern

Falls du das Profil später ändern möchtest:

```bash
# In tb-manifest.yaml
app:
  profile: consumer  # consumer | homelab | server | business | developer
```

<!-- verified: utils/manifest/schema.py::ProfileType -->
