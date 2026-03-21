# ToolBoxV2 Feature Packs Guide

## SHARED_CONTEXT
Du arbeitest im ToolBoxV2 Doc-Writer System. Deine einzige Wahrheitsquelle ist der Code. Nie etwas erfinden oder aus alten Docs übernehmen ohne Code-Verifikation.

**CODE-REFERENZ REGEL**: Jeden Claim mit belegen. Format: `<!-- verified: <dateiname>::<klasse_oder_funktion> -->`

---

# Feature Packs

## Was ist ein Feature Pack?

Ein **Feature Pack** ist ein verteilbares ZIP-Archiv, das:
- Konfigurationsdateien (`feature.yaml`)
- Python-Module
- Ressourcen (Templates, Assets)
- Requirements

...gebündelt als `tbv2-feature-{name}-{version}.zip` enthält.

<!-- verified: feature_loader.py::EXTRA_TO_FEATURES -->

---

## Verfügbare Features

| Feature | PIP Extra | Marker Packages |
|---------|-----------|-----------------|
| `core` | (immer) | - |
| `cli` | `[cli]` | prompt_toolkit, rich, readchar |
| `web` | `[web]` | starlette, uvicorn, httpx |
| `desktop` | `[desktop]` | PyQt6 |
| `exotic` | `[exotic]` | scipy, matplotlib, pandas |
| `isaa` | `[isaa]` | litellm, langchain_core, groq |

<!-- verified: feature_loader.py::FEATURE_DETECTION -->

### Installation

```bash
# Nur Core
pip install toolboxv2

# Mit Web-Support
pip install toolboxv2[web]

# Alle Features
pip install toolboxv2[all]
```

---

## feature.yaml Struktur

Jedes Feature benötigt eine `feature.yaml` im ZIP:

```yaml
# feature.yaml Template
name: mein-feature
version: "1.0.0"
description: "Beschreibung des Features"

# Features die dieses Feature benötigt
dependencies:
  - core

# Zielverzeichnisse für Dateien
targets:
  - name: modules
    path: toolboxv2/mods/
  - name: web
    path: toolboxv2/web/

# Dateien die im ZIP enthalten sind
files:
  - toolboxv2/mods/mein_mod/
  - toolboxv2/web/templates/mein_feature/
```

---

## ZIP-Format Spezifikation

### Erforderliche Dateien
```
tbv2-feature-mein-feature-1.0.0.zip
├── feature.yaml                    # Pflicht!
├── requirements.txt               # Optional
└── files/                          # Optional
    ├── toolboxv2/mods/mein_mod/__init__.py
    ├── toolboxv2/mods/mein_mod/types.py
    └── toolboxv2/mods/mein_mod/manager.py
```

<!-- verified: feature_loader.py::unpack_feature -->

### Entpack-Logik
```python
# feature_loader.py (vereinfacht)
with zipfile.ZipFile(zip_path, 'r') as zf:
    # 1. feature.yaml nach features/{name}/
    if "feature.yaml" in zf.namelist():
        (features_dir / name / "feature.yaml").write_bytes(...)

    # 2. requirements.txt nach features/{name}/
    if "requirements.txt" in zf.namelist():
        (features_dir / name / "requirements.txt").write_bytes(...)

    # 3. files/ nach toolboxv2/{relative_path}
    for name in zf.namelist():
        if name.startswith("files/") and not name.endswith("/"):
            rel_path = name[6:]  # "files/" entfernen
            target_file = package_root / rel_path
            target_file.write_bytes(...)
```

---

## Feature Pack erstellen

### Schritt 1: Vorbereitung

```bash
# Verzeichnisstruktur erstellen
mkdir -p mein-feature
cd mein-feature

# feature.yaml erstellen
cat > feature.yaml << 'EOF'
name: mein-feature
version: "1.0.0"
description: "Mein erstes Feature Pack"
EOF

# requirements.txt erstellen (optional)
cat > requirements.txt << 'EOF'
requests>=2.28.0
EOF

# files/ Verzeichnis erstellen
mkdir -p files/toolboxv2/mods/mein_mod
```

### Schritt 2: Python-Module erstellen

```python
# files/toolboxv2/mods/mein_mod/__init__.py
__version__ = "1.0.0"
# ... Modul Code hier
```


### Schritt 3: Packen mit CLI

<!-- verified: manifest_cli.py::p_pack -->

```bash
# CLI Command
tb manifest pack mein-feature --output ./features_sto/

# Ausgabe
# Created: features_sto/tbv2-feature-mein-feature-1.0.0.zip
```

---

## Feature Pack entpacken & lokal testen

### Option A: CLI Unpack

<!-- verified: manifest_cli.py::p_unpack -->

```bash
# Feature aus ZIP entpacken
tb manifest unpack ./features_sto/tbv2-feature-mein-feature-1.0.0.zip

# Zu spezifischem Ziel entpacken
tb manifest unpack ./mein-feature.zip --target ./my-features/
```

### Option B: Programmatisch entpacken

<!-- verified: feature_loader.py::unpack_feature -->

```python
from toolboxv2.feature_loader import unpack_feature, is_feature_packed

# Prüfe ob Feature verfügbar
if is_feature_packed("mein-feature"):
    # Entpacke
    success = unpack_feature("mein-feature", force=False)
    print(f"Entpackt: {success}")
else:
    print("Feature nicht gefunden")
```

### Option C: Manuell entpacken

```bash
# ZIP öffnen
unzip tbv2-feature-mein-feature-1.0.0.zip -d temp/

# feature.yaml nach toolboxv2/features/mein_feature/ kopieren
mkdir -p toolboxv2/features/mein_feature
cp temp/feature.yaml toolboxv2/features/mein_feature/

# files/ nach toolboxv2/ kopieren
cp -r temp/files/* toolboxv2/

# .installed Marker erstellen
touch toolboxv2/features/mein_feature/.installed

# Aufräumen
rm -rf temp/
```

---

## Feature im Manifest aktivieren

### Automatische Aktivierung (via pip extra)

<!-- verified: feature_loader.py::detect_installed_extras -->

Bei `pip install toolboxv2[web]` werden automatisch:
1. Alle `[web]`-Marker-Pakete geprüft
2. Feature `web` entpackt wenn vorhanden
3. Feature-Dateien nach `toolboxv2/` extrahiert

### Manuelle Aktivierung

<!-- verified: manifest_cli.py::p_enable -->

```bash
# Feature aktivieren
tb manifest enable mein-feature

# Feature deaktivieren
tb manifest disable mein-feature

# Feature Status anzeigen
tb manifest list
```

### Manifest-Eintrag

```yaml
# tb-manifest.yaml
mods:
  installed:
    mein_mod: "^1.0.0"

feature_flags:
  mein-feature:
    enabled: true
    auto_update: true
```

---

## Feature Registry

### Verfügbare Packs anzeigen

<!-- verified: manifest_cli.py::p_packed -->

```bash
# Liste alle verfügbaren Packs
tb manifest packed

# In spezifischem Verzeichnis suchen
tb manifest packed --dir ./my-features/
```

### Von Registry herunterladen

<!-- verified: feature_loader.py::download_feature_from_registry -->

```python
from toolboxv2.feature_loader import download_feature_from_registry

# Feature von registry.simplecore.app herunterladen
path = download_feature_from_registry("mein-feature")
if path:
    print(f"Downloaded: {path}")
```

---

## Troubleshooting

### Feature wird nicht entpackt

1. Prüfe ob ZIP existiert:
   ```bash
   ls toolboxv2/features_packed/
   ```

2. Prüfe feature.yaml im ZIP:
   ```bash
   unzip -l tbv2-feature-mein-feature.zip | grep feature.yaml
   ```

3. Erzwinge Neu-Entpacken:
   ```python
   unpack_feature("mein-feature", force=True)
   ```

### Modul nicht importierbar

1. Prüfe Pfad:
   ```bash
   ls toolboxv2/mods/mein_mod/
   ```

2. Prüfe `__init__.py`:
   ```python
   # toolboxv2/mods/mein_mod/__init__.py muss existieren
   ```

3. Prüfe Import-Pfad:
   ```python
   import toolboxv2.mods.mein_mod
   ```

### Registry Download fehlgeschlagen

1. Prüfe TB_REGISTRY_URL:
   ```bash
   echo $TB_REGISTRY_URL
   ```

2. Prüfe Netzwerk-Verbindung:
   ```bash
   curl https://registry.simplecore.app/api/v1/health
   ```
