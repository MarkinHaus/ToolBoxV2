# Import/Export System

## Datei
`toolboxv2/mods/isaa/module.py:515`

---

## Schnellstart

```python
# Export
success, manifest = await isaa.save_agent(\"my_agent\", \"/path/to/backup/\")

# Import
agent = await isaa.load_agent(\"/path/to/backup/my_agent.tar.gz\")
```

---

## `save_agent()`

**Signatur:** `module.py:515`

```python
async def save_agent(
    self,
    agent_name: str,
    path: str,
    include_checkpoint: bool = True,
    include_tools: bool = True,
    notes: str | None = None,
) -> tuple[bool, AgentExportManifest | str]
```

### Parameter

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|---------------|
| `agent_name` | `str` | - | Name des Agents |
| `path` | `str` | - | Output-Verzeichnis |
| `include_checkpoint` | `bool` | `True` | Agent-State mit speichern |
| `include_tools` | `bool` | `True` | Serialisierte Tools speichern |
| `notes` | `str` | `None` | Optionale Notizen |

### Return

```python
(True, AgentExportManifest)  # Bei Erfolg
(False, \"Error message\")     # Bei Fehler
```

### Logik (Code-Referenz)

```python
# 1. Pfad normalisieren (module.py:542)
if not path.endswith(\".tar.gz\"):
    path = f\"{path}.tar.gz\"

# 2. Agent holen (module.py:548)
agent = await self.get_agent(agent_name)
builder_config = self.agent_data.get(agent_name, {})

# 3. Manifest erstellen (module.py:555)
manifest = AgentExportManifest(
    agent_name=agent_name,
    version=\"1.0\",
    config=builder_config,
    checkpoint_included=include_checkpoint,
    tools_included=include_tools,
    exported_at=datetime.now().isoformat(),
    notes=notes
)

# 4. Archiv erstellen (module.py:570)
# Struktur:
# agent_name.tar.gz/
# ├── manifest.json
# ├── config.json
# ├── checkpoint.json      (optional)
# ├── tools.dill           (optional)
# └── tools_manifest.json  (optional)
```

### Archive-Struktur

```
my_agent.tar.gz/
├── manifest.json          # Export-Metadaten
├── config.json            # AgentConfig (Pydantic)
├── checkpoint.json       # Agent-State (CheckpointManager)
├── tools.dill            # Serialisierte Tools (dill)
└── tools_manifest.json   # Tool-Dependencies
```

### Manifest-Inhalt

```json
{
  \"agent_name\": \"my_agent\",
  \"version\": \"1.0\",
  \"config\": { ... },
  \"checkpoint_included\": true,
  \"tools_included\": true,
  \"exported_at\": \"2024-01-15T10:30:00\",
  \"notes\": \"Production ready\"
}
```

---

## `load_agent()`

**Signatur:** `module.py:655`

```python
async def load_agent(
    self,
    path: str,
    agent_name: str | None = None,
    overwrite: bool = False,
) -> FlowAgent
```

### Parameter

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|---------------|
| `path` | `str` | - | Pfad zur .tar.gz Datei |
| `agent_name` | `str` | `None` | Neuer Name (optional) |
| `overwrite` | `bool` | `False` | Existierenden Agent überschreiben |

### Return
`FlowAgent` - Geladene Agent-Instanz

### Logik (Code-Referenz)

```python
# 1. Archiv öffnen (module.py:678)
with tarfile.open(path, \"r:gz\") as tar:
    # manifest.json lesen
    manifest_data = json.loads(tar.extractfile(\"manifest.json\").read())
    manifest = AgentExportManifest(**manifest_data)

# 2. Name ermitteln (module.py:690)
final_name = agent_name or manifest.agent_name

# 3. Check ob existiert (module.py:696)
if final_name in row_agent_builder_sto and not overwrite:
    raise ValueError(f\"Agent '{final_name}' exists. Use overwrite=True.\")

# 4. Config laden (module.py:705)
config_data = json.loads(tar.extractfile(\"config.json\").read())
config = AgentConfig(**config_data)
config.name = final_name  # Rename

# 5. Tools laden (optional) (module.py:720)
if manifest.tools_included and \"tools.dill\" in tar.getnames():
    tools = dill.loads(tar.extractfile(\"tools.dill\").read())

# 6. Builder erstellen (module.py:728)
builder = FlowAgentBuilder(config=config)
if tools:
    builder.with_tools(tools)

# 7. Agent bauen (module.py:735)
agent = await builder.build()
await self.register_agent(builder)

# 8. Checkpoint laden (optional) (module.py:742)
if manifest.checkpoint_included:
    checkpoint_data = json.loads(tar.extractfile(\"checkpoint.json\").read())
    await agent.checkpoint_manager.load_checkpoint(checkpoint_data)
```

---

## Warum Archiv-Format?

### Vorteile gegenüber JSON/YAML

| Aspekt | .tar.gz | JSON | SQLite |
|--------|---------|------|--------|
| **Mehrere Dateien** | ✅ | ❌ | ❌ |
| **Binary-Tools** | ✅ dill | ❌ | ⚠️ BLOB |
| **Kompression** | ✅ gzip | ❌ | ❌ |
| **Incremental** | ❌ | ✅ | ✅ |
| **Streaming** | ✅ | ❌ | ❌ |

### Alternative: Plain JSON

```python
# Für einfache Configs ohne Tools/Checkpoint
config = agent.amd.model_dump()
with open(\"agent.json\", \"w\") as f:
    json.dump(config, f, indent=2)
```

---

## Praxis-Beispiele

### 1. Vollständiger Export/Import

```python
# EXPORT
success, manifest = await isaa.save_agent(
    \"production_agent\",
    \"/backups/agents/\",
    include_checkpoint=True,
    include_tools=True,
    notes=\"Version 2.0 - Ready for staging\"
)
print(f\"Exported to: {manifest.backup_path}\")

# IMPORT
loaded_agent = await isaa.load_agent(
    \"/backups/agents/production_agent.tar.gz\",
    agent_name=\"staging_agent\",  # Kopie mit neuem Namen
    overwrite=False
)
```

### 2. Config-only Export (Leichtgewicht)

```python
# Nur Config, kein Checkpoint/Tools
success, manifest = await isaa.save_agent(
    \"template_agent\",
    \"/templates/\",
    include_checkpoint=False,
    include_tools=False
)

# Importieren und anpassen
agent = await isaa.load_agent(\"/templates/template_agent.tar.gz\")
builder = isaa.get_agent_builder(\"template_agent\")
builder.with_models(\"gpt-4o\", \"gpt-4o\")
```

### 3. Backup-Script

```python
import asyncio
from datetime import datetime

async def backup_all_agents(isaa, backup_dir):
    agents = await isaa.get_agents()
    results = []
    
    for agent in agents:
        path = f\"{backup_dir}/{agent.amd.name}_{datetime.now().strftime('%Y%m%d')}.tar.gz\"
        success, manifest = await isaa.save_agent(
            agent.amd.name,
            path,
            notes=f\"Automatic backup\"
        )
        results.append((agent.amd.name, success))
    
    return results
```

---

## Einschränkungen & Bekannte Probleme

### 1. Tools-Serialisierung (dill)

```python
# Problem: Nicht alle Objects sind serialisierbar
# Lösung: Fallback zu Config-only

try:
    tools = dill.loads(tar.extractfile(\"tools.dill\").read())
except Exception as e:
    print(f\"Tool-Deserialisierung fehlgeschlagen: {e}\")
    tools = None  # Config-only
```

### 2. Plattform-spezifische Pfade

```python
# Problem: Pfade wie C:/Users/... nicht portable
# Lösung: Relative Pfade oder Neuberechnung

if platform.system() == \"Windows\":
    wheel_path = \"C:/.../toolboxv2.whl\"
else:
    wheel_path = \"/opt/toolboxv2/toolboxv2.whl\"
```

### 3. Model-Keys nicht exportiert

```python
# Problem: API-Keys sind NICHT im Export
# Grund: Security

# Nach Import: Keys manuell setzen
agent.amd.handler_path_or_dict = {
    \"api_keys\": {
        \"openai\": [\"sk-...\"],
        \"anthropic\": [\"sk-ant-...\"]
    }
}
```

---

## Sicherheitsaspekte

| Was | Gespeichert | NICHT Gespeichert |
|-----|-------------|-------------------|
| Config | ✅ | - |
| Checkpoint | ✅ | - |
| System Message | ✅ | - |
| Tool-Config | ✅ | - |
| API Keys | ❌ | ❌ |
| Credentials | ❌ | ❌ |
| Sensitive Data | ❌ | ❌ |

**Empfehlung:** Nach Import immer Keys setzen!

```python
agent = await isaa.load_agent(\"agent.tar.gz\")
agent.amd.handler_path_or_dict = {\"api_keys\": {\"openai\": [\"sk-REAL-KEY\"]}}
```

---

## Versionierung

### Export-Versionen

| Version | Änderung |
|---------|----------|
| 1.0 | Initiale Version |
| 1.1 | (TODO) Incremental Updates |
| 2.0 | (TODO) Streaming Export |

### Migration

```python
async def migrate_export(manifest_path):
    \"\"\"Konvertiert alte Exports zu neuer Version.\"\"\"
    manifest = load_manifest(manifest_path)
    
    if manifest.version == \"1.0\":
        # 1.0 → 1.1: Nur Metadaten aktualisieren
        manifest.version = \"1.1\"
        manifest.migrated_from = \"1.0\"
    
    return manifest
```

---

## Troubleshooting

### Fehler: \"Agent exists\"

```python
# Lösung 1: overwrite=True
agent = await isaa.load_agent(\"agent.tar.gz\", overwrite=True)

# Lösung 2: Anderen Namen
agent = await isaa.load_agent(\"agent.tar.gz\", agent_name=\"agent_v2\")
```

### Fehler: \"Invalid archive\"

```python
# Prüfen ob Datei vollständig
import os
size = os.path.getsize(\"agent.tar.gz\")
print(f\"Size: {size} bytes\")  # Sollte > 0 sein

# Archiv testen
import tarfile
try:
    with tarfile.open(\"agent.tar.gz\") as tar:
        print(tar.getnames())
except Exception as e:
    print(f\"Corrupted: {e}\")
```

### Fehler: \"Checkpoint corrupted\"

```python
# Fallback: Config-only Import
agent = await isaa.save_agent(
    \"broken_agent.tar.gz\",
    include_checkpoint=False,  # Checkpoint ignorieren
    include_tools=False
)
```

---

## Nächste Schritte

👉 **[FlowAgent API](./FLOWAGENT_API.md)** - Agent zur Laufzeit steuern