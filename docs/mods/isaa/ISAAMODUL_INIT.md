# ISAA Modul: app.get_mod(\"isaa\")

## Datei
`toolboxv2/mods/isaa/module.py`

---

## Wie funktioniert app.get_mod(\"isaa\")?

### Architektur

```
Application.get_mod(\"isaa\")
         │
         ▼
┌─────────────────────────────────┐
│  ModManager lädt Modul          │
│  toolboxv2/mods/isaa/module.py  │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  ISAA Klasse instanziiert       │
│  mit app-Referenz               │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  ISAA Modul returned            │
│  (Singleton pro App-Instanz)    │
└─────────────────────────────────┘
```

### Code-Referenz

```python
# toolboxv2/utils/toolbox.py:~2294
class Application:
    def get_mod(self, name, spec='app') -> ModuleType or MainToolType:
        if spec != \"app\":
            self.print(f\"Getting Module {name} spec: {spec}\")
        
        # Hier passiert die Magie:
        return self._mods.get(name) or self._load_mod(name)
```

---

## ISAA Klasse: Initialisierung

**Datei:** `toolboxv2/mods/isaa/module.py:100-200`

### \_\_init\_\_ Signatur

```python
class ISAA:
    def __init__(
        self, 
        app: \"Application\", 
        working_directory: str | Path = None,
        config: dict = None,
        **kwargs
    ):
        self.app = app
        self.config = config or {}
        self.working_directory = working_directory
        
        # Agent Registry
        self.agent_data = {}
        
        # Builder Cache (global!)
        global row_agent_builder_sto
        
        # ... mehr Initialisierung
```

### Globale Variablen (Module-Level)

```python
# module.py:96-97
row_agent_builder_sto = {}  # Agent Builder Registry (APP-Lebensdauer)

# Module-Level Constants
Name = \"isaa\"
version = \"0.3.0\"
```

---

## Initialisierungs-Reihenfolge

```
1. app.get_mod(\"isaa\")
       │
       ▼
2. ModManager._load_mod(\"isaa\")
       │
       ▼
3. ISAA.__init__(app) aufrufen
       │
       ├── self.app = app (Referenz auf Application)
       ├── self.config = config oder {}
       ├── self.agent_data = {} (Agent-Konfigurationen)
       ├── self.working_directory = app.data_dir + \"/Agents\"
       │
       ▼
4. Agent Builder Registry initialisieren
       │
       ├── row_agent_builder_sto = {}
       │
       ▼
5. JobScheduler initialisieren (falls vorhanden)
       │
       ▼
6. ISAA Instance in app.mods[\"isaa\"] speichern
```

---

## Wichtige ISAA-Eigenschaften

### Nach Initialisierung verfügbare Daten

```python
isaa = app.get_mod(\"isaa\")

# Application Referenz
isaa.app              # Die Application-Instanz

# Konfiguration
isaa.config           # Dict mit Agent-Konfigurationen
isaa.config[\"FASTMODEL\"]      # Default Fast LLM
isaa.config[\"COMPLEXMODEL\"]  # Default Complex LLM

# Agent Registry
isaa.agent_data       # Dict: agent_name -> config_dict

# Builder Cache
row_agent_builder_sto  # Globaler Cache (nicht isaa.attribut!)

# Arbeitsverzeichnis
isaa.working_directory  # data_dir/Agents
```

---

## Config-Keys (Typische Werte)

```python
# Aus app.config oder .env
{
    \"FASTMODEL\": \"gpt-4o-mini\",
    \"COMPLEXMODEL\": \"gpt-4o\",
    
    # Agent-spezifische Modelle
    \"SELFMODEL\": \"claude-3-5-sonnet\",
    \"ANALYZERMODEL\": \"gpt-4o\",
    
    # Agent-Registry
    \"agents-name-list\": [\"Normal\", \"self\", ...],
    
    # Instance Cache
    \"agent-instance-Normal\": <FlowAgent>,
    \"agent-instance-self\": <FlowAgent>,
}
```

---

## Zugriff auf Manifest

```python
# System-Message für \"self\" Agent aus Manifest
isaa.app.manifest.isaa.self_agent.system_message
```

### Manifest-Struktur

```python
# tb-manifest.yaml
isaa:
  self_agent:
    system_message: |
      Du bist ein autonomer Agent...
  agents:
    - name: Normal
      model: gpt-4o-mini
    - name: self
      model: claude-3-5-sonnet
```

---

## Singleton-Verhalten

### Wichtig: Nur EINE ISAA-Instanz pro App!

```python
# Gleiche Instanz wird returned
isaa1 = app.get_mod(\"isaa\")
isaa2 = app.get_mod(\"isaa\")

assert isa1 is isaa2  # True!

# Änderungen an isaa1 sind auch in isaa2 sichtbar
isaa1.config[\"test\"] = \"value\"
print(is aa2.config[\"test\"])  # \"value\"
```

---

## Praxis: ISAA Modul inspizieren

```python
from toolboxv2 import Application

app = Application()
isaa = app.get_mod(\"isaa\")

# Debug-Ausgabe
print(f\"ISAA Version: {isaa.version}\")
print(f\"Working Dir: {isaa.working_directory}\")
print(f\"Agents: {isaa.config.get('agents-name-list', [])}\")
print(f\"Registered Builders: {list(row_agent_builder_sto.keys())}\")

# Agent erstellen
agent = await isaa.get_agent(\"test_agent\")
```

---

## Verwandte Dokumentation

👉 **[Agent Management](./AGENT_MANAGEMENT.md)** - Agents verwalten
👉 **[Chain System](./CHAIN_SYSTEM.md)** - Agents verketten
👉 **[Import/Export](./IMPORT_EXPORT.md)** - Agent-Backup