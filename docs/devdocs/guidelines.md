# ToolBox V2 – Python Architecture & Style Guidelines
**Status:** Battle-Proven / Production Ready
**Version:** 2.1 (The "Anti-Break" Standard)

## 🛑 Das Problem (Warum wir das tun)
Wir lösen hiermit das historische Problem der ToolBox:
> *Update Modul A Besser Machen → Modul B funktioniert nicht mehr → Update B → Modul C kaputt → Update C → Modul A kaputt.*
> *Resultat: Alles wird besser aber 2 Jahre im Kreis drehen nichts grift ineinader und fuktoniert als einheit.*

**Die Lösung:**
1.  **Strikte Isolation:** Worker reden niemals miteinander.
2.  **Versionierung:** Module definieren explizit ihre Abhängigkeiten.
3.  **Clean Architecture:** Facade & Workers Pattern.

---

## 1. Die Architektur-Philosophie: "Facade & Workers"

Jedes Modul (`mod`) besteht aus genau einer **Facade** (der `Tools` Klasse), die von `MainTool` erbt. Diese Klasse ist der **einzige** Punkt, der mit der App oder anderen Modulen interagiert.

### Die 4 Schichten eines Moduls
1.  **Data Layer (`types.py`):** Dumme, speichereffiziente Datencontainer (`__slots__`).
2.  **Worker Layer (`workers.py`):** Zustandslose Logik. Hier passiert die CPU-Arbeit. **Kennt die App nicht.**
3.  **Manager Layer (`manager.py`):** Verwaltet State, Cache, I/O und Thread-Safety.
4.  **Facade Layer (`main.py`):** Erbt von `MainTool` & `FileHandler`. Orchestriert die Layer und stellt die API bereit.

### ⚠️ Die Eiserne Kommunikations-Regel
**Worker kommunizieren NIEMALS direkt mit anderen Workern oder Modulen.**

*   ❌ **Falsch:** `WorkerA` importiert `WorkerB` und ruft `b.process()` auf.
*   ❌ **Falsch:** `WorkerA` greift auf `self.app.get_mod("B")` zu.
*   ✅ **Richtig (Facade Pattern):**
    1.  `WorkerA` gibt Daten an `FacadeA` zurück.
    2.  `FacadeA` nutzt `self.app.a_run_any("ModulB", "funktion", ...)` oder definierte Interfaces.
    3.  `FacadeA` gibt das Ergebnis an `WorkerA` (oder den nächsten Schritt) weiter.

---

## 2. Die 5 Goldenen Code-Regeln

1.  **Speicher-Hygiene:** Nutze `@dataclass(slots=True)` für alle Datenobjekte.
2.  **Echte Asynchronität:** Blockierende Calls (`open`, `json.load`) gehören in `run_in_executor`. Die API ist `async`.
3.  **Data Integrity:** Nutze Atomic Writes (Write temp → `os.replace`).
4.  **Robustheit:** Fehler werden geloggt (`logger.warning`), nicht stillschweigend geschluckt.
5.  **AI-Readiness:** Module müssen "Context-Bundles" (Token-effizient) liefern können, statt ganze Dateien zu dumpen.

---

## 3. Stability & Versioning Strategy

Um den "Update-Kreislauf" zu brechen, muss jedes Modul seine Kompatibilität prüfen, **bevor** es startet.

### A. Versioning Layer (In `__init__.py` oder `main.py`)
```python
REQUIRED_CORE_VERSION = "0.1.25"
REQUIRED_DEPS = {
    "docs_system": "2.1.0",
    "minu": "1.5.0"
}

def check_compatibility(app):
    # Core Version prüfen
    if app.version < REQUIRED_CORE_VERSION:
        app.logger.error(f"Modul {Name} requires Core {REQUIRED_CORE_VERSION}")
        return False

    # Abhängigkeiten prüfen (Soft Checks)
    for mod, ver in REQUIRED_DEPS.items():
        if mod in app.functions:
            # Hier Logik zur Versionsprüfung der anderen Module
            pass
    return True
```

### B. Integration Tests (Automatisch)
Tests simulieren Updates und prüfen die Kette.
```python
async def test_module_chain(app):
    """Sichert ab, dass Update A nicht B bricht."""
    # 1. Modul A ausführen
    res_a = await app.a_run_any("ModulA", "get_data")
    assert res_a is not None

    # 2. Prüfen ob Modul B mit Ergebnis von A noch klarkommt
    res_b = await app.a_run_any("ModulB", "process_data", args=[res_a])
    assert res_b == "Success"
```

---

## 4. Das Implementation Blueprint (Code Template)

### Schritt A: Data Layer (`types.py`)
```python
from dataclasses import dataclass

@dataclass(slots=True)
class AnalysisResult:
    id: str
    score: float
    tags: tuple
```

### Schritt B: Worker Layer (`workers.py`)
Reine Logik. Keine `self.app`. Keine Imports anderer Module.
```python
class LogicWorker:
    def calculate(self, input_data: str) -> AnalysisResult:
        # Komplexe Logik hier
        return AnalysisResult(id="1", score=0.9, tags=("a", "b"))
```

### Schritt C: Manager Layer (`manager.py`)
Thread-Safe I/O und State.
```python
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

class StateManager:
    def __init__(self, path):
        self.path = path
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def save(self, data):
        async with self._lock:
            await asyncio.get_running_loop().run_in_executor(
                self._executor, self._atomic_save, data
            )

    def _atomic_save(self, data):
        # ... temp file write & replace logic ...
        pass
```

### Schritt D: Facade Layer & Registration (`main.py`)
Verwendet das ToolBox-Native Export Pattern und vererbt von `MainTool` & `FileHandler`.

```python
import time
import logging
from pathlib import Path
from toolboxv2 import FileHandler, MainTool, get_app

# --- 1. Export Pattern (Registrierung) ---
Name = 'CloudM'
version = "1.0.0"

# Holt den Magic-Export der Toolbox
export = get_app(f"{Name}.EXPORT").tb

# Registriert Instanzen
no_test = export(mod_name=Name, test=False, version=version)
to_api = export(mod_name=Name, api=True, version=version)

# --- 2. Imports der sauberen Layer ---
from .types import AnalysisResult
from .workers import LogicWorker
from .manager import StateManager

# --- 3. Die Facade Klasse ---
class Tools(MainTool, FileHandler):
    version = version

    def __init__(self, app=None):
        t0 = time.perf_counter()

        # Basis Setup
        self.version = version
        self.name = Name
        self.color = "CYAN"
        if app is None: app = get_app()
        self.logger = app.logger

        # Architektur-Komponenten init
        mod_root = Path(__file__).parent
        self.manager = StateManager(mod_root / "data.json")
        self.worker = LogicWorker()

        # FileHandler Config (modules.config)
        self.keys = {
            "API_ENDPOINT": "https://api.cloudm.com",
            "MAX_RETRIES": 3
        }

        # ToolBox Config Init
        FileHandler.__init__(self,
                             "modules.config",
                             app.id if app else __name__,
                             self.keys,
                             defaults={})

        # Command Mapping
        self.tools = {
            "name": self.name,
            "all": [
                ["status", "Zeigt Modul Status"],
                ["analyze", "Startet Analyse"]
            ],
            # Mapping auf Facade-Methoden (nicht direkt auf Worker!)
            "status": self.cmd_status,
            "analyze": self.cmd_analyze,
            "Version": self.get_version,
            "get_mod_snapshot": self.get_mod_snapshot,
        }

        # MainTool Init
        MainTool.__init__(self,
                          load=self.load_open_file, # Async Entry Point
                          v=self.version,
                          tool=self.tools,
                          name=self.name,
                          logs=self.logger,
                          color=self.color,
                          on_exit=self.on_exit)

        self.logger.info(f"Init {self.name} took {time.perf_counter() - t0:.4f}s")

    async def load_open_file(self):
        """Async Entry Point nach Start der Toolbox."""
        self.load_file_handler() # Config laden

        # Versions-Check (Anti-Break Strategy)
        # if not check_compatibility(self.app): return

        # Manager starten
        await self.manager.load()
        self.logger.info(f"{self.name} Async Ready.")

    # --- Facade Methods (Orchestrierung) ---

    async def cmd_analyze(self, *args):
        """Orchestriert Datenfluss."""
        # 1. State holen
        state = await self.manager.get_state()

        # 2. Worker rufen (Logic)
        result = self.worker.calculate(state)

        # 3. Inter-Modul Kommunikation (NUR HIER!)
        if hasattr(self.app, 'docs_writer'):
            # Wir nutzen die App-Schnittstelle, importieren Docs nicht direkt!
            await self.app.a_run_any("docs_writer", "log_analysis", args=[result])

        # 4. Speichern
        await self.manager.save(result)
        return result

    def cmd_status(self, *args):
        return f"Active. Config: {self.config.get('API_ENDPOINT')}"

    def on_exit(self):
        pass
```

---

## 5. Review-Checkliste (Definition of Done)

Ein Modul ist erst "Production Ready", wenn:

- [ ] **Struktur:** Facade (`Tools`), Worker, Manager und Types sind getrennt.
- [ ] **Isolation:** Der Worker kennt `self.app` nicht.
- [ ] **I/O:** Keine blockierenden Calls im Main-Thread (`run_in_executor` genutzt).
- [ ] **Registration:** Das `export = get_app(...).tb` Pattern wird genutzt.
- [ ] **Integrity:** `__slots__` verwendet und Atomic Writes implementiert.
- [ ] **Compatibility:** `REQUIRED_DEPS` definiert.
