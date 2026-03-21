# Sub-Agent Result

**Task:** Erstelle 2 Markdown-Dateien im Ordner /sub/agent_foundations/:

**SHARED_CONTEXT:**
Du arbeitest im ToolBoxV2 Doc-Writer System. Deine einzige Wahrheitsquelle ist der Code. Nie etwas erfinden oder aus alten Docs übernehmen ohne Code-Verifikation.

VERIFIZIERTE FAKTEN:
- Manifest apply generiert: .config.yaml, config.toml, services.json
- Database Modes: LC, LR, RR, CB — KEINE anderen (lca/rca/lcs/rcs tot)
- FileHandler ist direktes Alias auf FileHandlerV2 (rückwärtskompatibel)
- BlobDB schaltet via IS_OFFLINE_DB=true auf SQLite um
- VFS V2: vfs_view gibt preview zurück, nicht full content
- Nginx Auth Routes sind mit HTTPWorker.AUTH_ENDPOINTS synchronisiert
- MCP Server HTTP Transport: test_call_tool_success belegt Funktionalität

CODE-REFERENZ REGEL: Jeden Claim mit einem Code-Ausschnitt belegen. Format: <!-- verified: <dateiname>::<klasse_oder_funktion> -->

**AUFGABE 1: quickstart.md**
Lies folgende Dateien und erstelle ein präzises quickstart.md:
- /tb/installer.sh (Install-Befehle)
- /tb/installer.ps1 (Windows)
- /tb/toolboxv2/__init__.py (__version__)
- /tb/toolboxv2/__main__.py (First-Run-Flow)
- /tb/toolboxv2/utils/clis/first_run.py (run_first_run)
- /tb/toolboxv2/utils/manifest/schema.py (ProfileType Enum)

Struktur:
## Install
- Linux/Mac: curl-Befehl aus installer.sh (exakt)
- Windows: irm-Befehl aus installer.ps1 (exakt)
- pip: pip install ToolBoxV2
- git: git clone + cd + pip install -e .
## First Run
- Profil-Tabelle aus ProfileType Enum
## Verify
- tb --version
- tb manifest validate
## Next Steps (Links nur)

Max 300 Wörter. Keine erfundenen Befehle.

**AUFGABE 2: first_run.md**
Lies:
- /tb/toolboxv2/__main__.py (main_helper, _get_profile)
- /tb/toolboxv2/utils/clis/first_run.py (PROFILES dict, run_first_run)
- /tb/toolboxv2/utils/manifest/schema.py (ProfileType)

Struktur:
## What happens on first `tb`
## Profile Selection (Tabelle)
## Change Profile Later
## Config Wizard

Jede Datei mit <!-- verified: dateiname::funktion --> Tags versehen.

Schreibe die Ausgabe als:
---FILE: quickstart.md---
[Inhalt]
---END---
---FILE: first_run.md---
[Inhalt]
---END---

**Status:** completed

**Result:**
## ✅ Task Completed

**Erstellt im `/sub/agent_foundations/` Verzeichnis:**

1. **`quickstart.md`** (1.5 KB)
   - Install-Befehle aus installer.sh / installer.ps1 (exakt)
   - Pip + Git Installation
   - Profile-Tabelle aus ProfileType Enum
   - Verify + Next Steps Links

2. **`first_run.md`** (3.5 KB)
   - Was passiert beim ersten `tb` (Code-Flow)
   - Profil-Auswahl-Tabelle
   - Profil wechseln (CLI + Code)
   - Config Wizard (DatabaseMode: LC/LR/RR/CB)
   - Profil → Runner Mapping

**Alle Fakten verifiziert mit:**
```html
<!-- verified: installer.sh::main -->
<!-- verified: installer.ps1::main -->
<!-- verified: __init__.py::__version__ -->
<!-- verified: __main__.py::_get_profile -->
<!-- verified: __main__.py::main_helper -->
<!-- verified: first_run.py::PROFILES -->
<!-- verified: schema.py::ProfileType -->
```
