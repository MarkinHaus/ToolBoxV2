# FIX-30 – aus Karte K-API-09

Paket: **P3-VERALTET** (Test veraltet (Schnittstelle absichtlich geändert)) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 8 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Ist `/system_context.md` absichtlich entfallen? Dann stellen die Tests auf `/vfs_guide.md` um. Nach K-API-02 kommen 2 IDs dazu.

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-30  (aus Karte K-API-09)
Voraussetzung: FIX-17
Dateien:       toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_session.py, test_vfs_session_tools.py (+2 Tests aus test_vfs_v2.py, test_vfs_shell_tool.py)   (nur diese ändern)
Vorher lesen:  wo das VFS seine Systemdateien anlegt
```

**Schritte:**

1. Nach Ja: Systemdatei-Tests auf `/vfs_guide.md` umstellen (Schreibschutz, immer offen, Zähler).
2. Tests, die den Inhalt des Systemkontexts prüfen (`'TestAgent'`, `'System Context'`): prüfen, wo diese Information heute steht, und dort prüfen.

**Nicht tun:** – Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 8 (+2) IDs grün.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_session.py::TestVirtualFileSystem::test_06_system_file_readonly
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_session.py::TestVirtualFileSystem::test_13_context_building
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_session.py::TestVirtualFileSystem::test_14_list_files
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_session_tools.py::TestContextBuilding::test_closed_file_shown_in_closed_count
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_session_tools.py::TestContextBuilding::test_empty_vfs_contains_system_context
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_session_tools.py::TestSystemFiles::test_system_context_always_open
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_session_tools.py::TestSystemFiles::test_system_context_contains_agent_info
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_session_tools.py::TestSystemFiles::test_system_context_readonly
```
