# FIX-08 – aus Karte K-STATE-01

Paket: **P1-AUTO-MINI** (Minimaler Fix, automatisch) · Fix-Typ: **A** · Karte: GETESTET · betrifft 20 Test-IDs

```
Fix:           FIX-08  (aus Karte K-STATE-01)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_mini/test_system/test_session.py   (nur diese ändern)
Vorher lesen:  Datei ganz, besonders `setUp`/`asyncSetUp`/`tearDown` der drei Klassen
```

**Schritte:**

1. Oben `import tempfile` ergänzen und nach den Imports `_TEST_DATA_DIR = tempfile.mkdtemp(prefix="tb_test_session_")`.
2. Beide Stellen `{"TB_DATA_DIR": "/fake/dir"}` → `{"TB_DATA_DIR": _TEST_DATA_DIR}`.
3. Direkt nach jedem `self.env_patcher.start()`: `self.addCleanup(self.env_patcher.stop)`; die `self.env_patcher.stop()`-Zeilen in `tearDown`/`asyncTearDown` entfernen (sonst doppelter Stop).

**Nicht tun:** Keine Assertions ändern. `test_toolboxv2.py` nicht ändern. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- `python -m pytest toolboxv2/tests/test_mini/test_system/test_session.py toolboxv2/tests/test_mini/test_toolboxv2.py -q` → genau 7 rote IDs (K-API-13: 2, K-API-14: 5 inkl. Download), keine ERROR.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionAuth::test_login_invalid_token_triggers_successful_refresh
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionAuth::test_login_valid_token_sets_session_valid
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionAuth::test_login_with_magic_link_sends_correct_payload
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionAuth::test_login_without_token_returns_false
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionAuth::test_verify_magic_link_saves_token_on_success
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionNetwork::test_download_file_writes_chunks_to_disk
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionNetwork::test_fetch_get_adds_auth_headers_and_uses_session
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_01a_crypt
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_01b_file_handler
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_01c_main_tool
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_01d_styles
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_02a_save_instance
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_02b_load_mod
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_02c_get_mod
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_02d_mod_online
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_02e_get_all_mods
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_03a_run
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_03b_all_functions
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_04a_remove_mod
toolboxv2/tests/test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_04b_remove_all_modules
```
