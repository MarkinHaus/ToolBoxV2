# Karte K-API-22

```
Karte:       K-API-22
Paket:       P3-VERALTET – Test veraltet (Schnittstelle absichtlich geändert)
Status:      UNGETESTET (9 Einzelursachen, je Fall nur Meldung und Stelle belegt)
Fix-Typ:     B (Keine Grundsatzfrage; je Fall meldet die Instanz „Test veraltet“ oder „Produktfehler“ mit Beleg, Owner bestätigt gesammelt.)
Fix-Auftrag: FIX-34
```

**Ursache:** Einzelfälle in ISAA-Tests, je eine eigene kleine Abweichung zwischen Test und heutigem Verhalten.

**Betroffen:** 9 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-22

**Beleg:** `test_exec_code_harness.py:212` `ValueError: too many values to unpack (expected 2)`; `test_isaa_session_managers.py::test_15_save_load` `TypeError: Object of type MagicMock is not JSON serializable` (`checkpoint_manager.py:469`); `test_browser.py:228` `expected call not found`; `test_vfs_shell_bullet_proof.py:114` `'\\t' not found in 'col1\tcol2…'`; `test_agent_fetures.py`: `'TestWebAgentToolsLive' object has no attribute 'tool_manager'` (2), `Docs/AutoTest/AutoFix Tools wurden nicht geladen!` (3)

**Hypothese:** Je Fall: Rückgabewert hat ein Element mehr; Attrappe landet im Checkpoint; Aufrufsignatur geändert; Tabs werden jetzt echt geschrieben statt maskiert; `setUp` des Live-Tests bricht vorher ab; Feature-Werkzeuge werden nicht registriert (vermutlich Folge fehlender optionaler Pakete, K-ENV-05).

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_base/test_agent/test_exec_code_harness.py toolboxv2/tests/test_isaa/test_base/test_agent/test_isaa_session_managers.py toolboxv2/tests/test_isaa/test_web/test_browser.py -q
```

**Vorher:** je 1 rot

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py` – 5
- `toolboxv2/tests/test_isaa/test_base/test_agent/test_exec_code_harness.py` – 1
- `toolboxv2/tests/test_isaa/test_base/test_agent/test_isaa_session_managers.py` – 1
- `toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_bullet_proof.py` – 1
- `toolboxv2/tests/test_isaa/test_web/test_browser.py` – 1

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestAutoFixFeatureLive::test_tool_health_checks | FAILED | AssertionError | False is not true : AutoFix Tools wurden nicht geladen!
```
