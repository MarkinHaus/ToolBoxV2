# FIX-34 – aus Karte K-API-22

Paket: **P3-VERALTET** (Test veraltet (Schnittstelle absichtlich geändert)) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 9 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Keine Grundsatzfrage; je Fall meldet die Instanz „Test veraltet“ oder „Produktfehler“ mit Beleg, Owner bestätigt gesammelt.

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-34  (aus Karte K-API-22)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       die 5 genannten Testdateien   (nur diese ändern)
Vorher lesen:  je Fall die im Beleg genannte Zeile und die aufgerufene Produktfunktion
```

**Schritte:**

1. Je ID: Einzellauf, Traceback lesen, in einem Satz festhalten, was sich geändert hat.
2. Nur anpassen, wenn die Änderung im Produkt erkennbar gewollt ist (Commit, Docstring). Sonst als Frage melden.

**Nicht tun:** Keine Sammeländerung „Erwartung = Ist“. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 9 IDs grün oder als Frage gemeldet.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestAutoFixFeatureLive::test_tool_health_checks
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestAutoTestFeatureLive::test_tool_health_checks
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestDocsFeatureLive::test_tool_health_checks
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestWebAgentToolsLive::test_all_tools_have_health_coverage
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestWebAgentToolsLive::test_tool_health_checks
toolboxv2/tests/test_isaa/test_base/test_agent/test_exec_code_harness.py::LegacyParseCodeTests::test_h1_new_path_keeps_namespace
toolboxv2/tests/test_isaa/test_base/test_agent/test_isaa_session_managers.py::TestCheckpointManager::test_15_save_load
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_bullet_proof.py::TestWriteReadRoundtrip::test_write_with_tabs
toolboxv2/tests/test_isaa/test_web/test_browser.py::TestOOPWrapperSignatures::test_wrapper_calls_browser_call_tool
```
