# FIX-27 – aus Karte K-API-04

Paket: **P3-VERALTET** (Test veraltet (Schnittstelle absichtlich geändert)) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 15 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Sind diese Änderungen gewollt: Auto-Fokus entfernt; `_build_system_prompt` → `_build_static_system_prompt`; Skill `habits_tracking` entfernt; `think` erhöht `max_iterations` anders? Wenn ja: Tests umschreiben. `'Situation' not found in "'async_generator' object is not callable"` sieht nach einem Fehler im `think`-Werkzeug aus – bitte bestätigen oder als Bug führen.

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-27  (aus Karte K-API-04)
Voraussetzung: FIX-11
Dateien:       toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py, toolboxv2/tests/test_isaa/test_base/test_agent/test_executions_engine_tools.py   (nur diese ändern)
Vorher lesen:  `execution_engine.py`: `ExecutionContext`, `HistoryCompressor`, `SkillsManager`, `_build_static_system_prompt`, `think`-Behandlung
```

**Schritte:**

1. Nach Ja des Owners je Punkt: Test auf die neue Schnittstelle umschreiben (gleiche Absicht, neuer Name/Wert).
2. Tests für `AutoFocusTracker` (3) und `auto_focus` (2): nur wenn der Owner bestätigt, dass es die Funktion nicht mehr gibt, entfernen (Regel 1, Ausnahme).
3. `test_06_think_result_structure_has_sections`: zuerst klären, warum das Ergebnis `'async_generator' object is not callable` ist.

**Nicht tun:** Erwartungswerte nicht einfach auf den Ist-Wert setzen, ohne die Absicht des Tests zu lesen. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 15 IDs grün oder mit Owner-Bestätigung entfernt; Bericht je ID.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestAutoFocusTracker::test_get_focus_message
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestAutoFocusTracker::test_max_actions_limit
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestAutoFocusTracker::test_record_action
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestExecutionContext::test_context_creation
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestExecutionContext::test_get_least_relevant_tool
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestExecutionEngineIntegration::test_build_system_prompt_main
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestExecutionEngineIntegration::test_build_system_prompt_sub_agent
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestHistoryCompressor::test_compress_partial
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestSkillsManager::test_init_predefined_skills
toolboxv2/tests/test_isaa/test_base/test_agent/test_executions_engine_tools.py::TestFinalAnswerTool::test_18_final_answer_not_in_auto_focus
toolboxv2/tests/test_isaa/test_base/test_agent/test_executions_engine_tools.py::TestThinkTool::test_03_think_increments_max_iterations
toolboxv2/tests/test_isaa/test_base/test_agent/test_executions_engine_tools.py::TestThinkTool::test_04_think_records_in_auto_focus
toolboxv2/tests/test_isaa/test_base/test_agent/test_executions_engine_tools.py::TestThinkTool::test_06_think_result_structure_has_sections
toolboxv2/tests/test_isaa/test_base/test_agent/test_executions_engine_tools.py::TestThinkTool::test_11_think_multiple_calls_accumulate_iterations
toolboxv2/tests/test_isaa/test_base/test_agent/test_executions_engine_tools.py::TestToolCallIntegrity::test_45_loop_detection_called
```
