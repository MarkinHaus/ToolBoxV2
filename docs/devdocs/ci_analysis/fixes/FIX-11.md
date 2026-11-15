# FIX-11 – aus Karte K-API-03

Paket: **P1-AUTO-MINI** (Minimaler Fix, automatisch) · Fix-Typ: **A** · Karte: GETESTET · betrifft 14 Test-IDs

```
Fix:           FIX-11  (aus Karte K-API-03)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py   (nur diese ändern)
Vorher lesen:  `class MockAgent` (Zeile 260)
```

**Schritte:**

1. In `MockAgent.__init__` nach `self.skills_manager = None`: `self.obs = None  # FlowAgent.obs is optional (flow_agent.py:435)`.

**Nicht tun:** Produktcode nicht ändern. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 12 IDs grün; `test_think_decrements_iteration` und `test_think_records_in_auto_focus` bleiben rot → gehören dann zu K-API-04.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_final_answer_is_final
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_final_answer_not_in_history
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_list_tools_decrements_iteration
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_list_tools_returns_string
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_minimax_style_id_preserved_exactly
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_shift_focus_decrements_5
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_think_decrements_iteration
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_think_en
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_think_live
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_think_not_final
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_think_records_in_auto_focus
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_tool_call_sequence_no_orphans_in_history
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_tool_result_has_correct_structure
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_unknown_tool_no_exception
```
