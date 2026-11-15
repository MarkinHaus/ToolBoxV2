# Karte K-API-04

```
Karte:       K-API-04
Paket:       P3-VERALTET – Test veraltet (Schnittstelle absichtlich geändert)
Status:      UNGETESTET (Probe-Fix wäre das Umschreiben von 15 Tests auf die neue Schnittstelle; in der Analysephase nicht gemacht. Ursache je Test durch Einzellauf und Meldung belegt)
Fix-Typ:     B (Sind diese Änderungen gewollt: Auto-Fokus entfernt; `_build_system_prompt` → `_build_static_system_prompt`; Skill `habits_tracking` entfernt; `think` erhöht `max_iterations` anders? Wenn ja: Tests umschreiben. `'Situation' not found in "'async_generator' object is not callable"` sieht nach einem Fehler im `think`-Werkzeug aus – bitte bestätigen oder als Bug führen.)
Fix-Auftrag: FIX-27
```

**Ursache:** Tests der `ExecutionEngine` prüfen entfernte oder umbenannte Teile: `AutoFocusTracker`, `ctx.auto_focus`, `_build_system_prompt`, alte Zählwerte, alte Skill-Liste, alte `think`-Semantik.

**Betroffen:** 15 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-04

**Beleg:** CI: `module '…execution_engine' has no attribute 'AutoFocusTracker'` (3), `'ExecutionContext' object has no attribute 'auto_focus'` (2), `'ExecutionEngine' object has no attribute '_build_system_prompt'. Did you mean: '_build_static_system_prompt'?` (2), `'habits_tracking' not found in {…}`, `10 != 11`, `10 != 5`, `'high' != 'low'`, `2 != 1`, `14 != 12`, `'Situation' not found in "'async_generator' object is not callable"`, `Expected 'mock' to have been called once. Called 2 times.`

**Hypothese:** Die Engine wurde umgebaut (Auto-Fokus entfernt, Systemprompt in statisch/dynamisch geteilt, Vorgabewerte geändert). Jeder Test für sich ist einzeln reproduzierbar rot.

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py toolboxv2/tests/test_isaa/test_base/test_agent/test_executions_engine_tools.py -q
```

**Vorher:** `23 failed` + `6 failed` (davon 14 → K-API-03)

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py` – 9
- `toolboxv2/tests/test_isaa/test_base/test_agent/test_executions_engine_tools.py` – 6

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestAutoFocusTracker::test_get_focus_message | FAILED | AttributeError | module 'toolboxv2.mods.isaa.base.Agent.execution_engine' has no attribute 'AutoFocusTracker'
```
