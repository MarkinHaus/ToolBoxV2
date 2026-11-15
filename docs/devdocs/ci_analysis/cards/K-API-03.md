# Karte K-API-03

```
Karte:       K-API-03
Paket:       P1-AUTO-MINI – Minimaler Fix, automatisch
Status:      GETESTET
Fix-Typ:     A
Fix-Auftrag: FIX-11
```

**Ursache:** `ExecutionEngine` liest `self.agent.obs`; die Testattrappe `MockAgent` hat das Attribut nicht.

**Betroffen:** 14 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-03

**Beleg:** `toolboxv2/mods/isaa/base/Agent/execution_engine.py:3895` `if self.agent.obs: …` (auch `:3256, :3723, :3789`); `flow_agent.py:435` `self.obs = None`. `toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py:260-267`. CI: `AttributeError: 'MockAgent' object has no attribute 'obs'` (14)

**Hypothese:** Die Beobachtungsschicht kam dazu; echte Agenten haben immer `obs` (auch `None`), die Attrappe nicht.

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py -q
```

**Vorher:** `23 failed, 43 passed`

**Probe-Fix:** in `MockAgent.__init__`: `self.obs = None`

**Nachher:** `11 failed, 55 passed` – 12 der 14 grün; 2 zeigen danach ihre zweite Ursache (`test_think_decrements_iteration: 7 != 6`, `test_think_records_in_auto_focus: … no attribute 'auto_focus'` → K-API-04)

**Nachbarn:** keine neue rote ID außer den zwei Wechslern

## Test-IDs

- `toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py` – 14

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py::TestInternalToolExecution::test_final_answer_is_final | FAILED | AttributeError | 'MockAgent' object has no attribute 'obs'
```
