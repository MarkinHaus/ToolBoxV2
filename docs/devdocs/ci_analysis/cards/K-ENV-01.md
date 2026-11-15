# Karte K-ENV-01

```
Karte:       K-ENV-01
Paket:       P0-SIGNAL – CI-Signal und Installation
Status:      GETESTET
Fix-Typ:     A
Fix-Auftrag: FIX-01
```

**Ursache:** Der Fallback `|| python -m unittest discover` entscheidet das Job-Ergebnis; er findet 0 Tests. Bis 3.11 ist das Exit-Code 0 (grün), ab 3.12 Exit-Code 5 (rot).

**Betroffen:** 0 Test-IDs, aber alle 12 Jobs: das Job-Ergebnis sagt nichts über die Tests aus (H0 bestätigt).

**Beleg:** `.github/workflows/build.yml:154`. Log ubuntu/3.10 (Lauf 43742117802): `= 556 failed, 4052 passed, 238 skipped, 38 warnings, 16 errors … =`, danach `Ran 0 tests in 0.000s` / `OK` → Job grün. Log ubuntu/3.11: `= 529 failed, 4103 passed … =`, danach `Ran 0 tests in 0.000s` / `OK` → Job grün. Log ubuntu/3.12: `= 529 failed … =`, `Ran 0 tests in 0.000s`, `NO TESTS RAN`, `Process completed with exit code 5.`

**Hypothese:** pytest schlägt in allen 12 Jobs mit 509–556 Fehlern fehl. `unittest discover -s toolboxv2/tests/` findet nichts, weil die Unterordner keine `__init__.py` haben und die einzige Datei auf oberster Ebene (`test_gitignore.py`) pytest-Funktionen enthält. Python 3.12 hat für „0 Tests gelaufen“ den Exit-Code 5 eingeführt. Die roten 3.12/3.13-Jobs sind also kein 3.12-Problem, und die grünen 3.10/3.11-Jobs sind nicht geprüft.

**Repro:**

```
python3.10 -m unittest discover -s toolboxv2/tests/; echo exit=$?
python3.12 -m unittest discover -s toolboxv2/tests/; echo exit=$?
```

**Vorher:** 3.10: `Ran 0 tests in 0.000s` `OK` exit=0
3.12: `Ran 0 tests in 0.000s` `NO TESTS RAN` exit=5

**Probe-Fix:** build.yml:154 → `run: python -m pytest toolboxv2/tests/ -v --tb=short` (ohne Fallback)

**Nachher:** Nicht als CI-Lauf ausgeführt (Workflow-Änderung ist in der Analysephase verboten). Lokal: pytest allein liefert Exit-Code 1 bei Fehlern, unabhängig von der Python-Version.

**Nachbarn:** entfällt (kein Testcode berührt)
