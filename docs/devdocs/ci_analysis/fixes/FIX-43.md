# FIX-43 – aus Karte K-STATE-04

Paket: **P6-PLATTFORM** (Plattform- und versionsabhängig) · Fix-Typ: **A** · Karte: UNGETESTET · betrifft 3 Test-IDs

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-43  (aus Karte K-STATE-04)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       erst Analyse, dann je nach Befund   (nur diese ändern)
Vorher lesen:  `toolboxv2/utils/toolbox.py` (`get_mod`, Meldung „Could not find“), `toolboxv2/flows/isaa/icli.py`
```

**Schritte:**

1. Unter Python 3.10 den Befehl aus `Repro` mit `--debug`/Log-Ausgabe laufen lassen und die verschluckte Ausnahme sichtbar machen.
2. Ist es ein 3.10-Syntax-/Typing-Fehler in ISAA: dort korrigieren. Ist es Reihenfolge: Test abschirmen.

**Nicht tun:** Kein skip für 3.10. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 3 IDs grün unter 3.10 (ubuntu, windows) und windows/3.13.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_core/test_job_execution_state.py::JobExecutionStateTests::test_h1_create_execution_crash_leaves_ghost_run
toolboxv2/tests/test_core/test_job_execution_state.py::JobExecutionStateTests::test_h2_manual_job_marks_completed
toolboxv2/tests/test_core/test_job_execution_state.py::JobExecutionStateTests::test_h3_manual_job_failure_marks_failed
```
