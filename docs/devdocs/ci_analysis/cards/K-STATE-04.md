# Karte K-STATE-04

```
Karte:       K-STATE-04
Paket:       P6-PLATTFORM – Plattform- und versionsabhängig
Status:      UNGETESTET (unter 3.10 lokal nicht einzeln geprüft; Ursache der verschluckten Ausnahme unbekannt)
Fix-Typ:     A
Fix-Auftrag: FIX-43
```

**Ursache:** `test_job_execution_state.py`: `ValueError: Could not find isaa in ['isaa'] …` – das Modul `isaa` wird im Test über `get_mod` gesucht und nicht geladen.

**Betroffen:** 3 Tests in 6 Jobs (macos-3.10, ubuntu-3.10, windows-3.10, windows-3.11, windows-3.12, windows-3.13); Liste in catalog.csv, Spalte karte=K-STATE-04

**Beleg:** CI: 3 IDs, nur 3.10 (ubuntu, windows) und windows/3.13. Einzeln lokal unter 3.12: `3 passed`.

**Hypothese:** Abhängig von Reihenfolge oder Ladezeit: `get_mod('isaa')` findet das Modul in der Liste, kann es aber nicht laden (Importfehler in `isaa` unter 3.10, z. B. Syntax/Typing, oder Zeitüberschreitung unter Windows). Die eigentliche Ausnahme wird verschluckt („use --debug for infos“).

**Repro:**

```
python3.10 -m pytest toolboxv2/tests/test_core/test_job_execution_state.py -q
```

**Vorher:** CI: 3 rot

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_core/test_job_execution_state.py` – 3

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_core/test_job_execution_state.py::JobExecutionStateTests::test_h1_create_execution_crash_leaves_ghost_run | FAILED | ValueError | Could not find isaa in ['isaa'] pleas install the module, or its posibly broken use --debug for infos
```
