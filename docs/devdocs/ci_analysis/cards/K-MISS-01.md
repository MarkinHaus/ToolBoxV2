# Karte K-MISS-01

```
Karte:       K-MISS-01
Paket:       P1-AUTO-MINI – Minimaler Fix, automatisch
Status:      GETESTET
Fix-Typ:     A
Fix-Auftrag: FIX-09
```

**Ursache:** `test_minicli_macros.py` patcht `toolboxv2.flows.minicli.BlobFile`; das Modul heißt `toolboxv2.flows.mini.minicli`.

**Betroffen:** 26 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-MISS-01

**Beleg:** `toolboxv2/tests/test_clis/test_minicli_macros.py:30,96` u. a.; Import in derselben Datei schon richtig (`from toolboxv2.flows.mini.minicli import BeastCLI`, Zeile 20). CI: `AttributeError: module 'toolboxv2.flows' has no attribute 'minicli'` (26)

**Hypothese:** Das Modul wurde nach `flows/mini/` verschoben; die Imports wurden angepasst, die Patch-Ziele (Zeichenketten) nicht.

**Repro:**

```
python -m pytest toolboxv2/tests/test_clis/test_minicli_macros.py -q
```

**Vorher:** `26 failed, 2 passed`

**Probe-Fix:** `sed -i 's/"toolboxv2\.flows\.minicli\./"toolboxv2.flows.mini.minicli./' …` (4 Stellen)

**Nachher:** `28 passed, 1 warning`

**Nachbarn:** python -m pytest toolboxv2/tests/test_clis -q → nur K-API-16 (2) bleibt

## Test-IDs

- `toolboxv2/tests/test_clis/test_minicli_macros.py` – 26

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_clis/test_minicli_macros.py::TestControlFlow::test_execute_macro_commands_break_stops_loop | FAILED | ModuleNotFoundError | No module named 'toolboxv2.flows.minicli'
```
