# Karte K-API-16

```
Karte:       K-API-16
Paket:       P4-ENTSCHEIDUNG – Braucht Entscheidung des Owners
Status:      UNGETESTET (Erwartungswert-Frage)
Fix-Typ:     B (Ist `workers` jetzt `infrastructure`, und sind die auskommentierten `core`-Dienste endgültig weg?)
Fix-Auftrag: FIX-22
```

**Ursache:** `test_cli_service_manager.py` erwartet `workers` in der Kategorie `core` und mindestens 4 `core`-Dienste; heute ist `workers` `infrastructure`, mehrere `core`-Einträge sind auskommentiert.

**Betroffen:** 2 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-16

**Beleg:** `toolboxv2/utils/clis/service_manager.py:76-196` (Kategorien; `:111-131` auskommentiert). CI: `'infrastructure' != 'core'`, `False is not true` (`len(core_services) >= 4`)

**Hypothese:** Dienste wurden umsortiert, mit `a0e7dbc` (R2) oder früher.

**Repro:**

```
python -m pytest toolboxv2/tests/test_clis/test_cli_service_manager.py -q
```

**Vorher:** `2 failed, 17 passed`

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_clis/test_cli_service_manager.py` – 2

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_clis/test_cli_service_manager.py::TestServiceManagerStatus::test_get_service_info | FAILED | AssertionError | 'infrastructure' != 'core'
```
