# Karte K-STATE-03

```
Karte:       K-STATE-03
Paket:       P6-PLATTFORM – Plattform- und versionsabhängig
Status:      UNGETESTET (nur auf dem Ubuntu-Runner im Gesamtlauf; lokal als root nicht nachstellbar)
Fix-Typ:     A
Fix-Auftrag: FIX-42
```

**Ursache:** `test_state_system.py::TestTbState::test_get_state_from_app` schreibt nach `/root/toolboxv2/features`.

**Betroffen:** 1 Tests in 4 Jobs (ubuntu-3.10, ubuntu-3.11, ubuntu-3.12, ubuntu-3.13); Liste in catalog.csv, Spalte karte=K-STATE-03

**Beleg:** CI ubuntu (alle 4 Versionen): `PermissionError: [Errno 13] Permission denied: '/root/toolboxv2/features'`. Einzeln lokal (als root): `4 passed`.

**Hypothese:** Ein Pfad wird aus `$HOME`/Startverzeichnis eines früheren Tests gebildet (auf dem Ubuntu-Runner zeigt etwas auf `/root`), oder ein früherer Test hinterlässt `HOME`/`TB_*`-Variablen. Nur ubuntu betroffen, macOS und Windows nicht.

**Repro:**

```
python -m pytest toolboxv2/tests/test_mini/test_system/test_state_system.py -q   # allein grün
```

**Vorher:** nur im CI-Gesamtlauf rot

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_mini/test_system/test_state_system.py` – 1

Beispiel aus `raw/ubuntu-3.10.txt`:

```
toolboxv2/tests/test_mini/test_system/test_state_system.py::TestTbState::test_get_state_from_app | FAILED | PermissionError | [Errno 13] Permission denied: '/root/toolboxv2/features'
```
