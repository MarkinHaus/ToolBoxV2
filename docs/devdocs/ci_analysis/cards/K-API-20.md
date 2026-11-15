# Karte K-API-20

```
Karte:       K-API-20
Paket:       P3-VERALTET – Test veraltet (Schnittstelle absichtlich geändert)
Status:      UNGETESTET (Richtung offen)
Fix-Typ:     B (Sollen die Browser-Abläufe in pytest laufen (dann Fixture `tf` bauen; braucht Playwright und einen Server → eher eigener CI-Job) oder sind es Hilfsfunktionen (dann umbenennen, z. B. `run_login_…`)?)
Fix-Auftrag: FIX-33
```

**Ursache:** `test_web/test_web.py` enthält Hilfsfunktionen mit dem Namen `test_…(tf, …)`; pytest sammelt sie als Tests und findet keine Fixture `tf`.

**Betroffen:** 3 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-20

**Beleg:** CI: `ERROR … test_web.py::test_login_p…` – `fixture 'tf' not found` (3 ERROR)

**Hypothese:** Die Datei ist ein Skript für ein eigenes Web-Test-Gerüst (`AsyncWebTestFramework`, Zeile 445), kein pytest-Modul.

**Repro:**

```
python -m pytest toolboxv2/tests/test_web/test_web.py -q
```

**Vorher:** `1 passed, 3 errors`

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_web/test_web.py` – 3

Beispiel aus `raw/macos-3.11.txt`:

```
toolboxv2/tests/test_web/test_web.py::test_api_endpoint | ERROR |  | 
```
