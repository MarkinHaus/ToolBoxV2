# Karte K-PY-01

```
Karte:       K-PY-01
Paket:       P1-AUTO-MINI – Minimaler Fix, automatisch
Status:      GETESTET
Fix-Typ:     A
Fix-Auftrag: FIX-10
```

**Ursache:** `test_config_wizard.py` lädt `config_wizard` per Dateipfad in `sys.modules`; unter Python 3.10 löst `mock.patch("toolboxv2.utils.clis.config_wizard.…")` den Namen über `getattr` am Elternpaket auf und findet ihn nicht.

**Betroffen:** 28 Tests in 3 Jobs (macos-3.10, ubuntu-3.10, windows-3.10); Liste in catalog.csv, Spalte karte=K-PY-01

**Beleg:** `toolboxv2/tests/test_clis/test_config_wizard.py:47-60`. CI (nur 3.10, alle OS): `AttributeError: module 'toolboxv2.utils.clis' has no attribute 'config_wizard'` (28)

**Hypothese:** Ab 3.11 nutzt `mock` `pkgutil.resolve_name`, das importiert und `sys.modules` beachtet; 3.10 macht `import toolboxv2.utils.clis` und dann `getattr`. Das per `spec_from_file_location` geladene Modul ist aber nie als Attribut am Paket gesetzt worden.

**Repro:**

```
python3.10 -m pytest toolboxv2/tests/test_clis/test_config_wizard.py -q
```

**Vorher:** `32 failed, 36 passed` (lokal 3.10.20; CI: 28 der 32 – dieselbe Meldung)

**Probe-Fix:** nach `_spec.loader.exec_module(_wiz)`: `import toolboxv2.utils.clis as _clis_pkg; _clis_pkg.config_wizard = _wiz`

**Nachher:** `python3.10 -m pytest toolboxv2/tests/test_clis/ -q` → nur noch die 26 minicli- und 2 service_manager-Fehler; `test_config_wizard.py` grün. Unter 3.12 ebenfalls grün.

**Nachbarn:** ganzer Ordner `test_clis` unter 3.10 und 3.12: keine neue rote ID

## Test-IDs

- `toolboxv2/tests/test_clis/test_config_wizard.py` – 28

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_clis/test_config_wizard.py::TestEnvFileIO::test_set_vars_not_prompted_again | FAILED | AttributeError | module 'toolboxv2.utils.clis' has no attribute 'config_wizard'
```
