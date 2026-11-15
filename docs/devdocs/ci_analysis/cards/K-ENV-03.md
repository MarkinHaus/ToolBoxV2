# Karte K-ENV-03

```
Karte:       K-ENV-03
Paket:       P0-SIGNAL – CI-Signal und Installation
Status:      GETESTET
Fix-Typ:     B (Gehört `waitress` in die Basis-Abhängigkeiten (lokale Oberfläche ist Kernfunktion, Empfehlung) oder soll die CI `.[web]` installieren? Prüfen: installiert der SimpleCore-Installer `toolboxv2[web]`? (`test_installer_protocol` erwartet `ToolBoxV2[cli,web,isaa]`, dann ist nur die CI betroffen.))
Fix-Auftrag: FIX-04
```

**Ursache:** `fast_tb.serve()` importiert `waitress`; das Paket steht nur im Extra `web`, die CI installiert ohne Extras.

**Betroffen:** 8 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-ENV-03

**Beleg:** `toolboxv2/utils/workers/fast_tb.py:1073` `from waitress.server import create_server`; `pyproject.toml:166` (`web = [… "waitress>=3.0.2"]`). Im Testlog: `ModuleNotFoundError: No module named 'waitress'` im Server-Thread, im Test `urllib.error.URLError: <urlopen error [Errno 111] Connection refused>`

**Hypothese:** Der Server-Thread der lokalen Oberfläche stirbt beim Import, der Test wartet und bekommt „Connection refused“. Gilt auch für das Produkt: die Onboarding-Oberfläche (WP1-K3) startet nur, wenn `toolboxv2[web]` installiert ist.

**Repro:**

```
python -m pytest toolboxv2/tests/test_core/test_wizard_e2e_http.py -q
```

**Vorher:** `8 failed, 2 passed … in 81.19s`

**Probe-Fix:** `uv pip install waitress` in der Testumgebung

**Nachher:** `10 passed, 1 warning in 4.77s`

**Nachbarn:** python -m pytest toolboxv2/tests/test_core -q → keine neue rote ID durch waitress

## Test-IDs

- `toolboxv2/tests/test_core/test_wizard_e2e_http.py` – 8

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_core/test_wizard_e2e_http.py::test_browser_onboarding_flow_over_http | FAILED | urllib.error.URLError | <urlopen error [Errno 61] Connection refused>
```
