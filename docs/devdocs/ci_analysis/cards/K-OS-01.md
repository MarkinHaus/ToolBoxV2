# Karte K-OS-01

```
Karte:       K-OS-01
Paket:       P6-PLATTFORM – Plattform- und versionsabhängig
Status:      UNGETESTET (nur unter Windows; Analyseumgebung ist Linux, dort 16 passed bzw. 5 skipped)
Fix-Typ:     A
Fix-Auftrag: FIX-35
```

**Ursache:** Unter Windows schlagen die Tests des Installer-Protokolls (`installer.ps1`) und von `verify_install_windows.ps1` fehl.

**Betroffen:** 14 Tests in 4 Jobs (windows-3.10, windows-3.11, windows-3.12, windows-3.13); Liste in catalog.csv, Spalte karte=K-OS-01

**Beleg:** CI nur Windows (alle 4 Versionen): `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)` (7), `AssertionError: 1 != 0` (2), `FileNotFoundError: … \\Temp\\tb_update_…\\uv.log` (2), `'core_build: "new-build"' not found in 'installer_version: "1.0.0" …'`, `'ToolBoxV2[cli,web,isaa] @ file://' not found in ''`, `test_verify_install_windows: 'FAIL' != 'PASS'`

**Hypothese:** Die Tests starten `installer.ps1` und lesen JSON-Zeilen von stdout. `Expecting value … char 0` heißt: stdout ist leer oder beginnt nicht mit JSON. Kandidaten: Skript bricht früh ab (Exit-Code 1), PowerShell-Version/Ausführungsrichtlinie auf dem Runner, BOM/Kodierung (R9 hat ein Doppel-BOM behoben), oder stdout/stderr vertauscht. **Produktpfad von WP1** (Installation und Kern-Update unter Windows).

**Repro:**

```
python -m pytest toolboxv2/tests/test_core/test_installer_protocol.py toolboxv2/tests/test_core/test_verify_install_windows.py -q   # unter Windows
```

**Vorher:** CI: 14 rot

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_core/test_installer_protocol.py` – 13
- `toolboxv2/tests/test_core/test_verify_install_windows.py` – 1

Beispiel aus `raw/windows-3.10.txt`:

```
toolboxv2/tests/test_core/test_installer_protocol.py::TestCoreUpdateInPlace::test_exit_ok_and_done_event | FAILED | AssertionError | 1 != 0 :
```
