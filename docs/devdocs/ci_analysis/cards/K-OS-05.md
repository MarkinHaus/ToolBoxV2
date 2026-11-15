# Karte K-OS-05

```
Karte:       K-OS-05
Paket:       P6-PLATTFORM – Plattform- und versionsabhängig
Status:      UNGETESTET (braucht Windows mit winget/choco und Netz)
Fix-Typ:     B (Sollen Tests echte Pakete installieren (dann eigener, getrennter CI-Job) oder die Paketmanager mocken? Im normalen Lauf verändern sie das System des Entwicklers.)
Fix-Auftrag: FIX-40
```

**Ursache:** `test_depmanager.py` ruft echte Paketmanager auf (winget, choco; lokal apt, nix) und installiert Pakete.

**Betroffen:** 3 Tests in 4 Jobs (windows-3.10, windows-3.11, windows-3.12, windows-3.13); Liste in catalog.csv, Spalte karte=K-OS-05

**Beleg:** CI nur Windows: `winget list returned nothing`, `test_search_winget: False is not true`, `nano not found after install via choco`

**Hypothese:** Abhängig vom Runner-Abbild und vom Netz; winget ist auf `windows-latest` nicht nutzbar wie erwartet.

**Repro:**

```
nur Windows-Runner
```

**Vorher:** CI: 3 rot

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_mini/test_extras/test_depmanager.py` – 3

Beispiel aus `raw/windows-3.10.txt`:

```
toolboxv2/tests/test_mini/test_extras/test_depmanager.py::TestInstallUninstall::test_cycle_choco | FAILED | AssertionError | False is not true : nano not found after install via choco
```
