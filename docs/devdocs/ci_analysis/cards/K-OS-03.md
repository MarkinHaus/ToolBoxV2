# Karte K-OS-03

```
Karte:       K-OS-03
Paket:       P6-PLATTFORM – Plattform- und versionsabhängig
Status:      UNGETESTET (nur macOS/Windows)
Fix-Typ:     A
Fix-Auftrag: FIX-38
```

**Ursache:** Tests vergleichen Temp-Pfade als Text; das Betriebssystem liefert eine andere Schreibweise desselben Pfads (macOS `/var/…` ↔ `/private/var/…`, Windows `RUNNER~1` ↔ `runneradmin`).

**Betroffen:** 4 Tests in 8 Jobs (macos-3.10, macos-3.11, macos-3.12, macos-3.13, windows-3.10, windows-3.11, windows-3.12, windows-3.13); Liste in catalog.csv, Spalte karte=K-OS-03

**Beleg:** macOS und Windows: `test_endpoint_dist_dir.py` `PosixPath('/var/folders/…') != PosixPath('/private/var/folders/…')` (2). Windows: `test_base_coder.py` `'C:\\Users\\RUNNER~1\\AppData\\Local\\Temp\\coder_cp_…' not found in …`, `Worktree C:\Users\RUNNER~1\… not in: C:/Users/runneradmin/…` (2)

**Hypothese:** Eine Seite ruft `resolve()`/`realpath` auf, die andere nicht. Unter Linux fällt es nicht auf.

**Repro:**

```
nur macOS/Windows
```

**Vorher:** CI: 4 rot

**Probe-Fix:** beide Seiten des Vergleichs mit `Path(...).resolve()` normalisieren (im Test)

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_core/test_endpoint_dist_dir.py` – 2
- `toolboxv2/tests/test_isaa/test_codig/test_base_coder.py` – 2

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_core/test_endpoint_dist_dir.py::TestDistDirManifestBase::test_dist_dir_relative_manifest_value_resolved_against_root | FAILED | AssertionError | PosixPath('/var/folders/h5/t85nd63d71x27_8gl5gw3l_w00[34 chars]ist') != PosixPath('/private/var/folders/h5/t85nd63d71x27_8gl5[42 chars]ist')
```
