# FIX-37 – aus Karte K-OS-06

Paket: **P6-PLATTFORM** (Plattform- und versionsabhängig) · Fix-Typ: **A** · Karte: UNGETESTET · betrifft 6 Test-IDs

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-37  (aus Karte K-OS-06)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       VFS-Code unter toolboxv2/mods/isaa/base/Agent/ (Verzeichnisoperationen); toolboxv2/mods/CloudM/LiveSync (Rechte)   (nur diese ändern)
Vorher lesen:  `rmdir`, `rm`, `mv` im VFS; `ShareRegistry` Dateirechte
```

**Schritte:**

1. Unter Windows reproduzieren; im VFS alle Pfadoperationen auf `posixpath` prüfen.
2. Rechte-Test: unter Windows prüfen, was `ShareRegistry` zusichern kann; Produkt (ACL) oder Test (plattformbezogene Erwartung) nach Befund.

**Nicht tun:** Kein skip für Windows ohne Owner-Bestätigung. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 6 IDs grün unter Windows.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_session_tools.py::TestDirectoryOps::test_rmdir_force_removes_contents
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py::TestShellDirectoryCommands::test_rm_rf_directory
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVFSV2Directories::test_rmdir_force
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsShellWrite::test_mv_into_directory
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsShellWrite::test_rm_rf_directory
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_init.py::TestShareRegistry::test_store_file_is_encrypted_and_private
```
