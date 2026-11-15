# FIX-31 – aus Karte K-API-10

Paket: **P3-VERALTET** (Test veraltet (Schnittstelle absichtlich geändert)) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 9 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Sind `_scan_disk`/`_dispatch_burst` endgültig weg? Dann die 9 Tests auf das öffentliche Verhalten umschreiben oder entfernen.

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-31  (aus Karte K-API-10)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py   (nur diese ändern)
Vorher lesen:  `MountWatcher` heute
```

**Schritte:**

1. Tests auf beobachtbares Verhalten umstellen (Datei anlegen → im VFS sichtbar), private Methoden nicht mehr aufrufen.

**Nicht tun:** – Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 9 IDs grün oder bestätigt entfernt.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py::TestBurstMode::test_burst_triggers_refresh_mount
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py::TestBurstMode::test_large_changeset_calls_burst_not_individual
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py::TestBurstMode::test_small_changeset_calls_individual_not_burst
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py::TestMountWatcherScanDiff::test_diff_detects_created
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py::TestMountWatcherScanDiff::test_diff_detects_deleted
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py::TestMountWatcherScanDiff::test_diff_detects_modified
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py::TestMountWatcherScanDiff::test_diff_empty_when_nothing_changed
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py::TestMountWatcherScanDiff::test_scan_disk_excludes_pycache
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py::TestMountWatcherScanDiff::test_scan_disk_finds_all_files
```
