# Karte K-API-10

```
Karte:       K-API-10
Paket:       P3-VERALTET – Test veraltet (Schnittstelle absichtlich geändert)
Status:      UNGETESTET (Umschreiben von 9 Tests auf private Methoden, die es nicht mehr gibt)
Fix-Typ:     B (Sind `_scan_disk`/`_dispatch_burst` endgültig weg? Dann die 9 Tests auf das öffentliche Verhalten umschreiben oder entfernen.)
Fix-Auftrag: FIX-31
```

**Ursache:** `test_vfs_pooling.py` ruft `MountWatcher._scan_disk` und `_dispatch_burst` auf; beide gibt es nicht mehr.

**Betroffen:** 9 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-10

**Beleg:** CI: `AttributeError: 'MountWatcher' object has no attribute '_scan_disk'` (6), `'_dispatch_burst'` (3)

**Hypothese:** Der Watcher wurde auf ein anderes Verfahren umgestellt (Ereignisse statt Abfragen).

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py -q
```

**Vorher:** 9 dieser IDs rot (plus K-FLAKY-01)

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py` – 9

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py::TestBurstMode::test_burst_triggers_refresh_mount | FAILED | AttributeError | 'MountWatcher' object has no attribute '_dispatch_burst'
```
