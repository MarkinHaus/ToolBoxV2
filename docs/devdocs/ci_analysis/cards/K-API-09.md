# Karte K-API-09

```
Karte:       K-API-09
Paket:       P3-VERALTET – Test veraltet (Schnittstelle absichtlich geändert)
Status:      UNGETESTET (Entscheidung nötig, ob Datei zurückkommt oder Tests umgestellt werden)
Fix-Typ:     B (Ist `/system_context.md` absichtlich entfallen? Dann stellen die Tests auf `/vfs_guide.md` um. Nach K-API-02 kommen 2 IDs dazu.)
Fix-Auftrag: FIX-30
```

**Ursache:** Tests erwarten die Systemdatei `/system_context.md` im VFS; heute gibt es nur `/vfs_guide.md`.

**Betroffen:** 8 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-09

**Beleg:** CI: `KeyError: '/system_context.md'`, `'system_context.md' not found in '=== VFS … vfs_guide.md [OPEN]'`, `3 != 4`, `'System Context' not found in …`, `'TestAgent' not found in …`, `True is not false` (Schreibschutz-Test auf der fehlenden Datei)

**Hypothese:** Der Systemkontext wird nicht mehr als VFS-Datei geführt.

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_session.py toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_session_tools.py -q
```

**Vorher:** `3 failed` + `5 failed`

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_session.py` – 3
- `toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_session_tools.py` – 5

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_session.py::TestVirtualFileSystem::test_06_system_file_readonly | FAILED | AssertionError | True is not false
```
