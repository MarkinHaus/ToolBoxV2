# Karte K-OS-06

```
Karte:       K-OS-06
Paket:       P6-PLATTFORM – Plattform- und versionsabhängig
Status:      UNGETESTET (nur unter Windows)
Fix-Typ:     A
Fix-Auftrag: FIX-37
```

**Ursache:** Unter Windows schlagen VFS-Verzeichnisoperationen (`rmdir`, `rm -rf`, `mv` in ein Verzeichnis) und ein Rechte-Test (`'666' != '600'`) fehl.

**Betroffen:** 6 Tests in 4 Jobs (windows-3.10, windows-3.11, windows-3.12, windows-3.13); Liste in catalog.csv, Spalte karte=K-OS-06

**Beleg:** CI nur Windows: `test_vfs_v2.py::TestVFSV2Directories::test_rmdir_force…` `False is not true`, `::TestVfsShellWrite::test_mv_into_directo…`, `::test_rm_rf_directory` `True is not false`; `test_vfs_shell_tool.py::TestShellDirectoryCommands::…`; `test_vfs_session_tools.py::TestDirectoryOps::test_rmdir…`; `test_LiveSync_init.py::TestShareRegistry::test_store_file_i…` `'666' != '600'`

**Hypothese:** (1) Vermutlich Pfadtrenner: das VFS vergleicht Präfixe mit `/`, unter Windows entstehen an einer Stelle `\`-Pfade (z. B. `os.path.join`/`PurePath` statt `posixpath`). Wäre ein Produktfehler unter Windows. (2) `chmod 600` wirkt unter Windows nicht; `stat` meldet 666.

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py -q -k "rmdir or rm_rf or mv_into"   # unter Windows
```

**Vorher:** CI: 6 rot

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_session_tools.py` – 1
- `toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py` – 1
- `toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py` – 3
- `toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_init.py` – 1

Beispiel aus `raw/windows-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_session_tools.py::TestDirectoryOps::test_rmdir_force_removes_contents | FAILED | AssertionError | False is not true
```
