# Karte K-ENV-04

```
Karte:       K-ENV-04
Paket:       P0-SIGNAL – CI-Signal und Installation
Status:      GETESTET
Fix-Typ:     A
Fix-Auftrag: FIX-02
```

**Ursache:** `toolboxv2` ist in der CI nicht editierbar installiert; ein Unterprozess, der nicht im Checkout startet, importiert die Kopie aus `site-packages`, in der `mods/isaa` fehlt.

**Betroffen:** 1 Tests in 8 Jobs (macos-3.10, macos-3.11, macos-3.12, macos-3.13, ubuntu-3.10, ubuntu-3.11, ubuntu-3.12, ubuntu-3.13); Liste in catalog.csv, Spalte karte=K-ENV-04

**Beleg:** `build.yml:147` `uv pip install . --system`; `pyproject.toml:89` schließt `toolboxv2.mods.[!CDW]*` aus dem Paket aus. Test: `test_shell_tool.py:405-422` schreibt `harness.py` in ein Temp-Verzeichnis und startet es. CI: `AssertionError: 1 != 0 : launcher chain was killed`

**Hypothese:** O5 ist kein Plattform- oder Prozessketten-Problem. `harness.py` liegt in `/tmp/…`, also ist `sys.path[0]` das Temp-Verzeichnis und nicht der Checkout. `from toolboxv2.mods.isaa.module import …` trifft `site-packages/toolboxv2` → `ModuleNotFoundError: No module named 'toolboxv2.mods.isaa'` → Exit-Code 1. Lokal beim Owner grün, weil dort editierbar installiert ist. Unter Windows fällt der Test in der CI nicht auf, weil er dort übersprungen bzw. anders verzweigt wird (nicht in den Windows-Rohlisten).

**Repro:**

```
python -m pytest "toolboxv2/tests/test_isaa/test_shell_tool.py::TestKillProcess::test_killing_the_harness_grandparent_does_not_kill_the_harness" -q
# Gegenprobe ohne pytest:  cd /tmp && python -c "import toolboxv2; print(toolboxv2.__file__)"  → …/site-packages/toolboxv2/__init__.py
```

**Vorher:** `1 failed` – `AssertionError: 1 != 0 : launcher chain was killed`; direkter Aufruf des Harness: `ModuleNotFoundError: No module named 'toolboxv2.mods.isaa'`

**Probe-Fix:** frische venv: `uv pip install -e toolboxv2/mods/isaa` und `uv pip install -e .` (statt `.`)

**Nachher:** `python -m pytest toolboxv2/tests/test_isaa/test_shell_tool.py -q` → `44 passed`; `toolboxv2.__file__` zeigt aus jedem Verzeichnis auf den Checkout

**Nachbarn:** ganze Datei `test_shell_tool.py`: 44 passed

**Notiz:** Zweite Ursache derselben Art → ARCH-02: das editierbare `isaa`-Paket legt die Top-Level-Namen `base`, `extras`, `CodingAgent` in die Umgebung, nicht `toolboxv2.mods.isaa`.

## Test-IDs

- `toolboxv2/tests/test_isaa/test_shell_tool.py` – 1

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_shell_tool.py::TestKillProcess::test_killing_the_harness_grandparent_does_not_kill_the_harness | FAILED | AssertionError | 1 != 0 : launcher chain was killed
```
