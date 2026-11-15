# Karte K-OS-02

```
Karte:       K-OS-02
Paket:       P6-PLATTFORM – Plattform- und versionsabhängig
Status:      UNGETESTET (nur unter Windows)
Fix-Typ:     A
Fix-Auftrag: FIX-36
```

**Ursache:** Unter Windows enthält die Ausgabe des Shell-Werkzeugs den PowerShell-Prompt (`hiPS D:\a\ToolBoxV2\ToolBoxV2>`).

**Betroffen:** 4 Tests in 4 Jobs (windows-3.10, windows-3.11, windows-3.12, windows-3.13); Liste in catalog.csv, Spalte karte=K-OS-02

**Beleg:** CI nur Windows: `AssertionError: 'hiPS D:\\a\\ToolBoxV2\\ToolBoxV2>' != 'hi'`, `'okPS …>' != 'ok'`, `'bPS …>' != 'b'`, `'' != 'legacy'` (4 IDs in `test_shell_tool.py`)

**Hypothese:** Wahrscheinlich Produktfehler: die dauerhafte PowerShell-Sitzung schreibt nach dem Befehl den Prompt, und die Ende-Markierung schneidet ihn nicht ab. Ein Agent bekäme unter Windows bei jedem Befehl den Prompt im Ergebnis.

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_shell_tool.py -q   # unter Windows
```

**Vorher:** CI: 4 rot

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_isaa/test_shell_tool.py` – 4

Beispiel aus `raw/windows-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_shell_tool.py::TestCompletion::test_finished_status_and_exit_code | FAILED | AssertionError | 'hiPS D:\\a\\ToolBoxV2\\ToolBoxV2>' != 'hi'
```
