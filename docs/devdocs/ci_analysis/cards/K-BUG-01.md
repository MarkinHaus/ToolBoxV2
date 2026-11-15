# Karte K-BUG-01

```
Karte:       K-BUG-01
Paket:       P1-AUTO-MINI – Minimaler Fix, automatisch
Status:      GETESTET
Fix-Typ:     A
Fix-Auftrag: FIX-13
```

**Ursache:** `FileScanner._should_include` prüft die Ausschlussliste (`tmp`, `temp`, `build`, `dist`, `target`, …) gegen den **absoluten** Pfad. Liegt das Projekt selbst unter so einem Ordner, wird jede Datei ausgeschlossen.

**Betroffen:** 11 Tests in 4 Jobs (ubuntu-3.10, ubuntu-3.11, ubuntu-3.12, ubuntu-3.13); Liste in catalog.csv, Spalte karte=K-BUG-01

**Beleg:** `toolboxv2/utils/extras/mkdocs.py:2015-2018` (`parts = path.parts`), Liste `:1940-1964`. CI nur ubuntu (Temp = `/tmp/…`): `AssertionError: 'main.py' not found in []`, `0 not greater than 0`, `'rebuilt' != 'loaded'` (11)

**Hypothese:** Echter Produktfehler. Unter Linux liegt `tempfile.mkdtemp()` in `/tmp`, also enthält jeder Pfad den Teil `tmp`. macOS (`/var/folders/…`) und Windows (`…\Temp\…`) treffen die Liste nicht (Groß-/Kleinschreibung). Ein Nutzerprojekt unter `~/build/x` oder `/tmp/x` hätte einen leeren Doku-Index.

**Repro:**

```
python -m pytest toolboxv2/tests/test_mini/test_extras/test_mkdocs.py -q   # Linux
```

**Vorher:** `11 failed, 120 passed`

**Probe-Fix:** in `_should_include` die Teile relativ zu `self.root` bzw. `self.docs_root` bilden (`path.resolve().relative_to(base.resolve()).parts`), Rückfall auf den absoluten Pfad

**Nachher:** `131 passed in 0.91s`

**Nachbarn:** ganze Datei grün

## Test-IDs

- `toolboxv2/tests/test_mini/test_extras/test_mkdocs.py` – 11

Beispiel aus `raw/ubuntu-3.10.txt`:

```
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestDocsSystemIntegration::test_initialize_builds_index | FAILED | AssertionError | 0 not greater than 0
```
