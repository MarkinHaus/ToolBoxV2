# FIX-13 – aus Karte K-BUG-01

Paket: **P1-AUTO-MINI** (Minimaler Fix, automatisch) · Fix-Typ: **A** · Karte: GETESTET · betrifft 11 Test-IDs

```
Fix:           FIX-13  (aus Karte K-BUG-01)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/utils/extras/mkdocs.py   (nur diese ändern)
Vorher lesen:  `FileScanner` (`scan`, `_get_search_roots`, `_should_include`), `iter_files` (Zeile 1916)
```

**Schritte:**

1. `_should_include` ersetzen durch:

```python
    def _should_include(self, path: Path) -> bool:
        """Check if file should be included (exclude only check)."""
        # Only directories *inside* the project decide; a project that lives
        # under e.g. /tmp or .../build must not be excluded as a whole.
        parts = path.parts
        for base in (self.root, self.docs_root):
            try:
                parts = path.resolve().relative_to(Path(base).resolve()).parts
                break
            except (ValueError, OSError):
                continue
        return not any(exc in parts for exc in self.exclude_dirs)
```

**Nicht tun:** Ausschlussliste nicht kürzen. `iter_files` nicht ändern. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- `python -m pytest toolboxv2/tests/test_mini/test_extras/test_mkdocs.py -q` → `131 passed` (Linux).
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestDocsSystemIntegration::test_initialize_builds_index
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestDocsSystemIntegration::test_initialize_loads_existing
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestDocsSystemIntegration::test_lookup_code_python
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestDocsSystemIntegration::test_read_all_sections
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestDocsSystemIntegration::test_read_by_query
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestDocsSystemIntegration::test_read_markdown_format
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestDocsSystemLiveWorkflow::test_cross_reference_docs_and_code
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestDocsSystemLiveWorkflow::test_full_documentation_workflow
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestDocsSystemLiveWorkflow::test_multi_language_support
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestFileScanner::test_scan_markdown_files
toolboxv2/tests/test_mini/test_extras/test_mkdocs.py::TestFileScanner::test_scan_python_files
```
