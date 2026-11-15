# Karte K-MISS-03

```
Karte:       K-MISS-03
Paket:       P1-AUTO-MINI – Minimaler Fix, automatisch
Status:      GETESTET
Fix-Typ:     A
Fix-Auftrag: FIX-15
```

**Ursache:** `test_port_unification.py` baut den Pfad zu `utils/workers/fast_tb.py` vom Repo-Wurzelverzeichnis statt vom Paketordner `toolboxv2/`.

**Betroffen:** 1 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-MISS-03

**Beleg:** `toolboxv2/tests/test_core/test_port_unification.py:45` `base = Path(__file__).resolve().parents[3]`. CI: `FileNotFoundError: … '/home/runner/work/ToolBoxV2/ToolBoxV2/utils/workers/fast_tb.py'`

**Hypothese:** Ein Verzeichnis zu weit oben (`parents[3]` = Repo, `parents[2]` = `toolboxv2`).

**Repro:**

```
python -m pytest toolboxv2/tests/test_core/test_port_unification.py -q
```

**Vorher:** `1 failed, 6 passed`

**Probe-Fix:** `parents[3]` → `parents[2]`

**Nachher:** `7 passed`

**Nachbarn:** ganze Datei grün

## Test-IDs

- `toolboxv2/tests/test_core/test_port_unification.py` – 1

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_core/test_port_unification.py::test_embedded_js_fallback_matches_canon | FAILED | FileNotFoundError | [Errno 2] No such file or directory: '/Users/runner/work/ToolBoxV2/ToolBoxV2/utils/workers/fast_tb.py'
```
