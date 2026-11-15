# Karte K-API-08

```
Karte:       K-API-08
Paket:       P3-VERALTET – Test veraltet (Schnittstelle absichtlich geändert)
Status:      UNGETESTET (Umschreiben von 20 Tests)
Fix-Typ:     B (Ist die neue Overlay-Struktur der Soll-Zustand? `'thought A' not found in ['thought Athought B']` könnte ein echter Fehler sein (Gedanken werden zusammengeklebt) – gewollt?)
Fix-Auftrag: FIX-28
```

**Ursache:** `test_icli_overlay.py`/`test_icli_logic.py` prüfen eine alte `TaskOverlay`/`IterView`-Struktur (`_focus`, `_left_items`, `_effective_view`, `_right_scroll`, `pending_tool`) und alte Textausgaben.

**Betroffen:** 20 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-08

**Beleg:** CI: `'TaskOverlay' object has no attribute '_left_items'` (6), `'_effective_view'` (5), `'_focus'` (2), `'_right_scroll'. Did you mean: '_agent_scroll'?`, `'IterView' object has no attribute 'pending_tool'. Did you mean: 'pending_tools'?` (2); `test_icli_logic.py`: `'plain text' not found in ''`, `'thought A' not found in ['thought Athought B']`, `'fail' not found in …`, `'more' not found in …`

**Hypothese:** Die Overlay-Oberfläche wurde neu gebaut (mehrere gleichzeitige Tools, andere Scroll-Felder).

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_icli_overlay.py toolboxv2/tests/test_isaa/test_icli_logic.py -q
```

**Vorher:** `16 failed` + `4 failed`

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_isaa/test_icli_logic.py` – 4
- `toolboxv2/tests/test_isaa/test_icli_overlay.py` – 16

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_icli_logic.py::TestIngestChunkTypeRouting::test_reasoning_appended_to_iter | FAILED | AssertionError | 'thought A' not found in ['thought Athought B']
```
