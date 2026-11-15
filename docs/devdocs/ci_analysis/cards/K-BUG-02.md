# Karte K-BUG-02

```
Karte:       K-BUG-02
Paket:       P1-AUTO-MINI – Minimaler Fix, automatisch
Status:      GETESTET
Fix-Typ:     A
Fix-Auftrag: FIX-14
```

**Ursache:** In `add_data` wurde die Längenbegrenzung mitten in eine `if/elif`-Kette gesetzt. Jeder String bis 6000 Zeichen fällt in `else: raise ValueError("Unsupported data type")`.

**Betroffen:** 5 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-BUG-02

**Beleg:** `toolboxv2/mods/isaa/base/ai_semantic_memory.py:258-268`: `elif isinstance(data, str): texts = […]` / `if len(data) > 6000: data = data[:6000]` / `elif isinstance(data, list): …` / `else: raise ValueError("Unsupported data type")`. Eingeführt mit `2d51dd4` (2026-09-05). CI: `ValueError: Unsupported data type` (5)

**Hypothese:** Echter Produktfehler: Texte lassen sich nicht mehr ins semantische Gedächtnis schreiben. Das neue `if` beginnt eine eigene Kette; ein kurzer String ist weder > 6000 noch Liste noch dict. Bei langen Strings wird `data` gekürzt, aber `texts` enthält schon den ungekürzten Text.

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory.py -q
```

**Vorher:** `5 failed, 71 passed, 1 skipped`

**Probe-Fix:** Kürzung in den `str`-Zweig ziehen, vor die Zuweisung von `texts`

**Nachher:** `python -m pytest toolboxv2/tests/test_isaa/test_mem/ -q` → `1 failed, 134 passed, 1 skipped` (Rest: K-FLAKY-01)

**Nachbarn:** ganzer Ordner `test_mem`: keine neue rote ID

## Test-IDs

- `toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory.py` – 5

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory.py::TestAISemanticMemoryV1Compat::test_add_data_v1_signature | FAILED | ValueError | Unsupported data type
```
