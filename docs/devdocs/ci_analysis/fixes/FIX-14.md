# FIX-14 – aus Karte K-BUG-02

Paket: **P1-AUTO-MINI** (Minimaler Fix, automatisch) · Fix-Typ: **A** · Karte: GETESTET · betrifft 5 Test-IDs

```
Fix:           FIX-14  (aus Karte K-BUG-02)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/mods/isaa/base/ai_semantic_memory.py   (nur diese ändern)
Vorher lesen:  `add_data`, Zeilen 240–275
```

**Schritte:**

1. Den Block so umstellen:

```python
        elif isinstance(data, str):
            # fix4: cap single-string embeds - O(seq^2) attention blows up on long texts
            if len(data) > 6000:  # ponytail: add real chunking when recall quality drops
                data = data[:6000]
            texts = [data.replace("\\t", "").replace("\t", "")]
        elif isinstance(data, list):
```

**Nicht tun:** Grenze 6000 nicht ändern. Die anderen Zweige nicht anfassen. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- `python -m pytest toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory.py -q` → 0 failed.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory.py::TestAISemanticMemoryV1Compat::test_add_data_v1_signature
toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory.py::TestAISemanticMemoryV1Compat::test_delete_memory
toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory.py::TestAISemanticMemoryV1Compat::test_query_multi_space
toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory.py::TestAISemanticMemoryV1Compat::test_query_to_str
toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory.py::TestAISemanticMemoryV1Compat::test_query_v1_signature
```
