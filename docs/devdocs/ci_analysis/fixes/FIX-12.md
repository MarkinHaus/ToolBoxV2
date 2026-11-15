# FIX-12 – aus Karte K-MISS-02

Paket: **P1-AUTO-MINI** (Minimaler Fix, automatisch) · Fix-Typ: **A** · Karte: GETESTET · betrifft 9 Test-IDs

```
Fix:           FIX-12  (aus Karte K-MISS-02)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_isaa/test_base/test_embeddings.py, toolboxv2/tests/test_isaa/test_base/test_llm_router.py, toolboxv2/tests/test_isaa/test_base/test_adapters.py   (nur diese ändern)
Vorher lesen:  alle Vorkommen von `llm_router.` ohne `toolboxv2.`-Präfix
```

**Schritte:**

1. `patch("llm_router.` → `patch("toolboxv2.mods.isaa.base.llm_router.` (7 Stellen).
2. `from llm_router.` → `from toolboxv2.mods.isaa.base.llm_router.` (4 Stellen in `test_adapters.py`).

**Nicht tun:** Keine Assertions ändern. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 8 IDs grün; `test_registers_only_available_providers` bleibt rot mit `ProviderError` → melden, nicht „reparieren“.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_base/test_adapters.py::TestSetupDefaultAdapters::test_ollama_always_registered
toolboxv2/tests/test_isaa/test_base/test_adapters.py::TestSetupDefaultAdapters::test_registers_only_available_providers
toolboxv2/tests/test_isaa/test_base/test_embeddings.py::TestEmbedFunction::test_basic_embed
toolboxv2/tests/test_isaa/test_base/test_embeddings.py::TestEmbedFunction::test_dimension_truncation
toolboxv2/tests/test_isaa/test_base/test_embeddings.py::TestEmbedFunction::test_dimensions_passed_to_router
toolboxv2/tests/test_isaa/test_base/test_embeddings.py::TestEmbedFunction::test_input_type_passed
toolboxv2/tests/test_isaa/test_base/test_embeddings.py::TestEmbedFunction::test_litellm_embed_compat
toolboxv2/tests/test_isaa/test_base/test_embeddings.py::TestEmbedFunction::test_smart_embed_delegates
toolboxv2/tests/test_isaa/test_base/test_llm_router.py::TestRouter::test_resolve_unknown_fallback
```
