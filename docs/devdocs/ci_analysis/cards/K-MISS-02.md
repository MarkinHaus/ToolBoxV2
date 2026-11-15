# Karte K-MISS-02

```
Karte:       K-MISS-02
Paket:       P1-AUTO-MINI – Minimaler Fix, automatisch
Status:      GETESTET (8 von 9 grün; 1 ID hat eine zweite Ursache)
Fix-Typ:     A
Fix-Auftrag: FIX-12
```

**Ursache:** Tests patchen bzw. importieren `llm_router.…` als Top-Level-Modul; es heißt `toolboxv2.mods.isaa.base.llm_router.…`.

**Betroffen:** 9 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-MISS-02

**Beleg:** `toolboxv2/tests/test_isaa/test_base/test_embeddings.py:36,48,60,73,86,100` (`@patch("llm_router.embeddings._get_router")`), `test_llm_router.py:177`, `test_adapters.py:422,425,438,441` (`from llm_router.router import …`). CI: `ModuleNotFoundError: No module named 'llm_router'` (9)

**Hypothese:** Die Tests entstanden, als `llm_router` ein eigenständiges Paket war. Die übrigen Imports derselben Dateien nutzen schon den vollen Pfad.

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_base/test_embeddings.py toolboxv2/tests/test_isaa/test_base/test_adapters.py toolboxv2/tests/test_isaa/test_base/test_llm_router.py -q
```

**Vorher:** `12 failed` (9× `No module named 'llm_router'`, 3× K-API-06)

**Probe-Fix:** Präfix `llm_router.` → `toolboxv2.mods.isaa.base.llm_router.` in den Patch-Zielen und in den 4 Imports

**Nachher:** `4 failed`: 3× K-API-06 und neu sichtbar `test_adapters.py::TestSetupDefaultAdapters::test_registers_only_available_providers - ProviderError: No adapter for model 'groq/llama-3.3-70b' and strict_mode=True`

**Nachbarn:** s. Nachher – eine ID wechselt die Fehlermeldung (zweite Ursache, vorher verdeckt)

**Notiz:** Zweite Ursache für `test_registers_only_available_providers`: der Test erwartet einen registrierten groq-Adapter. Nach diesem Fix als eigene Karte unter K-API aufnehmen.

## Test-IDs

- `toolboxv2/tests/test_isaa/test_base/test_adapters.py` – 2
- `toolboxv2/tests/test_isaa/test_base/test_embeddings.py` – 6
- `toolboxv2/tests/test_isaa/test_base/test_llm_router.py` – 1

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_base/test_adapters.py::TestSetupDefaultAdapters::test_ollama_always_registered | FAILED | ModuleNotFoundError | No module named 'llm_router'
```
