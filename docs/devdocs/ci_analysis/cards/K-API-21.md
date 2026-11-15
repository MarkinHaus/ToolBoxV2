# Karte K-API-21

```
Karte:       K-API-21
Paket:       P3-VERALTET – Test veraltet (Schnittstelle absichtlich geändert)
Status:      UNGETESTET (Umschreiben auf den Router-Mock; 12 der 18 IDs liegen in `test_agent_fetures.py`)
Fix-Typ:     B (Ist `llm_router` der einzige LLM-Weg (kein `llm_handler` mehr)? Soll `parse_media_from_query` das Feld `format` liefern?)
Fix-Auftrag: FIX-26
```

**Ursache:** Tests des LLM-Pfads von `FlowAgent` mocken die alte Schnittstelle (`llm_handler.completion_with_rate_limiting`, litellm-Antworten); der Agent geht heute über `llm_router`. Dazu erwartet ein Test `image_url["format"]`.

**Betroffen:** 18 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-21

**Beleg:** Einzellauf: `flow_agent.py:1474 → :1042 internal_stream → llm_router/router.py:139 → adapters/litellm_fallback.py:28` `ProviderError: litellm is not installed …`; `live_harness.py:819` `TypeError: expected string or bytes-like object, got 'MagicMock'` (CI, 17); `test_flow_agent_class.py:64` `KeyError: 'format'`

**Hypothese:** Der Mock greift nicht mehr; der echte Router läuft und bekommt Attrappen als Modellnamen/Text. In der CI ist die Meldung `… got 'MagicMock'`, einzeln lokal `litellm is not installed` – gleicher Weg, anderes Ende (in der CI ist `litellm` eine Attrappe aus `conftest2.py`).

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_base/test_agent/test_flow_agent_class.py -q
```

**Vorher:** `4 failed, 7 passed`

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py` – 12
- `toolboxv2/tests/test_isaa/test_base/test_agent/test_flow_agent_class.py` – 4
- `toolboxv2/tests/test_isaa/test_base/test_agent/test_flowagent_streaming.py` – 2

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestCodeExecutorFeatureLive::test_A_loader_registers_feature | FAILED | TypeError | expected string or bytes-like object
```
