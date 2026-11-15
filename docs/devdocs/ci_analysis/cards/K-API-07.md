# Karte K-API-07

```
Karte:       K-API-07
Paket:       P3-VERALTET – Test veraltet (Schnittstelle absichtlich geändert)
Status:      UNGETESTET (Umschreiben von 19 Tests; experimenteller Bereich)
Fix-Typ:     B (Ist der Wegfall von `OmniSession(fallback=…)` und der Wechsel auf `GeminiLiveBackend` gewollt? Dann Tests umschreiben.)
Fix-Auftrag: FIX-29
```

**Ursache:** `test_omni.py` nutzt eine alte `OmniSession`-Schnittstelle (`fallback=…`, `FakeAgent.a_stream`, `CloudOmniBackend`).

**Betroffen:** 19 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-07

**Beleg:** CI: `TypeError: OmniSession.__init__() got an unexpected keyword argument 'fallback'` (13), `"Error: 'FakeAgent' object has no attribute 'a_stream'" != 'R:hello world'`, `'GeminiLiveBackend' != 'CloudOmniBackend'`, `'failed (new)' != 'completed'`, `0 != 1`, `1 != 2`; `test_omni_audio.py`: `Lists differ: [] != [b'f1', b'f2']`

**Hypothese:** Audio/Omni wurde umgebaut (Fallback-Parameter entfernt, Agent wird über `a_stream` angesprochen, anderes Standard-Backend).

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_audio.py -q
```

**Vorher:** `18 failed` + `1 failed`

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py` – 18
- `toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_audio.py` – 1

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py::TestAgentTools::test_agent_status_reflects_multiple | FAILED | AssertionError | 'failed (new)' != 'completed'
```
