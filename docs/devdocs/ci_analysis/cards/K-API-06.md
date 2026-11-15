# Karte K-API-06

```
Karte:       K-API-06
Paket:       P4-ENTSCHEIDUNG – Braucht Entscheidung des Owners
Status:      UNGETESTET (reine Erwartungswert-Frage)
Fix-Typ:     B (Sind 262 144 als Vorgabe für unbekannte Modelle und 2 000 000 für das geprüfte Modell gewollt?)
Fix-Auftrag: FIX-25
```

**Ursache:** `test_llm_router.py::TestModelInfo` erwartet alte Kontextgrenzen (1 000 000 / 128 000); das Produkt liefert 2 000 000 / 262 144.

**Betroffen:** 3 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-06

**Beleg:** CI: `AssertionError: 2000000 != 1000000`, `262144 != 128000` (2×)

**Hypothese:** Modelltabelle bzw. Vorgabewert für unbekannte Modelle wurde angehoben.

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_base/test_llm_router.py -q
```

**Vorher:** 3 dieser IDs rot

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_isaa/test_base/test_llm_router.py` – 3

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_base/test_llm_router.py::TestModelInfo::test_ctx_limit_known | FAILED | AssertionError | 2000000 != 1000000
```
