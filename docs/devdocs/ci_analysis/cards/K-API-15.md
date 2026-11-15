# Karte K-API-15

```
Karte:       K-API-15
Paket:       P2-AUTO-MEDIUM – Mittlerer Fix, automatisch
Status:      UNGETESTET (Erwartungswert; WS-Port durch Commit belegt, MCP-Port nicht)
Fix-Typ:     B (MCP-Vorgabeport 8791: gewollt? (WS 8468 ist durch `d1db5c9` belegt und kann ohne Rückfrage angepasst werden.) Außerdem prüft derselbe Test `http[0].port == 8000` – stimmt 8000 noch neben 8467?)
Fix-Auftrag: FIX-18
```

**Ursache:** Zwei Tests erwarten alte Vorgabeports: WebSocket 8100 (jetzt 8468) und MCP 8765 (jetzt 8791).

**Betroffen:** 2 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-15

**Beleg:** `toolboxv2/tests/test_mini/test_manifest/test_manifest_schema.py:106` `assert config.websocket[0].port == 8100` → `assert 8468 == 8100`; `toolboxv2/tests/test_isaa/test_mcp/test_models.py:272` → `8791 != 8765`. Commit `d1db5c9`: „One local port pair: HTTP 8467, WS 8468 (was 5000/8080/8100/5001)“.

**Hypothese:** Die Port-Vereinheitlichung ist dokumentiert und gewollt; `test_port_unification.py` prüft sie bereits (`'8100'` darf nicht mehr vorkommen). Für den MCP-Port 8791 gibt es keinen solchen Beleg.

**Repro:**

```
python -m pytest toolboxv2/tests/test_mini/test_manifest/test_manifest_schema.py toolboxv2/tests/test_isaa/test_mcp/test_models.py -q
```

**Vorher:** je 1 rot

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_isaa/test_mcp/test_models.py` – 1
- `toolboxv2/tests/test_mini/test_manifest/test_manifest_schema.py` – 1

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_mcp/test_models.py::TestMCPConfig::test_default_values | FAILED | AssertionError | 8791 != 8765
```
