# FIX-18 – aus Karte K-API-15

Paket: **P2-AUTO-MEDIUM** (Mittlerer Fix, automatisch) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 2 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: MCP-Vorgabeport 8791: gewollt? (WS 8468 ist durch `d1db5c9` belegt und kann ohne Rückfrage angepasst werden.) Außerdem prüft derselbe Test `http[0].port == 8000` – stimmt 8000 noch neben 8467?

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-18  (aus Karte K-API-15)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_mini/test_manifest/test_manifest_schema.py, toolboxv2/tests/test_isaa/test_mcp/test_models.py   (nur diese ändern)
Vorher lesen:  `toolboxv2/utils/manifest/schema.py` (`WorkersConfig`), `toolboxv2/mcp_server/models.py` (`MCPConfig`), `test_port_unification.py`
```

**Schritte:**

1. `test_manifest_schema.py:106`: `== 8100` → `== 8468`.
2. `test_models.py:272`: erst nach Ja `8765` → `8791`.

**Nicht tun:** Produktports nicht ändern. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 2 IDs grün.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_mcp/test_models.py::TestMCPConfig::test_default_values
toolboxv2/tests/test_mini/test_manifest/test_manifest_schema.py::TestWorkersConfig::test_default_workers
```
