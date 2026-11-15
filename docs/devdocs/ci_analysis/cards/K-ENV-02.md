# Karte K-ENV-02

```
Karte:       K-ENV-02
Paket:       P0-SIGNAL – CI-Signal und Installation
Status:      GETESTET
Fix-Typ:     B (Jetzt auf `mcp<2` pinnen (Typ-A-Teil, sofort) und die Migration auf die mcp-2-API als eigene Aufgabe planen? Oder direkt migrieren?)
Fix-Auftrag: FIX-03
```

**Ursache:** `mcp` ist ungepinnt (`mcp>=1.6.0`); die CI installiert 2.x, dort gibt es `Server.list_tools()` nicht mehr.

**Betroffen:** 45 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-ENV-02

**Beleg:** `toolboxv2/mcp_server/server.py:747` `@self.server.list_tools()`; `pyproject.toml:189` und `toolboxv2/mods/isaa/pyproject.toml:28` `mcp>=1.6.0`. CI: `AttributeError: 'Server' object has no attribute 'list_tools'`

**Hypothese:** Abhängigkeits-Drift. Der MCP-Server des Produkts ist gegen die 1.x-Dekorator-API geschrieben; mit mcp 2.3.0 scheitert schon `ToolBoxV2MCPServer()`. Betrifft nicht nur Tests: ein frisch installiertes ISAA hat denselben Fehler.

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_mcp -q   # mit mcp 2.3.0 installiert
```

**Vorher:** `46 failed, 156 passed` – 45× `AttributeError: 'Server' object has no attribute 'list_tools'`, 1× `8791 != 8765` (→ K-API-15)

**Probe-Fix:** `uv pip install 'mcp>=1.6,<2'` (installiert 1.30.0)

**Nachher:** `1 failed, 201 passed` (Rest: K-API-15)

**Nachbarn:** python -m pytest toolboxv2/tests/test_isaa/test_mcp -q → keine neue rote ID

## Test-IDs

- `toolboxv2/tests/test_isaa/test_mcp/test_e2.py` – 23
- `toolboxv2/tests/test_isaa/test_mcp/test_server.py` – 22

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EAPIKeyWorkflow::test_complete_api_key_lifecycle | FAILED | AttributeError | 'Server' object has no attribute 'list_tools'
```
