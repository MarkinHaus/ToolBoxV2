# Karte K-ENV-02

```
Karte:       K-ENV-02
Paket:       P0-SIGNAL – CI-Signal und Installation
Status:      GETESTET
Fix-Typ:     A
Fix-Auftrag: FIX-03
```

**Ursache:** `mcp` ist ungepinnt (`mcp>=1.6.0`); installiert wird 2.x. Dort gibt es die Dekorator-API (`Server.list_tools()` …) und `mcp.server.fastmcp` nicht mehr, und Modelle liefern Felder nur noch unter snake_case-Namen (`input_schema`, `uri_template`).

**Betroffen:** 45 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-ENV-02

**Beleg:** `toolboxv2/mcp_server/server.py:84, 747-790, 804-815`; `toolboxv2/mods/isaa/base/Agent/flow_agent.py:72, 429, 4518, 4529` (`FastMCP`); `toolboxv2/mods/isaa/extras/mcp_session_manager.py:306, 367, 395-397`; `pyproject.toml:189`, `toolboxv2/mods/isaa/pyproject.toml:28`. CI: `AttributeError: 'Server' object has no attribute 'list_tools'` (45). Unter mcp 2.3 zusätzlich: `No module named 'mcp.server.fastmcp'` und – ohne Fehlermeldung – der ISAA-Client meldet 0 statt 17 Werkzeuge eines MCP-Servers (`tool.inputSchema` wirft `AttributeError`, wird verschluckt).

**Hypothese:** Abhängigkeits-Drift, betrifft Nutzer genauso: der MCP-Server startet nicht, `FlowAgent.setup_mcp_server` meldet „MCP not available“, und ISAA sieht die Werkzeuge angebundener MCP-Server nicht. **Owner-Entscheidung 2026-11-15: auf mcp 2 migrieren** (nicht auf <2 pinnen).

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_mcp toolboxv2/tests/test_mini/test_system/feature_tests_progressive.py -q   # mit mcp 2.3.0
```

**Vorher:** `47 failed, 187 passed, 5 skipped` (45× `'Server' object has no attribute 'list_tools'`, 1× Port K-API-15, 1× `test_disk_not_mutated` – lokal schon vorher rot, nicht im Katalog)

**Probe-Fix:** Patch `probe_mcp2_migration.diff` (5 Dateien): Server-Handler über den Konstruktor (`Server(name, version=…, on_list_tools=…, on_list_resources=…, on_read_resource=…, on_call_tool=…)`), `run_stdio` mit `create_initialization_options`, `FastMCP` → `MCPServer`, Client `input_schema`/`uri_template`/`httpx2`, `mcp>=2.3,<3`

**Nachher:** `2 failed, 232 passed, 5 skipped` (Port K-API-15 und `test_disk_not_mutated`, beide ohne Patch ebenso rot). Protokoll-Probe mit echtem `ClientSession` über In-Memory-Streams: `init server: toolboxv2_mcp 3.0.0`, `tools: 17`, `resources: 5`, `read_resource: TextResourceContents`, `call_tool: False 4`. ISAA-Client `extract_capabilities`: vorher `tools: 0`, nachher `tools: 17`. `FlowAgent.setup_mcp_server`: `server: MCPServer`, `tools: ['agent_run']`, Aufruf liefert `R:hi`.

**Nachbarn:** `test_isaa/test_mcp` und `feature_tests_progressive.py` vollständig; Patch gilt auf `master` @ `ed32daf` und auch nach FIX-17

**Entscheidung des Owners (2026-11-15):** Migration auf mcp 2 (statt Pin <2).

## Test-IDs

- `toolboxv2/tests/test_isaa/test_mcp/test_e2.py` – 23
- `toolboxv2/tests/test_isaa/test_mcp/test_server.py` – 22

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EAPIKeyWorkflow::test_complete_api_key_lifecycle | FAILED | AttributeError | 'Server' object has no attribute 'list_tools'
```
