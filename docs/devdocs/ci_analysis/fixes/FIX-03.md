# FIX-03 – aus Karte K-ENV-02

Paket: **P0-SIGNAL** (CI-Signal und Installation) · Fix-Typ: **B** · Karte: GETESTET · betrifft 45 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Jetzt auf `mcp<2` pinnen (Typ-A-Teil, sofort) und die Migration auf die mcp-2-API als eigene Aufgabe planen? Oder direkt migrieren?

```
Fix:           FIX-03  (aus Karte K-ENV-02)
Voraussetzung: keine
Dateien:       pyproject.toml, toolboxv2/mods/isaa/pyproject.toml   (nur diese ändern)
Vorher lesen:  `pyproject.toml` Extra `isaa` (Zeile 189); `toolboxv2/mods/isaa/pyproject.toml` dependencies (Zeile 28); `toolboxv2/mcp_server/server.py:16-19, 747-790`
```

**Schritte:**

1. In beiden Dateien `"mcp>=1.6.0"` durch `"mcp>=1.6.0,<2"` ersetzen.
2. Falls eine `uv.lock`/Requirements-Datei mcp fest nennt: dort ebenfalls <2.

**Nicht tun:** `server.py` nicht anfassen. Keine Tests ändern. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- `python -m pytest toolboxv2/tests/test_isaa/test_mcp -q` → nur noch `test_models.py::TestMCPConfig::test_default_values` rot (K-API-15).
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EAPIKeyWorkflow::test_complete_api_key_lifecycle
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EAPIKeyWorkflow::test_multiple_keys_different_permissions
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2ECaching::test_cache_bypass
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2ECaching::test_cache_hit
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2ECompleteWorkflow::test_admin_workflow
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2ECompleteWorkflow::test_developer_workflow
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EPerformanceTracking::test_performance_metrics_recorded
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EPerformanceTracking::test_performance_to_dict
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EPythonExecution::test_app_available
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EPythonExecution::test_error_handling
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EPythonExecution::test_function_definition_and_use
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EPythonExecution::test_multiline_code
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EPythonExecution::test_persistent_variables
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EPythonExecution::test_simple_expression
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EPythonExecution::test_syntax_error
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EResourceReading::test_read_all_resources
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EServerLifecycle::test_resource_definitions_complete
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EServerLifecycle::test_server_creation_and_configuration
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2EServerLifecycle::test_tool_definitions_complete
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2ESystemInfo::test_toolbox_info_flows
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2ESystemInfo::test_toolbox_info_modules
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2ESystemInfo::test_toolbox_info_python_guide
toolboxv2/tests/test_isaa/test_mcp/test_e2.py::TestE2ESystemInfo::test_toolbox_status
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestLazyLoading::test_not_initialized_on_creation
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestLazyLoading::test_resource_definitions_without_loading
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestLazyLoading::test_tool_definitions_without_loading
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestPerformanceTracking::test_performance_recorded
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestResourceDefinitions::test_get_resource_definitions
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestResourceDefinitions::test_read_resource_discovery
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestResourceDefinitions::test_read_resource_performance
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestResourceDefinitions::test_read_resource_python_guide
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestResourceDefinitions::test_read_resource_unknown
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestResourceDefinitions::test_resource_structure
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolBoxV2MCPServerInit::test_custom_config
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolBoxV2MCPServerInit::test_default_config
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolBoxV2MCPServerInit::test_managers_initialized
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolBoxV2MCPServerInit::test_workers_initialized
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolDefinitions::test_get_tool_definitions_all_enabled
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolDefinitions::test_get_tool_definitions_docs_disabled
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolDefinitions::test_get_tool_definitions_flows_disabled
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolDefinitions::test_get_tool_definitions_python_disabled
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolDefinitions::test_tool_schema_structure
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolExecution::test_call_tool_toolbox_info
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolExecution::test_call_tool_toolbox_status
toolboxv2/tests/test_isaa/test_mcp/test_server.py::TestToolExecution::test_call_tool_unknown
```
