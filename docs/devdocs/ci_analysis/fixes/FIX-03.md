# FIX-03 – aus Karte K-ENV-02

Paket: **P0-SIGNAL** (CI-Signal und Installation) · Fix-Typ: **A** · Karte: GETESTET · betrifft 45 Test-IDs

```
Fix:           FIX-03  (aus Karte K-ENV-02)
Voraussetzung: keine
Dateien:       pyproject.toml (nur die mcp-Zeile im Extra `isaa`), toolboxv2/mods/isaa/pyproject.toml (nur die mcp-Zeile), toolboxv2/mcp_server/server.py, toolboxv2/mods/isaa/extras/mcp_session_manager.py, toolboxv2/mods/isaa/base/Agent/flow_agent.py (nur Zeilen 72, 429, 4518, 4529)   (nur diese ändern)
Vorher lesen:  `server.py:78-120, 740-820`; `mcp_session_manager.py:255-420`; `flow_agent.py:66-76, 4510-4535`
```

**Schritte:**

1. Patch aus diesem Auftrag anwenden (s. „Patch“). Gelingt `git apply --check` nicht, die Änderungen von Hand nach dem Patch machen – er ist die Vorlage.
2. venv neu installieren, damit mcp 2 drin ist: `uv pip install -e toolboxv2/mods/isaa` und `uv pip install -e .`; dann `python -c "import importlib.metadata as m; print(m.version('mcp'))"` → 2.3 oder höher.
3. Das Prüfskript aus „Zusätzliche Abnahme“ ausführen (prüft Protokoll, ISAA-Client und FlowAgent in einem Lauf).

**Nicht tun:** Keine Tests ändern. `test_isaa/test_mcp/test_models.py` (Port 8791) nicht anfassen – offene Frage. Kein Kompatibilitäts-Zweig für mcp 1.x. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- `python -m pytest toolboxv2/tests/test_isaa/test_mcp toolboxv2/tests/test_mini/test_system/feature_tests_progressive.py -q` → höchstens 2 rot (`test_models.py::TestMCPConfig::test_default_values`, `feature_tests_progressive.py::TestNoSingletonBleeding::test_disk_not_mutated`), und das Prüfskript gibt die Zeilen 1–7 mit den Werten aus „Nachher“ aus (`2 tools: 17`, `6 isaa client tools: 17`, `7 flowagent: MCPServer ['agent_run'] R:hi`).
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Patch (getestet, `git apply`)

Als Datei außerhalb des Repos speichern, im Repo-Wurzelordner `git apply --check <datei>` und dann `git apply <datei>`.

```diff
diff --git a/pyproject.toml b/pyproject.toml
index f42f12e..cc6d1f8 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -186,7 +186,7 @@ isaa = [
     "pypdf2>=3.0.1",
     "nest-asyncio>=1.5.0",
     "pyvis>=0.3.0",
-    "mcp>=1.6.0",
+    "mcp>=2.3,<3",
     "tenacity>=9.1.2",
     "playwright",
     "html2text>=2025.4.15",
diff --git a/toolboxv2/mcp_server/server.py b/toolboxv2/mcp_server/server.py
index 3179e01..9805891 100644
--- a/toolboxv2/mcp_server/server.py
+++ b/toolboxv2/mcp_server/server.py
@@ -14,7 +14,6 @@ from typing import Any, Dict, List, Optional, Union
 
 # MCP SDK imports
 from mcp.server import Server, NotificationOptions
-from mcp.server.models import InitializationOptions
 import mcp.server.stdio
 import mcp.types as types
 from pydantic import AnyUrl
@@ -81,7 +80,15 @@ class ToolBoxV2MCPServer:
         self.config = config or MCPConfig()
 
         # MCP Server instance
-        self.server = Server(self.config.server_name)
+        # mcp>=2: handlers are passed to the constructor (no decorator API)
+        self.server = Server(
+            self.config.server_name,
+            version=self.config.server_version,
+            on_list_tools=self._on_list_tools,
+            on_list_resources=self._on_list_resources,
+            on_read_resource=self._on_read_resource,
+            on_call_tool=self._on_call_tool,
+        )
 
         # State
         self._app: Optional[Any] = None  # ToolBoxV2 App (lazy loaded)
@@ -742,52 +749,36 @@ class ToolBoxV2MCPServer:
     # =========================================================================
 
     def _setup_handlers(self):
-        """Setup MCP protocol handlers."""
-
-        @self.server.list_tools()
-        async def handle_list_tools() -> List[types.Tool]:
-            """Return static tool definitions immediately (no app loading)."""
-            tools = self.get_tool_definitions()
-            return [
-                types.Tool(
-                    name=t["name"],
-                    description=t["description"],
-                    inputSchema=t["inputSchema"]
-                )
-                for t in tools
-            ]
-
-        @self.server.list_resources()
-        async def handle_list_resources() -> List[types.Resource]:
-            """Return resource definitions."""
-            resources = self.get_resource_definitions()
-            return [
-                types.Resource(
-                    uri=r["uri"],
-                    name=r["name"],
-                    description=r["description"],
-                    mimeType=r["mimeType"]
-                )
-                for r in resources
-            ]
-
-        @self.server.read_resource()
-        async def handle_read_resource(uri: AnyUrl) -> str:
-            """Read a resource."""
-            return await self.read_resource(str(uri))
-
-        @self.server.call_tool()
-        async def handle_call_tool(
-            name: str,
-            arguments: Dict[str, Any]
-        ) -> List[types.TextContent]:
-            """Execute a tool."""
-            results = await self.call_tool(name, arguments)
-
-            return [
-                types.TextContent(type="text", text=r.content)
-                for r in results
-            ]
+        """Handlers are registered in __init__ via Server(on_...=...) (mcp>=2)."""
+
+    async def _on_list_tools(self, ctx, params) -> types.ListToolsResult:
+        """Return static tool definitions immediately (no app loading)."""
+        return types.ListToolsResult(tools=[
+            types.Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
+            for t in self.get_tool_definitions()
+        ])
+
+    async def _on_list_resources(self, ctx, params) -> types.ListResourcesResult:
+        """Return resource definitions."""
+        return types.ListResourcesResult(resources=[
+            types.Resource(uri=r["uri"], name=r["name"], description=r["description"], mimeType=r["mimeType"])
+            for r in self.get_resource_definitions()
+        ])
+
+    async def _on_read_resource(self, ctx, params) -> types.ReadResourceResult:
+        """Read a resource."""
+        uri = str(params.uri)
+        text = await self.read_resource(uri)
+        mime = next((r["mimeType"] for r in self.get_resource_definitions() if r["uri"] == uri), "text/plain")
+        return types.ReadResourceResult(contents=[types.TextResourceContents(uri=uri, mimeType=mime, text=text)])
+
+    async def _on_call_tool(self, ctx, params) -> types.CallToolResult:
+        """Execute a tool."""
+        results = await self.call_tool(params.name, params.arguments or {})
+        return types.CallToolResult(
+            content=[types.TextContent(type="text", text=r.content) for r in results],
+            isError=any(not r.success for r in results),
+        )
 
     # =========================================================================
     # SERVER LIFECYCLE
@@ -805,13 +796,8 @@ class ToolBoxV2MCPServer:
                 await self.server.run(
                     read_stream,
                     write_stream,
-                    InitializationOptions(
-                        server_name=self.config.server_name,
-                        server_version=self.config.server_version,
-                        capabilities=self.server.get_capabilities(
-                            notification_options=NotificationOptions(),
-                            experimental_capabilities={}
-                        )
+                    self.server.create_initialization_options(
+                        notification_options=NotificationOptions(),
                     )
                 )
         finally:
diff --git a/toolboxv2/mods/isaa/base/Agent/flow_agent.py b/toolboxv2/mods/isaa/base/Agent/flow_agent.py
index 1fe7e15..c9500ba 100644
--- a/toolboxv2/mods/isaa/base/Agent/flow_agent.py
+++ b/toolboxv2/mods/isaa/base/Agent/flow_agent.py
@@ -69,7 +69,7 @@ except ImportError:
         pass
 
 try:
-    from mcp.server.fastmcp import FastMCP
+    from mcp.server.mcpserver import MCPServer  # mcp>=2: FastMCP was renamed
     MCP_AVAILABLE = True
 except ImportError:
     MCP_AVAILABLE = False
@@ -426,7 +426,7 @@ class FlowAgent:
 
         # Servers
         self.a2a_server: A2AServer | None = None
-        self.mcp_server: FastMCP | None = None
+        self.mcp_server: "MCPServer | None" = None
 
         # Execution engine instance (lazy loaded)
         self._execution_engine_cache = None
@@ -4515,7 +4515,7 @@ class FlowAgent:
 
     def setup_mcp_server(self, name: str | None = None):
         try:
-            from mcp.server.fastmcp import FastMCP
+            from mcp.server.mcpserver import MCPServer  # mcp>=2: FastMCP was renamed
 
             MCP_AVAILABLE = True
         except ImportError:
@@ -4526,7 +4526,7 @@ class FlowAgent:
             return
 
         server_name = name or f"{self.amd.name}_MCP"
-        self.mcp_server = FastMCP(server_name)
+        self.mcp_server = MCPServer(server_name)
 
         @self.mcp_server.tool()
         async def agent_run(query: str, session_id: str = "mcp_session") -> str:
diff --git a/toolboxv2/mods/isaa/extras/mcp_session_manager.py b/toolboxv2/mods/isaa/extras/mcp_session_manager.py
index d52c501..9cd234a 100644
--- a/toolboxv2/mods/isaa/extras/mcp_session_manager.py
+++ b/toolboxv2/mods/isaa/extras/mcp_session_manager.py
@@ -302,7 +302,7 @@ class MCPSessionManager:
     async def _create_http_session(self, server_name: str, server_config: dict[str, Any]) -> ClientSession | None:
         """Create HTTP MCP session with timeout"""
         try:
-            import httpx
+            import httpx2 as httpx  # mcp>=2 transports take an httpx2.AsyncClient
             from mcp.client.streamable_http import streamable_http_client
 
             url = server_config.get("url", f"http://localhost:{server_config.get('port', 8000)}/mcp")
@@ -364,7 +364,7 @@ class MCPSessionManager:
                     capabilities['tools'][tool.name] = {
                         'name': tool.name,
                         'description': tool.description or '',
-                        'input_schema': tool.inputSchema,
+                        'input_schema': tool.input_schema,
                         'output_schema': getattr(tool, 'outputSchema', None),
                         'display_name': getattr(tool, 'title', tool.name)
                     }
@@ -392,9 +392,9 @@ class MCPSessionManager:
             try:
                 templates_response = await asyncio.wait_for(session.list_resource_templates(), timeout=3.0)
                 for template in templates_response.resourceTemplates:
-                    capabilities['resource_templates'][template.uriTemplate] = {
-                        'uri_template': template.uriTemplate,
-                        'name': template.name or template.uriTemplate,
+                    capabilities['resource_templates'][template.uri_template] = {
+                        'uri_template': template.uri_template,
+                        'name': template.name or template.uri_template,
                         'description': template.description or ''
                     }
             except TimeoutError:
diff --git a/toolboxv2/mods/isaa/pyproject.toml b/toolboxv2/mods/isaa/pyproject.toml
index b1be63a..d086da2 100644
--- a/toolboxv2/mods/isaa/pyproject.toml
+++ b/toolboxv2/mods/isaa/pyproject.toml
@@ -25,7 +25,7 @@ dependencies = [
     "pyvis>=0.3.0",
     "redis>=5.0.0",
     # "python-a2a[all]>=0.5.1",
-    "mcp>=1.6.0",
+    "mcp>=2.3,<3",
     "google-cloud-aiplatform>=1.38.0",
     "pyyaml>=6.0",
     "docker>=7.0.0",
```

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
