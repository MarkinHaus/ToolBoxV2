"""
Tools REST routes (per-agent).

Operates on agent.tool_manager (ToolManager). Supports:
  - list all tools grouped by category, with enabled state
  - register CLI tools (register_cli_tool)
  - register MCP tools (via MCPSessionManager, best-effort)
  - unregister a tool
  - toggle a tool on/off (flags['disabled'])
  - toggle all on/off

Note: 'disabled' is a flag the UI sets. For it to actually exclude the tool
from the LLM, the ExecutionEngine's tool selection must filter
flags={'disabled': False}. The flag is persisted on the ToolEntry.
"""
from __future__ import annotations


async def _get_tool_manager(isaa, agent_name: str):
    agent = await isaa.get_agent(agent_name)
    tm = getattr(agent, "tool_manager", None)
    return agent, tm


def _entry_dict(entry) -> dict:
    return {
        "name": entry.name,
        "description": (entry.description or "")[:300],
        "category": entry.category,
        "source": entry.source,
        "flags": entry.flags,
        "enabled": not entry.flags.get("disabled", False),
        "health_status": getattr(entry, "health_status", "UNKNOWN"),
        "call_count": getattr(entry, "call_count", 0),
        "args_schema": getattr(entry, "args_schema", "()"),
        "server_name": getattr(entry, "server_name", None),
    }


def register(app, ctx):
    isaa = ctx["isaa"]

    @app.get("/api/agents/{name}/tools")
    async def list_tools(name: str):
        agent, tm = await _get_tool_manager(isaa, name)
        if tm is None:
            return {"categories": {}, "total": 0}
        # Group by primary (first) category
        groups: dict[str, list] = {}
        for entry in tm.get_all():
            cats = entry.category or ["uncategorized"]
            primary = cats[0]
            groups.setdefault(primary, []).append(_entry_dict(entry))
        return {
            "categories": groups,
            "total": tm.count(),
            "category_names": tm.list_categories(),
        }

    @app.post("/api/agents/{name}/tools/cli")
    async def add_cli_tool(name: str, request):
        body = request.json_data or {}
        agent, tm = await _get_tool_manager(isaa, name)
        if tm is None:
            return (400, {"error": "tool_manager unavailable"})
        tool_name = body.get("name")
        executable = body.get("executable")
        cli_tool_executable = body.get("cli_tool_executable", "")
        if not tool_name or not executable:
            return (400, {"error": "name and executable required"})
        try:
            entry = tm.register_cli_tool(
                name=tool_name,
                executable=executable,
                cli_tool_executable=cli_tool_executable,
                executable_args=body.get("executable_args") or [],
                post_script_args=body.get("post_script_args") or [],
                category=body.get("category"),
            )
        except FileNotFoundError as e:
            return (400, {"error": str(e)})
        except Exception as e:
            return (500, {"error": f"register failed: {e}"})
        return {"ok": True, "name": entry.name}

    @app.post("/api/agents/{name}/tools/mcp")
    async def add_mcp_tools(name: str, request):
        body = request.json_data or {}
        agent, tm = await _get_tool_manager(isaa, name)
        if tm is None:
            return (400, {"error": "tool_manager unavailable"})
        server_name = body.get("server_name")
        server_config = body.get("config")  # {command, args} or {url}
        if not server_name or not server_config:
            return (400, {"error": "server_name and config required"})
        try:
            from toolboxv2.mods.isaa.extras.mcp_session_manager import MCPSessionManager
        except ImportError:
            return (501, {"error": "MCP not available in this build"})
        try:
            mgr = MCPSessionManager()
            session = await mgr.get_session(server_name, server_config)
            if not session:
                return (502, {"error": f"could not connect to MCP server {server_name}"})
            caps = await mgr.extract_capabilities(session, server_name)
            tools = caps.get("tools", {})
            count = 0
            for tool_name, tool_info in tools.items():
                wrapper_name = f"{server_name}_{tool_name}"
                tm.register(
                    func=None,
                    name=wrapper_name,
                    description=tool_info.get("description", f"MCP tool: {tool_name}"),
                    category=[f"mcp_{server_name}", "mcp", server_name],
                    source="mcp",
                    server_name=server_name,
                    metadata={"input_schema": tool_info.get("input_schema", {})},
                )
                count += 1
            return {"ok": True, "registered": count, "server": server_name}
        except Exception as e:
            return (500, {"error": f"mcp register failed: {e}"})

    @app.delete("/api/agents/{name}/tools/{tool_name}")
    async def remove_tool(name: str, tool_name: str):
        agent, tm = await _get_tool_manager(isaa, name)
        if tm is None:
            return (400, {"error": "tool_manager unavailable"})
        if not tm.exists(tool_name):
            return (404, {"error": "tool not found"})
        ok = tm.unregister(tool_name)
        return {"ok": ok}

    @app.put("/api/agents/{name}/tools/{tool_name}/toggle")
    async def toggle_tool(name: str, tool_name: str, request):
        body = request.json_data or {}
        agent, tm = await _get_tool_manager(isaa, name)
        if tm is None:
            return (400, {"error": "tool_manager unavailable"})
        entry = tm.get(tool_name)
        if entry is None:
            return (404, {"error": "tool not found"})
        # explicit enabled in body, else flip
        if "enabled" in body:
            enabled = bool(body["enabled"])
        else:
            enabled = entry.flags.get("disabled", False)  # flip: was disabled → enable
        new_flags = dict(entry.flags)
        new_flags["disabled"] = not enabled
        tm.update(tool_name, flags=new_flags)
        return {"ok": True, "enabled": enabled}

    @app.post("/api/agents/{name}/tools/toggle_all")
    async def toggle_all(name: str, request):
        body = request.json_data or {}
        enabled = bool(body.get("enabled", True))
        agent, tm = await _get_tool_manager(isaa, name)
        if tm is None:
            return (400, {"error": "tool_manager unavailable"})
        changed = 0
        for entry in tm.get_all():
            new_flags = dict(entry.flags)
            new_flags["disabled"] = not enabled
            tm.update(entry.name, flags=new_flags)
            changed += 1
        return {"ok": True, "changed": changed, "enabled": enabled}
