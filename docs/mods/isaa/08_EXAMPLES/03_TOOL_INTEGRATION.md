# 03_TOOL_INTEGRATION - Custom Tools

## Problem
Wie integriere ich eigene Tools in den ISAA-Workflow?

## Loesung
1. Dynamische Tools - load_tools()
2. Custom Functions - def my_tool()
3. Docker - docker_run()

## Tool-Discovery

    all_tools = list_tools(category="filesystem")
    suitable = [t for t in all_tools if t.has_capability("read")]
    load_tools([t["name"] for t in suitable[:3]])

## Custom Tool

    def db_query(sql: str, database: str = "default") -> dict:
        if not sql.strip().upper().startswith("SELECT"):
            return {"success": False, "error": "Nur SELECT erlaubt"}
        result = docker_run(f"psql -h db -d {database} -c \"{sql}\"")
        return {"success": True, "data": result["stdout"]}
