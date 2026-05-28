"""
Agent REST routes.

Lists agents from isaa.config["agents-name-list"], serves config from
isaa.agent_data, updates via PUT (re-serialized into agent_data).
Updating config does NOT rebuild the live agent — that requires explicit
restart (we surface a 'rebuild_required' flag when the model changes).
"""
from __future__ import annotations

# Fields that can be hot-applied to an existing built agent.
HOT_FIELDS = {
    "system_message",
    "active_persona",
    "max_parallel_tasks",
    "temperature",
    "verbose_logging",
    "stream",
    "max_tokens_output",
    "max_tokens_input",
    "obs",
    "web_config",
    "context_config",
}

# Fields that force a rebuild.
REBUILD_FIELDS = {
    "fast_llm_model",
    "complex_llm_model",
    "name",
    "mcp",
    "a2a",
    "rate_limiter",
}


def register(app, ctx):
    isaa = ctx["isaa"]

    @app.get("/api/agents")
    async def list_agents():
        names = isaa.config.get("agents-name-list", []) or []
        out = []
        for n in names:
            cfg = isaa.agent_data.get(n, {})
            instance = isaa.config.get(f"agent-instance-{n}")
            out.append({
                "name": n,
                "model": cfg.get("fast_llm_model", ""),
                "complex_model": cfg.get("complex_llm_model", ""),
                "persona": cfg.get("active_persona", ""),
                "is_running": bool(getattr(instance, "is_running", False)) if instance else False,
                "built": instance is not None,
            })
        return out

    @app.get("/api/agents/{name}/config")
    async def get_config(name: str):
        cfg = isaa.agent_data.get(name)
        if cfg is None:
            return (404, {"error": "Agent not found"})
        return cfg

    @app.put("/api/agents/{name}/config")
    async def update_config(name: str, request):
        body = request.json_data or {}
        cfg = dict(isaa.agent_data.get(name) or {})
        if not cfg:
            return (404, {"error": "Agent not found"})

        changed_hot = []
        changed_rebuild = []
        for k, v in body.items():
            old = cfg.get(k)
            if old == v:
                continue
            cfg[k] = v
            if k in HOT_FIELDS:
                changed_hot.append(k)
            elif k in REBUILD_FIELDS:
                changed_rebuild.append(k)

        isaa.agent_data[name] = cfg

        # Hot-apply where possible.
        instance = isaa.config.get(f"agent-instance-{name}")
        if instance and changed_hot:
            for k in changed_hot:
                if k == "system_message":
                    instance.amd.system_message = cfg[k]
                elif k == "temperature":
                    instance.amd.temperature = cfg[k]
                elif k == "max_tokens_output":
                    instance.amd.max_tokens = cfg[k]
                elif k == "max_tokens_input":
                    instance.amd.max_input_tokens = cfg[k]

        # Drop instance for rebuild fields → next get_agent() rebuilds.
        if instance and changed_rebuild:
            isaa.config.pop(f"agent-instance-{name}", None)

        return {
            "ok": True,
            "applied_hot": changed_hot,
            "rebuild_required": changed_rebuild,
        }

    @app.post("/api/agents/{name}/duplicate")
    async def duplicate(name: str, request):
        body = request.json_data or {}
        new_name = body.get("new_name") or f"{name}_copy"
        if name not in isaa.agent_data:
            return (404, {"error": "Agent not found"})
        if new_name in isaa.agent_data:
            return (409, {"error": "Target name exists"})
        new_cfg = dict(isaa.agent_data[name])
        new_cfg["name"] = new_name
        isaa.agent_data[new_name] = new_cfg
        names = isaa.config.get("agents-name-list", []) or []
        if new_name not in names:
            names.append(new_name)
            isaa.config["agents-name-list"] = names
        return {"ok": True, "name": new_name}
