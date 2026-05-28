"""
Skills REST routes (per-agent).

Skills live on agent.session_manager.skills_manager. We touch them via that
path. The default session is created on-demand to ensure the manager exists.
"""
from __future__ import annotations

import io
import json
import zipfile
import urllib.request
import urllib.error


async def _get_skills_manager(isaa, agent_name: str):
    agent = await isaa.get_agent(agent_name)
    # Ensure a session exists so skills_manager is instantiated.
    await agent.session_manager.get_or_create("default")
    sm = getattr(agent.session_manager, "skills_manager", None)
    return agent, sm


def register(app, ctx):
    isaa = ctx["isaa"]

    @app.get("/api/agents/{name}/skills")
    async def list_skills(name: str):
        agent, sm = await _get_skills_manager(isaa, name)
        if sm is None:
            return []
        return [s.to_dict() for s in sm.skills.values()]

    @app.post("/api/agents/{name}/skills")
    async def add_skill(name: str, request):
        body = request.json_data or {}
        agent, sm = await _get_skills_manager(isaa, name)
        if sm is None:
            return (400, {"error": "skills_manager unavailable"})
        # Skill.from_dict expects a Skill-shaped dict
        try:
            from toolboxv2.mods.isaa.base.Agent.skills import Skill  # noqa
        except ImportError:
            return (500, {"error": "Skill class import failed"})
        try:
            skill = Skill.from_dict(body) if hasattr(Skill, "from_dict") else Skill(**body)
        except Exception as e:
            return (400, {"error": f"invalid skill: {e}"})
        if not getattr(skill, "id", None):
            return (400, {"error": "skill.id required"})
        skill.source = "custom"
        sm.skills[skill.id] = skill
        sm._skill_embeddings_dirty = True
        return {"ok": True, "skill_id": skill.id}

    @app.put("/api/agents/{name}/skills/{skill_id}")
    async def update_skill(name: str, skill_id: str, request):
        body = request.json_data or {}
        agent, sm = await _get_skills_manager(isaa, name)
        if sm is None or skill_id not in sm.skills:
            return (404, {"error": "skill not found"})
        skill = sm.skills[skill_id]
        for k, v in body.items():
            if hasattr(skill, k):
                setattr(skill, k, v)
        sm._skill_embeddings_dirty = True
        return {"ok": True}

    @app.delete("/api/agents/{name}/skills/{skill_id}")
    async def delete_skill(name: str, skill_id: str):
        agent, sm = await _get_skills_manager(isaa, name)
        if sm is None:
            return (404, {"error": "skills_manager unavailable"})
        if skill_id not in sm.skills:
            return (404, {"error": "skill not found"})
        # Predefined skills cannot be deleted (they auto-reappear).
        if sm.skills[skill_id].source == "predefined":
            return (400, {"error": "cannot delete predefined skill"})
        sm.skills.pop(skill_id, None)
        sm._skill_embeddings_dirty = True
        return {"ok": True}

    @app.get("/api/skills/library")
    async def library(request):
        """Cross-agent shareable skills."""
        out = []
        names = isaa.config.get("agents-name-list", []) or []
        for n in names:
            try:
                _, sm = await _get_skills_manager(isaa, n)
            except Exception:
                continue
            if sm is None:
                continue
            for s in sm.list_shareable_skills():
                s["agent"] = n
                out.append(s)
        return out

    @app.get("/api/agents/{name}/skills/export")
    async def export_skills(name: str):
        """Export all of an agent's skills as a ZIP of <skill_id>.json files."""
        agent, sm = await _get_skills_manager(isaa, name)
        if sm is None:
            return (404, {"error": "skills_manager unavailable"})
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for sid, skill in sm.skills.items():
                try:
                    data = skill.to_dict()
                    zf.writestr(f"{sid}.json", json.dumps(data, indent=2, default=str))
                except Exception:
                    continue
            # manifest
            zf.writestr("_manifest.json", json.dumps({
                "agent": name,
                "skill_ids": list(sm.skills.keys()),
                "format": "isaa-skills-v1",
            }, indent=2))
        body = buf.getvalue()
        headers = {
            "Content-Type": "application/zip",
            "Content-Length": str(len(body)),
            "Content-Disposition": f'attachment; filename="{name}_skills.zip"',
        }
        return (200, headers, body)

    @app.post("/api/agents/{name}/skills/import")
    async def import_skills(name: str, request):
        """Import skills from an uploaded ZIP (multipart) or a GitHub URL (json body).

        JSON body forms:
          {"github_url": "https://github.com/owner/repo/tree/main/skills"}
          {"github_url": "https://raw.githubusercontent.com/.../skill.json"}
          {"zip_url": "https://.../skills.zip"}
        Multipart form: file=<zip>
        """
        agent, sm = await _get_skills_manager(isaa, name)
        if sm is None:
            return (400, {"error": "skills_manager unavailable"})

        overwrite = True
        imported = []
        errors = []

        # Case 1: multipart ZIP upload
        fd = request.form_data or {}
        file_field = fd.get("file")
        if file_field:
            data = _file_bytes(file_field)
            imported, errors = _import_zip_bytes(sm, data, overwrite)
            return {"ok": True, "imported": imported, "errors": errors}

        # Case 2: JSON body with a URL
        body = request.json_data or {}
        github_url = body.get("github_url")
        zip_url = body.get("zip_url")

        if zip_url:
            try:
                data = _fetch_bytes(zip_url)
                imported, errors = _import_zip_bytes(sm, data, overwrite)
            except Exception as e:
                return (502, {"error": f"zip fetch failed: {e}"})
            return {"ok": True, "imported": imported, "errors": errors}

        if github_url:
            try:
                imported, errors = _import_github(sm, github_url, overwrite)
            except Exception as e:
                return (502, {"error": f"github import failed: {e}"})
            return {"ok": True, "imported": imported, "errors": errors}

        return (400, {"error": "provide file, zip_url, or github_url"})


# ============================================================================
# Import helpers
# ============================================================================

def _file_bytes(field) -> bytes:
    if isinstance(field, dict):
        d = field.get("data") or field.get("content") or b""
        return d.encode("utf-8") if isinstance(d, str) else bytes(d)
    if isinstance(field, str):
        return field.encode("utf-8")
    if isinstance(field, (bytes, bytearray)):
        return bytes(field)
    return b""


def _fetch_bytes(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "isaa-ui"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _skill_from_dict(sm, data: dict) -> bool:
    try:
        from toolboxv2.mods.isaa.base.Agent.skills import Skill
    except ImportError:
        return False
    try:
        skill = Skill.from_dict(data) if hasattr(Skill, "from_dict") else Skill(**data)
    except Exception:
        return False
    if not getattr(skill, "id", None):
        return False
    skill.source = "imported"
    sm.skills[skill.id] = skill
    sm._skill_embeddings_dirty = True
    return True


def _import_zip_bytes(sm, data: bytes, overwrite: bool):
    imported, errors = [], []
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except Exception as e:
        return [], [f"invalid zip: {e}"]
    for info in zf.infolist():
        fn = info.filename
        if not fn.endswith(".json") or fn.startswith("_") or "/" in fn.lstrip("./") and fn.split("/")[-1].startswith("_"):
            if fn.endswith("_manifest.json") or fn == "_manifest.json":
                continue
        if not fn.endswith(".json"):
            continue
        if fn.split("/")[-1] == "_manifest.json":
            continue
        try:
            raw = zf.read(info).decode("utf-8")
            obj = json.loads(raw)
        except Exception as e:
            errors.append(f"{fn}: {e}")
            continue
        # File could be a single skill or a list
        objs = obj if isinstance(obj, list) else [obj]
        for o in objs:
            if _skill_from_dict(sm, o):
                imported.append(o.get("id", fn))
            else:
                errors.append(f"{fn}: invalid skill shape")
    return imported, errors


def _import_github(sm, url: str, overwrite: bool):
    """Handle github tree URLs (folder) or raw .json/.zip URLs."""
    imported, errors = [], []

    # Raw single file
    if url.endswith(".json") and "raw.githubusercontent.com" in url:
        raw = _fetch_bytes(url).decode("utf-8")
        obj = json.loads(raw)
        objs = obj if isinstance(obj, list) else [obj]
        for o in objs:
            if _skill_from_dict(sm, o):
                imported.append(o.get("id", url))
            else:
                errors.append(f"{url}: invalid skill shape")
        return imported, errors

    if url.endswith(".zip"):
        return _import_zip_bytes(sm, _fetch_bytes(url), overwrite)

    # Folder: https://github.com/{owner}/{repo}/tree/{branch}/{path}
    import re
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.+)", url)
    if not m:
        # maybe https://github.com/owner/repo (root)
        m2 = re.match(r"https?://github\.com/([^/]+)/([^/]+)/?$", url)
        if not m2:
            raise ValueError("unsupported github url; use a tree/folder, raw .json, or .zip url")
        owner, repo = m2.group(1), m2.group(2)
        branch, path = "main", ""
    else:
        owner, repo, branch, path = m.group(1), m.group(2), m.group(3), m.group(4)

    api = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    listing = json.loads(_fetch_bytes(api).decode("utf-8"))
    if isinstance(listing, dict):
        listing = [listing]
    for item in listing:
        if item.get("type") != "file" or not item.get("name", "").endswith(".json"):
            continue
        dl = item.get("download_url")
        if not dl:
            continue
        try:
            obj = json.loads(_fetch_bytes(dl).decode("utf-8"))
            objs = obj if isinstance(obj, list) else [obj]
            for o in objs:
                if _skill_from_dict(sm, o):
                    imported.append(o.get("id", item["name"]))
                else:
                    errors.append(f"{item['name']}: invalid skill shape")
        except Exception as e:
            errors.append(f"{item.get('name')}: {e}")
    return imported, errors
