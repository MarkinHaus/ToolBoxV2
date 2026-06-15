# file: toolboxv2/mods/llama_lab/webapp/app.py
"""FastTB application — port of the llm-gateway server onto FastTB + llama_lab.

Auth split:
  * /v1/*           OpenAI-compatible, Bearer API-key (external clients)
  * /user|/admin|/playground  TB session auth (auth=True) — rides the native login
  * public          /, /api/*, /health, /api/signup
Inference is proxied to the served llama-server (OpenAI-compatible) selected by
the models bridge; no Ollama anywhere.
"""

import json
import time
from pathlib import Path

import httpx

from toolboxv2.utils.workers.fast_tb import FastTB
from toolboxv2.utils.workers.server_worker import ParsedRequest
from toolboxv2.utils.workers.session import SessionData

from . import db
from .live import LiveHandler, make_ws_class
from .models_bridge import Models

STATIC = Path(__file__).resolve().parent / "static"

app = FastTB(title="llama_lab")
app.mount_static("/static", str(STATIC))

_models: Models | None = None
_live: LiveHandler | None = None
_inited = False


def models() -> Models:
    global _models
    if _models is None:
        _models = Models()
    return _models


def live() -> LiveHandler:
    global _live
    if _live is None:
        _live = LiveHandler(models())
    return _live


async def _ensure_init():
    global _inited
    if not _inited:
        await db.init_db()
        _inited = True


def _page(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


# --- auth helpers -----------------------------------------------------------

async def _key(request: ParsedRequest):
    await _ensure_init()
    return await db.verify_api_key(request.headers.get("authorization"))


async def _web_user(session: SessionData):
    await _ensure_init()
    if not (session and session.is_authenticated):
        return None
    return  await db.user_by_email(session.user_name, create=True)


async def _admin(session: SessionData):
    return session.level == -1
    # u = await _web_user(session)
    # return u if u and u["tier"] == "admin" else None


def _err(status, msg):
    return (status, {"error": {"message": msg, "type": "error"}})


async def _proxy(base_url: str, path: str, payload: dict, timeout=300.0):
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(f"{base_url}{path}", json=payload)
        r.raise_for_status()
        return r.json()


# === OpenAI-compatible v1 (API key) =========================================

@app.get("/v1/models")
async def v1_models(request: ParsedRequest):
    if not await _key(request):
        return _err(401, "Invalid API key")
    return {"object": "list", "data": [
        {"id": m["name"], "object": "model", "created": int(time.time()),
         "owned_by": "local"} for m in models().active_models()]}


def _needs(messages):
    audio = vision = False
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for p in c:
                t = p.get("type", "") if isinstance(p, dict) else ""
                if t in ("image", "image_url"):
                    url = (p.get("image_url") or {}).get("url", "") if isinstance(p.get("image_url"), dict) else ""
                    audio = audio or "data:audio/" in url
                    vision = vision or "data:audio/" not in url
                if t in ("audio", "input_audio", "audio_url"):
                    audio = True
        elif isinstance(c, str):
            vision = vision or "data:image/" in c
            audio = audio or "data:audio/" in c
    return audio, vision


@app.post("/v1/chat/completions")
async def v1_chat(request: ParsedRequest):
    auth = await _key(request)
    if not auth:
        return _err(401, "Invalid API key")
    body = request.json_data or {}
    msgs = body.get("messages", [])
    needs_audio, needs_vision = _needs(msgs)
    m = models().find_for_request(body.get("model"), needs_vision, needs_audio)
    if not m:
        return _err(404, "No suitable model running. Load one in the admin panel.")
    payload = dict(body)
    payload["model"] = m["name"]
    start = time.time()

    if body.get("stream"):
        async def gen():
            tout = 0
            async with httpx.AsyncClient(timeout=300.0) as c:
                async with c.stream("POST", f"{m['base_url']}/chat/completions",
                                    json=payload) as r:
                    async for line in r.aiter_lines():
                        if line.startswith("data: "):
                            tout += 1
                            yield f"{line}\n\n"
            await db.log_usage(auth["key_id"], m["name"], 0, tout,
                               int((time.time() - start) * 1000))
            yield "data: [DONE]\n\n"
        return (200, {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}, gen())

    try:
        data = await _proxy(m["base_url"], "/chat/completions", payload)
        u = data.get("usage", {})
        await db.log_usage(auth["key_id"], m["name"], u.get("prompt_tokens", 0),
                           u.get("completion_tokens", 0), int((time.time() - start) * 1000))
        return data
    except httpx.HTTPStatusError as e:
        return _err(e.response.status_code, str(e))
    except Exception as e:
        return _err(500, str(e))


@app.post("/v1/embeddings")
async def v1_embeddings(request: ParsedRequest):
    auth = await _key(request)
    if not auth:
        return _err(401, "Invalid API key")
    m = models().find_embedding()
    if not m:
        return _err(404, "No embedding model running.")
    body = dict(request.json_data or {})
    body["model"] = m["name"]
    try:
        return await _proxy(m["base_url"], "/embeddings", body, timeout=120.0)
    except Exception as e:
        return _err(500, str(e))


@app.post("/v1/audio/transcriptions")
async def v1_transcribe(request: ParsedRequest):
    auth = await _key(request)
    if not auth:
        return _err(401, "Invalid API key")
    m = models().find_for_request(None, needs_audio=True)
    if not m:
        return _err(404, "No audio/omni model running.")
    try:
        return await _proxy(m["base_url"], "/audio/transcriptions", request.json_data or {})
    except Exception as e:
        return _err(500, str(e))


# --- live (omni) ---

@app.post("/v1/audio/live")
async def v1_live_create(request: ParsedRequest):
    auth = await _key(request)
    if not auth:
        return _err(401, "Invalid API key")
    body = request.json_data or {}
    res = live().create_session(body.get("model"), auth["key_id"])
    return res if "error" not in res else _err(400, res["error"])


@app.delete("/v1/audio/live/{session_token}")
async def v1_live_close(request: ParsedRequest, session_token: str):
    if not await _key(request):
        return _err(401, "Invalid API key")
    return {"closed": live().close(session_token)}


# === Admin (session) ========================================================

@app.get("/admin/", auth=True)
async def admin_page():
    return _page("admin.html")


@app.get("/admin/api/system", auth=True)
async def admin_system(session: SessionData):
    if not await _admin(session):
        return _err(403, "Admin required")
    return {"hardware": models().hardware(), "running": models().running()}


@app.get("/admin/api/models/installed", auth=True)
async def admin_installed(session: SessionData):
    if not await _admin(session):
        return _err(403, "Admin required")
    return {"models": models().installed()}


@app.get("/admin/api/models/running", auth=True)
async def admin_running(session: SessionData):
    if not await _admin(session):
        return _err(403, "Admin required")
    return {"running": models().running()}


@app.get("/admin/api/models/search-hf", auth=True)
async def admin_search_hf(session: SessionData, q: str = ""):
    if not await _admin(session):
        return _err(403, "Admin required")
    return {"results": models().search_hf(q)}


@app.post("/admin/api/models/download", auth=True)
async def admin_download(session: SessionData, repo_id: str = "", filename: str = ""):
    if not await _admin(session):
        return _err(403, "Admin required")
    return models().download(repo_id, filename)


@app.post("/admin/api/models/load", auth=True)
async def admin_load(session: SessionData, name: str = "", mode: str = "single",
                     port: int = 8080):
    if not await _admin(session):
        return _err(403, "Admin required")
    return models().load(name, mode, port)


@app.post("/admin/api/models/{name}/unload", auth=True)
async def admin_unload(session: SessionData, name: str):
    if not await _admin(session):
        return _err(403, "Admin required")
    return models().unload(name)


@app.post("/admin/api/models/remove", auth=True)
async def admin_remove(session: SessionData, path: str = ""):
    if not await _admin(session):
        return _err(403, "Admin required")
    return {"removed": models().remove(path)}


@app.get("/admin/api/users", auth=True)
async def admin_users(session: SessionData):
    if not await _admin(session):
        return _err(403, "Admin required")
    import aiosqlite
    async with aiosqlite.connect(db.DB_PATH) as d:
        d.row_factory = aiosqlite.Row
        cur = await d.execute("SELECT id,email,tier,balance,active FROM users ORDER BY id")
        return {"users": [dict(r) for r in await cur.fetchall()]}


# === User + Playground (session) ============================================

@app.get("/user/", auth=True)
async def user_page():
    return _page("user.html")


@app.get("/playground/", auth=True)
async def playground_page():
    # Single unified playground (LLM + media + live tab)
    return _page("playground.html")


@app.get("/user/api/me", auth=True)
async def user_me(session: SessionData):
    u = await _web_user(session)
    return {"email": u["email"], "tier": u["tier"], "balance": u["balance"]} if u else _err(401, "No session")


@app.get("/user/api/models", auth=True)
async def user_models(session: SessionData):
    if not await _web_user(session):
        return _err(401, "No session")
    return {"models": models().active_models()}


# --- file-tree chat sessions (new) ---

@app.get("/user/api/tree", auth=True)
async def chat_tree(session: SessionData):
    u = await _web_user(session)
    if not u:
        return _err(401, "No session")
    import aiosqlite
    async with aiosqlite.connect(db.DB_PATH) as d:
        d.row_factory = aiosqlite.Row
        f = await (await d.execute(
            "SELECT id,name,parent_id,sort FROM chat_folders WHERE user_id=? ORDER BY sort",
            (u["id"],))).fetchall()
        s = await (await d.execute(
            "SELECT id,title,folder_id,sort,updated_at FROM chat_sessions WHERE user_id=? ORDER BY sort",
            (u["id"],))).fetchall()
        return {"folders": [dict(x) for x in f], "sessions": [dict(x) for x in s]}


@app.post("/user/api/folders", auth=True)
async def chat_folder_create(session: SessionData, name: str = "new", parent_id: int = None):
    u = await _web_user(session)
    if not u:
        return _err(401, "No session")
    import aiosqlite
    async with aiosqlite.connect(db.DB_PATH) as d:
        cur = await d.execute(
            "INSERT INTO chat_folders (user_id,name,parent_id) VALUES (?,?,?)",
            (u["id"], name, parent_id))
        await d.commit()
        return {"id": cur.lastrowid, "name": name}


@app.post("/user/api/sessions", auth=True)
async def chat_session_save(request: ParsedRequest, session: SessionData):
    u = await _web_user(session)
    if not u:
        return _err(401, "No session")
    b = request.json_data or {}
    import aiosqlite
    async with aiosqlite.connect(db.DB_PATH) as d:
        if b.get("id"):
            await d.execute(
                "UPDATE chat_sessions SET title=?,messages=?,folder_id=?,sort=?,"
                "updated_at=CURRENT_TIMESTAMP WHERE id=? AND user_id=?",
                (b.get("title", "Chat"), json.dumps(b.get("messages", [])),
                 b.get("folder_id"), b.get("sort", 0), b["id"], u["id"]))
            sid = b["id"]
        else:
            cur = await d.execute(
                "INSERT INTO chat_sessions (user_id,folder_id,title,messages,sort) "
                "VALUES (?,?,?,?,?)", (u["id"], b.get("folder_id"),
                 b.get("title", "New chat"), json.dumps(b.get("messages", [])), b.get("sort", 0)))
            sid = cur.lastrowid
        await d.commit()
        return {"id": sid}


@app.get("/user/api/sessions/{sid}", auth=True)
async def chat_session_get(session: SessionData, sid: int):
    u = await _web_user(session)
    if not u:
        return _err(401, "No session")
    import aiosqlite
    async with aiosqlite.connect(db.DB_PATH) as d:
        d.row_factory = aiosqlite.Row
        r = await (await d.execute(
            "SELECT id,title,folder_id,messages FROM chat_sessions WHERE id=? AND user_id=?",
            (sid, u["id"]))).fetchone()
        if not r:
            return _err(404, "Not found")
        row = dict(r)
        row["messages"] = json.loads(row["messages"])
        return row


@app.delete("/user/api/sessions/{sid}", auth=True)
async def chat_session_delete(session: SessionData, sid: int):
    u = await _web_user(session)
    if not u:
        return _err(401, "No session")
    import aiosqlite
    async with aiosqlite.connect(db.DB_PATH) as d:
        await d.execute("DELETE FROM chat_sessions WHERE id=? AND user_id=?", (sid, u["id"]))
        await d.commit()
        return {"deleted": True}


# === Public =================================================================

@app.get("/")
async def landing():
    return _page("index.html")


@app.get("/docs/")
async def docs_page():
    return _page("docs.html")


@app.get("/health")
async def health():
    await _ensure_init()
    return {"status": "ok", "running": len(models().running())}


@app.get("/api/models")
async def public_models():
    return {"models": [{"id": m["name"], "type": m["modality"]} for m in models().running()]}


@app.post("/api/signup")
async def signup(request: ParsedRequest):
    await _ensure_init()
    b = request.json_data or {}
    email = (b.get("email") or "").strip()
    if not email or "@" not in email:
        return _err(400, "Valid email required")
    import aiosqlite
    async with aiosqlite.connect(db.DB_PATH) as d:
        await d.execute("INSERT INTO signup_requests (email,tier,message) VALUES (?,?,?)",
                        (email, b.get("tier", "payg"), b.get("message", "")))
        await d.commit()
    return {"status": "received"}


import secrets as _secrets


async def _aq(query, args=(), one=False, write=False):
    import aiosqlite
    async with aiosqlite.connect(db.DB_PATH) as d:
        d.row_factory = aiosqlite.Row
        cur = await d.execute(query, args)
        if write:
            await d.commit()
            return cur.lastrowid
        rows = await cur.fetchall()
        return (dict(rows[0]) if rows else None) if one else [dict(r) for r in rows]


# --- user: api keys + usage (session) ---

@app.get("/user/api/keys", auth=True)
async def user_keys(session: SessionData):
    u = await _web_user(session)
    if not u:
        return _err(401, "No session")
    return {"keys": await _aq("SELECT id,name,key_prefix,active,created_at FROM api_keys "
                              "WHERE user_id=? ORDER BY id", (u["id"],))}


@app.post("/user/api/keys", auth=True)
async def user_key_create(session: SessionData, name: str = "default"):
    u = await _web_user(session)
    if not u:
        return _err(401, "No session")
    key = "sk-" + _secrets.token_urlsafe(32)
    await _aq("INSERT INTO api_keys (user_id,key_hash,key_prefix,name) VALUES (?,?,?,?)",
              (u["id"], db.hash_key(key), key[:12], name), write=True)
    return {"key": key, "note": "shown once — store it now"}


@app.delete("/user/api/keys/{kid}", auth=True)
async def user_key_delete(session: SessionData, kid: int):
    u = await _web_user(session)
    if not u:
        return _err(401, "No session")
    await _aq("UPDATE api_keys SET active=0 WHERE id=? AND user_id=?", (kid, u["id"]), write=True)
    return {"revoked": True}


@app.get("/user/api/usage", auth=True)
async def user_usage(session: SessionData, days: int = 7):
    u = await _web_user(session)
    if not u:
        return _err(401, "No session")
    rows = await _aq(
        "SELECT us.model, SUM(us.tokens_in) tin, SUM(us.tokens_out) tout, "
        "SUM(us.cost) cost, COUNT(*) calls FROM usage us "
        "JOIN api_keys ak ON us.api_key_id=ak.id WHERE ak.user_id=? "
        "AND datetime(us.created_at) > datetime('now', ?) GROUP BY us.model",
        (u["id"], f"-{int(days)} days"))
    return {"usage": rows, "balance": u["balance"]}


@app.get("/user/api/ratelimit", auth=True)
async def user_ratelimit(session: SessionData):
    u = await _web_user(session)
    if not u:
        return _err(401, "No session")
    lim = db.load_config().get("rate_limits", {}).get(u["tier"], 5)
    return {"tier": u["tier"], "limit_per_min": lim}


# --- admin: users / apikeys / signups / config ---

@app.post("/admin/api/users", auth=True)
async def admin_user_create(session: SessionData, email: str = "", tier: str = "payg"):
    if not await _admin(session):
        return _err(403, "Admin required")
    if not email:
        return _err(400, "email required")
    uid = await _aq("INSERT INTO users (email,tier) VALUES (?,?)", (email, tier), write=True)
    return {"id": uid, "email": email, "tier": tier}


@app.patch("/admin/api/users/{uid}", auth=True)
async def admin_user_update(request: ParsedRequest, session: SessionData, uid: int):
    if not await _admin(session):
        return _err(403, "Admin required")
    b = request.json_data or {}
    sets, vals = [], []
    for f in ("tier", "active", "balance"):
        if f in b:
            sets.append(f"{f}=?")
            vals.append(b[f])
    if not sets:
        return _err(400, "nothing to update")
    vals.append(uid)
    await _aq(f"UPDATE users SET {','.join(sets)} WHERE id=?", tuple(vals), write=True)
    return {"updated": True}


@app.post("/admin/api/users/{uid}/apikey", auth=True)
async def admin_user_apikey(session: SessionData, uid: int, name: str = "default"):
    if not await _admin(session):
        return _err(403, "Admin required")
    key = "sk-" + _secrets.token_urlsafe(32)
    await _aq("INSERT INTO api_keys (user_id,key_hash,key_prefix,name) VALUES (?,?,?,?)",
              (uid, db.hash_key(key), key[:12], name), write=True)
    return {"key": key}


@app.delete("/admin/api/apikeys/{kid}", auth=True)
async def admin_apikey_delete(session: SessionData, kid: int):
    if not await _admin(session):
        return _err(403, "Admin required")
    await _aq("UPDATE api_keys SET active=0 WHERE id=?", (kid,), write=True)
    return {"revoked": True}


@app.get("/admin/api/signups", auth=True)
async def admin_signups(session: SessionData):
    if not await _admin(session):
        return _err(403, "Admin required")
    return {"signups": await _aq("SELECT id,email,tier,message,status,created_at "
                                 "FROM signup_requests ORDER BY id DESC")}


@app.post("/admin/api/signups/{sid}/approve", auth=True)
async def admin_signup_approve(session: SessionData, sid: int):
    if not await _admin(session):
        return _err(403, "Admin required")
    row = await _aq("SELECT * FROM signup_requests WHERE id=?", (sid,), one=True)
    if not row:
        return _err(404, "not found")
    u = await db.user_by_email(row["email"], create=True)
    key = "sk-" + _secrets.token_urlsafe(32)
    await _aq("INSERT INTO api_keys (user_id,key_hash,key_prefix,name) VALUES (?,?,?,?)",
              (u["id"], db.hash_key(key), key[:12], "default"), write=True)
    await _aq("UPDATE signup_requests SET status='approved' WHERE id=?", (sid,), write=True)
    return {"approved": row["email"], "key": key}


@app.get("/admin/api/config", auth=True)
async def admin_config_get(session: SessionData):
    if not await _admin(session):
        return _err(403, "Admin required")
    c = db.load_config()
    c.pop("admin_key", None)
    return c


@app.patch("/admin/api/config", auth=True)
async def admin_config_set(request: ParsedRequest, session: SessionData):
    if not await _admin(session):
        return _err(403, "Admin required")
    c = db.load_config()
    c.update(request.json_data or {})
    db.save_config(c)
    return {"updated": True}


# --- public uptime ---

@app.get("/api/uptime")
async def public_uptime(days: int = 30):
    rows = await _aq("SELECT status,active_models,latency_ms,created_at FROM uptime_checks "
                     "WHERE datetime(created_at) > datetime('now', ?) ORDER BY created_at",
                     (f"-{int(days)} days",))
    up = sum(1 for r in rows if r["status"] == "up")
    return {"checks": len(rows), "uptime_pct": round(up / len(rows) * 100, 2) if rows else 100.0,
            "recent": rows[-50:]}


# Register the live WebSocket bound to the shared LiveHandler singleton.
app.websocket("/v1/audio/live/ws/{session_token}")(make_ws_class(live(), app))
