# file: toolboxv2/mods/Kurse/app.py
"""Kurse — FastTB web app.

Coach routes (/api/*) ride the native TB session (self-host loopback-trust /
anonymous-local-root => coach is authenticated). Participant routes (/l/*,
/api/join, /api/session, /api/progress) are public — learners only type a name.
Non-coach hitting / gets a tiny loading page instead of the dashboard.
"""

from pathlib import Path

from toolboxv2 import get_app
from toolboxv2.utils.workers.fast_tb import FastTB
from toolboxv2.utils.workers.session import SessionData

from . import store as S

STATIC = Path(__file__).resolve().parent / "static"
app = FastTB(title="Kurse")
app.mount_static("/static", str(STATIC))


def _page(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _coach(session: SessionData) -> bool:
    return bool(session and (session.is_authenticated or session.level == -1))


DENY = (403, {"error": "coach login required"})


# ============================================================================
# pages
# ============================================================================

@app.get("/")
async def index(session: SessionData):
    # coach sees the dashboard; everyone else gets a mini loading page.
    return _page("coach.html") if _coach(session) else _page("loading.html")

@app.get("/login", auth=True)
async def login():
    return  _page("loading.html")

@app.get("/l/{coid}")
async def learn_page(coid: str):
    return _page("learn.html")


@app.get("/api/skill")
async def creation_skill():
    # the course-content creation skill, copy-pasteable on the coach page
    return (200, {"Content-Type": "text/markdown; charset=utf-8"},
            (STATIC / "skill.md").read_bytes())


# ============================================================================
# coach API — courses
# ============================================================================

@app.get("/api/courses")
async def courses(session: SessionData):
    if not _coach(session):
        return DENY
    return {"courses": await S.course_list(get_app())}


@app.post("/api/courses")
async def course_create(session: SessionData, name: str = ""):
    if not _coach(session):
        return DENY
    if not name.strip():
        return (400, {"error": "name required"})
    return (201, await S.course_create(get_app(), name.strip()))


@app.delete("/api/courses/{cid}")
async def course_delete(session: SessionData, cid: str):
    if not _coach(session):
        return DENY
    await S.course_delete(get_app(), cid)
    return {"deleted": cid}


# ---- files (flat HTML sheets) ---------------------------------------------

@app.get("/api/courses/{cid}/files")
async def files(session: SessionData, cid: str):
    if not _coach(session):
        return DENY
    fs = await S.file_list(get_app(), cid)
    for f in fs:                            # keep list payload small
        f["has_html"] = bool(f.get("html"))
        f["html"] = ""
    return {"files": fs}


@app.post("/api/files")
async def file_create(session: SessionData, cid: str = "", name: str = "",
                      html: str = ""):
    if not _coach(session):
        return DENY
    if not cid or not name.strip():
        return (400, {"error": "cid and name required"})
    return (201, await S.file_create(get_app(), cid, name.strip(), html))


@app.get("/api/files/{cid}/{fid}")
async def file_get(session: SessionData, cid: str, fid: str):
    if not _coach(session):
        return DENY
    f = await S.file_get(get_app(), cid, fid)
    return f or (404, {"error": "not found"})


@app.put("/api/files/{fid}")
async def file_update(session: SessionData, fid: str, cid: str = "",
                      name: str = None, order: float = None, html: str = None):
    if not _coach(session):
        return DENY
    f = await S.file_update(get_app(), cid, fid, name=name, order=order, html=html)
    return f or (404, {"error": "not found"})


@app.post("/api/files/{fid}/move")
async def file_move(session: SessionData, fid: str, cid: str = "", dir: int = 0):
    if not _coach(session):
        return DENY
    return {"files": await S.file_reorder(get_app(), cid, fid, int(dir))}


@app.delete("/api/files/{fid}")
async def file_delete(session: SessionData, fid: str, cid: str = ""):
    if not _coach(session):
        return DENY
    await S.file_delete(get_app(), cid, fid)
    return {"deleted": fid}


@app.get("/api/files/{cid}/{fid}/validate")
async def file_validate(session: SessionData, cid: str, fid: str):
    if not _coach(session):
        return DENY
    f = await S.file_get(get_app(), cid, fid)
    return S.validate_file(f) if f else (404, {"error": "not found"})


# ---- cohorts (share links) -------------------------------------------------

@app.get("/api/cohorts")
async def cohorts(session: SessionData, cid: str = ""):
    if not _coach(session):
        return DENY
    return {"cohorts": await S.cohort_list(get_app(), cid)}


@app.post("/api/cohorts")
async def cohort_create(session: SessionData, request):
    if not _coach(session):
        return DENY
    d = request.json_data or {}
    sids = d.get("session_ids") or []
    if not d.get("cid") or not sids:
        return (400, {"error": "cid and session_ids required"})
    co = await S.cohort_create(get_app(), d["cid"], d.get("name", "Kohorte"),
                               sids, d.get("anchor", 0))
    return (201, co)


@app.put("/api/cohorts/{coid}")
async def cohort_update(session: SessionData, coid: str, request):
    if not _coach(session):
        return DENY
    d = request.json_data or {}
    co = await S.cohort_update(get_app(), coid, name=d.get("name"),
                               session_ids=d.get("session_ids"),
                               anchor=d.get("anchor"))
    return co or (404, {"error": "not found"})


@app.delete("/api/cohorts/{coid}")
async def cohort_delete(session: SessionData, coid: str):
    if not _coach(session):
        return DENY
    try:
        await S.cohort_delete(get_app(), coid)
    except Exception as e:
        return (500, {"error": str(e)})
    return {"deleted": coid}


@app.get("/api/cohorts/{coid}/live")
async def cohort_live(session: SessionData, coid: str):
    # ponytail: polling, not WS. add a WS push if the roster gets big.
    if not _coach(session):
        return DENY
    a = get_app()
    co = await S.cohort_get(a, coid)
    stats = await S.cohort_live(a, coid)
    html_cache = {}                             # sheet html for dot counts
    for s in stats:
        sid = co["session_ids"][s["pos"]] if co and 0 <= s["pos"] < len(co["session_ids"]) else None
        if sid not in html_cache:
            f = await S.file_get(a, co["cid"], sid) if sid else None
            html_cache[sid] = f.get("html", "") if f else ""
        html = html_cache[sid]
        s["sheet_tasks"] = html.count('id="task-')
        # hints available for the learner's CURRENT task → fixed dot count
        s["cur_hints_total"] = html.count(f'id="hint-{s["task"]}-')
    return {"participants": stats}


# ============================================================================
# participant API — public
# ============================================================================

@app.post("/api/join")
async def join(request):
    d = request.json_data or {}
    coid, name = d.get("coid", ""), (d.get("name") or "").strip()
    if not coid or not name:
        return (400, {"error": "coid and name required"})
    a = get_app()
    p = await S.pp_join(a, coid, name)
    if not p:
        return (404, {"error": "unknown cohort"})
    co = await S.cohort_get(a, coid)
    sessions = []
    for sid in co["session_ids"]:
        f = await S.file_get(a, co["cid"], sid)
        sessions.append({"nid": sid, "name": f["name"] if f else "?"})
    return {"cid": co["cid"], "sessions": sessions, "anchor": co["anchor"],
            "resume": {"pos": p["pos"], "task": p["task"]}}


@app.get("/api/session/{coid}/{sidx}")
async def get_session(coid: str, sidx: int, name: str = ""):
    a = get_app()
    co = await S.cohort_get(a, coid)
    if not co or sidx < 0 or sidx >= len(co["session_ids"]):
        return (404, {"error": "out of range"})
    fid = co["session_ids"][sidx]
    f = await S.file_get(a, co["cid"], fid)
    if not f:
        return (404, {"error": "missing session"})
    if name:                                    # move sheet pointer, keep task
        await S.pp_touch_pos(a, coid, name, sidx)
    return {"name": f["name"], "html": f.get("html", ""),
            "idx": sidx, "count": len(co["session_ids"])}


@app.post("/api/progress")
async def progress(request):
    d = request.json_data or {}
    if not d.get("coid") or not d.get("name"):
        return (400, {"error": "coid and name required"})
    p = await S.pp_event(get_app(), d["coid"], d["name"], d.get("event", {}))
    return {"ok": bool(p)}
