# file: toolboxv2/mods/Kurse/store.py
"""Kurse — persistence layer on the tb native database (TBEF.DB).

Key scheme (``::`` separator, uuid4-hex ids so exact-key reads never
prefix-collide with a different id of equal length):

    KURSE::course::{cid}                 course meta
    KURSE::node::{cid}::{nid}            folder | session (Windows-like tree via parent ptr)
    KURSE::cohort::{coid}                a share link (session range + entry anchor)
    KURSE::pp::{coid}::{pslug}           one participant's progress + event log

Everything is stored as JSON strings; DB.set rejects falsy values, and a
json.dumps(dict) is always a non-empty string, so that is safe.
"""

import json
import re
import time
import uuid

from toolboxv2 import TBEF


def _id() -> str:
    return uuid.uuid4().hex


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return s or "anon"


def now() -> float:
    return round(time.time(), 3)


# --- raw db -----------------------------------------------------------------

async def _get(app, key: str):
    r = await app.a_run_any(TBEF.DB.GET, query=key, get_results=True)
    if r.is_error() or not r.get():
        return None
    data = r.get()
    if isinstance(data, list):          # startswith may return >1 — take exact
        data = data[0] if len(data) == 1 else next(
            (x for x in data if _match(x, key)), data[0])
    if isinstance(data, (bytes, bytearray)):
        data = data.decode()
    obj = json.loads(data) if isinstance(data, str) else data
    if isinstance(obj, dict) and obj.get("_deleted"):
        return None                     # tombstoned
    return obj


def _match(raw, key):
    try:
        return json.loads(raw).get("_k") == key
    except Exception:
        return False


async def _list(app, prefix: str) -> list:
    r = await app.a_run_any(TBEF.DB.GET, query=prefix + "*", get_results=True)
    if r.is_error() or not r.get():
        return []
    out = []
    for raw in r.get():
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode()
        try:
            obj = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("_deleted"):
            continue                    # tombstoned
        out.append(obj)
    return out


async def _set(app, key: str, obj: dict):
    obj["_k"] = key
    return await app.a_run_any(TBEF.DB.SET, query=key, data=json.dumps(obj),
                               get_results=True)


async def _del(app, key: str):
    # ponytail: the DB adapter's delete only works in dict/redis mode — in blob
    # mode it silently no-ops. So tombstone via set (works in every mode) FIRST,
    # then best-effort hard delete. Reads filter out _deleted records.
    # ceiling: tombstones linger in blob mode; a periodic compaction could purge
    # them if the keyspace ever grows large.
    try:
        await app.a_run_any(TBEF.DB.SET, query=key,
                            data=json.dumps({"_k": key, "_deleted": True}),
                            get_results=True)
    except Exception:
        pass
    try:
        return await app.a_run_any(TBEF.DB.DELETE, query=key, get_results=True)
    except Exception:
        return None


# --- courses ----------------------------------------------------------------

async def course_create(app, name: str) -> dict:
    c = {"id": _id(), "name": name, "created": now()}
    await _set(app, f"KURSE::course::{c['id']}", c)
    return c


async def course_list(app) -> list:
    return sorted(await _list(app, "KURSE::course::"), key=lambda x: x.get("created", 0))


async def course_delete(app, cid: str):
    for f in await file_list(app, cid):
        await _del(app, f"KURSE::file::{cid}::{f['id']}")
    await _del(app, f"KURSE::course::{cid}")


# --- files (flat HTML sheets per course) -----------------------------------
# ponytail: no folder tree — one course is a flat, ordered list of HTML files.

async def file_create(app, cid: str, name: str, html: str = "") -> dict:
    existing = await file_list(app, cid)
    nxt = (max((f.get("order", 0) for f in existing), default=-1) + 1) if existing else 0
    f = {"id": _id(), "cid": cid, "name": name, "html": html,
         "order": nxt, "created": now(), "updated": now()}
    await _set(app, f"KURSE::file::{cid}::{f['id']}", f)
    return f


async def file_list(app, cid: str) -> list:
    return sorted(await _list(app, f"KURSE::file::{cid}::"),
                  key=lambda x: x.get("order", 0))


async def file_get(app, cid: str, fid: str) -> dict | None:
    return await _get(app, f"KURSE::file::{cid}::{fid}")


async def file_update(app, cid: str, fid: str, **fields) -> dict | None:
    f = await file_get(app, cid, fid)
    if not f:
        return None
    for k in ("name", "order", "html"):
        if k in fields and fields[k] is not None:
            f[k] = fields[k]
    f["updated"] = now()
    await _set(app, f"KURSE::file::{cid}::{fid}", f)
    return f


async def file_reorder(app, cid: str, fid: str, direction: int) -> list:
    """Swap order with the neighbour above (-1) or below (+1). Reliable."""
    files = await file_list(app, cid)
    i = next((k for k, f in enumerate(files) if f["id"] == fid), None)
    j = None if i is None else i + direction
    if i is not None and j is not None and 0 <= j < len(files):
        a, b = files[i]["order"], files[j]["order"]
        await file_update(app, cid, files[i]["id"], order=b)
        await file_update(app, cid, files[j]["id"], order=a)
    return await file_list(app, cid)


async def file_delete(app, cid: str, fid: str):
    await _del(app, f"KURSE::file::{cid}::{fid}")


def validate_file(f: dict) -> dict:
    issues = []
    html = (f.get("html") or "").strip()
    if not html:
        issues.append("leerer Inhalt")
    elif "<" not in html:
        issues.append("kein HTML erkannt")
    # Kurse-compat hint (non-blocking): the naming model the shell hooks into
    if html and "reveal(" not in html and "data-kurse" not in html:
        issues.append("Hinweis: kein reveal()/data-kurse — Tipp-Tracking evtl. inaktiv")
    return {"ok": not any(x for x in issues if not x.startswith("Hinweis")),
            "issues": issues}


# --- cohorts (share links) --------------------------------------------------

async def cohort_create(app, cid: str, name: str, session_ids: list,
                        anchor: int) -> dict:
    co = {"id": _id(), "cid": cid, "name": name,
          "session_ids": list(session_ids),
          "anchor": max(0, min(int(anchor), max(0, len(session_ids) - 1))),
          "created": now()}
    await _set(app, f"KURSE::cohort::{co['id']}", co)
    return co


async def cohort_get(app, coid: str) -> dict | None:
    return await _get(app, f"KURSE::cohort::{coid}")


async def cohort_list(app, cid: str) -> list:
    return [c for c in await _list(app, "KURSE::cohort::") if c.get("cid") == cid]


async def cohort_update(app, coid: str, name=None, session_ids=None,
                        anchor=None) -> dict | None:
    """Change an existing link's range/anchor/name WITHOUT changing its id
    (the shared URL stays identical — this is how you release more sessions)."""
    co = await cohort_get(app, coid)
    if not co:
        return None
    if name is not None:
        co["name"] = name
    if session_ids is not None:
        co["session_ids"] = list(session_ids)
    if anchor is not None:
        co["anchor"] = max(0, min(int(anchor), max(0, len(co["session_ids"]) - 1)))
    await _set(app, f"KURSE::cohort::{coid}", co)
    return co


async def cohort_delete(app, coid: str):
    # delete the cohort FIRST so a hiccup in participant cleanup can't leave it
    # dangling; then best-effort remove participants by their real stored key.
    await _del(app, f"KURSE::cohort::{coid}")
    for p in await _list(app, f"KURSE::pp::{coid}::"):
        k = p.get("_k")
        if k:
            try:
                await _del(app, k)
            except Exception:
                pass


# --- participants + event log ----------------------------------------------

def _pkey(coid, name):
    return f"KURSE::pp::{coid}::{slug(name)}"


async def pp_join(app, coid: str, name: str) -> dict | None:
    co = await cohort_get(app, coid)
    if not co:
        return None
    key = _pkey(coid, name)
    p = await _get(app, key)
    if not p:
        p = {"name": name, "coid": coid, "cid": co["cid"],
             "pos": co["anchor"], "task": 0, "events": [], "last_seen": now()}
        await _set(app, key, p)
    else:
        p["last_seen"] = now()
        await _set(app, key, p)
    return p


async def pp_event(app, coid: str, name: str, ev: dict) -> dict | None:
    """Append an event. Server owns timestamps.

    Timing model:
      * opening a HINT closes only the previously open hint (the task timer
        keeps running — reading a tip is part of working on the task).
      * opening a TASK closes everything still open (previous task + any hint),
        so a task's duration includes the tips read during it.
    """
    key = _pkey(coid, name)
    p = await _get(app, key)
    if not p:
        return None
    t = now()
    etype = ev.get("type", "task")
    if etype == "task":
        for e in p["events"]:
            if "closed_at" not in e:
                e["closed_at"] = t
    else:  # hint — close only the last open hint
        for e in reversed(p["events"]):
            if e["type"] == "hint" and "closed_at" not in e:
                e["closed_at"] = t
                break
    entry = {"type": etype, "sIdx": int(ev.get("sIdx", p["pos"])),
             "taskIdx": int(ev.get("taskIdx", p["task"])), "opened_at": t}
    if etype == "hint":
        entry["hint"] = int(ev.get("hint", 0))
    p["events"].append(entry)
    p["pos"] = entry["sIdx"]
    p["task"] = entry["taskIdx"]
    p["last_seen"] = t
    await _set(app, key, p)
    return p


async def pp_touch_pos(app, coid: str, name: str, sidx: int) -> dict | None:
    """Update the sheet pointer without wiping progress.

    Same sheet (resume/refresh) → keep task + running timers.
    Different sheet → close open events, reset to task 0, start its timer.
    """
    key = _pkey(coid, name)
    p = await _get(app, key)
    if not p:
        return None
    if p["pos"] != sidx:
        t = now()
        for e in p["events"]:
            if "closed_at" not in e:
                e["closed_at"] = t
        p["pos"], p["task"] = sidx, 0
        p["events"].append({"type": "task", "sIdx": sidx, "taskIdx": 0, "opened_at": t})
    p["last_seen"] = now()
    await _set(app, key, p)
    return p


def _stats(p: dict) -> dict:
    """Derive live stats from the raw event log."""
    ev = p.get("events", [])
    pos, task = p["pos"], p["task"]
    cur_hints = [e for e in ev if e["type"] == "hint"
                 and e.get("sIdx") == pos and e.get("taskIdx") == task]
    cur_hints_used = len({e.get("hint", 0) for e in cur_hints})   # distinct levels
    cur_hints_open = sum(1 for e in cur_hints if "closed_at" not in e)
    hints_seen = sum(1 for e in ev if e["type"] == "hint")        # all-time (info)
    tasks = [e for e in ev if e["type"] == "task"]
    cur_since = prev_secs = 0
    if tasks:
        last = tasks[-1]
        cur_since = round((last.get("closed_at", now())) - last["opened_at"], 1)
        if len(tasks) >= 2:
            pe = tasks[-2]
            prev_secs = round(pe.get("closed_at", last["opened_at"]) - pe["opened_at"], 1)
    return {"name": p["name"], "pos": pos, "task": task,
            "cur_hints_used": cur_hints_used, "cur_hints_open": cur_hints_open,
            "hints_seen": hints_seen,
            "cur_task_since": cur_since, "prev_task_secs": prev_secs,
            "last_seen": p.get("last_seen", 0)}


async def cohort_live(app, coid: str) -> list:
    ps = await _list(app, f"KURSE::pp::{coid}::")
    return sorted((_stats(p) for p in ps), key=lambda s: s["name"])
