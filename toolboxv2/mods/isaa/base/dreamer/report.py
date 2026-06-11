"""
Dreamer V3 — Report Module

Builds the end-of-dream HTML dashboard report for the user:
what the Dreamer did, what the system learned, how it improved,
and what it can do now.

Pure functions — VFS / managers injected, fully unittest-able.
Output: one self-contained HTML file (no external assets, renders offline).

Author: FlowAgent V3
"""

import html
import json
import logging
import time
from datetime import datetime

_log = logging.getLogger("isaa.dreamer_v3.report")

TASKMAP_ROOT = "/global/.memory/taskmap"
REPORT_DIR = "/global/.memory/dreamer/reports"


# ═══════════════════════════════════════════════════════════════════
# SNAPSHOTS
# ═══════════════════════════════════════════════════════════════════

def collect_system_snapshot(skills_manager=None, rule_set=None, personas: dict | None = None) -> dict:
    """Counts + capability list. Call before AND after a dream cycle."""
    snap = {
        "skill_count": 0, "active_skills": 0, "avg_confidence": 0.0,
        "rule_count": 0, "pattern_count": 0, "persona_count": len(personas or {}),
        "skills": [],  # [{name, confidence, usage, triggers}]
    }
    try:
        skills = getattr(skills_manager, "skills", {}) or {}
        confs = []
        for s in skills.values():
            conf = float(getattr(s, "confidence", 0.0) or 0.0)
            confs.append(conf)
            active = True
            try:
                active = bool(s.is_active())
            except Exception:
                pass
            snap["skills"].append({
                "name": str(getattr(s, "name", "?")),
                "confidence": round(conf, 3),
                "usage": int(getattr(s, "usage_count", getattr(s, "total_uses", 0)) or 0),
                "active": active,
                "triggers": list(getattr(s, "triggers", []) or [])[:6],
            })
            if active:
                snap["active_skills"] += 1
        snap["skill_count"] = len(skills)
        snap["avg_confidence"] = round(sum(confs) / len(confs), 3) if confs else 0.0
    except Exception as e:
        _log.debug(f"skill snapshot failed: {e}")
    try:
        snap["rule_count"] = len(getattr(rule_set, "situation_rules", {}) or {})
        snap["pattern_count"] = len(getattr(rule_set, "learned_patterns", []) or [])
    except Exception:
        pass
    return snap


def collect_taskmap_overview(vfs) -> dict:
    """Top index + per-subtype indexes — what the system knows per task class."""
    out = {"task_types": {}}
    try:
        r = vfs.read(f"{TASKMAP_ROOT}/_index.json")
        if not r.get("success"):
            return out
        top = json.loads(r.get("content", "{}"))
        for tt, info in (top.get("task_types") or {}).items():
            entry = {
                "entry_count": info.get("entry_count", 0),
                "success_rate": info.get("success_rate", 0.0),
                "subtypes": {},
            }
            for st in info.get("subtypes", []):
                base = f"{TASKMAP_ROOT}/{tt}/{st}"
                sub = {}
                rr = vfs.read(f"{base}/_index.json")
                if rr.get("success"):
                    try:
                        sub = json.loads(rr.get("content", "{}"))
                    except Exception:
                        sub = {}
                sub["has_guid"] = bool(
                    vfs.read(f"{base}/guid.md").get("success", False)
                    and vfs.read(f"{base}/guid.md").get("content")
                )
                sub["has_happypath"] = bool(
                    vfs.read(f"{base}/happypath.md").get("success", False)
                )
                entry["subtypes"][st] = sub
            out["task_types"][tt] = entry
    except Exception as e:
        _log.debug(f"taskmap overview failed: {e}")
    return out


# ═══════════════════════════════════════════════════════════════════
# HTML BUILDER
# ═══════════════════════════════════════════════════════════════════

def _e(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _delta(before, after, suffix: str = "") -> str:
    d = round((after or 0) - (before or 0), 3)
    if d > 0:
        return f'<span class="up">+{d}{suffix}</span>'
    if d < 0:
        return f'<span class="down">{d}{suffix}</span>'
    return f'<span class="flat">±0{suffix}</span>'


def _card(label: str, before, after, suffix: str = "") -> str:
    return (
        f'<div class="card"><div class="card-label">{_e(label)}</div>'
        f'<div class="card-value">{_e(after)}{_e(suffix)}</div>'
        f'<div class="card-delta">{_e(before)}{_e(suffix)} &rarr; {_delta(before, after, suffix)}</div></div>'
    )


def _meter(frac: float) -> str:
    pct = max(0, min(100, int(round((frac or 0.0) * 100))))
    return (f'<div class="meter"><div class="meter-fill" style="width:{pct}%"></div>'
            f'<span class="meter-num">{pct}%</span></div>')


def _action_rows(report: dict) -> list[tuple[str, str, list]]:
    """(label, verb-class, items) for every non-empty action bucket."""
    mapping = [
        ("skills_created", "Skills erstellt", "add"),
        ("skills_evolved", "Skills verbessert", "mod"),
        ("skills_merged", "Skills zusammengeführt", "mod"),
        ("skills_split", "Skills aufgeteilt", "mod"),
        ("skills_compressed", "Skills komprimiert", "mod"),
        ("skills_deleted", "Skills gelöscht", "del"),
        ("skills_deactivated", "Skills deaktiviert", "del"),
        ("rules_created", "Regeln gelernt", "add"),
        ("rules_deleted", "Regeln gelöscht", "del"),
        ("patterns_added", "Patterns gespeichert", "add"),
        ("patterns_pruned", "Patterns entfernt", "del"),
        ("personas_evolved", "Personas angepasst", "mod"),
        ("personas_pruned", "Personas entfernt", "del"),
        ("memories_added", "Memory-Einträge", "add"),
        ("taskmap_guides_written", "Task-Guides geschrieben", "add"),
    ]
    rows = []
    for key, label, cls in mapping:
        items = report.get(key) or []
        if items:
            rows.append((label, cls, items))
    return rows


def build_dream_report_html(
    meta: dict,
    before: dict,
    after: dict,
    actions_report: dict,
    taskmap: dict,
    final_answer: str = "",
) -> str:
    """
    Self-contained dashboard. Sections:
      1 Nachtprotokoll (header)   2 System Health Δ (cards)
      3 Aktionen                  4 Gelernt (Task Map intel)
      5 Fähigkeiten jetzt         6 Dreamer-Report (final answer)
    """
    ts = meta.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M")
    agent = _e(meta.get("agent", "?"))
    dur = meta.get("duration_s", 0)

    cards = "".join([
        _card("Skills", before.get("skill_count", 0), after.get("skill_count", 0)),
        _card("Aktive Skills", before.get("active_skills", 0), after.get("active_skills", 0)),
        _card("&empty; Confidence", before.get("avg_confidence", 0.0), after.get("avg_confidence", 0.0)),
        _card("Regeln", before.get("rule_count", 0), after.get("rule_count", 0)),
        _card("Patterns", before.get("pattern_count", 0), after.get("pattern_count", 0)),
        _card("Personas", before.get("persona_count", 0), after.get("persona_count", 0)),
    ])

    # Actions
    act_html = []
    for label, cls, items in _action_rows(actions_report):
        lis = "".join(
            f"<li>{_e(i if isinstance(i, str) else json.dumps(i, ensure_ascii=False, default=str)[:240])}</li>"
            for i in items[:25]
        )
        act_html.append(
            f'<div class="act {cls}"><h3>{_e(label)} '
            f'<span class="count">{len(items)}</span></h3><ul>{lis}</ul></div>'
        )
    actions_block = "".join(act_html) or '<p class="muted">Keine Aktionen protokolliert.</p>'

    # Task map intel — what the system has learned per class
    tm_rows = []
    for tt, info in sorted((taskmap.get("task_types") or {}).items()):
        for st, sub in sorted((info.get("subtypes") or {}).items()):
            perf = float(sub.get("performance", 0.0) or 0.0)
            badges = []
            if sub.get("has_guid"):
                badges.append('<span class="badge guid">guide</span>')
            if sub.get("has_happypath"):
                badges.append('<span class="badge hp">happy path</span>')
            if sub.get("is_new"):
                badges.append('<span class="badge new">new</span>')
            tm_rows.append(
                f"<tr><td>{_e(tt)}/{_e(st)}</td>"
                f"<td>{_e(sub.get('entry_count', 0))}</td>"
                f"<td>{_meter(perf)}</td>"
                f"<td>{_e(sub.get('avg_trace_length', 0))}</td>"
                f"<td>{_e(sub.get('improvement_trend', 0))}</td>"
                f"<td>{' '.join(badges)}</td></tr>"
            )
    tm_block = (
        '<table><thead><tr><th>Task-Klasse</th><th>Runs</th><th>Erfolg</th>'
        '<th>&empty; Iterationen</th><th>Trend</th><th>Wissen</th></tr></thead>'
        f"<tbody>{''.join(tm_rows)}</tbody></table>"
        if tm_rows else '<p class="muted">Task Map noch leer — Background-Learning sammelt ab dem nächsten Run.</p>'
    )

    # Capabilities now
    skill_rows = []
    for s in sorted(after.get("skills", []), key=lambda x: -x.get("confidence", 0))[:40]:
        state = "aktiv" if s.get("active") else "inaktiv"
        skill_rows.append(
            f"<tr class=\"{'' if s.get('active') else 'dim'}\">"
            f"<td>{_e(s.get('name'))}</td><td>{_meter(s.get('confidence', 0.0))}</td>"
            f"<td>{_e(s.get('usage', 0))}</td><td>{_e(', '.join(s.get('triggers', [])))}</td>"
            f"<td>{state}</td></tr>"
        )
    skills_block = (
        '<table><thead><tr><th>Skill</th><th>Confidence</th><th>Nutzungen</th>'
        '<th>Trigger</th><th>Status</th></tr></thead>'
        f"<tbody>{''.join(skill_rows)}</tbody></table>"
        if skill_rows else '<p class="muted">Keine Skills vorhanden.</p>'
    )

    final_block = (
        f'<pre class="final">{_e(final_answer[:8000])}</pre>'
        if final_answer else '<p class="muted">Kein Abschluss-Report vom Dreamer.</p>'
    )

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dream Report — {agent} — {_e(ts)}</title>
<style>
  :root {{
    --ink:#0b0e14; --panel:#11151f; --line:#1f2735; --text:#c9d1d9;
    --muted:#5c6573; --dawn:#e8a33d; --up:#7fb069; --down:#d06a5e;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--ink); color:var(--text);
    font:14px/1.55 ui-monospace,'JetBrains Mono','Cascadia Mono',Consolas,monospace; }}
  main {{ max-width:1060px; margin:0 auto; padding:32px 20px 80px; }}
  header {{ border-bottom:1px solid var(--line); padding-bottom:18px; margin-bottom:28px; }}
  .cron {{ color:var(--muted); font-size:12px; letter-spacing:.08em; }}
  h1 {{ margin:.3em 0 0; font-size:26px; font-weight:600; color:var(--dawn); }}
  h1 .moon {{ color:var(--muted); font-weight:400; }}
  h2 {{ font-size:13px; letter-spacing:.18em; text-transform:uppercase;
        color:var(--muted); border-bottom:1px solid var(--line);
        padding-bottom:6px; margin:40px 0 16px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }}
  .card {{ background:var(--panel); border:1px solid var(--line); padding:12px 14px; }}
  .card-label {{ color:var(--muted); font-size:11px; letter-spacing:.1em; text-transform:uppercase; }}
  .card-value {{ font-size:24px; margin:4px 0 2px; }}
  .card-delta {{ font-size:12px; color:var(--muted); }}
  .up {{ color:var(--up); }} .down {{ color:var(--down); }} .flat {{ color:var(--muted); }}
  .act {{ background:var(--panel); border:1px solid var(--line);
          border-left:3px solid var(--muted); padding:10px 16px; margin-bottom:10px; }}
  .act.add {{ border-left-color:var(--up); }}
  .act.mod {{ border-left-color:var(--dawn); }}
  .act.del {{ border-left-color:var(--down); }}
  .act h3 {{ margin:0 0 6px; font-size:13px; font-weight:600; }}
  .act .count {{ color:var(--muted); font-weight:400; }}
  .act ul {{ margin:0; padding-left:18px; color:var(--text); }}
  .act li {{ margin:2px 0; font-size:13px; }}
  table {{ width:100%; border-collapse:collapse; background:var(--panel);
           border:1px solid var(--line); font-size:13px; }}
  th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }}
  th {{ color:var(--muted); font-size:11px; letter-spacing:.1em; text-transform:uppercase; font-weight:500; }}
  tr.dim td {{ color:var(--muted); }}
  .meter {{ position:relative; background:var(--ink); border:1px solid var(--line);
            height:14px; min-width:90px; }}
  .meter-fill {{ background:var(--dawn); height:100%; opacity:.55; }}
  .meter-num {{ position:absolute; inset:0; font-size:10px; line-height:14px;
                text-align:center; color:var(--text); }}
  .badge {{ font-size:10px; padding:1px 6px; border:1px solid var(--line); color:var(--muted); }}
  .badge.guid {{ color:var(--dawn); border-color:var(--dawn); }}
  .badge.hp {{ color:var(--up); border-color:var(--up); }}
  .badge.new {{ color:var(--down); border-color:var(--down); }}
  .final {{ background:var(--panel); border:1px solid var(--line); padding:16px;
            white-space:pre-wrap; word-break:break-word; font-size:13px; }}
  .muted {{ color:var(--muted); }}
  @media (prefers-reduced-motion:no-preference) {{
    .meter-fill {{ transition:width .6s ease; }}
  }}
</style>
</head>
<body>
<main>
  <header>
    <div class="cron"># dream cycle &middot; agent={agent} &middot; {_e(ts)} &middot; {_e(round(dur, 1))}s</div>
    <h1>Nachtprotokoll <span class="moon">— was das System gelernt hat</span></h1>
  </header>

  <h2>System Health &Delta;</h2>
  <div class="cards">{cards}</div>

  <h2>Aktionen dieses Zyklus</h2>
  {actions_block}

  <h2>Gelernt — Task Map Intel</h2>
  {tm_block}

  <h2>F&auml;higkeiten jetzt</h2>
  {skills_block}

  <h2>Dreamer-Abschlussreport</h2>
  {final_block}
</main>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════
# WRITE
# ═══════════════════════════════════════════════════════════════════

def write_dream_report(vfs, html_content: str, agent_name: str = "agent") -> str:
    """Write the report into VFS. Returns the VFS path ('' on failure)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{REPORT_DIR}/dream_{agent_name}_{ts}.html"
    try:
        try:
            vfs.mkdir(REPORT_DIR, parents=True)
        except Exception:
            pass
        r = vfs.write(path, html_content)
        if isinstance(r, dict) and not r.get("success", True):
            return ""
        return path
    except Exception as e:
        _log.warning(f"dream report write failed: {e}")
        return ""
