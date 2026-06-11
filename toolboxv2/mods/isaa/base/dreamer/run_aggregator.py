"""
Dreamer V3 — Run Aggregator

Background-learning aggregation after every run:
obs RunRecord  →  RunMetrics  →  VFS Task Map (/global/.memory/taskmap/).

Classification design (revised):
  - NO standalone classifier service. classifier_app is gone.
  - Run-start:  the Narrator classifies (one call) with fuzzy pre-selection
                support via the classify guide in global VFS
                (`fuzzy_preselect()` below is that support function).
  - Post-run:   the agent's FAST model classifies (one JSON call, candidates
                narrowed by the same fuzzy pre-selection).
  - Uncertainty sink: task_type "new" — runs classified "new" never inject
                happypath/guid into a run start, so false classifications
                degrade gracefully.

Pure logic: VFS and the LLM completion function are injected.
Fully unittest-able without an agent (see test_run_aggregator.py).

Author: FlowAgent V3
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict

_log = logging.getLogger("isaa.dreamer_v3.run_aggregator")

TASKMAP_ROOT = "/global/.memory/taskmap"
CLASSIFY_GUIDE_PATH = f"{TASKMAP_ROOT}/classify_guide.md"

# Seed labels. New labels are added by the Dreamer (guide + index), never
# hardcoded anywhere else.
DEFAULT_TASK_TYPES = ["coding", "conversational", "brainstorming", "homework", "freelancing"]
NEW_TYPE = "new"  # uncertainty sink — no happypath/guid injection for this class

_STOPWORDS = {
    "der", "die", "das", "und", "ist", "ein", "eine", "mit", "für", "von", "auf",
    "nicht", "wie", "was", "ich", "du", "den", "dem", "des", "im", "in", "zu",
    "the", "a", "an", "and", "is", "are", "to", "of", "for", "with", "on", "in",
    "it", "this", "that", "my", "me", "you", "be", "can", "do", "how", "what",
}


# ═══════════════════════════════════════════════════════════════════
# RUN METRICS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RunMetrics:
    """Aggregated intel of a single run — one row in the task map."""
    run_id: str = ""
    timestamp: float = 0.0
    query: str = ""
    success: bool = True
    duration_s: float = 0.0
    total_iterations: int = 0
    tool_call_sequence: list = field(default_factory=list)  # [{name, duration_s, status, error}]
    error_tools: list = field(default_factory=list)         # tool names that failed
    files_modified: list = field(default_factory=list)
    skills_matched: list = field(default_factory=list)
    resume_count: int = 0
    resume_type: str = ""              # "auto" | "max_iter" | "user_content" | ""
    user_provided_content: str = ""    # correction content on user resume — gold for the Dreamer
    topic_drift: float = -1.0          # narrator drift flag (1.0/0.0), -1 = n/a
    repeat: bool = False               # narrator repetition flag
    effort_ratio: float = -1.0         # iters / baseline(avg_trace_length), -1 = no baseline (<3 entries)
    plan_summary: str = ""             # narrator plan summary
    task_type: str = ""
    subtype: str = ""
    is_new_task_type: bool = False
    is_new_subtype: bool = False
    parent_run_id: str = ""
    sub_agent_run_ids: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_formatted_row(self) -> dict:
        """Compact row the Dreamer reads (formatted_row.jsonl)."""
        return {
            "run_id": self.run_id,
            "ts": self.timestamp,
            "ok": self.success,
            "iters": self.total_iterations,
            "dur": round(self.duration_s, 2),
            "tools": [t.get("name", "") for t in self.tool_call_sequence],
            "err_tools": self.error_tools,
            "files": self.files_modified,
            "skills": self.skills_matched,
            "resume": {
                "count": self.resume_count,
                "type": self.resume_type,
                "user_content": self.user_provided_content[:400],
            },
            "drift": self.topic_drift,
            "repeat": self.repeat,
            "effort": self.effort_ratio,
            "plan": self.plan_summary[:200],
            "query": self.query[:300],
            "sub_runs": self.sub_agent_run_ids,
            "new_flag": self.is_new_task_type or self.is_new_subtype,
        }


# ═══════════════════════════════════════════════════════════════════
# OBS RunRecord → RunMetrics extraction (defensive: dict or object)
# ═══════════════════════════════════════════════════════════════════

def _g(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_metrics(run_record, query: str = "", narrator_snapshot: dict | None = None) -> RunMetrics:
    """Build RunMetrics from an obs RunRecord (no LLM, no VFS, cheap)."""
    m = RunMetrics(
        run_id=str(_g(run_record, "run_id", "") or ""),
        timestamp=time.time(),
        query=query or str(_g(run_record, "query", "") or ""),
        success=bool(_g(run_record, "success", True)),
        duration_s=float(_g(run_record, "duration_s", 0.0) or 0.0),
        total_iterations=int(_g(run_record, "total_iterations", 0) or 0),
        files_modified=list(_g(run_record, "files_modified", []) or []),
        skills_matched=list(_g(run_record, "skills_matched", []) or []),
        parent_run_id=str(_g(run_record, "parent_run_id", "") or ""),
    )

    # Sub-agent runs: live-linked dicts [{run_id, status}] or plain ids
    for sub in _g(run_record, "sub_agent_runs", []) or []:
        sid = _g(sub, "run_id", sub if isinstance(sub, str) else "")
        if sid:
            m.sub_agent_run_ids.append(str(sid))

    # Steps → tool sequence + resume facts (facts, not judgements)
    for step in _g(run_record, "steps", []) or []:
        for tc in _g(step, "tool_calls", []) or []:
            entry = {
                "name": str(_g(tc, "name", "") or ""),
                "duration_s": round(float(_g(tc, "duration_s", 0.0) or 0.0), 3),
                "status": str(_g(tc, "status", "ok") or "ok"),
            }
            err = _g(tc, "error", "")
            if err:
                entry["error"] = str(err)[:200]
                m.error_tools.append(entry["name"])
            m.tool_call_sequence.append(entry)
        if _g(step, "is_resume_point", False):
            m.resume_count += 1
            m.resume_type = str(_g(step, "resume_type", "") or m.resume_type or "auto")
            upc = _g(step, "user_provided_content", "")
            if upc:
                m.resume_type = "user_content"
                m.user_provided_content = str(upc)[:1000]

    m.error_tools = list(dict.fromkeys(m.error_tools))

    if narrator_snapshot:
        m.topic_drift = 1.0 if narrator_snapshot.get("drift") else 0.0
        m.repeat = bool(narrator_snapshot.get("repeat"))
        m.plan_summary = str(narrator_snapshot.get("plan_summary", "") or "")

    return m


# ═══════════════════════════════════════════════════════════════════
# CLASSIFY GUIDE + FUZZY PRESELECT
# (the same functions serve the Narrator at run-start)
# ═══════════════════════════════════════════════════════════════════

def parse_classify_guide(guide_text: str) -> list[tuple[str, str, set]]:
    """
    Parse the classify guide into [(task_type, subtype, keyword_set), ...].

    Guide line format (machine-first, fuzzy-matchable, Dreamer-editable):
        coding/toolbox: tb mod flows export module framework
    Lines not matching the pattern are ignored (prose for the Dreamer/agent).
    """
    entries = []
    for line in (guide_text or "").splitlines():
        mt = re.match(r"^\s*([a-z0-9_-]+)/([a-z0-9_-]+)\s*:\s*(.+)$", line.strip(), re.IGNORECASE)
        if not mt:
            continue
        kws = {w for w in mt.group(3).lower().split() if w and w not in _STOPWORDS}
        entries.append((mt.group(1).lower(), mt.group(2).lower(), kws))
    return entries


def fuzzy_preselect(query: str, guide_text: str, top_n: int = 3) -> list[tuple[str, str, int]]:
    """
    Word-overlap pre-selection over the classify guide.
    Returns top_n [(task_type, subtype, score)] — score 0 entries excluded.

    Used by:
      - the Narrator at run-start (narrows candidates for its single
        classification call; with a dominant score it IS the fallback)
      - post-run fast-model classification (candidate narrowing)
    """
    q_words = {w for w in (query or "").lower().split() if w not in _STOPWORDS}
    scored = []
    for task_type, subtype, kws in parse_classify_guide(guide_text):
        score = len(q_words & kws)
        if score > 0:
            scored.append((task_type, subtype, score))
    scored.sort(key=lambda x: -x[2])
    return scored[:top_n]


def default_classify_guide() -> str:
    return (
        "# Classify Guide (auto-maintained)\n"
        "#\n"
        "# Machine format — one line per class, fuzzy-matchable:\n"
        "#   task_type/subtype: keyword keyword keyword ...\n"
        "# The aggregator appends query keywords after each run (bounded).\n"
        "# The Dreamer refines lines and adds new labels here.\n"
        "# Unclear runs are classified as new/general — they inject nothing\n"
        "# into a run start and have Dreamer priority.\n"
        "#\n"
        "coding/general: code script function bug fix implement refactor test python\n"
        "coding/toolbox: toolbox tb mod flows module export app fasttb\n"
        "coding/isaa: isaa agent flowagent skill persona dreamer vfs\n"
        "conversational/general: erkläre explain frage question chat hilfe help\n"
        "brainstorming/general: idee idea brainstorm konzept design vorschlag options\n"
        "homework/general: hausaufgabe homework aufgabe übung exercise uni lecture\n"
        "freelancing/general: kunde client angebot invoice projekt auftrag offer\n"
        f"{NEW_TYPE}/general: \n"
    )


def _top_keywords(query: str, n: int = 4) -> list[str]:
    words = [w for w in re.findall(r"[a-zA-ZäöüÄÖÜß_]{4,}", (query or "").lower()) if w not in _STOPWORDS]
    seen = list(dict.fromkeys(words))
    return seen[:n]


def update_classify_guide(guide_text: str, task_type: str, subtype: str, query: str,
                          max_kw_per_line: int = 24) -> str:
    """
    Append query keywords to the matching guide line (bounded), or add the
    line if the class is new. NEW_TYPE never accumulates keywords.
    """
    if task_type == NEW_TYPE:
        return guide_text
    new_kws = _top_keywords(query)
    if not new_kws:
        return guide_text
    lines = (guide_text or default_classify_guide()).splitlines()
    prefix = f"{task_type}/{subtype}:"
    for i, line in enumerate(lines):
        if line.strip().lower().startswith(prefix):
            existing = line.split(":", 1)[1].split()
            merged = list(dict.fromkeys(existing + new_kws))[:max_kw_per_line]
            lines[i] = f"{task_type}/{subtype}: {' '.join(merged)}"
            break
    else:
        lines.append(f"{task_type}/{subtype}: {' '.join(new_kws)}")
    return "\n".join(lines) + ("\n" if not guide_text.endswith("\n") else "")


# ═══════════════════════════════════════════════════════════════════
# RUN AGGREGATOR
# ═══════════════════════════════════════════════════════════════════

class RunAggregator:
    """
    Aggregates one finished run into the VFS task map.

    Injected dependencies (no agent coupling):
      vfs                  — VFS instance (read/write/mkdir, dict results)
      llm_completion_func  — async (messages, **kw) -> str  (the FAST model;
                             classification post-run). None → fuzzy-only.
    """

    def __init__(self, vfs, llm_completion_func=None):
        self.vfs = vfs
        self.llm = llm_completion_func

    # ── public entry ────────────────────────────────────────────────

    async def aggregate(self, run_record, query: str = "",
                        narrator_snapshot: dict | None = None) -> RunMetrics:
        """
        Steps: ① metrics ② classify (fast model + fuzzy) ③ lookup/upsert
        ④ effort_ratio only with entry_count ≥ 3 ⑤ write rows + index + happypath.
        Never raises — bg learning must not break a run.
        """
        m = extract_metrics(run_record, query=query, narrator_snapshot=narrator_snapshot)
        try:
            guide = self._read(CLASSIFY_GUIDE_PATH)
            if not guide:
                guide = default_classify_guide()
                self._write(CLASSIFY_GUIDE_PATH, guide)

            m.task_type, m.subtype = await self._classify(m.query, guide)
            self._lookup_and_flag(m)

            # effort_ratio: only with baseline (entry_count >= 3)
            sub_index = self._read_json(self._sub_index_path(m)) or {}
            if int(sub_index.get("entry_count", 0)) >= 3:
                baseline = float(sub_index.get("avg_trace_length", 0.0) or 0.0)
                if baseline > 0:
                    m.effort_ratio = round(m.total_iterations / baseline, 3)
            else:
                m.topic_drift = m.topic_drift if m.topic_drift >= 0 else -1.0

            self._persist(m)
            self._write(CLASSIFY_GUIDE_PATH,
                        update_classify_guide(guide, m.task_type, m.subtype, m.query))
        except Exception as e:
            _log.warning(f"aggregate failed for run {m.run_id}: {e}")
        return m

    # ── classification (post-run, fast model) ──────────────────────

    async def _classify(self, query: str, guide: str) -> tuple[str, str]:
        candidates = fuzzy_preselect(query, guide, top_n=3)
        known = sorted({f"{t}/{s}" for t, s, _ in parse_classify_guide(guide)})

        if self.llm is None:
            # fuzzy-only fallback: dominant match or NEW_TYPE sink
            if candidates and candidates[0][2] >= 2:
                return candidates[0][0], candidates[0][1]
            return NEW_TYPE, "general"

        cand_str = ", ".join(f"{t}/{s}" for t, s, _ in candidates) or "none"
        sys = (
            "Classify the task into task_type/subtype. Respond ONLY with JSON: "
            '{"task_type": "...", "subtype": "..."}\n'
            f"Known classes: {', '.join(known)}\n"
            f"Fuzzy pre-selection (likely): {cand_str}\n"
            f'If uncertain or the task fits no class well, use "{NEW_TYPE}"/"general". '
            "A new sensible class name is allowed if clearly warranted."
        )
        try:
            raw = await self.llm(
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": (query or "")[:1500]}],
                model_preference="fast", with_context=False, stream=False,
            )
            data = json.loads(re.search(r"\{.*\}", str(raw), re.DOTALL).group(0))
            tt = re.sub(r"[^a-z0-9_-]", "", str(data.get("task_type", NEW_TYPE)).lower()) or NEW_TYPE
            st = re.sub(r"[^a-z0-9_-]", "", str(data.get("subtype", "general")).lower()) or "general"
            return tt, st
        except Exception as e:
            _log.debug(f"fast-model classify failed ({e}), fuzzy fallback")
            if candidates and candidates[0][2] >= 2:
                return candidates[0][0], candidates[0][1]
            return NEW_TYPE, "general"

    # ── task map paths ──────────────────────────────────────────────

    @staticmethod
    def _type_dir(m: RunMetrics) -> str:
        return f"{TASKMAP_ROOT}/{m.task_type}"

    @staticmethod
    def _sub_dir(m: RunMetrics) -> str:
        return f"{TASKMAP_ROOT}/{m.task_type}/{m.subtype}"

    def _sub_index_path(self, m: RunMetrics) -> str:
        return f"{self._sub_dir(m)}/_index.json"

    # ── lookup / upsert ─────────────────────────────────────────────

    def _lookup_and_flag(self, m: RunMetrics) -> None:
        top = self._read_json(f"{TASKMAP_ROOT}/_index.json") or {"task_types": {}}
        tt = top["task_types"].get(m.task_type)
        if tt is None:
            m.is_new_task_type = True
            self._mkdir(self._type_dir(m))
        if tt is None or m.subtype not in (tt.get("subtypes") or []):
            m.is_new_subtype = True
            self._mkdir(self._sub_dir(m))

    def _persist(self, m: RunMetrics) -> None:
        sub_dir = self._sub_dir(m)
        self._mkdir(sub_dir)

        # tracks.jsonl (raw) + formatted_row.jsonl (Dreamer reads this)
        self._append_jsonl(f"{sub_dir}/tracks.jsonl", m.to_dict())
        self._append_jsonl(f"{sub_dir}/formatted_row.jsonl", m.to_formatted_row())

        # subtype _index.json
        idx_path = self._sub_index_path(m)
        idx = self._read_json(idx_path) or {
            "subtype": m.subtype, "performance": 0.0, "avg_trace_length": 0.0,
            "improvement_trend": 0.0, "best_iterations": 0,
            "last_updated": 0, "entry_count": 0, "is_new": True,
        }
        n = int(idx.get("entry_count", 0))
        perf = float(idx.get("performance", 0.0))
        avg_trace = float(idx.get("avg_trace_length", 0.0))
        new_perf = (perf * n + (1.0 if m.success else 0.0)) / (n + 1)
        new_avg = (avg_trace * n + m.total_iterations) / (n + 1)
        idx.update({
            "performance": round(new_perf, 4),
            "avg_trace_length": round(new_avg, 3),
            "improvement_trend": round(new_perf - perf, 4),
            "entry_count": n + 1,
            "last_updated": int(time.time()),
            "is_new": bool(idx.get("is_new", False)) and n + 1 < 3,
        })

        # happypath.md — only on improvement (success AND fewer iterations)
        best = int(idx.get("best_iterations", 0))
        if m.success and m.task_type != NEW_TYPE and (best == 0 or m.total_iterations < best):
            idx["best_iterations"] = m.total_iterations
            self._write(f"{sub_dir}/happypath.md", self._render_happypath(m))
        # guid.md is NEVER touched here — it belongs to the Dreamer.

        self._write(idx_path, json.dumps(idx, indent=2))

        # top-level _index.json
        top_path = f"{TASKMAP_ROOT}/_index.json"
        top = self._read_json(top_path) or {"task_types": {}}
        tt = top["task_types"].setdefault(m.task_type, {
            "subtypes": [], "success_rate": 0.0, "last_updated": 0, "entry_count": 0,
        })
        if m.subtype not in tt["subtypes"]:
            tt["subtypes"].append(m.subtype)
        tn = int(tt.get("entry_count", 0))
        tt["success_rate"] = round(
            (float(tt.get("success_rate", 0.0)) * tn + (1.0 if m.success else 0.0)) / (tn + 1), 4)
        tt["entry_count"] = tn + 1
        tt["last_updated"] = int(time.time())
        self._write(top_path, json.dumps(top, indent=2))

    @staticmethod
    def _render_happypath(m: RunMetrics) -> str:
        """Best accumulated happy path — mini syntax + reasoning (->)."""
        lines = [
            f"# Happy Path — {m.task_type}/{m.subtype}",
            f"<!-- run {m.run_id}, {m.total_iterations} iters, {round(m.duration_s, 1)}s -->",
            "",
            f"Query: {m.query[:200]}",
        ]
        if m.plan_summary:
            lines.append(f"Plan -> {m.plan_summary[:200]}")
        lines.append("")
        for tc in m.tool_call_sequence:
            mark = "" if tc.get("status") == "ok" else "  [FAILED — avoid]"
            lines.append(f"-> {tc.get('name')}{mark}")
        if m.files_modified:
            lines.append("")
            lines.append(f"Files: {', '.join(m.files_modified[:10])}")
        return "\n".join(lines) + "\n"

    # ── VFS helpers (dict-result API) ───────────────────────────────

    def _read(self, path: str) -> str:
        try:
            r = self.vfs.read(path)
            return r.get("content", "") if r.get("success") else ""
        except Exception:
            return ""

    def _read_json(self, path: str):
        raw = self._read(path)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def _write(self, path: str, content: str) -> None:
        try:
            self.vfs.write(path, content)
        except Exception as e:
            _log.warning(f"vfs write {path} failed: {e}")

    def _append_jsonl(self, path: str, row: dict) -> None:
        existing = self._read(path)
        line = json.dumps(row, ensure_ascii=False, default=str)
        self._write(path, (existing.rstrip("\n") + "\n" if existing else "") + line + "\n")

    def _mkdir(self, path: str) -> None:
        try:
            self.vfs.mkdir(path, parents=True)
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════════════
# PRE-INJECTION HOOK (run-start, flag-activated)
# ═══════════════════════════════════════════════════════════════════

GUIDE_SWAP_NOTICE = (
    "NOTE: This guide was selected by a fast pre-classification. If it does "
    "not fit your actual task, IGNORE it (close it mentally) or load a better "
    f"one yourself from {TASKMAP_ROOT}/<task_type>/<subtype>/guid.md via vfs. "
    "A wrong pre-classification is rare and must never constrain you."
)

# token budgets per spec: happypath ≤ 800 tok, guid ≤ 400 tok (~4 chars/tok)
_HAPPYPATH_MAX_CHARS = 3200
_GUID_MAX_CHARS = 1600


async def classify_for_injection(query: str, guide: str, narrator_call=None) -> tuple[str, str]:
    """
    Run-start classification: fuzzy pre-selection over the classify guide,
    confirmed by ONE narrator blitz call. Uncertainty → NEW_TYPE (inject nothing).

    narrator_call: async (system: str, candidates: list[str]) -> dict|None
                   (engine wraps narrator.blitz with a JSON schema).
    """
    candidates = fuzzy_preselect(query, guide, top_n=3)

    # dominant fuzzy match → no LLM call needed (zero latency)
    if candidates and (candidates[0][2] >= 3 or (len(candidates) == 1 and candidates[0][2] >= 2)):
        return candidates[0][0], candidates[0][1]

    if narrator_call is None:
        if candidates and candidates[0][2] >= 2:
            return candidates[0][0], candidates[0][1]
        return NEW_TYPE, "general"

    known = sorted({f"{t}/{s}" for t, s, _ in parse_classify_guide(guide)})
    cand_str = ", ".join(f"{t}/{s}" for t, s, _ in candidates) or "none"
    system = (
        "Classify this task into task_type/subtype. "
        f"Known classes: {', '.join(known)}. "
        f"Fuzzy pre-selection (likely): {cand_str}. "
        f'If uncertain, answer task_type="{NEW_TYPE}", subtype="general". '
        'Respond as JSON: {"task_type": "...", "subtype": "..."}'
    )
    try:
        data = await narrator_call(system, query)
        if isinstance(data, dict) and data.get("task_type"):
            tt = re.sub(r"[^a-z0-9_-]", "", str(data["task_type"]).lower()) or NEW_TYPE
            st = re.sub(r"[^a-z0-9_-]", "", str(data.get("subtype", "general")).lower()) or "general"
            return tt, st
    except Exception as e:
        _log.debug(f"narrator classify failed: {e}")
    if candidates and candidates[0][2] >= 2:
        return candidates[0][0], candidates[0][1]
    return NEW_TYPE, "general"


async def build_preinjection(vfs, query: str, narrator_call=None) -> str:
    """
    Flag-activated pre-context block for the run start.
    Returns "" whenever nothing should be injected:
      - classify guide missing / task map empty
      - task_type == NEW_TYPE (uncertainty sink — by design no injection)
      - class has neither happypath nor guid
    Never raises.
    """
    try:
        r = vfs.read(CLASSIFY_GUIDE_PATH)
        guide = r.get("content", "") if r.get("success") else ""
        if not guide:
            return ""

        task_type, subtype = await classify_for_injection(query, guide, narrator_call)
        if task_type == NEW_TYPE:
            return ""

        base = f"{TASKMAP_ROOT}/{task_type}/{subtype}"
        hp_r = vfs.read(f"{base}/happypath.md")
        gd_r = vfs.read(f"{base}/guid.md")
        happypath = (hp_r.get("content", "") if hp_r.get("success") else "")[:_HAPPYPATH_MAX_CHARS]
        guid = (gd_r.get("content", "") if gd_r.get("success") else "")[:_GUID_MAX_CHARS]
        if not happypath and not guid:
            return ""

        parts = [f"## TASK MAP PRE-CONTEXT ({task_type}/{subtype})", GUIDE_SWAP_NOTICE]
        if guid:
            parts.append(f"### Task Guide\n{guid}")
        if happypath:
            parts.append(f"### Known Happy Path\n{happypath}")
        return "\n\n".join(parts)
    except Exception as e:
        _log.debug(f"build_preinjection failed: {e}")
        return ""
