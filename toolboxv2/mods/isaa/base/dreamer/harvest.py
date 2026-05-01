"""
Dreamer V3 — Harvest Module

Parses execution logs from VFS into RunRecords.
Runs BEFORE the DreamerAgent starts (no LLM needed).

Migrated from dreamer.py [F1] section-based parser.

Author: FlowAgent V3
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional, List

_log = logging.getLogger("isaa.dreamer_v3.harvest")


# ═══════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RunRecord:
    """Parsed execution log."""
    run_id: str = ""
    query: str = ""
    tools_used: list = field(default_factory=list)
    success: bool = True
    error_traces: list = field(default_factory=list)
    summary: str = ""
    timestamp: str = ""
    log_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════
# LOG PARSER
# ═══════════════════════════════════════════════════════════════════

def parse_log(content: str, path: str) -> Optional[RunRecord]:
    """
    Parse a commit_run log file into a RunRecord.

    Format:
        # Execution Log: <run_id>
        Query: <short title>
        ----------------------------------------
        ### SYSTEM
        IDENTITY: ...
        ### USER
        <the actual user query>
        ### TOOL
        <tool calls as dict>
        ### ASSISTANT
        <agent response>

    Returns None for trivial/empty logs.
    """
    if not content or not content.strip():
        return None

    record = RunRecord(log_path=path)

    # ── Run-ID from header or filename ──
    header_match = re.search(r'#\s*Execution Log:\s*(\w+)', content)
    if header_match:
        record.run_id = header_match.group(1)

    # Fallback: ID from filename (YYYYMMDD_HHMMSS_runid.md)
    if not record.run_id:
        basename = path.rsplit("/", 1)[-1].replace(".md", "")
        parts = basename.split("_", 2)
        if len(parts) >= 3:
            record.run_id = parts[2]
            record.timestamp = f"{parts[0]}_{parts[1]}"

    # ── Split into sections ──
    sections = re.split(r'^###\s+(\w+)', content, flags=re.MULTILINE)

    user_queries: list = []
    tool_sections: list = []
    system_sections: list = []
    assistant_sections: list = []

    i = 1  # Skip header (index 0)
    while i < len(sections) - 1:
        section_type = sections[i].strip().upper()
        section_content = sections[i + 1].strip()
        i += 2

        if section_type == "USER":
            user_queries.append(section_content)
        elif section_type == "TOOL":
            tool_sections.append(section_content)
        elif section_type == "SYSTEM":
            system_sections.append(section_content)
        elif section_type == "ASSISTANT":
            assistant_sections.append(section_content)

    # ── Extract query: first non-trivial USER section ──
    trivial = {"hi", "hello", "hey", "hallo", ""}
    for uq in user_queries:
        cleaned = uq.strip()
        if len(cleaned) > 5 and cleaned.lower() not in trivial:
            record.query = cleaned[:500]
            break

    # Fallback: "Query: " line from header
    if not record.query:
        for line in content.split("\n")[:5]:
            if line.startswith("Query: "):
                q = line[7:].strip()
                if q and q.lower() not in trivial:
                    record.query = q[:500]
                    break

    # ── Extract tools from TOOL sections and SYSTEM ──
    for ts in tool_sections:
        tool_matches = re.findall(r"'name':\s*'([^']+)'", ts)
        record.tools_used.extend(tool_matches)
        func_matches = re.findall(r'`(\w+)\(`', ts)
        record.tools_used.extend(func_matches)

    for ss in system_sections:
        if "RUN SUMMARY" in ss:
            record.summary = ss[:500]
        tool_refs = re.findall(r'`Tool Call:\s*([^`]+)`', ss)
        record.tools_used.extend(t.strip() for t in tool_refs)
        func_refs = re.findall(
            r'\b(vfs_\w+|spawn_\w+|memory_\w+|analyze_\w+|load_tools|list_tools)\b', ss
        )
        record.tools_used.extend(func_refs)

    # Deduplicate tools
    record.tools_used = list(dict.fromkeys(record.tools_used))

    # ── Error detection ──
    error_patterns = re.compile(
        r'(Fehler|Error|failed|exception|traceback|❌|🔴)',
        re.IGNORECASE
    )
    false_positive_words = {"empfehlung", "beheben", "sofort", "level", "error_handling"}
    for line in content.split("\n"):
        if error_patterns.search(line):
            if any(fp in line.lower() for fp in false_positive_words):
                continue
            record.error_traces.append(line.strip()[:200])

    # ── Success heuristic ──
    if record.error_traces:
        real_errors = [
            e for e in record.error_traces
            if any(kw in e.lower() for kw in ("traceback", "exception", "❌", "failed"))
        ]
        error_ratio = len(real_errors) / max(len(record.tools_used), 1)
        record.success = error_ratio < 0.1
    else:
        record.success = True

    # ── Only return records with meaningful query ──
    if not record.query:
        return None

    return record


# ═══════════════════════════════════════════════════════════════════
# FILTERING
# ═══════════════════════════════════════════════════════════════════

def filter_records(
    records: List[RunRecord],
    query_filter: str = "",
    success_only: bool = False,
    failure_only: bool = False,
    limit: int = 0,
) -> List[RunRecord]:
    """Filter RunRecords by criteria."""
    result = records

    if query_filter:
        kw = query_filter.lower()
        result = [r for r in result if kw in r.query.lower()]

    if success_only:
        result = [r for r in result if r.success]
    elif failure_only:
        result = [r for r in result if not r.success]

    if limit > 0:
        result = result[:limit]

    return result


# ═══════════════════════════════════════════════════════════════════
# CUTOFF CALCULATION
# ═══════════════════════════════════════════════════════════════════

_DEFAULT_LOOKBACK_HOURS = 72


def get_cutoff(
    max_history_time: Optional[float] = None,
    last_run_ts: Optional[str] = None,
) -> datetime:
    """
    Calculate harvest cutoff time.

    Priority:
      1. Explicit max_history_time (hours)
      2. last_run_ts from VFS (ISO string)
      3. Default: 72h lookback
    """
    if max_history_time is not None:
        return datetime.now() - timedelta(hours=max_history_time)

    if last_run_ts:
        try:
            return datetime.fromisoformat(last_run_ts.strip())
        except (ValueError, AttributeError):
            pass

    return datetime.now() - timedelta(hours=_DEFAULT_LOOKBACK_HOURS)


# ═══════════════════════════════════════════════════════════════════
# VFS HARVEST ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════

def harvest_from_vfs(
    vfs,
    log_dir: str,
    cutoff: Optional[datetime],
) -> List[RunRecord]:
    """
    Read and parse all log files from VFS directory.

    Args:
        vfs: VFS instance with ls() and read() methods
        log_dir: Path to log directory (e.g. "/global/.memory/logs")
        cutoff: Only include files newer than this datetime

    Returns:
        List of parsed RunRecords
    """
    ls_result = vfs.ls(log_dir, recursive=False) if hasattr(vfs, 'ls') else vfs.ls(log_dir)
    if not ls_result.get("success"):
        return []

    contents = ls_result.get("contents", [])
    records: List[RunRecord] = []

    for entry in contents:
        name = entry.get("name", "")
        if not name.endswith(".md"):
            continue

        # Parse timestamp from filename: YYYYMMDD_HHMMSS_runid.md
        if cutoff is not None:
            try:
                basename = name.rsplit("/", 1)[-1]
                ts_str = basename[:15]
                file_ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                if file_ts < cutoff:
                    continue
            except (ValueError, IndexError):
                # Can't parse timestamp — include file anyway
                pass

        path = f"{log_dir}/{name}" if not name.startswith("/") else name
        read_result = vfs.read(path)
        if not read_result.get("success"):
            continue

        record = parse_log(read_result.get("content", ""), path)
        if record is not None:
            records.append(record)

    return records
