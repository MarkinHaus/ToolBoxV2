"""
toolboxv2/flows/doc_server.py
─────────────────────────────
MkDocs Documentation Hub — TBJS Paper Style

Dual-purpose documentation server:
  • Users  -> browse & search rendered docs, view changelogs
  • Devs   -> coverage dashboard, staleness tracker, missing-docs audit,
             live editor, doc-generation jobs with status tracking

Data comes from DocsSystem (mkdocs.py) — pre-built inverted index,
no filesystem scanning at request time.

Stack: FastTB + TBJS Paper neo-brutalism
Run  : `tb doc_server` or `python -m toolboxv2 -fg doc_server`
"""

import asyncio
import os
import re
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

from toolboxv2 import get_app, get_logger, tb_root_dir, Spinner, profile_code
from toolboxv2.utils.workers.fast_tb import FastTB
from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
from toolboxv2.utils.workers.server_worker import ParsedRequest
from toolboxv2.utils.extras.mkdocs import DocsSystem

NAME = "doc_server"
logger = get_logger()

# ── Paths (defaults — overridden by run() args) ─────────
PROJECT_ROOT = tb_root_dir.parent
DOCS_DIR = PROJECT_ROOT / "docs"
MKDOCS_YML = PROJECT_ROOT / "mkdocs.yml"

# ── State ────────────────────────────────────────────────
_docs: Optional[DocsSystem] = None
_tb_app = None     # ToolBoxV2 App instance

# ── FastTB app ───────────────────────────────────────────
web_app = FastTB(title="ToolBoxV2 Docs")


def _get_docs():
    """Get the initialized DocsSystem."""
    if _docs is None:
        logger.error("DocsSystem not initialized yet")
        raise RuntimeError("Documentation index is still loading — try again in a moment")
    return _docs


# ═════════════════════════════════════════════════════════
# JOB TRACKER — server-side, survives browser close
# ═════════════════════════════════════════════════════════


class JobTracker:
    """In-memory job tracker for doc-generation tasks."""

    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, file_path: str, update_existing: str = None) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "file_path": file_path,
                "update_existing": update_existing,
                "status": "queued",
                "created": time.time(),
                "started": None,
                "finished": None,
                "result": None,
                "error": None,
                "output_path": None,
            }
        logger.info(f"Job created: {job_id} for {file_path}")
        return job_id

    def update(self, job_id: str, **kwargs):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(kwargs)
                logger.debug(f"Job {job_id} updated: {list(kwargs.keys())}")

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            return self._jobs.get(job_id, {}).copy() if job_id in self._jobs else None

    def list_all(self) -> list[dict]:
        with self._lock:
            return [j.copy() for j in sorted(
                self._jobs.values(), key=lambda j: j["created"], reverse=True
            )]


_jobs = JobTracker()

# ── Context budget management for doc generation ──

# Approximate chars-per-token ratio for code (conservative)
_CHARS_PER_TOKEN = 3.5

# Model context limits (tokens). Override via LOGCONTEXTMODEL if available.
_DEFAULT_CONTEXT_TOKENS = 200_000
_RESERVED_SYSTEM_TOKENS = 6_000    # system prompt + output headroom
_RESERVED_OUTPUT_TOKENS = 8_000    # space for the generated doc
_MAX_EXISTING_DOC_CHARS = 5_000


def _estimate_tokens(text: str) -> int:
    """Rough token estimate from char count."""
    return int(len(text) / _CHARS_PER_TOKEN)


def _get_context_budget() -> int:
    """Get available tokens for code content in the prompt."""
    try:
        from toolboxv2.utils.extras.blobs import LOGCONTEXTMODEL
        if LOGCONTEXTMODEL is not None:
            # Use the complex model's context size if available
            ctx = getattr(LOGCONTEXTMODEL, 'context_length', None)
            if ctx and isinstance(ctx, int):
                return ctx - _RESERVED_SYSTEM_TOKENS - _RESERVED_OUTPUT_TOKENS
    except (ImportError, AttributeError):
        pass
    return _DEFAULT_CONTEXT_TOKENS - _RESERVED_SYSTEM_TOKENS - _RESERVED_OUTPUT_TOKENS


def _build_code_prompt_sections(elements: list, budget_tokens: int, job_id: str = "") -> list[str]:
    """
    Build code sections for the doc-gen prompt, respecting a token budget.

    Strategy (applied only when total exceeds budget):
    1. FULL MODE: All elements with code blocks (≤2000 chars each) — if fits, done.
    2. COMPACT MODE: Classes get signature+docstring+method-table, functions get signature+docstring.
       Code blocks only for small elements (≤500 chars). Large code omitted.
    3. SUMMARY MODE: Group by class → list methods as table rows. Loose functions grouped.
       No code blocks at all.
    4. CHUNKED: If even summary doesn't fit, split into class-groups and return
       only the first chunk (caller must handle multi-pass).
    """
    # First: try full mode
    full_sections = _format_elements_full(elements)
    full_tokens = sum(_estimate_tokens(s) for s in full_sections)

    if full_tokens <= budget_tokens:
        logger.debug(f"Job {job_id}: full mode — {full_tokens} tokens, budget {budget_tokens}")
        return full_sections

    # Compact mode: reduce code blocks
    compact_sections = _format_elements_compact(elements)
    compact_tokens = sum(_estimate_tokens(s) for s in compact_sections)

    if compact_tokens <= budget_tokens:
        logger.info(f"Job {job_id}: compact mode — {compact_tokens} tokens (full was {full_tokens})")
        return compact_sections

    # Summary mode: no code blocks
    summary_sections = _format_elements_summary(elements)
    summary_tokens = sum(_estimate_tokens(s) for s in summary_sections)

    if summary_tokens <= budget_tokens:
        logger.info(f"Job {job_id}: summary mode — {summary_tokens} tokens (full was {full_tokens})")
        return summary_sections

    # Chunked: take what fits from summary
    logger.warning(f"Job {job_id}: chunked mode — summary {summary_tokens} > budget {budget_tokens}, truncating")
    result = []
    running = 0
    for section in summary_sections:
        t = _estimate_tokens(section)
        if running + t > budget_tokens:
            result.append(f"\n\n> ⚠ **Truncated**: {len(summary_sections) - len(result)} elements omitted (context limit).")
            break
        result.append(section)
        running += t
    return result


def _format_elements_full(elements: list) -> list[str]:
    """Full detail: signature + metadata + docstring + code (capped at 2000 chars)."""
    sections = []
    for elem in elements:
        entry = f"### `{elem['signature']}`"
        if elem.get("parent"):
            entry = f"### `{elem['parent']}.{elem['name']}` — {elem['type']}"
        entry += f"\n- Type: {elem['type']}"
        entry += f"\n- Lines: {elem['lines'][0]}-{elem['lines'][1]}"
        entry += f"\n- Language: {elem.get('language', 'python')}"
        if elem.get("docstring"):
            entry += f"\n- Docstring: {elem['docstring']}"
        if elem.get("code"):
            code = elem["code"]
            if len(code) > 2000:
                code = code[:2000] + "\n# ... (truncated)"
            entry += f"\n```python\n{code}\n```"
        sections.append(entry)
    return sections


def _format_elements_compact(elements: list) -> list[str]:
    """
    Compact: classes get method tables instead of full code.
    Small elements (≤500 chars code) keep their code blocks.
    Large elements get signature+docstring only.
    """
    # Group by class
    classes: dict[str, list] = {}
    loose: list = []
    class_elems: dict[str, dict] = {}

    for elem in elements:
        if elem["type"] == "class":
            class_elems[elem["name"]] = elem
            classes.setdefault(elem["name"], [])
        elif elem.get("parent") and elem["parent"] in classes:
            classes[elem["parent"]].append(elem)
        else:
            loose.append(elem)

    sections = []

    for cls_name, methods in classes.items():
        cls = class_elems.get(cls_name, {})
        entry = f"### `{cls.get('signature', cls_name)}`"
        entry += f"\n- Type: class"
        if cls.get("docstring"):
            entry += f"\n- Docstring: {cls['docstring']}"

        # Small class with code → include it
        code = cls.get("code", "")
        if code and len(code) <= 500:
            entry += f"\n```python\n{code}\n```"

        # Methods as table
        if methods:
            entry += "\n\n| Method | Signature | Docstring |"
            entry += "\n|--------|-----------|-----------|"
            for m in methods:
                doc = (m.get("docstring") or "—")[:80].replace("|", "\\|").replace("\n", " ")
                sig = m.get("signature", m["name"]).replace("|", "\\|")
                entry += f"\n| `{m['name']}` | `{sig}` | {doc} |"

        sections.append(entry)

    for elem in loose:
        entry = f"### `{elem['signature']}`"
        entry += f"\n- Type: {elem['type']}"
        if elem.get("docstring"):
            entry += f"\n- Docstring: {elem['docstring']}"
        code = elem.get("code", "")
        if code and len(code) <= 500:
            entry += f"\n```python\n{code}\n```"
        sections.append(entry)

    return sections


def _format_elements_summary(elements: list) -> list[str]:
    """
    Minimal: no code blocks. Classes with method lists, functions as signatures.
    """
    classes: dict[str, list] = {}
    loose: list = []
    class_elems: dict[str, dict] = {}

    for elem in elements:
        if elem["type"] == "class":
            class_elems[elem["name"]] = elem
            classes.setdefault(elem["name"], [])
        elif elem.get("parent") and elem["parent"] in classes:
            classes[elem["parent"]].append(elem)
        else:
            loose.append(elem)

    sections = []

    for cls_name, methods in classes.items():
        cls = class_elems.get(cls_name, {})
        entry = f"### `{cls.get('signature', cls_name)}`"
        if cls.get("docstring"):
            entry += f"\n{cls['docstring'][:200]}"
        if methods:
            entry += f"\n\nMethods ({len(methods)}):"
            for m in methods:
                doc_hint = f" — {m['docstring'][:60]}" if m.get("docstring") else ""
                entry += f"\n- `{m.get('signature', m['name'])}`{doc_hint}"
        sections.append(entry)

    # Group loose functions
    if loose:
        entry = "### Module-level Functions\n"
        for fn in loose:
            doc_hint = f" — {fn['docstring'][:60]}" if fn.get("docstring") else ""
            entry += f"\n- `{fn.get('signature', fn['name'])}`{doc_hint}"
        sections.append(entry)

    return sections

# ═════════════════════════════════════════════════════════
# DATA LAYER — reads from DocsSystem index, zero I/O
# ═════════════════════════════════════════════════════════


def _get_doc_pages() -> list[dict]:
    """Get doc sections from the pre-built index."""
    docs = _get_docs()
    sections = docs.index_mgr.index.sections
    by_file = defaultdict(list)
    for s in sections.values():
        by_file[s.file_path].append(s)

    pages = []
    for file_path, secs in sorted(by_file.items()):
        fp = Path(file_path)
        try:
            rel = fp.relative_to(DOCS_DIR)
        except ValueError:
            try:
                rel = fp.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = fp
        word_count = sum(len(s.content.split()) for s in secs)
        pages.append({
            "name": fp.stem,
            "rel_path": str(rel),
            "full_path": file_path,
            "mtime": datetime.fromtimestamp(
                secs[0].last_modified, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M") if secs else "",
            "word_count": word_count,
            "headings": len(secs),
        })
    return pages


def _get_code_elements() -> dict:
    """Get code elements grouped by file from the index."""
    docs = _get_docs()
    elements = docs.index_mgr.index.code_elements
    by_file = defaultdict(list)
    for eid, elem in elements.items():
        by_file[elem.file_path].append(elem)
    return by_file


def _compute_coverage() -> dict:
    """Cross-reference code elements vs doc sections — all from index."""
    docs = _get_docs()
    idx = docs.index_mgr.index

    doc_names = set()
    doc_files = set()
    for s in idx.sections.values():
        doc_names.add(s.title.lower())
        doc_files.add(Path(s.file_path).stem.lower())
        for ref in s.source_refs:
            doc_names.add(ref.lower())

    by_file = defaultdict(list)
    for eid, elem in idx.code_elements.items():
        by_file[elem.file_path].append(elem)

    documented_files = []
    missing_files = []

    for file_path, elems in sorted(by_file.items()):
        fp = Path(file_path)
        file_stem = fp.stem.lower()

        classes = sum(1 for e in elems if e.element_type == "class")
        functions = sum(1 for e in elems if e.element_type in ("function", "method"))
        has_docstring = any(e.docstring for e in elems)

        info = {
            "name": fp.stem,
            "rel_path": str(Path(file_path).relative_to(PROJECT_ROOT)).replace("\\", "/") if file_path.startswith(str(PROJECT_ROOT)) else file_path.replace("\\", "/"),
            "classes": classes,
            "functions": functions,
            "has_docstring": has_docstring,
            "elements": len(elems),
        }

        is_documented = (
            file_stem in doc_files
            or any(e.name.lower() in doc_names for e in elems if e.element_type == "class")
        )

        if is_documented:
            documented_files.append(info)
        else:
            missing_files.append(info)

    total = max(len(by_file), 1)
    documented = len(documented_files)
    pct = int(documented / total * 100)

    stale = []
    for file_path, fhash in idx.file_hashes.items():
        fp = Path(file_path)
        if fp.suffix in ('.py', '.js', '.ts'):
            doc_stem = fp.stem.lower()
            if doc_stem in doc_files:
                for s in idx.sections.values():
                    if Path(s.file_path).stem.lower() == doc_stem:
                        try:
                            src_mtime = fp.stat().st_mtime
                        except OSError:
                            continue
                        if src_mtime > s.last_modified:
                            stale.append({
                                "doc_path": str(Path(s.file_path).relative_to(DOCS_DIR)) if DOCS_DIR in Path(s.file_path).parents else s.file_path,
                                "doc_mtime": datetime.fromtimestamp(s.last_modified, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                                "src_path": file_path,
                                "src_mtime": datetime.fromtimestamp(src_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                            })
                        break

    docstring_pct = sum(1 for f in documented_files + missing_files if f["has_docstring"]) / total * 100
    freshness = max(0, 100 - len(stale) * 10)
    pages = len(set(s.file_path for s in idx.sections.values()))
    avg_words = sum(len(s.content.split()) for s in idx.sections.values()) / max(pages, 1)
    depth_score = min(100, avg_words / 3)
    quality = pct * 0.4 + freshness * 0.2 + docstring_pct * 0.2 + depth_score * 0.2

    return {
        "total_modules": len(by_file),
        "documented": documented,
        "missing": missing_files,
        "stale": stale,
        "pages": pages,
        "pct": pct,
        "quality_score": round(quality, 1),
        "total_elements": len(idx.code_elements),
        "total_sections": len(idx.sections),
    }


# ═════════════════════════════════════════════════════════
# HEALTH — per-directory breakdown
# ═════════════════════════════════════════════════════════


def _compute_health() -> dict:
    """Per-directory breakdown of functions, classes, and doc coverage."""
    docs = _get_docs()
    idx = docs.index_mgr.index

    doc_names = set()
    for s in idx.sections.values():
        doc_names.add(s.title.lower())
        for ref in s.source_refs:
            doc_names.add(ref.lower())

    by_dir: dict[str, dict] = {}

    for eid, elem in idx.code_elements.items():
        fp = Path(elem.file_path)
        # Get relative directory from project root
        try:
            rel = fp.relative_to(PROJECT_ROOT)
            dir_key = str(rel.parent)
        except ValueError:
            dir_key = str(fp.parent)

        if dir_key not in by_dir:
            by_dir[dir_key] = {
                "classes": 0, "functions": 0, "methods": 0,
                "documented": 0, "undocumented": 0,
                "total": 0,
            }

        bucket = by_dir[dir_key]
        bucket["total"] += 1

        if elem.element_type == "class":
            bucket["classes"] += 1
        elif elem.element_type == "function":
            bucket["functions"] += 1
        elif elem.element_type == "method":
            bucket["methods"] += 1

        # Check if documented (has docstring OR appears in doc sections)
        is_doc = bool(elem.docstring) or elem.name.lower() in doc_names
        if is_doc:
            bucket["documented"] += 1
        else:
            bucket["undocumented"] += 1

    # Orphan docs — docs referencing non-existent code
    code_names = {elem.name.lower() for elem in idx.code_elements.values()}
    code_files = {Path(elem.file_path).stem.lower() for elem in idx.code_elements.values()}

    orphan_docs = []
    for sid, section in idx.sections.items():
        fp = Path(section.file_path)
        doc_stem = fp.stem.lower()
        # A doc is orphaned if its stem doesn't match any code file
        # AND none of its source_refs match known code names
        if doc_stem not in code_files:
            refs_valid = any(ref.lower() in code_names for ref in section.source_refs)
            title_valid = section.title.lower() in code_names
            if not refs_valid and not title_valid:
                try:
                    rel = str(fp.relative_to(DOCS_DIR))
                except ValueError:
                    try:
                        rel = str(fp.relative_to(PROJECT_ROOT))
                    except ValueError:
                        rel = str(fp)
                orphan_docs.append({
                    "rel_path": rel,
                    "title": section.title,
                    "words": len(section.content.split()),
                })

    # Totals
    total_all = sum(d["total"] for d in by_dir.values())
    doc_all = sum(d["documented"] for d in by_dir.values())

    return {
        "by_dir": dict(sorted(by_dir.items())),
        "total_elements": total_all,
        "total_documented": doc_all,
        "total_undocumented": total_all - doc_all,
        "doc_pct": int(doc_all / max(total_all, 1) * 100),
        "orphan_docs": orphan_docs,
    }


# ═════════════════════════════════════════════════════════
# TEMPLATE ENGINE — TBJS Paper Neo-Brutalism
# ═════════════════════════════════════════════════════════

PAPER_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

:root {
  --raw-primary: 55% 0.18 230;
  --raw-success: 65% 0.2 145;
  --raw-warning: 75% 0.18 85;
  --raw-error:   55% 0.22 25;

  --primary: oklch(55% 0.18 230);
  --success: oklch(65% 0.2 145);
  --warning: oklch(75% 0.18 85);
  --error:   oklch(55% 0.22 25);

  --paper-bg:      #f4f1ea;
  --paper-surface: #ffffff;
  --paper-sunken:  #ebe7dc;
  --ink:           #111111;
  --ink-muted:     #555555;
  --ink-faint:     #888888;
  --rule:          #111111;

  --font-display: 'IBM Plex Mono', ui-monospace, 'SF Mono', Consolas, monospace;
  --font-body:    'IBM Plex Sans', system-ui, -apple-system, sans-serif;

  --text-display: clamp(32px, 5vw, 48px);
  --text-h1: clamp(28px, 3.5vw, 36px);
  --text-h2: clamp(22px, 2.5vw, 28px);
  --text-h3: clamp(18px, 2vw, 22px);
  --text-base: 16px;
  --text-sm: 14px;
  --text-xs: 12px;

  --space-1: 4px;  --space-2: 8px;  --space-3: 12px;
  --space-4: 16px; --space-5: 24px; --space-6: 32px;
  --space-7: 40px; --space-8: 48px; --space-9: 64px;
  --space-10: 80px; --space-11: 96px; --space-12: 128px;
}

[data-theme="dark"] {
  --paper-bg:      #1a1a1a;
  --paper-surface: #2a2a2a;
  --paper-sunken:  #0f0f0f;
  --ink:           #f4f1ea;
  --ink-muted:     #b8b3a8;
  --ink-faint:     #7a7770;
  --rule:          #f4f1ea;
}

html { font-size: 16px; }

body {
  font-family: var(--font-body);
  font-size: var(--text-base);
  line-height: 1.6;
  color: var(--ink);
  background: var(--paper-bg);
  min-height: 100vh;
}

.nav {
  position: sticky; top: 0; z-index: 100;
  display: flex; align-items: center; gap: var(--space-5);
  padding: var(--space-4) var(--space-6);
  background: var(--paper-bg);
  border-block-end: 3px solid var(--ink);
}
.nav-brand {
  font-family: var(--font-display); font-size: var(--text-h3);
  font-weight: 700; color: var(--ink); text-decoration: none; letter-spacing: -0.02em;
}
.nav-links { display: flex; gap: var(--space-4); margin-left: auto; }
.nav-link {
  font-family: var(--font-display); font-size: var(--text-sm); font-weight: 500;
  text-transform: uppercase; letter-spacing: 1px; color: var(--ink); text-decoration: none;
  padding: var(--space-1) 0; border-bottom: 2px solid transparent; transition: border-color 80ms linear;
}
.nav-link:hover, .nav-link.active { border-bottom-color: var(--ink); }
.nav-toggle {
  display: none; background: none; border: 2px solid var(--ink); color: var(--ink);
  font-family: var(--font-display); font-size: var(--text-base);
  padding: var(--space-2) var(--space-3); cursor: pointer;
}

.container { max-width: 1100px; margin: 0 auto; padding: var(--space-7) var(--space-6); }
.page-title {
  font-family: var(--font-display); font-size: var(--text-h1); font-weight: 600;
  line-height: 1.15; letter-spacing: -0.02em; color: var(--ink); margin-bottom: var(--space-6);
}
.page-subtitle {
  font-family: var(--font-body); font-size: var(--text-base); color: var(--ink-muted);
  max-width: 68ch; margin-top: calc(-1 * var(--space-4)); margin-bottom: var(--space-6);
}

.card {
  padding: var(--space-5); background: var(--paper-surface); border: 2px solid var(--ink);
  border-radius: 0; box-shadow: 6px 6px 0 var(--ink); margin-bottom: var(--space-6); margin-right: 8px;
  transition: transform 100ms linear, box-shadow 100ms linear;
}
.card:hover { transform: translate(-2px, -2px); box-shadow: 8px 8px 0 var(--ink); }
.card-title { font-family: var(--font-display); font-size: var(--text-h3); font-weight: 600; margin: 0 0 var(--space-3); }
.card-eyebrow {
  font-family: var(--font-display); font-size: var(--text-xs); text-transform: uppercase;
  letter-spacing: 2px; color: var(--ink-muted); margin: 0 0 var(--space-3);
}
.card--flat { box-shadow: none; transition: none; }
.card--flat:hover { transform: none; box-shadow: none; }

.grid { display: grid; gap: var(--space-6); }
.grid-2 { grid-template-columns: repeat(2, 1fr); }
.grid-3 { grid-template-columns: repeat(3, 1fr); }
.grid-4 { grid-template-columns: repeat(4, 1fr); }

.badge {
  display: inline-flex; align-items: center; padding: 0.2rem 0.5rem;
  font-family: var(--font-display); font-size: var(--text-xs); font-weight: 600;
  text-transform: uppercase; letter-spacing: 1px;
  background: var(--ink); color: var(--paper-bg); border: 2px solid var(--ink); border-radius: 0;
}
.badge--success { background: var(--success); border-color: var(--success); color: #fff; }
.badge--warning { background: var(--warning); border-color: var(--warning); color: var(--ink); }
.badge--error   { background: var(--error); border-color: var(--error); color: #fff; }
.badge--ghost   { background: transparent; color: var(--ink); }

.btn {
  display: inline-flex; align-items: center; gap: 0.5rem;
  padding: 0.75rem 1.25rem; font-family: var(--font-display); font-size: var(--text-sm);
  font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; text-decoration: none;
  background: var(--paper-surface); color: var(--ink); border: 2px solid var(--ink);
  border-radius: 0; box-shadow: 4px 4px 0 var(--ink); cursor: pointer;
  transition: transform 80ms linear, box-shadow 80ms linear;
}
.btn:hover { transform: translate(-2px, -2px); box-shadow: 6px 6px 0 var(--ink); }
.btn:active { transform: translate(2px, 2px); box-shadow: 0 0 0 var(--ink); }
.btn--primary { background: var(--primary); color: #fff; }
.btn--danger  { background: var(--error); color: #fff; }
.btn--ghost   { background: transparent; }
.btn--sm { padding: 0.4rem 0.75rem; font-size: var(--text-xs); }

.progress { height: 12px; background: var(--paper-sunken); border: 2px solid var(--ink); border-radius: 0; overflow: hidden; }
.progress-fill { height: 100%; transition: width 600ms ease; }

.table-wrap { overflow-x: auto; margin-right: 8px; margin-bottom: 8px; }
table {
  width: 100%; border-collapse: separate; border-spacing: 0;
  font-size: var(--text-sm); font-family: var(--font-body); border: 2px solid var(--ink);
}
th {
  font-family: var(--font-display); font-size: var(--text-xs); font-weight: 600;
  text-transform: uppercase; letter-spacing: 1px; color: var(--paper-bg); background: var(--ink);
  text-align: left; padding: var(--space-3) var(--space-4); white-space: nowrap;
}
td {
  padding: var(--space-3) var(--space-4); border-bottom: 2px solid var(--paper-sunken);
  vertical-align: middle; line-height: 1.5; background: var(--paper-surface);
}
td:first-child { white-space: nowrap; font-weight: 500; }
tr:nth-child(even) td { background: var(--paper-sunken); }
tr:hover td { background: var(--ink); color: var(--paper-bg); }
tr:hover td a { color: var(--paper-bg); }
tr:hover td code { color: var(--paper-bg); }
tr:hover td .badge { border-color: var(--paper-bg); }
td.path-cell {
  max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  color: var(--ink-muted); font-size: var(--text-xs);
}

.md-content table { margin: var(--space-5) 0; box-shadow: 4px 4px 0 var(--ink); }
.md-content th { font-size: var(--text-xs); padding: var(--space-2) var(--space-3); }
.md-content td { font-size: var(--text-sm); padding: var(--space-2) var(--space-3); }
.md-content td code { font-size: 0.85em; font-weight: 500; }

.mermaid-container {
  margin: var(--space-5) 0; padding: var(--space-5); background: var(--paper-surface);
  border: 2px solid var(--ink); box-shadow: 4px 4px 0 var(--ink); overflow-x: auto; text-align: center;
}

.stat { text-align: center; }
.stat-value {
  font-family: var(--font-display); font-size: var(--text-display); font-weight: 700;
  line-height: 1; color: var(--ink);
}
.stat-label {
  font-family: var(--font-display); font-size: var(--text-xs); text-transform: uppercase;
  letter-spacing: 2px; color: var(--ink-muted); margin-top: var(--space-2);
}

textarea, input[type="text"], select {
  width: 100%; padding: 0.75rem 1rem; font-family: var(--font-body); font-size: var(--text-base);
  color: var(--ink); background: var(--paper-surface); border: 2px solid var(--ink);
  border-radius: 0; box-shadow: 4px 4px 0 var(--ink);
  transition: box-shadow 80ms linear, transform 80ms linear;
}
textarea:focus, input[type="text"]:focus {
  outline: none; transform: translate(-1px, -1px);
  box-shadow: 5px 5px 0 var(--primary); border-color: var(--primary);
}
textarea { font-family: var(--font-display); font-size: var(--text-sm); resize: vertical; }
label {
  display: block; font-family: var(--font-display); font-size: var(--text-sm); font-weight: 600;
  text-transform: uppercase; letter-spacing: 1px; margin-bottom: var(--space-2); color: var(--ink-muted);
}

.callout { padding: 1rem 1.25rem; margin: var(--space-5) 0; background: var(--paper-sunken); border-left: 6px solid var(--ink); }
.callout--warn  { border-left-color: var(--warning); }
.callout--error { border-left-color: var(--error); }
.callout--ok    { border-left-color: var(--success); }

.search-box { position: relative; max-width: 400px; }
.search-box input { padding-left: 2.5rem; }
.search-box::before {
  content: "⌕"; position: absolute; left: 0.75rem; top: 50%; transform: translateY(-50%);
  font-size: 1.2rem; color: var(--ink-muted); z-index: 1;
}

code {
  font-family: var(--font-display); font-size: 0.85em; background: var(--paper-sunken);
  padding: 0.1em 0.35em; border: none; border-radius: 0; white-space: nowrap;
}
td code { background: transparent; padding: 0; font-weight: 500; white-space: nowrap; }

pre {
  font-family: var(--font-display); font-size: var(--text-sm); line-height: 1.6;
  background: var(--paper-sunken); padding: 1.25rem 1.5rem; border: 2px solid var(--ink);
  box-shadow: 4px 4px 0 var(--ink); overflow-x: auto; margin: var(--space-5) 0;
  white-space: pre; tab-size: 4;
}
pre code { background: none; border: none; padding: 0; font-size: inherit; line-height: inherit; word-break: normal; white-space: pre; }

.kw  { color: oklch(55% 0.18 230); font-weight: 600; }
.fn  { color: oklch(55% 0.15 280); }
.cls { color: oklch(55% 0.18 230); font-weight: 600; }
.str { color: oklch(60% 0.18 145); }
.num { color: oklch(60% 0.18 85); }
.cmt { color: var(--ink-faint); font-style: italic; }
.dec { color: oklch(55% 0.15 280); }
.op  { color: var(--ink-muted); }
[data-theme="dark"] .kw  { color: oklch(70% 0.15 230); }
[data-theme="dark"] .fn  { color: oklch(70% 0.12 280); }
[data-theme="dark"] .cls { color: oklch(70% 0.15 230); }
[data-theme="dark"] .str { color: oklch(72% 0.15 145); }
[data-theme="dark"] .num { color: oklch(75% 0.15 85); }
[data-theme="dark"] .cmt { color: var(--ink-faint); }

a { color: var(--primary); text-decoration: underline; text-decoration-thickness: 2px; text-underline-offset: 3px; }
a:hover { background: var(--primary); color: var(--paper-surface); text-decoration-color: transparent; }

hr { border: none; border-block-start: 2px solid var(--ink); margin-block: var(--space-6); }

.md-content h1, .md-content h2, .md-content h3 {
  font-family: var(--font-display); font-weight: 600; line-height: 1.15; margin: var(--space-5) 0 var(--space-3);
}
.md-content h1 { font-size: var(--text-h1); }
.md-content h2 { font-size: var(--text-h2); border-bottom: 2px solid var(--ink); padding-bottom: var(--space-2); }
.md-content h3 { font-size: var(--text-h3); }
.md-content p  { max-inline-size: 68ch; margin-bottom: 1.2em; }
.md-content ul, .md-content ol { padding-left: var(--space-6); margin-bottom: 1.2em; }
.md-content li { margin-bottom: var(--space-2); }
.md-content blockquote { padding: 1rem 1.25rem; background: var(--paper-sunken); border-left: 6px solid var(--ink); margin: var(--space-4) 0; }

@media (max-width: 767px) {
  .grid-2, .grid-3, .grid-4 { grid-template-columns: 1fr; }
  .nav { flex-wrap: wrap; gap: var(--space-3); }
  .nav-links { width: 100%; }
  .nav-link { padding: var(--space-3) var(--space-4); }
  .container { padding: var(--space-5) var(--space-4); }
  .card { box-shadow: 3px 3px 0 var(--ink); margin-right: 4px; }
  .card:hover { box-shadow: 5px 5px 0 var(--ink); }
  .nav-toggle { display: block; margin-left: auto; }
  .nav-links.collapsed { display: none; }
}

.theme-toggle {
  background: none; border: 2px solid var(--ink); color: var(--ink);
  font-family: var(--font-display); font-size: var(--text-sm);
  padding: var(--space-1) var(--space-3); cursor: pointer;
  text-transform: uppercase; letter-spacing: 1px;
  transition: transform 80ms linear, box-shadow 80ms linear; box-shadow: 2px 2px 0 var(--ink);
}
.theme-toggle:hover { transform: translate(-1px, -1px); box-shadow: 3px 3px 0 var(--ink); }
.theme-toggle:active { transform: translate(1px, 1px); box-shadow: 0 0 0 var(--ink); }

.job-spinner {
  display: inline-block; width: 14px; height: 14px;
  border: 2px solid var(--ink-faint); border-top-color: var(--primary);
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
"""


# ── Base layout ─────────────────────────────

def _layout(content: str, title: str = "Docs", active: str = "") -> str:
    def _active(name: str) -> str:
        return "active" if name == active else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — ToolBoxV2</title>
<style>{PAPER_CSS}</style>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<script>
  document.addEventListener('DOMContentLoaded', function() {{
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    mermaid.initialize({{
      startOnLoad: true,
      theme: isDark ? 'dark' : 'neutral',
      fontFamily: "'IBM Plex Mono', monospace",
      themeVariables: isDark
        ? {{ primaryColor: '#58a6ff', primaryTextColor: '#f4f1ea', lineColor: '#f4f1ea', primaryBorderColor: '#f4f1ea' }}
        : {{ primaryColor: '#2563eb', primaryTextColor: '#111', lineColor: '#111', primaryBorderColor: '#111' }}
    }});
  }});
</script>
</head>
<body>

<nav class="nav">
  <a href="/" class="nav-brand">TB·DOCS</a>
  <button class="theme-toggle" id="theme-btn" onclick="toggleTheme()">◐</button>
  <button class="nav-toggle" onclick="document.querySelector('.nav-links').classList.toggle('collapsed')">MENU</button>
  <div class="nav-links">
    <a href="/" class="nav-link {_active('dashboard')}">Dashboard</a>
    <a href="/browse" class="nav-link {_active('browse')}">Browse</a>
    <a href="/search" class="nav-link {_active('search')}">Search</a>
    <a href="/health" class="nav-link {_active('health')}">Health</a>
    <a href="/inventory" class="nav-link {_active('inventory')}">Inventory</a>
    <a href="/relationships" class="nav-link {_active('relationships')}">Map</a>
    <a href="/coverage" class="nav-link {_active('coverage')}">Coverage</a>
    <a href="/jobs" class="nav-link {_active('jobs')}">Jobs<span id="job-badge" style="display:none;margin-left:4px;min-width:18px;height:18px;border-radius:50%;background:var(--error);color:#fff;font-size:10px;font-weight:700;text-align:center;line-height:18px;padding:0 4px"></span></a>
    <a href="/edit" class="nav-link {_active('edit')}">Editor</a>
  </div>
</nav>

<div class="container">
{content}
</div>

<script>
function toggleTheme() {{
  const html = document.documentElement;
  const isDark = html.getAttribute('data-theme') === 'dark';
  html.setAttribute('data-theme', isDark ? '' : 'dark');
  localStorage.setItem('tb-docs-theme', isDark ? 'light' : 'dark');
  document.getElementById('theme-btn').textContent = isDark ? '◐' : '◑';
}}
(function() {{
  const saved = localStorage.getItem('tb-docs-theme');
  if (saved === 'dark') {{
    document.documentElement.setAttribute('data-theme', 'dark');
    const btn = document.getElementById('theme-btn');
    if (btn) btn.textContent = '◑';
  }}
}})();

async function generateDoc(filePath, btn) {{
  btn.disabled = true;
  btn.innerHTML = '<span class="job-spinner"></span> Queuing...';
  try {{
    const resp = await fetch('/api/generate-doc', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{file_path: filePath}})
    }});
    const data = await resp.json();
    if (data.job_id) {{
      btn.innerHTML = '✓ Queued';
      btn.className = 'btn btn--sm btn--primary';
      setTimeout(() => window.location.href = '/jobs', 1000);
    }} else {{
      btn.innerHTML = '✗ ' + (data.error || 'Failed');
      btn.className = 'btn btn--sm btn--danger';
      btn.disabled = false;
    }}
  }} catch(e) {{
    btn.innerHTML = '✗ Error';
    btn.className = 'btn btn--sm btn--danger';
    btn.disabled = false;
  }}
}}

async function fixStaleDoc(docPath, srcPath, btn) {{
  btn.disabled = true;
  btn.innerHTML = '<span class="job-spinner"></span> Queuing...';
  try {{
    const resp = await fetch('/api/generate-doc', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{file_path: srcPath, update_existing: docPath}})
    }});
    const data = await resp.json();
    if (data.job_id) {{
      btn.innerHTML = '✓ Queued';
      btn.className = 'btn btn--sm btn--primary';
      setTimeout(() => window.location.href = '/jobs', 1000);
    }} else {{
      btn.innerHTML = '✗ ' + (data.error || 'Failed');
      btn.className = 'btn btn--sm btn--danger';
      btn.disabled = false;
    }}
  }} catch(e) {{
    btn.innerHTML = '✗ Error';
    btn.className = 'btn btn--sm btn--danger';
    btn.disabled = false;
  }}
}}

async function deleteDoc(docPath, btn) {{
  btn.disabled = true;
  btn.innerHTML = 'Deleting...';
  try {{
    const resp = await fetch('/api/doc/' + encodeURIComponent(docPath), {{
      method: 'DELETE'
    }});
    const data = await resp.json();
    if (data.ok) {{
      btn.closest('tr').style.opacity = '0.3';
      btn.innerHTML = '✓ Deleted';
      setTimeout(() => location.reload(), 1000);
    }} else {{
      btn.innerHTML = '✗ ' + (data.error || 'Failed');
      btn.disabled = false;
    }}
  }} catch(e) {{
    btn.innerHTML = '✗ Error';
    btn.disabled = false;
  }}
}}

(function pollJobs() {{
  fetch('/api/jobs').then(r=>r.json()).then(jobs=>{{
    const active = jobs.filter(j=>j.status==='running'||j.status==='queued').length;
    const badge = document.getElementById('job-badge');
    if (badge) {{
      if (active > 0) {{
        badge.textContent = active;
        badge.style.display = 'inline-block';
      }} else {{
        badge.style.display = 'none';
      }}
    }}
    if (active > 0) setTimeout(pollJobs, 3000);
    else setTimeout(pollJobs, 15000);
  }}).catch(()=>setTimeout(pollJobs, 15000));
}})();
</script>

</body>
</html>"""


def _render(content: str, title: str = "Docs", active: str = "") -> str:
    """Return full HTML page string."""
    return _layout(content, title, active)


# ═════════════════════════════════════════════════════════
# MARKDOWN RENDERING
# ═════════════════════════════════════════════════════════


def _highlight_python(code: str) -> str:
    """Single-pass Python syntax highlighter."""
    PY_KW = (
        r'def|class|return|if|elif|else|for|while|try|except|finally|with|as|'
        r'import|from|raise|yield|async|await|pass|break|continue|and|or|not|in|is|'
        r'None|True|False|self|cls|lambda|global|nonlocal'
    )
    TOKEN_RE = re.compile(
        r'(?P<cmt>#[^\n]*)'
        r'|(?P<tstr>"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')'
        r'|(?P<str>f?"[^"\\]*(?:\\.[^"\\]*)*"|f?\'[^\'\\]*(?:\\.[^\'\\]*)*\')'
        r'|(?P<dec>@\w+)'
        r'|(?P<defn>(?<=def\s)\w+)'
        r'|(?P<clsn>(?<=class\s)\w+)'
        r'|(?P<kw>\b(?:' + PY_KW + r')\b)'
        r'|(?P<num>\b\d+\.?\d*\b)'
    )

    def _replace(m):
        for name in ('cmt', 'tstr', 'str', 'dec', 'defn', 'clsn', 'kw', 'num'):
            val = m.group(name)
            if val is not None:
                css = {'tstr': 'str', 'defn': 'fn', 'clsn': 'cls'}.get(name, name)
                safe = val.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                return f'<span class="{css}">{safe}</span>'
        return m.group(0).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    result = TOKEN_RE.sub(_replace, code)
    parts = re.split(r'(<span class="[^"]+">|</span>)', result)
    out = []
    for part in parts:
        if part.startswith('<span') or part == '</span>':
            out.append(part)
        else:
            if '<span' not in part:
                part = part.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                part = part.replace("&amp;amp;", "&amp;").replace("&amp;lt;", "&lt;").replace("&amp;gt;", "&gt;")
            out.append(part)
    return "".join(out)


def _render_md_table(table_text: str) -> str:
    """Convert a markdown table string to an HTML table."""
    lines = [l.strip() for l in table_text.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return table_text

    def _parse_row(line: str) -> list[str]:
        cells = line.split("|")
        if cells and not cells[0].strip():
            cells = cells[1:]
        if cells and not cells[-1].strip():
            cells = cells[:-1]
        return [c.strip() for c in cells]

    is_sep = bool(re.match(r'^[|\s:\-]+$', lines[1])) if len(lines) > 1 else False
    header = _parse_row(lines[0])
    html = '<div class="table-wrap"><table>\n<thead><tr>'
    for cell in header:
        html += f"<th>{cell}</th>"
    html += "</tr></thead>\n<tbody>\n"

    start = 2 if is_sep else 1
    for line in lines[start:]:
        if re.match(r'^[|\s:\-]+$', line):
            continue
        cells = _parse_row(line)
        html += "<tr>"
        for cell in cells:
            cell = re.sub(r"`([^`]+)`", r"<code>\1</code>", cell)
            html += f"<td>{cell}</td>"
        html += "</tr>\n"

    html += "</tbody></table></div>"
    return html


def _render_md(raw: str) -> str:
    """Minimal markdown -> HTML with syntax highlighting, tables, and mermaid."""
    code_blocks = []
    def _stash_code(m):
        lang = m.group(1) or ""
        code = m.group(2)
        idx = len(code_blocks)
        placeholder = f"\x00CODE{idx}\x00"
        if lang == "mermaid":
            escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            code_blocks.append(f'<div class="mermaid-container"><pre class="mermaid">{escaped}</pre></div>')
        else:
            if lang in ("python", "py", ""):
                code = _highlight_python(code)
            else:
                code = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            code_blocks.append(f'<pre><code class="lang-{lang}">{code}</code></pre>')
        return placeholder

    html = re.sub(r"```(\w*)\n(.*?)```", _stash_code, raw, flags=re.S)

    table_blocks = []
    def _stash_table(m):
        idx = len(table_blocks)
        placeholder = f"\x00TABLE{idx}\x00"
        table_blocks.append(_render_md_table(m.group(0)))
        return placeholder

    html = re.sub(r"(?:^[|].*[|]\s*\n){2,}", _stash_table, html, flags=re.M)

    html = html.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = re.sub(r"^#### (.+)$", r"<h4>\1</h4>", html, flags=re.M)
    html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.M)
    html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.M)
    html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.M)
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
    html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)
    html = re.sub(r"^&gt;\s*(.+)$", r"<blockquote>\1</blockquote>", html, flags=re.M)
    html = re.sub(r"^- (.+)$", r"<li>\1</li>", html, flags=re.M)
    html = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', html)
    html = re.sub(r"\n\n+", "</p><p>", html)
    html = f"<p>{html}</p>"
    html = re.sub(r"<p>\s*</p>", "", html)

    for i, block in enumerate(code_blocks):
        html = html.replace(f"\x00CODE{i}\x00", block)
    for i, block in enumerate(table_blocks):
        html = html.replace(f"\x00TABLE{i}\x00", block)

    return html


# ═════════════════════════════════════════════════════════
# HELPER — code ref counting
# ═════════════════════════════════════════════════════════

_GENERIC_NAMES = frozenset({
    "self", "init", "main", "test", "args", "data", "name", "type",
    "none", "true", "false", "help", "list", "dict", "file", "path",
    "result", "value", "error", "config", "setup", "build", "load",
    "save", "read", "write", "update", "delete", "create", "close",
    "open", "start", "stop", "state", "event", "index", "parse",
    "handle", "process", "response", "request", "model", "query",
})


def _get_meaningful_code_names(docs_sys) -> set:
    names = set()
    for elem in docs_sys.index_mgr.index.code_elements.values():
        n = elem.name.lower()
        if len(n) > 4 and n not in _GENERIC_NAMES and not n.startswith("_"):
            names.add(n)
    return names


def _count_code_refs_for_file(docs_sys, file_path: str, code_names: set) -> int:
    inverted = docs_sys.index_mgr.index.inverted
    section_ids = inverted.file_to_sections.get(file_path, set())
    if not section_ids:
        return 0
    matched = set()
    for name in code_names:
        kw_sections = inverted.keyword_to_sections.get(name, set())
        if kw_sections & section_ids:
            matched.add(name)
    return len(matched)


# ═════════════════════════════════════════════════════════
# ROUTES — DASHBOARD
# ═════════════════════════════════════════════════════════


@web_app.get("/")
async def dashboard():
    cov = _compute_coverage()

    def _color(pct: int) -> str:
        if pct >= 70:
            return "var(--success)"
        return "var(--warning)" if pct >= 40 else "var(--error)"

    def _badge(pct: int) -> str:
        if pct >= 70:
            return "badge--success"
        return "badge--warning" if pct >= 40 else "badge--error"

    def _grade(score: float) -> str:
        if score >= 80:
            return "A"
        if score >= 60:
            return "B"
        if score >= 40:
            return "C"
        return "D" if score >= 20 else "F"

    content = f"""
<h1 class="page-title">Documentation Dashboard</h1>
<p class="page-subtitle">
  Overview of ToolBoxV2 documentation health — coverage, freshness, and quality at a glance.
</p>
<div class="grid grid-4">
  <div class="card">
    <div class="stat">
      <div class="stat-value">{cov["pct"]}%</div>
      <div class="stat-label">Coverage</div>
    </div>
  </div>
  <div class="card">
    <div class="stat">
      <div class="stat-value">{_grade(cov["quality_score"])}</div>
      <div class="stat-label">Quality</div>
    </div>
  </div>
  <div class="card">
    <div class="stat">
      <div class="stat-value">{len(cov["stale"])}</div>
      <div class="stat-label">Stale Docs</div>
    </div>
  </div>
  <div class="card">
    <div class="stat">
      <div class="stat-value">{len(cov["missing"])}</div>
      <div class="stat-label">Undocumented</div>
    </div>
  </div>
</div>

<div class="grid grid-2">
  <div class="card">
    <div class="card-eyebrow">Coverage Breakdown</div>
    <div class="card-title">Module Coverage</div>
    <div class="progress" style="margin: var(--space-4) 0">
      <div class="progress-fill" style="width:{cov["pct"]}%; background:{_color(cov["pct"])}"></div>
    </div>
    <p style="color: var(--ink-muted); font-size: var(--text-sm)">
      <strong>{cov["documented"]}</strong> of <strong>{cov["total_modules"]}</strong> modules documented
    </p>
    <div style="margin-top: var(--space-4); display:flex; align-items:center; gap:var(--space-3)">
      <span class="badge {_badge(cov["pct"])}">{cov["pct"]}% Covered</span>
      {"" if not cov["missing"] else f'<button class="btn btn--sm btn--primary" id="fix-all-btn" onclick="fixAll()">Fix All ({len(cov["missing"])})</button>'}
    </div>
    <div id="fix-all-progress" style="display:none; margin-top:var(--space-4)">
      <div style="display:flex; justify-content:space-between; font-family:var(--font-display); font-size:var(--text-xs); margin-bottom:var(--space-2)">
        <span id="fp-status">Queuing...</span>
        <span id="fp-time"></span>
      </div>
      <div class="progress">
        <div class="progress-fill" id="fp-bar" style="width:0%; background:var(--primary); transition:width 300ms ease"></div>
      </div>
      <div style="margin-top:var(--space-2); font-size:var(--text-xs); color:var(--ink-muted)">
        <span id="fp-counts">0 / 0</span> &mdash; <span id="fp-eta"></span>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-eyebrow">Quick Stats</div>
    <div class="card-title">System Health</div>
    <table>
      <tr><td style="border:none; font-weight:600">Doc Pages</td><td style="border:none">{cov["pages"]}</td></tr>
      <tr><td style="border:none; font-weight:600">Source Modules</td><td style="border:none">{cov["total_modules"]}</td></tr>
      <tr><td style="border:none; font-weight:600">Quality Score</td><td style="border:none">{cov["quality_score"]}/100</td></tr>
      <tr><td style="border:none; font-weight:600">Last Scan</td><td style="border:none">{datetime.now().strftime('%H:%M:%S')}</td></tr>
    </table>
  </div>
</div>

{"" if not cov["missing"] else f'''
<div class="callout callout--warn">
  <strong>{len(cov["missing"])} modules</strong> have no documentation.
  <a href="/coverage">View full coverage report -></a>
</div>
'''}

{"" if not cov["stale"] else f'''
<div class="callout callout--error">
  <strong>{len(cov["stale"])} docs</strong> may be outdated — source changed after last doc edit.
  <a href="/health">Review health -></a>
</div>
'''}
<script>
async function fixAll() {{
  const btn = document.getElementById('fix-all-btn');
  const prog = document.getElementById('fix-all-progress');
  btn.disabled = true;
  btn.innerHTML = '<span class="job-spinner"></span> Queuing...';
  prog.style.display = 'block';

  try {{
    // Get all missing file paths from API
    const covResp = await fetch('/api/coverage');
    const covData = await covResp.json();
    const paths = (covData.missing || []).map(m => m.path);

    if (!paths.length) {{
      btn.innerHTML = 'Nothing to fix';
      return;
    }}

    // Batch generate
    const resp = await fetch('/api/batch-generate', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{file_paths: paths}})
    }});
    const data = await resp.json();

    if (!data.ok) {{
      document.getElementById('fp-status').textContent = 'Error: ' + (data.error || 'Failed');
      btn.innerHTML = 'Fix All'; btn.disabled = false;
      return;
    }}

    btn.innerHTML = '\\u2713 ' + data.count + ' Queued';
    const totalJobs = data.count;
    const startTime = Date.now();

    // Poll progress
    function poll() {{
      fetch('/api/jobs').then(r => r.json()).then(jobs => {{
        const batchJobs = jobs.slice(0, totalJobs);
        const done = batchJobs.filter(j => j.status === 'done').length;
        const errors = batchJobs.filter(j => j.status === 'error').length;
        const running = batchJobs.filter(j => j.status === 'running').length;
        const completed = done + errors;
        const pct = Math.round(completed / totalJobs * 100);

        document.getElementById('fp-bar').style.width = pct + '%';
        document.getElementById('fp-counts').textContent = completed + ' / ' + totalJobs;

        if (errors > 0) {{
          document.getElementById('fp-status').textContent = done + ' done, ' + errors + ' errors, ' + running + ' running';
        }} else {{
          document.getElementById('fp-status').textContent = done + ' done, ' + running + ' running';
        }}

        // ETA calculation
        const elapsed = (Date.now() - startTime) / 1000;
        if (completed > 0 && completed < totalJobs) {{
          const perJob = elapsed / completed;
          const remaining = (totalJobs - completed) * perJob;
          const mins = Math.floor(remaining / 60);
          const secs = Math.round(remaining % 60);
          document.getElementById('fp-eta').textContent = '~' + (mins > 0 ? mins + 'm ' : '') + secs + 's remaining';
        }} else if (completed >= totalJobs) {{
          const mins = Math.floor(elapsed / 60);
          const secs = Math.round(elapsed % 60);
          document.getElementById('fp-eta').textContent = 'Done in ' + (mins > 0 ? mins + 'm ' : '') + secs + 's';
          document.getElementById('fp-bar').style.background = 'var(--success)';
          document.getElementById('fp-status').textContent = done + ' done' + (errors > 0 ? ', ' + errors + ' errors' : '') + ' \\u2014 complete!';
          btn.innerHTML = '\\u2713 Complete';
          return;
        }}

        // Elapsed time display
        const eMins = Math.floor(elapsed / 60);
        const eSecs = Math.round(elapsed % 60);
        document.getElementById('fp-time').textContent = (eMins > 0 ? eMins + 'm ' : '') + eSecs + 's elapsed';

        setTimeout(poll, 2000);
      }}).catch(() => setTimeout(poll, 5000));
    }}
    setTimeout(poll, 1500);

  }} catch(e) {{
    document.getElementById('fp-status').textContent = 'Network error';
    btn.innerHTML = 'Fix All'; btn.disabled = false;
  }}
}}
</script>

<div style="margin-top:var(--space-6); padding-top:var(--space-6); border-top:2px solid var(--ink-faint); display:flex; justify-content:flex-end">
  <button class="btn btn--sm btn--danger" onclick="stopServer()">Stop Server</button>
</div>

<script>
function stopServer() {{
  if (!confirm('Stop the doc server? This page will close.')) return;
  fetch('/api/shutdown', {{method:'POST'}}).then(()=>{{
    document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;font-family:monospace;font-size:24px;color:#555">Server stopped.</div>';
    setTimeout(()=>window.close(), 1500);
  }}).catch(()=>{{
    document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;font-family:monospace;font-size:24px;color:#555">Server stopped.</div>';
  }});
}}
</script>
"""
    return _render(content, "Dashboard", "dashboard")


# ═════════════════════════════════════════════════════════
# ROUTES — BROWSE
# ═════════════════════════════════════════════════════════


@web_app.get("/browse")
async def browse(q: str = ""):
    docs_sys = _get_docs()
    pages = _get_doc_pages()
    if q:
        q_lower = q.lower()
        pages = [d for d in pages if q_lower in d["name"].lower() or q_lower in d["rel_path"].lower()]

    code_names = _get_meaningful_code_names(docs_sys)

    rows = ""
    for d in pages:
        refs = _count_code_refs_for_file(docs_sys, d["full_path"], code_names)

        if refs == 0 and d["word_count"] < 20:
            badge = '<span class="badge badge--error">Empty</span>'
        elif refs == 0:
            badge = '<span class="badge badge--warning">No refs</span>'
        else:
            badge = f'<span class="badge badge--success">{refs} refs</span>'

        rows += f"""<tr>
          <td><a href="/view?p={d["rel_path"]}">{d["rel_path"]}</a></td>
          <td>{d["word_count"]}</td>
          <td>{d["headings"]}</td>
          <td>{d["mtime"]}</td>
          <td>{badge}</td>
        </tr>"""

    if not rows:
        rows = '<tr><td colspan="5" style="text-align:center; padding: var(--space-5)">No docs found.</td></tr>'

    content = f"""
<h1 class="page-title">Browse Documentation</h1>
<p class="page-subtitle">All markdown pages in the docs directory. Click a page to read it.</p>

<div style="margin-bottom: var(--space-6)">
  <form method="get" action="/browse" style="display:flex; gap: var(--space-3); align-items:end">
    <div class="search-box" style="flex:1">
      <input type="text" name="q" value="{q}" placeholder="Search pages...">
    </div>
    <button class="btn btn--sm" type="submit">Search</button>
  </form>
</div>

<div class="card card--flat">
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>Page</th><th>Words</th><th>Sections</th><th>Modified</th><th>Code Refs</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>
"""
    return _render(content, "Browse", "browse")


# ═════════════════════════════════════════════════════════
# ROUTES — VIEW PAGE
# ═════════════════════════════════════════════════════════


@web_app.get("/view")
async def view_page(p: str = ""):
    path = unquote(p)
    if not path:
        logger.warning("view_page called without ?p= parameter")
        return (400, _render("""
<h1 class="page-title">Missing Path</h1>
<div class="callout callout--error">No page specified. Use <code>/view?p=path/to/file.md</code></div>
<a href="/browse" class="btn btn--ghost">← Back to Browse</a>
""", "Error", "browse"))

    logger.info(f"view_page: requested path={path}")

    # Sanitize: resolve and ensure path stays within DOCS_DIR
    try:
        full = (DOCS_DIR / path).resolve()
    except (OSError, ValueError):
        full = None

    if full is None or not str(full).startswith(str(DOCS_DIR.resolve())):
        logger.warning(f"view_page: path escapes DOCS_DIR — {path}")
        return (403, _render(f"""
    <h1 class="page-title">Access Denied</h1>
    <div class="callout callout--error">
      Path <code>{path}</code> is outside the docs directory.
    </div>
    <a href="/browse" class="btn btn--ghost">← Back to Browse</a>
    """, "Forbidden", "browse"))

    if not full.exists():
        logger.warning(f"view_page: not found — {full}")
        return (404, _render(f"""
    <h1 class="page-title">Page Not Found</h1>
    <div class="callout callout--error">
      Could not find <code>{path}</code> in <code>{DOCS_DIR}</code>
    </div>
    <a href="/browse" class="btn btn--ghost">← Back to Browse</a>
    """, "Not Found", "browse"))

    raw = full.read_text(encoding="utf-8", errors="ignore")
    rendered = _render_md(raw)
    # Rewrite relative .md links to /view?p= links so doc-to-doc navigation works
    def _rewrite_md_link(m):
        link_path = m.group(1)
        # Resolve relative links against the current doc's directory
        try:
            resolved = (full.parent / link_path.split("#")[0]).resolve()
            if str(resolved).startswith(str(DOCS_DIR.resolve())):
                rel = str(resolved.relative_to(DOCS_DIR.resolve()))
                fragment = "#" + link_path.split("#")[1] if "#" in link_path else ""
                return f'href="/view?p={rel}{fragment}"'
        except (OSError, ValueError):
            pass
        # Link outside DOCS_DIR or broken — keep as dead link with warning
        return f'href="#" title="Link outside docs directory"'

    rendered = re.sub(r'href="([^"]*\.md(?:#[^"]*)?)"', _rewrite_md_link, rendered)

    breadcrumbs = " / ".join(
        f"<span>{p}</span>" for p in Path(path).parts
    )

    content = f"""
<div style="margin-bottom: var(--space-4)">
  <span style="font-family: var(--font-display); font-size: var(--text-xs);
    text-transform: uppercase; letter-spacing: 2px; color: var(--ink-muted)">
    {breadcrumbs}
  </span>
</div>

<div class="card card--flat">
  <div class="md-content">{rendered}</div>
</div>

<hr>
<div style="display:flex; gap: var(--space-3)">
  <a href="/edit?page={path}" class="btn btn--sm">Edit This Page</a>
  <a href="/browse" class="btn btn--sm btn--ghost">← Back</a>
</div>
"""
    return _render(content, path, "browse")


# ═════════════════════════════════════════════════════════
# ROUTES — SEARCH (unified docs + code)
# ═════════════════════════════════════════════════════════


@web_app.get("/search")
async def search_page(q: str = "", type: str = "all"):
    docs_sys = _get_docs()

    results_html = ""
    context_html = ""
    doc_count = 0
    code_count = 0
    context_files = set()

    if q:
        if type in ("all", "docs"):
            sections_result = await docs_sys.read(query=q, max_results=20)
            sections = sections_result.get("sections", [])
            doc_count = len(sections)
            if sections:
                results_html += '<div class="card-eyebrow" style="margin-top: var(--space-5)">Documentation Results</div>'
                for s in sections:
                    try:
                        rel = str(Path(s["file"]).relative_to(DOCS_DIR))
                    except (ValueError, KeyError):
                        rel = s.get("file", "")
                    snippet = s.get("content", "")[:200]
                    results_html += f"""
                    <div class="card" style="margin-bottom: var(--space-4)">
                      <div class="card-title" style="font-size: var(--text-base)">{s.get("title", "Untitled")}</div>
                      <div style="font-size: var(--text-xs); color: var(--ink-muted); margin-bottom: var(--space-2)">
                        <a href="/view?p={rel}">{rel}</a> · Level {s.get("level", 0)}
                      </div>
                      <p style="font-size: var(--text-sm); color: var(--ink-muted)">{snippet}...</p>
                    </div>"""

        if type in ("all", "code"):
            code_result = await docs_sys.lookup_code(name=q, max_results=20)
            elements = code_result.get("results", [])
            code_count = len(elements)
            if elements:
                # Collect files from top-ranked results only (exact/near matches)
                q_lower = q.lower()
                for elem in elements[:5]:
                    # Only include files where the element name closely matches the query
                    if q_lower in elem["name"].lower() or elem["name"].lower() in q_lower:
                        context_files.add(elem["file"])
                # Fallback: if no close match, take top 2 result files
                if not context_files:
                    for elem in elements[:2]:
                        context_files.add(elem["file"])

                results_html += '<div class="card-eyebrow" style="margin-top: var(--space-5)">Code Results</div>'
                for elem in elements:
                    type_badge = "badge--success" if elem["type"] == "class" else "badge--ghost"
                    doc_info = f'<code style="font-size:var(--text-xs)">{elem["docstring"][:100]}...</code>' if elem.get("docstring") else '<span style="color:var(--ink-faint)">No docstring</span>'
                    results_html += f"""
                    <div class="card" style="margin-bottom: var(--space-4)">
                      <div style="display:flex; align-items:center; gap: var(--space-3); margin-bottom: var(--space-2)">
                        <span class="badge {type_badge}">{elem["type"]}</span>
                        <code style="font-size: var(--text-base); font-weight: 600">{elem["signature"]}</code>
                      </div>
                      <div style="font-size: var(--text-xs); color: var(--ink-muted)">
                        {elem["file"]} · L{elem["lines"][0]}-{elem["lines"][1]} · {elem.get("language", "python")}
                      </div>
                      <div style="margin-top: var(--space-2)">{doc_info}</div>
                    </div>"""

        # Build context overview from found files
        if context_files:
            try:
                ctx = await docs_sys.get_task_context(
                    files=list(context_files)[:10],
                    intent=q,
                )
                graph = ctx.get("result", {}).get("context_graph", {})
                upstream = graph.get("upstream_dependencies", [])
                downstream = graph.get("downstream_usages", [])
                relevant_docs = ctx.get("result", {}).get("relevant_docs", [])

                ctx_parts = []

                if upstream:
                    ctx_parts.append('<div style="margin-bottom:var(--space-4)">')
                    ctx_parts.append('<div style="font-family:var(--font-display); font-size:var(--text-xs); text-transform:uppercase; letter-spacing:1px; color:var(--ink-muted); margin-bottom:var(--space-2)">Depends On</div>')
                    for u in upstream[:8]:
                        name = u.get("name", "?")
                        sig = u.get("signature", "")
                        imp = u.get("import_statement", "")
                        usage = u.get("usage_snippet", "")
                        ufile = u.get("file", "")
                        ctx_parts.append(f'<div style="padding:var(--space-2) 0; border-bottom:1px solid var(--paper-sunken)">')
                        ufile = u.get("file", "")
                        try:
                            ufile_rel = str(Path(ufile).relative_to(PROJECT_ROOT)).replace("\\", "/")
                        except (ValueError, TypeError):
                            ufile_rel = ufile
                        ctx_parts.append(
                            f'<code style="font-weight:600">{name}</code> <span style="color:var(--ink-muted); font-size:var(--text-xs)">({u.get("type", "?")}) in <code>{ufile_rel}</code></span>')
                        if sig:
                            ctx_parts.append(f'<div style="font-size:var(--text-xs); color:var(--ink-muted)"><code>{sig}</code></div>')
                        if imp:
                            ctx_parts.append(f'<div style="font-size:var(--text-xs); color:var(--ink-faint)"><code>{imp}</code></div>')
                        if usage:
                            ctx_parts.append(f'<div style="font-size:var(--text-xs); color:var(--ink-faint)">Usage: <code>{usage}</code></div>')
                        ctx_parts.append('</div>')
                    ctx_parts.append('</div>')

                if downstream:
                    ctx_parts.append('<div style="margin-bottom:var(--space-4)">')
                    ctx_parts.append('<div style="font-family:var(--font-display); font-size:var(--text-xs); text-transform:uppercase; letter-spacing:1px; color:var(--ink-muted); margin-bottom:var(--space-2)">Used By</div>')
                    for d in downstream[:8]:
                        imp_name = d.get("imported_name", "?")
                        dfile = d.get("file", "")
                        imp_stmt = d.get("import_statement", "")
                        snippets = d.get("usage_snippets", [])
                        ctx_parts.append(f'<div style="padding:var(--space-2) 0; border-bottom:1px solid var(--paper-sunken)">')
                        try:
                            dfile_rel = str(Path(dfile).relative_to(PROJECT_ROOT)).replace("\\", "/")
                        except (ValueError, TypeError):
                            dfile_rel = dfile
                        ctx_parts.append(
                            f'<code style="font-weight:600">{imp_name}</code> <span style="color:var(--ink-muted); font-size:var(--text-xs)">in <code>{dfile_rel}</code></span>')
                        if imp_stmt:
                            ctx_parts.append(f'<div style="font-size:var(--text-xs); color:var(--ink-faint)"><code>{imp_stmt}</code></div>')
                        for sn in snippets[:2]:
                            ctx_parts.append(f'<div style="font-size:var(--text-xs); color:var(--ink-faint)">→ <code>{sn}</code></div>')
                        ctx_parts.append('</div>')
                    ctx_parts.append('</div>')

                if relevant_docs:
                    ctx_parts.append('<div>')
                    ctx_parts.append('<div style="font-family:var(--font-display); font-size:var(--text-xs); text-transform:uppercase; letter-spacing:1px; color:var(--ink-muted); margin-bottom:var(--space-2)">Related Docs</div>')
                    for rd in relevant_docs[:3]:
                        title = rd.get("title", "")
                        rfile = rd.get("file", "")
                        try:
                            rd_rel = str(Path(rfile).relative_to(DOCS_DIR)).replace("\\", "/")
                        except (ValueError, TypeError):
                            try:
                                rd_rel = str(Path(rfile).relative_to(PROJECT_ROOT)).replace("\\", "/")
                            except (ValueError, TypeError):
                                rd_rel = rfile
                        # Only link if within DOCS_DIR
                        docs_resolved = DOCS_DIR.resolve()
                        try:
                            is_in_docs = str(Path(rfile).resolve()).startswith(str(docs_resolved))
                        except (OSError, ValueError):
                            is_in_docs = False
                        if is_in_docs:
                            ctx_parts.append(
                                f'<div style="font-size:var(--text-sm)"><a href="/view?p={rd_rel}">{title}</a></div>')
                        else:
                            ctx_parts.append(
                                f'<div style="font-size:var(--text-sm)">{title} <span style="color:var(--ink-faint); font-size:var(--text-xs)">({rd_rel})</span></div>')
                    ctx_parts.append('</div>')

                if ctx_parts:
                    files_list = ", ".join(f"<code>{Path(f).stem}</code>" for f in list(context_files)[:5])
                    if len(context_files) > 5:
                        files_list += f" +{len(context_files) - 5} more"
                    context_html = f"""
                    <div class="card" style="margin-bottom:var(--space-5); border-left:6px solid var(--primary)">
                      <div class="card-eyebrow">Context Overview</div>
                      <div class="card-title" style="font-size:var(--text-base)">How these modules connect</div>
                      <div style="font-size:var(--text-xs); color:var(--ink-muted); margin-bottom:var(--space-3)">
                        Based on: {files_list}
                      </div>
                      {"".join(ctx_parts)}
                    </div>"""

            except Exception as e:
                logger.warning(f"search: context generation failed — {e}")

    if q and not results_html and not context_html:
        results_html = f'<div class="callout">No results found for "<strong>{q}</strong>".</div>'

    content = f"""
<h1 class="page-title">Search</h1>
<p class="page-subtitle">Search documentation and code from the mkdocs index.</p>

<div style="margin-bottom: var(--space-6)">
  <form method="get" action="/search" style="display:flex; gap: var(--space-3); align-items:end; flex-wrap: wrap">
    <div class="search-box" style="flex:1; min-width: 200px">
      <input type="text" name="q" value="{q}" placeholder="Search docs & code...">
    </div>
    <select name="type" style="width: auto; min-width: 120px">
      <option value="all" {"selected" if type == "all" else ""}>All</option>
      <option value="docs" {"selected" if type == "docs" else ""}>Docs Only</option>
      <option value="code" {"selected" if type == "code" else ""}>Code Only</option>
    </select>
    <button class="btn btn--sm" type="submit">Search</button>
  </form>
</div>

{"" if not q else f'''
<div style="margin-bottom: var(--space-4); font-size: var(--text-sm); color: var(--ink-muted)">
  Found <strong>{doc_count}</strong> doc sections and <strong>{code_count}</strong> code elements
</div>
'''}

{context_html}

{results_html}
"""
    return _render(content, "Search", "search")

# ═════════════════════════════════════════════════════════
# ROUTES — HEALTH (per-directory breakdown + orphans)
# ═════════════════════════════════════════════════════════


@web_app.get("/health")
async def health_page():
    h = _compute_health()

    dir_rows = ""
    for dir_path, stats in h["by_dir"].items():
        pct = int(stats["documented"] / max(stats["total"], 1) * 100)
        badge_cls = "badge--success" if pct >= 70 else ("badge--warning" if pct >= 40 else "badge--error")
        dir_rows += f"""<tr>
          <td><code>{dir_path}</code></td>
          <td>{stats["classes"]}</td>
          <td>{stats["functions"]}</td>
          <td>{stats["methods"]}</td>
          <td>{stats["total"]}</td>
          <td>{stats["documented"]}</td>
          <td><span class="badge {badge_cls}">{pct}%</span></td>
        </tr>"""

    if not dir_rows:
        dir_rows = '<tr><td colspan="7" style="text-align:center; padding: var(--space-5)">No code elements indexed.</td></tr>'

    orphan_rows = ""
    for o in h["orphan_docs"]:
        orphan_rows += f"""<tr>
          <td><a href="/view?p={o["rel_path"]}">{o["rel_path"]}</a></td>
          <td>{o["title"]}</td>
          <td>{o["words"]}</td>
          <td><span class="badge badge--warning">Orphan</span></td>
        </tr>"""

    if not orphan_rows:
        orphan_rows = '<tr><td colspan="4" style="text-align:center; padding: var(--space-5)">No orphan docs found.</td></tr>'

    content = f"""
<h1 class="page-title">Documentation Health</h1>
<p class="page-subtitle">
  Per-directory breakdown of code elements and their documentation status.
  Orphan docs reference code that no longer exists.
</p>

<div class="grid grid-4" style="margin-bottom: var(--space-6)">
  <div class="card">
    <div class="stat">
      <div class="stat-value">{h["total_elements"]}</div>
      <div class="stat-label">Total Elements</div>
    </div>
  </div>
  <div class="card">
    <div class="stat">
      <div class="stat-value" style="color: var(--success)">{h["total_documented"]}</div>
      <div class="stat-label">Documented</div>
    </div>
  </div>
  <div class="card">
    <div class="stat">
      <div class="stat-value" style="color: var(--error)">{h["total_undocumented"]}</div>
      <div class="stat-label">Undocumented</div>
    </div>
  </div>
  <div class="card">
    <div class="stat">
      <div class="stat-value" style="color: var(--warning)">{len(h["orphan_docs"])}</div>
      <div class="stat-label">Orphan Docs</div>
    </div>
  </div>
</div>

<div class="card card--flat">
  <div class="card-eyebrow">Per-Directory Breakdown</div>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>Directory</th><th>Classes</th><th>Functions</th><th>Methods</th><th>Total</th><th>Documented</th><th>Coverage</th>
      </tr></thead>
      <tbody>{dir_rows}</tbody>
    </table>
  </div>
</div>

<div class="card card--flat" style="margin-top: var(--space-6)">
  <div class="card-eyebrow">Orphan Documentation</div>
  <p style="font-size: var(--text-sm); color: var(--ink-muted); margin-bottom: var(--space-4)">
    Documentation pages that don't match any existing code file or element.
  </p>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>Doc Page</th><th>Title</th><th>Words</th><th>Status</th>
      </tr></thead>
      <tbody>{orphan_rows}</tbody>
    </table>
  </div>
</div>
"""
    return _render(content, "Health", "health")

# ═════════════════════════════════════════════════════════
# ROUTES — INVENTORY (DocMap: What's Here)
# ═════════════════════════════════════════════════════════


@web_app.get("/inventory")
async def inventory_page(focus: str = ""):
    docs_sys = _get_docs()
    result = await docs_sys.generate_inventory(
        focus_dirs=focus.split(",") if focus else None,
        max_classes_per_file=5,
        max_methods_per_class=3,
        format_type="structured",
    )
    files = result.get("files", [])

    rows = ""
    for f in files:
        cls_details = ""
        for cls in f["detailed_classes"]:
            method_rows = ""
            for m in cls["top_methods"]:
                doc = f'<span style="color:var(--ink-muted)"> — {m["docstring"][:80]}</span>' if m.get("docstring") else ""
                method_rows += f"<tr><td><code>{m['name']}</code></td><td><code>{m['signature']}</code>{doc}</td></tr>"

            methods_table = ""
            if method_rows:
                methods_table = (
                    '<table style="margin-top:var(--space-2)">'
                    "<thead><tr><th>Method</th><th>Signature</th></tr></thead>"
                    f"<tbody>{method_rows}</tbody></table>"
                )

            docstr = f'<blockquote style="margin:var(--space-2) 0; padding:var(--space-2) var(--space-3); border-left:3px solid var(--ink-faint); color:var(--ink-muted); font-size:var(--text-sm)">{cls["docstring"][:150]}</blockquote>' if cls.get("docstring") else ""

            cls_details += (
                f'<div style="margin:var(--space-3) 0; padding:var(--space-3); background:var(--paper-sunken); border-left:4px solid var(--ink)">'
                f'<code style="font-weight:600">{cls["signature"]}</code>'
                f'{docstr}'
                f'<div style="font-size:var(--text-xs); color:var(--ink-muted)">{cls["method_count"]} methods total</div>'
                f'{methods_table}'
                f'</div>'
            )

        others = ""
        if f["other_classes"]:
            names = ", ".join(f"<code>{n}</code>" for n in f["other_classes"])
            others = f'<p style="margin-top:var(--space-2)"><strong>Also:</strong> {names}</p>'

        fns = ""
        if f["functions"]:
            fn_items = "".join(f"<li><code>{fn['signature']}</code></li>" for fn in f["functions"])
            fns = f'<div style="margin-top:var(--space-2)"><strong>Functions:</strong><ul style="padding-left:var(--space-5)">{fn_items}</ul></div>'

        rows += f"""
        <details class="card" style="margin-bottom:var(--space-4)">
          <summary style="cursor:pointer; font-family:var(--font-display); font-size:var(--text-sm); font-weight:500; list-style:none; display:flex; align-items:center; gap:var(--space-3)">
            <span class="badge">{f["language"]}</span>
            <code>{f["path"]}</code>
          </summary>
          <div style="padding-top:var(--space-3); border-top:1px solid var(--ink); margin-top:var(--space-3)">
            {cls_details}{others}{fns}
          </div>
        </details>"""

    if not rows:
        rows = '<div class="callout">No code elements found in the index.</div>'

    content = f"""
<h1 class="page-title">Project Inventory</h1>
<p class="page-subtitle">What's here — all files, classes, and functions grouped by module. Top classes ranked by usage.</p>

<div style="margin-bottom:var(--space-6)">
  <form method="get" action="/inventory" style="display:flex; gap:var(--space-3); align-items:end">
    <div style="flex:1">
      <label>Focus Directories (comma-separated, empty = all)</label>
      <input type="text" name="focus" value="{focus}" placeholder="e.g. toolboxv2/utils,toolboxv2/mods">
    </div>
    <button class="btn btn--sm" type="submit">Filter</button>
  </form>
</div>

<div style="margin-bottom:var(--space-4); font-size:var(--text-sm); color:var(--ink-muted)">
  {result["file_count"]} files | Generated in {result["time_ms"]:.0f}ms
</div>

{rows}
"""
    return _render(content, "Inventory", "inventory")


# ═════════════════════════════════════════════════════════
# ROUTES — RELATIONSHIP MAP (DocMap: How Components Connect)
# ═════════════════════════════════════════════════════════


@web_app.get("/relationships")
async def relationships_page(focus: str = "", classes: str = "", nodes: str = "40"):
    docs_sys = _get_docs()
    max_n = min(int(nodes) if nodes.isdigit() else 40, 80)

    result = await docs_sys.generate_relationship_map(
        focus_dirs=focus.split(",") if focus else None,
        focus_classes=classes.split(",") if classes else None,
        max_nodes=max_n,
        format_type="structured",
    )

    mermaid_code = result.get("mermaid", "graph LR\n    A[No relationships found]")
    escaped = mermaid_code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    edges = result.get("edges", [])
    edge_rows = ""
    for e in edges:
        type_badge = {"inherits": "badge--success", "uses": "badge--warning", "imports": "badge--ghost"}.get(e["type"], "badge--ghost")
        edge_rows += f'<tr><td><code>{e["from"]}</code></td><td><span class="badge {type_badge}">{e["type"]}</span></td><td><code>{e["to"]}</code></td></tr>'

    if not edge_rows:
        edge_rows = '<tr><td colspan="3" style="text-align:center; padding:var(--space-5)">No relationships found.</td></tr>'

    content = f"""
<h1 class="page-title">Relationship Map</h1>
<p class="page-subtitle">How components connect — inheritance, composition, and cross-file references as a Mermaid diagram.</p>

<div style="margin-bottom:var(--space-6)">
  <form method="get" action="/relationships" style="display:flex; gap:var(--space-3); align-items:end; flex-wrap:wrap">
    <div style="flex:1; min-width:200px">
      <label>Focus Directories</label>
      <input type="text" name="focus" value="{focus}" placeholder="e.g. toolboxv2/utils">
    </div>
    <div style="flex:1; min-width:200px">
      <label>Focus Classes</label>
      <input type="text" name="classes" value="{classes}" placeholder="e.g. DocsSystem,IndexManager">
    </div>
    <div style="width:100px">
      <label>Max Nodes</label>
      <input type="text" name="nodes" value="{max_n}" placeholder="40">
    </div>
    <button class="btn btn--sm" type="submit">Generate</button>
  </form>
</div>

<div style="margin-bottom:var(--space-4); font-size:var(--text-sm); color:var(--ink-muted)">
  {result["node_count"]} nodes | {result["edge_count"]} edges | Generated in {result["time_ms"]:.0f}ms
</div>

<div class="mermaid-container">
  <pre class="mermaid">{escaped}</pre>
</div>

<div class="card card--flat" style="margin-top:var(--space-6)">
  <div class="card-eyebrow">Edge List</div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>From</th><th>Type</th><th>To</th></tr></thead>
      <tbody>{edge_rows}</tbody>
    </table>
  </div>
</div>

<div class="callout" style="margin-top:var(--space-5)">
  <strong>Tips:</strong> Use "Focus Classes" to zoom into one class and see all its connections.
  For large projects, set Max Nodes to 20–30 and use Focus Directories to scope per subsystem.
</div>
"""
    return _render(content, "Relationships", "relationships")

# ═════════════════════════════════════════════════════════
# ROUTES — COVERAGE
# ═════════════════════════════════════════════════════════


@web_app.get("/coverage")
async def coverage_page():
    cov = _compute_coverage()

    # Build nested tree: dir -> subdir -> ... -> files
    tree: dict = {}
    for m in cov["missing"]:
        parts = Path(m["rel_path"]).parts
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = {"__file__": m}

    def _count_files(n: dict) -> int:
        c = 0
        for v in n.values():
            if isinstance(v, dict):
                if "__file__" in v:
                    c += 1
                else:
                    c += _count_files(v)
        return c

    def _render_tree(node: dict, path_prefix: str = "", depth: int = 0) -> str:
        html = ""
        dirs = {k: v for k, v in sorted(node.items()) if isinstance(v, dict) and "__file__" not in v}
        files = {k: v["__file__"] for k, v in sorted(node.items()) if isinstance(v, dict) and "__file__" in v}

        for dir_name, children in dirs.items():
            dir_path = f"{path_prefix}/{dir_name}" if path_prefix else dir_name
            count = _count_files(children)
            html += f'''
            <details class="tree-dir" {"open" if depth < 1 else ""}>
              <summary style="cursor:pointer; display:flex; align-items:center; gap:var(--space-2);
                padding:var(--space-2) var(--space-3); font-family:var(--font-display);
                font-size:var(--text-sm); font-weight:500; list-style:none;
                border-bottom:1px solid var(--ink-faint)">
                <input type="checkbox" class="dir-check" data-dir="{dir_path}"
                  onchange="toggleDir(this)" style="margin:0">
                <span style="color:var(--ink-muted)">&#128193;</span>
                <code>{dir_name}/</code>
                <span class="badge badge--ghost" style="margin-left:auto">{count}</span>
              </summary>
              <div style="padding-left:var(--space-5)">
                {_render_tree(children, dir_path, depth + 1)}
              </div>
            </details>'''

        for fname, m in files.items():
            escaped = m["rel_path"].replace("\\", "/").replace("'", "\\'")
            prio_cls = "badge--error" if m["classes"] > 0 or m["functions"] > 5 else "badge--warning"
            prio_lbl = "High" if m["classes"] > 0 or m["functions"] > 5 else "Med"
            html += f'''
            <div class="tree-file" style="display:flex; align-items:center; gap:var(--space-2);
              padding:var(--space-2) var(--space-3); border-bottom:1px solid var(--paper-sunken)">
              <input type="checkbox" class="file-check" value="{escaped}" style="margin:0">
              <code style="flex:1; font-size:var(--text-sm)">{m["name"]}</code>
              <span style="font-size:var(--text-xs); color:var(--ink-muted)">{m["elements"]}el</span>
              <span style="font-size:var(--text-xs); color:var(--ink-muted)">{m["classes"]}C {m["functions"]}F</span>
              <span class="badge {prio_cls}" style="font-size:10px">{prio_lbl}</span>
              <button class="btn btn--sm" style="padding:2px 8px; font-size:10px"
                onclick="generateDoc('{escaped}', this)">Gen</button>
            </div>'''

        return html

    tree_html = _render_tree(tree)
    if not tree_html:
        tree_html = '<div class="callout callout--ok">All modules documented!</div>'

    content = f"""
<h1 class="page-title">Coverage Report</h1>
<p class="page-subtitle">
  Undocumented modules as tree. Check files or directories, then batch-generate (3 concurrent).
</p>

<div class="grid grid-3" style="margin-bottom: var(--space-6)">
  <div class="card">
    <div class="stat">
      <div class="stat-value" style="color: var(--success)">{cov["documented"]}</div>
      <div class="stat-label">Documented</div>
    </div>
  </div>
  <div class="card">
    <div class="stat">
      <div class="stat-value" style="color: var(--error)">{len(cov["missing"])}</div>
      <div class="stat-label">Missing</div>
    </div>
  </div>
  <div class="card">
    <div class="stat">
      <div class="stat-value">{cov["total_modules"]}</div>
      <div class="stat-label">Total</div>
    </div>
  </div>
</div>

<div class="card card--flat">
  <div style="display:flex; align-items:center; gap:var(--space-3); padding:var(--space-3);
    border-bottom:2px solid var(--ink); background:var(--paper-sunken)">
    <input type="checkbox" id="select-all" onchange="toggleAll(this)" style="margin:0">
    <label for="select-all" style="margin:0; text-transform:none; letter-spacing:0; font-size:var(--text-sm)">Select All</label>
    <div style="margin-left:auto; display:flex; gap:var(--space-3); align-items:center">
      <span id="sel-count" style="font-family:var(--font-display); font-size:var(--text-sm); color:var(--ink-muted)">0 selected</span>
      <button class="btn btn--sm btn--primary" id="batch-btn" onclick="batchGenerate()" disabled>
        Batch Generate
      </button>
    </div>
  </div>
  <div id="tree-root">{tree_html}</div>
</div>

<div id="batch-result" style="margin-top:var(--space-4)"></div>

<script>
function updateCount() {{
  const n = document.querySelectorAll('.file-check:checked').length;
  document.getElementById('sel-count').textContent = n + ' selected';
  document.getElementById('batch-btn').disabled = n === 0;
}}
function toggleDir(cb) {{
  const det = cb.closest('details');
  det.querySelectorAll('.file-check').forEach(c => c.checked = cb.checked);
  det.querySelectorAll('.dir-check').forEach(c => {{ if (c !== cb) c.checked = cb.checked; }});
  updateCount();
}}
function toggleAll(cb) {{
  document.querySelectorAll('.file-check, .dir-check').forEach(c => c.checked = cb.checked);
  updateCount();
}}
document.addEventListener('change', function(e) {{
  if (e.target.classList.contains('file-check')) updateCount();
}});
async function batchGenerate() {{
  const paths = Array.from(document.querySelectorAll('.file-check:checked')).map(c => c.value);
  if (!paths.length) return;
  const btn = document.getElementById('batch-btn');
  const res = document.getElementById('batch-result');
  btn.disabled = true;
  btn.innerHTML = '<span class="job-spinner"></span> ' + paths.length + '...';
  try {{
    const r = await fetch('/api/batch-generate', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{file_paths: paths}})
    }});
    const d = await r.json();
    if (d.ok) {{
      res.innerHTML = '<div class="callout callout--ok"><strong>' + d.count + ' jobs queued</strong> (3 concurrent). Progress in nav badge.</div>';
      btn.innerHTML = '&#10003; ' + d.count + ' Queued';
    }} else {{
      res.innerHTML = '<div class="callout callout--error">' + (d.error||'Failed') + '</div>';
      btn.innerHTML = 'Batch Generate'; btn.disabled = false;
    }}
  }} catch(e) {{
    res.innerHTML = '<div class="callout callout--error">Network error</div>';
    btn.innerHTML = 'Batch Generate'; btn.disabled = false;
  }}
}}
</script>
"""
    return _render(content, "Coverage", "coverage")


# ═════════════════════════════════════════════════════════
# ROUTES — JOBS (doc generation queue + status)
# ═════════════════════════════════════════════════════════


@web_app.get("/jobs")
async def jobs_page():
    all_jobs = _jobs.list_all()

    rows = ""
    for j in all_jobs:
        status_badge = {
            "queued": "badge--ghost",
            "running": "badge--warning",
            "done": "badge--success",
            "error": "badge--error",
        }.get(j["status"], "badge--ghost")

        created = datetime.fromtimestamp(j["created"]).strftime("%H:%M:%S")
        duration = ""
        if j["started"] and j["finished"]:
            duration = f'{j["finished"] - j["started"]:.1f}s'
        elif j["started"]:
            duration = f'{time.time() - j["started"]:.0f}s...'

        output_link = ""
        if j["output_path"]:
            output_link = f'<a href="/view?p={j["output_path"]}" class="btn btn--sm btn--primary" style="margin-left:var(--space-2)">View Doc</a>'

        error_info = ""
        if j["error"]:
            error_info = f'<div style="font-size:var(--text-xs); color:var(--error); margin-top:var(--space-1)">{j["error"][:100]}</div>'

        rows += f"""<tr>
          <td><code style="font-size:var(--text-xs)">{j["id"]}</code></td>
          <td class="path-cell" title="{j["file_path"]}">{j["file_path"]}</td>
          <td><span class="badge {status_badge}">{j["status"]}</span></td>
          <td>{created}</td>
          <td>{duration}</td>
          <td>{output_link}{error_info}</td>
        </tr>"""

    if not rows:
        rows = '<tr><td colspan="6" style="text-align:center; padding: var(--space-5)">No jobs yet. Generate docs from the Coverage page.</td></tr>'

    running = sum(1 for j in all_jobs if j["status"] == "running")
    queued = sum(1 for j in all_jobs if j["status"] == "queued")
    done = sum(1 for j in all_jobs if j["status"] == "done")
    errors = sum(1 for j in all_jobs if j["status"] == "error")

    content = f"""
<h1 class="page-title">Doc Generation Jobs</h1>
<p class="page-subtitle">
  Background documentation generation tasks. Jobs run server-side and persist even when the browser is closed.
</p>

<div class="grid grid-4" style="margin-bottom: var(--space-6)">
  <div class="card">
    <div class="stat">
      <div class="stat-value">{running}</div>
      <div class="stat-label">Running</div>
    </div>
  </div>
  <div class="card">
    <div class="stat">
      <div class="stat-value">{queued}</div>
      <div class="stat-label">Queued</div>
    </div>
  </div>
  <div class="card">
    <div class="stat">
      <div class="stat-value" style="color: var(--success)">{done}</div>
      <div class="stat-label">Done</div>
    </div>
  </div>
  <div class="card">
    <div class="stat">
      <div class="stat-value" style="color: var(--error)">{errors}</div>
      <div class="stat-label">Errors</div>
    </div>
  </div>
</div>

<div style="margin-bottom: var(--space-5); display:flex; gap: var(--space-3)">
  <button class="btn btn--sm" onclick="location.reload()">Refresh</button>
  <script>{"if(" + str(running + queued) + ">0) setTimeout(()=>location.reload(), 5000);"}</script>
</div>

<div class="card card--flat">
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>Job ID</th><th>File</th><th>Status</th><th>Created</th><th>Duration</th><th>Result</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>
"""
    return _render(content, "Jobs", "jobs")


# ═════════════════════════════════════════════════════════
# ROUTES — EDITOR
# ═════════════════════════════════════════════════════════


@web_app.get("/edit")
async def edit_form(page: str = ""):
    current_path = page or "index.md"
    current_content = ""
    target = DOCS_DIR / current_path
    if target.exists():
        current_content = target.read_text(encoding="utf-8", errors="ignore")

    escaped = current_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    content = f"""
<h1 class="page-title">Page Editor</h1>
<p class="page-subtitle">Edit markdown documentation directly. Changes are saved to disk immediately.</p>

<div class="card card--flat">
  <div>
    <label>File Path (relative to docs/)</label>
    <input type="text" id="edit-path" value="{current_path}" style="margin-bottom: var(--space-5)">

    <label>Content (Markdown)</label>
    <textarea id="edit-content" rows="30" style="margin-bottom: var(--space-5)">{escaped}</textarea>

    <div style="display: flex; gap: var(--space-3)">
      <button class="btn btn--primary" onclick="savePage()">Save Page</button>
      <a href="/browse" class="btn btn--ghost">Cancel</a>
    </div>
    <div id="save-result" style="margin-top: var(--space-4)"></div>
  </div>
</div>

<script>
async function savePage() {{
  const path = document.getElementById('edit-path').value;
  const content = document.getElementById('edit-content').value;
  const resultDiv = document.getElementById('save-result');
  resultDiv.innerHTML = '<span class="job-spinner"></span> Saving...';
  try {{
    const resp = await fetch('/api/edit', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{path: path, content: content}})
    }});
    const data = await resp.json();
    if (data.ok) {{
      resultDiv.innerHTML = '<div class="callout callout--ok"><strong>' + path + '</strong> saved. <a href="/view?p=' + encodeURIComponent(path) + '">View Page -></a></div>';
    }} else {{
      resultDiv.innerHTML = '<div class="callout callout--error">' + (data.error || 'Failed') + '</div>';
    }}
  }} catch(e) {{
    resultDiv.innerHTML = '<div class="callout callout--error">Network error</div>';
  }}
}}
</script>
"""
    return _render(content, "Editor", "edit")


# ═════════════════════════════════════════════════════════
# API ROUTES — JSON
# ═════════════════════════════════════════════════════════


@web_app.get("/api/coverage")
async def api_coverage():
    cov = _compute_coverage()
    return {
        "total_modules": cov["total_modules"],
        "documented": cov["documented"],
        "missing_count": len(cov["missing"]),
        "stale_count": len(cov["stale"]),
        "pages": cov["pages"],
        "coverage_pct": cov["pct"],
        "quality_score": cov["quality_score"],
        "missing": [{"name": m["name"], "path": m["rel_path"]} for m in cov["missing"]],
        "stale": cov["stale"],
    }


@web_app.get("/api/pages")
async def api_pages():
    return _get_doc_pages()


@web_app.get("/api/modules")
async def api_modules():
    by_file = _get_code_elements()
    result = []
    for file_path, elems in sorted(by_file.items()):
        result.append({
            "name": Path(file_path).stem,
            "path": file_path,
            "elements": len(elems),
            "classes": sum(1 for e in elems if e.element_type == "class"),
            "functions": sum(1 for e in elems if e.element_type in ("function", "method")),
            "has_docstring": any(e.docstring for e in elems),
        })
    return result


@web_app.get("/api/search")
async def api_search(q: str = "", type: str = "all"):
    docs = _get_docs()
    results = {}
    if type in ("all", "docs"):
        sections = await docs.read(query=q, max_results=20)
        results["docs"] = sections.get("sections", [])
    if type in ("all", "code"):
        code = await docs.lookup_code(name=q, max_results=20)
        results["code"] = code.get("results", [])
    return results


@web_app.get("/api/suggestions")
async def api_suggestions():
    docs = _get_docs()
    return await docs.get_suggestions(max_suggestions=50)


@web_app.get("/api/health")
async def api_health():
    return _compute_health()

@web_app.get("/api/inventory")
async def api_inventory(focus: str = ""):
    docs = _get_docs()
    return await docs.generate_inventory(
        focus_dirs=focus.split(",") if focus else None,
        format_type="structured",
    )


@web_app.get("/api/relationships")
async def api_relationships(focus: str = "", classes: str = "", nodes: str = "40"):
    docs = _get_docs()
    max_n = min(int(nodes) if nodes.isdigit() else 40, 80)
    return await docs.generate_relationship_map(
        focus_dirs=focus.split(",") if focus else None,
        focus_classes=classes.split(",") if classes else None,
        max_nodes=max_n,
        format_type="structured",
    )

@web_app.post("/api/sync")
async def api_sync():
    logger.info("api_sync: starting index sync")
    docs = _get_docs()
    result = await docs.sync()
    logger.info(f"api_sync: done — {result}")
    return result


@web_app.post("/api/build")
async def api_build():
    if not MKDOCS_YML.exists():
        return (404, {"ok": False, "error": "mkdocs.yml not found"})
    try:
        proc = await asyncio.create_subprocess_exec(
            "mkdocs", "build", "--clean",
            cwd=str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        return {
            "ok": proc.returncode == 0,
            "stdout": stdout.decode(errors="ignore"),
            "stderr": stderr.decode(errors="ignore"),
        }
    except FileNotFoundError:
        return (500, {"ok": False, "error": "mkdocs binary not found"})
    except asyncio.TimeoutError:
        return (504, {"ok": False, "error": "build timed out (60s)"})


@web_app.post("/api/edit")
async def api_edit(request: ParsedRequest):
    data = request.json_data or {}
    path = data.get("path", "index.md")
    content = data.get("content", "")

    target = DOCS_DIR / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    logger.info(f"api_edit: saved {path} ({len(content)} chars)")

    try:
        docs = _get_docs()
        await docs._update_file(target)
        docs.context.clear_cache()
    except Exception as e:
        logger.warning(f"api_edit: re-index failed for {path} — {e}")

    return {"ok": True, "path": path, "saved_at": datetime.now().strftime("%H:%M:%S")}


@web_app.delete("/api/doc/{path}")
async def api_delete_doc(request: ParsedRequest):
    raw_path = request.path
    prefix = "/api/doc/"
    path = unquote(raw_path[len(prefix):]) if raw_path.startswith(prefix) else ""
    logger.info(f"api_delete_doc: path={path}")

    target = DOCS_DIR / path
    if not target.exists():
        logger.warning(f"api_delete_doc: not found — {target}")
        return (404, {"ok": False, "error": f"Not found: {path}"})
    target.unlink()
    logger.info(f"api_delete_doc: deleted {target}")
    try:
        _get_docs().index_mgr.remove_file(str(target))
        _get_docs().context.clear_cache()
    except Exception as e:
        logger.error(f"api_delete_doc: re-index failed — {e}")
    return {"ok": True, "deleted": path}


@web_app.get("/api/jobs")
async def api_jobs():
    return _jobs.list_all()


@web_app.get("/api/jobs/{job_id}")
async def api_job_status(request: ParsedRequest):
    raw_path = request.path
    prefix = "/api/jobs/"
    job_id = unquote(raw_path[len(prefix):]) if raw_path.startswith(prefix) else ""

    job = _jobs.get(job_id)
    if not job:
        return (404, {"error": f"Job not found: {job_id}"})
    return job


# ═════════════════════════════════════════════════════════
# DOC AGENT — Grounded documentation generation via ISAA
# ═════════════════════════════════════════════════════════

DOC_AGENT_PROMPT = """# ToolBoxV2 Documentation Agent

You are a precision documentation agent. You receive verified code data extracted from an AST-based index — signatures, docstrings, dependency graphs. This data is ground truth. You write documentation based ONLY on this data.

## ZERO-HALLUCINATION PROTOCOL

- Every class, function, parameter you mention MUST appear in the provided code data.
- If information is missing, OMIT it. Never guess, never interpolate, never invent.
- Words like "probably", "likely", "might", "presumably" are FORBIDDEN.
- A short correct doc beats a long hallucinated one.

## CHAIN-OF-THOUGHT PROCESS

Before writing, reason through these steps internally:
1. Identify all classes and their base classes from the data.
2. List all public methods per class (skip _private unless they have docstrings).
3. List all top-level functions with their full signatures.
4. Extract what docstrings actually say (quote, don't paraphrase).
5. Map the dependency graph: what does this module need, who uses it.
6. Only THEN write the documentation.

## OUTPUT FORMAT — CRITICAL

Your response must be ONLY the final markdown document — the COMPLETE documentation.
- NO preamble ("Here is the documentation...")
- NO ```markdown fences wrapping the output
- NO explanation or commentary after the document
- NO writing to files
- Your text output IS the final deliverable. It will be saved directly as the .md file.
- If you feel the document is too long: it is not. Include EVERYTHING. Completeness is mandatory.

## DOCUMENT STRUCTURE

Use this exact heading hierarchy — the DocsSystem indexes by # levels:

```
# <Module Name>

<1-2 sentences: what this module does, derived from module docstring or class/function analysis>

## Why This Matters

<User perspective: Why would someone care about this module? What problem does it solve?
 When would you reach for it? Keep it practical, 2-3 sentences max.>

## Quick Start

<Minimal usage example if derivable from signatures. If not derivable: skip this section entirely.
 Show the import and one representative call. Nothing invented.>

## Usage Guide

<Practical usage patterns derived from the public API signatures.
 Show 2-3 realistic scenarios based on different function/method combinations.
 Each scenario: one sentence what it does, then the code.
 Only use functions/classes that exist in the provided code data.
 If the module has fewer than 3 public functions: merge this into Quick Start and skip this section.>

### Basic Usage

```python | and or bash | and or in terminal
# Derived from the most common/simple function signatures
```

## Live Usage Examples

<Practical, runnable examples that show this module in real use.
 Each example should be a complete, copy-pasteable snippet.
 Where possible, show this module working TOGETHER with other ToolBoxV2 modules
 it depends on (from the upstream graph) or that depend on it (downstream graph).
 This section bridges the gap between API reference and real integration.

 Structure per example:
 1. One sentence: what this demonstrates
 2. Complete code block (imports included, runnable as-is)
 3. Expected output or behavior (as a comment)

 Include 1-3 examples depending on module complexity:
 - Simple module (1-2 public functions): 1 example showing basic + integration
 - Medium module (1 class, few methods): 2 examples — standalone + with another module
 - Complex module (multiple classes): 3 examples — basic, integration, advanced combo

 ONLY use modules and functions that exist in the provided code data or upstream/downstream graph.
 If no integration partners exist in the graph, show standalone usage only.
 A mini-demo that shows realistic data flow is worth more than isolated API calls.>

### Advanced Usage

```python | and or bash | and or in terminal
# Derived from more complex signatures (async, callbacks, optional params)
```

## How It Works

<Developer perspective: Architecture in 3-5 sentences.
 Internal design, data flow, key patterns used.
 Only what the code signatures and docstrings reveal.>

## API Reference

### Classes

#### `ClassName(BaseClass)`

<Class docstring if available. Otherwise 1 sentence derived from its methods.>

| Method | Signature | Description |
|--------|-----------|-------------|
| `method_name` | `def method_name(self, param: type) -> return` | <from docstring or parameter names> |

### Functions

#### `function_name(param1: type, param2: type) -> ReturnType`

<What it does — from docstring. If no docstring: derive from name + params.>

**Parameters:**
- `param1` — <type and purpose if docstring says>
- `param2` — <type and purpose if docstring says>

**Returns:** <return type and meaning>

## Dependencies

<From the upstream graph. List what this module imports/uses from other modules.
 Format: `ModuleName.function` from `path/to/file.py`
 If empty: "No indexed upstream dependencies.">

## Used By

<From the downstream graph. List what other modules call into this one.
 Format: Referenced by `CallerName` in `path/to/caller.py`
 If empty: "No indexed downstream usages.">
```

## QUALITY STANDARDS

- Headings use # hierarchy matching the DocsSystem parser (# = level 1, ## = level 2, etc.)
- Tables for methods — scannable, not walls of text.
- Parameter docs only when docstrings provide info — never invent parameter descriptions.
- Code examples only when directly derivable — a wrong example is worse than none.
- English throughout. Technical writing: clear, direct, no filler.

## CROSS-LINKING (mandatory)

- Links MUST reflect the actual docs/ directory structure, not flat names.
- Format: `[ModuleName](relative/path/to/modulename.md)` — match the source file's directory.
- In Dependencies section: EVERY upstream module MUST be a markdown link to its doc path.
- In Used By section: EVERY downstream caller MUST be a markdown link to its doc path.
- The file paths are provided in the code data — derive the doc path by replacing the source extension with `.md`.

## ARCHITECTURE DIAGRAMS (when applicable)

If the module has 3+ classes or a clear data flow, include a mermaid diagram.
Only include classes/functions that actually exist in the provided code data.
Keep diagrams small: max 8 nodes. Do NOT invent connections.

## KNOWN ISSUES / BUGS SECTION (when applicable)

If the code data reveals patterns like `except: pass`, TODO/FIXME in docstrings,
very long functions without docstrings — add a `## Known Issues` section.
Only include if there are actual issues found in the data. Do not invent issues.

## FOR UPDATES TO EXISTING DOCS

When updating an existing document:
1. Preserve the structure and any human-written context that is still accurate.
2. Update signatures that changed in the code.
3. Remove documentation for code elements that no longer exist.
4. Add new elements that exist in code but not in the doc.
5. Never mark anything as deprecated unless the code explicitly says so.
"""

init_buld = [False]

async def _get_or_build_doc_agent_sync():
    """Get or build the doc agent singleton (sync, runs in bg thread)."""

    logger.info("Building doc_agent ISAA agent (reason mode)...")
    app = _tb_app or get_app("doc_server agent init")

    isaa = app.get_mod("isaa")
    if isaa is None:
        logger.error("ISAA module not loaded")
        raise RuntimeError("ISAA module not loaded — start with: tb -m isaa -f doc_server")
    if init_buld[0]:
        return await isaa.get_agent("doc_agent")
    await isaa.init_isaa(name="doc_agent")

    builder = isaa.get_agent_builder(
        "doc_agent",
        add_base_tools=False,
        with_dangerous_shell=False,
    )
    builder.config.system_message = DOC_AGENT_PROMPT

    await isaa.register_agent(builder)
    init_buld[0] = True
    _doc_agent = await isaa.get_agent("doc_agent")
    logger.info("doc_agent ready")
    return _doc_agent

async def _run_doc_job(job_id: str, file_path: str, update_existing: str = None):
    """
    Synchronous doc generation — runs in background thread via app.run_bg_task_advanced.

    Pipeline:
    1. lookup_code(file_path, include_code=True)
    2. get_task_context([file_path], intent)
    3. If update_existing: read existing doc content
    4. Build grounded prompt
    5. Call ISAA agent
    6. Write to docs/ and re-index
    """
    logger.info(f"Job {job_id}: starting doc generation for {file_path}")
    _jobs.update(job_id, status="running", started=time.time())

    try:
        docs = _get_docs()
        fp = Path(file_path)

        try:
            # 1. Get code elements
            logger.debug(f"Job {job_id}: looking up code elements for {file_path}")
            code_result = await docs.lookup_code(file_path=file_path, include_code=True, max_results=100)
            elements = code_result.get("results", [])
            if not elements:
                code_result = await docs.lookup_code(file_path=fp.name, include_code=True, max_results=100)
                elements = code_result.get("results", [])

            if not elements:
                logger.warning(f"Job {job_id}: no code elements found for {file_path}")
                _jobs.update(job_id, status="error", finished=time.time(),
                             error=f"No code elements found in index for {file_path}")
                return

            logger.info(f"Job {job_id}: found {len(elements)} code elements")

            # 2. Get context graph
            try:
                ctx = await docs.get_task_context(files=[file_path], intent="Generate comprehensive module documentation")
                context_graph = ctx.get("result", {}).get("context_graph", {})
                logger.debug(f"Job {job_id}: context graph retrieved")
            except Exception as e:
                logger.warning(f"Job {job_id}: context graph failed — {e}")
                context_graph = {"upstream_dependencies": [], "downstream_usages": []}

            # 3. Read existing doc if updating
            existing_content = ""
            if update_existing:
                existing_path = DOCS_DIR / update_existing
                if existing_path.exists():
                    existing_content = existing_path.read_text(encoding="utf-8", errors="ignore")

            # 4. Build prompt with context-budget awareness
            budget = _get_context_budget()
            upstream = context_graph.get("upstream_dependencies", [])
            downstream = context_graph.get("downstream_usages", [])

            # Reserve tokens for non-code parts of the prompt
            graph_text = ""
            graph_text += "### Upstream (what this module imports and uses)\n"
            if upstream:
                for u in upstream:
                    graph_text += f"\n- **`{u.get('name', '?')}`** ({u.get('type', '?')}) from `{u.get('file', '?')}`"
                    if u.get("import_statement"):
                        graph_text += f"\n  Import: `{u['import_statement']}`"
                    if u.get("usage_snippet"):
                        graph_text += f"\n  Usage: `{u['usage_snippet']}`"
                    if u.get("signature"):
                        graph_text += f"\n  Signature: `{u['signature']}`"
            else:
                graph_text += "No indexed upstream dependencies."
            graph_text += "\n\n### Downstream (who imports and uses this module)\n"
            if downstream:
                for d in downstream:
                    graph_text += f"\n- **`{d.get('imported_name', '?')}`** used in `{d.get('file', '?')}`"
                    if d.get("import_statement"):
                        graph_text += f"\n  Import: `{d['import_statement']}`"
                    for snippet in d.get("usage_snippets", []):
                        graph_text += f"\n  Usage: `{snippet}`"
            else:
                graph_text += "No indexed downstream usages."

            existing_text = ""
            if existing_content:
                existing_text = f"\n## Existing Documentation (to update)\n\n```markdown\n{existing_content[:_MAX_EXISTING_DOC_CHARS]}\n```\n\nUpdate this documentation to match the current code above.\n"
            else:
                existing_text = "\nCreate a new documentation page for this module following the structure in your system prompt.\nYour COMPLETE response must be the final markdown document — nothing else.\n"

            # Calculate remaining budget for code sections
            overhead_tokens = _estimate_tokens(graph_text) + _estimate_tokens(
                existing_text) + 200  # 200 for framing
            code_budget = budget - overhead_tokens

            code_section = _build_code_prompt_sections(elements, code_budget, job_id=job_id)

            user_prompt = f"""Document this file: `{file_path}`

## Verified Code Elements (from AST index — this is ground truth)

{chr(10).join(code_section)}

## Dependency Graph (from index)

{graph_text}
{existing_text}"""

                    # Select model based on prompt size
            prompt_tokens = _estimate_tokens(user_prompt)
            model_hint = None
            log_model = os.getenv("LOGCONTEXTMODEL", "")
            if log_model:
                if prompt_tokens > 50_000:
                    model_hint = log_model  # Use the large-context model for big prompts
                    logger.info(
                        f"Job {job_id}: large prompt ({prompt_tokens} tokens) — using LOGCONTEXTMODEL={log_model}")
                else:
                    logger.info(f"Job {job_id}: normal prompt ({prompt_tokens} tokens) — using default model")

            # 5. Call ISAA agent
            logger.info(
                f"Job {job_id}: calling ISAA agent (prompt {len(user_prompt)} chars, ~{prompt_tokens} tokens)")

            # Build agent in THIS event loop (each bg thread has its own)
            app = _tb_app or get_app("doc_server agent")
            isaa = app.get_mod("isaa")
            if isaa is None:
                raise RuntimeError("ISAA module not loaded — start with: tb -m isaa -f doc_server")

            agent_name = f"doc_agent"
            from toolboxv2.mods.isaa.base.Agent.builder import FlowAgentBuilder
            agent = await (
                FlowAgentBuilder()
                .with_name(agent_name)
                .with_models(os.getenv("LOGCONTEXTMODEL", os.getenv("LIGHNIGMODEL", os.getenv("BLITZMODEL", "gemini/gemini-2.5-flash"))))
                .with_system_message(DOC_AGENT_PROMPT)
                .with_temperature(0.3)
                .build()
            )

            run_kwargs = dict(query=user_prompt, session_id=f"doc_session_{job_id}", human_online=False)
            if model_hint:
                run_kwargs["model"] = model_hint

            result = await agent.a_run(**run_kwargs)

            if not result or not isinstance(result, str):
                logger.error(f"Job {job_id}: agent returned empty/invalid result: {type(result)}")
                _jobs.update(job_id, status="error", finished=time.time(),
                             error="Agent returned empty result")
                return

            if len(result) < 50 or result.startswith("Es ist ein Fehler aufgetreten:") :
                logger.error(f"Job {job_id}: agent returned invalid result: {result}")
                _jobs.update(job_id, status="error", finished=time.time(),
                             error="Agent returned empty result")
                return

            logger.info(f"Job {job_id}: agent returned {len(result)} chars")

            # Clean markdown fences
            markdown_content = result.strip()
            if markdown_content.startswith("```markdown"):
                markdown_content = markdown_content[len("```markdown"):].strip()
            if markdown_content.startswith("```"):
                markdown_content = markdown_content[3:].strip()
            if markdown_content.endswith("```"):
                markdown_content = markdown_content[:-3].strip()

            # 6. Write to disk and re-index
            if update_existing:
                out_path = DOCS_DIR / update_existing
            else:
                try:
                    rel = fp.relative_to(PROJECT_ROOT)
                except ValueError:
                    rel = fp
                out_path = DOCS_DIR / rel.with_suffix(".md")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(markdown_content, encoding="utf-8")
            logger.info(f"Job {job_id}: wrote {len(markdown_content.split())} words -> {out_path}")

            # Re-index
            try:
                await docs._update_file(out_path)
                docs.context.clear_cache()
                logger.debug(f"Job {job_id}: re-indexed {out_path}")
            except Exception as e:
                logger.warning(f"Job {job_id}: re-index failed — {e}")

            # Compute relative path for the view link
            try:
                view_path = str(out_path.relative_to(DOCS_DIR))
            except ValueError:
                try:
                    view_path = str(out_path.relative_to(PROJECT_ROOT))
                except ValueError:
                    view_path = str(out_path)

            _jobs.update(
                job_id,
                status="done",
                finished=time.time(),
                output_path=view_path,
                result={
                    "path": view_path,
                    "full_path": str(out_path),
                    "elements_used": len(elements),
                    "words_generated": len(markdown_content.split()),
                    "mode": "update" if update_existing else "create",
                },
            )
            logger.info(f"Job {job_id}: done — {view_path}")

        finally:
            logger.info(f"Job {job_id}: close")

    except Exception as e:
        import traceback
        logger.error(f"Job {job_id}: failed — {e}\n{traceback.format_exc()}")
        _jobs.update(job_id, status="error", finished=time.time(), error=str(e))


@web_app.post("/api/generate-doc")
async def api_generate_doc(request: ParsedRequest):
    data = request.json_data
    if not data:
        logger.warning("api_generate_doc: invalid JSON body")
        return (400, {"ok": False, "error": "Invalid JSON body"})

    file_path = data.get("file_path")
    update_existing = data.get("update_existing")

    if not file_path:
        logger.warning("api_generate_doc: missing file_path")
        return (400, {"ok": False, "error": "file_path required"})

    job_id = _jobs.create(file_path, update_existing)
    logger.info(f"api_generate_doc: created job {job_id} for {file_path}")

    # Run in background via app.run_bg_task_advanced
    app = _tb_app or get_app("doc_server job dispatch")
    app.run_bg_task_advanced(_run_doc_job, job_id, file_path, update_existing)
    # await _run_doc_job(job_id, file_path, update_existing)
    return {"ok": True, "job_id": job_id, "status": "queued"}

@web_app.post("/api/batch-generate")
async def api_batch_generate(request: ParsedRequest):
    data = request.json_data
    if not data:
        return (400, {"ok": False, "error": "Invalid JSON body"})

    file_paths = data.get("file_paths", [])
    if not file_paths:
        return (400, {"ok": False, "error": "file_paths required (list)"})

    if len(file_paths) > 60000:
        return (400, {"ok": False, "error": "Max 60000 files per batch"})

    app = _tb_app or get_app("doc_server batch dispatch")
    job_ids = []

    for fp in file_paths:
        job_id = _jobs.create(fp)
        job_ids.append(job_id)
        app.run_bg_task_advanced(_run_doc_job, job_id, fp)

    return {"ok": True, "job_ids": job_ids, "count": len(job_ids)}

# ═════════════════════════════════════════════════════════
# FLOW ENTRY
# ═════════════════════════════════════════════════════════
@web_app.post("/api/shutdown")
async def api_shutdown():
    logger.info("Shutdown requested via dashboard")
    import _thread
    # Respond first, then kill
    def _delayed_exit():
        time.sleep(0.5)
        logger.info("Server stopping...")
        _thread.interrupt_main()  # Raises KeyboardInterrupt in main thread
        from toolboxv2 import get_app
        get_app().exit()
        import sys
        sys.exit(0)
    threading.Thread(target=_delayed_exit, daemon=True).start()
    return {"ok": True, "message": "Shutting down..."}

async def run(app, args):
    """Flow entry — initializes DocsSystem, starts WSGI server via FastTBHandler."""
    global _docs, _tb_app, PROJECT_ROOT, DOCS_DIR, MKDOCS_YML

    _tb_app = app

    # Optional args: project_root, docs_root
    project_root = getattr(args, "project_root", None)
    docs_root = getattr(args, "docs_root", None)

    if project_root:
        PROJECT_ROOT = Path(project_root).resolve()
    if docs_root:
        DOCS_DIR = Path(docs_root).resolve()
    else:
        DOCS_DIR = PROJECT_ROOT / "docs"

    MKDOCS_YML = PROJECT_ROOT / "mkdocs.yml"

    host = getattr(args, "host", None) or "0.0.0.0"
    port = getattr(args, "port", None) or 8088

    logger.info(f"PROJECT_ROOT = {PROJECT_ROOT}")
    logger.info(f"DOCS_DIR     = {DOCS_DIR}")
    logger.info(f"DOCS_DIR exists: {DOCS_DIR.exists()}")
    if DOCS_DIR.exists():
        doc_files = list(DOCS_DIR.rglob("*.md"))
        logger.info(f"DOCS_DIR contains {len(doc_files)} .md files")
        for f in doc_files[:10]:
            logger.debug(f"  {f.relative_to(DOCS_DIR)}")
        if len(doc_files) > 10:
            logger.debug(f"  ... and {len(doc_files) - 10} more")

    # ── Initialize DocsSystem (use existing index if available, refresh in background) ──
    from toolboxv2.utils.extras.mkdocs import create_docs_system

    logger.info("Initializing documentation index...")
    _docs = create_docs_system(
        project_root=str(PROJECT_ROOT),
        docs_root=str(DOCS_DIR),
        include_dirs=["toolboxv2", "mods", "flows", "utils"],
    )
    with Spinner("Initializing documentation index...", symbols="h"):
        result = await _docs.initialize(force_rebuild=False)
    logger.info(f"Index: {result['sections']} sections, {result['elements']} elements "
                f"({result['status']}, {result['time_ms']:.0f}ms)")

    # Background refresh: sync with git changes
    async def _bg_refresh():
        try:
            with Spinner("Sync documentation index...", symbols="e"):
                sync_result = await _docs.sync()
            if sync_result.get("files_updated", 0) > 0:
                logger.info(f"Background sync: {sync_result['files_updated']} files updated")
            else:
                logger.debug("Background sync: no changes")
        except Exception as e:
            import traceback
            logger.warning(f"Background sync failed: {e} {traceback.format_exc()}")

    if result["status"] == "loaded":
        # Index was loaded from cache — refresh in background
        logger.info("Index loaded from cache — scheduling background refresh")
        app.run_bg_task_advanced(_bg_refresh)

    # ── Start WSGI server ──
    handler = FastTBHandler(web_app)
    wsgi_app = handler.as_wsgi_app()

    pp = f"Doc Server -> http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}"
    print(pp)
    logger.info(pp)
    web_app.auth = True
    web_app.serve(host=host, port=port, enable_ws=True)
    # Register WS handlers if any
    # ws_handlers = web_app.get_websocket_handlers()
    # from waitress.server import create_server
    # server = create_server(wsgi_app, host=host, port=port)
    # logger.info("Press Ctrl+C to stop")
    # try:
    #     server.run()
    # except KeyboardInterrupt:
    #     logger.info("Shutting down doc server...")
    #     server.close()
