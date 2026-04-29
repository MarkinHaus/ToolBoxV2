"""
toolboxv2/flows/doc_server.py
─────────────────────────────
MkDocs Documentation Hub — TBJS Paper Style

Dual-purpose documentation server:
  • Users  → browse & search rendered docs, view changelogs
  • Devs   → coverage dashboard, staleness tracker, missing-docs audit,
             live editor, MkDocs build trigger

Data comes from DocsSystem (nbpaper_style.py) — pre-built inverted index,
no filesystem scanning at request time.

Stack: FastAPI + TBJS Paper neo-brutalism
Run  : `tb doc_server` or `python -m toolboxv2 -fg doc_server`
"""

import asyncio
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from toolboxv2 import tb_root_dir

NAME = "doc_server"

# ── Paths ───────────────────────────────────────────────
PROJECT_ROOT = tb_root_dir.parent
DOCS_DIR = PROJECT_ROOT / "docs"
MKDOCS_YML = PROJECT_ROOT / "mkdocs.yml"

# ── FastAPI ─────────────────────────────────────────────
from contextlib import asynccontextmanager

_docs = None  # type: Optional[object]
_tb_app = None  # ToolBoxV2 App instance — set in run(), needed for ISAA


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Initialize DocsSystem once before first request."""
    global _docs
    from toolboxv2.utils.extras.mkdocs import create_docs_system

    print("📚 Initializing documentation index...")
    _docs = create_docs_system(
        project_root=str(PROJECT_ROOT),
        docs_root=str(DOCS_DIR),
        include_dirs=["toolboxv2", "mods", "flows", "utils"],
    )
    result = await _docs.initialize()
    print(f"   Index: {result['sections']} sections, {result['elements']} elements "
          f"({result['status']}, {result['time_ms']:.0f}ms)")
    yield
    # shutdown — nothing to clean up


web_app = FastAPI(title="ToolBoxV2 Docs", version="2.0.0", lifespan=_lifespan)


def _get_docs():
    """Get the initialized DocsSystem."""
    if _docs is None:
        raise HTTPException(503, "Documentation index is still loading — try again in a moment")
    return _docs


# ═════════════════════════════════════════════════════════
# DATA LAYER — reads from DocsSystem index, zero I/O
# ═════════════════════════════════════════════════════════


def _get_doc_pages() -> list[dict]:
    """Get doc sections from the pre-built index."""
    docs = _get_docs()
    sections = docs.index_mgr.index.sections
    # Group by file, return one entry per file
    by_file = defaultdict(list)
    for s in sections.values():
        by_file[s.file_path].append(s)

    pages = []
    for file_path, secs in sorted(by_file.items()):
        fp = Path(file_path)
        try:
            rel = fp.relative_to(DOCS_DIR)
        except ValueError:
            rel = fp.name
        total_content = sum(len(s.content) for s in secs)
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

    # Doc sections indexed by name (lowercase)
    doc_names = set()
    doc_files = set()
    for s in idx.sections.values():
        doc_names.add(s.title.lower())
        doc_files.add(Path(s.file_path).stem.lower())
        # Also add referenced source names
        for ref in s.source_refs:
            doc_names.add(ref.lower())

    # Code elements grouped by file
    by_file = defaultdict(list)
    for eid, elem in idx.code_elements.items():
        by_file[elem.file_path].append(elem)

    documented_files = []
    missing_files = []

    for file_path, elems in sorted(by_file.items()):
        fp = Path(file_path)
        file_stem = fp.stem.lower()

        # Count classes and functions
        classes = sum(1 for e in elems if e.element_type == "class")
        functions = sum(1 for e in elems if e.element_type in ("function", "method"))
        has_docstring = any(e.docstring for e in elems)

        info = {
            "name": fp.stem,
            "rel_path": file_path,
            "classes": classes,
            "functions": functions,
            "has_docstring": has_docstring,
            "elements": len(elems),
        }

        # Check if this file has matching documentation
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

    # Staleness from suggestions
    stale = []
    # Check file hashes — if a code file's hash differs from when docs were written
    for file_path, fhash in idx.file_hashes.items():
        fp = Path(file_path)
        if fp.suffix in ('.py', '.js', '.ts'):
            doc_stem = fp.stem.lower()
            if doc_stem in doc_files:
                # Find matching doc section
                for s in idx.sections.values():
                    if Path(s.file_path).stem.lower() == doc_stem:
                        # Compare timestamps
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

    # Quality score
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
# TEMPLATE ENGINE — TBJS Paper Neo-Brutalism
# ═════════════════════════════════════════════════════════

PAPER_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

:root {
  /* OKLCH primitives */
  --raw-primary: 55% 0.18 230;
  --raw-success: 65% 0.2 145;
  --raw-warning: 75% 0.18 85;
  --raw-error:   55% 0.22 25;

  --primary: oklch(55% 0.18 230);
  --success: oklch(65% 0.2 145);
  --warning: oklch(75% 0.18 85);
  --error:   oklch(55% 0.22 25);

  /* Paper Light */
  --paper-bg:      #f4f1ea;
  --paper-surface: #ffffff;
  --paper-sunken:  #ebe7dc;
  --ink:           #111111;
  --ink-muted:     #555555;
  --ink-faint:     #888888;
  --rule:          #111111;

  /* Typography */
  --font-display: 'IBM Plex Mono', ui-monospace, 'SF Mono', Consolas, monospace;
  --font-body:    'IBM Plex Sans', system-ui, -apple-system, sans-serif;

  --text-display: clamp(32px, 5vw, 48px);
  --text-h1: clamp(28px, 3.5vw, 36px);
  --text-h2: clamp(22px, 2.5vw, 28px);
  --text-h3: clamp(18px, 2vw, 22px);
  --text-base: 16px;
  --text-sm: 14px;
  --text-xs: 12px;

  /* Spacing (8pt grid) */
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

/* ── Nav ──────────────────────────────────── */
.nav {
  position: sticky;
  top: 0;
  z-index: 100;
  display: flex;
  align-items: center;
  gap: var(--space-5);
  padding: var(--space-4) var(--space-6);
  background: var(--paper-bg);
  border-block-end: 3px solid var(--ink);
}
.nav-brand {
  font-family: var(--font-display);
  font-size: var(--text-h3);
  font-weight: 700;
  color: var(--ink);
  text-decoration: none;
  letter-spacing: -0.02em;
}
.nav-links { display: flex; gap: var(--space-4); margin-left: auto; }
.nav-link {
  font-family: var(--font-display);
  font-size: var(--text-sm);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--ink);
  text-decoration: none;
  padding: var(--space-1) 0;
  border-bottom: 2px solid transparent;
  transition: border-color 80ms linear;
}
.nav-link:hover, .nav-link.active {
  border-bottom-color: var(--ink);
}
.nav-toggle {
  display: none;
  background: none;
  border: 2px solid var(--ink);
  color: var(--ink);
  font-family: var(--font-display);
  font-size: var(--text-base);
  padding: var(--space-2) var(--space-3);
  cursor: pointer;
}

/* ── Layout ───────────────────────────────── */
.container {
  max-width: 1100px;
  margin: 0 auto;
  padding: var(--space-7) var(--space-6);
}
.page-title {
  font-family: var(--font-display);
  font-size: var(--text-h1);
  font-weight: 600;
  line-height: 1.15;
  letter-spacing: -0.02em;
  color: var(--ink);
  margin-bottom: var(--space-6);
}
.page-subtitle {
  font-family: var(--font-body);
  font-size: var(--text-base);
  color: var(--ink-muted);
  max-width: 68ch;
  margin-top: calc(-1 * var(--space-4));
  margin-bottom: var(--space-6);
}

/* ── Card ─────────────────────────────────── */
.card {
  padding: var(--space-5);
  background: var(--paper-surface);
  border: 2px solid var(--ink);
  border-radius: 0;
  box-shadow: 6px 6px 0 var(--ink);
  margin-bottom: var(--space-6);
  margin-right: 8px;
  transition: transform 100ms linear, box-shadow 100ms linear;
}
.card:hover {
  transform: translate(-2px, -2px);
  box-shadow: 8px 8px 0 var(--ink);
}
.card-title {
  font-family: var(--font-display);
  font-size: var(--text-h3);
  font-weight: 600;
  margin: 0 0 var(--space-3);
}
.card-eyebrow {
  font-family: var(--font-display);
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 2px;
  color: var(--ink-muted);
  margin: 0 0 var(--space-3);
}
.card--flat {
  box-shadow: none;
  transition: none;
}
.card--flat:hover {
  transform: none;
  box-shadow: none;
}

/* ── Grid ─────────────────────────────────── */
.grid { display: grid; gap: var(--space-6); }
.grid-2 { grid-template-columns: repeat(2, 1fr); }
.grid-3 { grid-template-columns: repeat(3, 1fr); }
.grid-4 { grid-template-columns: repeat(4, 1fr); }

/* ── Badge ────────────────────────────────── */
.badge {
  display: inline-flex;
  align-items: center;
  padding: 0.2rem 0.5rem;
  font-family: var(--font-display);
  font-size: var(--text-xs);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
  background: var(--ink);
  color: var(--paper-bg);
  border: 2px solid var(--ink);
  border-radius: 0;
}
.badge--success { background: var(--success); border-color: var(--success); color: #fff; }
.badge--warning { background: var(--warning); border-color: var(--warning); color: var(--ink); }
.badge--error   { background: var(--error); border-color: var(--error); color: #fff; }
.badge--ghost   { background: transparent; color: var(--ink); }

/* ── Button ───────────────────────────────── */
.btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1.25rem;
  font-family: var(--font-display);
  font-size: var(--text-sm);
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  text-decoration: none;
  background: var(--paper-surface);
  color: var(--ink);
  border: 2px solid var(--ink);
  border-radius: 0;
  box-shadow: 4px 4px 0 var(--ink);
  cursor: pointer;
  transition: transform 80ms linear, box-shadow 80ms linear;
}
.btn:hover {
  transform: translate(-2px, -2px);
  box-shadow: 6px 6px 0 var(--ink);
}
.btn:active {
  transform: translate(2px, 2px);
  box-shadow: 0 0 0 var(--ink);
}
.btn--primary { background: var(--primary); color: #fff; }
.btn--danger  { background: var(--error); color: #fff; }
.btn--ghost   { background: transparent; }
.btn--sm { padding: 0.4rem 0.75rem; font-size: var(--text-xs); }

/* ── Progress bar ─────────────────────────── */
.progress {
  height: 12px;
  background: var(--paper-sunken);
  border: 2px solid var(--ink);
  border-radius: 0;
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  transition: width 600ms ease;
}

/* ── Table ─────────────────────────────────── */
.table-wrap { overflow-x: auto; margin-right: 8px; margin-bottom: 8px; }
table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: var(--text-sm);
  font-family: var(--font-body);
  border: 2px solid var(--ink);
}
th {
  font-family: var(--font-display);
  font-size: var(--text-xs);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--paper-bg);
  background: var(--ink);
  text-align: left;
  padding: var(--space-3) var(--space-4);
  white-space: nowrap;
}
td {
  padding: var(--space-3) var(--space-4);
  border-bottom: 2px solid var(--paper-sunken);
  vertical-align: middle;
  line-height: 1.5;
  background: var(--paper-surface);
}
td:first-child { white-space: nowrap; font-weight: 500; }
tr:nth-child(even) td { background: var(--paper-sunken); }
tr:hover td { background: var(--ink); color: var(--paper-bg); }
tr:hover td a { color: var(--paper-bg); }
tr:hover td code { color: var(--paper-bg); }
tr:hover td .badge { border-color: var(--paper-bg); }
/* Path columns — truncate long paths */
td.path-cell {
  max-width: 300px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--ink-muted);
  font-size: var(--text-xs);
}

/* ── md-content tables (rendered from markdown) ── */
.md-content table {
  margin: var(--space-5) 0;
  box-shadow: 4px 4px 0 var(--ink);
}
.md-content th {
  font-size: var(--text-xs);
  padding: var(--space-2) var(--space-3);
}
.md-content td {
  font-size: var(--text-sm);
  padding: var(--space-2) var(--space-3);
}
.md-content td code {
  font-size: 0.85em;
  font-weight: 500;
}

/* ── Mermaid diagrams ─────────────────────── */
.mermaid-container {
  margin: var(--space-5) 0;
  padding: var(--space-5);
  background: var(--paper-surface);
  border: 2px solid var(--ink);
  box-shadow: 4px 4px 0 var(--ink);
  overflow-x: auto;
  text-align: center;
}

/* ── Stat block ───────────────────────────── */
.stat { text-align: center; }
.stat-value {
  font-family: var(--font-display);
  font-size: var(--text-display);
  font-weight: 700;
  line-height: 1;
  color: var(--ink);
}
.stat-label {
  font-family: var(--font-display);
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 2px;
  color: var(--ink-muted);
  margin-top: var(--space-2);
}

/* ── Form ─────────────────────────────────── */
textarea, input[type="text"], select {
  width: 100%;
  padding: 0.75rem 1rem;
  font-family: var(--font-body);
  font-size: var(--text-base);
  color: var(--ink);
  background: var(--paper-surface);
  border: 2px solid var(--ink);
  border-radius: 0;
  box-shadow: 4px 4px 0 var(--ink);
  transition: box-shadow 80ms linear, transform 80ms linear;
}
textarea:focus, input[type="text"]:focus {
  outline: none;
  transform: translate(-1px, -1px);
  box-shadow: 5px 5px 0 var(--primary);
  border-color: var(--primary);
}
textarea { font-family: var(--font-display); font-size: var(--text-sm); resize: vertical; }
label {
  display: block;
  font-family: var(--font-display);
  font-size: var(--text-sm);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: var(--space-2);
  color: var(--ink-muted);
}

/* ── Callout ──────────────────────────────── */
.callout {
  padding: 1rem 1.25rem;
  margin: var(--space-5) 0;
  background: var(--paper-sunken);
  border-left: 6px solid var(--ink);
}
.callout--warn  { border-left-color: var(--warning); }
.callout--error { border-left-color: var(--error); }
.callout--ok    { border-left-color: var(--success); }

/* ── Tabs ─────────────────────────────────── */
.tabs {
  display: flex;
  gap: 0;
  border-bottom: 3px solid var(--ink);
  margin-bottom: var(--space-6);
}
.tab {
  font-family: var(--font-display);
  font-size: var(--text-sm);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
  padding: var(--space-3) var(--space-5);
  background: transparent;
  color: var(--ink-muted);
  border: 2px solid transparent;
  border-bottom: none;
  cursor: pointer;
  text-decoration: none;
  transition: background 80ms;
}
.tab:hover { background: var(--paper-sunken); color: var(--ink); }
.tab.active {
  background: var(--paper-surface);
  color: var(--ink);
  border-color: var(--ink);
  border-bottom: 2px solid var(--paper-surface);
  margin-bottom: -3px;
}

/* ── Search ───────────────────────────────── */
.search-box {
  position: relative;
  max-width: 400px;
}
.search-box input {
  padding-left: 2.5rem;
}
.search-box::before {
  content: "⌕";
  position: absolute;
  left: 0.75rem;
  top: 50%;
  transform: translateY(-50%);
  font-size: 1.2rem;
  color: var(--ink-muted);
  z-index: 1;
}

/* ── Code ─────────────────────────────────── */
code {
  font-family: var(--font-display);
  font-size: 0.85em;
  background: var(--paper-sunken);
  padding: 0.1em 0.35em;
  border: none;
  border-radius: 0;
  white-space: nowrap;
}

/* Code inside tables — minimal treatment */
td code {
  background: transparent;
  padding: 0;
  font-weight: 500;
  white-space: nowrap;
}

pre {
  font-family: var(--font-display);
  font-size: var(--text-sm);
  line-height: 1.6;
  background: var(--paper-sunken);
  padding: 1.25rem 1.5rem;
  border: 2px solid var(--ink);
  box-shadow: 4px 4px 0 var(--ink);
  overflow-x: auto;
  margin: var(--space-5) 0;
  white-space: pre;
  tab-size: 4;
}

/* Reset inline code styling inside pre blocks */
pre code {
  background: none;
  border: none;
  padding: 0;
  font-size: inherit;
  line-height: inherit;
  word-break: normal;
  white-space: pre;
}

/* Syntax highlighting tokens */
.kw  { color: oklch(55% 0.18 230); font-weight: 600; } /* keywords */
.fn  { color: oklch(55% 0.15 280); }                    /* function names */
.cls { color: oklch(55% 0.18 230); font-weight: 600; } /* class names */
.str { color: oklch(60% 0.18 145); }                    /* strings */
.num { color: oklch(60% 0.18 85); }                     /* numbers */
.cmt { color: var(--ink-faint); font-style: italic; }   /* comments */
.dec { color: oklch(55% 0.15 280); }                    /* decorators */
.op  { color: var(--ink-muted); }                       /* operators */
[data-theme="dark"] .kw  { color: oklch(70% 0.15 230); }
[data-theme="dark"] .fn  { color: oklch(70% 0.12 280); }
[data-theme="dark"] .cls { color: oklch(70% 0.15 230); }
[data-theme="dark"] .str { color: oklch(72% 0.15 145); }
[data-theme="dark"] .num { color: oklch(75% 0.15 85); }
[data-theme="dark"] .cmt { color: var(--ink-faint); }

/* ── Links ────────────────────────────────── */
a {
  color: var(--primary);
  text-decoration: underline;
  text-decoration-thickness: 2px;
  text-underline-offset: 3px;
}
a:hover {
  background: var(--primary);
  color: var(--paper-surface);
  text-decoration-color: transparent;
}

/* ── HR ───────────────────────────────────── */
hr {
  border: none;
  border-block-start: 2px solid var(--ink);
  margin-block: var(--space-6);
}

/* ── Markdown rendered content ────────────── */
.md-content h1, .md-content h2, .md-content h3 {
  font-family: var(--font-display);
  font-weight: 600;
  line-height: 1.15;
  margin: var(--space-5) 0 var(--space-3);
}
.md-content h1 { font-size: var(--text-h1); }
.md-content h2 { font-size: var(--text-h2); border-bottom: 2px solid var(--ink); padding-bottom: var(--space-2); }
.md-content h3 { font-size: var(--text-h3); }
.md-content p  { max-inline-size: 68ch; margin-bottom: 1.2em; }
.md-content ul, .md-content ol { padding-left: var(--space-6); margin-bottom: 1.2em; }
.md-content li { margin-bottom: var(--space-2); }
.md-content blockquote {
  padding: 1rem 1.25rem;
  background: var(--paper-sunken);
  border-left: 6px solid var(--ink);
  margin: var(--space-4) 0;
}

/* ── Responsive ───────────────────────────── */
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

/* ── Dark mode toggle ─────────────────────── */
.theme-toggle {
  background: none;
  border: 2px solid var(--ink);
  color: var(--ink);
  font-family: var(--font-display);
  font-size: var(--text-sm);
  padding: var(--space-1) var(--space-3);
  cursor: pointer;
  text-transform: uppercase;
  letter-spacing: 1px;
  transition: transform 80ms linear, box-shadow 80ms linear;
  box-shadow: 2px 2px 0 var(--ink);
}
.theme-toggle:hover {
  transform: translate(-1px, -1px);
  box-shadow: 3px 3px 0 var(--ink);
}
.theme-toggle:active {
  transform: translate(1px, 1px);
  box-shadow: 0 0 0 var(--ink);
}

/* ── Doc agent job status ─────────────────── */
.job-status {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  font-family: var(--font-display);
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 1px;
}
.job-spinner {
  display: inline-block;
  width: 14px;
  height: 14px;
  border: 2px solid var(--ink-faint);
  border-top-color: var(--primary);
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
    <a href="/coverage" class="nav-link {_active('coverage')}">Coverage</a>
    <a href="/stale" class="nav-link {_active('stale')}">Staleness</a>
    <a href="/audit" class="nav-link {_active('audit')}">Audit</a>
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
  btn.innerHTML = '<span class="job-spinner"></span> Working...';
  try {{
    const resp = await fetch('/api/generate-doc', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{file_path: filePath}})
    }});
    const data = await resp.json();
    if (data.ok) {{
      btn.innerHTML = '✓ Done';
      btn.className = 'btn btn--sm btn--primary';
      setTimeout(() => location.reload(), 1500);
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
  btn.innerHTML = '<span class="job-spinner"></span> Fixing...';
  try {{
    const resp = await fetch('/api/generate-doc', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{file_path: srcPath, update_existing: docPath}})
    }});
    const data = await resp.json();
    if (data.ok) {{
      btn.innerHTML = '✓ Fixed';
      btn.className = 'btn btn--sm btn--primary';
      setTimeout(() => location.reload(), 1500);
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
</script>

</body>
</html>"""


def _render(content: str, title: str = "Docs", active: str = "") -> HTMLResponse:
    return HTMLResponse(_layout(content, title, active))


# ═════════════════════════════════════════════════════════
# ROUTES — DASHBOARD
# ═════════════════════════════════════════════════════════


@web_app.get("/", response_class=HTMLResponse)
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
        return score >= 20 and "D" or "F"

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
    <div style="margin-top: var(--space-4)">
      <span class="badge {_badge(cov["pct"])}">{cov["pct"]}% Covered</span>
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
  <a href="/coverage">View full coverage report →</a>
</div>
'''}

{"" if not cov["stale"] else f'''
<div class="callout callout--error">
  <strong>{len(cov["stale"])} docs</strong> may be outdated — source changed after last doc edit.
  <a href="/stale">Review staleness →</a>
</div>
'''}
"""
    return _render(content, "Dashboard", "dashboard")


# ═════════════════════════════════════════════════════════
# ROUTES — BROWSE (User-facing)
# ═════════════════════════════════════════════════════════

# Generic names that match too much text — skip in ref counting
_GENERIC_NAMES = frozenset({
    "self", "init", "main", "test", "args", "data", "name", "type",
    "none", "true", "false", "help", "list", "dict", "file", "path",
    "result", "value", "error", "config", "setup", "build", "load",
    "save", "read", "write", "update", "delete", "create", "close",
    "open", "start", "stop", "state", "event", "index", "parse",
    "handle", "process", "response", "request", "model", "query",
})


def _get_meaningful_code_names(docs_sys) -> set:
    """Get non-generic code element names for ref matching."""
    names = set()
    for elem in docs_sys.index_mgr.index.code_elements.values():
        n = elem.name.lower()
        if len(n) > 4 and n not in _GENERIC_NAMES and not n.startswith("_"):
            names.add(n)
    return names


def _count_code_refs_for_file(docs_sys, file_path: str, code_names: set) -> int:
    """Count how many unique code element names a doc file references — via inverted index O(k)."""
    inverted = docs_sys.index_mgr.index.inverted
    # Get all section_ids for this doc file
    section_ids = inverted.file_to_sections.get(file_path, set())
    if not section_ids:
        return 0

    # Check which code names have at least one keyword match in these sections
    matched = set()
    for name in code_names:
        # The inverted index keyword_to_sections maps keywords → section_ids
        kw_sections = inverted.keyword_to_sections.get(name, set())
        if kw_sections & section_ids:  # set intersection — O(min(a,b))
            matched.add(name)
    return len(matched)


@web_app.get("/browse", response_class=HTMLResponse)
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
          <td><a href="/view/{d["rel_path"]}">{d["rel_path"]}</a></td>
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
# ROUTES — VIEW PAGE (rendered markdown)
# ═════════════════════════════════════════════════════════


def _highlight_python(code: str) -> str:
    """Single-pass Python syntax highlighter — no cross-contamination between tokens."""
    PY_KW = (
        r'def|class|return|if|elif|else|for|while|try|except|finally|with|as|'
        r'import|from|raise|yield|async|await|pass|break|continue|and|or|not|in|is|'
        r'None|True|False|self|cls|lambda|global|nonlocal'
    )

    # Single combined regex — first match wins (priority order)
    TOKEN_RE = re.compile(
        r'(?P<cmt>#[^\n]*)'                                          # comments
        r'|(?P<tstr>"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')'            # triple strings
        r'|(?P<str>f?"[^"\\]*(?:\\.[^"\\]*)*"|f?\'[^\'\\]*(?:\\.[^\'\\]*)*\')'  # strings
        r'|(?P<dec>@\w+)'                                            # decorators
        r'|(?P<defn>(?<=def\s)\w+)'                                  # function name after def
        r'|(?P<clsn>(?<=class\s)\w+)'                                # class name after class
        r'|(?P<kw>\b(?:' + PY_KW + r')\b)'                          # keywords
        r'|(?P<num>\b\d+\.?\d*\b)'                                   # numbers
    )

    def _replace(m):
        for name in ('cmt', 'tstr', 'str', 'dec', 'defn', 'clsn', 'kw', 'num'):
            val = m.group(name)
            if val is not None:
                css = {'tstr': 'str', 'defn': 'fn', 'clsn': 'cls'}.get(name, name)
                safe = val.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                return f'<span class="{css}">{safe}</span>'
        return m.group(0).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Run tokenizer, then escape any unmatched text between tokens
    result = TOKEN_RE.sub(_replace, code)
    # Escape remaining unmatched text (between spans)
    # Split on span tags, escape non-tag parts
    parts = re.split(r'(<span class="[^"]+">|</span>)', result)
    out = []
    for part in parts:
        if part.startswith('<span') or part == '</span>':
            out.append(part)
        else:
            # Only escape if not already escaped by _replace
            if '<span' not in part:
                part = part.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                # Fix double-escaping from _replace
                part = part.replace("&amp;amp;", "&amp;").replace("&amp;lt;", "&lt;").replace("&amp;gt;", "&gt;")
            out.append(part)
    return "".join(out)


def _render_md(raw: str) -> str:
    """Minimal markdown → HTML with syntax highlighting, tables, and mermaid."""
    # 1. Extract code blocks + mermaid blocks first
    code_blocks = []
    def _stash_code(m):
        lang = m.group(1) or ""
        code = m.group(2)
        idx = len(code_blocks)
        placeholder = f"\x00CODE{idx}\x00"
        if lang == "mermaid":
            # Render as mermaid diagram container
            escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            code_blocks.append(
                f'<div class="mermaid-container"><pre class="mermaid">{escaped}</pre></div>'
            )
        else:
            if lang in ("python", "py", ""):
                # Highlight raw code first, then escape content inside spans
                code = _highlight_python(code)
            else:
                code = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            code_blocks.append(f'<pre><code class="lang-{lang}">{code}</code></pre>')
        return placeholder

    html = re.sub(r"```(\w*)\n(.*?)```", _stash_code, raw, flags=re.S)

    # 2. Extract markdown tables before HTML-escaping
    table_blocks = []
    def _stash_table(m):
        table_text = m.group(0)
        idx = len(table_blocks)
        placeholder = f"\x00TABLE{idx}\x00"
        table_blocks.append(_render_md_table(table_text))
        return placeholder

    html = re.sub(
        r"(?:^[|].*[|]\s*\n){2,}",
        _stash_table,
        html,
        flags=re.M,
    )

    # 3. HTML-escape everything else
    html = html.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 4. Markdown → HTML
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
    # Paragraphs
    html = re.sub(r"\n\n+", "</p><p>", html)
    html = f"<p>{html}</p>"
    html = re.sub(r"<p>\s*</p>", "", html)

    # 5. Restore stashed blocks
    for i, block in enumerate(code_blocks):
        html = html.replace(f"\x00CODE{i}\x00", block)
    for i, block in enumerate(table_blocks):
        html = html.replace(f"\x00TABLE{i}\x00", block)

    return html


def _render_md_table(table_text: str) -> str:
    """Convert a markdown table string to an HTML table."""
    lines = [l.strip() for l in table_text.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return table_text

    def _parse_row(line: str) -> list[str]:
        cells = line.split("|")
        # Strip leading/trailing empty cells from | delimiters
        if cells and not cells[0].strip():
            cells = cells[1:]
        if cells and not cells[-1].strip():
            cells = cells[:-1]
        return [c.strip() for c in cells]

    # Check if row 2 is a separator (|---|---|)
    sep_idx = 1
    is_sep = bool(re.match(r'^[|\s:\-]+$', lines[sep_idx])) if len(lines) > 1 else False

    header = _parse_row(lines[0])
    html = '<div class="table-wrap"><table>\n<thead><tr>'
    for cell in header:
        html += f"<th>{cell}</th>"
    html += "</tr></thead>\n<tbody>\n"

    start = 2 if is_sep else 1
    for line in lines[start:]:
        if re.match(r'^[|\s:\-]+$', line):
            continue  # skip separator lines
        cells = _parse_row(line)
        html += "<tr>"
        for cell in cells:
            # Inline code in cells
            cell = re.sub(r"`([^`]+)`", r"<code>\1</code>", cell)
            html += f"<td>{cell}</td>"
        html += "</tr>\n"

    html += "</tbody></table></div>"
    return html


@web_app.get("/view/{path:path}", response_class=HTMLResponse)
async def view_page(path: str):
    full = DOCS_DIR / path
    if not full.exists():
        raise HTTPException(404, f"Page not found: {path}")

    raw = full.read_text(encoding="utf-8", errors="ignore")
    rendered = _render_md(raw)

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
# ROUTES — COVERAGE (Dev-facing)
# ═════════════════════════════════════════════════════════


@web_app.get("/coverage", response_class=HTMLResponse)
async def coverage_page():
    cov = _compute_coverage()

    # Group missing by directory
    by_dir: dict[str, list] = {}
    for m in cov["missing"]:
        parts = Path(m["rel_path"]).parts
        dir_name = parts[0] if parts else "root"
        by_dir.setdefault(dir_name, []).append(m)

    missing_html = ""
    for dir_name, modules in sorted(by_dir.items()):
        missing_html += f"""
        <tr><td colspan="6" style="background: var(--paper-sunken); font-family: var(--font-display);
            font-weight: 600; text-transform: uppercase; letter-spacing: 1px; font-size: var(--text-xs)">
          {dir_name}/ — {len(modules)} undocumented
        </td></tr>"""
        for m in modules:
            priority = "badge--error" if m["classes"] > 0 or m["functions"] > 5 else "badge--warning"
            escaped_path = m["rel_path"].replace("\\", "\\\\").replace("'", "\\'")
            missing_html += f"""<tr>
              <td>{m["name"]}</td>
              <td class="path-cell" title="{m["rel_path"]}">{m["rel_path"]}</td>
              <td>{m["elements"]}</td>
              <td>{m["classes"]}C {m["functions"]}F</td>
              <td><span class="badge {priority}">{'High' if 'error' in priority else 'Med'}</span></td>
              <td><button class="btn btn--sm" onclick="generateDoc('{escaped_path}', this)">Generate</button></td>
            </tr>"""

    if not missing_html:
        missing_html = '<tr><td colspan="6" style="text-align:center; padding: var(--space-5)">All modules documented!</td></tr>'

    content = f"""
<h1 class="page-title">Coverage Report</h1>
<p class="page-subtitle">
  Every source module cross-referenced against existing documentation.
  High priority = classes or 5+ functions without any matching doc page.
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
      <div class="stat-label">Total Modules</div>
    </div>
  </div>
</div>

<div class="card card--flat">
  <div class="card-eyebrow">Undocumented Modules</div>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>Module</th><th>Path</th><th>Size</th><th>Entities</th><th>Priority</th><th>Action</th>
      </tr></thead>
      <tbody>{missing_html}</tbody>
    </table>
  </div>
</div>
"""
    return _render(content, "Coverage", "coverage")


# ═════════════════════════════════════════════════════════
# ROUTES — STALENESS (Dev-facing)
# ═════════════════════════════════════════════════════════


@web_app.get("/stale", response_class=HTMLResponse)
async def staleness_page():
    cov = _compute_coverage()

    stale_rows = ""
    for s in cov["stale"]:
        escaped_doc = s["doc_path"].replace("\\", "\\\\").replace("'", "\\'")
        escaped_src = s["src_path"].replace("\\", "\\\\").replace("'", "\\'")
        stale_rows += f"""<tr>
          <td><a href="/view/{s["doc_path"]}">{s["doc_path"]}</a></td>
          <td>{s["doc_mtime"]}</td>
          <td><code>{s["src_path"]}</code></td>
          <td>{s["src_mtime"]}</td>
          <td><span class="badge badge--warning">Stale</span></td>
          <td>
            <button class="btn btn--sm" onclick="fixStaleDoc('{escaped_doc}', '{escaped_src}', this)">Auto-Fix</button>
          </td>
        </tr>"""

    if not stale_rows:
        stale_rows = '<tr><td colspan="6" style="text-align:center; padding: var(--space-5)">All docs are fresh!</td></tr>'

    callout = ""
    if cov["stale"]:
        callout = f"""
        <div class="callout callout--warn">
          <strong>{len(cov["stale"])} documentation pages</strong> were last edited
          <em>before</em> their corresponding source files changed.
          These may contain outdated information.
        </div>"""

    content = f"""
<h1 class="page-title">Staleness Tracker</h1>
<p class="page-subtitle">
  Documentation pages where the source module was modified after the doc page.
  These likely need review and updating.
</p>

{callout}

<div class="card card--flat">
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>Doc Page</th><th>Doc Modified</th><th>Source File</th><th>Source Modified</th><th>Status</th><th>Action</th>
      </tr></thead>
      <tbody>{stale_rows}</tbody>
    </table>
  </div>
</div>
"""
    return _render(content, "Staleness", "stale")


# ═════════════════════════════════════════════════════════
# ROUTES — AUDIT (Doc quality check)
# ═════════════════════════════════════════════════════════


def _audit_doc_pages() -> list[dict]:
    """Analyze doc pages for quality issues using the inverted index — O(fast)."""
    docs = _get_docs()
    idx = docs.index_mgr.index
    inverted = idx.inverted

    code_names = _get_meaningful_code_names(docs)

    # Group sections by file
    by_file = defaultdict(list)
    for s in idx.sections.values():
        by_file[s.file_path].append(s)

    issues = []
    for file_path, secs in sorted(by_file.items()):
        fp = Path(file_path)
        try:
            rel = str(fp.relative_to(DOCS_DIR))
        except ValueError:
            rel = fp.name

        total_words = sum(len(s.content.split()) for s in secs)
        total_headings = len(secs)

        # Fast ref count via inverted index
        ref_count = _count_code_refs_for_file(docs, file_path, code_names)

        # Check for TODO/FIXME markers
        all_content = " ".join(s.content.lower() for s in secs)
        has_todos = any(marker in all_content for marker in ("todo", "fixme", "tbd", "placeholder", "xxx"))

        # Determine issue (priority order — first match wins)
        problem = None
        severity = "badge--ghost"

        if total_words < 20:
            problem = "Empty / stub"
            severity = "badge--error"
        elif total_words < 80 and total_headings <= 2:
            problem = "Too short"
            severity = "badge--warning"
        elif ref_count == 0:
            problem = "No code refs"
            severity = "badge--warning"
        elif has_todos:
            problem = "Has TODO/FIXME"
            severity = "badge--warning"
        elif total_words > 1500 and ref_count < 3:
            problem = "Bloat"
            severity = "badge--warning"
        elif total_headings == 1 and total_words > 100:
            problem = "No structure"
            severity = "badge--ghost"

        if problem:
            issues.append({
                "rel_path": rel,
                "full_path": file_path,
                "words": total_words,
                "headings": total_headings,
                "matched_code": ref_count,
                "problem": problem,
                "severity": severity,
            })

    return issues


@web_app.get("/audit", response_class=HTMLResponse)
async def audit_page():
    issues = _audit_doc_pages()

    rows = ""
    for item in issues:
        escaped_path = item["rel_path"].replace("\\", "\\\\").replace("'", "\\'")
        rows += f"""<tr>
          <td><a href="/view/{item["rel_path"]}">{item["rel_path"]}</a></td>
          <td>{item["words"]}</td>
          <td>{item["headings"]}</td>
          <td>{item["matched_code"]}</td>
          <td><span class="badge {item["severity"]}">{item["problem"]}</span></td>
          <td style="display:flex; gap:var(--space-2); flex-wrap:nowrap">
            <a href="/edit?page={item["rel_path"]}" class="btn btn--sm">Edit</a>
            <button class="btn btn--sm btn--danger"
              onclick="if(confirm('Delete {escaped_path}?')) deleteDoc('{escaped_path}', this)">Del</button>
            <button class="btn btn--sm"
              onclick="fixStaleDoc('{escaped_path}', '', this)">Fix</button>
          </td>
        </tr>"""

    if not rows:
        rows = '<tr><td colspan="6" style="text-align:center; padding: var(--space-5)">All docs look healthy!</td></tr>'

    empty_count = sum(1 for i in issues if "Empty" in i["problem"] or "short" in i["problem"])
    no_match = sum(1 for i in issues if "No code" in i["problem"])
    bloat = sum(1 for i in issues if "Bloat" in i["problem"])
    todos = sum(1 for i in issues if "TODO" in i["problem"])
    no_struct = sum(1 for i in issues if "structure" in i["problem"])

    content = f"""
<h1 class="page-title">Doc Audit</h1>
<p class="page-subtitle">
  Documentation pages with quality issues — empty, unlinked, bloated, or lacking structure.
</p>

<div class="grid grid-3" style="margin-bottom: var(--space-6)">
  <div class="card">
    <div class="stat">
      <div class="stat-value">{len(issues)}</div>
      <div class="stat-label">Total Issues</div>
    </div>
  </div>
  <div class="card">
    <div class="stat">
      <div class="stat-value" style="color: var(--error)">{empty_count}</div>
      <div class="stat-label">Empty / Stub</div>
    </div>
  </div>
  <div class="card">
    <div class="stat">
      <div class="stat-value" style="color: var(--warning)">{no_match}</div>
      <div class="stat-label">No Code Refs</div>
    </div>
  </div>
</div>
<div class="grid grid-3" style="margin-bottom: var(--space-6)">
  <div class="card">
    <div class="stat">
      <div class="stat-value">{bloat}</div>
      <div class="stat-label">Bloat</div>
    </div>
  </div>
  <div class="card">
    <div class="stat">
      <div class="stat-value" style="color: var(--warning)">{todos}</div>
      <div class="stat-label">Has TODO</div>
    </div>
  </div>
  <div class="card">
    <div class="stat">
      <div class="stat-value">{no_struct}</div>
      <div class="stat-label">No Structure</div>
    </div>
  </div>
</div>

<div class="card card--flat">
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>Page</th><th>Words</th><th>Sections</th><th>Code Refs</th><th>Issue</th><th>Actions</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>
"""
    return _render(content, "Audit", "audit")


@web_app.delete("/api/doc/{path:path}")
async def api_delete_doc(path: str):
    """Delete a doc page and remove from index."""
    target = DOCS_DIR / path
    if not target.exists():
        return JSONResponse({"ok": False, "error": f"Not found: {path}"}, status_code=404)
    target.unlink()
    try:
        _get_docs().index_mgr.remove_file(str(target))
        _get_docs().context.clear_cache()
    except Exception:
        pass
    return JSONResponse({"ok": True, "deleted": path})


# ═════════════════════════════════════════════════════════
# ROUTES — EDITOR
# ═════════════════════════════════════════════════════════


@web_app.get("/edit", response_class=HTMLResponse)
async def edit_form(page: str = ""):
    current_path = page or "index.md"
    current_content = ""
    target = DOCS_DIR / current_path
    if target.exists():
        current_content = target.read_text(encoding="utf-8", errors="ignore")

    # Escape for textarea
    escaped = current_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    content = f"""
<h1 class="page-title">Page Editor</h1>
<p class="page-subtitle">Edit markdown documentation directly. Changes are saved to disk immediately.</p>

<div class="card card--flat">
  <form method="POST" action="/edit">
    <label>File Path (relative to docs/)</label>
    <input type="text" name="path" value="{current_path}" style="margin-bottom: var(--space-5)">

    <label>Content (Markdown)</label>
    <textarea name="content" rows="30" style="margin-bottom: var(--space-5)">{escaped}</textarea>

    <div style="display: flex; gap: var(--space-3)">
      <button type="submit" class="btn btn--primary">Save Page</button>
      <a href="/browse" class="btn btn--ghost">Cancel</a>
    </div>
  </form>
</div>
"""
    return _render(content, "Editor", "edit")


@web_app.post("/edit", response_class=HTMLResponse)
async def edit_submit(request: Request):
    form = await request.form()
    path = form.get("path", "index.md")
    content = form.get("content", "")

    target = DOCS_DIR / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    # Re-index this file in the DocsSystem
    try:
        await _get_docs()._update_file(target)
        _get_docs().context.clear_cache()
    except Exception:
        pass

    saved_html = f"""
<h1 class="page-title">Saved</h1>
<div class="callout callout--ok">
  <strong>{path}</strong> saved successfully at {datetime.now().strftime('%H:%M:%S')}.
</div>
<div style="display: flex; gap: var(--space-3); margin-top: var(--space-5)">
  <a href="/view/{path}" class="btn btn--primary">View Page</a>
  <a href="/edit?page={path}" class="btn">Edit Again</a>
  <a href="/browse" class="btn btn--ghost">Browse All</a>
</div>
"""
    return _render(saved_html, "Saved", "edit")


# ═════════════════════════════════════════════════════════
# API ROUTES — JSON
# ═════════════════════════════════════════════════════════


@web_app.get("/api/coverage")
async def api_coverage():
    cov = _compute_coverage()
    return JSONResponse({
        "total_modules": cov["total_modules"],
        "documented": cov["documented"],
        "missing_count": len(cov["missing"]),
        "stale_count": len(cov["stale"]),
        "pages": cov["pages"],
        "coverage_pct": cov["pct"],
        "quality_score": cov["quality_score"],
        "missing": [{"name": m["name"], "path": m["rel_path"]} for m in cov["missing"]],
        "stale": cov["stale"],
    })


@web_app.get("/api/pages")
async def api_pages():
    docs = _get_doc_pages()
    return JSONResponse(docs)


@web_app.get("/api/modules")
async def api_modules():
    """Code elements from the index, grouped by file."""
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
    return JSONResponse(result)


@web_app.get("/api/search")
async def api_search(q: str = "", type: str = "all"):
    """Search the index — delegates to DocsSystem."""
    docs = _get_docs()
    results = {}
    if type in ("all", "docs"):
        sections = await docs.read(query=q, max_results=20)
        results["docs"] = sections.get("sections", [])
    if type in ("all", "code"):
        code = await docs.lookup_code(name=q, max_results=20)
        results["code"] = code.get("results", [])
    return JSONResponse(results)


@web_app.get("/api/suggestions")
async def api_suggestions():
    """Get doc improvement suggestions from the index."""
    docs = _get_docs()
    return JSONResponse(await docs.get_suggestions(max_suggestions=50))


@web_app.post("/api/sync")
async def api_sync():
    """Incremental sync via git change detection."""
    docs = _get_docs()
    return JSONResponse(await docs.sync())


@web_app.post("/api/build")
async def api_build():
    """Trigger MkDocs build."""
    if not MKDOCS_YML.exists():
        return JSONResponse({"ok": False, "error": "mkdocs.yml not found"}, status_code=404)
    try:
        proc = await asyncio.create_subprocess_exec(
            "mkdocs", "build", "--clean",
            cwd=str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        return JSONResponse({
            "ok": proc.returncode == 0,
            "stdout": stdout.decode(errors="ignore"),
            "stderr": stderr.decode(errors="ignore"),
        })
    except FileNotFoundError:
        return JSONResponse({"ok": False, "error": "mkdocs binary not found"}, status_code=500)
    except asyncio.TimeoutError:
        return JSONResponse({"ok": False, "error": "build timed out (60s)"}, status_code=504)


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

## OUTPUT FORMAT

Your response must be ONLY the final markdown document. No preamble, no "Here is the documentation", no ```markdown fences, no explanation after.

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
  - Source at `toolboxv2/utils/system/session.py` → link: `[Session](toolboxv2/utils/system/session.md)`
  - Source at `toolboxv2/mods/CloudM/` → link: `[CloudM](toolboxv2/mods/CloudM.md)`
  - Source at `flows/doc_server.py` → link: `[doc_server](flows/doc_server.md)`
- In Dependencies section: EVERY upstream module MUST be a markdown link to its doc path.
- In Used By section: EVERY downstream caller MUST be a markdown link to its doc path.
- In API Reference: link parameter types when they are classes from other modules:
  "Accepts a [`DocSection`](toolboxv2/utils/extras/mkdocs.md) instance"
- In Usage Guide: link to related modules the user might also need.
- The file paths are provided in the code data — derive the doc path by replacing the source extension with `.md`.

## ARCHITECTURE DIAGRAMS (when applicable)

If the module has 3+ classes or a clear data flow, include a mermaid diagram:

For class relationships:
```mermaid
classDiagram
    ParentClass <|-- ChildClass
    ChildClass --> DependencyClass : uses
```

For data/control flow:
```mermaid
flowchart LR
    Input --> ProcessorClass --> OutputClass
    ProcessorClass --> CacheLayer
```

Rules for diagrams:
- Only include classes/functions that actually exist in the provided code data.
- Keep diagrams small: max 8 nodes. If more, focus on the main flow.
- Use `flowchart LR` (left-to-right) for data flows, `classDiagram` for inheritance.
- Do NOT invent connections — only diagram relationships visible in the code/graph.

## KNOWN ISSUES / BUGS SECTION (when applicable)

If the code data reveals any of these patterns, add a `## Known Issues` section at the bottom:

- `except: pass` or `except Exception: pass` — swallowed errors
- `TODO`, `FIXME`, `HACK`, `XXX` in docstrings or comments
- Functions with no return type annotation and no docstring
- Very long functions (line_end - line_start > 100) with no docstring
- Circular-looking dependencies (A depends on B, B depends on A in the graph)

Format each issue as:
- **`function_name` (line X):** Brief factual description of the issue. No judgment, no fix suggestions.

Only include this section if there are actual issues found in the data. Do not invent issues.

## FOR UPDATES TO EXISTING DOCS

When updating an existing document:
1. Preserve the structure and any human-written context that is still accurate.
2. Update signatures that changed in the code.
3. Remove documentation for code elements that no longer exist.
4. Add new elements that exist in code but not in the doc.
5. Never mark anything as deprecated unless the code explicitly says so.
6. If the existing doc has custom sections (e.g. "Migration Guide"), keep them unless they reference removed code.
"""


# Agent singleton — built once, reused across requests
_doc_agent = None
_doc_agent_lock = asyncio.Lock()


async def _get_or_build_doc_agent():
    """Get or build the doc agent singleton. Thread-safe via asyncio.Lock."""
    global _doc_agent
    if _doc_agent is not None:
        return _doc_agent

    async with _doc_agent_lock:
        # Double-check after acquiring lock
        if _doc_agent is not None:
            return _doc_agent

        if _tb_app is None:
            raise RuntimeError("ToolBoxV2 app not available — start with: tb -m isaa -f doc_server")

        isaa = _tb_app.get_mod("isaa")
        if isaa is None:
            raise RuntimeError("ISAA module not loaded — start with: tb -m isaa -f doc_server")

        await isaa.init_isaa(name="doc_agent")

        builder = isaa.get_agent_builder(
            "doc_agent",
            add_base_tools=False,
            with_dangerous_shell=False,
        )
        builder.config.system_message = DOC_AGENT_PROMPT

        await isaa.register_agent(builder)
        _doc_agent = await isaa.get_agent("doc_agent")
        return _doc_agent


async def _run_doc_agent(file_path: str, update_existing: str = None) -> dict:
    """
    Generate or update documentation for a source file.

    Pipeline:
    1. lookup_code(file_path, include_code=True) → actual signatures + code
    2. get_task_context([file_path], intent) → upstream/downstream graph
    3. If update_existing: read existing doc content
    4. Build grounded prompt with ONLY real data
    5. Call ISAA agent.a_run → get markdown
    6. Write to docs/<name>.md and re-index
    """
    docs = _get_docs()
    fp = Path(file_path)

    # 1. Get all code elements for this file from the index
    code_result = await docs.lookup_code(
        file_path=file_path,
        include_code=True,
        max_results=100,
    )
    elements = code_result.get("results", [])
    if not elements:
        code_result = await docs.lookup_code(
            file_path=fp.name,
            include_code=True,
            max_results=100,
        )
        elements = code_result.get("results", [])

    if not elements:
        return {"ok": False, "error": f"No code elements found in index for {file_path}"}

    # 2. Get context graph (upstream/downstream dependencies)
    try:
        ctx = await docs.get_task_context(
            files=[file_path],
            intent="Generate comprehensive module documentation",
        )
        context_graph = ctx.get("result", {}).get("context_graph", {})
    except Exception:
        context_graph = {"upstream_dependencies": [], "downstream_usages": []}

    # 3. If updating, read existing doc
    existing_content = ""
    if update_existing:
        existing_path = DOCS_DIR / update_existing
        if existing_path.exists():
            existing_content = existing_path.read_text(encoding="utf-8", errors="ignore")

    # 4. Build the grounded prompt — ONLY verified data from the index
    code_section = []
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
        code_section.append(entry)

    upstream = context_graph.get("upstream_dependencies", [])
    downstream = context_graph.get("downstream_usages", [])

    user_prompt = f"""Document this file: `{file_path}`

## Verified Code Elements (from AST index — this is ground truth)

{chr(10).join(code_section)}

## Dependency Graph (from index)

### Upstream (what this module uses)
{chr(10).join(f"- `{u.get('name', '?')}` ({u.get('type', '?')}) in `{u.get('file', '?')}`" for u in upstream) if upstream else "No indexed upstream dependencies."}

### Downstream (what uses this module)
{chr(10).join(f"- `{d.get('name', '?')}` in `{d.get('file', '?')}`" for d in downstream) if downstream else "No indexed downstream usages."}
"""

    if existing_content:
        user_prompt += f"""
## Existing Documentation (to update — preserve accurate content, fix stale parts)

```markdown
{existing_content[:5000]}
```

Update this documentation to match the current code above.
Remove references to code that no longer exists. Add new elements. Fix changed signatures.
Keep any human-written context sections that are still accurate.
"""
    else:
        user_prompt += """
Create a new documentation page for this module following the structure in your system prompt.
Your COMPLETE response must be the final markdown document — nothing else.
"""

    # 5. Call ISAA agent
    try:
        agent = await _get_or_build_doc_agent()

        result = await agent.a_run(
            query=user_prompt,
            session_id="doc_agent_session",
            human_online=False,
        )

        if not result or not isinstance(result, str):
            return {"ok": False, "error": "Agent returned empty result"}

        # Clean up: strip markdown fences if the model wrapped it
        markdown_content = result.strip()
        if markdown_content.startswith("```markdown"):
            markdown_content = markdown_content[len("```markdown"):].strip()
        if markdown_content.startswith("```"):
            markdown_content = markdown_content[3:].strip()
        if markdown_content.endswith("```"):
            markdown_content = markdown_content[:-3].strip()

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": f"ISAA agent failed: {e}"}

    # 6. Write to disk and re-index
    if update_existing:
        out_path = DOCS_DIR / update_existing
    else:
        # Determine output path: docs/<dir>/<stem>.md
        try:
            rel = fp.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = fp
        out_path = DOCS_DIR / rel.with_suffix(".md")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown_content, encoding="utf-8")

    # Re-index
    try:
        await docs._update_file(out_path)
        docs.context.clear_cache()
    except Exception:
        pass

    return {
        "ok": True,
        "path": str(out_path.relative_to(DOCS_DIR)),
        "elements_used": len(elements),
        "words_generated": len(markdown_content.split()),
        "mode": "update" if update_existing else "create",
    }


@web_app.post("/api/generate-doc")
async def api_generate_doc(request: Request):
    """Generate or update documentation for a source file using ISAA."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON body"}, status_code=400)

    file_path = body.get("file_path")
    update_existing = body.get("update_existing")

    if not file_path:
        return JSONResponse({"ok": False, "error": "file_path required"}, status_code=400)

    result = await _run_doc_agent(file_path, update_existing)
    status = 200 if result["ok"] else 500
    return JSONResponse(result, status_code=status)


# ═════════════════════════════════════════════════════════
# FLOW ENTRY — async uvicorn
# ═════════════════════════════════════════════════════════


async def run(app, args):
    """Flow entry — starts uvicorn. DocsSystem init happens via lifespan."""
    import uvicorn

    global _tb_app
    _tb_app = app

    host = getattr(args, "host", None) or "0.0.0.0"
    port = getattr(args, "port", None) or 8088

    print(f"📚 Doc Server → http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}")

    config = uvicorn.Config(
        web_app,
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    task = asyncio.create_task(server.serve())

    try:
        await task
    except (asyncio.CancelledError, KeyboardInterrupt):
        server.should_exit = True
        await task
