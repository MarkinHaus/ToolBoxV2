"""
toolboxv2/clis/cli_mkdocs.py
─────────────────────────────
CLI for the MkDocs Documentation System.

Entry: `tb mkdocs <command> [options]`

Commands:
  index     Build or rebuild the documentation index
  sync      Update index from git changes
  search    Search docs and/or code elements
  lookup    Look up specific code elements by name/type/file
  gaps      Find undocumented code (missing docs audit)
  export    Export DocMap (inventory + relationships) to HTML or MD
  inventory Show project inventory (what's here)
  map       Show relationship map (how components connect)
  clean     Remove index file and caches
  info      Show index stats
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional


# ─── Path Resolution ────────────────────────────────────

def _resolve_paths(args) -> tuple[Path, Path]:
    """Resolve project_root and docs_root from CLI args."""
    # --path: project root
    if args.path and args.path != ".":
        project_root = Path(args.path).resolve()
    elif args.path == ".":
        project_root = Path.cwd()
    else:
        # Default: tb root
        try:
            from toolboxv2 import tb_root_dir
            project_root = tb_root_dir.parent
        except ImportError:
            project_root = Path.cwd()

    # --docs-path: docs location (relative to project_root or absolute)
    if args.docs_path:
        dp = Path(args.docs_path)
        docs_root = dp if dp.is_absolute() else (project_root / dp).resolve()
    else:
        docs_root = project_root / "docs"

    return project_root, docs_root


def _build_system(args):
    """Create DocsSystem + patch docmap."""
    from toolboxv2.utils.extras.mkdocs import DocsSystem
    project_root, docs_root = _resolve_paths(args)

    include = args.include.split(",") if hasattr(args, "include") and args.include else None

    system = DocsSystem(
        project_root=project_root,
        docs_root=docs_root,
        include_dirs=include,
    )

    # Patch docmap methods
    try:
        from toolboxv2.utils.extras.mkdocs_docmap import patch_docmap
        patch_docmap(system)
    except ImportError:
        pass  # docmap not installed — inventory/map/export won't work

    return system


# ─── Commands ───────────────────────────────────────────

async def cmd_index(args):
    """Build or rebuild the documentation index."""
    system = _build_system(args)
    force = getattr(args, "force", False)

    print(f"Project: {system.project_root}")
    print(f"Docs:    {system.docs_root}")
    print(f"Mode:    {'full rebuild' if force else 'load or build'}")
    print()

    result = await system.initialize(force_rebuild=force, show_tqdm=True)

    print(f"\nStatus:   {result['status']}")
    print(f"Sections: {result['sections']}")
    print(f"Elements: {result['elements']}")
    print(f"Time:     {result['time_ms']:.0f}ms")

    if result["status"] == "rebuilt":
        await system.index_mgr.save(force=True)
        print(f"Index saved to {system.index_mgr.index_path}")


async def cmd_sync(args):
    """Sync index with filesystem/git changes."""
    system = _build_system(args)
    await system.initialize()

    result = await system.sync()

    print(f"Changes detected: {result['changes_detected']}")
    print(f"Files updated:    {result['files_updated']}")
    print(f"Time:             {result['time_ms']:.0f}ms")

    if result["files_updated"] > 0:
        await system.index_mgr.save()
        print("Index saved.")


async def cmd_search(args):
    """Search documentation sections and/or code."""
    system = _build_system(args)
    await system.initialize()

    query = args.query
    target = getattr(args, "type", "all")

    if target in ("all", "docs"):
        result = await system.read(query=query, max_results=args.limit)
        sections = result.get("sections", [])
        if sections:
            print(f"── Documentation ({len(sections)} results) ──\n")
            for s in sections:
                tags = " ".join(f"#{t}" for t in s.get("tags", []))
                print(f"  [{s.get('level', 0)}] {s.get('title', '?')}")
                print(f"      {s.get('file', '?')}  {tags}")
                snippet = s.get("content", "")[:120].replace("\n", " ")
                print(f"      {snippet}...")
                print()
        elif target == "docs":
            print("No documentation results.\n")

    if target in ("all", "code"):
        result = await system.lookup_code(name=query, max_results=args.limit)
        elements = result.get("results", [])
        if elements:
            print(f"── Code ({len(elements)} results) ──\n")
            for e in elements:
                parent = f"{e['parent']}." if e.get("parent") else ""
                print(f"  [{e['type']}] {parent}{e['name']}")
                print(f"      {e['signature']}")
                print(f"      {e['file']}:{e['lines'][0]}-{e['lines'][1]}  ({e.get('language', '?')})")
                if e.get("docstring"):
                    print(f"      {e['docstring'][:100]}...")
                print()
        elif target == "code":
            print("No code results.\n")


async def cmd_lookup(args):
    """Look up code elements by name, type, file, or language."""
    system = _build_system(args)
    await system.initialize()

    result = await system.lookup_code(
        name=args.name or None,
        element_type=args.element_type or None,
        file_path=args.file or None,
        language=args.language or None,
        include_code=args.code,
        max_results=args.limit,
    )

    elements = result.get("results", [])
    print(f"{len(elements)} elements found ({result.get('time_ms', 0):.0f}ms)\n")

    for e in elements:
        parent = f"{e['parent']}." if e.get("parent") else ""
        print(f"  [{e['type']:8}] {parent}{e['name']}")
        print(f"             {e['signature']}")
        print(f"             {e['file']}:{e['lines'][0]}-{e['lines'][1]}")
        if e.get("docstring"):
            print(f"             doc: {e['docstring'][:120]}")
        if args.code and e.get("code"):
            print(f"             ─── code ───")
            for line in e["code"].split("\n")[:20]:
                print(f"             {line}")
            if e["code"].count("\n") > 20:
                print(f"             ... ({e['code'].count(chr(10)) - 20} more lines)")
        print()


async def cmd_gaps(args):
    """Find undocumented code elements (missing docs audit)."""
    system = _build_system(args)
    await system.initialize()

    result = await system.get_suggestions(max_suggestions=args.limit)
    suggestions = result.get("suggestions", [])

    missing = [s for s in suggestions if s["type"] == "missing_docs"]
    unclear = [s for s in suggestions if s["type"] == "unclear_section"]

    if missing:
        print(f"── Missing Documentation ({len(missing)}) ──\n")
        for s in missing:
            prio = s.get("priority", "?")
            marker = "!" if prio == "high" else "·"
            print(f"  {marker} [{s['element_type']:8}] {s['element']}  ({s.get('language', '?')})")
            print(f"    {s['file']}")
        print()

    if unclear:
        print(f"── Unclear Sections ({len(unclear)}) ──\n")
        for s in unclear:
            print(f"  ? {s['title']}")
            print(f"    {s.get('section_id', '?')}")
        print()

    if not missing and not unclear:
        print("No gaps found — all public elements documented.")

    print(f"\nTotal: {result.get('total', 0)} suggestions")


async def cmd_inventory(args):
    """Show project inventory — what files, classes, functions exist."""
    system = _build_system(args)
    await system.initialize()

    if not hasattr(system, "generate_inventory"):
        print("Error: mkdocs_docmap not installed. Cannot generate inventory.")
        return

    result = await system.generate_inventory(
        focus_dirs=args.focus.split(",") if args.focus else None,
        max_classes_per_file=args.max_classes,
        max_methods_per_class=args.max_methods,
        include_functions=not args.no_functions,
        format_type="markdown",
    )

    print(result.get("content", "No inventory generated."))
    print(f"\n({result['file_count']} files, {result['time_ms']:.0f}ms)")


async def cmd_map(args):
    """Show relationship map — how components connect."""
    system = _build_system(args)
    await system.initialize()

    if not hasattr(system, "generate_relationship_map"):
        print("Error: mkdocs_docmap not installed. Cannot generate map.")
        return

    result = await system.generate_relationship_map(
        focus_dirs=args.focus.split(",") if args.focus else None,
        focus_classes=args.classes.split(",") if args.classes else None,
        max_nodes=args.max_nodes,
        show_methods=args.methods,
        format_type="markdown",
    )

    print(result.get("content", "No relationships found."))
    print(f"\n({result['node_count']} nodes, {result['edge_count']} edges, {result['time_ms']:.0f}ms)")


async def cmd_export(args):
    """Export DocMap to HTML or MD file."""
    system = _build_system(args)
    await system.initialize()

    if not hasattr(system, "export_docmap"):
        print("Error: mkdocs_docmap not installed. Cannot export.")
        return

    fmt = args.format
    output = args.output
    if not output:
        output = f"docmap.{fmt}"

    result = await system.export_docmap(
        output_path=output,
        format_type=fmt,
        focus_dirs=args.focus.split(",") if args.focus else None,
        max_classes_per_file=args.max_classes,
        max_methods_per_class=args.max_methods,
        max_nodes=args.max_nodes,
        title=args.title or None,
    )

    print(f"Exported: {result.get('output_path', output)}")
    print(f"Format:   {fmt}")
    print(f"Files:    {result['inventory_files']}")
    print(f"Nodes:    {result['relationship_nodes']}")
    print(f"Edges:    {result['relationship_edges']}")
    print(f"Time:     {result['time_ms']:.0f}ms")


async def cmd_clean(args):
    """Remove index file and clear caches."""
    system = _build_system(args)
    idx_path = system.index_mgr.index_path

    if idx_path.exists():
        idx_path.unlink()
        print(f"Deleted: {idx_path}")
    else:
        print(f"No index at {idx_path}")

    system.doc_parser.clear_cache()
    system.code_analyzer.clear_cache()
    system.jsts_analyzer.clear_cache()
    system.scanner.clear_cache()
    print("Caches cleared.")


async def cmd_info(args):
    """Show index stats."""
    system = _build_system(args)

    idx_path = system.index_mgr.index_path
    if not idx_path.exists():
        print(f"No index at {idx_path}")
        print("Run: tb mkdocs index")
        return

    await system.initialize()
    idx = system.index_mgr.index

    print(f"Index:      {idx_path}")
    print(f"Version:    {idx.version}")
    print(f"Sections:   {len(idx.sections)}")
    print(f"Elements:   {len(idx.code_elements)}")
    print(f"Files:      {len(idx.file_hashes)}")
    print(f"Keywords:   {len(idx.inverted.keyword_to_sections)}")
    print(f"Last commit:{idx.last_git_commit or 'N/A'}")

    import datetime
    if idx.last_indexed:
        ts = datetime.datetime.fromtimestamp(idx.last_indexed).strftime("%Y-%m-%d %H:%M:%S")
        print(f"Last indexed: {ts}")

    # Size on disk
    size = idx_path.stat().st_size
    if size > 1_000_000:
        print(f"Disk:       {size / 1_000_000:.1f} MB")
    else:
        print(f"Disk:       {size / 1_000:.1f} KB")

    # Breakdown by language
    langs = {}
    for e in idx.code_elements.values():
        langs[e.language] = langs.get(e.language, 0) + 1
    if langs:
        parts = ", ".join(f"{l}: {c}" for l, c in sorted(langs.items()))
        print(f"Languages:  {parts}")

    # Breakdown by type
    types = {}
    for e in idx.code_elements.values():
        types[e.element_type] = types.get(e.element_type, 0) + 1
    if types:
        parts = ", ".join(f"{t}: {c}" for t, c in sorted(types.items()))
        print(f"Types:      {parts}")


# ─── Argument Parser ────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tb mkdocs",
        description="MkDocs Documentation System CLI",
    )
    # Global args
    p.add_argument("--path", default=None,
                   help="Project root. '.' = cwd, omit = tb root")
    p.add_argument("--docs-path", default=None,
                   help="Docs directory (relative to --path or absolute). Default: <path>/docs")
    p.add_argument("--include", default=None,
                   help="Comma-separated dirs to include (e.g. toolboxv2,flows)")

    sub = p.add_subparsers(dest="command", help="Command")

    # index
    s = sub.add_parser("index", help="Build or rebuild the documentation index")
    s.add_argument("--force", "-f", action="store_true", help="Force full rebuild")

    # sync
    sub.add_parser("sync", help="Update index from git changes")

    # search
    s = sub.add_parser("search", help="Search docs and/or code")
    s.add_argument("query", help="Search query")
    s.add_argument("--type", "-t", choices=["all", "docs", "code"], default="all")
    s.add_argument("--limit", "-n", type=int, default=15)

    # lookup
    s = sub.add_parser("lookup", help="Look up code elements")
    s.add_argument("--name", "-n", default="", help="Element name")
    s.add_argument("--element-type", "-t", default="",
                   choices=["", "class", "function", "method", "interface", "type"])
    s.add_argument("--file", "-f", default="", help="Filter by file path")
    s.add_argument("--language", "-l", default="", choices=["", "python", "javascript", "typescript"])
    s.add_argument("--code", "-c", action="store_true", help="Include source code")
    s.add_argument("--limit", type=int, default=15)

    # gaps
    s = sub.add_parser("gaps", help="Find undocumented code")
    s.add_argument("--limit", "-n", type=int, default=50)

    # inventory
    s = sub.add_parser("inventory", help="Project inventory (what's here)")
    s.add_argument("--focus", default="", help="Comma-separated focus dirs")
    s.add_argument("--max-classes", type=int, default=5)
    s.add_argument("--max-methods", type=int, default=3)
    s.add_argument("--no-functions", action="store_true")

    # map
    s = sub.add_parser("map", help="Relationship map (how components connect)")
    s.add_argument("--focus", default="", help="Comma-separated focus dirs")
    s.add_argument("--classes", default="", help="Comma-separated focus classes")
    s.add_argument("--max-nodes", type=int, default=40)
    s.add_argument("--methods", action="store_true", help="Show method-level edges")

    # export
    s = sub.add_parser("export", help="Export DocMap to file")
    s.add_argument("--format", choices=["html", "md"], default="html")
    s.add_argument("--output", "-o", default="", help="Output path (default: docmap.<format>)")
    s.add_argument("--focus", default="", help="Comma-separated focus dirs")
    s.add_argument("--max-classes", type=int, default=5)
    s.add_argument("--max-methods", type=int, default=3)
    s.add_argument("--max-nodes", type=int, default=40)
    s.add_argument("--title", default="", help="Document title")

    # clean
    sub.add_parser("clean", help="Remove index and clear caches")

    # info
    sub.add_parser("info", help="Show index stats")

    return p


# ─── Runner ─────────────────────────────────────────────

COMMANDS = {
    "index": cmd_index,
    "sync": cmd_sync,
    "search": cmd_search,
    "lookup": cmd_lookup,
    "gaps": cmd_gaps,
    "inventory": cmd_inventory,
    "map": cmd_map,
    "export": cmd_export,
    "clean": cmd_clean,
    "info": cmd_info,
}


async def run(app, _):
    """
    Flow entry point for ToolBoxV2.
    Called via: tb mkdocs <command> [options]
    """
    parser = build_parser()
    argv = ["tb", "mkdocs"]+sys.argv[1:]
    try:
        idx = argv.index("mkdocs")
        cli_args = parser.parse_args(argv[idx + 1:])
    except ValueError:
        cli_args = parser.parse_args([])

    if not cli_args.command:
        parser.print_help()
        return

    handler = COMMANDS.get(cli_args.command)
    if not handler:
        print(f"Unknown command: {cli_args.command}")
        parser.print_help()
        return

    await handler(cli_args)
