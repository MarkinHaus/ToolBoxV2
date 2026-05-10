"""
Profiler

- AST-based import analysis (no more BFS through importlib)
- Package Blame: parses actual import statements
- Grouped stats
- Interactive HTML force-graph with real edges
"""

import ast
import asyncio
import cProfile
import pstats
import os
import re
import json
import sys
from collections import defaultdict
from functools import wraps
from pathlib import Path


def profile_code(sort_by="cumulative", top_n=30,
                 module_filter=None, graph=True, graph_file="import_graph.html",
                 function_graph=True, function_graph_file="function_graph.html",
                 group_depth=2, min_time=0.01, blame_threshold=0.1, override=False):
    if os.getenv("PROFILING", "false") != "true":
        if not override:
            return lambda x: x

    def _generate_reports(profiler):
        stats = pstats.Stats(profiler)
        project_root = os.getcwd()

        # 1) Grouped
        groups = _group_by_package(stats.stats, project_root, group_depth)
        _print_grouped(groups)

        # 2) Project Module Imports
        norm_root = os.path.normpath(project_root)
        import_entries = {
            k: v for k, v in stats.stats.items()
            if k[0].startswith(norm_root) and k[2] == "<module>"
        }
        if import_entries:
            _print_section("MODULE IMPORT TIMES", sort_by,
                           _make_filtered_stats(profiler, import_entries), top_n)

        # 3) Package Blame (AST-based)
        _package_blame_ast(stats.stats, project_root, blame_threshold)

        # 4) Drill-Down
        if module_filter:
            for mod_name in module_filter:
                entries = {k: v for k, v in stats.stats.items() if mod_name in k[0]}
                if entries:
                    _print_section(f"DRILL-DOWN: *{mod_name}*", sort_by,
                                   _make_filtered_stats(profiler, entries), top_n)

        # 5) Import Graph
        if graph:
            html = _build_html_graph(stats.stats, project_root, min_time)
            out = os.path.join(project_root, graph_file)
            with open(out, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"\n  Import Graph: {out}")

        # 6) Function Call Graph
        if function_graph:
            html = _build_function_graph(stats.stats, project_root, min_time)
            out = os.path.join(project_root, function_graph_file)
            with open(out, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"  Function Graph: {out}")
            print(f"  open in Browser\n")

    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                sys.setprofile(None)
                profiler = cProfile.Profile()
                profiler.enable()
                try:
                    result = await func(*args, **kwargs)
                finally:
                    profiler.disable()
                _generate_reports(profiler)
                return result
            return wrapper
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                sys.setprofile(None)
                profiler = cProfile.Profile()
                profiler.enable()
                try:
                    result = func(*args, **kwargs)
                finally:
                    profiler.disable()
                _generate_reports(profiler)
                return result
            return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════
#  AST Import Parsing
# ═══════════════════════════════════════════════════════════════

def _parse_imports(filepath):
    """Parse a Python file and return dict of {package_name: [line_numbers]}."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            tree = ast.parse(f.read(), filename=filepath)
    except (SyntaxError, OSError, ValueError):
        return {}

    imports = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.name.split(".")[0]].append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                imports[node.module.split(".")[0]].append(node.lineno)
    return dict(imports)


def _resolve_import_to_file(imp_name, node_ids, norm_root):
    """
    Given an import name like 'litellm' or 'toolboxv2',
    find matching node file paths.
    """
    matches = set()
    for filepath, nid in node_ids.items():
        if filepath.startswith(norm_root):
            # Project file: check if import matches its module path
            rel = os.path.relpath(filepath, norm_root)
            dotted = rel.replace(os.sep, ".").replace(".__init__.py", "").replace(".py", "")
            top = dotted.split(".")[0]
            if imp_name == top:
                matches.add(nid)
        else:
            # External: match package name
            pkg = _get_package_name(filepath)
            if pkg and pkg == imp_name:
                matches.add(nid)
    return matches


# ═══════════════════════════════════════════════════════════════
#  Package Blame (AST-based)
# ═══════════════════════════════════════════════════════════════

def _package_blame_ast(all_stats, project_root, threshold):
    """
    For each expensive external package:
    1) Parse project files to find direct imports WITH line numbers
    2) Collect ALL ext package files for transitive dep resolution
    3) Multi-level transitive: project -> A -> B -> target
    """
    norm_root = os.path.normpath(project_root)

    # Collect ext package cumtimes and ALL their files
    pkg_times = defaultdict(float)
    pkg_all_files = defaultdict(list)  # pkg -> [all .py files]
    for (fn, ln, func), (cc, nc, tt, ct, callers) in all_stats.items():
        if func != "<module>":
            continue
        norm_fn = os.path.normpath(fn)
        if norm_fn.startswith(norm_root):
            continue
        pkg = _get_package_name(norm_fn)
        if not pkg:
            continue
        pkg_times[pkg] = max(pkg_times[pkg], ct)
        if os.path.exists(norm_fn):
            pkg_all_files[pkg].append(norm_fn)

    # Collect project files
    project_files = []
    for (fn, ln, func), (cc, nc, tt, ct, callers) in all_stats.items():
        if func == "<module>":
            norm_fn = os.path.normpath(fn)
            if norm_fn.startswith(norm_root) and os.path.exists(norm_fn):
                project_files.append(norm_fn)

    # Parse project files: {filepath: {pkg: [lines]}}
    file_imports = {}
    for pf in project_files:
        file_imports[pf] = _parse_imports(pf)

    # Build ext->ext dep map by scanning ALL files of each package
    ext_deps = defaultdict(set)  # pkg -> set of ext pkgs it imports
    for pkg, files in pkg_all_files.items():
        for f in files[:20]:  # limit per package for speed
            imps = _parse_imports(f)
            for imp in imps:
                if imp in pkg_times and imp != pkg:
                    ext_deps[pkg].add(imp)

    # Build transitive closure (up to 3 levels)
    ext_deps_deep = defaultdict(set)
    for pkg in ext_deps:
        visited = set()
        queue = list(ext_deps[pkg])
        while queue:
            dep = queue.pop(0)
            if dep in visited or dep == pkg:
                continue
            visited.add(dep)
            ext_deps_deep[pkg].add(dep)
            for sub in ext_deps.get(dep, set()):
                if sub not in visited:
                    queue.append(sub)

    # Sort packages by time
    sorted_pkgs = sorted(pkg_times.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*80}")
    print(f"  PACKAGE BLAME — Welche eigenen Dateien laden teure Packages?")
    print(f"{'='*80}")

    for pkg_name, pkg_time in sorted_pkgs:
        if pkg_time < threshold:
            continue

        # Direct: project files that `import <pkg>`
        direct = []
        for pf, imps in file_imports.items():
            if pkg_name in imps:
                lines = imps[pkg_name]
                direct.append((pf, lines))

        # Indirect: project file imports X, and X (transitively) imports pkg
        indirect = []  # (project_file, lines_of_via, via_pkg, chain)
        for pf, imps in file_imports.items():
            if pkg_name in imps:
                continue
            for imp, lines in imps.items():
                if imp not in pkg_times:
                    continue
                # Direct dep of imp?
                if pkg_name in ext_deps.get(imp, set()):
                    indirect.append((pf, lines, imp, [imp]))
                    break
                # Transitive dep?
                if pkg_name in ext_deps_deep.get(imp, set()):
                    # Reconstruct short chain
                    chain = _find_chain(imp, pkg_name, ext_deps, max_depth=3)
                    indirect.append((pf, lines, imp, chain))
                    break

        if direct or indirect:
            print(f"\n  {pkg_name} ({pkg_time:.3f}s)")

            if direct:
                print(f"    DIRECT:")
                for pf, lines in sorted(direct, key=lambda x: x[0])[:15]:
                    label = _short_label(pf, norm_root)
                    line_str = ",".join(str(l) for l in lines[:3])
                    print(f"      {label}:{line_str}")
                if len(direct) > 15:
                    print(f"      ... +{len(direct)-15} more")

            if indirect:
                print(f"    INDIRECT:")
                seen = set()
                for pf, lines, via, chain in sorted(indirect, key=lambda x: x[0])[:15]:
                    label = _short_label(pf, norm_root)
                    if label in seen:
                        continue
                    seen.add(label)
                    line_str = ",".join(str(l) for l in lines[:3])
                    chain_str = " -> ".join(chain + [pkg_name])
                    print(f"      {label}:{line_str} -> {chain_str}")
        else:
            print(f"\n  {pkg_name} ({pkg_time:.3f}s)")
            print(f"    (transitiv — kein Projekt-Import gefunden)")


def _find_chain(start_pkg, target_pkg, ext_deps, max_depth=3):
    """Find shortest chain from start_pkg to target_pkg through ext_deps."""
    queue = [(start_pkg, [start_pkg])]
    visited = {start_pkg}
    while queue:
        pkg, path = queue.pop(0)
        if len(path) > max_depth:
            continue
        for dep in ext_deps.get(pkg, set()):
            if dep == target_pkg:
                return path
            if dep not in visited:
                visited.add(dep)
                queue.append((dep, path + [dep]))
    return [start_pkg]


# ═══════════════════════════════════════════════════════════════
#  Build Edges (AST-based)
# ═══════════════════════════════════════════════════════════════

def _build_edges_ast(node_ids, norm_root):
    """Build import edges by parsing actual source files."""
    pkg_to_nodes = defaultdict(set)
    for filepath, nid in node_ids.items():
        if filepath.startswith(norm_root):
            rel = os.path.relpath(filepath, norm_root)
            dotted = rel.replace(os.sep, ".").replace(".__init__.py", "").replace(".py", "")
            parts = dotted.split(".")
            for i in range(len(parts)):
                pkg_to_nodes[".".join(parts[:i+1])].add(nid)
        else:
            pkg = _get_package_name(filepath)
            if pkg:
                pkg_to_nodes[pkg].add(nid)

    edges = []
    edge_set = set()

    for filepath, nid in node_ids.items():
        if not os.path.exists(filepath):
            continue
        imports = _parse_imports(filepath)  # returns {pkg: [lines]}
        for imp in imports:
            targets = pkg_to_nodes.get(imp, set())
            for tid in targets:
                if tid != nid:
                    ek = (nid, tid)
                    if ek not in edge_set:
                        edge_set.add(ek)
                        edges.append({"source": nid, "target": tid})

    return edges


# ═══════════════════════════════════════════════════════════════
#  Gruppierung
# ═══════════════════════════════════════════════════════════════

def _group_by_package(all_stats, project_root, depth):
    groups = defaultdict(lambda: {"tottime": 0.0, "cumtime": 0.0, "calls": 0, "entries": []})
    norm_root = os.path.normpath(project_root)
    for (filename, lineno, func_name), (cc, nc, tt, ct, callers) in all_stats.items():
        norm_file = os.path.normpath(filename)
        if not norm_file.startswith(norm_root):
            sp = re.search(r'site-packages[/\\]([^/\\]+)', norm_file)
            group_key = f"[ext] {sp.group(1)}" if sp else "[stdlib]"
        else:
            rel = os.path.relpath(norm_file, norm_root)
            parts = Path(rel).parts
            meaningful = [p for p in parts if p != "__init__.py" and not p.endswith(".py")]
            if not meaningful:
                meaningful = [parts[0] if parts else "root"]
            group_key = "/".join(meaningful[:depth])
        groups[group_key]["tottime"] += tt
        groups[group_key]["cumtime"] += ct
        groups[group_key]["calls"] += nc
        groups[group_key]["entries"].append((filename, lineno, func_name, tt, ct, nc))
    return groups


def _print_grouped(groups):
    print(f"\n{'='*80}")
    print(f"  GROUPED BY PACKAGE")
    print(f"{'='*80}")
    print(f"  {'Group':<45} {'tottime':>8} {'cumtime':>8} {'calls':>8}")
    print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8}")
    sg = sorted(groups.items(), key=lambda x: x[1]["cumtime"], reverse=True)
    for name, d in sg[:30]:
        if d["cumtime"] < 0.005:
            continue
        print(f"  {name:<45} {d['tottime']:>8.3f} {d['cumtime']:>8.3f} {d['calls']:>8}")

    print(f"\n  TOP ENTRIES PER GROUP (cumtime > 0.05s)")
    print(f"  {'-'*75}")
    for name, d in sg[:15]:
        hot = [(fn, ln, nm, tt, ct, nc) for fn, ln, nm, tt, ct, nc in d["entries"] if ct > 0.05]
        if not hot:
            continue
        hot.sort(key=lambda x: x[4], reverse=True)
        print(f"\n  +- {name} (total: {d['cumtime']:.3f}s)")
        for fn, ln, nm, tt, ct, nc in hot[:5]:
            print(f"  |  {ct:>7.3f}s  {_shorten_path(fn)}:{ln}({nm})")
        print(f"  +--")


# ═══════════════════════════════════════════════════════════════
#  Interactive HTML Graph
# ═══════════════════════════════════════════════════════════════

def _build_html_graph(all_stats, project_root, min_time):
    norm_root = os.path.normpath(project_root)
    nodes = []
    node_ids = {}

    for (fn, ln, func), (cc, nc, tt, ct, callers) in all_stats.items():
        if func != "<module>" or ct < min_time:
            continue
        norm_fn = os.path.normpath(fn)
        if norm_fn in node_ids:
            continue
        nid = len(nodes)
        node_ids[norm_fn] = nid
        is_proj = norm_fn.startswith(norm_root)
        if is_proj:
            rel = os.path.relpath(norm_fn, norm_root)
            parts = Path(rel).parts
            group = "/".join(parts[:2]) if len(parts) > 1 else parts[0]
        else:
            sp = re.search(r'site-packages[/\\]([^/\\]+)', norm_fn)
            group = sp.group(1) if sp else "stdlib"
        nodes.append({
            "id": nid, "label": _short_label(norm_fn, norm_root),
            "cumtime": round(ct, 3), "is_project": is_proj, "group": group,
        })

    # AST-based edges
    edges = _build_edges_ast(node_ids, norm_root)

    gd = json.dumps({"nodes": nodes, "edges": edges})
    n_nodes = len(nodes)
    n_edges = len(edges)

    return f'''<!DOCTYPE html><html><head><meta charset="utf-8"><title>Import Graph</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0f172a;font-family:system-ui,sans-serif;overflow:hidden}}
#gc{{position:relative;width:100vw;height:100vh}}
canvas{{display:block;width:100%;height:100%}}
.tip{{position:absolute;background:#1e293b;color:#e2e8f0;padding:8px 12px;
  border-radius:6px;font-size:13px;pointer-events:none;opacity:0;
  border:1px solid #475569;box-shadow:0 4px 12px rgba(0,0,0,0.5);max-width:300px}}
.ctrl{{position:fixed;top:12px;left:12px;background:#1e293b;
  padding:12px 16px;border-radius:8px;color:#94a3b8;font-size:12px;
  border:1px solid #334155;z-index:10;max-width:280px}}
.ctrl h3{{color:#e2e8f0;margin-bottom:6px;font-size:14px}}
.ctrl label{{display:block;margin:4px 0;cursor:pointer}}
.ctrl input[type=range]{{width:100%}}
.leg{{position:fixed;bottom:12px;left:12px;background:#1e293b;
  padding:10px 14px;border-radius:8px;color:#94a3b8;font-size:12px;
  border:1px solid #334155;z-index:10}}
.li{{display:flex;align-items:center;gap:6px;margin:3px 0}}
.ld{{width:12px;height:12px;border-radius:50%}}
#stats{{position:fixed;top:12px;right:12px;background:#1e293b;
  padding:8px 12px;border-radius:8px;color:#64748b;font-size:11px;
  border:1px solid #334155;z-index:10}}
</style></head><body>
<div id="gc"><canvas id="cv"></canvas></div>
<div class="ctrl"><h3>Import Graph</h3>
  <label>Min Time: <span id="ml">0.01</span>s
    <input type="range" id="ms" min="0" max="3" step="0.01" value="0.01"></label>
  <label><input type="checkbox" id="po"> Nur Projekt-Code</label>
  <div style="margin-top:8px;color:#64748b">Scroll=Zoom | Drag=Pan | Click=Highlight Chain</div>
</div>
<div class="leg">
  <div class="li"><div class="ld" style="background:#ef4444"></div> &gt; 1.0s (project)</div>
  <div class="li"><div class="ld" style="background:#f97316"></div> &gt; 0.3s</div>
  <div class="li"><div class="ld" style="background:#eab308"></div> &gt; 0.1s</div>
  <div class="li"><div class="ld" style="background:#22c55e"></div> &lt; 0.1s</div>
  <div class="li"><div class="ld" style="background:#64748b"></div> External</div>
  <div class="li" style="margin-top:4px;color:#60a5fa">--- upstream (who imports)</div>
  <div class="li" style="color:#f97316">--- downstream (what it imports)</div>
</div>
<div class="tip" id="tip"></div>
<div id="stats">{n_nodes} nodes, {n_edges} edges</div>
<script>
const D={gd};
const cv=document.getElementById("cv");
const ctx=cv.getContext("2d");
const tip=document.getElementById("tip");
const stEl=document.getElementById("stats");
let W,H;
function resize(){{W=innerWidth;H=innerHeight;cv.width=W*devicePixelRatio;cv.height=H*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0)}}
resize();window.addEventListener("resize",resize);

let sel=null,mt=0.01,po=false;
let pan={{x:0,y:0}},zm=1,lm=null,dn=null;

function nc(n){{if(!n.is_project)return"#64748b";if(n.cumtime>1)return"#ef4444";if(n.cumtime>.3)return"#f97316";if(n.cumtime>.1)return"#eab308";return"#22c55e"}}
function nr(n){{return Math.max(5,Math.min(28,4+n.cumtime*10))}}

const ns=D.nodes.map(n=>({{...n,x:W/2+(Math.random()-.5)*W*.5,y:H/2+(Math.random()-.5)*H*.5,vx:0,vy:0}}));
let alpha=1;

function tick(){{
  if(alpha<.001)return;
  const N=ns.length,a=alpha;
  for(const n of ns){{n.vx+=(W/2-n.x)*1e-4*a;n.vy+=(H/2-n.y)*1e-4*a}}
  for(let i=0;i<N;i++)for(let j=i+1;j<N;j++){{
    let dx=ns[j].x-ns[i].x,dy=ns[j].y-ns[i].y,d2=dx*dx+dy*dy;
    if(d2<1)d2=1;if(d2>250000)continue;
    let d=Math.sqrt(d2),f=Math.min(1.5,80*a/d2);
    ns[i].vx-=dx/d*f;ns[i].vy-=dy/d*f;ns[j].vx+=dx/d*f;ns[j].vy+=dy/d*f}}
  for(const e of D.edges){{
    const s=ns[e.source],t=ns[e.target];if(!s||!t)continue;
    let dx=t.x-s.x,dy=t.y-s.y,d=Math.sqrt(dx*dx+dy*dy)+.1;
    let f=(d-140)*.0006*a;
    s.vx+=dx/d*f;s.vy+=dy/d*f;t.vx-=dx/d*f;t.vy-=dy/d*f}}
  const gc={{}};
  for(const n of ns){{if(!gc[n.group])gc[n.group]={{x:0,y:0,c:0}};gc[n.group].x+=n.x;gc[n.group].y+=n.y;gc[n.group].c++}}
  for(const g in gc){{gc[g].x/=gc[g].c;gc[g].y/=gc[g].c}}
  for(const n of ns){{const c=gc[n.group];if(c){{n.vx+=(c.x-n.x)*4e-4*a;n.vy+=(c.y-n.y)*4e-4*a}}}}
  for(const n of ns){{if(dn===n)continue;
    n.vx*=.55;n.vy*=.55;
    const spd=Math.sqrt(n.vx*n.vx+n.vy*n.vy);
    if(spd>4){{n.vx=n.vx/spd*4;n.vy=n.vy/spd*4}}
    n.x+=n.vx;n.y+=n.vy}}
  alpha*=.993
}}

function getChain(id){{
  const u=new Set,d=new Set;
  function fu(i){{for(const e of D.edges)if(e.target===i&&!u.has(e.source)){{u.add(e.source);fu(e.source)}}}}
  function fd(i){{for(const e of D.edges)if(e.source===i&&!d.has(e.target)){{d.add(e.target);fd(e.target)}}}}
  fu(id);fd(id);return{{u,d}}
}}

function tx(x){{return(x+pan.x)*zm}}
function ty(y){{return(y+pan.y)*zm}}

function render(){{
  ctx.clearRect(0,0,W,H);
  const vis=ns.filter(n=>n.cumtime>=mt&&(!po||n.is_project));
  const vi=new Set(vis.map(n=>n.id));
  const ve=D.edges.filter(e=>vi.has(e.source)&&vi.has(e.target));
  const ch=sel!==null?getChain(sel):null;
  let visE=0;

  for(const e of ve){{
    const s=ns[e.source],t=ns[e.target];
    const sx=tx(s.x),sy=ty(s.y),ex=tx(t.x),ey=ty(t.y);
    let opacity=0.35,color="#475569",width=1.2;
    if(ch){{
      const sIn=ch.u.has(e.source)||e.source===sel;
      const tIn=ch.u.has(e.target)||e.target===sel;
      const sInD=ch.d.has(e.source)||e.source===sel;
      const tInD=ch.d.has(e.target)||e.target===sel;
      if(sIn&&tIn){{opacity=0.9;color="#60a5fa";width=2.5}}
      else if(sInD&&tInD){{opacity=0.9;color="#f97316";width=2.5}}
      else{{opacity=0.06}}
    }}

    const dx=ex-sx,dy=ey-sy,d=Math.sqrt(dx*dx+dy*dy);
    if(d<1)continue;
    const r=nr(t)*zm;
    const ax=ex-dx/d*r,ay=ey-dy/d*r;

    ctx.beginPath();ctx.moveTo(sx,sy);ctx.lineTo(ax,ay);
    ctx.strokeStyle=color;ctx.globalAlpha=opacity;ctx.lineWidth=width;ctx.stroke();

    if(d>20){{
      const aLen=8*Math.min(2,width);
      const angle=Math.atan2(ay-sy,ax-sx);
      ctx.beginPath();ctx.moveTo(ax,ay);
      ctx.lineTo(ax-aLen*Math.cos(angle-.4),ay-aLen*Math.sin(angle-.4));
      ctx.lineTo(ax-aLen*Math.cos(angle+.4),ay-aLen*Math.sin(angle+.4));
      ctx.closePath();ctx.fillStyle=color;ctx.fill();
    }}
    visE++;
  }}
  ctx.globalAlpha=1;

  for(const n of vis){{
    const x=tx(n.x),y=ty(n.y),r=nr(n)*zm;
    let opacity=1,sw=n.is_project?2:1;
    if(ch&&n.id!==sel&&!ch.u.has(n.id)&&!ch.d.has(n.id))opacity=0.1;
    if(n.id===sel)sw=3;

    ctx.globalAlpha=opacity;
    ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);
    ctx.fillStyle=nc(n);ctx.fill();
    ctx.strokeStyle=n.is_project?"#e2e8f0":"#475569";
    ctx.lineWidth=sw;ctx.stroke();

    const fs=Math.max(9,Math.min(13,r*.8));
    if(r>4||n.id===sel){{
      ctx.font=fs+"px system-ui";ctx.textAlign="center";
      ctx.fillStyle="#cbd5e1";
      const label=n.label.split("/").pop().replace(".py","");
      ctx.fillText(label,x,y+r+fs+3);
      ctx.font=(fs-1)+"px system-ui";ctx.fillStyle="#64748b";
      ctx.fillText(n.cumtime+"s",x,y+r+fs*2+3);
    }}
  }}
  ctx.globalAlpha=1;
  stEl.textContent=vis.length+" nodes, "+visE+" edges shown ("+D.nodes.length+" / "+D.edges.length+" total)";
}}

function nodeAt(mx,my){{
  for(let i=ns.length-1;i>=0;i--){{
    const n=ns[i];if(n.cumtime<mt||(po&&!n.is_project))continue;
    const dx=tx(n.x)-mx,dy=ty(n.y)-my,r=nr(n)*zm+4;
    if(dx*dx+dy*dy<r*r)return n;
  }}
  return null;
}}

cv.addEventListener("click",e=>{{
  const n=nodeAt(e.clientX,e.clientY);
  sel=n?(sel===n.id?null:n.id):null;
}});
cv.addEventListener("mousemove",e=>{{
  const n=nodeAt(e.clientX,e.clientY);
  if(n){{
    tip.style.opacity=1;tip.style.left=e.clientX+14+"px";tip.style.top=e.clientY-12+"px";
    tip.innerHTML="<b>"+n.label+"</b><br>"+n.cumtime+"s | "+n.group+"<br>"+(n.is_project?"Project":"External");
    cv.style.cursor="pointer";
  }}else{{tip.style.opacity=0;cv.style.cursor="grab"}}
  if(e.buttons===1&&!dn){{
    if(lm){{pan.x+=(e.clientX-lm.x)/zm;pan.y+=(e.clientY-lm.y)/zm}}
    lm={{x:e.clientX,y:e.clientY}};cv.style.cursor="grabbing";
  }}else if(!dn){{lm=null}}
}});
cv.addEventListener("mouseup",()=>{{lm=null}});
cv.addEventListener("wheel",e=>{{
  e.preventDefault();
  const mx=e.clientX,my=e.clientY;
  const wx=(mx/zm)-pan.x, wy=(my/zm)-pan.y;
  zm*=e.deltaY>0?.9:1.1;zm=Math.max(.05,Math.min(8,zm));
  pan.x=(mx/zm)-wx;pan.y=(my/zm)-wy;
}});
cv.addEventListener("mousedown",e=>{{
  const n=nodeAt(e.clientX,e.clientY);
  if(n)dn=n;
}});
document.addEventListener("mousemove",e=>{{
  if(dn){{dn.x=(e.clientX/zm)-pan.x;dn.y=(e.clientY/zm)-pan.y;dn.vx=0;dn.vy=0;alpha=Math.max(alpha,.05)}}
}});
document.addEventListener("mouseup",()=>{{dn=null}});
document.getElementById("ms").addEventListener("input",e=>{{
  mt=parseFloat(e.target.value);document.getElementById("ml").textContent=mt.toFixed(2)}});
document.getElementById("po").addEventListener("change",e=>{{po=e.target.checked}});

(function loop(){{tick();render();requestAnimationFrame(loop)}})()
</script></body></html>'''


# ═══════════════════════════════════════════════════════════════
#  Function Call Graph — TBJS Glass Style
# ═══════════════════════════════════════════════════════════════

def _build_function_graph(all_stats, project_root, min_time):
    """Build interactive function-level call graph with 3 concentric circles.

    Inner circle  = avg time per call  (tottime / ncalls)
    Middle circle = tottime            (time spent IN this function)
    Outer circle  = cumtime            (time including sub-calls)

    Edges from cProfile callers dict with line numbers.
    """
    norm_root = os.path.normpath(project_root)
    nodes = []
    node_map = {}  # (filename, lineno, funcname) -> node_id

    for (fn, ln, func), (cc, nc, tt, ct, callers) in all_stats.items():
        if ct < min_time and tt < min_time:
            continue
        key = (fn, ln, func)
        if key in node_map:
            continue

        nid = len(nodes)
        node_map[key] = nid
        norm_fn = os.path.normpath(fn)
        is_proj = norm_fn.startswith(norm_root)

        if is_proj:
            rel = os.path.relpath(norm_fn, norm_root)
            parts = Path(rel).parts
            group = "/".join(parts[:2]) if len(parts) > 1 else parts[0]
            label = _short_label(norm_fn, norm_root)
        else:
            sp = re.search(r'site-packages[/\\]([^/\\]+)', norm_fn)
            group = sp.group(1) if sp else "stdlib"
            label = _short_label(norm_fn, norm_root)

        avg_time = tt / nc if nc > 0 else 0

        nodes.append({
            "id": nid,
            "func": func,
            "label": label,
            "line": ln,
            "ncalls": nc,
            "tottime": round(tt, 6),
            "cumtime": round(ct, 6),
            "avgtime": round(avg_time, 6),
            "is_project": is_proj,
            "group": group,
        })

    # Build edges from callers dict
    edges = []
    edge_set = set()

    for (fn, ln, func), (cc, nc, tt, ct, callers) in all_stats.items():
        callee_key = (fn, ln, func)
        if callee_key not in node_map:
            continue
        callee_id = node_map[callee_key]

        for caller_key, (c_cc, c_nc, c_tt, c_ct) in callers.items():
            if caller_key not in node_map:
                continue
            caller_id = node_map[caller_key]
            if caller_id == callee_id:
                continue
            ek = (caller_id, callee_id)
            if ek in edge_set:
                continue
            edge_set.add(ek)
            edges.append({
                "source": caller_id,
                "target": callee_id,
                "calls": c_nc,
                "caller_line": caller_key[1],
            })

    gd = json.dumps({"nodes": nodes, "edges": edges})
    n_nodes = len(nodes)
    n_edges = len(edges)

    return _function_graph_html(gd, n_nodes, n_edges)


def _function_graph_html(graph_data_json, n_nodes, n_edges):
    """Generate the full HTML for the function call graph."""
    return f'''<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<title>Function Call Graph — tb_analyze</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {{
  --bg-base: #08080d;
  --bg-surface: rgba(10, 10, 18, 0.85);
  --bg-elevated: rgba(15, 15, 25, 0.95);
  --glass-border: rgba(255, 255, 255, 0.05);
  --glass-blur: 12px;
  --border-subtle: rgba(255, 255, 255, 0.08);
  --text-main: rgba(255, 255, 255, 0.85);
  --text-label: rgba(255, 255, 255, 0.4);
  --text-muted: rgba(255, 255, 255, 0.25);

  --c-avg: oklch(55% 0.18 230);    /* blue — avg per call */
  --c-tot: oklch(65% 0.2 145);     /* green — tottime */
  --c-cum: oklch(75% 0.18 85);     /* yellow — cumtime */
  --c-hot: oklch(55% 0.22 25);     /* red — hot functions */
  --c-ext: rgba(100, 116, 139, 0.6);
  --c-edge-up: oklch(60% 0.15 230);
  --c-edge-down: oklch(65% 0.18 55);

  --font-sans: 'IBM Plex Sans', system-ui, sans-serif;
  --font-mono: 'IBM Plex Mono', ui-monospace, monospace;
  --radius-md: 6px;
  --radius-lg: 12px;
}}
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: var(--bg-base);
  font-family: var(--font-sans);
  color: var(--text-main);
  overflow: hidden;
  -webkit-font-smoothing: antialiased;
}}
canvas {{ display: block; width: 100%; height: 100%; }}

/* Panels */
.panel {{
  position: fixed;
  background: var(--bg-surface);
  backdrop-filter: blur(var(--glass-blur));
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-lg);
  padding: 14px 18px;
  z-index: 10;
  font-size: 12px;
  color: var(--text-label);
  box-shadow: 0 4px 16px rgba(0,0,0,0.5);
}}
.panel h3 {{
  color: var(--text-main);
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 8px;
  font-family: var(--font-sans);
}}
.panel label {{
  display: block;
  margin: 5px 0;
  cursor: pointer;
  font-family: var(--font-mono);
  font-size: 11px;
}}
.panel input[type=range] {{ width: 100%; accent-color: oklch(55% 0.18 230); }}
.panel input[type=checkbox] {{ accent-color: oklch(55% 0.18 230); }}

.ctrl {{ top: 12px; left: 12px; max-width: 260px; }}
.legend {{ bottom: 12px; left: 12px; }}
.stats {{ top: 12px; right: 12px; font-family: var(--font-mono); font-size: 11px; color: var(--text-muted); }}

/* Legend items */
.leg-section {{
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid var(--border-subtle);
}}
.leg-section:first-child {{ margin-top: 0; padding-top: 0; border-top: none; }}
.leg-title {{
  font-family: var(--font-mono);
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: var(--text-muted);
  margin-bottom: 6px;
}}
.leg-item {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 4px 0;
  font-size: 11px;
  color: var(--text-label);
}}
.leg-circle {{
  width: 14px;
  height: 14px;
  border-radius: 50%;
  flex-shrink: 0;
}}
.leg-line {{
  width: 20px;
  height: 0;
  border-top: 2px dashed;
  flex-shrink: 0;
}}

/* Tooltip */
.tip {{
  position: absolute;
  background: var(--bg-elevated);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-md);
  padding: 10px 14px;
  font-size: 12px;
  color: var(--text-main);
  pointer-events: none;
  opacity: 0;
  box-shadow: 0 4px 16px rgba(0,0,0,0.6);
  max-width: 360px;
  line-height: 1.5;
  font-family: var(--font-sans);
  transition: opacity 0.1s ease;
}}
.tip b {{ color: var(--text-main); font-weight: 600; }}
.tip .mono {{ font-family: var(--font-mono); font-size: 11px; }}
.tip .dim {{ color: var(--text-label); }}
.tip .val {{ font-family: var(--font-mono); }}
.tip .row {{ display: flex; justify-content: space-between; gap: 12px; margin: 2px 0; }}
.tip .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }}

/* Detail panel for selected node */
.detail {{
  position: fixed;
  bottom: 12px;
  right: 12px;
  max-width: 340px;
  max-height: 50vh;
  overflow-y: auto;
  background: var(--bg-surface);
  backdrop-filter: blur(var(--glass-blur));
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-lg);
  padding: 14px 18px;
  z-index: 10;
  font-size: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.5);
  display: none;
}}
.detail h3 {{ color: var(--text-main); font-size: 13px; font-weight: 600; margin-bottom: 6px; font-family: var(--font-mono); }}
.detail .meta {{ color: var(--text-label); font-size: 11px; font-family: var(--font-mono); margin-bottom: 8px; }}
.detail table {{ width: 100%; border-collapse: collapse; font-size: 11px; font-family: var(--font-mono); }}
.detail td {{ padding: 3px 6px; color: var(--text-label); }}
.detail td:first-child {{ color: var(--text-muted); }}
.detail td:last-child {{ text-align: right; color: var(--text-main); }}
.detail .section-label {{
  font-size: 9px; text-transform: uppercase; letter-spacing: 2px;
  color: var(--text-muted); margin: 8px 0 4px; padding-top: 6px;
  border-top: 1px solid var(--border-subtle);
}}
.detail .caller-row {{
  font-size: 11px; color: var(--text-label); padding: 2px 0;
  font-family: var(--font-mono); cursor: pointer;
}}
.detail .caller-row:hover {{ color: var(--text-main); }}

/* Mode key indicators */
.mode-key {{
  display: inline-block;
  width: 18px; height: 18px;
  line-height: 18px;
  text-align: center;
  border-radius: 3px;
  background: rgba(255,255,255,0.06);
  border: 1px solid var(--border-subtle);
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 600;
  color: var(--text-muted);
  margin-right: 2px;
  transition: all 0.1s;
}}
.mode-key.active {{
  background: oklch(55% 0.18 230 / 0.3);
  border-color: oklch(55% 0.18 230);
  color: oklch(80% 0.12 230);
}}

/* Hotpath highlight badge */
.hotpath-badge {{
  display: inline-block;
  padding: 1px 5px;
  border-radius: 3px;
  font-size: 9px;
  font-weight: 600;
  font-family: var(--font-mono);
  margin-left: 4px;
}}
.hotpath-1 {{ background: oklch(55% 0.22 25 / 0.3); color: oklch(75% 0.15 25); }}
.hotpath-2 {{ background: oklch(65% 0.18 55 / 0.3); color: oklch(80% 0.12 55); }}
.hotpath-3 {{ background: oklch(75% 0.18 85 / 0.3); color: oklch(80% 0.12 85); }}
</style>
</head>
<body>
<canvas id="cv"></canvas>

<div class="panel ctrl">
  <h3>Function Call Graph</h3>
  <label>Min Time: <span id="ml">0.01</span>s
    <input type="range" id="ms" min="0" max="3" step="0.01" value="0.01">
  </label>
  <label>Spacing: <span id="sl">160</span>
    <input type="range" id="ss" min="40" max="500" step="10" value="160">
  </label>
  <label><input type="checkbox" id="po"> Project only</label>
  <label><input type="checkbox" id="hm"> Hide &lt;module&gt;</label>
  <div style="margin-top:10px;border-top:1px solid var(--border-subtle);padding-top:8px">
    <div class="leg-title">Selection modes (hold key)</div>
    <div style="margin:4px 0;font-family:var(--font-mono);font-size:11px;color:var(--text-label)">
      <span id="mk_u" class="mode-key">U</span> upstream only&ensp;
      <span id="mk_d" class="mode-key">D</span> downstream only<br>
      <span id="mk_i" class="mode-key">I</span> inspect — follow hotpath
    </div>
  </div>
  <div style="margin-top:8px;color:var(--text-muted);font-family:var(--font-mono);font-size:10px">
    Scroll = Zoom &middot; Drag = Pan<br>Click = Select &middot; Dbl = Reset
  </div>
</div>

<div class="panel legend">
  <div class="leg-section">
    <div class="leg-title">Circles (per function)</div>
    <div class="leg-item"><div class="leg-circle" style="background:oklch(55% 0.18 230);opacity:0.9"></div> Inner — avg/call (tottime&divide;ncalls)</div>
    <div class="leg-item"><div class="leg-circle" style="background:oklch(65% 0.2 145);opacity:0.5"></div> Middle — tottime (self only)</div>
    <div class="leg-item"><div class="leg-circle" style="background:oklch(75% 0.18 85);opacity:0.35"></div> Outer — cumtime (heat-colored)</div>
  </div>
  <div class="leg-section">
    <div class="leg-title">Outer ring heat</div>
    <div class="leg-item"><div class="leg-circle" style="background:oklch(55% 0.22 25)"></div> &gt; 1.0s cumtime</div>
    <div class="leg-item"><div class="leg-circle" style="background:oklch(65% 0.18 55)"></div> &gt; 0.3s</div>
    <div class="leg-item"><div class="leg-circle" style="background:oklch(75% 0.18 85)"></div> &gt; 0.1s</div>
    <div class="leg-item"><div class="leg-circle" style="background:oklch(65% 0.2 145)"></div> &lt; 0.1s</div>
    <div class="leg-item"><div class="leg-circle" style="background:var(--c-ext)"></div> External / stdlib</div>
  </div>
  <div class="leg-section">
    <div class="leg-title">Edges (call direction)</div>
    <div class="leg-item"><div class="leg-line" style="border-color:var(--c-edge-up)"></div> Upstream (who calls)</div>
    <div class="leg-item"><div class="leg-line" style="border-color:var(--c-edge-down)"></div> Downstream (what it calls)</div>
  </div>
</div>

<div class="panel stats" id="stats">{n_nodes} nodes, {n_edges} edges</div>
<div class="tip" id="tip"></div>
<div class="detail" id="detail"></div>

<script>
const D={graph_data_json};
const cv=document.getElementById("cv");
const ctx=cv.getContext("2d");
const tip=document.getElementById("tip");
const detailEl=document.getElementById("detail");
const stEl=document.getElementById("stats");

let W,H;
function resize(){{
  W=innerWidth;H=innerHeight;
  cv.width=W*devicePixelRatio;cv.height=H*devicePixelRatio;
  ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
}}
resize();window.addEventListener("resize",resize);

let sel=null, mt=0.01, projOnly=false, hideModule=false;
let pan={{x:0,y:0}}, zm=1, lm=null, dn=null;
let springLen=160;  // spacing slider
let mode="all";     // "all" | "up" | "down" | "inspect"
let hotpath=[];     // inspect mode: ordered list of node ids on the hot path

// Colors
const C_AVG="oklch(55% 0.18 230)";
const C_TOT="oklch(65% 0.2 145)";
const C_CUM="oklch(75% 0.18 85)";
const C_EXT="rgba(100,116,139,0.6)";
const C_UP="oklch(60% 0.15 230)";
const C_DN="oklch(65% 0.18 55)";

function nodeColor(n){{
  if(!n.is_project) return C_EXT;
  if(n.cumtime>1) return "oklch(55% 0.22 25)";
  if(n.cumtime>0.3) return "oklch(65% 0.18 55)";
  if(n.cumtime>0.1) return "oklch(75% 0.18 85)";
  return "oklch(65% 0.2 145)";
}}

// Radius scaling — based on cumtime, log-scaled for spread
function baseRadius(n){{ return Math.max(4, Math.min(22, 4 + Math.sqrt(n.cumtime) * 12)); }}

// 3 rings: outer=cumtime, middle=tottime, inner=avgtime
// Each proportional to its time share relative to cumtime
function ringRadii(n){{
  const base = baseRadius(n);
  const cum = base;
  const totRatio = n.cumtime > 1e-9 ? n.tottime / n.cumtime : 0.5;
  const tot = Math.max(2, cum * Math.max(0.15, Math.min(0.85, totRatio)));
  // avg scaled by how avgtime compares to tottime (small avg = tight inner circle)
  const avgRatio = n.tottime > 1e-9 ? n.avgtime / n.tottime : 0.5;
  const avg = Math.max(1.5, tot * Math.max(0.15, Math.min(0.7, avgRatio)));
  return {{ avg, tot, cum }};
}}

// Init positions
const ns=D.nodes.map(n=>({{...n, x:W/2+(Math.random()-.5)*W*.6, y:H/2+(Math.random()-.5)*H*.6, vx:0, vy:0}}));
let alpha=1;

function isVisible(n){{
  if(n.cumtime<mt && n.tottime<mt) return false;
  if(projOnly && !n.is_project) return false;
  if(hideModule && n.func==="<module>") return false;
  return true;
}}

function tick(){{
  if(alpha<0.001) return;
  const a=alpha;
  const vns=ns.filter(isVisible);
  const N=vns.length;
  // Center gravity
  for(const n of vns){{ n.vx+=(W/2-n.x)*1e-4*a; n.vy+=(H/2-n.y)*1e-4*a; }}
  // Repulsion — skip pairs beyond threshold, cap iterations for large graphs
  const maxPairs=Math.min(N,300);
  for(let i=0;i<maxPairs;i++) for(let j=i+1;j<maxPairs;j++){{
    let dx=vns[j].x-vns[i].x, dy=vns[j].y-vns[i].y, d2=dx*dx+dy*dy;
    if(d2<1)d2=1; if(d2>300000)continue;
    let d=Math.sqrt(d2), f=Math.min(2, 120*a/d2);
    vns[i].vx-=dx/d*f; vns[i].vy-=dy/d*f;
    vns[j].vx+=dx/d*f; vns[j].vy+=dy/d*f;
  }}
  // Spring (edges)
  for(const e of D.edges){{
    const s=ns[e.source], t=ns[e.target]; if(!s||!t) continue;
    let dx=t.x-s.x, dy=t.y-s.y, d=Math.sqrt(dx*dx+dy*dy)+0.1;
    let f=(d-springLen)*0.0005*a;
    s.vx+=dx/d*f; s.vy+=dy/d*f; t.vx-=dx/d*f; t.vy-=dy/d*f;
  }}
  // Group cohesion
  const gc={{}};
  for(const n of ns){{ if(!gc[n.group]) gc[n.group]={{x:0,y:0,c:0}}; gc[n.group].x+=n.x; gc[n.group].y+=n.y; gc[n.group].c++; }}
  for(const g in gc){{ gc[g].x/=gc[g].c; gc[g].y/=gc[g].c; }}
  for(const n of ns){{ const c=gc[n.group]; if(c){{ n.vx+=(c.x-n.x)*3e-4*a; n.vy+=(c.y-n.y)*3e-4*a; }} }}
  // Integrate
  for(const n of ns){{
    if(dn===n) continue;
    n.vx*=0.55; n.vy*=0.55;
    const spd=Math.sqrt(n.vx*n.vx+n.vy*n.vy);
    if(spd>5){{ n.vx=n.vx/spd*5; n.vy=n.vy/spd*5; }}
    n.x+=n.vx; n.y+=n.vy;
  }}
  alpha*=0.993;
}}

function getChain(id){{
  const u=new Set, d=new Set;
  if(mode==="up" || mode==="all"){{
    // Direct callers (1 level up)
    for(const e of D.edges) if(e.target===id) u.add(e.source);
  }}
  if(mode==="down" || mode==="all"){{
    // Direct callees (1 level down)
    for(const e of D.edges) if(e.source===id) d.add(e.target);
  }}
  if(mode==="inspect"){{
    // Recursive downstream: follow the heaviest callee at each level
    // Also collect ALL direct callees but highlight the hot path
    const visited=new Set();
    function followHot(nid, depth){{
      if(depth>30||visited.has(nid)) return;
      visited.add(nid);
      const callees=D.edges.filter(e=>e.source===nid);
      for(const e of callees) d.add(e.target);
      // Follow the heaviest callee
      if(callees.length>0){{
        const heaviest=callees.reduce((best,e)=>{{
          const cn=ns[e.target];
          const bn=ns[best.target];
          return cn&&bn&&cn.cumtime>bn.cumtime?e:best;
        }});
        const hn=ns[heaviest.target];
        if(hn && hn.cumtime>0.001){{
          hotpath.push(heaviest.target);
          followHot(heaviest.target, depth+1);
        }}
      }}
    }}
    hotpath=[id];
    followHot(id, 0);
  }}
  return {{u,d}};
}}

// Build top-3 hotpath nodes (by cumtime, excluding the selected node itself)
function getHotNodes(){{
  if(mode!=="inspect"||hotpath.length<2) return [];
  // hotpath[0] is the selected node, rest are the chain
  return hotpath.slice(1)
    .map(id=>ns[id])
    .filter(Boolean)
    .sort((a,b)=>b.cumtime-a.cumtime)
    .slice(0,3);
}}

function tx(x){{ return (x+pan.x)*zm; }}
function ty(y){{ return (y+pan.y)*zm; }}

function render(){{
  ctx.clearRect(0,0,W,H);
  const vis=ns.filter(isVisible);
  const vi=new Set(vis.map(n=>n.id));
  const ve=D.edges.filter(e=>vi.has(e.source)&&vi.has(e.target));
  const ch=sel!==null?getChain(sel):null;
  let visE=0;

  // Draw edges
  const hpSet=new Set(hotpath);
  for(const e of ve){{
    const s=ns[e.source], t=ns[e.target];
    const sx=tx(s.x), sy=ty(s.y), ex=tx(t.x), ey=ty(t.y);
    let opacity=0.25, color="#475569", width=1;

    if(ch){{
      if(mode==="inspect"){{
        // Inspect: highlight edges on the hotpath chain
        const srcOnPath=hpSet.has(e.source);
        const tgtOnPath=hpSet.has(e.target);
        if(srcOnPath&&tgtOnPath){{ opacity=0.95; color="oklch(55% 0.22 25)"; width=3.5; }}
        else if(ch.d.has(e.target)&&(e.source===sel||ch.d.has(e.source))){{ opacity=0.4; color=C_DN; width=1.5; }}
        else{{ opacity=0.03; }}
      }} else {{
        const isUp=(ch.u.has(e.source)||e.source===sel)&&(ch.u.has(e.target)||e.target===sel);
        const isDn=(ch.d.has(e.source)||e.source===sel)&&(ch.d.has(e.target)||e.target===sel);
        if(isUp){{ opacity=0.85; color=C_UP; width=2.5; }}
        else if(isDn){{ opacity=0.85; color=C_DN; width=2.5; }}
        else{{ opacity=0.04; }}
      }}
    }}

    const dx=ex-sx, dy=ey-sy, d=Math.sqrt(dx*dx+dy*dy);
    if(d<1) continue;
    const r=baseRadius(t)*zm;
    const ax=ex-dx/d*r, ay=ey-dy/d*r;

    ctx.beginPath(); ctx.moveTo(sx,sy); ctx.lineTo(ax,ay);
    ctx.strokeStyle=color; ctx.globalAlpha=opacity; ctx.lineWidth=width; ctx.stroke();

    // Arrow
    if(d>25){{
      const aLen=7*Math.min(2,width);
      const angle=Math.atan2(ay-sy,ax-sx);
      ctx.beginPath(); ctx.moveTo(ax,ay);
      ctx.lineTo(ax-aLen*Math.cos(angle-.35),ay-aLen*Math.sin(angle-.35));
      ctx.lineTo(ax-aLen*Math.cos(angle+.35),ay-aLen*Math.sin(angle+.35));
      ctx.closePath(); ctx.fillStyle=color; ctx.fill();
    }}

    // Edge label (calls count) — only when highlighted
    if(opacity>0.5 && e.calls>1){{
      const mx=(sx+ax)/2, my=(sy+ay)/2;
      ctx.font="9px 'IBM Plex Mono'";
      ctx.fillStyle=color; ctx.globalAlpha=0.7;
      ctx.textAlign="center";
      ctx.fillText(e.calls+"×",mx,my-4);
    }}

    visE++;
  }}
  ctx.globalAlpha=1;

  // Draw nodes — 3 concentric circles
  const topHot=getHotNodes();
  const topHotIds=new Set(topHot.map(n=>n.id));
  for(const n of vis){{
    const x=tx(n.x), y=ty(n.y);
    const rr=ringRadii(n);
    let opacity=1, strokeW=n.is_project?1.5:0.8;
    if(ch && n.id!==sel && !ch.u.has(n.id) && !ch.d.has(n.id)) opacity=0.08;
    if(mode==="inspect" && ch && hpSet.has(n.id)) strokeW=2.5;
    if(n.id===sel) strokeW=3;
    // Top-3 hotpath: extra glow
    if(topHotIds.has(n.id)){{ strokeW=3.5; }}
    ctx.globalAlpha=opacity;

    // Outer ring: cumtime — heat-colored (green→yellow→red by time)
    ctx.beginPath(); ctx.arc(x,y,rr.cum*zm,0,Math.PI*2);
    ctx.fillStyle=nodeColor(n); ctx.globalAlpha=opacity*0.2;
    ctx.fill();
    ctx.strokeStyle=n.is_project?"rgba(255,255,255,0.15)":"rgba(100,116,139,0.2)";
    ctx.lineWidth=strokeW; ctx.globalAlpha=opacity; ctx.stroke();

    // Middle ring: tottime — always green (fixed)
    ctx.beginPath(); ctx.arc(x,y,rr.tot*zm,0,Math.PI*2);
    ctx.fillStyle=n.is_project?C_TOT:C_EXT; ctx.globalAlpha=opacity*0.45;
    ctx.fill();

    // Inner ring: avg per call — always blue (fixed)
    ctx.beginPath(); ctx.arc(x,y,rr.avg*zm,0,Math.PI*2);
    ctx.fillStyle=n.is_project?C_AVG:C_EXT; ctx.globalAlpha=opacity*0.9;
    ctx.fill();

    ctx.globalAlpha=opacity;

    // Label
    const fs=Math.max(9,Math.min(12, rr.cum*zm*0.6));
    if(rr.cum*zm>6 || n.id===sel || topHotIds.has(n.id)){{
      ctx.font=fs+"px 'IBM Plex Mono'";
      ctx.textAlign="center";
      ctx.fillStyle="rgba(255,255,255,0.75)";
      const fname = n.func==="<module>" ? n.label.split("/").pop().replace(".py","") : n.func;
      ctx.fillText(fname, x, y+rr.cum*zm+fs+3);
      // Time below
      ctx.font=(fs-1)+"px 'IBM Plex Mono'";
      ctx.fillStyle="rgba(255,255,255,0.35)";
      ctx.fillText(n.cumtime.toFixed(3)+"s", x, y+rr.cum*zm+fs*2+4);
      // Hotpath rank badge
      const hpIdx=topHot.findIndex(h=>h.id===n.id);
      if(hpIdx>=0){{
        const colors=["oklch(55% 0.22 25)","oklch(65% 0.18 55)","oklch(75% 0.18 85)"];
        ctx.font="bold "+(fs-1)+"px 'IBM Plex Mono'";
        ctx.fillStyle=colors[hpIdx]||colors[2];
        ctx.fillText("★ #"+(hpIdx+1)+" hotpath", x, y+rr.cum*zm+fs*3+6);
      }}
    }}
  }}
  ctx.globalAlpha=1;
  stEl.textContent=vis.length+" / "+D.nodes.length+" nodes, "+visE+" / "+D.edges.length+" edges";
}}

function nodeAt(mx,my){{
  for(let i=ns.length-1;i>=0;i--){{
    const n=ns[i]; if(!isVisible(n)) continue;
    const dx=tx(n.x)-mx, dy=ty(n.y)-my, r=baseRadius(n)*zm+5;
    if(dx*dx+dy*dy<r*r) return n;
  }}
  return null;
}}

function showDetail(n){{
  if(!n){{ detailEl.style.display="none"; return; }}
  // Find callers & callees
  const callers=D.edges.filter(e=>e.target===n.id).map(e=>({{...ns[e.source], calls:e.calls, line:e.caller_line}}));
  const callees=D.edges.filter(e=>e.source===n.id).map(e=>({{...ns[e.target], calls:e.calls, line:e.caller_line}}));

  let html=`<h3>${{n.func}}</h3>`;
  html+=`<div class="meta">${{n.label}}:${{n.line}}</div>`;
  html+=`<table>
    <tr><td>avg/call</td><td>${{n.avgtime.toFixed(6)}}s</td></tr>
    <tr><td>tottime</td><td>${{n.tottime.toFixed(4)}}s</td></tr>
    <tr><td>cumtime</td><td>${{n.cumtime.toFixed(4)}}s</td></tr>
    <tr><td>ncalls</td><td>${{n.ncalls}}</td></tr>
    <tr><td>group</td><td>${{n.group}}</td></tr>
  </table>`;

  if(callers.length){{
    html+=`<div class="section-label">Called by (${{callers.length}})</div>`;
    callers.sort((a,b)=>b.cumtime-a.cumtime);
    for(const c of callers.slice(0,15)){{
      html+=`<div class="caller-row" data-id="${{c.id}}">&larr; ${{c.func}} <span style="color:var(--text-muted)">:${{c.line}} (${{c.calls}}&times;)</span></div>`;
    }}
  }}
  if(callees.length){{
    html+=`<div class="section-label">Calls (${{callees.length}})</div>`;
    callees.sort((a,b)=>b.cumtime-a.cumtime);
    for(const c of callees.slice(0,15)){{
      html+=`<div class="caller-row" data-id="${{c.id}}">&rarr; ${{c.func}} <span style="color:var(--text-muted)">:${{c.line}} (${{c.calls}}&times;)</span></div>`;
    }}
  }}

  detailEl.innerHTML=html;
  detailEl.style.display="block";

  // Click on caller/callee to navigate
  detailEl.querySelectorAll(".caller-row").forEach(el=>{{
    el.addEventListener("click",()=>{{
      const id=parseInt(el.dataset.id);
      sel=id; showDetail(ns[id]); alpha=Math.max(alpha,0.1);
    }});
  }});
}}

// Events
cv.addEventListener("click",e=>{{
  const n=nodeAt(e.clientX,e.clientY);
  if(n){{ sel=(sel===n.id?null:n.id); showDetail(sel!==null?n:null); }}
  else{{ sel=null; showDetail(null); }}
}});

cv.addEventListener("dblclick",e=>{{
  sel=null; showDetail(null); pan={{x:0,y:0}}; zm=1;
}});

cv.addEventListener("mousemove",e=>{{
  const n=nodeAt(e.clientX,e.clientY);
  if(n){{
    tip.style.opacity="1";
    tip.style.left=(e.clientX+16)+"px";
    tip.style.top=(e.clientY-14)+"px";
    tip.innerHTML=`<b>${{n.func}}</b> <span class="dim mono">${{n.label}}:${{n.line}}</span><br>`
      +`<div class="row"><span><span class="dot" style="background:var(--c-avg)"></span>avg/call</span><span class="val">${{n.avgtime.toFixed(6)}}s</span></div>`
      +`<div class="row"><span><span class="dot" style="background:var(--c-tot)"></span>tottime</span><span class="val">${{n.tottime.toFixed(4)}}s</span></div>`
      +`<div class="row"><span><span class="dot" style="background:var(--c-cum)"></span>cumtime</span><span class="val">${{n.cumtime.toFixed(4)}}s</span></div>`
      +`<div class="row"><span class="dim">ncalls</span><span class="val">${{n.ncalls}}</span></div>`
      +`<div class="row"><span class="dim">group</span><span class="val">${{n.group}}</span></div>`;
    cv.style.cursor="pointer";
  }} else {{
    tip.style.opacity="0"; cv.style.cursor="grab";
  }}
  // Pan
  if(e.buttons===1 && !dn){{
    if(lm){{ pan.x+=(e.clientX-lm.x)/zm; pan.y+=(e.clientY-lm.y)/zm; }}
    lm={{x:e.clientX,y:e.clientY}}; cv.style.cursor="grabbing";
  }} else if(!dn){{ lm=null; }}
}});

cv.addEventListener("mouseup",()=>{{lm=null}});
cv.addEventListener("wheel",e=>{{
  e.preventDefault();
  const mx=e.clientX, my=e.clientY;
  const wx=(mx/zm)-pan.x, wy=(my/zm)-pan.y;
  zm*=e.deltaY>0?0.9:1.1; zm=Math.max(0.05,Math.min(10,zm));
  pan.x=(mx/zm)-wx; pan.y=(my/zm)-wy;
}},{{passive:false}});
cv.addEventListener("mousedown",e=>{{
  const n=nodeAt(e.clientX,e.clientY);
  if(n) dn=n;
}});
document.addEventListener("mousemove",e=>{{
  if(dn){{ dn.x=(e.clientX/zm)-pan.x; dn.y=(e.clientY/zm)-pan.y; dn.vx=0; dn.vy=0; alpha=Math.max(alpha,0.05); }}
}});
document.addEventListener("mouseup",()=>{{dn=null}});

// Controls
document.getElementById("ms").addEventListener("input",e=>{{
  mt=parseFloat(e.target.value);
  document.getElementById("ml").textContent=mt.toFixed(2);
}});
document.getElementById("ss").addEventListener("input",e=>{{
  springLen=parseFloat(e.target.value);
  document.getElementById("sl").textContent=springLen.toFixed(0);
  alpha=Math.max(alpha,0.3);
}});
document.getElementById("po").addEventListener("change",e=>{{projOnly=e.target.checked}});
document.getElementById("hm").addEventListener("change",e=>{{hideModule=e.target.checked}});

// Keyboard modes: U=upstream, D=downstream, I=inspect (hold key)
const modeKeys={{"u":"up","d":"down","i":"inspect"}};
const mkEls={{u:document.getElementById("mk_u"),d:document.getElementById("mk_d"),i:document.getElementById("mk_i")}};
function setMode(m){{
  mode=m;
  for(const k in mkEls){{ mkEls[k].classList.toggle("active", modeKeys[k]===m); }}
  // Re-compute chain if node selected
  if(sel!==null){{ hotpath=[]; }}
}}
document.addEventListener("keydown",e=>{{
  const k=e.key.toLowerCase();
  if(k in modeKeys && mode!==modeKeys[k]){{
    setMode(modeKeys[k]);
    // Re-trigger chain computation on next render
    if(sel!==null) hotpath=[];
  }}
}});
document.addEventListener("keyup",e=>{{
  const k=e.key.toLowerCase();
  if(k in modeKeys && mode===modeKeys[k]){{
    setMode("all");
    hotpath=[];
  }}
}});

(function loop(){{ tick(); render(); requestAnimationFrame(loop); }})();
</script>
</body></html>'''

def _get_package_name(filepath):
    sp = re.search(r'site-packages[/\\]([^/\\]+)', filepath)
    if sp:
        name = sp.group(1)
        return re.sub(r'[-.].*', '', name)
    return None

def _short_label(filepath, root):
    norm = os.path.normpath(filepath)
    if norm.startswith(root):
        rel = os.path.relpath(norm, root)
        parts = Path(rel).parts
        return "/".join(parts[-3:]) if len(parts) > 3 else "/".join(parts)
    sp = re.search(r'site-packages[/\\](.+)', norm)
    if sp:
        parts = Path(sp.group(1)).parts
        return parts[0] + "/.../" + parts[-1] if len(parts) > 2 else "/".join(parts)
    return os.path.basename(norm)

def _shorten_path(fp, n=3):
    p = Path(fp).parts
    return ".../" + "/".join(p[-n:]) if len(p) > n else fp

def _make_filtered_stats(profiler, entries):
    f = pstats.Stats(profiler)
    f.stats = entries
    return f

def _print_section(title, sort_by, stats_obj, top_n):
    print(f"\n{'='*80}")
    print(f"  {title} -- sorted by {sort_by}")
    print(f"{'='*80}")
    stats_obj.sort_stats(sort_by)
    stats_obj.print_stats(top_n)
