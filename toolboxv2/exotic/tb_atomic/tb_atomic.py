"""
tb_atomic — dev/build tool that packs a single ToolBoxV2 component (and its
minimal transitive in-tree dependencies) into a standalone, install-free
Python distribution.

No runtime import hooks. The packer:
  1. crawls transitive in-tree imports of an entry file (absolute *and*
     relative imports are resolved),
  2. rewrites the `toolboxv2.` namespace to a neutral shared `tbv.` so multiple
     packed modules co-install without clashing with a real ToolBoxV2,
  3. splits modules into a shared `tbv-core` base (seeded from the exports
     re-declared in `toolboxv2/__init__.py`) and a per-entry leaf bundle,
  4. generates `tbv/__init__.py` mirroring the original top-level re-exports;
     framework-bound symbols (App, get_app) raise a clear ImportError.

CLI:
    python -m tb_atomic pack <entry.py> --root <repo> -o <outdir>
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

TOP = "toolboxv2"
NS = "tbv"

# Top-level symbols that cannot exist without a live framework -> stub.
FRAMEWORK_STUBS = {"App", "get_app", "ToolBox_over", "flows_dict"}

# Non-Python data files a module reads via `tb_root_dir / <relpath>` at import
# or runtime. When the keyed module lands in a closure, the files are vendored
# into the package's tbv/ dir so the shim's tb_root_dir resolves them.
_ASSETS: dict[str, list[str]] = {
    "toolboxv2.utils.workers.fast_tb_defaults": ["tbjs/dist/tbjs.css"],
}


# ── module <-> path ──────────────────────────────────────────────────────────


def mod_to_path(mod: str, root: Path) -> Path | None:
    """toolboxv2.a.b -> <root>/toolboxv2/a/b.py  or  .../b/__init__.py"""
    rel = mod.replace(".", "/")
    cand_mod = root / f"{rel}.py"
    cand_pkg = root / rel / "__init__.py"
    if cand_mod.exists():
        return cand_mod
    if cand_pkg.exists():
        return cand_pkg
    return None


def path_to_mod(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def is_package_path(path: Path) -> bool:
    return path.name == "__init__.py"


# ── import resolution (absolute + relative) ──────────────────────────────────


def _pkg_of(mod: str, is_pkg: bool) -> str:
    """Containing package of a module, per Python relative-import rules."""
    return mod if is_pkg else (mod.rsplit(".", 1)[0] if "." in mod else "")


def _resolve_relative(base_pkg: str, level: int, module: str | None) -> str:
    anchor = base_pkg
    for _ in range(level - 1):
        anchor = anchor.rsplit(".", 1)[0] if "." in anchor else ""
    if module:
        return f"{anchor}.{module}" if anchor else module
    return anchor


def _iter_imports(src: str, this_mod: str, is_pkg: bool, load_time_only: bool):
    """
    Yield ('import'|'from', absolute_module, [imported_names]) for in-tree
    imports. Relative imports are resolved to absolute. When load_time_only,
    imports nested in functions / classes / try blocks (deferred) are skipped.
    """
    base_pkg = _pkg_of(this_mod, is_pkg)
    tree = ast.parse(src)

    class V(ast.NodeVisitor):
        def __init__(self):
            self.depth = 0
            self.hits: list = []

        def _descend(self, node):
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        visit_FunctionDef = _descend
        visit_AsyncFunctionDef = _descend
        # NOTE: ClassDef bodies and module-level Try blocks execute at import
        # time, so they are NOT deferred. Only function bodies defer.

        def visit_Import(self, node):
            if load_time_only and self.depth:
                return
            for a in node.names:
                if a.name == TOP or a.name.startswith(TOP + "."):
                    self.hits.append(("import", a.name, []))

        def visit_ImportFrom(self, node):
            if load_time_only and self.depth:
                return
            if node.level == 0:
                m = node.module or ""
            else:
                m = _resolve_relative(base_pkg, node.level, node.module)
            if m == TOP or m.startswith(TOP + "."):
                self.hits.append(("from", m, [a.name for a in node.names]))

    v = V()
    v.visit(tree)
    return v.hits


def resolve_imports(src: str, this_mod: str, is_pkg: bool,
                    load_time_only: bool = False) -> set[str]:
    """All in-tree absolute module names referenced (for test-eligibility)."""
    out: set[str] = set()
    for kind, mod, names in _iter_imports(src, this_mod, is_pkg, load_time_only):
        out.add(mod)
        if kind == "from" and mod != TOP:
            for n in names:
                out.add(f"{mod}.{n}")
    return out


def module_deps(src: str, this_mod: str, is_pkg: bool,
                exports: dict[str, str], root: Path) -> set[str]:
    """
    Load-time in-tree *modules to vendor* for `src`. Bare `from toolboxv2
    import X` symbols are resolved to their source module via `exports`; the
    heavy top `__init__` itself is never vendored (a shim replaces it).
    """
    deps: set[str] = set()
    for kind, mod, names in _iter_imports(src, this_mod, is_pkg, load_time_only=True):
        if mod == TOP:
            for n in names:
                if n in FRAMEWORK_STUBS:
                    continue
                srcmod = exports.get(n)
                if srcmod:
                    deps.add(srcmod)
            continue  # never vendor toolboxv2/__init__ body
        if mod_to_path(mod, root):
            deps.add(mod)
        if kind == "from":
            for n in names:
                if mod_to_path(f"{mod}.{n}", root):
                    deps.add(f"{mod}.{n}")
    return deps


# ── __init__.py export map (symbol -> source module) ─────────────────────────


def parse_init_exports(root: Path) -> dict[str, str]:
    """
    Read toolboxv2/__init__.py and map each re-exported symbol to its absolute
    source module, e.g. {'Result': 'toolboxv2.utils.system.types', ...}.
    """
    init = root / TOP / "__init__.py"
    exports: dict[str, str] = {}
    tree = ast.parse(init.read_text(encoding="utf-8", errors="replace"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level and node.module:
            anchor = TOP  # __init__ is the toolboxv2 package itself
            target = f"{anchor}.{node.module}"
            for a in node.names:
                exports[a.asname or a.name] = target
    return exports


# ── crawl ────────────────────────────────────────────────────────────────────


def crawl(entry: Path, root: Path, exports: dict[str, str]) -> dict[str, Path]:
    """Transitive load-time in-tree closure. Returns {module_name: source_path}."""
    seen: dict[str, Path] = {}
    entry = entry.resolve()
    root = root.resolve()
    start = path_to_mod(entry, root)
    stack = [(start, entry)]
    while stack:
        mod, path = stack.pop()
        if mod in seen:
            continue
        seen[mod] = path
        src = path.read_text(encoding="utf-8", errors="replace")
        for dep in module_deps(src, mod, is_package_path(path), exports, root):
            if dep in seen:
                continue
            dpath = mod_to_path(dep, root)
            if dpath is not None:
                stack.append((dep, dpath))
    return seen


# ── namespace rewrite (toolboxv2. -> tbv., relative -> absolute tbv.) ─────────


SHIM = f"{NS}._shim"


def _map_mod(name: str) -> str:
    """toolboxv2 -> tbv._shim ; toolboxv2.x.y -> tbv.x.y"""
    if name == TOP:
        return SHIM
    return NS + name[len(TOP):]


class _Rewriter(ast.NodeTransformer):
    def __init__(self, this_mod: str, is_pkg: bool):
        self.base_pkg = _pkg_of(this_mod, is_pkg)

    def visit_Import(self, node: ast.Import):
        for a in node.names:
            if a.name == TOP or a.name.startswith(TOP + "."):
                # `import toolboxv2` -> `import tbv._shim as toolboxv2`
                if a.name == TOP and not a.asname:
                    a.asname = TOP
                a.name = _map_mod(a.name)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.level == 0:
            m = node.module or ""
            if m == TOP or m.startswith(TOP + "."):
                node.module = _map_mod(m)
            return node
        # resolve relative to absolute, then swap top namespace
        anchor = self.base_pkg
        for _ in range(node.level - 1):
            anchor = anchor.rsplit(".", 1)[0] if "." in anchor else ""
        target = f"{anchor}.{node.module}" if node.module else anchor
        if target == TOP or target.startswith(TOP + "."):
            node.module = _map_mod(target)
            node.level = 0
        return node


def rewrite_source(src: str, this_mod: str, is_pkg: bool) -> str:
    tree = ast.parse(src)
    tree = _Rewriter(this_mod, is_pkg).visit(tree)
    ast.fix_missing_locations(tree)
    out = ast.unparse(tree)
    # Catch string/attribute refs to the top namespace that are import-like.
    out = re.sub(rf'(["\'`]){TOP}(\.|["\'`])', rf"\1{NS}\2", out)
    return out


# ── tbv/__init__.py generation ───────────────────────────────────────────────


def gen_shim(exports: dict[str, str], available: set[str]) -> str:
    """
    Mirror toolboxv2/__init__ re-exports against vendored modules. Symbols whose
    source module is not in the packed set, or that are framework-bound, become
    stubs that raise on use.
    """
    lines = [
        "# Auto-generated by tb_atomic. Do not edit.",
        "import logging",
        "from pathlib import Path",
        "get_logger = logging.getLogger",
        "# Repo-root analogue: in a packed install, asset paths like",
        "# tb_root_dir / 'tbjs/dist/...' resolve inside the installed tbv/ dir",
        "# (assets vendored there by tb_atomic's _ASSETS manifest).",
        "tb_root_dir = __tb_root_dir__ = Path(__file__).parent",
        "",
        "class _TBUnavailable:",
        "    __slots__ = ('_n',)",
        "    def __init__(self, n): self._n = n",
        "    def __getattr__(self, k):",
        "        raise ImportError(",
        "            f'tbv: {self._n!r} needs a full ToolBoxV2 install '",
        "            '(framework-bound, not packed by tb_atomic).')",
        "    def __call__(self, *a, **k):",
        "        raise ImportError(",
        "            f'tbv: {self._n!r} needs a full ToolBoxV2 install.')",
        "",
    ]
    by_mod: dict[str, list[str]] = {}
    stubbed: list[str] = []
    for sym, mod in sorted(exports.items()):
        ns_mod = NS + mod[len(TOP):]
        if sym in FRAMEWORK_STUBS or mod not in available:
            stubbed.append(sym)
        else:
            by_mod.setdefault(ns_mod, []).append(sym)
    for ns_mod, syms in sorted(by_mod.items()):
        lines.append(f"from {ns_mod} import {', '.join(sorted(syms))}")
    for sym in stubbed:
        if sym in ("get_logger", "setup_logging"):
            continue
        lines.append(f"{sym} = _TBUnavailable({sym!r})")
    lines.append("")
    keep = sorted(set(exports) | {"get_logger"})
    lines.append("__all__ = " + repr(keep))
    lines.append("")
    return "\n".join(lines)


# import-name -> PyPI distribution name, where they differ
_DIST_MAP = {"faiss": "faiss-cpu", "cv2": "opencv-python", "PIL": "pillow",
             "yaml": "pyyaml", "sklearn": "scikit-learn", "bs4": "beautifulsoup4",
             "dotenv": "python-dotenv", "dateutil": "python-dateutil",
             "google": "google-auth", "googleapiclient": "google-api-python-client",
             "google_auth_oauthlib": "google-auth-oauthlib",
             "jose": "python-jose", "docx": "python-docx", "fitz": "pymupdf",
             "socks": "pysocks", "OpenSSL": "pyopenssl", "Crypto": "pycryptodome",
             "websocket": "websocket-client", "jwt": "pyjwt",
             "magic": "python-magic"}

# Import roots that resolve via local sys.path hacks / vendored dirs and must
# never be emitted as PyPI dependencies (installing the same-named PyPI dist
# would be wrong or malicious-typosquat territory).
_NOT_PYPI = {"models"}

_IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError"}


def _classify_externals(src: str, std: set[str]) -> tuple[set[str], set[str], set[str]]:
    """
    Return (hard, optional, lazy) external import roots for one module:
      hard     = unguarded, module-level (executes at import, required)
      optional = inside a try/except(ImportError) block anywhere (feature-gated)
      lazy     = unguarded, inside a function body (required-but-deferred or opt)
    """
    hard: set[str] = set()
    optional: set[str] = set()
    lazy: set[str] = set()
    tree = ast.parse(src)

    def root_of(name: str) -> str | None:
        r = name.split(".")[0]
        if not r or r == TOP or r in std or r.startswith("_"):
            return None
        return r

    class V(ast.NodeVisitor):
        def __init__(self):
            self.fn_depth = 0
            self.guard_depth = 0

        def _fn(self, n):
            self.fn_depth += 1
            self.generic_visit(n)
            self.fn_depth -= 1

        visit_FunctionDef = _fn
        visit_AsyncFunctionDef = _fn

        def visit_Try(self, n):
            guards = any(
                (h.type is None)
                or (isinstance(h.type, ast.Name) and h.type.id in _IMPORT_ERRORS)
                or (isinstance(h.type, ast.Tuple) and any(
                    isinstance(e, ast.Name) and e.id in _IMPORT_ERRORS for e in h.type.elts))
                for h in n.handlers
            )
            if guards:
                self.guard_depth += 1
                for stmt in n.body:
                    self.visit(stmt)
                self.guard_depth -= 1
                for stmt in n.orelse + n.finalbody:
                    self.visit(stmt)
                for h in n.handlers:
                    self.visit(h)
            else:
                self.generic_visit(n)

        def _names(self, node):
            if isinstance(node, ast.Import):
                return [a.name for a in node.names]
            return [node.module] if (node.level == 0 and node.module) else []

        def _emit(self, node):
            for name in self._names(node):
                r = root_of(name)
                if not r:
                    continue
                if self.guard_depth:
                    optional.add(r)
                elif self.fn_depth:
                    lazy.add(r)
                else:
                    hard.add(r)

        visit_Import = _emit
        visit_ImportFrom = _emit

    V().visit(tree)
    return hard, optional, lazy


def external_deps(closure: dict[str, Path]) -> tuple[list[str], list[str], list[str]]:
    """
    Aggregate external deps across the closure into three buckets, mapped to
    PyPI distribution names:
      hard     -> [project.dependencies]
      optional -> [project.optional-dependencies].optional
      full     -> [project.optional-dependencies].full  (everything; test-gate)
    """
    std = set(sys.stdlib_module_names)
    H, O, L = set(), set(), set()
    for path in closure.values():
        h, o, l = _classify_externals(
            path.read_text(encoding="utf-8", errors="replace"), std)
        H |= h; O |= o; L |= l
    optional = O - H
    full = H | O | L
    m = lambda s: sorted(_DIST_MAP.get(x, x) for x in s if x not in _NOT_PYPI)
    return m(H), m(optional), m(full)


# ── packaging ────────────────────────────────────────────────────────────────


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class PackResult:
    leaf_name: str
    entry_mod: str
    core_modules: list[str] = field(default_factory=list)
    leaf_modules: list[str] = field(default_factory=list)
    stubbed_symbols: list[str] = field(default_factory=list)
    out_dir: Path | None = None


def _write_module(pkg_root: Path, mod: str, src: str, is_pkg: bool = False) -> None:
    """Write rewritten source into the tbv namespace tree.

    Flat modules  -> <pkg_root>/tbv/<...>/<name>.py
    Packages      -> <pkg_root>/tbv/<...>/<name>/__init__.py

    Emitting a source package's __init__.py as ``<name>.py`` would shadow the
    sibling ``<name>/`` directory holding its submodules ('extras.py vs
    extras/' collision -> ModuleNotFoundError: '... is not a package').
    Intermediate directories stay PEP420 (no __init__.py), so co-install
    across wheels remains clean; only real source packages get an __init__.py.
    """
    ns_mod = NS + mod[len(TOP):]
    parts = ns_mod.split(".")
    if is_pkg:
        target = pkg_root / Path(*parts) / "__init__.py"
        # A previous (buggy or pre-fix) run may have left the flat shadow file.
        stale = pkg_root / Path(*parts).with_suffix(".py")
        if stale.exists():
            stale.unlink()
    else:
        target = pkg_root / Path(*parts).with_suffix(".py")
        # Inverse collision guard: a stale __init__.py from an earlier run
        # where this module was (mis)classified as a package.
        stale = pkg_root / Path(*parts) / "__init__.py"
        if stale.exists() and not any(
                p for p in stale.parent.iterdir() if p.name != "__init__.py"):
            stale.unlink()
            stale.parent.rmdir()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(src, encoding="utf-8")


def _vendor_assets(pkg_dir: Path, mods: list[str], root: Path) -> bool:
    """Copy _ASSETS files for any packed module into <pkg_dir>/tbv/<relpath>,
    mirroring the repo layout under toolboxv2/ so the shim's tb_root_dir
    (= installed tbv/ dir) resolves them. Returns True if anything copied."""
    copied = False
    for m in mods:
        for rel in _ASSETS.get(m, []):
            src = root / TOP / rel
            if not src.exists():
                print(f"WARNING: asset {rel} for {m} not found at {src}")
                continue
            dst = pkg_dir / NS / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            copied = True
    return copied


def pack(entry: Path, root: Path, out: Path, leaf_name: str | None = None,
         version: str = "0.1.0") -> PackResult:
    root = root.resolve()
    entry = entry.resolve()
    exports = parse_init_exports(root)
    closure = crawl(entry, root, exports)
    entry_mod = path_to_mod(entry, root)
    export_src_mods = set(exports.values())

    available = set(closure)
    # core = closure modules that are __init__-export sources (shared base).
    # Core MUST be import-closed: leaves only pin tbv-core, so every load-time
    # in-tree dep of a core module has to live in core too — otherwise
    # installing core + a different leaf breaks (e.g. manifest.loader in core
    # needing manifest.schema that only shipped inside another leaf).
    core_set = {m for m in closure if m in export_src_mods}
    grew = True
    while grew:
        grew = False
        for m in sorted(core_set):
            src_m = closure[m].read_text(encoding="utf-8", errors="replace")
            for dep in module_deps(src_m, m, is_package_path(closure[m]),
                                   exports, root):
                if dep in closure and dep not in core_set:
                    core_set.add(dep)
                    grew = True
    core = sorted(core_set)
    leaf = sorted(m for m in closure if m not in core_set)

    leaf_name = leaf_name or "tbv-" + entry_mod.split(".")[-1].replace("_", "-")
    out.mkdir(parents=True, exist_ok=True)

    core_dir = out / "tbv-core"
    leaf_dir = out / leaf_name

    # The shim (top-level symbol mirror) lives in core if there is one,
    # otherwise in the leaf. It is always present because deferred
    # `from toolboxv2 import X` refs are rewritten to `from tbv._shim import X`.
    shim_src = gen_shim(exports, available)

    # ---- core package (monotonic superset; merge, never shrink) ----
    core_pin = None
    if core:
        (core_dir / NS).mkdir(parents=True, exist_ok=True)
        manifest = _load_core_manifest(core_dir)
        prev_mods = dict(manifest.get("modules", {}))
        changed: list[str] = []
        added: list[str] = []
        for m in core:
            rsrc = rewrite_source(
                closure[m].read_text(encoding="utf-8", errors="replace"),
                m, is_package_path(closure[m]))
            sha = _sha(rsrc)
            _write_module(core_dir, m, rsrc, is_package_path(closure[m]))
            if m not in prev_mods:
                added.append(m)
            elif prev_mods[m] != sha:
                changed.append(m)
            prev_mods[m] = sha
        # shim reflects ALL core modules ever merged (superset), not just this pack
        merged_core = set(prev_mods)
        (core_dir / NS / "_shim.py").write_text(
            gen_shim(exports, available | merged_core), encoding="utf-8")
        old_ver = manifest.get("version")
        core_ver = old_ver or "0.1.0"
        if old_ver and (added or changed):
            core_ver = _bump_patch(old_ver)
        # Core carries its own external hard deps (it is import-closed and its
        # modules execute real imports at load time, e.g. security.cryp ->
        # python-dotenv). Merge monotonically like the module set: modules from
        # earlier merges may not be in this closure, so never drop their deps.
        core_hard, _co, _cf = external_deps({m: closure[m] for m in core})
        merged_deps = sorted(set(manifest.get("deps", [])) | set(core_hard))
        core_assets = _vendor_assets(core_dir, core, root)
        _save_core_manifest(core_dir, core_ver, prev_mods, merged_deps)
        _write_pyproject(core_dir, "tbv-core", core_ver, deps=merged_deps,
                         assets=core_assets)
        core_pin = f"tbv-core>={core_ver}"
        if changed:
            print(f"WARNING: tbv-core modules changed (shared!): {changed} "
                  f"-> bumped to {core_ver}; verify back-compat for dependents.")

    # ---- leaf package ----
    (leaf_dir / NS).mkdir(parents=True, exist_ok=True)
    for m in leaf:
        src = closure[m].read_text(encoding="utf-8", errors="replace")
        is_pkg = is_package_path(closure[m])
        _write_module(leaf_dir, m, rewrite_source(src, m, is_pkg), is_pkg)
    # hard/optional scoped to the leaf's own modules (core declares its own);
    # `full` stays closure-wide so the test-gate venv can exercise everything.
    hard, optional, _ = external_deps({m: closure[m] for m in leaf})
    _, _, full = external_deps(closure)
    if core:
        deps = [core_pin] + hard
    else:
        (leaf_dir / NS / "_shim.py").write_text(shim_src, encoding="utf-8")
        deps = hard
    leaf_assets = _vendor_assets(leaf_dir, leaf, root)
    _write_pyproject(leaf_dir, leaf_name, version, deps=deps,
                     optional=optional, full=full, assets=leaf_assets)

    stubbed = sorted(
        s for s, m in exports.items()
        if s in FRAMEWORK_STUBS or m not in available
    )
    return PackResult(
        leaf_name=leaf_name, entry_mod=entry_mod,
        core_modules=core, leaf_modules=leaf, stubbed_symbols=stubbed, out_dir=out,
    )


def _bump_patch(ver: str) -> str:
    parts = (ver.split(".") + ["0", "0", "0"])[:3]
    try:
        parts[2] = str(int(parts[2]) + 1)
    except ValueError:
        parts[2] = "1"
    return ".".join(parts)


def _load_core_manifest(core_dir: Path) -> dict:
    f = core_dir / ".tbv_core.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def _save_core_manifest(core_dir: Path, version: str, modules: dict,
                        deps: list[str] | None = None) -> None:
    (core_dir / ".tbv_core.json").write_text(
        json.dumps({"version": version, "modules": modules,
                    "deps": sorted(deps or [])}, indent=2,
                   sort_keys=True), encoding="utf-8")


def _write_pyproject(pkg_dir: Path, name: str, version: str, deps: list[str],
                     optional: list[str] | None = None,
                     full: list[str] | None = None,
                     assets: bool = False) -> None:
    dep_str = ", ".join(f'"{d}"' for d in deps)
    body = (
        "[build-system]\n"
        'requires = ["setuptools>=61"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        "[project]\n"
        f'name = "{name}"\n'
        f'version = "{version}"\n'
        f'description = "tb_atomic-packed ToolBoxV2 component ({name})."\n'
        'requires-python = ">=3.10"\n'
        f"dependencies = [{dep_str}]\n"
    )
    opt_lines = []
    if optional:
        opt_lines.append("optional = [" + ", ".join(f'"{d}"' for d in optional) + "]")
    if full:
        opt_lines.append("full = [" + ", ".join(f'"{d}"' for d in full) + "]")
    if opt_lines:
        body += "\n[project.optional-dependencies]\n" + "\n".join(opt_lines) + "\n"
    body += (
        "\n[tool.setuptools.packages.find]\n"
        f'include = ["{NS}*"]\n'
        "namespaces = true\n"
    )
    if assets:
        body += (
            "\n[tool.setuptools.package-data]\n"
            '"*" = ["*.css", "*.js", "*.json", "*.html", "*.svg", "*.txt"]\n'
        )
    (pkg_dir / "pyproject.toml").write_text(body, encoding="utf-8")


# ── test selection (import-closed) ───────────────────────────────────────────


def select_tests(closure: set[str], tests_root: Path, repo_root: Path) -> list[Path]:
    """
    A test file is eligible if every in-tree module it imports is inside the
    packed closure (so it can run against the standalone copy).
    """
    eligible: list[Path] = []
    packed = set(closure)
    for tf in tests_root.rglob("test_*.py"):
        try:
            src = tf.read_text(encoding="utf-8", errors="replace")
            mod = path_to_mod(tf, repo_root)
            imports = resolve_imports(src, mod, is_package_path(tf), load_time_only=True)
        except (SyntaxError, ValueError):
            continue
        intree = {i for i in imports if mod_to_path(i, repo_root)}
        if not intree:
            continue
        # only keep tests that touch the packed set and stay within it
        if intree & packed and intree <= (packed | {TOP}):
            eligible.append(tf)
    return eligible


# ── CLI ──────────────────────────────────────────────────────────────────────


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tb_atomic")
    sub = p.add_subparsers(dest="cmd", required=True)
    pk = sub.add_parser("pack", help="Pack an entry file into a standalone dist.")
    pk.add_argument("entry")
    pk.add_argument("--root", default=".", help="Repo root containing toolboxv2/.")
    pk.add_argument("-o", "--out", default="dist_atomic")
    pk.add_argument("--name", default=None)
    pk.add_argument("--version", default="0.1.0", help="Leaf package version.")
    args = p.parse_args(argv)

    if args.cmd == "pack":
        res = pack(Path(args.entry), Path(args.root), Path(args.out),
                   args.name, args.version)
        print(json.dumps({
            "leaf": res.leaf_name,
            "entry": res.entry_mod,
            "core_modules": res.core_modules,
            "leaf_modules": res.leaf_modules,
            "stubbed": res.stubbed_symbols,
            "out": str(res.out_dir),
        }, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
