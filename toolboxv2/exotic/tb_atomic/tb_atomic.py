"""
tb_atomic — isolated, install-free import of ToolBoxV2 components.

Fetches individual modules (and their transitive in-tree dependencies) from
remote sources (gist / GitHub raw / a registry endpoint) at import time,
caches them to disk, and exposes them under their original dotted names so
that ``from toolboxv2.mods.isaa.memory_layer import MemoryLayer`` works
without the full ToolBoxV2 install.

Two layers cooperate:

  * Remote modules  — real source pulled over HTTP, hash-pinned, cached.
  * Stubs           — lightweight stand-ins for top-level TB symbols
                      (e.g. ``toolboxv2.get_logger``) that would otherwise
                      drag in the whole framework.

Intermediate packages (``toolboxv2``, ``toolboxv2.mods``, ...) are synthesised
as empty namespace packages on demand so submodule imports resolve.

CLI:
    python -m tb_atomic crawl <entry.py> [--root DIR] [--base URL] [-o atomic.json]
    python -m tb_atomic clear-cache [MODULE ...]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.abc
import importlib.util
import json
import logging
import sys
import time
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any, Callable

__version__ = "0.1.0"
__all__ = [
    "register",
    "register_stub",
    "invalidate",
    "load_manifest",
    "clear_cache",
    "configure",
]

# ── Configuration ────────────────────────────────────────────────────────────

CACHE_DIR = Path("~/.tb_atomic_cache").expanduser()


def configure(cache_dir: str | Path | None = None) -> None:
    """Override the on-disk cache directory."""
    global CACHE_DIR
    if cache_dir is not None:
        CACHE_DIR = Path(cache_dir).expanduser()


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── Registries ───────────────────────────────────────────────────────────────

# module_name -> {url, sha256, version, ttl}
_REGISTRY: dict[str, dict] = {}
# module_name -> {attr_name: value}
_STUB_REGISTRY: dict[str, dict[str, Any]] = {}


# ── Cache helpers ────────────────────────────────────────────────────────────


def _safe(name: str) -> str:
    return name.replace(".", "_")


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{_safe(name)}.py"


def _meta_path(name: str) -> Path:
    return CACHE_DIR / f"{_safe(name)}.meta.json"


def _load_meta(name: str) -> dict:
    p = _meta_path(name)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (ValueError, OSError):
            return {}
    return {}


def _save_meta(name: str, meta: dict) -> None:
    _meta_path(name).write_text(json.dumps(meta), encoding="utf-8", errors="replace")


def _cache_valid(name: str) -> bool:
    """A cache entry is valid if the file exists and its TTL has not expired."""
    if not _cache_path(name).exists():
        return False
    meta = _load_meta(name)
    ttl = meta.get("ttl", 0)
    if ttl and (time.time() - meta.get("fetched_at", 0)) > ttl:
        return False
    # Version mismatch between registry and cache forces a re-fetch.
    reg = _REGISTRY.get(name, {})
    if reg.get("version") and meta.get("version") != reg["version"]:
        return False
    return True


# ── Public API ───────────────────────────────────────────────────────────────


def register(
    module_name: str,
    url: str,
    *,
    sha256: str | None = None,
    version: str | None = None,
    ttl: int = 0,
    is_package: bool = False,
    force: bool = False,
) -> None:
    """
    Register a remote source for a dotted module name.

    ttl        — seconds before the cached copy is stale (0 = never).
    is_package — treat the module as a package (sets __path__).
    force      — drop any cached copy immediately so the next import re-fetches.
    """
    _REGISTRY[module_name] = {
        "url": url,
        "sha256": sha256,
        "version": version,
        "ttl": ttl,
        "is_package": is_package,
    }
    if force:
        invalidate(module_name)


def register_stub(module_name: str, **names: Any) -> None:
    """
    Register a lightweight stub module exposing the given attributes.

        register_stub("toolboxv2", get_logger=logging.getLogger)
    """
    _STUB_REGISTRY[module_name] = dict(names)


def invalidate(module_name: str) -> None:
    """Force a re-fetch of a single module on its next import."""
    _cache_path(module_name).unlink(missing_ok=True)
    _meta_path(module_name).unlink(missing_ok=True)
    sys.modules.pop(module_name, None)


def clear_cache(*module_names: str) -> None:
    """Clear the whole cache, or only the named modules."""
    if module_names:
        for name in module_names:
            invalidate(name)
        return
    if CACHE_DIR.exists():
        for p in CACHE_DIR.glob("*"):
            p.unlink(missing_ok=True)
    for name in list(sys.modules):
        if name in _REGISTRY or name in _STUB_REGISTRY:
            sys.modules.pop(name, None)


def load_manifest(
    url_or_path: str,
    *,
    force: bool = False,
    prefetch: bool = False,
) -> dict:
    """
    Load an ``atomic.json`` manifest and register every module + stub it lists.

    Manifest shape::

        {
          "version": "0.4.2",
          "base_url": "https://raw.githubusercontent.com/.../main",
          "modules": {
            "toolboxv2.utils.types": {"path": "toolboxv2/utils/types.py",
                                      "sha256": "...", "deps": [...]},
            ...
          },
          "stubs": {"toolboxv2": ["get_logger", "get_app"]}
        }

    ``url`` may be given per-module, otherwise it is built from ``base_url`` +
    ``path``. force/prefetch behave as on :func:`register`.
    """
    if url_or_path.startswith(("http://", "https://")):
        import requests

        data = requests.get(url_or_path, timeout=15).json()
    else:
        data = json.loads(
            Path(url_or_path).read_text(encoding="utf-8", errors="replace")
        )

    base = data.get("base_url", "").rstrip("/")
    version = data.get("version")

    for name, entry in data.get("modules", {}).items():
        url = entry.get("url")
        if not url and base and entry.get("path"):
            url = f"{base}/{entry['path']}"
        if not url:
            raise ValueError(
                f"manifest entry {name!r} has neither url nor base_url+path"
            )
        register(
            name,
            url,
            sha256=entry.get("sha256"),
            version=entry.get("version", version),
            ttl=entry.get("ttl", 0),
            is_package=entry.get("is_package", False),
            force=force,
        )

    for stub_mod, attrs in data.get("stubs", {}).items():
        register_stub(stub_mod, **{a: _default_stub_attr(stub_mod, a) for a in attrs})

    if prefetch:
        for name in data.get("modules", {}):
            importlib.import_module(name)

    return data


def _default_stub_attr(module_name: str, attr: str) -> Callable[..., Any]:
    """Best-effort default stub: known names get real behaviour, rest raise."""
    if attr in ("get_logger", "getLogger"):
        return logging.getLogger

    def _unavailable(*_a: Any, **_k: Any) -> Any:
        raise ImportError(
            f"tb_atomic stub: '{module_name}.{attr}' is not available without a "
            f"full ToolBoxV2 install. Register a real stub via "
            f"tb_atomic.register_stub({module_name!r}, {attr}=...)."
        )

    return _unavailable


# ── Loaders ──────────────────────────────────────────────────────────────────


class _RemoteLoader(importlib.abc.Loader):
    def __init__(self, name: str, entry: dict):
        self.name = name
        self.entry = entry

    def create_module(self, spec: ModuleSpec):  # noqa: D401 - default semantics
        return None

    def exec_module(self, module) -> None:
        _ensure_cache_dir()
        cp = _cache_path(self.name)

        if not _cache_valid(self.name):
            import requests

            resp = requests.get(self.entry["url"], timeout=15)
            resp.raise_for_status()
            code = resp.text
            expected = self.entry.get("sha256")
            if expected:
                got = hashlib.sha256(code.encode("utf-8")).hexdigest()
                if got != expected:
                    raise ImportError(
                        f"tb_atomic: sha256 mismatch for {self.name} "
                        f"(expected {expected[:12]}…, got {got[:12]}…)"
                    )
            cp.write_text(code, encoding="utf-8", errors="replace")
            _save_meta(
                self.name,
                {
                    "fetched_at": time.time(),
                    "ttl": self.entry.get("ttl", 0),
                    "version": self.entry.get("version"),
                    "url": self.entry["url"],
                },
            )

        source = cp.read_text(encoding="utf-8", errors="replace")
        module.__file__ = str(cp)
        if _is_package(self.name, self.entry) and not hasattr(module, "__path__"):
            module.__path__ = []  # type: ignore[attr-defined]
        exec(compile(source, str(cp), "exec"), module.__dict__)


class _StubLoader(importlib.abc.Loader):
    def __init__(self, name: str, names: dict[str, Any]):
        self.name = name
        self.names = names

    def create_module(self, spec: ModuleSpec):
        return None

    def exec_module(self, module) -> None:
        # Acts as a package too, so submodule imports keep resolving.
        if not hasattr(module, "__path__"):
            module.__path__ = []  # type: ignore[attr-defined]
        for key, value in self.names.items():
            setattr(module, key, value)


class _NamespaceLoader(importlib.abc.Loader):
    """Synthesises an empty package so dotted-parent imports succeed."""

    def create_module(self, spec: ModuleSpec):
        return None

    def exec_module(self, module) -> None:
        if not hasattr(module, "__path__"):
            module.__path__ = []  # type: ignore[attr-defined]


# ── Finder ───────────────────────────────────────────────────────────────────


def _is_package(fullname: str, entry: dict) -> bool:
    """A registered module is a package if it is an __init__ file or a parent."""
    if entry.get("is_package"):
        return True
    url = entry.get("url", "")
    if url.endswith(("/__init__.py", "__init__.py")):
        return True
    prefix = fullname + "."
    managed = set(_REGISTRY) | set(_STUB_REGISTRY)
    return any(name.startswith(prefix) for name in managed)


class _TbAtomicFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):  # noqa: D401
        # Stubs win over remote sources: a deliberately registered stand-in
        # must override the real (heavy) module of the same name.
        if fullname in _STUB_REGISTRY:
            loader = _StubLoader(fullname, _STUB_REGISTRY[fullname])
            spec = ModuleSpec(fullname, loader, origin=f"<tb_atomic stub:{fullname}>")
            spec.submodule_search_locations = []  # mark as package
            return spec

        if fullname in _REGISTRY:
            entry = _REGISTRY[fullname]
            loader = _RemoteLoader(fullname, entry)
            spec = ModuleSpec(fullname, loader, origin=entry["url"])
            if _is_package(fullname, entry):
                spec.submodule_search_locations = []  # mark as package
            return spec

        # Parent package of something we manage → synthesise a namespace pkg.
        prefix = fullname + "."
        managed = set(_REGISTRY) | set(_STUB_REGISTRY)
        if any(name.startswith(prefix) for name in managed):
            spec = ModuleSpec(fullname, _NamespaceLoader(), origin="<tb_atomic ns>")
            spec.submodule_search_locations = []
            return spec

        return None


def _install() -> None:
    if not any(isinstance(f, _TbAtomicFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _TbAtomicFinder())


_install()


# ── AST crawler ──────────────────────────────────────────────────────────────


def _extract_imports(source: str) -> list[str]:
    tree = ast.parse(source)
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                out.append(node.module)
    return out


def crawl(
    entry_file: str | Path,
    root: str | Path = "toolboxv2",
    top_package: str = "toolboxv2",
    base_url: str = "",
    _seen: dict[str, Path] | None = None,
) -> dict[str, Path]:
    """
    Statically resolve all transitive in-tree imports of ``entry_file``.

    Returns ``{module_name: source_path}`` for every module under
    ``top_package`` reachable from the entry file. External / stdlib imports
    are ignored.
    """
    root = Path(root)
    if _seen is None:
        _seen = {}
    entry_file = Path(entry_file)
    src = entry_file.read_text(encoding="utf-8", errors="replace")

    for imp in _extract_imports(src):
        if not imp.startswith(top_package):
            continue
        if imp in _seen:
            continue
        rel = imp.replace(".", "/")
        # root already includes the top package directory; strip the prefix.
        sub = rel[len(top_package) + 1 :] if rel != top_package else ""
        candidates = [
            root / f"{sub}.py" if sub else root / "__init__.py",
            root / sub / "__init__.py",
        ]
        for cand in candidates:
            if cand.exists():
                _seen[imp] = cand
                crawl(cand, root, top_package, base_url, _seen)
                break
    return _seen


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_text(encoding="utf-8", errors="replace").encode("utf-8")
    ).hexdigest()


def _path_to_module(path: Path, root: Path, top_package: str) -> str:
    """toolboxv2/mods/isaa/memory_layer.py -> toolboxv2.mods.isaa.memory_layer"""
    rel = path.relative_to(root.parent) if root.parent in path.parents else path
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _build_manifest(
    entry_file: str | Path,
    root: str | Path,
    top_package: str,
    base_url: str,
    version: str | None,
) -> dict:
    root = Path(root)
    entry_file = Path(entry_file)
    deps = crawl(entry_file, root, top_package, base_url)
    # The entry module itself must be importable, not only its dependencies.
    deps.setdefault(_path_to_module(entry_file, root, top_package), entry_file)
    modules: dict[str, dict] = {}
    for name, path in deps.items():
        rel = path.relative_to(root.parent) if root.parent in path.parents else path
        entry = {"path": str(rel).replace("\\", "/"), "sha256": _sha256_file(path)}
        if path.name == "__init__.py":
            entry["is_package"] = True
        modules[name] = entry
    manifest: dict[str, Any] = {"modules": modules}
    if version:
        manifest["version"] = version
    if base_url:
        manifest["base_url"] = base_url.rstrip("/")
    return manifest


# ── CLI ──────────────────────────────────────────────────────────────────────


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tb_atomic")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_crawl = sub.add_parser("crawl", help="Build an atomic.json manifest via AST.")
    p_crawl.add_argument("entry", help="Entry .py file to start from.")
    p_crawl.add_argument(
        "--root",
        default="toolboxv2",
        help="Package source root dir (default: toolboxv2).",
    )
    p_crawl.add_argument(
        "--top",
        default="toolboxv2",
        help="Top-level package name (default: toolboxv2).",
    )
    p_crawl.add_argument(
        "--base",
        default="",
        help="base_url for raw fetches, e.g. "
        "https://raw.githubusercontent.com/USER/REPO/main",
    )
    p_crawl.add_argument("--version", default=None, help="Manifest version tag.")
    p_crawl.add_argument("-o", "--out", default="atomic.json", help="Output file.")

    p_clear = sub.add_parser("clear-cache", help="Clear the on-disk cache.")
    p_clear.add_argument("modules", nargs="*", help="Specific modules (default: all).")

    args = parser.parse_args(argv)

    if args.cmd == "crawl":
        manifest = _build_manifest(
            args.entry, args.root, args.top, args.base, args.version
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).touch(exist_ok=True)
        Path(args.out).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8", errors="replace"
        )
        print(f"Wrote {args.out} with {len(manifest['modules'])} modules.")
        return 0

    if args.cmd == "clear-cache":
        clear_cache(*args.modules)
        target = ", ".join(args.modules) if args.modules else "all"
        print(f"Cleared cache: {target}.")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
