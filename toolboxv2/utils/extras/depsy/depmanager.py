"""depmanager — cross-platform system package manager abstraction.

Detects available system package managers (winget, brew, apt, dnf, pacman,
zypper, choco, scoop, nix, uv, pip), lets the user pick a preferred one,
and exposes a unified functional API: search, install, uninstall, update, list.

Preferred manager: arg > env DEPX_MANAGER > auto-detect (first available).
sudo auto-prefixed on Linux when needed, with fallback + warning.
Sync core; optional threaded() wrapper for async-safe usage.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional, TypeVar

from toolboxv2 import get_logger

log = get_logger()

T = TypeVar("T")

# ── thread wrapper ──────────────────────────────────────────────────────────

_pool = ThreadPoolExecutor(max_workers=2)


def threaded(fn: Callable[..., T], *args, **kwargs) -> Future[T]:
    """Run any sync depmanager function in a background thread. Returns Future."""
    return _pool.submit(fn, *args, **kwargs)


# ── manager definitions ────────────────────────────────────────────────────

@dataclass(frozen=True)
class ManagerDef:
    name: str
    binary: str
    needs_sudo: bool = False  # typical; actual check is dynamic
    # command templates — {pkg} is replaced at call time
    search_cmd: tuple[str, ...] = ()
    install_cmd: tuple[str, ...] = ()
    uninstall_cmd: tuple[str, ...] = ()
    update_cmd: tuple[str, ...] = ()
    list_cmd: tuple[str, ...] = ()
    # flags that accept prompts / licenses silently
    accept_flags: tuple[str, ...] = ()


MANAGERS: dict[str, ManagerDef] = {}


def _reg(m: ManagerDef) -> None:
    MANAGERS[m.name] = m


_reg(ManagerDef(
    "winget", "winget",
    search_cmd=("winget", "search", "{pkg}"),
    install_cmd=("winget", "install", "--accept-source-agreements",
                 "--accept-package-agreements", "{pkg}"),
    uninstall_cmd=("winget", "uninstall", "{pkg}"),
    update_cmd=("winget", "upgrade", "{pkg}"),
    list_cmd=("winget", "list"),
))

_reg(ManagerDef(
    "choco", "choco",
    install_cmd=("choco", "install", "-y", "{pkg}"),
    uninstall_cmd=("choco", "uninstall", "-y", "{pkg}"),
    update_cmd=("choco", "upgrade", "-y", "{pkg}"),
    search_cmd=("choco", "search", "{pkg}"),
    list_cmd=("choco", "list", "--local-only"),
))

_reg(ManagerDef(
    "scoop", "scoop",
    search_cmd=("scoop", "search", "{pkg}"),
    install_cmd=("scoop", "install", "{pkg}"),
    uninstall_cmd=("scoop", "uninstall", "{pkg}"),
    update_cmd=("scoop", "update", "{pkg}"),
    list_cmd=("scoop", "list"),
))

_reg(ManagerDef(
    "brew", "brew",
    search_cmd=("brew", "search", "{pkg}"),
    install_cmd=("brew", "install", "{pkg}"),
    uninstall_cmd=("brew", "uninstall", "{pkg}"),
    update_cmd=("brew", "upgrade", "{pkg}"),
    list_cmd=("brew", "list"),
))

_reg(ManagerDef(
    "apt", "apt",
    needs_sudo=True,
    search_cmd=("apt", "search", "{pkg}"),
    install_cmd=("apt", "install", "-y", "{pkg}"),
    uninstall_cmd=("apt", "remove", "-y", "{pkg}"),
    update_cmd=("apt", "install", "--only-upgrade", "-y", "{pkg}"),
    list_cmd=("apt", "list", "--installed"),
))

_reg(ManagerDef(
    "dnf", "dnf",
    needs_sudo=True,
    search_cmd=("dnf", "search", "{pkg}"),
    install_cmd=("dnf", "install", "-y", "{pkg}"),
    uninstall_cmd=("dnf", "remove", "-y", "{pkg}"),
    update_cmd=("dnf", "upgrade", "-y", "{pkg}"),
    list_cmd=("dnf", "list", "installed"),
))

_reg(ManagerDef(
    "pacman", "pacman",
    needs_sudo=True,
    search_cmd=("pacman", "-Ss", "{pkg}"),
    install_cmd=("pacman", "-S", "--noconfirm", "{pkg}"),
    uninstall_cmd=("pacman", "-R", "--noconfirm", "{pkg}"),
    update_cmd=("pacman", "-Syu", "--noconfirm"),
    list_cmd=("pacman", "-Q"),
))

_reg(ManagerDef(
    "zypper", "zypper",
    needs_sudo=True,
    search_cmd=("zypper", "search", "{pkg}"),
    install_cmd=("zypper", "install", "-y", "{pkg}"),
    uninstall_cmd=("zypper", "remove", "-y", "{pkg}"),
    update_cmd=("zypper", "update", "-y", "{pkg}"),
    list_cmd=("zypper", "packages", "--installed-only"),
))

_reg(ManagerDef(
    "nix", "nix-env",
    search_cmd=("nix", "search", "nixpkgs", "{pkg}"),
    install_cmd=("nix-env", "-iA", "nixpkgs.{pkg}"),
    uninstall_cmd=("nix-env", "-e", "{pkg}"),
    update_cmd=("nix-env", "-u", "{pkg}"),
    list_cmd=("nix-env", "-q"),
))

_reg(ManagerDef(
    "uv", "uv",
    search_cmd=("uv", "pip", "search", "{pkg}"),  # may not exist yet
    install_cmd=("uv", "pip", "install", "--system", "--break-system-packages", "{pkg}"),
    uninstall_cmd=("uv", "pip", "uninstall", "--system", "--break-system-packages", "{pkg}"),
    update_cmd=("uv", "pip", "install", "--system", "--break-system-packages", "--upgrade", "{pkg}"),
    list_cmd=("uv", "pip", "list", "--system"),
))

_reg(ManagerDef(
    "pip", "pip",
    search_cmd=("pip", "index", "versions", "{pkg}"),
    install_cmd=("pip", "install", "--break-system-packages", "{pkg}"),
    uninstall_cmd=("pip", "uninstall", "--break-system-packages", "-y", "{pkg}"),
    update_cmd=("pip", "install", "--break-system-packages", "--upgrade", "{pkg}"),
    list_cmd=("pip", "list"),
))


# ── detection ───────────────────────────────────────────────────────────────

# ranked per platform — first match wins as default
_RANK = {
    "Windows": ["winget", "choco", "scoop", "uv", "pip"],
    "Darwin":  ["brew", "nix", "uv", "pip"],
    "Linux":   ["apt", "dnf", "pacman", "zypper", "nix", "brew", "uv", "pip"],
}


def detect_available() -> list[str]:
    """Return names of all available managers, ranked by platform preference."""
    system = platform.system()
    order = _RANK.get(system, list(MANAGERS.keys()))
    # also include any not in rank list (in case user installed e.g. brew on linux)
    all_names = order + [n for n in MANAGERS if n not in order]
    return [n for n in all_names if shutil.which(MANAGERS[n].binary)]


def resolve_manager(manager: Optional[str] = None) -> ManagerDef:
    """Resolve which manager to use: arg > env > auto-detect."""
    name = manager or os.environ.get("DEPX_MANAGER")
    if name:
        name = name.lower().strip()
        if name not in MANAGERS:
            raise ValueError(f"Unknown manager '{name}'. Known: {list(MANAGERS)}")
        if not shutil.which(MANAGERS[name].binary):
            raise RuntimeError(f"Manager '{name}' not found on PATH")
        return MANAGERS[name]
    available = detect_available()
    if not available:
        raise RuntimeError("No supported package manager found on this system")
    return MANAGERS[available[0]]


# ── sudo handling ───────────────────────────────────────────────────────────

def _needs_elevation(mdef: ManagerDef) -> bool:
    """Check if we need sudo for this manager on this system."""
    if not mdef.needs_sudo:
        return False
    if platform.system() == "Windows":
        return False
    if os.geteuid() == 0:
        return False
    return True


def _build_cmd(template: tuple[str, ...], pkg: str, mdef: ManagerDef) -> list[str]:
    """Build command list from template, inject sudo if needed."""
    cmd = [t.replace("{pkg}", pkg) for t in template]
    if _needs_elevation(mdef):
        if shutil.which("sudo"):
            cmd = ["sudo"] + cmd
        else:
            log.warning("sudo not found — running without elevation, may fail")
    return cmd


# ── core execution ──────────────────────────────────────────────────────────

@dataclass
class RunResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    cmd: list[str] = field(default_factory=list)


def _run(cmd: list[str], timeout: int = 300, interactive: bool = False) -> RunResult:
    """Execute a command, return structured result."""
    log.debug("exec: %s", " ".join(cmd))
    try:
        if interactive:
            # live output — stdin/stdout/stderr go to terminal
            r = subprocess.run(cmd, timeout=timeout)
            return RunResult(
                ok=r.returncode == 0,
                returncode=r.returncode,
                stdout="", stderr="",
                cmd=cmd,
            )
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        return RunResult(
            ok=r.returncode == 0,
            returncode=r.returncode,
            stdout=r.stdout,
            stderr=r.stderr,
            cmd=cmd,
        )
    except FileNotFoundError:
        return RunResult(False, -1, "", f"command not found: {cmd[0]}", cmd)
    except subprocess.TimeoutExpired:
        return RunResult(False, -2, "", f"timeout after {timeout}s", cmd)


# ── output parsing ──────────────────────────────────────────────────────────

def _parse_search(raw: str, manager_name: str) -> list[dict[str, str]]:
    """Best-effort parse of search output into list of dicts.

    Each manager has different output format. We extract what we can:
    {id, name, version, source} — fields may be empty.
    """
    results: list[dict[str, str]] = []
    lines = raw.strip().splitlines()

    if manager_name == "winget":
        # winget outputs fixed-width columns with a header + dashes separator
        # strip BOM / progress chars that winget may emit
        cleaned = []
        for line in lines:
            line = line.lstrip("\ufeff\x00")  # BOM
            line = line.replace("\r", "")
            if line.strip():
                cleaned.append(line)
        lines = cleaned

        header_idx = -1
        dash_line = ""
        for i, line in enumerate(lines):
            stripped = line.strip()
            # match a line that is predominantly dashes (may have spaces between column groups)
            if len(stripped) > 10 and stripped.replace("-", "").replace(" ", "") == "":
                header_idx = i
                dash_line = line
                break
        if header_idx < 1:
            pass
        else:
            # find where each dash-group starts
            col_starts = [m.start() for m in re.finditer(r"-+", dash_line)]
            data_lines = lines[header_idx + 1:]
            for line in data_lines:
                if not line.strip():
                    continue
                # extract fields by column positions
                fields = []
                for j, start in enumerate(col_starts):
                    end = col_starts[j + 1] if j + 1 < len(col_starts) else len(line)
                    if start < len(line):
                        fields.append(line[start:end].strip())
                    else:
                        fields.append("")
                if len(fields) >= 2 and fields[0]:
                    results.append({
                        "name": fields[0],
                        "id": fields[1],
                        "version": fields[2] if len(fields) > 2 else "",
                    })
    elif manager_name == "brew":
        for line in lines:
            line = line.strip()
            if line and not line.startswith("==>"):
                results.append({"id": line, "name": line, "version": ""})
    elif manager_name in ("apt", "dnf", "zypper"):
        for line in lines:
            # skip apt noise lines
            if line.startswith(("Sorting", "Full Text", "WARNING", "  ")):
                continue
            # apt: "pkg/repo version arch [status]"
            m = re.match(r"^([a-z0-9][a-z0-9.+\-]+)(?:/\S+)?\s+(\S+)", line)
            if m:
                results.append({"id": m.group(1), "name": m.group(1), "version": m.group(2)})
    elif manager_name == "pacman":
        # "repo/pkgname version"
        for line in lines:
            m = re.match(r"^(?:\S+/)?(\S+)\s+(\S+)", line)
            if m:
                results.append({"id": m.group(1), "name": m.group(1), "version": m.group(2)})
    elif manager_name in ("choco", "scoop"):
        for line in lines:
            parts = line.split()
            if len(parts) >= 1 and not line.startswith(("Chocolatey", "---", "Name")):
                results.append({
                    "id": parts[0],
                    "name": parts[0],
                    "version": parts[1] if len(parts) > 1 else "",
                })
    elif manager_name in ("pip", "uv"):
        for line in lines:
            parts = line.split()
            if len(parts) >= 1 and not line.startswith(("Package", "---")):
                results.append({
                    "id": parts[0],
                    "name": parts[0],
                    "version": parts[1] if len(parts) > 1 else "",
                })
    else:
        # generic fallback: one result per non-empty line
        for line in lines:
            line = line.strip()
            if line:
                results.append({"id": line, "name": line, "version": ""})

    return results


def _parse_list(raw: str, manager_name: str) -> list[dict[str, str]]:
    """Parse list output — reuse search parser, works for most."""
    return _parse_search(raw, manager_name)


# ── public API ──────────────────────────────────────────────────────────────

def search(
    pkg: str,
    manager: Optional[str] = None,
    interactive: bool = False,
) -> list[dict[str, str]] | Optional[str]:
    """Search for a package. Returns list[dict] or prints + returns user choice."""
    mdef = resolve_manager(manager)
    if not mdef.search_cmd:
        log.warning("%s has no search command", mdef.name)
        return []

    result = _run(_build_cmd(mdef.search_cmd, pkg, mdef))
    if not result.ok:
        log.warning("search failed (%d): %s", result.returncode, result.stderr.strip())
        return []

    entries = _parse_search(result.stdout, mdef.name)
    if not interactive:
        return entries

    # interactive mode: print + let user pick
    if not entries:
        print(f"No results for '{pkg}' via {mdef.name}")
        return None

    print(f"\n  [{mdef.name}] search results for '{pkg}':\n")
    for i, e in enumerate(entries):
        ver = f"  ({e['version']})" if e.get("version") else ""
        print(f"    {i + 1:>3}. {e['id']}{ver}")
    print(f"    {'':>3}  0. cancel\n")

    try:
        choice = int(input("  select #> ").strip())
    except (ValueError, EOFError, KeyboardInterrupt):
        print("  cancelled.")
        return None

    if choice < 1 or choice > len(entries):
        print("  cancelled.")
        return None

    return entries[choice - 1]["id"]


def install(
    pkg: str,
    manager: Optional[str] = None,
    interactive: bool = False,
) -> RunResult:
    """Install a package."""
    mdef = resolve_manager(manager)
    cmd = _build_cmd(mdef.install_cmd, pkg, mdef)
    result = _run(cmd, interactive=interactive)

    # sudo fallback: if failed with sudo, retry without + warn
    if not result.ok and cmd[0] == "sudo":
        log.warning("sudo install failed (rc=%d), retrying without sudo", result.returncode)
        result = _run(cmd[1:], interactive=interactive)
        if result.ok:
            log.warning("Installed without sudo — may be incomplete")

    return result


def uninstall(
    pkg: str,
    manager: Optional[str] = None,
    interactive: bool = False,
) -> RunResult:
    """Uninstall a package."""
    mdef = resolve_manager(manager)
    cmd = _build_cmd(mdef.uninstall_cmd, pkg, mdef)
    return _run(cmd, interactive=interactive)


def update(
    pkg: str,
    manager: Optional[str] = None,
    interactive: bool = False,
) -> RunResult:
    """Update/upgrade a package."""
    mdef = resolve_manager(manager)
    cmd = _build_cmd(mdef.update_cmd, pkg, mdef)
    return _run(cmd, interactive=interactive)


def list_installed(
    manager: Optional[str] = None,
    filter_str: Optional[str] = None,
) -> list[dict[str, str]]:
    """List installed packages, optionally filtered."""
    mdef = resolve_manager(manager)
    if not mdef.list_cmd:
        return []

    result = _run(list(mdef.list_cmd))
    if not result.ok:
        log.warning("list failed: %s", result.stderr.strip())
        return []

    entries = _parse_list(result.stdout, mdef.name)
    if filter_str:
        f = filter_str.lower()
        entries = [e for e in entries if f in e.get("id", "").lower()
                   or f in e.get("name", "").lower()]
    return entries


def is_installed(
    pkg: str,
    manager: Optional[str] = None,
) -> bool:
    """Check if a package, module, or program is installed.

    Strategy (stops at first hit):
      1. shutil.which(pkg) — finds any binary/script on PATH
      2. list_installed() with filter — checks the selected package manager
      3. Python importlib — checks if it's an importable Python module
    """
    # 1) binary on PATH?
    if shutil.which(pkg):
        return True

    # 2) known to the package manager?
    entries = list_installed(manager=manager, filter_str=pkg)
    if any(
        pkg.lower() == e.get("id", "").lower()
        or pkg.lower() == e.get("name", "").lower()
        or pkg.lower() in e.get("id", "").lower()
        for e in entries
    ):
        return True

    # 3) importable Python module?
    try:
        import importlib
        importlib.import_module(pkg.replace("-", "_"))
        return True
    except (ImportError, ModuleNotFoundError, ValueError):
        pass

    return False


# ── interactive workflow ────────────────────────────────────────────────────

def interactive_install(query: str, manager: Optional[str] = None) -> RunResult | None:
    """Full interactive flow: search → user picks → install with live output."""
    choice = search(query, manager=manager, interactive=True)
    if not choice:
        return None
    print(f"\n  installing '{choice}' via {resolve_manager(manager).name}...")
    return install(choice, manager=manager, interactive=True)


# ── CLI entry point ─────────────────────────────────────────────────────────

def _cli():
    import argparse
    p = argparse.ArgumentParser(prog="depmanager", description="Cross-platform package manager")
    p.add_argument("-m", "--manager", default=None, help="Force package manager")
    sub = p.add_subparsers(dest="cmd")

    s_search = sub.add_parser("search", help="Search for a package")
    s_search.add_argument("pkg")
    s_search.add_argument("-i", "--interactive", action="store_true")

    s_install = sub.add_parser("install", help="Install a package")
    s_install.add_argument("pkg")
    s_install.add_argument("-i", "--interactive", action="store_true")

    s_uninstall = sub.add_parser("uninstall", help="Uninstall a package")
    s_uninstall.add_argument("pkg")

    s_update = sub.add_parser("update", help="Update a package")
    s_update.add_argument("pkg")

    s_list = sub.add_parser("list", help="List installed packages")
    s_list.add_argument("filter", nargs="?", default=None)

    s_detect = sub.add_parser("detect", help="Show available managers")

    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if os.environ.get("DEPX_DEBUG") else logging.INFO)

    if args.cmd == "detect":
        available = detect_available()
        preferred = resolve_manager(args.manager)
        print(f"  available: {', '.join(available)}")
        print(f"  selected:  {preferred.name}")
        return

    if args.cmd == "search":
        if args.interactive:
            search(args.pkg, manager=args.manager, interactive=True)
        else:
            for e in search(args.pkg, manager=args.manager):
                ver = f"  ({e['version']})" if e.get("version") else ""
                print(f"  {e['id']}{ver}")
        return

    if args.cmd == "install":
        if args.interactive:
            interactive_install(args.pkg, manager=args.manager)
        else:
            r = install(args.pkg, manager=args.manager)
            sys.exit(0 if r.ok else r.returncode)
        return

    if args.cmd == "uninstall":
        r = uninstall(args.pkg, manager=args.manager)
        sys.exit(0 if r.ok else r.returncode)

    if args.cmd == "update":
        r = update(args.pkg, manager=args.manager)
        sys.exit(0 if r.ok else r.returncode)

    if args.cmd == "list":
        for e in list_installed(manager=args.manager, filter_str=args.filter):
            ver = f"  ({e['version']})" if e.get("version") else ""
            print(f"  {e['id']}{ver}")
        return

    p.print_help()


if __name__ == "__main__":
    _cli()
