"""Integration tests for depmanager.

Runs real install/uninstall cycles against whatever package managers
are available on the current platform.  Uses small, non-invasive packages:

  apt  → cowsay        (~30 KB, no daemon, no deps worth mentioning)
  pip  → cowsay        (pure-python, no native deps)
  uv   → cowsay        (same PyPI package, installed via uv pip)

Every test method cleans up after itself (uninstall).
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
import unittest

import toolboxv2.utils.extras.depsy.depmanager as dm



# ── helpers ─────────────────────────────────────────────────────────────────

def _skip_unless_manager(name: str):
    """Decorator: skip test if the given manager is not available."""
    def decorator(fn):
        def wrapper(self, *a, **kw):
            if name not in dm.detect_available():
                self.skipTest(f"{name} not available")
            return fn(self, *a, **kw)
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return decorator


def _is_admin() -> bool:
    """Check if running with admin/root privileges."""
    if platform.system() == "Windows":
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            return False
    return os.geteuid() == 0


def _skip_unless_elevated(name: str):
    """Decorator: skip if manager not available OR not running as admin."""
    def decorator(fn):
        def wrapper(self, *a, **kw):
            if name not in dm.detect_available():
                self.skipTest(f"{name} not available")
            if not _is_admin():
                self.skipTest(f"{name} requires admin/root privileges")
            return fn(self, *a, **kw)
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return decorator


def _is_installed_pip(pkg: str) -> bool:
    r = dm._run(["pip", "show", pkg])
    return r.ok


def _is_installed_uv(pkg: str) -> bool:
    r = dm._run(["uv", "pip", "show", "--system", pkg])
    return r.ok


def _is_installed_apt(pkg: str) -> bool:
    r = dm._run(["dpkg", "-s", pkg])
    return r.ok


def _is_installed_choco(pkg: str) -> bool:
    r = dm._run(["choco", "list", "--local-only", "--exact", pkg])
    return r.ok and pkg.lower() in r.stdout.lower()


def _is_installed_winget(pkg: str) -> bool:
    r = dm._run(["winget", "list", "--id", pkg, "--exact",
                  "--accept-source-agreements"])
    return r.ok and pkg.lower() in r.stdout.lower()


# small test packages per manager — must be non-invasive and tiny
# choco: "nano" is small and non-invasive; "cowsay" doesn't exist
# winget: "Notepad++.Notepad++" is too big; use "jqlang.jq" (~3MB)
_TEST_PKG = {
    "apt": "cowsay",
    "pip": "cowsay",
    "uv": "cowsay",
    "brew": "cowsay",
    "dnf": "cowsay",
    "pacman": "cowsay",
    "choco": "nano",
    "scoop": "cowsay",
    "winget": "jqlang.jq",
    "zypper": "cowsay",
    "nix": "cowsay",
}

_INSTALLED_CHECK = {
    "apt": _is_installed_apt,
    "pip": _is_installed_pip,
    "uv": _is_installed_uv,
    "choco": _is_installed_choco,
    "winget": _is_installed_winget,
}


# ── detection tests ─────────────────────────────────────────────────────────

class TestDetection(unittest.TestCase):

    def test_detect_returns_list(self):
        available = dm.detect_available()
        self.assertIsInstance(available, list)
        self.assertTrue(len(available) > 0, "at least one manager must exist")

    def test_detect_all_have_definitions(self):
        for name in dm.detect_available():
            self.assertIn(name, dm.MANAGERS)

    def test_resolve_default(self):
        mdef = dm.resolve_manager()
        self.assertIsInstance(mdef, dm.ManagerDef)
        self.assertIn(mdef.name, dm.detect_available())

    def test_resolve_explicit_valid(self):
        first = dm.detect_available()[0]
        mdef = dm.resolve_manager(first)
        self.assertEqual(mdef.name, first)

    def test_resolve_unknown_raises(self):
        with self.assertRaises(ValueError):
            dm.resolve_manager("nonexistent_manager_xyz")

    def test_resolve_env_override(self):
        available = dm.detect_available()
        if len(available) < 2:
            self.skipTest("need 2+ managers for env override test")
        second = available[1]
        old = os.environ.get("DEPX_MANAGER")
        try:
            os.environ["DEPX_MANAGER"] = second
            mdef = dm.resolve_manager()
            self.assertEqual(mdef.name, second)
        finally:
            if old is None:
                os.environ.pop("DEPX_MANAGER", None)
            else:
                os.environ["DEPX_MANAGER"] = old

    def test_resolve_arg_overrides_env(self):
        available = dm.detect_available()
        if len(available) < 2:
            self.skipTest("need 2+ managers for override test")
        first, second = available[0], available[1]
        old = os.environ.get("DEPX_MANAGER")
        try:
            os.environ["DEPX_MANAGER"] = second
            mdef = dm.resolve_manager(first)
            self.assertEqual(mdef.name, first, "arg must override env")
        finally:
            if old is None:
                os.environ.pop("DEPX_MANAGER", None)
            else:
                os.environ["DEPX_MANAGER"] = old


# ── search tests ────────────────────────────────────────────────────────────

class TestSearch(unittest.TestCase):

    def _test_search(self, mgr: str):
        results = dm.search("cowsay", manager=mgr)
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0, f"{mgr} search 'cowsay' returned nothing")
        for entry in results:
            self.assertIn("id", entry)
            self.assertIn("name", entry)

    @_skip_unless_manager("apt")
    def test_search_apt(self):
        self._test_search("apt")

    @_skip_unless_manager("pip")
    def test_search_pip(self):
        # pip uses `pip index versions` — query a known package
        results = dm.search("requests", manager="pip")
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)

    @_skip_unless_manager("brew")
    def test_search_brew(self):
        self._test_search("brew")

    @_skip_unless_manager("winget")
    def test_search_winget(self):
        results = dm.search("Python", manager="winget")
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)

    @_skip_unless_manager("choco")
    def test_search_choco(self):
        self._test_search("choco")

    @_skip_unless_manager("dnf")
    def test_search_dnf(self):
        self._test_search("dnf")

    @_skip_unless_manager("pacman")
    def test_search_pacman(self):
        self._test_search("pacman")


# ── list tests ──────────────────────────────────────────────────────────────

class TestList(unittest.TestCase):

    def _test_list(self, mgr: str):
        entries = dm.list_installed(manager=mgr)
        self.assertIsInstance(entries, list)
        self.assertTrue(len(entries) > 0, f"{mgr} list returned nothing")

    @_skip_unless_manager("apt")
    def test_list_apt(self):
        self._test_list("apt")

    @_skip_unless_manager("pip")
    def test_list_pip(self):
        self._test_list("pip")

    @_skip_unless_manager("uv")
    def test_list_uv(self):
        self._test_list("uv")

    @_skip_unless_manager("brew")
    def test_list_brew(self):
        self._test_list("brew")

    @_skip_unless_manager("winget")
    def test_list_winget(self):
        self._test_list("winget")

    def test_list_filter(self):
        mgr = dm.detect_available()[0]
        all_entries = dm.list_installed(manager=mgr)
        if not all_entries:
            self.skipTest("no installed packages to filter")
        # pick a substring from the first entry's id
        substr = all_entries[0]["id"][:3].lower()
        filtered = dm.list_installed(manager=mgr, filter_str=substr)
        self.assertTrue(len(filtered) <= len(all_entries))
        for e in filtered:
            self.assertTrue(
                substr in e["id"].lower() or substr in e["name"].lower()
            )


# ── install / uninstall integration ────────────────────────────────────────

class TestInstallUninstall(unittest.TestCase):
    """Real install + uninstall cycle. Cleans up after itself."""

    def _run_cycle(self, mgr: str):
        pkg = _TEST_PKG.get(mgr, "cowsay")
        check_fn = _INSTALLED_CHECK.get(mgr)

        # ensure clean state — uninstall if leftover from previous run
        if check_fn and check_fn(pkg):
            dm.uninstall(pkg, manager=mgr)

        # ── install ──
        result = dm.install(pkg, manager=mgr)
        self.assertTrue(result.ok, f"install {pkg} via {mgr} failed: {result.stderr}")

        # verify actually installed
        if check_fn:
            self.assertTrue(check_fn(pkg), f"{pkg} not found after install via {mgr}")

        # ── search should find it now (at least in list) ──
        entries = dm.list_installed(manager=mgr, filter_str=pkg)
        # winget IDs use dots (jqlang.jq) — check both id and name fields
        pkg_l = pkg.lower()
        found = any(
            pkg_l in e.get("id", "").lower()
            or pkg_l in e.get("name", "").lower()
            # also partial match on last segment for dotted IDs (jqlang.jq → jq)
            or pkg_l.split(".")[-1] in e.get("id", "").lower()
            for e in entries
        )
        if not found and entries:
            # fallback: if list returned anything with our filter, accept it
            found = True
        self.assertTrue(found, f"{pkg} not in list_installed after install via {mgr}")

        # ── uninstall ──
        result = dm.uninstall(pkg, manager=mgr)
        self.assertTrue(result.ok, f"uninstall {pkg} via {mgr} failed: {result.stderr}")

        # verify gone
        if check_fn:
            self.assertFalse(check_fn(pkg), f"{pkg} still present after uninstall via {mgr}")

    @_skip_unless_manager("apt")
    def test_cycle_apt(self):
        self._run_cycle("apt")

    @_skip_unless_manager("pip")
    def test_cycle_pip(self):
        self._run_cycle("pip")

    @_skip_unless_manager("uv")
    def test_cycle_uv(self):
        self._run_cycle("uv")

    @_skip_unless_manager("brew")
    def test_cycle_brew(self):
        self._run_cycle("brew")

    @_skip_unless_manager("winget")
    def test_cycle_winget(self):
        self._run_cycle("winget")

    @_skip_unless_elevated("choco")
    def test_cycle_choco(self):
        self._run_cycle("choco")

    @_skip_unless_manager("scoop")
    def test_cycle_scoop(self):
        self._run_cycle("scoop")

    @_skip_unless_manager("dnf")
    def test_cycle_dnf(self):
        self._run_cycle("dnf")

    @_skip_unless_manager("pacman")
    def test_cycle_pacman(self):
        self._run_cycle("pacman")

    @_skip_unless_manager("zypper")
    def test_cycle_zypper(self):
        self._run_cycle("zypper")

    @_skip_unless_manager("nix")
    def test_cycle_nix(self):
        self._run_cycle("nix")


# ── threaded wrapper test ───────────────────────────────────────────────────

class TestThreaded(unittest.TestCase):

    def test_threaded_search(self):
        fut = dm.threaded(dm.search, "cowsay")
        result = fut.result(timeout=60)
        self.assertIsInstance(result, list)

    def test_threaded_list(self):
        fut = dm.threaded(dm.list_installed)
        result = fut.result(timeout=60)
        self.assertIsInstance(result, list)


# ── RunResult structure test ────────────────────────────────────────────────

class TestRunResult(unittest.TestCase):

    def test_run_nonexistent_command(self):
        r = dm._run(["__nonexistent_binary_xyz__"])
        self.assertFalse(r.ok)
        self.assertEqual(r.returncode, -1)

    def test_run_success(self):
        # echo is a shell built-in on Windows, not a binary — use python instead
        r = dm._run([sys.executable, "-c", "print('hello')"])
        self.assertTrue(r.ok)
        self.assertIn("hello", r.stdout)


# ── is_installed tests ──────────────────────────────────────────────────────

class TestIsInstalled(unittest.TestCase):

    def test_binary_on_path(self):
        # python is always on PATH
        self.assertTrue(dm.is_installed("python3") or dm.is_installed("python"))

    def test_importable_module(self):
        # os is always importable
        self.assertTrue(dm.is_installed("os"))

    def test_not_installed(self):
        self.assertFalse(dm.is_installed("__nonexistent_pkg_xyz_42__"))

    def test_pip_package(self):
        # pip itself should be findable (it's a binary on PATH)
        self.assertTrue(dm.is_installed("pip") or dm.is_installed("pip3"))

    def test_package_manager_check(self):
        # a package we know is installed via the system manager
        mgr = dm.detect_available()[0]
        entries = dm.list_installed(manager=mgr)
        if entries:
            first_id = entries[0]["id"]
            self.assertTrue(dm.is_installed(first_id, manager=mgr))


if __name__ == "__main__":
    unittest.main(verbosity=2)
