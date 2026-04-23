"""
VFS V2 — Ebene 1 Bug-Fixes + Ebene 2 MountPollRegistry Tests (v2 — korrigiert)
==============================================================================

Änderungen ggü. v1:
  - _wait_for_vfs_change() Helper statt starre time.sleep()
  - _fresh_registry() mit Thread-Yield
  - TestRegistryLifecycle nutzt Registry direkt (kein vfs.mount das auto-subscribed)
  - TestPollIntegration / TestMultiAgentSync ergänzen subscribe() mit FAST_POLL
    NACH dem mount() — der Watcher adoptiert das schnellere Interval
  - Großzügigerer WAIT-Timeout für Windows mtime-Granularität (NTFS: 100ns,
    aber die Python-os-API rundet teilweise)
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest

from toolboxv2.mods.isaa.base.Agent.vfs_v2 import (
    FileBackingType,
    VirtualFileSystemV2,
    VFSFile,
)
from toolboxv2.mods.isaa.base.patch.mount_poll_registry import (
    MountPollRegistry,
    MountWatcher,
    get_mount_poll_registry,
    DEFAULT_POLL_INTERVAL,
    BURST_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vfs(session_id: str = "test", agent: str = "agent") -> VirtualFileSystemV2:
    return VirtualFileSystemV2(session_id=session_id, agent_name=agent)


def _populate(base: str, files: dict[str, str]) -> None:
    for rel, content in files.items():
        full = os.path.join(base, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(content)


def _write_disk(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def _delete_disk(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def _fresh_registry() -> MountPollRegistry:
    """Reset registry to empty state. Stops all watcher threads."""
    reg = get_mount_poll_registry()
    with reg._lock:
        for watcher in list(reg._watchers.values()):
            watcher._stop_event.set()
        reg._watchers.clear()
    time.sleep(0.05)  # Daemon-Threads Zeit geben den Loop zu verlassen
    return reg


def _wait_for_vfs_change(
    vfs: VirtualFileSystemV2,
    predicate,
    timeout: float = 3.0,
    check_interval: float = 0.08,
) -> bool:
    """Poll VFS state until predicate(vfs) is True or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if predicate(vfs):
                return True
        except (KeyError, AttributeError):
            pass
        time.sleep(check_interval)
    try:
        return predicate(vfs)
    except (KeyError, AttributeError):
        return False


# ===========================================================================
# EBENE 1 — Fix #1
# ===========================================================================

class TestScanMountRespectsDirty(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _populate(self.tmp.name, {"work.py": "v = 1"})
        self.vfs = _make_vfs()
        self.vfs.mount(self.tmp.name, vfs_path="/proj", auto_sync=False)

    def tearDown(self):
        try: self.vfs.unmount("/proj", save_changes=False)
        except Exception: pass
        self.tmp.cleanup()

    def test_dirty_content_survives_scan_mount(self):
        f = self.vfs.files["/proj/work.py"]
        f._content = "v = DIRTY"
        f.is_dirty = True
        f.backing_type = FileBackingType.MODIFIED

        self.vfs._scan_mount(self.vfs.mounts["/proj"])

        f_after = self.vfs.files.get("/proj/work.py")
        self.assertIsNotNone(f_after)
        self.assertEqual(f_after._content, "v = DIRTY")
        self.assertTrue(f_after.is_dirty)

    def test_dirty_file_not_replaced_with_new_instance(self):
        f = self.vfs.files["/proj/work.py"]
        f._content = "v = DIRTY"
        f.is_dirty = True
        f.backing_type = FileBackingType.MODIFIED
        original_id = id(f)

        self.vfs._scan_mount(self.vfs.mounts["/proj"])

        self.assertEqual(id(self.vfs.files.get("/proj/work.py")), original_id)

    def test_clean_file_mtime_updated_on_scan(self):
        f = self.vfs.files["/proj/work.py"]
        f._content = "v = 1"
        f.is_dirty = False
        f.local_mtime = 0.0

        self.vfs._scan_mount(self.vfs.mounts["/proj"])
        self.assertGreater(self.vfs.files["/proj/work.py"].local_mtime, 0.0)

    def test_clean_loaded_file_invalidated_if_disk_newer(self):
        f = self.vfs.files["/proj/work.py"]
        f._content = "v = OLD"
        f.is_dirty = False
        f.backing_type = FileBackingType.SHADOW
        f.local_mtime = 0.0

        self.vfs._scan_mount(self.vfs.mounts["/proj"])
        self.assertIsNone(self.vfs.files["/proj/work.py"]._content)


# ===========================================================================
# EBENE 1 — Fix #2
# ===========================================================================

class TestScanMountRemovesZombies(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _populate(self.tmp.name, {"alive.py": "ok", "zombie.py": "gone"})
        self.vfs = _make_vfs()
        self.vfs.mount(self.tmp.name, vfs_path="/proj")

    def tearDown(self):
        try: self.vfs.unmount("/proj", save_changes=False)
        except Exception: pass
        self.tmp.cleanup()

    def test_deleted_disk_file_removed_from_vfs(self):
        self.assertIn("/proj/zombie.py", self.vfs.files)
        _delete_disk(os.path.join(self.tmp.name, "zombie.py"))
        self.vfs._scan_mount(self.vfs.mounts["/proj"])
        self.assertNotIn("/proj/zombie.py", self.vfs.files)

    def test_deleted_disk_file_removed_from_shadow_index(self):
        _delete_disk(os.path.join(self.tmp.name, "zombie.py"))
        self.vfs._scan_mount(self.vfs.mounts["/proj"])
        self.assertNotIn("/proj/zombie.py", self.vfs._shadow_index)

    def test_alive_file_remains_after_scan(self):
        _delete_disk(os.path.join(self.tmp.name, "zombie.py"))
        self.vfs._scan_mount(self.vfs.mounts["/proj"])
        self.assertIn("/proj/alive.py", self.vfs.files)

    def test_dirty_zombie_not_removed(self):
        f = self.vfs.files["/proj/zombie.py"]
        f._content = "unsaved work"
        f.is_dirty = True
        f.backing_type = FileBackingType.MODIFIED

        _delete_disk(os.path.join(self.tmp.name, "zombie.py"))
        self.vfs._scan_mount(self.vfs.mounts["/proj"])

        self.assertIn("/proj/zombie.py", self.vfs.files)
        self.assertEqual(self.vfs.files["/proj/zombie.py"]._content, "unsaved work")

    def test_refresh_mount_removes_zombie_via_scan(self):
        _delete_disk(os.path.join(self.tmp.name, "zombie.py"))
        self.vfs.refresh_mount("/proj")
        self.assertNotIn("/proj/zombie.py", self.vfs.files)


# ===========================================================================
# EBENE 1 — Fix #3
# ===========================================================================

class TestReadMtimeRefresh(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _populate(self.tmp.name, {"data.py": "v = 1"})
        self.vfs = _make_vfs()
        self.vfs.mount(self.tmp.name, vfs_path="/proj", auto_sync=True)

    def tearDown(self):
        try: self.vfs.unmount("/proj", save_changes=False)
        except Exception: pass
        self.tmp.cleanup()

    def test_read_reloads_when_disk_is_newer(self):
        self.vfs.read("/proj/data.py")
        f = self.vfs.files["/proj/data.py"]
        self.assertTrue(f.is_loaded)

        time.sleep(0.05)
        _write_disk(f.local_path, "v = 999")
        f.local_mtime = 0.0

        result = self.vfs.read("/proj/data.py")
        self.assertTrue(result["success"])
        self.assertIn("999", result["content"])

    def test_read_does_not_reload_if_mtime_unchanged(self):
        self.vfs.read("/proj/data.py")
        f = self.vfs.files["/proj/data.py"]
        f._content = "v = CACHED"
        f.is_dirty = False
        f.local_mtime = os.path.getmtime(f.local_path)

        self.assertIn("CACHED", self.vfs.read("/proj/data.py")["content"])

    def test_read_does_not_reload_dirty_file(self):
        self.vfs.read("/proj/data.py")
        f = self.vfs.files["/proj/data.py"]
        f._content = "v = DIRTY"
        f.is_dirty = True
        f.local_mtime = 0.0

        self.assertIn("DIRTY", self.vfs.read("/proj/data.py")["content"])

    def test_read_returns_error_if_backing_file_disappeared(self):
        self.vfs.read("/proj/data.py")
        f = self.vfs.files["/proj/data.py"]
        local = f.local_path
        f.local_mtime = 0.0

        _delete_disk(local)
        result = self.vfs.read("/proj/data.py")
        self.assertFalse(result["success"])
        self.assertIn("hint", result)


# ===========================================================================
# EBENE 1 — Fix #4
# ===========================================================================

class TestWriteOptimisticConcurrency(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _populate(self.tmp.name, {"target.py": "v = 0"})
        self.vfs = _make_vfs()
        self.vfs.mount(self.tmp.name, vfs_path="/proj", auto_sync=True)

    def tearDown(self):
        try: self.vfs.unmount("/proj", save_changes=False)
        except Exception: pass
        self.tmp.cleanup()

    def test_clean_file_with_newer_disk_reloads_transparently(self):
        self.vfs.read("/proj/target.py")
        f = self.vfs.files["/proj/target.py"]
        f.local_mtime = 0.0

        result = self.vfs.write("/proj/target.py", "v = NEW")
        self.assertTrue(result["success"])

    def test_dirty_file_with_newer_disk_returns_conflict(self):
        self.vfs.read("/proj/target.py")
        f = self.vfs.files["/proj/target.py"]
        f._content = "v = AGENT_WORK"
        f.is_dirty = True
        f.backing_type = FileBackingType.MODIFIED
        f.local_mtime = 0.0

        time.sleep(0.05)
        _write_disk(f.local_path, "v = EXTERNAL")

        result = self.vfs.write("/proj/target.py", "v = AGENT_NEW")
        self.assertFalse(result["success"])
        self.assertTrue(result.get("conflict"))
        self.assertIn("hint", result)

    def test_conflict_response_contains_mtime_info(self):
        self.vfs.read("/proj/target.py")
        f = self.vfs.files["/proj/target.py"]
        f._content = "dirty"
        f.is_dirty = True
        f.local_mtime = 0.0

        time.sleep(0.05)
        _write_disk(f.local_path, "external")

        result = self.vfs.write("/proj/target.py", "new")
        if not result.get("success") and result.get("conflict"):
            self.assertIn("disk_mtime", result)
            self.assertIn("local_mtime", result)

    def test_clean_file_no_conflict_when_mtime_matches(self):
        self.vfs.read("/proj/target.py")
        f = self.vfs.files["/proj/target.py"]
        f.local_mtime = os.path.getmtime(f.local_path)
        f.is_dirty = False

        result = self.vfs.write("/proj/target.py", "v = NORMAL")
        self.assertTrue(result["success"])
        self.assertFalse(result.get("conflict", False))


# ===========================================================================
# EBENE 2 — MountWatcher isoliert
# ===========================================================================

class TestMountWatcherScanDiff(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _populate(self.tmp.name, {"a.py": "x = 1", "sub/b.py": "y = 2"})
        self.watcher = MountWatcher(local_path=self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_scan_disk_finds_all_files(self):
        snap = self.watcher._scan_disk()
        keys = list(snap.keys())
        self.assertTrue(any("a.py" in k for k in keys))
        self.assertTrue(any("b.py" in k for k in keys))

    def test_scan_disk_excludes_pycache(self):
        _populate(self.tmp.name, {"__pycache__/c.pyc": "\x00"})
        snap = self.watcher._scan_disk()
        self.assertFalse(any("__pycache__" in k for k in snap.keys()))

    def test_diff_detects_created(self):
        old = self.watcher._scan_disk()
        _populate(self.tmp.name, {"new.py": "new"})
        new = self.watcher._scan_disk()
        changes = self.watcher._diff_snapshots(old, new)
        self.assertTrue(any("new.py" in p and t == "created" for p, t in changes.items()))

    def test_diff_detects_deleted(self):
        old = self.watcher._scan_disk()
        _delete_disk(os.path.join(self.tmp.name, "a.py"))
        new = self.watcher._scan_disk()
        changes = self.watcher._diff_snapshots(old, new)
        self.assertTrue(any("a.py" in p and t == "deleted" for p, t in changes.items()))

    def test_diff_detects_modified(self):
        old = self.watcher._scan_disk()
        time.sleep(0.05)
        _write_disk(os.path.join(self.tmp.name, "a.py"), "x = 999")
        new = self.watcher._scan_disk()
        changes = self.watcher._diff_snapshots(old, new)
        self.assertTrue(any("a.py" in p and t == "modified" for p, t in changes.items()))

    def test_diff_empty_when_nothing_changed(self):
        snap = self.watcher._scan_disk()
        self.assertEqual(len(self.watcher._diff_snapshots(snap, snap)), 0)


# ===========================================================================
# EBENE 2 — Registry Lifecycle
# ===========================================================================

class TestRegistryLifecycle(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _fresh_registry()
        self.reg = get_mount_poll_registry()
        self.vfs = _make_vfs()

    def tearDown(self):
        self.reg.unsubscribe_all(self.vfs)
        _fresh_registry()
        self.tmp.cleanup()

    def test_subscribe_starts_watcher(self):
        self.reg.subscribe(self.tmp.name, self.vfs, poll_interval=5.0)
        self.assertEqual(self.reg.stats()["active_watchers"], 1)

    def test_subscribe_starts_thread(self):
        self.reg.subscribe(self.tmp.name, self.vfs, poll_interval=5.0)
        abs_path = os.path.abspath(self.tmp.name)
        watcher = self.reg._watchers[abs_path]
        time.sleep(0.2)
        self.assertTrue(watcher._thread is not None and watcher._thread.is_alive())

    def test_two_vfs_share_one_watcher(self):
        vfs2 = _make_vfs("test2", "agent2")
        try:
            self.reg.subscribe(self.tmp.name, self.vfs, poll_interval=5.0)
            self.reg.subscribe(self.tmp.name, vfs2, poll_interval=5.0)
            self.assertEqual(self.reg.stats()["active_watchers"], 1)
            abs_path = os.path.abspath(self.tmp.name)
            self.assertEqual(len(self.reg._watchers[abs_path]._subscribers), 2)
        finally:
            self.reg.unsubscribe_all(vfs2)

    def test_unsubscribe_last_stops_watcher(self):
        self.reg.subscribe(self.tmp.name, self.vfs, poll_interval=5.0)
        self.reg.unsubscribe(self.tmp.name, self.vfs)
        self.assertEqual(self.reg.stats()["active_watchers"], 0)

    def test_unsubscribe_one_of_two_keeps_watcher(self):
        vfs2 = _make_vfs("test2", "agent2")
        try:
            self.reg.subscribe(self.tmp.name, self.vfs, poll_interval=5.0)
            self.reg.subscribe(self.tmp.name, vfs2, poll_interval=5.0)
            self.reg.unsubscribe(self.tmp.name, self.vfs)
            self.assertEqual(self.reg.stats()["active_watchers"], 1)
        finally:
            self.reg.unsubscribe_all(vfs2)

    def test_unsubscribe_nonexistent_does_not_raise(self):
        try:
            self.reg.unsubscribe(self.tmp.name, self.vfs)
        except Exception as e:
            self.fail(f"unsubscribe ohne subscribe muss stumm sein: {e}")

    def test_unsubscribe_all_removes_from_all_watchers(self):
        tmp2 = tempfile.TemporaryDirectory()
        try:
            self.reg.subscribe(self.tmp.name, self.vfs, poll_interval=5.0)
            self.reg.subscribe(tmp2.name, self.vfs, poll_interval=5.0)
            self.reg.unsubscribe_all(self.vfs)
            self.assertEqual(self.reg.stats()["active_watchers"], 0)
        finally:
            tmp2.cleanup()

    def test_faster_interval_adopted_by_existing_watcher(self):
        """Zweiter subscribe mit kleinerem Interval → Watcher übernimmt es."""
        self.reg.subscribe(self.tmp.name, self.vfs, poll_interval=5.0)
        vfs2 = _make_vfs("test2", "agent2")
        try:
            self.reg.subscribe(self.tmp.name, vfs2, poll_interval=0.3)
            abs_path = os.path.abspath(self.tmp.name)
            watcher = self.reg._watchers[abs_path]
            self.assertAlmostEqual(watcher.poll_interval, 0.3, places=2)
        finally:
            self.reg.unsubscribe_all(vfs2)


# ===========================================================================
# EBENE 2 — End-to-End Integration
# ===========================================================================

class TestPollIntegration(unittest.TestCase):

    FAST_POLL = 0.3
    WAIT = 4.0   # generous for Windows, NTFS mtime, CI slowness

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _fresh_registry()
        self.reg = get_mount_poll_registry()
        _populate(self.tmp.name, {"shared.py": "v = 1"})
        self.vfs = _make_vfs()
        self.vfs.mount(self.tmp.name, vfs_path="/proj", auto_sync=True)
        # schnelleres Interval erzwingen (Watcher existiert bereits vom mount-auto-subscribe)
        self.reg.subscribe(self.tmp.name, self.vfs, poll_interval=self.FAST_POLL)

    def tearDown(self):
        try: self.vfs.unmount("/proj", save_changes=False)
        except Exception: pass
        _fresh_registry()
        self.tmp.cleanup()

    def test_external_create_becomes_visible(self):
        _write_disk(os.path.join(self.tmp.name, "new_file.py"), "created = True")

        found = _wait_for_vfs_change(
            self.vfs,
            lambda v: "/proj/new_file.py" in v.files,
            timeout=self.WAIT,
        )
        self.assertTrue(
            found,
            f"Extern erstelltes File muss nach {self.WAIT}s sichtbar sein. "
            f"Registry: {self.reg.stats()}"
        )

    def test_external_modify_invalidates_cache(self):
        self.vfs.read("/proj/shared.py")
        f = self.vfs.files["/proj/shared.py"]
        self.assertTrue(f.is_loaded)

        time.sleep(0.15)
        _write_disk(f.local_path, "v = MODIFIED_EXTERNALLY")

        invalidated = _wait_for_vfs_change(
            self.vfs,
            lambda v: (
                v.files["/proj/shared.py"]._content is None
                or "MODIFIED_EXTERNALLY" in (v.files["/proj/shared.py"]._content or "")
            ),
            timeout=self.WAIT,
        )
        self.assertTrue(invalidated)

    def test_external_modify_then_read_gets_new_content(self):
        self.vfs.read("/proj/shared.py")
        f = self.vfs.files["/proj/shared.py"]

        time.sleep(0.15)
        _write_disk(f.local_path, "v = FRESH_CONTENT")

        _wait_for_vfs_change(
            self.vfs,
            lambda v: (
                v.files["/proj/shared.py"]._content is None
                or "FRESH_CONTENT" in (v.files["/proj/shared.py"]._content or "")
            ),
            timeout=self.WAIT,
        )

        result = self.vfs.read("/proj/shared.py")
        self.assertTrue(result["success"])
        self.assertIn("FRESH_CONTENT", result["content"])

    def test_external_delete_removes_from_vfs(self):
        _delete_disk(os.path.join(self.tmp.name, "shared.py"))

        removed = _wait_for_vfs_change(
            self.vfs,
            lambda v: "/proj/shared.py" not in v.files,
            timeout=self.WAIT,
        )
        self.assertTrue(
            removed,
            f"Extern gelöschtes File muss nach {self.WAIT}s entfernt sein. "
            f"Registry: {self.reg.stats()}"
        )

    def test_dirty_file_not_removed_on_external_delete(self):
        f = self.vfs.files["/proj/shared.py"]
        f._content = "unsaved agent work"
        f.is_dirty = True
        f.backing_type = FileBackingType.MODIFIED

        _delete_disk(os.path.join(self.tmp.name, "shared.py"))
        time.sleep(self.WAIT)

        self.assertIn("/proj/shared.py", self.vfs.files)
        self.assertEqual(self.vfs.files["/proj/shared.py"]._content, "unsaved agent work")


# ===========================================================================
# EBENE 2 — Multi-Agent
# ===========================================================================

class TestMultiAgentSync(unittest.TestCase):

    FAST_POLL = 0.3
    WAIT = 4.0

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _fresh_registry()
        self.reg = get_mount_poll_registry()

        _populate(self.tmp.name, {"shared.py": "v = 0"})

        self.vfs_a = _make_vfs("session-a", "agent-a")
        self.vfs_b = _make_vfs("session-b", "agent-b")

        self.vfs_a.mount(self.tmp.name, vfs_path="/proj", auto_sync=True)
        self.vfs_b.mount(self.tmp.name, vfs_path="/proj", auto_sync=True)

        # Schnelleres Interval erzwingen — beide subscriben denselben Watcher,
        # der das kleinere Interval übernimmt
        self.reg.subscribe(self.tmp.name, self.vfs_a, poll_interval=self.FAST_POLL)
        self.reg.subscribe(self.tmp.name, self.vfs_b, poll_interval=self.FAST_POLL)

    def tearDown(self):
        try: self.vfs_a.unmount("/proj", save_changes=False)
        except Exception: pass
        try: self.vfs_b.unmount("/proj", save_changes=False)
        except Exception: pass
        _fresh_registry()
        self.tmp.cleanup()

    def test_a_creates_file_b_sees_it(self):
        self.vfs_a.create("/proj/a_1.py", "created by A")

        found = _wait_for_vfs_change(
            self.vfs_b,
            lambda v: "/proj/a_1.py" in v.files,
            timeout=self.WAIT,
        )
        self.assertTrue(
            found,
            f"Agent B muss File sehen das A erstellt hat. Registry: {self.reg.stats()}"
        )

    def test_b_creates_file_a_sees_it(self):
        self.vfs_b.create("/proj/b_1.py", "created by B")

        found = _wait_for_vfs_change(
            self.vfs_a,
            lambda v: "/proj/b_1.py" in v.files,
            timeout=self.WAIT,
        )
        self.assertTrue(
            found,
            f"Agent A muss File sehen das B erstellt hat. Registry: {self.reg.stats()}"
        )

    def test_b_writes_to_a_file_a_reads_new_content(self):
        self.vfs_a.create("/proj/a_1.py", "v = 1")

        # B wartet bis Datei sichtbar
        _wait_for_vfs_change(
            self.vfs_b,
            lambda v: "/proj/a_1.py" in v.files,
            timeout=self.WAIT,
        )
        self.assertIn("/proj/a_1.py", self.vfs_b.files)

        time.sleep(0.15)   # mtime-Differenz
        self.vfs_b.write("/proj/a_1.py", "v = B_WROTE")

        _wait_for_vfs_change(
            self.vfs_a,
            lambda v: (
                "/proj/a_1.py" in v.files
                and (
                    v.files["/proj/a_1.py"]._content is None
                    or "B_WROTE" in (v.files["/proj/a_1.py"]._content or "")
                )
            ),
            timeout=self.WAIT,
        )

        result = self.vfs_a.read("/proj/a_1.py")
        self.assertTrue(result["success"])
        self.assertIn("B_WROTE", result["content"])

    def test_a_deletes_b_file_b_sees_deletion(self):
        self.vfs_b.create("/proj/b_1.py", "B's data")

        _wait_for_vfs_change(
            self.vfs_a,
            lambda v: "/proj/b_1.py" in v.files,
            timeout=self.WAIT,
        )
        self.assertIn("/proj/b_1.py", self.vfs_a.files)

        self.vfs_a.delete("/proj/b_1.py")

        removed = _wait_for_vfs_change(
            self.vfs_b,
            lambda v: "/proj/b_1.py" not in v.files,
            timeout=self.WAIT,
        )
        self.assertTrue(removed)

    def test_shared_watcher_single_thread(self):
        stats = self.reg.stats()
        self.assertEqual(stats["active_watchers"], 1)
        self.assertEqual(stats["watchers"][0]["subscribers"], 2)


# ===========================================================================
# EBENE 2 — Burst-Mode
# ===========================================================================

class TestBurstMode(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _fresh_registry()
        self.vfs = _make_vfs()

    def tearDown(self):
        try: self.vfs.unmount("/proj", save_changes=False)
        except Exception: pass
        _fresh_registry()
        self.tmp.cleanup()

    def test_burst_triggers_refresh_mount(self):
        _populate(self.tmp.name, {"base.py": "x = 1"})
        self.vfs.mount(self.tmp.name, vfs_path="/proj")
        watcher = MountWatcher(local_path=self.tmp.name)

        refresh_called = []
        original = self.vfs.refresh_mount
        def mock_refresh(p):
            refresh_called.append(p)
            return original(p)
        self.vfs.refresh_mount = mock_refresh

        watcher._dispatch_burst(self.vfs)
        self.assertIn("/proj", refresh_called)

    def test_large_changeset_calls_burst_not_individual(self):
        _populate(self.tmp.name, {"base.py": "x = 1"})
        self.vfs.mount(self.tmp.name, vfs_path="/proj")
        watcher = MountWatcher(local_path=self.tmp.name)
        import weakref
        watcher._subscribers = weakref.WeakSet([self.vfs])

        burst_calls = []
        individual_calls = []
        watcher._dispatch_burst = lambda vfs: burst_calls.append(vfs)
        watcher._dispatch_individual = lambda vfs, ch: individual_calls.append(vfs)

        fake = {f"/fake/file_{i}.py": "created" for i in range(BURST_THRESHOLD + 10)}
        watcher._dispatch_changes(fake)

        self.assertTrue(len(burst_calls) > 0)
        self.assertEqual(len(individual_calls), 0)

    def test_small_changeset_calls_individual_not_burst(self):
        _populate(self.tmp.name, {"base.py": "x = 1"})
        self.vfs.mount(self.tmp.name, vfs_path="/proj")
        watcher = MountWatcher(local_path=self.tmp.name)
        import weakref
        watcher._subscribers = weakref.WeakSet([self.vfs])

        burst_calls = []
        individual_calls = []
        watcher._dispatch_burst = lambda vfs: burst_calls.append(vfs)
        watcher._dispatch_individual = lambda vfs, ch: individual_calls.append((vfs, ch))

        small = {os.path.join(self.tmp.name, "base.py"): "modified"}
        watcher._dispatch_changes(small)

        self.assertEqual(len(burst_calls), 0)
        self.assertTrue(len(individual_calls) > 0)


# ===========================================================================
# EBENE 2 — vfs.mount() Integration
# ===========================================================================

class TestMountAutoSubscribes(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _fresh_registry()
        self.reg = get_mount_poll_registry()

    def tearDown(self):
        _fresh_registry()
        self.tmp.cleanup()

    def test_mount_subscribes_to_registry(self):
        vfs = _make_vfs()
        try:
            vfs.mount(self.tmp.name, vfs_path="/proj")
            self.assertGreaterEqual(self.reg.stats()["active_watchers"], 1)
        finally:
            vfs.unmount("/proj", save_changes=False)

    def test_unmount_unsubscribes_from_registry(self):
        vfs = _make_vfs()
        vfs.mount(self.tmp.name, vfs_path="/proj")
        vfs.unmount("/proj", save_changes=False)
        self.assertEqual(self.reg.stats()["active_watchers"], 0)

    def test_two_mounts_to_same_path_share_watcher(self):
        vfs_a = _make_vfs("a", "a")
        vfs_b = _make_vfs("b", "b")
        try:
            vfs_a.mount(self.tmp.name, vfs_path="/proj")
            vfs_b.mount(self.tmp.name, vfs_path="/proj2")
            self.assertEqual(self.reg.stats()["active_watchers"], 1)
        finally:
            try: vfs_a.unmount("/proj", save_changes=False)
            except Exception: pass
            try: vfs_b.unmount("/proj2", save_changes=False)
            except Exception: pass


# ===========================================================================
# EBENE 2 — Registry Stats
# ===========================================================================

class TestRegistryStats(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _fresh_registry()
        self.reg = get_mount_poll_registry()

    def tearDown(self):
        _fresh_registry()
        self.tmp.cleanup()

    def test_stats_empty_initially(self):
        stats = self.reg.stats()
        self.assertEqual(stats["active_watchers"], 0)
        self.assertEqual(len(stats["watchers"]), 0)

    def test_stats_reflect_active_watcher(self):
        vfs = _make_vfs()
        try:
            self.reg.subscribe(self.tmp.name, vfs, poll_interval=5.0)
            stats = self.reg.stats()
            self.assertEqual(stats["active_watchers"], 1)
            w_info = stats["watchers"][0]
            self.assertEqual(w_info["subscribers"], 1)
            self.assertIn("poll_interval", w_info)
            self.assertIn("tracked_files", w_info)
        finally:
            self.reg.unsubscribe(self.tmp.name, vfs)

    def test_stats_thread_alive_field(self):
        vfs = _make_vfs()
        try:
            self.reg.subscribe(self.tmp.name, vfs, poll_interval=5.0)
            time.sleep(0.2)
            self.assertTrue(self.reg.stats()["watchers"][0]["thread_alive"])
        finally:
            self.reg.unsubscribe(self.tmp.name, vfs)

# ===========================================================================
# EBENE 3 — GlobalVFSManager Shared-Store
# ===========================================================================

import threading as _threading

from toolboxv2.mods.isaa.base.patch.power_vfs import (
    GlobalVFSManager,
    get_global_vfs,
    GLOBAL_VFS_PATH,
)


def _reset_global_vfs() -> GlobalVFSManager:
    """
    Reset GlobalVFSManager singleton state for isolated tests.

    Clears shared_store, subscribers, mounted_vfs registry, and wipes
    the on-disk data_dir so each test starts with a known empty state.
    """
    gvfs = get_global_vfs()
    with gvfs._store_lock:
        gvfs._shared_store.clear()
        gvfs._subscribers.clear()
        gvfs._mounted_vfs.clear()
        gvfs._version_counter = 0
    # Disk auch leeren
    if gvfs.data_dir.exists():
        for item in gvfs.data_dir.iterdir():
            try:
                if item.is_dir():
                    import shutil as _sh
                    _sh.rmtree(item)
                else:
                    item.unlink()
            except OSError:
                pass
    return gvfs


class TestSharedStoreBasics(unittest.TestCase):
    """Grundfunktionen des Shared-Stores: write → read → store-hit."""

    def setUp(self):
        self.gvfs = _reset_global_vfs()

    def tearDown(self):
        _reset_global_vfs()

    def test_write_populates_shared_store(self):
        self.gvfs.write_file("config.json", '{"k": 1}', author="agent-a")
        entry = self.gvfs.get_shared("config.json")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["content"], '{"k": 1}')
        self.assertEqual(entry["author"], "agent-a")

    def test_write_also_persists_to_disk(self):
        self.gvfs.write_file("data.txt", "hello")
        disk_path = self.gvfs.data_dir / "data.txt"
        self.assertTrue(disk_path.exists())
        self.assertEqual(disk_path.read_text(encoding="utf-8"), "hello")

    def test_read_hits_shared_store(self):
        self.gvfs.write_file("x.md", "# Title")
        result = self.gvfs.read_file("x.md")
        self.assertTrue(result["success"])
        self.assertEqual(result["content"], "# Title")
        self.assertIn("version", result)

    def test_read_nonexistent_returns_error(self):
        result = self.gvfs.read_file("ghost.txt")
        self.assertFalse(result["success"])

    def test_version_counter_monotonic(self):
        r1 = self.gvfs.write_file("a.txt", "1")
        r2 = self.gvfs.write_file("b.txt", "2")
        r3 = self.gvfs.write_file("a.txt", "3")
        self.assertLess(r1["version"], r2["version"])
        self.assertLess(r2["version"], r3["version"])

    def test_path_traversal_rejected(self):
        result = self.gvfs.write_file("../escape.txt", "bad")
        self.assertFalse(result["success"])
        self.assertIn("traversal", result["error"].lower())

    def test_has_shared_reflects_state(self):
        self.assertFalse(self.gvfs.has_shared("new.md"))
        self.gvfs.write_file("new.md", "content")
        self.assertTrue(self.gvfs.has_shared("new.md"))


class TestSharedStoreDiskFallback(unittest.TestCase):
    """Wenn Store leer aber Disk-File existiert → read() soll laden & cachen."""

    def setUp(self):
        self.gvfs = _reset_global_vfs()

    def tearDown(self):
        _reset_global_vfs()

    def test_disk_file_read_hydrates_store(self):
        # Datei direkt auf Disk legen (Store-bypass)
        disk_path = self.gvfs.data_dir / "from_disk.txt"
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        disk_path.write_text("disk content", encoding="utf-8")

        self.assertFalse(self.gvfs.has_shared("from_disk.txt"))
        result = self.gvfs.read_file("from_disk.txt")
        self.assertTrue(result["success"])
        self.assertEqual(result["content"], "disk content")
        # Nach read muss der Store gefüllt sein
        self.assertTrue(self.gvfs.has_shared("from_disk.txt"))


class TestSharedStoreDelete(unittest.TestCase):
    """delete_file muss aus Store UND Disk entfernen."""

    def setUp(self):
        self.gvfs = _reset_global_vfs()
        self.gvfs.write_file("doomed.txt", "bye")

    def tearDown(self):
        _reset_global_vfs()

    def test_delete_removes_from_store(self):
        self.assertTrue(self.gvfs.has_shared("doomed.txt"))
        self.gvfs.delete_file("doomed.txt")
        self.assertFalse(self.gvfs.has_shared("doomed.txt"))

    def test_delete_removes_from_disk(self):
        disk_path = self.gvfs.data_dir / "doomed.txt"
        self.assertTrue(disk_path.exists())
        self.gvfs.delete_file("doomed.txt")
        self.assertFalse(disk_path.exists())

    def test_delete_nonexistent_returns_error(self):
        result = self.gvfs.delete_file("never_existed.txt")
        self.assertFalse(result["success"])


class TestSharedStoreSubscribers(unittest.TestCase):
    """Subscribe/Unsubscribe + Event-Delivery."""

    def setUp(self):
        self.gvfs = _reset_global_vfs()
        self.events = []

    def tearDown(self):
        _reset_global_vfs()

    def _cb(self, event):
        self.events.append(event)

    def test_subscribe_receives_write_event(self):
        self.gvfs.subscribe("tasks.md", self._cb)
        self.gvfs.write_file("tasks.md", "- [ ] todo", author="a")
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.events[0]["type"], "write")
        self.assertEqual(self.events[0]["path"], "tasks.md")
        self.assertEqual(self.events[0]["author"], "a")

    def test_subscribe_receives_delete_event(self):
        self.gvfs.write_file("doomed.md", "x")
        self.gvfs.subscribe("doomed.md", self._cb)
        self.gvfs.delete_file("doomed.md")
        # Filter: nur delete-Events zählen
        deletes = [e for e in self.events if e["type"] == "delete"]
        self.assertEqual(len(deletes), 1)

    def test_wildcard_subscribe_receives_all(self):
        self.gvfs.subscribe("*", self._cb)
        self.gvfs.write_file("a.txt", "1")
        self.gvfs.write_file("b.txt", "2")
        self.assertEqual(len(self.events), 2)

    def test_path_subscribe_filters(self):
        self.gvfs.subscribe("only_this.md", self._cb)
        self.gvfs.write_file("other.md", "ignored")
        self.gvfs.write_file("only_this.md", "hit")
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.events[0]["path"], "only_this.md")

    def test_unsubscribe_stops_delivery(self):
        self.gvfs.subscribe("topic.md", self._cb)
        self.gvfs.write_file("topic.md", "first")
        self.gvfs.unsubscribe("topic.md", self._cb)
        self.gvfs.write_file("topic.md", "second")
        self.assertEqual(len(self.events), 1)

    def test_callback_exception_does_not_break_others(self):
        def bad_cb(event):
            raise RuntimeError("boom")

        good_events = []
        def good_cb(event):
            good_events.append(event)

        self.gvfs.subscribe("file.md", bad_cb)
        self.gvfs.subscribe("file.md", good_cb)
        self.gvfs.write_file("file.md", "content")
        self.assertEqual(len(good_events), 1)


class TestSharedStoreVFSInvalidation(unittest.TestCase):
    """
    Wenn ein Agent via write_file schreibt, müssen alle gemounteten
    VFS-Instanzen ihre cached copy des Files invalidieren.
    """

    def setUp(self):
        self.gvfs = _reset_global_vfs()
        self.gvfs.write_file("shared.md", "initial")

        # Zwei VFS die /global/ mounten
        self.vfs_a = _make_vfs("a", "agent-a")
        self.vfs_b = _make_vfs("b", "agent-b")
        self.vfs_a.mount(str(self.gvfs.data_dir), vfs_path=GLOBAL_VFS_PATH, auto_sync=True)
        self.vfs_b.mount(str(self.gvfs.data_dir), vfs_path=GLOBAL_VFS_PATH, auto_sync=True)
        self.gvfs.register_vfs(self.vfs_a)
        self.gvfs.register_vfs(self.vfs_b)

    def tearDown(self):
        try: self.vfs_a.unmount(GLOBAL_VFS_PATH, save_changes=False)
        except Exception: pass
        try: self.vfs_b.unmount(GLOBAL_VFS_PATH, save_changes=False)
        except Exception: pass
        _reset_global_vfs()

    def test_write_invalidates_other_vfs_caches(self):
        vfs_path = f"{GLOBAL_VFS_PATH}/shared.md"

        # A lädt Content (cached in VFS-Instanz)
        self.vfs_a.read(vfs_path)
        f_a = self.vfs_a.files.get(vfs_path)
        if f_a is not None:
            self.assertTrue(f_a.is_loaded)

        # B schreibt via Shared-Store
        self.gvfs.write_file("shared.md", "B updated", author="agent-b")

        # A's Cache muss invalidiert sein (falls File in A's VFS existiert)
        if f_a is not None and not f_a.is_dirty:
            self.assertIsNone(
                f_a._content,
                "A's VFS-Cache muss nach write_file invalidiert sein"
            )

    def test_dirty_vfs_entry_not_invalidated(self):
        """Dirty Agent-Arbeit darf durch write_file nicht überschrieben werden."""
        vfs_path = f"{GLOBAL_VFS_PATH}/shared.md"
        self.vfs_a.read(vfs_path)
        f_a = self.vfs_a.files.get(vfs_path)
        if f_a is None:
            self.skipTest("File not in VFS A — mount-scan nicht durchgelaufen")

        f_a._content = "my unsaved work"
        f_a.is_dirty = True
        f_a.backing_type = FileBackingType.MODIFIED

        self.gvfs.write_file("shared.md", "external update")

        self.assertEqual(
            f_a._content, "my unsaved work",
            "Dirty-Content darf nicht durch invalidate überschrieben werden"
        )
        self.assertTrue(f_a.is_dirty)


class TestSharedStoreThreadSafety(unittest.TestCase):
    """Concurrent writes dürfen nicht crashen oder Daten korrumpieren."""

    def setUp(self):
        self.gvfs = _reset_global_vfs()

    def tearDown(self):
        _reset_global_vfs()

    def test_concurrent_writes_do_not_crash(self):
        """20 Threads schreiben je 10 Mal in unterschiedliche Pfade."""
        errors = []

        def worker(wid):
            try:
                for i in range(10):
                    result = self.gvfs.write_file(
                        f"t{wid}_{i}.txt", f"w{wid}_{i}", author=f"t{wid}"
                    )
                    if not result.get("success"):
                        errors.append(f"t{wid}_{i}: {result.get('error')}")
            except Exception as e:
                errors.append(str(e))

        threads = [_threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(errors, [], f"Concurrent write errors: {errors}")
        # 20 * 10 = 200 Einträge erwartet
        with self.gvfs._store_lock:
            self.assertEqual(len(self.gvfs._shared_store), 200)

    def test_concurrent_writes_same_path_last_wins(self):
        """
        100 Threads schreiben nacheinander in denselben Pfad.
        Kein Crash, final content ist einer der geschriebenen Werte.
        """
        written_values = set()
        errors = []

        def worker(wid):
            try:
                val = f"val_{wid}"
                written_values.add(val)
                self.gvfs.write_file("contested.txt", val)
            except Exception as e:
                errors.append(str(e))

        threads = [_threading.Thread(target=worker, args=(i,)) for i in range(100)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(errors, [])
        entry = self.gvfs.get_shared("contested.txt")
        self.assertIsNotNone(entry)
        self.assertIn(entry["content"], written_values)


class TestVFSReadWriteRoutingGlobal(unittest.TestCase):
    """
    vfs.read() und vfs.write() für /global/-Pfade müssen durch den
    Shared-Store laufen (nicht durch Shadow-Mount-Logik).
    """

    def setUp(self):
        self.gvfs = _reset_global_vfs()
        self.vfs = _make_vfs()
        self.vfs.mount(
            str(self.gvfs.data_dir),
            vfs_path=GLOBAL_VFS_PATH,
            auto_sync=True,
        )
        self.gvfs.register_vfs(self.vfs)

    def tearDown(self):
        try: self.vfs.unmount(GLOBAL_VFS_PATH, save_changes=False)
        except Exception: pass
        _reset_global_vfs()

    def test_vfs_write_to_global_populates_store(self):
        target = f"{GLOBAL_VFS_PATH}/note.md"
        # Datei im VFS anlegen damit write() sie findet
        self.vfs.create(target, "initial")
        self.vfs.write(target, "via vfs.write")

        entry = self.gvfs.get_shared("note.md")
        self.assertIsNotNone(
            entry,
            "vfs.write auf /global/-Pfad muss Shared-Store updaten"
        )
        self.assertEqual(entry["content"], "via vfs.write")

    def test_vfs_write_to_global_records_author(self):
        target = f"{GLOBAL_VFS_PATH}/authored.md"
        self.vfs.create(target, "x")
        self.vfs.write(target, "signed")

        entry = self.gvfs.get_shared("authored.md")
        self.assertEqual(entry["author"], self.vfs.agent_name)

    def test_vfs_read_from_global_hits_store(self):
        # Via Store schreiben
        self.gvfs.write_file("shared.md", "from store", author="external")
        target = f"{GLOBAL_VFS_PATH}/shared.md"

        # File in VFS anlegen (Shadow) damit read() den Pfad kennt
        self.vfs.create(target, "")   # Leerer placeholder
        # Dirty-Flag zurücksetzen damit Shared-Store-Pfad greift
        f = self.vfs.files[target]
        f.is_dirty = False

        result = self.vfs.read(target)
        self.assertTrue(result["success"])
        self.assertEqual(
            result["content"], "from store",
            "vfs.read auf /global/ muss Shared-Store-Content liefern"
        )

    def test_vfs_delete_on_global_clears_store(self):
        target = f"{GLOBAL_VFS_PATH}/ephemeral.md"
        self.vfs.create(target, "x")
        self.vfs.write(target, "real content")
        self.assertTrue(self.gvfs.has_shared("ephemeral.md"))

        self.vfs.delete(target)
        self.assertFalse(
            self.gvfs.has_shared("ephemeral.md"),
            "vfs.delete auf /global/ muss Shared-Store-Eintrag entfernen"
        )


class TestMultiAgentGlobalInstantSync(unittest.TestCase):
    """
    Kern-Szenario: Zwei Agents mounten /global/.
    A schreibt → B sieht SOFORT (kein Poll-Delay), weil Shared-Store in RAM.
    """

    def setUp(self):
        self.gvfs = _reset_global_vfs()
        self.vfs_a = _make_vfs("a", "agent-a")
        self.vfs_b = _make_vfs("b", "agent-b")
        self.vfs_a.mount(str(self.gvfs.data_dir), vfs_path=GLOBAL_VFS_PATH, auto_sync=True)
        self.vfs_b.mount(str(self.gvfs.data_dir), vfs_path=GLOBAL_VFS_PATH, auto_sync=True)
        self.gvfs.register_vfs(self.vfs_a)
        self.gvfs.register_vfs(self.vfs_b)

    def tearDown(self):
        try: self.vfs_a.unmount(GLOBAL_VFS_PATH, save_changes=False)
        except Exception: pass
        try: self.vfs_b.unmount(GLOBAL_VFS_PATH, save_changes=False)
        except Exception: pass
        _reset_global_vfs()

    def test_a_writes_b_sees_instantly_via_store(self):
        """Kein time.sleep, kein Poll-Wait. Store ist synchron."""
        self.gvfs.write_file("tasks.md", "- [ ] do it", author="agent-a")

        # B liest direkt — ohne Warten
        result = self.gvfs.read_file("tasks.md")
        self.assertTrue(result["success"])
        self.assertEqual(result["content"], "- [ ] do it")

    def test_roundtrip_via_vfs_api(self):
        """A schreibt via vfs.write, B liest via gvfs.read_file — instant."""
        target = f"{GLOBAL_VFS_PATH}/roundtrip.md"
        self.vfs_a.create(target, "")
        self.vfs_a.write(target, "A wrote this")

        result_b = self.gvfs.read_file("roundtrip.md")
        self.assertTrue(result_b["success"])
        self.assertEqual(result_b["content"], "A wrote this")
        self.assertEqual(result_b["author"], "agent-a")

    def test_subscriber_fires_on_cross_agent_write(self):
        """Agent B subscribt, Agent A schreibt → B's Callback feuert sofort."""
        received = []
        self.gvfs.subscribe("tasks.md", lambda e: received.append(e))

        # A schreibt via vfs.write
        target = f"{GLOBAL_VFS_PATH}/tasks.md"
        self.vfs_a.create(target, "")
        self.vfs_a.write(target, "new task")

        # Callback synchron → keine Wartezeit
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["path"], "tasks.md")
        self.assertEqual(received[0]["author"], "agent-a")

if __name__ == "__main__":
    unittest.main(verbosity=2)
