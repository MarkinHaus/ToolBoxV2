"""WP1 S3: global ISAA folder as blobs + one entry point sync_all.

Doc criteria (wörtlich): global/notes/a.txt arrives at device B;
isaa/global/../x is never written; without device account -> skipped;
second run -> zero transfers. No power_vfs import (path built from
F22: <data_dir>/Agents/VFS/global), no deletions, files >50 MB and
unsafe paths skipped.
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from toolboxv2.mods.CloudM.sync.global_sync import sync_all


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class _MemBlobStore:
    def __init__(self, data=None, ts=None):
        self.data = dict(data or {})
        self.ts = dict(ts or {})
        self.writes = []

    async def list_blobs(self, prefix=""):
        return [{"blob_id": b, "size": len(d),
                 "updated_at": self.ts.get(b, 0.0)}
                for b, d in self.data.items() if b.startswith(prefix)]

    async def read_blob(self, blob_id):
        return self.data.get(blob_id)

    async def write_blob(self, blob_id, data, encrypted=True):
        self.data[blob_id] = bytes(data)
        self.writes.append(blob_id)
        return blob_id


def _app(tmp, with_account=True):
    app = SimpleNamespace(data_dir=str(tmp))
    if with_account:
        (tmp / "device_owner.json").write_text(json.dumps(
            {"user_id": "usr_acc1", "username": "markin"}), encoding="utf-8")
    return app


def _global_dir(tmp):
    d = Path(tmp) / "Agents" / "VFS" / "global"
    d.mkdir(parents=True, exist_ok=True)
    return d


class GlobalSyncTest(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.root = Path(self._t.name)
        self.tmp_a = self.root / "a"; self.tmp_a.mkdir()
        self.tmp_b = self.root / "b"; self.tmp_b.mkdir()
        self.cloud = _MemBlobStore()

    def _sync(self, app, local, tmp):
        return run(sync_all(app, local_blobs=local, remote_blobs=self.cloud,
                            state_file=Path(tmp) / "_global_sync.json"))

    def test_a_to_b_transfer(self):
        g = _global_dir(self.tmp_a)
        (g / "notes").mkdir(parents=True, exist_ok=True)
        (g / "notes" / "a.txt").write_text("hello sync", encoding="utf-8")
        res_a = self._sync(_app(self.tmp_a), _MemBlobStore(), self.tmp_a)
        self.assertEqual(res_a["export"]["pushed"], ["isaa/global/notes/a.txt"])
        res_b = self._sync(_app(self.tmp_b), _MemBlobStore(), self.tmp_b)
        arrived = self.tmp_b / "Agents" / "VFS" / "global" / "notes" / "a.txt"
        self.assertTrue(arrived.is_file())
        self.assertEqual(arrived.read_text(encoding="utf-8"), "hello sync")

    def test_traversal_blob_never_written_outside(self):
        evil = b"x"
        self.cloud.data["isaa/global/../x"] = evil
        self.cloud.ts["isaa/global/../x"] = 9.0
        res = self._sync(_app(self.tmp_b), _MemBlobStore(), self.tmp_b)
        outside = self.root / "x"
        self.assertFalse(outside.is_file())
        self.assertIn("isaa/global/../x", res["import"]["skipped"])

    def test_without_account_skipped(self):
        g = _global_dir(self.tmp_a)
        (g / "notes" / "a.txt") if False else None
        res = self._sync(_app(self.tmp_a, with_account=False),
                         _MemBlobStore(), self.tmp_a)
        self.assertTrue(res["skipped"])
        self.assertEqual(self.cloud.writes, [])
        self.assertIn("finished_at", res)

    def test_second_run_zero_transfers(self):
        g = _global_dir(self.tmp_a)
        (g / "notes").mkdir(parents=True, exist_ok=True)
        (g / "notes" / "a.txt").write_text("stable", encoding="utf-8")
        loc_a = _MemBlobStore()
        self._sync(_app(self.tmp_a), loc_a, self.tmp_a)
        self._sync(_app(self.tmp_b), _MemBlobStore(), self.tmp_b)
        res2 = self._sync(_app(self.tmp_a), loc_a, self.tmp_a)
        self.assertEqual(res2["blobs"]["pushed"], [])
        self.assertEqual(res2["blobs"]["pulled"], [])
        self.assertEqual(res2["import"]["written"], 0)


if __name__ == "__main__":
    unittest.main()
