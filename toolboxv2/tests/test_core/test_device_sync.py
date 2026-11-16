"""WP1 S1: account data sync local <-> cloud.

Decision table (WP1 S1):
 (1) local.new + remote.missing            -> push
 (2) remote.new + local.missing            -> pull
 (3) same version+ts                       -> noop
 (4) both changed, remote newer            -> pull
 (5) both changed, local newer             -> push
 (6) cycle suspicion (last direction push,
     now pull with same versions)          -> remote wins + local backup
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from toolboxv2.mods.CloudM.sync.device_sync import (
    EntryMeta, SyncDecision, decide, sync_account_data,
)


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class DecisionTableTest(unittest.TestCase):
    def t(self, key, local, remote):
        return decide(key, local, remote)

    def test_1_local_new_push(self):
        d = self.t("k", EntryMeta(version=1, updated_at=2.0), None)
        self.assertEqual(d.action, "push")

    def test_2_remote_new_pull(self):
        d = self.t("k", None, EntryMeta(version=1, updated_at=2.0))
        self.assertEqual(d.action, "pull")

    def test_3_equal_noop(self):
        d = self.t("k", EntryMeta(1, 2.0), EntryMeta(1, 2.0))
        self.assertEqual(d.action, "noop")

    def test_4_remote_newer_pull(self):
        d = self.t("k", EntryMeta(1, 1.0), EntryMeta(1, 5.0))
        self.assertEqual(d.action, "pull")

    def test_5_local_newer_push(self):
        d = self.t("k", EntryMeta(2, 5.0), EntryMeta(1, 1.0))
        self.assertEqual(d.action, "push")

    def test_6_same_ts_diff_version_conflict_remote_wins(self):
        d = self.t("k", EntryMeta(2, 5.0), EntryMeta(7, 5.0))
        self.assertEqual(d.action, "conflict-remote-wins")
        self.assertEqual(d.reason, "cycle-suspected")


class _Mem:
    """Local+remote stores as dicts, mirroring measured response shapes."""

    def __init__(self, data=None):
        self.data = dict(data or {})

    async def list_keys(self):
        return list(self.data)

    async def read(self, key):
        return self.data.get(key)

    async def write(self, key, payload):
        self.data[key] = payload


class SyncAccountDataTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def _run(self, local, remote, last=None):
        loc, rem = _Mem(local), _Mem(remote)
        res = run(sync_account_data(
            app=None, local_store=loc, remote_store=rem,
            state_file=self.tmp / "_last_sync.json", last_directions=last))
        return res, loc.data, rem.data

    def test_push_case(self):
        res, l, r = self._run({"a": {"v": 1}}, {})
        self.assertEqual(res["pushed"], 1)
        self.assertEqual(r["a"], {"v": 1})

    def test_pull_case(self):
        res, l, r = self._run({}, {"a": {"v": 1}})
        self.assertEqual(res["pulled"], 1)
        self.assertEqual(l["a"], {"v": 1})

    def test_noop_case(self):
        res, l, r = self._run({"a": {"v": 1}}, {"a": {"v": 1}})
        self.assertEqual(res["noop"], 1)
        self.assertEqual(res["pulled"] + res["pushed"], 0)

    def test_remote_newer_wins(self):
        res, l, r = self._run(
            {"a": {"v": 1, "updated_at": 1.0}},
            {"a": {"v": 2, "updated_at": 5.0}})
        self.assertEqual(res["pulled"], 1)
        self.assertEqual(l["a"]["v"], 2)

    def test_local_newer_pushes(self):
        res, l, r = self._run(
            {"a": {"v": 2, "updated_at": 5.0}},
            {"a": {"v": 1, "updated_at": 1.0}})
        self.assertEqual(res["pushed"], 1)
        self.assertEqual(r["a"]["v"], 2)

    def test_conflict_backs_up_local_then_remote_wins(self):
        res, l, r = self._run(
            {"a": {"v": 2, "updated_at": 5.0, "version": 2}},
            {"a": {"v": 9, "updated_at": 5.0, "version": 7}},
            last={"a": "push"})
        self.assertEqual(res["conflicts"], ["a"])
        self.assertEqual(l["a"]["v"], 9)
        backups = list(self.tmp.glob("backup_*.a.json"))
        self.assertEqual(len(backups), 1)

    def test_state_file_written(self):
        res, l, r = self._run({"a": {"v": 1}}, {})
        state = json.loads((self.tmp / "_last_sync.json").read_text())
        self.assertEqual(state["last_directions"]["a"], "push")
        self.assertIn("last_run", state)


if __name__ == "__main__":
    unittest.main()
