"""WP1 S2: blob sync local SQLite <-> cloud (MinIO).

WP1 S2 four cases, binary-safe: blobs are bytes, never JSON. Uses the
measured BlobManager forms: list -> dicts (blob_id, size, updated_at,
sync_status), read -> bytes, write -> (data, encrypted) -> blob_id.
Direction decisions reuse the S1 table (decide, EntryMeta).
"""
import asyncio
import tempfile
import unittest
from pathlib import Path

from toolboxv2.mods.CloudM.sync.blob_sync import sync_blobs


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class _MemBlobStore:
    """Mirrors measured forms: list_blobs/read_blob/write_blob."""

    def __init__(self, data=None):
        self.data = dict(data or {})  # blob_id -> bytes
        self.ts = {}                  # blob_id -> updated_at
        self.writes = []

    async def list_blobs(self, prefix=""):
        return [{"blob_id": b, "size": len(d), "updated_at": self.ts.get(b, 0.0),
                 "sync_status": "local_only"}
                for b, d in self.data.items()]

    async def read_blob(self, blob_id):
        return self.data.get(blob_id)

    async def write_blob(self, blob_id, data, encrypted=True):
        self.data[blob_id] = bytes(data)
        self.writes.append(blob_id)
        return blob_id


class BlobSyncTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def _run(self, local, remote, lts=None, rts=None):
        loc = _MemBlobStore(local)
        rem = _MemBlobStore(remote)
        loc.ts = dict(lts or {})
        rem.ts = dict(rts or {})
        res = run(sync_blobs(app=None, local_store=loc, remote_store=rem,
                             state_file=self.tmp / "_last_sync.json"))
        return res, loc, rem

    def test_case_1_local_only_pushes_binary(self):
        payload = bytes([0, 255, 0, 10]) + "ü".encode()
        res, loc, rem = self._run({"b1": payload}, {})
        self.assertEqual(res["pushed"], ["b1"])
        self.assertEqual(rem.data["b1"], payload)  # binary roundtrip intact

    def test_case_2_remote_only_pulls(self):
        res, loc, rem = self._run({}, {"b1": b"\x00\x01\x02"})
        self.assertEqual(res["pulled"], ["b1"])
        self.assertEqual(loc.data["b1"], b"\x00\x01\x02")

    def test_case_3_remote_newer_pulls(self):
        res, loc, rem = self._run({"b1": b"old"}, {"b1": b"new"},
                                  lts={"b1": 1.0}, rts={"b1": 5.0})
        self.assertEqual(res["pulled"], ["b1"])
        self.assertEqual(loc.data["b1"], b"new")

    def test_case_4_equal_noop(self):
        res, loc, rem = self._run({"b1": b"same"}, {"b1": b"same"},
                                  lts={"b1": 2.0}, rts={"b1": 2.0})
        self.assertEqual(res["noop"], ["b1"])
        self.assertEqual(loc.writes, [])
        self.assertEqual(rem.writes, [])

    def test_state_file_written(self):
        res, loc, rem = self._run({"b1": b"x"}, {})
        import json
        state = json.loads((self.tmp / "_last_sync.json").read_text(encoding="utf-8"))
        self.assertEqual(state["last_directions"]["b1"], "push")

    def test_never_json_encodes_binary(self):
        res, loc, rem = self._run({"b1": b"\x00raw"}, {})
        self.assertIsInstance(rem.data["b1"], bytes)


if __name__ == "__main__":
    unittest.main()
