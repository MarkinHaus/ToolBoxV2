# file: toolboxv2/mods/Kurse/tests/test_store.py
"""Behavioral invariants for the Kurse store — no real ToolBox needed.

Stubs `toolboxv2` (TBEF + Result) and drives the store against a fake DB that
mimics MiniDictDB semantics: exact key -> scalar, `prefix*` -> list.
Run: python -m unittest toolboxv2.mods.Kurse.tests.test_store -v
"""

import asyncio
import sys
import types
import unittest


# --- stub the toolboxv2 surface the store touches ---------------------------
def _install_stubs():
    tb = types.ModuleType("toolboxv2")

    class _Op:
        def __init__(self, n): self.n = n

    class DB:
        GET, SET, DELETE = _Op("get"), _Op("set"), _Op("delete")

    tb.TBEF = types.SimpleNamespace(DB=DB)
    tb.get_app = lambda *a, **k: _APP
    sys.modules["toolboxv2"] = tb
    return DB


class _Result:
    def __init__(self, data=None, err=False): self._d, self._e = data, err
    def is_error(self): return self._e
    def get(self): return self._d


class FakeApp:
    """MiniDictDB-like: get(exact)->scalar unwrap, get(prefix*)->list.
    hard_delete=False simulates blob mode where the adapter's delete no-ops."""
    def __init__(self, DB): self.data = {}; self.DB = DB; self.hard_delete = True

    async def a_run_any(self, op, query=None, data=None, matching=False, get_results=True):
        if op is self.DB.SET:
            self.data[query] = data; return _Result()
        if op is self.DB.DELETE:
            if self.hard_delete:
                self.data.pop(query, None)
            return _Result()
        # GET
        if query.endswith("*"):
            pre = query[:-1]
            vals = [v for k, v in self.data.items() if k.startswith(pre)]
            return _Result(vals) if vals else _Result(err=True)
        if query in self.data:
            return _Result(self.data[query])          # scalar unwrap
        hits = [v for k, v in self.data.items() if k.startswith(query)]
        return _Result(hits[0]) if len(hits) == 1 else (
            _Result(hits) if hits else _Result(err=True))


DB = _install_stubs()
_APP = FakeApp(DB)

import importlib.util, os                          # noqa: E402
_sp = os.path.join(os.path.dirname(__file__), "..", "store.py")
_spec = importlib.util.spec_from_file_location("kurse_store", _sp)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def run(coro): return asyncio.run(coro)


class StoreTest(unittest.TestCase):
    def setUp(self): _APP.data.clear()

    def test_course_roundtrip(self):
        c = run(S.course_create(_APP, "MP"))
        self.assertIn(c, run(S.course_list(_APP)))

    def test_files_flat_and_reorder(self):
        c = run(S.course_create(_APP, "JS"))
        a = run(S.file_create(_APP, c["id"], "S1", "<h1>x</h1>"))
        b = run(S.file_create(_APP, c["id"], "S2", "<p>y</p>"))
        order = [f["name"] for f in run(S.file_list(_APP, c["id"]))]
        self.assertEqual(order, ["S1", "S2"])
        run(S.file_reorder(_APP, c["id"], b["id"], -1))          # move S2 up
        order2 = [f["name"] for f in run(S.file_list(_APP, c["id"]))]
        self.assertEqual(order2, ["S2", "S1"])
        run(S.file_delete(_APP, c["id"], a["id"]))
        self.assertEqual(run(S.file_get(_APP, c["id"], a["id"])), None)

    def test_course_delete_cascades_files(self):
        c = run(S.course_create(_APP, "MP"))
        run(S.file_create(_APP, c["id"], "S1", "<h1>x</h1>"))
        run(S.course_delete(_APP, c["id"]))
        self.assertEqual(run(S.file_list(_APP, c["id"])), [])

    def test_validate(self):
        self.assertFalse(S.validate_file({"html": ""})["ok"])
        self.assertTrue(S.validate_file({"html": "<p>reveal(this,1,1)</p>"})["ok"])

    def test_cohort_anchor_clamped(self):
        co = run(S.cohort_create(_APP, "c", "K", ["a", "b", "c"], anchor=9))
        self.assertEqual(co["anchor"], 2)                   # clamped into range

    def test_participant_resume(self):
        co = run(S.cohort_create(_APP, "c", "K", ["a", "b", "c"], anchor=1))
        p1 = run(S.pp_join(_APP, co["id"], "Max"))
        self.assertEqual(p1["pos"], 1)                      # starts at anchor
        run(S.pp_event(_APP, co["id"], "Max", {"type": "task", "sIdx": 2, "taskIdx": 3}))
        p2 = run(S.pp_join(_APP, co["id"], "max"))          # slug-insensitive
        self.assertEqual((p2["pos"], p2["task"]), (2, 3))   # resumes

    def test_event_closes_previous(self):
        co = run(S.cohort_create(_APP, "c", "K", ["a"], anchor=0))
        run(S.pp_join(_APP, co["id"], "Ann"))
        run(S.pp_event(_APP, co["id"], "Ann", {"type": "hint", "taskIdx": 0, "hint": 1}))
        run(S.pp_event(_APP, co["id"], "Ann", {"type": "task", "taskIdx": 1}))
        stats = run(S.cohort_live(_APP, co["id"]))[0]
        self.assertEqual(stats["hints_seen"], 1)
        self.assertEqual(stats["cur_hints_open"], 0)            # hint got closed by next event

    def test_hint_open_counts_until_next(self):
        co = run(S.cohort_create(_APP, "c", "K", ["a"], anchor=0))
        run(S.pp_join(_APP, co["id"], "Bo"))
        run(S.pp_event(_APP, co["id"], "Bo", {"type": "hint", "taskIdx": 0, "hint": 1}))
        stats = run(S.cohort_live(_APP, co["id"]))[0]
        self.assertEqual(stats["cur_hints_open"], 1)            # still open

    def test_task_timer_survives_hint(self):
        co = run(S.cohort_create(_APP, "c", "K", ["a"], anchor=0))
        run(S.pp_join(_APP, co["id"], "Cy"))
        run(S.pp_event(_APP, co["id"], "Cy", {"type": "task", "taskIdx": 0}))
        run(S.pp_event(_APP, co["id"], "Cy", {"type": "hint", "taskIdx": 0, "hint": 1}))
        p = run(S.pp_join(_APP, co["id"], "Cy"))
        task_ev = [e for e in p["events"] if e["type"] == "task"][0]
        self.assertNotIn("closed_at", task_ev)              # hint did NOT close task
        run(S.pp_event(_APP, co["id"], "Cy", {"type": "task", "taskIdx": 1}))
        s = run(S.cohort_live(_APP, co["id"]))[0]
        self.assertEqual(s["cur_hints_open"], 0)                # new task closed the hint
        self.assertIn("prev_task_secs", s)

    def test_touch_pos_persists_task_same_sheet(self):
        co = run(S.cohort_create(_APP, "c", "K", ["a", "b"], anchor=0))
        run(S.pp_join(_APP, co["id"], "Di"))
        run(S.pp_event(_APP, co["id"], "Di", {"type": "task", "sIdx": 0, "taskIdx": 4}))
        run(S.pp_touch_pos(_APP, co["id"], "Di", 0))        # same sheet
        self.assertEqual(run(S.pp_join(_APP, co["id"], "Di"))["task"], 4)  # kept
        run(S.pp_touch_pos(_APP, co["id"], "Di", 1))        # new sheet
        self.assertEqual(run(S.pp_join(_APP, co["id"], "Di"))["task"], 0)  # reset

    def test_cohort_delete_removes(self):
        co = run(S.cohort_create(_APP, "c", "K", ["a"], anchor=0))
        run(S.pp_join(_APP, co["id"], "Ed"))
        run(S.cohort_delete(_APP, co["id"]))
        self.assertIsNone(run(S.cohort_get(_APP, co["id"])))
        self.assertEqual(run(S.cohort_list(_APP, "c")), [])

    def test_delete_works_in_blob_mode(self):
        _APP.hard_delete = False                    # adapter delete no-ops (blob)
        try:
            co = run(S.cohort_create(_APP, "c", "K", ["a"], anchor=0))
            run(S.cohort_delete(_APP, co["id"]))
            self.assertIsNone(run(S.cohort_get(_APP, co["id"])))   # tombstone filtered
            self.assertEqual(run(S.cohort_list(_APP, "c")), [])
        finally:
            _APP.hard_delete = True

    def test_cohort_update_keeps_id(self):
        co = run(S.cohort_create(_APP, "c", "K", ["a", "b"], anchor=0))
        up = run(S.cohort_update(_APP, co["id"], session_ids=["a", "b", "cc"], anchor=2))
        self.assertEqual(up["id"], co["id"])                # same link
        self.assertEqual(up["session_ids"], ["a", "b", "cc"])
        self.assertEqual(up["anchor"], 2)


if __name__ == "__main__":
    unittest.main()
