"""PathIndexDict: Ordner auflisten ueber den Kinder-Index - gleiches Ergebnis wie der fruehere
Durchlauf ueber alle Pfade, immer aktuell bei jeder Art von Aenderung, und schnell."""
import copy
import pickle
import random
import time
import unittest

from toolboxv2.mods.isaa.base.Agent.vfs_v2 import PathIndexDict


def old_children(keys, path):
    """Die fruehere Logik aus _list_directory_contents (Praefix + kein weiterer '/')."""
    prefix = path if path == "/" else path + "/"
    return sorted(k for k in keys if k != path and k.startswith(prefix) and "/" not in k[len(prefix):])


def new_children(d, path):
    return sorted(k for k in d.children(path) if k != path)


class PathIndexTest(unittest.TestCase):
    def check(self, d):
        keys = list(d)
        for path in {"/"} | {PathIndexDict.parent(k) or "/" for k in keys} | set(keys):
            self.assertEqual(new_children(d, path), old_children(keys, path), path)

    def test_random_operations_match_old_logic(self):
        rnd = random.Random(7)
        names = ["a", "b", "src", "x.py", "README.md", ".hidden", "ü"]
        d = PathIndexDict({"/": 1})
        for step in range(3000):
            depth = rnd.randint(1, 4)
            path = "/" + "/".join(rnd.choice(names) for _ in range(depth))
            op = rnd.random()
            if op < 0.5:
                d[path] = step
            elif op < 0.6:
                d.pop(path, None)
            elif op < 0.7 and path in d:
                del d[path]
            elif op < 0.75:
                d.setdefault(path, step)
            elif op < 0.8:
                d.update({path: step, path + "/n": step})
            elif op < 0.85:
                d |= {path + "/m": step}
            elif op < 0.86 and len(d) > 1:
                d.popitem()
            if step % 250 == 0:
                self.check(d)
        self.check(d)
        d.clear()
        self.assertEqual(d.children("/"), [])
        d["/neu"] = 1
        self.assertEqual(d.children("/"), ["/neu"])

    def test_pickle_copy_keep_index(self):
        d = PathIndexDict({"/a": 1, "/a/b": 2, "/c": 3})
        for clone in (pickle.loads(pickle.dumps(d)), copy.deepcopy(d), d.copy()):
            self.assertIsInstance(clone, PathIndexDict)
            self.assertEqual(sorted(clone.children("/")), ["/a", "/c"])
            clone["/a/z"] = 4
            self.assertEqual(sorted(clone.children("/a")), ["/a/b", "/a/z"])
        self.assertEqual(sorted(d.children("/a")), ["/a/b"])       # Kopien sind unabhaengig

    def test_vfs_lists_current_state_fast(self):
        from toolboxv2.mods.isaa.base.Agent.vfs_v2 import VirtualFileSystemV2, VFSDirectory
        vfs = VirtualFileSystemV2(session_id="t", agent_name="t")
        vfs.mkdir("/project/src", parents=True)
        vfs.write("/project/src/main.py", "print(1)")
        vfs.write("/project/README.md", "# x")
        names = lambda p: [(e["type"], e["name"]) for e in vfs._list_directory_contents(p)]
        self.assertEqual(names("/project"), [("directory", "src"), ("file", "README.md")])
        vfs.mv("/project/README.md", "/project/src/README.md")
        self.assertEqual(names("/project"), [("directory", "src")])
        self.assertIn(("file", "README.md"), names("/project/src"))
        vfs.directories["/project/extern"] = VFSDirectory(name="extern")   # Aenderung von aussen
        self.assertIn(("directory", "extern"), names("/project"))
        # alte Zustaende (plain dict) werden beim ersten Auflisten umgestellt
        vfs.files = dict(vfs.files)
        self.assertIn(("file", "main.py"), names("/project/src"))
        self.assertIsInstance(vfs.files, PathIndexDict)
        # Laufzeit: 100k Dateien, ein Ordner mit 10 Eintraegen
        for i in range(100_000):
            vfs.files[f"/big/d{i % 1000}/f{i}.txt"] = vfs.files["/project/src/main.py"]
        t0 = time.perf_counter()
        for _ in range(200):
            vfs._list_directory_contents("/project/src")
        per_call = (time.perf_counter() - t0) / 200
        self.assertLess(per_call, 0.002, f"{per_call * 1000:.2f} ms pro Aufruf")


if __name__ == "__main__":
    unittest.main()
