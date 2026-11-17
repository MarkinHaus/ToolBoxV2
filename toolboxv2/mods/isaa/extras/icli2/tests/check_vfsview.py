"""Pruefung von vfsview.py mit Fake-VFS (gleiche Methodennamen und Rueckgabeformen wie vfs_v2.py)."""
import asyncio
import stub
from toolboxv2.mods.isaa.extras.icli2.vfsview import VfsViewMixin, split_partial, list_dir


class FakeFile:
    def __init__(self, text, dirty=False):
        self._text = text
        self.size = len(text)
        self.is_dirty = dirty


class FakeVFS:
    def __init__(self):
        self.calls = []
        self.directories = {"/": None, "/project": None, "/project/src": None, "/project/docs": None}
        self.files = {
            "/project/src/main.py": FakeFile("print(1)\n"),
            "/project/src/util.py": FakeFile("x" * 9000, dirty=True),
            "/project/README.md": FakeFile("# Projekt\n"),
            "/notes.txt": FakeFile("Notiz\n"),
        }

    def _normalize_path(self, path):
        self.calls.append("_normalize_path")
        if not path:
            return "/"
        if not path.startswith("/"):
            path = "/" + path
        return path.rstrip("/") or "/"

    def _is_directory(self, path):
        self.calls.append("_is_directory")
        return self._normalize_path(path) in self.directories

    def _is_file(self, path):
        self.calls.append("_is_file")
        return self._normalize_path(path) in self.files

    def _list_directory_contents(self, path):
        self.calls.append("_list_directory_contents")
        prefix = path if path == "/" else path + "/"
        out = []
        for d in self.directories:
            if d != path and d.startswith(prefix) and "/" not in d[len(prefix):]:
                out.append({"name": d[len(prefix):], "type": "directory", "path": d})
        for p, f in self.files.items():
            if p.startswith(prefix) and "/" not in p[len(prefix):]:
                out.append({"name": p[len(prefix):], "type": "file", "path": p, "size": f.size})
        return sorted(out, key=lambda x: (x["type"] != "directory", x["name"]))

    def read(self, path, max_chars=25000):
        self.calls.append("read")
        if path in self.files:
            return {"success": True, "content": self.files[path]._text}
        return {"success": False, "error": f"File not found: {path}"}


class Sess:
    def __init__(self, vfs):
        self.vfs = vfs


class H(VfsViewMixin):
    def __init__(self, vfs):
        self.vfs = vfs

    async def _session(self):
        return Sess(self.vfs)


def check_split():
    assert split_partial("/project/src/ma") == ("/project/src", "ma")
    assert split_partial("/project/src/") == ("/project/src", "")
    assert split_partial("") == ("/", "")
    assert split_partial("/") == ("/", "")
    assert split_partial("pro") == ("/", "pro")


def check_list_dir(vfs):
    rows = list_dir(vfs, "/project")
    assert [r["name"] for r in rows] == ["docs", "src", "README.md"], rows
    assert [r["dir"] for r in rows] == [True, True, False]
    for r in rows:
        assert set(r.keys()) == {"path", "name", "dir", "dirty", "size"}, r
    assert list_dir(vfs, "/gibt/es/nicht") == []
    src = {r["name"]: r for r in list_dir(vfs, "/project/src")}
    assert src["util.py"]["dirty"] is True and src["util.py"]["size"] == 9000
    assert src["main.py"]["dirty"] is False and src["main.py"]["size"] == 9


async def check_async(h):
    title, rows = await h._rows_vfs("")
    assert title == "VFS /", title
    assert [r["label"] for r in rows] == ["project/", "notes.txt"], rows

    title, rows = await h._rows_vfs("/project/src/ma")
    assert title == "VFS /project/src", title
    assert rows == [{"value": "/project/src/main.py", "label": "main.py", "hint": "9 Zeichen"}], rows

    _, rows = await h._rows_vfs("/pro")
    assert rows == [{"value": "/project/", "label": "project/", "hint": "Ordner"}], rows

    _, rows_upper = await h._rows_vfs("/PRO")
    assert rows_upper == rows

    _, rows = await h._rows_vfs("/project/src/")
    hints = {r["label"]: r["hint"] for r in rows}
    assert hints == {"main.py": "9 Zeichen", "util.py": "9000 Zeichen · geändert"}, hints

    for arg in ("", "/", "/project/", "/project/src/"):
        _, rs = await h._rows_vfs(arg)
        for r in rs:
            assert set(r.keys()) == {"value", "label", "hint"}, r
            assert all(isinstance(v, str) for v in r.values()), r

    d = await h._do_vfs("/project/src")
    lines = d["text"].split("\n")
    assert lines[0] == "/project/src", lines[0]
    assert "  main.py   9 Zeichen" in lines, lines
    assert "  util.py   9000 Zeichen *" in lines, lines

    d = await h._do_vfs("/project/src/main.py")
    assert "print(1)" in d["text"]

    d = await h._do_vfs("/project/src/util.py")
    assert d["text"].endswith("… gekürzt (9000 Zeichen)"), d["text"][-80:]
    assert "x" * MAX_CHARS in d["text"] and "x" * (MAX_CHARS + 1) not in d["text"]

    d = await h._do_vfs("/project/docs")
    assert d["text"].endswith("  (leer)"), d["text"]

    d = await h._do_vfs("")
    assert d["text"].split("\n")[0] == "/"

    try:
        await h._do_vfs("/nix")
    except FileNotFoundError as e:
        assert "Nicht im VFS: /nix" in str(e), e
    else:
        raise AssertionError("FileNotFoundError erwartet")


MAX_CHARS = 6000  # spiegelt vfsview.MAX_FILE_CHARS


def main():
    check_split()
    vfs = FakeVFS()
    check_list_dir(vfs)
    asyncio.run(check_async(H(vfs)))
    assert vfs.calls, "Fake-VFS wurde nicht genutzt"
    for name in vfs.calls:
        for bad in ("write", "mount", "sync", "delete", "remove"):
            assert bad not in name, name
    print("vfsview ok")


if __name__ == "__main__":
    main()
