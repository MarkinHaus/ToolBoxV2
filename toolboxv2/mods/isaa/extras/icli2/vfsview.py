"""Nur-Lese-Ansicht des Agenten-VFS fuer icli2.

Listet Ordner, zeigt Dateien und liefert Pfad-Vorschlaege beim Tippen.
Nutzt nur: _normalize_path, _is_directory, _is_file, _list_directory_contents,
files (Attribut) und read. Das VFS wird nie veraendert.
"""

MAX_FILE_CHARS = 6000      # laengere Dateien werden gekuerzt angezeigt
MAX_ROWS = 40              # hoechstens so viele Vorschlaege


def split_partial(partial: str) -> tuple[str, str]:
    """Zerlegt einen angefangenen Pfad in Ordner und Namens-Praefix.

    "/project/src/ma" -> ("/project/src", "ma"); "/project/src/" -> ("/project/src", "");
    "" oder "/" -> ("/", ""); "pro" -> ("/", "pro"). Rein, ohne VFS.
    """
    head, _, tail = partial.rpartition("/")
    return "/" + head.lstrip("/"), tail


def list_dir(vfs, path: str) -> list[dict]:
    """Direkte Eintraege eines VFS-Ordners: Ordner zuerst, dann Dateien, je alphabetisch.

    Unbekannter Ordner -> []. Wirft nie.
    """
    try:
        folder = vfs._normalize_path(path)
        if not vfs._is_directory(folder):
            return []
        rows = []
        for entry in vfs._list_directory_contents(folder):
            is_dir = entry.get("type") == "directory"
            # ungespeicherte Aenderung nur bei Dateien
            f = None if is_dir else vfs.files.get(entry["path"])
            rows.append({
                "path": entry["path"],
                "name": entry["name"],
                "dir": is_dir,
                "dirty": bool(getattr(f, "is_dirty", False)),
                # Zeichenanzahl; 0 bei Ordnern oder unbekannter Groesse
                "size": 0 if is_dir else int(entry.get("size") or 0),
            })
        return sorted(rows, key=lambda r: (not r["dir"], r["name"].lower(), r["name"]))
    except Exception:
        return []


class VfsViewMixin:
    """Mixin fuer die icli2-App: stellt VFS-Vorschlaege und Anzeige bereit."""

    async def _rows_vfs(self, arg: str = "") -> tuple[str, list[dict]]:
        """Vorschlaege fuer einen angefangenen Pfad `arg` (kann leer sein)."""
        vfs = (await self._session()).vfs
        ordner, prefix = split_partial(arg or "")
        pfx = prefix.lower()
        rows = []
        for e in list_dir(vfs, ordner):
            if not e["name"].lower().startswith(pfx):
                continue
            if e["dir"]:
                value, label, hint = e["path"] + "/", e["name"] + "/", "Ordner"
            else:
                hint = f"{e['size']} Zeichen" + (" · geändert" if e["dirty"] else "")
                value, label = e["path"], e["name"]
            rows.append({"value": value, "label": label, "hint": hint})
            if len(rows) >= MAX_ROWS:
                break
        return "VFS " + ordner, rows

    async def _do_vfs(self, path: str = "") -> dict:
        """Auflistung eines Ordners oder Inhalt einer Datei als {"text": ...}."""
        vfs = (await self._session()).vfs
        p = vfs._normalize_path(path or "/")

        if vfs._is_directory(p):
            entries = list_dir(vfs, p)
            lines = [p]
            for e in entries:
                if e["dir"]:
                    lines.append(f"  {e['name']}/")
                else:
                    star = " *" if e["dirty"] else ""
                    lines.append(f"  {e['name']}   {e['size']} Zeichen{star}")
            if not entries:
                lines.append("  (leer)")
            return {"text": "\n".join(lines)}

        if not vfs._is_file(p):
            raise FileNotFoundError(f"Nicht im VFS: {p}")

        f = vfs.files.get(p)
        try:
            r = vfs.read(p, max_chars=10**9)  # kein Kuerzen im VFS, wir kuerzen selbst
            if not r.get("success"):
                raise OSError(r.get("error", "unbekannter Fehler"))
            body = str(r.get("content") or "")
            n = len(body)
        except Exception as e:
            body = f"(Inhalt nicht lesbar: {e})"
            n = int(getattr(f, "size", 0) or 0)

        head = f"{p} · {n} Zeichen" + (" · geändert" if getattr(f, "is_dirty", False) else "")
        if n > MAX_FILE_CHARS:
            body = body[:MAX_FILE_CHARS] + f"\n… gekürzt ({n} Zeichen)"
        return {"text": f"{head}\n\n{body}"}
