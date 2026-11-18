# -*- coding: utf-8 -*-
"""Schreibschutz fuer Tests: Tests duerfen echte Zustandsordner (Profil-Config, appdata, .data/.info/.config
des Checkouts) nie anlegen, aendern oder loeschen (testing_guide Regel 6.3).

Benutzt von conftest.py (ganze Sitzung) und icli2_betrieb.py (jeder Betriebstest). Nur Standardbibliothek.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]          # <repo>/toolboxv2 (Zustand im Source-Checkout)


def echte_zustandsordner() -> list[Path]:
    """Ordner, in die kein Test schreiben darf. VOR dem Umbiegen der Umgebung aufrufen."""
    home = Path.home()
    dirs = [home / ".config" / "ToolBoxV2", home / ".local" / "share" / "toolboxv2",
            home / "AppData" / "Roaming" / "ToolBoxV2", home / "AppData" / "Local" / "toolboxv2",
            home / "Library" / "Application Support" / "toolboxv2",
            _PKG / ".data", _PKG / ".info", _PKG / ".config", _PKG.parent / ".data"]
    for var in ("APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME"):
        if os.environ.get(var):
            dirs.append(Path(os.environ[var]) / "ToolBoxV2")
            dirs.append(Path(os.environ[var]) / "toolboxv2")
    return list(dict.fromkeys(dirs))


def checkout_stand(repo: Path = _PKG.parent) -> set[str] | None:
    """`git status` des Checkouts (neue unversionierte + geaenderte Dateien); None ohne git/Checkout."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return set(out.stdout.splitlines()) if out.returncode == 0 else None


#: beim ERSTEN Import bestimmt - conftest.py importiert dieses Modul, bevor es die Umgebung umbiegt
ECHTE_ZUSTANDSORDNER: list[Path] = echte_zustandsordner()


class Schreibschutz:
    """Stand (Pfad -> mtime, Groesse) der geschuetzten Ordner vorher/nachher; jeder Unterschied = Fehler."""

    def __init__(self, dirs: list[Path] | None = None, repo: Path | None = None):
        self.dirs = list(ECHTE_ZUSTANDSORDNER if dirs is None else dirs)
        self.vorher = self._stand()
        self.repo = repo or (_PKG.parent if dirs is None else None)     # Standard: das echte Checkout
        self.checkout = checkout_stand(self.repo) if self.repo is not None else None

    def _stand(self) -> dict[str, tuple[int, int]]:
        out = {}
        for d in self.dirs:
            if d.is_dir():
                out[str(d) + "/"] = (0, 0)           # Ordner selbst: Anlegen/Loeschen zaehlt auch
                for f in d.rglob("*"):
                    try:
                        if f.is_file():
                            st = f.stat()
                            out[str(f)] = (st.st_mtime_ns, st.st_size)
                        elif f.is_dir():
                            out[str(f) + "/"] = (0, 0)
                    except OSError:                  # waehrend des Laufs geloescht
                        continue
        return out

    def unterschiede(self) -> str:
        nachher = self._stand()
        neu = sorted(set(nachher) - set(self.vorher))
        weg = sorted(set(self.vorher) - set(nachher))
        geaendert = sorted(k for k in set(nachher) & set(self.vorher) if nachher[k] != self.vorher[k])
        im_checkout = []
        if self.checkout is not None:
            jetzt = checkout_stand(self.repo)
            im_checkout = sorted((jetzt or set()) - self.checkout)
        if not (neu or weg or geaendert or im_checkout):
            return ""
        return (f"Test hat echte Zustandsordner veraendert: neu={neu[:10]} geaendert={geaendert[:10]} "
                f"geloescht={weg[:10]} im_checkout={im_checkout[:10]}")

    def pruefen(self) -> None:
        text = self.unterschiede()
        if text:
            raise AssertionError(text)
