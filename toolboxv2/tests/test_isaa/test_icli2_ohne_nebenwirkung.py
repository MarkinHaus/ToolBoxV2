# -*- coding: utf-8 -*-
"""Kein Nebeneffekt beim Import, kein Schreiben ins echte Profil, kein litellm.

1. `import icli2/app.py` (die reine UI) legt keine ToolBox-App an und laedt den isaa-Mod nicht.
   Frueher: toolboxv2/mods/isaa/__init__.py importierte module.py, dort `get_app("isaa.Export")` auf
   Modulebene -> App mit echtem Profil, bevor ein Test seine Umgebung umbiegen konnte.
2. Der Mod-Lader findet isaa trotzdem (Tools/tools/version ueber das Paket).
3. Der Schreibschutz der Betriebstests erkennt jede Aenderung in geschuetzten Ordnern.
Kindprozesse mit eigener, umgebogener Umgebung (HOME, XDG_*, TB_DATA_DIR) - wie ein frischer Start.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from toolboxv2.tests.schreibschutz import Schreibschutz

pytestmark = [pytest.mark.icli2_betrieb, pytest.mark.timeout(120)]


def _python(code: str, root: Path) -> str:
    env = {**os.environ, "TB_DATA_DIR": str(root), "HOME": str(root / "home"),
           "XDG_CONFIG_HOME": str(root / "cfg"), "XDG_DATA_HOME": str(root / "share"),
           "APPDATA": str(root / "appdata"), "LOCALAPPDATA": str(root / "localappdata")}
    code = "import sys\nsys.modules['litellm'] = None\n" + code     # litellm gesperrt, eigener Prozess
    proc = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(root), capture_output=True,
                          text=True, timeout=100)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
    return proc.stdout


def _dateien(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_import_der_ui_legt_keine_app_an(tmp_path):
    schutz = Schreibschutz()
    out = _python(
        "import toolboxv2.mods.isaa.extras.icli2.app\n"
        "from toolboxv2.utils.system.getting_and_closing_app import registered_apps\n"
        "print('APP', registered_apps[0] is not None)\n"
        "print('ISAA_MODULE', 'toolboxv2.mods.isaa.module' in sys.modules)\n", tmp_path)
    assert "APP False" in out, out
    assert "ISAA_MODULE False" in out, out
    assert _dateien(tmp_path) == [], _dateien(tmp_path)   # nichts geschrieben, auch nicht im Temp-Ordner
    schutz.pruefen()


def test_mod_lader_findet_isaa_weiterhin(tmp_path):
    schutz = Schreibschutz()
    out = _python(
        "import toolboxv2.mods.isaa as pkg\n"
        "print('NAME', pkg.Name)\n"
        "print('HAS_TOOLS', hasattr(pkg, 'Tools'), pkg.Tools.__module__, pkg.tools is pkg.Tools)\n"
        "print('VERSION', pkg.version)\n"
        "from toolboxv2.mods.isaa import Tools\n"
        "print('IMPORT_TOOLS', Tools is pkg.Tools)\n", tmp_path)
    assert "NAME isaa" in out
    assert "HAS_TOOLS True toolboxv2.mods.isaa.module True" in out, out
    assert "VERSION 0.3.0" in out and "IMPORT_TOOLS True" in out, out
    schutz.pruefen()


def test_schreibschutz_erkennt_aenderungen(tmp_path):
    geschuetzt = tmp_path / "profil"
    geschuetzt.mkdir()
    (geschuetzt / "settings.json").write_text("{}")
    schutz = Schreibschutz([geschuetzt])
    schutz.pruefen()                                       # unveraendert: kein Fehler
    (geschuetzt / "widgets.json").write_text("{}")
    with pytest.raises(AssertionError, match="widgets.json"):
        schutz.pruefen()
    schutz = Schreibschutz([geschuetzt])
    (geschuetzt / "settings.json").write_text('{"max_iterations": 77}')
    with pytest.raises(AssertionError, match="settings.json"):
        schutz.pruefen()
    schutz = Schreibschutz([geschuetzt])
    (geschuetzt / "settings.json").unlink()
    with pytest.raises(AssertionError, match="geloescht"):
        schutz.pruefen()
    fehlt = tmp_path / "noch_nicht_da"                     # Profilordner, den es noch nicht gibt
    schutz = Schreibschutz([fehlt])
    fehlt.mkdir()                                          # z. B. App-Start mit makedirs(appdata)
    with pytest.raises(AssertionError, match="noch_nicht_da"):
        schutz.pruefen()


def test_schreibschutz_erkennt_neue_dateien_im_checkout(tmp_path):
    repo = tmp_path / "checkout"                           # eigenes git-Repo statt des echten Checkouts
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "docs").mkdir()
    schutz = Schreibschutz([], repo=repo)
    schutz.pruefen()
    (repo / "docs" / ".docs_index.json").write_text("{}")    # wie frueher TestDocsFeatureLive
    with pytest.raises(AssertionError, match="docs_index"):
        schutz.pruefen()

