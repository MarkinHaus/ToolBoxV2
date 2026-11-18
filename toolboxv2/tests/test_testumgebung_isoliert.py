# -*- coding: utf-8 -*-
"""Die Testumgebung (conftest.py) biegt ALLE Zustandsorte einer App in den Temp-Ordner um:
data_dir (TB_DATA_DIR) und appdata (XDG_CONFIG_HOME bzw. APPDATA). Kein Test schreibt ins echte Profil."""
import os
import tempfile
from pathlib import Path

from toolboxv2.tests.schreibschutz import ECHTE_ZUSTANDSORDNER


def test_app_der_tests_liegt_im_temp_ordner():
    from toolboxv2 import get_app
    app = get_app("isolation-check")
    tmp = Path(tempfile.gettempdir()).resolve()
    for name in ("appdata", "data_dir"):
        path = Path(getattr(app, name)).resolve()
        assert tmp in path.parents, f"{name}={path} liegt nicht im Temp-Ordner {tmp}"
        for real in ECHTE_ZUSTANDSORDNER:
            assert real.resolve() not in (path, *path.parents), f"{name}={path} liegt in {real}"
    for var in ("XDG_CONFIG_HOME", "APPDATA", "TB_DATA_DIR"):
        assert tmp in Path(os.environ[var]).resolve().parents, f"{var}={os.environ[var]}"
