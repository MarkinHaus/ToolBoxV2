"""Headless-Pruefungen der icli2 ohne installiertes toolboxv2 (nur textual, prompt_toolkit, dill).
Aufruf: python check_ui.py && python check_backend.py && python check_autopack.py"""
"""Laedt nur die echten Branding-Dateien, ohne das schwere toolboxv2-Paket."""
import sys, types, importlib.util
from pathlib import Path
R=str(Path(__file__).resolve().parents[5])
def pkg(n,path=None):
    m=types.ModuleType(n); m.__path__=[path] if path else []; sys.modules[n]=m; return m
pkg("toolboxv2"); pkg("toolboxv2.mods"); pkg("toolboxv2.mods.isaa"); pkg("toolboxv2.mods.isaa.base")
pkg("toolboxv2.mods.isaa.extras", R+"/mods/isaa/extras"); pkg("toolboxv2.mods.isaa.extras.zen", R+"/mods/isaa/extras/zen")
au=types.ModuleType("toolboxv2.mods.isaa.base.AgentUtils"); au.anything_from_str_to_dict=lambda *a,**k:{}; sys.modules[au.__name__]=au
import logging
sys.modules["toolboxv2"].get_logger=lambda *a,**k: logging.getLogger("tb")
