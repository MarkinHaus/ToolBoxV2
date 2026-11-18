# -*- coding: utf-8 -*-
"""T3: Subcommand-Autocomplete - suggest_args() reine Vorschlagslogik (textual-frei)."""
import ast
from pathlib import Path

import pytest

from toolboxv2.mods.isaa.extras.icli2 import autocomplete as ac


def test_suggest_args_exists():
    assert hasattr(ac, "suggest_args")


def test_plugin_first_sub():
    assert ac.suggest_args("/plugin ") == [
        "confirm ", "discard ", "install ", "market ", "markets ", "remove "]


def test_plugin_prefix_filter():
    assert ac.suggest_args("/plugin re") == ["remove "]
    assert ac.suggest_args("/plugin m") == ["market ", "markets "]


def test_set_keys():
    sug = ac.suggest_args("/set ")
    assert "max_iterations " in sug
    assert "voice_input_device " in sug
    assert "tts_voice " in sug


def test_model_targets_after_dynamic_name():
    """target kommt an 2. Position (name ist frei): nur unverbrauchte statische Tokens."""
    assert ac.suggest_args("/model x ") == ["both ", "complex ", "fast "]
    assert ac.suggest_args("/model x c") == ["complex "]
    assert ac.suggest_args("/model x fast c") == []   # kein 3. Argument im Dispatcher


def test_job_dynamic_first_token():
    # /job: erster Token ist frei (Jobname) -> kein statischer Vorschlag
    assert ac.suggest_args("/job ") == []


def test_no_subcommands():
    assert ac.suggest_args("/settings ") == []
    assert ac.suggest_args("/agent ") == []
    assert ac.suggest_args("ohne slash") == []
    assert ac.suggest_args("") == []


def test_full_completion_exhausted():
    """Nach statischen Tokens kommt freier Text (owner/repo) -> kein Vorschlag."""
    assert ac.suggest_args("/plugin install ") == []


def test_app_integrity_via_ast():
    """Alle SUBCOMMANDS-Keys sind echte Befehle in app.py:COMMANDS (AST, kein Import)."""
    src = Path(ac.__file__).with_name("app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    cmds = None
    subs = None
    cmds = None
    subs = None
    assigns = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            assigns += list(zip(node.targets, [node.value] * len(node.targets)))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            assigns.append((node.target, node.value))
    for tgt, value in assigns:
        name = tgt.id if isinstance(tgt, ast.Name) else ""
        if name == "COMMANDS" and isinstance(value, ast.Dict):
            cmds = {k.value for k in value.keys if isinstance(k, ast.Constant)}
        if name == "SUBCOMMANDS":
            subs = True                              # Zuweisung existiert (Alias)
    assert cmds is not None
    assert subs
    assert set(ac.SUBCOMMANDS) <= cmds
