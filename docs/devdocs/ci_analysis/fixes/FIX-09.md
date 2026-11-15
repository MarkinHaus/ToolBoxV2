# FIX-09 – aus Karte K-MISS-01

Paket: **P1-AUTO-MINI** (Minimaler Fix, automatisch) · Fix-Typ: **A** · Karte: GETESTET · betrifft 26 Test-IDs

```
Fix:           FIX-09  (aus Karte K-MISS-01)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_clis/test_minicli_macros.py   (nur diese ändern)
Vorher lesen:  alle `patch("toolboxv2.flows.minicli.` Stellen
```

**Schritte:**

1. Jede Zeichenkette `"toolboxv2.flows.minicli.` durch `"toolboxv2.flows.mini.minicli.` ersetzen (4 Stellen).

**Nicht tun:** Nichts sonst. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- `python -m pytest toolboxv2/tests/test_clis/test_minicli_macros.py -q` → `28 passed`.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_clis/test_minicli_macros.py::TestControlFlow::test_execute_macro_commands_break_stops_loop
toolboxv2/tests/test_clis/test_minicli_macros.py::TestControlFlow::test_execute_macro_commands_handles_comment
toolboxv2/tests/test_clis/test_minicli_macros.py::TestControlFlow::test_for_loop_iterates
toolboxv2/tests/test_clis/test_minicli_macros.py::TestControlFlow::test_if_break_sets_flag
toolboxv2/tests/test_clis/test_minicli_macros.py::TestControlFlow::test_if_false_skips_action
toolboxv2/tests/test_clis/test_minicli_macros.py::TestControlFlow::test_if_true_executes_action
toolboxv2/tests/test_clis/test_minicli_macros.py::TestControlFlow::test_while_respects_1000_limit
toolboxv2/tests/test_clis/test_minicli_macros.py::TestExportImportJSON::test_import_conflict_overwrite
toolboxv2/tests/test_clis/test_minicli_macros.py::TestExportImportJSON::test_import_invalid_format_prints_error
toolboxv2/tests/test_clis/test_minicli_macros.py::TestExportImportJSON::test_import_missing_file_prints_error
toolboxv2/tests/test_clis/test_minicli_macros.py::TestExportImportJSON::test_import_valid_json_creates_macros
toolboxv2/tests/test_clis/test_minicli_macros.py::TestMacroArguments::test_args_set_as_arg1_and_dollar1
toolboxv2/tests/test_clis/test_minicli_macros.py::TestMacroArguments::test_missing_macro_prints_error
toolboxv2/tests/test_clis/test_minicli_macros.py::TestMacroManagerRouting::test_create_dispatches
toolboxv2/tests/test_clis/test_minicli_macros.py::TestMacroManagerRouting::test_export_dispatches
toolboxv2/tests/test_clis/test_minicli_macros.py::TestMacroManagerRouting::test_import_dispatches_with_path
toolboxv2/tests/test_clis/test_minicli_macros.py::TestMacroManagerRouting::test_unknown_action_prints_error
toolboxv2/tests/test_clis/test_minicli_macros.py::TestMacroResultCapture::test_return_value_saved_to_quick_vars
toolboxv2/tests/test_clis/test_minicli_macros.py::TestMacroSaveLoad::test_load_empty_blob_leaves_macros_empty
toolboxv2/tests/test_clis/test_minicli_macros.py::TestMacroSaveLoad::test_load_macros_populates_dict
toolboxv2/tests/test_clis/test_minicli_macros.py::TestMacroSaveLoad::test_save_macros_writes_correct_structure
toolboxv2/tests/test_clis/test_minicli_macros.py::TestVariableSubstitution::test_loop_var_highest_priority
toolboxv2/tests/test_clis/test_minicli_macros.py::TestVariableSubstitution::test_macro_var_overrides_quick_var
toolboxv2/tests/test_clis/test_minicli_macros.py::TestVariableSubstitution::test_multiple_vars_in_one_command
toolboxv2/tests/test_clis/test_minicli_macros.py::TestVariableSubstitution::test_substitutes_quick_var
toolboxv2/tests/test_clis/test_minicli_macros.py::TestVariableSubstitution::test_unknown_var_left_as_is
```
