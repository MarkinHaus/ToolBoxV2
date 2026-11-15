# FIX-10 – aus Karte K-PY-01

Paket: **P1-AUTO-MINI** (Minimaler Fix, automatisch) · Fix-Typ: **A** · Karte: GETESTET · betrifft 28 Test-IDs

```
Fix:           FIX-10  (aus Karte K-PY-01)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_clis/test_config_wizard.py   (nur diese ändern)
Vorher lesen:  Zeilen 30–65 (Bootstrap)
```

**Schritte:**

1. Direkt nach `_spec.loader.exec_module(_wiz)` einfügen:

```python
import toolboxv2.utils.clis as _clis_pkg
_clis_pkg.config_wizard = _wiz  # py3.10: mock.patch resolves via getattr on the parent package
```

**Nicht tun:** Bootstrap nicht umbauen, keine Tests anfassen. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- `python3.10 -m pytest toolboxv2/tests/test_clis/test_config_wizard.py -q` → 0 failed; unter 3.12 ebenfalls.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_clis/test_config_wizard.py::TestEnvFileIO::test_set_vars_not_prompted_again
toolboxv2/tests/test_clis/test_config_wizard.py::TestEnvFileIO::test_wizard_env_variables_does_not_mutate_existing_dict
toolboxv2/tests/test_clis/test_config_wizard.py::TestFullWizardPipeline::test_app_name_preserved_end_to_end
toolboxv2/tests/test_clis/test_config_wizard.py::TestFullWizardPipeline::test_database_mode_lc_end_to_end
toolboxv2/tests/test_clis/test_config_wizard.py::TestFullWizardPipeline::test_isaa_none_when_declined_end_to_end
toolboxv2/tests/test_clis/test_config_wizard.py::TestFullWizardPipeline::test_produces_valid_tbmanifest
toolboxv2/tests/test_clis/test_config_wizard.py::TestFullWizardPipeline::test_save_load_after_full_pipeline
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardAppSchemaIntegration::test_app_name_stored_in_schema
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardAppSchemaIntegration::test_debug_is_bool_in_schema
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardAppSchemaIntegration::test_log_level_is_loglevel_enum
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardAppSchemaIntegration::test_model_validate_succeeds
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardDatabaseSchemaIntegration::test_CB_produces_valid_manifest
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardDatabaseSchemaIntegration::test_LC_mode_is_databasemode_enum
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardDatabaseSchemaIntegration::test_LC_produces_valid_manifest
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardDatabaseSchemaIntegration::test_LR_mode_is_databasemode_enum
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardDatabaseSchemaIntegration::test_RR_produces_valid_manifest
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardDatabaseSchemaIntegration::test_local_path_preserved_in_schema
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardDatabaseSchemaIntegration::test_minio_use_ssl_is_bool_in_schema
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardDatabaseSchemaIntegration::test_redis_db_index_is_int_in_schema
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardIsaaSchemaIntegration::test_a2a_enabled_is_bool_in_schema
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardIsaaSchemaIntegration::test_isaa_config_is_isaaconfig_instance
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardIsaaSchemaIntegration::test_isaa_enabled_flag_true
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardIsaaSchemaIntegration::test_isaa_none_when_declined
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardIsaaSchemaIntegration::test_max_tokens_is_int_in_schema
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardIsaaSchemaIntegration::test_mcp_enabled_is_bool_in_schema
toolboxv2/tests/test_clis/test_config_wizard.py::TestWizardIsaaSchemaIntegration::test_temperature_is_float_in_schema
toolboxv2/tests/test_clis/test_config_wizard.py::TestWorkersSchemaCompatibility::test_wizard_workers_raises_typeerror_on_schema_dump
toolboxv2/tests/test_clis/test_config_wizard.py::TestWorkersSchemaCompatibility::test_wizard_workers_works_with_empty_workers_dict
```
