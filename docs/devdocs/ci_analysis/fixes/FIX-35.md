# FIX-35 – aus Karte K-OS-01

Paket: **P6-PLATTFORM** (Plattform- und versionsabhängig) · Fix-Typ: **A** · Karte: UNGETESTET · betrifft 14 Test-IDs

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-35  (aus Karte K-OS-01)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       erst Analyse; voraussichtlich installer.ps1 / toolboxv2/simple-core/src-tauri/…/installer.ps1 oder die Tests   (nur diese ändern)
Vorher lesen:  `toolboxv2/tests/test_core/test_installer_protocol.py` (wie das Skript gestartet wird), `installer.ps1` (Protokoll auf fd 3/stdout), Handoff R8/R9
```

**Schritte:**

1. Auf dem Windows-Rechner des Owners: Befehl aus `Repro` ausführen. Sind die Tests dort grün, liegt es am Runner (PowerShell-Version, fehlendes `uv`, Ausführungsrichtlinie) → stderr des Skripts im Test mit ausgeben und einen CI-Lauf ansehen.
2. Sind sie dort rot: stdout/stderr des Skripts ansehen, erste Zeile, Exit-Code.
3. Karte mit dem Befund ergänzen (Status GETESTET), dann beheben.

**Nicht tun:** Keine `skipIf(win32)`. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 14 IDs grün in allen 4 Windows-Jobs.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_core/test_installer_protocol.py::TestCoreUpdateInPlace::test_exit_ok_and_done_event
toolboxv2/tests/test_core/test_installer_protocol.py::TestCoreUpdateInPlace::test_manifest_records_new_build_and_keeps_the_rest
toolboxv2/tests/test_core/test_installer_protocol.py::TestCoreUpdateInPlace::test_profile_and_env_untouched
toolboxv2/tests/test_core/test_installer_protocol.py::TestCoreUpdateInPlace::test_wheel_reinstalled_with_the_installed_features
toolboxv2/tests/test_core/test_installer_protocol.py::TestInstallerProtocol::test_bundled_uv_sidecar_is_used
toolboxv2/tests/test_core/test_installer_protocol.py::TestInstallerProtocol::test_ends_with_done
toolboxv2/tests/test_core/test_installer_protocol.py::TestInstallerProtocol::test_exit_ok_and_config_path_parsed
toolboxv2/tests/test_core/test_installer_protocol.py::TestInstallerProtocol::test_fail_emits_error_event
toolboxv2/tests/test_core/test_installer_protocol.py::TestInstallerProtocol::test_progress_is_fraction_and_monotonic_enough
toolboxv2/tests/test_core/test_installer_protocol.py::TestInstallerProtocol::test_stdout_is_only_json_with_type
toolboxv2/tests/test_core/test_installer_protocol.py::TestInstallerProtocol::test_tauri_bundle_copy_in_sync
toolboxv2/tests/test_core/test_installer_protocol.py::TestWheelAndFeaturesDryRun::test_dry_run_names_pkg_with_extras_and_file_url
toolboxv2/tests/test_core/test_installer_protocol.py::TestWheelAndFeaturesDryRun::test_exit_ok_and_json_only
toolboxv2/tests/test_core/test_verify_install_windows.py::TestVerifyInstallWindows::test_all_v1_to_v7_pass_and_exit_nonzero_due_v8
```
