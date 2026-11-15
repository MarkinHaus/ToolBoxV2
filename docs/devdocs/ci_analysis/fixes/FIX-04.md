# FIX-04 – aus Karte K-ENV-03

Paket: **P0-SIGNAL** (CI-Signal und Installation) · Fix-Typ: **B** · Karte: GETESTET · betrifft 8 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Gehört `waitress` in die Basis-Abhängigkeiten (lokale Oberfläche ist Kernfunktion, Empfehlung) oder soll die CI `.[web]` installieren? Prüfen: installiert der SimpleCore-Installer `toolboxv2[web]`? (`test_installer_protocol` erwartet `ToolBoxV2[cli,web,isaa]`, dann ist nur die CI betroffen.)

```
Fix:           FIX-04  (aus Karte K-ENV-03)
Voraussetzung: keine
Dateien:       pyproject.toml (Variante 1) oder .github/workflows/build.yml (Variante 2)   (nur diese ändern)
Vorher lesen:  `pyproject.toml:8-28` (dependencies), `:134-170` (Extras); `fast_tb.py:1065-1125`
```

**Schritte:**

1. Variante 1 (Empfehlung): `"waitress>=3.0.2",` in `[project].dependencies` aufnehmen; im Extra `web` stehen lassen.
2. Variante 2: in `build.yml` Schritt `Install toolboxv2 with ISAA` `.[web]` statt `.` installieren (zusammen mit FIX-02: `-e ".[web]"`).

**Nicht tun:** Kein try/except um den Import, das den Server stumm nicht starten lässt. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- `python -m pytest toolboxv2/tests/test_core/test_wizard_e2e_http.py -q` → `10 passed`.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_core/test_wizard_e2e_http.py::test_browser_onboarding_flow_over_http
toolboxv2/tests/test_core/test_wizard_e2e_http.py::test_e2e_live_health_over_http
toolboxv2/tests/test_core/test_wizard_e2e_http.py::test_e2e_live_redirect_or_disabled
toolboxv2/tests/test_core/test_wizard_e2e_http.py::test_e2e_loopback_guard_blocks_remote_environ
toolboxv2/tests/test_core/test_wizard_e2e_http.py::test_e2e_wizard_profile_over_http
toolboxv2/tests/test_core/test_wizard_e2e_http.py::test_e2e_wizard_state_over_http
toolboxv2/tests/test_core/test_wizard_e2e_http.py::test_root_serves_onboarding_on_first_run
toolboxv2/tests/test_core/test_wizard_e2e_http.py::test_setup_always_serves_onboarding
```
