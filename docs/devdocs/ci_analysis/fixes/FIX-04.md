# FIX-04 – aus Karte K-ENV-03

Paket: **P0-SIGNAL** (CI-Signal und Installation) · Fix-Typ: **A** · Karte: GETESTET · betrifft 8 Test-IDs

```
Fix:           FIX-04  (aus Karte K-ENV-03)
Voraussetzung: FIX-02 (ändert dieselbe Zeile vorher)
Dateien:       .github/workflows/build.yml   (nur diese ändern)
Vorher lesen:  Job `py-check`, Schritt `Install toolboxv2 with ISAA` (Zeilen 143–147)
```

**Schritte:**

1. Nach FIX-02 lautet Zeile 147 `python -m uv pip install -e . --system`. Ändern in:

```yaml
          python -m uv pip install -e ".[web]" --system
```

**Nicht tun:** `pyproject.toml` nicht ändern (Owner-Entscheidung). Keine weiteren Extras hinzufügen. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- CI ubuntu/3.12: alle 8 IDs aus `test_core/test_wizard_e2e_http.py` grün.
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
