# FIX-32 – aus Karte K-API-12

Paket: **P3-VERALTET** (Test veraltet (Schnittstelle absichtlich geändert)) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 6 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Sollen `.env`-Werte in einfachen Anführungszeichen geschrieben werden (`A='1'`)? Achtung: bestehende Einträge bleiben ohne (`X=0`), neue bekommen welche – gemischtes Format gewollt?

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-32  (aus Karte K-API-12)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_isaa/test_providers/test_paid.py, test_onboarding_v2.py, test_free_providers.py   (nur diese ändern)
Vorher lesen:  `OnboardingResult`, der `.env`-Schreiber
```

**Schritte:**

1. Tests auf die heutigen Felder umstellen.
2. `.env`-Format nach Antwort: Test oder Schreiber angleichen.

**Nicht tun:** – Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 6 IDs grün.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_providers/test_free_providers.py::TestAppendEnv::test_append_only
toolboxv2/tests/test_isaa/test_providers/test_free_providers.py::TestAppendEnv::test_handles_missing_newline
toolboxv2/tests/test_isaa/test_providers/test_onboarding_v2.py::TestScanReturnsEnvNames::test_shared_env_key_not_duplicated
toolboxv2/tests/test_isaa/test_providers/test_paid.py::TestBuildConfigPaid::test_coding_plan_limits
toolboxv2/tests/test_isaa/test_providers/test_paid.py::TestBuildConfigPaid::test_cost_table_from_config
toolboxv2/tests/test_isaa/test_providers/test_paid.py::TestBuildConfigPaid::test_payg_limits_shape
```
