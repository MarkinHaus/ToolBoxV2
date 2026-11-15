# FIX-16 – aus Karte K-API-17

Paket: **P1-AUTO-MINI** (Minimaler Fix, automatisch) · Fix-Typ: **A** · Karte: GETESTET · betrifft 1 Test-IDs

```
Fix:           FIX-16  (aus Karte K-API-17)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_core/test_tray_daemon_integration.py   (nur diese ändern)
Vorher lesen:  `test_fallback_tray_fetches_state`
```

**Schritte:**

1. Zeile 461: `"_fetch_tray_state"` → `"fetch_tray_state"`.

**Nicht tun:** Die zweite Prüfung (`"/tray/state"`) bleibt. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- `42 passed`.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_core/test_tray_daemon_integration.py::TestDaemonServiceE2EValidation::test_fallback_tray_fetches_state
```
