# FIX-15 – aus Karte K-MISS-03

Paket: **P1-AUTO-MINI** (Minimaler Fix, automatisch) · Fix-Typ: **A** · Karte: GETESTET · betrifft 1 Test-IDs

```
Fix:           FIX-15  (aus Karte K-MISS-03)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_core/test_port_unification.py   (nur diese ändern)
Vorher lesen:  `test_embedded_js_fallback_matches_canon`
```

**Schritte:**

1. Zeile 45: `parents[3]` → `parents[2]`.

**Nicht tun:** Die Assertions (`'8100'` nicht enthalten, `'8468'` enthalten) bleiben. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- `python -m pytest toolboxv2/tests/test_core/test_port_unification.py -q` → `7 passed`.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_core/test_port_unification.py::test_embedded_js_fallback_matches_canon
```
