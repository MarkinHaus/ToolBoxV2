# FIX-01 – aus Karte K-ENV-01

Paket: **P0-SIGNAL** (CI-Signal und Installation) · Fix-Typ: **A** · Karte: GETESTET · betrifft 0 Test-IDs

```
Fix:           FIX-01  (aus Karte K-ENV-01)
Voraussetzung: keine
Dateien:       .github/workflows/build.yml   (nur diese ändern)
Vorher lesen:  Job `py-check`, Schritt `Run Tests` (Zeilen 148–154)
```

**Schritte:**

1. Zeile 154 ersetzen durch `run: python -m pytest toolboxv2/tests/ -v --tb=short -rfE`.
2. Nichts anderes im Job ändern (FIX-02 ändert die Installation getrennt).

**Nicht tun:** Kein `continue-on-error`, kein `|| true`, keine Matrix-Einträge entfernen. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- CI-Lauf: alle 12 `py-check`-Jobs enden mit dem pytest-Exit-Code. Erwartung direkt nach diesem Fix: alle 12 rot (ehrliches Signal), Fehlerzahl je Job wie in README Tabelle 1.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.
