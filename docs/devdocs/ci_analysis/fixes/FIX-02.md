# FIX-02 – aus Karte K-ENV-04

Paket: **P0-SIGNAL** (CI-Signal und Installation) · Fix-Typ: **A** · Karte: GETESTET · betrifft 1 Test-IDs

```
Fix:           FIX-02  (aus Karte K-ENV-04)
Voraussetzung: keine
Dateien:       .github/workflows/build.yml   (nur diese ändern)
Vorher lesen:  Job `py-check`, Schritt `Install toolboxv2 with ISAA` (Zeilen 143–147)
```

**Schritte:**

1. Zeile 147: `python -m uv pip install -e . --system` (statt `.`).
2. Zeile 146 unverändert lassen.

**Nicht tun:** Nicht `PYTHONPATH` im Workflow setzen. Test nicht ändern. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- CI ubuntu/3.12: `test_shell_tool.py::TestKillProcess::test_killing_the_harness_grandparent_does_not_kill_the_harness` grün.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_shell_tool.py::TestKillProcess::test_killing_the_harness_grandparent_does_not_kill_the_harness
```
