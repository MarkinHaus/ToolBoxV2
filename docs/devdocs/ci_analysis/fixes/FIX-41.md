# FIX-41 – aus Karte K-PY-02

Paket: **P6-PLATTFORM** (Plattform- und versionsabhängig) · Fix-Typ: **A** · Karte: UNGETESTET · betrifft 2 Test-IDs

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-41  (aus Karte K-PY-02)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_isaa/test_providers/test_performance_and_limiter.py, toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_performance_stability.py   (nur diese ändern)
Vorher lesen:  die beiden Testmethoden
```

**Schritte:**

1. `async with asyncio.timeout(x): await c` → `await asyncio.wait_for(c, timeout=x)`.
2. Zweite ID unter 3.10 einzeln laufen lassen, Traceback lesen, Ursache in die Karte nachtragen, dann beheben.

**Nicht tun:** 3.10 nicht aus der Matrix nehmen (Owner-Entscheidung, s. README Fragen). Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- beide IDs grün unter 3.10.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_performance_stability.py::TestAgentPerformance::test_vfs_mount_blocking_analysis
toolboxv2/tests/test_isaa/test_providers/test_performance_and_limiter.py::TestAgentPerformance::test_rate_limiter_blocks_correctly
```
