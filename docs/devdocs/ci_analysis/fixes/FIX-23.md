# FIX-23 – aus Karte K-API-18

Paket: **P4-ENTSCHEIDUNG** (Braucht Entscheidung des Owners) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 1 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Hat das Dockerfile noch Feature-Build-Argumente (`FEATURE_*`), oder ist es bewusst ein Vollabbild?

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-23  (aus Karte K-API-18)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       Dockerfile oder toolboxv2/tests/test_core/test_docker_build.py   (nur diese ändern)
Vorher lesen:  `FEATURE_MAP` im Test, Dockerfile
```

**Schritte:**

1. Nach Antwort: fehlende `ARG FEATURE_…` ergänzen oder den Test auf das Vollabbild umstellen.

**Nicht tun:** – Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 1 ID grün.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_core/test_docker_build.py::TestDockerfileStructure::test_all_feature_build_args_declared
```
