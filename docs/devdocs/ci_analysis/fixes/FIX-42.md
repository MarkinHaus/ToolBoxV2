# FIX-42 – aus Karte K-STATE-03

Paket: **P6-PLATTFORM** (Plattform- und versionsabhängig) · Fix-Typ: **A** · Karte: UNGETESTET · betrifft 1 Test-IDs

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-42  (aus Karte K-STATE-03)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_mini/test_system/test_state_system.py (und der Test, der die Variable hinterlässt)   (nur diese ändern)
Vorher lesen:  `toolboxv2/utils/system/state_system.py` (`get_state_from_app`), woher `…/toolboxv2/features` kommt
```

**Schritte:**

1. Im Test vor dem Aufruf ausgeben, welche Variable auf `/root` zeigt (`HOME`, `TB_DATA_DIR`, `app.start_dir`).
2. Den Test, der sie setzt, mit `patch.dict`/`addCleanup` abschirmen (Muster wie FIX zu K-STATE-01).

**Nicht tun:** Nicht `/root` beschreibbar machen, kein skip. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- CI ubuntu: ID grün.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_mini/test_system/test_state_system.py::TestTbState::test_get_state_from_app
```
