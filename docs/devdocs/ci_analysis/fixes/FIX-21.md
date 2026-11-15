# FIX-21 – aus Karte K-BUG-03

Paket: **P4-ENTSCHEIDUNG** (Braucht Entscheidung des Owners) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 2 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Welcher Vorgabeport gilt für den Broadcast-Client (6568, 6588 oder der vereinheitlichte 8467)? Soll `host` der Name (`"Local test router"`) oder die IP sein? Wird der Broadcast noch gebraucht?

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-21  (aus Karte K-BUG-03)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/utils/brodcast/client.py; ggf. toolboxv2/tests/test_core/test_brodcast/test_bv_server_client.py   (nur diese ändern)
Vorher lesen:  `client.py` ganz, `server.py::make_known`, die drei `EventManager`-Stellen mit `TOOLBOXV2_BASE_PORT`
```

**Schritte:**

1. Nach Antwort: `int(os.getenv("TOOLBOXV2_BASE_PORT", <Vorgabe>))`.
2. `host`-Erwartung im Test bzw. Wert im Produkt nach Antwort angleichen.
3. macOS-Fehler `No route to host` getrennt prüfen (Broadcast-Adresse).

**Nicht tun:** Test nicht löschen. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- `python -m pytest toolboxv2/tests/test_core/test_brodcast -q` → 2 passed (ubuntu).
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_core/test_brodcast/test_bv_server_client.py::TestBrodcast::test_make_known
toolboxv2/tests/test_core/test_brodcast/test_bv_server_client.py::TestBrodcast::test_start_client
```
