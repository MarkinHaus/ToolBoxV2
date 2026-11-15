# FIX-33 – aus Karte K-API-20

Paket: **P3-VERALTET** (Test veraltet (Schnittstelle absichtlich geändert)) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 3 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Sollen die Browser-Abläufe in pytest laufen (dann Fixture `tf` bauen; braucht Playwright und einen Server → eher eigener CI-Job) oder sind es Hilfsfunktionen (dann umbenennen, z. B. `run_login_…`)?

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-33  (aus Karte K-API-20)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_web/test_web.py   (nur diese ändern)
Vorher lesen:  Datei ganz
```

**Schritte:**

1. Hilfsfunktionen: Präfix `test_` entfernen (Aufrufer in derselben Datei anpassen).
2. Oder: `@pytest.fixture async def tf()` anlegen und die Abläufe als eigene Marke (`@pytest.mark.e2e`) führen.

**Nicht tun:** Kein `skip`. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- keine ERROR mehr in der Datei.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_web/test_web.py::test_api_endpoint
toolboxv2/tests/test_web/test_web.py::test_login_page
toolboxv2/tests/test_web/test_web.py::test_main_content_authenticated
```
