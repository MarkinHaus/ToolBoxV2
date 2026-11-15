# FIX-22 – aus Karte K-API-16

Paket: **P4-ENTSCHEIDUNG** (Braucht Entscheidung des Owners) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 2 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Ist `workers` jetzt `infrastructure`, und sind die auskommentierten `core`-Dienste endgültig weg?

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-22  (aus Karte K-API-16)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_clis/test_cli_service_manager.py   (nur diese ändern)
Vorher lesen:  `ServiceRegistry.__init__`
```

**Schritte:**

1. Nach Antwort: Erwartungen an die heutige Einteilung anpassen (Kategorie von `workers`, Mindestzahl `core`).

**Nicht tun:** – Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 2 IDs grün.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_clis/test_cli_service_manager.py::TestServiceManagerStatus::test_get_service_info
toolboxv2/tests/test_clis/test_cli_service_manager.py::TestServiceRegistry::test_registry_get_by_category
```
