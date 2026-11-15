# FIX-25 – aus Karte K-API-06

Paket: **P4-ENTSCHEIDUNG** (Braucht Entscheidung des Owners) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 3 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Sind 262 144 als Vorgabe für unbekannte Modelle und 2 000 000 für das geprüfte Modell gewollt?

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-25  (aus Karte K-API-06)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_isaa/test_base/test_llm_router.py (oder die Modelltabelle im Router)   (nur diese ändern)
Vorher lesen:  `toolboxv2/mods/isaa/base/llm_router/` – wo `ctx_limit` herkommt
```

**Schritte:**

1. Nach Antwort: Erwartungswerte im Test oder Werte in der Tabelle anpassen.

**Nicht tun:** Nicht beides ändern. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 3 IDs grün.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_base/test_llm_router.py::TestModelInfo::test_ctx_limit_known
toolboxv2/tests/test_isaa/test_base/test_llm_router.py::TestModelInfo::test_ctx_limit_unknown_default
toolboxv2/tests/test_isaa/test_base/test_llm_router.py::TestModelInfo::test_ctx_limit_wildcard
```
