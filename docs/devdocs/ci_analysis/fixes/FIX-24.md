# FIX-24 – aus Karte K-API-19

Paket: **P4-ENTSCHEIDUNG** (Braucht Entscheidung des Owners) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 1 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Soll jede HTML-Zeichenkette in die Seitenhülle eingebettet werden, auch ein Fragment? (Für HTMX-Teilantworten wäre das falsch.)

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-24  (aus Karte K-API-19)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/utils/workers/fast_tb.py oder der Test   (nur diese ändern)
Vorher lesen:  `format_html`/Antwort-Formatierung in `fast_tb.py`
```

**Schritte:**

1. Nach Antwort: Test auf „enthält `<h1>Hi</h1>`“ umstellen, oder Fragment-Antworten ohne Hülle zulassen.

**Nicht tun:** – Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 1 ID grün.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_web/test_workers/test_fast/test_fast_tb.py::TestHandlerResponseFormatting::test_format_html_string
```
