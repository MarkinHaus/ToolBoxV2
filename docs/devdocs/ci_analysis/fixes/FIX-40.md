# FIX-40 – aus Karte K-OS-05

Paket: **P6-PLATTFORM** (Plattform- und versionsabhängig) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 3 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Sollen Tests echte Pakete installieren (dann eigener, getrennter CI-Job) oder die Paketmanager mocken? Im normalen Lauf verändern sie das System des Entwicklers.

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-40  (aus Karte K-OS-05)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_mini/test_extras/test_depmanager.py   (nur diese ändern)
Vorher lesen:  Datei ganz; welche Tests schon übersprungen werden, wenn der Manager fehlt
```

**Schritte:**

1. Nach Antwort: echte Installations-Tests in einen eigenen Job mit Marke (`@pytest.mark.system`) verschieben, oder Aufrufe mocken.

**Nicht tun:** – Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 3 IDs grün bzw. im getrennten Job.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_mini/test_extras/test_depmanager.py::TestInstallUninstall::test_cycle_choco
toolboxv2/tests/test_mini/test_extras/test_depmanager.py::TestList::test_list_winget
toolboxv2/tests/test_mini/test_extras/test_depmanager.py::TestSearch::test_search_winget
```
