# FIX-06 – aus Karte K-ENV-06

Paket: **P0-SIGNAL** (CI-Signal und Installation) · Fix-Typ: **A** · Karte: UNGETESTET · betrifft 0 Test-IDs

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-06  (aus Karte K-ENV-06)
Voraussetzung: keine
Dateien:       .github/workflows/release.yml   (nur diese ändern)
Vorher lesen:  alle drei Build-Jobs; der Linux-Job ist die Vorlage
```

**Schritte:**

1. In `build-windows-x64` und `build-macos-arm64` nach `setup-python` einen Schritt `Install build deps` einfügen: `python -m pip install --upgrade pip` und `pip install nuitka zstandard`.
2. Den Schritt `Install project` aus dem Linux-Job übernehmen (gleicher Befehl).

**Nicht tun:** Linux-Job nicht ändern. Release-Schritt (`Update nightly prerelease`) nicht ändern, das ist O3. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- `release.yml` manuell auslösen → alle drei Build-Jobs grün, Smoke-Test läuft.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.
