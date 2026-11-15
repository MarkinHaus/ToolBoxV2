# Karte K-ENV-06

```
Karte:       K-ENV-06
Paket:       P0-SIGNAL – CI-Signal und Installation
Status:      UNGETESTET (Workflow-Datei, in der Analysephase nicht änderbar; kein Runner für Windows/macOS verfügbar)
Fix-Typ:     A
Fix-Auftrag: FIX-06
```

**Ursache:** `release.yml`: die Jobs `build-windows-x64` und `build-macos-arm64` rufen `python -m nuitka` auf, installieren Nuitka aber nicht (O4).

**Betroffen:** 0 Test-IDs; Job `build-windows-x64` rot (Lauf 43742117826: `No module named nuitka`), `build-macos-arm64` hat denselben Aufbau.

**Beleg:** `.github/workflows/release.yml:100-110` (Windows) und `:126-137` (macOS): nach `setup-python` folgt direkt `python -m nuitka …`. Der Linux-Job hat `pip install nuitka zstandard` (Zeile 55) und einen Schritt `Install project`.

**Hypothese:** Kopierfehler beim Anlegen der beiden Jobs.

**Repro:**

```
nur in der CI (Workflow-Datei)
```

**Vorher:** `No module named nuitka`

**Probe-Fix:** in beiden Jobs vor dem Build: `python -m pip install --upgrade pip`, `pip install nuitka zstandard`, Projekt installieren wie im Linux-Job

**Nachher:** –

**Nachbarn:** –
