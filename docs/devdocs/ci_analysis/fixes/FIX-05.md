# FIX-05 – aus Karte K-ENV-07

Paket: **P0-SIGNAL** (CI-Signal und Installation) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 0 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Owner: Secret `TAURI_PRIVATE_KEY` neu eintragen (eine Zeile, kein Umbruch) und `build.yml` manuell auslösen. Liegt danach `latest.json` signiert im Release `nightly`?

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-05  (aus Karte K-ENV-07)
Voraussetzung: keine
Dateien:       keine Datei im Repo (GitHub-Secret)   (nur diese ändern)
Vorher lesen:  `build.yml:404-530` (Reset nightly, Tauri-Jobs)
```

**Schritte:**

1. Owner: Repository → Settings → Secrets → `TAURI_PRIVATE_KEY` neu setzen. Wert: der Inhalt der von `tauri signer generate` erzeugten Schlüsseldatei, als eine einzige Zeile.
2. `TAURI_KEY_PASSWORD` prüfen (kein Umbruch).
3. `build.yml` manuell auslösen.

**Nicht tun:** Schlüssel nicht in eine Datei im Repo oder in ein Dokument schreiben. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- Lauf auf `master`: drei Tauri-Jobs grün; Release `nightly` enthält Installer und `latest.json`.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.
