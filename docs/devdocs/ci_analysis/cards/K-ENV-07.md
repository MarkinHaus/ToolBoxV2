# Karte K-ENV-07

```
Karte:       K-ENV-07
Paket:       P0-SIGNAL – CI-Signal und Installation
Status:      UNGETESTET (Secret ist nur dem Owner zugänglich)
Fix-Typ:     B (Owner: Secret `TAURI_PRIVATE_KEY` neu eintragen (eine Zeile, kein Umbruch) und `build.yml` manuell auslösen. Liegt danach `latest.json` signiert im Release `nightly`?)
Fix-Auftrag: FIX-05
```

**Ursache:** Auf `master` schlägt `tauri build` beim Signieren fehl: das Secret `TAURI_PRIVATE_KEY` lässt sich nicht als Base64 lesen.

**Betroffen:** 0 Test-IDs; alle drei Jobs `Tauri Build (…)` des Laufs 43742117802 auf `ebed4d6` rot. Im PR-Lauf grün, weil dort nicht signiert wird.

**Beleg:** Log `Tauri Build (ubuntu-22.04)`: `Finished 3 bundles at: …` / `Invalid symbol 10, offset 97.` / `Error Invalid symbol 10, offset 97.` / `Command "npx ["tauri","build"]" failed with exit code 1`. `build.yml:518-519` setzt `TAURI_SIGNING_PRIVATE_KEY` nur bei `PUBLISH == 'true'`.

**Hypothese:** Zeichen 10 ist ein Zeilenumbruch. Der gespeicherte Schlüssel enthält ab Position 97 einen Umbruch (zweizeilig eingefügt, oder Schlüsseldatei statt Base64-Zeile). Folge: Der Job `Reset nightly release` hat das Release `nightly` schon gelöscht, die Tauri-Jobs laden nichts hoch. O1 (Setup aus `nightly`) und O2 (Updater) sind damit blockiert, bis das Secret stimmt.

**Repro:**

```
nur in der CI mit dem echten Secret
```

**Vorher:** s. Beleg

**Probe-Fix:** Secret neu setzen: Inhalt der `.key`-Datei als eine Zeile ohne Umbruch am Ende

**Nachher:** –

**Nachbarn:** –
