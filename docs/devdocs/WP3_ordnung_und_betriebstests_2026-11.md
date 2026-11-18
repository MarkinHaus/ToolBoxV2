# Arbeitspaket WP3 – Ordnung herstellen: erst Betriebstests, dann Fixes

> Stand 2026-11-17, Basis `master` @ `c36c3dc`. Befunde: Prüfbericht vom 2026-11-17 (Konto, Sync, Ende-zu-Ende, CI).
> Jeder Auftrag aus diesem Paket gilt nur zusammen mit `docs/devdocs/agent_arbeitsprotokoll.md` und `docs/devdocs/testing_guide.md`.
> Arbeitsbranch: legt der Owner fest (keine neuen Branches ohne seine Anweisung). Kein Merge nach `master` ohne Prüfung und Freigabe.

---

## 1. Warum es auseinandergefallen ist

1. **Die Aufträge ließen Spielraum, und der wurde für Umbauten genutzt.**
   - Aufgaben wurden ersetzt statt erfüllt: Blob-Sync und Kontodaten-Abgleich sind in `d672504` entfernt.
   - Es kamen Extras dazu, die niemand bestellt hat: `get_sync_key` mit Kontoschlüssel im Klartext auf dem Server, umbenannte Routen, eine andere Kontoseite.
2. **Die Tests patchten genau die Stellen weg, die im Betrieb kaputt sind.** Es gab keinen einzigen Test gegen das installierte Paket oder den laufenden Worker.
3. **Es wurde direkt nach `master` gemergt**, ohne Prüfung gegen die Abschlusskriterien.
4. **Kleine Reparaturen stapelten sich auf Reparaturen**, zum Beispiel derselbe Import zweimal im `try/except` in `server_worker.py:47-57`, statt die Ursache zu verstehen.

Gegenmaßnahmen in diesem Paket:
1. **Erst Betriebstests, die rot sind:** Sie belegen jeden Befund im echten Ablauf.
2. **Dann je Ursache ein Auftrag mit fester Dateiliste.** Fertig ist er erst, wenn sein Betriebstest grün ist und die Gegenprobe ihn wieder rot macht.
3. **Freigaben:** Neue Server-Funktionen, Schlüssel und das Entfernen von Funktionen nur mit Owner-Freigabe im Auftrag.
4. **Prüfung vor jedem Merge**, Bericht gegen Diff.

## 2. Entscheidungen des Owners (2026-11-17, gelten)

1. **Löschungen** werden zwischen Geräten übertragen.
2. **`agent_sandbox`** ist eine optionale Abhängigkeit. ISAA startet ohne sie; der Sandbox-Pfad meldet sichtbar „nicht verfügbar“.
3. **Kein Kontoschlüssel im Klartext auf dem Server und keine Schlüsselausgabe über eine neue `api=True`-Funktion.** `get_sync_key` wird zurückgebaut. Das Krypto-Verfahren für den Sync legt Abschnitt 6 fest, nach Entscheidung des Owners.
4. Weiter gelten alle Entscheidungen aus WP1, insbesondere: Sync nur über `CloudM.UserDataAPI`, keine MinIO-Daten auf dem Gerät, Kontopflicht ohne Schalter.

Nachtrag 2026-11-18:

5. **Krypto: Weg 2 (Geräte-Kopplung)** aus Abschnitt 6. Nur Funktionen aus `utils/security/cryp.py` und dem vorhandenen `mods/CloudM/LiveSync/crypto.py`.
6. **Blob-Sync gehört weiter zu Ziel 7** (Auftrag 10), über die Data-API. Entscheidung 1 (Löschungen) gilt auch für Blobs.
7. **Agenten pushen direkt auf `master`**; geprüft wird danach (Protokoll Abschnitt 1, Punkt 4).
8. **Kontoseite:** Das WP1-Dokument wird an die Seite angepasst (drei Anmeldeknöpfe, Routen `/api/account/*`). Ausnahme: Das Wort „Optional“ auf der Seite widerspricht der Kontopflicht und fällt weg (Auftrag 5).
9. Nacharbeit aus der Prüfung von `a2cae2b` geht in Auftrag 5 (siehe Abschnitt 9).

## 3. Reihenfolge

| Nr. | Auftrag | wer | blockiert durch |
|---|---|---|---|
| 1 | Signaturschlüssel `TAURI_PRIVATE_KEY` korrigieren (Abschnitt 7) | Owner | – |
| 2 | Zeitlimit in der CI gegen hängende Tests | Agent | – |
| 3 | Betriebstests Konto und Sicherheit anlegen (rot) | Agent | – |
| 4 | Betriebstests Paket und Sync anlegen (rot) | Agent | – |
| 5 | Eine Prüfung „nur lokal“, Kontopflicht an einer Stelle, WS-Origin | Agent | 3 |
| 6 | Konto im Betrieb: Datenordner, Gerätevertrauen, `base`, Reihenfolge in `link_account` | Agent | 3 |
| 7 | Paket: LiveSync aufnehmen, Multipart-Abhängigkeit | Agent | 4 |
| 8 | `get_sync_key` zurückbauen, Krypto-Verfahren nach Abschnitt 6 | Agent | Entscheidung zu Abschnitt 6 |
| 9 | Sync-Korrekturen: Konfliktkopie, Symlinks, 401, Löschungen, Dateinamen mit `:` | Agent | 4, 8 |
| 10 | Blob-Sync über die Data-API (falls weiter Ziel) | Agent | Entscheidung |
| 11 | CI-Reste: Modul-Stub `test_stubs.py`, `await` im Health-Check-Test, `agent_sandbox` optional | Agent | – |
| 12 | Windows-Reste: drei unwirksame Fixes, Dateisperre `device_owner.json` | Agent (Windows) | – |
| 13 | Kartenrunde veraltete Tests (rund 117 IDs) | Owner + Agent | Antworten je Karte |
| 14 | Kleinere Befunde (unten) | Agent | 5–9 |

Die Aufträge 3/4, 5/6/7 und 11/12 können parallel laufen. Ihre Dateilisten sind getrennt; die Aufträge legen sie fest.

## 4. Die Betriebstests (Aufträge 3 und 4)

**Neue Testdatei je Bereich**, `unittest`. Alle laufen in einer isolierten Umgebung (Protokoll, Abschnitt 3.1) gegen echte Teile.
- Der Worker läuft auf einem freien Port (Muster `test_wizard_e2e_http.py`).
- Für die Paket-Tests wird das gebaute Wheel in eine frische venv installiert.
- Doubles nur für `simplecore.app`, mit echten Antwortformen.

**Jeder Test muss auf `c36c3dc` rot sein**, mit der Fehlermeldung des Befunds. Der Auftrag liefert diesen roten Lauf als Beleg und **ändert keinen Produktcode**.

### Auftrag 3 – Konto und Sicherheit

| Nr. | Test | erwartet (Soll) | Befund auf `c36c3dc` |
|---|---|---|---|
| 1 | Ohne Gerätekonto: `GET /api/wizard/state`, `POST /api/wizard/profile`, `POST /api/wizard/save`, `/api/chats`, `/tray/command`, `/isaa`, `/obs`, WS 8468 | abgelehnt bzw. Kontoseite | nur `/` prüft; Profil `server` setzen öffnet das Dashboard |
| 2 | Anfrage an eine lokale Route mit fremder Gegenstelle und `Host: 127.0.0.1` | 403 | durchgelassen (`local_ui.py:35-43` prüft nur den Host-Header) |
| 3 | WS-Handshake auf den lokalen WS mit `Origin: http://evil.example` | abgelehnt | angenommen |
| 4 | Echte App, Arbeitsordner ≠ Datenordner, `device_owner.json` im Datenordner, Profil `local` | `onboarding_state()["needs_account"] is False`, `/` liefert das Dashboard | `needs_account: true` (belegt: `account_state(app)` meldet „linked“ und `onboarding_state()` meldet `needs_account: true`) |
| 5 | Worker läuft; danach Konto verknüpfen bzw. Kontodatei entfernen; **ohne Neustart** eine Anfrage | Vertrauen an bzw. aus | erst nach Neustart (`server_worker.py:1682-1690`) |
| 6 | Ohne Konto bzw. ohne Profil `local`/`developer` | kein Thread `account-sync` | Thread startet immer (`server_worker.py:1692-1710`, falsche Einrückung) |
| 7 | `link_account` mit fehlschlagendem `login()` | keine Tokens gespeichert, `session.base` unverändert | Tokens und `base` vorher gesetzt (`device_account.py:84-87`) |
| 8 | Konto-Sync nach dem Laden von `CloudM` | `session.base` ist gesetzt | `CloudM/module.py:142` setzt `None` → „insecure base“ |

### Auftrag 4 – Paket und Sync

| Nr. | Test | erwartet (Soll) | Befund auf `c36c3dc` |
|---|---|---|---|
| 1 | Wheel bauen, frische venv, `python -I` aus fremdem Ordner: `import toolboxv2.mods.CloudM.LiveSync.data_api_sync`; Worker starten, Log ohne `ModuleNotFoundError` | importierbar, Thread läuft | `ModuleNotFoundError` (`pyproject.toml:74, 95`) |
| 2 | Datei-Upload (multipart) an eine vorhandene Upload-Route des Workers | Datei kommt an | Uploads abgeschaltet: `server_worker.py:47-57` importiert `parse_form_data` aus `multipart`, deklariert ist `python-multipart` (hat diese Namen nicht) |
| 3 | Zwei Geräte (zwei Datenordner, gemeinsamer Fake der Data-API): Konflikt im Konto-Spiegel, dann zweiter `sync_all` | lokaler Wert bleibt erhalten | Konfliktkopie wird gelöscht (`account_bridge.py:67-79`) |
| 4 | Ordner-Symlink im globalen ISAA-Ordner, der nach außen zeigt | nichts außerhalb geschrieben, nichts von außen hochgeladen | außerhalb geschrieben |
| 5 | Datei auf A löschen, `sync_all` A, `sync_all` B | auf B gelöscht | bleibt auf B |
| 6 | Data-API antwortet 401 bei gecachtem Schlüssel | Zustand `reauth_required` | `offline` |
| 7 | Konto-Sync: Was liegt auf dem Server? | kein Kontoschlüssel im Klartext im Nutzerdatensatz | `user_pass_sync` im Klartext (`auth/api_session.py:172-195`) |

## 5. Die Fix-Aufträge (5–9)

Jeder Fix-Auftrag:
- nennt seinen Betriebstest aus Abschnitt 4, der grün werden muss;
- nennt seine feste Dateiliste;
- hält das Protokoll ein, alle Schritte 1–11.

**Hypothesen aus dem Prüfbericht sind Startpunkte, keine Wahrheit:** Der Agent bestätigt sie selbst (Schritt 4).

- **Auftrag 5 – Eine Lokal-Prüfung, eine Kontopflicht.**
  - Es gibt heute drei Loopback-Prüfungen. Die Prüfung auf Ebene der TCP-Gegenstelle (`live_proxy._peer_is_local`) wird die einzige; die beiden anderen werden entfernt. Diese Entfernung ist hiermit freigegeben.
  - Die Kontopflicht wird an **einer** Stelle durchgesetzt, dort, wo der lokale Worker Anfragen verteilt. Keine Prüfung pro Route, kein Wrapper pro Route. Welche Stelle das ist, findet der Agent durch Lesen und belegt es; der Planer prüft die Wahl vor dem Edit.
  - Zusätzlich die Origin-Prüfung für WS 8468.
- **Auftrag 6 – Konto im Betrieb.**
  - `onboarding_state` liest die Kontodatei über die App bzw. `_state_dir.data_root`, nie relativ.
  - Gerätevertrauen wird bei Änderung der Kontodatei neu bewertet; keine zweite Quelle.
  - Einrückung des Sync-Threads korrigieren.
  - `module.py:142` setzt `base` nicht auf `None`.
  - `link_account` schreibt erst nach erfolgreichem `login()`.
- **Auftrag 7 – Paket.**
  - LiveSync in das Paket aufnehmen.
  - Eine Multipart-Abhängigkeit, die zum Code passt: Der Code benutzt die API des PyPI-Pakets `multipart`. `python-multipart` bleibt nur, wenn es noch woanders gebraucht wird; das belegt der Agent mit `git grep`.
  - Den doppelten Import entfernen.
  - Abnahme nur mit dem Wheel (Regel 8).
- **Auftrag 8 – Krypto zurück auf die Owner-Linie.** `get_sync_key` entfernen und den Sync nicht mehr über `user_pass_sync` laufen lassen (freigegeben). **Korrektur 2026-11-18:** Das **Feld** `user_pass_sync` bleibt: Es ist älter als WP1 und der alte Magic-Link-Weg leitet daraus seinen Schlüssel ab (`AuthHelpers.py:108`, `extras.py:343`). Folge: Jeder vom Server ausgegebene Sync-Schlüssel war zugleich das Ausgangsmaterial dieses Links; der Auftrag prüft, ob der Weg noch erreichbar ist, und der Owner erneuert danach die betroffenen Werte auf dem Server. Das Verfahren nach der Entscheidung aus Abschnitt 6 mit den **vorhandenen** Funktionen aus `toolboxv2/utils/security/cryp.py`; keine neue Krypto-Bibliothek, keine neue `api=True`-Funktion.
- **Auftrag 9 – Sync-Korrekturen** nach den Betriebstests aus Auftrag 4 (Nr. 3–6), dazu Dateinamen mit `:` sichtbar behandeln statt still überspringen.

## 6. Entscheidung: Wie wird der Sync verschlüsselt?

Ausgangslage: `Code.DK()` liefert den **Geräteschlüssel**. Er entsteht pro Gerät (`device.enc`, abhängig von `TB_R_KEY`). Damit kann ein zweites Gerät nichts entschlüsseln, was das erste verschlüsselt hat. Der Geräteschlüssel allein reicht also nicht für den Abgleich zwischen Geräten, wohl aber, um einen gemeinsamen Schlüssel **lokal** sicher abzulegen.

| Weg | Wie | Server sieht Klartext? | Neues | Aufwand |
|---|---|---|---|---|
| 1 – keine Ende-zu-Ende-Verschlüsselung | Gerät überträgt per TLS an die Data-API; der Server speichert wie alle Nutzerdaten | ja | nichts (entspricht WP1-Entscheidung 7 wörtlich) | klein |
| 2 – Geräte-Kopplung mit vorhandener Krypto | Das erste Gerät erzeugt einen Kontoschlüssel (`Code.generate_symmetric_key`) und legt ihn lokal mit dem Geräteschlüssel verschlüsselt ab. Jedes Gerät hat ein Schlüsselpaar (`Code.generate_asymmetric_keys`); sein öffentlicher Schlüssel ist ein normaler Datensatz über die Data-API. Ein vorhandenes Gerät legt den Kontoschlüssel mit `Code.encrypt_asymmetric` für das neue Gerät ab, das neue entschlüsselt mit seinem privaten Schlüssel. Nutzdaten: `Code.encrypt_symmetric` | nein, auch nicht den Kontoschlüssel | ein Kopplungsschritt (ein vorhandenes Gerät muss das neue einmal bestätigen); keine neue Server-Funktion | mittel |
| 3 – Passphrase | Kontoschlüssel aus einer Nutzer-Passphrase abgeleitet | nein | Passphrase-Eingabe; Passphrase vergessen = Daten verloren | mittel |

**Empfehlung:**
- Weg 2, wenn die Sync-Daten vor dem Betreiber geschützt sein sollen. Er benutzt nur vorhandene Funktionen und keine neue Server-Funktion.
- Sonst Weg 1, das ist der einfachste und entspricht der ursprünglichen Entscheidung.

In beiden Fällen entfällt `get_sync_key`.

## 7. Signaturschlüssel für den Updater (Auftrag 1, Owner)

Ursache: Alle drei Tauri-Builds bündeln erfolgreich und scheitern beim Signieren mit `Invalid symbol 10, offset 97.`. Zeichen 10 ist ein Zeilenumbruch: Im Secret `TAURI_PRIVATE_KEY` steckt nach 97 Zeichen ein Umbruch. Der Schlüssel muss eine einzige Zeile sein.

Wo was hingehört:

| Was | Wo |
|---|---|
| privater Schlüssel (Inhalt der `.key`-Datei, eine Zeile) | GitHub → Repo → Settings → Secrets and variables → Actions → `TAURI_PRIVATE_KEY` |
| Passwort des Schlüssels | ebenda → `TAURI_KEY_PASSWORD` |
| öffentlicher Schlüssel (Inhalt der `.pub`-Datei, nicht der Pfad) | `toolboxv2/simple-core/src-tauri/tauri.conf.json` → `plugins.updater.pubkey` |
| Weitergabe an den Build | schon vorhanden: `.github/workflows/build.yml:518-519` setzt daraus `TAURI_SIGNING_PRIVATE_KEY` und `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` |
| die `.key`-Datei selbst | nur auf deinem Rechner und in deinem Passwort-Manager, nie im Repo |

**Fall 1 – du hast die alte `.key`-Datei noch.** Zuerst prüfen, ob sie zum Schlüssel in `tauri.conf.json` passt: Der Inhalt der zugehörigen `.pub`-Datei muss genau dem Wert `plugins.updater.pubkey` entsprechen. Dann nur das Secret neu setzen (siehe Befehle unten).

**Fall 2 – die Datei fehlt oder passt nicht.** Neu erzeugen und den öffentlichen Schlüssel eintragen. Das geht jetzt noch ohne Folgen: Wegen des Signierfehlers wurde nie ein signiertes Update veröffentlicht, also verlässt sich keine installierte App auf den alten Schlüssel. Ein Setup, das du schon aus einem PR-Artefakt installiert hast, enthält den alten öffentlichen Schlüssel und nimmt keine Updates mit dem neuen an: deinstallieren und das neue Setup aus `nightly` installieren.

```powershell
cd C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\simple-core
npm install                 # einmal, holt @tauri-apps/cli aus package.json
npx tauri signer generate -w "$HOME\.tauri\simplecore.key"
# fragt nach einem Passwort; erzeugt simplecore.key und simplecore.key.pub
```

Danach den Inhalt von `simplecore.key.pub` als `plugins.updater.pubkey` in `tauri.conf.json` eintragen (eine Zeile, in Anführungszeichen) und als eigenen Commit pushen.

**Secret setzen, ohne dass ein Umbruch hineinkommt** (mit der GitHub-CLI; `Trim()` entfernt einen Umbruch am Ende):

```powershell
$k = (Get-Content -Raw "$HOME\.tauri\simplecore.key").Trim()
$k -match '\s'          # muss False sein – sonst steckt ein Umbruch mitten im Schlüssel
$k.Length               # nur die Länge notieren, nie den Inhalt
gh secret set TAURI_PRIVATE_KEY --repo MarkinHaus/ToolBoxV2 --body $k
gh secret set TAURI_KEY_PASSWORD --repo MarkinHaus/ToolBoxV2   # fragt interaktiv, nichts in der History
Remove-Variable k
```

Ohne `gh`: Datei in einem Editor öffnen, alles markieren, kopieren und in GitHub einfügen. Vorher prüfen, dass die Datei genau eine Zeile hat, also kein Zeilenumbruch beim Kopieren aus einer Konsole mit Zeilenumbruch.

**Abnahme:**
1. `build.yml` manuell starten (Actions → Build & Test → Run workflow, Branch `master`). Die drei Tauri-Jobs werden grün.
2. Das Release `nightly` enthält die Installer, `latest.json` und je eine `.sig`-Datei.
3. Erst danach den Laptop-Test (Setup aus `nightly`, `scripts/verify_install_windows.ps1`) und für den Updater eine zweite Veröffentlichung.

Quellen: [Tauri v2 Updater](https://v2.tauri.app/plugin/updater/), [CrabNebula: Auto-Updates für Tauri](https://docs.crabnebula.dev/guides/auto-updates-tauri).

## 8. Kleinere Befunde (Auftrag 14)

1. `installer.sh --yes` ohne `--config` bricht mit `tb_features_cfg: unbound variable` ab.
2. `tb --version` startet die ganze App und zeigt die Version nur in der Modulliste (`RESimpleToolBox: 0.1.29`). Für Skripte und den Updater sollte eine Zeile `0.1.29` ohne App-Start kommen.
3. Die App versucht beim Start `dl.min.io` und `localhost:9000` zu erreichen, auch mit `IS_OFFLINE_DB=true`.
4. `tb-manifest.yaml` meldet `app.version 0.1.27` bei Paket 0.1.29; der Installer wählt Python 3.10.
5. Die Kontoseite weicht vom WP1-Dokument ab (Text „Optional“, drei Knöpfe, Routennamen). Entweder Seite oder Dokument anpassen; der Owner entscheidet.
6. `OSError: Bad file descriptor` (waitress) beim Stop.

## 9. Prüfung von `master` @ `a2cae2b`/`dca1ffa` (2026-11-18)

Erledigt und im echten Ablauf bestätigt: Auftrag 2 (Zeitlimits), 3 und 4 (Betriebstests angelegt), 6 (Konto im Betrieb), 7 (Paket). Testlauf: 8 rot (= Aufträge 5 und 9), 52 grün. Wheel frisch installiert, `python -I` aus fremdem Ordner: LiveSync importierbar, `MULTIPART_AVAILABLE True`. Echte App, fremder Arbeitsordner, `onboarding_state()` ohne `app`: `needs_account false`.

Nacharbeit (geht in Auftrag 5):
1. Vertrauen nach dem Verknüpfen ohne Neustart: Der Prüf-Thread startet nur, wenn beim Start schon ein Konto da war (`server_worker.py:1684`, 1707, 1732). Beim Erststart über den Wizard bleibt das Vertrauen bis zum Neustart aus. Nach dem Entfernen des Kontos läuft `account-sync` weiter.
2. A3 Nr. 3, 5, 6 lesen Quelltext statt auszuführen (Testleitfaden, Fälle 8–10). Neu als Tests am laufenden Worker.
3. A3 Nr. 1 prüft nur `/` und `/api/wizard/state`, nicht alle Routen aus Abschnitt 4.
4. A3 Nr. 4 ruft mit Fake-App statt ohne `app` wie der Betrieb.
5. `live_proxy._peer_is_local` fällt ohne `REMOTE_ADDR` auf `client_ip` zurück, das aus `X-Forwarded-For` stammen kann (Kommentar dort, `server_worker` L308).

Startprompts: `docs/devdocs/wp3_prompts/` (Aufträge 5, 8, 9, 10). Wellen: 5 und 9 parallel; danach 8; danach 10.
