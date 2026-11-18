# ISAA-Auftrag WP3-05 – Eine Lokal-Prüfung, eine Kontopflicht, WS-Origin, Vertrauen ohne Neustart

> Basis: `master` @ `dca1ffa` (geprüft 2026-11-18). Welle 1, läuft parallel zu WP3-09 (getrennte Dateien).
> Abnahme: `toolboxv2/tests/test_core/test_a3_betriebsbefunde.py` komplett grün, mit den hier verlangten neuen Fassungen von Nr. 1, 3, 4, 5, 6.

## 0. Rahmen (gleich in allen WP3-Startprompts)

**Du bist eine ISAA-Instanz und arbeitest diesen Auftrag selbstständig ab.** Verbindlich und vor dem ersten Edit vollständig zu lesen (liegen auf `master`):

1. `docs/devdocs/agent_arbeitsprotokoll.md` – Rahmen, die 11 Pflichtschritte je Änderung, isolierte Umgebung, Bericht.
2. `docs/devdocs/testing_guide.md` – Anti-Muster und Regeln. Besonders: Doubles nur an der Prozessgrenze (Regel 2), Eingang wie im Betrieb (Regel 3), **kein Quelltext als Testobjekt** (kein `inspect.getsource`, keine Wortsuche), Gegenprobe (Regel 7), Wheel-Test bei Paket-/Importänderungen (Regel 8).
3. `docs/devdocs/WP3_ordnung_und_betriebstests_2026-11.md` – Entscheidungen (Abschnitt 2), Befunde (Abschnitte 4, 9).

Kurzfassung der harten Regeln (die Dateien gehen vor):

- **Nur deine Dateien** (Abschnitt 1 unten). Alles andere, was dir auffällt: Bericht, „Gefunden, nicht geändert“.
- **Keine Dopplung:** Vor jeder neuen Funktion `git grep -n` nach dem **Zweck**; Fund in den Bericht. Kein Wrapper, kein Fallback-Zweig, keine zweite Prüfung neben einer vorhandenen.
- **Hypothese vor Edit** (Protokoll Schritte 3–4), **isolierter Lauf mit vollen Abhängigkeiten nach jedem Edit** (Schritt 7, Protokoll Abschnitt 3), **Gegenprobe** mit `git stash` (Schritt 8).
- **Kein Grün durch Wegsehen:** kein `skip`/`xfail`/`importorskip`, keine abgeschwächte Assertion, kein Test gelöscht, außer der Auftrag nennt ihn ausdrücklich zum Ersetzen.
- **Owner-Umgebung schützen:** Ports 8467/8468 nicht belegen, wenn die Owner-Instanz läuft (dann fragen, nicht ausweichen). Daten nur unter `$T` (Protokoll 3.1). Keine Geheimnisse in Dateien, Logs, Commits, Bericht.
- **Stoppen und melden** bei: Hypothese widerlegt und keine neue belegbar; Datei außerhalb der Liste nötig; Code sieht anders aus als hier beschrieben; dritter Reparaturversuch am selben Fehler.

### 0.1 Arbeitskopie (einmal)

```powershell
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2
git fetch origin
git worktree add -B isaa/05 C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa05 origin/master
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa05
uv venv -p 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install pytest pytest-asyncio pytest-timeout build
uv pip install -e toolboxv2/mods/isaa
uv pip install -e ".[web]"
$env:MOCK_EXTERNAL_SERVICES = "true"; $env:IS_OFFLINE_DB = "true"
python -c "import toolboxv2; print(toolboxv2.__file__)"   # muss in ...Toolboxv2-isaa05 liegen
```

Nichts von Hand in die venv installieren, was nicht in `pyproject.toml` steht. Sonst misst die Gegenprobe deine Umgebung statt des Codes (Testleitfaden, Fall 12).

### 0.2 Gate

Skript `C:\Users\Markin\Workspace\ISAABOX\tb_gate.py` (liegt aus WP2 bereits dort; Inhalt sonst: `git show origin/claude/keen-goldberg-o59mt9:docs/devdocs/ci_analysis/prompts/ISAA_06_session_per_user.md`, Abschnitt 0.4). Gate-Bereich: siehe Abschnitt 1 unten.

```powershell
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py baseline C:\Users\Markin\Workspace\ISAABOX\isaa05_baseline.json @GATE   # vor dem ersten Edit
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py check    C:\Users\Markin\Workspace\ISAABOX\isaa05_baseline.json @GATE   # nach jedem Edit, muss "NEU rot: 0"
```

### 0.3 Commit und Push (Owner-Entscheidung: direkt auf `master`)

Erst wenn alle Pflichtfelder des Protokolls belegt sind und der Bericht geschrieben ist. Ein Commit je Teilaufgabe; Dateien einzeln hinzufügen (`git add <pfad>`), nie `git add -A`/`.`/`-a`. Nachricht: erste Zeile `WP3 A05.<n>: <kurz>`, darunter Hypothese (ein Satz) und die Schlusszeile des Abnahmelaufs.

```powershell
git fetch origin
git rebase origin/master
# nach dem Rebase: Gate-Check erneut
git push origin HEAD:master
```

Abgelehnt (`non-fast-forward`): höchstens dreimal wiederholen, dann melden. Kein `--force`, kein Merge-Commit.

### 0.4 Bericht

`C:\Users\Markin\Workspace\ISAABOX\wp3_A05_bericht.md` (außerhalb des Repos), Format nach Protokoll Abschnitt 4: je Teilaufgabe die Tabelle mit den 11 Pflichtfeldern, Zustand, „Gefunden, nicht geändert“, Patch-Inventar der Tests. Denselben Text als Antwort ausgeben.


## 1. Deine Dateien

- `toolboxv2/utils/workers/fast/live_proxy.py` (nur `_is_loopback`, `_peer_is_local`, `_ensure_local`)
- `toolboxv2/utils/workers/fast/local_ui.py`
- `toolboxv2/utils/workers/fast/wizard_api.py` (nur `_loopback_guard` und die Guard-Aufrufe)
- `toolboxv2/utils/workers/fast/account_page.py` (nur das Wort „Optional.“ im Hinweistext)
- **eine** Dispatch-Stelle für die Kontopflicht (Teilaufgabe 2; welche, belegst du)
- `toolboxv2/utils/workers/ws_worker.py` (nur Handshake: `_process_request_new_api` :265, `_process_request_legacy` :283)
- `toolboxv2/utils/workers/server_worker.py` (nur der Block „Device trust“ ab ca. Zeile 1676 bis `Device trust off`)
- `toolboxv2/tests/test_core/test_a3_betriebsbefunde.py`

Gate-Bereich:
```powershell
$GATE = @('toolboxv2/tests/test_core/test_a3_betriebsbefunde.py','toolboxv2/tests/test_core/test_wizard_e2e_http.py','toolboxv2/tests/test_core/test_device_account.py','toolboxv2/tests/test_core/test_account_link_flow.py','toolboxv2/tests/test_core/test_first_run_onboarding.py','toolboxv2/tests/test_core/test_state_dir.py')
```

**Nicht anfassen:** `mods/CloudM/LiveSync/*` (WP3-09), `mods/CloudM/auth/*`, `pyproject.toml`, `docs/`.

## 2. Teilaufgaben in dieser Reihenfolge

### 2.1 Zuerst die Tests ehrlich machen (rot auf `dca1ffa`)

Die vorhandenen Fassungen von Nr. 3, 5 und 6 lesen Quelltext (`inspect.getsource`). Das ist verboten (Testleitfaden, Fälle 8–10). Nr. 4 ruft mit Fake-App statt wie der Betrieb ohne `app`. Nr. 1 prüft nur zwei Routen. **Ersetze** diese fünf Tests (freigegeben) durch Tests, die ausführen:

| Nr. | neuer Test (Muster: Klasse `_Server` in derselben Datei bzw. `test_wizard_e2e_http.py`) | muss auf `dca1ffa` |
|---|---|---|
| 1 | Profil `local`, **kein** `device_owner.json`. Per HTTP gegen den echten Worker auf freiem Port: `GET /` → Kontoseite (nicht Dashboard); `GET /api/wizard/state` → `needs_account: true`; `POST /api/wizard/profile` mit `server` und `POST /api/wizard/save` → abgelehnt (Teilaufgabe 2.2); jede weitere lokal registrierte Route, die nicht in der Ausnahmeliste (2.2) steht, → 403 bzw. Kontoseite. Die Routenliste **aus der App lesen** (`app` der FastTB-Instanz, alle registrierten Routen), nicht abschreiben. | rot |
| 3 | Echter WS-Worker auf freiem Port (wie `ws_worker` startet), Handshake mit `Origin: http://evil.example` → abgelehnt (HTTP 403 im Upgrade); mit `Origin: http://127.0.0.1:<port>` → angenommen. Echte Client-Bibliothek `websockets`. | rot |
| 4 | Echte App (`get_app`) mit `TB_DATA_DIR` in `$T`, `device_owner.json` im `app.data_dir`, Profil `local`, Arbeitsordner = ein **anderer** Temp-Ordner; Aufruf **ohne** `app` über `local_ui.get_installation_state()` → `needs_account is False`. Singleton nach dem Test zurücksetzen (Leitfaden Regel 6.4). | grün (Fix ist da) – Gegenprobe mit `git stash` auf A6 nicht möglich, also: einmal gegen `c36c3dc` laufen lassen und die rote Ausgabe in den Bericht |
| 5 | Worker starten **ohne** `device_owner.json` (Profil `local`). Danach Datei schreiben. Ohne Neustart innerhalb von 10 s: eine Anfrage von 127.0.0.1 ist vertraut (gleiche Prüfung wie WP1-K2: `GET /api_user_data` bzw. das, was `enable_local_trust` steuert – belegen). Danach Datei löschen → innerhalb von 10 s nicht mehr vertraut. | rot (An-Richtung) |
| 6 | Worker ohne Konto starten → in `threading.enumerate()` kein Thread `account-sync`. Konto anlegen → Thread da (≤ 10 s). Konto entfernen → Thread beendet (≤ 10 s nach seinem nächsten Takt; Takt für den Test über eine vorhandene Konstante oder einen Parameter, kein `time.sleep`-Patch). | rot (Beenden fehlt) |

Für 5 und 6: Worker im Testprozess wie in `test_a3`/`test_wizard_e2e_http` starten; wenn `HTTPWorker` dort nicht startbar ist, im Unterprozess mit isoliertem `TB_DATA_DIR` und Logdatei. Doubles nur für `simplecore.app` (Regel 2).

Beleg im Bericht: alle fünf rot auf `dca1ffa` (wörtliche Fehlermeldung), Commit `WP3 A05.1: Betriebstests Nr. 1,3,4,5,6 ausfuehrend statt Quelltext`. **Dieser Commit enthält keinen Produktcode** und darf rot gepusht werden (Ausnahme vom Gate nur für diese Test-IDs; im Bericht nennen).

### 2.2 Eine Lokal-Prüfung

Heute drei Prüfungen: `local_ui._ensure_local` (nur Host-Header, `local_ui.py:35-43`), `wizard_api._loopback_guard` (ruft die erste), `live_proxy._ensure_local`/`_peer_is_local` (Gegenstelle). Freigegeben: **`live_proxy._peer_is_local` wird die einzige**, die beiden anderen werden entfernt, alle Aufrufer rufen die eine.

Zusätzlich behebst du die Lücke in `_peer_is_local` (`live_proxy.py:116-125`): Fehlt `REMOTE_ADDR`, fällt sie auf `client_ip` zurück, das laut eigenem Kommentar aus `X-Forwarded-For` stammen kann. Hypothese zuerst belegen: Setzt der echte Worker `environ["REMOTE_ADDR"]` für jede Anfrage? (Schritt 4 mit echtem Worker.) Wenn ja: Rückfall entfernen, fehlendes `REMOTE_ADDR` = nicht lokal. Wenn nein: stoppen und melden.

Test Nr. 2 (vorhanden, prüft die Funktion direkt) bleibt; ergänze einen HTTP-Fall: Worker auf `0.0.0.0` und freiem Port, Anfrage über die eigene Nicht-Loopback-Adresse des Rechners (aus `socket.gethostbyname(socket.gethostname())`, falls das 127.x liefert: erste Adresse aus `socket.getaddrinfo`) mit `Host: 127.0.0.1` und `X-Forwarded-For: 127.0.0.1` → 403. Gibt es keine Nicht-Loopback-Adresse, steht das im Bericht; der Test meldet das dann als Fehler mit klarer Meldung, nicht als `skip`.

### 2.3 Kontopflicht an einer Stelle

Soll: Profil `local`/`developer` und kein Gerätekonto → **jede** lokale Route außer der Ausnahmeliste liefert für HTML eine Weiterleitung auf `/account`, sonst 403 `{"error": "account required"}`.

Ausnahmeliste (vollständig, nicht erweitern ohne Rückfrage):
- statische Assets für Kontoseite und Login (`/web/`, `/dist`-Mount, `/favicon.ico`), `/health`;
- `/`, `/account`, `/setup` (zeigen selbst die richtige Seite);
- `GET /api/wizard/state`, `GET /api/account/status`, `POST /api/account/start`;
- **während `first_run`** (noch kein Profil): zusätzlich `POST /api/wizard/profile` und `POST /api/wizard/save`. Ist ein Profil gesetzt und fehlt das Konto, sind sie gesperrt; damit lässt sich die Kontopflicht nicht mehr über ein Profil `server` umgehen. Profilwechsel geht dann über `tb -init` in der Konsole.

Stelle: Lies `fast_tb_handler.FastTBHandler.handle_request` (:241, dort steht schon das Auth-Gate „variant iii: global“) und finde heraus, ob **alle** Routen des lokalen Workers (`local_ui.app`, gemountete Apps `/isaa`, `/obs`, `tray_api`, `wizard_api`) durch genau diese Methode laufen. Belege das mit dem echten Worker: z. B. Logzeile oder Breakpoint-Ausgabe für jede Route aus Test Nr. 1. Ist es diese Methode: dort, neben dem vorhandenen Gate, **eine** Prüfung, die nur für die lokale App gilt (Erkennung über ein Attribut der FastTB-Instanz, das `local_ui` setzt – vorhanden ist z. B. `serve_login_assets`; ein neues Attribut wie `require_device_account = True` ist freigegeben). Zustand über `first_run.onboarding_state()`, keine zweite Quelle. Läuft eine Route **nicht** dort durch: stoppen und mit Beleg melden.

Danach: Die Kontopflicht-Zweige in den einzelnen Routen (`local_ui.root` :114 ff.) bleiben nur, soweit sie die richtige Seite **zeigen**; sie prüfen nichts Zusätzliches.

`account_page.py`: das Wort „Optional.“ im Hinweistext entfernen (Owner-Entscheidung 8, WP3 Abschnitt 2). Sonst nichts an der Seite.

### 2.4 WS-Origin

Im Handshake beider API-Varianten (`_process_request_new_api`, `_process_request_legacy`): Origin vorhanden und nicht `http://127.0.0.1:<port>`, `http://localhost:<port>` oder ein Tauri-Origin → 403. Welche Origins die Tauri-App schickt, belegst du aus `toolboxv2/simple-core/src-tauri/tauri.conf.json` und dem Frontend-Code (Datei:Zeile im Bericht); nicht raten. Fehlender Origin (Nicht-Browser-Client, z. B. CLI) bleibt erlaubt – prüfe vorher mit `git grep`, welche Clients 8468 benutzen, und nenne sie.

### 2.5 Vertrauen und Sync-Thread ohne Neustart

Heute (`server_worker.py:1684-1736`): Der Prüf-Thread `trust-revalidate` und `account-sync` starten nur, wenn **beim Start** ein Konto da war. Soll:
- `trust-revalidate` läuft bei Profil `local`/`developer` **immer** (einmal je Prozess, Namensprüfung wie bei `account-sync`), unabhängig davon, ob beim Start ein Konto da war.
- Er ist die **einzige** Stelle, die Vertrauen und `account-sync` an- und abschaltet. Beim Wechsel auf „Konto da“ startet er `account-sync` (falls nicht laufend), beim Wechsel auf „kein Konto“ signalisiert er ihm das Ende (`threading.Event`), `account-sync` prüft das Event statt `time.sleep(600)` (`event.wait(600)`).
- Kein zweiter Code-Pfad für den Start: Die Startlogik beim Boot ruft dieselbe Funktion wie der Prüf-Thread beim Wechsel.

Abnahme: Nr. 5 und Nr. 6 grün; Gegenprobe; isolierter Lauf nach Protokoll 3.3 mit `tb workers start` **nur**, wenn 8467/8468 frei sind.

## 3. Fertig, wenn

`test_a3_betriebsbefunde.py` 8/8 grün (mit den neuen Fassungen), Gate „NEU rot: 0“, Gegenprobe je Teilaufgabe im Bericht, Commits gepusht.
