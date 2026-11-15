# Arbeitspaket WP1 — Konto und Nutzung: Kontopflicht, Offline-Anmeldung, gezielter Sync

> Stand: 2026-11-15, geprüft gegen `master` @ `ebed4d6` (Merge von #14, R1–R9).
> Ersetzt die Abschnitte K und S aus `taskmap_stable_tauri_2026-11.md` und aus den Regeln v3. Die alten Entwürfe mit Kontoschlüssel und MinIO-Zugangsdaten auf dem Gerät sind **verworfen**.
> Läuft parallel zu WP2 (CI-Fehler). WP1 ändert Code, WP2 schreibt in der Analysephase nur Dokumente — so kommen sich die beiden nicht in die Quere.

---

## 1. Ziel

| # | Abnahmepunkt |
|---|---|
| Z3 | Beim ersten Start muss sich der Nutzer mit seinem SimpleCore-Konto anmelden. Danach ist er auf dem Gerät automatisch angemeldet, **auch ohne Netz**. |
| Z3a | Lokale Daten, die vor dem Verknüpfen schon da waren (lokaler Admin `root`), gehen in das Konto über. |
| Z8 | Zweites Gerät, gleiches Konto: Kontodaten, Blobs und der globale ISAA-Ordner sind da; Änderungen wandern in beide Richtungen. |
| Z8a | Übertragen wird nur, was sich geändert hat: Pfad für Pfad, Schlüssel für Schlüssel, Feld für Feld. Nie „alles als ein JSON". |

## 2. Entscheidungen des Owners (gelten, nicht neu verhandeln)

1. Kontopflicht beim ersten Start; danach automatische Anmeldung auf dem Gerät. Es gibt keinen Schalter, der die Pflicht umgeht.
2. **Sync ausschließlich über die vorhandene Data-API** `CloudM.UserDataAPI` (`get_data`, `set_data`, `list_data`, `get_mod_data`, `set_mod_data`) über HTTPS zu `simplecore.app`. **Kein neuer Server-Endpunkt, kein Kontoschlüssel, keine MinIO-Zugangsdaten auf dem Gerät.**
3. MinIO bleibt auf dem Server intern. `get_secure_storage_credentials` und `BlobStorage.sync()` mit `_cloud_minio` werden vom Gerät **nicht** benutzt.
4. Sync-Umfang: Kontodaten (private Daten und Mod-Daten), Blobs, globaler ISAA-Ordner. ISAA-Chats nicht.
5. Gezielt und dynamisch: zuerst `list_data` (Pfad, Größe, `updated_at`) vergleichen, dann nur geänderte Pfade einzeln holen bzw. setzen. Mod-Daten pro `key`, beim Setzen nur die geänderten Felder mit `merge=True`.
6. Neue lokale Routen und API-Funktionen im TB-Stil (`@export(mod_name=Name, api=True, version=version, request_as_kwarg=True)` bzw. die vorhandenen FastTB-Routen wie in `wizard_api.py`). Keine neuen Techniken.
7. Live-Tests gegen `simplecore.app` erst, **nachdem der Owner den Stand dort ausgerollt hat** (D1).

## 3. Grundregeln (aus v3, gelten unverändert)

- **Keine Dopplungen.** Vor jeder neuen Funktion mit `git grep` nach dem Zweck suchen. Was es gibt, wird benutzt.
- **Vor dem Bearbeiten:** die Datei ganz lesen, alle Aufrufer der geänderten Stellen lesen, den Ist-Zustand einmal messen.
- **FERTIG** heißt: Kriterium auf dem Rechner ausgeführt und beobachtet, Befehl und echte Ausgabe im Bericht, Commit gepusht. Sonst NICHT VERIFIZIERT / BLOCKIERT / OFFEN.
- Verboten: Tests überspringen oder abschwächen, Werte erfinden (URLs, Feldnamen, Antwortformen), Auth- oder Sicherheitsentscheidungen selbst treffen, mehrere Tasks in einem Commit.
- Keine Geheimnisse in Dateien, Logs oder Commits (Tokens, Refresh-Tokens, Schlüssel). `device_owner.json` enthält **kein** Token.
- **Schutz der Entwicklungsumgebung:** Der lokale Agent arbeitet nur unter `C:\Users\Markin\Workspace\ISAABOX\Toolboxv2`. Er löscht keine Daten der Entwicklungsinstanz des Owners und beendet keine fremden Prozesse. Echte Installationen nur isoliert (umgebogene `LOCALAPPDATA`/`APPDATA`/`USERPROFILE`, wie in R9).
- Branch: `claude/wizardly-cannon-o944j4` (neu ab `master` @ `ebed4d6`). Nie auf `master` pushen.
- Standard-Gate wie in v3 §2.5, zusätzlich `toolboxv2.tests.test_core.test_data_root` und `toolboxv2.tests.test_core.test_env_single_file`.

## 4. Gemessener Stand auf `master` @ `ebed4d6`

| # | Befund | Stelle | Folge |
|---|---|---|---|
| F8 | `SessionManager.enable_local_trust` wird aufgerufen, ist aber **nirgends definiert**. Der Aufruf landet im `except` → Log `Loopback trust not enabled`. | `utils/workers/server_worker.py:1682–1700` (Aufruf :1696) | K2 |
| F8a | Der Live-UI-Owner (FastTB, 8467) baut auf `HTTPWorker` auf und nutzt dessen `SessionManager` (`fast_tb_handler.py:644`, Session-Ermittlung :723–738). Eine Änderung in `utils/workers/session.py` wirkt also auch auf 8467. | `utils/workers/fast_tb_handler.py` | K2 trifft den richtigen Prozess |
| F16 | Lokale Anmeldung und Konto schließen sich aus: mit `TOOLBOXV2_REMOTE_BASE` kein lokaler Admin, ohne sie nur `root`. | `mods/CloudM/module.py:95–123` | K1, K2 |
| F17 | Anmeldung vom Gerät existiert: `cli_remote_login()` öffnet die Login-Seite und erhält `token`, `refresh_token`, `user_id`, `username` über `127.0.0.1:8765` (`TB_OAUTH_CALLBACK_PORT`). | `utils/clis/cli_oauth.py:65`, :78 | K3 benutzt das |
| F17a | Tokens speichern: vorhandener Weg `s._save_session_token(...)` + `await s.login()`. | `utils/clis/local_cli.py:262` (auch :198, :228) | K1 benutzt das |
| F22 | Globaler ISAA-Ordner: `<data_dir>/Agents/VFS/global`. | `mods/isaa/base/patch/power_vfs.py:110` | S3 |
| F23 | `list_data` gibt den Scope inzwischen richtig an den Speicher weiter (behoben in `5636d82`). **Es fehlt nur ein Test**, der das festhält. | `mods/CloudM/UserDataAPI.py:744–795` | S1 |
| F24 | Blob-Methoden: `create_blob` :697, `read_blob` :746, `update_blob` :809, `list_blobs` :923. | `utils/extras/blobs.py` | S2 |
| F25 | Remote-Aufruf eines Moduls: `App.run_http(...)`. | `utils/toolbox.py:2218` | S1–S3 |
| F26 | Nutzer laden/speichern: `_save_user` :15, `_load_user` :30, `_find_user_by_email` :45; `LOCAL_ADMIN_EMAIL` in `auth/config.py:80`. | `mods/CloudM/auth/user_store.py` | K1 |
| F27 | Session aus Anfrage: `get_session_from_request` :685 (anonymer Rückfall :735), `get_session_from_request_sync` :738 (anonymer Rückfall :783). | `utils/workers/session.py` | K2 |
| F28 | Onboarding-Zustand `onboarding_state()`; Wizard-Zustand-Route; Loopback-Prüfung; Route `/`. | `utils/clis/first_run.py:35`, `utils/workers/fast/wizard_api.py:98`, `utils/workers/fast/local_ui.py:36`, :114 | K3 |

Zeilennummern gelten für `ebed4d6`. Vor dem Ändern neu nachsehen.

## 5. Reihenfolge

```
K1 → K2 → K3 → S1 → S2 → S3 → T10 → (D1 Owner) → K0 → K4 → S4
```

Begründung: Ohne Gerätekonto (K1) gibt es keine automatische Anmeldung (K2) und keine Kontoseite (K3). Der Sync (S1–S3) braucht ein verknüpftes Konto. Live-Messung und Abnahme (K0, K4, S4) brauchen den ausgerollten Server-Stand (D1).

---

## 6. Tasks

### K1 — Gerätekonto: verknüpfen, merken, Altdaten übernehmen

- **Neue Datei:** `toolboxv2/mods/CloudM/auth/device_account.py` mit `device_owner_file(app)`, `read_device_owner(app) -> Optional[dict]`, `async link_account(app, payload) -> UserData`, `async unlink_account(app)`.
- **Merker** `<app.data_dir>/device_owner.json` mit genau `user_id`, `username`, `base_url`, `linked_at`, `migrated_from`. Kein Token, kein Schlüssel.
- **Tokens** nur über den vorhandenen Weg (F17a). `app.session.base` vorher auf `TOOLBOXV2_REMOTE_BASE`, Default `https://simplecore.app`. `login()` schlägt fehl → `RuntimeError`, nichts schreiben.
- **Nutzer** nur über `user_store.py` (F26), DB nur über `auth/db_helpers.py`. Lokales Level `-1`; dieses Level geht nie an den Server.
- **Altdaten:** Gibt es den lokalen Admin (`LOCAL_ADMIN_EMAIL`) mit anderer ID: `settings` (ohne `anonymous`, `local_admin`) und `mod_data` übernehmen, **Konto gewinnt bei Konflikt**. Danach alten Datensatz und E-Mail-Index löschen, `migrated_from` setzen.
- Gleiches Konto erneut → keine Änderung. Anderes Konto, während eines verknüpft ist → `RuntimeError("device already linked")`.
- **Abschlusskriterium:** `toolboxv2/tests/test_core/test_device_account.py` (unittest, DB-Helfer per `mock.patch` durch ein Dict ersetzt, `app.session` gemockt, Temp-`data_dir`):

| Fall | Erwartet |
|---|---|
| Verknüpfen ohne lokalen Admin | Datei existiert, Nutzer mit Level `-1` |
| Lokaler Admin mit `settings={"theme":"dark"}`, `mod_data={"m":{"a":1}}` | beides im neuen Nutzer; alter Datensatz und Index gelöscht; `migrated_from` gesetzt |
| Konto `theme: light`, Admin `theme: dark` | bleibt `light` |
| Zweiter Aufruf, gleiches Konto | keine Änderung, kein Fehler |
| Zweiter Aufruf, anderes Konto | `RuntimeError` |
| `login()` → `False` | `RuntimeError`, keine Datei |
| `device_owner.json` lesen | kein Feld enthält `token` |

### K2 — Automatische Anmeldung über das Gerätekonto (auch offline)

- **`utils/workers/session.py`:** `SessionManager.enable_local_trust(provider)` definieren (`provider`: Funktion ohne Argumente → `{"user_id","username","level"}` oder `None`) und `_local_trust_session(environ)`. Eingebunden in **beide** Funktionen aus F27, direkt vor dem anonymen Rückfall.
- **Alle** Bedingungen müssen gelten, sonst anonym: Provider liefert Konto mit `user_id`; `REMOTE_ADDR` ist `127.0.0.1` oder `::1`; kein `X-Forwarded-For`, `X-Real-Ip`, `Forwarded`; Host aus `HTTP_HOST` (ohne Port) ist `127.0.0.1`, `localhost` oder `[::1]`.
- **`server_worker.py:1682–1700`** ersetzen: nur bei Profil `local` oder `developer`; Provider liest `read_device_owner(self._app)` (Ergebnis darf 5 s gecacht werden). `ensure_local_admin` wird hier nicht mehr aufgerufen. Profil `server` oder kein Profil → kein Provider.
- **`mods/CloudM/module.py:95–123`:** Gibt es ein Gerätekonto, `app.session` daraus setzen (`username`, `user_id`, `valid=True`), `base` nicht auf `None`. Sonst bisheriges Verhalten (lokaler Admin für die CLI).
- **Offline:** Die lokale Anmeldung hängt nur an `device_owner.json`, nicht an einem gültigen Token. Ist das Token abgelaufen und der Refresh scheitert (kein Netz), bleibt der Nutzer lokal angemeldet. Nur der Sync meldet `"error": "offline"` bzw. `"reauth_required"`. Ein abgelaufenes Token führt **nie** zur Abmeldung.
- **Nicht tun:** keine Anmeldung für Nicht-Loopback-Anfragen; CSRF- und CORS-Regeln nicht anfassen.
- **Abschlusskriterium:**
  1. `toolboxv2/tests/test_core/test_local_trust.py` mit den Fällen: Konto + Loopback + Host `127.0.0.1:8467` → Session des Kontos (`anonymous False`, `validated True`); `REMOTE_ADDR=192.168.1.5` → anonym; Loopback + `X-Forwarded-For` → anonym; Loopback + `HTTP_HOST=evil.example` → anonym; Provider `None` → anonym; kein Provider → anonym.
  2. Isolierte echte Installation (wie R9) mit von Hand angelegtem `device_owner.json` und per `_save_user` angelegtem Testnutzer: `curl http://127.0.0.1:8467/api_user_data` → dieser Nutzer. Im Worker-Log steht kein `Loopback trust not enabled`. Über die LAN-IP → nicht angemeldet. Netz getrennt, Workers neu → weiter angemeldet.

### K3 — Konto ist Pflicht: Zustand, API, Seite

- `onboarding_state()` (F28) bekommt `needs_account` (Profil `local`/`developer` und kein Gerätekonto) und `account` (`username` oder `None`). `first_run` bleibt, wie es ist.
- `wizard_api.py`: `GET /api/wizard/state` gibt beide Felder mit aus. Neu `POST /api/wizard/account/start`: startet **einen** Hintergrund-Thread mit `cli_remote_login()` (F17), danach `link_account`. Neu `GET /api/wizard/account/status` → `{"state": "idle"|"waiting"|"linked"|"error", "error", "account"}`. Beide nur für Loopback, mit derselben Prüfung wie `local_ui.py:36`, im Stil der vorhandenen Wizard-Routen.
- Kontoseite als String-Konstante in `utils/workers/fast/account_page.py` (eigenständiges HTML, keine externen Skripte oder Stylesheets): Knopf „Mit SimpleCore anmelden", Statuszeile, alle 2 s Status abfragen; bei `linked` → `location.replace('/')`.
- `local_ui.py`: `GET /account`; Route `/`: zuerst `first_run` → Setup, sonst `needs_account` → Kontoseite, sonst Dashboard.
- Bestehende Tests, die nach gesetztem Profil das Dashboard erwarten, legen im `setUp` ein `device_owner.json` an. Das ist die neue Voraussetzung, keine Abschwächung.
- **Abschlusskriterium:** `test_wizard_api` und `test_wizard_e2e_http` grün, jeweils mit und ohne `device_owner.json`. Isolierte echte Installation ohne Gerätekonto: `/api/wizard/state` → `"needs_account":true`; `/` → Kontoseite. **Achtung:** `test_wizard_e2e_http` ist in der CI unter 3.12/3.13 heute rot (8 Fälle, WP2). Vor K3 die Ursache aus WP2 abfragen bzw. einmal selbst lokal reproduzieren und den Befund im Bericht nennen.

### S1 — Kontodaten gezielt über die Data-API abgleichen

- **Neue Datei:** `toolboxv2/mods/CloudM/auth/device_sync.py`. Remote nur über `app.run_http("CloudM.UserDataAPI", "<funktion>", ...)` (F25); lokal dieselbe Data-API des Owners.
- **Abgleich pro Pfad:** je Scope (`private`, `mod_data` je Mod) remote und lokal `list_data` → nach Pfad zusammenführen → nur Pfade, die einseitig fehlen oder deren `updated_at` neuer ist, **einzeln** mit `get_data`/`set_data`. Unveränderte Pfade nicht anfassen. Den letzten Stand (`path → updated_at` beider Seiten) lokal ablegen, über den vorhandenen FileHandler V2 bzw. die DB, keine neue Datei-Art. So prüft der nächste Lauf nur die Differenz.
- **Mod-Daten pro Schlüssel:** `get_mod_data(source_mod, key=…)` nur für geänderte Schlüssel. Setzen mit `set_mod_data(source_mod, key=…, data=<nur geänderte Felder>, merge=True)`.
- Welche Pfade und Schlüssel es gibt, kommt aus `list_data` und den vorhandenen FileHandler-Schlüsseln, nicht aus einer festen Liste.
- **Konfliktregel** (Vorschlag des Planers, Owner kann kippen): neueres `updated_at` gewinnt. Bei gleichem Zeitstempel und unterschiedlicher Größe gewinnt der Server, und der lokale Stand wird als `<pfad>.conflict-<zeit>` lokal behalten. Löschungen werden nicht übertragen.
- **Vorher messen** (lokal auf 8467): `set_data`/`get_data` mit dict, str und Binärinhalt; `list_data` für `private` und `mod_data`; `set_mod_data` mit `merge=True` und zwei Feldern nacheinander. Antwortformen ins Messprotokoll. Kommt Binärinhalt nicht unverändert zurück: **stoppen und melden**, kein eigenes Kodierschema erfinden.
- **Abschlusskriterium:** unittest mit gemocktem `run_http`:

| Fall | Erwartet |
|---|---|
| remote 3 Pfade, lokal 1 identischer | genau 2 `get_data` |
| lokal 1 Pfad geändert | genau 1 `set_data`, nur dieser Pfad |
| nichts geändert, zweiter Lauf | 0 `get_data`, 0 `set_data` (nur `list_data`) |
| ein Feld eines Mod-Schlüssels geändert | 1 `set_mod_data` mit genau diesem Feld und `merge=True` |
| kein Netz | kein Absturz, `error` im Ergebnis, lokale Daten unverändert |
| `list_data(scope="mod_data")` gegen die echte Funktion | liefert Mod-Pfade, nicht `private` (Test zu F23) |

### S2 — Blobs über die Data-API

- In `device_sync.py`: lokal `list_blobs()` (F24), remote `list_data(prefix=<festes Präfix, z. B. "blobs/">)`. Nur fehlende oder neuere Blobs **einzeln**: `read_blob` → `set_data` bzw. `get_data` → `create_blob`/`update_blob`. Inhalt in der Form, die S1 als verlustfrei gemessen hat. Keine Löschungen.
- **Nicht verwenden:** `BlobStorage.sync()` / `_cloud_minio`, `get_secure_storage_credentials`.
- **Abschlusskriterium:** unittest, zwei Temp-Blobspeicher und ein gemeinsamer Fake-`run_http`: Blob von A erscheint inhaltsgleich in B; zweiter Lauf ohne Änderung → 0 Übertragungen; von 3 Blobs wird nur der geänderte übertragen; ohne Gerätekonto → `skipped`.

### S3 — Globaler ISAA-Ordner und ein Einstieg

- Ordner aus F22 (`power_vfs` **nicht** importieren, der Import zieht ISAA mit). Dateien als Blobs `isaa/global/<pfad>` über S2, zurück als Dateien, nur bei geänderter Änderungszeit oder Größe. `..` und absolute Pfade überspringen und loggen; Dateien über 50 MB überspringen und loggen; keine Löschungen.
- `async sync_all(app)`: export → Blobs → import → Kontodaten. Rückgabe: Teilergebnisse und `finished_at`.
- **Auslöser:** nach `link_account`; beim Start des Owners (im Hintergrund); alle 10 min; `POST /api/wizard/account/sync` (nur Loopback, nur mit Gerätekonto). `GET /api/wizard/account/status` zeigt `last_sync`. Ein Fehler im Sync beendet den Owner nie.
- **Abschlusskriterium:** unittest mit Gerät A/B: `global/notes/a.txt` kommt bei B an; `isaa/global/../x` wird nicht geschrieben; ohne Konto → `skipped`; zweiter Lauf → 0 Übertragungen.

### T10 — ISAA antwortet, OBS zeigt den Lauf

- Nur Abnahme, mit einem auf dem Entwicklungsrechner vorhandenen LLM-Key. Code nur, wenn das Kriterium rot ist.
- **Abschlusskriterium:** isolierte echte Installation → Dashboard → ISAA → Nachricht → Antwort; WS-Verbindung auf 8468; `/obs` zeigt einen neuen Eintrag zu diesem Lauf. Antwort wörtlich oder als Bildschirmfoto im Bericht.

### D1 — Owner rollt den Stand auf `simplecore.app` aus

Der Agent stoppt hier und wartet auf die Bestätigung. Server-seitig ändert WP1 nur Tests. Ausgerollt werden muss trotzdem mindestens `5636d82` (Scope-Fix in `list_data`).

### K0 — Messung gegen `simplecore.app` (nach D1)

- `docs/devdocs/probe_simplecore_2026-11.md` mit Statuscode und Feldnamen, **ohne Token**, für: `/api_user_data`, `list_data` (`private` und `mod_data`), `get_data` (dict, str, binär), `set_data`, `get_mod_data(key=…)`, `set_mod_data(key=…, merge=True)`. Weicht etwas von der lokalen Messung aus S1 ab: melden.
- Ein temporäres Token liegt nur in einer Datei außerhalb des Repos und wird nach der Messung gelöscht.

### K4 — Abnahme Konto (live)

| Schritt | Erwartet |
|---|---|
| frische Installation, `/api/wizard/state` | `"needs_account":true` |
| `/` | Kontoseite |
| Login bei SimpleCore | Dashboard |
| `/api_user_data` | `user_id` wie in K0 |
| Netz trennen, Workers neu, `/` | Dashboard, angemeldet |
| Zugriff über die LAN-IP | nicht angemeldet |
| `device_owner.json` | existiert, enthält kein Token |
| `scripts/verify_install_windows.ps1` | V1–V12 PASS |

**Wenn rot:** Bleibt die Seite auf `waiting`, prüfen, ob eine Firewall Port 8765 sperrt. Den Befund melden, den Port nicht ändern.

### S4 — Abnahme Sync (live, zwei Installationen, gleiches Konto)

| Schritt | Erwartet |
|---|---|
| Gerät 1: `global\probe.txt` anlegen, `POST /api/wizard/account/sync` | kein `error`, mindestens 1 Upload |
| Gerät 2: frisch installieren, anmelden | `probe.txt` inhaltsgleich (Hash) |
| Gerät 2: `get_data` für einen Pfad aus K0 | bestehende Kontodaten |
| Gerät 2: `set_data("probe/b")`, Sync; Gerät 1: Sync, `get_data` | gleicher Inhalt |
| zweiter Sync ohne Änderung, beide Geräte | 0 `get_data`, 0 `set_data` |
| Gerät 1: Netz trennen, neu starten | angemeldet, `probe.txt` lesbar |

Ohne diese Tabelle ist der Sync NICHT VERIFIZIERT. Gerät 2 darf ein zweites Windows-Benutzerkonto auf demselben Rechner sein.

---

## 7. Was der Owner tut

| # | Was | Wann |
|---|---|---|
| D1 | Stand (mind. `5636d82`) auf `simplecore.app` ausrollen | nach T10 |
| — | Browser-Login bei SimpleCore | K0, K4, S4 |
| — | zweites Gerät oder zweites Windows-Konto | S4 |
| — | Konfliktregel S1 bestätigen oder ändern | vor S1 |

## 8. Nicht anfassen

`mods/CloudM/auth/api_session.py::update_user_data`, `utils/clis/local_cli.py:258` ff. (nur aufrufen), `utils/clis/cli_oauth.py` ab Zeile 105, `mods/SocketManager.py`, `UserDataAPI.get_secure_storage_credentials`, `BlobStorage.sync()`, `launch.rs`-Pfade für Linux/macOS, Dispatcher (Punkt E ist gestrichen).

## 9. Bericht je Task

An `docs/devdocs/agent_report_2026-11.md` anhängen, Format wie in v3 §2.5: Task, Zustand, Commit, Wiederverwendet/Entfernt, Kriterium (Befehl wörtlich), Ausgabe (wörtlich), Tests vorher/nachher, Seiteneffekte, Abweichung, Frage an den Owner.
