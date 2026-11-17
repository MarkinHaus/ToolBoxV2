# icli 2.0 – Plan

Stand: Schritte 1–5, 8 und 9 umgesetzt; 6 und 7 offen (laufen lokal). Die alte `flows/isaa/icli.py` bleibt unverändert und parallel nutzbar.

Start: `tb -m icli2` · Code: `toolboxv2/mods/isaa/extras/icli2/` · Starter: `toolboxv2/flows/isaa/icli2.py`

## Ziele

- Weniger Befehle, keine Subcommands, Auswahl über Picker.
- UI hängt nie: Ausführung und Oberfläche sind strikt getrennt.
- Schönes, ruhiges Streaming im ISAA-Branding.
- Volle Kontrolle: pausieren, stoppen, nachsteuern, parallele Läufe.
- Passt zu ToolBox, isaa-Mod und CloudM; vorhandene Funktionen werden genutzt statt neu gebaut.

## Entscheidungen aus der Fragerunde

| Thema | Entscheidung |
|---|---|
| Trennung | Engine in eigenem Thread mit eigenem asyncio-Loop, Protokoll JSON-fähig (später als Prozess möglich) |
| UI | Vollbild mit Textual |
| Layout | Chat-Spalte, Seitenleiste per Taste, Picker als Overlay |
| Befehle | Wenige flache Befehle plus Fuzzy-Picker und Palette |
| Streaming | Live-Markdown, Tool-Zeilen aufklappbar, Gedanken eingeklappt |
| Kontrolle | Pause/Stopp, Nachsteuern im Lauf, Parallel-Läufe (keine Tool-Freigaben) |
| Umfang v1 | Tasks & Jobs, Audio & Omni, Skills/MCP/Plugins/Features; Coder nicht |
| CloudM | Login & Identität, LiveSync für Mounts, ToolBox-Module aus der Palette |
| Ablage | Paket unter `mods/isaa/extras/icli2`, dünner Flow-Starter |
| Abhängigkeit | `textual` im bestehenden `isaa`-Extra |
| Migration | Agenten und Sessions werden geteilt; keine alte History, keine Alias-Befehle |
| Lieferung | Pro Schritt ein reiner Diff (`git apply`) |
| Agenten | Extern registrierte Agenten erscheinen in der icli2, eigene Tools werden automatisch gepackt |
| Discord | Bestehende Integration soll unverändert funktionieren |

## Architektur

```
app.py      Textual-UI        sendet Commands, holt Events pro Frame ab
picker.py   Fuzzy-Auswahl     wiederverwendbar für alle Listen
theme.py    Aussehen          nur aus isaa_branding.py und zen_renderer.py
   │  Command ▼      ▲ Event (Queue, UI zieht – wird nie aufgerufen)
engine.py   Engine-Thread     Läufe, Bündelung der Deltas, Herzschlag
fmt.py      Formatierung      Tool-Parameter/-Ergebnisse ohne JSON-Klammern
backend.py  isaa-Adapter      einzige Stelle mit isaa-Zugriff
protocol.py Nachrichten       reine Daten
```

Regeln, die für jeden weiteren Schritt gelten:

1. `app.py` wartet auf nichts: kein Agent, keine Datei, kein Netzwerk.
2. Nur `backend.py` kennt isaa. Die Engine kennt kein Textual.
3. Vor jeder neuen Funktion wird geprüft, ob es sie im Code schon gibt.
4. Alles, was dauern kann, läuft als eigener Task in der Engine.

## Befehle

| Befehl | Status | Ersetzt (alte icli) |
|---|---|---|
| `/q` | fertig | `/q`, `/Q`, `/quit`, `/exit` |
| `/agent [name]` | fertig | `/agent switch`, `/agent list` |
| `/pack [name]` | fertig | – (manuelles Vollpaket mit Checkpoint) |
| `/open n`, `/fold` | fertig | – |
| `/stop`, `/clear` | fertig | Interrupt-Menü, `/clear` |
| `/model` | fertig | `/agent model` |
| `/session [name]`, `/new`, `/reset` | fertig | `/session new/clear/show` |
| `/mount [pfad]` | fertig | `/vfs mount/unmount/save` |
| `/task` | fertig | `/task log/view`, Fokuswechsel |
| `/job [zeitplan name: auftrag]`, `/dream` | fertig | `/job add/list/remove`, `/job dream create` |
| `/network [datei]` | fertig | – (Agent-Network sichern/laden) |
| `/side` | fertig | – |
| `/feature`, `/skill`, `/mcp`, `/plugin` | Schritt 6 | gleichnamige Befehle, über Palette |
| `/discord` | Schritt 7 | unverändert durchgereicht |
| `/login`, `/logout` | fertig | – |
| `/tb [mod.fn key=wert]` | fertig | – (ToolBox-Funktion aufrufen) |
| `/share` | fertig | `/vfs share` (nur verbinden/trennen) |
| `/voice`, `F4` | fertig | `/audio`, `/omni` (Omni nur Grundfunktion) |

Tasten: `Esc` pausieren, `Esc Esc` abbrechen, `Strg+O` alle Tools auf/zu, `Strg+R` Gedanken auf/zu, `Alt+1–9` letzte n Tools öffnen, `Strg+L` leeren, `Strg+Q` beenden. `Strg+P` Palette, `↑/↓` Eingabe-History, `F2` Seitenleiste, `F3` nächster Lauf, `Umschalt+Enter`/`Strg+J` neue Zeile. Geplant: `F4` Sprache.

## Erledigt

### Schritt 1 – Grundgerüst (`icli2-01-grundgeruest.patch`)
- Protokoll, Engine-Thread, isaa-Adapter, Textual-UI, Flow-Starter.
- Stream-Deltas auf ca. 25 Events/s gebündelt, UI holt max. 200 Events pro Frame.
- Herzschlag: blockiert die Engine, zeigt die Fußzeile das an, die Eingabe bleibt frei.
- Nachsteuern nutzt das vorhandene Sidecar-Steering von `a_stream`.

### Schritt 2 – Branding und Agenten (`icli2-02-branding-agenten.patch`)
- Farben, Logo, Begrüßung, Symbole aus dem vorhandenen Branding.
- `/agent` mit Fuzzy-Picker; Liste bei jedem Aufruf live aus isaa und von Platte.

### Schritt 3 – Auto-Pack, Signal-Leiste, Tool-Darstellung (`icli2-03-autopack-tools.patch`)
- **Auto-Pack im isaa-Mod** (`mods/isaa/module.py`): Wird ein Agent gebaut, werden seine eigenen Tools automatisch nach `<data_dir>/Agents/<name>/<name>.tar.gz` gepackt. Baut ein anderer Prozess denselben Agenten, hängt `get_agent` die Tools wieder an. Gilt für das ganze isaa-System, nicht nur die icli2.
  - Als „eigen“ gelten lokale Tools, deren Modul nicht mit `toolboxv2.` beginnt; System-Tools baut der Builder selbst.
  - Ein bestehendes Paket wird nie durch eine Version mit weniger Tools überschrieben.
  - Abschaltbar mit `ISAA_AUTO_PACK=0`.
- **Fehler behoben:** `save_agent` las ein nicht mehr vorhandenes `tool_manager.tools` und konnte deshalb keine Tools exportieren.
- **Eingabeleiste** in Graublau mit Signalfarbe: Cyan läuft, Grün fertig, Gelb Frage/pausiert, Rot abgestürzt.
- **Tool-Zeilen** ohne JSON: Begründung, falls vorhanden, sonst `schlüssel using wert`; Details als eingerückte Liste; Ergebnis-Kurzinfo über den vorhandenen `ZenRenderer._extract_tool_meta`.
- Auf-/Zuklappen per Taste und Befehl.

### Schritt 4 – Modell, Session, Mount, Palette (`icli2-04-modell-session-mount.patch`)
- Engine und Protokoll kennen nur noch zwei generische Eingänge: `LIST` (Zeilen für einen Picker) und `DO` (Aktion). Neue Bereiche brauchen ab jetzt nur Code in `backend.py` und einen Eintrag in der Befehlstabelle.
- `/model`: Picker mit Modellen aus dem Register der alten icli (`MODEL_MAPPING`, per AST gelesen, ohne `icli.py` zu importieren) plus Gateway-Modelle; zweiter Picker für schnell/komplex/beide; wird in `agent.json` gespeichert.
- `/session`, `/new`, `/reset`: wechseln, neu anlegen, Verlauf leeren (gleiche Logik wie `/session clear`).
- `/mount ordner` hängt ein (erster Mount unter `/project`, weitere unter `/<ordnername>`); `/mount` ohne Argument listet, Enter hängt aus und speichert.
- Palette `Strg+P` (auch `/` + Enter), Eingabe-History mit `↑/↓`, gespeichert unter `<appdata>/icli2/history.txt`.
- Kopfzeile zeigt Anzahl der Mounts.

### Schritt 5 – Läufe, Seitenleiste, Sub-Agenten, Jobs, Networks (`icli2-05-laeufe-jobs-networks.patch`)
- **Parallele Läufe:** Jeder Lauf hat einen eigenen Block im Chat, Streams vermischen sich nicht. `F3` oder `/task` wechselt den Fokus; `Esc` wirkt nur auf den fokussierten Lauf.
- **Seitenleiste `F2`:** Läufe mit Status und Sub-Agenten, Jobs, Mounts, Kontext-Auslastung. Jobs und Mounts werden nur bei offener Leiste alle 2 s nachgeladen.
- **Sub-Agenten:** Chunks mit `_sub_agent_id` erscheinen als eingerückter Unterblock mit dem Auftrag als Überschrift; sie verändern nie Status oder Antwort des Elternlaufs.
- **Mehrzeilige Eingabe:** Enter sendet, `Umschalt+Enter` oder `Strg+J` neue Zeile, bis 8 Zeilen hoch.
- **Jobs:** Scheduler wird wie in der alten icli gestartet (`on_cli_start`, verpasste Jobs nachholen) und beim Beenden gestoppt. Job-Läufe gehen über `Engine.spawn` und erscheinen als normale Läufe. `/job` listet (ausführen, pausieren/fortsetzen, löschen); anlegen mit `/job alle 30m name: auftrag`, `/job um 09:00 name: auftrag`, `/job start name: auftrag`. `/dream` legt den nächtlichen Dream-Job an.
- **Agent-Networks:** `/network` sichert den aktiven Agenten samt gebundener Agenten nach `<data_dir>/Agents/_networks/<agent>.tar.gz`; gesicherte Networks stehen im Agent-Picker (`⧉ name`), `/network datei` lädt ein beliebiges Archiv.
- **Session-Rückblick:** Beim Agent- oder Session-Wechsel werden die letzten Nachrichten angezeigt.
- **Pausierte Läufe überleben den Neustart:** Beim Beenden nach `<appdata>/icli2/resume.json` gesichert, beim Start als „paused“ zurück in die Engine; die nächste Nachricht in der Session setzt fort.
- Behoben: `Esc` schließt jetzt offene Picker (wurde vorher vom Stopp-Kürzel abgefangen).
- Unabhängiges Code-Review eingearbeitet (10 Funde): übergebener Stream wurde verworfen (Dream-Jobs liefen als normaler Chat), früh scheiternde Läufe blieben in der UI hängen, Shutdown ließ Tasks zurück, Scheduler-Stopp, Resume-Datei atomar und pro Eintrag, Wertebereiche bei `/job`, Formatierer gegen kaputte/tief verschachtelte Werte, Zustand wechselt erst nach Erfolg.
- Prüfskripte liegen jetzt im Repo: `tests/check_ui.py`, `tests/check_backend.py`, `tests/check_autopack.py`.

### Schritt 8 – CloudM (`icli2-08-09-cloudm-voice.patch`, Datei `cloudm.py`)
- `/login` zeigt den angemeldeten Benutzer (`app.session`, sonst `_find_cli_session`); der Name steht in der Kopfzeile. **Anmelden selbst geht nicht in der icli2**, weil `cli_login` Tastatureingaben per `input()` braucht: `/login` nennt dann den Befehl `tb login` für das normale Terminal. `/logout` nutzt `cli_logout`.
- `/tb` öffnet einen Picker über `app.functions`; `/tb mod.fn key=wert …` ruft über `app.a_run_any((mod, fn), get_results=True, …)` auf (Werte als JSON, sonst Text), Ergebnis lesbar formatiert.
- `/share` listet die gespeicherten LiveSync-Shares der alten icli (`<appdata>/icli/vfs_shares.json`) und verbindet/trennt sie über `VFSSyncManager.connect/disconnect`. Neue Shares anlegen geht weiterhin nur in der alten icli (`/vfs share connect`).

### Schritt 9 – Sprache (gleicher Patch, Datei `voice.py`)
- Audio-Code wird erst beim ersten Gebrauch geladen (per Test abgesichert).
- `F4` startet/beendet die Aufnahme (`LocalMicRecorder`), transkribiert mit `Stt.transcribe` und setzt den Text in die Eingabe.
- `/voice`: Vorlesen an/aus (`AudioStreamPlayer`, Satz-Pufferung wie in der alten icli, Code-Blöcke werden übersprungen), Ein-/Ausgabegerät wählen, Omni-Session starten/stoppen.
- **Omni ist nur die Grundfunktion:** Backend, Mikrofon, Player und die Tools des Agenten. Es fehlen gegenüber der alten icli: Delegations-/VFS-/Chat-Tools, Weltmodell, JobManager, Zustandsspeicher, Silero-VAD, Sprecher-Erkennung, Pegelanzeige. Fest auf `omni_cloud` (Gemini Live).
- Aufnahme und Omni schließen sich gegenseitig aus (beide brauchen das Mikrofon).

## Offen – Arbeitsanleitung für die restlichen Schritte

Geschrieben für einen kleinen Agenten. Ein Schritt pro Durchgang, in dieser Reihenfolge.

### Vorgehen für jeden Schritt (immer gleich)

1. **Erst nachsehen, dann nutzen.** Zu jedem Punkt steht unten, wo die Funktion im System schon existiert. Stelle öffnen, Signatur lesen, erst dann schreiben. Nichts nachbauen, was es gibt. `flows/isaa/icli.py` nie importieren (lädt Audio/Omni) und nie ändern – nur lesen.
2. **Nur drei Stellen anfassen:**
   - `backend.py`: `_rows_<was>()` liefert `(titel, [ _row(value, label, hint) ])`, `_do_<aktion>(**args)` liefert `{"ready": …}` und/oder `{"text": …}`. Fehler als Exception werfen, die Engine zeigt sie an.
   - `app.py`: Eintrag in `COMMANDS`, ein `elif` in `_command` (`do("aktion", …)` oder `ask("was")`), bei Picker-Auswahl ein `elif` in `_pick`.
   - `engine.py` und `protocol.py` bleiben unverändert, außer der Schritt verlangt es ausdrücklich.
3. **Regeln:** `app.py` wartet auf nichts (kein Datei-, Netz- oder isaa-Zugriff). Blockierendes im Backend in `asyncio.to_thread`. Lang laufende Dinge über `self._spawn(agent, session, query, kind=…)`, damit sie als Lauf in der UI erscheinen.
4. **Prüfen:** In `tests/check_ui.py` die Fake-Klasse um die neuen `rows`/`do` erweitern und die Bedienung per `pilot.press` durchspielen; in `tests/check_backend.py` das Backend gegen Nachbauten (oder echte, leicht importierbare Module) testen. Alle drei Skripte müssen durchlaufen.
5. **Abschluss:** `PLAN.md` aktualisieren (Befehlstabelle, „Erledigt“, Lücken), committen, pushen, Patch als reinen Diff gegen den vorherigen Schritt erzeugen (`git diff <letzter-schritt-commit> HEAD > icli2-0N-name.patch`) und mit `git apply --check` auf dem Vorgänger prüfen.

### Schritt 6 – Features, Skills, MCP, Plugins

| Befehl | Vorhandenes im System (zuerst lesen) | Umsetzung |
|---|---|---|
| `/feature` | `icli.py:1912` `SimpleFeatureManager` (`list_features`, `enable`, `disable`, `is_enabled`, `set_agent`), Lader `load_*_feature(fm)` ab `icli.py:1950`; Nutzung `_cmd_feature` `icli.py:13494` | Die Klasse und die Lader hängen an `icli.py`. Kopie in neue Datei `icli2/features.py` (nur die Klasse und die kleinen Lader: desktop_auto, web_auto, isis, full_web_auto, coder_toolkit, chain_toolkit, execute; die großen docs/autodoc/autotest/autofix erst, wenn gebraucht). Picker mit `✓`/`·` im Label, Enter schaltet um. Aktive Features in READY als `features` und in der Kopfzeile. |
| `/skill` | `_cmd_skill` `icli.py:12688`, `_cmd_skill_invoke` `icli.py:9952`, `_refresh_skill_names` `icli.py:9918`; SkillsManager in `mods/isaa/base/Agent/skills.py` | Picker listet Skills des aktiven Agenten; Enter aktiviert den Skill für die nächste Nachricht (alte Logik: Instruktion wird der Anfrage vorangestellt, siehe `_active_skill_context` in `icli.py:14505`). Dafür im Backend `self._skill_context` halten und in `stream()` vor die Query setzen. |
| `/mcp` | `_cmd_mcp` `icli.py:9110`, `_mcp_connect_and_register` `icli.py:9382`, `mods/isaa/extras/mcp_session_manager.py` | Picker listet konfigurierte Server mit Status; Enter verbindet/trennt. Hinzufügen per `/mcp name befehl args…` (kein Wizard). |
| `/plugin` | `_load_all_plugins` `icli.py:9461`, `ClaudeCodePluginParser` `icli.py:4273`, `scan_plugin_dir` `icli.py:4354`, `_cmd_plugin` `icli.py:9625` | Nur Liste und An/Aus. Installation aus Git/Marketplace bleibt in der alten icli. |

Fertig, wenn: jeder der vier Befehle einen Picker öffnet, Umschalten eine Rückmeldung als Hinweis zeigt und der Zustand nach erneutem Öffnen stimmt.

### Schritt 7 – Discord und Telegram (unverändert weiterverwenden)

- **Vorhanden:** `mods/isaa/extras/discord_interface/integration_example.py:74` `patch_cli_for_discord(host)`, `discord_cli_extension.py` `DiscordCLIExtension(host)` mit `handle_command(args)` und `get_completer_dict`; Telegram: `mods/isaa/extras/telegram_interface/telegram_cli_extension.py` `TelegramCliExtension(host)` mit `handle_command(args_str)`. Aufruf in der alten icli: `icli.py:15427` (`run`).
- **Was die Extension vom Host braucht** (per Suche bestätigt): `host.isaa_tools`, `host.active_agent_name`, `host._handle_command`, `host._build_completer`, `host.run_agent_monitored`, `host.discord_ext`.
- **Umsetzung:** neue Datei `icli2/host_shim.py` mit Klasse `HostShim`:
  - `isaa_tools` = `backend.isaa`, `active_agent_name` als Property auf `backend.agent`.
  - `_handle_command(text)`: no-op (wird von `patch_cli_for_discord` umhüllt); `_build_completer()` gibt `({}, None)` zurück.
  - `run_agent_monitored(agent_name, query, kind=…, **kw)`: Signatur und Chunk-Format aus `icli.py:6211` übernehmen. Als Async-Generator umsetzen, der die Chunks von `agent.a_stream` durchreicht **und** denselben Lauf über `Engine.spawn(..., stream=<tee>)` in der UI zeigt. Dafür die Chunks in eine `asyncio.Queue` spiegeln.
  - Im Backend `boot()`: `patch_cli_for_discord(shim)` in `try/except ImportError`.
  - `_do_discord(args)` ruft `shim._handle_command("/discord " + args)`; Telegram analog über `shim.telegram_ext.handle_command`.
- **Ausgabe abfangen:** Die Extensions geben per `print`/`c_print` aus. Im `_do_discord` mit `contextlib.redirect_stdout(io.StringIO())` einfangen und als `{"text": …}` zurückgeben (ANSI mit `toolboxv2.remove_styles` entfernen).
- `/discord` ohne Argument: Picker mit den Unterbefehlen aus `get_completer_dict()`.
- Fertig, wenn: `/discord status` die Ausgabe der Extension als Hinweis zeigt und ein über den Shim gestarteter Lauf als Block mit `origin` „discord“ erscheint (im Test mit Fake-Extension prüfen).

### Danach

- Engine als eigener Prozess: `protocol.py` ist schon JSON-fähig; Transport analog `mods/icli_web/bridge.py`.
- `icli_web` an `Engine.spawn`/Events anschließen statt an `run_agent_for_web`.
- OpenRouter-Modellliste im Hintergrund nachladen (`fetch_openrouter_models` `icli.py:951`).

## Bekannte Lücken und Risiken

- **Review-Fixes** sind über die Prüfskripte abgedeckt, aber wie alles andere nicht gegen den echten isaa-Mod gelaufen.
- **Nur mit Fake-Agent getestet.** Der Pfad über den echten isaa-Mod (Agent bauen, `a_stream`, Pause, Auto-Pack mit echten Tools) ist noch nie gelaufen.
- **Datenordner:** `data_dir` hängt vom App-Namen ab. Läuft ein Flow unter einem anderen Namen als `isaa-host`, liegen seine Agenten evtl. in einem anderen Ordner und erscheinen nicht.
- **Nicht jedes Tool ist packbar:** Tools mit offenen Verbindungen, Prozessen oder Locks lassen sich nicht serialisieren und fehlen nach dem Laden.
- **Pakete führen Code aus:** Beim Laden wird der enthaltene Code ausgeführt. Geladen wird nur aus dem eigenen `Agents`-Ordner; fremde Pakete dort nicht ablegen.
- **Frage-Erkennung ist eine Heuristik:** Gelb erscheint bei pausiertem Lauf oder wenn die Antwort mit `?` endet.
- **`Alt+Ziffer`** wird nicht von jedem Terminal weitergegeben; `/open n` funktioniert immer.
- **Referenzbilder** für das Design konnten nicht geöffnet werden; das Aussehen folgt dem Branding im Code.
- `/mount` und `/model` sind gegen nachgebaute isaa-Objekte getestet, nicht gegen das echte VFS.

- **Jobs:** gegen den echten `JobScheduler` aus dem Repo getestet; Networks, Rückblick und Resume nur gegen Nachbauten.
- **Job-Fehler:** Scheitert ein Job-Lauf, meldet die icli2 das dem Scheduler als Fehler (die alte icli meldete „completed“).
- **`Umschalt+Enter`** kommt nicht in jedem Terminal an; `Strg+J` funktioniert überall.
- OpenRouter-Modellliste wird nicht nachgeladen (Netzwerkabruf); Modellnamen lassen sich mit `/model name` direkt setzen.
- Beendete Job-/Dream-Läufe aus `on_cli_exit` sind nach dem Schließen der UI nicht mehr sichtbar.

- **Schritt 8/9 von Subagenten geschrieben**, von mir gegengelesen und die genutzten Signaturen im Quellcode nachgeprüft. Getestet nur gegen Nachbauten: kein echter Login, kein echter LiveSync-Server, kein Mikrofon, kein TTS, kein Gemini Live.
- **Gerätewahl** bei `/voice` wird nicht gespeichert und gilt nur bis zum Beenden.
- **Vorlesen** nutzt fest Groq-TTS mit Stimme „autumn“ (wie die alte icli); die Emotions-Erkennung ist vereinfacht.
- **Patch 08-09 baut auf Patch 05 auf**, nicht auf 06/07. Beide Stränge ändern `app.py` und `backend.py`; beim Zusammenführen `git apply --3way` verwenden.

## Patches anwenden

```bash
git apply icli2-01-grundgeruest.patch
git apply icli2-02-branding-agenten.patch
git apply icli2-03-autopack-tools.patch
git apply icli2-04-modell-session-mount.patch
git apply icli2-05-laeufe-jobs-networks.patch
git apply --3way icli2-08-09-cloudm-voice.patch   # nach den lokalen Schritten 6/7
pip install "textual>=8"
```

## Mini-Prompt zum Fortfahren

```
Lies toolboxv2/mods/isaa/extras/icli2/PLAN.md, Abschnitt "Offen – Arbeitsanleitung".
Setze genau den nächsten offenen Schritt um und halte dich an "Vorgehen für jeden Schritt":
zuerst die genannten Stellen im System lesen und vorhandene Funktionen nutzen, nur
backend.py/app.py (plus die im Schritt genannte neue Datei) anfassen, icli.py nie ändern.
Prüfskripte in icli2/tests erweitern und alle drei ausführen, PLAN.md aktualisieren,
committen, pushen und mir den Schritt als .patch (reiner Diff auf dem Vorgänger) schicken.
```
