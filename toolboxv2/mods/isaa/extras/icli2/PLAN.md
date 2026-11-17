# icli 2.0 – Plan

Stand: alle Schritte umgesetzt und einmal real durchgefahren (Schritt 10). Die alte `flows/isaa/icli.py` bleibt unverändert und parallel nutzbar.

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
| `/settings`, `/set name wert` | fertig | `/set_max_iterations` |
| `/context` | fertig | `/context` (jetzt in der Seitenleiste) |
| `/vfs [pfad]` | fertig | `/vfs` (ansehen, mit Pfad-Vervollständigung) |
| `/feature`, `/skill`, `/mcp [name befehl …]`, `/plugin` | fertig | gleichnamige Befehle (Picker mit An/Aus) |
| `/plugin install <quelle>`, `confirm`, `discard`, `remove`, `market [quelle]`, `markets` | fertig | `/plugin install`, `/plugin marketplace`, `/plugin browse`, `/plugin remove` |
| `/discord [befehl …]`, `/telegram [befehl …]` | fertig | unverändert an die vorhandenen Erweiterungen durchgereicht |
| `/login`, `/logout` | fertig | – |
| `/tb [mod.fn key=wert]` | fertig | – (ToolBox-Funktion aufrufen) |
| `/share` | fertig | `/vfs share` (nur verbinden/trennen) |
| `/voice`, `F4`, `#audio` | fertig | `/omni`, F4-Aufnahme, `#audio` am Nachrichtenende |

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
- **Kein Dauer-Schalter fürs Vorlesen.** Gesprochen wird nur die Antwort auf eine Nachricht, die auf `#audio` endet (wie in der alten icli; das Kürzel kommt nicht beim Agenten an), oder in einer Omni-Session. `/voice`: Omni starten/stoppen, Ein-/Ausgabegerät wählen.
- **Omni ist nur die Grundfunktion:** Backend, Mikrofon, Player und die Tools des Agenten. Es fehlen gegenüber der alten icli: Delegations-/VFS-/Chat-Tools, Weltmodell, JobManager, Zustandsspeicher, Silero-VAD, Sprecher-Erkennung, Pegelanzeige. Fest auf `omni_cloud` (Gemini Live).
- Aufnahme und Omni schließen sich gegenseitig aus (beide brauchen das Mikrofon).

### Schritt 6 – Features, Skills, MCP, Plugins (`icli2-06-07-features-discord.patch`, Dateien `features.py`, `connect.py`)
- `/feature`: `SimpleFeatureManager` und die sieben kleinen Lader aus der alten icli (Kopie, alte Datei unverändert); aktive Features stehen in der Kopfzeile.
- `/skill`: Skills des Agenten über den vorhandenen `SkillsManager`; ein aktiver Skill wird jeder Anfrage vorangestellt, im Format der alten icli.
- `/mcp`: Server aus `agent.json` verbinden/trennen (`MCPSessionManager`, `_register_mcp_session`); `/mcp name befehl args…` oder `/mcp name https://…` legt einen neuen an.
- `/plugin`: Plugins aus `<appdata>/icli/plugins` an/aus (Skills und MCP-Server des Plugins); beim Start werden alle nicht abgeschalteten geladen.

### Schritt 7 – Discord und Telegram (gleicher Patch, Datei `bridges.py`)
- Die vorhandenen Erweiterungen laufen **unverändert**: `HostShim` bietet genau, was sie vom alten Host benutzen (`isaa_tools`, `active_agent_name`, `_handle_command`, `_build_completer`, `run_agent_monitored`).
- `/discord …` und `/telegram …` reichen den Befehl durch; die Terminal-Ausgabe der Erweiterung erscheint als Hinweis. Ohne Argument: Picker mit den Unterbefehlen.
- Läufe, die über Discord/Telegram hereinkommen, erscheinen als normale Läufe (Block im Chat, Seitenleiste) und lassen sich mit `Esc` stoppen.
- Die Erweiterungen werden erst beim ersten `/discord` bzw. `/telegram` geladen.

### Schritt 10 – Echter Durchlauf und Nacharbeiten (`icli2-10-e2e-settings-vfs.patch`)
- **Echter Durchlauf:** toolboxv2 samt isaa-Mod installiert und die icli2 gegen den echten Agenten gefahren, mit einem lokalen Ersatz-LLM (`tests/e2e/`). Geprüft: Start, Lauf mit `think`, echtem Tool und Antwort, alle Picker, alle Aktionen, Job-Lauf, Discord/Telegram-Status über die unveränderten Erweiterungen.
- **Dabei gefundene und behobene Fehler:**
  - Antwort kam nur als Tool-Zeile: der Agent antwortet über das Tool `final_answer`; `think` wird jetzt als Gedanken gezeigt.
  - Absturz des Modell-Pickers bei gleichem schnellem und komplexem Modell (doppelte Werte).
  - Absturz bei offener Auswahl: Kopf-/Fußzeile wurden im Picker statt im Hauptbild gesucht.
  - Session-Liste enthielt leere Namen und brachte den Picker zum Absturz.
  - `/dream` legte bei jedem Aufruf einen weiteren Dream-Job an.
  - Tool-Fehler, die nur als Text gemeldet werden, galten als Erfolg.
- **Modell-Picker dynamisch** (`models.py`): liest bei jedem Öffnen Gateway (`TB_LLM_GATEWAY_URL`) und alle in litellm registrierten Custom-Provider neu, dann das Register der alten icli; der Titel nennt die Router-Präfixe des Agenten. Nebenbefund: Gateway-Modelle erschienen bisher nie (auch nicht in der alten icli), weil dort das Modul statt einer Instanz abgefragt wurde.
- **Autovervollständigung** beim Tippen von `/befehl` (Tab übernimmt, ↑↓ wählt, Enter führt aus) und für Pfade bei `/vfs`.
- **Befehlsmenü:** ↑↓ oder W/S wählen, → / D / Enter öffnet, ← / A zurück in die Befehlsliste, `/` startet das Filtern.
- **`/settings`:** Max. Iterationen (auch `/set max_iterations 50`, wird gespeichert), Kontext, Anmeldestatus, Datenordner; Zugangsdaten sind als „kommt“ vorgemerkt.
- **`/context`** öffnet die Seitenleiste und zeigt dort unten die Aufschlüsselung (`agent.context_overview`).
- **`/vfs [pfad]`** (`vfsview.py`): Ordner auflisten, Datei anzeigen, nur lesend.
- Leerzeile zwischen eigener Nachricht und Antwort; erste Taste nach verlorenem Fokus wird nicht mehr verschluckt.

### Schritt 11 – Plugin-Installation (`icli2-11-plugin-install.patch`, Datei `plugin_install.py`)
- `/plugin install <quelle>`: Quelle ist `owner/repo`, eine Git-URL, ein lokaler Pfad oder `name@marketplace` (wie die alte icli). Geklont wird mit `git clone --depth 1` in `<appdata>/icli/plugins/.staging/<name>`.
- Wie die alte icli: Plugin = Ordner mit `.claude-plugin/plugin.json` (auch verschachtelt im Repo), installiert unter dem Namen aus `plugin.json`, danach sofort aktiv.
- **Statt der Rückfrage im Terminal zwei Schritte:** ohne Funde wird direkt installiert. Mit Funden (gefährliche Muster, Hooks, `.mcp.json`, Symlinks nach außen) erscheint der Sicherheitsbericht, das Plugin bleibt im Staging: `/plugin confirm <name>` installiert, `/plugin discard <name>` verwirft.
- `/plugin remove <name>` deaktiviert und löscht. `/plugin market owner/repo` merkt einen Marketplace (gleiche Datei wie die alte icli), `/plugin market` zeigt alle Plugins daraus – Enter installiert –, `/plugin markets` listet die gemerkten – Enter entfernt.
- **Sicherheit:** git ohne Shell und mit `--` vor der Quelle, Quellen mit `-` am Anfang oder `::` werden abgelehnt, nur die Protokolle file/https/http/ssh. Jeder Name wird geprüft und jeder Pfad muss im Plugin-Ordner liegen – auch nach Auflösen von Symlinks. Aus dem Repo wird nichts ausgeführt, Hooks laufen nicht.
- **Geprüft:** `tests/check_plugin_install.py` mit echten lokalen Git-Repos (sauber, riskant, Symlink, kein Plugin, verschachtelt, umbenannt, defektes Manifest, Pfad-Ausbrüche, Option-Injection, Marketplaces); per absichtlich eingebauter Fehler nachgewiesen, dass der Test bei fehlender Symlink-, Pfad- und Quellen-Prüfung anschlägt. Dazu der echte Durchlauf: Installieren, Aktivieren, Deaktivieren, Entfernen gegen den realen isaa-Agenten, einmal direkt und einmal über die Oberfläche (`tests/e2e/plug.py`, `e2e.py`).
- Geschrieben von einem Subagenten (vor dem Prüfskript wegen Rate-Limit abgebrochen); Code vollständig gegengelesen, an die alte icli angeglichen (Manifest-Name, verschachtelte Wurzel, `name@marketplace`), Prüfskript selbst geschrieben.

## Arbeitsanleitung für weitere Schritte

Alle geplanten Schritte sind umgesetzt. Das Vorgehen gilt für jede Erweiterung; offene Ideen stehen unter „Danach“.

### Vorgehen für jeden Schritt (immer gleich)

1. **Erst nachsehen, dann nutzen.** Zu jedem Punkt steht unten, wo die Funktion im System schon existiert. Stelle öffnen, Signatur lesen, erst dann schreiben. Nichts nachbauen, was es gibt. `flows/isaa/icli.py` nie importieren (lädt Audio/Omni) und nie ändern – nur lesen.
2. **Nur drei Stellen anfassen:**
   - `backend.py`: `_rows_<was>()` liefert `(titel, [ _row(value, label, hint) ])`, `_do_<aktion>(**args)` liefert `{"ready": …}` und/oder `{"text": …}`. Fehler als Exception werfen, die Engine zeigt sie an.
   - `app.py`: Eintrag in `COMMANDS`, ein `elif` in `_command` (`do("aktion", …)` oder `ask("was")`), bei Picker-Auswahl ein `elif` in `_pick`.
   - `engine.py` und `protocol.py` bleiben unverändert, außer der Schritt verlangt es ausdrücklich.
3. **Regeln:** `app.py` wartet auf nichts (kein Datei-, Netz- oder isaa-Zugriff). Blockierendes im Backend in `asyncio.to_thread`. Lang laufende Dinge über `self._spawn(agent, session, query, kind=…)`, damit sie als Lauf in der UI erscheinen.
4. **Prüfen:** In `tests/check_ui.py` die Fake-Klasse um die neuen `rows`/`do` erweitern und die Bedienung per `pilot.press` durchspielen; in `tests/check_backend.py` das Backend gegen Nachbauten (oder echte, leicht importierbare Module) testen. Alle drei Skripte müssen durchlaufen.
5. **Abschluss:** `PLAN.md` aktualisieren (Befehlstabelle, „Erledigt“, Lücken), committen, pushen, Patch als reinen Diff gegen den vorherigen Schritt erzeugen (`git diff <letzter-schritt-commit> HEAD > icli2-0N-name.patch`) und mit `git apply --check` auf dem Vorgänger prüfen.

### Danach

- **Echter Durchlauf mit echtem LLM und auf Windows** - bisher nur Ersatz-LLM unter Linux.
- **Zugangsdaten** in `/settings` verwalten.
- **Volles Omni** (bestehenden Adapter `toolboxv2/mods/isaa/base/audio_io/omni.py` verwenden - nichts Neues bauen, die bestehende Klasse elegant integrieren): Delegations-/VFS-/Chat-Tools (`_make_omni_tools` `icli.py:11374`), Weltmodell, VAD, Sprecher-Erkennung, Pegelanzeige.
- **Modellliste aller Anbieter über den IntelligentRateLimiter** (`mods/isaa/base/IntelligentRateLimiter/free_providers/fetch_models.py:84` `fetch_available_models(only_free=True)` -> Cache `models_cache/free_models.json` mit `{provider, model_id, ..., context_length}`; `registry.py` kennt Provider-IDs wie `openrouter`, `gemini-cli`, `groq`). Ersatz/Erweiterung des Modell-Pickers in `models.py` (Gateway + litellm-Customs + `MODEL_MAPPING`) - kein litellm, keine separate OpenRouter-REST-Abfrage.
- **Agenten-Konfiguration in `/settings` (nach dem Builder):** Agent-Config-Felder (Modell, `max_iterations`, Tools) über `/settings` editierbar; dazu max. Kontextgröße (auto-detektiert aus `context_length` der Modellliste) und Offloading-Limits (`cleanup_max_message_count`, `cleanup_summary_max_chars`, `conversation_max_context_size` aus `agent.json`). Wirkt nur, wenn `agent.json` existiert (`data_dir/Agents/<name>/agent.json` - existiert immer nach dem Bauen des Agenten); sonst Hinweis.
- **Autocompletion für Subcommands:** Befehlstabelle in `app.py` um Argumente erweitern (z.B. `/plugin install|confirm|discard|remove`, `/job alle|um|start`, `/mcp <name> <cmd>`, `/set <key>`); Tab/Pfeiltasten vervollständigt den vollen Befehl.
- **icli2-Tool für den Agenten `self`:** Settings lesen/setzen, Modell wechseln - auch ANDERER Agenten (Agent laden, `agent.json` patchen, speichern; Agent muss nicht aktiv laufen). Falls isaa keine Spawn-API für fremde Agenten bietet, auch Spawn+Konfiguration+Aufgaben über das Tool. Registrierung nur für Agent `self`.
- **Dynamische Widgets (interaktiv, nur Planung, keine Umsetzung):** NUR über das icli2-Tool des Agenten `self` verwendbar (nicht als eigenständiges Feature): Neuer Event-Typ `widget` in `protocol.py`; Agent liefert JSON (Titel, Text, Knöpfe/Felder, Callback-Id). `app.py` rendert Panel; Antwort geht als Nutzer-Eingabe mit Präfix `widget:<id>` zurück in den Lauf. Nur vordefinierte Aktionen aus dem Widget-JSON - kein Freitext.

### Verbindung zu `mods/isaa/ui` (Web-Oberfläche) - geplant

Beide Oberflächen bleiben unabhängig startbar und nutzen dieselben Sessions (das System verifiziert das bereits). Kopplung nur über:

1. **Gemeinsamen Datenordner** (`data_dir/Agents/...`): Sessions/Verläufe in denselben Dateien.
2. **isaa-Kern als Vermittler:** beide nutzen dieselbe isaa-API (`get_agent`, Lauf-Start, Events). Ein in der icli2 gestarteter Lauf erscheint auch in der Web-UI und umgekehrt.

Kein `icli_web`-Bezug, keine Brücke, keine gemeinsame Prozessarchitektur.

### Plugin-Hooks - sicher einbauen (Standpunkt Markin)

`hooks/hooks.json`-Ausführung bleibt standardmäßig aus. Einbau nur mit:

1. Freigabe pro Plugin beim Installationszeitpunkt (Confirm-Flow `plugin_install.py`), nicht beim ersten Lauf.
2. Hooks laufen im Sandbox-Profil des Plugins; kein Zugriff auf andere Agenten/Ordner.
3. Timeouts und Zwangsbeenden bei Hänger; Ausgaben ins Lauf-Log.
4. Kein Env-Zugriff des Hosts, keine Netzwerkfreigabe per Default.

## Bekannte Lücken und Risiken

- **Review-Fixes** sind über die Prüfskripte abgedeckt, aber wie alles andere nicht gegen den echten isaa-Mod gelaufen.
- **Echter Durchlauf nur mit Ersatz-LLM unter Linux**, ohne MinIO, ohne Embedding-Modell (kein Download möglich), ohne Mikrofon. Nicht real geprüft: echtes LLM, Windows-Terminal, verbundenes Discord/Telegram, MCP-Server, LiveSync, Sprache, `#audio`.
- **Früher Stand:** Der Pfad über den echten isaa-Mod (Agent bauen, `a_stream`, Pause, Auto-Pack mit echten Tools) ist noch nie gelaufen.
- **Datenordner:** `data_dir` hängt vom App-Namen ab. Läuft ein Flow unter einem anderen Namen als `isaa-host`, liegen seine Agenten evtl. in einem anderen Ordner und erscheinen nicht.
- **Nicht jedes Tool ist packbar:** Tools mit offenen Verbindungen, Prozessen oder Locks lassen sich nicht serialisieren und fehlen nach dem Laden.
- **Pakete führen Code aus:** Beim Laden wird der enthaltene Code ausgeführt. Geladen wird nur aus dem eigenen `Agents`-Ordner; fremde Pakete dort nicht ablegen.
- **Frage-Erkennung ist eine Heuristik:** Gelb erscheint bei pausiertem Lauf oder wenn die Antwort mit `?` endet.
- **`Alt+Ziffer`** wird nicht von jedem Terminal weitergegeben; `/open n` funktioniert immer.
- **Referenzbilder** für das Design konnten nicht geöffnet werden; das Aussehen folgt dem Branding im Code.
- `/mount` und `/model` sind gegen nachgebaute isaa-Objekte getestet, nicht gegen das echte VFS. **Bedeutung:** das echte VFS (VFS-Layer des isaa-Mods, Berechtigungen, Mount-Pfade) wurde nicht mit echten Objekten geprüft - die Tests nutzen Nachbauten mit gleicher Schnittstelle.
- **Jobs:** gegen den echten `JobScheduler` aus dem Repo getestet; Networks, Rückblick und Resume nur gegen Nachbauten.
- **Job-Fehler:** Scheitert ein Job-Lauf, meldet die icli2 das dem Scheduler als Fehler (die alte icli meldete "completed").
- **`Umschalt+Enter`** kommt nicht in jedem Terminal an; `Strg+J` funktioniert überall.
- Beendete Job-/Dream-Läufe aus `on_cli_exit` sind nach dem Schliessen der UI nicht mehr sichtbar.
- **Schritt 8/9 von Subagenten geschrieben**, von mir gegengelesen und die genutzten Signaturen im Quellcode nachgeprüft. Getestet nur gegen Nachbauten: kein echter Login, kein echter LiveSync-Server, kein Mikrofon, kein TTS, kein Gemini Live.
- **Gerätewahl bei `/voice`:** wird NICHT dauerhaft gespeichert - gilt nur bis zum Beenden. **Geplant:** `voice_input_device`, `voice_output_device` in `<appdata>/icli2/settings.json`, beim Start automatisch anwenden; zusätzlich in `/settings` anpassbar.
- **`#audio`:** nutzt fest Groq-TTS mit Stimme "autumn" (wie die alte icli); Emotions-Erkennung vereinfacht. **Geplant:** `tts_provider`, `tts_voice` (+ Emotions-Schärfe) in `/settings` anpassbar.
- **Schritt 6/7 von Subagenten geschrieben.** `bridges.py` vollständig gegengelesen und gegen die echte Engine getestet; bei `features.py` und `connect.py` genutzte Signaturen nachgeprüft, nicht Zeile für Zeile gelesen.
- **Discord/Telegram nie real verbunden.** Die Erweiterungen selbst sind unverändert; geprüft ist nur, dass der Host-Ersatz genau die fünf Dinge bietet, die sie benutzen, und dass Läufe darüber in der UI erscheinen. Der Moderator-Agent (`discord_moderator` bzw. der Telegram-Moderator) muss in isaa existieren.
- **Über Discord/Telegram gestartete Läufe** kennen kein `#new-ctx`, kein Fortsetzen pausierter Läufe und keine aktiven Skills (in der alten icli vorhanden).
- **Features:** nur die sieben kleinen (desktop_auto, mini_web_auto, full_web_auto, isis_auto, coder, chain, execute); docs/autodoc/autotest/autofix fehlen. Abschalten wartet jetzt auf asynchrone Aufräumfunktionen (die alte icli tat das nicht).
- **Plugins:** Hooks laufen nicht (auch nicht nach Installation). Abschaltliste `<appdata>/icli/plugins_disabled.json`; alle anderen Plugins werden beim Start geladen wie bisher.
- **MCP:** neue Server werden nur gespeichert, wenn `agent.json` des Agenten existiert (liegt unter `data_dir/Agents/<name>/agent.json` und existiert immer nach dem Bauen).
- **Plugins (Marketplace):** Die alte icli benutzt `<appdata>/icli/plugins/.staging` direkt als Staging und löscht es komplett; eine dort angefangene icli2-Installation geht verloren, wenn parallel in der alten icli installiert wird. Die icli2 überschreibt nie ein installiertes Plugin (die alte icli schon). **Marketplace-Einträge mit relativem Pfad (`./plugins/x`):** der Pfad wird gegen das Arbeitsverzeichnis aufgelöst statt gegen die Wurzel des geklonten Marketplace-Repos - deshalb scheitert die Installation. Die alte icli hat dieselbe Schwäche. **Geplant:** relativen Pfad gegen die Clone-Wurzel auflösen. Bei `git-subdir` wird wie in der alten icli das ganze Repo geklont.
- **UI-Prüfskript** ist zeitabhängig und kann unter hoher Last einmal fehlschlagen; ein zweiter Lauf geht dann durch.

## Patches anwenden

```bash
git apply icli2-01-grundgeruest.patch
git apply icli2-02-branding-agenten.patch
git apply icli2-03-autopack-tools.patch
git apply icli2-04-modell-session-mount.patch
git apply icli2-05-laeufe-jobs-networks.patch
git apply icli2-08-09-cloudm-voice.patch
git apply icli2-06-07-features-discord.patch   # baut auf 08-09 auf
git apply icli2-10-e2e-settings-vfs.patch
git apply icli2-11-plugin-install.patch
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
