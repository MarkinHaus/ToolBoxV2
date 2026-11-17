# icli 2.0 – Plan

Stand: Schritt 5 von 9 umgesetzt. Die alte `flows/isaa/icli.py` bleibt unverändert und parallel nutzbar.

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
| `/login` | Schritt 8 | – |
| `/voice` | Schritt 9 | `/audio`, `/omni` |

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

## Offen

### Schritt 6 – Skills, MCP, Plugins, Features
- Ein Picker je Bereich mit An/Aus-Schalter statt Subcommands.
- Feature-Lader aus der alten icli (`load_*_feature`) wiederverwenden; dafür ggf. aus `icli.py` in ein eigenes Modul auslagern (Kopie in neue Datei, alte Datei bleibt).
- Aktive Features in der Kopfzeile.

### Schritt 7 – Discord (und Telegram)
- Ziel: `patch_cli_for_discord(host)` und `DiscordCLIExtension` laufen unverändert.
- Die Integration braucht vom Host nur: `isaa_tools`, `active_agent_name`, `_handle_command`, `_build_completer`, `run_agent_monitored`. Dafür ein schlanker Host-Adapter im Engine-Thread, der genau diese fünf Punkte bereitstellt.
- `/discord …` wird unverändert an die Extension durchgereicht; ihre Textausgabe landet als Hinweis im Chat.
- Discord-Läufe erscheinen als normale Läufe in der Seitenleiste.
- Telegram funktioniert nach demselben Muster (`TelegramCliExtension`).
- Risiko: Die Extension gibt per `print` aus; die Ausgabe muss in der Engine abgefangen werden.

### Schritt 8 – CloudM
- Anmeldestatus in der Kopfzeile, `/login` über die vorhandene CloudM-Auth.
- `/mount` zeigt zusätzlich per LiveSync geteilte Ordner (`CloudM/LiveSync/vfs_adapter.py`).
- ToolBox-Mod-Funktionen aus der Palette aufrufen, Ergebnis im Chat.

### Schritt 9 – Audio und Omni
- `F4` Spracheingabe, Vorlesen, Omni-Session; alles erst beim ersten Gebrauch geladen.
- Pegelanzeige in der Fußzeile.

### Danach
- Engine als eigener Prozess (Läufe überleben `/q`, erneutes Verbinden).
- Web-Brücke (`icli_web`) an das neue Protokoll anschließen.
- Automatisierte Tests ins Repo übernehmen.

## Bekannte Lücken und Risiken

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

## Patches anwenden

```bash
git apply icli2-01-grundgeruest.patch
git apply icli2-02-branding-agenten.patch
git apply icli2-03-autopack-tools.patch
git apply icli2-04-modell-session-mount.patch
git apply icli2-05-laeufe-jobs-networks.patch
pip install "textual>=8"
```

## Mini-Prompt zum Fortfahren

```
Lies toolboxv2/mods/isaa/extras/icli2/PLAN.md und setze den nächsten offenen Schritt um.
Regeln aus dem Plan einhalten (UI wartet auf nichts, nur backend.py kennt isaa,
vorhandene Funktionen nutzen, alte icli.py nicht ändern). Headless testen,
PLAN.md aktualisieren, committen, pushen und mir den Schritt als .patch
(reiner Diff, aufbauend auf den vorherigen) schicken.
```
