# icli 2.0 – Plan

Stand: Schritt 3 von 9 umgesetzt. Die alte `flows/isaa/icli.py` bleibt unverändert und parallel nutzbar.

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
| `/model` | Schritt 4 | `/agent model` |
| `/session` | Schritt 4 | `/session new/clear/show/working` |
| `/mount [pfad]` | Schritt 4 | `/vfs mount/unmount/save` |
| `/task` | Schritt 5 | `/task log/view`, Fokuswechsel |
| `/job` | Schritt 5 | `/job`, `/job dream` |
| `/feature`, `/skill`, `/mcp`, `/plugin` | Schritt 6 | gleichnamige Befehle, über Palette |
| `/discord` | Schritt 7 | unverändert durchgereicht |
| `/login` | Schritt 8 | – |
| `/voice` | Schritt 9 | `/audio`, `/omni` |

Tasten: `Esc` pausieren, `Esc Esc` abbrechen, `Strg+O` alle Tools auf/zu, `Strg+R` Gedanken auf/zu, `Alt+1–9` letzte n Tools öffnen, `Strg+L` leeren, `Strg+Q` beenden. Geplant: `Strg+P` Palette, `F2` Seitenleiste, `F4` Sprache.

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

## Offen

### Schritt 4 – Modell, Session, Mount, Palette
- `/model`: Picker; Modellwechsel über `isaa.get_agent(name, model_override=…)`. Modellliste aus den Quellen der alten icli (`load_gateway_models`, `fetch_openrouter_models`) – vorher prüfen, ob sie ohne `icli.py`-Import nutzbar sind, sonst in ein gemeinsames Modul verschieben.
- `/session`: Picker über `agent.session_manager`; neu, wechseln, leeren. Verlauf der gewählten Session beim Wechsel anzeigen.
- `/mount pfad`: lokalen Ordner in das VFS der Session einhängen, ohne Argument Liste mit Aushängen/Speichern. Nutzt `vfs_v2`.
- Palette `Strg+P`: alle Befehle plus Fuzzy-Suche.
- Mehrzeilige Eingabe (`TextArea`, Enter senden, Umschalt+Enter Zeile) und Eingabe-History.

### Schritt 5 – Tasks, Jobs, Parallel-Läufe, Networks
- Seitenleiste `F2`: laufende und beendete Läufe, Jobs, Mounts, Kontext-Auslastung.
- Fokuswechsel zwischen Läufen; Lauf in den Hintergrund schieben; Stream je Lauf getrennt puffern.
- Sub-Agenten-Chunks (`_sub_agent_id`) als eingerückte Unterläufe statt wie bisher verworfen.
- Jobs über den vorhandenen `isaa.job_scheduler` inklusive `dream`.
- Agent-Networks: `export_agent_network`/`import_agent_network` in den Agent-Picker; ein Network erscheint als Gruppe.
- Paused-Läufe nach Neustart fortsetzen (`resume_from_disk`).

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
- Eingabe ist noch einzeilig, Sub-Agenten-Ausgaben werden noch verworfen.

## Patches anwenden

```bash
git apply icli2-01-grundgeruest.patch
git apply icli2-02-branding-agenten.patch
git apply icli2-03-autopack-tools.patch
pip install "textual>=8"
```
