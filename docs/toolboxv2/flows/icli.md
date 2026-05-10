# ISAA Host CLI (icli) — Benutzerhandbuch

`tb -m cli` startet die BeastCLI (Makros, Module). `tb` im ISAA-Kontext öffnet den
**ISAA Host** — ein eigenständiges Multi-Agent-System mit Echtzeit-Monitoring,
VFS, Audio-Eingabe und Job-Scheduler.

```bash
tb -m icli           # Normaler Start
```

---

## Konzept in 30 Sekunden

```
Du tippst Aufgaben  →  Self Agent denkt nach  →  spawnt Sub-Agents
                                     ↓
         Footer-Toolbar zeigt Fortschritt aller Tasks live
                                     ↓
              F2 öffnet ZEN+ Fullscreen-Overlay für Details
```

Der **Self Agent** ist der einzige Agent der direkten Shell-Zugriff hat.
Alle anderen Agents operieren in der VFS (virtuelles Dateisystem) und dürfen
nur über definierte Tools mit dem System kommunizieren.

---

## Prompt-Eingabe

Direkte Texteingabe → geht an den aktiven Agenten:

```
> Analysiere alle Python-Dateien in /src und erstelle eine Übersicht
```

Slash-Befehle steuern den Host (kein LLM-Aufruf):

```
> /agent list
> /vfs mount ./mein-projekt /proj
> /task view
```

Sonderzeichen am Eingabe-Anfang:

| Prefix | Wirkung |
|---|---|
| `/` | Host-Befehl |
| `#audio` am Ende | Einmalige Audio-Antwort |
| normaler Text | Geht an aktiven Agenten |

---

## F-Keys — Immer verfügbar

| Taste | Aktion |
|---|---|
| **F2** | ZEN+ Fullscreen-Overlay öffnen / schließen |
| **F4** | Audioaufnahme starten / stoppen |
| **F5** | Status-Dashboard: Agents, VFS, Skills, Session |
| **F6** | Fokus auf laufenden Task setzen / aufheben |
| **F7** | Zwischen laufenden Tasks wechseln (cycle) |
| **F8** | Fokussierten Task abbrechen |

**F6 + F7** sind das wichtigste Duo: wenn mehrere Tasks gleichzeitig laufen,
zeigt der Footer nur den fokussierten Task fett an. F7 wechselt durch alle.

---

## ZEN+ Overlay (F2) — Vollbild-Monitor

Öffnet einen Fullscreen-View mit zwei Panels.

```
┌─ Tasks ─────────────┬──────────────────────────────────────────────┐
│                     │                                              │
│  ▸ ● self           │  ● self  Analysiere die Docs...             │
│    ✦ sub_0d9b1fb7   │  Persona: fallback_analyst                  │
│    ✦ sub_b700359f   │  Tokens: [████████░░░░░░░░░░░░] 7%          │
│      sub_ae462d6d   │  ━━━━━━━━━━━━━━━━━━━━  iter 15/15          │
│                     │  ─────────────────────────────────────────  │
│                     │  ── iter 15 ▸ running ─────────────────    │
│  ↑↓=nav →/Enter=r  │  ◇ vfs_shell          ✓    0.00s           │
│                     │  ── iter 12 ────────────────────────────    │
└─────────────────────┴──────────────────────────────────────────────┘
  Tab=fokus links/rechts  Enter=iter drill  ←=zurück  Esc=schließen
```

### Navigation im Overlay

**Linkes Panel** (fokussiert mit Tab oder ←):

| Taste | Aktion |
|---|---|
| `↑` / `↓` | Task oder Sub-Agent auswählen |
| `→` / `Enter` | Fokus ins rechte Panel wechseln |

Sub-Agents erscheinen eingerückt unter ihrem Parent-Task mit `✦`. Wenn ein
Sub-Agent einen eigenen TaskView hat (eigene Iteration), wird sein Status
direkt angezeigt.

**Rechtes Panel** (fokussiert mit → oder Tab):

| Taste | Aktion |
|---|---|
| `↑` / `↓` | Einzelne Zeilen scrollen |
| `j` / `k` | 5 Zeilen scrollen |
| `PgDn` / `PgUp` | 5 Zeilen scrollen |
| `Enter` | Sichtbare Iteration im Detail öffnen (Drill-Down) |
| `1`–`9` | Direkt zu Iteration N springen |
| `←` / `Backspace` | Aus Drill-Down zurück zur Liste |

**Drill-Down Modus** (nach Enter auf einer Iteration):
Zeigt Thoughts und Tool-Ergebnisse ohne Truncating — volle Inhalte,
echte Zeilenumbrüche. Finale Antwort des Agenten am Ende vollständig.

---

## Agent Management

```bash
/agent list                      # Alle Agenten anzeigen
/agent switch <name>             # Aktiven Agenten wechseln
/agent spawn <name> <persona>    # Neuen Agenten erstellen
/agent stop <name>               # Tasks eines Agenten stoppen
/agent model fast <model>        # LLM-Modell live wechseln
/agent model complex <model>     # Komplexes Modell wechseln
/agent checkpoint save [name]    # Zustand speichern
/agent checkpoint load [name]    # Zustand laden
/agent stats [name]              # Token-Verbrauch und Kosten
/agent delete <name>             # Agenten und Daten löschen
```

### Modelle wechseln

Alle verfügbaren Kürzel:

```bash
/agent model fast gemini-3-flash     # Schnell + billig
/agent model complex sonnet-4.6      # Anthropic Claude
/agent model fast deepseek-v3.2      # DeepSeek
/agent model fast glm-4.7-flash      # Z-AI
/agent model fast qwen3.5-27b        # Qwen via OpenRouter
```

Lokale Modelle via Ollama:

```bash
/agent model fast qwen3_14b          # ollama/qwen3:14b
/agent model fast deepseek-r1_8b     # ollama/deepseek-r1:8b
```

---

## Session Management

```bash
/session list            # Alle Sessions anzeigen
/session new             # Neue Session starten (leere History)
/session switch <id>     # Session wechseln
/session show            # Letzte 10 Nachrichten anzeigen
/session show 25         # Letzte 25 Nachrichten
/session clear           # Session-History löschen
```

Sessions sind persistent. Wenn du die icli neu startest und die gleiche
Session-ID verwendest, ist der gesamte Kontext des Agenten wiederhergestellt.

---

## VFS — Virtuelles Dateisystem

Das VFS ist der primäre Arbeitsbereich aller Agents. Es isoliert Agent-Aktivität
vom echten Dateisystem bis du explizit synchronisierst.

```bash
/vfs                          # VFS-Baum anzeigen
/vfs /projekt/src/main.py     # Datei-Inhalt anzeigen
/vfs mount ./mein-projekt /proj   # Lokalen Ordner mounten
/vfs mount ./mein-projekt /proj --readonly   # Nur-Lesen
/vfs mount ./mein-projekt /proj --no-sync    # Manueller Sync
/vfs unmount /proj            # Mount entfernen
/vfs sync /proj               # Änderungen auf Disk schreiben
/vfs save /proj/out.py ./out.py   # Einzeldatei lokal speichern
/vfs pull /proj               # Änderungen von Disk nachladen
/vfs refresh /proj            # Mount neu einlesen
/vfs mounts                   # Aktive Mounts auflisten
/vfs dirty                    # Geänderte Dateien anzeigen (nicht gespeichert)
/vfs rm /proj/alt.py          # Datei/Verzeichnis entfernen
```

### System-Dateien (Read-Only für Agents)

Nützlich um Referenz-Dateien bereitzustellen die Agents lesen aber nicht
verändern können:

```bash
/vfs sys-add ./README.md /sys/readme       # Als System-Datei hinzufügen
/vfs sys-list                               # Alle System-Dateien
/vfs sys-refresh /sys/readme               # Neu einlesen
/vfs sys-remove /sys/readme                # Entfernen
```

### Tab-Completion im VFS

Alle `/vfs`-Pfade haben hierarchische Tab-Completion:

```
/vfs /pro[TAB]     → zeigt alle /proj*
/vfs /proj/[TAB]   → zeigt Kinder von /proj
```

Geänderte Dateien (`✱`) erscheinen zuerst in der Completion.

---

## Tasks im Hintergrund

Jede Agent-Anfrage erzeugt einen Task. Mehrere Tasks laufen parallel.

```bash
/task                    # Alle Tasks anzeigen
/task view               # Fullscreen-View (wie F2)
/task view <id>          # Bestimmten Task anzeigen
/task cancel <id>        # Task abbrechen
/task clean              # Abgeschlossene Tasks aufräumen
```

### Paralleles Arbeiten — optimaler Workflow

```
1. Aufgabe A an Agenten schicken → Task startet im Hintergrund
2. F6 → Task A bekommt Fokus (Footer zeigt Details)
3. Neue Aufgabe B tippen → Task B startet parallel
4. F7 → Fokus wechselt zu Task B
5. F2 → Beide Tasks im Overlay vergleichen
```

Der fokussierte Task streamt seine Ausgabe direkt ins Terminal.
Nicht-fokussierte Tasks laufen still im Hintergrund — sichtbar im Footer.

---

## Features — Agent-Fähigkeiten erweitern

```bash
/feature list                      # Verfügbare Features
/feature enable coder              # Code-Editor Toolkit (SEARCH/REPLACE)
/feature enable docs               # Dokumentations-System
/feature enable desktop_auto       # Desktop GUI Automation
/feature enable mini_web_auto      # Headless Browser (Playwright)
/feature enable full_web_auto      # Vollständiger Browser
/feature enable chain              # Agent-zu-Agent Chains
/feature disable coder             # Feature deaktivieren
```

Features werden zur Laufzeit aktiviert — kein Neustart nötig.

### Wann welches Feature?

| Feature | Wann aktivieren |
|---|---|
| `coder` | Code-Dateien bearbeiten, Git-Worktrees, SEARCH/REPLACE |
| `docs` | Dokumentation lesen/schreiben, Code-Lookup |
| `mini_web_auto` | Webseiten scrapen, Formulare ausfüllen (headless) |
| `full_web_auto` | Interaktiver Browser (sichtbar) |
| `desktop_auto` | GUI-Anwendungen steuern |
| `chain` | Mehrere Agents koordinieren |

---

## MCP — Model Context Protocol

Externe Tool-Server dynamisch anschließen:

```bash
/mcp list                           # Aktive MCP-Verbindungen
/mcp add filesystem npx @mcp/fs    # MCP-Server hinzufügen
/mcp add myserver python server.py  # Eigenen Python-Server
/mcp remove filesystem              # Trennen + Tools entfernen
/mcp reload                         # Alle MCP-Tools neu indizieren
```

Tools des MCP-Servers sind sofort nach `/mcp add` verfügbar.

---

## Tools — Feinsteuerung

```bash
/tools list                   # Alle Tools des aktiven Agenten
/tools list vfs               # Nur VFS-Kategorie
/tools info                   # Details zu allen Tools
/tools enable vfs_shell       # Einzelnes Tool aktivieren
/tools disable vfs_shell      # Einzelnes Tool deaktivieren
/tools enable-all             # Alle Tools aktivieren
/tools disable-all            # Alle nicht-System-Tools deaktivieren
```

---

## Skills — Agenten-Wissen

Skills sind gespeicherte Verhaltensanweisungen die der Agent automatisch
anwendet wenn die Situation passt.

```bash
/skill list                        # Skills des aktiven Agenten
/skill list --inactive             # Inaktive Skills
/skill show <id>                   # Inhalt eines Skills
/skill edit <id>                   # Skill bearbeiten
/skill delete <id>                 # Skill löschen
/skill merge <keep_id> <rm_id>     # Zwei Skills zusammenführen
/skill boost <id> 0.3              # Relevanz erhöhen
/skill import ./my_skills/         # Skills aus Verzeichnis importieren
/skill export <id> ./export/       # Skill exportieren
```

---

## Jobs — Automatisierung & Zeitplanung

```bash
/job list                     # Alle geplanten Jobs
/job add                      # Neuen Job erstellen (interaktiv)
/job remove <id>              # Job löschen
/job pause <id>               # Job pausieren
/job resume <id>              # Job fortsetzen
/job fire <id>                # Job sofort auslösen
/job detail <id>              # Job-Details anzeigen
```

### Dreamer — Nachtverarbeitung

Der Dreamer verarbeitet Erfahrungen und destilliert neue Skills:

```bash
/job dream create             # Nacht-Job für Self Agent erstellen (03:00)
/job dream create analyst     # Für spezifischen Agenten
/job dream status             # Alle Dream-Jobs anzeigen
/job dream live               # Jetzt ausführen mit Live-Visualisierung
```

### OS Auto-Wake (Server-Modus)

```bash
/job autowake install         # OS weckt System für Jobs auf
/job autowake remove          # Auto-Wake deaktivieren
/job autowake status          # Status prüfen
```

---

## Audio

```bash
/audio on                     # Audio-Antworten immer aktiviert
/audio off                    # Deaktivieren
/audio voice <name>           # Stimme wählen
/audio backend groq            # Groq Whisper STT
/audio backend piper           # Lokaler Piper TTS
/audio backend elevenlabs      # ElevenLabs TTS
/audio stop                   # Aktuelle Wiedergabe stoppen
```

**Einmalige Audio-Antwort** ohne globales `/audio on`:

```
> Erkläre mir diesen Code #audio
```

**F4** während Eingabe: Mikrofon-Aufnahme starten → F4 nochmal → Transcription
wird automatisch als Eingabe übernommen.

---

## Bind & Teach — Agent-Koordination

```bash
/bind self analyst             # Self Agent delegiert an Analyst
/teach analyst code_review     # Skill "code_review" auf Analyst übertragen
/context stats                 # Context-Fenster-Auslastung anzeigen
```

---

## Single-Run Modus (Non-Interaktiv)

Für Scripts und CI:

```bash
# Einmalige Anfrage, dann Exit
tb icli "Analysiere ./src und liste alle public APIs" --agent self

# Mit spezifischem Modell
tb icli "Refaktoriere main.py" --agent self --model gemini-3-flash

# Mit Feature
tb icli "Lese die README und erstelle eine Zusammenfassung" \
  --feature docs --session analysis_run_1

# Mit MCP
tb icli "Liste alle Dateien" \
  --mcp '{"name": "fs", "command": "npx", "args": ["@mcp/fs"]}'
```

`--remember <session_id>` speichert die History. Ohne `--remember` wird
sie nach dem Run gelöscht.

---

## Optimale Workflows

### Großes Refactoring

```
1. /vfs mount ./mein-projekt /proj
2. /feature enable coder
3. /feature enable docs
4. "Erstelle einen Plan für das Refactoring von /proj/src"
   → Agent plant, legt Plan in VFS ab
5. F6 → Fokus auf Plan-Task
6. Wenn done: "Führe den Plan aus, beginne mit Schritt 1"
   → Agent startet, spawnt Sub-Agents für Teilaufgaben
7. F2 → Sub-Agents im Overlay beobachten
8. Wenn fertig: /vfs sync /proj
```

### Parallele Analyse mehrerer Codebasen

```
1. /vfs mount ./projekt-a /a
2. /vfs mount ./projekt-b /b
3. "Analysiere /a und erstelle Bericht" → Task A startet
4. F6 zum Defokussieren
5. /agent spawn analyst_b fallback_analyst
6. /agent switch analyst_b
7. "Analysiere /b und erstelle Bericht" → Task B startet parallel
8. F7 → zwischen Tasks wechseln
9. F2 → beide im Overlay vergleichen
```

### Automatisierte Nacht-Routine

```
1. /job dream create          → Dreamer lernt aus heutigen Sessions
2. /job add                   → Eigenen Job konfigurieren (Cron-Syntax)
3. /job autowake install      → OS weckt Server um 3:00 Uhr
4. /job list                  → Überblick
```

### Debugging mit Audio

```
1. /audio on
2. /audio backend groq
3. F4 → sprechen → F4 → Aufnahme endet → Transcription erscheint
4. Agent antwortet → Antwort wird vorgelesen
```

---

## Footer-Toolbar verstehen

```
▸ ◯ self       ━━━━━━━━─── 15/15   fallback..   7% ◎ im ordner tb/docs...
▸ ● analyst    ────────────  0/0                    ● done 4m32s
```

| Symbol | Bedeutung |
|---|---|
| `▸` | Fokussierter Task |
| `◯` | Laufend |
| `●` | Abgeschlossen |
| `✗` | Fehler |
| `⏸` | Abgebrochen |
| `━━━━────` | Iterations-Fortschrittsbalken |
| `15/15` | Aktuelle / Max Iterationen |
| `fallback..` | Persona (gekürzt) |
| `7%` | Token-Verbrauch |
| `◎ text` | Aktueller Gedanke des Agenten |
| `◇ vfs_shell ✓` | Zuletzt ausgeführtes Tool |
| `✦2` | Anzahl aktiver Sub-Agents |

---

## Häufige Probleme

**Agent antwortet nicht mehr** → F8 (fokussierten Task abbrechen) → neu starten

**Footer zu voll** → F2 öffnen, dort navigieren statt im Footer lesen

**VFS-Änderungen verloren** → `/vfs dirty` zeigt was noch nicht gespeichert ist,
dann `/vfs sync`

**Falsches Modell aktiv** → `/agent stats` zeigt aktuelles Modell,
`/agent model fast <kürzel>` wechselt es sofort

**Sub-Agent hängt** → In F2-Overlay: Sub-Agent links auswählen, F8 drücken
(betrifft nur den fokussierten Task)
