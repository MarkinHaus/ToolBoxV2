# Arbeitspaket WP2 — CI-Fehler kategorisieren, analysieren, Fix-Plan erstellen

> Stand: 2026-11-15, Basis `master` @ `ebed4d6`.
> **Auftrag, keine Ergebnisse.** Die Analyse macht ein Agent (z. B. ISAA lokal). Das Ergebnis ist ein Fix-Plan, den ein kleineres Modell Schritt für Schritt umsetzen kann, und für Architekturfragen ein Satz Lösungsvorschläge, den der Owner mit dem Planer durchdenkt.
> Läuft parallel zu WP1. In der Analysephase ändert WP2 **keinen Produktcode und keine Tests im Repo**. Experimente laufen lokal und werden danach zurückgesetzt. Committet werden nur Dokumente unter `docs/devdocs/ci_analysis/`.

---

## 1. Ziel

Die CI soll ein verlässliches Signal liefern: Rot heißt kaputt, Grün heißt geprüft. Heute gilt beides nicht:

- Die Jobs `Python 3.12` und `Python 3.13` (ubuntu, windows, macos) sind auf `master` rot. Unter 3.12/ubuntu waren es 541 Fehlschläge vor #14 und 531 im Pull Request. Gesammelt sind bisher nur die Fehler unter 3.12/ubuntu; die anderen Jobs fehlen noch.
- Ob `Python 3.10` und `3.11` zu Recht grün sind, ist **nicht belegt**. Siehe H0.

Am Ende von WP2 liegen vor:

1. ein **Katalog**: jede fehlschlagende Test-ID aus allen 12 Jobs genau einer Ursachen-Kategorie zugeordnet;
2. je Kategorie eine **getestete Hypothese** zur Ursache;
3. **Fix-Aufträge** für das kleinere Modell (Typ A und B, siehe 4.3), jeder mit genauen Dateien, Schritten und Abnahme;
4. **Architektur-Notizen** (Typ C) mit mindestens zwei Lösungswegen, ihren Vor- und Nachteilen und den offenen Fragen an den Owner; Grundlage für das Brainstorming.

## 2. Regeln

1. **Kein Grün durch Wegsehen.** Kein `skip`, `xfail`, `continue-on-error`, kein Löschen und kein Abschwächen von Tests als Fix-Vorschlag. Ausnahme nur, wenn der Test etwas prüft, das es nicht mehr gibt, und der Owner das bestätigt (dann Typ B).
2. **„Getestete Hypothese" heißt:**
   1. Ein Repro-Befehl lässt genau die betroffenen Tests lokal fehlschlagen, mit derselben Fehlermeldung wie in der CI.
   2. Die vorgeschlagene Änderung wird lokal probehalber angewendet, und derselbe Befehl wird grün.
   3. Die Nachbartests (gleiche Datei, gleiches Modul) werden dabei nicht rot.
   4. Die Änderung wird wieder zurückgesetzt (`git stash`/`git checkout -- .`). Befehl, Ausgabe vorher und Ausgabe nachher kommen ins Dokument.

   Geht ein Schritt nicht (z. B. Fehler nur unter macOS), heißt die Hypothese **UNGETESTET**, mit Grund. Nie „wahrscheinlich gleichwertig".
3. **Ursache vor Symptom.** Kategorisiert wird nach Ursache, nicht nach Datei. 40 Fehler in einer Datei können eine Ursache haben, ein Fehler kann zwei haben (dann die erste Ursache, die zweite als Notiz).
4. **Schutz der Entwicklungsumgebung:** Arbeit nur unter `C:\Users\Markin\Workspace\ISAABOX\Toolboxv2`. Für andere Python-Versionen eigene, isolierte venvs (`uv venv -p 3.12 .venv312`, `uv venv -p 3.13 .venv313`). Keine Daten der Owner-Instanz löschen, keine fremden Prozesse beenden, Ports 8467/8468 nicht belegen, wenn die Owner-Instanz läuft. Keine Geheimnisse in Dokumenten (auch keine `OPENAI_API_KEY`-Werte aus CI-Logs).
5. Branch: `claude/wizardly-cannon-o944j4`, Commits nur unter `docs/devdocs/ci_analysis/`. Nie `master`.

## 3. Bekannte Ausgangslage (nicht neu herleiten, aber bestätigen)

### 3.1 Wie die CI testet (`.github/workflows/build.yml`, Job `py-check`, Zeilen 102–154)

| Punkt | Ist | Warum wichtig |
|---|---|---|
| Matrix | 3.10–3.13 × ubuntu/windows/macos, `fail-fast: false` | 12 Jobs |
| Installation | `uv pip install -e toolboxv2/mods/isaa --system`, dann `uv pip install . --system` (**nicht editierbar**) | Tests laufen aus dem Checkout, importiert wird evtl. aus `site-packages`. Zwei Stände von `toolboxv2` möglich |
| Testbefehl | `python -m pytest toolboxv2/tests/ -v --tb=short \|\| python -m unittest discover -s toolboxv2/tests/` | **Fällt pytest durch, entscheidet allein `unittest discover`.** Siehe H0 |
| Umgebung | `MOCK_EXTERNAL_SERVICES=true`, `IS_OFFLINE_DB=true`, `OPENAI_API_KEY` aus Secrets | Tests, die externe Dienste brauchen, sollten ohne Netz laufen |
| Zusatzpakete | nur `pytest pytest-asyncio Cython` | Optionale Module wie `pyautogui` fehlen |

### 3.2 Erste Zahlen (Python 3.12, ubuntu, PR #14 vor `91d984a`)

523 fehlschlagende Test-IDs in 65 Dateien. Rohliste: `docs/devdocs/ci_analysis/input_py312_ubuntu_pr14.txt`.

| Bereich | Anzahl |
|---|---|
| `tests/test_isaa` | 401 |
| `tests/test_mini` | 79 |
| `tests/test_clis` | 28 |
| `tests/test_core` | 12 |
| `tests/test_web` | 4 |

Die größten Dateien: `test_jobs_extended.py` 43, `test_job_manager.py` 40, `test_agent_fetures.py` 30, `test_minicli_macros.py` 26, `test_mcp/test_e2.py` 23, `test_execution_engine.py` 23, `test_mcp/test_server.py` 22, `test_vfs_v2.py` 19, `test_omni.py` 18, `test_dreamer_aggregator.py` 18, `test_icli_overlay.py` 16, `test_mini/test_toolboxv2.py` 13, `test_LiveSync_server.py` 13, `test_system/test_session.py` 12, `test_LiveSync_index.py` 12, `test_mkdocs.py` 11, `test_core/test_wizard_e2e_http.py` 8.

### 3.3 Beobachtete Fehlerbilder (Stichprobe, noch keine Kategorien)

- Event-Loop-Fehler in async-Tests (typisch für 3.12+: `asyncio.get_event_loop()` ohne laufende Schleife, Zusammenspiel mit `pytest-asyncio`).
- Geänderte Schnittstelle des MCP-Servers (Tests rufen alte Signaturen auf).
- Fehlende optionale Module (`pyautogui` u. a.).
- Fehlendes Modul `toolboxv2.flows.minicli` (betrifft vermutlich alle 26 Fälle in `test_minicli_macros.py`).
- `test_shell_tool::TestKillProcess::test_killing_the_harness_grandparent_does_not_kill_the_harness`: lokal grün, nur in der CI rot (andere Prozesskette auf dem Runner).
- `test_mini/test_toolboxv2.py::TestToolboxv2Mods::test_01*`: `ERROR` schon im Setup, also eine gemeinsame Ursache im `setUpClass`.

## 4. Vorgehen

### Schritt 0 — Hypothese H0 zuerst: Ist das CI-Signal echt?

**H0:** Die Jobs 3.10/3.11 sind grün, weil pytest dort durchfällt und danach `unittest discover` erfolgreich ist. In dem Fall würde der Fallback in `build.yml:154` die Fehler verdecken.

- Prüfen: Für einen 3.10- und einen 3.11-Job auf `ebed4d6` das Log lesen. Zählen, wie viele Tests pytest dort fehlschlagen lässt, und ob danach `unittest discover` lief.
- Prüfen: Laufen Tests gegen den Checkout oder gegen `site-packages`? Ein Test, der `toolboxv2.__file__` ausgibt, reicht als Beleg (nur lokal, nicht committen).
- Ergebnis als eigene Karte (siehe 4.4). Bestätigt sich H0, ist das der erste Fix-Auftrag, und die Zahlen für 3.10/3.11 kommen in den Katalog.

### Schritt 1 — Sammeln

Für **alle 12** `py-check`-Jobs des Laufs auf `ebed4d6` die fehlschlagenden Test-IDs mit Fehlertyp und erster Fehlerzeile sammeln (Job-Logs). Eine Datei je Job: `docs/devdocs/ci_analysis/raw/<os>-<py>.txt` mit dem Format `ID | FAILED/ERROR | Exception-Typ | erste Zeile der Meldung`.

Danach eine Matrix `Test-ID × Job`. Daraus ergibt sich, was **überall** fehlschlägt (Code oder Test), was **versionsabhängig** ist (3.12+), was **plattformabhängig** ist und was **sporadisch** ist (zwei Läufe vergleichen; der Lauf vor `91d984a` liegt als Rohliste vor).

### Schritt 2 — Kategorisieren

Jede ID bekommt genau eine Kategorie. Startliste; neue Kategorien sind erlaubt, wenn begründet:

| Kennung | Kategorie |
|---|---|
| K-ENV | CI-Umgebung: Installationsart, fehlende Pakete, Pfade, Secrets, `MOCK_*` wirkt nicht |
| K-PY | Python-Version: Verhalten ab 3.12/3.13 geändert (asyncio, entfernte APIs, Warnungen als Fehler) |
| K-OS | Plattform: Prozesse, Signale, Pfadtrenner, Kodierung (cp1252), Dateisperren |
| K-ASYNC | Test-Infrastruktur für async: pytest-asyncio-Modus, Schleifen zwischen Tests, unittest vs. pytest |
| K-API | Test prüft eine **beabsichtigt** geänderte Schnittstelle (Test veraltet) |
| K-BUG | echter Fehler im Produktcode |
| K-MISS | Test importiert Code, den es nicht (mehr) gibt |
| K-STATE | Reihenfolge oder globaler Zustand: Singleton `get_app`, gemeinsame `.data`, Ports, Umgebungsvariablen |
| K-NET | braucht Netz oder externe Dienste trotz Mock |
| K-FLAKY | Zeitabhängig, Rennen, Zufall |

### Schritt 3 — Je Kategorie: Ursache belegen, Fix-Typ bestimmen

| Typ | Bedeutung | Wer setzt um |
|---|---|---|
| **A — mechanisch** | eindeutige, örtlich begrenzte Änderung (Import, Fixture, Konfiguration, kleine Code-Korrektur). Hypothese getestet | kleineres Modell |
| **B — Entscheidung nötig** | Fix ist klar, hängt aber an einer Frage an den Owner (z. B. „Ist die neue MCP-Signatur gewollt? Dann Tests anpassen") | kleineres Modell nach Antwort |
| **C — Architektur** | Ursache liegt im Aufbau, z. B. globaler `get_app`-Singleton in Tests, zwei Test-Runner gleichzeitig, ISAA als Unterpaket mit eigener Installation, Event-Loop-Besitz in Bibliothekscode. Kein Einzel-Fix | Owner + Planer, Brainstorming |

Für Typ C: **nicht reparieren.** Stattdessen eine Notiz mit:

1. **Befund:** Was ist der Aufbau heute (Dateien, Aufrufkette)? Welche Tests und welches Produktverhalten hängen daran?
2. **Mindestens zwei Lösungswege**, je mit Aufwand (S/M/L), Risiko, betroffenen Modulen, Migrationsweg und dem Teil, den man sofort als Typ A machen könnte.
3. **Ein Experiment je Weg**, das die Machbarkeit zeigt (Spike, lokal, nicht committet), oder die Begründung, warum es keines gibt.
4. **Fragen an den Owner**, die die Wahl entscheiden.

### Schritt 4 — Dokumente schreiben

Alles unter `docs/devdocs/ci_analysis/`:

| Datei | Inhalt |
|---|---|
| `README.md` | Übersicht: Zahlen je Job vorher, Kategorien mit Anzahl, Typ-A/B/C-Liste mit Reihenfolge, offene Fragen an den Owner |
| `catalog.csv` | `test_id;job_list;kategorie;karte;fix_typ` — jede ID genau einmal |
| `cards/<kategorie>-<nr>.md` | eine **Karte** je Ursache (Format unten) |
| `fixes/FIX-<nr>.md` | ein **Fix-Auftrag** je Typ-A/B-Karte, für das kleinere Modell (Format unten) |
| `architecture/ARCH-<nr>.md` | eine Notiz je Typ-C-Ursache |
| `raw/` | Rohlisten je Job |

**Format Karte:**

```
Karte:       K-PY-01
Ursache:     <ein Satz>
Betroffen:   <n> Tests in <m> Jobs (Liste in catalog.csv)
Beleg:       <Datei:Zeile im Produktcode bzw. Test, CI-Fehlerzeile wörtlich>
Hypothese:   <was genau falsch ist>
Repro:       <Befehl, wörtlich>
Vorher:      <Ausgabe, wörtlich, gekürzt>
Probe-Fix:   <Diff, kurz>
Nachher:     <Ausgabe, wörtlich, gekürzt>
Nachbarn:    <Befehl + Ergebnis>
Status:      GETESTET | UNGETESTET (<Grund>)
Fix-Typ:     A | B (<Frage>) | C (-> ARCH-<nr>)
```

**Format Fix-Auftrag** (so geschrieben, dass ein kleineres Modell ihn ohne weitere Analyse umsetzen kann):

```
Fix:          FIX-03  (aus Karte K-PY-01)
Voraussetzung: <andere FIX-Nummern oder keine>
Dateien:      <genaue Pfade; nur diese ändern>
Vorher lesen: <Dateien/Funktionen, Aufrufer>
Schritte:     1. ... 2. ... (konkret, mit Codeausschnitt wo nötig)
Nicht tun:    <was ausdrücklich verboten ist>
Abnahme:      <Befehl>  -> diese Test-IDs grün: ...
              <Gate-Befehl> -> grün, keine neue rote ID gegenüber catalog.csv
Rückfall:     wenn rot: stoppen, Ausgabe melden
```

### Schritt 5 — Reihenfolge des Fix-Plans

Im `README.md` eine Reihenfolge vorschlagen und begründen. Grundsatz:

1. **Signal zuerst:** H0 bzw. K-ENV (sonst misst jeder spätere Fix falsch).
2. **Breite Ursachen vor einzelnen:** ein Fix für 100 IDs vor 10 Fixes für je 1.
3. **Produktpfade vor Experimentellem:** Kern, CLI, Wizard (`test_wizard_e2e_http` betrifft WP1-K3), CloudM-Auth, Session, LiveSync vor ISAA-Jobs, Audio, Dreamer.
4. Typ-C-Ursachen blockieren die von ihnen abhängigen Typ-A-Fixes nicht, wenn sich der Typ-A-Teil sauber abgrenzen lässt.

## 5. Abschlusskriterium WP2 (Analysephase)

| # | Kriterium |
|---|---|
| A1 | H0 beantwortet, mit Log-Zeilen aus einem 3.10- und einem 3.11-Job |
| A2 | `raw/` enthält alle 12 Jobs von `ebed4d6` |
| A3 | `catalog.csv`: jede fehlschlagende ID genau einmal, keine ohne Kategorie (Zählung im README: Summe = Summe der Rohlisten, ohne Doppelte) |
| A4 | Jede Karte hat Status GETESTET oder UNGETESTET mit Grund; mindestens 80 % der IDs liegen in GETESTET-Karten |
| A5 | Jede Typ-A/B-Karte hat einen Fix-Auftrag im Format oben |
| A6 | Jede Typ-C-Ursache hat eine Notiz mit mindestens zwei Wegen, Experiment oder Begründung und Fragen an den Owner |
| A7 | Arbeitsbaum sauber; Produktcode und Tests unverändert (`git diff --stat master -- toolboxv2 .github` leer) |

Danach **stoppen** und dem Owner melden: Zahlen, Typ-C-Liste, Fragen. Die Umsetzungsphase (kleineres Modell) startet erst nach seiner Freigabe der Reihenfolge und der Typ-B-Antworten.

## 6. Abschlusskriterium der späteren Umsetzungsphase (zur Orientierung)

- Jeder Fix-Auftrag ein Commit mit der Abnahme im Bericht.
- Kein Test übersprungen oder abgeschwächt (Diff-Prüfung auf `skip`, `xfail`, `continue-on-error`).
- Ziel: `py-check` auf allen 12 Jobs grün **ohne** den `|| unittest discover`-Fallback, oder mit einer begründeten, vom Owner akzeptierten Restliste aus Typ-C-Ursachen.
