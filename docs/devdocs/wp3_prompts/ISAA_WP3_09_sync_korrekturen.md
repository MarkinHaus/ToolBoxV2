# ISAA-Auftrag WP3-09 – Sync-Korrekturen: Konfliktkopie, Symlinks, Löschungen, 401, Dateinamen mit `:`

> Basis: `master` @ `dca1ffa` (geprüft 2026-11-18). Welle 1, läuft parallel zu WP3-05 (getrennte Dateien). WP3-08 und WP3-10 bauen danach auf deinem Stand auf.
> Abnahme: `toolboxv2/tests/test_core/test_a4_betriebsbefunde.py` Nr. 3, 4, 5, 6 grün (Nr. 7 gehört WP3-08 und bleibt rot), dazu ein neuer Wheel-Test für Nr. 2.

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
git worktree add -B isaa/09 C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa09 origin/master
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa09
uv venv -p 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install pytest pytest-asyncio pytest-timeout build
uv pip install -e toolboxv2/mods/isaa
uv pip install -e ".[web]"
$env:MOCK_EXTERNAL_SERVICES = "true"; $env:IS_OFFLINE_DB = "true"
python -c "import toolboxv2; print(toolboxv2.__file__)"   # muss in ...Toolboxv2-isaa09 liegen
```

Nichts von Hand in die venv installieren, was nicht in `pyproject.toml` steht. Sonst misst die Gegenprobe deine Umgebung statt des Codes (Testleitfaden, Fall 12).

### 0.2 Gate

Skript `C:\Users\Markin\Workspace\ISAABOX\tb_gate.py` (liegt aus WP2 bereits dort; Inhalt sonst: `git show origin/claude/keen-goldberg-o59mt9:docs/devdocs/ci_analysis/prompts/ISAA_06_session_per_user.md`, Abschnitt 0.4). Gate-Bereich: siehe Abschnitt 1 unten.

```powershell
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py baseline C:\Users\Markin\Workspace\ISAABOX\isaa09_baseline.json @GATE   # vor dem ersten Edit
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py check    C:\Users\Markin\Workspace\ISAABOX\isaa09_baseline.json @GATE   # nach jedem Edit, muss "NEU rot: 0"
```

### 0.3 Commit und Push (Owner-Entscheidung: direkt auf `master`)

Erst wenn alle Pflichtfelder des Protokolls belegt sind und der Bericht geschrieben ist. Ein Commit je Teilaufgabe; Dateien einzeln hinzufügen (`git add <pfad>`), nie `git add -A`/`.`/`-a`. Nachricht: erste Zeile `WP3 A09.<n>: <kurz>`, darunter Hypothese (ein Satz) und die Schlusszeile des Abnahmelaufs.

```powershell
git fetch origin
git rebase origin/master
# nach dem Rebase: Gate-Check erneut
git push origin HEAD:master
```

Abgelehnt (`non-fast-forward`): höchstens dreimal wiederholen, dann melden. Kein `--force`, kein Merge-Commit.

### 0.4 Bericht

`C:\Users\Markin\Workspace\ISAABOX\wp3_A09_bericht.md` (außerhalb des Repos), Format nach Protokoll Abschnitt 4: je Teilaufgabe die Tabelle mit den 11 Pflichtfeldern, Zustand, „Gefunden, nicht geändert“, Patch-Inventar der Tests. Denselben Text als Antwort ausgeben.


## 1. Deine Dateien

- `toolboxv2/mods/CloudM/LiveSync/data_api_sync.py` – **nur** `DataApiTransport` (:38–87), `_safe_rel` (:89), `sync_folder` (:109 ff.); **nicht** `ensure_sync_key` und `sync_all` (gehören WP3-08)
- `toolboxv2/mods/CloudM/LiveSync/account_bridge.py`
- `toolboxv2/mods/CloudM/LiveSync/index.py` (nur falls Löschungen den Index brauchen; belegen)
- `toolboxv2/tests/test_core/test_a4_betriebsbefunde.py` (Nr. 2–6)

Gate-Bereich:
```powershell
$GATE = @('toolboxv2/tests/test_core/test_a4_betriebsbefunde.py','toolboxv2/tests/test_core/test_data_api_sync.py','toolboxv2/tests/test_core/test_account_bridge.py','toolboxv2/tests/test_mini/test_cloudm/test_LiveSync')
```
Vor der Baseline mit `git ls-files toolboxv2/tests | Select-String "sync|bridge|livesync"` prüfen, ob es weitere Testdateien für diese Module gibt, und sie in `$GATE` aufnehmen (Liste in den Bericht).

**Nicht anfassen:** `mods/CloudM/auth/*`, `utils/workers/*` (WP3-05), `pyproject.toml`, `utils/security/cryp.py`, `LiveSync/crypto.py`, `docs/`.

Owner-Entscheidungen, die hier gelten: Sync nur über `CloudM.UserDataAPI` (vorhanden: `list_data`, `get_data`, `set_data`, **`delete_data`** `UserDataAPI.py:691`); **Löschungen werden übertragen**; kein neuer Server-Endpunkt.

## 2. Teilaufgaben in dieser Reihenfolge

Für jede: Protokoll-Schritte 1–11. Die Tests Nr. 3–6 sind vorhanden und rot; lies sie zuerst und prüfe sie gegen die Prüfliste des Testleitfadens (Abschnitt 3). Ein Test, der dort durchfällt (z. B. Patch im eigenen Code), wird **vor** dem Fix umgeschrieben (eigener Commit, Begründung im Bericht).

### 2.1 401 heißt „reauth_required“ (Nr. 6)

`DataApiTransport._call` (:52–70) macht aus `exec_code 401` und „Session fetch failed“ ein `TransferError("offline")`. Hypothese aufschreiben und prüfen: Was liefert `app.run_http` bei abgelaufener Sitzung wirklich (Antwortform, `exec_code`), und was bei fehlendem Netz? Die Formen stammen aus dem echten Code (`App.run_http`, `session.fetch`), nicht aus dem Test. Soll: abgelaufene/ungültige Anmeldung → `reauth_required`, kein Netz → `offline`. Der Rückgabewert von `sync_all` zeigt den Zustand (`{"error": "reauth_required"}`), die Kontoseite liest ihn über `account_state()` (`last_sync_error`) – prüfe, dass er dort ankommt (nur lesen, nicht ändern).

### 2.2 Konfliktkopien überleben den nächsten Export (Nr. 3)

`export_account` löscht im Block „delete vanished keys“ (`account_bridge.py:67-79`) jede Spiegeldatei, die nicht aus dem Konto kommt – also auch Konfliktkopien, die `sync_folder` über `save_conflict_copy` angelegt hat. Vorhandene Erkennung benutzen: `conflict.is_conflict_copy` (schon in `data_api_sync` importiert). Keine zweite Erkennung bauen. Klären und im Bericht festhalten: Was passiert danach mit einer Konfliktkopie (bleibt sie im Spiegel bis der Nutzer entscheidet? wird sie importiert?). Soll laut Test: lokaler Wert bleibt erhalten.

### 2.3 Walker bleibt im Ordner (Nr. 4)

`sync_folder` geht mit `local_dir.rglob("*")` (:135) durch den Ordner und folgt Symlinks/Junctions. Soll: nichts außerhalb von `local_dir` wird gelesen oder geschrieben. `followlinks=False` reicht unter Windows für Junctions nicht (Befund WP3-A4). Umsetzung: jede Datei und jedes Zielverzeichnis mit `Path.resolve()` gegen `local_dir.resolve()` prüfen (`is_relative_to`), außerhalb → überspringen und im Ergebnis melden. Gilt auch für den **Schreib**pfad (`_atomic_write`-Ziel), nicht nur fürs Lesen. Test unter Windows mit echter Junction (`mklink /J`), wie der vorhandene Test es tut.

### 2.4 Löschungen übertragen (Nr. 5)

Heute fehlt das komplett (kein `delete` im Transport). Soll:
- Datei war synchronisiert (Index-Eintrag `sync_state="synced"`), ist lokal weg und remote unverändert → `delete_data` remote, Index-Eintrag weg.
- Remote fehlt ein Eintrag, den der Index als synchronisiert kennt, und lokal ist die Datei unverändert → lokal löschen, Index-Eintrag weg.
- Lokal geändert **und** remote gelöscht → lokal behalten und neu hochladen (kein Datenverlust). Remote geändert und lokal gelöscht → remote gewinnt, Datei kommt zurück.
- **Schutz:** Löschentscheidungen nur, wenn `list` in diesem Lauf erfolgreich war (kein Löschen bei leerer Liste nach Fehler). Mehr als 50 % der bekannten Einträge würden gelöscht → nichts löschen, Ergebnis `{"error": "mass_delete_blocked", ...}`. Die Schwelle als Konstante neben `MAX_BYTES`.

`DataApiTransport.delete(path)` ruft `delete_data` mit derselben Antwortbehandlung wie die anderen Methoden (über `_call`, kein zweiter Parser). Antwortform von `delete_data` aus `UserDataAPI.py:691` lesen und im Bericht zitieren.

Abnahme zusätzlich zu Nr. 5: ein Zwei-Geräte-Test (zwei Datenordner, **ein** gemeinsamer Fake der Data-API, der `list_data`/`get_data`/`set_data`/`delete_data` mit den echten Antwortformen liefert). Fälle: Löschen A → weg auf B; lokal geändert + remote gelöscht → bleibt; Fehler bei `list` → nichts gelöscht; Massenlöschung → blockiert. Für den Fake gilt Leitfaden Regel 10: einmal dieselben Fälle gegen die echte `CloudM.UserDataAPI` mit Temp-Datenbank (`IS_OFFLINE_DB=true`) laufen lassen; geht das nicht, Grund in den Bericht.

### 2.5 Dateinamen mit `:` sichtbar

`_safe_rel` (:89–99) gibt bei `:` still `None` zurück; die Datei wird übersprungen, niemand merkt es. Soll: übersprungene Pfade stehen im Ergebnis von `sync_folder` (`"skipped": [{"path": ..., "reason": "colon"}]`) und als `logger.warning`. Gleiches Feld für 2.3 (`"reason": "outside"`) und für zu große Dateien (`"reason": "too_large"`, falls heute auch still). Keine neue Struktur, wenn es schon ein Ergebnisfeld dafür gibt – erst lesen.

### 2.6 Wheel-Test für Multipart (Nr. 2)

Der vorhandene Test Nr. 2 ist auch auf dem alten Code grün, sobald `multipart` von Hand installiert ist (Leitfaden, Fall 12). Ergänze ihn (nicht ersetzen) um einen Lauf gegen das Wheel: im Test das Wheel bauen (wie Nr. 1), in eine **frische venv** in `$T` mit `pip install <wheel>[web]` installieren, mit dieser venv `python -I` aus einem fremden Ordner `import toolboxv2.utils.workers.server_worker as w; print(w.MULTIPART_AVAILABLE)` → `True`. Gegenprobe: in `pyproject.toml` die Zeile `multipart>=0.2.4` lokal entfernen (nicht committen) → Test rot. Der Test ist langsam; Zeitlimit im Test setzen (Regel 12).

## 3. Fertig, wenn

Nr. 2 (mit Wheel), 3, 4, 5, 6 grün, der Zwei-Geräte-Test grün, Gate „NEU rot: 0“, Gegenproben im Bericht. Nr. 7 bleibt rot (WP3-08) und steht so im Bericht.
