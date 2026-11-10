# Handoff — lokaler Coding-Agent: stabile Tauri-Anwendung (2026-11-10)

> Du arbeitest **lokal auf dem Windows-Rechner des Owners**. Du hast Zugriff auf Windows und auf `simplecore.app`.
> Deine einzige Arbeitsanweisung ist [`taskmap_stable_tauri_2026-11.md`](taskmap_stable_tauri_2026-11.md). Dieses Dokument sagt dir nur, wie du anfängst, wo du stoppst und wie du berichtest.

---

## 1. Auftrag in einem Satz

Arbeite die Task-Map von T1 bis T14 in der dort festgelegten Reihenfolge ab, bis die Abnahmepunkte Z1–Z8 erfüllt sind: Installation, Start, Anmeldung mit SimpleCore-Konto, ISAA, OBS, Neustart, Update, Daten auf allen Geräten.

## 2. Start

```powershell
git fetch origin
git checkout claude/wizardly-cannon-o944j4
```

Falls du stattdessen den Patch bekommen hast (Basis: `master` @ `d1db5c9c`):

```powershell
git checkout -b claude/wizardly-cannon-o944j4 d1db5c9c
git am taskmap_stable_tauri_2026-11.patch
```

Der Patch enthält nur Dokumente (`docs/devdocs/`), keinen Code. Danach: Task-Map vollständig lesen, dann mit **T1** beginnen.

## 3. Reihenfolge

```
T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 → K0 → K1 → K2 → K3 → K4 → T10 → T11 → S1 → H5 → S2 → S3 → S4 → S5 → H1 → T12 → T13 → T14
```

| Block | Inhalt | Ohne Tauri prüfbar |
|---|---|---|
| T1–T4 | Umgebung, sauberer Arbeitsbaum, Nutzerzustand aus dem Paketordner, Abhängigkeiten | ja |
| T5–T8 | Installer nimmt das mitgelieferte Wheel, Wheel ins Bundle, Version, Start ohne `tb-worker` | T5–T7 ja, T8 braucht den Tauri-Build |
| K0–K4 | Messung gegen `simplecore.app`, Gerätekonto, automatische Anmeldung, Kontopflicht | ja |
| T10–T11 | ISAA und OBS im Owner-Prozess | ja |
| S1–S5 | Schlüssel beim Konto, Data-API-Sync, Blob-Sync, globaler ISAA-Ordner | ja, S5 braucht zwei Installationen |
| T12–T14 | App-Update, Kern-Update, Gesamtabnahme | nein |

## 4. Regeln, die am häufigsten gebrochen werden

Die vollständigen Regeln stehen in Abschnitt 1 der Task-Map. Diese hier gelten ohne Ausnahme:

1. **Ein Task, ein Commit, erst nach grünem Gate.** Kein Task wird begonnen, solange das Gate des vorigen rot ist.
2. **Rot bleibt rot → stoppen und melden.** Erst den Abschnitt „Wenn rot" ausführen. Hilft das nicht: Befehl und Ausgabe an den Owner, nichts weiter tun.
3. **Nichts erfinden.** Feldnamen, URLs und Antwortformen von `simplecore.app` kommen aus dem Messprotokoll von K0, nicht aus Vermutungen. Steht etwas dort nicht, miss es nach oder frag.
4. **Keine Tests abschwächen.** Neue Tests sind `unittest`.
5. **Keine Geheimnisse in Dateien, Logs oder Commits:** keine Tokens, keinen Kontoschlüssel, keine MinIO-Zugangsdaten, keinen Signaturschlüssel. `probe_token.tmp` aus K0 nach der Messung löschen.
6. **Kopien synchron halten:** `installer.sh` / `installer.ps1` (Root ≡ `src-tauri`), `installer.html` (≡ `web/installer/index.html`).
7. **Nie auf `master` pushen.** Nur auf `claude/wizardly-cannon-o944j4`.
8. **Entscheidungen des Owners nicht neu verhandeln** (Liste in Abschnitt 1 der Task-Map). Eine neue Architektur-, Auth- oder Sicherheitsfrage entscheidest du nicht selbst — du legst sie dem Owner vor.

## 5. Stellen, an denen du auf den Owner wartest

| # | Stelle | Was der Owner tut |
|---|---|---|
| K0 | Browser-Login für die Messung | meldet sich bei SimpleCore an |
| H0 | vor T10 | trägt einen LLM-API-Key ein |
| H4 | nur wenn K0-P3 rot ist | konfiguriert MinIO auf `simplecore.app` |
| H5 | nach S1 | rollt die Server-Änderung auf `simplecore.app` aus |
| H1 | vor T12 | hinterlegt den Signaturschlüssel als GitHub-Secrets |
| H2, H3 | in T14 | stößt `nightly.yml` zweimal an |

Während du auf H5 wartest, darfst du **nicht** mit S2 beginnen. H4 blockiert nur S2 und S4; K1–K4, T10, T11 und S3 gehen ohne MinIO.

## 6. Was gemessen ist und was nicht

- **Gemessen (Linux, 2026-11-10):** Wheel-Bau mit `tb build build`, Installation per `uv tool install` aus dem lokalen Wheel, `tb -init local`, Owner auf 8467, ISAA-Mount unter `/isaa`. Details: Abschnitt 2 der Task-Map (F1–F15).
- **Nur aus dem Code gelesen:** alles zu Konto und Sync (F16–F22).
- **Nicht gemessen:** Windows, `installer.ps1`, Tauri-Build, Updater, `simplecore.app`, MinIO auf dem Server. Deshalb beginnt der Konto-Block mit der Messung K0, und deshalb haben T5, T6 und T8 eigene Windows-Gates.

## 7. Bericht nach jedem Task

Genau dieses Format, nichts darüber hinaus:

```
Task:    <Kennung>
Status:  grün | rot | wartet auf <H-Kennung>
Commit:  <Hash>
Gate:    <Befehl> → <Ergebnis in einer Zeile>
Abweichung von der Map: <keine | was und warum>
Offene Frage an den Owner: <keine | eine Frage>
```

Am Ende (T14): die Tabelle Z1–Z8 mit „ok" oder der genauen Fehlermeldung.

## 8. Nicht anfassen

Alles aus Abschnitt 6 der Task-Map. Zusätzlich aufgefallen und dem Owner gemeldet, **nicht** von dir zu ändern:

- `mods/CloudM/auth/api_session.py:173` — `update_user_data` enthält selbst keine Prüfung des Aufrufers.
- `utils/clis/local_cli.py:258` — gibt die Login-Antwort samt Tokens per `print` aus.
- `utils/clis/cli_oauth.py` ab Zeile 105 — loser Textblock im Modul.
