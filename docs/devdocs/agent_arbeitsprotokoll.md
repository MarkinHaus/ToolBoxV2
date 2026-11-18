# Arbeitsprotokoll für Agenten (ISAA und andere)

> Stand 2026-11-17. Gilt für **jeden** Auftrag an einen Agenten in diesem Repo. Jeder Auftrag verweist auf diese Datei; was hier steht, muss im Auftrag nicht wiederholt werden.
> Testregeln: `docs/devdocs/testing_guide.md`.
>
> Grund: Die Prüfung von `master` @ `c36c3dc` hat gezeigt, wohin Handlungsspielraum führt.
> - Aufgaben wurden **ersetzt statt erfüllt**: Blob-Sync und Kontodaten-Abgleich wurden entfernt.
> - Es kamen **ungefragte Extras** dazu: die neue Server-Funktion `get_sync_key` mit Kontoschlüssel im Klartext, umbenannte Routen, eine andere Kontoseite.
> - Die Tests **patchten die fehlerhaften Stellen weg**.
>
> Dieses Protokoll erzwingt den langsamen Weg: lesen, verstehen, Hypothese, belegen, kleinster Eingriff, echte Ausführung.

---

## 1. Rahmen: was du darfst und was nicht

1. **Nur der Auftrag.** Du änderst nur die Dateien und Zeilen, die der Auftrag nennt. Alles andere, das dir auffällt, kommt in den Bericht unter „Gefunden, nicht geändert“, mit Datei:Zeile und einem Satz. Auch dann, wenn es „nur eine Kleinigkeit“ ist.
2. **Ohne ausdrückliche Freigabe im Auftrag verboten:**
   1. neue Funktionen mit `api=True`, neue Routen, neue Endpunkte, neue Dateien im Produktcode;
   2. Funktionen, Module, Tests oder Routen entfernen, umbenennen oder ersetzen;
   3. Wrapper, Adapter, Kompatibilitätsschichten, „Fallback“-Zweige um vorhandenen Code;
   4. neue Abhängigkeiten in `pyproject.toml`;
   5. Änderungen an Schlüsseln, Secrets, Verschlüsselung, Authentifizierung.
3. **Keine Dopplung.** Vor jeder neuen Funktion, Klasse oder Konstante: `git grep -n` nach dem **Zweck** (nicht nur dem Namen) und den Fund in den Bericht. Gibt es etwas Passendes, wird es benutzt. Gibt es zwei passende Stellen, wird **nicht** eine dritte gebaut: melden.
4. **Push nach `master` nur mit vollständigem Bericht** (Owner-Entscheidung 2026-11-18: Agenten pushen direkt auf `master`, geprüft wird danach). Vor dem Push müssen alle Pflichtfelder aus Abschnitt 2 belegt sein, das Gate „NEU rot: 0“ zeigen und der Bericht geschrieben sein. Kein Merge-Commit, kein `--force`. Ist ein Schritt nicht belegt: nicht pushen, Zustand `gestoppt`.
5. **Owner-Entscheidungen** (WP-Dokumente, Abschnitt „Entscheidungen“) werden nicht neu ausgelegt. Passt eine Entscheidung nicht zum Code: stoppen und fragen.
6. **Stoppen und melden** statt weitermachen, wenn:
   1. die Hypothese nicht bestätigt werden kann;
   2. die Abnahme rot bleibt;
   3. eine Datei außerhalb des Auftrags geändert werden müsste;
   4. der Code anders aussieht als im Auftrag beschrieben;
   5. du zum dritten Mal denselben Fehler „reparierst“.

## 2. Der Ablauf je Änderung (nicht abkürzbar)

Jeder Schritt hat ein Pflichtfeld im Bericht. Fehlt ein Feld, gilt der Auftrag als **nicht erledigt**.

| Nr. | Schritt | Pflichtfeld im Bericht |
|---|---|---|
| 1 | **Lesen.** Die zu ändernde Datei vollständig lesen; alle Aufrufer der betroffenen Funktion (`git grep -n "<name>("`) und die Tests, die sie benutzen | Liste: Aufrufer mit Datei:Zeile, und mit welchen Argumenten sie aufrufen |
| 2 | **Ist-Zustand messen.** Den Fehler im echten Ablauf auslösen (Abschnitt 3), nicht im Unit-Test | Befehl + echte Ausgabe |
| 3 | **Hypothese aufschreiben.** „Fehler X entsteht, weil Datei:Zeile …. Wenn das stimmt, zeigt Befehl Y die Ausgabe Z; wenn nicht, zeigt er W.“ | Text der Hypothese |
| 4 | **Hypothese prüfen** mit dem Befehl aus Schritt 3, **vor** jeder Änderung | Befehl + Ausgabe + „bestätigt“/„widerlegt“. Bei „widerlegt“ zurück zu Schritt 1 |
| 5 | **Roter Test.** Ein `unittest`, der den Fehler über den Betriebs-Eingang zeigt; Doubles nur an der Prozessgrenze (Testleitfaden, Regel 2) | Testname, Lauf mit roter Ausgabe (Fehlermeldung wie in Schritt 2) |
| 6 | **Kleinster Eingriff.** Nur so viel ändern, wie die Hypothese verlangt | `git diff --stat` |
| 7 | **Isolierter Lauf mit vollen Abhängigkeiten** (Abschnitt 3) – nach **jedem** Edit, nicht erst am Ende | Befehl + Ausgabe, die das erwartete Verhalten zeigt |
| 8 | **Gegenprobe.** `git stash` → Test aus Schritt 5 rot; `git stash pop` → grün | beide Ausgaben |
| 9 | **Nachbarn.** Alle Tests der geänderten Datei und ihrer Aufrufer, allein und zusammen | Schlusszeilen von pytest |
| 10 | **Gate.** Kein Test rot, der vorher grün war (Baseline vor Schritt 6) | Ausgabe „NEU rot: 0“ |
| 11 | **Commit.** Ein Commit je Auftrag-Nummer; die Nachricht nennt Hypothese und Beleg | Commit-Hash |

## 3. „Isoliert, aber mit vollen Abhängigkeiten“ – so geht es in diesem Repo

Ziel: Der geänderte Teil läuft so, wie er im Betrieb läuft, mit echter App, echten Dateien und echten Importen, aber ohne die Owner-Umgebung zu berühren.

### 3.1 Immer: eigene Umgebung (PowerShell)

```powershell
$T = Join-Path $env:TEMP ("tb_iso_" + [guid]::NewGuid().ToString("N").Substring(0,8))
New-Item -ItemType Directory -Force "$T\data","$T\home","$T\appdata","$T\local" | Out-Null
$env:TB_DATA_DIR  = "$T\data"
$env:USERPROFILE  = "$T\home"; $env:HOME = "$T\home"
$env:APPDATA      = "$T\appdata"; $env:LOCALAPPDATA = "$T\local"
$env:MOCK_EXTERNAL_SERVICES = "true"; $env:IS_OFFLINE_DB = "true"
```

Am Ende: Shell schließen (die Variablen gelten nur dort), `$T` löschen. Nie unter dem Haupt-Checkout oder im echten Profil schreiben.

### 3.2 Funktion oder Modul: über die echte App aufrufen

Kleines Skript **außerhalb** des Repos (`$T\probe.py`), das die echte App holt und die Funktion so aufruft, wie ihr Aufrufer es tut:

```python
import os, json
from toolboxv2 import get_app
app = get_app("probe")                      # echte App, Datenordner aus TB_DATA_DIR
from toolboxv2.utils.clis.first_run import onboarding_state
print(json.dumps(onboarding_state()))       # genau wie local_ui.py:89 – ohne app
```

Starten aus einem **anderen** Ordner als dem Datenordner (`cd $T\home; python $T\probe.py`), damit Fehler mit relativen Pfaden sichtbar werden.

### 3.3 Route: echter Worker auf freiem Port

Muster wie `toolboxv2/tests/test_core/test_wizard_e2e_http.py`: die echte FastTB-App (`toolboxv2.utils.workers.fast.local_ui.app`) mit `serve(host="127.0.0.1", port=<frei>)` in einem Thread starten, dann von außen mit HTTP prüfen. `tb workers start` (Ports 8467/8468) nur, wenn die Ports frei sind. Sind sie belegt, läuft vermutlich die Owner-Instanz: **nicht ausweichen, Owner fragen.**

### 3.4 Paket, Importe, `pyproject.toml`: das gebaute Wheel

Siehe Testleitfaden, Regel 8: Wheel bauen, frische venv in `$T`, Start mit `python -I` aus `$T\home`, betroffenen Pfad ausführen.

### 3.5 Server-Seite (`simplecore.app`)

Live-Aufrufe erst, wenn der Owner den Stand ausgerollt hat. Bis dahin ein Fake von `run_http`, der die **echten Antwortformen** liefert. Der Fake wird einmal gegen die echte `CloudM.UserDataAPI` mit Temp-Datenbank geprüft (Testleitfaden, Regel 10).

## 4. Bericht

Datei laut Auftrag. Je Auftrag-Nummer eine Tabelle mit den Pflichtfeldern aus Abschnitt 2. Danach:

1. **Zustand:** `erledigt` (nur wenn alle Pflichtfelder belegt sind), `gestoppt (Grund)`, `nicht verifiziert (Grund)`.
2. **Gefunden, nicht geändert:** nummerierte Liste.
3. **Patch-Inventar der neuen oder geänderten Tests:** je Patch das Ziel, die Begründung „Prozessgrenze: …“ und das Ergebnis der Gegenprobe.

Der Owner bzw. der Planer prüft den Bericht gegen den Diff, bevor gemergt wird. Behauptungen ohne Ausgabe zählen nicht.
