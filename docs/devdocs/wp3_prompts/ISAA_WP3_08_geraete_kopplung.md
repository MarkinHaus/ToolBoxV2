# ISAA-Auftrag WP3-08 – Sync-Schlüssel per Geräte-Kopplung, `get_sync_key` zurückbauen

> Basis: `master` **nach** WP3-05 und WP3-09 (beide gepusht). Welle 2. Owner-Entscheidung 2026-11-18: **Weg 2 – Geräte-Kopplung** (WP3-Dokument, Abschnitt 6).
> Abnahme: `test_a4_betriebsbefunde.py` Nr. 7 grün (neue Fassung, siehe 2.4) und der Kopplungstest aus 2.5 grün.

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
git worktree add -B isaa/08 C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa08 origin/master
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa08
uv venv -p 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install pytest pytest-asyncio pytest-timeout build
uv pip install -e toolboxv2/mods/isaa
uv pip install -e ".[web]"
$env:MOCK_EXTERNAL_SERVICES = "true"; $env:IS_OFFLINE_DB = "true"
python -c "import toolboxv2; print(toolboxv2.__file__)"   # muss in ...Toolboxv2-isaa08 liegen
```

Nichts von Hand in die venv installieren, was nicht in `pyproject.toml` steht. Sonst misst die Gegenprobe deine Umgebung statt des Codes (Testleitfaden, Fall 12).

### 0.2 Gate

Skript `C:\Users\Markin\Workspace\ISAABOX\tb_gate.py` (liegt aus WP2 bereits dort; Inhalt sonst: `git show origin/claude/keen-goldberg-o59mt9:docs/devdocs/ci_analysis/prompts/ISAA_06_session_per_user.md`, Abschnitt 0.4). Gate-Bereich: siehe Abschnitt 1 unten.

```powershell
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py baseline C:\Users\Markin\Workspace\ISAABOX\isaa08_baseline.json @GATE   # vor dem ersten Edit
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py check    C:\Users\Markin\Workspace\ISAABOX\isaa08_baseline.json @GATE   # nach jedem Edit, muss "NEU rot: 0"
```

### 0.3 Commit und Push (Owner-Entscheidung: direkt auf `master`)

Erst wenn alle Pflichtfelder des Protokolls belegt sind und der Bericht geschrieben ist. Ein Commit je Teilaufgabe; Dateien einzeln hinzufügen (`git add <pfad>`), nie `git add -A`/`.`/`-a`. Nachricht: erste Zeile `WP3 A08.<n>: <kurz>`, darunter Hypothese (ein Satz) und die Schlusszeile des Abnahmelaufs.

```powershell
git fetch origin
git rebase origin/master
# nach dem Rebase: Gate-Check erneut
git push origin HEAD:master
```

Abgelehnt (`non-fast-forward`): höchstens dreimal wiederholen, dann melden. Kein `--force`, kein Merge-Commit.

### 0.4 Bericht

`C:\Users\Markin\Workspace\ISAABOX\wp3_A08_bericht.md` (außerhalb des Repos), Format nach Protokoll Abschnitt 4: je Teilaufgabe die Tabelle mit den 11 Pflichtfeldern, Zustand, „Gefunden, nicht geändert“, Patch-Inventar der Tests. Denselben Text als Antwort ausgeben.


## 1. Deine Dateien

- `toolboxv2/mods/CloudM/auth/api_session.py` (nur `get_sync_key` entfernen, :172–195)
- `toolboxv2/mods/CloudM/LiveSync/data_api_sync.py` (nur `ensure_sync_key` und die Stelle in `sync_all`, die den Schlüssel holt)
- **neu, freigegeben:** `toolboxv2/mods/CloudM/LiveSync/pairing.py` (eine Datei, keine weitere)
- `toolboxv2/utils/workers/fast/wizard_api.py` (nur **zwei neue lokale Routen**, freigegeben: `GET /api/account/devices`, `POST /api/account/devices/approve`; im Stil der vorhandenen `/api/account/*` mit derselben einen Lokal-Prüfung aus WP3-05)
- `toolboxv2/utils/workers/fast/account_page.py` (nur ein Abschnitt „Wartende Geräte“ mit Bestätigen-Knopf, sichtbar wenn verknüpft)
- `toolboxv2/utils/clis/first_run.py` (nur `account_state`: Feld für den Kopplungszustand ergänzen)
- `toolboxv2/tests/test_core/test_sync_key.py` (ersetzen, freigegeben: testet die entfernte Funktion)
- `toolboxv2/tests/test_core/test_a4_betriebsbefunde.py` (nur Nr. 7)
- **neu:** `toolboxv2/tests/test_core/test_wp3_pairing.py`

Gate-Bereich:
```powershell
$GATE = @('toolboxv2/tests/test_core/test_a4_betriebsbefunde.py','toolboxv2/tests/test_core/test_data_api_sync.py','toolboxv2/tests/test_core/test_account_bridge.py','toolboxv2/tests/test_core/test_sync_key.py','toolboxv2/tests/test_core/test_wizard_e2e_http.py','toolboxv2/tests/test_core/test_a3_betriebsbefunde.py','toolboxv2/tests/test_mini/test_cloudm/test_LiveSync')
```

**Nicht anfassen:** `utils/security/cryp.py`, `LiveSync/crypto.py` (nur benutzen), das **Feld** `user_pass_sync` im Modell und in `AuthHelpers.py`/`extras.py`, `UserDataAPI.py`, `pyproject.toml`, `docs/`. Keine neue Funktion mit `api=True`, keine neue Krypto-Bibliothek.

## 2. Teilaufgaben in dieser Reihenfolge

### 2.1 Lesen und Ist-Zustand

1. `ensure_sync_key` (`data_api_sync.py:249-276`): holt den Schlüssel über `CloudM.Auth.get_sync_key` vom Server und legt ihn im Nutzerdatensatz `user_pass_sync` ab.
2. `get_sync_key` (`api_session.py:172-195`): erzeugt bei leerem Feld `Code.generate_symmetric_key()` und gibt ihn im Klartext aus.
3. **Wichtig:** `user_pass_sync` ist älter als WP1. Der alte Magic-Link-Weg leitet daraus seinen Schlüssel ab: `"01#" + Code.one_way_hash(user.user_pass_sync, "CM", "get_magic_link_email")` (`AuthHelpers.py:108`, `extras.py:343`). Klären und mit Datei:Zeile belegen: Wo wird ein `01#`-Schlüssel **geprüft** (`m_log_in.html` und die Route dahinter)? Ist dieser Weg auf `simplecore.app` erreichbar (nur aus dem Code, keine Live-Aufrufe)? Ergebnis in den Bericht unter „Gefunden, nicht geändert“ mit Einschätzung. **Nicht ändern.**
4. Welches Schlüsselformat erwartet `LiveSync.crypto.encrypt_bytes` (`_decode_key` :49)? Ist ein `Code.generate_symmetric_key()`-Schlüssel (Fernet, urlsafe base64) dort gültig, oder muss der Kontoschlüssel mit `crypto.generate_encryption_key()` (:39) entstehen? Mit einem Lauf belegen. Der Kontoschlüssel entsteht mit der Funktion, die zu `encrypt_bytes` passt.
5. `Code.generate_asymmetric_keys`, `encrypt_asymmetric`, `decrypt_asymmetric` (`cryp.py:210, 310, 341`): Schlüssellänge und maximale Klartextlänge für `encrypt_asymmetric` belegen (Kontoschlüssel muss hineinpassen). `Code.encode_code`/`decode_code` (`cryp.py:100-118`) verschlüsseln mit dem Geräteschlüssel – das ist der Weg, lokale Geheimnisse abzulegen.

### 2.2 Verfahren (festgelegt, nicht neu auslegen)

Datenablage über die vorhandene Data-API (`set_data`/`get_data`/`list_data`/`delete_data`, Scope `private`, über `DataApiTransport` – kein zweiter Parser):

| Pfad in der Data-API | Inhalt |
|---|---|
| `livesync/devices/<device_id>/pub` | öffentlicher RSA-Schlüssel des Geräts (PEM), Gerätename, Zeitpunkt |
| `livesync/devices/<device_id>/wrapped` | Kontoschlüssel, mit `Code.encrypt_asymmetric` für **dieses** Gerät verschlüsselt |

Lokal (in `app.data_dir/account_sync/keys/`, nie im Klartext): privater RSA-Schlüssel des Geräts und Kontoschlüssel, jeweils mit `Code().encode_code(...)` (Geräteschlüssel) verschlüsselt.

`device_id`: die vorhandene `sanitize_device_id(socket.gethostname())` wie in `sync_all` – keine zweite Ableitung.

Ablauf in `pairing.py` (eine öffentliche Funktion je Schritt, alle `async`, alle über den übergebenen Transport):
1. **Gerät vorbereiten:** Schlüsselpaar erzeugen, falls lokal keins; `pub` hochladen, falls remote keins oder abweichend.
2. **Kontoschlüssel holen:** lokal vorhanden → benutzen. Sonst `wrapped` für das eigene Gerät lesen und mit dem privaten Schlüssel entschlüsseln. Gibt es **kein einziges** `wrapped` im Konto (erstes Gerät) → Kontoschlüssel erzeugen, lokal ablegen, `wrapped` für sich selbst hochladen. Gibt es `wrapped` für andere Geräte, aber nicht für dieses → Zustand `pairing_pending`, kein Sync.
3. **Bestätigen** (auf einem gekoppelten Gerät): Liste der Geräte mit `pub` ohne `wrapped`; für ein bestätigtes Gerät den Kontoschlüssel mit dessen `pub` verschlüsseln und `wrapped` hochladen.
4. **Gerät entfernen** gehört nicht zu diesem Auftrag (im Bericht als offen nennen).

`sync_all`: statt `ensure_sync_key` Schritt 1 und 2. Bei `pairing_pending` → `{"skipped": True, "reason": "pairing_pending"}`; `account_state()` meldet das, die Kontoseite zeigt „Auf einem anderen Gerät bestätigen“. Entschlüsselung eines vorhandenen Remote-Eintrags schlägt fehl (z. B. Altdaten mit dem früheren Server-Schlüssel) → sichtbarer Fehler `key_mismatch` im Ergebnis, **nichts überschreiben, nichts löschen**.

Routen (lokal, Kontopflicht gilt): `GET /api/account/devices` → `[{device_id, name, since, paired}]`; `POST /api/account/devices/approve` mit `{"device_id"}` → Schritt 3. Kontoseite: Liste und Knopf „Bestätigen“. Kein Schlüssel und kein `pub` im Klartext in Antworten oder Logs (nur `device_id`, Name, Zeit).

### 2.3 `get_sync_key` zurückbauen

`get_sync_key` aus `api_session.py` entfernen (freigegeben). `git grep -n get_sync_key` danach: nur noch Treffer in `docs/` und im Bericht. `user_pass_sync` wird vom Sync weder gelesen noch geschrieben. Das Feld selbst bleibt.

### 2.4 Tests

- `test_sync_key.py` ersetzen (freigegeben): (a) `CloudM.Auth` hat keine Funktion mehr, die einen Schlüssel ausgibt – geprüft über die **registrierten** Funktionen des Moduls (`app.functions` bzw. die Export-Registry), nicht über Quelltext; (b) `get_user_data`-Antworten enthalten `user_pass_sync` nicht (vorhandener Fall behalten).
- `test_a4` Nr. 7 neu fassen: Zwei-Geräte-Lauf über einen gemeinsamen Fake der Data-API (echte Antwortformen, Vertragsprüfung nach Leitfaden Regel 10 – wenn WP3-09 schon einen solchen Fake hat, **den** benutzen). Danach: Im Fake liegt kein Eintrag, der den Kontoschlüssel im Klartext oder nur base64-kodiert enthält (alle gespeicherten Werte gegen den Kontoschlüssel prüfen), und der Nutzerdatensatz hat kein neues `user_pass_sync`.
- `test_wp3_pairing.py`: erstes Gerät erzeugt den Schlüssel; zweites Gerät → `pairing_pending`, kein Upload von Nutzdaten; Bestätigen auf Gerät A → B synchronisiert, Inhalte gleich; Server-Fake sieht nur Chiffrat; Neustart von B (neue App-Instanz, gleicher Datenordner) → Schlüssel lokal wieder lesbar; anderer Geräteschlüssel (`TB_R_KEY` anders) → lokale Datei nicht lesbar, sichtbarer Fehler, keine Löschung.
- Routen: per HTTP gegen den echten Worker (Muster `test_wizard_e2e_http.py`), mit und ohne Konto.

### 2.5 Abnahme im echten Ablauf

Zwei isolierte Umgebungen (`$T1`, `$T2`, je eigenes `TB_DATA_DIR`, `HOME`, `APPDATA`, `LOCALAPPDATA`, **verschiedenes** `TB_R_KEY`), gemeinsamer Fake-Server für `run_http` (Datei-gestützt, damit beide Prozesse ihn sehen). Gerät 1 `sync_all` → Gerät 2 `sync_all` (`pairing_pending`) → Gerät 1 bestätigt → Gerät 2 `sync_all` → Datei aus dem globalen Ordner von Gerät 1 liegt inhaltsgleich bei Gerät 2. Ausgaben in den Bericht.

### 2.6 Nach dem Push: Schritt für den Owner (in den Bericht)

Auf dem Server sind `user_pass_sync`-Werte, die `get_sync_key` erzeugt und an Geräte ausgegeben hat. Je nach Ergebnis von 2.1 Punkt 3 schreibst du dem Owner einen konkreten Schritt auf (welche Nutzer, welcher Befehl, um den Wert zu erneuern). **Selbst nichts auf dem Server ausführen.**

## 3. Fertig, wenn

Nr. 7 und `test_wp3_pairing.py` grün, `git grep get_sync_key` ohne Codetreffer, Abnahme 2.5 im Bericht, Gate „NEU rot: 0“, Owner-Schritt 2.6 beschrieben.
