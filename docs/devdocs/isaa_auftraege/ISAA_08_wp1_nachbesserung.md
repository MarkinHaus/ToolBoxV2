# ISAA-Auftrag 08 – WP1 nachbessern: Kontopflicht, Auto-Anmeldung, Sync über LiveSync und Data-API

> Komplexität: **4 von 5** · Teile 1–11 · Basis: `master` @ `d885fa2` (Stand 2026-11-16)
> Ersetzt die Umsetzung von S1–S3 (`63c2478`, `352e33f`, `8a29da0`) und bessert K2 (`4c5f97b`) und K3 (`5cf22e1`) nach.
> Grundlage: Prüfung von `master` am 2026-11-16 (94/94 Tests grün, aber K2, K3 und S1–S3 erfüllen die WP1-Kriterien nicht) und die Entscheidungen des Owners vom selben Tag (Abschnitt 0.1).
> Alte Namen (K2, K3, S1–S3, K0, K4, S4, T10, D1) bleiben, damit man sie im WP1-Dokument findet. Die Teile dieses Auftrags haben **nur Nummern**.

**Du bist eine ISAA-Instanz und arbeitest diesen Auftrag selbstständig ab.** Arbeite die Teile **genau in der angegebenen Reihenfolge** ab. Jeder Teil hat:

- **Vorher prüfen:** Befehle und die Ausgabe, die du sehen musst. Siehst du etwas anderes, ist das eine **Abweichung** → Abschnitt 0.6.
- **Schritte:** genau das tun, nicht mehr.
- **Tests:** genau diese Fälle, als `unittest`.
- **Abnahme:** Befehl und erwartete letzte Zeile.
- **Commit.**

Es gibt kein „gleichwertig“ und kein „sinngemäß“. Wenn ein Schritt nicht wörtlich passt: stoppen und melden.

---

## 0. Rahmen

### 0.1 Entscheidungen des Owners (2026-11-16, verbindlich, nicht neu verhandeln)

1. **LiveSync** (`toolboxv2/mods/CloudM/LiveSync/`) ist die einzige Sync-Maschine. Verwendet werden seine Teile `crypto.py` (AES-256-GCM), `conflict.py` (Konfliktkopie), `index.py` (lokaler Index), `protocol.should_ignore`. Die doppelten Neubauten `toolboxv2/mods/CloudM/sync/*` werden entfernt.
2. **Übertragung ausschließlich über die Data-API** `CloudM.UserDataAPI` (`list_data`, `get_data`, `set_data`) per HTTPS zu `simplecore.app`, über `app.run_http(...)`. Keine signierten MinIO-URLs, kein WebSocket für den Konto-Sync, **keine MinIO-Zugangsdaten auf dem Gerät**. MinIO muss nicht öffentlich erreichbar sein.
3. **Die Ausgabe von MinIO-Zugangsdaten an Clients wird abgeschaltet** (`UserDataAPI.get_secure_storage_credentials`, `LiveSync.tb_get_my_minio_credentials`).
4. **Nie Klartext, nie „nur kodiert“:** Jeder Inhalt verlässt das Gerät nur als AES-GCM-Chiffretext (`crypto.encrypt_bytes`). Base64 ist nur die Verpackung dieses Chiffretexts im JSON. Der Kanal ist immer HTTPS. Ist die Server-Basis nicht `https://` (Ausnahme: `127.0.0.1`/`localhost` in Tests), wird **nichts** gesendet.
5. **Kontoschlüssel = `user_pass_sync`**, wie im alten `User` (`CloudM/types.py:27,41`: `Code.generate_symmetric_key()`). Er kommt zurück in `UserData`. Der Server erzeugt ihn einmal pro Konto und gibt ihn nur über die neue Funktion `get_sync_key` an den angemeldeten Nutzer selbst heraus. Auf dem Gerät liegt er im lokalen Nutzerdatensatz (lokale DB, verschlüsselt abgelegt).
6. **Konfliktregel:**
   - Nur eine Seite geändert → übertragen, **kein** Backup.
   - Beide geändert, gleicher Inhalt → nichts tun.
   - Beide geändert, verschiedener Inhalt → **lokale Version als Konfliktkopie** sichern (`conflict.save_conflict_copy`), danach gilt die **Server-Version**.
   - „Geändert“ heißt: anders als beim letzten gemeinsamen Stand (Prüfsumme lokal, `updated_at` remote). Die Uhrzeit allein entscheidet **nie**.
7. **Kontodaten Schlüssel für Schlüssel:** Jeder Schlüssel in `settings` und jeder Schlüssel in `mod_data[<mod>]` ist ein eigener Eintrag. Ändert sich ein Wert, wird nur dieser eine Eintrag übertragen.
8. Löschungen werden nicht übertragen.

### 0.2 Eigene Arbeitskopie (einmal)

```powershell
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2
git fetch origin
git worktree add -B isaa/08 C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa08 origin/master
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa08
uv venv -p 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install pytest pytest-asyncio
uv pip install -e toolboxv2/mods/isaa
uv pip install -e ".[web]"
```

In **jeder** neuen Shell:

```powershell
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa08
.\.venv\Scripts\Activate.ps1
Get-ChildItem Env: | Where-Object { $_.Name -match '^(TOOLBOX|TB_)' } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:MOCK_EXTERNAL_SERVICES = "true"; $env:IS_OFFLINE_DB = "true"
```

Bekannte Fallen:
- Ohne das Entfernen der `TOOLBOX*`/`TB_*`-Variablen hängt pytest.
- Immer `.\.venv\Scripts\python.exe` bzw. die aktivierte venv verwenden.
- `python -c "..."` mit Anführungszeichen ist unter pwsh unzuverlässig: Prüfskripte als Datei mit `Set-Content` anlegen und mit `python <datei>` starten.

### 0.3 Regeln

1. **Nur die Dateien aus Abschnitt 1 ändern.** Jede andere Datei gehört einer anderen Instanz oder ist gesperrt.
2. **Kein Grün durch Wegsehen:**
   - kein `skip`, `skipIf`, `xfail`, `importorskip`, `continue-on-error`, `|| true`;
   - keine Assertion abschwächen und keinen Erwartungswert auf den Ist-Wert setzen.
   
   Ausnahmen gibt es nur dort, wo ein Teil ausdrücklich sagt „dieser Test wird geändert, weil …“ (Korrektur nach Owner-Entscheidung).
3. **Keine Dopplungen:** Vor jeder neuen Funktion `git grep -n "<zweck>"`. Gibt es etwas Passendes, das nicht in diesem Auftrag steht → Abweichung (0.6).
4. **Keine Geheimnisse** in Dateien, Logs, Commits oder im Bericht:
   - kein Kontoschlüssel, kein Token, keine MinIO-Daten;
   - Schlüssel in Belegen nur als Länge oder `sha256(...)[:8]`;
   - Log-Zeilen enthalten nie Werte aus `settings`/`mod_data`, nur Schlüsselnamen und Anzahlen.
5. **Ein Commit je Teil.** Dateien einzeln hinzufügen (`git add <pfad>`), nie `git add -A`, `git add .`, `git commit -a`. Vor jedem Commit `git status --short`: nur deine Dateien.
6. Kein `--force`, kein `--no-verify`, keine Merge-Commits, keine fremden Commits umschreiben.
7. **Owner-Umgebung schützen:** Ports 8467/8468 nicht belegen, keine fremden Prozesse beenden, nichts unter dem Haupt-Checkout oder im Benutzerprofil löschen. Keine Anfragen an `simplecore.app` in diesem Auftrag (Live-Teile erst nach D1).

### 0.4 Commit und Push

Commit-Nachricht:
- erste Zeile: `08-<Teil> (<alter Name>): <kurz>`, z. B. `08-2 (K3): needs_account richtig herum`;
- darunter: `Auftrag 08, Teil <n>.` und die letzte Zeile der Abnahme-Ausgabe.

```powershell
git fetch origin
git rebase origin/master
# nach einem Rebase mit fremden Commits: Gate (0.5) erneut laufen lassen
git push origin HEAD:master
```

Push abgelehnt (`non-fast-forward`) → die drei Schritte wiederholen, höchstens dreimal, danach melden. Rebase-Konflikt → `git rebase --abort`, stoppen, melden.

### 0.5 Gate: keine neue rote Test-ID

Gate-Liste (in Teil 1 als Ausgangsstand messen; neue Testdateien kommen hinzu, sobald sie existieren):

```powershell
$GATE = @(
  "toolboxv2/tests/test_core/test_device_account.py",
  "toolboxv2/tests/test_core/test_local_trust.py",
  "toolboxv2/tests/test_core/test_account_link_flow.py",
  "toolboxv2/tests/test_core/test_wizard_api.py",
  "toolboxv2/tests/test_core/test_wizard_e2e_http.py",
  "toolboxv2/tests/test_core/test_first_run_onboarding.py",
  "toolboxv2/tests/test_core/test_env_single_file.py",
  "toolboxv2/tests/test_core/test_data_root.py",
  "toolboxv2/tests/test_core/test_tray_daemon_integration.py",
  "toolboxv2/tests/test_web/test_workers/test_fast/test_session_custom.py",
  "toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/"
)
python -m pytest $GATE -q -p no:cacheprovider -rfE 2>&1 | Tee-Object -FilePath ..\isaa08_gate_<teil>.txt | Select-Object -Last 15
```

Nach jedem Teil:
- Die Menge der roten IDs muss eine **Teilmenge** der roten IDs aus Teil 1 sein.
- Ausnahme: IDs, die dieser Teil ausdrücklich entfernt.
- Jede neue rote ID ist eine Abweichung (0.6).

### 0.6 Abweichung: was tun

Eine Abweichung liegt vor, wenn:
- „Vorher prüfen“ etwas anderes zeigt als angegeben (andere Zeile, anderer Treffer, mehr oder weniger Treffer);
- ein Schritt eine Datei außerhalb von Abschnitt 1 bräuchte;
- ein Test aus diesem Auftrag nach den Schritten rot bleibt;
- das Gate eine neue rote ID zeigt.

Dann:
1. Sofort anhalten. Nicht „passend machen“, nicht umgehen.
2. Eigene Änderungen dieses Teils zurücknehmen: `git checkout -- <pfad>` bzw. `git restore --staged <pfad>`.
3. Im Bericht unter „Abweichungen“ eintragen: **Teil · erwartet (wörtlich) · gefunden (wörtlich, Befehl + Ausgabe) · was du zurückgenommen hast**.
4. Steht im Teil „Hängt ab von Teil X“ und X ist gestoppt → diesen Teil ebenfalls als „gestoppt (wegen X)“ melden.
5. Teile ohne Abhängigkeit weiter abarbeiten.

### 0.7 Bericht

`C:\Users\Markin\Workspace\ISAABOX\isaa08_bericht.md` (außerhalb des Repos), derselbe Text als Antwort:

| Teil | Alter Name | Ergebnis (`erledigt` / `gestoppt` / `nur live prüfbar`) | Commit | Abnahme (letzte Zeile, wörtlich) | Bemerkung |
|---|---|---|---|---|---|

Darunter:
- Abweichungen (Format 0.6);
- neue rote IDs (sollte leer sein);
- rote IDs aus fremden Commits;
- Gefundenes, das du nicht geändert hast.

---

## 1. Deine Dateien

**Ändern:**

- `toolboxv2/utils/clis/first_run.py`
- `toolboxv2/utils/workers/fast/wizard_api.py` (nur der Block `# --- Account link (WP1 K3)`)
- `toolboxv2/utils/workers/fast/local_ui.py` (nur Route `/` und `/account`)
- `toolboxv2/utils/workers/fast/account_page.py`
- `toolboxv2/utils/workers/session.py` (nur `get_session_from_request`, `get_session_from_request_sync`, der Block `Local Device Trust`)
- `toolboxv2/utils/workers/server_worker.py` (nur der Block „Loopback trust“, heute Zeilen 1681–1719)
- `toolboxv2/mods/CloudM/module.py` (nur Zeilen 93–112)
- `toolboxv2/mods/CloudM/auth/device_account.py`
- `toolboxv2/mods/CloudM/auth/models.py` (nur `class UserData`)
- `toolboxv2/mods/CloudM/auth/api_session.py` (nur `get_user_data`, `update_user_data`, neue Funktion `get_sync_key`)
- `toolboxv2/mods/CloudM/UserDataAPI.py` (nur `get_secure_storage_credentials`)
- `toolboxv2/mods/CloudM/LiveSync/__init__.py` (nur `tb_get_my_minio_credentials`)

**Neu:**

- `toolboxv2/mods/CloudM/LiveSync/data_api_sync.py`
- `toolboxv2/mods/CloudM/LiveSync/account_bridge.py`
- `toolboxv2/tests/test_core/test_sync_key.py`
- `toolboxv2/tests/test_core/test_data_api_sync.py`
- `toolboxv2/tests/test_core/test_account_bridge.py`
- `toolboxv2/tests/test_core/test_trust_provider.py`

**Tests ändern (nur wo der Teil es sagt):**

- `toolboxv2/tests/test_core/test_account_link_flow.py`
- `toolboxv2/tests/test_core/test_local_trust.py`
- `toolboxv2/tests/test_core/test_wizard_api.py`
- `toolboxv2/tests/test_core/test_wizard_e2e_http.py`
- `toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_credentials_api.py`

**Löschen (Teil 7):**

- `toolboxv2/mods/CloudM/sync/` (alle Dateien)
- `toolboxv2/tests/test_core/test_device_sync.py`
- `toolboxv2/tests/test_core/test_blob_sync.py`
- `toolboxv2/tests/test_core/test_global_sync.py`

**Nur lesen, nicht ändern:**

- `LiveSync/share_store.py` (gehört Auftrag 05)
- `LiveSync/crypto.py`, `conflict.py`, `index.py`, `protocol.py`, `client.py`, `server.py`, `minio_helper.py`
- `utils/extras/blobs.py`
- `auth/user_store.py`
- `utils/toolbox.py`

**Gesperrt:**
- alle Dateien der Aufträge 02–07;
- `pyproject.toml`, `docs/`, `.github/`;
- `cli_oauth.py` ab Zeile 105, `local_cli.py`;
- der Dispatcher.

---

## 2. Teile in dieser Reihenfolge

### Teil 1 – Ausgangsstand messen (kein Code)

**Schritte:**

1. Gate (0.5) laufen lassen und als `..\isaa08_gate_1.txt` sichern. Die roten IDs sind der Ausgangsstand.
2. Diese Anker prüfen:

```powershell
git log -1 --oneline
git grep -n '"needs_account": owner is not None' -- toolboxv2/utils/clis/first_run.py
git grep -n 'not os.environ.get("TOOLBOXV2_REMOTE_BASE")' -- toolboxv2/utils/workers/server_worker.py
git grep -n 'is_remote = bool(os.environ.get("TOOLBOXV2_REMOTE_BASE"))' -- toolboxv2/mods/CloudM/module.py
git grep -n 'the trust itself is not loopback-checked' -- toolboxv2/utils/workers/session.py
git grep -n 'remote = None' -- toolboxv2/mods/CloudM/sync/global_sync.py
git grep -n '"TOOLBOXV2_REMOTE_BASE": "https://simplecore.app"' -- toolboxv2/utils/clis/config_wizard.py
git grep -n "user_pass_sync" -- toolboxv2/mods/CloudM/auth/models.py
```

**Erwartet:**
- je genau **ein** Treffer für die ersten vier `git grep`;
- **zwei** Treffer für `remote = None` (`global_sync.py:156` und `:164`);
- **drei** Treffer in `config_wizard.py`;
- **kein** Treffer für `user_pass_sync` in `auth/models.py`;
- `git log` zeigt `d885fa2` oder einen neueren Commit.

Bei einem neueren Commit: `git log d885fa2..HEAD --stat` ansehen. Berührt er eine Datei aus Abschnitt 1 → Abweichung (0.6), **alles stoppen**.

Kein Commit.

---

### Teil 2 – K3: `needs_account` richtig herum

**Fehler:** `first_run.py:89` setzt `"needs_account": owner is not None`. Das ist umgekehrt: nach dem Verknüpfen `true`, vorher `false`.

**Vorher prüfen:**

```powershell
git grep -n "needs_account" -- toolboxv2/utils/clis/first_run.py toolboxv2/tests/test_core/test_account_link_flow.py toolboxv2/tests/test_core/test_first_run_onboarding.py
```

Erwartet (Zeilennummern ±2):
- `first_run.py:89`;
- `test_account_link_flow.py:103, 105, 107, 110, 121, 124, 145`;
- `test_first_run_onboarding.py:83`.

**Schritte:**

1. In `onboarding_state()` (`first_run.py`) die Zeile ersetzen durch:
   ```python
   "needs_account": profile in ("local", "developer") and owner is None,
   ```
2. Sonst nichts an `onboarding_state` ändern.

**Tests** (`test_account_link_flow.py`). **Dieser Test wird geändert, weil er das falsche Verhalten festschreibt (Owner-Entscheidung 8 aus dem Handoff: Kontopflicht).**

1. `test_needs_account_true_after_link` umbenennen in `test_needs_account_false_after_link` und `assertTrue(state["needs_account"])` → `assertFalse(...)`.
2. `test_needs_account_false_without_owner_file`:
   - umbenennen in `test_needs_account_true_without_owner_local_profile`;
   - im Test das Profil `local` setzen. Lies zuerst, wie `setUp` das Manifest bzw. Profil anlegt, und verwende **denselben Weg**. Gibt es keinen, `mock.patch` auf `ManifestLoader` wie in `test_first_run_onboarding.py` (dort nachsehen);
   - erwarten: `assertTrue(state["needs_account"])`.
3. Neuer Test `test_needs_account_false_for_server_profile`: Profil `server`, kein Gerätekonto → `False`.
4. Neuer Test `test_needs_account_false_without_profile`: kein Profil, kein Gerätekonto → `False`. Vor dem Onboarding gilt `first_run`, nicht die Kontopflicht.
5. `test_first_run_and_needs_account_independent`: nach `_link` gilt jetzt `needs_account` = `False`. Die Assertion ändern auf `assertFalse(state["needs_account"])`; `first_run` bleibt `True`.
6. `test_wizard_state_endpoint_reports_account`: nach `_link` → `assertFalse(res["needs_account"])`, `account` bleibt `"markin"`.
7. `test_first_run_onboarding.py:83` **nicht** ändern (Profil `server` → `False`, bleibt richtig).

**Abnahme:**

```powershell
python -m pytest toolboxv2/tests/test_core/test_account_link_flow.py toolboxv2/tests/test_core/test_first_run_onboarding.py -q -p no:cacheprovider
```

Letzte Zeile: `N passed`, `0 failed`. Dazu das Gate.

**Commit:** `08-2 (K3): needs_account nur bei Profil local/developer ohne Geraetekonto`

---

### Teil 3 – K3: Route `/` zeigt ohne Gerätekonto die Kontoseite

Hängt ab von Teil 2.

**Vorher prüfen:**

```powershell
git grep -n '@app.get("/", auth=False)' -- toolboxv2/utils/workers/fast/local_ui.py
git grep -n '@app.get("/account", auth=False)' -- toolboxv2/utils/workers/fast/local_ui.py
git grep -n "get_installation_state" -- toolboxv2/utils/workers/fast/local_ui.py
```

Erwartet: je ein Treffer für die Routen. `get_installation_state` ist importiert und wird in `root` benutzt.

Dann `git grep -n "def get_installation_state"` ausführen und die Funktion lesen. Gibt sie `needs_account` aus `onboarding_state()` weiter? Wenn **nicht**, ist sie eine Dopplung zu `onboarding_state` → Abweichung (0.6), nicht selbst zusammenführen.

**Schritte:**

1. In `local_ui.py` den Rumpf von `account(...)` (heute: Seite in eine Temp-Datei schreiben und ausliefern) in eine Hilfsfunktion `_serve_account_page()` verschieben. `account(...)` ruft nach `_ensure_local` nur noch `return _serve_account_page()` auf. Inhaltlich nichts ändern.
2. In `root(...)` direkt **nach** dem `first_run`-Block und **vor** dem Dashboard-Block einfügen:
   ```python
   if get_installation_state().get("needs_account", False):
       return _serve_account_page()
   ```

**Tests:**

1. `test_wizard_e2e_http.py` und `test_wizard_api.py` lesen. Jeder bestehende Test, der nach gesetztem Profil `local`/`developer` das Dashboard erwartet, bekommt im `setUp` (oder in der Fixture) ein `device_owner.json` im Daten-Ordner des Tests: `{"user_id":"u1","username":"t","base_url":"https://simplecore.app","linked_at":0,"migrated_from":null}`. Das ist die neue Voraussetzung aus dem WP1-Dokument, keine Abschwächung.
2. Neuer Test in `test_wizard_e2e_http.py`, im Stil der Nachbartests (gleicher Server-Start):
   - `test_root_serves_account_page_without_device_account`:
     - Profil `local`, **kein** `device_owner.json`;
     - `GET /` liefert 200;
     - der Body enthält `<title>ToolBoxV2 - Link account</title>`.
3. Neuer Test `test_root_serves_dashboard_with_device_account`:
   - Profil `local`, mit `device_owner.json`;
   - der Body enthält **nicht** `Link account`.

**Abnahme:**

```powershell
python -m pytest toolboxv2/tests/test_core/test_wizard_e2e_http.py toolboxv2/tests/test_core/test_wizard_api.py -q -p no:cacheprovider
```

`0 failed`, dazu das Gate.

Hinweis: `test_wizard_e2e_http` braucht `waitress`. Fehlt es, ist `.[web]` nicht installiert (0.2), das ist dann **kein** Testfehler.

**Commit:** `08-3 (K3): / zeigt ohne Geraetekonto die Kontoseite`

---

### Teil 4 – K3: Anmeldung im Hintergrund, Status abfragen, Loopback-Schutz der Konto-Routen

Hängt ab von Teil 2.

**Fehler:**
- `POST /api/account/start` blockiert die Anfrage bis zu 180 s;
- `/api/account/*` prüft nicht auf Loopback.

**Vorher prüfen:**

```powershell
git grep -n '@app.\(get\|post\)("/api/account' -- toolboxv2/utils/workers/fast/wizard_api.py
git grep -n "def _ensure_local" -- toolboxv2/utils/workers/fast/
git grep -n "fetch('/api/account" -- toolboxv2/utils/workers/fast/account_page.py
```

Erwartet:
- drei Routen (`status`, `sync`, `start`);
- `_ensure_local` nur in `local_ui.py`;
- zwei `fetch` in `account_page.py` (`start`, `status`).

**Schritte:**

1. `first_run.py`: unter `_start_account_link` einfügen:
   ```python
   import threading as _threading
   _LINK_LOCK = _threading.Lock()
   _LINK = {"state": "idle", "error": None}

   def start_account_link_bg(method=None) -> dict:
       """Start one background link run (WP1 K3). Never blocks."""
       with _LINK_LOCK:
           if _LINK["state"] == "waiting":
               return dict(_LINK)
           _LINK.update(state="waiting", error=None)

       def _run():
           owner = _start_account_link(None, method=method)
           with _LINK_LOCK:
               if owner:
                   _LINK.update(state="linked", error=None)
               else:
                   _LINK.update(state="error", error="remote login failed or timed out")

       _threading.Thread(target=_run, daemon=True, name="account-link").start()
       return dict(_LINK)
   ```
2. `account_state()`: In **beiden** Rückgaben die Felder `"state"` und `"error"` ergänzen:
   - mit Gerätekonto: `state = "linked"`;
   - sonst: `_LINK["state"]` und `_LINK["error"]`.
   - Unter dem Lock lesen.
3. `wizard_api.py`, `account_start`:
   - die Methodenprüfung bleibt;
   - statt `_start_account_link(...)` wird `start_account_link_bg(method)` aufgerufen;
   - Rückgabe: `return 202, account_state()`.
4. `wizard_api.py`: Jede der drei Konto-Routen beginnt mit:
   ```python
   from toolboxv2.utils.workers.fast.local_ui import _ensure_local
   err = _ensure_local(request)
   if err is not None:
       return err
   ```
   Der Import steht **in** der Funktion, nicht oben in der Datei (`local_ui` baut beim Import eine FastTB-App).
5. `account_page.py`:
   - nach dem `POST /api/account/start` **nicht** auf das Login warten;
   - alle 2000 ms `GET /api/account/status` abfragen;
   - bei `state == "linked"` → `location.replace('/')`;
   - bei `state == "error"` → `error` in der Statuszeile anzeigen und das Abfragen beenden.
   - Keine externen Skripte.

**Tests** (`test_account_link_flow.py`, neue Methoden):

1. `test_start_bg_runs_one_thread`:
   - `mock.patch` auf `first_run._start_account_link` mit einer Funktion, die auf ein `threading.Event` wartet;
   - zweimal `start_account_link_bg()` → beide Rückgaben `state == "waiting"`, die Patch-Funktion wurde **genau einmal** aufgerufen;
   - Event setzen, auf den Thread warten (Name `account-link`, `join(timeout=5)`).
2. `test_start_bg_error_state`: Patch liefert `None` → nach dem Join `account_state()["state"] == "error"`.
3. `test_start_bg_linked_state`: Patch legt `device_owner.json` an und liefert ein Dict → `account_state()["state"] == "linked"`.
4. `test_account_routes_refuse_non_loopback_host`:
   - Routen wie in `test_wizard_state_endpoint_reports_account` registrieren;
   - Request-Attrappe mit `headers = {"host": "evil.example"}`;
   - alle drei Routen liefern ein Fehlerergebnis mit `exec_code` 403.
   - Wie das Ergebnis von `_ensure_local` aussieht, im Code nachlesen und genau das prüfen.
5. Nach jedem Test `_LINK` auf `{"state":"idle","error":None}` zurücksetzen (`addCleanup`).

**Abnahme:** `python -m pytest toolboxv2/tests/test_core/test_account_link_flow.py -q -p no:cacheprovider` → `0 failed`, dazu das Gate.

**Commit:** `08-4 (K3): Konto-Login im Hintergrund, Status-Abfrage, Loopback-Schutz`

---

### Teil 5 – K2: Loopback-, Header- und Host-Prüfung für das Gerätevertrauen

**Fehler:** `SessionManager` vertraut **jeder** Anfrage ohne Anmeldung als Geräte-Nutzer (`session.py`, Docstring „the trust itself is not loopback-checked“).

**Vorher prüfen:**

```powershell
git grep -n "_device_trust" -- toolboxv2/utils/workers/session.py
```

Erwartet: Definition `self._device_trust`; zwei Stellen `if self._device_trust is not None:` (in `get_session_from_request` und `get_session_from_request_sync`); `enable_local_trust`; `_device_trust_session`.

**Schritte:**

1. Im Block `Local Device Trust (WP1 K2)` eine Methode hinzufügen:
   ```python
   _TRUST_HOSTS = ("127.0.0.1", "localhost", "[::1]", "::1")
   _TRUST_ADDRS = ("127.0.0.1", "::1")
   _PROXY_KEYS = ("HTTP_X_FORWARDED_FOR", "HTTP_X_REAL_IP", "HTTP_FORWARDED")

   def _device_trust_allowed(self, environ: Dict, headers: Optional[Dict[str, str]]) -> bool:
       """All must hold, else anonymous: loopback peer, no proxy headers, loopback Host."""
       if self._device_trust is None or not self._device_trust.get("user_id"):
           return False
       if environ.get("REMOTE_ADDR", "") not in self._TRUST_ADDRS:
           return False
       if any(environ.get(k) for k in self._PROXY_KEYS):
           return False
       h = {k.lower(): v for k, v in (headers or {}).items()}
       if any(h.get(k) for k in ("x-forwarded-for", "x-real-ip", "forwarded")):
           return False
       host = environ.get("HTTP_HOST", "") or h.get("host", "")
       if host.startswith("["):
           host = host.split("]")[0] + "]"
       else:
           host = host.split(":")[0]
       return host.lower() in self._TRUST_HOSTS
   ```
2. An **beiden** Stellen `if self._device_trust is not None:` ersetzen durch `if self._device_trust_allowed(environ, headers):`.
3. Im Docstring von `enable_local_trust` den Satz „the trust itself is not loopback-checked“ ersetzen durch „Each request is checked by _device_trust_allowed (loopback peer, no proxy headers, loopback Host).“

**Tests** (`test_local_trust.py`): Vorhandene Tests lesen. Tests, die heute nur mit einem leeren `environ` arbeiten und das Vertrauen erwarten, bekommen `environ = {"REMOTE_ADDR": "127.0.0.1", "HTTP_HOST": "127.0.0.1:8467"}`. Das ist die neue Bedingung aus dem WP1-Dokument, keine Abschwächung.

Neue Tests, je einmal für die asynchrone und die synchrone Funktion:

| Fall | Erwartet |
|---|---|
| Provider gesetzt, `REMOTE_ADDR=127.0.0.1`, `HTTP_HOST=127.0.0.1:8467` | `anonymous` False, `user_id` des Providers |
| dasselbe mit `REMOTE_ADDR=::1`, `HTTP_HOST=[::1]:8467` | Konto |
| `REMOTE_ADDR=192.168.1.5` | anonym |
| Loopback + `HTTP_X_FORWARDED_FOR=1.2.3.4` | anonym |
| Loopback + `HTTP_HOST=evil.example` | anonym |
| Provider `None` | anonym |
| Provider ohne `user_id` (`{"username":"x"}`) | anonym |
| kein `enable_local_trust`-Aufruf | anonym |

Der vorhandene Test `test_trust_does_not_shadow_api_key_auth` bleibt grün.

**Abnahme:** `python -m pytest toolboxv2/tests/test_core/test_local_trust.py -q -p no:cacheprovider` → `0 failed`, dazu das Gate.

**Commit:** `08-5 (K2): Geraetevertrauen nur fuer Loopback ohne Proxy-Header`

---

### Teil 6 – K2: Aktivierung nach Profil statt nach `TOOLBOXV2_REMOTE_BASE`

**Fehler:**
- Das Profil `local` schreibt `TOOLBOXV2_REMOTE_BASE=https://simplecore.app` in die `.env` (`config_wizard.py:226`).
- `server_worker.py:1685` und `CloudM/module.py:95` aktivieren das Gerätevertrauen aber nur **ohne** diese Variable.
- Auf einer echten Installation greift K2 also nie.

**Vorher prüfen:** die Anker aus Teil 1 für `server_worker.py` und `module.py`. Dazu:

```powershell
git grep -n "self.config.environment" -- toolboxv2/utils/workers/server_worker.py
git grep -n "def read_device_owner" -- toolboxv2/mods/CloudM/auth/device_account.py
```

Erwartet: Im Block 1681–1719 steht `self.config.environment != "production"`; `read_device_owner` ist vorhanden.

**Schritte:**

1. `device_account.py`, neue Funktion unter `read_device_owner`:
   ```python
   def local_trust_provider(app, is_production: bool) -> Optional[dict]:
       """Provider for SessionManager.enable_local_trust (WP1 K2).
       Only profile local/developer, never production, only with a linked
       device account. No local-admin fallback (account is mandatory)."""
       if is_production:
           return None
       profile = getattr(getattr(getattr(app, "manifest", None), "app", None), "profile", None)
       value = getattr(profile, "value", profile)
       if value not in ("local", "developer"):
           return None
       owner = read_device_owner(app)
       if not owner or not owner.get("user_id"):
           return None
       return {"user_id": owner["user_id"], "username": owner.get("username", ""), "level": -1}
   ```
2. `server_worker.py`: Den ganzen Block ab dem Kommentar `# Loopback trust: local installs` bis einschließlich der beiden `logger.warning(f"Loopback trust not enabled...")` ersetzen durch:
   ```python
   # Device trust (WP1 K2): profile local/developer + linked device account.
   try:
       from toolboxv2.mods.CloudM.auth.device_account import local_trust_provider
       provider = local_trust_provider(self._app, self.config.environment == "production")
   except Exception as e:
       provider = None
       logger.warning(f"Device trust check failed: {type(e).__name__}")
   self._session_manager.enable_local_trust(provider)
   if provider:
       logger.info(f"Device trust on for {provider.get('username', '')}")
       # Teil 10 haengt hier den Konto-Sync an.
   else:
       logger.info("Device trust off (no device account or not a local profile)")
   ```
   Damit entfallen `ensure_local_admin` und die `wp1-sync`-Schleife an dieser Stelle.
3. `CloudM/module.py`, Zeilen 93–112: Die Entscheidung ersetzen durch genau diese Reihenfolge:
   - Profil `server` → wie heute `await self.app.session.login()`.
   - Sonst, wenn `read_device_owner(self.app)` etwas liefert → `self._apply_local_admin_to_session(owner)`. **Kein** `session.login()`, damit es offline geht. `base` bleibt, wie `_apply_local_admin_to_session` es heute setzt.
   - Sonst wie bisher: `is_remote` → `await self.app.session.login()`, andernfalls lokaler Admin über `ensure_local_admin`. Das ist der alte Weg für die CLI vor dem Verknüpfen.
   
   Fehler in `read_device_owner` → wie heute `self.logger.warning("local_admin init skipped: ...")`, nur mit Fehlertyp.

**Tests** (`test_trust_provider.py`, neu, unittest; App-Attrappe mit `manifest.app.profile` als Objekt mit `.value`; `read_device_owner` per `mock.patch`):

| Fall | Erwartet |
|---|---|
| `is_production=True`, Profil local, Konto | `None` |
| Profil `server`, Konto | `None` |
| kein Profil (`profile=None`), Konto | `None` |
| Profil `local`, kein Konto | `None` |
| Profil `local`, Konto ohne `user_id` | `None` |
| Profil `local`, Konto | `{"user_id": …, "username": …, "level": -1}` |
| Profil `developer`, Konto | wie oben |

**Prüfung bei Abweichung:** Nach Schritt 2 wird `test_tray_daemon_integration.py` oder ein Test unter `test_web/test_workers/` rot, weil er sich auf das Vertrauen zum lokalen Admin verlassen hat? Dann **keinen** Rückfall auf den lokalen Admin einbauen → Abweichung (0.6) mit Test-ID und Fehlerzeile. Das entscheidet der Owner.

**Nur live prüfbar (K4):** dass ein Start ohne Netz mit Gerätekonto angemeldet bleibt. Im Bericht als „nur live prüfbar“ vermerken.

**Abnahme:** `python -m pytest toolboxv2/tests/test_core/test_trust_provider.py -q -p no:cacheprovider` → `7 passed`, dazu das Gate (`test_trust_provider.py` ab jetzt in `$GATE`).

**Commit:** `08-6 (K2): Geraetevertrauen nach Profil + Geraetekonto, kein REMOTE_BASE-Schalter`

---

### Teil 7 – Dopplungen aus S1–S3 entfernen

Hängt ab von Teil 6 (die `wp1-sync`-Schleife ist dort schon entfernt).

**Vorher prüfen:**

```powershell
git grep -n "mods\.CloudM\.sync\|CloudM\.sync\.\|run_sync_once" -- '*.py'
```

Erwartet, genau diese 8 Treffer (Zeilennummern ±3):
- `toolboxv2/mods/CloudM/auth/device_account.py:107` und `:108`
- `toolboxv2/mods/CloudM/sync/global_sync.py:144`
- `toolboxv2/tests/test_core/test_blob_sync.py:13`
- `toolboxv2/tests/test_core/test_device_sync.py:18`
- `toolboxv2/tests/test_core/test_global_sync.py:16`
- `toolboxv2/utils/workers/fast/wizard_api.py:177` und `:183`

`server_worker.py` darf hier **nicht** mehr auftauchen (in Teil 6 entfernt). Jeder andere Treffer → Abweichung (0.6).

Achtung: Namen wie `sync_all` gibt es auch im ISAA-VFS (`vfs_v2.py`, `flow_agent.py` …). Die sind **nicht** gemeint und werden nicht angefasst.

**Schritte:**

1. `git rm -r toolboxv2/mods/CloudM/sync`
2. `git rm toolboxv2/tests/test_core/test_device_sync.py toolboxv2/tests/test_core/test_blob_sync.py toolboxv2/tests/test_core/test_global_sync.py`. **Diese Tests werden gelöscht, weil der Code, den sie prüfen, auf Owner-Entscheidung 1 entfernt wird.** Die Fälle kommen in Teil 9/10 gegen LiveSync wieder.
3. `device_account.py`: Block `# WP1 S3: best-effort sync kick-off ...` (try/except mit `run_sync_once`) entfernen.
4. `wizard_api.py`: Route `@app.post("/api/account/sync", auth=False)` samt Funktion entfernen. Sie kommt in Teil 10 neu.
5. Die drei gelöschten Testdateien aus `$GATE` nehmen, falls sie dort stehen. Ihre IDs zählen nicht als „neu rot“.

**Abnahme:** Den Befehl aus „Vorher prüfen“ erneut ausführen → **0 Treffer**, dazu das Gate.

**Commit:** `08-7 (S1-S3): doppelte Sync-Maschine entfernt, LiveSync uebernimmt`

---

### Teil 8 – Kontoschlüssel `user_pass_sync`, keine Geheimnisse in Nutzer-Antworten, MinIO-Vergabe aus

**Vorher prüfen:**

```powershell
git grep -n "user_pass_sync\|generate_symmetric_key" -- toolboxv2/mods/CloudM/types.py toolboxv2/utils/security/cryp.py
git grep -n "return Result.ok(user.to_dict())" -- toolboxv2/mods/CloudM/auth/api_session.py
git grep -n "get_secure_storage_credentials\|tb_get_my_minio_credentials" -- ':!*.md'
```

Erwartet:
- `types.py:27` und `:41`; `cryp.py`: `def generate_symmetric_key` (Fernet-Schlüssel, URL-sicheres Base64, 32 Byte; dasselbe Format, das `LiveSync/crypto._decode_key` erwartet);
- `api_session.py`: zwei Treffer (`get_user_data`, `update_user_data`);
- MinIO-Vergabe: Treffer nur in `UserDataAPI.py`, `LiveSync/__init__.py`, Tests, Markdown/HTML-Doku. **Ein Aufruf aus Web-JS, Tauri oder einem anderen Modul** → Abweichung (0.6), nur die Abschaltung (Schritt 5) nicht machen, den Rest des Teils schon.

**Schritte:**

1. `auth/models.py`, `class UserData`: nach `mod_data` einfügen:
   ```python
   # Account sync key (Fernet/urlsafe-b64, 32 bytes). Leaves the server only via
   # CloudM.Auth.get_sync_key to the account owner. Never in API responses.
   user_pass_sync: str = ""
   ```
   und direkt unter `to_dict`:
   ```python
   _SECRET_FIELDS = ("user_pass_sync", "minio_credentials")

   def to_public_dict(self) -> dict:
       """to_dict without secrets - the only form that may leave via an API."""
       d = self.to_dict()
       for k in self._SECRET_FIELDS:
           d.pop(k, None)
       return d
   ```
   `to_dict` bleibt unverändert, denn die DB speichert alles.
2. `api_session.py`: in `get_user_data` und `update_user_data` `Result.ok(user.to_dict())` → `Result.ok(user.to_public_dict())`.
3. `api_session.py`, neue Funktion direkt unter `get_user_data`:
   ```python
   @export(mod_name=Name, version=version, api=True, request_as_kwarg=True)
   async def get_sync_key(app: App = None, request=None, data: dict = None) -> ApiResult:
       """Account sync key for the authenticated owner only (WP1, Auftrag 08).
       The user comes from the Bearer token, never from parameters."""
       if app is None:
           app = get_app(f"{Name}.get_sync_key")
       auth = await validate_session(app, request=request)
       if auth.is_error():
           return Result.default_user_error("Not authenticated", exec_code=401)
       user_id = auth.get().get("user_id")
       user = await _load_user(app, user_id)
       if not user:
           return Result.default_user_error("User not found", exec_code=404)
       if not user.user_pass_sync:
           from toolboxv2.utils.security.cryp import Code
           user.user_pass_sync = Code.generate_symmetric_key()
           await _save_user(app, user)
       try:
           app.audit_logger.log_action(user_id=user_id, action="user.sync_key",
                                       resource=f"/users/{user_id}", status="SUCCESS",
                                       details={})
       except Exception: pass
       return Result.ok({"key": user.user_pass_sync})
   ```
   Vorher mit `git grep -n "def validate_session" toolboxv2/mods/CloudM/auth/api_session.py` sicherstellen, dass `validate_session` im selben Modul liegt und `Result` mit `.is_error()`/`.get()` liefert. Wenn nicht → Abweichung.
4. `local_admin.py` wird **nicht** geändert (`get_local_admin` ist nur nativ, nie HTTP).
5. MinIO-Vergabe abschalten (nur wenn „Vorher prüfen“ sauber war):
   - `UserDataAPI.py`: `get_secure_storage_credentials` samt `@export` entfernen; `_get_credential_broker` nur entfernen, wenn `git grep -n "_get_credential_broker"` danach keinen Treffer mehr hat.
   - `LiveSync/__init__.py`: `tb_get_my_minio_credentials` samt `@export` entfernen.
   - `test_LiveSync_credentials_api.py`: Tests, die diese Funktion aufrufen, entfernen. **Gelöscht, weil die Funktion auf Owner-Entscheidung 3 entfernt wird.** Stattdessen ein Test `test_minio_credentials_not_exported`: Auf `toolboxv2.mods.CloudM.LiveSync` gilt `hasattr(..., "tb_get_my_minio_credentials")` → `False`, ebenso auf `toolboxv2.mods.CloudM.UserDataAPI` für `get_secure_storage_credentials`.
   - Tests in der Datei, die `tb_generate_share_token` prüfen, bleiben.

**Tests** (`test_sync_key.py`, neu; DB-Helfer wie in `test_device_account.py` per `mock.patch` durch ein Dict ersetzen, `validate_session` patchen):

| Fall | Erwartet |
|---|---|
| `validate_session` liefert Fehler | Fehlerergebnis mit 401, im Fake-DB-Dict kein `user_pass_sync` gesetzt |
| gültige Sitzung, leerer Schlüssel | Ergebnis `key`, `base64.urlsafe_b64decode(key)` hat 32 Byte; gespeichert |
| zweiter Aufruf | derselbe Schlüssel, `_save_user` nicht erneut gerufen |
| `get_user_data` für diesen Nutzer | weder `user_pass_sync` noch `minio_credentials` im Ergebnis |
| `get_sync_key(..., data={"user_id": "fremd"})` | liefert den Schlüssel des **angemeldeten** Nutzers, nicht von `fremd` |
| `UserData.from_dict` mit altem Datensatz ohne `user_pass_sync` | lädt, Feld `""` |

**Abnahme:** `python -m pytest toolboxv2/tests/test_core/test_sync_key.py toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/ -q -p no:cacheprovider` → `0 failed`, dazu das Gate (`test_sync_key.py` ab jetzt in `$GATE`).

**Commit:** `08-8: Kontoschluessel user_pass_sync + get_sync_key, Secrets nie in Nutzer-Antworten, MinIO-Vergabe aus`

---

### Teil 9 – LiveSync: Ordner-Abgleich über die Data-API

Hängt ab von Teil 7.

**Vorher prüfen:**

```powershell
git grep -n "def encrypt_bytes\|def decrypt_bytes\|def compute_checksum\b\|def compute_checksum_file" -- toolboxv2/mods/CloudM/LiveSync/crypto.py
git grep -n "def save_conflict_copy\|def is_conflict_copy" -- toolboxv2/mods/CloudM/LiveSync/conflict.py
git grep -n "async def upsert_file\|async def get_file\|async def get_all_files\|async def init\|async def close" -- toolboxv2/mods/CloudM/LiveSync/index.py
git grep -n "def should_ignore" -- toolboxv2/mods/CloudM/LiveSync/protocol.py
git grep -n "class TransferError" -- toolboxv2/mods/CloudM/LiveSync/minio_helper.py
git grep -n "async def run_http" -- toolboxv2/utils/toolbox.py
git grep -n "def result_from_dict" -- toolboxv2/utils/system/types.py
git grep -n "async def list_data\|async def get_data\|async def set_data" -- toolboxv2/mods/CloudM/UserDataAPI.py
```

Jeder Befehl muss genau die genannten Definitionen finden. Danach lesen:
- `run_http`: Mit `method="POST"` gehen die Keyword-Argumente als JSON-Body; mit `GET` gehen sie **verloren**;
- `list_data`: gibt `[{path, size, updated_at}]` zurück;
- `get_data`: dekodiert JSON;
- `set_data`: ein Dict wird als JSON gespeichert.

Sieht etwas anders aus → Abweichung.

**Schritte:** Neue Datei `toolboxv2/mods/CloudM/LiveSync/data_api_sync.py` mit genau diesen Teilen:

1. **Konstanten:**
   - `PREFIX_GLOBAL = "livesync/global/"`, `PREFIX_ACCOUNT = "livesync/account/"`;
   - `MAX_BYTES = 50 * 1024 * 1024`, `PAYLOAD_V = 1`.
2. **`class DataApiTransport`** (`__init__(self, app)`):
   - **HTTPS-Pflicht:**
     - `base = (app.session.base or "")`;
     - erlaubt: `base.startswith("https://")`, oder der Host von `base` ist `127.0.0.1` bzw. `localhost`;
     - sonst `raise TransferError("insecure base")` (aus `.minio_helper`), **bevor** irgendetwas gesendet wird.
   - `async def _call(self, fn, **kw)`:
     - `res = await self.app.run_http("CloudM.UserDataAPI", fn, method="POST", **kw)`;
     - `r = Result.result_from_dict(**res)` (Import aus `toolboxv2`);
     - bei `r.is_error()`: `raise TransferError("offline")`, wenn `res` die Meldung `Session fetch failed` trägt; sonst `TransferError(f"{fn} failed")`;
     - Rückgabe `r.get()`.
   - `async def list(self, prefix) -> dict`: `_call("list_data", prefix=prefix, scope="private")` → `{path: (size, updated_at)}`.
   - `async def put(self, path, ciphertext: bytes)`: `_call("set_data", path=path, scope="private", data={"v": PAYLOAD_V, "enc": base64.b64encode(ciphertext).decode()})`. **Nur** diese zwei Schlüssel, nie Klartext, nie Dateiinhalt unverschlüsselt.
   - `async def get(self, path) -> bytes`:
     - `_call("get_data", path=path, scope="private")`;
     - Ergebnis muss ein Dict mit `v == 1` und `enc` sein, sonst `TransferError("bad payload")`;
     - Rückgabe `base64.b64decode(enc)`.
3. **`def _safe_rel(rel) -> Optional[str]`:**
   - `None` bei: leer, beginnt mit `/` oder `\`, enthält `:`, ein Teil ist `..` oder leer;
   - sonst der Pfad mit `/`.
4. **`async def sync_folder(transport, local_dir: Path, prefix: str, index: LocalIndex, key: str, device_id: str) -> dict`.**

   Rückgabe: `{"pushed": [], "pulled": [], "conflicts": [], "skipped": [], "unchanged": 0}`.

   *Index-Belegung* (eigene DB-Datei, nur von dieser Funktion benutzt; so in einem Kommentar festhalten):

   | Spalte | Inhalt |
   |---|---|
   | `checksum` | `compute_checksum` des **Klartexts** beim letzten gemeinsamen Stand |
   | `mtime` | `updated_at` der Remote-Seite beim letzten gemeinsamen Stand |
   | `size` | Remote-Größe |
   | `remote_key` | Remote-Pfad |
   | `sync_state` | `"synced"` |

   *Ablauf:*

   1. `remote = await transport.list(prefix)`, dann auf `rel = path[len(prefix):]` umschlüsseln. Pfade mit `_safe_rel(rel) is None` → `skipped`, loggen, nie schreiben.
   2. Lokal: Dateien unter `local_dir`, ohne `should_ignore(rel)` und ohne `is_conflict_copy(rel)`, ohne unsichere Pfade. Größer als `MAX_BYTES` → `skipped` und loggen. Prüfsumme mit `compute_checksum_file`.
   3. Für jede `rel` aus der Vereinigung: `base = await index.get_file(rel)`.
      - `L_changed`: lokal vorhanden und (`base` fehlt oder Prüfsumme ≠ `base.checksum`).
      - `R_changed`: remote vorhanden und (`base` fehlt oder `updated_at` ≠ `base.mtime`).
   4. Entscheidung, **genau** diese Tabelle:

      | Nr. | Lage | Aktion |
      |---|---|---|
      | 1 | `L_changed` und nicht `R_changed` | **push**: `encrypt_bytes(klartext, key)` → `put(prefix+rel)` |
      | 2 | nicht `L_changed` und `R_changed` | **pull**: `get` → `decrypt_bytes` → atomar schreiben |
      | 3 | weder noch | `unchanged += 1` |
      | 4 | beide, entfernter Klartext == lokaler Klartext | nichts schreiben, nur Index aktualisieren |
      | 5 | beide, verschieden | `save_conflict_copy(str(local_dir), rel, device_id)`, dann den Remote-Inhalt atomar schreiben; `conflicts.append(rel)` |
      | 6 | lokal gelöscht (`base` da, Datei fehlt) und nicht `R_changed` | nichts (keine Löschung übertragen, nichts neu laden) |
      | 7 | lokal gelöscht und `R_changed` | wie Nr. 2 |

      *Atomar schreiben:* in `<ziel>.livesync-tmp` im selben Ordner schreiben, dann `os.replace`. `decrypt_bytes` schlägt fehl (falscher Schlüssel, manipuliert) → `skipped` mit Grund `decrypt`, lokale Datei **unverändert**.
   5. Nach allen Pushes **ein** weiteres `transport.list(prefix)`, um `updated_at`/`size` der hochgeladenen Pfade für den Index zu holen. Für jeden bearbeiteten Pfad `index.upsert_file(rel, mtime=<remote updated_at>, size=<remote size>, checksum=<klartext-prüfsumme>, sync_state="synced", remote_key=prefix+rel)`.
   6. `get_data` nur für die Fälle 2, 4, 5 und 7, `set_data` nur für Fall 1. Sonst keine Inhalte übertragen.

**Tests** (`test_data_api_sync.py`, neu):
- `FakeTransport` mit einem Dict `store[path] = (payload_dict, updated_at)` und einem Zähler pro Methode; `updated_at` ist ein steigender Zähler;
- `put` speichert das Payload-Dict, so wie `DataApiTransport.put` es bauen würde. Dafür die Bau-Logik in eine Funktion `_payload(ciphertext)` legen, die beide benutzen;
- zwei Temp-Ordner A/B, je ein eigener `LocalIndex` (Temp-DB), gleicher Schlüssel aus `crypto.generate_encryption_key()`.

| Fall | Erwartet |
|---|---|
| A hat eine Binärdatei (4096 Zufallsbytes), A sync, B sync | B hat byte-gleiche Datei |
| zweiter Lauf A und B ohne Änderung | `put` 0, `get` 0 (nur `list`) |
| 3 Dateien synchron, A ändert eine | A: genau 1 `put`; B: genau 1 `get` |
| nur A ändert (offline), dann sync | Push, **keine** Konfliktkopie |
| A und B ändern dieselbe Datei verschieden, A sync, dann B sync | B: Datei = Inhalt von A (Server), Konfliktkopie `*.conflict.*` mit Bs Inhalt; `conflicts == [rel]` |
| A und B ändern gleich | keine Konfliktkopie |
| Remote-Pfad `livesync/global/../x` und `livesync/global//etc/x` | nichts außerhalb von B geschrieben, in `skipped` |
| Transport wirft `TransferError("offline")` | Ergebnis enthält `error: "offline"`, lokale Dateien unverändert (Prüfsummen vorher = nachher). Dafür fängt der **Aufrufer** (Teil 10) den Fehler; hier `assertRaises` |
| alle gespeicherten Payloads | Schlüssel genau `{"v","enc"}`; der dekodierte Chiffretext enthält den Klartext-Marker `b"KLARTEXT-MARKER"` **nicht** |
| `DataApiTransport` mit `session.base = "http://example.com"` | `TransferError("insecure base")`, `run_http` nie aufgerufen (Mock) |
| Datei mit 50 MB + 1 Byte (`f.truncate(MAX_BYTES + 1)`) | `skipped`, kein `put` |
| falscher Schlüssel bei B | `skipped` mit Grund `decrypt`, B-Datei unverändert |

**Abnahme:** `python -m pytest toolboxv2/tests/test_core/test_data_api_sync.py -q -p no:cacheprovider` → `12 passed`, dazu das Gate (`test_data_api_sync.py` ab jetzt in `$GATE`).

**Commit:** `08-9 (S2/S3): LiveSync-Ordnerabgleich ueber Data-API, Ende-zu-Ende verschluesselt`

---

### Teil 10 – Kontodaten Schlüssel für Schlüssel, Einstieg `sync_all`, Auslöser

Hängt ab von Teil 8 und Teil 9.

**Vorher prüfen:**

```powershell
git grep -n "_ADMIN_ONLY_SETTINGS" -- toolboxv2/mods/CloudM/auth/device_account.py
git grep -n "async def _load_user\|async def _save_user" -- toolboxv2/mods/CloudM/auth/user_store.py
git grep -n "Agents.*VFS.*global\|\"global\"" -- toolboxv2/mods/isaa/base/patch/power_vfs.py
```

Erwartet:
- `_ADMIN_ONLY_SETTINGS = ("anonymous", "local_admin")`;
- beide Nutzerfunktionen vorhanden;
- der globale Ordner ist `<data_dir>/Agents/VFS/global` (F22). `power_vfs` **nicht** importieren.

**Schritte:**

1. **Neue Datei `LiveSync/account_bridge.py`.**

   *Layout im Spiegelordner:*
   - `settings/<quote(key)>.json`
   - `mod_data/<quote(mod)>/<quote(key)>.json`
   - mit `quote = urllib.parse.quote(x, safe="")`.

   *Inhalt einer Datei:* `json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")`.

   *Funktionen:*
   - `def export_account(user, mirror: Path) -> int`:
     - für jeden Schlüssel in `user.settings` (ohne `_ADMIN_ONLY_SETTINGS`) und jeden Schlüssel in `user.mod_data[mod]` (nur wenn `mod_data[mod]` ein Dict ist; sonst `mod_data/<quote(mod)>/__value__.json`) die Datei schreiben, **nur wenn sich die Bytes unterscheiden**;
     - Dateien in `settings/` und `mod_data/`, deren Schlüssel es im Nutzer nicht mehr gibt, löschen;
     - nicht JSON-fähige Werte überspringen und **nur den Schlüsselnamen** loggen;
     - Rückgabe: Anzahl geschriebener Dateien.
   - `def import_account(user, mirror: Path) -> bool`:
     - jede Datei lesen, `unquote`, `json.loads`, in `user.settings[key]` bzw. `user.mod_data[mod][key]` setzen, nur wenn der Wert sich unterscheidet;
     - Konfliktkopien (`is_conflict_copy`) und unlesbares JSON überspringen;
     - `__value__.json` setzt `user.mod_data[mod]` ganz;
     - Rückgabe: ob etwas geändert wurde.
2. **In `data_api_sync.py` ergänzen:**
   - Modul-Zustand: `_RUN_LOCK = threading.Lock()`, `_LAST = {}`.
   - `async def ensure_sync_key(app, user_id) -> str`:
     - lokalen Nutzer laden (`_load_user`); ist `user_pass_sync` gesetzt → zurückgeben;
     - sonst `res = await app.run_http("CloudM.Auth", "get_sync_key", method="POST")`, mit `Result.result_from_dict(**res)` auspacken, `key` in den lokalen Nutzer schreiben (`_save_user`) und zurückgeben;
     - Fehler → `TransferError("offline")` bzw. `TransferError("reauth_required")` bei 401;
     - den Schlüssel **nie** loggen.
   - `async def sync_all(app) -> dict` (wirft nie):
     1. Lock: `if not _RUN_LOCK.acquire(blocking=False): return {"skipped": True, "reason": "running"}`, sonst `try/finally` mit `release`.
     2. `owner = read_device_owner(app)`; ohne Konto → `{"skipped": True, "reason": "no-account"}`.
     3. `key = await ensure_sync_key(app, owner["user_id"])`; `transport = DataApiTransport(app)`.
     4. `data = Path(app.data_dir)`.
     5. Globaler Ordner:
        - `idx = LocalIndex(str(data / "account_sync" / "global_index.db"))`, `await idx.init()`;
        - `sync_folder(transport, data/"Agents"/"VFS"/"global", PREFIX_GLOBAL, idx, key, device_id)`;
        - `await idx.close()`.
     6. Kontodaten:
        - `user = await _load_user(app, owner["user_id"])`;
        - `mirror = data/"account_sync"/"mirror"`;
        - `export_account(user, mirror)`;
        - `sync_folder(transport, mirror, PREFIX_ACCOUNT, idx2, key, device_id)` mit `idx2` aus `account_index.db`;
        - `if import_account(user, mirror): await _save_user(app, user)`.
     7. `device_id`: `socket.gethostname()` durch `conflict.sanitize_device_id`.
     8. Ergebnis: `{"global": …, "account": …, "finished_at": time.time()}`. Bei `TransferError e` → `{"error": str(e), "finished_at": …}`; bei anderer Exception → `{"error": type(e).__name__, "finished_at": …}`.
     9. `_LAST.clear(); _LAST.update(result)`, Rückgabe.
   - `def run_sync_in_background(app) -> None`: Thread `name="account-sync-run"`, `daemon=True`, darin `asyncio.run(sync_all(app))`.
3. **Auslöser:**
   - `device_account.link_account`, ganz am Ende vor `return user`:
     ```python
     try:
         from toolboxv2.mods.CloudM.LiveSync.data_api_sync import run_sync_in_background
         run_sync_in_background(app)
     except Exception:
         pass
     ```
   - `wizard_api.py`, neue Route im Konto-Block (mit `_ensure_local` wie in Teil 4):
     `@app.post("/api/account/sync", auth=False)`:
     - ohne Gerätekonto → `409, {"error": "no-account"}`;
     - sonst mit `get_app("account.sync")` → `run_sync_in_background(app)` → `202, {"started": True}`.
   - `server_worker.py`, an der markierten Stelle aus Teil 6 (`# Teil 10 haengt hier den Konto-Sync an.`):
     - genau einmal pro Prozess einen Daemon-Thread `name="account-sync"` starten (vorher `threading.enumerate()` prüfen);
     - er wartet 30 s, ruft dann `asyncio.run(sync_all(self._app))` auf und wiederholt das alle 600 s;
     - jede Exception wird gefangen und nur mit Typ geloggt.
   - `first_run.account_state()`, Zweig mit Gerätekonto: `"last_sync": _LAST.get("finished_at")` und `"last_sync_error": _LAST.get("error")` ergänzen (Import aus `data_api_sync` in der Funktion, Fehler → beide `None`).

**Tests:**

`test_account_bridge.py`:

| Fall | Erwartet |
|---|---|
| export → import in einen leeren Nutzer | `settings` und `mod_data` gleich, ohne `anonymous`/`local_admin` |
| ein Wert in `mod_data["m"]["a"]` geändert, erneut exportiert | nur die Datei von `m/a` hat neue Bytes (Prüfsummen der anderen gleich) |
| Schlüssel entfernt | Datei weg |
| Schlüssel mit `/` und Umlaut | Rundweg verlustfrei |
| nicht JSON-fähiger Wert | übersprungen, Log enthält Schlüsselnamen, **nicht** den Wert |

`test_data_api_sync.py` (erweitern, mit `FakeTransport` aus Teil 9, Nutzer-DB als Dict gepatcht, `read_device_owner` gepatcht, `ensure_sync_key` gegen einen Fake-`run_http`):

| Fall | Erwartet |
|---|---|
| `sync_all` ohne Gerätekonto | `{"skipped": True, "reason": "no-account"}`, kein Transport-Aufruf |
| Gerät A setzt `settings["theme"]="dark"`, A sync, B sync | B-Nutzer hat `theme == "dark"` |
| B ändert nur `mod_data["m"]["a"]`, B sync, A sync | A hat das neue `a`, `m["b"]` unverändert, genau 1 `put` bei B |
| A und B ändern `mod_data["m"]["a"]` verschieden, A sync, B sync | B-Nutzer hat As Wert; im B-Spiegelordner liegt eine Konfliktkopie mit Bs Wert |
| zweiter `sync_all` während einer läuft (Lock vorher halten) | `{"skipped": True, "reason": "running"}` |
| zwei `sync_all`-Läufe | `get_sync_key` nur beim ersten aufgerufen |
| Fake-`run_http` für `get_sync_key` liefert 401 | Ergebnis `error == "reauth_required"`, nichts geschrieben |
| Logs aller Fälle (`assertLogs`) | enthalten weder den Schlüssel noch einen Wert aus `settings`/`mod_data` |

`test_account_link_flow.py`: `POST /api/account/sync` ohne Gerätekonto → 409; mit Gerätekonto → 202 (`run_sync_in_background` gepatcht, genau ein Aufruf).

**Abnahme:** `python -m pytest toolboxv2/tests/test_core/test_account_bridge.py toolboxv2/tests/test_core/test_data_api_sync.py toolboxv2/tests/test_core/test_account_link_flow.py -q -p no:cacheprovider` → `0 failed`, dazu das Gate (`test_account_bridge.py` ab jetzt in `$GATE`).

**Commit:** `08-10 (S1/S3): Kontodaten Schluessel fuer Schluessel, sync_all, Ausloeser`

---

### Teil 11 – Abschluss

1. Volles Gate ein letztes Mal, Ausgabe als `..\isaa08_gate_final.txt`.
2. Diese Prüfung muss **0 Treffer** liefern (keine Geheimnisse, keine alten Wege):
   ```powershell
   git grep -n "get_secure_storage_credentials\|tb_get_my_minio_credentials\|mods\.CloudM\.sync\|CloudM\.sync\.\|run_sync_once" -- '*.py' ':!toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_credentials_api.py'
   git grep -n "print(.*user_pass_sync\|log.*user_pass_sync" -- '*.py'
   ```
3. `git log origin/master --oneline -12` zeigt die Commits `08-2` … `08-10`.
4. Bericht (0.7) schreiben. Diese Punkte **ausdrücklich** als „nur live prüfbar, nach D1“ eintragen:
   - K0: Auf `simplecore.app` nimmt `POST /api/CloudM.UserDataAPI/list_data|get_data|set_data` die Parameter aus dem JSON-Body an. Dasselbe für `POST /api/CloudM.Auth/get_sync_key`. Statuscodes und Antwortformen ohne Token und ohne Schlüssel festhalten.
   - K4: Tabelle aus dem WP1-Dokument, dazu „Netz trennen, Workers neu → angemeldet“.
   - S4: Tabelle aus dem WP1-Dokument, mit `POST /api/account/sync`.
   - T10.
5. Arbeitskopie bleibt; der Owner entfernt sie mit `git worktree remove C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa08`.

---

## 3. Bekannt, aber nicht Teil dieses Auftrags (nur im Bericht bestätigen, nichts ändern)

1. `CloudM.Auth.get_user_data` nimmt `user_id` aus dem Aufruf, ohne zu prüfen, ob es der angemeldete Nutzer ist. Nach Teil 8 enthält die Antwort keine Geheimnisse mehr, aber weiter `settings`/`mod_data` fremder Nutzer, falls der Worker die Route nicht absichert. **Owner-Entscheidung nötig.**
2. LiveSync-Shares (icli): `tb_generate_share_token` hat keine Anmeldeprüfung, und der Token **mit AES-Schlüssel** geht bei `ws://` unverschlüsselt über die Leitung (`LiveSync/__init__.py:520–560`, `client.py:430`). Betrifft nicht den Konto-Sync, verstößt aber gegen Entscheidung 4. **Eigener Auftrag.**
3. Pfadnamen (z. B. `livesync/global/notes/plan.md`) sind für den Server sichtbar; Inhalte nicht. Der Kanal ist HTTPS.
4. `AdminDashboard.py` verwaltet weiter MinIO-Zugangsdaten pro Nutzer (nur Admin, serverseitig). Nicht angefasst.
