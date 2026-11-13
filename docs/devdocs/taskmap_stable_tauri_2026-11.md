# Task-Map — Stabile Tauri-Anwendung (Stand 2026-11-10)

> Basis: `master` @ `d1db5c9c`. Vorgänger: `handoff_onboarding_2026-11.md`.
> Zielgruppe: ein Coding-Agent, der **lokal auf dem Windows-Rechner des Owners** arbeitet.
> Diese Datei ist die einzige Arbeitsanweisung. Reihenfolge einhalten, nichts überspringen.

---

## 0. Ziel und Abnahme

Am Ende läuft auf einem Windows-Rechner **ohne vorher installiertes ToolBoxV2**:

| # | Abnahmepunkt | Prüfung |
|---|---|---|
| Z1 | Installation | SimpleCore-Setup installieren, App starten, Profil `local` wählen → Installation endet ohne Fehler |
| Z2 | Start | Danach zeigt das Hauptfenster das Dashboard von `http://127.0.0.1:8467/` (kein Text „dist/index.html not found", kein Timeout) |
| Z3 | Anmeldung | Erster Start verlangt die Anmeldung mit dem SimpleCore-Konto. Danach ist man auf diesem Gerät automatisch angemeldet, auch ohne Netz |
| Z4 | ISAA | Dashboard → Tab „Lokal" → ISAA öffnet `/isaa`, eine Chat-Nachricht wird beantwortet |
| Z5 | OBS | Dashboard → Tab „Lokal" → OBS öffnet `/obs` und zeigt Live-Daten des ISAA-Laufs |
| Z6 | Neustart | App schließen und neu starten → direkt Dashboard, kein Onboarding |
| Z7 | Update | Neuere App-Version wird gefunden und installiert; danach läuft der neue Kern, Profil, Konto und Daten sind erhalten |
| Z8 | Daten auf allen Geräten | Zweites Gerät, gleiches Konto: bestehende Kontodaten (Data-API), Blobs und der globale ISAA-Ordner sind nach dem Login da; Änderungen wandern in beide Richtungen |

Nicht Teil des Ziels: Sync von ISAA-Chatverläufen (heute keinem Nutzer zugeordnet), Linux/macOS-Abnahme, Punkt E (ein HTTP-Stack), README, Dead-Code, Installer-Konsolidierung, B2, B3, B9.

---

## 1. Regeln (immer gültig)

1. Arbeite nur auf dem Branch `claude/wizardly-cannon-o944j4`. Nie auf `master` pushen.
2. Ein Task = ein Commit. Commit erst, wenn das **Gate** des Tasks grün ist.
3. Ist ein Gate rot: Abschnitt „Wenn rot" des Tasks ausführen. Ist es danach noch rot: **stoppen**, Befehl + Ausgabe an den Owner melden. Nicht raten, nicht den nächsten Task beginnen.
4. Nie einen Test löschen, überspringen oder abschwächen, um ein Gate grün zu bekommen.
5. Kopien synchron halten (Tests erzwingen Byte-Gleichheit):
   - `installer.sh` ≡ `toolboxv2/simple-core/src-tauri/installer.sh`
   - `installer.ps1` ≡ `toolboxv2/simple-core/src-tauri/installer.ps1`
   - `toolboxv2/simple-core/src-tauri/installer.html` ≡ `toolboxv2/web/installer/index.html`
   Immer die Root-Datei bearbeiten, dann in die Kopie kopieren.
6. Keine Ports hart im Code. Quelle bleibt Config/Manifest (8467 HTTP, 8468 WS).
7. Keine Geheimnisse (Signaturschlüssel, API-Keys) in Dateien oder Commits.
8. Zwei verschiedene `dist`-Ordner nicht verwechseln: `dist/` im Repo-Root = Python-Wheel, `toolboxv2/dist/` = Web-Frontend.
9. Tasks mit Kennung **H** macht der Owner. Dort stoppen und warten.

**Aufgaben des Owners** (der Agent stoppt an der genannten Stelle):

| # | Was | Wann |
|---|---|---|
| H0 | LLM-API-Key eintragen | vor T10 |
| H4 | MinIO auf `simplecore.app` konfigurieren und von außen per HTTPS erreichbar machen | nur wenn K0-P3 rot ist |
| H5 | Server-Änderung aus S1 auf `simplecore.app` ausrollen | nach S1 |
| H1 | Signaturschlüssel als GitHub-Secrets | vor T12 |
| H2, H3 | `nightly.yml` anstoßen (zweimal) | in T14 |

**Entscheidungen des Owners** (gelten, nicht neu verhandeln): Wheel im App-Bundle · Updater über GitHub (`nightly`) · Konto beim ersten Start Pflicht, danach automatische Anmeldung auf dem Gerät · vorhandene lokale Daten gehen ins Konto · Schlüssel liegt beim Konto auf dem Server · Sync-Umfang: Data-API-Daten, Blobs, globaler ISAA-Ordner.

**Standard-Gate** (in jedem Task zusätzlich zum eigenen Gate):

```powershell
python -m unittest toolboxv2.tests.test_core.test_first_run_onboarding toolboxv2.tests.test_core.test_user_apps toolboxv2.tests.test_core.test_cors_policy toolboxv2.tests.test_core.test_live_ui_owner toolboxv2.tests.test_core.test_wizard_api toolboxv2.tests.test_core.test_installer_protocol
```

Erwartet: `OK`. Danach muss `git status --short` nur die Dateien des Tasks zeigen.

---

## 2. Gemessene Fakten (nicht neu herleiten)

Gemessen am 2026-11-10 unter Linux mit einem lokal gebauten Wheel. Windows ist **nicht** gemessen.

| # | Fakt | Folge |
|---|---|---|
| F1 | PyPI `toolboxv2 0.1.28` (13.07.) enthält weder `wizard_api.py` noch `user_apps.py`. `master` trägt dieselbe Nummer. | Installation von PyPI ergibt alten Kern → T5–T7 |
| F2 | Installer-Default `FEATURES="core cli"` → `ToolBoxV2[cli]`. `waitress` steckt nur im Extra `web`. Ohne `waitress` stirbt der Owner-Prozess (`ModuleNotFoundError`, `fast_tb.py:1029`). | T4, T5 |
| F3 | `installer.ps1` liest Features nie aus der Config. `installer.sh` liest sie nur aus der Zeile **nach** `features:` und löscht dabei Leerzeichen. | T5 |
| F4 | `tb build build` baut Wheel + Feature-ZIPs fehlerfrei (4 MB). Das Web-ZIP enthält `dist` nur, wenn vorher `npm run build` lief. Ohne `dist`: `/` liefert einen Text, `/setup` liefert 404. | T6 |
| F5 | `uv tool install "toolboxv2[web,isaa,cli] @ file:///<abs>/toolboxv2-0.1.28-py3-none-any.whl" --force` funktioniert (68 s, 580 MB). Danach `tb -init local` → `{"ok": true}`, Owner antwortet nach 4 s. | Gewählter Lieferweg trägt |
| F6 | Manifest, `.data` und `.config` liegen in `site-packages/toolboxv2/`. Nach Reinstall ist `.data` **weg**. | Jedes Update löscht Nutzerdaten → T3 |
| F7 | Das Repo-Manifest `toolboxv2/tb-manifest.yaml` ist eingecheckt mit `profile: developer` und wird ins Wheel gepackt. | Wheel-Installationen sehen nie „first run" → T3 |
| F8 | `SessionManager.enable_local_trust` wird in `server_worker.py:1697` aufgerufen, ist aber nirgends definiert. Log: `Loopback trust not enabled`. | Keine Auto-Anmeldung → K2 |
| F9 | `POST /tray/command` mit `{"command":"start_app","args":{"name":"isaa"}}` → `{"ok":true,"result":{"merged":true,"prefix":"/isaa","ws":true}}`, `/isaa` → 200. | ISAA-Mount funktioniert |
| F10 | `litellm` steht in `features/isaa/feature.yaml`, fehlt aber im Extra `isaa` in `pyproject.toml`. | ISAA-Chat ohne LLM-Bibliothek → T4 |
| F11 | OBS (`mods/isaa/extras/live_obs_server.py`, `app = FastTB(...)` Zeile 41) ist nicht in der App-Registry. | T11 |
| F12 | `tauri.conf.json` hat `plugins.updater`, aber `tauri-plugin-updater` fehlt in `Cargo.toml` und `lib.rs`. App-Version ist fest `0.1.0`. | Kein Update möglich → T12 |
| F13 | `bundle.externalBin` verlangt `binaries/tb-worker`; laut Handoff B10 funktioniert dieses Binary nicht. `worker_manager.rs` erlaubt Endpoint `local` nur, wenn der Sidecar da ist. | T8 |
| F14 | `launch.rs` wartet nach dem Start nur 25 s auf 8467. | Erster Start unter Windows kann länger dauern → T8 |
| F15 | Tests und `import toolboxv2` schreiben Marker in den Arbeitsbaum (`features/*/.installed`, `logs/`, `.info/`). | T2 |
| F16 | Lokale Anmeldung und SimpleCore-Konto schließen sich heute aus: mit `TOOLBOXV2_REMOTE_BASE` kein lokaler Admin (`CloudM/module.py:95`), ohne sie nur der Admin `root` mit Zufalls-ID. | K1–K3 |
| F17 | SimpleCore-Login vom Gerät existiert für die CLI: `utils/clis/cli_oauth.py:65` öffnet die Login-Seite im Browser und erhält `token`, `refresh_token`, `user_id`, `username` über `127.0.0.1:8765`. | Wird in K3 wiederverwendet |
| F18 | Ohne Nutzerschlüssel verschlüsseln `blobs.py:189–212` und `scoped_storage.py` mit dem **Geräteschlüssel**. | Anderes Gerät kann die Daten nicht lesen → S1, S2 |
| F19 | Der Server gibt pro Konto MinIO-Zugangsdaten aus: `CloudM/UserDataAPI.py:1097`. Braucht `MINIO_ENDPOINT` auf dem Server — **nicht geprüft**. | K0-P3 |
| F20 | `BlobStorage.sync()` (`blobs.py:1021`) gleicht lokal ↔ Cloud unter `{user_id}/` ab, ist aber nirgends an ein Konto angeschlossen. `toolbox.py:290` baut den Speicher im Server-Modus gegen `127.0.0.1:9000`. | S2, S4 |
| F21 | Bestehende Serverdaten der Data-API sind mit dem Schlüssel des Servers verschlüsselt (Server-`UserContext` ohne `encryption_key`). | Nur über die API lesbar → S3 |
| F22 | Der globale ISAA-Ordner ist `<data_dir>/Agents/VFS/global` (`power_vfs.py:110`), ein normaler Ordner. | S4 |
| F23 | `utils/clis/cli_oauth.py` enthält ab Zeile 105 einen losen Textblock („WORKING AGREEMENT") als String im Modul; `local_cli.py:258` gibt die Login-Antwort samt Tokens per `print` aus. | Nicht anfassen; dem Owner gemeldet |

---

## 3. Reihenfolge

```
T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 → K0 → K1 → K2 → K3 → K4 → T10 → T11 → S1 → H5 → S2 → S3 → S4 → S5 → H1 → T12 → T13 → T14
```

Begründung, rückwärts vom Ziel: Update (T12/T13) braucht eine installierbare App (T6–T8). Die App braucht einen Installer, der das mitgelieferte Wheel nimmt (T5), und einen Kern, der mit diesem Wheel startet (T4) und dessen Zustand ein Update überlebt (T3). Konto (K0–K4), ISAA und OBS (T10, T11) sind reine Kern-Themen und ohne Tauri prüfbar. Der Sync (S1–S5) braucht das Konto und steht deshalb danach; er braucht außerdem eine Server-Änderung (H5).

---

## 4. Tasks

### T1 — Arbeitsumgebung und Ausgangslage

**Schritte**

```powershell
git fetch origin claude/wizardly-cannon-o944j4
git checkout claude/wizardly-cannon-o944j4
uv venv .venv
.venv\Scripts\activate
uv pip install -e ".[web,cli,core,isaa]" litellm pytest pytest-asyncio build
```

Prüfe außerdem, dass vorhanden sind: `node --version` (≥ 20), `cargo --version`, `git --version`.

**Gate:** Standard-Gate → `OK`. `python -m pytest -q toolboxv2/tests/test_core/test_wizard_e2e_http.py` → alle bestanden.

**Wenn rot:** Ausgabe melden und stoppen. Nichts reparieren, was nicht in dieser Map steht.

---

### T2 — Arbeitsbaum sauber halten (F15)

**Datei:** `.gitignore`

**Schritte:** Diese Zeilen anhängen:

```
toolboxv2/features/*/.installed
toolboxv2/features_packed/
toolboxv2/.info/
toolboxv2/minio_cluster_config.json
toolboxv2/simple-core/src-tauri/wheels/*.whl
/logs/
```

**Gate:** Standard-Gate laufen lassen, dann `git status --short` → zeigt nur `.gitignore`.

**Nicht tun:** `toolboxv2/tb-manifest.yaml` nicht in `.gitignore` aufnehmen. Wenn ein Test sie verändert hat: `git checkout toolboxv2/tb-manifest.yaml`.

---

### T3 — Nutzerzustand aus dem Paketordner holen (F6, F7)

**Ziel:** Bei einer Wheel-Installation liegen Manifest und Daten im Installationsordner des Nutzers, nicht in `site-packages`. Im Source-Checkout bleibt alles wie bisher.

**Schritt 1 — neue Datei `toolboxv2/_state_dir.py`** (nur Standardbibliothek, keine `toolboxv2`-Importe):

- `is_packaged_install() -> bool`: `True`, wenn `"site-packages"` oder `"dist-packages"` in den Pfadteilen von `Path(__file__).resolve()` vorkommt.
- `user_state_dir() -> Path`: 1. Umgebungsvariable `TB_INSTALL_DIR`, wenn gesetzt und nicht leer. 2. sonst derselbe Standardpfad wie im Installer. Lies ihn ab aus `installer.sh` Zeilen 160–166 und `installer.ps1` Zeile 132 (Windows: `%LOCALAPPDATA%\toolboxv2`, Linux: `~/.local/share/toolboxv2`) und bilde ihn exakt nach.
- `manifest_base_dir() -> Path`: 1. Umgebungsvariable `TB_MANIFEST_DIR`, wenn gesetzt. 2. sonst `user_state_dir()`, wenn `is_packaged_install()`. 3. sonst `Path(__file__).parent`.

**Schritt 2 — `toolboxv2/__init__.py`:** Direkt nach Zeile 69 (`__tb_root_dir__ = tb_root_dir = ...`) einfügen: Wenn `is_packaged_install()` und `TB_DATA_DIR` nicht gesetzt ist, dann `os.environ["TB_DATA_DIR"] = str(user_state_dir())` und den Ordner anlegen. (`utils/toolbox.py:78` hängt selbst `.data` an.)

**Schritt 3 — alle Manifest-Aufrufe umstellen.** Finde sie mit:

```powershell
git grep -n "ManifestLoader(" -- "toolboxv2/*.py" ":!toolboxv2/tests"
```

Bekannte Stellen: `__init__.py:230`, `feature_loader_registry.py:72` und `:87`, `init_onboarding.py:198` und `:381`, `utils/clis/first_run.py:41` und `:57`, `utils/workers/fast/endpoint.py:155`, `flows/mini/toolbox_admin.py:364/387/406`. Überall, wo `tb_root_dir` (auch als `str(...)` oder `base_dir=str(root)`) übergeben wird, stattdessen `manifest_base_dir()` übergeben. Stellen, die einen **explizit übergebenen** Pfad nutzen (`path`-Parameter), unverändert lassen. In `endpoint.py` bleibt die Auflösung relativer `dist_dir`-Werte gegen `tb_root_dir` unverändert.

**Schritt 4 — Installer schreibt kein Manifest mehr.** In `installer.sh` den Block „tb-manifest.yaml (only write if not already present)" (ab ca. Zeile 884 bis zum zugehörigen `fi`) entfernen. Das Gegenstück in `installer.ps1` suchen (`git grep -n "tb-manifest.yaml" installer.ps1`) und ebenfalls entfernen. Grund: `tb -init <profile>` legt das Manifest an; ein vom Installer geschriebenes Manifest würde jetzt das echte werden. Beide Dateien in die `src-tauri`-Kopien kopieren.

**Schritt 5 — neuer Test `toolboxv2/tests/test_core/test_state_dir.py`:**

- Mit `TB_MANIFEST_DIR=<tmp>`: `onboarding_state()["first_run"]` ist `True`.
- Danach `apply_profile("local")`: `<tmp>/tb-manifest.yaml` existiert, `onboarding_state()` liefert `first_run False`, `profile "local"`.
- Die Datei `toolboxv2/tb-manifest.yaml` im Repo ist vor und nach dem Test byte-gleich.
- Ohne `TB_MANIFEST_DIR` im Source-Checkout: `manifest_base_dir()` ist gleich `tb_root_dir`.

**Gate:** Standard-Gate + `python -m unittest toolboxv2.tests.test_core.test_state_dir` → `OK`.

**Wenn rot:** Schlägt ein bestehender Test fehl, weil er das Manifest im Paketordner erwartet, dann setze in dessen `setUp` `TB_MANIFEST_DIR` auf ein Temp-Verzeichnis — die Prüfungen selbst nicht ändern.

---

### T4 — Abhängigkeiten vollständig machen (F2, F10)

**Datei:** `pyproject.toml`

**Schritte**

1. In `[project] dependencies` ergänzen: `"waitress>=3.0.2"` und `"websockets>=15.0.1"` (der Owner auf 8467 und der WS-Worker gehören zu jedem Profil).
2. Im Extra `isaa` ergänzen: `"litellm"`.

**Gate:** Standard-Gate. Zusätzlich in einer frischen Umgebung:

```powershell
uv venv .venv-min
uv pip install --python .venv-min -e .
.venv-min\Scripts\python -c "import waitress, websockets; print('ok')"
```

Erwartet: `ok`. Danach `.venv-min` löschen.

---

### T5 — Installer nimmt mitgeliefertes Wheel und die richtigen Features (F1–F3)

**Dateien:** `installer.sh`, `installer.ps1` (danach in `src-tauri` kopieren)

**Schritt 1 — Features aus der Config.** Neuer einzeiliger Config-Schlüssel `tb_features`, Wert kommagetrennt, z. B. `tb_features: cli,web,isaa`.
- `installer.sh`: im Config-Block (bei Zeile 340–350) lesen mit `yaml_get "$CONFIG_FILE" "tb_features" ""`; wenn nicht leer: Kommas durch Leerzeichen ersetzen und `FEATURES` setzen. Die alte `features:`-Logik bleibt.
- `installer.ps1`: an der Stelle, an der `profile`/`ui` aus der Config gelesen werden, dasselbe mit `Get-YamlField` tun und `$script:FEATURES` setzen.

**Schritt 2 — Wheel-Quelle.** Neue Umgebungsvariable `TB_WHEEL` (absoluter Pfad zu einer `.whl`-Datei).
- `installer.sh`, Funktion `_install_uv` (Zeile 706): Wenn `TB_WHEEL` gesetzt ist und die Datei existiert, dann lautet das Paket `toolboxv2[<extras>] @ file://<TB_WHEEL>` (ohne Extras: `toolboxv2 @ file://<TB_WHEEL>`). Sonst wie bisher.
- `installer.ps1`, Funktion `Install-Uv` (Zeile 584): gleiches Verhalten. Die URL bilden mit `([System.Uri](Resolve-Path $env:TB_WHEEL).Path).AbsoluteUri` (ergibt `file:///C:/...`).
- Gleiches für `_install_venv` / `Install-Venv`.

**Schritt 3 — Update mit Wheel.** In `action_update` (sh Zeile 957) und `Action-Update` (ps1 Zeile 765), Zweig `uv`: Wenn `TB_WHEEL` gesetzt ist, `FEATURES` aus `install.manifest` lesen (Schlüssel `features`), `_install_uv` / `Install-Uv` aufrufen und mit der zurückgegebenen Version `phase_write_manifests` (bzw. das ps1-Gegenstück) ausführen. Ohne `TB_WHEEL` bleibt `uv tool upgrade`.

**Schritt 4 — Tests.** In `toolboxv2/tests/test_core/test_installer_protocol.py` zwei Dry-Run-Fälle ergänzen (Muster der vorhandenen Fälle übernehmen): Config mit `tb_features: cli,web,isaa` und gesetztem `TB_WHEEL` auf eine leere Temp-Datei `toolboxv2-9.9.9-py3-none-any.whl` → die Ausgabe nennt das Paket mit `[cli,web,isaa]` und `file://`.

**Gate**

```powershell
python -m unittest toolboxv2.tests.test_core.test_installer_protocol
powershell -ExecutionPolicy Bypass -File installer.ps1 -JsonOutput -Config <tmp>\tb-install.yaml -DryRun
```

Erwartet: Test `OK`; der Dry-Run gibt nur JSON-Zeilen auf stdout aus und endet mit einer Zeile `"type":"done"`. (`<tmp>\tb-install.yaml` mit `install_mode: uv`, `profile: local`, `ui: none`, `assume: yes`, `action: install`, `tb_features: cli,web,isaa`.)

**Wenn rot:** Lehnt `uv` die `file:///`-URL unter Windows ab, dann stattdessen `uv tool install --from "<TB_WHEEL>" "toolboxv2[<extras>]"` verwenden und dieselbe Form in `installer.sh` übernehmen. Befund im Commit-Text nennen.

---

### T6 — Wheel bauen und in die Tauri-App legen (F4)

**Schritt 1 — Bauskript `scripts/build_wheel_for_tauri.sh`** (bash; unter Windows mit Git Bash ausführen), Inhalt in dieser Reihenfolge:

1. `cd toolboxv2 && npm install && npm install --prefix ./web && npm run build && cd ..`
2. Abbruch mit Fehler, wenn `toolboxv2/dist/index.html` fehlt oder kleiner als 500 Bytes ist (Schutz vor dem CI-Stub).
3. `rm -rf dist toolboxv2/features_packed`
4. `tb build build`
5. Abbruch mit Fehler, wenn das Web-ZIP `toolboxv2/features_packed/tbv2-feature-web-*.zip` keinen Eintrag `dist/index.html` enthält (prüfen mit `python -m zipfile -l`).
6. `mkdir -p toolboxv2/simple-core/src-tauri/wheels`, alte `*.whl` dort löschen, `cp dist/toolboxv2-*.whl` dorthin.

**Schritt 2 — Bundle.** Datei `toolboxv2/simple-core/src-tauri/wheels/.gitkeep` anlegen. In `tauri.conf.json` unter `bundle.resources` ergänzen: `"wheels/*"`.

**Schritt 3 — `installer.rs`.**
- Neue reine Funktion `bundled_wheel(resource_dir: &Path) -> Option<PathBuf>`: sucht in `<resource_dir>/wheels` Dateien, die mit `toolboxv2-` beginnen und auf `.whl` enden; gibt die mit dem alphabetisch größten Namen zurück.
- In `run_install` direkt nach dem `TB_UV_BIN`-Block (Zeile 201–204): wenn `bundled_wheel(...)` etwas findet, `cmd.env("TB_WHEEL", pfad)`.
- In `config_to_yaml`: für alle Profile außer `developer` die Zeile `tb_features: cli,web,isaa` schreiben.
- Tests im selben Modul ergänzen: `bundled_wheel_found_in_resources` (Temp-Ordner mit zwei Wheels → das größere), und in `yaml_contains_canon_keys` die neue Zeile prüfen.

**Schritt 4 — CI.** In `.github/workflows/nightly.yml`, `release.yml` (Job „Tauri") und `build.yml`: nach dem Schritt, der `npm run build` ausführt, und vor dem Tauri-Build einen Schritt einfügen:

```yaml
      - name: Build core wheel for bundle
        shell: bash
        run: |
          pip install build
          bash scripts/build_wheel_for_tauri.sh
```

**Gate**

```powershell
bash scripts/build_wheel_for_tauri.sh
dir toolboxv2\simple-core\src-tauri\wheels\*.whl
cd toolboxv2\simple-core\src-tauri; cargo test --lib installer::; cd ..\..\..
```

Erwartet: genau eine `.whl`-Datei; alle `installer::`-Tests bestanden. Plus Standard-Gate.

**Zusatz-Gate (echte Installation aus dem Wheel, ohne Tauri):**

```powershell
$env:TB_WHEEL = (Resolve-Path toolboxv2\simple-core\src-tauri\wheels\*.whl).Path
powershell -ExecutionPolicy Bypass -File installer.ps1 -JsonOutput -Config <tmp>\tb-install.yaml
& "$env:LOCALAPPDATA\toolboxv2\bin\tb.exe" workers start
```

In einem zweiten Fenster:

| Aufruf | Erwartet |
|---|---|
| `curl http://127.0.0.1:8467/health` | `"status":"healthy"` |
| `curl http://127.0.0.1:8467/api/wizard/state` | `"first_run":false`, `"profile":"local"` |
| `curl -s -o NUL -w "%{http_code} %{size_download}" http://127.0.0.1:8467/` | `200` und mehr als 1000 Bytes |
| `curl -s -o NUL -w "%{http_code}" http://127.0.0.1:8467/setup` | `200` |
| `dir "$env:LOCALAPPDATA\toolboxv2\tb-manifest.yaml"` | Datei existiert |
| `dir "$env:LOCALAPPDATA\toolboxv2\.data"` | Ordner existiert |

Workers danach stoppen. Die Installation für K4, T10, T11 und S5 stehen lassen.

**Wenn rot:** `/` liefert Text statt HTML → das Wheel wurde ohne `dist` gebaut, Schritt 1 des Skripts prüfen. `first_run:true` trotz Installer-Lauf → T3 Schritt 3 hat eine Aufrufstelle verfehlt; `git grep` wiederholen.

---

### T7 — Eine Versionsnummer für Kern und App (F1, F12)

**Ziel:** `0.1.29` überall, damit sich der Stand von PyPI `0.1.28` unterscheidet und App-Version = Kern-Version gilt.

**Schritte:** `git grep -n "0\.1\.28"` ausführen und in diesen Dateien auf `0.1.29` ändern: `pyproject.toml`, alle `toolboxv2/features/*/feature.yaml`. Zusätzlich `0.1.0` → `0.1.29` in `toolboxv2/simple-core/src-tauri/tauri.conf.json` (`version`), `toolboxv2/simple-core/src-tauri/Cargo.toml` (`[package] version`) und `toolboxv2/simple-core/package.json`. Treffer in Tests, Docs, Lock-Dateien und `tb-registry/` **nicht** ändern.

**Gate:** Standard-Gate. `bash scripts/build_wheel_for_tauri.sh` → Wheel heißt `toolboxv2-0.1.29-py3-none-any.whl`. `cargo test --lib` in `src-tauri` → bestanden.

---

### T8 — Tauri startet ohne `tb-worker` und wartet lange genug (F13, F14)

**Schritte**

1. `tauri.conf.json`: aus `bundle.externalBin` den Eintrag `"binaries/tb-worker"` entfernen (`"binaries/uv"` bleibt).
2. `capabilities/`: den `shell:allow-execute`-Eintrag für `binaries/tb-worker` entfernen.
3. `worker_manager.rs`, Funktion `set_endpoint`: `"local"` immer erlauben (die Bedingung `if self.sidecar_available` mit Rückfall auf `Remote` entfernen).
4. `lib.rs:918`: `wait_until_running(&base, 25)` → `wait_until_running(&base, 120)`.
5. CI (`nightly.yml`, `release.yml`, `build.yml`): im Schritt „Create Worker Sidecar Binary" alle PyInstaller-Zeilen und das Erzeugen von `worker_runner.py` entfernen. **Behalten:** das Anlegen von `binaries/` und den `toolboxv2/dist/index.html`-Stub. Den Schritt umbenennen in „Prepare build dirs". `pip install pyinstaller` entfernen.

**Gate**

```powershell
cd toolboxv2\simple-core\src-tauri; cargo test --lib; cd ..
bash ../../scripts/fetch_uv_sidecar.sh x86_64-pc-windows-msvc
npx tauri build --config '{\"bundle\":{\"createUpdaterArtifacts\":false}}'
```

Erwartet: Tests bestanden; Build endet mit einem Setup unter `src-tauri\target\release\bundle\nsis\`.

**Wenn rot:** Meldet der Build ein fehlendes `binaries/uv-...`, das Sidecar-Skript mit dem Triple aus der Fehlermeldung erneut ausführen.

---

### K0 — Messung gegen `simplecore.app` (kein Code)

**Ziel:** Vier Annahmen prüfen, auf denen K und S aufbauen. Ergebnisse in `docs/devdocs/probe_simplecore_2026-11.md` festhalten — **ohne** Tokens, Schlüssel oder Zugangsdaten (nur Statuscode, Feldnamen, Fehlermeldung).

**Schritte**

1. Token holen (öffnet den Browser, Owner meldet sich an):

```powershell
$env:TOOLBOXV2_REMOTE_BASE = "https://simplecore.app"
python -c "import asyncio; from toolboxv2.utils.clis.cli_oauth import cli_remote_login; p = asyncio.run(cli_remote_login()); print(sorted(p.keys()) if p else None); open('probe_token.tmp','w').write(p['access_token'] if p else '')"
$t = Get-Content probe_token.tmp
```

2. Vier Aufrufe, jeweils Statuscode und Feldnamen der Antwort notieren:

| # | Aufruf | Frage |
|---|---|---|
| P1 | `curl -H "Authorization: Bearer $t" https://simplecore.app/api_user_data` | Welche Felder hat der Nutzerdatensatz? |
| P2 | `curl -H "Authorization: Bearer $t" https://simplecore.app/api/CloudM.UserDataAPI/list_data` | Stimmt die URL-Form `/api/<Modul>/<Funktion>`? Wie sieht die Liste aus? |
| P3 | `curl -H "Authorization: Bearer $t" https://simplecore.app/api/CloudM.UserDataAPI/get_secure_storage_credentials` | Gibt der Server MinIO-Zugangsdaten aus? Welche Felder (`endpoint`, `secure`, `buckets`, …)? |
| P4 | `curl -H "Authorization: Bearer $t" "https://simplecore.app/api/CloudM.UserDataAPI/get_data?path=<ein Pfad aus P2>"` | In welcher Form kommen die Daten zurück? |

3. `probe_token.tmp` löschen. Prüfen, dass die Datei nicht in `git status` auftaucht.

**Gate / Entscheidung**

| Ergebnis | Weiter |
|---|---|
| P1–P4 liefern 200 | K1 |
| P2 liefert 404 | Stoppen. Die URL-Form ist eine ANNAHME; richtige Form beim Owner erfragen |
| P3 meldet „MINIO_ENDPOINT not configured" oder einen anderen Fehler | K-Block normal fortsetzen; **S2 und S4 sind blockiert**, bis der Owner MinIO auf dem Server konfiguriert (H4). Melden |
| P3 liefert `secure: false` oder einen `endpoint`, der nicht aus dem Internet erreichbar ist | Wie Zeile darüber |

---

### K1 — Gerätekonto: verknüpfen, merken, Altdaten übernehmen (F8, F16, F17)

**Neue Datei:** `toolboxv2/mods/CloudM/auth/device_account.py`

**Festlegungen**

- „Gerätekonto" = das SimpleCore-Konto, das auf diesem Gerät angemeldet wurde.
- Merker ist die Datei `<app.data_dir>/device_owner.json` mit genau diesen Feldern: `user_id`, `username`, `base_url`, `linked_at`, `migrated_from` (leer oder alte lokale ID). Keine Tokens, keine Schlüssel in dieser Datei.
- Auf dem eigenen Gerät hat das Gerätekonto lokal `level = -1` (wie bisher der lokale Admin `root`). Dieses lokale Level wird nie an den Server gesendet.

**Funktionen**

1. `device_owner_file(app=None) -> Path`: `Path(app.data_dir) / "device_owner.json"`; ohne `app` → `get_app()`.
2. `read_device_owner(app=None) -> Optional[dict]`: Datei lesen; bei fehlender oder ungültiger Datei `None`.
3. `async link_account(app, payload: dict) -> UserData`:
   1. `payload` muss `access_token` und `user_id` enthalten, sonst `ValueError`.
   2. Tokens speichern **genau wie** `utils/clis/local_cli.py:259–267` (`s.username = ...`, `s._save_session_token(...)`, `await s.login()`); vorher `app.session.base` auf `TOOLBOXV2_REMOTE_BASE` setzen, Default `https://simplecore.app`. Schlägt `login()` fehl → `RuntimeError`, nichts weiter schreiben.
   3. Lokalen Nutzerdatensatz laden (`_load_user(app, user_id)`) oder neu anlegen: `user_id` und `username` aus `payload`, `email` leer, `level = -1`, `settings = {"device_owner": True}`.
   4. **Altdaten übernehmen:** `old = await _find_user_by_email(app, LOCAL_ADMIN_EMAIL)`. Wenn `old` existiert und `old.user_id != user_id`: jeden Schlüssel aus `old.settings` (außer `anonymous`, `local_admin`) und aus `old.mod_data` in den neuen Datensatz übernehmen, **wenn er dort noch nicht existiert** (Konto gewinnt bei Konflikt). Danach `_db_delete` für `AUTH_USER::<old.user_id>` und `AUTH_USER_EMAIL::<LOCAL_ADMIN_EMAIL>`. `migrated_from = old.user_id`.
   5. `_save_user(app, user)`, dann `device_owner.json` schreiben.
   6. Mehrfacher Aufruf mit demselben Konto ändert nichts (idempotent). Aufruf mit einem **anderen** Konto, während schon eines verknüpft ist → `RuntimeError("device already linked")`.
4. `async unlink_account(app) -> None`: `device_owner.json` löschen und `await app.session.logout()`.

**Test `toolboxv2/tests/test_core/test_device_account.py`** (unittest; `_db_set`, `_db_get`, `_db_get_raw`, `_db_delete` in `user_store` und `device_account` per `unittest.mock.patch` durch ein Dict ersetzen; `app.session` durch ein Mock, dessen `login()` `True` liefert; `app.data_dir` = Temp-Ordner):

| Fall | Erwartet |
|---|---|
| Verknüpfen ohne lokalen Admin | Datei existiert, Nutzer mit `level -1` gespeichert |
| Verknüpfen mit lokalem Admin, der `settings={"theme":"dark"}` und `mod_data={"m":{"a":1}}` hat | Neuer Nutzer hat beides, alter Datensatz und E-Mail-Index gelöscht, `migrated_from` gesetzt |
| Konto hat `theme: light`, lokaler Admin `theme: dark` | bleibt `light` |
| Zweiter Aufruf, gleiches Konto | keine Änderung, kein Fehler |
| Zweiter Aufruf, anderes Konto | `RuntimeError` |
| `login()` liefert `False` | `RuntimeError`, keine Datei |

**Gate:** Standard-Gate + neuer Test `OK`.

---

### K2 — Automatische Anmeldung über das Gerätekonto (F8)

**Dateien:** `toolboxv2/utils/workers/session.py`, `toolboxv2/utils/workers/server_worker.py`

**Schritt 1 — `SessionManager` (session.py ab Zeile 524):**

- In `__init__`: `self._local_trust_provider = None`.
- Neue Methode `enable_local_trust(self, provider) -> None`: `provider` ist eine Funktion ohne Argumente, die ein Dict (`user_id`, `username`, `level`) oder `None` liefert.
- Neue Methode `_local_trust_session(self, environ, headers)`: gibt `None` zurück, außer **alle** Bedingungen gelten:
  1. Ein Provider ist gesetzt und liefert ein Dict mit nicht-leerer `user_id`.
  2. `environ.get("REMOTE_ADDR")` ist `127.0.0.1` oder `::1`.
  3. Keiner der Header `X-Forwarded-For`, `X-Real-Ip`, `Forwarded` ist vorhanden (Groß-/Kleinschreibung egal).
  4. Der Host aus `HTTP_HOST` (ohne Port) ist `127.0.0.1`, `localhost` oder `[::1]`.
  Dann eine `SessionData` zurückgeben mit `user_id`, `user_name`, `level`, `session_id="local-trust"`, `validated=True`, `anonymous=False`, `exp = jetzt + cookie_max_age`.
- In `get_session_from_request` (Zeile 685) **und** `get_session_from_request_sync` (Zeile 738): direkt vor „4. Return anonymous" `_local_trust_session` aufrufen und dessen Ergebnis zurückgeben, wenn es nicht `None` ist.

**Schritt 2 — `server_worker.py:1682–1700`:** Den ganzen Block ersetzen. Neu: Wenn das Profil `local` oder `developer` ist, `self._session_manager.enable_local_trust(provider)` aufrufen. `provider` liest `read_device_owner(self._app)` und gibt `{"user_id", "username", "level": -1}` oder `None` zurück; das Ergebnis darf 5 s zwischengespeichert werden. Die Bedingungen auf `TOOLBOXV2_REMOTE_BASE` und `environment` entfallen. `ensure_local_admin` wird hier **nicht** mehr aufgerufen. Profil `server` oder fehlendes Profil → kein Provider.

**Schritt 3 — `mods/CloudM/module.py:93–109`:** Wenn `read_device_owner(self.app)` ein Konto liefert, `app.session.username`, `app.session.user_id`, `app.session.valid = True` daraus setzen und `app.session.base` **nicht** auf `None` setzen. Nur wenn kein Gerätekonto existiert, bleibt das bisherige Verhalten (lokaler Admin für die CLI).

**Test `toolboxv2/tests/test_core/test_local_trust.py`** (unittest):

| Fall | Erwartet |
|---|---|
| Provider liefert Konto, `REMOTE_ADDR=127.0.0.1`, `HTTP_HOST=127.0.0.1:8467` | Session des Kontos, `anonymous False`, `validated True` |
| Provider liefert Konto, `REMOTE_ADDR=192.168.1.5` | anonym |
| Provider liefert Konto, Loopback, Header `X-Forwarded-For: 1.2.3.4` | anonym |
| Provider liefert Konto, Loopback, `HTTP_HOST=evil.example` | anonym |
| Provider liefert `None`, Loopback | anonym |
| Kein Provider gesetzt, Loopback | anonym |

**Gate:** Standard-Gate + beide neuen Tests `OK`. Im Worker-Log steht nicht mehr `Loopback trust not enabled`.

**Nicht tun:** Keine Anmeldung für Nicht-Loopback-Anfragen. CSRF- und CORS-Regeln nicht anfassen.

---

### K3 — Konto ist Pflicht: Zustand, API, Seite

**Schritt 1 — `utils/clis/first_run.py`, `onboarding_state()`:** Das Ergebnis bekommt zwei neue Felder: `needs_account` (`True`, wenn Profil `local` oder `developer` ist und `read_device_owner()` `None` liefert) und `account` (`username` des Gerätekontos oder `None`). Die Bedeutung von `first_run` bleibt unverändert.

**Schritt 2 — `utils/workers/fast/wizard_api.py`:**

- `GET /api/wizard/state` gibt zusätzlich `needs_account` und `account` zurück.
- Neu `POST /api/wizard/account/start` (`auth=False`): startet **einen** Hintergrund-Thread, der `cli_remote_login()` ausführt (öffnet den Standard-Browser auf der Login-Seite von SimpleCore), danach `link_account(app, payload)`. Vorher `TOOLBOXV2_REMOTE_BASE` bzw. Default `https://simplecore.app` als `app.session.base` setzen. Läuft schon ein Thread: nichts Neues starten. Antwort `{"ok": true, "state": "waiting"}`.
- Neu `GET /api/wizard/account/status` (`auth=False`): `{"state": "idle" | "waiting" | "linked" | "error", "error": <Text oder null>, "account": <username oder null>}`.
- Beide neuen Routen nur für Loopback (dieselbe Prüfung wie in `local_ui.py:36`).

**Schritt 3 — neue Datei `utils/workers/fast/account_page.py`:** Eine Konstante `ACCOUNT_PAGE_HTML` mit einer vollständigen, eigenständigen HTML-Seite (kein externes Skript, kein externes Stylesheet): Überschrift „Mit SimpleCore anmelden", ein Knopf, eine Statuszeile. Klick → `POST /api/wizard/account/start`, dann alle 2 s `GET /api/wizard/account/status`; bei `linked` → `location.replace('/')`; bei `error` → Fehlertext anzeigen und Knopf wieder aktivieren. HTML als Python-String, damit keine Paketdaten nötig sind.

**Schritt 4 — `utils/workers/fast/local_ui.py`:**

- Neue Route `GET /account` (`auth=False`): liefert `ACCOUNT_PAGE_HTML`.
- Route `/` (Zeile 114): Reihenfolge 1. `first_run` → Setup-Seite (wie bisher), 2. sonst `needs_account` → `ACCOUNT_PAGE_HTML`, 3. sonst Dashboard.

**Schritt 5 — Tests.** `test_wizard_api.py` und `test_wizard_e2e_http.py` ergänzen: Zustand mit Profil, ohne Gerätekonto → `needs_account true`, `/` liefert die Kontoseite; mit `device_owner.json` im Temp-Datenordner → `needs_account false`, `/` liefert das Dashboard. Bestehende Fälle, die nach gesetztem Profil das Dashboard erwarten, legen in ihrem `setUp` ein `device_owner.json` an (das ist die neue Voraussetzung, keine Abschwächung).

**Gate:** Standard-Gate + `python -m pytest -q toolboxv2/tests/test_core/test_wizard_e2e_http.py` bestanden.

**Nicht tun:** Keine Umgebungsvariable und kein Schalter, der die Kontopflicht umgeht.

---

### K4 — Abnahme Konto (manuell, mit der Installation aus T6)

Wheel neu bauen, `installer.ps1` erneut ausführen, Workers starten.

| Schritt | Erwartet |
|---|---|
| `curl http://127.0.0.1:8467/api/wizard/state` | `"first_run":false`, `"needs_account":true` |
| Browser `http://127.0.0.1:8467/` | Kontoseite |
| Knopf klicken, im geöffneten Browser bei SimpleCore anmelden | Seite springt auf das Dashboard |
| `curl http://127.0.0.1:8467/api_user_data` | Datensatz des SimpleCore-Kontos (`user_id` wie bei P1 aus K0) |
| Workers stoppen, **Netzwerk trennen**, Workers starten, `/` öffnen | Dashboard, angemeldet, keine Kontoseite |
| `dir "$env:LOCALAPPDATA\toolboxv2\.data"` (rekursiv nach `device_owner.json` suchen) | Datei existiert, enthält kein Token |

**Wenn rot:** Bleibt die Seite nach dem Login auf „waiting" → prüfen, ob Port 8765 (Rückkanal von `cli_oauth.py`) von einer Firewall blockiert wird; Befund melden, nicht den Port ändern.

---

### T10 — ISAA im Owner abnehmen

**Voraussetzung:** K4 ist grün (Gerätekonto verknüpft). **H0 (Owner):** einen LLM-API-Key bereitstellen und über `tb -init onboard` (Frage 2) eintragen.

**Schritte:** Kein Code, außer ein Gate ist rot.

**Gate** (Installation aus T6, Workers laufen):

```powershell
curl -X POST -H "Content-Type: application/json" -H "Origin: http://127.0.0.1:8467" -d "{\"command\":\"start_app\",\"args\":{\"name\":\"isaa\"}}" http://127.0.0.1:8467/tray/command
curl -s -o NUL -w "%{http_code}" http://127.0.0.1:8467/isaa
```

Erwartet: `"ok":true` mit `"prefix":"/isaa"`; danach `200`. Dann im Browser `http://127.0.0.1:8467/` → Tab „Lokal" → ISAA → eine Nachricht senden → Antwort erscheint. In den Entwicklerwerkzeugen (Netzwerk → WS) besteht eine Verbindung zu Port **8468**.

**Wenn rot**

| Symptom | Maßnahme |
|---|---|
| `start_app` meldet einen Routenkonflikt | Konflikt-Route und beide Quellen melden, stoppen |
| WS verbindet auf einen anderen Port als 8468 | Stelle im ISAA-Frontend (`mods/isaa/ui/static`) finden, die die WS-URL baut; auf den vom Owner gelieferten Port umstellen (Regel 6) |
| `ModuleNotFoundError` im Log | Fehlendes Paket in das Extra `isaa` in `pyproject.toml` eintragen, Wheel neu bauen, neu installieren |

---

### T11 — OBS als App im Owner (F11, B1)

**Schritt 1 — `mods/isaa/extras/live_obs_server.py:1537`:** `module_path="obs"` → `module_path="toolboxv2.mods.isaa.extras.live_obs_server"`.

**Schritt 2 — Start ohne `main()`.** Beim Mounten läuft `main()` nicht. Eine idempotente Funktion `ensure_started()` ergänzen, die genau einmal `start_disk_scanner(str(Path(get_app().data_dir) / "Agents"))` aufruft; sie am Anfang des Handlers von `@app.get("/")` (Zeile 691) aufrufen.

**Schritt 3 — WS-URL.** Bei Zeile 1320 (`ws=new WebSocket(url)`) nachsehen, wie `url` gebildet wird. Sie muss auf dieselbe Weise entstehen wie im ISAA-Frontend aus T10 (WS-Port des Owners, Pfad mit Präfix `/obs`). Wenn nicht: angleichen.

**Schritt 4 — Registry.** In `toolboxv2/features/isaa/feature.yaml` unter `user_facing` einen zweiten Eintrag ergänzen:

```yaml
- name: obs
  function: ui
  label: OBS
  icon: monitoring
  description: Live-Beobachtung der Agenten
  mount:
    module: toolboxv2.mods.isaa.extras.live_obs_server
    attr: app
    prefix: obs
```

**Schritt 5 — Test.** In `test_user_apps.py` analog zu `test_isaa_feature_declares_lazy_app` einen Test für den Eintrag `obs` ergänzen.

**Gate:** Standard-Gate. Wheel neu bauen, neu installieren, Workers starten:

```powershell
curl http://127.0.0.1:8467/tray/apps
curl -X POST -H "Content-Type: application/json" -H "Origin: http://127.0.0.1:8467" -d "{\"command\":\"start_app\",\"args\":{\"name\":\"obs\"}}" http://127.0.0.1:8467/tray/command
curl -s -o NUL -w "%{http_code}" http://127.0.0.1:8467/obs
```

Erwartet: `/tray/apps` listet `isaa` und `obs`; `"ok":true`; `200`. Im Browser zeigt `/obs` nach einer ISAA-Nachricht neue Einträge. Beide Apps gleichzeitig gemountet → beide weiter erreichbar.

**Wenn rot:** Kollidiert eine OBS-Route mit Owner oder ISAA, die Route melden und stoppen (kein Umbenennen auf eigene Faust).

---

### S1 — Server: ein Schlüssel pro Konto (F18)

**Entscheidung des Owners:** Der Schlüssel für synchronisierte Daten liegt beim Konto auf dem Server und wird nach dem Login an das Gerät gegeben.

**Datei:** `toolboxv2/mods/CloudM/UserDataAPI.py`

**Schritte**

1. Neue Funktion mit demselben Dekorator wie `get_secure_storage_credentials` (Zeile 1096): `async def get_user_data_key(app, request)`.
   - Nutzer über `_get_storage_provider(app, request)` bestimmen; ohne Nutzer → `Result.default_user_error(info="Nicht authentifiziert", exec_code=401)`.
   - Schlüssel unter dem DB-Schlüssel `AUTH_USER_DATAKEY::<user_id>` lesen (`_db_get_raw` aus `auth/db_helpers.py`). Fehlt er: mit `Code.generate_symmetric_key()` (`utils/security/cryp.py:148`) erzeugen und mit `_db_set` speichern.
   - Antwort `Result.ok(data={"key": <schlüssel>, "user_id": <user_id>})`.
2. Der Schlüssel wird nie geloggt.

**Test `toolboxv2/tests/test_core/test_user_data_key.py`** (unittest, DB wie in K1 durch ein Dict ersetzt, `_get_storage_provider` gepatcht):

| Fall | Erwartet |
|---|---|
| Nutzer A, zwei Aufrufe | zweimal derselbe Schlüssel |
| Nutzer A und Nutzer B | verschiedene Schlüssel |
| nicht angemeldet | Fehler 401, kein Schlüssel |

**Gate:** Standard-Gate + neuer Test `OK`.

**H5 (Owner):** Diese Änderung auf `simplecore.app` ausrollen. Der Agent stoppt nach S1, bis der Owner bestätigt, dass `curl -H "Authorization: Bearer <token>" https://simplecore.app/api/CloudM.UserDataAPI/get_user_data_key` einen Schlüssel liefert.

---

### S2 — Gerät: Schlüssel und Zugangsdaten holen, Blob-Speicher auf das Konto stellen (F19, F20)

**Voraussetzung:** K0-P3 war erfolgreich und H5 ist erledigt. Sonst stoppen.

**Neue Datei:** `toolboxv2/mods/CloudM/auth/device_sync.py`

**Funktionen**

1. `async fetch_account_key(app) -> str`: `app.session.fetch("/api/CloudM.UserDataAPI/get_user_data_key")`; den Schlüssel lokal verschlüsselt ablegen — mit `BlobFile` im selben Speicher und mit demselben Geräteschlüssel wie die Tokens (`utils/system/session.py:129–160` als Vorlage, Blob-Name `device_data_key`).
2. `load_cached_account_key(app) -> Optional[str]`: liest den abgelegten Schlüssel, ohne Netz.
3. `async fetch_cloud_credentials(app) -> dict`: `app.session.fetch("/api/CloudM.UserDataAPI/get_secure_storage_credentials")`. Ergebnis **nicht** auf Platte speichern (kurzlebig).
4. `build_account_blob_storage(app) -> Optional[BlobStorage]` (kein Netz): Wenn `read_device_owner(app)` und `load_cached_account_key(app)` beide etwas liefern: `BlobStorage(mode=StorageMode.OFFLINE, user_id=<Konto-ID>, encryption_key=<Schlüssel>, auto_sync=False)`. Sonst `None`.
5. `attach_cloud(storage, creds) -> None`: `storage._cloud_minio = Minio(creds["endpoint"], access_key=..., secret_key=..., secure=creds["secure"])` und `storage.bucket` auf den privaten Bucket aus `creds` setzen. Die Feldnamen stehen im Messprotokoll aus K0-P3 — exakt diese verwenden.
6. `migrate_local_blobs(old_storage, new_storage) -> int`: alle Blob-IDs aus `old_storage.list_blobs()` lesen, jeden mit `old_storage.read_blob(...)` entschlüsselt laden und mit `new_storage` neu schreiben. Rückgabe = Anzahl. (Grund: bisherige Blobs sind mit dem Geräteschlüssel verschlüsselt und wären sonst nach dem Umstellen unlesbar. Das ist zugleich die Übernahme der Altdaten.)

**Verdrahtung**

- `utils/toolbox.py:290`: Vor dem bestehenden Ausdruck `build_account_blob_storage(self)` versuchen; liefert es einen Speicher, wird er `self.root_blob_storage`. Sonst bleibt die Zeile wie sie ist.
- `device_account.link_account` (K1), ganz am Ende: `fetch_account_key` → neuen Speicher bauen → `migrate_local_blobs(alter, neuer)` → `app.root_blob_storage = neuer`. Ein Fehler hier bricht das Verknüpfen **nicht** ab; er wird geloggt und im Kontostatus als Hinweis gemeldet.

**Test `toolboxv2/tests/test_core/test_device_sync.py`** (unittest, Temp-Ordner, kein Netz):

| Fall | Erwartet |
|---|---|
| Kein Gerätekonto | `build_account_blob_storage` → `None` |
| Gerätekonto + Schlüssel im Cache | Speicher mit `user_id` = Konto-ID |
| Blob „a" mit Geräteschlüssel geschrieben, dann `migrate_local_blobs` | Neuer Speicher liest „a" mit identischem Inhalt |
| Zwei Speicher in zwei Temp-Ordnern mit **demselben** Kontoschlüssel: A schreibt Blob, Rohdaten nach B kopiert | B liest den Inhalt (beweist: Schlüssel ist nicht gerätegebunden) |

**Gate:** Standard-Gate + neuer Test `OK`.

**Wenn rot:** Weichen die Methodennamen von `BlobStorage` ab (`read_blob`, `list_blobs`, `create_blob`, `update_blob` stehen in `utils/extras/blobs.py:697–923`), dort die Signaturen lesen und exakt verwenden.

---

### S3 — Kontodaten über die Data-API laden und abgleichen (F21)

**Grundsatz:** Daten der Data-API laufen **immer über HTTPS zum Server** (dort entschlüsselt der Server), nie direkt nach MinIO. Grund: Bestehende Serverdaten sind mit dem Schlüssel des Servers verschlüsselt.

**Schritt 1 — Messung (kein Code).** Auf dem Gerät mit verknüpftem Konto prüfen und im Messprotokoll festhalten:

| Frage | Wie prüfen |
|---|---|
| M1: Bleibt ein lokales `set_data(path="probe/a", data={"x":1}, scope="private")` nach einem Neustart der Workers lesbar? | `curl -X POST http://127.0.0.1:8467/api/CloudM.UserDataAPI/set_data ...`, Neustart, `get_data` |
| M2: Dasselbe für `set_mod_data(source_mod="probe", data={"x":1})` / `get_mod_data` | wie M1 |

**Schritt 2 — lokale Ablage.** Wenn M1 „nein" ergibt: in `UserDataAPI._get_storage_provider` (Zeile 358–389) für Geräte mit Gerätekonto `local_db_path = <app.data_dir>/user_data.db` übergeben und im `UserContext` `encryption_key` = Kontoschlüssel (`load_cached_account_key`) setzen. Dann M1 wiederholen → muss „ja" sein.

**Schritt 3 — Abgleich in `device_sync.py`:**

- `async pull_account_data(app) -> dict`: entfernt `list_data` (Scope `private`) aufrufen, für jeden Pfad entfernt `get_data`, Ergebnis lokal mit dem lokalen Speicher schreiben, wenn der Pfad lokal fehlt oder der entfernte Stand neuer ist. Rückgabe `{"downloaded": n}`.
- `async push_account_data(app) -> dict`: lokal geänderte Einträge (`get_dirty_blobs()` der lokalen DB) per entferntem `set_data` hochladen und danach als synchron markieren. Rückgabe `{"uploaded": n}`.
- `async sync_account_data(app) -> dict`: erst `push`, dann `pull`.
- Die Form der Antworten steht im Messprotokoll (K0-P2, K0-P4) — exakt diese verwenden.

**Schritt 4 — Mod-Daten.** Nur wenn M2 „ja" ergibt: entferntes `get_mod_data` / `set_mod_data` analog einbinden. Ergibt M2 „nein": **stoppen und melden**, nichts erfinden.

**Test** (unittest, entfernte Aufrufe durch ein Mock von `app.session.fetch` ersetzt):

| Fall | Erwartet |
|---|---|
| Entfernt 2 Einträge, lokal leer | `pull` lädt 2, beide lokal lesbar |
| Lokal 1 geänderter Eintrag | `push` sendet genau 1 `set_data`, Eintrag danach nicht mehr „dirty" |
| Entfernter Aufruf schlägt fehl (kein Netz) | kein Absturz, Rückgabe enthält `error`, lokale Daten unverändert |

**Gate:** Standard-Gate + neuer Test `OK`.

---

### S4 — Blobs und der globale ISAA-Ordner (F20, F22)

**Schritt 1 — Blob-Sync.** In `device_sync.py`: `async sync_blobs(app) -> dict`: `fetch_cloud_credentials` → `attach_cloud(app.root_blob_storage, creds)` → `app.root_blob_storage.sync()` (existiert, `blobs.py:1021`). Ohne Netz oder ohne Gerätekonto: `{"status": "skipped"}`.

**Schritt 2 — globaler ISAA-Ordner.** Der Ordner ist `<app.data_dir>/Agents/VFS/global` (`mods/isaa/base/patch/power_vfs.py:110`). In `device_sync.py`:

- `global_dir(app) -> Path`: genau dieser Pfad. `power_vfs` **nicht** importieren (zieht ISAA-Abhängigkeiten).
- `export_global_folder(app) -> int`: jede Datei im Ordner (rekursiv) als Blob mit der ID `isaa/global/<relativer Pfad mit />` schreiben, wenn der Blob fehlt oder die Datei neuer ist als der Blob.
- `import_global_folder(app) -> int`: jeden Blob mit Präfix `isaa/global/` als Datei schreiben, wenn die Datei fehlt oder der Blob neuer ist.
- Pfade, die `..` enthalten oder absolut sind, werden übersprungen und geloggt.
- Löschungen werden **nicht** übertragen (eine lokal gelöschte Datei kommt beim nächsten Import zurück). Dateien über 50 MB werden übersprungen und geloggt.

**Schritt 3 — ein Einstieg.** `async sync_all(app) -> dict` in dieser Reihenfolge: `export_global_folder` → `sync_blobs` → `import_global_folder` → `sync_account_data`. Rückgabe: ein Dict mit den Teilergebnissen und `finished_at`.

**Schritt 4 — Auslöser.**

- Nach erfolgreichem `link_account` (K1): einmal `sync_all`.
- Beim Start des Owner-Prozesses, wenn ein Gerätekonto existiert: einmal `sync_all` im Hintergrund; danach alle 10 Minuten.
- Neu `POST /api/wizard/account/sync` (nur Loopback, nur mit Gerätekonto): stößt `sync_all` an und gibt das Ergebnis zurück. `GET /api/wizard/account/status` (K3) zeigt zusätzlich `last_sync`.
- Ein Fehler im Sync darf den Owner-Prozess nie beenden.

**Test** (unittest, zwei Temp-Ordner als „Gerät A" und „Gerät B", ein gemeinsamer Fake-Cloud-Speicher statt MinIO, gleicher Kontoschlüssel):

| Fall | Erwartet |
|---|---|
| A legt `global/notes/a.txt` an, A `export`, Blob in die Fake-Cloud, B `import` | B hat `global/notes/a.txt` mit gleichem Inhalt |
| Blob-ID `isaa/global/../x` | wird nicht geschrieben |
| Kein Gerätekonto | `sync_all` → `skipped`, kein Fehler |

**Gate:** Standard-Gate + neuer Test `OK`.

---

### S5 — Abnahme Sync (manuell, echtes Konto)

Zwei Installationen desselben Kontos (zweiter Windows-Rechner oder zweites Windows-Benutzerkonto).

| Schritt | Erwartet |
|---|---|
| Gerät 1: Datei `…\.data\<id>\Agents\VFS\global\probe.txt` anlegen, `curl -X POST -H "Origin: http://127.0.0.1:8467" http://127.0.0.1:8467/api/wizard/account/sync` | Antwort ohne `error`, mindestens 1 Upload |
| Gerät 2: frisch installieren, mit demselben Konto anmelden | Nach dem Login existiert `probe.txt` mit gleichem Inhalt |
| Gerät 2: `get_data` für einen Pfad aus K0-P2 lokal aufrufen | Liefert die bestehenden Kontodaten |
| Gerät 2: lokal `set_data(path="probe/b", ...)`, Sync auslösen; Gerät 1: Sync auslösen, `get_data("probe/b")` | Gleicher Inhalt |
| Gerät 1: Netz trennen, App neu starten | Startet angemeldet, `probe.txt` weiter lesbar |

**Wenn rot:** Gerät 2 kann Blobs nicht entschlüsseln → der Kontoschlüssel wurde nicht geholt oder nicht verwendet; S2 prüfen. Zugriff auf MinIO verweigert → Bucket-Name oder Präfix passt nicht zu den ausgegebenen Rechten; mit dem Messprotokoll K0-P3 vergleichen.

---

### H1 — Signaturschlüssel (Owner)

Der Agent stoppt hier, bis der Owner bestätigt:

1. Der private Schlüssel zum `pubkey` in `tauri.conf.json` liegt vor. Falls nicht: `npx tauri signer generate -w %USERPROFILE%\.tauri\simplecore.key` ausführen und dem Agenten den **öffentlichen** Schlüssel geben (der Agent trägt ihn als `plugins.updater.pubkey` ein).
2. GitHub-Secrets im Repo gesetzt: `TAURI_SIGNING_PRIVATE_KEY`, `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`.

---

### T12 — App-Update über GitHub (F12)

**Kanal dieser Session:** das Release `nightly` (einziger Weg, der heute Tauri-Bundles in ein GitHub-Release legt). Stabile Releases sind ein Folge-Task.

**Schritt 1 — `Cargo.toml`:** unter `[target.'cfg(not(any(target_os = "android")))'.dependencies]` ergänzen: `tauri-plugin-updater = "2"`.

**Schritt 2 — `lib.rs`:**
- Im Desktop-Zweig des Builders (dort, wo `tauri_plugin_autostart` registriert wird): `.plugin(tauri_plugin_updater::Builder::new().build())`.
- Neue Funktion `spawn_update_check(app: AppHandle)` (nur Desktop): 15 s warten, dann `app.updater()` → `check().await`. Bei `Ok(Some(update))`: Log-Eintrag mit der Version, `update.download_and_install(...)`, danach `app.restart()`. Bei `Ok(None)` oder `Err`: nur loggen, nie abstürzen.
- Aufruf in `setup` direkt nach `spawn_bootstrap(...)`.

**Schritt 3 — `capabilities/` (Desktop-Datei):** Berechtigung `updater:default` ergänzen.

**Schritt 4 — `tauri.conf.json`:**
- `bundle.createUpdaterArtifacts: true`
- `plugins.updater.endpoints`: `["https://github.com/MarkinHaus/ToolBoxV2/releases/download/nightly/latest.json"]`

**Schritt 5 — `nightly.yml`:**
- Vor dem Tauri-Build einen Schritt einfügen, der in `tauri.conf.json` die `version` auf `<Version aus pyproject.toml>-<github.run_number>` setzt (nur Ziffern nach dem Bindestrich; MSI verlangt das).
- Im Schritt „Build Tauri (nightly)" unter `env` ergänzen: `TAURI_SIGNING_PRIVATE_KEY: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY }}` und `TAURI_SIGNING_PRIVATE_KEY_PASSWORD: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY_PASSWORD }}`.

**Gate:** `cargo test --lib` bestanden. Lokaler Build wie in T8 (mit `createUpdaterArtifacts:false`) erfolgreich. Der echte Nachweis erfolgt in T14.

**Wenn rot:** Kompiliert der Updater-Aufruf nicht, die API in der Doku des installierten Plugins nachlesen (`cargo doc -p tauri-plugin-updater --open`) statt zu raten.

---

### T13 — Kern-Update beim App-Start

**Ziel:** Bringt die App ein neueres Wheel mit als installiert ist, aktualisiert sie den Kern vor dem Start.

**Schritt 1 — `installer.rs`:**
- Reine Funktion `wheel_version(path: &Path) -> Option<String>`: aus `toolboxv2-<version>-py3-none-any.whl` die Version lesen.
- Reine Funktion `installed_version(install_dir: &Path) -> Option<String>`: aus `<install_dir>/install.manifest` den Wert der Zeile `tb_version:` lesen (Anführungszeichen entfernen).
- Reine Funktion `core_needs_update(bundled: Option<String>, installed: Option<String>) -> bool`: `true` nur, wenn beide vorhanden und verschieden sind.
- Funktion `run_core_update(app)`: wie `run_install`, aber die Config enthält nur `action: update`, `assume: yes`, `ui: none`; `TB_WHEEL` und `TB_UV_BIN` werden gesetzt. Gemeinsamen Code aus `run_install` in eine Hilfsfunktion ziehen statt kopieren.
- Unit-Tests für die drei reinen Funktionen.

**Schritt 2 — `lib.rs`, `bootstrap_target()`:** Ganz am Anfang: wenn `launch::tb_on_path()` und `core_needs_update(...)`:
1. laufende Workers stoppen: `<tb> workers stop` (Befehl vorher mit `tb workers --help` bestätigen), dann warten, bis `probe_health` `false` liefert (max. 30 s);
2. `run_core_update` ausführen;
3. bei Fehler: loggen und normal weiter (alter Kern startet).
Danach läuft die bestehende Logik unverändert.

**Gate:** `cargo test --lib` bestanden. Manuell: `tb_version` in `%LOCALAPPDATA%\toolboxv2\install.manifest` von Hand auf `0.0.1` setzen, App starten → Log `install.log` zeigt einen Update-Lauf, danach steht dort wieder `0.1.29`; `curl http://127.0.0.1:8467/api/wizard/state` zeigt weiter `"profile":"local"`, `first_run:false`, `needs_account:false`.

**Wenn rot:** Scheitert das Update mit „Datei wird verwendet", läuft noch ein `tb`-Prozess — die Wartezeit in Punkt 1 prüfen, nicht `--force` oder Prozess-Kill einbauen.

---

### T14 — Gesamtabnahme

**Schritte**

1. Alles committen und pushen. Der Owner stößt `nightly.yml` an (**H2**).
2. Auf einem Windows-Konto ohne `%LOCALAPPDATA%\toolboxv2` das Nightly-Setup installieren.
3. Punkte Z1–Z6 aus Abschnitt 0 durchgehen; Z8 nach dem Ablauf von S5 mit der Nightly-App wiederholen.
4. Der Owner stößt `nightly.yml` ein zweites Mal an (**H3**). App neu starten → Z7: Update wird installiert, App startet neu; `install.manifest` zeigt die neue Version; ISAA-Chatverlauf und Profil sind erhalten.

**Gate:** Tabelle Z1–Z8 vollständig mit „ok" oder genauer Fehlermeldung ausgefüllt an den Owner melden.

**Wenn rot**

| Symptom | Zurück zu |
|---|---|
| Installer-Fenster meldet Fehler | `logs\install.log` neben der App-EXE lesen → T5/T6 |
| Fehlerseite „installed-stopped-timeout" | `%LOCALAPPDATA%\ToolBoxV2\app\logs\workers_start.log` lesen → T4/T8 |
| Onboarding erscheint erneut | T3 |
| Kontoseite erscheint nach dem ersten Login erneut | K1/K2 |
| Daten fehlen auf dem zweiten Gerät | S2–S4 |
| Kein Update gefunden | Prüfen, ob `latest.json` im Release `nightly` liegt und die Version höher ist → T12 |

---

## 5. Risiken, die unterwegs auftreten können

| Risiko | Erkennbar an | Vorgesehene Reaktion |
|---|---|---|
| `file:///`-Wheel-URL wird von `uv` unter Windows anders behandelt | T5-Gate | Ausweichform in T5 „Wenn rot" |
| Erste Installation lädt ~580 MB Abhängigkeiten, langsam oder offline | T6/T14 dauert Minuten | Kein Code; im Installer-Fenster läuft der Fortschritt. Offline-Installation ist nicht Teil des Ziels |
| Windows sperrt Dateien eines laufenden `tb` beim Update | T13-Gate | Workers vorher stoppen und auf Ende warten |
| ISAA lädt im Owner-Prozess, Start wird langsam | T10: `/health` antwortet erst spät | Messen und melden; die Wartezeit ist in T8 auf 120 s erhöht |
| Routenkonflikt ISAA/OBS/Owner | T10/T11 | Melden und stoppen |
| MinIO-Verbindungsversuche auf `localhost:9000` im Profil `local` | Warnungen im Log | Ignorieren, solange Z2 in unter 120 s erfüllt ist |
| Kein privater Signaturschlüssel vorhanden | H1 | Neues Schlüsselpaar; bereits verteilte Apps können dann nicht aktualisiert werden |
| CI-Windows-Runner scheitert an `fetch_uv_sidecar.sh` (Handoff V2) | H2 | Log des Schritts lesen, nur diesen Schritt reparieren |
| `simplecore.app` gibt keine MinIO-Zugangsdaten aus | K0-P3 | S2 und S4 blockiert bis H4; Konto und Data-API-Sync (S3) gehen trotzdem |
| MinIO des Servers ist von außen nicht per HTTPS erreichbar | K0-P3, S5 | Wie oben; Owner-Entscheidung |
| Kurzlebige MinIO-Zugangsdaten laufen während eines Syncs ab | Fehler im Sync-Ergebnis | Vor jedem `sync_blobs` neu holen (so in S4 vorgesehen) |
| Gleiche Daten auf zwei Geräten geändert | S5 | Bestehendes Verhalten: lokale Änderung wird hochgeladen, sonst gewinnt der neuere Stand. Keine Zusammenführung |
| Lokale Blobs nach dem Umstellen auf den Kontoschlüssel unlesbar | S2-Test | `migrate_local_blobs` vor dem Umschalten |
| Rückkanal-Port 8765 des Logins belegt oder blockiert | K4 | Melden; Port nicht ändern |
| Nutzer verliert Zugang zum Konto | — | Lokale Daten bleiben lesbar (Schlüssel liegt im Geräte-Cache); kein Code in dieser Session |

---

## 6. Bewusst offen (nicht anfassen)

Handoff B2, B3, B4, B5, B6, B7, B9; V5; Punkt E; Abschnitt 4 des Handoffs komplett. Dazu neu: ISAA-Chatverläufe pro Nutzer speichern und synchronisieren (`isaa/ui/chat_store.py` kennt keine Nutzer-ID); Löschungen im globalen Ordner übertragen; Abmelden/Kontowechsel in der Oberfläche; bestehende Serverdaten auf den Kontoschlüssel umstellen; `launch.rs` kennt unter Linux/macOS den Installationspfad nicht (`tb_candidates` deckt nur Windows ab); stabiler Update-Kanal über `release.yml`; Entwicklerprofil in der Tauri-App (Source-Installation braucht Git).
