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
| Z3 | Anmeldung | Ohne Eingabe als lokaler Admin angemeldet (kein Login-Formular) |
| Z4 | ISAA | Dashboard → Tab „Lokal" → ISAA öffnet `/isaa`, eine Chat-Nachricht wird beantwortet |
| Z5 | OBS | Dashboard → Tab „Lokal" → OBS öffnet `/obs` und zeigt Live-Daten des ISAA-Laufs |
| Z6 | Neustart | App schließen und neu starten → direkt Dashboard, kein Onboarding |
| Z7 | Update | Neuere App-Version wird gefunden und installiert; danach läuft der neue Kern, Profil und Daten sind erhalten |

Nicht Teil des Ziels: Konto-Login gegen `simplecore.app`, Linux/macOS-Abnahme, Punkt E (ein HTTP-Stack), README, Dead-Code, Installer-Konsolidierung, B2, B3, B9.

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
| F8 | `SessionManager.enable_local_trust` wird in `server_worker.py:1697` aufgerufen, ist aber nirgends definiert. Log: `Loopback trust not enabled`. | Keine Auto-Anmeldung → T9 |
| F9 | `POST /tray/command` mit `{"command":"start_app","args":{"name":"isaa"}}` → `{"ok":true,"result":{"merged":true,"prefix":"/isaa","ws":true}}`, `/isaa` → 200. | ISAA-Mount funktioniert |
| F10 | `litellm` steht in `features/isaa/feature.yaml`, fehlt aber im Extra `isaa` in `pyproject.toml`. | ISAA-Chat ohne LLM-Bibliothek → T4 |
| F11 | OBS (`mods/isaa/extras/live_obs_server.py`, `app = FastTB(...)` Zeile 41) ist nicht in der App-Registry. | T11 |
| F12 | `tauri.conf.json` hat `plugins.updater`, aber `tauri-plugin-updater` fehlt in `Cargo.toml` und `lib.rs`. App-Version ist fest `0.1.0`. | Kein Update möglich → T12 |
| F13 | `bundle.externalBin` verlangt `binaries/tb-worker`; laut Handoff B10 funktioniert dieses Binary nicht. `worker_manager.rs` erlaubt Endpoint `local` nur, wenn der Sidecar da ist. | T8 |
| F14 | `launch.rs` wartet nach dem Start nur 25 s auf 8467. | Erster Start unter Windows kann länger dauern → T8 |
| F15 | Tests und `import toolboxv2` schreiben Marker in den Arbeitsbaum (`features/*/.installed`, `logs/`, `.info/`). | T2 |

---

## 3. Reihenfolge

```
T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 → T9 → T10 → T11 → H1 → T12 → T13 → T14
```

Begründung, rückwärts vom Ziel: Update (T12/T13) braucht eine installierbare App (T6–T8). Die App braucht einen Installer, der das mitgelieferte Wheel nimmt (T5), und einen Kern, der mit diesem Wheel startet (T4) und dessen Zustand ein Update überlebt (T3). Anmeldung, ISAA und OBS (T9–T11) sind reine Kern-Themen und ohne Tauri prüfbar.

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

Workers danach stoppen. Die Installation für T9–T11 stehen lassen.

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

### T9 — Lokale Auto-Anmeldung (F8)

**Dateien:** `toolboxv2/utils/workers/session.py`, `toolboxv2/utils/workers/server_worker.py`

**Schritt 1 — `SessionManager` (session.py ab Zeile 524):**

- In `__init__`: `self._local_trust_user = None`.
- Neue Methode `enable_local_trust(self, user: dict) -> None`: speichert eine Kopie von `user`.
- Neue Methode `_local_trust_session(self, environ, headers)`: gibt `None` zurück, außer **alle** Bedingungen gelten:
  1. `self._local_trust_user` ist gesetzt.
  2. `environ.get("REMOTE_ADDR")` ist `127.0.0.1` oder `::1`.
  3. Keiner der Header `X-Forwarded-For`, `X-Real-Ip`, `Forwarded` ist vorhanden (Groß-/Kleinschreibung egal).
  4. Der Host aus `HTTP_HOST` (ohne Port) ist `127.0.0.1`, `localhost` oder `[::1]`.
  Dann eine `SessionData` zurückgeben mit `user_id`, `user_name` (aus `username`), `level` aus dem Nutzer, `session_id="local-trust"`, `validated=True`, `anonymous=False`, `exp = jetzt + cookie_max_age`.
- In `get_session_from_request` (Zeile 685) **und** `get_session_from_request_sync` (Zeile 738): direkt vor „4. Return anonymous" `_local_trust_session` aufrufen und dessen Ergebnis zurückgeben, wenn es nicht `None` ist.

**Schritt 2 — Bedingung in `server_worker.py:1685–1690`:** Die Tauri-Installation setzt `environment: production`. Die Bedingung ändern auf: kein `TOOLBOXV2_REMOTE_BASE` **und** Profil ist `local` oder `developer`. Die Prüfung auf `environment` entfällt. Ein fehlendes Profil (`None`) zählt als nicht erlaubt.

**Schritt 3 — neuer Test `toolboxv2/tests/test_core/test_local_trust.py`:**

| Fall | Erwartet |
|---|---|
| Trust aktiv, `REMOTE_ADDR=127.0.0.1`, `HTTP_HOST=127.0.0.1:8467` | Session des Admins, `anonymous False` |
| Trust aktiv, `REMOTE_ADDR=192.168.1.5` | anonym |
| Trust aktiv, Loopback, Header `X-Forwarded-For: 1.2.3.4` | anonym |
| Trust aktiv, Loopback, `HTTP_HOST=evil.example` | anonym |
| Trust nicht aktiv, Loopback | anonym |

**Gate:** Standard-Gate + neuer Test `OK`. Dann mit der Installation aus T6 (vorher `installer.ps1` mit dem neu gebauten Wheel erneut ausführen):

```powershell
& "$env:LOCALAPPDATA\toolboxv2\bin\tb.exe" workers start
curl http://127.0.0.1:8467/api_user_data
```

Erwartet: Antwort enthält den lokalen Admin (kein Fehler „nicht angemeldet"); im Worker-Log steht **nicht** mehr `Loopback trust not enabled`.

**Wenn rot:** Liefert `/api_user_data` trotz grünem Unit-Test „nicht angemeldet", dann im Log prüfen, ob `ensure_local_admin` einen Fehler wirft, und diesen Fehler melden. Stoppen.

**Nicht tun:** Keine Anmeldung für Nicht-Loopback-Anfragen freigeben. CSRF- und CORS-Regeln nicht anfassen.

---

### T10 — ISAA im Owner abnehmen

**Voraussetzung (H0, Owner):** einen LLM-API-Key bereitstellen und über `tb -init onboard` (Frage 2) eintragen.

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

**Gate:** `cargo test --lib` bestanden. Manuell: `tb_version` in `%LOCALAPPDATA%\toolboxv2\install.manifest` von Hand auf `0.0.1` setzen, App starten → Log `install.log` zeigt einen Update-Lauf, danach steht dort wieder `0.1.29`; `curl http://127.0.0.1:8467/api/wizard/state` zeigt weiter `"profile":"local"`, `first_run:false`.

**Wenn rot:** Scheitert das Update mit „Datei wird verwendet", läuft noch ein `tb`-Prozess — die Wartezeit in Punkt 1 prüfen, nicht `--force` oder Prozess-Kill einbauen.

---

### T14 — Gesamtabnahme

**Schritte**

1. Alles committen und pushen. Der Owner stößt `nightly.yml` an (**H2**).
2. Auf einem Windows-Konto ohne `%LOCALAPPDATA%\toolboxv2` das Nightly-Setup installieren.
3. Punkte Z1–Z6 aus Abschnitt 0 durchgehen.
4. Der Owner stößt `nightly.yml` ein zweites Mal an (**H3**). App neu starten → Z7: Update wird installiert, App startet neu; `install.manifest` zeigt die neue Version; ISAA-Chatverlauf und Profil sind erhalten.

**Gate:** Tabelle Z1–Z7 vollständig mit „ok" oder genauer Fehlermeldung ausgefüllt an den Owner melden.

**Wenn rot**

| Symptom | Zurück zu |
|---|---|
| Installer-Fenster meldet Fehler | `logs\install.log` neben der App-EXE lesen → T5/T6 |
| Fehlerseite „installed-stopped-timeout" | `%LOCALAPPDATA%\ToolBoxV2\app\logs\workers_start.log` lesen → T4/T8 |
| Onboarding erscheint erneut | T3 |
| Login-Formular erscheint | T9 |
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

---

## 6. Bewusst offen (nicht anfassen)

Handoff B2, B3, B4, B5, B6, B7, B9; V5; Punkt E; Abschnitt 4 des Handoffs komplett. Dazu neu: `launch.rs` kennt unter Linux/macOS den Installationspfad nicht (`tb_candidates` deckt nur Windows ab); stabiler Update-Kanal über `release.yml`; Entwicklerprofil in der Tauri-App (Source-Installation braucht Git).
