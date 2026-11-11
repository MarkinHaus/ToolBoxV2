

# Agent Report 2026-11 — R1: Secret-Autogen beim ersten Start (11.11.2026)

## Status: CODE FIX + INSTALLATION VALIDIERT (3/3 Teilbelege, Hash-Vergleich nach Server-Stopp ausstehend)

## Regelwerk-Kern (userpaste_1794353822)
Kein Secret in production mehr erzwingen. Stattdessen: beim ersten Start automatisch erzeugen,
persistieren, Env setzen. Persistenz-Ort: _state_dir.user_state_dir() (NIE site-packages).
Leere TB_JWT_SECRET=/TB_COOKIE_SECRET=-Zeilen aus Installer-.env duerfen Persistenz nicht blockieren.
Workflow: Dateien lesen VOR Aenderung -> Ist messen -> TDD rot -> kleinste Aenderung -> gruen ->
Gate -> echte Installation (3 Kriterien) -> sauberer Commit -> Push auf claude/wizardly-cannon-o944j4.

## Fix (config.py, init_onboarding.py)
1. utils/workers/config.py (production-Zweig): ValueError('TB_COOKIE_SECRET must be set...') entfernt ->
   lazy `from toolboxv2.init_onboarding import ensure_secret`; cookie_secret = ensure_secret().
   env_mapping laeuft weiterhin vorher (TB_COOKIE_SECRET="" => falsy => Autogen greift).
2. init_onboarding._writable_env_path: bei is_packaged_install() wird user_state_dir()/.env als
   Kandidat vor tb_root_dir.parent/.env eingereiht (site-packages-Schutz).
3. init_onboarding._append_env: bestehende LEERE Zeilen (TB_JWT_SECRET=) werden in-place gefuellt
   statt uebersprungen (sonst Re-Key bei jedem Restart). Nicht-Secret-Zeilen bleiben unberuehrt.
4. ensure_secret: os.environ['TB_COOKIE_SECRET'] nur setzen, wenn leer ("" zaehlt als unset).

## Tests (TDD, unittest, 4/4 gruen)
- tests/test_core/test_worker_secret_autogen.py: production load_config ohne Secret (leere Env-Zeilen)
  -> 128-Zeichen-Secret, zwei Aufrufe identisch, Persistenz in TB_DATA_DIR/.env; packaged-Pfad-Test
  (tb_root_dir+is_packaged_install gemockt) -> _writable_env_path == user_state_dir/.env.
- tests/test_core/test_cp1252_env.py: cp1252-kodierte .env (Em-Dash-Header, leere Secret-Zeilen)
  -> ensure_secret(persist=True) crasht nicht (surrogateescape round-trip), Fremd-Bytes ueberleben.
- Standard-Gate (8 Module, 62 Tests): vorher OK skipped=9, nachher OK skipped=9 (+4 neue, 0 Regressions).

## Bei der echten Installation gefundene und gefixte Blocker (alles genau verifiziert)
- IB1 installer.ps1:616 (uv tool list 2>null): PS5.1 macht stderr-Capture bei DEBUG-Ausgabe (uv,
  maschinen-RUST_LOG=debug) ein terminierendes NativeCommandError (EAP=Stop, Zeile 31) -> Install
  brach nach erfolgreichem Setup ab. Fix: stderr-sicheres Capturing (2>&1 | ForEach-Object { "$_" })
  an 616 und 683. docker-Pfad-Zeilen (220/803-806) nicht betroffen (kein uv-Scope).
- IB2 installer.ps1:930 (tb -init 2>&1 | Out-String) und 884 (tb access 2>null): natives stderr-
  Routing liest stderr der Child-Procs -> EAP=Stop NativeCommandError. Fix: cmd /c ... 2>&1 Merge auf
  native Ebene, PS sieht nur Strings.
- RT1 .env-Encoding: installer.ps1 schrieb Header mit Em-Dash (U+2013); PS5.1 Set-Content ohne
  -Encoding = ANSI -> dotenv liest utf-8 strict -> UnicodeDecodeError bereits beim tb-Import
  (crypt/mods/CloudM/auth/config.py:21 load_dotenv) => tb -init/Start crasht. Fix: Header ASCII-only
  + init_onboarding liest/schreibt .env mit errors=surrogateescape (round-trip-sicher, robust fuer
  alle Installationsvarianten).
- Wheel-Rebuild nach RT1 (init_onboarding ist im Wheel), uv cache clean + --no-cache reinstall
  (Cache-Falle: file-URL-Cache-Key, --force allein reicht nicht).

## Echte-Installation-Evidenz (Run8: installer.ps1 EXIT=0, tb -init local => {"ok": true})
- K1: tb http_worker -H 127.0.0.1 -p 8791 -w r1test => /health 200
  {"status":"healthy","worker_id":"r1test","pid":33388,...} deutlich unter 120s.
- Persistenz: 128-Zeichen-Hex-Secret in <install>/.env (TB_JWT_SECRET="..."), h1(SHA256)=FB9918804FE23BEC.
- K2 (Secret-Stabilitaet ueber Neustarts): 3 unabhaengige Prozesse (Python direct, tb -init, HTTP-Server)
  ohne Env-Vorgaben: identischer Secret-Head f2011eaba47c bzw. h1 => kein Re-Key. Wiederaufsetzen des
  Servers nach hartem Stopp steht noch aus (Harness-Schutz blockiert Prozess-Stop; Testserver haelt
  Port 8791 als Waisenprozess). Wiederholung nach Neustart: Server starten -> .env-Hash == h1.
- K3: keine .env unter site-packages (0 Treffer, recurse).

## Artefakte
- Geaendert: toolboxv2/utils/workers/config.py, toolboxv2/init_onboarding.py, installer.ps1
- Neu: tests/test_core/test_worker_secret_autogen.py, tests/test_core/test_cp1252_env.py
- Logs: %TEMP%/r1_gate_baseline.txt, r1_gate_after.txt, r1_install*.txt, r1_server1.log
