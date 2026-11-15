# WP2 — CI-Fehleranalyse: Ergebnis

> Stand 2026-11-15. Basis `master` @ `ebed4d6`, Lauf `Build & Test` 43742117802 (alle 12 `py-check`-Jobs).
> Zweiter Lauf zum Vergleich: 43741108838 auf `91d984a` (gleicher Dateibaum, 10 Jobs ausgewertet).
> Analysephase: kein Produktcode, kein Test, kein Workflow geändert. Alle Probe-Fixes liefen in einer getrennten Arbeitskopie und sind verworfen.
> Auftrag: `WP2_ci_fehleranalyse_2026-11.md`. Abweichung vom Auftrag: Branch ist `claude/keen-goldberg-o59mt9` (Vorgabe dieser Sitzung), nicht `claude/wizardly-cannon-o944j4`.

## 1. Das Wichtigste

1. **H0 ist bestätigt. Das CI-Signal ist in beide Richtungen falsch.** In *allen* 12 Jobs schlägt pytest mit 509–556 Fehlern fehl. Danach läuft `unittest discover`, findet 0 Tests und liefert bis Python 3.11 Exit-Code 0 (Job grün) und ab 3.12 Exit-Code 5 (Job rot). 3.10/3.11 sind also nicht geprüft, und 3.12/3.13 sind nicht wegen 3.12 rot. → [K-ENV-01](cards/K-ENV-01.md), [FIX-01](fixes/FIX-01.md).
2. **Es gibt kein „3.12-Problem“.** 483 der 593 fehlschlagenden Test-IDs schlagen in allen 12 Jobs gleich fehl. Nur 30 IDs hängen an einer Python-Version (alle an 3.10).
3. **Wenige Ursachen erklären die Masse.** Sieben Ursachen decken 345 IDs (58 %): geschlossene Event-Schleife (160), `mcp` 2.x (45), `vfs_view` jetzt async (46), falsches Patch-Ziel minicli (26), 3.10-`mock.patch` (28), `/fake`-Datenverzeichnis (20), `waitress` fehlt (8). Alle sieben sind getestet.
4. **Zwei echte Produktfehler** wurden gefunden und per Probe-Fix bestätigt: [K-BUG-02](cards/K-BUG-02.md) (`add_data` lehnt jeden Text bis 6000 Zeichen ab) und [K-BUG-01](cards/K-BUG-01.md) (Doku-Scanner schließt ganze Projekte aus, die unter `/tmp`, `build`, `dist` … liegen). Ein dritter ist wahrscheinlich: [K-OS-02](cards/K-OS-02.md) (PowerShell-Prompt in der Shell-Ausgabe unter Windows).
5. **Außerhalb der Tests, aber dringend:** Auf `master` schlagen alle drei Tauri-Builds beim Signieren fehl (`Invalid symbol 10, offset 97.`), nachdem das Release `nightly` schon gelöscht wurde. O1 und O2 sind damit blockiert. → [K-ENV-07](cards/K-ENV-07.md), [FIX-05](fixes/FIX-05.md) (Owner).
6. **O5** (`…grandparent…` nur in der CI rot) ist geklärt: kein Plattformproblem, sondern die nicht editierbare Installation. → [K-ENV-04](cards/K-ENV-04.md). **O4** (Nuitka fehlt in `release.yml`) → [K-ENV-06](cards/K-ENV-06.md).

## 2. Zahlen je Job (Lauf auf `ebed4d6`)

| Job | pytest-Zeile | IDs in raw/ | Fallback `unittest discover` | Job-Ergebnis |
|---|---|---|---|---|
| macos-3.10 | 540 failed, 4066 passed, 240 skipped, 40 warnings, 16 errors, 304 subtests passed | 539 | Ran 0 tests → OK, Exit 0 | **grün** |
| macos-3.11 | 509 failed, 4114 passed, 221 skipped, 69 warnings, 16 errors, 454 subtests passed | 510 | Ran 0 tests → OK, Exit 0 | **grün** |
| macos-3.12 | 512 failed, 4114 passed, 221 skipped, 69 warnings, 16 errors, 451 subtests passed | 511 | Ran 0 tests → Exit 5 | rot |
| macos-3.13 | 511 failed, 4115 passed, 221 skipped, 61 warnings, 16 errors, 451 subtests passed | 510 | Ran 0 tests → Exit 5 | rot |
| ubuntu-3.10 | 556 failed, 4052 passed, 238 skipped, 38 warnings, 16 errors, 304 subtests passed | 555 | Ran 0 tests → OK, Exit 0 | **grün** |
| ubuntu-3.11 | 529 failed, 4103 passed, 219 skipped, 52 warnings, 16 errors, 447 subtests passed | 524 | Ran 0 tests → OK, Exit 0 | **grün** |
| ubuntu-3.12 | 529 failed, 4104 passed, 219 skipped, 71 warnings, 16 errors, 446 subtests passed | 523 | Ran 0 tests → Exit 5 | rot |
| ubuntu-3.13 | 529 failed, 4105 passed, 219 skipped, 61 warnings, 16 errors, 445 subtests passed | 522 | Ran 0 tests → Exit 5 | rot |
| windows-3.10 | 555 failed, 4041 passed, 256 skipped, 50 warnings, 10 errors, 313 subtests passed | 548 | Ran 0 tests → OK, Exit 0 | **grün** |
| windows-3.11 | 528 failed, 4087 passed, 237 skipped, 54 warnings, 10 errors, 461 subtests passed | 521 | Ran 0 tests → OK, Exit 0 | **grün** |
| windows-3.12 | 531 failed, 4086 passed, 237 skipped, 57 warnings, 10 errors, 459 subtests passed | 523 | Ran 0 tests → Exit 1 | rot |
| windows-3.13 | 530 failed, 4087 passed, 237 skipped, 72 warnings, 10 errors, 459 subtests passed | 522 | Ran 0 tests → Exit 1 | rot |

„IDs in raw/“ zählt die Zeilen `FAILED`/`ERROR` der pytest-Kurzfassung. Die pytest-Zeile zählt zusätzlich fehlgeschlagene Subtests einzeln, darum ist `failed + errors` etwas größer.

**Zählung (A3):** Summe aller Zeilen in `raw/*.txt` = 6308. Ohne Doppelte = **593** eindeutige IDs = Zeilen in `catalog.csv`. Jede ID hat genau eine Karte.

### Wo eine ID fehlschlägt

| Muster | IDs |
|---|---|
| in allen 12 Jobs | 482 |
| nur Python 3.10 (alle OS) | 30 |
| nur windows (alle Versionen) | 29 |
| sonstige Muster (teils sporadisch) | 19 |
| ubuntu + macOS, nicht Windows | 19 |
| nur ubuntu (alle Versionen) | 13 |
| nur macos (alle Versionen) | 1 |

Zwischen den beiden Läufen desselben Baums unterscheiden sich 18 IDs (`raw/sporadic.txt`); 3 davon waren nur im zweiten Lauf rot und stehen deshalb nicht im Katalog (alle zu [K-FLAKY-01](cards/K-FLAKY-01.md)): `test_external_create_becomes_visible`, `test_call_counts_positive`, `test_caller_edges_for_compute`.

## 3. Kategorien

| Kategorie | IDs | Karten |
|---|---|---|
| K-API | 191 | K-API-02, K-API-03, K-API-04, K-API-06, K-API-07, K-API-08, K-API-09, K-API-10, K-API-11, K-API-12, K-API-13, K-API-14, K-API-15, K-API-16, K-API-17, K-API-18, K-API-19, K-API-20, K-API-21, K-API-22 |
| K-ASYNC | 160 | K-ASYNC-01 |
| K-ENV | 80 | K-ENV-01, K-ENV-02, K-ENV-03, K-ENV-04, K-ENV-05, K-ENV-06, K-ENV-07 |
| K-STATE | 38 | K-STATE-01, K-STATE-02, K-STATE-03, K-STATE-04 |
| K-MISS | 36 | K-MISS-01, K-MISS-02, K-MISS-03 |
| K-OS | 33 | K-OS-01, K-OS-02, K-OS-03, K-OS-04, K-OS-05, K-OS-06 |
| K-PY | 30 | K-PY-01, K-PY-02 |
| K-BUG | 18 | K-BUG-01, K-BUG-02, K-BUG-03 |
| K-FLAKY | 7 | K-FLAKY-01 |

## 4. Arbeitspakete für die ISAA-Instanzen

Jede Karte gehört zu genau einem Paket. Ein Paket kann einer eigenen Instanz gegeben werden; die Fix-Aufträge darin sind voneinander unabhängig, außer wo „Voraussetzung“ etwas anderes sagt. In `catalog.csv` steht das Paket in der Spalte `paket`.

| Paket | Bedeutung | braucht dich? |
|---|---|---|
| P0-SIGNAL | CI-Workflow, Abhängigkeiten | nur noch das Signing-Secret (FIX-05) |
| P1-AUTO-MINI | minimaler Fix, getestet | nein |
| P2-AUTO-MEDIUM | mittlerer Fix, Muster getestet | eine Rückfrage (MCP-Port 8791) |
| P3-VERALTET | Test prüft alte Schnittstelle | je Karte bestätigen (Entscheidung 4) |
| P4-ENTSCHEIDUNG | zwei gültige Richtungen | ja, je Karte |
| P5-ARCH | Aufbau | Brainstorming |
| P6-PLATTFORM | nur Windows/macOS/Linux-Runner/3.10 | Windows-Karten: lokal bei dir prüfbar |
| P7-FLAKY | sporadisch | nein |

### P0-SIGNAL – CI-Signal und Installation (54 IDs)

Zuerst. Ohne diese Fixes misst jeder spätere Lauf falsch. Wer: Planer / kleines Modell, Workflow- und pyproject-Änderungen.

| Karte | Fix | Typ | IDs | Status | Ursache |
|---|---|---|---|---|---|
| [K-ENV-01](cards/K-ENV-01.md) | [FIX-01](fixes/FIX-01.md) | A | 0 | getestet | Der Fallback `|| python -m unittest discover` entscheidet das Job-Ergebnis; er findet 0 Tests. Bis 3.11 ist das Exit-Code 0 (grün), ab 3.12 Exit-Code  |
| [K-ENV-04](cards/K-ENV-04.md) | [FIX-02](fixes/FIX-02.md) | A | 1 | getestet | `toolboxv2` ist in der CI nicht editierbar installiert; ein Unterprozess, der nicht im Checkout startet, importiert die Kopie aus `site-packages`, in  |
| [K-ENV-02](cards/K-ENV-02.md) | [FIX-03](fixes/FIX-03.md) | A | 45 | getestet | `mcp` ist ungepinnt (`mcp>=1.6.0`); installiert wird 2.x. Dort gibt es die Dekorator-API (`Server.list_tools()` …) und `mcp.server.fastmcp` nicht mehr |
| [K-ENV-03](cards/K-ENV-03.md) | [FIX-04](fixes/FIX-04.md) | A | 8 | getestet | `fast_tb.serve()` importiert `waitress`; das Paket steht nur im Extra `web`, die CI installiert ohne Extras. |
| [K-ENV-07](cards/K-ENV-07.md) | [FIX-05](fixes/FIX-05.md) | B | 0 | ungetestet | Auf `master` schlägt `tauri build` beim Signieren fehl: das Secret `TAURI_PRIVATE_KEY` lässt sich nicht als Base64 lesen. |
| [K-ENV-06](cards/K-ENV-06.md) | [FIX-06](fixes/FIX-06.md) | A | 0 | ungetestet | `release.yml`: die Jobs `build-windows-x64` und `build-macos-arm64` rufen `python -m nuitka` auf, installieren Nuitka aber nicht (O4). |

### P1-AUTO-MINI – Minimaler Fix, automatisch (275 IDs)

Getestete, örtlich begrenzte Änderung (1–3 Dateien, wenige Zeilen). Wer: ISAA-Instanz ohne Rückfrage.

| Karte | Fix | Typ | IDs | Status | Ursache |
|---|---|---|---|---|---|
| [K-ASYNC-01](cards/K-ASYNC-01.md) | [FIX-07](fixes/FIX-07.md) | A | 160 | getestet | Synchrone Tests rufen `asyncio.get_event_loop().run_until_complete(…)` auf; ein früherer Test hat die Schleife geschlossen oder abgemeldet. |
| [K-STATE-01](cards/K-STATE-01.md) | [FIX-08](fixes/FIX-08.md) | A | 20 | getestet | `test_session.py` setzt `TB_DATA_DIR=/fake/dir`; `Session()` legt dort wirklich Verzeichnisse an. Auf dem Runner ist das verboten, `asyncSetUp` bricht |
| [K-MISS-01](cards/K-MISS-01.md) | [FIX-09](fixes/FIX-09.md) | A | 26 | getestet | `test_minicli_macros.py` patcht `toolboxv2.flows.minicli.BlobFile`; das Modul heißt `toolboxv2.flows.mini.minicli`. |
| [K-PY-01](cards/K-PY-01.md) | [FIX-10](fixes/FIX-10.md) | A | 28 | getestet | `test_config_wizard.py` lädt `config_wizard` per Dateipfad in `sys.modules`; unter Python 3.10 löst `mock.patch("toolboxv2.utils.clis.config_wizard.…" |
| [K-API-03](cards/K-API-03.md) | [FIX-11](fixes/FIX-11.md) | A | 14 | getestet | `ExecutionEngine` liest `self.agent.obs`; die Testattrappe `MockAgent` hat das Attribut nicht. |
| [K-MISS-02](cards/K-MISS-02.md) | [FIX-12](fixes/FIX-12.md) | A | 9 | getestet | Tests patchen bzw. importieren `llm_router.…` als Top-Level-Modul; es heißt `toolboxv2.mods.isaa.base.llm_router.…`. |
| [K-BUG-01](cards/K-BUG-01.md) | [FIX-13](fixes/FIX-13.md) | A | 11 | getestet | `FileScanner._should_include` prüft die Ausschlussliste (`tmp`, `temp`, `build`, `dist`, `target`, …) gegen den **absoluten** Pfad. Liegt das Projekt  |
| [K-BUG-02](cards/K-BUG-02.md) | [FIX-14](fixes/FIX-14.md) | A | 5 | getestet | In `add_data` wurde die Längenbegrenzung mitten in eine `if/elif`-Kette gesetzt. Jeder String bis 6000 Zeichen fällt in `else: raise ValueError("Unsup |
| [K-MISS-03](cards/K-MISS-03.md) | [FIX-15](fixes/FIX-15.md) | A | 1 | getestet | `test_port_unification.py` baut den Pfad zu `utils/workers/fast_tb.py` vom Repo-Wurzelverzeichnis statt vom Paketordner `toolboxv2/`. |
| [K-API-17](cards/K-API-17.md) | [FIX-16](fixes/FIX-16.md) | A | 1 | getestet | Test sucht im Quelltext nach `_fetch_tray_state`; die Funktion heißt `fetch_tray_state`. |

### P2-AUTO-MEDIUM – Mittlerer Fix, automatisch (48 IDs)

Mechanisch, aber mehrere Dateien oder viele Stellen; Muster ist getestet. Wer: ISAA-Instanz ohne Rückfrage.

| Karte | Fix | Typ | IDs | Status | Ursache |
|---|---|---|---|---|---|
| [K-API-02](cards/K-API-02.md) | [FIX-17](fixes/FIX-17.md) | A | 46 | getestet | `vfs_view` ist jetzt eine Koroutine und `vfs_shell` hat die Signatur `(reason, command)`. Die Tests rufen beide noch synchron bzw. ohne `reason` auf – |
| [K-API-15](cards/K-API-15.md) | [FIX-18](fixes/FIX-18.md) | B | 2 | ungetestet | Zwei Tests erwarten alte Vorgabeports: WebSocket 8100 (jetzt 8468) und MCP 8765 (jetzt 8791). |

### P3-VERALTET – Test veraltet (Schnittstelle absichtlich geändert) (107 IDs)

Der Test prüft eine alte Schnittstelle. Owner bestätigt nur: Änderung war gewollt. Dann Tests auf die neue Schnittstelle umschreiben (nicht löschen, nicht abschwächen). Wer: ISAA-Instanz nach einem Ja/Nein des Owners je Karte.

| Karte | Fix | Typ | IDs | Status | Ursache |
|---|---|---|---|---|---|
| [K-API-21](cards/K-API-21.md) | [FIX-26](fixes/FIX-26.md) | B | 18 | ungetestet | Tests des LLM-Pfads von `FlowAgent` mocken die alte Schnittstelle (`llm_handler.completion_with_rate_limiting`, litellm-Antworten); der Agent geht heu |
| [K-API-04](cards/K-API-04.md) | [FIX-27](fixes/FIX-27.md) | B | 15 | ungetestet | Tests der `ExecutionEngine` prüfen entfernte oder umbenannte Teile: `AutoFocusTracker`, `ctx.auto_focus`, `_build_system_prompt`, alte Zählwerte, alte |
| [K-API-08](cards/K-API-08.md) | [FIX-28](fixes/FIX-28.md) | B | 20 | ungetestet | `test_icli_overlay.py`/`test_icli_logic.py` prüfen eine alte `TaskOverlay`/`IterView`-Struktur (`_focus`, `_left_items`, `_effective_view`, `_right_sc |
| [K-API-07](cards/K-API-07.md) | [FIX-29](fixes/FIX-29.md) | B | 19 | ungetestet | `test_omni.py` nutzt eine alte `OmniSession`-Schnittstelle (`fallback=…`, `FakeAgent.a_stream`, `CloudOmniBackend`). |
| [K-API-09](cards/K-API-09.md) | [FIX-30](fixes/FIX-30.md) | B | 8 | ungetestet | Tests erwarten die Systemdatei `/system_context.md` im VFS; heute gibt es nur `/vfs_guide.md`. |
| [K-API-10](cards/K-API-10.md) | [FIX-31](fixes/FIX-31.md) | B | 9 | ungetestet | `test_vfs_pooling.py` ruft `MountWatcher._scan_disk` und `_dispatch_burst` auf; beide gibt es nicht mehr. |
| [K-API-12](cards/K-API-12.md) | [FIX-32](fixes/FIX-32.md) | B | 6 | ungetestet | Provider-Tests nutzen alte `OnboardingResult`-Felder (`payg_providers`, `coding_providers`) und ein altes `.env`-Schreibformat (ohne Anführungszeichen |
| [K-API-20](cards/K-API-20.md) | [FIX-33](fixes/FIX-33.md) | B | 3 | ungetestet | `test_web/test_web.py` enthält Hilfsfunktionen mit dem Namen `test_…(tf, …)`; pytest sammelt sie als Tests und findet keine Fixture `tf`. |
| [K-API-22](cards/K-API-22.md) | [FIX-34](fixes/FIX-34.md) | B | 9 | ungetestet | Einzelfälle in ISAA-Tests, je eine eigene kleine Abweichung zwischen Test und heutigem Verhalten. |

### P4-ENTSCHEIDUNG – Braucht Entscheidung des Owners (15 IDs)

Fix ist klar, aber es gibt zwei gültige Richtungen (Produkt ändern oder Test ändern). Wer: ISAA-Instanz nach Antwort.

| Karte | Fix | Typ | IDs | Status | Ursache |
|---|---|---|---|---|---|
| [K-API-13](cards/K-API-13.md) | [FIX-19](fixes/FIX-19.md) | A | 3 | getestet | `Session._get_blob_name()` gibt immer `"user/_session.json"` zurück; der bereinigte Nutzername wird berechnet, aber nicht verwendet. Alle Nutzer eines |
| [K-API-14](cards/K-API-14.md) | [FIX-20](fixes/FIX-20.md) | B | 3 | ungetestet | Restliche Abweichungen in `test_session.py`: erwarteter Aufruf ohne `Authorization`-Header; Sitzung löschen erwartet `delete_blob`, das Produkt schrei |
| [K-BUG-03](cards/K-BUG-03.md) | [FIX-21](fixes/FIX-21.md) | B | 2 | ungetestet | `brodcast/client.py` ruft `int(os.getenv("TOOLBOXV2_BASE_PORT"))` ohne Vorgabewert auf; ohne die Variable stirbt der Client-Thread. |
| [K-API-16](cards/K-API-16.md) | [FIX-22](fixes/FIX-22.md) | B | 2 | ungetestet | `test_cli_service_manager.py` erwartet `workers` in der Kategorie `core` und mindestens 4 `core`-Dienste; heute ist `workers` `infrastructure`, mehrer |
| [K-API-18](cards/K-API-18.md) | [FIX-23](fixes/FIX-23.md) | B | 1 | ungetestet | `test_docker_build.py` verlangt für jedes Feature in `FEATURE_MAP` ein `ARG` im Dockerfile; `FEATURE_CLI` fehlt dort. |
| [K-API-19](cards/K-API-19.md) | [FIX-24](fixes/FIX-24.md) | B | 1 | ungetestet | `test_fast_tb.py::test_format_html_string` erwartet das rohe `<h1>Hi</h1>`; das Produkt liefert die Seite in der Hülle (`<html data-style="paper"> …`, |
| [K-API-06](cards/K-API-06.md) | [FIX-25](fixes/FIX-25.md) | B | 3 | ungetestet | `test_llm_router.py::TestModelInfo` erwartet alte Kontextgrenzen (1 000 000 / 128 000); das Produkt liefert 2 000 000 / 262 144. |

### P5-ARCH – Architektur (48 IDs)

Kein Einzel-Fix. Notiz unter architecture/. Wer: Owner + Planer, Brainstorming.

| Karte | Fix | Typ | IDs | Status | Ursache |
|---|---|---|---|---|---|
| [K-ENV-05](cards/K-ENV-05.md) | ARCH-03 | C | 26 | ungetestet | Tests brauchen optionale Pakete (`pyautogui`, `torch`, `fastapi`, `litellm`), die die CI nicht installiert. |
| [K-STATE-02](cards/K-STATE-02.md) | ARCH-01 | C | 14 | ungetestet | Testmodule ersetzen beim Import Einträge in `sys.modules` dauerhaft durch `MagicMock`; spätere Tests bekommen die Attrappen statt der echten Module. |
| [K-API-11](cards/K-API-11.md) | ARCH-01 | C | 8 | ungetestet | `FlowAgent.__init__` liest `amd.web_config.enable_web`; die Attrappe `MockAgentModelData` in `conftest2.py` hat `web_config` nicht – und `conftest2.py |

### P6-PLATTFORM – Plattform- und versionsabhängig (39 IDs)

Fehler treten nur unter Windows, macOS, Linux-Runner oder Python 3.10 auf. In dieser Analyse nicht reproduzierbar, Hypothese ungetestet. Wer: ISAA lokal auf Windows (Windows-Karten), sonst CI-Lauf als Beleg.

| Karte | Fix | Typ | IDs | Status | Ursache |
|---|---|---|---|---|---|
| [K-OS-01](cards/K-OS-01.md) | [FIX-35](fixes/FIX-35.md) | A | 14 | ungetestet | Unter Windows schlagen die Tests des Installer-Protokolls (`installer.ps1`) und von `verify_install_windows.ps1` fehl. |
| [K-OS-02](cards/K-OS-02.md) | [FIX-36](fixes/FIX-36.md) | A | 4 | ungetestet | Unter Windows enthält die Ausgabe des Shell-Werkzeugs den PowerShell-Prompt (`hiPS D:\a\ToolBoxV2\ToolBoxV2>`). |
| [K-OS-06](cards/K-OS-06.md) | [FIX-37](fixes/FIX-37.md) | A | 6 | ungetestet | Unter Windows schlagen VFS-Verzeichnisoperationen (`rmdir`, `rm -rf`, `mv` in ein Verzeichnis) und ein Rechte-Test (`'666' != '600'`) fehl. |
| [K-OS-03](cards/K-OS-03.md) | [FIX-38](fixes/FIX-38.md) | A | 4 | ungetestet | Tests vergleichen Temp-Pfade als Text; das Betriebssystem liefert eine andere Schreibweise desselben Pfads (macOS `/var/…` ↔ `/private/var/…`, Windows |
| [K-OS-04](cards/K-OS-04.md) | [FIX-39](fixes/FIX-39.md) | B | 2 | ungetestet | `test_job_os_scheduler.py::TestRealFilesystemIntegration` schreibt echte crontab-Einträge und liest sie zurück; auf dem Ubuntu-Runner kommen sie nicht |
| [K-OS-05](cards/K-OS-05.md) | [FIX-40](fixes/FIX-40.md) | B | 3 | ungetestet | `test_depmanager.py` ruft echte Paketmanager auf (winget, choco; lokal apt, nix) und installiert Pakete. |
| [K-PY-02](cards/K-PY-02.md) | [FIX-41](fixes/FIX-41.md) | A | 2 | ungetestet | Zwei Tests laufen nur unter 3.10 nicht: `asyncio.timeout` gibt es erst ab 3.11; `'NoneType' object has no attribute 'get_memory'`. |
| [K-STATE-03](cards/K-STATE-03.md) | [FIX-42](fixes/FIX-42.md) | A | 1 | ungetestet | `test_state_system.py::TestTbState::test_get_state_from_app` schreibt nach `/root/toolboxv2/features`. |
| [K-STATE-04](cards/K-STATE-04.md) | [FIX-43](fixes/FIX-43.md) | A | 3 | ungetestet | `test_job_execution_state.py`: `ValueError: Could not find isaa in ['isaa'] …` – das Modul `isaa` wird im Test über `get_mod` gesucht und nicht gelade |

### P7-FLAKY – Sporadisch (7 IDs)

Zeit- oder zufallsabhängig. Erst stabilisieren, dann bewerten. Wer: ISAA-Instanz, je Test einzeln.

| Karte | Fix | Typ | IDs | Status | Ursache |
|---|---|---|---|---|---|
| [K-FLAKY-01](cards/K-FLAKY-01.md) | [FIX-44](fixes/FIX-44.md) | A | 7 | ungetestet | Zeit- oder zufallsabhängige Tests: Datei-Watcher mit festen Wartezeiten, Profiler-Zählwerte, Backup-Aufräumen nach Zeitstempel, Vektorsuche, Audio-Rei |

## 5. Reihenfolge des Fix-Plans

Die Nummern der Fix-Aufträge sind die vorgeschlagene Reihenfolge.

1. **Signal zuerst (FIX-01 bis FIX-06).** FIX-01 entfernt den Fallback; danach sind alle 12 Jobs rot, und das ist richtig. FIX-02 (editierbare Installation) sorgt dafür, dass genau ein Stand von `toolboxv2` getestet wird. FIX-03 (`mcp<2`) und FIX-04 (`waitress`) beseitigen 53 IDs, die nichts mit dem Code zu tun haben. FIX-05 ist deine Aufgabe (Secret) und blockiert O1/O2. Ohne diese Schritte misst jeder spätere Fix gegen eine CI, die grün zeigt, obwohl 500 Tests rot sind.
2. **Breite, getestete Ursachen (FIX-07 bis FIX-16, Paket P1).** FIX-07 allein nimmt 160 IDs weg. Zusammen mit FIX-08 bis FIX-16 sind es 275. Jeder Auftrag ist eine Datei und wenige Zeilen.
3. **Mittlere (FIX-17, FIX-18).** 48 IDs.
4. **Produktpfade mit Entscheidung (FIX-19 bis FIX-25, Paket P4).** Sitzung/Login (K-API-13, K-API-14) zuerst, weil WP1 daran hängt.
5. **Veraltete Tests (FIX-26 bis FIX-34, Paket P3).** Fast nur ISAA; Reihenfolge nach Größe.
6. **Plattform (FIX-35 bis FIX-43, Paket P6).** FIX-35 (`installer.ps1` unter Windows, 14 IDs) ist ein **Produktpfad von WP1** und sollte trotz der späten Nummer früh an eine Instanz auf deinem Windows-Rechner gehen; es ist nur deshalb hinten, weil die Hypothese ungetestet ist.
7. **Sporadische (FIX-44).**

Typ-C-Ursachen blockieren nichts davon: der Typ-A-Teil von ARCH-01 ist FIX-07/FIX-08, von ARCH-02 FIX-02, von ARCH-03 FIX-03/FIX-04.

## 6. Wirkung der getesteten Probe-Fixes zusammen

Ein Lauf über die 77 Testdateien, in denen mindestens eine ID rot ist (plus `test_isaa_module.py` als bekannter Verschmutzer), Python 3.12, Linux, einmal unverändert und einmal mit allen Probe-Fixes der Karten unten (Arbeitskopie, editierbar installiert, `mcp<2`, `waitress`). Bezug: die IDs der Karte, die im Job ubuntu/3.12 rot sind.

- unverändert: `518 failed, 2360 passed, 91 skipped, 137 warnings, 6 errors, 448 subtests passed`
- mit Probe-Fixes: `209 failed, 2671 passed, 91 skipped, 25 warnings, 6 errors, 446 subtests passed`

| Karte | Fix | IDs (ubuntu/3.12) | rot unverändert | rot mit Probe-Fixes |
|---|---|---|---|---|
| K-ENV-02 | FIX-03 | 45 | 45 | 0 |
| K-ENV-03 | FIX-04 | 8 | 8 | 0 |
| K-ENV-04 | FIX-02 | 1 | 1 | 0 |
| K-ASYNC-01 | FIX-07 | 160 | 160 | 0 |
| K-STATE-01 | FIX-08 | 20 | 2 | 2 |
| K-MISS-01 | FIX-09 | 26 | 26 | 0 |
| K-PY-01 | FIX-10 | 0 | 0 | 0 |
| K-API-03 | FIX-11 | 14 | 14 | 2 |
| K-MISS-02 | FIX-12 | 9 | 9 | 1 |
| K-BUG-01 | FIX-13 | 11 | 11 | 0 |
| K-BUG-02 | FIX-14 | 5 | 5 | 0 |
| K-MISS-03 | FIX-15 | 1 | 1 | 0 |
| K-API-17 | FIX-16 | 1 | 1 | 0 |
| K-API-02 | FIX-17 | 46 | 46 | 13 |
| **Summe** | | **347** | **329** | **18** |

Rote IDs, die es nur mit den Probe-Fixes gibt (3; 314 IDs sind verschwunden):

- `toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py::TestPollIntegration::test_external_delete_removes_from_vfs` (Katalog: K-FLAKY-01)
- `toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory_realworld.py::TestHybridMemoryRealWorld::test_02_vector_search` (Katalog: K-FLAKY-01)
- `toolboxv2/tests/test_mini/test_extras/test_analyser/test_profiler.py::TestE2EFunctionGraphDataIntegrity::test_call_chain_integrity` (**nicht im Katalog**)

Lesehilfe: Die 18 verbleibenden sind erwartet. K-API-02: 13 = die `vfs_shell(reason, …)`-Stellen, die dieser Lauf noch nicht enthielt (danach einzeln auf `master` @ `ed32daf` nachgeprüft und grün, s. Karte), und 2 Wechsler zu K-API-09. K-API-03: 2 Wechsler zu K-API-04. K-MISS-02: 1 ID mit zweiter Ursache. K-STATE-01: 2 IDs, die als root ohnehin nicht über `/fake` scheitern. Die 3 neuen roten IDs sind sporadische Tests (Watcher, Vektorsuche, Profiler); `test_profiler.py::…::test_call_chain_integrity` gehört zur selben Datei wie die Profiler-IDs in K-FLAKY-01 und war in keinem CI-Job rot.

Einschränkungen: Der Teil-Lauf ist nicht der Gesamtlauf (andere Reihenfolge-Nachbarn); K-PY-01 ist nur unter 3.10 rot und hier deshalb 0/0 (Beleg in der Karte); K-STATE-01 ist als root nicht rot (Beleg in der Karte über `/sys/fake`). Ein vollständiger lokaler Gesamtlauf war in der Analyseumgebung zu langsam (einzelne ISAA-Dateien warten auf Netz-Zeitüberschreitungen).

## 7. Stand der Abschlusskriterien

| # | Kriterium | Stand |
|---|---|---|
| A1 | H0 beantwortet, Log-Zeilen aus 3.10 und 3.11 | erfüllt (K-ENV-01, Tabelle oben; lokal mit 3.10 und 3.12 nachgestellt) |
| A2 | `raw/` enthält alle 12 Jobs von `ebed4d6` | erfüllt |
| A3 | jede ID genau einmal, keine ohne Kategorie | erfüllt (593 IDs) |
| A4 | jede Karte GETESTET oder UNGETESTET mit Grund; ≥ 80 % der IDs in GETESTET-Karten | **nicht erfüllt: 378 von 593 = 64 %.** Jede Karte hat Status und Grund. Ungetestet sind: veraltete Tests, deren Probe-Fix das Umschreiben der Tests wäre (P3, 107 IDs), reine Windows/macOS/Runner-/3.10-Fehler (P6, 39), optionale Pakete (26), `sys.modules`-Verschmutzung (22), Entscheidungsfragen (P4, 15), Sporadische (7), Port-Erwartungen (2) |
| A5 | jede Typ-A/B-Karte hat einen Fix-Auftrag | erfüllt (44) |
| A6 | jede Typ-C-Ursache hat eine Notiz mit ≥ 2 Wegen | erfüllt (ARCH-01, -02, -03) |
| A7 | Arbeitsbaum sauber, `git diff --stat master -- toolboxv2 .github` leer | erfüllt |

## 8. Fragen an den Owner

### Entscheidungen vom 2026-11-15

| # | Frage | Entscheidung | Folge |
|---|---|---|---|
| 1 | mcp 2.x: pinnen oder migrieren? | **migrieren** | FIX-03 ist jetzt die Migration (Server, ISAA-Client, FlowAgent, `mcp>=2.3,<3`); getestet mit mcp 2.3.0, Prompt 07. Nebenbefund dabei: unter mcp 2 sah der ISAA-Client von einem MCP-Server 0 statt 17 Werkzeuge, ohne Fehlermeldung |
| 2 | `waitress`: Basis-Abhängigkeit oder CI-Extra? | **CI installiert `.[web]`** | FIX-04 ändert nur `build.yml`, Teil von Prompt 01 |
| 3 | Sitzungsdatei je Gerät oder je Nutzer? | **je Nutzer** | FIX-19: `user/<name>_session.json` (`BlobFile` verlangt einen Ordner im Namen) plus einmalige Übernahme der alten `user/_session.json`, nur wenn sie demselben Nutzer gehört; getestet, Prompt 06. Beleg für die Dringlichkeit: heute lädt ein anderer Nutzer auf demselben Gerät die Anmeldung des vorigen |
| 4 | Veraltete Tests: generelle Freigabe? | **je Karte einzeln** | P3 (11 Karten, 107 IDs) bleibt gesperrt, bis die Fragen der Karten beantwortet sind |

### Typ B (je Fix-Auftrag)

| Fix | Karte | IDs | Frage |
|---|---|---|---|
| FIX-05 | K-ENV-07 | 0 | Owner: Secret `TAURI_PRIVATE_KEY` neu eintragen (eine Zeile, kein Umbruch) und `build.yml` manuell auslösen. Liegt danach `latest.json` signiert im Release `nightly`? |
| FIX-18 | K-API-15 | 2 | MCP-Vorgabeport 8791: gewollt? (WS 8468 ist durch `d1db5c9` belegt und kann ohne Rückfrage angepasst werden.) Außerdem prüft derselbe Test `http[0].port == 8000` – stimmt 8000 noch neben 8467? |
| FIX-20 | K-API-14 | 3 | (a) Soll `validate_session` den Token zusätzlich als `Authorization: Bearer` senden (heutiges Verhalten)? (d) Soll „abmelden“ die Sitzungsdatei löschen oder nur leeren (heute: leeren)? |
| FIX-21 | K-BUG-03 | 2 | Welcher Vorgabeport gilt für den Broadcast-Client (6568, 6588 oder der vereinheitlichte 8467)? Soll `host` der Name (`"Local test router"`) oder die IP sein? Wird der Broadcast noch gebraucht? |
| FIX-22 | K-API-16 | 2 | Ist `workers` jetzt `infrastructure`, und sind die auskommentierten `core`-Dienste endgültig weg? |
| FIX-23 | K-API-18 | 1 | Hat das Dockerfile noch Feature-Build-Argumente (`FEATURE_*`), oder ist es bewusst ein Vollabbild? |
| FIX-24 | K-API-19 | 1 | Soll jede HTML-Zeichenkette in die Seitenhülle eingebettet werden, auch ein Fragment? (Für HTMX-Teilantworten wäre das falsch.) |
| FIX-25 | K-API-06 | 3 | Sind 262 144 als Vorgabe für unbekannte Modelle und 2 000 000 für das geprüfte Modell gewollt? |
| FIX-26 | K-API-21 | 18 | Ist `llm_router` der einzige LLM-Weg (kein `llm_handler` mehr)? Soll `parse_media_from_query` das Feld `format` liefern? |
| FIX-27 | K-API-04 | 15 | Sind diese Änderungen gewollt: Auto-Fokus entfernt; `_build_system_prompt` → `_build_static_system_prompt`; Skill `habits_tracking` entfernt; `think` erhöht `max_iterations` anders? Wenn ja: Tests umschreiben. `'Situation' not found in "'async_generator' object is not callable"` sieht nach einem Fehler im `think`-Werkzeug aus – bitte bestätigen oder als Bug führen. |
| FIX-28 | K-API-08 | 20 | Ist die neue Overlay-Struktur der Soll-Zustand? `'thought A' not found in ['thought Athought B']` könnte ein echter Fehler sein (Gedanken werden zusammengeklebt) – gewollt? |
| FIX-29 | K-API-07 | 19 | Ist der Wegfall von `OmniSession(fallback=…)` und der Wechsel auf `GeminiLiveBackend` gewollt? Dann Tests umschreiben. |
| FIX-30 | K-API-09 | 8 | Ist `/system_context.md` absichtlich entfallen? Dann stellen die Tests auf `/vfs_guide.md` um. Nach K-API-02 kommen 2 IDs dazu. |
| FIX-31 | K-API-10 | 9 | Sind `_scan_disk`/`_dispatch_burst` endgültig weg? Dann die 9 Tests auf das öffentliche Verhalten umschreiben oder entfernen. |
| FIX-32 | K-API-12 | 6 | Sollen `.env`-Werte in einfachen Anführungszeichen geschrieben werden (`A='1'`)? Achtung: bestehende Einträge bleiben ohne (`X=0`), neue bekommen welche – gemischtes Format gewollt? |
| FIX-33 | K-API-20 | 3 | Sollen die Browser-Abläufe in pytest laufen (dann Fixture `tf` bauen; braucht Playwright und einen Server → eher eigener CI-Job) oder sind es Hilfsfunktionen (dann umbenennen, z. B. `run_login_…`)? |
| FIX-34 | K-API-22 | 9 | Keine Grundsatzfrage; je Fall meldet die Instanz „Test veraltet“ oder „Produktfehler“ mit Beleg, Owner bestätigt gesammelt. |
| FIX-39 | K-OS-04 | 2 | Darf ein Test die echte crontab des Nutzers ändern? Auf dem Entwicklerrechner ist das riskant (Regel 4). Alternative: `crontab`-Aufruf im Test durch ein Schein-Programm im `PATH` ersetzen. |
| FIX-40 | K-OS-05 | 3 | Sollen Tests echte Pakete installieren (dann eigener, getrennter CI-Job) oder die Paketmanager mocken? Im normalen Lauf verändern sie das System des Entwicklers. |

### Grundsätzliches

1. **3.10 in der Matrix behalten?** `requires-python = ">=3.10"`. 30 IDs sind nur unter 3.10 rot (K-PY-01, K-PY-02); beide sind behebbar.
2. **Dürfen Tests das System des Entwicklers ändern?** Gefunden: echte crontab (K-OS-04), echte Paketinstallation über winget/choco/apt/nix (K-OS-05), Schreiben nach `/fake/dir` (K-STATE-01), `docs/.docs_index.json` wird beim Testlauf ins Repo geschrieben (nicht in `.gitignore`).
3. ARCH-01 bis ARCH-03: Fragen stehen jeweils in Abschnitt 4 der Notiz.
4. **O3** (zwei Workflows schreiben ins Release `nightly`) ist nicht Teil dieser Analyse; durch K-ENV-07 ist das Release auf `master` zurzeit leer, die Messung „welche Dateien liegen im Release“ geht erst nach FIX-05.

## 9. Nebenbefunde (keine Test-IDs)

- `toolboxv2/mods/isaa/base/Agent/flow_agent.py:512`: Vorgabewert ist ein Pfad auf dem Owner-Rechner (`C:/Users/Markin/Workspace/ToolBoxV2/dist/toolboxv2-0.1.24-py2.py3-none-any.whl`).
- Laufzeit `py-check`: 19–27 Minuten (ubuntu/macOS), 42–48 Minuten (Windows).
- pytest-asyncio läuft ohne Konfiguration (`asyncio_default_fixture_loop_scope is unset`); es gibt keine `pytest.ini`/`[tool.pytest.ini_options]`.

## 10. Dateien

| Datei | Inhalt |
|---|---|
| `catalog.csv` | `test_id;job_list;kategorie;karte;fix_typ;paket;fix;status` – 593 Zeilen |
| `cards/` | 47 Karten |
| `fixes/` | 44 Fix-Aufträge, Nummer = Reihenfolge |
| `architecture/` | ARCH-01 (globaler Testzustand), ARCH-02 (Installation, ISAA-Paket), ARCH-03 (optionale Abhängigkeiten) |
| `raw/` | `<os>-<py>.txt` je Job (`ebed4d6`), `second_run_91d984a/`, `sporadic.txt` |
| `prompts/` | Fünf eigenständige Start-Prompts für ISAA-Instanzen, aufsteigende Komplexität, ohne offene Owner-Fragen: 01 CI-Signal (FIX-01/02/06) · 02 minimale Fixes (FIX-07–16) · 03 VFS-Werkzeuge (FIX-17, FIX-18 Teil WS-Port) · 04 sporadische Tests (FIX-44) · 05 Windows/3.10 (FIX-35–38, 41–43) · nach den Entscheidungen: 06 Sitzung je Nutzer (FIX-19), 07 mcp-2-Migration (FIX-03); 01 enthält zusätzlich FIX-04. Dateilisten überschneiden sich nicht, außer zwei ausdrücklich zeilengetrennten Dateien (`test_session.py` 02/06, `flow_agent.py` 03/07); jede Instanz arbeitet in einem eigenen git worktree und pusht nach `master` |
| `gate.py` | Abnahme-Helfer: `python docs/devdocs/ci_analysis/gate.py [Pfad] [--card K-…]` meldet rote IDs außerhalb des Katalogs. Reines Hilfsskript, kein Produktcode |

Befehle in den Karten sind für eine POSIX-Shell geschrieben und setzen `MOCK_EXTERNAL_SERVICES=true IS_OFFLINE_DB=true` voraus (PowerShell: `$env:MOCK_EXTERNAL_SERVICES="true"; $env:IS_OFFLINE_DB="true"`). `gate.py` setzt beide selbst.
