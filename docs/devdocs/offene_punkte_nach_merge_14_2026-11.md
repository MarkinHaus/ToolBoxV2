# Offene Punkte nach dem Merge von #14 (Stand 2026-11-15)

> `master` @ `ebed4d6`: R1–R9 sind gemergt (stabile Installation, eine `.env`, ein Datenordner, Kern-Update über Build-ID, `litellm` optional, ein Build-Workflow, Windows-Nachweis R9).
> Gemergt trotz roter Python-3.12/3.13-Jobs. Die waren schon vorher auf `master` rot (→ WP2).

## Arbeitspakete

| Paket | Datei | Inhalt | Wer |
|---|---|---|---|
| WP1 | `WP1_konto_und_sync_2026-11.md` | Kontopflicht, Offline-Anmeldung, gezielter Sync über die Data-API (K1–K4, S1–S4, T10) | lokaler Agent; D1 und Live-Login macht der Owner |
| WP2 | `WP2_ci_fehleranalyse_2026-11.md` | CI-Fehler kategorisieren, Hypothesen testen, Fix-Plan; Architekturfragen als Lösungsvorschläge | Analyse: Agent; Umsetzung: kleineres Modell nach Freigabe |

## Einzelpunkte

| # | Punkt | Stand | Nächster Schritt | Wer |
|---|---|---|---|---|
| O1 | Laptop-Test (Weg 6): SimpleCore-Setup installieren, `tb` im Terminal global, `scripts/verify_install_windows.ps1` | offen | Setup aus dem Release `nightly` (oder Artefakt `simplecore-x86_64-pc-windows-msvc` des Laufs auf `ebed4d6`) installieren, Skript ausführen, Ausgabe zurückmelden | Owner |
| O2 | Updater Ende-zu-Ende (Z7): App findet neuere Version und installiert sie, Profil/Daten/Secret bleiben | offen, braucht zwei Veröffentlichungen | nach O1 eine zweite Veröffentlichung auslösen (Push auf `master` oder `build.yml` manuell), App neu starten, `verify_install_windows.ps1 -CompareBaseline` | Owner |
| O3 | **Zwei Workflows schreiben ins Release `nightly`:** `build.yml` löscht und erzeugt es neu (Tauri, `latest.json`), `release.yml` hängt bei jedem Push auf `master` die `tb`-Einzeldateien an („Update nightly prerelease", `release.yml:90`). Je nach Reihenfolge gehen Dateien verloren, oder `latest.json` fehlt kurzzeitig | Risiko, nicht gemessen | Nach dem ersten Lauf auf `ebed4d6` prüfen, welche Dateien im Release liegen und ob `latest.json` signiert dabei ist. Dann entscheiden: `release.yml` auf Tags beschränken (Owner-Entscheidung 8 sah das für die App so vor) oder getrenntes Release für die `tb`-Dateien | Planer, Entscheidung Owner |
| O4 | `release.yml` → `build-windows-x64` rot: `No module named nuitka` (Schritt „Build onefile", Lauf 43742117826) | gab es schon vor #14 | in WP2 als K-ENV aufnehmen oder direkt: Nuitka im Job installieren wie im Linux-Job | WP2 / klein |
| O5 | `test_shell_tool::TestKillProcess::…grandparent…` nur in der CI rot | aus dem portierten ISAA-Stand des Owners | in WP2 analysieren (K-OS) | WP2 |
| O6 | Gegenprüfung der R9-Commits des lokalen Agenten (`19019ff..7ec64ac`: `verify_install_windows.ps1`, NginxManager-Symlink-Fallback, `pids_dir`, ps1-Standardpfad, Doppel-BOM, Auto-Mount `app:<name>`, `mount_app` übernimmt Routen desselben Moduls) | gemergt, vom Planer nicht einzeln gegengeprüft | Review vor bzw. parallel zu WP1-K3, weil `mount_app` und `local_ui` dort berührt werden | Planer |
| O7 | Code-Signatur (SmartScreen-Warnung) | bewusst verschoben | wenn es zahlende Kunden gibt: OV-Zertifikat auf den eigenen Namen (Freelancer) oder Signierdienst; Updater-Signatur (`TAURI_PRIVATE_KEY`) ist davon unabhängig und bereits eingerichtet | Owner |
| O8 | Doppelter Async-Helper in `SocketManager`/`EventManager` | bewusst so gelassen | später zusammenführen (eigene Aufgabe) | — |
| O9 | `docs/devdocs/taskmap_stable_tauri_2026-11.md` und `handoff_agent_stable_tauri_2026-11.md` sind veraltet (K/S mit Kontoschlüssel und MinIO) | Hinweis | gelten nur noch als Hintergrund; maßgeblich sind WP1 und WP2 | — |

## Gestrichen

- Punkt E (Apps auf dem Server mounten): Apps laufen nur lokal im Live-UI-Owner; der Server lädt Module und stellt `api=True`-Funktionen bereit.
