# Karte K-API-17

```
Karte:       K-API-17
Paket:       P1-AUTO-MINI – Minimaler Fix, automatisch
Status:      GETESTET
Fix-Typ:     A
Fix-Auftrag: FIX-16
```

**Ursache:** Test sucht im Quelltext nach `_fetch_tray_state`; die Funktion heißt `fetch_tray_state`.

**Betroffen:** 1 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-17

**Beleg:** `toolboxv2/utils/extras/fallback_tray.py:76` `def fetch_tray_state() -> dict | None:`; `toolboxv2/tests/test_core/test_tray_daemon_integration.py:461`. CI: `'_fetch_tray_state' not found in …`

**Hypothese:** Umbenennung (öffentlich gemacht).

**Repro:**

```
python -m pytest toolboxv2/tests/test_core/test_tray_daemon_integration.py -q
```

**Vorher:** `1 failed, 41 passed`

**Probe-Fix:** `assertIn("fetch_tray_state", src)`

**Nachher:** `42 passed`

**Nachbarn:** ganze Datei grün

## Test-IDs

- `toolboxv2/tests/test_core/test_tray_daemon_integration.py` – 1

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_core/test_tray_daemon_integration.py::TestDaemonServiceE2EValidation::test_fallback_tray_fetches_state | FAILED | AssertionError | '_fetch_tray_state' not found in '"""Fallback system tray (pystray) for ToolBoxV2.\n\nUsed whenever the Tauri desktop app is not available: the tray is the only\nalways-on surface the user has, so eve
```
