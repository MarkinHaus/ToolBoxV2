# Karte K-API-13

```
Karte:       K-API-13
Paket:       P4-ENTSCHEIDUNG – Braucht Entscheidung des Owners
Status:      GETESTET
Fix-Typ:     A
Fix-Auftrag: FIX-19
```

**Ursache:** `Session._get_blob_name()` gibt immer `"user/_session.json"` zurück; der bereinigte Nutzername wird berechnet, aber nicht verwendet. Alle Nutzer eines Geräts teilen sich eine Sitzungsdatei.

**Betroffen:** 3 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-13

**Beleg:** `toolboxv2/utils/system/session.py:119-125`: `safe_name = …` / `return f"user/_session.json"`, eingeführt mit `8dbf40b`. `BlobFile` verlangt einen Namen der Form `<blob_id>/<datei>` – `alice_session.json` (was die Tests erwarten) wirft `ValueError: Filename must include a path within the blob, e.g., 'blob_id/file.txt'`. CI: `'user/_session.json' != 'default_session.json'`, `!= 'user_evil_______name__session.json'`, `test_cli_session.py: 'default' not found in 'user/_session.json'`

**Hypothese:** Vermutlich wurde `{safe_name}_session.json` wegen dieses `ValueError` durch einen festen Namen ersetzt und der Nutzername ging dabei verloren. **Owner-Entscheidung 2026-11-15: eine Sitzungsdatei je Nutzer.** Weil `BlobFile` einen Ordner verlangt: `user/<name>_session.json`. Die App übergibt immer einen Namen (`toolbox.py:226` `Session(self.get_username())`, Vorgabe `"loot"`). Bestehende Installationen haben ihre Sitzung in `user/_session.json`; ohne Übernahme wären alle nach dem Update abgemeldet.

**Repro:**

```
python -m pytest toolboxv2/tests/test_mini/test_system/test_session.py toolboxv2/tests/test_mini/test_cli_session.py toolboxv2/tests/test_mini/test_toolboxv2.py -q
```

**Vorher:** `8 failed, 28 passed` (3× K-API-13, 2× K-API-14 b/c, 3× K-API-14 a/d bzw. Netz). Szenario-Skript ohne Fix: `3 carol: {'access_token': 'B-TOKEN', 'username': 'bob'}` – **eine andere Nutzerin bekommt bobs Anmeldung**; `4 dave: None | user/_session.json`.

**Probe-Fix:** Patch `probe_session_per_user.diff`: `_get_blob_name` → `user/{safe_name}_session.json`; `_migrate_legacy_session()` übernimmt einmalig `user/_session.json`, wenn sie diesem Nutzer gehört oder keinen Besitzer hat, und leert sie; `_load_session_token` fällt darauf zurück. Tests: drei erwartete Namen mit `user/`-Präfix, `del os.environ[…]` → `os.environ.pop(…, None)`, Längentest misst nur den Dateinamen.

**Nachher:** `3 failed, 33 passed` – übrig: `test_clear_session_token_deletes_blob_and_clears_attributes`, `test_login_valid_token_sets_session_valid` (offene Frage K-API-14), `test_download_file_writes_chunks_to_disk` (braucht Netz). Szenario-Probe mit echter `BlobStorage`: (1) alte Datei von alice → alice lädt `A-TOKEN`, neue Datei angelegt, alte geleert; (2) zweiter Start lädt aus der neuen Datei; (3) alte Datei von bob, Nutzer carol → nichts übernommen, bobs Datei bleibt; (4) Speichern schreibt `user/dave_session.json`.

**Nachbarn:** `test_mini/test_toolboxv2.py` unverändert grün; Patch gilt auf `master` @ `ed32daf` und auch nach FIX-08

**Notiz:** Betrifft WP1 (Anmeldung nach Update). Nach dem Rollout im Bericht festhalten, ob bei einem echten Update die Übernahme im Log erscheint (`Session migrated from user/_session.json to …`).

**Entscheidung des Owners (2026-11-15):** Eine Sitzungsdatei je Nutzer, mit einmaliger Übernahme der alten Datei.

## Test-IDs

- `toolboxv2/tests/test_mini/test_cli_session.py` – 1
- `toolboxv2/tests/test_mini/test_system/test_session.py` – 2

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_mini/test_cli_session.py::TestSessionPathSafety::test_empty_username_uses_default | FAILED | AssertionError | 'default' not found in 'user/_session.json'
```
