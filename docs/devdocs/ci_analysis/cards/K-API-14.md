# Karte K-API-14

```
Karte:       K-API-14
Paket:       P4-ENTSCHEIDUNG – Braucht Entscheidung des Owners
Status:      UNGETESTET (vier Einzelursachen, kein Probe-Fix)
Fix-Typ:     B (Soll `validate_session` den Token zusätzlich als `Authorization: Bearer` senden (heutiges Verhalten)? Dann Test anpassen.)
Fix-Auftrag: FIX-20
```

**Ursache:** Vier weitere Abweichungen in `test_session.py`: erwarteter Aufruf ohne `Authorization`-Header, erwarteter Blob-Name, `del os.environ["TOOLBOXV2_REMOTE_BASE"]` ohne gesetzte Variable, Löschen der Sitzung mit ungültigem Schlüssel.

**Betroffen:** 3 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-14

**Beleg:** CI/lokal: `Expected: mock('POST', url=…, json={'token': 'valid_token'})` / `Actual: … headers={'Authorization': 'Bearer valid_token'}`; `Expected: BlobFile('testuser_session.json', …)` / `Actual: BlobFile('user/_session.json', …)`; `KeyError: 'TOOLBOXV2_REMOTE_BASE'` (`test_session.py:104`); `Error clearing session: Invalid symmetric key provided.` → `False is not true`

**Hypothese:** Gemischt: (a) Produkt sendet jetzt den Header mit – Test veraltet. (b) folgt K-API-13. (c) Testfehler: Variable wird gelöscht, die niemand gesetzt hat. (d) Der Test mockt den Geräteschlüssel nicht vollständig. Lokal schlägt zusätzlich der Download-Test fehl (`Cannot connect to host test.local`), in der CI nicht aufgeführt.

**Repro:**

```
python -m pytest toolboxv2/tests/test_mini/test_system/test_session.py -q
```

**Vorher:** lokal `7 failed` (davon 2 K-API-13, 1 nur lokal)

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_mini/test_system/test_session.py` – 3

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionStorage::test_clear_session_token_deletes_blob_and_clears_attributes | FAILED | AssertionError | False is not true
```
