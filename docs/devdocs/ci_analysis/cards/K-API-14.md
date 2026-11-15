# Karte K-API-14

```
Karte:       K-API-14
Paket:       P4-ENTSCHEIDUNG – Braucht Entscheidung des Owners
Status:      UNGETESTET (zwei offene Fragen)
Fix-Typ:     B ((a) Soll `validate_session` den Token zusätzlich als `Authorization: Bearer` senden (heutiges Verhalten)? (d) Soll „abmelden“ die Sitzungsdatei löschen oder nur leeren (heute: leeren)?)
Fix-Auftrag: FIX-20
```

**Ursache:** Restliche Abweichungen in `test_session.py`: erwarteter Aufruf ohne `Authorization`-Header; Sitzung löschen erwartet `delete_blob`, das Produkt schreibt `{}` (und scheitert im Test am Schlüssel). Zwei weitere Teile (Blob-Name, `del os.environ`) erledigt FIX-19.

**Betroffen:** 3 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-14

**Beleg:** `Expected: mock('POST', url=…, json={'token': 'valid_token'})` / `Actual: … headers={'Authorization': 'Bearer valid_token'}`; `Error clearing session: Invalid symmetric key provided.` → `False is not true`; `test_session.py:125` `delete_blob.assert_called_with(…)`. Lokal zusätzlich `test_download_file_writes_chunks_to_disk` (`Cannot connect to host test.local`), in der CI nicht aufgeführt.

**Hypothese:** (a) Produkt sendet jetzt den Header mit – Test veraltet? (d) `_clear_session_token` leert die Datei statt sie zu löschen; der Test erwartet Löschen.

**Repro:**

```
python -m pytest toolboxv2/tests/test_mini/test_system/test_session.py -q
```

**Vorher:** nach FIX-08 und FIX-19: 3 rot

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_mini/test_system/test_session.py` – 3

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionStorage::test_clear_session_token_deletes_blob_and_clears_attributes | FAILED | AssertionError | False is not true
```
