# FIX-20 – aus Karte K-API-14

Paket: **P4-ENTSCHEIDUNG** (Braucht Entscheidung des Owners) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 3 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Soll `validate_session` den Token zusätzlich als `Authorization: Bearer` senden (heutiges Verhalten)? Dann Test anpassen.

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-20  (aus Karte K-API-14)
Voraussetzung: FIX-08, FIX-19
Dateien:       toolboxv2/tests/test_mini/test_system/test_session.py   (nur diese ändern)
Vorher lesen:  `session.py` `login`, `_clear_session_token`, `_load_session_token`
```

**Schritte:**

1. (a) nach Antwort Erwartung um `headers=` ergänzen.
2. (c) `os.environ.pop("TOOLBOXV2_REMOTE_BASE", None)` statt `del`.
3. (d) Geräteschlüssel im Test so mocken, dass `_clear_session_token` einen gültigen Schlüssel bekommt.
4. (b) folgt der Entscheidung aus K-API-13.

**Nicht tun:** – Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 3 IDs der Karte grün (nach FIX zu K-STATE-01 und K-API-13).
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionStorage::test_clear_session_token_deletes_blob_and_clears_attributes
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionStorage::test_load_session_token_restores_state
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionStorage::test_save_session_token_writes_encrypted_json
```
