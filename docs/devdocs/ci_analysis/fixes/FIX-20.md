# FIX-20 – aus Karte K-API-14

Paket: **P4-ENTSCHEIDUNG** (Braucht Entscheidung des Owners) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 3 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: (a) Soll `validate_session` den Token zusätzlich als `Authorization: Bearer` senden (heutiges Verhalten)? (d) Soll „abmelden“ die Sitzungsdatei löschen oder nur leeren (heute: leeren)?

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-20  (aus Karte K-API-14)
Voraussetzung: FIX-08, FIX-19
Dateien:       toolboxv2/tests/test_mini/test_system/test_session.py; ggf. toolboxv2/utils/system/session.py (`_clear_session_token`)   (nur diese ändern)
Vorher lesen:  `session.py` `login`, `_clear_session_token`
```

**Schritte:**

1. (a) nach Antwort Erwartung um `headers=` ergänzen.
2. (d) nach Antwort: Produkt löscht (`self._storage.delete_blob(...)`) oder Test prüft das Leeren; Geräteschlüssel im Test so mocken, dass `BlobFile` arbeitet.

**Nicht tun:** – Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 2 IDs grün.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionStorage::test_clear_session_token_deletes_blob_and_clears_attributes
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionStorage::test_load_session_token_restores_state
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionStorage::test_save_session_token_writes_encrypted_json
```
