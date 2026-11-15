# FIX-19 – aus Karte K-API-13

Paket: **P4-ENTSCHEIDUNG** (Braucht Entscheidung des Owners) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 3 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Eine Sitzungsdatei je Nutzer (`{safe_name}_session.json`, wie die Tests es erwarten) oder genau eine je Gerät (`user/_session.json`)? Betrifft WP1 (Login/Sitzung nach Update).

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-19  (aus Karte K-API-13)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/utils/system/session.py oder die 3 Tests   (nur diese ändern)
Vorher lesen:  `session.py:100-205`, wer `_get_blob_name` aufruft; Auswirkungen auf vorhandene Sitzungsdateien
```

**Schritte:**

1. Je Nutzer: `return f"{safe_name}_session.json"` (und Migration der vorhandenen `user/_session.json` bedenken).
2. Je Gerät: tote Zeilen (`safe_name`) entfernen, 3 Tests umstellen (auch `…writes_encrypted_json` aus K-API-14).

**Nicht tun:** Nicht ohne Antwort ändern – bestehende Anmeldungen hängen daran. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 3 IDs grün.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_mini/test_cli_session.py::TestSessionPathSafety::test_empty_username_uses_default
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionStorage::test_get_blob_name_empty_username_uses_default
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionStorage::test_get_blob_name_sanitizes_special_characters
```
