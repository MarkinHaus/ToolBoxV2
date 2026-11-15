# Karte K-API-13

```
Karte:       K-API-13
Paket:       P4-ENTSCHEIDUNG – Braucht Entscheidung des Owners
Status:      UNGETESTET (zwei gültige Richtungen)
Fix-Typ:     B (Eine Sitzungsdatei je Nutzer (`{safe_name}_session.json`, wie die Tests es erwarten) oder genau eine je Gerät (`user/_session.json`)? Betrifft WP1 (Login/Sitzung nach Update).)
Fix-Auftrag: FIX-19
```

**Ursache:** `Session._get_blob_name()` gibt immer `"user/_session.json"` zurück; der bereinigte Nutzername wird berechnet, aber nicht verwendet.

**Betroffen:** 3 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-13

**Beleg:** `toolboxv2/utils/system/session.py:119-125`: `safe_name = …` / `return f"user/_session.json"` (f-String ohne Platzhalter). CI: `'user/_session.json' != 'default_session.json'`, `!= 'user_evil_______name__session.json'`, `test_cli_session.py: 'default' not found in 'user/_session.json'`

**Hypothese:** Entweder ein Versehen (dann teilen sich alle Nutzer eines Geräts eine Sitzungsdatei) oder eine bewusste Vereinfachung auf eine Sitzung je Gerät (dann sind `safe_name` und die Tests überholt). Der Code allein entscheidet das nicht.

**Repro:**

```
python -m pytest toolboxv2/tests/test_mini/test_system/test_session.py toolboxv2/tests/test_mini/test_cli_session.py -q
```

**Vorher:** 3 dieser IDs rot

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_mini/test_cli_session.py` – 1
- `toolboxv2/tests/test_mini/test_system/test_session.py` – 2

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_mini/test_cli_session.py::TestSessionPathSafety::test_empty_username_uses_default | FAILED | AssertionError | 'default' not found in 'user/_session.json'
```
