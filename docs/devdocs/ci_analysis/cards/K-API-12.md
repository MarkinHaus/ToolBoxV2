# Karte K-API-12

```
Karte:       K-API-12
Paket:       P3-VERALTET – Test veraltet (Schnittstelle absichtlich geändert)
Status:      UNGETESTET (Umschreiben; eine Frage zum `.env`-Format)
Fix-Typ:     B (Sollen `.env`-Werte in einfachen Anführungszeichen geschrieben werden (`A='1'`)? Achtung: bestehende Einträge bleiben ohne (`X=0`), neue bekommen welche – gemischtes Format gewollt?)
Fix-Auftrag: FIX-32
```

**Ursache:** Provider-Tests nutzen alte `OnboardingResult`-Felder (`payg_providers`, `coding_providers`) und ein altes `.env`-Schreibformat (ohne Anführungszeichen).

**Betroffen:** 6 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-12

**Beleg:** CI: `TypeError: OnboardingResult.__init__() got an unexpected keyword argument 'payg_providers'` (2), `'coding_providers'` (1); `'A=1' not found in "A='1'\nB='2'\n"`, `"X=0\nY='1'\n" != 'X=0\nY=1\n'`; `test_onboarding_v2.py: 0 != 1`

**Hypothese:** Onboarding v2 hat die Ergebnisstruktur geändert; der `.env`-Schreiber setzt Werte jetzt in Anführungszeichen.

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_providers -q
```

**Vorher:** 6 rot

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_isaa/test_providers/test_free_providers.py` – 2
- `toolboxv2/tests/test_isaa/test_providers/test_onboarding_v2.py` – 1
- `toolboxv2/tests/test_isaa/test_providers/test_paid.py` – 3

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_providers/test_free_providers.py::TestAppendEnv::test_append_only | FAILED | AssertionError | 'A=1' not found in "A='1'\nB='2'\n"
```
