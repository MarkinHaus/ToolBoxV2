# Karte K-BUG-03

```
Karte:       K-BUG-03
Paket:       P4-ENTSCHEIDUNG – Braucht Entscheidung des Owners
Status:      UNGETESTET (kein Probe-Fix: unklar, welcher Vorgabeport und welcher `host`-Wert gewollt sind)
Fix-Typ:     B (Welcher Vorgabeport gilt für den Broadcast-Client (6568, 6588 oder der vereinheitlichte 8467)? Soll `host` der Name (`"Local test router"`) oder die IP sein? Wird der Broadcast noch gebraucht?)
Fix-Auftrag: FIX-21
```

**Ursache:** `brodcast/client.py` ruft `int(os.getenv("TOOLBOXV2_BASE_PORT"))` ohne Vorgabewert auf; ohne die Variable stirbt der Client-Thread.

**Betroffen:** 2 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-BUG-03

**Beleg:** `toolboxv2/utils/brodcast/client.py:33`. Testlog: `TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'` im Thread, Test: `AssertionError: '' != 'Local test router'`. macOS zusätzlich `OSError: [Errno 65] No route to host` in `test_make_known`.

**Hypothese:** An den anderen Stellen hat die Variable einen Vorgabewert (`EventManager/module.py:281` → 6588, `:459` → 6568). Hier fehlt er. Der Test erwartet außerdem `host == "Local test router"`, das Produkt sendet `host_ip`; ob das je gestimmt hat, ist offen. macOS-Runner: UDP-Broadcast nicht erlaubt (K-NET).

**Repro:**

```
python -m pytest toolboxv2/tests/test_core/test_brodcast -q
```

**Vorher:** `1 failed, 1 passed`

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_core/test_brodcast/test_bv_server_client.py` – 2

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_core/test_brodcast/test_bv_server_client.py::TestBrodcast::test_make_known | FAILED | OSError | [Errno 65] No route to host
```
