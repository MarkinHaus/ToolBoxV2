# Karte K-STATE-01

```
Karte:       K-STATE-01
Paket:       P1-AUTO-MINI – Minimaler Fix, automatisch
Status:      GETESTET (Nachstellung als root mit `/sys/fake/dir`; in der CI ist es `/fake`)
Fix-Typ:     A
Fix-Auftrag: FIX-08
```

**Ursache:** `test_session.py` setzt `TB_DATA_DIR=/fake/dir`; `Session()` legt dort wirklich Verzeichnisse an. Auf dem Runner ist das verboten, `asyncSetUp` bricht ab, der Patch bleibt aktiv und trifft die nächste Datei.

**Betroffen:** 20 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-STATE-01

**Beleg:** `toolboxv2/tests/test_mini/test_system/test_session.py:26,133` (`patch.dict(os.environ, {"TB_DATA_DIR": "/fake/dir"})`), Stop erst in `asyncTearDown` (Zeile 157). Aufruf: `session.py:76 → :114 _init_storage → blobs.py:525 → os.makedirs`. CI: `PermissionError: [Errno 13] Permission denied: '/fake'` (ubuntu), `OSError: [Errno 30] Read-only file system: '/fake'` (macOS); 13× `ERROR` in `test_toolboxv2.py::…::setUpClass` (`toolbox.py:88`)

**Hypothese:** Zwei Fehler hintereinander. (1) Der Test ist nicht so abgeschirmt, wie sein Kommentar sagt: Geräteschlüssel und Sitzungsordner werden echt unter `/fake/dir` angelegt. (2) Schlägt `asyncSetUp` fehl, ruft unittest `asyncTearDown` nicht auf; `TB_DATA_DIR=/fake/dir` bleibt für den Rest des Laufs gesetzt. `test_toolboxv2.py` läuft direkt danach und scheitert in `setUpClass`.

**Repro:**

```
# als Nicht-root (CI) direkt; als root zum Nachstellen in einer Kopie "/fake/dir" durch "/sys/fake/dir" ersetzen
python -m pytest toolboxv2/tests/test_mini/test_system/test_session.py toolboxv2/tests/test_mini/test_toolboxv2.py -q
```

**Vorher:** `PermissionError: [Errno 1] Operation not permitted: '/sys/fake'` – 7 FAILED in `test_session.py` (dieselben 7 Tests wie CI) + 13 ERROR in `test_toolboxv2.py`

**Probe-Fix:** `_TEST_DATA_DIR = tempfile.mkdtemp(prefix="tb_test_session_")` statt `"/fake/dir"`, und `self.addCleanup(self.env_patcher.stop)` direkt nach `start()`

**Nachher:** `7 failed, 18 passed` – die 13 ERROR und die 7 `/fake`-Fehler sind weg; die 7 verbleibenden sind K-API-13 und K-API-14

**Nachbarn:** s. Nachher

## Test-IDs

- `toolboxv2/tests/test_mini/test_system/test_session.py` – 7
- `toolboxv2/tests/test_mini/test_toolboxv2.py` – 13

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionAuth::test_login_invalid_token_triggers_successful_refresh | FAILED | OSError | [Errno 30] Read-only file system: '/fake'
```
