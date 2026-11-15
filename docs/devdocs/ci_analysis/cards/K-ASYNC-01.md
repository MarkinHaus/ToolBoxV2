# Karte K-ASYNC-01

```
Karte:       K-ASYNC-01
Paket:       P1-AUTO-MINI – Minimaler Fix, automatisch
Status:      GETESTET
Fix-Typ:     A
Fix-Auftrag: FIX-07
```

**Ursache:** Synchrone Tests rufen `asyncio.get_event_loop().run_until_complete(…)` auf; ein früherer Test hat die Schleife geschlossen oder abgemeldet.

**Betroffen:** 160 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-ASYNC-01

**Beleg:** `toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py:49`; dasselbe Muster in 16 Testdateien. Verschmutzer z. B. `toolboxv2/tests/test_isaa/test_isaa_module.py:54-65` (`new_event_loop`/`set_event_loop`/`close`) und `toolboxv2/tests/a_util.py:10-22`. CI: `RuntimeError: Event loop is closed` (92) und `RuntimeError: There is no current event loop in thread 'MainThread'.` (68)

**Hypothese:** Reihenfolge-Fehler, kein 3.12-Fehler: die 160 IDs schlagen in allen 11 Jobs fehl, auch unter 3.10. Jede der 10 Dateien ist einzeln grün. Sobald ein Test davor die aktuelle Schleife schließt (`loop.close()`) oder mit `asyncio.run()` abmeldet, liefert `get_event_loop()` eine geschlossene oder keine Schleife.

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_isaa_module.py toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py -q
```

**Vorher:** 40× `RuntimeError: Event loop is closed` (dieselben 40 IDs wie in der CI). Allein: `test_job_manager.py` → `107 passed`

**Probe-Fix:** autouse-Fixture in `toolboxv2/tests/conftest.py`: vor jedem Test prüfen, ob die aktuelle Schleife fehlt oder geschlossen ist, dann `asyncio.set_event_loop(asyncio.new_event_loop())` (Diff im Fix-Auftrag)

**Nachher:** `python -m pytest toolboxv2/tests/test_isaa/test_isaa_module.py toolboxv2/tests/test_isaa/test_jobs/ -q` → `5 failed, 300 passed, 8 skipped` (die 5: `crontab` fehlt lokal, K-OS-04). `… test_isaa_module.py test_mini/test_cloudm/test_LiveSync test_isaa/test_base/test_agent/test_dreamer -q` → `317 passed, 3 skipped`

**Nachbarn:** s. Nachher; Gesamtlauf über die 77 betroffenen Dateien s. README Abschnitt 6

**Notiz:** Die Fixture behebt das Symptom für alle 160 IDs. Die Ursache im Aufbau (wer besitzt die Schleife, drei verschiedene async-Hilfen in den Tests) steht in ARCH-01.

## Test-IDs

- `toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py` – 18
- `toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p0_fixes.py` – 7
- `toolboxv2/tests/test_isaa/test_codig/test_coder_apply_back_hypotheses.py` – 9
- `toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py` – 40
- `toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py` – 43
- `toolboxv2/tests/test_isaa/test_jobs/test_jobs_offline_validation.py` – 9
- `toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_client.py` – 9
- `toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py` – 12
- `toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py` – 13

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_aggregate_never_raises | FAILED | RuntimeError | There is no current event loop in thread 'MainThread'.
```
