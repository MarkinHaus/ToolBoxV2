# FIX-44 – aus Karte K-FLAKY-01

Paket: **P7-FLAKY** (Sporadisch) · Fix-Typ: **A** · Karte: UNGETESTET · betrifft 7 Test-IDs

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-44  (aus Karte K-FLAKY-01)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       je Test die Testdatei; bei `test_vfs_pooling` evtl. der Watcher   (nur diese ändern)
Vorher lesen:  je Test die Wartelogik
```

**Schritte:**

1. Feste `sleep(x)` + einmalige Prüfung durch „warte bis Bedingung, höchstens T Sekunden“ ersetzen.
2. `test_rollback_manager`: Zeitstempel der Test-Backups ausdrücklich setzen statt auf die Uhr zu vertrauen.
3. `test_hybrid_memory_realworld`: Einbettungen im Test festlegen (deterministischer Mock) oder die Erwartung an die Rangfolge begründen.
4. Jede ID 20× hintereinander laufen lassen; erst dann als behoben melden.

**Nicht tun:** Kein `flaky`-Plugin, kein Wiederholen bis grün, keine längeren festen Wartezeiten als Lösung. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- jede ID 20 von 20 Läufen grün.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py::TestMultiAgentSync::test_a_creates_file_b_sees_it
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py::TestMultiAgentSync::test_b_creates_file_a_sees_it
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py::TestMultiAgentSync::test_b_writes_to_a_file_a_reads_new_content
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py::TestPollIntegration::test_external_delete_removes_from_vfs
toolboxv2/tests/test_isaa/test_base/test_audio/test_audio_core.py::TestAudioStreamPlayer::test_multiple_texts_ordered
toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory_realworld.py::TestHybridMemoryRealWorld::test_02_vector_search
toolboxv2/tests/test_mini/test_extras/test_install/test_rollback_manager.py::test_cleanup_old_backups
```
