# Karte K-FLAKY-01

```
Karte:       K-FLAKY-01
Paket:       P7-FLAKY – Sporadisch
Status:      UNGETESTET (sporadisch; kein gemeinsamer Probe-Fix)
Fix-Typ:     A
Fix-Auftrag: FIX-44
```

**Ursache:** Zeit- oder zufallsabhängige Tests: Datei-Watcher mit festen Wartezeiten, Profiler-Zählwerte, Backup-Aufräumen nach Zeitstempel, Vektorsuche, Audio-Reihenfolge.

**Betroffen:** 7 Tests in 7 Jobs (macos-3.11, macos-3.12, ubuntu-3.10, ubuntu-3.11, ubuntu-3.12, ubuntu-3.13, windows-3.12); Liste in catalog.csv, Spalte karte=K-FLAKY-01

**Beleg:** `test_vfs_pooling.py`: `Extern erstelltes File muss nach 4.0s sichtbar sein`, `Extern gelöschtes File muss nach 4.0s entfernt sein`, `Agent B muss File sehen das A erstellt hat` – je Job andere Auswahl; `test_profiler.py`: `22 != 30 : compute should have 30 calls` (nur ein Job); `test_rollback_manager.py`: `assert 4 == 7` (zwei Jobs, lokal `5 == 7`); `test_hybrid_memory_realworld.py`: `'SpaceX' not found in ['Tesla, Inc.', …]` (Ergebnisliste je Job anders); `test_audio_core.py::test_multiple_texts_ordered` (ein Job)

**Hypothese:** Vergleich der beiden Läufe (`91d984a` und `ebed4d6`, gleicher Baum) und der Jobs untereinander: diese IDs wechseln. Gemeinsame Art: feste `sleep`-Zeiten, Zeitstempel mit Sekundenauflösung, zufällige oder nicht festgelegte Einbettungen.

**Repro:**

```
for i in 1 2 3 4 5; do python -m pytest toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py -q | tail -1; done
```

**Vorher:** wechselnd

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py` – 4
- `toolboxv2/tests/test_isaa/test_base/test_audio/test_audio_core.py` – 1
- `toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory_realworld.py` – 1
- `toolboxv2/tests/test_mini/test_extras/test_install/test_rollback_manager.py` – 1

Beispiel aus `raw/ubuntu-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py::TestMultiAgentSync::test_a_creates_file_b_sees_it | FAILED | AssertionError | False is not true : Agent B muss File sehen das A erstellt hat. Registry: {'active_watchers': 1, 'watchers': [{'local_path': '/tmp/tmplir_ma5v', 'subscribers': 2, 'tracked_files': 2, 'poll_interval': 
```
