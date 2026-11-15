# FIX-29 – aus Karte K-API-07

Paket: **P3-VERALTET** (Test veraltet (Schnittstelle absichtlich geändert)) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 19 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Ist der Wegfall von `OmniSession(fallback=…)` und der Wechsel auf `GeminiLiveBackend` gewollt? Dann Tests umschreiben.

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-29  (aus Karte K-API-07)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py, test_omni_audio.py   (nur diese ändern)
Vorher lesen:  `OmniSession.__init__`, Backend-Auswahl, wie der Agent aufgerufen wird
```

**Schritte:**

1. Tests an die aktuelle Signatur anpassen; `FakeAgent` um `a_stream` ergänzen.
2. Fallback-Tests: nur entfernen, wenn es keinen Fallback mehr gibt (Owner).

**Nicht tun:** Keine Assertions streichen, um grün zu werden. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 19 IDs grün oder bestätigt entfernt.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py::TestAgentTools::test_agent_status_reflects_multiple
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py::TestAgentTools::test_delegate_returns_job_id_immediately_then_result
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py::TestCompress::test_compress_folds_history_into_summary
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py::TestCompress::test_compress_keeps_turns_arriving_during_call
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py::TestCompress::test_compress_no_merge_drops_prior
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py::TestCompress::test_compress_nothing_to_do
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py::TestCompress::test_compress_tool_schedules_and_runs
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py::TestEventDispatch::test_audio_event_wraps_pcm_and_queues_to_player
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py::TestEventDispatch::test_error_event_triggers_fallback
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py::TestEventDispatch::test_fallback_only_once
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py::TestRestart::test_do_restart_swaps_backend_and_reseeds
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py::TestRestart::test_should_restart_threshold
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py::TestRestart::test_turn_end_at_threshold_schedules_restart
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py::TestSeedAndArchive::test_archive_appends_one_liner
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py::TestSeedAndArchive::test_archive_noop_without_summary
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py::TestSessionIntegration::test_full_roundtrip_tool_call_then_followup_audio
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py::TestSessionIntegration::test_start_failure_falls_back
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni.py::TestVoiceModeConfig::test_omni_cloud_builds_without_connecting
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_audio.py::TestVadGate::test_pump_gates_silence_frames
```
