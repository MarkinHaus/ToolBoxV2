# FIX-07 – aus Karte K-ASYNC-01

Paket: **P1-AUTO-MINI** (Minimaler Fix, automatisch) · Fix-Typ: **A** · Karte: GETESTET · betrifft 160 Test-IDs

```
Fix:           FIX-07  (aus Karte K-ASYNC-01)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/conftest.py   (nur diese ändern)
Vorher lesen:  `toolboxv2/tests/conftest.py` ganz; `toolboxv2/tests/a_util.py:10-22`
```

**Schritte:**

1. Nach der Fixture `reset_app_state_before_test` diese Fixture einfügen:

```python
@pytest.fixture(autouse=True)
def ensure_open_event_loop():
    """Sync tests that call asyncio.get_event_loop().run_until_complete(...) need
    an open current loop. Earlier tests may have closed or unset it."""
    import asyncio
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            loop = asyncio.get_event_loop_policy().get_event_loop()
        stale = loop.is_closed()
    except RuntimeError:
        stale = True
    if stale:
        asyncio.set_event_loop(asyncio.new_event_loop())
    yield
```

**Nicht tun:** Keine Testdatei ändern. Kein `nest_asyncio`. Die Fixture schließt keine Schleife. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- `python -m pytest toolboxv2/tests/test_isaa/test_isaa_module.py toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py -q` → `0 failed`. Alle 160 IDs der Karte grün im Gesamtlauf.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_aggregate_never_raises
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_effort_ratio_gated_until_three_entries
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_fuzzy_only_classification_and_persist
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_guid_md_never_touched
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_guide_written_and_extended
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_happypath_only_on_improvement
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_index_stats_accumulate
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_llm_classification_used
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_llm_failure_falls_back_to_fuzzy_or_new
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_llm_slash_in_task_type_splits_correctly
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestAggregator::test_unclear_goes_to_new_sink
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestPreInjection::test_dominant_fuzzy_skips_llm
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestPreInjection::test_injection_block_contains_guide_happypath_and_swap_notice
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestPreInjection::test_narrator_call_used_when_ambiguous
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestPreInjection::test_never_raises
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestPreInjection::test_new_type_injects_nothing
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestPreInjection::test_no_data_injects_nothing
toolboxv2/tests/test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py::TestPreInjection::test_uncertain_returns_new
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p0_fixes.py::TestT2TextSerialize::test_flush_noop_when_empty
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p0_fixes.py::TestT2TextSerialize::test_passes_through_when_idle
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p0_fixes.py::TestT2TextSerialize::test_queues_while_speaking_then_flushes_once
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p0_fixes.py::TestT8ChatViewTools::test_chat_executions_returns_running
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p0_fixes.py::TestT8ChatViewTools::test_chat_history_returns_last_n
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p0_fixes.py::TestT8ChatViewTools::test_provider_errors_are_caught
toolboxv2/tests/test_isaa/test_base/test_audio/test_omni_p0_fixes.py::TestT9StatusReportsOnce::test_each_finished_job_new_once
toolboxv2/tests/test_isaa/test_codig/test_coder_apply_back_hypotheses.py::TestH1_ChangedFilesAfterCommit::test_changed_files_empty_after_commit
toolboxv2/tests/test_isaa/test_codig/test_coder_apply_back_hypotheses.py::TestH1_ChangedFilesAfterCommit::test_changed_files_nonempty_before_commit
toolboxv2/tests/test_isaa/test_codig/test_coder_apply_back_hypotheses.py::TestH3_ApplyBackSubfolder::test_apply_back_creates_new_subfolder_file
toolboxv2/tests/test_isaa/test_codig/test_coder_apply_back_hypotheses.py::TestH3_ApplyBackSubfolder::test_apply_back_modifies_subfolder_file
toolboxv2/tests/test_isaa/test_codig/test_coder_apply_back_hypotheses.py::TestH4_ApplyBackFallback::test_explicit_no_commit_no_changes_returns_zero
toolboxv2/tests/test_isaa/test_codig/test_coder_apply_back_hypotheses.py::TestH4_ApplyBackFallback::test_uncommitted_changes_apply_via_fallback
toolboxv2/tests/test_isaa/test_codig/test_coder_apply_back_hypotheses.py::TestH5_CommitStagingFailure::test_ignored_file_NOT_propagated_by_apply_back
toolboxv2/tests/test_isaa/test_codig/test_coder_apply_back_hypotheses.py::TestH6_ApplyBackCopySubfolderListing::test_subfolder_apply_back_propagates_new_nested_file
toolboxv2/tests/test_isaa/test_codig/test_coder_apply_back_hypotheses.py::TestH7_AcceptFlowCleanupRace::test_apply_back_persists_before_cleanup
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestCustomTrigger::test_always_fire_evaluator
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestCustomTrigger::test_evaluator_exception_returns_false
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestCustomTrigger::test_unknown_trigger_returns_false
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireJob::test_fire_emits_completed_event
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireJob::test_fire_emits_failed_event
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireJob::test_fire_exception_sets_failed
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireJob::test_fire_persists_state_to_file
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireJob::test_fire_removes_from_firing_set_on_failure
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireJob::test_fire_removes_from_firing_set_on_success
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireJob::test_fire_timeout_sets_fail
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireJob::test_fire_updates_run_count_and_result
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireJobWithLiveState::test_live_state_finish_written_on_failure
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireJobWithLiveState::test_live_state_finish_written_on_success
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireMissedJobs::test_fires_missed_jobs_and_returns_count
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireMissedJobs::test_fires_multiple_missed
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestFireMissedJobs::test_returns_zero_when_nothing_missed
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestJobChainIntegration::test_a_completes_b_fires
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestJobChainIntegration::test_job_b_does_not_fire_on_wrong_event
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestJobScheduler::test_paused_job_not_fired_during_tick
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnCliLifecycleEvaluator::test_mark_pending_and_evaluate
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnCliLifecycleEvaluator::test_multiple_jobs_independent
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnCliLifecycleEvaluator::test_teardown_clears_pending
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnIntervalEvaluator::test_does_not_fire_when_recent
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnIntervalEvaluator::test_exactly_at_boundary
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnIntervalEvaluator::test_fires_when_elapsed
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnIntervalEvaluator::test_fires_when_never_run
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnIntervalEvaluator::test_no_interval_returns_false
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnJobEventEvaluator::test_job_chain_fires
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnJobEventEvaluator::test_paused_job_not_notified
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnJobEventEvaluator::test_wrong_event_type_does_not_fire
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnJobEventEvaluator::test_wrong_source_does_not_fire
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnTimeEvaluator::test_does_not_fire_future
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnTimeEvaluator::test_fires_exactly_at_boundary
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnTimeEvaluator::test_fires_when_past
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnTimeEvaluator::test_invalid_datetime_returns_false
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnTimeEvaluator::test_naive_datetime_treated_as_utc
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestOnTimeEvaluator::test_no_datetime_returns_false
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestSchedulerLifecycle::test_double_start_is_idempotent
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestSchedulerLifecycle::test_fire_lifecycle_on_cli_start
toolboxv2/tests/test_isaa/test_jobs/test_job_manager.py::TestSchedulerLifecycle::test_start_and_stop
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestFireLifecycleCliExit::test_cli_exit_job_fires
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestFireLifecycleCliExit::test_cli_start_does_not_fire_exit_jobs
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestLastFiredTsAfterFire::test_interval_does_not_refire_immediately_after_completion
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestLastFiredTsAfterFire::test_last_fired_ts_updated_after_failed_fire
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestLastFiredTsAfterFire::test_last_fired_ts_updated_after_successful_fire
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestMultiJobPersistenceRoundTrip::test_run_count_survives_reload
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnAgentIdleEvaluator::test_does_not_fire_when_active_recently
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnAgentIdleEvaluator::test_does_not_fire_without_activity_record
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnAgentIdleEvaluator::test_fires_when_agent_idle_long_enough
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnAgentIdleEvaluator::test_mark_activity_resets_idle_state
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnAgentIdleEvaluator::test_multiple_agents_tracked_independently
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnAgentIdleEvaluator::test_same_agent_two_jobs_throttled_independently
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnAgentIdleEvaluator::test_setup_and_teardown_do_not_raise
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnAgentIdleEvaluator::test_teardown_clears_per_job_check_entry
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnAgentIdleEvaluator::test_throttle_prevents_rapid_checks
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnCronEvaluatorEdgeCases::test_missing_cron_expression_returns_false
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnDreamEventEvaluator::test_all_four_event_types_work
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnDreamEventEvaluator::test_all_jobs_dict_cleaned_on_teardown
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnDreamEventEvaluator::test_consumed_after_evaluate
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnDreamEventEvaluator::test_fires_after_notify_matching_event
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnDreamEventEvaluator::test_multiple_jobs_same_event_all_notified
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnDreamEventEvaluator::test_not_fires_without_notify
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnDreamEventEvaluator::test_paused_job_not_notified
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnDreamEventEvaluator::test_setup_registers_job
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnDreamEventEvaluator::test_teardown_clears_all_pending
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnDreamEventEvaluator::test_wrong_event_type_does_not_fire
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnJobEventMultipleWatchers::test_chain_length_3
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnJobEventMultipleWatchers::test_multiple_watchers_on_same_source_all_notified
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnJobEventMultipleWatchers::test_one_watcher_per_source_does_not_affect_other_source
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnSystemShutdownEvaluatorInit::test_evaluate_before_setup_returns_false
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnSystemShutdownEvaluatorInit::test_multiple_setups_accumulate_job_ids
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnSystemShutdownEvaluatorInit::test_setup_initializes_job_ids
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnSystemShutdownEvaluatorInit::test_teardown_before_setup_does_not_raise
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnWebhookEvaluator::test_consumed_after_evaluate
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnWebhookEvaluator::test_fires_after_trigger_webhook
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnWebhookEvaluator::test_multiple_webhooks_independent
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnWebhookEvaluator::test_not_fires_without_trigger
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnWebhookEvaluator::test_scheduler_trigger_webhook_delegates
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnWebhookEvaluator::test_teardown_clears_pending
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestOnWebhookEvaluator::test_wrong_job_id_does_not_fire
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestSchedulerWebhookIntegration::test_webhook_job_fires_via_scheduler
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestSchedulerWebhookIntegration::test_webhook_not_fired_for_wrong_job_id
toolboxv2/tests/test_isaa/test_jobs/test_jobs_extended.py::TestTriggerConfigEmptyList::test_interval_zero_round_trip_handled_safely
toolboxv2/tests/test_isaa/test_jobs/test_jobs_offline_validation.py::TestFullOfflineOnlineCycle::test_job_file_updated_after_fire_contains_correct_timestamps
toolboxv2/tests/test_isaa/test_jobs/test_jobs_offline_validation.py::TestFullOfflineOnlineCycle::test_job_fires_after_simulated_offline_period
toolboxv2/tests/test_isaa/test_jobs/test_jobs_offline_validation.py::TestFullOfflineOnlineCycle::test_multiple_restarts_accumulate_run_count
toolboxv2/tests/test_isaa/test_jobs/test_jobs_offline_validation.py::TestFullOfflineOnlineCycle::test_on_time_job_fires_after_offline_period
toolboxv2/tests/test_isaa/test_jobs/test_jobs_offline_validation.py::TestSchedulerOfflineCatchup::test_fire_missed_failed_job_increments_fail_count
toolboxv2/tests/test_isaa/test_jobs/test_jobs_offline_validation.py::TestSchedulerOfflineCatchup::test_fire_missed_fires_due_job
toolboxv2/tests/test_isaa/test_jobs/test_jobs_offline_validation.py::TestSchedulerOfflineCatchup::test_fire_missed_fires_multiple_jobs
toolboxv2/tests/test_isaa/test_jobs/test_jobs_offline_validation.py::TestSchedulerOfflineCatchup::test_fire_missed_returns_zero_when_nothing_missed
toolboxv2/tests/test_isaa/test_jobs/test_jobs_offline_validation.py::TestSchedulerOfflineCatchup::test_fire_missed_updates_run_count
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_client.py::TestSyncClient::test_compute_diff_with_conflicts
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_client.py::TestSyncClient::test_compute_local_diff_missing_remote
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_client.py::TestSyncClient::test_compute_local_diff_new_files
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_client.py::TestSyncClient::test_download_checksum_verification
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_client.py::TestSyncClient::test_download_pipeline
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_client.py::TestSyncClient::test_download_uses_supplied_url_without_asking
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_client.py::TestSyncClient::test_handle_file_deleted_moves_to_trash
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_client.py::TestSyncClient::test_upload_pipeline
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_client.py::TestSyncClient::test_upload_without_url_grant_marks_pending
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestExportIndex::test_export_gzipped
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestExportIndex::test_import_gzipped_roundtrip
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestLocalIndex::test_delete_file
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestLocalIndex::test_get_all_checksums
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestLocalIndex::test_get_pending
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestLocalIndex::test_init_creates_tables
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestLocalIndex::test_set_sync_state
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestLocalIndex::test_upsert_and_get
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestLocalIndex::test_upsert_overwrites
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestSyncLog::test_log_for_share
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestSyncLog::test_log_limit
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_index.py::TestSyncLog::test_log_sync_event
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestConflictDetection::test_claiming_new_when_we_hold_a_version_is_a_conflict
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestConflictDetection::test_divergent_base_is_a_conflict
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestConflictDetection::test_legacy_client_without_base_is_not_flagged
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestConflictDetection::test_new_file_is_not_a_conflict
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestConflictDetection::test_sequential_update_is_not_a_conflict
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestServerModes::test_index_does_not_scan_the_vault
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestServerModes::test_replica_client_connects_to_loopback
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestServerModes::test_replica_without_stored_token_falls_back_to_relay
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestSyncServer::test_broadcast_handles_dead_connection
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestSyncServer::test_broadcast_skips_originator
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestSyncServer::test_conflict_detection
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestSyncServer::test_handle_file_changed_updates_index
toolboxv2/tests/test_mini/test_cloudm/test_LiveSync/test_LiveSync_server.py::TestSyncServer::test_init_opens_index_without_scanning
```
