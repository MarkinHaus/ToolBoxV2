# FIX-26 – aus Karte K-API-21

Paket: **P3-VERALTET** (Test veraltet (Schnittstelle absichtlich geändert)) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 18 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Ist `llm_router` der einzige LLM-Weg (kein `llm_handler` mehr)? Soll `parse_media_from_query` das Feld `format` liefern?

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-26  (aus Karte K-API-21)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_isaa/test_base/test_agent/test_flow_agent_class.py, test_flowagent_streaming.py, test_agent_fetures.py   (nur diese ändern)
Vorher lesen:  `flow_agent.py:1000-1060, 1440-1480`, `llm_router/router.py`, `live_harness.py:800-830`
```

**Schritte:**

1. Einen gemeinsamen Test-Helfer bauen, der `CompletionRouter.complete`/`stream` mockt, und die Tests darauf umstellen.
2. `format`-Erwartung nach Antwort.

**Nicht tun:** `litellm` nicht per `sys.modules` stubben (ARCH-01). Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 18 IDs grün.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestCodeExecutorFeatureLive::test_A_loader_registers_feature
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestCodeExecutorFeatureLive::test_B_has_activation_function
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestCodeExecutorFeatureLive::test_C_has_deactivation_function
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestCodeExecutorFeatureLive::test_D_activation_callable
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestCodeExecutorFeatureLive::test_E_deactivation_callable
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestCodeExecutorFeatureLive::test_F_enable_calls_add_tools
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestCodeExecutorFeatureLive::test_G_registered_tools_have_tool_func
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestCodeExecutorFeatureLive::test_H_registered_tools_have_name
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestCodeExecutorFeatureLive::test_I_registered_tools_have_category
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestCodeExecutorFeatureLive::test_J_expected_tool_names_present
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestCodeExecutorFeatureLive::test_K_disable_calls_remove_tools
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestCodeExecutorFeatureLive::test_tool_health_checks
toolboxv2/tests/test_isaa/test_base/test_agent/test_flow_agent_class.py::TestFlowAgentLLM::test_a_run_llm_completion_max_tokens_triggers_auto_resume
toolboxv2/tests/test_isaa/test_base/test_agent/test_flow_agent_class.py::TestFlowAgentLLM::test_a_run_llm_completion_media_error_triggers_retry_without_media
toolboxv2/tests/test_isaa/test_base/test_agent/test_flow_agent_class.py::TestFlowAgentLLM::test_a_run_llm_completion_standard_text_returns_string
toolboxv2/tests/test_isaa/test_base/test_agent/test_flow_agent_class.py::TestMediaParser::test_parse_media_from_query_valid_image_returns_cleaned_query_and_vision_format
toolboxv2/tests/test_isaa/test_base/test_agent/test_flowagent_streaming.py::TestFlowAgentStreaming::test_a_stream_verbose_formats_text_correctly
toolboxv2/tests/test_isaa/test_base/test_agent/test_flowagent_streaming.py::TestFlowAgentStreaming::test_a_stream_yields_chunks_only_when_async_iterated
```
