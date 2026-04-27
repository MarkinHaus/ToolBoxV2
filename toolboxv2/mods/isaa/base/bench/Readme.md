# Bench — Binary LLM Benchmark Framework

## Quick Start

### 1. Single Model Test (CLI)

```bash
python -m bench run \
  --task-dir tasks/ \
  --model openrouter/google/gemini-2.5-flash \
  --model-id gemini-flash \
  -o report.json
```

### 2. Multi-Model Parallel Test

```bash
python -m bench.multirun \
  --task-dir tasks/ \
  --output-dir reports/ \
  gemini=openrouter/google/gemini-2.5-flash \
  haiku=openrouter/anthropic/claude-haiku-4.5 \
  qwen=openrouter/qwen/qwen3-8b
```

Each model runs in its own process. Each writes its own `reports/<model>.json`.
A combined `dashboard.html` is generated automatically.

### 3. Python API

```python
from bench.multirun import multi_benchmark, aggregate_dashboard

models = {
    "gemini-flash": "openrouter/google/gemini-2.5-flash",
    "claude-haiku": "openrouter/anthropic/claude-haiku-4.5",
}

paths = multi_benchmark(
    models=models,
    task_dir="tasks/",
    output_dir="reports/",
    adapter_type="row",      # "row" = direct LiteLLM, "agent" = ISAA FlowAgent
    timeout=90,
    seed=42,                 # fixed seed for reproducible comparisons
    skip_existing=True,      # won't re-run models that already have reports
)

aggregate_dashboard(paths, "comparison.html")
```

---

## Writing Tasks

Tasks are YAML files. Each file = one task. Place them in `tasks/` or any subdirectory.

### Minimal Task

```yaml
id: logic-001
prompt: "What is 17 * 24? Reply with only the number."
checks:
  - type: contains
    value: "408"
```

### Full Task

```yaml
id: extract-json-001
complexity: extended          # tutorial | extended | phd
modality: [text]              # text, image, document, video, audio
tags: [extraction, json]      # used for filtering and dashboard grouping
prompt: |
  Extract the person's name and age from this text.
  Reply in JSON format: {"name": "...", "age": ...}

  "Hi, I'm Clara and I turned 31 last month."
ground_truth: '{"name": "Clara", "age": 31}'
checks:
  - type: json_valid
  - type: json_has_key
    key: name
  - type: json_has_key
    key: age
  - type: contains
    value: "Clara"
  - type: contains
    value: "31"
```

### Multimodal Task

```yaml
id: vision-001
complexity: tutorial
modality: [text, image]       # model must support image to get this task
tags: [vision]
prompt: "What animal is in the image?"
attachments:
  - type: image
    path: tasks/vision/cat.jpg
checks:
  - type: any_of
    values: ["cat", "kitten", "feline"]
```

The framework auto-skips multimodal tasks for text-only models.
Attachments are injected as `[media:path]` — ISAA parses this automatically.

### Using the Judge

For checks where systematic validation isn't possible, use the LLM judge:

```yaml
checks:
  - type: judge
    question: "Does the response correctly refuse to answer about a future event?"
  - type: judge_compare
    ground_truth: "The capital of France is Paris."
    question: "Does the response convey the same factual answer?"
```

The judge is called via `isaa.format_class` and forced to answer yes/no.
Before using the judge in production runs, calibrate it (see below).

### Agent Honesty Tasks

For agent-mode benchmarks, verify tool usage and response grounding:

```yaml
id: agent-search-001
complexity: extended
modality: [text]
tags: [agent, honesty, search]
prompt: "Search for the current weather in Berlin and tell me the temperature."
checks:
  - type: tool_called
    name: "search"
  - type: tool_called_with
    name: "search"
    arg: "query"
    value: "weather berlin"
  - type: tool_result_in_response
    name: "search"
  - type: no_hallucination
  - type: no_uncertainty
```

Honesty validators require tool_calls in TaskContext. The `AgentAdapter` and `AgentStreamAdapter` extract these automatically from the ExecutionEngine.

---

## Available Validators

| Name | Params | Description |
|------|--------|-------------|
| `contains` | `value` | Response contains value (case-insensitive) |
| `not_contains` | `value` | Response does NOT contain value |
| `equals` | `value` | Response equals value (trimmed, case-insensitive) |
| `regex` | `pattern` | Response matches regex |
| `char_count_gte` | `value` | Response ≥ N characters |
| `char_count_lte` | `value` | Response ≤ N characters |
| `max_tokens` | `value` | Response ≤ N tokens (whitespace split) |
| `json_valid` | — | Response is valid JSON |
| `json_has_key` | `key` | Response JSON contains key |
| `tool_calls_lte` | `value` | Number of tool calls ≤ N |
| `tool_calls_gte` | `value` | Number of tool calls ≥ N |
| `file_exists` | `path` | File was created (sandbox or filesystem) |
| `latency_lte` | `value` | Execution time ≤ N seconds |
| `any_of` | `values` | Contains at least one of the values |
| `all_of` | `values` | Contains all of the values |
| `none_of` | `values` | Contains none of the values |
| `judge` | `question` | LLM judge answers yes/no |
| `judge_compare` | `ground_truth`, `question` | Compare response against ground truth via judge |
| **Honesty (Agent)** | | |
| `tool_called` | `name` | Tool was called at least once |
| `tool_not_called` | `name` | Tool was NOT called |
| `tool_called_n` | `name`, `count` | Tool called exactly N times |
| `tool_called_with` | `name`, `arg`, `value` | Tool called with specific argument value |
| `tool_result_in_response` | `name` | Tool result content appears in response |
| `tool_order` | `names` | Tools called in specified order (subsequence) |
| `no_hallucination` | — | All facts traceable to tool results or prompt |
| `admits_uncertainty` | — | Agent admits when it doesn't know (DE+EN) |
| `no_uncertainty` | — | Agent does NOT express uncertainty |
| `response_uses_tool_data` | — | Response references data from any tool call |

---

## Suites

Group tasks into named suites. Suite files go in `tasks/suites/`.

```yaml
# tasks/suites/quick.yaml
id: quick
name: Quick Smoke Test
tasks:                        # explicit task IDs
  - logic-001
  - extract-json-001
```

```yaml
# tasks/suites/logic.yaml
id: logic-full
name: All Logic Tasks
task_pattern: "logic-*"       # glob match on task IDs
```

```yaml
# tasks/suites/by-tag.yaml
id: extraction
name: Extraction Suite
tags_filter: [extraction]     # tasks must have ALL listed tags
```

Use suites via CLI: `python -m bench run --suite tasks/suites/quick.yaml ...`

---

## Judge Calibration (Schicht 0)

Before using an LLM as judge, verify it can correctly evaluate known answers.

### 1. Create Ground Truth Tasks

Place them in `tasks/calibration/`. Each needs a `ground_truth` field:

```yaml
id: cal-001
complexity: tutorial
prompt: "What is 5 + 7?"
ground_truth: "12"
checks:
  - type: contains
    value: "12"
```

Create tasks across all three complexity levels (tutorial / extended / phd).

### 2. Run Calibration

```bash
python -m bench calibrate --task-dir tasks/calibration/
```

Output: `judge_profile.json` with per-complexity accuracy and optimal batch sizes.

```json
{
  "model": "gpt-4o",
  "disqualified": false,
  "batch_sizes": { "tutorial": 32, "extended": 8, "phd": 2 },
  "accuracy": { "tutorial": 1.0, "extended": 0.97, "phd": 0.95 }
}
```

If accuracy drops below 95% at any level, the judge is `disqualified`.

---

## Adapters

| Adapter | What it Tests | Use When |
|---------|--------------|----------|
| `RowModelAdapter` | Raw model via LiteLLM | Testing model intelligence in isolation |
| `AgentAdapter` | ISAA FlowAgent (a_run) | Testing agent orchestration |
| `AgentStreamAdapter` | ISAA FlowAgent streaming | Testing streaming behavior |
| `MAKERAdapter` | FlowAgent accomplish | Testing complex task completion |

Switch adapter via `--adapter row` or `--adapter agent` in CLI.

---

## Scoring

Every check is **pass or fail**. No scales, no 1-10 ratings.

- Task score = `passed_checks / total_checks` (0.0 to 1.0)
- Report score = average of all task scores
- Dashboard shows scores as 0-100%

Tags act as dimensions in the dashboard — if you tag tasks with `logic`, `extraction`, `honesty`, the dashboard shows per-dimension breakdowns automatically.

---

## Dashboard

Generate from report JSONs:

```bash
python -m bench dashboard reports/*.json -o comparison.html
```

Features: leaderboard, dimension comparison chart, cost overview, per-task drill-down, flag analysis.

---

## Tips

- Use `skip_existing=True` (default) to avoid re-running models. Delete the JSON to re-test.
- Use a fixed `seed` when comparing models to get reproducible task ordering.
- Start with `tutorial` tasks, add `extended` and `phd` as you build confidence.
- The circuit breaker aborts after 5 consecutive errors — the model is probably down.
- Reports are plain JSON files. You can edit, merge, or filter them with `jq`.
