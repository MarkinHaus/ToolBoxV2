⏳ Initializing session...
⏳ Preparing execution...
⏳ Initializing (skills + tools + personas)...

🔄 Iter 1/3

🔧 Nutze Tool: think
[32m##[0m[32m Situation[0m[32m Assessment[0m[32m

[0m[32mThe[0m[32m agent[0m[32m has[0m[32m completed[0m[32m its[0m[32m analysis[0m[32m in[0m[32m `[0m[32mthink[0m[32m()`[0m[32m and[0m[32m identified[0m[32m all[0m[32m classes[0m[32m,[0m[32m methods[0m[32m,[0m[32m and[0m[32m functions[0m[32m.[0m[32m It[0m[32m has[0m[32m everything[0m[32m needed[0m[32m —[0m[32m the[0m[32m full[0m[32m code[0m[32m index[0m[32m with[0m[32m signatures[0m[32m,[0m[32m doc[0m[32mstrings[0m[32m,[0m[32m and[0m[32m the[0m[32m dependency[0m[32m graph[0m[32m are[0m[32m all[0m[32m in[0m[32m the[0m[32m conversation[0m[32m.[0m[32m No[0m[32m additional[0m[32m file[0m[32m reads[0m[32m required[0m[32m.

[0m[32m##[0m[32m Key[0m[32m Insights[0m[32m

[0m[32m-[0m[32m All[0m[32m [0m[32m5[0m[32m data[0m[32m model[0m[32m classes[0m[32m are[0m[32m clearly[0m[32m defined[0m[32m with[0m[32m complete[0m[32m field[0m[32m lists[0m[32m in[0m[32m the[0m[32m index[0m[32m.
[0m[32m-[0m[32m The[0m[32m upstream[0m[32m dependency[0m[32m list[0m[32m looks[0m[32m like[0m[32m noise[0m[32m ([0m[32mtest[0m[32m functions[0m[32m,[0m[32m unrelated[0m[32m modules[0m[32m)[0m[32m —[0m[32m likely[0m[32m a[0m[32m graph[0m[32m artifact[0m[32m.[0m[32m Only[0m[32m the[0m[32m downstream[0m[32m list[0m[32m ([0m[32mflows[0m[32m that[0m[32m call[0m[32m `[0m[32mrun[0m[32m`)[0m[32m is[0m[32m meaningful[0m[32m.
[0m[32m-[0m[32m The[0m[32m module[0m[32m has[0m[32m three[0m[32m layers[0m[32m:[0m[32m data[0m[32m models[0m[32m ([0m[32mlines[0m[32m [0m[32m34[0m[32m-[0m[32m76[0m[32m),[0m[32m core[0m[32m logic[0m[32m ([0m[32mlines[0m[32m [0m[32m79[0m[32m-[0m[32m5[0m[32m21[0m[32m,[0m[32m [0m[32m5[0m[32m33[0m[32m-[0m[32m110[0m[32m3[0m[32m),[0m[32m and[0m[32m UI[0m[32m ([0m[32mlines[0m[32m [0m[32m120[0m[32m3[0m[32m-[0m[32m171[0m[32m2[0m[32m).
[0m[32m-[0m[32m Inner[0m[32m UI[0m[32m callback[0m[32m functions[0m[32m (`[0m[32mset[0m[32m_mode[0m[32m_quick[0m[32m`,[0m[32m `[0m[32mprocess[0m[32m_dis[0m[32mcovery[0m[32m`,[0m[32m etc[0m[32m.)[0m[32m are[0m[32m nested[0m[32m inside[0m[32m `_[0m[32mbuild[0m[32m_ui[0m[32m`[0m[32m —[0m[32m they[0m[32m are[0m[32m NOT[0m[32m top[0m[32m-level[0m[32m functions[0m[32m despite[0m[32m the[0m[32m index[0m[32m listing[0m[32m them[0m[32m separately[0m[32m.

[0m[32m##[0m[32m Concrete[0m[32m Tips[0m[32m

[0m[32m1[0m[32m.[0m[32m Call[0m[32m `[0m[32mfinal[0m[32m_answer[0m[32m()`[0m[32m now[0m[32m with[0m[32m the[0m[32m complete[0m[32m markdown[0m[32m document[0m[32m.[0m[32m No[0m[32m more[0m[32m tool[0m[32m calls[0m[32m needed[0m[32m.
[0m[32m2[0m[32m.[0m[32m Group[0m[32m the[0m[32m UI[0m[32m rendering[0m[32m functions[0m[32m (`[0m[32m_[0m[32mrender[0m[32m_*[0m[32m`)[0m[32m and[0m[32m UI[0m[32m helpers[0m[32m (`[0m[32m_[0m[32mtechn[0m[32mique[0m[32m_card[0m[32m`)[0m[32m into[0m[32m a[0m[32m single[0m[32m "[0m[32mUI[0m[32m Rendering[0m[32m"[0m[32m subsection[0m[32m —[0m[32m don[0m[32m't[0m[32m list[0m[32m each[0m[32m one[0m[32m as[0m[32m a[0m[32m top[0m[32m-level[0m[32m function[0m[32m entry[0m[32m.
[0m[32m3[0m[32m.[0m[32m Do[0m[32m NOT[0m[32m list[0m[32m the[0m[32m nested[0m[32m callback[0m[32m closures[0m[32m (`[0m[32mset[0m[32m_mode[0m[32m_quick[0m[32m`,[0m[32m `[0m[32mset[0m[32m_mode[0m[32m_dis[0m[32mcovery[0m[32m`,[0m[32m `[0m[32mstart[0m[32m_dis[0m[32mcovery[0m[32m`,[0m[32m `[0m[32mprocess[0m[32m_dis[0m[32mcovery[0m[32m`,[0m[32m `[0m[32mgenerate[0m[32m_prompt[0m[32m`,[0m[32m `[0m[32mreset[0m[32m_ui[0m[32m`,[0m[32m `[0m[32mswitch[0m[32m_to[0m[32m_dis[0m[32mcovery[0m[32m`)[0m[32m as[0m[32m separate[0m[32m functions[0m[32m.[0m[32m They[0m[32m are[0m[32m implementation[0m[32m details[0m[32m of[0m[32m `_[0m[32mbuild[0m[32m_ui[0m[32m`.
[0m[32m4[0m[32m.[0m[32m For[0m[32m the[0m[32m upstream[0m[32m dependencies[0m[32m,[0m[32m note[0m[32m they[0m[32m appear[0m[32m to[0m[32m be[0m[32m index[0m[32m noise[0m[32m —[0m[32m say[0m[32m "[0m[32mNo[0m[32m direct[0m[32m upstream[0m[32m dependencies[0m[32m identified[0m[32m"[0m[32m or[0m[32m list[0m[32m them[0m[32m with[0m[32m a[0m[32m caveat[0m[32m.

[0m[32m##[0m[32m Partial[0m[32m Solution[0m[32m /[0m[32m H[0m[32mints[0m[32m

[0m[32mThe[0m[32m document[0m[32m structure[0m[32m should[0m[32m be[0m[32m:

[0m[32m``[0m[32m`
[0m[32m#[0m[32m adaptive[0m[32m_prompt[0m[32m_system[0m[32m

[0m[32m##[0m[32m Overview[0m[32m
[0m[32m2[0m[32m-[0m[32m3[0m[32m sentences[0m[32m:[0m[32m adaptive[0m[32m prompt[0m[32m generation[0m[32m with[0m[32m L[0m[32mLM[0m[32m system[0m[32m discovery[0m[32m,[0m[32m profiling[0m[32m,[0m[32m and[0m[32m technique[0m[32m injection[0m[32m ([0m[32mCo[0m[32mT[0m[32m/[0m[32mTo[0m[32mT[0m[32m/ad[0m[32mvers[0m[32marial[0m[32m).[0m[32m Dual[0m[32m interface[0m[32m:[0m[32m CLI[0m[32m and[0m[32m web[0m[32m UI[0m[32m.

[0m[32m##[0m[32m Classes[0m[32m
[0m[32m###[0m[32m L[0m[32mLM[0m[32mSystem[0m[32m(Enum[0m[32m)[0m[32m —[0m[32m table[0m[32m of[0m[32m values[0m[32m
[0m[32m###[0m[32m System[0m[32mProfile[0m[32m —[0m[32m field[0m[32m table[0m[32m from[0m[32m data[0m[32mclass[0m[32m
[0m[32m###[0m[32m Prompt[0m[32mTemplate[0m[32m —[0m[32m field[0m[32m table[0m[32m from[0m[32m data[0m[32mclass[0m[32m
[0m[32m###[0m[32m Prompt[0m[32mLibrary[0m[32m —[0m[32m method[0m[32m table[0m[32m ([0m[32mexclude[0m[32m _[0m[32mload[0m[32m_library[0m[32m or[0m[32m mark[0m[32m private[0m[32m)
[0m[32m###[0m[32m Adaptive[0m[32mPrompt[0m[32mGenerator[0m[32m —[0m[32m method[0m[32m table[0m[32m with[0m[32m signatures[0m[32m

[0m[32m##[0m[32m Functions[0m[32m
[0m[32m###[0m[32m run[0m[32m(...)[0m[32m —[0m[32m the[0m[32m main[0m[32m entry[0m[32m point[0m[32m
[0m[32m###[0m[32m _[0m[32mgenerate[0m[32m_opt[0m[32mimized[0m[32m_prompt[0m[32m(...)
[0m[32m###[0m[32m _[0m[32mcreate[0m[32m_quick[0m[32m_profile[0m[32m(...)
[0m[32m###[0m[32m _[0m[32mbuild[0m[32m_te[0m[32mchn[0m[32mique[0m[32m_instructions[0m[32m(...)
[0m[32m###[0m[32m get[0m[32m_dis[0m[32mcovery[0m[32m_prompt[0m[32m(...)
[0m[32m###[0m[32m create[0m[32m_f[0m[32mew[0m[32m_shot[0m[32m_example[0m[32m(...)
[0m[32m###[0m[32m _[0m[32mprint[0m[32m_result[0m[32m(...)
[0m[32m###[0m[32m _[0m[32mprofile[0m[32m_to[0m[32m_dict[0m[32m(...)
[0m[32m###[0m[32m _[0m[32mget[0m[32m_instances[0m[32m(...)
[0m[32m###[0m[32m ui[0m[32m(view[0m[32m)[0m[32m —[0m[32m UI[0m[32m entry[0m[32m point[0m[32m

[0m[32m##[0m[32m UI[0m[32m Rendering[0m[32m ([0m[32mPrivate[0m[32m)
[0m[32mB[0m[32mul[0m[32mleted[0m[32m list[0m[32m of[0m[32m _[0m[32mrender[0m[32m_*[0m[32m and[0m[32m _[0m[32mtechn[0m[32mique[0m[32m_card[0m[32m —[0m[32m no[0m[32m individual[0m[32m tables[0m[32m needed[0m[32m.

[0m[32m##[0m[32m Dependencies[0m[32m
[0m[32mUp[0m[32mstream[0m[32m:[0m[32m note[0m[32m the[0m[32m list[0m[32m appears[0m[32m unreliable[0m[32m
[0m[32mDown[0m[32mstream[0m[32m:[0m[32m list[0m[32m the[0m[32m flow[0m[32m modules[0m[32m

[0m[32m##[0m[32m Used[0m[32m By[0m[32m
[0m[32mList[0m[32m from[0m[32m downstream[0m[32m graph[0m[32m
[0m[32m``[0m[32m`

[0m[32m##[0m[32m Pit[0m[32mfalls[0m[32m to[0m[32m Avoid[0m[32m

[0m[32m-[0m[32m **[0m[32mDo[0m[32m not[0m[32m invent[0m[32m base[0m[32m classes[0m[32m**[0m[32m —[0m[32m `[0m[32mSystem[0m[32mProfile[0m[32m`[0m[32m and[0m[32m `[0m[32mPrompt[0m[32mTemplate[0m[32m`[0m[32m show[0m[32m `[0m[32m@[0m[32mdata[0m[32mclass[0m[32m`[0m[32m style[0m[32m fields[0m[32m but[0m[32m no[0m[32m explicit[0m[32m base[0m[32m class[0m[32m is[0m[32m in[0m[32m the[0m[32m index[0m[32m.[0m[32m Don[0m[32m't[0m[32m assume[0m[32m `[0m[32mdata[0m[32mclass[0m[32m`[0m[32m decorator[0m[32m is[0m[32m present[0m[32m unless[0m[32m you[0m[32m see[0m[32m it[0m[32m.
[0m[32m-[0m[32m **[0m[32mDo[0m[32m not[0m[32m duplicate[0m[32m nested[0m[32m callbacks[0m[32m**[0m[32m as[0m[32m top[0m[32m-level[0m[32m functions[0m[32m.
[0m[32m-[0m[32m **[0m[32mDo[0m[32m not[0m[32m include[0m[32m truncated[0m[32m code[0m[32m**[0m[32m in[0m[32m the[0m[32m doc[0m[32m —[0m[32m only[0m[32m reference[0m[32m signatures[0m[32m and[0m[32m doc[0m[32mstrings[0m[32m.
[0m[32m-[0m[32m **[0m[32mDo[0m[32m not[0m[32m describe[0m[32m `[0m[32mmock[0m[32m_interface[0m[32m`[0m[32m**[0m[32m —[0m[32m it[0m[32m's[0m[32m a[0m[32m local[0m[32m closure[0m[32m inside[0m[32m `[0m[32mrun[0m[32m`,[0m[32m not[0m[32m a[0m[32m module[0m[32m-level[0m[32m function[0m[32m worth[0m[32m documenting[0m[32m.[0m   ✓ Ergebnis: ## Situation Assessment

The agent has completed its analysis in `think()` and identified all classes, methods, and functions. It has everything needed — the full code index with signatures, docstring...

🔄 Iter 2/4
💭 Analyse des Codes abgeschlossen, Dokumentation erforderlich
🔧 Nutze Tool: final_answer


📝 
# Adaptive Prompt System

Generates system-optimized prompts by discovering LLM capabilities, building profiles, and injecting prompt-engineering techniques (Chain of Thought, Tree of Thoughts, Adversarial Validation). Provides both an interactive CLI and a web UI.

## Overview

The module implements a two-phase adaptive prompt pipeline: **Discovery** sends benchmark prompts to a target LLM and analyzes responses to build a `SystemProfile`; **Generation** uses that profile to construct meta-prompts tailored to the target system's strengths, format preferences, and context window. A `PromptLibrary` persists templates and profiles to disk. The module is structured in three layers: data models (`LLMSystem`, `SystemProfile`, `PromptTemplate`), core logic (`PromptLibrary`, `AdaptivePromptGenerator`, `run`), and UI (rendering helpers + `ui` entry point).

## Classes

### `LLMSystem(Enum)`

Known LLM systems with their characteristics.

| Value | String |
|-------|--------|
| `CLAUDE` | `"claude"` |
| `CHATGPT` | `"chatgpt"` |
| `GPT4` | `"gpt4"` |
| `GEMINI` | `"gemini"` |
| `LLAMA` | `"llama"` |
| `MISTRAL` | `"mistral"` |
| `UNKNOWN` | `"unknown"` |

---

### `SystemProfile`

Profil eines LLM-Systems basierend auf Discovery.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `system_type` | `LLMSystem` | `UNKNOWN` | Detected LLM system |
| `model_name` | `str` | `""` | Model name |
| `capabilities` | `list[str]` | `[]` | Detected capabilities |
| `limitations` | `list[str]` | `[]` | Detected limitations |
| `preferred_format` | `str` | `"markdown"` | Preferred output format |
| `supports_system_prompt` | `bool` | `True` | Whether system prompts are supported |
| `supports_tools` | `bool` | `False` | Whether tool use is supported |
| `supports_vision` | `bool` | `False` | Whether vision/image input is supported |
| `supports_code_execution` | `bool` | `False` | Whether code execution is supported |
| `context_window` | `int` | `8000` | Approximate context window in tokens |
| `strengths` | `list[str]` | `[]` | Identified strengths |
| `raw_discovery_response` | `str` | `""` | Raw discovery output |
| `discovered_at` | `str` | `datetime.now().isoformat()` | Timestamp of discovery |

---

### `PromptTemplate`

Template für system-spezifische Prompts.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | — | Template name |
| `description` | `str` | — | Template description |
| `persona` | `str` | — | Persona definition |
| `context_template` | `str` | — | Context template string |
| `format_instructions` | `str` | — | Output format instructions |
| `chain_of_thought` | `bool` | `False` | CoT technique flag |
| `tree_of_thoughts` | `bool` | `False` | ToT technique flag |
| `adversarial_validation` | `bool` | `False` | Adversarial validation flag |
| `few_shot_examples` | `list[dict]` | `[]` | Few-shot example pairs |
| `system_type` | `LLMSystem` | `UNKNOWN` | Target system type |
| `tags` | `list[str]` | `[]` | Searchable tags |

---

### `PromptLibrary`

Bibliothek für system-spezifische Prompt-Templates. Speichert und verwaltet optimierte Prompts pro LLM-System.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(storage_path: str = "./prompt_library")` | Initializes library, loads existing data from disk |
| `_load_library` | `() → None` | Lädt gespeicherte Templates und Profile from JSON files |
| `save_library` | `() → None` | Speichert Templates und Profile to JSON files |
| `add_template` | `(template: PromptTemplate) → None` | Fügt ein neues Template hinzu |
| `get_template` | `(name: str, system_type: LLMSystem = UNKNOWN) → PromptTemplate \| None` | Holt ein Template für ein bestimmtes System; falls back to generic template |
| `add_profile` | `(name: str, profile: SystemProfile) → None` | Speichert ein System-Profil |
| `get_profile` | `(name: str) → SystemProfile \| None` | Holt ein gespeichertes System-Profil |

Storage is backed by `templates.json` and `profiles.json` in `storage_path`.

---

### `AdaptivePromptGenerator`

Hauptklasse für die 2-Way Prompt-Generierung. Flow: Discovery → Profile Creation → Prompt Adaptation.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(library: PromptLibrary)` | Initializes with a PromptLibrary reference |
| `discover_system` | `(isaa, target_interface: Callable[[str], str], profile_name: str = "default") → SystemProfile` | STEP 1: Sends benchmark prompts to the target system, analyzes responses via ISAA, saves the resulting profile |
| `_analyze_discovery` | `(isaa, discovery_response: str) → SystemProfile` | Parses the combined discovery response into a structured `SystemProfile` using an LLM-based JSON extraction |
| `generate_adaptive_prompt` | `(isaa, user_request: str, profile: SystemProfile \| None, use_cot: bool, use_tot: bool, use_adversarial: bool, custom_persona: str \| None, few_shot_examples: list[dict] \| None) → dict[str, str]` | STEP 2: Builds a meta-prompt incorporating the system profile and selected techniques, returns `{"system_prompt", "user_prompt"}` |
| `_build_meta_prompt` | `(user_request, profile, use_cot, use_tot, use_adversarial, custom_persona, few_shot_examples) → str` | Constructs the full meta-prompt for the prompt-engineering agent |

## Functions

### `run(app, args_sto, mode, user_request, profile_name, ...) → dict[str, Any]`

```python
async def run(
    app, args_sto,
    mode: str = "interactive",
    user_request: str = "",
    profile_name: str = "default",
    target_response: Optional[str] = None,
    use_cot: bool = True,
    use_tot: bool = False,
    use_adversarial: bool = False,
    custom_persona: Optional[str] = None,
    few_shot_examples: Optional[list[dict]] = None,
    storage_path: str = "./prompt_library",
    target_system: str = "claude",
    **kwargs,
) -> dict[str, Any]:
```

Main entry point. Supports `"interactive"` (CLI menu loop), `"quick"` (direct generation with a predefined profile), and `"discover"` (system discovery from a provided response).

---

### `_generate_optimized_prompt(isaa, user_request, profile, use_cot, use_tot, use_adversarial, custom_persona) → dict[str, str]`

```python
async def _generate_optimized_prompt(
    isaa, user_request: str, profile: SystemProfile,
    use_cot: bool = True, use_tot: bool = False,
    use_adversarial: bool = False,
    custom_persona: Optional[str] = None,
) -> dict[str, str]:
```

Generiert einen optimierten Prompt mit verbessertem Meta-Prompt. Builds a German-language meta-prompt incorporating the system profile and technique instructions, dispatches it to the `AdaptivePromptGenerator` agent.

---

### `_create_quick_profile(system: str) → SystemProfile`

Erstellt ein vordefiniertes Profil basierend auf bekannten System-Charakteristiken. Returns a `SystemProfile` pre-populated with known capabilities for `"claude"`, `"chatgpt"`, `"gemini"`, `"llama"`, `"mistral"`, or a generic default.

---

### `_build_technique_instructions(use_cot: bool, use_tot: bool, use_adversarial: bool) → str`

Baut Technik-Anweisungen. Returns formatted instruction strings for each enabled technique (CoT, ToT, Adversarial), or a fallback message if none are active.

---

### `_get_instances(storage_path: str = "./prompt_library") → tuple[PromptLibrary, AdaptivePromptGenerator]`

Initialisiert oder holt globale Instanzen. Returns singleton `PromptLibrary` and `AdaptivePromptGenerator` instances (lazy-initialized on first call).

---

### `get_discovery_prompt(prompt_type: str = "identity") → str`

Gibt einen Discovery-Prompt zurück. Looks up the prompt from the module-level `DISCOVERY_PROMPTS` dict.

---

### `create_few_shot_example(input_text: str, output_text: str) → dict`

Erstellt ein Few-Shot Beispiel. Returns `{"input": input_text, "output": output_text}`.

---

### `_profile_to_dict(profile: SystemProfile) → dict`

Konvertiert SystemProfile zu dict. Selects key fields (`system_type`, `model_name`, `capabilities`, `limitations`, `strengths`, `preferred_format`, `context_window`, `supports_tools`, `supports_vision`) for serialization.

---

### `_print_result(result: dict, profile: SystemProfile) → None`

Gibt das Ergebnis formatiert aus. Prints the generated system prompt, user prompt, and reasoning to stdout.

---

### `ui(view) → Component`

Custom UI für Adaptive Prompt System. Entry point for the web UI; delegates to `_build_ui(view)` with view context management.

---

### UI Rendering Helpers (Private)

These functions construct specific tabs and states for the web UI:

- **`_build_ui(view)`** — Baut die eigentliche UI. Assembles all tabs, registers callbacks, and returns the layout.
- **`_render_quick_mode(view, vals, generate_handler)`** — Quick Generate Tab with target system selector, request input, and technique checkboxes.
- **`_render_discovery_mode(view, vals, step, start_handler, process_handler)`** — Discovery Tab with prompt type selector and instructions.
- **`_render_full_flow_mode(view, vals)`** — Full Flow Tab combining discovery and generation.
- **`_render_profiles_mode(view, switch_discovery_handler)`** — Profiles Tab for viewing saved profiles.
- **`_render_success(view, header, vals, reset_handler)`** — Success State mit Ergebnis-Anzeige. Renders the generated prompt or discovered profile.
- **`_technique_card(name: str, description: str, enabled: bool = False)`** — Rendert eine Technik-Karte. UI component for a single technique toggle.

## Dependencies

No direct upstream dependencies identified. The upstream entries in the index appear to be graph artifacts from unrelated modules.

## Used By

- `run` in `toolboxv2/flows/auto_git_commit.py`
- `run` in `toolboxv2/flows/bg.py`
- `run` in `toolboxv2/flows/bgws.py`
- `run` in `toolboxv2/flows/bottelUp.py`
- `run` in `toolboxv2/flows/chain.py`
- `cmd_run` in `toolboxv2/flows/chain.py`
- `run` in `toolboxv2/flows/core0.py`
- `run` in `toolboxv2/flows/docker.py`
- `run_image_build` in `toolboxv2/flows/docker.py`

💾 Saving context...

✅ Fertig