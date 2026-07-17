# Adaptive Prompt System

> **Flow**: `toolboxv2/flows/adaptive_prompt_system.py`
> **Zweck**: System-optimierte Prompt-Generierung durch LLM-Discovery, Profiling und Prompt-Engineering-Techniken (CoT, ToT, Adversarial Validation).

## Übersicht

Das Modul implementiert eine 2-Phase Adaptive-Prompt-Pipeline:

1. **Discovery**: Sendet Benchmark-Prompts an ein Ziel-LLM, analysiert die Antworten und erstellt ein `SystemProfile`.
2. **Generation**: Nutzt das Profil um Meta-Prompts zu bauen, die auf die Stärken, Format-Präferenzen und das Context-Window des Zielsystems zugeschnitten sind.

Eine `PromptLibrary` persistiert Templates und Profile auf Disk.

## Architektur

Drei Schichten:

| Schicht | Klassen | Zweck |
|---------|---------|-------|
| Data Models | `LLMSystem` (Enum), `SystemProfile`, `PromptTemplate` | Datencontainer |
| Core Logic | `PromptLibrary`, `AdaptivePromptGenerator` | Discovery + Generation |
| UI | `ui()` → `_build_ui()`, `_render_*` Helpers | Web-UI Komponenten |

## Klassen

### `LLMSystem(Enum)`

Bekannte LLM-Systeme mit Charakteristiken.

| Value | String |
|-------|--------|
| `CLAUDE` | `"claude"` |
| `CHATGPT` | `"chatgpt"` |
| `GPT4` | `"gpt4"` |
| `GEMINI` | `"gemini"` |
| `LLAMA` | `"llama"` |
| `MISTRAL` | `"mistral"` |
| `UNKNOWN` | `"unknown"` |

### `SystemProfile`

Profil eines LLM-Systems basierend auf Discovery.

| Feld | Typ | Default | Beschreibung |
|------|-----|---------|-------------|
| `system_type` | `LLMSystem` | `UNKNOWN` | Erkanntes LLM-System |
| `model_name` | `str` | `""` | Model-Name |
| `capabilities` | `list[str]` | `[]` | Erkannte Fähigkeiten |
| `limitations` | `list[str]` | `[]` | Erkannte Limitierungen |
| `preferred_format` | `str` | `"markdown"` | Bevorzugtes Output-Format |
| `supports_system_prompt` | `bool` | `True` | System-Prompts unterstützt? |
| `supports_tools` | `bool` | `False` | Tool-Use unterstützt? |
| `supports_vision` | `bool` | `False` | Vision/Input unterstützt? |
| `supports_code_execution` | `bool` | `False` | Code-Execution unterstützt? |
| `context_window` | `int` | `8000` | Approx. Context-Window (Tokens) |
| `strengths` | `list[str]` | `[]` | Identifizierte Stärken |
| `raw_discovery_response` | `str` | `""` | Roher Discovery-Output |
| `discovered_at` | `str` | `datetime.now().isoformat()` | Zeitstempel |

### `PromptTemplate`

Template für system-spezifische Prompts.

| Feld | Typ | Default | Beschreibung |
|------|-----|---------|-------------|
| `name` | `str` | — | Template-Name |
| `description` | `str` | — | Beschreibung |
| `persona` | `str` | — | Persona-Definition |
| `context_template` | `str` | — | Context-Template-String |
| `format_instructions` | `str` | — | Output-Format-Anweisungen |
| `chain_of_thought` | `bool` | `False` | CoT-Technik aktiviert |
| `tree_of_thoughts` | `bool` | `False` | ToT-Technik aktiviert |
| `adversarial_validation` | `bool` | `False` | Adversarial Validation aktiviert |
| `few_shot_examples` | `list[dict]` | `[]` | Few-Shot Beispiel-Paare |
| `system_type` | `LLMSystem` | `UNKNOWN` | Ziel-System |
| `tags` | `list[str]` | `[]` | Suchbare Tags |

### `PromptLibrary`

Bibliothek für system-spezifische Prompt-Templates. Speichert Templates und Profile in `templates.json` und `profiles.json`.

| Methode | Signatur | Beschreibung |
|---------|----------|-------------|
| `__init__` | `(storage_path="./prompt_library")` | Initialisiert, lädt von Disk |
| `save_library` | `() → None` | Speichert Templates + Profile |
| `add_template` | `(template) → None` | Template hinzufügen |
| `get_template` | `(name, system_type=UNKNOWN) → PromptTemplate \| None` | Template holen (mit Fallback) |
| `add_profile` | `(name, profile) → None` | Profil speichern |
| `get_profile` | `(name) → SystemProfile \| None` | Profil holen |

### `AdaptivePromptGenerator`

Hauptklasse für die 2-Way Prompt-Generierung.

| Methode | Signatur | Beschreibung |
|---------|----------|-------------|
| `__init__` | `(library: PromptLibrary)` | Initialisiert mit Library |
| `discover_system` | `(isaa, target_interface, profile_name="default") → SystemProfile` | **Phase 1**: Benchmark-Prompts senden, Antworten analysieren, Profil erstellen |
| `generate_adaptive_prompt` | `(isaa, user_request, profile, use_cot, use_tot, use_adversarial, custom_persona, few_shot_examples) → dict[str,str]` | **Phase 2**: Meta-Prompt bauen, gibt `{"system_prompt", "user_prompt"}` zurück |

## Funktionen

### `run()` — Main Entry Point

```python
async def run(
    app, args_sto,
    mode: str = "interactive",      # "interactive" | "quick" | "discover"
    user_request: str = "",
    profile_name: str = "default",
    use_cot: bool = True,
    use_tot: bool = False,
    use_adversarial: bool = False,
    custom_persona: str | None = None,
    storage_path: str = "./prompt_library",
    target_system: str = "claude",
    **kwargs,
) -> dict[str, Any]
```

- **`interactive`**: CLI-Menü mit Mode-Selection
- **`quick`**: Direkte Generierung mit vordefiniertem Profil
- **`discover`**: System-Discovery aus einer gegebenen Response

### Hilfsfunktionen

| Funktion | Beschreibung |
|----------|-------------|
| `_create_quick_profile(system)` | Vordefiniertes Profil für bekannte Systeme |
| `_build_technique_instructions(cot, tot, adversarial)` | Technik-Anweisungs-Strings generieren |
| `get_discovery_prompt(prompt_type="identity")` | Discovery-Prompt aus `DISCOVERY_PROMPTS` dict |
| `create_few_shot_example(input, output)` | Few-Shot Beispiel erstellen |
| `_get_instances(storage_path)` | Singleton `PromptLibrary` + `AdaptivePromptGenerator` |

## UI

`ui(view)` ist der Entry-Point für die Web-UI. Baut Tabs für Quick Mode, Discovery Mode, Full Flow und Profiles.

## Used By

- `flows/chain.py` — `run`, `cmd_run`
- `flows/bg.py` — `run`
- `flows/core0.py` — `run`
- `flows/docker.py` — `run`, `run_image_build`
