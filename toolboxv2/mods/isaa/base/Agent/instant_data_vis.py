"""
InstantDataVisualizer - Prompt System for LLM-Generated Data Visualization

Generates Python code that uses cli_printing utilities to visualize arbitrary dict data.
Designed for small LLMs (Qwen, Phi, Gemma) with One-Shot examples.

Usage:
    prompt = build_terminal_prompt(data_preview, data_type_hint)
    # Send to LLM, get code back, exec() it with data context
"""
import os

# =============================================================================
# SYSTEM PROMPT - Defines LLM role and constraints
# =============================================================================

SYSTEM_PROMPT_TERMINAL = """Du bist ein Code-Generator für Daten-Visualisierung im Terminal.

DEINE AUFGABE:
- Du erhältst eine Vorschau von Daten (erste ~250 Zeichen)
- Du schreibst Python-Code der diese Daten mit den bereitgestellten Utility-Funktionen visualisiert

REGELN:
1. NUR diese importierten Funktionen verwenden - KEINE anderen imports!
2. Die Variable `data` enthält die vollständigen Daten (nicht nur die Vorschau)
3. Code muss sofort ausführbar sein (kein if __name__, keine Funktionsdefinitionen)
4. Halte den Code minimal und direkt
5. Bei Listen: Iteriere und zeige relevante Felder
6. Bei verschachtelten Dicts: Nutze print_box_header für Sektionen
7. Zahlen/Scores: Verwende Farben (green=gut, yellow=mittel, red=schlecht)

VERFÜGBARE FUNKTIONEN:

```python
# Header/Footer für Sektionen
print_box_header(title: str, icon: str = "ℹ", width: int = 76)
print_box_footer(width: int = 76)

# Content mit Style
print_box_content(text: str, style: str = "", width: int = 76)
# styles: 'success', 'error', 'warning', 'info', '' (plain)

# Status-Zeilen mit Icons
print_status(message: str, status: str = "info")
# status: 'success', 'error', 'warning', 'info', 'progress', 'data', 'server', etc.

# Trennlinien
print_separator(char: str = "─", width: int = 76)

# Tabellen
print_table_header(columns: list, widths: list)
# columns = [("Name", 20), ("Value", 30), ...]
# widths = [20, 30, ...]

print_table_row(values: list, widths: list, styles: list = None)
# styles: ['white', 'grey', 'green', 'yellow', 'cyan', 'blue', 'red', 'magenta']

# Code/Config Blöcke
print_code_block(code: str, language: str = "text", show_line_numbers: bool = False)
# language: 'json', 'yaml', 'toml', 'env', 'text'
```

AUSGABE: Nur Python-Code, keine Erklärungen, keine Markdown-Blöcke."""

# =============================================================================
# ONE-SHOT EXAMPLES - Critical for small LLMs
# =============================================================================

EXAMPLE_INPUT_1 = '''{"users": [{"name": "Alice", "score": 95, "status": "active"}, {"name": "Bob", "score": 72, "status": "inactive"}], "total": 2}'''

EXAMPLE_OUTPUT_1 = '''print_box_header("Users Overview", "👥")
print_box_content(f"Total: {data.get('total', len(data.get('users', [])))}", "info")
print_box_footer()

columns = [("Name", 15), ("Score", 10), ("Status", 12)]
widths = [15, 10, 12]
print_table_header(columns, widths)

for user in data.get("users", []):
    name = str(user.get("name", "-"))
    score = user.get("score", 0)
    status = user.get("status", "-")

    score_style = "green" if score >= 80 else "yellow" if score >= 50 else "red"
    status_style = "green" if status == "active" else "grey"

    print_table_row([name, str(score), status], widths, ["white", score_style, status_style])'''

EXAMPLE_INPUT_2 = '''{"top_3_funde": [{"rang": 1, "titel": "Apple Siri 2.0", "details": {"modell": "Gemini 1.2T", "neue_features": ["Kontext", "Privacy"]}}'''

EXAMPLE_OUTPUT_2 = '''print_box_header("Top Funde", "🔍")

for item in data.get("top_3_funde", []):
    rang = item.get("rang", "?")
    titel = item.get("titel", "Unbekannt")

    print_status(f"#{rang}: {titel}", "info")

    details = item.get("details", {})
    if details:
        if "modell" in details:
            print_box_content(f"Modell: {details['modell']}", "info")

        features = details.get("neue_features", [])
        if features:
            print_box_content(f"Features: {', '.join(features[:3])}", "success")

    print_separator()

print_box_footer()'''

EXAMPLE_INPUT_3 = '''{"server": {"host": "0.0.0.0", "port": 8080, "status": "running"}, "metrics": {"cpu": 45, "memory": 72}}'''

EXAMPLE_OUTPUT_3 = '''print_box_header("Server Status", "🖥️")

server = data.get("server", {})
print_box_content(f"Host: {server.get('host', '-')}:{server.get('port', '-')}", "info")

status = server.get("status", "unknown")
status_style = "success" if status == "running" else "error"
print_box_content(f"Status: {status}", status_style)

print_separator()

metrics = data.get("metrics", {})
if metrics:
    print_status("Metrics", "data")
    cpu = metrics.get("cpu", 0)
    mem = metrics.get("memory", 0)

    cpu_style = "success" if cpu < 70 else "warning" if cpu < 90 else "error"
    mem_style = "success" if mem < 70 else "warning" if mem < 90 else "error"

    print_box_content(f"CPU: {cpu}%", cpu_style)
    print_box_content(f"Memory: {mem}%", mem_style)

print_box_footer()'''


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_terminal_prompt(
    data_preview: str,
    data_type_hint: str = None,
    max_preview_chars: int = 250,
    include_examples: int = 2
) -> list[dict]:
    """
    Build prompt messages for LLM to generate terminal visualization code.

    Args:
        data_preview: First N chars of str(data) or json.dumps(data)
        data_type_hint: Optional hint like "search_results", "metrics", "user_list"
        max_preview_chars: Truncate preview to this length
        include_examples: Number of one-shot examples (0-3), recommended: 2 for small LLMs

    Returns:
        List of message dicts for chat completion API
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
    """

    # Truncate preview
    preview = data_preview[:max_preview_chars]
    if len(data_preview) > max_preview_chars:
        preview += "..."

    messages = [{"role": "system", "content": SYSTEM_PROMPT_TERMINAL}]

    # Add one-shot examples
    examples = [
        (EXAMPLE_INPUT_1, EXAMPLE_OUTPUT_1),
        (EXAMPLE_INPUT_2, EXAMPLE_OUTPUT_2),
        (EXAMPLE_INPUT_3, EXAMPLE_OUTPUT_3),
    ]

    for i in range(min(include_examples, len(examples))):
        inp, out = examples[i]
        messages.append({"role": "user", "content": f"DATEN:\n{inp}"})
        messages.append({"role": "assistant", "content": out})

    # Build actual request
    user_content = f"DATEN:\n{preview}"
    if data_type_hint:
        user_content = f"TYP: {data_type_hint}\n{user_content}"

    messages.append({"role": "user", "content": user_content})

    return messages


# =============================================================================
# CODE EXECUTOR WITH FALLBACK
# =============================================================================

def execute_visualization(
    generated_code: str,
    data: dict,
    fallback_to_json: bool = True
) -> bool:
    """
    Execute LLM-generated visualization code safely.

    Args:
        generated_code: Python code string from LLM
        data: The actual data dict to visualize
        fallback_to_json: If True, print formatted JSON on error

    Returns:
        True if visualization succeeded, False if fallback was used
    """
    # Import utilities into execution context
    from toolboxv2.utils.clis.cli_printing import (
        print_box_header,
        print_box_footer,
        print_box_content,
        print_status,
        print_separator,
        print_table_header,
        print_table_row,
        print_code_block,
        c_print,
        Colors
    )

    # Build execution context
    exec_globals = {
        "data": data,
        "print_box_header": print_box_header,
        "print_box_footer": print_box_footer,
        "print_box_content": print_box_content,
        "print_status": print_status,
        "print_separator": print_separator,
        "print_table_header": print_table_header,
        "print_table_row": print_table_row,
        "print_code_block": print_code_block,
        "c_print": c_print,
        "Colors": Colors,
        # Minimal stdlib allowed
        "str": str,
        "int": int,
        "float": float,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "isinstance": isinstance,
        "list": list,
        "dict": dict,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
        "round": round,
    }

    try:
        # Clean code (remove markdown fences if LLM added them)
        clean_code = generated_code.strip()
        if clean_code.startswith("```"):
            lines = clean_code.split("\n")
            clean_code = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        exec(clean_code, exec_globals)
        return True

    except Exception as e:
        if fallback_to_json:
            import json
            print_box_header("Data (JSON Fallback)", "⚠")
            print_box_content(f"Visualization Error: {type(e).__name__}: {e}", "error")
            print_separator()
            try:
                formatted = json.dumps(data, indent=2, ensure_ascii=False, default=str)
                print_code_block(formatted, "json", show_line_numbers=False)
            except:
                print_code_block(str(data), "text")
            print_box_footer()
        return False


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

async def visualize_data_terminal(
    data: dict,
    agent: 'FlowAgent',  # Any client with .a_run_llm_completion()
    model: str = None,
    data_type_hint: str = None,
    include_examples: int = 2,
    max_preview_chars: int = 2000,
) -> bool:
    """
    Complete pipeline: Build prompt -> Call LLM -> Execute visualization

    Args:
        data: Dict to visualize
        agent: LLM client with chat completion capability
        model: Optional model override
        data_type_hint: Optional context hint
        include_examples: One-shot examples count
        max_preview_chars: Max characters in data preview

    Returns:
        True if LLM-generated viz worked, False if fallback
    """
    import json

    # Generate preview
    try:
        preview = json.dumps(data, ensure_ascii=False, default=str)
    except:
        preview = str(data)

    # Build prompt
    messages = build_terminal_prompt(
        data_preview=preview,
        data_type_hint=data_type_hint,
        include_examples=include_examples,
        max_preview_chars=max_preview_chars
    )

    # Call LLM (adapt to your client interface)
    # This is a generic example - adjust to your actual LLM client
    try:
        if hasattr(agent, 'a_run_llm_completion'):
            response = await agent.a_run_llm_completion(messages=messages, with_context=False, model=model or os.getenv("BLITZMODEL", os.getenv("FASTMODEL")))
        else:
            raise ValueError("LLM client must have .a_run_llm_completion() method")

        generated_code = response if isinstance(response, str) else response.get('content', str(response))

    except Exception as e:
        print_status(f"LLM Error: {e}", "error")
        return execute_visualization("", data, fallback_to_json=True)

    # Execute
    return execute_visualization(generated_code, data, fallback_to_json=True)


# =============================================================================
# PLACEHOLDER FOR HTML (Future Extension)
# =============================================================================

SYSTEM_PROMPT_HTML = """[RESERVED FOR HTML VISUALIZATION]
Same structure as terminal but generates HTML using html_utils functions.
Will be implemented when html_utils are consolidated.
"""


def build_html_prompt(data_preview: str, data_type_hint: str = None) -> list[dict]:
    """Placeholder for HTML visualization prompt builder."""
    raise NotImplementedError("HTML visualization coming soon - utilities being consolidated")


def visualize_data_html(data: dict, llm_client, **kwargs) -> str:
    """Placeholder for HTML visualization pipeline."""
    raise NotImplementedError("HTML visualization coming soon - utilities being consolidated")


# =============================================================================
# TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    # Demo without actual LLM - just show the prompt structure
    import json

    test_data = {
        "top_3_funde": [
            {
                "rang": 1,
                "titel": "Apple Siri 2.0 mit 1,2 Billionen Parameter",
                "details": {
                    "modell": "Google Gemini mit 1,2 Billionen Parametern",
                    "neue_features": [
                        "Kontextbewusstsein über mehrere Interaktionen",
                        "On-Device Processing für Privatsphäre",
                    ],
                    "relevanz": "Massiver Sprung für Consumer AI"
                }
            },
            {
                "rang": 2,
                "titel": "Agentic AI Framework",
                "details": {
                    "modell": "Multi-Agent System",
                    "neue_features": ["Self-coordination", "Task decomposition"],
                }
            }
        ]
    }

    preview = json.dumps(test_data, ensure_ascii=False)[:250]
    messages = build_terminal_prompt(preview, data_type_hint="search_results")

    print("=" * 80)
    print("GENERATED PROMPT STRUCTURE")
    print("=" * 80)

    for i, msg in enumerate(messages):
        role = msg['role'].upper()
        content = msg['content']
        if len(content) > 200:
            content = content[:200] + "..."
        print(f"\n[{i}] {role}:")
        print("-" * 40)
        print(content)

    print("\n" + "=" * 80)
    print("DEMO: Manual code execution with test data")
    print("=" * 80 + "\n")

    # Simulate what LLM would generate
    demo_code = EXAMPLE_OUTPUT_2

    # Execute (will fail without actual imports in this context)
    try:
        from toolboxv2.utils.clis.cli_printing import (
            print_box_header, print_box_footer, print_box_content,
            print_status, print_separator
        )

        data = test_data
        exec(demo_code)
    except ImportError:
        print("(Import failed - run within ToolBoxV2 context for full demo)")
        print("\nGenerated code would be:")
        print("-" * 40)
        print(demo_code)
