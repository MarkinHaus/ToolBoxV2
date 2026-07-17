# Style & Terminal Output (`utils/extras/Style.py`)

> **File:** `toolboxv2/utils/extras/Style.py` (~667 Zeilen)
> **Typ:** Reference
> ANSI-Farben, Spinner, Terminal-Clear, JSON-Extraction.

## Why This Matters

`Style` ist überall. Jede CLI-Ausgabe, jeder Logger-Eintrag, jedes ISAA-Tool nutzt `Style.RED()`, `Style.GREEN()`, `Style.BOLD()` für Terminal-Formatierung. Ohne `Style` gäbe es keine Farb-Ausgabe.

## API Reference

### Color Methods (Static)

Jede Methode nimmt `text: str` und gibt ANSI-formatierten String zurück.

| Category | Methods |
|----------|---------|
| **Red** | `RED`, `LIGHTRED`, `DARKRED`, `REDBG`, `REDBG2` |
| **Green** | `GREEN`, `LIGHTGREEN`, `DARKGREEN`, `GREENBG`, `GREENBG2` |
| **Blue** | `BLUE`, `LIGHTBLUE`, `DARKBLUE`, `BLUEBG`, `BLUEBG2` |
| **Yellow** | `YELLOW`, `LIGHTYELLOW`, `YELLOWBG`, `YELLOWBG2` |
| **Purple** | `PURPLE`, `LIGHTPURPLE`, `PURPLEBG`, `PURPLEBG2` |
| **Cyan** | `CYAN`, `CYANBG` |
| **White/Grey** | `WHITE`, `GREY`, `BEIGE`, `BEIGEBG`, `BEIGEBG2` |
| **Orange** | `ORANGE`, `ORANGEBG`, `ORANGEBG2` |
| **Base** | `BOLD`, `UNDERLINE`, `MIXED`, `END` |

```python
from toolboxv2.utils.extras.Style import Style

print(Style.RED("Error: connection failed"))
print(Style.GREEN("✓ Done") + " " + Style.GREY("(0.3s)"))
print(Style.BOLD(Style.UNDERLINE("Title")))
```

### Utility Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `cls` | `() → str` | Clear screen ANSI sequence |
| `print_to_console` | `(text)` | Strip ANSI + print to console |
| `remove_styles` | `(text: str) → str` | Remove all ANSI codes from text |
| `ansi_to_rich` | `(text: str) → str` | Convert ANSI → Rich markup |

### SpinnerManager

| Method | Description |
|--------|-------------|
| `start(label)` | Start animated spinner with label |
| `stop()` | Stop spinner, clear line |
| `update(label)` | Change spinner label |

```python
from toolboxv2.utils.extras.Style import SpinnerManager

spinner = SpinnerManager()
spinner.start("Loading modules...")
# ... long operation ...
spinner.stop()
```

### JSONExtractor

Extracts JSON from mixed text output (e.g., LLM responses with surrounding prose):

| Method | Description |
|--------|-------------|
| `extract(text) → dict?` | Find first valid JSON block in text |
| `extract_all(text) → list[dict]` | Find all JSON blocks |

## Common Pitfalls

- **Windows CMD**: Old `cmd.exe` doesn't support ANSI. Use Windows Terminal or `tb` in modern shell.
- **Chaining**: `Style.RED(Style.BOLD("text"))` works, but order matters — outer tag closes first.
- **Non-string input**: Methods call `str(text)` internally, but `None` becomes literal `"None"`.

## Used By

- Literally everything — `tb` CLI, ISAA agents, CloudM, logging

## Related

- [Core Types](types.md) — `AppType.logger` uses Style
- [All Functions Enums](all_functions_enums.md) — Mod dispatch
