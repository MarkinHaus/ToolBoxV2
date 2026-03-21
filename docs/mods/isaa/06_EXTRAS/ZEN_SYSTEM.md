# ISAA Zen System

## Übersicht

Zen ist das Meta-Learning und Dream-System für ISAA.

## Komponenten

| Komponente | Datei | Beschreibung |
|------------|-------|---------------|
| `ZenPlus` | `zen_plus.py` | Haupt-Zen Engine |
| `DreamZenAdapter` | `dream_zen_adapter.py` | Dream-Integration |
| `ZenRenderer` | `zen_renderer.py` | Output Rendering |

## Usage

```python
from isaa_mod.extras.zen import ZenPlus

zen = ZenPlus()

# Dream erstellen
dream = await zen.create_dream(
    context=\"Analyse von User-Verhalten\",
    depth=\"deep\"
)

# Traum verarbeiten
result = await zen.process(dream)
```
