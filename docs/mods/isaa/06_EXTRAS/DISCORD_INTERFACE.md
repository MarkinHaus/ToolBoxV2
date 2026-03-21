# ISAA Discord Interface

## Übersicht

Das Discord Interface ermöglicht die Steuerung von ISAA über Discord.

## Setup

```python
from isaa_mod.extras.discord_interface import DiscordInterface

interface = DiscordInterface(
    token=\"YOUR_DISCORD_BOT_TOKEN\"
)

# Mit ISAA verbinden
interface.connect(app.get_mod(\"isaa\"))

# Starten
await interface.start()
```

## Features

- **Voice Mode** - Sprachsteuerung
- **VFS Search** - Dateien durchsuchen
- **CLI Extension** - Bot-Commands

## Commands

| Command | Beschreibung |
|---------|-------------|
| `!ask <frage>` | Frage an Agent |
| `!search <term>` | VFS Suche |
| `!status` | Agent Status |
