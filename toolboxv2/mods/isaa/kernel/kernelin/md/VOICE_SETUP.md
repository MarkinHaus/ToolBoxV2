# Discord Voice Support Setup

## Overview

The Discord Kernel supports voice channels with full voice input/transcription, allowing the bot to:
- Join and leave voice channels
- **Listen to voice input and transcribe in real-time** (NEW!)
- Track voice state changes (users joining/leaving/moving)
- Monitor voice connection status
- Automatic speech-to-text using Groq Whisper API

## Requirements

### 1. Basic Voice Support (Join/Leave Channels)

Voice support requires the **PyNaCl** library for voice encryption.

```bash
pip install -U discord.py[voice]
```

This automatically installs:
- `discord.py` - Discord API wrapper
- `PyNaCl` - Voice encryption library
- `cffi` - C Foreign Function Interface

### 2. Voice Input/Transcription Support (NEW!)

For real-time voice transcription, you need:

```bash
pip install discord-ext-voice-recv groq
```

This installs:
- `discord-ext-voice-recv` - Voice receive extension for discord.py
- `groq` - Groq API client for Whisper transcription

### 3. Groq API Key (for Voice Transcription)

Get a free API key from [Groq Console](https://console.groq.com/):

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

Or add to your `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. TTS (Text-to-Speech) Support

#### Option A: ElevenLabs (Cloud, High Quality)

```bash
pip install elevenlabs
export ELEVENLABS_API_KEY="your_elevenlabs_api_key"
```

Get API key from [ElevenLabs](https://elevenlabs.io/)

#### Option B: Piper (Local, Free)

Download Piper from [GitHub](https://github.com/rhasspy/piper/releases)

```bash
# Set path to piper.exe
export PIPER_PATH="C:\Users\Markin\Workspace\piper_w\piper.exe"
```

Or use default path: `C:\Users\Markin\Workspace\piper_w\piper.exe`

### System Dependencies (Linux)

On Debian/Ubuntu-based systems, you may need:
```bash
sudo apt install libffi-dev libnacl-dev python3-dev
```

On Fedora/RHEL-based systems:
```bash
sudo dnf install libffi-devel libsodium-devel python3-devel
```

## Verification

After installation, restart your bot. You should see:
```
✓ Discord Kernel initialized (instance: default)
```

Instead of:
```
⚠️ PyNaCl not installed. Voice support disabled.
```

## Available Voice Commands

Once voice support is enabled, the following commands are available:

### `!join`
Joins the voice channel you're currently in.
```
User: !join
Bot: 🔊 Joined General Voice
```

### `!leave`
Leaves the current voice channel.
```
User: !leave
Bot: 👋 Left the voice channel
```

### `!voice_status`
Shows detailed voice connection status.
```
User: !voice_status
Bot: [Embed showing]:
     - Channel: General Voice
     - Connected: ✅
     - Playing: ❌
     - Paused: ❌
     - Latency: 45.23ms
     - Listening: ✅ (if voice input is active)
```

### `!listen` (NEW!)
Starts listening to voice input and transcribing in real-time.
```
User: !listen
Bot: 🎤 Started listening! Speak and I'll transcribe your voice in real-time.

[User speaks: "Hello, how are you?"]
Bot processes: "Hello, how are you?" as text input
```

**Features:**
- Real-time transcription using Groq Whisper (whisper-large-v3-turbo)
- Automatic language detection
- Multi-user support (tracks each speaker)
- Transcriptions sent directly to kernel as user input
- Transcription interval: 3 seconds

### `!stop_listening` (NEW!)
Stops listening to voice input.
```
User: !stop_listening
Bot: 🔇 Stopped listening to voice input.
```

### `!tts` (NEW!)
Toggle Text-to-Speech on/off and select TTS engine.
```
User: !tts
Bot: 🔊 TTS is disabled

User: !tts piper
Bot: 🔊 TTS enabled with piper

User: !tts elevenlabs
Bot: 🔊 TTS enabled with elevenlabs

User: !tts off
Bot: 🔇 TTS disabled
```

**Features:**
- Agent responses are spoken in voice channel
- Two TTS engines: ElevenLabs (cloud) or Piper (local)
- ElevenLabs: High quality, natural voices
- Piper: Free, local, fast
- Automatic voice playback when enabled

## Voice Events & VAD (Voice Activity Detection)

The kernel automatically tracks voice state changes with advanced VAD:

- **User joins voice channel**: Sends signal to kernel
- **User leaves voice channel**: Sends signal to kernel
- **User moves between channels**: Sends signal to kernel
- **User starts speaking**: VAD detects speech start (🎤)
- **User stops speaking**: VAD detects speech end (🔇)
- **Voice input received**: Transcribed and sent as user input signal

**Voice Activity Detection Features:**
- Real-time detection when users start/stop speaking
- Automatic transcription trigger on speech end
- Multi-speaker separation (tracks each user separately)
- Silence detection (1 second threshold)
- Per-user audio buffering

These events are logged and can be used for:
- Presence tracking
- Activity monitoring
- Automated responses
- Voice-based interactions
- Speaker identification in group calls

## How Voice Transcription Works

1. **User speaks** in voice channel
2. **Discord sends audio** to bot (PCM format, 48kHz stereo)
3. **Audio is buffered** for 3 seconds per user
4. **Audio is converted** to WAV format
5. **Groq Whisper API** transcribes the audio
6. **Transcription is sent** to kernel as user input
7. **Agent processes** the text and responds

**Performance:**
- Groq Whisper (whisper-large-v3-turbo) is extremely fast (~0.1-0.5s per 3s audio)
- Near real-time transcription
- Supports 99+ languages with automatic detection

## Troubleshooting

### "PyNaCl is not installed" Warning

**Solution**: Install discord.py with voice support:
```bash
pip install -U discord.py[voice]
```

### "discord-ext-voice-recv not installed" Warning

**Solution**: Install voice receive extension:
```bash
pip install discord-ext-voice-recv
```

### "Groq not installed" Warning

**Solution**: Install Groq client:
```bash
pip install groq
```

### "GROQ_API_KEY not set" Warning

**Solution**: Set environment variable:
```bash
export GROQ_API_KEY="your_api_key"
```

Or add to `.env` file in project root.

### "Voice support is not available" Error

**Cause**: PyNaCl is not installed or failed to load.

**Solution**:
1. Verify installation: `pip list | grep PyNaCl`
2. Reinstall: `pip install --force-reinstall PyNaCl`
3. Check system dependencies (see above)

### Voice input not working

**Possible causes:**
1. **Missing dependencies**: Install `discord-ext-voice-recv` and `groq`
2. **No API key**: Set `GROQ_API_KEY` environment variable
3. **Not listening**: Use `!listen` command after joining voice channel
4. **No audio**: Check microphone permissions in Discord

### Transcription is slow or inaccurate

**Solutions:**
- Groq Whisper is very fast (~0.1-0.5s), if slow check internet connection
- For better accuracy, speak clearly and reduce background noise
- Transcription interval is 3 seconds, adjust in code if needed

### Permission Errors

**Cause**: Bot lacks voice permissions.

**Solution**: Ensure bot has these permissions:
- `Connect` - Join voice channels
- `Speak` - Send audio (future feature)
- `Use Voice Activity` - Voice detection

### Can bot join DM voice channels?

**Yes!** The bot can join DM voice channels if invited by a user.

**How it works:**
1. User creates/joins a DM voice channel
2. User invites the bot (or bot is already in DM)
3. User types `!join` in DM
4. Bot joins the DM voice channel

**What bots CAN do:**
- Join guild (server) voice channels
- Join DM voice channels if invited by a user
- Listen to voice input in any joined channel (guild or DM)
- Speak in voice channels (TTS)
- Transcribe voice input in real-time

**What bots CANNOT do:**
- Initiate private calls (Discord API limitation)
- Call users directly
- Create DM voice channels

## Complete Setup Example

```bash
# 1. Install all dependencies
pip install -U discord.py[voice] discord-ext-voice-recv groq elevenlabs

# 2. Set API keys
export GROQ_API_KEY="gsk_your_api_key_here"
export ELEVENLABS_API_KEY="your_elevenlabs_api_key"  # Optional
export PIPER_PATH="C:\Users\Markin\Workspace\piper_w\piper.exe"  # Optional

# 3. Set Discord bot token
export DISCORD_BOT_TOKEN="your_discord_bot_token"

# 4. Run the bot
python -m toolboxv2.mods.isaa.kernel.kernelin.kernelin_discord

# 5. In Discord (Guild/Server):
# - Join a voice channel
# - Type: !join
# - Type: !listen
# - Type: !tts piper  (or !tts elevenlabs)
# - Start speaking!
# - Bot transcribes your voice and speaks responses

# 6. In Discord (DM):
# - Create/join a DM voice channel
# - Type: !join
# - Type: !listen
# - Start speaking!
```

## Feature Summary

| Feature | Status | Description |
|---------|--------|-------------|
| **Voice Join/Leave** | ✅ Implemented | Join/leave guild and DM voice channels |
| **Voice Input** | ✅ Implemented | Real-time voice transcription |
| **Voice Activity Detection** | ✅ Implemented | Detects when users start/stop speaking |
| **Multi-Speaker Separation** | ✅ Implemented | Tracks each speaker separately |
| **Text-to-Speech (TTS)** | ✅ Implemented | ElevenLabs + Piper support |
| **DM Voice Channels** | ✅ Implemented | Join DM voice channels |
| **Groq Whisper** | ✅ Implemented | Fast, accurate transcription |
| **Auto Language Detection** | ✅ Implemented | 99+ languages |
| **Initiate Private Calls** | ❌ Not Possible | Discord API limitation |

## Future Features

Planned voice features:
- 🎵 Music playback (YouTube, Spotify)
- 🎧 Audio effects and filters
- 📻 Audio streaming
- 🎙️ Advanced speaker diarization (AI-based)
- 🌍 Real-time translation
- 🎬 Voice cloning
- 🎮 Voice commands (wake word detection)

## Technical Details

### Voice Client Management

Each guild (server) has its own voice client:
```python
self.output_router.voice_clients[guild_id] = voice_client
```

### Voice State Tracking

Voice states are tracked via `on_voice_state_update` event:
- Before state: Previous voice channel
- After state: Current voice channel
- Signals sent to kernel for processing

### Intents Required

Voice support requires these intents:
```python
intents = discord.Intents.default()
intents.voice_states = True  # Track voice state changes
```

## Support

For issues or questions:
1. Check this documentation
2. Verify PyNaCl installation
3. Check bot permissions
4. Review Discord API status

## References

- [discord.py Voice Documentation](https://discordpy.readthedocs.io/en/stable/api.html#voice-related)
- [PyNaCl Documentation](https://pynacl.readthedocs.io/)
- [Discord Voice API](https://discord.com/developers/docs/topics/voice-connections)

