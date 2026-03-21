# tb_browser — Chrome Extension

> **Location**: `toolboxv2/tb_browser/`
> **Build**: `npm run dev` / `npm run build`
> **Backend**: ToolBoxV2 local server on `localhost:8080`

A production-ready browser extension integrating the full ToolBoxV2 stack into Chrome. Provides gesture control, ISAA AI assistant, voice input, live page search, and a password manager with 2FA.

## Features

| Feature | Description |
|---------|-------------|
| 🎯 Gesture Control | Mouse/touch swipe navigation (back, forward, scroll) |
| 🤖 ISAA AI Assistant | Page-aware AI chat, web automation, form filling |
| 🎤 Voice Input | Speech-to-text, text-to-speech, wake words |
| 🔍 Live Search | Auto-index page content, in-page highlighting, section jump |
| 🔐 Password Manager | Auto-fill, TOTP 2FA, browser import, secure generation |
| 🎨 Glass Morphism UI | Black/white glass look, marine blue accents (#2E86AB) |

## Project Structure

```
tb_browser/
├── src/
│   ├── popup.html          ← Main popup (three-panel: ISAA / Search / Passwords)
│   ├── popup.js            ← UI management + panel switching
│   ├── popup.css           ← Glass morphism styling
│   ├── content.js          ← Page indexing, highlighting, gesture detection
│   ├── content.css         ← Content script styling
│   ├── background.js       ← Service worker, API communication
│   └── gesture-detector.js ← Swipe/touch recognition
├── icons/
├── assets/
├── build/                  ← Output directory (load this in Chrome)
├── manifest.json           ← Extension manifest (Manifest V3)
├── package.json
└── build.js
```

## Installation

### Development

```bash
cd toolboxv2/tb_browser
npm run dev
```

Then in Chrome: `chrome://extensions/` → Enable Developer Mode → Load Unpacked → select `build/`

### Production

```bash
npm run package     # Creates distributable ZIP
npm run clean       # Clean build directory
```

### Native Messaging Host

For full ToolBoxV2 integration (beyond the HTTP API), install the native messaging host:

```bash
tb -c tb_browser install_native_host
```

=== "Linux / macOS"
    Copies `toolbox_native_host.json` to `~/.config/google-chrome/NativeMessagingHosts/`

=== "Windows"
    Registers registry key: `HKCU\Software\Google\Chrome\NativeMessagingHosts\com.toolboxv2.host`

!!! warning "Disconnected Status"
    If the extension shows **Disconnected**, verify:

    1. `toolbox_native_host.json` has the **absolute path** to `toolbox_native_host.py`
    2. `toolbox_native_host.py` is executable (`chmod +x`)
    3. Python shebang matches your active virtualenv

## Native Messaging Protocol

4-byte little-endian length prefix + JSON:

```
[4 bytes LE uint32: message_length][JSON bytes]
```

Example message:

```json
{"type": "run_command", "mod": "CloudM", "fn": "session_status", "args": {}}
```

## API Integration

The extension connects to `localhost:8080`. Required endpoints:

| Endpoint | Purpose |
|----------|---------|
| `/api/isaa/mini_task_completion` | ISAA AI processing |
| `/api/call/PasswordManager/list_passwords` | List passwords |
| `/api/call/PasswordManager/generate_password` | Generate password |
| `/api/call/PasswordManager/get_password_for_autofill` | Auto-fill data |
| `/api/system/status` | Connection status check |

## Gesture Configuration

```javascript
{
  enabled: true,
  sensitivity: 1.0,
  minSwipeDistance: 100,
  enableMouse: true,
  enableTouch: true
}
```

| Gesture | Action |
|---------|--------|
| Swipe Left | Browser back |
| Swipe Right | Browser forward |
| Swipe Up | Scroll up (length-dependent speed) |
| Double-Click | Open popup (focus search) |

## Voice Configuration

```javascript
{
  enabled: true,
  language: 'en-US',
  autoSpeak: false,
  wakeWords: ['toolbox', 'isaa', 'computer', 'assistant']
}
```

Voice commands: `"Search [query]"`, `"Password fill"`, or any free-form query → ISAA.

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Alt + T` | Toggle popup |
| `Ctrl + Shift + V` | Activate voice |
| `Ctrl + Shift + F` | Quick page search |
| `Ctrl + Shift + P` | Auto-fill password |

## Test Suite

```bash
# All 132 tests
tb -c tb_browser test

# Unit tests (105)
python -m unittest discover toolboxv2/tb_browser/tests/unit/

# E2E tests (27) — requires Chrome + ChromeDriver
python -m unittest discover toolboxv2/tb_browser/tests/e2e/
```

## Browser Compatibility

| Browser | Min Version | Notes |
|---------|-------------|-------|
| Chrome | 88+ | Full support (Manifest V3) |
| Edge | 88+ | Full support (Chromium) |
| Firefox | 109+ | Requires manifest adaptation |
| Safari | — | Not supported |

## Troubleshooting

**Extension not loading** → Check ToolBoxV2 server is running on `localhost:8080`, check browser console.

**Gestures not working** → Verify gesture detection enabled in settings, refresh the page.

**Voice not responding** → Check microphone permissions, verify HTTPS context (or `localhost`).

**Password autofill failing** → Check server accessibility, verify domain entry exists.

## Related

- [ISAA Agent Framework](../mods/isaa/README.md) — AI backend powering ISAA chat in the extension
- [CloudM Login System](../mods/CloudM/login_system.md) — Session tokens used for API auth
- [Worker System](../workers/index.md) — Local server the extension talks to
