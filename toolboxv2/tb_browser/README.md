# ToolBox Pro - Browser Extension

A production-ready browser extension with advanced gesture control, ISAA AI assistant, voice input, live search, and secure password management with 2FA support.

## 🚀 Features

### 🎯 Gesture Detection
- **Left Swipe**: Navigate back in browser history
- **Right Swipe**: Navigate forward in browser history  
- **Swipe Up**: Dynamic scroll up (length-dependent)
- **Double-Click**: Open ToolBox popup with focus on search

### 🤖 ISAA AI Assistant
- **Page Interaction**: Ask ISAA about current page content
- **Web Automation**: Navigate, download files, export data
- **Task Completion**: Form filling, data extraction
- **Context Awareness**: Understands current page and user intent

### 🎤 Voice Input System
- **Minimalistic Design**: Non-visual disturbing microphone icon
- **Voice-to-Text**: Convert speech to search queries
- **Text-to-Speech**: Direct audio feedback from ISAA
- **Wake Words**: "toolbox", "isaa", "computer", "assistant"

### 🔍 Live Search & Page Indexing
- **Auto-Indexing**: Automatically index page content on load
- **In-Page Highlighting**: Visual highlighting of search results
- **Section Navigation**: Scroll to specific sections
- **ISAA Integration**: Ask ISAA about specific page sections

### 🔐 Password Manager with 2FA
- **Auto-fill**: Intelligent form detection and password filling
- **2FA Codes**: TOTP support with countdown timer
- **Browser Import**: Import passwords from Chrome, Firefox, etc.
- **Secure Generation**: Cryptographically secure password generation

### 🎨 Glass Morphism UI
- **Modern Design**: Black and white glass look
- **Marine Blue Accents**: Subtle blue accent colors (#2E86AB)
- **Three-Panel Interface**: ISAA Chat, Live Search, Password Manager
- **Smooth Transitions**: Fluid animations and interactions

## 📦 Installation

### Development Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MarkinHaus/ToolBoxV2.git
   cd ToolBoxV2/toolboxv2/tb_browser
   ```

2. **Build the extension**:
   ```bash
   npm run dev
   ```

3. **Load in Chrome**:
   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select the `build` folder

### Production Installation

1. **Build production version**:
   ```bash
   npm run package
   ```

2. **Install the generated ZIP file** in your browser's extension store

## 🛠️ Development

### Project Structure

```
tb_browser/
├── src/
│   ├── popup.html          # Main popup interface
│   ├── popup.js            # Popup logic and UI management
│   ├── popup.css           # Glass morphism styling
│   ├── content.js          # Page interaction and indexing
│   ├── content.css         # Content script styling
│   ├── background.js       # Service worker and API communication
│   └── gesture-detector.js # Advanced gesture recognition
├── icons/                  # Extension icons
├── assets/                 # Additional assets
├── build/                  # Built extension files
├── manifest.json           # Extension manifest
├── package.json           # Build configuration
├── build.js               # Build system
└── README.md              # This file
```

### Build Commands

- `npm run dev` - Development build (no minification)
- `npm run build` - Production build with minification
- `npm run package` - Create distributable ZIP package
- `npm run clean` - Clean build directory

### API Integration

The extension connects to the ToolBox backend server running on `localhost:8080`. Ensure the ToolBox server is running for full functionality.

#### Required API Endpoints:
- `/api/isaa/mini_task_completion` - ISAA AI processing
- `/api/call/PasswordManager/list_passwords` - Password management
- `/api/call/PasswordManager/generate_password` - Password generation
- `/api/call/PasswordManager/get_password_for_autofill` - Auto-fill data
- `/api/system/status` - Connection status

## 🎮 Usage

### Keyboard Shortcuts
- `Alt + T` - Toggle ToolBox popup
- `Ctrl + Shift + V` - Activate voice command
- `Ctrl + Shift + F` - Quick page search
- `Ctrl + Shift + P` - Auto-fill password

### Gesture Controls
- **Touch/Mouse Gestures**: Swipe in any direction for navigation
- **Double-Click**: Anywhere on page to open popup
- **Voice Activation**: Click microphone or use keyboard shortcut

### Voice Commands
- "Search [query]" - Switch to search tab and search
- "Password fill" - Auto-fill password for current site
- General queries - Send to ISAA AI assistant

## 🔧 Configuration

### Gesture Settings
```javascript
{
  enabled: true,
  sensitivity: 1.0,
  minSwipeDistance: 100,
  enableMouse: true,
  enableTouch: true
}
```

### Voice Settings
```javascript
{
  enabled: true,
  language: 'en-US',
  autoSpeak: false,
  wakeWords: ['toolbox', 'isaa']
}
```

### Password Settings
```javascript
{
  autoFill: true,
  generateLength: 16,
  includeSymbols: true
}
```

## 🔒 Security

- **Encrypted Storage**: Passwords encrypted with device keys
- **Secure Communication**: HTTPS API calls to ToolBox server
- **Permission Model**: Minimal required permissions
- **Content Security Policy**: Strict CSP for popup and content scripts

## 🌐 Browser Compatibility

- **Chrome**: 88+ (Manifest V3 support)
- **Edge**: 88+ (Chromium-based)
- **Firefox**: 109+ (with manifest adaptation)
- **Safari**: Not supported (different extension system)

## 🐛 Troubleshooting

### Common Issues

1. **Extension not loading**:
   - Check if ToolBox server is running on localhost:8080
   - Verify all files are present in build directory
   - Check browser console for errors

2. **Gestures not working**:
   - Ensure gesture detection is enabled in settings
   - Check if page allows gesture events
   - Try refreshing the page

3. **Voice input not responding**:
   - Check microphone permissions
   - Verify browser supports Web Speech API
   - Ensure HTTPS context for voice features

4. **Password autofill failing**:
   - Check if ToolBox server is accessible
   - Verify password exists for current domain
   - Ensure form fields are properly detected

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For support and bug reports, please visit:
- GitHub Issues: https://github.com/MarkinHaus/ToolBoxV2/issues
- Documentation: https://toolbox.app/docs
- Community: https://discord.gg/toolbox
