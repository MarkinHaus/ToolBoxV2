# ToolBox Pro Browser Extension - Installation Guide

## 🚀 Quick Start

### Prerequisites
1. **ToolBox Server**: Ensure ToolBox server is running on `localhost:8080`
2. **Chrome Browser**: Version 88+ (Manifest V3 support required)
3. **Node.js**: Version 14+ (for building from source)

### Installation Steps

#### Option 1: Load Development Build (Recommended for Testing)

1. **Navigate to the extension directory**:
   ```bash
   cd ToolBoxV2/toolboxv2/tb_browser
   ```

2. **Build the extension**:
   ```bash
   node build.js dev
   ```

3. **Load in Chrome**:
   - Open Chrome and navigate to `chrome://extensions/`
   - Enable "Developer mode" (toggle in top-right corner)
   - Click "Load unpacked"
   - Select the `build` folder inside `tb_browser`

4. **Verify Installation**:
   - Extension icon should appear in Chrome toolbar
   - Click the icon to open the popup
   - Check connection status (should show "Connected" if ToolBox server is running)

#### Option 2: Production Build

1. **Create production build**:
   ```bash
   node build.js build
   ```

2. **Create distribution package**:
   ```bash
   node build.js zip
   ```

3. **Install the ZIP file** through Chrome Web Store or load as unpacked extension

## 🔧 Configuration

### Initial Setup

1. **Open the extension popup** (click the ToolBox icon)
2. **Check connection status** - should show "Connected" if ToolBox server is running
3. **Test basic functionality**:
   - Try voice input (microphone icon)
   - Switch between tabs (ISAA Chat, Live Search, Passwords)
   - Test gesture detection (double-click on any page)

### Permissions

The extension requires the following permissions:
- **Storage**: For saving settings and cached data
- **Active Tab**: For interacting with current page
- **Scripting**: For content script injection
- **Notifications**: For user feedback
- **Context Menus**: For right-click functionality
- **Web Navigation**: For gesture-based navigation
- **Clipboard**: For copying passwords and TOTP codes

### Settings Configuration

Access settings through the extension popup or Chrome's extension management page:

#### Gesture Settings
```javascript
{
  enabled: true,
  sensitivity: 1.0,
  minSwipeDistance: 100,
  enableMouse: true,
  enableTouch: true
}
```

#### Voice Settings
```javascript
{
  enabled: true,
  language: 'en-US',
  autoSpeak: false,
  wakeWords: ['toolbox', 'isaa']
}
```

#### Password Settings
```javascript
{
  autoFill: true,
  generateLength: 16,
  includeSymbols: true
}
```

## 🧪 Testing

### Using the Test Page

1. **Open the test page**:
   ```bash
   # From the tb_browser directory
   open test.html
   # or navigate to file:///path/to/tb_browser/test.html
   ```

2. **Follow the testing instructions** on the page to verify all features

### Manual Testing Checklist

#### ✅ Gesture Detection
- [ ] Left swipe navigates back
- [ ] Right swipe navigates forward
- [ ] Up swipe scrolls up dynamically
- [ ] Double-click opens popup
- [ ] Alt+Arrow keys work as shortcuts

#### ✅ Voice Input
- [ ] Microphone icon activates voice recognition
- [ ] Ctrl+Shift+V keyboard shortcut works
- [ ] Voice commands are processed correctly
- [ ] TTS feedback works (if enabled)
- [ ] Wake words trigger activation

#### ✅ ISAA AI Assistant
- [ ] Chat interface responds to queries
- [ ] Page context is understood
- [ ] Commands are executed properly
- [ ] Context menu integration works

#### ✅ Live Search
- [ ] Page indexing works automatically
- [ ] Search results are highlighted
- [ ] "Scroll to" functionality works
- [ ] "Ask ISAA" integration works
- [ ] Dynamic content is indexed

#### ✅ Password Manager
- [ ] Forms are detected automatically
- [ ] Auto-fill works on login forms
- [ ] Password generation works
- [ ] 2FA codes display correctly
- [ ] Import functionality works

#### ✅ UI/UX
- [ ] Glass morphism design displays correctly
- [ ] Marine blue accents are visible
- [ ] Tab switching is smooth
- [ ] Animations work properly
- [ ] Responsive design adapts to different sizes

## 🔍 Troubleshooting

### Common Issues

#### Extension Not Loading
**Symptoms**: Extension doesn't appear in Chrome toolbar
**Solutions**:
1. Check if build completed successfully
2. Verify all files are present in build directory
3. Try reloading the extension in Chrome
4. Check Chrome console for errors

#### Connection Issues
**Symptoms**: "Disconnected" status in popup
**Solutions**:
1. Ensure ToolBox server is running on localhost:8080
2. Check firewall settings
3. Verify API endpoints are accessible
4. Check browser console for network errors

#### Gesture Detection Not Working
**Symptoms**: Swipes and gestures don't trigger actions
**Solutions**:
1. Check if gesture detection is enabled in settings
2. Verify page allows gesture events (some sites block them)
3. Try refreshing the page
4. Check content script injection

#### Voice Input Not Responding
**Symptoms**: Microphone doesn't activate or recognize speech
**Solutions**:
1. Check microphone permissions in Chrome
2. Ensure HTTPS context (required for Web Speech API)
3. Verify browser supports Web Speech API
4. Check audio input device settings

#### Password Auto-fill Failing
**Symptoms**: Forms not detected or auto-fill doesn't work
**Solutions**:
1. Check if ToolBox server is accessible
2. Verify password exists for current domain
3. Ensure form fields are properly detected
4. Check content script injection

#### Search Not Working
**Symptoms**: Live search doesn't return results
**Solutions**:
1. Click "Index Page" button first
2. Wait for page to fully load before searching
3. Check if content script is injected
4. Verify search terms are not too short

### Debug Mode

Enable debug mode for detailed logging:

1. **Open Chrome DevTools** (F12)
2. **Go to Console tab**
3. **Look for ToolBox-related messages**:
   - `🚀 ToolBox Pro initializing...`
   - `✅ ToolBox Pro initialized`
   - `🎯 Gesture detected: ...`
   - `🔐 Autofilling password...`

### Log Files

Check these locations for detailed logs:
- **Extension Console**: Chrome DevTools → Extensions → ToolBox Pro → Inspect views
- **Background Script**: Chrome DevTools → Extensions → ToolBox Pro → Background page
- **Content Script**: Page DevTools → Console (look for ToolBox messages)

## 📞 Support

### Getting Help

1. **Check the README.md** for detailed feature documentation
2. **Use the test page** to isolate issues
3. **Check browser console** for error messages
4. **Verify ToolBox server** is running and accessible

### Reporting Issues

When reporting issues, please include:
- Chrome version
- Extension version
- ToolBox server version
- Steps to reproduce
- Console error messages
- Screenshots (if applicable)

### Community Resources

- **GitHub Issues**: https://github.com/MarkinHaus/ToolBoxV2/issues
- **Documentation**: https://toolbox.app/docs
- **Discord Community**: https://discord.gg/toolbox

## 🔄 Updates

### Updating the Extension

1. **Pull latest changes**:
   ```bash
   git pull origin main
   ```

2. **Rebuild extension**:
   ```bash
   node build.js dev
   ```

3. **Reload in Chrome**:
   - Go to `chrome://extensions/`
   - Click reload button for ToolBox Pro

### Version History

- **v3.0.0**: Initial production release
  - Complete gesture detection system
  - ISAA AI integration
  - Voice input and TTS
  - Live search with page indexing
  - Password manager with 2FA
  - Glass morphism UI design

## 🎯 Next Steps

After successful installation:

1. **Explore all features** using the test page
2. **Configure settings** to your preferences
3. **Import passwords** from your browser
4. **Set up 2FA codes** for enhanced security
5. **Customize gesture sensitivity** for optimal experience

Enjoy using ToolBox Pro! 🚀
