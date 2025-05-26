## tbjs Framework: Comprehensive Guide & Documentation

**Table of Contents**

1.  **Introduction**
    *   Key Design Principles & Features
2.  **Getting Started**
    *   Prerequisites
    *   Installation
    *   HTML Setup
    *   Application Initialization (`TB.init`)
3.  **Core Modules (`TB.*`)**
    *   `TB.config`: Configuration Management
    *   `TB.logger`: Logging Utility
    *   `TB.state`: Global State Management
    *   `TB.events`: Event Bus / Pub/Sub
    *   `TB.env`: Environment Detection
    *   `TB.api`: Backend Communication
    *   `TB.router`: SPA Routing
    *   `TB.crypto`: Cryptographic Utilities
    *   `TB.user`: User Session & Authentication
    *   `TB.sse`: Server-Sent Events
    *   `TB.sw`: Service Worker Management
    *   `TB.utils`: General Utilities
    *   `TB.graphics`: 3D Graphics (THREE.js)
4.  **UI System (`TB.ui.*`)**
    *   `TB.ui.theme`: Theming (Light/Dark Mode, Backgrounds)
    *   `TB.ui.htmxIntegration`: HTMX Event Handling
    *   `TB.ui.processDynamicContent`: Handling New DOM Content
    *   **UI Components:**
        *   `TB.ui.Modal`
        *   `TB.ui.Toast`
        *   `TB.ui.Loader`
        *   `TB.ui.Button`
        *   `TB.ui.DarkModeToggle`
        *   `TB.ui.CookieBanner`
        *   `TB.ui.MarkdownRenderer`
        *   `TB.ui.NavMenu`
        *   `TB.ui.AutocompleteWidget`
5.  **Styling with Tailwind CSS**
    *   Prefixing and CSS Variables
    *   Using `tbjs` Tailwind Config in Your Project
6.  **Advanced Topics**
    *   Tauri Integration
    *   Working with 3D Graphics
7.  **Example: Login Flow Walkthrough**
8.  **Building `tbjs` (For Developers)**

---

### 1. Introduction

`tbjs` is a modular frontend framework designed for building modern web applications, with special consideration for integration with Tauri for desktop applications and tools like HTMX and Three.js. It provides a comprehensive set of tools for managing configuration, state, API communication, routing, UI components, user authentication, and more.

**Key Design Principles & Features:**

*   **Modularity:** Clear separation of concerns into `core` and `ui` modules. You can use only the parts you need.
*   **Event-Driven:** Facilitates decoupled communication between modules via an event bus.
*   **Configuration-Centric:** Application behavior is heavily influenced by a central configuration object.
*   **State Management:** Centralized application state with optional persistence.
*   **SPA Router:** Handles client-side navigation and view loading.
*   **API Abstraction:** Simplifies backend communication, supporting both HTTP and Tauri `invoke` calls.
*   **UI System:** Includes theme management (light/dark mode), dynamic backgrounds, and reusable UI components.
*   **3D Graphics Integration:** Built-in support for THREE.js for dynamic backgrounds or scenes, managed by `TB.graphics`.
*   **User Authentication:** Robust support for various authentication flows, including device key (asymmetric crypto) and WebAuthn (passkeys).
*   **HTMX Friendly:** Designed to work seamlessly alongside HTMX for enhancing HTML with dynamic behaviors.
*   **Tauri-Aware:** Core functionalities can adapt to run optimally in a Tauri environment.
*   **Modern Tooling:** Built with Webpack, Babel, PostCSS, and Tailwind CSS.

---

### 2. Getting Started

#### Prerequisites

Before using `tbjs`, ensure you have the following (or plan to include them if using related features):

1.  **HTMX** (Recommended): `tbjs` integrates well with HTMX for server-rendered partials and dynamic updates.
    ```html
    <script defer src="https://unpkg.com/htmx.org@2.0.2/dist/htmx.min.js"></script>
    ```
2.  **Three.js** (Optional, if using `TB.graphics`):
    ```html
    <script defer src="https://cdnjs.cloudflare.com/ajax/libs/three.js/0.153.0/three.min.js"></script>
    ```
3.  **Marked & Highlight.js** (Optional, if using `TB.ui.MarkdownRenderer`): For rendering Markdown to HTML with syntax highlighting.
    ```html
    <script defer src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/marked-highlight/lib/index.umd.min.js"></script>
    <script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/default.min.css">
    ```
    *Ensure `window.marked`, `window.markedHighlight`, and `window.hljs` are available before `TB.ui.MarkdownRenderer` is used.*

4.  **Material Symbols** (Optional, used by some default UI components):
    ```html
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
    ```

#### Installation

1.  **Add `tbjs` to your project:**
    If `tbjs` were published to npm:
    ```bash
    npm install tbjs
    # or
    yarn add tbjs
    ```
    Since it's often used locally or as part of a larger monorepo, you'd typically:
    *   Build `tbjs` from source (see [Building `tbjs`](#8-building-tbjs-for-developers)) to get `dist/tbjs.js` and `dist/tbjs.css`.
    *   Or, if integrating into a build system, import directly from its source path (e.g., `import TB from 'path/to/tbjs/src/index.js';`).

2.  **Include files in your HTML (if using pre-built dist files):**
    ```html
    <link rel="stylesheet" href="path/to/your/tbjs/dist/tbjs.css">
    <!-- Load tbjs.js as a module or a global script depending on its build -->
    <script defer type="module" src="path/to/your/tbjs/dist/tbjs.js"></script> <!-- If ES Module build -->
    <!-- <script defer src="path/to/your/tbjs/dist/tbjs.js"></script> --> <!-- If UMD build -->
    ```

3.  **Peer Dependencies (Reminder):**
    Ensure you have `htmx.org` and `three` installed/included if you plan to use features that depend on them.
    ```bash
    npm install htmx.org three # Or yarn add
    ```

---

### 3. Core Modules (`TB.*`)

#### `TB.config`: Configuration Management

Manages application-wide settings. It's initialized by `TB.init()` with default values merged with your provided configuration.

*   **Initialization:** `TB.config.init(userAppConfig)` is called by `TB.init`.
    *   `userAppConfig` options:
        *   `appRootId` (string): ID of the main DOM element for router views. Default: `app-root`.
        *   `baseApiUrl` (string): Base URL for API calls. Default: `/api`. Normalized to be absolute (e.g., `/api` becomes `window.location.origin/api`).
        *   `baseFileUrl` (string): Base URL for fetching static HTML files for routing. Default: `window.location.origin`. Normalized to end with `/` if it has a path, and ensures it doesn't include file names.
        *   `initialState` (object): Initial state for `TB.state`.
        *   `themeSettings` (object): See `TB.ui.theme` section.
        *   `routes` (array): Predefined routes for `TB.router` (currently for reference/future use).
        *   `logLevel` (string): `debug`, `info`, `warn`, `error`, `none`. Default: `info`.
        *   `isProduction` (boolean): Inferred based on hostname (`localhost`, `127.0.0.1`) if not explicitly set.
        *   `serviceWorker` (object): `{ enabled: boolean, url: string, scope: string }`.
*   **Getting Configuration:**
    ```javascript
    const apiUrl = TB.config.get('baseApiUrl');
    const logLevel = TB.config.get('logLevel');
    const themePref = TB.config.get('themeSettings.defaultPreference'); // Dot notation for nested
    const allConfig = TB.config.getAll(); // Returns a copy of the entire config
    ```
*   **Setting Configuration (dynamically, use with caution after init):**
    ```javascript
    TB.config.set('myCustomSetting', 'someValue');
    TB.config.set('featureFlags.newFeature', true);
    ```

#### `TB.logger`: Logging Utility

Provides leveled, prefixed, and timestamped logging to the console.

*   **Initialization:** `TB.logger.init({ logLevel: '...' })` is called by `TB.init` based on `TB.config`.
*   **Setting Log Level:**
    ```javascript
    TB.logger.setLevel('debug'); // 'debug', 'info', 'warn', 'error', 'none'
    ```
*   **Logging Messages:**
    ```javascript
    TB.logger.debug('Detailed debug message:', { data: 123 });
    TB.logger.info('Informational message.'); // Alias: TB.logger.log()
    TB.logger.warn('Potential issue warning.');
    TB.logger.error('An error occurred:', new Error('Something went wrong'));
    ```
    Output includes a timestamp, `[tbjs]`, and the log level (e.g., `[DEBUG]`).

#### `TB.state`: Global State Management

A simple key-value store for global application state with optional persistence to `localStorage`.

*   **Initialization:** `TB.state.init(initialState)` is called by `TB.init` with `TB.config.get('initialState')`. Loads any persisted state.
*   **Getting State:**
    ```javascript
    const username = TB.state.get('user.username'); // Dot notation for nested
    const allState = TB.state.get(); // Returns a copy of the entire state
    ```
*   **Setting State:**
    ```javascript
    // Set a simple value
    TB.state.set('ui.darkMode', true);

    // Set a nested value, creating intermediate objects if they don't exist
    TB.state.set('user.profile.avatarUrl', '/path/to/avatar.png');

    // Persist the top-level key 'user' to localStorage
    TB.state.set('user.isLoggedIn', true, { persist: true });
    // Any change under 'user' (e.g., 'user.settings.notifications') will now persist 'user'.
    ```
    *   Emits `state:changed` event with `{ key, value, fullState }`.
    *   Emits specific event like `state:changed:user:profile:avatarUrl` with the `value`.
*   **Deleting State:**
    ```javascript
    TB.state.delete('user.profile.temporaryToken');
    TB.state.delete('featureFlags.oldFlag', { persist: true }); // Will update persisted 'featureFlags'
    ```
*   **Legacy "Var" Methods (for simple key-value persistence, prefer structured state with `persist` option):**
    *   `TB.state.initVar('myVar', 'defaultValue')`: Sets if not already defined, persists.
    *   `TB.state.delVar('myVar')`: Deletes and updates persisted state.
    *   `TB.state.getVar('myVar')`: Gets value.
    *   `TB.state.setVar('myVar', 'newValue')`: Sets value, persists.

#### `TB.events`: Event Bus / Pub/Sub

Allows modules to communicate without direct dependencies.

*   **Subscribing to Events:**
    ```javascript
    function handleThemeChange(eventData) {
        console.log('Theme changed to:', eventData.mode);
    }
    TB.events.on('theme:changed', handleThemeChange);

    // Subscribe only once
    TB.events.once('app:firstLogin', (userData) => { /* ... */ });
    ```
*   **Unsubscribing from Events:**
    ```javascript
    TB.events.off('theme:changed', handleThemeChange);
    ```
*   **Emitting Events:**
    ```javascript
    TB.events.emit('user:loggedIn', { userId: 123, username: 'testuser' });
    ```
    If a listener throws an error, `TB.logger.error` is called, and other listeners still execute.
*   **Common Framework Events:** `tbjs:initialized`, `state:changed`, `router:navigationSuccess`, `theme:changed`, `api:networkError`, `graphics:initialized`, `user:loggedOut`, etc.

#### `TB.env`: Environment Detection

Provides information about the runtime environment.

*   **Initialization:** `TB.env.detect()` is called by `TB.init`.
*   **Checking Environment:**
    ```javascript
    if (TB.env.isTauri()) {
        console.log('Running in Tauri environment.');
    } else if (TB.env.isWeb()) {
        console.log('Running in a web browser.');
    }
    if (TB.env.isMobile()) { // Currently implies Tauri mobile if detected
        console.log('Running on a mobile platform (Tauri).');
    }
    ```

#### `TB.api`: Backend Communication

Handles all HTTP and Tauri `invoke` calls, standardizing responses.

*   **Core `Result` Object:** All `TB.api` methods (and Tauri invokes) aim to return or be wrapped into a `Result` object:
    ```javascript
    // Structure of a Result object (simplified)
    // const result = {
    //   origin: Array<string>,    // e.g., ['http'], ['tauri']
    //   error: string,           // From TB.ToolBoxError (e.g., 'none', 'InternalError')
    //   result: {                // Instance of ToolBoxResult
    //     data_to: string,       // From TB.ToolBoxInterfaces (e.g., 'API', 'NATIVE')
    //     data_info: string|null,// Additional info
    //     data: any              // The actual payload
    //   },
    //   info: {                  // Instance of ToolBoxInfo
    //     exec_code: number,     // HTTP status or custom code (0 for success)
    //     help_text: string      // Descriptive message
    //   },
    //   get: function() { return this.result.data; }, // Helper to get payload
    //   log: function() { /* console logs details */ },
    //   html: function() { /* returns an HTML representation for debugging */ }
    // };
    ```
*   **`TB.api.request(moduleName, functionName, payload, method, useTauri, isSpecialAuthRoute)`:**
    The primary method for making backend requests.
    *   `moduleName` (string): Backend module/class OR full path (e.g., `/validateSession`).
    *   `functionName` (string|object): Backend function/method OR query params object if `moduleName` is a full path (for GET/DELETE).
    *   `payload` (object|string|null): Data to send. Object for JSON POST/PUT; string can be form-urlencoded or query params.
    *   `method` (string): HTTP method (`GET`, `POST`, etc.). Default: `POST`.
    *   `useTauri` (string): `auto` (default), `force` (Tauri only), `never` (HTTP only).
    *   `isSpecialAuthRoute` (boolean): If `true`, influences token handling (rarely needed).

    **URL Construction (HTTP):**
    *   Standard: `baseApiUrl/moduleName/functionName`
    *   Full path: `baseApiUrl` + `moduleName` (where `moduleName` starts with `/`, e.g., `/custom/endpoint`)

    ```javascript
    // POST request (HTTP or Tauri if available and 'auto')
    const userData = { name: 'John Doe', email: 'john@example.com' };
    let response = await TB.api.request('UserModule', 'createUser', userData); // Defaults to POST

    if (response.error === TB.ToolBoxError.none) {
        console.log('User created:', response.get());
    } else {
        TB.logger.error('Failed to create user:', response.info.help_text);
    }

    // GET request with query parameters from an object
    response = await TB.api.request('ProductModule', 'getProduct', { id: 123 }, 'GET');
    // URL: /api/ProductModule/getProduct?id=123

    // Full path GET (functionName is query params object)
    response = await TB.api.request('/custom/data', { type: 'summary' }, null, 'GET');
    // URL: /api/custom/data?type=summary (if baseApiUrl is /api)
    ```
*   **`TB.api.fetchHtml(path)`:** Fetches HTML content, typically for router views. Path is relative to `baseFileUrl`.
    ```javascript
    const htmlResult = await TB.api.fetchHtml('/about.html'); // Fetches /web/pages/about.html
    if (!htmlResult.startsWith('HTTP error!')) { /* ... */ }
    ```
*   **`TB.api.AuthHttpPostData(username)`:** Specific method for validating a session. Calls `/validateSession`.
*   **`TB.api.logoutServer()`:** Notifies the backend to invalidate the current user's session token (calls `/web/logoutS`).
*   **Events:**
    *   `api:networkError`: Emitted on fetch network failures. Payload: `{ url, error }`.

#### `TB.router`: SPA Routing

Manages client-side navigation and view rendering.

*   **Initialization:** `TB.router.init(rootElement, predefinedRoutes)` called by `TB.init`.
    *   `rootElement`: The DOM element where views will be rendered (from `TB.config.appRootId`).
    *   Automatically navigates to the initial URL (or `/index.html`).
*   **Navigating:**
    ```javascript
    // Navigate to a new path, updating browser history
    TB.router.navigateTo('/products/123'); // Fetches baseFileUrl + /products/123.html

    // Navigate and replace current history entry
    TB.router.navigateTo('/profile/settings', true);
    ```
    *   Fetches HTML from `TB.config.get('baseFileUrl') + path + '.html'` (by default, unless path includes an extension).
    *   Handles script loading within new views (external once, inline executed, `unsave` attribute for fresh execution, `global="true"` for potential preservation).
    *   Updates `appRootElement.innerHTML` with fetched content.
    *   Calls `TB.ui.processDynamicContent()` on the new content.
    *   Handles 404 errors by trying to navigate to `/web/assets/404.html`.
    *   Handles 401 errors by trying to navigate to `/web/assets/401.html`.
*   **Getting Current Path:** `TB.router.getCurrentPath()`
*   **Cache Management:**
    *   `TB.router.clearCache(path)`: Clears HTML cache for a specific path or all if `path` is omitted (uses `sessionStorage` if `USE_SESSION_CACHE` is true in router.js).
    *   `scriptCache` (Set of script `src` URLs) prevents re-fetching external scripts.
*   **Events:**
    *   `router:beforeNavigation`: `{ from, to }`
    *   `router:navigationSuccess`: `{ path, contentSource }` ('cache' or 'fetched')
    *   `router:navigationError`: `{ path, error }`
    *   `router:contentProcessed`: `{ path, element }`

#### `TB.crypto`: Cryptographic Utilities

Provides functions for various cryptographic operations, including WebAuthn. Relies on browser's Web Crypto API.

*   **Key Management & Signing:**
    *   `TB.crypto.generateAsymmetricKeys()`: Generates RSA-OAEP key pair (PEM & Base64).
    *   `TB.crypto.decryptAsymmetric(encryptedBase64Data, privateKeyBase64, convertHex = false)`: Decrypts RSA-OAEP encrypted data.
    *   `TB.crypto.signMessage(privateKeyBase64, message)`: Signs a message using RSA-PSS.
    *   `TB.crypto.storePrivateKey(privateKeyBase64, username)`: Stores private key in `localStorage`.
    *   `TB.crypto.retrievePrivateKey(username)`: Retrieves private key.
*   **Symmetric Encryption/Decryption:**
    *   `TB.crypto.generateSymmetricKey()`: Generates an AES-GCM key (Base64 of raw key).
    *   `TB.crypto.decryptSymmetric(encryptedDataB64, password)`: Decrypts AES-GCM data (assumes IV is prepended to ciphertext, password used for key derivation via PBKDF2).
*   **WebAuthn (Passkeys):**
    *   The Relying Party ID (`rpId`) is determined from `window.location.hostname` (or "localhost").
    *   `TB.crypto.registerWebAuthnCredential(registrationData, singData)`:
        *   `registrationData`: `{ challenge, userId, username }` from server.
        *   `singData`: Additional data (e.g., session token) to associate.
        *   Calls `navigator.credentials.create()`. Returns payload for server verification.
    *   `TB.crypto.authorizeWebAuthnCredential(rawIdAsBase64, challenge, username)`:
        *   `rawIdAsBase64`, `challenge`, `username` from server.
        *   Calls `navigator.credentials.get()`. Returns assertion payload for server verification.
*   **Data Conversions:** `arrayBufferToBase64`, `base64ToArrayBuffer`, `strToBase64`, etc.

#### `TB.user`: User Session & Authentication

Manages user state, authentication flows, and user-specific data. User state is stored under `TB.state.get('user')`.

*   **Initialization:** `TB.user.init(forceServerFetch = false)`:
    *   Called by `TB.init`. Loads session, validates with backend, synchronizes user data.
*   **Authentication State:** `TB.user.isAuthenticated()`, `TB.user.getUsername()`, `TB.user.getToken()`, etc.
*   **Login Methods:**
    *   `async TB.user.signup(username, email, initiationKey, registerAsPersona = false)`: Initiates user creation.
    *   `async TB.user.loginWithDeviceKey(username)`: Login using locally stored asymmetric key.
    *   `async TB.user.loginWithWebAuthn(username)`: Login using WebAuthn (passkey).
    *   `async TB.user.requestMagicLink(username)`: Requests a magic link email.
    *   `async TB.user.registerDeviceWithInvitation(username, invitationKey)`: Registers a new device.
    *   `async TB.user.registerWebAuthnForCurrentUser(username)`: Adds a WebAuthn credential for an authenticated user.
*   **Session Management:**
    *   `async TB.user.checkSessionValidity()`: Validates current token with server.
    *   `async TB.user.logout(notifyServer = true)`: Clears local session, notifies server.
*   **User-Specific Data:** `TB.user.getUserData(key)`, `TB.user.setUserData(keyOrObject, value, syncToServer = false)`, `async TB.user.syncUserData()`, `async TB.user.fetchUserData()`.
*   **Events:** `user:stateChanged`, `user:loggedOut`.

#### `TB.sse`: Server-Sent Events

Manages connections to Server-Sent Event streams.

*   **Connecting:** `TB.sse.connect(url, options = {})`
    *   `options`: `{ onOpen, onError, onMessage, listeners: { eventName: handler }, eventSourceOptions }`.
    ```javascript
    TB.sse.connect('/api/sse/updates', {
        listeners: {
            'user-update': (data) => TB.state.set('user.profile', data),
            'new-notification': (data) => TB.ui.Toast.showInfo(data.message)
        }
    });
    ```
*   **Disconnecting:** `TB.sse.disconnect(url)`, `TB.sse.disconnectAll()`
*   **Getting Connection:** `TB.sse.getConnection(url)`
*   **Events Emitted:** `sse:open:<url>`, `sse:error:<url>`, `sse:event:<url>:<eventName>`, etc.

#### `TB.sw`: Service Worker Management

Handles registration and communication with your application's service worker.

*   **Configuration (`TB.config.get('serviceWorker')`):** `enabled`, `url`, `scope`.
*   **Registration:** Called automatically by `TB.init` if enabled. Manual: `await TB.sw.register()`.
*   **Unregistration:** `await TB.sw.unregister()`
*   **Sending Messages:** `await TB.sw.sendMessage({ type: 'GET_VERSION' })`
*   **Events Emitted:** `sw:updateAvailable`, `sw:contentCached`.
    ```javascript
    TB.events.on('sw:updateAvailable', ({ registration }) => {
        if (confirm('New version available. Reload?')) {
            registration.waiting.postMessage({ type: 'SKIP_WAITING' });
            // Listen for controllerchange to reload
            navigator.serviceWorker.addEventListener('controllerchange', () => window.location.reload());
        }
    });
    ```

#### `TB.utils`: General Utilities

A collection of helper functions.

*   `TB.utils.autocomplete(inputElement, arrayOrFunctionSource)`: Basic autocomplete (prefer `TB.ui.AutocompleteWidget`).
*   `TB.utils.debounce(func, delay)`
*   `TB.utils.throttle(func, limit)`
*   `TB.utils.uniqueId(prefix = 'id-')`
*   `TB.utils.deepClone(obj)`
*   `TB.utils.cleanUrl(url)`: Basic URL cleaning.

#### `TB.graphics`: 3D Graphics (THREE.js)

Manages a THREE.js scene, typically for background effects.

*   **Initialization:** `TB.graphics.init(canvasContainerSelector, options = {})`
    *   `canvasContainerSelector`: CSS selector for the DOM element (e.g., `'#threeDScene'`).
    *   `options`: `{ cameraY, cameraZ, sierpinskiDepth, loaderHideDelay }`.
    *   Typically called if `themeSettings.background.type` is `'3d'`.
*   **Control Methods:**
    *   `TB.graphics.dispose()`, `TB.graphics.pause()`, `TB.graphics.resume()`.
    *   `TB.graphics.updateTheme(themeMode)`: Called by `TB.ui.theme`.
    *   `TB.graphics.setSierpinskiDepth(newDepth)`.
    *   `TB.graphics.setAnimationSpeed(x, y, z, factor)`.
    *   `TB.graphics.adjustCameraZoom(delta)`, `TB.graphics.setCameraZoom(absoluteZoomValue)`.
*   **Programmed Animation Sequences:**
    *   `TB.graphics.playAnimationSequence(sequenceString, onCompleteCallback, baseSpeedOverride, speedFactorOverride)`
        *   `sequenceString`: e.g., `"R1+32:P2-14"` (Type, Repeat, Direction, Speed, Complexity).
    *   `TB.graphics.stopAnimationSequence()`.
*   **Events:** `graphics:initialized`, `graphics:disposed`.

---

### 4. UI System (`TB.ui.*`)

#### `TB.ui.theme`: Theming

Manages light/dark mode and application background.

*   **Initialization:** `TB.ui.theme.init(themeSettings)` called by `TB.init`.
    *   `themeSettings`: `{ defaultPreference ('light'|'dark'|'system'), background: { type, light, dark, placeholder } }`.
    *   `background.type`: `'3d'`, `'image'`, `'color'`, `'none'`.
    *   `background.light/dark`: `{ color: string, image: string|null }`.
    *   `background.placeholder`: `{ image_light, image_dark, displayUntil3DReady }`.
*   **Interacting with Theme:**
    *   `TB.ui.theme.setPreference('dark')`, `TB.ui.theme.togglePreference()`.
    *   `TB.ui.theme.getCurrentMode()` ('light' or 'dark').
    *   `TB.ui.theme.getPreference()` ('light', 'dark', or 'system').
*   **Background Management:**
    *   Uses `#appBackgroundContainer` for image/color and `#threeDScene` for 3D.
*   **Events:** `theme:changed` (payload: `{ mode: 'light' | 'dark' }`).

#### `TB.ui.htmxIntegration`: HTMX Event Handling

Listens to HTMX events to integrate `tbjs` functionalities.

*   **Initialization:** `TB.ui.htmxIntegration.init()` is called by `TB.init`.
*   **`htmx:afterSwap`:** Calls `TB.ui.processDynamicContent` on the new HTMX target element.
*   **`htmx:afterRequest`:**
    *   Inspects XHR response. If JSON, wraps in `TB.api.Result`, shows toasts for errors.
    *   Handles `REMOTE` render commands.
    *   Emits `htmx:jsonResponse`.

#### `TB.ui.processDynamicContent(parentElement, options = {})`

Initializes `tbjs` features/components within newly added DOM content.

*   `parentElement`: The container of the new content.
*   `options`: `{ addScripts (default true), scriptCache }`.
*   **Actions:** Calls `window.htmx.process()`, handles scripts, calls `TB.ui.MarkdownRenderer.renderAllIn()`, initializes data-attribute driven components like `AutocompleteWidget`.

#### UI Components

##### `TB.ui.Modal`

Displays modal dialogs.

*   **Static Usage:** `TB.ui.Modal.show({ title, content, maxWidth, buttons, onOpen, onClose, ... })`
    *   `buttons`: `[{ text, action: (modalInstance) => {}, variant, className }]`
*   **Styling:** Uses Tailwind CSS, "milk glass" effect.
*   **Events:** `modal:shown`, `modal:closed`.

##### `TB.ui.Toast`

Displays "speech balloon" style toast notifications.

*   **Static Usage:**
    *   `TB.ui.Toast.showInfo(message, options)`
    *   `TB.ui.Toast.showSuccess(message, options)`
    *   `TB.ui.Toast.showWarning(message, options)`
    *   `TB.ui.Toast.showError(message, options)`
*   **Options:** `{ title, duration, position, actions, icon, closable, showDotOnHide, dotDuration }`.
*   **Events:** `toast:shown`, `toast:hidden`.

##### `TB.ui.Loader`

Displays a loading indicator.

*   **Static Usage (Global Page Loader):**
    *   `const loaderElement = TB.ui.Loader.show('Processing...');`
    *   `TB.ui.Loader.hide(loaderElement);` (or `TB.ui.Loader.hide()` for default).
*   **Options:** `{ text, fullscreen, customSpinnerHtml }`.

##### `TB.ui.Button`

Creates styled button elements programmatically.

*   **Static Usage:** `const myButtonElement = TB.ui.Button.create(text, onClickCallback, options)`
*   **Options:** `{ variant, size, iconLeft, iconRight, type, disabled, isLoading, ... }`.
*   **Instance Methods:** `setLoading(true)`, `setDisabled(true)`.

##### `TB.ui.DarkModeToggle`

UI component for switching themes, syncing with `TB.ui.theme`.

*   **HTML (Example):**
    ```html
    <div id="darkModeToggleContainer">
        <label for="darkModeSwitch"><span class="tb-toggle-icon material-symbols-outlined">light_mode</span></label>
        <input type="checkbox" id="darkModeSwitch" class="tb-sr-only">
    </div>
    ```
*   **Initialization:** `TB.ui.DarkModeToggle.init({ containerSelector, iconSelector, checkboxSelector, ... })`. Default init uses common selectors.
*   Updates icon and checkbox based on `theme:changed` event.

##### `TB.ui.CookieBanner`

Displays a cookie consent banner and settings modal.

*   **Static Usage:** `TB.ui.CookieBanner.show({ title, message, termsLink, onConsent, ... })`
*   `onConsent` callback receives `{ essential, preferences, analytics, source }`.
*   **Methods:** `CookieBanner.getConsent()`, `CookieBanner.clearConsent()`.
*   **Events:** `cookieConsent:updated`, `cookieBanner:shown`/`hidden`.

##### `TB.ui.MarkdownRenderer`

Renders Markdown to HTML, with optional `highlight.js` syntax highlighting.

*   **Dependencies:** `marked`, `highlight.js`, `marked-highlight` (global or loaded).
*   **Methods:**
    *   `TB.ui.MarkdownRenderer.render(markdownString)`
    *   `TB.ui.MarkdownRenderer.renderAllIn(parentElement)` (for elements with `.markdown` class)
    *   `TB.ui.MarkdownRenderer.renderElement(element)`
*   Adds Tailwind Prose classes (`prose dark:prose-invert`) for styling.

##### `TB.ui.NavMenu`

A slide-in (or modal-style) navigation menu.

*   **HTML Trigger (Example):**
    ```html
    <div id="Nav-Main"> <!-- Menu is appended here -->
        <div id="links"><span class="material-symbols-outlined">menu</span></div>
    </div>
    ```
*   **Initialization:** `TB.ui.NavMenu.init({ triggerSelector, menuContentHtml, ... })`.
*   **Events:** `navMenu:opened`, `navMenu:closed`.

##### `TB.ui.AutocompleteWidget`

Provides autocomplete suggestions for input fields.

*   **HTML (Declarative):**
    ```html
    <input type="text" data-tb-autocomplete data-tb-autocomplete-source='["Apple", "Banana"]'>
    <!-- Or data-tb-autocomplete-source="myGlobalFunctionName" -->
    ```
*   **Initialization:**
    *   Automatic: `TB.ui.AutocompleteWidget.initAll()` (called by `processDynamicContent`).
    *   Manual: `new TB.ui.AutocompleteWidget(inputEl, { source, minLength, onSelect, ... })`.
*   **Features:** Keyboard navigation, ARIA attributes.

---

### 5. Styling with Tailwind CSS

`tbjs` components are primarily styled using Tailwind CSS utility classes.

#### Prefixing and CSS Variables

*   **Prefix:** `tbjs`'s internal Tailwind configuration uses a `tb-` prefix (e.g., `tb-bg-primary-500`, `tb-text-lg`). This is crucial to avoid conflicts if your main application also uses Tailwind without a prefix or with a different one.
*   **Main CSS (`tbjs.css` or `tbjs-main.css`):**
    *   Imports Tailwind utilities generated with the `tb-` prefix.
    *   Defines CSS custom properties (variables) for theming (e.g., `--tb-color-primary-500`, `--theme-bg`, `--glass-bg`). These are used by the prefixed Tailwind classes.
    *   Includes light and dark theme definitions typically applied to `body[data-theme="dark"]` or `body.dark-mode`.
    *   Provides base styles and some component-specific styles hard to achieve with utilities alone (e.g., toast speech balloon tail).
*   **Customization:**
    *   Applications can override the CSS variables defined in `tbjs.css` in their own stylesheets to customize the look and feel.
    *   For deeper Tailwind customization (new colors, variants specific to `tbjs`), you would edit `tbjs/tailwind.config.js` and rebuild `tbjs`.

#### Using `tbjs` Tailwind Config in Your Project

If your project also uses Tailwind CSS, you have a few options:

1.  **Separate Builds (Recommended for Isolation):**
    *   Build `tbjs.css` using its own Tailwind configuration (with the `tb-` prefix).
    *   Build your application's CSS using its Tailwind configuration.
    *   Include both CSS files in your HTML. The `tb-` prefix prevents most conflicts.

2.  **Merging Configurations (Advanced):**
    If you want a single Tailwind build process, you might try to merge configurations. This can be complex due to prefixing and potential conflicts.
    *   You would need to ensure your main `tailwind.config.js` includes the `content` paths for `tbjs` source files.
    *   You'd also need to decide how to handle the `tb-` prefix. If your app doesn't use a prefix, `tbjs` components might not be styled correctly unless you manually adapt their classes or adjust the `tbjs` source.
    *   A simpler merge might involve including `tbjs`'s Tailwind plugin or preset if it were structured that way, but this is not the default.

    Example (Conceptual - requires careful setup):
    ```javascript
    // your-app/tailwind.config.js
    // const tbjsTailwindConfig = require('path/to/tbjs/tailwind.config.js'); // If CJS

    export default {
      content: [
        './src/**/*.{html,js,svelte,vue,jsx,tsx}', // Your app's content
        './node_modules/tbjs/src/**/*.{html,js}', // Or path to tbjs source
      ],
      // If your app uses a prefix, it might conflict or work alongside tb-
      // prefix: 'app-',
      theme: {
        extend: {
          // You might try to extend with tbjs colors if they are defined without prefix in its config
          // This part is tricky due to the 'tb-' prefix baked into tbjs's own build
        },
      },
      plugins: [],
    };
    ```
    *Generally, keeping `tbjs.css` separate with its `tb-` prefix is the most straightforward way to avoid styling conflicts.*

---

### 6. Advanced Topics

#### Tauri Integration

*   **Environment Check:** Use `TB.env.isTauri()` to execute Tauri-specific code.
*   **API Calls:** `TB.api.request()` automatically uses `window.__TAURI__.invoke` if `useTauri` is `'auto'` (default) or `'force'` and the environment is Tauri.
    *   The Tauri command invoked is typically `moduleName.functionName` (e.g., `MyRustModule.my_function`).
    ```javascript
    if (TB.env.isTauri()) {
        const result = await TB.api.request('my_rust_command', 'sub_command_or_payload_key', { data: 'payload' });
        // Effective Tauri invoke: window.__TAURI__.invoke('my_rust_command.sub_command_or_payload_key', { data: 'payload' });
    }
    ```
*   **Platform-Specific Features:** The `initializeApp` function shows a pattern for loading Tauri-specific listeners or UI adjustments.

#### Working with 3D Graphics

*   The `TB.graphics` module manages a THREE.js scene, typically for background effects.
*   **Integration with Theme:** If `themeSettings.background.type` is `'3d'`, `TB.ui.theme` will initialize `TB.graphics` (targeting `#threeDScene`) and call `TB.graphics.updateTheme()` on light/dark mode changes.
*   **Manual Control:**
    ```javascript
    TB.graphics.setSierpinskiDepth(3);
    TB.graphics.playAnimationSequence("R2+52:P1-31", () => console.log("3D Animation done!"));
    // Mouse/touch drag for interaction is usually enabled by default.
    ```

---

### 7. Example: Login Flow Walkthrough

This conceptual example (based on a typical `login.js` implementation with `tbjs`) demonstrates how various modules work together:

1.  **Initialization (e.g., in a `setupLogin` function called when the login page loads):**
    *   Wait for `tbjs:initialized` or check `TB.isInitialized`.
    *   Optionally, play an initial graphics animation: `TB.graphics.playAnimationSequence("Z0+12")`.
    *   Check session validity: `TB.user.checkSessionValidity()`. If valid, show a toast and offer navigation to a dashboard.

2.  **Form Submission (e.g., on login button click):**
    *   Prevent default form submission.
    *   Get username from input. Validate locally (show info/toast on error).
    *   Play a "login attempt" graphics animation: `TB.graphics.playAnimationSequence("R1+11:P1-11")`.
    *   Show a global loader: `TB.ui.Loader.show('Attempting login...')`.
    *   **Authentication Logic (Conditional):**
        *   If user opts for WebAuthn/Passkey: `await TB.user.loginWithWebAuthn(username)`.
        *   Else (e.g., device key): `await TB.user.loginWithDeviceKey(username)`.
            *   If `loginWithDeviceKey` fails due to no key: Show a sticky error toast with actions:
                *   "Try Passkey/WebAuthn": Calls `TB.user.loginWithWebAuthn()`.
                *   "Register with Invitation": Prompts for key, calls `TB.user.registerDeviceWithInvitation()`.
                *   "Send Magic Link": Calls `TB.user.requestMagicLink()`.
                *   Each action would have its own loader management and graphics animations.

3.  **Result Handling:**
    *   Based on `result.success` from `TB.user` login methods:
        *   **Success:**
            *   Show success toast: `TB.ui.Toast.showSuccess('Login successful!')`.
            *   Play success animation: `TB.graphics.playAnimationSequence("Z1+32:R0+50")`.
            *   Navigate: `TB.router.navigateTo('/dashboard')`.
        *   **Failure:**
            *   Show error toast: `TB.ui.Toast.showError(result.message)`.
            *   Play failure animation: `TB.graphics.playAnimationSequence("P2-42")`.
    *   Use `TB.logger` for detailed console logging throughout the process.
    *   Hide loader (`TB.ui.Loader.hide()`) and stop animations (`TB.graphics.stopAnimationSequence()`) in a `finally` block or after completion.

This flow showcases:
*   **Event-driven UI:** Graphics and toasts respond to login states.
*   **Module Orchestration:** `TB.user`, `TB.graphics`, `TB.ui.Toast`, `TB.ui.Loader`, `TB.router`, `TB.logger` working in concert.
*   **User Feedback:** Clear messages and visual cues for different scenarios.

---

### 8. Building `tbjs` (For Developers)

If you are modifying the `tbjs` framework itself or need to build it from source:

1.  **Prerequisites:**
    *   Node.js and npm (or yarn) installed.
2.  **Install Dependencies:**
    Navigate to the `tbjs` root directory in your terminal and run:
    ```bash
    npm install
    # or
    # yarn install
    ```
3.  **Build Scripts (examples from a typical `package.json`):**
    *   **Production Build:**
        ```bash
        npm run build
        ```
        This usually creates optimized, minified files in a `dist/` directory (e.g., `dist/tbjs.js` and `dist/tbjs.css`). The build process uses Webpack, configured in `webpack.config.js`.
    *   **Development Watch Mode:**
        ```bash
        npm run watch
        # or npm run dev
        ```
        This watches source files for changes and automatically rebuilds, often in a non-minified format for easier debugging.
    *   **Linting:**
        ```bash
        npm run lint
        ```
        Checks the JavaScript code for style consistency and potential errors using a linter like ESLint.

