# tbjs Mini Framework Documentation

**Version:** 0.1.0-alpha

`tbjs` is a lightweight, modular JavaScript framework designed for building modern web applications, with a focus on integrating with tools like HTMX and Three.js. It provides core functionalities such as routing, state management, API communication, and a UI component system.

## Table of Contents
1.  [Introduction](#introduction)
2.  [Getting Started](#getting-started)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
    *   [HTML Setup](#html-setup)
    *   [Initialization](#initialization)
3.  [Core Modules](#core-modules)
    *   [Configuration (`TB.config`)](#configuration-tbconfig)
    *   [State Management (`TB.state`)](#state-management-tbstate)
    *   [Routing (`TB.router`)](#routing-tbrouter)
    *   [API Communication (`TB.api`)](#api-communication-tbapi)
    *   [Event System (`TB.events`)](#event-system-tbevents)
    *   [Logging (`TB.logger`)](#logging-tblogger)
    *   [Environment Detection (`TB.env`)](#environment-detection-tbenv)
    *   [Cryptography (`TB.crypto`)](#cryptography-tbcrypto)
    *   [User Management (`TB.user`)](#user-management-tbuser)
    *   [Server-Sent Events (`TB.sse`)](#server-sent-events-tbsse)
    *   [Service Worker (`TB.sw`)](#service-worker-tbsw)
    *   [Utilities (`TB.utils`)](#utilities-tbutils)
4.  [UI System (`TB.ui`)](#ui-system-tbui)
    *   [Theme Management (`TB.ui.theme`)](#theme-management-tbuitheme)
    *   [Graphics (`TB.graphics`)](#graphics-tbgraphics)
    *   [HTMX Integration (`TB.ui.htmxIntegration`)](#htmx-integration-tbuihtmxintegration)
    *   [Dynamic Content Processing](#dynamic-content-processing)
    *   [Components](#components)
        *   [Modal (`TB.ui.Modal`)](#modal-tbuimodal)
        *   [Toast (`TB.ui.Toast`)](#toast-tbuitoast)
        *   [Loader (`TB.ui.Loader`)](#loader-tbuiloader)
        *   [Button (`TB.ui.Button`)](#button-tbuibutton)
        *   [DarkModeToggle (`TB.ui.DarkModeToggle`)](#darkmodetoggle-tbuidarkmodetoggle)
        *   [CookieBanner (`TB.ui.CookieBanner`)](#cookiebanner-tbuicookiebanner)
        *   [MarkdownRenderer (`TB.ui.MarkdownRenderer`)](#markdownrenderer-tbuimarkdownrenderer)
        *   [AutocompleteWidget (`TB.ui.AutocompleteWidget`)](#autocompletewidget-tbuiautocompletewidget)
        *   [NavMenu (`TB.ui.NavMenu`)](#navmenu-tbuinavmenu)
5.  [Usage Examples](#usage-examples)
    *   [Basic Application Setup](#basic-application-setup)
    *   [Fetching Data and Updating State](#fetching-data-and-updating-state)
    *   [Client-Side Routing](#client-side-routing)
    *   [Displaying a Modal](#displaying-a-modal)
    *   [User Authentication Flow](#user-authentication-flow)
6.  [Styling with Tailwind CSS](#styling-with-tailwind-css)
7.  [Building tbjs (For Developers)](#building-tbjs-for-developers)

---

## 1. Introduction
`tbjs` aims to provide a solid foundation for single-page applications (SPAs) and dynamic web experiences. It's built with modularity in mind, allowing you to use only the parts you need. Key features include:

*   **Configuration-driven**: Easily customize framework behavior.
*   **State Management**: Centralized application state.
*   **SPA Router**: Handles client-side navigation and view loading.
*   **API Abstraction**: Simplifies backend communication (HTTP & Tauri).
*   **Event Bus**: Decoupled communication between modules.
*   **UI System**: Includes theme management and reusable components.
*   **3D Graphics**: Integration with Three.js for dynamic backgrounds or scenes.
*   **User Authentication**: Built-in support for various authentication flows, including WebAuthn.
*   **HTMX Friendly**: Designed to work alongside HTMX for enhancing HTML.

---

## 2. Getting Started

### Prerequisites
Before using `tbjs`, ensure you have the following:

1.  **HTMX**: `tbjs` is designed to work well with HTMX. Include HTMX in your project.
    ```html
    <script src="https://unpkg.com/htmx.org@2.0.0/dist/htmx.min.js"></script>
    ```
2.  **Three.js** (Optional, if using `TB.graphics`):
    ```html
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/0.153.0/three.min.js"></script>
    ```
3.  **Marked & Highlight.js** (Optional, if using `TB.ui.MarkdownRenderer`): These should be available globally.
    ```html
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    <!-- Optional: Styles for highlight.js -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css">
    <!-- If using marked-highlight extension -->
    <script src="https://cdn.jsdelivr.net/npm/marked-highlight/lib/index.umd.js"></script>
    ```
    *Note: Ensure `window.marked` and `window.hljs` (and `window.markedHighlight` if using the extension) are available before `TB.ui.MarkdownRenderer.init()` is called or any markdown rendering is attempted.*

### Installation
`tbjs` is distributed as `tbjs.js` and `tbjs.css`.

1.  **Build `tbjs`**: If you have the source, run `npm run build` to generate the `dist/` folder.
2.  **Include files in your HTML**:
    ```html
    <link rel="stylesheet" href="path/to/your/tbjs/dist/tbjs.css">
    <script type="module" src="path/to/your/tbjs/dist/tbjs.js"></script>
    ```
    If you are using `tbjs` as a module in your own build system, you can import it:
    ```javascript
    import TB from 'path/to/your/tbjs/src/index.js'; // Or from dist/tbjs.js if UMD
    ```

### HTML Setup
Your main HTML file (`index.html`) should have a root element for your application content and optionally, containers for background effects or 3D graphics.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My tbjs App</title>

    <!-- tbjs CSS -->
    <link rel="stylesheet" href="path/to/tbjs/dist/tbjs.css">

    <!-- Tailwind (if you're using it directly in your app) -->
    <!-- <script src="https://cdn.tailwindcss.com"></script> -->
    <!-- Or your compiled Tailwind CSS -->
    <link rel="stylesheet" href="path/to/your/app.css">

    <!-- HTMX -->
    <script src="https://unpkg.com/htmx.org@2.0.0/dist/htmx.min.js"></script>

    <!-- Three.js (if using graphics module) -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/0.153.0/three.min.js"></script>

    <!-- Marked & Highlight.js (if using MarkdownRenderer) -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked-highlight/lib/index.umd.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/atom-one-dark.min.css">


    <!-- Material Symbols (used by some components) -->
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />

    <!-- tbjs Script (type="module" if loading the ES module directly) -->
    <!-- Option 1: Loading tbjs as a module -->
    <!-- <script type="module" src="path/to/tbjs/src/index.js"></script> -->
    <!-- Option 2: Loading the UMD build -->
    <script src="path/to/tbjs/dist/tbjs.js"></script>
</head>
<body>
    <!-- Dedicated background container (used by TB.ui.theme) -->
    <div id="appBackgroundContainer"></div>

    <!-- Optional: Container for 3D graphics (used by TB.graphics) -->
    <div id="threeDScene" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -10;"></div>

    <!-- Main application root -->
    <div id="app-root">
        <!-- Initial content or loader can go here -->
        <div class="loaderCenter" style="display: flex; justify-content: center; align-items: center; height: 100vh;">
            Loading application...
        </div>
    </div>

    <!-- Your main application script -->
    <script type="module">
        // Import TB if using ES Modules and not UMD global
        // import TB from 'path/to/tbjs/src/index.js'; // or from 'path/to/tbjs/dist/tbjs.js' if it's an ES module build

        // Wait for the DOM to be fully loaded
        document.addEventListener('DOMContentLoaded', () => {
            // Access TB (it's global if you included the UMD build)
            if (window.TB) {
                window.TB.init({
                    appRootId: 'app-root', // Matches the div above
                    baseApiUrl: '/api',    // Your backend API base URL
                    // baseFileUrl: window.location.origin + '/app/', // If HTML files are in a subfolder
                    initialState: {
                        appName: 'My Awesome App'
                    },
                    themeSettings: {
                        defaultPreference: 'system', // 'light', 'dark', or 'system'
                        background: {
                            type: 'color', // 'color', 'image', or '3d'
                            light: { color: '#f0f0f0' },
                            dark: { color: '#202020' },
                            // placeholder: { image: '/path/to/placeholder.jpg', displayUntil3DReady: true }
                        }
                    },
                    routes: [
                        // You can predefine routes, though router primarily fetches HTML
                        // { path: '/home', component: 'views/home.html' }
                    ],
                    logLevel: 'debug', // 'debug', 'info', 'warn', 'error'
                    serviceWorker: {
                        enabled: false, // Set to true to enable
                        url: '/sw.js'
                    }
                });
            } else {
                console.error("TB object not found. Ensure tbjs.js is loaded correctly.");
            }
        });
    </script>
</body>
</html>
```

### Initialization
Initialize `tbjs` by calling `TB.init()` with your application's configuration. This is typically done in a `<script type="module">` tag at the end of your `<body>` or after the `DOMContentLoaded` event.

```javascript
// In your main application script (e.g., app.js or inline script)
document.addEventListener('DOMContentLoaded', () => {
    TB.init({
        appRootId: 'app-root',
        baseApiUrl: '/api/v1', // Example: your backend API
        // baseFileUrl: window.location.origin, // Default
        initialState: {
            userPreferences: {
                notifications: true
            }
        },
        themeSettings: {
            defaultPreference: 'dark', // 'light', 'dark', or 'system'
            background: {
                type: '3d', // 'color', 'image', '3d'
                light: { color: '#FFFFFF' }, // Fallback if 3D fails or not used
                dark: { color: '#121212' },  // Fallback
                placeholder: { image: '/images/background-placeholder.jpg', displayUntil3DReady: true }
            }
        },
        routes: [
             // Example: Define a route that maps to an HTML file.
             // { path: '/dashboard', view: '/web/pages/dashboard.html' }, // Not directly used by current router.navigateTo structure
        ],
        logLevel: 'info', // 'debug', 'info', 'warn', 'error', 'none'
        isProduction: false, // Manually set or let tbjs infer
        serviceWorker: {
            enabled: true,
            url: '/custom-sw.js',
            scope: '/'
        }
    });

    // Example: Listen for tbjs initialization completion
    TB.events.on('tbjs:initialized', (tbInstance) => {
        console.log('tbjs is ready!', tbInstance.VERSION);
        // You can now safely use all TB modules
        // Example: TB.router.navigateTo('/home');
    });
});
```
If the `appRootId` element is not found on the page when `TB.init()` is called, `tbjs` will attempt to redirect to `/index.html` after storing the intended path in `sessionStorage`. This helps handle deep linking when the main HTML shell hasn't loaded yet.

---

## 3. Core Modules

### Configuration (`TB.config`)
The configuration module (`TB.config`) manages all framework and application settings. It's initialized by `TB.init()` with default values merged with your provided configuration.

**Default Configuration Structure:**
```javascript
{
    appRootId: 'app-root',
    baseApiUrl: '/api',
    baseFileUrl: window.location.origin, // Base URL for HTML views
    initialState: {},
    themeSettings: {
        defaultPreference: 'system', // 'light', 'dark', 'system'
        background: {
            type: 'color', // '3d', 'image', 'color', 'none'
            light: { color: '#FFFFFF', image: '' },
            dark: { color: '#121212', image: '' },
            placeholder: { image: '', displayUntil3DReady: true }
        }
    },
    routes: [], // Primarily for reference, router fetches HTML directly
    logLevel: 'info',
    isProduction: /* inferred or user-set */,
    serviceWorker: {
        enabled: false,
        url: '/sw.js',
        scope: '/'
    }
}
```

**Methods:**
*   `TB.config.init(initialUserConfig)`: Called internally by `TB.init()`. Merges user config with defaults.
    *   `baseApiUrl` is made absolute. If relative (e.g., `/api`), it's prepended with `window.location.origin`. If just `api`, it becomes `/api`.
    *   `baseFileUrl` ensures it ends with a `/` if it contains a path.
    *   `isProduction` is inferred based on hostname (`localhost`, `127.0.0.1`) if not explicitly set.
*   `TB.config.get(key)`: Retrieves a configuration value. Supports dot notation for nested properties (e.g., `TB.config.get('themeSettings.background.type')`).
*   `TB.config.getAll()`: Returns a copy of the entire configuration object.
*   `TB.config.set(key, value)`: Sets a configuration value. Supports dot notation. *Note: Use with caution after initialization, as not all modules may react to live changes.*

**Example:**
```javascript
const apiUrl = TB.config.get('baseApiUrl');
console.log('API URL:', apiUrl);

const isProd = TB.config.get('isProduction');
if (isProd) {
    TB.logger.setLevel('warn');
}
```

### State Management (`TB.state`)
`TB.state` provides a simple, centralized store for your application's state.

**Methods:**
*   `TB.state.init(initialState)`: Called internally by `TB.init()`. Loads any persisted state from `localStorage`.
*   `TB.state.get(key)`: Retrieves a state value. Supports dot notation. If `key` is undefined, returns a copy of the entire state.
*   `TB.state.set(key, value, options = { persist: false })`: Sets a state value.
    *   `options.persist`: If `true`, the top-level key of this state will be saved to `localStorage` and reloaded on next init.
    *   Emits `state:changed` event with `{ key, value, fullState }`.
    *   Emits `state:changed:your:key` (dots replaced with colons) for more specific listeners.
*   `TB.state.delete(key, options = { persist: false })`: Deletes a key from the state. Handles persisted state removal.
*   Legacy methods for simple key-value persistence (discouraged for new code, prefer structured state with `persist` option):
    *   `TB.state.initVar(v_name, v_value)`: Initializes if not set, persists.
    *   `TB.state.delVar(v_name)`: Deletes, persists change.
    *   `TB.state.getVar(v_name)`: Gets value.
    *   `TB.state.setVar(v_name, v_value)`: Sets value, persists.

**Example:**
```javascript
// Set initial user data (e.g., after login)
TB.state.set('user.profile', { name: 'Alice', theme: 'dark' }, { persist: true });

// Get user name
const userName = TB.state.get('user.profile.name'); // Alice

// Listen for changes to a specific part of the state
TB.events.on('state:changed:user:profile:theme', (newTheme) => {
    console.log('User theme changed to:', newTheme);
    // Update UI or apply theme
});

// Update the theme
TB.state.set('user.profile.theme', 'light'); // Listener above will be triggered
```

### Routing (`TB.router`)
`TB.router` handles client-side navigation for your SPA. It fetches HTML content for views and updates the DOM.

**Key Features:**
*   Loads HTML content from `TB.config.get('baseFileUrl') + path`.
*   Updates browser history (`pushState`, `replaceState`).
*   Handles `popstate` events (browser back/forward).
*   Intercepts clicks on local links.
*   Manages script execution for dynamically loaded content:
    *   Scripts with `src` are loaded and executed once (cached by URL).
    *   Inline scripts are executed.
    *   Inline scripts with `unsave="true"` are executed via a Blob URL (fresh execution each time).
    *   **Important**: It attempts to prevent re-execution of main application bundle scripts (e.g., `main.js`, `bundle.js`) if they are found within fetched HTML.
*   Optional session-based caching for HTML content.
*   Emits router events (`router:beforeNavigation`, `router:navigationSuccess`, `router:navigationError`, `router:contentProcessed`).

**Methods:**
*   `TB.router.init(rootElement, predefinedRoutes)`: Initializes the router.
    *   `rootElement`: The DOM element where views will be rendered.
    *   `predefinedRoutes`: (Currently for reference) Array of route objects.
    *   Automatically navigates to the initial URL.
*   `TB.router.navigateTo(path, replace = false, isInitialLoad = false)`: Navigates to the given path.
    *   `path`: The URL path (e.g., `/about`, `/users/123?param=value#section`). Can be absolute or relative.
    *   `replace`: If `true`, uses `history.replaceState` instead of `history.pushState`.
    *   `isInitialLoad`: Internal flag for the first navigation.
    *   Handles 404 errors by trying to navigate to `/web/assets/404.html`.
    *   Handles 401 errors by trying to navigate to `/web/assets/401.html`.
*   `TB.router.getCurrentPath()`: Returns the current normalized path.
*   `TB.router.clearCache(path)`: Clears the sessionStorage cache for a specific path or all cached pages if `path` is omitted. (Only if `USE_SESSION_CACHE` is true in `router.js`).

**Example: Navigating and Handling Events**
```html
<!-- In your main HTML -->
<nav>
    <a href="/home">Home</a>
    <a href="/products">Products</a>
    <a href="/web/assets/contact.html">Contact (relative to baseFileUrl)</a>
</nav>
<div id="app-root"></div>
```

```javascript
// Initialize router (typically done in TB.init)
// TB.router.init(document.getElementById('app-root'));

// Programmatic navigation
TB.router.navigateTo('/home');

TB.events.on('router:navigationSuccess', ({ path, contentSource }) => {
    console.log(`Navigated to ${path} (from ${contentSource})`);
    // TB.ui.processDynamicContent is called internally by the router for the appRootElement
    // but if you load content into other areas, you might call it manually:
    // TB.ui.processDynamicContent(document.getElementById('some-other-area'));
});

TB.events.on('router:navigationError', ({ path, error }) => {
    console.error(`Failed to navigate to ${path}:`, error);
    TB.ui.Toast.showError(`Could not load page: ${path}`);
});
```

### API Communication (`TB.api`)
`TB.api` is responsible for all backend communication, supporting standard HTTP requests and Tauri `invoke` calls.

**Key Structures:**
*   **`Result` Object**: Standardized wrapper for API responses.
    *   `origin`: Source of the data (e.g., `['http']`, `['tauri']`).
    *   `error`: Error type (e.g., `TB.api.ToolBoxError.none`, `TB.api.ToolBoxError.input_error`).
    *   `result`: A `ToolBoxResult` object.
        *   `data_to`: Interface target (e.g., `TB.api.ToolBoxInterfaces.api`, `.cli`).
        *   `data_info`: Additional info about the data.
        *   `data`: The actual payload.
    *   `info`: A `ToolBoxInfo` object.
        *   `exec_code`: Execution code (e.g., HTTP status or custom code).
        *   `help_text`: Descriptive message.
    *   Methods: `.log()`, `.html()`, `.get()` (returns `result.data`).
*   `ToolBoxError`: Enum for error types.
*   `ToolBoxInterfaces`: Enum for data destinations.

**Main Method: `TB.api.request()`**
```javascript
async TB.api.request(
    moduleName,      // string: Backend module/class OR full path (e.g., '/special/endpoint')
    functionName,    // string: Backend function/method (ignored if moduleName is full path)
                     // OR object: Query parameters for GET/DELETE if moduleName is full path
    payload = null,  // object|string: Data to send. If string, used as query params for GET/POST-URL.
    method = 'POST', // string: HTTP method ('GET', 'POST', 'PUT', 'DELETE', etc.)
    useTauri = 'auto', // string: 'auto', 'force' (Tauri only), or 'never' (HTTP only)
    isSpecialAuthRoute = false // boolean: For routes with custom auth header handling (rarely needed)
)
```
*   **Tauri Integration**: If `env.isTauri()` is true and `useTauri` is 'auto' or 'force', it attempts `window.__TAURI__.invoke("moduleName.functionName", payload)`. Falls back to HTTP if 'auto' and invoke fails.
*   **HTTP Requests**:
    *   URL construction:
        *   Standard: `baseApiUrl/moduleName/functionName`
        *   Full path: `baseApiUrl/moduleName` (where `moduleName` is e.g., `/login`)
    *   Headers:
        *   `Content-Type: application/json` and `Accept: application/json` by default for JSON.
        *   `Authorization: Bearer <token>` is added if `TB.state.get('user.token')` exists (unless `isSpecialAuthRoute` has special handling, though current `_getRequestHeaders` adds it generally).
    *   Payload:
        *   GET/DELETE: Payload (object or string) becomes URL query parameters.
        *   POST/PUT/PATCH: Object payload is JSON.stringified. String payload for POST can be query params if URL doesn't already have them.
    *   Response Handling:
        *   Automatically parses JSON responses.
        *   Handles non-JSON responses (e.g., 204 No Content) gracefully.
        *   Wraps responses (or errors) in a `Result` object.

**Helper Methods:**
*   `TB.api.fetchHtml(path)`: Fetches HTML content, typically used by the router. Path is relative to `baseFileUrl`.
*   `TB.api.httpPostData(module_name, function_name, data)`: Alias for `request(..., 'POST')`.
*   `TB.api.AuthHttpPostData(username)`: Specific method for session validation. Calls `/validateSession` with `jwt_claim_device` and `Username`.
*   `TB.api.logoutServer()`: Calls `/web/logoutS` to notify the server of logout.

**Example: Making a GET Request**
```javascript
async function fetchProducts() {
    TB.ui.Loader.show('Fetching products...');
    try {
        // GET /api/ProductManager/getProductsList?category=electronics&limit=10
        const result = await TB.api.request(
            'ProductManager',
            'getProductsList',
            { category: 'electronics', limit: 10 },
            'GET'
        );

        if (result.error === TB.api.ToolBoxError.none) {
            const products = result.get(); // products === result.result.data
            console.log('Products:', products);
            TB.state.set('products.list', products);
        } else {
            TB.logger.error('Failed to fetch products:', result.info.help_text);
            TB.ui.Toast.showError(result.info.help_text || 'Could not load products.');
        }
    } catch (error) { // Network errors or other exceptions
        TB.logger.error('Network error fetching products:', error);
        TB.ui.Toast.showError('Network error. Please try again.');
    } finally {
        TB.ui.Loader.hide();
    }
}

fetchProducts();
```

**Example: Tauri Invoke or HTTP POST**
```javascript
async function saveData(dataToSave) {
    const result = await TB.api.request(
        'DataManager',  // Tauri: DataManager.save_data
        'save_data',    // HTTP: /api/DataManager/save_data
        dataToSave,     // Payload
        'POST',         // HTTP Method
        'auto'          // Try Tauri first, then HTTP
    );

    result.log(); // Logs the structured result to console

    if (result.error === TB.api.ToolBoxError.none) {
        TB.ui.Toast.showSuccess('Data saved successfully!');
        return result.get(); // Return the data from server response
    } else {
        TB.ui.Toast.showError(`Save failed: ${result.info.help_text}`);
        return null;
    }
}
```

### Event System (`TB.events`)
A simple publish/subscribe system for decoupled communication.

**Methods:**
*   `TB.events.on(eventName, callback)`: Subscribes to an event.
*   `TB.events.off(eventName, callback)`: Unsubscribes from an event.
*   `TB.events.emit(eventName, data)`: Publishes an event with optional data.
*   `TB.events.once(eventName, callback)`: Subscribes to an event for a single occurrence.

**Example:**
```javascript
// Module A
function doSomething() {
    // ...
    TB.events.emit('user:actionCompleted', { action: 'saveSettings', status: 'success' });
}

// Module B
TB.events.on('user:actionCompleted', (eventData) => {
    if (eventData.action === 'saveSettings' && eventData.status === 'success') {
        console.log('User settings saved!');
    }
});
```
Framework Core Events (Examples):
*   `state:changed`
*   `state:changed:path:to:key`
*   `router:beforeNavigation`, `router:navigationSuccess`, `router:navigationError`, `router:contentProcessed`
*   `theme:changed`
*   `tbjs:initialized`
*   `api:networkError`
*   `graphics:initialized`, `graphics:disposed`
*   `cookieConsent:updated`
*   `user:stateChanged`, `user:loggedOut`

### Logging (`TB.logger`)
Provides prefixed and timestamped console logging with different levels.

**Methods:**
*   `TB.logger.init({ logLevel })`: Called internally.
*   `TB.logger.setLevel(levelName)`: Sets the minimum log level ('debug', 'info', 'warn', 'error', 'none').
*   `TB.logger.debug(...args)`
*   `TB.logger.log(...args)` (alias for `info`)
*   `TB.logger.info(...args)`
*   `TB.logger.warn(...args)`
*   `TB.logger.error(...args)`

**Example:**
```javascript
TB.logger.debug('This is a debug message with an object:', { id: 1, name: 'Test' });
TB.logger.info('Application started.');
TB.logger.warn('Something might be wrong here.');
TB.logger.error('A critical error occurred!', new Error('Oops'));
```
Log level is configured via `TB.config.logLevel` during `TB.init`.

### Environment Detection (`TB.env`)
Detects the current runtime environment.

**Methods:**
*   `TB.env.detect()`: Called internally by `TB.init()`.
*   `TB.env.isTauri()`: Returns `true` if running in a Tauri environment.
*   `TB.env.isWeb()`: Returns `true` if running in a standard web browser environment.
*   `TB.env.isMobile()`: (Placeholder) Intended for Tauri mobile detection, currently may not be fully implemented.

**Example:**
```javascript
if (TB.env.isTauri()) {
    TB.logger.log('Running in Tauri, specific Tauri features can be used.');
    // Example: window.__TAURI__.fs.readTextFile(...)
} else if (TB.env.isWeb()) {
    TB.logger.log('Running in a web browser.');
}
```

### Cryptography (`TB.crypto`)
Provides various cryptographic utilities, including WebAuthn support.

**Key Management:**
*   `TB.crypto.generateAsymmetricKeys()`: Generates RSA-OAEP key pair (PEM and base64).
*   `TB.crypto.decryptAsymmetric(encryptedTextBase64, privateKeyBase64, convertHex = false)`: Decrypts RSA-OAEP encrypted text.
*   `TB.crypto.signMessage(privateKeyBase64, message)`: Signs a message using RSA-PSS with the private key.
*   `TB.crypto.storePrivateKey(privateKeyBase64, username)`: Stores private key in `localStorage`.
*   `TB.crypto.retrievePrivateKey(username)`: Retrieves private key from `localStorage`.

**Symmetric Encryption (Example - requires careful IV handling):**
*   `TB.crypto.generateSymmetricKey()`: Generates an AES-GCM key (base64 raw).
*   `TB.crypto.decryptSymmetric(encryptedDataB64, password)`: Decrypts AES-GCM data. *Note: This implementation assumes the IV is prepended to the ciphertext (first 12 bytes). The `password` is used to derive the key via PBKDF2.*

**WebAuthn:**
*   **`getRpId()` (internal, used by WebAuthn functions):** Determines the Relying Party ID based on `TB.config.get('baseAppUrl')` or `window.location.hostname`. For localhost, it's "localhost".
*   `TB.crypto.registerWebAuthnCredential(registrationData, sing)`:
    *   `registrationData`: `{ challenge, userId, username }`.
        *   `challenge`: Server-provided challenge (string, base64url encoded then decoded to ArrayBuffer).
        *   `userId`: Server-provided user ID (string, base64url encoded then decoded to ArrayBuffer).
        *   `username`: User's display name.
    *   `sing`: Additional data (e.g., session token) that might be included in the payload sent to the server.
    *   Calls `navigator.credentials.create()`.
    *   Returns a promise resolving to the payload object to be sent to the server for `/register_user_personal_key` (or similar endpoint).
*   `TB.crypto.authorizeWebAuthnCredential(rawIdAsBase64, challenge, username)`:
    *   `rawIdAsBase64`: Base64 representation of the credential's rawId (from server).
    *   `challenge`: Server-provided challenge string.
    *   `username`: User's display name.
    *   Calls `navigator.credentials.get()`.
    *   Returns a promise resolving to the payload object to be sent to the server for `/validate_persona` (or similar endpoint).

**Helper Functions:**
*   `arrayBufferToBase64(buffer)`
*   `base64ToArrayBuffer(base64)`
*   `strToArrayBuffer(str)`
*   `arrayBufferToStr(arrayBuffer)`
*   And others...

**Example: Registering a WebAuthn Credential (Passkey)**
```javascript
// Assuming 'username' is known and user is authenticated to add a new key
async function registerNewPasskey(username) {
    try {
        // 1. Client requests challenge from server for this user
        const challengeRes = await TB.api.request('AuthManager', 'getWebAuthnRegistrationChallenge', { username }, 'POST');
        if (challengeRes.error !== TB.api.ToolBoxError.none || !challengeRes.get()?.challengeInfo) {
            throw new Error(challengeRes.info.help_text || "Failed to get registration challenge.");
        }
        const { challenge, userId } = challengeRes.get().challengeInfo; // Server provides its internal userId for WebAuthn

        // 2. Client uses TB.crypto to create credential
        const currentSessionToken = TB.user.getToken(); // Example for 'sing' parameter
        const webAuthnPayload = await TB.crypto.registerWebAuthnCredential(
            { challenge, userId, username },
            currentSessionToken
        );

        // 3. Client sends new credential to server for verification and storage
        const registrationResult = await TB.api.request('AuthManager', 'completeWebAuthnRegistration', webAuthnPayload, 'POST');

        if (registrationResult.error === TB.api.ToolBoxError.none) {
            TB.ui.Toast.showSuccess('Passkey registered successfully!');
        } else {
            TB.ui.Toast.showError(`Passkey registration failed: ${registrationResult.info.help_text}`);
        }
    } catch (error) {
        TB.logger.error('[WebAuthnDemo] Registration error:', error);
        TB.ui.Toast.showError(error.message || 'An error occurred during passkey registration.');
    }
}
```

### User Management (`TB.user`)
Manages user authentication state, sessions, and user-specific data.

**State:** The user's state (isAuthenticated, username, token, etc.) is stored under `TB.state.get('user')`.

**Methods:**
*   `TB.user.init(forceServerFetch = false)`: Initializes the user module.
    *   Loads session from `localStorage` (`tbjs_user_session`).
    *   If authenticated, validates the session with the server (`TB.api.AuthHttpPostData`).
    *   Synchronizes `userData` based on timestamps (`tbjs_user_data_timestamp`) if `forceServerFetch` is true or server data is newer.
*   `TB.user.signup(username, email, initiationKey, registerAsPersona = false)`: Placeholder for signup flow.
*   `TB.user.loginWithDeviceKey(username)`: Performs login using a locally stored asymmetric key.
    1.  Retrieves private key using `TB.crypto.retrievePrivateKey()`.
    2.  Requests challenge from `CloudM.AuthManager.get_to_sing_data`.
    3.  Signs challenge using `TB.crypto.signMessage()`.
    4.  Validates signature with `CloudM.AuthManager.validate_device`.
    5.  If successful, updates user state with token and user data.
*   `TB.user.loginWithWebAuthn(username)`: Performs WebAuthn (passkey) login.
    1.  Requests challenge & rawId from `CloudM.AuthManager.get_to_sing_data` (`personal_key: true`).
    2.  Calls `TB.crypto.authorizeWebAuthnCredential()`.
    3.  Sends assertion to `CloudM.AuthManager.validate_persona`.
*   `TB.user.requestMagicLink(username)`: Requests a magic link email via `CloudM.AuthManager.get_magic_link_email`.
*   `TB.user.registerDeviceWithInvitation(username, invitationKey)`: Registers a new device using an invitation key.
    1.  Generates new asymmetric keys (`TB.crypto.generateAsymmetricKeys()`).
    2.  Stores private key (`TB.crypto.storePrivateKey()`).
    3.  Sends public key and invitation to `CloudM.AuthManager.add_user_device`.
    4.  Attempts login via `loginWithDeviceKey()` upon success.
*   `TB.user.registerWebAuthnForCurrentUser(username)`: Registers a WebAuthn credential for an already authenticated user.
*   `TB.user.logout(notifyServer = true)`: Logs out the user, clears local session, and optionally notifies the server via `TB.api.logoutServer()`.
*   `TB.user.checkSessionValidity()`: Checks if current session token is valid via `/IsValidSession`.
*   **Getters:**
    *   `TB.user.isAuthenticated()`
    *   `TB.user.getUsername()`
    *   `TB.user.getUserLevel()`
    *   `TB.user.getToken()`
    *   `TB.user.isDeviceRegisteredWithKey()`
*   **User Data Management:**
    *   `TB.user.getUserData(key)`: Gets a specific piece of user data from `TB.state.get('user.userData')`.
    *   `TB.user.setUserData(keyOrObject, value, syncToServer = false)`: Sets user data locally. If `syncToServer` is true, calls `syncUserData`.
    *   `TB.user.syncUserData(updatedFields = null)`: Syncs `userData` (or specified fields) to the server via `UserManager.updateUserData`.
    *   `TB.user.fetchUserData()`: Fetches all user data from `UserManager.getUserData`.

**Example: Login and Accessing User Info**
```javascript
async function handleLogin(username) {
    const loginResult = await TB.user.loginWithDeviceKey(username);
    if (loginResult.success) {
        TB.ui.Toast.showSuccess(`Welcome, ${TB.user.getUsername()}!`);
        TB.router.navigateTo('/dashboard');
    } else {
        TB.ui.Toast.showError(loginResult.message);
    }
}

// Check if user is logged in before accessing a protected route
if (!TB.user.isAuthenticated()) {
    TB.router.navigateTo('/login');
} else {
    console.log('User Level:', TB.user.getUserLevel());
}
```

### Server-Sent Events (`TB.sse`)
Manages Server-Sent Event (SSE) connections.

**Methods:**
*   `TB.sse.connect(url, options = {})`: Establishes an SSE connection.
    *   `url`: The SSE endpoint URL.
    *   `options`:
        *   `eventSourceOptions`: Options passed directly to the `EventSource` constructor (e.g., `{ withCredentials: true }`).
        *   `onOpen`: Callback for `open` event.
        *   `onError`: Callback for `error` event.
        *   `onMessage`: Callback for generic `message` events.
        *   `listeners`: An object of `{ eventName: handlerFunction }` for custom named SSE events.
    *   Emits events like `sse:open:<url>`, `sse:error:<url>`, `sse:message:<url>`, `sse:event:<url>:<eventName>`.
*   `TB.sse.disconnect(url)`: Closes a specific SSE connection.
*   `TB.sse.disconnectAll()`: Closes all active SSE connections.
*   `TB.sse.getConnection(url)`: Returns the `EventSource` object for a given URL.

**Example:**
```javascript
const sseConnection = TB.sse.connect('/api/notifications', {
    onOpen: () => TB.logger.info('SSE connection for notifications opened.'),
    onError: (err) => TB.logger.error('SSE notifications error:', err),
    listeners: {
        'new_message': (data) => {
            TB.ui.Toast.showInfo(`New message: ${data.text}`);
        },
        'user_update': (data) => {
            TB.state.set('userDetails', data);
        }
    }
});

// To close
// TB.sse.disconnect('/api/notifications');
```

### Service Worker (`TB.sw`)
Manages the registration and communication with your application's Service Worker.

**Configuration (`TB.config.serviceWorker`):**
*   `enabled`: (boolean) Master switch for SW registration.
*   `url`: (string) Path to your `sw.js` file (default: `/sw.js`).
*   `scope`: (string) Scope for the Service Worker (default: `/`).

**Methods:**
*   `TB.sw.register()`: Registers the Service Worker based on configuration.
    *   Handles `updatefound` and state changes of the installing worker.
    *   Emits `sw:updateAvailable` or `sw:contentCached`.
*   `TB.sw.unregister()`: Unregisters all active Service Workers for the current origin.
*   `TB.sw.sendMessage(message)`: Sends a message to the active Service Worker controller and returns a Promise for the response.

**Example:**
```javascript
// In TB.init config:
// serviceWorker: { enabled: true, url: '/my-app-sw.js' }

// tbjs will attempt to register it automatically.

// Listening for updates
TB.events.on('sw:updateAvailable', ({ registration }) => {
    if (confirm('A new version is available. Reload to update?')) {
        // Logic to skip waiting and activate new SW
        registration.waiting.postMessage({ type: 'SKIP_WAITING' });
        // Usually, you'd listen for 'controllerchange' event then reload
        navigator.serviceWorker.addEventListener('controllerchange', () => {
            window.location.reload();
        });
    }
});

// Sending a message to SW
async function clearAppCacheViaSW() {
    try {
        const response = await TB.sw.sendMessage({ type: 'CLEAR_CACHE', cacheName: 'app-data-v1' });
        console.log('SW Cache Clear Response:', response);
        TB.ui.Toast.showSuccess('App cache cleared by Service Worker.');
    } catch (error) {
        TB.ui.Toast.showError(`Error communicating with SW: ${error}`);
    }
}
```

### Utilities (`TB.utils`)
A collection of general-purpose helper functions.

**Methods:**
*   `TB.utils.autocomplete(inputElement, array)`: Adds basic autocomplete to an input field.
*   `TB.utils.debounce(func, delay)`: Debounces a function.
*   `TB.utils.throttle(func, limit)`: Throttles a function.
*   `TB.utils.uniqueId(prefix = 'id-')`: Generates a simple unique ID.
*   `TB.utils.deepClone(obj)`: Deep clones an object or array.
*   `TB.utils.cleanUrl(url)`: Removes protocol from a URL.

**Example:**
```javascript
const myInput = document.getElementById('search');
const suggestions = ['Apple', 'Banana', 'Orange', 'Apricot'];
TB.utils.autocomplete(myInput, suggestions);

const debouncedSave = TB.utils.debounce((data) => {
    console.log('Saving data:', data);
    // TB.api.request(...) to save
}, 500);

myInput.addEventListener('input', (e) => debouncedSave(e.target.value));
```

---

## 4. UI System (`TB.ui`)
The `TB.ui` namespace contains modules and components for managing the user interface.

### Theme Management (`TB.ui.theme`)
Manages application themes (light/dark mode) and dynamic backgrounds.

**Configuration (`TB.config.themeSettings`):**
*   `defaultPreference`: 'light', 'dark', or 'system'.
*   `background`: Object defining background types and sources.
    *   `type`: 'color', 'image', '3d', 'none'.
    *   `light`: `{ color: '#hex', image: 'path/to/image.jpg' }`
    *   `dark`: `{ color: '#hex', image: 'path/to/image.jpg' }`
    *   `placeholder`: `{ image: 'path/to/placeholder.jpg', displayUntil3DReady: true }` (used when `type` is '3d').

**Initialization:**
`TB.ui.theme.init()` is called by `TB.init()` using `TB.config.get('themeSettings')`. It:
*   Loads user preference from `localStorage` or defaults.
*   Sets up a background container div (`#appBackgroundContainer`) if not present.
*   Applies the initial theme and background.
*   Listens for system theme changes and `graphics:initialized`/`graphics:disposed` events to update the background.

**Methods:**
*   `TB.ui.theme.setPreference(preference)`: Sets the theme preference ('light', 'dark', 'system') and saves to `localStorage`. Updates the theme immediately.
*   `TB.ui.theme.togglePreference()`: Toggles between 'light' and 'dark' modes. If current preference is 'system', it effectively picks the opposite of the current *effective* mode.
*   `TB.ui.theme.getCurrentMode()`: Returns the current active mode ('light' or 'dark').
*   `TB.ui.theme.getPreference()`: Returns the user's set preference ('light', 'dark', or 'system').
*   `TB.ui.theme.getBackgroundConfig()`: Returns the current background configuration.

**Events:**
*   `theme:changed`: Emitted with `{ mode: 'light'|'dark' }` when the effective theme changes.
*   The `TB.graphics.updateTheme(mode)` method is called internally when the theme changes and graphics are active.

**Example:**
```javascript
// In your app initialization (TB.init config)
// themeSettings: {
//   defaultPreference: 'system',
//   background: {
//     type: 'image',
//     light: { image: '/images/bg-light.jpg', color: '#E0E0E0' },
//     dark: { image: '/images/bg-dark.jpg', color: '#303030' }
//   }
// }

// Toggle theme with a button
const themeToggleButton = document.getElementById('theme-toggle-btn');
themeToggleButton.addEventListener('click', () => {
    TB.ui.theme.togglePreference();
});

// Listen for theme changes to update other UI elements
TB.events.on('theme:changed', (eventData) => {
    console.log('Theme is now:', eventData.mode);
    // Update icons, specific component styles, etc.
});
```
The background container `#appBackgroundContainer` is styled to be fixed and behind all other content (`z-index: -1`). The 3D graphics, if used as a background, are expected to render into `#threeDScene`.

### Graphics (`TB.graphics`)
Manages 3D rendering using Three.js, often used for dynamic backgrounds.

**Initialization:**
`TB.graphics.init(canvasContainerSelector, options = {})`
*   `canvasContainerSelector`: A CSS selector for the DOM element where the Three.js canvas will be appended (e.g., `#threeDScene`).
*   `options`:
    *   `cameraY`, `cameraZ`: Initial camera position.
    *   `sierpinskiDepth`: Depth for the default Sierpinski tetrahedron animation.
    *   `loaderHideDelay`: Delay in ms before hiding a `.loaderCenter` element after graphics init.

**Key Features & Methods:**
*   Creates a WebGLRenderer, Scene, and PerspectiveCamera.
*   Builds a default Sierpinski tetrahedron fractal.
*   Adds ambient and point lights.
*   `TB.graphics.updateTheme(themeMode)`: Adjusts light colors based on 'light' or 'dark' mode. Called automatically by `TB.ui.theme`.
*   `TB.graphics.setSierpinskiDepth(newDepth)`: Rebuilds the fractal with new depth.
*   `TB.graphics.setAnimationSpeed(x, y, z, factor)`: Controls the rotation speed of the main 3D object.
*   `TB.graphics.adjustCameraZoom(delta)` / `TB.graphics.setCameraZoom(value)`: Controls camera Z position.
*   Interactive rotation via mouse/touch drag.
*   **Animation Sequences:**
    *   `TB.graphics.playAnimationSequence(sequenceString, onComplete, baseSpeed, speedFactor)`: Plays a predefined animation sequence.
        *   `sequenceString`: Colon-separated steps, e.g., `"R1+32:P2-51:Y1+15"`.
            *   Format: `Type(1)Repeat(N)Direction(1)Speed(1)Complexity(1)`
            *   Type: `R` (Roll/X), `P` (Pan/Z), `Y` (Yaw/Y), `Z` (Zoom - placeholder).
            *   Repeat: Number of repetitions.
            *   Direction: `+` or `-`.
            *   Speed: 1-9.
            *   Complexity: 1-9 (influences duration).
    *   `TB.graphics.stopAnimationSequence()`: Stops the current sequence.
*   `TB.graphics.pause()` / `TB.graphics.resume()`: Pause/resume rendering loop.
*   `TB.graphics.dispose()`: Cleans up Three.js resources and removes event listeners.

**Events:**
*   `graphics:initialized`: Emitted when graphics setup is complete. `TB.ui.theme` listens to this.
*   `graphics:disposed`: Emitted on cleanup.

**Example: Initializing 3D Background**
```html
<!-- In your HTML -->
<div id="myThreeCanvasContainer" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -10;"></div>
```
```javascript
// In your TB.init config:
// themeSettings: {
//   background: {
//     type: '3d'
//   }
// }

// If TB.ui.theme is configured with type '3d', it will attempt to initialize graphics automatically.
// However, you might need to ensure the container exists and is ready.
// Or, initialize manually if you don't use themeSettings.background for it:
document.addEventListener('DOMContentLoaded', () => {
    if (TB.config.get('themeSettings.background.type') === '3d') {
        const graphicsContext = TB.graphics.init('#myThreeCanvasContainer', {
            sierpinskiDepth: 4,
            cameraZ: 8
        });
        if (graphicsContext) {
            console.log('3D Graphics initialized for background.');
        }
    }
});

// Example: Play an animation sequence
// TB.graphics.playAnimationSequence("R5+55:P2-22", () => console.log("Sequence done!"));
```

### HTMX Integration (`TB.ui.htmxIntegration`)
Manages interactions with HTMX.

**Initialization:**
`TB.ui.htmxIntegration.init()` is called by `TB.init()`. It sets up global event listeners for HTMX events.

**Event Handling:**
*   **`htmx:afterSwap`**:
    *   Triggered after HTMX swaps content into the DOM.
    *   Calls `TB.ui.processDynamicContent(event.detail.target)` to initialize `tbjs` components, run scripts, and apply Markdown rendering on the newly added content.
*   **`htmx:afterRequest`**:
    *   Triggered after an HTMX AJAX request completes.
    *   If the response is JSON:
        *   It attempts to wrap the JSON data in a `TB.api.Result` object.
        *   Logs errors and shows a `TB.ui.Toast` if the `Result` indicates an error.
        *   If the `Result.result.data_to` is `TB.api.ToolBoxInterfaces.remote` and contains a `render` command, it emits a `ws:renderCommand` event (for potential WebSocket/SSE driven rendering updates).
        *   Emits an `htmx:jsonResponse` event with the processed data.
    *   If the response is HTML, HTMX handles the swap, and `htmx:afterSwap` will subsequently fire.

**Usage:** This module works mostly in the background. Ensure your HTMX-powered components return appropriate HTML fragments or structured JSON that `tbjs` can understand.

**Example: HTMX with JSON Response**
Backend (Python/Flask example) returning JSON that might trigger a toast or update state:
```python
@app.route('/htmx/submit-form', methods=['POST'])
def submit_form_htmx():
    # ... process form ...
    if success:
        return jsonify({
            "error": "none",
            "result": {
                "data_to": "API", # Or "CLIENT"
                "data_info": "Form submitted successfully.",
                "data": {"new_id": 123, "message": "Item created."}
            },
            "info": {"exec_code": 0, "help_text": "Success"}
        })
    else:
        return jsonify({
            "error": "InputError",
            "result": {},
            "info": {"exec_code": 1, "help_text": "Invalid data provided."}
        }), 400
```
JavaScript to listen for custom JSON processing:
```javascript
TB.events.on('htmx:jsonResponse', ({ detail, data: tbResult }) => {
    console.log('HTMX JSON Response:', tbResult.get());
    if (tbResult.get() && tbResult.get().message) {
        TB.ui.Toast.showInfo(tbResult.get().message);
    }
});
```

### Dynamic Content Processing
`TB.ui.processDynamicContent(parentElement, options = {})`

This function is crucial when new HTML is added to the DOM, for example, by `TB.router` or HTMX after an `hx-swap`.

*   `parentElement`: The DOM element (or a wrapper around) the newly added content.
*   `options`:
    *   `addScripts` (boolean, default `true`): Whether to process `<script>` tags.
    *   `scriptCache` (Set, default `new Set()`): A Set to keep track of loaded script URLs to prevent re-execution. `TB.router` passes its cache.

**Actions Performed:**
1.  Calls `window.htmx.process(parentElement)` to initialize HTMX attributes on the new content.
2.  Handles `<script>` tags (see `TB.router` script handling for details).
3.  Initializes `tbjs` UI components found within `parentElement` (e.g., by looking for specific data attributes or classes).
4.  Calls `TB.ui.MarkdownRenderer.renderAllIn(parentElement)` if applicable.

**Usage:**
Generally, you don't need to call this manually if content is loaded via `TB.router` or standard HTMX swaps, as `TB.router` and `TB.ui.htmxIntegration` handle it. However, if you manually inject HTML that needs `tbjs` processing:
```javascript
const newContentContainer = document.getElementById('dynamic-area');
newContentContainer.innerHTML = '... new HTML with tbjs components, markdown, or scripts ...';
TB.ui.processDynamicContent(newContentContainer);
```

### Components

`tbjs` provides a set of pre-built UI components. They generally use Tailwind CSS classes for styling (often prefixed with `tb-` as per `tailwind.config.js`) and can be initialized programmatically or sometimes via data attributes.

#### Modal (`TB.ui.Modal`)
Displays content in a modal dialog.

**Options:**
*   `content`: HTML string or HTMLElement for the modal body.
*   `title`: (string) Optional modal title.
*   `closeOnOutsideClick`: (boolean, default `true`)
*   `closeOnEsc`: (boolean, default `true`)
*   `buttons`: Array of button config objects: `{ text, action, variant, className, size }`.
*   `onOpen`, `onClose`, `beforeClose`: Callbacks. `beforeClose` can return `false` to prevent closing.
*   `maxWidth`: (string, Tailwind class, default `'max-w-lg'`)
*   `modalId`: Custom ID for the modal element.
*   `customClasses`: Object to override default Tailwind classes for `overlay`, `modalContainer`, etc.

**Methods:**
*   `modal.show()`: Displays the modal.
*   `modal.close()`: Closes the modal.

**Static Method:**
*   `TB.ui.Modal.show(options)`: Creates and shows a modal.

**Example:**
```javascript
TB.ui.Modal.show({
    title: 'Confirm Action',
    content: '<p>Are you sure you want to delete this item?</p>',
    maxWidth: 'max-w-md', // Tailwind class
    buttons: [
        {
            text: 'Cancel',
            variant: 'secondary', // Uses TB.ui.Button styling
            action: (modal) => modal.close()
        },
        {
            text: 'Delete',
            variant: 'danger',
            action: (modal) => {
                console.log('Item deleted!');
                modal.close();
                TB.ui.Toast.showSuccess('Item deleted successfully.');
            }
        }
    ],
    onOpen: () => console.log('Confirmation modal opened.'),
    onClose: () => console.log('Confirmation modal closed.')
});
```

#### Toast (`TB.ui.Toast`)
Displays small, non-blocking notification messages.

**Options:**
*   `message`: (string) The message content.
*   `type`: `'info'`, `'success'`, `'warning'`, `'error'` (default `'info'`).
*   `duration`: (number) Milliseconds to display (0 for sticky, default `5000`).
*   `position`: `'top-right'`, `'top-center'`, etc. (default `'top-right'`).
*   `title`: (string) Optional title.
*   `actions`: Array of action objects: `[{ text: 'Undo', action: () => { /* ... */ } }]`.
*   `icon`: (boolean, default `true`) Show type-specific icon.
*   `closable`: (boolean, default `true`) Show a close button.
*   `customClasses`: For overriding default styling of toast elements.

**Static Methods:**
*   `TB.ui.Toast.showInfo(message, options)`
*   `TB.ui.Toast.showSuccess(message, options)`
*   `TB.ui.Toast.showWarning(message, options)`
*   `TB.ui.Toast.showError(message, options)`
*   `TB.ui.Toast.hideAll()`

**Example:**
```javascript
TB.ui.Toast.showSuccess('Profile updated successfully!', {
    duration: 3000,
    position: 'bottom-center'
});

TB.ui.Toast.showError('Failed to connect to server.', {
    title: 'Network Error',
    duration: 0, // Sticky
    actions: [
        { text: 'Retry', action: () => console.log('Retry clicked!') }
    ]
});
```

#### Loader (`TB.ui.Loader`)
Displays a loading indicator.

**Options:**
*   `text`: (string, default `'Loading...'`) Text displayed below the spinner.
*   `fullscreen`: (boolean, default `true`) If true, covers the whole page.
*   `customSpinnerHtml`: (string) Custom HTML for the spinner.
*   `customClasses`: For overriding default styling of overlay, spinner container, text.

**Static Methods:**
*   `TB.ui.Loader.show(textOrOptions)`: Shows a fullscreen loader. Returns the loader DOM element.
*   `TB.ui.Loader.hide(loaderElement)`: Hides a specific loader element or the default fullscreen loader.

**Example:**
```javascript
// Show default fullscreen loader
const myLoader = TB.ui.Loader.show('Processing your request...');

// Simulate an async operation
setTimeout(() => {
    TB.ui.Loader.hide(myLoader); // Hide the specific loader instance
    // Or TB.ui.Loader.hide(); to hide the default ID loader
}, 2000);
```
*Note: The Loader component injects its own minimal CSS for the spinner animation (`tbjs_spin`) and basic layout if not overridden by Tailwind classes.*

#### Button (`TB.ui.Button`)
A class to create styled button elements programmatically.

**Options:**
*   `text`: (string, default `'Button'`)
*   `action`: (function) Click handler `(event, buttonInstance) => {}`.
*   `variant`: `'primary'`, `'secondary'`, `'danger'`, `'outline'`, `'ghost'`, `'link'` (default `'primary'`).
*   `size`: `'sm'`, `'md'`, `'lg'` (default `'md'`).
*   `type`: `'button'`, `'submit'`, `'reset'` (default `'button'`).
*   `disabled`: (boolean, default `false`)
*   `isLoading`: (boolean, default `false`)
*   `iconLeft`, `iconRight`: HTML string for icons (e.g., Material Symbol span).
*   `customClasses`: Additional CSS classes.
*   `attributes`: Object of custom attributes.

**Instance Methods:**
*   `setText(text)`
*   `setLoading(isLoading, updateDom = true)`
*   `setDisabled(isDisabled, updateDom = true)`
*   `element`: The DOM element of the button.

**Static Method:**
*   `TB.ui.Button.create(text, action, options)`: Creates a button and returns its DOM element.

**Example:**
```javascript
const submitBtnElement = TB.ui.Button.create('Submit Form', async (event, btnInstance) => {
    btnInstance.setLoading(true);
    try {
        // await someApiCall();
        TB.ui.Toast.showSuccess('Form submitted!');
    } catch (e) {
        TB.ui.Toast.showError('Submission failed.');
    } finally {
        btnInstance.setLoading(false);
    }
}, {
    variant: 'primary',
    size: 'lg',
    iconLeft: '<span class="material-symbols-outlined">save</span>',
    attributes: { 'data-form-id': 'user-reg-form' }
});

document.getElementById('form-actions').appendChild(submitBtnElement);
```

#### DarkModeToggle (`TB.ui.DarkModeToggle`)
Manages a dark mode toggle button/UI element, syncing with `TB.ui.theme`.

**Options (passed to constructor or `TB.ui.DarkModeToggle.init()`):**
*   `containerSelector`: (string, default `'#darkModeToggleContainer'`) The main clickable element or wrapper.
*   `iconSelector`: (string, default `.tb-toggle-icon'`) Selector for the icon element (e.g., a `<span>` for Material Symbols).
*   `checkboxSelector`: (string, default `'#darkModeSwitch'`) Selector for an optional underlying `<input type="checkbox">`.
*   `lightModeIconClass`, `darkModeIconClass`: Text content for the icon (e.g., Material Symbol names like `'light_mode'`, `'dark_mode'`).
*   `rotationActiveDeg`, `rotationInactiveDeg`, `rotationTransition`: For icon rotation animation.

**Initialization:**
*   `new TB.ui.DarkModeToggle(options)`
*   `TB.ui.DarkModeToggle.init(optionsOrSelector)`: Static convenience method.

**HTML Structure Examples:**

1.  **Icon-only Toggle (recommended):**
    ```html
    <button id="darkModeToggleContainer" aria-label="Toggle dark mode" class="tb-p-2 tb-rounded-full hover:tb-bg-gray-200 dark:hover:tb-bg-gray-700">
        <span class="material-symbols-outlined tb-toggle-icon"></span>
    </button>
    ```
    Initialize with: `TB.ui.DarkModeToggle.init();` (uses default selectors) or `TB.ui.DarkModeToggle.init('#darkModeToggleContainer');`

2.  **Checkbox-driven Toggle:**
    ```html
    <label for="darkModeSwitch" id="darkModeToggleContainer" class="tb-inline-flex tb-items-center tb-cursor-pointer">
        <input type="checkbox" id="darkModeSwitch" class="tb-sr-only tb-peer">
        <div class="tb-relative tb-w-11 tb-h-6 tb-bg-gray-200 peer-focus:tb-outline-none peer-focus:tb-ring-4 peer-focus:tb-ring-blue-300 dark:peer-focus:tb-ring-blue-800 tb-rounded-full peer dark:tb-bg-gray-700 peer-checked:tb-after:tb-translate-x-full rtl:peer-checked:tb-after:-tb-translate-x-full peer-checked:tb-after:tb-border-white tb-after:tb-content-[''] tb-after:tb-absolute tb-after:tb-top-[2px] tb-after:tb-start-[2px] tb-after:tb-bg-white tb-after:tb-border-gray-300 tb-after:tb-border tb-after:tb-rounded-full tb-after:tb-h-5 tb-after:tb-w-5 tb-after:tb-transition-all dark:tb-border-gray-600 peer-checked:tb-bg-blue-600"></div>
        <span class="material-symbols-outlined tb-toggle-icon tb-ml-3 tb-text-gray-900 dark:tb-text-gray-300"></span>
    </label>
    ```
    Initialize with: `TB.ui.DarkModeToggle.init({ containerSelector: '#darkModeToggleContainer', iconSelector: '.tb-toggle-icon', checkboxSelector: '#darkModeSwitch' });`

**Functionality:**
*   Updates its visual state (icon, checkbox checked status) based on `TB.ui.theme.getCurrentMode()`.
*   Listens for clicks (on container or changes on checkbox) to call `TB.ui.theme.setPreference()` or `TB.ui.theme.togglePreference()`.
*   Reacts to `theme:changed` events to keep its visual state synchronized.

#### CookieBanner (`TB.ui.CookieBanner`)
Displays a cookie consent banner and optional settings modal.

**Options:**
*   `title`, `message`, `termsLink`, `termsLinkText`, `acceptMinimalText`, `showAdvancedOptions`, `advancedOptionsText`: Text and behavior customization.
*   `onConsent`: Callback `(consentSettings) => {}` triggered when consent is given/updated.
*   `customClasses`: For styling banner, modal, etc.

**Static Method:**
*   `TB.ui.CookieBanner.show(options)`: Creates and displays the banner if no consent is found in `localStorage` (`tbjs_cookie_consent`).
*   `TB.ui.CookieBanner.getConsent()`: Retrieves current consent status from `localStorage`.

**Functionality:**
*   Shows a banner at the bottom of the page.
*   Allows accepting recommended settings or opening a modal for granular control (Essential, Preferences, Analytics).
*   Saves consent to `localStorage`.
*   Emits `cookieConsent:updated` event with consent settings: `{ essential, preferences, analytics, source }`.

**Example:**
```javascript
// This can be called early in your application setup
TB.ui.CookieBanner.show({
    title: 'Our Cookie Policy',
    message: 'We use cookies to improve your experience. By clicking "Accept", you agree to our use of cookies.',
    termsLink: '/privacy-policy',
    onConsent: (settings) => {
        console.log('Cookie consent given:', settings);
        if (settings.analytics) {
            // Initialize analytics tools
        }
    }
});

// Check consent later
const consent = TB.ui.CookieBanner.getConsent();
if (consent && consent.analytics) {
    // ...
}
```

#### MarkdownRenderer (`TB.ui.MarkdownRenderer`)
Renders Markdown content to HTML, with optional syntax highlighting using `highlight.js`.

**Prerequisites:**
*   `marked.js` (e.g., `window.marked`) must be globally available.
*   `highlight.js` (e.g., `window.hljs`) must be globally available for syntax highlighting.
*   `marked-highlight` (e.g., `window.markedHighlight`) if using the extension for `hljs`.

**Methods:**
*   `TB.ui.MarkdownRenderer.init()`: Initializes `marked` with `highlight.js`. Called automatically on first render if needed, or can be called explicitly.
*   `TB.ui.MarkdownRenderer.render(markdownString)`: Converts a Markdown string to HTML.
*   `TB.ui.MarkdownRenderer.renderElement(element)`: Renders the content of a DOM element (if it has class `.markdown` and not already rendered). Adds Tailwind Prose classes.
*   `TB.ui.MarkdownRenderer.renderAllIn(parentElement)`: Finds all elements with class `.markdown` within `parentElement` (that haven't been rendered yet) and renders them.

**Usage:**
Typically used by `TB.ui.processDynamicContent` when new HTML containing elements with the class `markdown` is added to the DOM.

**Example:**
HTML:
```html
<div class="markdown">
# My Markdown Title

This is some **bold** text and a [link](https://example.com).

```javascript
console.log('This is JavaScript code');
```
</div>
```
JavaScript (rendering is often automatic after content swap):
```javascript
// If you add markdown content dynamically outside of router/HTMX swaps:
const myDiv = document.createElement('div');
myDiv.className = 'markdown';
myDiv.textContent = '## Subtitle\n* Item 1\n* Item 2';
document.body.appendChild(myDiv);
TB.ui.MarkdownRenderer.renderElement(myDiv); // Or TB.ui.MarkdownRenderer.renderAllIn(document.body);
```
The rendered output will be styled with Tailwind Prose classes (`prose dark:prose-invert max-w-none`).

#### AutocompleteWidget (`TB.ui.AutocompleteWidget`)
Provides autocomplete functionality for input fields.

**Options:**
*   `source`: Array of strings, or a function `(inputValue) => Promise<string[]>` or `(inputValue) => string[]`.
*   `minLength`: (number, default `1`) Minimum characters to type before suggestions appear.
*   `onSelect`: Callback `(value, inputElement) => {}` when an item is selected.
*   `customClasses`: Object to customize Tailwind classes for `list`, `item`, `activeItem`, `highlight`.

**Initialization:**
*   `new TB.ui.AutocompleteWidget(inputElement, options)`
*   `TB.ui.AutocompleteWidget.initAll(selector = 'input[data-tb-autocomplete]')`: Initializes for all matching elements.
    *   Uses `data-tb-autocomplete-source` attribute (JSON array or global function name) if present.

**Example:**
HTML:
```html
<div class="tb-relative"> <!-- Autocomplete list will be positioned relative to this -->
    <input type="text" id="myAutocompleteInput" class="tb-border tb-p-2 tb-w-full" placeholder="Search...">
</div>
<div class="tb-relative">
    <input type="text" data-tb-autocomplete data-tb-autocomplete-source='["Apple", "Banana", "Cherry"]' class="tb-border tb-p-2 tb-w-full" placeholder="Fruit Search...">
</div>
```
JavaScript:
```javascript
const acInput = document.getElementById('myAutocompleteInput');
const mySourceFunction = async (term) => {
    // In a real app, fetch from an API
    const items = ['JavaScript', 'Java', 'Python', 'PHP', 'Perl'];
    return items.filter(item => item.toLowerCase().includes(term.toLowerCase()));
};

new TB.ui.AutocompleteWidget(acInput, {
    source: mySourceFunction,
    minLength: 2,
    onSelect: (value, el) => {
        console.log(`Selected: ${value} from input:`, el);
    }
});

// Initialize declarative autocomplete inputs
TB.ui.AutocompleteWidget.initAll();
```

#### NavMenu (`TB.ui.NavMenu`)
Manages a responsive navigation menu, typically a slide-in or modal menu for mobile.

**Options:**
*   `triggerSelector`: (string, default `'#links'`) Selector for the menu toggle button.
*   `menuContentHtml`: (string) HTML content for the menu.
*   `menuId`: (string, default `'tb-nav-menu-modal'`) ID for the menu container.
*   `openIconClass`, `closeIconClass`: Material Symbols class names for the toggle icon.
*   `customClasses`: For styling `overlay`, `menuContainer`, `iconContainer`.

**HTML Structure Expectation (for default trigger):**
The trigger element (e.g., `#links`) should ideally contain a `<span>` (often with `class="material-symbols-outlined"`) for the icon. If empty, the component will append one. The menu itself is appended to an element with `id="Nav-Main"`.

**Example HTML:**
```html
<nav id="Nav-Main" class="tb-bg-gray-800 tb-text-white tb-p-4 tb-flex tb-justify-between tb-items-center">
    <a href="/" class="tb-text-xl tb-font-bold">MyApp</a>
    <button id="menuTrigger" class="tb-p-2 md:tb-hidden"> <!-- md:hidden to hide on larger screens -->
        <span class="material-symbols-outlined">menu</span>
    </button>
    <ul class="hidden md:tb-flex tb-space-x-4"> <!-- Desktop links -->
        <li><a href="/page1">Page 1</a></li>
        <li><a href="/page2">Page 2</a></li>
    </ul>
</nav>
```
**Example JavaScript:**
```javascript
// Initialize the NavMenu
const navMenu = TB.ui.NavMenu.init({
    triggerSelector: '#menuTrigger', // Custom trigger
    menuContentHtml: `
        <ul class="tb-space-y-2 tb-p-4">
            <li><a href="/home" class="tb-block tb-p-2 hover:tb-bg-gray-700 tb-rounded">Home</a></li>
            <li><a href="/about" class="tb-block tb-p-2 hover:tb-bg-gray-700 tb-rounded">About</a></li>
            <li><a href="/contact" class="tb-block tb-p-2 hover:tb-bg-gray-700 tb-rounded">Contact</a></li>
        </ul>
    `,
    // customize classes if needed, e.g., for a different background:
    // customClasses: {
    //   menuContainer: 'fixed top-0 left-0 h-full w-64 sm:w-72 bg-neutral-800 shadow-xl z-[1041] transform -translate-x-full transition-transform duration-300 ease-in-out text-white',
    // }
});

// The menu links will automatically close the menu upon navigation if handled by TB.router.
```

---

## 5. Usage Examples

### Basic Application Setup
This example shows a minimal setup to get `tbjs` running with a simple home page.

**`index.html`:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>tbjs App</title>
    <link rel="stylesheet" href="path/to/tbjs/dist/tbjs.css">
    <link rel="stylesheet" href="your-app-styles.css"> <!-- Your app specific styles -->
    <script src="https://unpkg.com/htmx.org@2.0.0/dist/htmx.min.js"></script>
    <script src="path/to/tbjs/dist/tbjs.js"></script>
</head>
<body>
    <header>
        <nav>
            <a href="/home">Home</a>
            <a href="/about">About</a>
        </nav>
    </header>
    <main id="app-root"></main>

    <script type="module">
        document.addEventListener('DOMContentLoaded', () => {
            TB.init({
                appRootId: 'app-root',
                baseApiUrl: '/api',
                logLevel: 'debug',
                // Default initial navigation will be to current path or /index.html handled by router
            });
        });
    </script>
</body>
</html>
```

**`/web/pages/home.html`** (assuming `baseFileUrl` allows fetching this via `/home` or similar)
```html
<h1>Welcome to the Home Page!</h1>
<p>This content was loaded by tbjs router.</p>
<button hx-get="/api/time" hx-target="#time-div">Get Server Time</button>
<div id="time-div"></div>
```
*   `TB.init()` sets up the router, which loads the initial page content (e.g., based on URL, or defaults to `/index.html` which might then redirect or show base content).
*   Router intercepts clicks on `<a>` tags.
*   HTMX integration is active.

### Fetching Data and Updating State
```javascript
// products.js
async function loadAndDisplayProducts() {
    TB.ui.Loader.show('Loading products...');
    const result = await TB.api.request('products', 'list', { limit: 5 }, 'GET');
    TB.ui.Loader.hide();

    if (result.error === TB.api.ToolBoxError.none) {
        const products = result.get();
        TB.state.set('shop.products', products);
        renderProductList(products);
    } else {
        TB.ui.Toast.showError('Failed to load products: ' + result.info.help_text);
    }
}

function renderProductList(products) {
    const container = document.getElementById('product-list-container');
    if (!container) return;
    container.innerHTML = `
        <ul class="tb-list-disc tb-pl-5">
            ${products.map(p => `<li class="tb-mb-2">${p.name} - $${p.price}</li>`).join('')}
        </ul>
    `;
}

// Listen for state changes to re-render if needed elsewhere
TB.events.on('state:changed:shop.products', (newProducts) => {
    // Potentially update other parts of the UI, or just log
    console.log('Product list updated in state:', newProducts);
});

// Initial load
// loadAndDisplayProducts(); // Call this when the relevant view is loaded
```

### Client-Side Routing
The router automatically handles link clicks and browser navigation.

**`/web/pages/about.html`:**
```html
<h2>About Us</h2>
<p>This is the about page, loaded dynamically by the router.</p>
<script>
    // This script will be executed when about.html is loaded
    console.log('About page script executed!');
    TB.logger.info('[AboutPage] Loaded and script ran.');
</script>
```
*   When a user clicks `<a href="/about">About</a>`, `TB.router` fetches `/web/pages/about.html` (assuming `baseFileUrl` + `/about` maps to it) and injects its content into `#app-root`.
*   The inline script in `about.html` is executed.

### Displaying a Modal
```javascript
document.getElementById('show-info-modal').addEventListener('click', () => {
    TB.ui.Modal.show({
        title: 'Important Information',
        content: `
            <p>This is some important information presented in a modal.</p>
            <p>Current time from state: ${TB.state.get('app.currentTime') || 'Not set'}</p>
        `,
        buttons: [
            { text: 'OK', action: (modal) => modal.close(), variant: 'primary' }
        ]
    });
});
```

### User Authentication Flow
A simplified example of a device key login.

**HTML for Login:**
```html
<!-- login.html -->
<h2>Login with Device Key</h2>
<input type="text" id="username" placeholder="Username" class="tb-border tb-p-2">
<button id="loginButton" class="tb-bg-blue-500 tb-text-white tb-p-2 tb-rounded">Login</button>
<div id="login-status"></div>
```

**JavaScript for login.html (or global script managing this view):**
```javascript
// Assuming this script runs when login.html is loaded
document.addEventListener('DOMContentLoaded', () => { // Or use TB.events router:contentProcessed
    const loginButton = document.getElementById('loginButton');
    const usernameInput = document.getElementById('username');
    const statusDiv = document.getElementById('login-status');

    if (loginButton) {
        loginButton.addEventListener('click', async () => {
            const username = usernameInput.value.trim();
            if (!username) {
                statusDiv.textContent = 'Please enter a username.';
                return;
            }

            statusDiv.textContent = 'Attempting login...';
            TB.ui.Loader.show('Logging in...');

            try {
                const result = await TB.user.loginWithDeviceKey(username);
                TB.ui.Loader.hide();

                if (result.success) {
                    statusDiv.textContent = `Login successful! Welcome ${TB.user.getUsername()}.`;
                    TB.ui.Toast.showSuccess('Login successful!');
                    TB.router.navigateTo('/dashboard'); // Navigate to a protected area
                } else {
                    statusDiv.textContent = `Login failed: ${result.message}`;
                    TB.ui.Toast.showError(result.message);
                }
            } catch (error) {
                TB.ui.Loader.hide();
                statusDiv.textContent = `Login error: ${error.message}`;
                TB.logger.error('Login process error:', error);
            }
        });
    }
});
```

---

## 6. Styling with Tailwind CSS
`tbjs` components are designed to be styled with Tailwind CSS. The framework includes a `tailwind.config.js` and `postcss.config.js`.

**Key Points:**
*   **Prefix:** The provided `tailwind.config.js` uses `prefix: 'tb-'`. This means all Tailwind utility classes used internally by `tbjs` components will be prefixed (e.g., `tb-bg-blue-500`, `tb-text-lg`). This helps avoid conflicts with your application's own Tailwind classes if it doesn't use a prefix or uses a different one.
*   **CSS Variables:** The configuration defines CSS variables for theming (e.g., `--tb-color-primary-500`, `--tb-color-background`). These are used in the `theme.extend.colors` section of `tailwind.config.js` and in `tbjs-main.css`. This allows themes (light/dark) to be easily applied and customized.
*   **`tbjs-main.css`:** This file imports Tailwind utilities and defines the base CSS variables and some default styles for components like the Loader.
*   **Customization:** You can customize the `tbjs` Tailwind configuration (`tbjs/tailwind.config.js`) or integrate its plugin settings into your main application's Tailwind config.

**Using `tbjs` Tailwind Config in Your Project:**
If your project also uses Tailwind CSS, you can either:
1.  **Run two PostCSS processes**: One for `tbjs` (using its config) and one for your app (using your app's config).
2.  **Merge configurations**: If your app's Tailwind setup can consume the `tbjs` Tailwind config (e.g., as a preset or by merging `content` paths and `theme` extensions).

    Example of merging in your app's `tailwind.config.js`:
    ```javascript
    // your-app/tailwind.config.js
    import tbjsTailwindConfig from 'path/to/tbjs/tailwind.config.js';

    export default {
      content: [
        './src/**/*.{html,js}', // Your app's content
        './node_modules/tbjs/src/**/*.{html,js}', // Include tbjs components for scanning
      ],
      darkMode: 'class', // Ensure consistency
      prefix: '', // Or your app's prefix
      theme: {
        extend: {
          // Merge tbjs theme extensions if needed, being mindful of prefix differences
          colors: {
            ...tbjsTailwindConfig.theme.extend.colors, // May need adjustment if your app doesn't use 'tb-' prefix
            // Your app-specific colors
          },
          // ... other extensions
        },
      },
      plugins: [
        // Your app's plugins
      ],
    };
    ```
    *The key is to ensure Tailwind processes the classes used in `tbjs` components.* Using the `tb-` prefix in `tbjs` helps isolate its styles.

---

## 7. Building `tbjs` (For Developers)
If you are modifying the `tbjs` framework itself:

*   **Dependencies**: Install development dependencies: `npm install`
*   **Build**: `npm run build` (creates production build in `dist/`)
*   **Watch**: `npm run watch` (watches for changes and rebuilds in production mode)
*   **Lint**: `npm run lint` (checks JavaScript code style)

The build process uses Webpack, configured in `webpack.config.js`. It bundles the JavaScript into `dist/tbjs.js` (UMD format) and extracts CSS into `dist/tbjs.css`.

---
