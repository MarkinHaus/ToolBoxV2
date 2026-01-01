// tbjs/src/core/config.js
import logger from './logger.js'; // Ensure logger is imported if used here

let _config = {};


const Config = {
    init: (initialUserConfig = {}) => {
        const defaultConfig = {
            appRootId: 'app-root',
            baseApiUrl: '/api',
            baseFileUrl: window.location.origin,
            initialState: {},
            themeSettings: {
                defaultPreference: 'system',
                background: {
                    type: 'color',
                    light: { color: '#FFFFFF', image: '' },
                    dark: { color: '#121212', image: '' },
                    placeholder: { image: '', displayUntil3DReady: true }
                }
            },
            routes: [],
            logLevel: 'info',
            // isProduction: process.env.NODE_ENV === 'production', // Better set by user config
            serviceWorker: {
                enabled: false,
                url: '/sw.js',
                scope: '/'
            }
        };

        // Simple deep merge for specific known nested objects
        _config = {
            ...defaultConfig,
            ...initialUserConfig,
            themeSettings: {
                ...(defaultConfig.themeSettings || {}),
                ...(initialUserConfig.themeSettings || {}),
                background: {
                    ...(defaultConfig.themeSettings?.background || {}),
                    ...(initialUserConfig.themeSettings?.background || {}),
                    light: {
                        ...(defaultConfig.themeSettings?.background?.light || {}),
                        ...(initialUserConfig.themeSettings?.background?.light || {}),
                    },
                    dark: {
                        ...(defaultConfig.themeSettings?.background?.dark || {}),
                        ...(initialUserConfig?.themeSettings?.background?.dark || {}),
                    },
                    placeholder: {
                        ...(defaultConfig.themeSettings?.background?.placeholder || {}),
                        ...(initialUserConfig.themeSettings?.background?.placeholder || {}),
                    }
                }
            },
            serviceWorker: {
                ...(defaultConfig.serviceWorker || {}),
                ...(initialUserConfig.serviceWorker || {})
            }
        };

        if (typeof _config.isProduction === 'undefined') {
            // Infer if not explicitly set, e.g. based on hostname or a global var
             _config.isProduction = !(window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
        }

        // Detect Tauri environment and set API URLs
        const isTauri = window.location.hostname === 'tauri.localhost' ||
                        window.location.protocol === 'tauri:' ||
                        (typeof window.__TAURI__ !== 'undefined');

        // Remote API fallback (for mobile or when worker unavailable)
        const remoteApiUrl = initialUserConfig.remoteApiUrl || 'https://simplecore.app/api';
        const remoteWsUrl = initialUserConfig.remoteWsUrl || 'wss://simplecore.app';

        if (isTauri && !initialUserConfig.baseApiUrl) {
            const workerHttpPort = initialUserConfig.workerHttpPort || 5000;
            const workerWsPort = initialUserConfig.workerWsPort || 5001;
            const localApiUrl = `http://localhost:${workerHttpPort}/api`;
            const localWsUrl = `ws://localhost:${workerWsPort}`;

            // Detect platform from user agent
            const ua = navigator.userAgent.toLowerCase();
            const isIOS = /iphone|ipad|ipod/.test(ua) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
            const isAndroid = /android/.test(ua);
            const isMobile = isIOS || isAndroid;

            _config.isTauri = true;
            _config.isMobile = isMobile;
            _config.isIOS = isIOS;
            _config.isAndroid = isAndroid;

            if (isIOS) {
                // iOS: Always use remote API (no local worker possible)
                _config.baseApiUrl = remoteApiUrl;
                _config.baseWsUrl = remoteWsUrl;
                _config.useRemoteApi = true;
                console.log('[Config] Tauri iOS detected, using remote API:', remoteApiUrl);
            } else if (isAndroid) {
                // Android: User can choose between local, remote, or custom endpoint
                // Store available endpoints for switching
                _config._localApiUrl = localApiUrl;
                _config._localWsUrl = localWsUrl;
                _config._remoteApiUrl = remoteApiUrl;
                _config._remoteWsUrl = remoteWsUrl;
                _config.canSwitchEndpoint = true; // Flag for UI to show settings

                // Load user preference from localStorage
                const savedEndpoint = localStorage.getItem('tb_api_endpoint_mode') || 'local';
                const customApiUrl = localStorage.getItem('tb_custom_api_url');
                const customWsUrl = localStorage.getItem('tb_custom_ws_url');

                if (savedEndpoint === 'remote') {
                    _config.baseApiUrl = remoteApiUrl;
                    _config.baseWsUrl = remoteWsUrl;
                    _config.useRemoteApi = true;
                    _config.endpointMode = 'remote';
                    console.log('[Config] Tauri Android: Using saved remote API:', remoteApiUrl);
                } else if (savedEndpoint === 'custom' && customApiUrl) {
                    _config.baseApiUrl = customApiUrl;
                    _config.baseWsUrl = customWsUrl || remoteWsUrl;
                    _config.useRemoteApi = true;
                    _config.endpointMode = 'custom';
                    console.log('[Config] Tauri Android: Using custom API:', customApiUrl);
                } else {
                    // Default to local, with fallback check
                    _config.baseApiUrl = localApiUrl;
                    _config.baseWsUrl = localWsUrl;
                    _config.useRemoteApi = false;
                    _config.endpointMode = 'local';
                    // Async worker check - will update config if worker unavailable
                    Config._checkWorkerAndFallback(localApiUrl, remoteApiUrl, remoteWsUrl);
                    console.log('[Config] Tauri Android: Using local API with fallback check');
                }
            } else {
                // Desktop: Use local worker
                _config.baseApiUrl = localApiUrl;
                _config.baseWsUrl = localWsUrl;
                _config.useRemoteApi = false;
                console.log('[Config] Tauri Desktop detected, using local worker:', localApiUrl);
            }
        } else {
            _config.isTauri = isTauri;
        }

        // Ensure baseFileUrl ends with a slash if it's not just the origin and contains a path
        if (_config.baseFileUrl && new URL(_config.baseFileUrl).pathname !== '/' && !_config.baseFileUrl.endsWith('/')) {
            _config.baseFileUrl += '/';
        }

        // Only convert relative URLs to absolute - skip if already absolute (http/https)
        // This preserves the Tauri-specific URLs set above
        if (_config.baseApiUrl && !_config.baseApiUrl.startsWith('http')) {
            if (!_config.baseApiUrl.startsWith('/')) {
                _config.baseApiUrl = '/' + _config.baseApiUrl;
            }
            // Only resolve relative URLs for non-Tauri environments
            // In Tauri, we've already set absolute URLs above
            if (!_config.isTauri) {
                _config.baseApiUrl = new URL(_config.baseApiUrl, window.location.origin).href;
            } else {
                // For Tauri with relative URL (shouldn't happen, but fallback to local worker)
                const workerHttpPort = initialUserConfig.workerHttpPort || 5000;
                _config.baseApiUrl = `http://localhost:${workerHttpPort}${_config.baseApiUrl}`;
            }
        }

        // Override global fetch in Tauri to redirect API/Auth requests to the worker backend
        // Static assets (HTML, CSS, JS, images) are served by Tauri from frontendDist
        // Also injects Authorization headers for authenticated requests
        if (_config.isTauri && typeof window !== 'undefined') {
            const originalFetch = window.fetch;

            // Auth endpoints that need to be redirected to the worker
            const AUTH_ENDPOINTS = ['/validateSession', '/IsValidSession', '/web/logoutS', '/api_user_data'];

            // Static file extensions that should NOT be redirected (served by Tauri)
            const STATIC_EXTENSIONS = [
                '.html', '.htm', '.css', '.js', '.mjs', '.json', '.map',
                '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.webp',
                '.woff', '.woff2', '.ttf', '.eot', '.otf',
                '.mp3', '.mp4', '.webm', '.ogg', '.wav',
                '.pdf', '.txt', '.xml'
            ];

            // Get base URL without /api suffix
            const getBaseUrl = () => _config.baseApiUrl.replace(/\/api\/?$/, '');

            // Check if path is a static asset
            const isStaticAsset = (path) => {
                const lowerPath = path.toLowerCase().split('?')[0]; // Remove query string
                return STATIC_EXTENSIONS.some(ext => lowerPath.endsWith(ext));
            };

            // Get fresh auth token from Clerk or stored state
            const getAuthToken = async () => {
                try {
                    // Try to get fresh token from Clerk via TB.user
                    if (window.TB?.user?.getSessionToken) {
                        const token = await window.TB.user.getSessionToken();
                        if (token) return token;
                    }
                    // Fallback to stored token
                    if (window.TB?.state?.get) {
                        return window.TB.state.get('user.token');
                    }
                } catch (e) {
                    console.debug('[Config] Could not get auth token:', e);
                }
                return null;
            };

            // Merge auth headers into init options
            const mergeAuthHeaders = async (init, needsAuth) => {
                if (!needsAuth) return init;

                const token = await getAuthToken();
                if (!token) return init;

                const newInit = { ...init };
                newInit.headers = new Headers(init?.headers || {});

                // ALWAYS set fresh Authorization token (override any stale token)
                // This ensures we always use the latest token from Clerk
                newInit.headers.set('Authorization', `Bearer ${token}`);

                // Ensure credentials are included
                if (!newInit.credentials) {
                    newInit.credentials = 'include';
                }

                return newInit;
            };

            window.fetch = async function(input, init) {
                let url = input;

                // Handle Request objects
                if (input instanceof Request) {
                    url = input.url;
                }

                // Convert to string for checking
                const urlStr = typeof url === 'string' ? url : url.toString();

                // Extract path from URL (handles both relative and absolute URLs)
                let path = urlStr;
                if (urlStr.includes('tauri.localhost')) {
                    path = urlStr.replace(/^https?:\/\/tauri\.localhost/, '');
                } else if (urlStr.startsWith('http://') || urlStr.startsWith('https://')) {
                    try {
                        path = new URL(urlStr).pathname;
                    } catch (e) {
                        // Keep original path if URL parsing fails
                    }
                }

                // NEVER redirect static assets - let Tauri serve them
                if (isStaticAsset(path)) {
                    return originalFetch.call(this, input, init);
                }

                // Check if it's an auth endpoint
                const isAuthEndpoint = AUTH_ENDPOINTS.some(ep => path === ep || path.startsWith(ep + '?'));
                if (isAuthEndpoint) {
                    const newUrl = getBaseUrl() + path;
                    const authInit = await mergeAuthHeaders(init, true);
                    console.log('[Config] Fetch redirect (auth):', urlStr, '->', newUrl);

                    if (input instanceof Request) {
                        return originalFetch.call(this, new Request(newUrl, input), authInit);
                    }
                    return originalFetch.call(this, newUrl, authInit);
                }

                // Check if it's an /api/ request
                if (path.startsWith('/api/') || path === '/api') {
                    const newUrl = getBaseUrl() + path;
                    const authInit = await mergeAuthHeaders(init, true);
                    console.log('[Config] Fetch redirect (api):', urlStr, '->', newUrl);

                    if (input instanceof Request) {
                        return originalFetch.call(this, new Request(newUrl, input), authInit);
                    }
                    return originalFetch.call(this, newUrl, authInit);
                }

                // Pass through all other requests (static assets, external URLs, etc.)
                return originalFetch.call(this, input, init);
            };
            console.log('[Config] Global fetch override installed for Tauri (API + Auth with auto-injection)');
        }

        // After logger is initialized in TB.init, this can be moved there or use a preliminary log.
        // For now, console.log is fine for config init.
        console.log('[Config] Initialized Config:', JSON.parse(JSON.stringify(_config))); // Use JSON stringify/parse for clean log
    },

    get: (key) => {
        if (!key) return undefined;

        if (key.includes('.')) {
            const keys = key.split('.');
            let current = _config;
            for (let i = 0; i < keys.length; i++) {
                if (current && typeof current === 'object' && keys[i] in current) {
                    current = current[keys[i]];
                } else {
                    // logger.warn(`[Config] Key "${key}" not found (path: ${keys.slice(0, i+1).join('.')})`);
                    // console.warn(`[Config] GET: Key "${key}" not found (path: ${keys.slice(0, i+1).join('.')}). Available config:`, _config);
                    return undefined;
                }
            }
            return current;
        }

        const value = _config[key];
        // if (value === undefined) {
            // logger.warn(`[Config] Key "${key}" not found.`);
            // console.warn(`[Config] GET: Key "${key}" not found. Available config:`, _config);
        // }
        return value;
    },

    getAll: () => {
        return { ..._config };
    },

    set: (key, value) => {
        if (key.includes('.')) {
            const keys = key.split('.');
            let current = _config;
            for (let i = 0; i < keys.length - 1; i++) {
                if (!current[keys[i]] || typeof current[keys[i]] !== 'object') {
                    current[keys[i]] = {};
                }
                current = current[keys[i]];
            }
            current[keys[keys.length - 1]] = value;
        } else {
            _config[key] = value;
        }
        logger.debug(`[Config] Set: ${key} =`, value);
    },

    /**
     * Check if local worker is available (Android), fallback to remote API if not.
     * @param {string} localApiUrl - Local worker API URL
     * @param {string} remoteApiUrl - Remote API URL fallback
     * @param {string} remoteWsUrl - Remote WebSocket URL fallback
     */
    _checkWorkerAndFallback: async (localApiUrl, remoteApiUrl, remoteWsUrl) => {
        try {
            // Try to reach the local worker with a health check
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 3000); // 3 second timeout

            const response = await fetch(`${localApiUrl.replace('/api', '')}/health`, {
                method: 'GET',
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            if (response.ok) {
                console.log('[Config] Local worker is available');
                _config.useRemoteApi = false;
                return;
            }
        } catch (error) {
            console.log('[Config] Local worker not available, falling back to remote API:', error.message);
        }

        // Fallback to remote API
        _config.baseApiUrl = remoteApiUrl;
        _config.baseWsUrl = remoteWsUrl;
        _config.useRemoteApi = true;
        _config.endpointMode = 'remote';
        console.log('[Config] Using remote API:', remoteApiUrl);
    },

    /**
     * Switch API endpoint (Android only).
     * @param {'local' | 'remote' | 'custom'} mode - The endpoint mode
     * @param {string} [customApiUrl] - Custom API URL (required if mode is 'custom')
     * @param {string} [customWsUrl] - Custom WebSocket URL (optional for custom mode)
     * @returns {boolean} - True if switch was successful
     */
    setApiEndpoint: (mode, customApiUrl = null, customWsUrl = null) => {
        if (!_config.canSwitchEndpoint) {
            console.warn('[Config] API endpoint switching is not available on this platform');
            return false;
        }

        const localApiUrl = _config._localApiUrl;
        const localWsUrl = _config._localWsUrl;
        const remoteApiUrl = _config._remoteApiUrl;
        const remoteWsUrl = _config._remoteWsUrl;

        switch (mode) {
            case 'local':
                _config.baseApiUrl = localApiUrl;
                _config.baseWsUrl = localWsUrl;
                _config.useRemoteApi = false;
                _config.endpointMode = 'local';
                localStorage.setItem('tb_api_endpoint_mode', 'local');
                localStorage.removeItem('tb_custom_api_url');
                localStorage.removeItem('tb_custom_ws_url');
                console.log('[Config] Switched to local API:', localApiUrl);
                break;

            case 'remote':
                _config.baseApiUrl = remoteApiUrl;
                _config.baseWsUrl = remoteWsUrl;
                _config.useRemoteApi = true;
                _config.endpointMode = 'remote';
                localStorage.setItem('tb_api_endpoint_mode', 'remote');
                localStorage.removeItem('tb_custom_api_url');
                localStorage.removeItem('tb_custom_ws_url');
                console.log('[Config] Switched to remote API:', remoteApiUrl);
                break;

            case 'custom':
                if (!customApiUrl) {
                    console.error('[Config] Custom API URL is required for custom mode');
                    return false;
                }
                // Ensure URL has /api suffix if not present
                const apiUrl = customApiUrl.endsWith('/api') ? customApiUrl :
                               customApiUrl.endsWith('/') ? customApiUrl + 'api' : customApiUrl + '/api';
                _config.baseApiUrl = apiUrl;
                _config.baseWsUrl = customWsUrl || remoteWsUrl;
                _config.useRemoteApi = true;
                _config.endpointMode = 'custom';
                localStorage.setItem('tb_api_endpoint_mode', 'custom');
                localStorage.setItem('tb_custom_api_url', apiUrl);
                if (customWsUrl) {
                    localStorage.setItem('tb_custom_ws_url', customWsUrl);
                }
                console.log('[Config] Switched to custom API:', apiUrl);
                break;

            default:
                console.error('[Config] Invalid endpoint mode:', mode);
                return false;
        }

        // Notify Tauri backend if available
        if (window.__TAURI__?.invoke) {
            window.__TAURI__.invoke('set_api_endpoint', { endpoint: _config.baseApiUrl })
                .catch(err => console.warn('[Config] Failed to notify Tauri backend:', err));
        }

        // Emit event for other components to react
        if (typeof window !== 'undefined') {
            window.dispatchEvent(new CustomEvent('tb:apiEndpointChanged', {
                detail: {
                    mode: _config.endpointMode,
                    apiUrl: _config.baseApiUrl,
                    wsUrl: _config.baseWsUrl
                }
            }));
        }

        return true;
    },

    /**
     * Get available endpoint options (for settings UI).
     * @returns {object|null} - Endpoint options or null if not available
     */
    getEndpointOptions: () => {
        if (!_config.canSwitchEndpoint) {
            return null;
        }
        return {
            currentMode: _config.endpointMode || 'local',
            local: _config._localApiUrl,
            remote: _config._remoteApiUrl,
            custom: localStorage.getItem('tb_custom_api_url') || null
        };
    }
};

export default Config;
