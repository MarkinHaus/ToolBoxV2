// tbjs/core/logger.js
// Centralized logging with cookie-consent, remote batching, and audit trail.
// Replaces: scattered console.log statements.

const LogLevel = {
    DEBUG: 0,
    INFO: 1,
    WARN: 2,
    ERROR: 3,
    NONE: 4,
};

const LOG_LEVEL_NAMES = { 0: 'DEBUG', 1: 'INFO', 2: 'WARN', 3: 'ERROR', 4: 'NONE' };

let currentLogLevel = LogLevel.DEBUG;

// ============================================================================
// Cookie Consent — 'all' | 'essential' | 'errors' | 'none'
// ============================================================================

const CONSENT_COOKIE = 'tb_log_consent';
const CONSENT_MAX_AGE = 365 * 86400; // 1 year

function getLogConsent() {
    try {
        const m = document.cookie.match(new RegExp(`(?:^|;\\s*)${CONSENT_COOKIE}=([^;]+)`));
        return m ? m[1] : 'none';
    } catch {
        return 'none'; // SSR / node fallback
    }
}

function setLogConsent(level) {
    const valid = ['all', 'essential', 'errors', 'none'];
    const safe = valid.includes(level) ? level : 'none';
    try {
        document.cookie = `${CONSENT_COOKIE}=${safe};path=/;max-age=${CONSENT_MAX_AGE};SameSite=Lax`;
    } catch { /* node env */ }
    return safe;
}

// ============================================================================
// Remote Buffer — batched POST to /api/client-logs
// ============================================================================

const RemoteBuffer = {
    _queue: [],
    _timer: null,
    _flushInterval: 5000,
    _maxBatch: 50,
    _endpoint: '/api/client-logs',
    _sending: false,

    /** Check consent and push entry if allowed. */
    push(entry) {
        const consent = getLogConsent();
        if (consent === 'none') return;
        if (consent === 'errors' && entry.levelNum < LogLevel.ERROR) return;
        if (consent === 'essential' && entry.levelNum < LogLevel.WARN) return;
        // 'all' → everything

        this._queue.push(entry);
        if (this._queue.length >= this._maxBatch) this._flush();
        if (!this._timer) {
            this._timer = setInterval(() => this._flush(), this._flushInterval);
        }
    },

    /** Flush current queue to server. */
    _flush() {
        if (!this._queue.length || this._sending) return;
        const batch = this._queue.splice(0, this._maxBatch);
        const payload = JSON.stringify({ logs: batch });
        this._sending = true;

        // Primary: fetch with keepalive (survives page unload)
        const done = () => { this._sending = false; };
        try {
            fetch(this._endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: payload,
                keepalive: true,
                credentials: 'include',
            }).catch(() => {
                // Fallback: sendBeacon (fire-and-forget, no CORS issues)
                try { navigator.sendBeacon(this._endpoint, payload); } catch { /* lost */ }
            }).finally(done);
        } catch {
            try { navigator.sendBeacon(this._endpoint, payload); } catch { /* lost */ }
            done();
        }
    },

    /** Force-flush (call on page hide / before unload). */
    flush() {
        this._flush();
    },

    /** Update endpoint (e.g. when config loads late). */
    setEndpoint(url) {
        this._endpoint = url;
    },

    /** Stop the periodic timer. */
    destroy() {
        if (this._timer) { clearInterval(this._timer); this._timer = null; }
        this._flush(); // drain remaining
    },
};

// Flush on page hide (tab switch, close, navigate away)
if (typeof document !== 'undefined') {
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden') RemoteBuffer.flush();
    });
    window.addEventListener('pagehide', () => RemoteBuffer.flush());
}

// ============================================================================
// Audit Logger — structured frontend events
// ============================================================================

const AuditLogger = {
    /**
     * Log a structured audit event.
     * Audit entries bypass the log-level filter but still respect consent != 'none'.
     *
     * @param {string} action    - e.g. 'PAGE_VIEW', 'FORM_SUBMIT', 'CLICK', 'API_CALL'
     * @param {string} resource  - e.g. '/dashboard', 'login-form', '/api/projects'
     * @param {string} [status]  - 'SUCCESS' | 'FAILURE'
     * @param {object} [details] - arbitrary metadata
     */
    log(action, resource, status = 'SUCCESS', details = null) {
        if (getLogConsent() === 'none') return;

        const entry = {
            type: 'audit',
            timestamp: new Date().toISOString(),
            level: 'AUDIT',
            levelNum: LogLevel.WARN, // treat as essential for consent filtering
            source: 'tbjs',
            action,
            resource,
            status,
            details,
            url: typeof location !== 'undefined' ? location.pathname : '',
        };
        RemoteBuffer.push(entry);
    },
};

// ============================================================================
// Core Logger
// ============================================================================

function _serialize(arg) {
    if (arg === null) return 'null';
    if (arg === undefined) return 'undefined';
    if (arg instanceof Error) return `${arg.name}: ${arg.message}`;
    if (typeof arg === 'object') {
        try { return JSON.stringify(arg); } catch { return String(arg); }
    }
    return String(arg);
}

const Logger = {
    LogLevel,

    init(options = { logLevel: 'debug' }) {
        Logger.setLevel(options.logLevel || 'debug');
        if (options.remoteEndpoint) {
            RemoteBuffer.setEndpoint(options.remoteEndpoint);
        }
    },

    setLevel(levelName) {
        const key = String(levelName).toUpperCase();
        if (LogLevel.hasOwnProperty(key)) {
            currentLogLevel = LogLevel[key];
        } else {
            const ts = new Date().toLocaleTimeString();
            console.warn(`[${ts}] [tbjs] [WARN]`, `[Logger] Invalid log level: ${levelName}. Defaulting to DEBUG.`);
            currentLogLevel = LogLevel.DEBUG;
        }
    },

    _log(level, messagePrefix, consoleMethodName, ...args) {
        if (level < currentLogLevel) return;

        const ts = new Date().toLocaleTimeString();
        const csl = console[consoleMethodName] || console.log;
        csl(`[${ts}] [tbjs] [${messagePrefix}]`, ...args);

        // Remote reporting
        RemoteBuffer.push({
            type: 'log',
            timestamp: new Date().toISOString(),
            level: messagePrefix,
            levelNum: level,
            source: 'tbjs',
            message: args.map(_serialize).join(' '),
            url: typeof location !== 'undefined' ? location.pathname : '',
        });
    },

    debug: (...args) => Logger._log(LogLevel.DEBUG, 'DEBUG', 'debug', ...args),
    log:   (...args) => Logger._log(LogLevel.INFO,  'INFO',  'log',   ...args),
    info:  (...args) => Logger._log(LogLevel.INFO,  'INFO',  'log',   ...args),
    warn:  (...args) => Logger._log(LogLevel.WARN,  'WARN',  'warn',  ...args),
    error: (...args) => Logger._log(LogLevel.ERROR, 'ERROR', 'error', ...args),

    // Consent helpers exposed on Logger for convenience
    getConsent: getLogConsent,
    setConsent: setLogConsent,

    // Audit sub-logger
    audit: AuditLogger,

    // Remote buffer control
    remote: RemoteBuffer,

    /** Teardown — call on SPA unmount / app shutdown. */
    destroy() {
        RemoteBuffer.destroy();
    },
};

export { AuditLogger, RemoteBuffer, getLogConsent, setLogConsent };
export default Logger;
