// tbjs/core/logger.js
// Centralized logging utility.
// Original: Scattered console.log statements.

const LogLevel = {
    DEBUG: 0,
    INFO: 1,
    WARN: 2,
    ERROR: 3,
    NONE: 4,
};

let currentLogLevel = LogLevel.DEBUG; // Default

const Logger = {
    init: (options = {logLevel: 'debug'}) => {
        Logger.setLevel(options.logLevel);
    },

    setLevel: (levelName) => {
        const levelKey = String(levelName).toUpperCase();
        if (LogLevel.hasOwnProperty(levelKey)) {
            currentLogLevel = LogLevel[levelKey];
        } else {
            console.warn(`[Logger] Invalid log level: ${levelName}. Defaulting to DEBUG.`);
            currentLogLevel = LogLevel.DEBUG;
        }
    },

    _log: (level, prefix, ...args) => {
        if (level >= currentLogLevel) {
            const timestamp = new Date().toLocaleTimeString();
            console[prefix.toLowerCase()](`[${timestamp}] [tbjs] [${prefix}]`, ...args);
        }
    },

    debug: (...args) => Logger._log(LogLevel.DEBUG, 'DEBUG', ...args),
    log: (...args) => Logger._log(LogLevel.INFO, 'INFO', ...args), // alias for info
    info: (...args) => Logger._log(LogLevel.INFO, 'INFO', ...args),
    warn: (...args) => Logger._log(LogLevel.WARN, 'WARN', ...args),
    error: (...args) => Logger._log(LogLevel.ERROR, 'ERROR', ...args),
};

export default Logger;
