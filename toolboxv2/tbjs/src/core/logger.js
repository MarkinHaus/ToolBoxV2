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
        // If init() is called, options.logLevel is 'debug'.
        // If init({}) is called, options.logLevel is undefined.
        //   In this case, String(undefined).toUpperCase() = 'UNDEFINED', which is invalid
        //   and will correctly default to DEBUG and issue a warning.
        // If init({logLevel: 'info'}) is called, it uses 'info'.
        Logger.setLevel(options.logLevel || 'debug'); // Ensure 'debug' if options.logLevel is falsy (e.g. undefined)
    },

    setLevel: (levelName) => {
        const levelKey = String(levelName).toUpperCase();
        if (LogLevel.hasOwnProperty(levelKey)) {
            currentLogLevel = LogLevel[levelKey];
        } else {
            const timestamp = new Date().toLocaleTimeString();
            // Call console.warn with two arguments: formatted prefix and the message,
            // to match test expectations and the general logging format.
            console.warn(
                `[${timestamp}] [tbjs] [WARN]`,
                `[Logger] Invalid log level: ${levelName}. Defaulting to DEBUG.`
            );
            currentLogLevel = LogLevel.DEBUG;
        }
    },

    // _log now takes messagePrefix (for the string output) and consoleMethodName (for the console function)
    _log: (level, messagePrefix, consoleMethodName, ...args) => {
        if (level >= currentLogLevel) {
            const timestamp = new Date().toLocaleTimeString();
            // Ensure the console method exists; fallback to console.log if not (e.g., console.debug might not be standard in all environments)
            const cslMethod = console[consoleMethodName] ? console[consoleMethodName] : console.log;
            cslMethod(`[${timestamp}] [tbjs] [${messagePrefix}]`, ...args);
        }
    },

    debug: (...args) => Logger._log(LogLevel.DEBUG, 'DEBUG', 'debug', ...args),
    // Logger.log and Logger.info will use 'console.log' but display '[INFO]' in the message
    log: (...args) => Logger._log(LogLevel.INFO, 'INFO', 'log', ...args), // alias for info
    info: (...args) => Logger._log(LogLevel.INFO, 'INFO', 'log', ...args),
    warn: (...args) => Logger._log(LogLevel.WARN, 'WARN', 'warn', ...args),
    error: (...args) => Logger._log(LogLevel.ERROR, 'ERROR', 'error', ...args),
};

export default Logger;
