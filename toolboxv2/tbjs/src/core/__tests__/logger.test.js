import Logger from '../logger.js'; // Passe den Pfad an

describe('Logger', () => {
  let consoleSpyDebug;
  let consoleSpyLog;
  let consoleSpyWarn;
  let consoleSpyError;

  beforeEach(() => {
    // Spy auf console Methoden vor jedem Test
    consoleSpyDebug = jest.spyOn(console, 'debug').mockImplementation(() => {});
    consoleSpyLog = jest.spyOn(console, 'log').mockImplementation(() => {});
    consoleSpyWarn = jest.spyOn(console, 'warn').mockImplementation(() => {});
    consoleSpyError = jest.spyOn(console, 'error').mockImplementation(() => {});
    // Standard LogLevel für Tests setzen
    Logger.init({ logLevel: 'debug' }); // This will call setLevel('debug')
  });

  afterEach(() => {
    // Spies nach jedem Test wiederherstellen
    jest.restoreAllMocks();
  });

  it('should initialize with a default log level', () => {
    Logger.init(); // Ruft init ohne Optionen auf, sollte auf DEBUG defaulten via {logLevel: 'debug'} default param
    expect(consoleSpyWarn).not.toHaveBeenCalledWith(
        expect.stringMatching(/\[.+\] \[tbjs\] \[WARN\]/), // Check for the formatted warning
        expect.stringContaining('[Logger] Invalid log level')
    );
    Logger.debug('test');
    expect(consoleSpyDebug).toHaveBeenCalledWith(
        expect.stringMatching(/\[.+] \[tbjs\] \[DEBUG\]/), // Zeitstempel und Präfix
        'test'
    );
  });

    it('should set log level correctly', () => {
      Logger.setLevel('warn');
      Logger.info('Info message'); // Should not be logged
      Logger.warn('Warn message'); // Should be logged
      expect(consoleSpyLog).not.toHaveBeenCalled();
      expect(consoleSpyWarn).toHaveBeenCalledWith(
        expect.stringMatching(/\[.+] \[tbjs] \[WARN\]/), // Zeitstempel und Präfix
        'Warn message'
      );
    });

    it('should default to DEBUG if invalid log level is set', () => {
      Logger.setLevel('invalid');
      // Expect the warning about invalid log level (now called with two arguments by logger.js)
      expect(consoleSpyWarn).toHaveBeenCalledWith(
        expect.stringMatching(/\[.+\] \[tbjs\] \[WARN\]/),
        '[Logger] Invalid log level: invalid. Defaulting to DEBUG.'
      );

      consoleSpyWarn.mockClear(); // Clear spy for next check
      consoleSpyDebug.mockClear(); // Clear spy for next check

      // Logger is now in DEBUG mode
      Logger.debug('Debug message after invalid set');
      // Expect a standard debug message, not the warning message
      expect(consoleSpyDebug).toHaveBeenCalledWith(
        expect.stringMatching(/\[.+] \[tbjs\] \[DEBUG\]/),
        'Debug message after invalid set'
      );
      expect(consoleSpyWarn).not.toHaveBeenCalled(); // Ensure no new warnings
    });

  it('should log debug messages if level is DEBUG', () => {
    Logger.setLevel('debug');
    Logger.debug('Test debug');
    expect(consoleSpyDebug).toHaveBeenCalledWith(
        expect.stringMatching(/\[.+] \[tbjs\] \[DEBUG\]/), // Zeitstempel und Präfix
        'Test debug'
    );
  });

  it('should not log debug messages if level is INFO', () => {
    Logger.setLevel('info');
    Logger.debug('Test debug');
    expect(consoleSpyDebug).not.toHaveBeenCalled();
  });

    it('should log info messages if level is INFO or DEBUG', () => {
      Logger.setLevel('info');
      Logger.info('Test info');
      // Logger.info now calls console.log, with "[INFO]" in the message
      expect(consoleSpyLog).toHaveBeenCalledWith(
          expect.stringMatching(/\[.+\] \[tbjs\] \[INFO\]/),
          'Test info'
      );

      consoleSpyLog.mockClear();
      Logger.setLevel('debug');
      Logger.info('Test info debug');
      expect(consoleSpyLog).toHaveBeenCalledWith(
          expect.stringMatching(/\[.+\] \[tbjs\] \[INFO\]/),
          'Test info debug'
      );
    });

  it('should log warn messages if level is WARN, INFO or DEBUG', () => {
    Logger.setLevel('warn'); // Current level is WARN
    Logger.warn('Test warn');
    // Expect console.warn to be called with 2 arguments
    expect(consoleSpyWarn).toHaveBeenCalledWith(
        expect.stringMatching(/\[.+] \[tbjs\] \[WARN\]/),
        'Test warn'
    );
  });

  it('should log error messages unless level is NONE', () => {
    Logger.setLevel('error'); // Current level is ERROR
    Logger.error('Test error');
    // Expect console.error to be called with 2 arguments
    expect(consoleSpyError).toHaveBeenCalledWith(
        expect.stringMatching(/\[.+] \[tbjs\] \[ERROR\]/),
        'Test error'
    );
  });

  it('should not log any messages if level is NONE', () => {
    Logger.setLevel('none');

    // Clear any potential calls from setLevel or beforeEach (though they shouldn't warn/log here)
    consoleSpyDebug.mockClear();
    consoleSpyLog.mockClear();
    consoleSpyWarn.mockClear();
    consoleSpyError.mockClear();

    Logger.debug('No debug');
    Logger.info('No info');
    Logger.warn('No warn');
    Logger.error('No error');

    expect(consoleSpyDebug).not.toHaveBeenCalled();
    expect(consoleSpyLog).not.toHaveBeenCalled();
    expect(consoleSpyWarn).not.toHaveBeenCalled(); // Simpler: no warn calls should happen at all
    expect(consoleSpyError).not.toHaveBeenCalled();
  });

    it('log alias should work like info', () => {
      Logger.setLevel('info');
      Logger.log('Test log alias'); // Logger.log is an alias for Logger.info, uses console.log
      // Expect console.log to be called, with "[INFO]" in the message
      expect(consoleSpyLog).toHaveBeenCalledWith(
          expect.stringMatching(/\[.+\] \[tbjs\] \[INFO\]/),
          'Test log alias'
      );
    });
});
