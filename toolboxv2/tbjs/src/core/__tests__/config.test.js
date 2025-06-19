import Config from '../config.js';
import Logger from '../logger.js';

jest.mock('../logger.js', () => ({
  debug: jest.fn(),
  warn: jest.fn(),
  log: jest.fn(), // Mock console.log for config init
}));

// Mock window.location für baseFileUrl und isProduction Inferenz
const mockLocation = (hostname = 'localhost', origin = 'http://localhost', pathname = '/') => {
  Object.defineProperty(window, 'location', {
    writable: true,
    value: {
      hostname,
      origin,
      pathname,
      href: `${origin}${pathname}`
    }
  });
};


describe('Config', () => {
  const originalConsoleLog = console.log;
  const originalConsoleWarn = console.warn;

  beforeEach(() => {
    Config._config = {}; // Reset internal config state
    Logger.debug.mockClear();
    Logger.warn.mockClear();
    // console.log wird in Config.init verwendet
    console.log = jest.fn();
    console.warn = jest.fn();
    mockLocation(); // Default localhost
  });

  afterEach(() => {
    console.log = originalConsoleLog;
    console.warn = originalConsoleWarn;
  });

  describe('init', () => {
    it('should initialize with default values', () => {
      Config.init({});
      expect(Config.get('appRootId')).toBe('app-root');
      expect(Config.get('baseApiUrl')).toBe('http://localhost/api'); // Normalisiert
      expect(Config.get('logLevel')).toBe('info');
      expect(Config.get('isProduction')).toBe(false); // Inferred from localhost
    });

    it('should allow overriding default values', () => {
      const userConfig = {
        appRootId: 'my-app',
        baseApiUrl: '/my-api',
        logLevel: 'warn',
        isProduction: true,
      };
      Config.init(userConfig);
      expect(Config.get('appRootId')).toBe('my-app');
      expect(Config.get('baseApiUrl')).toBe('http://localhost/my-api');
      expect(Config.get('logLevel')).toBe('warn');
      expect(Config.get('isProduction')).toBe(true);
    });

    it('should deeply merge themeSettings', () => {
      const userConfig = {
        themeSettings: {
          defaultPreference: 'dark',
          background: {
            light: { color: '#FAFAFA' },
            dark: { image: '/dark.png' },
          },
        },
      };
      Config.init(userConfig);
      const theme = Config.get('themeSettings');
      expect(theme.defaultPreference).toBe('dark');
      expect(theme.background.light.color).toBe('#FAFAFA');
      expect(theme.background.dark.color).toBe('#121212'); // Default dark color
      expect(theme.background.dark.image).toBe('/dark.png');
    });

    it('should normalize baseApiUrl to be absolute', () => {
      Config.init({ baseApiUrl: 'api/v1' }); // Relative
      expect(Config.get('baseApiUrl')).toBe('http://localhost/api/v1');

      Config.init({ baseApiUrl: '/abs/api' }); // Absolute path
      expect(Config.get('baseApiUrl')).toBe('http://localhost/abs/api');

      Config.init({ baseApiUrl: 'https://example.com/api' }); // Full URL
      expect(Config.get('baseApiUrl')).toBe('https://example.com/api');
    });

    it('should normalize baseFileUrl to end with / if it has a path', () => {
      mockLocation('test.com', 'https://test.com');
      Config.init({ baseFileUrl: 'https://test.com/app' });
      expect(Config.get('baseFileUrl')).toBe('https://test.com/app/');

      Config.init({ baseFileUrl: 'https://test.com/app/' }); // Already ends with /
      expect(Config.get('baseFileUrl')).toBe('https://test.com/app/');

      Config.init({ baseFileUrl: 'https://test.com' }); // Just origin
      expect(Config.get('baseFileUrl')).toBe('https://test.com');
    });

    it('should infer isProduction based on hostname if not set', () => {
      mockLocation('production.com', 'https://production.com');
      Config.init({});
      expect(Config.get('isProduction')).toBe(true);

      mockLocation('localhost', 'http://localhost');
      Config.init({});
      expect(Config.get('isProduction')).toBe(false);
    });
  });

  describe('get', () => {
    beforeEach(() => {
      Config.init({
        topLevel: 'value1',
        nested: { level2: 'value2', deep: { level3: 'value3' } },
      });
    });

    it('should get a top-level key', () => {
      expect(Config.get('topLevel')).toBe('value1');
    });

    it('should get a nested key using dot notation', () => {
      expect(Config.get('nested.level2')).toBe('value2');
      expect(Config.get('nested.deep.level3')).toBe('value3');
    });

    it('should return undefined for non-existent keys', () => {
      expect(Config.get('nonExistent')).toBeUndefined();
      expect(Config.get('nested.nonExistent')).toBeUndefined();
      expect(Config.get('nested.deep.nonExistent.level4')).toBeUndefined();
    });

    it('should return undefined if no key is passed', () => {
        expect(Config.get()).toBeUndefined();
    });
  });

    describe('getAll', () => {
      it('should return a copy of the entire config object', () => {
        const initialUserConf = { test: 'data', logLevel: 'debug' }; // Verwende ein Test-Config-Objekt
        Config.init(initialUserConf);
        const allConfig = Config.getAll();

        // Erstelle ein erwartetes Objekt, das die Defaults und die User-Config merged,
        // so wie es Config.init intern tun würde.
        const expectedConfigShape = {
          appRootId: 'app-root',
          baseApiUrl: 'http://localhost/api', // Normalisiert durch init
          baseFileUrl: 'http://localhost',    // Normalisiert durch init
          initialState: {},
          isProduction: false, // Inferred
          logLevel: 'debug',   // Aus initialUserConf
          routes: [],
          serviceWorker: {
            enabled: false,
            scope: '/',
            url: '/sw.js',
          },
          themeSettings: {
            defaultPreference: 'system',
            background: {
              type: 'color',
              light: { color: '#FFFFFF', image: '' },
              dark: { color: '#121212', image: '' },
              placeholder: { image: '', displayUntil3DReady: true }
            }
          },
          test: 'data' // Aus initialUserConf
        };

        expect(allConfig).toEqual(expectedConfigShape);
        expect(allConfig).not.toBe(Config._config); // Referenzprüfung bleibt wichtig
      });
    });

  describe('set', () => {
    beforeEach(() => Config.init({}));

    it('should set a top-level key', () => {
      Config.set('newKey', 'newValue');
      expect(Config.get('newKey')).toBe('newValue');
      expect(Logger.debug).toHaveBeenCalledWith('[Config] Set: newKey =', 'newValue');
    });

    it('should set a nested key using dot notation', () => {
      Config.set('parent.child.grandchild', 'nestedValue');
      expect(Config.get('parent.child.grandchild')).toBe('nestedValue');
      expect(Config.get('parent.child')).toEqual({ grandchild: 'nestedValue' });
    });

    it('should create intermediate objects if they do not exist when setting nested keys', () => {
      Config.set('newParent.newChild', 'childValue');
      expect(Config.get('newParent')).toEqual({ newChild: 'childValue' });
    });
  });
});
