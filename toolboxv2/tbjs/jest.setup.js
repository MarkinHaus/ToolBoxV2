// tbjs/jest.setup.js
// Mock localStorage
const localStorageMock = (function() {
  let store = {};
  return {
    getItem(key) {
      return store[key] || null;
    },
    setItem(key, value) {
      store[key] = value.toString();
    },
    removeItem(key) {
      delete store[key];
    },
    clear() {
      store = {};
    },
    get length() {
      return Object.keys(store).length;
    },
    key(i) {
      const keys = Object.keys(store);
      return keys[i] || null;
    }
  };
})();
Object.defineProperty(window, 'localStorage', { value: localStorageMock });
Object.defineProperty(window, 'sessionStorage', { value: localStorageMock }); // Kann auch einen separaten Mock haben

// Mock fetch
global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({}),
    text: () => Promise.resolve(''),
    headers: {
      get: jest.fn(header => {
        if (header === 'content-type') return 'application/json';
        return null;
      })
    }
  })
);

// Mock window.__TAURI__
global.window.__TAURI__ = {
  invoke: jest.fn(() => Promise.resolve({})),
  // Füge hier weitere Tauri-spezifische Mocks hinzu, falls benötigt
};

// Mock window.location
// delete window.location; // Zuerst löschen, da es nicht direkt überschreibbar ist
// window.location = {
//   href: 'http://localhost/',
//   origin: 'http://localhost',
//   pathname: '/',
//   search: '',
//   hash: '',
//   assign: jest.fn(),
//   replace: jest.fn(),
//   reload: jest.fn(),
// };
// Bessere Methode für window.location Mocking:
Object.defineProperty(window, 'location', {
  writable: true,
  value: {
    href: 'http://localhost/',
    origin: 'http://localhost',
    pathname: '/',
    search: '',
    hash: '',
    assign: jest.fn(),
    replace: jest.fn(),
    reload: jest.fn(),
    ancestorOrigins: {},
    protocol: 'http:',
    host: 'localhost',
    hostname: 'localhost',
    port: ''
  }
});


// Mock matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // deprecated
    removeListener: jest.fn(), // deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock für HTMX (sehr rudimentär)
global.window.htmx = {
  process: jest.fn(),
};

// Mock für TB-Struktur (damit Importe nicht fehlschlagen)
// Wird in spezifischen Tests detaillierter gemockt
global.TB = {
  config: {
    get: jest.fn(key => {
      if (key === 'baseApiUrl') return 'http://localhost/api';
      if (key === 'baseFileUrl') return 'http://localhost';
      return undefined;
    }),
    init: jest.fn(),
  },
  state: {
    get: jest.fn(),
    set: jest.fn(),
    init: jest.fn(),
    delete: jest.fn(),
    initVar: jest.fn(),
    delVar: jest.fn(),
    getVar: jest.fn(),
    setVar: jest.fn(),
  },
  logger: {
    log: jest.fn(),
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    init: jest.fn(),
    setLevel: jest.fn(),
  },
  events: {
    on: jest.fn(),
    off: jest.fn(),
    emit: jest.fn(),
    once: jest.fn(),
  },
  // ... weitere Core-Module rudimentär mocken
  env: {
    isTauri: jest.fn(() => false),
    isWeb: jest.fn(() => true),
  },
  api: { // Wird oft spezifischer gemockt pro Test
    request: jest.fn(),
    fetchHtml: jest.fn(),
    AuthHttpPostData: jest.fn(),
    logoutServer: jest.fn(),
  },
  ui: { // UI-Komponenten und -Module
    theme: {
      getCurrentMode: jest.fn(() => 'light'),
      setPreference: jest.fn(),
      init: jest.fn(),
    },
    Modal: {
      show: jest.fn(() => ({ close: jest.fn(), _modalElement: document.createElement('div') })),
    },
    Toast: {
        showInfo: jest.fn(),
        showError: jest.fn(),
    },
    Loader: {
        show: jest.fn(() => document.createElement('div')),
        hide: jest.fn(),
    },
    Button: {
        create: jest.fn(() => document.createElement('button'))
    },
    processDynamicContent: jest.fn(),
    // Weitere UI-Module hier mocken
  },
  utils: { // Rudimentär, da utils selbst getestet wird
    uniqueId: jest.fn(prefix => `${prefix}${Math.random().toString(36).substr(2, 9)}`),
    // ...
  }
};
// Mock für window.crypto.subtle (vereinfacht)
if (!global.window.crypto) {
    global.window.crypto = {};
}
if(!global.window.crypto.subtle){
    global.window.crypto.subtle = {
        generateKey: jest.fn(() => Promise.resolve({ publicKey: {}, privateKey: {} })),
        exportKey: jest.fn(() => Promise.resolve(new ArrayBuffer(8))),
        importKey: jest.fn(() => Promise.resolve({})),
        decrypt: jest.fn(() => Promise.resolve(new ArrayBuffer(8))),
        sign: jest.fn(() => Promise.resolve(new ArrayBuffer(8))),
        deriveKey: jest.fn(() => Promise.resolve({})),
    };
}
if(!global.window.crypto.getRandomValues){
    global.window.crypto.getRandomValues = jest.fn((array) => {
        for (let i = 0; i < array.length; i++) {
            array[i] = Math.floor(Math.random() * 256);
        }
        return array;
    });
}


// Mock für TextEncoder/TextDecoder
global.TextEncoder = class { encode() { return new Uint8Array(); } };
global.TextDecoder = class { decode() { return ''; } };

// Mock für URL.createObjectURL und revokeObjectURL
global.URL.createObjectURL = jest.fn();
global.URL.revokeObjectURL = jest.fn();

// Mock für navigator.credentials
if (!global.navigator.credentials) {
    global.navigator.credentials = {
        create: jest.fn(),
        get: jest.fn(),
    };
}
