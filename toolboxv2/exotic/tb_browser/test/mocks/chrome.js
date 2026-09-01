/**
 * chrome API mock — covers all APIs used by tb_browser
 */

const makeCallbackStore = () => {
  const listeners = [];
  const api = {
    addListener: (fn) => listeners.push(fn),
    removeListener: (fn) => { const i = listeners.indexOf(fn); if (i > -1) listeners.splice(i, 1); },
    hasListeners: () => listeners.length > 0,
    _fire: (...args) => listeners.forEach(fn => fn(...args)),
    _listeners: listeners,
  };
  return api;
};

// In-memory storage
const _syncStore = {};
const _localStore = {};
const _sessionStore = {};

const makeStorage = (store) => ({
  get: jest.fn(async (keys) => {
    if (typeof keys === 'string') return { [keys]: store[keys] };
    if (Array.isArray(keys)) return Object.fromEntries(keys.map(k => [k, store[k]]));
    if (typeof keys === 'object' && keys !== null) {
      return Object.fromEntries(Object.keys(keys).map(k => [k, store[k] ?? keys[k]]));
    }
    return { ...store };
  }),
  set: jest.fn(async (items) => { Object.assign(store, items); }),
  remove: jest.fn(async (keys) => {
    (Array.isArray(keys) ? keys : [keys]).forEach(k => delete store[k]);
  }),
  clear: jest.fn(async () => { Object.keys(store).forEach(k => delete store[k]); }),
  _store: store,
});

const chrome = {
  runtime: {
    id: 'test-extension-id',
    sendMessage: jest.fn(),
    sendNativeMessage: jest.fn(),
    onMessage: makeCallbackStore(),
    onInstalled: makeCallbackStore(),
    getURL: jest.fn((path) => `chrome-extension://test-id/${path}`),
    lastError: null,
  },
  storage: {
    sync: makeStorage(_syncStore),
    local: makeStorage(_localStore),
    session: makeStorage(_sessionStore),
  },
  tabs: {
    query: jest.fn(async () => [{ id: 1, url: 'https://example.com', windowId: 1 }]),
    sendMessage: jest.fn(async () => ({ success: true })),
    create: jest.fn(async (opts) => ({ id: 99, ...opts })),
    get: jest.fn(async (id) => ({ id, url: 'https://example.com', windowId: 1 })),
    onUpdated: makeCallbackStore(),
    onActivated: makeCallbackStore(),
  },
  action: {
    setBadgeText: jest.fn(),
    setBadgeBackgroundColor: jest.fn(),
    onClicked: makeCallbackStore(),
  },
  sidePanel: {
    open: jest.fn(async () => {}),
  },
  commands: {
    onCommand: makeCallbackStore(),
  },
  scripting: {
    executeScript: jest.fn(async () => {}),
    insertCSS: jest.fn(async () => {}),
  },
  notifications: {
    create: jest.fn(),
    clear: jest.fn(),
    onButtonClicked: makeCallbackStore(),
  },
  webNavigation: {
    onCompleted: makeCallbackStore(),
  },
  debugger: {
    attach: jest.fn(async () => {}),
    detach: jest.fn(async () => {}),
    sendCommand: jest.fn(async () => ({ nodes: [] })),
  },

  // helpers for test reset
  _reset() {
    Object.keys(_syncStore).forEach(k => delete _syncStore[k]);
    Object.keys(_localStore).forEach(k => delete _localStore[k]);
    Object.keys(_sessionStore).forEach(k => delete _sessionStore[k]);
    jest.clearAllMocks();
    chrome.runtime.id = 'test-extension-id';
    chrome.runtime.lastError = null;
  },
};

global.chrome = chrome;
module.exports = chrome;
