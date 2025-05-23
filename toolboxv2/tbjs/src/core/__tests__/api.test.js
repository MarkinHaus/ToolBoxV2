// tbjs/core/__tests__/api.test.js

import Api, { Result, ToolBoxError, ToolBoxInterfaces, ToolBoxResult, ToolBoxInfo } from '../api.js';
import TB from '../../index.js'; // Still needed for TB.state mock

// Mock direct dependencies of api.js
jest.mock('../config.js', () => ({
  get: jest.fn(),
}));
jest.mock('../env.js'); // Will be auto-mocked, methods will be jest.fn()
jest.mock('../logger.js', () => ({
  debug: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
  log: jest.fn(),
}));
jest.mock('../events.js', () => ({
  emit: jest.fn(),
}));

// Import the mocked versions to configure them
import config from '../config.js';
import env from '../env.js';
import logger from '../logger.js';
import events from '../events.js';


// Mock TB's relevant parts that api.js *does* use (TB.state)
// Other parts of TB that api.js doesn't directly use for config/logging/events are not needed here.
jest.mock('../../index.js', () => ({
  state: {
    get: jest.fn(),
  },
  // These are useful if test assertions compare against them, but api.js uses its own exports for these.
  ToolBoxError: { none: "none", internal_error: "InternalError" },
  ToolBoxInterfaces: { native: "NATIVE" },
}));


describe('API Module', () => {
  const mockBaseApiUrl = 'http://localhost:3000/api';
  const mockBaseFileUrl = 'http://localhost:3000';

  beforeEach(() => {
    jest.clearAllMocks(); // Clears all mocks, including those from jest.mock

    // Configure the direct config mock
    config.get.mockImplementation(key => {
      if (key === 'baseApiUrl') return mockBaseApiUrl;
      if (key === 'baseFileUrl') return mockBaseFileUrl;
      return undefined;
    });

    // Configure TB.state.get mock (used by api.js)
    TB.state.get.mockReturnValue(null); // Standardmäßig kein Token

    // Default mock for env.isTauri()
    env.isTauri.mockReturnValue(false);


    global.fetch.mockClear();
    if (global.window && global.window.__TAURI__ && global.window.__TAURI__.invoke) {
        global.window.__TAURI__.invoke.mockClear();
    }
  });

  describe('Result Classes', () => {
    it('Result constructor should create a default Result object', () => {
      const res = new Result();
      expect(res.error).toBe(ToolBoxError.none);
      expect(res.result).toBeInstanceOf(ToolBoxResult);
      expect(res.info).toBeInstanceOf(ToolBoxInfo);
      expect(res.info.exec_code).toBe(-1);
    });

    it('Result.get() should return result.data', () => {
        const data = { test: 'value' };
        const res = new Result([], ToolBoxError.none, new ToolBoxResult(ToolBoxInterfaces.cli, null, data));
        expect(res.get()).toEqual(data);
    });
  });

  describe('wrapApiResponse', () => {
    it('should correctly wrap a standard API success response', () => {
      const mockResponseData = {
        origin: ['server'],
        error: ToolBoxError.none,
        result: { data_to: ToolBoxInterfaces.api, data_info: 'Success', data: { id: 1 } },
        info: { exec_code: 0, help_text: 'Operation successful' },
      };
      const result = Api.wrapApiResponse(mockResponseData, 'http');
      expect(result).toBeInstanceOf(Result);
      expect(result.error).toBe(ToolBoxError.none);
      expect(result.result.data).toEqual({ id: 1 });
      expect(result.info.help_text).toBe('Operation successful');
      expect(result.origin).toEqual(['server']);
    });

    it('should wrap direct data as a native Result', () => {
      const directData = { message: 'Hello' };
      const result = Api.wrapApiResponse(directData, 'customSource');
      expect(result).toBeInstanceOf(Result);
      expect(result.error).toBe(ToolBoxError.none);
      // ToolBoxInterfaces.native is defined in api.js and imported by the test
      expect(result.result.data_to).toBe(ToolBoxInterfaces.native);
      expect(result.result.data).toEqual(directData);
      expect(result.origin).toEqual(['customSource', 'wrapped']);
      expect(result.info.exec_code).toBe(0);
    });
  });


  describe('_getRequestHeaders', () => {
    it('should include Content-Type and Accept for JSON by default', () => {
      const headers = Api._getRequestHeaders();
      expect(headers['Content-Type']).toBe('application/json');
      expect(headers['Accept']).toBe('application/json');
    });

    it('should include Accept for HTML if isJson is false', () => {
      const headers = Api._getRequestHeaders(false);
      expect(headers['Content-Type']).toBeUndefined();
      expect(headers['Accept']).toBe('text/html');
    });

    it('should include Authorization header if token exists', () => {
      TB.state.get.mockReturnValueOnce('test-token-123');
      const headers = Api._getRequestHeaders();
      expect(headers['Authorization']).toBe('Bearer test-token-123');
    });

    it('should not include Authorization header if no token', () => {
      TB.state.get.mockReturnValueOnce(null);
      const headers = Api._getRequestHeaders();
      expect(headers['Authorization']).toBeUndefined();
    });
  });

  describe('request - HTTP Fetch', () => {
    it('should make a POST request with JSON payload', async () => {
      const payload = { data: 'test' };
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ error: ToolBoxError.none, result: { data: 'response' }, info: { exec_code: 0 } }),
        headers: { get: () => 'application/json' }
      });

      const result = await Api.request('TestModule', 'testFunc', payload, 'POST');

      expect(global.fetch).toHaveBeenCalledWith(
        `${mockBaseApiUrl}/TestModule/testFunc`,
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify(payload),
          headers: expect.any(Object),
        })
      );
      expect(result.get()).toBe('response');
    });

    it('should make a GET request with query string from object payload', async () => {
      const payload = { param1: 'value1', param2: 'value2' };
       global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ error: ToolBoxError.none, result: { data: 'response' }, info: { exec_code: 0 } }),
        headers: { get: () => 'application/json' }
      });

      await Api.request('TestModule', 'testFunc', payload, 'GET');
      // Corrected URLSearchParams behavior for param2, ensure & is used
      const expectedUrl = `${mockBaseApiUrl}/TestModule/testFunc?param1=value1&param2=value2`; // FIX: ¶ to &
      expect(global.fetch).toHaveBeenCalledWith(
        expectedUrl,
        expect.objectContaining({ method: 'GET', headers: expect.any(Object) })
      );
    });

     it('should handle x-www-form-urlencoded POST payload', async () => {
      const payload = 'key1=value1&key2=value2';
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ error: ToolBoxError.none, result: { data: 'response' }, info: { exec_code: 0 } }),
        headers: { get: () => 'application/json' }
      });

      await Api.request('TestModule', 'testFunc', payload, 'POST');
      expect(global.fetch).toHaveBeenCalledWith(
        `${mockBaseApiUrl}/TestModule/testFunc`,
        expect.objectContaining({
          method: 'POST',
          body: payload,
          headers: expect.objectContaining({
            'Content-Type': 'application/x-www-form-urlencoded',
          }),
        })
      );
    });

    it('should handle full path GET request with payload as query string', async () => {
        const moduleName = '/custom/path';
        const functionName = { query: 'param' }; // functionName is payload for full path GET
        global.fetch.mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({ error: ToolBoxError.none, result: { data: 'response' }, info: { exec_code: 0 } }),
            headers: { get: () => 'application/json' }
        });

        await Api.request(moduleName, functionName, null, 'GET');
        const expectedUrl = `${mockBaseApiUrl}/custom/path?query=param`;
        expect(global.fetch).toHaveBeenCalledWith(
            expectedUrl,
            expect.objectContaining({ method: 'GET' })
        );
    });

     it('should handle full path IsValidSession/validateSession URLs correctly', async () => {
        const moduleNameValidate = '/validateSession';
        global.fetch.mockResolvedValueOnce({
            ok: true, json: () => Promise.resolve({ error: ToolBoxError.none, result: { data: 'ok' }, info: {exec_code: 0} }), headers: { get: () => 'application/json' }
        });
        await Api.request(moduleNameValidate, null, {}, 'POST');
        expect(global.fetch).toHaveBeenCalledWith(
            `${mockBaseApiUrl.replace('/api', '')}/validateSession`,
            expect.objectContaining({ method: 'POST' })
        );
    });

    it('should handle network errors', async () => {
      global.fetch.mockRejectedValueOnce(new Error('Network failure'));
      const result = await Api.request('NetModule', 'netFunc', {}, 'POST');
      expect(result.error).toBe(ToolBoxError.internal_error);
      expect(result.info.help_text).toContain('Network error: Network failure');
      expect(events.emit).toHaveBeenCalledWith('api:networkError', expect.objectContaining({
          url: `${mockBaseApiUrl}/NetModule/netFunc`, // or whatever the constructed URL is
          error: new Error('Network failure')
      }));
    });

    it('should handle HTTP error responses', async () => {
        global.fetch.mockResolvedValueOnce({
            ok: false,
            status: 401,
            statusText: 'Unauthorized',
            json: () => Promise.resolve({ error: "AuthError", info: { help_text: "Token invalid" } }),
            headers: { get: () => 'application/json' }
        });
        const result = await Api.request('AuthModule', 'authFunc', {});
        expect(result.error).toBe("AuthError");
        expect(result.info.help_text).toBe("Token invalid");
    });

    it('should handle non-JSON error responses', async () => {
        global.fetch.mockResolvedValueOnce({
            ok: false,
            status: 500,
            statusText: 'Server Error',
            text: () => Promise.resolve('Internal Server Error HTML page'),
            headers: { get: () => 'text/html' }
        });
        const result = await Api.request('ErrorModule', 'errorFunc', {});
        expect(result.error).toBe(ToolBoxError.internal_error);
        expect(result.info.help_text).toBe('Internal Server Error HTML page'); // This comes from responseData.message
        expect(result.info.exec_code).toBe(500);
    });

     it('should handle 204 No Content responses', async () => {
        global.fetch.mockResolvedValueOnce({
            ok: true,
            status: 204,
            headers: { get: () => null }
        });
        const result = await Api.request('LogoutModule', 'logoutFunc', null, 'POST');
        expect(result.error).toBe(ToolBoxError.none);
        expect(result.get()).toEqual({ success: true });
    });
  });

  describe('request - Tauri Invoke', () => {
    beforeEach(() => {
      env.isTauri.mockReturnValue(true);
      // Ensure __TAURI__ is on window for these tests if not globally set up by Jest
      if (!global.window.__TAURI__) {
        global.window.__TAURI__ = { invoke: jest.fn() };
      }
      global.window.__TAURI__.invoke.mockClear();
    });

    afterEach(() => {
      env.isTauri.mockReturnValue(false); // Reset for other tests
    });

    it('should use Tauri invoke if available and useTauri is "auto" or "force"', async () => {
      global.window.__TAURI__.invoke.mockResolvedValueOnce({
        error: ToolBoxError.none, result: { data: 'tauri_response' }, info: { exec_code: 0 }
      });
      const payload = { id: 1 };
      const result = await Api.request('TauriModule', 'tauriFunc', payload, 'POST', 'auto');

      expect(global.window.__TAURI__.invoke).toHaveBeenCalledWith('TauriModule.tauriFunc', payload);
      expect(global.fetch).not.toHaveBeenCalled();
      expect(result.get()).toBe('tauri_response');
      expect(result.origin).toEqual(['tauri']);
    });

    it('should fallback to HTTP if Tauri invoke fails and useTauri is "auto"', async () => {
      global.window.__TAURI__.invoke.mockRejectedValueOnce(new Error('Tauri error'));
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ error: ToolBoxError.none, result: { data: 'http_fallback' }, info: { exec_code: 0 } }),
        headers: { get: () => 'application/json' }
      });

      const result = await Api.request('TauriModule', 'tauriFunc', {}, 'POST', 'auto');

      expect(global.window.__TAURI__.invoke).toHaveBeenCalled();
      expect(logger.warn).toHaveBeenCalledWith('[API] Tauri invoke failed for TauriModule.tauriFunc:', new Error('Tauri error'));
      expect(global.fetch).toHaveBeenCalled();
      expect(result.get()).toBe('http_fallback');
    });

    it('should return an error if Tauri invoke fails and useTauri is "force"', async () => {
      global.window.__TAURI__.invoke.mockRejectedValueOnce(new Error('Forced Tauri error'));
      const result = await Api.request('TauriModule', 'tauriFunc', {}, 'POST', 'force');

      expect(global.window.__TAURI__.invoke).toHaveBeenCalled();
      expect(global.fetch).not.toHaveBeenCalled();
      expect(result.error).toBe(ToolBoxError.internal_error);
      expect(result.info.help_text).toContain('Tauri invoke TauriModule.tauriFunc failed: Forced Tauri error');
    });

    it('should parse string payload to object for Tauri invoke', async () => {
        global.window.__TAURI__.invoke.mockResolvedValueOnce({
            error: ToolBoxError.none, result: { data: 'tauri_response_parsed' }, info: { exec_code: 0 }
        });
        // Corrected payload string with '&'
        const payloadString = 'param1=value1&param2=value2'; // FIX: ¶ to &
        const expectedTauriPayload = { param1: 'value1', param2: 'value2' };

        await Api.request('TauriParseModule', 'parseFunc', payloadString, 'POST', 'force');
        expect(global.window.__TAURI__.invoke).toHaveBeenCalledWith('TauriParseModule.parseFunc', expectedTauriPayload);
    });
  });

  describe('fetchHtml', () => {
    it('should fetch HTML content successfully', async () => {
      const htmlContent = '<h1>Hello</h1>';
      global.fetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(htmlContent),
      });

      const result = await Api.fetchHtml('/page.html');
      expect(global.fetch).toHaveBeenCalledWith(`${mockBaseFileUrl}/page.html`, expect.objectContaining({
          headers: expect.objectContaining({'Accept': 'text/html'})
      }));
      expect(result).toBe(htmlContent);
    });

    it('should handle errors when fetching HTML', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found',
      });
      const result = await Api.fetchHtml('/nonexistent.html');
      expect(result).toBe('HTTP error! status: 404');
      expect(logger.warn).toHaveBeenCalledWith( // Use the imported logger
        '[API] HTTP error 404 fetching HTML from http://localhost:3000/nonexistent.html'
      );
    });
  });

  describe('AuthHttpPostData', () => {
    it('should make a POST request to /validateSession with token and username', async () => {
        const username = 'testuser';
        const token = 'fake-jwt-token';
        TB.state.get.mockImplementation(key => (key === 'user.token' ? token : (key === 'user.username' ? username : null)));

        global.fetch.mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({ error: ToolBoxError.none, result: { data: { valid: true } }, info: { exec_code: 0 } }),
            headers: { get: () => 'application/json' }
        });

        // Assuming AuthHttpPostData uses TB.state to get username if not passed or for consistency
        // The original function signature was AuthHttpPostData(username), so we'll stick to that.
        // The payload in api.js uses window.TB.state.get('user.token'), which is mocked by TB.state.get
        const result = await Api.AuthHttpPostData(username);


        expect(global.fetch).toHaveBeenCalledWith(
            `${mockBaseApiUrl.replace('/api', '')}/validateSession`,
            expect.objectContaining({
                method: 'POST',
                body: JSON.stringify({ Jwt_claim: token, Username: username }),
            })
        );
        expect(result.get()).toEqual({ valid: true });
    });

    it('should return an error if no token is found for AuthHttpPostData', async () => {
        TB.state.get.mockReturnValue(null); // No token
        const result = await Api.AuthHttpPostData('testuser'); // Pass username as per function signature
        expect(result.error).toBe(ToolBoxError.internal_error);
        expect(result.info.help_text).toContain('no token found');
        expect(global.fetch).not.toHaveBeenCalled();
    });
  });

  describe('logoutServer', () => {
     it('should make a POST request to /web/logoutS with token', async () => {
        const token = 'fake-logout-token';
        TB.state.get.mockImplementation(key => (key === 'user.token' ? token : null));
        global.fetch.mockResolvedValueOnce({
            ok: true, status: 204, headers: { get: () => null }
        });

        const result = await Api.logoutServer();
        expect(global.fetch).toHaveBeenCalledWith(
            `${mockBaseApiUrl.replace('/api','')}/web/logoutS`,
            expect.objectContaining({
                method: 'POST',
                body: JSON.stringify({ token: token }) // As per api.js implementation
            })
        );
        expect(result.error).toBe(ToolBoxError.none);
    });

    it('should return a client-only success if no token for logoutServer', async () => {
        TB.state.get.mockReturnValue(null);
        const result = await Api.logoutServer();
        expect(result.origin).toEqual(['logout', 'client-only']);
        expect(result.error).toBe(ToolBoxError.none);
        expect(global.fetch).not.toHaveBeenCalled();
    });
  });

});
