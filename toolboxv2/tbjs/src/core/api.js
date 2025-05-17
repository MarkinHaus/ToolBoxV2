// tbjs/core/api.js
// Handles all backend communication (HTTP, Tauri invoke).
// Original: httpSender.js, AuthHttpPostData, parts of router's fetch logic.

import config from './config.js';
import env from './env.js';
import events from './events.js';
import logger from './logger.js';
// Import Result, ToolBoxError, ToolBoxInterfaces structures (can be defined here or in utils.js)
// For now, let's assume they are available, e.g., from utils.js or defined inline

export const ToolBoxError = { /* ... from httpSender.js ... */ none: "none", input_error: "InputError", internal_error: "InternalError", custom_error: "CustomError" };
export const ToolBoxInterfaces = { /* ... from httpSender.js ... */ cli: "CLI", api: "API", remote: "REMOTE", native: "NATIVE" };
export class ToolBoxResult { /* ... from httpSender.js ... */ constructor(data_to = ToolBoxInterfaces.cli, data_info = null, data = null) { this.data_to = data_to; this.data_info = data_info; this.data = data; } }
export class ToolBoxInfo { /* ... from httpSender.js ... */ constructor(exec_code, help_text) { this.exec_code = exec_code; this.help_text = help_text; } }
export class Result { /* ... from httpSender.js ... */ constructor(origin = [], error = ToolBoxError.none, result = new ToolBoxResult(), info = new ToolBoxInfo(-1, "")) { this.origin = origin; this.error = error; this.result = result; this.info = info; } log(){ console.log("======== Result ========\nFunction Exec coed:", this.info.exec_code, "\nInfo's:",this.info.help_text, "<|>", this.result.data_info, "\nOrigin:",this.origin, "\nData_to:",this.result.data_to, "\nData:",this.result.data, "\nerror:",this.error, "\n------- EndOfD -------", ) } html(){ return `<div style="background-color: var(--background-color); border-radius: 5px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); padding: 2px;"> <div style="background-color: var(--background-color); padding: 5px;text-align:center"> <p>======== Result ========</p> <p>Function Exec code: <span id="execCode">`+this.info.exec_code+`</span></p> <p>Info's: <span id="infoText">`+this.info.help_text+`</span> <|> <span id="dataInfo">`+this.result.data_info+`</span></p> <p>Origin: <span id="originText">`+this.origin+`</span></p> <p>Data_to: <span id="dataTo">`+this.result.data_to+`</span></p> <p>Data: <span id="data">`+this.result.data+`</span></p> <p>Error: <span id="errorText">`+this.error+`</span></p> <p>------- EndOfD -------</p> </div></div>` } get() { return this.result.data; } }


function wrapApiResponse(data, source = "http") {
    // Simplified version of original wrapInResult
    // In production, ensure this robustly handles various backend response structures
    if (data && typeof data === 'object' && 'error' in data && 'result' in data && 'info' in data) {
        return new Result(data.origin || [source], data.error, new ToolBoxResult(data.result.data_to, data.result.data_info, data.result.data), new ToolBoxInfo(data.info.exec_code, data.info.help_text));
    }
    // Fallback for non-standard or direct data responses
    return new Result([source, 'wrapped'], ToolBoxError.none, new ToolBoxResult(ToolBoxInterfaces.native, "Direct data", data), new ToolBoxInfo(0, "Data wrapped by client"));
}


const Api = {
    _getRequestHeaders: (isJson = true) => {
        const headers = {
            'Accept': isJson ? 'application/json' : 'text/html',
        };
        if (isJson) {
            headers['Content-Type'] = 'application/json';
        }
        // Add Auth token if available
        // const token = TB.state.get('session.token'); // Example state path
        // if (token) headers['Authorization'] = `Bearer ${token}`;
        return headers;
    },

    /**
     * Makes a request to the backend.
     * @param {string} moduleName - The backend module name.
     * @param {string} functionName - The backend function name.
     * @param {object|string} [payload=null] - Data to send. If string, used as query params for GET/POST-URL.
     * @param {string} [method='POST'] - HTTP method.
     * @param {string} [useTauri='auto'] - 'auto', 'force', or 'never'.
     * @returns {Promise<Result>} - A promise resolving to a Result object.
     */
    request: async (moduleName, functionName, payload = null, method = 'POST', useTauri = 'auto') => {
        const command = `${moduleName}.${functionName}`; // For Tauri invoke

        if ((useTauri === 'auto' || useTauri === 'force') && env.isTauri()) {
            try {
                logger.debug(`[API] Attempting Tauri invoke: ${command}`, payload);
                // const tauriPayload = method === 'GET' && typeof payload === 'string' ? Api.parseQueryParams(payload) : payload;
                const tauriPayload = {};
                if(payload){
                    if(typeof payload === 'string'){
                         Api.parseQueryParams(payload, tauriPayload);
                    } else {
                        Object.assign(tauriPayload, payload);
                    }
                }

                const response = await window.__TAURI__.invoke(command, tauriPayload);
                logger.debug(`[API] Tauri invoke success for ${command}:`, response);
                return wrapApiResponse(response, 'tauri');
            } catch (error) {
                logger.warn(`[API] Tauri invoke failed for ${command}:`, error);
                if (useTauri === 'force') {
                    return new Result(['tauri', 'error'], ToolBoxError.internal_error, new ToolBoxResult(), new ToolBoxInfo(-1, `Tauri invoke ${command} failed: ${error.message || error}`));
                }
                // Fallback to HTTP if 'auto'
            }
        }

        // HTTP Fetch
        let url = `${config.get('baseApiUrl')}/${moduleName}/${functionName}`;
        const options = {
            method,
            headers: Api._getRequestHeaders(),
        };

        if (method.toUpperCase() === 'GET' || method.toUpperCase() === 'DELETE') {
            if (payload && typeof payload === 'string') url += `?${payload}`;
            else if (payload && typeof payload === 'object') url += `?${new URLSearchParams(payload).toString()}`;
        } else { // POST, PUT, PATCH
            if (payload && typeof payload === 'string' && method.toUpperCase() === 'POST') { // POST with URL-encoded form style params in URL
                 url += `?${payload}`;
            } else if (payload) {
                options.body = JSON.stringify(payload);
            }
        }


        logger.debug(`[API] HTTP ${method} request to: ${url}`, payload);
        try {
            const response = await fetch(url, options);
            const responseData = await response.json(); // Assuming JSON response
            if (!response.ok) {
                logger.error(`[API] HTTP error ${response.status} for ${url}:`, responseData);
                return wrapApiResponse(responseData || {
                    error: ToolBoxError.internal_error,
                    info: { exec_code: response.status, help_text: response.statusText },
                    result: {}
                });
            }
            logger.debug(`[API] HTTP success for ${url}:`, responseData);
            return wrapApiResponse(responseData);
        } catch (error) {
            logger.error(`[API] HTTP fetch error for ${url}:`, error);
            events.emit('api:networkError', { url, error });
            return new Result(['http', 'error'], ToolBoxError.internal_error, new ToolBoxResult(), new ToolBoxInfo(-1, `Network error: ${error.message || error}`));
        }
    },

    /**
     * Fetches HTML content for a given path.
     * @param {string} path - The path to the HTML file (e.g., /web/pages/home.html).
     * @returns {Promise<string>} - HTML content or error string.
     */
    fetchHtml: async (path) => {
        // Original: loadHtmlFile and its fallbacks from index.js
        // This simplified version assumes files are served from baseFileUrl
        const url = path.startsWith('http') ? path : `${config.get('baseFileUrl')}${path.startsWith('/') ? '' : '/'}${path}`;
        logger.debug(`[API] Fetching HTML from: ${url}`);
        try {
            const response = await fetch(url, { headers: Api._getRequestHeaders(false) });
            if (!response.ok) {
                logger.warn(`[API] HTTP error ${response.status} fetching HTML from ${url}`);
                return `HTTP error! status: ${response.status}`;
            }
            return await response.text();
        } catch (error) {
            logger.error(`[API] Error fetching HTML from ${url}:`, error);
            events.emit('api:networkError', { url, error });
            return `HTTP error! status: ${error.message || 'Network Error'}`;
        }
    },

    // Helper to parse query params for Tauri if needed (Tauri commands usually take objects)
    parseQueryParams: (paramString, targetObject = {}) => {
        new URLSearchParams(paramString).forEach((value, key) => {
            targetObject[key] = value;
        });
        return targetObject;
    },

    // Specific methods from original httpSender.js can be mapped here:
    httpPostUrl: (module_name, function_name, params, from_string = false) => {
        // The `request` method is more generic. This specific mapping might be less needed.
        // If `from_string` is true, it implies the backend returns a string that needs to be parsed as Result.
        // This is complex. Better for backend to always return structured JSON for API calls.
        // For now, let's assume standard JSON response.
        return Api.request(module_name, function_name, params, 'POST');
    },
    httpPostData: (module_name, function_name, data) => {
        return Api.request(module_name, function_name, data, 'POST');
    },
    AuthHttpPostData: (username) => { // Original AuthHttpPostData
        const endpoint = '/validateSession'; // Not module/function based
        const url = `${config.get('baseApiUrl')}${endpoint}`;
        const payload = {
            'Jwt_claim': localStorage.getItem('jwt_claim_device'), // Direct localStorage access, consider moving to TB.state
            'Username': username
        };
        logger.debug(`[API] Auth request to: ${url}`, payload);
        return fetch(url, { method: 'POST', headers: Api._getRequestHeaders(), body: JSON.stringify(payload) })
            .then(res => res.json().then(data => wrapApiResponse(data)))
            .catch(error => {
                logger.error(`[API] Auth fetch error for ${url}:`, error);
                return new Result(['auth', 'error'], ToolBoxError.internal_error, new ToolBoxResult(), new ToolBoxInfo(-1, `Auth Network error: ${error.message}`));
            });
    }
};

export default Api;
