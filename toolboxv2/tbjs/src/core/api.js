// tbjs/core/api.js
// Handles all backend communication (HTTP, Tauri invoke).
// Original: httpSender.js, AuthHttpPostData, parts of router's fetch logic.
import TB from '../index.js'; // Access TB.api, TB.config, TB.logger

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
    wrapApiResponse: wrapApiResponse,
    _getRequestHeaders: (isJson = true) => {
        const headers = {
            'Accept': isJson ? 'application/json' : 'text/html',
        };
        if (isJson) {
            headers['Content-Type'] = 'application/json';
        }
        // Add Auth token if available and NOT for an auth-specific path that shouldn't have it
        // This logic needs to be more nuanced if some auth paths need the token and others don't.
        const token = TB.state.get('user.token'); // Assuming user state is under 'user' namespace
        if (token) {
            // Check if the current operation is one that should *not* receive a token
            // For now, let's assume validateSession might be one such case if it's establishing a new session from a claim.
            // This needs to be decided based on your backend's requirements.
            // if ( !(moduleName === 'AuthManager' && functionName === 'validateSession') ) { // Example exclusion
                 headers['Authorization'] = `Bearer ${token}`;
            // }
        }
        return headers;
    },

    /**
     * Makes a request to the backend.
     * @param {string} moduleName - The backend module name, OR a full path starting with '/' for special routes.
     * @param {string} functionName - The backend function name (ignored if moduleName is a full path).
     * @param {object|string} [payload=null] - Data to send. If string, used as query params for GET/POST-URL.
     * @param {string} [method='POST'] - HTTP method.
     * @param {string} [useTauri='auto'] - 'auto', 'force', or 'never'.
     * @param {boolean} [isSpecialAuthRoute=false] - Indicates if this is a special auth route that might have different token handling.
     * @returns {Promise<Result>} - A promise resolving to a Result object.
     */
    request: async (moduleName, functionName, payload = null, method = 'POST', useTauri = 'auto', isSpecialAuthRoute = false) => {
        let command = `${moduleName}.${functionName}`; // For Tauri invoke
        let isFullPath = moduleName.startsWith('/');

        if (!isFullPath && (useTauri === 'auto' || useTauri === 'force') && env.isTauri()) {
            try {
                logger.debug(`[API] Attempting Tauri invoke: ${command}`, payload);
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
        let url;
        if (isFullPath) {
            if (moduleName.includes("IsValidSession") ||moduleName.includes("validateSession") || moduleName.startsWith("/web/")){
                url = `${config.get('baseApiUrl').replace('/api', '')}${moduleName}`;
                if (method==="POST" && moduleName.includes("IsValidSession")){
                    method = "GET"
                }
            }else{
                url = `${config.get('baseApiUrl')}${moduleName}`;
            }// moduleName is the path itself
             if (functionName && typeof functionName === 'string' && functionName.length > 0 && (method.toUpperCase() === 'GET' || method.toUpperCase() === 'DELETE')) {
                // If functionName is provided for a full path GET/DELETE, treat it as query string
                url += `?${functionName}`;
            } else if (functionName && typeof functionName === 'object' && (method.toUpperCase() === 'GET' || method.toUpperCase() === 'DELETE')) {
                 url += `?${new URLSearchParams(functionName).toString()}`;
            }

        } else {
            url = `${config.get('baseApiUrl')}/${moduleName}/${functionName}`;
        }

        const options = {
            method,
            // Headers now depend on whether it's a special auth route.
            // For AuthHttpPostData, it passed its own headers. We can mimic this by not adding auth token for it.
            headers: Api._getRequestHeaders(true), // Let _getRequestHeaders handle token logic generally
        };

        // Special handling for AuthHttpPostData which originally had its own Content-Type and Accept.
        // And it didn't send Authorization Bearer token.
        // This is tricky to generalize. If `isSpecialAuthRoute` is true, we might want different header logic.
        // For now, _getRequestHeaders handles token addition. If /validateSession should NOT have it,
        // the backend should simply ignore it, or _getRequestHeaders needs more specific rules.

        if (method.toUpperCase() === 'GET' || method.toUpperCase() === 'DELETE') {
            if (payload && typeof payload === 'string' && !isFullPath)
                url += `?${payload}`; // Only add payload as query if not already handled by functionName for full paths
            else if (payload && typeof payload === 'object' && !isFullPath) url += `?${new URLSearchParams(payload).toString()}`;
        } else { // POST, PUT, PATCH
             if (payload && typeof payload === 'string' && payload.includes('=')) { // Simple check for key=value pairs
            options.headers['Content-Type'] = 'application/x-www-form-urlencoded';
            options.body = payload;
            } else if (payload && typeof payload === 'string' && method.toUpperCase() === 'POST' && !isFullPath) {
                 url += `?${payload}`;
            }  else if (payload) {
                options.body = JSON.stringify(payload);
            }
        }


        logger.debug(`[API] HTTP ${method} request to: ${url}`, payload);
        try {
            const response = await fetch(url, options);
            // Handle cases where response might not be JSON (e.g. logout might return 204 No Content)
            let responseData;
            const contentType = response.headers.get("content-type");
            if (contentType && contentType.includes("application/json")) {
                responseData = await response.json();
            } else if (response.status === 204 || response.status === 205) { // No Content / Reset Content
                responseData = { success: true }; // Create a minimal success object
            } else {
                // Attempt to read as text if not JSON and not empty
                const textResponse = await response.text();
                try {
                    responseData = JSON.parse(textResponse); // Try parsing if it happens to be JSON anyway
                } catch (e) {
                     responseData = { message: textResponse }; // Fallback to text message
                }
            }


            if (!response.ok) {
                logger.error(`[API] HTTP error ${response.status} for ${url}:`, responseData);
                // Ensure wrapApiResponse can handle responseData that might not be a full Result structure
                const errorPayload = (responseData && typeof responseData === 'object') ? responseData : {};
                return wrapApiResponse({
                    error: errorPayload.error || ToolBoxError.internal_error,
                    info: errorPayload.info || { exec_code: response.status, help_text: (errorPayload?.message || response.statusText || "HTTP Error") },
                    result: errorPayload.result || {}
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
    httpPostUrl: (module_name, function_name, params= null, from_string = false) => {
        // The `request` method is more generic. This specific mapping might be less needed.
        // If `from_string` is true, it implies the backend returns a string that needs to be parsed as Result.
        // This is complex. Better for backend to always return structured JSON for API calls.
        // For now, let's assume standard JSON response.
        return Api.request(module_name, function_name, params, 'POST');
    },
    httpPostData: (module_name, function_name, data= null) => {
        return Api.request(module_name, function_name, data, 'POST');
    },
    // Updated AuthHttpPostData to use the new `request` method with a full path
    AuthHttpPostData: (username) => {
        const token = TB.state.get('user.token');
        if (!token) {
            return new Result(['token', 'error'], ToolBoxError.internal_error, new ToolBoxResult(), new ToolBoxInfo(-1, `no token found for ${username}`));
        }
        const endpoint = '/validateSession'; // This is the full path relative to baseApiUrl
        const payload = {
            'Jwt_claim': token, // Consider getting from TB.state if populated
            'Username': username
        };
        // Call `request` with moduleName as the path, and functionName as null or empty.
        // Set isSpecialAuthRoute if specific header handling is needed (e.g. no auth token).
        return Api.request(endpoint, null, payload, 'POST', 'never', true);
    },

    // Example for /web/logoutS which might be a GET or POST without a body
    logoutServer: async () => {
        const endpoint = '/web/logoutS'; // Or whatever the actual path is
        // Assuming it's a POST request that needs the current token to invalidate it on the server.
        // If it's GET and token is in cookie/header automatically, payload might be empty.
        const token = TB.state.get('user.token');
        if (!token) {
            logger.warn('[API] No token found for server logout.');
            // Still return a successful-like result for client-side logout to proceed
            return new Result(['logout', 'client-only'], ToolBoxError.none, new ToolBoxResult(), new ToolBoxInfo(0, "Client logout, no token for server."));
        }
        // The `request` method's _getRequestHeaders should add the token.
        return Api.request(endpoint, null, { token: token } /* or empty if server gets token from header/cookie */, 'POST', 'never', true);
    }
};

export default Api;

