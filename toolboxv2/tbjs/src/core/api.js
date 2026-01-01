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

    _readFileAsBase64: (file) => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                // reader.result is "data:mime/type;base64,THE_BASE64_STRING"
                // We need to extract THE_BASE64_STRING
                const base64String = reader.result.split(',')[1];
                if (typeof base64String === 'string') {
                    resolve(base64String);
                } else {
                    // This case should ideally not happen with readAsDataURL if file is valid
                    logger.warn("[API _readFileAsBase64] Failed to extract base64 string from FileReader result for file:", file.name);
                    reject(new Error("Failed to extract base64 string from FileReader result."));
                }
            };
            reader.onerror = (error) => {
                logger.error("[API _readFileAsBase64] FileReader error:", error);
                reject(error);
            };
            reader.readAsDataURL(file);
        });
    },

    _getRequestHeaders: (isJson = true) => {
        const headers = {
            'Accept': isJson ? 'application/json' : 'text/html',
        };
        if (isJson) {
            headers['Content-Type'] = 'application/json';
        }

        // Add Auth token if available
        const token = TB.state.get('user.token');
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        return headers;
    },

    /**
     * Makes a request to the backend.
     * @param {string} moduleName - The backend module name, OR a full path starting with '/' for special routes.
     * @param {string} functionName - The backend function name (ignored if moduleName is a full path).
     * @param {object|string|FormData} [payload=null] - Data to send. If string, used as query params for GET/POST-URL.
     * @param {string} [method='POST'] - HTTP method.
     * @param {string} [useTauri='auto'] - 'auto', 'force', or 'never'.
     * @param {boolean} [isSpecialAuthRoute=false] - Indicates if this is a special auth route that might have different token handling.
     * @returns {Promise<Result>} - A promise resolving to a Result object.
     */
    request: async (moduleName, functionName, payload = null, method = 'POST', useTauri = 'never', isSpecialAuthRoute = false, retryCount = 0) => {
        let command = `${moduleName}.${functionName}`; // For Tauri invoke
        let isFullPath = moduleName.startsWith('/');
        const isFormDataPayload = payload instanceof FormData;

        // _getRequestHeaders() likely sets 'Accept' and 'Authorization'.
        // It might also set a default 'Content-Type', which we'll need to override or remove for FormData.
        const baseHeaders = Api._getRequestHeaders();
        const options = {
            method,
            headers: { ...baseHeaders }, // Start with base headers
        };

        if (!isFullPath && (useTauri === 'auto' || useTauri === 'force') && env.isTauri()) {
            try {
                logger.debug(`[API] Attempting Tauri invoke: ${command}`, isFormDataPayload ? "[FormData]" : payload);
                let tauriInvokePayloadArgs; // This will be the direct argument(s) for invoke

                if (isFormDataPayload) {
                    // The Rust parser produces a flat HashMap<String, serde_json::Value>.
                    // We need to construct an equivalent JS object.
                    // The original JS sent { form_data: { ... } }, so we'll stick to that outer structure
                    // and make the inner object compatible.
                    const compatibleFormDataMap = {};
                    for (const [key, value] of payload.entries()) {
                        if (value instanceof File) {
                            try {
                                compatibleFormDataMap[key] = {
                                    filename: value.name,
                                    content_type: value.type || 'application/octet-stream', // Provide a default
                                    content_base64: await Api._readFileAsBase64(value)
                                };
                                logger.debug(`[API] Tauri FormData: Processed file field '${key}': ${value.name}`);
                            } catch (fileReadError) {
                                logger.error(`[API] Error reading file ${value.name} (field: ${key}) for Tauri invoke:`, fileReadError);
                                // Decide how to handle: skip field, error out, or send null
                                // For now, let's send null as per Rust parser's error handling for decode
                                compatibleFormDataMap[key] = null;
                                // Optionally, could return an error immediately:
                                // return new Result(['tauri', 'file_error'], ToolBoxError.input_error, new ToolBoxResult(), new ToolBoxInfo(-1, `Error reading file ${value.name} for Tauri: ${fileReadError.message}`));
                            }
                        } else {
                            // Regular field, Rust parser expects a string value (which serde_json::json!() will handle)
                            compatibleFormDataMap[key] = value;
                            logger.debug(`[API] Tauri FormData: Processed text field '${key}': ${value}`);
                        }
                    }
                    // Assuming the Tauri command expects an object with a 'form_data' field
                    // which then contains our map.
                    tauriInvokePayloadArgs = { form_data: compatibleFormDataMap };

                } else if (payload !== null && payload !== undefined) {
                    tauriInvokePayloadArgs = {}; // Initialize as an empty object for kwargs
                    if (typeof payload === 'string') {
                         Api.parseQueryParams(payload, tauriInvokePayloadArgs);
                    } else { // Assumes payload is an object to be spread as kwargs
                        Object.assign(tauriInvokePayloadArgs, payload);
                    }
                } else { // payload is null or undefined
                    tauriInvokePayloadArgs = {}; // Send empty object as kwargs
                }

                logger.debug(`[API] Tauri invoke payload for ${command}:`, tauriInvokePayloadArgs);
                // Use @tauri-apps/api/core for Tauri 2.x
                const { invoke } = await import('@tauri-apps/api/core');
                const response = await invoke(command, tauriInvokePayloadArgs);
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
            if (moduleName.includes("IsValidSession") || moduleName.includes("validateSession") || moduleName.startsWith("/web/")){
                url = `${config.get('baseApiUrl').replace('/api', '')}${moduleName}`;
                if (method === "POST" && moduleName.includes("IsValidSession")){ // Note: case sensitive
                    method = "GET"; // Update method for options too
                    options.method = "GET";
                }
            } else {
                url = `${config.get('baseApiUrl')}${moduleName}`;
            }
            if (functionName && typeof functionName === 'string' && functionName.length > 0 && (method.toUpperCase() === 'GET' || method.toUpperCase() === 'DELETE')) {
                url += `?${functionName}`;
            } else if (functionName && typeof functionName === 'object' && (method.toUpperCase() === 'GET' || method.toUpperCase() === 'DELETE')) {
                 url += `?${new URLSearchParams(functionName).toString()}`;
            }
        } else {
            url = `${config.get('baseApiUrl')}/${moduleName}/${functionName}`;
        }

        if (method.toUpperCase() === 'GET' || method.toUpperCase() === 'DELETE') {
            if (payload && typeof payload === 'string' && !isFullPath) {
                url += `?${payload}`;
            } else if (payload && typeof payload === 'object' && !isFullPath) {
                url += `?${new URLSearchParams(payload).toString()}`;
            }
            // For GET/DELETE, payload is in URL, so options.body is not set.
            // Content-Type is typically not needed for GET/DELETE requests without a body.
            delete options.headers['Content-Type'];
        } else { // POST, PUT, PATCH
             if (isFormDataPayload) {
                // For FormData, DO NOT set Content-Type header manually.
                // The browser will do it correctly, including the boundary.
                // Remove any Content-Type that might have been set by _getRequestHeaders or previous logic.
                delete options.headers['Content-Type'];
                options.body = payload; // payload is already a FormData object
            } else if (payload && typeof payload === 'string' && payload.includes('=')) {
                options.headers['Content-Type'] = 'application/x-www-form-urlencoded';
                options.body = payload;
            } else if (payload && typeof payload === 'string' && method.toUpperCase() === 'POST' && !isFullPath) {
                 url += `?${payload}`;
                 delete options.headers['Content-Type']; // No body, so no content type for body
            }  else if (payload !== null && payload !== undefined) {
                options.headers['Content-Type'] = 'application/json';
                options.body = JSON.stringify(payload);
            } else {
                // If payload is null or undefined for POST/PUT/PATCH, options.body remains unset.
                // And we might not need a Content-Type, or the server might expect one (e.g., application/json for an empty object).
                // For safety, if a default was set by _getRequestHeaders, it might remain.
                // If no body, often Content-Type is omitted or can be 'application/json' if an empty {} is conventional.
                // If options.headers['Content-Type'] was already set (e.g. to application/json), leave it.
                // If not, and we want to be explicit for an empty body POST:
                // options.headers['Content-Type'] = 'application/json';
                // options.body = JSON.stringify({}); // Or let it be truly empty if the server handles it
                // For now, let's assume if payload is null, no body is intended, and Content-Type for body is not relevant.
                delete options.headers['Content-Type'];
            }
        }

        logger.debug(`[API] HTTP ${options.method} request to: ${url}`, isFormDataPayload ? "[FormData (details in network tab)]" : payload, "Headers:", options.headers);
        try {
            const response = await fetch(url, options);
            let responseData;
            const contentTypeHeader = response.headers.get("content-type");

            if (response.status === 204 || response.status === 205) { // No Content / Reset Content
                responseData = { success: true }; // Create a minimal success object
            } else if (contentTypeHeader && contentTypeHeader.includes("application/json")) {
                responseData = await response.json();
            } else {
                const textResponse = await response.text();
                if (textResponse) {
                    try {
                        responseData = JSON.parse(textResponse);
                    } catch (e) {
                         responseData = { success: response.ok, message: textResponse }; // Fallback to text message
                    }
                } else {
                    responseData = { success: response.ok }; // Empty response, but use status for success
                }
            }

            if (!response.ok) {
                // --- Session Refresh Logic ---
                if (response.status === 401 && retryCount === 0) {
                    logger.warn(`[API] Received 401 Unauthorized for ${url}. Attempting session refresh...`);
                    try {
                        await TB.user._refreshToken(); // This will update TB.state.get('user.token')
                        logger.info(`[API] Session refreshed. Retrying original request for ${url}.`);
                        // Retry the original request with the new token (retryCount = 1)
                        return Api.request(moduleName, functionName, payload, method, useTauri, isSpecialAuthRoute, 1);
                    } catch (refreshError) {
                        logger.error(`[API] Session refresh failed for ${url}.`, refreshError);
                        // If refresh fails, it means the user is logged out by _refreshToken.
                        // So, we just return the original 401 error.
                    }
                }
                // --- End Session Refresh Logic ---

                logger.error(`[API] HTTP error ${response.status} for ${url}:`, responseData);
                const errorPayload = (responseData && typeof responseData === 'object') ? responseData : {};
                return wrapApiResponse({
                    error: errorPayload.error || ToolBoxError.internal_error,
                    info: errorPayload.info || { exec_code: response.status, help_text: (errorPayload?.message || response.statusText || "HTTP Error") },
                    result: errorPayload.result || {}
                }, 'http');
            }
            logger.debug(`[API] HTTP success for ${url}:`, responseData);
            return wrapApiResponse(responseData, 'http');
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

