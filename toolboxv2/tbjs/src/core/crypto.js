// tbjs/core/crypto.js
// Cryptographic utilities.
// Original: web/scripts/cryp.js

// import { httpPostData } from './api.js'; // If TB.api is not ready/preferred. Better to use TB.api.
// import config from './config.js'; // For rpIdUrl logic
import TB from '../index.js'; // Access TB.api, TB.config once fully initialized


// const rpIdUrl_f = ()=> { // This logic should now come from TB.config.get('baseFileUrl') or similar
//     const baseUrl = TB.config.get('baseFileUrl') || window.location.origin;
//     const url = new URL(baseUrl);
//     if (url.hostname.includes("localhost")) {
//         return "localhost"; // WebAuthn rp.id is usually the domain, not 'localhost:port'
//     } else {
//         return url.hostname;
//     }
// }
// let rpIdUrl; // Will be initialized after TB.config

function getRpId() {
    if (!TB || !TB.config || !TB.config.get('baseFileUrl')) {
        // Fallback for early calls before full TB init, less reliable
        const hostname = window.location.hostname;
        return hostname.includes("localhost") ? "localhost" : hostname;
    }
    const baseUrl = TB.config.get('baseFileUrl'); // This should be the registrable domain part
    const url = new URL(baseUrl);
    return url.hostname; // e.g., "simplecore.app" or "localhost"
}


// ... (rest of your cryp.js content) ...
// Example change for httpPostData:
// Replace: httpPostData('CloudM.AuthManager', 'validate_persona', credential, errorCallback, sucessCallback);
// With: TB.api.request('CloudM.AuthManager', 'validate_persona', credential).then(sucessCallback).catch(errorCallback);
// Note: The success/error callbacks for TB.api.request will receive the Result object directly.

// Ensure all functions from original cryp.js are here.
// All crypto functions like arrayBufferToBase64, generateAsymmetricKeys, encryptAsymmetric, etc.

// Export functions
// export async function generateAsymmetricKeys() { ... }
// ... and so on for all exported functions from original cryp.js

// --- START OF PASTE FROM web/scripts/cryp.js ---
// Ensure imports within this pasted code are adapted.
// For example, httpPostData should use TB.api if TB is initialized.

// import {httpPostData} from "./httpSender.js"; // Will be TB.api.httpPostData or TB.api.request

// const rpIdUrl_f = ()=> { ... } // Replaced by getRpId()
// const rpIdUrl = rpIdUrl_f() // Replaced by getRpId() in functions

// console.log("[rpIdUrl]:", rpIdUrl) // Use TB.logger.debug

function arrayBufferToBase64(buffer) { /* ... */ }
function base64ToArrayBuffer(base64) { /* ... */ }
function strToArrayBuffer(str) { /* ... */ }
function arrayBufferToStr(arrayBuffer) { /* ... */ }
function strToBase64(str) { /* ... */ }
function hexStringToArrayBuffer(hexString) { /* ... */ }
function convertToPem(keyBuffer, type) { /* ... */ }

export async function generateAsymmetricKeys() { /* ... */ }
// async function encryptAsymmetric(text, publicKeyBase64) { /* ... */ } // If not exported, make it internal
export async function decryptAsymmetric(encryptedTextBase64, privateKeyBase64, convert=false) { /* ... */ }
export async function signMessage(privateKeyBase64, message) { /* ... */ }
export async function generateSymmetricKey() { /* ... */ }
// async function encryptSymmetric(data, password) { /* ... */ } // If not exported
export async function decryptSymmetric(encryptedData, password) { /* ... requires fix in original to return decryptedData */ }
export async function storePrivateKey(privateKey, username) { /* ... */ }
export async function retrievePrivateKey(username) { /* ... */ }

export async function registerUser(registrationData, sing, errorCallback, sucessCallback) {
    const rpId = getRpId();
    // ... rest of function, using rpId
    // Replace httpPostData with TB.api.request or TB.api.httpPostData
    // e.g., TB.api.httpPostData('CloudM.AuthManager', 'register_user_personal_key', newCredential)
    //      .then(sucessCallback).catch(errorCallback);
}

export async function authorisazeUser(rawId, challenge, username, errorCallback, sucessCallback) {
    const rpId = getRpId();
    // ... rest of function, using rpId
    // Replace httpPostData
}

async function sendLoginResponseToServer(credential, errorCallback, sucessCallback) {
    // Replace httpPostData
    // TB.api.httpPostData('CloudM.AuthManager', 'validate_persona', credential)
    //  .then(sucessCallback).catch(errorCallback);
}
async function sendRegistrationResponseToServer(credential, errorCallback, sucessCallback) {
    // Replace httpPostData
    // TB.api.httpPostData('CloudM.AuthManager', 'register_user_personal_key', credential)
    //  .then(sucessCallback).catch(errorCallback);
}
// --- END OF PASTE ---
