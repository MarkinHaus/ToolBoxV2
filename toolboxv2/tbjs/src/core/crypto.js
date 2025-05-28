// tbjs/core/crypto.js
// Cryptographic utilities.
// Original: web/scripts/cryp.js

import TB from '../index.js'; // Access TB.api, TB.config, TB.logger

function getRpId() {
    let rpId = "localhost"; // Default
    try {
        if (TB && TB.config && TB.config.get('baseAppUrl')) { // baseAppUrl should be your main application URL
            const baseUrl = TB.config.get('baseAppUrl');
            const url = new URL(baseUrl);
            // WebAuthn rp.id is usually the domain, not 'localhost:port'
            // For localhost development, it should just be "localhost"
            // For production, it's the eTLD+1 (e.g., "example.com", not "www.example.com")
            if (url.hostname.includes("localhost") || url.hostname === "127.0.0.1") {
                rpId = "localhost";
            } else {
                rpId = url.hostname; // This might need to be refined to be eTLD+1 for production
            }
        } else {
            // Fallback for early calls before full TB init or if baseAppUrl is not set
            const hostname = window.location.hostname;
            rpId = (hostname.includes("localhost") || hostname === "127.0.0.1") ? "localhost" : hostname;
        }
    } catch (e) {
        // TB might not be fully initialized, or TB.config might not be ready
        // Fallback to window.location based logic
        const hostname = window.location.hostname;
        rpId = (hostname.includes("localhost") || hostname === "127.0.0.1") ? "localhost" : hostname;
        if (TB && TB.logger) {
            TB.logger.warn(`[Crypto] getRpId fallback used: ${rpId}. Error: ${e.message}`);
        } else {
            console.warn(`[Crypto] getRpId fallback used: ${rpId}. Error: ${e.message}`);
        }
    }
    if (TB && TB.logger) TB.logger.debug(`[Crypto] RP ID determined as: ${rpId}`);
    else console.log(`[Crypto] RP ID determined as: ${rpId}`);
    return rpId;
}


// --- START OF PASTE FROM web/scripts/cryp.js (adapted) ---

export function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
}

export function base64ToArrayBuffer(base64) {
    const binaryString = window.atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}

export function strToArrayBuffer(str) {
    const encoder = new TextEncoder();
    return encoder.encode(str);
}

export function arrayBufferToStr(arrayBuffer) {
    const decoder = new TextDecoder(); // Standardmäßig 'utf-8'
    return decoder.decode(arrayBuffer);
}

export function strToBase64(str) {
    const utf8Str = new TextEncoder().encode(str);
    const asciiStr = Array.from(utf8Str).map(byte => String.fromCharCode(byte)).join('');
    return btoa(asciiStr);
}

export function hexStringToArrayBuffer(hexString) {
    if (hexString.length % 2 !== 0) {
        throw new Error("Invalid hexadecimal string length");
    }
    var arrayBuffer = new Uint8Array(hexString.length / 2);
    for (var i = 0; i < hexString.length; i += 2) {
        var byteValue = parseInt(hexString.substring(i, i + 2), 16);
        if (isNaN(byteValue)) {
            throw new Error("Invalid hexadecimal string");
        }
        arrayBuffer[i / 2] = byteValue;
    }
    return arrayBuffer;
}

export function convertToPem(keyBuffer, type) {
    let typeString;
    if (type === 'public') {
        typeString = 'PUBLIC KEY';
    } else if (type === 'private') {
        typeString = 'PRIVATE KEY';
    } else {
        throw new Error('Invalid key type');
    }
    const base64Key = arrayBufferToBase64(keyBuffer);
    const formattedKey = base64Key.match(/.{1,64}/g).join('\n');
    return `-----BEGIN ${typeString}-----\n${formattedKey}\n-----END ${typeString}-----\n`;
}

export async function generateAsymmetricKeys() {
    const keyPair = await window.crypto.subtle.generateKey(
        {
            name: "RSA-OAEP",
            modulusLength: 2048 * 3, // Consider 2048 or 3072 for broader compatibility/performance
            publicExponent: new Uint8Array([1, 0, 1]),
            hash: "SHA-512", // SHA-256 is often sufficient and more performant
        },
        true,
        ["encrypt", "decrypt"]
    );
    const publicKey = await window.crypto.subtle.exportKey("spki", keyPair.publicKey);
    const privateKey = await window.crypto.subtle.exportKey("pkcs8", keyPair.privateKey);
    return {
        publicKey: convertToPem(publicKey, 'public'),
        publicKey_base64: arrayBufferToBase64(publicKey),
        privateKey: convertToPem(privateKey, 'private'),
        privateKey_base64: arrayBufferToBase64(privateKey),
    };
}


export async function encryptAsymmetric(text, publicKeyBase64) {
    const publicKey = await window.crypto.subtle.importKey(
        "spki",
        base64ToArrayBuffer(publicKeyBase64),
        {
            name: "RSA-OAEP",
            hash: "SHA-512"
        },
        false,
        ["encrypt"]
    );

    const encrypted = await window.crypto.subtle.encrypt(
        { name: "RSA-OAEP" },
        publicKey,
        new TextEncoder().encode(text)
    );

    return arrayBufferToBase64(encrypted);
}

export async function decryptAsymmetric(encryptedTextBase64, privateKeyBase64, convert = false) {
    const privateKey = await window.crypto.subtle.importKey(
        "pkcs8",
        base64ToArrayBuffer(privateKeyBase64),
        { name: "RSA-OAEP", hash: "SHA-512" }, // Match hash with generation
        false,
        ["decrypt"]
    );
    let ciphertext = convert ? hexStringToArrayBuffer(encryptedTextBase64) : base64ToArrayBuffer(encryptedTextBase64);
    try {
        const decrypted = await window.crypto.subtle.decrypt({ name: "RSA-OAEP" }, privateKey, ciphertext);
        return new TextDecoder().decode(decrypted);
    } catch (error) {
        (TB.logger || console).error("[Crypto] Error decrypting:", error);
        throw error; // Re-throw to be handled by caller
    }
}

export async function signMessage(privateKeyBase64, message) {
    // For RSA-PSS, the key usage must include "sign".
    // If keys are generated with "encrypt/decrypt", they can't be used for RSA-PSS "sign".
    // We need to import the key with "sign" usage.
    // This assumes the PKCS8 key was generated with sign capability or is a general RSA key.
    let keyAlgorithm = { name: "RSA-PSS", hash: "SHA-512" };
    // Try to import with RSA-PSS first
    let privateKey;
    try {
        privateKey = await window.crypto.subtle.importKey(
            "pkcs8",
            base64ToArrayBuffer(privateKeyBase64),
            keyAlgorithm,
            false,
            ["sign"]
        );
    } catch (e) {
        // Fallback for keys possibly generated with different intentions (e.g. RSA-OAEP only)
        // This is tricky. Ideally, keys are generated for their intended purpose.
        // If the key was for RSA-OAEP, it might not be suitable for RSA-PSS signing directly.
        // For simplicity, we'll assume the provided key *can* be used for signing,
        // and if importKey fails, it's a fundamental issue.
        (TB.logger || console).error("[Crypto] Error importing private key for signing:", e);
        throw e;
    }

    const encodedMessage = strToArrayBuffer(message);
    const signature = await window.crypto.subtle.sign(
        { name: "RSA-PSS", saltLength: 32 }, // Salt length for SHA-512 is typically 64
        privateKey,
        encodedMessage
    );
    return arrayBufferToBase64(signature);
}


export async function generateSymmetricKey() {
    const key = await window.crypto.subtle.generateKey(
        { name: "AES-GCM", length: 256 },
        true,
        ["encrypt", "decrypt"]
    );
    const exportedKey = await window.crypto.subtle.exportKey("raw", key);
    return window.btoa(String.fromCharCode(...new Uint8Array(exportedKey))); // Base64 of raw key
}

// Not exported in original
// async function encryptSymmetric(data, password) { ... }

export async function decryptSymmetric(encryptedDataB64, password) {
    const enc = new TextEncoder();
    const dec = new TextDecoder();
    const keyMaterial = await window.crypto.subtle.importKey(
        "raw", enc.encode(password), { name: "PBKDF2" }, false, ["deriveKey"]
    );
    const key = await window.crypto.subtle.deriveKey(
        { name: "PBKDF2", salt: enc.encode("some-salt"), iterations: 100000, hash: "SHA-512" },
        keyMaterial, { name: "AES-GCM", length: 256 }, false, ["decrypt"]
    );

    // Assuming encryptedDataB64 is base64 encoded ciphertext
    const encryptedData = base64ToArrayBuffer(encryptedDataB64);

    try {
        // IV needs to be extracted from encryptedData or passed separately.
        // AES-GCM typically prepends or appends the IV to the ciphertext.
        // Assuming a 12-byte IV prepended. This is a common practice.
        // If your original encryptSymmetric used a fixed or randomly generated IV *not* stored with ciphertext,
        // this part will fail or need adjustment.
        // For the original 'window.crypto.getRandomValues(new Uint8Array(12))' in decrypt,
        // it means a *new* random IV was generated for decryption, which is incorrect.
        // The IV used for encryption MUST be used for decryption.

        // Let's assume the IV is the first 12 bytes of the encryptedData
        if (encryptedData.byteLength <= 12) throw new Error("Encrypted data too short to contain IV.");
        const iv = encryptedData.slice(0, 12);
        const ciphertext = encryptedData.slice(12);

        const decryptedDataBuffer = await window.crypto.subtle.decrypt(
            { name: "AES-GCM", iv: iv }, // Use the extracted IV
            key,
            ciphertext
        );
        return dec.decode(decryptedDataBuffer);
    } catch (e) {
        (TB.logger || console).error("[Crypto] Symmetric decryption failed:", e);
        // Original returned "invalid key", better to throw or return specific error object
        throw new Error("Symmetric decryption failed, possibly invalid key or tampered data.");
    }
}


export async function storePrivateKey(privateKeyBase64, username) {
    // Use a prefix for TB-specific keys to avoid collision and for easier management
    const keyName = `tb_pk_${username}`;
    // Instead of hashing, just use the username. Hashing here doesn't add much security for localStorage.
    // The private key itself should be handled securely by the browser's crypto API if imported,
    // or if stored as base64, it's "as is".
    try {
        localStorage.setItem(keyName, privateKeyBase64);
        (TB.logger || console).debug(`[Crypto] Private key stored for ${username}`);
    } catch (e) {
        (TB.logger || console).error(`[Crypto] Failed to store private key for ${username}:`, e);
        throw new Error("Failed to store private key in localStorage.");
    }
}

export async function retrievePrivateKey(username) {
    const keyName = `tb_pk_${username}`;
    const privateKeyBase64 = localStorage.getItem(keyName);
    if (!privateKeyBase64) {
        (TB.logger || console).warn(`[Crypto] No private key found for ${username}`);
        return null; // Return null or throw an error, rather than "Invalid user name..." string
    }
    (TB.logger || console).debug(`[Crypto] Private key retrieved for ${username}`);
    return privateKeyBase64;
}

/**
 * WebAuthn Registration
 * @param {object} registrationData - { challenge, userId, username }
 * @param {string} sing - Additional data to be signed or included (purpose unclear from original)
 * @returns {Promise<object>} Resolves with newCredential object for server, or throws error
 */
export async function registerWebAuthnCredential(registrationData, sing) {
    if (!TB || !TB.api) throw new Error("TB.api is not available for WebAuthn server communication.");

    const { challenge, userId, username } = registrationData;
    const rpId = getRpId();

    const publicKeyCredentialCreationOptions = {
        challenge: strToArrayBuffer(challenge), // Server challenge should be base64url decoded then strToArrayBuffer
        rp: { name: "SimpleCore", id: rpId, "ico": "/favicon.ico" }, // Customize rp.name
        user: {
            id: base64ToArrayBuffer(userId), // Server User ID should be base64url string
            name: username,
            displayName: username
        },
        pubKeyCredParams: [
            { type: "public-key", alg: -7 },  // ES256
            { type: "public-key", alg: -257 } // RS256 (common)
            // { type: "public-key", alg: -256 }, typo, should be -257 for RS256, -258 RS384, -259 RS512
            // { type: "public-key", alg: -512 } // This is not a standard COSE algorithm identifier. Perhaps meant -259 (PS512) or similar.
        ],
        timeout: 60000,
        authenticatorSelection: {
            // residentKey: "preferred", // discoverable credentials
            requireResidentKey: true, // For passkeys
            userVerification: "preferred"
        },
        // extensions: { credProps: true } // If you need to know if a credential is discoverable
    };

    try {
        const credential = await navigator.credentials.create({ publicKey: publicKeyCredentialCreationOptions });

        const newCredentialPayload = {
            userId: userId, // Pass back original userId to server
            username: username,
            sing: sing, // Include original 'sing' parameter
            registration_credential: {
                id: credential.id, // This is the credential ID, base64url encoded by browser
                rawId: arrayBufferToBase64(credential.rawId), // rawId as base64
                type: credential.type, // "public-key"
                response: {
                    clientDataJSON: arrayBufferToBase64(credential.response.clientDataJSON),
                    attestationObject: arrayBufferToBase64(credential.response.attestationObject),
                    // authenticatorData: arrayBufferToBase64(credential.response.getAuthenticatorData()), // Part of attestationObject
                    // publicKey: arrayBufferToBase64(credential.response.getPublicKey()), // Part of attestationObject
                    // publicKeyAlgorithm: credential.response.getPublicKeyAlgorithm() // COSE alg ID
                },
                authenticatorAttachment: credential.authenticatorAttachment, // "platform" or "cross-platform" or null
            }
            // The fields pk, pkAlgo from your original sendRegistrationResponseToServer seem redundant
            // if you are sending the full registration_credential structure, as the server
            // will parse the attestationObject to get the public key.
            // If your server *specifically* needs them pre-parsed, add them.
        };
        // This function now just prepares the payload. The calling function (e.g. in user.js) will send it.
        return newCredentialPayload;

    } catch (error) {
        (TB.logger || console).error('[Crypto] WebAuthn registration failed:', error);
        throw error;
    }
}

/**
 * WebAuthn Authorization/Login
 * @param {string} rawIdAsBase64 - The base64 encoded rawId of the credential to use.
 * @param {string} challenge - The server-provided challenge.
 * @param {string} username - The username attempting to log in.
 * @returns {Promise<object>} Resolves with credential object for server, or throws error
 */
export async function authorizeWebAuthnCredential(rawIdAsBase64, challenge, username) {
    if (!TB || !TB.api) throw new Error("TB.api is not available for WebAuthn server communication.");
    const rpId = getRpId();

    const publicKeyCredentialRequestOptions = {
        challenge: strToArrayBuffer(challenge), // Server challenge, base64url decoded then strToArrayBuffer
        rpId: rpId,
        allowCredentials: [{
            type: "public-key",
            id: base64ToArrayBuffer(rawIdAsBase64) // rawId from server, base64 decoded then to ArrayBuffer
        }],
        userVerification: "preferred",
        timeout: 60000,
    };

    try {
        const assertion = await navigator.credentials.get({ publicKey: publicKeyCredentialRequestOptions });

        const loginCredentialPayload = {
            username: username, // Pass username for server context
            authentication_credential: {
                id: assertion.id, // Credential ID, base64url encoded by browser
                rawId: arrayBufferToBase64(assertion.rawId),
                type: assertion.type,
                response: {
                    clientDataJSON: arrayBufferToBase64(assertion.response.clientDataJSON),
                    authenticatorData: arrayBufferToBase64(assertion.response.authenticatorData),
                    signature: arrayBufferToBase64(assertion.response.signature),
                    userHandle: assertion.response.userHandle ? arrayBufferToBase64(assertion.response.userHandle) : null,
                }
            }
        };
        // This function now just prepares the payload. The calling function (e.g. in user.js) will send it.
        return loginCredentialPayload;

    } catch (error) {
        (TB.logger || console).error('[Crypto] WebAuthn authorization failed:', error);
        throw error;
    }
}
// --- END OF ADAPTED PASTE ---
