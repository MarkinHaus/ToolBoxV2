// tbjs/core/cryptoV2.js
// Simplified Crypto Module - WebAuthn Only (No Custom Crypto)
// Version: 2.0.0

import TB from '../index.js';

/**
 * Get RP ID for WebAuthn
 * @returns {string} RP ID (e.g., "localhost" or "example.com")
 */
function getRpId() {
    try {
        const hostname = window.location.hostname;
        return (hostname.includes("localhost") || hostname === "127.0.0.1") ? "localhost" : hostname;
    } catch (e) {
        TB.logger?.warn(`[CryptoV2] getRpId error: ${e.message}`);
        return "localhost";
    }
}

/**
 * Get origin for WebAuthn
 * @returns {string} Origin (e.g., "http://localhost:8080")
 */
function getOrigin() {
    return window.location.origin;
}

// =================== Base64 Utilities ===================

/**
 * Convert ArrayBuffer to Base64
 * @param {ArrayBuffer} buffer
 * @returns {string} Base64 string
 */
export function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
}

/**
 * Convert Base64 to ArrayBuffer
 * @param {string} base64
 * @returns {ArrayBuffer}
 */
export function base64ToArrayBuffer(base64) {
    const binaryString = window.atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}

/**
 * Convert ArrayBuffer to Base64URL (WebAuthn standard)
 * @param {ArrayBuffer} buffer
 * @returns {string} Base64URL string
 */
export function arrayBufferToBase64URL(buffer) {
    const base64 = arrayBufferToBase64(buffer);
    return base64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
}

/**
 * Convert Base64URL to ArrayBuffer
 * @param {string} base64url
 * @returns {ArrayBuffer}
 */
export function base64URLToArrayBuffer(base64url) {
    // Add padding
    let base64 = base64url.replace(/-/g, '+').replace(/_/g, '/');
    while (base64.length % 4) {
        base64 += '=';
    }
    return base64ToArrayBuffer(base64);
}

// =================== WebAuthn Registration ===================

/**
 * Register a new WebAuthn credential (Passkey)
 * @param {Object} options - WebAuthn registration options from server
 * @returns {Promise<Object>} Credential response for server
 */
export async function registerWebAuthnCredential(options) {
    try {
        TB.logger?.info('[CryptoV2] Starting WebAuthn registration');
        
        // Convert base64url strings to ArrayBuffers
        const publicKeyOptions = {
            challenge: base64URLToArrayBuffer(options.challenge),
            rp: options.rp,
            user: {
                id: base64URLToArrayBuffer(options.user.id),
                name: options.user.name,
                displayName: options.user.displayName
            },
            pubKeyCredParams: options.pubKeyCredParams,
            timeout: options.timeout,
            authenticatorSelection: options.authenticatorSelection
        };
        
        // Call WebAuthn API
        const credential = await navigator.credentials.create({
            publicKey: publicKeyOptions
        });
        
        if (!credential) {
            throw new Error('No credential returned from WebAuthn API');
        }
        
        // Convert response to format expected by server
        const response = {
            id: credential.id,
            rawId: arrayBufferToBase64URL(credential.rawId),
            type: credential.type,
            response: {
                clientDataJSON: arrayBufferToBase64URL(credential.response.clientDataJSON),
                attestationObject: arrayBufferToBase64URL(credential.response.attestationObject),
                transports: credential.response.getTransports ? credential.response.getTransports() : []
            }
        };
        
        TB.logger?.info('[CryptoV2] WebAuthn registration successful');
        return response;
        
    } catch (error) {
        TB.logger?.error('[CryptoV2] WebAuthn registration failed:', error);
        throw error;
    }
}

// =================== WebAuthn Authentication ===================

/**
 * Authenticate with WebAuthn credential (Passkey)
 * @param {Object} options - WebAuthn authentication options from server
 * @returns {Promise<Object>} Assertion response for server
 */
export async function authenticateWebAuthn(options) {
    try {
        TB.logger?.info('[CryptoV2] Starting WebAuthn authentication');
        
        // Convert base64url strings to ArrayBuffers
        const publicKeyOptions = {
            challenge: base64URLToArrayBuffer(options.challenge),
            timeout: options.timeout,
            rpId: options.rpId,
            allowCredentials: options.allowCredentials.map(cred => ({
                id: base64URLToArrayBuffer(cred.id),
                type: cred.type,
                transports: cred.transports
            })),
            userVerification: options.userVerification
        };
        
        // Call WebAuthn API
        const assertion = await navigator.credentials.get({
            publicKey: publicKeyOptions
        });
        
        if (!assertion) {
            throw new Error('No assertion returned from WebAuthn API');
        }
        
        // Convert response to format expected by server
        const response = {
            id: assertion.id,
            rawId: arrayBufferToBase64URL(assertion.rawId),
            type: assertion.type,
            response: {
                clientDataJSON: arrayBufferToBase64URL(assertion.response.clientDataJSON),
                authenticatorData: arrayBufferToBase64URL(assertion.response.authenticatorData),
                signature: arrayBufferToBase64URL(assertion.response.signature),
                userHandle: assertion.response.userHandle ? arrayBufferToBase64URL(assertion.response.userHandle) : null
            }
        };
        
        TB.logger?.info('[CryptoV2] WebAuthn authentication successful');
        return response;
        
    } catch (error) {
        TB.logger?.error('[CryptoV2] WebAuthn authentication failed:', error);
        throw error;
    }
}

// =================== WebAuthn Availability Check ===================

/**
 * Check if WebAuthn is available in this browser
 * @returns {boolean} True if WebAuthn is supported
 */
export function isWebAuthnAvailable() {
    return !!(window.PublicKeyCredential && navigator.credentials && navigator.credentials.create);
}

/**
 * Check if platform authenticator (e.g., Touch ID, Face ID, Windows Hello) is available
 * @returns {Promise<boolean>} True if platform authenticator is available
 */
export async function isPlatformAuthenticatorAvailable() {
    if (!isWebAuthnAvailable()) {
        return false;
    }
    
    try {
        return await window.PublicKeyCredential.isUserVerifyingPlatformAuthenticatorAvailable();
    } catch (error) {
        TB.logger?.warn('[CryptoV2] Error checking platform authenticator:', error);
        return false;
    }
}

