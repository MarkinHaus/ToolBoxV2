// tbjs/core/crypto.js
// Cryptographic utilities.
// Original: web/scripts/cryp.js
import {openDB} from 'idb';
import TB from '../index.js'; // Access TB.api, TB.config, TB.logger
const DEFAULT_TTL_MS = 3 * 24 * 60 * 60 * 1000; // 3 days

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

export function convertToPem(keyBuffer, type, is_base=false) {
    let typeString;
    if (type === 'public') {
        typeString = 'PUBLIC KEY';
    } else if (type === 'private') {
        typeString = 'PRIVATE KEY';
    } else {
        throw new Error('Invalid key type');
    }
    let base64Key
    if (!is_base) {
        base64Key = arrayBufferToBase64(keyBuffer);
    }else{
        base64Key = keyBuffer
    }

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


export async function storePublicKey(privateKeyBase64, username) {
    // Use a prefix for TB-specific keys to avoid collision and for easier management
    const keyName = `tb_pb_${username}`;
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

export async function retrievePublicKey(username) {
    const keyName = `tb_pb_${username}`;
    const privateKeyBase64 = localStorage.getItem(keyName);
    if (!privateKeyBase64) {
        (TB.logger || console).warn(`[Crypto] No private key found for ${username}`);
        return null; // Return null or throw an error, rather than "Invalid user name..." string
    }
    (TB.logger || console).debug(`[Crypto] Private key retrieved for ${username}`);
    return  convertToPem(privateKeyBase64, 'public', true);
}
async function getEncryptedKeyId(username, salt) {
    const key = await deriveKeyFromSaltedUsername(username, salt);
    const enc = new TextEncoder();
    const encrypted = await crypto.subtle.encrypt(
        { name: "AES-GCM", iv: salt.slice(0, 12) },
        key,
        enc.encode(username)
    );
    return `tb_pk_${toHex(encrypted)}`;
}
async function deriveKeyFromSaltedUsername(username, salt) {
    const enc = new TextEncoder();
    const keyMaterial = await crypto.subtle.importKey(
        "raw",
        enc.encode(username),
        { name: "PBKDF2" },
        false,
        ["deriveKey"]
    );
    return crypto.subtle.deriveKey(
        {
            name: "PBKDF2",
            salt,
            iterations: 150000,
            hash: "SHA-256",
        },
        keyMaterial,
        { name: "AES-GCM", length: 256 },
        false,
        ["encrypt", "decrypt"]
    );
}

export async function storePrivateKey(privateKeyBase64, username, ttlMs = DEFAULT_TTL_MS) {
    const salt = crypto.getRandomValues(new Uint8Array(16));
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const enc = new TextEncoder();
    const data = enc.encode(atob(privateKeyBase64));

    try {
        // First, remove all existing private keys for this user
        await removePrivateKeysForUser(username);

        const aesKey = await deriveKeyFromSaltedUsername(username, salt);
        const encrypted = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, aesKey, data);

        const keyId = await getEncryptedKeyId(username, salt);
        const db = await getDB();

        await db.put("keys", {
            encrypted: arrayBufferToBase64(encrypted),
            iv: arrayBufferToBase64(iv),
            salt: arrayBufferToBase64(salt),
            username: username, // Store username for easier identification
            createdAt: Date.now(),
            ttlMs
        }, keyId);

        (TB.logger || console).debug(`[Crypto] Encrypted key stored under ${keyId} for user ${username}`);
    } catch (e) {
        (TB.logger || console).error(`[Crypto] Failed to store encrypted key:`, e);
        throw new Error("Private key storage failed.");
    }
}

// Assuming getDB, DEFAULT_TTL_MS, getEncryptedKeyId, base64ToArrayBuffer,
// deriveKeyFromSaltedUsername, TB.logger, and crypto.subtle.decrypt are defined elsewhere.

export async function retrievePrivateKey(username) {
    try {
        const db = await getDB();
        let matchedKeyEntry = null; // Store the full entry of the matched key
        const keysToDelete = [];
        const potentialEntries = []; // To store entries that are not expired

        // --- Phase 1: Read all relevant data from DB ---
        let readTx = db.transaction("keys", "readonly");
        let store = readTx.objectStore("keys"); // Use objectStore() method

        let cursor = await store.openCursor();
        while (cursor) {
            const entry = cursor.value; // { salt, iv, encrypted, createdAt, ttlMs }
            const createdAt = entry.createdAt || 0;
            const ttlMs = entry.ttlMs || DEFAULT_TTL_MS;

            if (Date.now() - createdAt > ttlMs) {
                keysToDelete.push(cursor.key);
            } else {
                // Store the cursor's key and the entry's salt for later processing
                potentialEntries.push({
                    idbKey: cursor.key, // The actual key in IndexedDB
                    salt: entry.salt, // Needed to compute candidateKeyId
                    iv: entry.iv,     // Needed for decryption if this is the one
                    encrypted: entry.encrypted, // Needed for decryption
                    // No need to store the full entry if we store components
                });
            }
            cursor = await cursor.continue();
        }
        await readTx.done; // Wait for the readonly transaction to complete

        // --- Phase 2: Process potential entries (async crypto operations outside IDB transaction) ---
        for (const potential of potentialEntries) {
            const candidateKeyId = await getEncryptedKeyId(username, base64ToArrayBuffer(potential.salt));
            if (potential.idbKey === candidateKeyId) {
                matchedKeyEntry = potential; // We found our match
                break; // Stop searching
            }
        }

        // --- Phase 3: Delete expired keys (in a new readwrite transaction) ---
        if (keysToDelete.length > 0) {
            const deleteTx = db.transaction("keys", "readwrite");
            store = deleteTx.objectStore("keys");
            for (const key of keysToDelete) {
                (TB.logger || console).info(`[Crypto] Deleting expired key: ${key}`);
                await store.delete(key); // This await is fine, it's an IDB request
            }
            await deleteTx.done; // Wait for the delete transaction to complete
        }

        if (!matchedKeyEntry) {
            (TB.logger || console).warn(`[Crypto] No valid key found for ${username}`);
            return null;
        }

        // Now use matchedKeyEntry which contains salt, iv, encrypted
        const salt = base64ToArrayBuffer(matchedKeyEntry.salt);
        const iv = base64ToArrayBuffer(matchedKeyEntry.iv);
        const encrypted = base64ToArrayBuffer(matchedKeyEntry.encrypted);

        const aesKey = await deriveKeyFromSaltedUsername(username, salt);
        const decrypted = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, aesKey, encrypted);

        // If the decrypted data is meant to be a UTF-8 string that is then base64 encoded:
        const dec = new TextDecoder(); // Assumes decrypted data is UTF-8 text
        const decryptedString = dec.decode(decrypted);
        const keyBase64 = btoa(decryptedString); // btoa works on strings of single-byte characters

        // If the decrypted data is raw binary and needs to be base64 encoded:
        // function arrayBufferToBase64(buffer) {
        //     let binary = '';
        //     const bytes = new Uint8Array(buffer);
        //     for (let i = 0; i < bytes.byteLength; i++) {
        //         binary += String.fromCharCode(bytes[i]);
        //     }
        //     return btoa(binary);
        // }
        // const keyBase64 = arrayBufferToBase64(decrypted);


        (TB.logger || console).debug(`[Crypto] Private key retrieved for ${username}`);
        return keyBase64;

    } catch (e) {
        (TB.logger || console).error(`[Crypto] Key retrieval failed:`, e);
        // More specific error logging if possible
        if (e.name === 'TransactionInactiveError') {
            (TB.logger || console).error('[Crypto] TransactionInactiveError encountered. This usually means an await for a non-IDB operation happened inside an IDB transaction loop.');
        }
        return null;
    }
}


export async function removePrivateKey(username) {
    try {
        const db = await getDB();

        // Step 1: Collect all entries in a readonly transaction
        const readTransaction = db.transaction("keys", "readonly");
        const entries = [];

        for (let cursor = await readTransaction.store.openCursor(); cursor; cursor = await cursor.continue()) {
            entries.push({
                key: cursor.key,
                value: cursor.value
            });
        }

        await readTransaction.complete;

        // Step 2: Determine which keys belong to this user (outside of transaction)
        const keysToDelete = [];
        for (const entry of entries) {
            const salt = base64ToArrayBuffer(entry.value.salt);
            const candidateKeyId = await getEncryptedKeyId(username, salt);

            if (entry.key === candidateKeyId) {
                keysToDelete.push(entry.key);
            }
        }

        // Step 3: Delete the identified keys in a new transaction
        if (keysToDelete.length > 0) {
            const writeTransaction = db.transaction("keys", "readwrite");
            for (const keyId of keysToDelete) {
                await writeTransaction.store.delete(keyId);
            }
            await writeTransaction.complete;
        }

        (TB.logger || console).debug(`[Crypto] Removed ${keysToDelete.length} private key(s) for user ${username}`);
        return keysToDelete.length;
    } catch (e) {
        (TB.logger || console).error(`[Crypto] Failed to remove private key for ${username}:`, e);
        throw new Error("Private key removal failed.");
    }
}

async function removePrivateKeysForUser(username) {
    try {
        const db = await getDB();

        // Step 1: Collect all entries in a readonly transaction
        const readTransaction = db.transaction("keys", "readonly");
        const entries = [];

        for (let cursor = await readTransaction.store.openCursor(); cursor; cursor = await cursor.continue()) {
            entries.push({
                key: cursor.key,
                value: cursor.value
            });
        }

        await readTransaction.complete;

        // Step 2: Determine which keys belong to this user (outside of transaction)
        const keysToDelete = [];
        for (const entry of entries) {
            const salt = base64ToArrayBuffer(entry.value.salt);
            const candidateKeyId = await getEncryptedKeyId(username, salt);

            if (entry.key === candidateKeyId) {
                keysToDelete.push(entry.key);
            }
        }

        // Step 3: Delete the identified keys in a new transaction
        if (keysToDelete.length > 0) {
            const writeTransaction = db.transaction("keys", "readwrite");
            for (const keyId of keysToDelete) {
                await writeTransaction.store.delete(keyId);
            }
            await writeTransaction.complete;

            (TB.logger || console).debug(`[Crypto] Removed ${keysToDelete.length} existing private key(s) for user ${username} before storing new key`);
        }
    } catch (e) {
        (TB.logger || console).error(`[Crypto] Failed to remove existing private keys for ${username}:`, e);
        throw new Error("Failed to clean up existing private keys.");
    }
}

function toHex(buffer) {
    return Array.from(new Uint8Array(buffer)).map(b => b.toString(16).padStart(2, '0')).join('');
}

async function getDB() {
    return openDB("TBKeys", 1, {
        upgrade(db) {
            db.createObjectStore("keys");
        }
    });
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
            pk: await retrievePublicKey(username), // PEM format from generateAsymmetricKeys
            sing: sing, // Include original 'sing' parameter
            client_json: {challenge, origin: window.location.origin},
            raw_id: arrayBufferToBase64(credential.rawId),
            registration_credential: {
                id: credential.id, // This is the credential ID, base64url encoded by browser
                raw_id: credential.id, // rawId as base64
                type: credential.type, // "public-key"
                authenticator_attachment: credential.authenticatorAttachment,
                response: {
                    client_data_json: arrayBufferToBase64(credential.response.clientDataJSON),
                    attestation_object: arrayBufferToBase64(credential.response.attestationObject),
                    // authenticatorData: arrayBufferToBase64(credential.response.getAuthenticatorData()), // Part of attestationObject
                    // publicKey: arrayBufferToBase64(credential.response.getPublicKey()), // Part of attestationObject
                    // publicKeyAlgorithm: credential.response.getPublicKeyAlgorithm() // COSE alg ID
                },
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
const asBase64 = ab => btoa(String.fromCharCode(...new Uint8Array(ab)))
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
        // This function now just prepares the payload. The calling function (e.g. in user.js) will send it.
        return {
            signature: arrayBufferToBase64(assertion.response.signature),
            username: username,
            authentication_credential: {
                id: assertion.id,
                raw_id: assertion.id,
                response: {
                    client_data_json: asBase64(assertion.response.clientDataJSON),
                    signature: asBase64(assertion.response.signature),
                    authenticator_data: asBase64(assertion.response.authenticatorData),
                    user_handle: assertion.response.userHandle ? arrayBufferToBase64(assertion.response.userHandle) : null,
                },
                type: "public-key"
            }
        };

    } catch (error) {
        (TB.logger || console).error('[Crypto] WebAuthn authorization failed:', error);
        throw error;
    }
}
// --- END OF ADAPTED PASTE ---
