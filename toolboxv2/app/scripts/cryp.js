async function generateAsymmetricKeys() {
    const keyPair = await window.crypto.subtle.generateKey(
        {
            name: "RSA-OAEP",
            modulusLength: 2048,
            publicExponent: new Uint8Array([1, 0, 1]),
            hash: "SHA-512",
        },
        true,
        ["encrypt", "decrypt"]
    );

    const publicKey = await window.crypto.subtle.exportKey("spki", keyPair.publicKey);
    const privateKey = await window.crypto.subtle.exportKey("pkcs8", keyPair.privateKey);

    return {
        publicKey: arrayBufferToBase64(publicKey),
        privateKey: arrayBufferToBase64(privateKey)
    };
}

function arrayBufferToBase64(buffer) {
    return btoa(String.fromCharCode.apply(null, new Uint8Array(buffer)));
}

async function encryptAsymmetric(text, publicKeyBase64) {
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

function base64ToArrayBuffer(base64) {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}

async function decryptAsymmetric(encryptedTextBase64, privateKeyBase64) {
    const privateKey = await window.crypto.subtle.importKey(
        "pkcs8",
        base64ToArrayBuffer(privateKeyBase64),
        {
            name: "RSA-OAEP",
            hash: "SHA-512"
        },
        false,
        ["decrypt"]
    );

    const decrypted = await window.crypto.subtle.decrypt(
        { name: "RSA-OAEP" },
        privateKey,
        base64ToArrayBuffer(encryptedTextBase64)
    );

    return new TextDecoder().decode(decrypted);
}

async function encryptSymmetric(data, password) {
    const enc = new TextEncoder();
    const keyMaterial = await window.crypto.subtle.importKey(
        "raw", enc.encode(password), { name: "PBKDF2" }, false, ["deriveKey"]
    );

    const key = await window.crypto.subtle.deriveKey(
        { name: "PBKDF2", salt: enc.encode("some-salt"), iterations: 100000, hash: "SHA-512" },
        keyMaterial, { name: "AES-GCM", length: 256 }, false, ["encrypt"]
    );

    const encryptedData = await window.crypto.subtle.encrypt(
        { name: "AES-GCM", iv: window.crypto.getRandomValues(new Uint8Array(12)) },
        key, enc.encode(data)
    );

    return btoa(String.fromCharCode(...new Uint8Array(encryptedData)));
}

async function decryptSymmetric(encryptedData, password) {
    const enc = new TextEncoder();
    const dec = new TextDecoder();
    const keyMaterial = await window.crypto.subtle.importKey(
        "raw", enc.encode(password), { name: "PBKDF2" }, false, ["deriveKey"]
    );

    const key = await window.crypto.subtle.deriveKey(
        { name: "PBKDF2", salt: enc.encode("some-salt"), iterations: 100000, hash: "SHA-512" },
        keyMaterial, { name: "AES-GCM", length: 256 }, false, ["decrypt"]
    );

    const decryptedData = await window.crypto.subtle.decrypt(
        { name: "AES-GCM", iv: window.crypto.getRandomValues(new Uint8Array(12)) },
        key, base64ToArrayBuffer(encryptedData)
    );

    return dec.decode(decryptedData);
}

async function storePrivateKey(privateKey, deviceID) {
    const encryptedKey = await encryptSymmetric(privateKey, deviceID);
    localStorage.setItem('encryptedPrivateKey', encryptedKey);
}

async function retrievePrivateKey(deviceID) {
    const encryptedKey = localStorage.getItem('encryptedPrivateKey');
    if (!encryptedKey) return null;
    return await decryptSymmetric(encryptedKey, deviceID);
}


// Generieren von Schlüsseln
generateAsymmetricKeys().then(keys => {
    console.log("Public Key:", keys.publicKey);
    console.log("Private Key:", keys.privateKey);

    // Verschlüsseln eines Textes
    const text = "Hello, World!";
    encryptAsymmetric(text, keys.publicKey).then(encrypted => {
        console.log("Encrypted:", encrypted);

        // Entschlüsseln des Textes
        decryptAsymmetric(encrypted, keys.privateKey).then(decrypted => {
            console.log("Decrypted:", decrypted);
        });
    });
});
