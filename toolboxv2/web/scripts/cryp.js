import {httpPostData} from "./httpSender.js";

const rpIdUrl_f = ()=> {
    if (window.location.href.match("localhost")) {
        return "localhost"
    } else {
        return window.location.origin
    }
}
const rpIdUrl = rpIdUrl_f()

console.log("[rpIdUrl]:", rpIdUrl)

function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
}

function base64ToArrayBuffer(base64) {
    const binaryString = window.atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}

function strToArrayBuffer(str) {
    const encoder = new TextEncoder();
    return encoder.encode(str);
}

function arrayBufferToStr(arrayBuffer) {
    const decoder = new TextDecoder(); // Standardmäßig 'utf-8'
    return decoder.decode(arrayBuffer);
}

function strToBase64(str) {
    // Erstelle einen UTF-8-kodierten String
    const utf8Str = new TextEncoder().encode(str);

    // Konvertiere den UTF-8-kodierten String in einen ASCII-String
    const asciiStr = Array.from(utf8Str).map(byte => String.fromCharCode(byte)).join('');

    // Kodiere den ASCII-String in Base64
    return btoa(asciiStr);
}
function hexStringToArrayBuffer(hexString) {
    if (hexString.length % 2 !== 0) {
        throw "Ungültige Hexadezimalstring-Länge";
    }
    var arrayBuffer = new Uint8Array(hexString.length / 2);
    for (var i = 0; i < hexString.length; i += 2) {
        var byteValue = parseInt(hexString.substring(i, i + 2), 16);
        if (isNaN(byteValue)) {
            throw "Ungültiger Hexadezimalstring";
        }
        arrayBuffer[i / 2] = byteValue;
    }
    return arrayBuffer;
}

function convertToPem(keyBuffer, type) {
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
            modulusLength: 2048*3,
            publicExponent: new Uint8Array([1, 0, 1]),
            hash: "SHA-512",
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

export async function decryptAsymmetric(encryptedTextBase64, privateKeyBase64, convert=false) {

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

    let ciphertext
    if (convert){
        ciphertext = hexStringToArrayBuffer(encryptedTextBase64);
    }else{
        ciphertext = base64ToArrayBuffer(encryptedTextBase64)
    }

    try {
        const decrypted = await window.crypto.subtle.decrypt(
            { name: "RSA-OAEP" },
            privateKey,
            ciphertext
        );
        return new TextDecoder().decode(decrypted);
    } catch (error) {
        console.error("Fehler beim Entschlüsseln:", error);
        return encryptedTextBase64
    }

}

export async function signMessage(privateKeyBase64, message) {
    const privateKey = await window.crypto.subtle.importKey(
        "pkcs8",
        base64ToArrayBuffer(privateKeyBase64),
        {
            name: "RSA-PSS",
            hash: "SHA-512"
        },
        false,
        ["sign"]
    );
    const encodedMessage = strToArrayBuffer(message);
    return arrayBufferToBase64(await window.crypto.subtle.sign(
        {
            name: "RSA-PSS",
            saltLength: 32, // Die Länge des Salzes in Bytes
        },
        privateKey,
        encodedMessage
    ));
}

export async function generateSymmetricKey() {
    const key = await window.crypto.subtle.generateKey(
        { name: "AES-GCM", length: 256 },
        true,
        ["encrypt", "decrypt"]
    );

    const exportedKey = await window.crypto.subtle.exportKey("raw", key);
    return window.btoa(String.fromCharCode(...new Uint8Array(exportedKey)));
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

export async function decryptSymmetric(encryptedData, password) {
    const enc = new TextEncoder();
    const dec = new TextDecoder();
    const keyMaterial = await window.crypto.subtle.importKey(
        "raw", enc.encode(password), { name: "PBKDF2" }, false, ["deriveKey"]
    );

    const key = await window.crypto.subtle.deriveKey(
        { name: "PBKDF2", salt: enc.encode("some-salt"), iterations: 100000, hash: "SHA-512" },
        keyMaterial, { name: "AES-GCM", length: 256 }, false, ["decrypt"]
    );

    try {
        const decryptedData = await window.crypto.subtle.decrypt(
            { name: "AES-GCM", iv: window.crypto.getRandomValues(new Uint8Array(12)) },
            key, encryptedData
        );
    }catch (e){
        return "invalid key"
    }


return dec.decode(decryptedData);
}

export async function storePrivateKey(privateKey, username) {
    const key = await window.crypto.subtle.digest("SHA-256", new TextEncoder().encode(username+"PK"))
    localStorage.setItem(arrayBufferToBase64(key), privateKey);
}

export async function retrievePrivateKey(username) {
    const key = await window.crypto.subtle.digest("SHA-256", new TextEncoder().encode(username+"PK"))
    const encryptedKey = localStorage.getItem(arrayBufferToBase64(key));
    console.log("encryptedPrivateKey:", encryptedKey)
    if (!encryptedKey) return "Invalid user name device not registered";
    return encryptedKey
}



export async function registerUser(registrationData, sing, errorCallback, sucessCallback) {
    // Schritt 1: Anfrage an den Server senden, um die Registrierungsdaten zu erhalten
    // const registrationData = {
    //     challenge:"Y2hhbGxlbmdlUmFuZG9tU3R12yaW5n",
    //     userId:"asda123124155768jgh",
    //     username:"TestSimp"} //await fetch('/path/to/registration').then(response => response.json());

    // Schritt 2: Umwandeln der Herausforderung und der Benutzer-ID von Base64 in Uint8Array
    console.log("[registerUser registrationData]:", registrationData)
    const challenge = strToArrayBuffer(registrationData.challenge);
    const userId = base64ToArrayBuffer(registrationData.userId)
    // Schritt 3: PublicKeyCredentialCreationOptions für die Registrierung vorbereiten
    const publicKeyCredentialCreationOptions = {
        challenge: challenge,
        rp: {
            "name": "SimpleCore",
            "id": rpIdUrl, //"localhost",
            "ico": "/favicon.ico"
        },
        user: {
            id: userId,
            name: registrationData.username,
            displayName: registrationData.username
        },
        pubKeyCredParams: [
            {type: "public-key",alg: -7 }, // -7 steht für ES256
            {type: "public-key",alg: -256},
            {type: "public-key",alg: -512}
        ],
        timeout: 60000,
        // Weitere Optionen können hier hinzugefügt werden
        excludeCredentials: [],
        authenticatorSelection: {
            // ... andere Auswahlkriterien ...
            //residentKey: "preferred",
            //authenticatorAttachment: "platform",
            requireResidentKey: true, // Setzen Sie dies auf true, um einen residenten Schlüssel zu erfordern
            userVerification: "preferred" // Kann "required", "preferred" oder "discouraged" sein
        },
    };

    // Schritt 4: Registrierung mit WebAuthn durchführen

    const asBase64 = ab => btoa(String.fromCharCode(...new Uint8Array(ab)))

    try {
        await navigator.credentials.create({
            publicKey: publicKeyCredentialCreationOptions
        }).then((publicKeyCredential) => {
            const response = publicKeyCredential.response;

            // Access attestationObject ArrayBuffer
            const attestationObj = response.attestationObject;

            // Access client JSON
            const clientJSON = response.clientDataJSON;

            // Return authenticator data ArrayBuffer
            const authenticatorData = response.getAuthenticatorData();

            // Return public key ArrayBuffer
            const pk = response.getPublicKey();

            // Return public key algorithm identifier
            const pkAlgo = response.getPublicKeyAlgorithm();

            // Return public key algorithm identifier
            const rawId = arrayBufferToBase64(publicKeyCredential.rawId);

            // Return permissible transports array
            const transports = response.getTransports();
            //const attestation = response.assertion()

            console.log("[attestationObj]:", arrayBufferToBase64(attestationObj))
            console.log("[clientJSON]:", arrayBufferToBase64(clientJSON))
            console.log("[pk]:", arrayBufferToStr(pk))
            console.log("[pkAlgo]:", pkAlgo)
            console.log("[transports]:", transports)
            console.log("[authenticatorData]:", arrayBufferToStr(authenticatorData))
            //console.log("[authenticatorData]:", response.getClientExtensionResults())
            //console.log("[attestation]:", attestation)
            const newCredential = {
                userId: registrationData.userId,
                username: registrationData.username,
                pk:convertToPem(pk, 'public'),
                pkAlgo:pkAlgo,
                clientJson:arrayBufferToBase64(clientJSON),
                authenticatorData:arrayBufferToBase64(authenticatorData),
                rawId:rawId,
                sing,
                registration_credential: {
                     id: publicKeyCredential.id,
                     raw_id: asBase64(publicKeyCredential.rawId),
                     response: {
                         client_data_json: asBase64(clientJSON),
                         attestation_object: asBase64(attestationObj),
                     },
                     authenticator_attachment: "platform",
                     type: "public-key"
                }
            }
            sendRegistrationResponseToServer(newCredential, errorCallback, sucessCallback);
        });
        return true
        // Schritt 5: Registrierungsantwort an den Server senden
        //await sendRegistrationResponseToServer(newCredential);
    } catch (error) {
        console.error('Fehler bei der Registrierung:', error);
        return false
    }
}

export async function authorisazeUser(rawId, challenge, username, errorCallback, sucessCallback) {
    //const challenge = strToArrayBuffer(registrationData.challenge);
    const publicKey = {
        challenge: base64ToArrayBuffer(strToBase64(challenge)),
        rpId: rpIdUrl,
        allowCredentials: [{
            type: "public-key",
            id: base64ToArrayBuffer(rawId)
        }],
        userVerification: "preferred",
    }
    const asBase64 = ab => btoa(String.fromCharCode(...new Uint8Array(ab)))
    try {
        await navigator.credentials.get({ publicKey }).then((publicKeyCredential) => {
            const response = publicKeyCredential.response;

            // Access authenticator data ArrayBuffer
            const authenticatorData = response.authenticatorData;

            // Access client JSON
            const clientJSON = response.clientDataJSON;

            // Access signature ArrayBuffer
            const signature = response.signature;

            // Access userHandle ArrayBuffer
            const userHandle = response.userHandle;
            const userCredential = {
                signature: arrayBufferToBase64(signature),
                username: username,
                authentication_credential: {
                    id: publicKeyCredential.id,
                    raw_id: asBase64(publicKeyCredential.rawId),
                    response: {
                        client_data_json: asBase64(clientJSON),
                        signature: asBase64(signature),
                        authenticator_data: asBase64(authenticatorData),
                        user_handle: arrayBufferToBase64(userHandle),
                    },
                    type:"public-key"
                }
            }
            sendLoginResponseToServer(userCredential, errorCallback, sucessCallback);
        });
        return true
    } catch (error) {
        console.error('Fehler beim der einloggen:', error);
        return false
    }
}


async function sendLoginResponseToServer(credential, errorCallback, sucessCallback) {
    console.log("[credential]:", credential)
    httpPostData('CloudM.AuthManager',
        'validate_persona',
        credential, errorCallback, sucessCallback);
    // Implementieren Sie eine Funktion, um die Registrierungsantwort an Ihren Server zu senden register_user_personal_key
    // Sie müssen Teile von `credential` in ein Format umwandeln, das über HTTP gesendet werden kann
}
async function sendRegistrationResponseToServer(credential, errorCallback, sucessCallback) {
    console.log("[credential]:", credential)
    httpPostData('CloudM.AuthManager',
        'register_user_personal_key',
        credential, errorCallback, sucessCallback);
    // Implementieren Sie eine Funktion, um die Registrierungsantwort an Ihren Server zu senden register_user_personal_key
    // Sie müssen Teile von `credential` in ein Format umwandeln, das über HTTP gesendet werden kann
}

// Generieren von Schlüsseln
//generateAsymmetricKeys().then(keys => {
//    console.log("Public Key:", keys.publicKey);
//    console.log("Private Key:", keys.privateKey);
//
//    // Verschlüsseln eines Textes
//    const text = "Hello, World!";
//    encryptAsymmetric(text, keys.publicKey).then(encrypted => {
//        console.log("Encrypted:", encrypted);
//
//        // Entschlüsseln des Textes
//        decryptAsymmetric(encrypted, keys.privateKey).then(decrypted => {
//            console.log("Decrypted:", decrypted);
//        });
//    });
//});
//
