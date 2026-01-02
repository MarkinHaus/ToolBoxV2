// tbjs/src/core/__tests__/crypto.test.js
// Tests für das Kryptographie/WebAuthn Modul

// Mock TB before importing crypto functions
jest.mock('../index.js', () => ({
    logger: {
        log: jest.fn(),
        debug: jest.fn(),
        info: jest.fn(),
        warn: jest.fn(),
        error: jest.fn(),
    },
}));

import {
    arrayBufferToBase64,
    base64ToArrayBuffer,
    arrayBufferToBase64URL,
    base64URLToArrayBuffer,
    isWebAuthnAvailable,
    isPlatformAuthenticatorAvailable,
    registerWebAuthnCredential,
    authenticateWebAuthn,
} from '../crypto.js';
import TB from '../index.js';

describe('Crypto Module', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    describe('Base64 Utilities', () => {
        describe('arrayBufferToBase64', () => {
            it('should convert ArrayBuffer to Base64 string', () => {
                const buffer = new Uint8Array([72, 101, 108, 108, 111]).buffer; // "Hello"
                const result = arrayBufferToBase64(buffer);

                expect(result).toBe('SGVsbG8=');
            });

            it('should handle empty ArrayBuffer', () => {
                const buffer = new ArrayBuffer(0);
                const result = arrayBufferToBase64(buffer);

                expect(result).toBe('');
            });

            it('should handle binary data', () => {
                const buffer = new Uint8Array([0, 255, 128, 64]).buffer;
                const result = arrayBufferToBase64(buffer);

                expect(typeof result).toBe('string');
                expect(result.length).toBeGreaterThan(0);
            });
        });

        describe('base64ToArrayBuffer', () => {
            it('should convert Base64 string to ArrayBuffer', () => {
                const base64 = 'SGVsbG8='; // "Hello"
                const result = base64ToArrayBuffer(base64);

                const bytes = new Uint8Array(result);
                expect(Array.from(bytes)).toEqual([72, 101, 108, 108, 111]);
            });

            it('should handle empty string', () => {
                const result = base64ToArrayBuffer('');

                expect(result.byteLength).toBe(0);
            });

            it('should be inverse of arrayBufferToBase64', () => {
                const original = new Uint8Array([1, 2, 3, 4, 5]).buffer;
                const base64 = arrayBufferToBase64(original);
                const restored = base64ToArrayBuffer(base64);

                expect(new Uint8Array(restored)).toEqual(new Uint8Array(original));
            });
        });

        describe('arrayBufferToBase64URL', () => {
            it('should convert ArrayBuffer to Base64URL string', () => {
                const buffer = new Uint8Array([72, 101, 108, 108, 111]).buffer;
                const result = arrayBufferToBase64URL(buffer);

                // Base64URL should not contain +, /, or =
                expect(result).not.toContain('+');
                expect(result).not.toContain('/');
                expect(result).not.toContain('=');
            });

            it('should replace + with - and / with _', () => {
                // Create buffer that would produce + and / in standard Base64
                const buffer = new Uint8Array([251, 255, 254]).buffer;
                const base64 = arrayBufferToBase64(buffer);
                const base64url = arrayBufferToBase64URL(buffer);

                if (base64.includes('+')) {
                    expect(base64url).toContain('-');
                }
                if (base64.includes('/')) {
                    expect(base64url).toContain('_');
                }
            });
        });

        describe('base64URLToArrayBuffer', () => {
            it('should convert Base64URL string to ArrayBuffer', () => {
                const base64url = 'SGVsbG8'; // "Hello" without padding
                const result = base64URLToArrayBuffer(base64url);

                const bytes = new Uint8Array(result);
                expect(Array.from(bytes)).toEqual([72, 101, 108, 108, 111]);
            });

            it('should handle Base64URL with - and _', () => {
                // First create a known buffer, convert to base64url, then back
                const original = new Uint8Array([251, 255, 254]).buffer;
                const base64url = arrayBufferToBase64URL(original);
                const restored = base64URLToArrayBuffer(base64url);

                expect(new Uint8Array(restored)).toEqual(new Uint8Array(original));
            });

            it('should be inverse of arrayBufferToBase64URL', () => {
                const original = new Uint8Array([10, 20, 30, 40, 50]).buffer;
                const base64url = arrayBufferToBase64URL(original);
                const restored = base64URLToArrayBuffer(base64url);

                expect(new Uint8Array(restored)).toEqual(new Uint8Array(original));
            });
        });
    });

    describe('WebAuthn Availability', () => {
        describe('isWebAuthnAvailable', () => {
            it('should return true when WebAuthn APIs are available', () => {
                // Mock WebAuthn APIs
                global.window.PublicKeyCredential = jest.fn();
                global.navigator.credentials = {
                    create: jest.fn(),
                    get: jest.fn(),
                };

                expect(isWebAuthnAvailable()).toBe(true);
            });

            it('should return false when PublicKeyCredential is not available', () => {
                const original = global.window.PublicKeyCredential;
                delete global.window.PublicKeyCredential;

                expect(isWebAuthnAvailable()).toBe(false);

                global.window.PublicKeyCredential = original;
            });
        });

        describe('isPlatformAuthenticatorAvailable', () => {
            it('should return false when WebAuthn is not available', async () => {
                const original = global.window.PublicKeyCredential;
                delete global.window.PublicKeyCredential;

                const result = await isPlatformAuthenticatorAvailable();

                expect(result).toBe(false);

                global.window.PublicKeyCredential = original;
            });

            it('should call isUserVerifyingPlatformAuthenticatorAvailable', async () => {
                const mockCheck = jest.fn().mockResolvedValue(true);
                global.window.PublicKeyCredential = {
                    isUserVerifyingPlatformAuthenticatorAvailable: mockCheck
                };
                global.navigator.credentials = { create: jest.fn(), get: jest.fn() };

                const result = await isPlatformAuthenticatorAvailable();

                expect(mockCheck).toHaveBeenCalled();
                expect(result).toBe(true);
            });

            it('should return false and log warning on error', async () => {
                global.window.PublicKeyCredential = {
                    isUserVerifyingPlatformAuthenticatorAvailable: jest.fn().mockRejectedValue(new Error('Test'))
                };
                global.navigator.credentials = { create: jest.fn(), get: jest.fn() };

                const result = await isPlatformAuthenticatorAvailable();

                expect(result).toBe(false);
                expect(TB.logger.warn).toHaveBeenCalled();
            });
        });
    });

    describe('WebAuthn Registration', () => {
        const mockRegistrationOptions = {
            challenge: 'dGVzdC1jaGFsbGVuZ2U', // base64url encoded
            rp: { name: 'Test RP', id: 'localhost' },
            user: {
                id: 'dXNlci1pZA', // base64url encoded
                name: 'test@example.com',
                displayName: 'Test User'
            },
            pubKeyCredParams: [{ type: 'public-key', alg: -7 }],
            timeout: 60000,
            authenticatorSelection: { userVerification: 'preferred' }
        };

        beforeEach(() => {
            global.window.PublicKeyCredential = jest.fn();
            global.navigator.credentials = {
                create: jest.fn(),
                get: jest.fn(),
            };
        });

        it('should call navigator.credentials.create with correct options', async () => {
            const mockCredential = {
                id: 'credential-id',
                rawId: new ArrayBuffer(16),
                type: 'public-key',
                response: {
                    clientDataJSON: new ArrayBuffer(100),
                    attestationObject: new ArrayBuffer(200),
                    getTransports: () => ['internal']
                }
            };
            global.navigator.credentials.create.mockResolvedValue(mockCredential);

            await registerWebAuthnCredential(mockRegistrationOptions);

            expect(global.navigator.credentials.create).toHaveBeenCalledWith({
                publicKey: expect.objectContaining({
                    challenge: expect.any(ArrayBuffer),
                    rp: mockRegistrationOptions.rp,
                    user: expect.objectContaining({
                        name: 'test@example.com',
                        displayName: 'Test User'
                    })
                })
            });
        });

        it('should return formatted credential response', async () => {
            const mockCredential = {
                id: 'credential-id',
                rawId: new ArrayBuffer(16),
                type: 'public-key',
                response: {
                    clientDataJSON: new ArrayBuffer(100),
                    attestationObject: new ArrayBuffer(200),
                    getTransports: () => ['internal', 'usb']
                }
            };
            global.navigator.credentials.create.mockResolvedValue(mockCredential);

            const result = await registerWebAuthnCredential(mockRegistrationOptions);

            expect(result).toHaveProperty('id', 'credential-id');
            expect(result).toHaveProperty('rawId');
            expect(result).toHaveProperty('type', 'public-key');
            expect(result.response).toHaveProperty('clientDataJSON');
            expect(result.response).toHaveProperty('attestationObject');
            expect(result.response.transports).toEqual(['internal', 'usb']);
        });

        it('should throw error when no credential returned', async () => {
            global.navigator.credentials.create.mockResolvedValue(null);

            await expect(registerWebAuthnCredential(mockRegistrationOptions))
                .rejects.toThrow('No credential returned');
        });

        it('should log error and rethrow on failure', async () => {
            const error = new Error('User cancelled');
            global.navigator.credentials.create.mockRejectedValue(error);

            await expect(registerWebAuthnCredential(mockRegistrationOptions))
                .rejects.toThrow('User cancelled');
            expect(TB.logger.error).toHaveBeenCalled();
        });
    });

    describe('WebAuthn Authentication', () => {
        const mockAuthOptions = {
            challenge: 'YXV0aC1jaGFsbGVuZ2U', // base64url encoded
            timeout: 60000,
            rpId: 'localhost',
            allowCredentials: [
                { id: 'Y3JlZC1pZA', type: 'public-key', transports: ['internal'] }
            ],
            userVerification: 'preferred'
        };

        beforeEach(() => {
            global.window.PublicKeyCredential = jest.fn();
            global.navigator.credentials = {
                create: jest.fn(),
                get: jest.fn(),
            };
        });

        it('should call navigator.credentials.get with correct options', async () => {
            const mockAssertion = {
                id: 'assertion-id',
                rawId: new ArrayBuffer(16),
                type: 'public-key',
                response: {
                    clientDataJSON: new ArrayBuffer(100),
                    authenticatorData: new ArrayBuffer(50),
                    signature: new ArrayBuffer(64),
                    userHandle: new ArrayBuffer(8)
                }
            };
            global.navigator.credentials.get.mockResolvedValue(mockAssertion);

            await authenticateWebAuthn(mockAuthOptions);

            expect(global.navigator.credentials.get).toHaveBeenCalledWith({
                publicKey: expect.objectContaining({
                    challenge: expect.any(ArrayBuffer),
                    rpId: 'localhost',
                    userVerification: 'preferred'
                })
            });
        });

        it('should return formatted assertion response', async () => {
            const mockAssertion = {
                id: 'assertion-id',
                rawId: new ArrayBuffer(16),
                type: 'public-key',
                response: {
                    clientDataJSON: new ArrayBuffer(100),
                    authenticatorData: new ArrayBuffer(50),
                    signature: new ArrayBuffer(64),
                    userHandle: new ArrayBuffer(8)
                }
            };
            global.navigator.credentials.get.mockResolvedValue(mockAssertion);

            const result = await authenticateWebAuthn(mockAuthOptions);

            expect(result).toHaveProperty('id', 'assertion-id');
            expect(result).toHaveProperty('rawId');
            expect(result).toHaveProperty('type', 'public-key');
            expect(result.response).toHaveProperty('clientDataJSON');
            expect(result.response).toHaveProperty('authenticatorData');
            expect(result.response).toHaveProperty('signature');
        });

        it('should handle null userHandle', async () => {
            const mockAssertion = {
                id: 'assertion-id',
                rawId: new ArrayBuffer(16),
                type: 'public-key',
                response: {
                    clientDataJSON: new ArrayBuffer(100),
                    authenticatorData: new ArrayBuffer(50),
                    signature: new ArrayBuffer(64),
                    userHandle: null
                }
            };
            global.navigator.credentials.get.mockResolvedValue(mockAssertion);

            const result = await authenticateWebAuthn(mockAuthOptions);

            expect(result.response.userHandle).toBeNull();
        });

        it('should throw error when no assertion returned', async () => {
            global.navigator.credentials.get.mockResolvedValue(null);

            await expect(authenticateWebAuthn(mockAuthOptions))
                .rejects.toThrow('No assertion returned');
        });
    });
});

