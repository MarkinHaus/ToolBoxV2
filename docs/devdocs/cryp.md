# Crypto Utilities (`utils/security/cryp.py`)

> **File:** `toolboxv2/utils/security/cryp.py` (~469+ Zeilen)
> Verschlüsselung, Signatur, Hashing via `Code`-Klasse.

## Why This Matters

Die `Code`-Klasse ist die **einzige** Verschlüsselungs-Schnittstelle in ToolBoxV2. Sie wird verwendet von:
- Session-Management (Cookie-Signing mit HMAC)
- FileHandler (Config-Verschlüsselung mit AES)
- CloudM Auth (RSA-Signaturen, JWT)
- BlobStorage (verschlüsselte Blob-Übertragung)

Wenn du irgendwo in ToolBoxV2 mit Verschlüsselung zu tun hast, geht es durch `Code`.

## Overview

Die `Code`-Klasse bietet statische Methoden für:

| Bereich | Algorithmen |
|---------|------------|
| **Symmetrisch** | Fernet (AES-CBC + HMAC-SHA256) |
| **Asymmetrisch** | RSA (6144-bit = 3×2048), OAEP Padding, SHA-512 |
| **Signaturen** | RSA-PSS, SHA-512 (oder SHA-256 für Web) |
| **Hashing** | SHA-256 mit Salt + Pepper |
| **Key-Management** | Generierung, Datei-Speicherung, PEM-Konvertierung |

## DEVICE_KEY()

Systemweiter Geräteschlüssel. Wird als Default-Key verwendet wenn kein Key angegeben:

```python
from toolboxv2.utils.security.cryp import DEVICE_KEY
```

## API Reference

### Symmetrische Verschlüsselung (Fernet/AES)

| Method | Signature | Description |
|--------|-----------|-------------|
| `generate_symmetric_key` | `(as_str=True) → str\|bytes` | Fernet-Key generieren |
| `encrypt_symmetric` | `(text: str\|bytes, key: str) → str` | AES verschlüsseln |
| `decrypt_symmetric` | `(enc_text: str, key: str, to_str=True) → str\|bytes` | AES entschlüsseln |
| `encode_code` | `(data, key=None) → str` | Convenience: verschlüsseln mit DEVICE_KEY |
| `decode_code` | `(enc_data, key=None) → str` | Convenience: entschlüsseln mit DEVICE_KEY |

```python
from toolboxv2.utils.security.cryp import Code

# Generate key
key = Code.generate_symmetric_key()
# Encrypt
cipher = Code.encrypt_symmetric("secret data", key)
# Decrypt
plain = Code.decrypt_symmetric(cipher, key)

# With DEVICE_KEY (no explicit key needed)
enc = Code().encode_code("payload")
dec = Code().decode_code(enc)
```

### Asymmetrische Verschlüsselung (RSA)

| Method | Signature | Description |
|--------|-----------|-------------|
| `generate_asymmetric_keys` | `() → (pub_pem: str, priv_pem: str)` | RSA 6144-bit Key-Pair |
| `encrypt_asymmetric` | `(text: str, pub_key_pem: str) → str` | RSA-OAEP encrypt → hex |
| `decrypt_asymmetric` | `(enc_hex: str, priv_key_pem: str) → str` | RSA-OAEP decrypt |
| `save_keys_to_files` | `(pub, priv, dir="keys") → None` | Private Key wird mit DEVICE_KEY verschlüsselt |
| `load_keys_from_files` | `(dir="keys") → (pub, priv)` | Lädt + entschlüsselt Private Key |

### Signaturen

| Method | Signature | Description |
|--------|-----------|-------------|
| `create_signature` | `(msg: str, priv_key: str, row=False) → str\|bytes` | RSA-PSS Signatur (SHA-512) |
| `verify_signature` | `(sig, msg, pub_key, salt_length=PSS.MAX_LENGTH) → bool` | RSA-PSS verifizieren |
| `verify_signature_web_algo` | `(sig, msg, pub_key, algo=-512) → bool` | ECDSA für WebAuthn/Passkey |

### Hashing & Utility

| Method | Signature | Description |
|--------|-----------|-------------|
| `generate_random_string` | `(length: int) → str` | Sichere Zufallszeichen (`secrets.token_urlsafe`) |
| `generate_seed` | `() → int` | Zufalls-Seed (2³² – 2⁶⁴) |
| `one_way_hash` | `(text, salt='', pepper='') → str` | SHA-256 Hash mit Salt + Pepper |
| `pem_to_public_key` | `(pem: str) → RSAPublicKey` | PEM → Key-Objekt |
| `public_key_to_pem` | `(pub_key: RSAPublicKey) → str` | Key-Objekt → PEM |

## Usage Examples

### Symmetrisch (Config/FileHandler-Verschlüsselung)

```python
key = Code.generate_symmetric_key()
encrypted = Code.encrypt_symmetric('{"theme": "dark"}', key)
# Later...
config = Code.decrypt_symmetric(encrypted, key)
```

### Asymmetrisch (Auth/User-Verification)

```python
pub, priv = Code.generate_asymmetric_keys()
encrypted = Code.encrypt_asymmetric("user data", pub)
decrypted = Code.decrypt_asymmetric(encrypted, priv)
```

### Signatur (JWT/Auth)

```python
pub, priv = Code.generate_asymmetric_keys()
message = "authenticate user_123"
signature = Code.create_signature(message, priv)
is_valid = Code.verify_signature(signature, message, pub)
```

### Key-Storage (Sicher)

```python
# Private Key wird mit DEVICE_KEY verschlüsselt gespeichert
Code.save_keys_to_files(pub, priv, directory="keys")

# Later
pub, priv = Code.load_keys_from_files("keys")
# priv ist automatisch entschlüsselt
```

## Used By

- [Session Management](../runtime/session.md) — `pem_to_public_key`
- [FileHandlerV2](file_handler.md) — `encrypt_symmetric` (verschlüsselte Configs)
- [BlobStorage](../storage/ref_blobdb.md) — `Code` für Blob-Verschlüsselung
- [CloudM Auth](../mods/CloudM/auth.md) — Signaturen, JWT
- [CloudM FolderSync](../mods/CloudM/folder_sync.md) — AES-Verschlüsselung

## Related

- [Core Types](types.md)
- [FileHandlerV2](file_handler.md)
- [CloudM Auth](../mods/CloudM/auth.md)
