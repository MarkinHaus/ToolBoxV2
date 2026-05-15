"""Tests for crypto.py — AES-256-GCM + zlib compression."""
import os
import tempfile
import unittest


class TestCryptoKeyGen(unittest.TestCase):
    def test_generate_key_length(self):
        from toolboxv2.mods.CloudM.LiveSync.crypto import generate_encryption_key
        key_b64 = generate_encryption_key()
        self.assertIsInstance(key_b64, str)
        import base64
        raw = base64.urlsafe_b64decode(key_b64)
        self.assertEqual(len(raw), 32)  # AES-256 = 32 bytes

    def test_keys_are_unique(self):
        from toolboxv2.mods.CloudM.LiveSync.crypto import generate_encryption_key
        k1 = generate_encryption_key()
        k2 = generate_encryption_key()
        self.assertNotEqual(k1, k2)


class TestEncryptDecryptBytes(unittest.TestCase):
    def setUp(self):
        from toolboxv2.mods.CloudM.LiveSync.crypto import generate_encryption_key
        self.key = generate_encryption_key()

    def test_roundtrip_bytes(self):
        from toolboxv2.mods.CloudM.LiveSync.crypto import encrypt_bytes, decrypt_bytes
        original = b"Hello LiveSync! " * 100
        encrypted = encrypt_bytes(original, self.key)
        self.assertNotEqual(encrypted, original)
        decrypted = decrypt_bytes(encrypted, self.key)
        self.assertEqual(decrypted, original)

    def test_empty_bytes(self):
        from toolboxv2.mods.CloudM.LiveSync.crypto import encrypt_bytes, decrypt_bytes
        encrypted = encrypt_bytes(b"", self.key)
        decrypted = decrypt_bytes(encrypted, self.key)
        self.assertEqual(decrypted, b"")

    def test_large_data(self):
        from toolboxv2.mods.CloudM.LiveSync.crypto import encrypt_bytes, decrypt_bytes
        original = os.urandom(1024 * 1024)  # 1 MB random
        encrypted = encrypt_bytes(original, self.key)
        decrypted = decrypt_bytes(encrypted, self.key)
        self.assertEqual(decrypted, original)

    def test_compression_shrinks_text(self):
        from toolboxv2.mods.CloudM.LiveSync.crypto import encrypt_bytes
        # Highly compressible data
        original = b"AAAA" * 10000  # 40 KB of repeats
        encrypted = encrypt_bytes(original, self.key)
        # encrypted should be much smaller due to zlib
        self.assertLess(len(encrypted), len(original))

    def test_wrong_key_fails(self):
        from toolboxv2.mods.CloudM.LiveSync.crypto import encrypt_bytes, decrypt_bytes, generate_encryption_key
        encrypted = encrypt_bytes(b"secret", self.key)
        wrong_key = generate_encryption_key()
        with self.assertRaises(Exception):
            decrypt_bytes(encrypted, wrong_key)

    def test_tampered_data_fails(self):
        from toolboxv2.mods.CloudM.LiveSync.crypto import encrypt_bytes, decrypt_bytes
        encrypted = encrypt_bytes(b"secret", self.key)
        tampered = encrypted[:-1] + bytes([encrypted[-1] ^ 0xFF])
        with self.assertRaises(Exception):
            decrypt_bytes(tampered, self.key)


class TestEncryptDecryptFile(unittest.TestCase):
    def setUp(self):
        from toolboxv2.mods.CloudM.LiveSync.crypto import generate_encryption_key
        self.key = generate_encryption_key()
        self.tmpdir = tempfile.mkdtemp()

    def test_encrypt_file_roundtrip(self):
        from toolboxv2.mods.CloudM.LiveSync.crypto import encrypt_file, decrypt_to_file
        # Create test file
        src = os.path.join(self.tmpdir, "test.md")
        content = "# Hello\nThis is a test document.\n" * 50
        with open(src, "w") as f:
            f.write(content)

        # Encrypt
        encrypted = encrypt_file(src, self.key)
        self.assertIsInstance(encrypted, bytes)
        self.assertNotIn(b"Hello", encrypted)

        # Decrypt to new file
        dst = os.path.join(self.tmpdir, "restored.md")
        decrypt_to_file(encrypted, self.key, dst)

        with open(dst) as f:
            restored = f.read()
        self.assertEqual(restored, content)

    def test_encrypt_binary_file(self):
        from toolboxv2.mods.CloudM.LiveSync.crypto import encrypt_file, decrypt_to_file
        src = os.path.join(self.tmpdir, "test.bin")
        data = os.urandom(4096)
        with open(src, "wb") as f:
            f.write(data)

        encrypted = encrypt_file(src, self.key)
        dst = os.path.join(self.tmpdir, "restored.bin")
        decrypt_to_file(encrypted, self.key, dst)

        with open(dst, "rb") as f:
            restored = f.read()
        self.assertEqual(restored, data)

    def test_atomic_write_uses_tmp(self):
        """decrypt_to_file should use .sync-tmp then rename."""
        from toolboxv2.mods.CloudM.LiveSync.crypto import decrypt_to_file, encrypt_bytes
        encrypted = encrypt_bytes(b"data", self.key)
        dst = os.path.join(self.tmpdir, "sub", "deep", "file.txt")
        # Should create parent dirs
        decrypt_to_file(encrypted, self.key, dst)
        self.assertTrue(os.path.exists(dst))
        # No .sync-tmp left behind
        self.assertFalse(os.path.exists(dst + ".sync-tmp"))


class TestChecksum(unittest.TestCase):
    def test_checksum_deterministic(self):
        from toolboxv2.mods.CloudM.LiveSync.crypto import compute_checksum
        c1 = compute_checksum(b"hello world")
        c2 = compute_checksum(b"hello world")
        self.assertEqual(c1, c2)

    def test_checksum_length(self):
        from toolboxv2.mods.CloudM.LiveSync.crypto import compute_checksum
        c = compute_checksum(b"test")
        self.assertEqual(len(c), 16)  # first 16 chars of sha256

    def test_different_data_different_checksum(self):
        from toolboxv2.mods.CloudM.LiveSync.crypto import compute_checksum
        c1 = compute_checksum(b"aaa")
        c2 = compute_checksum(b"bbb")
        self.assertNotEqual(c1, c2)

    def test_checksum_file(self):
        from toolboxv2.mods.CloudM.LiveSync.crypto import compute_checksum_file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"hello world")
            path = f.name
        try:
            c = compute_checksum_file(path)
            self.assertEqual(len(c), 16)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
