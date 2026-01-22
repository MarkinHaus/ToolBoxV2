#!/usr/bin/env python3
"""
ToolBox Password Manager Tests
Comprehensive testing for password management functionality
"""

import unittest
import tempfile
import os
import json
import time
from dataclasses import asdict
from unittest.mock import Mock, patch

from toolboxv2 import App, Result
from toolboxv2.mods.PasswordManager import (
    PasswordEntry, PasswordManagerCore, PasswordImporter, TOTPManager,
    ImportResult, add_password, get_password, search_passwords,
    list_passwords, generate_password, import_passwords,
    generate_totp_code, add_totp_secret, parse_totp_qr_code
)
from toolboxv2.tests.a_util import IsolatedTestCase


class TestPasswordEntry(IsolatedTestCase):
    """Test PasswordEntry data structure"""

    def test_password_entry_creation(self):
        """Test creating a password entry"""
        entry = PasswordEntry(
            id="test123",
            url="https://example.com",
            username="testuser",
            password="testpass123",
            title="Example Site"
        )

        self.assertEqual(entry.id, "test123")
        self.assertEqual(entry.url, "https://example.com")
        self.assertEqual(entry.username, "testuser")
        self.assertEqual(entry.password, "testpass123")
        self.assertEqual(entry.title, "Example Site")
        self.assertIsInstance(entry.created_at, float)
        self.assertIsInstance(entry.updated_at, float)

    def test_password_entry_auto_id(self):
        """Test automatic ID generation"""
        entry = PasswordEntry(
            id="",
            url="https://test.com",
            username="user",
            password="pass"
        )

        self.assertIsNotNone(entry.id)
        self.assertNotEqual(entry.id, "")
        self.assertEqual(len(entry.id), 16)

    def test_password_update_history(self):
        """Test password update with history"""
        entry = PasswordEntry(
            id="test",
            url="https://test.com",
            username="user",
            password="oldpass"
        )

        original_time = entry.updated_at
        time.sleep(0.01)  # Small delay to ensure time difference

        entry.update_password("newpass")

        self.assertEqual(entry.password, "newpass")
        self.assertGreater(entry.updated_at, original_time)
        self.assertEqual(len(entry.password_history), 1)
        self.assertEqual(entry.password_history[0]['password'], "oldpass")

    def test_get_domain(self):
        """Test domain extraction"""
        entry = PasswordEntry(
            id="test",
            url="https://www.example.com/login",
            username="user",
            password="pass"
        )

        self.assertEqual(entry.get_domain(), "www.example.com")

    def test_to_dict_from_dict(self):
        """Test serialization and deserialization"""
        original = PasswordEntry(
            id="test123",
            url="https://example.com",
            username="testuser",
            password="testpass123",
            title="Example Site",
            notes="Test notes",
            tags=["work", "important"]
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = PasswordEntry.from_dict(data)

        self.assertEqual(original.id, restored.id)
        self.assertEqual(original.url, restored.url)
        self.assertEqual(original.username, restored.username)
        self.assertEqual(original.password, restored.password)
        self.assertEqual(original.title, restored.title)
        self.assertEqual(original.notes, restored.notes)
        self.assertEqual(original.tags, restored.tags)


class TestTOTPManager(IsolatedTestCase):
    """Test TOTP functionality"""

    def test_generate_totp_code(self):
        """Test TOTP code generation"""
        # Use a known test secret
        secret = "JBSWY3DPEHPK3PXP"  # "Hello!" in base32

        code = TOTPManager.generate_totp_code(secret)

        self.assertIsInstance(code, str)
        self.assertEqual(len(code), 6)
        self.assertTrue(code.isdigit())

    def test_parse_totp_uri(self):
        """Test TOTP URI parsing"""
        uri = "otpauth://totp/Example:alice@google.com?secret=JBSWY3DPEHPK3PXP&issuer=Example"

        parsed = TOTPManager.parse_totp_uri(uri)

        self.assertEqual(parsed['secret'], 'JBSWY3DPEHPK3PXP')
        self.assertEqual(parsed['issuer'], 'Example')
        self.assertEqual(parsed['account'], 'alice@google.com')
        self.assertEqual(parsed['algorithm'], 'SHA1')
        self.assertEqual(parsed['digits'], '6')
        self.assertEqual(parsed['period'], '30')

    def test_generate_qr_code_uri(self):
        """Test QR code URI generation"""
        uri = TOTPManager.generate_qr_code_uri(
            secret="JBSWY3DPEHPK3PXP",
            account="alice@google.com",
            issuer="Example"
        )

        self.assertTrue(uri.startswith("otpauth://totp/"))
        self.assertIn("secret=JBSWY3DPEHPK3PXP", uri)
        self.assertIn("issuer=Example", uri)


class TestPasswordImporter(IsolatedTestCase):
    """Test password import functionality"""

    def setUp(self):
        """Set up test environment"""
        super().setUp()
        # Don't use spec=App to avoid mock contamination issues
        self.app = Mock()
        self.app.config = {'blob_servers': ['http://localhost:8080']}
        self.app.logger = Mock()

    def test_import_generic_csv(self):
        """Test generic CSV import"""
        csv_content = """url,username,password,title,notes
https://example.com,testuser,testpass123,Example Site,Test notes
https://test.com,user2,pass456,Test Site,"""

        with patch('toolboxv2.mods.PasswordManager.PasswordManagerCore') as mock_core:
            mock_instance = Mock()
            mock_core.return_value = mock_instance
            mock_instance.add_password.return_value = Result.ok()

            importer = PasswordImporter(self.app)
            result = importer._import_csv(csv_content, "Imported")

            self.assertTrue(result.success)
            self.assertEqual(result.imported_count, 2)

    def test_import_invalid_csv(self):
        """Test handling of invalid CSV data"""
        csv_content = """url,username
https://example.com,testuser
,incomplete_entry,"""

        with patch('toolboxv2.mods.PasswordManager.PasswordManagerCore') as mock_core:
            mock_instance = Mock()
            mock_core.return_value = mock_instance

            importer = PasswordImporter(self.app)
            result = importer._import_csv(csv_content, "Imported")

            self.assertTrue(result.success)
            self.assertEqual(result.imported_count, 0)
            self.assertGreater(result.skipped_count, 0)


class TestPasswordGeneration(IsolatedTestCase):
    """Test password generation"""

    def setUp(self):
        super().setUp()
        # Don't use spec=App to avoid mock contamination issues
        self.app = Mock()

    def test_generate_numbers_only(self):
        """Test numbers-only password generation"""
        result = generate_password(
            self.app,
            length=8,
            include_symbols=False,
            include_uppercase=False,
            include_lowercase=False,
            include_numbers=True
        ).as_result()

        self.assertTrue(result.is_ok())
        password = result.get()['password']
        self.assertTrue(password.isdigit())

    def test_invalid_length(self):
        """Test invalid password length"""
        result = generate_password(self.app, length=2).as_result()
        self.assertTrue(result.is_error())

        result = generate_password(self.app, length=200).as_result()
        self.assertTrue(result.is_error())

    def test_no_character_types(self):
        """Test password generation with no character types"""
        result = generate_password(
            self.app,
            include_symbols=False,
            include_numbers=False,
            include_uppercase=False,
            include_lowercase=False
        )

        self.assertTrue(result.is_error())


# API tests removed - PasswordManager now uses ToolBox's built-in API system
# All functions are automatically exposed via @export(api=True) decorators


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
