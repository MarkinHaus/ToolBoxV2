# file: toolboxv2/tests/test_mods/test_cloudm/test_types.py
"""
Tests for CloudM types module.

Tests the User and UserCreator dataclasses including:
- User creation and initialization
- UserCreator automatic key generation
- Field validation and defaults
- Encryption/decryption of user data
"""

import unittest
import uuid
import time
from unittest.mock import patch, MagicMock

from toolboxv2.mods.CloudM.types import User, UserCreator, UserPersonaPubKey
from toolboxv2 import Code


class TestUserPersonaPubKey(unittest.TestCase):
    """Tests for UserPersonaPubKey dataclass"""

    def test_user_persona_pub_key_creation(self):
        """Test creating a UserPersonaPubKey instance"""
        pub_key = UserPersonaPubKey(
            public_key=b"test_public_key",
            sign_count=1,
            credential_id=b"test_credential",
            rawId="test_raw_id",
            attestation_object=b"test_attestation"
        )
        
        self.assertEqual(pub_key.public_key, b"test_public_key")
        self.assertEqual(pub_key.sign_count, 1)
        self.assertEqual(pub_key.credential_id, b"test_credential")
        self.assertEqual(pub_key.rawId, "test_raw_id")
        self.assertEqual(pub_key.attestation_object, b"test_attestation")

    def test_user_persona_pub_key_fields(self):
        """Test all required fields are present"""
        pub_key = UserPersonaPubKey(
            public_key=b"key",
            sign_count=0,
            credential_id=b"cred",
            rawId="raw",
            attestation_object=b"att"
        )
        
        self.assertTrue(hasattr(pub_key, 'public_key'))
        self.assertTrue(hasattr(pub_key, 'sign_count'))
        self.assertTrue(hasattr(pub_key, 'credential_id'))
        self.assertTrue(hasattr(pub_key, 'rawId'))
        self.assertTrue(hasattr(pub_key, 'attestation_object'))


class TestUser(unittest.TestCase):
    """Tests for User dataclass"""

    def test_user_default_creation(self):
        """Test creating a User with default values"""
        user = User()
        
        # Check that UID is generated
        self.assertIsNotNone(user.uid)
        self.assertTrue(len(user.uid) > 0)
        
        # Check default values
        self.assertEqual(user.pub_key, "")
        self.assertEqual(user.email, "")
        self.assertEqual(user.name, "")
        self.assertEqual(user.user_pass_pub, "")
        self.assertEqual(user.user_pass_pri, "")
        self.assertEqual(user.user_pass_sync, "")
        self.assertEqual(user.challenge, "")
        self.assertFalse(user.is_persona)
        self.assertEqual(user.level, 0)
        self.assertEqual(user.log_level, "INFO")
        self.assertIsInstance(user.settings, dict)
        self.assertEqual(len(user.settings), 0)

    def test_user_with_custom_values(self):
        """Test creating a User with custom values"""
        custom_uid = str(uuid.uuid4())
        user = User(
            uid=custom_uid,
            email="test@example.com",
            name="Test User",
            level=5,
            log_level="DEBUG",
            is_persona=True
        )
        
        self.assertEqual(user.uid, custom_uid)
        self.assertEqual(user.email, "test@example.com")
        self.assertEqual(user.name, "Test User")
        self.assertEqual(user.level, 5)
        self.assertEqual(user.log_level, "DEBUG")
        self.assertTrue(user.is_persona)

    def test_user_creation_time_format(self):
        """Test that creation_time is in correct format"""
        user = User()
        
        # Check format: YYYY-MM-DD::HH:MM:SS
        self.assertIsNotNone(user.creation_time)
        self.assertIn("::", user.creation_time)
        parts = user.creation_time.split("::")
        self.assertEqual(len(parts), 2)
        
        # Verify date part
        date_part = parts[0]
        self.assertEqual(len(date_part.split("-")), 3)

    def test_user_persona_pub_devices_default(self):
        """Test that user_pass_pub_devices is initialized as empty list"""
        user = User()
        
        self.assertIsInstance(user.user_pass_pub_devices, list)
        self.assertEqual(len(user.user_pass_pub_devices), 0)

    def test_user_persona_pub_dict_default(self):
        """Test that user_pass_pub_persona is initialized as empty dict"""
        user = User()
        
        self.assertIsInstance(user.user_pass_pub_persona, dict)
        self.assertEqual(len(user.user_pass_pub_persona), 0)

    def test_user_settings_modification(self):
        """Test modifying user settings"""
        user = User()
        
        user.settings['theme'] = 'dark'
        user.settings['language'] = 'en'
        
        self.assertEqual(user.settings['theme'], 'dark')
        self.assertEqual(user.settings['language'], 'en')
        self.assertEqual(len(user.settings), 2)

    def test_user_devices_modification(self):
        """Test adding devices to user"""
        user = User()
        
        user.user_pass_pub_devices.append("device1")
        user.user_pass_pub_devices.append("device2")
        
        self.assertEqual(len(user.user_pass_pub_devices), 2)
        self.assertIn("device1", user.user_pass_pub_devices)
        self.assertIn("device2", user.user_pass_pub_devices)


class TestUserCreator(unittest.TestCase):
    """Tests for UserCreator dataclass"""

    def test_user_creator_generates_keys(self):
        """Test that UserCreator automatically generates keys"""
        user = UserCreator()
        
        # Check that keys are generated
        self.assertIsNotNone(user.user_pass_pub)
        self.assertIsNotNone(user.user_pass_pri)
        self.assertIsNotNone(user.user_pass_sync)
        self.assertIsNotNone(user.challenge)
        
        # Check that keys are not empty
        self.assertTrue(len(user.user_pass_pub) > 0)
        self.assertTrue(len(user.user_pass_pri) > 0)
        self.assertTrue(len(user.user_pass_sync) > 0)
        self.assertTrue(len(user.challenge) > 0)

    def test_user_creator_asymmetric_keys_are_different(self):
        """Test that public and private keys are different"""
        user = UserCreator()
        
        self.assertNotEqual(user.user_pass_pub, user.user_pass_pri)

    def test_user_creator_multiple_instances_different_keys(self):
        """Test that different UserCreator instances have different keys"""
        user1 = UserCreator()
        user2 = UserCreator()
        
        self.assertNotEqual(user1.user_pass_pub, user2.user_pass_pub)
        self.assertNotEqual(user1.user_pass_pri, user2.user_pass_pri)
        self.assertNotEqual(user1.user_pass_sync, user2.user_pass_sync)
        self.assertNotEqual(user1.challenge, user2.challenge)

    def test_user_creator_challenge_is_encrypted(self):
        """Test that challenge is encrypted with public key"""
        user = UserCreator()
        
        # Challenge should be encrypted, so it should be different from a plain UUID
        self.assertIsNotNone(user.challenge)
        self.assertTrue(len(user.challenge) > 0)
        
        # Try to decrypt the challenge with private key
        try:
            decrypted = Code.decrypt_asymmetric(user.challenge, user.user_pass_pri)
            # Should be a valid UUID format
            uuid.UUID(decrypted)
        except Exception as e:
            self.fail(f"Challenge decryption failed: {e}")

    def test_user_creator_inherits_user_fields(self):
        """Test that UserCreator inherits all User fields"""
        user = UserCreator(
            email="creator@example.com",
            name="Creator User",
            level=10
        )
        
        self.assertEqual(user.email, "creator@example.com")
        self.assertEqual(user.name, "Creator User")
        self.assertEqual(user.level, 10)
        self.assertIsNotNone(user.uid)

    def test_user_creator_symmetric_key_format(self):
        """Test that symmetric key is in correct format"""
        user = UserCreator()
        
        # Symmetric key should be a valid base64 string
        self.assertIsNotNone(user.user_pass_sync)
        self.assertTrue(len(user.user_pass_sync) > 0)
        
        # Test that it can be used for encryption/decryption
        test_data = b"test data"
        try:
            encrypted = Code.encrypt_symmetric(test_data, user.user_pass_sync)
            decrypted = Code.decrypt_symmetric(encrypted, user.user_pass_sync, to_str=False)
            self.assertEqual(test_data, decrypted)
        except Exception as e:
            self.fail(f"Symmetric encryption/decryption failed: {e}")

    def test_user_creator_with_custom_values(self):
        """Test UserCreator with custom initialization values"""
        custom_uid = str(uuid.uuid4())
        user = UserCreator(
            uid=custom_uid,
            email="custom@example.com",
            name="Custom User",
            is_persona=True,
            log_level="ERROR"
        )
        
        # Custom values should be preserved
        self.assertEqual(user.uid, custom_uid)
        self.assertEqual(user.email, "custom@example.com")
        self.assertEqual(user.name, "Custom User")
        self.assertTrue(user.is_persona)
        self.assertEqual(user.log_level, "ERROR")
        
        # Keys should still be generated
        self.assertIsNotNone(user.user_pass_pub)
        self.assertIsNotNone(user.user_pass_pri)
        self.assertIsNotNone(user.user_pass_sync)
        self.assertIsNotNone(user.challenge)


class TestUserIntegration(unittest.TestCase):
    """Integration tests for User types"""

    def test_user_to_user_creator_conversion(self):
        """Test converting User data to UserCreator"""
        # Create a basic user
        basic_user = User(
            email="basic@example.com",
            name="Basic User",
            level=3
        )
        
        # Create UserCreator with same data
        creator = UserCreator(
            uid=basic_user.uid,
            email=basic_user.email,
            name=basic_user.name,
            level=basic_user.level
        )
        
        # Basic fields should match
        self.assertEqual(creator.uid, basic_user.uid)
        self.assertEqual(creator.email, basic_user.email)
        self.assertEqual(creator.name, basic_user.name)
        self.assertEqual(creator.level, basic_user.level)
        
        # But creator should have keys
        self.assertTrue(len(creator.user_pass_pub) > 0)
        self.assertTrue(len(creator.user_pass_pri) > 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

