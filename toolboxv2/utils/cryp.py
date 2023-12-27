import base64
import random
import time
import os
import hashlib

from cryptography.exceptions import InvalidSignature
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes

from toolboxv2.utils.tb_logger import get_logger

from platform import node


class Code:
    application_key = None

    def decode_code(self, data, key=None):

        if not isinstance(data, str):
            data = str(data)

        if key is None:
            key = base64.urlsafe_b64encode((node() + '0' * min(32 - len(node()), 32)).encode() if len(node()) < 32 else
                                           node()[:32].encode()).decode()
        return self.decrypt_symmetric(data, key)

    def encode_code(self, data, key=None):

        if not isinstance(data, str):
            data = str(data)

        if key is None:
            key = base64.urlsafe_b64encode((node() + '0' * min(32 - len(node()), 32)).encode() if len(node()) < 32 else
                                           node()[:32].encode()).decode()
        return self.encrypt_symmetric(data, key)

    @staticmethod
    def generate_seed() -> int:
        """
        Erzeugt eine zufällige Zahl als Seed.

        Returns:
            int: Eine zufällige Zahl.
        """
        time.sleep(random.uniform(0, 0.001))  # Mikroverzögerung
        return random.randint(2 ** 32 - 1, 2 ** 64 - 1)

    @staticmethod
    def one_way_hash(text: str, salt: str = '', pepper: str = '') -> str:
        """
        Erzeugt einen Hash eines gegebenen Textes mit Salt, Pepper und optional einem Seed.

        Args:
            text (str): Der zu hashende Text.
            salt (str): Der Salt-Wert.
            pepper (str): Der Pepper-Wert.
            seed (int, optional): Ein optionaler Seed-Wert. Standardmäßig None.

        Returns:
            str: Der resultierende Hash-Wert.
        """
        return hashlib.sha256((salt + text + pepper).encode()).hexdigest()

    @staticmethod
    def generate_symmetric_key() -> str:
        """
        Generiert einen Schlüssel für die symmetrische Verschlüsselung.

        Args:
            seed (int, optional): Ein optionaler Seed-Wert. Standardmäßig None.

        Returns:
            str: Der generierte Schlüssel.
        """
        return Fernet.generate_key().decode()

    @staticmethod
    def encrypt_symmetric(text: str, key: str) -> str:
        """
        Verschlüsselt einen Text mit einem gegebenen symmetrischen Schlüssel.

        Args:
            text (str): Der zu verschlüsselnde Text.
            key (str): Der symmetrische Schlüssel.

        Returns:
            str: Der verschlüsselte Text.
        """
        time.sleep(random.uniform(0, 0.001))  # Mikroverzögerung
        try:
            fernet = Fernet(key.encode())
            return fernet.encrypt(text.encode()).decode()
        except Exception as e:
            get_logger().error(f"Error encrypt_symmetric {e}")
            return "Error encrypt"

    @staticmethod
    def decrypt_symmetric(encrypted_text: str, key: str) -> str:
        """
        Entschlüsselt einen Text mit einem gegebenen symmetrischen Schlüssel.

        Args:
            encrypted_text (str): Der zu entschlüsselnde Text.
            key (str): Der symmetrische Schlüssel.

        Returns:
            str: Der entschlüsselte Text.
        """
        time.sleep(random.uniform(0, 0.001))  # Mikroverzögerung
        try:
            fernet = Fernet(key.encode())
            return fernet.decrypt(encrypted_text.encode()).decode()
        except Exception as e:
            get_logger().error(f"Error decrypt_symmetric {e}")
            return f"Error decoding"

    @staticmethod
    def generate_asymmetric_keys() -> (str, str):
        """
        Generiert ein Paar von öffentlichen und privaten Schlüsseln für die asymmetrische Verschlüsselung.

        Args:
            seed (int, optional): Ein optionaler Seed-Wert. Standardmäßig None.

        Returns:
            (str, str): Ein Tupel aus öffentlichem und privatem Schlüssel.
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        public_key = private_key.public_key()

        # Serialisieren der Schlüssel
        pem_private_key = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode()

        pem_public_key = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()

        return pem_public_key, pem_private_key

    @staticmethod
    def encrypt_asymmetric(text: str, public_key_str: str) -> str:
        """
        Verschlüsselt einen Text mit einem gegebenen öffentlichen Schlüssel.

        Args:
            text (str): Der zu verschlüsselnde Text.
            public_key_str (str): Der öffentliche Schlüssel als String.

        Returns:
            str: Der verschlüsselte Text.
        """
        try:
            public_key = serialization.load_pem_public_key(public_key_str.encode())
            encrypted = public_key.encrypt(
                text.encode(),
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA512()),
                    algorithm=hashes.SHA512(),
                    label=None
                )
            )
            return encrypted.hex()
        except Exception as e:
            get_logger().error(f"Error encrypt_asymmetric {e}")
            return "Invalid"

    @staticmethod
    def decrypt_asymmetric(encrypted_text_hex: str, private_key_str: str) -> str:
        """
        Entschlüsselt einen Text mit einem gegebenen privaten Schlüssel.

        Args:
            encrypted_text_hex (str): Der verschlüsselte Text als Hex-String.
            private_key_str (str): Der private Schlüssel als String.

        Returns:
            str: Der entschlüsselte Text.
        """
        try:
            private_key = serialization.load_pem_private_key(private_key_str.encode(), password=None)
            decrypted = private_key.decrypt(
                bytes.fromhex(encrypted_text_hex),
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA512()),
                    algorithm=hashes.SHA512(),
                    label=None
                )
            )
            return decrypted.decode()

        except Exception as e:
            get_logger().error(f"Error encrypt_asymmetric {e}")
        return "Invalid"

    @staticmethod
    def verify_signature(signature: str, message: str, public_key_str: str) -> bool:
        try:
            public_key = serialization.load_pem_public_key(public_key_str.encode())
            public_key.verify(
                signature.encode(),
                message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA512()
            )
            return True
        except InvalidSignature:
            pass
        return False

    @staticmethod
    def create_signature(message: str, private_key_str: str) -> str:
        try:
            private_key = serialization.load_pem_private_key(private_key_str.encode(), password=None)
            signature = private_key.sign(
                message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return signature.decode()
        except Exception as e:
            get_logger().error(f"Error encrypt_asymmetric {e}")
        return "Invalid Key"
