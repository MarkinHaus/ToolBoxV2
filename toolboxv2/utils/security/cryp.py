import base64
import random
import os
import hashlib

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.ec import ECDSA
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

from ..system.tb_logger import get_logger

DEVICE_KEY_DIR = "./.info"
DEVICE_KEY_PATH = os.path.join(DEVICE_KEY_DIR, "device.key")


def ensure_device_key_dir_exists():
    if not os.path.exists(DEVICE_KEY_DIR):
        os.makedirs(DEVICE_KEY_DIR)


def get_or_create_device_key():
    ensure_device_key_dir_exists()
    if os.path.exists(DEVICE_KEY_PATH):
        with open(DEVICE_KEY_PATH, "r") as key_file:
            return key_file.read()
    else:
        key = Fernet.generate_key()
        with open(DEVICE_KEY_PATH, "wb") as key_file:
            key_file.write(key)
        return key.decode()


DEVICE_KEY = get_or_create_device_key


class Code:
    application_key = None

    @staticmethod
    def DK():
        return DEVICE_KEY

    def decode_code(self, encrypted_data, key=None):

        if not isinstance(encrypted_data, str):
            encrypted_data = str(encrypted_data)

        if key is None:
            key = DEVICE_KEY()

        return self.decrypt_symmetric(encrypted_data, key)

    def encode_code(self, data, key=None):

        if not isinstance(data, str):
            data = str(data)

        if key is None:
            key = DEVICE_KEY()

        return self.encrypt_symmetric(data, key)

    @staticmethod
    def generate_seed() -> int:
        """
        Erzeugt eine zufällige Zahl als Seed.

        Returns:
            int: Eine zufällige Zahl.
        """
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

        Returns:
            str: Der generierte Schlüssel.
        """
        return Fernet.generate_key().decode()

    @staticmethod
    def encrypt_symmetric(text: str or bytes, key: str) -> str:
        """
        Verschlüsselt einen Text mit einem gegebenen symmetrischen Schlüssel.

        Args:
            text (str): Der zu verschlüsselnde Text.
            key (str): Der symmetrische Schlüssel.

        Returns:
            str: Der verschlüsselte Text.
        """
        if isinstance(text, str):
            text = text.encode()

        try:
            fernet = Fernet(key.encode())
            return fernet.encrypt(text).decode()
        except Exception as e:
            get_logger().error(f"Error encrypt_symmetric #{str(e)}#")
            return "Error encrypt"

    @staticmethod
    def decrypt_symmetric(encrypted_text: str, key: str, to_str=True) -> str or bytes:
        """
        Entschlüsselt einen Text mit einem gegebenen symmetrischen Schlüssel.

        Args:
            encrypted_text (str): Der zu entschlüsselnde Text.
            key (str): Der symmetrische Schlüssel.
            to_str (bool): default true returns str if false returns bytes
        Returns:
            str: Der entschlüsselte Text.
        """

        if isinstance(key, str):
            key = key.encode()

        try:
            fernet = Fernet(key)
            text_b = fernet.decrypt(encrypted_text)
            if not to_str:
                return text_b
            return text_b.decode()
        except Exception as e:
            get_logger().error(f"Error decrypt_symmetric {e}")
            if not to_str:
                return f"Error decoding".encode()
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
            key_size=2048 * 3,
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
            public_key_str (str): Der öffentliche Schlüssel als String oder im pem format.

        Returns:
            str: Der verschlüsselte Text.
        """
        try:
            public_key: RSAPublicKey = serialization.load_pem_public_key(public_key_str.encode())
        except Exception as e:
            get_logger().error(f"Error encrypt_asymmetric {e}")
        try:
            public_key: RSAPublicKey = serialization.load_pem_public_key(public_key_str.encode())
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
            get_logger().error(f"Error decrypt_asymmetric {e}")
        return "Invalid"

    @staticmethod
    def verify_signature(signature: str or bytes, message: str or bytes, public_key_str: str,
                         salt_length=padding.PSS.MAX_LENGTH) -> bool:
        if isinstance(signature, str):
            signature = signature.encode()
        if isinstance(message, str):
            message = message.encode()
        try:
            public_key: RSAPublicKey = serialization.load_pem_public_key(public_key_str.encode())
            public_key.verify(
                signature=signature,
                data=message,
                padding=padding.PSS(
                    mgf=padding.MGF1(hashes.SHA512()),
                    salt_length=salt_length
                ),
                algorithm=hashes.SHA512()
            )
            return True
        except Exception as e:
            get_logger().error(f"Error validating signature: {e}")
            print(e)
        return False

    @staticmethod
    def verify_signature_web_algo(signature: str or bytes, message: str or bytes, public_key_str: str,
                                  algo: int = -512) -> bool:
        signature_algorithm = ECDSA(hashes.SHA512())
        if algo != -512:
            signature_algorithm = ECDSA(hashes.SHA256())

        if isinstance(signature, str):
            signature = signature.encode()
        if isinstance(message, str):
            message = message.encode()
        try:
            public_key = serialization.load_pem_public_key(public_key_str.encode())
            public_key.verify(
                signature=signature,
                data=message,
                # padding=padding.PSS(
                #    mgf=padding.MGF1(hashes.SHA512()),
                #    salt_length=padding.PSS.MAX_LENGTH
                # ),
                signature_algorithm=signature_algorithm
            )
            return True
        except Exception as e:
            print(e)
            get_logger().error(f"Error validating signature: {e}")
        return False

    @staticmethod
    def create_signature(message: str, private_key_str: str, salt_length=padding.PSS.MAX_LENGTH,
                         row=False) -> str or bytes:
        try:
            private_key = serialization.load_pem_private_key(private_key_str.encode(), password=None)
            signature = private_key.sign(
                message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA512()),
                    salt_length=salt_length
                ),
                hashes.SHA512()
            )
            if row:
                return signature
            return base64.b64encode(signature).decode()
        except Exception as e:
            get_logger().error(f"Error create_signature {e}")
            print(e)
        return "Invalid Key"

    @staticmethod
    def pem_to_public_key(pem_key: str):
        """
        Konvertiert einen PEM-kodierten öffentlichen Schlüssel in ein PublicKey-Objekt.

        Args:
            pem_key (str): Der PEM-kodierte öffentliche Schlüssel.

        Returns:
            PublicKey: Das PublicKey-Objekt.
        """
        public_key = serialization.load_pem_public_key(pem_key.encode())
        return public_key

    @staticmethod
    def public_key_to_pem(public_key: RSAPublicKey):
        """
        Konvertiert ein PublicKey-Objekt in einen PEM-kodierten String.

        Args:
            public_key (PublicKey): Das PublicKey-Objekt.

        Returns:
            str: Der PEM-kodierte öffentliche Schlüssel.
        """
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.decode()
