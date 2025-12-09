import asyncio
import base64
import hashlib
import os
import queue
import random
import secrets
from collections.abc import Callable
from functools import wraps

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.ec import ECDSA
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from dotenv import load_dotenv

from ..system.tb_logger import get_logger

load_dotenv()


DEVICE_KEY_DIR = "./.info"
DEVICE_KEY_PATH = os.path.join(DEVICE_KEY_DIR, "device.enc")


def ensure_device_key_dir_exists():
    if not os.path.exists(DEVICE_KEY_DIR):
        os.makedirs(DEVICE_KEY_DIR)

TB_R_KEY = os.getenv("TB_R_KEY", "randomstring")

def derive_aes_key(tb_r_key: str) -> bytes:
    return hashlib.sha256(tb_r_key.encode()).digest()  # 32 Byte AES-Schlüssel

def encrypt_with_key(data: bytes, key: bytes) -> bytes:
    nonce = secrets.token_bytes(12)  # AES-GCM benötigt 12-Byte Nonce
    aesgcm = AESGCM(key)
    encrypted = aesgcm.encrypt(nonce, data, None)
    return nonce + encrypted  # nonce vorne anhängen

def decrypt_with_key(encrypted_data: bytes, key: bytes) -> bytes:
    nonce = encrypted_data[:12]
    ciphertext = encrypted_data[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)

def get_or_create_device_key():
    try:
        ensure_device_key_dir_exists()
        aes_key = derive_aes_key(TB_R_KEY)

        if os.path.exists(DEVICE_KEY_PATH):
            with open(DEVICE_KEY_PATH, "rb") as key_file:
                encrypted_data = key_file.read()
            decrypted_key = decrypt_with_key(encrypted_data, aes_key)
            return decrypted_key.decode()
        else:
            key = Fernet.generate_key()  # 32 Byte Base64
            encrypted = encrypt_with_key(key, aes_key)
            with open(DEVICE_KEY_PATH, "wb") as key_file:
                key_file.write(encrypted)
            return key.decode()
    except Exception as e:
        get_logger().error(f"Error get_or_create_device_key {e}")
        import traceback
        traceback.print_exc()
        raise e


DEVICE_KEY = get_or_create_device_key


class Code:

    @staticmethod
    def DK():
        return DEVICE_KEY

    @staticmethod
    def generate_random_string(length: int) -> str:
        """
        Generiert eine zufällige Zeichenkette der angegebenen Länge.

        Args:
            length (int): Die Länge der zu generierenden Zeichenkette.

        Returns:
            str: Die generierte Zeichenkette.
        """
        return secrets.token_urlsafe(length)

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
    def generate_symmetric_key(as_str=True) -> str or bytes:
        """
        Generiert einen Schlüssel für die symmetrische Verschlüsselung.

        Returns:
            str: Der generierte Schlüssel.
        """
        key = Fernet.generate_key()
        if as_str:
            key = key.decode()
        return key

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
        if isinstance(key, str):
            key = key.encode()

        fernet = Fernet(key)
        return fernet.encrypt(text).decode()

    @staticmethod
    def decrypt_symmetric(encrypted_text: str, key: str, to_str=True, mute=False) -> str or bytes:
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

        #try:
        fernet = Fernet(key)
        text_b = fernet.decrypt(encrypted_text)
        if not to_str:
            return text_b
        return text_b.decode()
        # except Exception as e:
        #     get_logger().error(f"Error decrypt_symmetric {e}")
        #     if not mute:
        #         raise e
        #     if not to_str:
        #         return f"Error decoding".encode()
        #     return f"Error decoding"

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
    def save_keys_to_files(public_key: str, private_key: str, directory: str = "keys") -> None:
        """
        Speichert die generierten Schlüssel in separate Dateien.
        Der private Schlüssel wird mit dem Device Key verschlüsselt.

        Args:
            public_key (str): Der öffentliche Schlüssel im PEM-Format
            private_key (str): Der private Schlüssel im PEM-Format
            directory (str): Das Verzeichnis, in dem die Schlüssel gespeichert werden sollen
        """
        # Erstelle das Verzeichnis, falls es nicht existiert
        os.makedirs(directory, exist_ok=True)

        # Hole den Device Key
        device_key = DEVICE_KEY()

        # Verschlüssele den privaten Schlüssel mit dem Device Key
        encrypted_private_key = Code.encrypt_symmetric(private_key, device_key)

        # Speichere den öffentlichen Schlüssel
        public_key_path = os.path.join(directory, "public_key.pem")
        with open(public_key_path, "w") as f:
            f.write(public_key)

        # Speichere den verschlüsselten privaten Schlüssel
        private_key_path = os.path.join(directory, "private_key.pem")
        with open(private_key_path, "w") as f:
            f.write(encrypted_private_key)

        print("Saved keys in ", public_key_path)

    @staticmethod
    def load_keys_from_files(directory: str = "keys") -> (str, str):
        """
        Lädt die Schlüssel aus den Dateien.
        Der private Schlüssel wird mit dem Device Key entschlüsselt.

        Args:
            directory (str): Das Verzeichnis, aus dem die Schlüssel geladen werden sollen

        Returns:
            (str, str): Ein Tupel aus öffentlichem und privatem Schlüssel

        Raises:
            FileNotFoundError: Wenn die Schlüsseldateien nicht gefunden werden können
        """
        # Pfade zu den Schlüsseldateien
        public_key_path = os.path.join(directory, "public_key.pem")
        private_key_path = os.path.join(directory, "private_key.pem")

        # Prüfe ob die Dateien existieren
        if not os.path.exists(public_key_path) or not os.path.exists(private_key_path):
            return "", ""

        # Hole den Device Key
        device_key = DEVICE_KEY()

        # Lade den öffentlichen Schlüssel
        with open(public_key_path) as f:
            public_key = f.read()

        # Lade und entschlüssele den privaten Schlüssel
        with open(private_key_path) as f:
            encrypted_private_key = f.read()
            private_key = Code.decrypt_symmetric(encrypted_private_key, device_key)

        return public_key, private_key

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
        # try:
        #    public_key: RSAPublicKey = serialization.load_pem_public_key(public_key_str.encode())
        #  except Exception as e:
        #     get_logger().error(f"Error encrypt_asymmetric {e}")
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
        except:
            pass
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
        except:
            pass
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


class E2EEncryption:
    def __init__(self, r_send=None, r_recv=None):
        self.code = Code()
        self.device_key = self.code.DK()
        self.session_key = None
        self.remote_public_key = None
        def r_send(*args, **kwargs):
            return None if r_send is None else r_send
        r_recv: queue.Queue = queue.Queue() if r_recv is None else r_recv
        self.row_function = [r_send, r_recv]

    async def _exchange_keys(self):
        public_key, private_key = self.code.generate_asymmetric_keys()
        # Here you would implement the actual key exchange protocol
        # For example, send your public key and receive the remote public key
        # This is a placeholder and should be replaced with actual implementation
        self.row_function[0](public_key)
        try:
            self.remote_public_key = self.row_function[1].get(timeout=15)
        except queue.Empty:
            print("exchange_keys Failure")
            return
        self.remote_public_key = "REMOTE_PUBLIC_KEY_PLACEHOLDER"
        self.session_key = self.code.generate_symmetric_key()
        self.code.encrypt_asymmetric(self.session_key, self.remote_public_key)
        # Send encrypted_session_key to the remote party
        # Again, this is a placeholder and should be replaced with actual sending logic

    def encrypt_wrapper(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not self.session_key:
                await self._exchange_keys()

            data = args[0] if args else kwargs.get('data')
            encrypted_data = self.code.encrypt_symmetric(data, self.session_key)

            if asyncio.iscoroutinefunction(func):
                return await func(encrypted_data, *args[1:], **kwargs)
            else:
                return func(encrypted_data, *args[1:], **kwargs)

        return wrapper

    def decrypt_wrapper(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not self.session_key:
                await self._exchange_keys()

            if asyncio.iscoroutinefunction(func):
                encrypted_data = await func(*args, **kwargs)
            else:
                encrypted_data = func(*args, **kwargs)

            decrypted_data = self.code.decrypt_symmetric(encrypted_data, self.session_key)
            return decrypted_data

        return wrapper

    @staticmethod
    def create_encrypted_channel(send_func: Callable, receive_func: Callable) -> dict[str, Callable]:
        e2e = E2EEncryption()
        return {
            'send': e2e.encrypt_wrapper(send_func),
            'receive': e2e.decrypt_wrapper(receive_func)
        }
