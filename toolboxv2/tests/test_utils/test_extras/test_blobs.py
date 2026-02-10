import unittest
import os
import sys
import shutil
import tempfile
import time
import json
import threading
import hashlib
import pickle
import socket
import uuid
from unittest.mock import MagicMock, patch, ANY

from toolboxv2.utils.extras import blobs

# --- IMPORT FIX ---
# Wir müssen sicherstellen, dass die lokalen Module gefunden werden.
# Wenn blobs.py Importe wie "from ..security.cryp" hat, müssen wir diese mocken,
# bevor wir blobs importieren, falls die Ordnerstruktur nicht existiert.

# Mocking der externen Abhängigkeiten (Code, DEVICE_KEY, get_logger),
# damit das Skript auch ohne die gesamte Projektumgebung läuft.
sys.modules['reedsolo'] = MagicMock()  # Optional mocken

mock_security = MagicMock()
mock_cryp = MagicMock()
mock_system = MagicMock()


# Mock Implementation für Code.encrypt/decrypt
class MockCodeImpl:
    @staticmethod
    def encrypt_symmetric(data, key):
        if isinstance(data, str): data = data.encode()
        return b"ENC:" + data

    @staticmethod
    def decrypt_symmetric(data, key, to_str=False):
        if data.startswith(b"ENC:"):
            res = data[4:]
        else:
            res = data
        return res.decode() if to_str else res


mock_cryp.Code = MockCodeImpl
mock_cryp.DEVICE_KEY = lambda: "test_device_key"

# Logger Mock
mock_logger = MagicMock()
mock_system.getting_and_closing_app.get_logger = lambda: mock_logger

# Patching modules into sys.modules
sys.modules['..security'] = mock_security
sys.modules['..security.cryp'] = mock_cryp
sys.modules['..system'] = mock_system
sys.modules['..system.getting_and_closing_app'] = mock_system.getting_and_closing_app

# Jetzt importieren wir blobs (wir gehen davon aus, dass blobs.py im selben Ordner liegt)
# Falls es ein Import-Fehler gibt, muss blobs.py angepasst oder PYTHONPATH gesetzt werden.

# --- HELPER FUNCTIONS ---

def is_server_active(host="127.0.0.1", port=3000):
    """Prüft, ob der Blob-Server erreichbar ist."""
    try:
        sock = socket.create_connection((host, port), timeout=1)
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


SERVER_AVAILABLE = is_server_active()
SERVER_URL = "http://127.0.0.1:3000"


# --- UNIT TESTS (ATOMIC, MOCKED) ---

class TestWatchCallback(unittest.TestCase):
    def test_initialization_and_expiration(self):
        cb_func = MagicMock()
        wc = blobs.WatchCallback(callback=cb_func, blob_id="test", max_idle_timeout=1)

        self.assertFalse(wc.is_expired())

        # Warte bis Timeout
        time.sleep(1.1)
        self.assertTrue(wc.is_expired())

    def test_update_timestamp(self):
        wc = blobs.WatchCallback(callback=MagicMock(), blob_id="test", max_idle_timeout=10)
        old_time = wc.last_update
        time.sleep(0.1)
        wc.update_timestamp()
        self.assertGreater(wc.last_update, old_time)


class TestWatchManager(unittest.TestCase):
    def setUp(self):
        self.mock_storage = MagicMock()
        self.manager = blobs.WatchManager(self.mock_storage)

    def tearDown(self):
        self.manager.remove_all_watches()

    def test_add_and_remove_watch(self):
        cb = MagicMock()
        self.manager.add_watch("blob1", cb)

        self.assertIn("blob1", self.manager._watches)
        self.assertEqual(len(self.manager._watches["blob1"]), 1)
        self.assertTrue(self.manager._running)  # Thread sollte gestartet sein

        self.manager.remove_watch("blob1", cb)
        self.assertNotIn("blob1", self.manager._watches)

    def test_dispatch_callbacks(self):
        """Testet, ob Callbacks gefeuert werden, ohne den Thread zu nutzen."""
        cb = MagicMock()
        self.manager.add_watch("blob1", cb)

        # Mock BlobFile Erstellung, da _dispatch_callbacks versucht BlobFile zu lesen
        with patch('toolboxv2.utils.extras.blobs.BlobFile') as MockBlobFile:
            self.manager._dispatch_callbacks("blob1")

            cb.assert_called_once()
            args = cb.call_args[0]
            self.assertIsInstance(args[0], MagicMock)  # Das MockBlobFile Objekt

    def test_cleanup_expired(self):
        cb = MagicMock()
        # Timeout auf 0 setzen damit es sofort abläuft
        self.manager.add_watch("blob1", cb, max_idle_timeout=0)
        time.sleep(0.1)  # Sicherstellen, dass Zeit vergangen ist

        self.manager._cleanup_expired_callbacks()
        self.assertNotIn("blob1", self.manager._watches)


if __name__ == '__main__':
    # Verbose Output
    unittest.main(verbosity=2)
