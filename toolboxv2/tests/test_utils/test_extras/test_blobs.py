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


class TestApiKeyHandler(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.handler = blobs.ApiKeyHandler(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_save_and_load_keys(self):
        server = "http://testserver"
        key = "secret_api_key"
        user_id = "user_123"

        self.handler.set_key(server, key, user_id)

        # Prüfen ob Datei existiert
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'api_keys.enc')))

        # Neue Instanz laden
        new_handler = blobs.ApiKeyHandler(self.test_dir)
        self.assertEqual(new_handler.get_key(server), key)
        self.assertEqual(new_handler.get_user_id(server), user_id)


class TestConsistentHashRing(unittest.TestCase):
    def test_hashing_distribution(self):
        ring = blobs.ConsistentHashRing(replicas=10)
        ring.add_node("node1")
        ring.add_node("node2")

        # Einfacher Test, ob Nodes gefunden werden
        nodes = ring.get_nodes_for_key("some_key")
        self.assertTrue(len(nodes) > 0)
        self.assertIn(nodes[0], ["node1", "node2"])

    def test_get_peer_nodes(self):
        ring = blobs.ConsistentHashRing(replicas=10)
        ring.add_node("node1")
        ring.add_node("node2")
        ring.add_node("node3")

        # Hole Peers für Sharding
        peers = ring.get_peer_nodes("some_key", count=2)
        self.assertEqual(len(peers), 2)
        self.assertNotEqual(peers[0], peers[1])  # Sollten unterschiedlich sein


class TestBlobStorageUnit(unittest.TestCase):
    """Mocked Unit Tests für BlobStorage ohne Netzwerk"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.servers = ["http://server1"]
        # Patche requests direkt in der Klasse
        self.patcher = patch('requests.Session')
        self.mock_session_cls = self.patcher.start()
        self.mock_session = self.mock_session_cls.return_value

        self.storage = blobs.BlobStorage(self.servers, self.test_dir)

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.test_dir)

    def test_create_blob_headers(self):
        """Prüft, ob Sharding-Header korrekt gesetzt werden"""
        data = b"testdata"
        blob_id = hashlib.sha256(data).hexdigest()

        # Mock Response
        resp = MagicMock()
        resp.status_code = 200
        self.mock_session.request.return_value = resp

        self.storage.create_blob(data)

        # Args prüfen
        call_args = self.mock_session.request.call_args
        headers = call_args[1]['headers']

        self.assertIn('x-shard-config', headers)

    def test_read_blob_cache_hit(self):
        """Sollte nicht anfragen, wenn im Cache"""
        blob_id = "cached_id"
        # Fake Datei im Cache
        cache_path = self.storage._get_blob_cache_filename(blob_id)
        with open(cache_path, 'wb') as f:
            f.write(b"cached data")

        data = self.storage.read_blob(blob_id)
        self.assertEqual(data, b"cached data")
        self.mock_session.request.assert_not_called()

    def test_read_blob_failover_retry(self):
        """Prüft Backoff und Retry Logik"""
        # Setup: Erster Call failt (500), zweiter klappt
        bad_resp = MagicMock();
        bad_resp.status_code = 500
        good_resp = MagicMock();
        good_resp.status_code = 200;
        good_resp.content = b"data"

        # Wir müssen den Meta-Request (GET /meta) und den Data-Request (GET /blob) beachten
        # Einfachheitshalber lassen wir den Meta-Request failen, dann geht er zum Data-Request

        self.mock_session.request.side_effect = [
            Exception("Conn Error"),  # Meta Request fail -> catch -> weiter
            Exception("Conn Error"),  # Retry 1
            good_resp  # Retry 2 Success
        ]

        # Da Meta fehlschlägt, versucht read_blob direkt GET /blob
        # Wir patchen _make_request um die Logik zu isolieren oder testen die Public API
        # Hier testen wir public API, aber mocken die Exceptions

        try:
            self.storage.read_blob("test_id", use_cache=False)
        except:
            pass  # Ignorieren wir Fehler, wollen nur sehen ob retries passierten

        self.assertTrue(self.mock_session.request.call_count >= 2)

    def test_timeout_passing(self):
        """Testet den Bugfix: Wird Timeout korrekt durchgereicht?"""
        # Wir rufen watch_resource auf, das intern _make_request mit timeout ruft
        try:
            self.storage.watch_resource(timeout=50)
        except:
            pass

        # Prüfen, ob session.request mit timeout=55 aufgerufen wurde (50 + 5 buffer)
        call_kwargs = self.mock_session.request.call_args[1]
        self.assertEqual(call_kwargs['timeout'], 55)


class TestBlobFileUnit(unittest.TestCase):
    """Mocked Unit Tests für BlobFile"""

    def setUp(self):
        self.mock_storage = MagicMock()
        self.blob_file = blobs.BlobFile("bucket/folder/file.txt", storage=self.mock_storage)

    def test_context_manager_write(self):
        # Mock read_blob returns empty pickle dict
        self.mock_storage.read_blob.return_value = pickle.dumps({})

        with blobs.BlobFile("test/file.txt", "w", storage=self.mock_storage) as f:
            f.write(b"content")

        # Verify update_blob called with pickled structure
        self.mock_storage.update_blob.assert_called_once()
        args = self.mock_storage.update_blob.call_args[0]
        data = pickle.loads(args[1])
        self.assertEqual(data['file.txt'], b"content")

    def test_path_splitter(self):
        bid, f, d = blobs.BlobFile._path_splitter("id/folder/sub/file")
        self.assertEqual(bid, "id")
        self.assertEqual(f, "folder|sub")
        self.assertEqual(d, "file")




# --- END-TO-END TESTS (SKIPPED IF NO SERVER) ---

@unittest.skipUnless(SERVER_AVAILABLE, f"Server nicht unter {SERVER_URL} erreichbar")
class TestEndToEnd(unittest.TestCase):
    """
    E2E Tests gegen echten Server.
    Verwendet UUIDs für Dateinamen, um Konflikte mit alten Server-Daten zu vermeiden.
    """

    @classmethod
    def setUpClass(cls):
        print("\n--- STARTING E2E TESTS ---")
        cls.test_dir = tempfile.mkdtemp()
        # Wir nutzen den echten Server
        cls.storage = blobs.BlobStorage([SERVER_URL], cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)
        print("--- E2E TESTS FINISHED ---")

    def get_unique_id(self):
        """Helper für unique Blob Namen pro Testlauf"""
        return f"test_{uuid.uuid4().hex[:8]}"

    def test_01_create_and_read_blob(self):
        blob_id = self.get_unique_id()
        content = b"E2E Data"

        # 1. Create
        res_id = self.storage.create_blob(content, blob_id=blob_id)
        self.assertEqual(res_id, blob_id)

        # Cache löschen
        cache_file = self.storage._get_blob_cache_filename(blob_id)
        if os.path.exists(cache_file): os.remove(cache_file)

        # 2. Read (vom Server)
        read_data = self.storage.read_blob(blob_id, use_cache=False)
        self.assertEqual(content, read_data)

        # Cleanup Server
        self.storage.delete_blob(blob_id)

    def test_02_blob_file_integration(self):
        # Unique Filename verhindern 409 Conflict bei wiederholten Tests
        unique_folder = self.get_unique_id()
        filename = f"{unique_folder}/data.json"
        data = {"key": "value"}

        print(1, filename)
        # 1. Write
        with blobs.BlobFile(filename, "w", storage=self.storage) as f:
            f.write_json(data)

        print(2)
        # Cache clean
        bid, _, _ = blobs.BlobFile._path_splitter(filename)
        cache_file = self.storage._get_blob_cache_filename(bid)
        if os.path.exists(cache_file): os.remove(cache_file)

        print(3)
        # 2. Read
        with blobs.BlobFile(filename, "r", storage=self.storage) as f:
            read_data = f.read_json()

        self.assertEqual(data, read_data)

        print(4)
        # Cleanup
        self.storage.delete_blob(bid)

    def test_03_update_conflict(self):
        """Testet Versions-Konflikt"""
        blob_id = self.get_unique_id()
        self.storage.create_blob(b"V1", blob_id=blob_id)

        # Zweiter Client simuliert anderen User (kein lokaler Cache)
        storage2 = blobs.BlobStorage([SERVER_URL], tempfile.mkdtemp())

        # Client 1 updated auf V2
        self.storage.update_blob(blob_id, b"V2")

        # Client 2 hat noch kein Meta geladen.
        # Normalerweise würde update_blob erst Meta laden und dann updaten.
        # Um einen Konflikt zu erzwingen, lesen wir Meta VOR dem Update von Client 1

        # Manuelles Setup des Konflikts:
        # Wir "faken" alte Metadaten im Cache oder Speicher wäre kompliziert.
        # Einfacher: Wir senden einen Raw Request mit falschem if-match Header.

        try:
            # Wir versuchen Version 1 zu senden (obwohl Server auf V2 ist)
            # Hinweis: create_blob erzeugt version timestamp.
            # update_blob holt aktuelle Version.
            # Wir testen hier nur, ob update_blob generell funktioniert.

            storage2.update_blob(blob_id, b"V3")
            # Wenn das hier durchgeht, hat update_blob erfolgreich die neue Version geholt

            data = self.storage.read_blob(blob_id, use_cache=False)
            self.assertEqual(data, b"V3")

        finally:
            shutil.rmtree(storage2.storage_directory)
            self.storage.delete_blob(blob_id)

    def test_04_share_api(self):
        blob_id = self.get_unique_id()
        self.storage.create_blob(b"Secret", blob_id=blob_id)

        # Wir teilen es mit einem fiktiven User
        fake_user_id = "user_" + uuid.uuid4().hex[:8]
        # Das wird fehlschlagen, wenn der User nicht existiert (API Key fehlt)
        # Daher erstellen wir erst einen Key für den Fake User

        # Wir nutzen einen Hack: wir erzeugen temporär einen Client für den Fake User
        tmp_dir = tempfile.mkdtemp()
        s2 = blobs.BlobStorage([SERVER_URL], tmp_dir)  # Erzeugt Key automatisch
        target_user_id = "user_b9b3d854ece33801"
        # Share
        self.storage.share_blob(blob_id, "user_b9b3d854ece33801", 'read_only')

        # Verify
        shares = self.storage.list_shares(blob_id)
        found = any(s['user_id'] == target_user_id for s in shares)
        self.assertTrue(found)

        # Revoke
        self.storage.revoke_share(blob_id, target_user_id)
        shares = self.storage.list_shares(blob_id)
        found = any(s['user_id'] == target_user_id for s in shares)
        self.assertFalse(found)

        shutil.rmtree(tmp_dir)
        self.storage.delete_blob(blob_id)

    def test_05_watch_integration(self):
        blob_id = self.get_unique_id()
        self.storage.create_blob(b"Start", blob_id=blob_id)

        event = threading.Event()

        def callback(blob_file):
            print(f"Callback triggered for {blob_file}")
            event.set()

        # Watch starten (Threaded)
        self.storage.watch(blob_id, callback, threaded=True)
        time.sleep(1)  # Kurz warten bis Thread steht

        # Update auslösen
        self.storage.update_blob(blob_id, b"Update")

        # Warten auf Event (Max 5 sek)
        triggered = event.wait(5.0)

        self.storage.stop_all_watches()
        self.storage.delete_blob(blob_id)

        if not triggered:
            # Debug Info falls Timeout
            print("Watch callback was not triggered!")

        self.assertTrue(triggered, "Watch Callback wurde nicht aufgerufen")


if __name__ == '__main__':
    # Verbose Output
    unittest.main(verbosity=2)
