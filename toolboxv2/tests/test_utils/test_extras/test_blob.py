import pickle
import unittest
from unittest.mock import MagicMock, patch

from toolboxv2 import Code
from toolboxv2.utils.extras.blobs import BlobFile, BlobStorage, ConsistentHashRing
# file: test_blobs.py
import unittest
import os
import pickle
import random
import hashlib
import tempfile
import shutil
from unittest.mock import patch, MagicMock, call, ANY

import requests

# --- Mock Dependencies ---
# These mock classes and functions are used to isolate the code under test
# from external systems like the network, filesystem, or logging.

class MockResponse:
    """A mock for requests.Response to simulate server responses."""
    def __init__(self, content=b"", status_code=200, reason="OK"):
        self.content = content
        self.status_code = status_code
        self.reason = reason
        self.text = content.decode('utf-8') if isinstance(content, bytes) else content

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise requests.exceptions.HTTPError(f"{self.status_code} {self.reason}", response=self)

# Mocking the Code class dependency for encryption/decryption
class MockCode:
    """A mock for the Code dependency to simulate encryption."""
    @staticmethod
    def generate_symmetric_key():
        return "a_very_secret_key"

    @staticmethod
    def encrypt_symmetric(data, key):
        return b"encrypted:" + data

    @staticmethod
    def decrypt_symmetric(data, key, to_str=False):
        decrypted = data.replace(b"encrypted:", b"")
        return decrypted.decode() if to_str else decrypted

# --- Test Cases ---

class TestConsistentHashRing(unittest.TestCase):
    """Tests the ConsistentHashRing implementation for node distribution."""

    def test_add_node(self):
        """Verify that adding a node correctly populates the ring."""
        ring = ConsistentHashRing(replicas=50)
        ring.add_node("http://server1")
        self.assertEqual(len(ring._keys), 50)
        self.assertEqual(len(ring._nodes), 50)
        self.assertTrue(all(v == "http://server1" for v in ring._nodes.values()))

    def test_get_nodes_for_key_single_node(self):
        """Test that with one node, it's always returned."""
        ring = ConsistentHashRing()
        ring.add_node("http://server1")
        nodes = ring.get_nodes_for_key("my_blob_id")
        self.assertEqual(nodes, ["http://server1"])

    def test_get_nodes_for_key_no_nodes(self):
        """Test that an empty list is returned if the ring has no nodes."""
        ring = ConsistentHashRing()
        self.assertEqual(ring.get_nodes_for_key("any_key"), [])

    def test_get_nodes_for_key_multiple_nodes_is_deterministic(self):
        """Test that the node list is ordered and deterministic for a given key."""
        ring = ConsistentHashRing()
        servers = ["http://server1", "http://server2", "http://server3"]
        for s in servers:
            ring.add_node(s)

        nodes1 = ring.get_nodes_for_key("my_blob_id")
        nodes2 = ring.get_nodes_for_key("my_blob_id")

        self.assertEqual(nodes1, nodes2)
        self.assertEqual(len(nodes1), 3)
        self.assertCountEqual(nodes1, servers) # Checks that all servers are present

    def test_distribution_is_reasonably_even(self):
        """Check if keys are distributed across nodes, not all mapped to one."""
        servers = [f"http://server{i}" for i in range(5)]
        ring = ConsistentHashRing(replicas=10)
        for s in servers:
            ring.add_node(s)

        counts = {s: 0 for s in servers}
        for i in range(1000):
            blob_id = f"blob_{i}"
            primary_node = ring.get_nodes_for_key(blob_id)[0]
            counts[primary_node] += 1

        # Check that every server got at least one key.
        # With 1000 keys and 5 servers, it's statistically impossible for one to get 0.
        self.assertTrue(all(count > 0 for count in counts.values()))
        # A perfect distribution would be 200 per server. We check it's not wildly skewed.
        self.assertTrue(all(count > 100 for count in counts.values()))


@patch('toolboxv2.utils.extras.blobs.get_logger', MagicMock())
@patch('time.sleep', MagicMock())
class TestBlobStorage(unittest.TestCase):
    """Tests the BlobStorage client for network, cache, and failover logic."""

    def setUp(self):
        """Set up a temporary directory for caching and mock servers."""
        self.test_dir = tempfile.mkdtemp()
        self.servers = ["http://server1:8000", "http://server2:8000", "http://server3:8000"]
        self.storage = BlobStorage(servers=self.servers, storage_directory=self.test_dir)

        # Mock the requests session for all tests in this class
        self.session_patcher = patch('requests.Session')
        self.mock_session = self.session_patcher.start()
        self.mock_request = self.mock_session.return_value.request
        # This allows us to use self.storage.session as the mocked object
        self.storage.session = self.mock_session.return_value

    def tearDown(self):
        """Clean up the temporary directory and stop patches."""
        shutil.rmtree(self.test_dir)
        self.session_patcher.stop()

    def test_create_blob_success(self):
        """Test successfully creating a blob and caching it."""
        data = b"This is my blob content."
        blob_id = hashlib.sha256(data).hexdigest()
        self.mock_request.return_value = MockResponse(content=blob_id.encode(), status_code=200)

        result_id = self.storage.create_blob(data)

        self.assertEqual(result_id, blob_id)
        # Verify the network call
        args, kwargs = self.mock_request.call_args

        # Standardprüfungen
        self.assertEqual(args[0], 'PUT')
        self.assertEqual(kwargs['timeout'], 10)
        self.assertEqual(kwargs['data'], b'This is my blob content.')

        # Prüfen, ob die URL korrekt auf blob_id endet
        self.assertTrue(args[1].endswith(f"/{blob_id}"), f"URL {args[1]} endet nicht auf /{blob_id}")

        # Verify it was cached
        cached_path = os.path.join(self.test_dir, blob_id + '.blobcache')
        self.assertTrue(os.path.exists(cached_path))
        with open(cached_path, 'rb') as f:
            self.assertEqual(f.read(), data)

    def test_read_blob_from_cache(self):
        """Test that reading an existing blob uses the cache and avoids network calls."""
        blob_id = "cached_blob"
        data = b"i am in the cache"
        with open(os.path.join(self.test_dir, blob_id + '.blobcache'), 'wb') as f:
            f.write(data)

        result = self.storage.read_blob(blob_id)

        self.assertEqual(result, data)
        # Assert no network call was made
        self.mock_request.assert_not_called()

    def test_read_blob_from_network_and_caches(self):
        """Test reading a blob from the network when not in cache."""
        blob_id = "network_blob"
        data = b"i came from the network"
        self.mock_request.return_value = MockResponse(content=data, status_code=200)

        result = self.storage.read_blob(blob_id)

        self.assertEqual(result, data)
        # Verify the network call
        args, kwargs = self.mock_request.call_args

        self.assertEqual(args[0], 'GET')
        self.assertEqual(kwargs['timeout'], 10)
        self.assertTrue(args[1].endswith(f"/{blob_id}"), f"URL {args[1]} endet nicht auf /{blob_id}")

        # Verify it is now cached
        cached_path = os.path.join(self.test_dir, blob_id + '.blobcache')
        self.assertTrue(os.path.exists(cached_path))
        with open(cached_path, 'rb') as f:
            self.assertEqual(f.read(), data)

    def test_delete_blob_and_removes_from_cache(self):
        """Test that deleting a blob also removes its cached version."""
        blob_id = "to_be_deleted"
        cache_file = os.path.join(self.test_dir, blob_id + '.blobcache')
        with open(cache_file, 'wb') as f:
            f.write(b"delete me")
        self.assertTrue(os.path.exists(cache_file))

        self.mock_request.return_value = MockResponse(status_code=204)
        self.storage.delete_blob(blob_id)

        args, kwargs = self.mock_request.call_args

        self.assertEqual(args[0], 'DELETE')
        self.assertEqual(kwargs['timeout'], 10)
        self.assertTrue(args[1].endswith(f"/{blob_id}"), f"URL {args[1]} endet nicht auf /{blob_id}")

        self.assertFalse(os.path.exists(cache_file))

    def test_request_failover_and_success(self):
        """Test the retry logic: first server fails, second succeeds."""
        blob_id = "failover_test"
        # The consistent hash ring will determine the order. Let's mock it to be predictable.
        with patch.object(self.storage.hash_ring, 'get_nodes_for_key', return_value=self.servers):
            # First server call raises an error, second returns a success response.
            self.mock_request.side_effect = [
                requests.exceptions.ConnectionError("Connection timed out"),
                MockResponse(content=b"success", status_code=200)
            ]

            response = self.storage.read_blob(blob_id)
            self.assertEqual(response, b"success")

            # Verify it tried two servers
            expected_calls = [
                call('GET', f"{self.servers[0]}/blob/{blob_id}", timeout=10),
                call('GET', f"{self.servers[1]}/blob/{blob_id}", timeout=10)
            ]
            self.mock_request.assert_has_calls(expected_calls)
            self.assertEqual(self.mock_request.call_count, 2)

    def test_request_all_servers_fail(self):
        """Test that a ConnectionError is raised after all retries fail."""
        blob_id = "total_failure"
        self.mock_request.side_effect = requests.exceptions.RequestException("All failed")

        with self.assertRaises(ConnectionError) as cm:
            self.storage.read_blob(blob_id)

        self.assertIn("Failed to execute request after 2 attempts", str(cm.exception))
        # It tries all 3 servers on the first attempt, then all 3 again on the retry.
        self.assertEqual(self.mock_request.call_count, len(self.servers) * 2)

    @patch('random.sample')
    def test_share_blobs_uses_random_server(self, mock_random_sample):
        """Verify that coordination tasks like 'share' do not use the hash ring."""
        mock_random_sample.return_value = self.servers
        self.mock_request.return_value = MockResponse(status_code=200)
        blob_ids_to_share = ["id1", "id2"]

        self.storage.share_blobs(blob_ids_to_share)

        # Verify it used a random server and not the hash ring
        mock_random_sample.assert_called_once_with(self.servers, len(self.servers))
        args, kwargs = self.mock_request.call_args

        self.assertEqual(args[0], 'POST')
        self.assertEqual(kwargs['timeout'], 10)
        self.assertTrue(args[1].endswith("/share"), f"URL {args[1]} endet nicht auf /share")
        self.assertEqual(kwargs['json'], {"blob_ids": blob_ids_to_share})


@patch('toolboxv2.utils.extras.blobs.Code', MockCode)
class TestBlobFile(unittest.TestCase):
    """Tests the BlobFile interface which acts upon a BlobStorage instance."""

    def setUp(self):
        """Set up a mock BlobStorage and a default BlobFile instance."""
        self.mock_storage = MagicMock(spec=BlobStorage)
        self.blob_id = "my_blob_id"
        self.folder = "config"
        self.datei = "settings.json"
        self.filename = f"{self.blob_id}/{self.folder}/{self.datei}"
        self.key = MockCode.generate_symmetric_key()

    def test_path_splitter(self):
        """Verify that paths are correctly parsed into blob_id, folder, and datei."""
        blob_id, folder, datei = BlobFile._path_splitter("blob1/file.txt")
        self.assertEqual(blob_id, "blob1")
        self.assertEqual(folder, "")
        self.assertEqual(datei, "file.txt")

        blob_id, folder, datei = BlobFile._path_splitter("blob2/folder/file.txt")
        self.assertEqual(blob_id, "blob2")
        self.assertEqual(folder, "folder")
        self.assertEqual(datei, "file.txt")

        blob_id, folder, datei = BlobFile._path_splitter("blob3/f1/f2/f3/file.txt")
        self.assertEqual(blob_id, "blob3")
        self.assertEqual(folder, "f1|f2|f3")
        self.assertEqual(datei, "file.txt")

        with self.assertRaises(ValueError):
            BlobFile._path_splitter("just_a_blob_id")

    def test_write_and_exit(self):
        """Test writing data to a file and the __exit__ method saving it."""
        # Blob is initially empty
        self.mock_storage.read_blob.return_value = pickle.dumps({})
        file_content = b"hello world"

        with BlobFile(self.filename, 'w', storage=self.mock_storage) as bf:
            bf.write(file_content)

        # On __exit__, the data should be structured and saved
        expected_blob_content = {
            self.folder: {
                self.datei: file_content
            }
        }
        self.mock_storage.update_blob.assert_called_once_with(
            self.blob_id, pickle.dumps(expected_blob_content)
        )

    def test_write_with_encryption(self):
        """Test that data is encrypted before being written to the blob."""
        self.mock_storage.read_blob.return_value = pickle.dumps({})
        file_content = b"secret data"
        encrypted_content = MockCode.encrypt_symmetric(file_content, self.key)

        with BlobFile(self.filename, 'w', storage=self.mock_storage, key=self.key) as bf:
            bf.write(file_content)

        expected_blob_content = {
            self.folder: {
                self.datei: encrypted_content
            }
        }
        self.mock_storage.update_blob.assert_called_once_with(
            self.blob_id, pickle.dumps(expected_blob_content)
        )

    def test_read_and_enter(self):
        """Test reading data from an existing blob file."""
        file_content = b"here is my data"
        blob_content = {self.folder: {self.datei: file_content}}
        self.mock_storage.read_blob.return_value = pickle.dumps(blob_content)

        with BlobFile(self.filename, 'r', storage=self.mock_storage) as bf:
            self.assertEqual(bf.read(), file_content)

        self.mock_storage.read_blob.assert_called_once_with(self.blob_id)

    def test_read_with_decryption(self):
        """Test that data is decrypted after being read from the blob."""
        file_content = b"top secret info"
        encrypted_content = MockCode.encrypt_symmetric(file_content, self.key)
        blob_content = {self.folder: {self.datei: encrypted_content}}
        self.mock_storage.read_blob.return_value = pickle.dumps(blob_content)

        with BlobFile(self.filename, 'r', storage=self.mock_storage, key=self.key) as bf:
            self.assertEqual(bf.read(), file_content)

    def test_enter_on_nonexistent_blob(self):
        """Test that opening a file from a non-existent blob results in an empty file."""
        # Simulate the server returning 404 Not Found
        self.mock_storage.read_blob.side_effect = requests.exceptions.HTTPError(
            "404 Not Found", response=MockResponse(status_code=404)
        )

        with BlobFile(self.filename, 'r', storage=self.mock_storage) as bf:
            self.assertEqual(bf.read(), b"") # Should be empty, not raise an error

    def test_exists_method(self):
        """Test the 'exists' method for various scenarios."""
        file_content = b"i exist"
        blob_content = {self.folder: {self.datei: file_content}}
        self.mock_storage.read_blob.return_value = pickle.dumps(blob_content)

        bf = BlobFile(self.filename, 'r', storage=self.mock_storage)
        self.assertTrue(bf.exists())

        bf_bad_path = BlobFile(f"{self.blob_id}/{self.folder}/wrong_file.txt", 'r', storage=self.mock_storage)
        self.assertFalse(bf_bad_path.exists())

        # Test when the blob itself doesn't exist
        self.mock_storage.read_blob.side_effect = requests.exceptions.HTTPError(
            "404 Not Found", response=MockResponse(status_code=404)
        )
        bf_bad_blob = BlobFile("nonexistent_blob/some/file.txt", 'r', storage=self.mock_storage)
        self.assertFalse(bf_bad_blob.exists())

if __name__ == '__main__':
    unittest.main(verbosity=2)
