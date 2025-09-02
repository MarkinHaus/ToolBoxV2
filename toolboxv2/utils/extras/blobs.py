# file: blobs.py
import bisect
import hashlib
import io
import json
import os
import pickle
import random
import time
from pathlib import Path

import requests
import yaml

# These are assumed to exist from your project structure
from ..security.cryp import Code
from ..system.getting_and_closing_app import get_logger


class ConsistentHashRing:
    """
    A consistent hash ring implementation to map keys (blob_ids) to nodes (servers).
    It uses virtual nodes (replicas) to ensure a more uniform distribution of keys.
    """
    def __init__(self, replicas=100):
        """
        :param replicas: The number of virtual nodes for each physical node.
                         Higher values lead to more balanced distribution.
        """
        self.replicas = replicas
        self._keys = []  # Sorted list of hash values (the ring)
        self._nodes = {} # Maps hash values to physical node URLs

    def _hash(self, key: str) -> int:
        """Hashes a key to an integer using md5 for speed and distribution."""
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)

    def add_node(self, node: str):
        """Adds a physical node to the hash ring."""
        for i in range(self.replicas):
            vnode_key = f"{node}:{i}"
            h = self._hash(vnode_key)
            bisect.insort(self._keys, h)
            self._nodes[h] = node

    def get_nodes_for_key(self, key: str) -> list[str]:
        """
        Returns an ordered list of nodes responsible for the given key.
        The first node in the list is the primary, the rest are failover candidates
        in preferential order.
        """
        if not self._nodes:
            return []

        h = self._hash(key)
        start_idx = bisect.bisect_left(self._keys, h)

        # Collect unique physical nodes by iterating around the ring
        found_nodes = []
        for i in range(len(self._keys)):
            idx = (start_idx + i) % len(self._keys)
            node_hash = self._keys[idx]
            physical_node = self._nodes[node_hash]
            if physical_node not in found_nodes:
                found_nodes.append(physical_node)
            # Stop when we have found all unique physical nodes
            if len(found_nodes) == len(set(self._nodes.values())):
                break
        return found_nodes


class BlobStorage:
    """
    A production-ready client for the distributed blob storage server.
    It handles communication with a list of server instances, manages a local cache,
    and implements backoff/retry logic for resilience.
    """

    def __init__(self, servers: list[str], storage_directory: str = './.data/blob_cache'):


        self.servers = servers
        self.session = requests.Session()
        self.storage_directory = storage_directory
        self.blob_ids = []
        os.makedirs(storage_directory, exist_ok=True)

        # Initialize the consistent hash ring
        self.hash_ring = ConsistentHashRing()
        for server in self.servers:
            self.hash_ring.add_node(server)

    def _make_request(self, method, endpoint, blob_id: str = None, max_retries=2, **kwargs):
        """
        Makes a resilient HTTP request to the server cluster.
        - If a blob_id is provided, it uses the consistent hash ring to find the
          primary server and subsequent backup servers in a predictable order.
        - If no blob_id is given (e.g., for broadcast actions), it tries servers randomly.
        - Implements exponential backoff on server errors.
        """
        if not self.servers:
            res = requests.Response()
            res.status_code = 503
            res.reason = "No servers available"
            return res

        if blob_id:
            # Get the ordered list of servers for this specific blob
            preferred_servers = self.hash_ring.get_nodes_for_key(blob_id)
        else:
            # For non-specific requests, shuffle all servers
            preferred_servers = random.sample(self.servers, len(self.servers))

        last_error = None
        for attempt in range(max_retries):
            for server in preferred_servers:
                url = f"{server.rstrip('/')}{endpoint}"
                try:
                    # In a targeted request, print which server we are trying
                    response = self.session.request(method, url, timeout=10, **kwargs)

                    if 500 <= response.status_code < 600:
                        get_logger().warning(f"Warning: Server {server} returned status {response.status_code}. Retrying...")
                        continue
                    response.raise_for_status()
                    return response
                except requests.exceptions.RequestException as e:
                    last_error = e
                    get_logger().warning(f"Warning: Could not connect to server {server}: {e}. Trying next server.")

            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt*0.1)
                get_logger().warning(f"Warning: All preferred servers failed. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                if len(preferred_servers) == 1 and len(self.servers) > 1:
                    preferred_servers = random.sample(self.servers, len(self.servers))

        raise ConnectionError(f"Failed to execute request after {max_retries} attempts. Last error: {last_error}")


    def create_blob(self, data: bytes, blob_id=None) -> str:
        """
        Creates a new blob. The blob_id is calculated client-side by hashing
        the content, and the data is sent to the correct server determined
        by the consistent hash ring. This uses a PUT request, making creation
        idempotent.
        """
        # The blob ID is the hash of its content, ensuring content-addressable storage.
        if not blob_id:
            blob_id = hashlib.sha256(data).hexdigest()

        # Use PUT, as we now know the blob's final ID/URL.
        # Pass blob_id to _make_request so it uses the hash ring.
        print(f"Creating blob {blob_id} on {self._make_request('PUT', f'/blob/{blob_id}',blob_id=blob_id, data=data).status_code}")
        # blob_id = response.text
        self._save_blob_to_cache(blob_id, data)
        return blob_id

    def read_blob(self, blob_id: str) -> bytes:
        cached_data = self._load_blob_from_cache(blob_id)
        if cached_data is not None:
            return cached_data

        get_logger().info(f"Info: Blob '{blob_id}' not in cache, fetching from network.")
        # Pass blob_id to _make_request to target the correct server(s).
        response = self._make_request('GET', f'/blob/{blob_id}', blob_id=blob_id)

        blob_data = response.content
        self._save_blob_to_cache(blob_id, blob_data)
        return blob_data

    def update_blob(self, blob_id: str, data: bytes):
        # Pass blob_id to _make_request to target the correct server(s).
        response = self._make_request('PUT', f'/blob/{blob_id}', blob_id=blob_id, data=data)
        self._save_blob_to_cache(blob_id, data)
        return response

    def delete_blob(self, blob_id: str):
        # Pass blob_id to _make_request to target the correct server(s).
        self._make_request('DELETE', f'/blob/{blob_id}', blob_id=blob_id)
        cache_file = self._get_blob_cache_filename(blob_id)
        if os.path.exists(cache_file):
            os.remove(cache_file)

    # NOTE: share_blobs and recover_blob are coordination endpoints. They do not
    # act on a single blob, so they will continue to use the non-targeted (random)
    # request mode to contact any available server to act as a coordinator.
    def share_blobs(self, blob_ids: list[str]):
        get_logger().info(f"Info: Instructing a server to share blobs for recovery: {blob_ids}")
        payload = {"blob_ids": blob_ids}
        # No blob_id passed, will try any server as a coordinator.
        self._make_request('POST', '/share', json=payload)
        get_logger().info("Info: Sharing command sent successfully.")

    def recover_blob(self, lost_blob_id: str) -> bytes:
        get_logger().info(f"Info: Attempting to recover '{lost_blob_id}' from the cluster.")
        payload = {"blob_id": lost_blob_id}
        # No blob_id passed, recovery can be initiated by any server.
        response = self._make_request('POST', '/recover', json=payload)

        recovered_data = response.content
        get_logger().info(f"Info: Successfully recovered blob '{lost_blob_id}'.")
        self._save_blob_to_cache(lost_blob_id, recovered_data)
        return recovered_data

    def _get_blob_cache_filename(self, blob_id: str) -> str:
        return os.path.join(self.storage_directory, blob_id + '.blobcache')

    def _save_blob_to_cache(self, blob_id: str, data: bytes):
        if blob_id not in self.blob_ids:
            self.blob_ids.append(blob_id)
        with open(self._get_blob_cache_filename(blob_id), 'wb') as f:
            f.write(data)

    def _load_blob_from_cache(self, blob_id: str) -> bytes | None:
        cache_file = self._get_blob_cache_filename(blob_id)
        if not os.path.exists(cache_file):
            return None
        with open(cache_file, 'rb') as f:
            return f.read()

    def exit(self):
        if len(self.blob_ids) < 5:
            return
        for _i in range(len(self.servers)//2+1):
            self.share_blobs(self.blob_ids)


# The BlobFile interface remains unchanged as it's a high-level abstraction
class BlobFile(io.IOBase):
    def __init__(self, filename: str, mode: str = 'r', storage: BlobStorage = None, key: str = None,
                 servers: list[str] = None):
        if not isinstance(filename, str) or not filename:
            raise ValueError("Filename must be a non-empty string.")
        if not filename.startswith('/'): filename = '/' + filename
        self.filename = filename.lstrip('/\\')
        self.blob_id, self.folder, self.datei = self._path_splitter(self.filename)
        self.mode = mode

        if storage is None:
            # In a real app, dependency injection or a global factory would be better
            # but this provides a fallback for simple scripts.
            if not servers:
                from toolboxv2 import get_app
                storage = get_app(from_="BlobStorage").root_blob_storage
            else:
                storage = BlobStorage(servers=servers)

        self.storage = storage
        self.data_buffer = b""
        self.key = key
        if key:
            try:
                assert Code.decrypt_symmetric(Code.encrypt_symmetric(b"test", key), key, to_str=False) == b"test"
            except Exception:
                raise ValueError("Invalid symmetric key provided.")

    @staticmethod
    def _path_splitter(filename):
        parts = Path(filename).parts
        if not parts: raise ValueError("Filename cannot be empty.")
        blob_id = parts[0]
        if len(parts) == 1: raise ValueError("Filename must include a path within the blob, e.g., 'blob_id/file.txt'")
        datei = parts[-1]
        folder = '|'.join(parts[1:-1])
        return blob_id, folder, datei

    def create(self):
        self.storage.create_blob(pickle.dumps({}), self.blob_id)
        return self

    def __enter__(self):
        try:
            raw_blob_data = self.storage.read_blob(self.blob_id)
            blob_content = pickle.loads(raw_blob_data)
        except (requests.exceptions.HTTPError, EOFError, pickle.UnpicklingError, ConnectionError) as e:
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                blob_content = {}  # Blob doesn't exist yet, treat as empty
            elif isinstance(e, EOFError | pickle.UnpicklingError):
                blob_content = {}  # Blob is empty or corrupt, treat as empty for writing
            else:
                self.storage.create_blob(blob_id=self.blob_id, data=pickle.dumps({}))
                blob_content = {}

        if 'r' in self.mode:
            path_key = self.folder if self.folder else self.datei
            if self.folder:
                file_data = blob_content.get(self.folder, {}).get(self.datei)
            else:
                file_data = blob_content.get(self.datei)

            if file_data:
                self.data_buffer = file_data
                if self.key:
                    self.data_buffer = Code.decrypt_symmetric(self.data_buffer, self.key, to_str=False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if 'w' in self.mode:
            final_data = self.data_buffer
            if self.key:
                final_data = Code.encrypt_symmetric(final_data, self.key)

            try:
                raw_blob_data = self.storage.read_blob(self.blob_id)
                blob_content = pickle.loads(raw_blob_data)
            except Exception:
                blob_content = {}

            # Safely navigate and create path
            current_level = blob_content
            if self.folder:
                if self.folder not in current_level:
                    current_level[self.folder] = {}
                current_level = current_level[self.folder]

            current_level[self.datei] = final_data
            self.storage.update_blob(self.blob_id, pickle.dumps(blob_content))




    def exists(self) -> bool:
        """
        Checks if the specific file path exists within the blob without reading its content.
        This is an efficient, read-only operation.

        Returns:
            bool: True if the file exists within the blob, False otherwise.
        """
        try:
            # Fetch the raw blob data. This leverages the local cache if available.
            raw_blob_data = self.storage.read_blob(self.blob_id)
            # Unpickle the directory structure.
            if raw_blob_data:
                blob_content = pickle.loads(raw_blob_data)
            else:
                return False
        except (requests.exceptions.HTTPError, EOFError, pickle.UnpicklingError, ConnectionError):
            # If the blob itself doesn't exist, is empty, or can't be reached,
            # then the file within it cannot exist.
            return False

        # Navigate the dictionary to check for the file's existence.
        current_level = blob_content
        if self.folder:
            if self.folder not in current_level:
                return False
            current_level = current_level[self.folder]

        return self.datei in current_level

    def clear(self):
        self.data_buffer = b''

    def write(self, data):
        if 'w' not in self.mode: raise OSError("File not opened in write mode.")
        if isinstance(data, str):
            self.data_buffer += data.encode()
        elif isinstance(data, bytes):
            self.data_buffer += data
        else:
            raise TypeError("write() argument must be str or bytes")

    def read(self):
        if 'r' not in self.mode: raise OSError("File not opened in read mode.")
        return self.data_buffer

    def read_json(self):
        if 'r' not in self.mode: raise ValueError("File not opened in read mode.")
        if self.data_buffer == b"": return {}
        return json.loads(self.data_buffer.decode())

    def write_json(self, data):
        if 'w' not in self.mode: raise ValueError("File not opened in write mode.")
        self.data_buffer += json.dumps(data).encode()

    def read_pickle(self):
        if 'r' not in self.mode: raise ValueError("File not opened in read mode.")
        if self.data_buffer == b"": return {}
        return pickle.loads(self.data_buffer)

    def write_pickle(self, data):
        if 'w' not in self.mode: raise ValueError("File not opened in write mode.")
        self.data_buffer += pickle.dumps(data)

    def read_yaml(self):
        if 'r' not in self.mode: raise ValueError("File not opened in read mode.")
        if self.data_buffer == b"": return {}
        return yaml.safe_load(self.data_buffer)

    def write_yaml(self, data):
        if 'w' not in self.mode: raise ValueError("File not opened in write mode.")
        yaml.dump(data, self)

