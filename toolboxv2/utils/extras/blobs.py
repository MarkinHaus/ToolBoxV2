# file: blobs.py
import bisect
import hashlib
import io
import json
import os
import pickle
import platform
import random
import time
import threading
from pathlib import Path
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field

import requests
import yaml

# These are assumed to exist from your project structure
from ..security.cryp import Code, DEVICE_KEY
from ..system.getting_and_closing_app import get_logger
from toolboxv2.utils.singelton_class import Singleton

# Try to import Reed-Solomon library for recovery
try:
    import reedsolo
    HAS_REEDSOLO = True
except ImportError:
    HAS_REEDSOLO = False
    get_logger().warning("reedsolo library not available. Shard recovery will not work.")


@dataclass
class WatchCallback:
    """
    Wrapper for a watch callback with metadata.
    Tracks when the callback was last triggered and manages timeout.
    """
    callback: Callable[['BlobFile'], None]
    blob_id: str
    last_update: float = field(default_factory=time.time)
    max_idle_timeout: int = 600  # 10 minutes default
    folder: Optional[str] = None
    filename: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if this callback has exceeded its idle timeout"""
        return (time.time() - self.last_update) > self.max_idle_timeout

    def update_timestamp(self):
        """Update the last update timestamp"""
        self.last_update = time.time()


class WatchManager:
    """
    Manages watch operations for blob changes.
    Runs in a separate thread and dispatches callbacks when blobs change.
    """
    def __init__(self, storage: 'BlobStorage'):
        self.storage = storage
        self._watches: Dict[str, List[WatchCallback]] = {}  # blob_id -> callbacks
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._running = False

    def add_watch(self, blob_id: str, callback: Callable[['BlobFile'], None],
                  max_idle_timeout: int = 600, **kwargs):
        """
        Register a callback for blob changes.

        Args:
            blob_id: The blob to watch
            callback: Function to call when blob changes, receives BlobFile object
            max_idle_timeout: Seconds of inactivity before auto-removing callback
        """
        with self._lock:
            if blob_id not in self._watches:
                self._watches[blob_id] = []

            watch_cb = WatchCallback(
                callback=callback,
                blob_id=blob_id,
                max_idle_timeout=max_idle_timeout,
                **kwargs
            )
            self._watches[blob_id].append(watch_cb)

            get_logger().info(f"Added watch for blob '{blob_id}' (timeout: {max_idle_timeout}s)")

            # Start watch thread if not running
            if not self._running:
                self._start_watch_thread()

    def remove_watch(self, blob_id: str, callback: Optional[Callable] = None):
        """
        Remove watch callback(s) for a blob.

        Args:
            blob_id: The blob to stop watching
            callback: Specific callback to remove, or None to remove all
        """
        with self._lock:
            if blob_id not in self._watches:
                return

            if callback is None:
                # Remove all callbacks for this blob
                del self._watches[blob_id]
                get_logger().info(f"Removed all watches for blob '{blob_id}'")
            else:
                # Remove specific callback
                self._watches[blob_id] = [
                    w for w in self._watches[blob_id]
                    if w.callback != callback
                ]
                if not self._watches[blob_id]:
                    del self._watches[blob_id]
                get_logger().info(f"Removed specific watch for blob '{blob_id}'")

            # Stop thread if no more watches
            if not self._watches and self._running:
                self._stop_watch_thread()

    def remove_all_watches(self):
        """Remove all watch callbacks and stop the watch thread"""
        with self._lock:
            self._watches.clear()
            get_logger().info("Removed all watches")

        if self._running:
            self._stop_watch_thread()

    def _start_watch_thread(self):
        """Start the background watch thread"""
        if self._running:
            return

        self._stop_event.clear()
        self._running = True
        self._watch_thread = threading.Thread(
            target=self._watch_loop,
            name="BlobWatchThread",
            daemon=True
        )
        self._watch_thread.start()
        get_logger().info("Started watch thread")

    def _stop_watch_thread(self):
        """Stop the background watch thread"""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=5)

        get_logger().info("Stopped watch thread")

    def _watch_loop(self):
        """
        Main watch loop running in background thread.
        Makes long-polling requests and dispatches callbacks.
        """
        while not self._stop_event.is_set():
            try:
                # Check if we have any watches
                with self._lock:
                    if not self._watches:
                        break

                # Make watch request to server (60s timeout on server side)
                result = self.storage.watch_resource(blob_id=None, timeout=60)
                # Handle timeout (no changes)
                if result.get('timeout'):
                    self._cleanup_expired_callbacks()
                    continue

                # Handle errors
                if result.get('error'):
                    get_logger().warning(f"Watch error: {result['error']}")
                    time.sleep(5)  # Back off on error
                    continue

                # Extract changed blob_id from response
                # Server returns just the blob_id string
                changed_blob_id = result if isinstance(result, str) else result.get('blob_id')

                if changed_blob_id and changed_blob_id != 'timeout':
                    self._dispatch_callbacks(changed_blob_id)

                # Cleanup expired callbacks
                self._cleanup_expired_callbacks()

            except Exception as e:
                get_logger().error(f"Watch loop error: {e}")
                time.sleep(5)  # Back off on error

        self._running = False
        get_logger().info("Watch loop exited")

    def _dispatch_callbacks(self, blob_id: str):
        """
        Dispatch callbacks for a changed blob.
        Creates BlobFile object and calls all registered callbacks.
        """
        with self._lock:
            callbacks = self._watches.get(blob_id, []).copy()

        if not callbacks:
            return

        get_logger().info(f"Dispatching {len(callbacks)} callbacks for blob '{blob_id}'")

        # Create BlobFile object for this blob
        try:
            # Use 'r' mode to read the updated blob
            row_data = self.storage.read_blob(blob_id, use_cache=False)
            # Call each callback
            for watch_cb in callbacks:

                if watch_cb.filename:
                    if not watch_cb.folder:
                        watch_cb.folder = "/"
                    if not watch_cb.folder.startswith('/'):
                        watch_cb.folder = '/' + watch_cb.folder

                try:
                    blob_file = BlobFile(blob_id + watch_cb.folder + '/' + watch_cb.filename, 'r', storage=self.storage) if watch_cb.filename else row_data
                    watch_cb.callback(blob_file)
                    watch_cb.update_timestamp()
                except Exception as e:
                    get_logger().error(f"Callback error for blob '{blob_id}': {e}")
                    import traceback
                    get_logger().error(traceback.format_exc())

        except Exception as e:
            get_logger().error(f"Failed to create BlobFile for '{blob_id}': {e}")

    def _cleanup_expired_callbacks(self):
        """Remove callbacks that have exceeded their idle timeout"""
        with self._lock:
            expired_blobs = []

            for blob_id, callbacks in self._watches.items():
                # Filter out expired callbacks
                active_callbacks = [cb for cb in callbacks if not cb.is_expired()]

                if len(active_callbacks) < len(callbacks):
                    removed_count = len(callbacks) - len(active_callbacks)
                    get_logger().info(f"Removed {removed_count} expired callbacks for blob '{blob_id}'")

                if active_callbacks:
                    self._watches[blob_id] = active_callbacks
                else:
                    expired_blobs.append(blob_id)

            # Remove blobs with no active callbacks
            for blob_id in expired_blobs:
                del self._watches[blob_id]
                get_logger().info(f"Removed blob '{blob_id}' from watch list (no active callbacks)")

            # Stop thread if no more watches
            if not self._watches and self._running:
                get_logger().info("No more active watches, stopping watch thread")
                self._stop_event.set()


class ApiKeyHandler(metaclass=Singleton):
    """
    Manages API keys for distributed blob storage servers.
    Keys and user IDs are stored encrypted with the device key.
    """
    def __init__(self, storage_directory: str):
        self.storage_directory = storage_directory
        self.keys_file = os.path.join(storage_directory, 'api_keys.enc')
        self._keys: Dict[str, Dict[str, str]] = {}  # server_url -> {api_key, user_id}
        self._load_keys()

    def _load_keys(self):
        """Load encrypted API keys from disk"""
        if not os.path.exists(self.keys_file):
            return

        try:
            with open(self.keys_file, 'r') as f:
                encrypted_data = f.read()

            if encrypted_data:
                device_key = DEVICE_KEY()
                decrypted = Code.decrypt_symmetric(encrypted_data, device_key)
                loaded = json.loads(decrypted)

                # Handle old format (just api_key string) and new format (dict)
                for server, value in loaded.items():
                    if isinstance(value, str):
                        # Old format: just api_key
                        self._keys[server] = {'api_key': value, 'user_id': None}
                    else:
                        # New format: dict with api_key and user_id
                        self._keys[server] = value

                get_logger().info(f"Loaded {len(self._keys)} API keys from storage")
        except Exception as e:
            get_logger().error(f"Failed to load API keys: {e}")
            self._keys = {}

    def _save_keys(self):
        """Save API keys encrypted to disk"""
        try:
            device_key = DEVICE_KEY()
            data = json.dumps(self._keys)
            encrypted = Code.encrypt_symmetric(data, device_key)

            with open(self.keys_file, 'w') as f:
                f.write(encrypted)
        except Exception as e:
            get_logger().error(f"Failed to save API keys: {e}")

    def get_key(self, server_url: str) -> Optional[str]:
        """Get API key for a server"""
        server_data = self._keys.get(server_url)
        if server_data:
            return server_data.get('api_key')
        return None

    def get_user_id(self, server_url: str) -> Optional[str]:
        """Get User ID for a server"""
        server_data = self._keys.get(server_url)
        if server_data:
            return server_data.get('user_id')
        return None

    def set_key(self, server_url: str, api_key: str, user_id: Optional[str] = None):
        """Set API key and user ID for a server and persist"""
        self._keys[server_url] = {
            'api_key': api_key,
            'user_id': user_id
        }
        self._save_keys()

    def has_key(self, server_url: str) -> bool:
        """Check if we have an API key for this server"""
        return server_url in self._keys


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

    def get_peer_nodes(self, key: str, count: int = 3) -> list[str]:
        """
        Get peer nodes for sharding (excluding primary).
        Returns up to 'count' peer servers for the given key.
        """
        all_nodes = self.get_nodes_for_key(key)
        if len(all_nodes) <= 1:
            return []
        # Return peers (excluding primary which is first)
        return all_nodes[1:count+1]


class BlobStorage:
    """
    A production-ready client for the distributed blob storage server.
    It handles communication with a list of server instances, manages a local cache,
    and implements backoff/retry logic for resilience.

    Now supports:
    - API key authentication
    - Reed-Solomon sharding with peer distribution
    - Optimistic locking with version control
    - Self-healing via shard reconstruction
    """

    def __init__(self, servers: list[str], storage_directory: str = './.data/blob_cache',
                 data_shards: int = 4, parity_shards: int = 2):
        self.servers = servers
        self.session = requests.Session()
        self.storage_directory = storage_directory
        self.blob_ids = []
        self.data_shards = data_shards
        self.parity_shards = parity_shards
        os.makedirs(storage_directory, exist_ok=True)

        # Initialize API key handler
        self.api_key_handler = ApiKeyHandler(storage_directory)

        # Initialize the consistent hash ring
        self.hash_ring = ConsistentHashRing()
        for server in self.servers:
            self.hash_ring.add_node(server)

        # Initialize watch manager
        self.watch_manager = WatchManager(self)

        # Ensure all servers have API keys
        self._ensure_api_keys()

    def _ensure_api_keys(self):
        """Ensure we have API keys for all servers, creating them if needed"""
        for server in self.servers:
            if not self.api_key_handler.has_key(server):
                try:
                    self._create_api_key(server)
                except Exception as e:
                    get_logger().warning(f"Failed to create API key for {server}: {e}")

    def _create_api_key(self, server: str, device_name: Optional[str] = None):
        """Create a new API key on the server with device name for user_id generation"""
        url = f"{server.rstrip('/')}/keys"

        # Get device name if not provided
        if device_name is None:
            device_name = platform.node()

        payload = {"device_name": device_name}

        try:

            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            api_key = data.get('key')
            user_id = data.get('user_id')

            if api_key:
                self.api_key_handler.set_key(server, api_key, user_id)
                get_logger().info(f"Created API key for {server} (user_id: {user_id})")
        except Exception as e:
            get_logger().error(f"Failed to create API key for {server}: {e}")
            raise

    def get_user_id(self, server: Optional[str] = None) -> Optional[str]:
        """
        Get the Public User ID for this client.

        Args:
            server: Specific server to get user_id for, or None for first available

        Returns:
            User ID string (e.g., 'user_abc123') or None
        """
        if server:
            return self.api_key_handler.get_user_id(server)

        # Return first available user_id
        for srv in self.servers:
            user_id = self.api_key_handler.get_user_id(srv)
            if user_id:
                return user_id

        return None

    def _make_request(self, method, endpoint, blob_id: str = None, max_retries=2,
                      include_auth: bool = True, **kwargs):
        """
        Makes a resilient HTTP request to the server cluster.
        """
        if not self.servers:
            res = requests.Response()
            res.status_code = 503
            res.reason = "No servers available"
            return res

        if blob_id:
            preferred_servers = self.hash_ring.get_nodes_for_key(blob_id)
        else:
            preferred_servers = random.sample(self.servers, len(self.servers))

        # FIX: Extrahiere timeout aus kwargs oder nutze Standardwert 10
        request_timeout = kwargs.pop('timeout', 10)

        last_error = None
        for attempt in range(max_retries):
            for server in preferred_servers:
                url = f"{server.rstrip('/')}{endpoint}"

                if 'headers' not in kwargs:
                    kwargs['headers'] = {}

                if include_auth:
                    api_key = self.api_key_handler.get_key(server)
                    if api_key:
                        kwargs['headers']['x-api-key'] = api_key

                try:
                    # FIX: Nutze request_timeout hier
                    response = self.session.request(method, url, timeout=request_timeout, **kwargs)

                    if 500 <= response.status_code < 600:
                        get_logger().warning(
                            f"Warning: Server {server} returned status {response.status_code}. Retrying...")
                        continue

                    if response.status_code == 401 and include_auth:
                        get_logger().warning(f"API key invalid for {server}, creating new one...")
                        raise Exception("API key invalid")
                        #try:
                        #    self._create_api_key(server)
                        #    api_key = self.api_key_handler.get_key(server)
                        #    if api_key:
                        #        kwargs['headers']['x-api-key'] = api_key
                        #        response = self.session.request(method, url, timeout=request_timeout, **kwargs)
                        #except Exception as e:
                        #    get_logger().error(f"Failed to refresh API key: {e}")

                    response.raise_for_status()
                    return response
                except requests.exceptions.RequestException as e:
                    last_error = e
                    get_logger().warning(f"Warning: Could not connect to server {server}: {e}. Trying next server.")

            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt * 0.1)
                get_logger().warning(f"Warning: All preferred servers failed. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                if len(preferred_servers) == 1 and len(self.servers) > 1:
                    preferred_servers = random.sample(self.servers, len(self.servers))

        raise ConnectionError(f"Failed to execute request after {max_retries} attempts. Last error: {last_error}")

    def create_blob(self, data: bytes, blob_id=None) -> str:
        """
        Creates a new blob with Reed-Solomon sharding.
        The blob_id is calculated client-side by hashing the content.
        The server handles shard distribution to peers.
        Sharding configuration is sent via x-shard-config header as JSON.
        """
        # The blob ID is the hash of its content, ensuring content-addressable storage.
        if not blob_id:
            blob_id = hashlib.sha256(data).hexdigest()

        # Get peer servers for sharding
        peers = self.hash_ring.get_peer_nodes(blob_id, count=self.data_shards + self.parity_shards - 1)

        # Build sharding configuration as JSON
        shard_config = {
            'data_shards': self.data_shards,
            'parity_shards': self.parity_shards,
            'peers': peers
        }

        # Send to primary server with sharding config in header
        response = self._make_request(
            'PUT',
            f'/blob/{blob_id}',
            blob_id=blob_id,
            data=data,
            headers={'x-shard-config': json.dumps(shard_config)}
        )

        get_logger().info(f"Created blob {blob_id} with status {response.status_code}")
        self._save_blob_to_cache(blob_id, data)
        return blob_id

    def get_blob_meta(self, blob_id: str) -> dict:
        """Fetch metadata for a blob to retrieve current version and shards."""
        response = self._make_request('GET', f'/blob/{blob_id}/meta', blob_id=blob_id)
        return response.json()

    def update_blob(self, blob_id: str, data: bytes) -> dict:
        """
        Updates an existing blob with new data using optimistic locking.
        """
        # Get current version for optimistic locking
        try:
            meta = self.get_blob_meta(blob_id)
            current_version = meta.get('version', 0)
        except Exception as e:
            # Falls Blob nicht existiert oder Netzwerkfehler, nehmen wir 0 an.
            # Aber wir loggen es, um "AttributeError" (fehlende Methode) zu sehen
            get_logger().debug(f"Could not fetch meta for update: {e}")
            current_version = 0

        peers = self.hash_ring.get_peer_nodes(blob_id, count=self.data_shards + self.parity_shards - 1)

        shard_config = {
            'data_shards': self.data_shards,
            'parity_shards': self.parity_shards,
            'peers': peers
        }

        response = self._make_request(
            'PUT',
            f'/blob/{blob_id}',
            blob_id=blob_id,
            data=data,
            headers={
                'if-match': str(current_version),
                'x-shard-config': json.dumps(shard_config)
            }
        )

        get_logger().info(f"Updated blob {blob_id} with status {response.status_code}")
        self._save_blob_to_cache(blob_id, data)
        return response.json()

    def read_blob(self, blob_id: str, use_cache: bool = True) -> bytes:
        """
        Read blob with metadata-first approach and self-healing.

        1. Check cache if enabled
        2. Fetch metadata to get version and shard locations
        3. Try to read from primary server
        4. If primary fails, attempt shard reconstruction
        """
        # Check cache first
        if use_cache:
            cached_data = self._load_blob_from_cache(blob_id)
            if cached_data is not None:
                return cached_data

        get_logger().info(f"Blob '{blob_id}' not in cache, fetching from network.")

        # Fetch metadata first
        try:
            meta_response = self._make_request('GET', f'/blob/{blob_id}/meta', blob_id=blob_id)
            metadata = meta_response.json()
            version = metadata.get('version')
            shard_locations = metadata.get('shard_locations', [])
            data_shards = metadata.get('data_shards', self.data_shards)
            parity_shards = metadata.get('parity_shards', self.parity_shards)
            original_size = metadata.get('size', 0)
        except Exception as e:
            get_logger().warning(f"Failed to fetch metadata for {blob_id}: {e}")
            metadata = None
            version = None
            shard_locations = []

        # Try to read from primary server
        try:
            response = self._make_request('GET', f'/blob/{blob_id}', blob_id=blob_id)
            blob_data = response.content

            # Validate ETag if we have version info
            if version:
                etag = response.headers.get('ETag')
                if etag and str(version) != etag:
                    get_logger().warning(f"Version mismatch for {blob_id}: expected {version}, got {etag}")

            self._save_blob_to_cache(blob_id, blob_data)
            return blob_data

        except Exception as e:
            get_logger().warning(f"Failed to read blob {blob_id} from primary: {e}")

            # Attempt self-healing via shard reconstruction
            if shard_locations and HAS_REEDSOLO:
                get_logger().info(f"Attempting shard reconstruction for {blob_id}")
                try:
                    reconstructed = self._reconstruct_from_shards(
                        blob_id,
                        shard_locations,
                        data_shards,
                        parity_shards,
                        original_size
                    )
                    if reconstructed:
                        self._save_blob_to_cache(blob_id, reconstructed)
                        return reconstructed
                except Exception as shard_error:
                    get_logger().error(f"Shard reconstruction failed: {shard_error}")

            # Re-raise original error if recovery failed
            raise

    def _reconstruct_from_shards(self, blob_id: str, shard_locations: List[str],
                                  data_shards: int, parity_shards: int, original_size: int) -> Optional[bytes]:
        """
        Reconstruct blob data from shards using Reed-Solomon decoding.
        """
        if not HAS_REEDSOLO:
            get_logger().error("reedsolo library not available for reconstruction")
            return None

        total_shards = data_shards + parity_shards
        shards = [None] * total_shards

        # Fetch shards from peer servers
        for i in range(total_shards):
            for server in shard_locations:
                try:
                    # Try to fetch shard from this server
                    response = self._make_request(
                        'GET',
                        f'/shard/{blob_id}/{i}',
                        blob_id=None,  # Don't use hash ring for shard requests
                        max_retries=1
                    )
                    shards[i] = bytearray(response.content)
                    get_logger().info(f"Fetched shard {i} from {server}")
                    break
                except Exception as e:
                    get_logger().warning(f"Failed to fetch shard {i} from {server}: {e}")
                    continue

        # Check if we have enough shards to reconstruct
        available_shards = sum(1 for s in shards if s is not None)
        if available_shards < data_shards:
            get_logger().error(f"Not enough shards: have {available_shards}, need {data_shards}")
            return None

        # Reconstruct using Reed-Solomon
        try:
            # Create RS codec
            rs = reedsolo.RSCodec(parity_shards)

            # Reconstruct data
            # Note: This is a simplified reconstruction. The actual implementation
            # depends on how the Rust server encodes the shards.
            reconstructed_shards = []
            for shard in shards[:data_shards]:
                if shard:
                    reconstructed_shards.append(bytes(shard))

            # Concatenate data shards
            reconstructed = b''.join(reconstructed_shards)

            # Trim to original size (remove padding)
            reconstructed = reconstructed[:original_size]

            get_logger().info(f"Successfully reconstructed blob {blob_id}")
            return reconstructed

        except Exception as e:
            get_logger().error(f"Reed-Solomon reconstruction failed: {e}")
            return None


    def delete_blob(self, blob_id: str):
        """Delete a blob from the storage"""
        self._make_request('DELETE', f'/blob/{blob_id}', blob_id=blob_id)
        cache_file = self._get_blob_cache_filename(blob_id)
        if os.path.exists(cache_file):
            os.remove(cache_file)

    def watch_resource(self, blob_id: Optional[str] = None, timeout: int = 60):
        """
        Low-level watch method for single long-polling request.

        Args:
            blob_id: Specific blob to watch (Note: server ignores this, watches all)
            timeout: How long to wait for changes (seconds)

        Returns:
            String with blob_id that changed, or 'timeout', or dict with error
        """
        try:
            response = self._make_request(
                'GET',
                '/watch',
                blob_id=None,  # Server doesn't use blob_id parameter
                timeout=timeout + 5  # Add buffer to request timeout
            )
            # Server returns plain text blob_id or "timeout"
            return {"blob_id":response.text.strip('"')}  # Remove quotes if JSON string
        except requests.exceptions.Timeout:
            # Timeout is expected for long-polling
            return {'timeout': True}
        except Exception as e:
            get_logger().error(f"Watch failed: {e}")
            return {'error': str(e)}

    def watch(self, blob_id: str, callback: Callable[['BlobFile'], None],
              max_idle_timeout: int = 600, threaded: bool = True, **kwargs):
        """
        Watch for changes to a blob with automatic callback execution.

        This method registers a callback that will be called whenever the blob changes.
        The callback receives a BlobFile object with the updated data.

        Features:
        - Runs in background thread (non-blocking)
        - Automatic batching of multiple watches
        - Auto-cleanup after max_idle_timeout seconds without updates
        - Thread-safe callback execution

        Args:
            blob_id: The blob to watch for changes
            callback: Function(blob_file: BlobFile) -> None
                     Called when blob changes, receives BlobFile object
            max_idle_timeout: Seconds without updates before auto-removing callback (default: 600 = 10 min)
            threaded: If True, runs in background thread (default: True)

        Example:
            def on_change(blob_file: BlobFile):
                data = blob_file.read_json()
                print(f"Blob updated: {data}")

            storage.watch('myblob/data.json', on_change, max_idle_timeout=600)
            # Callback will be called automatically when blob changes
            # Auto-removed after 10 minutes without updates
        """
        if not threaded:
            get_logger().warning("Non-threaded watch not yet implemented, using threaded mode")

        self.watch_manager.add_watch(blob_id, callback, max_idle_timeout, **kwargs)
        get_logger().info(f"Started watching blob '{blob_id}'")

    def stop_watch(self, blob_id: str, callback: Optional[Callable] = None):
        """
        Stop watching a blob.

        Args:
            blob_id: The blob to stop watching
            callback: Specific callback to remove, or None to remove all callbacks for this blob

        Example:
            # Stop specific callback
            storage.stop_watch('myblob/data.json', on_change)

            # Stop all callbacks for a blob
            storage.stop_watch('myblob/data.json')
        """
        self.watch_manager.remove_watch(blob_id, callback)
        get_logger().info(f"Stopped watching blob '{blob_id}'")

    def stop_all_watches(self):
        """
        Stop all active watches and shutdown the watch thread.

        This is useful for cleanup when shutting down the application.
        """
        self.watch_manager.remove_all_watches()
        get_logger().info("Stopped all watches")

    def share_blob(self, blob_id: str, target_user_id: str, access_level: str = 'read_only'):
        """
        Share a blob with another user.

        Args:
            blob_id: The blob to share
            target_user_id: Public User ID of the target user (e.g., 'user_abc123')
            access_level: 'read_only' or 'read_write'

        Returns:
            Dict with share information

        Security:
            - You must have write access to the blob
            - Cannot share with yourself
            - Target user must exist (have an API key)
            - One party cannot have both keys (enforced by user_id system)

        Example:
            # Share config file with another user (read-only)
            storage.share_blob('app/config.json', 'user_abc123', 'read_only')

            # Share data file with write access
            storage.share_blob('app/data.json', 'user_xyz789', 'read_write')
        """
        if access_level not in ['read_only', 'read_write']:
            raise ValueError("access_level must be 'read_only' or 'read_write'")

        payload = {
            'user_id': target_user_id,
            'access_level': access_level
        }

        response = self._make_request(
            'POST',
            f'/share/{blob_id}',
            blob_id=blob_id,
            json=payload
        )

        result = response.json()
        get_logger().info(f"Shared blob '{blob_id}' with user '{target_user_id}' ({access_level})")
        return result

    def revoke_share(self, blob_id: str, target_user_id: str):
        """
        Revoke share access for a user.

        Args:
            blob_id: The blob to revoke access from
            target_user_id: Public User ID to revoke access from

        Example:
            storage.revoke_share('app/config.json', 'user_abc123')
        """
        response = self._make_request(
            'DELETE',
            f'/share/{blob_id}/{target_user_id}',
            blob_id=blob_id
        )

        get_logger().info(f"Revoked share for blob '{blob_id}' from user '{target_user_id}'")
        return response.status_code == 204

    def list_shares(self, blob_id: str) -> List[Dict]:
        """
        List all users who have access to a blob.

        Args:
            blob_id: The blob to list shares for

        Returns:
            List of share information dicts with:
            - user_id: Public User ID
            - access_level: 'read_only' or 'read_write'
            - granted_by: User ID who granted access
            - granted_at: Unix timestamp

        Example:
            shares = storage.list_shares('app/config.json')
            for share in shares:
                print(f"{share['user_id']}: {share['access_level']}")
        """
        response = self._make_request(
            'GET',
            f'/share/{blob_id}',
            blob_id=blob_id
        )

        result = response.json()
        return result.get('shares', [])

    # NOTE: share_blobs and recover_blob are legacy coordination endpoints
    def share_blobs(self, blob_ids: list[str]):
        """Legacy method - sharing is now handled automatically by the server"""
        get_logger().info(f"Instructing a server to share blobs for recovery: {blob_ids}")
        payload = {"blob_ids": blob_ids}
        self._make_request('POST', '/share', json=payload)
        get_logger().info("Sharing command sent successfully.")

    def recover_blob(self, lost_blob_id: str) -> bytes:
        """Legacy method - recovery is now handled via shard reconstruction"""
        get_logger().info(f"Attempting to recover '{lost_blob_id}' from the cluster.")
        payload = {"blob_id": lost_blob_id}
        response = self._make_request('POST', '/recover', json=payload)
        recovered_data = response.content
        get_logger().info(f"Successfully recovered blob '{lost_blob_id}'.")
        self._save_blob_to_cache(lost_blob_id, recovered_data)
        return recovered_data

    def _get_blob_cache_filename(self, blob_id: str) -> str:
        return os.path.join(self.storage_directory, blob_id + '.blobcache')

    def _save_blob_to_cache(self, blob_id: str, data: bytes):
        if not data or data is None:
            return
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
                 servers: list[str] = None, use_cache = True):
        self.use_cache = use_cache
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
            raw_blob_data = self.storage.read_blob(self.blob_id, use_cache=self.use_cache)
            if raw_blob_data != b'' and (not raw_blob_data or raw_blob_data is None):
                raw_blob_data = b""
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

    def watch(self, callback: Callable[['BlobFile'], None],
              max_idle_timeout: int = 600, threaded: bool = True):
        """
        Watch for changes to this blob file with automatic callback execution.

        This is a high-level convenience method that watches this specific blob
        and calls the provided callback whenever it changes.

        Features:
        - Non-blocking (runs in background thread)
        - Automatic callback execution with updated BlobFile object
        - Auto-cleanup after max_idle_timeout seconds without updates

        Args:
            callback: Function(blob_file: BlobFile) -> None
                     Called when this blob changes, receives updated BlobFile object
            max_idle_timeout: Seconds without updates before auto-removing callback (default: 600 = 10 min)
            threaded: If True, runs in background thread (default: True)

        Example:
            def on_data_change(blob_file: BlobFile):
                data = blob_file.read_json()
                print(f"Data updated: {data}")

            # Start watching
            blob = BlobFile('myblob/data.json', 'r')
            blob.watch(on_data_change, max_idle_timeout=600)

            # Callback will be called automatically when blob changes
            # Auto-removed after 10 minutes without updates

            # To stop watching:
            blob.stop_watch(on_data_change)
        """
        self.storage.watch(self.blob_id, callback, max_idle_timeout, threaded)

    def stop_watch(self, callback: Optional[Callable] = None):
        """
        Stop watching this blob file.

        Args:
            callback: Specific callback to remove, or None to remove all callbacks

        Example:
            blob.stop_watch(on_data_change)  # Stop specific callback
            blob.stop_watch()  # Stop all callbacks for this blob
        """
        self.storage.stop_watch(self.blob_id, callback)

