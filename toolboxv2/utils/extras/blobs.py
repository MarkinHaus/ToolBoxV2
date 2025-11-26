# file: blobs.py
# Production-ready BlobStorage client with robust error handling
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
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

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


class ConnectionState(Enum):
    """Connection state for a server"""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    UNAUTHORIZED = "unauthorized"
    UNREACHABLE = "unreachable"
    ERROR = "error"


@dataclass
class ServerStatus:
    """Status information for a server"""
    url: str
    state: ConnectionState = ConnectionState.UNKNOWN
    last_check: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None

    def is_healthy(self) -> bool:
        return self.state == ConnectionState.HEALTHY

    def mark_healthy(self):
        self.state = ConnectionState.HEALTHY
        self.error_count = 0
        self.last_error = None
        self.last_check = time.time()

    def mark_error(self, error: str, state: ConnectionState = ConnectionState.ERROR):
        self.state = state
        self.error_count += 1
        self.last_error = error
        self.last_check = time.time()


@dataclass
class WatchCallback:
    """Wrapper for a watch callback with metadata."""
    callback: Callable[['BlobFile'], None]
    blob_id: str
    last_update: float = field(default_factory=time.time)
    max_idle_timeout: int = 600
    folder: Optional[str] = None
    filename: Optional[str] = None

    def is_expired(self) -> bool:
        return (time.time() - self.last_update) > self.max_idle_timeout

    def update_timestamp(self):
        self.last_update = time.time()


class WatchManager:
    """Manages watch operations for blob changes."""

    def __init__(self, storage: 'BlobStorage'):
        self.storage = storage
        self._watches: Dict[str, List[WatchCallback]] = {}
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._running = False
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5
        self._backoff_time = 1.0

    def add_watch(self, blob_id: str, callback: Callable[['BlobFile'], None],
                  max_idle_timeout: int = 600, **kwargs):
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

            if not self._running:
                self._start_watch_thread()

    def remove_watch(self, blob_id: str, callback: Optional[Callable] = None):
        with self._lock:
            if blob_id not in self._watches:
                return

            if callback is None:
                del self._watches[blob_id]
                get_logger().info(f"Removed all watches for blob '{blob_id}'")
            else:
                self._watches[blob_id] = [
                    w for w in self._watches[blob_id]
                    if w.callback != callback
                ]
                if not self._watches[blob_id]:
                    del self._watches[blob_id]
                get_logger().info(f"Removed specific watch for blob '{blob_id}'")

            if not self._watches and self._running:
                self._stop_watch_thread()

    def remove_all_watches(self):
        with self._lock:
            self._watches.clear()
            get_logger().info("Removed all watches")
        if self._running:
            self._stop_watch_thread()

    def _start_watch_thread(self):
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._consecutive_failures = 0
        self._backoff_time = 1.0
        self._watch_thread = threading.Thread(
            target=self._watch_loop,
            name="BlobWatchThread",
            daemon=True
        )
        self._watch_thread.start()
        get_logger().info("Started watch thread")

    def _stop_watch_thread(self):
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=5)
        get_logger().info("Stopped watch thread")

    def _watch_loop(self):
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    if not self._watches:
                        break

                result = self.storage.watch_resource(timeout=60)

                if result.get('timeout'):
                    self._consecutive_failures = 0
                    self._backoff_time = 1.0
                    self._cleanup_expired_callbacks()
                    continue

                if result.get('error'):
                    error_msg = result['error']
                    get_logger().warning(f"Watch error: {error_msg}")

                    # Handle specific error types
                    if '401' in str(error_msg) or 'Unauthorized' in str(error_msg):
                        # Try to re-validate keys
                        get_logger().info("Watch received 401, attempting key re-validation...")
                        self.storage._revalidate_keys()

                    self._consecutive_failures += 1
                    if self._consecutive_failures >= self._max_consecutive_failures:
                        get_logger().error(f"Watch failed {self._consecutive_failures} times, backing off...")
                        self._backoff_time = min(self._backoff_time * 2, 60.0)

                    time.sleep(self._backoff_time)
                    continue

                # Success - reset failure counters
                self._consecutive_failures = 0
                self._backoff_time = 1.0

                changed_blob_id = result if isinstance(result, str) else result.get('blob_id')
                if changed_blob_id and changed_blob_id != 'timeout':
                    self._dispatch_callbacks(changed_blob_id)

                self._cleanup_expired_callbacks()

            except Exception as e:
                get_logger().error(f"Watch loop error: {e}")
                self._consecutive_failures += 1
                time.sleep(self._backoff_time)

        self._running = False
        get_logger().info("Watch loop exited")

    def _dispatch_callbacks(self, blob_id: str):
        with self._lock:
            callbacks = self._watches.get(blob_id, []).copy()

        if not callbacks:
            return

        get_logger().info(f"Dispatching {len(callbacks)} callbacks for blob '{blob_id}'")

        try:
            row_data = self.storage.read_blob(blob_id, use_cache=False)
            for watch_cb in callbacks:
                if watch_cb.filename:
                    if not watch_cb.folder:
                        watch_cb.folder = "/"
                    if not watch_cb.folder.startswith('/'):
                        watch_cb.folder = '/' + watch_cb.folder
                try:
                    blob_file = BlobFile(
                        blob_id + watch_cb.folder + '/' + watch_cb.filename,
                        'r',
                        storage=self.storage
                    ) if watch_cb.filename else row_data
                    watch_cb.callback(blob_file)
                    watch_cb.update_timestamp()
                except Exception as e:
                    get_logger().error(f"Callback error for blob '{blob_id}': {e}")
                    import traceback
                    get_logger().error(traceback.format_exc())
        except Exception as e:
            get_logger().error(f"Failed to create BlobFile for '{blob_id}': {e}")

    def _cleanup_expired_callbacks(self):
        with self._lock:
            expired_blobs = []
            for blob_id, callbacks in self._watches.items():
                active_callbacks = [cb for cb in callbacks if not cb.is_expired()]
                if len(active_callbacks) < len(callbacks):
                    removed_count = len(callbacks) - len(active_callbacks)
                    get_logger().info(f"Removed {removed_count} expired callbacks for blob '{blob_id}'")
                if active_callbacks:
                    self._watches[blob_id] = active_callbacks
                else:
                    expired_blobs.append(blob_id)

            for blob_id in expired_blobs:
                del self._watches[blob_id]
                get_logger().info(f"Removed blob '{blob_id}' from watch list (no active callbacks)")

            if not self._watches and self._running:
                get_logger().info("No more active watches, stopping watch thread")
                self._stop_event.set()


class ApiKeyHandler(metaclass=Singleton):
    """Manages API keys for distributed blob storage servers."""

    def __init__(self, storage_directory: str):
        self.storage_directory = storage_directory
        os.makedirs(storage_directory, exist_ok=True)
        self.keys_file = os.path.join(storage_directory, 'api_keys.enc')
        self._keys: Dict[str, Dict[str, str]] = {}
        self._load_keys()

    def _load_keys(self):
        if not os.path.exists(self.keys_file):
            return
        try:
            with open(self.keys_file, 'r') as f:
                encrypted_data = f.read()
            if encrypted_data:
                device_key = DEVICE_KEY()
                decrypted = Code.decrypt_symmetric(encrypted_data, device_key)
                loaded = json.loads(decrypted)
                for server, value in loaded.items():
                    if isinstance(value, str):
                        self._keys[server] = {'api_key': value, 'user_id': None}
                    else:
                        self._keys[server] = value
                get_logger().info(f"Loaded {len(self._keys)} API keys from storage")
        except Exception as e:
            get_logger().error(f"Failed to load API keys: {e}")
            self._keys = {}

    def _save_keys(self):
        try:
            device_key = DEVICE_KEY()
            data = json.dumps(self._keys)
            encrypted = Code.encrypt_symmetric(data, device_key)
            with open(self.keys_file, 'w') as f:
                f.write(encrypted)
        except Exception as e:
            get_logger().error(f"Failed to save API keys: {e}")

    def get_key(self, server_url: str) -> Optional[str]:
        server_data = self._keys.get(server_url)
        if server_data:
            return server_data.get('api_key')
        return None

    def get_user_id(self, server_url: str) -> Optional[str]:
        server_data = self._keys.get(server_url)
        if server_data:
            return server_data.get('user_id')
        return None

    def set_key(self, server_url: str, api_key: str, user_id: Optional[str] = None):
        self._keys[server_url] = {
            'api_key': api_key,
            'user_id': user_id
        }
        self._save_keys()

    def has_key(self, server_url: str) -> bool:
        return server_url in self._keys

    def remove_key(self, server_url: str):
        """Remove an API key for a server"""
        if server_url in self._keys:
            del self._keys[server_url]
            self._save_keys()
            get_logger().info(f"Removed API key for {server_url}")

    def get_all_servers(self) -> List[str]:
        """Get all servers with stored keys"""
        return list(self._keys.keys())


class ConsistentHashRing:
    """Consistent hash ring for mapping keys to nodes."""

    def __init__(self, replicas=100):
        self.replicas = replicas
        self._keys = []
        self._nodes = {}

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)

    def add_node(self, node: str):
        for i in range(self.replicas):
            vnode_key = f"{node}:{i}"
            h = self._hash(vnode_key)
            bisect.insort(self._keys, h)
            self._nodes[h] = node

    def get_nodes_for_key(self, key: str) -> list[str]:
        if not self._nodes:
            return []
        h = self._hash(key)
        start_idx = bisect.bisect_left(self._keys, h)
        found_nodes = []
        for i in range(len(self._keys)):
            idx = (start_idx + i) % len(self._keys)
            node_hash = self._keys[idx]
            physical_node = self._nodes[node_hash]
            if physical_node not in found_nodes:
                found_nodes.append(physical_node)
            if len(found_nodes) == len(set(self._nodes.values())):
                break
        return found_nodes

    def get_peer_nodes(self, key: str, count: int = 3) -> list[str]:
        all_nodes = self.get_nodes_for_key(key)
        if len(all_nodes) <= 1:
            return []
        return all_nodes[1:count + 1]


class BlobStorage:
    """
    Production-ready client for distributed blob storage.

    Features:
    - Robust API key management with validation
    - Automatic key recovery after server restarts
    - Connection health tracking
    - Reed-Solomon sharding
    - Optimistic locking with version control
    """

    def __init__(self, servers: list[str], storage_directory: str = './.data/blob_cache',
                 data_shards: int = 4, parity_shards: int = 2, api_key_dir: str = './.data/api_keys',
                 auto_validate_keys: bool = True):
        self.servers = servers
        self.session = requests.Session()
        self.storage_directory = storage_directory
        self.blob_ids = []
        self.data_shards = data_shards
        self.parity_shards = parity_shards
        self._auto_validate_keys = auto_validate_keys
        os.makedirs(storage_directory, exist_ok=True)

        # Initialize API key handler
        self.api_key_handler = ApiKeyHandler(api_key_dir)

        # Server status tracking
        self._server_status: Dict[str, ServerStatus] = {
            server: ServerStatus(url=server) for server in servers
        }

        # Initialize hash ring
        self.hash_ring = ConsistentHashRing()
        for server in self.servers:
            self.hash_ring.add_node(server)

        # Initialize watch manager
        self.watch_manager = WatchManager(self)

        # Ensure all servers have valid API keys
        self._ensure_api_keys()

        # Validate keys on startup
        if auto_validate_keys:
            self._validate_all_keys()

    def _ensure_api_keys(self):
        """Ensure we have API keys for all servers, creating them if needed"""
        for server in self.servers:
            if not self.api_key_handler.has_key(server):
                try:
                    self._create_api_key(server)
                except Exception as e:
                    get_logger().warning(f"Failed to create API key for {server}: {e}")
                    self._server_status[server].mark_error(str(e), ConnectionState.UNREACHABLE)

    def _create_api_key(self, server: str, device_name: Optional[str] = None):
        """Create a new API key on the server"""
        url = f"{server.rstrip('/')}/keys"
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
                self._server_status[server].mark_healthy()
                get_logger().info(f"Created API key for {server} (user_id: {user_id})")
        except Exception as e:
            get_logger().error(f"Failed to create API key for {server}: {e}")
            raise

    def _validate_key(self, server: str) -> bool:
        """Validate an API key with the server"""
        api_key = self.api_key_handler.get_key(server)
        if not api_key:
            return False

        url = f"{server.rstrip('/')}/keys/validate"
        try:
            response = requests.get(
                url,
                headers={'x-api-key': api_key},
                timeout=5
            )
            if response.status_code == 200:
                self._server_status[server].mark_healthy()
                get_logger().debug(f"API key validated for {server}")
                return True
            elif response.status_code == 401:
                get_logger().warning(f"API key invalid for {server}, will re-create")
                self._server_status[server].mark_error("Key invalid", ConnectionState.UNAUTHORIZED)
                return False
            else:
                get_logger().warning(f"Key validation returned {response.status_code} for {server}")
                return False
        except requests.exceptions.ConnectionError:
            get_logger().debug(f"Server {server} unreachable during key validation")
            self._server_status[server].mark_error("Unreachable", ConnectionState.UNREACHABLE)
            return False
        except Exception as e:
            get_logger().warning(f"Key validation failed for {server}: {e}")
            return False

    def _validate_all_keys(self):
        """Validate all API keys and re-create if needed"""
        for server in self.servers:
            if self.api_key_handler.has_key(server):
                if not self._validate_key(server):
                    # Key is invalid, remove and re-create
                    self.api_key_handler.remove_key(server)
                    try:
                        self._create_api_key(server)
                    except Exception as e:
                        get_logger().error(f"Failed to re-create key for {server}: {e}")

    def _revalidate_keys(self):
        """Re-validate all keys (called after 401 errors)"""
        get_logger().info("Re-validating all API keys...")
        self._validate_all_keys()

    def get_user_id(self, server: Optional[str] = None) -> Optional[str]:
        if server:
            return self.api_key_handler.get_user_id(server)
        for srv in self.servers:
            user_id = self.api_key_handler.get_user_id(srv)
            if user_id:
                return user_id
        return None

    def _get_healthy_servers(self) -> List[str]:
        """Get list of servers that are currently healthy"""
        return [
            server for server in self.servers
            if self._server_status[server].is_healthy()
        ]

    def _make_request(self, method, endpoint, blob_id: str = None, max_retries=2,
                      include_auth: bool = True, **kwargs):
        """Makes a resilient HTTP request to the server cluster."""
        if not self.servers:
            res = requests.Response()
            res.status_code = 503
            res.reason = "No servers available"
            return res

        if blob_id:
            preferred_servers = self.hash_ring.get_nodes_for_key(blob_id)
        else:
            # Prefer healthy servers first
            healthy = self._get_healthy_servers()
            unhealthy = [s for s in self.servers if s not in healthy]
            preferred_servers = healthy + unhealthy
            if not preferred_servers:
                preferred_servers = random.sample(self.servers, len(self.servers))

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
                    else:
                        get_logger().warning(f"No API key for {server}, skipping")
                        continue

                try:
                    from toolboxv2 import get_app
                    get_app().sprint(
                        f"[BloBDB] make_request {method} {url} {kwargs if method != 'PUT' else 'PUT_DATA'}")
                    response = self.session.request(method, url, timeout=request_timeout, **kwargs)
                    get_app().sprint(f"response {response.status_code}")

                    if 500 <= response.status_code < 600:
                        self._server_status[server].mark_error(f"Server error {response.status_code}")
                        get_logger().warning(f"Server {server} returned {response.status_code}. Retrying...")
                        continue

                    if response.status_code == 401 and include_auth:
                        error_msg = f"API key invalid - {response.text}"
                        get_logger().error(f"API key invalid for {server} - {response.text} ({endpoint}, {method})")
                        self._server_status[server].mark_error(error_msg, ConnectionState.UNAUTHORIZED)

                        # Try to re-validate/recreate key for this server
                        if self._auto_validate_keys:
                            self.api_key_handler.remove_key(server)
                            try:
                                self._create_api_key(server)
                                # Retry with new key
                                api_key = self.api_key_handler.get_key(server)
                                if api_key:
                                    kwargs['headers']['x-api-key'] = api_key
                                    retry_response = self.session.request(method, url, timeout=request_timeout,
                                                                          **kwargs)
                                    if retry_response.status_code != 401:
                                        self._server_status[server].mark_healthy()
                                        retry_response.raise_for_status()
                                        return retry_response
                            except Exception as e:
                                get_logger().error(f"Failed to recreate key: {e}")

                        # Fall through to try next server

                    response.raise_for_status()
                    self._server_status[server].mark_healthy()
                    return response

                except requests.exceptions.RequestException as e:
                    last_error = e
                    self._server_status[server].mark_error(str(e))
                    get_logger().warning(
                        f"Could not connect to {server}: {e}. "
                        f"{'Trying next server.' if len(self.servers) > 1 else 'No more servers.'}"
                    )

            if len(self.servers) <= 1:
                break

            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt * 0.1)
                get_logger().warning(f"All servers failed. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                if len(preferred_servers) == 1 and len(self.servers) > 1:
                    preferred_servers = random.sample(self.servers, len(self.servers))

        raise ConnectionError(f"Failed to execute request after {max_retries} attempts. Last error: {last_error}")

    def watch_resource(self, timeout: int = 60) -> Dict[str, Any]:
        """Watch for changes with proper error handling"""
        try:
            response = self._make_request('GET', '/watch', timeout=timeout + 5)
            result = response.text
            if result == 'timeout':
                return {'timeout': True}
            return {'blob_id': result}
        except ConnectionError as e:
            return {'error': str(e)}
        except Exception as e:
            return {'error': str(e)}

    def watch(self, blob_id: str, callback: Callable[['BlobFile'], None],
              max_idle_timeout: int = 600, threaded: bool = True, **kwargs):
        """Register a watch callback for a blob"""
        self.watch_manager.add_watch(blob_id, callback, max_idle_timeout, **kwargs)

    def stop_watch(self, blob_id: str, callback: Optional[Callable] = None):
        """Stop watching a blob"""
        self.watch_manager.remove_watch(blob_id, callback)

    def create_blob(self, data: bytes, blob_id=None) -> str:
        """Creates a new blob with Reed-Solomon sharding."""
        if not blob_id:
            blob_id = hashlib.sha256(data).hexdigest()

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
            headers={'x-shard-config': json.dumps(shard_config)}
        )

        get_logger().info(f"Created blob {blob_id} with status {response.status_code}")
        self._save_blob_to_cache(blob_id, data)
        return blob_id

    def get_blob_meta(self, blob_id: str) -> dict:
        """Fetch metadata for a blob."""
        response = self._make_request('GET', f'/blob/{blob_id}/meta', blob_id=blob_id)
        return response.json()

    def update_blob(self, blob_id: str, data: bytes) -> dict:
        """Updates an existing blob with optimistic locking."""
        try:
            meta = self.get_blob_meta(blob_id)
            current_version = meta.get('version', 0)
        except Exception as e:
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
        """Read blob with caching and self-healing."""
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
        except ConnectionError as e:
            if '404' in str(e):
                raise e
            get_logger().warning(f"Failed to fetch metadata for {blob_id}: {e}")
            metadata = None
            version = None
            shard_locations = []
            data_shards = self.data_shards
            parity_shards = self.parity_shards
            original_size = 0
        except Exception as e:
            get_logger().warning(f"Failed to fetch metadata for {blob_id}: {e}")
            metadata = None
            version = None
            shard_locations = []
            data_shards = self.data_shards
            parity_shards = self.parity_shards
            original_size = 0

        # Try to read from primary server
        try:
            response = self._make_request('GET', f'/blob/{blob_id}', blob_id=blob_id)
            blob_data = response.content

            if version:
                etag = response.headers.get('ETag')
                if etag and str(version) != etag:
                    get_logger().warning(f"Version mismatch for {blob_id}: expected {version}, got {etag}")

            self._save_blob_to_cache(blob_id, blob_data)
            return blob_data

        except Exception as e:
            get_logger().warning(f"Failed to read blob {blob_id} from primary: {e}")

            if shard_locations and HAS_REEDSOLO:
                get_logger().info(f"Attempting shard reconstruction for {blob_id}")
                try:
                    reconstructed = self._reconstruct_from_shards(
                        blob_id, shard_locations, data_shards, parity_shards, original_size
                    )
                    if reconstructed:
                        self._save_blob_to_cache(blob_id, reconstructed)
                        return reconstructed
                except Exception as shard_error:
                    get_logger().error(f"Shard reconstruction failed: {shard_error}")
            raise

    def delete_blob(self, blob_id: str) -> bool:
        """Delete a blob"""
        try:
            response = self._make_request('DELETE', f'/blob/{blob_id}', blob_id=blob_id)
            self._delete_blob_from_cache(blob_id)
            return response.status_code == 204
        except Exception as e:
            get_logger().error(f"Failed to delete blob {blob_id}: {e}")
            return False

    def _reconstruct_from_shards(self, blob_id: str, shard_locations: List[str],
                                 data_shards: int, parity_shards: int, original_size: int) -> Optional[bytes]:
        """Reconstruct blob from shards using Reed-Solomon."""
        if not HAS_REEDSOLO:
            get_logger().error("reedsolo library not available")
            return None

        total_shards = data_shards + parity_shards
        shards = [None] * total_shards

        for i in range(total_shards):
            for server in shard_locations:
                try:
                    response = self._make_request(
                        'GET', f'/shard/{blob_id}/{i}',
                        blob_id=blob_id,
                        include_auth=False
                    )
                    shards[i] = response.content
                    break
                except Exception:
                    continue

        # Check if we have enough shards
        available = sum(1 for s in shards if s is not None)
        if available < data_shards:
            get_logger().error(f"Not enough shards: {available}/{data_shards}")
            return None

        # Reed-Solomon decode
        try:
            rs = reedsolo.RSCodec(parity_shards)
            # Combine shards and decode
            combined = b''.join(s for s in shards if s is not None)
            decoded = rs.decode(combined)
            return decoded[:original_size]
        except Exception as e:
            get_logger().error(f"RS decode failed: {e}")
            return None

    # Cache methods
    def _get_cache_path(self, blob_id: str) -> str:
        return os.path.join(self.storage_directory, f"{blob_id}.blob")

    def _save_blob_to_cache(self, blob_id: str, data: bytes):
        try:
            with open(self._get_cache_path(blob_id), 'wb') as f:
                f.write(data)
        except Exception as e:
            get_logger().warning(f"Failed to cache blob {blob_id}: {e}")

    def _load_blob_from_cache(self, blob_id: str) -> Optional[bytes]:
        cache_path = self._get_cache_path(blob_id)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return f.read()
            except Exception as e:
                get_logger().warning(f"Failed to read cached blob {blob_id}: {e}")
        return None

    def _delete_blob_from_cache(self, blob_id: str):
        cache_path = self._get_cache_path(blob_id)
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except Exception as e:
                get_logger().warning(f"Failed to delete cached blob {blob_id}: {e}")

    def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all servers"""
        return {
            server: {
                'state': status.state.value,
                'is_healthy': status.is_healthy(),
                'error_count': status.error_count,
                'last_error': status.last_error,
                'last_check': status.last_check
            }
            for server, status in self._server_status.items()
        }


class BlobFile:
    """File-like interface for blob storage."""

    def __init__(self, filename, mode='r', storage=None, key=None, servers=None, use_cache=True):
        self.mode = mode
        self.use_cache = use_cache
        self.blob_id, self.folder, self.datei = self._path_splitter(filename)

        if storage is None:
            if servers is None:
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
        if not parts:
            raise ValueError("Filename cannot be empty.")
        blob_id = parts[0]
        if len(parts) == 1:
            raise ValueError("Filename must include a path within the blob, e.g., 'blob_id/file.txt'")
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
            if isinstance(e, requests.exceptions.HTTPError):
                if e.response.status_code == 404:
                    blob_content = {}
                else:
                    raise
            elif isinstance(e, (EOFError, pickle.UnpicklingError)):
                blob_content = {}
            elif isinstance(e, ConnectionError):
                if '404' in str(e):
                    blob_content = {}
                else:
                    raise
            else:
                raise

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

            current_level = blob_content
            if self.folder:
                if self.folder not in current_level:
                    current_level[self.folder] = {}
                current_level = current_level[self.folder]

            current_level[self.datei] = final_data
            self.storage.update_blob(self.blob_id, pickle.dumps(blob_content))

    def exists(self) -> bool:
        try:
            raw_blob_data = self.storage.read_blob(self.blob_id)
            if raw_blob_data:
                blob_content = pickle.loads(raw_blob_data)
            else:
                return False
        except (requests.exceptions.HTTPError, EOFError, pickle.UnpicklingError, ConnectionError):
            return False

        current_level = blob_content
        if self.folder:
            if self.folder not in current_level:
                return False
            current_level = current_level[self.folder]

        return self.datei in current_level

    def clear(self):
        self.data_buffer = b''

    def write(self, data):
        if 'w' not in self.mode:
            raise OSError("File not opened in write mode.")
        if isinstance(data, str):
            self.data_buffer += data.encode()
        elif isinstance(data, bytes):
            self.data_buffer += data
        else:
            raise TypeError("write() argument must be str or bytes")

    def read(self):
        if 'r' not in self.mode:
            raise OSError("File not opened in read mode.")
        return self.data_buffer

    def read_json(self):
        if 'r' not in self.mode:
            raise ValueError("File not opened in read mode.")
        if self.data_buffer == b"":
            return {}
        return json.loads(self.data_buffer.decode())

    def write_json(self, data):
        if 'w' not in self.mode:
            raise ValueError("File not opened in write mode.")
        self.data_buffer += json.dumps(data).encode()

    def read_pickle(self):
        if 'r' not in self.mode:
            raise ValueError("File not opened in read mode.")
        if self.data_buffer == b"":
            return {}
        return pickle.loads(self.data_buffer)

    def write_pickle(self, data):
        if 'w' not in self.mode:
            raise ValueError("File not opened in write mode.")
        self.data_buffer += pickle.dumps(data)

    def read_yaml(self):
        if 'r' not in self.mode:
            raise ValueError("File not opened in read mode.")
        if self.data_buffer == b"":
            return {}
        return yaml.safe_load(self.data_buffer)

    def write_yaml(self, data):
        if 'w' not in self.mode:
            raise ValueError("File not opened in write mode.")
        yaml.dump(data, self)

    def watch(self, callback: Callable[['BlobFile'], None],
              max_idle_timeout: int = 600, threaded: bool = True):
        """Watch for changes to this blob file."""
        self.storage.watch(self.blob_id, callback, max_idle_timeout, threaded)

    def stop_watch(self, callback: Optional[Callable] = None):
        """Stop watching this blob file."""
        self.storage.stop_watch(self.blob_id, callback)
