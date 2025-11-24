# Assumed to be in a file like: toolboxv2/utils/db/mini_db.py

import os
import pickle

from toolboxv2 import Result
from toolboxv2.mods.DB.types import AuthenticationTypes

# Import the new networked blob storage system
from toolboxv2.utils.extras.blobs import BlobFile, BlobStorage


class BlobDB:
    """
    A persistent, encrypted dictionary-like database that uses the BlobStorage
    system as its backend, making it networked and fault-tolerant.

    This implementation uses multiple BlobFiles instead of a single virtual file.
    Keys are structured like USER::XYZ::: or MANAGER::SPACE::DATA::XYZ and are
    converted to file paths by replacing :: with /.
    """
    auth_type = AuthenticationTypes.location

    def __init__(self):
        self.data: dict = {}  # In-memory cache of all data
        self.key: str | None = None
        self.db_path: str | None = None  # Base path for the database
        self.storage_client: BlobStorage | None = None


    def _key_to_blob_path(self, key: str) -> str:
        """
        Converts a database key to a blob file path.
        Replaces :: with / to create a hierarchical file structure.

        Examples:
            USER::XYZ::: -> db_path/USER/XYZ.json
            MANAGER::SPACE::DATA::XYZ -> db_path/MANAGER/SPACE/DATA/XYZ.json
        """
        # Replace :: with / and remove trailing/leading colons and slashes
        path_parts = key.replace('::', '/').strip('/').strip(':').strip('/')
        # Combine with base db_path
        return f"{self.db_path}/{path_parts}.json"

    def _load_blob_file(self, blob_path: str) -> dict:
        """
        Loads data from a specific blob file.
        Returns empty dict if file doesn't exist.
        """
        try:
            db_file = BlobFile(blob_path, mode='r', storage=self.storage_client, key=self.key)
            if not db_file.exists():
                return {}
            with db_file as f:
                data = f.read_json()
                return data if data else {}
        except Exception as e:
            print(f"Warning: Could not load blob file '{blob_path}'. Error: {e}")
            return {}

    def _save_blob_file(self, blob_path: str, data: dict) -> bool:
        """
        Saves data to a specific blob file.
        Returns True on success, False on failure.
        """
        try:
            # Ensure the blob exists first
            db_file = BlobFile(blob_path, mode='r', storage=self.storage_client, key=self.key)
            if not db_file.exists():
                db_file.create()

            with BlobFile(blob_path, mode='w', storage=self.storage_client, key=self.key) as f:
                f.write_json(data)
            return True
        except Exception as e:
            print(f"Error: Could not save blob file '{blob_path}'. Error: {e}")
            return False

    def initialize(self, db_path: str, key: str, storage_client: BlobStorage) -> Result:
        """
        Initializes the database from a location within the blob storage.

        Args:
            db_path (str): The base path within the blob storage,
                           e.g., "my_database_blob".
            key (str): The encryption key for the database content.
            storage_client (BlobStorage): An initialized BlobStorage client instance.

        Returns:
            Result: An OK result if successful.
        """
        self.db_path = db_path.rstrip('/')
        self.key = key
        self.storage_client = storage_client

        print(f"Initializing BlobDB with base path: '{self.db_path}'...")

        # Initialize with empty data - data will be loaded on-demand
        self.data = {}

        print("Successfully initialized database with multi-file storage.")
        return Result.ok().set_origin("Blob Dict DB")

    def exit(self) -> Result:
        """
        Saves the current state of the database back to the blob storage.
        Each key is saved to its own blob file based on the key structure.
        """
        print("BLOB DB on exit ", not all([self.key, self.db_path, self.storage_client]))
        if not all([self.key, self.db_path, self.storage_client]):
            return Result.default_internal_error(
                info="Database not initialized. Cannot exit."
            ).set_origin("Blob Dict DB")

        print(f"Saving database to blob storage at base path: '{self.db_path}'...")

        errors = []
        saved_count = 0

        try:
            # Group keys by their blob file path
            blob_files_data = {}
            for key, value in self.data.items():
                blob_path = self._key_to_blob_path(key)
                if blob_path not in blob_files_data:
                    blob_files_data[blob_path] = {}
                blob_files_data[blob_path][key] = value

            # Save each blob file
            for blob_path, data in blob_files_data.items():
                if self._save_blob_file(blob_path, data):
                    saved_count += 1
                else:
                    errors.append(f"Failed to save {blob_path}")

            if errors:
                return Result.custom_error(
                    data=errors,
                    info=f"Saved {saved_count} blob files, but {len(errors)} failed: {errors}"
                ).set_origin("Blob Dict DB")

            print(f"Success: Database saved to {saved_count} blob file(s).")
            return Result.ok().set_origin("Blob Dict DB")

        except Exception as e:
            return Result.custom_error(
                data=e,
                info=f"Error saving database to blob storage: {e}"
            ).set_origin("Blob Dict DB")

    # --- Data Manipulation Methods ---
    # These methods now handle loading data on-demand from multiple blob files

    def _ensure_key_loaded(self, key: str):
        """
        Ensures that data for a specific key is loaded into memory.
        If the key contains wildcards, loads all matching blob files.
        """
        if key.endswith('*'):
            # For wildcard searches, scan blob storage and load matching files
            prefix = key.replace('*', '')

            # First, check if we already have matching keys in memory
            matching_keys_in_memory = [k for k in self.data.keys() if k.startswith(prefix)]

            # Scan blob storage for matching files
            matching_blob_paths = self._scan_blob_storage_for_prefix(prefix)

            # Load each matching blob file that isn't already fully loaded
            for blob_path in matching_blob_paths:
                # Check if this blob file's data is already in memory
                # by checking if any key from this blob file is missing
                blob_data = self._load_blob_file(blob_path)

                # Only merge if we found new data
                if blob_data:
                    # Check if any keys from this blob are not in memory
                    new_keys = [k for k in blob_data.keys() if k not in self.data]
                    if new_keys:
                        self.data.update(blob_data)

            return

        # Check if key is already in memory
        if key in self.data:
            return

        # Load the blob file for this key
        blob_path = self._key_to_blob_path(key)
        blob_data = self._load_blob_file(blob_path)

        # Merge loaded data into memory
        self.data.update(blob_data)

    def _scan_blob_storage_for_prefix(self, prefix: str) -> list[str]:
        """
        Scans the blob storage directory for files matching the given prefix.
        Returns a list of blob file paths that match the prefix pattern.

        Args:
            prefix (str): The key prefix to search for (e.g., "USER::alice")

        Returns:
            list[str]: List of matching blob file paths
        """
        if not self.storage_client or not hasattr(self.storage_client, 'storage_directory'):
            # If storage client doesn't have a storage_directory, we can't scan
            return []

        matching_paths = []
        storage_dir = self.storage_client.storage_directory

        # Convert prefix to path pattern
        # e.g., "USER::alice" -> "USER/alice"
        prefix_path = prefix.replace('::', '/').strip('/').strip(':').strip('/')

        # Build the expected directory path
        search_base = os.path.join(self.db_path, prefix_path) if prefix_path else self.db_path

        try:
            # Walk through the storage directory looking for matching blob files
            if os.path.exists(storage_dir):
                for root, dirs, files in os.walk(storage_dir):
                    for file in files:
                        if file.endswith('.blobcache'):
                            # Extract blob_id from filename
                            blob_id = file.replace('.blobcache', '')

                            # Try to reconstruct what this blob might contain
                            # by checking if it matches our db_path pattern
                            # This is a heuristic approach - we load and check the content
                            cache_file = os.path.join(root, file)

                            # Read the blob to check if it contains keys with our prefix
                            try:
                                with open(cache_file, 'rb') as f:
                                    blob_content = pickle.loads(f.read())

                                    # Check if this blob contains data for our prefix
                                    # The blob structure is nested: blob_id -> folder -> file -> data
                                    # We need to check if any keys in the stored JSON match our prefix
                                    if self._blob_contains_prefix(blob_id, prefix, blob_content):
                                        # Construct the blob path that would be used to load this
                                        # We need to find the actual path within our db structure
                                        potential_paths = self._extract_blob_paths_from_content(blob_id, prefix, blob_content)
                                        matching_paths.extend(potential_paths)
                            except Exception as e:
                                # Skip blobs we can't read
                                print(f"Warning: Could not scan blob cache file '{cache_file}': {e}")
                                continue
        except Exception as e:
            print(f"Warning: Error scanning blob storage directory: {e}")

        return list(set(matching_paths))  # Remove duplicates

    def _blob_contains_prefix(self, blob_id: str, prefix: str, blob_content: dict) -> bool:
        """
        Checks if a blob contains any keys matching the given prefix.

        Args:
            blob_id (str): The blob identifier
            prefix (str): The key prefix to search for
            blob_content (dict): The deserialized blob content

        Returns:
            bool: True if the blob contains matching keys
        """
        # The blob content structure is: {folder: {file: data}}
        # We need to check if any stored keys match our prefix
        for folder, files in blob_content.items():
            if isinstance(files, dict):
                for file, data in files.items():
                    # Reconstruct the potential blob path
                    potential_path = f"{self.db_path}/{folder}/{file}"

                    # Check if this path could contain keys with our prefix
                    if prefix in potential_path or potential_path.startswith(self.db_path):
                        return True
        return False

    def _extract_blob_paths_from_content(self, blob_id: str, prefix: str, blob_content: dict) -> list[str]:
        """
        Extracts blob file paths from blob content that match the prefix.

        Args:
            blob_id (str): The blob identifier
            prefix (str): The key prefix to search for
            blob_content (dict): The deserialized blob content

        Returns:
            list[str]: List of blob paths that might contain matching keys
        """
        paths = []
        prefix_path = prefix.replace('::', '/').strip('/').strip(':').strip('/')

        for folder, files in blob_content.items():
            if isinstance(files, dict):
                for file, data in files.items():
                    # Reconstruct the blob path
                    if file.endswith('.json'):
                        file_without_ext = file[:-5]  # Remove .json
                    else:
                        file_without_ext = file

                    # Build the full path
                    if folder:
                        full_path = f"{self.db_path}/{folder}/{file_without_ext}.json"
                    else:
                        full_path = f"{self.db_path}/{file_without_ext}.json"

                    # Check if this path matches our prefix
                    path_key = full_path.replace(f"{self.db_path}/", "").replace(".json", "").replace("/", "::")
                    if path_key.startswith(prefix):
                        paths.append(full_path)

        return paths

    def get(self, key: str) -> Result:
        data = []

        if key == 'all':
            # Load all data - this is expensive, consider pagination in production
            data_info = "Returning all data available"
            data = list(self.data.items())
        elif key == "all-k":
            data_info = "Returning all keys"
            data = list(self.data.keys())
        else:
            # Ensure the key or matching keys are loaded
            self._ensure_key_loaded(key)
            data_info = f"Returning values for keys starting with '{key.replace('*', '')}'"
            data = [self.data[k] for k in self.scan_iter(key)]

        if not data:
            return Result.default_internal_error(info=f"No data found for key '{key}'").set_origin("Blob Dict DB")

        return Result.ok(data=data, data_info=data_info).set_origin("Blob Dict DB")

    def set(self, key: str, value) -> Result:
        if not isinstance(key, str) or not key:
            return Result.default_user_error(info="Key must be a non-empty string.").set_origin("Blob Dict DB")

        # Ensure existing data for this key is loaded first
        self._ensure_key_loaded(key)

        self.data[key] = value
        return Result.ok().set_origin("Blob Dict DB")

    def scan_iter(self, search: str = ''):
        if not self.data:
            return []
        prefix = search.replace('*', '')
        return [key for key in self.data if key.startswith(prefix)]

    def append_on_set(self, key: str, value: list) -> Result:
        # Ensure existing data for this key is loaded first
        self._ensure_key_loaded(key)

        if key not in self.data:
            self.data[key] = []

        if not isinstance(self.data[key], list):
            return Result.default_user_error(info=f"Existing value for key '{key}' is not a list.").set_origin(
                "Blob Dict DB")

        # Use a set for efficient checking to avoid duplicates
        existing_set = set(self.data[key])
        new_items = [item for item in value if item not in existing_set]
        self.data[key].extend(new_items)
        return Result.ok().set_origin("Blob Dict DB")

    def if_exist(self, key: str) -> int:
        # Ensure data is loaded for existence check
        self._ensure_key_loaded(key)

        if key.endswith('*'):
            return len(self.scan_iter(key))
        return 1 if key in self.data else 0

    def delete(self, key: str, matching: bool = False) -> Result:
        # Ensure data is loaded before deletion
        self._ensure_key_loaded(key)

        keys_to_delete = []
        if matching:
            keys_to_delete = self.scan_iter(key)
        elif key in self.data:
            keys_to_delete.append(key)

        if not keys_to_delete:
            return Result.default_internal_error(info=f"No keys found to delete for pattern '{key}'").set_origin(
                "Blob Dict DB")

        deleted_items = {k: self.data.pop(k) for k in keys_to_delete}
        return Result.ok(
            data=list(deleted_items.items()),
            data_info=f"Successfully removed {len(deleted_items)} item(s)."
        ).set_origin("Blob Dict DB")
