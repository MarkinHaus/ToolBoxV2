# Assumed to be in a file like: toolboxv2/utils/db/mini_db.py

import json
from toolboxv2 import Result
from toolboxv2.mods.DB.types import AuthenticationTypes
from toolboxv2.utils.security.cryp import Code

# Import the new networked blob storage system
from toolboxv2.utils.extras.blobs import BlobFile, BlobStorage


class BlobDB:
    """
    A persistent, encrypted dictionary-like database that uses the BlobStorage
    system as its backend, making it networked and fault-tolerant.
    """
    auth_type = AuthenticationTypes.location

    def __init__(self):
        self.data: dict = {}
        self.key: str | None = None
        self.db_path: str | None = None
        self.storage_client: BlobStorage | None = None


    def initialize(self, db_path: str, key: str, storage_client: BlobStorage) -> Result:
        """
        Initializes the database from a location within the blob storage.

        Args:
            db_path (str): The virtual path within the blob storage,
                           e.g., "my_database_blob/database.json".
            key (str): The encryption key for the database content.
            storage_client (BlobStorage): An initialized BlobStorage client instance.

        Returns:
            Result: An OK result if successful.
        """
        self.db_path = db_path
        self.key = key
        self.storage_client = storage_client

        print(f"Initializing BlobDB from blob path: '{self.db_path}'...")

        try:
            # Use BlobFile for reading. It handles caching, networking, and decryption.
            db_file = BlobFile(self.db_path, mode='r', storage=self.storage_client, key=self.key)
            if not db_file.exists():
                print(f"Database file not found at '{self.db_path}'. Starting with an empty database.")
                db_file.create()
                self.data = {}
            else:
                with db_file as f:
                    # read_json safely loads the content.
                    self.data = f.read_json()
                    if not self.data:  # Handle case where file exists but is empty
                        self.data = {}
                print("Successfully initialized database.")

        except Exception as e:
            print(f"Warning: Could not initialize BlobDB from '{self.db_path}'. Error: {e}. Starting fresh.")
            self.data = {}

        return Result.ok().set_origin("Blob Dict DB")

    def exit(self) -> Result:
        """
        Saves the current state of the database back to the blob storage.
        """
        print("BLOB DB on exit ", not all([self.key, self.db_path, self.storage_client]))
        if not all([self.key, self.db_path, self.storage_client]):
            return Result.default_internal_error(
                info="Database not initialized. Cannot exit."
            ).set_origin("Blob Dict DB")

        print(f"Saving database to blob path: '{self.db_path}'...")
        try:
            # Use BlobFile for writing. It handles encryption, networking, and updates.
            with BlobFile(self.db_path, mode='w', storage=self.storage_client, key=self.key) as f:
                f.write_json(self.data)

            print("Success: Database saved to blob storage.")
            return Result.ok().set_origin("Blob Dict DB")

        except Exception as e:
            return Result.custom_error(
                data=e,
                info=f"Error saving database to blob storage: {e}"
            ).set_origin("Blob Dict DB")

    # --- Data Manipulation Methods (Unchanged Logic) ---
    # These methods operate on the in-memory `self.data` dictionary and do not
    # need to be changed, as the loading/saving is handled by initialize/exit.

    def get(self, key: str) -> Result:
        if not self.data:
            return Result.default_internal_error(info=f"No data found for key '{key}' (database is empty).").set_origin(
                "Blob Dict DB")

        data = []
        if key == 'all':
            data_info = "Returning all data available"
            data = list(self.data.items())
        elif key == "all-k":
            data_info = "Returning all keys"
            data = list(self.data.keys())
        else:
            data_info = f"Returning values for keys starting with '{key.replace('*', '')}'"
            data = [self.data[k] for k in self.scan_iter(key)]

        if not data:
            return Result.default_internal_error(info=f"No data found for key '{key}'").set_origin("Blob Dict DB")

        return Result.ok(data=data, data_info=data_info).set_origin("Blob Dict DB")

    def set(self, key: str, value) -> Result:
        if not isinstance(key, str) or not key:
            return Result.default_user_error(info="Key must be a non-empty string.").set_origin("Blob Dict DB")

        self.data[key] = value
        return Result.ok().set_origin("Blob Dict DB")

    def scan_iter(self, search: str = ''):
        if not self.data:
            return []
        prefix = search.replace('*', '')
        return [key for key in self.data if key.startswith(prefix)]

    def append_on_set(self, key: str, value: list) -> Result:
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
        if key.endswith('*'):
            return len(self.scan_iter(key))
        return 1 if key in self.data else 0

    def delete(self, key: str, matching: bool = False) -> Result:
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
