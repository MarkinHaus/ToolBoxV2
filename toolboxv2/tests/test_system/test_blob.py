import pickle
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os

from toolboxv2 import Code
from toolboxv2.utils.extras.blobs import BlobStorage, BlobFile


class TestBlobStorage(unittest.TestCase):

    def setUp(self):
        self.mock_storage_directory = Path("/tmp/test_blob_storage")
        self.blob_storage = BlobStorage(storage_directory=str(self.mock_storage_directory))

    @patch('os.path.exists')
    @patch('pickle.dump')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_save_blob(self, mock_open, mock_pickle_dump, mock_path_exists):
        blob_id = "test_blob_id.blob"
        blob_data = {"data": b"test data", "links": {"self": b""}}
        mock_path_exists.return_value = False

        self.blob_storage._save_blob(blob_id, blob_data)

        mock_open.assert_called_once_with(str(self.mock_storage_directory) + "\\" + blob_id, 'wb')
        mock_pickle_dump.assert_called_once_with(blob_data, mock_open())

    @patch('os.path.exists')
    @patch('pickle.load')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data=b'')
    def test_load_blob(self, mock_open, mock_pickle_load, mock_path_exists):
        blob_id = "test_blob.blob"
        mock_path_exists.return_value = True
        mock_pickle_load.return_value = {"data": b"test data"}

        result = self.blob_storage._load_blob(blob_id)

        mock_open.assert_called_once_with(str(self.mock_storage_directory) + "\\" + blob_id, 'rb')
        self.assertEqual(result, {"data": b"test data"})


class TestBlobFile(unittest.TestCase):

    def setUp(self):
        self.mock_blob_id = "test_blob_id"
        self.mock_folder = "test_folder"
        self.mock_datei = "test_file.txt"
        self.mock_filename = f"{self.mock_blob_id}/{self.mock_folder}/{self.mock_datei}"
        self.mock_storage = MagicMock(spec=BlobStorage)
        self.mock_key = Code.generate_symmetric_key()
        self.mock_data = b"test data"
        self.mock_encrypted_data = Code.encrypt_symmetric(self.mock_data, self.mock_key)
        self.blob_file = BlobFile(filename=self.mock_filename, mode='r', storage=self.mock_storage, key=self.mock_key)

    @patch('toolboxv2.utils.extras.blobs.Code')
    @patch('pickle.loads')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data=b'')
    def test_enter_read_mode(self, mock_open, mock_pickle_loads, mock_code):
        mock_blob_data = {self.mock_folder: {self.mock_datei: self.mock_encrypted_data}}
        mock_pickle_loads.return_value = mock_blob_data
        mock_code.decrypt_symmetric.return_value = self.mock_data

        with self.blob_file as file:
            self.assertEqual(file.data, self.mock_data)

        mock_pickle_loads.assert_called_once_with(self.mock_storage.read_blob(self.mock_blob_id))
        mock_code.decrypt_symmetric.assert_called_once_with(self.mock_encrypted_data, self.mock_key)

    @patch('toolboxv2.utils.extras.blobs.Code')
    @patch('pickle.loads')
    @patch('pickle.dumps')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_enter_write_mode(self, mock_open, mock_pickle_loads, mock_pickle_dumps, mock_code):
        self.blob_file.mode = 'w'

        with self.blob_file as file:
            self.assertEqual(file.data, b"")

    def test_exit_write_mode(self):
        self.blob_file.mode = 'w'
        self.mock_storage.read_blob.return_value = pickle.dumps({})
        with self.blob_file as file:
            file.clear()
            file.write(self.mock_data)

        self.mock_storage.update_blob.assert_called_once()


if __name__ == '__main__':
    unittest.main()
