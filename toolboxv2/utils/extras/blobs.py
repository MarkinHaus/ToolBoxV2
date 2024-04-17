import io
import json
import os
import pickle
from pathlib import Path
from typing import List

import reedsolo
import yaml

from ..security.cryp import Code
from ... import get_app, Singleton


class BlobStorage(metaclass=Singleton):

    def __init__(self, storage_directory=None, Fehlerkorrekturbytes=10):
        if storage_directory is None:
            storage_directory = get_app(from_="BlobStorage").data_dir
        self.storage_directory = storage_directory
        self.rs = reedsolo.RSCodec(Fehlerkorrekturbytes)  # Reed-Solomon-Code mit 10 Fehlerkorrekturbytes

    def update_self_link(self, blob_id):
        blob_data = self._load_blob(blob_id)
        blob_data["links"]["self"] = self._generate_recovery_bytes(blob_id)
        self._save_blob(blob_id, blob_data)
        return blob_data["links"]["self"]

    def add_link(self, blob_id, link_id, link_data):
        blob_data = self._load_blob(blob_id)
        blob_data["links"][link_id] = link_data
        self._save_blob(blob_id, blob_data)

    def chair_link(self, blob_ids: List[str]):

        all_links = [link for link in [self.update_self_link(_id) for _id in blob_ids]]
        all_links_len = len(all_links)
        current_blob_id = 0
        for all_link in all_links:
            link_len = len(all_link)
            splitter = link_len // all_links_len-1
            index_ = 0
            for i in range(0, link_len, splitter):
                if index_ == current_blob_id:
                    index_ += 1
                link_port = all_link[i:i + splitter]
                self.add_link(blob_ids[index_], blob_ids[current_blob_id], {
                    "row": link_port,
                    "index": index_,
                    "max": all_links_len})
                index_ += 1
                if index_ + 1 > len(blob_ids) and len(all_link[i + splitter:]) > 1:
                    self.add_link(blob_ids[current_blob_id], blob_ids[current_blob_id], link_port)
            current_blob_id += 1

    def recover_blob(self, blob_ids, check_blobs_ids):
        s = self.get_recover_blob_sorted(blob_ids, check_blobs_ids)
        r = self.sorted_to_keys(s)
        blob_data_v = self.get_data_versions(r)
        lengths = [len(b) for b in blob_data_v]
        return blob_data_v[lengths.index(max(lengths))]

    def get_recover_blob_sorted(self, blob_id, check_blobs_ids=None):

        if check_blobs_ids is None:
            check_blobs_ids = self._get_all_blob_ids()

        all_links = [self._load_blob(_id).get("links", {}).get.get(blob_id, None) for _id in check_blobs_ids]
        all_links = [_ for _ in all_links if _ is not None]
        links = sorted(all_links, key=lambda x: x.get("max", -1))
        sorted_link = {

        }
        for link in links:
            if link.get("max", -1) == -1:
                continue
            max_ = link.get("max")
            key = str(max_)
            if key not in sorted_link:
                sorted_link[key] = ["#404#"] * max_
            sorted_link[key][link.get("index")] = link.get("row")

        return sorted_link

    @staticmethod
    def sorted_to_keys(sorted_link):

        recovery_keys = []

        for key, value in sorted_link.items():
            if "#404#" in value:
                continue
            recovery_keys.append(''.join(value))

        return recovery_keys

    def get_data_versions(self, recovery_keys):

        version_data = []
        for r_keys in recovery_keys:
            try:
                version_data.append(self.rs.decode(r_keys))
            except:
                print(f"Could not decode with key {recovery_keys.index(r_keys)}:{len(recovery_keys)}")

        return version_data

    def create_blob(self, data: bytes, blob_id=None):
        blob_data = {"data": data, "links": {"self": b""}}
        if blob_id is None:
            blob_id = self._generate_blob_id()
            self._save_blob(blob_id, blob_data)
            return blob_id
        else:
            return blob_data

    def read_blob(self, blob_id):
        blob_data = self._load_blob(blob_id)
        return blob_data["data"]

    def update_blob(self, blob_id, data):
        blob_data = self._load_blob(blob_id)
        blob_data["data"] = data
        self._save_blob(blob_id, blob_data)

    def delete_blob(self, blob_id):
        blob_file = self._get_blob_filename(blob_id)
        if os.path.exists(blob_file):
            os.remove(blob_file)

    @staticmethod
    def _generate_blob_id():
        return str(hash(os.urandom(32)))

    def _get_blob_filename(self, blob_id):
        return os.path.join(self.storage_directory, blob_id)

    def _get_all_blob_ids(self):
        filenames = []
        for root, dirs, files in os.walk(self.storage_directory):
            for file in files:
                filenames.append(file)
        return filenames

    def _save_blob(self, blob_id, blob_data):
        blob_file = self._get_blob_filename(blob_id)
        with open(blob_file, 'wb') as f:
            pickle.dump(blob_data, f)

    def _load_blob(self, blob_id):
        blob_file = self._get_blob_filename(blob_id)
        if not os.path.exists(blob_file):
            return self.create_blob(pickle.dumps({}), blob_id)
        with open(blob_file, 'rb') as f:
            return pickle.load(f)

    def _generate_recovery_bytes(self, blob_id):
        blob_data = self._load_blob(blob_id).get("data", b"")
        return self.rs.encode(blob_data)


class BlobFile(io.IOBase):
    def __init__(self, filename:str, mode='r', storage=None, key=None):
        if filename.startswith('/') or filename.startswith('\\'):
            filename = filename[1:]
        self.filename = filename
        self.blob_id, self.folder, self.datei = self.path_splitter(filename)
        self.mode = mode
        if storage is None:
            storage = BlobStorage()
        self.storage = storage
        self.data = b""
        if key is not None:
            if Code.decrypt_symmetric(Code.encrypt_symmetric("test", key), key) != "test":
                raise ValueError("Invalid Key")
        self.key = key

    @staticmethod
    def path_splitter(filename):
        pfad_obj = Path(filename)
        # Extrahieren der Bestandteile
        pfad_teile = pfad_obj.parts
        # Das erste Element
        erstes_element = pfad_teile[0]
        # Die Datei (oder das letzte Element)
        datei = pfad_teile[-1]
        # Alle Elemente in der Mitte
        mittel_teile = pfad_teile[1:-1] if len(pfad_teile) > 2 else []
        blob_id = erstes_element
        folder = '|'.join(mittel_teile)
        return blob_id, folder, datei

    def __enter__(self):
        if 'r' in self.mode:
            blob_data = pickle.loads(self.storage.read_blob(self.blob_id))
            if self.folder in blob_data:
                blob_folder = blob_data[self.folder]
                if self.datei in blob_folder:
                    self.data = blob_folder[self.datei]
                if self.key is not None:
                    self.data = Code.decrypt_symmetric(self.data, self.key)
        elif 'w' in self.mode:
            self.data = b""
        else:
            raise ValueError("Invalid mode. Only 'r' and 'w' modes are supported.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if 'w' in self.mode:
            data = self.data
            if self.key is not None:
                data = Code.encrypt_symmetric(data, self.key)
            blob_data = pickle.loads(self.storage.read_blob(self.blob_id))
            if self.folder not in blob_data:
                blob_data[self.folder] = {self.datei: data}
            else:
                blob_data[self.folder][self.datei] = data

            self.storage.update_blob(self.blob_id, pickle.dumps(blob_data))

    def write(self, data: str or bytes or dict):
        if 'w' not in self.mode:
            raise ValueError("File not opened in write mode.")
        if isinstance(data, str):
            self.data += data.encode()
        elif isinstance(data, bytes):
            self.data += data
        elif isinstance(data, dict):
            self.write_yaml(data)
        else:
            raise ValueError("Invalid Data type not supported")

    # def add_save_on_disk(self, storage_id, one_time_token):
    #     self.storage.save(self.filename, storage_id, one_time_token)

    def clear(self):
        self.data = b""

    def read(self):
        if 'r' not in self.mode:
            raise ValueError("File not opened in read mode.")
        return self.data

    def read_json(self):
        if 'r' not in self.mode:
            raise ValueError("File not opened in read mode.")
        if self.data == b"":
            return {}
        return json.loads(self.data.decode())

    def write_json(self, data):
        if 'w' not in self.mode:
            raise ValueError("File not opened in write mode.")
        self.data += json.dumps(data).encode()

    def read_pickle(self):
        if 'r' not in self.mode:
            raise ValueError("File not opened in read mode.")
        if self.data == b"":
            return {}
        return pickle.loads(self.data)

    def write_pickle(self, data):
        if 'w' not in self.mode:
            raise ValueError("File not opened in write mode.")
        self.data += pickle.dumps(data)

    def read_yaml(self):
        if 'r' not in self.mode:
            raise ValueError("File not opened in read mode.")
        if self.data == b"":
            return {}
        return yaml.safe_load(self.data)

    def write_yaml(self, data):
        if 'w' not in self.mode:
            raise ValueError("File not opened in write mode.")
        yaml.dump(data, self)
