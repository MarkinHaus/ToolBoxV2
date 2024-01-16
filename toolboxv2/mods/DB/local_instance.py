import json
import os

from toolboxv2 import Result, get_app
from .types import AuthenticationTypes
from ...utils.cryp import Code


class MiniDictDB:
    auth_type = AuthenticationTypes.location

    def __init__(self):
        self.data = {}
        self.key = ""
        self.location = ""

    def scan_iter(self, serch=''):
        # print(self.data)
        if not self.data:
            return []
        return [key for key in self.data.keys() if key.startswith(serch.replace('*', ''))]

    def initialize(self, location, key):
        if os.path.exists(location + 'MiniDictDB.json'):
            try:
                self.data = eval(Code.decrypt_symmetric(load_from_json(location + 'MiniDictDB.json').get('data'), key))
            except Exception as er:
                print(f"Data is currupted error : {er}")
                self.data = {}
        else:
            print(f'Could not initialize MiniDictDB with data from MiniDictDB.json')
            self.data = {}
        self.key = key
        self.location = location + 'MiniDictDB.json'
        return Result.ok()

    def get(self, key: str) -> Result:
        data = []
        data_info = ""

        if key == 'all':
            data_info = "Returning all data available "
            for key_ in self.data.items():
                data.append(key_)

        elif key == "all-k":
            data_info = "Returning all keys "
            data = list(self.data.keys())
        else:
            data_info = "Returning subset of keys "
            for key_ in self.scan_iter(key):
                val = self.data.get(key_)
                data.append(val)

        if not data:
            return Result.default_internal_error(info=f"No data found for key {key}")

        return Result.ok(data=data, data_info=data_info + key)

    def set(self, key, value):
        if key and value:
            self.data[key] = value
            return Result.ok()
        return Result.default_user_error(info=f"key is {key}, type{type(key)}, value is {value}, type{type(value)}")

    def append_on_set(self, key, value):
        if key in self.data:
            self.data[key].append(value)
            return Result.ok()
        return Result.default_user_error(info=f"key not found {key}")

    def delete(self, key, matching=False) -> Result:

        del_list = []
        n = 0

        if matching:
            for key_ in self.scan_iter():
                # Check if the key contains the substring
                if key_ in str(key, 'utf-8'):
                    n += 1
                    # Delete the key if it contains the substring
                    v = self.data.pop(key)
                    del_list.append((key_, v))
        else:
            v = self.data.pop(key)
            del_list.append((key, v))
            n += 1

        return Result.ok(data=del_list, data_info=f"Data deleted successfully removed {n} items")

    def exit(self) -> Result:
        if self.key == "":
            return Result.default_internal_error(info="No cryptographic key available")
        if self.location == "":
            return Result.default_internal_error(info="No file location available")
        data = Code().encode_code(str(self.data), self.key)
        try:
            save_to_json({"data": data}, self.location)
        except PermissionError and FileNotFoundError as f:
            return Result.custom_error(data=f, info=f"Error Exiting local DB instance {f}")

        return Result.ok()


def save_to_json(data, filename):
    """
    Speichert die übergebenen Daten in einer JSON-Datei.

    :param data: Die zu speichernden Daten.
    :param filename: Der Dateiname oder Pfad, in dem die Daten gespeichert werden sollen.
    """
    if not os.path.exists(filename):
        open(filename, 'a').close()

    with open(filename, 'w+') as file:
        json.dump(data, file, indent=4)


def load_from_json(filename):
    """
    Lädt Daten aus einer JSON-Datei.

    :param filename: Der Dateiname oder Pfad der zu ladenden Datei.
    :return: Die geladenen Daten.
    """
    if not os.path.exists(filename):
        return {'data': ''}

    with open(filename, 'r') as file:
        return json.load(file)
