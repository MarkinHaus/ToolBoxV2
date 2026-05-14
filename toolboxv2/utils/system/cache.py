import os
import shelve
import time


class FileCache:
    def __init__(self, folder='./FileCache/', filename='cache.db'):
        self.filename = filename
        self.folder = folder
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    def get(self, key):
        try:
            with shelve.open(self.filename) as db:
                return db.get(key.replace('\x00', ''))
        except Exception:
            return None

    def set(self, key, value):
        try:
            with shelve.open(self.filename, writeback=True) as db:
                db[key.replace('\x00', '')] = value
        except Exception:
            return None

    def cleanup(self):
        # Check if the folder exists and is empty
        # Also, remove the file associated with the cache
        if os.path.exists(self.filename):
            os.remove(self.filename)
        if len(os.listdir(self.folder)) == 3:
            for filename in os.listdir(self.folder):
                os.remove(self.folder+filename)
            os.rmdir(self.folder)



class MemoryCache:
    def __init__(self, maxsize=100, ttl=300):
        self._data = {}
        self._maxsize = maxsize
        self._ttl = ttl

    def get(self, key):
        entry = self._data.get(key)
        if entry is None:
            return None
        if time.monotonic() - entry[1] > self._ttl:
            del self._data[key]
            return None
        return entry[0]

    def set(self, key, value):
        if len(self._data) >= self._maxsize:
            # Ältesten raus
            oldest = min(self._data, key=lambda k: self._data[k][1])
            del self._data[oldest]
        self._data[key] = (value, time.monotonic())
