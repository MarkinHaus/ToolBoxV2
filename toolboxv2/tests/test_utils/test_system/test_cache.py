import os
import time
import unittest
from tempfile import TemporaryDirectory

from toolboxv2.utils.system import FileCache, MemoryCache


class TestFileCache(unittest.TestCase):

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.cache_file = os.path.join(self.temp_dir.name, 'test_cache.db')

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_set_and_get(self):
        cache = FileCache(folder=self.temp_dir.name, filename='test_cache.db')
        cache.set('key', 'value')
        self.assertEqual(cache.get('key'), 'value')
        cache.cleanup()

    def test_get_nonexistent_key(self):
        cache = FileCache(folder=self.temp_dir.name, filename='test_cache.db')
        self.assertIsNone(cache.get('nonexistent_key'))
        cache.cleanup()

    def test_folder_creation(self):
        cache_folder = os.path.join(self.temp_dir.name, 'cache_folder')
        self.assertFalse(os.path.exists(cache_folder))
        self.assertFalse(os.path.isdir(cache_folder))
        cache = FileCache(folder=cache_folder, filename='test_cache.db')
        self.assertTrue(os.path.exists(cache_folder))
        self.assertTrue(os.path.isdir(cache_folder))
        cache.cleanup()

    def test_folder_cleanup(self):
        cache_folder = os.path.join(self.temp_dir.name, 'cache_folder')
        os.makedirs(cache_folder, exist_ok=True)
        self.assertTrue(os.path.exists(cache_folder))
        cache = FileCache(folder=cache_folder, filename='test_cache.db')
        self.assertTrue(os.path.exists(cache_folder))
        cache.cleanup()  # Remove reference to the FileCache instance
        self.assertFalse(os.path.exists(cache_folder+'/test_cache.db'))


class TestMemoryCache(unittest.TestCase):

    def test_set_and_get(self):
        cache = MemoryCache(maxsize=100, ttl=300)
        cache.set('key', 'value')
        self.assertEqual(cache.get('key'), 'value')

    def test_get_nonexistent_key(self):
        cache = MemoryCache(maxsize=100, ttl=300)
        self.assertIsNone(cache.get('nonexistent_key'))

    def test_maxsize_zero(self):
        cache = MemoryCache(maxsize=0, ttl=300)
        # Attempt to set a key-value pair
        with self.assertRaises(ValueError):
            cache.set('key', 'value')
        # Attempt to get the value for the key
        value = cache.get('key')
        # Verify that the value is None since the cache size is 0
        self.assertIsNone(value)

    def test_maxsize_negative(self):
        cache = MemoryCache(maxsize=-1, ttl=300)
        # Attempt to set a key-value pair
        with self.assertRaises(ValueError):
            cache.set('key', 'value')
        # Attempt to get the value for the key
        value = cache.get('key')
        # Verify that the value is None since the cache size is negative
        self.assertIsNone(value)

    def test_ttl_timer(self):
        cache = MemoryCache(maxsize=100, ttl=1)  # TTL set to 1 second
        # Set a key-value pair
        cache.set('key', 'value')
        # Wait for 2 seconds
        time.sleep(2)
        # Attempt to get the value for the key after TTL expiry
        value = cache.get('key')
        # Verify that the value is None since the TTL has expired
        self.assertIsNone(value)
