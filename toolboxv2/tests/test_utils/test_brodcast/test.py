import threading
import unittest
import time

from toolboxv2.utils.brodcast.server import make_known
from toolboxv2.utils.brodcast.client import start_client


class TestBrodcast(unittest.TestCase):

    def test_start_client(self):
        threading.Thread(target=self.threddet_client, daemon=True).start()
        time.sleep(2)
        s = make_known("tom", port=44666, get_flag=b"R")
        print("CRCVED", s)
        self.assertIsInstance(s, dict)
        self.assertIn("host", s.keys())
        self.assertEqual(s.get("host"), "Local test router")

    def threddet_client(self):
        router = start_client("Local test router", port=44666)
        source_id, connection = next(router)
        print(f"Infos :{source_id}, connection :{connection}")
        router.send("e")
        router.close()
        self.assertEqual(source_id, "tom")

    def test_make_known(self):
        res = make_known("tom", port=44666)
        self.assertIsInstance(res, dict)
        self.assertIn("port", res.keys())
        self.assertIn("host", res.keys())
        self.assertEqual(res.get("port"), 0)
