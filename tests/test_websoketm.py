import json
import queue
import time
import unittest
from unittest.mock import MagicMock, patch

from toolboxv2 import App
from toolboxv2.mods.WebSocketManager import Tools


class TestWebSocketManager(unittest.TestCase):
    t0 = None
    app = None

    @classmethod
    def setUpClass(cls):
        # Code, der einmal vor allen Tests ausgeführt wird
        cls.t0 = time.time()
        cls.app = App('test-WebSocketManager')
        cls.app.mlm = "I"
        cls.app.debug = True
        cls.app.load_mod('WebSocketManager')
        cls.tool = cls.app.get_mod('WebSocketManager')
        cls.app.new_ac_mod('WebSocketManager')
        cls.websocket_id = 'app-HotReload-DESKTOP-CI57V1L4cb26814-887a-49e1-9e74-8e560a7b7b14CloudM-Signed'
        cls.url = 'ws://localhost:5000/ws'

    @classmethod
    def tearDownClass(cls):
        cls.app.logger.info('Closing APP')
        cls.app.config_fh.delete_file()
        cls.app.remove_all_modules()
        cls.app.save_exit()
        cls.app.exit()
        cls.app.logger.info(f'Accomplished in {time.time() - cls.t0}')

    def test_get_vt(self):
        result = self.tool.get_vt('test_uid')
        self.assertEqual(result, 'test_uidVTInstance')

    def test_show_version(self):
        result = self.tool.show_version()
        self.assertEqual(result, self.tool.version)

    def test_create_websocket(self):
        # result = self.tool.create_websocket(self.websocket_id)
        # time.sleep(5)
        # self.assertNotEqual(result, None)
        # self.tool.close_websocket(self.websocket_id)
        #
        # self.assertEqual(self.tool.active_connections_client, {})
        pass

    def test_get_sender_receiver_que_ws(self):
        # send_queue, recv_queue = self.tool.get_sender_receiver_que_ws(self.url, self.websocket_id)
        # time.sleep(5)
        ## Überprüfen Sie, ob die zurückgegebenen Queues Instanzen der Klasse Queue sind
        # self.assertIsInstance(send_queue, queue.Queue)
        # self.assertIsInstance(recv_queue, queue.Queue)
        #
        ## Überprüfen Sie, ob die Queues leer sind
        # self.assertTrue(send_queue.empty())
        # self.assertTrue(recv_queue.empty())
        # print("active_connections_client", self.tool.active_connections_client)
        #
        # send_queue.put('exit')
        # recv_queue.put('exit')
        pass
