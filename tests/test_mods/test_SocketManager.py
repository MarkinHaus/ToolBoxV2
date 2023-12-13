import threading
import unittest
import time

from toolboxv2 import App
from toolboxv2.utils.toolbox import ApiOb
from toolboxv2.mods.SocketManager import Tools, SocketType


class TestRestrictor(unittest.TestCase):
    t0 = None
    app = None

    @classmethod
    def setUpClass(cls):
        # Code, der einmal vor allen Tests ausgeführt wird
        cls.t0 = time.perf_counter()
        cls.app = App('test-SocketManager')
        cls.app.mlm = 'I'
        cls.app.debug = True
        cls.app.load_mod('SocketManager')
        cls.tool: Tools = cls.app.get_mod('SocketManager')

    @classmethod
    def tearDownClass(cls):
        cls.app.logger.info('Closing APP')
        cls.app.config_fh.delete_file()
        cls.app.remove_all_modules()
        cls.app.save_exit()
        cls.app.exit()
        cls.app.logger.info(f'Accomplished in {time.perf_counter() - cls.t0}')

    def test_show_version(self):
        command = ApiOb(
            data={'username': 'test', 'password': 'test', 'email': 'test@test.com', 'invitation': 'test'},
            token='')
        res = self.app.run_function('Version', [command, ])
        self.assertEqual(res, self.tool.version)

    def test_echo_ip_server(self):
        s_thread = threading.Thread(target=self.tool.run_as_ip_echo_server_a, args=("test-server", "", 8080, 2))
        s_thread.start()

        time.sleep(2)

        self.tool.create_socket("test-client", host="localhost", port=8080)

        time.sleep(2)

        self.tool.create_socket("test-client2", host="localhost", port=8080)

        time.sleep(2)

        # self.tool.create_socket("exit", host="localhost", port=8080)

        time.sleep(4)

        self.assertEqual(self.tool.version, self.tool.version)

    def test_peer2peer_connection(self):
        sender_p1 = self.tool.create_socket("peer1-client",
                                                        host="localhost",
                                                        port=8080,
                                                        endpoint_port=8081,
                                                        type_id=SocketType.peer)
        self.assertIsNone(sender_p1)
