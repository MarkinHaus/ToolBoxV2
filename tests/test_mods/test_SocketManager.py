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
        cls.app.new_ac_mod('SocketManager')

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

    def test_tb_controller_server2_demon_client(self):
        s, r = self.tool.create_socket("test-client", host="localhost", port=62436)

        s({"module_name": 'SocketManager', "function_name": 'Version'})
        s({"exit": True})

        if r.not_empty:
            data = r.get()
            print(data)
            self.assertEqual(data, {'data': self.tool.version})
        if r.not_empty:
            data = r.get()
            print(data)
            self.assertEqual(data, {'data': self.tool.version})

    def test_peer2peer_connection(self):
        sender_p1, reciver_p1 = self.tool.create_socket("peer1-client",
                                                        host="localhost",
                                                        port=8080,
                                                        endpoint_port=8081,
                                                        type_id=SocketType.peer)
        sender_p2, reciver_p2 = self.tool.create_socket("peer2-client",
                                                        host="localhost",
                                                        port=8081,
                                                        endpoint_port=8082,
                                                        type_id=SocketType.peer)

        sender_p1({'data': "hallo dats ist eine kleiner test von p1 -> p2"})
        sender_p2({'data': "hallo dats ist eine kleiner test von p2 -> p1"})

        p_que_data1 = reciver_p1.get(block=True)
        p_que_data2 = reciver_p2.get(block=True)

        print("Que data = ", p_que_data1)
        print("Que data = ", p_que_data2)
