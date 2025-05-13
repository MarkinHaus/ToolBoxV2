import asyncio
import threading
import time
import unittest

from toolboxv2 import get_app
from toolboxv2.tests.a_util import async_test
from toolboxv2.utils.daemon import DaemonUtil
from toolboxv2.utils.proxy import ProxyUtil


class MiniHelper:

    def __init__(self, name):
        self.name = name

    def me(self):
        print(self.name)
        return {"0": "Iam " + self.name}

    def you(self, s):
        print(s, "called you on", self.name)
        return {self.name: self.name}


class TestDPIntegration(unittest.TestCase):

    async def helper(self):
        daemon = await DaemonUtil(class_instance=MiniHelper("tom"), host='127.0.0.1', port=6183, t=True,
                                  app=None, peer=False, name='tom',
                                  on_register=None, on_client_exit=None, on_server_exit=None,
                                  unix_socket=False, test_override=True)
        await asyncio.sleep(20)
        await daemon.a_exit()

    async def test_helper(self):
        if not get_app(name="test").local_test:
            return
        app = get_app(name="test")
        await app.get_mod("SocketManager")


        # await app.init_module(socketManager)

        threading.Thread(target=self.helper, daemon=True).start()

        await asyncio.sleep(1)

        proxy_class = await ProxyUtil(class_instance=MiniHelper("mia"), host='127.0.0.1', port=6183, timeout=5,
                                      app=app,
                                      remote_functions=["you"], peer=False, name='mia', do_connect=True,
                                      unix_socket=False, test_override=True)
        await proxy_class.verify()
        time.sleep(1)
        p = await proxy_class.you(proxy_class.name)
        m = proxy_class.me()
        self.assertEqual(p, {'tom': 'tom'})
        self.assertEqual(m, {'0': 'Iam mia'})
        nd = proxy_class.r
        self.assertEqual(nd, "No data")
        print("Proxy class", p)
        print("Proxy class 2", nd)
        print("Proxy class 3", m)

        await proxy_class.reconnect()
        await proxy_class.verify()
        time.sleep(1)
        p = await proxy_class.you(proxy_class.name)
        m = proxy_class.me()
        self.assertEqual(p, {'tom': 'tom'})
        self.assertEqual(m, {'0': 'Iam mia'})
        nd = proxy_class.r
        self.assertEqual(nd, "No data")
        print("# Proxy class", p)
        print("# Proxy class 2", nd)
        print("# Proxy class 3", m)

        await proxy_class.disconnect()


TestDPIntegration.test_helper = async_test(TestDPIntegration.test_helper)
TestDPIntegration.helper = async_test(TestDPIntegration.helper)
