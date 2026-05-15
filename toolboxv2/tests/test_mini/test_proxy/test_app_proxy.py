import unittest

from toolboxv2.tests.a_util import async_test
from toolboxv2.utils.proxy import ProxyUtil


class TestProxyUtil(unittest.TestCase):

    async def test_proxy_util(self):

        proxy_util = ProxyUtil(class_instance=None, host='0.0.0.0', port=6581, timeout=15, app=None,
                               remote_functions=None, peer=False, name='daemonApp-client', do_connect=False,
                               unix_socket=False)

        # Test initialization
        self.assertFalse(proxy_util.async_initialized)

        with self.assertRaises(Exception):
            await ProxyUtil(class_instance=None, host='0.0.0.0', port=6581, timeout=15, app=None,
                                         remote_functions=None, peer=False, name='daemonApp-client', do_connect=True,
                                         unix_socket=False)

        with self.assertRaises(Exception):
            await ProxyUtil(class_instance=None, host='0.0.0.0', port=6581, timeout=15, app=None,
                                         remote_functions=None, peer=False, name='daemonApp-client', do_connect=True,
                                         unix_socket=False, test_override=True)

        proxy_util = await ProxyUtil(class_instance=None, host='0.0.0.0', port=6581, timeout=15, app=None,
                                     remote_functions=None, peer=False, name='daemonApp-client', do_connect=False,
                                     unix_socket=False, test_override=True)

        self.assertTrue(proxy_util.async_initialized)


# Apply async_test decorator to each async test method
TestProxyUtil.test_proxy_util = async_test(TestProxyUtil.test_proxy_util)
