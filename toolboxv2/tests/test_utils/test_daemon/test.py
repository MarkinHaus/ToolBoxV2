import unittest
from toolboxv2.tests.a_util import async_test
from toolboxv2.utils.daemon import DaemonUtil


class TestDaemonUtil(unittest.TestCase):

    async def test_daemon_util(self):
        daemon_util = DaemonUtil(class_instance=None, host='0.0.0.0', port=6582, t=False,
                                 app=None, peer=False, name='daemonApp-server',
                                 on_register=None, on_client_exit=None, on_server_exit=None,
                                 unix_socket=False)

        # Test initialization
        self.assertFalse(daemon_util.async_initialized)

        with self.assertRaises(Exception):
            await DaemonUtil(class_instance=None, host='0.0.0.0', port=6582, t=False,
                             app=None, peer=False, name='daemonApp-server',
                             on_register=None, on_client_exit=None, on_server_exit=None,
                             unix_socket=False)

        daemon_util = await DaemonUtil(class_instance=None, host='0.0.0.0', port=6582, t=False,
                                       app=None, peer=False, name='daemonApp-server',
                                       on_register=None, on_client_exit=None, on_server_exit=None,
                                       unix_socket=False, test_override=True)

        self.assertTrue(daemon_util.async_initialized)
        # Test send
        data = {'key': 'value'}
        result = await daemon_util.send(data)
        self.assertEqual(result, "Data Transmitted")
        print(result)

        # Test connect
        # app = MagicMock()
        # await daemon_util.connect(app)

        # Test a_exit

        self.assertTrue(daemon_util.alive)

        await daemon_util.a_exit()

        self.assertFalse(daemon_util.alive)


# Apply async_test decorator to each async test method
TestDaemonUtil.test_daemon_util = async_test(TestDaemonUtil.test_daemon_util)
