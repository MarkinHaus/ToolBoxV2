from toolboxv2.tests.a_util import async_test
from toolboxv2.utils.system.session import Session
import unittest


class TestSession(unittest.TestCase):

    async def test_helper_session_invalid(self):
        s = Session('root')

        t = await s.init_log_in_mk_link("/")
        print(t)
        t1 = await s.login()
        print(t1)
        assert t1 == False

    async def test_session_invalid_log_in(self):
        s = Session('root')
        t1 = await s.login()
        print(t1)
        assert t1 == False


TestSession.test_session_invalid_log_in = async_test(TestSession.test_session_invalid_log_in)
TestSession.test_helper_session_invalid = async_test(TestSession.test_helper_session_invalid)
