import time

from toolboxv2 import get_app
from toolboxv2.mods.CloudM.AuthManager import (
    crate_local_account,
    get_invitation,
    get_user_by_name,
)
from toolboxv2.mods.CloudM.extras import create_magic_log_in
from toolboxv2.tests.a_util import async_test
from toolboxv2.tests.web_util import AsyncWebTestFramework, WebTestFramework
from toolboxv2.utils.system.session import Session


def ensure_test_user(user_name="testUser"):
    app = get_app(name="test")
    app.get_mod("DB").edit_cli("LC")
    user = get_user_by_name(app, user_name)
    if user.is_error():
        crate_local_account(app, user_name, user_name + "@simpleCore.app", get_invitation(app).get())
    link = create_magic_log_in(app, user_name)
    app.get_mod("DB").close_db()
    app.get_mod("DB").edit_cli("LC")
    return link


start_urls_init = [None, None]


def ensure_web_server_online():
    app = get_app(name="test")
    if 'test_api' not in app.get_mod("FastApi").running_apis:
        start_urls_init[0] = ensure_test_user("loot")
        start_urls_init[1] = ensure_test_user("testUser")
        app.run_any(("FastApi", "startDUG"), api_name="test_api", get_results=True).print()
        time.sleep(10)


def ensure_moc_session(session: Session):
    pass
    from fastapi.testclient import TestClient

    from toolboxv2.mods.FastApi.fast_api_main import app
    tc = TestClient(app)
    session.init()
    session.session.request = tc.request
    session.session.get = tc.get
    session.session.post = tc.post


async def test_run_valid_session_tests(headless=True):
    if not get_app(name="test").local_test:
        return
    from toolboxv2.tests.web_test import valid_session_tests
    # setup
    ensure_web_server_online()
    start_url = start_urls_init[1]
    async with AsyncWebTestFramework(headless=headless) as tf:
        await tf.create_context(
            viewport={'width': 1280, 'height': 720},
            user_agent="Mozilla/5.0 (Custom Test Agent)"
        )
        init_test_res = await tf.mimic_user_interaction(
            [{'type': 'goto', 'url': start_url}, {"type": "sleep", "time": 10},
             {"type": "test", "selector": "#inputField.inputField"}])
        tf.eval_r(init_test_res)
        await tf.save_state("valid_session_tests")
        await tf.run_tests(*valid_session_tests, evaluation=True)


test_run_valid_session_tests = async_test(test_run_valid_session_tests)


def run_valid_session_tests(valid_session_tests, headless=True, evaluation=True):
    # setup
    ensure_web_server_online()
    with WebTestFramework(headless=headless) as tf:
        tf.create_context(
            viewport={'width': 1280, 'height': 720},
            user_agent="Mozilla/5.0 (Custom Test Agent)"
        )
        if not tf.load_state("valid_session_tests"):
            test_run_valid_session_tests()
        if not tf.load_state("valid_session_tests"):
            return "No session tests loaded"
        return tf.run_tests(*valid_session_tests, evaluation=evaluation)


async def test_run_loot_session_tests(headless=True):
    if not get_app(name="test").local_test:
        return
    from toolboxv2.tests.web_test import loot_session_tests
    # setup
    ensure_web_server_online()
    start_url = start_urls_init[0]
    async with AsyncWebTestFramework(headless=headless) as tf:
        await tf.create_context(
            viewport={'width': 1280, 'height': 720},
            user_agent="Mozilla/5.0 (Custom Test Agent)"
        )
        tf.eval_r(await tf.mimic_user_interaction([{'type': 'goto', 'url': start_url}, {"type": "sleep", "time": 10},
                                                   {"type": "test", "selector": "input#inputField.inputField"}]))
        await tf.save_state("loot_session_tests")
        await tf.run_tests(*loot_session_tests, evaluation=True)


test_run_loot_session_tests = async_test(test_run_loot_session_tests)


def run_loot_session_tests(loot_session_tests, headless=True, evaluation=True):
    # setup
    ensure_web_server_online()
    with WebTestFramework(headless=headless) as tf:
        tf.create_context(
            viewport={'width': 1280, 'height': 720},
            user_agent="Mozilla/5.0 (Custom Test Agent)"
        )
        if not tf.load_state("loot_session_tests"):
            test_run_loot_session_tests()
        if not tf.load_state("loot_session_tests"):
            return "No session tests loaded"
        return tf.run_tests(*loot_session_tests, evaluation=evaluation)


async def test_run_in_valid_session_tests(headless=True):
    if not get_app(name="test").local_test:
        return
    from toolboxv2.tests.web_test import in_valid_session_tests
    # setup
    ensure_web_server_online()
    async with AsyncWebTestFramework(headless=headless) as tf:
        await tf.create_context(
            viewport={'width': 1280, 'height': 720},
            user_agent="Mozilla/5.0 (Custom Test Agent)"
        )
        await tf.run_tests(*in_valid_session_tests, evaluation=True)


test_run_in_valid_session_tests = async_test(test_run_in_valid_session_tests)


async def run_in_valid_session_tests(in_valid_session_tests, headless=True, evaluation=True):
    # setup
    ensure_web_server_online()
    async with AsyncWebTestFramework(headless=headless) as tf:
        await tf.create_context(
            viewport={'width': 1280, 'height': 720},
            user_agent="Mozilla/5.0 (Custom Test Agent)"
        )
        return await tf.run_tests(*in_valid_session_tests, evaluation=evaluation)
