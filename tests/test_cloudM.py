#!/usr/bin/env python
import time

from rich.traceback import install
import os
import threading
import mock
from unittest.mock import patch

from toolboxv2.utils.toolbox import ApiOb

install(show_locals=True)

"""Tests for `cloudM` package."""
from coverage.annotate import os

from toolboxv2 import App, Style

import unittest

from toolboxv2.mods.cloudM import Tools


class TestCloudM(unittest.TestCase):
    t0 = None
    api = None
    app = None

    @classmethod
    def setUpClass(cls):
        # Code, der einmal vor allen Tests ausgeführt wird
        cls.t0 = time.time()
        cls.app = App("test")
        # api = threading.Thread(target=cls.app.run_any, args=('api_manager', 'start-api', ['start-api', 'test-api']))
        # api.start()
        time.sleep(0.6)
        cls.app.mlm = "I"
        cls.app.debug = True
        cls.app.load_mod("cloudM")
        cls.tool = cls.app.get_mod("cloudM")
        cls.app.new_ac_mod("cloudM")
        #cls.app.run_function("first-web-connection", ['first-web-connection', 'http://127.0.0.1:5000/api'])
        #cls.app.HELPER['api-thread'] = api

    @classmethod
    def tearDownClass(cls):
        cls.app.logger.info('Closing API')
        cls.app.config_fh.delete_file()

        if 'api-thread' in cls.app.HELPER.keys():
            cls.app.run_any('api_manager', 'stop-api', ['start-api', 'test-api'])
            cls.app.HELPER['api-thread'].join()
            del cls.app.HELPER['api-thread']

        cls.app.remove_all_modules()
        cls.app.save_exit()
        cls.app.exit()

        cls.app.logger.info(f"Accomplished in {time.time() - cls.t0}")

    def test_show_version(self):
        comd = []
        res = self.app.run_function("Version", comd)
        self.assertEqual(res, "0.0.1")

    def test_new_module(self):
        comd = ["", "test_module"]
        res = self.app.run_function("NEW", comd)
        self.assertTrue(res)
        res = self.app.run_function("NEW", comd)
        self.assertFalse(res)
        comd = ["", "cloudM"]
        res = self.app.run_function("NEW", comd)
        self.assertFalse(res)
        self.assertTrue(os.path.exists("./mods_dev/test_module.py"))
        os.remove("./mods_dev/test_module.py")
        self.assertFalse(os.path.exists("./mods_dev/test_module.py"))

    def test_prep_system_initial(self):
        self.assertTrue(self.tool.prep_system_initial(['do-root'], self.app))

    def test_delete_user_instance(self):

        res = self.tool.delete_user_instance("00000")
        self.assertEqual("User instance not found", res)
        self.assertEqual(self.tool.user_instances, {})
        self.assertEqual(self.tool.live_user_instances, {})

    def test_get_user_instance_valid(self):
        uid = '123'
        username = 'test_user'
        token = 'abc'
        result = self.tool.get_user_instance(uid, username, token)

        self.assertIsInstance(result, dict)

        result_keys = list(result.keys())
        self.assertTrue('token' in result_keys)
        self.assertTrue('SiID' in result_keys)
        self.assertTrue('webSocketID' in result_keys)
        self.assertTrue('live' in result_keys)
        self.assertTrue('save' in result_keys)
        self.assertTrue(uid == result['save']['uid'])

        res = self.tool.delete_user_instance(uid)
        self.assertEqual("Instance deleted successfully", res)
        self.assertEqual(self.tool.user_instances, {})
        self.assertEqual(self.tool.live_user_instances, {})

        # instance = {
        #     'save': {'uid': uid, 'level': 0, 'mods': [], 'username': username},
        #     'live': {},
        #     'webSocketID': self.get_web_socket_id(uid),
        #     'SiID': self.get_si_id(uid),
        #     'token': token
        # }

    def test_save_user_instances(self):
        uid = '123'
        username = 'test_user'
        token = 'abc'
        instance = self.tool.get_user_instance(uid, username, token)

        self.assertIn(instance['SiID'], self.tool.live_user_instances.keys())

        instance_c = self.tool.get_user_instance(uid)

        self.assertEqual(instance_c, instance)
        self.assertEqual(instance['save']['username'], instance_c['save']['username'])

        self.assertEqual(self.tool.user_instances[instance['SiID']], instance['webSocketID'])
        self.assertEqual(self.tool.live_user_instances[instance['SiID']], instance)
        res = self.tool.delete_user_instance(uid)
        self.assertEqual("Instance deleted successfully", res)
        self.assertEqual(self.tool.user_instances, {})
        self.assertEqual(self.tool.live_user_instances, {})
    def test_create_user(self):

        self.tool.prep_system_initial(['do-root'], self.app)

        command = ApiOb(
            data={'username': 'test', 'password': 'test', 'email': 'test@test.com', 'invitation': 'test'},
            token='')
        result = self.tool.create_user([command], self.app)

        self.assertTrue(result not in self.tool.user_instances.keys())

        self.app.run_any("db", "set", ["test", "Valid"])
        result = self.tool.create_user([command], self.app)

        valid_id = False
        for key, ids in self.tool.user_instances.items():
            if result == ids:
                valid_id = True
                result = key
                break

        self.assertTrue(valid_id)
        self.assertIn(result, self.tool.user_instances.keys())

        valid_id = False
        for key, ids in self.tool.live_user_instances.items():
            if result == key:
                valid_id = True
                break

        self.assertTrue(valid_id)
        self.assertIn(result, self.tool.live_user_instances.keys())
        instance = self.tool.live_user_instances[result]

        self.assertIsInstance(result, str)
        self.assertIsInstance(instance, dict)  # Überprüfen, ob das Ergebnis eine Zeichenkette ist
        res = self.tool.delete_user_instance(instance['save']['uid'])
        self.assertEqual("Instance deleted successfully", res)
        self.assertEqual(self.tool.user_instances, {})
        self.assertEqual(self.tool.live_user_instances, {})

    def test_user_welcome(self):

        self.tool.prep_system_initial(['do-root'], self.app)

        command = ApiOb(
            data={'username': 'test', 'password': 'test', 'email': 'test@test.com', 'invitation': 'test'},
            token='')

        self.app.run_any("db", "set", ["test", "Valid"])
        webSocketID = self.tool.create_user([command], self.app)  # Erstellen Sie zuerst einen Benutzer
        command.data["webSocketID"] = webSocketID

        valid_id = False
        result = ""
        for key, ids in self.tool.user_instances.items():
            if webSocketID == ids:
                valid_id = True
                result = key
                break

        self.assertTrue(valid_id)
        self.assertIn(result, self.tool.user_instances.keys())

        valid_id = False
        for key, ids in self.tool.live_user_instances.items():
            if result == key:
                valid_id = True
                break

        self.assertTrue(valid_id)
        self.assertIn(result, self.tool.live_user_instances.keys())
        instance = self.tool.live_user_instances[result]

        instance['save']['mods'] = ['welcome']

        instance = self.tool.hydrate_instance(instance)
        self.tool.save_user_instances(instance)

        self.app.new_ac_mod("VirtualizationTool")
        res = None
        if self.app.run_function('set-ac', instance['live']['v-welcome']):
            res = self.app.run_function('api_version', command)

        print("V-Welcome version:", res)
        self.assertIsInstance(res, str)

        self.tool.close_user_instance(instance['save']['uid'])

        self.assertEqual(instance['live'], {})

        res = self.tool.delete_user_instance(instance['save']['uid'])
        self.assertEqual("Instance deleted successfully", res)
        self.assertEqual(self.tool.user_instances, {})
        self.assertEqual(self.tool.live_user_instances, {})

# def test_upload(self):
#   comd = ["", "test_module"]
#   res = self.app.run_function("NEW", command=comd)
#   self.assertTrue(res)
#   res = self.app.run_function("upload", command=comd)
#   self.assertIsNone(res)
#   self.assertTrue(os.path.exists("./mods_dev/test_module.py"))
#   os.remove("./mods_dev/test_module.py")
#   self.assertFalse(os.path.exists("./mods_dev/test_module.py"))
#
# def test_download(self):
#    comd = []
#    res = self.app.run_function("download", command=comd)
#    self.assertFalse(res)
#

#
# def test_create_account(self):
#    pass
#    #comd = []
#    #res = self.app.run_function("create-account", command=comd)
#
#
# def test_log_in(self):
#    comd = []
#    res = self.app.run_function("login", command=comd)
#    assert res == ""
#
# def test_create_user(self):
#    comd = []
#    res = self.app.run_function("create_user", command=comd)
#    assert res == ""
#
# def test_log_in_user(self):
#    comd = []
#    res = self.app.run_function("log_in_user", command=comd)
#    assert res == ""
#
# def test_validate_jwt(self):
#    comd = []
#    res = self.app.run_function("validate_jwt", command=comd)
#    assert res == ""
#
# def test_download_api_files(self):
#    comd = []
#    res = self.app.run_function("download_api_files", command=comd)
#    assert res == ""
#
# def test_update_core(self):
#    comd = []
#    res = self.app.run_function("#update-core", command=comd)
#    assert res == ""
