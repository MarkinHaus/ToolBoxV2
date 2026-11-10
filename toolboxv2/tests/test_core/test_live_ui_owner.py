"""Live-UI owner: WS bridge on demand (A), persisted mounts (C), nginx-gated
HTTP workers (D), loopback-only bind.

Run: python -m unittest toolboxv2.tests.test_core.test_live_ui_owner -v
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml


class TestAttachWsBridge(unittest.TestCase):
    def _app_with_ws(self):
        from toolboxv2.utils.workers.fast_tb import FastTB
        app = FastTB(title="owner")

        @app.websocket("/ws/open_chat")
        class Chat:
            async def on_message(self, *a, **k):
                return None

        return app

    def test_no_context_means_no_ws(self):
        from toolboxv2.utils.workers.fast_tb import FastTB
        self.assertFalse(FastTB(title="x").attach_ws_bridge())

    def test_registers_handlers_and_starts_bridge_once(self):
        app = self._app_with_ws()
        handler = mock.Mock()
        tb_app = SimpleNamespace()
        app._ws_ctx = (handler, "cfg", tb_app, "worker", "loopfn")
        self.assertTrue(app.attach_ws_bridge())
        self.assertTrue(app.attach_ws_bridge())
        handler._start_ws_bridge.assert_called_once_with("cfg", tb_app, "worker", "loopfn")
        self.assertIn("ws/open_chat", tb_app.websocket_handlers)

    def test_already_running_infra_is_not_restarted(self):
        app = self._app_with_ws()
        handler = mock.Mock()
        app._ws_ctx = (handler, "cfg", SimpleNamespace(), "worker", "loopfn")
        app._ws_bridge_started = True
        self.assertTrue(app.attach_ws_bridge())
        handler._start_ws_bridge.assert_not_called()


class _Owner:
    def __init__(self):
        self._mounted_sources = {}
        self.ws_attached = 0

    def mount_app(self, joiner, prefix, source):
        self._mounted_sources[source] = {"prefix": prefix}
        return []

    def unmount_app(self, source):
        return self._mounted_sources.pop(source, None) is not None

    def attach_ws_bridge(self):
        self.ws_attached += 1
        return True


class _FakeDBApp:
    """Interface-level stand-in for app.run_any(TBEF.DB.GET/SET)."""
    def __init__(self):
        self.store = {}

    def run_any(self, fn, query=None, data=None, get_results=False):
        name = getattr(fn, "value", str(fn))
        if name == "set":
            self.store[query] = data
            return SimpleNamespace(is_error=lambda: False, get=lambda: None)
        if query in self.store:
            return SimpleNamespace(is_error=lambda: False, get=lambda: [self.store[query]])
        return SimpleNamespace(is_error=lambda: True, get=lambda: None)


class TestOwnerMounts(unittest.TestCase):
    def setUp(self):
        from toolboxv2.utils.workers.fast import tray_api
        from toolboxv2.utils.system import user_apps
        self.tray = tray_api
        root = Path(tempfile.mkdtemp(prefix="tb_owner_"))
        d = root / "isaa"
        d.mkdir()
        (d / "feature.yaml").write_text(yaml.safe_dump({"name": "isaa", "user_facing": [
            {"name": "isaa", "mount": {"module": "m.ui", "attr": "app", "prefix": "isaa"}}]}))
        (d / ".installed").touch()
        self.db = _FakeDBApp()
        self.patches = [
            mock.patch.object(user_apps, "_features_dir", return_value=root),
            mock.patch("toolboxv2.get_app", return_value=self.db),
        ]
        for p in self.patches:
            p.start()
        self.owner = _Owner()
        tray_api.set_owner_app(self.owner)

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tray.set_owner_app(None)

    def _joiner(self, ws=True):
        return SimpleNamespace(_ws_routes=["/ws/open_chat"] if ws else [], title="ISAA")

    def test_ws_app_attaches_bridge(self):
        with mock.patch.object(self.tray, "reload_from_src", return_value=self._joiner(ws=True)):
            res = self.tray._cmd_start_app("isaa")
        self.assertTrue(res["merged"])
        self.assertTrue(res["ws"])
        self.assertEqual(self.owner.ws_attached, 1)

    def test_plain_app_does_not_touch_ws(self):
        with mock.patch.object(self.tray, "reload_from_src", return_value=self._joiner(ws=False)):
            res = self.tray._cmd_start_app("isaa")
        self.assertNotIn("ws", res)
        self.assertEqual(self.owner.ws_attached, 0)

    def test_started_app_is_persisted_and_restored_by_new_owner(self):
        with mock.patch.object(self.tray, "reload_from_src", return_value=self._joiner()):
            self.tray._cmd_start_app("isaa")
            self.assertEqual(self.tray.load_mounted_apps(), ["isaa"])
            # failover: a fresh owner takes over the port
            new_owner = _Owner()
            self.tray.set_owner_app(new_owner)
            restored = self.tray.restore_mounted_apps()
        self.assertTrue(restored["isaa"]["merged"])
        self.assertIn("app:isaa", new_owner._mounted_sources)
        self.assertEqual(json.loads(self.db.store[self.tray._MOUNTS_KEY]), ["isaa"])

    def test_unmount_removes_from_persisted_list(self):
        with mock.patch.object(self.tray, "reload_from_src", return_value=self._joiner()):
            self.tray._cmd_start_app("isaa")
        self.assertTrue(self.tray._cmd_unmount_app("app:isaa")["unmounted"])
        self.assertEqual(self.tray.load_mounted_apps(), [])

    def test_db_failure_never_breaks_start(self):
        with mock.patch("toolboxv2.get_app", side_effect=RuntimeError("db down")), \
             mock.patch.object(self.tray, "reload_from_src", return_value=self._joiner()):
            self.assertTrue(self.tray._cmd_start_app("isaa")["merged"])
            self.assertEqual(self.tray.load_mounted_apps(), [])


class TestWorkerManagerPolicy(unittest.TestCase):
    def test_live_ui_is_loopback_only(self):
        from toolboxv2.utils.clis.cli_worker_manager import live_ui_bind_host
        cfg = lambda h: SimpleNamespace(manager=SimpleNamespace(live_ui_host=h))  # noqa: E731
        self.assertEqual(live_ui_bind_host(cfg("0.0.0.0")), "127.0.0.1")
        self.assertEqual(live_ui_bind_host(cfg("192.168.1.5")), "127.0.0.1")
        self.assertEqual(live_ui_bind_host(cfg("127.0.0.1")), "127.0.0.1")
        self.assertEqual(live_ui_bind_host(cfg("localhost")), "localhost")

    def test_http_workers_only_with_nginx(self):
        from toolboxv2.utils.clis.cli_worker_manager import WorkerManager
        mgr = WorkerManager.__new__(WorkerManager)
        mgr.config = SimpleNamespace(http_worker=SimpleNamespace(workers=4),
                                     nginx=SimpleNamespace(enabled=False))
        self.assertEqual(mgr._http_worker_count(), 0)
        mgr.config.nginx.enabled = True
        self.assertEqual(mgr._http_worker_count(), 4)



class TestSingleLocalWsPort(unittest.TestCase):
    """Locally ONE WS port: worker manager == Tauri == FastTB == frontends."""

    def _cfg(self, ws=8468, http=8000, workers=4, nginx=False):
        return SimpleNamespace(ws_worker=SimpleNamespace(port=ws),
                               http_worker=SimpleNamespace(port=http, workers=workers),
                               nginx=SimpleNamespace(enabled=nginx))

    def test_local_uses_canonical_port(self):
        from toolboxv2.utils.clis.cli_worker_manager import ws_base_port
        self.assertEqual(ws_base_port(self._cfg()), 8468)  # was http+100 = 8100 before

    def test_nginx_collision_moves_behind_http_range(self):
        from toolboxv2.utils.clis.cli_worker_manager import ws_base_port
        self.assertEqual(ws_base_port(self._cfg(ws=8001, nginx=True)), 8004)
        self.assertEqual(ws_base_port(self._cfg(ws=8468, nginx=True)), 8468)

    def test_all_local_defaults_agree(self):
        import re
        from pathlib import Path
        import yaml
        from toolboxv2 import tb_root_dir
        from toolboxv2.utils.workers.config import WSWorkerConfig
        from toolboxv2.utils.manifest.schema import WSWorkerInstance
        port = WSWorkerConfig().port
        self.assertEqual(port, 8468)
        self.assertEqual(WSWorkerInstance().port, port)
        shipped = yaml.safe_load((tb_root_dir / "tb-manifest.yaml").read_text(encoding="utf-8"))
        self.assertEqual(shipped["ws_worker"]["port"], port)
        example = yaml.safe_load((tb_root_dir / "utils/manifest/example-manifest.yaml").read_text(encoding="utf-8"))
        self.assertEqual(example["workers"]["websocket"][0]["port"], port)
        for rel in ("mods/isaa/ui/static/ws.js", "mods/isaa/extras/live_obs_server.py",
                    "utils/workers/fast_tb.py", "exotic/tb_browser/src/background.js"):
            src = (tb_root_dir / rel).read_text(encoding="utf-8")
            self.assertNotRegex(src, r"__TB_WS_PORT__\s*\|\|\s*'(?!%d)\d+'" % port, rel)
            self.assertNotRegex(src, r":5001'|:8100'", rel)
        rust = (tb_root_dir / "simple-core/src-tauri/src/worker_manager.rs").read_text(encoding="utf-8")
        self.assertRegex(rust, r"DEFAULT_WS_PORT: u16 = %d;" % port)


if __name__ == "__main__":
    unittest.main()
