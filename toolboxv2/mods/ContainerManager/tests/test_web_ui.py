"""
Unit tests for web_ui.py

Tests route registration, backend functions (docker_health, list_all_docker_containers,
update_container, reconcile_status), and dashboard HTML structure.
"""

import json
import os
import unittest

from toolboxv2.mods.ContainerManager.docker_ops import set_docker_ops
from toolboxv2.mods.ContainerManager.tests.fakes import (
    FakeDockerOps, FakeDB, FakeApp,
    make_container_spec, seed_container_in_db,
)


# ============================================================================
# Route Registration
# ============================================================================

class TestWebUIRouteRegistration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._fake_ops = FakeDockerOps()
        set_docker_ops(cls._fake_ops)
        from toolboxv2.mods.ContainerManager.web_ui import container_ui_app
        cls.app = container_ui_app

    @classmethod
    def tearDownClass(cls):
        set_docker_ops(FakeDockerOps())

    def test_route_get_cm(self):
        self.assertTrue(self.app.has_route("/cm", "GET"))

    def test_route_get_cm_slash(self):
        self.assertTrue(self.app.has_route("/cm/", "GET"))

    def test_route_get_containers(self):
        self.assertTrue(self.app.has_route("/cm/api/containers", "GET"))

    def test_route_post_containers(self):
        self.assertTrue(self.app.has_route("/cm/api/containers", "POST"))

    def test_route_get_container_detail(self):
        self.assertTrue(self.app.has_route("/cm/api/container/abc123", "GET"))

    def test_route_post_start(self):
        self.assertTrue(self.app.has_route("/cm/api/container/abc123/start", "POST"))

    def test_route_post_stop(self):
        self.assertTrue(self.app.has_route("/cm/api/container/abc123/stop", "POST"))

    def test_route_post_restart(self):
        self.assertTrue(self.app.has_route("/cm/api/container/abc123/restart", "POST"))

    def test_route_post_delete(self):
        self.assertTrue(self.app.has_route("/cm/api/container/abc123/delete", "POST"))

    def test_route_get_logs(self):
        self.assertTrue(self.app.has_route("/cm/api/container/abc123/logs", "GET"))

    def test_route_get_docker_health(self):
        self.assertTrue(self.app.has_route("/cm/api/docker-health", "GET"),
                        msg="Missing GET /cm/api/docker-health")

    def test_route_get_all_containers(self):
        self.assertTrue(self.app.has_route("/cm/api/all-containers", "GET"),
                        msg="Missing GET /cm/api/all-containers")

    def test_route_post_update(self):
        self.assertTrue(self.app.has_route("/cm/api/container/abc123/update", "POST"),
                        msg="Missing POST /cm/api/container/{id}/update")

    def test_route_post_reconcile(self):
        self.assertTrue(self.app.has_route("/cm/api/container/abc123/reconcile", "POST"),
                        msg="Missing POST /cm/api/container/{id}/reconcile")

    def test_route_get_topology(self):
        self.assertTrue(self.app.has_route("/cm/api/topology", "GET"),
                        msg="Missing GET /cm/api/topology")


# ============================================================================
# docker_health
# ============================================================================

class TestDockerHealthFunction(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.ops = FakeDockerOps()
        set_docker_ops(self.ops)
        self.db = FakeDB()
        self.app = FakeApp(self.db)

    async def asyncTearDown(self):
        set_docker_ops(FakeDockerOps())

    async def test_online_no_containers(self):
        from toolboxv2.mods.ContainerManager import docker_health
        result = await docker_health(app=self.app, admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))
        self.assertTrue(result.is_ok())
        data = result.get()
        self.assertTrue(data["docker_available"])
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["total_containers"], 0)

    async def test_online_mixed_containers(self):
        self.ops.add_container("c1", "tb", "tbv2:latest",
                               labels={"managed-by": "ContainerManager"})
        self.ops.add_container("c2", "redis", "redis:7", labels={})
        from toolboxv2.mods.ContainerManager import docker_health
        data = (await docker_health(app=self.app, admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))).get()
        self.assertEqual(data["total_containers"], 2)
        self.assertEqual(data["tb_managed"], 1)
        self.assertEqual(data["external"], 1)

    async def test_offline(self):
        self.ops.set_available(False)
        from toolboxv2.mods.ContainerManager import docker_health
        data = (await docker_health(app=self.app, admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))).get()
        self.assertFalse(data["docker_available"])
        self.assertEqual(data["status"], "offline")

    async def test_bad_key(self):
        from toolboxv2.mods.ContainerManager import docker_health
        result = await docker_health(app=self.app, admin_key="wrong")
        self.assertTrue(result.is_error())


# ============================================================================
# list_all_docker_containers
# ============================================================================

class TestListAllDockerContainers(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.ops = FakeDockerOps()
        set_docker_ops(self.ops)
        self.app = FakeApp(FakeDB())

    async def asyncTearDown(self):
        set_docker_ops(FakeDockerOps())

    async def test_returns_tb_flag(self):
        self.ops.add_container("c1", "tb", "tbv2:latest",
                               labels={"managed-by": "ContainerManager"})
        self.ops.add_container("c2", "redis", "redis:7")
        from toolboxv2.mods.ContainerManager import list_all_docker_containers
        data = (await list_all_docker_containers(
            app=self.app, admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))).get()
        self.assertTrue(data["docker_available"])
        self.assertEqual(len(data["containers"]), 2)
        tb = next(c for c in data["containers"] if c["name"] == "tb")
        self.assertTrue(tb["is_tb_managed"])

    async def test_truncated_and_full_id(self):
        self.ops.add_container("abcdef1234567890full", "web", "nginx")
        from toolboxv2.mods.ContainerManager import list_all_docker_containers
        c = (await list_all_docker_containers(
            app=self.app, admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))).get()["containers"][0]
        self.assertEqual(c["container_id"], "abcdef123456")
        self.assertEqual(c["container_id_full"], "abcdef1234567890full")

    async def test_empty_when_offline(self):
        self.ops.set_available(False)
        from toolboxv2.mods.ContainerManager import list_all_docker_containers
        data = (await list_all_docker_containers(
            app=self.app, admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))).get()
        self.assertFalse(data["docker_available"])
        self.assertEqual(data["containers"], [])

    async def test_bad_key(self):
        from toolboxv2.mods.ContainerManager import list_all_docker_containers
        result = await list_all_docker_containers(app=self.app, admin_key="nope")
        self.assertTrue(result.is_error())


# ============================================================================
# update_container
# ============================================================================

class TestUpdateContainer(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.ops = FakeDockerOps()
        set_docker_ops(self.ops)
        self.db = FakeDB()
        self.app = FakeApp(self.db)

    async def asyncTearDown(self):
        set_docker_ops(FakeDockerOps())

    async def _seed(self, cid="test-abc123"):
        spec = make_container_spec(container_id=cid)
        seed_container_in_db(self.db, spec)
        self.ops.add_container(cid, spec["container_name"], spec["image"],
                               status="running",
                               labels={"managed-by": "ContainerManager"})
        return spec

    async def test_update_full_cycle(self):
        await self._seed()
        from toolboxv2.mods.ContainerManager import update_container
        result = await update_container(app=self.app, container_id="test-abc123",
                                        admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"), pull=True)
        self.assertTrue(result.is_ok(), msg=f"failed: {result.info}")
        data = result.get()
        self.assertEqual(data["old_container_id"], "test-abc123"[:12])
        self.assertNotEqual(data["old_container_id"], data["new_container_id"])
        self.assertEqual(self.ops.get_container_status("test-abc123"), "not_found")
        self.assertIn("toolboxv2:latest", self.ops._images)

    async def test_update_no_pull(self):
        await self._seed()
        from toolboxv2.mods.ContainerManager import update_container
        result = await update_container(app=self.app, container_id="test-abc123",
                                        admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"), pull=False)
        self.assertTrue(result.is_ok())
        self.assertNotIn("toolboxv2:latest", self.ops._images)

    async def test_update_bad_key(self):
        await self._seed()
        from toolboxv2.mods.ContainerManager import update_container
        result = await update_container(app=self.app, container_id="test-abc123",
                                        admin_key="wrong")
        self.assertTrue(result.is_error())

    async def test_update_missing_container(self):
        from toolboxv2.mods.ContainerManager import update_container
        result = await update_container(app=self.app, container_id="nope",
                                        admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))
        self.assertTrue(result.is_error())

    async def test_update_docker_offline(self):
        await self._seed()
        self.ops.set_available(False)
        from toolboxv2.mods.ContainerManager import update_container
        result = await update_container(app=self.app, container_id="test-abc123",
                                        admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))
        self.assertTrue(result.is_error())


# ============================================================================
# reconcile_status
# ============================================================================

class TestReconcileStatus(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.ops = FakeDockerOps()
        set_docker_ops(self.ops)
        self.db = FakeDB()
        self.app = FakeApp(self.db)

    async def asyncTearDown(self):
        set_docker_ops(FakeDockerOps())

    async def test_updates_db_on_change(self):
        spec = make_container_spec(container_id="c1", status="running")
        seed_container_in_db(self.db, spec)
        self.ops.add_container("c1", "web", "nginx", status="exited")
        from toolboxv2.mods.ContainerManager import reconcile_status
        result = await reconcile_status(app=self.app, container_id="c1",
                                        admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))
        self.assertEqual(result.get()["status"], "exited")
        stored = json.loads(self.db.get("CONTAINER::c1").get())
        self.assertEqual(stored["status"], "exited")

    async def test_no_change(self):
        spec = make_container_spec(container_id="c1", status="running")
        seed_container_in_db(self.db, spec)
        self.ops.add_container("c1", "web", "nginx", status="running")
        from toolboxv2.mods.ContainerManager import reconcile_status
        result = await reconcile_status(app=self.app, container_id="c1",
                                        admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))
        self.assertEqual(result.get()["status"], "running")

    async def test_not_found(self):
        spec = make_container_spec(container_id="c1", status="running")
        seed_container_in_db(self.db, spec)
        from toolboxv2.mods.ContainerManager import reconcile_status
        result = await reconcile_status(app=self.app, container_id="c1",
                                        admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))
        self.assertEqual(result.get()["status"], "not_found")

    async def test_docker_offline(self):
        spec = make_container_spec(container_id="c1", status="running")
        seed_container_in_db(self.db, spec)
        self.ops.set_available(False)
        from toolboxv2.mods.ContainerManager import reconcile_status
        result = await reconcile_status(app=self.app, container_id="c1",
                                        admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))
        self.assertEqual(result.get()["status"], "docker_offline")

    async def test_bad_key(self):
        from toolboxv2.mods.ContainerManager import reconcile_status
        result = await reconcile_status(app=self.app, container_id="c1",
                                        admin_key="nope")
        self.assertTrue(result.is_error())

    async def test_missing_id(self):
        from toolboxv2.mods.ContainerManager import reconcile_status
        result = await reconcile_status(app=self.app, container_id=None,
                                        admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))
        self.assertTrue(result.is_error())


# ============================================================================
# Dashboard HTML structure
# ============================================================================

class TestDashboardHTML(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        set_docker_ops(FakeDockerOps())
        from toolboxv2.mods.ContainerManager.web_ui import _DASHBOARD_HTML
        cls.html = _DASHBOARD_HTML

    @classmethod
    def tearDownClass(cls):
        set_docker_ops(FakeDockerOps())

    def test_doctype(self):
        self.assertTrue(self.html.strip().startswith("<!DOCTYPE html>"))

    def test_admin_key_password_field(self):
        self.assertIn('type="password"', self.html)
        self.assertIn('id="adminKey"', self.html)

    def test_all_tabs_present(self):
        for tab in ["dashboard", "topology", "create", "logs"]:
            self.assertIn(f'data-tab="{tab}"', self.html, msg=f"Missing tab: {tab}")

    def test_docker_health_banner_element(self):
        self.assertIn("dockerHealth", self.html)

    def test_js_docker_health_endpoint(self):
        self.assertIn("/cm/api/docker-health", self.html)

    def test_js_all_containers_endpoint(self):
        self.assertIn("/cm/api/all-containers", self.html)

    def test_js_reconcile_endpoint(self):
        self.assertIn("/reconcile", self.html)

    def test_js_topology_endpoint(self):
        self.assertIn("/cm/api/topology", self.html)

    def test_update_action_exists(self):
        self.assertIn("update", self.html.lower())

    def test_visibility_change_polling(self):
        self.assertIn("visibilitychange", self.html)

    def test_no_10s_full_refresh(self):
        n = self.html.replace(" ", "")
        self.assertNotIn("setInterval(refresh,10000)", n)

    def test_is_tb_managed_used(self):
        self.assertIn("is_tb_managed", self.html)

    def test_oklch_colors(self):
        self.assertIn("oklch", self.html)

    def test_ibm_plex_mono(self):
        self.assertIn("IBM Plex Mono", self.html)


# ============================================================================
# Topology data
# ============================================================================

class TestTopologyData(unittest.TestCase):

    def setUp(self):
        self.ops = FakeDockerOps()
        set_docker_ops(self.ops)

    def tearDown(self):
        set_docker_ops(FakeDockerOps())

    def test_topology_from_networks(self):
        from toolboxv2.mods.ContainerManager.docker_ops import NetworkInfo
        self.ops.add_container("c1", "web", "nginx", networks=["bridge", "app"])
        self.ops.add_container("c2", "db", "pg", networks=["app"])
        self.ops._networks = [
            NetworkInfo("n1", "bridge", "bridge", ["c1"]),
            NetworkInfo("n2", "app", "bridge", ["c1", "c2"]),
        ]
        nets = self.ops.list_networks()
        app_net = next(n for n in nets if n.name == "app")
        self.assertIn("c1", app_net.containers)
        self.assertIn("c2", app_net.containers)


if __name__ == "__main__":
    unittest.main()
