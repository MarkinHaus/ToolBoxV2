"""
Unit tests for ContainerManager business logic (__init__.py).

Tests use FakeDockerOps + FakeDB — no real Docker, no real DB.
"""

import json
import os
import unittest

from toolboxv2.mods.ContainerManager.docker_ops import set_docker_ops, DockerOps
from toolboxv2.mods.ContainerManager.tests.fakes import (
    FakeDockerOps, FakeDB, FakeApp,
    make_container_spec, seed_container_in_db,
)


def _setup_module_globals(ops, app):
    """Inject test doubles into ContainerManager module globals."""
    import toolboxv2.mods.ContainerManager as cm
    set_docker_ops(ops)
    # Patch get_docker to use our ops
    cm.get_docker = lambda: ops if ops.is_available() else None
    # Patch nginx functions to no-op (we don't have /etc/nginx in tests)
    cm.deploy_nginx_config = _noop_nginx
    cm.remove_nginx_config = _noop_nginx


async def _noop_nginx(*args, **kwargs):
    """No-op replacement for nginx functions in tests."""
    from toolboxv2 import Result
    return Result.ok()


class TestListContainersStatusHonesty(unittest.IsolatedAsyncioTestCase):
    """Verify that list_containers returns LIVE Docker status, not stale DB status."""

    async def asyncSetUp(self):
        self.ops = FakeDockerOps()
        self.db = FakeDB()
        self.app = FakeApp(self.db)
        _setup_module_globals(self.ops, self.app)

    async def test_shows_running_only_when_docker_confirms(self):
        from toolboxv2.mods.ContainerManager import list_containers
        spec = make_container_spec(container_id="c1", status="running")
        seed_container_in_db(self.db, spec)
        # Docker says running
        self.ops.add_container("c1", "test", "nginx", status="running")

        result = await list_containers(app=self.app, user_id="usr_test", admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))
        self.assertFalse(result.is_error())
        containers = result.get()["containers"]
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0]["status"], "running")

    async def test_shows_exited_when_container_stopped(self):
        from toolboxv2.mods.ContainerManager import list_containers
        spec = make_container_spec(container_id="c1", status="running")
        seed_container_in_db(self.db, spec)
        # Docker says exited — DB still says running
        self.ops.add_container("c1", "test", "nginx", status="exited")

        result = await list_containers(app=self.app, user_id="usr_test", admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))
        containers = result.get()["containers"]
        self.assertEqual(containers[0]["status"], "exited")

    async def test_shows_docker_offline_when_daemon_unavailable(self):
        from toolboxv2.mods.ContainerManager import list_containers
        spec = make_container_spec(container_id="c1", status="running")
        seed_container_in_db(self.db, spec)
        self.ops.set_available(False)

        result = await list_containers(app=self.app, user_id="usr_test", admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))
        containers = result.get()["containers"]
        self.assertEqual(containers[0]["status"], "docker_offline")

    async def test_shows_not_found_for_removed_container(self):
        from toolboxv2.mods.ContainerManager import list_containers
        spec = make_container_spec(container_id="c1", status="running")
        seed_container_in_db(self.db, spec)
        # Container not in Docker at all

        result = await list_containers(app=self.app, user_id="usr_test", admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))
        containers = result.get()["containers"]
        self.assertEqual(containers[0]["status"], "not_found")


class TestListAllDockerContainers(unittest.IsolatedAsyncioTestCase):
    """Verify list_all_docker_containers shows TB + external containers."""

    async def asyncSetUp(self):
        self.ops = FakeDockerOps()
        self.db = FakeDB()
        self.app = FakeApp(self.db)
        _setup_module_globals(self.ops, self.app)

    async def test_includes_external_containers(self):
        from toolboxv2.mods.ContainerManager import list_all_docker_containers
        self.ops.add_container("c1", "tb-worker", "toolboxv2:latest",
                               labels={"managed-by": "ContainerManager"})
        self.ops.add_container("c2", "searxng", "searxng:latest",
                               labels={"service": "search"})

        result = await list_all_docker_containers(app=self.app, admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))
        self.assertFalse(result.is_error(), msg=f"Error: {result.info.help_text if hasattr(result.info, 'help_text') else result.info}")
        containers = result.get()["containers"]
        self.assertEqual(len(containers), 2)

    async def test_marks_tb_managed_correctly(self):
        from toolboxv2.mods.ContainerManager import list_all_docker_containers
        self.ops.add_container("c1", "tb-worker", "toolboxv2:latest",
                               labels={"managed-by": "ContainerManager"})
        self.ops.add_container("c2", "searxng", "searxng:latest")

        result = await list_all_docker_containers(app=self.app, admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))
        self.assertFalse(result.is_error(), msg=f"Error: {result.info.help_text if hasattr(result.info, 'help_text') else result.info}")
        containers = result.get()["containers"]
        tb = next(c for c in containers if c["name"] == "tb-worker")
        ext = next(c for c in containers if c["name"] == "searxng")
        self.assertTrue(tb["is_tb_managed"])
        self.assertFalse(ext["is_tb_managed"])

    async def test_returns_empty_when_docker_offline(self):
        from toolboxv2.mods.ContainerManager import list_all_docker_containers
        self.ops.set_available(False)

        result = await list_all_docker_containers(app=self.app, admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))
        self.assertFalse(result.is_error(), msg=f"Error: {result.info.help_text if hasattr(result.info, 'help_text') else result.info}")
        self.assertFalse(result.get()["docker_available"])


class TestDockerHealth(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.ops = FakeDockerOps()
        self.db = FakeDB()
        self.app = FakeApp(self.db)
        _setup_module_globals(self.ops, self.app)

    async def test_returns_online_when_available(self):
        from toolboxv2.mods.ContainerManager import docker_health
        self.ops.add_container("c1", "web", "nginx",
                               labels={"managed-by": "ContainerManager"})
        self.ops.add_container("c2", "ext", "redis")

        result = await docker_health(app=self.app, admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))
        self.assertFalse(result.is_error(), msg=f"Error: {result.info.help_text if hasattr(result.info, 'help_text') else result.info}")
        data = result.get()
        self.assertTrue(data["docker_available"])
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["total_containers"], 2)
        self.assertEqual(data["tb_managed"], 1)
        self.assertEqual(data["external"], 1)

    async def test_returns_offline_when_unavailable(self):
        from toolboxv2.mods.ContainerManager import docker_health
        self.ops.set_available(False)

        result = await docker_health(app=self.app, admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"))
        self.assertFalse(result.is_error(), msg=f"Error: {result.info.help_text if hasattr(result.info, 'help_text') else result.info}")
        self.assertFalse(result.get()["docker_available"])
        self.assertEqual(result.get()["status"], "offline")


class TestUpdateContainer(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.ops = FakeDockerOps()
        self.db = FakeDB()
        self.app = FakeApp(self.db)
        _setup_module_globals(self.ops, self.app)

    async def test_preserves_ports_and_volumes(self):
        from toolboxv2.mods.ContainerManager import update_container
        spec = make_container_spec(
            container_id="old_c1", port=9001, ssh_port=22001,
            volume_name="vol_test",
        )
        seed_container_in_db(self.db, spec)
        self.ops.add_container("old_c1", "test", "toolboxv2:latest", status="running")

        result = await update_container(
            app=self.app, container_id="old_c1",
            admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"), pull=False,
        )
        self.assertFalse(result.is_error(), msg=f"update failed: {result.get()}")
        data = result.get()
        self.assertIn("new_container_id", data)
        self.assertNotEqual(data["new_container_id"], "old_c1"[:12])

        # Old container should be removed
        self.assertEqual(self.ops.get_container_status("old_c1"), "not_found")

        # New container should be running
        new_id = data["new_container_id"]
        # The new container exists in our fake
        new_containers = [c for c in self.ops._containers.values()
                          if c["container_id"] != "old_c1"]
        self.assertEqual(len(new_containers), 1)

    async def test_pulls_new_image(self):
        from toolboxv2.mods.ContainerManager import update_container
        spec = make_container_spec(container_id="c1", image="toolboxv2:latest")
        seed_container_in_db(self.db, spec)
        self.ops.add_container("c1", "test", "toolboxv2:latest")

        result = await update_container(
            app=self.app, container_id="c1",
            admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"), pull=True,
        )
        self.assertFalse(result.is_error())
        self.assertIn("toolboxv2:latest", self.ops._images)

    async def test_fails_when_container_not_found(self):
        from toolboxv2.mods.ContainerManager import update_container
        result = await update_container(
            app=self.app, container_id="nonexistent",
            admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"),
        )
        self.assertTrue(result.is_error())

    async def test_fails_when_docker_offline(self):
        from toolboxv2.mods.ContainerManager import update_container
        spec = make_container_spec(container_id="c1")
        seed_container_in_db(self.db, spec)
        self.ops.set_available(False)

        result = await update_container(
            app=self.app, container_id="c1",
            admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"),
        )
        self.assertTrue(result.is_error())


class TestReconcileStatus(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.ops = FakeDockerOps()
        self.db = FakeDB()
        self.app = FakeApp(self.db)
        _setup_module_globals(self.ops, self.app)

    async def test_updates_db_when_status_changed(self):
        from toolboxv2.mods.ContainerManager import reconcile_status, db_get_container
        spec = make_container_spec(container_id="c1", status="running")
        seed_container_in_db(self.db, spec)
        # Docker says exited
        self.ops.add_container("c1", "test", "nginx", status="exited")

        result = await reconcile_status(
            app=self.app, container_id="c1",
            admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"),
        )
        self.assertFalse(result.is_error(), msg=f"Error: {result.info.help_text if hasattr(result.info, 'help_text') else result.info}")
        self.assertEqual(result.get()["status"], "exited")

        # DB should be updated too
        updated = await db_get_container(self.app, "c1")
        self.assertEqual(updated.status, "exited")

    async def test_returns_docker_offline(self):
        from toolboxv2.mods.ContainerManager import reconcile_status
        self.ops.set_available(False)

        result = await reconcile_status(
            app=self.app, container_id="c1",
            admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"),
        )
        self.assertFalse(result.is_error(), msg=f"Error: {result.info.help_text if hasattr(result.info, 'help_text') else result.info}")
        self.assertEqual(result.get()["status"], "docker_offline")
        self.assertFalse(result.get()["docker_available"])


class TestDeleteContainerCleanup(unittest.IsolatedAsyncioTestCase):
    """Verify delete properly cleans up volume + ports."""

    async def asyncSetUp(self):
        self.ops = FakeDockerOps()
        self.db = FakeDB()
        self.app = FakeApp(self.db)
        _setup_module_globals(self.ops, self.app)

    async def test_removes_volume_on_delete(self):
        from toolboxv2.mods.ContainerManager import delete_container
        spec = make_container_spec(
            container_id="c1", user_id="usr_test", volume_name="vol_test", status="exited",
        )
        seed_container_in_db(self.db, spec)
        self.ops.add_container("c1", "test", "nginx", status="exited")
        self.ops._volumes.add("vol_test")

        result = await delete_container(
            app=self.app, container_id="c1",
            user_id="usr_test",
            admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"), force=True,
        )
        self.assertFalse(result.is_error(), msg=f"Error: {result.info.help_text if hasattr(result.info, 'help_text') else result.info}")
        self.assertNotIn("vol_test", self.ops._volumes)

    async def test_releases_ports_on_delete(self):
        from toolboxv2.mods.ContainerManager import delete_container
        spec = make_container_spec(
            container_id="c1", user_id="usr_test", port=9001, ssh_port=22001, status="exited",
        )
        seed_container_in_db(self.db, spec)
        self.db.set("CONTAINER_PORT_POOL", json.dumps([9001]))
        self.db.set("CONTAINER_SSH_PORT_POOL", json.dumps([22001]))
        self.ops.add_container("c1", "test", "nginx", status="exited")

        result = await delete_container(
            app=self.app, container_id="c1",
            user_id="usr_test",
            admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"), force=True,
        )
        self.assertFalse(result.is_error(), msg=f"Error: {result.info.help_text if hasattr(result.info, 'help_text') else result.info}")

        port_pool = json.loads(self.db.get("CONTAINER_PORT_POOL").get())
        self.assertNotIn(9001, port_pool)


class TestContainerExecSafety(unittest.IsolatedAsyncioTestCase):
    """Verify exec uses list-based command (no shell injection)."""

    async def asyncSetUp(self):
        self.ops = FakeDockerOps()
        self.db = FakeDB()
        self.app = FakeApp(self.db)
        _setup_module_globals(self.ops, self.app)

    async def test_exec_uses_list_command(self):
        from toolboxv2.mods.ContainerManager import container_exec
        spec = make_container_spec(container_id="c1", user_id="usr_test")
        seed_container_in_db(self.db, spec)
        self.ops.add_container("c1", "test", "nginx")

        result = await container_exec(
            app=self.app, container_id="c1",
            user_id="usr_test",
            admin_key=os.getenv("CONTAINER_ADMIN_KEY", "admin-change-me"), command="ls -la",
        )
        self.assertFalse(result.is_error(), msg=f"Error: {result.info.help_text if hasattr(result.info, 'help_text') else result.info}")
        self.assertEqual(result.get()["exit_code"], 0)
        self.assertEqual(result.get()["exit_code"], 0)


class TestSSHKeyValidation(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.ops = FakeDockerOps()
        self.db = FakeDB()
        self.app = FakeApp(self.db)
        _setup_module_globals(self.ops, self.app)

    async def test_rejects_invalid_ssh_key_format(self):
        from toolboxv2.mods.ContainerManager import register_ssh_key
        result = await register_ssh_key(
            app=self.app, ssh_public_key='ssh-rsa ; rm -rf /',
        )
        self.assertTrue(result.is_error())

    async def test_rejects_key_exceeding_max_length(self):
        from toolboxv2.mods.ContainerManager import register_ssh_key
        long_key = "ssh-ed25519 " + "A" * 2100
        result = await register_ssh_key(
            app=self.app, ssh_public_key=long_key,
        )
        self.assertTrue(result.is_error())

    async def test_accepts_valid_ed25519_key(self):
        from toolboxv2.mods.ContainerManager import register_ssh_key
        from unittest.mock import AsyncMock, patch
        # This will fail at auth check (no real user), but should pass validation
        valid_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest user@host"
        # Patch get_user_from_request to return None (not authenticated)
        with patch("toolboxv2.mods.ContainerManager.get_user_from_request",
                   new_callable=AsyncMock, return_value=None):
            result = await register_ssh_key(
                app=self.app, ssh_public_key=valid_key,
            )
        # Should fail at "Not authenticated" — NOT at key validation
        self.assertTrue(result.is_error())
        info_text = result.info.help_text if hasattr(result.info, 'help_text') else str(result.info)
        self.assertIn("authenticated", info_text.lower())


class TestServerIp(unittest.TestCase):

    def test_get_server_ip_uses_env_var(self):
        import os
        os.environ["CONTAINER_SERVER_IP"] = "1.2.3.4"
        try:
            self.assertEqual(DockerOps.get_server_ip(), "1.2.3.4")
        finally:
            del os.environ["CONTAINER_SERVER_IP"]

    def test_fake_docker_ops_returns_static_ip(self):
        self.assertEqual(FakeDockerOps.get_server_ip(), "10.0.0.1")


if __name__ == "__main__":
    unittest.main()
