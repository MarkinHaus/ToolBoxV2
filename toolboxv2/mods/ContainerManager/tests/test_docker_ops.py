"""
Unit tests for docker_ops.py

Tests the DockerOps interface contract using FakeDockerOps.
No real Docker daemon needed.
"""

import unittest

from toolboxv2.mods.ContainerManager.docker_ops import (
    ContainerInfo, NetworkInfo, get_docker_ops, set_docker_ops,
)
from toolboxv2.mods.ContainerManager.tests.fakes import (
    FakeDockerOps, make_container_info,
)


class TestDockerOpsAvailability(unittest.TestCase):

    def setUp(self):
        self.ops = FakeDockerOps()

    def test_is_available_returns_true_by_default(self):
        self.assertTrue(self.ops.is_available())

    def test_is_available_returns_false_when_daemon_offline(self):
        self.ops.set_available(False)
        self.assertFalse(self.ops.is_available())


class TestDockerOpsListContainers(unittest.TestCase):

    def setUp(self):
        self.ops = FakeDockerOps()

    def test_list_all_containers_empty_when_none_exist(self):
        result = self.ops.list_all_containers()
        self.assertEqual(result, [])

    def test_list_all_containers_returns_seeded_containers(self):
        self.ops.add_container("c1", "web", "nginx:latest", status="running")
        self.ops.add_container("c2", "db", "postgres:16", status="exited")
        result = self.ops.list_all_containers()
        self.assertEqual(len(result), 2)

    def test_list_all_containers_excludes_stopped_when_requested(self):
        self.ops.add_container("c1", "web", "nginx:latest", status="running")
        self.ops.add_container("c2", "db", "postgres:16", status="exited")
        result = self.ops.list_all_containers(include_stopped=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "web")

    def test_list_all_containers_returns_empty_when_docker_offline(self):
        self.ops.add_container("c1", "web", "nginx:latest")
        self.ops.set_available(False)
        result = self.ops.list_all_containers()
        self.assertEqual(result, [])

    def test_list_all_containers_sets_is_tb_managed_from_labels(self):
        self.ops.add_container("c1", "tb-worker", "toolboxv2:latest",
                               labels={"managed-by": "ContainerManager"})
        self.ops.add_container("c2", "searxng", "searxng/searxng:latest",
                               labels={"com.docker.compose.service": "searxng"})
        result = self.ops.list_all_containers()
        tb = next(c for c in result if c.name == "tb-worker")
        ext = next(c for c in result if c.name == "searxng")
        self.assertTrue(tb.is_tb_managed)
        self.assertFalse(ext.is_tb_managed)

    def test_container_info_has_correct_fields(self):
        self.ops.add_container(
            "c1", "web", "nginx:latest", status="running",
            ports={"80/tcp": 8080}, networks=["bridge", "app-net"],
            labels={"env": "prod"},
        )
        result = self.ops.list_all_containers()
        c = result[0]
        self.assertIsInstance(c, ContainerInfo)
        self.assertEqual(c.container_id, "c1")
        self.assertEqual(c.name, "web")
        self.assertEqual(c.image, "nginx:latest")
        self.assertEqual(c.status, "running")
        self.assertEqual(c.ports, {"80/tcp": 8080})
        self.assertIn("bridge", c.networks)
        self.assertIn("app-net", c.networks)


class TestDockerOpsContainerStatus(unittest.TestCase):

    def setUp(self):
        self.ops = FakeDockerOps()

    def test_get_status_returns_running_for_running_container(self):
        self.ops.add_container("c1", "web", "nginx", status="running")
        self.assertEqual(self.ops.get_container_status("c1"), "running")

    def test_get_status_returns_exited_for_stopped_container(self):
        self.ops.add_container("c1", "web", "nginx", status="exited")
        self.assertEqual(self.ops.get_container_status("c1"), "exited")

    def test_get_status_returns_not_found_for_missing_container(self):
        self.assertEqual(self.ops.get_container_status("nonexistent"), "not_found")

    def test_get_status_returns_docker_offline_when_unavailable(self):
        self.ops.add_container("c1", "web", "nginx")
        self.ops.set_available(False)
        self.assertEqual(self.ops.get_container_status("c1"), "docker_offline")


class TestDockerOpsLifecycle(unittest.TestCase):

    def setUp(self):
        self.ops = FakeDockerOps()

    def test_create_container_returns_id(self):
        cid = self.ops.create_container(
            name="test", image="nginx:latest",
            labels={"managed-by": "ContainerManager"},
        )
        self.assertTrue(cid.startswith("fake_"))
        self.assertEqual(self.ops.get_container_status(cid), "running")

    def test_create_container_raises_when_docker_offline(self):
        self.ops.set_available(False)
        with self.assertRaises(RuntimeError):
            self.ops.create_container(name="test", image="nginx")

    def test_stop_changes_status_to_exited(self):
        self.ops.add_container("c1", "web", "nginx", status="running")
        self.assertTrue(self.ops.stop("c1"))
        self.assertEqual(self.ops.get_container_status("c1"), "exited")

    def test_start_changes_status_to_running(self):
        self.ops.add_container("c1", "web", "nginx", status="exited")
        self.assertTrue(self.ops.start("c1"))
        self.assertEqual(self.ops.get_container_status("c1"), "running")

    def test_restart_keeps_status_running(self):
        self.ops.add_container("c1", "web", "nginx", status="running")
        self.assertTrue(self.ops.restart("c1"))
        self.assertEqual(self.ops.get_container_status("c1"), "running")

    def test_remove_deletes_container(self):
        self.ops.add_container("c1", "web", "nginx", status="exited")
        self.assertTrue(self.ops.remove("c1"))
        self.assertEqual(self.ops.get_container_status("c1"), "not_found")

    def test_remove_without_force_fails_on_running(self):
        self.ops.add_container("c1", "web", "nginx", status="running")
        self.assertFalse(self.ops.remove("c1", force=False))
        self.assertEqual(self.ops.get_container_status("c1"), "running")

    def test_remove_with_force_succeeds_on_running(self):
        self.ops.add_container("c1", "web", "nginx", status="running")
        self.assertTrue(self.ops.remove("c1", force=True))
        self.assertEqual(self.ops.get_container_status("c1"), "not_found")

    def test_lifecycle_on_missing_container_returns_false(self):
        self.assertFalse(self.ops.start("missing"))
        self.assertFalse(self.ops.stop("missing"))
        self.assertFalse(self.ops.restart("missing"))
        self.assertFalse(self.ops.remove("missing"))

    def test_lifecycle_when_docker_offline_returns_false(self):
        self.ops.add_container("c1", "web", "nginx")
        self.ops.set_available(False)
        self.assertFalse(self.ops.start("c1"))
        self.assertFalse(self.ops.stop("c1"))
        self.assertFalse(self.ops.restart("c1"))
        self.assertFalse(self.ops.remove("c1"))


class TestDockerOpsVolumes(unittest.TestCase):

    def setUp(self):
        self.ops = FakeDockerOps()

    def test_create_container_registers_volume(self):
        self.ops.create_container(
            name="test", image="nginx",
            volumes={"my_vol": {"bind": "/data", "mode": "rw"}},
        )
        self.assertIn("my_vol", self.ops._volumes)

    def test_remove_volume_succeeds(self):
        self.ops._volumes.add("my_vol")
        self.assertTrue(self.ops.remove_volume("my_vol"))
        self.assertNotIn("my_vol", self.ops._volumes)

    def test_remove_volume_returns_false_for_unknown(self):
        self.assertFalse(self.ops.remove_volume("nonexistent"))


class TestDockerOpsExec(unittest.TestCase):

    def setUp(self):
        self.ops = FakeDockerOps()

    def test_exec_run_returns_default_success(self):
        self.ops.add_container("c1", "web", "nginx")
        code, output = self.ops.exec_run("c1", ["echo", "hello"])
        self.assertEqual(code, 0)

    def test_exec_run_returns_preprogrammed_result(self):
        self.ops.add_container("c1", "web", "nginx")
        self.ops.set_exec_result("c1", 1, "error: file not found")
        code, output = self.ops.exec_run("c1", ["cat", "/missing"])
        self.assertEqual(code, 1)
        self.assertIn("file not found", output)

    def test_exec_run_on_missing_container_returns_error(self):
        code, output = self.ops.exec_run("missing", ["ls"])
        self.assertEqual(code, -1)

    def test_exec_run_when_docker_offline_returns_error(self):
        self.ops.add_container("c1", "web", "nginx")
        self.ops.set_available(False)
        code, output = self.ops.exec_run("c1", ["ls"])
        self.assertEqual(code, -1)


class TestDockerOpsLogs(unittest.TestCase):

    def setUp(self):
        self.ops = FakeDockerOps()

    def test_logs_returns_output_for_existing_container(self):
        self.ops.add_container("c1", "web", "nginx")
        logs = self.ops.logs("c1")
        self.assertIn("c1", logs)

    def test_logs_returns_empty_for_missing_container(self):
        self.assertEqual(self.ops.logs("missing"), "")

    def test_logs_returns_empty_when_docker_offline(self):
        self.ops.add_container("c1", "web", "nginx")
        self.ops.set_available(False)
        self.assertEqual(self.ops.logs("c1"), "")


class TestDockerOpsNetworks(unittest.TestCase):

    def setUp(self):
        self.ops = FakeDockerOps()

    def test_get_container_networks_returns_network_list(self):
        self.ops.add_container("c1", "web", "nginx", networks=["bridge", "app"])
        nets = self.ops.get_container_networks("c1")
        self.assertEqual(nets, ["bridge", "app"])

    def test_get_container_networks_returns_empty_when_offline(self):
        self.ops.add_container("c1", "web", "nginx", networks=["bridge"])
        self.ops.set_available(False)
        self.assertEqual(self.ops.get_container_networks("c1"), [])


class TestDockerOpsImages(unittest.TestCase):

    def setUp(self):
        self.ops = FakeDockerOps()

    def test_pull_image_succeeds(self):
        self.assertTrue(self.ops.pull_image("toolboxv2:latest"))
        self.assertIn("toolboxv2:latest", self.ops._images)

    def test_pull_image_fails_when_offline(self):
        self.ops.set_available(False)
        self.assertFalse(self.ops.pull_image("toolboxv2:latest"))


class TestDockerOpsStats(unittest.TestCase):

    def setUp(self):
        self.ops = FakeDockerOps()

    def test_get_stats_returns_dict_for_existing_container(self):
        self.ops.add_container("c1", "web", "nginx")
        stats = self.ops.get_container_stats("c1")
        self.assertIsInstance(stats, dict)
        self.assertIn("cpu_percent", stats)
        self.assertIn("memory_mb", stats)

    def test_get_stats_returns_none_for_missing(self):
        self.assertIsNone(self.ops.get_container_stats("missing"))

    def test_get_stats_returns_none_when_offline(self):
        self.ops.add_container("c1", "web", "nginx")
        self.ops.set_available(False)
        self.assertIsNone(self.ops.get_container_stats("c1"))


class TestDockerOpsSingleton(unittest.TestCase):

    def test_set_and_get_docker_ops(self):
        fake = FakeDockerOps()
        set_docker_ops(fake)
        self.assertIs(get_docker_ops(), fake)

    def tearDown(self):
        # Reset singleton to a fresh Fake — never leave as None
        # (None would cause next get_docker_ops() to create a REAL DockerOps
        #  which tries docker.from_env() and hangs if Docker is offline)
        set_docker_ops(FakeDockerOps())


class TestDockerOpsServerIp(unittest.TestCase):

    def test_fake_returns_static_ip(self):
        self.assertEqual(FakeDockerOps.get_server_ip(), "10.0.0.1")


class TestDockerOpsStatsParsing(unittest.TestCase):

    def test_parse_stats_correct_cpu_delta(self):
        """Verify CPU calculation uses delta between current and previous sample."""
        from toolboxv2.mods.ContainerManager.docker_ops import DockerOps

        raw = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 200_000_000},
                "system_cpu_usage": 2_000_000_000,
                "online_cpus": 4,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 100_000_000},
                "system_cpu_usage": 1_000_000_000,
            },
            "memory_stats": {"usage": 256 * 1024 * 1024, "limit": 1024 * 1024 * 1024},
            "networks": {
                "eth0": {"rx_bytes": 5000, "tx_bytes": 3000},
            },
        }
        result = DockerOps._parse_stats(raw)

        # CPU delta: (200M-100M)/(2G-1G) * 4 * 100 = 40%
        self.assertAlmostEqual(result["cpu_percent"], 40.0, places=1)
        # Memory: 256MB / 1024MB = 25%
        self.assertAlmostEqual(result["memory_mb"], 256.0, places=0)
        self.assertAlmostEqual(result["memory_percent"], 25.0, places=1)
        # Network
        self.assertEqual(result["network_rx_bytes"], 5000)
        self.assertEqual(result["network_tx_bytes"], 3000)

    def test_parse_stats_zero_system_delta_returns_zero_cpu(self):
        from toolboxv2.mods.ContainerManager.docker_ops import DockerOps

        raw = {
            "cpu_stats": {"cpu_usage": {"total_usage": 100}, "system_cpu_usage": 500, "online_cpus": 1},
            "precpu_stats": {"cpu_usage": {"total_usage": 100}, "system_cpu_usage": 500},
            "memory_stats": {"usage": 0, "limit": 1},
            "networks": {},
        }
        result = DockerOps._parse_stats(raw)
        self.assertEqual(result["cpu_percent"], 0.0)


if __name__ == "__main__":
    unittest.main()
