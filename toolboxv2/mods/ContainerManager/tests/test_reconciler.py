"""
Unit tests for reconciler.py
"""

import unittest

from toolboxv2.mods.ContainerManager.reconciler import ContainerReconciler
from toolboxv2.mods.ContainerManager.tests.fakes import FakeDockerOps


class TestReconcilerRoundRobin(unittest.TestCase):

    def setUp(self):
        self.ops = FakeDockerOps()
        self.ops.add_container("c1", "web1", "nginx", status="running")
        self.ops.add_container("c2", "web2", "nginx", status="exited")
        self.ops.add_container("c3", "web3", "nginx", status="running")
        self.reconciler = ContainerReconciler(docker_ops=self.ops)
        self.reconciler.set_container_ids(["c1", "c2", "c3"])

    def test_reconcile_next_returns_first_container(self):
        result = self.reconciler.reconcile_next_sync()
        self.assertIsNotNone(result)
        self.assertEqual(result["container_id"], "c1")
        self.assertEqual(result["new_status"], "running")

    def test_reconcile_next_advances_round_robin(self):
        self.reconciler.reconcile_next_sync()  # c1
        result = self.reconciler.reconcile_next_sync()  # c2
        self.assertEqual(result["container_id"], "c2")
        self.assertEqual(result["new_status"], "exited")

    def test_reconcile_next_wraps_around(self):
        self.reconciler.reconcile_next_sync()  # c1
        self.reconciler.reconcile_next_sync()  # c2
        self.reconciler.reconcile_next_sync()  # c3
        result = self.reconciler.reconcile_next_sync()  # wraps → c1
        self.assertEqual(result["container_id"], "c1")

    def test_reconcile_next_returns_none_when_no_containers(self):
        empty = ContainerReconciler(docker_ops=self.ops)
        self.assertIsNone(empty.reconcile_next_sync())


class TestReconcilerStatusDetection(unittest.TestCase):

    def setUp(self):
        self.ops = FakeDockerOps()
        self.reconciler = ContainerReconciler(docker_ops=self.ops)

    def test_detects_running_status(self):
        self.ops.add_container("c1", "web", "nginx", status="running")
        self.reconciler.set_container_ids(["c1"])
        result = self.reconciler.reconcile_next_sync()
        self.assertEqual(result["new_status"], "running")

    def test_detects_stopped_status(self):
        self.ops.add_container("c1", "web", "nginx", status="exited")
        self.reconciler.set_container_ids(["c1"])
        result = self.reconciler.reconcile_next_sync()
        self.assertEqual(result["new_status"], "exited")

    def test_detects_removed_container(self):
        # Container in reconciler list but not in Docker
        self.reconciler.set_container_ids(["ghost"])
        result = self.reconciler.reconcile_next_sync()
        self.assertEqual(result["new_status"], "not_found")

    def test_detects_docker_offline(self):
        self.ops.add_container("c1", "web", "nginx", status="running")
        self.reconciler.set_container_ids(["c1"])
        self.ops.set_available(False)
        result = self.reconciler.reconcile_next_sync()
        self.assertEqual(result["new_status"], "docker_offline")

    def test_status_changes_when_container_stops(self):
        self.ops.add_container("c1", "web", "nginx", status="running")
        self.reconciler.set_container_ids(["c1"])

        result1 = self.reconciler.reconcile_next_sync()
        self.assertEqual(result1["new_status"], "running")

        # Container stops between polls
        self.ops._containers["c1"]["status"] = "exited"
        self.reconciler._current_index = 0  # Reset for re-check

        result2 = self.reconciler.reconcile_next_sync()
        self.assertEqual(result2["new_status"], "exited")


class TestReconcileAll(unittest.TestCase):

    def setUp(self):
        self.ops = FakeDockerOps()
        self.ops.add_container("c1", "web1", "nginx", status="running")
        self.ops.add_container("c2", "web2", "nginx", status="exited")
        self.reconciler = ContainerReconciler(docker_ops=self.ops)
        self.reconciler.set_container_ids(["c1", "c2"])

    def test_reconcile_all_returns_all_statuses(self):
        results = self.reconciler.reconcile_all_sync()
        self.assertEqual(len(results), 2)
        statuses = {r["container_id"]: r["new_status"] for r in results}
        self.assertEqual(statuses["c1"], "running")
        self.assertEqual(statuses["c2"], "exited")

    def test_reconcile_all_empty_list(self):
        empty = ContainerReconciler(docker_ops=self.ops)
        self.assertEqual(empty.reconcile_all_sync(), [])


class TestSetContainerIds(unittest.TestCase):

    def setUp(self):
        self.ops = FakeDockerOps()
        self.ops.add_container("c1", "web1", "nginx")
        self.ops.add_container("c2", "web2", "nginx")
        self.ops.add_container("c3", "web3", "nginx")
        self.reconciler = ContainerReconciler(docker_ops=self.ops)

    def test_set_container_ids_resets_index(self):
        self.reconciler.set_container_ids(["c1", "c2"])
        self.reconciler.reconcile_next_sync()  # c1
        self.reconciler.reconcile_next_sync()  # c2

        # Reset with new list
        self.reconciler.set_container_ids(["c3", "c1"])
        result = self.reconciler.reconcile_next_sync()
        self.assertEqual(result["container_id"], "c3")

    def test_container_count_reflects_list_size(self):
        self.reconciler.set_container_ids(["c1", "c2", "c3"])
        self.assertEqual(self.reconciler.container_count, 3)

        self.reconciler.set_container_ids(["c1"])
        self.assertEqual(self.reconciler.container_count, 1)


class TestDockerHealth(unittest.TestCase):

    def setUp(self):
        self.ops = FakeDockerOps()
        self.reconciler = ContainerReconciler(docker_ops=self.ops)

    def test_docker_health_online(self):
        self.reconciler.set_container_ids(["c1", "c2"])
        health = self.reconciler.docker_health()
        self.assertTrue(health["docker_available"])
        self.assertEqual(health["status"], "online")
        self.assertEqual(health["container_count"], 2)

    def test_docker_health_offline(self):
        self.ops.set_available(False)
        health = self.reconciler.docker_health()
        self.assertFalse(health["docker_available"])
        self.assertEqual(health["status"], "offline")


if __name__ == "__main__":
    unittest.main()
