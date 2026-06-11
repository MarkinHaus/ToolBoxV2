"""
test_sandbox_backend.py — Phase 6.

unittest only (no pytest). Live tests against a real AIO container are
gated behind TB_SANDBOX_LIVE=1; everything else runs offline.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from toolboxv2.mods.isaa.base.patch import sandbox_backend as sb


class TestRemoteKey(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = mock.patch.object(sb, "STATE_DIR", Path(self._tmp.name))
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_invalid_format_rejected(self):
        for bad in ("nodot", "a.b.c", "x.y", "", "ab cd.efgh"):
            with self.assertRaises(ValueError):
                sb.resolve_remote_key(bad)

    def test_unknown_conn_key_rejected(self):
        (Path(self._tmp.name) / "remotes.json").write_text("{}")
        with self.assertRaises(ValueError):
            sb.resolve_remote_key("k7f2x9aa.proj0001")

    def test_register_and_resolve(self):
        sb.register_remote("k7f2x9aa", "https://sbx.example.com:7053", token="tok")
        tgt = sb.resolve_remote_key("k7f2x9aa.proj0001")
        self.assertEqual(tgt.base_url, "https://sbx.example.com:7053")
        self.assertEqual(tgt.token, "tok")
        self.assertEqual(tgt.workdir, "/work/proj0001")


class TestDockerLifecycle(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = mock.patch.object(sb, "STATE_DIR", Path(self._tmp.name))
        self._gpatch = mock.patch.object(sb, "GLOBAL_DIR", Path(self._tmp.name) / "global")
        self._patch.start(); self._gpatch.start()

    def tearDown(self):
        self._patch.stop(); self._gpatch.stop()
        self._tmp.cleanup()

    def test_container_name_slug(self):
        lc = sb.DockerLifecycle("My Agent/01")
        self.assertEqual(lc.container, "tb-sbx-my-agent-01")

    def test_no_docker_returns_error_dict(self):
        lc = sb.DockerLifecycle("a1")
        with mock.patch.object(lc, "docker_available", return_value=False):
            r = lc.ensure()
        self.assertFalse(r["success"])
        self.assertIn("docker not available", r["stderr"])

    def test_reuse_running_container(self):
        lc = sb.DockerLifecycle("a2")
        lc._save({lc.container: {"port": 7053, "mounts": sb.configured_mounts()}})
        with mock.patch.object(lc, "docker_available", return_value=True), \
             mock.patch.object(lc, "_state", return_value="running"):
            r = lc.ensure()
        self.assertTrue(r["success"])
        self.assertEqual(r["port"], 7053)
        self.assertTrue(r["reused"])
        self.assertEqual(r["base_url"], "http://127.0.0.1:7053")

    def test_free_port_in_range(self):
        port = sb._free_port()
        self.assertGreaterEqual(port, 7000)
        self.assertLessEqual(port, 7099)


class _FakeBackend:
    """Protocol-shape stub for registry / schleuse logic tests."""
    def __init__(self):
        self.workdir = "/work/s1"
        self.events = []
        self.label = "fake"
    def _abs(self, p):
        return p if p.startswith("/") else f"{self.workdir}/{p}"
    def download(self, s, h):
        return {"success": True, "stdout": "", "stderr": "", "returncode": 0}
    def upload(self, s, h):
        return {"success": True, "stdout": "", "stderr": "", "returncode": 0}
    def exec(self, cmd, cwd=None, timeout=None):
        return {"success": True, "stdout": "FILE", "stderr": "", "returncode": 0}
    def health(self):
        return {"success": True, "stdout": "ok", "stderr": "", "returncode": 0}


class TestSchleusePolicy(unittest.TestCase):
    def _session(self, be):
        s = mock.Mock()
        s._sandbox_backend = be
        s.sandbox_allowed_dirs = None
        s.session_id = "s1"
        s.obs = None
        s.agent = None
        return s

    def test_export_blocked_outside_out(self):
        from toolboxv2.mods.isaa.base.patch.sandbox_tools import make_sandbox_export
        be = _FakeBackend()
        tool = make_sandbox_export(self._session(be))
        r = tool("src/secret.py", "/tmp/x.py")
        self.assertFalse(r["success"])
        self.assertIn("export blocked", r["stderr"])

    def test_export_allowed_from_out(self):
        from toolboxv2.mods.isaa.base.patch.sandbox_tools import make_sandbox_export
        be = _FakeBackend()
        tool = make_sandbox_export(self._session(be))
        r = tool("out/result.txt", "/tmp/result.txt")
        self.assertTrue(r["success"])

    def test_import_blocked_outside_allowed_dirs(self):
        from toolboxv2.mods.isaa.base.patch.sandbox_tools import make_sandbox_import
        be = _FakeBackend()
        s = self._session(be)
        s.sandbox_allowed_dirs = ["/srv/allowed"]
        tool = make_sandbox_import(s)
        r = tool("/etc/passwd")
        self.assertFalse(r["success"])
        self.assertIn("import blocked", r["stderr"])


class TestObsEmitNeverRaises(unittest.TestCase):
    def test_emit_swallows_exceptions(self):
        be = object.__new__(sb.AIOSandboxBackend)  # skip __init__ (no SDK call)
        be.workdir = "/work/x"
        be.label = "t"
        be._emit_fn = mock.Mock(side_effect=RuntimeError("obs down"))
        be._emit("sandbox.exec", {"command": "ls"})  # must not raise


@unittest.skipUnless(os.getenv("TB_SANDBOX_LIVE") == "1",
                     "live test — requires docker + TB_SANDBOX_LIVE=1")
class TestLiveRoundtrip(unittest.TestCase):
    """End-to-end against a real AIO container (one per agent)."""

    @classmethod
    def setUpClass(cls):
        reg = sb.SandboxRegistry.for_agent("unittest-agent")
        be = reg.backend_for_session("live-s1")
        if isinstance(be, dict):
            raise unittest.SkipTest(be["stderr"])
        cls.reg, cls.be = reg, be

    @classmethod
    def tearDownClass(cls):
        cls.reg.shutdown()

    def test_exec_echo(self):
        r = self.be.exec("echo hello")
        self.assertTrue(r["success"])
        self.assertIn("hello", r["stdout"])

    def test_write_read_roundtrip(self):
        self.assertTrue(self.be.write("t.txt", "abc\ndef")["success"])
        r = self.be.read("t.txt")
        self.assertIn("abc", r["stdout"])

    def test_schleuse_roundtrip(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("payload")
            host = f.name
        self.assertTrue(self.be.upload(host, "out/in.txt")["success"])
        dst = host + ".back"
        self.assertTrue(self.be.download("out/in.txt", dst)["success"])
        self.assertEqual(Path(dst).read_text(), "payload")

    def test_jupyter(self):
        r = self.be.code("print(21*2)")
        self.assertTrue(r["success"])
        self.assertIn("42", r["stdout"])

    def test_per_agent_container_shared_across_sessions(self):
        be2 = self.reg.backend_for_session("live-s2")
        self.assertEqual(be2.base_url, self.be.base_url)      # same container
        self.assertNotEqual(be2.workdir, self.be.workdir)     # different workdir


if __name__ == "__main__":
    unittest.main()


class TestMountsAndDirs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._p1 = mock.patch.object(sb, "STATE_DIR", Path(self._tmp.name))
        self._p2 = mock.patch.object(sb, "GLOBAL_DIR", Path(self._tmp.name) / "global")
        self._p1.start(); self._p2.start()

    def tearDown(self):
        self._p1.stop(); self._p2.stop(); self._tmp.cleanup()

    def test_global_mount_always_first(self):
        with mock.patch.dict(os.environ, {"TB_SANDBOX_MOUNTS": "/srv/obsidian:/work/vault"}):
            m = sb.configured_mounts()
        self.assertEqual(m[0][1], "/work/global")
        self.assertIn(("/srv/obsidian", "/work/vault"), m)

    def test_mount_change_triggers_recreate(self):
        lc = sb.DockerLifecycle("a3")
        lc._save({lc.container: {"port": 7053, "mounts": [["/old", "/work/old"]]}})
        calls = []
        def fake_docker(*args, **kw):
            calls.append(args)
            r = mock.Mock(); r.returncode = 1; r.stderr = "stop-here"; r.stdout = ""
            return r
        with mock.patch.object(lc, "docker_available", return_value=True), \
             mock.patch.object(lc, "_state", return_value="running"), \
             mock.patch.object(lc, "_image_present", return_value=True), \
             mock.patch.object(sb, "_docker", side_effect=fake_docker):
            lc.ensure()
        self.assertTrue(any(c[:2] == ("rm", "-f") for c in calls))  # recreate happened


class TestShellSession404(unittest.TestCase):
    def _backend(self):
        be = object.__new__(sb.AIOSandboxBackend)
        be.workdir = "/work/s1"; be.label = "t"; be._emit_fn = None
        be._shell_session = "tb-s1"; be._shell_ready = False
        be.client = mock.Mock()
        return be

    def test_session_created_before_first_exec(self):
        be = self._backend()
        r = mock.Mock(); r.data.output = "ok"; r.data.exit_code = 0
        be.client.shell.exec_command.return_value = r
        res = be.exec("echo ok")
        be.client.shell.create_session.assert_called_once_with(id="tb-s1", exec_dir="/work/s1")
        self.assertTrue(res["success"])

    def test_404_triggers_recreate_and_retry(self):
        be = self._backend()
        be._shell_ready = True  # session believed alive
        good = mock.Mock(); good.data.output = "ok"; good.data.exit_code = 0
        # order: failing exec -> bootstrap one-off exec -> retried exec
        be.client.shell.exec_command.side_effect = [RuntimeError("404 Session not found"), good, good]
        res = be.exec("echo ok")
        self.assertTrue(res["success"])
        be.client.shell.create_session.assert_called_once()
        self.assertEqual(be.client.shell.exec_command.call_count, 3)  # exec(404) + bootstrap + retry


class TestBootstrapAndPolicy(unittest.TestCase):
    def _backend(self):
        be = object.__new__(sb.AIOSandboxBackend)
        be.workdir = "/work/s1"; be.label = "t"; be._emit_fn = None
        be._shell_session = "tb-s1"; be._shell_ready = False
        be.client = mock.Mock()
        ok = mock.Mock(); ok.data.output = "ok"; ok.data.exit_code = 0
        be.client.shell.exec_command.return_value = ok
        return be

    def test_bootstrap_runs_before_create_session(self):
        be = self._backend()
        order = []
        be._bootstrap_workdir = lambda: order.append("bootstrap")
        be.client.shell.create_session.side_effect = lambda **kw: order.append("create")
        be._ensure_shell()
        self.assertEqual(order, ["bootstrap", "create"])

    def test_upload_falls_back_to_base64_write(self):
        be = self._backend()
        be._shell_ready = True
        be.client.file.upload_file.side_effect = RuntimeError("pydantic mess / PermissionError")
        be.client.file.write_file.return_value = mock.Mock()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"data"); host = f.name
        r = be.upload(host, "x.bin")
        self.assertTrue(r["success"])
        self.assertIn("b64 fallback", r["stdout"])
        kw = be.client.file.write_file.call_args.kwargs
        self.assertEqual(kw["encoding"], "base64")

    def test_policy_session_overrides_env(self):
        s = mock.Mock(spec=[])
        s.sandbox_policy = sb.SandboxPolicy(export_prefix="", allowed_import_dirs=["/srv"])
        pol = sb.get_policy(s)
        self.assertEqual(pol.export_prefix, "")
        self.assertEqual(pol.allowed_import_dirs, ["/srv"])

    def test_policy_legacy_allowed_dirs_mapped(self):
        s = mock.Mock(spec=["sandbox_allowed_dirs"])
        s.sandbox_allowed_dirs = ["/legacy"]
        pol = sb.get_policy(s)
        self.assertEqual(pol.allowed_import_dirs, ["/legacy"])
        self.assertEqual(pol.allowed_export_dirs, ["/legacy"])

    def test_export_prefix_empty_allows_whole_workdir(self):
        from toolboxv2.mods.isaa.base.patch.sandbox_tools import make_sandbox_export
        be = _FakeBackend()
        s = mock.Mock(spec=[]); s.session_id = "s1"
        s.sandbox_policy = sb.SandboxPolicy(export_prefix="")
        s._sandbox_backend = be
        r = make_sandbox_export(s)("src/anything.py", "/tmp/a.py")
        self.assertTrue(r["success"])

    def test_edit_passthrough(self):
        be = self._backend()
        d = mock.Mock(); d.output = "edited";
        rr = mock.Mock(); rr.data = d
        be.client.file.str_replace_editor.return_value = rr
        r = be.edit("str_replace", "a.py", old_str="x", new_str="y")
        self.assertTrue(r["success"])
        kw = be.client.file.str_replace_editor.call_args.kwargs
        self.assertEqual(kw["command"], "str_replace")
        self.assertEqual(kw["path"], "/work/s1/a.py")
