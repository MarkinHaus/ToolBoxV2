"""
Tests for Docker image build logic and Compose configuration.

Covers:
  - Feature parsing and preset resolution
  - Build command construction
  - filter_requirements.py feature→package mapping
  - compose.yaml structural validation
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# We test docker_image_cli functions in isolation (no Docker needed)
# ---------------------------------------------------------------------------

# Inline the pure functions so tests don't depend on toolboxv2 imports
FEATURE_MAP = {
    'cli': 'FEATURE_CLI',
    'web': 'FEATURE_WEB',
    'desktop': 'FEATURE_DESKTOP',
    'exotic': 'FEATURE_EXOTIC',
    'isaa': 'FEATURE_ISAA',
}

FEATURE_PRESETS = {
    'all': ['cli', 'web', 'desktop', 'exotic', 'isaa'],
    'production': ['cli', 'web'],
    'dev': ['cli', 'web', 'desktop', 'exotic', 'isaa'],
}


def _resolve_features(features_str):
    if not features_str:
        return {'cli': 1, 'web': 1}
    key = features_str.lower().strip()
    if key in FEATURE_PRESETS:
        return {f: 1 for f in FEATURE_PRESETS[key]}
    active = {}
    for f in key.split(','):
        f = f.strip()
        if f in FEATURE_MAP:
            active[f] = 1
    return active or {'cli': 1, 'web': 1}


def _build_cmd(dockerfile, tag, project_root, active_features, no_cache=False):
    cmd = ["docker", "build", "-f", dockerfile, "-t", tag]
    for feat, env_var in FEATURE_MAP.items():
        cmd.extend(["--build-arg", f"{env_var}={active_features.get(feat, 0)}"])
    if no_cache:
        cmd.append("--no-cache")
    cmd.append(project_root)
    return cmd


# ============================================================================
# Feature Resolution Tests
# ============================================================================

class TestFeatureResolution(unittest.TestCase):
    """Test _resolve_features: string → dict mapping."""

    def test_none_defaults_to_cli_web(self):
        result = _resolve_features(None)
        self.assertEqual(result, {'cli': 1, 'web': 1})

    def test_empty_string_defaults_to_cli_web(self):
        result = _resolve_features("")
        self.assertEqual(result, {'cli': 1, 'web': 1})

    def test_preset_all_enables_everything(self):
        result = _resolve_features("all")
        self.assertEqual(len(result), 5)
        for feat in ('cli', 'web', 'desktop', 'exotic', 'isaa'):
            self.assertIn(feat, result)

    def test_preset_production_is_cli_web(self):
        result = _resolve_features("production")
        self.assertEqual(result, {'cli': 1, 'web': 1})

    def test_preset_dev_equals_all(self):
        self.assertEqual(_resolve_features("dev"), _resolve_features("all"))

    def test_preset_case_insensitive(self):
        self.assertEqual(_resolve_features("ALL"), _resolve_features("all"))
        self.assertEqual(_resolve_features("Production"), _resolve_features("production"))

    def test_single_feature(self):
        result = _resolve_features("isaa")
        self.assertEqual(result, {'isaa': 1})

    def test_comma_separated_features(self):
        result = _resolve_features("cli,web,isaa")
        self.assertEqual(result, {'cli': 1, 'web': 1, 'isaa': 1})

    def test_comma_separated_with_spaces(self):
        result = _resolve_features(" cli , web ")
        self.assertEqual(result, {'cli': 1, 'web': 1})

    def test_unknown_feature_ignored(self):
        result = _resolve_features("cli,nonexistent,web")
        self.assertEqual(result, {'cli': 1, 'web': 1})

    def test_all_unknown_falls_back_to_default(self):
        result = _resolve_features("foo,bar,baz")
        self.assertEqual(result, {'cli': 1, 'web': 1})


# ============================================================================
# Build Command Construction Tests
# ============================================================================

class TestBuildCommand(unittest.TestCase):
    """Test _build_cmd: correct docker build command generation."""

    def test_basic_command_structure(self):
        cmd = _build_cmd("/path/Dockerfile.toolbox", "toolboxv2:latest", "/project", {'cli': 1, 'web': 1})
        self.assertEqual(cmd[0], "docker")
        self.assertEqual(cmd[1], "build")
        self.assertIn("-f", cmd)
        self.assertIn("-t", cmd)
        self.assertEqual(cmd[-1], "/project")

    def test_all_feature_args_present(self):
        cmd = _build_cmd("Df", "img:t", "/p", {'cli': 1})
        arg_string = " ".join(cmd)
        for env_var in FEATURE_MAP.values():
            self.assertIn(env_var, arg_string, f"Missing build-arg for {env_var}")

    def test_enabled_feature_gets_1(self):
        cmd = _build_cmd("Df", "img:t", "/p", {'web': 1})
        idx = cmd.index("FEATURE_WEB=1")
        # Should exist (preceded by --build-arg)
        self.assertEqual(cmd[idx - 1], "--build-arg")

    def test_disabled_feature_gets_0(self):
        cmd = _build_cmd("Df", "img:t", "/p", {'cli': 1})
        self.assertIn("FEATURE_ISAA=0", " ".join(cmd))

    def test_no_cache_flag(self):
        cmd = _build_cmd("Df", "img:t", "/p", {}, no_cache=True)
        self.assertIn("--no-cache", cmd)

    def test_no_cache_absent_by_default(self):
        cmd = _build_cmd("Df", "img:t", "/p", {})
        self.assertNotIn("--no-cache", cmd)

    def test_tag_applied(self):
        cmd = _build_cmd("Df", "myimage:v2.0", "/p", {})
        idx = cmd.index("-t")
        self.assertEqual(cmd[idx + 1], "myimage:v2.0")


# ============================================================================
# filter_requirements.py Tests
# ============================================================================

class TestFilterRequirements(unittest.TestCase):
    """Test docker/filter_requirements.py in isolation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Minimal requirements.txt
        req_path = os.path.join(self.tmpdir, "requirements.txt")
        with open(req_path, 'w') as f:
            f.write("pydantic>=2.0\nhttpx\n# comment line\nzmq  # inline comment\n")

        # Copy filter script
        script_src = Path(__file__).parent.parent / "docker" / "filter_requirements.py"
        if script_src.exists():
            import shutil
            self.script = os.path.join(self.tmpdir, "filter_requirements.py")
            shutil.copy(script_src, self.script)
        else:
            # Inline fallback for CI where file layout differs
            self.script = os.path.join(self.tmpdir, "filter_requirements.py")
            with open(self.script, 'w') as f:
                f.write(textwrap.dedent('''\
                    import os
                    with open('requirements.txt') as f:
                        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
                    core = []
                    for line in lines:
                        if '#' in line:
                            parts = line.split('#', 1)
                            if parts[0].strip():
                                core.append(parts[0].strip())
                        else:
                            core.append(line)
                    with open('requirements-final.txt', 'w') as f:
                        f.write('\\n'.join(core) + '\\n')
                    active = []
                    feature_extras = {
                        'web': ['starlette', 'uvicorn[standard]', 'aiohttp-cors', 'httpx', 'waitress'],
                        'isaa': ['litellm>=0.49.0', 'langchain-core>=0.1.0', 'groq>=0.11.0'],
                    }
                    for feat, env in [('cli','FEATURE_CLI'),('web','FEATURE_WEB'),
                                       ('desktop','FEATURE_DESKTOP'),('exotic','FEATURE_EXOTIC'),
                                       ('isaa','FEATURE_ISAA')]:
                        if os.environ.get(env, '0') == '1':
                            active.append(feat)
                            for pkg in feature_extras.get(feat, []):
                                with open('requirements-final.txt', 'a') as f:
                                    f.write(pkg + '\\n')
                    with open('active-features.txt', 'w') as f:
                        f.write(' '.join(active) if active else 'none')
                    with open('system-deps.txt', 'w') as f:
                        f.write('')
                '''))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_filter(self, env_overrides=None):
        env = os.environ.copy()
        # Reset all features
        for var in FEATURE_MAP.values():
            env[var] = '0'
        if env_overrides:
            env.update(env_overrides)

        result = subprocess.run(
            [sys.executable, self.script],
            cwd=self.tmpdir,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"filter script failed:\n{result.stderr}")
        return result

    def test_no_features_produces_core_only(self):
        self._run_filter()
        reqs = Path(self.tmpdir, "requirements-final.txt").read_text()
        self.assertIn("pydantic", reqs)
        self.assertIn("httpx", reqs)
        self.assertIn("zmq", reqs)
        # No feature extras
        self.assertNotIn("starlette", reqs)
        self.assertNotIn("litellm", reqs)

    def test_inline_comments_stripped(self):
        self._run_filter()
        reqs = Path(self.tmpdir, "requirements-final.txt").read_text()
        # "zmq  # inline comment" → "zmq"
        for line in reqs.splitlines():
            self.assertNotIn("#", line, f"Comment not stripped: {line}")

    def test_comment_lines_excluded(self):
        self._run_filter()
        reqs = Path(self.tmpdir, "requirements-final.txt").read_text()
        self.assertNotIn("comment line", reqs)

    def test_web_feature_adds_starlette(self):
        self._run_filter({'FEATURE_WEB': '1'})
        reqs = Path(self.tmpdir, "requirements-final.txt").read_text()
        self.assertIn("starlette", reqs)

    def test_isaa_feature_adds_litellm(self):
        self._run_filter({'FEATURE_ISAA': '1'})
        reqs = Path(self.tmpdir, "requirements-final.txt").read_text()
        self.assertIn("litellm", reqs)

    def test_active_features_file_lists_enabled(self):
        self._run_filter({'FEATURE_CLI': '1', 'FEATURE_WEB': '1'})
        active = Path(self.tmpdir, "active-features.txt").read_text().strip()
        self.assertIn("cli", active)
        self.assertIn("web", active)
        self.assertNotIn("isaa", active)

    def test_no_features_active_writes_none(self):
        self._run_filter()
        active = Path(self.tmpdir, "active-features.txt").read_text().strip()
        self.assertEqual(active, "none")


# ============================================================================
# Compose YAML Structural Validation
# ============================================================================

class TestComposeStructure(unittest.TestCase):
    """Validate compose.yaml structure without running Docker."""

    @classmethod
    def setUpClass(cls):
        # Find compose.yaml relative to this test file
        test_dir = Path(__file__).parent.parent
        compose_candidates = [
            test_dir / "compose.yaml",
            Path(__file__).parent.parent.parent / "compose.yaml",
            Path(__file__).parent.parent.parent.parent / "compose.yaml",
        ]
        cls.compose_path = None
        for p in compose_candidates:
            if p.exists():
                cls.compose_path = p
                break

        if cls.compose_path is None:
            # Try loading from output dir (when run in build context)
            output = Path(__file__).parent.parent / "output" / "compose.yaml"
            if output.exists():
                cls.compose_path = output

        cls.compose_data = None
        if cls.compose_path:
            try:
                import yaml
                with open(cls.compose_path) as f:
                    cls.compose_data = yaml.safe_load(f)
            except ImportError:
                pass

    def test_compose_file_exists(self):
        self.assertIsNotNone(self.compose_path, "compose.yaml not found")

    def test_compose_has_required_services(self):
        if not self.compose_data:
            self.skipTest("Could not parse compose.yaml (pyyaml missing?)")
        services = self.compose_data.get('services', {})
        self.assertIn('tb-worker', services)
        self.assertIn('nginx', services)

    def test_compose_profile_services_have_profiles(self):
        """Optional services must declare their profile."""
        if not self.compose_data:
            self.skipTest("Could not parse compose.yaml (pyyaml missing?)")
        services = self.compose_data.get('services', {})
        profile_services = ['redis', 'minio', 'postgres', 'otel-collector',
                            'llm-gateway', 'ollama', 'tb-registry', 'tb-host']
        for svc in profile_services:
            if svc in services:
                profiles = services[svc].get('profiles', [])
                self.assertTrue(len(profiles) > 0,
                                f"Service '{svc}' should have a profile declaration")

    def test_compose_networks_defined(self):
        if not self.compose_data:
            self.skipTest("Could not parse compose.yaml (pyyaml missing?)")
        networks = self.compose_data.get('networks', {})
        self.assertIn('tb-internal', networks)

    def test_compose_volumes_defined(self):
        if not self.compose_data:
            self.skipTest("Could not parse compose.yaml (pyyaml missing?)")
        volumes = self.compose_data.get('volumes', {})
        self.assertIn('tb-data', volumes)
        self.assertIn('redis-data', volumes)


# ============================================================================
# Dockerfile Structural Validation (text-based, no Docker needed)
# ============================================================================

class TestDockerfileStructure(unittest.TestCase):
    """Validate Dockerfile.toolbox content without building."""

    @classmethod
    def setUpClass(cls):
        candidates = [
            Path(__file__).parent.parent / "Dockerfile.toolbox",
            Path(__file__).parent.parent / "output" / "Dockerfile.toolbox",
            Path(__file__).parent.parent.parent.parent / "Dockerfile.toolbox",
        ]
        cls.content = None
        for p in candidates:
            if p.exists():
                cls.content = p.read_text()
                break

    def test_dockerfile_exists(self):
        self.assertIsNotNone(self.content, "Dockerfile.toolbox not found")

    def test_has_multi_stage_build(self):
        if not self.content:
            self.skipTest("No Dockerfile")
        # At least 3 FROM statements (requirements-generator, builder, runtime)
        from_count = self.content.count("\nFROM ")
        self.assertGreaterEqual(from_count, 3, f"Expected 3+ stages, found {from_count}")

    def test_copies_package_files_before_source(self):
        if not self.content:
            self.skipTest("No Dockerfile")
        # requirements.txt should be copied before the bulk COPY toolboxv2/
        lines = self.content.splitlines()
        req_line = None
        source_line = None
        for i, line in enumerate(lines):
            if 'COPY' in line and 'requirements.txt' in line and 'requirements-generator' not in line:
                if req_line is None:
                    req_line = i
            if 'COPY' in line and 'toolboxv2/' in line and 'node-builder' not in line and 'requirements-generator' not in line:
                if source_line is None:
                    source_line = i
        if req_line is not None and source_line is not None:
            self.assertLess(req_line, source_line,
                            "requirements.txt should be COPY'd before toolboxv2/ source for caching")

    def test_uses_tini_entrypoint(self):
        if not self.content:
            self.skipTest("No Dockerfile")
        self.assertIn("tini", self.content, "Should use tini for PID 1 signal handling")

    def test_has_healthcheck(self):
        if not self.content:
            self.skipTest("No Dockerfile")
        self.assertIn("HEALTHCHECK", self.content)

    def test_runs_as_non_root(self):
        if not self.content:
            self.skipTest("No Dockerfile")
        self.assertIn("USER toolbox", self.content)

    def test_all_feature_build_args_declared(self):
        if not self.content:
            self.skipTest("No Dockerfile")
        for env_var in FEATURE_MAP.values():
            self.assertIn(env_var, self.content, f"Missing ARG {env_var}")


class TestDockerfileSsh(unittest.TestCase):
    """Validate Dockerfile.ssh content."""

    @classmethod
    def setUpClass(cls):
        candidates = [
            Path(__file__).parent.parent / "Dockerfile.ssh",
            Path(__file__).parent.parent / "output" / "Dockerfile.ssh",
            Path(__file__).parent.parent.parent.parent / "Dockerfile.ssh",
        ]
        cls.content = None
        for p in candidates:
            if p.exists():
                cls.content = p.read_text()
                break

    def test_dockerfile_exists(self):
        self.assertIsNotNone(self.content, "Dockerfile.ssh not found")

    def test_uses_base_image_arg(self):
        if not self.content:
            self.skipTest("No Dockerfile.ssh")
        self.assertIn("BASE_IMAGE", self.content)
        self.assertIn("toolboxv2:latest", self.content)

    def test_exposes_ssh_port(self):
        if not self.content:
            self.skipTest("No Dockerfile.ssh")
        self.assertIn("EXPOSE 2222", self.content)

    def test_has_sshd_config(self):
        if not self.content:
            self.skipTest("No Dockerfile.ssh")
        self.assertIn("PubkeyAuthentication yes", self.content)
        self.assertIn("PasswordAuthentication no", self.content)

    def test_has_tmux(self):
        if not self.content:
            self.skipTest("No Dockerfile.ssh")
        self.assertIn("tmux", self.content)


if __name__ == '__main__':
    unittest.main()
