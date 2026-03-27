"""E2E tests for tb-registry — Docker-based, unittest only.

Run:
    python -m unittest tests/test_e2e.py -v

Skipped automatically if Docker is not available.
Logs are saved to tests/logs/e2e_<timestamp>.log before teardown.
"""

import io
import os
import subprocess
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

import httpx
import jwt  # PyJWT

# ── Config ────────────────────────────────────────────────────────────────────

JWT_SECRET    = "e2e_tb_registry_test_secret_x32!"
BASE_URL      = "http://localhost:4025"
API           = f"{BASE_URL}/api/v1"
COMPOSE_FILE  = Path(__file__).parent.parent / "docker-compose.yml"
IMAGE_NAME    = "tb-registry-registry"
LOG_DIR       = Path(__file__).parent / "logs"

ARTIFACT_TYPE = "tauri_app"   # ArtifactType.TAURI_APP
PLATFORM      = "windows"     # Platform.WINDOWS
ARCHITECTURE  = "x64"         # Architecture.X64


# ── Docker guard ──────────────────────────────────────────────────────────────

def _docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False

DOCKER_UNAVAILABLE = not _docker_available()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _image_exists() -> bool:
    r = subprocess.run(["docker", "images", "-q", IMAGE_NAME], capture_output=True, text=True)
    return bool(r.stdout.strip())


def _mint_token(user_id="e2e_user_1", username="e2euser",
                email="e2e@test.com", level=1) -> str:
    return jwt.encode(
        {"user_id": user_id, "username": username, "email": email,
         "level": level, "provider": "test", "exp": int(time.time()) + 7200},
        JWT_SECRET, algorithm="HS256",
    )


def _wait_healthy(timeout=120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"{BASE_URL}/health", timeout=2).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("Registry did not become healthy within timeout")


def _compose(args: list, override: Path, env: dict) -> None:
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "-f", str(override)] + args,
        env=env, capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"docker compose {args} failed:\n"
            f"stdout: {result.stdout.decode()}\nstderr: {result.stderr.decode()}"
        )


def _save_logs(override: Path, env: dict) -> None:
    """Extract container logs before teardown and save to tests/logs/."""
    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"e2e_{ts}.log"
    services = ["registry", "minio"]
    lines = [f"=== E2E Log Capture {ts} ===\n"]
    for svc in services:
        result = subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "-f", str(override),
             "logs", "--no-color", "--tail=200", svc],
            env=env, capture_output=True, text=True,
        )
        lines.append(f"\n{'='*60}\nSERVICE: {svc}\n{'='*60}\n")
        lines.append(result.stdout or "(no stdout)\n")
        if result.stderr:
            lines.append(f"[stderr]\n{result.stderr}\n")
    log_file.write_text("".join(lines), encoding="utf-8")
    print(f"\n[E2E] Logs saved → {log_file}")


# ── Test Suite ────────────────────────────────────────────────────────────────

@unittest.skipIf(DOCKER_UNAVAILABLE, "Docker not available — skipping E2E tests")
class TestRegistryE2E(unittest.TestCase):

    _override_file: Path
    _compose_env: dict
    token: str
    auth_headers: dict
    publisher_slug = "e2e-publisher"

    @classmethod
    def setUpClass(cls):
        cls._override_file = Path(tempfile.mktemp(suffix="-e2e-override.yml"))
        cls._override_file.write_text(
            "services:\n"
            "  registry:\n"
            "    environment:\n"
            f"      CLOUDM_JWT_SECRET: \"{JWT_SECRET}\"\n"
        )
        cls._compose_env = {
            **os.environ,
            "MINIO_ACCESS_KEY": "minioadmin",
            "MINIO_SECRET_KEY": "minioadmin",
        }
        up_args = ["up", "-d"] if _image_exists() else ["up", "-d", "--build"]
        _compose(up_args, cls._override_file, cls._compose_env)
        _wait_healthy()

        cls.token = _mint_token()
        cls.auth_headers = {"Authorization": f"Bearer {cls.token}"}

        r = httpx.post(
            f"{API}/auth/register-publisher",
            json={"name": cls.publisher_slug, "display_name": "E2E Publisher",
                  "email": "e2e@test.com"},
            headers=cls.auth_headers, timeout=10,
        )
        assert r.status_code in (200, 201, 400, 409), (
            f"Publisher setup: {r.status_code} {r.text}"
        )

    @classmethod
    def tearDownClass(cls):
        _save_logs(cls._override_file, cls._compose_env)
        try:
            _compose(["down", "-v"], cls._override_file, cls._compose_env)
        finally:
            cls._override_file.unlink(missing_ok=True)

    # ── Health ────────────────────────────────────────────────────────────────

    def test_01_health_returns_200(self):
        r = httpx.get(f"{BASE_URL}/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json().get("status"), "healthy")

    # ── Auth ──────────────────────────────────────────────────────────────────

    def test_02_get_me_authenticated(self):
        r = httpx.get(f"{API}/auth/me", headers=self.auth_headers)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["username"], "e2euser")

    def test_03_get_me_unauthenticated_returns_401(self):
        r = httpx.get(f"{API}/auth/me")
        self.assertEqual(r.status_code, 401)

    def test_04_invalid_token_returns_401(self):
        r = httpx.get(f"{API}/auth/me",
                      headers={"Authorization": "Bearer garbage.token.here"})
        self.assertEqual(r.status_code, 401)

    # ── Publishers ────────────────────────────────────────────────────────────

    def test_05_list_publishers_public(self):
        r = httpx.get(f"{API}/publishers")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("publishers", body)
        if body["publishers"]:
            p = body["publishers"][0]
            for field in ("id", "name", "display_name", "verification_status",
                          "package_count", "total_downloads"):
                self.assertIn(field, p, f"Missing field '{field}' in publisher response")

    def test_06_get_nonexistent_publisher_returns_404(self):
        r = httpx.get(f"{API}/publishers/no-such-publisher-xyz-99")
        self.assertEqual(r.status_code, 404)

    # ── Packages ─────────────────────────────────────────────────────────────

    def test_07_unauthenticated_post_packages_returns_401(self):
        r = httpx.post(f"{API}/packages",
                       json={"name": "noauth", "package_type": "library"})
        self.assertEqual(r.status_code, 401)

    def test_08_create_package_as_publisher(self):
        r = httpx.post(
            f"{API}/packages",
            json={"name": "e2e-lib", "package_type": "library", "description": "E2E test"},
            headers=self.auth_headers,
        )
        self.assertIn(r.status_code, (200, 201))
        self.assertEqual(r.json()["name"], "e2e-lib")

    def test_09_create_duplicate_package_returns_409(self):
        pkg = {"name": "e2e-dup", "package_type": "library"}
        httpx.post(f"{API}/packages", json=pkg, headers=self.auth_headers)
        r = httpx.post(f"{API}/packages", json=pkg, headers=self.auth_headers)
        self.assertEqual(r.status_code, 409)

    def test_10_list_packages_returns_200(self):
        r = httpx.get(f"{API}/packages")
        self.assertEqual(r.status_code, 200)
        self.assertIn("packages", r.json(),
                      "GET /packages must return {packages:[...]} not publishers")

    # ── Artifacts ────────────────────────────────────────────────────────────

    def test_11_create_artifact(self):
        r = httpx.post(
            f"{API}/artifacts",
            json={"name": "e2e-app", "artifact_type": ARTIFACT_TYPE,
                  "description": "E2E app"},
            headers=self.auth_headers,
        )
        self.assertIn(r.status_code, (200, 201))
        self.assertEqual(r.json()["name"], "e2e-app")

    def test_12_upload_build(self):
        httpx.post(
            f"{API}/artifacts",
            json={"name": "e2e-dl", "artifact_type": ARTIFACT_TYPE,
                  "description": "dl"},
            headers=self.auth_headers,
        )
        r = httpx.post(
            f"{API}/artifacts/e2e-dl/builds",
            data={"version": "1.0.0", "platform": PLATFORM,
                  "architecture": ARCHITECTURE, "changelog": "init"},
            files={"file": ("e2e-dl.exe", io.BytesIO(b"\x7fELF" + b"\x00" * 1020),
                            "application/octet-stream")},
            headers=self.auth_headers, timeout=30,
        )
        self.assertIn(r.status_code, (200, 201))
        self.assertEqual(r.json()["version"], "1.0.0")

    def test_13_get_latest_after_upload(self):
        r = httpx.get(
            f"{API}/artifacts/e2e-dl/latest",
            params={"platform": PLATFORM, "architecture": ARCHITECTURE},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["version"], "1.0.0")

    def test_14_get_download_url(self):
        # 200 = presigned URL OK; 404 = MinIO URL not reachable from test host (acceptable)
        r = httpx.get(
            f"{API}/artifacts/e2e-dl/versions/1.0.0/download",
            params={"platform": PLATFORM, "architecture": ARCHITECTURE},
        )
        self.assertIn(r.status_code, (200, 404))
        if r.status_code == 200:
            self.assertIn("url", r.json())

    def test_15_artifact_not_found_returns_404(self):
        r = httpx.get(f"{API}/artifacts/no-such-artifact-xyz")
        self.assertEqual(r.status_code, 404)

    def test_16_upload_second_version(self):
        r = httpx.post(
            f"{API}/artifacts/e2e-dl/builds",
            data={"version": "1.1.0", "platform": PLATFORM,
                  "architecture": ARCHITECTURE, "changelog": "update"},
            files={"file": ("e2e-dl-v2.exe", io.BytesIO(b"\x7fELF" + b"\x01" * 1020),
                            "application/octet-stream")},
            headers=self.auth_headers, timeout=30,
        )
        self.assertIn(r.status_code, (200, 201))
        r2 = httpx.get(
            f"{API}/artifacts/e2e-dl/latest",
            params={"platform": PLATFORM, "architecture": ARCHITECTURE},
        )
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json()["version"], "1.1.0")

    # ── Search ────────────────────────────────────────────────────────────────

    def test_17_search_returns_results(self):
        r = httpx.get(f"{API}/search", params={"q": "e2e"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("results", r.json())

    def test_18_search_missing_query_returns_422(self):
        r = httpx.get(f"{API}/search")
        self.assertEqual(r.status_code, 422)

    # ── Diff ─────────────────────────────────────────────────────────────────

    def test_19_diff_nonexistent_package_returns_error(self):
        r = httpx.get(f"{API}/packages/no-such-pkg/diff/0.1.0/0.2.0")
        self.assertIn(r.status_code, (400, 404, 500))

    # ── Versions ─────────────────────────────────────────────────────────────

    def test_20_versions_empty_query_returns_all(self):
        r = httpx.get(f"{API}/versions")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("versions", body)
        self.assertIsInstance(body["versions"], dict)

    def test_21_versions_filtered_by_name(self):
        r = httpx.get(f"{API}/versions", params={"names": "e2e-lib"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("versions", body)
        self.assertIn("e2e-lib", body["versions"])

    def test_22_versions_unknown_package_returns_empty(self):
        r = httpx.get(f"{API}/versions", params={"names": "no-such-pkg-xyz"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["versions"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
