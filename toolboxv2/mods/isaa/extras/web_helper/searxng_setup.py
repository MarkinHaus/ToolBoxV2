#!/usr/bin/env python3
"""
SearXNG Setup Script - Cross-Platform (Windows/Linux/Mac/Docker)
================================================================

Minimales Setup für lokale SearXNG Instanz mit JSON API.

Usage:
    python searxng_setup.py              # Auto-detect & setup
    python searxng_setup.py --docker     # Force Docker
    python searxng_setup.py --start      # Start existing
    python searxng_setup.py --stop       # Stop
    python searxng_setup.py --status     # Check status
    python searxng_setup.py --test       # Test API
"""

import argparse
import os
import platform
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

SEARXNG_PORT = 8072
SEARXNG_DIR = Path.home() / ".searxng"
DOCKER_COMPOSE_FILE = SEARXNG_DIR / "docker-compose.yml"
SETTINGS_FILE = SEARXNG_DIR / "searxng" / "settings.yml"

# Minimal settings.yml with JSON API enabled
SETTINGS_YML = """# SearXNG Settings - Minimal Configuration for API Usage
# Documentation: https://docs.searxng.org/admin/settings/settings.html

use_default_settings: true

general:
  instance_name: "SearXNG Local"
  debug: false
  enable_metrics: false

server:
  bind_address: "0.0.0.0:8080"
  secret_key: "{secret_key}"
  limiter: false  # Important: Disable rate limiting for local use
  image_proxy: true
  method: "GET"

ui:
  static_use_hash: true
  default_theme: simple
  theme_args:
    simple_style: auto

search:
  safe_search: 0
  autocomplete: "google"
  default_lang: "auto"
  formats:
    - html
    - json    # Required for API!
    - csv
    - rss

# Enabled search engines
engines:
  - name: google
    engine: google
    shortcut: g
    disabled: false

  - name: bing
    engine: bing
    shortcut: b
    disabled: false

  - name: duckduckgo
    engine: duckduckgo
    shortcut: ddg
    disabled: false

  - name: brave
    engine: brave
    shortcut: br
    disabled: false

  - name: wikipedia
    engine: wikipedia
    shortcut: w
    disabled: false

  - name: github
    engine: github
    shortcut: gh
    disabled: false

  - name: stackoverflow
    engine: stackoverflow
    shortcut: st
    disabled: false

outgoing:
  request_timeout: 5.0
  max_request_timeout: 15.0
  useragent_suffix: ""
  pool_connections: 100
  pool_maxsize: 20
"""

# Minimal Docker Compose
DOCKER_COMPOSE_YML = """# SearXNG Docker Compose - Minimal Setup
# Start: docker compose up -d
# Stop:  docker compose down

services:
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    restart: unless-stopped
    ports:
      - "{port}:8080"
    volumes:
      - ./searxng:/etc/searxng:rw
    environment:
      - SEARXNG_BASE_URL=http://localhost:{port}/
      - UWSGI_WORKERS=4
      - UWSGI_THREADS=4
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - SETGID
      - SETUID
    logging:
      driver: "json-file"
      options:
        max-size: "1m"
        max-file: "1"
"""


# ============================================================================
# HELPERS
# ============================================================================

class Colors:
    """ANSI colors for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

    @classmethod
    def disable(cls):
        cls.GREEN = cls.YELLOW = cls.RED = cls.BLUE = cls.BOLD = cls.END = ''


def log(msg: str, level: str = "info"):
    """Print colored log message."""
    icons = {
        "info": f"{Colors.BLUE}ℹ{Colors.END}",
        "success": f"{Colors.GREEN}✓{Colors.END}",
        "warning": f"{Colors.YELLOW}⚠{Colors.END}",
        "error": f"{Colors.RED}✗{Colors.END}",
        "step": f"{Colors.BOLD}→{Colors.END}",
    }
    print(f"{icons.get(level, '•')} {msg}")


def run_cmd(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run command and return result."""
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        if capture:
            log(f"Command failed: {e.stderr}", "error")
        raise
    except FileNotFoundError:
        return None


def get_platform() -> str:
    """Detect current platform."""
    system = platform.system().lower()
    if system == "darwin":
        return "mac"
    elif system == "windows":
        return "windows"
    else:
        return "linux"


def check_docker() -> bool:
    """Check if Docker is available."""
    result = run_cmd(["docker", "--version"], check=False, capture=True)
    if result and result.returncode == 0:
        return True
    return False


def check_docker_compose() -> bool:
    """Check if Docker Compose is available."""
    # Try new syntax first (docker compose)
    result = run_cmd(["docker", "compose", "version"], check=False, capture=True)
    if result and result.returncode == 0:
        return True

    # Try old syntax (docker-compose)
    result = run_cmd(["docker-compose", "--version"], check=False, capture=True)
    if result and result.returncode == 0:
        return True

    return False


def get_docker_compose_cmd() -> list[str]:
    """Get the correct docker compose command."""
    result = run_cmd(["docker", "compose", "version"], check=False, capture=True)
    if result and result.returncode == 0:
        return ["docker", "compose"]
    return ["docker-compose"]


# ============================================================================
# SETUP FUNCTIONS
# ============================================================================

def create_directories():
    """Create necessary directories."""
    log("Creating directories...", "step")

    SEARXNG_DIR.mkdir(parents=True, exist_ok=True)
    (SEARXNG_DIR / "searxng").mkdir(exist_ok=True)

    log(f"Created: {SEARXNG_DIR}", "success")


def create_settings():
    """Create settings.yml with JSON API enabled."""
    log("Creating settings.yml...", "step")

    # Generate secret key
    secret_key = secrets.token_hex(32)

    settings_content = SETTINGS_YML.format(secret_key=secret_key)

    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(settings_content)

    log(f"Created: {SETTINGS_FILE}", "success")
    log("JSON API enabled ✓", "success")


def create_docker_compose():
    """Create docker-compose.yml."""
    log("Creating docker-compose.yml...", "step")

    compose_content = DOCKER_COMPOSE_YML.format(port=SEARXNG_PORT)
    DOCKER_COMPOSE_FILE.write_text(compose_content)

    log(f"Created: {DOCKER_COMPOSE_FILE}", "success")


def start_docker():
    """Start SearXNG with Docker Compose."""
    log("Starting SearXNG with Docker...", "step")

    os.chdir(SEARXNG_DIR)

    compose_cmd = get_docker_compose_cmd()
    run_cmd(compose_cmd + ["up", "-d"])

    log("SearXNG container started", "success")


def stop_docker():
    """Stop SearXNG Docker container."""
    log("Stopping SearXNG...", "step")

    if not DOCKER_COMPOSE_FILE.exists():
        log("No docker-compose.yml found", "warning")
        return

    os.chdir(SEARXNG_DIR)

    compose_cmd = get_docker_compose_cmd()
    run_cmd(compose_cmd + ["down"], check=False)

    log("SearXNG stopped", "success")


def check_status() -> bool:
    """Check if SearXNG is running."""
    result = run_cmd(
        ["docker", "ps", "--filter", "name=searxng", "--format", "{{.Status}}"],
        check=False,
        capture=True
    )

    if result and result.returncode == 0 and result.stdout.strip():
        log(f"SearXNG is running: {result.stdout.strip()}", "success")
        return True
    else:
        log("SearXNG is not running", "warning")
        return False


def wait_for_ready(timeout: int = 60) -> bool:
    """Wait for SearXNG to be ready."""
    log(f"Waiting for SearXNG to be ready (max {timeout}s)...", "step")

    if not HTTPX_AVAILABLE:
        # Fallback ohne httpx
        time.sleep(10)
        return True

    url = f"http://localhost:{SEARXNG_PORT}/search?q=test&format=json"

    start = time.time()
    while time.time() - start < timeout:
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    log("SearXNG is ready!", "success")
                    return True
        except:
            pass
        time.sleep(2)
        print(".", end="", flush=True)

    print()
    log("Timeout waiting for SearXNG", "warning")
    return False


def test_api():
    """Test the SearXNG JSON API."""
    log("Testing SearXNG API...", "step")

    if not HTTPX_AVAILABLE:
        log("httpx not installed, skipping test. Run: pip install httpx", "warning")
        return False

    url = f"http://localhost:{SEARXNG_PORT}/search"
    params = {"q": "test", "format": "json"}

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(url, params=params)

            if resp.status_code != 200:
                log(f"HTTP {resp.status_code}: {resp.text[:100]}", "error")
                return False

            data = resp.json()
            results = data.get("results", [])

            log(f"API Response OK - {len(results)} results", "success")

            if results:
                log(f"First result: {results[0].get('title', 'N/A')[:50]}...", "info")

            return True

    except Exception as e:
        log(f"API test failed: {e}", "error")
        return False


# ============================================================================
# PLATFORM-SPECIFIC DOCKER INSTALL INSTRUCTIONS
# ============================================================================

def print_docker_install_instructions():
    """Print Docker installation instructions for current platform."""
    plat = get_platform()

    print(f"\n{Colors.BOLD}Docker Installation Instructions ({plat}){Colors.END}\n")

    if plat == "windows":
        print("""
1. Download Docker Desktop:
   https://www.docker.com/products/docker-desktop/

2. Install and restart

3. Run this script again

Alternative (WSL2):
   wsl --install
   # Then install Docker in WSL2
""")

    elif plat == "mac":
        print("""
Option 1 - Docker Desktop:
   https://www.docker.com/products/docker-desktop/

Option 2 - Homebrew:
   brew install --cask docker
   # Open Docker.app to complete setup

Option 3 - Colima (lightweight):
   brew install colima docker
   colima start
""")

    else:  # Linux
        print("""
# Ubuntu/Debian:
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in

# Arch:
sudo pacman -S docker docker-compose
sudo systemctl enable --now docker

# Fedora:
sudo dnf install docker docker-compose
sudo systemctl enable --now docker
""")


# ============================================================================
# MAIN SETUP
# ============================================================================

def setup():
    """Main setup function."""
    print(f"\n{Colors.BOLD}═══════════════════════════════════════════════════════════{Colors.END}")
    print(f"{Colors.BOLD}  SearXNG Local Setup{Colors.END}")
    print(f"{Colors.BOLD}═══════════════════════════════════════════════════════════{Colors.END}\n")

    # Check platform
    plat = get_platform()
    log(f"Platform: {plat}", "info")

    # Check Docker
    if not check_docker():
        log("Docker not found!", "error")
        print_docker_install_instructions()
        sys.exit(1)

    log("Docker found ✓", "success")

    if not check_docker_compose():
        log("Docker Compose not found!", "error")
        print_docker_install_instructions()
        sys.exit(1)

    log("Docker Compose found ✓", "success")

    # Create files
    create_directories()
    create_settings()
    create_docker_compose()

    # Start
    start_docker()

    # Wait and test
    if wait_for_ready():
        test_api()

    # Print summary
    print(f"\n{Colors.BOLD}═══════════════════════════════════════════════════════════{Colors.END}")
    print(f"{Colors.GREEN}SearXNG is ready!{Colors.END}")
    print(f"{Colors.BOLD}═══════════════════════════════════════════════════════════{Colors.END}")
    print(f"""
  Web UI:  http://localhost:{SEARXNG_PORT}
  API:     http://localhost:{SEARXNG_PORT}/search?q=test&format=json

  Directory: {SEARXNG_DIR}

  Commands:
    python {sys.argv[0]} --start   # Start
    python {sys.argv[0]} --stop    # Stop
    python {sys.argv[0]} --status  # Check status
    python {sys.argv[0]} --test    # Test API

  In Python:
    from web_agent import WebAgent
    agent = WebAgent(searxng_url="http://localhost:{SEARXNG_PORT}")
""")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SearXNG Local Setup - Cross-Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python searxng_setup.py              # Full setup
  python searxng_setup.py --start      # Start existing
  python searxng_setup.py --stop       # Stop
  python searxng_setup.py --test       # Test API
        """
    )

    parser.add_argument("--start", action="store_true", help="Start SearXNG")
    parser.add_argument("--stop", action="store_true", help="Stop SearXNG")
    parser.add_argument("--status", action="store_true", help="Check status")
    parser.add_argument("--test", action="store_true", help="Test API")
    parser.add_argument("--port", type=int, default=8072, help="Port (default: 8072)")
    parser.add_argument("--no-color", action="store_true", help="Disable colors")

    args = parser.parse_args()

    global SEARXNG_PORT
    SEARXNG_PORT = args.port

    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    # Handle commands
    if args.stop:
        stop_docker()
    elif args.status:
        check_status()
    elif args.test:
        test_api()
    elif args.start:
        if DOCKER_COMPOSE_FILE.exists():
            start_docker()
            wait_for_ready()
        else:
            log("No existing setup found. Running full setup...", "warning")
            setup()
    else:
        setup()


if __name__ == "__main__":
    main()
