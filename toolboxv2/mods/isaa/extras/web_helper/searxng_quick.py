#!/usr/bin/env python3
"""
SearXNG Quick Start - Ultra Minimal
====================================

Ein Befehl zum Starten von SearXNG mit Docker.

Usage:
    python searxng_quick.py              # Start
    python searxng_quick.py --stop       # Stop
    python searxng_quick.py --test       # Test API
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PORT = 8072
CONTAINER_NAME = "searxng-quick"
CONFIG_DIR = Path.home() / ".searxng-quick"

# Minimale settings.yml als String
SETTINGS = """use_default_settings: true
server:
  limiter: false
  secret_key: "{secret}"
search:
  formats: [html, json]
"""


def run(cmd: str, check: bool = True) -> bool:
    """Shell command ausführen."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return result.returncode == 0


def start():
    """SearXNG starten."""
    print("🚀 Starting SearXNG...")

    # Config erstellen
    CONFIG_DIR.mkdir(exist_ok=True)
    settings_file = CONFIG_DIR / "settings.yml"

    import secrets
    settings_file.write_text(SETTINGS.format(secret=secrets.token_hex(16)))

    # Alten Container entfernen falls vorhanden
    run(f"docker rm -f {CONTAINER_NAME}", check=False)

    # Container starten
    cmd = f"""docker run -d \
        --name {CONTAINER_NAME} \
        -p {PORT}:8072 \
        -v {CONFIG_DIR}:/etc/searxng:rw \
        searxng/searxng:latest"""

    if not run(cmd):
        print("❌ Failed to start container")
        print("\nIs Docker running? Try: docker info")
        return False

    print(f"✅ SearXNG started!")
    print(f"\n   Web UI: http://localhost:{PORT}")
    print(f"   API:    http://localhost:{PORT}/search?q=test&format=json")

    # Kurz warten und testen
    print("\n⏳ Waiting for startup...")
    time.sleep(5)
    test()

    return True


def stop():
    """SearXNG stoppen."""
    print("🛑 Stopping SearXNG...")
    run(f"docker stop {CONTAINER_NAME}", check=False)
    run(f"docker rm {CONTAINER_NAME}", check=False)
    print("✅ Stopped")


def test():
    """API testen."""
    try:
        import urllib.request
        import json

        url = f"http://localhost:{PORT}/search?q=test&format=json"

        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            results = data.get("results", [])
            print(f"✅ API working - {len(results)} results")
            if results:
                print(f"   First: {results[0].get('title', 'N/A')[:50]}")
            return True

    except Exception as e:
        print(f"❌ API test failed: {e}")
        print(f"   Make sure SearXNG is running: docker ps")
        return False


def status():
    """Status prüfen."""
    result = subprocess.run(
        f"docker ps --filter name={CONTAINER_NAME} --format '{{{{.Status}}}}'",
        shell=True, capture_output=True, text=True
    )
    if result.stdout.strip():
        print(f"✅ Running: {result.stdout.strip()}")
    else:
        print("❌ Not running")


def main():
    parser = argparse.ArgumentParser(description="SearXNG Quick Start")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--port", type=int, default=8072)

    args = parser.parse_args()

    global PORT
    PORT = args.port

    if args.stop:
        stop()
    elif args.test:
        test()
    elif args.status:
        status()
    else:
        start()


if __name__ == "__main__":
    main()
