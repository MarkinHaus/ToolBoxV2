#!/usr/bin/env python3
"""
ProjectDeveloper Studio - Launcher Script

Usage:
    python run.py [--port PORT] [--host HOST] [--mock]

Options:
    --port PORT    Port to run on (default: 8501)
    --host HOST    Host to bind to (default: localhost)
    --mock         Force mock mode (no LLM)
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    # Parse arguments
    port = "8501"
    host = "localhost"

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--port" and i + 1 < len(args):
            port = args[i + 1]
        elif arg == "--host" and i + 1 < len(args):
            host = args[i + 1]

    # Get the app directory
    app_dir = Path(__file__).parent
    app_file = app_dir / "app.py"

    if not app_file.exists():
        print(f"Error: {app_file} not found!")
        sys.exit(1)

    # Check dependencies
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit>=1.40.0"])

    # Build command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_file),
        "--server.port", port,
        "--server.address", host,
        "--theme.base", "dark",
        "--theme.primaryColor", "#6366f1",
        "--theme.backgroundColor", "#0a0e17",
        "--theme.secondaryBackgroundColor", "#1a2332",
        "--theme.textColor", "#f1f5f9",
    ]

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🚀 ProjectDeveloper Studio                                 ║
║                                                              ║
║   Starting on http://{host}:{port}                          ║
║                                                              ║
║   Press Ctrl+C to stop                                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Run Streamlit
    try:
        subprocess.run(cmd, cwd=str(app_dir))
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")


if __name__ == "__main__":
    main()
