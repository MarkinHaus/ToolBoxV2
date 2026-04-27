#!/usr/bin/env python3
"""
jsxserve — Serve JSX files locally. Zero Node.js dependency.

Transpilation happens in-browser via esm.sh + Babel standalone.
Just point it at a .jsx file and open your browser.

Usage:
    python jsxserve.py                          # serves all .jsx in cwd
    python jsxserve.py app.jsx                  # serves specific file
    python jsxserve.py app.jsx --port 5173      # custom port
    python jsxserve.py --dir components/        # serve from directory
    python jsxserve.py app.jsx --no-reload      # disable auto-reload

Features:
    - In-browser JSX transpilation (Babel standalone)
    - React 18 + ReactDOM via esm.sh CDN
    - Auto-reload on file changes (WebSocket)
    - Tailwind CSS via CDN (utility classes ready)
    - Zero npm, zero node, zero build step
    - Serves static assets (images, CSS, JSON) from same directory
"""

import argparse
import asyncio
import hashlib
import http.server
import json
import mimetypes
import os
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.parse import unquote

# ─── Config ───

DEFAULT_PORT = 5173
RELOAD_INTERVAL = 0.5  # seconds between file checks

# CDN URLs — pinned versions for stability
REACT_CDN = "https://esm.sh/react@18.3.1"
REACT_DOM_CDN = "https://esm.sh/react-dom@18.3.1/client"
BABEL_CDN = "https://unpkg.com/@babel/standalone@7.26.10/babel.min.js"
TAILWIND_CDN = "https://cdn.tailwindcss.com"

# Additional libraries available via esm.sh (same as Claude artifacts)
ESM_LIBS = {
    "lucide-react": "https://esm.sh/lucide-react@0.383.0",
    "recharts": "https://esm.sh/recharts@2.15.3?external=react,react-dom",
    "mathjs": "https://esm.sh/mathjs@13.2.3",
    "lodash": "https://esm.sh/lodash@4.17.21",
    "d3": "https://esm.sh/d3@7.9.0",
    "three": "https://esm.sh/three@0.170.0",
    "papaparse": "https://esm.sh/papaparse@5.5.2",
    "chart.js": "https://esm.sh/chart.js@4.4.8",
    "tone": "https://esm.sh/tone@15.1.22",
}


def _build_importmap(jsx_source: str) -> dict:
    """Scan JSX source for imports and build an import map."""
    imports = {
        "react": REACT_CDN,
        "react-dom": REACT_DOM_CDN + "/../",
        "react-dom/client": REACT_DOM_CDN,
        "react/": REACT_CDN + "/",
    }
    for lib, url in ESM_LIBS.items():
        # Only include libs that are actually imported
        if f'from "{lib}"' in jsx_source or f"from '{lib}'" in jsx_source:
            imports[lib] = url
    return {"imports": imports}


def _build_html(jsx_path: Path, jsx_source: str, ws_port: int | None = None) -> str:
    """Wrap a JSX file in a full HTML shell with in-browser transpilation."""
    importmap = _build_importmap(jsx_source)
    title = jsx_path.stem.replace("-", " ").replace("_", " ").title()

    # Escape JSX for embedding in script tag
    # We use a separate fetch instead to avoid escaping issues
    reload_script = ""
    if ws_port:
        reload_script = f"""
    <script>
      (function() {{
        let ws;
        function connect() {{
          ws = new WebSocket('ws://localhost:{ws_port}');
          ws.onmessage = function(e) {{
            if (e.data === 'reload') window.location.reload();
          }};
          ws.onclose = function() {{
            setTimeout(connect, 1000);
          }};
        }}
        connect();
      }})();
    </script>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="{TAILWIND_CDN}"></script>
    <script src="{BABEL_CDN}"></script>
    <script type="importmap">
    {json.dumps(importmap, indent=2)}
    </script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        #root {{ min-height: 100vh; }}
    </style>{reload_script}
</head>
<body>
    <div id="root"></div>
    <script>var exports = {{}}, module = {{ exports: exports }};</script>
    <script type="text/babel" data-type="module" data-plugins="transform-modules-commonjs">
{jsx_source}

// ─── Mount ───
const _mod = typeof App !== 'undefined' ? App
           : typeof BenchConfigUI !== 'undefined' ? BenchConfigUI
           : null;

// Check for default export pattern
const _default = (() => {{
  // Babel commonjs transform puts default export on module.exports or exports.default
  if (typeof exports !== 'undefined' && exports.default) return exports.default;
  return _mod;
}})();

if (_default) {{
  const React = require('react');
  const ReactDOM = require('react-dom/client');
  const root = ReactDOM.createRoot(document.getElementById('root'));
  root.render(React.createElement(_default));
}} else {{
  document.getElementById('root').innerHTML =
    '<div style="padding:40px;color:#888;font-family:monospace;">' +
    'No default export or App/BenchConfigUI component found.' +
    '</div>';
}}
    </script>
</body>
</html>"""


# ─── File Watcher (no dependencies) ───

class FileWatcher:
    """Watch files for changes using polling. No watchdog dependency."""

    def __init__(self, paths: list[Path], callback, interval: float = RELOAD_INTERVAL):
        self.paths = paths
        self.callback = callback
        self.interval = interval
        self._hashes: dict[str, str] = {}
        self._running = False
        self._thread = None
        self._init_hashes()

    def _hash_file(self, path: Path) -> str:
        try:
            return hashlib.md5(path.read_bytes()).hexdigest()
        except (OSError, IOError):
            return ""

    def _init_hashes(self):
        for p in self.paths:
            self._hashes[str(p)] = self._hash_file(p)

    def _check(self):
        for p in self.paths:
            key = str(p)
            new_hash = self._hash_file(p)
            if new_hash and new_hash != self._hashes.get(key):
                self._hashes[key] = new_hash
                self.callback(p)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            self._check()
            time.sleep(self.interval)

    def stop(self):
        self._running = False


# ─── WebSocket server (minimal, RFC 6455) ───

class SimpleWSServer:
    """Minimal WebSocket server for reload notifications."""

    def __init__(self, port: int):
        self.port = port
        self.clients: list[socket.socket] = []
        self._server = None
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", self.port))
        self._server.listen(5)
        self._server.settimeout(1.0)

        while True:
            try:
                conn, addr = self._server.accept()
                t = threading.Thread(target=self._handshake, args=(conn,), daemon=True)
                t.start()
            except socket.timeout:
                continue
            except OSError:
                break

    def _handshake(self, conn: socket.socket):
        """Perform WebSocket handshake."""
        try:
            data = conn.recv(4096).decode("utf-8", errors="replace")
            if "Upgrade: websocket" not in data:
                conn.close()
                return

            # Extract Sec-WebSocket-Key
            key = ""
            for line in data.split("\r\n"):
                if line.startswith("Sec-WebSocket-Key:"):
                    key = line.split(": ", 1)[1].strip()
                    break

            if not key:
                conn.close()
                return

            import base64
            import struct

            magic = "258EAFA5-E914-47DA-95CA-5AB5DC11B85A"
            accept = base64.b64encode(
                hashlib.sha1((key + magic).encode()).digest()
            ).decode()

            response = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
            )
            conn.sendall(response.encode())
            self.clients.append(conn)

            # Keep connection alive — read and discard
            while True:
                try:
                    d = conn.recv(1024)
                    if not d:
                        break
                except (OSError, ConnectionError):
                    break

            if conn in self.clients:
                self.clients.remove(conn)
            conn.close()
        except Exception:
            try:
                conn.close()
            except Exception:
                pass

    def send_reload(self):
        """Send 'reload' message to all connected WebSocket clients."""
        import struct

        msg = b"reload"
        frame = bytearray()
        frame.append(0x81)  # text frame, FIN
        frame.append(len(msg))
        frame.extend(msg)

        dead = []
        for client in self.clients:
            try:
                client.sendall(bytes(frame))
            except (OSError, ConnectionError):
                dead.append(client)

        for d in dead:
            if d in self.clients:
                self.clients.remove(d)
            try:
                d.close()
            except Exception:
                pass


# ─── HTTP Handler ───

class JSXHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that serves JSX files wrapped in HTML."""

    jsx_files: dict[str, Path] = {}
    base_dir: Path = Path(".")
    ws_port: int | None = None

    def do_GET(self):
        path = unquote(self.path.lstrip("/"))

        # Root — serve index or file listing
        if not path or path == "/":
            if len(self.jsx_files) == 1:
                # Single file mode — serve directly
                name = list(self.jsx_files.keys())[0]
                self._serve_jsx(self.jsx_files[name])
            else:
                self._serve_index()
            return

        # Check if it's a JSX route
        # Strip .html suffix if present
        route = path.removesuffix(".html")
        if route in self.jsx_files:
            self._serve_jsx(self.jsx_files[route])
            return

        # Also try with .jsx extension stripped
        if route.removesuffix(".jsx") in self.jsx_files:
            self._serve_jsx(self.jsx_files[route.removesuffix(".jsx")])
            return

        # Static file
        static_path = self.base_dir / path
        if static_path.is_file():
            self._serve_static(static_path)
            return

        self.send_error(404)

    def _serve_jsx(self, jsx_path: Path):
        try:
            source = jsx_path.read_text(encoding="utf-8")
            html = _build_html(jsx_path, source, ws_port=self.ws_port)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        except Exception as e:
            self.send_error(500, str(e))

    def _serve_index(self):
        items = ""
        for name, path in sorted(self.jsx_files.items()):
            items += (
                f'<a href="/{name}" style="display:block;padding:12px 20px;'
                f'border:1px solid rgba(255,255,255,0.06);border-radius:4px;'
                f'color:#00e6d2;text-decoration:none;font-family:monospace;'
                f'font-size:14px;margin-bottom:8px;transition:all 0.15s;"'
                f' onmouseover="this.style.borderColor=\'rgba(0,230,210,0.3)\'"'
                f' onmouseout="this.style.borderColor=\'rgba(255,255,255,0.06)\'"'
                f'>{name}<span style="color:rgba(255,255,255,0.2);'
                f'margin-left:12px;font-size:11px;">{path}</span></a>\n'
            )
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>jsxserve</title></head>
<body style="background:#08080d;color:#e2e2e8;font-family:sans-serif;padding:40px;">
<div style="max-width:600px;margin:0 auto;">
<h1 style="font-size:18px;font-weight:300;margin-bottom:24px;color:rgba(255,255,255,0.4);">
jsxserve</h1>
{items}
</div></body></html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_static(self, path: Path):
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application/octet-stream"
        try:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(500, str(e))

    def log_message(self, format, *args):
        # Quieter logging
        if args and "404" in str(args[0]):
            return
        path = args[0] if args else ""
        sys.stderr.write(f"  {path}\n")


# ─── Main ───

def find_free_port(start: int = 9100) -> int:
    for port in range(start, start + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    return start + 50


def discover_jsx(directory: Path) -> dict[str, Path]:
    """Find all .jsx files in directory."""
    files = {}
    for p in sorted(directory.rglob("*.jsx")):
        rel = p.relative_to(directory)
        name = str(rel).removesuffix(".jsx")
        files[name] = p
    return files


def main():
    parser = argparse.ArgumentParser(
        prog="jsxserve",
        description="Serve JSX files locally. Zero Node.js dependency.",
    )
    parser.add_argument("file", nargs="?", help="JSX file to serve (default: all in cwd)")
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT, help=f"HTTP port (default: {DEFAULT_PORT})")
    parser.add_argument("--dir", "-d", default=".", help="Directory to serve from")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")

    args = parser.parse_args()
    base_dir = Path(args.dir).resolve()

    # Discover JSX files
    if args.file:
        jsx_path = Path(args.file).resolve()
        if not jsx_path.exists():
            print(f"File not found: {jsx_path}")
            sys.exit(1)
        jsx_files = {jsx_path.stem: jsx_path}
        base_dir = jsx_path.parent
    else:
        jsx_files = discover_jsx(base_dir)
        if not jsx_files:
            print(f"No .jsx files found in {base_dir}")
            sys.exit(1)

    # WebSocket for auto-reload
    ws_port = None
    ws_server = None
    watcher = None

    if not args.no_reload:
        ws_port = find_free_port()
        ws_server = SimpleWSServer(ws_port)
        ws_server.start()

        def on_change(path):
            print(f"  ↻ {path.name} changed, reloading...")
            ws_server.send_reload()

        watcher = FileWatcher(list(jsx_files.values()), on_change)
        watcher.start()

    # Configure handler
    JSXHandler.jsx_files = jsx_files
    JSXHandler.base_dir = base_dir
    JSXHandler.ws_port = ws_port

    # Start HTTP server
    server = http.server.HTTPServer((args.host, args.port), JSXHandler)

    print(f"\n  jsxserve")
    print(f"  {'─' * 40}")
    for name, path in jsx_files.items():
        print(f"  → http://{args.host}:{args.port}/{name}")
    if len(jsx_files) > 1:
        print(f"  → http://{args.host}:{args.port}/")
    print(f"  {'─' * 40}")
    if ws_port:
        print(f"  auto-reload: ws://localhost:{ws_port}")
    print(f"  serving from: {base_dir}")
    print(f"  Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
        if watcher:
            watcher.stop()
        server.server_close()


if __name__ == "__main__":
    main()
