#!/usr/bin/env python3
"""
jsxserve — Serve JSX files locally. Zero Node.js dependency.

Transpilation happens in-browser via Babel standalone.
React/ReactDOM loaded as UMD globals — no import maps needed.

Usage:
    python jsxserve.py                          # serves all .jsx in cwd
    python jsxserve.py app.jsx                  # serves specific file
    python jsxserve.py app.jsx --port 5173      # custom port
    python jsxserve.py --dir components/        # serve from directory
    python jsxserve.py app.jsx --no-reload      # disable auto-reload
"""

import argparse
import hashlib
import http.server
import json
import mimetypes
import os
import re
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.parse import unquote

# ─── Config ───

DEFAULT_PORT = 5173
RELOAD_INTERVAL = 0.5

# UMD builds — React/ReactDOM become window.React / window.ReactDOM
REACT_CDN = "https://unpkg.com/react@18/umd/react.production.min.js"
REACT_DOM_CDN = "https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"
BABEL_CDN = "https://unpkg.com/@babel/standalone@7.26.10/babel.min.js"
TAILWIND_CDN = "https://cdn.tailwindcss.com"

# Additional libs loaded as UMD globals via script tags
ESM_LIBS = {
    "lucide-react": "https://esm.sh/lucide-react@0.383.0?bundle&external=react,react-dom",
    "recharts": "https://esm.sh/recharts@2.15.3?bundle&external=react,react-dom",
    "mathjs": "https://esm.sh/mathjs@13.2.3?bundle",
    "lodash": "https://unpkg.com/lodash@4.17.21/lodash.min.js",
    "d3": "https://unpkg.com/d3@7.9.0/dist/d3.min.js",
    "three": "https://unpkg.com/three@0.170.0/build/three.min.js",
    "papaparse": "https://unpkg.com/papaparse@5.5.2/papaparse.min.js",
    "chart.js": "https://unpkg.com/chart.js@4.4.8/dist/chart.umd.min.js",
    "tone": "https://unpkg.com/tone@15.1.22/build/Tone.js",
}

# Map import names to the global variable they expose
LIB_GLOBALS = {
    "lucide-react": "lucideReact",
    "recharts": "Recharts",
    "mathjs": "math",
    "lodash": "_",
    "d3": "d3",
    "three": "THREE",
    "papaparse": "Papa",
    "chart.js": "Chart",
    "tone": "Tone",
}


def _detect_imports(jsx_source: str) -> list[str]:
    """Detect which libraries are imported in the JSX source."""
    found = []
    for lib in ESM_LIBS:
        if f'from "{lib}"' in jsx_source or f"from '{lib}'" in jsx_source:
            found.append(lib)
    return found


def _strip_imports(jsx_source: str) -> tuple[str, str | None]:
    """
    Strip ES module import/export statements from JSX source.
    Returns (cleaned_source, default_export_name).

    Handles:
    - import { x, y } from 'react';
    - import React from 'react';
    - import * as X from 'lib';
    - Multi-line imports
    - export default ComponentName;
    - export default function ...
    - Named destructured imports from libs → mapped to globals
    """
    lines = jsx_source.split('\n')
    cleaned = []
    default_export_name = None
    shim_lines = []  # Global destructuring shims

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines at top
        # Multi-line import: collect until closing ;
        if re.match(r'^import\s+', stripped):
            # Collect full import statement (may span multiple lines)
            full_import = stripped
            while not full_import.rstrip().endswith(';') and i + 1 < len(lines):
                i += 1
                full_import += ' ' + lines[i].strip()

            # Parse what's being imported and from where
            # Extract lib name
            lib_match = re.search(r'''from\s+['"]([^'"]+)['"]''', full_import)
            if lib_match:
                lib_name = lib_match.group(1)

                # For non-react libs, generate destructuring shims
                if lib_name in LIB_GLOBALS:
                    global_name = LIB_GLOBALS[lib_name]
                    # Extract named imports: import { X, Y as Z } from 'lib'
                    named_match = re.search(r'\{([^}]+)\}', full_import)
                    if named_match:
                        names = named_match.group(1)
                        # Parse "X, Y as Z" → "const { X, Y: Z } = global"
                        parts = []
                        for part in names.split(','):
                            part = part.strip()
                            if ' as ' in part:
                                orig, alias = part.split(' as ', 1)
                                parts.append(f"{orig.strip()}: {alias.strip()}")
                            else:
                                parts.append(part)
                        shim_lines.append(
                            f"const {{ {', '.join(parts)} }} = window.{global_name};"
                        )
                    # Default import: import X from 'lib'
                    default_match = re.match(
                        r'''import\s+(\w+)\s+from\s+['"]''', full_import
                    )
                    if default_match:
                        alias = default_match.group(1)
                        shim_lines.append(f"const {alias} = window.{global_name};")

                # React imports → destructure from window.React
                elif lib_name == 'react':
                    named_match = re.search(r'\{([^}]+)\}', full_import)
                    if named_match:
                        names = named_match.group(1)
                        parts = []
                        for part in names.split(','):
                            part = part.strip()
                            if ' as ' in part:
                                orig, alias = part.split(' as ', 1)
                                parts.append(f"{orig.strip()}: {alias.strip()}")
                            else:
                                parts.append(part)
                        shim_lines.append(
                            f"const {{ {', '.join(parts)} }} = React;"
                        )

                elif lib_name == 'react-dom/client':
                    # import { createRoot } from 'react-dom/client' → skip,
                    # mount logic handles this
                    pass

                # Unknown lib → just strip, hope for the best
            i += 1
            continue

        # export default ComponentName;
        export_match = re.match(r'^export\s+default\s+(\w+)\s*;?\s*$', stripped)
        if export_match:
            default_export_name = export_match.group(1)
            i += 1
            continue

        # export default function ComponentName(...)  { → keep as function, extract name
        export_fn_match = re.match(
            r'^export\s+default\s+(function\s+(\w+))', stripped
        )
        if export_fn_match:
            # Replace "export default function X" → "function X"
            cleaned.append(line.replace('export default ', '', 1))
            default_export_name = export_fn_match.group(2)
            i += 1
            continue

        # export default () => ... or export default class ...
        if stripped.startswith('export default '):
            # Generic: assign to a temp name
            rest = stripped.replace('export default ', '', 1)
            cleaned.append(f"const __DefaultExport__ = {rest}")
            default_export_name = '__DefaultExport__'
            i += 1
            continue

        # Named exports: export function X / export const X → just strip 'export'
        if re.match(r'^export\s+(function|const|let|var|class)\s+', stripped):
            cleaned.append(line.replace('export ', '', 1))
            i += 1
            continue

        cleaned.append(line)
        i += 1

    # Prepend shims at top
    result = '\n'.join(shim_lines + cleaned)
    return result, default_export_name


def _build_html(jsx_path: Path, jsx_source: str, ws_port: int | None = None) -> str:
    """Wrap a JSX file in a full HTML shell with in-browser transpilation."""
    title = jsx_path.stem.replace("-", " ").replace("_", " ").title()

    # Detect which extra libs are needed
    extra_libs = _detect_imports(jsx_source)

    # Strip imports/exports, get default export name
    cleaned_source, default_export_name = _strip_imports(jsx_source)

    # Build script tags for extra libs
    lib_scripts = ""
    for lib in extra_libs:
        if lib in ESM_LIBS:
            lib_scripts += f'    <script src="{ESM_LIBS[lib]}"></script>\n'

    # Build mount logic — try multiple detection strategies
    component_name = default_export_name or "App"
    # Also try to detect common component names from the source
    # Look for: function ComponentName( or const ComponentName =
    detected_names = re.findall(
        r'(?:function|const|let|var|class)\s+([A-Z]\w+)',
        cleaned_source
    )
    # Build fallback chain
    candidates = [component_name]
    for name in detected_names:
        if name not in candidates and name not in ('React', 'ReactDOM', 'Component'):
            candidates.append(name)

    fallback_checks = " : ".join(
        f"typeof {n} !== 'undefined' ? {n}" for n in candidates
    ) + " : null"

    # Escape </script> in JSX source to prevent premature tag closing
    safe_source = cleaned_source.replace('</script>', '<\\/script>')

    reload_script = ""
    if ws_port:
        reload_script = f"""
    <script>
      (function() {{
        var ws;
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
    <script src="{REACT_CDN}"></script>
    <script src="{REACT_DOM_CDN}"></script>
    <script src="{TAILWIND_CDN}"></script>
    <script src="{BABEL_CDN}"></script>
{lib_scripts}    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        #root {{ min-height: 100vh; }}
    </style>{reload_script}
</head>
<body>
    <div id="root"></div>
    <script type="text/babel">
{safe_source}

// ─── Mount ───
const _component = {fallback_checks};

if (_component) {{
  const root = ReactDOM.createRoot(document.getElementById('root'));
  root.render(React.createElement(_component));
}} else {{
  document.getElementById('root').innerHTML =
    '<div style="padding:40px;color:#888;font-family:monospace;">' +
    'No component found. Export a default or define a PascalCase function.' +
    '</div>';
}}
    </script>
</body>
</html>"""


# ─── File Watcher (no dependencies) ───

class FileWatcher:
    """Watch files for changes using polling."""

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
                conn, _ = self._server.accept()
                t = threading.Thread(target=self._handshake, args=(conn,), daemon=True)
                t.start()
            except socket.timeout:
                continue
            except OSError:
                break

    def _handshake(self, conn: socket.socket):
        try:
            data = conn.recv(4096).decode("utf-8", errors="replace")
            if "Upgrade: websocket" not in data:
                conn.close()
                return

            key = ""
            for line in data.split("\r\n"):
                if line.startswith("Sec-WebSocket-Key:"):
                    key = line.split(": ", 1)[1].strip()
                    break

            if not key:
                conn.close()
                return

            import base64

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
        msg = b"reload"
        frame = bytearray()
        frame.append(0x81)
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
    jsx_files: dict[str, Path] = {}
    base_dir: Path = Path(".")
    ws_port: int | None = None

    def do_GET(self):
        path = unquote(self.path.lstrip("/"))

        if not path or path == "/":
            if len(self.jsx_files) == 1:
                name = list(self.jsx_files.keys())[0]
                self._serve_jsx(self.jsx_files[name])
            else:
                self._serve_index()
            return

        route = path.removesuffix(".html")
        if route in self.jsx_files:
            self._serve_jsx(self.jsx_files[route])
            return

        if route.removesuffix(".jsx") in self.jsx_files:
            self._serve_jsx(self.jsx_files[route.removesuffix(".jsx")])
            return

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
    parser.add_argument("file", nargs="?", help="JSX file to serve")
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dir", "-d", default=".")
    parser.add_argument("--no-reload", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")

    args = parser.parse_args()
    base_dir = Path(args.dir).resolve()

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

    JSXHandler.jsx_files = jsx_files
    JSXHandler.base_dir = base_dir
    JSXHandler.ws_port = ws_port

    server = http.server.HTTPServer((args.host, args.port), JSXHandler)

    print(f"\n  jsxserve")
    print(f"  {'─' * 40}")
    for name in jsx_files:
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
