"""
Preview Server - HTTP Server for Live Preview of Generated Apps

Features:
- Serves static files from project workspace
- Hot-reload support via WebSocket
- React/Vue app support with ESM modules
- CORS enabled for iframe embedding
- Auto-injects necessary scripts for frameworks
- FIXED: Better file detection, consistent preview behavior
"""

import asyncio
import json
import mimetypes
import os
import re
import socket
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional, Dict, Callable, List
from urllib.parse import urlparse, parse_qs
import hashlib


class SilentHTTPServer(HTTPServer):
    """HTTPServer that silently handles client disconnection errors"""

    def handle_error(self, request, client_address):
        """Silently ignore connection errors (common on Windows)"""
        import sys
        exc_type = sys.exc_info()[0]
        # Silently ignore connection-related errors
        if exc_type in (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            return
        # Also ignore OSError with specific Windows error codes
        if exc_type == OSError:
            return


class PreviewRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for preview server"""

    # Class-level config (set by PreviewServer)
    workspace_path: Path = Path(".")
    file_overrides: Dict[str, str] = {}
    on_request: Optional[Callable] = None

    def __init__(self, *args, **kwargs):
        # Set directory before calling parent __init__
        super().__init__(*args, directory=str(self.workspace_path), **kwargs)

    def log_message(self, format, *args):
        """Suppress logging or redirect to callback"""
        if self.on_request:
            self.on_request(f"{args[0]} {args[1]}")

    def end_headers(self):
        """Add CORS and cache headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        """Handle GET requests with special processing"""
        try:
            self._handle_get()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            # Client disconnected - ignore silently (common on Windows)
            pass
        except Exception as e:
            # Log but don't crash
            if self.on_request:
                self.on_request(f"Error: {e}")

    def _handle_get(self):
        """Internal GET handler - FIXED: subdirectory support, HTML injection, cache control"""
        parsed = urlparse(self.path)
        path = parsed.path.lstrip('/')

        # Strip query params for file lookup (cache-busting ?_t=xxx)
        # but keep path clean
        path = path.split('?')[0] if '?' in path else path

        # Check for file overrides (in-memory files)
        if path in self.file_overrides:
            content = self.file_overrides[path]
            self._serve_content(content, path)
            return

        # Special endpoint for hot-reload status
        if path == '__preview__/status':
            self._serve_json({"status": "running", "timestamp": time.time()})
            return

        # Special endpoint for file list
        if path == '__preview__/files':
            files = self._list_files_with_content()
            self._serve_json({"files": files})
            return

        # Root request → smart index resolution
        if path == '' or path == 'index.html':
            self._serve_index()
            return

        # Directory request (e.g. /minecraft-calculator/) → look for index.html inside
        full_path = self.workspace_path / path
        if full_path.is_dir():
            sub_index = full_path / 'index.html'
            if sub_index.exists():
                self._serve_html_file(sub_index)
            else:
                # Show file browser scoped to this subdirectory
                content = self._generate_file_browser(sub_dir=path)
                content = self._inject_preview_scripts(content)
                self._serve_content(content, 'index.html')
            return

        # ANY .html/.htm file → serve with hot-reload injection
        if path.endswith('.html') or path.endswith('.htm'):
            html_path = self.workspace_path / path
            if html_path.exists():
                self._serve_html_file(html_path)
                return

        # Handle JSX/TSX files - transform to JS
        if path.endswith('.jsx') or path.endswith('.tsx'):
            self._serve_transformed_react(path)
            return

        # Handle Vue SFC files
        if path.endswith('.vue'):
            self._serve_transformed_vue(path)
            return

        # Default file serving (CSS, JS, images, etc.)
        super().do_GET()

    def _serve_html_file(self, file_path: Path):
        """Serve any HTML file with hot-reload and framework scripts injected"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='replace')
        except Exception:
            content = file_path.read_text(encoding='latin-1', errors='replace')

        content = self._inject_preview_scripts(content)
        self._serve_content(content, file_path.name)

    def _serve_content(self, content: str, filename: str):
        """Serve string content with appropriate mime type"""
        try:
            content_bytes = content.encode('utf-8')
            mime_type = self._get_mime_type(filename)

            self.send_response(200)
            self.send_header('Content-Type', f'{mime_type}; charset=utf-8')
            self.send_header('Content-Length', len(content_bytes))
            self.end_headers()
            self.wfile.write(content_bytes)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            # Client disconnected - ignore silently
            pass
        except Exception:
            pass

    def _serve_json(self, data: dict):
        """Serve JSON response"""
        content = json.dumps(data)
        self._serve_content(content, 'response.json')

    def _get_mime_type(self, filename: str) -> str:
        """Get MIME type for file"""
        ext = Path(filename).suffix.lower()
        mime_map = {
            '.html': 'text/html',
            '.css': 'text/css',
            '.js': 'application/javascript',
            '.mjs': 'application/javascript',
            '.jsx': 'application/javascript',
            '.ts': 'application/javascript',
            '.tsx': 'application/javascript',
            '.json': 'application/json',
            '.svg': 'image/svg+xml',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.woff': 'font/woff',
            '.woff2': 'font/woff2',
            '.ttf': 'font/ttf',
        }
        return mime_map.get(ext, 'text/plain')

    def _list_files_with_content(self) -> list:
        """List all files in workspace with size info - FIXED for better detection"""
        files = []

        # First, check if we have any web-related files
        web_extensions = {'.html', '.htm', '.css', '.js', '.jsx', '.ts', '.tsx', '.vue', '.json'}

        try:
            for path in self.workspace_path.rglob('*'):
                if path.is_file():
                    # Skip common non-web directories
                    parts = path.parts
                    if any(skip in str(path) for skip in ['node_modules', '.git', '__pycache__',
                                                          '.pytest_cache', '.venv', 'venv']):
                        continue

                    rel_path = str(path.relative_to(self.workspace_path))
                    stat = path.stat()
                    files.append({
                        "path": rel_path,
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "is_web": path.suffix.lower() in web_extensions
                    })
        except Exception:
            pass

        # Sort: web files first, then by name
        files.sort(key=lambda f: (not f['is_web'], f['path'].lower()))
        return files

    def _serve_index(self):
        """Serve index.html with smart resolution:
        1. Root index.html exists → serve it
        2. No root index but subfolder has index.html → redirect there
        3. Web content exists but no index.html → file browser
        4. No content at all → welcome page
        """
        root_index = self.workspace_path / 'index.html'

        # 1. Root index exists
        if root_index.exists():
            self._serve_html_file(root_index)
            return

        # 2. Search subdirectories for index.html
        found_index = self._find_best_index()
        if found_index:
            # Redirect to the subfolder's index
            rel_path = str(found_index.relative_to(self.workspace_path))
            # Use 302 redirect so browser navigates there
            self.send_response(302)
            self.send_header('Location', f'/{rel_path}')
            self.end_headers()
            return

        # 3. Has web content → file browser
        if self._has_web_content():
            content = self._generate_file_browser()
            content = self._inject_preview_scripts(content)
            self._serve_content(content, 'index.html')
            return

        # 4. No content → welcome page
        content = self._generate_welcome_page()
        content = self._inject_preview_scripts(content)
        self._serve_content(content, 'index.html')

    def _find_best_index(self) -> Optional[Path]:
        """Find the best index.html in subdirectories (closest to root first)"""
        try:
            candidates = []
            for path in self.workspace_path.rglob('index.html'):
                if path.is_file():
                    if any(skip in str(path) for skip in ['node_modules', '.git', '__pycache__', '.venv']):
                        continue
                    # Score by depth (fewer parts = closer to root = better)
                    depth = len(path.relative_to(self.workspace_path).parts)
                    candidates.append((depth, path))

            if candidates:
                candidates.sort(key=lambda x: x[0])
                return candidates[0][1]
        except Exception:
            pass
        return None

    def _has_web_content(self) -> bool:
        """Check if workspace has any web-related content"""
        web_extensions = {'.html', '.htm', '.css', '.js', '.jsx', '.ts', '.tsx', '.vue'}

        try:
            for path in self.workspace_path.rglob('*'):
                if path.is_file() and path.suffix.lower() in web_extensions:
                    # Skip node_modules and similar
                    if any(skip in str(path) for skip in ['node_modules', '.git']):
                        continue
                    return True
        except Exception:
            pass

        return False

    def _generate_welcome_page(self) -> str:
        """Generate a welcome page when no content exists"""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Workspace</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e2e8f0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }}
        .container {{
            max-width: 600px;
            text-align: center;
        }}
        h1 {{
            font-size: 2rem;
            margin-bottom: 1rem;
            color: #818cf8;
        }}
        .status {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 2rem;
            margin-top: 2rem;
        }}
        .status-icon {{
            font-size: 48px;
            margin-bottom: 1rem;
            opacity: 0.5;
        }}
        .info {{
            color: #94a3b8;
            margin-top: 1rem;
            line-height: 1.6;
        }}
        .path {{
            font-family: monospace;
            background: rgba(0,0,0,0.3);
            padding: 0.5rem 1rem;
            border-radius: 6px;
            margin-top: 1rem;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Project Workspace</h1>
        <div class="status">
            <div class="status-icon">📁</div>
            <p><strong>Workspace is ready</strong></p>
            <p class="info">
                Generate HTML, CSS, JavaScript, or other web files<br>
                to see them appear in the preview.
            </p>
            <p class="path">{self.workspace_path}</p>
        </div>
    </div>
</body>
</html>'''

    def _generate_file_browser(self, sub_dir: str = None) -> str:
        """Generate an interactive file browser.
        - Shows ALL files (web files highlighted)
        - Clicking HTML files opens them in-place
        - Subdirectory navigation support
        - Back button for navigation
        """
        base_path = self.workspace_path
        if sub_dir:
            base_path = self.workspace_path / sub_dir

        files = []
        dirs = set()

        try:
            for item in sorted(base_path.iterdir()):
                # Skip hidden and common non-project dirs
                if item.name.startswith('.') or item.name in ('node_modules', '__pycache__', '.venv', 'venv', '.git'):
                    continue

                if item.is_dir():
                    # Count files in dir
                    count = sum(1 for _ in item.rglob('*') if _.is_file())
                    dirs.add((item.name, count))
                elif item.is_file():
                    rel_path = str(item.relative_to(self.workspace_path))
                    files.append({
                        "name": item.name,
                        "path": rel_path,
                        "size": item.stat().st_size,
                        "ext": item.suffix.lower(),
                    })
        except Exception:
            pass

        # Build directory items
        dir_items = ""
        for dname, count in sorted(dirs):
            dir_path = f"{sub_dir}/{dname}" if sub_dir else dname
            dir_items += f'''
                <li class="file-item dir" onclick="window.location.href='/{dir_path}/'">
                    <span class="icon">📁</span>
                    <span class="name">{dname}/</span>
                    <span class="size">{count} files</span>
                </li>
            '''

        # Build file items
        viewable_exts = {'.html', '.htm', '.svg', '.json', '.txt', '.md'}
        file_items = ""
        for f in files:
            icon = self._get_file_icon(f['name'])
            size_kb = f['size'] / 1024

            # HTML/viewable files → navigate in browser
            if f['ext'] in viewable_exts:
                file_items += f'''
                    <li class="file-item viewable" onclick="window.location.href='/{f['path']}'">
                        <span class="icon">{icon}</span>
                        <span class="name">{f['name']}</span>
                        <span class="badge">view</span>
                        <span class="size">{size_kb:.1f} KB</span>
                    </li>
                '''
            else:
                file_items += f'''
                    <li class="file-item">
                        <span class="icon">{icon}</span>
                        <span class="name">{f['name']}</span>
                        <span class="size">{size_kb:.1f} KB</span>
                    </li>
                '''

        # Back button
        back_btn = ""
        if sub_dir:
            parent = str(Path(sub_dir).parent)
            parent_url = f"/{parent}/" if parent != '.' else '/'
            back_btn = f'''
                <li class="file-item dir" onclick="window.location.href='{parent_url}'" style="border-bottom: 2px solid rgba(255,255,255,0.1);">
                    <span class="icon">⬆️</span>
                    <span class="name">..</span>
                    <span class="size">parent</span>
                </li>
            '''

        display_path = sub_dir or "/"
        total_items = len(dirs) + len(files)

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Files - {display_path}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e2e8f0;
            min-height: 100vh;
            padding: 1.5rem;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        .header {{
            display: flex; align-items: center; gap: 1rem;
            margin-bottom: 1.5rem; flex-wrap: wrap;
        }}
        h1 {{ font-size: 1.3rem; color: #818cf8; }}
        .breadcrumb {{
            font-family: monospace; font-size: 0.85rem; color: #94a3b8;
            background: rgba(0,0,0,0.3); padding: 0.3rem 0.8rem; border-radius: 4px;
        }}
        .count {{ color: #64748b; font-size: 0.85rem; }}
        .file-list {{ list-style: none; background: rgba(255,255,255,0.03); border-radius: 8px; overflow: hidden; }}
        .file-item {{
            display: flex; align-items: center; gap: 0.75rem;
            padding: 0.6rem 1rem; border-bottom: 1px solid rgba(255,255,255,0.05);
            transition: all 0.15s; font-size: 0.9rem;
        }}
        .file-item.viewable, .file-item.dir {{ cursor: pointer; }}
        .file-item.viewable:hover {{ background: rgba(129, 140, 248, 0.15); }}
        .file-item.dir:hover {{ background: rgba(251, 191, 36, 0.1); }}
        .file-item:last-child {{ border-bottom: none; }}
        .icon {{ font-size: 1.1rem; flex-shrink: 0; }}
        .name {{ flex: 1; font-family: monospace; font-size: 0.85rem; }}
        .size {{ color: #64748b; font-size: 0.75rem; flex-shrink: 0; }}
        .badge {{
            font-size: 0.65rem; padding: 1px 6px; border-radius: 3px;
            background: rgba(129, 140, 248, 0.2); color: #a5b4fc;
            text-transform: uppercase; letter-spacing: 0.5px; flex-shrink: 0;
        }}
        .no-files {{
            text-align: center; padding: 3rem; color: #64748b;
            background: rgba(255,255,255,0.03); border-radius: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📁 Project Files</h1>
            <span class="breadcrumb">{display_path}</span>
            <span class="count">{total_items} item(s)</span>
        </div>
        {f'<ul class="file-list">{back_btn}{dir_items}{file_items}</ul>' if total_items > 0 else '<div class="no-files">No files here yet.<br><small>Generate files to see them appear.</small></div>'}
    </div>
</body>
</html>'''

    def _get_file_icon(self, filename: str) -> str:
        """Get emoji icon for file type"""
        ext = Path(filename).suffix.lower()
        icons = {
            '.py': '🐍', '.js': '📜', '.jsx': '⚛️', '.ts': '📘', '.tsx': '⚛️',
            '.html': '🌐', '.css': '🎨', '.json': '📋', '.vue': '💚',
            '.md': '📝', '.txt': '📄', '.svg': '🎭',
        }
        return icons.get(ext, '📄')

    def _inject_preview_scripts(self, html: str) -> str:
        """Inject preview support scripts into HTML"""

        # Hot reload script
        hot_reload_script = '''
<script>
// Preview Hot Reload
(function() {
    let lastCheck = Date.now();
    const checkInterval = 1000;

    async function checkForUpdates() {
        try {
            const res = await fetch('/__preview__/status');
            const data = await res.json();
            if (data.timestamp > lastCheck + 5000) {
                console.log('[Preview] Changes detected, reloading...');
                location.reload();
            }
            lastCheck = Date.now();
        } catch (e) {
            console.log('[Preview] Server disconnected');
        }
    }

    setInterval(checkForUpdates, checkInterval);
    console.log('[Preview] Hot reload enabled');
})();
</script>
'''

        # React support (via esm.sh CDN)
        react_support = '''
<script type="importmap">
{
    "imports": {
        "react": "https://esm.sh/react@18",
        "react-dom": "https://esm.sh/react-dom@18",
        "react-dom/client": "https://esm.sh/react-dom@18/client",
        "react/jsx-runtime": "https://esm.sh/react@18/jsx-runtime"
    }
}
</script>
'''

        # Vue support
        vue_support = '''
<script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
'''

        # Detect what frameworks are needed
        needs_react = '.jsx' in html or '.tsx' in html or 'react' in html.lower()
        needs_vue = '.vue' in html or 'vue' in html.lower()

        inject = hot_reload_script
        if needs_react:
            inject = react_support + inject
        if needs_vue:
            inject = vue_support + inject

        # Inject before </head> or at start of <body>
        if '</head>' in html:
            html = html.replace('</head>', inject + '</head>')
        elif '<body>' in html:
            html = html.replace('<body>', '<body>' + inject)
        else:
            html = inject + html

        return html

    def _serve_transformed_react(self, path: str):
        """Transform and serve JSX/TSX files"""
        file_path = self.workspace_path / path

        if not file_path.exists():
            self.send_error(404, f"File not found: {path}")
            return

        try:
            content = file_path.read_text(encoding='utf-8', errors='replace')
        except:
            content = file_path.read_text(encoding='latin-1', errors='replace')

        # Wrap as ES module
        wrapped = f'''
// Transformed from {path}
import React from 'react';
import ReactDOM from 'react-dom/client';

{content}

// Auto-mount if default export is a component
if (typeof App !== 'undefined') {{
    const root = document.getElementById('root') || document.getElementById('app');
    if (root) {{
        ReactDOM.createRoot(root).render(React.createElement(App));
    }}
}}
'''

        self._serve_content(wrapped, path.replace('.jsx', '.js').replace('.tsx', '.js'))

    def _serve_transformed_vue(self, path: str):
        """Transform and serve Vue SFC files"""
        file_path = self.workspace_path / path

        if not file_path.exists():
            self.send_error(404, f"File not found: {path}")
            return

        try:
            content = file_path.read_text(encoding='utf-8', errors='replace')
        except:
            content = file_path.read_text(encoding='latin-1', errors='replace')

        # Parse Vue SFC (simplified)
        template_match = re.search(r'<template>(.*?)</template>', content, re.DOTALL)
        script_match = re.search(r'<script[^>]*>(.*?)</script>', content, re.DOTALL)
        style_match = re.search(r'<style[^>]*>(.*?)</style>', content, re.DOTALL)

        template = template_match.group(1).strip() if template_match else ''
        script = script_match.group(1).strip() if script_match else ''
        style = style_match.group(1).strip() if style_match else ''

        # Generate runtime component
        component_name = Path(path).stem

        transformed = f'''
// Transformed Vue SFC: {path}
const {component_name} = {{
    template: `{template}`,
    {script.replace('export default', '...(')} or {{}}
}};

// Inject styles
if (document.getElementById('vue-style-{component_name}') === null) {{
    const style = document.createElement('style');
    style.id = 'vue-style-{component_name}';
    style.textContent = `{style}`;
    document.head.appendChild(style);
}}

// Auto-mount
if (typeof Vue !== 'undefined') {{
    const app = Vue.createApp({component_name});
    const mountPoint = document.getElementById('app') || document.body;
    app.mount(mountPoint);
}}

export default {component_name};
'''

        self._serve_content(transformed, path.replace('.vue', '.js'))


class PreviewServer:
    """
    Preview server manager for serving generated apps.

    Usage:
        server = PreviewServer(workspace_path="/path/to/project")
        server.start()
        print(f"Preview at: {server.url}")

        # Update files in memory (without disk write)
        server.update_file("app.js", "console.log('updated')")

        server.stop()
    """

    def __init__(
        self,
        workspace_path: str,
        port: int = 0,  # 0 = auto-select
        host: str = "127.0.0.1",
        on_request: Optional[Callable[[str], None]] = None
    ):
        self.workspace_path = Path(workspace_path)
        self.host = host
        self.port = port or self._find_free_port()
        self.on_request = on_request

        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._file_overrides: Dict[str, str] = {}
        self._running = False

    def _find_free_port(self) -> int:
        """Find an available port"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    @property
    def url(self) -> str:
        """Get the server URL"""
        return f"http://{self.host}:{self.port}"

    @property
    def is_running(self) -> bool:
        """Check if server is running"""
        return self._running

    def start(self):
        """Start the preview server"""
        if self._running:
            return

        # FIX: Create a unique handler subclass for THIS server instance
        # This prevents multiple servers from overwriting each other's workspace_path
        ws_path = self.workspace_path
        overrides = self._file_overrides
        on_req = self.on_request

        class BoundHandler(PreviewRequestHandler):
            workspace_path = ws_path
            file_overrides = overrides
            on_request = on_req

        # Create and start server with the bound handler
        self._server = SilentHTTPServer((self.host, self.port), BoundHandler)
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        self._running = True
        self._bound_handler = BoundHandler  # Keep reference for update_file

        # Wait a moment for server to start
        time.sleep(0.1)

    def _run_server(self):
        """Server thread function"""
        try:
            self._server.serve_forever()
        except Exception:
            pass  # Silently ignore server errors
        finally:
            self._running = False

    def stop(self):
        """Stop the preview server"""
        if self._server:
            try:
                self._server.shutdown()
            except Exception:
                pass
            try:
                self._server.server_close()
            except Exception:
                pass
            self._server = None
        self._running = False

    def update_file(self, path: str, content: str):
        """Update a file in memory (instant preview update)"""
        self._file_overrides[path] = content
        # Update the bound handler's overrides (not the global class)
        if hasattr(self, '_bound_handler'):
            self._bound_handler.file_overrides = self._file_overrides

    def remove_file_override(self, path: str):
        """Remove in-memory file override (use disk version)"""
        if path in self._file_overrides:
            del self._file_overrides[path]

    def clear_overrides(self):
        """Clear all in-memory file overrides"""
        self._file_overrides.clear()

    def get_preview_html(self, height: int = 600) -> str:
        """Get HTML for embedding preview in iframe"""
        return f'''
<iframe
    src="{self.url}"
    width="100%"
    height="{height}px"
    style="border: none; border-radius: 8px; background: white;"
    sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-modals"
    allow="clipboard-read; clipboard-write"
></iframe>
'''


# Singleton manager for multiple project servers
class PreviewServerManager:
    """Manages multiple preview servers for different projects"""

    _servers: Dict[str, PreviewServer] = {}
    _port_base = 8600

    @classmethod
    def get_server(cls, project_id: str, workspace_path: str) -> PreviewServer:
        """Get or create a preview server for a project"""
        if project_id not in cls._servers:
            port = cls._port_base + len(cls._servers)
            server = PreviewServer(
                workspace_path=workspace_path,
                port=port
            )
            server.start()
            cls._servers[project_id] = server
        return cls._servers[project_id]

    @classmethod
    def stop_server(cls, project_id: str):
        """Stop a specific project's server"""
        if project_id in cls._servers:
            cls._servers[project_id].stop()
            del cls._servers[project_id]

    @classmethod
    def stop_all(cls):
        """Stop all preview servers"""
        for server in cls._servers.values():
            server.stop()
        cls._servers.clear()


# Convenience function
def create_preview_server(workspace_path: str, port: int = 0) -> PreviewServer:
    """Create a new preview server"""
    server = PreviewServer(workspace_path=workspace_path, port=port)
    server.start()
    return server


if __name__ == "__main__":
    # Test server
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        Path(tmpdir, "index.html").write_text("""
<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
    <div id="app"></div>
    <script type="module" src="app.jsx"></script>
</body>
</html>
""")

        Path(tmpdir, "app.jsx").write_text("""
function App() {
    const [count, setCount] = React.useState(0);
    return (
        <div>
            <h1>React Counter</h1>
            <p>Count: {count}</p>
            <button onClick={() => setCount(c => c + 1)}>Increment</button>
        </div>
    );
}
""")

        server = create_preview_server(tmpdir)
        print(f"Preview server running at: {server.url}")
        print("Press Ctrl+C to stop")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            server.stop()
            print("\nServer stopped")
