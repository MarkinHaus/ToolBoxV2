"""
WebAppDisplay - Local development demo for web app display

Provides a simple iframe-based display for Docker-hosted web apps.
This is a minimal implementation for local development.

For production (simplecor.app), session-based routing will be added.

Author: FlowAgent V2
"""
from __future__ import annotations

import asyncio
import secrets
import socket
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.docker_vfs import DockerVFS


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class WebDisplayConfig:
    """Configuration for web app display"""
    host: str = "localhost"
    proxy_port_start: int = 9000
    proxy_port_end: int = 9100
    session_timeout_minutes: int = 60
    max_sessions: int = 10
    enable_auth: bool = False  # For local dev, no auth needed


@dataclass
class DisplaySession:
    """A display session for accessing a web app"""
    session_id: str
    token: str
    container_url: str
    proxy_port: int
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    active: bool = True
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


# =============================================================================
# WEB APP DISPLAY
# =============================================================================

class WebAppDisplay:
    """
    Local development display for Docker-hosted web apps.
    
    Features:
    - Simple port forwarding for local access
    - Session token generation
    - HTML iframe embed code generation
    
    For production use with simplecor.app:
    - Add session-based routing through nginx/traefik
    - Implement proper authentication
    - Use secure tokens with expiration
    """
    
    def __init__(
        self,
        config: WebDisplayConfig | None = None
    ):
        """
        Initialize WebAppDisplay.
        
        Args:
            config: Display configuration
        """
        self.config = config or WebDisplayConfig()
        
        # Active sessions
        self._sessions: dict[str, DisplaySession] = {}
        
        # Port allocation
        self._used_ports: set[int] = set()
    
    # =========================================================================
    # PORT MANAGEMENT
    # =========================================================================
    
    def _find_free_port(self) -> int | None:
        """Find a free port in the configured range"""
        for port in range(self.config.proxy_port_start, self.config.proxy_port_end):
            if port in self._used_ports:
                continue
            
            # Check if port is actually available
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((self.config.host, port))
                    self._used_ports.add(port)
                    return port
                except OSError:
                    continue
        
        return None
    
    def _release_port(self, port: int):
        """Release a port"""
        self._used_ports.discard(port)
    
    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================
    
    def _generate_token(self) -> str:
        """Generate a secure session token"""
        return secrets.token_urlsafe(32)
    
    def _generate_session_id(self) -> str:
        """Generate a session ID"""
        return secrets.token_hex(8)
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        expired = [
            sid for sid, session in self._sessions.items()
            if session.is_expired()
        ]
        for sid in expired:
            self._release_port(self._sessions[sid].proxy_port)
            del self._sessions[sid]
    
    # =========================================================================
    # DISPLAY METHODS
    # =========================================================================
    
    def create_session(
        self,
        container_url: str,
        timeout_minutes: int | None = None
    ) -> dict:
        """
        Create a display session for a container app.
        
        Args:
            container_url: URL of the app inside the container (e.g., http://localhost:8080)
            timeout_minutes: Session timeout (default from config)
            
        Returns:
            Session info with access URL
        """
        self._cleanup_expired_sessions()
        
        # Check max sessions
        if len(self._sessions) >= self.config.max_sessions:
            return {"success": False, "error": "Maximum sessions reached"}
        
        # Allocate port
        proxy_port = self._find_free_port()
        if proxy_port is None:
            return {"success": False, "error": "No available ports"}
        
        # Create session
        session_id = self._generate_session_id()
        token = self._generate_token()
        
        timeout = timeout_minutes or self.config.session_timeout_minutes
        expires_at = datetime.now() + timedelta(minutes=timeout)
        
        session = DisplaySession(
            session_id=session_id,
            token=token,
            container_url=container_url,
            proxy_port=proxy_port,
            expires_at=expires_at
        )
        
        self._sessions[session_id] = session
        
        access_url = f"http://{self.config.host}:{proxy_port}"
        
        return {
            "success": True,
            "session_id": session_id,
            "token": token,
            "access_url": access_url,
            "container_url": container_url,
            "expires_at": expires_at.isoformat(),
            "iframe_html": self._generate_iframe_html(access_url, session_id)
        }
    
    def get_session(self, session_id: str) -> DisplaySession | None:
        """Get a session by ID"""
        session = self._sessions.get(session_id)
        if session and not session.is_expired():
            return session
        return None
    
    def close_session(self, session_id: str) -> dict:
        """Close a display session"""
        if session_id not in self._sessions:
            return {"success": False, "error": "Session not found"}
        
        session = self._sessions[session_id]
        session.active = False
        self._release_port(session.proxy_port)
        del self._sessions[session_id]
        
        return {"success": True, "message": f"Session {session_id} closed"}
    
    def list_sessions(self) -> list[dict]:
        """List all active sessions"""
        self._cleanup_expired_sessions()
        
        return [
            {
                "session_id": s.session_id,
                "container_url": s.container_url,
                "proxy_port": s.proxy_port,
                "access_url": f"http://{self.config.host}:{s.proxy_port}",
                "created_at": s.created_at.isoformat(),
                "expires_at": s.expires_at.isoformat() if s.expires_at else None,
                "active": s.active
            }
            for s in self._sessions.values()
        ]
    
    # =========================================================================
    # HTML GENERATION
    # =========================================================================
    
    def _generate_iframe_html(
        self,
        url: str,
        session_id: str,
        width: str = "100%",
        height: str = "600px"
    ) -> str:
        """Generate HTML iframe embed code"""
        return f'''<iframe 
    src="{url}" 
    id="vfs-app-{session_id}"
    width="{width}" 
    height="{height}" 
    frameborder="0"
    sandbox="allow-scripts allow-forms allow-same-origin"
    style="border: 1px solid #ccc; border-radius: 4px;"
></iframe>'''
    
    def generate_full_html_page(
        self,
        session_id: str,
        title: str = "VFS Web App"
    ) -> str | None:
        """Generate a full HTML page with the iframe"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        url = f"http://{self.config.host}:{session.proxy_port}"
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            min-height: 100vh;
        }}
        .header {{
            background: #1a1a2e;
            color: white;
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{
            font-size: 18px;
            font-weight: 500;
        }}
        .status {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .status-dot {{
            width: 8px;
            height: 8px;
            background: #4caf50;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        .container {{
            padding: 24px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .info-bar {{
            background: white;
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .info-bar code {{
            background: #f0f0f0;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 13px;
        }}
        .iframe-container {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        iframe {{
            display: block;
            width: 100%;
            height: calc(100vh - 200px);
            min-height: 500px;
            border: none;
        }}
        .footer {{
            text-align: center;
            padding: 16px;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🐳 {title}</h1>
        <div class="status">
            <div class="status-dot"></div>
            <span>Session: {session_id[:8]}...</span>
        </div>
    </div>
    
    <div class="container">
        <div class="info-bar">
            <span>Container URL: <code>{session.container_url}</code></span>
            <span>Access URL: <code>{url}</code></span>
        </div>
        
        <div class="iframe-container">
            <iframe src="{url}" id="app-frame"></iframe>
        </div>
    </div>
    
    <div class="footer">
        VFS Docker Display • Session expires: {session.expires_at.strftime('%Y-%m-%d %H:%M:%S') if session.expires_at else 'Never'}
    </div>
    
    <script>
        // Auto-refresh on connection loss
        const iframe = document.getElementById('app-frame');
        let retries = 0;
        const maxRetries = 5;
        
        iframe.onerror = function() {{
            if (retries < maxRetries) {{
                retries++;
                setTimeout(() => {{
                    iframe.src = iframe.src;
                }}, 2000);
            }}
        }};
    </script>
</body>
</html>'''
    
    # =========================================================================
    # LOCAL DEV SERVER (SIMPLE PROXY)
    # =========================================================================
    
    async def start_simple_proxy(self, session_id: str) -> dict:
        """
        Start a simple HTTP proxy for local development.
        
        This is a basic implementation for local testing.
        For production, use nginx/traefik with proper routing.
        
        Note: This requires aiohttp. Falls back to direct URL if not available.
        """
        session = self.get_session(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}
        
        try:
            from aiohttp import web, ClientSession
            
            async def proxy_handler(request: web.Request) -> web.Response:
                """Proxy requests to container"""
                path = request.path
                query = request.query_string
                
                target_url = session.container_url.rstrip('/') + path
                if query:
                    target_url += f"?{query}"
                
                async with ClientSession() as client:
                    async with client.request(
                        method=request.method,
                        url=target_url,
                        headers=dict(request.headers),
                        data=await request.read()
                    ) as resp:
                        body = await resp.read()
                        return web.Response(
                            body=body,
                            status=resp.status,
                            headers=dict(resp.headers)
                        )
            
            app = web.Application()
            app.router.add_route('*', '/{path:.*}', proxy_handler)
            
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, self.config.host, session.proxy_port)
            await site.start()
            
            return {
                "success": True,
                "message": f"Proxy started on port {session.proxy_port}",
                "url": f"http://{self.config.host}:{session.proxy_port}"
            }
            
        except ImportError:
            # aiohttp not available, return direct URL
            return {
                "success": True,
                "message": "Direct access (no proxy)",
                "url": session.container_url,
                "note": "Install aiohttp for proxy support"
            }
    
    # =========================================================================
    # FUTURE: SIMPLECOR.APP INTEGRATION
    # =========================================================================
    
    def generate_simplecor_config(self, session_id: str) -> dict | None:
        """
        Generate configuration for SimpleCor.app integration.
        
        This is a placeholder for future production deployment.
        The actual implementation will depend on SimpleCor's routing setup.
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        return {
            "session_id": session_id,
            "token": session.token,
            "container_url": session.container_url,
            "routing_rules": {
                # Example routing configuration for nginx/traefik
                "path_prefix": f"/app/{session_id}",
                "upstream": session.container_url,
                "auth_required": self.config.enable_auth,
                "auth_token": session.token if self.config.enable_auth else None
            },
            "iframe_url": f"https://simplecor.app/embed/{session_id}",
            "note": "This is a configuration template for SimpleCor integration"
        }
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    
    def cleanup(self):
        """Clean up all sessions and release ports"""
        for session in self._sessions.values():
            self._release_port(session.proxy_port)
        self._sessions.clear()


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

async def create_web_display_for_docker(
    docker_vfs: 'DockerVFS',
    entrypoint: str,
    title: str = "VFS Web App"
) -> dict:
    """
    Helper function to start a web app and create a display session.
    
    Args:
        docker_vfs: DockerVFS instance
        entrypoint: Command to start the web app
        title: Display title
        
    Returns:
        Dict with display info and HTML page
    """
    # Start web app in container
    app_result = await docker_vfs.start_web_app(entrypoint)
    
    if not app_result.get("success"):
        return app_result
    
    # Create display session
    display = WebAppDisplay()
    
    container_url = app_result.get("url") or f"http://localhost:{app_result.get('host_port', 8080)}"
    session_result = display.create_session(container_url)
    
    if not session_result["success"]:
        return session_result
    
    # Generate full HTML page
    html_page = display.generate_full_html_page(session_result["session_id"], title)
    
    return {
        **session_result,
        "html_page": html_page,
        "app_info": app_result
    }
