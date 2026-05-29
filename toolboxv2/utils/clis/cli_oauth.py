# file: toolboxv2/utils/clis/cli_oauth.py
"""
CLI local-callback auth: Discord/Google OAuth + WebAuthn passkey (register+login).

A tiny loopback HTTP server on 127.0.0.1 receives the OAuth callback or runs the
WebAuthn ceremony; the code/credential is exchanged IN-PROCESS against
CloudM.Auth and the resulting payload is handed back to the caller. No tb worker
/ nginx required — the CLI owns the receiver for its own port.

Passkey note: rp_id is pinned to "localhost" and origin to the page port, so a
passkey REGISTERED via the CLI is the one USABLE via the CLI (WebAuthn scopes
credentials to rp_id). Web-registered passkeys (rp_id=<domain>) are separate.
"""

import asyncio
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlparse, parse_qs

from toolboxv2 import get_app
from toolboxv2.mods.CloudM.auth.config import get_passkey_config

_HOST = "127.0.0.1"
_DONE_PAGE = (
    b"<!DOCTYPE html><html><body style='font-family:sans-serif;background:#1a1a2e;"
    b"color:#fff;text-align:center;padding-top:18vh'><h2>Login complete</h2>"
    b"<p>You can close this tab and return to the CLI.</p></body></html>"
)


# =============================================================================
# OAuth (Discord / Google)
# =============================================================================

class _OAuthCapture:
    def __init__(self):
        self.code: Optional[str] = None
        self.state: Optional[str] = None
        self.error: Optional[str] = None
        self.done = threading.Event()


def _oauth_handler(provider: str, cap: _OAuthCapture):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            if u.path != f"/auth/{provider}/callback":
                self.send_response(404); self.end_headers(); return
            q = parse_qs(u.query)
            cap.code = (q.get("code") or [None])[0]
            cap.state = (q.get("state") or [None])[0]
            cap.error = (q.get("error") or [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_DONE_PAGE)
            cap.done.set()
    return H


async def cli_oauth_login(provider: str, timeout: float = 180.0) -> Optional[dict]:
    """Discord/Google OAuth via local loopback callback. provider: 'discord'|'google'."""
    if provider not in ("discord", "google"):
        return None
    app = get_app(f"local_cli.oauth.{provider}")
    port = int(os.getenv("TB_OAUTH_CALLBACK_PORT", "8765"))
    redirect = f"http://{_HOST}:{port}/auth/{provider}/callback"
    env_key = f"{provider.upper()}_REDIRECT_URI"
    _prev = os.environ.get(env_key)
    os.environ[env_key] = redirect              # token exchange must reuse this value
    try:
        res = await app.a_run_any(
            ("CloudM.Auth", f"get_{provider}_auth_url"),
            redirect_after=redirect, get_results=True,
        )
        if hasattr(res, "is_error") and res.is_error():
            return None
        data = res.get() if hasattr(res, "get") else res
        auth_url = (data or {}).get("auth_url")
        if not auth_url:
            return None

        cap = _OAuthCapture()
        server = ThreadingHTTPServer((_HOST, port), _oauth_handler(provider, cap))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            webbrowser.open(auth_url)
            ok = await asyncio.to_thread(cap.done.wait, timeout)
            if not ok or cap.error or not cap.code:
                return None
            lr = await app.a_run_any(
                ("CloudM.Auth", f"login_{provider}"),
                code=cap.code, state=cap.state, get_results=True,
            )
            if hasattr(lr, "is_error") and lr.is_error():
                return None
            payload = lr.get() if hasattr(lr, "get") else lr
            return payload if isinstance(payload, dict) and payload.get("authenticated") else None
        finally:
            server.shutdown()
            server.server_close()
    finally:
        _restore_env(env_key, _prev)


# =============================================================================
# Passkey (WebAuthn) — shared loopback ceremony runner
# =============================================================================

# Shared base64url <-> ArrayBuffer helpers + finish POST.
_JS_HELPERS = r"""
const b64uToBuf=s=>{s=s.replace(/-/g,'+').replace(/_/g,'/');const p='='.repeat((4-s.length%4)%4);
const b=atob(s+p),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a.buffer;};
const bufToB64u=b=>{const a=new Uint8Array(b);let s='';for(const x of a)s+=String.fromCharCode(x);
return btoa(s).replace(/\+/g,'-').replace(/\//g,'_').replace(/=+$/,'');};
const setMsg=t=>{document.getElementById('m').textContent=t;};
const finish=async body=>(await(await fetch('/finish',{method:'POST',
headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json());
"""

_PAGE_LOGIN = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Passkey login</title>
<style>body{font-family:sans-serif;background:#1a1a2e;color:#fff;text-align:center;padding-top:16vh}</style>
</head><body><h2 id="m">Touch your authenticator…</h2><script>%s
(async()=>{try{
const opts=await(await fetch('/start',{method:'POST'})).json();
if(opts.error){setMsg('Failed: '+opts.error);return;}
const challenge=opts.challenge;
opts.challenge=b64uToBuf(opts.challenge);
if(opts.allowCredentials)for(const c of opts.allowCredentials)c.id=b64uToBuf(c.id);
const cred=await navigator.credentials.get({publicKey:opts});
const out={id:cred.id,rawId:bufToB64u(cred.rawId),type:cred.type,
response:{authenticatorData:bufToB64u(cred.response.authenticatorData),
clientDataJSON:bufToB64u(cred.response.clientDataJSON),
signature:bufToB64u(cred.response.signature),
userHandle:cred.response.userHandle?bufToB64u(cred.response.userHandle):null},
clientExtensionResults:cred.getClientExtensionResults()};
const r=await finish({challenge,credential:out});
setMsg(r.ok?'Login complete — return to the CLI.':'Failed: '+(r.error||'unknown'));
}catch(e){setMsg('Error: '+e);try{await finish({error:String(e)});}catch(_){}}})();
</script></body></html>""" % _JS_HELPERS

_PAGE_REGISTER = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Passkey register</title>
<style>body{font-family:sans-serif;background:#1a1a2e;color:#fff;text-align:center;padding-top:16vh}</style>
</head><body><h2 id="m">Create your passkey…</h2><script>%s
(async()=>{try{
const opts=await(await fetch('/start',{method:'POST'})).json();
if(opts.error){setMsg('Failed: '+opts.error);return;}
const challenge=opts.challenge;
opts.challenge=b64uToBuf(opts.challenge);
opts.user.id=b64uToBuf(opts.user.id);
if(opts.excludeCredentials)for(const c of opts.excludeCredentials)c.id=b64uToBuf(c.id);
const cred=await navigator.credentials.create({publicKey:opts});
const out={id:cred.id,rawId:bufToB64u(cred.rawId),type:cred.type,
response:{attestationObject:bufToB64u(cred.response.attestationObject),
clientDataJSON:bufToB64u(cred.response.clientDataJSON),
transports:(cred.response.getTransports?cred.response.getTransports():[])},
clientExtensionResults:cred.getClientExtensionResults()};
const r=await finish({challenge,credential:out});
setMsg((r.success||r.ok)?'Passkey registered — return to the CLI.':'Failed: '+(r.error||'unknown'));
}catch(e){setMsg('Error: '+e);try{await finish({error:String(e)});}catch(_){}}})();
</script></body></html>""" % _JS_HELPERS


class _PkState:
    def __init__(self):
        self.payload: Optional[dict] = None   # login: token payload; register: {"success":True}
        self.ok = False
        self.done = threading.Event()


def _pk_handler(page: str, start_fn: str, start_kwargs: dict,
                finish_fn: str, want_payload: bool, state: _PkState, loop, app):
    def call(fn, **kw):
        fut = asyncio.run_coroutine_threadsafe(
            app.a_run_any(("CloudM.Auth", fn), get_results=True, **kw), loop)
        r = fut.result(timeout=30)
        if hasattr(r, "is_error") and r.is_error():
            err = "auth error"
            if hasattr(r, "info") and hasattr(r.info, "help_text"):
                err = r.info.help_text
            return {"error": err}
        return r.get() if hasattr(r, "get") else r

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _json(self, obj, status=200):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if urlparse(self.path).path != "/":
                self.send_response(404); self.end_headers(); return
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/start":
                self._json(call(start_fn, **start_kwargs))
                return
            if path == "/finish":
                length = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    body = json.loads(raw.decode() or "{}")
                except Exception:
                    body = {}
                if body.get("error"):
                    self._json({"ok": False, "error": body["error"]})
                    state.done.set(); return
                res = call(finish_fn, challenge=body.get("challenge"),
                           credential=body.get("credential"))
                if want_payload:
                    if isinstance(res, dict) and res.get("authenticated"):
                        state.payload, state.ok = res, True
                        self._json({"ok": True})
                    else:
                        self._json({"ok": False, "error": (res or {}).get("error", "verification failed")})
                else:
                    if isinstance(res, dict) and res.get("success"):
                        state.payload, state.ok = res, True
                        self._json({"ok": True, "success": True})
                    else:
                        self._json({"ok": False, "error": (res or {}).get("error", "registration failed")})
                state.done.set()
                return
            self.send_response(404); self.end_headers()
    return H


def _restore_env(key: str, val: Optional[str]):
    if val is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = val


async def _run_passkey(page, start_fn, start_kwargs, finish_fn, want_payload,
                       port, timeout) -> Optional[dict]:
    # Use the deployment's rp_id so ONE passkey is shared with the web UI.
    # Only loopback rp_ids can be served locally; a real domain rp_id must run
    # the ceremony on that origin in the browser (web flow), not here.
    try:
        rp = get_passkey_config().get("rp_id", "")
    except Exception:
        rp = ""
    if rp not in ("localhost", "127.0.0.1"):
        print(f"[passkey] deployment rp_id='{rp}' is not loopback — register/login "
              f"the passkey via the web UI on that origin. CLI passkey supports "
              f"local (loopback) deployments only.")
        return None
    host = rp

    app = get_app("local_cli.passkey")
    loop = asyncio.get_running_loop()
    _prev_rp = os.environ.get("PASSKEY_RP_ID")
    _prev_origin = os.environ.get("PASSKEY_ORIGIN")
    os.environ["PASSKEY_RP_ID"] = host                      # == web rp_id -> shared credential
    os.environ["PASSKEY_ORIGIN"] = f"http://{host}:{port}"  # per-ceremony origin
    state = _PkState()
    server = ThreadingHTTPServer(
        (host, port),
        _pk_handler(page, start_fn, start_kwargs, finish_fn, want_payload, state, loop, app),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        webbrowser.open(f"http://{host}:{port}/")
        ok = await asyncio.to_thread(state.done.wait, timeout)
        return state.payload if (ok and state.ok) else None
    finally:
        server.shutdown()
        server.server_close()
        _restore_env("PASSKEY_RP_ID", _prev_rp)
        _restore_env("PASSKEY_ORIGIN", _prev_origin)


async def cli_passkey_login(timeout: float = 180.0) -> Optional[dict]:
    """WebAuthn passkey login (discoverable credential). Returns auth payload or None."""
    port = int(os.getenv("TB_PASSKEY_CALLBACK_PORT", "8766"))
    return await _run_passkey(
        _PAGE_LOGIN, "passkey_login_start", {},
        "passkey_login_finish", want_payload=True, port=port, timeout=timeout,
    )


async def cli_passkey_register(user_id: str, username: str, timeout: float = 180.0) -> bool:
    """Register a passkey for an already-logged-in user. Returns True on success."""
    if not user_id or not username:
        return False
    port = int(os.getenv("TB_PASSKEY_CALLBACK_PORT", "8766"))
    res = await _run_passkey(
        _PAGE_REGISTER, "passkey_register_start", {"user_id": user_id, "username": username},
        "passkey_register_finish", want_payload=False, port=port, timeout=timeout,
    )
    return bool(res and res.get("success"))
