"""account_page.py - minimal account-link page served by the local UI (WP1 K3).

Self-contained HTML+JS: POSTs to /api/account/start with the chosen method
and polls /api/account/status. Loopback-guarded via live_proxy._ensure_local.
"""

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ToolBoxV2 - Link account</title>
<style>
  body { font-family: system-ui, sans-serif; background: #0f1115; color: #e6e6e6;
         display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
  .card { background: #1a1d24; padding: 2rem 2.5rem; border-radius: 12px; min-width: 320px; }
  h1 { font-size: 1.2rem; margin-top: 0; }
  button { display: block; width: 100%; margin: .5rem 0; padding: .6rem; border: 0;
           border-radius: 8px; background: #4c8dff; color: #fff; font-size: 1rem; cursor: pointer; }
  button:hover { background: #3a76e0; }
  #status { margin-top: 1rem; font-size: .9rem; color: #9aa0a6; white-space: pre-line; }
</style>
</head>
<body>
<div class="card">
  <h1>Link a SimpleCore account</h1>
  <p style="color:#9aa0a6;font-size:.9rem">Keeps your local admin data and migrates
     settings to the account. Tokens stay in the local token store.</p>
  <button onclick="start('discord')">Discord</button>
  <button onclick="start('google')">Google</button>
  <button onclick="start('passkey')">Passkey</button>
  <div id="status"></div>
</div>
<script>
const status = document.getElementById('status');
let polling = null;
async function poll() {
  const r = await fetch('/api/account/status');
  const d = await r.json();
  if (d.state === 'linked') {
    status.textContent = 'Linked as ' + (d.username || d.user_id);
    if (polling) { clearInterval(polling); polling = null; }
  } else if (d.state === 'error') {
    status.textContent = 'Failed: ' + (d.error || 'unknown');
    if (polling) { clearInterval(polling); polling = null; }
  } else {
    status.textContent = 'Waiting for login ...';
  }
}
async function start(method) {
  status.textContent = 'Starting background login ...';
  try {
    const r = await fetch('/api/account/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({method})
    });
    const d = await r.json();
    if (!r.ok) { status.textContent = 'Failed: ' + (d.error || r.status); return; }
    if (polling) clearInterval(polling);
    polling = setInterval(poll, 1500);
  } catch (e) {
    status.textContent = 'Failed: ' + e;
  }
}
poll();
</script>
</body>
</html>
"""


def account_page() -> str:
    return _PAGE
