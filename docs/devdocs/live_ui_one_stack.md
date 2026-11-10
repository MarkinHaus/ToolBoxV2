# Live-UI Owner & One Stack (Notizen, Stand 2026-11)

## Heute (nach A + C + D)

```
lokal (nginx aus)                       server (nginx an)
─────────────────                       ─────────────────
127.0.0.1:8467  Live-UI Owner           nginx :443 ─► HTTP-Worker 8000+ (dist, /api)
  ├─ dist/ + /api/* (in-process)                 └► WS-Worker  8468+
  ├─ /tray/*, /api/wizard/*             127.0.0.1:8467  Live-UI Owner (nur lokal,
  └─ gemountete Apps (/isaa …)                          nie über nginx)
ZMQ-Broker + WS-Worker :8468
  └─ Owner hängt als WS-Bridge dran,
     sobald eine App mit WS-Routen gemountet wird
```

- **A** — `FastTB.attach_ws_bridge()`: Owner abonniert den bestehenden Broker erst beim
  ersten Mount einer App mit WS-Routen. Standby-Replicas starten nichts → keine
  Doppelverarbeitung.
- **C** — gestartete Apps stehen in `TBEF.DB` unter `LIVE_UI::mounted_apps`; eine Replica,
  die den Port übernimmt, mountet sie neu (`restore_mounted_apps`).
- **D** — HTTP-Worker nur mit `nginx.enabled`; lokal bedient der Owner alles.
- 8467 bindet immer Loopback (`live_ui_bind_host`), egal was konfiguriert ist.

## E — ein Stack statt zwei (nicht umgesetzt, Optionen)

Problem: HTTP-Worker (`server_worker.HTTPWorker`, raw WSGI) und FastTB-Owner sind zwei
HTTP-Stacks mit eigener Routing-Logik. FastTB nutzt intern bereits einen HTTPWorker
für Built-ins (`as_wsgi_app`) — der Owner *ist* also schon ein HTTPWorker plus Routen.

1. **HTTP-Worker = FastTB-Owner ohne `/`** (empfohlen, mittel)
   Jeder HTTP-Worker-Prozess startet als `FastTB` mit `as_wsgi_app` statt `HTTPWorker.run`.
   Apps werden per Registry (`user_facing.mount`) in *alle* Worker gemountet — Mount-Liste
   aus TBEF.DB (wie C). nginx bleibt Load-Balancer. Kein Collective/Standby mehr nötig,
   der lokale Owner ist dann nur „ein Worker ohne nginx“.
2. **Mount-Broadcast über ZMQ** (klein, Zwischenschritt)
   `start_app` publiziert ein `APP_MOUNT`-Event; jeder HTTP-Worker mit FastTB-Hülle mountet
   selbst. Server-Betrieb bekommt die Apps hinter nginx, ohne 8467 zu öffnen.
3. **Owner als nginx-Upstream** — verworfen: 8467 darf nicht exposed werden.

Offen/zu klären vor E:
- WS-Port: erledigt — lokal EIN Port (`ws_worker.port`, 8468) für Worker-Manager,
  Tauri, FastTB und alle Frontends; `ws_base_port()` ist die eine Quelle für
  Worker-Start und nginx-Upstreams. Separater Port bleibt Pflicht (Waitress/WSGI
  kann kein WS-Upgrade).
- `/tray/command mount_app` importiert beliebige Module (nur noch CSRF-geschützt);
  bei E auf Registry-Namen beschränken.
