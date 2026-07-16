# ToolBoxV2 Worker Manager — Operations Docs

## Voraussetzungen

```bash
# nginx installieren
sudo apt install nginx apache2-utils

# ADMIN_UI_PASSWORD setzen (persistent in ~/.bashrc oder /etc/environment)
export ADMIN_UI_PASSWORD="dein_passwort"

# ToolBoxV2 installiert und tb im PATH
```

---

## 1. Erstmalige Server-Einrichtung

```bash
# Schritt 1: nginx.conf include patchen (einmalig, non-invasiv)
sudo tb workers nginx-init

# Schritt 2: Site-Config + htpasswd schreiben
sudo tb workers nginx-config --write-htpasswd

# Schritt 3: Prüfen ob alles korrekt ist
tb workers nginx-check

# Erwartete Ausgabe:
# nginx.conf path:     /etc/nginx/nginx.conf
# conf exists:         ✓
# box-enabled include: ✓
# box-available dir:   ✓
# box-enabled dir:     ✓
# toolbox config:      ✓
# toolbox symlink:     ✓
# nginx installed:     ✓
# nginx version:       nginx/1.24.0
# htpasswd:            ✓

# Schritt 4: Alles starten
tb workers start
```

Nach `start` läuft:
- ZMQ Broker (IPC zwischen Workern)
- N HTTP Worker (aus config)
- 1 WS Worker
- Health Checker (alle 5s)
- Metrics Collector (alle 60s)
- Web UI auf `http://127.0.0.1:9005` (via `/admin/manager/` durch nginx mit Basic Auth)

---

## 2. Config-Struktur verstehen

```
/etc/nginx/
  nginx.conf                    ← nur gelesen, nie überschrieben
  box-available/
    toolbox                     ← generierte Site-Config (hier stehen ALLE ports vorab)
  box-enabled/
    toolbox -> ../box-available/toolbox   ← Symlink, nginx lädt diese
```

**Wichtig:** Die Config wird **einmalig** mit pre-allokierten Port-Ranges geschrieben. Nginx weiß von Anfang an von allen möglichen Worker-Ports (`8000–8003` für HTTP, `8100–8103` für WS). Tote Worker werden automatisch via passive health check übersprungen (`max_fails=1 fail_timeout=5s`). **Kein nginx-Reload beim Skalieren nötig.**

---

## 3. Worker-Lifecycle

### Starten (alle auf einmal)
```bash
tb workers start              # foreground, Ctrl+C stoppt alles
tb workers start --no-ui      # ohne Web UI
```

### Stoppen
```bash
tb workers stop               # graceful shutdown aller Worker + Broker
```

### Restart
```bash
tb workers restart            # stop → 2s warten → start
```

### Einzelnen Worker starten
```bash
tb workers worker-start -t http              # neuer HTTP Worker, auto-ID + auto-Port
tb workers worker-start -t http -w main_http # mit fester ID
tb workers worker-start -t ws                # neuer WS Worker
```

### Einzelnen Worker stoppen
```bash
tb workers worker-stop -w http_a3f2b1c0
```

### Worker-ID herausfinden
```bash
tb workers status | grep worker_id
# oder Web UI: http://your-server/admin/manager/
```

---

## 4. Skalieren

Skalierung erfolgt zur Laufzeit. Nginx braucht keinen Reload — die neuen Ports waren schon in der Config vorab registriert.

```bash
# Via Web UI: POST /admin/manager/api/scale
curl -X POST http://127.0.0.1:9005/admin/manager/api/scale \
  -H "Content-Type: application/json" \
  -d '{"type": "http", "count": 4}'

# Antwort:
# {"status": "ok", "started": ["http_x1", "http_x2"], "stopped": []}
```

**Limit:** Nur bis zur pre-allokierten Port-Range skalierbar (konfiguriert in `config.http_worker.workers`). Darüber hinaus → Config neu schreiben mit `--force`.

---

## 5. Rolling Update (Zero-Downtime)

Startet pro bestehendem HTTP-Worker einen neuen, validiert ihn, schaltet nginx um, fährt alten runter.

```bash
tb workers update

# Oder via API:
curl -X POST http://127.0.0.1:9005/admin/manager/api/rolling-update
```

**Ablauf pro Worker:**
1. Neuer Worker startet auf nächstem freien Port
2. 2s warten
3. `/health` Check auf neuem Worker
4. nginx reload
5. Alter Worker draining → stop

---

## 6. Cluster (mehrere Server)

Jeder Server läuft `tb workers start`. Server A kennt Server B als Remote-Node. Nginx auf Server A leitet bei Bedarf an Server B weiter (als `backup`).

```bash
# Auf Server A: Server B hinzufügen
tb workers cluster-join \
  --host server-b.example.com \
  --port 9005 \
  --secret dein_cluster_secret

# Cluster-Secret auf allen Servern gleich setzen:
export CLUSTER_SECRET="shared_secret_alle_server"
tb workers start
```

**Was Cluster-Mode macht:**
- Server A pollt Server B alle 10s nach Worker-Status
- Gesunde HTTP-Worker von B werden als `backup` in nginx upstream eingetragen
- Fällt B aus → backup entfernt sich automatisch via passive health check

```bash
# Status aller Nodes prüfen
tb workers status | python3 -m json.tool | grep -A5 cluster
```

---

## 7. nginx-Config neu schreiben (nach Konfig-Änderung)

```bash
# Worker-Anzahl in config geändert → Config neu generieren
sudo tb workers nginx-config --force

# nginx testen + reloaden
tb workers nginx-reload
```

```bash
# Nur htpasswd erneuern (neues Passwort)
export ADMIN_UI_PASSWORD="neues_passwort"
sudo tb workers nginx-config --write-htpasswd
```

---

## 8. Health & Metrics

```bash
# Alle Worker + Health-Status
tb workers status

# Kompakt: nur unhealthy Worker
tb workers status | python3 -c "
import json, sys
s = json.load(sys.stdin)
for wid, w in s['workers'].items():
    if not w['healthy']:
        print(f\"UNHEALTHY: {wid} state={w['state']}\")
"
```

```bash
# Via API
curl http://127.0.0.1:9005/admin/manager/api/health
curl http://127.0.0.1:9005/admin/manager/api/metrics
curl http://127.0.0.1:9005/admin/manager/api/workers
```

**Health-Check-Logik:**
- HTTP Worker: `GET /health` → erwartet HTTP 200
- WS Worker: TCP connect + `GET /health` → erwartet HTTP 200 (kein WS-Handshake nötig)
- Interval: 5 Sekunden
- Ergebnis in `worker.healthy` + `worker.health_latency_ms`

---

## 9. Systemd-Service (Production)

```ini
# /etc/systemd/system/tb-workers.service
[Unit]
Description=ToolBoxV2 Worker Manager
After=network.target nginx.service

[Service]
Type=simple
User=www-data
Environment=ADMIN_UI_PASSWORD=dein_passwort
Environment=CLUSTER_SECRET=dein_cluster_secret
ExecStartPre=tb workers nginx-check
ExecStart=tb workers start --no-ui
ExecStop=tb workers stop
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable tb-workers
sudo systemctl start tb-workers
sudo systemctl status tb-workers
```

---

## 10. Troubleshooting

| Problem | Diagnose | Fix |
|---|---|---|
| nginx leitet nicht weiter | `tb workers nginx-check` | `sudo tb workers nginx-init` |
| Admin UI 401 | htpasswd fehlt | `sudo tb workers nginx-config --write-htpasswd` |
| Worker startet nicht | `tb workers status` → `state: failed` | `tb workers worker-stop -w <id>` dann `worker-start` |
| Alle Worker unhealthy | nginx gibt 502 | `tb workers restart` |
| Config-Änderung greift nicht | ports außerhalb Range | `sudo tb workers nginx-config --force && tb workers nginx-reload` |
| Cluster-Node nicht erreichbar | `status` → `healthy_nodes: 0` | Firewall Port 9005, CLUSTER_SECRET prüfen |

```bash
# nginx Error-Log live
sudo tail -f /var/log/nginx/toolbox_error.log

# Worker Manager Log
journalctl -u tb-workers -f
```
