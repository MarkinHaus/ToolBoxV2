#!/usr/bin/env bash
# =============================================================================
# ToolBoxV2 — Registry Setup: CNAME + SSL + Nginx
# =============================================================================
# Verwendung:
#   bash setup_registry.sh
#   bash setup_registry.sh --registry-domain registry.simplecore.app \
#                          --main-domain simplecore.app \
#                          --email admin@simplecore.app \
#                          --mode same-machine
#
# Modi:
#   same-machine   Registry läuft als TB-Mod auf demselben Server (Port 4025)
#   remote         Registry auf anderer Maschine — nur Nginx-Proxy einrichten
#   new-user       Neuen dedizierten User für Registry anlegen
#
# Was dieses Script tut:
#   1. Modus wählen (same-machine / remote / new-user)
#   2. User anlegen + SSH-Key (new-user Modus)
#   3. Existierendes Zertifikat erkennen (Let's Encrypt / custom)
#      → Falls vorhanden: direkt verwenden
#      → Falls nicht:     certbot für CNAME-Subdomain
#   4. Nginx-Config aus registry.conf Template generieren
#      → Pfade + Domain + SSL-Cert-Pfade einsetzen
#   5. Site aktivieren + Nginx reload
#   6. tb manifest set registry.url aktualisieren
# =============================================================================

set -euo pipefail

C_RESET='\033[0m'; C_RED='\033[0;31m'; C_GREEN='\033[0;32m'
C_YELLOW='\033[0;33m'; C_CYAN='\033[0;36m'; C_BOLD='\033[1m'

log_info()    { echo -e "${C_CYAN}ℹ  ${*}${C_RESET}"; }
log_ok()      { echo -e "${C_GREEN}✅ ${*}${C_RESET}"; }
log_warn()    { echo -e "${C_YELLOW}⚠  ${*}${C_RESET}"; }
log_error()   { echo -e "${C_RED}❌ ${*}${C_RESET}"; exit 1; }
log_section() { echo -e "\n${C_BOLD}${C_CYAN}══ ${*} ══${C_RESET}"; }

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
REGISTRY_DOMAIN=""
MAIN_DOMAIN=""
EMAIL=""
REGISTRY_PORT=4025          # Port aus Dockerfile EXPOSE + registry.conf upstream
SETUP_MODE=""               # same-machine | remote | new-user
REMOTE_HOST=""              # nur bei mode=remote
REGISTRY_USER="tbregistry"
REGISTRY_HOME="/opt/tb-registry"
NGINX_BOX_AVAILABLE="/etc/nginx/box-available"
NGINX_BOX_ENABLED="/etc/nginx/box-enabled"
CERTBOT_WEBROOT="/var/lib/letsencrypt/webroot"

# -----------------------------------------------------------------------------
# Argumente
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --registry-domain) REGISTRY_DOMAIN="$2"; shift 2 ;;
        --main-domain)     MAIN_DOMAIN="$2";     shift 2 ;;
        --email)           EMAIL="$2";            shift 2 ;;
        --mode)            SETUP_MODE="$2";       shift 2 ;;
        --remote-host)     REMOTE_HOST="$2";      shift 2 ;;
        --port)            REGISTRY_PORT="$2";    shift 2 ;;
        *) log_warn "Unbekanntes Argument: $1"; shift ;;
    esac
done

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
log_section "ToolBoxV2 Registry Setup"
echo ""

# -----------------------------------------------------------------------------
# Interaktive Eingaben
# -----------------------------------------------------------------------------
if [[ -z "$REGISTRY_DOMAIN" ]]; then
    read -r -p "$(echo -e "${C_BOLD}Registry-Domain (z.B. registry.simplecore.app): ${C_RESET}")" REGISTRY_DOMAIN
    [[ -z "$REGISTRY_DOMAIN" ]] && log_error "Registry-Domain darf nicht leer sein."
fi

if [[ -z "$MAIN_DOMAIN" ]]; then
    read -r -p "$(echo -e "${C_BOLD}Haupt-Domain (z.B. simplecore.app, oder leer lassen): ${C_RESET}")" MAIN_DOMAIN
fi

if [[ -z "$EMAIL" ]]; then
    read -r -p "$(echo -e "${C_BOLD}E-Mail für SSL-Zertifikat: ${C_RESET}")" EMAIL
    [[ -z "$EMAIL" ]] && log_error "E-Mail darf nicht leer sein."
fi

# Modus wählen
if [[ -z "$SETUP_MODE" ]]; then
    echo ""
    echo -e "  ${C_BOLD}Wie läuft die Registry?${C_RESET}"
    echo -e "  ${C_CYAN}1)${C_RESET} same-machine  — auf diesem Server, selber TB-User"
    echo -e "  ${C_CYAN}2)${C_RESET} remote        — auf anderem Server, nur Nginx-Proxy hier"
    echo -e "  ${C_CYAN}3)${C_RESET} new-user      — auf diesem Server, neuer dedizierter User"
    echo ""
    read -r -p "$(echo -e "${C_BOLD}Wahl [1]: ${C_RESET}")" mode_choice
    case "${mode_choice:-1}" in
        2) SETUP_MODE="remote" ;;
        3) SETUP_MODE="new-user" ;;
        *) SETUP_MODE="same-machine" ;;
    esac
fi

# Remote-Host bei mode=remote
if [[ "$SETUP_MODE" == "remote" && -z "$REMOTE_HOST" ]]; then
    read -r -p "$(echo -e "${C_BOLD}IP/Hostname des Registry-Servers: ${C_RESET}")" REMOTE_HOST
    [[ -z "$REMOTE_HOST" ]] && log_error "Remote-Host darf nicht leer sein."
fi

echo ""
log_info "Registry-Domain: $REGISTRY_DOMAIN"
log_info "Modus:           $SETUP_MODE"
[[ "$SETUP_MODE" == "remote" ]] && log_info "Remote-Host:     $REMOTE_HOST:$REGISTRY_PORT"
log_info "Registry-Port:   $REGISTRY_PORT"
echo ""

read -r -p "$(echo -e "${C_BOLD}Fortfahren? [Y/n]: ${C_RESET}")" confirm
[[ "${confirm,,}" == "n" ]] && echo "Abgebrochen." && exit 0

# Root-Check
[[ "$(id -u)" -ne 0 ]] && log_error "Benötigt root (sudo bash $0)"

# =============================================================================
# SCHRITT 1 — Abhängigkeiten
# =============================================================================
log_section "Schritt 1: Abhängigkeiten"

install_pkg() {
    if command -v apt-get &>/dev/null; then apt-get install -y -q "$@"
    elif command -v dnf &>/dev/null; then dnf install -y -q "$@"
    elif command -v yum &>/dev/null; then yum install -y -q "$@"
    else log_warn "Bitte manuell installieren: $*"; fi
}

command -v nginx   &>/dev/null || { log_info "Installiere nginx...";   install_pkg nginx; }
command -v certbot &>/dev/null || { log_info "Installiere certbot..."; install_pkg certbot python3-certbot-nginx; }
command -v openssl &>/dev/null || install_pkg openssl

log_ok "nginx, certbot, openssl verfügbar"

# =============================================================================
# SCHRITT 2 — User anlegen (nur new-user Modus)
# =============================================================================
log_section "Schritt 2: User-Setup (Modus: $SETUP_MODE)"

if [[ "$SETUP_MODE" == "new-user" ]]; then
    if id "$REGISTRY_USER" &>/dev/null; then
        log_ok "User '$REGISTRY_USER' existiert bereits."
    else
        useradd \
            --system \
            --create-home \
            --home-dir "$REGISTRY_HOME" \
            --shell /bin/bash \
            --comment "TB Registry Service User" \
            "$REGISTRY_USER"
        log_ok "User '$REGISTRY_USER' erstellt (Home: $REGISTRY_HOME)"
    fi

    # SSH-Key
    REGISTRY_SSH_DIR="$REGISTRY_HOME/.ssh"
    REGISTRY_KEY_PATH="$REGISTRY_SSH_DIR/id_ed25519"
    if [[ ! -f "$REGISTRY_KEY_PATH" ]]; then
        mkdir -p "$REGISTRY_SSH_DIR"
        chmod 700 "$REGISTRY_SSH_DIR"
        ssh-keygen -t ed25519 -C "tbregistry@$REGISTRY_DOMAIN" -f "$REGISTRY_KEY_PATH" -N ""
        cat "$REGISTRY_KEY_PATH.pub" >> "$REGISTRY_SSH_DIR/authorized_keys"
        chmod 600 "$REGISTRY_SSH_DIR/authorized_keys"
        chown -R "$REGISTRY_USER:$REGISTRY_USER" "$REGISTRY_SSH_DIR"
        log_ok "SSH-Key: $REGISTRY_KEY_PATH"
        echo ""
        echo -e "${C_BOLD}Öffentlicher Schlüssel:${C_RESET}"
        echo -e "${C_CYAN}$(cat "$REGISTRY_KEY_PATH.pub")${C_RESET}"
    else
        log_ok "SSH-Key existiert: $REGISTRY_KEY_PATH"
    fi

elif [[ "$SETUP_MODE" == "remote" ]]; then
    log_info "Remote-Modus: Kein lokaler User nötig."
    log_info "Stelle sicher dass Port $REGISTRY_PORT auf $REMOTE_HOST erreichbar ist."

    # SSH-Connectivity-Test (optional)
    read -r -p "$(echo -e "${C_BOLD}SSH-Verbindung zu $REMOTE_HOST testen? [y/N]: ${C_RESET}")" test_ssh
    if [[ "${test_ssh,,}" == "y" ]]; then
        SSH_USER="${SUDO_USER:-$(logname 2>/dev/null || echo root)}"
        read -r -p "$(echo -e "${C_BOLD}SSH-User für $REMOTE_HOST [$SSH_USER]: ${C_RESET}")" ssh_user_input
        SSH_USER="${ssh_user_input:-$SSH_USER}"

        if ssh -o ConnectTimeout=5 -o BatchMode=yes "$SSH_USER@$REMOTE_HOST" "echo ok" 2>/dev/null; then
            log_ok "SSH-Verbindung zu $REMOTE_HOST funktioniert."
        else
            log_warn "SSH-Verbindung fehlgeschlagen — bitte manuell prüfen."
            log_info "SSH-Key generieren für $SSH_USER:"
            echo -e "  ${C_CYAN}ssh-keygen -t ed25519 -C 'nginx-proxy@$(hostname)'${C_RESET}"
            echo -e "  ${C_CYAN}ssh-copy-id $SSH_USER@$REMOTE_HOST${C_RESET}"
        fi
    fi
else
    # same-machine — nutze existierenden toolbox-User
    REGISTRY_USER="${SUDO_USER:-toolbox}"
    log_ok "same-machine: Registry läuft als User '$REGISTRY_USER' auf Port $REGISTRY_PORT."
fi

# =============================================================================
# SCHRITT 3 — SSL-Zertifikat erkennen oder beschaffen
# =============================================================================
log_section "Schritt 3: SSL-Zertifikat für $REGISTRY_DOMAIN"

# --- Erkennungs-Logik (analog SSLManager.discover() aus cli_worker_manager.py) ---
CERT_PATH=""
KEY_PATH=""
CERT_SOURCE=""

# Priorität 1: Let's Encrypt für registry-domain
LE_CERT="/etc/letsencrypt/live/$REGISTRY_DOMAIN/fullchain.pem"
LE_KEY="/etc/letsencrypt/live/$REGISTRY_DOMAIN/privkey.pem"

# Priorität 2: Wildcard von main-domain
if [[ -n "$MAIN_DOMAIN" ]]; then
    WILDCARD_CERT="/etc/letsencrypt/live/$MAIN_DOMAIN/fullchain.pem"
    WILDCARD_KEY="/etc/letsencrypt/live/$MAIN_DOMAIN/privkey.pem"
else
    WILDCARD_CERT=""
    WILDCARD_KEY=""
fi

# Priorität 3: Custom-Cert-Pfade
CUSTOM_CERT="/etc/nginx/ssl/$REGISTRY_DOMAIN.crt"
CUSTOM_KEY="/etc/nginx/ssl/$REGISTRY_DOMAIN.key"

if [[ -f "$LE_CERT" && -f "$LE_KEY" ]]; then
    CERT_PATH="$LE_CERT"
    KEY_PATH="$LE_KEY"
    CERT_SOURCE="Let's Encrypt (registry-domain)"
    log_ok "Existierendes Let's Encrypt Zertifikat gefunden."
    expiry=$(openssl x509 -enddate -noout -in "$CERT_PATH" 2>/dev/null | cut -d= -f2)
    log_info "  Läuft ab: $expiry"

elif [[ -n "$WILDCARD_CERT" && -f "$WILDCARD_CERT" && -f "$WILDCARD_KEY" ]]; then
    # Prüfe ob Wildcard das registry-subdomain abdeckt
    cn=$(openssl x509 -noout -subject -in "$WILDCARD_CERT" 2>/dev/null | grep -oP '(?<=CN=)[^\s,]+')
    san=$(openssl x509 -noout -text -in "$WILDCARD_CERT" 2>/dev/null | grep -A1 "Subject Alternative" | tail -1)
    if echo "$san $cn" | grep -qE "\*\.${MAIN_DOMAIN}|${REGISTRY_DOMAIN}"; then
        CERT_PATH="$WILDCARD_CERT"
        KEY_PATH="$WILDCARD_KEY"
        CERT_SOURCE="Wildcard von $MAIN_DOMAIN"
        log_ok "Wildcard-Zertifikat von $MAIN_DOMAIN abdeckt $REGISTRY_DOMAIN."
    else
        log_info "Wildcard-Zertifikat deckt $REGISTRY_DOMAIN nicht ab."
    fi

elif [[ -f "$CUSTOM_CERT" && -f "$CUSTOM_KEY" ]]; then
    CERT_PATH="$CUSTOM_CERT"
    KEY_PATH="$CUSTOM_KEY"
    CERT_SOURCE="Custom (/etc/nginx/ssl/)"
    log_ok "Custom-Zertifikat gefunden: $CUSTOM_CERT"
fi

# Kein Zertifikat gefunden → certbot
if [[ -z "$CERT_PATH" ]]; then
    log_info "Kein Zertifikat gefunden — starte certbot für $REGISTRY_DOMAIN..."

    # DNS-Check vor certbot
    DNS_IP=$(dig +short "$REGISTRY_DOMAIN" 2>/dev/null | tail -1)
    SERVER_IP=$(curl -s --max-time 5 https://api.ipify.org 2>/dev/null || echo "")
    if [[ -n "$DNS_IP" && -n "$SERVER_IP" && "$DNS_IP" != "$SERVER_IP" ]]; then
        log_warn "DNS für $REGISTRY_DOMAIN zeigt auf $DNS_IP, aber dieser Server ist $SERVER_IP"
        log_warn "CNAME/A-Record muss zuerst auf diesen Server zeigen!"
        echo ""
        echo -e "${C_BOLD}DNS-Konfiguration (bei deinem DNS-Provider):${C_RESET}"
        echo -e "  ${C_CYAN}$REGISTRY_DOMAIN.  IN  CNAME  $MAIN_DOMAIN.${C_RESET}"
        echo -e "  oder:"
        echo -e "  ${C_CYAN}$REGISTRY_DOMAIN.  IN  A      $SERVER_IP${C_RESET}"
        echo ""
        read -r -p "$(echo -e "${C_BOLD}Trotzdem certbot versuchen? [y/N]: ${C_RESET}")" force_certbot
        [[ "${force_certbot,,}" != "y" ]] && {
            log_warn "Certbot übersprungen. Nach DNS-Propagation erneut ausführen."
            CERT_PATH="$LE_CERT"   # Platzhalter — wird nicht existieren
            KEY_PATH="$LE_KEY"
        }
    fi

    if [[ ! -f "$LE_CERT" ]]; then
        # Temp-HTTP-Config für ACME-Challenge
        mkdir -p "$CERTBOT_WEBROOT"
        TEMP_CONF="/etc/nginx/sites-available/certbot_registry_$REGISTRY_DOMAIN"
        cat > "$TEMP_CONF" <<NGINX_EOF
server {
    listen 80;
    server_name $REGISTRY_DOMAIN;
    location /.well-known/acme-challenge/ {
        root $CERTBOT_WEBROOT;
    }
    location / { return 444; }
}
NGINX_EOF
        ln -sf "$TEMP_CONF" "/etc/nginx/sites-enabled/certbot_registry"
        nginx -t && nginx -s reload 2>/dev/null || true

        certbot certonly \
            --webroot \
            --webroot-path "$CERTBOT_WEBROOT" \
            --domain "$REGISTRY_DOMAIN" \
            --email "$EMAIL" \
            --agree-tos \
            --non-interactive \
            && {
                CERT_PATH="$LE_CERT"
                KEY_PATH="$LE_KEY"
                CERT_SOURCE="Let's Encrypt (neu ausgestellt)"
                log_ok "Zertifikat ausgestellt: $CERT_PATH"
            } || log_warn "Certbot fehlgeschlagen — Nginx-Config ohne SSL generieren."

        rm -f "/etc/nginx/sites-enabled/certbot_registry" "$TEMP_CONF"
        nginx -s reload 2>/dev/null || true
    fi
fi

log_info "Zertifikat-Quelle: ${CERT_SOURCE:-'keines'}"

# =============================================================================
# SCHRITT 4 — Nginx-Config aus registry.conf Template generieren
# =============================================================================
log_section "Schritt 4: Nginx-Config generieren"

mkdir -p "$NGINX_BOX_AVAILABLE"

# Upstream-Adresse je nach Modus
if [[ "$SETUP_MODE" == "remote" ]]; then
    UPSTREAM_ADDR="$REMOTE_HOST:$REGISTRY_PORT"
else
    UPSTREAM_ADDR="127.0.0.1:$REGISTRY_PORT"
fi

REGISTRY_CONF="$NGINX_BOX_AVAILABLE/registry_$REGISTRY_DOMAIN"

# SSL-Block nur wenn Zertifikat wirklich existiert
if [[ -n "$CERT_PATH" && -f "$CERT_PATH" ]]; then
    SSL_REDIRECT_BLOCK="return 301 https://\$server_name\$request_uri;"
    SSL_SERVER_BLOCK=$(cat <<SSL_EOF

server {
    listen 443 ssl http2;
    server_name $REGISTRY_DOMAIN;

    ssl_certificate      $CERT_PATH;
    ssl_certificate_key  $KEY_PATH;
    ssl_protocols        TLSv1.2 TLSv1.3;
    ssl_ciphers          ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    ssl_session_cache    shared:SSL_REG:10m;
    ssl_session_timeout  1d;

    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Strict-Transport-Security "max-age=63072000" always;

    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types application/json application/javascript text/css text/plain;

    client_max_body_size 100M;

    # Health check (kein Rate-Limit)
    location /health {
        proxy_pass http://registry_${REGISTRY_PORT};
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }

    # API
    location /api/ {
        limit_req zone=registry_api_${REGISTRY_PORT} burst=20 nodelay;
        proxy_pass http://registry_${REGISTRY_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Connection "";
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Downloads (separates Rate-Limit)
    location ~ ^/api/packages/.*/download {
        limit_req zone=registry_dl_${REGISTRY_PORT} burst=10 nodelay;
        proxy_pass http://registry_${REGISTRY_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Connection "";
        proxy_read_timeout 300s;
    }

    # Package-Metadata (gecacht)
    location ~ ^/api/packages/[^/]+$ {
        proxy_pass http://registry_${REGISTRY_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Connection "";
        proxy_cache_valid 200 5m;
        add_header X-Cache-Status \$upstream_cache_status;
    }

    error_page 502 503 504 /50x.html;
    location = /50x.html {
        root /usr/share/nginx/html;
        internal;
    }
}
SSL_EOF
)
else
    SSL_REDIRECT_BLOCK="# SSL nicht verfügbar — HTTP only"
    SSL_SERVER_BLOCK=""
    log_warn "SSL nicht konfiguriert — nur HTTP-Config wird generiert."
fi

# Config schreiben
cat > "$REGISTRY_CONF" <<CONF_EOF
# ToolBoxV2 Registry — Nginx Config
# Domain:  $REGISTRY_DOMAIN
# Backend: $UPSTREAM_ADDR (Port $REGISTRY_PORT)
# SSL:     ${CERT_SOURCE:-'nicht konfiguriert'}
# Generiert: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Regenerieren: bash setup_registry.sh --registry-domain $REGISTRY_DOMAIN --mode $SETUP_MODE

upstream registry_${REGISTRY_PORT} {
    server $UPSTREAM_ADDR;
    keepalive 32;
}

# Rate-Limit Zonen (eindeutige Namen pro Port)
limit_req_zone \$binary_remote_addr zone=registry_api_${REGISTRY_PORT}:10m rate=10r/s;
limit_req_zone \$binary_remote_addr zone=registry_dl_${REGISTRY_PORT}:10m  rate=5r/s;

server {
    listen 80;
    server_name $REGISTRY_DOMAIN;
    $SSL_REDIRECT_BLOCK
}
$SSL_SERVER_BLOCK
CONF_EOF

log_ok "Config geschrieben: $REGISTRY_CONF"

# =============================================================================
# SCHRITT 5 — Site aktivieren + Nginx reload
# =============================================================================
log_section "Schritt 5: Site aktivieren"

mkdir -p "$NGINX_BOX_ENABLED"
SYMLINK_DST="$NGINX_BOX_ENABLED/registry_$REGISTRY_DOMAIN"
ln -sf "$REGISTRY_CONF" "$SYMLINK_DST"
log_ok "Symlink: $SYMLINK_DST → $REGISTRY_CONF"

if nginx -t 2>/dev/null; then
    nginx -s reload
    log_ok "Nginx erfolgreich neu geladen."
else
    nginx -t  # Zeigt den echten Fehler
    log_error "Nginx config ungültig."
fi

# =============================================================================
# SCHRITT 6 — tb manifest set registry.url
# =============================================================================
log_section "Schritt 6: Manifest aktualisieren"

TB_CMD=""
for candidate in \
    "/opt/toolboxv2/.venv/bin/tb" \
    "$(command -v tb 2>/dev/null || echo '')" \
    "/usr/local/bin/tb"; do
    [[ -x "$candidate" ]] && TB_CMD="$candidate" && break
done

REGISTRY_URL="$( [[ -n "$CERT_PATH" && -f "$CERT_PATH" ]] && echo "https" || echo "http" )://$REGISTRY_DOMAIN"

if [[ -n "$TB_CMD" ]]; then
    TB_RUNNER="${SUDO_USER:-toolbox}"
    sudo -u "$TB_RUNNER" "$TB_CMD" manifest set registry.url "$REGISTRY_URL" || \
        log_warn "Manifest-Update fehlgeschlagen — manuell: tb manifest set registry.url $REGISTRY_URL"
    log_ok "registry.url = $REGISTRY_URL"
else
    log_warn "tb nicht gefunden — manuell ausführen:"
    echo -e "  ${C_CYAN}tb manifest set registry.url $REGISTRY_URL${C_RESET}"
fi

# Certbot auto-renew (falls neu ausgestellt)
if [[ "${CERT_SOURCE:-}" == *"neu ausgestellt"* ]]; then
    if ! crontab -l 2>/dev/null | grep -q certbot; then
        (crontab -l 2>/dev/null; echo "0 3 * * * certbot renew --quiet && nginx -s reload") | crontab -
        log_ok "Certbot auto-renew cron eingerichtet."
    fi
fi

# =============================================================================
# ZUSAMMENFASSUNG
# =============================================================================
echo ""
echo -e "${C_BOLD}${C_GREEN}══════════════════════════════════════════${C_RESET}"
echo -e "${C_BOLD}${C_GREEN}  Registry Setup abgeschlossen!${C_RESET}"
echo -e "${C_BOLD}${C_GREEN}══════════════════════════════════════════${C_RESET}"
echo ""
echo -e "  ${C_BOLD}Registry-URL:${C_RESET}    $REGISTRY_URL"
echo -e "  ${C_BOLD}Backend:${C_RESET}         $UPSTREAM_ADDR"
echo -e "  ${C_BOLD}Modus:${C_RESET}           $SETUP_MODE"
echo -e "  ${C_BOLD}SSL:${C_RESET}             ${CERT_SOURCE:-'⚠️  nicht konfiguriert'}"
echo -e "  ${C_BOLD}Nginx-Config:${C_RESET}    $REGISTRY_CONF"
echo ""
echo -e "${C_BOLD}Nächste Schritte:${C_RESET}"

if [[ "$SETUP_MODE" == "same-machine" ]]; then
    echo -e "  1. Registry starten:  ${C_CYAN}tb registry start --host 127.0.0.1 --port $REGISTRY_PORT --background${C_RESET}"
    echo -e "  2. Health prüfen:     ${C_CYAN}curl $REGISTRY_URL/health${C_RESET}"
elif [[ "$SETUP_MODE" == "remote" ]]; then
    echo -e "  1. Auf $REMOTE_HOST:  ${C_CYAN}tb registry start --host 0.0.0.0 --port $REGISTRY_PORT --background${C_RESET}"
    echo -e "  2. Firewall prüfen:   Port $REGISTRY_PORT von diesem Server erreichbar?"
    echo -e "  3. Health prüfen:     ${C_CYAN}curl $REGISTRY_URL/health${C_RESET}"
elif [[ "$SETUP_MODE" == "new-user" ]]; then
    echo -e "  1. Als $REGISTRY_USER: ${C_CYAN}sudo -u $REGISTRY_USER tb registry start --host 127.0.0.1 --port $REGISTRY_PORT --background${C_RESET}"
    echo -e "  2. Health prüfen:     ${C_CYAN}curl $REGISTRY_URL/health${C_RESET}"
fi

if [[ -z "$CERT_PATH" || ! -f "$CERT_PATH" ]]; then
    echo ""
    echo -e "  ${C_YELLOW}DNS-Konfiguration erforderlich:${C_RESET}"
    echo -e "  ${C_CYAN}$REGISTRY_DOMAIN.  IN  CNAME  $MAIN_DOMAIN.${C_RESET}"
    SERVER_IP=$(curl -s --max-time 5 https://api.ipify.org 2>/dev/null || echo "<server-ip>")
    echo -e "  oder:  ${C_CYAN}$REGISTRY_DOMAIN.  IN  A  $SERVER_IP${C_RESET}"
    echo ""
    echo -e "  Nach DNS-Propagation (~5-60 min) SSL nachholen:"
    echo -e "  ${C_CYAN}certbot certonly --webroot -w $CERTBOT_WEBROOT -d $REGISTRY_DOMAIN --email $EMAIL --agree-tos -n && nginx -s reload${C_RESET}"
fi
echo ""
