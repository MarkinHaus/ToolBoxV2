#!/usr/bin/env bash
# =============================================================================
# ToolBoxV2 — Server Setup: Nginx + SSL (certbot) + System User
# =============================================================================
# Verwendung:
#   bash setup_tb_server.sh
#   bash setup_tb_server.sh --domain simplecore.app --email admin@simplecore.app
#
# Was dieses Script tut:
#   1. Prüft / installiert Abhängigkeiten (nginx, certbot, python3)
#   2. Legt einen dedizierten 'toolbox' System-User an (falls nicht vorhanden)
#   3. Generiert SSH-Key für den toolbox-User
#   4. Richtet Nginx-Verzeichnisse ein (box-available, box-enabled)
#   5. Setzt Domain + Email im tb-manifest.yaml
#   6. Läuft certbot (oder erkennt existierendes Zertifikat)
#   7. Ruft 'tb manifest apply' auf → NginxManager generiert Site-Config
#   8. Gibt Berechtigungen an toolbox-User
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Farben & Logging
# -----------------------------------------------------------------------------
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
TB_USER="toolbox"
TB_HOME="/opt/toolboxv2"
NGINX_BOX_AVAILABLE="/etc/nginx/box-available"
NGINX_BOX_ENABLED="/etc/nginx/box-enabled"
NGINX_SITES_AVAILABLE="/etc/nginx/sites-available"
NGINX_SITES_ENABLED="/etc/nginx/sites-enabled"
CERTBOT_WEBROOT="/var/lib/letsencrypt/webroot"

DOMAIN=""
EMAIL=""
HTTP_PORT=8000
WS_PORT=8100
SKIP_CERTBOT=false
DRY_RUN=false

# -----------------------------------------------------------------------------
# Argumente
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --domain)   DOMAIN="$2"; shift 2 ;;
        --email)    EMAIL="$2";  shift 2 ;;
        --http-port) HTTP_PORT="$2"; shift 2 ;;
        --ws-port)   WS_PORT="$2"; shift 2 ;;
        --skip-certbot) SKIP_CERTBOT=true; shift ;;
        --dry-run)  DRY_RUN=true; shift ;;
        *) log_warn "Unbekanntes Argument: $1"; shift ;;
    esac
done

# -----------------------------------------------------------------------------
# Interaktive Eingabe wenn nötig
# -----------------------------------------------------------------------------
log_section "ToolBoxV2 Server Setup"
echo ""

if [[ -z "$DOMAIN" ]]; then
    read -r -p "$(echo -e "${C_BOLD}Domain (z.B. simplecore.app): ${C_RESET}")" DOMAIN
    [[ -z "$DOMAIN" ]] && log_error "Domain darf nicht leer sein."
fi

if [[ -z "$EMAIL" ]]; then
    read -r -p "$(echo -e "${C_BOLD}E-Mail für SSL-Zertifikat: ${C_RESET}")" EMAIL
    [[ -z "$EMAIL" ]] && log_error "E-Mail darf nicht leer sein."
fi

log_info "Domain:    $DOMAIN"
log_info "Email:     $EMAIL"
log_info "HTTP-Port: $HTTP_PORT"
log_info "WS-Port:   $WS_PORT"
log_info "TB-User:   $TB_USER"
log_info "TB-Home:   $TB_HOME"
echo ""

# Bestätigung
read -r -p "$(echo -e "${C_BOLD}Fortfahren? [Y/n]: ${C_RESET}")" confirm
[[ "${confirm,,}" == "n" ]] && echo "Abgebrochen." && exit 0

# -----------------------------------------------------------------------------
# Root-Check
# -----------------------------------------------------------------------------
[[ "$(id -u)" -ne 0 ]] && log_error "Script muss als root ausgeführt werden (sudo bash $0)"

# =============================================================================
# SCHRITT 1 — Abhängigkeiten prüfen / installieren
# =============================================================================
log_section "Schritt 1: Abhängigkeiten"

install_pkg() {
    if command -v apt-get &>/dev/null; then
        apt-get install -y -q "$@"
    elif command -v dnf &>/dev/null; then
        dnf install -y -q "$@"
    elif command -v yum &>/dev/null; then
        yum install -y -q "$@"
    else
        log_warn "Kein bekannter Paketmanager — bitte '$*' manuell installieren."
    fi
}

# Nginx
if ! command -v nginx &>/dev/null; then
    log_info "Installiere nginx..."
    install_pkg nginx
fi
log_ok "nginx $(nginx -v 2>&1 | grep -oP '[\d.]+')"

# Certbot
if ! command -v certbot &>/dev/null; then
    log_info "Installiere certbot..."
    install_pkg certbot python3-certbot-nginx
fi
log_ok "certbot $(certbot --version 2>&1 | grep -oP '[\d.]+')"

# Python3
if ! command -v python3 &>/dev/null; then
    log_info "Installiere python3..."
    install_pkg python3 python3-pip python3-venv
fi
log_ok "python3 $(python3 --version)"

# =============================================================================
# SCHRITT 2 — System-User anlegen
# =============================================================================
log_section "Schritt 2: System-User '$TB_USER'"

if id "$TB_USER" &>/dev/null; then
    log_ok "User '$TB_USER' existiert bereits."
else
    log_info "Lege User '$TB_USER' an..."
    useradd \
        --system \
        --create-home \
        --home-dir "$TB_HOME" \
        --shell /bin/bash \
        --comment "ToolBoxV2 Service User" \
        "$TB_USER"
    log_ok "User '$TB_USER' erstellt (Home: $TB_HOME)"
fi

# www-data Gruppe → nginx kann statische Dateien lesen
usermod -aG www-data "$TB_USER" 2>/dev/null || true

# =============================================================================
# SCHRITT 3 — SSH-Key für toolbox-User
# =============================================================================
log_section "Schritt 3: SSH-Key"

TB_SSH_DIR="$TB_HOME/.ssh"
TB_KEY_PATH="$TB_SSH_DIR/id_ed25519"

if [[ ! -f "$TB_KEY_PATH" ]]; then
    mkdir -p "$TB_SSH_DIR"
    chmod 700 "$TB_SSH_DIR"
    ssh-keygen -t ed25519 -C "toolbox@$DOMAIN" -f "$TB_KEY_PATH" -N ""
    cat "$TB_KEY_PATH.pub" >> "$TB_SSH_DIR/authorized_keys"
    chmod 600 "$TB_SSH_DIR/authorized_keys"
    chown -R "$TB_USER:$TB_USER" "$TB_SSH_DIR"
    log_ok "SSH-Key generiert: $TB_KEY_PATH"
    echo ""
    echo -e "${C_BOLD}Öffentlicher Schlüssel (für authorized_keys auf anderen Servern):${C_RESET}"
    echo -e "${C_CYAN}$(cat "$TB_KEY_PATH.pub")${C_RESET}"
    echo ""
    log_warn "Privaten Key sicher aufbewahren: $TB_KEY_PATH"
else
    log_ok "SSH-Key existiert bereits: $TB_KEY_PATH"
fi

# =============================================================================
# SCHRITT 4 — Nginx-Verzeichnisse + Berechtigungen
# =============================================================================
log_section "Schritt 4: Nginx-Verzeichnisse"

for dir in "$NGINX_BOX_AVAILABLE" "$NGINX_BOX_ENABLED" \
           "$NGINX_SITES_AVAILABLE" "$NGINX_SITES_ENABLED"; do
    mkdir -p "$dir"
    # toolbox-User darf box-* schreiben (für tb manifest apply)
    if [[ "$dir" == *"box"* ]]; then
        chown root:"$TB_USER" "$dir"
        chmod 775 "$dir"
        log_ok "$dir  (toolbox: rw)"
    else
        log_ok "$dir  (root: rw)"
    fi
done

# Nginx-Logs
mkdir -p /var/log/nginx
chown -R www-data:adm /var/log/nginx 2>/dev/null || true

# Webroot für certbot ACME-Challenge
mkdir -p "$CERTBOT_WEBROOT"
chown -R www-data:www-data "$CERTBOT_WEBROOT"

# Nginx main.conf: box-enabled einbinden (falls noch nicht)
if ! grep -q "box-enabled" /etc/nginx/nginx.conf 2>/dev/null; then
    log_info "Füge box-enabled include zu nginx.conf hinzu..."
    # Füge vor dem letzten } im http-Block ein
    sed -i '/^}/{ /http/!{ s|^}|    include /etc/nginx/box-enabled/*;\n}| } }' \
        /etc/nginx/nginx.conf 2>/dev/null || \
        log_warn "nginx.conf manuell anpassen: 'include /etc/nginx/box-enabled/*;' im http-Block"
fi

# =============================================================================
# SCHRITT 5 — tb-manifest.yaml aktualisieren
# =============================================================================
log_section "Schritt 5: Manifest konfigurieren"

# Suche tb-Executable (im venv des toolbox-Users oder global)
TB_CMD=""
for candidate in \
    "$TB_HOME/.venv/bin/tb" \
    "$TB_HOME/venv/bin/tb" \
    "$(command -v tb 2>/dev/null || echo '')" \
    "/usr/local/bin/tb"; do
    [[ -x "$candidate" ]] && TB_CMD="$candidate" && break
done

if [[ -z "$TB_CMD" ]]; then
    log_warn "tb-Executable nicht gefunden — Manifest-Update wird übersprungen."
    log_warn "Bitte nach der TB-Installation erneut ausführen: bash $0 --domain $DOMAIN --skip-certbot"
else
    log_info "tb gefunden: $TB_CMD"
    sudo -u "$TB_USER" "$TB_CMD" manifest set nginx.server_name "$DOMAIN"    || true
    sudo -u "$TB_USER" "$TB_CMD" manifest set nginx.enabled true              || true
    sudo -u "$TB_USER" "$TB_CMD" manifest set nginx.ssl_enabled true          || true
    sudo -u "$TB_USER" "$TB_CMD" manifest set app.environment production      || true
    log_ok "Manifest aktualisiert."
fi

# =============================================================================
# SCHRITT 6 — SSL: Existierendes Zertifikat erkennen ODER certbot
# =============================================================================
log_section "Schritt 6: SSL-Zertifikat"

CERT_PATH="/etc/letsencrypt/live/$DOMAIN/fullchain.pem"
KEY_PATH="/etc/letsencrypt/live/$DOMAIN/privkey.pem"
CERT_EXISTS=false

if [[ -f "$CERT_PATH" && -f "$KEY_PATH" ]]; then
    log_ok "Existierendes Let's Encrypt Zertifikat gefunden:"
    log_info "  Cert: $CERT_PATH"
    log_info "  Key:  $KEY_PATH"
    # Ablaufdatum
    expiry=$(openssl x509 -enddate -noout -in "$CERT_PATH" 2>/dev/null | cut -d= -f2)
    log_info "  Läuft ab: $expiry"
    CERT_EXISTS=true
elif [[ "$SKIP_CERTBOT" == "true" ]]; then
    log_warn "Certbot übersprungen (--skip-certbot). SSL nicht konfiguriert."
else
    log_info "Kein Zertifikat für $DOMAIN gefunden — starte certbot..."

    # Minimale HTTP-Config zum Verifizieren (certbot-webroot)
    TEMP_CONF="/etc/nginx/sites-available/certbot_$DOMAIN"
    cat > "$TEMP_CONF" <<NGINX_EOF
server {
    listen 80;
    server_name $DOMAIN www.$DOMAIN;
    location /.well-known/acme-challenge/ {
        root $CERTBOT_WEBROOT;
    }
    location / { return 444; }
}
NGINX_EOF
    ln -sf "$TEMP_CONF" "/etc/nginx/sites-enabled/certbot_$DOMAIN"
    nginx -t && nginx -s reload

    certbot certonly \
        --webroot \
        --webroot-path "$CERTBOT_WEBROOT" \
        --domain "$DOMAIN" \
        --domain "www.$DOMAIN" \
        --email "$EMAIL" \
        --agree-tos \
        --non-interactive \
        && CERT_EXISTS=true \
        || log_warn "Certbot fehlgeschlagen. SSL manuell einrichten."

    # Temp-Config entfernen
    rm -f "/etc/nginx/sites-enabled/certbot_$DOMAIN" "$TEMP_CONF"
fi

# Certbot auto-renew (systemd timer bevorzugt, cron als Fallback)
if [[ "$CERT_EXISTS" == "true" ]]; then
    if systemctl list-timers 2>/dev/null | grep -q certbot; then
        log_ok "Certbot auto-renew (systemd timer) ist aktiv."
    else
        if ! crontab -l 2>/dev/null | grep -q certbot; then
            (crontab -l 2>/dev/null; echo "0 3 * * * certbot renew --quiet && nginx -s reload") | crontab -
            log_ok "Certbot cron-Job eingerichtet (täglich 3:00 Uhr)."
        else
            log_ok "Certbot cron-Job existiert bereits."
        fi
    fi
fi

# =============================================================================
# SCHRITT 7 — tb manifest apply → NginxManager generiert Site-Config
# =============================================================================
log_section "Schritt 7: Site-Config generieren"

if [[ -n "$TB_CMD" ]]; then
    sudo -u "$TB_USER" "$TB_CMD" manifest apply || \
        log_warn "manifest apply fehlgeschlagen — bitte manuell ausführen: sudo -u $TB_USER tb manifest apply"

    # Site-Config aktivieren (Symlink)
    SITE_SRC="$NGINX_BOX_AVAILABLE/toolbox"
    SITE_DST="$NGINX_BOX_ENABLED/toolbox"
    if [[ -f "$SITE_SRC" ]]; then
        ln -sf "$SITE_SRC" "$SITE_DST"
        log_ok "Site aktiviert: $SITE_DST → $SITE_SRC"
    else
        log_warn "Site-Config nicht gefunden: $SITE_SRC"
        log_warn "tb manager nginx-config manuell aufrufen."
    fi
else
    log_warn "tb nicht verfügbar — Schritt 7 nach TB-Installation wiederholen."
fi

# =============================================================================
# SCHRITT 8 — Nginx testen + reload
# =============================================================================
log_section "Schritt 8: Nginx reload"

if nginx -t 2>/dev/null; then
    nginx -s reload
    log_ok "Nginx erfolgreich neu geladen."
else
    log_warn "Nginx config ungültig — bitte manuell prüfen: nginx -t"
fi

# =============================================================================
# ZUSAMMENFASSUNG
# =============================================================================
echo ""
echo -e "${C_BOLD}${C_GREEN}══════════════════════════════════════════${C_RESET}"
echo -e "${C_BOLD}${C_GREEN}  Setup abgeschlossen!${C_RESET}"
echo -e "${C_BOLD}${C_GREEN}══════════════════════════════════════════${C_RESET}"
echo ""
echo -e "  ${C_BOLD}Domain:${C_RESET}       https://$DOMAIN"
echo -e "  ${C_BOLD}TB-User:${C_RESET}      $TB_USER  (Home: $TB_HOME)"
echo -e "  ${C_BOLD}SSH-Key:${C_RESET}      $TB_KEY_PATH"
echo -e "  ${C_BOLD}Nginx-Dirs:${C_RESET}   $NGINX_BOX_AVAILABLE"
echo -e "  ${C_BOLD}SSL:${C_RESET}          $( [[ "$CERT_EXISTS" == "true" ]] && echo "✅ aktiv" || echo "⚠️  nicht konfiguriert" )"
echo ""
echo -e "${C_BOLD}Nächste Schritte:${C_RESET}"
echo -e "  1. TB als toolbox-User starten:  ${C_CYAN}sudo -u $TB_USER tb workers${C_RESET}"
echo -e "  2. Status prüfen:                ${C_CYAN}sudo -u $TB_USER tb status${C_RESET}"
echo -e "  3. Manifest anpassen:            ${C_CYAN}sudo -u $TB_USER tb manifest show${C_RESET}"
if [[ "$CERT_EXISTS" == "false" ]]; then
    echo ""
    echo -e "  ${C_YELLOW}SSL fehlt — nach DNS-Propagation:${C_RESET}"
    echo -e "  ${C_CYAN}certbot certonly --nginx -d $DOMAIN -d www.$DOMAIN --email $EMAIL --agree-tos -n${C_RESET}"
    echo -e "  ${C_CYAN}bash $0 --domain $DOMAIN --skip-certbot${C_RESET}"
fi
echo ""
