#!/bin/bash
# TB-Registry Uninstall Script
# Usage: sudo ./uninstall.sh [--keep-data] [--with-minio]

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

INSTALL_DIR="/opt/tb-registry"
MINIO_DIR="/opt/minio"
LOG_DIR="/var/log/tb-registry"

KEEP_DATA=false
WITH_MINIO=false

for arg in "$@"; do
    case $arg in
        --keep-data) KEEP_DATA=true ;;
        --with-minio) WITH_MINIO=true ;;
    esac
done

log() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

[[ $EUID -ne 0 ]] && { echo -e "${RED}[ERROR]${NC} Run as root"; exit 1; }

log "=== TB-Registry Uninstall ==="

# Stop and disable services
log "Stopping services..."
systemctl stop tb-registry 2>/dev/null || true
systemctl disable tb-registry 2>/dev/null || true
rm -f /etc/systemd/system/tb-registry.service

if [[ "$WITH_MINIO" == true ]]; then
    systemctl stop minio 2>/dev/null || true
    systemctl disable minio 2>/dev/null || true
    rm -f /etc/systemd/system/minio.service
fi

systemctl daemon-reload

# Remove files
if [[ "$KEEP_DATA" == true ]]; then
    log "Keeping data directory..."
    find ${INSTALL_DIR} -mindepth 1 -maxdepth 1 ! -name 'data' -exec rm -rf {} +
else
    log "Removing all files..."
    rm -rf ${INSTALL_DIR}
    if [[ "$WITH_MINIO" == true ]]; then
        rm -rf ${MINIO_DIR}
    fi
fi

rm -rf ${LOG_DIR}

# Remove users
userdel tbregistry 2>/dev/null || true
[[ "$WITH_MINIO" == true ]] && userdel minio 2>/dev/null || true

log "=== Uninstall Complete ==="

