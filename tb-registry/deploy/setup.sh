#!/bin/bash
# TB-Registry Auto-Setup Script for Linux
# Usage: sudo ./setup.sh [--with-minio] [--dev]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="/opt/tb-registry"
MINIO_DIR="/opt/minio"
LOG_DIR="/var/log/tb-registry"
DATA_DIR="${INSTALL_DIR}/data"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
WITH_MINIO=false
DEV_MODE=false
for arg in "$@"; do
    case $arg in
        --with-minio) WITH_MINIO=true ;;
        --dev) DEV_MODE=true ;;
        --help|-h)
            echo "Usage: sudo ./setup.sh [--with-minio] [--dev]"
            echo "  --with-minio  Install and configure MinIO"
            echo "  --dev         Development mode (skip service installation)"
            exit 0
            ;;
    esac
done

log() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Check root
[[ $EUID -ne 0 ]] && error "This script must be run as root (use sudo)"

log "=== TB-Registry Setup Script ==="
log "Install directory: ${INSTALL_DIR}"

# Install dependencies
log "Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    apt-get update -qq
    apt-get install -y -qq curl wget python3 python3-pip python3-venv
elif command -v dnf &> /dev/null; then
    dnf install -y -q curl wget python3 python3-pip
elif command -v yum &> /dev/null; then
    yum install -y -q curl wget python3 python3-pip
else
    warn "Unknown package manager. Please install: curl, wget, python3, python3-pip"
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    log "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create user and directories
log "Creating user and directories..."
id -u tbregistry &>/dev/null || useradd -r -s /bin/false -d ${INSTALL_DIR} tbregistry
mkdir -p ${INSTALL_DIR} ${DATA_DIR} ${LOG_DIR}

# Copy registry files
log "Copying registry files..."
cp -r ${REPO_DIR}/registry ${INSTALL_DIR}/
cp ${REPO_DIR}/pyproject.toml ${INSTALL_DIR}/
cp ${REPO_DIR}/uv.lock ${INSTALL_DIR}/ 2>/dev/null || true

# Create virtual environment and install dependencies
log "Setting up Python environment..."
cd ${INSTALL_DIR}
uv venv
uv sync

# Create default .env if not exists
if [[ ! -f ${INSTALL_DIR}/.env ]]; then
    log "Creating default .env configuration..."
    cat > ${INSTALL_DIR}/.env << 'EOF'
# TB-Registry Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=4025
DATABASE_URL=sqlite:///./data/registry.db
DEBUG=false

# MinIO Configuration
MINIO_PRIMARY_ENDPOINT=localhost:9000
MINIO_PRIMARY_ACCESS_KEY=minioadmin
MINIO_PRIMARY_SECRET_KEY=minioadmin
MINIO_PRIMARY_BUCKET=tb-registry
MINIO_PRIMARY_SECURE=false

# Clerk Authentication (optional)
CLERK_SECRET_KEY=
CLERK_PUBLISHABLE_KEY=

# CORS
CORS_ORIGINS=["*"]
EOF
    warn "Please edit ${INSTALL_DIR}/.env with your configuration!"
fi

# Set permissions
chown -R tbregistry:tbregistry ${INSTALL_DIR} ${LOG_DIR}
chmod 600 ${INSTALL_DIR}/.env

# Install MinIO if requested
if [[ "$WITH_MINIO" == true ]]; then
    log "Installing MinIO..."
    mkdir -p ${MINIO_DIR}/data
    id -u minio &>/dev/null || useradd -r -s /bin/false -d ${MINIO_DIR} minio
    
    if ! command -v minio &> /dev/null; then
        wget -q https://dl.min.io/server/minio/release/linux-amd64/minio -O /usr/local/bin/minio
        chmod +x /usr/local/bin/minio
    fi
    
    chown -R minio:minio ${MINIO_DIR}
    
    if [[ "$DEV_MODE" == false ]]; then
        cp ${SCRIPT_DIR}/minio.service /etc/systemd/system/
        systemctl daemon-reload
        systemctl enable minio
        systemctl start minio
        log "MinIO started on http://localhost:9000 (console: :9001)"
    fi
fi

# Install systemd service
if [[ "$DEV_MODE" == false ]]; then
    log "Installing systemd service..."
    cp ${SCRIPT_DIR}/tb-registry.service /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable tb-registry
    log "Service installed. Start with: systemctl start tb-registry"
else
    log "Dev mode: Skipping service installation"
    log "Run manually: cd ${INSTALL_DIR} && uv run python -m registry"
fi

log "=== Setup Complete ==="
log "Configuration: ${INSTALL_DIR}/.env"
log "Logs: ${LOG_DIR}/"
log "Data: ${DATA_DIR}/"
echo ""
log "Next steps:"
echo "  1. Edit ${INSTALL_DIR}/.env with your settings"
echo "  2. Start the service: sudo systemctl start tb-registry"
echo "  3. Check status: sudo systemctl status tb-registry"
echo "  4. View logs: sudo journalctl -u tb-registry -f"

