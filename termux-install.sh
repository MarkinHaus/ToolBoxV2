#!/data/data/com.termux/files/usr/bin/bash
# ============================================================================
# ToolBoxV2 - Termux Android Installer
# ============================================================================
# Dieses Script installiert die Python Core von ToolBoxV2 auf Android
# via Termux. Es richtet eine vollständige Entwicklungsumgebung ein.
#
# Verwendung:
#   curl -sSL https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/master/termux-install.sh | bash
#
# Oder manuell:
#   wget -qO install.sh https://raw.githubusercontent.com/MarkinHaus/ToolBoxV2/master/termux-install.sh
#   chmod +x install.sh
#   ./install.sh
#
# Optionen:
#   --full    Installiert zusätzlich Nuitka für native Builds
#   --server  Installiert zusätzlich den Rust Server
#   --dev     Installiert Entwickler-Abhängigkeiten
# ============================================================================

set -e

# Farben
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Argumente parsen
INSTALL_FULL=false
INSTALL_SERVER=false
INSTALL_DEV=false

for arg in "$@"; do
    case $arg in
        --full)
            INSTALL_FULL=true
            ;;
        --server)
            INSTALL_SERVER=true
            ;;
        --dev)
            INSTALL_DEV=true
            ;;
        --help|-h)
            echo "ToolBoxV2 Termux Installer"
            echo ""
            echo "Optionen:"
            echo "  --full    Installiert Nuitka für native Builds"
            echo "  --server  Installiert den Rust Server"
            echo "  --dev     Installiert Entwickler-Tools"
            echo "  --help    Zeigt diese Hilfe"
            exit 0
            ;;
    esac
done

# Banner
clear
echo -e "${CYAN}"
cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║    ████████╗ ██████╗  ██████╗ ██╗     ██████╗  ██████╗ ██╗  ██╗██╗   ██╗██████╗  ║
║    ╚══██╔══╝██╔═══██╗██╔═══██╗██║     ██╔══██╗██╔═══██╗╚██╗██╔╝██║   ██║╚════██╗ ║
║       ██║   ██║   ██║██║   ██║██║     ██████╔╝██║   ██║ ╚███╔╝ ██║   ██║ █████╔╝ ║
║       ██║   ██║   ██║██║   ██║██║     ██╔══██╗██║   ██║ ██╔██╗ ╚██╗ ██╔╝██╔═══╝  ║
║       ██║   ╚██████╔╝╚██████╔╝███████╗██████╔╝╚██████╔╝██╔╝ ██╗ ╚████╔╝ ███████╗ ║
║       ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝  ╚═══╝  ╚══════╝ ║
║                                                                                  ║
║                           Termux Android Installer                               ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Prüfe Termux
echo -e "${YELLOW}[CHECK] Prüfe Termux-Umgebung...${NC}"
if [ ! -d "/data/data/com.termux" ]; then
    echo -e "${RED}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  FEHLER: Dieses Script muss in Termux ausgeführt werden!        ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Bitte installiere Termux von F-Droid (empfohlen):"
    echo "  https://f-droid.org/packages/com.termux/"
    echo ""
    echo "WICHTIG: Die Play Store Version ist veraltet und funktioniert nicht!"
    exit 1
fi
echo -e "${GREEN}✓ Termux erkannt${NC}"

# Prüfe Architektur
ARCH=$(uname -m)
echo -e "${BLUE}  Architektur: ${ARCH}${NC}"

# Speicher prüfen
AVAILABLE_SPACE=$(df -h /data/data/com.termux/files/home | awk 'NR==2 {print $4}')
echo -e "${BLUE}  Verfügbarer Speicher: ${AVAILABLE_SPACE}${NC}"
echo ""

# Schritt 1: Repositories aktualisieren
echo -e "${YELLOW}[1/7] Aktualisiere Paket-Repositories...${NC}"
pkg update -y 2>/dev/null || apt update -y
pkg upgrade -y 2>/dev/null || apt upgrade -y
echo -e "${GREEN}✓ Repositories aktualisiert${NC}"
echo ""

# Schritt 2: Basis-Pakete installieren
echo -e "${YELLOW}[2/7] Installiere Basis-Pakete...${NC}"
pkg install -y \
    python \
    python-pip \
    git \
    wget \
    curl \
    openssl \
    libffi \
    2>/dev/null || apt install -y python python-pip git wget curl openssl libffi-dev

echo -e "${GREEN}✓ Basis-Pakete installiert${NC}"
echo ""

# Schritt 3: Build-Tools installieren
echo -e "${YELLOW}[3/7] Installiere Build-Tools...${NC}"
pkg install -y \
    clang \
    make \
    cmake \
    binutils \
    pkg-config \
    2>/dev/null || apt install -y clang make cmake binutils pkg-config

# Zusätzliche Termux-spezifische Tools
pkg install -y patchelf ccache ldd termux-elf-cleaner 2>/dev/null || true

echo -e "${GREEN}✓ Build-Tools installiert${NC}"
echo ""

# Schritt 4: Python-Umgebung einrichten
echo -e "${YELLOW}[4/7] Richte Python-Umgebung ein...${NC}"
python -m pip install --upgrade pip
pip install wheel setuptools

# Python Version anzeigen
PYTHON_VERSION=$(python --version 2>&1)
echo -e "${BLUE}  ${PYTHON_VERSION}${NC}"
echo -e "${GREEN}✓ Python-Umgebung bereit${NC}"
echo ""

# Schritt 5: ToolBoxV2 installieren
echo -e "${YELLOW}[5/7] Installiere ToolBoxV2...${NC}"

# Versuche verschiedene Installationsmethoden
if pip install ToolBoxV2 2>/dev/null; then
    echo -e "${GREEN}✓ ToolBoxV2 von PyPI installiert${NC}"
elif pip install "ToolBoxV2[isaa]" 2>/dev/null; then
    echo -e "${GREEN}✓ ToolBoxV2 mit ISAA von PyPI installiert${NC}"
else
    echo -e "${YELLOW}  PyPI-Installation fehlgeschlagen, versuche GitHub...${NC}"
    if pip install git+https://github.com/MarkinHaus/ToolBoxV2.git 2>/dev/null; then
        echo -e "${GREEN}✓ ToolBoxV2 von GitHub installiert${NC}"
    else
        echo -e "${RED}✗ Installation fehlgeschlagen${NC}"
        echo ""
        echo "Versuche manuelle Installation:"
        echo "  git clone https://github.com/MarkinHaus/ToolBoxV2.git"
        echo "  cd ToolBoxV2"
        echo "  pip install -e ."
        exit 1
    fi
fi
echo ""

# Schritt 6: Optional - Erweiterte Installation
if [ "$INSTALL_FULL" = true ]; then
    echo -e "${YELLOW}[6/7] Installiere Nuitka für native Builds...${NC}"
    pip install nuitka zstandard ordered-set 2>/dev/null || true
    echo -e "${GREEN}✓ Nuitka installiert${NC}"
else
    echo -e "${BLUE}[6/7] Überspringe Nuitka (verwende --full für Installation)${NC}"
fi
echo ""

if [ "$INSTALL_SERVER" = true ]; then
    echo -e "${YELLOW}[6b/7] Installiere Rust für Server-Build...${NC}"
    pkg install -y rust 2>/dev/null || apt install -y rust

    # Klone und baue Server
    if [ ! -d "$HOME/ToolBoxV2" ]; then
        git clone --depth 1 https://github.com/MarkinHaus/ToolBoxV2.git "$HOME/ToolBoxV2"
    fi

    cd "$HOME/ToolBoxV2/toolboxv2/src-core"
    cargo build --release 2>/dev/null && {
        mkdir -p "$HOME/.local/bin"
        cp target/release/simple-core-server "$HOME/.local/bin/"
        echo -e "${GREEN}✓ Rust Server gebaut und installiert${NC}"
    } || {
        echo -e "${YELLOW}⚠ Server-Build fehlgeschlagen (optional)${NC}"
    }
    cd "$HOME"
fi

if [ "$INSTALL_DEV" = true ]; then
    echo -e "${YELLOW}[6c/7] Installiere Entwickler-Tools...${NC}"
    pip install pytest black ruff mypy ipython 2>/dev/null || true
    pkg install -y neovim tmux 2>/dev/null || true
    echo -e "${GREEN}✓ Dev-Tools installiert${NC}"
fi
echo ""

# Schritt 7: ToolBoxV2 initialisieren
echo -e "${YELLOW}[7/7] Initialisiere ToolBoxV2...${NC}"
tb -init main 2>/dev/null || python -m toolboxv2 -init main 2>/dev/null || {
    echo -e "${YELLOW}⚠ Automatische Initialisierung übersprungen${NC}"
    echo "  Führe später aus: tb -init main"
}
echo ""

# PATH konfigurieren
echo -e "${YELLOW}Konfiguriere PATH...${NC}"
mkdir -p "$HOME/.local/bin"

# Füge zu .bashrc hinzu falls nicht vorhanden
if ! grep -q "HOME/.local/bin" "$HOME/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi

# Erstelle tb Wrapper falls nötig
if ! command -v tb &> /dev/null; then
    cat > "$HOME/.local/bin/tb" << 'WRAPPER'
#!/data/data/com.termux/files/usr/bin/bash
python -m toolboxv2 "$@"
WRAPPER
    chmod +x "$HOME/.local/bin/tb"
fi

# Termux:Widget Shortcut erstellen
echo -e "${YELLOW}Erstelle Termux Shortcuts...${NC}"
mkdir -p "$HOME/.shortcuts"
cat > "$HOME/.shortcuts/ToolBoxV2" << 'SHORTCUT'
#!/data/data/com.termux/files/usr/bin/bash
cd ~
tb
SHORTCUT
chmod +x "$HOME/.shortcuts/ToolBoxV2"

cat > "$HOME/.shortcuts/TB-Server" << 'SHORTCUT'
#!/data/data/com.termux/files/usr/bin/bash
cd ~
tb api start
SHORTCUT
chmod +x "$HOME/.shortcuts/TB-Server"

# Termux:Boot Autostart (optional)
mkdir -p "$HOME/.termux/boot"
cat > "$HOME/.termux/boot/toolboxv2" << 'BOOT'
#!/data/data/com.termux/files/usr/bin/bash
# Uncomment to enable autostart:
# termux-wake-lock
# tb api start --background
BOOT
chmod +x "$HOME/.termux/boot/toolboxv2"

# Speichere Installationsinfo
mkdir -p "$HOME/.toolboxv2"
cat > "$HOME/.toolboxv2/install.info" << INFO
Installation: $(date)
Python: $(python --version 2>&1)
Architecture: $(uname -m)
Installer: termux-install.sh
Options: full=$INSTALL_FULL server=$INSTALL_SERVER dev=$INSTALL_DEV
INFO

# Abschluss
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                                  ║${NC}"
echo -e "${GREEN}║            Installation erfolgreich abgeschlossen!               ║${NC}"
echo -e "${GREEN}║                                                                  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Verwendung:${NC}"
echo "  tb              - Startet ToolBoxV2"
echo "  tb -h           - Zeigt Hilfe"
echo "  tb api start    - Startet den API Server"
echo "  tb --test       - Führt Tests aus"
echo ""
echo -e "${CYAN}Termux Widgets:${NC}"
echo "  Installiere 'Termux:Widget' von F-Droid für Home-Screen Shortcuts"
echo ""
echo -e "${CYAN}Dokumentation:${NC}"
echo "  https://github.com/MarkinHaus/ToolBoxV2"
echo ""
echo -e "${YELLOW}Hinweis: Starte ein neues Terminal oder führe aus:${NC}"
echo "  source ~/.bashrc"
echo ""

# Frage ob Terminal neu geladen werden soll
read -p "Terminal jetzt neu laden? (j/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Jj]$ ]]; then
    exec bash
fi
