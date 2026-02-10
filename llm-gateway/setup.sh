#!/bin/bash
set -e

# LLM Gateway Setup Script for Linux/macOS
# This script sets up the Python environment and optionally installs Ollama

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "LLM Gateway Setup"
echo "========================================="
echo

# Check Python version
echo "[1/5] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "Please install Python 3.12 or later and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.12"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "ERROR: Python $REQUIRED_VERSION or later is required."
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION detected"
echo

# Create virtual environment
echo "[2/5] Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo

# Activate virtual environment and install dependencies
echo "[3/5] Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip > /dev/null
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo

# Check for Ollama
echo "[4/5] Checking for Ollama..."
if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>&1 | head -n1 || echo "unknown")
    echo "✓ Ollama is already installed: $OLLAMA_VERSION"
else
    echo "Ollama is not installed."
    read -p "Would you like to install Ollama now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        echo "✓ Ollama installed"
    else
        echo "Skipping Ollama installation."
        echo "You can install it later with: curl -fsSL https://ollama.com/install.sh | sh"
    fi
fi
echo

# Generate initial config if it doesn't exist
echo "[5/5] Setting up configuration..."
mkdir -p data

if [ -f "data/config.json" ]; then
    echo "Configuration file already exists. Skipping generation."
else
    # Generate a random admin key
    ADMIN_KEY=$(openssl rand -hex 32 2>/dev/null || cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 64 | head -n 1)

    cat > data/config.json << EOF
{
  "api_keys": {
    "$ADMIN_KEY": {
      "name": "admin",
      "role": "admin",
      "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
      "rate_limit": {
        "requests_per_minute": 100,
        "tokens_per_minute": 100000
      }
    }
  },
  "ollama": {
    "base_url": "http://localhost:11434",
    "timeout": 300
  },
  "server": {
    "host": "0.0.0.0",
    "port": 4000,
    "log_level": "info"
  }
}
EOF

    echo "✓ Configuration file created with admin API key"
    echo
    echo "========================================="
    echo "IMPORTANT: Save your admin API key!"
    echo "========================================="
    echo "$ADMIN_KEY"
    echo "========================================="
    echo
fi

echo "Setup complete!"
echo
echo "Next steps:"
echo "1. Start Ollama (if not already running): ollama serve"
echo "2. Pull a model: ollama pull llama3.2:latest"
echo "3. Activate the virtual environment: source venv/bin/activate"
echo "4. Run the gateway: python server.py"
echo "   Or with uvicorn: uvicorn server:app --host 0.0.0.0 --port 4000"
echo
echo "The gateway will be available at: http://localhost:4000"
echo "API documentation: http://localhost:4000/docs"
echo
echo "For Docker deployment:"
echo "  Bare mode (gateway only): docker compose up gateway"
echo "  Docker mode (both): docker compose --profile ollama up"
echo

deactivate 2>/dev/null || true
