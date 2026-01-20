#!/bin/bash
set -e

# === LLM Gateway Setup Script ===
# For Ryzen CPU (12 cores, 48GB RAM, no GPU)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
MODELS_DIR="$DATA_DIR/models"
BUILD_DIR="$SCRIPT_DIR/build"

echo "=== LLM Gateway Setup ==="
echo "Script directory: $SCRIPT_DIR"

# Create directories
mkdir -p "$DATA_DIR" "$MODELS_DIR" "$BUILD_DIR" "$SCRIPT_DIR/static"

# === 1. Install dependencies ===
echo "[1/5] Installing dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    libcurl4-openssl-dev \
    python3 \
    python3-pip \
    python3-venv \
    jq

# === 2. Build llama.cpp ===
echo "[2/5] Building llama.cpp..."
cd "$BUILD_DIR"

if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git
fi

cd llama.cpp
git pull

# Build with CPU optimizations for Ryzen (AVX2)
cmake -B build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=ON \
    -DGGML_CPU_ARM_ARCH=native \
    -DLLAMA_CURL=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc) --target llama-server llama-cli

# Copy binaries
cp build/bin/llama-server "$SCRIPT_DIR/llama-server"
cp build/bin/llama-cli "$SCRIPT_DIR/llama-cli"

# === 3. Build whisper.cpp ===
echo "[3/5] Building whisper.cpp..."
cd "$BUILD_DIR"

if [ ! -d "whisper.cpp" ]; then
    git clone https://github.com/ggerganov/whisper.cpp.git
fi

cd whisper.cpp
git pull

cmake -B build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc) --target whisper-server

cp build/bin/whisper-server "$SCRIPT_DIR/whisper-server"

# === 4. Setup Python environment ===
echo "[4/5] Setting up Python environment..."
cd "$SCRIPT_DIR"

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install \
    fastapi \
    uvicorn[standard] \
    httpx \
    aiosqlite \
    psutil \
    huggingface_hub \
    pydantic \
    python-multipart \
    passlib[bcrypt]

# === 5. Create initial config ===
echo "[5/5] Creating initial configuration..."

if [ ! -f "$DATA_DIR/config.json" ]; then
    # Generate secure admin key
    ADMIN_KEY="sk-admin-$(openssl rand -hex 24)"

    cat > "$DATA_DIR/config.json" << EOF
{
    "slots": {
        "4801": null,
        "4802": null,
        "4803": null,
        "4804": null,
        "4805": null,
        "4806": null,
        "4807": null
    },
    "hf_token": null,
    "admin_key": "$ADMIN_KEY",
    "default_threads": 10,
    "default_ctx_size": 8192,
    "pricing": {
        "input_per_1k": 0.0001,
        "output_per_1k": 0.0002
    }
}
EOF
    echo ""
    echo "=========================================="
    echo "  ADMIN API KEY (save this!):"
    echo "  $ADMIN_KEY"
    echo "=========================================="
    echo ""
fi

# === Create systemd service ===
echo "Creating systemd service..."

sudo tee /etc/systemd/system/llm-gateway.service > /dev/null << EOF
[Unit]
Description=LLM Gateway API Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SCRIPT_DIR
Environment="PATH=$SCRIPT_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$SCRIPT_DIR/venv/bin/uvicorn server:app --host 0.0.0.0 --port 4000 --workers 1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable llm-gateway

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Commands:"
echo "  Start:   sudo systemctl start llm-gateway"
echo "  Stop:    sudo systemctl stop llm-gateway"
echo "  Status:  sudo systemctl status llm-gateway"
echo "  Restart:  sudo systemctl restart llm-gateway"
echo "  Logs:    journalctl -u llm-gateway -f"
echo ""
echo "Dev mode:"
echo "  source venv/bin/activate"
echo "  uvicorn server:app --host 0.0.0.0 --port 4000 --reload"
echo ""
echo "Access:"
echo "  API:    http://localhost:4000/v1/"
echo "  Admin:  http://localhost:4000/admin/"
echo "  User:   http://localhost:4000/user/"
