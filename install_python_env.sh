#!/usr/bin/env bash

set -e

TOOLBOX_EXTRAS="[isaa]"
PYTHON_VERSION="3.12.9"
TARGET_DIR="$(pwd)/python_env"
OS="$(uname -s)"
PYTHON_EXE="$TARGET_DIR/bin/python3"
PIP_EXE="$PYTHON_EXE -m pip"

mkdir -p "$TARGET_DIR"

download_file() {
    local url=$1
    local dest=$2
    if [ ! -f "$dest" ]; then
        echo "Downloading $url..."
        curl -L "$url" -o "$dest"
    fi
}

install_python_linux_or_mac() {
    local archive_url="https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
    local archive_path="$TARGET_DIR/python.tgz"
    download_file "$archive_url" "$archive_path"

    tar -xzf "$archive_path" -C "$TARGET_DIR"
    cd "$TARGET_DIR/Python-${PYTHON_VERSION}"

    ./configure --prefix="$TARGET_DIR"
    make -j$(nproc)
    make install

    cd -
    rm -rf "$archive_path" "$TARGET_DIR/Python-${PYTHON_VERSION}"
}

install_python_windows() {
    local exe_url="https://www.python.org/ftp/python/${PYTHON_VERSION}/python-${PYTHON_VERSION}-amd64.exe"
    local installer_path="$TARGET_DIR/python-installer.exe"
    download_file "$exe_url" "$installer_path"

    "$installer_path" /quiet TargetDir="$TARGET_DIR" InstallAllUsers=0 PrependPath=1
    rm -f "$installer_path"
    PYTHON_EXE="$TARGET_DIR/python.exe"
}

install_pip() {
    if ! $PYTHON_EXE -m pip --version >/dev/null 2>&1; then
        echo "Installing pip..."
        curl -sS https://bootstrap.pypa.io/get-pip.py -o "$TARGET_DIR/get-pip.py"
        $PYTHON_EXE "$TARGET_DIR/get-pip.py"
        rm -f "$TARGET_DIR/get-pip.py"
    else
        echo "pip already installed."
    fi
}

install_current_package() {
   if [[ "$TOOLBOX_EXTRAS" == *"isaa"* ]]; then
         $PYTHON_EXE -m pip install -e "toolboxv2/mods/isaa" --no-warn-script-location
    fi
    $PYTHON_EXE -m pip install -e "." --no-warn-script-location
}

install_extra_packages() {
    $PIP_EXE install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    $PIP_EXE install \
        starlette litellm pebble transformers networkx numpy tiktoken \
        chromadb openai pyaudio whisper gtts pydub websockets keyboard \
        pyperclip pygments beautifulsoup4 duckduckgo_search langchain \
        langchain_community langchain_core Pebble Requests tqdm utils \
        tokenizers sentence-transformers "browser-use>=0.1.40" \
        "python-git>=2018.2.1" "langchain-experimental>=0.3.4" \
        "rapidfuzz>=3.12.2" "astor>=0.8.1" "taichi>=1.7.3" \
        "nest-asyncio>=1.6.0" "schedule>=1.2.2" qdrant-client[fastembed] \
        "python-levenshtein>=0.27.1" "langchain-google-genai>=2.1.2"

    $PIP_EXE install websockets schedule mailjet_rest mockito
}

main() {
    echo "Installing Python $PYTHON_VERSION in $TARGET_DIR..."

    case "$OS" in
        Linux|Darwin)
            install_python_linux_or_mac
            ;;
        MINGW*|CYGWIN*|MSYS*|Windows_NT)
            install_python_windows
            ;;
        *)
            echo "Unsupported OS: $OS"
            exit 1
            ;;
    esac

    install_pip
    install_current_package

    if [[ "$TOOLBOX_EXTRAS" == *"isaa"* ]]; then
        install_extra_packages
    fi

    echo "✅ Python installed in $TARGET_DIR"
    echo "✅ Run your Python with: $PYTHON_EXE"
}

main
